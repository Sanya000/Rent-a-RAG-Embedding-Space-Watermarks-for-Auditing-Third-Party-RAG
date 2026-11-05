"""KGW/WARD-style token watermark: generation-time logits processor and aligned detection."""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


# ---------------------------------------------------------------------------
# WARD/KGW-compatible PRF seeding (mirrors WARD's free-form seeding scheme API)
# ---------------------------------------------------------------------------


def _seeding_scheme_lookup(seeding_scheme: str) -> tuple[str, int, bool, int]:
    if not isinstance(seeding_scheme, str):
        raise ValueError("Seeding scheme should be a string.")

    if seeding_scheme in {"simple_1", "lefthash"}:
        return ("additive_prf", 1, False, 15485863)
    if seeding_scheme in {"algorithm-3", "selfhash"}:
        return ("anchored_minhash_prf", 4, True, 15485863)
    if seeding_scheme == "minhash":
        return ("minhash_prf", 4, False, 15485863)
    if seeding_scheme == "skipgram":
        return ("skipgram_prf", 5, False, 15485863)
    if seeding_scheme == "gptwm":
        return ("constant_prf", 0, False, 0)
    if seeding_scheme.startswith("ff") or seeding_scheme.startswith("hard-"):
        # Example: ff-position_prf-2-False-1548585
        parts = seeding_scheme.split("-")
        if len(parts) < 4:
            raise ValueError(f"Malformed seeding scheme: {seeding_scheme}")
        prf_type = str(parts[1])
        context_width = int(parts[2])
        self_salt = parts[3] == "True"
        hash_key = int(parts[4]) if len(parts) >= 5 else 15485863
        return (prf_type, context_width, self_salt, hash_key)

    raise ValueError(f"Invalid seeding scheme: {seeding_scheme}")


_RNG_TABLE = torch.Generator(device=torch.device("cpu"))
_RNG_TABLE.manual_seed(2971215073)
_HASH_TABLE_SIZE = 1_000_003
_FIXED_TABLE = torch.randperm(_HASH_TABLE_SIZE, generator=_RNG_TABLE, device=torch.device("cpu"))


def _hashint(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    return _FIXED_TABLE[integer_tensor.cpu() % _HASH_TABLE_SIZE] + 1


def _multiplicative_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return int(salt_key * input_ids.prod().item())


def _additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return int(salt_key * input_ids.sum().item())


def _minfunc_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return int(salt_key * input_ids.min().item())


def _simple_skip_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    return int(_hashint(salt_key * input_ids[::k]).prod().item())


def _skipgram_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return int(_hashint(salt_key * input_ids[0]).item())


def _anchored_skipgram_prf(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    return int((_hashint(salt_key * input_ids[0]) * _hashint(salt_key * input_ids[anchor])).item())


def _minhash_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return int(_hashint(salt_key * input_ids).min().item())


def _anchored_minhash_prf(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    arr = salt_key * _hashint(input_ids) * _hashint(input_ids[anchor])
    return int(arr.min().item())


def _minskipgram_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    skipgrams = torch.as_tensor(list(combinations(_hashint(salt_key * input_ids), 2)))
    return int(skipgrams.prod(dim=1).min().item())


def _noncomm_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    key = torch.as_tensor(salt_key, dtype=torch.long)
    for entry in input_ids:
        key *= _hashint(key * entry)
        key %= 2**32
    return int(key.item())


def _position_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    # WARD-compatible position PRF
    return int(
        (salt_key * input_ids * torch.arange(1, len(input_ids) + 1, device=input_ids.device))
        .sum()
        .item()
    )


def _constant_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return int(salt_key)


_PRF_LOOKUP = {
    "multiplicative_prf": _multiplicative_prf,
    "additive_prf": _additive_prf,
    "minfunc_prf": _minfunc_prf,
    "simple_skip_prf": _simple_skip_prf,
    "skipgram_prf": _skipgram_prf,
    "anchored_skipgram_prf": _anchored_skipgram_prf,
    "minhash_prf": _minhash_prf,
    "anchored_minhash_prf": _anchored_minhash_prf,
    "minskipgram_prf": _minskipgram_prf,
    "noncomm_prf": _noncomm_prf,
    "position_prf": _position_prf,
    "constant_prf": _constant_prf,
}


def _keyhex_to_int(key_hex: str) -> int:
    if not key_hex:
        return 0
    try:
        kb = bytes.fromhex(key_hex)
    except Exception:
        kb = key_hex.encode("utf-8", errors="ignore")
    if len(kb) == 0:
        return 0
    return int.from_bytes(kb[:8].ljust(8, b"\x00"), "big", signed=False)


def _effective_hash_key(base_hash_key: int, key_hex: str) -> int:
    # Keep WARD PRF structure, but key providers independently via salt-key personalization.
    k = _keyhex_to_int(key_hex)
    eff = int(base_hash_key) ^ int(k & 0x7FFFFFFF)
    if eff == 0:
        eff = int(base_hash_key) or 15485863
    return eff


@dataclass
class KGWParams:
    # Defaults from WARD main experiment config.
    seeding_scheme: str = "ff-position_prf-2-False-1548585"
    gamma: float = 0.25
    delta: float = 3.5
    ignore_repeated_ngrams: bool = True
    # Keep generation and detection on the exact same vocabulary size.
    # If None, we infer from tokenizer (or runtime logits for generation).
    vocab_size: Optional[int] = None
    # Optional speed knob; None keeps faithful full-vocab greenlist biasing.
    topk_mask: Optional[int] = None


def _vocab_size(tokenizer: PreTrainedTokenizer, override: Optional[int] = None) -> int:
    if override is not None and int(override) > 0:
        return int(override)
    # tokenizer.vocab_size can exclude added tokens; len(tokenizer) is often closer.
    v_tok = int(getattr(tokenizer, "vocab_size", 0) or 0)
    try:
        v_len = int(len(tokenizer))
    except Exception:
        v_len = int(len(tokenizer.get_vocab()))
    return max(v_tok, v_len)


def _ngrams(seq: Sequence[int], n: int) -> Iterable[tuple[int, ...]]:
    if n <= 0:
        return []
    if len(seq) < n:
        return []
    return (tuple(seq[i : i + n]) for i in range(0, len(seq) - n + 1))


class KGWLogitsProcessor(LogitsProcessor):
    """
    WARD/KGW-compatible watermark logits processor.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        key_hex: str,
        params: KGWParams,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.params = params
        self.vocab_size = _vocab_size(tokenizer, override=params.vocab_size)
        self.key_hex = key_hex  # kept for API parity / metadata
        self.device = device

        (
            self.prf_type,
            self.context_width,
            self.self_salt,
            self.hash_key,
        ) = _seeding_scheme_lookup(params.seeding_scheme)
        self.hash_key = _effective_hash_key(self.hash_key, key_hex=self.key_hex)
        if self.prf_type not in _PRF_LOOKUP:
            raise ValueError(f"Unsupported PRF type: {self.prf_type}")

        # Must stay on CPU so generation-time and detector-time greenlists match.
        self.rng: Optional[torch.Generator] = None

    def _ensure_rng(self) -> torch.Generator:
        if self.rng is None:
            self.rng = torch.Generator(device=torch.device("cpu"))
        return self.rng

    def _seed_rng(self, prefix_ids: torch.LongTensor, rng: torch.Generator) -> None:
        if prefix_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"Seeding scheme requires at least {self.context_width} tokens of context."
            )
        # Keep PRF inputs on CPU for exact parity with detector-side oracle.
        pref = prefix_ids[-self.context_width :].detach().to(
            device=torch.device("cpu"), dtype=torch.long
        )
        prf_key = _PRF_LOOKUP[self.prf_type](
            pref, salt_key=self.hash_key
        )
        rng.manual_seed(int(prf_key) % (2**64 - 1))

    def _get_greenlist_ids(
        self, prefix_ids: torch.LongTensor, rng: torch.Generator
    ) -> torch.LongTensor:
        self._seed_rng(prefix_ids, rng)
        greenlist_size = int(self.vocab_size * float(self.params.gamma))
        perm = torch.randperm(self.vocab_size, device=torch.device("cpu"), generator=rng)
        return perm[:greenlist_size]

    def _score_rejection_sampling(
        self, prefix_ids: torch.LongTensor, scores_1d: torch.Tensor, rng: torch.Generator
    ) -> torch.LongTensor:
        sorted_scores, greedy_predictions = scores_1d.sort(dim=-1, descending=True)
        accepted: List[torch.Tensor] = []
        for idx, tok in enumerate(greedy_predictions):
            gl = self._get_greenlist_ids(torch.cat([prefix_ids, tok[None]], dim=0), rng)
            tok_i = int(tok.item())
            if tok_i in set(int(x) for x in gl.tolist()):
                accepted.append(tok)
            # Same fixed-compute stopping default used by upstream processor.
            if idx == 40:
                break
        if not accepted:
            return torch.empty((0,), dtype=torch.long, device=prefix_ids.device)
        return torch.stack(accepted, dim=0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        # Use runtime logits size if tokenizer metadata was incomplete.
        if self.params.vocab_size is None and int(scores.shape[-1]) != int(self.vocab_size):
            self.vocab_size = int(scores.shape[-1])

        rng = self._ensure_rng()
        list_of_greenlist_ids: List[torch.LongTensor] = [None] * input_ids.shape[0]  # type: ignore

        for b_idx, prefix in enumerate(input_ids):
            if self.self_salt:
                green_ids = self._score_rejection_sampling(prefix, scores[b_idx], rng)
            else:
                green_ids = self._get_greenlist_ids(prefix, rng)
            list_of_greenlist_ids[b_idx] = green_ids

        if self.params.topk_mask is None:
            green_mask = torch.zeros_like(scores, dtype=torch.bool)
            for b_idx, green_ids in enumerate(list_of_greenlist_ids):
                if len(green_ids) > 0:
                    green_mask[b_idx, green_ids.to(device=scores.device)] = True
            scores[green_mask] = scores[green_mask] + float(self.params.delta)
            return scores

        # Optional speed mode: bias only among top-K logits.
        k = max(1, min(int(self.params.topk_mask), scores.shape[-1]))
        for b_idx, green_ids in enumerate(list_of_greenlist_ids):
            if len(green_ids) == 0:
                continue
            _, top_idx = torch.topk(scores[b_idx], k=k, dim=-1)
            # membership test in greenlist
            green_set = set(int(x) for x in green_ids.tolist())
            keep = [i for i, tok in enumerate(top_idx.tolist()) if int(tok) in green_set]
            if keep:
                scores[b_idx, top_idx[keep]] = scores[b_idx, top_idx[keep]] + float(
                    self.params.delta
                )
        return scores


@dataclass
class KGWDetectionResult:
    T: int
    greens: int
    z: float
    p_one_sided: float
    green_fraction: float


def _z_and_p(T: int, greens: int, gamma: float) -> tuple[float, float]:
    if int(T) <= 0:
        return (0.0, 1.0)
    mu = float(gamma) * int(T)
    var = float(gamma) * (1.0 - float(gamma)) * int(T)
    z = (float(greens) - mu) / math.sqrt(var + 1e-12)
    try:
        from scipy.stats import norm

        p = float(norm.sf(z))
    except Exception:
        p = float(0.5 * math.erfc(z / math.sqrt(2.0)))
    return (float(z), float(p))


class KGWGreenlistOracle:
    """
    Fast detector-time helper: deterministic greenlist membership under a fixed key.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, key_hex: str, params: KGWParams):
        self.tokenizer = tokenizer
        self.key_hex = key_hex
        self.params = params
        self.vocab_size = _vocab_size(tokenizer, override=params.vocab_size)
        (
            self.prf_type,
            self.context_width,
            self.self_salt,
            self.hash_key,
        ) = _seeding_scheme_lookup(params.seeding_scheme)
        self.hash_key = _effective_hash_key(self.hash_key, key_hex=self.key_hex)
        if self.prf_type not in _PRF_LOOKUP:
            raise ValueError(f"Unsupported PRF type: {self.prf_type}")
        self._rng = torch.Generator(device=torch.device("cpu"))
        self._green_cache: Dict[tuple[int, ...], set[int]] = {}

    def _seed_from_prefix(self, prefix_ids: tuple[int, ...]) -> int:
        pref = torch.as_tensor(prefix_ids, dtype=torch.long, device=torch.device("cpu"))
        if pref.shape[-1] < self.context_width:
            raise ValueError(
                f"Seeding scheme requires at least {self.context_width} context tokens."
            )
        prf_key = _PRF_LOOKUP[self.prf_type](
            pref[-self.context_width :], salt_key=self.hash_key
        )
        return int(prf_key) % (2**64 - 1)

    def _greenlist_set(self, prefix_ids: tuple[int, ...]) -> set[int]:
        got = self._green_cache.get(prefix_ids)
        if got is not None:
            return got
        seed = self._seed_from_prefix(prefix_ids)
        self._rng.manual_seed(seed)
        green_sz = int(self.vocab_size * float(self.params.gamma))
        perm = torch.randperm(self.vocab_size, generator=self._rng, device=torch.device("cpu"))
        out = set(int(x) for x in perm[:green_sz].tolist())
        self._green_cache[prefix_ids] = out
        return out

    def is_green(self, prefix_ids: tuple[int, ...], token_id: int) -> bool:
        return int(token_id) in self._greenlist_set(prefix_ids)


def detect_kgw_text(
    text: str,
    key_hex: str,
    tokenizer: PreTrainedTokenizer,
    params: KGWParams,
) -> KGWDetectionResult:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return KGWDetectionResult(T=0, greens=0, z=0.0, p_one_sided=1.0, green_fraction=0.0)

    oracle = KGWGreenlistOracle(tokenizer=tokenizer, key_hex=key_hex, params=params)
    n = oracle.context_width + 1 - int(oracle.self_salt)
    if len(ids) < n:
        return KGWDetectionResult(T=0, greens=0, z=0.0, p_one_sided=1.0, green_fraction=0.0)

    ng = list(_ngrams(ids, n))
    freq = collections.Counter(ng)

    lookup: Dict[tuple[int, ...], bool] = {}
    for g in freq.keys():
        prefix = g if oracle.self_salt else g[:-1]
        target = int(g[-1])
        if len(prefix) < oracle.context_width:
            lookup[g] = False
            continue
        lookup[g] = oracle.is_green(tuple(prefix), target)

    if params.ignore_repeated_ngrams:
        T = len(freq.keys())
        greens = int(sum(1 for v in lookup.values() if v))
    else:
        T = int(sum(freq.values()))
        greens = int(sum(int(v) * int(freq[g]) for g, v in lookup.items()))

    z, p = _z_and_p(T=T, greens=greens, gamma=float(params.gamma))
    gf = (float(greens) / float(T)) if T > 0 else 0.0
    return KGWDetectionResult(T=T, greens=greens, z=z, p_one_sided=p, green_fraction=gf)


def _ids_no_special(tok: PreTrainedTokenizer, text: str) -> List[int]:
    ids = tok.encode(text, add_special_tokens=False)
    out: List[int] = []
    for i in ids:
        if tok.pad_token_id is not None and i == tok.pad_token_id:
            continue
        if tok.eos_token_id is not None and i == tok.eos_token_id:
            continue
        out.append(int(i))
    return out


def align_by_ngrams_spans(
    src_ids: Sequence[int],
    ans_ids: Sequence[int],
    n_min: int = 3,
) -> List[tuple[int, int, int]]:
    """
    Greedy exact n-gram span alignment.
    Returns spans as (src_start, ans_start, length).
    """
    if len(src_ids) < n_min or len(ans_ids) < n_min:
        return []

    index: Dict[tuple[int, ...], List[int]] = {}
    for i in range(0, len(src_ids) - n_min + 1):
        key = tuple(src_ids[i : i + n_min])
        index.setdefault(key, []).append(i)

    used_ans = [False] * len(ans_ids)
    spans: List[tuple[int, int, int]] = []

    j = 0
    while j <= len(ans_ids) - n_min:
        if used_ans[j]:
            j += 1
            continue
        key = tuple(ans_ids[j : j + n_min])
        cand = index.get(key)
        if not cand:
            j += 1
            continue

        best_i = -1
        best_len = 0
        for i in cand:
            L = n_min
            while (
                i + L < len(src_ids)
                and j + L < len(ans_ids)
                and src_ids[i + L] == ans_ids[j + L]
                and (not used_ans[j + L])
            ):
                L += 1
            if L > best_len:
                best_len = L
                best_i = i

        if best_len > 0:
            spans.append((best_i, j, best_len))
            for k in range(0, best_len):
                used_ans[j + k] = True
            j += best_len
        else:
            j += 1

    return spans


def detect_kgw_multi_source_aligned(
    *,
    answer_text: str,
    sources: Iterable[Tuple[int, str]],
    provider_keys_by_pid: Dict[int, str],
    tokenizer: PreTrainedTokenizer,
    params: KGWParams,
    n_min: int = 3,
    competitive: bool = True,
) -> tuple[Dict[int, KGWDetectionResult], Dict[str, object]]:
    """
    Provider-attribution detector for mixed-source answers:
      - Align answer tokens to each source via exact n-gram spans.
      - If competitive=True, each answer token is assigned to one provider
        (largest aligned span-length wins).
      - Evaluate KGW green membership under source-side prefix context.
    """
    # Keep all retrieved sources per provider (not just first) to avoid
    # penalizing aligned KGW when multiple docs from the same provider are present.
    source_docs_by_pid: Dict[int, List[List[int]]] = {}
    for pid, txt in sources:
        pid_i = int(pid)
        ids = _ids_no_special(tokenizer, txt)
        if not ids:
            continue
        source_docs_by_pid.setdefault(pid_i, []).append(ids)

    ans_ids = _ids_no_special(tokenizer, answer_text)
    P = sorted(provider_keys_by_pid.keys())

    if len(ans_ids) == 0 or len(source_docs_by_pid) == 0:
        zeros = {
            int(pid): KGWDetectionResult(T=0, greens=0, z=0.0, p_one_sided=1.0, green_fraction=0.0)
            for pid in P
        }
        meta = {
            "windows_total": len(ans_ids),
            "assigned_total": 0,
            "dropped_below_thresh_windows": len(ans_ids),
            "best_cos_stats": {"mean": None},
            "second_best_cos_stats": {"mean": None},
            "assigned_tokens_by_pid": {int(pid): 0 for pid in P},
        }
        return zeros, meta

    # answer token idx -> list[(pid, doc_idx, src_idx, span_len)]
    candidates: Dict[int, List[tuple[int, int, int, int]]] = {i: [] for i in range(len(ans_ids))}
    for pid, src_docs in source_docs_by_pid.items():
        for doc_idx, src_ids in enumerate(src_docs):
            spans = align_by_ngrams_spans(src_ids=src_ids, ans_ids=ans_ids, n_min=n_min)
            for s0, a0, L in spans:
                for k in range(0, L):
                    ai = a0 + k
                    si = s0 + k
                    if ai < len(ans_ids):
                        candidates[ai].append((int(pid), int(doc_idx), int(si), int(L)))

    # Per provider: tuples are (doc_idx, src_token_idx, ans_token_idx)
    assigned: Dict[int, List[tuple[int, int, int]]] = {int(pid): [] for pid in P}
    assigned_total = 0
    for ai in range(0, len(ans_ids)):
        cands = candidates.get(ai) or []
        if not cands:
            continue
        if competitive:
            # larger aligned span gets priority; deterministic tie-break on provider/doc/src index
            cands_sorted = sorted(cands, key=lambda t: (t[3], -t[0], -t[1], -t[2]), reverse=True)
            pid, doc_idx, si, _L = cands_sorted[0]
            assigned.setdefault(pid, []).append((doc_idx, si, ai))
            assigned_total += 1
        else:
            for pid, doc_idx, si, _L in cands:
                assigned.setdefault(pid, []).append((doc_idx, si, ai))
                assigned_total += 1

    oracles: Dict[int, KGWGreenlistOracle] = {
        int(pid): KGWGreenlistOracle(tokenizer=tokenizer, key_hex=provider_keys_by_pid[int(pid)], params=params)
        for pid in P
    }

    results: Dict[int, KGWDetectionResult] = {}
    for pid in P:
        pairs = assigned.get(int(pid), [])
        src_docs = source_docs_by_pid.get(int(pid), [])
        oracle = oracles[int(pid)]
        seen_ngrams: set[tuple[int, ...]] = set()
        greens = 0
        T = 0
        for doc_idx, si, ai in pairs:
            if doc_idx < 0 or doc_idx >= len(src_docs):
                continue
            src_ids = src_docs[doc_idx]
            if si < oracle.context_width:
                continue
            prefix = tuple(src_ids[si - oracle.context_width : si])
            tok = int(ans_ids[ai])
            gram = tuple(list(prefix) + [tok])
            if params.ignore_repeated_ngrams:
                if gram in seen_ngrams:
                    continue
                seen_ngrams.add(gram)
            if oracle.is_green(prefix, tok):
                greens += 1
            T += 1
        z, p = _z_and_p(T=T, greens=greens, gamma=float(params.gamma))
        gf = (float(greens) / float(T)) if T > 0 else 0.0
        results[int(pid)] = KGWDetectionResult(
            T=int(T),
            greens=int(greens),
            z=float(z),
            p_one_sided=float(p),
            green_fraction=float(gf),
        )

    meta = {
        "windows_total": int(len(ans_ids)),
        "assigned_total": int(assigned_total),
        "dropped_below_thresh_windows": int(max(0, len(ans_ids) - assigned_total)),
        # Keep keys parallel to dirbucket detector metadata so higher-level code can stay generic.
        "best_cos_stats": {"mean": None},
        "second_best_cos_stats": {"mean": None},
        "assigned_tokens_by_pid": {int(pid): int(len(assigned.get(int(pid), []))) for pid in P},
    }
    return results, meta
