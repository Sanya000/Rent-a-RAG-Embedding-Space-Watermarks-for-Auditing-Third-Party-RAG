"""Audit harness behind every Sec. 5 benchmark result (one audit = one JSON + one trial log).

--exp sanity_single | rag_multi | rag_absent_target | style_sweep
--wm_scheme dirbucket_naive | kgw_ward | semstamp | sim_attr | overlap_attr | entail_attr

Each trial makes two OpenAI calls (question generation at temperature 0.2 and the RAG answer
at --answer_temp), so OPENAI_API_KEY must be set. The CLI defaults are not the paper's
operating point; the drivers (run_watermark_scheme_compare.py, run_paper_rag_multi_dirbucket.py)
and reproduce/scripts/1*.sh pass the paper arguments explicitly (aligned/global_corpus,
tau_sim 0.82, tie margin 0.05, T_min 25, ...).

This file is the full version of the runner. The E1/E2 run cards name a DirBucket-only copy
of it (submission_repro_dirbucket/); its dirbucket_naive path is the same (replaying recorded
trials through both gives byte-identical trial logs).
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Must resolve to your existing implementation (in experiments/watermark_dirbucket.py or watermark_dirbucket.py)
from watermark_dirbucket import DetectSummary, MultiDirBucketDetector, split_sentences, _windows_within_sentence
from watermark_kirchenbauer import KGWParams, detect_kgw_multi_source_aligned, detect_kgw_text
from watermark_semstamp import (
    SemStampClusterer,
    SemStampParams,
    detect_semstamp_multi_source_aligned,
    detect_semstamp_text,
)
from attribution_baselines import (
    EntailmentBaselineParams,
    EntailmentScorer,
    LexicalOverlapScorer,
    OverlapBaselineParams,
    SimilarityBaselineParams,
    score_entailment_by_provider,
    score_overlap_by_provider,
    score_similarity_by_provider,
)

_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}


# ---------------- JSON helpers ----------------
def json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy/scalars/paths into JSON-serializable Python objects.
    Fixes errors like: TypeError: Object of type bool_ is not JSON serializable
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)

    # numpy scalar -> python scalar
    if isinstance(obj, (np.generic,)):
        return obj.item()

    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, set):
        return [json_safe(x) for x in sorted(obj)]

    # fallback: try stringifying
    return str(obj)


# ---------------- IO ----------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_alignment_manifest_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build a fast lookup for alignment source docs by doc id.
    Expected row fields: id, text_src (or text), provider_id (optional).
    """
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            did = str(row.get("id") or row.get("doc_id") or "").strip()
            if not did:
                continue
            txt = str(row.get("text_src") or row.get("text") or "").strip()
            if not txt:
                continue
            pid = row.get("provider_id")
            pid_int: Optional[int] = None
            try:
                if pid is not None:
                    pid_int = int(pid)
            except Exception:
                pid_int = None
            out[did] = {"text": txt, "provider_id": pid_int}
    return out


def load_workdir(
    work_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[str], str, str, str, str, Dict[str, Any]]:
    """
    Returns:
      rows, keys_hex, default_source_field, default_watermarked_field,
      default_wm_scheme, default_direction_scheme, keys_meta

    Supports:
      - legacy: provider_keys.json + text_wm
      - unified: provider_keys_dirbucket.json + text_src + text_wm_dirbucket
    """
    corpus_path = work_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")
    rows = load_jsonl(corpus_path)
    if not rows:
        raise RuntimeError("corpus.jsonl is empty")

    key_candidates = [
        work_dir / "provider_keys_kgw.json",
        work_dir / "provider_keys_dirbucket.json",
        work_dir / "provider_keys.json",
    ]
    key_path = next((p for p in key_candidates if p.exists()), None)
    if key_path is None:
        raise FileNotFoundError(
            "Missing keys file. Expected one of: provider_keys_kgw.json, "
            "provider_keys_dirbucket.json, provider_keys.json."
        )
    meta = json.loads(key_path.read_text(encoding="utf-8"))
    keys_hex = list(meta["keys_hex"])

    sample = rows[0]
    if "text_src" in sample:
        default_source_field = "text_src"
    else:
        default_source_field = "text_wm"

    if "text_wm_dirbucket" in sample:
        default_wm_field = "text_wm_dirbucket"
    elif "text_wm_semstamp" in sample:
        default_wm_field = "text_wm_semstamp"
    elif "text_wm_kgw" in sample:
        default_wm_field = "text_wm_kgw"
    else:
        default_wm_field = "text_wm"

    default_direction_scheme = str(meta.get("direction_scheme") or "naive").lower()
    declared = str(meta.get("watermark_scheme") or "").strip().lower()
    if declared in {"kgw_ward", "dirbucket_naive", "semstamp"}:
        default_wm_scheme = declared
    elif declared.startswith("dirbucket"):
        default_wm_scheme = "dirbucket_naive"
    elif declared.startswith("semstamp"):
        default_wm_scheme = "semstamp"
    elif key_path.name == "provider_keys_kgw.json":
        default_wm_scheme = "kgw_ward"
    elif "direction_scheme" in meta:
        default_wm_scheme = "dirbucket_naive"
    else:
        # Backward-compat fallback.
        default_wm_scheme = "dirbucket_naive"

    return (
        rows,
        keys_hex,
        default_source_field,
        default_wm_field,
        default_wm_scheme,
        default_direction_scheme,
        meta,
    )


def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def read_trial_log(path: Optional[Path], *, scheme: str, audit_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    if path is None or (not path.exists()):
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if str(obj.get("wm_scheme") or "") != str(scheme):
                continue
            if audit_mode is not None:
                mode = str(obj.get("audit_mode") or "aligned")
                if mode != str(audit_mode):
                    continue
            idx = obj.get("trial_idx")
            if isinstance(idx, int) and idx >= 1:
                out.append(obj)
    out.sort(key=lambda r: int(r["trial_idx"]))
    return out


def append_trial_log(path: Optional[Path], rec: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(rec), ensure_ascii=False) + "\n")


def default_trial_log_path(out_json: Optional[Path]) -> Optional[Path]:
    if out_json is None:
        return None
    if out_json.suffix == ".json":
        return out_json.with_suffix(".trials.jsonl")
    return Path(str(out_json) + ".trials.jsonl")


def get_tokenizer_cached(model_name: str) -> AutoTokenizer:
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _TOKENIZER_CACHE[model_name] = tok
    return tok


# ---------------- Retrieval ----------------
def ensure_embeddings(
    work_dir: Path,
    rows: List[Dict[str, Any]],
    field: str,
    model_name: str,
    max_chars: int = 1000,
) -> Tuple[np.ndarray, SentenceTransformer]:
    cache = work_dir / f"emb_{field}.npz"
    enc = SentenceTransformer(model_name)

    if cache.exists():
        dat = np.load(cache, allow_pickle=True)
        return dat["X"], enc

    docs = [((r.get(field) or "")[:max_chars]) for r in rows]
    X = enc.encode(docs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    np.savez(cache, X=X)
    return X, enc


def topk_retrieve_idxs(
    rows: List[Dict[str, Any]],
    X: np.ndarray,
    enc: SentenceTransformer,
    query_text: str,
    include_doc_id: Optional[str],
    k: int,
    min_unique_providers: int = 1,
    force_cross_provider: bool = False,
    query_vec: Optional[np.ndarray] = None,
    avoid_provider: Optional[int] = None,
) -> List[int]:
    q = (
        np.asarray(query_vec, dtype=np.float32)
        if query_vec is not None
        else enc.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    )
    sims = X @ q
    order = np.argsort(-sims)

    target_idx: Optional[int] = None
    target_provider: Optional[int] = None
    if include_doc_id is not None:
        id2idx = {str(r["id"]): i for i, r in enumerate(rows)}
        target_idx = id2idx.get(str(include_doc_id))
        if target_idx is None:
            raise KeyError(f"include_doc_id={include_doc_id} not found in rows")
        target_provider = int(rows[target_idx]["provider_id"])
        if avoid_provider is not None and int(target_provider) == int(avoid_provider):
            raise ValueError(
                f"include_doc_id provider {target_provider} conflicts with avoid_provider={avoid_provider}"
            )

    picked: List[int] = []
    if target_idx is not None:
        picked.append(int(target_idx))
        if k <= 1:
            return picked

    others = [int(i) for i in order if (target_idx is None or int(i) != int(target_idx))]
    if avoid_provider is not None:
        others = [i for i in others if int(rows[i]["provider_id"]) != int(avoid_provider)]

    if force_cross_provider and target_provider is not None:
        primary = [i for i in others if int(rows[i]["provider_id"]) != int(target_provider)]
        secondary = [i for i in others if int(rows[i]["provider_id"]) == int(target_provider)]
    else:
        primary, secondary = others, []

    def add_from(pool: List[int], picked_: List[int], need_k: int) -> None:
        have = set(int(rows[j]["provider_id"]) for j in picked_)
        for i in pool:
            if len(picked_) >= need_k:
                break
            pid = int(rows[i]["provider_id"])
            if len(have) < min_unique_providers and pid in have:
                continue
            picked_.append(i)
            have.add(pid)

    add_from(primary, picked, k)
    if len(picked) < k:
        add_from(secondary, picked, k)
    if len(picked) < k:
        for i in others:
            if i not in picked:
                picked.append(i)
                if len(picked) >= k:
                    break

    return picked


def _canon_aligned_source_mode(mode: str) -> str:
    m = str(mode or "oracle").strip().lower()
    if m in {"oracle", "prompt_wm"}:
        return "oracle"
    return m


def _canon_global_source_selector(selector: str) -> str:
    s = str(selector or "query").strip().lower()
    if s not in {"query", "answer", "max"}:
        raise ValueError(
            f"Unknown --global_source_selector={selector}. "
            "Use one of: query, answer, max."
        )
    return s


def _global_source_candidates(
    *,
    rows: List[Dict[str, Any]],
    sims: np.ndarray,
    source_field: str,
    sim_threshold: float,
    max_docs: int,
    max_docs_per_provider: int,
) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    order = np.argsort(-sims)
    above_thr = int(np.sum(sims >= float(sim_threshold)))

    chunks_by_pid: Dict[int, List[str]] = {}
    selected_doc_ids: List[str] = []
    selected_sims: List[float] = []
    per_provider_docs: Dict[int, int] = {}

    cap_total = int(max_docs) if int(max_docs) > 0 else None
    cap_per_pid = int(max_docs_per_provider) if int(max_docs_per_provider) > 0 else None

    for idx in order:
        sim = float(sims[int(idx)])
        if sim < float(sim_threshold):
            break
        if cap_total is not None and len(selected_doc_ids) >= cap_total:
            break

        row = rows[int(idx)]
        pid = int(row.get("provider_id", -1))
        if cap_per_pid is not None and per_provider_docs.get(pid, 0) >= cap_per_pid:
            continue
        txt = str(row.get(source_field) or "").strip()
        if not txt:
            continue

        did = str(row.get("id") or "")
        chunks_by_pid.setdefault(pid, []).append(txt)
        per_provider_docs[pid] = per_provider_docs.get(pid, 0) + 1
        selected_doc_ids.append(did)
        selected_sims.append(sim)

    diag: Dict[str, Any] = {
        "candidate_pool_docs": int(len(rows)),
        "above_threshold_docs": int(above_thr),
        "selected_docs": int(len(selected_doc_ids)),
        "selected_providers": int(len(per_provider_docs)),
        "selected_docs_by_provider": {str(pid): int(n) for pid, n in sorted(per_provider_docs.items())},
        "selected_sim_mean": (float(np.mean(selected_sims)) if selected_sims else None),
        "selected_sim_min": (float(np.min(selected_sims)) if selected_sims else None),
        "selected_sim_max": (float(np.max(selected_sims)) if selected_sims else None),
        "selected_doc_ids_sample": selected_doc_ids[:30],
    }
    return chunks_by_pid, diag


def build_alignment_sources(
    *,
    rows: List[Dict[str, Any]],
    picked_idxs: List[int],
    prompt_docs: List[Tuple[int, str]],
    retrieved_docs: List[Dict[str, Any]],
    source_field: str,
    aligned_source_mode: str,
    align_manifest_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    query_vec: Optional[np.ndarray] = None,
    answer_vec: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    global_source_query_sim_threshold: float = 0.55,
    global_source_max_docs: int = 120,
    global_source_max_docs_per_provider: int = 0,
    global_source_selector: str = "query",
) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
    mode = _canon_aligned_source_mode(aligned_source_mode)
    selector = _canon_global_source_selector(global_source_selector)
    chunks_by_pid: Dict[int, List[str]] = {}
    diag: Dict[str, Any] = {
        "mode": str(mode),
        "query_sim_threshold": None,
        "max_docs": None,
        "max_docs_per_provider": None,
        "global_source_selector": (selector if mode == "global_corpus" else None),
    }

    if mode == "oracle":
        for pid, txt in prompt_docs:
            if txt:
                chunks_by_pid.setdefault(int(pid), []).append(str(txt))
    elif mode == "retrieved_src":
        for i in picked_idxs:
            pid = int(rows[i]["provider_id"])
            txt = str(rows[i].get(source_field) or "").strip()
            if txt:
                chunks_by_pid.setdefault(pid, []).append(txt)
    elif mode == "manifest_src":
        if align_manifest_by_id is None:
            raise ValueError("align_manifest_by_id is required for aligned_source_mode=manifest_src")
        for rec in retrieved_docs:
            did = str(rec.get("doc_id") or "")
            pid = int(rec.get("provider_id"))
            m = align_manifest_by_id.get(did)
            if not m:
                continue
            txt = str(m.get("text") or "").strip()
            if txt:
                chunks_by_pid.setdefault(pid, []).append(txt)
    elif mode == "global_corpus":
        if X is None:
            raise ValueError("X is required for aligned_source_mode=global_corpus")
        qv = None if query_vec is None else np.asarray(query_vec, dtype=np.float32)
        av = None if answer_vec is None else np.asarray(answer_vec, dtype=np.float32)
        if selector == "query":
            if qv is None:
                raise ValueError(
                    "query_vec is required for aligned_source_mode=global_corpus "
                    "when --global_source_selector=query"
                )
            sims = X @ qv
        elif selector == "answer":
            if av is None:
                raise ValueError(
                    "answer_vec is required for aligned_source_mode=global_corpus "
                    "when --global_source_selector=answer"
                )
            sims = X @ av
        else:  # selector == "max"
            if qv is None or av is None:
                raise ValueError(
                    "query_vec and answer_vec are required for aligned_source_mode=global_corpus "
                    "when --global_source_selector=max"
                )
            sims = np.maximum(X @ qv, X @ av)
        chunks_by_pid, extra = _global_source_candidates(
            rows=rows,
            sims=sims,
            source_field=source_field,
            sim_threshold=float(global_source_query_sim_threshold),
            max_docs=int(global_source_max_docs),
            max_docs_per_provider=int(global_source_max_docs_per_provider),
        )
        diag.update(extra)
        diag["query_sim_threshold"] = float(global_source_query_sim_threshold)
        diag["max_docs"] = int(global_source_max_docs)
        diag["max_docs_per_provider"] = int(global_source_max_docs_per_provider)
    else:
        raise ValueError(
            f"Unknown --aligned_source_mode={aligned_source_mode}. "
            f"Use one of: oracle, global_corpus, retrieved_src, manifest_src."
        )

    sources: List[Tuple[int, str]] = []
    selected_docs = 0
    per_provider_selected: Dict[str, int] = {}
    for pid in sorted(chunks_by_pid.keys()):
        docs = [t for t in chunks_by_pid[pid] if str(t).strip()]
        if not docs:
            continue
        merged = "\n\n".join(docs)
        if not merged.strip():
            continue
        sources.append((int(pid), merged))
        selected_docs += len(docs)
        per_provider_selected[str(int(pid))] = int(len(docs))

    if "selected_docs" not in diag:
        diag["selected_docs"] = int(selected_docs)
    if "selected_providers" not in diag:
        diag["selected_providers"] = int(len(sources))
    if "selected_docs_by_provider" not in diag:
        diag["selected_docs_by_provider"] = per_provider_selected
    if "above_threshold_docs" not in diag:
        diag["above_threshold_docs"] = None
    if "candidate_pool_docs" not in diag:
        diag["candidate_pool_docs"] = None
    if "selected_sim_mean" not in diag:
        diag["selected_sim_mean"] = None
    if "selected_sim_min" not in diag:
        diag["selected_sim_min"] = None
    if "selected_sim_max" not in diag:
        diag["selected_sim_max"] = None
    if "selected_doc_ids_sample" not in diag:
        diag["selected_doc_ids_sample"] = []

    return sources, diag


def sample_unrelated_bundle(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    k: int,
    avoid_provider: int,
    min_unique_providers: int,
) -> List[int]:
    """Null mode: pick k docs NOT from avoid_provider, trying to reach min_unique_providers."""
    idxs = [i for i, r in enumerate(rows) if int(r["provider_id"]) != int(avoid_provider)]
    rng.shuffle(idxs)

    picked: List[int] = []
    have: set[int] = set()
    for i in idxs:
        pid = int(rows[i]["provider_id"])
        if len(have) < min_unique_providers and pid in have:
            continue
        picked.append(i)
        have.add(pid)
        if len(picked) >= k:
            break

    if len(picked) < k:
        for i in idxs:
            if i not in picked:
                picked.append(i)
                if len(picked) >= k:
                    break
    return picked


def select_retrieval_bundle_idxs(
    *,
    rows: List[Dict[str, Any]],
    X: np.ndarray,
    enc: SentenceTransformer,
    question: str,
    query_vec: np.ndarray,
    target_doc_id: str,
    target_provider: int,
    k: int,
    min_unique_providers: int,
    force_cross_provider: bool,
    null_unrelated: bool,
    exclude_target_provider_from_retrieval: bool,
    rng: random.Random,
) -> List[int]:
    if null_unrelated:
        return sample_unrelated_bundle(
            rows,
            rng=rng,
            k=k,
            avoid_provider=target_provider,
            min_unique_providers=min_unique_providers,
        )
    return topk_retrieve_idxs(
        rows,
        X,
        enc,
        query_text=question,
        include_doc_id=(None if exclude_target_provider_from_retrieval else str(target_doc_id)),
        k=k,
        min_unique_providers=min_unique_providers,
        force_cross_provider=force_cross_provider,
        query_vec=query_vec,
        avoid_provider=(int(target_provider) if exclude_target_provider_from_retrieval else None),
    )


# ---------------- OpenAI ----------------
def openai_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.55,
    *,
    max_attempts: int = 8,
    backoff_base_s: float = 2.0,
) -> str:
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError

    # Keep SDK-level retries off so we control retry budget/logging explicitly.
    client = OpenAI(max_retries=0)
    attempts = max(1, int(max_attempts))

    for attempt in range(1, attempts + 1):
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            return r.choices[0].message.content
        except Exception as e:
            status = getattr(e, "status_code", None)
            retryable = False
            if isinstance(e, (RateLimitError, APITimeoutError)):
                retryable = True
            elif isinstance(e, APIError):
                retryable = (status is None) or (int(status) >= 500) or (int(status) in {408, 409, 429})
            if (not retryable) or (attempt >= attempts):
                raise

            sleep_s = backoff_base_s * (2 ** (attempt - 1))
            sleep_s = min(60.0, sleep_s) * (0.85 + 0.30 * random.random())
            print(
                f"[OPENAI-RETRY] attempt={attempt}/{attempts} status={status} sleep={sleep_s:.1f}s",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError("openai_chat failed unexpectedly after retries")


_QGEN_PROMPT = (
    "Read the document below and write ONE question whose answer must cite at least three specific details "
    "(names, dates, numbers, or short phrases) from the document. Avoid yes/no or vague prompts. "
    "Return ONLY the question.\n\nDocument:\n{doc}\n\nQuestion:"
)


def gen_question_openai(doc_text: str, model: str) -> str:
    prompt = _QGEN_PROMPT.format(doc=doc_text[:1500])
    return openai_chat(model, [{"role": "user", "content": prompt}], temperature=0.2).strip()


def make_rag_messages(docs_texts: List[Tuple[int, str]], question: str, style: str) -> List[Dict[str, str]]:
    docs_block = "\n\n".join(
        [f"[DOC {i}] (provider={pid})\n{text}" for i, (pid, text) in enumerate(docs_texts)]
    )

    if style == "qa":
        sys = "You are a RAG assistant. Answer using ONLY the provided documents."
        user = f"Here are retrieved documents:\n{docs_block}\n\nQuestion: {question}\n\nAnswer:"
    elif style == "ownwords":
        sys = (
            "You are a RAG assistant. Answer using ONLY the provided documents. "
            "Synthesize across multiple docs. Use your OWN WORDS; avoid quoting exact spans. "
            "Provide 6–8 sentences."
        )
        user = f"Here are retrieved documents:\n{docs_block}\n\nQuestion: {question}\n\nAnswer:"
    elif style == "bullets":
        sys = (
            "You are a RAG assistant. Answer using ONLY the provided documents. "
            "Write 6–10 bullet points. Use your OWN WORDS (no quotes)."
        )
        user = f"Documents:\n{docs_block}\n\nQuestion: {question}\n\nBullet-point answer:"
    elif style == "eli5":
        sys = (
            "You are a RAG assistant. Answer using ONLY the provided documents. "
            "Explain like I'm 5. Use simple words, short sentences, and no quotes."
        )
        user = f"Documents:\n{docs_block}\n\nQuestion: {question}\n\nELI5 answer:"
    else:
        raise ValueError(f"Unknown answer_style: {style}")

    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ---------------- Stats ----------------
def norm_sf(z: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.sf(z))
    except Exception:
        from math import erfc
        return float(0.5 * erfc(z / math.sqrt(2.0)))


def bh_adjust(pvals: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    m = len(pvals)
    idx = np.argsort(pvals)
    p_sorted = np.array(pvals, dtype=float)[idx]
    q_sorted = np.empty(m, dtype=float)
    running = 1.0
    for j in range(m, 0, -1):
        val = (p_sorted[j - 1] * m) / j
        running = min(running, val)
        q_sorted[j - 1] = running
    qvals = np.empty(m, dtype=float)
    qvals[idx] = q_sorted

    sig_sorted = p_sorted <= (np.arange(1, m + 1) / m) * alpha
    k = int(np.max(np.where(sig_sorted)[0]) + 1) if np.any(sig_sorted) else 0
    sig = np.zeros(m, dtype=bool)
    if k > 0:
        sig[idx[:k]] = True

    return [float(x) for x in qvals.tolist()], [bool(x) for x in sig.tolist()]


def gate_significance_by_min_pooled_T(
    z_raw: List[float],
    pooled_T: np.ndarray,
    *,
    min_pooled_T: int,
    alpha: float = 0.05,
) -> Tuple[List[float], List[float], List[float], List[bool], List[bool]]:
    """
    Gate pooled significance by minimum evidence:
      - if T < min_pooled_T: force z=0, p=1, BH_sig=False
      - else: keep raw z and p=sf(z)
    """
    min_t = max(0, int(min_pooled_T))
    z_gated: List[float] = []
    pvals_gated: List[float] = []
    eligible: List[bool] = []
    for z, T in zip(z_raw, pooled_T.tolist()):
        ok = int(T) >= min_t
        eligible.append(bool(ok))
        if ok:
            z_gated.append(float(z))
            pvals_gated.append(float(norm_sf(float(z))))
        else:
            z_gated.append(0.0)
            pvals_gated.append(1.0)
    qvals, bh_sig = bh_adjust(pvals_gated, alpha=alpha)
    return z_gated, pvals_gated, qvals, bh_sig, eligible


def pooled_z(det: MultiDirBucketDetector, pooled_T: np.ndarray, pooled_sumproj: np.ndarray) -> List[float]:
    D = det.embedder.st.get_sentence_embedding_dimension()
    out: List[float] = []
    for T, s in zip(pooled_T.tolist(), pooled_sumproj.tolist()):
        if int(T) <= 0:
            out.append(0.0)
        else:
            mean_proj = float(s / T)
            out.append(mean_proj * math.sqrt(D * int(T)))
    return out


def pooled_z_kgw(pooled_T: np.ndarray, pooled_greens: np.ndarray, gamma: float) -> List[float]:
    out: List[float] = []
    g = float(gamma)
    for T, G in zip(pooled_T.tolist(), pooled_greens.tolist()):
        T_i = int(T)
        if T_i <= 0:
            out.append(0.0)
            continue
        mu = g * T_i
        var = g * (1.0 - g) * T_i + 1e-12
        z = (float(G) - mu) / math.sqrt(var)
        out.append(float(z))
    return out


def pooled_z_centered_empirical(
    pooled_T: np.ndarray,
    pooled_sum: np.ndarray,
    pooled_sumsq: np.ndarray,
    *,
    min_var: float = 1e-8,
) -> List[float]:
    """
    Standardize centered per-unit provider scores using empirical variance.

    For each provider p, we assume we pooled centered evidence increments x_{t,p}
    (centered across providers within unit), and maintain:
      T_p = count, S_p = sum x, SS_p = sum x^2.
    We report z_p = mean / (std/sqrt(T)).
    """
    out: List[float] = []
    for T_i, S_i, SS_i in zip(pooled_T.tolist(), pooled_sum.tolist(), pooled_sumsq.tolist()):
        T = int(T_i)
        if T <= 1:
            out.append(0.0)
            continue
        mean = float(S_i) / float(T)
        # Unbiased sample variance
        var = (float(SS_i) - (float(S_i) * float(S_i) / float(T))) / float(max(1, T - 1))
        var = max(float(min_var), float(var))
        z = mean / math.sqrt(var / float(T))
        out.append(float(z))
    return out


def detect_dirbucket_answer_only(
    *,
    det: MultiDirBucketDetector,
    text: str,
    win_words: int,
    stride_words: int,
    competitive: bool,
    tie_margin: Optional[float],
) -> Tuple[Dict[int, DetectSummary], Dict[str, Any]]:
    """
    Black-box answer-only detector:
      - build answer windows
      - compute per-provider directional projection for each window
      - if competitive=True, each window contributes to one provider (best score),
        with null-calibrated competitive z to avoid winner-selection bias.
    """
    sents = split_sentences(text)
    wins: List[str] = []
    for s in sents:
        wins.extend(_windows_within_sentence(s, win_words=win_words, stride_words=stride_words))

    pids = sorted(int(pid) for pid in det.provider_keys_by_pid.keys())
    requested_competitive = bool(competitive)
    D = int(det.embedder.st.get_sentence_embedding_dimension())
    if not wins:
        empty = {
            int(pid): DetectSummary(
                T=0,
                D=D,
                mean_proj=0.0,
                z=0.0,
                p_one_sided=1.0,
                proj_mean=0.0,
                proj_median=0.0,
                proj_p90=0.0,
                extras={"mode": "answer_only"},
            )
            for pid in pids
        }
        meta = {
            "mode": "answer_only",
            "assignment_mode": ("competitive" if requested_competitive else "non-competitive"),
            "requested_competitive": requested_competitive,
            "windows_total": 0,
            "assigned_total": 0,
            "dropped_below_thresh_windows": 0,
            "per_provider_assigned": {str(pid): 0 for pid in pids},
            "best_cos_stats": {"mean": None},
            "second_best_cos_stats": {"mean": None},
            "tie_margin": (float(tie_margin) if (requested_competitive and tie_margin is not None) else None),
            "null_mu": 0.0,
            "null_var": 0.0,
            "comp_stat_by_pid": [0.0 for _ in pids],
        }
        return empty, meta

    U = det.embedder.embed(wins)
    if requested_competitive:
        pid_to_idx = {int(pid): i for i, pid in enumerate(pids)}
        comp_stat = np.zeros(len(pids), dtype=np.float64)  # sum(top1_score where pid wins)
        assigned_counts = np.zeros(len(pids), dtype=np.int64)

        dropped = 0
        best_vals: List[float] = []
        second_vals: List[float] = []

        tie_margin_eff = None
        if tie_margin is not None:
            # Projection scale is roughly O(1/sqrt(D)); rescale user tie margin accordingly.
            tie_margin_eff = float(tie_margin) / max(1.0, math.sqrt(float(D)))

        top1_sum = 0.0
        top1_sq_sum = 0.0

        for i in range(U.shape[0]):
            u = U[i]
            b = det.bucket.code(u)
            scores: List[Tuple[int, float]] = []
            for pid in pids:
                v = det._direction(int(pid), b)  # internal helper on detector
                scores.append((int(pid), float(np.dot(u, v))))
            scores.sort(key=lambda t: t[1], reverse=True)

            pid_best, s_best = scores[0]
            s_second = scores[1][1] if len(scores) > 1 else -1e9
            best_vals.append(float(s_best))
            if len(scores) > 1:
                second_vals.append(float(s_second))

            if tie_margin_eff is not None and (s_best - s_second) < float(tie_margin_eff):
                dropped += 1
                continue

            idx = pid_to_idx[int(pid_best)]
            comp_stat[idx] += float(s_best)
            assigned_counts[idx] += 1
            top1_sum += float(s_best)
            top1_sq_sum += float(s_best) * float(s_best)

        p0 = 1.0 / max(1, len(pids))
        null_mu = float(top1_sum * p0)
        null_var = float(top1_sq_sum * p0 * (1.0 - p0))
        denom = math.sqrt(null_var + 1e-12)

        res_by_pid: Dict[int, DetectSummary] = {}
        for pid in pids:
            idx = pid_to_idx[int(pid)]
            stat = float(comp_stat[idx])
            T = int(assigned_counts[idx])
            mean_proj = stat / max(1, T)
            z = float((stat - null_mu) / denom) if denom > 0 else 0.0
            p = norm_sf(z)
            res_by_pid[int(pid)] = DetectSummary(
                T=T,
                D=D,
                mean_proj=mean_proj,
                z=float(z),
                p_one_sided=float(p),
                proj_mean=float(mean_proj),
                proj_median=float(mean_proj),
                proj_p90=float(mean_proj),
                extras={"mode": "answer_only", "score_pool": "winner_sum_calibrated"},
            )

        meta = {
            "mode": "answer_only",
            "assignment_mode": "competitive",
            "requested_competitive": True,
            "windows_total": int(len(wins)),
            "assigned_total": int(np.sum(assigned_counts)),
            "dropped_below_thresh_windows": int(dropped),
            "per_provider_assigned": {
                str(pid): int(assigned_counts[pid_to_idx[int(pid)]]) for pid in pids
            },
            "best_cos_stats": {"mean": (float(np.mean(best_vals)) if best_vals else None)},
            "second_best_cos_stats": {"mean": (float(np.mean(second_vals)) if second_vals else None)},
            "tie_margin": (float(tie_margin_eff) if tie_margin_eff is not None else None),
            "null_mu": float(null_mu),
            "null_var": float(null_var),
            "comp_stat_by_pid": [float(comp_stat[pid_to_idx[int(pid)]]) for pid in pids],
        }
        return res_by_pid, meta

    # Non-competitive answer-only baseline: every provider scores every answer window.
    proj_by_pid: Dict[int, List[float]] = {int(pid): [] for pid in pids}
    for i in range(U.shape[0]):
        u = U[i]
        b = det.bucket.code(u)
        for pid in pids:
            v = det._direction(int(pid), b)
            proj_by_pid[int(pid)].append(float(np.dot(u, v)))

    res_by_pid: Dict[int, DetectSummary] = {}
    for pid in pids:
        arr = np.array(proj_by_pid[int(pid)], dtype=np.float32)
        T = int(arr.size)
        mean_proj = float(arr.mean()) if T > 0 else 0.0
        z, p = det._z(mean_proj, T)
        res_by_pid[int(pid)] = DetectSummary(
            T=T,
            D=D,
            mean_proj=mean_proj,
            z=float(z),
            p_one_sided=float(p),
            proj_mean=float(arr.mean()) if T > 0 else 0.0,
            proj_median=float(np.median(arr)) if T > 0 else 0.0,
            proj_p90=float(np.percentile(arr, 90)) if T > 0 else 0.0,
            extras={"mode": "answer_only", "score_pool": "all_windows"},
        )

    meta = {
        "mode": "answer_only",
        "assignment_mode": "non-competitive",
        "requested_competitive": False,
        "windows_total": int(len(wins)),
        "assigned_total": int(len(wins)),
        "dropped_below_thresh_windows": 0,
        "per_provider_assigned": {str(pid): int(len(proj_by_pid[int(pid)])) for pid in pids},
        "best_cos_stats": {"mean": None},
        "second_best_cos_stats": {"mean": None},
        "tie_margin": None,
        "null_mu": 0.0,
        "null_var": 0.0,
        "comp_stat_by_pid": [0.0 for _ in pids],
    }
    return res_by_pid, meta


# ---------------- Experiment core ----------------
def run_dirbucket_audit(
    *,
    work_dir: Path,
    rows: List[Dict[str, Any]],
    keys: List[str],
    source_field: str,
    watermarked_field: str,
    retrieval_model: str,
    target_provider: int,
    trials: int,
    topk: int,
    min_unique_providers: int,
    force_cross_provider: bool,
    exclude_target_provider_from_retrieval: bool,
    answer_style: str,
    openai_model: str,
    answer_temp: float,
    bucket_bits: int,
    bucket_seed: int,
    embed_model: str,
    win_words: int,
    stride_words: int,
    sim_threshold: float,
    competitive: bool,
    tie_margin: float,
    neighbor_smoothing: bool,
    ham_radius: int,
    smooth_agg: str,
    audit_mode: str,
    aligned_source_mode: str,
    alignment_manifest: Optional[Path],
    global_source_query_sim_threshold: float,
    global_source_max_docs: int,
    global_source_max_docs_per_provider: int,
    global_source_selector: str,
    direction_scheme: str,
    do_wrongkey: bool,
    null_unrelated: bool,
    z_threshold: float,
    min_pooled_T: int,
    log_traj: bool,
    seed: int,
    wm_scheme: str,
    trial_log_jsonl: Optional[Path],
    resume_trials: bool,
) -> Dict[str, Any]:
    P = len(keys)
    if P <= 0:
        raise RuntimeError("No provider keys loaded")
    if not (0 <= target_provider < P):
        raise ValueError(f"--target_provider={target_provider} out of range (P={P})")

    X, enc = ensure_embeddings(work_dir, rows, field=source_field, model_name=retrieval_model)
    aligned_source_mode = _canon_aligned_source_mode(aligned_source_mode)
    global_source_selector = _canon_global_source_selector(global_source_selector)
    if aligned_source_mode not in {"oracle", "retrieved_src", "manifest_src", "global_corpus"}:
        raise ValueError(
            f"Unknown --aligned_source_mode={aligned_source_mode}. "
            f"Use one of: oracle, global_corpus, retrieved_src, manifest_src."
        )
    align_manifest_by_id: Dict[str, Dict[str, Any]] = {}
    if audit_mode == "aligned" and aligned_source_mode == "manifest_src":
        if alignment_manifest is None:
            raise ValueError("--alignment_manifest is required when --aligned_source_mode=manifest_src")
        if not alignment_manifest.exists():
            raise FileNotFoundError(f"Alignment manifest not found: {alignment_manifest}")
        align_manifest_by_id = load_alignment_manifest_map(alignment_manifest)

    det = MultiDirBucketDetector(
        provider_keys_by_pid={i: keys[i] for i in range(P)},
        bucket_bits=bucket_bits,
        bucket_seed=bucket_seed,
        embed_model=embed_model,
        direction_scheme=direction_scheme,
    )

    wrong_det = None
    wrong_keys = None
    if do_wrongkey:
        wrong_keys = keys[1:] + keys[:1]
        wrong_det = MultiDirBucketDetector(
            provider_keys_by_pid={i: wrong_keys[i] for i in range(P)},
            bucket_bits=bucket_bits,
            bucket_seed=bucket_seed,
            embed_model=embed_model,
            direction_scheme=direction_scheme,
        )

    candidates = [r for r in rows if int(r.get("provider_id", -1)) == int(target_provider)]
    if not candidates:
        raise RuntimeError(f"No candidate docs found for provider {target_provider}")

    pooled_T = np.zeros(P, dtype=np.int64)
    pooled_sumproj = np.zeros(P, dtype=np.float64)
    wrong_pooled_T = np.zeros(P, dtype=np.int64)
    wrong_pooled_sumproj = np.zeros(P, dtype=np.float64)
    winners: List[int] = []
    mix_counts: List[int] = []

    crossed_at: List[Optional[int]] = [None] * P
    crossed_at_raw: List[Optional[int]] = [None] * P
    wrong_crossed_at: List[Optional[int]] = [None] * P
    wrong_crossed_at_raw: List[Optional[int]] = [None] * P

    assign_total = 0
    windows_total = 0
    dropped_total = 0
    best_cos_means: List[float] = []
    second_cos_means: List[float] = []
    source_docs_selected: List[int] = []
    source_providers_selected: List[int] = []
    source_above_threshold_docs: List[int] = []
    source_sim_means: List[float] = []
    traj: List[Dict[str, Any]] = []
    use_comp_pool = bool(audit_mode == "answer_only" and competitive)
    pooled_comp_stat = np.zeros(P, dtype=np.float64)
    pooled_comp_mu = 0.0
    pooled_comp_var = 0.0
    wrong_pooled_comp_stat = np.zeros(P, dtype=np.float64)
    wrong_pooled_comp_mu = 0.0
    wrong_pooled_comp_var = 0.0

    def _update_crossed(
        crossed: List[Optional[int]],
        crossed_raw: List[Optional[int]],
        z_vec: List[float],
        T_vec: np.ndarray,
        q_eff: int,
    ) -> None:
        for pid in range(P):
            T = int(T_vec[pid])
            z_pool = float(z_vec[pid])
            if crossed_raw[pid] is None and z_pool >= z_threshold:
                crossed_raw[pid] = int(q_eff)
            if crossed[pid] is None and T >= int(min_pooled_T) and z_pool >= z_threshold:
                crossed[pid] = int(q_eff)

    def _pooled_z_now() -> List[float]:
        if use_comp_pool:
            denom = math.sqrt(max(float(pooled_comp_var), 1e-12))
            return [float((float(pooled_comp_stat[pid]) - float(pooled_comp_mu)) / denom) for pid in range(P)]
        return pooled_z(det, pooled_T, pooled_sumproj)

    def _pooled_z_wrong_now() -> Optional[List[float]]:
        if wrong_det is None:
            return None
        if use_comp_pool:
            denom = math.sqrt(max(float(wrong_pooled_comp_var), 1e-12))
            return [
                float((float(wrong_pooled_comp_stat[pid]) - float(wrong_pooled_comp_mu)) / denom)
                for pid in range(P)
            ]
        return pooled_z(wrong_det, wrong_pooled_T, wrong_pooled_sumproj)

    # Optional trial-log resume
    existing_trials: List[Dict[str, Any]] = (
        read_trial_log(trial_log_jsonl, scheme=wm_scheme, audit_mode=audit_mode)
        if resume_trials
        else []
    )
    if existing_trials:
        last = existing_trials[-1]
        cum = last.get("cumulative", {}) or {}
        try:
            pt = cum.get("pooled_T")
            ps = cum.get("pooled_sumproj")
            if isinstance(pt, list) and isinstance(ps, list) and len(pt) == P and len(ps) == P:
                pooled_T = np.array(pt, dtype=np.int64)
                pooled_sumproj = np.array(ps, dtype=np.float64)
                if wrong_det is not None:
                    wpt = cum.get("wrong_pooled_T")
                    wps = cum.get("wrong_pooled_sumproj")
                    if isinstance(wpt, list) and isinstance(wps, list) and len(wpt) == P and len(wps) == P:
                        wrong_pooled_T = np.array(wpt, dtype=np.int64)
                        wrong_pooled_sumproj = np.array(wps, dtype=np.float64)
                assign_total = int(cum.get("assign_total", 0) or 0)
                windows_total = int(cum.get("windows_total", 0) or 0)
                dropped_total = int(cum.get("dropped_total", 0) or 0)
                if use_comp_pool:
                    pcs = cum.get("pooled_comp_stat")
                    pcm = cum.get("pooled_comp_mu")
                    pcv = cum.get("pooled_comp_var")
                    if isinstance(pcs, list) and len(pcs) == P:
                        pooled_comp_stat = np.array(pcs, dtype=np.float64)
                    if pcm is not None:
                        pooled_comp_mu = float(pcm)
                    if pcv is not None:
                        pooled_comp_var = float(pcv)
                    if wrong_det is not None:
                        wpcs = cum.get("wrong_pooled_comp_stat")
                        wpcm = cum.get("wrong_pooled_comp_mu")
                        wpcv = cum.get("wrong_pooled_comp_var")
                        if isinstance(wpcs, list) and len(wpcs) == P:
                            wrong_pooled_comp_stat = np.array(wpcs, dtype=np.float64)
                        if wpcm is not None:
                            wrong_pooled_comp_mu = float(wpcm)
                        if wpcv is not None:
                            wrong_pooled_comp_var = float(wpcv)
        except Exception:
            existing_trials = []

        for tr in existing_trials:
            w = tr.get("winner_pid")
            if isinstance(w, int) and w >= 0:
                winners.append(int(w))
            m = tr.get("mix_unique_providers")
            if isinstance(m, int):
                mix_counts.append(int(m))
            al = tr.get("align", {}) or {}
            if al.get("best_cos_mean") is not None:
                best_cos_means.append(float(al["best_cos_mean"]))
            if al.get("second_cos_mean") is not None:
                second_cos_means.append(float(al["second_cos_mean"]))
            srcdiag = tr.get("aligned_source") or {}
            if isinstance(srcdiag, dict):
                sd = srcdiag.get("selected_docs")
                sp = srcdiag.get("selected_providers")
                sat = srcdiag.get("above_threshold_docs")
                sm = srcdiag.get("selected_sim_mean")
                if isinstance(sd, int):
                    source_docs_selected.append(int(sd))
                if isinstance(sp, int):
                    source_providers_selected.append(int(sp))
                if isinstance(sat, int):
                    source_above_threshold_docs.append(int(sat))
                if isinstance(sm, (int, float)):
                    source_sim_means.append(float(sm))
            if log_traj:
                snap = tr.get("traj_snapshot")
                if isinstance(snap, dict):
                    traj.append(snap)

        for tr in existing_trials:
            cum = tr.get("cumulative", {}) or {}
            z_vec = cum.get("z")
            t_vec = cum.get("pooled_T")
            q_eff = tr.get("q_effective")
            if isinstance(z_vec, list) and isinstance(t_vec, list) and len(z_vec) == P and len(t_vec) == P:
                _update_crossed(
                    crossed=crossed_at,
                    crossed_raw=crossed_at_raw,
                    z_vec=[float(z) for z in z_vec],
                    T_vec=np.array(t_vec, dtype=np.int64),
                    q_eff=int(q_eff) if isinstance(q_eff, int) else len(winners),
                )
            if wrong_det is not None:
                z_w = cum.get("z_wrong")
                t_w = cum.get("wrong_pooled_T")
                if isinstance(z_w, list) and isinstance(t_w, list) and len(z_w) == P and len(t_w) == P:
                    _update_crossed(
                        crossed=wrong_crossed_at,
                        crossed_raw=wrong_crossed_at_raw,
                        z_vec=[float(z) for z in z_w],
                        T_vec=np.array(t_w, dtype=np.int64),
                        q_eff=int(q_eff) if isinstance(q_eff, int) else len(winners),
                    )

    start_trial = 1
    if existing_trials:
        start_trial = int(existing_trials[-1]["trial_idx"]) + 1

    if use_comp_pool:
        print(
            "[INFO] DirBucket answer_only uses competitive assignment with null-calibrated pooling.",
            flush=True,
        )
    if exclude_target_provider_from_retrieval:
        print(
            "[INFO] Retrieval excludes target-provider docs (target-absent stress mode).",
            flush=True,
        )
    if audit_mode == "aligned":
        extra = ""
        if aligned_source_mode == "manifest_src" and alignment_manifest is not None:
            extra = f", manifest={alignment_manifest}"
        if aligned_source_mode == "global_corpus":
            extra = (
                f", selector={global_source_selector}, "
                f"query_sim_threshold={float(global_source_query_sim_threshold):.3f}, "
                f"max_docs={int(global_source_max_docs)}, "
                f"max_docs_per_provider={int(global_source_max_docs_per_provider)}"
            )
        print(
            f"[INFO] DirBucket aligned mode uses answer + source docs "
            f"(source_mode={aligned_source_mode}{extra}).",
            flush=True,
        )

    pbar = tqdm(range(start_trial, trials + 1), desc=f"audit:{answer_style}")
    for trial_idx in pbar:
        trial_rng = random.Random((int(seed) * 1315423911 + int(trial_idx) * 2654435761) & 0xFFFFFFFF)
        target = trial_rng.choice(candidates)

        q_doc_text = (target.get(source_field) or target.get(watermarked_field) or "")
        if not q_doc_text:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "status": "skipped_no_query_doc",
                    "winner_pid": None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_sumproj": pooled_sumproj.tolist(),
                        "pooled_comp_stat": (pooled_comp_stat.tolist() if use_comp_pool else None),
                        "pooled_comp_mu": (float(pooled_comp_mu) if use_comp_pool else None),
                        "pooled_comp_var": (float(pooled_comp_var) if use_comp_pool else None),
                        "wrong_pooled_T": wrong_pooled_T.tolist() if wrong_det is not None else None,
                        "wrong_pooled_sumproj": wrong_pooled_sumproj.tolist() if wrong_det is not None else None,
                        "wrong_pooled_comp_stat": (wrong_pooled_comp_stat.tolist() if (use_comp_pool and wrong_det is not None) else None),
                        "wrong_pooled_comp_mu": (float(wrong_pooled_comp_mu) if (use_comp_pool and wrong_det is not None) else None),
                        "wrong_pooled_comp_var": (float(wrong_pooled_comp_var) if (use_comp_pool and wrong_det is not None) else None),
                        "z": _pooled_z_now(),
                        "z_wrong": _pooled_z_wrong_now(),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        question = gen_question_openai(q_doc_text, model=openai_model)
        query_vec = enc.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]

        picked_idxs = select_retrieval_bundle_idxs(
            rows=rows,
            X=X,
            enc=enc,
            question=question,
            query_vec=query_vec,
            target_doc_id=str(target["id"]),
            target_provider=int(target_provider),
            k=int(topk),
            min_unique_providers=int(min_unique_providers),
            force_cross_provider=bool(force_cross_provider),
            null_unrelated=bool(null_unrelated),
            exclude_target_provider_from_retrieval=bool(exclude_target_provider_from_retrieval),
            rng=trial_rng,
        )

        providers_in_bundle = [int(rows[i]["provider_id"]) for i in picked_idxs]
        mix_counts.append(len(set(providers_in_bundle)))

        prompt_docs: List[Tuple[int, str]] = []
        retrieved_docs: List[Dict[str, Any]] = []
        for i in picked_idxs:
            pid = int(rows[i]["provider_id"])
            txt = (rows[i].get(watermarked_field) or "")
            if txt:
                prompt_docs.append((pid, txt))
            retrieved_docs.append({"doc_id": str(rows[i].get("id")), "provider_id": pid})

        if not prompt_docs:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "target_doc_id": str(target.get("id")),
                    "status": "skipped_no_prompt_docs",
                    "retrieved_docs": retrieved_docs,
                    "winner_pid": None,
                    "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_sumproj": pooled_sumproj.tolist(),
                        "pooled_comp_stat": (pooled_comp_stat.tolist() if use_comp_pool else None),
                        "pooled_comp_mu": (float(pooled_comp_mu) if use_comp_pool else None),
                        "pooled_comp_var": (float(pooled_comp_var) if use_comp_pool else None),
                        "wrong_pooled_T": wrong_pooled_T.tolist() if wrong_det is not None else None,
                        "wrong_pooled_sumproj": wrong_pooled_sumproj.tolist() if wrong_det is not None else None,
                        "wrong_pooled_comp_stat": (wrong_pooled_comp_stat.tolist() if (use_comp_pool and wrong_det is not None) else None),
                        "wrong_pooled_comp_mu": (float(wrong_pooled_comp_mu) if (use_comp_pool and wrong_det is not None) else None),
                        "wrong_pooled_comp_var": (float(wrong_pooled_comp_var) if (use_comp_pool and wrong_det is not None) else None),
                        "z": _pooled_z_now(),
                        "z_wrong": _pooled_z_wrong_now(),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        messages = make_rag_messages(prompt_docs, question, style=answer_style)
        answer = openai_chat(openai_model, messages, temperature=answer_temp).strip()

        sources: List[Tuple[int, str]] = []
        source_diag: Dict[str, Any] = {}
        answer_vec: Optional[np.ndarray] = None
        if (
            audit_mode == "aligned"
            and aligned_source_mode == "global_corpus"
            and global_source_selector in {"answer", "max"}
        ):
            answer_vec = enc.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0]
        if audit_mode == "aligned":
            sources, source_diag = build_alignment_sources(
                rows=rows,
                picked_idxs=picked_idxs,
                prompt_docs=prompt_docs,
                retrieved_docs=retrieved_docs,
                source_field=source_field,
                aligned_source_mode=aligned_source_mode,
                align_manifest_by_id=align_manifest_by_id if aligned_source_mode == "manifest_src" else None,
                query_vec=query_vec,
                answer_vec=answer_vec,
                X=X,
                global_source_query_sim_threshold=float(global_source_query_sim_threshold),
                global_source_max_docs=int(global_source_max_docs),
                global_source_max_docs_per_provider=int(global_source_max_docs_per_provider),
                global_source_selector=global_source_selector,
            )

        if audit_mode == "answer_only":
            # Black-box auditor mode: detector only sees final answer text.
            res_by_pid, meta = detect_dirbucket_answer_only(
                det=det,
                text=answer,
                win_words=win_words,
                stride_words=stride_words,
                competitive=competitive,
                tie_margin=tie_margin,
            )
        else:
            res_by_pid, meta = det.detect_windows_multi_aligned(
                text=answer,
                sources=sources,
                sim_threshold=sim_threshold,
                competitive=competitive,
                tie_margin=tie_margin,
                neighbor_smoothing=neighbor_smoothing,
                ham_radius=(ham_radius if neighbor_smoothing else None),
                smooth_agg=smooth_agg,
                win_words=win_words,
                stride_words=stride_words,
            )

        assign_total += int(meta.get("assigned_total", 0) or 0)
        windows_total += int(meta.get("windows_total", 0) or 0)
        dropped_total += int(meta.get("dropped_below_thresh_windows", 0) or 0)
        bs = meta.get("best_cos_stats", {}) or {}
        ss = meta.get("second_best_cos_stats", {}) or {}
        if bs.get("mean") is not None:
            best_cos_means.append(float(bs["mean"]))
        if ss.get("mean") is not None:
            second_cos_means.append(float(ss["mean"]))
        if audit_mode == "aligned":
            sd = source_diag.get("selected_docs")
            sp = source_diag.get("selected_providers")
            sat = source_diag.get("above_threshold_docs")
            sm = source_diag.get("selected_sim_mean")
            if isinstance(sd, int):
                source_docs_selected.append(int(sd))
            if isinstance(sp, int):
                source_providers_selected.append(int(sp))
            if isinstance(sat, int):
                source_above_threshold_docs.append(int(sat))
            if isinstance(sm, (int, float)):
                source_sim_means.append(float(sm))
        if use_comp_pool:
            cs = meta.get("comp_stat_by_pid")
            if isinstance(cs, list) and len(cs) == P:
                pooled_comp_stat += np.array([float(x) for x in cs], dtype=np.float64)
            pooled_comp_mu += float(meta.get("null_mu", 0.0) or 0.0)
            pooled_comp_var += float(meta.get("null_var", 0.0) or 0.0)

        per_answer_z = np.zeros(P, dtype=float)
        per_answer_T = np.zeros(P, dtype=np.int64)
        per_answer_mean_proj = np.zeros(P, dtype=float)
        for pid in range(P):
            detsum = res_by_pid.get(pid)
            if detsum is None:
                continue
            T_pid = int(detsum.T)
            mp = float(detsum.mean_proj)
            pooled_T[pid] += T_pid
            pooled_sumproj[pid] += mp * T_pid
            per_answer_T[pid] = T_pid
            per_answer_mean_proj[pid] = mp
            per_answer_z[pid] = float(detsum.z)
        winner_pid = int(np.argmax(per_answer_z))
        winners.append(winner_pid)

        wrong_per_answer_z = np.zeros(P, dtype=float)
        wrong_per_answer_T = np.zeros(P, dtype=np.int64)
        wrong_per_answer_mean_proj = np.zeros(P, dtype=float)
        if wrong_det is not None:
            wrong_meta: Dict[str, Any] = {}
            if audit_mode == "answer_only":
                res_null, wrong_meta = detect_dirbucket_answer_only(
                    det=wrong_det,
                    text=answer,
                    win_words=win_words,
                    stride_words=stride_words,
                    competitive=competitive,
                    tie_margin=tie_margin,
                )
            else:
                res_null, wrong_meta = wrong_det.detect_windows_multi_aligned(
                    text=answer,
                    sources=sources,
                    sim_threshold=sim_threshold,
                    competitive=competitive,
                    tie_margin=tie_margin,
                    neighbor_smoothing=neighbor_smoothing,
                    ham_radius=(ham_radius if neighbor_smoothing else None),
                    smooth_agg=smooth_agg,
                    win_words=win_words,
                    stride_words=stride_words,
                )
            for pid in range(P):
                nd = res_null.get(pid)
                if nd is None:
                    continue
                T_pid = int(nd.T)
                mp = float(nd.mean_proj)
                wrong_pooled_T[pid] += T_pid
                wrong_pooled_sumproj[pid] += mp * T_pid
                wrong_per_answer_T[pid] = T_pid
                wrong_per_answer_mean_proj[pid] = mp
                wrong_per_answer_z[pid] = float(nd.z)
            if use_comp_pool:
                wcs = wrong_meta.get("comp_stat_by_pid")
                if isinstance(wcs, list) and len(wcs) == P:
                    wrong_pooled_comp_stat += np.array([float(x) for x in wcs], dtype=np.float64)
                wrong_pooled_comp_mu += float(wrong_meta.get("null_mu", 0.0) or 0.0)
                wrong_pooled_comp_var += float(wrong_meta.get("null_var", 0.0) or 0.0)

        z_vec = _pooled_z_now()
        z_w_vec = _pooled_z_wrong_now()
        q_eff = int(len(winners))

        _update_crossed(crossed_at, crossed_at_raw, z_vec=z_vec, T_vec=pooled_T, q_eff=q_eff)
        if wrong_det is not None and z_w_vec is not None:
            _update_crossed(
                wrong_crossed_at,
                wrong_crossed_at_raw,
                z_vec=z_w_vec,
                T_vec=wrong_pooled_T,
                q_eff=q_eff,
            )

        snap: Optional[Dict[str, Any]] = None
        if log_traj:
            snap = {
                "q": q_eff,
                "pooled_T": pooled_T.tolist(),
                "z": [float(x) for x in z_vec],
                "assign_rate_sofar": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
                "mix_unique_providers_last": int(mix_counts[-1]) if mix_counts else None,
            }
            if z_w_vec is not None:
                snap["T_wrong"] = wrong_pooled_T.tolist()
                snap["z_wrong"] = [float(x) for x in z_w_vec]
            traj.append(snap)

        append_trial_log(
            trial_log_jsonl,
            {
                "trial_idx": int(trial_idx),
                "wm_scheme": wm_scheme,
                "audit_mode": str(audit_mode),
                "target_provider": int(target_provider),
                "target_doc_id": str(target.get("id")),
                "question": question,
                "question_hash": sha256_text(question),
                "retrieved_docs": retrieved_docs,
                "answer_style": str(answer_style),
                "answer": answer,
                "answer_hash": sha256_text(answer),
                "answer_tokens_est": int(len(answer.split())),
                "winner_pid": int(winner_pid),
                "q_effective": int(q_eff),
                "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                "align": {
                    "mode": str(audit_mode),
                    "source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
                    "assignment_mode": (
                        str(meta.get("assignment_mode"))
                        if meta.get("assignment_mode") is not None
                        else ("competitive" if competitive else "non-competitive")
                    ),
                    "windows_total": int(meta.get("windows_total", 0) or 0),
                    "assigned_total": int(meta.get("assigned_total", 0) or 0),
                    "dropped_total": int(meta.get("dropped_below_thresh_windows", 0) or 0),
                    "best_cos_mean": (None if bs.get("mean") is None else float(bs["mean"])),
                    "second_cos_mean": (None if ss.get("mean") is None else float(ss["mean"])),
                    "source_selected_docs": (
                        int(source_diag.get("selected_docs", 0)) if audit_mode == "aligned" else None
                    ),
                    "source_selected_providers": (
                        int(source_diag.get("selected_providers", 0)) if audit_mode == "aligned" else None
                    ),
                },
                "aligned_source": (source_diag if audit_mode == "aligned" else None),
                "per_provider": {
                    "T": per_answer_T.tolist(),
                    "mean_proj": [float(x) for x in per_answer_mean_proj.tolist()],
                    "z": [float(x) for x in per_answer_z.tolist()],
                },
                "wrongkey": (
                    {
                        "enabled": True,
                        "T": wrong_per_answer_T.tolist(),
                        "mean_proj": [float(x) for x in wrong_per_answer_mean_proj.tolist()],
                        "z": [float(x) for x in wrong_per_answer_z.tolist()],
                    }
                    if wrong_det is not None
                    else {"enabled": False}
                ),
                "traj_snapshot": snap,
                "cumulative": {
                    "pooled_T": pooled_T.tolist(),
                    "pooled_sumproj": pooled_sumproj.tolist(),
                    "pooled_comp_stat": (pooled_comp_stat.tolist() if use_comp_pool else None),
                    "pooled_comp_mu": (float(pooled_comp_mu) if use_comp_pool else None),
                    "pooled_comp_var": (float(pooled_comp_var) if use_comp_pool else None),
                    "wrong_pooled_T": wrong_pooled_T.tolist() if wrong_det is not None else None,
                    "wrong_pooled_sumproj": wrong_pooled_sumproj.tolist() if wrong_det is not None else None,
                    "wrong_pooled_comp_stat": (wrong_pooled_comp_stat.tolist() if (use_comp_pool and wrong_det is not None) else None),
                    "wrong_pooled_comp_mu": (float(wrong_pooled_comp_mu) if (use_comp_pool and wrong_det is not None) else None),
                    "wrong_pooled_comp_var": (float(wrong_pooled_comp_var) if (use_comp_pool and wrong_det is not None) else None),
                    "z": [float(x) for x in z_vec],
                    "z_wrong": ([float(x) for x in z_w_vec] if z_w_vec is not None else None),
                    "assign_total": int(assign_total),
                    "windows_total": int(windows_total),
                    "dropped_total": int(dropped_total),
                },
            },
        )

    final_z_raw = _pooled_z_now()
    pvals_raw = [norm_sf(z) for z in final_z_raw]
    final_z, pvals, qvals, bh_sig, bh_eligible = gate_significance_by_min_pooled_T(
        final_z_raw,
        pooled_T,
        min_pooled_T=min_pooled_T,
        alpha=0.05,
    )

    wrong_final_z_raw: Optional[List[float]] = _pooled_z_wrong_now()
    wrong_final_z: Optional[List[float]] = None
    if wrong_final_z_raw is not None:
        wrong_final_z = [
            float(z) if int(pooled_T_i) >= int(min_pooled_T) else 0.0
            for z, pooled_T_i in zip(wrong_final_z_raw, wrong_pooled_T.tolist())
        ]

    out: Dict[str, Any] = {
        "P": int(P),
        "target_provider": int(target_provider),
        "trials": int(trials),
        "topk": int(topk),
        "min_unique_providers": int(min_unique_providers),
        "force_cross_provider": bool(force_cross_provider),
        "exclude_target_provider_from_retrieval": bool(exclude_target_provider_from_retrieval),
        "answer_style": str(answer_style),
        "wm_scheme": str(wm_scheme),
        "audit_mode": str(audit_mode),
        "direction_scheme": str(direction_scheme),
        "z_threshold": float(z_threshold),
        "min_pooled_T": int(min_pooled_T),
        "winner_fraction_target": float(np.mean(np.array(winners) == int(target_provider))) if winners else 0.0,
        "mix_unique_providers_mean": float(np.mean(mix_counts)) if mix_counts else 0.0,
        "mix_unique_providers_median": float(np.median(mix_counts)) if mix_counts else 0.0,
        "pooled_T": pooled_T.tolist(),
        "pooled_sumproj": pooled_sumproj.tolist(),
        "pooled_comp_stat": (pooled_comp_stat.tolist() if use_comp_pool else None),
        "pooled_comp_mu": (float(pooled_comp_mu) if use_comp_pool else None),
        "pooled_comp_var": (float(pooled_comp_var) if use_comp_pool else None),
        "z_raw": [float(x) for x in final_z_raw],
        "z": [float(x) for x in final_z],
        "p_one_sided_raw": [float(x) for x in pvals_raw],
        "p_one_sided": [float(x) for x in pvals],
        "q_bh": [float(x) for x in qvals],
        "bh_sig": [bool(x) for x in bh_sig],
        "bh_eligible": [bool(x) for x in bh_eligible],
        "crossed_at": crossed_at,
        "crossed_at_raw": crossed_at_raw,
        "align": {
            "mode": str(audit_mode),
            "source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
            "assignment_mode": ("competitive" if competitive else "non-competitive"),
            "windows_total": int(windows_total),
            "assigned_total": int(assign_total),
            "dropped_total": int(dropped_total),
            "assign_rate": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
            "best_cos_mean": float(np.mean(best_cos_means)) if best_cos_means else None,
            "second_cos_mean": float(np.mean(second_cos_means)) if second_cos_means else None,
            "source_docs_selected_mean": (
                float(np.mean(source_docs_selected)) if source_docs_selected else None
            ),
            "source_docs_selected_median": (
                float(np.median(source_docs_selected)) if source_docs_selected else None
            ),
            "source_docs_selected_p90": (
                float(np.percentile(source_docs_selected, 90)) if source_docs_selected else None
            ),
            "source_providers_selected_mean": (
                float(np.mean(source_providers_selected)) if source_providers_selected else None
            ),
            "global_above_threshold_docs_mean": (
                float(np.mean(source_above_threshold_docs)) if source_above_threshold_docs else None
            ),
            "global_selected_sim_mean": (
                float(np.mean(source_sim_means)) if source_sim_means else None
            ),
            "global_query_sim_threshold": (
                float(global_source_query_sim_threshold)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_max_docs": (
                int(global_source_max_docs)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_max_docs_per_provider": (
                int(global_source_max_docs_per_provider)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_source_selector": (
                str(global_source_selector)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
        },
        "wrongkey": {
            "enabled": bool(wrong_det is not None),
            "z_raw": [float(x) for x in wrong_final_z_raw] if wrong_final_z_raw is not None else None,
            "z": [float(x) for x in wrong_final_z] if wrong_final_z is not None else None,
            "pooled_T": wrong_pooled_T.tolist() if wrong_det is not None else None,
            "pooled_sumproj": wrong_pooled_sumproj.tolist() if wrong_det is not None else None,
            "pooled_comp_stat": (wrong_pooled_comp_stat.tolist() if (use_comp_pool and wrong_det is not None) else None),
            "pooled_comp_mu": (float(wrong_pooled_comp_mu) if (use_comp_pool and wrong_det is not None) else None),
            "pooled_comp_var": (float(wrong_pooled_comp_var) if (use_comp_pool and wrong_det is not None) else None),
            "crossed_at": wrong_crossed_at if wrong_det is not None else None,
            "crossed_at_raw": wrong_crossed_at_raw if wrong_det is not None else None,
        },
        "traj": traj if log_traj else None,
    }
    return out


def run_kgw_audit(
    *,
    work_dir: Path,
    rows: List[Dict[str, Any]],
    keys: List[str],
    source_field: str,
    watermarked_field: str,
    retrieval_model: str,
    target_provider: int,
    trials: int,
    topk: int,
    min_unique_providers: int,
    force_cross_provider: bool,
    exclude_target_provider_from_retrieval: bool,
    answer_style: str,
    openai_model: str,
    answer_temp: float,
    kgw_tokenizer_model: str,
    kgw_params: KGWParams,
    kgw_n_min: int,
    competitive: bool,
    audit_mode: str,
    aligned_source_mode: str,
    alignment_manifest: Optional[Path],
    global_source_query_sim_threshold: float,
    global_source_max_docs: int,
    global_source_max_docs_per_provider: int,
    global_source_selector: str,
    do_wrongkey: bool,
    null_unrelated: bool,
    z_threshold: float,
    min_pooled_T: int,
    log_traj: bool,
    seed: int,
    wm_scheme: str,
    trial_log_jsonl: Optional[Path],
    resume_trials: bool,
) -> Dict[str, Any]:
    P = len(keys)
    if P <= 0:
        raise RuntimeError("No provider keys loaded")
    if not (0 <= target_provider < P):
        raise ValueError(f"--target_provider={target_provider} out of range (P={P})")

    X, enc = ensure_embeddings(work_dir, rows, field=source_field, model_name=retrieval_model)
    tok = get_tokenizer_cached(kgw_tokenizer_model)
    aligned_source_mode = _canon_aligned_source_mode(aligned_source_mode)
    global_source_selector = _canon_global_source_selector(global_source_selector)
    if aligned_source_mode not in {"oracle", "retrieved_src", "manifest_src", "global_corpus"}:
        raise ValueError(
            f"Unknown --aligned_source_mode={aligned_source_mode}. "
            f"Use one of: oracle, global_corpus, retrieved_src, manifest_src."
        )
    align_manifest_by_id: Dict[str, Dict[str, Any]] = {}
    if audit_mode == "aligned" and aligned_source_mode == "manifest_src":
        if alignment_manifest is None:
            raise ValueError("--alignment_manifest is required when --aligned_source_mode=manifest_src")
        if not alignment_manifest.exists():
            raise FileNotFoundError(f"Alignment manifest not found: {alignment_manifest}")
        align_manifest_by_id = load_alignment_manifest_map(alignment_manifest)

    key_map = {i: keys[i] for i in range(P)}
    wrong_key_map = None
    if do_wrongkey:
        wrong_keys = keys[1:] + keys[:1]
        wrong_key_map = {i: wrong_keys[i] for i in range(P)}

    candidates = [r for r in rows if int(r.get("provider_id", -1)) == int(target_provider)]
    if not candidates:
        raise RuntimeError(f"No candidate docs found for provider {target_provider}")

    pooled_T = np.zeros(P, dtype=np.int64)
    pooled_greens = np.zeros(P, dtype=np.int64)
    wrong_pooled_T = np.zeros(P, dtype=np.int64)
    wrong_pooled_greens = np.zeros(P, dtype=np.int64)
    winners: List[int] = []
    mix_counts: List[int] = []

    crossed_at: List[Optional[int]] = [None] * P
    crossed_at_raw: List[Optional[int]] = [None] * P
    wrong_crossed_at: List[Optional[int]] = [None] * P
    wrong_crossed_at_raw: List[Optional[int]] = [None] * P

    assign_total = 0
    windows_total = 0
    dropped_total = 0
    source_docs_selected: List[int] = []
    source_providers_selected: List[int] = []
    source_above_threshold_docs: List[int] = []
    source_sim_means: List[float] = []
    traj: List[Dict[str, Any]] = []

    def _update_crossed(
        crossed: List[Optional[int]],
        crossed_raw: List[Optional[int]],
        z_vec: List[float],
        T_vec: np.ndarray,
        q_eff: int,
    ) -> None:
        for pid in range(P):
            T = int(T_vec[pid])
            z_pool = float(z_vec[pid])
            if crossed_raw[pid] is None and z_pool >= z_threshold:
                crossed_raw[pid] = int(q_eff)
            if crossed[pid] is None and T >= int(min_pooled_T) and z_pool >= z_threshold:
                crossed[pid] = int(q_eff)

    existing_trials: List[Dict[str, Any]] = (
        read_trial_log(trial_log_jsonl, scheme=wm_scheme, audit_mode=audit_mode)
        if resume_trials
        else []
    )
    if existing_trials:
        last = existing_trials[-1]
        cum = last.get("cumulative", {}) or {}
        try:
            pt = cum.get("pooled_T")
            pg = cum.get("pooled_greens")
            if isinstance(pt, list) and isinstance(pg, list) and len(pt) == P and len(pg) == P:
                pooled_T = np.array(pt, dtype=np.int64)
                pooled_greens = np.array(pg, dtype=np.int64)
                if do_wrongkey:
                    wpt = cum.get("wrong_pooled_T")
                    wpg = cum.get("wrong_pooled_greens")
                    if isinstance(wpt, list) and isinstance(wpg, list) and len(wpt) == P and len(wpg) == P:
                        wrong_pooled_T = np.array(wpt, dtype=np.int64)
                        wrong_pooled_greens = np.array(wpg, dtype=np.int64)
                assign_total = int(cum.get("assign_total", 0) or 0)
                windows_total = int(cum.get("windows_total", 0) or 0)
                dropped_total = int(cum.get("dropped_total", 0) or 0)
        except Exception:
            existing_trials = []

        for tr in existing_trials:
            w = tr.get("winner_pid")
            if isinstance(w, int) and w >= 0:
                winners.append(int(w))
            m = tr.get("mix_unique_providers")
            if isinstance(m, int):
                mix_counts.append(int(m))
            srcdiag = tr.get("aligned_source") or {}
            if isinstance(srcdiag, dict):
                sd = srcdiag.get("selected_docs")
                sp = srcdiag.get("selected_providers")
                sat = srcdiag.get("above_threshold_docs")
                sm = srcdiag.get("selected_sim_mean")
                if isinstance(sd, int):
                    source_docs_selected.append(int(sd))
                if isinstance(sp, int):
                    source_providers_selected.append(int(sp))
                if isinstance(sat, int):
                    source_above_threshold_docs.append(int(sat))
                if isinstance(sm, (int, float)):
                    source_sim_means.append(float(sm))
            if log_traj:
                snap = tr.get("traj_snapshot")
                if isinstance(snap, dict):
                    traj.append(snap)

        for tr in existing_trials:
            cum = tr.get("cumulative", {}) or {}
            z_vec = cum.get("z")
            t_vec = cum.get("pooled_T")
            q_eff = tr.get("q_effective")
            if isinstance(z_vec, list) and isinstance(t_vec, list) and len(z_vec) == P and len(t_vec) == P:
                _update_crossed(
                    crossed=crossed_at,
                    crossed_raw=crossed_at_raw,
                    z_vec=[float(z) for z in z_vec],
                    T_vec=np.array(t_vec, dtype=np.int64),
                    q_eff=int(q_eff) if isinstance(q_eff, int) else len(winners),
                )
            if do_wrongkey:
                z_w = cum.get("z_wrong")
                t_w = cum.get("wrong_pooled_T")
                if isinstance(z_w, list) and isinstance(t_w, list) and len(z_w) == P and len(t_w) == P:
                    _update_crossed(
                        crossed=wrong_crossed_at,
                        crossed_raw=wrong_crossed_at_raw,
                        z_vec=[float(z) for z in z_w],
                        T_vec=np.array(t_w, dtype=np.int64),
                        q_eff=int(q_eff) if isinstance(q_eff, int) else len(winners),
                    )

    start_trial = 1
    if existing_trials:
        start_trial = int(existing_trials[-1]["trial_idx"]) + 1

    if audit_mode == "answer_only" and competitive:
        print(
            "[INFO] KGW answer_only scores full answers under every provider key; "
            "--competitive does not change token assignment in this mode.",
            flush=True,
        )
    if exclude_target_provider_from_retrieval:
        print(
            "[INFO] Retrieval excludes target-provider docs (target-absent stress mode).",
            flush=True,
        )
    if audit_mode == "aligned":
        extra = ""
        if aligned_source_mode == "manifest_src" and alignment_manifest is not None:
            extra = f", manifest={alignment_manifest}"
        if aligned_source_mode == "global_corpus":
            extra = (
                f", selector={global_source_selector}, "
                f"query_sim_threshold={float(global_source_query_sim_threshold):.3f}, "
                f"max_docs={int(global_source_max_docs)}, "
                f"max_docs_per_provider={int(global_source_max_docs_per_provider)}"
            )
        print(
            "[INFO] KGW aligned mode uses source-token alignment (white-box upper bound). "
            f"Use --audit_mode answer_only for strict black-box auditing. "
            f"(source_mode={aligned_source_mode}{extra})",
            flush=True,
        )

    pbar = tqdm(range(start_trial, trials + 1), desc=f"audit:{answer_style}")
    for trial_idx in pbar:
        trial_rng = random.Random((int(seed) * 1315423911 + int(trial_idx) * 2654435761) & 0xFFFFFFFF)
        target = trial_rng.choice(candidates)

        q_doc_text = (target.get(source_field) or target.get(watermarked_field) or "")
        if not q_doc_text:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "status": "skipped_no_query_doc",
                    "winner_pid": None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_greens": pooled_greens.tolist(),
                        "wrong_pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
                        "wrong_pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
                        "z": pooled_z_kgw(pooled_T, pooled_greens, gamma=kgw_params.gamma),
                        "z_wrong": (
                            pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=kgw_params.gamma)
                            if do_wrongkey
                            else None
                        ),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        question = gen_question_openai(q_doc_text, model=openai_model)
        query_vec = enc.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]

        picked_idxs = select_retrieval_bundle_idxs(
            rows=rows,
            X=X,
            enc=enc,
            question=question,
            query_vec=query_vec,
            target_doc_id=str(target["id"]),
            target_provider=int(target_provider),
            k=int(topk),
            min_unique_providers=int(min_unique_providers),
            force_cross_provider=bool(force_cross_provider),
            null_unrelated=bool(null_unrelated),
            exclude_target_provider_from_retrieval=bool(exclude_target_provider_from_retrieval),
            rng=trial_rng,
        )

        providers_in_bundle = [int(rows[i]["provider_id"]) for i in picked_idxs]
        mix_counts.append(len(set(providers_in_bundle)))

        prompt_docs: List[Tuple[int, str]] = []
        retrieved_docs: List[Dict[str, Any]] = []
        for i in picked_idxs:
            pid = int(rows[i]["provider_id"])
            txt = (rows[i].get(watermarked_field) or "")
            if txt:
                prompt_docs.append((pid, txt))
            retrieved_docs.append({"doc_id": str(rows[i].get("id")), "provider_id": pid})
        if not prompt_docs:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "target_doc_id": str(target.get("id")),
                    "status": "skipped_no_prompt_docs",
                    "retrieved_docs": retrieved_docs,
                    "winner_pid": None,
                    "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_greens": pooled_greens.tolist(),
                        "wrong_pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
                        "wrong_pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
                        "z": pooled_z_kgw(pooled_T, pooled_greens, gamma=kgw_params.gamma),
                        "z_wrong": (
                            pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=kgw_params.gamma)
                            if do_wrongkey
                            else None
                        ),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        messages = make_rag_messages(prompt_docs, question, style=answer_style)
        answer = openai_chat(openai_model, messages, temperature=answer_temp).strip()

        sources: List[Tuple[int, str]] = []
        source_diag: Dict[str, Any] = {}
        answer_vec: Optional[np.ndarray] = None
        if (
            audit_mode == "aligned"
            and aligned_source_mode == "global_corpus"
            and global_source_selector in {"answer", "max"}
        ):
            answer_vec = enc.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0]
        if audit_mode == "aligned":
            sources, source_diag = build_alignment_sources(
                rows=rows,
                picked_idxs=picked_idxs,
                prompt_docs=prompt_docs,
                retrieved_docs=retrieved_docs,
                source_field=source_field,
                aligned_source_mode=aligned_source_mode,
                align_manifest_by_id=align_manifest_by_id if aligned_source_mode == "manifest_src" else None,
                query_vec=query_vec,
                answer_vec=answer_vec,
                X=X,
                global_source_query_sim_threshold=float(global_source_query_sim_threshold),
                global_source_max_docs=int(global_source_max_docs),
                global_source_max_docs_per_provider=int(global_source_max_docs_per_provider),
                global_source_selector=global_source_selector,
            )

        if audit_mode == "answer_only":
            # Black-box auditor mode: score answer text directly under each provider key.
            res_by_pid = {
                int(pid): detect_kgw_text(
                    text=answer,
                    key_hex=key_map[int(pid)],
                    tokenizer=tok,
                    params=kgw_params,
                )
                for pid in range(P)
            }
            trial_windows = 0
            if res_by_pid:
                trial_windows = int(next(iter(res_by_pid.values())).T)
            meta = {
                "mode": "answer_only",
                "assignment_mode": "all_providers",
                "windows_total": int(trial_windows),
                "assigned_total": int(trial_windows),
                "dropped_below_thresh_windows": 0,
                "best_cos_stats": {"mean": None},
                "second_best_cos_stats": {"mean": None},
            }
        else:
            res_by_pid, meta = detect_kgw_multi_source_aligned(
                answer_text=answer,
                sources=sources,
                provider_keys_by_pid=key_map,
                tokenizer=tok,
                params=kgw_params,
                n_min=int(kgw_n_min),
                competitive=bool(competitive),
            )
        assign_total += int(meta.get("assigned_total", 0) or 0)
        windows_total += int(meta.get("windows_total", 0) or 0)
        dropped_total += int(meta.get("dropped_below_thresh_windows", 0) or 0)
        if audit_mode == "aligned":
            sd = source_diag.get("selected_docs")
            sp = source_diag.get("selected_providers")
            sat = source_diag.get("above_threshold_docs")
            sm = source_diag.get("selected_sim_mean")
            if isinstance(sd, int):
                source_docs_selected.append(int(sd))
            if isinstance(sp, int):
                source_providers_selected.append(int(sp))
            if isinstance(sat, int):
                source_above_threshold_docs.append(int(sat))
            if isinstance(sm, (int, float)):
                source_sim_means.append(float(sm))

        per_answer_z = np.zeros(P, dtype=float)
        per_answer_T = np.zeros(P, dtype=np.int64)
        per_answer_greens = np.zeros(P, dtype=np.int64)
        for pid in range(P):
            r = res_by_pid.get(pid)
            if r is None:
                continue
            T_pid = int(r.T)
            G_pid = int(r.greens)
            pooled_T[pid] += T_pid
            pooled_greens[pid] += G_pid
            per_answer_T[pid] = T_pid
            per_answer_greens[pid] = G_pid
            per_answer_z[pid] = float(r.z)
        winner_pid = int(np.argmax(per_answer_z))
        winners.append(winner_pid)

        wrong_per_answer_z = np.zeros(P, dtype=float)
        wrong_per_answer_T = np.zeros(P, dtype=np.int64)
        wrong_per_answer_greens = np.zeros(P, dtype=np.int64)
        if do_wrongkey and wrong_key_map is not None:
            if audit_mode == "answer_only":
                wrong_res = {
                    int(pid): detect_kgw_text(
                        text=answer,
                        key_hex=wrong_key_map[int(pid)],
                        tokenizer=tok,
                        params=kgw_params,
                    )
                    for pid in range(P)
                }
            else:
                wrong_res, _ = detect_kgw_multi_source_aligned(
                    answer_text=answer,
                    sources=sources,
                    provider_keys_by_pid=wrong_key_map,
                    tokenizer=tok,
                    params=kgw_params,
                    n_min=int(kgw_n_min),
                    competitive=bool(competitive),
                )
            for pid in range(P):
                r = wrong_res.get(pid)
                if r is None:
                    continue
                T_pid = int(r.T)
                G_pid = int(r.greens)
                wrong_pooled_T[pid] += T_pid
                wrong_pooled_greens[pid] += G_pid
                wrong_per_answer_T[pid] = T_pid
                wrong_per_answer_greens[pid] = G_pid
                wrong_per_answer_z[pid] = float(r.z)

        z_vec = pooled_z_kgw(pooled_T, pooled_greens, gamma=kgw_params.gamma)
        z_w_vec = (
            pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=kgw_params.gamma)
            if do_wrongkey
            else None
        )
        q_eff = int(len(winners))
        _update_crossed(crossed_at, crossed_at_raw, z_vec=z_vec, T_vec=pooled_T, q_eff=q_eff)
        if do_wrongkey and z_w_vec is not None:
            _update_crossed(
                wrong_crossed_at,
                wrong_crossed_at_raw,
                z_vec=z_w_vec,
                T_vec=wrong_pooled_T,
                q_eff=q_eff,
            )

        snap: Optional[Dict[str, Any]] = None
        if log_traj:
            snap = {
                "q": q_eff,
                "pooled_T": pooled_T.tolist(),
                "z": [float(x) for x in z_vec],
                "assign_rate_sofar": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
                "mix_unique_providers_last": int(mix_counts[-1]) if mix_counts else None,
            }
            if z_w_vec is not None:
                snap["T_wrong"] = wrong_pooled_T.tolist()
                snap["z_wrong"] = [float(x) for x in z_w_vec]
            traj.append(snap)

        append_trial_log(
            trial_log_jsonl,
            {
                "trial_idx": int(trial_idx),
                "wm_scheme": wm_scheme,
                "audit_mode": str(audit_mode),
                "target_provider": int(target_provider),
                "target_doc_id": str(target.get("id")),
                "question": question,
                "question_hash": sha256_text(question),
                "retrieved_docs": retrieved_docs,
                "answer_style": str(answer_style),
                "answer": answer,
                "answer_hash": sha256_text(answer),
                "answer_tokens_est": int(len(answer.split())),
                "winner_pid": int(winner_pid),
                "q_effective": int(q_eff),
                "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                "align": {
                    "mode": str(audit_mode),
                    "source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
                    "assignment_mode": (
                        str(meta.get("assignment_mode"))
                        if meta.get("assignment_mode") is not None
                        else ("competitive" if competitive else "non-competitive")
                    ),
                    "windows_total": int(meta.get("windows_total", 0) or 0),
                    "assigned_total": int(meta.get("assigned_total", 0) or 0),
                    "dropped_total": int(meta.get("dropped_below_thresh_windows", 0) or 0),
                    "best_cos_mean": None,
                    "second_cos_mean": None,
                    "source_selected_docs": (
                        int(source_diag.get("selected_docs", 0)) if audit_mode == "aligned" else None
                    ),
                    "source_selected_providers": (
                        int(source_diag.get("selected_providers", 0)) if audit_mode == "aligned" else None
                    ),
                },
                "aligned_source": (source_diag if audit_mode == "aligned" else None),
                "per_provider": {
                    "T": per_answer_T.tolist(),
                    "greens": per_answer_greens.tolist(),
                    "z": [float(x) for x in per_answer_z.tolist()],
                },
                "wrongkey": (
                    {
                        "enabled": True,
                        "T": wrong_per_answer_T.tolist(),
                        "greens": wrong_per_answer_greens.tolist(),
                        "z": [float(x) for x in wrong_per_answer_z.tolist()],
                    }
                    if do_wrongkey
                    else {"enabled": False}
                ),
                "traj_snapshot": snap,
                "cumulative": {
                    "pooled_T": pooled_T.tolist(),
                    "pooled_greens": pooled_greens.tolist(),
                    "wrong_pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
                    "wrong_pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
                    "z": [float(x) for x in z_vec],
                    "z_wrong": ([float(x) for x in z_w_vec] if z_w_vec is not None else None),
                    "assign_total": int(assign_total),
                    "windows_total": int(windows_total),
                    "dropped_total": int(dropped_total),
                },
            },
        )

    final_z_raw = pooled_z_kgw(pooled_T, pooled_greens, gamma=kgw_params.gamma)
    pvals_raw = [norm_sf(z) for z in final_z_raw]
    final_z, pvals, qvals, bh_sig, bh_eligible = gate_significance_by_min_pooled_T(
        final_z_raw,
        pooled_T,
        min_pooled_T=min_pooled_T,
        alpha=0.05,
    )
    wrong_final_z_raw = (
        pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=kgw_params.gamma)
        if do_wrongkey
        else None
    )
    wrong_final_z = None
    if wrong_final_z_raw is not None:
        wrong_final_z = [
            float(z) if int(pooled_T_i) >= int(min_pooled_T) else 0.0
            for z, pooled_T_i in zip(wrong_final_z_raw, wrong_pooled_T.tolist())
        ]

    out: Dict[str, Any] = {
        "P": int(P),
        "target_provider": int(target_provider),
        "trials": int(trials),
        "topk": int(topk),
        "min_unique_providers": int(min_unique_providers),
        "force_cross_provider": bool(force_cross_provider),
        "exclude_target_provider_from_retrieval": bool(exclude_target_provider_from_retrieval),
        "answer_style": str(answer_style),
        "wm_scheme": str(wm_scheme),
        "audit_mode": str(audit_mode),
        "z_threshold": float(z_threshold),
        "min_pooled_T": int(min_pooled_T),
        "winner_fraction_target": float(np.mean(np.array(winners) == int(target_provider))) if winners else 0.0,
        "mix_unique_providers_mean": float(np.mean(mix_counts)) if mix_counts else 0.0,
        "mix_unique_providers_median": float(np.median(mix_counts)) if mix_counts else 0.0,
        "pooled_T": pooled_T.tolist(),
        "pooled_greens": pooled_greens.tolist(),
        # Compatibility field used by old CSV writer/consumers.
        "pooled_sumproj": [float(x) for x in pooled_greens.tolist()],
        "z_raw": [float(x) for x in final_z_raw],
        "z": [float(x) for x in final_z],
        "p_one_sided_raw": [float(x) for x in pvals_raw],
        "p_one_sided": [float(x) for x in pvals],
        "q_bh": [float(x) for x in qvals],
        "bh_sig": [bool(x) for x in bh_sig],
        "bh_eligible": [bool(x) for x in bh_eligible],
        "crossed_at": crossed_at,
        "crossed_at_raw": crossed_at_raw,
        "align": {
            "mode": str(audit_mode),
            "source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
            "assignment_mode": ("competitive" if competitive else "non-competitive"),
            "windows_total": int(windows_total),
            "assigned_total": int(assign_total),
            "dropped_total": int(dropped_total),
            "assign_rate": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
            "best_cos_mean": None,
            "second_cos_mean": None,
            "source_docs_selected_mean": (
                float(np.mean(source_docs_selected)) if source_docs_selected else None
            ),
            "source_docs_selected_median": (
                float(np.median(source_docs_selected)) if source_docs_selected else None
            ),
            "source_docs_selected_p90": (
                float(np.percentile(source_docs_selected, 90)) if source_docs_selected else None
            ),
            "source_providers_selected_mean": (
                float(np.mean(source_providers_selected)) if source_providers_selected else None
            ),
            "global_above_threshold_docs_mean": (
                float(np.mean(source_above_threshold_docs)) if source_above_threshold_docs else None
            ),
            "global_selected_sim_mean": (
                float(np.mean(source_sim_means)) if source_sim_means else None
            ),
            "global_query_sim_threshold": (
                float(global_source_query_sim_threshold)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_max_docs": (
                int(global_source_max_docs)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_max_docs_per_provider": (
                int(global_source_max_docs_per_provider)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_source_selector": (
                str(global_source_selector)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
        },
        "wrongkey": {
            "enabled": bool(do_wrongkey),
            "z_raw": [float(x) for x in wrong_final_z_raw] if wrong_final_z_raw is not None else None,
            "z": [float(x) for x in wrong_final_z] if wrong_final_z is not None else None,
            "pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
            "pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
            "crossed_at": wrong_crossed_at if do_wrongkey else None,
            "crossed_at_raw": wrong_crossed_at_raw if do_wrongkey else None,
        },
        "traj": traj if log_traj else None,
        "kgw": {
            "tokenizer_model": str(kgw_tokenizer_model),
            "seeding_scheme": str(kgw_params.seeding_scheme),
            "gamma": float(kgw_params.gamma),
            "delta": float(kgw_params.delta),
            "ignore_repeated_ngrams": bool(kgw_params.ignore_repeated_ngrams),
            "vocab_size": (None if kgw_params.vocab_size is None else int(kgw_params.vocab_size)),
            "n_min": int(kgw_n_min),
            "competitive": bool(competitive),
            "audit_mode": str(audit_mode),
            "aligned_source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
        },
    }
    return out


def run_semstamp_audit(
    *,
    work_dir: Path,
    rows: List[Dict[str, Any]],
    keys: List[str],
    source_field: str,
    watermarked_field: str,
    retrieval_model: str,
    target_provider: int,
    trials: int,
    topk: int,
    min_unique_providers: int,
    force_cross_provider: bool,
    exclude_target_provider_from_retrieval: bool,
    answer_style: str,
    openai_model: str,
    answer_temp: float,
    semstamp_clusterer: SemStampClusterer,
    semstamp_params: SemStampParams,
    semstamp_min_sent_words: int,
    sim_threshold: float,
    tie_margin: float,
    competitive: bool,
    audit_mode: str,
    aligned_source_mode: str,
    alignment_manifest: Optional[Path],
    global_source_query_sim_threshold: float,
    global_source_max_docs: int,
    global_source_max_docs_per_provider: int,
    global_source_selector: str,
    do_wrongkey: bool,
    null_unrelated: bool,
    z_threshold: float,
    min_pooled_T: int,
    log_traj: bool,
    seed: int,
    wm_scheme: str,
    trial_log_jsonl: Optional[Path],
    resume_trials: bool,
) -> Dict[str, Any]:
    P = len(keys)
    if P <= 0:
        raise RuntimeError("No provider keys loaded")
    if not (0 <= target_provider < P):
        raise ValueError(f"--target_provider={target_provider} out of range (P={P})")

    X, enc = ensure_embeddings(work_dir, rows, field=source_field, model_name=retrieval_model)
    aligned_source_mode = _canon_aligned_source_mode(aligned_source_mode)
    global_source_selector = _canon_global_source_selector(global_source_selector)
    if aligned_source_mode not in {"oracle", "retrieved_src", "manifest_src", "global_corpus"}:
        raise ValueError(
            f"Unknown --aligned_source_mode={aligned_source_mode}. "
            f"Use one of: oracle, global_corpus, retrieved_src, manifest_src."
        )
    align_manifest_by_id: Dict[str, Dict[str, Any]] = {}
    if audit_mode == "aligned" and aligned_source_mode == "manifest_src":
        if alignment_manifest is None:
            raise ValueError("--alignment_manifest is required when --aligned_source_mode=manifest_src")
        if not alignment_manifest.exists():
            raise FileNotFoundError(f"Alignment manifest not found: {alignment_manifest}")
        align_manifest_by_id = load_alignment_manifest_map(alignment_manifest)

    key_map = {i: keys[i] for i in range(P)}
    wrong_key_map = None
    if do_wrongkey:
        wrong_keys = keys[1:] + keys[:1]
        wrong_key_map = {i: wrong_keys[i] for i in range(P)}

    candidates = [r for r in rows if int(r.get("provider_id", -1)) == int(target_provider)]
    if not candidates:
        raise RuntimeError(f"No candidate docs found for provider {target_provider}")

    pooled_T = np.zeros(P, dtype=np.int64)
    pooled_greens = np.zeros(P, dtype=np.int64)
    wrong_pooled_T = np.zeros(P, dtype=np.int64)
    wrong_pooled_greens = np.zeros(P, dtype=np.int64)
    winners: List[int] = []
    mix_counts: List[int] = []

    crossed_at: List[Optional[int]] = [None] * P
    crossed_at_raw: List[Optional[int]] = [None] * P
    wrong_crossed_at: List[Optional[int]] = [None] * P
    wrong_crossed_at_raw: List[Optional[int]] = [None] * P

    assign_total = 0
    windows_total = 0
    dropped_total = 0
    best_cos_means: List[float] = []
    second_cos_means: List[float] = []
    source_docs_selected: List[int] = []
    source_providers_selected: List[int] = []
    source_above_threshold_docs: List[int] = []
    source_sim_means: List[float] = []
    traj: List[Dict[str, Any]] = []

    def _update_crossed(
        crossed: List[Optional[int]],
        crossed_raw: List[Optional[int]],
        z_vec: List[float],
        T_vec: np.ndarray,
        q_eff: int,
    ) -> None:
        for pid in range(P):
            T = int(T_vec[pid])
            z_pool = float(z_vec[pid])
            if crossed_raw[pid] is None and z_pool >= z_threshold:
                crossed_raw[pid] = int(q_eff)
            if crossed[pid] is None and T >= int(min_pooled_T) and z_pool >= z_threshold:
                crossed[pid] = int(q_eff)

    existing_trials: List[Dict[str, Any]] = (
        read_trial_log(trial_log_jsonl, scheme=wm_scheme, audit_mode=audit_mode)
        if resume_trials
        else []
    )
    if existing_trials:
        last = existing_trials[-1]
        cum = last.get("cumulative", {}) or {}
        try:
            pt = cum.get("pooled_T")
            pg = cum.get("pooled_greens")
            if isinstance(pt, list) and isinstance(pg, list) and len(pt) == P and len(pg) == P:
                pooled_T = np.array(pt, dtype=np.int64)
                pooled_greens = np.array(pg, dtype=np.int64)
                if do_wrongkey:
                    wpt = cum.get("wrong_pooled_T")
                    wpg = cum.get("wrong_pooled_greens")
                    if isinstance(wpt, list) and isinstance(wpg, list) and len(wpt) == P and len(wpg) == P:
                        wrong_pooled_T = np.array(wpt, dtype=np.int64)
                        wrong_pooled_greens = np.array(wpg, dtype=np.int64)
                assign_total = int(cum.get("assign_total", 0) or 0)
                windows_total = int(cum.get("windows_total", 0) or 0)
                dropped_total = int(cum.get("dropped_total", 0) or 0)
        except Exception:
            existing_trials = []

        for tr in existing_trials:
            w = tr.get("winner_pid")
            if isinstance(w, int) and w >= 0:
                winners.append(int(w))
            m = tr.get("mix_unique_providers")
            if isinstance(m, int):
                mix_counts.append(int(m))
            al = tr.get("align", {}) or {}
            if al.get("best_cos_mean") is not None:
                best_cos_means.append(float(al["best_cos_mean"]))
            if al.get("second_cos_mean") is not None:
                second_cos_means.append(float(al["second_cos_mean"]))
            srcdiag = tr.get("aligned_source") or {}
            if isinstance(srcdiag, dict):
                sd = srcdiag.get("selected_docs")
                sp = srcdiag.get("selected_providers")
                sat = srcdiag.get("above_threshold_docs")
                sm = srcdiag.get("selected_sim_mean")
                if isinstance(sd, int):
                    source_docs_selected.append(int(sd))
                if isinstance(sp, int):
                    source_providers_selected.append(int(sp))
                if isinstance(sat, int):
                    source_above_threshold_docs.append(int(sat))
                if isinstance(sm, (int, float)):
                    source_sim_means.append(float(sm))
            if log_traj:
                snap = tr.get("traj_snapshot")
                if isinstance(snap, dict):
                    traj.append(snap)

        for tr in existing_trials:
            cum = tr.get("cumulative", {}) or {}
            z_vec = cum.get("z")
            t_vec = cum.get("pooled_T")
            q_eff = tr.get("q_effective")
            if isinstance(z_vec, list) and isinstance(t_vec, list) and len(z_vec) == P and len(t_vec) == P:
                _update_crossed(
                    crossed=crossed_at,
                    crossed_raw=crossed_at_raw,
                    z_vec=[float(z) for z in z_vec],
                    T_vec=np.array(t_vec, dtype=np.int64),
                    q_eff=int(q_eff) if isinstance(q_eff, int) else len(winners),
                )
            if do_wrongkey:
                z_w = cum.get("z_wrong")
                t_w = cum.get("wrong_pooled_T")
                if isinstance(z_w, list) and isinstance(t_w, list) and len(z_w) == P and len(t_w) == P:
                    _update_crossed(
                        crossed=wrong_crossed_at,
                        crossed_raw=wrong_crossed_at_raw,
                        z_vec=[float(z) for z in z_w],
                        T_vec=np.array(t_w, dtype=np.int64),
                        q_eff=int(q_eff) if isinstance(q_eff, int) else len(winners),
                    )

    start_trial = 1
    if existing_trials:
        start_trial = int(existing_trials[-1]["trial_idx"]) + 1

    if audit_mode == "answer_only" and competitive:
        print(
            "[INFO] SemStamp answer_only scores full answers under every provider key; "
            "--competitive only affects aligned mode.",
            flush=True,
        )
    if exclude_target_provider_from_retrieval:
        print(
            "[INFO] Retrieval excludes target-provider docs (target-absent stress mode).",
            flush=True,
        )
    if audit_mode == "aligned":
        extra = ""
        if aligned_source_mode == "manifest_src" and alignment_manifest is not None:
            extra = f", manifest={alignment_manifest}"
        if aligned_source_mode == "global_corpus":
            extra = (
                f", selector={global_source_selector}, "
                f"query_sim_threshold={float(global_source_query_sim_threshold):.3f}, "
                f"max_docs={int(global_source_max_docs)}, "
                f"max_docs_per_provider={int(global_source_max_docs_per_provider)}"
            )
        print(
            "[INFO] SemStamp aligned mode uses semantic sentence alignment against source docs. "
            f"(source_mode={aligned_source_mode}{extra})",
            flush=True,
        )

    pbar = tqdm(range(start_trial, trials + 1), desc=f"audit:{answer_style}")
    for trial_idx in pbar:
        trial_rng = random.Random((int(seed) * 1315423911 + int(trial_idx) * 2654435761) & 0xFFFFFFFF)
        target = trial_rng.choice(candidates)

        q_doc_text = (target.get(source_field) or target.get(watermarked_field) or "")
        if not q_doc_text:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "status": "skipped_no_query_doc",
                    "winner_pid": None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_greens": pooled_greens.tolist(),
                        "wrong_pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
                        "wrong_pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
                        "z": pooled_z_kgw(pooled_T, pooled_greens, gamma=semstamp_params.gamma),
                        "z_wrong": (
                            pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=semstamp_params.gamma)
                            if do_wrongkey
                            else None
                        ),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        question = gen_question_openai(q_doc_text, model=openai_model)
        query_vec = enc.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]

        picked_idxs = select_retrieval_bundle_idxs(
            rows=rows,
            X=X,
            enc=enc,
            question=question,
            query_vec=query_vec,
            target_doc_id=str(target["id"]),
            target_provider=int(target_provider),
            k=int(topk),
            min_unique_providers=int(min_unique_providers),
            force_cross_provider=bool(force_cross_provider),
            null_unrelated=bool(null_unrelated),
            exclude_target_provider_from_retrieval=bool(exclude_target_provider_from_retrieval),
            rng=trial_rng,
        )

        providers_in_bundle = [int(rows[i]["provider_id"]) for i in picked_idxs]
        mix_counts.append(len(set(providers_in_bundle)))

        prompt_docs: List[Tuple[int, str]] = []
        retrieved_docs: List[Dict[str, Any]] = []
        for i in picked_idxs:
            pid = int(rows[i]["provider_id"])
            txt = (rows[i].get(watermarked_field) or "")
            if txt:
                prompt_docs.append((pid, txt))
            retrieved_docs.append({"doc_id": str(rows[i].get("id")), "provider_id": pid})
        if not prompt_docs:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "target_doc_id": str(target.get("id")),
                    "status": "skipped_no_prompt_docs",
                    "retrieved_docs": retrieved_docs,
                    "winner_pid": None,
                    "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_greens": pooled_greens.tolist(),
                        "wrong_pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
                        "wrong_pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
                        "z": pooled_z_kgw(pooled_T, pooled_greens, gamma=semstamp_params.gamma),
                        "z_wrong": (
                            pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=semstamp_params.gamma)
                            if do_wrongkey
                            else None
                        ),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        messages = make_rag_messages(prompt_docs, question, style=answer_style)
        answer = openai_chat(openai_model, messages, temperature=answer_temp).strip()

        sources: List[Tuple[int, str]] = []
        source_diag: Dict[str, Any] = {}
        answer_vec: Optional[np.ndarray] = None
        if (
            audit_mode == "aligned"
            and aligned_source_mode == "global_corpus"
            and global_source_selector in {"answer", "max"}
        ):
            answer_vec = enc.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0]
        if audit_mode == "aligned":
            sources, source_diag = build_alignment_sources(
                rows=rows,
                picked_idxs=picked_idxs,
                prompt_docs=prompt_docs,
                retrieved_docs=retrieved_docs,
                source_field=source_field,
                aligned_source_mode=aligned_source_mode,
                align_manifest_by_id=align_manifest_by_id if aligned_source_mode == "manifest_src" else None,
                query_vec=query_vec,
                answer_vec=answer_vec,
                X=X,
                global_source_query_sim_threshold=float(global_source_query_sim_threshold),
                global_source_max_docs=int(global_source_max_docs),
                global_source_max_docs_per_provider=int(global_source_max_docs_per_provider),
                global_source_selector=global_source_selector,
            )

        if audit_mode == "answer_only":
            res_by_pid = {
                int(pid): detect_semstamp_text(
                    text=answer,
                    key_hex=key_map[int(pid)],
                    clusterer=semstamp_clusterer,
                    params=semstamp_params,
                    min_sent_words=int(semstamp_min_sent_words),
                )
                for pid in range(P)
            }
            trial_windows = 0
            if res_by_pid:
                trial_windows = int(next(iter(res_by_pid.values())).T)
            meta = {
                "mode": "answer_only",
                "assignment_mode": "all_providers",
                "windows_total": int(trial_windows),
                "assigned_total": int(trial_windows),
                "dropped_below_thresh_windows": 0,
                "best_cos_stats": {"mean": None},
                "second_best_cos_stats": {"mean": None},
            }
        else:
            res_by_pid, meta = detect_semstamp_multi_source_aligned(
                answer_text=answer,
                sources=sources,
                provider_keys_by_pid=key_map,
                clusterer=semstamp_clusterer,
                params=semstamp_params,
                sim_threshold=float(sim_threshold),
                tie_margin=float(tie_margin),
                competitive=bool(competitive),
                min_sent_words=int(semstamp_min_sent_words),
            )

        assign_total += int(meta.get("assigned_total", 0) or 0)
        windows_total += int(meta.get("windows_total", 0) or 0)
        dropped_total += int(meta.get("dropped_below_thresh_windows", 0) or 0)
        if audit_mode == "aligned":
            bcm = (meta.get("best_cos_stats", {}) or {}).get("mean")
            scm = (meta.get("second_best_cos_stats", {}) or {}).get("mean")
            if bcm is not None:
                best_cos_means.append(float(bcm))
            if scm is not None:
                second_cos_means.append(float(scm))
            sd = source_diag.get("selected_docs")
            sp = source_diag.get("selected_providers")
            sat = source_diag.get("above_threshold_docs")
            sm = source_diag.get("selected_sim_mean")
            if isinstance(sd, int):
                source_docs_selected.append(int(sd))
            if isinstance(sp, int):
                source_providers_selected.append(int(sp))
            if isinstance(sat, int):
                source_above_threshold_docs.append(int(sat))
            if isinstance(sm, (int, float)):
                source_sim_means.append(float(sm))

        per_answer_z = np.zeros(P, dtype=float)
        per_answer_T = np.zeros(P, dtype=np.int64)
        per_answer_greens = np.zeros(P, dtype=np.int64)
        for pid in range(P):
            r = res_by_pid.get(pid)
            if r is None:
                continue
            T_pid = int(r.T)
            G_pid = int(r.greens)
            pooled_T[pid] += T_pid
            pooled_greens[pid] += G_pid
            per_answer_T[pid] = T_pid
            per_answer_greens[pid] = G_pid
            per_answer_z[pid] = float(r.z)
        winner_pid = int(np.argmax(per_answer_z))
        winners.append(winner_pid)

        wrong_per_answer_z = np.zeros(P, dtype=float)
        wrong_per_answer_T = np.zeros(P, dtype=np.int64)
        wrong_per_answer_greens = np.zeros(P, dtype=np.int64)
        if do_wrongkey and wrong_key_map is not None:
            if audit_mode == "answer_only":
                wrong_res = {
                    int(pid): detect_semstamp_text(
                        text=answer,
                        key_hex=wrong_key_map[int(pid)],
                        clusterer=semstamp_clusterer,
                        params=semstamp_params,
                        min_sent_words=int(semstamp_min_sent_words),
                    )
                    for pid in range(P)
                }
            else:
                wrong_res, _ = detect_semstamp_multi_source_aligned(
                    answer_text=answer,
                    sources=sources,
                    provider_keys_by_pid=wrong_key_map,
                    clusterer=semstamp_clusterer,
                    params=semstamp_params,
                    sim_threshold=float(sim_threshold),
                    tie_margin=float(tie_margin),
                    competitive=bool(competitive),
                    min_sent_words=int(semstamp_min_sent_words),
                )
            for pid in range(P):
                r = wrong_res.get(pid)
                if r is None:
                    continue
                T_pid = int(r.T)
                G_pid = int(r.greens)
                wrong_pooled_T[pid] += T_pid
                wrong_pooled_greens[pid] += G_pid
                wrong_per_answer_T[pid] = T_pid
                wrong_per_answer_greens[pid] = G_pid
                wrong_per_answer_z[pid] = float(r.z)

        z_vec = pooled_z_kgw(pooled_T, pooled_greens, gamma=semstamp_params.gamma)
        z_w_vec = (
            pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=semstamp_params.gamma)
            if do_wrongkey
            else None
        )
        q_eff = int(len(winners))
        _update_crossed(crossed_at, crossed_at_raw, z_vec=z_vec, T_vec=pooled_T, q_eff=q_eff)
        if do_wrongkey and z_w_vec is not None:
            _update_crossed(
                wrong_crossed_at, wrong_crossed_at_raw, z_vec=z_w_vec, T_vec=wrong_pooled_T, q_eff=q_eff
            )

        snap: Optional[Dict[str, Any]] = None
        if log_traj:
            snap = {
                "q": q_eff,
                "pooled_T": pooled_T.tolist(),
                "z": [float(x) for x in z_vec],
                "assign_rate_sofar": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
                "mix_unique_providers_last": int(mix_counts[-1]) if mix_counts else None,
            }
            if z_w_vec is not None:
                snap["T_wrong"] = wrong_pooled_T.tolist()
                snap["z_wrong"] = [float(x) for x in z_w_vec]
            traj.append(snap)

        append_trial_log(
            trial_log_jsonl,
            {
                "trial_idx": int(trial_idx),
                "wm_scheme": wm_scheme,
                "audit_mode": str(audit_mode),
                "target_provider": int(target_provider),
                "target_doc_id": str(target.get("id")),
                "question": question,
                "question_hash": sha256_text(question),
                "retrieved_docs": retrieved_docs,
                "answer_style": str(answer_style),
                "answer": answer,
                "answer_hash": sha256_text(answer),
                "answer_tokens_est": int(len(answer.split())),
                "winner_pid": int(winner_pid),
                "q_effective": int(q_eff),
                "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                "align": {
                    "mode": str(audit_mode),
                    "source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
                    "assignment_mode": (
                        str(meta.get("assignment_mode"))
                        if meta.get("assignment_mode") is not None
                        else ("competitive" if competitive else "non-competitive")
                    ),
                    "windows_total": int(meta.get("windows_total", 0) or 0),
                    "assigned_total": int(meta.get("assigned_total", 0) or 0),
                    "dropped_total": int(meta.get("dropped_below_thresh_windows", 0) or 0),
                    "best_cos_mean": (
                        None if (meta.get("best_cos_stats", {}) or {}).get("mean") is None else
                        float((meta.get("best_cos_stats", {}) or {}).get("mean"))
                    ),
                    "second_cos_mean": (
                        None if (meta.get("second_best_cos_stats", {}) or {}).get("mean") is None else
                        float((meta.get("second_best_cos_stats", {}) or {}).get("mean"))
                    ),
                    "source_selected_docs": (
                        int(source_diag.get("selected_docs", 0)) if audit_mode == "aligned" else None
                    ),
                    "source_selected_providers": (
                        int(source_diag.get("selected_providers", 0)) if audit_mode == "aligned" else None
                    ),
                },
                "aligned_source": (source_diag if audit_mode == "aligned" else None),
                "per_provider": {
                    "T": per_answer_T.tolist(),
                    "greens": per_answer_greens.tolist(),
                    "z": [float(x) for x in per_answer_z.tolist()],
                },
                "wrongkey": (
                    {
                        "enabled": True,
                        "T": wrong_per_answer_T.tolist(),
                        "greens": wrong_per_answer_greens.tolist(),
                        "z": [float(x) for x in wrong_per_answer_z.tolist()],
                    }
                    if do_wrongkey
                    else {"enabled": False}
                ),
                "traj_snapshot": snap,
                "cumulative": {
                    "pooled_T": pooled_T.tolist(),
                    "pooled_greens": pooled_greens.tolist(),
                    "wrong_pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
                    "wrong_pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
                    "z": [float(x) for x in z_vec],
                    "z_wrong": ([float(x) for x in z_w_vec] if z_w_vec is not None else None),
                    "assign_total": int(assign_total),
                    "windows_total": int(windows_total),
                    "dropped_total": int(dropped_total),
                },
            },
        )

    final_z_raw = pooled_z_kgw(pooled_T, pooled_greens, gamma=semstamp_params.gamma)
    pvals_raw = [norm_sf(z) for z in final_z_raw]
    final_z, pvals, qvals, bh_sig, bh_eligible = gate_significance_by_min_pooled_T(
        final_z_raw, pooled_T, min_pooled_T=min_pooled_T, alpha=0.05
    )
    wrong_final_z_raw = (
        pooled_z_kgw(wrong_pooled_T, wrong_pooled_greens, gamma=semstamp_params.gamma) if do_wrongkey else None
    )
    wrong_final_z = None
    if wrong_final_z_raw is not None:
        wrong_final_z = [
            float(z) if int(pooled_T_i) >= int(min_pooled_T) else 0.0
            for z, pooled_T_i in zip(wrong_final_z_raw, wrong_pooled_T.tolist())
        ]

    out: Dict[str, Any] = {
        "P": int(P),
        "target_provider": int(target_provider),
        "trials": int(trials),
        "topk": int(topk),
        "min_unique_providers": int(min_unique_providers),
        "force_cross_provider": bool(force_cross_provider),
        "exclude_target_provider_from_retrieval": bool(exclude_target_provider_from_retrieval),
        "answer_style": str(answer_style),
        "wm_scheme": str(wm_scheme),
        "audit_mode": str(audit_mode),
        "z_threshold": float(z_threshold),
        "min_pooled_T": int(min_pooled_T),
        "winner_fraction_target": float(np.mean(np.array(winners) == int(target_provider))) if winners else 0.0,
        "mix_unique_providers_mean": float(np.mean(mix_counts)) if mix_counts else 0.0,
        "mix_unique_providers_median": float(np.median(mix_counts)) if mix_counts else 0.0,
        "pooled_T": pooled_T.tolist(),
        "pooled_greens": pooled_greens.tolist(),
        "pooled_sumproj": [float(x) for x in pooled_greens.tolist()],
        "z_raw": [float(x) for x in final_z_raw],
        "z": [float(x) for x in final_z],
        "p_one_sided_raw": [float(x) for x in pvals_raw],
        "p_one_sided": [float(x) for x in pvals],
        "q_bh": [float(x) for x in qvals],
        "bh_sig": [bool(x) for x in bh_sig],
        "bh_eligible": [bool(x) for x in bh_eligible],
        "crossed_at": crossed_at,
        "crossed_at_raw": crossed_at_raw,
        "align": {
            "mode": str(audit_mode),
            "source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
            "assignment_mode": ("competitive" if competitive else "non-competitive"),
            "windows_total": int(windows_total),
            "assigned_total": int(assign_total),
            "dropped_total": int(dropped_total),
            "assign_rate": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
            "best_cos_mean": (float(np.mean(best_cos_means)) if best_cos_means else None),
            "second_cos_mean": (float(np.mean(second_cos_means)) if second_cos_means else None),
            "source_docs_selected_mean": (float(np.mean(source_docs_selected)) if source_docs_selected else None),
            "source_docs_selected_median": (
                float(np.median(source_docs_selected)) if source_docs_selected else None
            ),
            "source_docs_selected_p90": (
                float(np.percentile(source_docs_selected, 90)) if source_docs_selected else None
            ),
            "source_providers_selected_mean": (
                float(np.mean(source_providers_selected)) if source_providers_selected else None
            ),
            "global_above_threshold_docs_mean": (
                float(np.mean(source_above_threshold_docs)) if source_above_threshold_docs else None
            ),
            "global_selected_sim_mean": (float(np.mean(source_sim_means)) if source_sim_means else None),
            "global_query_sim_threshold": (
                float(global_source_query_sim_threshold)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_max_docs": (
                int(global_source_max_docs)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_max_docs_per_provider": (
                int(global_source_max_docs_per_provider)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
            "global_source_selector": (
                str(global_source_selector)
                if (audit_mode == "aligned" and aligned_source_mode == "global_corpus")
                else None
            ),
        },
        "wrongkey": {
            "enabled": bool(do_wrongkey),
            "z_raw": [float(x) for x in wrong_final_z_raw] if wrong_final_z_raw is not None else None,
            "z": [float(x) for x in wrong_final_z] if wrong_final_z is not None else None,
            "pooled_T": wrong_pooled_T.tolist() if do_wrongkey else None,
            "pooled_greens": wrong_pooled_greens.tolist() if do_wrongkey else None,
            "crossed_at": wrong_crossed_at if do_wrongkey else None,
            "crossed_at_raw": wrong_crossed_at_raw if do_wrongkey else None,
        },
        "traj": traj if log_traj else None,
        "semstamp": {
            "embed_model": str(semstamp_params.embed_model),
            "num_clusters": int(semstamp_params.num_clusters),
            "gamma": float(semstamp_params.gamma),
            "min_sent_words": int(semstamp_min_sent_words),
            "competitive": bool(competitive),
            "audit_mode": str(audit_mode),
            "aligned_source_mode": (str(aligned_source_mode) if audit_mode == "aligned" else None),
            "align_sim_threshold": float(sim_threshold),
            "align_tie_margin": float(tie_margin),
        },
    }
    return out


def run_nonwm_baseline_audit(
    *,
    baseline_kind: str,  # sim_attr | overlap_attr | entail_attr
    work_dir: Path,
    rows: List[Dict[str, Any]],
    keys: List[str],
    source_field: str,
    watermarked_field: str,
    retrieval_model: str,
    target_provider: int,
    trials: int,
    topk: int,
    min_unique_providers: int,
    force_cross_provider: bool,
    exclude_target_provider_from_retrieval: bool,
    null_unrelated: bool,
    answer_style: str,
    openai_model: str,
    answer_temp: float,
    audit_mode: str,
    aligned_source_mode: str,
    alignment_manifest: Optional[Path],
    global_source_query_sim_threshold: float,
    global_source_max_docs: int,
    global_source_max_docs_per_provider: int,
    global_source_selector: str,
    z_threshold: float,
    min_pooled_T: int,
    log_traj: bool,
    seed: int,
    wm_scheme: str,
    trial_log_jsonl: Optional[Path],
    resume_trials: bool,
    sim_params: SimilarityBaselineParams,
    overlap_params: OverlapBaselineParams,
    overlap_scorer: Optional[LexicalOverlapScorer],
    entail_params: EntailmentBaselineParams,
    entail_scorer: Optional[EntailmentScorer],
) -> Dict[str, Any]:
    if audit_mode != "aligned":
        raise ValueError(
            f"{wm_scheme} only supports --audit_mode=aligned (it is a source-aware non-watermark baseline)."
        )
    if baseline_kind == "overlap_attr" and overlap_scorer is None:
        raise ValueError("Overlap baseline requires overlap_scorer.")
    if baseline_kind == "entail_attr" and entail_scorer is None:
        raise ValueError("Entailment baseline requires entail_scorer.")

    P = len(keys)
    if P <= 0:
        raise RuntimeError("No provider keys loaded")
    if not (0 <= target_provider < P):
        raise ValueError(f"--target_provider={target_provider} out of range (P={P})")

    X, enc = ensure_embeddings(work_dir, rows, field=source_field, model_name=retrieval_model)
    aligned_source_mode = _canon_aligned_source_mode(aligned_source_mode)
    global_source_selector = _canon_global_source_selector(global_source_selector)
    if aligned_source_mode not in {"oracle", "retrieved_src", "manifest_src", "global_corpus"}:
        raise ValueError(
            f"Unknown --aligned_source_mode={aligned_source_mode}. "
            f"Use one of: oracle, global_corpus, retrieved_src, manifest_src."
        )
    align_manifest_by_id: Dict[str, Dict[str, Any]] = {}
    if aligned_source_mode == "manifest_src":
        if alignment_manifest is None:
            raise ValueError("--alignment_manifest is required when --aligned_source_mode=manifest_src")
        if not alignment_manifest.exists():
            raise FileNotFoundError(f"Alignment manifest not found: {alignment_manifest}")
        align_manifest_by_id = load_alignment_manifest_map(alignment_manifest)

    candidates = [r for r in rows if int(r.get("provider_id", -1)) == int(target_provider)]
    if not candidates:
        raise RuntimeError(f"No candidate docs found for provider {target_provider}")

    pooled_T = np.zeros(P, dtype=np.int64)
    pooled_sum = np.zeros(P, dtype=np.float64)
    pooled_sumsq = np.zeros(P, dtype=np.float64)
    winners: List[int] = []
    mix_counts: List[int] = []

    crossed_at: List[Optional[int]] = [None] * P
    crossed_at_raw: List[Optional[int]] = [None] * P

    assign_total = 0
    windows_total = 0
    dropped_total = 0
    best_cos_means: List[float] = []
    second_cos_means: List[float] = []
    source_docs_selected: List[int] = []
    source_providers_selected: List[int] = []
    source_above_threshold_docs: List[int] = []
    source_sim_means: List[float] = []
    traj: List[Dict[str, Any]] = []

    def _update_crossed(
        crossed: List[Optional[int]],
        crossed_raw: List[Optional[int]],
        z_vec: List[float],
        T_vec: np.ndarray,
        q_eff: int,
    ) -> None:
        for pid in range(P):
            T = int(T_vec[pid])
            z_pool = float(z_vec[pid])
            if crossed_raw[pid] is None and z_pool >= z_threshold:
                crossed_raw[pid] = int(q_eff)
            if crossed[pid] is None and T >= int(min_pooled_T) and z_pool >= z_threshold:
                crossed[pid] = int(q_eff)

    existing_trials: List[Dict[str, Any]] = (
        read_trial_log(trial_log_jsonl, scheme=wm_scheme, audit_mode=audit_mode)
        if resume_trials
        else []
    )
    if existing_trials:
        last = existing_trials[-1]
        cum = last.get("cumulative", {}) or {}
        try:
            pt = cum.get("pooled_T")
            ps = cum.get("pooled_sum")
            pss = cum.get("pooled_sumsq")
            if (
                isinstance(pt, list)
                and isinstance(ps, list)
                and isinstance(pss, list)
                and len(pt) == P
                and len(ps) == P
                and len(pss) == P
            ):
                pooled_T = np.array(pt, dtype=np.int64)
                pooled_sum = np.array(ps, dtype=np.float64)
                pooled_sumsq = np.array(pss, dtype=np.float64)
                assign_total = int(cum.get("assign_total", 0) or 0)
                windows_total = int(cum.get("windows_total", 0) or 0)
                dropped_total = int(cum.get("dropped_total", 0) or 0)
        except Exception:
            existing_trials = []

        for tr in existing_trials:
            w = tr.get("winner_pid")
            if isinstance(w, int) and w >= 0:
                winners.append(int(w))
            m = tr.get("mix_unique_providers")
            if isinstance(m, int):
                mix_counts.append(int(m))
            al = tr.get("align", {}) or {}
            if al.get("best_cos_mean") is not None:
                best_cos_means.append(float(al["best_cos_mean"]))
            if al.get("second_cos_mean") is not None:
                second_cos_means.append(float(al["second_cos_mean"]))
            srcdiag = tr.get("aligned_source") or {}
            if isinstance(srcdiag, dict):
                sd = srcdiag.get("selected_docs")
                sp = srcdiag.get("selected_providers")
                sat = srcdiag.get("above_threshold_docs")
                sm = srcdiag.get("selected_sim_mean")
                if isinstance(sd, int):
                    source_docs_selected.append(int(sd))
                if isinstance(sp, int):
                    source_providers_selected.append(int(sp))
                if isinstance(sat, int):
                    source_above_threshold_docs.append(int(sat))
                if isinstance(sm, (int, float)):
                    source_sim_means.append(float(sm))
            if log_traj:
                snap = tr.get("traj_snapshot")
                if isinstance(snap, dict):
                    traj.append(snap)

        for tr in existing_trials:
            cum = tr.get("cumulative", {}) or {}
            z_vec = cum.get("z")
            t_vec = cum.get("pooled_T")
            q_eff = tr.get("q_effective")
            if isinstance(z_vec, list) and isinstance(t_vec, list) and len(z_vec) == P and len(t_vec) == P:
                _update_crossed(
                    crossed=crossed_at,
                    crossed_raw=crossed_at_raw,
                    z_vec=[float(z) for z in z_vec],
                    T_vec=np.array(t_vec, dtype=np.int64),
                    q_eff=int(q_eff) if isinstance(q_eff, int) else len(winners),
                )

    start_trial = 1
    if existing_trials:
        start_trial = int(existing_trials[-1]["trial_idx"]) + 1

    extra = ""
    if aligned_source_mode == "manifest_src" and alignment_manifest is not None:
        extra = f", manifest={alignment_manifest}"
    if aligned_source_mode == "global_corpus":
        extra = (
            f", selector={global_source_selector}, "
            f"query_sim_threshold={float(global_source_query_sim_threshold):.3f}, "
            f"max_docs={int(global_source_max_docs)}, "
            f"max_docs_per_provider={int(global_source_max_docs_per_provider)}"
        )
    print(
        f"[INFO] {wm_scheme} aligned non-watermark attribution baseline "
        f"(source_mode={aligned_source_mode}{extra})",
        flush=True,
    )
    if exclude_target_provider_from_retrieval:
        print(
            "[INFO] Retrieval excludes target-provider docs (target-absent stress mode).",
            flush=True,
        )

    def _pad_vec(vec: np.ndarray) -> np.ndarray:
        out = np.zeros(P, dtype=np.float32)
        n = min(P, int(vec.shape[0]))
        if n > 0:
            out[:n] = np.asarray(vec[:n], dtype=np.float32)
        return out

    pbar = tqdm(range(start_trial, trials + 1), desc=f"audit:{answer_style}")
    for trial_idx in pbar:
        trial_rng = random.Random((int(seed) * 1315423911 + int(trial_idx) * 2654435761) & 0xFFFFFFFF)
        target = trial_rng.choice(candidates)

        q_doc_text = (target.get(source_field) or target.get(watermarked_field) or "")
        if not q_doc_text:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "status": "skipped_no_query_doc",
                    "winner_pid": None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_sum": [float(x) for x in pooled_sum.tolist()],
                        "pooled_sumsq": [float(x) for x in pooled_sumsq.tolist()],
                        "z": pooled_z_centered_empirical(pooled_T, pooled_sum, pooled_sumsq),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        question = gen_question_openai(q_doc_text, model=openai_model)
        query_vec = enc.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]

        picked_idxs = select_retrieval_bundle_idxs(
            rows=rows,
            X=X,
            enc=enc,
            question=question,
            query_vec=query_vec,
            target_doc_id=str(target["id"]),
            target_provider=int(target_provider),
            k=int(topk),
            min_unique_providers=int(min_unique_providers),
            force_cross_provider=bool(force_cross_provider),
            null_unrelated=bool(null_unrelated),
            exclude_target_provider_from_retrieval=bool(exclude_target_provider_from_retrieval),
            rng=trial_rng,
        )

        providers_in_bundle = [int(rows[i]["provider_id"]) for i in picked_idxs]
        mix_counts.append(len(set(providers_in_bundle)))

        prompt_docs: List[Tuple[int, str]] = []
        retrieved_docs: List[Dict[str, Any]] = []
        for i in picked_idxs:
            pid = int(rows[i]["provider_id"])
            txt = str(rows[i].get(watermarked_field) or "")
            if txt:
                prompt_docs.append((pid, txt))
            retrieved_docs.append({"doc_id": str(rows[i].get("id")), "provider_id": pid})
        if not prompt_docs:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "target_doc_id": str(target.get("id")),
                    "status": "skipped_no_prompt_docs",
                    "retrieved_docs": retrieved_docs,
                    "winner_pid": None,
                    "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                    "q_effective": int(len(winners)),
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_sum": [float(x) for x in pooled_sum.tolist()],
                        "pooled_sumsq": [float(x) for x in pooled_sumsq.tolist()],
                        "z": pooled_z_centered_empirical(pooled_T, pooled_sum, pooled_sumsq),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        messages = make_rag_messages(prompt_docs, question, style=answer_style)
        answer = openai_chat(openai_model, messages, temperature=answer_temp).strip()
        answer_vec: Optional[np.ndarray] = None
        if (
            aligned_source_mode == "global_corpus"
            and global_source_selector in {"answer", "max"}
        ):
            answer_vec = enc.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0]

        sources, source_diag = build_alignment_sources(
            rows=rows,
            picked_idxs=picked_idxs,
            prompt_docs=prompt_docs,
            retrieved_docs=retrieved_docs,
            source_field=source_field,
            aligned_source_mode=aligned_source_mode,
            align_manifest_by_id=align_manifest_by_id if aligned_source_mode == "manifest_src" else None,
            query_vec=query_vec,
            answer_vec=answer_vec,
            X=X,
            global_source_query_sim_threshold=float(global_source_query_sim_threshold),
            global_source_max_docs=int(global_source_max_docs),
            global_source_max_docs_per_provider=int(global_source_max_docs_per_provider),
            global_source_selector=global_source_selector,
        )

        if not sources:
            append_trial_log(
                trial_log_jsonl,
                {
                    "trial_idx": int(trial_idx),
                    "wm_scheme": wm_scheme,
                    "audit_mode": str(audit_mode),
                    "target_provider": int(target_provider),
                    "target_doc_id": str(target.get("id")),
                    "question": question,
                    "question_hash": sha256_text(question),
                    "retrieved_docs": retrieved_docs,
                    "answer_style": str(answer_style),
                    "answer": answer,
                    "answer_hash": sha256_text(answer),
                    "answer_tokens_est": int(len(answer.split())),
                    "status": "skipped_no_alignment_sources",
                    "winner_pid": None,
                    "q_effective": int(len(winners)),
                    "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                    "aligned_source": source_diag,
                    "cumulative": {
                        "pooled_T": pooled_T.tolist(),
                        "pooled_sum": [float(x) for x in pooled_sum.tolist()],
                        "pooled_sumsq": [float(x) for x in pooled_sumsq.tolist()],
                        "z": pooled_z_centered_empirical(pooled_T, pooled_sum, pooled_sumsq),
                        "assign_total": int(assign_total),
                        "windows_total": int(windows_total),
                        "dropped_total": int(dropped_total),
                    },
                },
            )
            continue

        raw_scores = np.zeros(P, dtype=np.float32)
        unit_mat = np.zeros((0, P), dtype=np.float32)
        score_meta: Dict[str, Any] = {"units_total": 0, "unit_kind": "trial", "per_provider_passages": {}}
        if baseline_kind == "sim_attr":
            vec, score_meta = score_similarity_by_provider(
                answer_text=answer,
                sources=sources,
                encoder=enc,
                params=sim_params,
            )
            raw_scores = _pad_vec(vec)
            unit_mat = raw_scores.reshape(1, P)
        elif baseline_kind == "overlap_attr":
            assert overlap_scorer is not None
            vec, score_meta = score_overlap_by_provider(
                answer_text=answer,
                sources=sources,
                scorer=overlap_scorer,
                params=overlap_params,
            )
            raw_scores = _pad_vec(vec)
            unit_mat = raw_scores.reshape(1, P)
        elif baseline_kind == "entail_attr":
            assert entail_scorer is not None
            vec, unit_u, score_meta = score_entailment_by_provider(
                answer_text=answer,
                sources=sources,
                encoder=enc,
                nli=entail_scorer,
                params=entail_params,
            )
            raw_scores = _pad_vec(vec)
            if unit_u.size > 0:
                if unit_u.shape[1] < P:
                    tmp = np.zeros((unit_u.shape[0], P), dtype=np.float32)
                    tmp[:, : unit_u.shape[1]] = unit_u
                    unit_mat = tmp
                else:
                    unit_mat = unit_u[:, :P].astype(np.float32)
            else:
                unit_mat = np.zeros((0, P), dtype=np.float32)
        else:
            raise ValueError(f"Unknown baseline_kind={baseline_kind}")

        psg_counts = {
            int(pid_str): int(n)
            for pid_str, n in ((score_meta.get("per_provider_passages") or {}).items())
            if str(pid_str).isdigit()
        }
        active_pids = [pid for pid, n in sorted(psg_counts.items()) if 0 <= int(pid) < P and int(n) > 0]

        units_total_trial = int(score_meta.get("units_total", unit_mat.shape[0]) or 0)
        if units_total_trial <= 0 and unit_mat.shape[0] > 0:
            units_total_trial = int(unit_mat.shape[0])
        windows_total += int(units_total_trial)

        units_used_trial = 0
        per_answer_T = np.zeros(P, dtype=np.int64)
        per_answer_sum = np.zeros(P, dtype=np.float64)
        per_answer_sumsq = np.zeros(P, dtype=np.float64)
        if unit_mat.shape[0] > 0 and active_pids:
            active_pids_arr = np.array(active_pids, dtype=np.int64)
            for urow in unit_mat:
                vals = urow[active_pids_arr].astype(np.float64)
                if vals.size == 0:
                    continue
                mu = float(np.mean(vals))
                centered = vals - mu
                for idx_local, pid in enumerate(active_pids):
                    x = float(centered[idx_local])
                    pooled_T[pid] += 1
                    pooled_sum[pid] += x
                    pooled_sumsq[pid] += x * x
                    per_answer_T[pid] += 1
                    per_answer_sum[pid] += x
                    per_answer_sumsq[pid] += x * x
                units_used_trial += 1
        assign_total += int(units_used_trial)
        dropped_total += max(0, int(units_total_trial) - int(units_used_trial))

        # Candidate-source diagnostics (same format as aligned watermark runs)
        sd = source_diag.get("selected_docs")
        sp = source_diag.get("selected_providers")
        sat = source_diag.get("above_threshold_docs")
        sm = source_diag.get("selected_sim_mean")
        if isinstance(sd, int):
            source_docs_selected.append(int(sd))
        if isinstance(sp, int):
            source_providers_selected.append(int(sp))
        if isinstance(sat, int):
            source_above_threshold_docs.append(int(sat))
        if isinstance(sm, (int, float)):
            source_sim_means.append(float(sm))

        per_answer_centered = np.zeros(P, dtype=np.float32)
        if active_pids:
            vals = raw_scores[np.array(active_pids, dtype=np.int64)].astype(np.float64)
            mu = float(np.mean(vals))
            per_answer_centered[np.array(active_pids, dtype=np.int64)] = (vals - mu).astype(np.float32)
        winner_pid = int(np.argmax(raw_scores)) if raw_scores.size else 0
        winners.append(winner_pid)

        z_vec = pooled_z_centered_empirical(pooled_T, pooled_sum, pooled_sumsq)
        q_eff = int(len(winners))
        _update_crossed(crossed_at, crossed_at_raw, z_vec=z_vec, T_vec=pooled_T, q_eff=q_eff)

        snap: Optional[Dict[str, Any]] = None
        if log_traj:
            snap = {
                "q": q_eff,
                "pooled_T": pooled_T.tolist(),
                "z": [float(x) for x in z_vec],
                "assign_rate_sofar": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
                "mix_unique_providers_last": int(mix_counts[-1]) if mix_counts else None,
            }
            traj.append(snap)

        append_trial_log(
            trial_log_jsonl,
            {
                "trial_idx": int(trial_idx),
                "wm_scheme": wm_scheme,
                "audit_mode": str(audit_mode),
                "target_provider": int(target_provider),
                "target_doc_id": str(target.get("id")),
                "question": question,
                "question_hash": sha256_text(question),
                "retrieved_docs": retrieved_docs,
                "answer_style": str(answer_style),
                "answer": answer,
                "answer_hash": sha256_text(answer),
                "answer_tokens_est": int(len(answer.split())),
                "winner_pid": int(winner_pid),
                "q_effective": int(q_eff),
                "mix_unique_providers": int(mix_counts[-1]) if mix_counts else None,
                "align": {
                    "mode": "aligned",
                    "source_mode": str(aligned_source_mode),
                    "assignment_mode": "all_providers_scored",
                    "windows_total": int(units_total_trial),
                    "assigned_total": int(units_used_trial),
                    "dropped_total": int(max(0, units_total_trial - units_used_trial)),
                    "best_cos_mean": None,
                    "second_cos_mean": None,
                    "source_selected_docs": int(source_diag.get("selected_docs", 0)),
                    "source_selected_providers": int(source_diag.get("selected_providers", 0)),
                },
                "aligned_source": source_diag,
                "per_provider": {
                    "T": per_answer_T.tolist(),
                    "score_raw": [float(x) for x in raw_scores.tolist()],
                    "score_centered": [float(x) for x in per_answer_centered.tolist()],
                    "sum_centered": [float(x) for x in per_answer_sum.tolist()],
                },
                "baseline_meta": score_meta,
                "wrongkey": {"enabled": False},
                "traj_snapshot": snap,
                "cumulative": {
                    "pooled_T": pooled_T.tolist(),
                    "pooled_sum": [float(x) for x in pooled_sum.tolist()],
                    "pooled_sumsq": [float(x) for x in pooled_sumsq.tolist()],
                    "z": [float(x) for x in z_vec],
                    "assign_total": int(assign_total),
                    "windows_total": int(windows_total),
                    "dropped_total": int(dropped_total),
                },
            },
        )

    final_z_raw = pooled_z_centered_empirical(pooled_T, pooled_sum, pooled_sumsq)
    pvals_raw = [norm_sf(z) for z in final_z_raw]
    final_z, pvals, qvals, bh_sig, bh_eligible = gate_significance_by_min_pooled_T(
        final_z_raw, pooled_T, min_pooled_T=min_pooled_T, alpha=0.05
    )

    baseline_cfg: Dict[str, Any]
    if baseline_kind == "sim_attr":
        baseline_cfg = {
            "kind": "similarity",
            "top_m": int(sim_params.top_m),
            "min_sent_words": int(sim_params.min_sent_words),
        }
    elif baseline_kind == "overlap_attr":
        baseline_cfg = {
            "kind": "overlap",
            "top_m": int(overlap_params.top_m),
            "ngram_n": int(overlap_params.ngram_n),
            "min_sent_words": int(overlap_params.min_sent_words),
            "idf_floor": float(overlap_params.idf_floor),
        }
    else:
        baseline_cfg = {
            "kind": "entailment",
            "nli_model": str(entail_params.nli_model),
            "topk_passages_per_claim": int(entail_params.topk_passages_per_claim),
            "claim_min_words": int(entail_params.claim_min_words),
            "passage_min_words": int(entail_params.passage_min_words),
            "max_claims": int(entail_params.max_claims),
            "batch_size": int(entail_params.batch_size),
            "device": str(entail_params.device),
        }

    out: Dict[str, Any] = {
        "P": int(P),
        "target_provider": int(target_provider),
        "trials": int(trials),
        "topk": int(topk),
        "min_unique_providers": int(min_unique_providers),
        "force_cross_provider": bool(force_cross_provider),
        "exclude_target_provider_from_retrieval": bool(exclude_target_provider_from_retrieval),
        "answer_style": str(answer_style),
        "wm_scheme": str(wm_scheme),
        "audit_mode": str(audit_mode),
        "z_threshold": float(z_threshold),
        "min_pooled_T": int(min_pooled_T),
        "winner_fraction_target": float(np.mean(np.array(winners) == int(target_provider))) if winners else 0.0,
        "mix_unique_providers_mean": float(np.mean(mix_counts)) if mix_counts else 0.0,
        "mix_unique_providers_median": float(np.median(mix_counts)) if mix_counts else 0.0,
        "pooled_T": pooled_T.tolist(),
        "pooled_sumproj": [float(x) for x in pooled_sum.tolist()],
        "pooled_sumsq": [float(x) for x in pooled_sumsq.tolist()],
        "z_raw": [float(x) for x in final_z_raw],
        "z": [float(x) for x in final_z],
        "p_one_sided_raw": [float(x) for x in pvals_raw],
        "p_one_sided": [float(x) for x in pvals],
        "q_bh": [float(x) for x in qvals],
        "bh_sig": [bool(x) for x in bh_sig],
        "bh_eligible": [bool(x) for x in bh_eligible],
        "crossed_at": crossed_at,
        "crossed_at_raw": crossed_at_raw,
        "align": {
            "mode": "aligned",
            "source_mode": str(aligned_source_mode),
            "assignment_mode": "all_providers_scored",
            "windows_total": int(windows_total),
            "assigned_total": int(assign_total),
            "dropped_total": int(dropped_total),
            "assign_rate": float(assign_total / max(1, windows_total)) if windows_total > 0 else 0.0,
            "best_cos_mean": (float(np.mean(best_cos_means)) if best_cos_means else None),
            "second_cos_mean": (float(np.mean(second_cos_means)) if second_cos_means else None),
            "source_docs_selected_mean": (float(np.mean(source_docs_selected)) if source_docs_selected else None),
            "source_docs_selected_median": (
                float(np.median(source_docs_selected)) if source_docs_selected else None
            ),
            "source_docs_selected_p90": (
                float(np.percentile(source_docs_selected, 90)) if source_docs_selected else None
            ),
            "source_providers_selected_mean": (
                float(np.mean(source_providers_selected)) if source_providers_selected else None
            ),
            "global_above_threshold_docs_mean": (
                float(np.mean(source_above_threshold_docs)) if source_above_threshold_docs else None
            ),
            "global_selected_sim_mean": (float(np.mean(source_sim_means)) if source_sim_means else None),
            "global_query_sim_threshold": (
                float(global_source_query_sim_threshold)
                if aligned_source_mode == "global_corpus"
                else None
            ),
            "global_max_docs": (int(global_source_max_docs) if aligned_source_mode == "global_corpus" else None),
            "global_max_docs_per_provider": (
                int(global_source_max_docs_per_provider) if aligned_source_mode == "global_corpus" else None
            ),
            "global_source_selector": (
                str(global_source_selector) if aligned_source_mode == "global_corpus" else None
            ),
        },
        "wrongkey": {"enabled": False},
        "traj": traj if log_traj else None,
        "baseline": baseline_cfg,
    }
    return out


def write_results_csv(path: Path, results: Dict[str, Any], scheme_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    P = int(results["P"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scheme",
                "provider",
                "pooled_T",
                "pooled_sumproj",
                "z",
                "p_one_sided",
                "q_bh",
                "BH_sig",
                "crossed_at",
                "target_provider",
                "trials",
                "topk",
                "min_unique_providers",
                "force_cross_provider",
                "answer_style",
                "z_threshold",
                "min_pooled_T",
            ]
        )
        for pid in range(P):
            w.writerow(
                [
                    scheme_name,
                    pid,
                    int(results["pooled_T"][pid]),
                    float(results["pooled_sumproj"][pid]),
                    float(results["z"][pid]),
                    float(results["p_one_sided"][pid]),
                    float(results["q_bh"][pid]),
                    bool(results["bh_sig"][pid]),
                    results["crossed_at"][pid],
                    int(results["target_provider"]),
                    int(results["trials"]),
                    int(results["topk"]),
                    int(results["min_unique_providers"]),
                    bool(results["force_cross_provider"]),
                    str(results["answer_style"]),
                    float(results["z_threshold"]),
                    int(results.get("min_pooled_T", 1)),
                ]
            )

        wk = results.get("wrongkey", {})
        if wk and wk.get("enabled"):
            w.writerow([])
            w.writerow(["scheme", "provider", "pooled_T", "z_wrongkey", "crossed_at_wrongkey"])
            z_w = wk.get("z") or [0.0] * P
            T_w = wk.get("pooled_T") or [0] * P
            c_w = wk.get("crossed_at") or [None] * P
            for pid in range(P):
                w.writerow([f"{scheme_name}_wrongkey", pid, int(T_w[pid]), float(z_w[pid]), c_w[pid]])


def print_summary(res: Dict[str, Any], title: str) -> None:
    P = int(res["P"])
    z = res["z"]
    print(f"\n=== {title} ===")
    print(f"Target provider: {res['target_provider']}   trials={res['trials']}   topk={res['topk']}")
    print(
        f"Retrieval unique providers: mean={res['mix_unique_providers_mean']:.2f}  "
        f"median={res['mix_unique_providers_median']:.0f}"
    )
    print(f"Winner fraction (target has max per-answer z): {res['winner_fraction_target']:.3f}")
    print("Pooled z by provider:", [f"{i}:{z[i]:.2f}" for i in range(P)])
    print(
        f"Crossed_at (z>={res['z_threshold']:.1f} AND T>={int(res.get('min_pooled_T', 1))}) by provider:",
        res["crossed_at"],
    )

    print("\nBH/FDR @ alpha=0.05 (one-sided):")
    print(f"  (min pooled T gate for z/p/BH: T>={int(res.get('min_pooled_T', 1))})")
    header = f"{'prov':>4}  {'z':>8}  {'p':>10}  {'q(BH)':>10}  {'BH_sig':>7}  {'T':>8}"
    print(header)
    for pid in range(P):
        print(
            f"{pid:>4}  {float(z[pid]):8.2f}  {res['p_one_sided'][pid]:10.2e}  {res['q_bh'][pid]:10.2e}  "
            f"{str(bool(res['bh_sig'][pid])):>7}  {int(res['pooled_T'][pid]):8d}"
        )

    al = res["align"]
    print("\nAlignment summary:")
    print(
        f"  windows_total={al['windows_total']}  assigned_total={al['assigned_total']}  "
        f"dropped_total={al['dropped_total']}  assign_rate={al['assign_rate']:.3f}"
    )
    if al.get("best_cos_mean") is not None:
        print(f"  mean(best_cos)={al['best_cos_mean']:.3f}")
    if al.get("second_cos_mean") is not None:
        print(f"  mean(second_cos)={al['second_cos_mean']:.3f}")
    if al.get("source_docs_selected_mean") is not None:
        print(
            f"  source_docs_selected mean/median/p90="
            f"{float(al['source_docs_selected_mean']):.2f}/"
            f"{float(al['source_docs_selected_median']):.2f}/"
            f"{float(al['source_docs_selected_p90']):.2f}"
        )
    if al.get("source_providers_selected_mean") is not None:
        print(f"  source_providers_selected_mean={float(al['source_providers_selected_mean']):.2f}")
    if al.get("global_above_threshold_docs_mean") is not None:
        print(f"  global_above_threshold_docs_mean={float(al['global_above_threshold_docs_mean']):.2f}")
    if al.get("global_selected_sim_mean") is not None:
        print(f"  global_selected_sim_mean={float(al['global_selected_sim_mean']):.3f}")

    wk = res.get("wrongkey", {})
    if wk and wk.get("enabled"):
        z_w = wk.get("z") or []
        print("\nWrong-key pooled z (should be ~0 for all providers):")
        print([f"{i}:{z_w[i]:.2f}" for i in range(P)])
        print(
            f"Wrong-key crossed_at (z>={res['z_threshold']:.1f} AND T>={int(res.get('min_pooled_T', 1))}):",
            wk.get("crossed_at"),
        )


# ---------------- CLI ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["sanity_single", "rag_multi", "rag_absent_target", "style_sweep"], required=True)
    ap.add_argument("--work_dir", type=Path, required=True)
    ap.add_argument("--target_provider", type=int, default=0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--source_field", type=str, default=None)
    ap.add_argument("--watermarked_field", type=str, default=None)
    ap.add_argument(
        "--wm_scheme",
        choices=["auto", "dirbucket_naive", "kgw_ward", "semstamp", "sim_attr", "overlap_attr", "entail_attr"],
        default="auto",
    )

    ap.add_argument("--retrieval_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--answer_temp", type=float, default=0.55)
    ap.add_argument("--answer_style", choices=["qa", "ownwords", "bullets", "eli5"], default="ownwords")

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_unique_providers", type=int, default=3)
    ap.add_argument("--force_cross_provider", action="store_true")

    ap.add_argument("--bucket_bits", type=int, default=96)
    ap.add_argument("--bucket_seed", type=int, default=2025)
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--direction_scheme", choices=["auto", "naive", "ortho"], default="auto")
    ap.add_argument("--win_words", type=int, default=28)
    ap.add_argument("--stride_words", type=int, default=14)
    ap.add_argument("--sim_threshold", type=float, default=0.86)
    ap.add_argument("--competitive", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--audit_mode",
        choices=["answer_only", "aligned"],
        default="answer_only",
        help="answer_only = strict black-box (answer text only); aligned = answer + source-doc alignment.",
    )
    ap.add_argument(
        "--aligned_source_mode",
        choices=["oracle", "global_corpus", "prompt_wm", "retrieved_src", "manifest_src"],
        default="oracle",
        help=(
            "When --audit_mode=aligned: oracle (alias prompt_wm) uses the true prompted docs; "
            "global_corpus selects candidate source docs by query similarity from the full corpus; "
            "retrieved_src aligns against source_field text for retrieved docs; "
            "manifest_src aligns against --alignment_manifest by doc id."
        ),
    )
    ap.add_argument(
        "--alignment_manifest",
        type=Path,
        default=None,
        help="Optional source-doc manifest for --aligned_source_mode=manifest_src.",
    )
    ap.add_argument(
        "--global_source_query_sim_threshold",
        type=float,
        default=0.55,
        help="For --aligned_source_mode=global_corpus: keep docs with sim(query, doc) >= this threshold.",
    )
    ap.add_argument(
        "--global_source_max_docs",
        type=int,
        default=120,
        help="For --aligned_source_mode=global_corpus: cap selected source docs across all providers (<=0 means no cap).",
    )
    ap.add_argument(
        "--global_source_max_docs_per_provider",
        type=int,
        default=0,
        help="For --aligned_source_mode=global_corpus: optional per-provider cap (<=0 disables).",
    )
    ap.add_argument(
        "--global_source_selector",
        choices=["query", "answer", "max"],
        default="query",
        help=(
            "For --aligned_source_mode=global_corpus: selector used to score candidate source docs "
            "(query=sim(query,doc), answer=sim(answer,doc), max=max(query,answer))."
        ),
    )
    ap.add_argument("--tie_margin", type=float, default=0.02)
    ap.add_argument("--neighbor_smoothing", action="store_true")
    ap.add_argument("--ham_radius", type=int, default=1)
    ap.add_argument("--smooth_agg", choices=["max", "mean"], default="max")

    ap.add_argument("--kgw_tokenizer_model", type=str, default=None)
    ap.add_argument("--kgw_seeding_scheme", type=str, default="ff-position_prf-2-False-1548585")
    ap.add_argument("--kgw_gamma", type=float, default=0.25)
    ap.add_argument("--kgw_delta", type=float, default=3.5)
    ap.add_argument("--kgw_ignore_repeated_ngrams", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--kgw_n_min", type=int, default=3)
    ap.add_argument("--kgw_vocab_size", type=int, default=None)

    ap.add_argument("--semstamp_centroids", type=Path, default=None)
    ap.add_argument("--semstamp_embed_model", type=str, default=None)
    ap.add_argument("--semstamp_num_clusters", type=int, default=None)
    ap.add_argument("--semstamp_gamma", type=float, default=None)
    ap.add_argument("--semstamp_min_sent_words", type=int, default=1)

    # Non-watermark attribution baselines
    ap.add_argument("--simattr_top_m", type=int, default=3)
    ap.add_argument("--simattr_min_sent_words", type=int, default=4)

    ap.add_argument("--overlap_top_m", type=int, default=3)
    ap.add_argument("--overlap_ngram_n", type=int, default=3)
    ap.add_argument("--overlap_min_sent_words", type=int, default=4)
    ap.add_argument("--overlap_idf_floor", type=float, default=0.5)

    ap.add_argument("--entail_model", type=str, default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--entail_topk_passages_per_claim", type=int, default=5)
    ap.add_argument("--entail_claim_min_words", type=int, default=4)
    ap.add_argument("--entail_passage_min_words", type=int, default=4)
    ap.add_argument("--entail_max_claims", type=int, default=8)
    ap.add_argument("--entail_batch_size", type=int, default=16)
    ap.add_argument("--entail_device", choices=["auto", "cpu", "cuda"], default="auto")

    ap.add_argument("--z_threshold", type=float, default=4.0)
    ap.add_argument(
        "--min_pooled_T",
        type=int,
        default=1,
        help=(
            "Minimum pooled windows required to count z-threshold crossings, and to consider "
            "pooled z / p / BH significance (providers below this are gated to z=0, p=1)."
        ),
    )
    ap.add_argument("--do_wrongkey", action="store_true")
    ap.add_argument("--null_unrelated", action="store_true")
    ap.add_argument("--log_traj", action="store_true", help="Log pooled z (and T) after each audited answer.")
    ap.add_argument("--resume_trials", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trial_log_jsonl", type=Path, default=None)

    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--out_json", type=Path, default=None)

    args = ap.parse_args()

    (
        rows,
        keys,
        default_source,
        default_wm,
        default_wm_scheme,
        default_direction_scheme,
        keys_meta,
    ) = load_workdir(args.work_dir)
    source_field = args.source_field or default_source
    watermarked_field = args.watermarked_field or default_wm
    wm_scheme = (default_wm_scheme if args.wm_scheme == "auto" else args.wm_scheme)
    direction_scheme = (default_direction_scheme if args.direction_scheme == "auto" else args.direction_scheme)
    if wm_scheme in {"kgw_ward", "semstamp", "sim_attr", "overlap_attr", "entail_attr"}:
        direction_scheme = "naive"
    if wm_scheme in {"sim_attr", "overlap_attr", "entail_attr"} and args.audit_mode != "aligned":
        print(
            f"[WARN] {wm_scheme} requires source-aware scoring; overriding --audit_mode={args.audit_mode} -> aligned",
            flush=True,
        )
        args.audit_mode = "aligned"

    if source_field not in rows[0]:
        raise ValueError(f"source_field '{source_field}' not present in corpus rows.")
    if watermarked_field not in rows[0]:
        raise ValueError(f"watermarked_field '{watermarked_field}' not present in corpus rows.")

    trial_log_base = args.trial_log_jsonl or default_trial_log_path(args.out_json)
    if trial_log_base is not None and (not args.resume_trials) and trial_log_base.exists():
        trial_log_base.unlink()

    if wm_scheme == "kgw_ward":
        meta_kgw = keys_meta.get("kgw") or {}
        meta_vocab = meta_kgw.get("vocab_size")
        kgw_vocab_size = int(args.kgw_vocab_size if args.kgw_vocab_size is not None else (meta_vocab or 0)) or None
        kgw_tok_model = (
            args.kgw_tokenizer_model
            or str(meta_kgw.get("tokenizer_model") or keys_meta.get("wm_model") or "")
            or "Qwen/Qwen2.5-1.5B-Instruct"
        )
        kgw_params = KGWParams(
            seeding_scheme=str(args.kgw_seeding_scheme),
            gamma=float(args.kgw_gamma),
            delta=float(args.kgw_delta),
            ignore_repeated_ngrams=bool(args.kgw_ignore_repeated_ngrams),
            vocab_size=kgw_vocab_size,
        )
    else:
        kgw_tok_model = ""
        kgw_params = KGWParams()

    semstamp_clusterer: Optional[SemStampClusterer]
    semstamp_params: Optional[SemStampParams]
    if wm_scheme == "semstamp":
        meta_sem = keys_meta.get("semstamp") or {}
        impl_variant = str(meta_sem.get("impl_variant") or "")
        if impl_variant != "ksemstamp_sequence_conditional_v3_answerseq_margin_minwords":
            print(
                "[WARN] SemStamp corpus/keys metadata does not indicate the current "
                "sequence-conditioned k-SemStamp-style implementation "
                "(expected impl_variant=ksemstamp_sequence_conditional_v3_answerseq_margin_minwords). "
                "Rebuild SemStamp corpus to avoid mismatched generator/detector behavior.",
                flush=True,
            )
        sem_embed_model = (
            args.semstamp_embed_model
            or str(meta_sem.get("embed_model") or args.embed_model)
        )
        sem_num_clusters = int(
            args.semstamp_num_clusters
            if args.semstamp_num_clusters is not None
            else (meta_sem.get("num_clusters") or 256)
        )
        sem_gamma = float(
            args.semstamp_gamma if args.semstamp_gamma is not None else (meta_sem.get("gamma") or 0.25)
        )
        sem_centroids_path = args.semstamp_centroids
        if sem_centroids_path is None:
            meta_centroids = str(meta_sem.get("centroids_path") or "semstamp_centroids.npz")
            p = Path(meta_centroids)
            sem_centroids_path = (args.work_dir / p) if not p.is_absolute() else p
        if not sem_centroids_path.exists():
            raise FileNotFoundError(
                f"SemStamp centroids not found: {sem_centroids_path} "
                f"(override with --semstamp_centroids)"
            )
        semstamp_clusterer = SemStampClusterer.load(sem_centroids_path)
        if sem_embed_model and str(sem_embed_model) != str(semstamp_clusterer.embedder.model_name):
            print(
                f"[WARN] --semstamp_embed_model={sem_embed_model} differs from centroid file embed_model="
                f"{semstamp_clusterer.embedder.model_name}; using centroid file embedder for consistency.",
                flush=True,
            )
        semstamp_params = SemStampParams(
            embed_model=str(semstamp_clusterer.embedder.model_name),
            num_clusters=int(sem_num_clusters),
            gamma=float(sem_gamma),
        )
        if int(semstamp_clusterer.num_clusters) != int(semstamp_params.num_clusters):
            print(
                f"[WARN] SemStamp centroids K={semstamp_clusterer.num_clusters} but "
                f"--semstamp_num_clusters/meta={semstamp_params.num_clusters}; using centroid file K.",
                flush=True,
            )
            semstamp_params.num_clusters = int(semstamp_clusterer.num_clusters)
    else:
        semstamp_clusterer = None
        semstamp_params = None

    sim_baseline_params = SimilarityBaselineParams(
        top_m=int(args.simattr_top_m),
        min_sent_words=int(args.simattr_min_sent_words),
    )
    overlap_baseline_params = OverlapBaselineParams(
        top_m=int(args.overlap_top_m),
        ngram_n=int(args.overlap_ngram_n),
        min_sent_words=int(args.overlap_min_sent_words),
        idf_floor=float(args.overlap_idf_floor),
    )
    entail_baseline_params = EntailmentBaselineParams(
        nli_model=str(args.entail_model),
        topk_passages_per_claim=int(args.entail_topk_passages_per_claim),
        claim_min_words=int(args.entail_claim_min_words),
        passage_min_words=int(args.entail_passage_min_words),
        max_claims=int(args.entail_max_claims),
        batch_size=int(args.entail_batch_size),
        device=str(args.entail_device),
    )
    overlap_scorer: Optional[LexicalOverlapScorer] = None
    entail_scorer: Optional[EntailmentScorer] = None
    if wm_scheme == "overlap_attr":
        overlap_scorer = LexicalOverlapScorer(
            rows,
            source_field=source_field,
            ngram_n=int(overlap_baseline_params.ngram_n),
            idf_floor=float(overlap_baseline_params.idf_floor),
        )
    if wm_scheme == "entail_attr":
        entail_scorer = EntailmentScorer(
            model_name=str(entail_baseline_params.nli_model),
            device=str(entail_baseline_params.device),
        )

    if args.exp == "sanity_single":
        args.topk = 1
        args.min_unique_providers = 1
        args.force_cross_provider = False
    exclude_target_provider_from_retrieval = bool(args.exp == "rag_absent_target")

    def _style_log_path(base: Optional[Path], style: str) -> Optional[Path]:
        if base is None:
            return None
        if base.suffix:
            return base.with_name(f"{base.stem}.{style}{base.suffix}")
        return Path(str(base) + f".{style}.jsonl")

    def _run_one(style: str, trial_log_path: Optional[Path]) -> Dict[str, Any]:
        if wm_scheme == "kgw_ward":
            return run_kgw_audit(
                work_dir=args.work_dir,
                rows=rows,
                keys=keys,
                source_field=source_field,
                watermarked_field=watermarked_field,
                retrieval_model=args.retrieval_model,
                target_provider=args.target_provider,
                trials=args.trials,
                topk=args.topk,
                min_unique_providers=args.min_unique_providers,
                force_cross_provider=args.force_cross_provider,
                exclude_target_provider_from_retrieval=exclude_target_provider_from_retrieval,
                answer_style=style,
                openai_model=args.openai_model,
                answer_temp=args.answer_temp,
                kgw_tokenizer_model=kgw_tok_model,
                kgw_params=kgw_params,
                kgw_n_min=args.kgw_n_min,
                competitive=args.competitive,
                audit_mode=args.audit_mode,
                aligned_source_mode=args.aligned_source_mode,
                alignment_manifest=args.alignment_manifest,
                global_source_query_sim_threshold=args.global_source_query_sim_threshold,
                global_source_max_docs=args.global_source_max_docs,
                global_source_max_docs_per_provider=args.global_source_max_docs_per_provider,
                global_source_selector=args.global_source_selector,
                do_wrongkey=args.do_wrongkey,
                null_unrelated=args.null_unrelated,
                z_threshold=args.z_threshold,
                min_pooled_T=args.min_pooled_T,
                log_traj=args.log_traj,
                seed=args.seed,
                wm_scheme=wm_scheme,
                trial_log_jsonl=trial_log_path,
                resume_trials=args.resume_trials,
            )

        if wm_scheme == "semstamp":
            assert semstamp_clusterer is not None and semstamp_params is not None
            return run_semstamp_audit(
                work_dir=args.work_dir,
                rows=rows,
                keys=keys,
                source_field=source_field,
                watermarked_field=watermarked_field,
                retrieval_model=args.retrieval_model,
                target_provider=args.target_provider,
                trials=args.trials,
                topk=args.topk,
                min_unique_providers=args.min_unique_providers,
                force_cross_provider=args.force_cross_provider,
                exclude_target_provider_from_retrieval=exclude_target_provider_from_retrieval,
                answer_style=style,
                openai_model=args.openai_model,
                answer_temp=args.answer_temp,
                semstamp_clusterer=semstamp_clusterer,
                semstamp_params=semstamp_params,
                semstamp_min_sent_words=args.semstamp_min_sent_words,
                sim_threshold=args.sim_threshold,
                tie_margin=args.tie_margin,
                competitive=args.competitive,
                audit_mode=args.audit_mode,
                aligned_source_mode=args.aligned_source_mode,
                alignment_manifest=args.alignment_manifest,
                global_source_query_sim_threshold=args.global_source_query_sim_threshold,
                global_source_max_docs=args.global_source_max_docs,
                global_source_max_docs_per_provider=args.global_source_max_docs_per_provider,
                global_source_selector=args.global_source_selector,
                do_wrongkey=args.do_wrongkey,
                null_unrelated=args.null_unrelated,
                z_threshold=args.z_threshold,
                min_pooled_T=args.min_pooled_T,
                log_traj=args.log_traj,
                seed=args.seed,
                wm_scheme=wm_scheme,
                trial_log_jsonl=trial_log_path,
                resume_trials=args.resume_trials,
            )

        if wm_scheme in {"sim_attr", "overlap_attr", "entail_attr"}:
            return run_nonwm_baseline_audit(
                baseline_kind=str(wm_scheme),
                work_dir=args.work_dir,
                rows=rows,
                keys=keys,
                source_field=source_field,
                watermarked_field=watermarked_field,
                retrieval_model=args.retrieval_model,
                target_provider=args.target_provider,
                trials=args.trials,
                topk=args.topk,
                min_unique_providers=args.min_unique_providers,
                force_cross_provider=args.force_cross_provider,
                exclude_target_provider_from_retrieval=exclude_target_provider_from_retrieval,
                null_unrelated=args.null_unrelated,
                answer_style=style,
                openai_model=args.openai_model,
                answer_temp=args.answer_temp,
                audit_mode=args.audit_mode,
                aligned_source_mode=args.aligned_source_mode,
                alignment_manifest=args.alignment_manifest,
                global_source_query_sim_threshold=args.global_source_query_sim_threshold,
                global_source_max_docs=args.global_source_max_docs,
                global_source_max_docs_per_provider=args.global_source_max_docs_per_provider,
                global_source_selector=args.global_source_selector,
                z_threshold=args.z_threshold,
                min_pooled_T=args.min_pooled_T,
                log_traj=args.log_traj,
                seed=args.seed,
                wm_scheme=wm_scheme,
                trial_log_jsonl=trial_log_path,
                resume_trials=args.resume_trials,
                sim_params=sim_baseline_params,
                overlap_params=overlap_baseline_params,
                overlap_scorer=overlap_scorer,
                entail_params=entail_baseline_params,
                entail_scorer=entail_scorer,
            )

        return run_dirbucket_audit(
            work_dir=args.work_dir,
            rows=rows,
            keys=keys,
            source_field=source_field,
            watermarked_field=watermarked_field,
            retrieval_model=args.retrieval_model,
            target_provider=args.target_provider,
            trials=args.trials,
            topk=args.topk,
            min_unique_providers=args.min_unique_providers,
            force_cross_provider=args.force_cross_provider,
            exclude_target_provider_from_retrieval=exclude_target_provider_from_retrieval,
            answer_style=style,
            openai_model=args.openai_model,
            answer_temp=args.answer_temp,
            bucket_bits=args.bucket_bits,
            bucket_seed=args.bucket_seed,
            embed_model=args.embed_model,
            win_words=args.win_words,
            stride_words=args.stride_words,
            sim_threshold=args.sim_threshold,
            competitive=args.competitive,
            audit_mode=args.audit_mode,
            tie_margin=args.tie_margin,
            neighbor_smoothing=args.neighbor_smoothing,
            ham_radius=args.ham_radius,
            smooth_agg=args.smooth_agg,
            direction_scheme=direction_scheme,
            aligned_source_mode=args.aligned_source_mode,
            alignment_manifest=args.alignment_manifest,
            global_source_query_sim_threshold=args.global_source_query_sim_threshold,
            global_source_max_docs=args.global_source_max_docs,
            global_source_max_docs_per_provider=args.global_source_max_docs_per_provider,
            global_source_selector=args.global_source_selector,
            do_wrongkey=args.do_wrongkey,
            null_unrelated=args.null_unrelated,
            z_threshold=args.z_threshold,
            min_pooled_T=args.min_pooled_T,
            log_traj=args.log_traj,
            seed=args.seed,
            wm_scheme=wm_scheme,
            trial_log_jsonl=trial_log_path,
            resume_trials=args.resume_trials,
        )

    if args.exp == "style_sweep":
        styles = ["qa", "ownwords", "bullets", "eli5"]
        all_runs = []
        for st in styles:
            style_log = _style_log_path(trial_log_base, st)
            if style_log is not None and (not args.resume_trials) and style_log.exists():
                style_log.unlink()
            res = _run_one(style=st, trial_log_path=style_log)
            print_summary(res, title=f"{wm_scheme} STYLE={st}")
            all_runs.append(res)

        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(json_safe({"runs": all_runs}), indent=2), encoding="utf-8")

        if args.out_csv:
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["style", "winner_frac_target", "z_target", "T_target", "crossed_at_target", "mix_mean", "align_rate"])
                for res in all_runs:
                    tp = int(res["target_provider"])
                    w.writerow(
                        [
                            res["answer_style"],
                            float(res["winner_fraction_target"]),
                            float(res["z"][tp]),
                            int(res["pooled_T"][tp]),
                            res["crossed_at"][tp],
                            float(res["mix_unique_providers_mean"]),
                            float(res["align"]["assign_rate"]),
                        ]
                    )
        return

    res = _run_one(style=args.answer_style, trial_log_path=trial_log_base)
    print_summary(res, title=f"{wm_scheme} EXP={args.exp}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(json_safe(res), indent=2), encoding="utf-8")

    if args.out_csv:
        write_results_csv(args.out_csv, res, scheme_name=wm_scheme)


if __name__ == "__main__":
    main()
