"""Build the KGW-adapted corpus view (work/farad_kgw_ward) by watermarked LLM paraphrasing.

Uses a local HF causal LM (paper: Qwen/Qwen2.5-1.5B-Instruct) with the KGW logits processor;
no OpenAI calls. See reproduce/scripts/03_build_baseline_corpora.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from watermark_kirchenbauer import KGWParams


def _stable_shard(doc_id: str, num_shards: int, seed: int = 13) -> int:
    if num_shards <= 1:
        return 0
    h = hashlib.sha256(f"shard:{seed}:{doc_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % num_shards


def _shard_quota(max_docs_total: int, num_shards: int, shard_id: int) -> int:
    if num_shards <= 1:
        return max_docs_total
    base = max_docs_total // num_shards
    rem = max_docs_total % num_shards
    return base + (1 if shard_id < rem else 0)


def _iter_manifest_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            did = rec.get("id")
            if did is not None:
                done.add(str(did))
    return done


def _merge_shards(out_dir: Path, num_shards: int, out_path: Path) -> int:
    recs: Dict[str, Dict[str, Any]] = {}
    for sid in range(0, num_shards):
        sp = out_dir / f"corpus.shard{sid:02d}.jsonl"
        if not sp.exists():
            continue
        with sp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                did = rec.get("id")
                if did is None:
                    continue
                did_s = str(did)
                if did_s not in recs:
                    recs[did_s] = rec
    items = sorted(recs.values(), key=lambda r: str(r.get("id", "")))
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return len(items)


def _choose_device(spec: str) -> str:
    import torch

    s = (spec or "auto").lower()
    if s != "auto":
        return s
    return "cuda" if torch.cuda.is_available() else "cpu"


def _choose_dtype(spec: str, device: str) -> torch.dtype:
    import torch

    s = (spec or "auto").lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    if device == "cuda":
        # Works well on modern GPUs; user can override.
        return torch.bfloat16
    return torch.float32


_PARAPHRASE_PROMPT = (
    "You are an expert rewriter. Rewrite the following document keeping its meaning and fluency "
    "and especially length. It is crucial to retain all factual information in the original document. "
    "DO NOT MAKE THE TEXT SHORTER. Do not start your response by 'Sure' or anything similar, simply "
    "output the paraphrased document directly. Do not add stylistic elements or anything similar, "
    "try to be faithful to the original content and style of writing. Do not be too formal. "
    "Keep all the factual information.\n\nDOCUMENT:\n{doc}"
)


def _build_prompt(tokenizer: AutoTokenizer, doc_text: str) -> str:
    user_msg = _PARAPHRASE_PROMPT.format(doc=doc_text)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return user_msg


def _generate_watermarked_text(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text_src: str,
    key_hex: str,
    kgw_params: KGWParams,
    device: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    import torch
    from transformers import LogitsProcessorList

    from watermark_kirchenbauer import KGWLogitsProcessor

    prompt = _build_prompt(tokenizer, text_src)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    proc = KGWLogitsProcessor(
        tokenizer=tokenizer,
        key_hex=key_hex,
        params=kgw_params,
        device=torch.device(device),
    )
    proc_list = LogitsProcessorList([proc])

    gen_kwargs: Dict[str, Any] = {
        "logits_processor": proc_list,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": bool(do_sample),
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    new_ids = out[0, prompt_len:]
    text_wm = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if text_wm:
        return text_wm

    # Fallback: decode the full output and remove the prompt prefix if possible.
    full = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if full.startswith(prompt):
        full = full[len(prompt) :].strip()
    return full


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a KGW/WARD-style token-watermarked corpus from a shared manifest."
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--providers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_docs", type=int, default=400)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--merge_shards", action="store_true")

    ap.add_argument("--wm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    ap.add_argument("--warmup_only", action="store_true")

    ap.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)

    ap.add_argument("--seeding_scheme", type=str, default="ff-position_prf-2-False-1548585")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=3.5)
    ap.add_argument("--ignore_repeated_ngrams", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--topk_mask",
        type=int,
        default=None,
        help="Optional speed mode for generation-time bias (non-faithful). Default None is faithful.",
    )
    ap.add_argument("--timing_log", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise SystemExit(f"Invalid --shard_id={shard_id} for --num_shards={num_shards}")

    if args.merge_shards:
        out_path = args.out_dir / "corpus.jsonl"
        n = _merge_shards(args.out_dir, num_shards=num_shards, out_path=out_path)
        print(f"[MERGE] Wrote {n} docs → {out_path}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from watermark_kirchenbauer import KGWDetectionResult, KGWParams, detect_kgw_text

    kgw_params = KGWParams(
        seeding_scheme=str(args.seeding_scheme),
        gamma=float(args.gamma),
        delta=float(args.delta),
        ignore_repeated_ngrams=bool(args.ignore_repeated_ngrams),
        topk_mask=(None if args.topk_mask is None else int(args.topk_mask)),
    )

    keys_path = args.out_dir / "provider_keys.json"
    keys_meta: Dict[str, Any]
    provider_keys: List[str]
    if keys_path.exists():
        keys_meta = json.loads(keys_path.read_text(encoding="utf-8"))
        provider_keys = list(keys_meta["keys_hex"])
        if int(keys_meta.get("providers", len(provider_keys))) != int(args.providers):
            raise SystemExit(
                f"providers mismatch: keys file has {keys_meta.get('providers')} but --providers={args.providers}"
            )
    else:
        prng = random.Random(int(args.seed))
        provider_keys = [prng.randbytes(32).hex() for _ in range(int(args.providers))]
        keys_meta = {
            "keys_hex": provider_keys,
            "providers": int(args.providers),
            "seed": int(args.seed),
            "watermark_scheme": "kgw_ward",
        }

    device = _choose_device(args.device)
    torch_dtype = _choose_dtype(args.dtype, device=device)

    print(f"[KGW] device={device} dtype={torch_dtype} model={args.wm_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.wm_model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.wm_model, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    # Use the model's logits vocab size for both generation and detection parity.
    model_vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    if model_vocab_size <= 0:
        model_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    kgw_params.vocab_size = int(model_vocab_size)

    # Keep key metadata synchronized even when resuming from existing files.
    kgw_meta = dict(keys_meta.get("kgw") or {})
    kgw_meta.update(
        {
            "seeding_scheme": kgw_params.seeding_scheme,
            "gamma": float(kgw_params.gamma),
            "delta": float(kgw_params.delta),
            "ignore_repeated_ngrams": bool(kgw_params.ignore_repeated_ngrams),
            "topk_mask": kgw_params.topk_mask,
            "vocab_size": int(kgw_params.vocab_size),
        }
    )
    keys_meta.update(
        {
            "keys_hex": provider_keys,
            "providers": int(args.providers),
            "seed": int(args.seed),
            "watermark_scheme": "kgw_ward",
            "wm_model": str(args.wm_model),
            "kgw": kgw_meta,
        }
    )
    keys_path.write_text(json.dumps(keys_meta, indent=2), encoding="utf-8")

    if args.warmup_only:
        print("[WARMUP] model+tokenizer loaded and cached.")
        return

    merged_out = args.out_dir / "corpus.jsonl"
    out_path = merged_out if num_shards <= 1 else (args.out_dir / f"corpus.shard{shard_id:02d}.jsonl")
    done_ids = _load_done_ids(out_path) if args.resume else set()
    quota = _shard_quota(int(args.max_docs), num_shards=num_shards, shard_id=shard_id)
    remaining = max(0, quota - len(done_ids))
    if remaining <= 0:
        print(f"[DONE] shard output already has {len(done_ids)} docs (quota={quota}): {out_path}")
        return

    mode = "a" if args.resume else "w"
    written = 0
    sum_z = 0.0
    sum_gf = 0.0
    pbar = tqdm(total=remaining, desc=f"Build KGW corpus (shard{shard_id:02d})", unit="doc")
    with out_path.open(mode, encoding="utf-8") as f_out:
        for row in _iter_manifest_rows(args.manifest):
            if written >= remaining:
                break

            did = str(row.get("id") or "").strip()
            if not did:
                continue
            if did in done_ids:
                continue
            if num_shards > 1 and _stable_shard(did, num_shards=num_shards, seed=int(args.seed)) != shard_id:
                continue

            pid = int(row.get("provider_id", -1))
            if pid < 0 or pid >= len(provider_keys):
                continue
            text_src = str(row.get("text_src") or "").strip()
            if not text_src:
                continue

            t0 = time.perf_counter()
            text_wm = _generate_watermarked_text(
                model=model,
                tokenizer=tokenizer,
                text_src=text_src,
                key_hex=provider_keys[pid],
                kgw_params=kgw_params,
                device=device,
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_new_tokens=int(args.max_new_tokens),
            )
            det: KGWDetectionResult = detect_kgw_text(
                text=text_wm,
                key_hex=provider_keys[pid],
                tokenizer=tokenizer,
                params=kgw_params,
            )
            dt = time.perf_counter() - t0
            if args.timing_log:
                print(
                    f"[KGW-TIME] id={did[:24]} shard={shard_id:02d} total={dt:.2f}s "
                    f"T={det.T} z={det.z:.2f}",
                    flush=True,
                )

            rec: Dict[str, Any] = {
                "id": did,
                "title": str(row.get("title") or ""),
                "provider_id": pid,
                "text_src": text_src,
                "text_wm": text_wm,
                "wm_T": int(det.T),
                "wm_greens": int(det.greens),
                "wm_z": float(det.z),
                "wm_green_fraction": float(det.green_fraction),
                "wm_latency_s": float(dt),
                "src_chars": int(len(text_src)),
                "wm_chars": int(len(text_wm)),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            sum_z += float(det.z)
            sum_gf += float(det.green_fraction)
            done_ids.add(did)
            pbar.update(1)
            if written % 10 == 0:
                mean_z = sum_z / max(1, written)
                mean_gf = sum_gf / max(1, written)
                print(
                    f"[KGW-QC] shard={shard_id:02d} done={written} "
                    f"mean_z={mean_z:.2f} mean_green_frac={mean_gf:.3f} gamma={kgw_params.gamma:.3f}",
                    flush=True,
                )
                if written >= 20 and mean_gf <= float(kgw_params.gamma) + 0.01:
                    print(
                        "[KGW-WARN] Mean green fraction is near gamma baseline; watermark "
                        "bias may be ineffective (check RNG/vocab consistency).",
                        flush=True,
                    )
            if (written % 3) == 0:
                f_out.flush()
    pbar.close()

    if num_shards > 1:
        print(
            f"[DONE] Built shard shard{shard_id:02d}: +{written} docs → {out_path} "
            f"(quota={quota}, resume={args.resume})"
        )
        print(
            f"[NOTE] Merge shards: python experiments/build_corpora_kgw.py "
            f"--out_dir {args.out_dir} --merge_shards --num_shards {num_shards}"
        )
    else:
        print(f"[DONE] Built/extended KGW corpus: +{written} docs → {out_path}")


if __name__ == "__main__":
    main()
