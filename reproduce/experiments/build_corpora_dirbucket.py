"""Build a DirBucket-watermarked provider corpus (work/farad_dirbucket_{naive,ortho}).

Selects up to --summary_max_sents sentences per document (MMR), paraphrases each with the
OpenAI API and keeps the candidate that maximises the keyed projection gain. Requires
OPENAI_API_KEY. See reproduce/scripts/02_build_dirbucket.sh for the paper arguments.
"""
from __future__ import annotations
import argparse, json, os, random, gzip, re, hashlib, time
from pathlib import Path
from typing import Dict, List, Iterable, Any, Tuple, Optional, Set

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from sentence_transformers import SentenceTransformer

# DirBucket watermark (provider-side)
from watermark_dirbucket import (
    DirBucketWatermarker, GenParams, split_sentences
)

# =========================
# Robust FARAD loaders
# =========================

def _open_any(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def _yield_records_from_file(p: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON objects from .jsonl / .ndjson / .json (+ .gz).
    Skips malformed lines but continues scanning.
    """
    try:
        if p.suffix in {".jsonl", ".ndjson"} or str(p).endswith((".jsonl.gz", ".ndjson.gz")):
            with _open_any(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        else:
            with _open_any(p) as f:
                text = f.read()
            obj = json.loads(text)
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for rec in obj:
                    if isinstance(rec, dict):
                        yield rec
    except Exception:
        return

_PREFERRED_MODELS = ["gpt4o", "gpt-4o", "gpt4", "gpt-4", "claude", "llama", "qwen"]

def _pick_generated_article(arts: Dict[str, Any]) -> Tuple[str, str] | None:
    if not isinstance(arts, dict):
        return None
    for pref in _PREFERRED_MODELS:
        for key, val in arts.items():
            if pref in key.lower() and isinstance(val, dict):
                art = val.get("article")
                if isinstance(art, str) and art.strip():
                    return (key, art)
    for key, val in arts.items():
        if isinstance(val, dict):
            art = val.get("article")
            if isinstance(art, str) and art.strip():
                return (key, art)
    return None

def _extract_article(rec: Dict[str, Any]) -> tuple[str, str, str] | None:
    doc_id = str(rec.get("id") or rec.get("_id") or rec.get("doc_id") or rec.get("uuid") or "")
    title = rec.get("title") or rec.get("topic") or ""

    orig = rec.get("original_doc")
    if isinstance(orig, str) and orig.strip():
        return (doc_id or title or "NO_ID", title or "NO_TITLE", orig)

    arts = rec.get("articles")
    picked = _pick_generated_article(arts) if isinstance(arts, dict) else None
    if picked:
        _, art = picked
        return (doc_id or title or "NO_ID", title or "NO_TITLE", art)

    for k in ("article", "text", "content"):
        val = rec.get(k)
        if isinstance(val, str) and val.strip():
            return (doc_id or title or "NO_ID", title or "NO_TITLE", val)

    return None

def iter_farad_articles(farad_dir: Path):
    patterns = ["*.jsonl", "*.ndjson", "*.json", "*.jsonl.gz", "*.ndjson.gz", "*.json.gz"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(farad_dir.rglob(pat)))

    print(f"[FARAD] Found {len(files)} candidate files under {farad_dir}")
    shown = 0
    for p in files:
        for rec in _yield_records_from_file(p):
            got = _extract_article(rec)
            if not got:
                continue
            doc_id, title, text = got
            if shown < 3:
                print(f"[FARAD] Example doc id @{p.name}: {doc_id[:24]}… (title='{(title or '')[:40]}')")
                shown += 1
            yield {"id": doc_id, "title": title, "text": text}


def iter_manifest_articles(manifest_path: Path):
    """
    Yield rows from a shared manifest jsonl with fields:
      - id
      - title (optional)
      - text_src (preferred) or text
      - provider_id (optional; used if present)
    """
    shown = 0
    with manifest_path.open("r", encoding="utf-8") as f:
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
            doc_id = str(row.get("id") or row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            title = str(row.get("title") or "")
            text = str(row.get("text_src") or row.get("text") or "").strip()
            if not text:
                continue
            out = {"id": doc_id, "title": title, "text": text}
            if "provider_id" in row:
                try:
                    out["provider_id"] = int(row["provider_id"])
                except Exception:
                    pass
            if shown < 3:
                print(f"[MANIFEST] Example doc id: {doc_id[:24]}… (title='{title[:40]}')")
                shown += 1
            yield out

# =========================
# Sentence selection (MMR)
# =========================

def sent_tokenize_fallback(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if len(s.strip()) > 2]

def mmr_select(
    sentences: list[str],
    k: int,
    lambda_mult: float,
    mmr_encoder: SentenceTransformer,
) -> list[int]:
    if len(sentences) <= k:
        return list(range(len(sentences)))

    emb = mmr_encoder.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
    doc_vec = emb.mean(axis=0, keepdims=True)

    selected: List[int] = []
    cand = list(range(len(sentences)))
    sims_doc = (emb @ doc_vec.T).ravel()

    first = int(np.argmax(sims_doc))
    selected.append(first)
    cand.remove(first)

    while len(selected) < k and cand:
        max_score, best = -1e9, None
        for c in cand:
            relevance = float(sims_doc[c])
            diversity = max(float(emb[c] @ emb[s]) for s in selected)
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > max_score:
                max_score, best = score, c
        selected.append(best)
        cand.remove(best)
    return sorted(selected)

# =========================
# Provider assignment & sharding
# =========================

def stable_partition(doc_id: str, P: int, seed: int = 13) -> int:
    h = hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % P

def stable_shard(doc_id: str, num_shards: int, seed: int = 13) -> int:
    """Deterministically assign a doc to a build shard (independent of provider partition)."""
    if num_shards <= 1:
        return 0
    h = hashlib.sha256(f"shard:{seed}:{doc_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % num_shards

def shard_quota(max_docs_total: int, num_shards: int, shard_id: int) -> int:
    """Split max_docs_total into per-shard quotas summing exactly to max_docs_total."""
    if num_shards <= 1:
        return max_docs_total
    base = max_docs_total // num_shards
    rem = max_docs_total % num_shards
    return base + (1 if shard_id < rem else 0)

def merge_shards(out_dir: Path, num_shards: int, out_path: Path) -> int:
    """Merge corpus.shardXX.jsonl files into corpus.jsonl. Dedup by id. Returns count."""
    shard_paths = [out_dir / f"corpus.shard{sid:02d}.jsonl" for sid in range(num_shards)]
    recs: Dict[str, Dict[str, Any]] = {}
    for sp in shard_paths:
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
                if not did:
                    continue
                if did not in recs:
                    recs[did] = rec

    items = list(recs.values())
    items.sort(key=lambda r: str(r.get("id", "")))

    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return len(items)

def choose_device(mmr_device: str) -> str:
    mmr_device = (mmr_device or "auto").lower()
    if mmr_device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return mmr_device


def _require_openai_key() -> None:
    """Abort before a paraphrasing build when no OpenAI credentials are configured.

    The reference paraphraser in watermark_dirbucket.py swallows API errors and keeps the
    source sentence, so a build without a key would silently write an unwatermarked corpus.
    A custom OPENAI_BASE_URL (e.g. a local vLLM server) still needs a (dummy) OPENAI_API_KEY.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. The corpus build paraphrases every sentence with the "
            "OpenAI API (or an OpenAI-compatible server via OPENAI_BASE_URL); without a key it "
            "would silently produce an unwatermarked corpus. Export OPENAI_API_KEY and retry."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Append to an existing corpus file and skip already processed doc IDs.")
    ap.add_argument("--farad_dir", type=Path, required=False,
                    help="FARAD directory (required unless --manifest is provided).")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Optional shared manifest jsonl with id/title/text_src/provider_id.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--providers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_docs", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=4000, help="truncate source text before processing")
    ap.add_argument("--rewrite_mode", choices=["summary", "full"], default="summary",
                    help="summary = select top-N salient sentences first; full = watermark all sentences")
    ap.add_argument("--summary_select", choices=["mmr", "firstk"], default="mmr")
    ap.add_argument("--summary_max_sents", type=int, default=12)
    ap.add_argument("--summary_lambda", type=float, default=0.7)
    ap.add_argument("--timing_log", action="store_true",
                    help="Log timing per sentence/doc to diagnose slowness.")
    ap.add_argument("--timing_every", type=int, default=1,
                    help="Log every N sentences when --timing_log is set.")

    # DirBucket params (generator + buckets)
    ap.add_argument("--bucket_bits", type=int, default=96)
    ap.add_argument("--bucket_seed", type=int, default=2025)
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--direction_scheme", choices=["naive", "ortho"], default="naive")

    ap.add_argument("--gen_mode", choices=["batch", "contrastive"], default="batch")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k_batch", type=int, default=24)
    ap.add_argument("--k_stage1", type=int, default=16)
    ap.add_argument("--k_stage2", type=int, default=8)
    ap.add_argument("--min_sem_sim", type=float, default=0.70)
    ap.add_argument("--min_delta_proj", type=float, default=5e-4)
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--force_accept_last_round", action="store_true")

    ap.add_argument("--mmr_embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--mmr_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Parallelism knobs (within a single shard process)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inflight", type=int, default=0, help="0 => 2*workers")

    # Multi-process sharding (useful for multi-GPU / multi-node parallel builds)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total number of build shards (processes).")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Which shard this process is responsible for: 0..num_shards-1")
    ap.add_argument("--merge_shards", action="store_true",
                    help="Merge corpus.shardXX.jsonl files into corpus.jsonl and exit.")
    ap.add_argument("--warmup_only", action="store_true",
                    help="Download/cache embedding models then exit.")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Early-exit utilities
    if args.warmup_only:
        try:
            SentenceTransformer(args.embed_model)
        except Exception:
            pass
        try:
            SentenceTransformer(args.mmr_embed_model)
        except Exception:
            pass
        print("[WARMUP] sentence-transformers models are cached.")
        return

    if args.merge_shards:
        out_path = args.out_dir / "corpus.jsonl"
        n = merge_shards(args.out_dir, max(1, int(args.num_shards)), out_path)
        print(f"[MERGE] Wrote {n} docs → {out_path}")
        return

    if args.manifest is None and args.farad_dir is None:
        raise SystemExit("Provide either --farad_dir or --manifest.")
    _require_openai_key()

    # Sharding setup
    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise SystemExit(f"Invalid --shard_id={shard_id} for --num_shards={num_shards}")
    shard_tag = f"shard{shard_id:02d}" if num_shards > 1 else None

    # Provider keys
    keys_path = args.out_dir / "provider_keys.json"

    # In sharded mode multiple processes may start simultaneously. Make shard 0 create keys; others wait.
    provider_keys: List[str]
    if keys_path.exists():
        meta = json.loads(keys_path.read_text(encoding="utf-8"))
        provider_keys = list(meta["keys_hex"])
        assert len(provider_keys) == args.providers, "providers mismatch vs existing keys"
        if "direction_scheme" in meta:
            # If resuming an existing out_dir, keep the stored direction scheme for consistency.
            args.direction_scheme = str(meta["direction_scheme"]).lower()
    else:
        if num_shards > 1 and shard_id != 0:
            # Wait briefly for shard 0 to create the keys file.
            t0 = time.time()
            while not keys_path.exists():
                if time.time() - t0 > 120:
                    raise RuntimeError(f"Timed out waiting for {keys_path} to be created by shard 0.")
                time.sleep(0.25)
            meta = json.loads(keys_path.read_text(encoding="utf-8"))
            provider_keys = list(meta["keys_hex"])
        else:
            prng = random.Random(args.seed)
            provider_keys = [prng.randbytes(32).hex() for _ in range(args.providers)]
            keys_path.write_text(
                json.dumps(
                    {
                        "keys_hex": provider_keys,
                        "providers": args.providers,
                        "seed": args.seed,
                        "direction_scheme": args.direction_scheme,
                        "bucket_bits": args.bucket_bits,
                        "bucket_seed": args.bucket_seed,
                        "embed_model": args.embed_model,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # GenParams used for ALL providers (key differs per provider, scheme changes directions)
    gparams = GenParams(
        bucket_bits=args.bucket_bits,
        bucket_seed=args.bucket_seed,
        embed_model=args.embed_model,
        min_sem_sim=args.min_sem_sim,
        min_delta_proj=args.min_delta_proj,
        rounds=args.rounds,
        k_batch=args.k_batch,
        k_stage1=args.k_stage1,
        k_stage2=args.k_stage2,
        strategy=args.gen_mode,
        openai_model=args.openai_model,
    )

    # Output file: either a single corpus.jsonl or per-shard corpus.shardXX.jsonl
    merged_out_path = args.out_dir / "corpus.jsonl"
    out_path = merged_out_path if num_shards <= 1 else (args.out_dir / f"corpus.shard{shard_id:02d}.jsonl")

    # Fast skip: if a merged corpus already exists with enough docs, do nothing.
    if num_shards > 1 and merged_out_path.exists():
        try:
            n_exist = 0
            with merged_out_path.open("r", encoding="utf-8") as _f:
                for _line in _f:
                    if _line.strip():
                        n_exist += 1
                    if n_exist >= args.max_docs:
                        break
            if n_exist >= args.max_docs:
                print(f"[DONE] merged corpus already has {n_exist} docs (>= max_docs={args.max_docs}): {merged_out_path}")
                return
        except Exception:
            pass

    # Warm MMR encoder once per process (with fallback if assigned GPU is not usable).
    device = choose_device(args.mmr_device)
    try:
        mmr_encoder = SentenceTransformer(args.mmr_embed_model, device=device)
    except Exception as e:
        if device == "cuda":
            print(f"[WARN] Failed to init MMR encoder on CUDA ({type(e).__name__}: {e}). Falling back to CPU.")
            mmr_encoder = SentenceTransformer(args.mmr_embed_model, device="cpu")
        else:
            raise

    # Resume tracking (per-output-file)
    done_ids: Set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        done_ids.add(str(rec["id"]))
                except Exception:
                    break  # stop at first bad/truncated line

    quota = shard_quota(args.max_docs, num_shards, shard_id)
    remaining = max(0, quota - len(done_ids))
    if remaining <= 0:
        print(f"[DONE] shard output already has {len(done_ids)} docs (quota={quota}): {out_path}")
        return

    mode = "a" if args.resume else "w"

    # Small cache: watermarker per provider id
    wm_cache: Dict[int, DirBucketWatermarker] = {}

    def get_watermarker(pid: int) -> DirBucketWatermarker:
        wm = wm_cache.get(pid)
        if wm is None:
            key_hex = provider_keys[pid]
            wm = DirBucketWatermarker(
                key_hex,
                gparams,
                direction_scheme=args.direction_scheme,
                provider_id=int(pid) if args.direction_scheme == "ortho" else None,
                all_provider_keys=provider_keys if args.direction_scheme == "ortho" else None,
                force_accept_last_round=args.force_accept_last_round,
            )
            wm_cache[pid] = wm
        return wm

    def process_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        t_doc0 = time.perf_counter()
        raw_text = (row.get("text") or row.get("text_src") or "").strip()
        if not raw_text:
            return None
        raw_text = raw_text[: args.max_chars]

        doc_id = str(row["id"])
        if "provider_id" in row:
            try:
                pid = int(row["provider_id"])
            except Exception:
                pid = stable_partition(doc_id, args.providers, args.seed)
        else:
            pid = stable_partition(doc_id, args.providers, args.seed)
        if pid < 0 or pid >= int(args.providers):
            return None

        # Construct source chunk (summary/full)
        sel_dt = 0.0
        sents_total = None
        kept_total = None
        if args.rewrite_mode == "summary":
            t_sel0 = time.perf_counter()
            try:
                sents = split_sentences(raw_text) or []
            except Exception:
                sents = []
            if not sents:
                sents = sent_tokenize_fallback(raw_text)
            sents_total = len(sents)

            if args.summary_select == "firstk":
                keep_idx = list(range(min(args.summary_max_sents, len(sents))))
            else:
                keep_idx = mmr_select(
                    sents,
                    k=min(args.summary_max_sents, len(sents)),
                    lambda_mult=args.summary_lambda,
                    mmr_encoder=mmr_encoder,
                )
            kept_total = len(keep_idx)
            text_for_wm = " ".join([sents[i] for i in keep_idx])
            sel_dt = time.perf_counter() - t_sel0
        else:
            text_for_wm = raw_text

        wm = get_watermarker(pid)
        t_wm0 = time.perf_counter()
        wm_text, summary = wm.watermark(
            text_for_wm,
            timing=args.timing_log,
            timing_doc_id=doc_id,
            timing_every=args.timing_every,
        )
        wm_dt = time.perf_counter() - t_wm0
        doc_dt = time.perf_counter() - t_doc0
        if args.timing_log:
            parts = [
                f"[DOC-TIME] id={doc_id[:24]}",
                f"shard={shard_tag or 'single'}",
                f"wm={wm_dt:.2f}s",
                f"total={doc_dt:.2f}s",
            ]
            if sents_total is not None:
                parts.extend([
                    f"sents={sents_total}",
                    f"kept={kept_total}",
                    f"select={sel_dt:.2f}s",
                ])
            print(" ".join(parts), flush=True)

        return {
            "id": doc_id,
            "title": row.get("title", ""),
            "provider_id": pid,
            "text_wm": wm_text,
            "wm_gain_mean": float(getattr(summary, "gain_mean", 0.0)),
            "wm_changed_frac": float(getattr(summary, "changed_frac", 0.0)),
        }

    inflight = args.inflight if args.inflight and args.inflight > 0 else max(1, 2 * args.workers)

    completed = 0
    submitted = 0

    pbar = tqdm(total=remaining, desc=f"Build corpus ({shard_tag or 'single'})", unit="doc")

    with out_path.open(mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures: Set[Future] = set()

            src_iter = iter_manifest_articles(args.manifest) if args.manifest else iter_farad_articles(args.farad_dir)
            for row in src_iter:
                if completed >= remaining:
                    break

                doc_id = row["id"]

                if num_shards > 1 and stable_shard(doc_id, num_shards, args.seed) != shard_id:
                    continue

                if doc_id in done_ids:
                    continue

                # throttle futures
                while len(futures) >= inflight:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        rec = fut.result()
                        if rec is None:
                            continue
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done_ids.add(rec["id"])
                        completed += 1
                        pbar.update(1)
                        if (completed % 5) == 0:
                            f_out.flush()

                        if completed >= remaining:
                            break

                futures.add(ex.submit(process_row, row))
                submitted += 1

            # drain remaining futures
            while futures and completed < remaining:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    rec = fut.result()
                    if rec is None:
                        continue
                    if rec["id"] in done_ids:
                        continue
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done_ids.add(rec["id"])
                    completed += 1
                    pbar.update(1)
                    if (completed % 5) == 0:
                        f_out.flush()

                    if completed >= remaining:
                        break

    pbar.close()

    if num_shards > 1:
        print(f"[DONE] Built shard {shard_tag}: +{completed} docs → {out_path} (quota={quota}, resume={args.resume})")
        print(f"[NOTE] Merge shards: python experiments/build_corpora_dirbucket.py --out_dir {args.out_dir} --merge_shards --num_shards {num_shards}")
    else:
        print(f"[DONE] Built/extended DirBucket corpus: +{completed} docs → {out_path}")

if __name__ == "__main__":
    main()
