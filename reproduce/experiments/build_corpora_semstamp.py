"""Build the SemStamp-adapted corpus view (paper: work/farad_semstamp_k8_relaxed, K=8, gamma=0.25).

Trains k-means centroids on manifest sentences, then rejection-samples OpenAI paraphrases whose
semantic cluster falls in the provider's green set. Requires OPENAI_API_KEY for the build step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tqdm import tqdm


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


def _collect_training_sentences(
    *,
    manifest: Path,
    max_sents: int,
    per_doc_cap: int,
    seed: int,
    min_words: int,
) -> List[str]:
    from watermark_semstamp import split_sentences

    all_sents: List[str] = []
    for row in _iter_manifest_rows(manifest):
        txt = str(row.get("text_src") or "").strip()
        if not txt:
            continue
        sents = [s.strip() for s in split_sentences(txt) if s and s.strip()]
        if min_words > 1:
            sents = [s for s in sents if len(s.split()) >= int(min_words)]
        if per_doc_cap > 0 and len(sents) > per_doc_cap:
            did = str(row.get("id") or "")
            h = hashlib.sha256(f"{seed}:{did}".encode()).digest()
            rng = random.Random(int.from_bytes(h[:8], "big"))
            rng.shuffle(sents)
            sents = sents[:per_doc_cap]
        all_sents.extend(sents)
    if max_sents > 0 and len(all_sents) > max_sents:
        rng = random.Random(int(seed))
        rng.shuffle(all_sents)
        all_sents = all_sents[:max_sents]
    return all_sents



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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a SemStamp-watermarked corpus from a shared manifest."
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

    # SemStamp core params
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--num_clusters", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Centroid training / loading
    ap.add_argument("--centroids_path", type=Path, default=None)
    ap.add_argument("--centroid_max_sents", type=int, default=20000)
    ap.add_argument("--centroid_max_sents_per_doc", type=int, default=64)
    ap.add_argument("--centroid_min_words", type=int, default=4)
    ap.add_argument("--train_centroids_only", action="store_true")
    ap.add_argument("--warmup_only", action="store_true")

    # Paraphrase search (SemStamp adaptation to our pipeline)
    ap.add_argument("--gen_strategy", choices=["batch", "contrastive"], default="batch")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k_batch", type=int, default=20)
    ap.add_argument("--k_stage1", type=int, default=16)
    ap.add_argument("--k_stage2", type=int, default=8)
    ap.add_argument("--min_sem_sim", type=float, default=0.78)
    ap.add_argument("--cluster_margin", type=float, default=0.02)
    ap.add_argument(
        "--gen_min_sent_words",
        type=int,
        default=1,
        help="Minimum words per sentence for SemStamp generation and build-time QC detection.",
    )
    ap.add_argument("--keep_if_already_green", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force_accept_last_round", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--timing_log", action="store_true")
    ap.add_argument("--timing_every", type=int, default=1)
    ap.add_argument("--qc_every_docs", type=int, default=5)
    ap.add_argument("--qc_rolling_docs", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise SystemExit(f"Invalid --shard_id={shard_id} for --num_shards={num_shards}")

    if args.merge_shards:
        out_path = args.out_dir / "corpus.jsonl"
        n = _merge_shards(args.out_dir, num_shards=num_shards, out_path=out_path)
        print(f"[MERGE] Wrote {n} docs -> {out_path}")
        return

    # Lazy import to keep --merge_shards fast and dependency-light.
    from watermark_semstamp import (
        SemStampClusterer,
        SemStampGenParams,
        SemStampParams,
        SemStampWatermarker,
        detect_semstamp_text,
    )

    centroids_path = args.centroids_path or (args.out_dir / "semstamp_centroids.npz")
    if centroids_path.exists():
        clusterer = SemStampClusterer.load(centroids_path, device=str(args.device))
        print(
            f"[SEMSTAMP] loaded centroids={centroids_path} K={clusterer.num_clusters} "
            f"dim={clusterer.dim} embed_model={clusterer.embedder.model_name}"
        )
    else:
        print(
            f"[SEMSTAMP] training centroids K={int(args.num_clusters)} "
            f"from manifest={args.manifest} (max_sents={int(args.centroid_max_sents)})"
        )
        train_sents = _collect_training_sentences(
            manifest=args.manifest,
            max_sents=int(args.centroid_max_sents),
            per_doc_cap=int(args.centroid_max_sents_per_doc),
            seed=int(args.seed),
            min_words=int(args.centroid_min_words),
        )
        if len(train_sents) < int(args.num_clusters):
            raise SystemExit(
                f"Not enough training sentences to fit {int(args.num_clusters)} clusters: "
                f"{len(train_sents)}"
            )
        clusterer = SemStampClusterer.train_kmeans(
            sentences=train_sents,
            embed_model=str(args.embed_model),
            num_clusters=int(args.num_clusters),
            random_state=int(args.seed),
            device=str(args.device),
        )
        clusterer.save(centroids_path)
        print(
            f"[SEMSTAMP] saved centroids -> {centroids_path} "
            f"(train_sents={len(train_sents)})"
        )

    sem_params = SemStampParams(
        embed_model=str(clusterer.embedder.model_name),
        num_clusters=int(clusterer.num_clusters),
        gamma=float(args.gamma),
    )
    gen_params = SemStampGenParams(
        rounds=int(args.rounds),
        k_batch=int(args.k_batch),
        k_stage1=int(args.k_stage1),
        k_stage2=int(args.k_stage2),
        strategy=str(args.gen_strategy),
        min_sem_sim=float(args.min_sem_sim),
        cluster_margin=float(args.cluster_margin),
        min_sent_words=max(1, int(args.gen_min_sent_words)),
        keep_if_already_green=bool(args.keep_if_already_green),
        force_accept_last_round=bool(args.force_accept_last_round),
        openai_model=str(args.openai_model),
    )

    keys_path = args.out_dir / "provider_keys.json"
    keys_meta: Dict[str, Any]
    provider_keys: List[str]
    if keys_path.exists():
        keys_meta = json.loads(keys_path.read_text(encoding="utf-8"))
        provider_keys = list(keys_meta["keys_hex"])
        if int(keys_meta.get("providers", len(provider_keys))) != int(args.providers):
            raise SystemExit(
                f"providers mismatch: keys file has {keys_meta.get('providers')} "
                f"but --providers={args.providers}"
            )
    else:
        prng = random.Random(int(args.seed))
        provider_keys = [prng.randbytes(32).hex() for _ in range(int(args.providers))]
        keys_meta = {"keys_hex": provider_keys, "providers": int(args.providers), "seed": int(args.seed)}

    sem_meta = dict(keys_meta.get("semstamp") or {})
    sem_meta.update(
        {
            "impl_variant": "ksemstamp_sequence_conditional_v3_answerseq_margin_minwords",
            "notes": (
                "k-SemStamp-style sequence-conditioned cluster validity baseline "
                "(k-means partitions, generation uses cluster-margin delta and sentence min-word filtering)"
            ),
            "embed_model": str(sem_params.embed_model),
            "num_clusters": int(sem_params.num_clusters),
            "gamma": float(sem_params.gamma),
            "centroids_path": str(centroids_path),
            "centroid_file_sha256": hashlib.sha256(centroids_path.read_bytes()).hexdigest(),
            "gen": {
                "strategy": str(gen_params.strategy),
                "rounds": int(gen_params.rounds),
                "k_batch": int(gen_params.k_batch),
                "k_stage1": int(gen_params.k_stage1),
                "k_stage2": int(gen_params.k_stage2),
                "min_sem_sim": float(gen_params.min_sem_sim),
                "cluster_margin": float(gen_params.cluster_margin),
                "min_sent_words": int(gen_params.min_sent_words),
                "keep_if_already_green": bool(gen_params.keep_if_already_green),
                "force_accept_last_round": bool(gen_params.force_accept_last_round),
                "openai_model": str(gen_params.openai_model),
            },
        }
    )
    keys_meta.update(
        {
            "keys_hex": provider_keys,
            "providers": int(args.providers),
            "seed": int(args.seed),
            "watermark_scheme": "semstamp",
            "semstamp": sem_meta,
        }
    )
    keys_path.write_text(json.dumps(keys_meta, indent=2), encoding="utf-8")

    if args.train_centroids_only:
        print("[DONE] Centroids trained/loaded and key metadata synced.")
        return

    if args.warmup_only:
        # Embed one short sentence to force model cache + a tiny centroid lookup.
        _ = clusterer.assign_texts(["Warmup sentence."])
        print("[WARMUP] SemStamp embedder+centroids ready.")
        return

    _require_openai_key()

    merged_out = args.out_dir / "corpus.jsonl"
    out_path = merged_out if num_shards <= 1 else (args.out_dir / f"corpus.shard{shard_id:02d}.jsonl")
    done_ids: Set[str] = set()
    if args.resume:
        done_ids_raw = _load_done_ids(out_path)
        if num_shards > 1:
            # If resume happens with a different num_shards than a previous run,
            # only count IDs that truly belong to this shard under current settings.
            done_ids = {
                did
                for did in done_ids_raw
                if _stable_shard(did, num_shards=num_shards, seed=int(args.seed)) == shard_id
            }
            foreign = int(len(done_ids_raw) - len(done_ids))
            if foreign > 0:
                print(
                    f"[RESUME-WARN] shard{shard_id:02d}: ignoring {foreign} pre-existing IDs "
                    f"that hash to other shards under num_shards={num_shards}."
                )
        else:
            done_ids = done_ids_raw
    quota = _shard_quota(int(args.max_docs), num_shards=num_shards, shard_id=shard_id)
    remaining = max(0, quota - len(done_ids))
    if remaining <= 0:
        print(f"[DONE] shard output already has {len(done_ids)} docs (quota={quota}): {out_path}")
        return

    mode = "a" if args.resume else "w"
    written = 0
    pbar = tqdm(total=remaining, desc=f"Build SemStamp corpus (shard{shard_id:02d})", unit="doc")
    qc_every = max(0, int(args.qc_every_docs))
    qc_roll = max(1, int(args.qc_rolling_docs))
    qc_wm_z: List[float] = []
    qc_wm_T: List[int] = []
    qc_wm_gf: List[float] = []
    qc_green_rate: List[float] = []
    qc_fail_rate: List[float] = []
    qc_sem_sim: List[float] = []
    qc_cluster_gap: List[float] = []
    with out_path.open(mode, encoding="utf-8") as f_out:
        wm_by_pid: Dict[int, SemStampWatermarker] = {}
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

            wm = wm_by_pid.get(pid)
            if wm is None:
                wm = SemStampWatermarker(
                    provider_key_hex=provider_keys[pid],
                    clusterer=clusterer,
                    params=sem_params,
                    gen_params=gen_params,
                )
                wm_by_pid[pid] = wm

            t0 = time.perf_counter()
            text_wm, gen_summary = wm.watermark(
                text_src,
                timing=bool(args.timing_log),
                timing_doc_id=did,
                timing_every=int(args.timing_every),
            )
            det = detect_semstamp_text(
                text=text_wm,
                key_hex=provider_keys[pid],
                clusterer=clusterer,
                params=sem_params,
                min_sent_words=max(1, int(args.gen_min_sent_words)),
            )
            dt = time.perf_counter() - t0
            if args.timing_log:
                print(
                    f"[SEMSTAMP-DOC] id={did[:24]} shard={shard_id:02d} total={dt:.2f}s "
                    f"T={det.T} z={det.z:.2f} changed={gen_summary.changed_sentences}/{gen_summary.n_sentences} "
                    f"min_words={int(args.gen_min_sent_words)}",
                    flush=True,
                )

            rec: Dict[str, Any] = {
                "id": did,
                "title": str(row.get("title") or ""),
                "provider_id": pid,
                "text_src": text_src,
                "text_wm": text_wm,
                "text_wm_semstamp": text_wm,
                "wm_T": int(det.T),
                "wm_greens": int(det.greens),
                "wm_z": float(det.z),
                "wm_green_fraction": float(det.green_fraction),
                "wm_latency_s": float(dt),
                "src_chars": int(len(text_src)),
                "wm_chars": int(len(text_wm)),
                "semstamp_changed_sentences": int(gen_summary.changed_sentences),
                "semstamp_n_sentences": int(gen_summary.n_sentences),
                "semstamp_already_green_sentences": int(gen_summary.already_green_sentences),
                "semstamp_green_after_sentences": int(gen_summary.green_after_sentences),
                "semstamp_failed_to_green_sentences": int(gen_summary.failed_to_green_sentences),
                "semstamp_mean_sem_sim_accepted": float(gen_summary.mean_sem_sim_accepted),
                "semstamp_mean_cluster_gap_accepted": float(gen_summary.mean_cluster_gap_accepted),
                "semstamp_min_cluster_gap_accepted": float(gen_summary.min_cluster_gap_accepted),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            done_ids.add(did)
            pbar.update(1)
            qc_wm_z.append(float(det.z))
            qc_wm_T.append(int(det.T))
            qc_wm_gf.append(float(det.green_fraction))
            n_s = max(1, int(gen_summary.n_sentences))
            qc_green_rate.append(float(gen_summary.green_after_sentences) / float(n_s))
            qc_fail_rate.append(float(gen_summary.failed_to_green_sentences) / float(n_s))
            qc_sem_sim.append(float(gen_summary.mean_sem_sim_accepted))
            qc_cluster_gap.append(float(gen_summary.mean_cluster_gap_accepted))
            if qc_every > 0 and (written % qc_every) == 0:
                sl = slice(max(0, len(qc_wm_z) - qc_roll), len(qc_wm_z))
                z_roll = qc_wm_z[sl]
                T_roll = qc_wm_T[sl]
                gf_roll = qc_wm_gf[sl]
                gr_roll = qc_green_rate[sl]
                fr_roll = qc_fail_rate[sl]
                ss_roll = qc_sem_sim[sl]
                cg_roll = qc_cluster_gap[sl]
                pos_frac_roll = (
                    sum(1 for x in z_roll if x > 0.0) / float(len(z_roll)) if z_roll else 0.0
                )
                z2_frac_roll = (
                    sum(1 for x in z_roll if x >= 2.0) / float(len(z_roll)) if z_roll else 0.0
                )
                print(
                    "[SEMSTAMP-QC] "
                    f"shard={shard_id:02d} docs={written}/{remaining} "
                    f"cum_z_mean={sum(qc_wm_z)/len(qc_wm_z):.2f} cum_pos_frac="
                    f"{(sum(1 for x in qc_wm_z if x > 0.0)/len(qc_wm_z)):.2f} "
                    f"roll{len(z_roll)}_z_mean={sum(z_roll)/len(z_roll):.2f} "
                    f"roll_pos_frac={pos_frac_roll:.2f} roll_z>=2_frac={z2_frac_roll:.2f} "
                    f"roll_T_mean={sum(T_roll)/len(T_roll):.1f} "
                    f"roll_gf_mean={sum(gf_roll)/len(gf_roll):.3f} "
                    f"roll_green_after_rate={sum(gr_roll)/len(gr_roll):.3f} "
                    f"roll_fail_rate={sum(fr_roll)/len(fr_roll):.3f} "
                    f"roll_sem_sim={sum(ss_roll)/len(ss_roll):.3f} "
                    f"roll_cluster_gap={sum(cg_roll)/len(cg_roll):.3f} "
                    f"gen_min_words={int(args.gen_min_sent_words)}",
                    flush=True,
                )
            if (written % 3) == 0:
                f_out.flush()
    pbar.close()

    if num_shards > 1:
        print(
            f"[DONE] Built shard shard{shard_id:02d}: +{written} docs -> {out_path} "
            f"(quota={quota}, resume={args.resume})"
        )
        print(
            f"[NOTE] Merge shards: python experiments/build_corpora_semstamp.py "
            f"--manifest {args.manifest} --out_dir {args.out_dir} --num_shards {num_shards} --merge_shards"
        )
    else:
        print(f"[DONE] Built/extended SemStamp corpus: +{written} docs -> {out_path}")


if __name__ == "__main__":
    main()
