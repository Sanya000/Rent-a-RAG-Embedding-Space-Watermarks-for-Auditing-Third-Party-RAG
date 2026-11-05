"""E3 staleness re-audit (paper App. app:corpus-evolution, Table tab:corpus-evolution).

Replays the primary run's logged answers (generated from the cached deployed
corpus v1) through the exact primary audit pipeline, with the auditor's
reference corpus replaced by an evolved corpus v2 in which a fraction f of the
documents was rewritten without watermark (``build_evolved_corpus.py``).
No API calls; CPU-bound (about 1 h per condition and seed on CPU).

Fidelity: uses the primary harness functions (build_alignment_sources,
MultiDirBucketDetector.detect_windows_multi_aligned, pooled_z,
gate_significance_by_min_pooled_T) with the paper's primary parameters. For
f=0 the replay is checked against the cumulative statistics stored in the
trial logs (``--validate_f0``).

Two output layouts:
  * single seed (``--seed_tag 01``, pilot):  <out_dir>/reaudit_<tag>.json and
    <out_dir>/staleness_summary.json
  * pooled seeds (``--seeds 01,02,03,04,05``, paper table):
    <out_dir>/reaudit_<tag>_allseeds.json and <out_dir>/staleness_full_summary.json.
    Finished (condition, seed) replays are kept in reaudit_<tag>_allseeds.json and
    skipped on restart, provided they were made with the same work_dir, --max_trials and
    --trial_logs_dir (stored under "_params"); otherwise they are recomputed.
``--merge`` combines <out_dir>/<tag>/staleness_full_summary.json files written by
per-condition jobs into <out_dir>/staleness_full_summary.json.

Paper table (one job per condition, then merge):
  python experiments/staleness_reaudit.py --conditions f10:work/staleness/f10 \
      --seeds 01,02,03,04,05 --max_trials 80 --out_dir results/staleness_full/f10
  python experiments/staleness_reaudit.py --merge --out_dir results/staleness_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dirbucket_quick_experiments import (  # noqa: E402
    build_alignment_sources,
    ensure_embeddings,
    gate_significance_by_min_pooled_T,
    load_workdir,
    pooled_z,
    read_trial_log,
)

# Primary-setting audit parameters (paper §5.1 / paper_rag_multi_config.json).
PARAMS = dict(
    retrieval_model="all-MiniLM-L6-v2",
    sim_threshold=0.82,
    tie_margin=0.05,
    win_words=28,
    stride_words=14,
    global_source_query_sim_threshold=0.48,
    global_source_max_docs=120,
    global_source_max_docs_per_provider=0,
    global_source_selector="query",
    z_threshold=4.0,
    min_pooled_T=25,
    alpha=0.05,
    bucket_bits=96,
    bucket_seed=2025,
    embed_model="sentence-transformers/all-mpnet-base-v2",
    direction_scheme="naive",
)


def replay_condition(work_dir: Path, trial_logs_dir: Path, targets: List[int],
                     seed_tag: str, max_trials: int, validate: bool) -> Dict[str, Any]:
    from watermark_dirbucket import MultiDirBucketDetector

    rows, keys, source_field, _, _, _, _ = load_workdir(work_dir)
    P = len(keys)
    if source_field != "text_wm":
        raise RuntimeError(f"Expected legacy corpus (text_wm); got source_field={source_field}")
    X, enc = ensure_embeddings(work_dir, rows, field=source_field,
                               model_name=PARAMS["retrieval_model"])
    det = MultiDirBucketDetector(
        provider_keys_by_pid={i: keys[i] for i in range(P)},
        bucket_bits=PARAMS["bucket_bits"],
        bucket_seed=PARAMS["bucket_seed"],
        embed_model=PARAMS["embed_model"],
        direction_scheme=PARAMS["direction_scheme"],
    )

    out: Dict[str, Any] = {"work_dir": str(work_dir), "targets": {}}
    for target in targets:
        log_path = trial_logs_dir / f"rag_multi_t{target}_seed{seed_tag}.trials.jsonl"
        trials = read_trial_log(log_path, scheme="dirbucket_naive", audit_mode="aligned")
        if not trials:
            raise FileNotFoundError(f"No trials in {log_path}")
        trials = trials[:max_trials]

        pooled_T = np.zeros(P, dtype=np.int64)
        pooled_sumproj = np.zeros(P, dtype=np.float64)
        crossed_at = None
        assign_total = 0
        windows_total = 0

        for q_eff, tr in enumerate(trials, 1):
            question = str(tr["question"])
            answer = str(tr["answer"])
            query_vec = enc.encode([question], normalize_embeddings=True,
                                   convert_to_numpy=True)[0]
            sources, _diag = build_alignment_sources(
                rows=rows, picked_idxs=[], prompt_docs=[], retrieved_docs=[],
                source_field=source_field, aligned_source_mode="global_corpus",
                query_vec=query_vec, X=X,
                global_source_query_sim_threshold=PARAMS["global_source_query_sim_threshold"],
                global_source_max_docs=PARAMS["global_source_max_docs"],
                global_source_max_docs_per_provider=PARAMS["global_source_max_docs_per_provider"],
                global_source_selector=PARAMS["global_source_selector"],
            )
            res_by_pid, meta = det.detect_windows_multi_aligned(
                text=answer, sources=sources,
                sim_threshold=PARAMS["sim_threshold"], competitive=True,
                tie_margin=PARAMS["tie_margin"], neighbor_smoothing=False,
                ham_radius=None, smooth_agg="max",
                win_words=PARAMS["win_words"], stride_words=PARAMS["stride_words"],
            )
            assign_total += int(meta.get("assigned_total", 0) or 0)
            windows_total += int(meta.get("windows_total", 0) or 0)
            for pid in range(P):
                s = res_by_pid.get(pid)
                if s is None:
                    continue
                pooled_T[pid] += int(s.T)
                pooled_sumproj[pid] += float(s.mean_proj) * int(s.T)

            z_now = pooled_z(det, pooled_T, pooled_sumproj)
            if (crossed_at is None and pooled_T[target] >= PARAMS["min_pooled_T"]
                    and z_now[target] >= PARAMS["z_threshold"]):
                crossed_at = q_eff

        z_raw = pooled_z(det, pooled_T, pooled_sumproj)
        z_gated, pvals, qvals, bh_sig, eligible = gate_significance_by_min_pooled_T(
            z_raw, pooled_T, min_pooled_T=PARAMS["min_pooled_T"], alpha=PARAMS["alpha"])
        nontarget_z = [z_gated[p] for p in range(P) if p != target]
        nontarget_sig = [bool(bh_sig[p]) for p in range(P) if p != target]

        rec: Dict[str, Any] = {
            "n_trials": len(trials),
            "pooled_T": pooled_T.tolist(),
            "z_raw": [float(z) for z in z_raw],
            "z_gated": [float(z) for z in z_gated],
            "bh_sig": [bool(b) for b in bh_sig],
            "target_z": float(z_raw[target]),
            "target_T": int(pooled_T[target]),
            "target_bh_detected": bool(bh_sig[target]),
            "queries_to_detection": crossed_at,
            "max_nontarget_z_gated": float(max(nontarget_z)) if nontarget_z else None,
            "any_nontarget_bh_sig": bool(any(nontarget_sig)),
            "assign_rate": (assign_total / windows_total) if windows_total else 0.0,
        }

        if validate:
            cum = (trials[-1].get("cumulative") or {})
            ref_T = cum.get("pooled_T")
            ref_S = cum.get("pooled_sumproj")
            if isinstance(ref_T, list) and isinstance(ref_S, list):
                dT = int(np.abs(pooled_T - np.array(ref_T, dtype=np.int64)).max())
                dS = float(np.abs(pooled_sumproj - np.array(ref_S, dtype=np.float64)).max())
                rec["validation"] = {"max_abs_dT": dT, "max_abs_dS": dS,
                                     "ok": bool(dT == 0 and dS < 1e-3)}
                print(f"    [validate] t{target}: dT={dT} dS={dS:.2e} "
                      f"{'OK' if rec['validation']['ok'] else 'MISMATCH'}")

        out["targets"][str(target)] = rec
        print(f"  [t{target}] Z={rec['target_z']:.2f} T={rec['target_T']} "
              f"BH={rec['target_bh_detected']} q2d={rec['queries_to_detection']} "
              f"maxNT={rec['max_nontarget_z_gated']:.2f}")
    return out


def _seed_summary(res: Dict[str, Any], targets: List[int]) -> Dict[str, Any]:
    tz = [res["targets"][str(t)]["target_z"] for t in targets]
    det_n = sum(res["targets"][str(t)]["target_bh_detected"] for t in targets)
    nt = max(res["targets"][str(t)]["max_nontarget_z_gated"] for t in targets)
    ntsig = any(res["targets"][str(t)]["any_nontarget_bh_sig"] for t in targets)
    return {
        "median_target_z": float(np.median(tz)),
        "min_target_z": float(min(tz)),
        "bh_detected": f"{det_n}/{len(targets)}",
        "max_nontarget_z_gated": float(nt),
        "any_nontarget_bh_sig": bool(ntsig),
        "queries_to_detection": [res["targets"][str(t)]["queries_to_detection"]
                                 for t in targets],
    }


def run_single_seed(args: argparse.Namespace, targets: List[int],
                    conditions: List[List[str]]) -> None:
    """Single-seed (pilot) layout: one seed, one reaudit file per condition."""
    summary: Dict[str, Any] = {"params": PARAMS, "seed_tag": args.seed_tag,
                               "max_trials": args.max_trials, "conditions": {}}
    for tag, wd in conditions:
        out_path = args.out_dir / f"reaudit_{tag}.json"
        if out_path.exists():
            print(f"[condition {tag}] SKIP (exists: {out_path})")
            res = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            print(f"[condition {tag}] work_dir={wd}")
            res = replay_condition(Path(wd), args.trial_logs_dir, targets,
                                   args.seed_tag, args.max_trials,
                                   validate=(args.validate_f0 and tag in {"f0", "f00"}))
            out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        summary["conditions"][tag] = _seed_summary(res, targets)
        c = summary["conditions"][tag]
        print(f"[{tag}] median Z={c['median_target_z']:.2f} "
              f"BH={c['bh_detected']} maxNT={c['max_nontarget_z_gated']:.2f}")

    (args.out_dir / "staleness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] {args.out_dir}/staleness_summary.json")


# Run parameters of the shipped paper replays (results/staleness_full/f*/reaudit_*_allseeds.json),
# which were written before the parameters were stored with each seed record.
_PAPER_REPLAY_PARAMS = {
    "max_trials": 80,
    "trial_logs_dir": str(Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive")),
}


def _run_params(work_dir: str, trial_logs_dir: Path, max_trials: int) -> Dict[str, Any]:
    return {"work_dir": str(Path(work_dir)), "max_trials": int(max_trials),
            "trial_logs_dir": str(Path(trial_logs_dir))}


def _cached_params(res: Dict[str, Any]) -> Dict[str, Any]:
    if "_params" in res:
        return res["_params"]
    # Legacy record: only work_dir was stored; it was produced with the paper settings.
    return {"work_dir": str(Path(res.get("work_dir", ""))), **_PAPER_REPLAY_PARAMS}


def run_multiseed(conditions: List[List[str]], trial_logs_dir: Path, targets: List[int],
                  seed_tags: List[str], max_trials: int, out_dir: Path) -> Dict[str, Any]:
    """Paper layout: pool each condition over ``seed_tags`` (resumable per seed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"seeds": seed_tags, "max_trials": max_trials,
                               "conditions": {}}
    for tag, wd in conditions:
        target_z: List[float] = []
        det = fpr = n = 0
        max_nt = 0.0
        q2d: List[int] = []
        per_seed_cache = out_dir / f"reaudit_{tag}_allseeds.json"
        allseed_records: Dict[str, Any] = (
            json.loads(per_seed_cache.read_text(encoding="utf-8"))
            if per_seed_cache.exists() else {}
        )
        expected = _run_params(wd, trial_logs_dir, max_trials)
        for seed_tag in seed_tags:
            res = allseed_records.get(seed_tag)
            reuse = res is not None and all(str(t) in res.get("targets", {}) for t in targets)
            if reuse and _cached_params(res) != expected:
                print(f"[{tag} seed {seed_tag}] cached record in {per_seed_cache} was made with "
                      f"{_cached_params(res)}, not {expected}: recomputing")
                reuse = False
            if reuse:
                print(f"[{tag} seed {seed_tag}] SKIP (cached in {per_seed_cache})")
            else:
                res = replay_condition(Path(wd), trial_logs_dir, targets,
                                       seed_tag, max_trials,
                                       validate=(tag in {"f0", "f00"} and seed_tag == "01"))
                res["_params"] = expected
                allseed_records[seed_tag] = res
                per_seed_cache.write_text(json.dumps(allseed_records, indent=1),
                                          encoding="utf-8")
            for t in targets:
                r = res["targets"][str(t)]
                target_z.append(r["target_z"])
                det += bool(r["target_bh_detected"])
                fpr += bool(r["any_nontarget_bh_sig"])
                max_nt = max(max_nt, r["max_nontarget_z_gated"] or 0.0)
                n += 1
                if r["queries_to_detection"] is not None:
                    q2d.append(int(r["queries_to_detection"]))
        summary["conditions"][tag] = {
            "n_target_runs": n,
            "median_target_z": round(median(target_z), 2),
            "min_target_z": round(min(target_z), 2),
            "bh_detection": f"{det}/{n}",
            "any_nontarget_bh": f"{fpr}/{n}",
            "max_nontarget_gated_z": round(max_nt, 3),
            "median_queries_to_detection": (median(q2d) if q2d else None),
        }
        c = summary["conditions"][tag]
        print(f"[{tag}] medZ={c['median_target_z']} minZ={c['min_target_z']} "
              f"BH={c['bh_detection']} maxNT={c['max_nontarget_gated_z']} "
              f"q2d={c['median_queries_to_detection']}")

    (out_dir / "staleness_full_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] {out_dir}/staleness_full_summary.json")
    return summary


def merge_condition_summaries(out_dir: Path, order: List[str]) -> Dict[str, Any]:
    """Merge <out_dir>/<tag>/staleness_full_summary.json into one summary."""
    merged: Dict[str, Any] = {"seeds": None, "max_trials": None, "conditions": {}}
    for tag in order:
        p = out_dir / tag / "staleness_full_summary.json"
        if not p.exists():
            raise SystemExit(f"[FATAL] missing {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        merged["seeds"] = merged["seeds"] or d.get("seeds")
        merged["max_trials"] = merged["max_trials"] or d.get("max_trials")
        merged["conditions"].update(d["conditions"])
    (out_dir / "staleness_full_summary.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8")
    for tag in order:
        c = merged["conditions"][tag]
        print(f"[{tag}] medZ={c['median_target_z']} minZ={c['min_target_z']} "
              f"BH={c['bh_detection']} maxNT={c['max_nontarget_gated_z']} "
              f"q2d={c['median_queries_to_detection']}")
    print(f"[DONE] {out_dir}/staleness_full_summary.json")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 staleness re-audit (replay, no API).")
    ap.add_argument("--conditions", type=str,
                    default="f00:work/farad_dirbucket_naive,f10:work/staleness/f10,"
                            "f25:work/staleness/f25,f50:work/staleness/f50",
                    help="Comma-separated tag:work_dir pairs.")
    ap.add_argument("--trial_logs_dir", type=Path,
                    default=Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"))
    ap.add_argument("--targets", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--seed_tag", type=str, default="01",
                    help="Single-seed (pilot) mode; ignored when --seeds is given.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seed tags (e.g. 01,02,03,04,05): pooled mode.")
    ap.add_argument("--max_trials", type=int, default=80)
    ap.add_argument("--out_dir", type=Path, default=Path("results/staleness"))
    ap.add_argument("--validate_f0", action="store_true",
                    help="Single-seed mode: compare the f00 replay with the trial-log "
                         "cumulative stats (pooled mode always validates f00 seed 01).")
    ap.add_argument("--merge", action="store_true",
                    help="Only merge <out_dir>/<tag>/staleness_full_summary.json for the "
                         "condition tags in --conditions.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [int(t) for t in args.targets.split(",")]
    conditions = [c.split(":", 1) for c in args.conditions.split(",")]

    if args.merge:
        merge_condition_summaries(args.out_dir, [tag for tag, _ in conditions])
    elif args.seeds:
        seed_tags = [s.strip() for s in args.seeds.split(",") if s.strip()]
        run_multiseed(conditions, args.trial_logs_dir, targets, seed_tags,
                      args.max_trials, args.out_dir)
    else:
        run_single_seed(args, targets, conditions)


if __name__ == "__main__":
    main()
