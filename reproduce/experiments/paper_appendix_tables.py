#!/usr/bin/env python3
"""Print the appendix and additional-experiment tables of the paper from stored results.

Pure aggregation over the JSON outputs of the experiment scripts (no models, no
API). Every subcommand reads the files at their default ``results/...`` paths
relative to ``reproduce/`` and prints the rows of the table.

  answer-variation    tab:answer-variation  pools post_answer_laundering_attack.py
                      per-pair results over all out dirs (seeds 1-3 + 4-5)
  laundering-utility  tab:laundering-utility (laundering_utility_judge.py)
  llm-judge           app:llm-judge         (llm_judge_utility.py)
  cost-strength       tab:cost-strength     (watermark_cost_ablation.py)
  sensitivity         tab:sensitivity + app:sensitivity text (detector_sensitivity_sweep.py)
  tmin                tab:tmin-crossbenchmark (e6_tmin_crossbenchmark.py) + dense sweep
  marketplace-scaling tab:marketplace-scaling (provider_scaling_experiment.py)
  model-robustness    tab:model-robustness  (model_robustness_check.py)
  paraphrase-model    tab:paraphrase-model  (e5_compare_paraphrasers.py)
  multidomain         tab:multidomain       (cr_e1_domain_analysis.py)
  coverage            tab:coverage          (from the per-run JSONs; ``--write`` stores
                      results/coverage/coverage_full_summary.json)
  corpus-evolution    tab:corpus-evolution  (staleness_reaudit.py / cr_staleness_multiseed.py)
  all                 everything whose inputs exist

Usage: python experiments/paper_appendix_tables.py <table> [--results_dir results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, List, Optional

R = Path("results")


def _load(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _row(*cells: Any, w: int = 12) -> str:
    return "".join(f"{c!s:<{w}}" for c in cells)


def _f(x: Optional[float], nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Sec. 5.5 / App. laundering
# ---------------------------------------------------------------------------
LAUNDER_ORDER = ["original", "neutral", "neutral2", "aggressive", "compress50", "compress25"]


def answer_variation(args: argparse.Namespace) -> Dict[str, Any]:
    """Pool laundering per-pair results across out dirs (later dirs do not override earlier pairs)."""
    pooled: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for d in args.launder_dirs.split(","):
        res = _load(R / d.strip() / "laundering_results.json")
        for cond, pairs in res.items():
            for key, v in pairs.items():
                pooled.setdefault(cond, {}).setdefault(key, v)
    out: Dict[str, Any] = {}
    print(_row("Condition", "Median Z", "BH det.", "Max NT Z", "Med Q2BH", "pairs", w=12))
    for cond in [c for c in LAUNDER_ORDER if c in pooled] + sorted(set(pooled) - set(LAUNDER_ORDER)):
        vals = list(pooled[cond].values())
        z = [float(v["final_target_z"]) for v in vals]
        det = sum(bool(v["bh_detected"]) for v in vals)
        nt = max(float(v["final_nontarget_z"]) for v in vals)
        q = [int(v["queries_to_bh"]) for v in vals if v.get("queries_to_bh") is not None]
        out[cond] = {"n_pairs": len(vals), "median_target_z": round(median(z), 2),
                     "bh_detection": f"{det}/{len(vals)}", "max_nontarget_z": nt,
                     "median_queries_to_bh": median(q) if q else None}
        print(_row(cond, _f(median(z)), f"{det}/{len(vals)}", _f(nt), out[cond]["median_queries_to_bh"],
                   len(vals), w=12))
    if any(v["n_pairs"] != 30 for v in out.values()):
        n = [v["n_pairs"] for v in out.values()]
        print(f"[INFO] stored laundering results cover {min(n)}-{max(n)} pairs per condition; "
              "scripts/20_laundering.sh with the laundering_cache bundle scores all 30 (seeds 1-5 x 6 targets).")
    return out


def laundering_utility(args: argparse.Namespace) -> Dict[str, Any]:
    d = _load(R / "laundering_attack" / "utility_judge_results.json")
    print(_row("Condition", "n", "Words", "Acc.", "Compl.", "Fluency", "Pref.orig", "Pref.eq", w=11))
    out = {}
    for cond, r in d.items():
        out[cond] = {k: r[k] for k in ("n", "laundered_words_median", "accuracy_mean",
                                       "completeness_mean", "fluency_mean", "pref_orig", "pref_equal")}
        print(_row(cond, r["n"], f"{r['laundered_words_median']:.0f}", _f(r["accuracy_mean"]),
                   _f(r["completeness_mean"]), _f(r["fluency_mean"]),
                   f"{100 * r['pref_orig']:.1f}%", f"{100 * r['pref_equal']:.1f}%", w=11))
    return out


def llm_judge(args: argparse.Namespace) -> Dict[str, Any]:
    d = _load(R / args.judge_dir / "judge_results.json")
    s, a = d["sentence_summary"], d["answer_summary"]
    out = {
        "n_sentence_pairs": s["n_pairs"],
        "meaning": s["meaning_preservation"]["mean"], "naturalness": s["naturalness"]["mean"],
        "factual": s["factual_consistency_rate"], "no_omission": s["no_omission_rate"],
        "n_answer_pairs": a["n_trials"],
        "acc_watermarked": a["watermarked_scores"]["accuracy"]["mean"],
        "acc_source": a["source_scores"]["accuracy"]["mean"],
        "pref_watermarked": a["preference_rate_watermarked"],
        "pref_source": a["preference_rate_source"],
        "pref_equivalent": a["preference_rate_equivalent"],
    }
    for k, v in out.items():
        print(f"{k:<20} {v if isinstance(v, int) else round(v, 3)}")
    return out


# ---------------------------------------------------------------------------
# Cost / sensitivity / T_min
# ---------------------------------------------------------------------------
def cost_strength(args: argparse.Namespace) -> Dict[str, Any]:
    d = _load(R / "watermark_cost_ablation" / "ablation_results.json")
    print(f"n_docs={d['n_docs']} n_sentences={d['n_sentences']} z_anchor={d['z_anchor']}")
    print(_row("K", "rho", "Gain %", "Target Z", "Sem.sim", "Rel.cost", w=10))
    out = {}
    for K in (24, 12, 6):
        for rho in (6, 3, 1):
            r = d["grid"][f"K{K}_rho{rho}"]
            out[f"K{K}_rho{rho}"] = r
            print(_row(K, rho, f"{100 * r['gain_ratio']:.1f}", _f(r["z_estimated"]),
                       _f(r["sem_sim_mean"], 3), f"{r['rel_api_calls']:.2f}x", w=10))
    return out


def _sens_key(W: int, ts: float, tm: float, tmin: int) -> str:
    return f"W{W}_ts{ts:.2f}_tm{tm:.2f}_Tmin{tmin}"


def sensitivity(args: argparse.Namespace) -> Dict[str, Any]:
    base = _load(R / "sensitivity_sweep" / "sensitivity_results.json")
    ext = _load(R / "sensitivity_sweep" / "sensitivity_extended.json")
    grid = {**base, **ext}
    n_combos = len(ext)
    stable = lambda r: r["detection_rate"] == 1.0 and r["fpr"] == 0.0  # noqa: E731

    def show(label: str, keys: List[str]) -> None:
        print(f"-- {label}")
        print(f"{'combo':<28}" + _row("det", "FPR", "med Z", "med Q2BH", w=10))
        for k in keys:
            r = grid[k]
            print(f"{k:<28}" + _row(_f(r["detection_rate"]), _f(r["fpr"]),
                                    _f(r["target_z_median"]), r.get("median_q2bh"), w=10))

    print(f"extended grid: {n_combos} combos; base grid: {len(base)} combos")
    show("W (tau_sim .82, tau_margin .05, T_min 25)", [_sens_key(W, .82, .05, 25) for W in (20, 28, 40)])
    ts_vals = sorted({r["tau_sim"] for r in ext.values()})
    tm_vals = sorted({r["tau_margin"] for r in ext.values()})
    show("tau_sim (tau_margin .05, T_min 25)", [_sens_key(28, t, .05, 25) for t in ts_vals])
    show("tau_sim (tau_margin 0, T_min 25)", [_sens_key(28, t, 0.0, 25) for t in ts_vals])
    show("tau_margin (tau_sim .82, T_min 25)", [_sens_key(28, .82, m, 25) for m in tm_vals])
    show("T_min (tau_sim .82, tau_margin .05)", [_sens_key(28, .82, .05, t) for t in (10, 25, 50)])
    stable_ts = [t for t in ts_vals if stable(grid[_sens_key(28, t, .05, 25)])]
    stable_tm = [m for m in tm_vals if stable(grid[_sens_key(28, .82, m, 25)])]
    out = {"n_extended_combos": n_combos,
           "tau_sim_stable": [min(stable_ts), max(stable_ts)] if stable_ts else None,
           "tau_margin_stable": [min(stable_tm), max(stable_tm)] if stable_tm else None}
    print(f"stable tau_sim range (tm .05, T_min 25): {out['tau_sim_stable']}")
    print(f"stable tau_margin range (ts .82, T_min 25): {out['tau_margin_stable']}")
    return out


def tmin(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    e6 = R / "e6_tmin" / "summary_cameraready.json"
    if e6.exists():
        d = _load(e6)
        for bench, rows in d["benchmarks"].items():
            print(f"-- E6 {bench} ({next(iter(rows.values()))['n_runs']} audits)")
            print(_row("T_min", "det", "non-target", "wrong-key", w=12))
            for t, r in rows.items():
                print(_row(t, r["target_detection_rate"], r["any_nontarget_rate"],
                           r["wrongkey_any_sig_rate"], w=12))
        out["e6"] = d["benchmarks"]
    dense = R / "sensitivity_sweep" / "tmin_sweep_dense.json"
    if dense.exists():
        d = _load(dense)
        print("-- dense T_min sweep (sensitivity re-implementation, seeds 1-3, 18 audits)")
        print(_row("T_min", "det", "FPR", "med Z", "med Q2BH", w=10))
        for t, r in d.items():
            print(_row(t, _f(r["detection_rate"]), _f(r["fpr"]), _f(r["target_z_median"]),
                       r.get("median_q2bh"), w=10))
        out["dense"] = d
    if not out:
        raise FileNotFoundError(e6)
    return out


# ---------------------------------------------------------------------------
# Re-watermarked mini-corpora
# ---------------------------------------------------------------------------
def marketplace_scaling(args: argparse.Namespace) -> Dict[str, Any]:
    out = {}
    print(_row("m", "T_min", "Med Z", "BH det.", "Max NT Z", "Q to BH", "targets", "trials", w=10))
    for p in sorted((R / "provider_scaling").glob("m*/summary*.json")):
        r = _load(p)
        n_t = r.get("n_targets")
        det = round(r["detection_rate"] * n_t) if n_t else None
        out[str(p.relative_to(R))] = r
        print(_row(r["m"], r["T_min"], _f(r["median_target_z"]), f"{det}/{n_t}",
                   _f(r.get("max_nontarget_z")), r.get("median_q2bh"), n_t, r["n_trials"], w=10))
    if not out:
        raise FileNotFoundError(R / "provider_scaling")
    print("[NOTE] more targets and audit seeds per m: scripts/26_provider_scaling.sh (RUNS, AUDIT_SEEDS).")
    return out


def model_robustness(args: argparse.Namespace) -> Dict[str, Any]:
    files = [("MiniLM-L6-v2", R / "model_robustness/part_a_minilm/summary_a.json"),
             ("GTR-T5-base", R / "model_robustness_gtr/summary_gtr.json"),
             ("GPT-3.5-turbo", R / "model_robustness/part_b_gpt4o/summary_b.json"),
             ("Qwen2.5-1.5B", R / "model_robustness/part_c_qwen/summary_c.json")]
    out = {}
    print(_row("Model", "d", "rho", "Det.", "Z", "FPR", "trials", w=14))
    for name, p in files:
        if not p.exists():
            continue
        r = _load(p)
        n_t = r["n_targets"]
        out[name] = r
        print(_row(name, r["D"], r.get("rho", 6), f"{round(r['detection_rate'] * n_t)}/{n_t}",
                   _f(r["median_target_z"]), _f(r["fpr"]), r["n_trials"], w=14))
    if not out:
        raise FileNotFoundError(files[0][1])
    print("[NOTE] the 'mpnet (primary)' row is the laundering 'original' condition on seeds 1-3 "
          "(answer-variation).")
    return out


def paraphrase_model(args: argparse.Namespace) -> Dict[str, Any]:
    d = _load(R / "e5_paraphraser" / "compare.json")
    print(_row("Paraphraser", "Dir.gain", "Sem.sim", "Changed", "n_docs", w=12))
    for tag in ("gpt4omini", "llama8b"):
        if tag in d:
            r = d[tag]
            print(_row(tag, f"{r['mean_gain']:.4f}", f"{r['mean_doc_semsim']:.3f}",
                       _f(r["mean_changed_frac"]), r["n_docs"], w=12))
    for k, v in d.get("relative_gain", {}).items():
        print(f"relative gain {k}: {100 * v:.1f}%")
    return d


# ---------------------------------------------------------------------------
# E1-E3
# ---------------------------------------------------------------------------
def multidomain(args: argparse.Namespace) -> Dict[str, Any]:
    d = _load(R / "multidomain_full" / "e1_domain_analysis.json")
    bh = d.get("per_domain_bh_detection", {})
    if not bh or "median_target_Z_all" not in d:  # older analysis file: recompute from run JSONs
        doms = {0: "clinical", 1: "clinical", 2: "cti", 3: "cti", 4: "legal", 5: "legal"}
        z_all, det = [], {}
        for f in sorted((R / "multidomain_full" / "dirbucket_naive").glob("rag_multi_t*_seed*.json")):
            run = json.loads(f.read_text())
            t = int(run["target_provider"])
            z_all.append(float(run["z_raw"][t]))
            det.setdefault(doms[t], []).append(bool((run.get("bh_sig") or [False] * 6)[t]))
        bh = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(det.items())}
        d["median_target_Z_all"] = round(median(z_all), 2) if z_all else None
    print(_row("Domain", "Median Z", "BH det.", w=12))
    for dom, z in d["per_domain_median_target_Z"].items():
        print(_row(dom, _f(z), bh.get(dom, "-"), w=12))
    print(_row("all", _f(d.get("median_target_Z_all")), d["bh_detection"], w=12))
    sib, cross = d["within_domain_sibling_nontarget_gated_Z"], d["cross_domain_nontarget_gated_Z"]
    print(f"sibling gated Z max {sib['max']} (n={sib['n']}); cross-domain max {cross['max']} "
          f"(n={cross['n']}); FPR {d['fpr_runs_with_any_nontarget_bh']}")
    return d


COVERAGE_LEVELS = [("c25", 0.30), ("c50", 0.53), ("c75", 0.78)]


def coverage_rows(results_dir: Path) -> List[Dict[str, Any]]:
    """Cross-coverage table from the per-run JSONs (the paper aggregation).

    ``realized_wm_frac`` is the realized watermarked-sentence fraction reported by
    build_coverage_corpus.py (0.299 / 0.534 / 0.783), rounded as in the paper.
    """
    rows = []
    for cov, wm in COVERAGE_LEVELS:
        files = sorted((results_dir / "coverage" / cov / "dirbucket_naive").glob("rag_multi_t*_seed*.json"))
        if not files:
            raise FileNotFoundError(results_dir / "coverage" / cov / "dirbucket_naive")
        zt: List[float] = []
        det = fpr = n = 0
        q: List[int] = []
        for f in files:
            d = json.loads(f.read_text())
            t = int(d["target_provider"])
            z = d["z_raw"]
            bh = d.get("bh_sig") or []
            zt.append(float(z[t]))
            n += 1
            det += bool(bh[t]) if bh else 0
            if bh and any(bh[p] for p in range(len(z)) if p != t):
                fpr += 1
            # crossed_at is the runner's first query with T >= T_min and Z >= z_threshold (4.0),
            # not the BH decision. The JSON keys below keep the names of the original
            # coverage_full_summary.json ("..._bh") for compatibility; they count Z>=4 crossings.
            ca = d.get("crossed_at")
            if isinstance(ca, list) and ca[t] is not None:
                q.append(int(ca[t]))
        rows.append({"coverage": cov, "realized_wm_frac": wm, "n_runs": n,
                     "bh_detection": f"{det}/{n}", "median_target_Z": round(median(zt), 2),
                     "fpr_runs": f"{fpr}/{n}",
                     "median_queries_to_bh": (median(q) if q else None),
                     "n_reached_bh": len(q)})
    return rows


def coverage(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = coverage_rows(R)
    # "Q to Z>=4" / "reached Z>=4" are the JSON keys median_queries_to_bh / n_reached_bh
    # (first query with Z >= 4 and T >= T_min, see coverage_rows).
    print(_row("Coverage", "BH det.", "Median Z", "FPR", "Q to Z>=4", "reached Z>=4", w=12))
    for r in sorted(rows, key=lambda r: -r["realized_wm_frac"]):
        print(_row(r["realized_wm_frac"], r["bh_detection"], _f(r["median_target_Z"]),
                   r["fpr_runs"], r["median_queries_to_bh"], r["n_reached_bh"], w=12))
    if args.write:
        out = R / "coverage" / "coverage_full_summary.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"[WRITE] {out}")
    return rows


def corpus_evolution(args: argparse.Namespace) -> Dict[str, Any]:
    d = _load(R / "staleness_full" / "staleness_full_summary.json")
    print(_row("f", "Med Z", "Min Z", "BH det.", "Max NT Z", "NT BH", "Q2D", w=10))
    for tag, r in d["conditions"].items():
        print(_row(tag, _f(r["median_target_z"]), _f(r["min_target_z"]), r["bh_detection"],
                   _f(r["max_nontarget_gated_z"]), r["any_nontarget_bh"],
                   r["median_queries_to_detection"], w=10))
    return d


TABLES: Dict[str, Callable[[argparse.Namespace], Any]] = {
    "answer-variation": answer_variation,
    "laundering-utility": laundering_utility,
    "llm-judge": llm_judge,
    "cost-strength": cost_strength,
    "sensitivity": sensitivity,
    "tmin": tmin,
    "marketplace-scaling": marketplace_scaling,
    "model-robustness": model_robustness,
    "paraphrase-model": paraphrase_model,
    "multidomain": multidomain,
    "coverage": coverage,
    "corpus-evolution": corpus_evolution,
}


def main() -> None:
    global R
    ap = argparse.ArgumentParser(description="Print appendix tables from stored results.")
    ap.add_argument("table", choices=[*TABLES, "all"])
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--launder_dirs", type=str, default="laundering_attack,laundering_attack_seeds45",
                    help="answer-variation: result dirs (under --results_dir) to pool.")
    ap.add_argument("--judge_dir", type=str, default="llm_judge_gpt4o",
                    help="llm-judge: result dir (paper: llm_judge_gpt4o).")
    ap.add_argument("--write", action="store_true",
                    help="coverage: also write results/coverage/coverage_full_summary.json.")
    ap.add_argument("--json_out", type=Path, default=None, help="Also dump the table data as JSON.")
    args = ap.parse_args()
    R = args.results_dir

    names = list(TABLES) if args.table == "all" else [args.table]
    dump: Dict[str, Any] = {}
    for name in names:
        print(f"\n=== {name} ===")
        try:
            dump[name] = TABLES[name](args)
        except FileNotFoundError as e:
            if args.table != "all":
                raise SystemExit(f"[FATAL] missing input: {e}")
            print(f"[SKIP] missing input: {e}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(dump, indent=2, default=str))


if __name__ == "__main__":
    main()
