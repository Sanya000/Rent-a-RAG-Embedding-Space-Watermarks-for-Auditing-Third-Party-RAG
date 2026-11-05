"""Plot the naive-vs-ortho direction ablation (results/direction_compare -> results/figs/fig_compare_*)."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def mean_ci(x: List[float]) -> Tuple[float, float]:
    """Return mean and 95% CI half-width (normal approx)."""
    if not x:
        return (float("nan"), float("nan"))
    a = np.array(x, dtype=float)
    m = float(np.nanmean(a))
    if len(a) <= 1:
        return (m, 0.0)
    s = float(np.nanstd(a, ddof=1))
    hw = 1.96 * s / math.sqrt(len(a))
    return (m, hw)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_files(results_root: Path, scheme: str, pattern: str) -> List[Path]:
    d = results_root / scheme
    if not d.exists():
        return []
    return sorted(d.glob(pattern))

def plot_rag_power(results_root: Path, out_dir: Path):
    """
    Uses rag_multi_* files and their 'traj' to plot pooled z vs #queries for target provider,
    with mean ± CI band for naive vs ortho.
    """
    figpath = out_dir / "fig_compare_rag_power.png"

    schemes = ["naive", "ortho"]
    series: Dict[str, Dict[int, List[float]]] = {s: {} for s in schemes}

    for s in schemes:
        files = find_files(results_root, s, "rag_multi_seed*.json")
        for fp in files:
            r = load_json(fp)
            tp = int(r.get("target_provider", 0))
            traj = r.get("traj") or []
            for snap in traj:
                q = int(snap["q"])
                z = snap["z"][tp]
                series[s].setdefault(q, []).append(float(z))

    max_q = 0
    for s in schemes:
        if series[s]:
            max_q = max(max_q, max(series[s].keys()))

    if max_q == 0:
        print("[WARN] No rag trajectories found (did you run rag_multi with --log_traj?)")
        return

    xs = np.arange(1, max_q + 1)

    plt.figure(figsize=(7.2, 4.2))
    for s in schemes:
        means = []
        lows = []
        highs = []
        for q in xs:
            m, hw = mean_ci(series[s].get(int(q), []))
            means.append(m)
            lows.append(m - hw)
            highs.append(m + hw)
        means = np.array(means, dtype=float)
        lows = np.array(lows, dtype=float)
        highs = np.array(highs, dtype=float)

        plt.plot(xs, means, marker=None, linewidth=2.2, label=s)
        plt.fill_between(xs, lows, highs, alpha=0.18)

    plt.axhline(4.0, linestyle="--", linewidth=1.5)
    plt.xlabel("# audited answers")
    plt.ylabel("pooled z (target provider)")
    plt.title("RAG power curve: naive vs orthogonal directions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpath, dpi=220)
    plt.close()
    print(f"[OK] {figpath}")

def plot_style_sweep(results_root: Path, out_dir: Path):
    """
    Reads style_sweep_* files, aggregates target pooled z + alignment assign_rate per style.
    """
    figpath = out_dir / "fig_compare_style_sweep.png"
    schemes = ["naive", "ortho"]
    styles = ["qa", "ownwords", "bullets", "eli5"]

    # style -> list of values
    z_by: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}
    a_by: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}

    for s in schemes:
        files = find_files(results_root, s, "style_sweep_seed*.json")
        for fp in files:
            obj = load_json(fp)
            runs = obj.get("runs", [])
            for r in runs:
                st = str(r.get("answer_style"))
                if st not in styles:
                    continue
                tp = int(r.get("target_provider", 0))
                z = float(r["z"][tp])
                ar = float((r.get("align", {}) or {}).get("assign_rate") or 0.0)
                z_by[s][st].append(z)
                a_by[s][st].append(ar)

    x = np.arange(len(styles))
    width = 0.36

    # two-panel plot: top z, bottom alignment
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # --- Top: pooled z ---
    for i, s in enumerate(schemes):
        means = []
        errs = []
        for st in styles:
            m, hw = mean_ci(z_by[s][st])
            means.append(m)
            errs.append(hw)
        offs = (i - 0.5) * width
        axes[0].bar(x + offs, means, width=width, yerr=errs, capsize=3, edgecolor="black", linewidth=0.6, label=s)

    axes[0].axhline(4.0, linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("pooled z (target)")
    axes[0].set_title("Style robustness: naive vs orthogonal directions")
    axes[0].legend()

    # --- Bottom: alignment assign_rate ---
    for i, s in enumerate(schemes):
        means = []
        errs = []
        for st in styles:
            m, hw = mean_ci(a_by[s][st])
            means.append(m)
            errs.append(hw)
        offs = (i - 0.5) * width
        axes[1].bar(x + offs, means, width=width, yerr=errs, capsize=3, edgecolor="black", linewidth=0.6, label=s)

    axes[1].set_ylabel("align assign_rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(styles)
    axes[1].set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(figpath, dpi=220)
    plt.close()
    print(f"[OK] {figpath}")

def plot_sanity_provider_bars(results_root: Path, out_dir: Path):
    """
    For each provider p:
      - cooperative z_true: z[p] from sanity_coop_t{p}_*
      - wrongkey z: wrongkey.z[p] from same files
      - unrelated-null z when p is NOT the target: from sanity_unrel_t{t}_* for t != p, take z[p]

    Produces a 2-row figure:
      - Top: cooperative z_true (big)
      - Bottom: wrong-key and unrelated-null (near 0) for visibility
    """
    figpath = out_dir / "fig_compare_sanity_pooled_z.png"
    schemes = ["naive", "ortho"]

    # infer provider count from available sanity files
    P = 0
    for s in schemes:
        files = find_files(results_root, s, "sanity_coop_t*_seed*.json")
        for fp in files:
            r = load_json(fp)
            z = r.get("z") or []
            if len(z) > P:
                P = len(z)
    if P <= 0:
        print("[WARN] No sanity files found.")
        return

    # scheme -> provider -> list
    coop_true: Dict[str, Dict[int, List[float]]] = {s: {p: [] for p in range(P)} for s in schemes}
    coop_wrong: Dict[str, Dict[int, List[float]]] = {s: {p: [] for p in range(P)} for s in schemes}
    unrel_notarget: Dict[str, Dict[int, List[float]]] = {s: {p: [] for p in range(P)} for s in schemes}

    # cooperative
    for s in schemes:
        files = find_files(results_root, s, "sanity_coop_t*_seed*.json")
        for fp in files:
            m = re.search(r"sanity_coop_t(\d+)_seed", fp.name)
            if not m:
                continue
            t = int(m.group(1))
            r = load_json(fp)
            z = r["z"]
            coop_true[s][t].append(float(z[t]))
            wz = (r.get("wrongkey", {}) or {}).get("z")
            if wz is not None:
                coop_wrong[s][t].append(float(wz[t]))

    # unrelated-null: take z[p] for runs where target != p
    for s in schemes:
        files = find_files(results_root, s, "sanity_unrel_t*_seed*.json")
        for fp in files:
            m = re.search(r"sanity_unrel_t(\d+)_seed", fp.name)
            if not m:
                continue
            t = int(m.group(1))
            r = load_json(fp)
            z = r["z"]
            for p in range(min(len(z), P)):
                if p == t:
                    continue
                unrel_notarget[s][p].append(float(z[p]))
    x = np.arange(P)
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # --- Top: cooperative correct-key ---
    for i, s in enumerate(schemes):
        means = []
        errs = []
        for p in range(P):
            m, hw = mean_ci(coop_true[s][p])
            means.append(m)
            errs.append(hw)
        offs = (i - 0.5) * width
        axes[0].bar(x + offs, means, width=width, yerr=errs, capsize=3, edgecolor="black", linewidth=0.6, label=s)

    axes[0].set_ylabel("pooled z (coop, correct key)")
    axes[0].set_title("Sanity pooled z by provider: naive vs orthogonal directions")
    axes[0].legend()

    # --- Bottom: wrong-key and unrelated (both near 0, show clearly) ---
    # For each scheme we plot two bars per provider: wrong-key and unrelated
    subw = width / 2.0
    for i, s in enumerate(schemes):
        offs0 = (i - 0.5) * width
        # wrong-key
        means_w = []
        errs_w = []
        # unrelated
        means_u = []
        errs_u = []
        for p in range(P):
            m, hw = mean_ci(coop_wrong[s][p])
            means_w.append(m); errs_w.append(hw)
            m, hw = mean_ci(unrel_notarget[s][p])
            means_u.append(m); errs_u.append(hw)

        axes[1].bar(x + offs0 - subw/2, means_w, width=subw, yerr=errs_w, capsize=3,
                    edgecolor="black", linewidth=0.6, label=f"{s} wrong-key", alpha=0.85)
        axes[1].bar(x + offs0 + subw/2, means_u, width=subw, yerr=errs_u, capsize=3,
                    edgecolor="black", linewidth=0.6, label=f"{s} unrelated", alpha=0.55)

    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_ylabel("pooled z (null)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(p) for p in range(P)])
    axes[1].set_xlabel("provider id")
    # zoom around 0 for visibility
    axes[1].set_ylim(-4.0, 4.0)

    plt.tight_layout()
    plt.savefig(figpath, dpi=220)
    plt.close()
    print(f"[OK] {figpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=Path("results/direction_compare"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/figs"))
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    plot_rag_power(args.results_root, args.out_dir)
    plot_style_sweep(args.results_root, args.out_dir)
    plot_sanity_provider_bars(args.results_root, args.out_dir)

if __name__ == "__main__":
    main()
