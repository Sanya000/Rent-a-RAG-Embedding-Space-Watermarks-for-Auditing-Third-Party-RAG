#!/usr/bin/env python3
"""Optional sensitivity figures for App. app:sensitivity.

  1. sensitivity_landscape.pdf: three-zone stability map over the (tau_sim,
     tau_margin) grid with FPR annotated, from sensitivity_extended.json.
  2. tmin_sensitivity.pdf: detection rate and FPR vs T_min from
     tmin_sweep_dense.json (falls back to the 3-point base sweep).

Usage:
  python experiments/plot_sensitivity_heatmap.py \
      [--extended results/sensitivity_sweep/sensitivity_extended.json] \
      [--original results/sensitivity_sweep/sensitivity_results.json] \
      [--tmin_dense results/sensitivity_sweep/tmin_sweep_dense.json] \
      [--out_dir results/sensitivity_sweep/figures]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator


# ─── Palette ──────────────────────────────────────────────────────────────────
GREEN   = "#2ca02c"    # stable
RED     = "#d62728"    # FPR > 0
BLUE    = "#1f77b4"    # starvation (det < 1, fpr = 0)
GOLD    = "#ff7f0e"    # paper default marker
LIGHT_G = "#c7e9c0"   # light stable (for gradient within stable)


def load_grid(extended_path: Path, W: int = 28, T_min: int = 25):
    data = json.load(open(extended_path))
    grid = {}
    for k, v in data.items():
        if v.get("W") == W and v.get("T_min") == T_min:
            ts = round(v["tau_sim"], 4)
            tm = round(v["tau_margin"], 4)
            grid[(ts, tm)] = v
    return grid


def categorise(det: float, fpr: float):
    """Returns ('stable'|'fpr'|'starve', severity_0_to_1)."""
    eps = 1e-9
    if fpr > eps:
        return "fpr", fpr
    if det < 1.0 - eps:
        return "starve", 1.0 - det
    return "stable", 0.0


# ─── Figure 1: Stability landscape ────────────────────────────────────────────
def make_landscape(grid: dict, out_path: Path):
    ts_vals = sorted(set(k[0] for k in grid))
    tm_vals = sorted(set(k[1] for k in grid))
    nts, ntm = len(ts_vals), len(tm_vals)

    # Build category + severity arrays
    cat_arr = np.empty((ntm, nts), dtype=object)
    sev_arr = np.zeros((ntm, nts))
    for i, tm in enumerate(tm_vals):
        for j, ts in enumerate(ts_vals):
            v = grid.get((ts, tm), {})
            det = v.get("detection_rate", 0.0)
            fpr = v.get("fpr", 1.0)
            cat, sev = categorise(det, fpr)
            cat_arr[i, j] = cat
            sev_arr[i, j] = sev

    # Colour matrix (RGBA)
    rgba = np.zeros((ntm, nts, 4))
    from matplotlib.colors import to_rgba
    for i in range(ntm):
        for j in range(nts):
            cat = cat_arr[i, j]
            sev = sev_arr[i, j]
            if cat == "stable":
                rgba[i, j] = to_rgba(GREEN, alpha=0.85)
            elif cat == "fpr":
                # Light red at low FPR, deep red at high FPR
                base_r = np.array(to_rgba("#ffcccc"))
                dark_r = np.array(to_rgba(RED))
                rgba[i, j] = (1 - sev) * base_r + sev * dark_r
            else:  # starve
                base_b = np.array(to_rgba("#d4e4f7"))
                dark_b = np.array(to_rgba(BLUE))
                rgba[i, j] = (1 - sev * 0.5) * base_b + sev * 0.5 * dark_b

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Draw pixels
    ax.imshow(rgba, aspect="auto", origin="lower",
              extent=[-0.5, nts - 0.5, -0.5, ntm - 0.5], interpolation="none")

    # Grid lines
    for x in np.arange(-0.5, nts, 1):
        ax.axvline(x, color="white", lw=0.4, alpha=0.5)
    for y in np.arange(-0.5, ntm, 1):
        ax.axhline(y, color="white", lw=0.4, alpha=0.5)

    # Annotate FPR values in the FPR zone (skip if 0)
    for i, tm in enumerate(tm_vals):
        for j, ts in enumerate(ts_vals):
            v = grid.get((ts, tm), {})
            fpr = v.get("fpr", 0.0)
            det = v.get("detection_rate", 0.0)
            if fpr > 0.005:
                ax.text(j, i, f"{fpr:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black", fontweight="bold")
            elif det < 0.995:
                ax.text(j, i, f"{det:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")

    # Stable-region boundary contour (draw manually using rectangles)
    for i, tm in enumerate(tm_vals):
        for j, ts in enumerate(ts_vals):
            if cat_arr[i, j] == "stable":
                # Check right neighbour
                if j + 1 < nts and cat_arr[i, j + 1] != "stable":
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5],
                            color="black", lw=2.0)
                # Check left neighbour
                if j - 1 >= 0 and cat_arr[i, j - 1] != "stable":
                    ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5],
                            color="black", lw=2.0)
                # Check top
                if i + 1 < ntm and cat_arr[i + 1, j] != "stable":
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5],
                            color="black", lw=2.0)
                # Check bottom
                if i - 1 >= 0 and cat_arr[i - 1, j] != "stable":
                    ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5],
                            color="black", lw=2.0)

    # Paper operating point
    pi = tm_vals.index(0.05)
    pj = ts_vals.index(0.82)
    ax.scatter([pj], [pi], s=300, marker="*", c=GOLD, edgecolors="black",
               linewidths=0.8, zorder=10, label=r"Paper default ($\tau_{\rm sim}$=0.82, $\tau_{\rm margin}$=0.05)")

    # Zone labels
    ax.text(1.5, 7.5, "False positive\nzone (FPR > 0)", ha="center", va="center",
            fontsize=9, color="darkred", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))
    ax.text(10.5, 4.0, "Starvation\n(det < 1)", ha="center", va="center",
            fontsize=9, color="darkblue", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))
    ax.text(6.5, 4.2, "Stable\n(det=1, FPR=0)", ha="center", va="center",
            fontsize=10, color="darkgreen", fontweight="bold")

    # Axes
    ax.set_xticks(range(nts))
    ax.set_xticklabels([f"{v:.2f}" for v in ts_vals], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(ntm))
    ax.set_yticklabels([f"{v:.2f}" for v in tm_vals], fontsize=8)
    ax.set_xlabel(r"Similarity threshold $\tau_{\rm sim}$", fontsize=11)
    ax.set_ylabel(r"Margin threshold $\tau_{\rm margin}$", fontsize=11)
    ax.set_title(
        r"DirBucket parameter stability landscape ($W$=28, $T_{\min}$=25)"
        "\nNumbers in coloured cells show FPR (red) or detection rate (blue)",
        fontsize=10.5)

    # Legend
    patches = [
        mpatches.Patch(fc=GREEN, alpha=0.85, label="Stable (det=1.0, FPR=0.0)"),
        mpatches.Patch(fc="#ff7777",          label="FPR > 0 (false positives)"),
        mpatches.Patch(fc="#aaccee",          label="Starvation (det < 1.0, FPR=0)"),
    ]
    ax.legend(handles=patches + [ax.collections[0]],
              loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ─── Figure 2: T_min dual-axis ────────────────────────────────────────────────
def make_tmin_plot(original_path: Path, dense_path: Path | None, out_path: Path):

    if dense_path and dense_path.exists():
        raw = json.load(open(dense_path))
        rows = [v for v in raw.values()]
        rows.sort(key=lambda r: r["T_min"])
        source = "dense sweep"
    else:
        raw = json.load(open(original_path))
        rows = []
        for k, v in raw.items():
            if (v.get("W") == 28
                    and abs(v.get("tau_sim", 0) - 0.82) < 0.001
                    and abs(v.get("tau_margin", 0) - 0.05) < 0.001):
                rows.append(v)
        rows.sort(key=lambda r: r["T_min"])
        source = "3-point sweep (dense not yet ready)"

    T_vals  = [r["T_min"]         for r in rows]
    det_vals = [r["detection_rate"] for r in rows]
    fpr_vals = [r["fpr"]           for r in rows]
    q2bh_vals = [r.get("median_q2bh") for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    # Stable region shading
    stable_T = min((T for T, d, f in zip(T_vals, det_vals, fpr_vals)
                    if d >= 1.0 and f == 0.0), default=None)
    if stable_T:
        ax1.axvspan(stable_T, max(T_vals) + 5, alpha=0.10, color=GREEN, zorder=0)
        ax1.axvline(stable_T, color=GREEN, lw=1.5, linestyle=":", alpha=0.8, zorder=1)

    # Detection rate (left axis)
    ax1.plot(T_vals, det_vals, "o-", color=GREEN, lw=2.5, markersize=6,
             label="Detection rate", zorder=3)
    ax1.set_ylabel("Detection rate", color=GREEN, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=GREEN)
    ax1.set_ylim(-0.05, 1.12)
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))

    # FPR (right axis)
    ax2.plot(T_vals, fpr_vals, "s--", color=RED, lw=2.5, markersize=6,
             label="FPR", zorder=3)
    ax2.set_ylabel("False positive rate (FPR)", color=RED, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(-0.05, 1.12)
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))

    # Optional: q2bh as a third line (right axis, scaled to [0,1])
    valid_q = [(T, q) for T, q in zip(T_vals, q2bh_vals) if q is not None]
    if valid_q and len(valid_q) > 2:
        Tq, Qq = zip(*valid_q)
        q_max = max(Qq)
        ax2.plot(Tq, [q / q_max for q in Qq], "^:", color="gray", lw=1.5,
                 markersize=5, alpha=0.7, label=f"Queries-to-detect (÷{q_max:.0f})")

    # Paper T_min marker
    ax1.scatter([25], [next((d for T, d in zip(T_vals, det_vals) if T == 25), 1.0)],
                s=220, marker="*", c=GOLD, edgecolors="black", linewidths=0.8,
                zorder=5, label=r"Paper default ($T_{\min}$=25)")

    ax1.set_xlabel(r"Minimum evidence gate $T_{\min}$", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_xticks(T_vals)
    ax1.set_xticklabels([str(t) for t in T_vals], rotation=45, fontsize=7.5)
    ax1.set_title(
        r"$T_{\min}$ sensitivity: detection rate and FPR"
        "\n" + r"($W$=28, $\tau_{\rm sim}$=0.82, $\tau_{\rm margin}$=0.05) — " + source,
        fontsize=10)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8.5,
               framealpha=0.9)

    # Annotations
    fpr_boundary = next((T for T, f in zip(T_vals, fpr_vals) if f == 0.0), None)
    if fpr_boundary:
        ax2.annotate(f"FPR=0\nat T={fpr_boundary}",
                     xy=(fpr_boundary, 0), xytext=(fpr_boundary * 1.5, 0.25),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.3),
                     fontsize=8, color=RED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path} ({source})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended",   type=Path,
                    default=Path("results/sensitivity_sweep/sensitivity_extended.json"))
    ap.add_argument("--original",   type=Path,
                    default=Path("results/sensitivity_sweep/sensitivity_results.json"))
    ap.add_argument("--tmin_dense", type=Path,
                    default=Path("results/sensitivity_sweep/tmin_sweep_dense.json"))
    ap.add_argument("--out_dir",    type=Path,
                    default=Path("results/sensitivity_sweep/figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid(args.extended)
    make_landscape(grid, args.out_dir / "sensitivity_landscape.pdf")
    make_tmin_plot(args.original,
                   args.tmin_dense if args.tmin_dense.exists() else None,
                   args.out_dir / "tmin_sensitivity.pdf")


if __name__ == "__main__":
    main()
