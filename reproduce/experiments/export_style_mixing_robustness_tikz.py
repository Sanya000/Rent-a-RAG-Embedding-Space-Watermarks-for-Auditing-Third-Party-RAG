#!/usr/bin/env python3
"""Export fig:style-mixing-robustness-main as TikZ by capturing the aggregator's matplotlib figure.

Needs the rag_multi trial logs of the primary root and of the mixing-stress root.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import matplotlib.pyplot as plt

import aggregate_watermark_compare_and_plot as agg  # noqa: E402


def fmt(value: float) -> str:
    return f"{float(value):.12f}"


def fmt_tick(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def visible_ticks(ax, axis: str) -> list[float]:
    if axis == "x":
        lo, hi = ax.get_xlim()
        ticks = ax.get_xticks()
    else:
        lo, hi = ax.get_ylim()
        ticks = ax.get_yticks()
    return [float(t) for t in ticks if lo - 1e-9 <= float(t) <= hi + 1e-9]


def coords_block(points: Sequence[Sequence[float]], *, per_line: int = 3) -> str:
    chunks: list[str] = []
    for idx in range(0, len(points), per_line):
        row = " ".join(f"({fmt(x)},{fmt(y)})" for x, y in points[idx : idx + per_line])
        chunks.append(f"  {row}")
    return "\n".join(chunks)


def tikz_path_block(points: Sequence[Sequence[float]], *, per_line: int = 3) -> str:
    chunks: list[str] = []
    for idx in range(0, len(points), per_line):
        row = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in points[idx : idx + per_line])
        if idx + per_line < len(points):
            row += " --"
        chunks.append(f"  {row}")
    return "\n".join(chunks)


def line_to_coords(line) -> list[tuple[float, float]]:
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    return list(zip(x.tolist(), y.tolist()))


def polygon_vertices(poly_collection) -> list[tuple[float, float]]:
    path = poly_collection.get_paths()[0]
    verts = np.asarray(path.vertices, dtype=float)
    return [(float(x), float(y)) for x, y in verts]


def capture_figure(
    results_root: Path,
    out_dir: Path,
    schemes: list[str],
    *,
    mixing_stress_root: Path | None = None,
):
    saved: dict[str, object] = {}
    old_save = agg._save_fig

    def fake_save(fig, out_base, tight_rect=None):
        saved["fig"] = fig
        saved["tight_rect"] = tight_rect

    agg._save_fig = fake_save
    try:
        agg.plot_style_mixing_robustness_main(
            results_root,
            out_dir,
            schemes,
            mixing_stress_root=mixing_stress_root,
        )
    finally:
        agg._save_fig = old_save

    fig = saved.get("fig")
    if fig is None:
        raise RuntimeError("Failed to capture robustness figure from plotting pipeline.")
    fig.canvas.draw()
    return fig


def export_tikz(out_path: Path, fig) -> None:
    ax_a, ax_b = fig.axes

    a_fill = polygon_vertices(ax_a.collections[0])
    a_line = line_to_coords(ax_a.lines[0])
    a_thresh_y = float(np.asarray(ax_a.lines[1].get_ydata(), dtype=float)[0])

    b_line = line_to_coords(ax_b.lines[0])
    b_low = np.asarray(ax_b.lines[1].get_ydata(), dtype=float)
    b_high = np.asarray(ax_b.lines[2].get_ydata(), dtype=float)
    b_thresh_y = float(np.asarray(ax_b.lines[3].get_ydata(), dtype=float)[0])

    ax_a_xlim = ax_a.get_xlim()
    ax_a_ylim = ax_a.get_ylim()
    ax_b_xlim = ax_b.get_xlim()
    ax_b_ylim = ax_b.get_ylim()
    ax_a_xticks = visible_ticks(ax_a, "x")
    ax_a_yticks = visible_ticks(ax_a, "y")
    ax_b_xticks = visible_ticks(ax_b, "x")
    ax_b_yticks = visible_ticks(ax_b, "y")

    cap_halfwidth = 0.045

    lines: list[str] = []
    lines.append("% Auto-generated from results by script.")
    lines.append("% Figure: fig_style_mixing_robustness_main converted to TikZ/PGFPlots.")
    lines.append("\\begin{tikzpicture}")
    lines.append("\\definecolor{MixBlue}{HTML}{1F77B4}")
    lines.append("\\definecolor{FPRRed}{HTML}{D62728}")
    lines.append("\\pgfplotsset{compat=1.18}")
    lines.append("\\begin{groupplot}[")
    lines.append("  group style={group size=1 by 2, vertical sep=1.85cm},")
    lines.append("  scale only axis,")
    lines.append("  width=0.805000\\linewidth,")
    lines.append("  tick label style={font=\\scriptsize},")
    lines.append("  label style={font=\\small},")
    lines.append("  title style={font=\\small},")
    lines.append("  tick style={line width=0.8pt, color=black},")
    lines.append("  major tick length=2.8pt,")
    lines.append("  minor tick num=0,")
    lines.append("  line width=0.9pt")
    lines.append("]")

    lines.append("\\nextgroupplot[")
    lines.append("  height=1.48cm,")
    lines.append(f"  xmin={fmt(ax_a_xlim[0])}, xmax={fmt(ax_a_xlim[1])},")
    lines.append(f"  ymin={fmt(ax_a_ylim[0])}, ymax={fmt(ax_a_ylim[1])},")
    lines.append("  xtick={" + ",".join(fmt_tick(v) for v in ax_a_xticks) + "},")
    lines.append("  ytick={" + ",".join(fmt_tick(v) for v in ax_a_yticks) + "},")
    lines.append("  xlabel={Mixing level (\\# providers)},")
    lines.append("  ylabel={Pooled Z},")
    lines.append("  title={A) Mixing Robustness},")
    lines.append("  ymajorgrids=true,")
    lines.append("  xmajorgrids=false,")
    lines.append("  grid style={dotted, draw=gray!35},")
    lines.append("  major grid style={dotted, draw=gray!35}")
    lines.append("]")
    lines.append("\\path[draw=none, fill=MixBlue, fill opacity=0.16]")
    lines.append(tikz_path_block(a_fill, per_line=3))
    lines.append("-- cycle;")
    lines.append("\\addplot[MixBlue, line width=1.8pt, mark=*, mark size=2.2pt] coordinates {")
    lines.append(coords_block(a_line, per_line=4))
    lines.append("};")
    lines.append("\\addplot[black!65, dashed, line width=0.95pt, forget plot] coordinates {")
    lines.append(
        coords_block(
            [(float(ax_a_xlim[0]), a_thresh_y), (float(ax_a_xlim[1]), a_thresh_y)],
            per_line=2,
        )
    )
    lines.append("};")

    lines.append("\\nextgroupplot[")
    lines.append("  height=1.31cm,")
    lines.append(f"  xmin={fmt(ax_b_xlim[0])}, xmax={fmt(ax_b_xlim[1])},")
    lines.append(f"  ymin={fmt(ax_b_ylim[0])}, ymax={fmt(ax_b_ylim[1])},")
    lines.append("  xtick={" + ",".join(fmt_tick(v) for v in ax_b_xticks) + "},")
    lines.append("  ytick={" + ",".join(fmt_tick(v) for v in ax_b_yticks) + "},")
    lines.append("  xlabel={Mixing level (\\# providers)},")
    lines.append("  ylabel={P(non-target BH sig)},")
    lines.append("  title={B) BH False-Positive Rate},")
    lines.append("  ymajorgrids=true,")
    lines.append("  xmajorgrids=false,")
    lines.append("  grid style={dotted, draw=gray!35},")
    lines.append("  major grid style={dotted, draw=gray!35}")
    lines.append("]")
    lines.append("\\addplot[FPRRed, line width=1.7pt, mark=*, mark size=2.2pt] coordinates {")
    lines.append(coords_block(b_line, per_line=4))
    lines.append("};")
    for (x, y), lo, hi in zip(b_line, b_low.tolist(), b_high.tolist()):
        lines.append(
            f"\\draw[FPRRed, line width=0.95pt] "
            f"(axis cs:{fmt(x)},{fmt(lo)}) -- (axis cs:{fmt(x)},{fmt(hi)});"
        )
        lines.append(
            f"\\draw[FPRRed, line width=0.95pt] "
            f"(axis cs:{fmt(x - cap_halfwidth)},{fmt(lo)}) -- (axis cs:{fmt(x + cap_halfwidth)},{fmt(lo)});"
        )
        lines.append(
            f"\\draw[FPRRed, line width=0.95pt] "
            f"(axis cs:{fmt(x - cap_halfwidth)},{fmt(hi)}) -- (axis cs:{fmt(x + cap_halfwidth)},{fmt(hi)});"
        )
    lines.append("\\addplot[black!65, dashed, line width=0.9pt, forget plot] coordinates {")
    lines.append(
        coords_block(
            [(float(ax_b_xlim[0]), b_thresh_y), (float(ax_b_xlim[1]), b_thresh_y)],
            per_line=2,
        )
    )
    lines.append("};")

    lines.append("\\end{groupplot}")
    lines.append("\\end{tikzpicture}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"[OK] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the provider-mixing robustness figure to TikZ.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results/watermark_compare_global_tuned_mT20_kgwn3"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/figs/watermark_compare_global_tuned_mT20_kgwn3"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figs/watermark_compare_global_tuned_mT20_kgwn3/fig_style_mixing_robustness_main_tikz.tex"),
    )
    parser.add_argument("--mixing_stress_root", type=Path, default=None)
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["dirbucket_naive", "kgw_ward", "semstamp", "sim_attr", "overlap_attr", "entail_attr"],
    )
    args = parser.parse_args()

    fig = capture_figure(
        args.results_root,
        args.out_dir,
        args.schemes,
        mixing_stress_root=args.mixing_stress_root,
    )
    export_tikz(args.out, fig)


if __name__ == "__main__":
    main()
