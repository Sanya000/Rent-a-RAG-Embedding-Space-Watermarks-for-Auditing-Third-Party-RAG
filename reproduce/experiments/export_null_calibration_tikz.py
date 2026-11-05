#!/usr/bin/env python3
"""Export fig:null-calibration (two panels) and fig:null-calibration-single (panel A) as TikZ.

Reads <results-root>/dirbucket_naive/sanity_{coop,unrel}_t*_seed*.json (120 + 120 audits).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: float) -> str:
    return f"{float(value):.12f}"


def fmt_tick(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def visible_ticks(ax: plt.Axes, axis: str) -> list[float]:
    if axis == "x":
        lo, hi = ax.get_xlim()
        ticks = ax.get_xticks()
    else:
        lo, hi = ax.get_ylim()
        ticks = ax.get_yticks()
    out = [float(t) for t in ticks if lo - 1e-9 <= float(t) <= hi + 1e-9]
    return out


def coords_block(points: Sequence[Sequence[float]], *, per_line: int = 3) -> str:
    chunks: list[str] = []
    for idx in range(0, len(points), per_line):
        # PGFPlots `\addplot ... coordinates` expects plain `(x,y)` pairs.
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


def polygon_vertices(patch) -> list[tuple[float, float]]:
    if hasattr(patch, "get_path"):
        verts = np.asarray(patch.get_path().vertices, dtype=float)
    else:
        verts = np.asarray(patch.get_paths()[0].vertices, dtype=float)
    return [(float(x), float(y)) for x, y in verts]


def percent_from_alpha(alpha: float) -> str:
    value = 100.0 * float(alpha)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


@dataclass
class NullCalibrationData:
    z_correct: list[float]
    z_wrong: list[float]
    z_unrel: list[float]
    support_correct: list[int]
    support_wrong: list[int]
    support_unrel: list[int]
    evidence_correct: list[float]
    evidence_wrong: list[float]
    evidence_unrel: list[float]
    min_t: int


@dataclass
class HexCloud:
    label: str
    color_name: str
    points: np.ndarray
    centers: np.ndarray | None
    prototype: np.ndarray | None
    counts: np.ndarray | None
    point_opacity: float
    hex_alpha_min: float | None
    hex_alpha_max: float | None


def build_data(results_root: Path) -> NullCalibrationData:
    scheme_dir = results_root / "dirbucket_naive"
    coop_files = sorted(scheme_dir.glob("sanity_coop_t*_seed*.json"))
    unrel_files = sorted(scheme_dir.glob("sanity_unrel_t*_seed*.json"))
    if not coop_files or not unrel_files:
        raise FileNotFoundError(f"Missing sanity files under {scheme_dir}")

    z_correct: list[float] = []
    z_wrong: list[float] = []
    z_unrel: list[float] = []
    support_correct: list[int] = []
    support_wrong: list[int] = []
    support_unrel: list[int] = []
    evidence_correct: list[float] = []
    evidence_wrong: list[float] = []
    evidence_unrel: list[float] = []
    min_t_vals: list[int] = []

    for fp in coop_files:
        record = load_json(fp)
        tp = int(record.get("target_provider", 0))
        min_t = int(record.get("min_pooled_T", 1))
        min_t_vals.append(min_t)

        z_raw = record.get("z_raw") or record.get("z") or []
        pooled_t = record.get("pooled_T") or []
        if isinstance(z_raw, list) and z_raw and 0 <= tp < len(z_raw):
            z_correct.append(float(z_raw[tp]))
            if isinstance(pooled_t, list) and tp < len(pooled_t):
                support_correct.append(int(pooled_t[tp]))
                evidence_correct.append(float(z_raw[tp]))

        wrong = record.get("wrongkey") or {}
        wrong_z = wrong.get("z_raw") or wrong.get("z") or []
        wrong_t = wrong.get("pooled_T") or []
        if isinstance(wrong_z, list) and wrong_z and 0 <= tp < len(wrong_z):
            z_wrong.append(float(wrong_z[tp]))
            if isinstance(wrong_t, list) and tp < len(wrong_t):
                support_wrong.append(int(wrong_t[tp]))
                evidence_wrong.append(float(wrong_z[tp]))

    for fp in unrel_files:
        record = load_json(fp)
        z_raw = record.get("z_raw") or record.get("z") or []
        pooled_t = record.get("pooled_T") or []
        if isinstance(z_raw, list) and z_raw:
            z_unrel.extend(float(value) for value in z_raw)
            if isinstance(pooled_t, list) and len(pooled_t) == len(z_raw):
                support_unrel.extend(int(value) for value in pooled_t)
                evidence_unrel.extend(float(value) for value in z_raw)

    if not z_correct or not z_wrong or not z_unrel:
        raise RuntimeError("Insufficient data to build null calibration figure.")

    min_t = int(round(float(np.median(np.asarray(min_t_vals, dtype=float))))) if min_t_vals else 25
    return NullCalibrationData(
        z_correct=z_correct,
        z_wrong=z_wrong,
        z_unrel=z_unrel,
        support_correct=support_correct,
        support_wrong=support_wrong,
        support_unrel=support_unrel,
        evidence_correct=evidence_correct,
        evidence_wrong=evidence_wrong,
        evidence_unrel=evidence_unrel,
        min_t=min_t,
    )


def alpha_cmap(base_color: str, name: str, *, min_alpha: float, max_alpha: float) -> LinearSegmentedColormap:
    r, g, b, _ = to_rgba(base_color)
    return LinearSegmentedColormap.from_list(name, [(r, g, b, min_alpha), (r, g, b, max_alpha)])


def hex_cloud(
    ax: plt.Axes,
    xs: Sequence[int],
    ys: Sequence[float],
    *,
    label: str,
    color: str,
    color_name: str,
    max_alpha: float,
    zorder: int,
) -> HexCloud:
    x_np = np.asarray(xs, dtype=float)
    y_np = np.asarray(ys, dtype=float)
    if len(x_np) < 8:
        ax.scatter(
            x_np,
            y_np,
            s=26,
            marker="o",
            color=color,
            edgecolor="none",
            alpha=max_alpha,
            zorder=zorder,
        )
        return HexCloud(
            label=label,
            color_name=color_name,
            points=np.column_stack([x_np, y_np]),
            centers=None,
            prototype=None,
            counts=None,
            point_opacity=max_alpha,
            hex_alpha_min=None,
            hex_alpha_max=None,
        )

    min_alpha = 0.42 if label != "Unrelated null" else 0.34
    cmap = alpha_cmap(
        color,
        f"null_hex_{label.replace(' ', '_').lower()}",
        min_alpha=min_alpha,
        max_alpha=max_alpha,
    )
    hb = ax.hexbin(
        x_np,
        y_np,
        gridsize=12,
        mincnt=1,
        cmap=cmap,
        linewidths=0.0,
        edgecolors="none",
        zorder=zorder,
    )
    point_opacity = 0.48 if label != "Unrelated null" else 0.44
    ax.scatter(
        x_np,
        y_np,
        s=11.0,
        marker="o",
        color=color,
        edgecolor="none",
        alpha=point_opacity,
        zorder=zorder + 0.05,
    )

    counts = hb.get_array()
    if counts is None or int(np.size(counts)) == 0:
        return HexCloud(
            label=label,
            color_name=color_name,
            points=np.column_stack([x_np, y_np]),
            centers=None,
            prototype=None,
            counts=None,
            point_opacity=point_opacity,
            hex_alpha_min=None,
            hex_alpha_max=None,
        )

    counts = np.asarray(counts, dtype=float)
    max_count = float(np.nanmax(counts))
    if not np.isfinite(max_count) or max_count <= 1.0:
        hb.remove()
        point_opacity = max(0.52, 0.70 * max_alpha)
        ax.scatter(
            x_np,
            y_np,
            s=24,
            marker="o",
            color=color,
            edgecolor="none",
            alpha=point_opacity,
            zorder=zorder,
        )
        return HexCloud(
            label=label,
            color_name=color_name,
            points=np.column_stack([x_np, y_np]),
            centers=None,
            prototype=None,
            counts=None,
            point_opacity=point_opacity,
            hex_alpha_min=None,
            hex_alpha_max=None,
        )

    return HexCloud(
        label=label,
        color_name=color_name,
        points=np.column_stack([x_np, y_np]),
        centers=np.asarray(hb.get_offsets(), dtype=float),
        prototype=np.asarray(hb.get_paths()[0].vertices, dtype=float),
        counts=counts,
        point_opacity=point_opacity,
        hex_alpha_min=min_alpha,
        hex_alpha_max=max_alpha,
    )


def build_figure(data: NullCalibrationData):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.35, 3.25), gridspec_kw={"height_ratios": [0.72, 1.0]})
    fig.subplots_adjust(left=0.18, right=0.985, top=0.975, bottom=0.105, hspace=0.28)

    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.8
    dist_data = [data.z_correct, data.z_wrong, data.z_unrel]
    dist_labels = ["Correct key", "Wrong key", "Unrelated null"]
    dist_colors = ["#0072B2", "#D55E00", "#4D4D4D"]
    xpos = np.arange(1, len(dist_data) + 1)

    violin = ax_a.violinplot(
        dist_data,
        positions=xpos,
        vert=True,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], dist_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.36)

    box = ax_a.boxplot(
        dist_data,
        positions=xpos,
        vert=True,
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"linewidth": 0.9, "color": "black"},
        capprops={"linewidth": 0.9, "color": "black"},
    )
    for patch, color in zip(box["boxes"], dist_colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.15)

    all_vals = np.asarray([float(value) for arr in dist_data for value in arr], dtype=float)
    ymin = min(float(np.nanmin(all_vals)) - 0.8, -2.8)
    ymax = max(float(np.nanmax(all_vals)) + 0.8, 5.2)
    ax_a.set_ylim(ymin, ymax)
    ax_a.set_xlim(0.45, len(dist_data) + 0.55)
    ax_a.set_xticks(xpos)
    ax_a.set_yticks([-2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5])
    ax_a.set_xticklabels(dist_labels, fontsize=tick_fs)
    ax_a.axhline(0.0, linestyle="--", linewidth=0.9, color="#666666", alpha=0.9)
    ax_a.axhline(4.0, linestyle="--", linewidth=0.9, color="#666666", alpha=0.9)
    ax_a.set_ylabel("Pooled Z", fontsize=label_fs)
    ax_a.set_title("A) Null z Distributions", fontsize=title_fs, pad=4.2)
    ax_a.grid(axis="y", linestyle=":", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#0072B2", markeredgecolor="#0072B2", markersize=7.2, label="Correct key"),
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#D55E00", markeredgecolor="#D55E00", markersize=7.2, label="Wrong key"),
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#4D4D4D", markeredgecolor="#4D4D4D", markersize=7.2, label="Unrelated null"),
        ],
        loc="upper right",
        fontsize=legend_fs,
        frameon=True,
    )

    ax_b.axvline(float(data.min_t), linestyle="--", linewidth=0.95, color="#666666", alpha=0.90)
    ax_b.axhline(4.0, linestyle="--", linewidth=0.95, color="#666666", alpha=0.90)
    ax_b.axhline(0.0, linestyle=":", linewidth=0.8, color="#888888", alpha=0.65)

    x_all = np.asarray(
        [float(v) for v in (data.support_correct + data.support_wrong + data.support_unrel)] or [0.0, float(data.min_t)],
        dtype=float,
    )
    y_all = np.asarray(
        [float(v) for v in (data.evidence_correct + data.evidence_wrong + data.evidence_unrel)] or [0.0, 4.0],
        dtype=float,
    )
    x_hi = max(float(np.nanmax(x_all)) + 4.0, float(data.min_t) + 6.0)
    y_lo = min(float(np.nanmin(y_all)) - 0.8, -2.8)
    y_hi = max(float(np.nanmax(y_all)) + 0.8, 5.2)
    ax_b.set_xlim(-0.5, x_hi)
    ax_b.set_ylim(y_lo, y_hi)
    ax_b.set_yticks([-2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5])

    if data.min_t > 0:
        n_cols = 520
        grad = np.zeros((2, n_cols, 4), dtype=float)
        grad[..., 0] = 0xE1 / 255.0
        grad[..., 1] = 0x57 / 255.0
        grad[..., 2] = 0x59 / 255.0
        xfade = np.linspace(0.0, 1.0, n_cols, dtype=float)
        alpha = 0.33 * np.power(1.0 - xfade, 0.62) + 0.035
        grad[..., 3] = alpha[None, :]
        ax_b.imshow(
            grad,
            extent=(0.0, float(data.min_t), y_lo, y_hi),
            origin="lower",
            aspect="auto",
            interpolation="bicubic",
            zorder=0,
        )
        ax_b.text(
            0.36 * float(data.min_t),
            y_lo + 0.40 * (y_hi - y_lo),
            "Low-evidence region",
            color="#b22222",
            fontsize=7.8,
            ha="left",
            va="center",
            rotation=90,
            weight="bold",
            zorder=1,
        )

    clouds = {
        "correct": hex_cloud(
            ax_b,
            data.support_correct,
            data.evidence_correct,
            color="#0072B2",
            color_name="CorrectBlue",
            label="Correct key",
            max_alpha=1.00,
            zorder=3,
        ),
        "wrong": hex_cloud(
            ax_b,
            data.support_wrong,
            data.evidence_wrong,
            color="#D55E00",
            color_name="WrongTeal",
            label="Wrong key",
            max_alpha=1.00,
            zorder=3,
        ),
        "unrel": hex_cloud(
            ax_b,
            data.support_unrel,
            data.evidence_unrel,
            color="#4D4D4D",
            color_name="NullGray",
            label="Unrelated null",
            max_alpha=0.96,
            zorder=2,
        ),
    }

    ax_b.set_xlabel("Support (pooled $|T_i|$) at evaluated provider", fontsize=label_fs)
    ax_b.set_ylabel("Pooled Z", fontsize=label_fs)
    ax_b.set_title("B) Null Evidence vs Support", fontsize=title_fs, pad=4.2)
    ax_b.grid(True, linestyle=":", alpha=0.25)
    ax_b.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)
    fig.canvas.draw()
    return fig, ax_a, ax_b, violin, box, clouds, (y_lo, y_hi)


def shading_definition() -> str:
    stops: list[str] = []
    for pct in range(0, 101, 10):
        xfade = pct / 100.0
        alpha = 0.33 * ((1.0 - xfade) ** 0.62) + 0.035
        stops.append(f"  color({pct}bp)=(LowShadeBase!{percent_from_alpha(alpha)})")
    return "\\pgfdeclarehorizontalshading{LowEvidenceFade}{100bp}{%\n" + ";\n".join(stops) + "\n}"


def hex_polygons(cloud: HexCloud) -> str:
    if cloud.centers is None or cloud.prototype is None or cloud.counts is None:
        return ""
    count_min = float(np.nanmin(cloud.counts))
    count_max = float(np.nanmax(cloud.counts))
    if count_max <= count_min:
        alphas = np.full_like(cloud.counts, fill_value=float(cloud.hex_alpha_max), dtype=float)
    else:
        alphas = cloud.hex_alpha_min + (cloud.counts - count_min) * (
            (cloud.hex_alpha_max - cloud.hex_alpha_min) / (count_max - count_min)
        )
    order = np.argsort(alphas)
    lines = [f"% {cloud.label} hexbin cells"]
    for idx in order:
        cx, cy = cloud.centers[idx]
        verts = cloud.prototype + np.array([cx, cy], dtype=float)
        verts_block = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in verts)
        lines.append(
            f"\\path[draw=none, fill={cloud.color_name}, fill opacity={alphas[idx]:.6f}] {verts_block} -- cycle;"
        )
    return "\n".join(lines)


def scatter_plot(name: str, color_name: str, points: np.ndarray, opacity: float) -> str:
    if points.size == 0:
        return ""
    return (
        f"\\addplot[\n"
        f"  only marks,\n"
        f"  mark=*,\n"
        f"  mark size=1.15pt,\n"
        f"  draw=none,\n"
        f"  fill={color_name},\n"
        f"  fill opacity={opacity:.6f}\n"
        f"] coordinates {{\n"
        f"{coords_block(points.tolist(), per_line=5)}\n"
        f"}};\n"
    )


def legend_entries() -> str:
    return "\n".join(
        [
            "\\addlegendimage{only marks, mark=square*, mark size=3.0pt, draw=CorrectBlue, draw opacity=1, fill=CorrectBlue, fill opacity=1}",
            "\\addlegendentry{Correct key}",
            "\\addlegendimage{only marks, mark=square*, mark size=3.0pt, draw=WrongTeal, draw opacity=1, fill=WrongTeal, fill opacity=1}",
            "\\addlegendentry{Wrong key}",
            "\\addlegendimage{only marks, mark=square*, mark size=3.0pt, draw=NullGray, draw opacity=1, fill=NullGray, fill opacity=1}",
            "\\addlegendentry{Unrelated null}",
        ]
    )


def export_tikz(
    out_path: Path,
    fig,
    ax_a: plt.Axes,
    ax_b: plt.Axes,
    violin,
    box,
    clouds: dict[str, HexCloud],
    b_ylim: tuple[float, float],
    min_t: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_bounds = ax_a.get_position().bounds
    bottom_bounds = ax_b.get_position().bounds
    axis_width_frac = top_bounds[2]
    top_height_cm = fig.get_figheight() * 2.54 * top_bounds[3]
    bottom_height_cm = fig.get_figheight() * 2.54 * bottom_bounds[3]
    axis_vsep_cm = fig.get_figheight() * 2.54 * (
        top_bounds[1] - (bottom_bounds[1] + bottom_bounds[3])
    )

    a_xticks = [1, 2, 3]
    a_yticks = visible_ticks(ax_a, "y")
    b_xticks = visible_ticks(ax_b, "x")
    b_yticks = visible_ticks(ax_b, "y")
    b_xticks = [tick for tick in b_xticks if tick >= 0.0]

    violin_paths = [polygon_vertices(body) for body in violin["bodies"]]
    box_polys = [polygon_vertices(patch) for patch in box["boxes"]]
    medians = [line_to_coords(line) for line in box["medians"]]
    whiskers = [line_to_coords(line) for line in box["whiskers"]]
    caps = [line_to_coords(line) for line in box["caps"]]

    y_lo, y_hi = b_ylim
    label_x = 0.36 * float(min_t)
    label_y = y_lo + 0.40 * (y_hi - y_lo)

    lines: list[str] = []
    lines.append("% Auto-generated from results by script.")
    lines.append("% Figure: fig_null_calibration converted to TikZ/PGFPlots.")
    lines.append("\\begin{tikzpicture}")
    lines.append("\\definecolor{CorrectBlue}{HTML}{0072B2}")
    lines.append("\\definecolor{WrongTeal}{HTML}{D55E00}")
    lines.append("\\definecolor{NullGray}{HTML}{4D4D4D}")
    lines.append("\\definecolor{LowShadeBase}{HTML}{E15759}")
    lines.append("\\definecolor{AnnotRed}{HTML}{B22222}")
    lines.append("\\pgfplotsset{compat=1.18}")
    lines.append(shading_definition())
    lines.append("\\begin{groupplot}[")
    lines.append(f"  group style={{group size=1 by 2, vertical sep={axis_vsep_cm:.3f}cm}},")
    lines.append("  scale only axis,")
    lines.append(f"  width={axis_width_frac:.6f}\\linewidth,")
    lines.append("  tick label style={font=\\scriptsize},")
    lines.append("  label style={font=\\small},")
    lines.append("  title style={font=\\small},")
    lines.append("  tick style={line width=0.8pt, color=black},")
    lines.append("  major tick length=2.8pt,")
    lines.append("  minor tick num=0")
    lines.append("]")
    lines.append("\\nextgroupplot[")
    lines.append(f"  height={top_height_cm:.3f}cm,")
    lines.append(f"  xmin={fmt(ax_a.get_xlim()[0])}, xmax={fmt(ax_a.get_xlim()[1])},")
    lines.append(f"  ymin={fmt(ax_a.get_ylim()[0])}, ymax={fmt(ax_a.get_ylim()[1])},")
    lines.append(f"  xtick={{{','.join(fmt_tick(x) for x in a_xticks)}}},")
    lines.append(f"  ytick={{{','.join(fmt_tick(y) for y in a_yticks)}}},")
    lines.append("  xticklabels={Correct key,Wrong key,Unrelated null},")
    lines.append("  ylabel={Pooled Z},")
    lines.append("  title={A) Null z Distributions},")
    lines.append("  ymajorgrids=true,")
    lines.append("  xmajorgrids=false,")
    lines.append("  grid style={dotted, draw=gray!35},")
    lines.append("  major grid style={dotted, draw=gray!35},")
    lines.append(
        "  legend style={font=\\scriptsize, draw=gray!35, fill=white, at={(0.97,0.97)}, anchor=north east, cells={anchor=west}}"
    )
    lines.append("]")
    for color_name, path in zip(["CorrectBlue", "WrongTeal", "NullGray"], violin_paths):
        lines.append(
            f"\\path[draw={color_name}, draw opacity=0.50, fill={color_name}, fill opacity=0.36, line width=1.0pt]"
        )
        lines.append(tikz_path_block(path, per_line=3))
        lines.append("-- cycle;")
    for color_name, poly in zip(["CorrectBlue", "WrongTeal", "NullGray"], box_polys):
        lines.append(f"\\path[draw={color_name}, fill=white, line width=1.15pt]")
        lines.append(tikz_path_block(poly, per_line=5))
        lines.append("-- cycle;")
    for line in whiskers:
        pts = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in line)
        lines.append(f"\\draw[black, line width=0.9pt] {pts};")
    for line in caps:
        pts = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in line)
        lines.append(f"\\draw[black, line width=0.9pt] {pts};")
    for line in medians:
        pts = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in line)
        lines.append(f"\\draw[black, line width=1.1pt] {pts};")
    lines.append(
        f"\\draw[black!70, dashed, line width=0.9pt] (axis cs:{fmt(ax_a.get_xlim()[0])},0.0) -- (axis cs:{fmt(ax_a.get_xlim()[1])},0.0);"
    )
    lines.append(
        f"\\draw[black!70, dashed, line width=0.9pt] (axis cs:{fmt(ax_a.get_xlim()[0])},4.0) -- (axis cs:{fmt(ax_a.get_xlim()[1])},4.0);"
    )
    lines.append(legend_entries())

    lines.append("\\nextgroupplot[")
    lines.append(f"  height={bottom_height_cm:.3f}cm,")
    lines.append(f"  xmin={fmt(ax_b.get_xlim()[0])}, xmax={fmt(ax_b.get_xlim()[1])},")
    lines.append(f"  ymin={fmt(ax_b.get_ylim()[0])}, ymax={fmt(ax_b.get_ylim()[1])},")
    lines.append(f"  xtick={{{','.join(fmt_tick(x) for x in b_xticks)}}},")
    lines.append(f"  ytick={{{','.join(fmt_tick(y) for y in b_yticks)}}},")
    lines.append("  xlabel={Support (pooled $|T_i|$) at evaluated provider},")
    lines.append("  ylabel={Pooled Z},")
    lines.append("  title={B) Null Evidence vs Support},")
    lines.append("  grid=both,")
    lines.append("  grid style={dotted, draw=gray!35},")
    lines.append("  major grid style={dotted, draw=gray!35}")
    lines.append("]")
    lines.append(
        f"\\path[draw=none, shading=LowEvidenceFade] (axis cs:0.0,{fmt(y_lo)}) rectangle (axis cs:{fmt(float(min_t))},{fmt(y_hi)});"
    )
    lines.append(
        f"\\node[font=\\fontsize{{7.8}}{{8.6}}\\selectfont\\bfseries, text=AnnotRed, rotate=90] at (axis cs:{fmt(label_x)},{fmt(label_y)}) {{Low-evidence region}};"
    )
    lines.append(
        f"\\draw[black!70, dashed, line width=0.95pt] (axis cs:{fmt(float(min_t))},{fmt(y_lo)}) -- (axis cs:{fmt(float(min_t))},{fmt(y_hi)});"
    )
    lines.append(
        f"\\draw[black!70, dashed, line width=0.95pt] (axis cs:{fmt(ax_b.get_xlim()[0])},4.0) -- (axis cs:{fmt(ax_b.get_xlim()[1])},4.0);"
    )
    lines.append(
        f"\\draw[black!55, dotted, line width=0.8pt] (axis cs:{fmt(ax_b.get_xlim()[0])},0.0) -- (axis cs:{fmt(ax_b.get_xlim()[1])},0.0);"
    )

    for key in ["unrel", "correct", "wrong"]:
        cloud = clouds[key]
        hex_text = hex_polygons(cloud)
        if hex_text:
            lines.append(hex_text)
        scatter_text = scatter_plot(cloud.label, cloud.color_name, cloud.points, cloud.point_opacity)
        if scatter_text:
            lines.append(scatter_text.rstrip())
    lines.append("\\end{groupplot}")
    lines.append("\\end{tikzpicture}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_panel_a_only_tikz(
    out_path: Path,
    fig,
    ax_a: plt.Axes,
    violin,
    box,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_bounds = ax_a.get_position().bounds
    axis_width_frac = top_bounds[2]
    top_height_cm = fig.get_figheight() * 2.54 * top_bounds[3]

    a_xticks = [1, 2, 3]
    a_yticks = visible_ticks(ax_a, "y")

    violin_paths = [polygon_vertices(body) for body in violin["bodies"]]
    box_polys = [polygon_vertices(patch) for patch in box["boxes"]]
    medians = [line_to_coords(line) for line in box["medians"]]
    whiskers = [line_to_coords(line) for line in box["whiskers"]]
    caps = [line_to_coords(line) for line in box["caps"]]

    lines: list[str] = []
    lines.append("% Auto-generated from results by script.")
    lines.append("% Figure: fig_null_calibration panel A only converted to TikZ/PGFPlots.")
    lines.append("\\begin{tikzpicture}")
    lines.append("\\definecolor{CorrectBlue}{HTML}{0072B2}")
    lines.append("\\definecolor{WrongTeal}{HTML}{D55E00}")
    lines.append("\\definecolor{NullGray}{HTML}{4D4D4D}")
    lines.append("\\pgfplotsset{compat=1.18}")
    lines.append("\\begin{axis}[")
    lines.append("  scale only axis,")
    lines.append(f"  width={axis_width_frac:.6f}\\linewidth,")
    lines.append(f"  height={top_height_cm:.3f}cm,")
    lines.append("  tick label style={font=\\scriptsize},")
    lines.append("  label style={font=\\small},")
    lines.append("  title style={font=\\small},")
    lines.append("  tick style={line width=0.8pt, color=black},")
    lines.append("  major tick length=2.8pt,")
    lines.append("  minor tick num=0,")
    lines.append(f"  xmin={fmt(ax_a.get_xlim()[0])}, xmax={fmt(ax_a.get_xlim()[1])},")
    lines.append(f"  ymin={fmt(ax_a.get_ylim()[0])}, ymax={fmt(ax_a.get_ylim()[1])},")
    lines.append(f"  xtick={{{','.join(fmt_tick(x) for x in a_xticks)}}},")
    lines.append(f"  ytick={{{','.join(fmt_tick(y) for y in a_yticks)}}},")
    lines.append("  xticklabels={Correct key,Wrong key,Unrelated null},")
    lines.append("  ylabel={Pooled Z},")
    lines.append("  title={A) Null z Distributions},")
    lines.append("  ymajorgrids=true,")
    lines.append("  xmajorgrids=false,")
    lines.append("  grid style={dotted, draw=gray!35},")
    lines.append("  major grid style={dotted, draw=gray!35}")
    lines.append("]")
    for color_name, path in zip(["CorrectBlue", "WrongTeal", "NullGray"], violin_paths):
        lines.append(
            f"\\path[draw={color_name}, draw opacity=0.50, fill={color_name}, fill opacity=0.36, line width=1.0pt]"
        )
        lines.append(tikz_path_block(path, per_line=3))
        lines.append("-- cycle;")
    for color_name, poly in zip(["CorrectBlue", "WrongTeal", "NullGray"], box_polys):
        lines.append(f"\\path[draw={color_name}, fill=white, line width=1.15pt]")
        lines.append(tikz_path_block(poly, per_line=5))
        lines.append("-- cycle;")
    for line in whiskers:
        pts = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in line)
        lines.append(f"\\draw[black, line width=0.9pt] {pts};")
    for line in caps:
        pts = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in line)
        lines.append(f"\\draw[black, line width=0.9pt] {pts};")
    for line in medians:
        pts = " -- ".join(f"(axis cs:{fmt(x)},{fmt(y)})" for x, y in line)
        lines.append(f"\\draw[black, line width=1.1pt] {pts};")
    lines.append(
        f"\\draw[black!70, dashed, line width=0.9pt] (axis cs:{fmt(ax_a.get_xlim()[0])},0.0) -- (axis cs:{fmt(ax_a.get_xlim()[1])},0.0);"
    )
    lines.append(
        f"\\draw[black!70, dashed, line width=0.9pt] (axis cs:{fmt(ax_a.get_xlim()[0])},4.0) -- (axis cs:{fmt(ax_a.get_xlim()[1])},4.0);"
    )
    lines.append("\\end{axis}")
    lines.append("\\end{tikzpicture}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fig_null_calibration as TikZ/PGFPlots.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/watermark_compare_global_tuned_mT20_kgwn3"),
        help="Root directory containing the DirBucket sanity JSONs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figs/watermark_compare_global_tuned_mT20_kgwn3/fig_null_calibration_tikz.tex"),
        help="Output TikZ path.",
    )
    parser.add_argument(
        "--out-panel-a-only",
        type=Path,
        default=None,
        help="Optional output path for a panel-A-only legend-free TikZ export.",
    )
    args = parser.parse_args()

    data = build_data(args.results_root)
    fig, ax_a, ax_b, violin, box, clouds, b_ylim = build_figure(data)
    try:
        export_tikz(args.out, fig, ax_a, ax_b, violin, box, clouds, b_ylim, data.min_t)
        if args.out_panel_a_only is not None:
            export_panel_a_only_tikz(args.out_panel_a_only, fig, ax_a, violin, box)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
