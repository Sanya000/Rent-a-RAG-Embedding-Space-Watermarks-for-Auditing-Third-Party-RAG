"""Render the README result figures from the shipped paper TikZ sources.

The curves are parsed directly from the PGFPlots coordinates in
``reproduce/results/figs/`` (the same data that produced the paper's figures),
so nothing here is re-simulated.

    python docs/assets/make_figures.py

writes ``results-{light,dark}.png`` and ``comparison-{light,dark}.png`` next to
this file. Backgrounds are opaque (GitHub's light and dark page colours), so each
image stays readable where ``<picture>`` theme switching is not supported.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[1] / "reproduce" / "results" / "figs"
EVIDENCE_TEX = FIGS / "watermark_compare_global_tuned_mT20_kgwn3_rag160" / "fig_evidence_accumulation_main_tikz.tex"
FRONTIER_TEX = FIGS / "watermark_compare_global_tuned_mT20_kgwn3" / "fig_wm_frontier_z_q2d_tikz.tex"

THEMES = {
    "light": dict(
        text="#1f2328", muted="#59636e", grid="#d8dee4", axis="#8c959f",
        target="#2a78d6", nontarget="#e34948", ref="#59636e",
        kgw="#eb6834", sem="#1baf7a", band_alpha=0.20, bg="#ffffff",
    ),
    "dark": dict(
        text="#e6edf3", muted="#9198a1", grid="#30363d", axis="#656c76",
        target="#4d9bf0", nontarget="#f07070", ref="#9198a1",
        kgw="#f0834f", sem="#2bbf8a", band_alpha=0.26, bg="#0d1117",
    ),
}

COORD = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
COORD_PM = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*\+-\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")


def _addplots(tex: str) -> list[tuple[str, str]]:
    """Return (options, coordinate-body) for every \\addplot ... coordinates {...}."""
    out = []
    for m in re.finditer(r"\\addplot\[(.*?)\]\s*coordinates\s*\{(.*?)\}", tex, flags=re.S):
        out.append((m.group(1), m.group(2)))
    return out


def _xy(body: str) -> np.ndarray:
    return np.array([(float(a), float(b)) for a, b in COORD.findall(body)])


def _band(body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A closed fill polygon: lower edge left->right, then upper edge right->left."""
    pts = _xy(body)
    n = len(pts) // 2
    lo, hi = pts[:n], pts[n:][::-1]
    assert np.allclose(lo[:, 0], hi[:, 0])
    return lo[:, 0], lo[:, 1], hi[:, 1]


def load_evidence():
    tex = EVIDENCE_TEX.read_text()
    plots = _addplots(tex)
    z_band = next(b for o, b in plots if "fill=DBBlue" in o)
    z_med = next(b for o, b in plots if o.startswith("DBBlue"))
    non = next(b for o, b in plots if o.startswith("NonRed"))
    cdf_band = next(b for o, b in plots if "fill=BHGreen" in o)
    cdf_med = next(b for o, b in plots if o.startswith("BHGreen"))
    return dict(
        z_band=_band(z_band), z_med=_xy(z_med), non=_xy(non),
        cdf_band=_band(cdf_band), cdf_med=_xy(cdf_med),
    )


def load_frontier():
    tex = FRONTIER_TEX.read_text()
    points = {}
    for m in re.finditer(r"coordinates \{(\([^}]*\+-[^}]*)\};\s*\\addlegendentry\{(\w+)\}", tex):
        x, y, xe, ye = map(float, COORD_PM.search(m.group(1)).groups())
        points[m.group(2)] = dict(non=x, non_ci=xe, target=y, target_ci=ye)
    curves = {}
    for o, b in _addplots(tex):
        for key, name in (("DBBlue,", "DirBucket"), ("KGWOrange,", "KGW"), ("SemTeal,", "SemStamp")):
            if o.startswith(key):
                curves[name] = _xy(b)
    return points, curves


def _style_axes(ax, t):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=t["muted"], labelsize=10, length=3, width=0.8)
    ax.grid(True, color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])


def _title(ax, text, sub, t):
    ax.text(0.0, 1.13, text, transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=t["text"], ha="left", va="bottom")
    ax.text(0.0, 1.05, sub, transform=ax.transAxes, fontsize=10, color=t["muted"],
            ha="left", va="bottom")


def results_figure(theme: str, out: Path) -> None:
    t = THEMES[theme]
    d = load_evidence()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=200,
                                   gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.24))
    fig.patch.set_facecolor(t["bg"])

    # Panel A: pooled Z vs audited answers.
    x, lo, hi = d["z_band"]
    ax1.fill_between(x, lo, hi, color=t["target"], alpha=t["band_alpha"], linewidth=0)
    ax1.plot(d["z_med"][:, 0], d["z_med"][:, 1], color=t["target"], lw=2.2, solid_capstyle="round")
    ax1.plot(d["non"][:, 0], d["non"][:, 1], color=t["nontarget"], lw=2.2, solid_capstyle="round",
             clip_on=False, zorder=5)
    ax1.axhline(4, color=t["ref"], lw=1.1, ls=(0, (5, 4)))
    ax1.set_xlim(1, 160)
    ax1.set_ylim(-0.4, 18.5)
    ax1.set_xticks([1, 40, 80, 120, 160])
    ax1.set_yticks([0, 4, 8, 12, 16])
    ax1.set_xlabel("audited answers")
    ax1.set_ylabel("pooled Z")
    _style_axes(ax1, t)
    _title(ax1, "Evidence accumulates with every answer",
           "median pooled Z (band: IQR) over target audits", t)
    xm, ym = d["z_med"][-1]
    ax1.text(xm - 2, ym + 1.25, "target provider", color=t["text"], fontsize=10,
             ha="right", va="bottom", fontweight="bold")
    ax1.text(158, 0.6, "max non-target provider", color=t["text"], fontsize=10,
             ha="right", va="bottom")
    ax1.text(158, 4.45, "Z = 4", color=t["muted"], fontsize=9.5, ha="right", va="bottom")
    # first q where the median crosses Z=4
    zq = d["z_med"]
    q4 = int(zq[np.argmax(zq[:, 1] >= 4.0), 0])
    ax1.plot([q4], [zq[q4 - 1, 1]], "o", ms=6.5, color=t["target"], mec="white" if theme == "light" else "#0d1117",
             mew=1.6, zorder=6)
    ax1.annotate(f"crosses Z = 4 after {q4} answers", xy=(q4, zq[q4 - 1, 1]), xytext=(q4 + 12, 1.6),
                 fontsize=9.5, color=t["text"],
                 va="center", arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.9, shrinkA=3, shrinkB=5, relpos=(0, 0.5)))

    # Panel B: CDF of answers to first BH-significant detection.
    x, lo, hi = d["cdf_band"]
    ax2.fill_between(x, lo, hi, color=t["target"], alpha=t["band_alpha"], linewidth=0, step=None)
    cm = d["cdf_med"]
    ax2.plot(cm[:, 0], cm[:, 1], color=t["target"], lw=2.2, solid_capstyle="round")
    q_all = int(cm[np.argmax(cm[:, 1] >= 1.0), 0])
    ax2.axvline(q_all, color=t["ref"], lw=1.1, ls=(0, (2, 3)))
    ax2.set_xlim(1, 40)
    ax2.set_ylim(0, 1.04)
    ax2.set_xticks([1, 10, 20, 30, 40])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_xlabel("audited answers")
    ax2.set_ylabel("target detected (BH)")
    _style_axes(ax2, t)
    _title(ax2, "Fast, reliable detection",
           "median share of audits flagged (BH), band: IQR", t)
    ax2.text(q_all + 1.0, 0.42, f"median reaches 100%\nat q = {q_all}", color=t["text"],
             fontsize=9.5, ha="left", va="center")

    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.15)
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


def comparison_figure(theme: str, out: Path) -> None:
    t = THEMES[theme]
    points, curves = load_frontier()
    order = ["DirBucket", "KGW", "SemStamp"]
    labels = {"DirBucket": "DirBucket (ours)", "KGW": "KGW-adapted", "SemStamp": "SemStamp-adapted"}
    colors = {"DirBucket": t["target"], "KGW": t["kgw"], "SemStamp": t["sem"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7), dpi=200,
                                   gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.28))
    fig.patch.set_facecolor(t["bg"])

    # Panel A: grouped bars, target vs max non-target pooled Z (mean, 95% CI).
    xpos = np.arange(len(order), dtype=float)
    w = 0.36
    err = dict(ecolor=t["muted"], elinewidth=1.0, capsize=3)
    for xi, m in zip(xpos, order):
        p = points[m]
        ax1.bar(xi - w / 2 - 0.02, p["target"], width=w, color=colors[m], yerr=p["target_ci"],
                error_kw=err, zorder=3)
        ax1.bar(xi + w / 2 + 0.02, p["non"], width=w, color=colors[m], alpha=0.35,
                yerr=p["non_ci"], error_kw=err, zorder=3)
        for dx, val, ci, style in ((-w / 2 - 0.02, p["target"], p["target_ci"], dict(color=t["text"], fontweight="bold")),
                                   (w / 2 + 0.02, p["non"], p["non_ci"], dict(color=t["muted"]))):
            # label above a positive bar, below a negative one
            y, va = (val + ci + 0.35, "bottom") if val >= 0 else (val - ci - 0.3, "top")
            ax1.text(xi + dx, y, f"{val:.1f}", ha="center", va=va, fontsize=9.5, **style)
    ax1.axhline(4, color=t["ref"], lw=1.1, ls=(0, (5, 4)), zorder=2)
    ax1.axhline(0, color=t["axis"], lw=0.9, zorder=4)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([labels[m].replace("-adapted", "") for m in order], fontsize=10.5,
                        color=t["text"])
    ax1.set_xlim(-0.6, len(order) - 0.4)
    ax1.set_ylim(-2.6, 14.5)
    ax1.set_yticks([0, 4, 8, 12])
    ax1.set_ylabel("mean pooled Z")
    _style_axes(ax1, t)
    ax1.grid(False, axis="x")
    ax1.tick_params(axis="x", length=0)
    ax1.spines["bottom"].set_visible(False)
    ax1.text(len(order) - 0.42, 4.3, "Z = 4", color=t["muted"], fontsize=9, ha="right", va="bottom")
    from matplotlib.patches import Patch
    leg = ax1.legend(handles=[Patch(color=t["muted"], label="target provider"),
                              Patch(color=t["muted"], alpha=0.35, label="max non-target")],
                     loc="upper right", frameon=False, fontsize=9.5, handlelength=1.1,
                     handleheight=0.8, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    _title(ax1, "Strong on the target, silent elsewhere",
           "mean pooled Z with 95% CI", t)

    # Panel B: P(target BH-significant by q).
    for m in order:
        c = curves[m]
        ax2.plot(c[:, 0], c[:, 1], color=colors[m], lw=2.2 if m == "DirBucket" else 1.8,
                 solid_capstyle="round", zorder=4 if m == "DirBucket" else 3)
        ax2.text(c[-1, 0] + 1.5, c[-1, 1], labels[m].replace(" (ours)", "").replace("-adapted", ""), color=t["text"],
                 fontsize=9.5, va="center", ha="left")
    ax2.set_xlim(1, 80)
    ax2.set_ylim(0, 1.04)
    ax2.set_xticks([1, 20, 40, 60, 80])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_xlabel("audited answers")
    ax2.set_ylabel("target detected (BH)")
    _style_axes(ax2, t)
    _title(ax2, "Time to a BH decision", "share of target audits flagged by q answers", t)

    fig.subplots_adjust(left=0.065, right=0.90, top=0.80, bottom=0.12)
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


def main() -> None:
    for theme in THEMES:
        results_figure(theme, HERE / f"results-{theme}.png")
        comparison_figure(theme, HERE / f"comparison-{theme}.png")


if __name__ == "__main__":
    main()
