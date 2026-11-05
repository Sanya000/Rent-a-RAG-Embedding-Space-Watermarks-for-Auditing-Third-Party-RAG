#!/usr/bin/env python3
"""Exporters for paper figures (TikZ) and tables (JSON + printed rows).

Subcommands (run from reproduce/):
  frontier             fig:wm-frontier-z-q2d       -> fig_wm_frontier_z_q2d_tikz.tex
  evidence             fig:evidence-accumulation-main -> fig_evidence_accumulation_main_tikz.tex
  postselection-null   tab:postselection-null      (JSON + printed rows)
  baseline-diagnostics tab:baseline-diagnostics    (JSON + printed rows)
  ortho-table          tab:ortho-ablation          (JSON + printed rows)

All statistics come from the helpers of aggregate_watermark_compare_and_plot.py, so the numbers
match the matplotlib figures; the TikZ templates reproduce the paper's files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402

import aggregate_watermark_compare_and_plot as agg  # noqa: E402

MAIN_ROOT = Path("results/watermark_compare_global_tuned_mT20_kgwn3")
RAG160_ROOT = Path("results/watermark_compare_global_tuned_mT20_kgwn3_rag160")
ORACLE_ROOT = Path("results/watermark_compare_oracle_aligned_baselines")
DIRECTION_ROOT = Path("results/direction_compare")
FIGS = Path("results/figs")
DIM = 768  # all-mpnet-base-v2


def _f(x: float, nd: int = 12) -> str:
    return f"{float(x):.{nd}f}"


def _coords(xs: Sequence[float], ys: Sequence[float], per_line: int = 8) -> str:
    pts = [f"({int(x)},{_f(y)})" for x, y in zip(xs, ys)]
    return "\n".join("    " + " ".join(pts[i : i + per_line]) for i in range(0, len(pts), per_line))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] {path}")


# --------------------------------------------------------------------------------------------
# fig:wm-frontier-z-q2d
# --------------------------------------------------------------------------------------------
_FRONTIER_STYLE = {
    "dirbucket_naive": ("DBBlue", "DirBucket", "4.2pt", "1.8pt"),
    "kgw_ward": ("KGWOrange", "KGW", "3.5pt", "1.5pt"),
    "semstamp": ("SemTeal", "SemStamp", "3.5pt", "1.5pt"),
}


def frontier_tikz(results_root: Path) -> str:
    methods = [s for s in _FRONTIER_STYLE if (results_root / s).exists()]
    summ = agg._wm_frontier_summary(results_root, methods)
    methods = [s for s in methods if s in summ]
    xs, curves, max_q = agg._target_bh_cdf_from_rag_traj(results_root, methods)
    ys = [summ[s]["target_z_mean"] for s in methods]
    ymin, ymax = min(ys) - 1.2, max(ys) + 1.8
    xmin, xmax = -1.0, 4.6
    yticks = ",".join(str(v) for v in range(2 * math.ceil(ymin / 2), int(math.floor(ymax / 2)) * 2 + 1, 2))
    xticks = ",".join(str(v) for v in range(10, max_q + 1, 10))
    out: List[str] = [
        "% Auto-generated from results by a script.",
        "% Figure: fig_wm_frontier_z_q2d converted to TikZ/PGFPlots.",
        "\\begin{tikzpicture}",
        "\\definecolor{DBBlue}{HTML}{1F77B4}",
        "\\definecolor{KGWOrange}{HTML}{FF7F0E}",
        "\\definecolor{SemTeal}{HTML}{17BECF}",
        "\\definecolor{RegGreen}{HTML}{59A14F}",
        "\\definecolor{RegRed}{HTML}{E15759}",
        "\\definecolor{RegYellow}{HTML}{F2CF5B}",
        "\\pgfplotsset{compat=1.18}",
        "\\begin{groupplot}[",
        "  group style={group size=1 by 2, vertical sep=1.60cm},",
        "  tick label style={font=\\scriptsize},",
        "  label style={font=\\small},",
        "  title style={font=\\small},",
        "  grid=both,",
        "  grid style={dotted, draw=gray!35},",
        "  major grid style={dotted, draw=gray!35},",
        "  minor tick num=0,",
        "  line width=0.9pt",
        "]",
        "\\nextgroupplot[",
        "  width=0.88\\linewidth,",
        "  height=3.45cm,",
        f"  xmin={xmin:.6f}, xmax={xmax:.6f},",
        f"  ymin={ymin:.6f}, ymax={ymax:.6f},",
        "  xtick={-1,0,1,2,3,4},",
        f"  ytick={{{yticks}}},",
        "  ytick pos=left,",
        "  xtick pos=bottom,",
        "  xlabel={Max non-target pooled z (mean)},",
        "  ylabel={Target pooled z (mean)},",
        "  title={A) z-Power vs z-Specificity Frontier},",
        "  clip=false,",
        "  legend to name=wmfrontierlegend,",
        "  legend columns=1,",
        "  legend style={",
        "    font=\\scriptsize,",
        "    draw=gray!60,",
        "    fill=white,",
        "    cells={anchor=west}",
        "  }",
        "]",
        f"\\path[fill=RegYellow, opacity=0.20] (axis cs:{xmin:.6f},{ymin:.6f}) rectangle (axis cs:4.0,4.0);",
        f"\\path[fill=RegRed, opacity=0.20] (axis cs:4.0,{ymin:.6f}) rectangle (axis cs:{xmax:.6f},{ymax:.6f});",
        f"\\path[fill=RegGreen, opacity=0.26] (axis cs:{xmin:.6f},4.0) rectangle (axis cs:{2.0:.6f},{ymax:.6f});",
        f"\\shade[left color=RegGreen!80,right color=RegRed!80,opacity=0.22] (axis cs:{2.0:.6f},4.0) rectangle (axis cs:{4.0:.6f},{ymax:.6f});",
        f"\\addplot[black!65, dashed, line width=0.9pt, forget plot] coordinates {{(4.0,{ymin:.6f}) (4.0,{ymax:.6f})}};",
        f"\\addplot[black!65, dashed, line width=0.9pt, forget plot] coordinates {{({xmin:.6f},4.0) ({xmax:.6f},4.0)}};",
        f"\\node[anchor=west, align=left, text=green!45!black, font=\\scriptsize\\bfseries] at (axis cs:{-0.7:.6f},{5.5:.6f}) {{Desired region\\\\(target z>4,\\\\non-target z<4)}};",
        f"\\node[anchor=west, text=brown!70!black, font=\\scriptsize\\bfseries] at (axis cs:{0.9:.6f},{0.45:.6f}) {{Insufficient attribution}};",
        f"\\node[anchor=center, rotate=90, text=red!70!black, font=\\scriptsize\\bfseries, align=center] at (axis cs:{4.33:.6f},{ymin + 0.72 * (ymax - ymin):.6f}) {{False-positive\\\\region}};",
    ]
    for s in methods:
        color, label, msize, _ = _FRONTIER_STYLE[s]
        m = summ[s]
        out += [
            "\\addplot[",
            "  only marks,",
            "  mark=*,",
            f"  mark size={msize},",
            f"  color={color},",
            f"  mark options={{draw=black, fill={color}}},",
            "  error bars/.cd, x dir=both, x explicit, y dir=both, y explicit,",
            f"  error bar style={{line width=0.9pt, draw={color}}}",
            f"] coordinates {{({_f(m['max_non_z_mean'])},{_f(m['target_z_mean'])}) +- ({_f(m['max_non_z_ci'])},{_f(m['target_z_ci'])})}};",
            f"\\addlegendentry{{{label}}}",
        ]
    out += [
        "\\nextgroupplot[",
        "  width=0.73\\linewidth,",
        "  height=2.55cm,",
        "  xshift=-0.05\\linewidth,",
        f"  xmin=1, xmax={max_q},",
        "  ymin=0.0, ymax=1.02,",
        f"  xtick={{{xticks}}},",
        "  ytick={0,0.2,0.4,0.6,0.8,1.0},",
        "  ytick pos=left,",
        "  xtick pos=bottom,",
        "  xlabel={\\# audited answers (q)},",
        "  ylabel={P(target BH sig by q)},",
        "  title={B) Time to BH Decision}",
        "]",
    ]
    for s in methods:
        color, _, _, lw = _FRONTIER_STYLE[s]
        pts = " ".join(f"({int(q)},{_f(p)})" for q, p in zip(xs, curves[s]))
        out.append(f"\\addplot[{color}, line width={lw}] coordinates {{{pts}}};")
    out += [
        "\\end{groupplot}",
        "\\node[anchor=west] at ($(group c1r2.east)+(0.08cm,0cm)$) {\\pgfplotslegendfromname{wmfrontierlegend}};",
        "\\end{tikzpicture}",
    ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------
# fig:evidence-accumulation-main
# --------------------------------------------------------------------------------------------
def _bh_sig_eligible_only(z: List[float], pooled_T: List[int], *, min_pooled_T: int, alpha: float = 0.05) -> List[bool]:
    """BH over the providers with T >= T_min only (m = #eligible), all others not significant."""
    if not isinstance(z, list) or not isinstance(pooled_T, list) or len(z) != len(pooled_T):
        return []
    idx = [i for i, t in enumerate(pooled_T) if int(t) >= int(min_pooled_T)]
    sig = [False] * len(z)
    if idx:
        for i, s in zip(idx, agg._bh_mask([agg._norm_sf(float(z[i])) for i in idx], alpha=alpha)):
            sig[i] = bool(s)
    return sig


def _capture_evidence_series(results_root: Path, bh_family: str = "eligible") -> Dict[str, np.ndarray]:
    """Run agg.plot_evidence_accumulation_main and record the data it draws.

    bh_family="eligible" applies BH over the T_min-eligible providers only (the paper figure);
    "all" keeps the aggregator/runner rule (ineligible providers enter BH with p=1, m=P).
    """
    calls: List[Tuple[str, Tuple[Any, ...]]] = []
    orig_fill, orig_plot, orig_save = Axes.fill_between, Axes.plot, agg._save_fig
    orig_bh = agg._bh_sig_from_snapshot

    def fill(self: Axes, x: Any, y1: Any, y2: Any = 0, *a: Any, **k: Any) -> Any:
        calls.append(("fill", (np.asarray(x, float), np.asarray(y1, float), np.asarray(y2, float))))
        return orig_fill(self, x, y1, y2, *a, **k)

    def plot(self: Axes, *args: Any, **k: Any) -> Any:
        calls.append(("plot", (np.asarray(args[0], float), np.asarray(args[1], float))))
        return orig_plot(self, *args, **k)

    Axes.fill_between, Axes.plot = fill, plot  # type: ignore[method-assign]
    agg._save_fig = lambda *a, **k: None  # type: ignore[assignment]
    if bh_family == "eligible":
        agg._bh_sig_from_snapshot = _bh_sig_eligible_only  # type: ignore[assignment]
    elif bh_family != "all":
        raise ValueError(f"unknown bh_family {bh_family!r}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            agg.plot_evidence_accumulation_main(results_root, Path(tmp), ["dirbucket_naive"])
    finally:
        Axes.fill_between, Axes.plot, agg._save_fig = orig_fill, orig_plot, orig_save  # type: ignore[method-assign]
        agg._bh_sig_from_snapshot = orig_bh  # type: ignore[assignment]
    kinds = [c[0] for c in calls]
    if kinds != ["fill", "plot", "fill", "plot", "fill", "plot"]:
        raise RuntimeError(f"unexpected drawing sequence in plot_evidence_accumulation_main: {kinds}")
    (x, q1_t, q3_t), (_, med_t), (_, q1_n, q3_n), (_, med_n), (_, lo, hi), (_, cdf) = [c[1] for c in calls]
    return {"x": x, "q1_t": q1_t, "q3_t": q3_t, "med_t": med_t, "q1_n": q1_n, "q3_n": q3_n,
            "med_n": med_n, "cdf_lo": lo, "cdf_hi": hi, "cdf": cdf}


def evidence_tikz(results_root: Path, ymax: Optional[float] = None, bh_family: str = "eligible") -> str:
    d = _capture_evidence_series(results_root, bh_family)
    x = d["x"].astype(int)
    max_q = int(x[-1])
    if ymax is None:
        ymax = math.ceil(2.0 * (float(np.nanmax(d["q3_t"])) + 0.5)) / 2.0
    full = [int(q) for q, p in zip(x, d["cdf"]) if p >= 0.999999]
    q_full = full[0] if full else None
    xticks = ",".join(str(v) for v in range(20, max_q + 1, 20))

    def band(lo: np.ndarray, hi: np.ndarray) -> str:
        xs = list(x) + list(x[::-1])
        ys = list(lo) + list(hi[::-1])
        return _coords(xs, ys)

    flat_non = bool(np.allclose(d["med_n"], d["med_n"][0]) and np.allclose(d["q1_n"], d["q3_n"]))
    out = [
        "% Auto-generated from results by script.",
        "% Figure: fig_evidence_accumulation_main converted to TikZ/PGFPlots.",
        "\\begin{tikzpicture}",
        "\\definecolor{DBBlue}{HTML}{1F77B4}",
        "\\definecolor{NonRed}{HTML}{E15759}",
        "\\definecolor{BHGreen}{HTML}{2E8B57}",
        "\\pgfplotsset{compat=1.18}",
        "\\begin{axis}[",
        "  width=\\linewidth,",
        "  height=3.85cm,",
        f"  xmin=1, xmax={max_q},",
        f"  ymin=0.0, ymax={ymax},",
        f"  xtick={{{xticks}}},",
        "  xlabel={\\# audited answers},",
        "  ylabel={Pooled z},",
        "  ytick pos=left,",
        "  tick label style={font=\\scriptsize},",
        "  label style={font=\\small},",
        "  xlabel style={yshift=0.6pt},",
        "  grid=both,",
        "  grid style={dotted, draw=gray!35},",
        "  major grid style={dotted, draw=gray!35},",
        "  minor tick num=0,",
        "  line width=0.9pt",
        "]",
        "\\addplot[draw=none, fill=DBBlue, fill opacity=0.22] coordinates {",
        band(d["q1_t"], d["q3_t"]),
        "} -- cycle;",
        "\\addplot[DBBlue, line width=2.4pt] coordinates {",
        _coords(x, d["med_t"]),
        "};",
    ]
    if flat_non:
        out.append(f"\\addplot[NonRed, line width=2.0pt] coordinates {{(1,{_f(d['med_n'][0])}) ({max_q},{_f(d['med_n'][-1])})}};")
    else:
        out += [
            "\\addplot[draw=none, fill=NonRed, fill opacity=0.20] coordinates {",
            band(d["q1_n"], d["q3_n"]),
            "} -- cycle;",
            "\\addplot[NonRed, line width=2.0pt] coordinates {",
            _coords(x, d["med_n"]),
            "};",
        ]
    out.append(f"\\addplot[black!65, dashed, line width=0.95pt] coordinates {{(1,4.0) ({max_q},4.0)}};")
    if q_full is not None:
        out.append(f"\\addplot[black!70, dotted, line width=1.05pt] coordinates {{({q_full},0.0) ({q_full},{ymax})}};")
    out += [
        "\\end{axis}",
        "",
        "\\begin{axis}[",
        "  width=\\linewidth,",
        "  height=3.85cm,",
        f"  xmin=1, xmax={max_q},",
        "  ymin=0.0, ymax=1.02,",
        "  axis x line=none,",
        "  axis y line*=right,",
        "  ytick={0,0.2,0.4,0.6,0.8,1.0},",
        "  ytick pos=right,",
        "  ylabel={P(target BH sig by q)},",
        "  ylabel style={at={(axis description cs:1.18,0.5)},anchor=south},",
        "  yticklabel style={font=\\scriptsize},",
        "  tick label style={font=\\scriptsize},",
        "  label style={font=\\small},",
        "  minor tick num=0,",
        "  line width=0.9pt",
        "]",
        "\\addplot[draw=none, fill=BHGreen, fill opacity=0.17] coordinates {",
        band(d["cdf_lo"], d["cdf_hi"]),
        "} -- cycle;",
        "\\addplot[BHGreen!90!black, line width=2.2pt] coordinates {",
        _coords(x, d["cdf"]),
        "};",
        "\\end{axis}",
        "",
        "\\node[anchor=north, inner sep=0pt, outer sep=0pt] at ($(current axis.south)+(0,-0.98cm)$) {",
        "  \\begin{tikzpicture}",
        "    \\matrix[column sep=0.22cm, row sep=0.02cm] {",
        "      \\draw[DBBlue, line width=2.4pt] (0,0) -- (0.45,0); & \\node[font=\\scriptsize] {Target z (med+IQR)}; &",
        "      \\draw[NonRed, line width=2.0pt] (0,0) -- (0.45,0); & \\node[font=\\scriptsize] {Max non-target z}; \\\\",
        "      \\draw[BHGreen!90!black, line width=2.2pt] (0,0) -- (0.45,0); & \\node[font=\\scriptsize] {Target BH sig. prob. (med+IQR)}; &",
        "      \\draw[black!65, dashed, line width=0.95pt] (0,0) -- (0.45,0); & \\node[font=\\scriptsize] {$z=4$ ref.}; \\\\",
        "      \\draw[black!70, dotted, line width=1.05pt] (0,0) -- (0.45,0); & \\node[font=\\scriptsize] {$q$ at CDF=1}; &",
        "      \\node{}; & \\node{}; \\\\",
        "    };",
        "  \\end{tikzpicture}",
        "};",
        "\\end{tikzpicture}",
    ]
    return "\n".join(out) + "\n"


def evidence_headline(results_root: Path, bh_family: str = "eligible") -> Dict[str, Any]:
    """Sec. 5.4 / abstract numbers: first q with median target z >= 4, q at which all audits are detected."""
    d = _capture_evidence_series(results_root, bh_family)
    x = d["x"].astype(int)
    first_med4 = next((int(q) for q, m in zip(x, d["med_t"]) if m >= 4.0), None)
    full = next((int(q) for q, p in zip(x, d["cdf"]) if p >= 0.999999), None)
    return {
        "audits": len(agg.find_rag_multi_files(results_root, "dirbucket_naive")),
        "first_q_median_target_z_ge_4": first_med4,
        "q_all_detected": full,
        "final_detected_fraction": float(d["cdf"][-1]),
        "max_median_non_target_z": float(np.nanmax(d["med_n"])),
    }


# --------------------------------------------------------------------------------------------
# tab:postselection-null
# --------------------------------------------------------------------------------------------
def _audit_files(root: Path, kind: str) -> List[Path]:
    pat = re.compile(rf"{kind}_t\d+_seed\d+\.json$")
    return sorted(p for p in (root / "dirbucket_naive").glob(f"{kind}_t*_seed*.json") if pat.search(p.name))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _wrongkey_increments(trial_log: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Per-query wrong-key support T and summed projections (mean_proj * T), shape (q, P)."""
    rows = [r for r in _read_jsonl(trial_log) if r.get("wrongkey")]
    T = np.array([r["wrongkey"]["T"] for r in rows], dtype=float)
    S = np.array([r["wrongkey"]["mean_proj"] for r in rows], dtype=float) * T
    return T, S


def _any_bh(T: np.ndarray, S: np.ndarray, min_pooled_T: int) -> bool:
    z = [float(s / t * math.sqrt(DIM * t)) if t > 0 else 0.0 for t, s in zip(T, S)]
    return any(agg._bh_sig_from_snapshot(z, [int(t) for t in T], min_pooled_T=min_pooled_T))


def postselection_null(root: Path, n_resamples: int = 300, block: int = 5, seed: int = 0) -> Dict[str, Any]:
    coop, unrel = _audit_files(root, "sanity_coop"), _audit_files(root, "sanity_unrel")
    wk_support, wk_p, wk_bh = [], [], 0
    for fp in coop:
        r = agg.load_json(fp)
        tp, mt, wk = int(r["target_provider"]), int(r["min_pooled_T"]), r["wrongkey"]
        T_t = int(wk["pooled_T"][tp])
        wk_support.append(T_t)
        wk_p.append(agg._norm_sf(float(wk["z"][tp])) if T_t >= mt else 1.0)
        wk_bh += int(any(agg._bh_sig_from_snapshot(wk["z_raw"], wk["pooled_T"], min_pooled_T=mt)))
    un_bh, un_support = 0, []
    for fp in unrel:
        r = agg.load_json(fp)
        un_bh += int(any(bool(b) for b in r.get("bh_sig") or []))
        un_support.append(int(max(r["pooled_T"])))
    out: Dict[str, Any] = {
        "wrong_key": {
            "audits": len(coop),
            "any_bh": wk_bh,
            "support_median": float(np.median(wk_support)) if wk_support else None,
            "support_min": int(min(wk_support)) if wk_support else None,
            "support_max": int(max(wk_support)) if wk_support else None,
            "P(p<=0.05)": float(np.mean(np.array(wk_p) <= 0.05)) if wk_p else None,
            "P(p<=0.01)": float(np.mean(np.array(wk_p) <= 0.01)) if wk_p else None,
        },
        "unrelated": {
            "audits": len(unrel),
            "any_bh": un_bh,
            "max_support": int(max(un_support)) if un_support else None,
        },
    }
    logs = [fp.with_suffix(".trials.jsonl") for fp in coop]
    if not logs or not all(p.exists() for p in logs):
        out["trajectory_and_resampling"] = "skipped: needs the sanity_coop *.trials.jsonl logs (bundle trial_logs_main)"
        return out

    ever = 0
    audits: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for fp, log in zip(coop, logs):
        r = agg.load_json(fp)
        mt = int(r["min_pooled_T"])
        for row in _read_jsonl(log):
            c = row.get("cumulative") or {}
            if c.get("z_wrong") and any(agg._bh_sig_from_snapshot(c["z_wrong"], c["wrong_pooled_T"], min_pooled_T=mt)):
                ever += 1
                break
        T, S = _wrongkey_increments(log)
        audits.append((T, S, mt))
    out["wrong_key"]["ever_bh_along_trajectory"] = ever

    # Resampling diagnostics: circular query-block bootstrap (block size 5) and per-query
    # provider-label permutation of the wrong-key increments.
    rng = np.random.default_rng(seed)
    n = boot = perm = perm_eligible = 0
    for T, S, mt in audits:
        q = T.shape[0]
        n_blocks = int(math.ceil(q / block))
        for _ in range(n_resamples):
            starts = rng.integers(0, q, size=n_blocks)
            idx = np.concatenate([(np.arange(s, s + block) % q) for s in starts])[:q]
            boot += int(_any_bh(T[idx].sum(0), S[idx].sum(0), mt))
            perms = np.array([rng.permutation(T.shape[1]) for _ in range(q)])
            Tp = np.take_along_axis(T, perms, 1).sum(0)
            Sp = np.take_along_axis(S, perms, 1).sum(0)
            perm += int(_any_bh(Tp, Sp, mt))
            perm_eligible += int(any(int(t) >= mt for t in Tp))
            n += 1
    out["resampling_reconstructed"] = {
        "note": "circular query-block bootstrap and per-query provider-label permutation (Monte Carlo)",
        "resamples": n,
        "block_bootstrap_any_bh": boot / n,
        "label_permutation_any_bh": perm / n,
        "label_permutation_any_bh_given_eligible": perm / max(1, perm_eligible),
        "seed": seed,
    }
    return out


# --------------------------------------------------------------------------------------------
# tab:baseline-diagnostics
# --------------------------------------------------------------------------------------------
def _target_and_max_non(r: Dict[str, Any]) -> Tuple[float, float]:
    tp, z = int(r.get("target_provider", 0)), r["z"]
    return float(z[tp]), float(max(float(z[i]) for i in range(len(z)) if i != tp))


def _mean_pair(rows: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    pairs = [_target_and_max_non(r) for r in rows]
    if not pairs:
        return None
    return float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))


def _seed_ok(path: Path, seeds: Sequence[int]) -> bool:
    m = re.search(r"seed(\d+)", path.name)
    return bool(m) and int(m.group(1)) in set(seeds)


def baseline_diagnostics(main_root: Path, oracle_root: Path, seeds: Sequence[int] = (1, 2, 3, 4, 5)) -> Dict[str, Any]:
    """Mean gated target z and mean max non-target z per setting (cooperative sanity, style, oracle)."""
    out: Dict[str, Any] = {}
    for scheme in ("dirbucket_naive", "kgw_ward", "semstamp"):
        row: Dict[str, Any] = {}
        coop = [agg.load_json(p) for p in sorted((main_root / scheme).glob("sanity_coop_t*_seed*.json")) if _seed_ok(p, seeds)]
        row["cooperative_sanity"] = _mean_pair(coop)
        per_style: Dict[str, List[Dict[str, Any]]] = {}
        for p in sorted((main_root / scheme).glob("style_sweep_seed*.json")):
            if not _seed_ok(p, seeds):
                continue
            for res in agg.load_json(p).get("runs") or []:
                per_style.setdefault(str(res.get("answer_style")), []).append(res)
        style_means = {s: _mean_pair(v) for s, v in per_style.items()}
        if style_means:
            vals = [v for v in style_means.values() if v is not None]
            row["style_sweep_avg"] = (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
            row["style_sweep_per_style"] = style_means
        oracle = [agg.load_json(p) for p in agg.find_rag_multi_files(oracle_root, scheme)]
        row["oracle_aligned"] = _mean_pair(oracle)
        out[scheme] = row
    return out


# --------------------------------------------------------------------------------------------
# tab:ortho-ablation
# --------------------------------------------------------------------------------------------
def ortho_table(root: Path) -> Dict[str, Any]:
    """Naive vs ortho rag_multi metrics, using the paper's seed conventions."""
    runs = {s: sorted((root / s).glob("rag_multi_seed*.json")) for s in ("naive", "ortho")}
    common = sorted(set(p.name for p in runs["naive"]) & set(p.name for p in runs["ortho"]))
    out: Dict[str, Any] = {"common_seed_files": common}
    for scheme, files in runs.items():
        rs_all = [agg.load_json(p) for p in files]
        rs = [agg.load_json(root / scheme / name) for name in common]
        tp = [int(r.get("target_provider", 0)) for r in rs]
        crossed = [r["crossed_at"][int(r.get("target_provider", 0))] for r in rs_all]
        out[scheme] = {
            "target_pooled_z": float(np.mean([r["z"][t] for r, t in zip(rs, tp)])),
            "target_support": float(np.mean([r["pooled_T"][t] for r, t in zip(rs, tp)])),
            "assign_rate": float(np.mean([r["align"]["assign_rate"] for r in rs])),
            "queries_to_cross_z4_median_all_seeds": float(np.median([c for c in crossed if c is not None])),
            "max_non_target_z_mean": float(np.mean([max(r["z"][i] for i in range(len(r["z"])) if i != t) for r, t in zip(rs, tp)])),
            "seeds_all": len(rs_all),
        }
    return out


# --------------------------------------------------------------------------------------------
def _dump(obj: Any, out: Optional[Path]) -> None:
    text = json.dumps(obj, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else str(x))
    print(text)
    if out is not None:
        _write(out, text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("frontier", help="fig:wm-frontier-z-q2d")
    p.add_argument("--results_root", type=Path, default=MAIN_ROOT)
    p.add_argument("--out", type=Path, default=FIGS / MAIN_ROOT.name / "fig_wm_frontier_z_q2d_tikz.tex")

    p = sub.add_parser("evidence", help="fig:evidence-accumulation-main (+ Sec. 5.4 headline numbers)")
    p.add_argument("--results_root", type=Path, default=RAG160_ROOT)
    p.add_argument("--out", type=Path, default=FIGS / RAG160_ROOT.name / "fig_evidence_accumulation_main_tikz.tex")
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--bh_family", choices=["eligible", "all"], default="eligible",
                   help="eligible: BH over T_min-eligible providers (paper TikZ); all: runner rule (m=P)")

    p = sub.add_parser("postselection-null", help="tab:postselection-null")
    p.add_argument("--results_root", type=Path, default=MAIN_ROOT)
    p.add_argument("--n_resamples", type=int, default=300, help="per audit (paper: 300 x 120 = 36,000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("baseline-diagnostics", help="tab:baseline-diagnostics")
    p.add_argument("--results_root", type=Path, default=MAIN_ROOT)
    p.add_argument("--oracle_root", type=Path, default=ORACLE_ROOT)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("ortho-table", help="tab:ortho-ablation")
    p.add_argument("--results_root", type=Path, default=DIRECTION_ROOT)
    p.add_argument("--out", type=Path, default=None)

    args = ap.parse_args()
    handlers: Dict[str, Callable[[], None]] = {
        "frontier": lambda: _write(args.out, frontier_tikz(args.results_root)),
        "evidence": lambda: (_write(args.out, evidence_tikz(args.results_root, args.ymax, args.bh_family)),
                             _dump(evidence_headline(args.results_root, args.bh_family), None)),
        "postselection-null": lambda: _dump(
            postselection_null(args.results_root, n_resamples=args.n_resamples, seed=args.seed), args.out),
        "baseline-diagnostics": lambda: _dump(baseline_diagnostics(args.results_root, args.oracle_root), args.out),
        "ortho-table": lambda: _dump(ortho_table(args.results_root), args.out),
    }
    handlers[args.cmd]()


if __name__ == "__main__":
    main()
