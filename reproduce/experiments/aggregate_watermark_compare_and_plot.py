"""Aggregate a watermark_compare results root into the Sec. 5 / appendix figures (PNG/PDF).

Paper figures produced here: fig_wm_frontier_z_q2d (fig:wm-frontier-z-q2d),
fig_evidence_accumulation_main (fig:evidence-accumulation-main, run on the rag160 root),
fig_benchmark_task_difficulty.pdf (fig:benchmark_difficulty), fig_null_calibration,
fig_style_mixing_robustness_main and fig_nonwm_critical_failure (on the nonwm_natural_full root).
TikZ versions come from export_*_tikz.py. Trial-log based panels need the *.trials.jsonl files.
find_rag_trial_logs pools every rag_multi*.trials.jsonl of a scheme directory, i.e. both the
untargeted rag_multi_seedNN logs and the per-target rag_multi_t{T}_seedNN logs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
import textwrap


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_ci(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    m = float(np.nanmean(arr))
    if len(arr) <= 1:
        return (m, 0.0)
    s = float(np.nanstd(arr, ddof=1))
    hw = 1.96 * s / math.sqrt(len(arr))
    return (m, hw)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def find_files(results_root: Path, scheme: str, pattern: str) -> List[Path]:
    d = results_root / scheme
    if not d.exists():
        return []
    return sorted(d.glob(pattern))


def find_rag_multi_files(results_root: Path, scheme: str) -> List[Path]:
    """
    Backward-compatible loader for rag_multi outputs.

    Old format:  rag_multi_seedSS.json              (single target provider, usually t=0)
    New format:  rag_multi_tT_seedSS.json           (all target providers)

    If both exist for the same seed and a t=0 file exists, drop the old file to avoid
    double-counting t=0 after backfilling newer per-target runs into older result dirs.
    """
    d = results_root / scheme
    if not d.exists():
        return []

    t_files = sorted(d.glob("rag_multi_t*_seed*.json"))
    old_files = sorted(d.glob("rag_multi_seed*.json"))
    if not t_files:
        return old_files

    t_pat = re.compile(r"rag_multi_t(\d+)_seed(\d+)\.json$")
    old_pat = re.compile(r"rag_multi_seed(\d+)\.json$")
    have_t0_by_seed: set[int] = set()
    have_any_t_by_seed: set[int] = set()
    for fp in t_files:
        m = t_pat.match(fp.name)
        if not m:
            continue
        t = int(m.group(1))
        seed = int(m.group(2))
        have_any_t_by_seed.add(seed)
        if t == 0:
            have_t0_by_seed.add(seed)

    keep_old: List[Path] = []
    for fp in old_files:
        m = old_pat.match(fp.name)
        if not m:
            keep_old.append(fp)
            continue
        seed = int(m.group(1))
        if seed in have_t0_by_seed:
            continue
        # Keep legacy file if this seed has no per-target t0 file yet (including partial backfills).
        keep_old.append(fp)
    return sorted(t_files + keep_old)


def find_rag_absent_files(results_root: Path, scheme: str) -> List[Path]:
    d = results_root / scheme
    if not d.exists():
        return []
    return sorted(d.glob("rag_absent_t*_seed*.json"))


def find_rag_trial_logs(results_root: Path, scheme: str) -> List[Path]:
    d = results_root / scheme
    if not d.exists():
        return []
    return sorted(d.glob("rag_multi*.trials.jsonl"))


def _save_fig(
    fig: plt.Figure,
    out_base: Path,
    tight_rect: Tuple[float, float, float, float] | None = None,
) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(out_base.with_suffix(".png"), dpi=240)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[OK] {out_base.with_suffix('.png')}")
    print(f"[OK] {out_base.with_suffix('.pdf')}")


def _scheme_label(scheme: str) -> str:
    if scheme == "dirbucket_naive":
        return "DirBucket (naive)"
    if scheme == "kgw_ward":
        return "KGW / WARD"
    if scheme == "semstamp":
        return "SemStamp"
    if scheme == "sim_attr":
        return "Similarity Attribution"
    if scheme == "overlap_attr":
        return "Overlap / Quote Attribution"
    if scheme == "entail_attr":
        return "Entailment Attribution"
    return scheme


def _scheme_short_label(scheme: str) -> str:
    if scheme == "dirbucket_naive":
        return "DirBucket"
    if scheme == "kgw_ward":
        return "KGW"
    if scheme == "semstamp":
        return "SemStamp"
    if scheme == "sim_attr":
        return "SimAttr"
    if scheme == "overlap_attr":
        return "Overlap"
    if scheme == "entail_attr":
        return "Entail"
    return scheme


def _load_corpus_text_map(work_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    p = work_dir / "corpus.jsonl"
    if not p.exists():
        return out

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            did = str(row.get("id") or "").strip()
            if not did:
                continue
            txt = str(
                row.get("text_wm")
                or row.get("text_wm_semstamp")
                or row.get("text_wm_kgw")
                or row.get("text")
                or ""
            )
            if txt:
                out[did] = txt
    return out


def _load_manifest_source_map(manifest_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not manifest_path.exists():
        return out
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            did = str(row.get("id") or "").strip()
            if not did:
                continue
            txt = str(row.get("text_src") or row.get("text") or "").strip()
            if txt:
                out[did] = txt
    return out


def _simple_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _simple_sentence_spans(text: str) -> List[str]:
    if not text:
        return []
    # Lightweight sentence split for plotting diagnostics (not used for scoring).
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.search(r"[A-Za-z0-9]", p):
            out.append(p)
    return out


def _estimate_tokens_per_sentence_from_trial_outputs(
    results_root: Path,
    schemes: List[str],
    *,
    max_records: int = 5000,
) -> float:
    """
    Estimate an answer-level tokens-per-sentence factor from saved trial JSONL outputs.

    Uses `answer_tokens_est` when available (already logged by the experiment runner)
    and a lightweight sentence split on the answer text. This is only for visual x-axis
    normalization in the RAG FP anatomy plot.
    """
    total_tok = 0.0
    total_sent = 0.0
    seen = 0
    globs = ("rag_multi*.trials.jsonl", "style_sweep*.trials.*.jsonl")
    for s in schemes:
        d = results_root / s
        if not d.exists():
            continue
        files: List[Path] = []
        for pat in globs:
            files.extend(sorted(d.glob(pat)))
        for fp in sorted(files):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        if seen >= max_records:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        ans = str(row.get("answer") or "")
                        if not ans:
                            continue
                        sents = _simple_sentence_spans(ans)
                        n_sent = len(sents)
                        if n_sent <= 0:
                            continue
                        tok_est = row.get("answer_tokens_est")
                        if not isinstance(tok_est, (int, float)) or float(tok_est) <= 0:
                            tok_est = len(_simple_tokens(ans))
                        total_tok += float(tok_est)
                        total_sent += float(n_sent)
                        seen += 1
                if seen >= max_records:
                    break
            except Exception:
                continue
        if seen >= max_records:
            break
    if total_sent <= 0:
        return 18.0
    # Clamp to a reasonable range to avoid pathologies from malformed answers.
    est = float(total_tok / total_sent)
    return min(40.0, max(6.0, est))


def _ngram_set(tokens: List[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def _norm_sf(z: float) -> float:
    # One-sided standard normal survival function using erfc.
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _bh_mask(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    n = len(pvals)
    if n <= 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    thresh_k = -1
    for rank, idx in enumerate(order, start=1):
        if float(pvals[idx]) <= alpha * rank / n:
            thresh_k = rank
    out = [False] * n
    if thresh_k <= 0:
        return out
    cutoff = alpha * thresh_k / n
    for i, p in enumerate(pvals):
        if float(p) <= cutoff:
            out[i] = True
    return out


def _bh_sig_from_snapshot(
    z: List[float],
    pooled_T: List[int],
    *,
    min_pooled_T: int,
    alpha: float = 0.05,
) -> List[bool]:
    if not isinstance(z, list) or not isinstance(pooled_T, list) or len(z) != len(pooled_T):
        return []
    pvals: List[float] = []
    for zi, Ti in zip(z, pooled_T):
        if int(Ti) < int(min_pooled_T):
            pvals.append(1.0)
        else:
            pvals.append(_norm_sf(float(zi)))
    return _bh_mask(pvals, alpha=alpha)


def _target_non_target_stats(r: Dict[str, Any]) -> Dict[str, Any]:
    z = r.get("z") or []
    z_raw = r.get("z_raw") or z
    bh = r.get("bh_sig") or []
    T = r.get("pooled_T") or []
    tp = int(r.get("target_provider", 0))
    if not z or tp < 0 or tp >= len(z):
        return {
            "target_z": float("nan"),
            "target_z_raw": float("nan"),
            "target_bh": False,
            "target_T": 0,
            "max_non_z": float("nan"),
            "max_non_z_raw": float("nan"),
            "max_non_T": 0,
            "any_non_bh": False,
            "max_non_pid": None,
        }
    non = [i for i in range(len(z)) if i != tp]
    if non:
        max_non_pid = max(non, key=lambda i: float(z_raw[i] if i < len(z_raw) else z[i]))
        max_non_z = float(max(float(z[i]) for i in non))
        max_non_z_raw = float(max(float((z_raw[i] if i < len(z_raw) else z[i])) for i in non))
        any_non_bh = bool(any(bool(bh[i]) for i in non if i < len(bh)))
        max_non_T = int(T[max_non_pid]) if max_non_pid < len(T) else 0
    else:
        max_non_pid = None
        max_non_z = float("nan")
        max_non_z_raw = float("nan")
        any_non_bh = False
        max_non_T = 0
    return {
        "target_z": float(z[tp]),
        "target_z_raw": float(z_raw[tp] if tp < len(z_raw) else z[tp]),
        "target_bh": bool(bh[tp]) if tp < len(bh) else False,
        "target_T": int(T[tp]) if tp < len(T) else 0,
        "max_non_z": max_non_z,
        "max_non_z_raw": max_non_z_raw,
        "max_non_T": max_non_T,
        "any_non_bh": any_non_bh,
        "max_non_pid": max_non_pid,
    }


def _is_watermark_scheme(scheme: str) -> bool:
    return scheme in {"dirbucket_naive", "kgw_ward", "semstamp"}


def _is_heuristic_scheme(scheme: str) -> bool:
    return scheme in {"sim_attr", "overlap_attr", "entail_attr"}


def _setting_display_name(setting_key: str) -> str:
    if setting_key == "single_source_sanity":
        return "Single-source\nsanity"
    if setting_key == "mixed_provider_rag":
        return "Mixed-provider\nRAG"
    if setting_key == "style_rewrite_suite":
        return "Style rewrite\nsuite"
    return setting_key


def _collect_primary_screen_stats(results_root: Path, scheme: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {
        "single_source_sanity": [],
        "mixed_provider_rag": [],
        "style_rewrite_suite": [],
    }
    for fp in find_files(results_root, scheme, "sanity_coop_t*_seed*.json"):
        out["single_source_sanity"].append(_target_non_target_stats(load_json(fp)))
    for fp in find_rag_multi_files(results_root, scheme):
        out["mixed_provider_rag"].append(_target_non_target_stats(load_json(fp)))
    for fp in find_files(results_root, scheme, "style_sweep_seed*.json"):
        obj = load_json(fp)
        for run in obj.get("runs", []):
            if isinstance(run, dict):
                out["style_rewrite_suite"].append(_target_non_target_stats(run))
    return out


def _aggregate_screen_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {
            "target_bh_rate": float("nan"),
            "non_target_bh_rate": float("nan"),
            "target_z_mean": float("nan"),
            "max_non_z_mean": float("nan"),
            "n": 0.0,
        }
    t_bh = [1.0 if bool(r.get("target_bh")) else 0.0 for r in records]
    nt_bh = [1.0 if bool(r.get("any_non_bh")) else 0.0 for r in records]
    t_z = [float(r.get("target_z", float("nan"))) for r in records]
    nt_z = [float(r.get("max_non_z", float("nan"))) for r in records]
    return {
        "target_bh_rate": float(np.nanmean(np.array(t_bh, dtype=float))),
        "non_target_bh_rate": float(np.nanmean(np.array(nt_bh, dtype=float))),
        "target_z_mean": float(np.nanmean(np.array(t_z, dtype=float))),
        "max_non_z_mean": float(np.nanmean(np.array(nt_z, dtype=float))),
        "n": float(len(records)),
    }


def _collect_rag_absent_metrics(results_root: Path, scheme: str) -> Dict[str, float]:
    target_bh_vals: List[float] = []
    any_bh_vals: List[float] = []
    target_z_vals: List[float] = []
    max_any_z_vals: List[float] = []
    for fp in find_rag_absent_files(results_root, scheme):
        r = load_json(fp)
        st = _target_non_target_stats(r)
        target_bh_vals.append(1.0 if bool(st["target_bh"]) else 0.0)
        target_z_vals.append(float(st["target_z"]))
        bh = r.get("bh_sig") or []
        if isinstance(bh, list) and bh:
            any_bh_vals.append(1.0 if any(bool(x) for x in bh) else 0.0)
        z = r.get("z") or []
        if isinstance(z, list) and z:
            max_any_z_vals.append(float(np.max(np.array(z, dtype=float))))
    return {
        "target_bh_rate": float(np.nanmean(np.array(target_bh_vals, dtype=float))) if target_bh_vals else float("nan"),
        "any_provider_bh_rate": float(np.nanmean(np.array(any_bh_vals, dtype=float))) if any_bh_vals else float("nan"),
        "target_z_mean": float(np.nanmean(np.array(target_z_vals, dtype=float))) if target_z_vals else float("nan"),
        "max_any_z_mean": float(np.nanmean(np.array(max_any_z_vals, dtype=float))) if max_any_z_vals else float("nan"),
        "n": float(len(target_bh_vals)),
    }


def plot_benchmark_task_difficulty(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Benchmark/task difficulty characterization (paper-facing, method-agnostic diagnostics):
      Panel A: retrieval mixing level M(q) distribution
      Panel B: alignment ambiguity distribution via trial-level mean margin
               margin = best_cos_mean - second_cos_mean
      Panel C: non-oracle candidate-set size vs retrieved-provider recall

    Uses one canonical scheme's rag_multi trial logs to avoid cross-method duplication.
    Preference order picks a watermark scheme first (DirBucket > KGW > SemStamp).
    """
    figpath = out_dir / "fig_benchmark_task_difficulty"
    preferred = [
        "dirbucket_naive",
        "kgw_ward",
        "semstamp",
        "sim_attr",
        "overlap_attr",
        "entail_attr",
    ]

    chosen_scheme: str | None = None
    trial_files: List[Path] = []
    for s in preferred:
        if s not in schemes:
            continue
        files = find_rag_trial_logs(results_root, s)
        if files:
            chosen_scheme = s
            trial_files = files
            break

    if chosen_scheme is None or not trial_files:
        print("[WARN] No rag_multi trial logs found for benchmark difficulty plot.")
        return

    mix_levels: List[int] = []
    ambiguity_margins: List[float] = []
    candidate_sizes: List[float] = []
    candidate_sizes_for_recall: List[float] = []
    provider_recalls: List[float] = []
    max_docs_vals: List[int] = []

    for fp in trial_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue

                    retrieved = row.get("retrieved_docs") or []
                    if isinstance(retrieved, list) and retrieved:
                        rset: set[int] = set()
                        for rec in retrieved:
                            if not isinstance(rec, dict):
                                continue
                            pid = rec.get("provider_id")
                            if isinstance(pid, (int, float)):
                                rset.add(int(pid))
                        if rset:
                            mix_levels.append(len(rset))

                    al = row.get("align") or {}
                    if isinstance(al, dict):
                        b = al.get("best_cos_mean")
                        s = al.get("second_cos_mean")
                        if isinstance(b, (int, float)) and isinstance(s, (int, float)):
                            ambiguity_margins.append(float(b) - float(s))

                    src = row.get("aligned_source") or {}
                    if not isinstance(src, dict):
                        continue
                    selected_docs = src.get("selected_docs")
                    by_provider = src.get("selected_docs_by_provider")
                    selected_docs_val: float | None = None
                    if isinstance(selected_docs, (int, float)):
                        selected_docs_val = float(selected_docs)
                        candidate_sizes.append(selected_docs_val)
                    if isinstance(by_provider, dict) and isinstance(retrieved, list) and retrieved:
                        cand_set: set[int] = set()
                        for k in by_provider.keys():
                            try:
                                cand_set.add(int(k))
                            except Exception:
                                continue
                        rset: set[int] = set()
                        for rec in retrieved:
                            if not isinstance(rec, dict):
                                continue
                            pid = rec.get("provider_id")
                            if isinstance(pid, (int, float)):
                                rset.add(int(pid))
                        if rset:
                            provider_recalls.append(float(len(rset & cand_set)) / float(len(rset)))
                            if selected_docs_val is not None:
                                candidate_sizes_for_recall.append(selected_docs_val)
                    md = src.get("max_docs")
                    if isinstance(md, (int, float)) and int(md) > 0:
                        max_docs_vals.append(int(md))
        except Exception:
            continue

    if not mix_levels:
        print("[WARN] Missing retrieval-mixing stats for benchmark difficulty plot.")
        return

    # Camera-ready styling for ACL two-column (intended for figure* full-width).
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.45))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    anno_fs = 6.8

    # Panel A: retrieval mixing level (histogram).
    ax = axs[0]
    mix_arr = np.array(mix_levels, dtype=int)
    max_m = max(1, int(np.max(mix_arr))) if mix_arr.size else 1
    bins = np.arange(0.5, max_m + 1.5, 1.0)
    ax.hist(
        mix_arr,
        bins=bins,
        density=True,
        color="#4c78a8",
        alpha=0.86,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(np.arange(1, max_m + 1))
    ax.set_xlim(0.9, max_m + 0.1)
    ax.set_xlabel("M(q): providers in retrieved bundle", fontsize=label_fs)
    ax.set_ylabel("Density", fontsize=label_fs)
    ax.set_title("A) Retrieval Mixing Level", fontsize=title_fs, pad=2.0)
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.axvline(float(np.mean(mix_arr)), color="black", linestyle="--", linewidth=1.0, alpha=0.85)

    # Panel B: ambiguity margin distribution (ECDF).
    ax = axs[1]
    tie_margin_ref = 0.05
    if ambiguity_margins:
        arr = np.array(ambiguity_margins, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = np.sort(arr)
        yy = np.arange(1, len(arr) + 1, dtype=float) / float(len(arr))
        ax.plot(arr, yy, color="#f58518", linewidth=2.0)
        frac_amb = float(np.mean(arr <= tie_margin_ref))
    else:
        ax.text(0.5, 0.5, "No ambiguity margins found", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(tie_margin_ref, color="black", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_xlabel("mean(best cos - second cos) per query", fontsize=label_fs)
    ax.set_ylabel("ECDF", fontsize=label_fs)
    ax.set_title("B) Alignment Ambiguity", fontsize=title_fs, pad=2.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.25)

    # Panel C: candidate-set size vs retrieved-provider recall (camera-ready).
    ax = axs[2]
    if candidate_sizes_for_recall and provider_recalls and len(candidate_sizes_for_recall) == len(provider_recalls):
        cs = np.array(candidate_sizes_for_recall, dtype=float)
        rec = np.array(provider_recalls, dtype=float)
        # Use log-scaled x-axis to keep full range readable without hard clipping artifacts.
        cs = np.maximum(cs, 1.0)
        main = np.isfinite(cs) & np.isfinite(rec)
        cs = cs[main]
        rec = rec[main]
        ax.scatter(
            cs,
            rec,
            s=9,
            color="#4c78a8",
            alpha=0.18,
            edgecolors="none",
            label="queries",
        )

        if cs.size >= 30 and float(np.max(cs)) > float(np.min(cs)):
            # Log-space bins produce a stable trend across small/large candidate sets.
            q = np.logspace(np.log10(float(np.min(cs))), np.log10(float(np.max(cs))), num=9)
            centers: List[float] = []
            y_med: List[float] = []
            y_q25: List[float] = []
            y_q75: List[float] = []
            for lo, hi in zip(q[:-1], q[1:]):
                m = (cs >= lo) & (cs < hi if hi < q[-1] else cs <= hi)
                if int(np.sum(m)) < 8:
                    continue
                centers.append(float(np.median(cs[m])))
                y_med.append(float(np.median(rec[m])))
                y_q25.append(float(np.quantile(rec[m], 0.25)))
                y_q75.append(float(np.quantile(rec[m], 0.75)))
            if centers:
                xx = np.array(centers, dtype=float)
                med = np.array(y_med, dtype=float)
                q25 = np.array(y_q25, dtype=float)
                q75 = np.array(y_q75, dtype=float)
                ax.fill_between(xx, q25, q75, color="#e45756", alpha=0.20, linewidth=0)
                ax.plot(xx, med, color="#e45756", marker="o", markersize=3.5, linewidth=1.8, label="median recall")
        ax.set_xscale("log")
        ax.legend(loc="lower right", fontsize=anno_fs, frameon=True)
    else:
        ax.text(0.5, 0.5, "No recall data found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("candidate-set size (docs, log scale)", fontsize=label_fs)
    ax.set_ylabel("retrieved-provider recall", fontsize=label_fs)
    ax.set_title("C) Candidate-Set Difficulty", fontsize=title_fs, pad=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.25)

    for ax in axs:
        ax.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)

    fig.suptitle(
        "Benchmark Difficulty Characterization (6 providers, IID data split)",
        fontsize=9.4,
        y=0.995,
    )
    _save_fig(fig, figpath)


def _plot_primary_panel(
    *,
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    panel_title: str,
    panel_schemes: List[str],
    stats: Dict[str, Dict[str, Dict[str, float]]],
    setting_keys: List[str],
    colors: Dict[str, str],
) -> None:
    if not panel_schemes:
        ax_top.axis("off")
        ax_bottom.axis("off")
        return

    x = np.arange(len(setting_keys))
    width = 0.82 / max(1, len(panel_schemes))

    for i, s in enumerate(panel_schemes):
        offs = (i - (len(panel_schemes) - 1) / 2.0) * width
        tgt = [stats[s][k]["target_bh_rate"] for k in setting_keys]
        nt = [stats[s][k]["non_target_bh_rate"] for k in setting_keys]
        tz = [stats[s][k]["target_z_mean"] for k in setting_keys]
        nz = [stats[s][k]["max_non_z_mean"] for k in setting_keys]
        c = colors.get(s, "#777777")
        ax_top.bar(
            x + offs,
            tgt,
            width=width,
            edgecolor="black",
            linewidth=0.6,
            color=c,
            label=_scheme_short_label(s),
        )
        ax_bottom.bar(
            x + offs,
            nt,
            width=width,
            edgecolor="black",
            linewidth=0.6,
            color=c,
            alpha=0.85,
        )
        for xx, yy, zt, zn in zip(x + offs, tgt, tz, nz):
            if np.isfinite(yy):
                txt = "n/a"
                if np.isfinite(zt) and np.isfinite(zn):
                    txt = f"{zt:.1f}/{zn:.1f}"
                ax_top.text(
                    float(xx),
                    min(1.02, float(yy) + 0.035),
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    ax_top.set_title(panel_title)
    ax_top.set_ylabel("target BH rate")
    ax_top.set_ylim(0.0, 1.05)
    ax_top.grid(axis="y", linestyle=":", alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=8, frameon=True)
    ax_top.text(
        0.01,
        0.98,
        "bar labels: z_target / z_max_non_target",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    ax_bottom.set_ylabel("non-target BH rate")
    ax_bottom.set_ylim(0.0, 1.05)
    ax_bottom.grid(axis="y", linestyle=":", alpha=0.25)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([_setting_display_name(k) for k in setting_keys], fontsize=9)


def plot_baseline_screen_overview(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Figure 1 (Sec. 6.2): baseline viability screen in the primary setting.
    2x2 layout aggregated over targets x seeds from mixed-provider RAG runs:
      TL: target pooled z
      TR: max non-target pooled z
      BL: target BH-significant rate (with 95% CI)
      BR: non-target BH-significant rate (with 95% CI)
    """
    figpath = out_dir / "fig_baseline_screen_overview"
    method_order = [
        "dirbucket_naive",
        "kgw_ward",
        "semstamp",
    ]
    methods = [m for m in method_order if m in schemes]
    if not methods:
        print("[WARN] No methods available for baseline screen.")
        return

    # Keep method colors consistent with the rest of the paper plots.
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    labels = {
        "dirbucket_naive": "DirBucket",
        "kgw_ward": "KGW",
        "semstamp": "SemStamp",
        "overlap_attr": "Lexical overlap",
        "sim_attr": "Semantic similarity",
        "entail_attr": "Entailment verifier",
    }

    rag_stats: Dict[str, List[Dict[str, Any]]] = {}
    for m in methods:
        vals: List[Dict[str, Any]] = []
        for fp in find_rag_multi_files(results_root, m):
            vals.append(_target_non_target_stats(load_json(fp)))
        rag_stats[m] = vals

    if not any(len(v) > 0 for v in rag_stats.values()):
        print("[WARN] No mixed-provider RAG files found for baseline screen.")
        return

    def _rate_ci(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return float("nan"), float("nan"), float("nan")
        arr = np.array(vals, dtype=float)
        n = int(arr.size)
        p = float(np.mean(arr))
        se = float(np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1)))
        lo = max(0.0, p - 1.96 * se)
        hi = min(1.0, p + 1.96 * se)
        return p, lo, hi

    target_z_data: List[List[float]] = []
    non_target_z_data: List[List[float]] = []
    target_bh_mean: List[float] = []
    target_bh_lo: List[float] = []
    target_bh_hi: List[float] = []
    non_target_bh_mean: List[float] = []
    non_target_bh_lo: List[float] = []
    non_target_bh_hi: List[float] = []
    ns: List[int] = []

    for m in methods:
        recs = rag_stats[m]
        tz = [float(r["target_z"]) for r in recs if np.isfinite(float(r["target_z"]))]
        nz = [float(r["max_non_z"]) for r in recs if np.isfinite(float(r["max_non_z"]))]
        tb = [1.0 if bool(r["target_bh"]) else 0.0 for r in recs]
        nb = [1.0 if bool(r["any_non_bh"]) else 0.0 for r in recs]

        target_z_data.append(tz)
        non_target_z_data.append(nz)
        t_m, t_lo, t_hi = _rate_ci(tb)
        n_m, n_lo, n_hi = _rate_ci(nb)
        target_bh_mean.append(t_m)
        target_bh_lo.append(t_lo)
        target_bh_hi.append(t_hi)
        non_target_bh_mean.append(n_m)
        non_target_bh_lo.append(n_lo)
        non_target_bh_hi.append(n_hi)
        ns.append(len(recs))

    x = np.arange(len(methods))
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    ax_tz, ax_nz = axs[0, 0], axs[0, 1]
    ax_tb, ax_nb = axs[1, 0], axs[1, 1]

    # Top-left: target pooled z distribution (violin + jitter)
    for i, m in enumerate(methods):
        vals = target_z_data[i]
        if vals:
            vp = ax_tz.violinplot(
                [vals],
                positions=[i],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in vp.get("bodies", []):
                body.set_facecolor(colors[m])
                body.set_edgecolor("black")
                body.set_linewidth(0.8)
                body.set_alpha(0.35)
            q1, med, q3 = np.percentile(np.array(vals, dtype=float), [25, 50, 75])
            ax_tz.vlines(i, q1, q3, color="black", linewidth=2.0, zorder=3)
            ax_tz.scatter([i], [med], color="black", s=14, zorder=4)
            rng = np.random.default_rng(1000 + i)
            xx = rng.normal(i, 0.05, size=len(vals))
            ax_tz.scatter(xx, vals, s=9, color=colors[m], alpha=0.45, linewidths=0)
    ax_tz.axhline(4.0, color="#d95f02", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_tz.set_title("Target Pooled z (higher is better)")
    ax_tz.set_ylabel("pooled z")
    ax_tz.grid(axis="y", linestyle=":", alpha=0.25)

    # Top-right: max non-target pooled z distribution (violin + jitter)
    for i, m in enumerate(methods):
        vals = non_target_z_data[i]
        if vals:
            vp = ax_nz.violinplot(
                [vals],
                positions=[i],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in vp.get("bodies", []):
                body.set_facecolor(colors[m])
                body.set_edgecolor("black")
                body.set_linewidth(0.8)
                body.set_alpha(0.35)
            q1, med, q3 = np.percentile(np.array(vals, dtype=float), [25, 50, 75])
            ax_nz.vlines(i, q1, q3, color="black", linewidth=2.0, zorder=3)
            ax_nz.scatter([i], [med], color="black", s=14, zorder=4)
            rng = np.random.default_rng(2000 + i)
            xx = rng.normal(i, 0.05, size=len(vals))
            ax_nz.scatter(xx, vals, s=9, color=colors[m], alpha=0.45, linewidths=0)
    ax_nz.axhline(4.0, color="#d95f02", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_nz.set_title("Max Non-target Pooled z (lower is better)")
    ax_nz.set_ylabel("pooled z")
    ax_nz.grid(axis="y", linestyle=":", alpha=0.25)

    # Bottom-left: target BH-significant rate with CI (bars + error bars)
    for i, m in enumerate(methods):
        p, lo, hi = target_bh_mean[i], target_bh_lo[i], target_bh_hi[i]
        if np.isfinite(p):
            ax_tb.bar(i, p, width=0.66, color=colors[m], edgecolor="black", linewidth=0.7, alpha=0.82, zorder=2)
            ax_tb.errorbar(
                i,
                p,
                yerr=[[p - lo], [hi - p]],
                fmt="none",
                markersize=6,
                color="black",
                ecolor="black",
                elinewidth=1.3,
                capsize=3,
                zorder=3,
            )
            ax_tb.text(i, min(1.02, p + 0.06), f"n={ns[i]}", ha="center", va="bottom", fontsize=7)
    ax_tb.axhspan(0.80, 1.00, color="#2ca02c", alpha=0.08, zorder=0)
    ax_tb.set_ylim(0.0, 1.05)
    ax_tb.set_title("Target BH-significant Rate (95% CI)")
    ax_tb.set_ylabel("rate")
    ax_tb.grid(axis="y", linestyle=":", alpha=0.25)

    # Bottom-right: non-target BH-significant rate with CI (bars + error bars)
    for i, m in enumerate(methods):
        p, lo, hi = non_target_bh_mean[i], non_target_bh_lo[i], non_target_bh_hi[i]
        if np.isfinite(p):
            ax_nb.bar(i, p, width=0.66, color=colors[m], edgecolor="black", linewidth=0.7, alpha=0.82, zorder=2)
            ax_nb.errorbar(
                i,
                p,
                yerr=[[p - lo], [hi - p]],
                fmt="none",
                markersize=6,
                color="black",
                ecolor="black",
                elinewidth=1.3,
                capsize=3,
                zorder=3,
            )
            ax_nb.text(i, min(1.02, p + 0.06), f"n={ns[i]}", ha="center", va="bottom", fontsize=7)
    ax_nb.axhspan(0.00, 0.05, color="#2ca02c", alpha=0.08, zorder=0)
    ax_nb.axhline(0.05, color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.6)
    ax_nb.set_ylim(0.0, 1.05)
    ax_nb.set_title("Non-target BH-significant Rate (95% CI)")
    ax_nb.set_ylabel("rate")
    ax_nb.grid(axis="y", linestyle=":", alpha=0.25)

    xticklabels = [labels[m] for m in methods]
    for ax in [ax_tz, ax_nz, ax_tb, ax_nb]:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=18, ha="right")

    fig.suptitle(
        "Baseline Viability Overview (Primary Setting: non-oracle aligned mixed-provider RAG)",
        fontsize=12,
    )
    _save_fig(fig, figpath)


def plot_null_calibration(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Section 6.3: null calibration and key dependence (DirBucket-only).
      Panel A: pooled-z distributions under correct key / wrong key / unrelated null.
      Panel B: BH activation rates under the same conditions, with ungated vs gated pairs.
    """
    figpath = out_dir / "fig_null_calibration"
    scheme = "dirbucket_naive"
    if scheme not in schemes:
        print("[WARN] DirBucket not in --schemes; skipping null calibration figure.")
        return

    coop_files = find_files(results_root, scheme, "sanity_coop_t*_seed*.json")
    unrel_files = find_files(results_root, scheme, "sanity_unrel_t*_seed*.json")
    if not coop_files or not unrel_files:
        print("[WARN] Missing DirBucket sanity files for null calibration figure.")
        return

    def _safe_bh(z_vals: List[float], t_vals: List[int], min_t: int) -> List[bool]:
        if not isinstance(z_vals, list) or not isinstance(t_vals, list):
            return []
        if len(z_vals) != len(t_vals) or len(z_vals) == 0:
            return []
        return _bh_sig_from_snapshot(z_vals, t_vals, min_pooled_T=min_t, alpha=0.05)

    z_correct: List[float] = []
    z_wrong: List[float] = []
    z_unrel: List[float] = []
    support_correct: List[int] = []
    support_wrong: List[int] = []
    support_unrel_all: List[int] = []
    evidence_correct: List[float] = []
    evidence_wrong: List[float] = []
    evidence_unrel_all: List[float] = []
    min_t_vals: List[int] = []

    rate_correct_gated: List[float] = []
    rate_correct_ungated: List[float] = []
    rate_wrong_gated: List[float] = []
    rate_wrong_ungated: List[float] = []
    rate_unrel_any_gated: List[float] = []
    rate_unrel_any_ungated: List[float] = []

    for fp in coop_files:
        r = load_json(fp)
        tp = int(r.get("target_provider", 0))
        min_t = int(r.get("min_pooled_T", 1))
        min_t_vals.append(min_t)

        zr = r.get("z_raw") or r.get("z") or []
        Tr = r.get("pooled_T") or []
        if isinstance(zr, list) and zr and 0 <= tp < len(zr):
            z_correct.append(float(zr[tp]))
            if isinstance(Tr, list) and tp < len(Tr):
                support_correct.append(int(Tr[tp]))
                evidence_correct.append(float(zr[tp]))
            bh_g = _safe_bh(zr, Tr, min_t=min_t)
            bh_u = _safe_bh(zr, Tr, min_t=0)
            if bh_g and tp < len(bh_g):
                rate_correct_gated.append(1.0 if bool(bh_g[tp]) else 0.0)
            elif isinstance(r.get("bh_sig"), list) and tp < len(r.get("bh_sig") or []):
                rate_correct_gated.append(1.0 if bool((r.get("bh_sig") or [])[tp]) else 0.0)
            if bh_u and tp < len(bh_u):
                rate_correct_ungated.append(1.0 if bool(bh_u[tp]) else 0.0)

        wk = r.get("wrongkey") or {}
        zwr = wk.get("z_raw") or wk.get("z") or []
        Tw = wk.get("pooled_T") or []
        if isinstance(zwr, list) and zwr and 0 <= tp < len(zwr):
            z_wrong.append(float(zwr[tp]))
            if isinstance(Tw, list) and tp < len(Tw):
                support_wrong.append(int(Tw[tp]))
                evidence_wrong.append(float(zwr[tp]))
            bh_g = _safe_bh(zwr, Tw, min_t=min_t)
            bh_u = _safe_bh(zwr, Tw, min_t=0)
            if bh_g and tp < len(bh_g):
                rate_wrong_gated.append(1.0 if bool(bh_g[tp]) else 0.0)
            if bh_u and tp < len(bh_u):
                rate_wrong_ungated.append(1.0 if bool(bh_u[tp]) else 0.0)

    for fp in unrel_files:
        r = load_json(fp)
        min_t = int(r.get("min_pooled_T", 1))
        zr = r.get("z_raw") or r.get("z") or []
        Tr = r.get("pooled_T") or []
        if isinstance(zr, list) and zr:
            z_unrel.extend(float(v) for v in zr)
            if isinstance(Tr, list) and len(Tr) == len(zr):
                support_unrel_all.extend(int(ti) for ti in Tr)
                evidence_unrel_all.extend(float(zi) for zi in zr)
            bh_g = _safe_bh(zr, Tr, min_t=min_t)
            bh_u = _safe_bh(zr, Tr, min_t=0)
            if bh_g:
                rate_unrel_any_gated.append(1.0 if any(bool(v) for v in bh_g) else 0.0)
            elif isinstance(r.get("bh_sig"), list):
                bh_saved = r.get("bh_sig") or []
                rate_unrel_any_gated.append(1.0 if any(bool(v) for v in bh_saved) else 0.0)
            if bh_u:
                rate_unrel_any_ungated.append(1.0 if any(bool(v) for v in bh_u) else 0.0)

    if not z_correct or not z_wrong or not z_unrel:
        print("[WARN] Insufficient DirBucket null-calibration data after parsing.")
        return

    # Camera-ready ACL style (aligned with recent figures).
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.95))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.8
    suptitle_fs = 9.4

    # Panel A: pooled-z distributions under correct / wrong / unrelated.
    dist_data = [z_correct, z_wrong, z_unrel]
    dist_labels = [
        "Correct key",
        "Wrong key",
        "Unrelated null",
    ]
    dist_colors = ["#1f77b4", "#72b7b2", "#7f7f7f"]
    xpos = np.arange(1, len(dist_data) + 1)

    v = ax_a.violinplot(
        dist_data,
        positions=xpos,
        vert=True,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, c in zip(v["bodies"], dist_colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.24)

    b = ax_a.boxplot(
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
    for patch, c in zip(b["boxes"], dist_colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.15)

    all_vals = np.array([float(vv) for arr in dist_data for vv in arr], dtype=float)
    ymin = min(float(np.nanmin(all_vals)) - 0.8, -2.8)
    ymax = max(float(np.nanmax(all_vals)) + 0.8, 5.2)
    ax_a.set_ylim(ymin, ymax)
    ax_a.set_xlim(0.45, len(dist_data) + 0.55)
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels(dist_labels, fontsize=tick_fs)
    ax_a.axhline(0.0, linestyle="--", linewidth=0.9, color="#666666", alpha=0.9)
    ax_a.axhline(4.0, linestyle="--", linewidth=0.9, color="#666666", alpha=0.9)
    ax_a.set_ylabel("Pooled z (raw)", fontsize=label_fs)
    ax_a.set_title("A) Null z Distributions", fontsize=title_fs, pad=4.2)
    ax_a.grid(axis="y", linestyle=":", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)

    # Panel B: evidence-support map under null/key perturbation.
    t_min = int(round(float(np.median(np.array(min_t_vals, dtype=float))))) if min_t_vals else 25
    ax_b.axvline(float(t_min), linestyle="--", linewidth=0.95, color="#666666", alpha=0.90)
    ax_b.axhline(4.0, linestyle="--", linewidth=0.95, color="#666666", alpha=0.90)
    ax_b.axhline(0.0, linestyle=":", linewidth=0.8, color="#888888", alpha=0.65)

    x_all = np.array(
        [float(v) for v in (support_correct + support_wrong + support_unrel_all)] or [0.0, float(t_min)],
        dtype=float,
    )
    y_all = np.array(
        [float(v) for v in (evidence_correct + evidence_wrong + evidence_unrel_all)] or [0.0, 4.0],
        dtype=float,
    )
    x_hi = max(float(np.nanmax(x_all)) + 4.0, float(t_min) + 6.0)
    y_lo = min(float(np.nanmin(y_all)) - 0.8, -2.8)
    y_hi = max(float(np.nanmax(y_all)) + 0.8, 5.2)
    ax_b.set_xlim(-0.5, x_hi)
    ax_b.set_ylim(y_lo, y_hi)

    # Low-evidence region shading (x < T_min): smooth red fade toward T_min.
    # Keep a visible tint close to T_min so the full low-support span reads clearly.
    if t_min > 0:
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
            extent=(0.0, float(t_min), y_lo, y_hi),
            origin="lower",
            aspect="auto",
            interpolation="bicubic",
            zorder=0,
        )
        ax_b.text(
            0.36 * float(t_min),
            y_lo + 0.40 * (y_hi - y_lo),
            "Low-evidence region",
            color="#b22222",
            fontsize=6.2,
            ha="left",
            va="center",
            rotation=90,
            weight="bold",
            zorder=1,
        )

    def _alpha_cmap(
        base_color: str,
        name: str,
        *,
        min_alpha: float,
        max_alpha: float,
    ) -> LinearSegmentedColormap:
        r, g, b, _ = to_rgba(base_color)
        return LinearSegmentedColormap.from_list(
            name,
            [
                (r, g, b, min_alpha),
                (r, g, b, max_alpha),
            ],
        )

    def _hex_cloud(
        xs: List[int],
        ys: List[float],
        *,
        color: str,
        label: str,
        max_alpha: float,
        zorder: int,
    ) -> None:
        if not xs or not ys:
            return
        x_np = np.array(xs, dtype=float)
        y_np = np.array(ys, dtype=float)
        if len(x_np) < 8:
            ax_b.scatter(
                x_np,
                y_np,
                s=26,
                marker="o",
                color=color,
                edgecolor="none",
                alpha=max_alpha,
                zorder=zorder,
            )
            return
        cmap = _alpha_cmap(
            color,
            f"null_hex_{label.replace(' ', '_').lower()}",
            min_alpha=0.30 if label != "Unrelated null" else 0.26,
            max_alpha=max_alpha,
        )
        hb = ax_b.hexbin(
            x_np,
            y_np,
            gridsize=12,
            mincnt=1,
            cmap=cmap,
            linewidths=0.0,
            edgecolors="none",
            zorder=zorder,
        )
        # Ensure low-density points remain visible in print.
        ax_b.scatter(
            x_np,
            y_np,
            s=11.0,
            marker="o",
            color=color,
            edgecolor="none",
            alpha=0.33 if label != "Unrelated null" else 0.35,
            zorder=zorder + 0.05,
        )
        counts = hb.get_array()
        if counts is None or int(np.size(counts)) == 0:
            return
        max_count = float(np.nanmax(np.asarray(counts, dtype=float)))
        # If the cloud has no repeated bins, hexbin can look effectively invisible.
        # Fall back to visible semi-transparent points in that case.
        if not np.isfinite(max_count) or max_count <= 1.0:
            hb.remove()
            ax_b.scatter(
                x_np,
                y_np,
                s=24,
                marker="o",
                color=color,
                edgecolor="none",
                alpha=max(0.40, 0.58 * max_alpha),
                zorder=zorder,
            )

    _hex_cloud(
        support_correct,
        evidence_correct,
        color="#1f77b4",
        label="Correct key",
        max_alpha=1.00,
        zorder=3,
    )
    _hex_cloud(
        support_wrong,
        evidence_wrong,
        color="#72b7b2",
        label="Wrong key",
        max_alpha=1.00,
        zorder=3,
    )
    _hex_cloud(
        support_unrel_all,
        evidence_unrel_all,
        color="#7f7f7f",
        label="Unrelated null",
        max_alpha=0.96,
        zorder=2,
    )

    ax_b.set_xlabel("Support (pooled $T$) at evaluated provider", fontsize=label_fs)
    ax_b.set_ylabel("Pooled z (raw)", fontsize=label_fs)
    ax_b.set_title("B) Null Evidence vs Support", fontsize=title_fs, pad=4.2)
    ax_b.grid(True, linestyle=":", alpha=0.25)
    ax_b.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=6.8, label="Correct key"),
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#72b7b2", markeredgecolor="none", markersize=6.8, label="Wrong key"),
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#7f7f7f", markeredgecolor="none", markersize=6.8, label="Unrelated null"),
        ],
        loc="upper left",
        fontsize=legend_fs,
        frameon=True,
    )

    fig.suptitle("Null Calibration and Key Dependence (DirBucket)", fontsize=suptitle_fs, y=0.992)
    _save_fig(fig, figpath, tight_rect=(0.01, 0.02, 0.995, 1.0))


def _iter_trial_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except Exception:
        return []
    return out


def _collect_nonwm_trial_level(
    results_root: Path,
    scheme: str,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    vals: Dict[str, List[float]] = {
        "coop_winner": [],
        "unrel_winner": [],
        "rag_winner": [],
        "coop_margin": [],
        "unrel_margin": [],
        "rag_margin": [],
        "unrel_target_present": [],
        "unrel_selected_docs": [],
        "unrel_selected_providers": [],
        "unrel_target_bh": [],
    }
    proto: Dict[str, Any] = {}

    def _consume(pattern: str, setting: str) -> None:
        for fp in find_files(results_root, scheme, pattern):
            rows = _iter_trial_jsonl(fp)
            for row in rows:
                status = str(row.get("status") or "")
                if status.startswith("skipped"):
                    continue
                tp = row.get("target_provider")
                if not isinstance(tp, int) or tp < 0:
                    continue

                winner = row.get("winner_pid")
                if isinstance(winner, int) and winner >= 0:
                    vals[f"{setting}_winner"].append(1.0 if int(winner) == int(tp) else 0.0)

                raw = ((row.get("per_provider") or {}).get("score_raw") or [])
                if isinstance(raw, list) and len(raw) > tp:
                    tscore = float(raw[tp])
                    non = [float(raw[i]) for i in range(len(raw)) if i != tp]
                    if non:
                        vals[f"{setting}_margin"].append(float(tscore - max(non)))

                if setting != "unrel":
                    continue

                asrc = row.get("aligned_source") or {}
                if not isinstance(asrc, dict):
                    continue

                sdp = asrc.get("selected_docs_by_provider") or {}
                present = 0.0
                if isinstance(sdp, dict):
                    n = sdp.get(str(tp))
                    try:
                        present = 1.0 if int(n) > 0 else 0.0
                    except Exception:
                        present = 0.0
                vals["unrel_target_present"].append(float(present))

                sd = asrc.get("selected_docs")
                sp = asrc.get("selected_providers")
                if isinstance(sd, (int, float)):
                    vals["unrel_selected_docs"].append(float(sd))
                if isinstance(sp, (int, float)):
                    vals["unrel_selected_providers"].append(float(sp))

    _consume("sanity_coop_t*_seed*.trials.jsonl", "coop")
    _consume("sanity_unrel_t*_seed*.trials.jsonl", "unrel")
    _consume("rag_multi*.trials.jsonl", "rag")

    # Auxiliary summary-level BH false-accusation.
    for fp in find_files(results_root, scheme, "sanity_unrel_t*_seed*.json"):
        r = load_json(fp)
        tp = int(r.get("target_provider", 0))
        bh = r.get("bh_sig") or []
        if isinstance(bh, list) and len(bh) > tp >= 0:
            vals["unrel_target_bh"].append(1.0 if bool(bh[tp]) else 0.0)
        if not proto:
            align = r.get("align") or {}
            if isinstance(align, dict):
                proto = {
                    "source_mode": align.get("source_mode"),
                    "selector": align.get("global_source_selector"),
                    "query_thr": align.get("global_query_sim_threshold"),
                    "max_docs": align.get("global_max_docs"),
                }

    # Fallback to summary winner fractions if trial logs are missing.
    if not vals["coop_winner"]:
        for fp in find_files(results_root, scheme, "sanity_coop_t*_seed*.json"):
            r = load_json(fp)
            wf = r.get("winner_fraction_target")
            if isinstance(wf, (int, float)):
                vals["coop_winner"].append(float(wf))
    if not vals["unrel_winner"]:
        for fp in find_files(results_root, scheme, "sanity_unrel_t*_seed*.json"):
            r = load_json(fp)
            wf = r.get("winner_fraction_target")
            if isinstance(wf, (int, float)):
                vals["unrel_winner"].append(float(wf))
    if not vals["rag_winner"]:
        for fp in find_rag_multi_files(results_root, scheme):
            r = load_json(fp)
            wf = r.get("winner_fraction_target")
            if isinstance(wf, (int, float)):
                vals["rag_winner"].append(float(wf))

    return vals, proto


def plot_nonwm_critical_failure(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Main Sec. 6.2 non-watermark figure:
      Panel A: uncalibrated operating points (false accusation vs utility).
      Panel B: calibration curves (FPR on target-absent null vs TPR on rag_multi).
    """
    figpath = out_dir / "fig_nonwm_critical_failure"
    methods = [s for s in schemes if _is_heuristic_scheme(s)]
    if not methods:
        print("[WARN] No heuristic methods found for non-watermark failure plot.")
        return

    colors = {
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }

    labels = {
        "sim_attr": "Semantic similarity",
        "overlap_attr": "Lexical overlap",
        "entail_attr": "Entailment verifier",
    }

    by_method: Dict[str, Dict[str, List[float]]] = {}
    protocol_meta: Dict[str, Any] = {}
    for s in methods:
        vals, proto = _collect_nonwm_trial_level(results_root, s)
        by_method[s] = vals
        if not protocol_meta and proto:
            protocol_meta = proto

    have_any = any(
        by_method[s]["coop_winner"]
        or by_method[s]["unrel_winner"]
        or by_method[s]["rag_winner"]
        for s in methods
    )
    if not have_any:
        print("[WARN] No data for non-watermark critical-failure plot.")
        return

    # Camera-ready styling to match benchmark-difficulty figure family.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.95))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.8
    suptitle_fs = 9.4

    # Panel A: uncalibrated operating points (false accusation vs utility).
    point_rows: List[Dict[str, Any]] = []
    for s in methods:
        x_m, x_hw = mean_ci(by_method[s]["unrel_winner"])
        y_rag, y_rag_hw = mean_ci(by_method[s]["rag_winner"])
        y_coop, y_coop_hw = mean_ci(by_method[s]["coop_winner"])
        if not (np.isfinite(x_m) and np.isfinite(y_rag)):
            continue
        point_rows.append(
            {
                "scheme": s,
                "color": colors.get(s, "#777777"),
                "x": float(x_m),
                "x_hw": float(x_hw if np.isfinite(x_hw) else 0.0),
                "y_rag": float(y_rag),
                "y_rag_hw": float(y_rag_hw if np.isfinite(y_rag_hw) else 0.0),
                "y_coop": float(y_coop) if np.isfinite(y_coop) else float("nan"),
                "y_coop_hw": float(y_coop_hw if np.isfinite(y_coop_hw) else 0.0),
            }
        )

        # Print candidate leakage summary for paper text/caption.
        leak_m, leak_hw = mean_ci(by_method[s]["unrel_target_present"])
        if np.isfinite(leak_m):
            print(
                f"[NONWM] {s} target present in candidate set under sanity_unrel: "
                f"{leak_m:.3f} +/- {leak_hw:.3f}"
            )

    def _nice_range(vals: List[float], *, lo_floor: float, hi_ceil: float, min_span: float) -> Tuple[float, float]:
        finite = [float(v) for v in vals if np.isfinite(float(v))]
        if not finite:
            return (lo_floor, hi_ceil)
        lo = min(finite)
        hi = max(finite)
        pad = max(0.02, 0.20 * (hi - lo))
        out_lo = max(lo_floor, lo - pad)
        out_hi = min(hi_ceil, hi + pad)
        if (out_hi - out_lo) < min_span:
            mid = 0.5 * (out_hi + out_lo)
            out_lo = max(lo_floor, mid - 0.5 * min_span)
            out_hi = min(hi_ceil, mid + 0.5 * min_span)
        if (out_hi - out_lo) < min_span:
            out_lo = max(lo_floor, out_hi - min_span)
            out_hi = min(hi_ceil, out_lo + min_span)
        return (out_lo, out_hi)

    if point_rows:
        x_vals = [r["x"] for r in point_rows]
        y_vals = [r["y_rag"] for r in point_rows]
        for r in point_rows:
            if np.isfinite(r["y_coop"]):
                y_vals.append(r["y_coop"])
        # Keep full [0,1] x-range so confidence bars are never clipped and
        # the x=0.05 threshold remains visible in-context.
        x_lo, x_hi = 0.0, 1.0
        y_lo, y_hi = _nice_range(y_vals, lo_floor=0.0, hi_ceil=1.03, min_span=0.20)
    else:
        x_lo, x_hi = 0.0, 1.0
        y_lo, y_hi = 0.0, 1.03

    # Region overlays: desired (low false accusation + high utility) and observed high-FP area.
    if y_hi > 0.8:
        ax_a.add_patch(
            plt.Rectangle(
                (0.0, 0.8),
                0.05,
                y_hi - 0.8,
                facecolor="#59a14f",
                alpha=0.22,
                edgecolor="none",
                zorder=0,
            )
        )
        ax_a.text(
            0.025,
            0.806,
            "Desired region",
            color="#2f7d32",
            fontsize=6.2,
            rotation=90,
            ha="center",
            va="bottom",
            weight="bold",
            zorder=1,
        )
        # High false-accusation region starts at x=0.05.
        high_fp_x0 = 0.05
        # Smooth transition from x=0.05 to x=0.20 via alpha ramp.
        grad_x1 = min(0.20, x_hi)
        if grad_x1 > high_fp_x0:
            n_grad = 10
            step = (grad_x1 - high_fp_x0) / float(n_grad)
            for i in range(n_grad):
                x0 = high_fp_x0 + i * step
                a = 0.05 + 0.12 * (float(i + 1) / float(n_grad))
                ax_a.add_patch(
                    plt.Rectangle(
                        (x0, 0.8),
                        step,
                        y_hi - 0.8,
                        facecolor="#e15759",
                        alpha=a,
                        edgecolor="none",
                        zorder=0,
                    )
                )
        if x_hi > max(grad_x1, high_fp_x0):
            ax_a.add_patch(
                plt.Rectangle(
                    (max(grad_x1, high_fp_x0), 0.8),
                    x_hi - max(grad_x1, high_fp_x0),
                    y_hi - 0.8,
                    facecolor="#e15759",
                    alpha=0.18,
                    edgecolor="none",
                    zorder=0,
                )
            )
            ax_a.text(
                0.12,
                min(y_hi - 0.006, 0.912),
                "High false-accusation region",
                color="#b22222",
                fontsize=6.2,
                ha="left",
                va="top",
                weight="bold",
                zorder=1,
            )
    ax_a.axvline(0.05, linestyle="--", linewidth=0.9, color="#666666", alpha=0.9)
    ax_a.axhline(0.80, linestyle="--", linewidth=0.9, color="#666666", alpha=0.9)

    method_handles: List[Line2D] = []
    for r in point_rows:
        s = str(r["scheme"])
        c = str(r["color"])
        x = float(r["x"])
        x_hw = float(r["x_hw"])
        y_rag = float(r["y_rag"])
        y_rag_hw = float(r["y_rag_hw"])
        y_coop = float(r["y_coop"])
        y_coop_hw = float(r["y_coop_hw"])

        ax_a.errorbar(
            x,
            y_rag,
            xerr=x_hw,
            yerr=y_rag_hw,
            fmt="o",
            markersize=8,
            color=c,
            markeredgecolor="black",
            markeredgewidth=0.8,
            ecolor=c,
            elinewidth=1.0,
            capsize=2,
            zorder=4,
        )
        if np.isfinite(y_coop):
            ax_a.errorbar(
                x,
                y_coop,
                xerr=x_hw,
                yerr=y_coop_hw,
                fmt="^",
                markersize=7,
                color=c,
                markerfacecolor="white",
                markeredgecolor=c,
                markeredgewidth=1.1,
                ecolor=c,
                elinewidth=1.0,
                capsize=2,
                zorder=3,
            )

        method_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=c,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=labels[s],
                markersize=7.5,
            )
        )

    ax_a.set_xlim(x_lo, x_hi)
    ax_a.set_ylim(y_lo, y_hi)
    ax_a.set_xlabel("False-accusation rate: P(target wins | target-absent null)", fontsize=label_fs)
    ax_a.set_ylabel("Attribution utility:\nP(target wins | mixed-provider RAG)", fontsize=label_fs, labelpad=4.0)
    ax_a.set_title("A) Operating Points (Uncalibrated)", fontsize=title_fs, pad=4.2)
    ax_a.grid(True, linestyle=":", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="None", markerfacecolor="black", markersize=6, label="Mixed-provider RAG"),
            Line2D([0], [0], marker="^", color="black", linestyle="None", markerfacecolor="white", markersize=6, label="Single-source sanity"),
        ],
        loc="upper left",
        fontsize=legend_fs,
        frameon=True,
    )

    # Panel B: calibration curves on margin threshold.
    op_handles: List[Line2D] = []
    for s in methods:
        c = colors.get(s, "#777777")
        unrel = np.array(by_method[s]["unrel_margin"], dtype=float)
        rag = np.array(by_method[s]["rag_margin"], dtype=float)
        unrel = unrel[np.isfinite(unrel)]
        rag = rag[np.isfinite(rag)]
        if len(unrel) < 2 or len(rag) < 2:
            continue

        thresholds = np.unique(np.quantile(np.concatenate([unrel, rag]), np.linspace(0.0, 1.0, 160)))
        fprs = np.array([float(np.mean(unrel > t)) for t in thresholds], dtype=float)
        tprs = np.array([float(np.mean(rag > t)) for t in thresholds], dtype=float)
        order = np.argsort(fprs)
        ax_b.plot(fprs[order], tprs[order], color=c, linewidth=2.0, label=labels[s], zorder=2)

        t_fpr5 = float(np.quantile(unrel, 0.95))
        fpr5 = float(np.mean(unrel > t_fpr5))
        tpr5 = float(np.mean(rag > t_fpr5))
        ax_b.scatter([fpr5], [tpr5], color=c, edgecolor="black", linewidth=0.7, s=34, zorder=4)
        op_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=c,
                markeredgecolor="black",
                markersize=5.4,
                label=f"TPR={tpr5:.2f}",
            )
        )

    ax_b.axvline(0.05, linestyle="--", linewidth=0.9, color="#444444", alpha=0.85)
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_xlabel("FPR on target-absent null", fontsize=label_fs)
    ax_b.set_ylabel("TPR on mixed-provider RAG", fontsize=label_fs)
    ax_b.set_title("B) Calibration Curves (threshold on target margin)", fontsize=title_fs, pad=4.2)
    ax_b.grid(True, linestyle=":", alpha=0.25)
    ax_b.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)
    if op_handles:
        ax_b.legend(handles=op_handles, loc="lower right", fontsize=legend_fs, frameon=True)

    # Shared method legend beneath both panels.
    if method_handles:
        fig.legend(
            handles=method_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=max(1, min(3, len(method_handles))),
            fontsize=legend_fs,
            frameon=True,
        )

    fig.suptitle("Non-watermark Baselines: Tradeoff + Calibration", fontsize=suptitle_fs, y=0.992)
    _save_fig(fig, figpath, tight_rect=(0.01, 0.10, 0.995, 1.0))


def plot_nonwm_margin_distributions(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Support Sec. 6.2 figure (appendix/main optional):
    target margin distributions per method under target-present vs target-absent.
    """
    figpath = out_dir / "fig_nonwm_margin_distributions"
    methods = [s for s in schemes if _is_heuristic_scheme(s)]
    if not methods:
        print("[WARN] No heuristic methods found for non-watermark margin distribution figure.")
        return

    colors = {
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    labels = {
        "sim_attr": "Semantic similarity",
        "overlap_attr": "Lexical overlap",
        "entail_attr": "Entailment verifier",
    }

    by_method: Dict[str, Dict[str, List[float]]] = {}
    for s in methods:
        vals, _ = _collect_nonwm_trial_level(results_root, s)
        by_method[s] = vals

    positions: List[float] = []
    datas: List[List[float]] = []
    styles: List[Tuple[str, str]] = []
    for i, s in enumerate(methods):
        coop = [float(v) for v in by_method[s]["coop_margin"] if np.isfinite(float(v))]
        unrel = [float(v) for v in by_method[s]["unrel_margin"] if np.isfinite(float(v))]
        if coop:
            positions.append(float(i) - 0.18)
            datas.append(coop)
            styles.append((s, "coop"))
        if unrel:
            positions.append(float(i) + 0.18)
            datas.append(unrel)
            styles.append((s, "unrel"))

    if not datas:
        print("[WARN] No margin data for non-watermark distribution figure.")
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bp = ax.boxplot(
        datas,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0, "color": "black"},
        capprops={"linewidth": 1.0, "color": "black"},
    )

    for patch, (scheme, kind) in zip(bp["boxes"], styles):
        c = colors.get(scheme, "#777777")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)
        if kind == "coop":
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        else:
            patch.set_facecolor("white")
            patch.set_hatch("//")
            patch.set_alpha(1.0)

    rng = np.random.default_rng(42)
    for pos, vals, (scheme, kind) in zip(positions, datas, styles):
        c = colors.get(scheme, "#777777")
        n_show = min(len(vals), 220)
        sample = np.array(vals, dtype=float)
        if len(sample) > n_show:
            idx = rng.choice(len(sample), size=n_show, replace=False)
            sample = sample[idx]
        xx = rng.normal(float(pos), 0.03, size=len(sample))
        ax.scatter(xx, sample, s=8, color=c, alpha=0.18 if kind == "coop" else 0.12, linewidths=0)

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(methods), dtype=float))
    ax.set_xticklabels([labels[s] for s in methods], rotation=15, ha="right")
    ax.set_ylabel("target margin (score_target - max score_non_target)")
    ax.set_title("Target-Margin Distributions: Target Present vs Target Absent")
    ax.grid(axis="y", linestyle=":", alpha=0.25)

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", alpha=0.35),
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//"),
        ],
        labels=["sanity_coop (target present)", "sanity_unrel (target absent)"],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    _save_fig(fig, figpath)


def _load_wm_frontier_records(results_root: Path, scheme: str) -> List[Dict[str, float]]:
    """
    Load rag_multi records for keyed watermark schemes and extract frontier metrics.
    q2d_target is censored to trials+1 when crossed_at[target] is missing.
    """
    out: List[Dict[str, float]] = []
    for fp in find_rag_multi_files(results_root, scheme):
        try:
            r = load_json(fp)
        except Exception:
            continue
        tp = int(r.get("target_provider", 0))
        bh = r.get("bh_sig") or []
        z = r.get("z") or []
        crossed = r.get("crossed_at") or []
        trials = int(r.get("trials", 0) or 0)
        if not isinstance(bh, list) or not isinstance(z, list):
            continue
        if tp < 0 or tp >= len(bh) or tp >= len(z):
            continue

        non_ids = [i for i in range(len(bh)) if i != tp and i < len(z)]
        if not non_ids:
            continue

        target_bh = 1.0 if bool(bh[tp]) else 0.0
        any_non_bh = 1.0 if any(bool(bh[i]) for i in non_ids if i < len(bh)) else 0.0
        target_z = float(z[tp])
        max_non_z = float(max(float(z[i]) for i in non_ids))

        # q2d_target: censor missing crossing to trials+1.
        q2d_target = float(trials + 1) if trials > 0 else float("nan")
        detected = 0.0
        if isinstance(crossed, list) and tp < len(crossed):
            v = crossed[tp]
            if isinstance(v, (int, float)):
                q2d_target = float(v)
                detected = 1.0

        winner = r.get("winner_fraction_target")
        winner_f = float(winner) if isinstance(winner, (int, float)) else float("nan")

        out.append(
            {
                "target_bh": float(target_bh),
                "any_non_bh": float(any_non_bh),
                "target_z": float(target_z),
                "max_non_z": float(max_non_z),
                "q2d_target": float(q2d_target),
                "winner_fraction_target": float(winner_f),
                "trials": float(trials),
                "detected": float(detected),
            }
        )
    return out


def _median_bootstrap_ci(
    values: List[float],
    *,
    n_boot: int = 2000,
    seed: int = 13,
) -> Tuple[float, float, float]:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.array(vals, dtype=float)
    med = float(np.median(arr))
    if len(arr) <= 1:
        return (med, med, med)
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot), dtype=float)
    n = int(arr.size)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.median(arr[idx]))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return (med, lo, hi)


def _wm_frontier_summary(results_root: Path, methods: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for s in methods:
        recs = _load_wm_frontier_records(results_root, s)
        if not recs:
            continue

        def _vals(key: str) -> List[float]:
            return [float(r[key]) for r in recs if key in r and np.isfinite(float(r[key]))]

        t_bh_m, t_bh_hw = mean_ci(_vals("target_bh"))
        n_bh_m, n_bh_hw = mean_ci(_vals("any_non_bh"))
        t_z_m, t_z_hw = mean_ci(_vals("target_z"))
        n_z_m, n_z_hw = mean_ci(_vals("max_non_z"))
        q_med, q_lo, q_hi = _median_bootstrap_ci(_vals("q2d_target"))
        det_m, det_hw = mean_ci(_vals("detected"))
        w_m, w_hw = mean_ci(_vals("winner_fraction_target"))
        trials_v = _vals("trials")
        trials_med = float(np.median(np.array(trials_v, dtype=float))) if trials_v else float("nan")
        n = float(len(recs))

        out[s] = {
            "n": n,
            "target_bh_mean": float(t_bh_m),
            "target_bh_ci": float(t_bh_hw),
            "non_bh_mean": float(n_bh_m),
            "non_bh_ci": float(n_bh_hw),
            "target_z_mean": float(t_z_m),
            "target_z_ci": float(t_z_hw),
            "max_non_z_mean": float(n_z_m),
            "max_non_z_ci": float(n_z_hw),
            "q2d_med": float(q_med),
            "q2d_lo": float(q_lo),
            "q2d_hi": float(q_hi),
            "detect_frac_mean": float(det_m),
            "detect_frac_ci": float(det_hw),
            "winner_mean": float(w_m),
            "winner_ci": float(w_hw),
            "trials_med": float(trials_med),
        }
    return out


def _target_bh_cdf_from_rag_traj(
    results_root: Path,
    methods: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Compute CDF-style target BH trajectories by query index q for rag_multi runs.
    Returns:
      xs: query indices 1..max_q
      curves: method -> P(target BH sig by q)
      max_q: maximum q observed in trajectories
    """
    target_bh_q: Dict[str, List[Optional[int]]] = {s: [] for s in methods}
    max_q = 0
    for s in methods:
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            tp = int(r.get("target_provider", 0))
            min_pooled_T = int(r.get("min_pooled_T", 1))
            traj = r.get("traj") or []
            if not traj:
                continue
            max_q = max(max_q, max(int(snap.get("q", 0) or 0) for snap in traj))
            t_cross: Optional[int] = None
            for snap in traj:
                q = int(snap.get("q", 0) or 0)
                z = snap.get("z") or []
                T = snap.get("pooled_T") or []
                bh = _bh_sig_from_snapshot(z, T, min_pooled_T=min_pooled_T, alpha=0.05)
                if not bh or tp < 0 or tp >= len(bh):
                    continue
                if bool(bh[tp]):
                    t_cross = q
                    break
            target_bh_q[s].append(t_cross)

    if max_q <= 0:
        return np.array([], dtype=int), {s: np.array([], dtype=float) for s in methods}, 0

    xs = np.arange(1, max_q + 1, dtype=int)
    curves: Dict[str, np.ndarray] = {}
    for s in methods:
        vals = target_bh_q[s]
        if not vals:
            curves[s] = np.full(xs.shape, np.nan, dtype=float)
            continue
        curves[s] = np.array(
            [
                sum(1 for v in vals if isinstance(v, int) and v <= int(q)) / float(len(vals))
                for q in xs
            ],
            dtype=float,
        )
    return xs, curves, max_q


def _plot_wm_frontier_panel_b(
    ax: plt.Axes,
    *,
    summary: Dict[str, Dict[str, float]],
    methods: List[str],
    colors: Dict[str, str],
    labels: Dict[str, str],
    title_fs: float,
    label_fs: float,
    tick_fs: float,
    legend_fs: float,
) -> None:
    q_vals = [summary[s]["q2d_med"] for s in methods if s in summary and np.isfinite(summary[s]["q2d_med"])]
    if q_vals:
        q_lo_all = min(summary[s]["q2d_lo"] for s in methods if s in summary and np.isfinite(summary[s]["q2d_lo"]))
        q_hi_all = max(summary[s]["q2d_hi"] for s in methods if s in summary and np.isfinite(summary[s]["q2d_hi"]))
        q_pad = max(2.0, 0.10 * (q_hi_all - q_lo_all))
        ax.set_xlim(max(0.0, q_lo_all - q_pad), q_hi_all + q_pad)

    z_vals = [summary[s]["target_z_mean"] for s in methods if s in summary and np.isfinite(summary[s]["target_z_mean"])]
    if z_vals:
        z_lo_all = min(summary[s]["target_z_mean"] - summary[s]["target_z_ci"] for s in methods if s in summary)
        z_hi_all = max(summary[s]["target_z_mean"] + summary[s]["target_z_ci"] for s in methods if s in summary)
        z_pad = max(0.8, 0.12 * (z_hi_all - z_lo_all))
        ax.set_ylim(z_lo_all - z_pad, z_hi_all + z_pad)

    # Reference line for N (median trials across methods).
    trial_candidates = [summary[s]["trials_med"] for s in methods if s in summary and np.isfinite(summary[s]["trials_med"])]
    if trial_candidates:
        N = float(np.median(np.array(trial_candidates, dtype=float)))
        ax.axvline(N, linestyle="--", linewidth=0.9, color="#666666", alpha=0.85)
        ax.text(
            N + 0.8,
            ax.get_ylim()[0] + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "nondetections censored at N+1",
            fontsize=6.2,
            color="#555555",
            ha="left",
            va="bottom",
        )

    for s in methods:
        if s not in summary:
            continue
        sm = summary[s]
        x = float(sm["q2d_med"])
        xerr = np.array([[x - float(sm["q2d_lo"])], [float(sm["q2d_hi"]) - x]], dtype=float)
        y = float(sm["target_z_mean"])
        yerr = float(sm["target_z_ci"])
        c = colors.get(s, "#777777")
        ms = 8.6 if s == "dirbucket_naive" else 7.3
        alpha = 1.0 if s == "dirbucket_naive" else 0.95
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=ms,
            color=c,
            markeredgecolor="black",
            markeredgewidth=0.8,
            ecolor=c,
            elinewidth=1.0,
            capsize=2,
            alpha=alpha,
            zorder=4,
        )
        # Detection fraction annotation per point.
        det = float(sm.get("detect_frac_mean", float("nan")))
        if np.isfinite(det):
            ax.text(
                x + 0.7,
                y + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"det={det:.2f}",
                fontsize=6.1,
                color=c,
                ha="left",
                va="center",
            )

    ax.set_xlabel("Median queries-to-detection (censored)", fontsize=label_fs)
    ax.set_ylabel("Target pooled z (mean ±95% CI)", fontsize=label_fs)
    ax.set_title("B) Detection Speed vs Power", fontsize=title_fs, pad=4.2)
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)


def plot_wm_frontier_bh_q2d(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Section 6.2 Figure 2 (Version A):
      Panel A: BH specificity/power frontier
      Panel B: Q2D vs target z frontier
    """
    figpath = out_dir / "fig_wm_frontier_bh_q2d"
    methods = [s for s in ["dirbucket_naive", "kgw_ward", "semstamp"] if s in schemes]
    if not methods:
        print("[WARN] No watermark schemes available for BH frontier plot.")
        return

    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#17becf",
    }
    labels = {
        "dirbucket_naive": "DirBucket",
        "kgw_ward": "KGW",
        "semstamp": "SemStamp",
    }
    summary = _wm_frontier_summary(results_root, methods)
    have = [s for s in methods if s in summary]
    if not have:
        print("[WARN] No rag_multi data for watermark BH frontier plot.")
        return

    # Camera-ready ACL style.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.95))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.8
    suptitle_fs = 9.4

    # Panel A
    ax_a.axvline(0.05, linestyle="--", linewidth=0.9, color="#666666", alpha=0.85)
    ax_a.axhline(0.80, linestyle="--", linewidth=0.9, color="#666666", alpha=0.85)
    ax_a.add_patch(plt.Rectangle((0.0, 0.80), 0.05, 0.20, facecolor="#59a14f", alpha=0.16, edgecolor="none", zorder=0))
    for s in have:
        sm = summary[s]
        x = float(sm["non_bh_mean"])
        y = float(sm["target_bh_mean"])
        xerr = float(sm["non_bh_ci"])
        yerr = float(sm["target_bh_ci"])
        c = colors[s]
        ms = 8.6 if s == "dirbucket_naive" else 7.3
        alpha = 1.0 if s == "dirbucket_naive" else 0.95
        ax_a.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=ms,
            color=c,
            markeredgecolor="black",
            markeredgewidth=0.8,
            ecolor=c,
            elinewidth=1.0,
            capsize=2,
            alpha=alpha,
            zorder=4,
        )

    ax_a.set_xlim(-0.01, 1.0)
    ax_a.set_ylim(-0.01, 1.01)
    ax_a.set_xlabel("Non-target BH-significant rate", fontsize=label_fs)
    ax_a.set_ylabel("Target BH-significant rate", fontsize=label_fs)
    ax_a.set_title("A) BH Power vs Specificity Frontier", fontsize=title_fs, pad=4.2)
    ax_a.grid(True, linestyle=":", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)

    _plot_wm_frontier_panel_b(
        ax_b,
        summary=summary,
        methods=have,
        colors=colors,
        labels=labels,
        title_fs=title_fs,
        label_fs=label_fs,
        tick_fs=tick_fs,
        legend_fs=legend_fs,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[s],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7.4,
            label=labels[s],
        )
        for s in have
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=max(1, min(3, len(handles))),
        fontsize=legend_fs,
        frameon=True,
    )

    fig.suptitle("Watermark Baselines Frontier (non-oracle aligned)", fontsize=suptitle_fs, y=0.992)
    _save_fig(fig, figpath, tight_rect=(0.01, 0.10, 0.995, 1.0))


def plot_wm_frontier_z_q2d(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Section 6.2 Figure 2 (Version B):
      Panel A: pooled-z frontier (target vs max non-target)
      Panel B: Q2D vs target z frontier
    """
    figpath = out_dir / "fig_wm_frontier_z_q2d"
    methods = [s for s in ["dirbucket_naive", "kgw_ward", "semstamp"] if s in schemes]
    if not methods:
        print("[WARN] No watermark schemes available for z frontier plot.")
        return

    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#17becf",
    }
    labels = {
        "dirbucket_naive": "DirBucket",
        "kgw_ward": "KGW",
        "semstamp": "SemStamp",
    }
    summary = _wm_frontier_summary(results_root, methods)
    have = [s for s in methods if s in summary]
    if not have:
        print("[WARN] No rag_multi data for watermark z frontier plot.")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.95))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.8
    suptitle_fs = 9.4

    # Panel A
    for s in have:
        sm = summary[s]
        x = float(sm["max_non_z_mean"])
        y = float(sm["target_z_mean"])
        xerr = float(sm["max_non_z_ci"])
        yerr = float(sm["target_z_ci"])
        c = colors[s]
        ms = 8.6 if s == "dirbucket_naive" else 7.3
        alpha = 1.0 if s == "dirbucket_naive" else 0.95
        ax_a.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=ms,
            color=c,
            markeredgecolor="black",
            markeredgewidth=0.8,
            ecolor=c,
            elinewidth=1.0,
            capsize=2,
            alpha=alpha,
            zorder=4,
        )

    x_vals = [summary[s]["max_non_z_mean"] for s in have]
    y_vals = [summary[s]["target_z_mean"] for s in have]
    x_lo = min(min(x_vals) - 0.8, -1.0)
    x_hi = max(max(x_vals) + 1.2, 4.6)
    y_lo = min(min(y_vals) - 1.2, -1.2)
    y_hi = max(max(y_vals) + 1.8, 5.0)
    ax_a.set_xlim(x_lo, x_hi)
    ax_a.set_ylim(y_lo, y_hi)

    # Region coding around z-thresholds (x=4 non-target, y=4 target).
    x_thr = 4.0
    y_thr = 4.0
    xl = ax_a.get_xlim()[0]
    xh = ax_a.get_xlim()[1]
    yl = ax_a.get_ylim()[0]
    yh = ax_a.get_ylim()[1]

    # Quadrant Q3 (x<4, y<4): insufficient attribution.
    q3_w = max(0.0, min(x_thr, xh) - xl)
    q3_h = max(0.0, min(y_thr, yh) - yl)
    if q3_w > 0 and q3_h > 0:
        ax_a.add_patch(
            plt.Rectangle(
                (xl, yl),
                q3_w,
                q3_h,
                facecolor="#f2cf5b",
                alpha=0.20,
                edgecolor="none",
                zorder=0,
            )
        )
        ax_a.text(
            xl + 0.38 * q3_w,
            yl + 0.30 * q3_h,
            "Insufficient attribution",
            color="#8a6d1d",
            fontsize=6.2,
            ha="left",
            va="bottom",
            weight="bold",
            zorder=1,
        )

    # False-positive region: all x>=4 (both y>=4 and y<4).
    if xh > x_thr:
        ax_a.add_patch(
            plt.Rectangle(
                (x_thr, yl),
                xh - x_thr,
                yh - yl,
                facecolor="#e15759",
                alpha=0.20,
                edgecolor="none",
                zorder=0,
            )
        )
        ax_a.text(
            x_thr + 0.55 * (xh - x_thr),
            yl + 0.72 * (yh - yl),
            "False-positive\nregion",
            color="#b22222",
            fontsize=6.2,
            ha="center",
            va="center",
            rotation=90,
            weight="bold",
            zorder=1,
        )

    # Desired region upper-left with x-gradient from 2 -> 4 blending into red.
    if yh > y_thr and xl < x_thr:
        green = np.array([0x59, 0xA1, 0x4F], dtype=float) / 255.0
        red = np.array([0xE1, 0x57, 0x59], dtype=float) / 255.0

        # Strong-green block for x < 2.
        x_green_hi = min(2.0, x_thr, xh)
        if x_green_hi > xl:
            ax_a.add_patch(
                plt.Rectangle(
                    (xl, y_thr),
                    x_green_hi - xl,
                    yh - y_thr,
                    facecolor="#59a14f",
                    alpha=0.26,
                    edgecolor="none",
                    zorder=0,
                )
            )

        # Blend band 2 <= x < 4 (green -> red).
        x_blend_lo = max(xl, 2.0)
        x_blend_hi = min(xh, x_thr)
        if x_blend_hi > x_blend_lo:
            n_grad = 16
            step = (x_blend_hi - x_blend_lo) / float(n_grad)
            for i in range(n_grad):
                x0 = x_blend_lo + i * step
                t = float(i + 1) / float(n_grad)
                rgb = (1.0 - t) * green + t * red
                ax_a.add_patch(
                    plt.Rectangle(
                        (x0, y_thr),
                        step,
                        yh - y_thr,
                        facecolor=(float(rgb[0]), float(rgb[1]), float(rgb[2])),
                        alpha=0.22,
                        edgecolor="none",
                        zorder=0,
                    )
                )

        # Desired label in the upper-left safe area.
        x_txt = xl + 0.06 * max(0.01, min(x_thr, xh) - xl)
        y_txt = y_thr + 0.10 * max(0.01, yh - y_thr)
        ax_a.text(
            x_txt,
            y_txt,
            "Desired region\n(target z>4,\nnon-target z<4)",
            color="#2f7d32",
            fontsize=6.2,
            ha="left",
            va="bottom",
            weight="bold",
            zorder=1,
        )

    # Threshold references.
    ax_a.axvline(x_thr, linestyle="--", linewidth=0.9, color="#666666", alpha=0.85)
    ax_a.axhline(y_thr, linestyle="--", linewidth=0.9, color="#666666", alpha=0.85)

    ax_a.set_xlabel("Max non-target pooled z (mean)", fontsize=label_fs)
    ax_a.set_ylabel("Target pooled z (mean)", fontsize=label_fs)
    ax_a.set_title("A) z-Power vs z-Specificity Frontier", fontsize=title_fs, pad=4.2)
    ax_a.grid(True, linestyle=":", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)

    # Panel B: upper subplot equivalent of fig_watermark_compare_rag_queries_to_bh
    # (time to BH decision on target).
    xs, curves, max_q = _target_bh_cdf_from_rag_traj(results_root, have)
    if max_q <= 0 or xs.size == 0:
        ax_b.text(0.5, 0.5, "No rag trajectories found", ha="center", va="center", transform=ax_b.transAxes)
        ax_b.set_axis_off()
    else:
        for s in have:
            yy = curves.get(s)
            if yy is None or yy.size == 0:
                continue
            c = colors.get(s, "#777777")
            lw = 2.5 if s == "dirbucket_naive" else 2.1
            ax_b.plot(xs, yy, linewidth=lw, color=c, alpha=1.0 if s == "dirbucket_naive" else 0.95)
        ax_b.set_xlim(1, max_q)
        ax_b.set_ylim(0.0, 1.02)
        ax_b.set_xlabel("# audited answers (q)", fontsize=label_fs)
        ax_b.set_ylabel("P(target BH sig by q)", fontsize=label_fs)
        ax_b.set_title("B) Time to BH Decision", fontsize=title_fs, pad=4.2)
        ax_b.grid(True, linestyle=":", alpha=0.25)
        ax_b.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[s],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7.4,
            label=labels[s],
        )
        for s in have
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=max(1, min(3, len(handles))),
        fontsize=legend_fs,
        frameon=True,
    )

    fig.suptitle("Watermark Baselines Frontier (non-oracle aligned)", fontsize=suptitle_fs, y=0.992)
    _save_fig(fig, figpath, tight_rect=(0.01, 0.10, 0.995, 1.0))


def plot_rag_power(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_rag_power"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }

    series: Dict[str, Dict[int, List[float]]] = {s: {} for s in schemes}
    for s in schemes:
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            tp = int(r.get("target_provider", 0))
            traj = r.get("traj") or []
            for snap in traj:
                q = int(snap["q"])
                z = float(snap["z"][tp])
                series[s].setdefault(q, []).append(z)

    max_q = 0
    for s in schemes:
        if series[s]:
            max_q = max(max_q, max(series[s].keys()))

    if max_q == 0:
        print("[WARN] No rag trajectories found. Re-run compare with rag_multi + --log_traj.")
        return

    xs = np.arange(1, max_q + 1)
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for s in schemes:
        means = []
        lows = []
        highs = []
        for q in xs:
            m, hw = mean_ci(series[s].get(int(q), []))
            means.append(m)
            lows.append(m - hw)
            highs.append(m + hw)
        means_np = np.array(means, dtype=float)
        lows_np = np.array(lows, dtype=float)
        highs_np = np.array(highs, dtype=float)
        c = colors.get(s)
        ax.plot(xs, means_np, linewidth=2.2, label=_scheme_label(s), color=c)
        ax.fill_between(xs, lows_np, highs_np, alpha=0.18, color=c)

    ax.axhline(4.0, linestyle="--", linewidth=1.3, color="black")
    ax.set_xlabel("# audited answers")
    ax.set_ylabel("pooled z (target provider)")
    ax.set_title("RAG Power Curve: DirBucket vs KGW/WARD")
    ax.legend()
    _save_fig(fig, figpath)


def plot_evidence_accumulation_main(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Section 5.4 main (DirBucket-centric) figure:
      Panel A: target vs max non-target pooled-z trajectories over audit budget.
      Panel B: CDF of queries-to-first-target-BH decision.
    """
    figpath = out_dir / "fig_evidence_accumulation_main"
    scheme = "dirbucket_naive"
    if scheme not in schemes:
        print("[WARN] DirBucket not in --schemes; skipping fig_evidence_accumulation_main.")
        return

    rag_files = find_rag_multi_files(results_root, scheme)
    if not rag_files:
        print("[WARN] No DirBucket rag_multi files found; skipping fig_evidence_accumulation_main.")
        return

    target_by_q: Dict[int, List[float]] = {}
    max_non_by_q: Dict[int, List[float]] = {}
    target_bh_cross_q: List[Optional[int]] = []
    max_q = 0

    for fp in rag_files:
        r = load_json(fp)
        tp = int(r.get("target_provider", 0))
        min_pooled_T = int(r.get("min_pooled_T", 1))
        traj = r.get("traj") or []
        if not isinstance(traj, list) or not traj:
            continue

        t_cross: Optional[int] = None
        for snap in traj:
            q = int(snap.get("q", 0) or 0)
            if q <= 0:
                continue
            z = snap.get("z") or []
            T = snap.get("pooled_T") or []
            if not isinstance(z, list) or not z or tp < 0 or tp >= len(z):
                continue
            z_t = float(z[tp])
            # Decision-eligible non-target evidence under the same T gate used by BH.
            eligible_non_target = [
                float(z[i])
                for i in range(len(z))
                if i != tp and isinstance(T, list) and i < len(T) and int(T[i]) >= int(min_pooled_T)
            ]
            z_n = max(eligible_non_target) if eligible_non_target else 0.0
            target_by_q.setdefault(q, []).append(z_t)
            if np.isfinite(z_n):
                max_non_by_q.setdefault(q, []).append(z_n)
            max_q = max(max_q, q)

            if t_cross is None:
                bh = _bh_sig_from_snapshot(z, T, min_pooled_T=min_pooled_T, alpha=0.05)
                if bh and tp < len(bh) and bool(bh[tp]):
                    t_cross = q
        target_bh_cross_q.append(t_cross)

    if max_q <= 0 or not target_by_q:
        print("[WARN] No usable DirBucket rag trajectories for fig_evidence_accumulation_main.")
        return

    xs = np.arange(1, max_q + 1, dtype=int)

    def _median_iqr(values: List[float]) -> Tuple[float, float, float]:
        if not values:
            return (float("nan"), float("nan"), float("nan"))
        arr = np.array(values, dtype=float)
        return (
            float(np.nanmedian(arr)),
            float(np.nanquantile(arr, 0.25)),
            float(np.nanquantile(arr, 0.75)),
        )

    def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n <= 0:
            return (float("nan"), float("nan"))
        p = float(k) / float(n)
        z2 = z * z
        denom = 1.0 + z2 / float(n)
        center = (p + z2 / (2.0 * float(n))) / denom
        half = (z / denom) * math.sqrt((p * (1.0 - p) / float(n)) + (z2 / (4.0 * float(n) * float(n))))
        return (max(0.0, center - half), min(1.0, center + half))

    med_t: List[float] = []
    q1_t: List[float] = []
    q3_t: List[float] = []
    med_n: List[float] = []
    q1_n: List[float] = []
    q3_n: List[float] = []
    for q in xs:
        m, l, h = _median_iqr(target_by_q.get(int(q), []))
        med_t.append(m)
        q1_t.append(l)
        q3_t.append(h)
        m, l, h = _median_iqr(max_non_by_q.get(int(q), []))
        med_n.append(m)
        q1_n.append(l)
        q3_n.append(h)

    cdf: List[float] = []
    cdf_lo: List[float] = []
    cdf_hi: List[float] = []
    n_runs = len(target_bh_cross_q)
    for q in xs:
        k = sum(1 for v in target_bh_cross_q if isinstance(v, int) and int(v) <= int(q))
        p = float(k) / float(n_runs) if n_runs > 0 else float("nan")
        lo, hi = _wilson_ci(k, n_runs)
        cdf.append(p)
        cdf_lo.append(lo)
        cdf_hi.append(hi)

    detected = [int(v) for v in target_bh_cross_q if isinstance(v, int)]
    med_detect_q = int(np.median(np.array(detected, dtype=int))) if detected else None

    # Camera-ready style (aligned with current ACL figures).
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.90))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.8
    suptitle_fs = 9.4

    x_np = xs.astype(float)
    med_t_np = np.array(med_t, dtype=float)
    q1_t_np = np.array(q1_t, dtype=float)
    q3_t_np = np.array(q3_t, dtype=float)
    med_n_np = np.array(med_n, dtype=float)
    q1_n_np = np.array(q1_n, dtype=float)
    q3_n_np = np.array(q3_n, dtype=float)

    c_target = "#1f77b4"
    c_non = "#e15759"

    ax_a.fill_between(x_np, q1_t_np, q3_t_np, color=c_target, alpha=0.22, linewidth=0)
    ax_a.plot(x_np, med_t_np, color=c_target, linewidth=2.4, label="Target pooled z (median)")
    ax_a.fill_between(x_np, q1_n_np, q3_n_np, color=c_non, alpha=0.20, linewidth=0)
    ax_a.plot(x_np, med_n_np, color=c_non, linewidth=2.0, label="Max non-target pooled z (median)")
    ax_a.axhline(4.0, linestyle="--", linewidth=0.95, color="#666666", alpha=0.90)
    ax_a.set_xlim(1, max_q)
    ax_a.set_xlabel("# audited answers (q)", fontsize=label_fs)
    ax_a.set_ylabel("Pooled z", fontsize=label_fs)
    ax_a.set_title("A) Evidence Separation Over Audit Budget", fontsize=title_fs, pad=4.2)
    ax_a.grid(True, linestyle=":", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)

    cdf_np = np.array(cdf, dtype=float)
    lo_np = np.array(cdf_lo, dtype=float)
    hi_np = np.array(cdf_hi, dtype=float)
    q_full_detect: Optional[int] = None
    for i, p in enumerate(cdf_np.tolist()):
        if np.isfinite(p) and p >= 0.999999:
            q_full_detect = int(xs[i])
            break

    if q_full_detect is not None:
        ax_a.axvline(float(q_full_detect), linestyle=":", linewidth=1.05, color="#4a4a4a", alpha=0.95)

    ax_b.fill_between(x_np, lo_np, hi_np, color=c_target, alpha=0.20, linewidth=0)
    ax_b.plot(x_np, cdf_np, color=c_target, linewidth=2.4, label="P(target BH sig by q)")
    if q_full_detect is not None:
        ax_b.axvline(float(q_full_detect), linestyle=":", linewidth=1.05, color="#4a4a4a", alpha=0.95)
    if med_detect_q is not None:
        ax_b.axvline(float(med_detect_q), linestyle="--", linewidth=0.9, color="#444444", alpha=0.85)
    ax_b.set_xlim(1, max_q)
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_xlabel("# audited answers (q)", fontsize=label_fs)
    ax_b.set_ylabel("P(target BH sig by q)", fontsize=label_fs)
    ax_b.set_title("B) Queries to First Target BH Detection", fontsize=title_fs, pad=4.2)
    ax_b.grid(True, linestyle=":", alpha=0.25)
    ax_b.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.8)

    # Shared legend.
    shared_handles = [
        Line2D([0], [0], color=c_target, linewidth=2.4, label="Target pooled z (median + IQR)"),
        Line2D([0], [0], color=c_non, linewidth=2.0, label="Decision-eligible max non-target z (T>=T_min)"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=0.95, label="z = 4 reference"),
    ]
    if q_full_detect is not None:
        shared_handles.append(
            Line2D([0], [0], color="#4a4a4a", linestyle=":", linewidth=1.05, label="q where detection CDF = 1")
        )
    fig.legend(
        handles=shared_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=legend_fs,
        frameon=True,
    )

    fig.suptitle("Main Results: Non-oracle Aligned Multi-provider RAG (DirBucket)", fontsize=suptitle_fs, y=0.992)
    _save_fig(fig, figpath, tight_rect=(0.01, 0.10, 0.995, 1.0))


def plot_style_paraphrase_intensity(
    results_root: Path,
    out_dir: Path,
    schemes: List[str],
    work_dirs: Dict[str, Path],
) -> None:
    figpath = out_dir / "fig_watermark_compare_style_paraphrase_intensity"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    styles = ["qa", "ownwords", "bullets", "eli5"]
    style_pat = re.compile(r"style_sweep_seed\d+\.trials\.(qa|ownwords|bullets|eli5)\.jsonl$")

    # 1) Load corpus text maps once per scheme.
    corpus_by_scheme: Dict[str, Dict[str, str]] = {}
    for s in schemes:
        wd = work_dirs.get(s)
        if wd is None:
            continue
        corpus_by_scheme[s] = _load_corpus_text_map(wd)

    # 2) Aggregate trial-level paraphrase intensity metrics.
    unigram_overlap: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}
    trigram_copy: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}
    compression_ratio: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}

    for s in schemes:
        cmap = corpus_by_scheme.get(s) or {}
        if not cmap:
            continue
        for fp in find_files(results_root, s, "style_sweep_seed*.trials.*.jsonl"):
            m = style_pat.search(fp.name)
            if not m:
                continue
            st = m.group(1)
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tr = json.loads(line)
                    except Exception:
                        continue
                    ans = str(tr.get("answer") or "")
                    ans_toks = _simple_tokens(ans)
                    if not ans_toks:
                        continue

                    src_chunks: List[str] = []
                    for rec in tr.get("retrieved_docs") or []:
                        did = str((rec or {}).get("doc_id") or "")
                        txt = cmap.get(did)
                        if txt:
                            src_chunks.append(txt)
                    if not src_chunks:
                        continue

                    src_text = "\n".join(src_chunks)
                    src_toks = _simple_tokens(src_text)
                    if not src_toks:
                        continue

                    src_uni = set(src_toks)
                    ans_uni_hits = sum(1 for t in ans_toks if t in src_uni)
                    uov = float(ans_uni_hits) / float(len(ans_toks))

                    ans_tri = _ngram_set(ans_toks, 3)
                    src_tri = _ngram_set(src_toks, 3)
                    tcopy = (float(len(ans_tri & src_tri)) / float(len(ans_tri))) if ans_tri else 0.0

                    comp = float(len(ans_toks)) / float(max(1, len(src_toks)))

                    unigram_overlap[s][st].append(uov)
                    trigram_copy[s][st].append(tcopy)
                    compression_ratio[s][st].append(comp)

    # 3) Plot grouped bars with CIs.
    have_any = False
    for s in schemes:
        for st in styles:
            if unigram_overlap[s][st]:
                have_any = True
                break
        if have_any:
            break
    if not have_any:
        print("[WARN] No style trial logs/corpus data for paraphrase-intensity plot.")
        return

    x = np.arange(len(styles))
    n_s = max(1, len(schemes))
    width = 0.82 / n_s
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)

    def _plot_metric(ax: plt.Axes, by_scheme: Dict[str, Dict[str, List[float]]], ylabel: str) -> None:
        for i, s in enumerate(schemes):
            means: List[float] = []
            errs: List[float] = []
            for st in styles:
                m, hw = mean_ci(by_scheme[s][st])
                means.append(m)
                errs.append(hw)
            offs = (i - (n_s - 1) / 2.0) * width
            ax.bar(
                x + offs,
                means,
                width=width,
                yerr=errs,
                capsize=3,
                edgecolor="black",
                linewidth=0.6,
                color=colors.get(s),
                label=_scheme_label(s),
            )
        ax.set_ylabel(ylabel)

    _plot_metric(axes[0], unigram_overlap, "unigram overlap")
    axes[0].set_title("Style Paraphrase Intensity (Answer vs Retrieved Docs)")
    axes[0].legend()

    _plot_metric(axes[1], trigram_copy, "trigram copy ratio")
    _plot_metric(axes[2], compression_ratio, "length ratio (ans/src)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(styles)
    axes[2].set_xlabel("answer style")

    _save_fig(fig, figpath)


def plot_style_robustness_main(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """Section 5.4 Block B main figure (DirBucket-centric, simple bar view)."""
    figpath = out_dir / "fig_style_robustness_main"
    scheme = "dirbucket_naive"
    if scheme not in schemes:
        print("[WARN] DirBucket not in --schemes; skipping fig_style_robustness_main.")
        return

    styles = ["qa", "ownwords", "bullets", "eli5"]
    labels = {"qa": "QA", "ownwords": "Own\nWords", "bullets": "Bullets", "eli5": "ELI5"}
    target_vals: Dict[str, List[float]] = {st: [] for st in styles}
    gated_vals: Dict[str, List[float]] = {st: [] for st in styles}

    for fp in find_files(results_root, scheme, "style_sweep_seed*.json"):
        obj = load_json(fp)
        for run in obj.get("runs", []):
            st = str(run.get("answer_style") or "")
            if st not in styles:
                continue
            tp = int(run.get("target_provider", 0))
            z_raw = run.get("z_raw") or run.get("z") or []
            z = run.get("z") or z_raw
            if not isinstance(z_raw, list) or not z_raw or tp < 0 or tp >= len(z_raw):
                continue
            non = [float(z_raw[i]) for i in range(len(z_raw)) if i != tp]
            if not non:
                continue
            z_t = float(z_raw[tp])
            target_vals[st].append(z_t)
            if isinstance(z, list) and tp < len(z):
                gated_vals[st].append(float(z[tp]))

    if not any(target_vals[st] for st in styles):
        print("[WARN] No style_sweep DirBucket records found for fig_style_robustness_main.")
        return

    # ACL-friendly typography, aligned with the existing camera-ready figures.
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.45))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.8
    x = np.arange(len(styles))
    means: List[float] = []
    errs: List[float] = []
    for st in styles:
        m, hw = mean_ci(target_vals[st])
        means.append(m)
        errs.append(hw)
    ax.bar(
        x,
        means,
        yerr=errs,
        capsize=3,
        width=0.62,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
        label="Target pooled z",
    )
    ax.axhline(4.0, linestyle="--", linewidth=1.0, color="#555555", zorder=1, label="z = 4")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(st, st) for st in styles], fontsize=6.8)
    ax.tick_params(axis="y", labelsize=tick_fs, width=0.8, length=2.8)
    ax.grid(axis="y", linestyle=":", alpha=0.26, zorder=0)
    ax.set_ylabel("Target pooled Z", fontsize=label_fs)
    ax.set_title("Style Robustness (DirBucket)", fontsize=title_fs, pad=4.0)
    # Keep a small headroom for high bars and avoid clipping around z=4.
    vals_flat = np.array([v for st in styles for v in target_vals[st]], dtype=float)
    y_hi = max(4.8, float(np.nanmax(vals_flat)) + 0.9) if vals_flat.size else 6.0
    ax.set_ylim(0.0, y_hi)
    ax.legend(loc="upper right", fontsize=legend_fs, frameon=True)

    # Print raw-vs-gated diagnostics for text (ELI5 drop is gate-driven in tuned runs).
    for st in styles:
        rz = target_vals[st]
        gz = gated_vals[st]
        if rz:
            mr = float(np.nanmean(np.array(rz, dtype=float)))
            mg = float(np.nanmean(np.array(gz, dtype=float))) if gz else float("nan")
            print(f"[INFO] style={st:8s} mean(z_raw)={mr:.3f} mean(z_gated)={mg:.3f} n={len(rz)}")
    _save_fig(fig, figpath)


def plot_rag_specificity(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_rag_specificity"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }

    z_target: Dict[str, Dict[int, List[float]]] = {s: {} for s in schemes}
    z_max_non: Dict[str, Dict[int, List[float]]] = {s: {} for s in schemes}
    z_margin: Dict[str, Dict[int, List[float]]] = {s: {} for s in schemes}

    for s in schemes:
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            tp = int(r.get("target_provider", 0))
            traj = r.get("traj") or []
            for snap in traj:
                z = snap.get("z") or []
                if not isinstance(z, list) or not z:
                    continue
                if tp < 0 or tp >= len(z):
                    continue
                q = int(snap.get("q", 0))
                if q <= 0:
                    continue

                z_t = float(z[tp])
                z_n = max(float(z[i]) for i in range(len(z)) if i != tp) if len(z) > 1 else float("nan")
                z_target[s].setdefault(q, []).append(z_t)
                z_max_non[s].setdefault(q, []).append(z_n)
                z_margin[s].setdefault(q, []).append(z_t - z_n)

    max_q = 0
    for s in schemes:
        if z_target[s]:
            max_q = max(max_q, max(z_target[s].keys()))

    if max_q == 0:
        print("[WARN] No rag trajectories found. Re-run compare with rag_multi + --log_traj.")
        return

    xs = np.arange(1, max_q + 1)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.0), sharex=True)

    def _plot_band(ax: plt.Axes, series: Dict[int, List[float]], scheme: str) -> None:
        means = []
        lows = []
        highs = []
        for q in xs:
            m, hw = mean_ci(series.get(int(q), []))
            means.append(m)
            lows.append(m - hw)
            highs.append(m + hw)
        means_np = np.array(means, dtype=float)
        lows_np = np.array(lows, dtype=float)
        highs_np = np.array(highs, dtype=float)
        c = colors.get(scheme)
        ax.plot(xs, means_np, linewidth=2.2, label=_scheme_label(scheme), color=c)
        ax.fill_between(xs, lows_np, highs_np, alpha=0.18, color=c)

    for s in schemes:
        _plot_band(axes[0], z_target[s], s)
    axes[0].axhline(4.0, linestyle="--", linewidth=1.2, color="black")
    axes[0].set_ylabel("target z")
    axes[0].set_title("RAG Specificity: Target vs Max Non-Target vs Margin")
    axes[0].legend()

    for s in schemes:
        _plot_band(axes[1], z_max_non[s], s)
    axes[1].axhline(4.0, linestyle="--", linewidth=1.2, color="black")
    axes[1].set_ylabel("max non-target z")

    for s in schemes:
        _plot_band(axes[2], z_margin[s], s)
    axes[2].axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    axes[2].set_ylabel("margin (target - max non)")
    axes[2].set_xlabel("# audited answers")

    _save_fig(fig, figpath)


def plot_rag_queries_to_crossing(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_rag_queries_to_crossing"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }

    target_cross: Dict[str, List[Optional[int]]] = {s: [] for s in schemes}
    non_target_cross: Dict[str, List[Optional[int]]] = {s: [] for s in schemes}
    max_q = 0

    for s in schemes:
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            tp = int(r.get("target_provider", 0))
            crossed = r.get("crossed_at") or []
            if not crossed:
                continue
            max_q = max(max_q, int(r.get("trials", 0) or 0))

            t_q: Optional[int] = None
            if 0 <= tp < len(crossed) and isinstance(crossed[tp], int):
                t_q = int(crossed[tp])
            target_cross[s].append(t_q)

            non_qs: List[int] = []
            for i, v in enumerate(crossed):
                if i == tp:
                    continue
                if isinstance(v, int):
                    non_qs.append(int(v))
            non_target_cross[s].append(min(non_qs) if non_qs else None)

    if max_q <= 0:
        print("[WARN] No rag_multi crossing data found.")
        return

    xs = np.arange(1, max_q + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.5), sharex=True)

    def _cdf(vals: List[Optional[int]]) -> np.ndarray:
        if not vals:
            return np.full_like(xs, np.nan, dtype=float)
        out: List[float] = []
        for q in xs:
            hit = sum(1 for v in vals if isinstance(v, int) and v <= int(q))
            out.append(float(hit) / float(len(vals)))
        return np.array(out, dtype=float)

    for s in schemes:
        y = _cdf(target_cross[s])
        axes[0].plot(xs, y, linewidth=2.2, color=colors.get(s), label=_scheme_label(s))
    axes[0].set_ylabel("P(target crossed by q)")
    axes[0].set_title("RAG Time-to-Decision (crossed_at, z>=4 + min_T gate)")
    axes[0].legend()
    axes[0].set_ylim(0.0, 1.0)

    for s in schemes:
        y = _cdf(non_target_cross[s])
        axes[1].plot(xs, y, linewidth=2.2, color=colors.get(s), label=_scheme_label(s))
    axes[1].set_ylabel("P(any non-target crossed by q)")
    axes[1].set_xlabel("# audited answers (q)")
    axes[1].set_ylim(0.0, 1.0)

    _save_fig(fig, figpath)


def plot_rag_queries_to_bh(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_rag_queries_to_bh"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }

    target_bh_q: Dict[str, List[Optional[int]]] = {s: [] for s in schemes}
    non_bh_q: Dict[str, List[Optional[int]]] = {s: [] for s in schemes}
    max_q = 0

    for s in schemes:
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            tp = int(r.get("target_provider", 0))
            min_pooled_T = int(r.get("min_pooled_T", 1))
            traj = r.get("traj") or []
            if not traj:
                continue
            max_q = max(max_q, max(int(snap.get("q", 0) or 0) for snap in traj))
            t_cross: Optional[int] = None
            n_cross: Optional[int] = None
            for snap in traj:
                q = int(snap.get("q", 0) or 0)
                z = snap.get("z") or []
                T = snap.get("pooled_T") or []
                bh = _bh_sig_from_snapshot(z, T, min_pooled_T=min_pooled_T, alpha=0.05)
                if not bh or tp < 0 or tp >= len(bh):
                    continue
                if t_cross is None and bool(bh[tp]):
                    t_cross = q
                if n_cross is None and any(bool(bh[i]) for i in range(len(bh)) if i != tp):
                    n_cross = q
                if t_cross is not None and n_cross is not None:
                    break
            target_bh_q[s].append(t_cross)
            non_bh_q[s].append(n_cross)

    if max_q <= 0:
        print("[WARN] No rag trajectories found for BH time-to-decision plot.")
        return

    xs = np.arange(1, max_q + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.9, 6.5), sharex=True)

    def _cdf(vals: List[Optional[int]]) -> np.ndarray:
        if not vals:
            return np.full_like(xs, np.nan, dtype=float)
        return np.array(
            [sum(1 for v in vals if isinstance(v, int) and v <= int(q)) / float(len(vals)) for q in xs],
            dtype=float,
        )

    for s in schemes:
        axes[0].plot(xs, _cdf(target_bh_q[s]), linewidth=2.2, color=colors.get(s), label=_scheme_label(s))
    axes[0].set_ylabel("P(target BH sig by q)")
    axes[0].set_title("RAG Time-to-BH Decision (alpha=0.05, min_T-gated)")
    axes[0].legend()
    axes[0].set_ylim(0.0, 1.0)

    for s in schemes:
        axes[1].plot(xs, _cdf(non_bh_q[s]), linewidth=2.2, color=colors.get(s), label=_scheme_label(s))
    axes[1].set_ylabel("P(any non-target BH sig by q)")
    axes[1].set_xlabel("# audited answers (q)")
    axes[1].set_ylim(0.0, 1.0)

    _save_fig(fig, figpath)


def plot_rag_fp_anatomy(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_rag_fp_anatomy"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    markers = {
        "dirbucket_naive": "o",
        "kgw_ward": "s",
        "semstamp": "^",
        "sim_attr": "D",
        "overlap_attr": "P",
        "entail_attr": "X",
    }

    pts_non: Dict[str, List[Dict[str, Any]]] = {s: [] for s in schemes}
    pts_tar: Dict[str, List[Dict[str, Any]]] = {s: [] for s in schemes}
    z_thr = 4.0
    min_t_gate = 1
    # KGW pooled_T counts token-level evidence, while DirBucket/SemStamp pooled_T are
    # sentence/window-level units. For visual comparability on this anatomy plot, show
    # KGW x-values in rough sentence-equivalent units.
    kgw_tokens_per_sentence_equiv = _estimate_tokens_per_sentence_from_trial_outputs(results_root, schemes)
    print(
        f"[INFO] RAG FP anatomy x-axis normalization: KGW pooled_T scaled by "
        f"{kgw_tokens_per_sentence_equiv:.2f} answer tokens/sentence (estimated from trial outputs)"
    )
    for s in schemes:
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            tp = int(r.get("target_provider", 0))
            z = r.get("z") or []
            z_raw = r.get("z_raw") or z
            T = r.get("pooled_T") or []
            bh = r.get("bh_sig") or []
            if not isinstance(z, list) or not isinstance(T, list) or tp < 0 or tp >= len(z):
                continue
            # Target point (one per run)
            pts_tar[s].append(
                {
                    "seed": fp.stem,
                    "provider": tp,
                    "T": int(T[tp]) if tp < len(T) else 0,
                    "z_raw": float(z_raw[tp] if tp < len(z_raw) else z[tp]),
                    "bh": bool(bh[tp]) if tp < len(bh) else False,
                }
            )
            # All non-target points (one per provider per run)
            for i in range(len(z)):
                if i == tp:
                    continue
                pts_non[s].append(
                    {
                        "seed": fp.stem,
                        "provider": i,
                        "T": int(T[i]) if i < len(T) else 0,
                        "z_raw": float(z_raw[i] if i < len(z_raw) else z[i]),
                        "bh": bool(bh[i]) if i < len(bh) else False,
                    }
                )
            z_thr = float(r.get("z_threshold", z_thr))
            min_t_gate = int(r.get("min_pooled_T", min_t_gate))

    have_any = any(pts_non[s] or pts_tar[s] for s in schemes)
    if not have_any:
        print("[WARN] No rag_multi files found for FP anatomy plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax_non, ax_tar = axes

    def _x_t_equiv(scheme: str, t_raw: float) -> float:
        if scheme == "kgw_ward":
            return float(t_raw) / kgw_tokens_per_sentence_equiv
        return float(t_raw)

    for s in schemes:
        for rec in pts_non[s]:
            face = colors.get(s) if bool(rec.get("bh")) else "none"
            x_non = _x_t_equiv(s, float(rec.get("T", 0)))
            ax_non.scatter(
                x_non,
                float(rec.get("z_raw", np.nan)),
                s=70,
                marker=markers.get(s, "o"),
                edgecolor=colors.get(s),
                facecolor=face,
                linewidth=1.4,
                alpha=0.65,
            )

        for rec in pts_tar[s]:
            face_t = colors.get(s) if bool(rec.get("bh")) else "none"
            x_tar = _x_t_equiv(s, float(rec.get("T", 0)))
            ax_tar.scatter(
                x_tar,
                float(rec.get("z_raw", np.nan)),
                s=70,
                marker=markers.get(s, "o"),
                edgecolor=colors.get(s),
                facecolor=face_t,
                linewidth=1.4,
                alpha=0.95,
            )

    for ax in axes:
        ax.axhline(z_thr, linestyle="--", linewidth=1.1, color="black")
        ax.axvline(min_t_gate, linestyle="--", linewidth=1.0, color="black", alpha=0.75)
        ax.set_xlabel(
            f"pooled T (sentence-equivalent; KGW scaled by ~{kgw_tokens_per_sentence_equiv:.1f} tok/sent)"
        )
        ax.set_ylabel("pooled z_raw")

    ax_non.set_title("RAG Non-Target Provider Anatomy (all non-targets)\n(filled = non-target BH sig)")
    ax_tar.set_title("RAG Target Detection Support\n(filled = target BH sig)")

    # Legend by scheme + fill semantics
    handles = []
    labels = []
    for s in schemes:
        h = ax_non.scatter([], [], s=70, marker=markers.get(s, "o"), edgecolor=colors.get(s), facecolor="none", linewidth=1.4)
        handles.append(h)
        labels.append(_scheme_label(s))
    h_fill = ax_non.scatter([], [], s=70, marker="o", edgecolor="black", facecolor="black", linewidth=1.0)
    h_open = ax_non.scatter([], [], s=70, marker="o", edgecolor="black", facecolor="none", linewidth=1.0)
    handles.extend([h_fill, h_open])
    labels.extend(["BH significant", "not BH significant"])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    # Use a custom tight-layout rect here so the figure-level legend does not overlap
    # the x-axis labels (the generic _save_fig() has no legend-aware bottom padding).
    out_base = figpath
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(out_base.with_suffix(".png"), dpi=240)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[OK] {out_base.with_suffix('.png')}")
    print(f"[OK] {out_base.with_suffix('.pdf')}")


def plot_rag_fp_anatomy_main(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Section 5.4 Block C main figure (DirBucket-focused).

    Panel A: non-target winner rate vs retrieval mixing level.
    Panel B: non-target winner rate vs alignment ambiguity margin bins.
    Panel C: representative hardest trajectories (target vs max non-target z_raw over q).

    Uses existing rag_multi trial logs and aggregate json outputs only.
    """
    figpath = out_dir / "fig_rag_fp_anatomy_main"
    scheme = "dirbucket_naive"
    if scheme not in schemes:
        print("[WARN] DirBucket not in --schemes; skipping fig_rag_fp_anatomy_main.")
        return

    trial_files = find_rag_trial_logs(results_root, scheme)
    agg_files = find_rag_multi_files(results_root, scheme)
    if not trial_files or not agg_files:
        print("[WARN] Missing rag_multi outputs for fig_rag_fp_anatomy_main.")
        return

    mix_to_nt_win: Dict[int, List[int]] = {}
    margin_vals: List[float] = []
    margin_nt: List[int] = []

    for fp in trial_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    tp = int(row.get("target_provider", 0))
                    wp = row.get("winner_pid")
                    non_target_win = 0 if (isinstance(wp, int) and wp == tp) else 1

                    mix = int(row.get("mix_unique_providers") or 0)
                    if mix <= 0:
                        retrieved = row.get("retrieved_docs") or []
                        rset: set[int] = set()
                        if isinstance(retrieved, list):
                            for rec in retrieved:
                                if not isinstance(rec, dict):
                                    continue
                                pid = rec.get("provider_id")
                                if isinstance(pid, (int, float)):
                                    rset.add(int(pid))
                        mix = len(rset)
                    if mix > 0:
                        mix_to_nt_win.setdefault(int(mix), []).append(int(non_target_win))

                    al = row.get("align") or {}
                    b = al.get("best_cos_mean")
                    s = al.get("second_cos_mean")
                    if isinstance(b, (int, float)) and isinstance(s, (int, float)):
                        margin_vals.append(float(b) - float(s))
                        margin_nt.append(int(non_target_win))
        except Exception:
            continue

    if not mix_to_nt_win:
        print("[WARN] No trial-level mixing stats found for fig_rag_fp_anatomy_main.")
        return

    # Representative hardest trajectories by final max non-target z_raw.
    hard_runs: List[Dict[str, Any]] = []
    run_pat = re.compile(r"rag_multi_t(\d+)_seed(\d+)\.json$")
    for fp in agg_files:
        r = load_json(fp)
        tp = int(r.get("target_provider", 0))
        traj = r.get("traj") or []
        if not isinstance(traj, list) or not traj:
            continue
        qs: List[int] = []
        z_t: List[float] = []
        z_n: List[float] = []
        for snap in traj:
            q = int(snap.get("q", 0) or 0)
            if q <= 0:
                continue
            z = snap.get("z_raw") or snap.get("z") or []
            if not isinstance(z, list) or tp < 0 or tp >= len(z):
                continue
            non = [float(z[i]) for i in range(len(z)) if i != tp]
            if not non:
                continue
            qs.append(q)
            z_t.append(float(z[tp]))
            z_n.append(float(max(non)))
        if not qs:
            continue
        m = run_pat.match(fp.name)
        if m:
            run_id = f"t{int(m.group(1))}/s{int(m.group(2)):02d}"
        else:
            run_id = fp.stem
        hard_runs.append(
            {
                "run_id": run_id,
                "q": qs,
                "z_t": z_t,
                "z_n": z_n,
                "final_max_non": float(z_n[-1]),
            }
        )

    if not hard_runs:
        print("[WARN] No aggregate trajectories found for fig_rag_fp_anatomy_main.")
        return
    hard_runs = sorted(hard_runs, key=lambda x: float(x["final_max_non"]), reverse=True)[:3]

    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.55))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.7
    anno_fs = 6.6

    # Panel A: non-target winner rate vs mixing level (with Wilson CI).
    ax = axs[0]
    mix_levels = sorted(k for k in mix_to_nt_win.keys() if k > 0)
    xs = np.arange(len(mix_levels))
    rates: List[float] = []
    lo_err: List[float] = []
    hi_err: List[float] = []
    for m in mix_levels:
        vals = mix_to_nt_win[m]
        n = len(vals)
        k = int(sum(vals))
        p = float(k) / float(n) if n > 0 else float("nan")
        lo, hi = _wilson_ci(k, n)
        rates.append(p)
        lo_err.append(max(0.0, p - lo))
        hi_err.append(max(0.0, hi - p))
    ax.bar(xs, rates, width=0.62, color="#d62728", edgecolor="black", linewidth=0.6, alpha=0.85, zorder=2)
    ax.errorbar(xs, rates, yerr=np.vstack([lo_err, hi_err]), fmt="none", ecolor="black", elinewidth=0.9, capsize=2.8, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(m) for m in mix_levels], fontsize=tick_fs)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.24, zorder=0)
    ax.set_xlabel("Mixing level (# providers)", fontsize=label_fs)
    ax.set_ylabel("P(non-target wins)", fontsize=label_fs)
    ax.set_title("A. Event Rate vs Mixing", fontsize=title_fs, pad=4.0)

    # Panel B: non-target winner rate vs alignment margin bins (ambiguity anatomy).
    ax = axs[1]
    if margin_vals and len(margin_vals) >= 20:
        marr = np.array(margin_vals, dtype=float)
        yarr = np.array(margin_nt, dtype=float)
        edges = np.quantile(marr, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Guard against duplicate quantile edges by adding tiny jitter.
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        bx = np.arange(len(edges) - 1)
        br: List[float] = []
        blo: List[float] = []
        bhi: List[float] = []
        blabels: List[str] = []
        for i in range(len(edges) - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if i == len(edges) - 2:
                mask = (marr >= lo) & (marr <= hi)
            else:
                mask = (marr >= lo) & (marr < hi)
            sub = yarr[mask]
            n = int(sub.size)
            k = int(np.sum(sub)) if n > 0 else 0
            p = float(k) / float(n) if n > 0 else float("nan")
            ci_lo, ci_hi = _wilson_ci(k, n)
            br.append(p)
            blo.append(max(0.0, p - ci_lo))
            bhi.append(max(0.0, ci_hi - p))
            blabels.append(f"{lo:.02f}-{hi:.02f}")
        ax.plot(bx, br, color="#ff7f0e", marker="o", linewidth=1.7, markersize=4.2, zorder=3)
        ax.errorbar(bx, br, yerr=np.vstack([blo, bhi]), fmt="none", ecolor="#ff7f0e", elinewidth=0.9, capsize=2.6, zorder=2)
        ax.set_xticks(bx)
        ax.set_xticklabels(blabels, fontsize=tick_fs, rotation=18, ha="right")
    else:
        ax.text(0.5, 0.5, "insufficient margin data", ha="center", va="center", fontsize=anno_fs)
        ax.set_xticks([])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.24, zorder=0)
    ax.set_xlabel("Alignment margin bin", fontsize=label_fs)
    ax.set_ylabel("P(non-target wins)", fontsize=label_fs)
    ax.set_title("B. Event Rate vs Ambiguity", fontsize=title_fs, pad=4.0)

    # Panel C: representative hardest trajectories over audit budget.
    ax = axs[2]
    traj_colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, run in enumerate(hard_runs):
        c = traj_colors[idx % len(traj_colors)]
        q = np.array(run["q"], dtype=float)
        z_t = np.array(run["z_t"], dtype=float)
        z_n = np.array(run["z_n"], dtype=float)
        ax.plot(q, z_t, color=c, linewidth=1.6, alpha=0.95)
        ax.plot(q, z_n, color=c, linewidth=1.4, alpha=0.95, linestyle="--")
        ax.text(
            float(q[-1]) + 0.8,
            float(z_n[-1]),
            str(run["run_id"]),
            color=c,
            fontsize=anno_fs,
            va="center",
            ha="left",
        )
    ax.axhline(4.0, linestyle="--", linewidth=0.9, color="#666666")
    ax.grid(axis="y", linestyle=":", alpha=0.24)
    ax.set_xlabel("Audited answers (q)", fontsize=label_fs)
    ax.set_ylabel("Pooled z_raw", fontsize=label_fs)
    ax.set_title("C. Hardest Trajectories", fontsize=title_fs, pad=4.0)
    ax.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.6)
    handles = [
        Line2D([0], [0], color="black", linewidth=1.6, linestyle="-", label="target"),
        Line2D([0], [0], color="black", linewidth=1.4, linestyle="--", label="max non-target"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=legend_fs, frameon=True)

    for a in axs[:2]:
        a.tick_params(axis="both", labelsize=tick_fs, width=0.8, length=2.6)

    _save_fig(fig, figpath)


def _collect_mixing_envelope_stats(
    results_root: Path,
    scheme: str,
    *,
    min_pooled_t_fallback: int = 25,
) -> Dict[int, Dict[str, List[float]]]:
    """Collect trial-level mixing stats for DirBucket robustness diagnostics."""
    trial_files = find_rag_trial_logs(results_root, scheme)
    if not trial_files:
        return {}

    min_pooled_t = int(min_pooled_t_fallback)
    for fp in find_rag_multi_files(results_root, scheme):
        r = load_json(fp)
        mt = r.get("min_pooled_T")
        if isinstance(mt, (int, float)):
            min_pooled_t = int(mt)
            break

    by_mix: Dict[int, Dict[str, List[float]]] = {}
    for fp in trial_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    tp = int(row.get("target_provider", 0))
                    mix = int(row.get("mix_unique_providers") or 0)
                    if mix <= 0:
                        retrieved = row.get("retrieved_docs") or []
                        rset: set[int] = set()
                        if isinstance(retrieved, list):
                            for rec in retrieved:
                                if not isinstance(rec, dict):
                                    continue
                                pid = rec.get("provider_id")
                                if isinstance(pid, (int, float)):
                                    rset.add(int(pid))
                        mix = len(rset)
                    if mix <= 0:
                        continue

                    snap = row.get("traj_snapshot") or {}
                    z_snap = snap.get("z") or []
                    t_snap = snap.get("pooled_T") or []
                    if not isinstance(z_snap, list) or tp < 0 or tp >= len(z_snap):
                        continue
                    non_snap = [float(z_snap[i]) for i in range(len(z_snap)) if i != tp]
                    if not non_snap:
                        continue
                    z_t = float(z_snap[tp])
                    z_n = float(max(non_snap))

                    bh = _bh_sig_from_snapshot(z_snap, t_snap, min_pooled_T=min_pooled_t, alpha=0.05)
                    target_bh = bool(bh[tp]) if bh and tp < len(bh) else False
                    any_non_bh = bool(any(bool(bh[i]) for i in range(len(bh)) if i != tp)) if bh else False

                    rec = by_mix.setdefault(
                        int(mix),
                        {
                            "target_bh": [],
                            "non_bh": [],
                            "target_z": [],
                            "max_non_z": [],
                        },
                    )
                    rec["target_bh"].append(1.0 if target_bh else 0.0)
                    rec["non_bh"].append(1.0 if any_non_bh else 0.0)
                    rec["target_z"].append(z_t)
                    rec["max_non_z"].append(z_n)
        except Exception:
            continue
    return by_mix


def plot_mixing_envelope_main(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    DirBucket-only mixing envelope plot for robustness checks.

    Panel A: BH rates by mixing level (target BH, any non-target BH).
    Panel B: per-answer evidence by mixing level (target z vs max non-target z).
    """
    figpath = out_dir / "fig_mixing_envelope_main"
    scheme = "dirbucket_naive"
    if scheme not in schemes:
        print("[WARN] DirBucket not in --schemes; skipping fig_mixing_envelope_main.")
        return

    by_mix = _collect_mixing_envelope_stats(results_root, scheme, min_pooled_t_fallback=25)
    if not by_mix:
        print("[WARN] No rag trial logs found for fig_mixing_envelope_main.")
        return

    mix_levels = sorted(by_mix.keys())
    if not mix_levels:
        print("[WARN] No usable mixing-level samples for fig_mixing_envelope_main.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.7

    # Panel A: BH rates vs mixing.
    ax = axs[0]
    xs = np.array(mix_levels, dtype=float)
    t_rate: List[float] = []
    t_lo: List[float] = []
    t_hi: List[float] = []
    n_rate: List[float] = []
    n_lo: List[float] = []
    n_hi: List[float] = []
    for m in mix_levels:
        rec = by_mix[m]
        n = len(rec["target_bh"])
        kt = int(sum(rec["target_bh"]))
        kn = int(sum(rec["non_bh"]))
        pt = float(kt) / float(n) if n > 0 else float("nan")
        pn = float(kn) / float(n) if n > 0 else float("nan")
        lo_t, hi_t = _wilson_ci(kt, n)
        lo_n, hi_n = _wilson_ci(kn, n)
        t_rate.append(pt)
        t_lo.append(max(0.0, pt - lo_t))
        t_hi.append(max(0.0, hi_t - pt))
        n_rate.append(pn)
        n_lo.append(max(0.0, pn - lo_n))
        n_hi.append(max(0.0, hi_n - pn))

    ax.errorbar(xs, t_rate, yerr=np.vstack([t_lo, t_hi]), marker="o", color="#1f77b4", linewidth=1.7, markersize=4.0, capsize=2.4, label="Target BH rate")
    ax.errorbar(xs, n_rate, yerr=np.vstack([n_lo, n_hi]), marker="s", color="#d62728", linewidth=1.5, markersize=3.8, capsize=2.4, label="Any non-target BH rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs, width=0.8, length=2.6)
    ax.grid(axis="y", linestyle=":", alpha=0.24)
    ax.set_xlabel("Mixing level (# providers)", fontsize=label_fs)
    ax.set_ylabel("BH rate", fontsize=label_fs)
    ax.set_title("A. Decision Robustness", fontsize=title_fs, pad=4.0)
    ax.legend(loc="lower left", fontsize=legend_fs, frameon=True)

    # Panel B: per-answer z separation vs mixing.
    ax = axs[1]
    t_med: List[float] = []
    t_q1: List[float] = []
    t_q3: List[float] = []
    n_med: List[float] = []
    n_q1: List[float] = []
    n_q3: List[float] = []
    for m in mix_levels:
        rec = by_mix[m]
        tz = np.array(rec["target_z"], dtype=float)
        nz = np.array(rec["max_non_z"], dtype=float)
        t_med.append(float(np.nanmedian(tz)))
        t_q1.append(float(np.nanquantile(tz, 0.25)))
        t_q3.append(float(np.nanquantile(tz, 0.75)))
        n_med.append(float(np.nanmedian(nz)))
        n_q1.append(float(np.nanquantile(nz, 0.25)))
        n_q3.append(float(np.nanquantile(nz, 0.75)))

    ax.plot(xs, t_med, color="#1f77b4", marker="o", linewidth=1.8, markersize=4.0, label="Target z")
    ax.fill_between(xs, np.array(t_q1, dtype=float), np.array(t_q3, dtype=float), color="#1f77b4", alpha=0.16)
    ax.plot(xs, n_med, color="#ff7f0e", marker="s", linewidth=1.6, markersize=3.8, linestyle="--", label="Max non-target z")
    ax.fill_between(xs, np.array(n_q1, dtype=float), np.array(n_q3, dtype=float), color="#ff7f0e", alpha=0.16)
    ax.axhline(4.0, linestyle="--", linewidth=0.9, color="#666666")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs, width=0.8, length=2.6)
    ax.grid(axis="y", linestyle=":", alpha=0.24)
    ax.set_xlabel("Mixing level (# providers)", fontsize=label_fs)
    ax.set_ylabel("Per-answer z", fontsize=label_fs)
    ax.set_title("B. Evidence Separation", fontsize=title_fs, pad=4.0)
    ax.legend(loc="upper right", fontsize=legend_fs, frameon=True)

    _save_fig(fig, figpath)


def _pick_mixing_stress_root(primary_root: Path, override: Path | None = None) -> Path | None:
    candidates: List[Path] = []
    if override is not None:
        candidates.append(override)
    candidates.extend(
        [
            Path("results/watermark_compare_global_mixstress_k9_u5"),
            Path("results/watermark_compare_global_mixstress_pilot_k9_u5"),
            primary_root.parent / "watermark_compare_global_mixstress_k9_u5",
            primary_root.parent / "watermark_compare_global_mixstress_pilot_k9_u5",
        ]
    )
    best_root: Path | None = None
    best_count = 0
    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        n = len(find_rag_trial_logs(c, "dirbucket_naive"))
        if n > best_count:
            best_count = n
            best_root = c
    return best_root if best_root is not None and best_count > 0 else None


def plot_style_mixing_robustness_main(
    results_root: Path,
    out_dir: Path,
    schemes: List[str],
    *,
    mixing_stress_root: Path | None = None,
) -> None:
    """
    Two-panel Section 5.4 robustness figure:
      Panel A: target pooled z vs mixing (baseline extended with stress data).
      Panel B: non-target BH significance probability vs mixing.
    """
    figpath = out_dir / "fig_style_mixing_robustness_main"
    scheme = "dirbucket_naive"
    if scheme not in schemes:
        print("[WARN] DirBucket not in --schemes; skipping fig_style_mixing_robustness_main.")
        return

    # ---- Panel A data (mixing robustness baseline + stress) ----
    base_mix = _collect_mixing_envelope_stats(results_root, scheme, min_pooled_t_fallback=25)
    stress_root = _pick_mixing_stress_root(results_root, mixing_stress_root)
    stress_mix: Dict[int, Dict[str, List[float]]] = {}
    if stress_root is not None:
        stress_mix = _collect_mixing_envelope_stats(stress_root, scheme, min_pooled_t_fallback=25)
        if not stress_mix:
            print(f"[WARN] No stress trial logs in {stress_root}; plotting baseline only.")
    else:
        print("[WARN] No stress root found; plotting baseline only for mixing panel.")

    if not base_mix:
        print("[WARN] Missing rag trial logs for baseline mixing panel in fig_style_mixing_robustness_main.")
        return

    # Collect q-indexed target-z values so Panel B can control for audit budget q.
    def _collect_mixing_q_target(root: Path) -> Dict[int, Dict[int, List[float]]]:
        out: Dict[int, Dict[int, List[float]]] = {}
        for fp in find_rag_trial_logs(root, scheme):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        tp = int(row.get("target_provider", 0))
                        mix = int(row.get("mix_unique_providers") or 0)
                        if mix <= 0:
                            continue
                        snap = row.get("traj_snapshot") or {}
                        q = int(snap.get("q") or row.get("q_effective") or 0)
                        z = snap.get("z") or []
                        if q <= 0 or not isinstance(z, list) or tp < 0 or tp >= len(z):
                            continue
                        out.setdefault(int(mix), {}).setdefault(int(q), []).append(float(z[tp]))
            except Exception:
                continue
        return out

    base_mix_q = _collect_mixing_q_target(results_root)
    stress_mix_q = _collect_mixing_q_target(stress_root) if stress_root is not None else {}

    # ---- Figure ----
    fig, axs = plt.subplots(2, 1, figsize=(4.35, 5.15))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.7
    suptitle_fs = 9.3

    # Panel A
    ax = axs[0]
    mix_levels = sorted(set(list(base_mix.keys()) + list(stress_mix.keys())))
    xs = np.array(mix_levels, dtype=float)

    def _summ(by_mix: Dict[int, Dict[str, List[float]]], key: str) -> Tuple[List[float], List[float], List[float]]:
        med: List[float] = []
        q1: List[float] = []
        q3: List[float] = []
        for m in mix_levels:
            arr = np.array((by_mix.get(m) or {}).get(key) or [], dtype=float)
            if arr.size == 0:
                med.append(float("nan"))
                q1.append(float("nan"))
                q3.append(float("nan"))
            else:
                med.append(float(np.nanmedian(arr)))
                q1.append(float(np.nanquantile(arr, 0.25)))
                q3.append(float(np.nanquantile(arr, 0.75)))
        return med, q1, q3

    # Merge baseline + stress samples by mixing level (no separate styling).
    merged_mix: Dict[int, Dict[str, List[float]]] = {}
    for m in mix_levels:
        mt = list((base_mix.get(m) or {}).get("target_z") or [])
        mn = list((base_mix.get(m) or {}).get("max_non_z") or [])
        nbh = list((base_mix.get(m) or {}).get("non_bh") or [])
        if stress_mix:
            mt.extend(list((stress_mix.get(m) or {}).get("target_z") or []))
            mn.extend(list((stress_mix.get(m) or {}).get("max_non_z") or []))
            nbh.extend(list((stress_mix.get(m) or {}).get("non_bh") or []))
        if mt or mn:
            merged_mix[m] = {"target_z": mt, "max_non_z": mn, "non_bh": nbh}

    # Merge q-indexed target-z streams and compute q-controlled summaries:
    # each mix is summarized over the same shared q support to remove budget confounding.
    merged_mix_q: Dict[int, Dict[int, List[float]]] = {}
    for m in mix_levels:
        by_q: Dict[int, List[float]] = {}
        for q, vals in (base_mix_q.get(m) or {}).items():
            by_q.setdefault(int(q), []).extend(list(vals))
        for q, vals in (stress_mix_q.get(m) or {}).items():
            by_q.setdefault(int(q), []).extend(list(vals))
        if by_q:
            merged_mix_q[int(m)] = by_q

    shared_q: List[int] = []
    if mix_levels and all((m in merged_mix_q and merged_mix_q[m]) for m in mix_levels):
        q_sets = [set(merged_mix_q[m].keys()) for m in mix_levels]
        shared_q = sorted(list(set.intersection(*q_sets))) if q_sets else []

    m_t: List[float] = []
    m_tq1: List[float] = []
    m_tq3: List[float] = []
    if len(shared_q) >= 5:
        for m in mix_levels:
            per_q_medians: List[float] = []
            by_q = merged_mix_q.get(m) or {}
            for q in shared_q:
                arr = np.array(by_q.get(q) or [], dtype=float)
                if arr.size > 0:
                    per_q_medians.append(float(np.nanmedian(arr)))
            if per_q_medians:
                arr_m = np.array(per_q_medians, dtype=float)
                m_t.append(float(np.nanmean(arr_m)))
                m_tq1.append(float(np.nanquantile(arr_m, 0.25)))
                m_tq3.append(float(np.nanquantile(arr_m, 0.75)))
            else:
                m_t.append(float("nan"))
                m_tq1.append(float("nan"))
                m_tq3.append(float("nan"))
    else:
        # Fallback to the original aggregate summary if shared q support is too small.
        m_t, m_tq1, m_tq3 = _summ(merged_mix, "target_z")

    ax.plot(xs, m_t, color="#1f77b4", marker="o", linewidth=1.8, markersize=4.0, label="Target z")
    ax.fill_between(xs, np.array(m_tq1, dtype=float), np.array(m_tq3, dtype=float), color="#1f77b4", alpha=0.14)

    ax.axhline(4.0, linestyle="--", linewidth=0.95, color="#666666")
    ax.grid(axis="y", linestyle=":", alpha=0.24)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(v)) for v in xs], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs, width=0.8, length=2.6)
    ax.set_xlabel("Mixing level (# providers)", fontsize=label_fs)
    ax.set_ylabel("Pooled Z", fontsize=label_fs)
    ax.set_title("A. Mixing Robustness", fontsize=title_fs, pad=4.0)

    # Panel B: non-target BH significance probability by mixing level.
    ax = axs[1]
    p_non: List[float] = []
    e_lo: List[float] = []
    e_hi: List[float] = []
    for m in mix_levels:
        arr = np.array((merged_mix.get(m) or {}).get("non_bh") or [], dtype=float)
        n = int(arr.size)
        k = int(np.sum(arr)) if n > 0 else 0
        p = (float(k) / float(n)) if n > 0 else float("nan")
        lo, hi = _wilson_ci(k, n)
        p_non.append(p)
        e_lo.append(max(0.0, p - lo))
        e_hi.append(max(0.0, hi - p))
    ax.errorbar(
        xs,
        np.array(p_non, dtype=float),
        yerr=np.vstack([np.array(e_lo, dtype=float), np.array(e_hi, dtype=float)]),
        color="#d62728",
        marker="o",
        linewidth=1.7,
        markersize=4.0,
        capsize=2.6,
    )
    ax.axhline(0.05, linestyle="--", linewidth=0.9, color="#666666")
    ymax = 0.052
    if p_non:
        ymax = max(ymax, float(np.nanmax(np.array(p_non, dtype=float) + np.array(e_hi, dtype=float))) + 0.004)
    ax.set_ylim(0.0, min(1.0, ymax))
    ax.grid(axis="y", linestyle=":", alpha=0.24)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(v)) for v in xs], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs, width=0.8, length=2.6)
    ax.set_xlabel("Mixing level (# providers)", fontsize=label_fs)
    ax.set_ylabel("P(non-target BH sig)", fontsize=label_fs)
    ax.set_title("B. BH False-Positive Rate", fontsize=title_fs, pad=4.0)

    fig.suptitle("Dirbucket Robustness to Provider Mixing", fontsize=suptitle_fs, y=0.975)
    _save_fig(fig, figpath, tight_rect=(0.0, 0.0, 1.0, 0.955))


def plot_utility_tradeoff(
    results_root: Path,
    out_dir: Path,
    schemes: List[str],
    work_dirs: Dict[str, Path],
    *,
    manifest_src: Path,
    retrieval_model_name: str = "all-MiniLM-L6-v2",
    topk: int = 5,
    max_queries: int = 0,
    sample_seed: int = 13,
) -> None:
    """
    Utility preservation plot (DirBucket-only):
      Panel A: retrieval utility on shared query set (source vs watermarked corpus).
      Panel B: corpus preservation (source->watermarked similarity/retention metrics).
    """
    figpath = out_dir / "fig_utility_tradeoff"
    scheme = "dirbucket_naive"
    if scheme not in schemes:
        print("[WARN] DirBucket not in --schemes; skipping fig_utility_tradeoff.")
        return

    work_dir = work_dirs.get(scheme)
    if work_dir is None:
        print("[WARN] Missing work_dir for DirBucket; skipping fig_utility_tradeoff.")
        return
    corpus_path = work_dir / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"[WARN] Missing {corpus_path}; skipping fig_utility_tradeoff.")
        return

    src_map = _load_manifest_source_map(manifest_src)
    if not src_map:
        print(f"[WARN] Missing/empty source manifest {manifest_src}; skipping fig_utility_tradeoff.")
        return

    wm_text_by_id: Dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            did = str(row.get("id") or "").strip()
            if not did:
                continue
            txt = str(
                row.get("text_wm")
                or row.get("text_wm_dirbucket")
                or row.get("text")
                or ""
            ).strip()
            if not txt:
                continue
            wm_text_by_id[did] = txt

    common_ids = sorted(set(src_map.keys()) & set(wm_text_by_id.keys()))
    if len(common_ids) < 50:
        print(
            f"[WARN] Too few source/watermarked matches ({len(common_ids)}); "
            "skipping fig_utility_tradeoff."
        )
        return

    id_to_idx = {did: i for i, did in enumerate(common_ids)}
    src_texts = [src_map[did] for did in common_ids]
    wm_texts = [wm_text_by_id[did] for did in common_ids]

    # Build shared query set from existing rag_multi trials (no new API calls).
    q_items: List[Tuple[str, str]] = []  # (question, target_doc_id)
    seen_keys: set[Tuple[str, str]] = set()
    for fp in find_rag_trial_logs(results_root, scheme):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    q = str(row.get("question") or "").strip()
                    did = str(row.get("target_doc_id") or "").strip()
                    if (not q) or (did not in id_to_idx):
                        continue
                    qh = str(row.get("question_hash") or "").strip()
                    key = (qh if qh else q, did)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    q_items.append((q, did))
        except Exception:
            continue

    if not q_items:
        print("[WARN] No query set found in rag trial logs; skipping fig_utility_tradeoff.")
        return

    if max_queries > 0 and len(q_items) > int(max_queries):
        rng = np.random.default_rng(int(sample_seed))
        idx = rng.choice(len(q_items), size=int(max_queries), replace=False)
        q_items = [q_items[int(i)] for i in sorted(idx.tolist())]

    q_texts = [q for q, _ in q_items]
    q_targets = [did for _, did in q_items]

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"[WARN] sentence-transformers unavailable ({e}); skipping fig_utility_tradeoff.")
        return

    model = SentenceTransformer(retrieval_model_name)
    src_emb = np.asarray(
        model.encode(src_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    wm_emb = np.asarray(
        model.encode(wm_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    q_emb = np.asarray(
        model.encode(q_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )

    # Retrieval utility (source vs watermarked).
    k = int(max(1, min(int(topk), len(common_ids))))
    sim_src = q_emb @ src_emb.T
    sim_wm = q_emb @ wm_emb.T
    top_src = np.argpartition(-sim_src, kth=k - 1, axis=1)[:, :k]
    top_wm = np.argpartition(-sim_wm, kth=k - 1, axis=1)[:, :k]

    hit_src: List[float] = []
    hit_wm: List[float] = []
    mrr_src: List[float] = []
    mrr_wm: List[float] = []
    topk_jaccard: List[float] = []
    rank_consistency: List[float] = []

    for qi, did in enumerate(q_targets):
        tgt_idx = id_to_idx.get(did, -1)
        if tgt_idx < 0:
            continue
        src_ord = top_src[qi][np.argsort(-sim_src[qi, top_src[qi]])]
        wm_ord = top_wm[qi][np.argsort(-sim_wm[qi, top_wm[qi]])]

        src_rank: int | None = None
        wm_rank: int | None = None
        for r, d_idx in enumerate(src_ord.tolist(), start=1):
            if int(d_idx) == int(tgt_idx):
                src_rank = r
                break
        for r, d_idx in enumerate(wm_ord.tolist(), start=1):
            if int(d_idx) == int(tgt_idx):
                wm_rank = r
                break

        hit_src.append(1.0 if src_rank is not None else 0.0)
        hit_wm.append(1.0 if wm_rank is not None else 0.0)
        mrr_src.append((1.0 / float(src_rank)) if src_rank is not None else 0.0)
        mrr_wm.append((1.0 / float(wm_rank)) if wm_rank is not None else 0.0)
        s_set = set(int(i) for i in src_ord.tolist())
        w_set = set(int(i) for i in wm_ord.tolist())
        topk_jaccard.append(float(len(s_set & w_set)) / float(max(1, len(s_set | w_set))))
        src_rank_eff = int(src_rank) if src_rank is not None else (k + 1)
        wm_rank_eff = int(wm_rank) if wm_rank is not None else (k + 1)
        rank_consistency.append(max(0.0, 1.0 - (abs(src_rank_eff - wm_rank_eff) / float(k))))

    # Corpus preservation metric (doc-level).
    sem_sim = np.sum(src_emb * wm_emb, axis=1).astype(float).tolist()

    # Summaries + CI.
    def _rate_and_ci(v: List[float]) -> Tuple[float, float, float]:
        arr = np.array(v, dtype=float)
        if arr.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        k_ = int(np.sum(arr > 0.5))
        n_ = int(arr.size)
        p_ = float(k_) / float(n_)
        lo_, hi_ = _wilson_ci(k_, n_)
        return (p_, max(0.0, p_ - lo_), max(0.0, hi_ - p_))

    hit_src_m, hit_src_lo, hit_src_hi = _rate_and_ci(hit_src)
    hit_wm_m, hit_wm_lo, hit_wm_hi = _rate_and_ci(hit_wm)

    mrr_src_m, mrr_src_hw = mean_ci(mrr_src)
    mrr_wm_m, mrr_wm_hw = mean_ci(mrr_wm)

    sem_m, sem_hw = mean_ci(sem_sim)
    jac_m, jac_hw = mean_ci(topk_jaccard)
    rnk_m, rnk_hw = mean_ci(rank_consistency)

    # ---- Plot (ACL two-column style) ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.9))
    title_fs = 8.4
    label_fs = 7.5
    tick_fs = 7.0
    legend_fs = 6.7

    # Panel A: retrieval utility (before/after watermarking).
    labels_a = ["Target hit@5", "Target MRR@5"]
    x = np.arange(len(labels_a), dtype=float)
    w = 0.34
    src_vals = np.array([hit_src_m, mrr_src_m], dtype=float)
    wm_vals = np.array([hit_wm_m, mrr_wm_m], dtype=float)
    src_err = np.array([[hit_src_lo, mrr_src_hw], [hit_src_hi, mrr_src_hw]], dtype=float)
    wm_err = np.array([[hit_wm_lo, mrr_wm_hw], [hit_wm_hi, mrr_wm_hw]], dtype=float)

    ax_a.bar(
        x - w / 2.0,
        src_vals,
        width=w,
        yerr=src_err,
        capsize=2.8,
        color="#9fa2a6",
        edgecolor="black",
        linewidth=0.6,
        label="Source corpus",
        zorder=2,
    )
    ax_a.bar(
        x + w / 2.0,
        wm_vals,
        width=w,
        yerr=wm_err,
        capsize=2.8,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
        label="DirBucket-watermarked",
        zorder=2,
    )
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels_a, fontsize=tick_fs)
    ax_a.tick_params(axis="y", labelsize=tick_fs, width=0.8, length=2.6)
    ax_a.grid(axis="y", linestyle=":", alpha=0.24, zorder=0)
    ax_a.set_ylabel("Score", fontsize=label_fs)
    ax_a.set_title("A. Retrieval Utility", fontsize=title_fs, pad=4.0)
    ax_a.legend(loc="lower left", fontsize=legend_fs, frameon=True)

    # Panel B: source-vs-watermarked consistency metrics (ideal = 1).
    labels_b = ["Top-k doc\noverlap", "Target-rank\nconsistency", "Doc semantic\nsimilarity"]
    y = np.array([jac_m, rnk_m, sem_m], dtype=float)
    yerr = np.array([jac_hw, rnk_hw, sem_hw], dtype=float)
    xb = np.arange(len(labels_b), dtype=float)
    ax_b.bar(
        xb,
        y,
        yerr=yerr,
        capsize=2.8,
        width=0.62,
        color="#4e79a7",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    ax_b.axhline(1.0, linestyle="--", linewidth=0.9, color="#666666", alpha=0.9, zorder=1)
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(labels_b, fontsize=tick_fs)
    ax_b.tick_params(axis="y", labelsize=tick_fs, width=0.8, length=2.6)
    ax_b.grid(axis="y", linestyle=":", alpha=0.24, zorder=0)
    ax_b.set_ylabel("Relative score", fontsize=label_fs)
    ax_b.set_title("B. Retrieval/Content Consistency", fontsize=title_fs, pad=4.0)

    fig.text(
        0.5,
        0.015,
        f"DirBucket utility diagnostics: n_docs={len(common_ids)}, n_queries={len(hit_src)}, top-k={k}",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#444444",
    )
    _save_fig(fig, figpath, tight_rect=(0.0, 0.04, 1.0, 1.0))

    print(
        "[INFO] fig_utility_tradeoff "
        f"hit@{k}: src={hit_src_m:.4f}, wm={hit_wm_m:.4f}; "
        f"MRR@{k}: src={mrr_src_m:.4f}, wm={mrr_wm_m:.4f}; "
        f"topk_overlap={jac_m:.4f}, rank_consistency={rnk_m:.4f}, sem={sem_m:.4f}"
    )


def plot_utility_snippet_examples(
    out_dir: Path,
    *,
    manifest_src: Path,
    work_dir_dirbucket: Path,
    n_examples: int = 2,
    snippet_sentences: int = 3,
) -> None:
    """
    Qualitative utility figure: side-by-side source vs DirBucket-watermarked snippets.
    """
    figpath = out_dir / "fig_utility_snippets"
    corpus_path = work_dir_dirbucket / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"[WARN] Missing {corpus_path}; skipping fig_utility_snippets.")
        return

    src_rows: Dict[str, Dict[str, Any]] = {}
    with manifest_src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            did = str(row.get("id") or "").strip()
            txt = str(row.get("text_src") or row.get("text") or "").strip()
            if did and txt:
                src_rows[did] = {
                    "text": txt,
                    "title": str(row.get("title") or ""),
                    "provider_id": row.get("provider_id"),
                }

    wm_rows: Dict[str, Dict[str, Any]] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            did = str(row.get("id") or "").strip()
            txt = str(row.get("text_wm") or row.get("text_wm_dirbucket") or "").strip()
            if not did or not txt:
                continue
            wm_rows[did] = {
                "text": txt,
                "changed_frac": float(row.get("wm_changed_frac")) if isinstance(row.get("wm_changed_frac"), (int, float)) else None,
                "title": str(row.get("title") or ""),
            }

    common_ids = sorted(set(src_rows.keys()) & set(wm_rows.keys()))
    if len(common_ids) == 0:
        print("[WARN] No source/watermarked overlap for fig_utility_snippets.")
        return

    # Pick representative examples by semantic-aligned local windows:
    # preserve meaning while still showing paraphrastic changes.
    def _jaccard_unigram(a: str, b: str) -> float:
        sa = set(_simple_tokens(a))
        sb = set(_simple_tokens(b))
        den = len(sa | sb)
        return (float(len(sa & sb)) / float(den)) if den > 0 else 0.0

    def _clean_sentence_split(txt: str) -> List[str]:
        sents = _simple_sentence_spans(txt)
        if not sents:
            return []
        # Merge common abbreviation fragments split by simple regex.
        abbr_end = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.$", re.IGNORECASE)
        merged: List[str] = []
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if merged and abbr_end.search(merged[-1]):
                merged[-1] = f"{merged[-1]} {s}"
            else:
                merged.append(s)
        return merged

    def _window_chunks(sents: List[str], n_sent: int) -> List[Tuple[int, str]]:
        n = max(1, int(n_sent))
        out: List[Tuple[int, str]] = []
        if not sents:
            return out
        if len(sents) <= n:
            chunk = " ".join(sents).strip()
            if len(_simple_tokens(chunk)) >= 18:
                out.append((0, chunk))
            return out
        for i in range(len(sents) - n + 1):
            chunk = " ".join(sents[i : i + n]).strip()
            if len(_simple_tokens(chunk)) >= 18:
                out.append((i, chunk))
        if not out:
            chunk = " ".join(sents[:n]).strip()
            if chunk:
                out.append((0, chunk))
        return out

    try:
        from sentence_transformers import SentenceTransformer

        sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        sem_model = None

    picked_rows: List[Dict[str, Any]] = []
    if sem_model is not None:
        candidates: List[Dict[str, Any]] = []
        for did in common_ids:
            src_s = _clean_sentence_split(str(src_rows[did]["text"]))
            wm_s = _clean_sentence_split(str(wm_rows[did]["text"]))
            src_w = _window_chunks(src_s, snippet_sentences)
            wm_w = _window_chunks(wm_s, snippet_sentences)
            if not src_w or not wm_w:
                continue

            src_chunks = [x[1] for x in src_w]
            wm_chunks = [x[1] for x in wm_w]
            vec = sem_model.encode(src_chunks + wm_chunks, normalize_embeddings=True, convert_to_numpy=True)
            src_vec = vec[: len(src_chunks)]
            wm_vec = vec[len(src_chunks) :]
            sim_m = np.matmul(src_vec, wm_vec.T)
            flat_ord = np.argsort(sim_m, axis=None)[::-1]

            best: Optional[Dict[str, Any]] = None
            for flat_idx in flat_ord[: min(len(flat_ord), 120)]:
                i, j = np.unravel_index(int(flat_idx), sim_m.shape)
                sim = float(sim_m[i, j])
                if sim < 0.88:
                    break
                src_chunk = src_chunks[int(i)]
                wm_chunk = wm_chunks[int(j)]
                src_len = len(_simple_tokens(src_chunk))
                wm_len = len(_simple_tokens(wm_chunk))
                if src_len < 18 or wm_len < 18:
                    continue
                len_ratio = float(wm_len) / float(max(1, src_len))
                if len_ratio < 0.72 or len_ratio > 1.35:
                    continue
                ov = _jaccard_unigram(src_chunk, wm_chunk)
                if ov < 0.20 or ov > 0.58:
                    continue
                # Prefer high semantic match, similar snippet length, and non-trivial paraphrase.
                score = sim - 0.18 * abs(len_ratio - 1.0) - 0.30 * abs(ov - 0.38)
                best = {
                    "did": did,
                    "src_txt": src_chunk,
                    "wm_txt": wm_chunk,
                    "sim": sim,
                    "ov": ov,
                    "len_ratio": len_ratio,
                    "score": float(score),
                }
                break
            if best is not None:
                candidates.append(best)

        candidates.sort(key=lambda r: float(r["score"]), reverse=True)
        seen_did: set[str] = set()
        for row in candidates:
            did = str(row["did"])
            if did in seen_did:
                continue
            row["title"] = str(src_rows[did].get("title") or wm_rows[did].get("title") or "")
            picked_rows.append(row)
            seen_did.add(did)
            if len(picked_rows) >= int(max(1, n_examples)):
                break

    if not picked_rows:
        # Fallback: lexical-overlap mid-quantile picks using first sentences.
        cand: List[Tuple[float, str]] = []
        for did in common_ids:
            ov = _jaccard_unigram(str(src_rows[did]["text"]), str(wm_rows[did]["text"]))
            cand.append((float(ov), did))
        cand.sort(key=lambda x: x[0])
        q_positions = np.linspace(0.25, 0.75, num=max(1, int(n_examples)))
        picked = []
        for q in q_positions.tolist():
            idx = int(round(q * (len(cand) - 1)))
            picked.append(cand[max(0, min(len(cand) - 1, idx))][1])

        for did in picked[: max(1, int(n_examples))]:
            src_s = _clean_sentence_split(str(src_rows[did]["text"]))
            wm_s = _clean_sentence_split(str(wm_rows[did]["text"]))
            src_txt = " ".join(src_s[: max(1, int(snippet_sentences))]).strip() if src_s else str(src_rows[did]["text"])
            wm_txt = " ".join(wm_s[: max(1, int(snippet_sentences))]).strip() if wm_s else str(wm_rows[did]["text"])
            ov = _jaccard_unigram(src_txt, wm_txt)
            picked_rows.append(
                {
                    "did": did,
                    "src_txt": src_txt,
                    "wm_txt": wm_txt,
                    "sim": float("nan"),
                    "ov": ov,
                    "len_ratio": float(len(_simple_tokens(wm_txt)) / max(1, len(_simple_tokens(src_txt)))),
                    "score": ov,
                    "title": str(src_rows[did].get("title") or wm_rows[did].get("title") or ""),
                }
            )

    picked_rows = picked_rows[: max(1, int(n_examples))]
    n_rows = len(picked_rows)

    fig, axes = plt.subplots(n_rows, 2, figsize=(6.9, 1.6 * n_rows + 0.85), squeeze=False)
    title_fs = 8.2
    cell_title_fs = 7.0
    body_fs = 6.15
    row_meta_fs = 6.2

    axes[0, 0].set_title("Source snippet ($x_{src}$)", fontsize=title_fs, pad=3.0)
    axes[0, 1].set_title("DirBucket snippet ($x_{db}$)", fontsize=title_fs, pad=3.0)

    for r, row in enumerate(picked_rows):
        did = str(row["did"])
        src_txt = str(row["src_txt"]).strip()
        wm_txt = str(row["wm_txt"]).strip()
        sim = row.get("sim")
        ov = float(row.get("ov", 0.0))
        len_ratio = float(row.get("len_ratio", 1.0))
        title = str(row.get("title") or "").strip()
        title_short = (title[:58] + "...") if len(title) > 61 else title
        meta = f"Doc {did}"
        if isinstance(sim, (int, float)) and np.isfinite(float(sim)):
            meta += f" | sim={float(sim):.2f}"
        meta += f" | overlap={ov:.2f}"
        meta += f" | len-ratio={len_ratio:.2f}"
        if title_short:
            meta += f" | {title_short}"

        for c in range(2):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_facecolor("#f7f7f7")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#c9c9c9")

        axes[r, 0].text(0.02, 0.92, textwrap.fill(src_txt, width=58), ha="left", va="top", fontsize=body_fs)
        axes[r, 1].text(0.02, 0.92, textwrap.fill(wm_txt, width=58), ha="left", va="top", fontsize=body_fs)
        axes[r, 0].text(0.00, -0.10, meta, transform=axes[r, 0].transAxes, ha="left", va="top", fontsize=row_meta_fs, color="#444444")

    _save_fig(fig, figpath, tight_rect=(0.0, 0.01, 1.0, 1.0))
    print(f"[OK] {figpath.with_suffix('.png')}")
    print(f"[OK] {figpath.with_suffix('.pdf')}")


def plot_rag_target_vs_nontarget_summary(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Main RAG mixing summary plot for the common current setup where rag_multi is run
    for a single target provider (typically provider 0) across seeds.

    Top: max non-target z bar + target z marker/connector (gated z).
    Bottom: BH rates (target detected vs any non-target activated).
    """
    figpath = out_dir / "fig_watermark_compare_rag_target_vs_nontarget_summary"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }

    recs: Dict[str, List[Dict[str, Any]]] = {s: [] for s in schemes}
    targets_seen: set[int] = set()
    for s in schemes:
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            st = _target_non_target_stats(r)
            st["seed"] = fp.stem
            st["target_provider"] = int(r.get("target_provider", 0))
            recs[s].append(st)
            targets_seen.add(int(r.get("target_provider", 0)))

    if not any(recs.values()):
        print("[WARN] No rag_multi files found for RAG target-vs-nontarget summary plot.")
        return

    if len(targets_seen) > 1:
        print(
            "[WARN] rag_multi contains multiple target providers. "
            "Current summary plot assumes one target provider across seeds and will aggregate together."
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.1]},
    )
    ax_top, ax_bot = axes
    x = np.arange(len(schemes))
    width = 0.68

    # Top panel: base=max non-target z, then a connector/marker to target z.
    for i, s in enumerate(schemes):
        vals = recs[s]
        if not vals:
            continue
        max_non = [float(v.get("max_non_z", np.nan)) for v in vals]
        target = [float(v.get("target_z", np.nan)) for v in vals]
        m_non, hw_non = mean_ci([v for v in max_non if not np.isnan(v)])
        m_tar, hw_tar = mean_ci([v for v in target if not np.isnan(v)])
        c = colors.get(s)

        # Base bar = max non-target z (gated).
        ax_top.bar(
            x[i],
            m_non,
            width=width,
            color=c,
            alpha=0.25,
            edgecolor="black",
            linewidth=0.7,
            yerr=(None if np.isnan(hw_non) else hw_non),
            capsize=4,
        )
        # If target is above non-target, show a continuation cap.
        if not (np.isnan(m_tar) or np.isnan(m_non)) and m_tar > m_non:
            ax_top.bar(
                x[i],
                m_tar - m_non,
                bottom=m_non,
                width=width,
                color=c,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.7,
            )
        # Connector to actual target mean (handles negative or below-base values).
        if not (np.isnan(m_tar) or np.isnan(m_non)):
            ax_top.plot([x[i], x[i]], [m_non, m_tar], color=c, linewidth=2.0, zorder=3)
            ax_top.scatter(
                [x[i]],
                [m_tar],
                s=70,
                marker="o",
                facecolor=c,
                edgecolor="black",
                linewidth=0.8,
                zorder=4,
                label=(_scheme_label(s) if i == 0 else None),
            )
            # Target CI as errorbar on marker.
            if not np.isnan(hw_tar):
                ax_top.errorbar(
                    [x[i]],
                    [m_tar],
                    yerr=[hw_tar],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=4,
                    zorder=5,
                )
        # Per-seed faint dots for transparency.
        jitter = np.linspace(-0.10, 0.10, num=max(1, len(vals)))
        for j, v in enumerate(vals):
            xx = x[i] + (jitter[j] if j < len(jitter) else 0.0)
            ax_top.scatter(xx, float(v.get("max_non_z", np.nan)), s=18, color=c, alpha=0.35, marker="s")
            ax_top.scatter(xx, float(v.get("target_z", np.nan)), s=18, color=c, alpha=0.55, marker="o")

    ax_top.axhline(4.0, linestyle="--", linewidth=1.1, color="black")
    ax_top.axhline(0.0, linewidth=0.9, color="black")
    ax_top.set_ylabel("pooled z (gated)")
    t_lab = (
        f"target provider={sorted(targets_seen)[0]}"
        if len(targets_seen) == 1
        else "mixed targets"
    )
    ax_top.set_title(f"RAG Mixing Summary ({t_lab}): target vs max non-target")

    # Small inline legend explaining marks.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    scheme_handles = [
        Patch(facecolor=colors.get(s), edgecolor="black", alpha=0.85, label=_scheme_label(s))
        for s in schemes
        if recs[s]
    ]
    extra_handles = [
        Patch(facecolor="black", edgecolor="black", alpha=0.25, label="bar base = max non-target z"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", linestyle="-", label="target z mean"),
    ]
    ax_top.legend(handles=scheme_handles + extra_handles, fontsize=8, loc="best")

    # Bottom panel: BH rates.
    bar_w = 0.34
    for i, s in enumerate(schemes):
        vals = recs[s]
        if not vals:
            continue
        target_bh_rate = float(np.mean([1.0 if bool(v.get("target_bh")) else 0.0 for v in vals]))
        non_bh_rate = float(np.mean([1.0 if bool(v.get("any_non_bh")) else 0.0 for v in vals]))
        c = colors.get(s)
        ax_bot.bar(
            x[i] - bar_w / 2.0,
            target_bh_rate,
            width=bar_w,
            color=c,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
        )
        ax_bot.bar(
            x[i] + bar_w / 2.0,
            non_bh_rate,
            width=bar_w,
            color=c,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.35,
        )
        ax_bot.text(x[i] - bar_w / 2.0, target_bh_rate + 0.03, f"{target_bh_rate:.2f}", ha="center", va="bottom", fontsize=8)
        ax_bot.text(x[i] + bar_w / 2.0, non_bh_rate + 0.03, f"{non_bh_rate:.2f}", ha="center", va="bottom", fontsize=8)

    ax_bot.set_ylim(0.0, 1.05)
    ax_bot.set_ylabel("BH rate")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([_scheme_label(s) for s in schemes], rotation=10, ha="right")
    ax_bot.set_xlabel("scheme")
    ax_bot.legend(
        handles=[
            Patch(facecolor="black", edgecolor="black", alpha=0.95, label="target BH significant"),
            Patch(facecolor="black", edgecolor="black", alpha=0.35, label="any non-target BH significant"),
        ],
        fontsize=8,
        loc="upper right",
    )

    _save_fig(fig, figpath)


def plot_style_false_positive_rates(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_style_false_positives"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    styles = ["qa", "ownwords", "bullets", "eli5"]
    fp_by: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}

    for s in schemes:
        for fp in find_files(results_root, s, "style_sweep_seed*.json"):
            obj = load_json(fp)
            for run in obj.get("runs", []):
                st = str(run.get("answer_style") or "")
                if st not in styles:
                    continue
                bh = run.get("bh_sig") or []
                tp = int(run.get("target_provider", 0))
                if not bh or tp < 0 or tp >= len(bh):
                    continue
                non = [i for i in range(len(bh)) if i != tp]
                fp_by[s][st].append(1.0 if any(bool(bh[i]) for i in non) else 0.0)

    x = np.arange(len(styles))
    n_s = max(1, len(schemes))
    width = 0.82 / n_s

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, s in enumerate(schemes):
        means: List[float] = []
        errs: List[float] = []
        for st in styles:
            m, hw = mean_ci(fp_by[s][st])
            means.append(m)
            errs.append(hw)
        offs = (i - (n_s - 1) / 2.0) * width
        ax.bar(
            x + offs,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            edgecolor="black",
            linewidth=0.6,
            color=colors.get(s),
            label=_scheme_label(s),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(styles)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("non-target BH activation rate")
    ax.set_title("Style Robustness Specificity (Lower is Better)")
    ax.legend()

    _save_fig(fig, figpath)


def plot_style_z_anatomy_by_level(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    """
    Style-level plot with x=paraphrase level, showing:
      - target z (gated) with target-BH fill status
      - max non-target z (gated) with any-non-target-BH fill status
    This is intended as a quick visual for "power survives / specificity stays clean".
    """
    figpath = out_dir / "fig_watermark_compare_style_z_anatomy"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    markers = {
        "dirbucket_naive": "o",
        "kgw_ward": "s",
        "semstamp": "^",
        "sim_attr": "D",
        "overlap_attr": "P",
        "entail_attr": "X",
    }
    styles = ["qa", "ownwords", "bullets", "eli5"]
    x = np.arange(len(styles))

    # rows[scheme][style] -> list of stats dicts
    rows_by: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        s: {st: [] for st in styles} for s in schemes
    }
    for s in schemes:
        for fp in find_files(results_root, s, "style_sweep_seed*.json"):
            obj = load_json(fp)
            for run in obj.get("runs", []):
                st_name = str(run.get("answer_style") or "")
                if st_name not in styles:
                    continue
                st = _target_non_target_stats(run)
                st["seed"] = fp.stem
                rows_by[s][st_name].append(st)

    have_any = any(rows_by[s][st] for s in schemes for st in styles)
    if not have_any:
        print("[WARN] No style_sweep files found for style z anatomy plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True, gridspec_kw={"height_ratios": [2, 2]})
    ax_t, ax_n = axes
    n_s = max(1, len(schemes))
    offsets = {
        s: (i - (n_s - 1) / 2.0) * (0.20 if n_s >= 3 else 0.24)
        for i, s in enumerate(schemes)
    }

    # Plot per-seed points + scheme means.
    for s in schemes:
        c = colors.get(s)
        mkr = markers.get(s, "o")
        mean_target: List[float] = []
        mean_non: List[float] = []
        for si, st_name in enumerate(styles):
            vals = rows_by[s][st_name]
            tvals = [float(v.get("target_z", np.nan)) for v in vals]
            nvals = [float(v.get("max_non_z", np.nan)) for v in vals]
            mt, _ = mean_ci([v for v in tvals if not np.isnan(v)])
            mn, _ = mean_ci([v for v in nvals if not np.isnan(v)])
            mean_target.append(mt)
            mean_non.append(mn)

            if vals:
                jitter = np.linspace(-0.05, 0.05, num=max(1, len(vals)))
                for j, v in enumerate(vals):
                    xx = x[si] + offsets[s] + (jitter[j] if j < len(jitter) else 0.0)
                    t_bh = bool(v.get("target_bh"))
                    n_bh = bool(v.get("any_non_bh"))
                    ax_t.scatter(
                        xx,
                        float(v.get("target_z", np.nan)),
                        s=58,
                        marker=mkr,
                        edgecolor=c,
                        facecolor=(c if t_bh else "none"),
                        linewidth=1.3,
                        alpha=0.95,
                    )
                    ax_n.scatter(
                        xx,
                        float(v.get("max_non_z", np.nan)),
                        s=58,
                        marker=mkr,
                        edgecolor=c,
                        facecolor=(c if n_bh else "none"),
                        linewidth=1.3,
                        alpha=0.95,
                    )

        xs = x + offsets[s]
        ax_t.plot(xs, np.array(mean_target, dtype=float), color=c, linewidth=1.8, alpha=0.9, label=_scheme_label(s))
        ax_n.plot(xs, np.array(mean_non, dtype=float), color=c, linewidth=1.8, alpha=0.9, label=_scheme_label(s))

    for ax in (ax_t, ax_n):
        ax.axhline(4.0, linestyle="--", linewidth=1.1, color="black")
        ax.axhline(0.0, linewidth=0.9, color="black")

    ax_t.set_ylabel("target z (gated)")
    ax_t.set_title("Style Robustness Anatomy: target power and non-target specificity by paraphrase level")
    ax_t.legend(loc="best")
    ax_t.text(
        0.99,
        0.03,
        "filled marker = target BH significant",
        transform=ax_t.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax_n.set_ylabel("max non-target z (gated)")
    ax_n.set_xticks(x)
    ax_n.set_xticklabels(styles)
    ax_n.set_xlabel("paraphrase level / answer style")
    ax_n.text(
        0.99,
        0.03,
        "filled marker = any non-target BH significant",
        transform=ax_n.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    _save_fig(fig, figpath)


def plot_style_sweep(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_style_sweep"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    styles = ["qa", "ownwords", "bullets", "eli5"]

    z_by: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}
    fp_by: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}

    for s in schemes:
        for fp in find_files(results_root, s, "style_sweep_seed*.json"):
            obj = load_json(fp)
            for r in obj.get("runs", []):
                st = str(r.get("answer_style"))
                if st not in styles:
                    continue
                tp = int(r.get("target_provider", 0))
                z_by[s][st].append(float(r["z"][tp]))
                bh = r.get("bh_sig") or []
                if bh and 0 <= tp < len(bh):
                    non = [i for i in range(len(bh)) if i != tp]
                    fp_by[s][st].append(1.0 if any(bool(bh[i]) for i in non) else 0.0)

    x = np.arange(len(styles))
    n_s = max(1, len(schemes))
    width = 0.82 / n_s

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for i, s in enumerate(schemes):
        means = []
        errs = []
        for st in styles:
            m, hw = mean_ci(z_by[s][st])
            means.append(m)
            errs.append(hw)
        offs = (i - (n_s - 1) / 2.0) * width
        axes[0].bar(
            x + offs,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            edgecolor="black",
            linewidth=0.6,
            color=colors.get(s),
            label=_scheme_label(s),
        )

    axes[0].axhline(4.0, linestyle="--", linewidth=1.2, color="black")
    axes[0].set_ylabel("pooled z (target, gated)")
    axes[0].set_title("Style Robustness: DirBucket vs KGW/WARD (min_T-gated z + BH specificity)")
    axes[0].legend()

    for i, s in enumerate(schemes):
        means = []
        errs = []
        for st in styles:
            m, hw = mean_ci(fp_by[s][st])
            means.append(m)
            errs.append(hw)
        offs = (i - (n_s - 1) / 2.0) * width
        axes[1].bar(
            x + offs,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            edgecolor="black",
            linewidth=0.6,
            color=colors.get(s),
            label=_scheme_label(s),
        )

    axes[1].set_ylabel("non-target BH activation rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(styles)
    axes[1].set_ylim(0.0, 1.0)

    _save_fig(fig, figpath)


def plot_style_specificity(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_style_specificity"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    styles = ["qa", "ownwords", "bullets", "eli5"]

    z_target: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}
    z_max_non: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}
    z_margin: Dict[str, Dict[str, List[float]]] = {s: {st: [] for st in styles} for s in schemes}

    for s in schemes:
        for fp in find_files(results_root, s, "style_sweep_seed*.json"):
            obj = load_json(fp)
            for r in obj.get("runs", []):
                st = str(r.get("answer_style"))
                if st not in styles:
                    continue
                tp = int(r.get("target_provider", 0))
                # Use min_pooled_T-gated z for main specificity plots; raw-z spikes are shown in dedicated diagnostics.
                z = r.get("z") or r.get("z_raw") or []
                if not z or tp < 0 or tp >= len(z):
                    continue
                non = [i for i in range(len(z)) if i != tp]
                if not non:
                    continue
                zt = float(z[tp])
                zn = float(max(float(z[i]) for i in non))
                z_target[s][st].append(zt)
                z_max_non[s][st].append(zn)
                z_margin[s][st].append(zt - zn)

    x = np.arange(len(styles))
    n_s = max(1, len(schemes))
    width = 0.82 / n_s
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.0), sharex=True)

    def _plot_metric(ax: plt.Axes, by_scheme: Dict[str, Dict[str, List[float]]], ylabel: str) -> None:
        for i, s in enumerate(schemes):
            means = []
            errs = []
            for st in styles:
                m, hw = mean_ci(by_scheme[s][st])
                means.append(m)
                errs.append(hw)
            offs = (i - (n_s - 1) / 2.0) * width
            ax.bar(
                x + offs,
                means,
                width=width,
                yerr=errs,
                capsize=3,
                edgecolor="black",
                linewidth=0.6,
                color=colors.get(s),
                label=_scheme_label(s),
            )
        ax.set_ylabel(ylabel)

    _plot_metric(axes[0], z_target, "target z (gated)")
    axes[0].axhline(4.0, linestyle="--", linewidth=1.2, color="black")
    axes[0].set_title("Style Specificity (min_T-gated): Target vs Max Non-Target vs Margin")
    axes[0].legend()

    _plot_metric(axes[1], z_max_non, "max non-target z (gated)")
    axes[1].axhline(4.0, linestyle="--", linewidth=1.2, color="black")

    _plot_metric(axes[2], z_margin, "margin (target - max non)")
    axes[2].axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(styles)
    axes[2].set_xlabel("answer style")

    _save_fig(fig, figpath)


def plot_non_target_activation_rates(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_non_target_activation_rates"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    settings = ["sanity_unrel", "rag_multi", "style_qa", "style_ownwords", "style_bullets", "style_eli5"]
    labels = ["sanity-unrel", "rag-multi", "style-qa", "style-own", "style-bul", "style-eli5"]

    rate_by_scheme: Dict[str, Dict[str, float]] = {s: {k: float("nan") for k in settings} for s in schemes}

    for s in schemes:
        # sanity_unrel: any BH positive at all (target absent; non-target activation).
        vals: List[float] = []
        for fp in find_files(results_root, s, "sanity_unrel_t*_seed*.json"):
            r = load_json(fp)
            bh = r.get("bh_sig") or []
            vals.append(1.0 if any(bool(x) for x in bh) else 0.0)
        if vals:
            rate_by_scheme[s]["sanity_unrel"] = float(np.mean(vals))

        # rag_multi: any non-target BH positive
        vals = []
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            bh = r.get("bh_sig") or []
            tp = int(r.get("target_provider", 0))
            non = [i for i in range(len(bh)) if i != tp]
            vals.append(1.0 if any(bool(bh[i]) for i in non) else 0.0)
        if vals:
            rate_by_scheme[s]["rag_multi"] = float(np.mean(vals))

        # style_*: any non-target BH positive per seed/style run
        style_vals: Dict[str, List[float]] = {k: [] for k in settings if k.startswith("style_")}
        for fp in find_files(results_root, s, "style_sweep_seed*.json"):
            obj = load_json(fp)
            for run in obj.get("runs", []):
                st = str(run.get("answer_style"))
                key = f"style_{st}"
                if key not in style_vals:
                    continue
                bh = run.get("bh_sig") or []
                tp = int(run.get("target_provider", 0))
                non = [i for i in range(len(bh)) if i != tp]
                style_vals[key].append(1.0 if any(bool(bh[i]) for i in non) else 0.0)
        for key, vals in style_vals.items():
            if vals:
                rate_by_scheme[s][key] = float(np.mean(vals))

    x = np.arange(len(settings))
    n_s = max(1, len(schemes))
    width = 0.82 / n_s

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, s in enumerate(schemes):
        offs = (i - (n_s - 1) / 2.0) * width
        ys = [rate_by_scheme[s][k] for k in settings]
        ax.bar(
            x + offs,
            ys,
            width=width,
            edgecolor="black",
            linewidth=0.6,
            color=colors.get(s),
            label=_scheme_label(s),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("non-target activation rate (any non-target BH)")
    ax.set_title("Non-Target Activation by Setting (Lower is Better for attribution)")
    ax.legend()

    _save_fig(fig, figpath)


def plot_decision_scorecard_heatmap(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_decision_scorecard"

    row_keys = [
        "sanity_coop_target_bh",
        "sanity_coop_specificity",
        "sanity_unrel_specificity",
        "rag_target_bh",
        "rag_specificity",
        "style_target_bh",
        "style_specificity",
    ]
    row_labels = [
        "Sanity coop: target BH rate",
        "Sanity coop: no non-target BH",
        "Sanity unrelated: no BH",
        "RAG multi: target BH rate",
        "RAG multi: no non-target BH",
        "Style sweep: target BH rate",
        "Style sweep: no non-target BH",
    ]

    vals_by_scheme: Dict[str, Dict[str, float]] = {s: {k: float("nan") for k in row_keys} for s in schemes}

    for s in schemes:
        # sanity_coop
        coop_target: List[float] = []
        coop_spec: List[float] = []
        for fp in find_files(results_root, s, "sanity_coop_t*_seed*.json"):
            r = load_json(fp)
            st = _target_non_target_stats(r)
            coop_target.append(1.0 if bool(st["target_bh"]) else 0.0)
            coop_spec.append(0.0 if bool(st["any_non_bh"]) else 1.0)
        if coop_target:
            vals_by_scheme[s]["sanity_coop_target_bh"] = float(np.mean(coop_target))
        if coop_spec:
            vals_by_scheme[s]["sanity_coop_specificity"] = float(np.mean(coop_spec))

        # sanity_unrel
        su_spec: List[float] = []
        for fp in find_files(results_root, s, "sanity_unrel_t*_seed*.json"):
            r = load_json(fp)
            bh = r.get("bh_sig") or []
            su_spec.append(0.0 if any(bool(x) for x in bh) else 1.0)
        if su_spec:
            vals_by_scheme[s]["sanity_unrel_specificity"] = float(np.mean(su_spec))

        # rag_multi
        rag_t: List[float] = []
        rag_spec: List[float] = []
        for fp in find_rag_multi_files(results_root, s):
            r = load_json(fp)
            st = _target_non_target_stats(r)
            rag_t.append(1.0 if bool(st["target_bh"]) else 0.0)
            rag_spec.append(0.0 if bool(st["any_non_bh"]) else 1.0)
        if rag_t:
            vals_by_scheme[s]["rag_target_bh"] = float(np.mean(rag_t))
        if rag_spec:
            vals_by_scheme[s]["rag_specificity"] = float(np.mean(rag_spec))

        # style sweep (aggregate all style runs)
        style_t: List[float] = []
        style_spec: List[float] = []
        for fp in find_files(results_root, s, "style_sweep_seed*.json"):
            obj = load_json(fp)
            for r in obj.get("runs", []):
                st = _target_non_target_stats(r)
                style_t.append(1.0 if bool(st["target_bh"]) else 0.0)
                style_spec.append(0.0 if bool(st["any_non_bh"]) else 1.0)
        if style_t:
            vals_by_scheme[s]["style_target_bh"] = float(np.mean(style_t))
        if style_spec:
            vals_by_scheme[s]["style_specificity"] = float(np.mean(style_spec))

    mat = np.array([[vals_by_scheme[s][k] for s in schemes] for k in row_keys], dtype=float)
    if np.isnan(mat).all():
        print("[WARN] No data for decision scorecard heatmap.")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(schemes)))
    ax.set_xticklabels([_scheme_label(s) for s in schemes], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Decision Scorecard (Higher is Better)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "nan" if np.isnan(v) else f"{v:.2f}"
            color = "black" if (np.isnan(v) or v < 0.75) else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("rate")

    _save_fig(fig, figpath)


def plot_sanity_provider_bars(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_sanity_pooled_z"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }

    P = 0
    for s in schemes:
        for fp in find_files(results_root, s, "sanity_coop_t*_seed*.json"):
            r = load_json(fp)
            P = max(P, len(r.get("z") or []))
    if P <= 0:
        print("[WARN] No sanity files found.")
        return

    coop_true: Dict[str, Dict[int, List[float]]] = {s: {p: [] for p in range(P)} for s in schemes}
    coop_wrong: Dict[str, Dict[int, List[float]]] = {s: {p: [] for p in range(P)} for s in schemes}
    unrel_notarget: Dict[str, Dict[int, List[float]]] = {s: {p: [] for p in range(P)} for s in schemes}

    for s in schemes:
        for fp in find_files(results_root, s, "sanity_coop_t*_seed*.json"):
            m = re.search(r"sanity_coop_t(\d+)_seed", fp.name)
            if not m:
                continue
            t = int(m.group(1))
            r = load_json(fp)
            z = r.get("z") or []
            if t < len(z):
                coop_true[s][t].append(float(z[t]))
            wz = (r.get("wrongkey", {}) or {}).get("z")
            if wz is not None and t < len(wz):
                coop_wrong[s][t].append(float(wz[t]))

    for s in schemes:
        for fp in find_files(results_root, s, "sanity_unrel_t*_seed*.json"):
            m = re.search(r"sanity_unrel_t(\d+)_seed", fp.name)
            if not m:
                continue
            t = int(m.group(1))
            r = load_json(fp)
            z = r.get("z") or []
            for p in range(min(len(z), P)):
                if p == t:
                    continue
                unrel_notarget[s][p].append(float(z[p]))

    x = np.arange(P)
    n_s = max(1, len(schemes))
    width = 0.82 / n_s

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for i, s in enumerate(schemes):
        means = []
        errs = []
        for p in range(P):
            m, hw = mean_ci(coop_true[s][p])
            means.append(m)
            errs.append(hw)
        offs = (i - (n_s - 1) / 2.0) * width
        axes[0].bar(
            x + offs,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            edgecolor="black",
            linewidth=0.6,
            color=colors.get(s),
            label=_scheme_label(s),
        )

    axes[0].axhline(4.0, linestyle="--", linewidth=1.2, color="black")
    axes[0].set_ylabel("pooled z (coop, correct key)")
    axes[0].set_title("Sanity Pooled z by Provider: DirBucket vs KGW/WARD")
    axes[0].legend()

    subw = width / 2.0
    for i, s in enumerate(schemes):
        offs = (i - (n_s - 1) / 2.0) * width
        means_w: List[float] = []
        errs_w: List[float] = []
        means_u: List[float] = []
        errs_u: List[float] = []
        for p in range(P):
            m, hw = mean_ci(coop_wrong[s][p])
            means_w.append(m)
            errs_w.append(hw)
            m, hw = mean_ci(unrel_notarget[s][p])
            means_u.append(m)
            errs_u.append(hw)

        c = colors.get(s)
        axes[1].bar(
            x + offs - subw / 2.0,
            means_w,
            width=subw,
            yerr=errs_w,
            capsize=3,
            edgecolor="black",
            linewidth=0.6,
            color=c,
            alpha=0.9,
            label=f"{_scheme_label(s)} wrong-key",
        )
        axes[1].bar(
            x + offs + subw / 2.0,
            means_u,
            width=subw,
            yerr=errs_u,
            capsize=3,
            edgecolor="black",
            linewidth=0.6,
            color=c,
            alpha=0.45,
            label=f"{_scheme_label(s)} unrelated",
        )

    axes[1].axhline(0.0, linewidth=1.0, color="black")
    axes[1].set_ylabel("pooled z (null)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(p) for p in range(P)])
    axes[1].set_xlabel("provider id")
    axes[1].set_ylim(-4.0, 4.0)

    _save_fig(fig, figpath)


def plot_aligned_source_scope(results_root: Path, out_dir: Path, schemes: List[str]) -> None:
    figpath = out_dir / "fig_watermark_compare_aligned_source_scope"
    colors = {
        "dirbucket_naive": "#1f77b4",
        "kgw_ward": "#ff7f0e",
        "semstamp": "#2ca02c",
        "sim_attr": "#d62728",
        "overlap_attr": "#9467bd",
        "entail_attr": "#8c564b",
    }
    settings = [
        "sanity_coop",
        "sanity_unrel",
        "rag_multi",
        "style_qa",
        "style_ownwords",
        "style_bullets",
        "style_eli5",
    ]
    labels = ["san-coop", "san-unrel", "rag", "st-qa", "st-own", "st-bul", "st-eli5"]

    docs_by: Dict[str, Dict[str, List[float]]] = {s: {k: [] for k in settings} for s in schemes}
    prov_by: Dict[str, Dict[str, List[float]]] = {s: {k: [] for k in settings} for s in schemes}
    assign_by: Dict[str, Dict[str, List[float]]] = {s: {k: [] for k in settings} for s in schemes}

    def _collect(s: str, key: str, obj: Dict[str, Any]) -> None:
        al = (obj.get("align") or {}) if isinstance(obj, dict) else {}
        d = al.get("source_docs_selected_mean")
        p = al.get("source_providers_selected_mean")
        a = al.get("assign_rate")
        if isinstance(d, (int, float)):
            docs_by[s][key].append(float(d))
        if isinstance(p, (int, float)):
            prov_by[s][key].append(float(p))
        if isinstance(a, (int, float)):
            assign_by[s][key].append(float(a))

    for s in schemes:
        for fp in find_files(results_root, s, "sanity_coop_t*_seed*.json"):
            _collect(s, "sanity_coop", load_json(fp))
        for fp in find_files(results_root, s, "sanity_unrel_t*_seed*.json"):
            _collect(s, "sanity_unrel", load_json(fp))
        for fp in find_rag_multi_files(results_root, s):
            _collect(s, "rag_multi", load_json(fp))
        for fp in find_files(results_root, s, "style_sweep_seed*.json"):
            obj = load_json(fp)
            for run in obj.get("runs", []):
                st = str(run.get("answer_style") or "")
                key = f"style_{st}"
                if key in docs_by[s]:
                    _collect(s, key, run)

    have_any = False
    for s in schemes:
        for k in settings:
            if docs_by[s][k] or prov_by[s][k] or assign_by[s][k]:
                have_any = True
                break
        if have_any:
            break
    if not have_any:
        print("[WARN] No aligned-source scope stats found.")
        return

    x = np.arange(len(settings))
    n_s = max(1, len(schemes))
    width = 0.82 / n_s
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True)

    def _plot_metric(ax: plt.Axes, by_scheme: Dict[str, Dict[str, List[float]]], ylabel: str) -> None:
        for i, s in enumerate(schemes):
            means: List[float] = []
            errs: List[float] = []
            for k in settings:
                m, hw = mean_ci(by_scheme[s][k])
                means.append(m)
                errs.append(hw)
            offs = (i - (n_s - 1) / 2.0) * width
            ax.bar(
                x + offs,
                means,
                width=width,
                yerr=errs,
                capsize=3,
                edgecolor="black",
                linewidth=0.6,
                color=colors.get(s),
                label=_scheme_label(s),
            )
        ax.set_ylabel(ylabel)

    _plot_metric(axes[0], docs_by, "source docs (mean)")
    axes[0].set_title("Aligned Source Scope Diagnostics")
    axes[0].legend()

    _plot_metric(axes[1], prov_by, "source providers (mean)")

    _plot_metric(axes[2], assign_by, "assign_rate")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].set_xlabel("setting")

    _save_fig(fig, figpath)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate and plot watermark comparison results.")
    ap.add_argument("--results_root", type=Path, default=Path("results/watermark_compare"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/figs/watermark_compare"))
    ap.add_argument(
        "--schemes",
        nargs="+",
        default=["dirbucket_naive", "kgw_ward", "semstamp", "sim_attr", "overlap_attr", "entail_attr"],
    )
    ap.add_argument("--work_dir_dirbucket", type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument("--work_dir_kgw", type=Path, default=Path("work/farad_kgw_ward"))
    ap.add_argument("--work_dir_semstamp", type=Path, default=Path("work/farad_semstamp"))
    ap.add_argument("--work_dir_simattr", type=Path, default=None)
    ap.add_argument("--work_dir_overlap", type=Path, default=None)
    ap.add_argument("--work_dir_entail", type=Path, default=None)
    ap.add_argument("--mixing_stress_root", type=Path, default=None)
    ap.add_argument("--manifest_src", type=Path, default=Path("work/farad_shared_manifest_p6_d400_seed13.jsonl"))
    ap.add_argument("--utility_retrieval_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--utility_topk", type=int, default=5)
    ap.add_argument("--utility_max_queries", type=int, default=0)
    ap.add_argument("--utility_query_seed", type=int, default=13)
    args = ap.parse_args()

    schemes = [s for s in args.schemes if (args.results_root / s).exists()]
    if not schemes:
        print(f"[WARN] No scheme result directories found under {args.results_root}")
        return

    work_dirs: Dict[str, Path] = {
        "dirbucket_naive": args.work_dir_dirbucket,
        "kgw_ward": args.work_dir_kgw,
        "semstamp": args.work_dir_semstamp,
        "sim_attr": (args.work_dir_simattr if args.work_dir_simattr is not None else args.work_dir_dirbucket),
        "overlap_attr": (args.work_dir_overlap if args.work_dir_overlap is not None else args.work_dir_dirbucket),
        "entail_attr": (args.work_dir_entail if args.work_dir_entail is not None else args.work_dir_dirbucket),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmark_task_difficulty(args.results_root, args.out_dir, schemes)
    plot_baseline_screen_overview(args.results_root, args.out_dir, schemes)
    plot_null_calibration(args.results_root, args.out_dir, schemes)
    plot_nonwm_critical_failure(args.results_root, args.out_dir, schemes)
    plot_nonwm_margin_distributions(args.results_root, args.out_dir, schemes)
    plot_wm_frontier_bh_q2d(args.results_root, args.out_dir, schemes)
    plot_wm_frontier_z_q2d(args.results_root, args.out_dir, schemes)
    plot_evidence_accumulation_main(args.results_root, args.out_dir, schemes)
    plot_rag_power(args.results_root, args.out_dir, schemes)
    plot_rag_specificity(args.results_root, args.out_dir, schemes)
    plot_rag_queries_to_crossing(args.results_root, args.out_dir, schemes)
    plot_rag_queries_to_bh(args.results_root, args.out_dir, schemes)
    plot_rag_fp_anatomy_main(args.results_root, args.out_dir, schemes)
    plot_mixing_envelope_main(args.results_root, args.out_dir, schemes)
    plot_style_mixing_robustness_main(
        args.results_root,
        args.out_dir,
        schemes,
        mixing_stress_root=args.mixing_stress_root,
    )
    plot_rag_fp_anatomy(args.results_root, args.out_dir, schemes)
    plot_rag_target_vs_nontarget_summary(args.results_root, args.out_dir, schemes)
    plot_style_robustness_main(args.results_root, args.out_dir, schemes)
    plot_style_sweep(args.results_root, args.out_dir, schemes)
    plot_style_specificity(args.results_root, args.out_dir, schemes)
    plot_style_z_anatomy_by_level(args.results_root, args.out_dir, schemes)
    plot_style_false_positive_rates(args.results_root, args.out_dir, schemes)
    intensity_schemes = [s for s in schemes if s in {"dirbucket_naive", "kgw_ward", "semstamp"}]
    if intensity_schemes:
        plot_style_paraphrase_intensity(args.results_root, args.out_dir, intensity_schemes, work_dirs)
    plot_utility_tradeoff(
        args.results_root,
        args.out_dir,
        schemes,
        work_dirs,
        manifest_src=args.manifest_src,
        retrieval_model_name=args.utility_retrieval_model,
        topk=args.utility_topk,
        max_queries=args.utility_max_queries,
        sample_seed=args.utility_query_seed,
    )
    plot_utility_snippet_examples(
        args.out_dir,
        manifest_src=args.manifest_src,
        work_dir_dirbucket=args.work_dir_dirbucket,
        n_examples=2,
        snippet_sentences=3,
    )
    plot_non_target_activation_rates(args.results_root, args.out_dir, schemes)
    plot_decision_scorecard_heatmap(args.results_root, args.out_dir, schemes)
    plot_sanity_provider_bars(args.results_root, args.out_dir, schemes)
    plot_aligned_source_scope(args.results_root, args.out_dir, schemes)


if __name__ == "__main__":
    main()
