"""Utility diagnostics for DirBucket answers (tab:utility-fidelity): support, coverage, numeric fidelity.

Writes dirbucket_utility_extra_stats.json from the rag_multi trial logs and the source manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def mean_ci(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    m = float(np.nanmean(arr))
    if arr.size <= 1:
        return (m, 0.0)
    s = float(np.nanstd(arr, ddof=1))
    hw = 1.96 * s / math.sqrt(arr.size)
    return (m, hw)


def _resolve_local_model(model_name: str) -> Path:
    if Path(model_name).exists():
        return Path(model_name)

    repo_dir = f"models--sentence-transformers--{model_name}"
    search_roots = [
        Path(".hf/hub"),
        Path(".hf/transformers"),
        Path.home() / ".cache/huggingface/hub",
        Path.home() / ".cache/huggingface/transformers",
    ]
    for root in search_roots:
        base = root / repo_dir
        ref_main = base / "refs" / "main"
        if not ref_main.exists():
            continue
        snap = ref_main.read_text(encoding="utf-8").strip()
        snap_path = base / "snapshots" / snap
        if snap_path.exists():
            return snap_path
    raise FileNotFoundError(
        f"Could not resolve a local cache entry for sentence-transformers/{model_name}. "
        "Pass --model_path explicitly if the snapshot lives elsewhere."
    )


def _load_manifest_source_map(manifest_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            did = str(row.get("id") or "").strip()
            txt = str(row.get("text_src") or row.get("text") or "").strip()
            if did and txt:
                out[did] = txt
    return out


def _load_watermarked_map(corpus_path: Path) -> Tuple[Dict[str, str], Dict[str, float]]:
    text_by_id: Dict[str, str] = {}
    changed_frac_by_id: Dict[str, float] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            did = str(row.get("id") or "").strip()
            txt = str(row.get("text_wm") or row.get("text") or "").strip()
            if not did or not txt:
                continue
            text_by_id[did] = txt
            cf = row.get("wm_changed_frac")
            if isinstance(cf, (int, float)):
                changed_frac_by_id[did] = float(cf)
    return text_by_id, changed_frac_by_id


def _load_shared_answers(results_root: Path, valid_doc_ids: set[str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for fp in sorted((results_root / "dirbucket_naive").glob("rag_multi*.trials.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                did = str(row.get("target_doc_id") or "").strip()
                ans = str(row.get("answer") or "").strip()
                if did not in valid_doc_ids or not ans:
                    continue
                q = str(row.get("question") or "").strip()
                qh = str(row.get("question_hash") or "").strip()
                key = (qh if qh else q, did)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((did, ans))
    return rows


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMERIC_TOKEN = re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?\b")
_WORD_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(str(text or ""))
    out: List[str] = []
    for part in parts:
        part = part.strip()
        if part and re.search(r"[A-Za-z0-9]", part):
            out.append(part)
    return out


def _numeric_token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _NUMERIC_TOKEN.finditer(text or "")}


def _word_count(text: str) -> int:
    return len(_WORD_TOKEN.findall((text or "").lower()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute extra DirBucket utility diagnostics from cached artifacts.")
    ap.add_argument("--results_root", type=Path, required=True)
    ap.add_argument("--manifest_src", type=Path, required=True)
    ap.add_argument("--work_dir_dirbucket", type=Path, required=True)
    ap.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--model_path", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_path = args.model_path if args.model_path is not None else _resolve_local_model(args.model_name)

    src_map = _load_manifest_source_map(args.manifest_src)
    wm_map, changed_frac_by_id = _load_watermarked_map(args.work_dir_dirbucket / "corpus.jsonl")
    common_ids = sorted(set(src_map.keys()) & set(wm_map.keys()))
    if not common_ids:
        raise RuntimeError("No matched source/watermarked documents found.")

    shared_answers = _load_shared_answers(args.results_root, set(common_ids))
    if not shared_answers:
        raise RuntimeError("No cached shared-answer diagnostics found in rag_multi*.trials.jsonl.")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(f"sentence-transformers import failed: {exc}") from exc

    model = SentenceTransformer(str(model_path))

    all_src_sents: List[str] = []
    all_wm_sents: List[str] = []
    src_offsets: Dict[str, Tuple[int, int]] = {}
    wm_offsets: Dict[str, Tuple[int, int]] = {}

    cur = 0
    for did in common_ids:
        sents = _split_sentences(src_map[did])
        src_offsets[did] = (cur, cur + len(sents))
        all_src_sents.extend(sents)
        cur += len(sents)

    cur = 0
    for did in common_ids:
        sents = _split_sentences(wm_map[did])
        wm_offsets[did] = (cur, cur + len(sents))
        all_wm_sents.extend(sents)
        cur += len(sents)

    src_sent_emb = np.asarray(
        model.encode(all_src_sents, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    wm_sent_emb = np.asarray(
        model.encode(all_wm_sents, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )

    src_to_wm_sent_cov: List[float] = []
    wm_to_src_sent_cov: List[float] = []
    numeric_src_total = 0
    numeric_src_hit = 0
    numeric_wm_total = 0
    numeric_wm_hit = 0
    for did in common_ids:
        a0, a1 = src_offsets[did]
        b0, b1 = wm_offsets[did]
        src_doc = src_sent_emb[a0:a1]
        wm_doc = wm_sent_emb[b0:b1]
        if len(src_doc) and len(wm_doc):
            sim = src_doc @ wm_doc.T
            src_to_wm_sent_cov.extend(sim.max(axis=1).astype(float).tolist())
            wm_to_src_sent_cov.extend(sim.max(axis=0).astype(float).tolist())

        src_nums = _numeric_token_set(src_map[did])
        wm_nums = _numeric_token_set(wm_map[did])
        overlap = src_nums & wm_nums
        numeric_src_total += len(src_nums)
        numeric_src_hit += len(overlap)
        numeric_wm_total += len(wm_nums)
        numeric_wm_hit += len(overlap)

    answer_doc_ids = [did for did, _ in shared_answers]
    answer_texts = [ans for _, ans in shared_answers]
    answer_src_texts = [src_map[did] for did in answer_doc_ids]
    answer_wm_texts = [wm_map[did] for did in answer_doc_ids]
    all_doc_level_texts = answer_texts + answer_src_texts + answer_wm_texts
    all_doc_level_emb = np.asarray(
        model.encode(all_doc_level_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    n_answers = len(shared_answers)
    ans_emb = all_doc_level_emb[:n_answers]
    src_doc_emb = all_doc_level_emb[n_answers : 2 * n_answers]
    wm_doc_emb = all_doc_level_emb[2 * n_answers :]
    ans_to_src_doc = np.sum(ans_emb * src_doc_emb, axis=1).astype(float).tolist()
    ans_to_wm_doc = np.sum(ans_emb * wm_doc_emb, axis=1).astype(float).tolist()

    matched_src_tokens = [_word_count(src_map[did]) for did in common_ids]
    matched_wm_tokens = [_word_count(wm_map[did]) for did in common_ids]
    length_ratios = [
        float(w) / float(max(1, s))
        for s, w in zip(matched_src_tokens, matched_wm_tokens)
    ]
    changed_vals = [
        float(changed_frac_by_id[did])
        for did in common_ids
        if did in changed_frac_by_id
    ]

    ans_to_src_doc_m, ans_to_src_doc_hw = mean_ci(ans_to_src_doc)
    ans_to_wm_doc_m, ans_to_wm_doc_hw = mean_ci(ans_to_wm_doc)
    src_to_wm_sent_cov_m, src_to_wm_sent_cov_hw = mean_ci(src_to_wm_sent_cov)
    wm_to_src_sent_cov_m, wm_to_src_sent_cov_hw = mean_ci(wm_to_src_sent_cov)

    stats = {
        "matched_docs": len(common_ids),
        "shared_answers": len(shared_answers),
        "answer_doc_support_source_mean": ans_to_src_doc_m,
        "answer_doc_support_source_ci95": ans_to_src_doc_hw,
        "answer_doc_support_watermarked_mean": ans_to_wm_doc_m,
        "answer_doc_support_watermarked_ci95": ans_to_wm_doc_hw,
        "source_to_watermarked_sentence_coverage_mean": src_to_wm_sent_cov_m,
        "source_to_watermarked_sentence_coverage_ci95": src_to_wm_sent_cov_hw,
        "watermarked_to_source_sentence_coverage_mean": wm_to_src_sent_cov_m,
        "watermarked_to_source_sentence_coverage_ci95": wm_to_src_sent_cov_hw,
        "source_to_watermarked_numeric_recall": float(numeric_src_hit) / float(max(1, numeric_src_total)),
        "watermarked_to_source_numeric_precision": float(numeric_wm_hit) / float(max(1, numeric_wm_total)),
        "changed_fraction_mean": float(np.mean(changed_vals)) if changed_vals else float("nan"),
        "changed_fraction_ge_0_9_rate": float(np.mean([v >= 0.9 for v in changed_vals])) if changed_vals else float("nan"),
        "matched_source_tokens_total": int(sum(matched_src_tokens)),
        "matched_watermarked_tokens_total": int(sum(matched_wm_tokens)),
        "watermarked_to_source_token_ratio_mean": float(np.mean(length_ratios)) if length_ratios else float("nan"),
        "watermarked_to_source_token_ratio_median": float(np.median(length_ratios)) if length_ratios else float("nan"),
    }

    print(json.dumps(stats, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
