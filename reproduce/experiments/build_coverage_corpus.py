"""Build the partial-coverage corpora for E2 (paper App. app:coverage, Table tab:coverage).

Deployment story being simulated: a provider with wording-sensitive spans
exempts a fraction (1-c) of sentences from watermarking — those sentences stay
VERBATIM from the source document (the same code path as Algorithm 1's
no-candidate fallback). We approximate this by splicing the existing
watermarked corpus with the source manifest at the sentence level:

  - The deployed watermarked doc is a summary-mode rewrite: each of its
    sentences paraphrases one (MMR-selected) source sentence. The selection map
    was not stored, so each watermarked sentence is aligned to its source
    counterpart by nearest-neighbor cosine in embedding space (paraphrases keep
    sem-sim >= 0.7 by construction, so 1-1 greedy NN matching is reliable;
    unalignable sentences below --align_threshold stay watermarked and are
    counted).
  - Per sentence, a keyed hash h(doc_id, idx) in [0,1) decides: h < c -> keep
    the watermarked sentence; else -> substitute the aligned source sentence.
    Same hash across conditions => kept-watermarked sets are NESTED
    (c=25% ⊂ 50% ⊂ 75%), making the curve monotone-interpretable.

No API calls; embedding-only (MiniLM). Emits standard work dirs usable by the
primary audit harness: work/coverage/c{25,50,75}/{corpus.jsonl,provider_keys.json}.

Usage:
  python experiments/build_coverage_corpus.py \
      --src_work_dir work/farad_dirbucket_naive \
      --manifest work/farad_shared_manifest_p6_d400_seed13.jsonl \
      --out_root work/coverage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watermark_dirbucket import split_sentences  # noqa: E402


def keep_prob(doc_id: str, idx: int) -> float:
    h = hashlib.sha256(f"cov:{doc_id}:{idx}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def align_greedy(sim: np.ndarray, threshold: float) -> Dict[int, int]:
    """Greedy 1-1 matching wm-sentence -> src-sentence by descending cosine."""
    pairs = [(float(sim[i, j]), i, j) for i in range(sim.shape[0]) for j in range(sim.shape[1])]
    pairs.sort(reverse=True)
    used_i: set[int] = set()
    used_j: set[int] = set()
    out: Dict[int, int] = {}
    for s, i, j in pairs:
        if s < threshold:
            break
        if i in used_i or j in used_j:
            continue
        out[i] = j
        used_i.add(i)
        used_j.add(j)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_work_dir", type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("work/farad_shared_manifest_p6_d400_seed13.jsonl"))
    ap.add_argument("--out_root", type=Path, default=Path("work/coverage"))
    ap.add_argument("--coverages", type=str, default="0.25,0.50,0.75")
    ap.add_argument("--align_model", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--align_threshold", type=float, default=0.50)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.align_model)

    rows = [json.loads(l) for l in (args.src_work_dir / "corpus.jsonl").open(encoding="utf-8")]
    src_by_id = {str(json.loads(l)["id"]): json.loads(l)["text_src"]
                 for l in args.manifest.open(encoding="utf-8")}
    coverages = sorted(float(x) for x in args.coverages.split(","))

    # ---- Align once; splice per coverage ----
    per_cov_rows: Dict[float, List[dict]] = {c: [] for c in coverages}
    align_cos: List[float] = []
    n_unalignable = 0
    n_sent_total = 0

    for k, r in enumerate(rows, 1):
        doc_id = str(r["id"])
        wm_sents = split_sentences(str(r["text_wm"]))
        src_text = src_by_id.get(doc_id, "")
        src_sents = split_sentences(src_text) if src_text else []
        n_sent_total += len(wm_sents)

        mapping: Dict[int, int] = {}
        if wm_sents and src_sents:
            E_wm = enc.encode(wm_sents, normalize_embeddings=True, convert_to_numpy=True)
            E_src = enc.encode(src_sents, normalize_embeddings=True, convert_to_numpy=True)
            sim = E_wm @ E_src.T
            mapping = align_greedy(sim, args.align_threshold)
            align_cos.extend(float(sim[i, j]) for i, j in mapping.items())
        n_unalignable += sum(1 for i in range(len(wm_sents)) if i not in mapping)

        probs = [keep_prob(doc_id, i) for i in range(len(wm_sents))]
        for c in coverages:
            spliced: List[str] = []
            kept = 0
            for i, ws in enumerate(wm_sents):
                # keep watermarked sentence if hash < c OR no source counterpart
                if probs[i] < c or i not in mapping:
                    spliced.append(ws)
                    kept += 1
                else:
                    spliced.append(src_sents[mapping[i]])
            rec = dict(r)
            rec["text_wm"] = " ".join(spliced)
            rec["wm_cov_frac"] = kept / max(1, len(wm_sents))
            per_cov_rows[c].append(rec)
        if k % 100 == 0:
            print(f"  [align] {k}/{len(rows)} docs")

    print(f"[align] mean cosine of matched pairs: {np.mean(align_cos):.3f} "
          f"(min {np.min(align_cos):.3f}); unalignable {n_unalignable}/{n_sent_total} sentences")

    for c in coverages:
        tag = f"c{int(round(100 * c)):02d}"
        out_dir = args.out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
            for rec in per_cov_rows[c]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        shutil.copy(args.src_work_dir / "provider_keys.json", out_dir / "provider_keys.json")
        realized = float(np.mean([rec["wm_cov_frac"] for rec in per_cov_rows[c]]))
        print(f"[emit] {out_dir}: target coverage {c:.2f}, realized {realized:.3f}")

    print("[DONE]")


if __name__ == "__main__":
    main()
