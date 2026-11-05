"""E5: compare an open-weight paraphraser with the API paraphraser (paper Table tab:paraphrase-model).

Both corpora were built on the SAME 50-doc sub-manifest with identical settings
(K=24, rho=3, same encoder/bucketizer/keys); only the paraphrase model differs.
Reports per corpus (``relative_gain`` = gain of the first tag / gain of the second):
  - mean per-doc directional gain (wm_gain_mean, as stored by the builder)
  - mean changed-sentence fraction
  - doc-level semantic similarity source vs. watermarked (mpnet cosine)
  - token ratio (watermarked / source, whitespace tokens)

Usage:
  python experiments/e5_compare_paraphrasers.py \
      --corpora llama8b:work/e5_llama8b,gpt4omini:work/e5_ref_gpt4omini \
      --manifest work/e5_sub50_manifest.jsonl --out results/e5_paraphraser/compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", type=str, required=True,
                    help="Comma-separated tag:work_dir pairs.")
    ap.add_argument("--manifest", type=Path, default=Path("work/e5_sub50_manifest.jsonl"))
    ap.add_argument("--embed_model", type=str,
                    default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--out", type=Path, default=Path("results/e5_paraphraser/compare.json"))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.embed_model)

    src_by_id = {str(json.loads(l)["id"]): str(json.loads(l)["text_src"])
                 for l in args.manifest.open(encoding="utf-8")}

    report = {}
    for spec in args.corpora.split(","):
        tag, wd = spec.split(":", 1)
        rows = [json.loads(l) for l in (Path(wd) / "corpus.jsonl").open(encoding="utf-8")]
        gains = [float(r.get("wm_gain_mean", 0.0)) for r in rows]
        changed = [float(r.get("wm_changed_frac", 0.0)) for r in rows]
        srcs = [src_by_id[str(r["id"])] for r in rows]
        wms = [str(r["text_wm"]) for r in rows]
        E_src = enc.encode(srcs, normalize_embeddings=True, convert_to_numpy=True)
        E_wm = enc.encode(wms, normalize_embeddings=True, convert_to_numpy=True)
        cos = (E_src * E_wm).sum(axis=1)
        tok_ratio = [len(w.split()) / max(1, len(s.split())) for w, s in zip(wms, srcs)]
        report[tag] = {
            "work_dir": wd,
            "n_docs": len(rows),
            "mean_gain": float(np.mean(gains)),
            "median_gain": float(np.median(gains)),
            "mean_changed_frac": float(np.mean(changed)),
            "mean_doc_semsim": float(np.mean(cos)),
            "min_doc_semsim": float(np.min(cos)),
            "mean_token_ratio": float(np.mean(tok_ratio)),
        }
        print(f"[{tag}] n={len(rows)} gain={report[tag]['mean_gain']:.4f} "
              f"changed={report[tag]['mean_changed_frac']:.3f} "
              f"semsim={report[tag]['mean_doc_semsim']:.3f} "
              f"tokratio={report[tag]['mean_token_ratio']:.3f}")

    tags = list(report)
    if len(tags) == 2:
        a, b = tags
        if report[b]["mean_gain"] > 0:
            report["relative_gain"] = {
                f"{a}_vs_{b}": float(report[a]["mean_gain"] / report[b]["mean_gain"])}
            print(f"[rel] {a} achieves "
                  f"{100 * report['relative_gain'][f'{a}_vs_{b}']:.1f}% of {b}'s gain")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] {args.out}")


if __name__ == "__main__":
    main()
