#!/usr/bin/env python3
"""Build the 50-document E5 sub-manifest (paper Table tab:paraphrase-model).

Rule used for the paper: sort the FARAD shared manifest by
``sha256("e5:" + id)`` and keep the first ``--n_docs`` rows unchanged. The
paper's ``work/e5_sub50_manifest.jsonl`` was produced by this rule.

Usage:
  python experiments/make_e5_manifest.py \
      --manifest work/farad_shared_manifest_p6_d400_seed13.jsonl \
      --out work/e5_sub50_manifest.jsonl --n_docs 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def e5_subset(rows: List[Dict[str, Any]], n_docs: int, salt: str = "e5") -> List[Dict[str, Any]]:
    """First ``n_docs`` rows in sha256(f"{salt}:{id}") order."""
    key = lambda r: hashlib.sha256(f"{salt}:{r['id']}".encode()).hexdigest()  # noqa: E731
    return sorted(rows, key=key)[:n_docs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path,
                    default=Path("work/farad_shared_manifest_p6_d400_seed13.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("work/e5_sub50_manifest.jsonl"))
    ap.add_argument("--n_docs", type=int, default=50)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    sub = e5_subset(rows, args.n_docs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_pid = dict(sorted(Counter(int(r["provider_id"]) for r in sub).items()))
    print(f"[DONE] {len(sub)}/{len(rows)} docs -> {args.out} (per provider: {per_pid})")


if __name__ == "__main__":
    main()
