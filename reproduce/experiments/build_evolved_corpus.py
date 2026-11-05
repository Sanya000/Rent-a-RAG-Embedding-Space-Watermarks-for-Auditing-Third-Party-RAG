"""Build the "evolved" reference corpora for E3 (paper App. app:corpus-evolution, Table tab:corpus-evolution).

Scenario: the operator caches the provider's deployed corpus v1
and answers queries from it; the provider's corpus meanwhile evolves — edited
documents whose sentences are NOT yet re-watermarked ("edited or newly added
sentences remain unwatermarked until re-processed"). The auditor holds the
CURRENT corpus v2 and audits the stale answers against it.

Implementation: fraction f of the deployed watermarked corpus docs get their
text_wm replaced by an unwatermarked meaning-preserving rewrite (the paper's
`neutral` laundering prompt, A.4.3, applied to the deployed doc text). The
rewrite preserves facts (retrieval/alignment still topically plausible) but
destroys the directional watermark and shifts window embeddings (the two
degradation channels: alignment and reference-bucket anchoring).

Evolved-doc sets are NESTED (f=10% ⊂ 25% ⊂ 50%) via a deterministic hash order,
so degradation is monotone-interpretable. Each condition becomes a full work
dir (corpus.jsonl + provider_keys.json) usable by the standard audit machinery.

Rewrites are cached in ``<out_root>/evolved_rewrites.json`` (200 gpt-4o-mini
rewrites for the paper; shipped in the ``corpora_ext`` bundle). With a complete
cache the script is API-free and re-emits the paper's f10/f25/f50 corpora
byte-identically; regenerating rewrites is not deterministic.

Usage:
  python experiments/build_evolved_corpus.py \
      --src_work_dir work/farad_dirbucket_naive --out_root work/staleness
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

# Paper A.4.3 `neutral` condition prompt.
NEUTRAL_PROMPT = """\
Rewrite the following text in different wording while preserving every fact and detail.
Do not add or remove information.
Return only the rewritten text with no preamble.

{text}"""


def evolve_order(doc_ids: List[str]) -> List[str]:
    return sorted(doc_ids, key=lambda i: hashlib.sha256(f"evolve:{i}".encode()).hexdigest())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_work_dir", type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument("--out_root", type=Path, default=Path("work/staleness"))
    ap.add_argument("--fractions", type=str, default="0.10,0.25,0.50")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.src_work_dir / "corpus.jsonl").open(encoding="utf-8")]
    fractions = sorted(float(x) for x in args.fractions.split(","))
    order = evolve_order([str(r["id"]) for r in rows])
    n_max = round(fractions[-1] * len(rows))
    to_evolve = order[:n_max]
    print(f"[evolve] corpus={len(rows)} docs; rewriting {len(to_evolve)} "
          f"(max fraction {fractions[-1]})")

    # ---- Rewrite (resume-safe cache) ----
    args.out_root.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_root / "evolved_rewrites.json"
    cache: Dict[str, str] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    by_id = {str(r["id"]): r for r in rows}
    todo = [d for d in to_evolve if d not in cache]
    if todo:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(f"[FATAL] {len(todo)} rewrites are not cached in {cache_path} "
                             "and OPENAI_API_KEY is not set.")
        import openai
        client = openai.OpenAI()

        def _rewrite(doc_id: str) -> None:
            text = by_id[doc_id]["text_wm"]
            prompt = NEUTRAL_PROMPT.format(text=text)
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=args.model, temperature=args.temperature,
                        messages=[{"role": "user", "content": prompt}])
                    out = (resp.choices[0].message.content or "").strip()
                    if out:
                        cache[doc_id] = out
                    return
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    time.sleep(3.0 * (attempt + 1))

        print(f"[evolve] rewriting {len(todo)} docs with {args.model}...")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for i, _ in enumerate(ex.map(_rewrite, todo), 1):
                if i % 25 == 0:
                    print(f"  [evolve] {i}/{len(todo)}")
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False),
                                          encoding="utf-8")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    missing = [d for d in to_evolve if d not in cache]
    if missing:
        raise SystemExit(f"[FATAL] {len(missing)} rewrites missing (e.g. {missing[:3]})")

    # ---- Emit per-fraction work dirs ----
    for f in fractions:
        n = round(f * len(rows))
        evolved = set(order[:n])
        tag = f"f{int(round(100 * f)):02d}"
        out_dir = args.out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                rec = dict(r)
                if str(r["id"]) in evolved:
                    rec["text_wm"] = cache[str(r["id"])]
                    rec["evolved"] = True
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        shutil.copy(args.src_work_dir / "provider_keys.json", out_dir / "provider_keys.json")
        print(f"[emit] {out_dir}: {n}/{len(rows)} docs evolved")

    print("[DONE]")


if __name__ == "__main__":
    main()
