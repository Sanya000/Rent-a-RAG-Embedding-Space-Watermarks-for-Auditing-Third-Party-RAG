"""Build the shared FARAD manifest (work/farad_shared_manifest_p6_d400_seed13.jsonl).

Each row is {id, title, provider_id, text_src}; provider_id = sha256("13:" + id) mod P.
No API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

from tqdm import tqdm

from build_corpora_dirbucket import iter_farad_articles, stable_partition


def _load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            did = obj.get("id")
            if did:
                done.add(str(did))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a shared manifest (id/title/provider_id/text_src) so multiple watermark "
            "schemes can be built on the exact same document set."
        )
    )
    ap.add_argument("--farad_dir", type=Path, required=True)
    ap.add_argument("--out_manifest", type=Path, required=True)
    ap.add_argument("--providers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_docs", type=int, default=400)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(args.out_manifest) if args.resume else set()

    if len(done_ids) >= int(args.max_docs):
        print(
            f"[DONE] manifest already has {len(done_ids)} docs (>= max_docs={args.max_docs}): "
            f"{args.out_manifest}"
        )
        return

    mode = "a" if args.resume else "w"
    seen_this_run: Set[str] = set()
    written = 0
    remaining = max(0, int(args.max_docs) - len(done_ids))

    with args.out_manifest.open(mode, encoding="utf-8") as f_out:
        pbar = tqdm(total=remaining, desc="Build shared manifest", unit="doc")
        for row in iter_farad_articles(args.farad_dir):
            if written >= remaining:
                break
            doc_id = str(row.get("id") or "").strip()
            if not doc_id:
                continue
            if doc_id in done_ids or doc_id in seen_this_run:
                continue

            text_src = str(row.get("text") or "").strip()
            if not text_src:
                continue
            text_src = text_src[: int(args.max_chars)]
            provider_id = int(stable_partition(doc_id, int(args.providers), int(args.seed)))

            rec: Dict[str, object] = {
                "id": doc_id,
                "title": str(row.get("title") or ""),
                "provider_id": provider_id,
                "text_src": text_src,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen_this_run.add(doc_id)
            written += 1
            pbar.update(1)
            if (written % 10) == 0:
                f_out.flush()
        pbar.close()

    print(
        f"[DONE] shared manifest +{written} docs → {args.out_manifest} "
        f"(total_target={args.max_docs}, resume={args.resume})"
    )


if __name__ == "__main__":
    main()

