"""Build the E1 real-domain manifest (paper App. app:multidomain, Table tab:multidomain).

m=6 providers from three real-world domains, two providers per domain so that
every target has a same-domain sibling:

  - clinical (P0, P1): PubMed abstracts      [HF qiaojin/PubMedQA, pqa_labeled contexts]
  - cti      (P2, P3): analyst CTI reports   [HF guychuk/cyber-threat-intelligence-reports]
  - legal    (P4, P5): case summaries        [HF joelniklaus/legal_case_document_summarization]

Documents are fetched through the HF datasets-server rows API (stdlib only, no
``datasets`` dependency), filtered by length, deduplicated by content prefix and
truncated to ``--max_chars`` (as in the FARAD manifest). Within a domain, docs
are sorted by sha256(id) and assigned alternately to the domain's two providers.

Output schema matches build_corpora_shared_manifest.py:
  {id, title, provider_id, text_src, domain}   ("domain" is ignored downstream)

LICENCES. qiaojin/PubMedQA is MIT-licensed. guychuk/cyber-threat-intelligence-reports
and joelniklaus/legal_case_document_summarization list no licence on the HF Hub
(the latter derives from Shukla et al., AACL 2022). The manifest and anything
built from it (watermarked corpus, trial logs) are therefore NOT redistributed
with this repository; check the upstream terms before sharing them.

REPRODUCIBILITY. The rows API is not revision-pinned, so upstream edits can
change the selection. ``configs/e1_multidomain_manifest_record.jsonl`` lists
id, provider, domain, character count and sha256 of ``text_src`` of the paper
manifest (no text). ``--check_record`` compares a fresh build against it;
``--write_record`` writes such a record for any manifest.

Usage (paper):
  python experiments/build_manifest_multidomain.py \
      --out_manifest work/multidomain_manifest_p6_d300.jsonl --docs_per_provider 50 \
      --check_record configs/e1_multidomain_manifest_record.jsonl
  # record of an existing manifest, without fetching:
  python experiments/build_manifest_multidomain.py \
      --out_manifest work/multidomain_manifest_p6_d300.jsonl --record_only \
      --write_record configs/e1_multidomain_manifest_record.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def _get_json(url: str, retries: int = 5) -> Dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rentarag-e1/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def iter_rows(dataset: str, config: str, split: str, max_rows: int) -> Iterable[Dict[str, Any]]:
    fetched = 0
    offset = 0
    while fetched < max_rows:
        length = min(PAGE, max_rows - fetched)
        url = (
            f"{ROWS_API}?dataset={urllib.parse.quote(dataset, safe='')}"
            f"&config={config}&split={split}&offset={offset}&length={length}"
        )
        data = _get_json(url)
        rows = data.get("rows", [])
        if not rows:
            return
        for rr in rows:
            yield rr["row"]
            fetched += 1
        offset += len(rows)
        total = data.get("num_rows_total")
        if total is not None and offset >= int(total):
            return


_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    # Normalize whitespace but keep paragraph breaks so sentence splitting behaves.
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text[:500].encode("utf-8", errors="ignore")).hexdigest()


def collect_clinical(need: int, min_chars: int, max_chars: int) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    seen: set[str] = set()
    for row in iter_rows("qiaojin/PubMedQA", "pqa_labeled", "train", max_rows=1000):
        ctx = row.get("context") or {}
        parts = ctx.get("contexts") if isinstance(ctx, dict) else None
        if not parts:
            continue
        text = _clean(" ".join(str(p) for p in parts))
        if len(text) < min_chars:
            continue
        h = _content_hash(text)
        if h in seen:
            continue
        seen.add(h)
        title = _clean(str(row.get("question") or ""))[:120] or "PubMed abstract"
        docs.append({"id": f"med-{row.get('pubid', h[:10])}", "title": title,
                     "text_src": text[:max_chars], "domain": "clinical"})
        if len(docs) >= need:
            break
    return docs


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_FIG_RE = re.compile(r"^\s*(figure|table)\s+\d+\s*[:.]", re.IGNORECASE)


def _clean_cti(text: str) -> str:
    """Drop PDF-extraction noise lines (URLs, page footers, figure captions)."""
    kept: List[str] = []
    for line in text.split("\n"):
        if _FIG_RE.match(line):
            continue
        stripped = _URL_RE.sub(" ", line)
        n_alpha = sum(c.isalpha() for c in stripped)
        # Footer/pagination lines are mostly URLs, digits, and punctuation.
        if n_alpha < 30:
            continue
        kept.append(stripped)
    return _clean("\n".join(kept))


def collect_cti(need: int, min_chars: int, max_chars: int) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    seen: set[str] = set()
    for row in iter_rows("guychuk/cyber-threat-intelligence-reports", "default", "train",
                         max_rows=978):
        text = _clean_cti(_clean(str(row.get("text") or "")))
        if len(text) < min_chars:
            continue
        h = _content_hash(text)
        if h in seen:
            continue
        seen.add(h)
        title = _clean(str(row.get("title") or ""))[:120] or "CTI report"
        docs.append({"id": f"cti-{h[:10]}", "title": title,
                     "text_src": text[:max_chars], "domain": "cti"})
        if len(docs) >= need:
            break
    return docs


def collect_legal(need: int, min_chars: int, max_chars: int) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    seen: set[str] = set()
    # Case *summaries* (the editorial layer), not judgements (wording-sensitive).
    for row in iter_rows("joelniklaus/legal_case_document_summarization", "default", "train",
                         max_rows=3000):
        text = _clean(str(row.get("summary") or ""))
        if len(text) < min_chars:
            continue
        h = _content_hash(text)
        if h in seen:
            continue
        seen.add(h)
        name = _clean(str(row.get("dataset_name") or ""))[:40]
        docs.append({"id": f"leg-{h[:10]}", "title": f"Case summary ({name})",
                     "text_src": text[:max_chars], "domain": "legal"})
        if len(docs) >= need:
            break
    return docs


def assign_providers(docs: List[Dict[str, str]], base_pid: int) -> None:
    """Deterministic, exactly balanced within-domain split across two providers."""
    docs.sort(key=lambda d: hashlib.sha256(d["id"].encode()).hexdigest())
    for i, d in enumerate(docs):
        d["provider_id"] = base_pid + (i % 2)


def manifest_record(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Text-free fingerprint of a manifest: id, provider, domain, length, sha256."""
    return [{"id": r["id"], "provider_id": int(r["provider_id"]), "domain": r.get("domain"),
             "n_chars": len(r["text_src"]),
             "sha256": hashlib.sha256(r["text_src"].encode("utf-8")).hexdigest()}
            for r in rows]


def check_record(rows: List[Dict[str, Any]], record_path: Path) -> bool:
    """Compare a manifest with a stored record; print and return whether they match."""
    want = {r["id"]: r for r in (json.loads(l) for l in record_path.open(encoding="utf-8") if l.strip())}
    got = {r["id"]: r for r in manifest_record(rows)}
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    changed = sorted(i for i in set(want) & set(got) if want[i] != got[i])
    ok = not (missing or extra or changed)
    print(f"[RECORD] {record_path}: {len(want)} expected, {len(got)} built, "
          f"{len(missing)} missing, {len(extra)} extra, {len(changed)} differ "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    for name, ids in (("missing", missing), ("extra", extra), ("differ", changed)):
        if ids:
            print(f"  {name}: {ids[:10]}{' ...' if len(ids) > 10 else ''}")
    return ok


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build multi-domain manifest (E1).")
    ap.add_argument("--out_manifest", type=Path, required=True)
    ap.add_argument("--docs_per_provider", type=int, default=50)
    ap.add_argument("--min_chars", type=int, default=1000)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--write_record", type=Path, default=None,
                    help="Write the text-free id/sha256 record of the manifest here.")
    ap.add_argument("--check_record", type=Path, default=None,
                    help="Compare the manifest with this record; exit 1 on mismatch.")
    ap.add_argument("--record_only", action="store_true",
                    help="Do not fetch: read --out_manifest and only write/check records.")
    args = ap.parse_args()

    if args.record_only:
        rows = [json.loads(l) for l in args.out_manifest.open(encoding="utf-8") if l.strip()]
        _finish_records(rows, args)
        return

    need = 2 * args.docs_per_provider
    domains = [
        ("clinical", collect_clinical, 0),
        ("cti", collect_cti, 2),
        ("legal", collect_legal, 4),
    ]

    all_docs: List[Dict[str, str]] = []
    for name, collector, base_pid in domains:
        docs = collector(need, args.min_chars, args.max_chars)
        if len(docs) < need:
            raise SystemExit(f"[FATAL] domain '{name}': only {len(docs)}/{need} docs "
                             f"after filters (min_chars={args.min_chars}).")
        docs = docs[:need]
        assign_providers(docs, base_pid)
        all_docs.extend(docs)
        lens = [len(d["text_src"]) for d in docs]
        print(f"[{name}] {len(docs)} docs -> providers {base_pid},{base_pid+1} | "
              f"chars min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

    rows = [{"id": d["id"], "title": d["title"], "provider_id": d["provider_id"],
             "text_src": d["text_src"], "domain": d["domain"]} for d in all_docs]
    _write_jsonl(args.out_manifest, rows)

    per_pid: Dict[int, int] = {}
    for d in all_docs:
        per_pid[d["provider_id"]] = per_pid.get(d["provider_id"], 0) + 1
    print(f"[DONE] {len(all_docs)} docs -> {args.out_manifest}")
    print(f"[DONE] per-provider counts: {dict(sorted(per_pid.items()))}")
    _finish_records(rows, args)


def _finish_records(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if args.write_record is not None:
        _write_jsonl(args.write_record, manifest_record(rows))
        print(f"[RECORD] wrote {args.write_record}")
    if args.check_record is not None and not check_record(rows, args.check_record):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
