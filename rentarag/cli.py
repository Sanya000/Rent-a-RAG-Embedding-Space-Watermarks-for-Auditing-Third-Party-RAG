"""Command-line interface: ``rentarag init-config | keygen | watermark | queries | audit``.

Every command accepts ``--config FILE`` (see ``rentarag init-config``);
flags given on the command line override the config file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audit import AUDIT_PRESETS, AuditConfig, Auditor, answer_text
from .config import RentaragConfig, load_config, write_example_config
from .corpus import _read_rows, load_corpus, watermark_corpus
from .keys import ProviderKeys, UnknownProviderError, _parse_providers
from .llm import LLMError
from .paraphrase import _LLM_ONLY_OPTIONS, CallableParaphraser, Paraphraser, get_paraphraser
from .queries import generate_audit_queries
from .watermark import WATERMARK_PRESETS, WatermarkConfig

_S = argparse.SUPPRESS


def _parse_tie_margin(s: str) -> float | None:
    return None if s.strip().lower() in {"none", "off", "null"} else float(s)


def _config(args: argparse.Namespace) -> RentaragConfig:
    return load_config(args.config) if getattr(args, "config", None) else RentaragConfig()


def _keys(args: argparse.Namespace, cfg: RentaragConfig) -> tuple[ProviderKeys, Path]:
    if args.keys is not None:
        path = Path(args.keys)
    elif cfg.keys:
        path = cfg.resolve_path(cfg.keys)
    else:
        sys.exit(f"rentarag {args.cmd}: error: no key file; pass --keys or set 'keys' in the config")
    return ProviderKeys.load(path), path


def _embedder(args: argparse.Namespace, cfg: RentaragConfig) -> Any:
    spec = args.embedder or args.embed_model
    return cfg.make_embedder() if spec is None else cfg.make_embedder(spec=spec)


def _corpus_opt(args: argparse.Namespace, cfg: RentaragConfig, name: str, default: Any) -> Any:
    v = getattr(args, name, None)
    return v if v is not None else cfg.corpus.get(name, default)


def _overrides(args: argparse.Namespace, mapping: dict[str, str]) -> dict[str, Any]:
    return {field: getattr(args, flag) for flag, field in mapping.items() if hasattr(args, flag)}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_init_config(args: argparse.Namespace) -> None:
    p = write_example_config(args.out, overwrite=args.force)
    print(f"wrote {p}; edit it, then pass --config {p}")


def cmd_keygen(args: argparse.Namespace) -> None:
    ids = _parse_providers(args.providers)
    if not ids:
        sys.exit("rentarag keygen: error: --providers needs a count or a comma-separated list of names")
    keys = ProviderKeys.generate(ids, seed=args.seed)
    meta: dict[str, Any] = {"seed": args.seed} if args.seed is not None else {}
    keys.save(args.out, **meta)
    names = "" if keys.is_indexed else f" ({', '.join(map(str, keys.ids))})"
    print(f"wrote {len(keys)} keys{names} to {args.out}" + (" (seeded: NOT secret)" if args.seed is not None else ""))


_WM_FLAGS = {
    "candidates": "n_candidates",
    "rounds": "rounds",
    "min_sem_sim": "min_sem_sim",
    "min_delta_proj": "min_delta_proj",
    "max_chars": "max_chars",
    "max_sentences": "max_sentences",
    "selection": "sentence_selection",
    "coverage": "coverage",
}


class _LazyParaphraser(CallableParaphraser):
    """Builds the real paraphraser on first use, so a completed ``--resume`` needs no credentials."""

    def __init__(self, build: Callable[[], Paraphraser]):
        super().__init__(lambda sentence, n: [], takes_n=True)
        self._build = build
        self._inner: Paraphraser | None = None
        self._lock = threading.Lock()

    def __call__(self, sentence: str, n: int) -> list[str]:
        if self._inner is None:
            with self._lock:
                if self._inner is None:
                    self._inner = self._build()
        return self._inner(sentence, n)


def _paraphraser(args: argparse.Namespace, cfg: RentaragConfig) -> Paraphraser:
    """--paraphraser / --toy / --model+--base-url, else the config file, else openai/gpt-4o-mini."""
    generic = {k: v for k, v in cfg.paraphraser.items() if k in _LLM_ONLY_OPTIONS}
    if args.toy:
        spec, kw = "toy", {}
    elif args.paraphraser:
        spec, kw = args.paraphraser, generic
    elif args.model or args.base_url:
        spec, kw = f"openai/{args.model or 'gpt-4o-mini'}", dict(generic)
        if args.base_url:
            kw["base_url"] = args.base_url
    else:
        kw = {k: v for k, v in cfg.paraphraser.items() if k != "spec"}
        spec = cfg.paraphraser["spec"]
    if args.prompt:
        kw["prompt"] = args.prompt
    return get_paraphraser(spec, **kw)


def cmd_watermark(args: argparse.Namespace) -> None:
    cfg = _config(args)
    if not args.input.exists():
        sys.exit(f"rentarag watermark: error: input file not found: {args.input}")
    if args.out.exists() and args.input.resolve() == args.out.resolve():
        sys.exit("rentarag watermark: error: --input and --out must be different files")
    keys, key_path = _keys(args, cfg)
    base = WatermarkConfig.preset(args.preset, **cfg.watermark_overrides) if args.preset else cfg.watermark
    wcfg = base.replace(**_overrides(args, _WM_FLAGS))
    if args.resume and args.out.exists():
        paraphraser: Paraphraser = _LazyParaphraser(lambda: _paraphraser(args, cfg))
    else:
        paraphraser = _paraphraser(args, cfg)
    embedder = _embedder(args, cfg)
    if not args.resume and args.out.exists():
        args.out.unlink()
    try:
        corpus = watermark_corpus(
            args.input,
            keys,
            paraphraser=paraphraser,
            embedder=embedder,
            config=wcfg,
            max_workers=args.workers or cfg.workers,
            cache_path=args.out,
            progress=not args.quiet,
            id_field=_corpus_opt(args, cfg, "id_field", "id"),
            provider_field=_corpus_opt(args, cfg, "provider_field", None),
            text_field=_corpus_opt(args, cfg, "text_field", "text"),
            out_field=_corpus_opt(args, cfg, "out_field", "text_wm"),
            default_provider=args.provider if args.provider is not None else cfg.corpus.get("default_provider"),
        )
    except UnknownProviderError as e:
        sys.exit(f"rentarag watermark: error: {e} in {key_path}")
    corpus.save(args.out)
    for pid, s in corpus.summary().items():
        print(
            f"provider={pid} docs={s['documents']} sentences={s['sentences']} "
            f"changed={s['changed_frac']:.2f} gain={s['gain_mean']:.4f}"
        )
    print(f"wrote {len(corpus)} documents to {args.out}")


def cmd_queries(args: argparse.Namespace) -> None:
    cfg = _config(args)
    llm = args.llm or cfg.queries.get("llm", "openai/gpt-4o-mini")
    n = args.n if args.n is not None else cfg.queries.get("n")
    text_field = args.text_field or cfg.corpus.get("text_field", "text")
    docs = load_corpus(
        args.corpus,
        text_field=(text_field, "text", "text_wm"),
        id_field=_corpus_opt(args, cfg, "id_field", "id"),
        provider_field=_corpus_opt(args, cfg, "provider_field", None),
        default_provider="",
    )
    qs = generate_audit_queries([d.text for d in docs], llm, n, seed=args.seed, max_workers=args.workers or cfg.workers)
    if args.out is None:
        print("\n".join(qs))
        return
    if args.out.suffix == ".jsonl":
        args.out.write_text("".join(json.dumps({"query": q}, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    else:
        args.out.write_text("\n".join(qs) + "\n", encoding="utf-8")
    print(f"wrote {len(qs)} queries to {args.out}")


_AUDIT_FLAGS = {
    "sim_threshold": "sim_threshold",
    "tie_margin": "tie_margin",
    "min_support": "min_support",
    "alpha": "alpha",
    "retrieval_model": "query_embedder",
    "query_sim_threshold": "query_sim_threshold",
    "max_reference_docs": "max_reference_docs",
}


def _read_queries(path: Path) -> list[str]:
    if path.suffix.lower() in (".txt", ""):
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [str(r["query"]) for r in _read_rows(path)]


def _http_ask(url: str, field: str, headers: list[str], timeout: float) -> Callable[[str], Any]:
    hdrs = {"Content-Type": "application/json"}
    for h in headers:
        k, _, v = h.partition(":")
        hdrs[k.strip()] = v.strip()

    def ask(query: str) -> Any:
        req = urllib.request.Request(url, data=json.dumps({field: query}).encode("utf-8"), headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    return ask


def _cmd_ask(cmd: str, timeout: float) -> Callable[[str], str]:
    def ask(query: str) -> str:
        # the query goes to stdin; it is never interpolated into the command line
        r = subprocess.run(cmd, shell=True, input=query, capture_output=True, text=True, timeout=timeout, check=True)
        return r.stdout.strip()

    return ask


def cmd_audit(args: argparse.Namespace) -> None:
    cfg = _config(args)
    keys, key_path = _keys(args, cfg)
    base = AuditConfig.preset(args.preset, **cfg.audit_overrides) if args.preset else cfg.audit
    over = _overrides(args, _AUDIT_FLAGS)
    if args.select_by_query:
        over["reference_selection"] = "query"
    acfg = base.replace(**over)
    text_field = args.text_field or (cfg.corpus.get("out_field", "text_wm"), "text")
    live = args.rag_url or args.rag_cmd
    if args.answers is not None and live:
        sys.exit("rentarag audit: error: use either --answers or --rag-url/--rag-cmd, not both")
    if args.answers is None and not live:
        sys.exit("rentarag audit: error: pass --answers FILE, or --queries FILE with --rag-url URL / --rag-cmd CMD")
    if live and args.queries is None:
        sys.exit("rentarag audit: error: a live audit needs --queries FILE (see `rentarag queries`)")
    try:
        auditor = Auditor.from_corpus(
            args.corpus,
            keys,
            acfg,
            _embedder(args, cfg),
            text_field=text_field,
            provider_field=_corpus_opt(args, cfg, "provider_field", None),
            id_field=_corpus_opt(args, cfg, "id_field", "id"),
        )
    except UnknownProviderError as e:
        sys.exit(f"rentarag audit: error: corpus {e} in {key_path}")

    if args.answers is not None:
        for row in _read_rows(args.answers):
            q = row.get(args.query_field) if isinstance(row, dict) else None
            auditor.add_answer(answer_text(row[args.answer_field]), query=q or None)
    else:
        if args.rag_url:
            ask = _http_ask(args.rag_url, args.rag_field, args.rag_header, args.timeout)
        else:
            ask = _cmd_ask(args.rag_cmd, args.timeout)
        auditor.audit(
            ask,
            _read_queries(args.queries),
            stop_when_detected=not args.no_stop,
            max_queries=args.max_queries,
            on_error="skip" if args.skip_errors else "raise",
            progress=not args.quiet,
        )
    report = auditor.report()
    print(report)
    if report.trajectory:
        first = report.first_detected_at
        print(f"queries asked: {len(report.trajectory)}; first detected at query: " + (", ".join(f"{p}@{k}" for p, k in first.items()) or "-"))
    if args.out:
        args.out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rentarag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("init-config", help="write a commented example config (paper defaults)")
    c.add_argument("-o", "--out", type=Path, default=Path("rentarag.yaml"))
    c.add_argument("--force", action="store_true", help="overwrite an existing file")
    c.set_defaults(func=cmd_init_config)

    k = sub.add_parser("keygen", help="generate provider keys")
    k.add_argument("--providers", required=True, help="a count (ids 0..n-1) or comma-separated names, e.g. acme,globex,initech")
    k.add_argument("--seed", type=int, default=None, help="deterministic (public!) keys; the paper used 13")
    k.add_argument("-o", "--out", type=Path, default=Path("provider_keys.json"))
    k.set_defaults(func=cmd_keygen)

    w = sub.add_parser("watermark", help="watermark a corpus (JSONL/JSON/CSV file or a directory of .txt files)")
    w.add_argument("--config", type=Path, default=None)
    w.add_argument("--keys", type=Path, default=None)
    w.add_argument("-i", "--input", type=Path, required=True)
    w.add_argument("-o", "--out", type=Path, required=True, help="output JSONL (also the resume cache)")
    w.add_argument(
        "--paraphraser",
        default=None,
        help="LLM spec: openai/gpt-4o-mini, anthropic/claude-haiku-5-5, gemini/gemini-2.5-flash, ollama/llama3.1, "
        "vllm/MODEL, hf/Qwen/Qwen2.5-1.5B-Instruct, litellm/MODEL, toy",
    )
    w.add_argument("--embedder", default=None, help="encoder spec (default all-mpnet-base-v2; 'hashing' = toy)")
    w.add_argument(
        "--preset", choices=sorted(WATERMARK_PRESETS), default=None, help="replaces the config preset; other config keys still apply"
    )
    w.add_argument("--workers", type=int, default=None, help="documents processed in parallel threads")
    w.add_argument("--provider", default=None, help="provider for rows without a provider field")
    w.add_argument("--id-field", default=None)
    w.add_argument("--provider-field", default=None)
    w.add_argument("--text-field", default=None)
    w.add_argument("--out-field", default=None)
    w.add_argument("--prompt", default=None, help="corpus | k-variants | a template with {sentence}")
    w.add_argument("--model", default=None, help="shorthand for --paraphraser openai/MODEL")
    w.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    w.add_argument("--toy", action="store_true", help="offline rule-based paraphraser (same as --paraphraser toy)")
    w.add_argument("--embed-model", default=None, help=_S)
    w.add_argument("--candidates", type=int, default=_S, help="K: candidates per round")
    w.add_argument("--rounds", type=int, default=_S, help="rho: hill-climbing rounds")
    w.add_argument("--min-sem-sim", type=float, default=_S, help="tau_sem")
    w.add_argument("--min-delta-proj", type=float, default=_S, help="delta_min")
    w.add_argument("--max-chars", type=int, default=_S, help="truncate documents")
    w.add_argument("--max-sentences", type=int, default=_S, help="keep at most this many sentences")
    w.add_argument("--selection", choices=["first", "mmr", "all"], default=_S, help="which sentences --max-sentences keeps")
    w.add_argument("--coverage", type=float, default=_S, help="fraction of sentences to watermark")
    w.add_argument("--resume", action="store_true", help="keep documents already in --out and watermark only the rest")
    w.add_argument("-q", "--quiet", action="store_true")
    w.set_defaults(func=cmd_watermark)

    q = sub.add_parser("queries", help="generate audit queries from a corpus with the paper's prompt")
    q.add_argument("--config", type=Path, default=None)
    q.add_argument("--corpus", type=Path, required=True)
    q.add_argument("--text-field", default=None)
    q.add_argument("--id-field", default=None)
    q.add_argument("--provider-field", default=None)
    q.add_argument("--llm", default=None, help="LLM spec (default openai/gpt-4o-mini)")
    q.add_argument("-n", type=int, default=None, help="number of queries (default: one per document)")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--workers", type=int, default=None)
    q.add_argument("-o", "--out", type=Path, default=None, help=".txt (one per line) or .jsonl")
    q.set_defaults(func=cmd_queries)

    a = sub.add_parser("audit", help="test every provider on RAG answers (from a file or a live system)")
    a.add_argument("--config", type=Path, default=None)
    a.add_argument("--keys", type=Path, default=None)
    a.add_argument("--corpus", type=Path, required=True, help="the watermarked corpus as published")
    a.add_argument("--text-field", default=None, help="field with the watermarked text (default text_wm, else text)")
    a.add_argument("--provider-field", default=None, help="field with the provider id (default provider, else provider_id)")
    a.add_argument("--id-field", default=None)
    a.add_argument("--answers", type=Path, default=None, help="JSONL/JSON/CSV with an answer (and optional query) per row")
    a.add_argument("--answer-field", default="answer")
    a.add_argument("--query-field", default="query")
    a.add_argument("--queries", type=Path, default=None, help="queries for a live audit (.txt or .jsonl)")
    a.add_argument("--rag-url", default=None, help="POST {\"query\": ...} as JSON; the answer is read from the response")
    a.add_argument("--rag-field", default="query", help="JSON field name of the query in --rag-url requests")
    a.add_argument("--rag-header", action="append", default=[], help="extra HTTP header 'Name: value' (repeatable)")
    a.add_argument("--rag-cmd", default=None, help="shell command that reads the query on stdin and prints the answer")
    a.add_argument("--timeout", type=float, default=120.0, help="seconds per RAG call")
    a.add_argument("--max-queries", type=int, default=None)
    a.add_argument("--no-stop", action="store_true", help="ask every query instead of stopping at the first detection")
    a.add_argument("--skip-errors", action="store_true", help="continue when the RAG system fails on a query")
    a.add_argument("--embedder", default=None, help="encoder spec; must be the one used to watermark")
    a.add_argument("--embed-model", default=None, help=_S)
    a.add_argument(
        "--preset", choices=sorted(AUDIT_PRESETS), default=None, help="replaces the config preset; other config keys still apply"
    )
    a.add_argument("--select-by-query", action="store_true", help="per-query reference selection (paper audits)")
    a.add_argument("--retrieval-model", default=_S, help="encoder for --select-by-query")
    a.add_argument("--query-sim-threshold", type=float, default=_S)
    a.add_argument("--max-reference-docs", type=int, default=_S)
    a.add_argument("--sim-threshold", type=float, default=_S, help="tau_sim")
    a.add_argument("--tie-margin", type=_parse_tie_margin, default=_S, help="tau_margin; 'none' or 'off' disables it")
    a.add_argument("--min-support", type=int, default=_S, help="T_min")
    a.add_argument("--alpha", type=float, default=_S, help="BH FDR level")
    a.add_argument("-q", "--quiet", action="store_true")
    a.add_argument("-o", "--out", type=Path, default=None)
    a.set_defaults(func=cmd_audit)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except LLMError as e:
        sys.exit(f"rentarag {args.cmd}: error: {e}")
    except FileNotFoundError as e:
        sys.exit(f"rentarag {args.cmd}: error: file not found: {e.filename or e}")
    except FileExistsError as e:
        sys.exit(f"rentarag {args.cmd}: error: {e.filename} exists (use --force to overwrite)")
    except UnknownProviderError as e:
        sys.exit(f"rentarag {args.cmd}: error: {e}")
    except KeyError as e:
        sys.exit(f"rentarag {args.cmd}: error: missing field {e} (check --text-field / --answer-field)")
    except (ValueError, TypeError, ImportError, subprocess.CalledProcessError) as e:  # includes json.JSONDecodeError
        sys.exit(f"rentarag {args.cmd}: error: {e}")
    except urllib.error.HTTPError as e:
        sys.exit(f"rentarag {args.cmd}: error: RAG endpoint returned HTTP {e.code} {e.reason}; use --skip-errors to continue past failed queries")
    except urllib.error.URLError as e:
        sys.exit(f"rentarag {args.cmd}: error: cannot reach the RAG endpoint ({e.reason}); is it running?")
    except subprocess.TimeoutExpired as e:
        sys.exit(f"rentarag {args.cmd}: error: {e}; raise --timeout or use --skip-errors")
    except OSError as e:  # includes socket timeouts
        sys.exit(f"rentarag {args.cmd}: error: {e}")


if __name__ == "__main__":
    main()
