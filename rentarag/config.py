"""One config file for keys, components and parameters.

YAML (``.yaml``/``.yml``), TOML (``.toml``) and JSON (``.json``) files are
supported, with the sections::

    keys: provider_keys.json          # key file (paths are relative to the config file)
    paraphraser: {spec: openai/gpt-4o-mini, prompt: corpus}
    embedder: {spec: sentence-transformers/all-mpnet-base-v2}
    watermark: {preset: paper, rounds: 6, ...}     # WatermarkConfig fields
    audit: {preset: paper, min_support: 25, ...}   # AuditConfig fields
    corpus: {id_field: id, provider_field: provider, text_field: text}
    queries: {llm: openai/gpt-4o-mini, n: 100}
    workers: 8

``rentarag init-config`` writes a commented example with the paper defaults.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import AuditConfig
from .embed import DEFAULT_EMBED_MODEL, Embedder, get_embedder
from .keys import ProviderKeys
from .paraphrase import Paraphraser, get_paraphraser
from .watermark import WatermarkConfig

__all__ = ["RentaragConfig", "load_config", "EXAMPLE_CONFIG", "write_example_config"]

_SECTIONS = ("keys", "paraphraser", "embedder", "watermark", "audit", "corpus", "queries", "workers")
_CORPUS_KEYS = ("id_field", "provider_field", "text_field", "out_field", "default_provider")


def _spec_dict(value: Any, default: str) -> dict[str, Any]:
    if value is None:
        return {"spec": default}
    if isinstance(value, str):
        return {"spec": value}
    if isinstance(value, Mapping):
        d = dict(value)
        d.setdefault("spec", default)
        return d
    raise TypeError(f"expected a spec string or a mapping with 'spec', got {value!r}")


def _explicit(section: Any) -> dict[str, Any]:
    return {k: v for k, v in section.items() if k != "preset"} if isinstance(section, Mapping) else {}


@dataclass
class RentaragConfig:
    """Parsed config file. Build components with the ``make_*`` methods."""

    keys: str | None = None
    paraphraser: dict[str, Any] = field(default_factory=lambda: {"spec": "openai/gpt-4o-mini"})
    embedder: dict[str, Any] = field(default_factory=lambda: {"spec": DEFAULT_EMBED_MODEL})
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    corpus: dict[str, Any] = field(default_factory=dict)
    queries: dict[str, Any] = field(default_factory=dict)
    workers: int = 1
    base_dir: Path | None = None
    # explicit (non-preset) keys of the watermark / audit sections, re-applied
    # on top of a preset chosen on the command line
    watermark_overrides: dict[str, Any] = field(default_factory=dict, repr=False)
    audit_overrides: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, base_dir: str | Path | None = None) -> RentaragConfig:
        data = dict(data or {})
        unknown = sorted(set(data) - set(_SECTIONS))
        if unknown:
            raise ValueError(f"unknown config section(s) {unknown}; valid sections: {', '.join(_SECTIONS)}")
        corpus = dict(data.get("corpus") or {})
        bad = sorted(set(corpus) - set(_CORPUS_KEYS))
        if bad:
            raise ValueError(f"unknown corpus option(s) {bad}; valid options: {', '.join(_CORPUS_KEYS)}")
        return cls(
            keys=data.get("keys"),
            paraphraser=_spec_dict(data.get("paraphraser"), "openai/gpt-4o-mini"),
            embedder=_spec_dict(data.get("embedder"), DEFAULT_EMBED_MODEL),
            watermark=WatermarkConfig.coerce(data.get("watermark")),
            audit=AuditConfig.coerce(data.get("audit")),
            corpus=corpus,
            queries=dict(data.get("queries") or {}),
            workers=int(data.get("workers") or 1),
            base_dir=Path(base_dir) if base_dir is not None else None,
            watermark_overrides=_explicit(data.get("watermark")),
            audit_overrides=_explicit(data.get("audit")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "keys": self.keys,
            "paraphraser": dict(self.paraphraser),
            "embedder": dict(self.embedder),
            "watermark": self.watermark.to_dict(),
            "audit": self.audit.to_dict(),
            "corpus": dict(self.corpus),
            "queries": dict(self.queries),
            "workers": self.workers,
        }

    def resolve_path(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() or self.base_dir is None else self.base_dir / p

    def make_paraphraser(self, **overrides: Any) -> Paraphraser:
        kw = {**self.paraphraser, **overrides}
        return get_paraphraser(kw.pop("spec"), **kw)

    def make_embedder(self, **overrides: Any) -> Embedder:
        kw = {**self.embedder, **overrides}
        return get_embedder(kw.pop("spec"), **kw)

    def load_keys(self) -> ProviderKeys:
        if not self.keys:
            raise ValueError("the config has no 'keys' entry")
        return ProviderKeys.load(self.resolve_path(self.keys))


def _parse(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("reading YAML configs needs PyYAML: pip install pyyaml (or use .toml / .json)") from exc
        return yaml.safe_load(text)
    if suffix == ".toml":
        import tomllib

        return tomllib.loads(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"unsupported config format {suffix!r}: use .yaml, .toml or .json")


def load_config(path: str | Path) -> RentaragConfig:
    """Read a config file (``.yaml``, ``.toml`` or ``.json``); relative paths resolve against its directory."""
    p = Path(path)
    data = _parse(p)
    if data is not None and not isinstance(data, Mapping):
        raise ValueError(f"{p}: the config must be a mapping of sections")
    return RentaragConfig.from_dict(data, base_dir=p.parent)


EXAMPLE_CONFIG = """\
# rentarag configuration (paper defaults). Used by the CLI via --config and by
# rentarag.load_config(path) in Python. Command-line flags override these values.

# Provider key file, created with `rentarag keygen`. Keep it secret.
keys: provider_keys.json

# Paraphrase LLM used to watermark (provider side). Any of:
#   openai/<model>      OPENAI_API_KEY; add base_url for any OpenAI-compatible server
#   anthropic/<model>   ANTHROPIC_API_KEY            e.g. anthropic/claude-haiku-5-5
#   gemini/<model>      GEMINI_API_KEY               e.g. gemini/gemini-2.5-flash
#   ollama/<model>      local Ollama                 e.g. ollama/llama3.1
#   vllm/<model>        local vLLM server            e.g. vllm/Qwen/Qwen2.5-7B-Instruct
#   hf/<model id>       local transformers model     e.g. hf/Qwen/Qwen2.5-1.5B-Instruct
#   litellm/<model>     anything LiteLLM supports
#   toy                 offline stand-in for smoke tests (not fluent)
paraphraser:
  spec: openai/gpt-4o-mini
  prompt: corpus            # corpus (paper) | k-variants (asks for n_candidates variants) | your template with {sentence}
  temperatures: [0.7, 1.0]  # one call per temperature (the second only if more candidates are needed)
  max_retries: 5            # rate limits / timeouts / 5xx are retried with exponential backoff
  # base_url: http://localhost:8000/v1
  # api_key: ...            # prefer environment variables

# Sentence encoder. Provider and auditor MUST use the same one.
# Any sentence-transformers id, openai/text-embedding-3-small, or hashing (toy).
embedder:
  spec: sentence-transformers/all-mpnet-base-v2

# Watermarking. Presets: paper | paper-corpus | fast | strong. Keys below override the preset.
watermark:
  preset: paper
  n_candidates: 24          # K: candidates per round (more = larger gains, more LLM tokens)
  rounds: 6                 # rho: hill-climbing rounds (more = stronger signal, more calls and drift)
  min_sem_sim: 0.70         # tau_sem: min cosine to the current sentence (higher = closer to the source)
  min_delta_proj: 0.0005    # delta_min: min score gain to accept a rewrite
  bucket_bits: 96           # SimHash bits h (must match the audit section)
  bucket_seed: 2025         # public SimHash seed (must match the audit section)
  max_chars: null           # truncate documents (paper corpus build: 4000)
  max_sentences: null       # keep at most this many sentences (paper corpus build: 12)
  sentence_selection: first # first | mmr | all
  coverage: 1.0             # fraction of sentences to watermark

# Auditing. Presets: paper | paper-query | small-corpus. Keys below override the preset.
audit:
  preset: paper
  win_words: 28             # W: window length in words
  stride_words: 14          # s: window stride
  sim_threshold: 0.82       # tau_sim: min cosine for an answer window to align
  tie_margin: 0.05          # tau_margin: min best-vs-runner-up margin (null = off)
  min_support: 25           # T_min: aligned windows needed before a provider is tested
  alpha: 0.05               # Benjamini-Hochberg FDR level
  reference_selection: all  # all | query (per-query references, as in the paper's audits)
  bucket_bits: 96
  bucket_seed: 2025

# Input field names for corpus files (JSONL / JSON / CSV).
corpus:
  id_field: id
  provider_field: null      # null = 'provider', else 'provider_id'
  text_field: text
  out_field: text_wm

# Audit-query generation (`rentarag queries`).
queries:
  llm: openai/gpt-4o-mini
  n: 100

# Parallel worker threads for API calls.
workers: 8
"""


def write_example_config(path: str | Path, *, overwrite: bool = False) -> Path:
    """Write :data:`EXAMPLE_CONFIG` to ``path`` (refuses to overwrite unless asked)."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(17, "refusing to overwrite (use --force)", str(p))
    p.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    return p
