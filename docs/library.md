# Using `rentarag`

`rentarag` implements **DirBucket**, the embedding-space watermark from *Rent-a-RAG:
Embedding-Space Watermarks for Auditing Third-Party RAG* (EMNLP 2026). Providers watermark the
documents they rent to RAG services; an auditor queries a black-box RAG service and decides,
with false-discovery-rate control, whose content it is using.

Every component is pluggable: any LLM or API for paraphrasing, any sentence encoder, any corpus
format, any number of providers and any parameter set. This page is the practical guide; the
compact API reference is in [api.md](api.md).

```bash
pip install -e .                    # core: numpy, scipy, sentence-transformers, tqdm, pyyaml
pip install -e ".[openai]"          # + OpenAI / Gemini / Ollama / vLLM / any OpenAI-compatible server
pip install -e ".[anthropic]"       # + Anthropic
pip install -e ".[hf]"              # + local transformers models
pip install -e ".[litellm]"         # + everything LiteLLM supports (Bedrock, Vertex, Azure, Mistral, ...)
pip install -e ".[all]"             # all of the above
pip install -e ".[dev]" && pytest   # offline test suite
```

## In ten lines

```python
import rentarag

keys = rentarag.ProviderKeys.generate(["acme", "globex", "initech"])        # one secret key per provider
keys.save("provider_keys.json")

marked = rentarag.watermark_corpus("corpus.jsonl", keys,                     # provider side
                                   paraphraser="openai/gpt-4o-mini", max_workers=16)
marked.save("corpus_wm.jsonl")                                               # publish text_wm

auditor = rentarag.Auditor.from_corpus(marked, keys)                         # auditor side
queries = rentarag.generate_audit_queries(marked, "openai/gpt-4o-mini", n=200)
report = auditor.audit(rag_chain.invoke, queries)                            # any query -> answer function
print(report)                                                                # Z, p, q(BH), DETECTED per provider
```

Everything above runs offline with `paraphraser="toy"`, `embedder="hashing"` and a toy RAG
function; see [`examples/`](../examples).

## Concepts

| Term | Meaning |
|---|---|
| **Provider key** | 256-bit secret per provider. Only the provider and the auditor hold it. |
| **Encoder** | Sentence embedding model (paper: `all-mpnet-base-v2`, d = 768). Provider and auditor use the same one. |
| **Bucket / direction** | Each embedding falls in a SimHash bucket (96 public hyperplanes); each (key, bucket) pair defines a secret unit direction. |
| **Watermarking** | Hill climbing per sentence: an LLM proposes `K` paraphrases per round for `ρ` rounds; the rewrite that moves the embedding furthest along the secret direction while keeping cosine ≥ `τ_sem` is kept. |
| **Alignment** | Answer windows (28 words) are matched to the providers' published documents (cosine ≥ `τ_sim`, margin ≥ `τ_margin` over the runner-up provider) and scored along the matched reference bucket's direction. |
| **Pooled Z and decision** | Scores are pooled per provider over all answers: `Z = mean · sqrt(d · T)` with support `T`. Providers with `T ≥ T_min` are tested; Benjamini-Hochberg at level `α` decides which are **detected**. |

## Bring your own LLM

A paraphraser turns a sentence into candidate rewrites. Pick one with a spec string; the SDK is
imported only when used.

| Spec | Backend | Credentials / server | Extra |
|---|---|---|---|
| `openai/gpt-4o-mini` | OpenAI Chat Completions (paper default) | `OPENAI_API_KEY` | `[openai]` |
| `openai/<model>` + `base_url=...` | any OpenAI-compatible server (llama.cpp, LM Studio, TGI, Together, Groq, OpenRouter, ...) | the server's key, if any | `[openai]` |
| `anthropic/claude-haiku-5-5` | Anthropic Messages API | `ANTHROPIC_API_KEY` | `[anthropic]` |
| `gemini/gemini-2.5-flash` | Gemini (OpenAI-compatible endpoint) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `[openai]` |
| `ollama/llama3.1` | local Ollama | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | `[openai]` |
| `vllm/Qwen/Qwen2.5-7B-Instruct` | local `vllm serve` | `VLLM_BASE_URL` (default `http://localhost:8000/v1`) | `[openai]` |
| `hf/Qwen/Qwen2.5-1.5B-Instruct` | local `transformers` model, chat template, sampling | `device=`, `dtype=`, `seed=` | `[hf]` |
| `litellm/<model>` | any LiteLLM provider (`litellm/bedrock/...`, `litellm/azure/...`) | as LiteLLM expects | `[litellm]` |
| `toy` | rule-based offline stand-in (not fluent; tests and demos) | none | none |

```python
para = rentarag.get_paraphraser("anthropic/claude-haiku-5-5")
para = rentarag.get_paraphraser("ollama/llama3.1", prompt="k-variants", max_retries=8)
para = rentarag.get_paraphraser("openai/my-model", base_url="http://gpu-box:8000/v1")
para = rentarag.get_paraphraser("hf/Qwen/Qwen2.5-1.5B-Instruct", device="cuda", seed=0)
para("The council approved the budget on Monday.", 8)   # -> up to 8 rewrites
```

**Your own code.** Any function works:

```python
rentarag.get_paraphraser(lambda sentence, n: my_model.rewrite(sentence, k=n))   # (sentence, n) -> list[str]
rentarag.get_paraphraser(lambda sentence: my_model.rewrite_one(sentence))       # sentence -> str, sampled up to n times
rentarag.LLMParaphraser(lambda prompt: my_gateway.complete(prompt))             # prompt -> text, with the paper prompt
```

**Prompts and call pattern.** `prompt="corpus"` (default) is the prompt that built the paper's
corpora: it asks for 8 variants (2 light, 3 medium, 3 strong) per call, with one call at
temperature 0.7 and a second at 1.0 when more candidates are needed. `prompt="k-variants"` asks
for exactly `n_candidates` variants and always makes both calls. Your own template is any string
with `{sentence}` and optionally `{n}`. `temperatures=` and `max_tokens=` set the calls.

**Retries.** Rate limits, timeouts, connection errors and 5xx responses are retried with
exponential backoff and jitter (`max_retries=5`, `backoff=1.0` s, capped at 60 s; a server's
`retry-after` is honoured). Other errors raise `ParaphraseError`; with `on_error="warn"` the
sentence is left unchanged and counted in `para.failures`.

**Throughput.** `watermark_corpus(..., max_workers=16)` processes documents in parallel threads,
which is what API backends need: most time is network latency. Results come back in input order.
Local `hf/` models serialise generation on one model instance.

The same backends drive audit-query generation: `rentarag.get_llm(spec)` returns the bare chat
model (`.complete(prompt, temperature=..., max_tokens=...)`).

## Bring your own encoder

```python
rentarag.get_embedder()                                        # all-mpnet-base-v2 (paper)
rentarag.get_embedder("BAAI/bge-small-en-v1.5")               # any sentence-transformers / HF id
rentarag.get_embedder("openai/text-embedding-3-small")         # embeddings API, batched, retried
rentarag.get_embedder("ollama/nomic-embed-text")               # local embeddings server
rentarag.get_embedder(lambda texts: my_encoder.encode(texts))  # any texts -> (n, d) array
rentarag.get_embedder("hashing")                               # toy bag-of-words encoder for tests
```

Rows are L2-normalised automatically and the dimension is detected, so buckets and directions
follow whatever encoder you choose. Every function that takes `embedder=` accepts the same specs.
The provider and the auditor must use the same encoder: `watermark_corpus` records it in every
output row (`wm_embedder`, `wm_dim`), and `Auditor.from_corpus` raises an error on a different
dimension and warns on a different model name.
`τ_sem = 0.70` and `τ_sim = 0.82` were tuned for `all-mpnet-base-v2`; for another encoder,
calibrate them on a few documents (watermark gain and alignment support are reported directly).

## Bring your own corpus

`watermark_corpus`, `Auditor.from_corpus` and `generate_audit_queries` take documents in any of
these shapes:

| Input | Provider | Text | Id |
|---|---|---|---|
| `.jsonl` / `.jsonl.gz` / `.json` list | `provider` or `provider_id` field | `text` | `id` (else `doc-<i>`) |
| `.csv` / `.tsv` with a header | column | column | column |
| `.json` mapping `{provider: [texts]}` | key | list item | `<provider>-<j>` |
| directory of `.txt` / `.md` files | first sub-directory (`corpus/acme/a.txt`) | file content | relative path |
| Python: dicts, `(provider, text)`, `(id, provider, text)`, `{provider: [texts]}`, `Document` | | | |

Field names are configurable in all three functions: `id_field=`, `provider_field=`,
`text_field=` (e.g. `text_field="body"`), and `default_provider=` for rows without one (corpus
loading and watermarking). `rentarag.load_corpus(path, ...)` returns the parsed
`Document` list if you want to inspect or filter it first.

`watermark_corpus` returns a `WatermarkedCorpus` (a list of `WatermarkedDoc` with `.text`,
`.original`, `.provider`, `.stats`), with `.save(path)` (JSONL: your input fields plus `text_wm`
and `wm_*` statistics), `.by_provider()` and `.summary()`.

**Resumable runs.** Pass `cache_path="corpus_wm.cache.jsonl"`: each finished document is appended
as soon as it is done, and a re-run skips documents already there (same id, provider and text).
A paraphraser given as a spec string is only built when documents remain, so re-running a finished
job needs no API key.

**Preprocessing.** The config can truncate documents (`max_chars`), keep a subset of sentences
(`max_sentences` with `sentence_selection="first"` or `"mmr"`, the paper's corpus build; `"all"`
ignores `max_sentences`) and
watermark only a fraction of sentences (`coverage`). The `paper-corpus` preset reproduces the
paper's build: 4000 characters, 12 sentences chosen by MMR with `all-MiniLM-L6-v2`.

## Any number of providers

```python
keys = rentarag.ProviderKeys.generate(["acme", "globex", "initech"])   # names
keys = rentarag.ProviderKeys.generate(50)                              # ids 0..49
keys.save("provider_keys.json"); keys = rentarag.ProviderKeys.load("provider_keys.json")
keys["acme"]                                                           # key hex
```

Provider ids in documents may be names or numbers; `"7"` and `7` resolve to the same provider.
An unknown provider raises `UnknownProviderError`. The key file also stores the keys as
`keys_hex`, the layout of the paper's `provider_keys.json`. Every provider in the key file is
tested by the auditor, including providers with no published documents.

## Parameters and presets

Configs are frozen dataclasses with presets, dict round trips and validation (out-of-range
values such as a cosine threshold above 1 raise `ValueError` naming the field):

```python
cfg = rentarag.WatermarkConfig.preset("fast")                       # or "paper", "strong", "paper-corpus"
cfg = rentarag.WatermarkConfig.from_dict({"preset": "fast", "rounds": 4})
cfg.to_dict(); cfg.replace(min_sem_sim=0.75)
rentarag.watermark_corpus(docs, keys, paraphraser=..., config="strong")   # a preset name or dict works too
```

### Watermarking (`WatermarkConfig`)

| Field | Paper | Trade-off |
|---|---|---|
| `n_candidates` (K) | 24 | More candidates per round find larger gains; more LLM tokens. |
| `rounds` (ρ) | 6 | Each accepted round adds signal; each costs 1–2 LLM calls per sentence and moves the text further from the source. |
| `min_sem_sim` (τ_sem) | 0.70 | Higher keeps rewrites closer to the original; lower admits stronger rewrites and more signal. |
| `min_delta_proj` (δ_min) | 5e-4 | Minimum gain for a rewrite to be accepted; rejects edits that add no signal. |
| `bucket_bits`, `bucket_seed` | 96, 2025 | SimHash resolution and public seed; the auditor uses the same values. |
| `max_chars`, `max_sentences`, `sentence_selection`, `mmr_lambda` | off | Shorter deployed documents and fewer LLM calls. |
| `coverage` | 1.0 | Fraction of sentences watermarked; the rest stay verbatim, giving less evidence per answer. |
| `force_accept_last_round` | False | Keep the best valid candidate even if no round improved the score. |

| Preset | Settings | Use |
|---|---|---|
| `paper` | defaults | the paper's operating point |
| `paper-corpus` | `paper` + 4000 chars, 12 MMR sentences | rebuild corpora like the paper |
| `fast` | K = 8, ρ = 3 | about a quarter of the LLM calls; less signal per sentence |
| `strong` | K = 32, ρ = 10 | more signal per sentence; pair with `prompt="k-variants"` |

### Auditing (`AuditConfig`)

| Field | Paper | Trade-off |
|---|---|---|
| `win_words`, `stride_words` (W, s) | 28, 14 | Shorter windows align more easily after rewording; each carries less signal. |
| `sim_threshold` (τ_sim) | 0.82 | Lower gathers support from loosely reworded answers and admits more unrelated text. |
| `tie_margin` (τ_margin) | 0.05 | Drops windows that several providers explain equally well; `None` disables it. |
| `min_support` (T_min) | 25 | Evidence required before a provider is tested; lower decides earlier on less evidence. |
| `alpha` | 0.05 | BH false-discovery rate across providers. |
| `reference_selection` | `"all"` | `"query"` picks references per audit query by retrieval similarity (`query_embedder`, `query_sim_threshold` 0.48, `max_reference_docs` 120), as in the paper's audits; it scales to large corpora. |
| `competitive`, `direction_scheme`, `z_threshold` | True, `"naive"`, 4.0 | assignment rule, direction scheme, reference line for `first_crossed_at` |

| Preset | Settings | Use |
|---|---|---|
| `paper` | defaults | every published document is a reference for every answer |
| `paper-query` | `reference_selection="query"` | the paper's audit setting; large corpora |
| `small-corpus` | `min_support=10` | demos and small corpora that cannot reach 25 aligned windows |

## Auditing a live RAG system

The system under audit is any function from a query to an answer:

```python
auditor = rentarag.Auditor.from_corpus("corpus_wm.jsonl", "provider_keys.json", config="paper-query")
queries = rentarag.generate_audit_queries("corpus.jsonl", "openai/gpt-4o-mini", n=300, max_workers=8)

report = auditor.audit(chain.invoke, queries)                  # LangChain runnable
report = auditor.audit(query_engine.query, queries)            # LlamaIndex query engine
report = auditor.audit(lambda q: requests.post(URL, json={"query": q}).json(), queries)

report.detected            # providers detected at FDR alpha
report.first_detected_at   # {provider: query number}
report.trajectory          # per query: query, answer, pooled Z and support per provider, detections
report.to_dict()           # JSON-ready
```

Return values are read with `rentarag.answer_text`: strings, dicts with `answer` / `result` /
`output` / `response` / `text`, chat-message objects (`.content`) and LlamaIndex responses
(`.response`) all work.

`audit` stops at the first detection by default. `stop_when_detected="acme"` or
`stop_when_detected=["acme", "globex"]` stops once those are detected; `stop_when_detected=False, max_queries=300` runs a fixed budget, which is
the setting for a fixed-size test. `on_error="skip"` records failed calls and continues.
Evidence accumulates on the auditor across calls; `auditor.reset()` starts over.

**Queries.** `generate_audit_queries` uses the paper's prompt: one question per sampled document
that must cite several specific details of it, so honest answers reproduce its content. Any LLM
spec or `prompt -> text` function works. Diverse queries give independent evidence: the pooled
Z treats aligned windows as independent, so asking many different questions is better than
repeating a few. With `n` larger than the number of documents, documents are reused.

**Answers you already have.** `auditor.add_answers(answers)` takes strings, `(query, answer)`
pairs or dicts with `answer` (and `query`), then `auditor.report()`.

## Command line and config file

```bash
rentarag init-config -o rentarag.yaml                          # commented config with paper defaults
rentarag keygen --providers acme,globex,initech -o provider_keys.json    # or --providers 6
rentarag watermark --config rentarag.yaml -i corpus.csv -o corpus_wm.jsonl --workers 16
rentarag watermark --keys provider_keys.json -i corpus/ -o corpus_wm.jsonl \
    --paraphraser ollama/llama3.1 --preset fast --resume
rentarag queries --config rentarag.yaml --corpus corpus.csv -n 300 -o queries.txt
rentarag audit --config rentarag.yaml --corpus corpus_wm.jsonl --answers answers.jsonl -o report.json
rentarag audit --config rentarag.yaml --corpus corpus_wm.jsonl --queries queries.txt \
    --rag-url http://localhost:8080/ask -o report.json          # POST {"query": ...}, answer read from the JSON reply
rentarag audit ... --queries queries.txt --rag-cmd "python my_rag.py"   # query on stdin, answer on stdout
```

Command-line flags override the config file, field by field (`--preset` replaces the section's
preset; the section's other keys still apply). The corpus field names can also be given as flags
(`--id-field`, `--provider-field`, `--text-field`) to `watermark`, `queries` and `audit`. The
config has one section per component
([`examples/config.yaml`](../examples/config.yaml)):

```yaml
keys: provider_keys.json                     # relative to the config file
paraphraser: {spec: openai/gpt-4o-mini, prompt: corpus}
embedder: {spec: sentence-transformers/all-mpnet-base-v2}
watermark: {preset: paper, rounds: 6}
audit: {preset: paper, min_support: 25}
corpus: {id_field: id, text_field: text, out_field: text_wm}
queries: {llm: openai/gpt-4o-mini, n: 100}
workers: 8
```

In Python, `cfg = rentarag.load_config("rentarag.yaml")` gives `cfg.make_paraphraser()`,
`cfg.make_embedder()`, `cfg.load_keys()`, `cfg.watermark` and `cfg.audit`. YAML, TOML and JSON
files are supported.

## Lower-level building blocks

- `DirBucketWatermarker(key_hex, paraphraser, embedder, config)` watermarks one text:
  `.watermark(text)` returns a `WatermarkResult` with per-sentence logs (`original`, `text`, `gain`,
  `sem_sim`, ...); `keep_candidates=True` logs every scored candidate.
- `Auditor(providers, config, embedder)` with `ProviderReference(id, key_hex, docs)` is the
  explicit form of `Auditor.from_corpus`; `auditor.add_answer(answer, query=..., sources=...)`
  scores one answer against chosen references.
- `AlignedScorer(keys, embedder, config).score(answer, sources)` scores a single answer statelessly.
- `pooled_z`, `gate_by_min_support`, `bh_adjust`, `norm_sf` are the statistics.
- `OrthoDirectionManager` provides the jointly orthonormalised directions of the paper's ablation.

The library reproduces the paper implementation in `reproduce/experiments/watermark_dirbucket.py`
numerically; the test suite checks this against that file.

## Security notes

- **Keys are secrets.** Whoever holds a provider's key can compute its directions. Generate keys
  with `ProviderKeys.generate(...)` (OS randomness) and store the key file like any credential.
  Keys made with a seed are public: `seed=13` regenerates the paper's research keys, which ship
  with the released corpora and tests.
- **Provider and auditor share the encoder and SimHash parameters** (`bucket_bits`, `bucket_seed`).
  The SimHash hyperplanes are public by design; secrecy lives in the keyed directions.
- **The auditor holds the published corpus and the keys** of the providers it audits; the RAG
  operator needs neither.
- API keys for LLM backends are read from environment variables or passed explicitly, never from
  files in the repository.
