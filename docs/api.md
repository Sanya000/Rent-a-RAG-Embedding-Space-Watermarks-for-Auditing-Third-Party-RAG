# `rentarag` API reference

Everything below is importable from the top-level package (`import rentarag`). See
[library.md](library.md) for the guide.

## Factories

| Name | Signature | Returns |
|---|---|---|
| `get_paraphraser` | `(spec="openai/gpt-4o-mini", **kw)` | paraphraser `(sentence, n) -> list[str]` from a spec, a chat model or a function |
| `get_llm` | `(spec, **kw)` | `ChatModel` with `.complete(prompt, *, temperature=None, max_tokens=None) -> str` |
| `get_embedder` | `(spec=None, **kw)` | `Embedder`: `(texts) -> float32 (n, dim)` unit rows, with `.dim` |
| `load_config` | `(path)` | `RentaragConfig` from `.yaml` / `.toml` / `.json` |
| `RentaragConfig` | dataclass `keys, paraphraser, embedder, watermark, audit, corpus, queries, workers` | `.make_paraphraser(**kw)`, `.make_embedder(**kw)`, `.load_keys()`, `.resolve_path(p)`, `.from_dict(d)`, `.to_dict()` |
| `write_example_config` | `(path, *, overwrite=False)` | writes the commented paper-default config |

Spec prefixes: `openai/`, `anthropic/`, `gemini/`, `ollama/`, `vllm/`, `hf/`, `litellm/`, `toy`
(LLMs); any sentence-transformers id, `st/`, `hf/`, `openai/`, `ollama/`, `vllm/`, `gemini/`,
`hashing[:dim]` (encoders).

## Keys and corpora

| Name | Signature | Notes |
|---|---|---|
| `ProviderKeys` | `(mapping or list)` | read-only `{provider: key_hex}`; `.generate(providers, *, seed=None)`, `.load(path)` (key files from `.save`, the paper's `provider_keys.json`, a list or `{name: hex}`), `.save(path)`, `.resolve(pid)`, `.ids`, `.keys_hex` |
| `generate_provider_keys` | `(n, seed=None) -> list[str]` | list form; `seed=13` gives the paper's public research keys |
| `load_corpus` | `(path, *, id_field="id", provider_field=None, text_field="text", default_provider=None) -> list[Document]` | `.jsonl[.gz]`, `.json`, `.csv`, `.tsv`, directory of `.txt`/`.md` |
| `iter_documents` | `(docs, **fields) -> Iterator[Document]` | normalises paths, dicts, tuples, mappings |
| `watermark_corpus` | `(docs, keys, *, paraphraser, embedder=None, config=None, max_workers=1, cache_path=None, progress=False, id_field="id", provider_field=None, text_field="text", out_field="text_wm", default_provider=None, selection_embedder=None, keep_sentences=False) -> WatermarkedCorpus` | parallel, resumable, input order; rows record `wm_embedder` and `wm_dim` |
| `WatermarkedCorpus` | `list[WatermarkedDoc]` | `.save(path)`, `.rows()`, `.by_provider()`, `.summary()` |
| `WatermarkedDoc` | dataclass | `id, provider, text, original, stats, meta, provider_field, result`; `.to_row(...)` |
| `Document` | dataclass | `id, provider, text, meta, provider_field` |

## Watermarking

| Name | Signature |
|---|---|
| `WatermarkConfig` | frozen dataclass: `n_candidates=24, rounds=6, min_sem_sim=0.70, min_delta_proj=5e-4, bucket_bits=96, bucket_seed=2025, force_accept_last_round=False, max_chars=None, max_sentences=None, sentence_selection="first", mmr_lambda=0.7, selection_model=None, coverage=1.0`; `.preset(name, **kw)`, `.from_dict(d)`, `.to_dict()`, `.replace(**kw)` |
| `WATERMARK_PRESETS` | `paper`, `paper-corpus`, `fast`, `strong` |
| `Paraphraser` | type alias `Callable[[str, int], list[str]]`: `(sentence, n) -> candidates` |
| `DirBucketWatermarker` | `(key_hex, paraphraser, embedder=None, config=None, *, directions=None, selection_embedder=None, keep_candidates=False, progress=False)`; `.watermark(text, *, doc_id=None, max_workers=1) -> WatermarkResult`, `.watermark_sentence(sentence)`, `.prepare(text)`, `.score(u)` |
| `WatermarkResult` | `text, sentences: list[SentenceLog]`; `.summary()`, `.to_dict()`, `gain_mean`, `changed_frac`, ... |
| `LLMParaphraser` | `(llm, *, prompt="corpus", temperatures=(0.7, 1.0), stop_when_enough=None, max_tokens=None, max_retries=5, backoff=1.0, max_backoff=60.0, on_error="raise", **llm_kw)` |
| `OpenAIParaphraser` | `(model="gpt-4o-mini", *, base_url=None, api_key=None, prompt=PROMPT_CORPUS, ...)`; `.k_variants(model, **kw)` |
| `CallableParaphraser` | `(fn, *, takes_n=None, on_error="raise")` |
| `ToyParaphraser` | `(seed=0, edits=2)` |
| `PROMPT_CORPUS`, `PROMPT_K_VARIANTS` | paraphrase prompts of the paper |

## Auditing

| Name | Signature |
|---|---|
| `AuditConfig` | frozen dataclass: `win_words=28, stride_words=14, sim_threshold=0.82, tie_margin=0.05, competitive=True, min_support=25, alpha=0.05, z_threshold=4.0, bucket_bits=96, bucket_seed=2025, direction_scheme="naive", reference_selection="all", query_embedder="sentence-transformers/all-MiniLM-L6-v2", query_sim_threshold=0.48, max_reference_docs=120, max_reference_docs_per_provider=0`; same helpers as `WatermarkConfig` |
| `AUDIT_PRESETS` | `paper`, `paper-query`, `small-corpus` |
| `Auditor.from_corpus` | `(corpus, keys, config=None, embedder=None, *, selector=None, text_field=("text_wm", "text"), provider_field=None, id_field="id") -> Auditor`; raises `ValueError` if the corpus records a different encoder dimension (`wm_dim`), warns on a different encoder name |
| `Auditor` | `(providers: list[ProviderReference], config=None, embedder=None, *, selector=None)` |
| `Auditor.audit` | `(ask, queries, *, stop_when_detected=True, max_queries=None, on_error="raise", progress=False, callback=None) -> AuditReport`; `stop_when_detected` is `True`, `False`, one provider id or a list of ids |
| `Auditor.add_answer` / `add_answers` | `(answer, query=None, *, sources=None) -> AnswerScore` / `(iterable, *, progress=False)` |
| `Auditor.report` / `reset` / `pooled_z` | current `AuditReport` / clear evidence / running Z |
| `AuditReport` | `n_answers, dim, config, providers: list[ProviderResult], trajectory: list[AuditStep]`; `.detected`, `.first_detected_at`, `[pid]`, `.to_dict()` |
| `ProviderResult` | `provider_id, support, sum_proj, mean_proj, z_raw, z, p_value, q_value, eligible, detected, first_crossed_at` |
| `AuditStep` | `step, query, answer, support, z, detected, error` |
| `AnswerScore` | `providers: dict[pid, ProviderAnswerScore], windows_total, assigned_total, dropped, best_cos_mean, second_cos_mean`; `.winner` |
| `ProviderAnswerScore` | `support, mean_proj, z`; `.sum_proj` (one answer, one provider) |
| `generate_audit_queries` | `(docs, llm="openai/gpt-4o-mini", n=None, *, seed=0, temperature=0.2, max_tokens=None, max_chars=1500, prompt=PROMPT_AUDIT_QUERY, max_workers=1, max_retries=5, text_field=("text", "text_wm"), id_field="id", provider_field=None, **llm_kw) -> list[str]` |
| `answer_text` | `(raw) -> str` for strings, dicts, chat messages, LlamaIndex responses |
| `ProviderReference` | `(provider_id, key_hex, texts)` |
| `QueryReferenceSelector` | `(documents, embedder=None, *, sim_threshold=0.48, max_docs=120, max_docs_per_provider=0, max_chars=1000, doc_embeddings=None)`; `.from_providers(...)`, `.select(query)` |
| `AlignedScorer` | `(keys, embedder=None, config=None, *, cache_size=64)`; `.score(answer, sources) -> AnswerScore` |

## LLM backends and encoders

| Name | Signature |
|---|---|
| `OpenAIChat` | `(model="gpt-4o-mini", *, base_url=None, api_key=None, api_key_env="OPENAI_API_KEY", timeout=120.0, client=None, extra_body=None, require_key=None)` |
| `AnthropicChat` | `(model, *, api_key=None, base_url=None, timeout=120.0, client=None, send_temperature=False, min_max_tokens=4096, **create_kwargs)` |
| `HFChat` | `(model, *, device="auto", dtype="auto", top_p=0.95, default_max_tokens=512, seed=None, model_kwargs=None, tokenizer=None, model_obj=None)` |
| `LiteLLMChat` | `(model, **completion_kwargs)` |
| `CallableChat`, `ToyChat` | `(fn)`, `(seed=0)` |
| `SentenceTransformerEmbedder` | `(model_name=DEFAULT_EMBED_MODEL, device="auto", batch_size=32)` |
| `OpenAIEmbedder` | `(model="text-embedding-3-small", *, dimensions=None, base_url=None, api_key=None, api_key_env="OPENAI_API_KEY", require_key=None, batch_size=256, max_retries=5, timeout=120.0, client=None)` |
| `CallableEmbedder` | `(fn, dim=None, *, batch_size=None)` |
| `HashingEmbedder` | `(dim=256, seed=0)` |

Errors: `LLMError` (base), `ParaphraseError`, `MissingDependencyError` (names the extra to
install), `MissingCredentialsError`, `UnknownProviderError`.

## Text, keys and statistics

`split_sentences`, `sentence_windows`, `text_windows`, `SimHashCoder`, `DirectionManager`,
`OrthoDirectionManager`, `save_provider_keys`, `load_provider_keys`, `pooled_z`,
`gate_by_min_support`, `bh_adjust`, `norm_sf`.
