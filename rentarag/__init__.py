"""rentarag: DirBucket embedding-space watermarks for auditing third-party RAG.

Reference implementation of "Rent-a-RAG: Embedding-Space Watermarks for
Auditing Third-Party RAG" (EMNLP 2026).

Providers watermark their documents::

    keys = rentarag.ProviderKeys.generate(["acme", "globex"])
    marked = rentarag.watermark_corpus(docs, keys, paraphraser="openai/gpt-4o-mini")

and an auditor tests a black-box RAG system::

    auditor = rentarag.Auditor.from_corpus(marked, keys)
    report = auditor.audit(rag_chain.invoke, queries)

Any LLM (:func:`get_paraphraser`, :func:`get_llm`), encoder
(:func:`get_embedder`), corpus format (:func:`load_corpus`), number of
providers and parameter set (:class:`WatermarkConfig`, :class:`AuditConfig`,
:func:`load_config`) plugs in.
"""

from .audit import (
    AUDIT_PRESETS,
    AlignedScorer,
    AnswerScore,
    AuditConfig,
    Auditor,
    AuditReport,
    AuditStep,
    ProviderAnswerScore,
    ProviderReference,
    ProviderResult,
    QueryReferenceSelector,
    answer_text,
)
from .config import RentaragConfig, load_config, write_example_config
from .corpus import Document, WatermarkedCorpus, WatermarkedDoc, iter_documents, load_corpus, watermark_corpus
from .embed import (
    DEFAULT_EMBED_MODEL,
    CallableEmbedder,
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from .keys import (
    DirectionManager,
    OrthoDirectionManager,
    ProviderKeys,
    SimHashCoder,
    UnknownProviderError,
    generate_provider_keys,
    load_provider_keys,
    save_provider_keys,
)
from .llm import (
    AnthropicChat,
    CallableChat,
    ChatModel,
    HFChat,
    LiteLLMChat,
    LLMError,
    MissingCredentialsError,
    MissingDependencyError,
    OpenAIChat,
    ToyChat,
    get_llm,
)
from .paraphrase import (
    PROMPT_CORPUS,
    PROMPT_K_VARIANTS,
    CallableParaphraser,
    LLMParaphraser,
    OpenAIParaphraser,
    Paraphraser,
    ParaphraseError,
    ToyParaphraser,
    get_paraphraser,
)
from .queries import PROMPT_AUDIT_QUERY, generate_audit_queries
from .stats import bh_adjust, gate_by_min_support, norm_sf, pooled_z
from .text import sentence_windows, split_sentences, text_windows
from .watermark import WATERMARK_PRESETS, DirBucketWatermarker, SentenceLog, WatermarkConfig, WatermarkResult

__version__ = "0.1.0"

__all__ = [
    "AUDIT_PRESETS", "AlignedScorer", "AnswerScore", "AnthropicChat", "AuditConfig", "AuditReport", "AuditStep",
    "Auditor", "CallableChat", "CallableEmbedder", "CallableParaphraser", "ChatModel", "DEFAULT_EMBED_MODEL",
    "DirBucketWatermarker", "DirectionManager", "Document", "Embedder", "HFChat", "HashingEmbedder", "LLMError",
    "LLMParaphraser", "LiteLLMChat", "MissingCredentialsError", "MissingDependencyError", "OpenAIChat",
    "OpenAIEmbedder", "OpenAIParaphraser", "OrthoDirectionManager", "PROMPT_AUDIT_QUERY", "PROMPT_CORPUS",
    "PROMPT_K_VARIANTS", "ParaphraseError", "Paraphraser", "ProviderAnswerScore", "ProviderKeys",
    "ProviderReference", "ProviderResult", "QueryReferenceSelector", "RentaragConfig", "SentenceLog",
    "SentenceTransformerEmbedder", "SimHashCoder", "ToyChat", "ToyParaphraser", "UnknownProviderError",
    "WATERMARK_PRESETS", "WatermarkConfig", "WatermarkResult", "WatermarkedCorpus", "WatermarkedDoc",
    "answer_text", "bh_adjust", "gate_by_min_support", "generate_audit_queries", "generate_provider_keys",
    "get_embedder", "get_llm", "get_paraphraser", "iter_documents", "load_config", "load_corpus",
    "load_provider_keys", "norm_sf", "pooled_z", "save_provider_keys", "sentence_windows", "split_sentences",
    "text_windows", "watermark_corpus", "write_example_config",
]
