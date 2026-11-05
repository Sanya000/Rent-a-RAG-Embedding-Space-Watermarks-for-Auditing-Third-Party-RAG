"""Embedder factory, adapters, normalisation and dimension handling."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import PROVIDER_DOCS

from rentarag import (
    AlignedScorer,
    CallableEmbedder,
    DirBucketWatermarker,
    HashingEmbedder,
    OpenAIEmbedder,
    ToyParaphraser,
    get_embedder,
)


def _unit_rows(X):
    return np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)


def test_hashing_specs():
    assert get_embedder("hashing").dim == 256
    assert get_embedder("hashing:64").dim == 64
    assert get_embedder("hashing/32").dim == 32
    e = HashingEmbedder(16)
    assert get_embedder(e) is e


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, "sentence-transformers/all-mpnet-base-v2"),
        ("default", "sentence-transformers/all-mpnet-base-v2"),
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ("BAAI/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
        ("st/intfloat/e5-base-v2", "intfloat/e5-base-v2"),
        ("hf/thenlper/gte-small", "thenlper/gte-small"),
        ("all-mpnet-base-v2", "all-mpnet-base-v2"),
    ],
)
def test_sentence_transformer_specs(monkeypatch, spec, expected):
    seen = {}

    class Stub:
        def __init__(self, name, **kw):
            seen["name"] = name

    monkeypatch.setattr("rentarag.embed.SentenceTransformerEmbedder", Stub)
    get_embedder(spec)
    assert seen["name"] == expected


def test_callable_embedder_normalises_and_probes_dim():
    calls = []

    def raw(texts):
        calls.append(len(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    emb = get_embedder(raw)
    assert isinstance(emb, CallableEmbedder) and emb.dim == 3
    X = emb(["a", "abcd", "xyz"])
    assert X.dtype == np.float32 and X.shape == (3, 3) and _unit_rows(X)
    assert emb([]).shape == (0, 3)
    batched = CallableEmbedder(raw, dim=3, batch_size=2)
    calls.clear()
    np.testing.assert_allclose(batched(["a", "abcd", "xyz"]), X)
    assert calls == [2, 1]


def test_callable_embedder_rejects_bad_shapes():
    emb = CallableEmbedder(lambda texts: np.ones((len(texts) + 1, 4)), dim=4)
    with pytest.raises(ValueError):
        emb(["a"])
    emb = CallableEmbedder(lambda texts: np.ones((len(texts), 5)), dim=4)
    with pytest.raises(ValueError):
        emb(["a"])


def test_zero_vector_stays_zero():
    emb = CallableEmbedder(lambda texts: np.zeros((len(texts), 4)), dim=4)
    assert not np.isnan(emb(["a"])).any()


class _FakeEmbeddingsClient:
    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input, **kw):
        self.calls.append((model, list(input), kw))
        rng = np.random.default_rng(len(self.calls))
        data = [SimpleNamespace(index=i, embedding=(rng.standard_normal(self.dim) * 3).tolist()) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))  # out of order on purpose


def test_openai_embedder_batches_and_normalises():
    client = _FakeEmbeddingsClient(dim=8)
    emb = OpenAIEmbedder("my-embedder", client=client, batch_size=2)
    assert emb.dim == 8 and len(client.calls) == 1  # dim probe for an unknown model
    X = emb(["a", "b", "c"])
    assert X.shape == (3, 8) and _unit_rows(X)
    assert [c[1] for c in client.calls[1:]] == [["a", "b"], ["c"]]
    known = OpenAIEmbedder("text-embedding-3-small", client=_FakeEmbeddingsClient(), dimensions=256)
    assert known.dim == 256
    known(["x"])
    assert known.client.calls[-1][2] == {"dimensions": 256}


def test_openai_embedder_spec(monkeypatch):
    created = []

    class FakeOpenAI(_FakeEmbeddingsClient):
        def __init__(self, **kw):
            super().__init__(dim=1536)
            created.append(kw)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    emb = get_embedder("openai/text-embedding-3-small")
    assert isinstance(emb, OpenAIEmbedder) and emb.dim == 1536 and created[-1]["api_key"] == "sk"
    get_embedder("ollama/nomic-embed-text")
    assert created[-1]["base_url"] == "http://localhost:11434/v1"


@pytest.mark.parametrize("dim", [16, 48, 384])
def test_directions_follow_embedder_dim(seed13_keys, dim):
    base = HashingEmbedder(dim)
    emb = get_embedder(lambda texts: base(texts) * 5.0)  # unnormalised custom encoder
    wm = DirBucketWatermarker(seed13_keys[0], ToyParaphraser(), emb)
    assert wm.coder.R.shape == (96, dim)
    res = wm.watermark(PROVIDER_DOCS[0][0])
    assert res.gain_mean > 0
    scorer = AlignedScorer({0: seed13_keys[0]}, emb)
    assert scorer.dim == dim and scorer.coder.R.shape[1] == dim
    assert scorer.score(res.text, {0: [res.text]}).providers[0].support > 0


def test_watermarker_accepts_specs(seed13_keys):
    wm = DirBucketWatermarker(seed13_keys[0], "toy", "hashing:64", "fast")
    assert wm.embedder.dim == 64 and wm.config.rounds == 3 and isinstance(wm.paraphraser, ToyParaphraser)
