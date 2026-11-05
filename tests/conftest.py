"""Shared fixtures.

The paper's frozen reference implementation lives in ``reproduce/experiments``;
tests import it from there and check that the library reproduces it exactly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rentarag import HashingEmbedder

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "reproduce" / "experiments"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_MODEL = "sentence-transformers/all-mpnet-base-v2"

if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

FAKE_DIM = 64


class FakeReferenceEmbedder:
    """Stand-in for ``watermark_dirbucket.Embedder`` backed by :class:`HashingEmbedder`."""

    dim = FAKE_DIM

    def __init__(self, model_name: str = "", device: str = "auto"):
        self._emb = HashingEmbedder(self.dim)
        self.st = SimpleNamespace(get_sentence_embedding_dimension=lambda: self.dim)

    def embed(self, texts):
        return self._emb(list(texts))


@pytest.fixture(scope="session")
def ref():
    """The frozen reference module ``watermark_dirbucket``."""
    import watermark_dirbucket

    return watermark_dirbucket


@pytest.fixture
def ref_fake(ref, monkeypatch):
    """Reference module with its encoder replaced by the fake hashing embedder."""
    monkeypatch.setattr(ref, "Embedder", FakeReferenceEmbedder)
    return ref


@pytest.fixture
def fake_embedder():
    return HashingEmbedder(FAKE_DIM)


@pytest.fixture(scope="session")
def seed13_keys() -> list[str]:
    return json.loads((FIXTURES / "provider_keys_seed13.json").read_text())["keys_hex"]


@pytest.fixture(scope="session")
def farad_sample() -> list[dict]:
    with (FIXTURES / "farad_dirbucket_naive_sample.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _real_model_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(REAL_MODEL, "config.json") is not None
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") == "1" and _real_model_available():
        return
    reason = "slow test: set RUN_SLOW=1 (needs all-mpnet-base-v2 in the HF cache)"
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=reason))

