"""Text segmentation, SimHash, keyed directions and key generation."""

import hashlib

import numpy as np
import pytest
from helpers import PROVIDER_DOCS, unit

from rentarag import (
    DirectionManager,
    OrthoDirectionManager,
    SimHashCoder,
    generate_provider_keys,
    load_provider_keys,
    save_provider_keys,
    sentence_windows,
    split_sentences,
    text_windows,
)

TEXTS = [
    "",
    "   ",
    "One sentence without a final stop",
    "Dr. Smith arrived at 5 p.m. on Monday. He said: no. It was e.g. a test? Yes! Short. Another rather long sentence follows here.",
    "Prices rose 3.5% in Q1.\n\nThe U.S. office said so. Ok.",
    " ".join(["word"] * 70) + ". " + " ".join(f"w{i}" for i in range(29)) + ".",
    *[d for docs in PROVIDER_DOCS.values() for d in docs],
]


@pytest.mark.parametrize("text", TEXTS)
def test_split_sentences_matches_reference(ref, text):
    assert split_sentences(text) == ref.split_sentences(text)


@pytest.mark.parametrize("n_words", [0, 1, 14, 27, 28, 29, 41, 42, 43, 56, 100])
@pytest.mark.parametrize("win,stride", [(28, 14), (12, 6), (10, 0), (5, 7)])
def test_sentence_windows_match_reference(ref, n_words, win, stride):
    sent = " ".join(f"t{i}" for i in range(n_words))
    assert sentence_windows(sent, win, stride) == ref._windows_within_sentence(sent, win, stride)


def test_text_windows_cover_every_sentence():
    text = PROVIDER_DOCS[0][0]
    wins = text_windows(text)
    assert len(wins) >= len(split_sentences(text))
    assert all(len(w.split()) <= 28 for w in wins)


def test_simhash_hyperplanes_are_the_paper_ones(ref):
    c = SimHashCoder(768)
    assert (c.bits, c.seed) == (96, 2025)
    assert hashlib.sha256(c.R.tobytes()).hexdigest() == "760d2bee0c34da0e5eed4fdba109d2a234f2e94cc6eb66aea7c722227ec82865"
    np.testing.assert_array_equal(c.R, ref.SimHashCoder(768, 96, 2025).R)


@pytest.mark.parametrize("dim,bits", [(768, 96), (384, 96), (64, 13), (32, 8)])
def test_simhash_codes_match_reference(ref, dim, bits):
    rng = np.random.default_rng(dim + bits)
    mine, theirs = SimHashCoder(dim, bits, 7), ref.SimHashCoder(dim, bits, 7)
    for _ in range(50):
        u = unit(rng, dim)
        assert mine.code(u) == theirs.code(u)
    assert len(mine.code(u)) == 2 * ((bits + 7) // 8)


def test_direction_golden_vector(seed13_keys):
    u = np.random.RandomState(0).normal(size=768).astype(np.float32)
    u /= np.linalg.norm(u)
    code = SimHashCoder(768).code(u)
    assert code == b"23af876cc196654952698880"
    v = DirectionManager(seed13_keys[0], 768).direction(code)
    assert v.dtype == np.float32 and v.shape == (768,)
    np.testing.assert_allclose(
        v[:4], [-0.034589920192956924, -0.06867329776287079, -0.0008871094323694706, -0.05305555462837219], rtol=1e-6
    )
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_directions_match_reference(ref, seed13_keys):
    rng = np.random.default_rng(1)
    coder = SimHashCoder(768)
    for key in seed13_keys:
        mine, theirs = DirectionManager(key, 768), ref.DirectionManager(key, 768)
        for _ in range(5):
            code = coder.code(unit(rng, 768))
            np.testing.assert_array_equal(mine.direction(code), theirs.direction(code))


def test_ortho_directions_match_reference(ref, seed13_keys):
    keys = dict(enumerate(seed13_keys))
    mine, theirs = OrthoDirectionManager(keys, 768), ref.OrthoDirectionManager(keys, 768)
    coder = SimHashCoder(768)
    rng = np.random.default_rng(2)
    for _ in range(5):
        code = coder.code(unit(rng, 768))
        V = np.stack([mine.direction_for(p, code) for p in keys])
        for p in keys:
            np.testing.assert_array_equal(mine.for_provider(p).direction(code), theirs.direction(p, code))
        np.testing.assert_allclose(V @ V.T, np.eye(len(keys)), atol=1e-5)


def test_keygen_seed13_reproduces_paper_keys(seed13_keys):
    assert generate_provider_keys(6, seed=13) == seed13_keys


def test_keygen_secret_by_default():
    a, b = generate_provider_keys(3), generate_provider_keys(3)
    assert a != b and all(len(k) == 64 for k in a)
    bytes.fromhex(a[0])


def test_key_file_roundtrip(tmp_path, seed13_keys):
    p = tmp_path / "keys.json"
    save_provider_keys(p, seed13_keys, seed=13)
    assert load_provider_keys(p) == seed13_keys
