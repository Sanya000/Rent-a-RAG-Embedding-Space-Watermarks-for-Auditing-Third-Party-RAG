"""Equivalence with the reference implementation under the paper's real encoder.

Skipped unless ``RUN_SLOW=1`` and ``all-mpnet-base-v2`` is in the local HF cache.
The fixture documents are four watermarked FARAD articles from the paper's
primary corpus (providers 0-3, seed-13 research keys).
"""

import numpy as np
import pytest
from helpers import PROVIDER_DOCS

from rentarag import (
    AlignedScorer,
    AuditConfig,
    Auditor,
    DirBucketWatermarker,
    ProviderReference,
    SentenceTransformerEmbedder,
    ToyParaphraser,
    WatermarkConfig,
    split_sentences,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def mpnet():
    return SentenceTransformerEmbedder()


@pytest.fixture(scope="module")
def sample_sources(farad_sample):
    return {r["provider_id"]: [r["text_wm"]] for r in farad_sample}


def test_embedder_matches_reference(ref, mpnet, farad_sample):
    texts = split_sentences(farad_sample[0]["text_wm"])[:6]
    np.testing.assert_array_equal(mpnet(texts), ref.Embedder().embed(texts))
    assert mpnet.dim == 768


def test_watermark_matches_reference_real_encoder(ref, monkeypatch, mpnet, seed13_keys):
    text = " ".join(PROVIDER_DOCS[0][:2])
    para = ToyParaphraser(seed=3)
    monkeypatch.setattr(ref, "paraphrase_diverse_batch", lambda s, model="gpt-4o-mini", n_total=16, **_: para(s, n_total))
    params = ref.GenParams(bucket_bits=96, bucket_seed=2025, min_sem_sim=0.70, min_delta_proj=5e-4, rounds=2, k_batch=8)
    ref_text, ref_summary = ref.DirBucketWatermarker(seed13_keys[0], params).watermark(text)
    cfg = WatermarkConfig(n_candidates=8, rounds=2)
    res = DirBucketWatermarker(seed13_keys[0], ToyParaphraser(seed=3), mpnet, cfg).watermark(text)
    assert res.text == ref_text
    assert res.summary()["gain_mean"] == ref_summary.gain_mean
    assert res.summary()["changed_sentences"] == ref_summary.changed_sentences


def _answer_mix(farad_sample) -> list[str]:
    """Two answers that copy sentences from several fixture documents."""
    s = [split_sentences(r["text_wm"]) for r in farad_sample]
    return [" ".join(s[0][:4] + s[1][:2]), " ".join(s[2][3:7] + s[3][:3])]


def test_alignment_matches_reference_real_encoder(ref, mpnet, seed13_keys, farad_sample, sample_sources):
    keys = {pid: seed13_keys[pid] for pid in sorted(sample_sources)}
    det = ref.MultiDirBucketDetector(keys)
    scorer = AlignedScorer(keys, mpnet, AuditConfig())
    sources = [(pid, "\n\n".join(docs)) for pid, docs in sample_sources.items()]
    for answer in _answer_mix(farad_sample):
        res_ref, meta = det.detect_windows_multi_aligned(
            answer, sources=sources, sim_threshold=0.82, competitive=True, tie_margin=0.05
        )
        res = scorer.score(answer, sample_sources)
        assert res.assigned_total == meta["assigned_total"] > 0
        for pid in keys:
            mine, theirs = res.providers[pid], res_ref[pid]
            assert (mine.support, mine.mean_proj, mine.z) == (theirs.T, theirs.mean_proj, theirs.z)


def test_paper_corpus_documents_are_detected(mpnet, seed13_keys, farad_sample, sample_sources):
    """Pooled over the four documents, every owner is detected with its key and not with a wrong one."""
    pids = sorted(sample_sources)
    for keys, expect in [(seed13_keys, True), (seed13_keys[1:] + seed13_keys[:1], False)]:
        auditor = Auditor(
            [ProviderReference(p, keys[p], sample_sources[p]) for p in pids],
            AuditConfig(min_support=10),
            mpnet,
        )
        for r in farad_sample:
            auditor.add_answer(r["text_wm"])
        rep = auditor.report()
        assert all(rep[p].support >= 10 for p in pids)
        assert (rep.detected == pids) is expect, str(rep)


def test_hf_backend_real_model():
    """Local transformers paraphraser and query generator (Qwen2.5-1.5B-Instruct, if cached)."""
    from huggingface_hub import try_to_load_from_cache

    from rentarag import generate_audit_queries, get_paraphraser

    model = "Qwen/Qwen2.5-1.5B-Instruct"
    if try_to_load_from_cache(model, "config.json") is None:
        pytest.skip(f"{model} not in the HF cache")
    para = get_paraphraser(f"hf/{model}", device="cpu", seed=0, max_tokens=200, temperatures=(0.7,))
    sentence = "The harbour authority approved a new ferry terminal on Monday after a long public debate."
    out = para(sentence, 4)
    assert 1 <= len(out) <= 4 and sentence not in out
    q = generate_audit_queries([sentence], para.llm, max_tokens=60)
    assert len(q) == 1 and q[0]
