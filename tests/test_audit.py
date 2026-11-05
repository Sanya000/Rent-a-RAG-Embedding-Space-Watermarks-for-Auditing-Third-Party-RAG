"""Aligned scoring, pooling and end-to-end detection."""

import random
import numpy as np
import pytest
from helpers import rotate, synthetic_corpus

from rentarag import (
    AlignedScorer,
    AuditConfig,
    Auditor,
    DirBucketWatermarker,
    HashingEmbedder,
    ProviderReference,
    QueryReferenceSelector,
    ToyParaphraser,
    split_sentences,
)


def _answers(docs, n, seed, target=0, other=None, keep=0.8):
    """Synthetic RAG answers: 3 fresh target sentences + 2 from another provider, lightly perturbed."""
    rng = random.Random(seed)
    pool = {pid: [s for d in ds for s in split_sentences(d)] for pid, ds in docs.items()}
    for sents in pool.values():
        rng.shuffle(sents)
    out = []
    for _ in range(n):
        o = other if other is not None else rng.choice([p for p in pool if p != target])
        sents = [pool[target].pop() for _ in range(3)] + [pool[o].pop() for _ in range(2)]
        rng.shuffle(sents)
        words = " ".join(sents).split()
        out.append(" ".join(w for w in words if rng.random() < keep or len(w) > 6))
    return out


@pytest.fixture(scope="module")
def plain_docs():
    """3 providers x 6 docs x 8 distinct random-word sentences (unwatermarked)."""
    return synthetic_corpus(3, 6, 8, seed=42)


@pytest.fixture(scope="module")
def wm_docs(seed13_keys, plain_docs):
    """Toy-watermarked ``plain_docs`` (fake 64-d embedder)."""
    emb = HashingEmbedder(64)
    return {
        pid: [DirBucketWatermarker(seed13_keys[pid], ToyParaphraser(seed=pid), emb).watermark(d).text for d in docs]
        for pid, docs in plain_docs.items()
    }


def _ref_sources(sources):
    return [(pid, "\n\n".join(docs)) for pid, docs in sources.items()]


CONFIGS = [
    dict(sim_threshold=0.82, competitive=True, tie_margin=0.05),
    dict(sim_threshold=0.5, competitive=True, tie_margin=0.05),
    dict(sim_threshold=0.6, competitive=True, tie_margin=None),
    dict(sim_threshold=0.6, competitive=False, tie_margin=None),
    dict(sim_threshold=0.3, competitive=True, tie_margin=0.2, win_words=8, stride_words=3),
]


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("scheme", ["naive", "ortho"])
def test_aligned_scoring_matches_reference(ref_fake, seed13_keys, wm_docs, fake_embedder, cfg, scheme):
    keys = {pid: seed13_keys[pid] for pid in wm_docs}
    win = dict(win_words=cfg.get("win_words", 28), stride_words=cfg.get("stride_words", 14))
    det = ref_fake.MultiDirBucketDetector(keys, direction_scheme=scheme)
    scorer = AlignedScorer(keys, fake_embedder, AuditConfig(direction_scheme=scheme, **cfg))
    subsets = [wm_docs, {0: wm_docs[0], 2: wm_docs[2][:1]}, {1: wm_docs[1]}]
    for k, answer in enumerate(_answers(wm_docs, 8, seed=1) + [""]):
        sources = subsets[k % len(subsets)]
        res_ref, meta = det.detect_windows_multi_aligned(
            answer,
            sources=_ref_sources(sources),
            sim_threshold=cfg["sim_threshold"],
            competitive=cfg["competitive"],
            tie_margin=cfg["tie_margin"],
            **win,
        )
        res = scorer.score(answer, sources)
        assert res.windows_total == meta["windows_total"]
        assert res.assigned_total == meta["assigned_total"]
        assert res.dropped == meta["dropped_below_thresh_windows"]
        assert res.best_cos_mean == meta["best_cos_stats"]["mean"]
        assert res.second_cos_mean == meta["second_best_cos_stats"]["mean"]
        for pid in keys:
            mine, theirs = res.providers[pid], res_ref[pid]
            assert (mine.support, mine.mean_proj, mine.z) == (theirs.T, theirs.mean_proj, theirs.z)


def test_pooled_auditor_matches_runner(ref_fake, seed13_keys, wm_docs, fake_embedder):
    runner = pytest.importorskip("run_dirbucket_quick_experiments")
    P = len(wm_docs)
    keys = [seed13_keys[p] for p in range(P)]
    cfg = AuditConfig(sim_threshold=0.6, min_support=10)
    det = ref_fake.MultiDirBucketDetector(dict(enumerate(keys)))
    auditor = Auditor([ProviderReference(p, keys[p], wm_docs[p]) for p in range(P)], cfg, fake_embedder)

    pooled_T = np.zeros(P, dtype=np.int64)
    pooled_S = np.zeros(P, dtype=np.float64)
    sources = [(p, "\n\n".join(wm_docs[p])) for p in range(P)]
    for answer in _answers(wm_docs, 8, seed=3, target=1):
        res_by_pid, _ = det.detect_windows_multi_aligned(
            answer, sources=sources, sim_threshold=0.6, competitive=True, tie_margin=0.05
        )
        for p in range(P):
            pooled_T[p] += int(res_by_pid[p].T)
            pooled_S[p] += float(res_by_pid[p].mean_proj) * int(res_by_pid[p].T)
        auditor.add_answer(answer)

    z_ref = runner.pooled_z(det, pooled_T, pooled_S)
    zg, pv, qv, sig, elig = runner.gate_significance_by_min_pooled_T(z_ref, pooled_T, min_pooled_T=10, alpha=0.05)
    rep = auditor.report()
    assert [r.support for r in rep.providers] == pooled_T.tolist()
    assert [r.sum_proj for r in rep.providers] == pooled_S.tolist()
    assert [r.z_raw for r in rep.providers] == z_ref
    assert [r.z for r in rep.providers] == zg
    assert [r.p_value for r in rep.providers] == pv
    assert [r.q_value for r in rep.providers] == qv
    assert [r.detected for r in rep.providers] == sig
    assert [r.eligible for r in rep.providers] == elig
    assert rep.detected == [1]


def test_end_to_end_detects_used_providers_only(seed13_keys, wm_docs):
    """Answers reuse providers 2 (3 sentences each) and 0 (2 each); provider 1 is never used."""
    emb = HashingEmbedder(64)
    keys = [seed13_keys[p] for p in sorted(wm_docs)]
    answers = _answers(wm_docs, 15, seed=7, target=2, other=0)

    auditor = Auditor([ProviderReference(p, keys[p], wm_docs[p]) for p in wm_docs], embedder=emb)
    for a in answers:
        auditor.add_answer(a)
    rep = auditor.report()
    assert rep.detected == [0, 2], str(rep)
    assert rep[2].support >= 25 and rep[2].z_raw > 4 and rep[2].first_crossed_at is not None
    assert rep[1].support == 0 and not rep[1].eligible

    wrong = Auditor([ProviderReference(p, k, wm_docs[p]) for p, k in zip(wm_docs, rotate(keys))], embedder=emb)
    for a in answers:
        wrong.add_answer(a)
    assert wrong.report().detected == []
    assert abs(wrong.report()[2].z_raw) < 3


def test_unwatermarked_text_not_detected(seed13_keys, plain_docs):
    emb = HashingEmbedder(64)
    keys = [seed13_keys[p] for p in sorted(plain_docs)]
    auditor = Auditor([ProviderReference(p, keys[p], plain_docs[p]) for p in plain_docs], embedder=emb)
    for a in _answers(plain_docs, 15, seed=7, target=2):
        auditor.add_answer(a)
    rep = auditor.report()
    assert rep.detected == []
    assert rep[2].support >= 25


def test_report_serialisation(seed13_keys, wm_docs):
    auditor = Auditor(
        [ProviderReference(p, seed13_keys[p], wm_docs[p]) for p in wm_docs], embedder=HashingEmbedder(64)
    )
    auditor.add_answer(_answers(wm_docs, 1, seed=0)[0])
    d = auditor.report().to_dict()
    assert d["n_answers"] == 1 and len(d["providers"]) == 3
    assert d["config"]["min_support"] == 25
    assert "answers audited: 1" in str(auditor.report())


def test_audit_defaults_are_paper_values():
    c = AuditConfig()
    assert (c.win_words, c.stride_words, c.sim_threshold, c.tie_margin, c.competitive) == (28, 14, 0.82, 0.05, True)
    assert (c.min_support, c.alpha, c.z_threshold, c.direction_scheme) == (25, 0.05, 4.0, "naive")


def test_unknown_provider_in_sources(seed13_keys, fake_embedder):
    scorer = AlignedScorer({0: seed13_keys[0]}, fake_embedder)
    with pytest.raises(ValueError):
        scorer.score("Some answer text here.", {7: ["doc"]})


def test_selector_requires_query(seed13_keys, wm_docs, fake_embedder):
    providers = [ProviderReference(p, seed13_keys[p], wm_docs[p]) for p in wm_docs]
    sel = QueryReferenceSelector.from_providers(providers, fake_embedder, sim_threshold=0.0)
    auditor = Auditor(providers, embedder=fake_embedder, selector=sel)
    with pytest.raises(ValueError):
        auditor.add_answer("x.")
    auditor.add_answer("The hospital triage system.", query="hospital triage")


@pytest.mark.parametrize("thr,max_docs,per_pid", [(0.0, 120, 0), (0.1, 3, 0), (0.05, 120, 1), (0.99, 120, 0)])
def test_query_selector_matches_runner(wm_docs, thr, max_docs, per_pid):
    runner = pytest.importorskip("run_dirbucket_quick_experiments")
    emb = HashingEmbedder(64)
    rows = [{"id": f"{p}-{i}", "provider_id": p, "text_wm": d} for p in wm_docs for i, d in enumerate(wm_docs[p])]
    X = emb([r["text_wm"][:1000] for r in rows])
    sel = QueryReferenceSelector([(r["provider_id"], r["text_wm"]) for r in rows], emb, sim_threshold=thr, max_docs=max_docs, max_docs_per_provider=per_pid)
    np.testing.assert_array_equal(sel.X, X)
    for query in ["new budget for the park", "seafood restaurant harbour", "school roof vegetables nutrition"]:
        qv = emb([query])[0]
        sources, _ = runner.build_alignment_sources(
            rows=rows,
            picked_idxs=[],
            prompt_docs=[],
            retrieved_docs=[],
            source_field="text_wm",
            aligned_source_mode="global_corpus",
            query_vec=qv,
            X=X,
            global_source_query_sim_threshold=thr,
            global_source_max_docs=max_docs,
            global_source_max_docs_per_provider=per_pid,
            global_source_selector="query",
        )
        chosen = sel.select(query)
        assert [(p, "\n\n".join(chosen[p])) for p in sorted(chosen)] == sources


def test_cache_does_not_change_scores(seed13_keys, wm_docs, fake_embedder):
    keys = {p: seed13_keys[p] for p in wm_docs}
    a = AlignedScorer(keys, fake_embedder, cache_size=0)
    b = AlignedScorer(keys, fake_embedder)
    for ans in _answers(wm_docs, 3, seed=11):
        ra, rb = a.score(ans, wm_docs), b.score(ans, wm_docs)
        assert ra == rb


def test_embedder_protocol():
    from rentarag import Embedder

    assert isinstance(HashingEmbedder(8), Embedder)


def test_bare_str_texts_rejected(seed13_keys, fake_embedder):
    doc = "The quick brown fox jumps over the lazy dog near the river bank today."
    with pytest.raises(TypeError):
        ProviderReference(0, seed13_keys[0], doc)
    scorer = AlignedScorer({0: seed13_keys[0]}, fake_embedder)
    with pytest.raises(TypeError):
        scorer.score("Some answer text here.", {0: doc})
    ref = ProviderReference(0, seed13_keys[0], [doc])
    ref.texts = doc  # mutated after construction
    with pytest.raises(TypeError):
        QueryReferenceSelector.from_providers([ref], fake_embedder)


def test_winner_none_without_alignment(seed13_keys, wm_docs, fake_embedder):
    providers = [ProviderReference(p, seed13_keys[p], wm_docs[p]) for p in wm_docs]
    auditor = Auditor(providers, AuditConfig(min_support=1), fake_embedder)
    assert auditor.add_answer("").winner is None
    unrelated = "zebra quantum " * 40
    sc = auditor.add_answer(unrelated)
    assert sc.assigned_total == 0 and sc.winner is None
    sc = auditor.add_answer(wm_docs[2][0])
    assert sc.assigned_total > 0 and sc.winner == 2
