"""Auditor.from_corpus, add_answers, live audits through ask(query) and audit-query generation."""

import json
import random
from types import SimpleNamespace

import pytest
from helpers import synthetic_corpus

from rentarag import (
    AuditConfig,
    Auditor,
    HashingEmbedder,
    ProviderKeys,
    QueryReferenceSelector,
    answer_text,
    generate_audit_queries,
    split_sentences,
    watermark_corpus,
)
from rentarag.queries import PROMPT_AUDIT_QUERY

EMB = HashingEmbedder(64)
NAMES = ["acme", "globex", "initech", "umbrella"]


@pytest.fixture(scope="module")
def setup():
    keys = ProviderKeys.generate(NAMES, seed=7)
    corpus = synthetic_corpus(len(NAMES), 6, 8, seed=11)
    docs = [{"id": f"{n}-{i}", "provider": n, "text": t} for p, n in enumerate(NAMES) for i, t in enumerate(corpus[p])]
    marked = watermark_corpus(docs, keys, paraphraser="toy", embedder=EMB)
    return keys, marked


class ToyRAG:
    """Answers each query with three fresh sentences from the indexed providers' documents."""

    def __init__(self, marked, indexed, seed=0):
        rng = random.Random(seed)
        self.pool = [s for d in marked if d.provider in indexed for s in split_sentences(d.text)]
        rng.shuffle(self.pool)
        self.calls = 0

    def __call__(self, query):
        k = self.calls
        self.calls += 1
        return " ".join(self.pool[3 * k : 3 * k + 3])


def test_from_corpus_accepts_many_shapes(setup):
    keys, marked = setup
    a = Auditor.from_corpus(marked, keys, embedder=EMB)
    assert [p.provider_id for p in a.providers] == NAMES
    rows = marked.rows()
    b = Auditor.from_corpus(rows, keys, embedder=EMB)
    c = Auditor.from_corpus(marked.by_provider(), keys, embedder=EMB)
    for other in (b, c):
        assert {p.provider_id: list(p.texts) for p in other.providers} == {p.provider_id: list(p.texts) for p in a.providers}


def test_provider_without_documents_is_still_tested(setup):
    keys, marked = setup
    a = Auditor.from_corpus([d for d in marked if d.provider != "umbrella"], keys, embedder=EMB)
    assert a.report()["umbrella"].support == 0


def test_live_audit_stops_at_detection(setup):
    keys, marked = setup
    auditor = Auditor.from_corpus(marked, keys, embedder=EMB)
    rag = ToyRAG(marked, {"globex"})
    report = auditor.audit(rag, [f"q{i}" for i in range(50)])
    assert report.detected == ["globex"]
    n = len(report.trajectory)
    assert rag.calls == n < 50 and report.n_answers == n
    assert report.first_detected_at == {"globex": n}
    steps = report.trajectory
    assert [s.step for s in steps] == list(range(1, n + 1))
    assert steps[-1].support["globex"] >= 25 and steps[-1].detected == ["globex"]
    assert all(not s.detected for s in steps[:-1])
    json.dumps(report.to_dict())  # serialisable, trajectory included
    assert len(report.to_dict()["trajectory"]) == n


def test_live_audit_targets_budget_and_no_stop(setup):
    keys, marked = setup
    a = Auditor.from_corpus(marked, keys, embedder=EMB)
    rep = a.audit(ToyRAG(marked, {"acme", "initech"}), [f"q{i}" for i in range(40)], stop_when_detected=["acme", "initech"])
    assert {"acme", "initech"} <= set(rep.detected) and "globex" not in rep.detected
    b = Auditor.from_corpus(marked, keys, embedder=EMB)
    rep = b.audit(ToyRAG(marked, {"acme"}), [f"q{i}" for i in range(40)], stop_when_detected=False, max_queries=12)
    assert len(rep.trajectory) == 12
    with pytest.raises(ValueError):
        b.audit(lambda q: "", ["q"], stop_when_detected=["nobody"])


def test_live_audit_error_policy(setup):
    keys, marked = setup
    a = Auditor.from_corpus(marked, keys, embedder=EMB)

    def flaky(q):
        if q == "bad":
            raise ConnectionError("down")
        return "some answer text"

    with pytest.raises(ConnectionError):
        a.audit(flaky, ["ok", "bad"])
    a.reset()
    rep = a.audit(flaky, ["ok", "bad", "ok"], on_error="skip", stop_when_detected=False)
    assert [s.error is not None for s in rep.trajectory] == [False, True, False] and rep.n_answers == 2


def test_answer_text_adapters():
    assert answer_text("x") == "x"
    assert answer_text({"answer": "a", "sources": []}) == "a"
    assert answer_text({"result": "r"}) == "r"
    assert answer_text({"output": {"content": "nested"}}) == "nested"
    assert answer_text(SimpleNamespace(content="msg")) == "msg"  # chat message objects
    assert answer_text(SimpleNamespace(response="li")) == "li"  # LlamaIndex Response
    assert answer_text(None) == ""


def test_add_answers_forms(setup):
    keys, marked = setup
    sents = [s for d in marked if d.provider == "acme" for s in split_sentences(d.text)]
    a = Auditor.from_corpus(marked, keys, embedder=EMB)
    res = a.add_answers([sents[0], ("q", sents[1]), {"answer": sents[2], "query": "q"}, SimpleNamespace(content=sents[3])])
    assert len(res) == 4 and a.n_answers == 4 and all(r.winner == "acme" for r in res)


def test_query_reference_selection_from_config(setup):
    keys, marked = setup
    cfg = AuditConfig(reference_selection="query", query_embedder="hashing:64", query_sim_threshold=0.0)
    a = Auditor.from_corpus(marked, keys, cfg, EMB)
    assert isinstance(a.selector, QueryReferenceSelector) and a.selector.sim_threshold == 0.0
    with pytest.raises(ValueError, match="query"):
        a.add_answer("text")
    rep = a.audit(ToyRAG(marked, {"initech"}), [f"what about item {i}" for i in range(40)])
    assert rep.detected == ["initech"]


def test_generate_queries_with_paper_prompt(setup):
    _, marked = setup
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return 'Question: "Which council approved the plan?"\nextra line'

    qs = generate_audit_queries(marked, llm, n=5, seed=1)
    assert qs == ["Which council approved the plan?"] * 5
    originals = {d.original[:1500] for d in marked}
    for p in prompts:
        doc = p.split("Document:\n", 1)[1].rsplit("\n\nQuestion:", 1)[0]
        assert doc in originals
        assert p == PROMPT_AUDIT_QUERY.format(doc=doc)
    assert len(set(prompts)) == 5  # sampled without replacement


def test_generate_queries_sampling_and_parallel():
    docs = [f"Document number {i} talks about topic {i}." for i in range(4)]
    seq = generate_audit_queries(docs, "toy", n=10, seed=3)
    par = generate_audit_queries(docs, "toy", n=10, seed=3, max_workers=4)
    assert seq == par and len(seq) == 10
    assert generate_audit_queries(docs, "toy") == [generate_audit_queries([d], "toy")[0] for d in docs]
    temps = []
    generate_audit_queries(docs[:1], lambda p, temperature=None: temps.append(temperature) or "Q?")
    assert temps == [0.2]
