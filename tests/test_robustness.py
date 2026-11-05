"""Custom field names end to end, input validation and clear error messages."""

import json
import warnings

import pytest
from helpers import synthetic_corpus

from rentarag import (
    AuditConfig,
    Auditor,
    HashingEmbedder,
    ProviderKeys,
    ToyParaphraser,
    WatermarkConfig,
    generate_audit_queries,
    load_corpus,
    split_sentences,
    watermark_corpus,
)
from rentarag.cli import main

EMB = HashingEmbedder(64)


class CountingParaphraser:
    def __init__(self):
        self.calls = 0
        self.toy = ToyParaphraser()

    def __call__(self, sentence, n):
        self.calls += 1
        return self.toy(sentence, n)


def _custom_rows(n_prov=2, n_docs=3):
    corpus = synthetic_corpus(n_prov, n_docs, 4, seed=5)
    return [
        {"doc_id": f"{p}-{i}", "owner": f"prov{p}", "body": d}
        for p, ds in corpus.items()
        for i, d in enumerate(ds)
    ]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_resume_with_custom_id_field(tmp_path):
    rows = _custom_rows()
    keys = ProviderKeys.generate(["prov0", "prov1"], seed=1)
    cache = tmp_path / "wm.jsonl"
    kw = dict(embedder=EMB, config="fast", cache_path=cache, id_field="doc_id", provider_field="owner", text_field="body")
    first = watermark_corpus(rows, keys, paraphraser=CountingParaphraser(), **kw)
    n_lines = len(cache.read_text().splitlines())
    para = CountingParaphraser()
    again = watermark_corpus(rows, keys, paraphraser=para, **kw)
    assert para.calls == 0
    assert len(cache.read_text().splitlines()) == n_lines
    assert [d.text for d in again] == [d.text for d in first]


def test_completed_resume_does_not_build_the_paraphraser(tmp_path):
    rows = _custom_rows(1, 2)
    keys = ProviderKeys.generate(["prov0"], seed=1)
    kw = dict(embedder=EMB, config="fast", cache_path=tmp_path / "wm.jsonl", id_field="doc_id", provider_field="owner", text_field="body")
    watermark_corpus(rows, keys, paraphraser="toy", **kw)
    # a spec that would fail to build (no credentials) is never built when nothing is left to do
    watermark_corpus(rows, keys, paraphraser="openai/gpt-4o-mini", **{**kw, "config": "fast"})


def test_queries_accept_field_names(tmp_path):
    rows = _custom_rows()
    qs = generate_audit_queries(rows, lambda prompt: "Question: what happened?", n=3, text_field="body")
    assert qs == ["what happened?"] * 3


def test_cli_custom_fields_round_trip(tmp_path):
    rows = _custom_rows()
    corpus, keys, wm, answers, report = (tmp_path / n for n in ("c.jsonl", "k.json", "w.jsonl", "a.jsonl", "r.json"))
    _write_jsonl(corpus, rows)
    main(["keygen", "--providers", "prov0,prov1", "--seed", "3", "-o", str(keys)])
    fields = ["--id-field", "doc_id", "--provider-field", "owner"]
    main(["watermark", "--keys", str(keys), "-i", str(corpus), "-o", str(wm), "--toy", "--embedder", "hashing",
          "--preset", "fast", "--text-field", "body", "-q", *fields])
    out = [json.loads(line) for line in wm.read_text().splitlines()]
    assert all(r["wm_dim"] == 256 and r["wm_embedder"] == "hashing" for r in out)
    sents = [s for r in out if r["owner"] == "prov1" for s in split_sentences(r["text_wm"])]
    _write_jsonl(answers, [{"answer": " ".join(sents[k : k + 3])} for k in range(0, len(sents), 3)] * 3)
    main(["audit", "--keys", str(keys), "--corpus", str(wm), "--answers", str(answers), "--embedder", "hashing",
          "--preset", "small-corpus", "--min-support", "5", "-o", str(report), *fields])
    assert json.loads(report.read_text())["detected"] == ["prov1"]
    q = tmp_path / "q.txt"
    main(["queries", "--corpus", str(corpus), "--llm", "toy", "-n", "2", "--text-field", "body", "-o", str(q), *fields])
    assert len(q.read_text().splitlines()) == 2


def test_cli_preset_keeps_config_overrides(tmp_path):
    from rentarag import load_config

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("watermark: {preset: paper, min_sem_sim: 0.75}\naudit: {preset: paper, sim_threshold: 0.8}\n")
    cfg = load_config(cfg_path)
    assert cfg.watermark_overrides == {"min_sem_sim": 0.75}
    assert WatermarkConfig.preset("fast", **cfg.watermark_overrides).min_sem_sim == 0.75
    assert AuditConfig.preset("small-corpus", **cfg.audit_overrides).sim_threshold == 0.8


def test_auditor_checks_the_encoder(tmp_path):
    rows = _custom_rows(1, 2)
    keys = ProviderKeys.generate(["prov0"], seed=1)
    marked = watermark_corpus(rows, keys, paraphraser="toy", embedder=EMB, config="fast",
                              provider_field="owner", text_field="body", id_field="doc_id")
    with pytest.raises(ValueError, match="64-dimensional"):
        Auditor.from_corpus(marked, keys, embedder=HashingEmbedder(32))
    path = tmp_path / "wm.jsonl"
    marked.save(path)
    with pytest.raises(ValueError, match="dimension 32"):
        Auditor.from_corpus(path, keys, embedder=HashingEmbedder(32), provider_field="owner", id_field="doc_id")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Auditor.from_corpus(path, keys, embedder=EMB, provider_field="owner", id_field="doc_id")

    class Named:
        dim = 64
        model_name = "other-model"

        def __call__(self, texts):
            return EMB(texts)

    with pytest.warns(UserWarning, match="other-model"):
        Auditor.from_corpus(marked, keys, embedder=Named())


@pytest.mark.parametrize(
    "kw",
    [{"min_sem_sim": 1.5}, {"min_sem_sim": -2}, {"min_delta_proj": -0.1}, {"mmr_lambda": 2}],
)
def test_watermark_config_validation(kw):
    with pytest.raises(ValueError, match=next(iter(kw))):
        WatermarkConfig(**kw)


@pytest.mark.parametrize(
    "kw",
    [{"sim_threshold": 2.0}, {"stride_words": 0}, {"stride_words": -3}, {"min_support": -1}, {"tie_margin": -0.1}],
)
def test_audit_config_validation(kw):
    with pytest.raises(ValueError, match=next(iter(kw))):
        AuditConfig(**kw)


def test_stop_when_detected_accepts_a_single_id():
    keys = ProviderKeys.generate(["prov0", "prov1"], seed=2)
    corpus = synthetic_corpus(2, 3, 6, seed=9)
    marked = watermark_corpus({f"prov{p}": ds for p, ds in corpus.items()}, keys, paraphraser="toy", embedder=EMB, config="fast")
    aud = Auditor.from_corpus(marked, keys, AuditConfig(min_support=3), EMB)
    sents = [s for d in marked.by_provider()["prov1"] for s in split_sentences(d)]
    answers = iter(" ".join(sents[k : k + 3]) for k in range(0, len(sents), 3))
    rep = aud.audit(lambda q: next(answers, ""), [f"q{k}" for k in range(40)], stop_when_detected="prov1")
    assert rep.detected == ["prov1"] and len(rep.trajectory) < 40
    with pytest.raises(ValueError, match="prov9"):
        aud.audit(lambda q: "", ["q"], stop_when_detected="prov9")


def test_key_file_formats(tmp_path):
    keys = ProviderKeys.generate(["acme", "globex"], seed=4)
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(dict(zip(keys.ids, keys.keys_hex))))
    assert ProviderKeys.load(flat).keys_hex == keys.keys_hex
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"acme": "not hex"}))
    with pytest.raises(ValueError, match="not a provider key file"):
        ProviderKeys.load(bad)
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"acme": "abcd"}))
    with pytest.raises(ValueError, match="at least 16 bytes"):
        ProviderKeys.load(short)


def test_bad_jsonl_line_is_located(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": 1, "provider": "a", "text": "x"}\n{not json}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2: invalid JSON"):
        load_corpus(p)


def test_cli_unreachable_rag_url_is_a_clean_error(tmp_path, capsys):
    rows = _custom_rows(1, 2)
    corpus, keys, q = tmp_path / "c.jsonl", tmp_path / "k.json", tmp_path / "q.txt"
    _write_jsonl(corpus, [{"id": r["doc_id"], "provider": r["owner"], "text": r["body"]} for r in rows])
    main(["keygen", "--providers", "prov0", "-o", str(keys)])
    q.write_text("what?\n")
    with pytest.raises(SystemExit) as e:
        main(["audit", "--keys", str(keys), "--corpus", str(corpus), "--queries", str(q), "--embedder", "hashing",
              "--rag-url", "http://127.0.0.1:9/ask", "--timeout", "5", "-q"])
    assert "cannot reach the RAG endpoint" in str(e.value)


def test_unknown_encoder_prefix_suggests_a_fix():
    from rentarag import get_embedder

    with pytest.raises(ValueError, match="Did you mean 'openai/text-embedding-3-small'"):
        get_embedder("openia/text-embedding-3-small")
