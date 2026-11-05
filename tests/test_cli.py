"""Offline round trip through the command-line interface."""

import json
import random

import pytest

from helpers import synthetic_corpus

from rentarag import split_sentences
from rentarag.cli import main


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_keygen_watermark_audit(tmp_path, seed13_keys):
    keys, docs, wm, answers, report = (tmp_path / n for n in ("k.json", "d.jsonl", "w.jsonl", "a.jsonl", "r.json"))
    _write_jsonl(
        docs,
        [{"id": f"{p}-{i}", "provider_id": p, "text": d} for p, ds in synthetic_corpus(3, 6, 8, seed=42).items() for i, d in enumerate(ds)],
    )
    main(["keygen", "--providers", "3", "--seed", "13", "-o", str(keys)])
    assert json.loads(keys.read_text())["keys_hex"] == seed13_keys[:3]

    main(["watermark", "--keys", str(keys), "-i", str(docs), "-o", str(wm), "--toy", "--embed-model", "hashing"])
    rows = _read_jsonl(wm)
    assert len(rows) == 18 and all(r["text_wm"] != r["text"] for r in rows)

    main(["watermark", "--keys", str(keys), "-i", str(docs), "-o", str(wm), "--toy", "--embed-model", "hashing", "--resume"])
    assert len(_read_jsonl(wm)) == 18

    rng = random.Random(0)
    pool = {p: [s for r in rows if r["provider_id"] == p for s in split_sentences(r["text_wm"])] for p in (0, 2)}
    for s in pool.values():
        rng.shuffle(s)
    _write_jsonl(answers, [{"answer": " ".join(pool[2][3 * k : 3 * k + 3] + pool[0][2 * k : 2 * k + 2])} for k in range(15)])
    main(["audit", "--keys", str(keys), "--corpus", str(wm), "--answers", str(answers), "--embed-model", "hashing", "-o", str(report)])
    rep = json.loads(report.read_text())
    assert rep["detected"] == [0, 2]
    assert rep["providers"][1]["support"] == 0


def _toy_setup(tmp_path, n_keys=2):
    keys, docs = tmp_path / "k.json", tmp_path / "d.jsonl"
    main(["keygen", "--providers", str(n_keys), "--seed", "13", "-o", str(keys)])
    rows = [{"id": f"0-{i}", "provider_id": 0, "text": d} for i, d in enumerate(synthetic_corpus(1, 2, 4, seed=1)[0])]
    _write_jsonl(docs, rows)
    return keys, docs


def _wm(keys, inp, out, *extra):
    return ["watermark", "--keys", str(keys), "-i", str(inp), "-o", str(out), "--toy", "--embed-model", "hashing", *extra]


def test_watermark_same_input_output_refused(tmp_path):
    keys, docs = _toy_setup(tmp_path)
    before = docs.read_bytes()
    with pytest.raises(SystemExit) as e:
        main(_wm(keys, docs, docs))
    assert "different files" in str(e.value.code)
    assert docs.read_bytes() == before


@pytest.mark.parametrize("pid", [-1, 2, 7])
def test_watermark_provider_id_out_of_range(tmp_path, pid):
    keys, docs = _toy_setup(tmp_path, n_keys=2)
    _write_jsonl(docs, [{"id": "x", "provider_id": pid, "text": "A short sentence about the harbour market today."}])
    with pytest.raises(SystemExit) as e:
        main(_wm(keys, docs, tmp_path / "w.jsonl"))
    assert "out of range" in str(e.value.code)


def test_audit_provider_id_out_of_range(tmp_path):
    keys, _ = _toy_setup(tmp_path, n_keys=2)
    corpus, answers = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
    _write_jsonl(corpus, [{"provider_id": 5, "text_wm": "Some document text."}])
    _write_jsonl(answers, [{"answer": "Some answer."}])
    with pytest.raises(SystemExit) as e:
        main(["audit", "--keys", str(keys), "--corpus", str(corpus), "--answers", str(answers), "--embed-model", "hashing"])
    assert "out of range" in str(e.value.code)


def test_resume_tolerates_partial_last_line(tmp_path):
    keys, docs = _toy_setup(tmp_path)
    wm = tmp_path / "w.jsonl"
    main(_wm(keys, docs, wm))
    full = _read_jsonl(wm)
    lines = wm.read_text(encoding="utf-8").splitlines(keepends=True)
    wm.write_text(lines[0] + lines[1][:30], encoding="utf-8")  # interrupted mid-write
    main(_wm(keys, docs, wm, "--resume"))
    resumed = _read_jsonl(wm)
    assert resumed[0] == full[0]
    assert [r["id"] for r in resumed] == [r["id"] for r in full]
    assert all("text_wm" in r for r in resumed)


def test_clean_errors_instead_of_tracebacks(tmp_path, monkeypatch):
    keys, docs = _toy_setup(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(_wm(tmp_path / "missing.json", docs, tmp_path / "w.jsonl"))
    assert "error" in str(e.value.code) and "missing.json" in str(e.value.code)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    pytest.importorskip("openai")
    with pytest.raises(SystemExit) as e:
        main(["watermark", "--keys", str(keys), "-i", str(docs), "-o", str(tmp_path / "w2.jsonl"), "--embed-model", "hashing"])
    assert "OPENAI_API_KEY" in str(e.value.code)
    with pytest.raises(SystemExit) as e:
        main(_wm(keys, docs, tmp_path / "w3.jsonl", "--text-field", "nope"))
    assert "nope" in str(e.value.code)


@pytest.mark.parametrize("val,expected", [("none", None), ("OFF", None), ("0.1", 0.1)])
def test_tie_margin_accepts_none(val, expected):
    from rentarag.cli import build_parser

    args = build_parser().parse_args(["audit", "--keys", "k", "--corpus", "c", "--answers", "a", "--tie-margin", val])
    assert args.tie_margin == expected


# ---------------------------------------------------------------------------
# config file, named providers, queries and live audits
# ---------------------------------------------------------------------------

_RAG_SCRIPT = """
import hashlib, json, sys
query = sys.stdin.read()
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
sents = [s.strip() + "." for r in rows if r["provider"] == sys.argv[2] for s in r["text_wm"].split(". ") if s.strip()]
k = int(hashlib.sha256(query.encode()).hexdigest(), 16)
print(" ".join(sents[(k + 3 * i) % len(sents)] for i in range(3)))
"""


def _named_setup(tmp_path):
    import yaml

    from rentarag import write_example_config

    cfg_path = write_example_config(tmp_path / "rentarag.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(
        keys="keys.json",
        paraphraser={"spec": "toy"},
        embedder={"spec": "hashing:64"},
        workers=2,
        queries={"llm": "toy", "n": 30},
    )
    cfg["watermark"] = {"preset": "fast"}
    cfg["audit"] = {"preset": "small-corpus"}
    cfg["corpus"] = {"id_field": "doc", "provider_field": "owner", "text_field": "body", "out_field": "text_wm"}
    cfg_path.write_text(yaml.safe_dump(cfg))
    main(["keygen", "--providers", "acme,globex,initech", "-o", str(tmp_path / "keys.json")])
    corpus = synthetic_corpus(3, 6, 8, seed=42)
    names = ["acme", "globex", "initech"]
    csv_path = tmp_path / "docs.csv"
    lines = ["doc,owner,body"] + [f"{n}-{i},{n},{d}" for p, n in enumerate(names) for i, d in enumerate(corpus[p])]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg_path, csv_path


def test_config_driven_round_trip(tmp_path, capsys):
    cfg_path, csv_path = _named_setup(tmp_path)
    keys = json.loads((tmp_path / "keys.json").read_text())
    assert keys["provider_names"] == ["acme", "globex", "initech"] and len(keys["keys_hex"]) == 3
    wm = tmp_path / "wm.jsonl"
    main(["watermark", "--config", str(cfg_path), "-i", str(csv_path), "-o", str(wm), "-q"])
    rows = _read_jsonl(wm)
    assert [r["doc"] for r in rows] == [f"{n}-{i}" for n in ("acme", "globex", "initech") for i in range(6)]
    assert all(r["text_wm"] != r["body"] and r["owner"] in ("acme", "globex", "initech") for r in rows)

    rng = random.Random(1)
    pool = [s for r in rows if r["owner"] == "initech" for s in split_sentences(r["text_wm"])]
    rng.shuffle(pool)
    answers = tmp_path / "answers.jsonl"
    _write_jsonl(answers, [{"answer": " ".join(pool[3 * k : 3 * k + 3])} for k in range(12)])
    report = tmp_path / "report.json"
    main(["audit", "--config", str(cfg_path), "--corpus", str(wm), "--answers", str(answers), "-o", str(report), "-q"])
    rep = json.loads(report.read_text())
    assert rep["detected"] == ["initech"] and rep["config"]["min_support"] == 10
    assert [p["provider_id"] for p in rep["providers"]] == ["acme", "globex", "initech"]


def test_cli_flags_override_config(tmp_path):
    cfg_path, csv_path = _named_setup(tmp_path)
    wm = tmp_path / "wm.jsonl"
    main(["watermark", "--config", str(cfg_path), "-i", str(csv_path), "-o", str(wm), "-q", "--rounds", "1", "--max-sentences", "2"])
    rows = _read_jsonl(wm)
    assert all(r["wm_n_sentences"] == 2 for r in rows)
    with pytest.raises(SystemExit) as e:
        main(["watermark", "--config", str(cfg_path), "-i", str(csv_path), "-o", str(wm), "--paraphraser", "nonsense/x"])
    assert "unknown LLM spec" in str(e.value.code)


def test_queries_and_live_audit_with_rag_cmd(tmp_path):
    import sys

    cfg_path, csv_path = _named_setup(tmp_path)
    wm = tmp_path / "wm.jsonl"
    main(["watermark", "--config", str(cfg_path), "-i", str(csv_path), "-o", str(wm), "-q"])
    # the toy RAG script reads provider names from a "provider" field
    rag_corpus = tmp_path / "rag_corpus.jsonl"
    _write_jsonl(rag_corpus, [{"provider": r["owner"], "text_wm": r["text_wm"]} for r in _read_jsonl(wm)])
    queries = tmp_path / "queries.txt"
    main(["queries", "--config", str(cfg_path), "--corpus", str(csv_path), "-o", str(queries)])
    qs = queries.read_text().splitlines()
    assert len(qs) == 30 and all(q.endswith("?") for q in qs)

    script = tmp_path / "rag.py"
    script.write_text(_RAG_SCRIPT)
    report = tmp_path / "live.json"
    cmd = f'"{sys.executable}" "{script}" "{rag_corpus}" globex'
    main(["audit", "--config", str(cfg_path), "--corpus", str(wm), "--queries", str(queries), "--rag-cmd", cmd, "-o", str(report), "-q"])
    rep = json.loads(report.read_text())
    assert rep["detected"] == ["globex"]
    assert 0 < len(rep["trajectory"]) <= 30 and rep["trajectory"][-1]["detected"] == ["globex"]


def test_init_config_refuses_overwrite(tmp_path):
    p = tmp_path / "c.yaml"
    main(["init-config", "-o", str(p)])
    with pytest.raises(SystemExit) as e:
        main(["init-config", "-o", str(p)])
    assert "exists" in str(e.value.code)
    main(["init-config", "-o", str(p), "--force"])


def test_audit_requires_an_answer_source(tmp_path):
    keys, _ = _toy_setup(tmp_path)
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"provider_id": 0, "text_wm": "Some document text."}])
    with pytest.raises(SystemExit) as e:
        main(["audit", "--keys", str(keys), "--corpus", str(corpus), "--embedder", "hashing"])
    assert "--answers" in str(e.value.code)
    with pytest.raises(SystemExit) as e:
        main(["audit", "--keys", str(keys), "--corpus", str(corpus), "--embedder", "hashing", "--rag-cmd", "cat"])
    assert "--queries" in str(e.value.code)
