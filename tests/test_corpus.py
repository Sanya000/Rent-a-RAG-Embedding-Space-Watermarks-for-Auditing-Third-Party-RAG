"""Provider keys, corpus loading, corpus watermarking (parallel, resumable) and preprocessing."""

import hashlib
import json

import numpy as np
import pytest
from helpers import PROVIDER_DOCS, synthetic_corpus

from rentarag import (
    DirBucketWatermarker,
    Document,
    HashingEmbedder,
    ProviderKeys,
    ToyParaphraser,
    UnknownProviderError,
    WatermarkConfig,
    generate_provider_keys,
    iter_documents,
    load_corpus,
    load_provider_keys,
    split_sentences,
    watermark_corpus,
)
from rentarag.watermark import coverage_keep, mmr_select

EMB = HashingEmbedder(64)


class StatelessParaphraser:
    """Deterministic candidates that depend only on (sentence, n): safe to compare across thread schedules."""

    def __init__(self):
        self.calls = 0

    def __call__(self, sentence, n):
        self.calls += 1
        toy = ToyParaphraser(seed=int(hashlib.sha256(sentence.encode()).hexdigest()[:8], 16))
        return toy(sentence, n)


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


def test_provider_keys_named_and_indexed(tmp_path, seed13_keys):
    named = ProviderKeys.generate("acme, globex,initech")
    assert named.ids == ["acme", "globex", "initech"] and not named.is_indexed
    assert len(set(named.values())) == 3 and all(len(k) == 64 for k in named.values())
    assert "acme" in repr(named) and named["acme"] not in repr(named)

    seeded = ProviderKeys.generate(6, seed=13)
    assert seeded.is_indexed and seeded.keys_hex == seed13_keys == generate_provider_keys(6, seed=13)
    assert ProviderKeys.generate(["a", "b"], seed=13).keys_hex == seed13_keys[:2]

    p = tmp_path / "k.json"
    named.save(p)
    assert ProviderKeys.load(p) == named
    assert load_provider_keys(p) == named.keys_hex  # paper loader still reads the file
    seeded.save(p, seed=13)
    assert ProviderKeys.load(p).ids == list(range(6))
    assert "provider_names" not in json.loads(p.read_text())


def test_provider_keys_resolve():
    keys = ProviderKeys.generate(3)
    assert keys.resolve("1") == 1 and keys["2"] == keys[2] and "0" in keys and 7 not in keys
    with pytest.raises(UnknownProviderError, match="out of range 0..2"):
        keys.resolve(5)
    named = ProviderKeys({"7": "ab" * 32, "x": "cd" * 32})
    assert named.resolve(7) == "7"
    with pytest.raises(UnknownProviderError, match="unknown provider 'y'"):
        named["y"]


def test_provider_keys_validation():
    with pytest.raises(ValueError):
        ProviderKeys({"a": "not-hex"})
    with pytest.raises(ValueError):
        ProviderKeys({"a": "abcd"})  # too short
    with pytest.raises(ValueError):
        ProviderKeys.generate(["a", "a"])


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------

ROWS = [
    {"id": "d1", "provider": "acme", "text": "First document. It has two sentences here.", "title": "T1"},
    {"id": "d2", "provider": "globex", "text": "Second document text.", "title": "T2"},
]


def _check(docs):
    assert [(d.id, d.provider, d.text) for d in docs] == [(r["id"], r["provider"], r["text"]) for r in ROWS]


def test_load_jsonl_json_csv_tsv(tmp_path):
    (tmp_path / "c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    (tmp_path / "c.json").write_text(json.dumps(ROWS))
    (tmp_path / "w.json").write_text(json.dumps({"documents": ROWS}))
    header = "id,provider,text,title\n"
    (tmp_path / "c.csv").write_text(header + "".join(f'{r["id"]},{r["provider"]},"{r["text"]}",{r["title"]}\n' for r in ROWS))
    (tmp_path / "c.tsv").write_text(header.replace(",", "\t") + "".join("\t".join(r.values()) + "\n" for r in ROWS))
    for name in ("c.jsonl", "c.json", "w.json", "c.csv", "c.tsv"):
        docs = load_corpus(tmp_path / name)
        _check(docs)
        assert docs[0].meta["title"] == "T1"


def test_load_json_mapping_and_custom_fields(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"acme": ["a b c.", "d e f."], "globex": ["g h i."]}))
    docs = load_corpus(tmp_path / "m.json")
    assert [(d.id, d.provider) for d in docs] == [("acme-0", "acme"), ("acme-1", "acme"), ("globex-0", "globex")]
    rows = [{"doc": "x", "owner": "acme", "body": "Text."}]
    (tmp_path / "r.jsonl").write_text(json.dumps(rows[0]) + "\n")
    d = load_corpus(tmp_path / "r.jsonl", id_field="doc", provider_field="owner", text_field="body")[0]
    assert (d.id, d.provider, d.text, d.provider_field) == ("x", "acme", "Text.", "owner")
    with pytest.raises(ValueError, match="'nope'"):
        load_corpus(tmp_path / "r.jsonl", text_field="nope")


def test_load_directory(tmp_path):
    for prov, name, text in [("acme", "a.txt", "Alpha text."), ("acme", "sub/b.md", "Beta text."), ("globex", "c.txt", "Gamma.")]:
        f = tmp_path / prov / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    (tmp_path / "ignore.json").write_text("{}")
    docs = load_corpus(tmp_path)
    assert [(d.id, d.provider, d.text) for d in docs] == [
        ("acme/a", "acme", "Alpha text."),
        ("acme/sub/b", "acme", "Beta text."),
        ("globex/c", "globex", "Gamma."),
    ]
    (tmp_path / "top.txt").write_text("Loose file.")
    with pytest.raises(ValueError, match="default_provider"):
        load_corpus(tmp_path)
    assert load_corpus(tmp_path, default_provider="misc")[-1].provider == "misc"


def test_iter_documents_shapes():
    docs = list(iter_documents([("acme", "t1"), ("id9", "globex", "t2"), {"provider_id": 3, "text": "t3"}]))
    assert [(d.id, d.provider, d.text) for d in docs] == [("doc-0", "acme", "t1"), ("id9", "globex", "t2"), ("doc-2", 3, "t3")]
    assert docs[2].provider_field == "provider_id"
    with pytest.raises(ValueError, match="default_provider"):
        list(iter_documents(["plain"]))
    assert next(iter_documents(["plain"], default_provider="p")).provider == "p"
    with pytest.raises(TypeError):
        list(iter_documents({"acme": "single string"}))
    with pytest.raises(ValueError, match="provider"):
        list(iter_documents([{"text": "x"}]))


def test_unsupported_format(tmp_path):
    (tmp_path / "c.xml").write_text("<x/>")
    with pytest.raises(ValueError, match="unsupported corpus format"):
        load_corpus(tmp_path / "c.xml")
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# watermark_corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ten_providers():
    names = [f"provider-{c}" for c in "abcdefghij"]
    keys = ProviderKeys.generate(names, seed=99)
    corpus = synthetic_corpus(10, 2, 4, seed=5)
    docs = [{"id": f"{names[p]}/{i}", "provider": names[p], "text": t} for p in range(10) for i, t in enumerate(corpus[p])]
    return keys, docs


def test_ten_named_providers_match_single_watermarker(ten_providers):
    keys, docs = ten_providers
    out = watermark_corpus(docs, keys, paraphraser=StatelessParaphraser(), embedder=EMB, config="fast")
    assert [d.id for d in out] == [d["id"] for d in docs]
    assert set(out.by_provider()) == set(keys)
    for d, src in zip(out, docs):
        ref = DirBucketWatermarker(keys[src["provider"]], StatelessParaphraser(), EMB, WatermarkConfig.preset("fast"))
        assert d.text == ref.watermark(src["text"]).text
        assert d.original == src["text"] and d.stats["n_sentences"] == 4
    s = out.summary()
    assert len(s) == 10 and all(v["documents"] == 2 and v["sentences"] == 8 for v in s.values())


def test_parallel_equals_sequential(ten_providers):
    keys, docs = ten_providers
    seq = watermark_corpus(docs, keys, paraphraser=StatelessParaphraser(), embedder=EMB, config="fast", max_workers=1)
    par = watermark_corpus(docs, keys, paraphraser=StatelessParaphraser(), embedder=EMB, config="fast", max_workers=8)
    assert [(d.id, d.provider, d.text, d.stats) for d in seq] == [(d.id, d.provider, d.text, d.stats) for d in par]


def test_sentence_parallel_equals_sequential(seed13_keys):
    text = " ".join(PROVIDER_DOCS[1])
    wm = DirBucketWatermarker(seed13_keys[1], StatelessParaphraser(), EMB)
    assert wm.watermark(text, max_workers=4).text == wm.watermark(text).text


def test_resume_from_cache(tmp_path, ten_providers):
    keys, docs = ten_providers
    cache = tmp_path / "cache.jsonl"
    para = StatelessParaphraser()
    full = watermark_corpus(docs[:6], keys, paraphraser=para, embedder=EMB, config="fast", cache_path=cache)
    first_calls = para.calls
    lines = cache.read_text().splitlines(keepends=True)
    assert len(lines) == 6
    cache.write_text("".join(lines[:4]) + lines[4][:25])  # interrupted mid-write

    para2 = StatelessParaphraser()
    resumed = watermark_corpus(docs[:6], keys, paraphraser=para2, embedder=EMB, config="fast", cache_path=cache)
    assert [d.text for d in resumed] == [d.text for d in full]
    assert [d.stats for d in resumed] == [d.stats for d in full]
    assert 0 < para2.calls < first_calls  # only the two missing documents were redone
    assert len(cache.read_text().splitlines()) == 6

    para3 = StatelessParaphraser()
    changed = [dict(d) for d in docs[:6]]
    changed[0]["text"] += " An extra sentence was appended to this document later on."
    watermark_corpus(changed, keys, paraphraser=para3, embedder=EMB, config="fast", cache_path=cache)
    assert para3.calls > 0  # edited document is recomputed


def test_cache_requires_unique_ids(tmp_path, ten_providers):
    keys, docs = ten_providers
    with pytest.raises(ValueError, match="duplicate"):
        watermark_corpus([docs[0], docs[0]], keys, paraphraser="toy", embedder=EMB, cache_path=tmp_path / "c.jsonl")


def test_rows_and_save(tmp_path, seed13_keys):
    keys = ProviderKeys(seed13_keys[:2])
    rows = [{"id": "a", "provider_id": 0, "text": PROVIDER_DOCS[0][0], "title": "x"}]
    out = watermark_corpus(rows, keys, paraphraser="toy", embedder=EMB)
    row = out.rows()[0]
    assert row["provider_id"] == 0 and row["title"] == "x" and row["text"] == rows[0]["text"]
    assert row["text_wm"] == out[0].text and {"wm_gain_mean", "wm_changed_frac", "wm_n_sentences"} <= set(row)
    out.save(tmp_path / "o.jsonl")
    back = load_corpus(tmp_path / "o.jsonl", text_field="text_wm")
    assert back[0].text == out[0].text and back[0].provider == 0


def test_unknown_provider_in_corpus(seed13_keys):
    with pytest.raises(UnknownProviderError):
        watermark_corpus([("nobody", "Text.")], ProviderKeys(seed13_keys[:2]), paraphraser="toy", embedder=EMB)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


class _RefEncoder:
    """``SentenceTransformer.encode`` stand-in for the paper build's ``mmr_select``."""

    def encode(self, sents, normalize_embeddings, convert_to_numpy):
        return EMB(sents)


@pytest.mark.parametrize("k,lam", [(3, 0.7), (5, 0.5), (12, 0.7), (1, 0.9)])
def test_mmr_matches_paper_build(k, lam):
    build = pytest.importorskip("build_corpora_dirbucket")
    sents = split_sentences(" ".join(PROVIDER_DOCS[0] + PROVIDER_DOCS[2]))
    assert mmr_select(EMB(sents), k, lam) == build.mmr_select(sents, k, lam, _RefEncoder())


def test_prepare_selection(seed13_keys):
    text = " ".join(PROVIDER_DOCS[0] + PROVIDER_DOCS[1])
    sents = split_sentences(text)
    first = DirBucketWatermarker(seed13_keys[0], "toy", EMB, {"max_sentences": 3})
    assert first.prepare(text) == " ".join(sents[:3])
    mmr = DirBucketWatermarker(seed13_keys[0], "toy", EMB, {"max_sentences": 3, "sentence_selection": "mmr"})
    assert mmr.prepare(text) == " ".join(sents[i] for i in mmr_select(EMB(sents), 3, 0.7))
    trunc = DirBucketWatermarker(seed13_keys[0], "toy", EMB, {"max_chars": 120})
    assert trunc.prepare(text) == text[:120]
    assert trunc.watermark(text).n_sentences == len(split_sentences(text[:120]))
    assert DirBucketWatermarker(seed13_keys[0], "toy", EMB).prepare(text) == text


def test_coverage(seed13_keys):
    text = " ".join(PROVIDER_DOCS[0] + PROVIDER_DOCS[1] + PROVIDER_DOCS[2])
    full = DirBucketWatermarker(seed13_keys[0], StatelessParaphraser(), EMB).watermark(text, doc_id="d")
    half = DirBucketWatermarker(seed13_keys[0], StatelessParaphraser(), EMB, {"coverage": 0.5}).watermark(text, doc_id="d")
    kept = [coverage_keep("d", i, 0.5) for i in range(half.n_sentences)]
    assert 0 < sum(kept) < len(kept)
    for s, f, k in zip(half.sentences, full.sentences, kept):
        assert s.watermarked == k
        assert s.text == (f.text if k else s.original)
    nested = [coverage_keep("d", i, 0.25) for i in range(half.n_sentences)]
    assert all(k for k, n in zip(kept, nested) if n)  # 25% subset of 50%


def test_config_validation():
    with pytest.raises(ValueError):
        WatermarkConfig(coverage=1.5)
    with pytest.raises(ValueError):
        WatermarkConfig(sentence_selection="random")
    with pytest.raises(ValueError):
        WatermarkConfig(max_sentences=0)


def test_document_dataclass_passthrough(seed13_keys):
    docs = [Document("x", 1, PROVIDER_DOCS[1][0])]
    out = watermark_corpus(docs, seed13_keys[:2], paraphraser="toy", embedder=EMB)
    assert out[0].provider == 1 and np.isfinite(out[0].stats["gain_mean"])
