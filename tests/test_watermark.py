"""Hill-climbing watermarker and paraphrasers against the reference implementation."""

from types import SimpleNamespace

import pytest
from helpers import PROVIDER_DOCS

from rentarag import (
    PROMPT_CORPUS,
    CallableParaphraser,
    DirBucketWatermarker,
    OpenAIParaphraser,
    OrthoDirectionManager,
    ParaphraseError,
    ToyParaphraser,
    WatermarkConfig,
)

REF_PARAMS = dict(bucket_bits=96, bucket_seed=2025, min_sem_sim=0.70, min_delta_proj=5e-4, rounds=6, k_batch=24)


def _patch_reference_paraphraser(ref, monkeypatch, para):
    monkeypatch.setattr(ref, "paraphrase_diverse_batch", lambda s, model="gpt-4o-mini", n_total=16, **_: para(s, n_total))


def _assert_same(result, ref_text, ref_summary):
    assert result.text == ref_text
    mine = result.summary()
    for k in ("n_sentences", "changed_sentences", "changed_frac", "gain_mean", "gain_median", "gain_p90", "bucket_flips"):
        assert mine[k] == getattr(ref_summary, k), k


def test_paper_defaults():
    c = WatermarkConfig()
    assert (c.bucket_bits, c.bucket_seed, c.n_candidates, c.rounds, c.min_sem_sim, c.min_delta_proj) == (
        96,
        2025,
        24,
        6,
        0.70,
        5e-4,
    )
    assert c.force_accept_last_round is False


@pytest.mark.parametrize("pid", [0, 1, 2])
@pytest.mark.parametrize("force", [False, True])
def test_watermark_matches_reference(ref_fake, monkeypatch, fake_embedder, seed13_keys, pid, force):
    text = " ".join(PROVIDER_DOCS[pid])
    _patch_reference_paraphraser(ref_fake, monkeypatch, ToyParaphraser(seed=pid))
    ref_wm = ref_fake.DirBucketWatermarker(
        seed13_keys[pid], ref_fake.GenParams(**REF_PARAMS), force_accept_last_round=force
    )
    ref_text, ref_summary = ref_wm.watermark(text)

    cfg = WatermarkConfig(force_accept_last_round=force)
    res = DirBucketWatermarker(seed13_keys[pid], ToyParaphraser(seed=pid), fake_embedder, cfg).watermark(text)
    _assert_same(res, ref_text, ref_summary)
    assert res.gain_mean > 0 and res.changed_frac > 0


@pytest.mark.parametrize(
    "params",
    [dict(rounds=1, k_batch=4, min_sem_sim=0.9, min_delta_proj=1e-2), dict(rounds=3, k_batch=8, min_sem_sim=0.5, min_delta_proj=0.0)],
)
def test_watermark_matches_reference_other_params(ref_fake, monkeypatch, fake_embedder, seed13_keys, params):
    text = PROVIDER_DOCS[1][0]
    _patch_reference_paraphraser(ref_fake, monkeypatch, ToyParaphraser(seed=5))
    ref_text, ref_summary = ref_fake.DirBucketWatermarker(seed13_keys[3], ref_fake.GenParams(**params)).watermark(text)
    cfg = WatermarkConfig(
        n_candidates=params["k_batch"],
        rounds=params["rounds"],
        min_sem_sim=params["min_sem_sim"],
        min_delta_proj=params["min_delta_proj"],
    )
    res = DirBucketWatermarker(seed13_keys[3], ToyParaphraser(seed=5), fake_embedder, cfg).watermark(text)
    _assert_same(res, ref_text, ref_summary)


def test_watermark_ortho_matches_reference(ref_fake, monkeypatch, fake_embedder, seed13_keys):
    text = PROVIDER_DOCS[2][1]
    _patch_reference_paraphraser(ref_fake, monkeypatch, ToyParaphraser(seed=9))
    ref_wm = ref_fake.DirBucketWatermarker(
        seed13_keys[2],
        ref_fake.GenParams(**REF_PARAMS),
        direction_scheme="ortho",
        provider_id=2,
        all_provider_keys=seed13_keys,
    )
    ref_text, ref_summary = ref_wm.watermark(text)
    dirs = OrthoDirectionManager(dict(enumerate(seed13_keys)), fake_embedder.dim).for_provider(2)
    res = DirBucketWatermarker(None, ToyParaphraser(seed=9), fake_embedder, directions=dirs).watermark(text)
    _assert_same(res, ref_text, ref_summary)


def test_sentence_logs(fake_embedder, seed13_keys):
    wm = DirBucketWatermarker(seed13_keys[0], ToyParaphraser(), fake_embedder, keep_candidates=True)
    res = wm.watermark(PROVIDER_DOCS[0][0])
    for s in res.sentences:
        assert s.gain == pytest.approx(s.final_proj - s.base_proj)
        if s.text != s.original:
            assert s.changed
        assert len(s.rounds) == WatermarkConfig().rounds
        assert all(c.accepted == (c.reject_reason is None) for r in s.rounds for c in r)
        if not s.changed:
            assert s.gain == 0.0 and s.sem_sim == pytest.approx(1.0, abs=1e-5)
    d = res.to_dict()
    assert set(d) == {"text", "summary", "sentences"} and "rounds" not in d["sentences"][0]


def test_no_candidates_leaves_text_unchanged(fake_embedder, seed13_keys):
    res = DirBucketWatermarker(seed13_keys[0], lambda s, n: [], fake_embedder).watermark(PROVIDER_DOCS[0][0])
    assert res.changed_frac == 0.0 and res.gain_mean == 0.0
    assert res.text == " ".join(s.original for s in res.sentences)


def test_empty_text(fake_embedder, seed13_keys):
    res = DirBucketWatermarker(seed13_keys[0], ToyParaphraser(), fake_embedder).watermark("  ")
    assert res.text == "  " and res.n_sentences == 0 and res.summary()["changed_frac"] == 0.0


class _FakeClient:
    """Records chat.completions.create kwargs and replays canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


REPLY_A = "1. Alpha one.\n2) Beta two.\n- Gamma three.\n4. Alpha one.\n\n5. Delta four."
REPLY_B = "1. Beta two.\n2. Epsilon five.\n3. Zeta six."


@pytest.mark.parametrize("n", [2, 4, 5, 8, 24])
def test_openai_call_pattern_matches_reference(ref, monkeypatch, n):
    ref_calls = []
    replies = [REPLY_A, REPLY_B]

    def fake_chat_once(model, prompt, temperature=0.8, max_tokens=512):
        ref_calls.append(dict(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens))
        return replies[len(ref_calls) - 1]

    monkeypatch.setattr(ref, "_chat_once", fake_chat_once)
    expected = ref.paraphrase_diverse_batch("The cat sat.", model="gpt-4o-mini", n_total=n)

    client = _FakeClient([REPLY_A, REPLY_B])
    got = OpenAIParaphraser(client=client)("The cat sat.", n)
    assert got == expected
    assert len(client.calls) == len(ref_calls)
    for mine, theirs in zip(client.calls, ref_calls):
        assert mine["model"] == theirs["model"]
        assert mine["temperature"] == theirs["temperature"]
        assert mine["max_tokens"] == theirs["max_tokens"]
        assert mine["messages"] == [{"role": "user", "content": theirs["prompt"]}]


def test_prompt_corpus_is_reference_prompt():
    assert PROMPT_CORPUS.format(sentence="X").endswith("Sentence:\nX")
    assert "Produce 8 variants: 2 very light, 3 medium, 3 strong rewrites." in PROMPT_CORPUS


def test_k_variants_prompt_matches_cost_ablation():
    client = _FakeClient([REPLY_A, REPLY_B])
    para = OpenAIParaphraser.k_variants(client=client)
    out = para("S.", 24)
    assert [c["temperature"] for c in client.calls] == [0.7, 1.0]
    assert all(c["max_tokens"] == 900 for c in client.calls)
    prompt = client.calls[0]["messages"][0]["content"]
    assert prompt == (
        "Rewrite the sentence in a diverse set of paraphrases while preserving all facts.\n"
        "Produce 24 variants: 8 very light, 8 medium, 8 strong rewrites.\n"
        "Vary voice, ordering, connectives, nominalization, and tone.\n"
        "Return one variant per line, numbered.\n\nSentence:\nS."
    )
    assert out == ["Alpha one.", "Beta two.", "Gamma three.", "Delta four.", "Epsilon five.", "Zeta six."]


def test_openai_errors_raise_by_default():
    para = OpenAIParaphraser(client=_FakeClient([RuntimeError("rate limited")]))
    with pytest.raises(ParaphraseError):
        para("The cat sat.", 8)
    assert para.failures == 1


def test_openai_errors_can_be_counted():
    para = OpenAIParaphraser(client=_FakeClient([RuntimeError("boom")] * 2), on_error="warn")
    assert para("a.", 8) == [] and para("b.", 8) == []
    assert para.failures == 2


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ParaphraseError, match="OPENAI_API_KEY"):
        OpenAIParaphraser()
    OpenAIParaphraser(base_url="http://localhost:8000/v1")


def test_callable_paraphraser():
    para = CallableParaphraser(lambda s: [s, " x ", "x", "y", "z"], takes_n=False)
    assert para("s", 2) == ["s", "x"]
    bad = CallableParaphraser(lambda s, n: 1 / 0, on_error="warn")
    assert bad("s", 3) == [] and bad.failures == 1


def test_toy_paraphraser_deterministic():
    a, b = ToyParaphraser(seed=1), ToyParaphraser(seed=1)
    s = PROVIDER_DOCS[0][0].split(". ")[0] + "."
    assert a(s, 8) == b(s, 8)
    out = a(s, 8)
    assert 0 < len(out) <= 8 and s not in out and len(set(out)) == len(out)
