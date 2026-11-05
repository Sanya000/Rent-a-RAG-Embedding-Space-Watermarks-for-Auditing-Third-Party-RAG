"""LLM backends, spec parsing, retries and paraphraser adapters (all offline, mocked clients)."""

import sys
from types import SimpleNamespace

import pytest

from rentarag import (
    AnthropicChat,
    CallableChat,
    CallableParaphraser,
    HFChat,
    LiteLLMChat,
    LLMParaphraser,
    MissingCredentialsError,
    MissingDependencyError,
    OpenAIChat,
    OpenAIParaphraser,
    ParaphraseError,
    ToyChat,
    ToyParaphraser,
    get_llm,
    get_paraphraser,
)
from rentarag.llm import GEMINI_BASE_URL, call_with_retries, is_retryable

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _chat_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeOpenAIClient:
    """Stand-in for ``openai.OpenAI``: records constructor and create() kwargs."""

    instances: list = []

    def __init__(self, **kw):
        self.init = kw
        self.calls = []
        self.replies = ["1. Alpha.\n2. Beta."]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        FakeOpenAIClient.instances.append(self)

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return _chat_response(r)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAIClient.instances = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_BASE_URL", "VLLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return FakeOpenAIClient


class FakeAnthropicClient:
    def __init__(self, **kw):
        self.init = kw
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="1. One.\n2. Two.")]
        return SimpleNamespace(content=blocks)


@pytest.fixture
def fake_anthropic(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropicClient))


class RateLimitError(Exception):
    pass


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------


def test_openai_spec(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = get_llm("openai/gpt-4o-mini")
    assert isinstance(llm, OpenAIChat) and llm.model == "gpt-4o-mini"
    init = fake_openai.instances[-1].init
    assert init["api_key"] == "sk-test" and init["base_url"] is None and init["max_retries"] == 0


def test_openai_compatible_server_needs_no_key(fake_openai):
    get_llm("openai/my-model", base_url="http://localhost:9000/v1")
    assert fake_openai.instances[-1].init == {
        "api_key": "EMPTY",
        "base_url": "http://localhost:9000/v1",
        "max_retries": 0,
        "timeout": 120.0,
    }


def test_openai_without_key_fails_fast(fake_openai):
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        get_llm("openai/gpt-4o-mini")


@pytest.mark.parametrize(
    "spec,base_url",
    [
        ("ollama/llama3.1", "http://localhost:11434/v1"),
        ("vllm/Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1"),
    ],
)
def test_local_server_specs(fake_openai, spec, base_url):
    llm = get_llm(spec)
    assert llm.model == spec.split("/", 1)[1]
    assert fake_openai.instances[-1].init["base_url"] == base_url
    assert fake_openai.instances[-1].init["api_key"] == "EMPTY"


def test_local_server_env_override(fake_openai, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    get_llm("ollama/llama3.1")
    assert fake_openai.instances[-1].init["base_url"] == "http://gpu-box:11434/v1"


def test_gemini_spec(fake_openai, monkeypatch):
    with pytest.raises(MissingCredentialsError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        get_llm("gemini/gemini-2.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    llm = get_llm("gemini/gemini-2.5-flash")
    assert llm.model == "gemini-2.5-flash"
    init = fake_openai.instances[-1].init
    assert init["base_url"] == GEMINI_BASE_URL and init["api_key"] == "g-key"


def test_anthropic_spec_and_call_shape(fake_anthropic):
    llm = get_llm("anthropic/claude-haiku-5-5", api_key="a-key")
    assert isinstance(llm, AnthropicChat)
    assert llm.client.init["api_key"] == "a-key" and llm.client.init["max_retries"] == 0
    out = llm.complete("Hi", temperature=0.7, max_tokens=700)
    assert out == "1. One.\n2. Two."
    call = llm.client.calls[-1]
    assert call["model"] == "claude-haiku-5-5"
    assert call["messages"] == [{"role": "user", "content": "Hi"}]
    assert "temperature" not in call and call["max_tokens"] == 4096
    llm2 = get_llm("anthropic/claude-haiku-5-5", send_temperature=True, min_max_tokens=0, output_config={"effort": "low"})
    llm2.complete("Hi", temperature=0.7, max_tokens=700)
    call = llm2.client.calls[-1]
    assert call["temperature"] == 0.7 and call["max_tokens"] == 700 and call["output_config"] == {"effort": "low"}


def test_litellm_spec(monkeypatch):
    calls = []
    fake = SimpleNamespace(completion=lambda **kw: calls.append(kw) or _chat_response("1. X."))
    monkeypatch.setitem(sys.modules, "litellm", fake)
    llm = get_llm("litellm/bedrock/anthropic.claude-x", api_base="http://proxy")
    assert isinstance(llm, LiteLLMChat) and llm.model == "bedrock/anthropic.claude-x"
    assert llm.complete("p", temperature=1.0, max_tokens=5) == "1. X."
    assert calls[-1] == {
        "model": "bedrock/anthropic.claude-x",
        "messages": [{"role": "user", "content": "p"}],
        "api_base": "http://proxy",
        "temperature": 1.0,
        "max_tokens": 5,
    }


@pytest.mark.parametrize("spec,module,extra", [("anthropic/x", "anthropic", "anthropic"), ("litellm/x", "litellm", "litellm")])
def test_missing_sdk_names_the_extra(monkeypatch, spec, module, extra):
    monkeypatch.setitem(sys.modules, module, None)  # makes `import module` fail
    with pytest.raises(MissingDependencyError, match=rf'rentarag\[{extra}\]'):
        get_llm(spec)


class _FakeTensor:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids))

    def to(self, device):
        return self


class _FakeTokenizer:
    chat_template = "x"
    pad_token_id = None
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, return_tensors, add_special_tokens):
        self.last_text = text
        return {"input_ids": _FakeTensor([1, 2, 3])}

    def decode(self, ids, skip_special_tokens):
        return "1. Decoded." if list(ids) == [7, 8] else "?"


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, **kw):
        self.calls.append(kw)
        return [list(input_ids.ids) + [7, 8]]


def test_hf_chat_with_injected_model():
    pytest.importorskip("torch")
    tok, model = _FakeTokenizer(), _FakeModel()
    llm = HFChat("Qwen/Qwen2.5-1.5B-Instruct", tokenizer=tok, model_obj=model, device="cpu")
    assert llm.complete("Hello", temperature=0.7, max_tokens=50) == "1. Decoded."
    assert tok.last_text == "<user>Hello<assistant>"
    assert model.calls[-1] == {"max_new_tokens": 50, "do_sample": True, "temperature": 0.7, "top_p": 0.95, "pad_token_id": 0}
    llm.complete("Hello", temperature=0.0)
    assert model.calls[-1]["do_sample"] is False and model.calls[-1]["max_new_tokens"] == 512


def test_hf_spec_routes_to_hfchat(monkeypatch):
    seen = {}

    class Stub:
        def __init__(self, model, **kw):
            seen.update(model=model, **kw)

        def complete(self, prompt, *, temperature=None, max_tokens=None):
            return ""

    monkeypatch.setattr("rentarag.llm.HFChat", Stub)
    get_llm("hf/Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
    assert seen == {"model": "Qwen/Qwen2.5-1.5B-Instruct", "device": "cpu"}


def test_toy_and_callable_llms():
    assert isinstance(get_llm("toy"), ToyChat)
    out = get_llm("toy").complete("Produce variants.\n\nSentence:\nThe big council approved the plan.")
    assert out.startswith("1. ")
    fn_llm = get_llm(lambda prompt: prompt.upper())
    assert isinstance(fn_llm, CallableChat) and fn_llm.complete("ab", temperature=0.3) == "AB"
    seen = {}
    get_llm(lambda prompt, temperature=None, max_tokens=None: seen.update(t=temperature, m=max_tokens) or "x").complete(
        "p", temperature=0.5, max_tokens=9
    )
    assert seen == {"t": 0.5, "m": 9}


@pytest.mark.parametrize("spec", ["gpt-4o-mini", "nonsense/model", "openai/", ""])
def test_bad_specs(spec):
    with pytest.raises((ValueError, TypeError)) as e:
        get_llm(spec)
    if spec == "gpt-4o-mini":
        assert "openai/gpt-4o-mini" in str(e.value)


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (RateLimitError("slow down"), True),
        (HTTPError(429), True),
        (HTTPError(503), True),
        (HTTPError(529), True),
        (HTTPError(400), False),
        (HTTPError(401), False),
        (TimeoutError(), True),
        (ConnectionError(), True),
        (RuntimeError("rate limited"), False),
        (ValueError("bad"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


def test_call_with_retries_backs_off_exponentially():
    sleeps, attempts = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 4:
            raise HTTPError(429)
        return "ok"

    assert call_with_retries(flaky, max_retries=5, backoff=1.0, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 3
    for k, s in enumerate(sleeps):
        assert 2**k <= s < 1.25 * 2**k


def test_call_with_retries_gives_up_and_skips_client_errors():
    sleeps = []
    with pytest.raises(HTTPError):
        call_with_retries(lambda: (_ for _ in ()).throw(HTTPError(500)), max_retries=2, sleep=sleeps.append)
    assert len(sleeps) == 2
    with pytest.raises(HTTPError):
        call_with_retries(lambda: (_ for _ in ()).throw(HTTPError(400)), max_retries=5, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_retry_after_header_is_respected():
    sleeps = []
    err = HTTPError(429)
    err.response = SimpleNamespace(headers={"retry-after": "7"})
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise err
        return 1

    call_with_retries(fn, sleep=sleeps.append)
    assert sleeps == [7.0]


def test_paraphraser_retries_rate_limits(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = FakeOpenAIClient()
    client.replies = [RateLimitError("429"), RateLimitError("429"), "1. A.\n2. B.\n3. C."]
    para = OpenAIParaphraser(client=client)
    assert para("S.", 3) == ["A.", "B.", "C."]
    assert len(client.calls) == 3 and para.failures == 0


def test_paraphraser_error_policy_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = FakeOpenAIClient()
    client.replies = [HTTPError(503)] * 10
    para = OpenAIParaphraser(client=client, max_retries=2, on_error="warn")
    assert para("S.", 3) == [] and para.failures == 1
    assert len(client.calls) == 3


# ---------------------------------------------------------------------------
# get_paraphraser
# ---------------------------------------------------------------------------


def test_get_paraphraser_openai_default_pattern(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    para = get_paraphraser("openai/gpt-4o-mini")
    assert isinstance(para, OpenAIParaphraser)
    assert (para.temperatures, para.stop_when_enough, para.max_tokens) == ((0.7, 1.0), True, 700)
    kv = get_paraphraser("openai/gpt-4o-mini", prompt="k-variants")
    assert (kv.stop_when_enough, kv.max_tokens) == (False, 900)
    assert kv.prompt == OpenAIParaphraser.k_variants(client=FakeOpenAIClient()).prompt


@pytest.mark.parametrize("spec", ["ollama/llama3.1", "vllm/m", "anthropic/claude-haiku-5-5", "gemini/gemini-2.5-flash"])
def test_get_paraphraser_backends(fake_openai, fake_anthropic, monkeypatch, spec):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    para = get_paraphraser(spec, prompt="k-variants", max_retries=1, on_error="warn")
    assert isinstance(para, LLMParaphraser)
    assert para.model == spec.split("/", 1)[1]
    assert (para.max_retries, para.on_error, para.stop_when_enough) == (1, "warn", False)


def test_get_paraphraser_splits_llm_and_paraphraser_options(fake_openai):
    para = get_paraphraser("openai/m", base_url="http://x/v1", temperatures=(0.2,), max_tokens=50)
    assert fake_openai.instances[-1].init["base_url"] == "http://x/v1"
    assert para.temperatures == (0.2,) and para.max_tokens == 50
    para("Sentence one.", 4)
    call = fake_openai.instances[-1].calls[-1]
    assert call["temperature"] == 0.2 and call["max_tokens"] == 50


def test_custom_prompt_template():
    seen = []
    para = LLMParaphraser(lambda p: seen.append(p) or "1. a\n2. b", prompt="Give {n} rewrites of: {sentence}")
    assert para("Hello.", 2) == ["a", "b"]
    assert seen == ["Give 2 rewrites of: Hello."]
    with pytest.raises(ValueError):
        LLMParaphraser(lambda p: "", prompt="no placeholder")


def test_llm_paraphraser_over_toy_llm_parses_lists():
    para = get_paraphraser(get_llm("toy"))
    out = para("The council approved a large budget for the new park on Monday.", 8)
    assert 0 < len(out) <= 8 and len(set(out)) == len(out)
    assert para.calls == 1  # 8 candidates from the first call: no second call


def test_get_paraphraser_toy_and_instances():
    toy = get_paraphraser("toy:3")
    assert isinstance(toy, ToyParaphraser) and toy.seed == 3
    assert get_paraphraser("toy", prompt="corpus").seed == 0  # LLM-only options are ignored
    assert get_paraphraser(toy) is toy
    with pytest.raises(TypeError):
        get_paraphraser(toy, prompt="corpus")


def test_get_paraphraser_callables():
    p2 = get_paraphraser(lambda s, n: [s + "!"] * n + ["x"])
    assert isinstance(p2, CallableParaphraser) and p2.takes_n and p2("a", 3) == ["a!", "x"]
    p1 = get_paraphraser(lambda s: [s.upper(), s.lower()])
    assert not p1.takes_n and p1("Ab", 5) == ["AB", "ab"]


def test_str_returning_callable_is_sampled_until_n():
    import itertools

    counter = itertools.count()
    para = CallableParaphraser(lambda s: f"{s} v{next(counter)}")
    assert para("s", 4) == ["s v0", "s v1", "s v2", "s v3"]
    const = []
    para = CallableParaphraser(lambda s: const.append(1) or "same")
    assert para("s", 10) == ["same"] and len(const) == 4  # stops after 3 calls add nothing


def test_paraphrase_error_wraps_missing_key(fake_openai):
    with pytest.raises(ParaphraseError, match="OPENAI_API_KEY"):
        get_paraphraser("openai/gpt-4o-mini")
