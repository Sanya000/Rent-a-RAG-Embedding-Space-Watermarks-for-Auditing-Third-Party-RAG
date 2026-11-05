"""Presets, dict round trips and config files."""

import json

import pytest

from rentarag import (
    AUDIT_PRESETS,
    WATERMARK_PRESETS,
    AuditConfig,
    HashingEmbedder,
    RentaragConfig,
    ToyParaphraser,
    WatermarkConfig,
    load_config,
    write_example_config,
)
from rentarag.config import EXAMPLE_CONFIG


def test_paper_presets_are_defaults():
    assert WatermarkConfig.preset("paper") == WatermarkConfig()
    assert AuditConfig.preset("paper") == AuditConfig()
    assert set(WATERMARK_PRESETS) == {"paper", "paper-corpus", "fast", "strong"}
    assert set(AUDIT_PRESETS) == {"paper", "paper-query", "small-corpus"}


def test_presets_and_overrides():
    fast = WatermarkConfig.preset("fast")
    assert (fast.n_candidates, fast.rounds, fast.min_sem_sim) == (8, 3, 0.70)
    assert WatermarkConfig.preset("strong", rounds=12).rounds == 12
    pc = WatermarkConfig.preset("paper-corpus")
    assert (pc.max_chars, pc.max_sentences, pc.sentence_selection) == (4000, 12, "mmr")
    assert AuditConfig.preset("small-corpus").min_support == 10
    assert AuditConfig.preset("paper-query").reference_selection == "query"
    assert AuditConfig.from_dict({"preset": "small-corpus", "min_support": 5}).min_support == 5


@pytest.mark.parametrize("cls", [WatermarkConfig, AuditConfig])
def test_dict_round_trip_and_errors(cls):
    for name in (WATERMARK_PRESETS if cls is WatermarkConfig else AUDIT_PRESETS):
        c = cls.preset(name)
        assert cls.from_dict(c.to_dict()) == c
        assert cls.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    with pytest.raises(ValueError, match="unknown .* option"):
        cls.from_dict({"roundz": 3})
    with pytest.raises(ValueError, match="preset"):
        cls.from_dict({"preset": "nope"})
    assert cls.coerce(None) == cls() and cls.coerce("paper") == cls()


def test_audit_config_validation():
    with pytest.raises(ValueError):
        AuditConfig(reference_selection="some")
    with pytest.raises(ValueError):
        AuditConfig(direction_scheme="random")
    assert AuditConfig.from_dict({"tie_margin": None}).tie_margin is None


def test_example_config_loads_to_paper_defaults(tmp_path):
    p = write_example_config(tmp_path / "rentarag.yaml")
    cfg = load_config(p)
    assert cfg.watermark == WatermarkConfig() and cfg.audit == AuditConfig()
    assert cfg.paraphraser["spec"] == "openai/gpt-4o-mini" and cfg.paraphraser["prompt"] == "corpus"
    assert cfg.embedder == {"spec": "sentence-transformers/all-mpnet-base-v2"}
    assert cfg.resolve_path(cfg.keys) == tmp_path / "provider_keys.json"
    assert cfg.workers == 8 and cfg.queries == {"llm": "openai/gpt-4o-mini", "n": 100}
    with pytest.raises(FileExistsError):
        write_example_config(p)
    write_example_config(p, overwrite=True)


def test_shipped_example_matches_template():
    from pathlib import Path

    shipped = Path(__file__).resolve().parents[1] / "examples" / "config.yaml"
    assert shipped.read_text(encoding="utf-8") == EXAMPLE_CONFIG


@pytest.mark.parametrize("fmt", ["yaml", "toml", "json"])
def test_config_formats(tmp_path, fmt):
    data = {
        "keys": "keys.json",
        "paraphraser": {"spec": "toy", "seed": 4},
        "embedder": "hashing:32",
        "watermark": {"preset": "fast", "min_sem_sim": 0.6},
        "audit": {"preset": "small-corpus"},
        "corpus": {"text_field": "body"},
        "workers": 3,
    }
    p = tmp_path / f"c.{fmt}"
    if fmt == "json":
        p.write_text(json.dumps(data))
    elif fmt == "yaml":
        import yaml

        p.write_text(yaml.safe_dump(data))
    else:
        p.write_text(
            'keys = "keys.json"\nembedder = "hashing:32"\nworkers = 3\n'
            '[paraphraser]\nspec = "toy"\nseed = 4\n'
            '[watermark]\npreset = "fast"\nmin_sem_sim = 0.6\n'
            '[audit]\npreset = "small-corpus"\n[corpus]\ntext_field = "body"\n'
        )
    cfg = load_config(p)
    assert cfg.watermark == WatermarkConfig.preset("fast", min_sem_sim=0.6)
    assert cfg.audit.min_support == 10 and cfg.workers == 3 and cfg.corpus == {"text_field": "body"}
    para, emb = cfg.make_paraphraser(), cfg.make_embedder()
    assert isinstance(para, ToyParaphraser) and para.seed == 4
    assert isinstance(emb, HashingEmbedder) and emb.dim == 32
    assert RentaragConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_config_rejects_unknown_sections(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"watermarks": {}}))
    with pytest.raises(ValueError, match="unknown config section"):
        load_config(p)
    p.write_text(json.dumps({"watermark": {"round": 2}}))
    with pytest.raises(ValueError, match="option"):
        load_config(p)
    p.write_text(json.dumps({"corpus": {"txt_field": "x"}}))
    with pytest.raises(ValueError, match="corpus option"):
        load_config(p)
    (tmp_path / "c.ini").write_text("")
    with pytest.raises(ValueError, match="unsupported config format"):
        load_config(tmp_path / "c.ini")
