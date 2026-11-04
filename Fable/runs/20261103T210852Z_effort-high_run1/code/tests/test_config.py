import json

import pytest

from fable_pipeline.config import PipelineConfig


def test_disable_and_roundtrip(tmp_path):
    cfg = PipelineConfig()
    cfg.disable("boilerplate", "tables", "ocr", "columns", "llm")
    assert not cfg.boilerplate.enabled and not cfg.tables.enabled
    assert not cfg.figures.ocr_enabled and not cfg.ingestion.detect_columns
    d = cfg.to_dict()
    assert d["boilerplate"]["enabled"] is False
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.to_dict() == d
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"figures": {"ocr_lang": "deu"}, "fail_fast": True}), encoding="utf-8")
    cfg3 = PipelineConfig.from_json_file(str(path))
    assert cfg3.figures.ocr_lang == "deu" and cfg3.fail_fast is True
    assert cfg3.tables.enabled is True  # untouched defaults


def test_unknown_disable_name_raises():
    with pytest.raises(ValueError):
        PipelineConfig().disable("nonsense")


def test_enable_llm_sets_flags():
    cfg = PipelineConfig().enable_llm()
    assert cfg.figures.describe_with_llm and cfg.tables.use_llm
    assert cfg.ollama.temperature == 0.0 and isinstance(cfg.ollama.seed, int)
