from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_config_module():
    path = ROOT / "src" / "ae_research" / "diffusion" / "config.py"
    spec = importlib.util.spec_from_file_location("dit_config_standalone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dit_console_scripts_are_registered():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'ae-dit-train = "ae_research.cli.train_dit:main"' in pyproject
    assert 'ae-dit-evaluate = "ae_research.cli.evaluate_dit:main"' in pyproject
    assert 'ae-prepare-dit-tags = "ae_research.cli.prepare_dit_tags:main"' in pyproject


def test_reference_dit_config_uses_the_canonical_schema():
    config = _load_config_module().load_dit_config(
        ROOT / "configs" / "dit_mert95m_1k_5s.yaml"
    )

    assert config["dit"]["depth"] == 11
    assert config["diffusion"]["prediction_type"] == "x_prediction"
    assert config["conditioning"]["text_encoder"]["cache_dir"]
    assert config["training"]["num_validation_samples"] == 5
    assert config["evaluation"]["sampling_steps"] == 50


def test_trainer_uses_dit_collator_and_evaluator_module():
    trainer = (
        ROOT / "src" / "ae_research" / "diffusion" / "trainer.py"
    ).read_text(encoding="utf-8")

    assert 'getattr(module, "create_dit_dataloader")' in trainer
    assert "batch = collate_fn(items)" in trainer
    assert 'import_module("ae_research.diffusion.evaluation")' in trainer
