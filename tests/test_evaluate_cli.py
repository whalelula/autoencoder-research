from __future__ import annotations

from argparse import Namespace

import pytest

from ae_research.cli.evaluate import (
    _apply_checkpoint_overrides,
    _is_checkpoint_model,
    _is_sao_model,
    _normalise_models,
    _same_model_names,
)


def test_unified_evaluate_model_dispatch_helpers():
    assert _same_model_names(["same"]) == ("same-s", "same-l")
    assert _same_model_names(["same-l"]) == ("same-l",)
    assert _same_model_names(["same-s", "same-l"]) == ("same-s", "same-l")
    assert _same_model_names(["sao"]) is None
    assert _is_sao_model(["sao"])
    assert _is_checkpoint_model(["dam_mert330"])


def test_evaluate_defaults_to_checkpoint_when_config_or_checkpoint_is_present():
    args = Namespace(config="configs/base.yaml", checkpoint=None)
    assert _normalise_models(None, args) == ["checkpoint"]


def test_evaluate_requires_model_without_checkpoint_inputs():
    args = Namespace(config=None, checkpoint=None)
    with pytest.raises(SystemExit, match="requires --model"):
        _normalise_models(None, args)


def test_checkpoint_cli_overrides_do_not_mutate_source_config():
    source = {
        "data": {
            "sample_rate": 24_000,
            "duration_seconds": 5.0,
            "channels": 1,
            "num_workers": 4,
            "pin_memory": True,
        },
        "evaluation": {
            "output_dir": "outputs/evaluation",
            "batch_size": 4,
            "export_audio": True,
        },
    }
    args = Namespace(
        output_dir="outputs/evaluation/override",
        batch_size=2,
        max_batches=3,
        no_export_audio=True,
        sample_rate=48_000,
        duration_seconds=10.0,
        channels=2,
        num_workers=0,
        no_pin_memory=True,
    )

    updated = _apply_checkpoint_overrides(source, args)

    assert source["evaluation"]["output_dir"] == "outputs/evaluation"
    assert updated["evaluation"]["output_dir"] == "outputs/evaluation/override"
    assert updated["evaluation"]["batch_size"] == 2
    assert updated["evaluation"]["max_batches"] == 3
    assert updated["evaluation"]["export_audio"] is False
    assert updated["data"]["sample_rate"] == 48_000
    assert updated["data"]["duration_seconds"] == 10.0
    assert updated["data"]["channels"] == 2
    assert updated["data"]["num_workers"] == 0
    assert updated["data"]["pin_memory"] is False
