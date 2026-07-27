from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from ae_research.diffusion.trainer import (  # noqa: E402
    DiTTrainer,
    DiffusionHistoryWriter,
    _warmup_cosine_multiplier,
    atomic_save_checkpoint,
    build_checkpoint_state,
    plot_diffusion_history,
    select_fixed_listening_indices,
)


class _PromptDataset:
    def __init__(self) -> None:
        self.records = [
            {"text": "rock, guitar"},
            {"text": "rock, guitar"},
            {"text": "jazz, piano"},
            {"text": "ambient, synthesizer"},
            {"text": "classical, strings"},
            {"text": "folk, acoustic guitar"},
            {"text": "electronic, energetic"},
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return self.records[index]


def test_shared_warmup_cosine_multiplier_preserves_lr_ratio():
    muon_base = 1e-5
    adamw_base = 1e-6
    multipliers = [
        _warmup_cosine_multiplier(
            step,
            total_steps=20,
            warmup_steps=4,
            min_ratio=0.1,
        )
        for step in range(20)
    ]
    assert multipliers[:4] == [0.25, 0.5, 0.75, 1.0]
    assert math.isclose(multipliers[-1], 0.1)
    for multiplier in multipliers:
        assert math.isclose(
            (muon_base * multiplier) / (adamw_base * multiplier), 10.0
        )


def test_fixed_listening_indices_are_deterministic_and_text_unique():
    dataset = _PromptDataset()
    first = select_fixed_listening_indices(dataset, count=5, seed=123)
    second = select_fixed_listening_indices(dataset, count=5, seed=123)
    assert first == second
    assert len(first) == len(set(first)) == 5
    prompts = [dataset.records[index]["text"] for index in first]
    assert len(set(prompts)) == 5


def test_plot_diffusion_history_writes_loss_curve(tmp_path):
    history_path = tmp_path / "dit_history.csv"
    output_path = tmp_path / "loss_curves.png"
    history = DiffusionHistoryWriter(history_path)
    history.append(
        split="train",
        epoch=1,
        step=20,
        metrics={
            "total": 1.0,
            "x_prediction": 0.75,
            "repa_internal_guidance": 0.25,
        },
    )
    history.append(
        split="val",
        epoch=1,
        step=20,
        metrics={
            "total": 1.1,
            "x_prediction": 0.8,
            "repa_internal_guidance": 0.3,
        },
    )

    plot_diffusion_history(history_path, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_checkpoint_state_contains_full_resume_and_fixed_listening_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = build_checkpoint_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config={"dit": {"model_dim": 8}},
        epoch=3,
        batch_in_epoch=7,
        epoch_complete=False,
        global_step=19,
        best_val=0.25,
        best_step=18,
        listening_indices=[1, 3, 5, 7, 9],
        listening_seeds={"selection": 11, "noise": 12, "validation": 13},
        last_validation_step=18,
        optimizer_parameter_names={"adamw": ("weight", "bias")},
    )
    path = tmp_path / "checkpoints" / "step_00000019.pt"
    atomic_save_checkpoint(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["global_step"] == 19
    assert loaded["batch_in_epoch"] == 7
    assert loaded["epoch_complete"] is False
    assert loaded["listening_indices"] == [1, 3, 5, 7, 9]
    assert loaded["listening_seeds"]["noise"] == 12
    assert {"model", "optimizer", "scheduler", "scaler", "config", "rng_state"} <= set(
        loaded
    )


def test_interrupted_checkpoints_save_last_without_overwriting_best(tmp_path):
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.checkpoint_dir = tmp_path
    state = {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
        "config": {},
        "epoch": 1,
        "global_step": 7,
        "listening_indices": [],
        "listening_seeds": {},
    }
    trainer._checkpoint_state = lambda *, epoch_complete: {
        **state,
        "epoch_complete": epoch_complete,
    }
    best_path = tmp_path / "best.pt"
    best_path.write_bytes(b"existing validated best")

    trainer.save_interrupted_checkpoints()

    assert best_path.read_bytes() == b"existing validated best"
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert last["global_step"] == 7
    assert last["epoch_complete"] is False

    best_path.unlink()
    trainer.save_interrupted_checkpoints()
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert best["global_step"] == 7
    assert best["epoch_complete"] is False
