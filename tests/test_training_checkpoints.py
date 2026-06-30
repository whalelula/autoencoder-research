from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from ae_research.training.trainer import (  # noqa: E402
    Trainer,
    _warmup_cosine_multiplier,
)


def test_warmup_cosine_reaches_peak_and_minimum():
    values = [
        _warmup_cosine_multiplier(
            step,
            total_steps=1000,
            warmup_steps=150,
            min_ratio=0.05,
        )
        for step in range(1000)
    ]

    assert values[0] == pytest.approx(1 / 150)
    assert values[149] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.05)
    assert values[:150] == sorted(values[:150])
    assert values[149:] == sorted(values[149:], reverse=True)


def test_save_resume_checkpoints_writes_best_and_last(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.model = SimpleNamespace(decoder=torch.nn.Linear(2, 1))
    trainer.optimizer = torch.optim.AdamW(trainer.model.decoder.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.global_step = 3
    trainer.best_val = 1.25
    trainer.config = {"training": {"output_dir": str(tmp_path)}}

    trainer._save_resume_checkpoints(epoch=2)

    best_path = trainer.checkpoint_dir / "best.pt"
    last_path = trainer.checkpoint_dir / "last.pt"
    assert best_path.is_file()
    assert last_path.is_file()
    assert best_path.stat().st_size > 0
    assert last_path.stat().st_size > 0

    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["global_step"] == 3
    assert checkpoint["best_val"] == 1.25
    assert "scheduler" in checkpoint


def test_load_checkpoint_restores_scheduler_state(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    source = Trainer.__new__(Trainer)
    source.model = SimpleNamespace(decoder=torch.nn.Linear(2, 1))
    source.optimizer = torch.optim.AdamW(source.model.decoder.parameters(), lr=2e-4)
    source.scheduler = torch.optim.lr_scheduler.LambdaLR(
        source.optimizer, lambda step: (step + 1) / 10
    )
    source.scaler = torch.amp.GradScaler("cuda", enabled=False)
    source.global_step = 3
    source.best_val = 0.75
    source.config = {"training": {}}
    for _ in range(source.global_step):
        source.optimizer.step()
        source.scheduler.step()
    source.save_checkpoint(checkpoint_path, epoch=2)

    restored = Trainer.__new__(Trainer)
    restored.model = SimpleNamespace(decoder=torch.nn.Linear(2, 1))
    restored.optimizer = torch.optim.AdamW(
        restored.model.decoder.parameters(), lr=2e-4
    )
    restored.scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored.optimizer, lambda step: (step + 1) / 10
    )
    restored.scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored.load_checkpoint(checkpoint_path)

    assert restored.global_step == 3
    assert restored.start_epoch == 3
    assert restored.scheduler.last_epoch == source.scheduler.last_epoch
    assert restored.scheduler.get_last_lr() == pytest.approx(
        source.scheduler.get_last_lr()
    )
