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
    trainer.best_step = 3
    trainer.listening_indices = [4, 1]
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
    assert checkpoint["best_step"] == 3
    assert checkpoint["listening_indices"] == [4, 1]
    assert "scheduler" in checkpoint


def test_listening_batch_uses_fixed_random_validation_indices():
    trainer = Trainer.__new__(Trainer)
    trainer.config = {"seed": 7}
    trainer.train_config = {"num_listen_samples": 3}
    trainer.val_loader = SimpleNamespace(
        dataset=[
            {
                "audio": torch.full((1, 2), float(index)),
                "track_id": f"track-{index}",
                "path": f"{index}.wav",
            }
            for index in range(10)
        ]
    )

    trainer.listening_indices = trainer._select_listening_indices()
    batch = trainer._listening_batch()

    assert trainer.listening_indices == [5, 2, 6]
    assert batch["track_id"] == ["track-5", "track-2", "track-6"]
    assert batch["audio"][:, 0, 0].tolist() == [5.0, 2.0, 6.0]


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
    source.best_step = 3
    source.listening_indices = [5, 2, 6]
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
    restored.listening_indices = []
    restored.load_checkpoint(checkpoint_path)

    assert restored.global_step == 3
    assert restored.best_step == 3
    assert restored.start_epoch == 3
    assert restored.listening_indices == [5, 2, 6]
    assert restored.scheduler.last_epoch == source.scheduler.last_epoch
    assert restored.scheduler.get_last_lr() == pytest.approx(
        source.scheduler.get_last_lr()
    )


def test_failed_checkpoint_validation_preserves_existing_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer = Trainer.__new__(Trainer)
    trainer.model = SimpleNamespace(decoder=torch.nn.Linear(2, 1))
    trainer.optimizer = torch.optim.AdamW(trainer.model.decoder.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.global_step = 3
    trainer.best_val = 1.25
    trainer.best_step = 3
    trainer.listening_indices = []
    trainer.config = {"training": {}}

    trainer.save_checkpoint(checkpoint_path, epoch=1)

    def fail_validation(self, path):  # noqa: ANN001
        raise RuntimeError(f"bad checkpoint: {path}")

    trainer.global_step = 4
    monkeypatch.setattr(Trainer, "_validate_checkpoint_file", fail_validation)
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        trainer.save_checkpoint(checkpoint_path, epoch=1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_final_checkpoint_validation_restores_existing_checkpoint(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer = Trainer.__new__(Trainer)
    trainer.model = SimpleNamespace(decoder=torch.nn.Linear(2, 1))
    trainer.optimizer = torch.optim.AdamW(trainer.model.decoder.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.global_step = 3
    trainer.best_val = 1.25
    trainer.best_step = 3
    trainer.listening_indices = []
    trainer.config = {"training": {}}

    trainer.save_checkpoint(checkpoint_path, epoch=1)
    trainer.global_step = 4

    import ae_research.training.trainer as trainer_module

    original_replace = trainer_module.os.replace

    def corrupting_replace(src, dst):  # noqa: ANN001
        original_replace(src, dst)
        if str(src).endswith(".tmp") and str(dst).endswith("checkpoint.pt"):
            checkpoint_path.write_bytes(b"incomplete checkpoint")

    monkeypatch.setattr(trainer_module.os, "replace", corrupting_replace)
    with pytest.raises(RuntimeError, match="Checkpoint is not readable after save"):
        trainer.save_checkpoint(checkpoint_path, epoch=1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 3
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.bak"))


def test_save_resume_checkpoints_rewrites_unreadable_best(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_dir.mkdir()
    trainer.model = SimpleNamespace(decoder=torch.nn.Linear(2, 1))
    trainer.optimizer = torch.optim.AdamW(trainer.model.decoder.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.global_step = 7
    trainer.best_val = 0.5
    trainer.best_step = 7
    trainer.listening_indices = []
    trainer.config = {"training": {}}
    (trainer.checkpoint_dir / "best.pt").write_bytes(b"incomplete checkpoint")

    trainer._save_resume_checkpoints(epoch=3)

    best = torch.load(
        trainer.checkpoint_dir / "best.pt", map_location="cpu", weights_only=False
    )
    last = torch.load(
        trainer.checkpoint_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert best["global_step"] == 7
    assert last["global_step"] == 7


def test_early_stopping_uses_optimizer_steps():
    trainer = Trainer.__new__(Trainer)
    trainer.train_config = {"early_stopping_patience_steps": 500}
    trainer.best_step = 1000

    trainer.global_step = 1499
    assert not trainer._early_stopping_reached()

    trainer.global_step = 1500
    assert trainer._early_stopping_reached()
