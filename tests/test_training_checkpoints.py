from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from ae_research.training.trainer import Trainer


def test_save_resume_checkpoints_writes_best_and_last(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.model = SimpleNamespace(decoder=torch.nn.Linear(2, 1))
    trainer.optimizer = torch.optim.AdamW(trainer.model.decoder.parameters())
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
