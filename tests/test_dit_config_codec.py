from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from ae_research.diffusion.codec import (  # noqa: E402
    FrozenAutoencoderCodec,
    LatentNormalizer,
)
from ae_research.diffusion.config import (  # noqa: E402
    DEFAULT_DIT_DEPTH,
    load_dit_config,
    validate_dit_config,
)


def _valid_dit_config(tmp_path: Path) -> dict:
    return {
        "data": {
            "root": "future/audio",
            "sample_rate": 24_000,
            "channels": 1,
            "duration_seconds": 5.0,
            "manifest_dir": "does/not/need/to/exist",
            "dit_manifest_dir": "future/dit/manifests",
            "text_embedding_dir": "future/text/embeddings",
            "num_workers": 0,
            "pin_memory": False,
        },
        "autoencoder": {
            "type": "semantic",
            "config_path": "future/ae.yaml",
            "checkpoint_path": "future/best.pt",
        },
        "dit": {
            "model_dim": 64,
            "text_dim": 32,
            "num_heads": 8,
            "mlp_multiplier": 4.0,
            "repa": {"enabled": True, "layer": 5, "loss_weight": 1.0},
        },
        # Direct fields are the backwards-compatible fallback.  The loader
        # normalizes these into conditioning.text_encoder for data tooling.
        "conditioning": {
            "model_name": "google/t5gemma-s-s-ul2",
            "max_length": 64,
            "batch_size": 4,
            "cache_dir": "future/text/embeddings",
            "cross_attention": True,
            "duration_conditioning": True,
        },
        "diffusion": {
            "prediction_type": "x_prediction",
            "t_eps": 1.0e-5,
            "timestep_sampler": {
                "type": "truncated_logit_normal",
                "mean": 0.0,
                "std": 1.0,
                "truncation": 0.075,
                "rescale": True,
                "flip": False,
            },
            "sampling": {
                "steps": 20,
                "guidance_scale": 1.0,
                "guidance_interval": [0.1, 1.0],
            },
        },
        "optimizer": {
            "type": "muon_adamw",
            "muon": {"lr": 1.0e-5, "momentum": 0.95, "ns_steps": 5},
            "adamw": {
                "lr": 1.0e-6,
                "betas": [0.9, 0.95],
                "weight_decay": 0.01,
            },
            "parameter_assignment": "qkv_and_ff",
        },
        "training": {
            "output_dir": str(tmp_path / "run"),
            "epochs": 2,
            "lr_scheduler": "warmup_cosine",
            "warmup_steps": 10,
            "batch_size": 2,
            "num_validation_samples": 5,
            "cfg_enabled": False,
        },
        "evaluation": {
            "output_dir": str(tmp_path / "evaluation"),
            "metrics": ["gfad"],
            "gfad": {"enabled": True},
        },
    }


def test_load_dit_config_defaults_depth_and_normalizes_text_encoder(tmp_path):
    config = _valid_dit_config(tmp_path)
    config_path = tmp_path / "dit.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_dit_config(config_path)

    assert loaded["dit"]["depth"] == DEFAULT_DIT_DEPTH
    assert loaded["conditioning"]["text_encoder"] == {
        "model_name": "google/t5gemma-s-s-ul2",
        "max_length": 64,
        "batch_size": 4,
        "cache_dir": "future/text/embeddings",
        "frozen": True,
    }
    # Validation intentionally does not require referenced paths to exist.
    assert not Path(loaded["autoencoder"]["checkpoint_path"]).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["dit"].update(depth=0), "dit.depth"),
        (lambda c: c["dit"].update(model_dim=65), "divisible"),
        (
            lambda c: c["conditioning"].update(cfg_enabled=True),
            "CFG is disabled",
        ),
        (
            lambda c: c["diffusion"]["timestep_sampler"].update(truncation=0.1),
            "truncation",
        ),
        (
            lambda c: c["optimizer"]["muon"].update(lr=2.0e-5),
            "optimizer.muon.lr",
        ),
        (
            lambda c: c["training"].update(lr_scheduler="inverse_lr"),
            "warmup_cosine",
        ),
        (
            lambda c: c["dit"]["repa"].update(layer=11),
            "smaller than dit.depth",
        ),
        (
            lambda c: c["dit"]["repa"].update(loss_weight=0.0),
            "positive when enabled",
        ),
    ],
)
def test_dit_config_rejects_incompatible_training_choices(
    tmp_path, mutation, message
):
    config = _valid_dit_config(tmp_path)
    config["dit"]["depth"] = DEFAULT_DIT_DEPTH
    config["conditioning"]["text_encoder"] = {
        "model_name": config["conditioning"].pop("model_name"),
        "max_length": config["conditioning"].pop("max_length"),
        "batch_size": config["conditioning"].pop("batch_size"),
        "cache_dir": config["conditioning"].pop("cache_dir"),
        "frozen": True,
    }
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_dit_config(config)


def test_latent_normalizer_round_trip_and_stats_file(tmp_path):
    stats_path = tmp_path / "latent_stats.pt"
    torch.save(
        {
            "channel_mean": torch.tensor([1.0, -2.0, 0.5]),
            "channel_std": torch.tensor([2.0, 4.0, 0.25]),
        },
        stats_path,
    )
    normalizer = LatentNormalizer.from_stats(stats_path, latent_dim=3)
    latent = torch.randn(2, 3, 7)

    restored = normalizer.denormalize(normalizer.normalize(latent))

    assert torch.allclose(restored, latent, atol=1.0e-6, rtol=1.0e-6)
    assert not tuple(normalizer.parameters())
    with pytest.raises(ValueError, match="expected 3"):
        normalizer.normalize(torch.randn(2, 4, 7))


class _DummySemanticEncoder(nn.Module):
    hidden_size = 3

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def preprocess(self, waveform):
        return waveform.mean(dim=1)

    def encode_normalized(self, normalized):
        return torch.stack(
            (normalized, normalized * self.scale, normalized + 3.0), dim=-1
        )


class _DummyDetailAware(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.5))

    def forward(self, normalized, semantic_features):
        del normalized
        return semantic_features + self.offset


class _DummySemanticDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 1, bias=False)
        nn.init.constant_(self.projection.weight, 0.25)
        self.calls = 0
        self.last_features = None

    def forward(self, semantic_features, target_num_samples):
        self.calls += 1
        self.last_features = semantic_features.detach().clone()
        projected = self.projection(semantic_features).transpose(1, 2)
        waveform = projected
        if waveform.shape[-1] < target_num_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_num_samples - waveform.shape[-1])
            )
        else:
            waveform = waveform[..., :target_num_samples]
        return waveform, projected


class _DummySemanticAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _DummySemanticEncoder()
        self.detail_aware = _DummyDetailAware()
        self.decoder = _DummySemanticDecoder()
        self.data_sample_rate = 24_000
        self.native_sample_rate = 24_000

    def _to_native_rate(self, waveform):
        return waveform

    def _from_native_rate(self, waveform, target_num_samples):
        return waveform[..., :target_num_samples]


def test_semantic_codec_uses_modulated_features_and_decoder_forward():
    model = _DummySemanticAutoencoder()
    codec = FrozenAutoencoderCodec.from_model(
        model, kind="semantic", sample_rate=24_000, channels=1
    )
    waveform = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)

    latent = codec.encode_raw(waveform)
    expected_features = model.encoder.encode_normalized(waveform[:, 0]) + 0.5

    assert latent.shape == (1, 3, 6)
    assert torch.equal(latent, expected_features.transpose(1, 2))
    reconstruction = codec.decode_raw(latent, target_num_samples=6)
    assert reconstruction.shape == waveform.shape
    assert model.decoder.calls == 1
    assert torch.equal(model.decoder.last_features, expected_features)
    assert codec.latent_dim == 3
    assert not codec.training
    assert not codec.autoencoder.training
    assert all(not parameter.requires_grad for parameter in codec.parameters())
    codec.train(True)
    assert not codec.training
    assert not codec.autoencoder.training


class _IdentityPretransform(nn.Module):
    channels = 2

    def encode(self, value):
        return value

    def decode(self, value):
        return value


class _IdentityEncoder(nn.Module):
    latent_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 2, 1))

    def forward(self, value):
        return value * self.scale


class _RandomizingBottleneck(nn.Module):
    noise_regularize = True
    noise_augment_dim = 0

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running_std", torch.tensor(2.0))
        self.decode_calls = 0

    def encode(self, value):
        return value / self.running_std

    def decode(self, value):
        self.decode_calls += 1
        return value * self.running_std + torch.randn_like(value)


class _DummySameAutoencoder(nn.Module):
    latent_dim = 2
    output_activation = "none"

    def __init__(self) -> None:
        super().__init__()
        self.pretransform = _IdentityPretransform()
        self.encoder = _IdentityEncoder()
        self.bottleneck = _RandomizingBottleneck()
        self.decoder = nn.Identity()

    @staticmethod
    def _match_length(waveform, target_num_samples):
        return waveform[..., :target_num_samples]


def test_same_codec_decode_is_deterministic_and_bypasses_noise_regularize():
    model = _DummySameAutoencoder()
    codec = FrozenAutoencoderCodec.from_model(
        model, kind="same", sample_rate=44_100, channels=2
    )
    waveform = torch.randn(2, 2, 8)
    latent = codec.encode_raw(waveform)

    torch.manual_seed(1)
    first = codec.decode_raw(latent, target_num_samples=8)
    torch.manual_seed(999)
    second = codec.decode_raw(latent, target_num_samples=8)

    assert torch.equal(first, second)
    assert torch.allclose(first, latent * model.bottleneck.running_std)
    assert model.bottleneck.decode_calls == 0


def _same_ae_config() -> dict:
    return {
        "data": {"sample_rate": 44_100, "channels": 2},
        "model": {"type": "same"},
    }


def test_codec_constructor_loads_checkpoint_and_checks_audio_format(tmp_path):
    config = _same_ae_config()
    config_path = tmp_path / "same.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source = _DummySameAutoencoder()
    source.encoder.scale.data.fill_(3.0)
    checkpoint_path = tmp_path / "same.pt"
    torch.save(
        {"model": source.state_dict(), "config": copy.deepcopy(config)}, checkpoint_path
    )

    def factory(model_config, **kwargs):
        assert model_config["type"] == "same"
        assert kwargs["kind"] == "same"
        return _DummySameAutoencoder()

    codec = FrozenAutoencoderCodec(
        config_path,
        checkpoint_path,
        expected_sample_rate=44_100,
        expected_channels=2,
        model_factory=factory,
    )
    assert codec.latent_dim == 2
    assert torch.equal(codec.autoencoder.encoder.scale, source.encoder.scale)
    with pytest.raises(ValueError, match="sample rate"):
        FrozenAutoencoderCodec(
            config_path,
            checkpoint_path,
            expected_sample_rate=24_000,
            expected_channels=2,
            model_factory=factory,
        )


def test_semantic_constructor_accepts_project_trainer_checkpoint(tmp_path):
    config = {
        "data": {"sample_rate": 24_000, "channels": 1},
        "model": {"type": "semantic_mert_autoencoder"},
    }
    config_path = tmp_path / "semantic.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source = _DummySemanticAutoencoder()
    source.decoder.projection.weight.data.fill_(0.75)
    source.detail_aware.offset.data.fill_(2.0)
    checkpoint_path = tmp_path / "semantic.pt"
    torch.save(
        {
            "decoder": source.decoder.state_dict(),
            "detail_aware": source.detail_aware.state_dict(),
            "config": copy.deepcopy(config),
        },
        checkpoint_path,
    )

    codec = FrozenAutoencoderCodec(
        config_path,
        checkpoint_path,
        model_factory=lambda model_config, **kwargs: _DummySemanticAutoencoder(),
    )

    assert codec.kind == "semantic"
    assert codec.latent_dim == 3
    assert torch.equal(
        codec.autoencoder.decoder.projection.weight, source.decoder.projection.weight
    )
    assert torch.equal(
        codec.autoencoder.detail_aware.offset, source.detail_aware.offset
    )
