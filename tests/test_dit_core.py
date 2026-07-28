from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ae_research.diffusion.dit import AudioDiffusionTransformer  # noqa: E402
from ae_research.diffusion.flow_matching import (  # noqa: E402
    XPredictionObjective,
    euler_sample,
    flow_interpolate,
    internal_guidance_prediction,
    resolve_internal_guidance,
    sample_truncated_logit_normal,
)
from ae_research.diffusion.optim import build_muon_adamw  # noqa: E402


def _small_model(*, depth: int = 3, repa_layer: int | None = 1):
    return AudioDiffusionTransformer(
        latent_dim=12,
        model_dim=24,
        text_dim=16,
        depth=depth,
        num_heads=4,
        mlp_multiplier=2.0,
        repa_layer=repa_layer,
    )


def test_dit_shape_depth_and_early_head():
    model = _small_model(depth=4, repa_layer=1)
    output = model(
        torch.randn(2, 12, 9),
        torch.tensor([0.2, 0.8]),
        duration=torch.tensor([5.0, 5.0]),
        text_embedding=torch.randn(2, 6, 16),
        text_mask=torch.tensor(
            [[True, True, True, False, False, False], [True] * 6]
        ),
    )
    assert len(model.blocks) == 4
    assert output["x_pred"].shape == (2, 12, 9)
    assert output["base_x_pred"].shape == (2, 12, 9)


def test_padding_tokens_do_not_affect_cross_attention():
    model = _small_model(depth=1, repa_layer=None).eval()
    with torch.no_grad():
        model.blocks[0].cross_attn.output_projection.weight.normal_()
        model.output_projection.weight.normal_()
    latent = torch.randn(1, 12, 5)
    text = torch.randn(1, 4, 16)
    changed = text.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 1000
    conditions = {
        "duration": torch.tensor([5.0]),
        "text_mask": torch.tensor([[True, True, False, False]]),
    }
    first = model(
        latent, torch.tensor([0.5]), text_embedding=text, **conditions
    )["x_pred"]
    second = model(
        latent, torch.tensor([0.5]), text_embedding=changed, **conditions
    )["x_pred"]
    torch.testing.assert_close(first, second)


def test_x_prediction_loss_matches_velocity_formula():
    clean = torch.tensor([[[2.0]], [[-1.0]]])
    noise = torch.tensor([[[6.0]], [[3.0]]])
    timestep = torch.tensor([0.5, 0.25])
    noisy = flow_interpolate(clean, noise, timestep)
    prediction = clean + torch.tensor([[[1.0]], [[2.0]]])
    objective = XPredictionObjective(t_eps=1e-5, base_loss_weight=0.5)
    losses = objective(
        {"x_pred": prediction, "base_x_pred": clean},
        noisy_latent=noisy,
        clean_latent=clean,
        timestep=timestep,
    )
    expected = ((prediction - clean) / timestep[:, None, None]).square().mean()
    torch.testing.assert_close(losses["x_prediction"], expected)
    torch.testing.assert_close(
        losses["total"], losses["x_prediction"] + 0.5 * losses["repa_internal_guidance"]
    )


def test_internal_guidance_uses_base_to_full_extrapolation():
    base = torch.tensor([1.0, 2.0])
    full = torch.tensor([3.0, 6.0])
    guided = internal_guidance_prediction(full, base, scale=1.5)
    torch.testing.assert_close(guided, base + 1.5 * (full - base))
    assert internal_guidance_prediction(full, base, scale=1.0) is full


class _GuidanceModel(torch.nn.Module):
    def __init__(self, *, return_base: bool) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.return_base = return_base

    def forward(self, value, timestep, **conditions):
        del timestep, conditions
        outputs = {"x_pred": torch.zeros_like(value) + self.anchor}
        if self.return_base:
            outputs["base_x_pred"] = torch.ones_like(value) + self.anchor
        return outputs


def test_euler_sample_really_applies_and_requires_internal_guidance():
    conditions = {
        "shape": (1, 2, 3),
        "duration": torch.tensor([5.0]),
        "text_embedding": torch.zeros(1, 1, 4),
        "text_mask": torch.ones(1, 1, dtype=torch.bool),
        "steps": 1,
        "t_eps": 0.05,
        "internal_guidance_enabled": True,
        "guidance_scale": 2.0,
    }
    guided = euler_sample(_GuidanceModel(return_base=True), **conditions)
    torch.testing.assert_close(guided, torch.full_like(guided, -1.0))
    with pytest.raises(RuntimeError, match="did not return"):
        euler_sample(_GuidanceModel(return_base=False), **conditions)


def test_internal_guidance_config_is_explicit_and_legacy_compatible():
    explicit = resolve_internal_guidance(
        {
            "sampling": {"guidance_scale": 1.0},
            "internal_guidance": {
                "enabled": True,
                "scale": 1.2,
                "interval": [0.1, 0.9],
            },
        }
    )
    assert explicit == {"enabled": True, "scale": 1.2, "interval": (0.1, 0.9)}
    legacy = resolve_internal_guidance(
        {"sampling": {"guidance_scale": 1.1, "guidance_interval": [0.2, 1.0]}}
    )
    assert legacy == {"enabled": True, "scale": 1.1, "interval": (0.2, 1.0)}


def test_truncated_logit_normal_is_rescaled_to_unit_interval():
    generator = torch.Generator().manual_seed(7)
    values = sample_truncated_logit_normal(
        4096, device="cpu", minimum=0.075, rescale=True, generator=generator
    )
    assert torch.all((0 <= values) & (values <= 1))
    assert 0.35 < float(values.mean()) < 0.65


def test_hybrid_optimizer_is_name_aware_and_covers_every_parameter():
    model = _small_model()
    config = {
        "muon": {"lr": 1e-5, "momentum": 0.95},
        "adamw": {"lr": 1e-6, "betas": [0.9, 0.95], "weight_decay": 0.01},
    }
    optimizer, names = build_muon_adamw(model, config)
    all_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(names["muon"]).isdisjoint(names["adamw"])
    assert set(names["muon"]).union(names["adamw"]) == all_names
    assert all(
        "qkv_projection" in name
        or "query_projection" in name
        or "key_value_projection" in name
        or ".ff.in_proj.weight" in name
        or ".ff.out_proj.weight" in name
        for name in names["muon"]
    )
    assert [group["algorithm"] for group in optimizer.param_groups] == [
        "muon",
        "adamw",
    ]


def test_zero_initialized_projections_use_high_lr_groups():
    model = _small_model()
    config = {
        "zero_init_lr_multiplier": 10.0,
        "muon": {"lr": 1e-5, "momentum": 0.95},
        "adamw": {"lr": 1e-6, "betas": [0.9, 0.95], "weight_decay": 0.01},
    }
    optimizer, names = build_muon_adamw(model, config)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["muon_zero_init"]["lr"] == pytest.approx(1e-4)
    assert groups["adamw_zero_init"]["lr"] == pytest.approx(1e-5)
    high_lr_names = set(names["muon_zero_init"]) | set(names["adamw_zero_init"])
    assert "output_projection.weight" in high_lr_names
    assert "repa_head.1.weight" in high_lr_names
    assert "blocks.0.cross_attn.output_projection.weight" in high_lr_names
    all_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert set().union(*(set(group) for group in names.values())) == all_names
