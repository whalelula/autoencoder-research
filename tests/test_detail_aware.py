from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

from torch import nn  # noqa: E402

from ae_research.models.detail_aware import AudioDetailAwareModule  # noqa: E402


class DummyMERTFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(1, 4, kernel_size=2, stride=2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.convolution(waveform.unsqueeze(1))


class DummyMERTFeatureProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 8)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def make_module() -> tuple[
    AudioDetailAwareModule,
    DummyMERTFeatureExtractor,
    DummyMERTFeatureProjection,
]:
    extractor = DummyMERTFeatureExtractor()
    projection = DummyMERTFeatureProjection()
    module = AudioDetailAwareModule(
        feature_extractor=extractor,
        feature_projection=projection,
        dim=8,
        depth=2,
        num_heads=2,
        cross_attention=True,
    )
    return module, extractor, projection


def test_detail_aware_shape_and_zero_initialized_sft():
    module, _, _ = make_module()
    waveform = torch.randn(2, 16)
    semantic_features = torch.randn(2, 8, 8)

    output = module(waveform, semantic_features)

    assert output.shape == semantic_features.shape
    assert torch.count_nonzero(module.fusion_projection.weight) == 0
    assert torch.count_nonzero(module.fusion_projection.bias) == 0
    assert torch.equal(output, semantic_features)


def test_detail_frontend_is_an_independent_trainable_copy():
    module, extractor, projection = make_module()

    assert module.feature_extractor is not extractor
    assert module.feature_projection is not projection
    assert all(parameter.requires_grad for parameter in module.feature_extractor.parameters())
    assert all(parameter.requires_grad for parameter in module.feature_projection.parameters())


def test_detail_and_semantic_token_counts_must_match():
    module, _, _ = make_module()

    with pytest.raises(ValueError, match="token grids must align"):
        module(torch.randn(2, 16), torch.randn(2, 7, 8))


def test_zero_initialized_fusion_head_receives_gradient():
    module, _, _ = make_module()
    output = module(torch.randn(2, 16), torch.randn(2, 8, 8))
    target = torch.randn_like(output)

    (output - target).square().mean().backward()

    assert module.fusion_projection.weight.grad is not None
    assert torch.isfinite(module.fusion_projection.weight.grad).all()
    assert torch.count_nonzero(module.fusion_projection.weight.grad) > 0
