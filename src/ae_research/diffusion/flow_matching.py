from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def sample_truncated_logit_normal(
    batch_size: int,
    *,
    device: torch.device | str,
    minimum: float = 0.075,
    rescale: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``sigmoid(N(0, 1))`` samples truncated below ``minimum``.

    Stable Audio 3 describes the retained interval as being rescaled to
    ``[0, 1]``. Setting ``rescale=False`` exposes the unscaled truncated
    samples for ablations.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= minimum < 1.0:
        raise ValueError("minimum must be in [0, 1)")

    result = torch.empty(batch_size, device=device, dtype=torch.float32)
    remaining = torch.arange(batch_size, device=device)
    while remaining.numel():
        candidates = torch.sigmoid(
            torch.randn(
                remaining.numel(),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )
        accepted = candidates >= minimum
        if accepted.any():
            result[remaining[accepted]] = candidates[accepted]
        remaining = remaining[~accepted]
    if rescale:
        result = (result - minimum) / (1.0 - minimum)
    return result.clamp(0.0, 1.0)


def flow_interpolate(
    clean_latent: torch.Tensor,
    noise_latent: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Return ``x_t = (1 - t) * x_clean + t * x_noise``."""
    if clean_latent.shape != noise_latent.shape:
        raise ValueError("clean_latent and noise_latent must have the same shape")
    if timestep.shape != (clean_latent.shape[0],):
        raise ValueError("timestep must have shape [batch]")
    expanded = timestep.reshape(-1, *([1] * (clean_latent.ndim - 1)))
    return (1.0 - expanded) * clean_latent + expanded * noise_latent


def x_prediction_to_velocity(
    noisy_latent: torch.Tensor,
    x_prediction: torch.Tensor,
    timestep: torch.Tensor,
    *,
    t_eps: float,
) -> torch.Tensor:
    if noisy_latent.shape != x_prediction.shape:
        raise ValueError("noisy_latent and x_prediction must have the same shape")
    if t_eps <= 0:
        raise ValueError("t_eps must be positive")
    expanded = timestep.clamp_min(t_eps).reshape(
        -1, *([1] * (noisy_latent.ndim - 1))
    )
    return (noisy_latent - x_prediction) / expanded


class XPredictionObjective(nn.Module):
    """RAEv2 x-prediction objective with an optional early/base head.

    In the semantic RAE path, the early head is both the REPA head and the
    model used for single-pass internal guidance. It therefore has one loss,
    not two independently added objectives.
    """

    def __init__(self, *, t_eps: float = 1e-5, base_loss_weight: float = 1.0) -> None:
        super().__init__()
        if t_eps <= 0:
            raise ValueError("t_eps must be positive")
        if base_loss_weight < 0:
            raise ValueError("base_loss_weight must be non-negative")
        self.t_eps = float(t_eps)
        self.base_loss_weight = float(base_loss_weight)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        *,
        noisy_latent: torch.Tensor,
        clean_latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_prediction = outputs["x_pred"]
        target_velocity = x_prediction_to_velocity(
            noisy_latent,
            clean_latent,
            timestep,
            t_eps=self.t_eps,
        )
        predicted_velocity = x_prediction_to_velocity(
            noisy_latent,
            x_prediction,
            timestep,
            t_eps=self.t_eps,
        )
        full_loss = (predicted_velocity - target_velocity).square().mean()
        losses = {"x_prediction": full_loss}
        total = full_loss

        base_prediction = outputs.get("base_x_pred")
        if base_prediction is None:
            base_prediction = outputs.get("repa_pred")
        if base_prediction is not None:
            base_velocity = x_prediction_to_velocity(
                noisy_latent,
                base_prediction,
                timestep,
                t_eps=self.t_eps,
            )
            base_loss = (base_velocity - target_velocity).square().mean()
            losses["repa_internal_guidance"] = base_loss
            total = total + self.base_loss_weight * base_loss
        losses["total"] = total
        return losses


def internal_guidance_prediction(
    full_prediction: torch.Tensor,
    base_prediction: torch.Tensor | None,
    *,
    scale: float,
) -> torch.Tensor:
    """RAEv2 single-forward guidance: ``base + scale * (full - base)``."""
    if base_prediction is None or scale == 1.0:
        return full_prediction
    if full_prediction.shape != base_prediction.shape:
        raise ValueError("Full and base predictions must have the same shape")
    return base_prediction + float(scale) * (full_prediction - base_prediction)


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    *,
    shape: Sequence[int],
    duration: torch.Tensor,
    text_embedding: torch.Tensor,
    text_mask: torch.Tensor,
    steps: int,
    t_eps: float,
    guidance_scale: float = 1.0,
    guidance_interval: tuple[float, float] = (0.0, 1.0),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integrate the x-prediction rectified-flow ODE from noise to clean latent."""
    if len(shape) != 3:
        raise ValueError("shape must be [batch, channels, frames]")
    if int(shape[0]) != duration.shape[0]:
        raise ValueError("shape batch dimension must match conditions")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if t_eps <= 0:
        raise ValueError("t_eps must be positive")
    interval_start, interval_end = guidance_interval
    if not 0 <= interval_start <= interval_end <= 1:
        raise ValueError("guidance_interval must lie within [0, 1]")

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:  # pragma: no cover - real DiT always has parameters
        model_dtype = text_embedding.dtype
    state = torch.randn(
        tuple(int(value) for value in shape),
        device=duration.device,
        dtype=model_dtype,
        generator=generator,
    )
    times = torch.linspace(
        1.0, 0.0, steps + 1, device=state.device, dtype=torch.float32
    )
    for index in range(steps):
        current = times[index]
        following = times[index + 1]
        timestep = current.expand(shape[0])
        outputs: Mapping[str, Any] = model(
            state,
            timestep,
            duration=duration,
            text_embedding=text_embedding,
            text_mask=text_mask,
        )
        base_prediction = outputs.get("base_x_pred")
        if base_prediction is None:
            base_prediction = outputs.get("repa_pred")
        use_guidance = interval_start <= float(current) <= interval_end
        prediction = internal_guidance_prediction(
            outputs["x_pred"],
            base_prediction if use_guidance else None,
            scale=guidance_scale,
        )
        velocity = x_prediction_to_velocity(
            state, prediction, timestep, t_eps=t_eps
        )
        state = state + (following - current).to(state.dtype) * velocity
    return state


__all__ = [
    "XPredictionObjective",
    "euler_sample",
    "flow_interpolate",
    "internal_guidance_prediction",
    "sample_truncated_logit_normal",
    "x_prediction_to_velocity",
]
