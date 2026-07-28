from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def _zeroth_power_newton_schulz(
    gradient: torch.Tensor, *, steps: int
) -> torch.Tensor:
    """Approximate the nearest semi-orthogonal matrix used by Muon."""
    if gradient.ndim != 2:
        raise ValueError("Muon is only defined for 2-D parameter matrices")
    value = gradient.float()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.mT
    value = value / value.norm().clamp_min(1e-7)
    # Quintic Newton-Schulz coefficients used by the reference Muon optimizer.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = value @ value.mT
        value = a * value + (b * gram + c * (gram @ gram)) @ value
    if transposed:
        value = value.mT
    return value.to(dtype=gradient.dtype)


class MuonAdamW(Optimizer):
    """One checkpointable optimizer containing Muon and AdamW parameter groups."""

    def __init__(
        self,
        muon_parameters: Iterable[nn.Parameter],
        adamw_parameters: Iterable[nn.Parameter],
        *,
        muon_high_lr_parameters: Iterable[nn.Parameter] = (),
        adamw_high_lr_parameters: Iterable[nn.Parameter] = (),
        high_lr_multiplier: float = 1.0,
        muon_lr: float = 1e-5,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.0,
        muon_ns_steps: int = 5,
        adamw_lr: float = 1e-6,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_weight_decay: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        muon = list(muon_parameters)
        adamw = list(adamw_parameters)
        muon_high_lr = list(muon_high_lr_parameters)
        adamw_high_lr = list(adamw_high_lr_parameters)
        if not muon:
            raise ValueError("Muon parameter group is empty")
        if not adamw:
            raise ValueError("AdamW parameter group is empty")
        if any(parameter.ndim != 2 for parameter in (*muon, *muon_high_lr)):
            raise ValueError("Every Muon parameter must be a 2-D matrix")
        all_groups = (muon, adamw, muon_high_lr, adamw_high_lr)
        all_ids = [id(parameter) for group in all_groups for parameter in group]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Optimizer parameter groups must be disjoint")
        if not math.isfinite(high_lr_multiplier) or high_lr_multiplier < 1.0:
            raise ValueError("high_lr_multiplier must be finite and at least 1")
        if not 0 <= muon_momentum < 1:
            raise ValueError("muon_momentum must be in [0, 1)")
        if muon_ns_steps <= 0:
            raise ValueError("muon_ns_steps must be positive")
        if any(not 0 <= beta < 1 for beta in adamw_betas):
            raise ValueError("AdamW betas must be in [0, 1)")

        groups = [
            {
                "params": muon,
                "algorithm": "muon",
                "name": "muon",
                "lr": float(muon_lr),
                "momentum": float(muon_momentum),
                "weight_decay": float(muon_weight_decay),
                "ns_steps": int(muon_ns_steps),
            },
            {
                "params": adamw,
                "algorithm": "adamw",
                "name": "adamw",
                "lr": float(adamw_lr),
                "betas": tuple(float(value) for value in adamw_betas),
                "weight_decay": float(adamw_weight_decay),
                "eps": float(eps),
            },
        ]
        if muon_high_lr:
            groups.append(
                {
                    "params": muon_high_lr,
                    "algorithm": "muon",
                    "name": "muon_zero_init",
                    "lr": float(muon_lr) * float(high_lr_multiplier),
                    "momentum": float(muon_momentum),
                    "weight_decay": float(muon_weight_decay),
                    "ns_steps": int(muon_ns_steps),
                }
            )
        if adamw_high_lr:
            groups.append(
                {
                    "params": adamw_high_lr,
                    "algorithm": "adamw",
                    "name": "adamw_zero_init",
                    "lr": float(adamw_lr) * float(high_lr_multiplier),
                    "betas": tuple(float(value) for value in adamw_betas),
                    "weight_decay": float(adamw_weight_decay),
                    "eps": float(eps),
                }
            )
        super().__init__(groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001, ANN201
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._step_muon(group)
            elif group["algorithm"] == "adamw":
                self._step_adamw(group)
            else:  # pragma: no cover - guarded by construction/state loading
                raise RuntimeError(f"Unknown optimizer algorithm: {group['algorithm']}")
        return loss

    def _step_muon(self, group: dict[str, Any]) -> None:
        learning_rate = float(group["lr"])
        momentum = float(group["momentum"])
        weight_decay = float(group["weight_decay"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(gradient)
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(gradient)
            update = gradient.add(buffer, alpha=momentum)
            update = _zeroth_power_newton_schulz(
                update, steps=int(group["ns_steps"])
            )
            # Rectangular matrices receive the standard Muon aspect-ratio scale.
            update.mul_(math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1])))
            if weight_decay:
                parameter.mul_(1.0 - learning_rate * weight_decay)
            parameter.add_(update, alpha=-learning_rate)

    def _step_adamw(self, group: dict[str, Any]) -> None:
        learning_rate = float(group["lr"])
        beta1, beta2 = group["betas"]
        weight_decay = float(group["weight_decay"])
        eps = float(group["eps"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            if weight_decay:
                parameter.mul_(1.0 - learning_rate * weight_decay)
            parameter.addcdiv_(
                exp_avg,
                denominator,
                value=-learning_rate / bias_correction1,
            )


def build_muon_adamw(
    model: nn.Module, config: dict[str, Any]
) -> tuple[MuonAdamW, dict[str, tuple[str, ...]]]:
    """Build the paper-specified name-aware hybrid optimizer split."""
    if not hasattr(model, "muon_parameters"):
        raise TypeError("Model must expose muon_parameters()")
    requested = list(model.muon_parameters())
    muon_ids = {id(parameter) for parameter in requested}
    if len(muon_ids) != len(requested):
        raise ValueError("muon_parameters() returned duplicate parameters")

    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    known_ids = {id(parameter) for _, parameter in named_trainable}
    unknown = muon_ids.difference(known_ids)
    if unknown:
        raise ValueError("muon_parameters() returned values not owned by the model")
    high_lr_multiplier = float(config.get("zero_init_lr_multiplier", 1.0))
    high_lr_requested = []
    if high_lr_multiplier > 1.0 and hasattr(model, "high_lr_parameters"):
        high_lr_requested = list(model.high_lr_parameters())
    high_lr_ids = {id(parameter) for parameter in high_lr_requested}
    if len(high_lr_ids) != len(high_lr_requested):
        raise ValueError("high_lr_parameters() returned duplicate parameters")
    if high_lr_ids.difference(known_ids):
        raise ValueError("high_lr_parameters() returned values not owned by the model")
    muon_named = [
        (name, parameter)
        for name, parameter in named_trainable
        if id(parameter) in muon_ids and id(parameter) not in high_lr_ids
    ]
    adamw_named = [
        (name, parameter)
        for name, parameter in named_trainable
        if id(parameter) not in muon_ids and id(parameter) not in high_lr_ids
    ]
    muon_high_lr_named = [
        (name, parameter)
        for name, parameter in named_trainable
        if id(parameter) in muon_ids and id(parameter) in high_lr_ids
    ]
    adamw_high_lr_named = [
        (name, parameter)
        for name, parameter in named_trainable
        if id(parameter) not in muon_ids and id(parameter) in high_lr_ids
    ]
    muon_config = config["muon"]
    adamw_config = config["adamw"]
    optimizer = MuonAdamW(
        (parameter for _, parameter in muon_named),
        (parameter for _, parameter in adamw_named),
        muon_high_lr_parameters=(
            parameter for _, parameter in muon_high_lr_named
        ),
        adamw_high_lr_parameters=(
            parameter for _, parameter in adamw_high_lr_named
        ),
        high_lr_multiplier=high_lr_multiplier,
        muon_lr=float(muon_config.get("lr", 1e-5)),
        muon_momentum=float(muon_config.get("momentum", 0.95)),
        muon_weight_decay=float(muon_config.get("weight_decay", 0.0)),
        muon_ns_steps=int(muon_config.get("ns_steps", 5)),
        adamw_lr=float(adamw_config.get("lr", 1e-6)),
        adamw_betas=tuple(adamw_config.get("betas", (0.9, 0.95))),
        adamw_weight_decay=float(adamw_config.get("weight_decay", 0.01)),
        eps=float(adamw_config.get("eps", 1e-8)),
    )
    names = {
        "muon": tuple(name for name, _ in muon_named),
        "adamw": tuple(name for name, _ in adamw_named),
        "muon_zero_init": tuple(name for name, _ in muon_high_lr_named),
        "adamw_zero_init": tuple(name for name, _ in adamw_high_lr_named),
    }
    return optimizer, names


__all__ = ["MuonAdamW", "build_muon_adamw"]
