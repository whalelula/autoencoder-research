from __future__ import annotations

import os

import torch


# Minimal helper adapted from Stability-AI/stable-audio-tools
# stable_audio_tools/models/utils.py.

try:
    torch._dynamo.config.cache_size_limit = max(
        64, torch._dynamo.config.cache_size_limit
    )
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"


def compile(function, *args, **kwargs):
    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function
    return function

