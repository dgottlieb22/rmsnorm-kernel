"""PyTorch implementations: the baselines the custom kernels are measured against.

Three eager formulations and the compiled version of the first:

* ``rmsnorm_torch_eager``: upcast to float32, normalize, multiply by the
  weight in float32, round to the input dtype once. This is also what
  ``torch.nn.functional.rms_norm`` computes (ATen's composite implementation
  does the weight multiply in the upcast dtype and casts at the end).
* ``rmsnorm_torch_eager_hf``: the formulation in Hugging Face's
  ``LlamaRMSNorm``: round the normalized activations back to the input dtype
  first, then multiply by the (bf16) weight. Two roundings instead of one;
  the numerics section of the README measures the difference.
* ``rmsnorm_torch_rms_norm``: ``torch.nn.functional.rms_norm`` (torch >= 2.4).
* ``rmsnorm_torch_compiled``: ``torch.compile`` of the eager function. On
  CUDA, TorchInductor emits its own Triton kernel for this graph, so the
  comparison against the hand-written kernels is Triton against Triton.

Every function takes ``x`` of shape ``(..., N)``, ``weight`` of shape ``(N,)``
and returns a tensor with x's shape and dtype. Weights may be in a different
dtype from the activations (float32 weights with bf16 activations are common);
the output dtype always follows ``x``.
"""
from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn.functional as F

from .reference import _check_shapes


def rmsnorm_torch_eager(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Upcast, normalize and scale in float32, round once at the end.

    For a bf16 input this is a chain of separate elementwise and reduction
    kernels (cast, square, mean, add, rsqrt, two multiplies, cast); the
    profiler section of the README counts them on real hardware. Every
    intermediate is a full-size float32 tensor, so eager mode moves several
    times the data the operation needs.
    """
    _check_shapes(x, weight)
    h = x.to(torch.float32)
    mean_sq = h.pow(2).mean(dim=-1, keepdim=True)
    h = h * torch.rsqrt(mean_sq + eps)
    return (h * weight.to(torch.float32)).to(x.dtype)


def rmsnorm_torch_eager_hf(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Hugging Face ``LlamaRMSNorm`` ordering: round, then multiply by the weight.

    ``weight * hidden_states.to(input_dtype)`` rounds the normalized value to
    bf16 before the weight multiply, so the result carries two roundings. HF
    keeps the weight in the model dtype; this version casts it to x's dtype to
    keep the output dtype equal to the input dtype when they differ.
    """
    _check_shapes(x, weight)
    input_dtype = x.dtype
    h = x.to(torch.float32)
    mean_sq = h.pow(2).mean(dim=-1, keepdim=True)
    h = h * torch.rsqrt(mean_sq + eps)
    return weight.to(input_dtype) * h.to(input_dtype)


def rmsnorm_torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``torch.nn.functional.rms_norm``: the library's own implementation."""
    _check_shapes(x, weight)
    # F.rms_norm promotes the output to the weight dtype when it is wider than
    # x; keep the output dtype tied to x like every other implementation here.
    w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def rmsnorm_torch_eager_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual add then RMSNorm as two eager steps, the way a plain decoder layer does it.

    The sum is formed in the input dtype (one rounding) and written out as the
    new residual stream; the norm then reads it back.
    """
    s = x + residual
    return rmsnorm_torch_eager(s, weight, eps), s


@functools.lru_cache(maxsize=None)
def _compiled(fn: Callable) -> Callable:
    # Dynamo caches one graph per distinct shape; the benchmark matrix has
    # dozens of shapes, so lift the default recompile limit.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    return torch.compile(fn)


def rmsnorm_torch_compiled(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``torch.compile(rmsnorm_torch_eager)``. The first call per shape compiles."""
    return _compiled(rmsnorm_torch_eager)(x, weight, eps)


def rmsnorm_torch_compiled_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.compile`` of the two-step residual version; Inductor fuses the add into the norm."""
    return _compiled(rmsnorm_torch_eager_residual)(x, residual, weight, eps)
