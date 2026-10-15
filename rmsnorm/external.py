"""Vendor RMSNorm implementations, registered when their package is importable.

These are the kernels serving stacks ship, and the comparison the hand-written
kernels are measured against. None of them is a dependency;
each adapter is attempted at import and silently skipped when the package is
missing. The adapters were written against the public signatures of the
versions named below and have not been run on this machine (no GPU, packages
not installed); the benchmark's correctness gate runs each one against the
oracle before timing it, so a signature or semantics mismatch surfaces as an
ERROR row in the results, never as a number.

* vLLM ``_custom_ops.rms_norm`` / ``fused_add_rms_norm`` (CUDA, in place for
  the fused variant).
* Liger Kernel ``liger_rms_norm`` (Triton; ``casting_mode="llama"`` rounds the
  normalized value before the weight multiply, the HF ordering).
* flash-attn ``rms_norm_fn`` (Triton, supports a fused residual).
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

Adapter = Tuple[str, Callable, str]  # name, fn, description


def _vllm() -> tuple[list[Adapter], list[Adapter]]:
    from vllm import _custom_ops as ops  # type: ignore

    def rms_norm(x, w, eps):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty_like(x2)
        ops.rms_norm(out, x2, w.to(x.dtype).contiguous(), eps)
        return out.view(x.shape)

    def fused(x, r, w, eps):
        # vLLM's kernel is in place: x becomes y and r becomes x + r. Work on
        # copies so the caller's tensors survive; the benchmark notes that the
        # two copies are part of what it times for this entry.
        x2 = x.reshape(-1, x.shape[-1]).clone()
        r2 = r.reshape(-1, r.shape[-1]).clone()
        ops.fused_add_rms_norm(x2, r2, w.to(x.dtype).contiguous(), eps)
        return x2.view(x.shape), r2.view(x.shape)

    return ([("vllm_rms_norm", rms_norm, "vLLM CUDA rms_norm")],
            [("vllm_fused_add_rms_norm", fused, "vLLM CUDA fused_add_rms_norm (in place; timed with two input copies)")])


def _liger() -> tuple[list[Adapter], list[Adapter]]:
    from liger_kernel.transformers.functional import liger_rms_norm  # type: ignore

    def rms_norm(x, w, eps):
        return liger_rms_norm(x, w.to(x.dtype), eps)

    return [("liger_rms_norm", rms_norm, "Liger Kernel Triton RMSNorm (HF ordering)")], []


def _flash_attn() -> tuple[list[Adapter], list[Adapter]]:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn  # type: ignore

    def rms_norm(x, w, eps):
        return rms_norm_fn(x, w.to(x.dtype), None, eps=eps)

    def fused(x, r, w, eps):
        y, s = rms_norm_fn(x, w.to(x.dtype), None, residual=r, eps=eps, prenorm=True)
        return y, s

    return ([("flash_attn_rms_norm", rms_norm, "flash-attn Triton rms_norm_fn")],
            [("flash_attn_rms_norm_residual", fused, "flash-attn rms_norm_fn with fused residual")])


def discover() -> tuple[Dict[str, Adapter], Dict[str, Adapter]]:
    """Return ``(rmsnorm adapters, residual adapters)`` for whichever vendor packages import."""
    plain: Dict[str, Adapter] = {}
    residual: Dict[str, Adapter] = {}
    for probe in (_vllm, _liger, _flash_attn):
        try:
            p, r = probe()
        except Exception:  # noqa: BLE001 - missing package or incompatible version: skip
            continue
        for name, fn, desc in p:
            plain[name] = (name, fn, desc)
        for name, fn, desc in r:
            residual[name] = (name, fn, desc)
    return plain, residual
