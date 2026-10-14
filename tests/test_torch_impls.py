"""Numerics of the PyTorch baselines that the README's numerics section relies on."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rmsnorm import torch_impls

from test_correctness import make_inputs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_library_rms_norm_rounds_once(dtype, device):
    """``F.rms_norm`` multiplies by the weight before rounding to the input dtype.

    Observed bit-for-bit on CPU with torch 2.8.0. On CUDA a fused kernel may
    order the reduction differently, so there the check is within tolerance;
    the ordering claim itself is only asserted exactly on CPU.
    """
    x, w, eps = make_inputs(64, 4096, dtype, device, seed=11)
    lib = F.rms_norm(x, (4096,), w, eps)
    once = torch_impls.rmsnorm_torch_eager(x, w, eps)
    if device == "cpu":
        assert torch.equal(lib, once)
    else:
        assert torch.allclose(lib.float(), once.float(), rtol=2 * torch.finfo(dtype).eps, atol=2 * torch.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_hf_ordering_differs_in_a_quarter_of_elements(dtype, device):
    """Rounding before the weight multiply changes a sizeable fraction of outputs.

    Both orderings are within tolerance of the oracle (test_tolerance.py); this
    test pins that they are not the same function, so a kernel claiming to
    match Hugging Face bit for bit has to use the HF ordering.
    """
    x, w, eps = make_inputs(64, 4096, dtype, device, seed=12)
    once = torch_impls.rmsnorm_torch_eager(x, w, eps)
    twice = torch_impls.rmsnorm_torch_eager_hf(x, w, eps)
    frac = (once != twice).float().mean().item()
    assert 0.1 < frac < 0.5, f"{frac:.3f} of elements differ"
