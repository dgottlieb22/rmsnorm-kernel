"""RMSNorm implementations, from a PyTorch baseline to Triton kernels.

``IMPLEMENTATIONS`` maps a stable name to a callable ``f(x, weight, eps)``;
``RESIDUAL_IMPLEMENTATIONS`` maps names to ``f(x, residual, weight, eps)``
returning ``(y, new_residual)``. Tests and the benchmark iterate over these
registries so adding a variant is one line here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from . import torch_impls, triton_impls
from .reference import rmsnorm_reference, rmsnorm_residual_reference


@dataclass(frozen=True)
class Impl:
    name: str
    fn: Callable
    family: str  # "torch" | "compile" | "triton" | "vendor"
    description: str

    @property
    def needs_triton(self) -> bool:
        return self.family == "triton"


IMPLEMENTATIONS: Dict[str, Impl] = {
    i.name: i
    for i in [
        Impl("torch_eager", torch_impls.rmsnorm_torch_eager, "torch",
             "float32 upcast, separate eager kernels, one rounding at the end"),
        Impl("torch_eager_hf", torch_impls.rmsnorm_torch_eager_hf, "torch",
             "Hugging Face LlamaRMSNorm ordering: round, then multiply by weight"),
        Impl("torch_rms_norm", torch_impls.rmsnorm_torch_rms_norm, "torch",
             "torch.nn.functional.rms_norm"),
        Impl("torch_compiled", torch_impls.rmsnorm_torch_compiled, "compile",
             "torch.compile of torch_eager (Inductor-generated kernel)"),
        Impl("triton_v0_naive", triton_impls.rmsnorm_triton_v0_naive, "triton",
             "one row per program, whole row in one block, 4 warps"),
        Impl("triton_v1_warps", triton_impls.rmsnorm_triton_v1_warps, "triton",
             "v0 with num_warps scaled to about 8 elements per thread"),
        Impl("triton_v2_looped", triton_impls.rmsnorm_triton_v2_looped, "triton",
             "two passes over the row in fixed chunks; bounded registers"),
        Impl("triton_v3_multirow", triton_impls.rmsnorm_triton_v3_multirow, "triton",
             "several rows per program; weight loaded once per program"),
        Impl("triton_v3b_wtile", triton_impls.rmsnorm_triton_v3b_wtile, "triton",
             "v3 with the weight read as a tile: no shared-memory layout change"),
        Impl("triton", triton_impls.rmsnorm_triton, "triton",
             "dispatcher: picks a variant by row length"),
    ]
}

RESIDUAL_IMPLEMENTATIONS: Dict[str, Impl] = {
    i.name: i
    for i in [
        Impl("torch_eager_residual", torch_impls.rmsnorm_torch_eager_residual, "torch",
             "eager add, then torch_eager"),
        Impl("torch_compiled_residual", torch_impls.rmsnorm_torch_compiled_residual, "compile",
             "torch.compile of the add-then-norm sequence"),
        Impl("triton_v4_fused_residual", triton_impls.rmsnorm_triton_v4_fused_residual, "triton",
             "residual add fused into the row kernel; writes both outputs"),
    ]
}


def _register_external() -> None:
    """Add vendor implementations (vLLM, Liger, flash-attn) when their packages import."""
    from . import external

    plain, residual = external.discover()
    for name, fn, desc in plain.values():
        IMPLEMENTATIONS[name] = Impl(name, fn, "vendor", desc)
    for name, fn, desc in residual.values():
        RESIDUAL_IMPLEMENTATIONS[name] = Impl(name, fn, "vendor", desc)


_register_external()

__all__ = [
    "IMPLEMENTATIONS",
    "RESIDUAL_IMPLEMENTATIONS",
    "Impl",
    "rmsnorm_reference",
    "rmsnorm_residual_reference",
]
