"""Shared test setup.

Without a CUDA device the Triton kernels run through Triton's interpreter
(``TRITON_INTERPRET=1``), which executes each program in numpy on the CPU.
The variable has to be set before ``triton`` is first imported, which is why
it is set here, before any ``rmsnorm`` import. Interpreter mode checks what
the kernels compute, not how fast; timing tests only run on a GPU.
"""
from __future__ import annotations

import os

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

import rmsnorm  # noqa: E402  (imports triton after the variable above is set)
from rmsnorm import torch_impls  # noqa: E402

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"


def interpreting() -> bool:
    """True when Triton kernels run through the CPU interpreter."""
    return os.environ.get("TRITON_INTERPRET") == "1"


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_gpu)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """A variant that does not cover a shape (``UnsupportedShape``) is a skip, not a failure.

    The row-loop kernel stops at N = 1024 by design; the registry-wide
    correctness matrix still runs it on every shape it does cover, and the
    skip reason names the shape it does not.
    """
    try:
        return (yield)
    except rmsnorm.triton_impls.UnsupportedShape as e:
        pytest.skip(f"variant does not cover this shape: {e}")


@pytest.fixture(scope="session")
def device() -> str:
    return DEVICE


@pytest.fixture(scope="session")
def compile_available() -> bool:
    """Whether ``torch.compile`` can build a kernel here.

    Inductor's CPU backend needs a C++ toolchain with the Python headers; a
    machine without them raises on the first compiled call. Rather than
    fail every compile test with the same traceback, probe once and skip
    them with the reason.
    """
    try:
        x = torch.randn(2, 8, device=DEVICE)
        w = torch.ones(8, device=DEVICE)
        torch_impls.rmsnorm_torch_compiled(x, w, 1e-6)
        return True
    except Exception as e:  # noqa: BLE001 - any build failure means "not here"
        pytest.skip(f"torch.compile unavailable on this host: {type(e).__name__}")
        return False


def impl_items(registry):
    """Registry entries as pytest params, skipping Triton ones when triton is missing."""
    params = []
    for impl in registry.values():
        marks = []
        if impl.needs_triton and not rmsnorm.triton_impls.HAS_TRITON:
            marks.append(pytest.mark.skip(reason="triton not installed"))
        params.append(pytest.param(impl, id=impl.name, marks=marks))
    return params
