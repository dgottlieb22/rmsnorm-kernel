# rmsnorm-kernels: RMSNorm from a PyTorch baseline to Triton, one change at a time

A GPU performance engineering study of one transformer inference operation,
RMSNorm, taken from a straightforward PyTorch implementation through
`torch.compile` to hand-written Triton kernels. The goal is not a speedup
number. It is to be able to say, for every implementation and every shape,
why it runs as fast as it does.

Status: implementations and correctness tests are complete and run on a
machine without a GPU (the Triton kernels run through Triton's CPU
interpreter). Benchmarks and profiles need a CUDA device and are PENDING
HARDWARE; no performance number in this repository is made up.

Layout:

- `rmsnorm/reference.py`: float64 oracle.
- `rmsnorm/torch_impls.py`: eager, Hugging Face ordering, `F.rms_norm`, `torch.compile`.
- `rmsnorm/triton_impls.py`: Triton kernels v0 to v4 and the dispatcher.
- `rmsnorm/tolerance.py`: dtype-derived tolerances.
- `tests/`: correctness matrix, mutant tests that prove the tolerances have teeth, numerics pins.

Run the tests with `make venv-cpu && make test` (CPU) or `make venv-gpu && make test` (GPU).
