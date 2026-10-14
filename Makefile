PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: venv-cpu venv-gpu test test-fast bench-quick bench-full report clean

# CPU-only development environment (Triton interpreter runs the kernels).
venv-cpu:
	python3 -m venv .venv
	$(PIP) install --quiet -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
	$(PIP) install --quiet -e .

venv-gpu:
	python3 -m venv .venv
	$(PIP) install --quiet -r requirements-gpu.txt
	$(PIP) install --quiet -e .

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

# Benchmarks only make sense on a GPU host; the CLI refuses to time on CPU.
bench-quick:
	$(PY) -m rmsnorm.cli bench --matrix quick --out results

bench-full:
	$(PY) -m rmsnorm.cli bench --matrix full --out results

report:
	$(PY) -m rmsnorm.cli report $$(ls -d results/* | tail -n 1)

clean:
	rm -rf .pytest_cache rmsnorm/__pycache__ rmsnorm/*/__pycache__ tests/__pycache__
