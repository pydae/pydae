# AGENTS.md

See `CLAUDE.md` for full technical documentation.

## Workflow & Commands

*   **Setup**: `uv sync --all-packages`. Never `pip install`.
*   **Testing**: `uv run pytest`.
    *   Fast (no compiler): `uv run pytest -m "parse or symbolic or codegen"`
    *   Full (requires GCC/MinGW): `uv run pytest -m "build or model"`
    *   Single package: `uv run pytest tests/core/`, `uv run pytest tests/bps/`
    *   By name: `uv run pytest -k "test_name"`
*   **Linting**: `uv run ruff check .`

## Architecture

*   **Monorepo**: `packages/pydae-core` (PyPI: `pydae`), `packages/pydae-bps`, `packages/pydae-uds`.
*   **Namespace packages**: `packages/*/src/pydae/` has NO `__init__.py`.
*   **Pipeline**: `Builder(sys_dict)` → `build()` → `Model.load()`.
*   **Sparse solvers**: `sparse=False` (dense LU), `sparse='klu'` (SuiteSparse), `sparse='pardiso'` (Intel MKL). KLU and MKL require conda-installed libs.
*   **Parallel codegen**: `PYDAE_PARALLEL=1` for > 200 expressions.

## Conventions

*   **ini/run swap**: Swap variable values in-place at the same index in `y_ini` — never delete and re-append.
*   **LaTeX docstrings**: Use raw strings `r"""..."""`.
*   **Component tests**: `def test():` at bottom of component modules, verified against sibling `.hjson`.
*   **Cross-platform**: Use `pathlib.Path` or `/` for all paths; no hard-coded backslashes.
*   **Line endings**: Force LF via `.gitattributes`. Do not override.
*   **Branching**: Default branch is `master`.

## Compiler Requirements (Build / Model Tests)

*   Linux: `gcc` (from `build-essential`) + `suitesparse` + `mkl` (conda).
*   macOS: Clang + `suitesparse` (no MKL).
*   Windows: MinGW (`m2w64-toolchain` + `libpython`) + `suitesparse` + `mkl`. **CFFI preferred over ctypes on Windows**.

## Common Issues & Fixes

### Initial Guess Physics
Never set `lam=0` (tension) in pendulum/test models:
*   The Jacobian entry `dFx/dlam = -2*p_x` becomes exactly zero, causing singular matrix errors.
*   Use `lam >= 10.0` or estimate: `lam ≈ M*G/cos(theta)`.
*   Include velocities in initial guess: `v_x=0, v_y=0` for steady-state convergence.

### Windows KLU + ctypes
*   KLU with ctypes may fail on Windows even when cffi+KLU passes.
*   Diagnostic shows "Near-zero diagonal" despite correct Ap/Ai.
*   **Fix**: Use `dense` or `cffi` backend on Windows until resolved.

### Buffer Overflow Shield
*   `model_class.py` has `PAD = 50` guard band on arrays passed to C.
*   If tests crash with `STATUS_HEAP_CORRUPTION` (Windows), the C code may write past array bounds.
*   Padding absorbs overflows safely on Linux/macOS but not Windows heap canaries.