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
*   **Sparse solvers**: `sparse=False` (dense LU), `sparse='klu'` (SuiteSparse, all OSes), `sparse='pardiso'` (Intel MKL, Intel/AMD only). KLU requires SuiteSparse headers.
*   **Parallel codegen**: `PYDAE_PARALLEL=1` for > 200 expressions.

## Conventions

*   **ini/run swap**: Swap variable values in-place at the same index in `y_ini` — never delete and re-append (downstream components reference by integer index).
*   **LaTeX docstrings**: Use raw strings `r"""..."""`.
*   **Component tests**: `def test():` at bottom of component modules, verified against sibling `.hjson`.
*   **Cross-platform**: `pathlib.Path` or `/` for all paths; no hard-coded backslashes.
*   **Line endings**: Force LF via `.gitattributes`. Do not override.
*   **Branching**: Default branch is `master`.

## Compiler Requirements (Build / Model Tests)

*   Linux: `gcc` (from `build-essential`).
*   Windows: MinGW (`m2w64-toolchain` + `libpython`). CFFI backend preferred over ctypes on Windows.
