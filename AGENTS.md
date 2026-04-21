# AGENTS.md

Compact instructions for OpenCode sessions in `pydae`. See `CLAUDE.md` for full technical documentation.

## Workflow & Commands

*   **Setup**: `uv sync --all-packages`. Never use `pip`.
*   **Testing**: `uv run pytest`
    *   Fast: `uv run pytest -m "parse or symbolic or codegen"` (no C compiler)
    *   Full: `uv run pytest -m "build or model"` (needs GCC/MSVC)
    *   Single file: `uv run pytest tests/core/test_pendulum.py`
    *   Single package: `uv run --package pydae pytest tests/core/`
*   **Linting**: `uv run ruff check .`

## Architecture

*   **Monorepo**: 3 packages — `pydae-core` (solver), `pydae-bps` (balanced PS), `pydae-uds` (unbalanced).
*   **Imports**: `from pydae.core import Builder, Model` | `from pydae.bps import BpsBuilder` | `from pydae.uds import UdsBuilder`.
*   **Namespace**: `packages/*/src/pydae/` MUST NOT have `__init__.py`.

## Quirks

*   **DAE Pipeline**: `define symbolic` → `build (C-compile)` → `simulate`.
*   **Sparse solvers**: `sparse=False` (dense), `sparse=True` or `'klu'` (SuiteSparse), `'pardiso'` (MKL).
*   **Parallelism**: `PYDAE_PARALLEL=1` for >200 expressions.
*   **ini/run Swap**: Swap at same index in `y_ini` (replace in-place, never append-then-delete).
*   **Line Endings**: Forced LF. Do not override.

## Constraints

*   **LaTeX Docs**: Raw strings (`r"""..."""`) — escape errors otherwise.
*   **Component Tests**: `def test()` at bottom of modules validates against sibling `.hjson`.
*   **Branching**: Default branch is `master` (not `main`).
*   **Cross-platform**: Use `pathlib.Path` or `/`, never hard-coded backslashes.
*   **C Compiler Required**: Linux needs `build-essential` (gcc), Windows needs MSVC or MinGW.
