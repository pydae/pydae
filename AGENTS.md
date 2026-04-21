# AGENTS.md

Compact instructions for OpenCode sessions in `pydae`. See `CLAUDE.md` for full technical documentation.

## Workflow & Commands

*   **Setup**: Always use `uv sync --all-packages`. Never use `pip`.
*   **Testing**: Use `uv run pytest`.
    *   Fast (No C compiler): `uv run pytest -m "parse or symbolic or codegen"`
    *   Full (Requires GCC/MSVC): `uv run pytest -m "build or model"`
    *   Targeting: `uv run pytest tests/core/`, `uv run pytest tests/bps/`
*   **Linting**: `uv run ruff check .`

## Architecture & Quirks

*   **Monorepo**: Packages in `packages/` (`pydae-core`, `pydae-bps`, `pydae-uds`).
*   **Namespace Packages**: `packages/*/src/pydae/` MUST NOT have `__init__.py`.
*   **Pipeline**: `define symbolic` → `build (C-compile)` → `simulate`.
*   **Parallelism**: Set `PYDAE_PARALLEL=1` for systems with > 200 expressions.
*   **ini/run Swap**: Never delete/re-append variables in `y_ini` to maintain index integrity for downstream components; swap in-place.
*   **Line Endings**: Repository is forced LF. Do not override.

## Constraints

*   **LaTeX Docs**: Docstrings must be raw strings (`r"""..."""`) to prevent escape character errors.
*   **Test Patterns**:
    *   Use `def test():` at the bottom of component modules for verification against sibling `.hjson` files.
    *   Test compilation backends (`cffi`/`ctypes`) and sparse modes (`dense`/`klu`) using the pattern in `tests/core/test_compilation_modes.py`.
*   **Branching**: Repo default is `master` (not `main`).
*   **Environment**: Repo is cross-platform; use `pathlib.Path` or `/` for paths, avoid hard-coded backslashes.
*   **Compiler**: Build pipeline requires `gcc` (Linux) or MSVC/MinGW (Windows).
