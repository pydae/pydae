# AGENTS.md

Compact instructions for OpenCode sessions in `pydae`. See `CLAUDE.md` for full technical details.

## Workflow & Commands
- **Setup**: `uv sync --all-packages`. NEVER use `pip install`.
- **Testing**: `uv run pytest`.
  - **Fast** (no compiler): `uv run pytest -m "parse or symbolic or codegen"`
  - **Full** (requires GCC/MinGW): `uv run pytest -m "build or model"`
- **Linting**: `uv run ruff check .`

## Architecture & Monorepo
- **Packages**: `pydae-core` (published as `pydae`), `pydae-bps`, `pydae-uds`.
- **Namespace Constraint**: `packages/*/src/pydae/` MUST NOT contain `__init__.py`. Adding one breaks the namespace merging of the three packages.
- **Pipeline**: `Builder(sys_dict)` → `build()` → `Model.load()`.
- **Parallel Codegen**: Set `PYDAE_PARALLEL=1` for systems with > 200 expressions to speed up SymPy translation.

## Windows & Environment Gotchas
- **Build Backend**: On Windows, `Builder(..., target='ctypes')` is more reliable than `cffi`.
- **Python 3.13 Warning**: `ctypes` and `cffi` can trigger heap corruption (`0xc0000374`) during codegen on Windows 3.13.
- **Sparse Solvers**: Sparse solvers (KLU) via `ctypes` are unstable on Windows. Use `sparse=False` (dense) for local development on Windows.
- **Line Endings**: Force LF. Do not convert to CRLF, as it breaks CFFI caching hashes.

## Physics & Solver Stability
- **Initial Guesses**: Never set `lam=0` (causes singular Jacobian). Use `lam >= 10.0` (50.0 is better).
- **Steady State**: Always include velocities (e.g., `v_x=0, v_y=0`) in initial guesses.
- **Diagnostics**: If `model.ini()` fails, it automatically prints a **DAE SOLVER DIAGNOSTIC REPORT** and saves `jacobian_diagnostic.png`. Check for zero rows/columns or large residuals.
- **ini/run Swap**: When initializing PV buses, replace variables in `y_ini` in-place at the same index to keep downstream integer-index references stable.

## Conventions
- **Docstrings**: Use raw strings `r"""..."""` to avoid LaTeX escape sequence errors (e.g., `\xi`).
- **Component Tests**: New components in `bps`/`uds` should have a `test()` function at the bottom and a sibling `.hjson` fixture.
- **Paths**: Use `pathlib.Path` or forward slashes `/`. No hard-coded backslashes.
