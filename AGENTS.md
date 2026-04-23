# AGENTS.md

Compact instructions for OpenCode sessions in this monorepo. See `CLAUDE.md` for full technical documentation.

## Workflow & Commands
*   **Setup**: `uv sync --all-packages`. Never use `pip install`.
*   **Testing**: `uv run pytest`.
    *   Fast (no compiler): `uv run pytest -m "parse or symbolic or codegen"`
    *   Full (GCC/MinGW required): `uv run pytest -m "build or model"`
*   **Linting**: `uv run ruff check .`

## Architecture
*   **Monorepo**: Workspace with three independent packages (`pydae-core`, `pydae-bps`, `pydae-uds`).
*   **Namespace**: `packages/*/src/pydae/` MUST NOT have `__init__.py`. Only subpackages (e.g., `core/`, `bps/`) have them.
*   **Pipeline**: `Builder(sys_dict)` → `build()` → `Model.load()`.
*   **Parallel Codegen**: Set `PYDAE_PARALLEL=1` for > 200 expressions.

## Conventions
*   **Docstrings**: Use raw strings `r"""..."""` for LaTeX support.
*   **Component Tests**: Include `def test():` at the bottom of component modules; verify against sibling `.hjson`.
*   **Paths**: Use `pathlib.Path` or `/`. No hard-coded backslashes for cross-platform compatibility.
*   **ini/run swap**: Swap variable values in-place at the same index in `y_ini`. Never delete/re-append to keep indices stable for component reference.

## Environment & Gotchas

### Windows Build Backend
*   **Codegen crashes**: On Windows Python 3.13, both ctypes and cffi fail in `sym2xyup` (regex heap corruption `0xc0000374`).
*   **CI skips builds**: `ci.yml` explicitly skips build/model tests on Windows. Fast tests only (`parse`, `symbolic`, `codegen`) run.
*   **Use `sparse=False`**: If building locally on Windows, use dense backend.

### Initial Guess Physics
*   **Never set `lam=0`**: Causes singular Jacobian (`dFx/dlam = -2*p_x` becomes exactly zero).
*   Use `lam >= 10.0` (50.0 is more robust) and include velocities: `v_x=0, v_y=0`.

### Sparse Solvers on Windows
*   KLU with ctypes may fail even when cffi+KLU passes.
*   **Fix**: Use `dense` backend on Windows (no sparse solver).

### Buffer Overflow Shield
*   `model_class.py` has `PAD = 50` guard band.
*   `STATUS_HEAP_CORRUPTION` on Windows indicates C array index-out-of-bounds.

### Line Endings
*   Force LF via `.gitattributes`. Do not override.