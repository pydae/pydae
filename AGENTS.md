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

## CasADi Backend
- **Location**: `pydae.core.builder.casadi.{CasadiBuilder, CasadiModel}`
- **No C compiler needed** — uses CasADi JIT and IDAS integrator.
- **API parity**: `CasadiModel` exposes `ini()`, `run()`, `post()`, `A_eval()`, `BCD_eval()`, `eval_eigenvalues()`.
- **Initialization**: `ca.rootfinder('newton')` for algebraic solves.
- **Integration**: `ca.integrator('idas')` for DAE time-stepping.
- **Tolerance parameters**: Pass to `CasadiModel(builder, newton_tol=1e-12, integrator_reltol=1e-10)` or override in `model.ini(newton_tol=1e-12)`.
- **External Newton**: Enable with `model.use_external_newton(True)` to use `_newton_solve()` with `_residual_fn`/`_jacobian_fn` instead of `ca.rootfinder`.
- **Output evaluation**: `h_dict` outputs (e.g., `E_p`, `E_k`) are evaluated via `_h_fn` function, automatically created by `CasadiBuilder` and transferred to `CasadiModel`.
- **Init substitution bug fix**: In `casadi_builder.py`, `y_run-only` substitution now excludes `u_ini` variables to prevent `theta` inputs from being hardcoded to 0.0.
- **Shadowing fix**: Builder stores evaluators as `A_eval_fn`, `B_eval_fn`, etc. to avoid `self.A_eval` shadowing the method.

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
