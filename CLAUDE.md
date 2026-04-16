# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all packages in development mode (use uv, not pip)
uv sync --all-packages

# Run full test suite
uv run pytest

# Run tests by marker (avoids slow compile steps)
uv run pytest -m parse        # parser/validation only
uv run pytest -m symbolic     # Jacobian computation only
uv run pytest -m codegen      # C code generation only
uv run pytest -m build        # full compile (requires gcc)
uv run pytest -m model        # end-to-end simulation
uv run pytest -m bps          # balanced power systems
uv run pytest -m uds          # unbalanced distribution

# Run a single test file
uv run pytest tests/core/test_pendulum.py

# Lint
uv run ruff check .
```

## Architecture

**Monorepo with three independent PyPI packages** sharing a single `uv` workspace. All three use Python [native namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) under the `pydae` namespace — there is intentionally **no `__init__.py`** in `packages/*/src/pydae/`. Never add one there; it would break namespace merging.

| Package dir | PyPI name | Import root | Purpose |
|---|---|---|---|
| `packages/pydae-core` | `pydae` | `pydae.core` | DAE solver engine |
| `packages/pydae-bps` | `pydae-bps` | `pydae.bps` | Balanced power systems |
| `packages/pydae-uds` | `pydae-uds` | `pydae.uds` | Unbalanced distribution systems |

### DAE System Form

pydae uses semi-explicit index-1 DAEs:

```
ẋ = f(x, y, u, p)    differential equations
 0 = g(x, y, u, p)   algebraic constraints
 z = h(x, y, u, p)   output equations
```

`x` = dynamic states, `y` = algebraic variables, `u` = inputs, `p` = parameters, `z` = outputs.

### DAE Solver Pipeline (pydae-core)

The core workflow: **define symbolically → build (compile C) → simulate at runtime**.

```
Builder(sys_dict)
   └── builder/core.py          orchestrates all phases
       ├── parser.py             validates & parses system dict to SymPy
       ├── symbolic.py           computes Jacobians (Fx, Fy, Gx, Gy, Fu, Gu, Hx, Hy)
       └── codegen/
           ├── cffi_builder.py   preferred backend; generates + compiles C via CFFI
           └── ctypes_builder.py alternative backend; uses ctypes + GCC/Clang
```

Output of a build: `{name}_data.json` (metadata, variable lists, sparsity patterns) + a compiled shared library (`.dll`/`.so`/`.dylib`).

Set `PYDAE_PARALLEL=1` to parallelize `sympy.ccode` translation across a `ProcessPoolExecutor` (useful when the system has > 200 expressions).

**Sparse solver options** (passed as `sparse=` to `Builder`):
- `False` — dense LAPACK LU (default)
- `True` / `'klu'` — SuiteSparse KLU, 0-based CSC
- `'pardiso'` — Intel MKL PARDISO, 1-based CSR
- `'accelerate'` — Apple Accelerate, macOS only

The ini and run Jacobians have **independent sparsity patterns** (`Ap_ini/Ai_ini` vs `Ap_trap/Ai_trap`) because `y_ini` and `y_run` may differ, producing different structurally-zero entries. PARDISO requires `iparm[11]=2` (solve Aᵀx=b) because the pattern is emitted in CSC form (= CSR of the transpose).

**CFFI vs ctypes:** CFFI links at compile time (more reliable on Windows); ctypes compiles a standalone `.dll`/`.so` loaded at runtime (simpler, no CFFI dependency). Both produce identical results.

### System Dictionary Structure

```python
sys_dict = {
    'name': 'model_name',
    'params_dict': {'L': 1.0, ...},       # parameter name → default value
    'f_list': [...],                        # differential equations (ẋ = f)
    'g_list': [...],                        # algebraic equations (0 = g)
    'x_list': [...],                        # differential state variables
    'y_ini_list': [...],                    # algebraic vars during initialization
    'y_run_list': [...],                    # algebraic vars during time simulation
    'u_ini_dict': {'var': value, ...},      # inputs during initialization
    'u_run_dict': {'var': value, ...},      # inputs during time simulation
    'h_dict': {'output': expr, ...},        # output equations
}
```

`y_ini_list` and `y_run_list` can differ. After `ini()` converges, `ini2run()` transfers values automatically: variables in `y_ini` that appear in `u_run` are copied over, and variables in `u_ini` that appear in `y_run` are seeded as starting points. The `f` and `g` equations are the same in both phases; only the Jacobian structure changes.

### Runtime Model API (model_class.py)

```python
from pydae.core import Model

model = Model('model_name')                        # loads JSON + compiled lib
model.ini(params_dict, xy_0)                       # Newton-Raphson initialization
model.run(t_end, inputs_dict)                      # trapezoidal time integration
model.post()                                       # truncate pre-allocated buffers

model.get_value('var')                             # scalar at current time
model.get_values('var')                            # full time series (after post)
model.set_value('var', val)
model.load_xy_0('file.json') / model.save_xy_0()  # initial condition I/O
```

Key tunables on the model instance: `Dt` (step size), `max_it`, `itol`, `decimation`, `alpha` (0.5 = trapezoidal, 1.0 = backward Euler).

Multiple `run()` calls chain seamlessly — each continues from the previous endpoint, enabling piecewise-constant input changes without re-initializing.

`jac_run_eval()` recovers continuous-time Jacobian blocks (`Fx`, `Fy`, `Gx`, `Gy`) from the compiled trapezoidal Jacobian. `A_eval()` computes the reduced state matrix `A = Fx - Fy·Gy⁻¹·Gx` for eigenvalue analysis.

On `ini()` failure, diagnostics run automatically — outputs a Jacobian heatmap (`jacobian_diagnostic.png`) and a terminal report checking for zero rows/columns, near-zero pivots, and condition number.

### Power Systems Builders (pydae-bps, pydae-uds)

These builders read HJSON network descriptions and assemble `sys_dict` objects for `pydae-core`. Each component module (e.g., `syns/milano2ord.py`) returns partial equation lists that `BpsBuilder` / `UdsBuilder` concatenates. Key component families in `pydae-bps`: synchronous generators (`syns/`), voltage source converters (`vscs/`), AVRs (`avrs/`), governors (`govs/`), wind turbines (`wecs/`), loads, lines.

### SSA Module (pydae.ssa)

Small-signal analysis via linearization. `eval_ss(model)` computes the state-space matrices:
- `A = Fx - Fy·Gy⁻¹·Gx`
- `B = Fu - Fy·Gy⁻¹·Gu`
- `C = Hx - Hy·Gy⁻¹·Gx`
- `D = Hu - Hy·Gy⁻¹·Gu`

Called after `model.ini()` when the Jacobian sub-blocks are populated.

## Import Paths

```python
from pydae.core import Builder, Model
from pydae.bps import BpsBuilder
from pydae.uds import UdsBuilder
from pydae.ssa import eval_ss, eval_A
```

Old-style imports (`import pydae.build_cffi`, `from pydae.bmapu import ...`) no longer work. See `MIGRATION_GUIDE.md` for full mapping.
