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

The core workflow: **define symbolically → build (compile C *or* fold into CasADi) → simulate at runtime**.

```
pydae.core.Builder(sys_dict)            ← from pydae.core import Builder
   └── builder/sympy_builder.py          orchestrates SymPy → C pipeline
       ├── common/parser.py               validates & parses system dict to SymPy
       ├── common/symbolic.py             computes Jacobians (Fx, Fy, Gx, Gy, Fu, Gu, Hx, Hy)
       └── builder/codegen/
           ├── cffi_builder.py             preferred backend; generates + compiles C via CFFI
           └── ctypes_builder.py           alternative backend; ctypes + GCC/Clang/MSVC

builder/casadi_builder.py                 alternative: folds SymPy → CasADi SX graph (no C compile)
                                          paired with model/casadi_model.py runtime (IDAS integrator)

diagnostics/dae_check.py                  Jacobian-health report emitted on ini() failure
src/pydae/daesolver/                      C runtime sources (daesolver.c/.h, *_dense.c, *_dlu_klu.c, *_run_lapack.c)
src/pydae/ssa/                            small-signal analysis (eval_ss, eval_A, damp)
```

Output of a SymPy build: `{name}_data.json` (metadata, variable lists, sparsity patterns) + a compiled shared library (`.dll`/`.so`/`.dylib`). The CasADi backend skips C compilation entirely and serializes the SX graph instead.

Set `PYDAE_PARALLEL=1` to parallelize `sympy.ccode` translation across a `ProcessPoolExecutor` (useful when the system has > 200 expressions).

**Sparse solver options** (passed as `sparse=` to `Builder`):
- `False` — dense LAPACK LU (default)
- `True` / `'klu'` — SuiteSparse KLU, 0-based CSC
- `'pardiso'` — Intel MKL PARDISO, 1-based CSR
- `'accelerate'` — Apple Accelerate, macOS only

The ini and run Jacobians have **independent sparsity patterns** (`Ap_ini/Ai_ini` vs `Ap_trap/Ai_trap`) because `y_ini` and `y_run` may differ, producing different structurally-zero entries. PARDISO requires `iparm[11]=2` (solve Aᵀx=b) because the pattern is emitted in CSC form (= CSR of the transpose).

**CFFI vs ctypes:** CFFI links at compile time (more reliable on Windows now that CI uses MSVC); ctypes compiles a standalone `.dll`/`.so` loaded at runtime (simpler, no CFFI dependency). Both produce identical results.

**SymPy vs CasADi backend:** the default `Builder` (from `builder/sympy_builder.py`) emits C and uses the trapezoidal `daesolver.c` runtime via `Model` (from `model/ctypes_model.py`). The CasADi path (`builder/casadi_builder.py` + `model/casadi_model.py`) constructs an `SX` graph and integrates via SUNDIALS/IDAS — no compiler required, slower per-step but useful where a C toolchain is unavailable. `feat: ss_num2sym supports both SymPy and CasADi backends` (commit e00c1b8) is the latest cross-backend convergence work.

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

### Runtime Model API (model/ctypes_model.py)

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

On `ini()` failure, diagnostics from `diagnostics/dae_check.py` run automatically — outputs a Jacobian heatmap (`jacobian_diagnostic.png`) and a terminal report checking for zero rows/columns, near-zero pivots, and condition number.

**Dual-buffer architecture**: The C solver writes directly into private arrays `_Time`, `_X`, `_Y`, `_Z` (allocated with `PAD=50` guard elements). `post()` creates the public `Time`, `X`, `Y`, `Z` arrays via `np.copy()` — these are safe to pass to Matplotlib or other libraries. Accessing `_*` arrays before `post()` is intentional (hot-path, no copy); accessing them after `post()` will see zeroed buffers because `_post_called` triggers a reset on the next `run()`.

### Power Systems Builders (pydae-bps, pydae-uds)

These builders read HJSON network descriptions and assemble `sys_dict` objects for `pydae-core`. Each component module (e.g., `syns/milano2ord.py`) returns partial equation lists that `BpsBuilder` / `UdsBuilder` concatenates. Key component families in `pydae-bps`: synchronous generators (`syns/`), voltage source converters (`vscs/`, `vsc_models/`), AVRs (`avrs/`), governors (`govs/`), PSSs (`psss/`), PODs (`pods/`), wind turbines (`wecs/`), loads, lines, reactive power banks (`miscellaneous/`). In `pydae-uds`: lines (`lines/`), grid-forming VSCs (`vsgs/`).

**AGC (Automatic Generation Control)**: activated by an `agc` key in the HJSON data dict. `BpsBuilder.construct()` calls `add_agc(self)` after all other builders, so the governor's `p_c_{gen}` (or `p_m_{gen}`) already exists in `u_ini_dict`/`u_run_dict` — `add_agc` pops it and replaces it with an algebraic variable driven by a PI on rotor speed. Config format:

```hjson
agc: {gen: "2", K_p_agc: 10.0, K_i_agc: 2.0}
```

`gen` is the generator name (matches the bus name used in `syns`). Outputs `p_agc` and `xi_agc` in `h_dict`.

### SSA Module (src/pydae/ssa/)

Small-signal analysis via linearization, imported as `from pydae.ssa import eval_ss, eval_A, damp`. `eval_ss(model)` computes the state-space matrices:
- `A = Fx - Fy·Gy⁻¹·Gx`
- `B = Fu - Fy·Gy⁻¹·Gu`
- `C = Hx - Hy·Gy⁻¹·Gx`
- `D = Hu - Hy·Gy⁻¹·Gu`

Called after `model.ini()` when the Jacobian sub-blocks are populated.

### Real-time API (pydae.api)

`packages/pydae-core/src/pydae/api/realtime_api.py` — soft real-time FastAPI server wrapping a `CasadiModel` in a background thread.

**Usage pattern:**
```python
from pydae.api.realtime_api import app
app.state.model        = model          # CasadiModel, already ini()-ed
app.state.chunk_ms     = 50.0           # integration chunk in ms
app.state.ramp_chunks  = 20             # chunks to spread each setpoint change over
app.state.cosim_config = grid.data.get("configs", {})  # optional co-sim config
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/measurements` | All h_dict outputs (browser-friendly) |
| `POST` | `/measurements` | Selective h_dict outputs by name list |
| `GET` | `/cosim/measurements` | Only measurements declared in `configs` |
| `POST` | `/setpoints` | Write any u_run inputs (ramped); body: `{"setpoints": {...}, "timestamp": float\|null}` |
| `POST` | `/cosim/setpoints` | Write setpoints restricted to `configs` — rejects unknown keys with 422 |
| `POST` | `/set_input` | Write a single input (ramped) |
| `GET` | `/status` | `{"is_running": bool, "t_sim": float}` |

**Key design points:**
- Setpoint changes are ramped over `ramp_chunks` chunks to avoid `IDA_LINESEARCH_FAIL` on large steps.
- The background thread catches `RuntimeError` from failed IDAS steps and logs them rather than crashing.
- CORS is open (`allow_origins=["*"]`) for browser and co-simulation client access.
- `cosim_config` uses the `configs` section of the co-simulation JSON. Variable names are resolved via `emec_name` (exact), `emec_prefix` (prefix match), or `emec_template` (regex with `<emec_id>` backreference). Measurements resolve against `h_dict`; setpoints against `u_run_names`.

## Import Paths

```python
from pydae.core import Builder, Model
from pydae.bps import BpsBuilder
from pydae.uds import UdsBuilder
from pydae.ssa import eval_ss, eval_A
from pydae import utils   # shared helpers (e.g., unit conversions, grid utilities)
```

Old-style imports (`import pydae.build_cffi`, `from pydae.bmapu import ...`) no longer work. See `MIGRATION_GUIDE.md` for full mapping.

## Cross-platform notes

This repo is developed on both Windows and Linux (Debian). Claude Code may be invoked from either — keep commands portable.

**Toolchain**
- `uv` is the package manager. Do not use `pip install -e .` — the workspace layout relies on `uv sync --all-packages`.
- A C compiler is required for `-m build` / `-m model` tests and for `Builder.build()` (only when using the SymPy backend — the CasADi backend has no C dependency):
  - Linux: `sudo apt install build-essential` (provides `gcc`).
  - Windows: MSVC (Visual Studio Build Tools) is the supported toolchain on CI as of commit 4a58af4. MinGW still works locally for `target='ctypes'` but CFFI requires MSVC against modern (Python 3.12+) MSVC-built CPython.
  - macOS: Apple Clang (pre-installed).
- Python ≥ 3.10. `uv run` auto-selects the workspace interpreter; outside `uv`, use `python -m pytest` rather than a bare `pytest` to avoid picking up a different interpreter's scripts.

**Windows CI / Build Limitations**
- CFFI: Now built with MSVC on Windows CI (commit 4a58af4). MinGW-only environments must use `target='ctypes'`.
- ctypes+KLU: Unstable on Windows (Python 3.13 regex heap corruption) — CI excludes the `windows × ctypes × sparse` cell. Use `target='ctypes', sparse=False` or `target='cffi', sparse='klu'` on Linux/macOS.
- SuiteSparse for KLU on Windows is installed via `conda install -c conda-forge suitesparse`.

**Path and shell**
- Inside Git Bash / WSL / Linux: forward slashes, `/dev/null`.
- Inside `cmd.exe` / PowerShell: prefer forward slashes where tools accept them; use `NUL` only for native Windows commands.
- All Python paths in the codebase should use `pathlib.Path` or forward slashes — no hard-coded backslashes.

**Line endings**
- Repo is normalised to LF via `.gitattributes` (`* text=auto eol=lf`). Do not override locally. C sources, HJSON fixtures, and Python files must stay LF so that hash-based caching (e.g., the CFFI source-unchanged skip in `daesolver`) is stable across OSes.

**Parallel symbolic codegen**
- `PYDAE_PARALLEL=1` fans out `sympy.ccode` translation across a `ProcessPoolExecutor`. Helpful for large systems (> 200 expressions). On Windows, the spawn-start-method overhead is higher; leave it off for small models.

**Running tests on a cold machine**
```bash
uv sync --all-packages              # one-time; installs all three packages + dev deps
uv run pytest -m "parse or symbolic or codegen"   # fast, no C compiler required
uv run pytest -m "build or model"   # requires gcc / MSVC
```

## Repo layout landmarks

- `packages/pydae-core/src/pydae/core/` — engine. `builder/` (sympy + casadi + codegen), `model/` (ctypes_model + casadi_model), `common/` (parser, symbolic), `diagnostics/`.
- `packages/pydae-core/src/pydae/ssa/` — small-signal analysis (separate sibling namespace, not under `core/`).
- `packages/pydae-core/src/pydae/daesolver/` — C runtime: `daesolver.c/.h`, `daesolver_dense.{c,h}`, `daesolver_dlu_klu.c`, `daesolver_run_lapack.{c,h}`.
- `packages/pydae-bps/src/pydae/bps/` — power-system component library. `avrs/`, `govs/`, `syns/`, `vscs/`, `wecs/`, `loads/`, `lines/`, `psss/`, `pods/`, `sources/`. Each module ships with a sibling `.hjson` fixture used by its in-module `test()`.
- `packages/pydae-uds/src/pydae/uds/` — unbalanced distribution builder. Component families: `lines/`, `vsgs/` (grid-forming VSCs).
- `tests/{core,bps,uds}/` — pytest suite. Markers declared in `tests/conftest.py`.
- `examples/` — standalone scripts (`pendulum.py`, `milano*ord*.py`). Build artefacts (`*_data.json`, `*.svg`) are gitignored.
- `docs/pydae-{core,bps,uds}/` — three independent Sphinx projects (each with its own `conf.py` and `.readthedocs.yaml`).

## Conventions

- **Docstrings with LaTeX**: use raw strings (`r"""..."""`). A non-raw docstring with `\xi`, `\n`, etc. will raise `SyntaxError: (unicodeescape) truncated \xXX escape`.
- **Component modules** (`avrs/`, `govs/`, etc.) expose a `descriptions()` list (the single source of truth for parameters/inputs/states/outputs, consumed by docs autosummary) and a builder function taking `(dae, data, name, bus_name)` that appends into `dae['f']`, `dae['g']`, `dae['x']`, `dae['y_ini']`, `dae['y_run']`, `dae['u_ini_dict']`, `dae['u_run_dict']`, `dae['params_dict']`, `dae['xy_0_dict']`, `dae['h_dict']`.
- **ini/run variable swap**: for PV-bus initialisation, swap `V_bus ↔ v_ref` at the same index in `y_ini` (replace in place, do not append-then-delete) so that downstream components that reference `y_ini` by integer index — notably `vsource`'s `g[idx_V] = ...` — keep targeting the correct equation. This pattern is preferred over the legacy `xi_v` dummy-integrator approach. See `avrs/kundur.py`, `avrs/kundur_tgr.py`, `avrs/sexs.py`, `avrs/avr_1.py` for the canonical form.
- **In-module tests**: component files carry a `def test():` at the bottom that builds a minimal network from the sibling `.hjson`, runs `ini()` and `run()`, and asserts on observables. Pattern: reuse for any new component.
- **Namespace packages**: never add `__init__.py` under `packages/*/src/pydae/` — only inside the subpackage (`pydae/core/`, `pydae/bps/`, etc.).
- **Commit style**: use conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`). Do not add `Co-Authored-By: Claude` trailers in this repo.
- **Active-power dispatch**: use `p_c_lc` (desired grid injection in machine pu) at the syn level, not `p_m`. The LC integrator (`load_controller.py`) wraps the governor setpoint and compensates armature losses automatically so that `p_g = p_c_lc` at steady state. `p_m` is kept as a direct input only when no LC is used. See `pydae.bps.miscellaneous.load_controller`.
- **Three-level active-power control**: LC (slow, ~100 s) compensates losses → `p_c_lc` → `ctrl_sym`. AGC (secondary, seconds) drives `dp_lc` (fast additive channel on the AGC-designated generator). All other generators use LC. Never attach both LC and AGC to the same machine.
- **vsource (infinite bus)**: `sources:[{type:"vsource",bus:"N",...}]` pins V and θ at bus N, replaces the AGC machine, and contributes H=10⁶ to the COI. When a vsource is present all generators use LC; remove the `agc:` key. Seed `v_ref_{name}` near the expected bus voltage in `model.ini()` to keep the AVR out of saturation during the cold start.
- **Parallel codegen**: set `PYDAE_PARALLEL=1` (not `'8'`) and `PYDAE_MAX_WORKERS=N` as separate env-vars before importing pydae. The check is `os.environ.get("PYDAE_PARALLEL") == "1"`.
- **UTF-8 data files**: `BpsBuilder`, `read_data()`, `reporter._load_data()`, and `model_class._load_metadata()` all open files with `encoding='utf-8'`. HJSON files may contain non-ASCII characters in comments without issues.
- **damp() sort**: `ssa.damp(A, sort='damp')` or `sort='freq'` sorts the printed table and the returned dict by damping ratio or frequency.

## Docs

```bash
# build one subproject
uv run sphinx-build -b html docs/pydae-core docs/pydae-core/_build/html
uv run sphinx-build -b html docs/pydae-bps  docs/pydae-bps/_build/html
uv run sphinx-build -b html docs/pydae-uds  docs/pydae-uds/_build/html
```

ReadTheDocs is configured per subproject. The repo's default branch is `master` (not `main`) — the RTD admin must match, or builds fail with `fatal: couldn't find remote ref refs/heads/main`.

## Releases

```bash
# bump version in packages/<pkg>/pyproject.toml, commit with tag per package
git tag pydae-core-vX.Y.Z
git tag pydae-bps-vX.Y.Z

# build + publish (PyPI token in ~/.pypirc)
uv build --package pydae     packages/pydae-core
uv build --package pydae-bps packages/pydae-bps
uvx twine upload dist/pydae-X.Y.Z*       --repository pypi
uvx twine upload dist/pydae_bps-X.Y.Z*   --repository pypi
```

Per-package tags (not a single repo-wide tag) so subpackages can release independently.

## Troubleshooting & Diagnostics

### Automatic DAE Diagnostics
On `ini()` failure, `model/ctypes_model.py` (via `diagnostics/dae_check.py`) outputs a **DAE SOLVER DIAGNOSTIC REPORT** to stdout:

- **Residual magnitudes** — large values = bad initial guess
- **Diagonal analysis** — near-zero diagonals = singular Jacobian
- **Condition number** — >10⁶ = ill-conditioned
- **Sparsity statistics** — NNZ fill ratio

Check for:
1. **Zero tension** (`lam=0`) — Jacobian entry `dFx/dlam` becomes zero → singular. Use `lam >= 10.0` (50.0 recommended).
2. **Missing velocities** — include `v_x=0, v_y=0` in initial guess for steady state.
3. **Windows CFFI fails**: Use `target='ctypes'` (MinGW cannot link Python 3.12+ MSVC-built Python).
4. **Windows ctypes+KLU unstable**: Use `target='ctypes', sparse=False` or test on Linux/macOS.

### Padding Shield
`model/ctypes_model.py` has `PAD = 50` guard band on arrays passed to C. This catches off-by-one overflows on Linux/macOS but not Windows heap canaries.

### CI Workflow
The workflow outputs JUnit XML (`--junit-xml`) and generates a Markdown summary table via `$GITHUB_STEP_SUMMARY`. Failure diagnostics are in test logs — search for "DAE SOLVER DIAGNOSTIC REPORT".
