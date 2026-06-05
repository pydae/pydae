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
uv run pytest -m core         # core solver / DAE pipeline
uv run pytest -m bps          # balanced power systems
uv run pytest -m uds          # unbalanced distribution
uv run pytest -m slow         # tests that take > 5 seconds

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

**Cross-backend accessor parity** (ctypes `Model` and `CasadiModel`): both expose the same scalar/batch lookup surface — `get_value(name)`, `get_values(name)` (post-`run()` time series), `get_mvalue([names])` (batch list-getter), and the `*_list` attributes `x_list`, `y_ini_list`, `y_run_list`, `u_ini_list`, `u_run_list`, `params_list`, `z_list` / `outputs_list` (alias). Code that walks the model (reporters like `pydae.uds.utils.reports.report_v`, `pydae.uds.utils.model2svg`, SSA, plotting helpers) is written against this shared surface and works on either backend without per-call shims. On `CasadiModel` the `*_list` attributes are aliases for the canonical `*_names` (string) lists.

### Power Systems Builders (pydae-bps, pydae-uds)

These builders read HJSON/JSON network descriptions and assemble `sys_dict` objects for `pydae-core`. Each component module (e.g., `syns/milano2ord.py`) returns partial equation lists that `BpsBuilder` / `UdsBuilder` concatenates. Key component families in `pydae-bps`: synchronous generators (`syns/`), voltage source converters (`vscs/`, `vsc_models/`, `vsc_ctrls/`), AVRs (`avrs/`), governors (`govs/`), PSSs (`psss/`), PODs (`pods/`), wind turbines (`wecs/`), PV systems (`pvs/` — `pv_dq`, `pv_dq_d`, `pv_dq_ss`, `pv_dq_vrt`, `pv_pq_ss`), WECC renewable converters (`weccs/`), WECC plant controllers (`ppcs/`), loads, lines, reactive power banks (`miscellaneous/`), grid-forming VSCs (`vsgs/`). In `pydae-uds` (three-phase, per-phase modelling): `vscs/` is the largest family, plus `vsgs/` (grid-forming), `genapes/`, `ess/` (storage), `fcs/`, `pvs/`, `loads/`, `lines/`, `transformers/`, `shunts/`, `sources/`, `vsc_ctrls/`, `miscellaneous/`.

**Synchronous machines — model choice.** `syns/genrou.py` is the canonical
round-rotor 6th-order machine (IEEE 1110-2019 Model 2.2 / Anderson-Fouad /
PSS/E `GENROU`). It takes industry-standard parameter tables (NTS Tabla 45,
IEEE 115, PSS/E) literally: $X_d''$, $X_q''$ are *terminal-referred*
(IEEE 115 Eq. (88): $X_{ds} = X_{ads} + X_l$), so the model has **no `X_l`
field**. Use `genrou` for new work. `syns/milano{2,3,4,6}ord.py` keep
Marconato / Sauer-Pai conventions (subtransient-minus-leakage stator
form, Marconato cross-coupling on the rotor) and are retained for
back-compat with existing benchmarks; they are deprecated but not
aliased — the two families produce different operating points and modes
on identical NTS-style input.

**Builder invocation workflow** (identical shape for both `BpsBuilder` and `UdsBuilder`):

```python
grid = BpsBuilder(data_or_path)        # or UdsBuilder(data, use_casadi=False)
grid.construct(name)                    # builds grid.sys_dict + grid.dae (concatenated equations)
# SymPy → C → trapezoidal runtime:
Builder(grid.sys_dict, target="cffi", sparse=False).build()
m = Model(name); m.ini({}, grid.dae["xy_0_dict"]); m.run(t_end, {}); m.post()
# OR CasADi → SX graph → IDAS runtime (no compiler):
mc = CasadiModel(CasadiBuilder(grid.sys_dict).build())
mc.ini({}, xy_0=grid.dae["xy_0_dict"]); mc.run(t_end, {}); mc.post()
```

Both builders accept `use_casadi=` and expose `.construct(name)`, `.sys_dict`, `.dae`, and `.build(name)`. `construct()` is misspelled `contruct_grid()` internally for the grid-assembly step — that name is load-bearing, do not "fix" it.

**pydae-uds specifics** (three-phase, 4-wire by default): each bus carries `N_nodes` (defaults to 4 = phases a,b,c + neutral n). Node voltages are modelled in rectangular form as `V_{bus}_{node}_{r|i}` (real/imag per node), with phase-neutral magnitudes exposed as `h_dict` outputs like `V_A1_an`. The builder assembles branch incidence matrices (`A`, `At`) and primitive admittances (`G_primitive`, `B_primitive`) through a `MathBackend` (`pydae.core.builder.casadi_builder`) so the same construction code emits either SymPy or CasADi graphs. Backend parity (both must converge to the same node voltages) is enforced by `tests/uds/test_uds_grid.py`. The UDS data dict uses top-level `system` (`S_base`), `buses`, `lines`, `transformers`, `shunts`, `sources`, `loads` keys.

**Dual-backend migration status (pydae-uds).** Components ported to `MathBackend` work on both SymPy and CasADi. Currently ported: `lines/`, `transformers/Dyn11`+`Dyg11`, `loads/load_ac` and `load_dc`, `sources/ac3ph4w_ideal` and `ac3ph3w_ideal`, `shunts/`, `miscellaneous/breaker`, `vsc_ctrls/ctrl_3ph_4w_droop`, `vscs/{acdc_3ph_4w_vdc_q, acdc_3ph_4w_pq, ac_3ph_4w}`, `vsgs/gflpfzv`, `ess/bess_dcdc_gf`. The `pydae_examples/grids/grid_urisi/acdc_7bus/` reference network exercises this whole ported stack on CasADi end-to-end. Other components (most `vscs/` variants, other `vsgs/`/`pvs/`/`fcs/`/`genapes/`, legacy non-Dyn11/Dyg11 transformer connections via `trafo_yprim`, `add_trafo_monitors` for those legacy connections) still use raw `sym.Symbol` and only work on the SymPy backend.

**Porting a new uds component to dual-backend** — the established pattern (see `vsgs/gflpfzv.py`, `vscs/acdc_3ph_4w_vdc_q.py`, `ess/bess_dcdc_gf.py` as canonical examples):
- Replace `sym.Symbol(name, real=True)` → `grid.backend.symbols(name)` (and `sym.symbols('a,b,c', real=True)` → three separate `bk.symbols` calls).
- Replace `sym.cos/sin/sqrt/exp/atan2` → `bk.cos/sin/sqrt/exp/atan2`. Use `bk.sqrt(x**2 + eps)` (small `eps`) as a smooth `|x|`.
- **Expand complex algebra in real form** — CasADi `SX` has no complex type. `s = v · conj(i)` becomes `re_s = v_r·i_r + v_i·i_i`, `im_s = v_i·i_r − v_r·i_i`. `Z·i` with `Z = R + jX` becomes `re = R·i_r − X·i_i`, `im = R·i_i + X·i_r`. Never construct `1j*x` or `sym.I*x` symbolically; either gate that branch behind `if not grid.use_casadi:` or remove it.
- Read branch currents from `grid.I_lines_re[idx]` / `grid.I_lines_im[idx]` (backend-agnostic), not `sym.re(grid.I_lines[idx])`.
- **Saturations use `bk.hard_limits(x, x_min, x_max)`** (clean `min(max(...))`), not `bk.Piecewise`. Reserve `bk.Piecewise` for true step functions (0/1 anti-windup indicators) and piecewise-linear interpolation (see the `_piecewise_linear` helper in `ess/bess_dcdc_gf.py`, which replaces `sympy.interpolating_spline`).
- COI accumulators (`grid.omega_coi_numerator`, `_denominator`) start as Python floats but become SX once any backend symbol is added; do not compare them to numerics with `==`. The builder already guards this with `isinstance(..., (int, float))`.

**WECC renewable model stack** (three-layer, all in `pydae-bps`):
```
ppcs/repc_a.py    — plant-level controller; monitors POI bus, commands Pref/Qext to one or more converters
weccs/reec_b.py   — local electrical control; sits between REPC_A and REGC_A, commands Ipcmd/Iqcmd
weccs/regc_a.py   — generator/converter interface; injects Ip/Iq into the network
```
REEC_B and REGC_A are nested inside a `weccs` HJSON entry (same pattern as an AVR inside `syns`). REPC_A is a separate plant-level entry in `ppcs`. Available variants: `reec_b`, `reec_e`; `regc_a`, `regc_b`, `regfm_a1`, `regfm_b1`; `repc_a`, `repc_d`.

**AGC (Automatic Generation Control)**: activated by an `agc` key in the HJSON data dict. `BpsBuilder.construct()` calls `add_agc(self)` after all other builders, so the governor's `p_c_{gen}` (or `p_m_{gen}`) already exists in `u_ini_dict`/`u_run_dict` — `add_agc` pops it and replaces it with an algebraic variable driven by a PI on rotor speed. Config format:

```hjson
agc: {gen: "2", K_p_agc: 10.0, K_i_agc: 2.0}
```

`gen` is the generator name (matches the bus name used in `syns`). Outputs `p_agc` and `xi_agc` in `h_dict`.

### CasADi backend solver internals

**Automatic differentiation, not analytic Jacobians.** `CasadiBuilder` exposes `_residual_fn` and `_jacobian_fn` built via `ca.jacobian(eq_ini, v_ini)` — pure reverse-mode AD. There is no `sympy.diff`-derived `Fx`/`Fy`/`Gx`/`Gy` codegen on the CasADi path (that is the SymPy backend's job). Adding new components or expressions to a CasADi-built model does not require deriving Jacobians by hand or invoking any analytic-Jacobian step. When choosing among solver paths below, remember: every one of them differentiates the residual via CasADi AD; only the iteration/step control differs.

**`CasadiModel.ini()` fallback chain** (`casadi_model.py:ini`):

1. `ca.rootfinder('rf', 'newton', ...)` — fast plain-Newton plugin. Succeeds on simple networks from any reasonable seed; fails on stiff or saturated DAEs (raises `RuntimeError`).
2. `_newton_solve` — small Python loop that calls `_residual_fn` / `_jacobian_fn` and solves the step with `np.linalg.solve`. Same AD-derived Jacobian as the plugin, looser iteration control. Handles distribution networks like `acdc_7bus` where the bundled `'newton'` plugin gives up but a plain Newton iterate still converges. Tolerates partial seeds.
3. `_calc_ic_init` — IDAS `calcIC` over a 1e-12 s integration; last resort, often fails on stiff models with `IDA_LINESEARCH_FAIL`.

KINSOL (`'kinsol'` plugin, SUNDIALS' nonlinear solver) was evaluated as a more sophisticated step 2 but trips on `KIN_MXNEWT_5X_EXCEEDED` for distribution networks that span per-unit (~1) and SI bus voltages (~10⁴) — even with `u_scale` set from the seed. The simple `_newton_solve` is more robust in practice; do not swap it out without re-evaluating on a stiff network.

**Seed quality.** Only the step-1 `'newton'` plugin is sensitive to the initial guess. If `mc.ini()` is slow or escalates to `calcIC`, harvest a SymPy-converged operating point and pass it as `xy_0`:
```python
m = Model(name); m.ini({}, "xy_0.json")
xy = {n: float(m.get_value(n)) for n in m.x_list + m.y_run_list}
mc.ini({}, xy_0=xy)   # full-coverage seed → step 1 converges
```

### SSA Module (src/pydae/ssa/)

Small-signal analysis via linearization, imported as `from pydae.ssa.ssa import eig, damp, participation`. `eval_ss(model)` computes the state-space matrices:
- `A = Fx - Fy·Gy⁻¹·Gx`
- `B = Fu - Fy·Gy⁻¹·Gu`
- `C = Hx - Hy·Gy⁻¹·Gx`
- `D = Hu - Hy·Gy⁻¹·Gu`

Called after `model.ini()` when the Jacobian sub-blocks are populated.

**`eig(model)`** computes eigenvalues and eigenvectors and stores three attributes on `model`:
- `model.eigenvalues` — complex eigenvalue array, shape `(N_x,)`
- `model.right_eigenvectors` / `model.left_eigenvectors` — matrices `V`, `W = inv(V)`
- `model.participation` — Kundur-normalised participation matrix, shape `(N_x, N_x)`; element `[k, i]` is `|φ_ki · ψ_ik|` divided by the column sum so each column sums to 1. Large values identify which states drive each mode.

**`participation(model, method='kundur')`** returns a labelled DataFrame. The `'kundur'` method returns the column-normalised absolute value `|P_raw| / ‖P_raw‖_col` (corrected in v1.4.0 — earlier versions returned the raw complex product).

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
from pydae.core.builder import CasadiBuilder, CasadiModel, MathBackend  # CasADi backend
from pydae.bps import BpsBuilder
from pydae.uds import UdsBuilder
from pydae.ssa.ssa import eig, damp, participation
from pydae.bps.utils.visualizer import PowerSystemVisualizer
from pydae import utils   # shared helpers (e.g., unit conversions, grid utilities)
```

Install the real-time API extras with `pip install pydae[api]` (adds FastAPI + uvicorn).

Old-style imports (`import pydae.build_cffi`, `from pydae.bmapu import ...`) no longer work. See `MIGRATION_GUIDE.md` for full mapping.

## Editable install into an external Python (Jupyter / Anaconda)

To make the in-tree packages visible to a system-wide or Anaconda interpreter without using the `uv` workspace `.venv`:

```bash
# from the repo root, using the target interpreter explicitly
python -m pip install -e packages/pydae-core -e packages/pydae-bps
# or on Windows Anaconda:
"C:/ProgramData/anaconda3/python.exe" -m pip install -e packages/pydae-core -e packages/pydae-bps
```

Edits under `packages/*/src/` are picked up on the next import — no reinstall needed. Note: `pydae` is a namespace package so `pydae.__version__` is undefined; check versions with `from importlib.metadata import version; version('pydae')`.

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
- `packages/pydae-core/src/pydae/svg_tools/` — SVG topology annotator (sibling namespace, restored from the `v0-legacy` tag). Exposes the `svg` class (`from pydae.svg_tools import svg`) used by `pydae.uds.utils.model2svg`. Depends on `svgwrite` and `xml.etree`.
- `packages/pydae-core/src/pydae/daesolver/` — C runtime: `daesolver.c/.h`, `daesolver_dense.{c,h}`, `daesolver_dlu_klu.c`, `daesolver_run_lapack.{c,h}`.
- `packages/pydae-bps/src/pydae/bps/` — power-system component library. `avrs/`, `govs/`, `syns/`, `vscs/`, `vsc_ctrls/`, `vsc_models/`, `vsgs/`, `wecs/`, `weccs/`, `ppcs/`, `pvs/`, `psss/`, `pssdesigner/`, `pods/`, `loads/`, `lines/`, `sources/`, `miscellaneous/`. Each module ships with a sibling `.hjson` fixture used by its in-module `test()`. `utils/` contains `visualizer.py` (`PowerSystemVisualizer` — draws reactance-weighted topology diagrams via networkx), `ss_num2sym.py`, `reporter.py`, and `validator.py`.
- `packages/pydae-uds/src/pydae/uds/` — unbalanced (three-phase) distribution builder. Component families: `vscs/` (largest), `vsgs/` (grid-forming), `genapes/`, `ess/`, `fcs/`, `pvs/`, `loads/`, `lines/`, `transformers/`, `shunts/`, `sources/`, `vsc_ctrls/`, `miscellaneous/`.
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
- **`N_nodes` / `N_branches` derivation (pydae-uds)**: a DC bus is specified by `{nodes:[0,1]}` and a DC line by `{bus_j_nodes:[0,1], bus_k_nodes:[0,1]}` — neither carries an explicit `N_nodes` / `N_branches` field, so any post-processor that reads the raw hjson **must** derive these from `len(nodes)` / `len(bus_j_nodes)` first (otherwise a default-to-4 will route DC entries through the 4-wire AC code path and reach for non-existent `V_*_3_r` or `i_l_*_2_*` variables). `UdsBuilder.preprocess` handles this for the build path; reporters like `model2svg` follow the same pattern.
- **damp() sort**: `ssa.damp(A, sort='damp')` or `sort='freq'` sorts the printed table and the returned dict by damping ratio or frequency.
- **eig() side-effects**: `ssa.eig(model)` populates `model.eigenvalues`, `model.right_eigenvectors`, `model.left_eigenvectors`, and `model.participation` in one call — prefer it over computing eigenvalues manually with `numpy.linalg.eig`.

## Docs

```bash
# build one subproject
uv run sphinx-build -b html docs/pydae-core docs/pydae-core/_build/html
uv run sphinx-build -b html docs/pydae-bps  docs/pydae-bps/_build/html
uv run sphinx-build -b html docs/pydae-uds  docs/pydae-uds/_build/html
```

**Auto-generated model pages** — `pydae-bps` and `pydae-uds` ship a model-doc generator that walks the component sources, extracts each file's raw module docstring (preserving LaTeX) and its `descriptions()` list, and writes one Markdown page per model under `docs/<pkg>/models/<family>/`. Regenerate any time a model is added or its docstring/descriptions changes:

```bash
python docs/pydae-bps/_scripts/generate_model_pages.py
python docs/pydae-uds/_scripts/generate_model_pages.py
```

The bps generator skips the same-named family dispatcher (`syns.py`, `avrs.py`, …) and documents its siblings. The uds version makes one exception: when a family directory contains only the same-named file (`shunts/shunts.py`, `lines/lines.py`), that file *is* the model and is documented. Canonical reference implementations of the `descriptions()` schema: `packages/pydae-bps/src/pydae/bps/syns/milano4ord.py` (transmission) and any of the documented uds components used by `pydae_examples/grids/grid_urisi/acdc_7bus/acdc_7bus.hjson` (distribution).

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
