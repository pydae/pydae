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

**Synchronous machines — model choice.** Two families coexist:

- **PSS/E-style family** (preferred for new work; terminal-referred
  $X_d''$, $X_q''$ per IEEE 115 Eq. (88): $X_{ds} = X_{ads} + X_l$, so
  **no `X_l` field**):
  - `syns/gencls.py` — classical 2nd-order ($\delta, \omega$, constant
    $E$ behind single $X'$). PSS/E `GENCLS`. First-swing studies,
    distant-equivalent machines.
  - `syns/gensal.py` — salient-pole 5th-order. IEEE 1110 Model 2.1 /
    PSS/E `GENSAL`. Hydro units. D-axis saturation only ($S_q=0$).
  - `syns/genrou.py` — round-rotor 6th-order with *quadratic-offset*
    saturation $S(E)=B_{sat}\max(E-A_{sat},0)^2/E$. IEEE 1110 Model 2.2 /
    Anderson-Fouad / PSS/E `GENROU`. Thermal/turbo units, default for
    NTS Tabla 45 / IEEE 115 / PSS/E `GENROU` data.
  - `syns/genroe.py` — identical to `genrou` except for the *geometric*
    saturation $S(E)=B_{sat}E^{N_{sat}}$ (PSS/E `GENROE`). Use when
    the data source is PSS/E `GENROE` or PES-TR18 (two-area, IEEE 39).
    Collapses to `genrou` when $S_{10}=S_{12}=0$.
- **Marconato / Sauer-Pai family** — `syns/milano{2,3,4,6}ord.py`
  (subtransient-minus-leakage stator form, Marconato rotor coupling).
  A fully supported modelling option, not deprecated. Use these when
  the source power-system data is itself in Marconato / Sauer-Pai form
  (e.g. existing milano-based benchmarks). The two families produce
  different operating points and mode shapes on identical NTS-style
  input, so do not freely substitute one for the other — pick the
  family that matches how the network was modelled.

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

Both builders accept `use_casadi=` and expose `.construct(name)`, `.sys_dict`, `.dae`, `.build(name)`, and `.build_mlx(name, **kwargs)` — the last being `construct()` + `MlxBuilder(sys_dict, **kwargs).build()` in one call (no C compilation; kwargs forwarded to `MlxBuilder`). `MathBackend` itself is still a two-way SymPy/CasADi switch and will stay that way: MLX has no symbolic layer, so it is a *lowering target* consumed after `sys_dict` exists, not a third construction backend. `construct()` is misspelled `contruct_grid()` internally for the grid-assembly step — that name is load-bearing, do not "fix" it.

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
REEC_B and REGC_A are nested inside a `weccs` HJSON entry (same pattern as an AVR inside `syns`). REPC_A is a separate plant-level entry in `ppcs`. Available variants: `reec_b`, `reec_e`; `regc_a`, `regc_b`, `regfm_a1`, `regfm_b1`; `repc_a`, `repc_d`, `sppc`. **`ppcs/sppc.py`** is a minimal *open-loop* plant controller: its builder just registers `p_poi_ref`/`q_poi_ref` inputs and records a `pvs`/`bess` converter list in `sys_dict['sppc']`; its dispatch is a **discrete** controller (`SppcController`, `make_discrete_controller`) running on the **`"high"` (~200 ms) tier**, broadcasting those refs to each converter's `p_s_ppc`/`q_s_ppc` (or splitting by count with `share:true`). It **freezes** the dispatch while the regulated bus voltage (`reg_bus`, default the ppc name) is below `freeze_v` (default 0.8 pu) so it doesn't fight the local FRT relay during a fault, resuming on recovery. No DAE states/equations.

**AGC (Automatic Generation Control)**: activated by an `agc` key in the HJSON data dict. `BpsBuilder.construct()` calls `add_agc(self)` after all other builders, so the governor's `p_c_{gen}` (or `p_m_{gen}`) already exists in `u_ini_dict`/`u_run_dict` — `add_agc` pops it and replaces it with an algebraic variable driven by a PI on rotor speed. Config format:

```hjson
agc: {gen: "2", K_p_agc: 10.0, K_i_agc: 2.0}
```

`gen` is the generator name (matches the bus name used in `syns`). Outputs `p_agc` and `xi_agc` in `h_dict`.

### Dynamic thermal rating (DTR) lines — `pydae-bps/lines/`

A `dtr` block on a `lines[]` entry (or a shared `dtr_codes` entry it references) turns an ordinary π-section into a **thermally coupled** line: the conductor temperature becomes a DAE state and the series admittance `g_{line}`/`b_{line}` becomes an *algebraic variable* evaluated from `R(T) = R_20·(1 + α_R·(T − 20))`, so temperature feeds back into losses and voltage drop. One `R_20` (from the line's `R_km`) and one `alpha_R` drive both the admittance and the heat balance.

Two thermal models, dispatched on the resolved `dtr` data:

| Model | Module | Physics |
|---|---|---|
| `overhead` (default) | `lines/line_dtr.py` | IEEE-738 bare conductor in air: Joule + solar in, forced/natural convection + radiation out. One conductor node. |
| `cable` (`dtr: {model: "cable", …}`) | `lines/line_dtr_cable.py` | Buried cable: heat conducted out through insulation/sheath/jacket into soil. IEC 60287 resistances, IEC 60853 two-node ladder — **or** an n-stage Cauer ladder. |

**Two-node vs n-stage — and when the two-node form is not enough.** The IEC 60853 form is `θ_c —T1— θ_e —T4— θ_soil` with Van Wormer-split capacitances `Q1`/`Q2`. Its two time constants on the CIGRE MV benchmark are ~2 min and ~1.5 days, and **a daily load cycle falls straight into the gap between them**: the two-node model reproduces the steady state exactly (same DC thermal resistance) but under-predicts the daily conductor swing by about 2×, hence the peak — the unsafe direction for a rating study. Supplying `R` and `C` as *lists* instead of `T1/T4/Q1/Q2/p` builds an n-stage Cauer ladder that closes that gap. The two parameter sets are **mutually exclusive** and giving both raises rather than silently dropping one. In the ladder form `t_sheath` does not exist; the interior nodes are `t_ladder_{i}_{line}` and node 1 is always `t_conductor_{line}` (the admittance equation is written against that name).

Three things here are load-bearing:
- **The ladder takes the *total* loss `(1+λ₁)·W_c + W_d` at node 1**, where the two-node form injects `W_c` at the conductor and `λ₁·W_c` at the external node. That is not a simplification: `cable_fem.identify_ladder` drives the FEM with the real *spatial* loss distribution (W_c in the conductor, W_s at the screen) and reduces the result, so the identified R/C already carry the split. Re-splitting it here would double-count it.
- **A DTR line appends algebraic equations during `add_lines()`, i.e. *before* the bus block.** Device injectors write into `dae['g'][idx_bus*2]` assuming the bus block sits at offset 0, and the vsource/AVR ini-run swaps assume `g[i]` is paired with `y_ini[i]`. `contruct_grid()` therefore **rotates the pre-bus entries to the back of `g`/`y_ini`/`y_run`** (no-op when nothing precedes the bus block). Without that rotation a DTR model fails `ini()` with a singular Jacobian that looks like a thermal-model bug and is not one.
- **Ladder node temperatures are time constants, not radii.** The stages come from a Foster→Cauer identification of the FEM step response, so an interior node is *not* a physical layer. Only the conductor node and the total `sum(R)` (the exact DC conductor-to-soil resistance per unit of total loss) have a direct physical reading. Locating a physical surface in the ladder means walking the resistance out from the conductor — see `cover_temperature()` in `cable_fem_example.py`.

**FEM identification — `lines/cable_fem.py`** works out *what the dtr parameters should be* for a real, geometrically resolved circuit, in four stages:

```
1. geometry     Layer / CableDesign / Installation / Backfill / CableCase
                concentric layer stack, flat or trefoil, native soil or backfilled trench
                CABLE_LIBRARY ships the CIGRE TB 575 European-MV benchmark (NA2XS2Y_120_AL)
2. field        ThermalFEM — graded tensor mesh + adaptive refinement (no external mesher),
                -div(k grad T) = q_v, Robin ground surface, Dirichlet far field
3. ladder       identify_ladder() — rational Krylov reduction of Z(s) = cᵀ(K+sM)⁻¹b,
                Foster form (positive residues => realisable) -> Cauer by continued fractions
                => LadderModel; ladder_dtr() emits the R/C block
4. two-node     fit_two_node() — T1 analytic from the layer stack, T4 from the FEM conductor
                temperature, Q1/Q2 least-squares on the step response with DC pinned
                => TwoNodeFit; .as_dtr() emits the block; verify_with_pydae() builds the real
                   pydae-bps model with it and checks against the FEM
```

- **`T4` from the FEM embeds the mutual heating of the neighbouring phases** (~30 % of the conductor rise on the benchmark), which the isolated-cable Kennelly formula cannot. `library_defaults()` / `compare_with_library()` quantify the difference — on the benchmark the library defaults would rate the cable 459 A against the FEM's 297 A.
- **The single-phase equivalent is exact only under balance.** `pydae.bps` is positive-sequence (one current per line); thermally the three cables are coupled, but under balanced loading superposition collapses the 3×3 mutual matrix to `Z_hot = Z_self + 2·Z_mutual`, so one ladder suffices *provided T4 carries that sum*. `LadderModel.z_self` / `z_mutual` are reported separately because the collapse is invalid the moment the phases are unbalanced.
- **Soil capacitance is lumped over a finite shell** of outer diameter `soil_shell × De` (default 3) rather than IEC 60853's exponential-integral response — the standard simplification for real-time/control studies. Tune `soil_shell`, or override `Q2`/`T4`, to match a reference.
- **`scikit-fem`/`scipy`/`matplotlib` are imported lazily**, inside the functions that need them, so importing `pydae.bps` never pays for them: `pip install "pydae-bps[fem]"`.

**Worked example — `lines/cable_fem_example.py`** narrates the whole path end to end (geometry → field → ladder → both dtr blocks → ampacity → `ini()` → a 24 h load cycle through the real simulator → figures), building *both* thermal models side by side so the cost of the truncation is visible rather than asserted:

```bash
python -m pydae.bps.lines.cable_fem_example --plot .     # ~17 s, writes cable_cycle.png + cable_field.png
python -m pydae.bps.lines.cable_fem_example --dt 3600 --no-step-check
```

The cycle is set by `CYCLE_BASE` / `CYCLE_AMPLITUDE` (amperes, mean ± swing) and starts from the steady state at the *mean* current so the day carries no start transient of its own. `CURRENT` is separate — it is the characterisation current for steps 1–6, not the cycle. The script also self-checks: it pushes the model's own `W_total` series through the same R/C integrated in plain Python and confirms the residual is first order in `dt` (a wrong network sits on a floor instead).

The same path expressed as assertions is `tests/bps/test_cable_fem_dtr.py` (markers `bps`, `slow`, some `model`), where every link is checked against something that does *not* come from that link: the FEM field against Kennelly + the analytic layer-stack formula, the Cauer ladder against a direct backward-Euler transient FEM solve, the exported block against the parameter contract `line_dtr_cable.descriptions()` publishes, and the `run()` trajectory against an independent scipy integration.

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

### MLX backend (Apple Silicon) — `builder/mlx_builder.py`, `builder/mlx_sparsity.py`, `builder/mlx_sweep.py`

A third backend alongside SymPy→C and CasADi. `MlxBuilder(sys_dict).build()` lowers the equations onto `mlx.core` graphs (accepting **either** SymPy or CasADi `SX` expressions — `_sympy_to_mlx` uses `lambdify` onto an mlx-backed numpy shim, `_casadi_to_mlx` replays the `SX` instruction tape), takes Jacobians by MLX autodiff, and wraps the Newton iteration in `@mx.compile`. Both classes are exported lazily from `pydae.core.builder` so importing pydae on non-Apple platforms still works; install with `pip install pydae[mlx]`.

**Do not reach for it as a faster scalar solver** — a single small system cannot keep a GPU busy and the per-op dispatch cost is paid either way. Its reason to exist is `MlxSweep`.

**Structural sparsity (`builder/mlx_sparsity.py`, `sparse=True` by default).** Every Jacobian — ini, trapezoidal, `Fx`/`Fy`/`Gx`/`Gy`/`Hx`… and the augmented SSA block — is built by Curtis-Powell-Reid **column compression**: columns that share no equation are perturbed together, so one forward-mode `jvp` per colour replaces one reverse-mode `vjp` per equation. Measured: `genrou` 16 columns → 7 passes, `k12p6` 87 → 8, a 24-machine system 507 → 11. `model.patterns['ini']` reports nnz, colours and pass count.

Four things here are load-bearing:
- **Dense rows go to reverse mode instead.** One dense equation makes every pair of columns conflict and collapses the colouring to one colour per column. The COI speed `omega_coi = sum(H_i omega_i)/sum(H_i)` is exactly that row and gets denser as the network grows — on a 24-machine system it is the difference between 11 passes and 29. The densest few rows are excluded from the conflict graph and recovered with one `vjp` each (a cut-down Coleman-Verma bicolouring).
- **The pattern must be deduplicated.** Recovery accumulates, so a repeated `(row, col)` doubles that entry. The trapezoidal pattern hits this whenever a state appears in its own derivative: the step's explicit `x_n` and the equation's own reference land on the same diagonal slot.
- **Recovery uses structured ops only** (`take` along the colour axis, transpose, multiply by a mask), never a flat list of `n*n` scalar slot indices — the per-element form measured *slower* than dense reverse mode on a 121-equation model and threw the whole saving away.
- **The scalar choice is measured, not ruled.** At B=1 `vmap` batches the identity rows into one dispatch, so 87 passes cost barely more than 8 and the fixed recovery cost can dominate; compression is 1.3-1.6x ahead on trigonometric power-system models and ~15% behind on models built from many trivially cheap equations. `build()` times both on the seed point and keeps the winner (`model._sparse_used`). Batched Jacobians (`_jac_ini_batched`, `_jac_trap_batched`, used by `MlxSweep`) are always compressed — there each pass carries real work, worth up to 106x on the Jacobian alone at B=10,000.

**The linear solve stays dense**, deliberately and by measurement: MLX arrays are immutable, so the in-place update of any sparse factorization (`values.at[dst].add(...)`) copies the whole value array per pivot. A batched sparse LU of the `k12p6` Jacobian needs 170 stages = 331 ms over a `(2246, 10000)` array, against 603 ms for the dense `mx.linalg.solve` it would replace, plus a static pivot order that can fail. Don't rebuild it. Every build checks `J @ tangent` against a plain `jvp` and refuses to build on a mismatch; `sparse=False` forces plain reverse-mode AD.

**`MlxSweep` — B copies of one DAE at once** (`model.sweep({...})`). The lowered functions are elementwise, so passing `(n_vars, B)` arrays instead of `(n_vars,)` evaluates B independent systems in one graph with no change to the lowering; B=1 reproduces the scalar path exactly. Measured on `genrou`: 10,000 steady states in 0.29 s (381x the scalar `ini`), 50,000 in 0.84 s (704x); batched transients reach 0.013 ms/step/case at B=10,000.

```python
m.ini({'p_m_1': 0.5})                                     # nominal anchor + seed
sw = m.sweep({'p_m_1': np.linspace(0.05, 0.95, 10_000)})
sw.ini(); q = sw.get_values('q_g_1')                      # (B,), NaN where infeasible
sw.run(5.0, {'p_m_1': 0.65}, record=['omega_1'])          # (n_samples, B)
sw.A_eval(); sw.eigenvalues()                             # damping locus vs the swept value
```

**Memory.** Scalar simulation is free (`k12p6`: 0.42 MB of MLX allocation, 0.53 MB of trajectory buffers for 10 s at Dt=0.01; the ~287 MB RSS is ~214 MB of Python + numpy/sympy/casadi). Batched peak tracks **`chunk`, never `B`** — 10k and 50k cases peak identically — at about `8·n²·itemsize` per case (measured 245.6 kB at n=87 float32 against a 250 kB estimate; 10.6 vs 10.5 kB at n=16). `_auto_chunk` caps at **4096** because throughput saturates there: on `k12p6` chunk 4096 is 0.96 s/996 MB against 8192's 0.97 s/1988 MB and 16384's 1.05 s/3943 MB — flat in time, linear in memory, slower beyond. `memory_budget_gb` (2.0) is a ceiling on top of that; lower it for memory, never raise it for speed. Compression cut peak memory 16-33× as well as time (chunk 4096: 986 MB sparse vs 15.4 GB dense). **`mx.get_peak_memory()` under-reports the CPU stream**, so it cannot check float64 (k12p6 float64 measures 253 kB/case vs float32's 246 kB, where it should be double) — use RSS there.

Three things load-bearing enough not to "simplify":
- **A swept name is written to every block that carries it** (`_locate`), unlike `_route_dict`'s first-match. `p_m_1` is in both `u_ini_dict` and `u_run_dict`; writing only the ini copy initialises correctly and then runs at nominal dispatch.
- **Converged/failed cases are frozen by substituting an identity system inside the compiled kernel.** MLX aborts the *entire* batched `linalg.solve` if any one element is singular, so masking only the update is not enough.
- **Non-convergence is data, not an error.** A wide sweep legitimately contains infeasible points — `p_m` at fixed `v_f` stops solving at the pull-out limit `E·V/X_d`, which `feasible_range()` reports (0.702 measured vs 0.7026 analytic). `ini()` never raises unless `require_all=True`.

MLX limits worth knowing: `float64` is CPU-only, `linalg.solve`/`eigvals` are CPU-only at any dtype, and `sin`/`cos`/`exp` are float32-accurate *even on float64 arrays*, so residuals floor near 1e-8 regardless — hence the Newton driver also accepts a solve that demonstrably stops improving (`stall_tol`).

**Sparse small-signal analysis** (`builder/mlx_ssa.py`, `model.small_signal(Shift=...)`). Solves the *generalized* problem `Λ_ξ φ = λ E φ` on the augmented state `ξ = [x; y]` with `E = diag(I, 0)`, so `A = Fx - Fy·Gy⁻¹·Gx` is never formed and `Gy` is never inverted. The pencil's `N_x` finite eigenvalues equal `eigvals(A)` (verified to 1e-13); its `N_y` infinite ones collapse to `μ = 0` under shift-invert and filter themselves out. `Λ_ξ v` is an `mx.jvp` pass and `Λ_ξᵀ w` an `mx.vjp` pass (the latter gives the left eigenvectors), both `@mx.compile`d; the shift-invert solve is matrix-free GMRES over those products. Returns eigenvalues, right/left eigenvectors and Kundur participation, all matching the dense path to ~1e-12.

Four things here are load-bearing:
- **Modes are selected on the Ritz residual, not on `|μ|`.** A float32 Krylov basis loses orthogonality (measured `‖VᵀV − I‖ = 0.9`) and emits ghost Ritz pairs with plausible `|μ|`; ranking by `|μ|` let one displace three genuine eigenvalues.
- **`Restart` defaults to the full system order.** Restarted GMRES stagnates on an augmented power-system Jacobian — on Kundur two-area (n=87), `Restart=60` flatlines at 4e-1 and the eigenvalues come out wrong by 1.55, while full GMRES reaches 5e-12.
- **Left eigenvectors normalise with the plain transpose `ψᵀEφ = 1`, never `ψᴴEφ`.** For a complex mode `conj(ψ)` is the left vector of the *conjugate* eigenvalue, so the conjugate-transpose product is ~0 by biorthogonality and normalising against it divides by noise.
- **Unconverged modes are returned but flagged** (`mode_converged`, `mode_residual`, plus a `RuntimeWarning`) — silently returning the least-bad Ritz pairs would look exactly like eigenvalues.

`matrix_free=False` instead forms `Λ_ξ` (compressed, via the `'aug'` pattern) and factorises it on the CPU: 0.02 s vs 0.83 s on genrou, 22 s vs 74 s on Kundur two-area, same answers. Matrix-free is for when memory, not time, is the constraint.

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

### Continuous Control Blocks (pydae.control)

`packages/pydae-core/src/pydae/control/continuos/` — standalone control-system building blocks (`pi`, `pid`, `lpf1`, `lpf2`, `leadlag`, `washout`, `deadband`, `saturation`, `rate_limiter`, `integrator`, `notch`, `tf`, `lti`, `pll`, `park`/`park_inv`, `kalman`, `luenberger`, `matrix_gain`, `product`, `summation`, `pi_feedforward`, `stab_1lpf_2wo_3ll`), each sharing the signature `block(dae, name, backend, data_dict=None, input=None, output=None)`, which appends into a `dae`-shaped dict (`f_list`, `x_list`, `g_list`, `y_ini_list`, `y_run_list`, `u_ini_dict`, `u_run_dict`, `params_dict`, `h_dict`, `xy_0_dict`). `ControlBuilder` (`pydae.control.control_builder`) plays the same role as `BpsBuilder` for a standalone `"control": [...]` list, dispatching each item by `"type"` through the `BUILDERS` registry.

Because `grid.sys_dict` (after `BpsBuilder.construct()`) has the same keys, `pydae.control.continuos` blocks can be appended **directly to a power-system `sys_dict`** — e.g. wiring a POD/stabilizer onto a generator speed signal, using `grid.backend` (the same `MathBackend` instance, which caches symbols by name) to retrieve existing symbols like `omega_1`.

**`stab_1lpf_2wo_3ll(dae, name, backend, data_dict=None, input=None, output=None)`** — 2-washout/3-lead-lag PSS/POD stabilizer chain (`lpf1 -> washout -> washout -> leadlag x3 -> saturation`, zero steady-state gain). `input`/`output` alias *existing* symbols (a state like `omega_1`, or an existing `u_ini`/`u_run` setpoint) directly, skipping the extra `u_stab_.../z_stab_...` wrapper symbols and connecting equations — if `output` names an existing input it is promoted to an algebraic variable driven by the saturation. `K_stab` (default 1.0) is an overall gain applied *before* the output saturation via `z = K_stab * clip(z_ll3, -Limit/K_stab, Limit/K_stab)` — written this way (not `clip(K_stab*z_ll3, -Limit, Limit)`) so the clip always operates on an O(1) signal even when `K_stab` is a large unit-conversion factor.

**Algebraic variables must stay O(1) for IDAS `calc_ic`.** When wiring a control-block output into `y_run_list` via `output=`, do not let its physical units (e.g. W, with `K_stab ~ 1e8`) put it directly into the algebraic-variable vector — `model.run()`'s `IDACalcIC` (`calc_ic: True`) re-solves all algebraic variables with uniform tolerances on every `set_value`/step, and a ~10⁶ value mixed with O(1) pu variables causes `IDA_LINESEARCH_FAIL`. Instead, keep the block's output in pu (`output=None`, modest `K_stab`/`Limit`) and add the unit-converted signal as a **plain `h_dict` expression** (`grid.sys_dict['h_dict']['P_5'] = z_pu * 1e8`) — `h_dict` outputs are computed post-hoc from `(x, y, p)` and never participate in `calc_ic`.

### Real-time API (pydae.api)

`packages/pydae-core/src/pydae/api/realtime_api.py` — soft real-time FastAPI server wrapping a `CasadiModel` in a background thread.

**Usage pattern:**
```python
from pydae.api.realtime_api import app
app.state.model        = model          # CasadiModel, already ini()-ed
app.state.chunk_ms     = 50.0           # integration chunk in ms
app.state.solver       = "idas"         # per-chunk integrator: "idas" | "trapezoidal" (faster)
app.state.ramp_chunks  = 20             # chunks to spread each setpoint change over
app.state.cosim_config = grid.data.get("configs", {})  # optional co-sim config
# optional injected bps layers (duck-typed; core never imports pydae-bps):
app.state.discrete     = DiscreteOrchestrator.from_model(model)  # per-chunk discrete control
app.state.faults       = FaultController(model, iq=iq_po123)     # fault apply/clear
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/measurements` | All h_dict outputs (browser-friendly) |
| `POST` | `/measurements` | Selective h_dict outputs by name list |
| `GET` | `/inputs` | All runtime input (u_run setpoint) values |
| `GET` | `/cosim/measurements` | Only measurements declared in `configs` |
| `POST` | `/setpoints` | Write any u_run inputs (ramped); body: `{"setpoints": {...}, "timestamp": float\|null}` **or** a flat `{name: value}` mapping (no-`setpoints`-key body is treated as flat) |
| `POST` | `/cosim/setpoints` | Write setpoints restricted to `configs` — rejects unknown keys with 422 |
| `POST` | `/set_input` | Write a single input (ramped) |
| `GET` | `/status` | `{"is_running": bool, "t_sim": float}` |
| `GET` | `/faults` | `{"declared": [...], "active": [...], "t_sim": float}` (501 if no fault controller) |
| `POST` | `/faults/apply` | Apply a shunt fault; body `{"bus": str, "t_clear": float\|null, "X_fault": float\|null, "fault_steps": int\|null}` (422 unknown bus). `X_fault` overrides the reactance (larger = softer); fewer `fault_steps` = faster soft fault |
| `POST` | `/faults/clear` | Clear a shunt fault; body `{"bus": str}` |

**Key design points:**
- **Per-chunk solver** is selectable via `app.state.solver`: `"idas"` (default; adaptive SUNDIALS, robust) or `"trapezoidal"` (fixed-step implicit trapezoid). The trapezoid reuses the model's cached fixed-step CasADi functions (`_build_fixed_step_fns`) + a modified-Newton with factorization reuse, so it avoids IDAS's per-chunk cold restart and runs ~3× faster on a settled plant with the same trajectory. It advances `trapezoidal_substeps` sub-steps/chunk (default `round(chunk_sec/Dt)`); the factorization is refreshed on schedule, adaptively, and after a fault. The Newton linear backend is `app.state.trapezoidal_linsol` (`"scipy"` default, or a `ca.Linsol` plugin like `"qr"`).
- Setpoint changes are ramped over `ramp_chunks` chunks to avoid `IDA_LINESEARCH_FAIL` on large steps.
- The background thread catches `RuntimeError` from failed IDAS steps and logs them rather than crashing.
- CORS is open (`allow_origins=["*"]`) for browser and co-simulation client access.
- `cosim_config` uses the `configs` section of the co-simulation JSON. Variable names are resolved via `emec_name` (exact), `emec_prefix` (prefix match), or `emec_template` (regex with `<emec_id>` backreference). Measurements resolve against `h_dict`; setpoints against `u_run_names`.
- **`discrete` / `faults` are injected duck-typed** (`app.state.discrete` needs `step(model[, t])`; `app.state.faults` needs `apply(bus)`/`clear(bus)`/`buses`), so `pydae-core` keeps **no dependency on `pydae-bps`**. The loop calls `discrete.step(model, t_sim)` before each chunk and services fault apply/clear + auto-clear **under the integrator lock**. See "Discrete control & faults (pydae-bps)" below.
- A fault apply/clear runs a blocking continuation under the lock, so afterwards the wall-clock pacing is **re-baselined** (`_rebase_pacing`) — the lag is dropped, not fast-forwarded, so real time resumes cleanly and a short `t_clear` fault stays observable to a polling client. Sub-millisecond overruns are not reported.

### Discrete control & faults (pydae-bps)

Two cooperating subsystems in `pydae-bps` model **discrete control** and **shunt faults**. Both keep a *generic* layer separate from the *particular* per-model behaviour, and both are consumed identically by the offline `model.run` loop and the real-time API (the API only ever sees the generic, duck-typed hooks — it never imports `pydae-bps`).

**Discrete control orchestration** (`pydae.bps.discrete.DiscreteOrchestrator`). A DAE model is continuous; some elements also need a per-step discrete part (read outputs, write inputs before the next integration). `DiscreteOrchestrator` is the generic hook: `from_model(model)` discovers discrete-capable models (an explicit `DEFAULT_DISCRETE_MODULES` list + a `register_discrete` registry — **no blind package scan**, since the model dirs hold dev/test scripts), builds each model's controller, and `step(model, t)` runs them all. A controller is any object with `step(model)` (+ optional `sync(model)`), built by a module-level `make_discrete_controller(model)` factory returning `None` when absent.
- **Latency tiers**: each controller declares `level` (`"low"`=every step / `"mid"`=0.05 s / `"high"`=0.2 s, overridable `tiers=` dict) or an explicit `period` (seconds); the orchestrator gates each by the sim clock so slow centralized controls (e.g. the `ppcs/sppc.py` plant dispatcher on the `"high"` tier) run less often than a local relay (`vrt`, `"low"`). Cadence is in **seconds** (ports across offline `Dt` and RT `chunk_sec`). Tiers gate cadence, not order; `step` returns a `controllers`-aligned list (`None` for skipped). `reset_schedule()` re-fires all after a discontinuity.

**Faults** (`pydae.bps.utils.faults`). A fault is a balanced shunt impedance $Z_f=R+jX$ switched onto a bus, declared in a `"faults"` data section (`{bus, R_pu, X_pu, S_mva}`) that `BpsBuilder` turns into an *off* shunt plus a target admittance in `model.sys_dict['faults']`. The shunt exposes `g_shunt_<bus>`/`b_shunt_<bus>` **parameters**.
- `FaultManager(model, bus)` — **generic**: robustly ramps the bus admittance in/out via frozen-time **continuation** (adaptive step + snapshot/restore + per-step fixed-point relaxation of an `on_step` callback). Deep faults (V→0 with constant-power converters) only converge because the converters reach bounded current injection *through* the continuation, not a time-domain ramp. `recalculate_algebraics` mutates only `y_run`.
- The converter's *particular* ride-through is injected as `on_step`, and lives as **shared grid-following-converter behaviour** in `utils/frt.py`, re-exported by both `pvs/pv_pq_ss.py` and `vscs/bess_pq_ss.py` (they share the `lvrt_ext_<name>`/`i_sr_ref_<name>`/`i_sa_ref_<name>` interface), so one controller — discovered via the single `utils/frt` factory (`DEFAULT_DISCRETE_MODULES`) — drives both PV and BESS. **Two grid-code laws** are available, selected by `DiscreteOrchestrator.from_model(model, law=...)` (factories take `**_ignored`, so the kwarg broadcast is harmless):

| `law` | Time-domain relay | Continuation callback | Curve |
|---|---|---|---|
| `"po123"` (default) | `vrt()` / `VrtController` | `lvrt_response(devices, iq=...)` | `iq_po123(V)` — absolute PO 12.3 dead-band curve, latched over a 0.85/0.9 voltage hysteresis band |
| `"ted749"` | `Ted749Controller` | `ted749_response(devices, ...)` | `iq_ted749(...)` — Orden TED/749/2020 (BOE-A-2020-8965) apartado 2.3.1 / figura 8 |

- **Prefer `"ted749"` for deep faults on converter-dominated plants.** It is not just a different curve, it is a *continuous proportional* control — `ΔI1 = K_1·(U1i − U1)` incremental to the pre-disturbance reactive current, total clipped to `I1_limU/I1_limL` (defaults ±0.9 pu), `2 ≤ K_1 ≤ 6` — plus active current from the apparent-current headroom `sqrt(I_max² − I1²)` capped at the pre-fault value (apartado 3.3.1). Two consequences are load-bearing for the solver, not just for compliance:
  - **No binary voltage threshold**, so the `FaultManager` fixed-point relaxation has nothing to chatter across. The PO 12.3 callback's bare `V <= v_lvrt` test makes converters flip in and out of ride-through between sweeps and stalls the continuation *at* the threshold.
  - **The converter is never commanded to inject exactly zero current.** At `V == U1i` the law returns the pre-disturbance current. A grid-following converter at exactly zero current has an **unobservable terminal angle** — a structurally singular `Gy` (null space all `I_*`, singular rows all `theta_*`). Adding a hysteresis band to the PO 12.3 callback trades the chatter for exactly this singularity, because `iq_po123(V)=0` *and* `i_sa_ref=0` over the whole band.
  - Arming is `V ∉ [V_min, V_max]` (0.85/1.15) **or** an abrupt change > 0.1 pu; disarming is a **time** latch — voltage back in range *and* 5 s since onset (30 s thermal ceiling), with the count restarting on any re-trigger (apartado 2.3.1 e), so the fault *clearance* transient legitimately restarts it.
  - `Ted749Controller.sync(model)` must latch the pre-disturbance operating point (`U1i`, `I1i`, active current) from its last undisturbed sample when the continuation armed the control, or the relay resumes against its defaults, commands zero active current, and steps straight off the point `apply()` just reached.
- `FaultController(model, iq=...)` — the **live-model** wrapper (apply/clear a fault without owning the time loop); it is what the real-time API takes on `app.state.faults`, running the continuation under the integrator lock. `apply(bus, X_fault=, fault_steps=, ...)` can override the reactance (softer fault) and continuation resolution per event; `fault_admittance(R_pu, X_pu, S_mva, S_base)` recomputes the shunt `(g, b)` for an overridden `X` (the builder stores `R_pu`/`X_pu`/`S_mva`/`S_base` in `sys_dict['faults'][bus]` for this). `apply_fault(model, bus, t_clear, ...)` is the older offline convenience wrapper that advances `model.run` itself.

## Import Paths

```python
from pydae.core import Builder, Model
from pydae.core.builder import CasadiBuilder, CasadiModel, MathBackend  # CasADi backend
from pydae.bps import BpsBuilder
from pydae.bps.discrete import DiscreteOrchestrator          # per-step discrete control (latency tiers)
from pydae.bps.utils.faults import FaultManager, FaultController, lvrt_response
from pydae.bps.utils.faults import ted749_response               # Orden TED/749/2020 continuation callback
from pydae.bps.utils.frt import vrt, VrtController, iq_po123     # shared FRT relay + PO 12.3 curve (also re-exported by pv_pq_ss / bess_pq_ss)
from pydae.bps.utils.frt import Ted749Controller, iq_ted749      # Orden TED/749/2020 relay + law
from pydae.bps.lines.cable_fem import CableCase, run_suite, ladder_dtr  # FEM -> dtr identification (needs pydae-bps[fem])
from pydae.uds import UdsBuilder
from pydae.ssa.ssa import eig, damp, participation
from pydae.control.control_builder import ControlBuilder
from pydae.control.continuos.stab_1lpf_2wo_3ll import stab_1lpf_2wo_3ll  # and pi, pid, lpf1, lpf2, leadlag, washout
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
- `packages/pydae-bps/src/pydae/bps/` — power-system component library. `avrs/`, `govs/`, `syns/`, `vscs/`, `vsc_ctrls/`, `vsc_models/`, `vsgs/`, `wecs/`, `weccs/`, `ppcs/`, `pvs/`, `psss/`, `pssdesigner/`, `pods/`, `loads/`, `lines/`, `sources/`, `miscellaneous/`. Each module ships with a sibling `.hjson` fixture used by its in-module `test()`. `discrete.py` (the `DiscreteOrchestrator` per-step control hook) sits at the package root. `lines/` also holds the DTR stack: `line_dtr.py` (IEEE-738 overhead), `line_dtr_cable.py` (IEC 60853 two-node / n-stage Cauer cable), `cable_fem.py` (the scikit-fem identification suite behind the `[fem]` extra) and `cable_fem_example.py` (the narrated end-to-end walkthrough; `--plot` writes `cable_cycle.png` / `cable_field.png`, both gitignorable artefacts). `utils/` contains `visualizer.py` (`PowerSystemVisualizer` — draws reactance-weighted topology diagrams via networkx), `faults.py` (`FaultManager` / `FaultController` / `lvrt_response`), `frt.py` (the shared grid-following FRT relays: `vrt` / `VrtController` / `iq_po123` for PO 12.3 and `Ted749Controller` / `iq_ted749` for Orden TED/749/2020, re-exported by `pvs/pv_pq_ss` and `vscs/bess_pq_ss`), `ss_num2sym.py`, `reporter.py`, and `validator.py`.
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
