# pydae-core: Architecture and Internal Design

## What is pydae-core

pydae-core is the foundational package of the pydae monorepo. It provides a
complete pipeline for turning a symbolic Differential-Algebraic Equation (DAE)
system into a compiled C shared library or a CasADi-native solver, and then
driving that solver from Python to perform steady-state initialization and
time-domain simulation.

The package is organized around two orthogonal axes — **Builder** (symbolic
input) and **Model** (runtime engine):

```
pydae/core/
├── builder/
│   ├── sympy_builder.py      ← SymPy → C pipeline (Builder)
│   ├── casadi_builder.py     ← CasADi → SX graph (CasadiBuilder)
│   └── codegen/              ← C code generation (cffi / ctypes)
├── model/
│   ├── ctypes_model.py       ← ctypes/CFFI runtime (Model)
│   └── casadi_model.py       ← CasADi runtime (CasadiModel)
├── common/
│   ├── parser.py             ← system dict parsing & validation
│   └── symbolic.py           ← symbolic Jacobian computation
└── diagnostics/              ← Jacobian health checks
```

A DAE system in pydae has the semi-explicit index-1 form:

```
dx/dt = f(x, y, u, p)        (differential equations)
    0 = g(x, y, u, p)        (algebraic constraints)
    z = h(x, y, u, p)        (output equations)
```

where `x` is the vector of dynamic states, `y` is the vector of algebraic
variables, `u` is the vector of inputs, `p` is the vector of parameters, and
`z` is the vector of outputs.

## The Build Pipelines

### SymPy Pipeline: `Builder` → `Model`

The user defines the system as a plain Python dictionary containing SymPy
expressions. For example:

```python
sys_dict = {
    "name": "pendulum",
    "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3},
    "f_list": [v_x, v_y, (-2*p_x*lam + f_x - K_d*v_x)/M, ...],
    "g_list": [p_x**2 + p_y**2 - L**2 - lam*1e-6, ...],
    "x_list": [p_x, p_y, v_x, v_y],
    "y_ini_list": [lam, f_x],
    "y_run_list": [lam, theta],
    "u_ini_dict": {"theta": 0.087},
    "u_run_dict": {"f_x": 0},
    "h_dict": {"E_p": M*G*(p_y + L), "E_k": 0.5*M*(v_x**2 + v_y**2)},
}
Builder(sys_dict, target="ctypes", sparse=False).build()
```

Calling `Builder.build()` triggers a four-phase pipeline:

**Phase 1 — Parsing (`common/parser.py`).** The `check_system` function
validates the dictionary: it ensures that both `f_list` and `g_list` are
present (adding dummy equations when necessary), checks for duplicate
variables, and determines whether a separate initialization/run split is
needed. Then `process_system_dict` converts all variable name strings into the
exact SymPy symbols that appear in the expressions, and wraps the equation
lists into SymPy row matrices.

**Phase 2 — Symbolic Jacobians (`common/symbolic.py`).** The
`compute_base_jacobians` function differentiates the equation vectors
symbolically to produce the six sub-Jacobian blocks: `Fx`, `Fy_ini`,
`Fy_run`, `Gx`, `Gy_ini`, and `Gy_run`. Then `build_large_jacobians` assembles
the two global Jacobians that the solver needs. The initialization Jacobian
is:

```
jac_ini = [[F_x,  Fy_ini],
           [G_x,  Gy_ini]]
```

The trapezoidal (run) Jacobian encodes the implicit integration rule and
includes the time step and blending parameter:

```
jac_trap = [[I - alpha*Dt*F_x,  -alpha*Dt*Fy_run],
            [G_x,                Gy_run          ]]
```

For sparse builds, this phase also computes the CSC sparsity pattern (column
pointers `Ap` and row indices `Ai`) of each Jacobian independently.

**Phase 3 — C code generation (`codegen/cffi_builder.py` /
`codegen/ctypes_builder.py`).** The `sym2c` function converts each SymPy
expression into a C code string using `sympy.ccode`. Then `sym2xyup` replaces
symbolic variable names with C array indexing (`x[0]`, `y[3]`, `u[1]`, `p[5]`,
etc.). The `_get_c_funcs` helper wraps these translated strings into C
function bodies. This produces seven C functions: `f_ini_eval`, `f_run_eval`,
`g_ini_eval`, `g_run_eval`, `h_eval`, `jac_ini_eval`, and `jac_trap_eval`.

When the number of expressions exceeds 200, the translation can be
parallelised by setting the `PYDAE_PARALLEL=1` environment variable, which
distributes `sym.ccode` calls across a `ProcessPoolExecutor`.

**Phase 4 — Compilation.** The generated model C code is compiled together
with `daesolver.c` (the generic solver) into a shared library. This can
happen through CFFI (`ffi.set_source` followed by `ffi.compile`, producing a
Python extension module) or through ctypes (invoking `gcc` or `clang` directly
to produce a `.dll`, `.so`, or `.dylib`).

The build also writes a JSON metadata file (`{name}_data.json`) containing
variable names, dimensions, parameter defaults, the sparse backend identifier,
NNZ counts, and the sparsity patterns (Ap, Ai arrays).

### CasADi Pipeline: `CasadiBuilder` → `CasadiModel`

The CasADi pipeline skips C code generation entirely. Instead, it builds a
CasADi SX computational graph:

**Phase 1 — Symbol extraction.** CasADi SX symbols are matched from the
expressions in `f_list` and `g_list`. Variables are classified into `x`,
`y_ini`, `y_run`, `p`, `u_ini`, `u_run`.

**Phase 2 — Runtime substitution.** Variables that appear in `y_ini_list` but
not in `y_run_list` (e.g., a solved reference like `v_ref`) are substituted
with their corresponding `u_run` symbols in the `f` and `g` expressions, so
the ODEs reference runtime inputs rather than initialization variables.

**Phase 3 — Solver construction.** A CasADi `rootfinder('newton')` is built
for initialization, and an `integrator('idas')` is configured for runtime
simulation. Jacobian evaluators (`A_eval`, `B_eval`, `C_eval`, `D_eval`) are
constructed symbolically via `ca.jacobian` and `ca.solve`.

The CasADi pipeline can also export the graph to C code via
`CasadiBuilder.export_code()` and load it as a precompiled binary through
`CasadiModel.from_binary()`.

## The DAE Solver: daesolver.c

`daesolver.c` is the heart of the SymPy runtime. It is a generic, reusable C
file that is compiled alongside every model. It provides two entry points:
`ini()` and `run()`.

### The `ini()` function

`ini()` implements a Newton-Raphson iteration to find the steady-state
operating point where `f(x, y, u, p) = 0` and `g(x, y, u, p) = 0`
simultaneously. On each iteration it:

1. Evaluates the residual vector `fg = [-f; -g]` by calling `f_ini_eval` and
   `g_ini_eval`.
2. Evaluates and factorises the initialization Jacobian by calling
   `jac_ini_eval` and then the appropriate linear solver.
3. Solves `jac_ini * Dxy = fg` for the correction vector `Dxy`.
4. Updates `xy += Dxy` and checks convergence via the squared norm of `Dxy`.

### The `run()` function

`run()` advances the solution in time using the implicit trapezoidal rule (or
backward Euler, depending on `alpha`). The discretized equations at each time
step are:

```
x_{n+1} - x_n - Dt * [alpha * f(x_{n+1}, y_{n+1}) + (1-alpha) * f(x_n, y_n)] = 0
g(x_{n+1}, y_{n+1}) = 0
```

At each time step, a Newton-Raphson loop:

1. Evaluates `f_run_eval` and `g_run_eval` to form the residual.
2. Evaluates `jac_trap_eval` to form the trapezoidal Jacobian and factorises
   it.
3. Solves `jac_trap * Dxy = fg` and updates the solution.
4. After convergence, optionally stores `Time`, `X`, `Y`, `Z` at the current
   step (subject to a decimation counter).

Both `ini()` and `run()` support a Jacobian re-use strategy controlled by
`intparams[0]`: value 0 means re-evaluate and refactor the Jacobian at every
Newton iteration, value 1 means only on the first iteration of each step
("dishonest Newton").

### How ctypes_model.py calls daesolver.c

The `Model` class in Python loads the compiled shared library at construction
time (either via CFFI's `importlib.import_module` or via `ctypes.CDLL`). It
allocates all working arrays as NumPy `float64` arrays and passes them to the
C functions through pointer casting. The `_d()` method handles the backend
abstraction: for CFFI, it uses `ffi.cast("double *", ffi.from_buffer(arr))`;
for ctypes, it passes the NumPy array directly (with `ndpointer` argtypes
already configured).

The `ini()` Python method prepares the initial guess vector `xy_0`,
pre-allocates the time-history storage arrays (`Time`, `X`, `Y`, `Z`), and
calls `self.solver_lib.ini(...)` with 18 arguments. After a successful solve,
it calls `ini2run()` to transfer the converged state into the runtime vectors.

The `run()` Python method accepts a target time `t_end` and an input
dictionary. It ensures enough storage capacity exists, and calls
`self.solver_lib.run(...)` with 27 arguments. Multiple `run()` calls can be
chained to simulate piecewise-constant input changes, with each call
continuing from the previous endpoint.

### How casadi_model.py runs

`CasadiModel` uses CasADi's built-in `rootfinder` for `ini()` and, for `run()`,
one of several time-domain strategies selected by
`run(t_end, update_dict=None, method=None, **opts)`. The `ini2run()` method
transfers values between initialization and runtime phases, including:

- Mapping solved `y_ini` variables to `u_run` inputs (`yini2urun`).
- Mapping `u_ini` inputs to `y_run` states (`uini2yrun`).
- Copying `u_ini` values to `u_run` for shared inputs.

#### Time-domain solver selection

`method` (falling back to the instance attribute `self.run_method`, default
`'idas_grid'`) picks one of four integration strategies. All four append to the
same `Time/X/Y/Z` storage and advance `x/y_run/t` identically, so they are
drop-in interchangeable:

| `method` | strategy | when to use |
|---|---|---|
| `'idas_grid'` (default) | one IDAS call over a *grid* of output times | one long continuous run — IDAS accuracy, ~30× faster than `'idas'` |
| `'idas'` | legacy: stateless IDAS call once per `Dt` | exact bit-for-bit reproduction of historical behaviour |
| `'trapezoidal'` | fixed-step implicit θ-method, `alpha=0.5` (2nd order) | many short / interactive / real-time steps |
| `'backward_euler'` | fixed-step implicit θ-method, `alpha=1.0` (1st order) | stiff systems needing a robust, damped step |

**Why the legacy per-`Dt` loop was slow.** A CasADi integrator is a *stateless*
function `(x0, z0, p) → (xf, zf)`: it does **not** carry SUNDIALS' internal BDF
state (past states, current order, current internal step size, the factorized
iteration matrix) between calls. The old `run()` called `ca.integrator('idas')`
once per `Dt`, so every call **cold-restarted** BDF — reset to order 1, picked a
tiny conservative initial step, and re-did the entire startup transient. On a
731-state / 981-algebraic model this measured ~11 internal steps and ~11 sparse
Jacobian factorizations per 0.01 s (≈900 ms/step, ~90 s for 1 s of simulation),
all thrown away for the next `Dt`.

**`'idas_grid'`** integrates to `t_end` in a **single** IDAS call over the grid
of output times `t + Dt*arange(1, n+1)`. SUNDIALS initializes once, ramps BDF
up, keeps and reuses its factorization across internal steps, and simply reports
the (interpolated) state at each grid point — the internal step size is chosen
adaptively and is unrelated to `Dt`, which now only sets output resolution. Same
trajectory, ~15 factorizations instead of ~110 for a 10-point span. Because the
DAE is autonomous, the integrator is built over a *relative* grid from `t0=0` and
shifted by the current time, so the object can be reused across equal-length
segments (`reuse_integrator=True`). Cold-start knobs `step0` (initial step-size
hint ≈ `Dt`) and `calc_ic` (consistent-IC recompute) are exposed for chaining
many short segments.

**`'trapezoidal'` / `'backward_euler'`** are fixed-step implicit integrators
built from `self.dae_dict` via CasADi AD — the CasADi-backend analogue of the C
`daesolver.c` trapezoidal loop, sharing its `alpha` convention (0.5 / 1.0). The
residual and iteration matrix are

```
R = [ x_{n+1} - x_n - Dt*((1-alpha)*f_n + alpha*f(x_{n+1},z_{n+1})) ]   (n_x rows)
    [ g(x_{n+1}, z_{n+1})                                           ]   (n_z rows)
J = dR/d[x,z]   (from ca.jacobian, sparse)
```

Each step runs a modified Newton iteration: `J` is factorized once and the
factorization is **reused** for `refactor_every` steps. It is refreshed on
schedule, after a `set_value` discontinuity (which marks the linear-solver
backend stale), or adaptively when a step exceeds `refactor_iters` Newton
iterations; a step that fails is retried once with a fresh factorization before
raising. Constant per-step cost, no adaptive restart.

#### Pluggable linear solver (`linsol`)

The Newton linear solve goes through a pluggable backend selected by the
`linsol` option (`model.run(..., method='trapezoidal', linsol=...)`). Both
backends follow a factorize-once / solve-many contract, since `J`'s sparsity is
fixed for the whole simulation:

| `linsol` | backend | how it factorizes |
|---|---|---|
| `'scipy'` (default) | `scipy.sparse.linalg.splu` | convert `J` (CasADi `DM`) → `scipy.sparse.csc`, then `splu` |
| `'qr'` (recommended CasADi) | `ca.Linsol('qr', …)` | stays in CasADi `DM`, no conversion |
| any `ca.Linsol` plugin, or `'casadi:<plugin>'` | `ca.Linsol(<plugin>, …)` | e.g. `lapacklu`, `csparse`, `ma27` |

The CasADi-native path uses `ca.Linsol`'s three-tier reuse API — `sfact()`
(symbolic factorization, **once**, bound to the fixed sparsity), `nfact()`
(numeric factorization, only when `J` is refreshed), and `solve()` (cheap
back-substitution, every Newton iteration, reusing `nfact`). `'qr'` is the
recommended plugin — in this reuse pattern it measured 10–60× faster per step
than `'csparse'` or `'lapacklu'`. A requested plugin that is unavailable (e.g.
an HSL solver like `ma27` with no HSL installed) falls back to SciPy with a
warning, so a run never hard-fails on a missing plugin.

`'scipy'` stays the default and its numerics are unchanged. Note that the linear
solve is only a small fraction of a trapezoidal step — per-iteration CasADi
`Function` calls and `DM ↔ numpy` marshaling dominate — so the CasADi backend is
mainly a cleanliness / no-scipy-conversion win, not a large end-to-end speedup.
A real end-to-end gain would come from reducing per-iteration Python round-trips
(evaluating `R`, `J`, and the solve in as few native CasADi calls as possible).

#### Consistent starting points and discontinuities

The fixed-step solvers and `'idas_grid'` assume a consistent starting point over
each continuous segment. Run `ini()` (or, after a parameter jump, call
`recalculate_algebraics()` to re-solve `g(x, y, p) = 0` with `x` held fixed)
before integrating. After a `set_value` discontinuity the frozen fixed-step
factorization is invalidated automatically, so the pattern is:

```python
model.run(t_step, method='trapezoidal')     # continuous segment
model.set_value('p_c_lc_G1', new_ref)        # discontinuity (auto-invalidates LU)
model.recalculate_algebraics(tol=1e-8)       # re-consistent algebraic ICs
model.run(t_end, method='trapezoidal')       # next continuous segment
```

## Why y_ini, u_ini and y_run, u_run

Many DAE systems need different unknowns and inputs during initialization
versus time-domain simulation. The pendulum example in the pydae test suite
illustrates this clearly. The pendulum has four dynamic states (`p_x`, `p_y`,
`v_x`, `v_y`) and two algebraic variables, but the role of those algebraic
variables changes between stages:

```python
"y_ini_list": [lam, f_x],
"y_run_list": [lam, theta],
"u_ini_dict": {"theta": np.deg2rad(5.0)},
"u_run_dict": {"f_x": 0},
```

During initialization, the user specifies the angle `theta` as a known input
(`u_ini`) and the solver finds the Lagrange multiplier `lam` and the
horizontal force `f_x` as unknowns (`y_ini`). This answers the question: "what
force is needed to hold the pendulum at this angle in steady state?"

During simulation, the roles swap. The angle `theta` becomes an unknown that
the solver computes at every time step (`y_run`), while the horizontal force
`f_x` becomes a known input (`u_run`) that the user can vary to apply
disturbances. This answers the question: "given this applied force, how does
the pendulum swing over time?"

pydae handles this through two separate sets of algebraic variables and inputs:

`y_ini_list` / `u_ini_dict` define the unknowns and known inputs for the
initialization stage. Newton-Raphson solves for `x` and `y_ini` given
`u_ini`.

`y_run_list` / `u_run_dict` define the unknowns and known inputs for the
time-domain stage. The trapezoidal integrator solves for `x` and `y_run` given
`u_run`.

The two lists may overlap partially or entirely (in this example, `lam`
appears in both). After initialization converges, the `ini2run()` method
transfers values between the two stages automatically: `f_x` was an unknown in
`y_ini` and also appears in `u_run`, so its converged value is copied into
`u_run`; `theta` was a known input in `u_ini` and also appears in `y_run`, so
its value is copied into `y_run` as a starting point.

This mechanism gives the user full control over the physical meaning of each
variable in each stage without duplicating the system equations. The `f` and
`g` equation vectors are the same; only the Jacobian structure changes because
the algebraic unknowns (`y_ini` vs `y_run`) may differ.

## Dense and Sparse Linear Algebra

The linear system `J * Dxy = fg` is the computational bottleneck at every
Newton iteration in the SymPy pipeline. pydae offers four backends, selectable
at build time.

### Dense mode (default)

When `sparse=False`, the Jacobian is stored as a flat `N_xy × N_xy` row-major
array. The solver in `daesolver.c` uses a built-in LU decomposition with
partial pivoting (`solve_dense`). This is the portable baseline that requires
no external libraries and works everywhere. It is suitable for systems up to a
few hundred unknowns.

In `ctypes_model.py`, the Jacobian buffer is allocated as
`np.zeros(N_xy * N_xy)` and passed directly to the C functions.

### Sparse mode with SuiteSparse KLU

When `sparse='klu'` (or `sparse=True`), the build phase computes a CSC
(Compressed Sparse Column) sparsity pattern with 0-based indexing. The
generated C file emits `Ap[]`, `Ai[]` arrays and a `NNZ` constant. At compile
time, the `USE_SPARSE` preprocessor define activates the KLU code paths in
`daesolver.c`.

### Sparse mode with Intel MKL PARDISO

When `sparse='pardiso'`, the build phase produces the same 0-based CSC pattern
in the JSON metadata. The C code generator converts `Ap` and `Ai` to 1-based
indexing at compile time, since PARDISO uses Fortran-style indexing.

### Sparse mode with Apple Accelerate

When `sparse='accelerate'` (macOS only), the build uses 0-based CSC with
`long` column pointers (as required by the Accelerate framework).

### How ctypes_model.py handles sparse backends

The `Model` class reads the `sparse_backend` field from the JSON metadata to
determine which backend was used at build time. This affects:

1. **Buffer sizing**: `jac_size_ini` and `jac_size_trap` are set to `NNZ` for
   sparse backends or `N_xy²` for dense.
2. **Jacobian reconstruction**: The `_sparse_to_dense_jac()` method converts a
   flat NNZ-length array back to a dense matrix using `scipy.sparse.csc_matrix`
   (for KLU and Accelerate) or `scipy.sparse.csr_matrix` with 1-based offset
   correction (for PARDISO).

## Cross-Platform Portability

pydae is designed to work on Windows, Linux, and macOS without requiring the
user to change their build scripts.

### Compiler selection

The ctypes builder detects the operating system via `platform.system()` and
selects the appropriate compiler: `clang` on macOS, `gcc` on Windows and
Linux. The shared library extension is `.dll`, `.so`, or `.dylib` accordingly.

### Shared library loading

`Model.__init__` first tries to load a CFFI extension module via
`importlib.import_module`. If that fails, it falls back to `ctypes.CDLL` with
the platform-appropriate extension. On Windows with Conda environments, it
calls `os.add_dll_directory` to add the Conda `Library/bin` path.

### CFFI vs ctypes trade-offs

CFFI produces a Python extension module that links against the solver at
compile time and avoids runtime symbol resolution. This is generally more
reliable on Windows where DLL loading can be fragile. ctypes compiles a
standalone shared library and loads it at runtime, which is simpler and does
not require CFFI to be installed. Both targets produce functionally identical
results; the choice is a matter of deployment convenience.

## Linearization: jac_run_eval and A_eval

Beyond simulation, `Model` provides methods for small-signal analysis.
`jac_run_eval()` calls the compiled `jac_trap_eval` C function directly at the
current operating point to obtain the trapezoidal Jacobian, then recovers the
continuous-time Jacobian blocks:

```
F_x = (I - jac_trap[:Nx, :Nx]) / (alpha * Dt)
F_y = -jac_trap[:Nx, Nx:] / (alpha * Dt)
G_x =  jac_trap[Nx:, :Nx]
G_y =  jac_trap[Nx:, Nx:]
```

`A_eval()` then computes the reduced state matrix by eliminating the algebraic
variables through the Schur complement: `A = F_x - F_y @ inv(G_y) @ G_x`. The
eigenvalues of `A` give the system modes (natural frequencies and damping
ratios) for small-signal stability analysis.

`CasadiModel` provides the same `A_eval()`, `BCD_eval()`, and
`eval_eigenvalues()` methods, computing them directly from the CasADi SX graph
rather than from compiled C functions.
