# pydae-core: Architecture and Internal Design

## What is pydae-core

pydae-core is the foundational package of the pydae monorepo. It provides a complete pipeline for turning a symbolic Differential-Algebraic Equation (DAE) system into a compiled C shared library and then driving that library from Python to perform steady-state initialization and time-domain simulation. The pipeline has two stages: a build stage (handled by the `Builder` class) and a runtime stage (handled by the `Model` class).

A DAE system in pydae has the semi-explicit index-1 form:

```
dx/dt = f(x, y, u, p)        (differential equations)
    0 = g(x, y, u, p)        (algebraic constraints)
    z = h(x, y, u, p)        (output equations)
```

where `x` is the vector of dynamic states, `y` is the vector of algebraic variables, `u` is the vector of inputs, `p` is the vector of parameters, and `z` is the vector of outputs.


## The Build Pipeline: from SymPy to a Shared Library

The user defines the system as a plain Python dictionary containing SymPy expressions. For example:

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

**Phase 1 — Parsing (`parser.py`).** The `check_system` function validates the dictionary: it ensures that both `f_list` and `g_list` are present (adding dummy equations when necessary), checks for duplicate variables, and determines whether a separate initialization/run split is needed. Then `process_system_dict` converts all variable name strings into the exact SymPy symbols that appear in the expressions, and wraps the equation lists into SymPy row matrices.

**Phase 2 — Symbolic Jacobians (`symbolic.py`).** The `compute_base_jacobians` function differentiates the equation vectors symbolically to produce the six sub-Jacobian blocks: `Fx`, `Fy_ini`, `Fy_run`, `Gx`, `Gy_ini`, and `Gy_run`. Then `build_large_jacobians` assembles the two global Jacobians that the solver needs. The initialization Jacobian is:

```
jac_ini = [[F_x,  Fy_ini],
           [G_x,  Gy_ini]]
```

The trapezoidal (run) Jacobian encodes the implicit integration rule and includes the time step and blending parameter:

```
jac_trap = [[I - alpha*Dt*F_x,  -alpha*Dt*Fy_run],
            [G_x,                Gy_run          ]]
```

For sparse builds, this phase also computes the CSC sparsity pattern (column pointers `Ap` and row indices `Ai`) of each Jacobian independently.

**Phase 3 — C code generation (`cffi_builder.py` / `ctypes_builder.py`).** The `sym2c` function converts each SymPy expression into a C code string using `sympy.ccode`. Then `sym2xyup` replaces symbolic variable names with C array indexing (`x[0]`, `y[3]`, `u[1]`, `p[5]`, etc.). The `_get_c_funcs` helper wraps these translated strings into C function bodies. This produces seven C functions: `f_ini_eval`, `f_run_eval`, `g_ini_eval`, `g_run_eval`, `h_eval`, `jac_ini_eval`, and `jac_trap_eval`. Each function writes its results into a flat `double *data` array. For dense mode, the Jacobian functions index into a row-major `N_xy × N_xy` flat array. For sparse mode, they index into a packed `NNZ`-length array using the sparse position `sp_idx`.

When the number of expressions exceeds 200, the translation can be parallelised by setting the `PYDAE_PARALLEL=1` environment variable, which distributes `sym.ccode` calls across a `ProcessPoolExecutor`.

**Phase 4 — Compilation.** The generated model C code is compiled together with `daesolver.c` (the generic solver) into a shared library. This can happen through CFFI (`ffi.set_source` followed by `ffi.compile`, producing a Python extension module) or through ctypes (invoking `gcc` or `clang` directly to produce a `.dll`, `.so`, or `.dylib`). Sparse-backend-specific compiler flags (`-DUSE_SPARSE`, `-DUSE_PARDISO`, or `-DUSE_ACCELERATE`) and linker flags are injected at this stage.

The build also writes a JSON metadata file (`{name}_data.json`) containing variable names, dimensions, parameter defaults, the sparse backend identifier, NNZ counts, and the sparsity patterns (Ap, Ai arrays).


## The Generated C Files

For a model called `pendulum`, the build produces a C file (`temp_ctypes_pendulum.c` or compiled inline by CFFI) containing:

Seven functions, all with the same signature: `void func(double *data, double *x, double *y, double *u, double *p, double Dt)`. The `data` pointer is the output buffer. For the equation evaluators (`f_ini_eval`, `g_run_eval`, etc.), `data` receives the equation residuals. For the Jacobian evaluators (`jac_ini_eval`, `jac_trap_eval`), `data` receives either the flat dense matrix or the packed sparse values array.

When sparse mode is active, the file also contains two independent sets of sparsity-pattern arrays: `Ap_ini[]`, `Ai_ini[]`, `NNZ_ini` for the initialization Jacobian, and `Ap_trap[]`, `Ai_trap[]`, `NNZ_trap` for the trapezoidal (run) Jacobian. These two patterns are different because the ini and run Jacobians have different algebraic blocks (`Fy_ini`/`Gy_ini` vs `Fy_run`/`Gy_run`), which means different entries can be structurally zero in each phase. Returning to the pendulum example, the ini Jacobian has columns for `lam` and `f_x`, while the run Jacobian has columns for `lam` and `theta` — these produce different sparsity structures and generally different numbers of nonzeros. Each set of arrays is declared as `extern` in `daesolver.c`: the `ini()` function references `Ap_ini`/`Ai_ini`, and the `run()` function references `Ap_trap`/`Ai_trap`, so each solver phase uses its own correct sparsity pattern.


## The DAE Solver: daesolver.c

`daesolver.c` is the heart of the runtime. It is a generic, reusable C file that is compiled alongside every model. It provides two entry points: `ini()` and `run()`.

### The `ini()` function

`ini()` implements a Newton-Raphson iteration to find the steady-state operating point where `f(x, y, u, p) = 0` and `g(x, y, u, p) = 0` simultaneously. On each iteration it:

1. Evaluates the residual vector `fg = [-f; -g]` by calling `f_ini_eval` and `g_ini_eval`.
2. Evaluates and factorises the initialization Jacobian by calling `jac_ini_eval` and then the appropriate linear solver.
3. Solves `jac_ini * Dxy = fg` for the correction vector `Dxy`.
4. Updates `xy += Dxy` and checks convergence via the squared norm of `Dxy`.

### The `run()` function

`run()` advances the solution in time using the implicit trapezoidal rule (or backward Euler, depending on `alpha`). The discretized equations at each time step are:

```
x_{n+1} - x_n - Dt * [alpha * f(x_{n+1}, y_{n+1}) + (1-alpha) * f(x_n, y_n)] = 0
g(x_{n+1}, y_{n+1}) = 0
```

At each time step, a Newton-Raphson loop:

1. Evaluates `f_run_eval` and `g_run_eval` to form the residual.
2. Evaluates `jac_trap_eval` to form the trapezoidal Jacobian and factorises it.
3. Solves `jac_trap * Dxy = fg` and updates the solution.
4. After convergence, optionally stores `Time`, `X`, `Y`, `Z` at the current step (subject to a decimation counter).

Both `ini()` and `run()` support a Jacobian re-use strategy controlled by `intparams[0]`: value 0 means re-evaluate and refactor the Jacobian at every Newton iteration, value 1 means only on the first iteration of each step ("dishonest Newton").

### How model_class.py calls daesolver.c

The `Model` class in Python loads the compiled shared library at construction time (either via CFFI's `importlib.import_module` or via `ctypes.CDLL`). It allocates all working arrays as NumPy `float64` arrays and passes them to the C functions through pointer casting. The `_d()` method handles the backend abstraction: for CFFI, it uses `ffi.cast("double *", ffi.from_buffer(arr))`; for ctypes, it passes the NumPy array directly (with `ndpointer` argtypes already configured).

The `ini()` Python method prepares the initial guess vector `xy_0`, pre-allocates the time-history storage arrays (`Time`, `X`, `Y`, `Z`), and calls `self.solver_lib.ini(...)` with 18 arguments. After a successful solve, it calls `ini2run()` to transfer the converged state into the runtime vectors.

The `run()` Python method accepts a target time `t_end` and an input dictionary. It ensures enough storage capacity exists (doubling the buffers when needed), and calls `self.solver_lib.run(...)` with 27 arguments. Multiple `run()` calls can be chained to simulate piecewise-constant input changes, with each call continuing from the previous endpoint.


## Why y_ini, u_ini and y_run, u_run

Many DAE systems need different unknowns and inputs during initialization versus time-domain simulation. The pendulum example in the pydae test suite illustrates this clearly. The pendulum has four dynamic states (`p_x`, `p_y`, `v_x`, `v_y`) and two algebraic variables, but the role of those algebraic variables changes between stages:

```python
"y_ini_list": [lam, f_x],
"y_run_list": [lam, theta],
"u_ini_dict": {"theta": np.deg2rad(5.0)},
"u_run_dict": {"f_x": 0},
```

During initialization, the user specifies the angle `theta` as a known input (`u_ini`) and the solver finds the Lagrange multiplier `lam` and the horizontal force `f_x` as unknowns (`y_ini`). This answers the question: "what force is needed to hold the pendulum at this angle in steady state?"

During simulation, the roles swap. The angle `theta` becomes an unknown that the solver computes at every time step (`y_run`), while the horizontal force `f_x` becomes a known input (`u_run`) that the user can vary to apply disturbances. This answers the question: "given this applied force, how does the pendulum swing over time?"

pydae handles this through two separate sets of algebraic variables and inputs:

`y_ini_list` / `u_ini_dict` define the unknowns and known inputs for the initialization stage. Newton-Raphson solves for `x` and `y_ini` given `u_ini`.

`y_run_list` / `u_run_dict` define the unknowns and known inputs for the time-domain stage. The trapezoidal integrator solves for `x` and `y_run` given `u_run`.

The two lists may overlap partially or entirely (in this example, `lam` appears in both). After initialization converges, the `ini2run()` method transfers values between the two stages automatically: `f_x` was an unknown in `y_ini` and also appears in `u_run`, so its converged value is copied into `u_run`; `theta` was a known input in `u_ini` and also appears in `y_run`, so its value is copied into `y_run` as a starting point.

This mechanism gives the user full control over the physical meaning of each variable in each stage without duplicating the system equations. The `f` and `g` equation vectors are the same; only the Jacobian structure changes because the algebraic unknowns (`y_ini` vs `y_run`) may differ.


## Dense and Sparse Linear Algebra

The linear system `J * Dxy = fg` is the computational bottleneck at every Newton iteration. pydae offers four backends, selectable at build time.

### Dense mode (default)

When `sparse=False`, the Jacobian is stored as a flat `N_xy × N_xy` row-major array. The solver in `daesolver.c` uses a built-in LU decomposition with partial pivoting (`solve_dense`). This is the portable baseline that requires no external libraries and works everywhere. It is suitable for systems up to a few hundred unknowns.

In `model_class.py`, the Jacobian buffer is allocated as `np.zeros(N_xy * N_xy)` and passed directly to the C functions.

### Sparse mode with SuiteSparse KLU

When `sparse='klu'` (or `sparse=True`), the build phase computes a CSC (Compressed Sparse Column) sparsity pattern with 0-based indexing. The generated C file emits `Ap[]`, `Ai[]` arrays and a `NNZ` constant. At compile time, the `USE_SPARSE` preprocessor define activates the KLU code paths in `daesolver.c`.

At runtime, `ini()` calls `klu_analyze` to perform symbolic analysis of the sparsity pattern (once), then `klu_factor` to compute numerical factorization at each Newton iteration, and `klu_solve` to solve the linear system. Since KLU natively expects CSC format — which is exactly what the build phase produces — no transpose is needed.

In `model_class.py`, the Jacobian buffer is allocated as `np.zeros(NNZ)` — only the nonzero entries, not the full matrix.

### Sparse mode with Intel MKL PARDISO

When `sparse='pardiso'`, the build phase produces the same 0-based CSC pattern in the JSON metadata. The C code generator (`ctypes_builder.py` / `cffi_builder.py`) converts `Ap` and `Ai` to 1-based indexing at compile time, since PARDISO uses Fortran-style indexing. The `USE_PARDISO` define activates the PARDISO code path.

PARDISO operates in three phases: phase 11 (symbolic analysis), phase 22 (numerical factorization), and phase 33 (forward/backward substitution). The solver calls all three phases explicitly. A critical detail is `iparm[11] = 2`, which tells PARDISO to solve `A^T x = b` instead of `A x = b`. This is necessary because the sparsity pattern is emitted in CSC form (which is equivalent to CSR of the transpose), so PARDISO must be told to use the transpose to get the correct solution.

### Sparse mode with Apple Accelerate

When `sparse='accelerate'` (macOS only), the build uses 0-based CSC with `long` column pointers (as required by the Accelerate framework). The `USE_ACCELERATE` define activates this path, which uses `SparseFactor(SparseFactorizationQR, ...)` for the initial factorization and `SparseRefactor` for subsequent re-factorizations at the same pattern.

### How model_class.py handles sparse backends

The `Model` class reads the `sparse_backend` field from the JSON metadata to determine which backend was used at build time. This affects:

1. **Buffer sizing**: `jac_size_ini` and `jac_size_trap` are set to `NNZ` for sparse backends or `N_xy²` for dense.
2. **Jacobian reconstruction**: The `_sparse_to_dense_jac()` method converts a flat NNZ-length array back to a dense matrix using `scipy.sparse.csc_matrix` (for KLU and Accelerate) or `scipy.sparse.csr_matrix` with 1-based offset correction (for PARDISO). This is used by `jac_run_eval()` to extract the F_x, F_y, G_x, G_y blocks for eigenvalue analysis.


## Cross-Platform Portability

pydae is designed to work on Windows, Linux, and macOS without requiring the user to change their build scripts.

### Compiler selection

The ctypes builder detects the operating system via `platform.system()` and selects the appropriate compiler: `clang` on macOS (required for the Accelerate framework), `gcc` on Windows and Linux. The shared library extension is `.dll`, `.so`, or `.dylib` accordingly. Platform-specific flags are applied: `-shared` on Windows and Linux, `-dynamiclib -fPIC` on macOS.

### Shared library loading

`Model.__init__` first tries to load a CFFI extension module via `importlib.import_module`. If that fails, it falls back to `ctypes.CDLL` with the platform-appropriate extension. On Windows with Conda environments, it calls `os.add_dll_directory` to add the Conda `Library/bin` path, ensuring that compiler-related DLLs (such as those from MinGW or MKL) can be found.

### Sparse library discovery

The build system searches for KLU and MKL headers/libraries through a priority chain: (1) the `CONDA_PREFIX` environment variable, using Conda's standard layout (`Library/include/suitesparse` on Windows, `include/suitesparse` elsewhere); (2) system-wide paths (`/opt/homebrew` on macOS, `/usr/include/suitesparse` on Linux); (3) for PARDISO, the `MKLROOT` environment variable and recursive file search under the Python data path. This allows pydae to find the sparse libraries whether they were installed via Conda, Homebrew, apt, or Intel oneAPI.

### MKL library naming

On Windows, MKL ships import libraries with a `_dll` suffix (`mkl_intel_lp64_dll.lib`), while Linux and macOS use the standard names (`libmkl_intel_lp64.so` / `.dylib`). The builder handles this transparently by appending the correct library names based on the platform.

### CFFI vs ctypes trade-offs

CFFI produces a Python extension module that links against the solver at compile time and avoids runtime symbol resolution. This is generally more reliable on Windows where DLL loading can be fragile. ctypes compiles a standalone shared library and loads it at runtime, which is simpler and does not require CFFI to be installed. Both targets produce functionally identical results; the choice is a matter of deployment convenience.


## Linearization: jac_run_eval and A_eval

Beyond simulation, `Model` provides methods for small-signal analysis. `jac_run_eval()` calls the compiled `jac_trap_eval` C function directly at the current operating point to obtain the trapezoidal Jacobian, then recovers the continuous-time Jacobian blocks:

```
F_x = (I - jac_trap[:Nx, :Nx]) / (alpha * Dt)
F_y = -jac_trap[:Nx, Nx:] / (alpha * Dt)
G_x =  jac_trap[Nx:, :Nx]
G_y =  jac_trap[Nx:, Nx:]
```

`A_eval()` then computes the reduced state matrix by eliminating the algebraic variables through the Schur complement: `A = F_x - F_y @ inv(G_y) @ G_x`. The eigenvalues of `A` give the system modes (natural frequencies and damping ratios) for small-signal stability analysis.
