# MLX backend and batched sweeps (Apple Silicon)

`pydae.core.builder.mlx_builder` is a third solver backend alongside SymPy→C
and CasADi. It lowers a `sys_dict` onto [MLX](https://ml-explore.github.io/mlx)
graphs, takes Jacobians by automatic differentiation, and compiles the
Newton-Raphson iteration into Metal GPU kernels — no C toolchain involved.

Its reason to exist is **`MlxSweep`**: solving thousands of copies of the same
DAE at once, one per parameter set. A capability curve of 10,000 operating
points takes about a second instead of several minutes.

:::{note}
This backend requires an Apple Silicon Mac and the optional `mlx` extra:

```bash
pip install pydae[mlx]
```

Both classes are imported lazily, so `import pydae` keeps working everywhere
else.
:::

---

## When to use it

```{list-table}
:header-rows: 1
:widths: 30 70

* - Situation
  - Backend
* - One model, one operating point, fastest possible
  - **SymPy→C** (`Builder` + `Model`)
* - One model, no C compiler available
  - **CasADi** (`CasadiBuilder` + `CasadiModel`)
* - **Many parameter sets of the same model**
  - **MLX** (`MlxBuilder.sweep`)
```

:::{warning}
As a *scalar* solver MLX is not competitive with a C-compiled or CasADi model,
whatever the Jacobian costs — a single small system cannot keep a GPU busy, and
the per-op dispatch overhead is paid either way. Reach for this backend when you
have a **batch**.
:::

---

## The DAE, unchanged

The MLX backend solves exactly the same semi-explicit index-1 system as the
other backends,

$$
\begin{aligned}
\dot{x} &= f(x, y, u, p) \\
0 &= g(x, y, u, p) \\
z &= h(x, y, u, p)
\end{aligned}
$$

in the same two phases:

- **Backward initialization** — with derivatives set to zero, solve
  $0 = f(x, y_{ini}, u_{ini}, p)$ and $0 = g(x, y_{ini}, u_{ini}, p)$
  for the unknowns $v = [x;\, y_{ini}]$.
- **Forward simulation** — advance with the implicit trapezoidal rule using
  the run-phase split $(y_{run}, u_{run})$.

It accepts a `sys_dict` built with **either** SymPy or CasADi expressions, so
`BpsBuilder`, `UdsBuilder` and hand-written systems all work as they are.

---

## Quick start: a single model

```python
from pydae.bps import BpsBuilder

model = BpsBuilder("genrou.hjson").build_mlx("genrou", dtype="float32", Dt=0.01)
model.ini({})                        # seeded from sys_dict['xy_0_dict']
model.run(5.0, {"p_m_1": 0.65})
model.post()

model.get_value("q_g_1")        # scalar, current value
model.get_values("omega_1")     # time series after post()
model.A_eval()                  # state matrix, for SSA
```

`build_mlx(name, **kwargs)` is a convenience on `BpsBuilder` and `UdsBuilder`:
it runs `construct(name)` and hands the resulting `sys_dict` to `MlxBuilder`,
forwarding any keyword arguments. Nothing is compiled to C. The long form is
still there when you want the `sys_dict` in between:

```python
from pydae.core.builder import MlxBuilder

grid = BpsBuilder("genrou.hjson")
grid.construct("genrou")
model = MlxBuilder(grid.sys_dict, dtype="float32").build()
```

Either way the model accepts **both** symbolic front ends — this builder's
default SymPy expressions, or CasADi `SX` from `BpsBuilder(..., use_casadi=True)`
— because `MlxBuilder` detects which it was handed and lowers accordingly.

The accessor surface matches `Model` and `CasadiModel` — `get_value`,
`get_values`, `get_mvalue`, and the `x_list` / `y_run_list` / `z_list`
attributes — so reporters and plotting helpers work unchanged.

---

## Batched sweeps

`model.sweep(...)` returns an `MlxSweep`: **B copies of the model**, differing
only in the values you sweep.

```python
import numpy as np

model.ini({})                                        # nominal anchor and seed

sweep = model.sweep({"p_m_1": np.linspace(0.05, 0.95, 10_000)})
sweep.ini()

q = sweep.get_values("q_g_1")        # (10000,) reactive power per case
```

Values are `name -> scalar | sequence`. Sequences define the batch width B,
scalars broadcast to every case:

```python
model.sweep({"p_m_1": np.linspace(0.1, 0.9, 500),   # swept
             "v_f_1": 1.5})                          # held for all 500 cases
```

Several names can be swept at once, and there are two different things that can
mean. `sweep()` pairs them — equal-length arrays are walked together, giving
one case per index:

```python
model.sweep({"p_m_1": np.linspace(0.2, 0.7, 100),
             "v_f_1": np.linspace(1.3, 1.9, 100)})   # B = 100 paired points
```

:::{warning}
Two arrays of length 100 give **100 cases, not 10,000**. `sweep()` never takes
a product; that is what `grid` is for. Arrays of differing length raise rather
than broadcast, so the mistake surfaces immediately — but two axes that happen
to be the same length will quietly pair up.
:::

For a full factorial study, `MlxSweep.grid` takes the cartesian product
instead, over any number of axes:

```python
from pydae.core.builder import MlxSweep

sweep = MlxSweep.grid(model,
                      p_m_1=np.linspace(0.2, 0.8, 100),
                      v_f_1=[1.3, 1.5, 1.7])          # B = 300

sweep = MlxSweep.grid(model,                          # three axes, B = 600
                      p_m_1=np.linspace(0.2, 0.8, 100),
                      v_f_1=[1.3, 1.5, 1.7],
                      H_1=[3.0, 6.0])
```

`sweep.cases` holds the per-case value of every swept name as a `(B,)` array,
so a factorial result reshapes back to its axes with
`sweep.get_values("q_g_1").reshape(100, 3)`.

:::{admonition} Solve the nominal point first
:class: tip

`sweep.ini()` seeds every case from the model's converged operating point, so
call `model.ini()` before creating the sweep. Pass `seed=` a dict or a JSON
`xy_0` path to override, or `seed='previous'` to continue from the sweep's own
last solution.
:::

### Reading results

```python
sweep.get_values("q_g_1")        # (B,) — states, algebraics, inputs, params, h_dict
sweep.get_mvalue(["p_g_1", "q_g_1"])
sweep.to_frame(["q_g_1", "V_1"]) # DataFrame: one row per case
sweep.cases["p_m_1"]             # the swept values themselves
```

`to_frame` carries the swept columns plus `converged` and `status`:

```
   p_m_1     q_g_1      V_1  converged    status
0.050000  0.187426 1.000740       True converged
0.499865  0.005845 1.000395       True converged
0.799775       NaN      NaN      False  max_iter
```

### Cases that have no solution

A wide sweep will contain operating points that simply do not exist — and
finding that boundary is often the point of the study. Non-convergence is
therefore reported as **data**, never raised:

```python
sweep.converged                  # (B,) bool mask
sweep.status                     # 0 converged, 1 max_iter, 2 non_finite, 3 singular
sweep.feasible_range("p_m_1")    # (0.05, 0.702) — the span that solved
sweep.failures()                 # DataFrame of the cases that did not
```

Non-converged cases read back as `NaN`, so a plot breaks cleanly at the limit
instead of drawing nonsense. Pass `require_all=True` to `ini()` if you would
rather have a `SweepError` (it carries the failure table).

In the example below, sweeping mechanical power at *fixed excitation* stops
converging at `p_m = 0.702 pu`, which is the machine's analytic pull-out limit
$E V / X_d = 1.5/2.135 = 0.7026$ — the solver found the steady-state stability
boundary on its own.

### Batched transients

The same batched state can be integrated in lockstep. Declare which variables
to record: storing the full state of a large sweep would run to gigabytes.

```python
sweep.run(1.0, record=["omega_1", "V_1"])   # settle
sweep.run(6.0, {"p_m_1": 0.65})             # step the input, all cases at once

sweep.get_series("omega_1")                 # (n_samples, B)
sweep.Time                                  # (n_samples,)
```

Notes:

- `update_dict` values may be scalars or `(B,)` arrays, so each case can take
  its own input step.
- `decimation=N` stores every Nth sample.
- All cases share one time grid, so **event times are common to the batch**.
  Group cases by event time into separate sweeps if they must differ.
- A case that fails mid-run is frozen and its remaining samples are `NaN`.

## Sparse small-signal analysis

For a *single* operating point there is a second route that never forms the
reduced state matrix at all. `model.small_signal()` solves the **generalized**
problem against the augmented Jacobian:

$$
\xi = \begin{bmatrix} x \\ y \end{bmatrix}, \quad
\lambda(\xi) = \begin{bmatrix} f(\xi) \\ g(\xi) \end{bmatrix}, \quad
\Lambda_\xi = \begin{bmatrix} F_x & F_y \\ G_x & G_y \end{bmatrix}, \quad
\Lambda_\xi \phi = \lambda E \phi, \quad
E = \begin{bmatrix} I & 0 \\ 0 & 0 \end{bmatrix}
$$

The pencil $(\Lambda_\xi, E)$ has exactly `N_x` finite eigenvalues — identical
to those of $A = F_x - F_y G_y^{-1} G_x$, verified to 1e-13 — plus `N_y`
infinite ones from the algebraic block, which shift-invert maps to zero and
filters out automatically.

```python
model.ini({}, xy_0=...)
result = model.small_signal(Shift=0.0, N_modes=6)

result.eigenvalues          # (N_modes,) nearest the shift first
result.right_eigenvectors   # (N_x + N_y, N_modes) -- dynamic block on top
result.left_eigenvectors    # normalised so psi^T E phi = 1
result.participation        # (N_x, N_modes) Kundur-normalised
result.damp_frame()         # Real / Imag / Freq. / Damp / Residual / Converged
result.participation_frame(mode=1)
```

`Shift` is the target: modes closest to it converge first. Use `0.0` for the
slow electromechanical modes, a large negative value for fast converter modes.

### Why it is built this way

* **Never $G_y^{-1}$.** Inverting the algebraic Jacobian destroys the network's
  sparsity, which is the one structural advantage a power-system model has.
* **Matrix free by default.** $\Lambda_\xi v$ is a forward-mode `mx.jvp` pass
  over $\lambda(\xi)$ — a Jacobian-*vector* product, so no Jacobian is built.
  $\Lambda_\xi^\mathsf{T} w$ is a reverse-mode `mx.vjp` pass, and that is what
  produces the left eigenvectors. Both are wrapped in `mx.compile`.
* **Shift-invert Arnoldi** on $M = (\Lambda_\xi - \sigma E)^{-1} E$, with the
  inner solve done by GMRES built purely from those products. Ritz values map
  back as $\lambda = \sigma + 1/\mu$.

:::{warning}
Every candidate mode carries a residual
$\|\Lambda_\xi\phi - \lambda E\phi\| / \|\Lambda_\xi\phi\|$, and modes are
selected on it rather than on $|\mu|$. This is not decoration: in float32 the
Krylov basis loses orthogonality (measured $\|V^\mathsf{T}V - I\| = 0.9$) and
throws off **ghost Ritz pairs** with entirely plausible $|\mu|$. Ranking by
$|\mu|$ alone let one displace three genuine eigenvalues. Always check
`result.mode_converged`; a mode that fails `Mode_tolerance` is still returned,
but it is not an eigenvalue, and a `RuntimeWarning` says so.
:::

### `matrix_free`, and when to turn it off

| | `matrix_free=True` (default) | `matrix_free=False` |
|---|---|---|
| Shift-invert solve | GMRES from `jvp` products, on GPU | one AD pass forms $\Lambda_\xi$, then LU on CPU |
| Forms an `n x n` matrix | never | $\Lambda_\xi$ only (never $A$, never $G_y^{-1}$) |
| genrou (16 unknowns) | 0.83 s | **0.02 s** |
| Kundur two-area (87) | 74 s | **22 s** |

Both agree with the dense reference to 5e-15 on the Kundur system. The
factorised path is far faster at these sizes — the matrix-free path pays a full
GMRES solve per Arnoldi step, and on vectors this short the GPU is bound by
dispatch, not arithmetic. Use `matrix_free=True` when memory is the binding
constraint rather than time.

:::{admonition} `Restart` defaults to the full system order — leave it there
:class: caution

Restarted GMRES stagnates badly on an augmented power-system Jacobian. On the
Kundur two-area system (n = 87), `Restart=60` flatlines at a relative residual
of 4e-1 and the eigenvalues come out wrong by 1.55; full GMRES reaches 5e-12 in
two cycles. Lowering `Restart` trades Krylov memory for convergence and needs a
preconditioner to be safe.
:::

### Batched small-signal analysis

Every case is linearised in one batched pass, which makes a damping locus
against a swept parameter cheap:

```python
A = sweep.A_eval()          # (B, N_x, N_x)   A = Fx - Fy Gy⁻¹ Gx
eigs = sweep.eigenvalues()  # (B, N_x) complex
sweep.damp(case=0)          # delegates to pydae.ssa.ssa.damp for one case
```

The AD blocks are produced batched on the GPU; the reduction and the
eigen-decomposition run in float64 on the CPU, which is both more accurate and
faster than MLX's CPU-only `solve` would be for a multi-column right-hand side.

---

## Worked example

`examples/mlx_sweep.py` runs three studies on the `genrou` fixture and saves
SVG plots:

```bash
python examples/mlx_sweep.py
```

1. **Capability curve** — 10,000 initializations sweeping `p_m_1` at fixed
   excitation, in ~1.3 s, with the infeasible region shaded.
2. **Inertia study** — 200 transients in lockstep over `H_1`, showing lighter
   machines swinging further after a power step.
3. **Eigenvalue locus** — the slowest oscillatory mode tracked across 60
   operating points.

Its output begins:

```text
--- capability curve: 10,000 initializations ---
ini: 7245/10000 cases converged (max residual 9.2e-05); 2755 did not: 2755 max_iter
  feasible p_m_1: 0.05 .. 0.702
solved in 1.38 s (0.138 ms per case)
analytic limit E*V/X_d = 0.7026 pu
```

---

## Performance

Measured on `genrou` (16 unknowns, float32, M-series GPU) against the same
model's own scalar `ini` at 11 ms per case:

```{list-table}
:header-rows: 1
:widths: 20 20 25 20

* - Cases (B)
  - Wall time
  - Per case
  - vs scalar
* - 100
  - 0.04 s
  - 0.39 ms
  - 29×
* - 1 000
  - 0.06 s
  - 0.055 ms
  - 200×
* - 10 000
  - 0.29 s
  - 0.029 ms
  - 381×
* - 50 000
  - 0.84 s
  - 0.017 ms
  - 704×
```

Batched transients reach 0.013 ms per step per case at B = 10,000, against
roughly 10 ms per step for a single case.

The speedup grows with B because what is being amortised is the fixed per-op
dispatch cost. Small sweeps are barely worth batching; large ones are worth a
great deal.

---

## Memory

**A scalar simulation costs nothing worth measuring.** `k12p6` (87 unknowns,
float32) allocates **0.42 MB** of MLX memory for `ini` and for `run`, plus
0.53 MB of trajectory buffers over 10 s at `Dt=0.01`. Process RSS lands near
287 MB, but ~214 MB of that is the Python interpreter with numpy, sympy and
casadi imported — the model itself is a rounding error.

**Batched sweeps are where memory lives, and `chunk` is the dial.** Peak scales
with `chunk`, never with `B`: 10,000 cases and 50,000 cases peak at the same
place, because the sweep chunks. Per case in flight, measured with
`mx.get_peak_memory()`:

| model | unknowns | per case | ≈ `8n²·itemsize` |
|---|---|---|---|
| `genrou` (float32) | 16 | 10.6 kB | 10.5 kB |
| `k12p6` (float32) | 87 | 245.6 kB | 250 kB |

That is roughly eight `n`×`n` temporaries — the expanded Jacobian, its
`(B, n, n)` transpose, the masked-and-ridged copy, and the factorisation inside
`mx.linalg.solve`. A batched transient peaks about 5 % above `ini`, and recorded
series are negligible (3 MB for two variables over 2 s at B = 2000).

So `k12p6` at the default `chunk` of 4096 peaks near **1.0 GB**, whatever B is.

### Why the default chunk is not larger

Throughput saturates well before memory does, so a bigger chunk is memory spent
for nothing. Measured on `ini()`:

| chunk | `genrou`, B=32768 | `k12p6`, B=16384 |
|---|---|---|
| 1024 | 0.48 s / 11 MB | 1.04 s / 250 MB |
| 2048 | 0.38 s / 22 MB | 0.98 s / 498 MB |
| **4096** | **0.33 s / 45 MB** | **0.96 s / 996 MB** |
| 8192 | 0.34 s / 89 MB | 0.97 s / 1988 MB |
| 16384 | — | 1.05 s / 3943 MB |

Flat in time past ~4096, linear in memory, and *slower* beyond — hence the cap
there. `memory_budget_gb` (default 2.0) is a ceiling on top of that, so it
rarely binds on a small model; lower it when memory is tight rather than raise
it to go faster:

```python
model.sweep(values, memory_budget_gb=0.5)   # n=87 -> chunk 2048, ~500 MB
model.sweep(values, chunk=1024)             # or set it outright
```

Neither changes any result — only how many cases are resident at once.

### Compression cut memory too

The structural sparsity above is a memory win as much as a speed one, because
the dense reverse-mode Jacobian materialised a full tape at width `chunk`. Peak
MLX bytes for `k12p6` at a fixed chunk:

| chunk | `sparse=True` | `sparse=False` | reduction |
|---|---|---|---|
| 256 | 46 MB | 1.51 GB | 33× |
| 1024 | 247 MB | 4.35 GB | 18× |
| 4096 | 986 MB | 15.4 GB | 16× |

:::{warning}
`mx.get_peak_memory()` under-reports the CPU stream, so it cannot be used to
check float64: `k12p6` in float64 measures 253 kB/case against float32's
246 kB, where it should be about double. Measure process RSS instead.
:::

---

## Structural sparsity

Every Jacobian on this backend is built by **compressed AD**: instead of one
reverse-mode pass per equation, the columns are grouped by Curtis-Powell-Reid
colouring and each group is probed with a single forward-mode pass. Two columns
can share a pass whenever no equation contains both of their variables, which on
a power-system Jacobian is nearly all of them.

It is on by default (`sparse=True`) and needs no configuration:

```python
model = grid.build_mlx("k12p6")
model.patterns["ini"]
# <JacobianPattern 87x87 nnz=350 (4.6%) passes=8 (8 colours + 0 rows) vs 87 dense>
```

| model | unknowns | density | AD passes |
|---|---|---|---|
| `genrou` | 16 | 23.8 % | 7 |
| `k12p6` (Kundur two-area) | 87 | 4.6 % | 8 |
| 6 × `k12p6` | 507 | 0.8 % | 11 (10 colours + 1 row) |

### Dense rows

One dense equation would ruin the colouring: if a row contains every variable
then every pair of columns conflicts and the colouring collapses to one colour
per column. Power systems have exactly such a row — the centre-of-inertia speed
`omega_coi = sum(H_i * omega_i) / sum(H_i)` touches every machine, and gets
denser as the network grows.

So the densest few rows are pulled out of the colouring and recovered the other
way round, one reverse-mode pass each, which yields a whole row however dense it
is. On the 24-machine system above that single row is the difference between 11
passes and 29.

### What is *not* sparse, and why

**The linear solve stays dense.** This is deliberate and it was measured, not
assumed: MLX arrays are immutable, so the in-place update at the heart of any
sparse factorization (`values.at[dst].add(...)`) reallocates and copies the
whole value array on every pivot. A batched sparse LU of the `k12p6` Jacobian
needs 170 such stages, which over a `(2246, 10000)` value array costs 331 ms —
against 603 ms for the dense `mx.linalg.solve` it would have replaced, in
exchange for a static pivot order that can fail on a badly scaled operating
point. Not a good trade. Sparsity pays in the AD passes, which is where the
time actually was: at B = 10,000 on `k12p6` the Jacobian was 2049 ms of a
2670 ms Newton iteration.

### Speedups, and where they come from

Compression removes AD passes, so it pays in proportion to what a pass costs.
At B = 1 that is very little — `mx.vmap` batches the identity rows into one
dispatch, so 87 passes cost barely more than 8 — while at large B each pass is
real work. Measured on the Jacobian alone (`k12p6`, float32):

| batch width | dense | compressed | speedup |
|---|---|---|---|
| 1 | 19.0 ms | 12.4 ms | 1.5× |
| 100 | 23.9 ms | 13.2 ms | 1.8× |
| 1 000 | 267.9 ms | 14.5 ms | 18× |
| 10 000 | 2329.6 ms | 22.0 ms | 106× |

End to end the dense solve now dominates, so the whole-iteration gains are
smaller — and larger on the larger model, which is the point:

| | `genrou` (n=16) | `k12p6` (n=87) |
|---|---|---|
| `ini` | 1.34× | 1.29× |
| trapezoidal step | 1.28× | 1.56× |
| `sweep.ini()`, B=2000 | 1.87× | 2.90× |
| `sweep.run()`, B=2000 | 1.46× | 2.48× |
| `sweep.A_eval()` | 1.04× | 3.44× |

Because the scalar payoff is model-dependent — compression is ahead on
trigonometric power-system models and about 15 % *behind* on a model built from
many trivially cheap equations — the builder times both on the seed point and
keeps the winner, logging the choice:

```
sparsity (ini)   : 350 nnz (4.6%), 8 AD passes vs 87 dense
sparsity (ini)   : scalar 19.01 ms dense vs 12.38 ms compressed -> compressed
```

Batched Jacobians are always compressed; there the AD passes carry real work
and no fixed overhead catches up.

### Correctness

The pattern is a structural *over*-approximation — an entry counts as nonzero
when the variable's symbol occurs in the equation, without asking whether it
cancels — so it can add entries but never miss one, and a compressed Jacobian is
exact wherever the dense one is. What could still go wrong is the recovery
(colouring, mask, dense-row split), so every build checks `J @ tangent` against
a plain `jvp` for a random tangent and refuses to build on a mismatch. Pass
`sparse=False` to force plain reverse-mode AD if you ever need to isolate it.

---

## Numerical precision

Two MLX properties bound accuracy, and they are worth knowing before choosing
a tolerance:

- **`float64` is CPU-only.** `dtype="float32"` (the default for sweeps) keeps
  everything on the GPU; `dtype="float64"` runs identical code on the CPU
  stream, still vectorised over the batch but without Metal.
- **`sin`, `cos` and `exp` are float32-accurate even on float64 arrays**
  (relative error ~2e-8; `sqrt`, `log`, `atan` and plain arithmetic are exact
  to double precision). Every power-system model is trigonometric, so its
  residual bottoms out near 1e-8 in *either* dtype.

Because of the second point, the Newton driver accepts a solve that
demonstrably stops improving (`stall_tol`) in addition to the usual residual
and step tests. A residual of 3e-8 on a power-system model is convergence, not
failure.

Validation against `CasadiModel` on the same models: trajectories and all
Jacobian blocks agree to ~1e-8 in float64 and ~1e-6 in float32.

---

## Gotchas

**A swept name is applied everywhere it appears.** `p_m_1` is registered in
both `u_ini_dict` and `u_run_dict`; a sweep writes both, because the value
defines the *case* rather than a phase-specific override. (This differs from
`model.set_value()`, which targets one phase.)

**Sweep values must be consistent in length.** Mixing a 500-point axis with a
3-point axis raises; use `MlxSweep.grid` for a factorial study instead.

**`BpsBuilder.construct()` writes `xy_0.json` into the working directory.**
Run examples and tests from a scratch directory if that matters.

**`small_signal` and `sweep.eigenvalues()` answer different questions.** The
sweep linearises *many* operating points and forms a small dense `A` for each,
which is the right trade when `N_x` is small and `B` is large.
`small_signal` linearises *one* point without forming anything, which is the
right trade when the model is large or you only want a few modes near a shift.

**Convergence checking costs a host synchronisation.** `run()` defaults to a
fixed 3 Newton iterations per step with no residual read-back, auditing
convergence once per output sample. Pass `newton_iters=None` for the adaptive
driver (residual-tested every iteration) if a model needs it.

---

## API summary

```{list-table}
:header-rows: 1
:widths: 35 65

* - Call
  - Purpose
* - `grid.build_mlx(name, **kwargs)`
  - `construct()` + `MlxBuilder(...).build()` in one call (bps and uds)
* - `MlxBuilder(sys_dict, dtype=..., sparse=True).build()`
  - Lower the equations, compile the Newton kernels
* - `model.patterns['ini' \| 'trap' \| 'aug' \| 'Fx' \| ...]`
  - Structural sparsity of a Jacobian block: nnz, colours, AD passes
* - `model.ini(params, xy_0=...)`
  - Backward initialization of one operating point
* - `model.run(t_end, inputs)` / `model.post()`
  - Trapezoidal time-domain simulation
* - `model.sweep(values)`
  - Create a batched `MlxSweep`
* - `MlxSweep.grid(model, **axes)`
  - Cartesian product of swept axes
* - `sweep.ini(stages=1, seed=..., require_all=False)`
  - Batched backward initialization
* - `sweep.run(t_end, inputs, record=[...])`
  - Batched trapezoidal simulation
* - `sweep.get_values(name)` / `get_series(name)`
  - `(B,)` current values / `(n_samples, B)` recorded series
* - `sweep.converged`, `status`, `feasible_range`, `failures()`
  - Per-case outcome
* - `sweep.to_frame([names])`
  - One DataFrame row per case
* - `sweep.A_eval()`, `sweep.eigenvalues()`
  - Batched small-signal analysis (dense A, one per case)
* - `model.small_signal(Shift=...)`
  - Sparse generalized SSA: no A, no `Gy^-1`, matrix-free shift-invert Arnoldi
* - `result.eigenvalues`, `right_eigenvectors`, `left_eigenvectors`
  - Modes of the pencil `(Lambda_xi, E)`
* - `result.participation`, `participation_frame()`
  - Kundur-normalised participation over the dynamic states
* - `result.mode_converged`, `mode_residual`
  - Per-mode residual check -- consult before trusting a mode
```
