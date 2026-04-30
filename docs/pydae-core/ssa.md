# Small-Signal Analysis (SSA)

After a model has been initialised and its Jacobians evaluated, pydae can
linearise the DAE around the current operating point to produce a standard
state-space representation. This enables eigenvalue analysis, participation
factors, damping ratios, and controller design.

## Linearised DAE

The nonlinear DAE system

$$
\begin{aligned}
\dot{x} &= f(x, y, u, p) \\
0 &= g(x, y, u, p) \\
z &= h(x, y, u, p)
\end{aligned}
$$

is linearised to incremental form around a steady state
$(x_0, y_0, u_0)$:

$$
\begin{aligned}
\Delta\dot{x} &= F_x \Delta x + F_y \Delta y + F_u \Delta u \\
0 &= G_x \Delta x + G_y \Delta y + G_u \Delta u \\
\Delta z &= H_x \Delta x + H_y \Delta y + H_u \Delta u
\end{aligned}
$$

where each block is a Jacobian evaluated at the operating point, e.g.
$F_x = \left.\frac{\partial f}{\partial x}\right|_{0}$.

## Prerequisites

```python
model.ini(params_dict, xy_0_dict)   # Newton-Raphson steady state
model.run(0.1, {})                  # single step to populate jac_trap
```

The state-space functions require:

1. A successful `ini()` call (algebraic variables converged).
2. At least one `run()` call (to populate the trapezoidal Jacobian).
3. The model built with `uz_jacs=True` (the default) for B, C, D computation.

## The A matrix

Eliminating $\Delta y$ from the linearised equations via the Schur complement
gives the reduced continuous-time state matrix:

$$
A = F_x - F_y \, G_y^{-1} \, G_x
$$

The `Model` class recovers $F_x$, $F_y$, $G_x$, $G_y$ from the compiled
trapezoidal Jacobian and computes $A$:

```python
A = model.A_eval()  # returns (N_x, N_x) numpy array
```

`A_eval()` calls `jac_run_eval()` lazily, so you only need to call
`A_eval()` directly.

### Eigenvalue report

The `pydae.ssa` module provides a damping report:

```python
from pydae.ssa.ssa import damp

damp(model.A)
# → prints Mode, Real, Imag, Freq, Damp for each eigenvalue
```

For a sorted table with optional participation factors:

```python
ssstudy = damp(model.A, model=model, sort='damp')
```

### Participation factors

```python
from pydae.ssa.ssa import participation

PF = participation(model, method='kundur')  # or 'milano'
print(PF)
```

## The B, C, and D matrices

When input-output matrices are needed, `BCD_eval()` evaluates the remaining
Jacobian blocks from the compiled shared library and completes the state-space
description:

$$
\begin{aligned}
B &= F_u - F_y \, G_y^{-1} \, G_u \\
C &= H_x - H_y \, G_y^{-1} \, G_x \\
D &= H_u - H_y \, G_y^{-1} \, G_u
\end{aligned}
$$

```python
B, C, D = model.BCD_eval()
```

The shapes are:

| Matrix | Shape | Description |
|--------|-------|-------------|
| $B$ | $(N_x, N_u)$ | State-to-input map |
| $C$ | $(N_z, N_x)$ | State-to-output map |
| $D$ | $(N_z, N_u)$ | Direct feed-through |

### Dense vs. sparse backends

The UZ Jacobians ($F_u, G_u, H_x, H_y, H_u$) are **always generated as dense
C functions**, regardless of whether the main solver uses KLU, PARDISO, or
dense LAPACK. The B, C, D algebra is performed in NumPy and is identical
across all backends.

### Complete example

```python
from pydae.core import Builder, Model
from pydae.ssa.ssa import damp

bld = Builder(sys_dict, target='ctypes', sparse=False)
bld.build()

model = Model(sys_dict['name'])
model.ini(params_dict, xy_0_dict)
model.run(0.1, {})

# Full state-space
A = model.A_eval()
B, C, D = model.BCD_eval()

# Eigenvalue analysis
damp(A, model=model, sort='damp')
```

## Re-evaluation

Every call to `A_eval()` or `BCD_eval()` triggers a fresh evaluation of the
underlying Jacobian at the **current** $(x, y, u)$ values. If you change
inputs or run additional simulation steps, re-evaluate to capture the new
operating point:

```python
model.run(5.0, {'p_c': 1.0})   # step to a new condition
A_new = model.A_eval()          # re-linearised at the new state
```
