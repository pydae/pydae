# Getting started

This page walks through simulating a driven, damped pendulum with `pydae-core`.
It is the canonical "hello world" for the library.

## 1. Install

```bash
pip install pydae
```

You will also need a working C compiler on the PATH (`gcc` or MSVC on Windows).

## 2. Describe the system symbolically

The pendulum has two position variables ($p_x, p_y$), two velocity variables
($v_x, v_y$), and an algebraic Lagrange multiplier $\lambda$ that enforces the
length constraint $p_x^2 + p_y^2 = L^2$.

```python
import numpy as np
import sympy as sym

# Parameters
L, G, M, K_d = sym.symbols("L,G,M,K_d", real=True)

# States and algebraic unknowns
p_x, p_y, v_x, v_y = sym.symbols("p_x,p_y,v_x,v_y", real=True)
lam, f_x, theta = sym.symbols("lam,f_x,theta", real=True)

# Differential equations
dp_x = v_x
dp_y = v_y
dv_x = (-2*p_x*lam + f_x - K_d*v_x) / M
dv_y = (-M*G - 2*p_y*lam - K_d*v_y) / M

# Algebraic constraints
g_1 = p_x**2 + p_y**2 - L**2 - lam*1e-6
g_2 = -theta + sym.atan2(p_x, -p_y)

sys_dict = {
    "name": "pendulum",
    "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3},
    "f_list": [dp_x, dp_y, dv_x, dv_y],
    "g_list": [g_1, g_2],
    "x_list": [p_x, p_y, v_x, v_y],
    "y_ini_list": [lam, f_x],
    "y_run_list": [lam, theta],
    "u_ini_dict": {"theta": np.deg2rad(5.0)},
    "u_run_dict": {"f_x": 0},
    "h_dict": {
        "E_p": M*G*(p_y + L),
        "E_k": 0.5*M*(v_x**2 + v_y**2),
    },
}
```

## 3. Build

This step generates C code, compiles it, and writes a small Python shim next to
your current working directory.

```python
from pydae.core import Builder

bld = Builder(sys_dict, target="ctypes")
bld.build()
```

## 4. Initialise and simulate

```python
from pydae.core import Model

model = Model("pendulum")

# Solve the steady-state / initial-value problem
model.ini(
    {"theta": np.deg2rad(10)},
    xy_0={"p_x": 0.9, "p_y": -5.1, "lam": 0, "f_x": 1},
)

# Run for 1 s at the ini input, then 20 s with a different force
model.run(1.0, {})
model.run(20.0, {"f_x": 0.0})
model.post()
```

## 5. Inspect results

`model.Time`, `model.X`, `model.Y`, and `model.Z` hold the recorded series:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(model.Time, model.get_values("p_x"), label="p_x")
ax[0].plot(model.Time, model.get_values("p_y"), label="p_y")
ax[0].legend()
ax[1].plot(model.Time, model.get_values("E_p") + model.get_values("E_k"),
           label="total energy")
ax[1].legend()
plt.show()
```

## Next steps

- Read [Overview](overview.md) for the conceptual model of builder and
  runtime.
- Browse [API](api.md) for the full class reference.
- For power-system users, skip hand-writing `sys_dict` and use
  [pydae-bps](https://pydae-bps.readthedocs.io/) or [pydae-uds](https://pydae-uds.readthedocs.io/) instead.
