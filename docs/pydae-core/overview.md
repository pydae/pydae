# Overview

## What pydae-core does

`pydae-core` takes a symbolic description of a DAE system and turns it into a
fast, standalone simulation object. The package supports **two independent
pipelines** — a compiled C pipeline and a CasADi-native pipeline — each with
its own Builder and Model, plus an optional [MLX backend](mlx.md) for running
many parameter sets in parallel on Apple Silicon:

```{mermaid}
flowchart TD
    A[sys_dict] --> B{Choose backend}
    B -->|SymPy| C[SymPyBuilder]
    B -->|CasADi| D[CasadiBuilder]
    B -->|MLX| K[MlxBuilder]
    C --> E[Generated C code]
    E --> F[Compiled .so / .dll]
    F --> G[CtypesModel / CasadiModel\nbinary load]
    D --> H[CasADi SX graph]
    H --> I[CasadiModel\nsymbolic or precompiled]
    K --> L[MlxSweep\nB cases at once]
    G --> J[ini / run / post]
    I --> J
    L --> J
```

The key idea is that **building is symbolic and slow, simulation is numerical
and fast**. You pay the compilation or graph-construction cost once and then
run arbitrarily many scenarios.

## Main objects

### `Builder` (SymPy → C)

Consumes a `sys_dict` of SymPy expressions and produces compiled C code plus a
Python wrapper.

```python
from pydae.core import Builder

bld = Builder(sys_dict, target="ctypes")
bld.build()
```

### `CasadiBuilder` (CasADi → SX graph)

Consumes a `sys_dict` of CasADi SX expressions and builds a CasADi rootfinder
and IDAS integrator.

```python
from pydae.core.builder import CasadiBuilder

bld = CasadiBuilder(sys_dict).build()
```

### `Model` (ctypes/CFFI runtime)

The runtime interface to a C-compiled system. Supports steady-state
initialisation (`ini`), time-domain simulation (`run`), and post-processing
(`post`).

```python
from pydae.core import Model

model = Model("my_system")
model.ini({...}, xy_0={...})
model.run(10.0, {"u_1": 1.0})
model.post()
```

### `CasadiModel` (CasADi runtime)

The runtime interface for the CasADi pipeline. Same API as `Model`.

```python
from pydae.core.model import CasadiModel

model = CasadiModel(bld)
model.ini({...}, xy_0={...})
model.run(10.0)
model.post()
```

`run(t_end, update_dict=None, method=None, **opts)` selects the time-domain
integration strategy (default `'idas_grid'`):

```python
model.run(10.0)                              # 'idas_grid': one IDAS grid call, ~30x faster
model.run(10.0, method='idas')               # legacy per-Dt IDAS loop (exact reproduction)
model.run(10.0, method='trapezoidal')        # fixed-step implicit, 2nd order
model.run(10.0, method='backward_euler')     # fixed-step implicit, 1st order (stiff)
model.run(10.0, method='trapezoidal', linsol='qr')   # CasADi-native ca.Linsol solve
```

See [architecture](architecture.md) ("How casadi_model.py runs") for why the
per-`Dt` loop was slow, the fixed-step formulation, and the `linsol` backends.

### Diagnostics

Jacobian-health utilities (rank, conditioning, sparsity) live in
`pydae.core.diagnostics` and are useful for debugging ill-posed DAEs before
trying to run them.

## Typical sys_dict shape

```python
sys_dict = {
    "name": "my_system",
    "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0},
    "f_list": [...],          # differential equations dx/dt = f
    "g_list": [...],          # algebraic equations 0 = g
    "x_list": [...],          # differential state variables
    "y_ini_list": [...],      # algebraic unknowns at initialisation
    "y_run_list": [...],      # algebraic unknowns during simulation
    "u_ini_dict": {...},      # inputs fixed at ini
    "u_run_dict": {...},      # inputs that may vary during run
    "h_dict": {...},          # output expressions to record
}
```

See [Getting started](getting_started.md) for a complete worked example.

## Where to go next

- Hands-on walkthrough: [Getting started](getting_started.md).
- API reference: [API](api.md).
- Building a balanced power system instead of hand-writing equations:
  [pydae-bps](https://pydae-bps.readthedocs.io/).
- Building an unbalanced distribution system:
  [pydae-uds](https://pydae-uds.readthedocs.io/).
