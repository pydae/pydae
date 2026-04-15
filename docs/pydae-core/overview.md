# Overview

## What pydae-core does

`pydae-core` takes a symbolic description of a DAE system and turns it into a
fast, standalone simulation object. Internally the pipeline is:

```{mermaid}
flowchart LR
    A[sys_dict<br/>SymPy expressions] --> B[Builder]
    B --> C[Generated C code]
    C --> D[Compiled .so / .dll<br/>ctypes or CFFI]
    D --> E[Model<br/>runtime API]
    E --> F[ini / run / post<br/>analysis]
```

The key idea is that **building is symbolic and slow, simulation is numerical
and fast**. You pay the compilation cost once and then run arbitrarily many
scenarios through the compiled binary.

## Main objects

### `Builder`

Consumes a `sys_dict` of SymPy expressions and produces compiled C code plus a
Python wrapper.

```python
from pydae.core import Builder

bld = Builder(sys_dict, target="ctypes")
bld.build()
```

### `Model`

The runtime interface to a built system. Supports steady-state initialisation
(`ini`), time-domain simulation (`run`), and post-processing (`post`).

```python
from pydae.core import Model

model = Model("my_system")
model.ini({...}, xy_0={...})
model.run(10.0, {"u_1": 1.0})
model.post()
```

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
