# API reference

```{note}
This is a scaffold. To enable auto-generated API docs from docstrings:

1. Install `pydae-core` in the environment that builds the docs
   (`pip install -e packages/pydae-core` or rely on the `.readthedocs.yaml`
   which does this automatically on RTD).
2. Flip `autosummary_generate = True` in `conf.py`.
3. Uncomment the `autosummary` block below.
```

The main public names are:

- **`pydae.core.Builder`** — SymPy → C pipeline. Builds compiled DAE models
  from a `sys_dict`.
- **`pydae.core.Model`** — ctypes/CFFI runtime interface for a built model
  (`ini`, `run`, `post`, `get_values`).
- **`pydae.core.builder.CasadiBuilder`** — CasADi → SX graph pipeline.
  No C compiler required.
- **`pydae.core.model.CasadiModel`** — CasADi runtime (IDAS integrator).
- **`pydae.core.common`** — shared parser and symbolic utilities.
- **`pydae.core.diagnostics`** — Jacobian health checks and structural
  analysis helpers.
- **`pydae.api.realtime_api`** — soft real-time simulation with a FastAPI
  REST interface (requires ``pydae[api]``).  See the
  [Real-time API](realtime_api.md) page for usage and design details.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary
    :nosignatures:
    :recursive:

    pydae.core
    pydae.core.builder.sympy_builder
    pydae.core.builder.casadi_builder
    pydae.core.model.ctypes_model
    pydae.core.model.casadi_model
    pydae.core.common.parser
    pydae.core.common.symbolic
    pydae.core.diagnostics
```

## pydae.ssa

```{eval-rst}
.. automodule:: pydae.ssa.ssa
    :members:
    :undoc-members:
    :show-inheritance:
```
