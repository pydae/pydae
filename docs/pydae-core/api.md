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

- **`pydae.core.Builder`** — builds compiled DAE models from a `sys_dict`.
- **`pydae.core.Model`** — runtime interface for a built model
  (`ini`, `run`, `post`, `get_values`).
- **`pydae.core.diagnostics`** — Jacobian health checks and structural
  analysis helpers.

<!--
```{eval-rst}
.. autosummary::
    :toctree: _autosummary
    :recursive:

    pydae.core
```
-->
