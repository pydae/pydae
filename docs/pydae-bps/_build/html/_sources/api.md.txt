# API reference

```{note}
This is a scaffold. To enable auto-generated API docs from docstrings:

1. Install `pydae-bps` (and its `pydae` dependency) in the environment that
   builds the docs. The `.readthedocs.yaml` in this folder does this
   automatically on RTD.
2. Flip `autosummary_generate = True` in `conf.py`.
3. Uncomment the `autosummary` block below.
```

The top-level entry point is:

- **`pydae.bps.BpsBuilder`** — consumes a network JSON/HJSON file and
  populates `sys_dict`.

Element-family subpackages (each with its own element-level API):

- `pydae.bps.syns` — synchronous machines (Milano 2/3/4/6-order, SCIB, …)
- `pydae.bps.avrs` — automatic voltage regulators (SEXS, NTSST1, NTSST4, …)
- `pydae.bps.govs` — turbine governors (HYGOV, NTSIEEEG1, …)
- `pydae.bps.vsgs` — virtual synchronous generators (LEON, REGFM-B1, uVSG, …)
- `pydae.bps.vscs` — voltage-source converters (grid-following / forming)
- `pydae.bps.vsc_ctrls` — VSC outer-loop controls
- `pydae.bps.vsc_models` — VSC inner-loop / physical models
- `pydae.bps.wecs` — wind energy conversion systems
- `pydae.bps.pvs` — PV plants
- `pydae.bps.loads` — ZIP loads
- `pydae.bps.lines` — lines and line-DTR
- `pydae.bps.pods` — power oscillation dampers
- `pydae.bps.sources` — ideal voltage sources, GENAPE
- `pydae.bps.pssdesigner` — PSS design helper

<!--
```{eval-rst}
.. autosummary::
    :toctree: _autosummary
    :recursive:

    pydae.bps
```
-->
