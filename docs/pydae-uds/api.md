# API reference

```{note}
This is a scaffold. To enable auto-generated API docs from docstrings:

1. Install `pydae-uds` (and its `pydae` dependency) in the environment that
   builds the docs. The `.readthedocs.yaml` in this folder does this
   automatically on RTD.
2. Flip `autosummary_generate = True` in `conf.py`.
3. Uncomment the `autosummary` block below.
```

The top-level entry point is:

- **`pydae.uds.UdsBuilder`** — consumes a network JSON/HJSON file and
  populates `sys_dict`.

Element-family subpackages:

- `pydae.uds.lines` — three-phase line models (including DTR variants)
- `pydae.uds.transformers` — Dyn11, Dyg11, and related 3-phase transformers
- `pydae.uds.loads` — unbalanced loads
- `pydae.uds.sources` — ideal 3-phase sources, DC sources
- `pydae.uds.vscs` — VSCs (3-phase, 4-wire, grid-forming / following)
- `pydae.uds.vsc_ctrls` — VSC outer-loop controls
- `pydae.uds.vsgs` — virtual synchronous generators
- `pydae.uds.ess` — battery energy storage systems (DC/DC and DC/AC)
- `pydae.uds.fcs` — fuel cells (SOFC, PEMFC) and DC/AC stages
- `pydae.uds.pvs` — PV plants with MPPT, DC/DC, DC/AC
- `pydae.uds.miscellaneous` — breakers, fault models, utilities
- `pydae.uds.utils` — reporting helpers (`report_v`, `get_v`, `get_i`,
  `get_power`)

<!--
```{eval-rst}
.. autosummary::
    :toctree: _autosummary
    :recursive:

    pydae.uds
```
-->
