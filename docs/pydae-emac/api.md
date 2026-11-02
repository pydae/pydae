# API reference

```{note}
This is a scaffold. To enable auto-generated API docs from docstrings, flip
`autosummary_generate = True` in `conf.py` and uncomment the block below.
```

The top-level entry point is:

- **`pydae.emac.EmacBuilder`** — consumes a drive JSON/HJSON file or dict and
  populates `sys_dict`; `build(name)` returns a ready `CasadiModel`.

Subpackages:

- `pydae.emac.machines` — DAE machine models (`pmsm_dq`, `pmsm_lut`) and the
  rotor (`mechanics`)
- `pydae.emac.ctrls` — DAE controllers (`foc_pi`)
- `pydae.emac.luts` — FEA tables: `load_fea_lut`, `build_flux_tables`,
  `mean_tables`, `save_fea_lut`, and synthetic tables with known answers
  (`linear_fea_lut`)
- `pydae.emac.drive` — the switched multirate solver: `build_machine`,
  `build_window`, `build_control_period`, `simulate`, `PIController`,
  `PredictiveController`
- `pydae.emac.utils.analysis` — `harmonics`, `switching_ripple`,
  `incremental_inductances`, `time_average`

<!--
```{eval-rst}
.. autosummary::
    :toctree: _autosummary
    :recursive:

    pydae.emac
```
-->
