# line_dtr

*Lines — pydae-bps model.*

## Model description

Dynamic Thermal Rating (DTR) lines for pydae-bps.

A DTR line augments the ordinary :math:`\pi`-section with a *thermal* model of
the conductor, so its temperature becomes a state of the DAE. Two thermal
models are available, dispatched on the resolved ``dtr`` data:

- **overhead** (default) — IEEE-738 bare-conductor heat balance in air
  (:func:`DTR_Symbolic`, this module). Single conductor node cooled by forced
  + natural convection and radiation, heated by Joule and solar gain.
- **cable** — IEC 60853 two-node buried-cable ladder
  (:func:`pydae.bps.lines.line_dtr_cable.DTR_Cable_Symbolic`), selected with
  ``dtr: {model: "cable", ...}``.

**Closed-loop coupling.** The series admittance ``g_{line}``/``b_{line}`` is
*not* a constant parameter: it is an algebraic variable evaluated from the
temperature-dependent resistance

.. math::  R(T) = R_{20}\,\bigl(1 + \alpha_R (T - 20)\bigr),

so the conductor temperature feeds back into the network resistance, losses and
voltage drop (see :func:`add_line_dtr`). The same ``R_20`` (from the line's
``R_km``) and single coefficient ``alpha_R`` drive both the network admittance
and the conductor heat balance — one resistance per line.

**Data interface.** Cable/ambient parameters are resolved per line by
:func:`_resolve_dtr_data` from the ``dtr`` field, which may be:

- ``true``                       — built-in defaults;
- an inline ``{...}`` dict       — those values;
- ``"code_name"``                — copy a named entry from the top-level
                                   ``dtr_codes`` section;
- ``{code: "code_name", ...}``   — that named code with per-line overrides.

This lets a cable type be defined once in ``dtr_codes`` and reused by many
lines. Results are summarised by :mod:`pydae.bps.utils.dtr_report`.

Originally created 2022-08-10 (@jmmauricio).

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `line_dtr` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $D_{cond}$ | `D_conductor` | 0.0277 | m | Conductor overall diameter |
| $D_{core}$ | `D_core` | 0.00308 | m | Steel-core diameter (0 if homogeneous) |
| $\epsilon$ | `Emissivity` | 0.8 | - | Surface emissivity (radiation) |
| $\alpha_s$ | `Absorptivity` | 0.8 | - | Solar absorptivity |
| $H_{elev}$ | `Elevation` | 0.0 | m | Line elevation above sea level |
| $m$ | `Mass_linear` | 1.62 | kg/m | Conductor mass per unit length |
| $C_p$ | `Cp` | 900.0 | J/(kg·K) | Conductor specific heat |
| $K_{dtr}$ | `K_dtr` | 0.0 | - | Network-current to conductor-current scaling |
| $R_{ac,20}$ | `R_ac_20` | 0.0005 | Ω/m | AC resistance at 20 °C (defaults to R_km/1000) |
| $\alpha_R$ | `alpha_R` | 0.004 | 1/K | Resistance temperature coefficient |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{amb}$ | `t_ambient` | 25.0 | °C | Ambient air temperature |
| $v_{wind}$ | `v_wind` | 1.0 | m/s | Wind speed |
| $\phi$ | `angle_wind` | 90.0 | deg | Wind attack angle to the line |
| $Q_s$ | `irradiance` | 1000.0 | W/m² | Solar irradiance |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{cond}$ | `t_conductor` |  | °C | Conductor temperature |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\rho_{air}$ | `dens` |  | kg/m³ | Air density |
| $Re$ | `reynolds` |  | - | Reynolds number |
| $Ra$ | `rayleigh` |  | - | Rayleigh number |
| $Nu$ | `nu` |  | - | Nusselt number |
| $q_s$ | `p_s` |  | W/m | Solar heat gain |
| $q_c$ | `p_c` |  | W/m | Convective heat loss |
| $g$ | `g` |  | pu | Series conductance G(T) (temperature-coupled) |
| $b$ | `b` |  | pu | Series susceptance B(T) (temperature-coupled) |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R_{ac}$ | `R_ac` |  | Ω/m | AC resistance at conductor temperature |
| $q_j$ | `p_j` |  | W/m | Joule heat gain |


## Source

- Module: `pydae.bps.lines.line_dtr`
- File: [`packages/pydae-bps/src/pydae/bps/lines/line_dtr.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/lines/line_dtr.py)
