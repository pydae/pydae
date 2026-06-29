# line_dtr_cable

*Lines — pydae-bps model.*

## Model description

Dynamic Thermal Rating for **underground cables** — IEC 60853 two-node model.

This is the buried-cable counterpart of the overhead IEEE-738 model in
:mod:`pydae.bps.lines.line_dtr`. A cable is not air-cooled: heat is *conducted*
outward through the insulation, sheath and jacket into the surrounding soil.
IEC 60287 gives the steady-state thermal resistances; IEC 60853 reduces the
transient response to a two-loop (two-node) lumped RC ladder via the Van Wormer
capacitance-splitting factor.

Network (per unit length)::

    W_c + p·W_d                     (1-p)·W_d + λ1·W_c
         │                                  │
       [θ_c]───T1───[θ_e]───────T4────────[θ_soil]
        Q1            Q2

    θ_c   conductor temperature                 [°C]   (state)
    θ_e   external / sheath-region temperature  [°C]   (state)
    θ_soil  ambient soil temperature            [°C]   (input)
    W_c   conductor loss  I²·R_ac(θ_c)          [W/m]
    W_d   dielectric loss                       [W/m]
    T1    insulation thermal resistance         [K·m/W]  (IEC 60287)
    T4    external/soil thermal resistance       [K·m/W]  (IEC 60287)
    Q1,Q2 lumped thermal capacitances           [J/(m·K)] (Van Wormer split)
    p     Van Wormer factor                      [-]
    λ1    sheath loss factor                      [-]

State equations::

    Q1·dθ_c/dt = (W_c + p·W_d)            − (θ_c − θ_e)/T1
    Q2·dθ_e/dt = (θ_c − θ_e)/T1 + (1−p)·W_d + λ1·W_c − (θ_e − θ_soil)/T4

T1, T4, Q1, Q2 and p are derived from the cable geometry/materials in Python
(:func:`_cable_thermal_network`) and registered as **per-line parameters** with
those numeric defaults; any of them can be overridden directly in the ``dtr``
data (or a ``dtr_codes`` entry). The temperature-dependent conductor resistance,
the network current and the soil temperature remain symbolic.

Selected by ``dtr: {model: "cable", ...}`` (or a ``dtr_codes`` entry whose
``model`` is ``"cable"``) on a ``lines[]`` entry.

.. note::
   The soil capacitance ``Q_soil`` is lumped over a finite shell of outer
   diameter ``soil_shell × De`` — a transparent approximation of the
   semi-infinite-medium response. IEC 60853 proper uses the exponential-integral
   soil response; the two-node reduction trades that for a single dominant soil
   time constant, which is the standard simplification for real-time / control
   studies. Tune ``soil_shell`` (or override ``Q2``/``T4``) to match a reference.

Default geometry follows the CIGRE-575 European MV cable NA2XS2Y 120 mm²
(Table 6.11); soil resistivity, burial depth and material heat capacities use
IEC typical values (not given by the benchmark).

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `line_dtr_cable` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $d_c$ | `dc` | 0.0124 | m | Conductor diameter |
| $t_i$ | `t_ins` | 0.0055 | m | Insulation thickness |
| $D_e$ | `De` | 0.0342 | m | Cable overall diameter |
| $\rho_{ins}$ | `rho_ins` | 3.5 | K·m/W | Insulation thermal resistivity |
| $\rho_{soil}$ | `rho_soil` | 1.0 | K·m/W | Soil thermal resistivity |
| $L$ | `L_depth` | 0.8 | m | Laying depth to cable axis |
| $c_{v,c}$ | `cv_cond` | 2500000.0 | J/(m³·K) | Conductor volumetric heat capacity |
| $c_{v,i}$ | `cv_ins` | 2400000.0 | J/(m³·K) | Insulation volumetric heat capacity |
| $c_{v,s}$ | `cv_soil` | 2000000.0 | J/(m³·K) | Soil volumetric heat capacity |
| $k_{soil}$ | `soil_shell` | 3.0 | - | Soil-capacitance shell, outer Ø = k·De |
| $R_{ac,20}$ | `R_ac_20` | 0.000264 | Ω/m | AC resistance at 20 °C (defaults to R_km/1000) |
| $\alpha_R$ | `alpha_R` | 0.004 | 1/K | Resistance temperature coefficient |
| $\lambda_1$ | `lambda1` | 0.0 | - | Sheath loss factor |
| $W_d$ | `Wd` | 0.0 | W/m | Dielectric loss |
| $K_{dtr}$ | `K_dtr` | 1.0 | - | Network- to conductor-current scaling |
| $T_1$ | `T1` |  | K·m/W | Insulation thermal resistance (IEC 60287) |
| $T_4$ | `T4` |  | K·m/W | External/soil thermal resistance (IEC 60287) |
| $Q_1$ | `Q1` |  | J/(m·K) | Conductor-node thermal capacitance |
| $Q_2$ | `Q2` |  | J/(m·K) | External-node thermal capacitance |
| $p$ | `p_vw` |  | - | Van Wormer capacitance-split factor |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{soil}$ | `t_soil` | 15.0 | °C | Ambient soil temperature |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{cond}$ | `t_conductor` |  | °C | Conductor temperature |
| $T_{sheath}$ | `t_sheath` |  | °C | External/sheath-region temperature |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $W_c$ | `W_c` |  | W/m | Conductor (Joule) loss |
| $W_{tot}$ | `W_total` |  | W/m | Total heat input |
| $R_{ac}$ | `R_ac` |  | Ω/m | AC resistance at conductor temperature |
| $I_{cond}$ | `I_cond` |  | A | Conductor current |


## Source

- Module: `pydae.bps.lines.line_dtr_cable`
- File: [`packages/pydae-bps/src/pydae/bps/lines/line_dtr_cable.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/lines/line_dtr_cable.py)
