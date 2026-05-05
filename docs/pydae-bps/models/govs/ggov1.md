# ggov1

*Turbine governors — pydae-bps model.*

## Model description

IEEE / GE GGOV1 general-purpose governor.

Three-state model combining a PI speed governor with droop, a
rate-limited fuel valve actuator, and a lead-lag turbine block.
The model matches the PSS/E GGOV1 definition and is widely used for
gas and diesel turbine prime movers.

**Signal path**

Speed deviation and permanent-droop error:

$$e = \frac{1 - \omega}{R}$$

The governor PI produces the fuel demand correction:

$$\frac{d x_{gov}}{dt} = K_{igov}\, e$$
$$u_{gov} = K_{pgov}\, e + x_{gov}$$

The fuel demand is clamped to the valve position range:

$$u_{fuel} = \mathrm{sat}(p_c + u_{gov},\; V_{min},\; V_{max})$$

The fuel valve actuator is a rate-limited integrator.  Valve rate is
clamped to $[R_{close},\, R_{open}]$:

$$v_{act} = \mathrm{sat}\!\left(\frac{u_{fuel} - x_{fuel}}{T_{act}},\;
            R_{close},\; R_{open}\right)$$
$$\frac{d x_{fuel}}{dt} = v_{act}$$

The turbine is a lead-lag block $(1 + T_c\,s)/(1 + T_b\,s)$ driven
by the valve position $x_{fuel}$.  Lead-lag state $x_{tb}$:

$$\frac{d x_{tb}}{dt} = \frac{x_{fuel} - x_{tb}}{T_b}$$
$$y_{turb} = x_{tb} + \frac{T_c}{T_b}\,(x_{fuel} - x_{tb})$$

Mechanical power with no-load fuel offset and self-damping:

$$0 = K_{turb}\,(y_{turb} - W_{fnl}) - D_m\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$): $e = 0$, $x_{gov} = 0$,
$u_{fuel} = p_c$, $x_{fuel} = p_c$, $x_{tb} = p_c$,
$y_{turb} = p_c$, and

$$p_m = K_{turb}\,(p_c - W_{fnl})$$

The ``p_c`` field therefore represents the **fuel valve setpoint**.
For the test fixture ($K_{turb} = 1$, $W_{fnl} = 0$) this simplifies
to $p_m = p_c$.  For a typical gas turbine ($K_{turb} = 1.5$,
$W_{fnl} = 0.1$) set $p_c = p_m / K_{turb} + W_{fnl}$.

**Note on $T_b$**

$T_b$ must be $> 0$.  Setting $T_c = 0$ gives a pure lag (common for
combustion chambers); setting $T_c = T_b$ gives unity (direct
pass-through).

**Configuration**

Example data entry (typical gas-turbine defaults)::

    "gov": {"type": "ggov1",
            "R": 0.05,
            "K_pgov": 10.0,  "K_igov": 2.0,
            "T_act": 0.5,
            "R_open": 0.1,   "R_close": -0.1,
            "V_max": 1.0,    "V_min": 0.0,
            "K_turb": 1.5,   "W_fnl": 0.1,
            "T_b": 0.5,      "T_c": 0.0,
            "D_m": 0.0,
            "p_c": 0.633}

The ``p_c`` value seeds the valve and turbine states at ini() and sets
the steady-state operating point.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `ggov1` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R_gov` | 0.05 | pu | Permanent droop (speed regulation) |
| $K_{pgov}$ | `K_pgov_gov` | 10.0 | pu | Governor proportional gain |
| $K_{igov}$ | `K_igov_gov` | 2.0 | pu/s | Governor integral gain |
| $T_{act}$ | `T_act_gov` | 0.5 | s | Fuel valve actuator time constant |
| $R_{open}$ | `R_open_gov` | 0.1 | pu/s | Maximum valve opening rate |
| $R_{close}$ | `R_close_gov` | -0.1 | pu/s | Maximum valve closing rate (negative) |
| $V_{max}$ | `V_max_gov` | 1.0 | pu | Maximum valve (fuel) position |
| $V_{min}$ | `V_min_gov` | 0.0 | pu | Minimum valve (fuel) position |
| $K_{turb}$ | `K_turb_gov` | 1.5 | pu | Turbine gain |
| $W_{fnl}$ | `W_fnl_gov` | 0.1 | pu | No-load fuel flow |
| $T_b$ | `T_b_gov` | 0.5 | s | Turbine lag time constant (must be > 0) |
| $T_c$ | `T_c_gov` | 0.0 | s | Turbine lead time constant (0 = pure lag) |
| $D_m$ | `D_m_gov` | 0.0 | pu | Mechanical damping coefficient |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.633 | pu | Fuel valve setpoint (load reference). At steady state: p_m = K_turb*(p_c - W_fnl). |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_{gov}$ | `x_gov_gov` |  | pu | Governor PI integral state |
| $x_{fuel}$ | `x_fuel_gov` |  | pu | Fuel valve actuator state (valve position) |
| $x_{tb}$ | `x_tb_gov` |  | pu | Turbine lead-lag lag state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` |  | pu | Mechanical power delivered to the synchronous machine |


## Source

- Module: `pydae.bps.govs.ggov1`
- File: [`packages/pydae-bps/src/pydae/bps/govs/ggov1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/ggov1.py)
