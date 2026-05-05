# tgov1

*Turbine governors — pydae-bps model.*

## Model description

IEEE TGOV1 steam turbine-governor.

A simple steam turbine governor with a first-order lag valve block and a
lead-lag turbine representation.  The model matches the PSS/E TGOV1
definition and the IEEE recommended practice.

**Signal path**

Speed error feeds the valve reference,

$$u_1 = p_c + \frac{1 - \omega}{R}$$

The valve position is a first-order lag of $u_1$ with position limits
$[V_{min}, V_{max}]$.  Let $x_1$ be the integrator state and
$y_1 = \mathrm{sat}(x_1, V_{min}, V_{max})$:

$$\frac{d x_1}{dt} = \frac{u_1 - x_1}{T_1} + K_{awu}\,(y_1 - x_1)$$

The anti-windup term $K_{awu}$ drives $x_1$ back when the position
limiter saturates.

The turbine is represented by a lead-lag $(1 + T_2\,s)/(1 + T_3\,s)$
with state $x_2$:

$$\frac{d x_2}{dt} = \frac{y_1 - x_2}{T_3}$$

The mechanical power output includes turbine damping:

$$0 = x_2 + \frac{T_2}{T_3}\,(y_1 - x_2) - D_t\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$): $u_1 = p_c$, $x_1 = y_1 = p_c$,
$x_2 = p_c$, and $p_m = p_c$ (for any $T_2$, $T_3$, $D_t = 0$).

**Configuration**

Example data entry (typical defaults)::

    "gov": {"type": "tgov1",
            "R": 0.05,
            "T_1": 0.5,
            "V_max": 1.0, "V_min": 0.0,
            "T_2": 2.1,  "T_3": 7.0,
            "D_t": 0.0,
            "p_c": 0.8}

The ``p_c`` field is the scheduled dispatch — equals the mechanical
power output at steady state when $\omega = 1$.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `tgov1` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R_gov` | 0.05 | pu | Permanent droop (speed regulation) |
| $T_1$ | `T_1_gov` | 0.5 | s | Steam chest (valve actuator) time constant |
| $V_{max}$ | `V_max_gov` | 1.0 | pu | Maximum valve position limit |
| $V_{min}$ | `V_min_gov` | 0.0 | pu | Minimum valve position limit |
| $T_2$ | `T_2_gov` | 2.1 | s | Lead time constant of turbine lead-lag |
| $T_3$ | `T_3_gov` | 7.0 | s | Lag time constant of turbine lead-lag |
| $D_t$ | `D_t_gov` | 0.0 | pu | Turbine damping coefficient |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.8 | pu | Scheduled dispatch (load reference). Equals p_m at steady state when omega = 1. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_1$ | `x_1_gov` |  | pu | Valve position integrator state |
| $x_2$ | `x_2_gov` |  | pu | Lead-lag turbine lag state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` |  | pu | Mechanical power delivered to the synchronous machine |


## Source

- Module: `pydae.bps.govs.tgov1`
- File: [`packages/pydae-bps/src/pydae/bps/govs/tgov1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/tgov1.py)
