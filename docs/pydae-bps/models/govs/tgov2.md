# tgov2

*Turbine governors — pydae-bps model.*

## Model description

IEEE TGOV2 steam turbine-governor with explicit valve rate limits.

Extends [tgov1](tgov1.md) by adding a separate rate limiter on the valve
change rate, allowing independent tuning of valve position limits and opening
/ closing speeds.

**Signal path**

Speed error feeds the valve reference:

$$u_1 = p_c + \frac{1 - \omega}{R}$$

The unsaturated valve change rate is:

$$v_1^{(0)} = \frac{u_1 - x_1}{T_1}$$

A rate limiter constrains the valve movement:

$$v_1 = \mathrm{sat}\!\left(v_1^{(0)},\; V_{R,min},\; V_{R,max}\right)$$

The valve state with anti-windup ($y_1 = \mathrm{sat}(x_1, V_{min}, V_{max})$):

$$\frac{d x_1}{dt} = v_1 + K_{awu}\,(y_1 - x_1)$$

The turbine lead-lag:

$$\frac{d x_2}{dt} = \frac{y_1 - x_2}{T_3}$$

Mechanical power with turbine damping:

$$0 = x_2 + \frac{T_2}{T_3}\,(y_1 - x_2) - D_t\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$): $p_m = p_c$.

**Difference from tgov1**

TGOV1 lumps the valve rate limiting inside the actuator time constant $T_1$
(through anti-windup only).  TGOV2 adds explicit rate limits $V_{R,min}$,
$V_{R,max}$ on the derivative $\dot{x}_1$, giving tighter control over how
fast the valve can open or close.

## Usage

```hjson
gov: {
  type: "tgov2",
  R: 0.05,
  T_1: 0.5,
  V_max: 1.0,    V_min: 0.0,
  V_R_max: 0.2,  V_R_min: -0.2,
  T_2: 2.1,      T_3: 7.0,
  D_t: 0.0,
  p_c: 0.8
}
```

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R` | 0.05 | pu | Permanent droop (speed regulation) |
| $T_1$ | `T_1` | 0.5 | s | Steam chest (valve actuator) time constant |
| $V_{max}$ | `V_max` | 1.0 | pu | Maximum valve position limit |
| $V_{min}$ | `V_min` | 0.0 | pu | Minimum valve position limit |
| $V_{R,max}$ | `V_R_max` | 0.2 | pu/s | Maximum valve opening rate limit |
| $V_{R,min}$ | `V_R_min` | -0.2 | pu/s | Maximum valve closing rate limit (negative) |
| $T_2$ | `T_2` | 2.1 | s | Lead time constant of turbine lead-lag |
| $T_3$ | `T_3` | 7.0 | s | Lag time constant of turbine lead-lag |
| $D_t$ | `D_t` | 0.0 | pu | Turbine damping coefficient |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.8 | pu | Scheduled dispatch (load reference). Equals $p_m$ at steady state when $\omega = 1$. |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $x_1$ | `x_1_gov` | pu | Valve position integrator state |
| $x_2$ | `x_2_gov` | pu | Lead-lag turbine lag state |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_m$ | `p_m` | pu | Mechanical power delivered to the synchronous machine |

## Source

- Module: `pydae.bps.govs.tgov2`
- File: [`packages/pydae-bps/src/pydae/bps/govs/tgov2.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/tgov2.py)
