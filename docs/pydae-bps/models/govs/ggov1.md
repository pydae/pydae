# ggov1

*Turbine governors — pydae-bps model.*

## Model description

IEEE / GE GGOV1 general-purpose turbine-governor.

A PI speed governor followed by a rate-limited fuel-valve actuator and a
lead-lag turbine representation.  Originally developed for gas turbines but
widely used for any fast-acting prime mover where both PI speed control and
fuel-valve rate limits matter.

**Signal path**

Speed error through permanent droop:

$$e = \frac{1 - \omega}{R}$$

Governor PI (integral state $x_{gov}$, proportional gain $K_{pgov}$):

$$\frac{d x_{gov}}{dt} = K_{igov}\, e, \qquad u_{gov} = K_{pgov}\, e + x_{gov}$$

Fuel demand with position limits $[V_{min}, V_{max}]$:

$$u_{fuel} = \mathrm{sat}\!\left(p_c + u_{gov},\; V_{min},\; V_{max}\right)$$

Fuel valve actuator (rate-limited integrator, rates $[R_{close}, R_{open}]$):

$$v_{act} = \mathrm{sat}\!\left(\frac{u_{fuel} - x_{fuel}}{T_{act}},\;
            R_{close},\; R_{open}\right)$$

$$\frac{d x_{fuel}}{dt} = v_{act}$$

Turbine lead-lag $(1 + T_c\,s)/(1 + T_b\,s)$ with lag state $x_{tb}$:

$$\frac{d x_{tb}}{dt} = \frac{x_{fuel} - x_{tb}}{T_b}$$

$$y_{turb} = x_{tb} + \frac{T_c}{T_b}\,(x_{fuel} - x_{tb})$$

Mechanical power:

$$0 = K_{turb}\,(y_{turb} - W_{fnl}) - D_m\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$): $e = 0$, $x_{gov} = 0$, $u_{fuel} = p_c$,
$x_{fuel} = p_c$, $y_{turb} = p_c$, and:

$$p_m = K_{turb}\,(p_c - W_{fnl})$$

With the default $K_{turb} = 1.5$, $W_{fnl} = 0.1$:
$p_m = 1.5\,(p_c - 0.1)$, so set `p_c = p_m/1.5 + 0.1`.

## Usage

```hjson
gov: {
  type: "ggov1",
  R: 0.05,
  K_pgov: 10.0,    K_igov: 2.0,
  T_act: 0.5,
  R_open: 0.1,     R_close: -0.1,
  V_max: 1.0,      V_min: 0.0,
  K_turb: 1.5,     W_fnl: 0.1,
  T_b: 0.5,        T_c: 0.0,
  D_m: 0.0,
  p_c: 0.633
}
```

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R` | 0.05 | pu | Permanent droop (speed regulation) |
| $K_{pgov}$ | `K_pgov` | 10.0 | pu | Governor proportional gain |
| $K_{igov}$ | `K_igov` | 2.0 | pu/s | Governor integral gain |
| $T_{act}$ | `T_act` | 0.5 | s | Fuel valve actuator time constant |
| $R_{open}$ | `R_open` | 0.1 | pu/s | Maximum valve opening rate |
| $R_{close}$ | `R_close` | -0.1 | pu/s | Maximum valve closing rate (negative) |
| $V_{max}$ | `V_max` | 1.0 | pu | Maximum valve (fuel) position |
| $V_{min}$ | `V_min` | 0.0 | pu | Minimum valve (fuel) position |
| $K_{turb}$ | `K_turb` | 1.5 | pu | Turbine gain |
| $W_{fnl}$ | `W_fnl` | 0.1 | pu | No-load fuel flow |
| $T_b$ | `T_b` | 0.5 | s | Turbine lag time constant (must be > 0) |
| $T_c$ | `T_c` | 0.0 | s | Turbine lead time constant (0 = pure lag) |
| $D_m$ | `D_m` | 0.0 | pu | Mechanical damping coefficient |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.633 | pu | Fuel valve setpoint (load reference). At steady state: $p_m = K_{turb}(p_c - W_{fnl})$. |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $x_{gov}$ | `x_gov_gov` | pu | Governor PI integral state |
| $x_{fuel}$ | `x_fuel_gov` | pu | Fuel valve actuator state (valve position) |
| $x_{tb}$ | `x_tb_gov` | pu | Turbine lead-lag lag state |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_m$ | `p_m` | pu | Mechanical power delivered to the synchronous machine |

## Source

- Module: `pydae.bps.govs.ggov1`
- File: [`packages/pydae-bps/src/pydae/bps/govs/ggov1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/ggov1.py)
