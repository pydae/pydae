# dgov

*Turbine governors — pydae-bps model.*

## Model description

DGOV diesel engine governor.

Three-state proportional governor for diesel prime movers.  Unlike GGOV1
(which has PI integral action), DGOV uses a purely proportional speed path
filtered through a single governor lag, followed by a rate-limited fuel
actuator and a combustion delay.  This matches the fast-response,
droop-governed behaviour typical of diesel generator sets.

**Signal path**

Speed error through permanent droop and governor gain:

$$e = \frac{K\,(1 - \omega)}{R}$$

The governor proportional lag $T_1$ smooths the speed signal:

$$\frac{d x_{gov}}{dt} = \frac{e - x_{gov}}{T_1}$$

The fuel demand sums the load reference and the governor output, clamped to
the valve range $[V_{min}, V_{max}]$:

$$u_{fuel} = \mathrm{sat}(p_c + x_{gov},\; V_{min},\; V_{max})$$

The actuator is a rate-limited integrator.  Valve rate is clamped to
$[R_{close}, R_{open}]$:

$$v_{act} = \mathrm{sat}\!\left(\frac{u_{fuel} - x_{act}}{T_2},\;
            R_{close},\; R_{open}\right)$$

$$\frac{d x_{act}}{dt} = v_{act}$$

The combustion / engine lag $T_3$ filters the fuel flow into shaft torque:

$$\frac{d x_{eng}}{dt} = \frac{x_{act} - x_{eng}}{T_3}$$

Mechanical power with no-load fuel and self-damping:

$$0 = K_{turb}\,(x_{eng} - W_{fnl}) - D_m\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$): $e = 0$, $x_{gov} = 0$,
$u_{fuel} = p_c$, $x_{act} = p_c$, $x_{eng} = p_c$, and:

$$p_m = K_{turb}\,(p_c - W_{fnl})$$

With $K_{turb} = 1$, $W_{fnl} = 0$ (test defaults): $p_m = p_c$.

**DGOV vs GGOV1**

DGOV has no governor integral state: speed error correction vanishes at
$\omega = 1$ regardless of load, so a non-zero steady-state speed deviation is
possible under load change.  GGOV1's integral forces $\omega \to 1$
asymptotically.  DGOV is therefore appropriate when the droop characteristic
governs frequency in island or parallel operation without isochronous
correction.

## Usage

```hjson
gov: {
  type: "dgov",
  R: 0.05,     K: 1.0,
  T_1: 0.02,   T_2: 0.3,
  R_open: 0.3, R_close: -0.3,
  V_max: 1.0,  V_min: 0.0,
  T_3: 0.5,
  K_turb: 1.0, W_fnl: 0.0,
  D_m: 0.0,
  p_c: 0.8
}
```

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R` | 0.05 | pu | Permanent droop (speed regulation) |
| $K$ | `K` | 1.0 | pu | Governor proportional gain |
| $T_1$ | `T_1` | 0.02 | s | Governor lag time constant |
| $T_2$ | `T_2` | 0.3 | s | Fuel valve actuator time constant |
| $R_{open}$ | `R_open` | 0.3 | pu/s | Maximum valve opening rate |
| $R_{close}$ | `R_close` | -0.3 | pu/s | Maximum valve closing rate (negative) |
| $V_{max}$ | `V_max` | 1.0 | pu | Maximum fuel valve position |
| $V_{min}$ | `V_min` | 0.0 | pu | Minimum fuel valve position |
| $T_3$ | `T_3` | 0.5 | s | Engine combustion lag time constant |
| $K_{turb}$ | `K_turb` | 1.0 | pu | Engine / turbine gain |
| $W_{fnl}$ | `W_fnl` | 0.0 | pu | No-load fuel flow |
| $D_m$ | `D_m` | 0.0 | pu | Mechanical damping coefficient |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.8 | pu | Fuel valve setpoint (load reference). At steady state: $p_m = K_{turb}(p_c - W_{fnl})$. |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $x_{gov}$ | `x_gov_gov` | pu | Governor proportional lag state |
| $x_{act}$ | `x_act_gov` | pu | Fuel valve actuator state |
| $x_{eng}$ | `x_eng_gov` | pu | Engine combustion lag state |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_m$ | `p_m` | pu | Mechanical power delivered to the synchronous machine |

## Source

- Module: `pydae.bps.govs.dgov`
- File: [`packages/pydae-bps/src/pydae/bps/govs/dgov.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/dgov.py)
