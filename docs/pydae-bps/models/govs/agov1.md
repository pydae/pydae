# agov1

*Turbine governors — pydae-bps model.*

## Model description

AGOV1 auxiliary governor with load-sharing and secondary frequency control.

Two-lag turbine model with a droop governor, an integral MW (load-sharing)
controller, and secondary frequency control participation.  Intended for
thermal units where load-sharing between parallel generators is required.

**Signal path**

The load reference integrates load-sharing error and secondary control:

$$\frac{d \xi_{imw}}{dt} = K_{imw}\,(p_c - p_g) - \varepsilon_{leak}\,\xi_{imw}$$

where $\varepsilon_{leak} = 10^{-6}$ prevents integrator wind-up at
steady state without load sharing.

Power reference:

$$0 = -p_{m,ref} + \xi_{imw} + p_r + K_{sec}\,p_{agc}
      - \frac{1}{Droop}\,(\omega - \omega_{ref})$$

Turbine two-lag cascade ($T_1$, $T_3$):

$$\frac{d x_1}{dt} = \frac{p_{m,ref} - x_1}{T_1}, \qquad
  \frac{d x_2}{dt} = \frac{x_1 - x_2}{T_3}$$

Mechanical power (lead-lag $(1 + T_2\,s)/(1 + T_3\,s)$):

$$0 = \frac{T_2}{T_3}\,(x_1 - x_2) + x_2 - p_m$$

**Inputs**

- $p_c$ — load setpoint (from dispatch or user).
- $p_r$ — power ramp reference (exogenous, default 0).
- $p_{agc}$ — system AGC signal (shared symbol `p_agc`, zero if no AGC).
- $p_g$ — active grid injection (algebraic, from synchronous machine).

**Steady-state relation**

At $\omega = 1$, $\varepsilon_{leak} \to 0$, $K_{imw} \to 0$:
$p_{m,ref} = p_c + p_r + K_{sec}\,p_{agc}$, $x_1 = x_2 = p_{m,ref}$,
$p_m = p_{m,ref}$.

With load sharing ($K_{imw} > 0$): integrator drives $p_g \to p_c$.

## Usage

```hjson
gov: {
  type: "agov1",
  Droop: 0.05,
  T_1: 1.0,    T_2: 0.5,    T_3: 10.0,
  K_imw: 0.0,
  p_c: 0.8
}
```

The `K_sec` parameter is read from the parent synchronous-machine entry
(`K_sec: 0.0` by default) and scales the system AGC signal.

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $Droop$ | `Droop` | 0.05 | pu | Permanent droop (speed regulation) |
| $T_1$ | `T_gov_1` | 1.0 | s | First turbine lag time constant |
| $T_2$ | `T_gov_2` | 0.5 | s | Lead-lag numerator time constant |
| $T_3$ | `T_gov_3` | 10.0 | s | Second turbine lag time constant |
| $K_{imw}$ | `K_imw` | 0.0 | pu/s | Load-sharing integral gain (0 = disabled) |
| $K_{sec}$ | `K_sec` | 0.0 | pu | Secondary frequency control participation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.8 | pu | Load setpoint (dispatch reference) |
| $p_r$ | `p_r` | 0.0 | pu | Power ramp reference |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $x_1$ | `x_gov_1` | pu | First turbine lag state |
| $x_2$ | `x_gov_2` | pu | Second turbine lag state |
| $\xi_{imw}$ | `xi_imw` | pu | Load-sharing integral state |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_{m,ref}$ | `p_m_ref` | pu | Power reference (droop + AGC + load-sharing) |
| $p_m$ | `p_m` | pu | Mechanical power delivered to the synchronous machine |

## Source

- Module: `pydae.bps.govs.agov1`
- File: [`packages/pydae-bps/src/pydae/bps/govs/agov1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/agov1.py)
