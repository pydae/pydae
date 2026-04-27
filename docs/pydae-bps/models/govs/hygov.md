# hygov

*Turbine governors — pydae-bps model.*

## Model description

IEEE HYGOV hydraulic turbine-governor.

Four-state model combining a droop governor with a dashpot (transient-droop)
network, a rate- and position-limited gate servomotor, an inelastic
water-column penstock, and an ideal hydraulic turbine.

**Signal path**

Speed deviation:

$$\Delta\omega = \omega - 1$$

The dashpot state $c_d$ tracks the gate integrator $x_g$.  Its time-derivative
provides the transient droop signal:

$$\frac{d c_d}{dt} = \frac{x_g - c_d}{T_r}, \qquad
  c_{d,out} = R_r\,\frac{x_g - c_d}{T_r}$$

The pilot valve demand sums the load reference, permanent droop, and transient
droop correction:

$$u_{pv} = p_c - \frac{\Delta\omega}{R} - c_{d,out}$$

A first-order filter with time constant $T_f$ smooths the demand:

$$\frac{d c_{pv}}{dt} = \frac{u_{pv} - c_{pv}}{T_f}$$

The gate servomotor is a rate-limited integrator with position limits
$[G_{min}, G_{max}]$.  Let $y_g = \mathrm{sat}(x_g, G_{min}, G_{max})$:

$$v_g = \mathrm{sat}\!\left(\frac{c_{pv} - x_g}{T_g},\; -V_{g,max},\; V_{g,max}\right)$$

$$\frac{d x_g}{dt} = v_g + K_{awu}\,(y_g - x_g)$$

The penstock obeys the inelastic water-column equation with hydraulic head
$h = (q / y_g)^2$:

$$\frac{d q}{dt} = \frac{1 - h}{T_w}$$

Mechanical power with turbine self-damping:

$$0 = A_t\,h\,(q - Q_{nl}) - D_{turb}\,\Delta\omega - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$, dashpot fully reset $c_d = x_g$):
$c_{d,out} = 0$, $u_{pv} = p_c$, $c_{pv} = x_g = y_g = p_c$,
$q = p_c$ (from $dq/dt = 0 \Rightarrow h = 1$), and:

$$p_m = A_t\,(p_c - Q_{nl})$$

So `p_c` is the **gate position setpoint**.  With $A_t = 1$, $Q_{nl} = 0$:
$p_m = p_c$.

```{note}
Keep $G_{min} > 0$ (default 0.01) to avoid the head singularity
$h = (q/y_g)^2 \to \infty$ when the gate closes to zero.
```

## Usage

```hjson
gov: {
  type: "hygov",
  R: 0.05,    R_r: 0.3,   T_r: 5.0,
  T_f: 0.05,  T_g: 0.5,
  V_g_max: 0.2,
  G_max: 1.0, G_min: 0.01,
  T_w: 1.0,
  A_t: 1.0,   D_turb: 0.0,  Q_nl: 0.0,
  p_c: 0.8
}
```

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R` | 0.05 | pu | Permanent droop (speed regulation) |
| $R_r$ | `R_r` | 0.3 | pu | Transient (temporary) droop |
| $T_r$ | `T_r` | 5.0 | s | Dashpot reset (transient droop) time constant |
| $T_f$ | `T_f` | 0.05 | s | Pilot valve filter time constant |
| $T_g$ | `T_g` | 0.5 | s | Gate servomotor time constant |
| $V_{g,max}$ | `V_g_max` | 0.2 | pu/s | Gate velocity (rate) limit |
| $G_{max}$ | `G_max` | 1.0 | pu | Maximum gate position |
| $G_{min}$ | `G_min` | 0.01 | pu | Minimum gate position (keep > 0 to avoid head singularity) |
| $T_w$ | `T_w` | 1.0 | s | Water starting time (penstock inertia) |
| $A_t$ | `A_t` | 1.0 | pu | Turbine gain |
| $D_{turb}$ | `D_turb` | 0.0 | pu | Turbine self-damping |
| $Q_{nl}$ | `Q_nl` | 0.0 | pu | No-load water flow |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.8 | pu | Gate position setpoint (load reference). At steady state: $p_m = A_t(p_c - Q_{nl})$. |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $c_d$ | `c_d_gov` | pu | Dashpot (transient-droop) state |
| $c_{pv}$ | `c_pv_gov` | pu | Pilot-valve filter state |
| $x_g$ | `x_g_gov` | pu | Gate servomotor position state |
| $q$ | `q_gov` | pu | Water column flow |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_m$ | `p_m` | pu | Mechanical power delivered to the synchronous machine |

## Source

- Module: `pydae.bps.govs.hygov`
- File: [`packages/pydae-bps/src/pydae/bps/govs/hygov.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/hygov.py)
