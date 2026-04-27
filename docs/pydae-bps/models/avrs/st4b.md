# st4b

*Automatic voltage regulators — pydae-bps model.*

## Model description

IEEE Std 421.5 Type ST4B static excitation system with PV-bus initialisation.

Dual-PI structure: an outer voltage-error loop drives an inter-stage lag, which
feeds an inner field-current loop.  Used in Spanish REE NTS network studies
with a simplified parameter set ($K_G = 0$, $X_L = 0$, $K_I = 0$,
$\theta_P = 0$) that reduces to a proportional exciter voltage calculation.

**Signal path**

Terminal voltage sensor:

$$\frac{d v_c}{dt} = \frac{V - v_c}{T_R}$$

Outer voltage-regulator PI (leakage $\varepsilon_{leak} = 10^{-6}$):

$$\frac{d \xi_r}{dt} = K_{IR}\,(v^\star - v_c + v_s) - \varepsilon_{leak}\,\xi_r$$

$$v_r^{nosat} = K_{PR}\,(v^\star - v_c + v_s) + \xi_r, \qquad
  v_r = \mathrm{sat}(v_r^{nosat}, V_{RMIN}, V_{RMAX})$$

Inter-stage lag:

$$\frac{d x_a}{dt} = \frac{v_r - x_a}{T_A}$$

Inner field-current regulator PI ($\varepsilon_m = x_a - K_G v_f$,
defaults to $x_a$ when $K_G = 0$):

$$\frac{d \xi_m}{dt} = K_{IM}\,\varepsilon_m - \varepsilon_{leak}\,\xi_m$$

$$v_m^{nosat} = K_{PM}\,\varepsilon_m + \xi_m, \qquad
  v_m = \mathrm{sat}(v_m^{nosat}, V_{MMIN}, V_{MMAX})$$

Exciter voltage (REE simplified: $V_E = K_P V$, $F_{EX} = 1$):

$$e_{fd}^{nosat} = v_m\,K_P\,V, \qquad
  0 = \min(V_{BMAX},\; e_{fd}^{nosat}) - v_f$$

**The ini/run variable swap**

Same PV-bus swap used by all AVR models:

|          | ini | run |
|----------|-----|-----|
| `v_f`    | $y_{ini}$ | $y_{run}$ (algebraic, always solved) |
| `v_ref`  | $y_{ini}$ | $u_{run}$ (unknown in ini, input in run) |
| `V_bus`  | $u_{ini}$ | $y_{run}$ (pinned in ini, solved in run) |

## Usage

```hjson
avr: {
  type: "st4b",
  T_R: 0.02,
  K_PR: 3.15,   K_IR: 3.15,
  V_RMAX: 1.0,  V_RMIN: -0.87,
  T_A: 0.02,
  K_PM: 1.0,    K_IM: 0.0,
  V_MMAX: 1.0,  V_MMIN: -0.87,
  K_G: 0.0,
  K_P: 6.5,
  V_BMAX: 8.0,
  v_ref: 1.0
}
```

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_R$ | `T_R` | 0.02 | s | Terminal-voltage sensor time constant |
| $K_{PR}$ | `K_PR` | 3.15 | pu | Outer voltage-regulator proportional gain |
| $K_{IR}$ | `K_IR` | 3.15 | pu/s | Outer voltage-regulator integral gain |
| $V_{RMAX}$ | `V_RMAX` | 1.0 | pu | Upper limit on outer-regulator output |
| $V_{RMIN}$ | `V_RMIN` | -0.87 | pu | Lower limit on outer-regulator output |
| $T_A$ | `T_A` | 0.02 | s | Inter-stage lag time constant |
| $K_{PM}$ | `K_PM` | 1.0 | pu | Inner field-current regulator proportional gain |
| $K_{IM}$ | `K_IM` | 0.0 | pu/s | Inner field-current regulator integral gain (REE default: 0) |
| $V_{MMAX}$ | `V_MMAX` | 1.0 | pu | Upper limit on inner-regulator output |
| $V_{MMIN}$ | `V_MMIN` | -0.87 | pu | Lower limit on inner-regulator output |
| $K_G$ | `K_G` | 0.0 | pu | Field-voltage feedback gain into inner loop (REE default: 0) |
| $K_P$ | `K_P` | 6.5 | pu | Exciter voltage gain ($V_E = K_P V$) |
| $V_{BMAX}$ | `V_BMAX` | 8.0 | pu | Rectifier ceiling on field-voltage command |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^\star$ | `v_ref` | 1.0 | pu | Voltage reference. PV-bus setpoint during ini (solved for); input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output). |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $v_c$ | `v_c` | pu | Sensed terminal voltage ($T_R$ lag output) |
| $\xi_r$ | `xi_r` | pu | Outer-PI integrator state |
| $x_a$ | `x_a` | pu | Inter-stage lag output |
| $\xi_m$ | `xi_m` | pu | Inner-PI integrator state (pinned ≈ 0 when $K_{IM} = 0$) |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $v_f$ | `v_f` | pu | Field-voltage command sent to the synchronous machine |

## Source

- Module: `pydae.bps.avrs.st4b`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/st4b.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/st4b.py)
