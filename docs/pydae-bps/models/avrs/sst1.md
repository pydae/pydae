# sst1

*Automatic voltage regulators — pydae-bps model.*

## Model description

ST1-style static excitation system with lead-lag regulator and output filter.

A three-state AVR that closely follows the SEXS simplified topology but
replaces the explicit exciter state with an output filter on $v_f$ and uses
the same time constant $T_r$ for both the sensor lag and the output filter.
Intended as a simpler, numerically stable alternative to NTSST1 for
studies where the output should be filtered.

**Signal path**

Terminal voltage sensor:

$$\frac{d v_r}{dt} = \frac{V - v_r}{T_r}$$

Voltage error and lead-lag $(1 + T_c\,s)/(1 + T_b\,s)$ with state $x_{cb}$:

$$v_1 = v^\star - V + v_s, \qquad
  \frac{d x_{cb}}{dt} = \frac{v_1 - x_{cb}}{T_b}$$

$$z_{cb} = (v_1 - x_{cb})\,\frac{T_c}{T_b} + x_{cb}$$

Proportional gain and hard limiter:

$$e_{fd}^{nosat} = K_a\,z_{cb}, \qquad
  e_{fd} = \mathrm{sat}\!\left(e_{fd}^{nosat},\; V_{fmin},\; V_{fmax}\right)$$

Output filter (same $T_r$ as the sensor):

$$\frac{d v_f}{dt} = \frac{e_{fd} - v_f}{T_r}$$

Note that $v_f$ is a **dynamic state** here, not an algebraic variable.

**The ini/run variable swap**

Same PV-bus swap used by all AVR models:

|          | ini | run |
|----------|-----|-----|
| `v_f`    | — (dynamic state) | — (dynamic state) |
| `v_ref`  | $y_{ini}$ | $u_{run}$ (unknown in ini, input in run) |
| `V_bus`  | $u_{ini}$ | $y_{run}$ (pinned in ini, solved in run) |

## Usage

```hjson
avr: {
  type: "sst1",
  K_a: 200.0,
  T_r: 0.02,
  T_b: 10.0,  T_c: 1.0,
  V_f_min: -100.0,  V_f_max: 100.0,
  v_ref: 1.0
}
```

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_a$ | `K_a` | 200.0 | pu | AVR main proportional gain |
| $T_r$ | `T_r` | 0.02 | s | Sensor and output-filter time constant (shared) |
| $T_b$ | `T_b` | 10.0 | s | Lead-lag denominator time constant |
| $T_c$ | `T_c` | 1.0 | s | Lead-lag numerator time constant |
| $V_{fmin}$ | `V_f_min` | -100.0 | pu | Lower field-voltage limit |
| $V_{fmax}$ | `V_f_max` | 100.0 | pu | Upper field-voltage limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^\star$ | `v_ref` | 1.0 | pu | Voltage reference. PV-bus setpoint during ini (solved for); input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output). |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $v_r$ | `v_r` | pu | Sensed terminal voltage ($T_r$ lag, diagnostic) |
| $x_{cb}$ | `x_cb` | pu | Lead-lag internal state |
| $v_f$ | `v_f` | pu | Filtered field-voltage command (output state) |

## Source

- Module: `pydae.bps.avrs.sst1`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/sst1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/sst1.py)
