# pss2a

*Power system stabilizers — pydae-bps model.*

## Model description

IEEE PSS2A dual-input power system stabilizer (REE NTS parameter set).

The PSS2A model is the IEEE 421.5 Type PSS2A dual-input stabilizer:
speed deviation is combined with an accelerating-power proxy derived
from electrical power to produce a voltage-regulator modulation
signal $v_{pss}$. The dual-input design cancels the synchronizing
torque component of the power signal so that only the damping
component remains, avoiding interaction with turbine torsional
modes.

**Signal path**

The speed-channel signal is the speed deviation from synchronism,

$$V_1 = \omega - \omega_{ref}$$

with $\omega_{ref} = 1$ pu. It passes through two cascaded washouts,

$$\frac{d x_{w1}}{dt} = \frac{V_1 - x_{w1}}{T_{w1}},\quad z_{w1} = V_1 - x_{w1}$$
$$\frac{d x_{w2}}{dt} = \frac{z_{w1} - x_{w2}}{T_{w2}},\quad z_{w2} = z_{w1} - x_{w2}$$

The transducer lag $1/(1 + s T_6)$ is a unity pass-through under the
REE spec ($T_6 = 0$), so $y_1 = z_{w2}$.

The power-channel signal $V_2 = p_g$ passes through a single washout
($T_{w4} = 0$ collapses the second),

$$\frac{d x_{w3}}{dt} = \frac{V_2 - x_{w3}}{T_{w3}},\quad z_{w3} = V_2 - x_{w3}$$

followed by the accelerating-power scale $K_{s2}$ and a lag with time
constant $T_7$ (typically chosen so that $K_{s2} T_7 = 1/(2H)$ for
cancellation of the synchronizing torque),

$$\frac{d x_p}{dt} = \frac{K_{s2}\, z_{w3} - x_p}{T_7},\quad y_3 = x_p$$

The ramp-tracking filter input combines the speed channel with the
weighted power proxy,

$$u_{rt} = y_1 + K_{s3}\, y_3$$

and the filter has transfer function
$F(s) = \left[(1 + s T_8)/(1 + s T_9)\right]^M \cdot 1/(1 + s T_9)^N$.
Under the REE spec $T_8 = 0$, $M = 5$, $N = 1$, so
$F(s) = 1/(1 + s T_9)^6$ — six cascaded first-order lags with state
variables $x_{r1} \dots x_{r6}$ and output $y_{rt} = x_{r6}$.

The dual-input cancellation applies the main gain,

$$u_{ll} = K_{s1}\,(y_{rt} - y_3)$$

followed by two series lead-lag compensators,

$$\frac{d x_{l1}}{dt} = \frac{u_{ll} - x_{l1}}{T_2},\quad
  z_{l1} = (u_{ll} - x_{l1})\frac{T_1}{T_2} + x_{l1}$$
$$\frac{d x_{l2}}{dt} = \frac{z_{l1} - x_{l2}}{T_4},\quad
  z_{l2} = (z_{l1} - x_{l2})\frac{T_3}{T_4} + x_{l2}$$

The output is saturated by the hard output limits,

$$0 = \mathrm{sat}(z_{l2},\; V_{STmin},\; V_{STmax}) - v_{pss}$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**Simplifications under the REE parameter set**

- $T_6 = 0$ — speed transducer is a unity pass-through; no state.
- $T_{w4} = 0$ — second power washout collapses; only $T_{w3}$ active.
- $T_8 = 0$, $M = 5$, $N = 1$ — ramp-tracker reduces to six cascaded
  lags $1/(1 + s T_9)^6$. Generalising to $T_8 \neq 0$ would require
  replacing each lag with a lead-lag, which is out of scope here.

**No ini/run swap**

The PSS output is an algebraic variable $v_{pss}$ that feeds the AVR
summing junction. The PSS does not pin a voltage setpoint, so the
``y_ini``/``y_run`` partitions are identical. At steady state
($\omega = 1$, $p_g$ constant) every state settles to zero and the
PSS output is zero — the stabiliser only responds to transients.

**Configuration**

Example data entry (REE NTS defaults)::

    "pss": {"type": "pss2a",
            "T_w1": 2.0, "T_w2": 2.0, "T_w3": 2.0, "T_w4": 0.0,
            "T_6": 0.0, "T_7": 2.0, "T_8": 0.0, "T_9": 0.1,
            "K_s1": 17.069, "K_s2": 0.158, "K_s3": 1.0,
            "T_1": 0.28, "T_2": 0.04, "T_3": 0.28, "T_4": 0.12,
            "V_STmax": 0.1, "V_STmin": -0.1,
            "M": 5, "N": 1}

``M`` and ``N`` are recorded in the parameter dict for documentation
but do not enter the symbolic equations — the ramp-tracker order is
baked in at six lags. Change the structure if a different order is
required.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pss2a` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{w1}$ | `T_w1_pss` | 2.0 | s | First speed washout time constant |
| $T_{w2}$ | `T_w2_pss` | 2.0 | s | Second speed washout time constant |
| $T_{w3}$ | `T_w3_pss` | 2.0 | s | First power washout time constant |
| $T_7$ | `T_7_pss` | 2.0 | s | Power-channel lag (chosen so that K_s2 T_7 ≈ 1/(2H) for synchronizing torque cancellation) |
| $T_9$ | `T_9_pss` | 0.1 | s | Ramp-tracker lag time constant |
| $K_{s1}$ | `K_s1_pss` | 17.069 | pu | Main stabiliser gain |
| $K_{s2}$ | `K_s2_pss` | 0.158 | pu | Power-channel gain |
| $K_{s3}$ | `K_s3_pss` | 1.0 | pu | Cross-path gain at ramp-tracker input |
| $T_1$ | `T_1_pss` | 0.28 | s | Lead-lag 1 numerator time constant |
| $T_2$ | `T_2_pss` | 0.04 | s | Lead-lag 1 denominator time constant |
| $T_3$ | `T_3_pss` | 0.28 | s | Lead-lag 2 numerator time constant |
| $T_4$ | `T_4_pss` | 0.12 | s | Lead-lag 2 denominator time constant |
| $V_{STmax}$ | `V_STmax_pss` | 0.1 | pu | Upper PSS output limit |
| $V_{STmin}$ | `V_STmin_pss` | -0.1 | pu | Lower PSS output limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\omega$ | `omega` | 1.0 | pu | Machine rotor speed (taken from the synchronous machine). Used as ω - 1 in the first channel. |
| $p_g$ | `p_g` | 0.0 | pu | Generator electrical power (taken from the synchronous machine's algebraic output). Second channel input. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_{w1}$ | `x_w1_pss` |  | pu | First speed washout state |
| $x_{w2}$ | `x_w2_pss` |  | pu | Second speed washout state |
| $x_{w3}$ | `x_w3_pss` |  | pu | Power washout state |
| $x_p$ | `x_p_pss` |  | pu | Power-channel lag state (accelerating-power proxy) |
| $x_{r1}$ | `x_r1_pss` |  | pu | Ramp-tracker lag state 1/6 |
| $x_{r2}$ | `x_r2_pss` |  | pu | Ramp-tracker lag state 2/6 |
| $x_{r3}$ | `x_r3_pss` |  | pu | Ramp-tracker lag state 3/6 |
| $x_{r4}$ | `x_r4_pss` |  | pu | Ramp-tracker lag state 4/6 |
| $x_{r5}$ | `x_r5_pss` |  | pu | Ramp-tracker lag state 5/6 |
| $x_{r6}$ | `x_r6_pss` |  | pu | Ramp-tracker lag state 6/6 |
| $x_{l1}$ | `x_l1_pss` |  | pu | Lead-lag 1 internal state |
| $x_{l2}$ | `x_l2_pss` |  | pu | Lead-lag 2 internal state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{pss}$ | `v_pss` |  | pu | PSS output signal added to the AVR summing junction (saturated). |


## Source

- Module: `pydae.bps.psss.pss2a`
- File: [`packages/pydae-bps/src/pydae/bps/psss/pss2a.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/psss/pss2a.py)
