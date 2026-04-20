# pss_kundur_2

*Power system stabilizers — pydae-bps model.*

## Model description

Kundur two-stage power system stabilizer (Figure E12.9).

The canonical textbook PSS from Kundur's *Power System Stability and
Control*, Figure E12.9: speed deviation is passed through a washout,
two series lead-lag compensators, scaled by the stabiliser gain, and
saturated at the output. The two compensators provide the phase lead
needed over the ~0.1–2 Hz range of electromechanical modes.

**Signal path**

The input is the speed deviation from synchronism,

$$\Delta\omega = \omega - \omega_{ref}$$

with $\omega_{ref} = 1$ pu. It passes through a washout
$sT_w/(1 + sT_w)$ with state $x_{wo}$,

$$\frac{d x_{wo}}{dt} = \frac{\Delta\omega - x_{wo}}{T_w},\quad
  z_{wo} = \Delta\omega - x_{wo}$$

followed by two lead-lag compensators in series,

$$\frac{d x_{12}}{dt} = \frac{z_{wo} - x_{12}}{T_2},\quad
  z_{12} = (z_{wo} - x_{12})\frac{T_1}{T_2} + x_{12}$$
$$\frac{d x_{34}}{dt} = \frac{z_{12} - x_{34}}{T_4},\quad
  z_{34} = (z_{12} - x_{34})\frac{T_3}{T_4} + x_{34}$$

The main gain is applied to the final compensator output and the
result is clipped by the hard output limits,

$$0 = \mathrm{sat}(K_{stab}\, z_{34},\; V_{Smin},\; V_{Smax}) - v_{pss}$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**No ini/run swap**

The PSS output is an algebraic variable $v_{pss}$ that feeds the AVR
summing junction. The PSS does not pin a voltage setpoint, so the
``y_ini``/``y_run`` partitions are identical. At steady state
($\omega = 1$) every state settles to zero and the PSS output is
zero — the stabiliser only responds to transients.

**Configuration**

Example data entry:

    "pss": {"type": "pss_kundur_2",
            "K_stab": 20.0,
            "T_w": 10.0,
            "T_1": 0.05, "T_2": 0.02,
            "T_3": 3.0,  "T_4": 5.4,
            "V_Smax": 0.1, "V_Smin": -0.1}

Reference: Kundur, *Power System Stability and Control*, Example 12.6
(Figure E12.9).

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pss_kundur_2` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_{stab}$ | `K_stab` | 20.0 | pu | Main stabiliser gain |
| $T_w$ | `T_w` | 10.0 | s | Washout time constant |
| $T_1$ | `T_1` | 0.05 | s | Lead-lag 1 numerator time constant |
| $T_2$ | `T_2` | 0.02 | s | Lead-lag 1 denominator time constant |
| $T_3$ | `T_3` | 3.0 | s | Lead-lag 2 numerator time constant |
| $T_4$ | `T_4` | 5.4 | s | Lead-lag 2 denominator time constant |
| $V_{Smax}$ | `V_Smax` | 0.1 | pu | Upper PSS output limit |
| $V_{Smin}$ | `V_Smin` | -0.1 | pu | Lower PSS output limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\omega$ | `omega` | 1.0 | pu | Machine rotor speed (from the synchronous machine). Used as ω - 1. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_{wo}$ | `x_wo_pss` |  | pu | Washout state |
| $x_{12}$ | `x_12_pss` |  | pu | Lead-lag 1 internal state |
| $x_{34}$ | `x_34_pss` |  | pu | Lead-lag 2 internal state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{pss}$ | `v_pss` |  | pu | PSS output signal added to the AVR summing junction (saturated). |

## Source

- Module: `pydae.bps.psss.pss_kundur_2`
- File: [`packages/pydae-bps/src/pydae/bps/psss/pss_kundur_2.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/psss/pss_kundur_2.py)
