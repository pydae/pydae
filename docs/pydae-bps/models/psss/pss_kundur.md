# pss_kundur

*Power system stabilizers — pydae-bps model.*

## Model description

Kundur single-stage power system stabilizer.

The simplest form of Kundur's PSS: speed deviation is passed through
a washout and a single lead-lag compensator, scaled by the main
stabiliser gain, and saturated at the output. The model is the
pedagogical starting point for Kundur Chapter 12 — a two-state
supplementary controller that adds damping torque through the AVR's
summing junction.

**Signal path**

The input is the speed deviation from synchronism,

$$\Delta\omega = \omega - \omega_{ref}$$

with $\omega_{ref} = 1$ pu. It passes through a washout
$sT_w/(1 + sT_w)$ with state $x_{wo}$,

$$\frac{d x_{wo}}{dt} = \frac{\Delta\omega - x_{wo}}{T_w},\quad
  z_{wo} = \Delta\omega - x_{wo}$$

and a single lead-lag $(1 + s T_1)/(1 + s T_2)$ with state $x_{lead}$,

$$\frac{d x_{lead}}{dt} = \frac{z_{wo} - x_{lead}}{T_2},\quad
  z_{lead} = (z_{wo} - x_{lead})\frac{T_1}{T_2} + x_{lead}$$

The main gain is applied to the compensator output and the result is
clipped by the hard output limits,

$$0 = \mathrm{sat}(K_{stab}\, z_{lead},\; V_{Smin},\; V_{Smax}) - v_{pss}$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**No ini/run swap**

The PSS output is an algebraic variable $v_{pss}$ that feeds the AVR
summing junction. The PSS does not pin a voltage setpoint, so the
``y_ini``/``y_run`` partitions are identical. At steady state
($\omega = 1$) every state settles to zero and the PSS output is
zero — the stabiliser only responds to transients.

**Configuration**

Example data entry:

    "pss": {"type": "pss_kundur",
            "K_stab": 20.0,
            "T_w": 10.0, "T_1": 0.05, "T_2": 0.02,
            "V_Smax": 0.1, "V_Smin": -0.1}

Reference: Kundur, *Power System Stability and Control*, Fig. 12.16
(single-stage PSS).

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pss_kundur` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_{stab}$ | `K_stab` | 20.0 | pu | Main stabiliser gain |
| $T_w$ | `T_w` | 10.0 | s | Washout time constant |
| $T_1$ | `T_1` | 0.05 | s | Lead-lag numerator time constant |
| $T_2$ | `T_2` | 0.02 | s | Lead-lag denominator time constant |
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
| $x_{lead}$ | `x_lead_pss` |  | pu | Lead-lag internal state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{pss}$ | `v_pss` |  | pu | PSS output signal added to the AVR summing junction (saturated). |


## Source

- Module: `pydae.bps.psss.pss_kundur`
- File: [`packages/pydae-bps/src/pydae/bps/psss/pss_kundur.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/psss/pss_kundur.py)
