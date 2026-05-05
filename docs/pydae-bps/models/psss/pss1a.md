# pss1a

*Power system stabilizers — pydae-bps model.*

## Model description

IEEE PSS1A power system stabilizer (single-input, two lead-lag).

Matches the IEEE Std 421.5 PSS1A topology: speed deviation (or power)
passes through a washout, two series lead-lag compensators, a gain stage,
and a hard output limiter.  An optional output low-pass filter with time
constant $T_6$ is supported but omitted when $T_6 = 0$ (most common case).

**Signal path**

Speed deviation:

$$\Delta\omega = \omega - 1$$

Washout $s T_5 / (1 + s T_5)$ with state $x_{wo}$:

$$\frac{d x_{wo}}{dt} = \frac{\Delta\omega - x_{wo}}{T_5}, \qquad
  z_{wo} = \Delta\omega - x_{wo}$$

First lead-lag $(1 + T_1 s)/(1 + T_2 s)$ with state $x_{12}$:

$$\frac{d x_{12}}{dt} = \frac{z_{wo} - x_{12}}{T_2}, \qquad
  z_{12} = (z_{wo} - x_{12})\frac{T_1}{T_2} + x_{12}$$

Second lead-lag $(1 + T_3 s)/(1 + T_4 s)$ with state $x_{34}$:

$$\frac{d x_{34}}{dt} = \frac{z_{12} - x_{34}}{T_4}, \qquad
  z_{34} = (z_{12} - x_{34})\frac{T_3}{T_4} + x_{34}$$

Gain and hard limits:

$$0 = \mathrm{sat}\!\left(K_s\, z_{34},\; V_{stmin},\; V_{stmax}\right) - v_{pss}$$

The output filter ($T_6 > 0$) is not yet modelled as a separate dynamic
state; set $T_6 = 0$ (default) to reproduce the standard two-lag PSS1A
behaviour used in most benchmark studies.

**Steady-state**

At $\omega = 1$: all states and $v_{pss}$ are zero.

**Configuration**

Example data entry (IEEE PES TR18 3MIB benchmark)::

    "pss": {"type": "pss1a",
            "K_s": 1.0, "T_5": 10.0,
            "T_1": 0.2, "T_2": 0.05,
            "T_3": 0.2, "T_4": 0.05,
            "T_6": 0.0,
            "V_stmax": 0.1, "V_stmin": -0.1}

Parameter names follow IEEE 421.5 Table D.15.  The ``pss_kundur_2``
model is structurally identical but uses the Kundur textbook names
(``K_stab``, ``T_w``, symmetric ``V_lim``).

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pss1a` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_s$ | `K_s_pss` | 1.0 | pu | Main stabiliser gain |
| $T_5$ | `T_5_pss` | 10.0 | s | Washout time constant |
| $T_1$ | `T_1_pss` | 0.2 | s | Lead-lag 1 numerator time constant |
| $T_2$ | `T_2_pss` | 0.05 | s | Lead-lag 1 denominator time constant |
| $T_3$ | `T_3_pss` | 0.2 | s | Lead-lag 2 numerator time constant |
| $T_4$ | `T_4_pss` | 0.05 | s | Lead-lag 2 denominator time constant |
| $V_{stmax}$ | `V_stmax_pss` | 0.1 | pu | Upper PSS output limit |
| $V_{stmin}$ | `V_stmin_pss` | -0.1 | pu | Lower PSS output limit |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_{wo}$ | `x_wo_pss` |  | pu | Washout state |
| $x_{12}$ | `x_12_pss` |  | pu | Lead-lag 1 internal state |
| $x_{34}$ | `x_34_pss` |  | pu | Lead-lag 2 internal state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{pss}$ | `v_pss` |  | pu | PSS output sent to the AVR summing junction |


## Source

- Module: `pydae.bps.psss.pss1a`
- File: [`packages/pydae-bps/src/pydae/bps/psss/pss1a.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/psss/pss1a.py)
