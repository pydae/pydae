# genroe

*Synchronous machines — pydae-bps model.*

## Model description

GENROE — round-rotor 6th-order synchronous machine, PSS/E ``GENROE`` variant.

Identical to :mod:`pydae.bps.syns.genrou` (IEEE 1110 Model 2.2 / Anderson-Fouad,
terminal-referred subtransient — no ``X_l`` parameter) **except for the
saturation function**.

``genrou`` (PSS/E ``GENROU``) uses the *quadratic-offset* form,

$$S(E) = \frac{B_{sat}\,\max(E - A_{sat},\,0)^2}{E}.$$

``genroe`` (PSS/E ``GENROE``) uses the *geometric / exponential* form documented
in the PSS/E Program Operation Manual (model ``GENROE``):

$$S(E) = B_{sat}\, E^{\,N_{sat}}.$$

Both forms are fit to the two saturation points $S(1.0) = S_{10}$ and
$S(1.2) = S_{12}$ from the manufacturer data sheet. For ``GENROE``,

$$N_{sat} = \frac{\ln\!\bigl(S_{12}/S_{10}\bigr)}{\ln 1.2},\qquad
  B_{sat} = S_{10}.$$

So $S(1.0) = S_{10}$ and $S(1.2) = S_{10}\,(1.2)^{N_{sat}} = S_{12}$ by
construction, irrespective of $N_{sat}$.

**When to use GENROE vs GENROU.** GENROE is the saturation form used by the
PES-TR18 benchmark systems (§5.3 "two-area 4-generator" and §5.2 "IEEE 39-bus"
both quote it explicitly: *"saturation function represented as a geometric
(exponential) function"*). If your data source is PSS/E ``GENROE`` records or
PES-TR18 dynamic data tables, use this model. If it quotes ``GENROU``, use
``genrou`` instead.

When saturation is off ($S_{10} = S_{12} = 0$) the two models are identical.

All other dynamic, algebraic, and steady-state behaviour follows
``genrou`` — refer to that module's docstring for the equations.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `genroe` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H$ | `H` | 5.0 | s | Inertia constant |
| $D$ | `D` | 0.0 | - | Damping coefficient |
| $X_d$ | `X_d` | 1.8 | pu-m | d-axis synchronous reactance |
| $X_q$ | `X_q` | 1.7 | pu-m | q-axis synchronous reactance |
| $X'_d$ | `X1d` | 0.3 | pu-m | d-axis transient reactance |
| $X'_q$ | `X1q` | 0.55 | pu-m | q-axis transient reactance |
| $X''_d$ | `X2d` | 0.2 | pu-m | d-axis subtransient reactance (terminal-referred) |
| $X''_q$ | `X2q` | 0.25 | pu-m | q-axis subtransient reactance (terminal-referred) |
| $T'_{d0}$ | `T1d0` | 8.0 | s | d-axis open-circuit transient time constant |
| $T'_{q0}$ | `T1q0` | 0.4 | s | q-axis open-circuit transient time constant |
| $T''_{d0}$ | `T2d0` | 0.03 | s | d-axis open-circuit subtransient time constant |
| $T''_{q0}$ | `T2q0` | 0.05 | s | q-axis open-circuit subtransient time constant |
| $R_a$ | `R_a` | 0.0 | pu-m | Armature resistance |
| $S_{1.0}$ | `S_10` | 0.0 | - | Saturation factor at E=1.0 (geometric form) |
| $S_{1.2}$ | `S_12` | 0.0 | - | Saturation factor at E=1.2 (geometric form) |
| $K_{\delta}$ | `K_delta` | 0.0 | - | Reference-machine constant |
| $K_{sec}$ | `K_sec` | 0.0 | - | Secondary-frequency control participation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` | 0.5 | pu-m | Mechanical power |
| $v_f$ | `v_f` | 1.0 | pu-m | Field voltage |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\delta$ | `delta` |  | rad | Rotor angle |
| $\omega$ | `omega` |  | pu | Rotor speed |
| $e'_q$ | `e1q` |  | pu-m | q-axis transient EMF |
| $e'_d$ | `e1d` |  | pu-m | d-axis transient EMF |
| $e''_q$ | `e2q` |  | pu-m | q-axis subtransient EMF |
| $e''_d$ | `e2d` |  | pu-m | d-axis subtransient EMF |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_d$ | `i_d` |  | pu-m | d-axis current |
| $i_q$ | `i_q` |  | pu-m | q-axis current |
| $p_g$ | `p_g` |  | pu-m | Active power injection (machine base) |
| $q_g$ | `q_g` |  | pu-m | Reactive power injection (machine base) |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_e$ | `p_e` |  | pu-m | Electrical power |
| $v_f$ | `v_f` |  | pu-m | Field voltage (echo) |
| $v_d$ | `v_d` |  | pu-m | d-axis terminal voltage |
| $v_q$ | `v_q` |  | pu-m | q-axis terminal voltage |
| $S_{at}$ | `S_at` |  | - | Saturation factor S(psi_AT) |


## Source

- Module: `pydae.bps.syns.genroe`
- File: [`packages/pydae-bps/src/pydae/bps/syns/genroe.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/genroe.py)
