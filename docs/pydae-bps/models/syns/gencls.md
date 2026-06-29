# gencls

*Synchronous machines — pydae-bps model.*

## Model description

GENCLS — classical 2nd-order synchronous machine (PSS/E ``GENCLS``).

The simplest useful machine model: just rotor swing dynamics with a
**constant internal voltage** $E$ behind a **single transient
reactance** $X'$. No field winding, no AVR, no damper windings, no
saturation, no saliency.

Used as a "back-of-the-envelope" machine for first-swing transient
stability studies, for the *infinite-machine equivalent* of distant
generation, and for benchmarks where rotor flux dynamics are not
relevant. Two states: $\delta$, $\omega$. The internal voltage $E$
(stored under the name ``e1q``) is an external input — typically
pinned to the magnitude obtained from a power-flow solution and held
fixed throughout the simulation.

Choose ``gencls`` when you want only the swing equation. Choose
``gensal`` for hydro and ``genrou`` for thermal/turbo where rotor flux
dynamics matter.

**Auxiliar equations**

$$v_d = V \sin(\delta - \theta), \quad v_q = V \cos(\delta - \theta)$$
$$\tau_e = (v_d + R_a i_d)\,i_d + (v_q + R_a i_q)\,i_q$$
$$\omega_s = \omega_{coi}$$

**Dynamic equations**

$$\frac{d\delta}{dt} = \Omega_b (\omega - \omega_s) - K_\delta \delta$$
$$\frac{d\omega}{dt} = \frac{1}{2H}\bigl(\tau_m - \tau_e - D(\omega - \omega_s)\bigr)$$

**Algebraic equations** — constant internal voltage $E = e_q^{(1)}$
behind a single transient reactance $X'$ (no saliency, $X_d' = X_q' = X'$):

$$0 = v_q + R_a i_q - E + X'\,i_d$$
$$0 = v_d + R_a i_d - X'\,i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

**Compared with ``milano2ord``** — `milano2ord` lets you set distinct
$X'_d$ and $X'_q$ (i.e. it can model salient-pole transient saliency
even at the classical-model level). `gencls` is the strict PSS/E
`GENCLS` form with a single $X'$. When $X_q' = X_d'$, `milano2ord`
collapses onto `gencls` exactly.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `gencls` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H$ | `H` | 5.0 | s | Inertia constant |
| $D$ | `D` | 0.0 | - | Damping coefficient |
| $X'$ | `X1d` | 0.3 | pu-m | Single transient reactance (X'_d = X'_q = X') |
| $R_a$ | `R_a` | 0.0 | pu-m | Armature resistance |
| $K_{\delta}$ | `K_delta` | 0.0 | - | Reference-machine constant |
| $K_{sec}$ | `K_sec` | 0.0 | - | Secondary-frequency control participation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` | 0.5 | pu-m | Mechanical power |
| $E$ | `e1q` | 1.0 | pu-m | Internal voltage magnitude (constant) |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\delta$ | `delta` |  | rad | Rotor angle |
| $\omega$ | `omega` |  | pu | Rotor speed |

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
| $v_d$ | `v_d` |  | pu-m | d-axis terminal voltage |
| $v_q$ | `v_q` |  | pu-m | q-axis terminal voltage |


## Source

- Module: `pydae.bps.syns.gencls`
- File: [`packages/pydae-bps/src/pydae/bps/syns/gencls.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/gencls.py)
