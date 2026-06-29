# gensal

*Synchronous machines — pydae-bps model.*

## Model description

GENSAL — salient-pole 5th-order synchronous machine (IEEE 1110 Model 2.1 /
PSS/E ``GENSAL``).

Salient-pole counterpart of :mod:`pydae.bps.syns.genrou`. Used for hydro
generators. Five states (no q-axis transient EMF — there is no field
winding on the q-axis of a salient-pole rotor):

- $\delta, \omega$ — rotor angle, electrical speed
- $e_q'$ — d-axis transient EMF
- $e_q''$, $e_d''$ — subtransient EMFs

Same convention as ``genrou``: $X_d''$, $X_q''$ are **terminal-referred**
(IEEE 115-2019 Eq. (88)). No ``X_l`` field. Saturation is applied on the
d-axis only — the salient q-axis has no field winding and is dominated
by reluctance, so $S_q = 0$ is the standard PSS/E choice.

**Auxiliar equations**

$$v_d = V \sin(\delta - \theta), \quad v_q = V \cos(\delta - \theta)$$
$$\tau_e = (v_d + R_a i_d)\,i_d + (v_q + R_a i_q)\,i_q$$
$$\psi_{AT} = \sqrt{e_q'^{\,2} + \epsilon}, \quad
  S(\psi_{AT}) = \frac{B_{sat}\,\max(\psi_{AT} - A_{sat},\,0)^2}{\psi_{AT}}$$
$$S_d = S(\psi_{AT}), \quad S_q = 0$$
$$\omega_s = \omega_{coi}$$

**Dynamic equations**

$$\frac{d\delta}{dt} = \Omega_b (\omega - \omega_s) - K_\delta \delta$$
$$\frac{d\omega}{dt} = \frac{1}{2H}\bigl(\tau_m - \tau_e - D(\omega - \omega_s)\bigr)$$
$$T_{d0}' \frac{de_q'}{dt} = -e_q' - (X_d - X_d')\,i_d - S_d\,e_q' + v_f$$
$$T_{d0}'' \frac{de_q''}{dt} = -e_q'' + e_q' - (X_d' - X_d'')\,i_d$$
$$T_{q0}'' \frac{de_d''}{dt} = -e_d'' + (X_q - X_q'')\,i_q$$

The q-axis subtransient is driven directly from $i_q$ (single damper
winding; no transient stage), which is the structural change vs
``genrou``. The d-axis sub-stack is unchanged.

**Algebraic equations** (terminal-referred subtransient, identical to
``genrou``):

$$0 = v_q + R_a i_q - e_q'' + X_d''\,i_d$$
$$0 = v_d + R_a i_d - e_d'' - X_q''\,i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

**Choosing between gensal and genrou.** Pick ``gensal`` for hydro units
(salient poles, $X_q < X_d$, no q-axis field) and ``genrou`` for turbo
units (round rotor, $X_q \approx X_d$, q-axis field present). PSS/E
GENSAL data tables drop straight in: ``X_d, X_q, X1d, X2d, X2q, T1d0,
T2d0, T2q0, S_10, S_12, H, D, R_a``.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `gensal` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H$ | `H` | 4.0 | s | Inertia constant |
| $D$ | `D` | 0.0 | - | Damping coefficient |
| $X_d$ | `X_d` | 1.1 | pu-m | d-axis synchronous reactance |
| $X_q$ | `X_q` | 0.7 | pu-m | q-axis synchronous reactance (salient: X_q < X_d) |
| $X'_d$ | `X1d` | 0.25 | pu-m | d-axis transient reactance |
| $X''_d$ | `X2d` | 0.2 | pu-m | d-axis subtransient reactance (terminal-referred) |
| $X''_q$ | `X2q` | 0.2 | pu-m | q-axis subtransient reactance (terminal-referred) |
| $T'_{d0}$ | `T1d0` | 8.0 | s | d-axis open-circuit transient time constant |
| $T''_{d0}$ | `T2d0` | 0.03 | s | d-axis open-circuit subtransient time constant |
| $T''_{q0}$ | `T2q0` | 0.05 | s | q-axis open-circuit subtransient time constant |
| $R_a$ | `R_a` | 0.0 | pu-m | Armature resistance |
| $S_{1.0}$ | `S_10` | 0.0 | - | Saturation factor at E=1.0 |
| $S_{1.2}$ | `S_12` | 0.0 | - | Saturation factor at E=1.2 |
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
| $e'_q$ | `e1q` |  | pu-m | q-axis transient EMF (d-axis flux) |
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

- Module: `pydae.bps.syns.gensal`
- File: [`packages/pydae-bps/src/pydae/bps/syns/gensal.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/gensal.py)
