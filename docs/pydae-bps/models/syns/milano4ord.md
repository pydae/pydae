# milano4ord

*Synchronous machines — pydae-bps model.*

## Model description

Synchronous machine model of order 4 (Two-Axis Model) with PSAT Saturation.

**Auxiliar equations**

$$v_d = V \sin(\delta - \theta)$$
$$v_q = V \cos(\delta - \theta)$$
$$p_e = i_d \left(v_d + R_a i_d\right) + i_q \left(v_q + R_a i_q\right)$$
$$v_{sat} = \sqrt{e'^2_q + e'^2_d + \epsilon}$$
$$S_{at} = \frac{B_{sat} \max(v_{sat} - A_{sat},\, 0)^2}{v_{sat}}$$
$$S_d = S_{at}$$
$$S_q = \frac{X_q}{X_d} S_{at}$$
$$\omega_s = \omega_{coi}$$

**Dynamic equations**

$$\frac{ d\delta}{dt} = \Omega_b \left(\omega - \omega_s \right) - K_{\delta} \delta$$
$$\frac{ d\omega}{dt} = \frac{1}{2H} \left(p_m - p_e - D \left(\omega - \omega_s \right) \right)$$
$$\frac{ de'_q}{dt} = \frac{1}{T'_{d0}} \left(-e'_q(1 + S_d) - (X_d - X'_d)i_d + v_f \right)$$
$$\frac{ de'_d}{dt} = \frac{1}{T'_{q0}} \left(-e'_d(1 + S_q) + (X_q - X'_q)i_q \right)$$

**Algebraic equations**

$$0 = e'_q - R_a i_q - X'_d i_d - v_q$$
$$0 = e'_d - R_a i_d + X'_q i_q - v_d$$
$$0 = i_d v_d + i_q v_q - p_g$$$$0 = i_d v_q - i_q v_d - q_g$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `milano4ord` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H$ | `H` | 5.0 | s | Inertia constant |
| $D$ | `D` | 1.0 | s | Damping coefficient |
| $X_d$ | `X_d` | 1.8 | pu-m | d-axis synchronous reactance |
| $X_q$ | `X_q` | 1.7 | pu-m | q-axis synchronous reactance |
| $X'_d$ | `X1d` | 0.3 | pu-m | d-axis transient reactance |
| $X'_q$ | `X1q` | 0.55 | pu-m | q-axis transient reactance |
| $T'_{d0}$ | `T1d0` | 8.0 | s | d-axis open circuit transient time constant |
| $T'_{q0}$ | `T1q0` | 0.4 | s | q-axis open circuit transient time constant |
| $R_a$ | `R_a` | 0.01 | pu-m | Armature resistance |
| $S_{1.0}$ | `S_10` | 0.0 | - | Saturation factor at 1.0 pu |
| $S_{1.2}$ | `S_12` | 0.0 | - | Saturation factor at 1.2 pu |
| $K_{\delta}$ | `K_delta` | 0.0 | - | Reference machine constant |
| $K_{sec}$ | `K_sec` | 0.0 | - | Secondary frequency control participation |

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
| $e'_q$ | `e1q` |  | pu-m | q-axis transient voltage |
| $e'_d$ | `e1d` |  | pu-m | d-axis transient voltage |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_d$ | `i_d` |  | pu-m | d-axis current |
| $i_q$ | `i_q` |  | pu-m | q-axis current |
| $p_g$ | `p_g` |  | pu-m | Active power |
| $q_g$ | `q_g` |  | pu-m | Reactive power |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_e$ | `p_e` |  | pu-m | Electrical power |
| $v_f$ | `v_f` |  | pu-m | Field voltage |
| $p_m$ | `p_m` |  | pu-m | Mechanical power |
| $S_{at}$ | `S_at` |  | - | Evaluated Saturation Factor |


## Source

- Module: `pydae.bps.syns.milano4ord`
- File: [`packages/pydae-bps/src/pydae/bps/syns/milano4ord.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/milano4ord.py)
