# milano2ord

*Synchronous machines — pydae-bps model.*

## Model description

Created on Thu August 10 23:52:55 2026

@author: jmmauricio

Synchronous machine model of order 2 (Classical Model).

**Auxiliar equations**

$$v_d = V \sin(\delta - \theta)$$
$$v_q = V \cos(\delta - \theta)$$
$$p_e = i_d \left(v_d + R_a i_d\right) + i_q \left(v_q + R_a i_q\right)$$
$$\omega_s = \omega_{coi}$$

**Dynamic equations**

$$\frac{ d\delta}{dt} = \Omega_b \left(\omega - \omega_s \right) - K_{\delta} \delta$$
$$\frac{ d\omega}{dt} = \frac{1}{2H} \left(p_m - p_e - D \left(\omega - \omega_s \right) \right)$$

**Algebraic equations**

$$0 = v_q + R_a i_q + X'_d i_d - e'_q$$
$$0 = v_d + R_a i_d - X'_q i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `milano2ord` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H$ | `H` | 5.0 | s | Inertia constant |
| $D$ | `D` | 1.0 | s | Damping coefficient |
| $X'_q$ | `X1q` | 0.55 | pu-m | q-axis transient reactance |
| $X'_d$ | `X1d` | 0.3 | pu-m | d-axis transient reactance |
| $R_a$ | `R_a` | 0.01 | pu-m | Armature resistance |
| $K_{\delta}$ | `K_delta` | 0.0 | - | Reference machine constant |
| $K_{sec}$ | `K_sec` | 0.0 | - | Secondary frequency control participation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` | 0.5 | pu-m | Mechanical power |
| $e'_q$ | `e1q` | 1.0 | pu-m | q-axis transient voltage |

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
| $p_g$ | `p_g` |  | pu-m | Active power |
| $q_g$ | `q_g` |  | pu-m | Reactive power |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_e$ | `p_e` |  | pu-m | Electrical power |


## Source

- Module: `pydae.bps.syns.milano2ord`
- File: [`packages/pydae-bps/src/pydae/bps/syns/milano2ord.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/milano2ord.py)
