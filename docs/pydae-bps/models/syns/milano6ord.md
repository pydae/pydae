# milano6ord

*Synchronous machines — pydae-bps model.*

## Model description

Synchronous machine model of order 6 (Subtransient Model) with PSAT Saturation.

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
$$\frac{ de'_q}{dt} = \frac{1}{T'_{d0}} \left(-e'_q(1 + S_d) - \left(X_d - X'_d - \frac{T''_{d0}}{T'_{d0}} \frac{X''_d}{X'_d} (X_d - X'_d)\right)i_d + \left(1 - \frac{T_{AA}}{T'_{d0}}\right)v_f \right)$$
$$\frac{ de'_d}{dt} = \frac{1}{T'_{q0}} \left(-e'_d(1 + S_q) + \left(X_q - X'_q - \frac{T''_{q0}}{T'_{q0}} \frac{X''_q}{X'_q} (X_q - X'_q)\right)i_q \right)$$
$$\frac{ de''_q}{dt} = \frac{1}{T''_{d0}} \left(-e''_q + e'_q - \left(X'_d - X''_d + \frac{T''_{d0}}{T'_{d0}} \frac{X''_d}{X'_d} (X_d - X'_d)\right)i_d + \frac{T_{AA}}{T'_{d0}} v_f \right)$$
$$\frac{ de''_d}{dt} = \frac{1}{T''_{q0}} \left(-e''_d + e'_d + \left(X'_q - X''_q + \frac{T''_{q0}}{T'_{q0}} \frac{X''_q}{X'_q} (X_q - X'_q)\right)i_q \right)$$**Algebraic equations**$$0 = v_q + R_a i_q - e''_q + (X''_d - X_l)i_d$$$$0 = v_d + R_a i_d - e''_d - (X''_q - X_l)i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `milano6ord` model is instantiated by including an entry in the relevant
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
| $X''_d$ | `X2d` | 0.2 | pu-m | d-axis subtransient reactance |
| $X''_q$ | `X2q` | 0.25 | pu-m | q-axis subtransient reactance |
| $X_l$ | `X_l` | 0.1 | pu-m | Leakage reactance |
| $T'_{d0}$ | `T1d0` | 8.0 | s | d-axis open circuit transient time constant |
| $T'_{q0}$ | `T1q0` | 0.4 | s | q-axis open circuit transient time constant |
| $T''_{d0}$ | `T2d0` | 0.03 | s | d-axis open circuit subtransient time constant |
| $T''_{q0}$ | `T2q0` | 0.05 | s | q-axis open circuit subtransient time constant |
| $T_{AA}$ | `T_AA` | 0.0 | s | d-axis additional time constant |
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
| $e''_q$ | `e2q` |  | pu-m | q-axis subtransient voltage |
| $e''_d$ | `e2d` |  | pu-m | d-axis subtransient voltage |

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

- Module: `pydae.bps.syns.milano6ord`
- File: [`packages/pydae-bps/src/pydae/bps/syns/milano6ord.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/milano6ord.py)
