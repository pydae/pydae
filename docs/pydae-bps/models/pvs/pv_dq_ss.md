# pv_dq_ss

*PV plants — pydae-bps model.*

## Model description

PV inverter model with L-filter coupling, PQ control, and simplified PV array
(steady-state / purely algebraic).

**Auxiliar equations**

$$v_{sD} = V \sin(\theta)$$
$$v_{sQ} = V \cos(\theta)$$
$$v_{sd} = v_{sD} \cos(\delta) - v_{sQ} \sin(\delta)$$
$$v_{sq} = v_{sD} \sin(\delta) + v_{sQ} \cos(\delta)$$
$$v_m = \sqrt{v_{sd}^2 + v_{sq}^2}$$
$$\text{lvrt} = \begin{cases} 0.0 & v_m \geq v_{\text{lvrt}} \\ 1.0 & v_m < v_{\text{lvrt}} \end{cases}$$
$$+ \text{lvrt}_{\text{ext}}$$
$$V_{oc,t} = N_{pv,s} V_{oc} \left(1 + \frac{K_{vt}}{100}(T - T_{stc})\right)$$
$$V_{mp,t} = N_{pv,s} V_{mp} \left(1 + \frac{K_{vt}}{100}(T - T_{stc})\right)$$
$$I_{mp,t} = N_{pv,p} I_{mp} \left(1 + \frac{K_{it}}{100}(T - T_{stc})\right)$$
$$I_{mp,i} = I_{mp,t} \frac{G}{1000}$$
$$v_{dc,v} = v_1 - \frac{(i_1 - i_{pv})(v_1 - v_2)}{i_1 - i_2}$$
$$p_{mp} = \frac{V_{mp,t} I_{mp,i}}{S_n}$$

**Dynamic equations** (optional state-space filter for p/q references)

$$\dot{x} = A_{pq} x + B_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$
$$\begin{bmatrix} p_{s,ppc,d} \\ q_{s,ppc,d} \end{bmatrix} =$$
$$C_{pq} x + D_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$

**Algebraic equations**

$$0 = v_{dc,v}/V_{dc,b} - v_{dc}$$
$$0 = -i_{sd,ref} + \text{sat}\left((1-\text{lvrt}) i_{sd,pq,ref} + \text{lvrt} \cdot i_{sd,ar,ref}\right)$$
$$0 = -i_{sq,ref} + \text{sat}\left((1-\text{lvrt}) i_{sq,pq,ref} + \text{lvrt} \cdot i_{sq,ar,ref}\right)$$
$$0 = v_{ti} - R_s i_{si} + X_s i_{sr} - v_{si}$$
$$0 = v_{tr} - R_s i_{sr} - X_s i_{si} - v_{sr}$$
$$0 = i_{si} v_{si} + i_{sr} v_{sr} - p_s$$
$$0 = i_{si} v_{sr} - i_{sr} v_{si} - q_s$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pv_dq_ss` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 1000000.0 | VA | Nominal power |
| $U_n$ | `U_n` | 400.0 | V | Nominal RMS line-to-line voltage |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $X_s$ | `X_s` | 0.1 | pu | Coupling reactance (pu, S_n base) |
| $R_s$ | `R_s` | 0.01 | pu | Coupling resistance (pu, S_n base) |
| $I_{sc}$ | `I_sc` | 8.0 | A | Short-circuit current at STC |
| $V_{oc}$ | `V_oc` | 42.1 | V | Open-circuit voltage at STC |
| $I_{mp}$ | `I_mp` | 3.56 | A | MPP current at STC |
| $V_{mp}$ | `V_mp` | 33.7 | V | MPP voltage at STC |
| $K_{vt}$ | `K_vt` | -0.16 | %/C | Voltage temperature coefficient |
| $K_{it}$ | `K_it` | 0.065 | %/C | Current temperature coefficient |
| $N_{pv,s}$ | `N_pv_s` | 25 | - | Number of series PV cells |
| $N_{pv,p}$ | `N_pv_p` | 250 | - | Number of parallel PV strings |
| $v_{\text{lvrt}}$ | `v_lvrt` | 0.8 | pu | LVRT voltage threshold |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{s,ppc}$ | `p_s_ppc` | 0.5 | pu | Active power reference from PPC |
| $q_{s,ppc}$ | `q_s_ppc` | 0.0 | pu | Reactive power reference from PPC |
| $v_{dc}$ | `v_dc` | 1.5 | pu | DC-link voltage (pu) |
| $i_{sa,ref}$ | `i_sa_ref` | 0.0 | pu | Active current reference (arbitrary mode) |
| $i_{sr,ref}$ | `i_sr_ref` | 0.0 | pu | Reactive current reference (arbitrary mode) |
| $\text{lvrt}_{\text{ext}}$ | `lvrt_ext` | 0.0 | - | External LVRT trigger |
| $T$ | `temp_deg` | 25.0 | C | Cell temperature |
| $G$ | `irrad` | 1000.0 | W/m2 | Irradiance |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_{pq}$ | `x_pq` |  | - | State-space filter states (when A_pq provided) |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{dc}$ | `v_dc` |  | pu | DC-link voltage (normalized) |
| $i_{sd,ref}$ | `i_sd_ref` |  | pu | d-axis current reference |
| $i_{sq,ref}$ | `i_sq_ref` |  | pu | q-axis current reference |
| $i_{si}$ | `i_si` |  | pu | Inverter active-axis current |
| $i_{sr}$ | `i_sr` |  | pu | Inverter reactive-axis current |
| $p_s$ | `p_s` |  | pu | Injected active power |
| $q_s$ | `q_s` |  | pu | Injected reactive power |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $m_{ref}$ | `m_ref` |  | - | Modulation index reference |
| $\theta_{t,ref}$ | `theta_t_ref` |  | rad | Inverter voltage angle reference |
| $v_{sd}$ | `v_sd` |  | pu | d-axis bus voltage |
| $v_{sq}$ | `v_sq` |  | pu | q-axis bus voltage |
| $\text{lvrt}$ | `lvrt` |  | - | LVRT active flag |
| $p_{mp}$ | `p_mp` |  | pu | Maximum power point (pu) |
| $i_{pv}$ | `i_pv` |  | pu | PV array current (pu) |
| $v_{dc,v}$ | `v_dc_v` |  | V | PV array voltage (V) |
| $v_{ac,v}$ | `v_ac_v` |  | V | AC terminal voltage (V) |
| $i_s$ | `i_s` |  | pu | Inverter current magnitude |


## Source

- Module: `pydae.bps.pvs.pv_dq_ss`
- File: [`packages/pydae-bps/src/pydae/bps/pvs/pv_dq_ss.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/pvs/pv_dq_ss.py)
