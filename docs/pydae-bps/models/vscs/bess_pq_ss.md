# bess_pq_ss

*Voltage-source converters — pydae-bps model.*

## Model description

Battery energy storage system (BESS) with VSC PQ control and SOC dynamics
(steady-state DC side, filtered AC power references).

**Auxiliar equations**

$$H = \frac{E_{kWh} \cdot 1000 \cdot 3600}{S_n}$$
$$\epsilon = soc_{ref} - soc$$
$$p_{soc} = -(K_p \epsilon + K_i \xi_{soc})$$
$$e = f_{spline}(soc) \quad \text{(OCV-SOC interpolation)}$$
$$s_s = \sqrt{p_s^2 + q_s^2}$$
$$i_s = s_s / V$$
$$p_{loss} = A_{loss} i_s^2 + B_{loss} i_s + C_{loss}$$

**Dynamic equations**

$$\frac{d\,soc}{dt} = \frac{1}{H} (-i_{dc} \cdot e)$$
$$\frac{d\,\xi_{soc}}{dt} = soc_{ref} - soc$$
$$\dot{x} = A_{pq} x + B_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$

**Algebraic equations**

$$0 = p_s + p_{loss} - p_{dc}$$
$$0 = v_{dc} i_{dc} - p_{dc}$$
$$0 = e - i_{dc} R_{bat} - v_{dc}$$

**Power reference with SOC limits**

$$p_s = \begin{cases} p_{s,ref} & p_{s,ref} \leq 0 \land soc < soc_{max} \\
p_{s,ref} & p_{s,ref} > 0 \land soc > soc_{min} \\
0 & \text{otherwise} \end{cases} + p_{soc}$$
$$q_s = q_{s,ref}$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `bess_pq_ss` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 1000000.0 | VA | Nominal power |
| $E_{kWh}$ | `E_kWh` | 100.0 | kWh | Battery energy capacity |
| $soc_{min}$ | `soc_min` | 0.0 | pu | Minimum state of charge |
| $soc_{max}$ | `soc_max` | 1.0 | pu | Maximum state of charge |
| $R_{bat}$ | `R_bat` | 0.0 | ohm | Battery internal resistance |
| $A_{loss}$ | `A_loss` | 0.0001 | - | Quadratic loss coefficient |
| $B_{loss}$ | `B_loss` | 0.0 | - | Linear loss coefficient |
| $C_{loss}$ | `C_loss` | 0.0001 | - | Constant loss coefficient |
| $K_p$ | `K_p` | 1e-06 | - | SOC proportional gain |
| $K_i$ | `K_i` | 1e-06 | - | SOC integral gain |
| $e_{soc\_order}$ | `e_soc_order` | 1 | - | Spline interpolation order for OCV-SOC |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{s,ppc}$ | `p_s_ppc` | 0.0 | pu | Active power reference from PPC |
| $q_{s,ppc}$ | `q_s_ppc` | 0.0 | pu | Reactive power reference from PPC |
| $soc_{ref}$ | `soc_ref` | 0.5 | pu | SOC reference setpoint |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $soc$ | `soc` |  | pu | State of charge |
| $\xi_{soc}$ | `xi_soc` |  | pu | SOC integral error state |
| $x_{pq}$ | `x_pq` |  | - | State-space filter states |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{dc}$ | `p_dc` |  | pu | DC power |
| $i_{dc}$ | `i_dc` |  | pu | DC current |
| $v_{dc}$ | `v_dc` |  | pu | DC voltage |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{loss}$ | `p_loss` |  | pu | VSC power losses |
| $i_s$ | `i_s` |  | pu | AC current magnitude |
| $e$ | `e` |  | pu | Battery OCV (from SOC curve) |
| $p_s$ | `p_s` |  | pu | Injected active power |
| $q_s$ | `q_s` |  | pu | Injected reactive power |
| $p_{s,ref}$ | `p_s_ref` |  | pu | Filtered active power reference |
| $q_{s,ref}$ | `q_s_ref` |  | pu | Filtered reactive power reference |


## Source

- Module: `pydae.bps.vscs.bess_pq_ss`
- File: [`packages/pydae-bps/src/pydae/bps/vscs/bess_pq_ss.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/vscs/bess_pq_ss.py)
