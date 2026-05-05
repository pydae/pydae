# bess_dq_vrt

*Voltage-source converters — pydae-bps model.*

## Model description

Battery energy storage system (BESS) with VSC dq control, SOC dynamics,
and voltage ride-through (VRT) capability.

**Auxiliar equations**

$$H = \frac{E_{kWh} \cdot 1000 \cdot 3600}{S_n}$$
$$\epsilon = soc_{ref} - soc$$
$$p_{soc} = -(K_p \epsilon + K_i \xi_{soc})$$
$$e = f_{spline}(soc) \quad \text{(OCV-SOC interpolation)}$$
$$v_{sr} = V \cos(\theta), \quad v_{si} = V \sin(\theta)$$
$$v_{sq\_mag} = v_{sr}^2 + v_{si}^2 + \epsilon_{reg}$$

**Dynamic equations**

$$\frac{d\,soc}{dt} = \frac{1}{H} (-i_{dc} \cdot e)$$
$$\frac{d\,\xi_{soc}}{dt} = soc_{ref} - soc$$
$$\frac{d\,\text{lvrt\_ext\_ramp}}{dt} = \frac{\text{lvrt\_ext} - \text{lvrt\_ext\_ramp}}{T_{lvrt}}$$

**Algebraic equations (DC side)**

$$0 = p_s + p_{loss} - p_{dc}$$
$$0 = v_{dc} i_{dc} - p_{dc}$$
$$0 = e - i_{dc} R_{bat} - v_{dc}$$

**Algebraic equations (AC side)**

$$0 = v_{sr} i_{sr} + v_{si} i_{si} - p_s$$
$$0 = v_{si} i_{sr} - v_{sr} i_{si} - q_s$$

**Current blending with soft saturation**

$$i_{sr,nosat} = (1 - \text{lvrt\_ramp}) i_{sr,pq} + \text{lvrt\_ramp} \cdot i_{sr,ar}$$
$$i_{si,nosat} = (1 - \text{lvrt\_ramp}) i_{si,pq} + \text{lvrt\_ramp} \cdot i_{si,ar}$$
$$i_{mod} = \sqrt{i_{sr,nosat}^2 + i_{si,nosat}^2 + \epsilon}$$
$$i_{mod,sat} = \frac{1}{2} \left(i_{mod} + I_{max} - \sqrt{(i_{mod} - I_{max})^2 + \epsilon}\right)$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `bess_dq_vrt` model is instantiated by including an entry in the relevant
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
| $I_{max}$ | `I_max` | 1.2 | pu | Maximum current magnitude |
| $T_{lvrt}$ | `T_lvrt` | 0.02 | s | LVRT ramp time constant |
| $\epsilon$ | `Epsilon` | 1e-08 | - | Regularization for soft saturation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{s,ppc}$ | `p_s_ppc` | 0.0 | pu | Active power reference from PPC |
| $q_{s,ppc}$ | `q_s_ppc` | 0.0 | pu | Reactive power reference from PPC |
| $soc_{ref}$ | `soc_ref` | 0.5 | pu | SOC reference setpoint |
| $\text{lvrt}_{\text{ext}}$ | `lvrt_ext` | 0.0 | - | External LVRT trigger |
| $i_{sa,ref}$ | `i_sa_ref` | 0.0 | pu | Active current reference (arbitrary mode) |
| $i_{sr,ref}$ | `i_sr_ref` | 0.0 | pu | Reactive current reference (arbitrary mode) |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $soc$ | `soc` |  | pu | State of charge |
| $\xi_{soc}$ | `xi_soc` |  | pu | SOC integral error state |
| $\text{lvrt}_{\text{ext,ramp}}$ | `lvrt_ext_ramp` |  | - | LVRT trigger ramped signal |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{dc}$ | `p_dc` |  | pu | DC power |
| $i_{dc}$ | `i_dc` |  | pu | DC current |
| $v_{dc}$ | `v_dc` |  | pu | DC voltage |
| $p_s$ | `p_s` |  | pu | Injected active power |
| $q_s$ | `q_s` |  | pu | Injected reactive power |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{loss}$ | `p_loss` |  | pu | VSC power losses |
| $i_s$ | `i_s` |  | pu | AC current magnitude |
| $e$ | `e` |  | pu | Battery OCV (from SOC curve) |
| $i_{mod}$ | `i_mod` |  | pu | Current magnitude before saturation |
| $i_{mod,sat}$ | `i_mod_sat` |  | pu | Saturated current magnitude |


## Source

- Module: `pydae.bps.vscs.bess_dq_vrt`
- File: [`packages/pydae-bps/src/pydae/bps/vscs/bess_dq_vrt.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/vscs/bess_dq_vrt.py)
