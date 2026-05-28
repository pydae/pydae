# pv_pq_ss

*PV plants — pydae-bps model.*

## Model description

PV inverter with VRT and optional state-space p/q reference filter.

Combines the VRT ramp and soft-saturation blending of ``pv_dq_vrt`` with the
optional state-space low-pass filter for active/reactive power references from
``pv_dq_ss``.  No DC-link, PV-array physics, or L-filter — the model is purely
algebraic in p/q injection with one VRT dynamic state plus the optional SS
filter states.

**Auxiliary equations**

$$v_{sr} = V \cos(\theta), \quad v_{si} = V \sin(\theta)$$
$$v_{sq\_mag} = v_{sr}^2 + v_{si}^2 + \epsilon$$
$$v_m = \sqrt{v_{sq\_mag}}$$

**Dynamic equations**

$$\frac{d\,\text{lvrt\_ext\_ramp}}{dt} = \frac{\text{lvrt\_ext} - \text{lvrt\_ext\_ramp}}{T_{lvrt}}$$

Optional state-space filter for p/q references (active when ``A_pq`` or
``A_p`` are supplied in data_dict):

$$\dot{x} = A_{pq}\, x + B_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$
$$\begin{bmatrix} p_{s,ppc,d} \\ q_{s,ppc,d} \end{bmatrix} =
C_{pq}\, x + D_{pq} \begin{bmatrix} p_{s,ppc} \\ q_{s,ppc} \end{bmatrix}$$

**Current references (PQ mode)**

$$i_{sr,pq} = \frac{p_{s,ppc,d}\, v_{sr} + q_{s,ppc,d}\, v_{si}}{v_{sq\_mag}}$$
$$i_{si,pq} = \frac{p_{s,ppc,d}\, v_{si} - q_{s,ppc,d}\, v_{sr}}{v_{sq\_mag}}$$

**Current references (arbitrary mode)**

$$i_{sr,ar} = \frac{i_{sa,ref}\, v_{sr} + i_{sr,ref}\, v_{si}}{v_m}$$
$$i_{si,ar} = \frac{i_{sa,ref}\, v_{si} - i_{sr,ref}\, v_{sr}}{v_m}$$

**Blending and soft saturation**

$$i_{sr,nosat} = (1 - \text{lvrt\_ramp})\, i_{sr,pq} + \text{lvrt\_ramp}\cdot i_{sr,ar}$$
$$i_{si,nosat} = (1 - \text{lvrt\_ramp})\, i_{si,pq} + \text{lvrt\_ramp}\cdot i_{si,ar}$$
$$i_{mod} = \sqrt{i_{sr,nosat}^2 + i_{si,nosat}^2 + \epsilon}$$
$$i_{mod,sat} = \tfrac{1}{2}\!\left(i_{mod} + I_{max}
    - \sqrt{(i_{mod} - I_{max})^2 + \epsilon}\right)$$

**Algebraic equations**

$$0 = v_{sr} i_{sr} + v_{si} i_{si} - p_s$$
$$0 = v_{si} i_{sr} - v_{sr} i_{si} - q_s$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pv_pq_ss` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 1000000.0 | VA | Nominal power |
| $I_{max}$ | `I_max` | 1.2 | pu | Maximum current magnitude |
| $T_{lvrt}$ | `T_lvrt` | 0.02 | s | LVRT ramp time constant |
| $\epsilon$ | `Epsilon` | 1e-08 | - | Regularization for soft saturation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{s,ppc}$ | `p_s_ppc` | 1.0 | pu | Active power reference from PPC |
| $q_{s,ppc}$ | `q_s_ppc` | 0.0 | pu | Reactive power reference from PPC |
| $\text{lvrt}_{\text{ext}}$ | `lvrt_ext` | 0.0 | - | External LVRT trigger |
| $i_{sa,ref}$ | `i_sa_ref` | 0.0 | pu | Active current reference (arbitrary mode) |
| $i_{sr,ref}$ | `i_sr_ref` | 0.0 | pu | Reactive current reference (arbitrary mode) |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\text{lvrt}_{\text{ext,ramp}}$ | `lvrt_ext_ramp` |  | - | LVRT trigger ramped signal |
| $x_{pq}$ | `x_pq` |  | - | State-space filter states (when A_pq/A_p provided) |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_s$ | `p_s` |  | pu | Injected active power |
| $q_s$ | `q_s` |  | pu | Injected reactive power |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_{mod}$ | `i_mod` |  | pu | Current magnitude before saturation |
| $i_{mod,sat}$ | `i_mod_sat` |  | pu | Saturated current magnitude |


## Source

- Module: `pydae.bps.pvs.pv_pq_ss`
- File: [`packages/pydae-bps/src/pydae/bps/pvs/pv_pq_ss.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/pvs/pv_pq_ss.py)
