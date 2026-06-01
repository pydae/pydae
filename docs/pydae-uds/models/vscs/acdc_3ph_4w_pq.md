# acdc_3ph_4w_pq

*Voltage-source converters — pydae-uds model.*

## Model description

Bidirectional AC-DC converter, 3 phase + neutral on the AC side, two poles
on the DC side, controlled in **per-phase active and reactive power**.

Unlike `acdc_3ph_4w_vdc_q` this model does **not** impose a DC voltage; it
tracks per-phase $p_{ac,\varphi}$, $q_{ac,\varphi}$ setpoints and the DC
side balances passively. The model pairs naturally with the
`ctrl_3ph_4w_droop` outer-loop control which sets the $p$ references from
an AC/DC voltage difference.

**AC balance** (one $p$ and one $q$ equation per phase, in real form):

$$\Re(s_\varphi) = p_\varphi^{ref}, \qquad \Im(s_\varphi) = q_\varphi^{ref}$$

with $s_\varphi = (v_\varphi - v_n)\overline{i_\varphi}$ and $\sum_\varphi i_\varphi + i_n = 0$.

**Loss model** — same conduction-loss polynomial as `acdc_3ph_4w_vdc_q`
(see its docstring), with a slightly larger smoothing constant
$|i|^2 \to |i|^2 + 10^{-2}$ to stabilise during start-up.

**DC balance**:

$$0 = -p_{dc} + p_{ac,total} + p_{loss,total}$$
$$i_{d} = \frac{p_{dc}}{v_{dc}} \quad \text{(injected into the DC poles)}$$

**HJSON snippet**

```hjson
vscs: [
    {bus_ac: "A4", bus_dc: "D4", type: "acdc_3ph_4w_pq",
     A: 350, B: 0, C: 0.03,
     vsc_ctrl: {type: "ctrl_3ph_4w_droop"}}
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `acdc_3ph_4w_pq` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $A_{loss}$ | `A_loss_{bus_ac}` |  | W | No-load loss |
| $B_{loss}$ | `B_loss_{bus_ac}` |  | V | Linear-loss coefficient |
| $C_{loss}$ | `C_loss_{bus_ac}` |  | \Omega | Quadratic-loss coefficient |
| $R_{dc}$ | `R_dc_{bus_dc}` | 1e-06 | \Omega | DC series resistance |
| $R_{gdc}$ | `R_gdc_{bus_dc}` | 3.0 | \Omega | DC neutral-to-ground resistance |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_\varphi^{ref}$ | `p_vsc_{ph}_{bus_ac}` | 0.0 | W | Per-phase active-power setpoint (replaced when a vsc_ctrl is attached) |
| $q_\varphi^{ref}$ | `q_vsc_{ph}_{bus_ac}` | 0.0 | var | Per-phase reactive-power setpoint |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_\varphi^{r,i}$ | `i_vsc_{bus_ac}_{ph}_{r,i}` |  | A | Per-phase converter current (a/b/c/n) |
| $p_{dc}$ | `p_vsc_{bus_dc}` |  | W | DC-side active power |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{vsc}$ | `p_vsc_{bus_ac}` |  | W | Total AC active power |
| $q_{vsc}$ | `q_vsc_{bus_ac}` |  | var | Total AC reactive power |
| $\|s_{vsc}\|$ | `s_vsc_{bus_ac}` |  | VA | Total AC apparent power magnitude |
| $v_{dc}$ | `v_dc_{bus_dc}` |  | V | DC pole-to-pole voltage |
| $\|v_{an}\|$ | `v_anm_{bus_ac}` |  | V | Phase-a to neutral voltage magnitude |


## Source

- Module: `pydae.uds.vscs.acdc_3ph_4w_pq`
- File: [`packages/pydae-uds/src/pydae/uds/vscs/acdc_3ph_4w_pq.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/vscs/acdc_3ph_4w_pq.py)
