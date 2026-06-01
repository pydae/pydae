# acdc_3ph_4w_vdc_q

*Voltage-source converters — pydae-uds model.*

## Model description

Bidirectional AC-DC converter, 3 phase + neutral on the AC side, two poles
on the DC side, controlled in **DC-bus voltage / per-phase reactive power**.
Used as a grid-forming converter on the DC link: it imposes the DC-pole
voltage and absorbs/injects whatever AC power balances it.

**AC side** — per-phase complex power on the converter terminals
($s_\varphi = (v_\varphi - v_n)\overline{i_\varphi}$, plus the neutral
$s_n = v_n \overline{i_n}$), expanded into real form to be backend-agnostic.

**Loss model** — per-wire conduction-loss polynomial in the rms current
(`A + B|i| + C|i|^2`), with a small smoothing constant inside the square
root so the Jacobian is well-defined at zero current:

$$|i_\varphi|_{rms} = \sqrt{(i_\varphi^r)^2 + (i_\varphi^i)^2 + 10^{-3}}$$
$$p_{loss,\varphi} = A_{loss} + B_{loss}|i|_{rms} + C_{loss}|i|^2_{rms}$$

**AC balance** — for each phase $\varphi$ the active power balance pulls
the DC active power $p_{dc}$ (split across phases by weights $C_\varphi$,
defaulting to 1/3 each) and adds the local + neutral conduction losses:

$$\Re(s_\varphi) = C_\varphi p_{dc} - p_{loss,\varphi} - p_{loss,n}/3$$
$$\Im(s_\varphi) = q_\varphi^{ref}$$

with $\sum_\varphi i_\varphi + i_n = 0$ closing the neutral KCL.

**DC side** — DC droop on the pole-to-pole voltage:

$$v_{dc} = v_{dc}^{ref} - K_{droop}\, p_{dc}$$

and a Thevenin-like coupling to the DC bus pole voltages through small
series resistances and a neutral-to-ground impedance $R_{gdc}$:

$$0 = v_{og} + v_{dc}/2 - R_{dc} i_{pos} - v_{pos}$$
$$0 = v_{og} - v_{dc}/2 - R_{dc} i_{neg} - v_{neg}$$
$$0 = -v_{og}/R_{gdc} - i_{pos} - i_{neg}$$
$$0 = -p_{dc} - v_{dc}/2 \cdot (i_{pos} - i_{neg})$$

**HJSON snippet**

```hjson
vscs: [
    {bus_ac: "A2", bus_dc: "D2", type: "acdc_3ph_4w_vdc_q",
     A: 0.0, B: 0.0, C: 0.0}
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `acdc_3ph_4w_vdc_q` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $A_{loss}$ | `A_{bus_ac}` |  | W | No-load loss |
| $B_{loss}$ | `B_{bus_ac}` |  | V | Linear-loss coefficient |
| $C_{loss}$ | `C_{bus_ac}` |  | \Omega | Quadratic-loss coefficient |
| $C_\varphi$ | `C_{ph}_{bus_ac}` | 0.3333333333333333 | - | DC-power share to phase ph (a/b/c) |
| $K_{droop}$ | `K_droop_{bus_ac}` | 0.0 | \Omega | DC voltage / power droop gain |
| $R_{dc}$ | `R_dc_{bus_dc}` | 1e-06 | \Omega | DC series resistance per pole |
| $R_{gdc}$ | `R_gdc_{bus_dc}` | 3.0 | \Omega | DC neutral-to-ground resistance |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{dc}^{ref}$ | `v_dc_{bus_dc}_ref` | 800.0 | V | DC voltage setpoint |
| $q_\varphi^{ref}$ | `q_vsc_{ph}_{bus_ac}` | 0.0 | var | Per-phase reactive-power setpoint |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_\varphi^{r,i}$ | `i_vsc_{bus_ac}_{ph}_{r,i}` |  | A | Per-phase converter current (a/b/c/n) |
| $p_{dc}$ | `p_vsc_{bus_dc}` |  | W | DC-side active power |
| $i_{pos}$ | `i_vsc_pos_{bus_dc}_sp` |  | A | DC positive-pole current |
| $i_{neg}$ | `i_vsc_{bus_dc}_sn` |  | A | DC negative-pole current |
| $v_{og}$ | `v_og_{bus_dc}` |  | V | DC neutral-to-ground voltage |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{vsc}$ | `p_vsc_{bus_ac}` |  | W | Total AC active power |
| $p_{vsc}^{loss}$ | `p_vsc_loss_{bus_ac}` |  | W | Total conduction loss |


## Source

- Module: `pydae.uds.vscs.acdc_3ph_4w_vdc_q`
- File: [`packages/pydae-uds/src/pydae/uds/vscs/acdc_3ph_4w_vdc_q.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/vscs/acdc_3ph_4w_vdc_q.py)
