# bess_dcdc_gf

*Energy storage systems — pydae-uds model.*

## Model description

Battery-energy-storage system with DC-DC grid-forming converter.

Two stages: a low-voltage battery (with an open-circuit-voltage curve and
internal resistance) feeds a DC-DC converter that imposes the high-voltage
DC bus with a $P/V$ linear droop and a slow SOC-balancing loop.

**Battery stage** — OCV curve as a piecewise-linear interpolant of
`(socs, es)` HJSON samples (replacement for the legacy SymPy
`interpolating_spline`), plus internal resistance $R_{bat}$:

$$e_{ocv}(\text{soc}) = \text{PWL}(\text{soc};\, \text{socs}, \text{es})$$
$$0 = e_{ocv} - i_l R_{bat} - v_{bat}$$
$$\dot{\text{soc}} = -\frac{i_l \cdot e_{ocv}}{1000 \cdot 3600 \cdot E_{kWh}}$$
$$\dot{\xi_{soc}} = \text{soc}_{ref} - \text{soc}$$

The SOC-PI ($K_p, K_i$ both default to $10^{-6}$ for slow action) produces
a $p_{soc}$ signal that can modulate the HV-side voltage reference.

**DC-DC stage** — HV pole-to-pole voltage with linear current/power droop:

$$v_h = v_{hp} - v_{hn}, \qquad
  e_h = e_h^{ref} - D_i i_{hf} - D_p i_{hf} v_h + K_{soc}\,p_{soc}$$

Currents into the positive and negative HV poles are derived from a small
$R_h, R_g$ Thevenin equivalent:

$$i_{hp} = (v_{og} + e_h/2 - v_{hp})/R_h, \qquad
  i_{hn} = (v_{og} - e_h/2 - v_{hn})/R_h$$

with $v_{og} = i_g R_g, \; i_g = (v_{hn} + v_{hp})/(2 R_g + R_h)$.

**LV-side balance** — HV input power plus conduction loss equals LV battery
power:

$$p_l = p_h + A\,i_{hp}^2 + B\,|i_{hp}| + C, \qquad i_l = p_l/v_{bat} - i_{charger}$$

with $|i_{hp}|$ approximated by $\sqrt{i_{hp}^2 + 10^{-12}}$ for
differentiability.

**HJSON snippet**

```hjson
ess: [
    {bus: "D3", type: "bess_dcdc_gf", E_kWh: 1000, soc_0: 0.1, v_ref: 800,
     A: 0.0, B: 0.0, C: 0.0,
     socs: [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
     es:   [600, 650, 680, 700, 710, 750]}
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `bess_dcdc_gf` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $E_{kWh}$ | `E_kWh_{name}` |  | kWh | Battery capacity |
| $R_h$ | `R_h_{name}` | 0.01 | \Omega | HV-side series resistance |
| $R_g$ | `R_g_{name}` | 3 | \Omega | HV neutral-to-ground resistance |
| $A$ | `A_{name}` |  | \Omega | Quadratic conduction-loss coefficient |
| $B$ | `B_{name}` |  | V | Linear conduction-loss coefficient |
| $C$ | `C_{name}` |  | W | No-load loss |
| $D_p$ | `Droop_p_{name}` | 0.0 | \Omega^{-1} | Power droop gain |
| $D_i$ | `Droop_i_{name}` | 0.0 | \Omega | Current droop gain |
| $T_f$ | `T_f_{name}` | 0.1 | s | Current LPF time constant |
| $K_p$ | `K_p_{name}` | 1e-06 | - | SOC-PI proportional gain |
| $K_i$ | `K_i_{name}` | 1e-06 | 1/s | SOC-PI integral gain |
| $K_{soc}$ | `K_soc_{name}` | 0.001 | V/W | SOC-to-voltage coupling |
| $K_{charger}$ | `K_charger_{name}` | 0.0 | - | Charger injection gain |
| $R_{bat}$ | `R_bat_{name}` | 0.0 | \Omega | Battery internal resistance |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $e_h^{ref}$ | `e_h_ref_{name}` | v_ref | V | HV voltage reference |
| $\text{soc}_{ref}$ | `soc_ref_{name}` | 0.5 | - | SOC reference |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_{hf}$ | `i_h_f_{name}` |  | A | HV-side filtered current |
| $\text{soc}$ | `soc_{name}` | soc_0 | - | Battery state of charge |
| $\xi_{soc}$ | `xi_soc_{name}` | 0.5 | -·s | SOC-PI integrator state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{bat}$ | `v_bat_{name}` |  | V | Battery terminal voltage |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_l$ | `i_l_{name}` |  | A | Battery-side current |
| $p_h$ | `p_h_{name}` |  | W | HV-side power |
| $e_h$ | `e_h_{name}` |  | V | HV reference (after droop / SOC) |
| $i_{charger}$ | `i_charger_{name}` |  | A | Charger current injection |


## Source

- Module: `pydae.uds.ess.bess_dcdc_gf`
- File: [`packages/pydae-uds/src/pydae/uds/ess/bess_dcdc_gf.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/ess/bess_dcdc_gf.py)
