# ctrl_3ph_4w_droop

*VSC outer-loop controls — pydae-uds model.*

## Model description

AC/DC linear droop outer-loop control for a 4-wire AC-DC VSC.

The control couples the DC-bus per-unit voltage to the AC-bus per-unit
phase-neutral voltage through a per-phase droop gain, generating the
per-phase active-power command that drives the underlying VSC. With a zero
droop gain the controller passes the reference signals straight through.

**Per-unit voltages**

$$v_{dc}^{pu} = \frac{|v_{dc}|}{V_{dc,b}}, \qquad
  V_{phn,\varphi}^{pu} = \frac{|V_{\varphi n}|}{V_{ac,b}}$$

with $V_{ac,b} = U_{ac,b}/\sqrt{3}$ and $|v_{dc}|, |V_{\varphi n}|$ the
DC pole-to-pole and AC phase-to-neutral voltage magnitudes (already
emitted as 2-norms of the rectangular node voltages).

**Droop equation** (one per phase $\varphi \in \{a, b, c\}$):

$$p_{ac,\varphi} = K_{acdc}\, K_{acdc,\varphi}\, (v_{dc}^{pu} - V_{phn,\varphi}^{pu})
                  + p_{vsc,\varphi}^{ref}$$

**Algebraic balance**: the control turns the per-phase `p_vsc_{abc}_{bus_ac}`
variables (inputs of the host VSC) into algebraic states by adding the
equations $p_{ac,\varphi} - p_{vsc,\varphi} = 0$ to `g_list`, and pops the
corresponding entries from `u_ini_dict` / `u_run_dict`. The reference
channel `p_vsc_{abc}_ref_{bus_ac}` becomes the new input.

**HJSON snippet** (nested inside an AC-DC VSC entry):

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

The `ctrl_3ph_4w_droop` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_{acdc}$ | `K_acdc_{bus_ac}` | 0.0 | - | Common droop gain (set >0 to enable the AC/DC coupling) |
| $K_{acdc,\varphi}$ | `K_acdc_{ph}_{bus_ac}` | 1.0 | - | Per-phase droop weight |
| $U_{ac,b}$ | `U_ac_b_{bus_ac}` | bus U_kV * 1e3 | V | AC base voltage (auto-set from bus U_kV) |
| $V_{dc,b}$ | `V_dc_b_{bus_dc}` | bus U_kV * 1e3 | V | DC base voltage |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{vsc,\varphi}^{ref}$ | `p_vsc_{ph}_ref_{bus_ac}` | 0.0 | W | Per-phase reference active-power injection |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{vsc,\varphi}$ | `p_vsc_{ph}_{bus_ac}` |  | W | Per-phase active-power injection commanded to the VSC |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{ac,\varphi}^{pu}$ | `v_ac_{ph}_pu_{bus_ac}` |  | pu | AC phase-neutral per-unit voltage (monitor) |
| $v_{dc}^{pu}$ | `v_dc_pu_{bus_dc}` |  | pu | DC pole-to-pole per-unit voltage (monitor) |


## Source

- Module: `pydae.uds.vsc_ctrls.ctrl_3ph_4w_droop`
- File: [`packages/pydae-uds/src/pydae/uds/vsc_ctrls/ctrl_3ph_4w_droop.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/vsc_ctrls/ctrl_3ph_4w_droop.py)
