# load_dc

*Loads — pydae-uds model.*

## Model description

Constant-power DC load between a bus's positive pole (node 0) and negative
pole (node 1).

**Algebraic equation** (one per DC load):

$$0 = i_p (v_p - v_n) - p$$

with $v_p, v_n$ the positive- and negative-pole voltages, $i_p$ the load
current entering at $v_p$, and $p$ the load active power (input).

**Bus current injections**:

$$g_{\text{bus}, \text{node}=0} \mathrel{{+}{=}} +i_p, \qquad
  g_{\text{bus}, \text{node}=1} \mathrel{{+}{=}} -i_p$$

**HJSON snippet**

```hjson
loads: [
    {bus: "D2", kW: 1.0, type: "DC", model: "ZIP"},
]
```

The `kW` field becomes the default `p_load_{bus}` input (in W) and can be
overridden at runtime via `model.ini({"p_load_D2": 5e3}, ...)`.

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `load_dc` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p$ | `p_load_{bus}` | kW*1e3 | W | Constant-power load setpoint |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_p$ | `i_load_{bus}_p_r` |  | A | Load current at the positive pole |
| $v_p$ | `V_{bus}_0_r` |  | V | Positive-pole voltage (already in the nodal vector) |
| $v_n$ | `V_{bus}_1_r` |  | V | Negative-pole voltage (already in the nodal vector) |


## Source

- Module: `pydae.uds.loads.load_dc`
- File: [`packages/pydae-uds/src/pydae/uds/loads/load_dc.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/loads/load_dc.py)
