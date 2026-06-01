# shunts

*Shunts — pydae-uds model.*

## Model description

Single-node shunt admittance attached to one node of one bus.

The shunt is specified in `(R, X)` per-Ω form and converted internally to a
nodal-admittance contribution. Only one bus node is referenced (`bus_nodes[0]`);
the implicit return is system ground.

**Admittance**

$$Z = R + j X$$
$$Y_{jk} = \frac{1}{Z} = g_{jk} + j\, b_{jk}$$

The conductance and susceptance enter the branch primitive matrices at the
shunt's branch index:

$$G_{\text{primitive}}[i, i] = g_{jk}, \qquad
  B_{\text{primitive}}[i, i] = b_{jk}$$

and the bus-to-branch incidence row marks the connected node with `+1`.
The shunt therefore adds the admittance `g_{jk} + j b_{jk}` between the
selected node and ground via the standard nodal-admittance assembly
performed by `UdsBuilder.contruct_grid()`.

**HJSON snippet**

```hjson
shunts: [
    {bus: "A1", R: 3.0, X: 0.0, bus_nodes: [3, 0]},
]
```

`bus_nodes[0]` is the connected node index (here `3`, the neutral); the
second entry is kept for backward compatibility but currently ignored
(ground is always the return path).

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `shunts` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R` |  | \Omega | Shunt resistance (HJSON field) |
| $X$ | `X` |  | \Omega | Shunt reactance (HJSON field) |
| $g_{jk}$ | `g_shunt_{bus}_{node}` | 1/R | S | Derived conductance, $\Re(1/(R+jX))$ |
| $b_{jk}$ | `b_shunt_{bus}_{node}` | -X/(R^2+X^2) | S | Derived susceptance, $\Im(1/(R+jX))$ |
| $n$ | `bus_nodes` |  | - | Connected node index on the bus (HJSON field) |


## Source

- Module: `pydae.uds.shunts.shunts`
- File: [`packages/pydae-uds/src/pydae/uds/shunts/shunts.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/shunts/shunts.py)
