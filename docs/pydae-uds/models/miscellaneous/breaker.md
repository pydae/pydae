# breaker

*Miscellaneous — pydae-uds model.*

## Model description

Four-wire breaker (or short series link) between two buses.

The breaker is modelled as a small series R+jX impedance per wire with a
boolean-like input `u_brk_{bus_1}` that scales the bus-side current
injections. Setting `u_brk = 0` disconnects both buses; `u_brk = 1` (default)
behaves as a permanent link with the small impedance.

**Per-phase branch equation** (one per phase $\varphi \in \{a, b, c, n\}$):

$$0 = v_{1,\varphi} - Z\, i_{\varphi} - v_{2,\varphi}, \qquad Z = R + jX$$

split into real form so it works on both SymPy and CasADi backends:

$$0 = v_{1,\varphi}^r - (R\,i_\varphi^r - X\,i_\varphi^i) - v_{2,\varphi}^r$$
$$0 = v_{1,\varphi}^i - (R\,i_\varphi^i + X\,i_\varphi^r) - v_{2,\varphi}^i$$

**Bus current injections** (added into the nodal `g` rows at each bus):

$$g_{\text{bus}_1, \varphi} \mathrel{{+}{=}} -u_{\text{brk}}\, i_\varphi$$
$$g_{\text{bus}_2, \varphi} \mathrel{{+}{=}} +u_{\text{brk}}\, i_\varphi$$

**HJSON snippet**

```hjson
breakers: [{bus_1: "A1", bus_2: "A2"}]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `breaker` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R_{bus_1}` | 0.0001 | \Omega | Series resistance per wire |
| $X$ | `X_{bus_1}` | 0.0001 | \Omega | Series reactance per wire |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $u_{brk}$ | `u_brk_{bus_1}` | 1.0 | - | Switching state (0 = open, 1 = closed) |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i^r_\varphi$ | `i_brk_{bus_1}_{phase}_r` |  | A | Per-phase real branch current |
| $i^i_\varphi$ | `i_brk_{bus_1}_{phase}_i` |  | A | Per-phase imag branch current |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\|i_\varphi\|$ | `i_brk_{bus}_{phase}_m` |  | A | Per-phase current magnitude (emitted on both buses) |


## Source

- Module: `pydae.uds.miscellaneous.breaker`
- File: [`packages/pydae-uds/src/pydae/uds/miscellaneous/breaker.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/miscellaneous/breaker.py)
