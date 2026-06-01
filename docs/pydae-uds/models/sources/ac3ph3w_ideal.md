# ac3ph3w_ideal

*Sources — pydae-uds model.*

## Model description

Ideal three-phase three-wire voltage source (slack/infinite bus) on a 3-wire AC bus.

Each phase EMF is a controlled sinusoid behind a series impedance
$Z_s = R + jX$. There is no neutral wire; the bus has `N_nodes = 3`.

**Phase EMFs** (rectangular, with per-phase magnitude inputs and a common
phase reference $\phi$):

$$e_a = v_{pu} e_{ao}^m \angle (\phi)$$
$$e_b = v_{pu} e_{bo}^m \angle (\phi - 2\pi/3)$$
$$e_c = v_{pu} e_{co}^m \angle (\phi - 4\pi/3)$$

**Per-phase branch equation** ($e_\varphi - Z_s i_\varphi - v_\varphi = 0$,
expanded to real form):

$$0 = e_\varphi^r - (R\,i_\varphi^r - X\,i_\varphi^i) - v_\varphi^r$$
$$0 = e_\varphi^i - (R\,i_\varphi^i + X\,i_\varphi^r) - v_\varphi^i$$

**Outputs** — total active and reactive power injected at the bus
($s = v \overline{i}$, summed over the three phases) plus per-phase
current magnitudes.

**HJSON snippet**

```hjson
buses: [{name: "A0", U_kV: 20.0, N_nodes: 3, phi_deg_0: 30.0}]
sources: [
    {type: "ac3ph3w_ideal", bus: "A0", S_n: 100e3, U_n: 20e3,
     R: 0.01, X: 0.1}
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `ac3ph3w_ideal` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R_s$ | `R_{bus}_s` | 0.01 | \Omega | Series resistance per phase |
| $X_s$ | `X_{bus}_s` | 0.1 | \Omega | Series reactance per phase |
| $U_n$ | `U_n` |  | V | Nominal line-to-line voltage (sets default EMF) |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $e_{\varphi o}^m$ | `e_{a,b,c}o_m_{bus}` | U_n/\sqrt{3} | V | Per-phase EMF magnitude |
| $v_{pu}$ | `v_pu_{bus}` | 1.0 | pu | Common per-unit voltage multiplier |
| $\phi$ | `phi_{bus}` | 0.0 | rad | Common phase reference |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_\varphi^r$ | `i_vsc_{bus}_{ph}_r` |  | A | Per-phase source current, real |
| $i_\varphi^i$ | `i_vsc_{bus}_{ph}_i` |  | A | Per-phase source current, imag |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\|i_\varphi\|$ | `i_vsc_{bus}_{ph}_m` |  | A | Per-phase current magnitude |
| $P$ | `p_{bus}` |  | W | Total injected active power |
| $Q$ | `q_{bus}` |  | var | Total injected reactive power |


## Source

- Module: `pydae.uds.sources.ac3ph3w_ideal`
- File: [`packages/pydae-uds/src/pydae/uds/sources/ac3ph3w_ideal.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/sources/ac3ph3w_ideal.py)
