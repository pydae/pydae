# load_ac

*Loads — pydae-uds model.*

## Model description

Three-phase + neutral ZIP load on a 4-wire AC bus.

For each phase $\varphi \in \{a, b, c\}$ the model superposes a
**constant-power** part $S_{s,\varphi}$ and a **constant-impedance** part
$S_{z,\varphi}$, both expressed against the phase-to-neutral voltage
$v_{\varphi n} = v_\varphi - v_n$ and split into real form to be backend-
agnostic.

**Constant-power part** ($s_s = v_{\varphi n} \overline{i_\varphi}$):

$$\Re(s_{s,\varphi}) = v_{\varphi n}^r i_\varphi^r + v_{\varphi n}^i i_\varphi^i$$
$$\Im(s_{s,\varphi}) = v_{\varphi n}^i i_\varphi^r - v_{\varphi n}^r i_\varphi^i$$

**Constant-impedance part** ($s_z = \overline{(g+jb)\, v_{\varphi n}}\, v_{\varphi n}$):

$$\Re(s_{z,\varphi}) =  g_\varphi |v_{\varphi n}|^2, \qquad
  \Im(s_{z,\varphi}) = -b_\varphi |v_{\varphi n}|^2$$

**Low-voltage stalling** — to keep the load tractable during ini() when a
phase voltage collapses, the constant-power numerator is divided by a
ramped factor that saturates at unity above $V_{th} = 0.7\,\text{pu}$:

$$K_v = \begin{cases} v_\varphi^{m} + 0.3 & v_\varphi^{m} < V_{th} \\
                       1 & v_\varphi^{m} \ge V_{th} \end{cases}$$

**Algebraic balance equations** (one $p$ and one $q$ per phase):

$$0 = K_{abc}\left(p_\varphi + p_{z,\varphi} + p_{s,\varphi}/K_v\right)$$
$$0 = K_{abc}\left(q_\varphi + q_{z,\varphi} + q_{s,\varphi}/K_v\right)$$

plus the neutral KCL $\sum_\varphi i_\varphi + i_n = 0$ in real / imag parts.

**HJSON snippet**

```hjson
loads: [
    {bus: "A2", kVA: 10.0, pf: 0.95, type: "3P+N", model: "ZIP"},
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `load_ac` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_{abc}$ | `K_abc_{bus}` | 1.0 | - | Scaling on the per-phase balance equations |
| $g_\varphi$ | `g_load_{bus}_{ph}` | 0.0 | S | Constant-Z conductance per phase |
| $b_\varphi$ | `b_load_{bus}_{ph}` | 0.0 | S | Constant-Z susceptance per phase |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_\varphi$ | `p_load_{bus}_{ph}` | kVA*pf*1e3/3 | W | Constant-P setpoint per phase (initialised from kVA and pf) |
| $q_\varphi$ | `q_load_{bus}_{ph}` | \sqrt{S^2-P^2}/3 | var | Constant-Q setpoint per phase |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_\varphi^r$ | `i_load_{bus}_{ph}_r` |  | A | Per-phase load current, real |
| $i_\varphi^i$ | `i_load_{bus}_{ph}_i` |  | A | Per-phase load current, imag |
| $i_n^r$ | `i_load_{bus}_n_r` |  | A | Neutral return current, real |
| $i_n^i$ | `i_load_{bus}_n_i` |  | A | Neutral return current, imag |


## Source

- Module: `pydae.uds.loads.load_ac`
- File: [`packages/pydae-uds/src/pydae/uds/loads/load_ac.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/loads/load_ac.py)
