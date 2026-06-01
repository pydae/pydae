# lines

*Lines — pydae-uds model.*

## Model description

Three-phase (and DC) line model and admittance-primitive assembly.

Each line carries an `N_branches × N_branches` per-unit-length primitive
admittance matrix $Y_\text{primitive}$ that depends on the conductor
geometry (R, X matrices per length unit) and length. From this primitive,
the global system admittance $Y = G + jB$ is built by the standard
incidence-based assembly performed in `UdsBuilder.contruct_grid()`:

$$Y = A^T\, Y_\text{primitive}\, A$$

with $A$ the bus-to-branch incidence matrix.

**Symbolic (sym=true) vs numeric (sym=false) admittances.** When the line
HJSON entry sets `sym: true` (default for transmission-like lines), each
non-zero off-diagonal admittance entry becomes a backend symbol
$g_{jk}, b_{jk}$ registered in `params_dict` — useful for line-parameter
SSA studies. With `sym: false`, the numeric R/X values are baked into the
primitive directly.

**Line-current monitors.** `add_line_monitors` emits, for any line with
`monitor: true`, the per-branch current outputs
$\Re(i_{jk}), \Im(i_{jk}), |i_{jk}|$ as `i_l_{bus_j}_{nj}_{bus_k}_{nk}_*`,
read off the backend-agnostic real-form arrays
`grid.I_lines_re[idx]`, `grid.I_lines_im[idx]` (see
`UdsBuilder.add_branch_monitor`).

**HJSON snippets**

A length-coded AC line with a 4×4 R/X matrix referenced by `code`:

```hjson
lines: [
    {bus_j: "A2", bus_k: "A3", code: "UG1", m: 100.0, monitor: true, sym: true},
]
line_codes: {
    UG1: {R: [[…], [...], [...], [...]], X: [[…], [...], [...], [...]], I_max: 430.0}
}
```

A simple 4-wire AC line specified directly with scalar R, X (per Ω):

```hjson
lines: [
    {bus_j: "A1", bus_k: "A2", R: 0.1, X: 0.1, N_branches: 4},
]
```

A 2-wire DC line:

```hjson
lines: [
    {bus_j: "D2", bus_k: "D3", code: "UG1dc", m: 100.0, bus_j_nodes:[0,1], bus_k_nodes:[0,1]},
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `lines` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $R$ | `R` |  | \Omega/km | Per-length resistance matrix (used with `m`) |
| $X$ | `X` |  | \Omega/km | Per-length reactance matrix |
| $m$ | `m` |  | km | Line length |
| $I_{max}$ | `I_max` |  | A | Thermal ampacity (used by `model2svg` colouring) |
| $g_{jk}$ | `g_{bus_j}_{nj}_{bus_k}_{nk}_{col}` | \Re(Y_{prim}) | S | Symbolic conductance entry (with `sym:true`) |
| $b_{jk}$ | `b_{bus_j}_{nj}_{bus_k}_{nk}_{col}` | \Im(Y_{prim}) | S | Symbolic susceptance entry (with `sym:true`) |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_{jk}^r$ | `i_l_{bus_j}_{nj}_{bus_k}_{nk}_r` |  | A | Branch current, real (with `monitor:true`) |
| $i_{jk}^i$ | `i_l_{bus_j}_{nj}_{bus_k}_{nk}_i` |  | A | Branch current, imag |
| $\|i_{jk}\|$ | `i_l_{bus_j}_{nj}_{bus_k}_{nk}_m` |  | A | Branch current magnitude |


## Source

- Module: `pydae.uds.lines.lines`
- File: [`packages/pydae-uds/src/pydae/uds/lines/lines.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/lines/lines.py)
