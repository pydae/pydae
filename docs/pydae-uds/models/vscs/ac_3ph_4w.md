# ac_3ph_4w

*Voltage-source converters — pydae-uds model.*

## Model description

AC-only 4-wire VSC carrier model. The converter terminal voltages
$v_{t,\varphi}$ are inputs by default — set them directly to use this as a
fixed-source converter, or nest a `vsg` (e.g. `gflpfzv`) which converts
them into algebraic states driven by the chosen grid-forming control.

**Per-phase branch equation** ($\varphi \in \{a, b, c, n\}$, real form):

$$0 = v_{og} + v_{t,\varphi} - Z_\varphi i_{s,\varphi} - v_{s,\varphi}$$

with $Z_a = Z_b = Z_c = R_s + jX_s$ (phase impedance) and
$Z_n = R_{sn} + jX_{sn}$ (neutral impedance).

**Neutral-to-ground loop** — the converter neutral $v_{og}$ closes to ground
through $Z_{ng} = R_{ng} + jX_{ng}$:

$$0 = i_{sa} + i_{sb} + i_{sc} + i_{sn} + v_{og}/Z_{ng}$$

(expanded into real form by multiplying numerator and denominator by
$\overline{Z_{ng}}$ so the equation lives on CasADi `SX`).

**Conduction loss** (per wire, same polynomial as `acdc_3ph_4w_*`):

$$p_{loss,\varphi} = A_{loss}|i_\varphi|^2 + B_{loss}|i_\varphi| + C_{loss}$$
$$p_{dc} = p_{ac,total} + p_{loss,total}$$

**Defaults** — on construction the terminal voltages are seeded with a
balanced three-phase set at $V_n / \sqrt{3} \angle \phi$ and the loss
coefficients are estimated from the rated apparent power (1% conduction
loss assumption) unless overridden via the HJSON `A_loss`, `B_loss`,
`C_loss` fields.

**HJSON snippet**

```hjson
vscs: [
    {type: "ac_3ph_4w", bus: "A3", S_n: 100e3, U_n: 400,
     R: 0.01, X: 0.05, R_n: 0.01, X_n: 0.1, R_ng: 0.01, X_ng: 3.0,
     vsg: {type: "gflpfzv", ...}}
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `ac_3ph_4w` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` |  | VA | Rated apparent power |
| $U_n$ | `U_n` |  | V | Rated line-to-line voltage |
| $R_s$ | `R_s_{bus}` |  | \Omega | Phase resistance |
| $X_s$ | `X_s_{bus}` |  | \Omega | Phase reactance |
| $R_{sn}$ | `R_sn_{bus}` |  | \Omega | Neutral wire resistance |
| $X_{sn}$ | `X_sn_{bus}` |  | \Omega | Neutral wire reactance |
| $R_{ng}$ | `R_ng_{bus}` |  | \Omega | Neutral-to-ground resistance |
| $X_{ng}$ | `X_ng_{bus}` |  | \Omega | Neutral-to-ground reactance |
| $A_{loss}$ | `A_loss_{bus}` | S_n-derived | \Omega | Quadratic-loss coefficient |
| $B_{loss}$ | `B_loss_{bus}` | 1.0 | V | Linear-loss coefficient |
| $C_{loss}$ | `C_loss_{bus}` | S_n-derived | W | No-load loss |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{t,\varphi}^{r,i}$ | `v_t{a,b,c,n}_{r,i}_{bus}` | balanced | V | Terminal voltage components (turned into states by a nested vsg) |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_{s,\varphi}^{r,i}$ | `i_vsc_{bus}_{ph}_{r,i}` |  | A | Per-phase converter current |
| $v_{og}^{r,i}$ | `v_{bus}_o_{r,i}` |  | V | Neutral-to-ground voltage |
| $p_{dc}$ | `p_dc_{bus}` |  | W | DC-side power equivalent (after losses) |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_{vsc,\varphi}, q_{vsc,\varphi}$ | `p_vsc_{bus}_{ph}, q_vsc_{bus}_{ph}` |  | W / var | Per-phase grid-side complex power |


## Source

- Module: `pydae.uds.vscs.ac_3ph_4w`
- File: [`packages/pydae-uds/src/pydae/uds/vscs/ac_3ph_4w.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/vscs/ac_3ph_4w.py)
