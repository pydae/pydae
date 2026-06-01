# Dyn11

*Transformers — pydae-uds model.*

## Model description

Three-phase Dyn11 transformer (delta primary, wye-neutral secondary, +30° group).

The primitive admittance is built in real form so it lives on either the
SymPy or CasADi backend. The construction follows Rodríguez del Nozal,
Romero-Ramos & Trigo-García, *Accurate Assessment of Decoupled OLTC
Transformers to Optimize the Operation of Low-Voltage Networks*, Energies
12, 2173 (2019).

**Per-winding admittance** (one short-circuit branch + one magnetising
branch per phase, all the same value by default):

$$Y_t = G_t + j B_t = \frac{1}{R_{cc} + j X_{cc}} \cdot \frac{1}{Z_b},
  \qquad Z_b = U_{jn}^2 / S_n$$

$$Y_m = G_m + j B_m = \frac{1}{R_{fe} \cdot Z_b} - j \frac{1}{X_{mu} \cdot Z_b}$$

**Winding-to-terminal incidence** $N \in \mathbb{R}^{6 \times 7}$ encodes
the Dyn11 vector group (columns = $[A, B, C, a, b, c, n]$):

$$N = \begin{bmatrix}
 1 & -1 &  0 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 &  1 &  0 &  0 & -1 \\
 0 &  1 & -1 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 &  0 &  1 &  0 & -1 \\
-1 &  0 &  1 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 &  0 &  0 &  1 & -1
\end{bmatrix}$$

**Per-winding primitive** (with turns ratio $r$ applied symmetrically):

$$Y_p = \begin{bmatrix}
 Y_t + Y_m & -r\, Y_t \\
-r\, Y_t & r^2\, Y_t
\end{bmatrix} \text{ (one 2×2 block per phase, block-diagonal in 6×6)}$$

**Terminal primitive**:

$$Y_\text{primitive} = N^T Y_p N$$

with the LV neutral node closed to ground through $R_g$:
$Y_\text{primitive}[6, 6] \mathrel{{+}{=}} 1/R_g$.

The real and imaginary parts $G_\text{primitive}, B_\text{primitive}$ are
written directly into the grid's branch-admittance matrices at this
transformer's branch slice; the standard nodal assembly in
`UdsBuilder.contruct_grid()` picks them up.

**Per-phase tap ratios** `Ratio_{a,b,c}` enter the primitive as multipliers
on $Y_t$, so the model supports independent per-phase OLTC.

**HJSON snippet**

```hjson
transformers: [
    {bus_j: "MV0", bus_k: "I01", S_n_kVA: 100, U_j_kV: 20, U_k_kV: 0.4,
     R_cc_pu: 0.01, X_cc_pu: 0.04,
     connection: "Dyn11", conductors_j: 3, conductors_k: 4, monitor: true}
]
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `Dyn11` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n_kVA` |  | kVA | Nominal apparent power |
| $U_j$ | `U_j_kV` |  | kV | Primary (delta) line-to-line voltage |
| $U_k$ | `U_k_kV` |  | kV | Secondary (wye) line-to-line voltage |
| $R_{cc}$ | `R_cc_pu` |  | pu | Short-circuit resistance |
| $X_{cc}$ | `X_cc_pu` |  | pu | Short-circuit reactance |
| $R_{fe}$ | `R_fe_pu` | \infty | pu | Iron-loss resistance (optional) |
| $X_\mu$ | `X_mu_pu` | \infty | pu | Magnetising reactance (optional) |
| $R_g$ | `R_g_{bus_j}_{bus_k}` | 3.0 | \Omega | Neutral-to-ground resistance |
| $G_t$ | `G_t_{bus_j}_{bus_k}` |  | S | Per-winding conductance (derived) |
| $B_t$ | `B_t_{bus_j}_{bus_k}` |  | S | Per-winding susceptance (derived) |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $r_a$ | `Ratio_a_{bus_j}_{bus_k}` | 1.0 | - | Phase-a OLTC tap ratio |
| $r_b$ | `Ratio_b_{bus_j}_{bus_k}` | 1.0 | - | Phase-b OLTC tap ratio |
| $r_c$ | `Ratio_c_{bus_j}_{bus_k}` | 1.0 | - | Phase-c OLTC tap ratio |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $i_{t,j}^{\varphi r/i}$ | `i_t_{bus_j}_{bus_k}_1_{ph}_{r,i,m}` |  | A | Primary-side branch currents (with `monitor:true`) |
| $i_{t,k}^{\varphi r/i}$ | `i_t_{bus_j}_{bus_k}_2_{ph}_{r,i,m}` |  | A | Secondary-side branch currents |


## Source

- Module: `pydae.uds.transformers.Dyn11`
- File: [`packages/pydae-uds/src/pydae/uds/transformers/Dyn11.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/transformers/Dyn11.py)
