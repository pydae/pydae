# change_line

*Lines — runtime parameter update — `pydae.bps.lines.lines`*

---

## Purpose

`change_line` updates the series admittance and (optionally) shunt susceptance
of a transmission line **after** `ini()` has been called, without rebuilding
the model.  This enables mid-simulation network switching, contingency studies,
and parameter sweeps.

The function accepts a single dict that mirrors the `lines[]` entry in the
system HJSON exactly, so the same data used at build time can be reused at
runtime with modified impedance values.

---

## Signature

```python
from pydae.bps.lines.lines import change_line

change_line(model, line_dict)
```

| Argument | Type | Description |
|---|---|---|
| `model` | `pydae.core.Model` | Initialised model instance |
| `line_dict` | `dict` | HJSON-style line entry (see formats below) |

---

## Supported impedance formats

The function accepts the same three formats as the `BpsBuilder`:

### Per-unit on S_mva base

```python
change_line(model, {
    "bus_j": "2", "bus_k": "3",
    "X_pu": 0.01, "R_pu": 0.0,
    "Bs_pu": 0.0,              # optional — omit to leave shunt unchanged
    "S_mva": 100
})
```

Scaling:
$$R_{sys} = R_{pu} \cdot \frac{S_{base}}{S_{mva}}, \quad
  X_{sys} = X_{pu} \cdot \frac{S_{base}}{S_{mva}}, \quad
  Bs_{sys} = Bs_{pu} \cdot \frac{S_{mva}}{S_{base}}$$

### Absolute ohms

```python
change_line(model, {
    "bus_j": "5", "bus_k": "6",
    "X": 26.4, "R": 0.53
})
```

Uses the bus nominal voltage to compute $Z_{base} = U_{kV}^2 / S_{base}$.

### Specific impedance × length

```python
change_line(model, {
    "bus_j": "7", "bus_k": "8",
    "X_km": 0.529, "R_km": 0.0529,
    "Bs_km": 2.1e-6, "km": 110
})
```

---

## What is updated

For a branch between buses $j$ and $k$ the model stores three parameters:
$g_{jk}$, $b_{jk}$, and $bs_{jk}$ (per-side shunt).  `change_line` recomputes
and writes these directly:

$$G_{jk} = \frac{R}{R^2 + X^2}, \quad B_{jk} = \frac{-X}{R^2 + X^2}$$

Omitting `Bs_pu` / `Bs_km` leaves $bs_{jk}$ at its current value — useful when
only the series impedance changes (e.g. line switching without reactive
compensation change).

---

## Bus ordering

The builder stores parameters under the key `{bus_j}_{bus_k}` as written in the
HJSON.  `change_line` tries both orderings (`bus_j_bus_k` and `bus_k_bus_j`)
when looking up the parameter names, so the call is valid regardless of the
order in which the buses appear in the HJSON.

---

## Usage examples

### Contingency: weaken the 2–3 interconnection

```python
from pydae.bps.lines.lines import change_line

model.ini({'p_c_lc_1': 0.9}, 'xy_0.json')
model.run(1.0, {})                        # steady state

# trip the stronger parallel path → X jumps from 0.01 to 0.6
change_line(model, {
    "bus_j": "2", "bus_k": "3",
    "X_pu": 0.6, "R_pu": 0.0, "Bs_pu": 0.0, "S_mva": 100
})
model.run(30.0, {})
model.post()
```

### Parameter sweep: vary line reactance

```python
import numpy as np
from pydae.bps.lines.lines import change_line

results = []
for x in np.linspace(0.01, 0.6, 20):
    model.ini({'p_c_lc_1': 0.9}, 'xy_0.json')
    change_line(model, {
        "bus_j": "2", "bus_k": "3",
        "X_pu": x, "R_pu": 0.0, "S_mva": 100
    })
    model.run(0.0, {})          # re-solve algebraic equations
    results.append(model.get_value('p_g_1'))
```

### Restore original parameters

```python
# restore
change_line(model, {
    "bus_j": "2", "bus_k": "3",
    "X_pu": 0.01, "R_pu": 0.0, "Bs_pu": 0.0, "S_mva": 100
})
```

---

## Notes

- `change_line` only updates **parameters** in the compiled model — the
  symbolic structure (sparsity pattern, Jacobian) is unchanged.  This means
  it cannot add or remove branches; it can only modify the admittance of an
  existing branch.
- For zero-impedance switching (ideal breaker), use a very small but non-zero
  reactance (e.g. `X_pu: 1e-6`) to avoid a singular Jacobian.
- After calling `change_line` during a running simulation (`run()` already in
  progress), the next integration step automatically uses the new parameters
  — no `ini()` or rebuild is required.

---

## Source

- Module: `pydae.bps.lines.lines`
- File: [`packages/pydae-bps/src/pydae/bps/lines/lines.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/lines/lines.py)

```{eval-rst}
.. autofunction:: pydae.bps.lines.lines.change_line
```
