# Getting started

This walkthrough takes one of the worked examples shipped with the package —
a grid-following VSC feeding a 3-phase 4-wire line — and drives it end-to-end.

## 1. Install

```bash
pip install pydae-uds
```

This also installs `pydae` (the core solver). A C compiler must be on PATH.

## 2. Pick a built-in example

`pydae-uds` ships many small example JSON / HJSON files next to the element
they exercise — for instance
`packages/pydae-uds/src/pydae/uds/vscs/ac_3ph_4w_pq_ib.hjson`. Start from one
of those and copy it to your working directory as `feeder.hjson`.

## 3. Build

```python
from pydae.uds import UdsBuilder
from pydae.core import Builder, Model

grid = UdsBuilder("feeder.hjson")
grid.checker()
grid.construct("feeder")

bld = Builder(grid.sys_dict, target="ctypes")
bld.build()
```

## 4. Initialise and simulate

```python
model = Model("feeder")

model.ini({}, xy_0=1.0)
model.run(1.0, {})
model.post()
```

## 5. Inspect results

The helpers in `pydae.uds.utils` work directly with 3-phase quantities:

```python
from pydae.uds.utils import report_v, get_v, get_i, get_power
import matplotlib.pyplot as plt

report_v(grid, data_input=grid.data, show=True, model="uds")

fig, ax = plt.subplots()
for ph in ("a", "b", "c"):
    ax.plot(model.Time, model.get_values(f"v_B_{ph}_m"), label=f"|V_B_{ph}|")
ax.legend()
plt.show()
```

## Next steps

- The
  [`packages/pydae-uds/src/pydae/uds/`](https://github.com/pydae/pydae/tree/main/packages/pydae-uds/src/pydae/uds)
  directory contains example notebooks for VSCs, BESS, SOFC, PEMFC, PV and
  line / transformer models.
- To build balanced transmission models instead, see
  [pydae-bps getting started](https://pydae-bps.readthedocs.io/en/latest/getting_started.html).
- To drop down to hand-written DAEs, see
  [pydae-core getting started](https://pydae-core.readthedocs.io/en/latest/getting_started.html).
