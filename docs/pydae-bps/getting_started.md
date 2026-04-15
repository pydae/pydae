# Getting started

This walkthrough builds a two-bus system with a 4th-order Milano synchronous
machine, compiles it through `pydae-core`, and runs a 1 s steady-state
simulation.

## 1. Install

```bash
pip install pydae-bps
```

This also installs `pydae` (the core solver). A working C compiler must be
available.

## 2. Describe the network

Save the following as `test_milano4ord.json`:

```json
{
  "system": {
    "name": "test_milano4ord",
    "S_base": 100000000.0,
    "K_p_agc": 0.0, "K_i_agc": 0.0, "K_xif": 0.01
  },
  "buses": [
    {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
    {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0}
  ],
  "lines": [
    {"bus_j": "1", "bus_k": "2", "X_pu": 0.05, "R_pu": 0.0, "Bs_pu": 0.0, "S_mva": 100.0}
  ],
  "syns": [
    {"bus": "1", "S_n": 100e6, "X_d": 1.8, "X1d": 0.3, "T1d0": 8.0,
     "X_q": 1.7, "X1q": 0.55, "T1q0": 0.4,
     "R_a": 0.0025, "X_l": 0.2, "H": 5.0, "D": 1.0,
     "Omega_b": 314.159265, "omega_s": 1.0,
     "model": "milano4ord"}
  ],
  "avrs": [{"bus": "1", "type": "sexs", "K_a": 100.0, "T_a": 0.1,
            "T_b": 10.0, "T_e": 0.1, "E_min": -5.0, "E_max": 5.0}],
  "govs": [{"bus": "1", "type": "hygov", "R_perm": 0.05, "R_temp": 0.3,
            "T_f": 0.1, "T_r": 5.0, "T_g": 0.5, "T_w": 1.0,
            "G_min": 0.0, "G_max": 1.0}]
}
```

## 3. Build the DAE system

```python
from pydae.bps import BpsBuilder
from pydae.core import Builder, Model

grid = BpsBuilder("test_milano4ord.json")
grid.checker()
grid.uz_jacs = False
grid.construct("test_milano4ord")     # populates grid.sys_dict

bld = Builder(grid.sys_dict, target="ctypes")
bld.build()
```

## 4. Initialise and simulate

```python
model = Model("test_milano4ord")

model.ini(
    {"p_m_1": 0.5, "e1q_1": 1.2, "v_ref_1": 1.0},
    xy_0=1.0,
)

model.run(1.0, {})
model.post()
```

## 5. Inspect results

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(model.Time, model.get_values("omega_1"), label="omega_1 [pu]")
ax[0].legend()
ax[1].plot(model.Time, model.get_values("V_1"), label="|V_1| [pu]")
ax[1].legend()
plt.show()
```

## Next steps

- The [`packages/pydae-bps/src/pydae/bps/syns/`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns)
  directory contains example notebooks for each Milano order.
- Try swapping the synchronous machine for a virtual synchronous generator
  (`bps.vsgs`) or a grid-forming VSC (`bps.vscs`).
- Drop down to the underlying solver:
  [pydae-core getting started](https://pydae-core.readthedocs.io/en/latest/getting_started.html).
