# vsource

*Sources — pydae-bps model.*

## Model description

Ideal voltage source — infinite-bus equivalent.

A `vsource` connected to a bus pins that bus's voltage magnitude $V$ and
angle $\theta$ to their reference values, acting as a stiff Thévenin
source with zero internal impedance.

**Typical applications**

- **Infinite bus** in single-machine or multi-machine stability studies.
- **NY system equivalent** in the IEEE 39-bus New England benchmark,
  where bus 39 connects to a large external grid.
- **Angle and frequency reference** when no synchronous machine is
  designated as slack.

---

## Equations

The bus $k$ equations are replaced with pin constraints:

$$0 = V_k - v_{ref}$$

$$0 = \theta_k - \theta_{ref}$$

A dummy dynamic state provides consistent vector sizing:

$$\dot{V}_{dummy} = v_{ref} - V_{dummy}$$

The vsource also contributes $H = 10^6$ s to the Centre-of-Inertia (COI)
computation, making it the dominant term and establishing the absolute
angle reference for all other machines in the network.

---

## Inputs

| Symbol | Variable | Default | Units | Description |
| --- | --- | --- | --- | --- |
| $v_{ref}$ | `v_ref_{name}` | 1.0 | pu | Bus voltage magnitude setpoint |
| $\theta_{ref}$ | `theta_ref_{name}` | 0.0 | rad | Bus voltage angle setpoint |

Both inputs can be changed at runtime via `model.ini()` or `model.run()`.

---

## HJSON configuration

Minimal (bus "6", V = 1 pu, θ = 0):

```hjson
sources: [{type: "vsource", bus: "6"}]
```

With explicit parameters (additional keys accepted for documentation but
currently not used — source is always ideal):

```hjson
sources: [{"type": "vsource", "bus": "39",
           "S_n": 10000e9, "F_n": 60,
           "X_v": 0.0001, "R_v": 0.0,
           "K_delta": 0.01, "K_alpha": 0.01}]
```

---

## Usage in Python

```python
from pydae.bps import BpsBuilder
from pydae.core import Builder, Model

grid = BpsBuilder("my_network.hjson")
grid.construct("my_system")

bld = Builder(grid.sys_dict, target="ctypes")
bld.build()

model = Model("my_system")

# Pin bus "39" at V = 1.03 pu, θ = 0 (default angle reference)
model.ini({"v_ref_39": 1.03}, "xy_0.json")

# Step the source voltage at t = 5 s
model.run(5.0, {})
model.run(30.0, {"v_ref_39": 1.05})
model.post()
```

---

## Notes

- **Zero impedance**: the source is ideal — $V_k$ and $\theta_k$ are
  pinned regardless of the power injected.  There is no droop or
  frequency-dependent voltage variation.
- **COI domination**: with $H = 10^6$ s the vsource essentially fixes
  $\omega_{COI} \approx 1$ pu, removing the need for an AGC loop when
  the bus represents an external infinite grid.
- **No AGC needed**: use a vsource as the angle reference instead of
  designating one generator as the AGC machine.  All remaining
  generators can then carry LC controllers.
- **Parameters in HJSON** (`S_n`, `F_n`, `X_v`, `R_v`, `K_delta`,
  `K_alpha`) are recorded for future extension (non-ideal source with
  finite impedance or droop) but are **not yet implemented**.

---

## Source

- Module: `pydae.bps.sources.vsource`
- File: [`packages/pydae-bps/src/pydae/bps/sources/vsource.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/sources/vsource.py)

```{eval-rst}
.. automodule:: pydae.bps.sources.vsource
   :no-members:
```
