# Automatic Generation Control (AGC)

*System-level secondary frequency controller — `pydae.bps.miscellaneous.agc`*

---

## Role in the control hierarchy

AGC is the **secondary** frequency controller.  After the governor (primary)
arrests a frequency excursion, a residual steady-state error remains because
the droop characteristic trades frequency deviation for load sharing.  AGC
eliminates this error by integrating the speed deviation and adjusting one
generator's active-power setpoint until $\omega \to 1$ pu.

In practice, AGC represents the plant-level Automatic Generation Control or a
simplified Area Control Error (ACE) loop used in interconnected-system studies.

---

## Mathematical model

Let $\omega_{gen}$ be the rotor speed of the designated generator and
$p_{ctrl}$ the variable AGC drives.  The controller is a PI on the speed
error $\varepsilon = 1 - \omega_{gen}$:

$$\dot{\xi}_{agc} = 1 - \omega_{gen}$$

$$0 = -p_{ctrl} + K_p\left(1 - \omega_{gen}\right) + K_i\,\xi_{agc}$$

**At ini():** $\dot{\xi}_{agc} = 0$ is trivially satisfied when $\omega = 1$,
so Newton–Raphson determines $p_{ctrl}$ from the network power balance and
$\xi_{agc} = p_{ctrl}/K_i$ follows.  No prior knowledge of the operating
point is needed.

**At steady state (run):**

$$\omega_{gen} = 1, \quad p_{ctrl} = K_i\,\xi_{agc}$$

The integral term holds the correct power setpoint indefinitely even as
network impedances or loads change.

A pure-integral setting ($K_p = 0$, default) is common in academic studies.
Adding a small proportional term ($K_p > 0$) improves damping during large
frequency excursions but introduces a residual at intermediate timescales.

---

## Control variable priority

``add_agc`` is called **last** in ``BpsBuilder.construct``, after all
synchronous machines, governors, and load controllers have been built.  It
inspects ``u_ini_dict`` and resolves the controlled variable in this order:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Key found in `u_ini_dict`
  - What AGC drives

* - `dp_lc_{gen}`
  - **LC fast channel** (preferred when LC is present).
    The AGC signal bypasses the slow LC integrator and reaches the
    governor/machine at the AGC timescale.

* - `p_c_{gen}`
  - Governor load reference (governor present, no LC).

* - `p_m_{gen}`
  - Direct mechanical power (no governor, no LC).
```

---

## Interaction with the Load Controller

When both LC and AGC are active on the same machine, the net
governor/machine setpoint is:

$$p_{ctrl} = x_{lc} + \Delta p_{lc}$$

AGC drives $\Delta p_{lc}$ (fast channel, $\tau_{agc} \approx 1$–30 s), and
the LC integrator drives $x_{lc}$ (slow channel, $\tau_{lc} \approx 100$ s).

```{mermaid}
sequenceDiagram
    participant DIST as Disturbance
    participant GOV as Governor
    participant AGC as AGC (dp_lc)
    participant LC as Load Controller (x_lc)

    DIST->>GOV: Load increases → ω drops
    GOV->>GOV: Droop: p_m ↑ partially (seconds)
    Note over GOV: ω settles at ω < 1 (droop error)
    AGC->>GOV: dp_lc ↑ until ω → 1 (1–2 min)
    Note over AGC: dp_lc now non-zero
    LC->>GOV: x_lc slowly absorbs dp_lc (10–30 min)
    Note over LC: dp_lc → 0, x_lc holds new dispatch
```

The three stages correspond to the classical three-level frequency restoration
process described in grid codes:

1. **Primary** (governor, <30 s): arrests the frequency nadir.
2. **Secondary** (AGC, <15 min): restores frequency to nominal.
3. **Tertiary** (LC / economic dispatch, >15 min): restores the
   loss-compensated dispatch setpoint and frees the secondary reserve.

---

## Variables added to the DAE

| Symbol | ini | run | Description |
|--------|-----|-----|-------------|
| `xi_agc` | state | state | PI integrator state |
| `ctrl_sym` | algebraic | algebraic | Controlled variable (dp_lc, p_c, or p_m) |
| `K_p_agc` | parameter | parameter | Proportional gain |
| `K_i_agc` | parameter | parameter | Integral gain |

---

## HJSON configuration

System-level key at the top of the data dict (not inside a `syns` entry):

```hjson
{
  system: {name: "two_gen", S_base: 100e6},
  buses:  [...],
  lines:  [...],
  syns: [
    {bus: "1", ...,
     gov: {type: "tgov1", ..., p_c: 0.8},
     lc:  {K_i: 0.01, p_c: 0.8}},    // LC exposes dp_lc_1
    {bus: "2", ...,
     gov: {type: "tgov1", ..., p_c: 0.4}}
  ],
  agc: {gen: "1", K_p_agc: 0.0, K_i_agc: 2.0}
  // AGC connects to dp_lc_1 because LC is present on gen "1"
}
```

`gen` must match the generator's `name` field (or bus name when no `name` is
given).  AGC acts on a **single** generator; in a multi-machine system the
other machines follow via the network and their own governors.

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
model.ini({"V_1": 1.02, "p_c_lc_1": 0.8}, "xy_0.json")

# Simulate a load step on bus 2
model.run(5.0, {})
model.run(305.0, {"P_2": 80e6})   # +80 MW load
model.post()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(model.Time, model.get_values("omega_1"), label="ω gen 1")
axes[0].set_ylabel("Speed (pu)")
axes[1].plot(model.Time, model.get_values("p_g_1"),  label="p_g gen 1")
axes[1].plot(model.Time, model.get_values("xi_agc"), label="ξ_agc")
axes[1].set_ylabel("Power (pu)")
for ax in axes:
    ax.legend(); ax.grid(True)
fig.savefig("agc_response.svg")
```

Expected response:

- $\omega$ drops at $t = 5$ s (load step), governor arrests it.
- AGC drives `dp_lc_1` upward; $\omega$ recovers to 1 pu in 1–5 min
  (depending on $K_i$).
- If LC is present, `x_lc_1` slowly absorbs `dp_lc_1` over the next
  10–30 min.

---

## Tuning guidelines

| Parameter | Typical range | Effect |
|-----------|--------------|--------|
| `K_i_agc` | 0.5 – 10 | Higher = faster frequency restoration, more oscillation |
| `K_p_agc` | 0 – 2 | Adds damping; too large causes overshoot |

For stability, the AGC closed-loop bandwidth must be well below the governor
bandwidth.  A conservative rule of thumb: set the AGC integral time constant
$T_{i,agc} = 1/K_{i,agc}$ to at least 5 times the governor lag $T_3$.

---

## Source

- Module: `pydae.bps.miscellaneous.agc`
- File: [`packages/pydae-bps/src/pydae/bps/miscellaneous/agc.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/miscellaneous/agc.py)

```{eval-rst}
.. autofunction:: pydae.bps.miscellaneous.agc.add_agc
```
