# Load Controller (LC)

*Per-machine active-power load controller — `pydae.bps.miscellaneous.load_controller`*

---

## Why the LC exists

Every synchronous machine has armature resistance $R_a$.  In steady state the
mechanical power $p_m$ and the grid injection $p_g$ satisfy

$$p_m = p_g + R_a\left(i_d^2 + i_q^2\right)$$

The loss term $R_a(i_d^2 + i_q^2)$ depends on the operating point and cannot
be pre-computed.  This means that a dispatch setpoint $p_c$ fed directly into
the governor's load reference gives

$$p_g = p_c - \underbrace{R_a(i_d^2 + i_q^2)}_{\approx 1\text{–}3\%} < p_c$$

The load controller (LC) is a slow integrator that drives the governor
(or $p_m$ when no governor is present) until the loss correction is absorbed
and $p_g = p_{c,lc}$ exactly.

---

## Mathematical model

The LC has one integrator state $x_{lc}$ and one algebraic equation:

$$\dot{x}_{lc} = K_{i,lc}\left(p_{c,lc} - p_g\right)$$

$$0 = x_{lc} + \Delta p_{lc} - p_{ctrl}$$

where $p_{ctrl}$ is the governor's load reference $p_c$ (or direct $p_m$ when
no governor is present), $p_{c,lc}$ is the user-specified desired grid
injection, and $\Delta p_{lc}$ is the fast additive channel (zero by default,
driven by AGC when present).

**At ini():** Newton–Raphson solves the full steady-state system including
$\dot{x}_{lc} = 0$, which forces $p_g = p_{c,lc}$ regardless of $K_{i,lc}$.
The solver finds $x_{lc}$ (and thus $p_{ctrl}$) that achieves this
automatically — no manual loss calculation needed.

**During run():** the integrator keeps $p_g \approx p_{c,lc}$ with time
constant $\tau_{lc} = 1/K_{i,lc}$.  With the default $K_{i,lc} = 0.01$ pu/s/pu
this gives $\tau_{lc} = 100$ s, which is well separated from governor and
swing dynamics.

---

## Fast additive channel for AGC

When the system-level [AGC](agc.md) is also present, a slow LC would delay
AGC signals by $\tau_{lc} \approx 100$ s.  To avoid this, the LC exposes a
**fast additive channel** `dp_lc`:

$$p_{ctrl} = \underbrace{x_{lc}}_{\text{slow base}} + \underbrace{\Delta p_{lc}}_{\text{fast AGC}}$$

AGC writes to $\Delta p_{lc}$ (bypassing the LC integrator entirely) so the
generator sees the full AGC step in seconds, not minutes.  Over time, $x_{lc}$
slowly absorbs the AGC increment, restoring $\Delta p_{lc} \to 0$ at the new
steady state.

```{mermaid}
flowchart LR
    PC["p_c_lc\n(desired p_g)"]
    INT["∫ K_i_lc\nx_lc"]
    DP["dp_lc\n(AGC fast)"]
    SUM("+")
    CTRL["p_ctrl\n(gov p_c or p_m)"]
    GOV["Governor /\nmachine"]
    PG["p_g"]

    PC -->|"error = p_c_lc − p_g"| INT
    INT -->|"x_lc"| SUM
    DP -->|"dp_lc"| SUM
    SUM --> CTRL
    CTRL --> GOV
    GOV --> PG
    PG -->|"feedback"| INT
```

---

## Timescale separation

| Controller | Timescale | Driven by |
|------------|-----------|-----------|
| Governor (primary) | 0.5 – 10 s | Frequency droop |
| AGC (secondary) | 1 – 120 s | Speed error $1 - \omega$ |
| LC (tertiary) | 60 – 600 s | Power error $p_{c,lc} - p_g$ |

The LC time constant ($\tau_{lc} \approx 100$ s) is at least one order of
magnitude larger than governor time constants and the swing period, so it does
not appear in eigenvalue analyses of inter-area or local modes and does not
affect fault-ride-through behaviour.

---

## Variables

| Symbol | ini phase | run phase | Description |
|--------|-----------|-----------|-------------|
| `x_lc_{name}` | state | state | Integrator output (= base setpoint) |
| `ctrl_sym` | algebraic | algebraic | Governor `p_c` or direct `p_m` |
| `p_c_lc_{name}` | input | input | Desired grid injection $p_g$ |
| `dp_lc_{name}` | input (0) | input | Fast additive channel for AGC |
| `K_i_lc_{name}` | parameter | parameter | Integrator gain (pu/s per pu) |

---

## HJSON configuration

Place `lc` after `gov` in the synchronous machine entry:

**Explicit `lc:` block** — full control over K_i:

```hjson
syns: [
  {
    bus: "1",
    ...,
    gov: {
      type: "tgov1",
      R: 0.05, T_1: 0.5,
      V_max: 1.0, V_min: 0.0,
      T_2: 2.1, T_3: 7.0,
      D_t: 0.0,
      p_c: 0.8           // initial guess only; overridden by LC
    },
    lc: {
      K_i: 0.01,         // slow integrator gain (τ = 100 s)
      p_c_lc: 0.8        // desired p_g in machine-base pu
    }
  }
]
```

**Bare `p_c_lc:` shorthand** — auto-creates `lc: {K_i: 0.001, p_c_lc: ...}`:

```hjson
syns: [
  {
    bus: "1",
    ...,
    gov: {type: "tgov1", ..., p_c: 0.8},
    p_c_lc: 0.8   // shorthand: desired p_g; LC added automatically with K_i=0.001
  }
]
```

`lc.p_c_lc` (or bare `p_c_lc`) is the **desired grid injection** $p_g$, not $p_m$.
`gov.p_c` serves as an initial guess and is replaced by the integrator.
If `p_m` is present instead of `p_c_lc`, the LC is **not** added and `p_m` is
used directly as the mechanical power input.

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

# p_c_lc_1 is the desired p_g for generator "1"
model.ini({"V_1": 1.02, "p_c_lc_1": 0.8}, "xy_0.json")

# At ini() p_g_1 ≈ 0.8 pu exactly; p_m_1 > 0.8 (armature losses included)
print(model.get_value("p_g_1"))   # → 0.800
print(model.get_value("p_m_1"))   # → 0.808 (example)

# Change dispatch setpoint
model.run(10.0, {})
model.run(610.0, {"p_c_lc_1": 0.6})   # LC settles in ~100 s
model.post()
```

---

## Without a governor

When no `gov` key is present the LC wraps `p_m` directly.
Use either the explicit block or the bare shorthand:

```hjson
// explicit block
syns: [{bus: "1", ..., lc: {K_i: 0.01, p_c_lc: 0.5}}]

// bare shorthand (K_i defaults to 0.001)
syns: [{bus: "1", ..., p_c_lc: 0.5}]
```

The integrator drives `p_m_{name}` so that $p_g = p_{c,lc}$ at steady state.
The `dp_lc_{name}` fast channel remains available for external setpoint
increments via `model.run(t, {"dp_lc_1": delta_p})`.

---

## Source

- Module: `pydae.bps.miscellaneous.load_controller`
- File: [`packages/pydae-bps/src/pydae/bps/miscellaneous/load_controller.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/miscellaneous/load_controller.py)

```{eval-rst}
.. autofunction:: pydae.bps.miscellaneous.load_controller.add_lc
```
