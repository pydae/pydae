# sexs

*Automatic voltage regulators — pydae-bps model.*

## Model description

SEXS simplified excitation system with PV-bus initialisation.

The model implements an IEEE-style *Simplified Excitation System* (SEXS):
a lead-lag voltage regulator feeding a first-order exciter with hard
field-voltage limits. The controller is written as two dynamic states
plus one algebraic equation, with no artificial initialisation
integrator — the initialisation is handled by the ``ini``/``run``
variable swap described below.

**Signal path**

The voltage error is

$$v_2 = v^{\star} - V + v_s$$

where $V$ is the terminal (or remote) bus voltage magnitude,
$v^{\star}$ is the voltage reference, and $v_s$ is the supplementary
PSS input. The error feeds a lead-lag block with state $x_{ab}$,

$$\frac{d x_{ab}}{dt} = \frac{v_2 - x_{ab}}{T_b}$$
$$z_{ab} = (v_2 - x_{ab}) \frac{T_a}{T_b} + x_{ab}$$

whose output drives a first-order exciter with state $x_e$ and gain
$K_a$,

$$\frac{d x_e}{dt} = \frac{K_a z_{ab} - x_e}{T_e}$$

The unsaturated field command is offset by one so that the operating
point sits near $v_f \approx 1$ at no-load:

$$e_{fd}^{nosat} = x_e + 1$$

The actual field voltage is produced by a hard limiter and returned as
the algebraic variable $v_f$ via the residual

$$0 = \mathrm{sat}\!\left(e_{fd}^{nosat},\; E_{min},\; E_{max}\right) - v_f$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialisation
problem is PV (active power and voltage magnitude specified, reactive
power and voltage reference unknown) while the subsequent time-domain
simulation is reference-driven (the reference is an input and the
voltage magnitude is solved from the network). ``pydae`` supports this
by allowing the ``y`` (algebraic) and ``u`` (input) partitions to
differ between the ``ini`` and ``run`` phases while sharing the same
set of residual equations $g$.

The table below shows how each quantity is classified in each phase:

                    ini                 run
    v_f             y_ini               y_run      (algebraic, always solved)
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

List cardinalities are preserved: ``|y_ini| == |y_run|`` and ``|g|`` is
unchanged. The swap is performed in place at the existing ``y_ini``
index of ``V_bus`` (added earlier by the bus builder) so that
downstream components that reference ``y_ini`` by integer index — for
example ``vsource``'s ``g[idx_V] = ...`` override — continue to target
the correct equations.

This replaces the earlier ``xi_v`` dummy-integrator approach (a fake
state with $d\xi_v/dt = v^{\star} - V$ and a tiny gain $K_{ai}$) that
was used to pin $V$ to $v^{\star}$ during ``ini``. The swap is cleaner
(one fewer state, one fewer equation, no spurious near-zero eigenvalue)
and strictly equivalent at the steady state.

**Value transfer between phases**

After ``ini()`` converges, ``ini2run()`` automatically copies solved
values to the run-phase state:

- ``v_ref`` is in ``y_ini`` and appears in ``u_run``: its solved value
  becomes the run-phase input, so the operating point is preserved
  without the user having to pass ``v_ref`` manually.
- ``V_bus`` is in ``u_ini`` and appears in ``y_run``: its pinned
  setpoint is used as the starting value of the run-phase solver.

**Configuration**

Example data entry::

    "avr": {"type": "sexs", "K_a": 200.0, "T_a": 0.015, "T_b": 10.0,
            "T_e": 0.1, "E_min": -5.0, "E_max": 5.0, "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `sexs` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_a$ | `K_a` | 200.0 | pu | AVR main gain |
| $T_a$ | `T_a` | 0.015 | s | Lead-lag numerator time constant |
| $T_b$ | `T_b` | 10.0 | s | Lead-lag denominator time constant |
| $T_e$ | `T_e` | 0.1 | s | Exciter time constant |
| $E_{min}$ | `E_min` | -5.0 | pu | Lower field-voltage limit |
| $E_{max}$ | `E_max` | 5.0 | pu | Upper field-voltage limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^{\star}$ | `v_ref` | 1.0 | pu | Voltage reference. Acts as the PV-bus setpoint during ini (where it is solved for) and as an input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output), added to the voltage error. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_{ab}$ | `x_ab` |  | pu | Lead-lag internal state |
| $x_e$ | `x_e` |  | pu | Exciter state — field command before the +1 offset and saturation |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_f$ | `v_f` |  | pu | Field-voltage command sent to the synchronous machine exciter (saturated). |


## Source

- Module: `pydae.bps.avrs.sexs`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/sexs.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/sexs.py)
