# kundur_tgr

*Automatic voltage regulators — pydae-bps model.*

## Model description

Kundur AVR with Transient Gain Reduction (TGR) and PV-bus initialisation.

The model is a classical high-gain proportional AVR with a first-order
terminal-voltage sensor and a TGR lead-lag compensator, following the
reference implementation in Kundur, *Power System Stability and
Control*, chapter 8. It has two dynamic states and one algebraic
equation, with no artificial initialisation integrator — the
initialisation is handled by the ``ini``/``run`` variable swap
described below.

**Signal path**

The terminal voltage is sensed through a first-order lag with time
constant $T_r$:

$$\frac{d v_r}{dt} = \frac{V - v_r}{T_r}$$

The voltage error and gain form the signal driving the TGR block:

$$u_{ab} = K_a \left( v^{\star} - v_r + v_s \right)$$

The TGR is a lead-lag $(1 + s T_a)/(1 + s T_b)$ with internal state
$x_{ab}$:

$$\frac{d x_{ab}}{dt} = \frac{u_{ab} - x_{ab}}{T_b}$$
$$z_{ab} = (u_{ab} - x_{ab})\frac{T_a}{T_b} + x_{ab}$$

Choosing $T_a \ll T_b$ reduces the high-frequency gain to $K_a T_a/T_b$
while preserving the steady-state gain $K_a$ — this is the *transient
gain reduction* that gives the model its name and improves
small-signal damping.

The field-voltage command is the TGR output, returned as the algebraic
variable $v_f$ via the residual

$$0 = z_{ab} - v_f$$

No limits are imposed — pair with a saturating exciter downstream if
hard field limits are needed.

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
state with $d\xi_v/dt = v^{\star} - v_r$ and a tiny gain $K_{ai}$) that
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

    "avr": {"type": "kundur_tgr", "K_a": 200.0, "T_r": 0.01,
            "T_a": 1.0, "T_b": 10.0, "v_ref": 1.0}

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

The `kundur_tgr` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_a$ | `K_a` | 200.0 | pu | AVR proportional gain |
| $T_r$ | `T_r` | 0.01 | s | Terminal-voltage sensor time constant |
| $T_a$ | `T_a` | 1.0 | s | TGR lead-lag numerator time constant |
| $T_b$ | `T_b` | 10.0 | s | TGR lead-lag denominator time constant |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^{\star}$ | `v_ref` | 1.0 | pu | Voltage reference. Acts as the PV-bus setpoint during ini (where it is solved for) and as an input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output), added to the voltage error. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_r$ | `v_r` |  | pu | Sensed terminal voltage (first-order lag output) |
| $x_{ab}$ | `x_ab` |  | pu | TGR lead-lag internal state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_f$ | `v_f` |  | pu | Field-voltage command sent to the synchronous machine exciter. |


## Source

- Module: `pydae.bps.avrs.kundur_tgr`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/kundur_tgr.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/kundur_tgr.py)
