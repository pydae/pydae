# avr_1

*Automatic voltage regulators — pydae-bps model.*

## Model description

Pure-algebraic automatic voltage regulator with PV-bus behaviour.

The controller is a single static-gain equation with no internal state and
no output limits. The field-voltage command $v_f$ is driven from the
voltage error at the regulated bus:

$$v_f = K_a \left( v^{\star} - V + v_s \right)$$

written as the residual

$$0 = v_f - K_a \left( v^{\star} - V + v_s \right)$$

where $V$ is the terminal (or a remote) bus voltage magnitude,
$v^{\star}$ is the voltage reference, $v_s$ is a supplementary PSS
input, and $v_f$ is the field-voltage command fed to the synchronous
machine. The large gain $K_a$ provides effectively stiff voltage control.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialization
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
index of ``V_bus`` (added earlier by the bus builder) so that downstream
components that reference ``y_ini`` by integer index — for example
``vsource``'s ``g[idx_V] = ...`` override — continue to target the
correct equations.

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

    "avr": {"type": "avr_1", "K_a": 100.0, "v_ref": 1.0, "bus": "2"}

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

The `avr_1` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_a$ | `K_a` | 100.0 | pu | AVR proportional gain |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^{\star}$ | `v_ref` | 1.0 | pu | Voltage reference. Acts as the PV-bus setpoint during ini (where it is solved for) and as an input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output), added to the voltage error. |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_f$ | `v_f` |  | pu | Field-voltage command sent to the synchronous machine exciter. |


## Source

- Module: `pydae.bps.avrs.avr_1`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/avr_1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/avr_1.py)
