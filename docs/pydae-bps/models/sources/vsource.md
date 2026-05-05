# vsource

*Sources — pydae-bps model.*

## Model description

Ideal voltage source — infinite-bus equivalent for pydae-bps.

A ``vsource`` connected to a bus pins that bus's voltage magnitude and
angle to their reference values, acting as a **stiff Thevenin source**
with zero internal impedance.  It is used to model:

- An **infinite bus** (slack node) in single-machine or multi-machine
  studies where one terminal is an ideal grid equivalent.
- The **New York system equivalent** in the IEEE 39-bus benchmark, where
  the New England area is connected to an external, very stiff grid.

**Auxiliar equations**

None.

**Dynamic equations**

$$\dot{V}_{dummy} = v_{ref} - V_{dummy}$$

A dummy dynamic state is added so that the state vector has a consistent
dimension.  This state has no physical meaning; it simply settles to
:math:`v_{ref}` at steady state.

**Algebraic equations**

$$0 = V_k - v_{ref}$$
$$0 = \theta_k - \theta_{ref}$$

These replace the standard network bus equations for bus :math:`k`,
treating :math:`V_k` and :math:`\theta_k` as fixed inputs rather than
unknown algebraic states.

**COI contribution**

The vsource contributes :math:`H = 10^6` s to the centre-of-inertia
(COI) computation, pinning :math:`\omega_{COI} \approx 1` pu and
establishing the absolute angle reference.

**Inputs (runtime settable)**

| Symbol | Default | Description | Units |
|---|---|---|---|
| ``v_ref_{name}`` | 1.0 pu | Bus voltage magnitude setpoint | pu |
| ``theta_ref_{name}`` | 0.0 rad | Bus voltage angle setpoint | rad |

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `vsource` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_{ref}$ | `v_ref` | 1.0 | pu | Bus voltage magnitude setpoint |
| $\theta_{ref}$ | `theta_ref` | 0.0 | rad | Bus voltage angle setpoint |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $V_{dummy}$ | `V_dummy` |  | pu | Dummy state for consistent vector sizing |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $V_{dummy}$ | `V_dummy_out` |  | pu | Dummy state output |


## Source

- Module: `pydae.bps.sources.vsource`
- File: [`packages/pydae-bps/src/pydae/bps/sources/vsource.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/sources/vsource.py)
