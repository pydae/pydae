# sst1

*Automatic voltage regulators — pydae-bps model.*

## Model description

ST1-style static excitation system with PV-bus initialisation.

The model implements a lightweight IEEE ST1 / NTSST1 variant: a
first-order terminal-voltage sensor, a lead-lag voltage regulator, a
proportional main-regulator gain, a hard field-voltage limiter, and a
first-order output filter. There are three dynamic states and no
algebraic output — the field-voltage command ``v_f`` is filtered, not
algebraic. Initialisation is handled by the ``ini``/``run`` variable
swap described below; no artificial ``xi_v`` integrator is used.

**Signal path**

The sensed terminal voltage follows $V$ with a first-order lag of time
constant $T_r$:

$$\frac{d v_r}{dt} = \frac{V - v_r}{T_r}$$

(The sensed state ``v_r`` is exported via ``h_dict`` for diagnostics;
the summing junction uses the instantaneous $V$, matching the
historical ST1 implementation in this codebase.)

The voltage error feeds a lead-lag $(1 + s T_c)/(1 + s T_b)$ with
internal state $x_{cb}$:

$$v_1 = v^{\star} - V + v_s$$
$$\frac{d x_{cb}}{dt} = \frac{v_1 - x_{cb}}{T_b}$$
$$z_{cb} = (v_1 - x_{cb})\frac{T_c}{T_b} + x_{cb}$$

The proportional gain $K_a$ produces the unsaturated field command,
which is passed through a hard limiter:

$$e_{fd}^{nosat} = K_a \, z_{cb}$$
$$e_{fd} = \mathrm{sat}\!\left(e_{fd}^{nosat},\; V_{fmin},\; V_{fmax}\right)$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

The output filter with time constant $T_r$ produces the field-voltage
command seen downstream:

$$\frac{d v_f}{dt} = \frac{e_{fd} - v_f}{T_r}$$

(The original ST1 implementation reuses $T_r$ for both the sensor lag
and the output filter; that choice is preserved here.)

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
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

List cardinalities are preserved: ``|y_ini| == |y_run|`` and ``|g|`` is
unchanged. The swap is performed in place at the existing ``y_ini``
index of ``V_bus`` (added earlier by the bus builder) so that
downstream components that reference ``y_ini`` by integer index — for
example ``vsource``'s ``g[idx_V] = ...`` override — continue to target
the correct equations. Because ``v_f`` is a dynamic state (not
algebraic), this module contributes no new entries to ``g`` / ``y``
beyond the swap itself.

This replaces the earlier ``xi_v`` dummy-integrator approach (a fake
state with $d\xi_v/dt = v^{\star} - V$ and a tiny gain $K_{ai}$) and
the ``k_sat`` ini/run saturation blend. The swap is cleaner (one fewer
state, one fewer input, no spurious near-zero eigenvalue) and strictly
equivalent at the steady state.

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

    "avr": {"type": "sst1", "K_a": 200.0, "T_r": 0.02, "T_b": 10.0,
            "T_c": 1.0, "V_f_min": -100.0, "V_f_max": 100.0,
            "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.
``V_f_min`` / ``V_f_max`` default to ±100 (effectively unbounded) to
match the original ``sst1`` defaults.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `sst1` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K_a$ | `K_a` | 200.0 | pu | AVR main proportional gain |
| $T_r$ | `T_r` | 0.02 | s | Terminal-voltage sensor and output-filter time constant (shared) |
| $T_b$ | `T_b` | 10.0 | s | Lead-lag denominator time constant |
| $T_c$ | `T_c` | 1.0 | s | Lead-lag numerator time constant |
| $V_{fmin}$ | `V_f_min` | -100.0 | pu | Lower field-voltage limit |
| $V_{fmax}$ | `V_f_max` | 100.0 | pu | Upper field-voltage limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^{\star}$ | `v_ref` | 1.0 | pu | Voltage reference. Acts as the PV-bus setpoint during ini (where it is solved for) and as an input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output), added to the voltage error. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_r$ | `v_r` |  | pu | Sensed terminal voltage (first-order lag output, diagnostic only) |
| $x_{cb}$ | `x_cb` |  | pu | Lead-lag internal state |
| $v_f$ | `v_f` |  | pu | Filtered field-voltage command sent to the synchronous machine exciter (post-saturation). |


## Source

- Module: `pydae.bps.avrs.sst1`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/sst1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/sst1.py)
