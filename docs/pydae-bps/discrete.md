# Discrete control orchestration

A pydae model captures a system's **continuous** dynamics. Many grid-following
elements also carry a small **discrete** control part that must run once per
solver step — reading the current operating point and rewriting runtime inputs
before the next integration. The canonical example is the grid-code
fault-ride-through relay {func}`~pydae.bps.utils.frt.vrt`, which switches a
converter between PQ and current-injection mode based on its terminal voltage.

The solver driving the model — the offline `model.run` loop, or the real-time
API background thread — should not have to know *which* elements have a discrete
part or *how* to invoke them. It needs one generic hook it can call every step.
{class}`~pydae.bps.discrete.DiscreteOrchestrator` is that hook.

```{admonition} Generic layer, particular behaviour
:class: note
The orchestrator is **generic**: it holds a list of controllers and, on
`step(model)`, calls each. The *particular* behaviour lives with each model as a
controller object (anything with `step(model)`, optionally `sync(model)`), built
by a module-level `make_discrete_controller(model)` factory. This mirrors the
{doc}`fault <faults>` design, where {class}`~pydae.bps.utils.faults.FaultManager`
is the generic numerics and the converter's `vrt`/`iq_po123` law is the
particular behaviour injected into it.
```

## Using the orchestrator

```python
from pydae.bps.discrete import DiscreteOrchestrator

discrete = DiscreteOrchestrator.from_model(model)   # auto-discovers controllers
for t in times:
    discrete.step(model)                    # run every element's discrete part
    model.run(t, {}, method="trapezoidal")
```

`from_model` probes a small list of known discrete-capable model modules
(`DEFAULT_DISCRETE_MODULES`) plus any factory registered with
{func}`~pydae.bps.discrete.register_discrete`, calls each factory with the
model, and keeps the non-`None` controllers. Discovery is **explicit and
side-effect-safe** — it does not blind-scan the model packages (which hold dev
and test scripts that must not be imported).

One factory can cover several element types that share an interface: the shared
FRT factory in {mod}`pydae.bps.utils.frt` discovers every converter exposing
`lvrt_ext_<name>`, which both `pv_pq_ss` inverters and `bess_pq_ss` VSCs do, so a
single {class}`~pydae.bps.utils.frt.VrtController` drives them all. (`pv_pq_ss`
and `bess_pq_ss` re-export `vrt` / `VrtController` / `iq_po123`, so you can
import the relay from either model or from `pydae.bps.utils.frt` directly.)

## Latency tiers

Not every discrete part runs at the same rate. A local fast control (the FRT
relay) reacts every solver step; a centralized controller (a plant PPC) runs
only every ~200 ms. Each controller declares a **latency tier** and the
orchestrator runs it at that cadence — so the solver loop stays oblivious to
*how often* each part should fire.

| Tier | Default period | Typical use |
|---|---|---|
| `low` | `0.0` s (every step) | local fast control (FRT relay, `vrt`) |
| `mid` | `0.05` s | fast measurement / feeder-level control |
| `high` | `0.2` s | centralized controller (plant PPC) |

A controller's period (seconds) is resolved in precedence order:

1. an explicit `controller.period` (seconds), else
2. `tiers[controller.level]` for a named `controller.level`, else
3. `0.0` — **low**, every step (the default when neither attribute is set).

Cadence is expressed in **seconds** so it ports across the offline loop
(`model.Dt`) and the real-time loop (`chunk_sec`), whose step sizes differ. The
gate needs a clock, so pass the current simulation time to `step`:

```python
discrete = DiscreteOrchestrator.from_model(model)   # or tiers={...} to override
for t in times:
    discrete.step(model, t)              # gate each controller by its tier
    model.run(t, {}, method="trapezoidal")
```

If `t` is omitted it falls back to `model.t`; with no clock at all every
controller runs (ungated). Tiers gate **cadence, not order** — the controllers
due on a step still run in registration order. `step` returns a list aligned to
`controllers`, with `None` for the ones skipped this step. `reset_schedule()`
clears the timers so every tier fires again on the next step (useful right after
a fault or a mode change).

Override the periods per orchestrator:

```python
discrete = DiscreteOrchestrator.from_model(model, tiers={"low": 0.0,
                                                         "mid": 0.02,
                                                         "high": 0.1})
```

## With a fault

When a fault is applied through {class}`~pydae.bps.utils.faults.FaultManager`,
the continuation leaves the converters in a ride-through operating point. Call
`discrete.sync(model)` afterwards so each controller re-seeds its latched state
from the model and does not fight the continuation on the next step:

```python
from pydae.bps.utils.faults import FaultManager, lvrt_response
from pydae.bps.pvs.pv_pq_ss import iq_po123

fault    = FaultManager(model, "POI")
discrete = DiscreteOrchestrator.from_model(model)
converters = discrete.controllers[0].devices
frt      = lvrt_response(converters, v_lvrt=0.85, iq=iq_po123)

for t in times:
    if apply_now:
        fault.apply(on_step=frt); discrete.sync(model)
    if clear_now:
        fault.clear(on_step=frt); discrete.sync(model)
    discrete.step(model)
    model.run(t, {}, method="trapezoidal")
```

## In the real-time API

The `pydae-core` real-time API takes the same hook, duck-typed, on
`app.state.discrete`; its background loop calls `step(model)` once per chunk,
before integrating. See the
[real-time API guide](https://pydae-core.readthedocs.io/en/latest/realtime_api.html).

```python
app.state.model    = model
app.state.discrete = DiscreteOrchestrator.from_model(model)
```

## Writing a discrete controller

A controller is any object with:

- `step(model)` — run the discrete update once (read outputs, write inputs);
  return value is passed back through the orchestrator.
- `sync(model)` *(optional)* — re-seed internal state from the model.
- `level` *(optional)* — its latency tier (`"low"` / `"mid"` / `"high"`);
  and/or `period` *(optional, seconds)* to override the tier. Absent → `low`
  (every step).

Expose it from your model module through a factory that returns `None` when the
system has no such element, and register it:

```python
from pydae.bps.discrete import register_discrete

class MyController:
    level = "high"                                  # runs at the slow tier
    def __init__(self, devices): self.devices = devices
    def step(self, model): ...
    def sync(self, model): ...

@register_discrete
def make_discrete_controller(model, **kwargs):
    devices = [...discover from model...]
    return MyController(devices, **kwargs) if devices else None
```

Two concrete controllers ship in the tree, discovered via
`DEFAULT_DISCRETE_MODULES`: the `"low"` FRT relay
({class}`~pydae.bps.utils.frt.VrtController`, every step) and the `"high"`
open-loop plant dispatcher {class}`~pydae.bps.ppcs.sppc.SppcController` (~200 ms),
which broadcasts `p_poi_ref`/`q_poi_ref` to each converter's `p_s_ppc`/`q_s_ppc`.

## Python API

```{eval-rst}
.. autoclass:: pydae.bps.discrete.DiscreteOrchestrator
   :members:
   :undoc-members:

.. autofunction:: pydae.bps.discrete.register_discrete

.. autoclass:: pydae.bps.utils.frt.VrtController
   :members:
   :undoc-members:

.. autofunction:: pydae.bps.utils.frt.vrt

.. autofunction:: pydae.bps.utils.frt.iq_po123
```
