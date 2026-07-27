# Shunt faults and converter ride-through

A **fault** in `pydae-bps` is a balanced shunt impedance switched onto a bus for
a short time. Two layers cooperate to simulate one:

- {class}`~pydae.bps.utils.faults.FaultManager` — **generic**. It robustly ramps
  a bus's fault admittance in and out (adaptive continuation) and knows nothing
  about converters or grid codes.
- The **converter's own response** — *particular*. It is injected into
  `FaultManager` as an `on_step` callback, and applied again during the
  time-domain integration by a per-step relay ({func}`~pydae.bps.pvs.pv_pq_ss.vrt`).

```{admonition} Why a fault is not just one solve
:class: note
Grid-following converters (`pv_pq_ss`, `bess_pq_ss`) are **constant power** in
normal operation, so their AC current $i_s \sim S/V$ diverges as the faulted
bus voltage collapses and the algebraic Jacobian goes singular. Applying a deep
fault is therefore a *continuation*, not a single Newton step, and the
converters have to switch to bounded current injection to survive it.
```

## Declaring a fault

Add a `faults` section. Each entry is a shunt impedance $Z_f = R_{pu} + jX_{pu}$
(per unit on its own `S_mva` base) at a bus:

```hjson
faults: [
  {bus: "POI"},                                       // defaults: deep, low-R fault
  {bus: "MV01", R_pu: 0.002, X_pu: 0.02, S_mva: 100}  // shallower, explicit
]
```

| Field | Default | Meaning |
|---|---|---|
| `bus` | — | bus the fault is applied to (required) |
| `R_pu` | `0.0005` | fault resistance (pu on `S_mva`) |
| `X_pu` | `0.005` | fault reactance (pu on `S_mva`) |
| `S_mva` | `100` | base power for `R_pu`/`X_pu` |

The builder turns each fault into an **off** shunt at the bus (admittance
$\approx 0$, so the base case is unchanged) and exposes the parameters
`g_shunt_<bus>` / `b_shunt_<bus>`. It also records the fault's target admittance

$$ Y_f = G_f + jB_f = \frac{1}{R_{pu} + jX_{pu}} $$

(referred to the system base) in `model.sys_dict['faults']`. A small but nonzero
`R_pu` keeps $Y_f$ finite and gives a defined residual voltage, avoiding the
singular $V \to 0$ operating point of a bolted ($Z_f = 0$) fault.

```{admonition} A fault is a shunt
:class: tip
A `faults` entry is exactly a `shunts` entry that starts open and carries a
switching target. Keep ordinary always-on `shunts` and `faults` in the same
network; just don't declare both at the same bus (they share the
`g_shunt_<bus>` / `b_shunt_<bus>` parameters).
```

## Applying the fault

The one-call convenience wrapper — apply, hold `t_clear`, clear:

```python
from pydae.bps.utils.faults import apply_fault

model.ini({}, xy_0="xy_0.json")
model.Dt = 10e-3

model.run(1.0, {})                       # settle to fault inception
apply_fault(model, "POI", t_clear=0.2)   # apply -> hold 0.2 s -> clear
model.run(10.0, {})                      # post-fault recovery
model.post()
```

`apply_fault(model, bus_name, t_clear, lvrt_iq=0.5, lvrt_id=0.0, v_lvrt=0.85, fault_steps=100)`
auto-discovers the converters (any unit exposing `lvrt_ext_<name>` /
`i_sr_ref_<name>` / `i_sa_ref_<name>`), builds the standard
{func}`~pydae.bps.utils.faults.lvrt_response`, and drives `FaultManager`.
`lvrt_iq` may be a scalar **or a callable grid-code curve** `iq(V)`.

## The generic core: `FaultManager`

For event-driven simulation — variable fault duration, inspecting the plant
mid-fault, or a custom element response — use `FaultManager` directly. It splits
the event into `apply()` / `clear()` so **the caller owns the hold**, and takes
an `on_step` callback that it relaxes to a fixed point at each continuation step:

```python
from pydae.bps.utils.faults import FaultManager, lvrt_response

fault = FaultManager(model, "POI")                         # generic, no LVRT knowledge
response = lvrt_response(converters, v_lvrt=0.85, iq=0.5)  # the particular behaviour

model.run(1.0, {})
fault.apply(on_step=response)      # ramp Z_f in; relax `response` each step
model.run(1.2, response and {})    # hold as long as you like
fault.clear(on_step=response)      # ramp Z_f out
model.run(10.0, {})
```

`FaultManager` contributes only the numerics:

- **adaptive continuation** — a step that fails to converge is rolled back and
  halved, then grown on success, so fine resolution is spent only where the
  operating point is stiff (the large first admittance increment of a deep
  fault);
- **snapshot/restore** — a failed branch is abandoned cleanly;
- **per-step fixed-point relaxation** of the `on_step(model) -> max_change`
  callback, so the elements reach their response *through the continuation*
  rather than a time-domain ramp — which is exactly what lets deep faults
  converge.

Pass `on_step=None` for a system with no fault-sensitive elements (e.g.
synchronous-only): `FaultManager` just ramps the admittance.

## Low-voltage ride-through response

In LVRT a converter blends from constant power to **bounded current injection**.
Setting $p_s = V\,i_{sa,ref}$ and $q_s = V\,i_{sr,ref}$ makes the AC current

$$ i_s = \frac{\sqrt{p_s^2 + q_s^2}}{V} = \sqrt{i_{sa,ref}^2 + i_{sr,ref}^2}, $$

which stays bounded as $V \to 0$ — unlike the constant-power current $S/V$ that
diverges. The blend is driven by `lvrt_ext` $\in [0,1]$ per converter.

{func}`~pydae.bps.utils.faults.lvrt_response` builds the `on_step` callback:
each converter enters LVRT (`lvrt_ext` → 1) when its **own** terminal voltage is
at or below `v_lvrt`, injecting reactive current `iq` — a scalar or a callable
curve `iq(V)`:

```python
lvrt_response(devices, v_lvrt=0.85, iq=0.5, i_active=0.0, gain=0.5)
```

## The converter law lives with the model

The *particular* control — the grid-code reactive-current characteristic and the
discrete time-domain relay — belongs to the converter model, in
{mod}`pydae.bps.pvs.pv_pq_ss`:

- {func}`~pydae.bps.pvs.pv_pq_ss.iq_po123` — the Spanish REE **PO 12.3**
  reactive-current-vs-voltage curve: a dead band above `v_dead` (0.85), a linear
  ramp down to `v_full` (0.5), saturated at `iq_max` below it.
- {func}`~pydae.bps.pvs.pv_pq_ss.vrt` — the discrete FRT relay run **once per
  time step**: hysteresis (`v_enter`/`v_exit`) plus rate-limited commutation of
  both `lvrt_ext` and `i_sr_ref = iq_po123(V)`.

The **same** `iq_po123` law is applied two ways by two owners: relaxed to a
fixed point by `FaultManager` during the fault switch, and rate-limited by `vrt`
during the integration. Defining it once keeps them consistent:

```python
from pydae.bps.utils.faults import FaultManager, lvrt_response
from pydae.bps.pvs.pv_pq_ss import vrt, iq_po123

fault = FaultManager(model, "POI")
frt   = lvrt_response(converters, v_lvrt=0.85, iq=iq_po123)   # curve in the callback

# --- fault inception (once) ---
fault.apply(on_step=frt)

# --- every time step ---
vrt(model, target, level, iq_level, v_enter=0.85, v_exit=0.9)
model.run(t, {}, method="trapezoidal")

# --- fault clearing (once) ---
fault.clear(on_step=frt)
```

```{admonition} Why the curve must go through the continuation
:class: note
The reactive support that stabilises a deep fault (full reactive current from
many converters) can only be reached by the **continuation** — ramping it in
during fixed-step integration folds the solve, because the plant crosses a
bifurcation between constant-power and current-injection operating points. So
`FaultManager` relaxes `iq(V)` to a fixed point as it applies the fault; `vrt`
then only *maintains* it. Sweeping the fault depth this way traces the whole
PO 12.3 characteristic — full reactive for deep dips, tapering through the
linear band, dead band above ~0.85 — with every depth converging.
```

## Notes and limitations

- The fault is an idealised instantaneous switching event (frozen-time algebraic
  continuation), consistent with grid-following converter models that have no
  LVRT detection lag. Recovery is handled by the integrator in the post-fault
  `run`.
- For a true *bolted* fault ($V = 0$) there is no phasor solution; use a small
  `R_pu`/`X_pu` for a defined residual voltage instead.
- In a perfectly **symmetric** plant all converters cross a threshold at the
  same instant — the worst case for a discrete relay. The continuation-reached
  `iq(V)` response avoids it; a realistic asymmetric plant desynchronises the
  crossings and is easier still.
```
