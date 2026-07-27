# Shunt faults and converter ride-through

A **fault** in `pydae-bps` is a balanced shunt impedance switched onto a bus for
a short time. Declaring it in the network data creates the shunt; the helper
{mod}`pydae.bps.utils.faults` (`apply_fault`) drives the whole event — apply,
hold, clear — while managing the low-voltage ride-through (LVRT) of every
grid-following converter in the plant.

```{admonition} Why a fault is not just one solve
:class: note
Grid-following converters (`pv_pq_ss`, `bess_pq_ss`) are **constant power** in
normal operation, so their AC current $i_s \sim S/V$ diverges as the faulted
bus voltage collapses and the algebraic Jacobian goes singular. Applying a deep
fault is therefore a *continuation*, not a single Newton step, and the
converters generally have to switch to current injection to survive it.
```

## Declaring a fault

Add a `faults` section. Each entry is a shunt impedance $Z_f = R_{pu} + jX_{pu}$
(per unit on its own `S_mva` base) at a bus:

```hjson
faults: [
  {bus: "POI"},                                    // defaults: deep, low-R fault
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
switching target. You can keep ordinary always-on `shunts` and `faults` in the
same network; just don't declare both at the same bus (they share the
`g_shunt_<bus>` / `b_shunt_<bus>` parameters).
```

## Applying the fault

```python
from pydae.bps.utils.faults import apply_fault

model.ini({}, xy_0="xy_0.json")
model.Dt = 10e-3

model.run(1.0, {})                       # settle to fault inception
apply_fault(model, "POI", t_clear=0.2)   # apply -> hold 0.2 s -> clear
model.run(10.0, {})                      # post-fault recovery
model.post()
```

`apply_fault` assumes the model is already integrated up to fault inception. It
then:

1. **applies** the fault instantaneously — an algebraic continuation at frozen
   time that ramps `g_shunt_<bus>` / `b_shunt_<bus>` from 0 to $Y_f$ while the
   converters move into LVRT;
2. **holds** it for `t_clear` seconds (`model.run` is advanced);
3. **clears** it instantaneously — the reverse continuation, releasing the
   converters back to PQ.

The caller resumes the post-fault simulation afterwards. The function returns the
per-device hold command applied during the fault.

### Decoupled apply/clear ({class}`~pydae.bps.utils.faults.FaultManager`)

`apply_fault` is a thin wrapper over `FaultManager`, which splits the event into
separate `apply()` and `clear()` calls so **the caller owns the hold** — the
natural shape for event-driven simulation, variable fault durations, or
inspecting the plant mid-fault:

```python
from pydae.bps.utils.faults import FaultManager

fm = FaultManager(model, "POI")     # same options as apply_fault
model.run(1.0, {})

hold_cmd = fm.apply()               # apply the fault (frozen time)
model.run(1.2, hold_cmd)            # hold for as long as you like
fm.clear()                          # clear (frozen time)
model.run(10.0, {})
```

`apply()` returns the LVRT hold command to pass to `model.run` during the fault;
`clear()` returns the converters to PQ. The manager keeps its own fault target
and device list, so repeated apply/clear cycles are cheap.

### Signature

```python
apply_fault(model, bus_name, t_clear,
            lvrt_iq=0.5, lvrt_id=0.0, v_lvrt=0.5,
            lvrt_soft=False, lvrt_band=0.2, fault_steps=100, verbose=True)
```

| Argument | Default | Meaning |
|---|---|---|
| `lvrt_iq` | `0.5` | reactive current commanded in LVRT (pu, voltage-supporting) |
| `lvrt_id` | `0.0` | active current commanded in LVRT (pu) |
| `v_lvrt` | `0.5` | terminal-voltage threshold separating LVRT from PQ |
| `lvrt_soft` | `False` | use the continuous relay instead of the hard latched one |
| `lvrt_band` | `0.2` | voltage band of the soft relay |
| `fault_steps` | `100` | nominal continuation steps (adaptive) |

The converters are **auto-discovered** from the model: any unit exposing the
`lvrt_ext_<name>` / `i_sr_ref_<name>` / `i_sa_ref_<name>` interface participates,
so no PV/BESS bus list is passed in. Models without that interface are ignored
and the event degrades to a plain shunt fault.

## Low-voltage ride-through

In LVRT a converter blends from constant power to **bounded current injection**.
Setting $p_s = V\,i_{sa,ref}$ and $q_s = V\,i_{sr,ref}$ makes the AC current

$$ i_s = \frac{\sqrt{p_s^2 + q_s^2}}{V} = \sqrt{i_{sa,ref}^2 + i_{sr,ref}^2}, $$

which stays bounded as $V \to 0$ — unlike the constant-power current $S/V$ that
diverges. A positive `lvrt_iq` injects reactive (voltage-supporting) current.
The blend is driven by `lvrt_ext` $\in [0,1]$ per converter (see
{doc}`models/index` for `pv_pq_ss` / `bess_pq_ss`).

`apply_fault` decides each converter's `lvrt_ext` with a per-device relay keyed
on its **own** terminal voltage. Two strategies are available.

### Hard relay (default)

Every unit enters LVRT together with the fault; then any unit whose terminal has
recovered above `v_lvrt` latches back to PQ. The relay is **one-way**
(LVRT → PQ only), so each unit switches at most once — the pass is monotone and
cannot chatter, even when many terminals settle right at the threshold.

```
lvrt_ext = 1  if  V <= v_lvrt   (latched: never re-enters once it leaves)
         = 0  otherwise
```

### Soft relay (`lvrt_soft=True`)

Each unit's `lvrt_ext` follows a continuous ramp of its own undervoltage,

$$ \texttt{lvrt\_ext} = \mathrm{clip}\!\left(\frac{v_{lvrt} - V}{\texttt{lvrt\_band}},\, 0,\, 1\right), $$

solved as a damped fixed point inside each continuation step. Because there is no
mode discontinuity, units slide gradually between PQ and LVRT — which lets the
soft relay cross the *intermediate* fault-severity band (partial, mixed
`lvrt_ext`) that makes a perfectly symmetric plant fold the hard relay's
simultaneous switch.

```{admonition} Which relay to use
:class: tip
The hard relay is the robust default for shallow faults (no LVRT needed) and
deep faults (full LVRT). Reach for the soft relay when a fault leaves the plant
near the threshold and you want a graded, mixed response — or when the hard
relay reports a convergence failure in that band.
```

## How the continuation stays robust

The fault ramp uses an **adaptive** step: a step that fails to converge is rolled
back and halved, then grown again on success, so fine resolution is spent only
where the operating point is stiff (the large first admittance increment of a
deep fault). State is snapshotted so a failed branch can be abandoned cleanly —
for instance, the hard relay first tries to bring the *whole* plant into LVRT
together with the fault and, if that overvolts for a milder fault, rolls back and
applies the fault in plain PQ instead.

## Example

A complete fault study — settle, fault at the point of interconnection, recover —
and a deep-fault voltage trace:

```python
from pydae.bps import BpsBuilder
from pydae.core.builder import CasadiBuilder
from pydae.core.model import CasadiModel
from pydae.bps.utils.faults import apply_fault

grid = BpsBuilder("plant.hjson", use_casadi=True)   # plant.hjson has a faults: section
grid.construct("plant")
model = CasadiModel(CasadiBuilder(grid.sys_dict))

model.ini({}, xy_0="xy_0.json")
model.Dt = 10e-3

model.run(1.0, {})
mode = apply_fault(model, "POI", t_clear=0.2)   # 200 ms fault
model.run(10.0, {})
model.post()
# -> "LVRT relay: 14/14 devices in LVRT, 0 in PQ (threshold V = 0.5)"
```

## Notes and limitations

- `apply_fault` performs the fault as an idealised instantaneous switching event
  (frozen-time algebraic continuation), consistent with grid-following converter
  models that have no LVRT detection lag. Recovery is handled by the integrator
  in the post-fault `run`.
- For a true *bolted* fault ($V = 0$) there is no phasor solution; use a small
  `R_pu`/`X_pu` for a defined residual voltage instead.
- In a perfectly **symmetric** plant all converters cross the threshold at the
  same instant, which is the worst case for either relay; a realistic asymmetric
  plant desynchronises the crossings and converges far more easily.
```
