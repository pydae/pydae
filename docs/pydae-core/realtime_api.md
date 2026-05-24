# Real-time simulation API

`pydae.api.realtime_api` wraps a
[CasadiModel](api.md#pydae.core.model.casadi_model.CasadiModel) in a
background thread that steps the IDAS integrator at a fixed wall-clock rate
and exposes the simulation state through a FastAPI REST interface.

The design targets *soft* real-time: timing errors are bounded but not
eliminated. A step that runs longer than the configured chunk prints a warning
and the scheduler self-corrects on the next iteration.

:::{note}
This module requires the optional `api` extra:

```bash
pip install pydae[api]
```
:::

---

## Concepts

### DAE outputs vs. states

pydae distinguishes three kinds of quantities at runtime:

| Kind | Where | Access from the API |
|---|---|---|
| **Differential states** `x` | `model.x` | — not exposed |
| **Algebraic variables** `y_run` | `model.y_run` | — not exposed |
| **Outputs** `z = h(x, y, u, p)` | `h_dict` in `sys_dict` | `POST /measurements` |

Only variables declared in `h_dict` are readable through `/measurements`.
This is intentional: the output map acts as a contract — if a signal must be
visible to external controllers or dashboards, it must be explicitly declared
as an output.

Setpoints (writable inputs) are the entries in `u_run_dict`.

### Drift-proof pacing

The simulation loop tracks an *absolute* wall-clock target, not a relative
sleep duration:

```
wall_t0 = time.perf_counter()          # captured once at thread start

while running:
    step()                              # IDAS integration
    t_sim += chunk_sec

    target = wall_t0 + t_sim           # absolute, not "now + dt"
    sleep_time = target - perf_counter()
    if sleep_time > 0:
        sleep(sleep_time)
    else:
        print(f"overrun {-sleep_time*1e3:.1f} ms")
```

If one step overshoots by Δ ms, the *next* target is still `wall_t0 + t_sim`.
The scheduler reclaims the Δ ms on the next cycle rather than letting the
error accumulate.

### Memory safety

The background thread never appends to `model.Time`, `model.X`, `model.Y`,
or `model.Z`. Only the two "hot" arrays `model.x` and `model.y_run` are
updated in place. Memory consumption is therefore O(1) regardless of how long
the simulation runs.

### Thread safety

A single `threading.Lock` protects all shared state:

- `_step()` — integrator call and array writes
- `set_input()` / `setpoints()` — input writes
- `measurements()` — output reads

The lock is held only for the minimum critical section. The pacing sleep
happens **outside** the lock so that HTTP handlers are never blocked waiting
for `time.sleep()` to return.

---

## Architecture

```{mermaid}
sequenceDiagram
    participant Client as HTTP Client
    participant API as FastAPI handlers
    participant Lock as threading.Lock
    participant Thread as Background Thread
    participant IDAS as CasADi IDAS

    loop every chunk_sec
        Thread->>Lock: acquire
        Thread->>IDAS: integrator(x, y_run, p)
        IDAS-->>Thread: x_next, y_run_next
        Thread->>Thread: t_sim += chunk_sec
        Thread->>Lock: release
        Thread->>Thread: sleep(target - now)
    end

    Client->>API: POST /setpoints
    API->>Lock: acquire
    API->>API: u_run_vals[i] = value
    API->>Lock: release
    API-->>Client: {"status": "ok"}

    Client->>API: POST /measurements
    API->>Lock: acquire
    API->>API: h_fn(x, y_run, p)  ← one call for all names
    API->>Lock: release
    API-->>Client: {"data": {...}, "t_sim": ...}
```

---

## Quick start

```python
import numpy as np
import casadi as ca
import uvicorn
from pydae.core.builder import CasadiBuilder, CasadiModel
from pydae.api.realtime_api import app

# ── 1. Build the model ──────────────────────────────────────────────────────
L, G, M, K_d, K_lam = [ca.SX.sym(n) for n in ["L", "G", "M", "K_d", "K_lam"]]
p_x, p_y, v_x, v_y  = [ca.SX.sym(n) for n in ["p_x", "p_y", "v_x", "v_y"]]
lam, f_x, theta      = [ca.SX.sym(n) for n in ["lam", "f_x", "theta"]]

sys_dict = {
    "name": "pendulum_rt",
    "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3, "K_lam": 1e-2},
    "f_list": [v_x, v_y,
               (-2*p_x*lam + f_x - K_d*v_x) / M,
               (-M*G - 2*p_y*lam - K_d*v_y) / M],
    "g_list": [p_x**2 + p_y**2 - L**2 - lam*K_lam,
               -theta + ca.atan2(p_x, -p_y)],
    "x_list": [p_x, p_y, v_x, v_y],
    "y_ini_list": [lam, f_x],
    "y_run_list": [lam, theta],
    "u_ini_dict": {"theta": np.deg2rad(10.0)},
    "u_run_dict": {"f_x": 0.0},
    # Only h_dict entries are readable via /measurements
    "h_dict": {
        "E_p":   M * G * (p_y + L),
        "E_k":   0.5 * M * (v_x**2 + v_y**2),
        "theta": theta,
    },
}

builder = CasadiBuilder(sys_dict).build()
model   = CasadiModel(builder)

# ── 2. Initialise ───────────────────────────────────────────────────────────
L_val, deg = 5.21, 10
model.ini(
    {"M": 10.0, "L": L_val, "K_lam": 1e-2, "theta": np.deg2rad(deg)},
    xy_0={
        "p_x": L_val * np.sin(np.deg2rad(deg)),
        "p_y": -L_val * np.cos(np.deg2rad(deg)),
        "lam": 50.0, "f_x": 0.0, "v_x": 0.0, "v_y": 0.0,
    },
)

# ── 3. Attach to the FastAPI app and serve ──────────────────────────────────
app.state.model    = model
app.state.chunk_ms = 50.0      # 50 ms chunks

uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## REST endpoints

### `GET /status`

Returns the simulator health and current simulation time.

**Response**

```json
{
  "is_running": true,
  "t_sim": 3.15
}
```

---

### `POST /set_input`

Write a single runtime input.

**Request body**

```json
{"name": "f_x", "value": 5.0}
```

**Response**

```json
{"status": "ok", "name": "f_x", "value": 5.0}
```

---

### `POST /setpoints`

Write several runtime inputs atomically in one lock acquisition.  All
values are applied before the next integration step, so the solver never sees
a partially-updated input vector.

**Request body**

```json
{
  "setpoints": {
    "f_x": 5.0,
    "K_d": 0.05
  }
}
```

**Response**

```json
{"status": "ok", "setpoints": {"f_x": 5.0, "K_d": 0.05}}
```

---

### `POST /measurements`

Read current values of output variables declared in `h_dict`.  Names absent
from `h_dict` are returned as `null`.  `_h_fn` is evaluated once per request
regardless of how many names are listed.

**Request body**

```json
{"names": ["E_p", "E_k", "theta"]}
```

**Response**

```json
{
  "data": {
    "E_p":   498.3,
    "E_k":   0.12,
    "theta": 0.174
  },
  "t_sim": 3.15
}
```

---

## Python API

```{eval-rst}
.. autoclass:: pydae.api.realtime_api.RealTimeSimulator
   :members: start, stop, set_input, setpoints, measurements
   :undoc-members:
   :show-inheritance:
```

### Pydantic schemas

```{eval-rst}
.. autopydantic_model:: pydae.api.realtime_api.InputPayload
   :inherited-members: BaseModel

.. autopydantic_model:: pydae.api.realtime_api.SetpointsPayload
   :inherited-members: BaseModel

.. autopydantic_model:: pydae.api.realtime_api.MeasurementsRequest
   :inherited-members: BaseModel
```

---

## Design notes

### Why only `h_dict` in measurements?

Exposing raw states and algebraic variables through the API would break
encapsulation: the internal variable layout (indices in `model.x`,
`model.y_run`) is an implementation detail of the model.  `h_dict` is the
explicit, user-defined output contract.  Any signal that must be observable
externally should be declared there, with a meaningful engineering name.

### Why a fixed chunk size rather than variable step?

Variable-step integrators (like IDAS in its adaptive mode) produce steps of
unpredictable duration, making it impossible to target a wall-clock rate.
The fixed chunk maps directly to a fixed sleep duration, enabling the
drift-proof pacing scheme.  IDAS still uses adaptive sub-steps *within* the
chunk; the chunk is only the time interval handed to `ca.integrator()`.

### Tuning `chunk_ms`

| Value | Trade-off |
|---|---|
| **10–25 ms** | Finer time resolution, higher CPU overhead (more integrator calls per second, more OS wakeups). |
| **50 ms** (default) | Good balance for most power-system models. |
| **100–200 ms** | Lower overhead for slow dynamics; coarser setpoint latency. |

A chunk that is too small relative to the integrator's computation time will
generate continuous overrun warnings.  If that happens, increase `chunk_ms`
or reduce the model size.
