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
| **Outputs** `z = h(x, y, u, p)` | `h_dict` in `sys_dict` | `GET /measurements`, `POST /measurements` |

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
- `measurements()` / `cosim_measurements()` — output reads

The lock is held only for the minimum critical section. The pacing sleep
happens **outside** the lock so that HTTP handlers are never blocked waiting
for `time.sleep()` to return.

### Setpoint ramping

Applying a large setpoint step instantaneously can cause IDAS
(`IDA_LINESEARCH_FAIL`) to fail to find consistent algebraic initial
conditions for the new operating point.  To prevent this, every setpoint
change is spread evenly over `ramp_chunks` integration chunks (default 20).

With `chunk_ms = 50` ms and `ramp_chunks = 20` the ramp duration is 1 s.
Tune `ramp_chunks` to match the dynamics of your model:

```python
app.state.ramp_chunks = 10   # faster ramp: 10 × chunk_ms
```

If a step still fails (e.g. model instability), the background thread logs the
error and continues from the next chunk rather than crashing.

### CORS

`CORSMiddleware` is enabled with `allow_origins=["*"]` so the API is
accessible from browsers and co-simulation clients running on any origin.

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
        Thread->>Thread: apply ramp step (if pending)
        Thread->>IDAS: integrator(x, y_run, p)
        IDAS-->>Thread: x_next, y_run_next
        Thread->>Thread: t_sim += chunk_sec
        Thread->>Lock: release
        Thread->>Thread: sleep(target - now)
    end

    Client->>API: POST /setpoints
    API->>Lock: acquire
    API->>API: enqueue ramp toward target values
    API->>Lock: release
    API-->>Client: {"status": "ok"}

    Client->>API: GET /measurements
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
app.state.model       = model
app.state.chunk_ms    = 50.0   # integration chunk in milliseconds
app.state.ramp_chunks = 20     # setpoint ramp duration in chunks

uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Co-simulation interface

When pydae is used as a power system simulator inside a co-simulation
framework (e.g. coupled with Mininet for communication-network emulation), the
co-simulation JSON can include a `configs` section that declares which
measurements and setpoints belong to each field-device type:

```json
{
  "configs": {
    "poi": {
      "measurements": [
        {"emec_name": "U_POI",           "emec_scale": 1.0,  "fdii": true},
        {"emec_name": "p_line_POI_POIHV","emec_scale": 1.0,  "fdii": true}
      ],
      "setpoints": []
    },
    "inv": {
      "measurements": [
        {"emec_prefix": "p_s",           "emec_scale": 1.0,  "fdii": true},
        {"emec_template": "p_line_<emec_id>_<emec_id>LV", "emec_scale": 1.0, "fdii": true}
      ],
      "setpoints": [
        {"emec_prefix": "p_s_ppc", "emec_scale": 3.33e-07, "fdii": true},
        {"emec_prefix": "q_s_ppc", "emec_scale": 3.33e-07, "fdii": true}
      ]
    }
  }
}
```

### Variable name resolution

Each entry in the `configs` section uses one of three patterns to reference
pydae variables:

| Pattern | Example | Resolves to |
|---|---|---|
| `emec_name` | `"U_POI"` | Exact h_dict key |
| `emec_prefix` | `"p_s"` | All h_dict keys starting with `p_s_` |
| `emec_template` | `"p_line_<emec_id>_<emec_id>LV"` | h_dict keys where both `<emec_id>` tokens are the same device id (e.g. `p_line_SS1_SS1LV`) |

Measurements are resolved against `h_dict`; setpoints against `u_run_names`.
Duplicates are removed preserving first-seen order.

### `emec_scale`

`emec_scale` is a unit-conversion factor defined by the field device
(Modbus register encoding). The API returns raw pydae values — apply
`emec_scale` on the client side:

- **Measurement** → `field_value = pydae_value * emec_scale`
- **Setpoint** → `pydae_value = field_value * emec_scale`

### Wiring the config

Pass the parsed `configs` dict to `app.state.cosim_config` before starting
the server.  When `BpsBuilder` is used, the full JSON is available at
`grid.data`:

```python
from pydae.bps import BpsBuilder
from pydae.api.realtime_api import app

grid = BpsBuilder('my_network.json', use_casadi=True)
grid.construct('my_network')

# ... build model, ini() ...

app.state.model        = model
app.state.cosim_config = grid.data.get("configs", {})

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## REST endpoints

### `GET /status`

Returns the simulator health and current simulation time.

**Response**

```json
{"is_running": true, "t_sim": 3.15}
```

---

### `GET /measurements`

Returns current values of **all** output variables declared in `h_dict`.
Browser-friendly — no request body needed.

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

### `POST /measurements`

Returns current values of a **selected subset** of `h_dict` outputs.
Names absent from `h_dict` are returned as `null`.  `_h_fn` is evaluated
once per request regardless of how many names are listed.

**Request body**

```json
{"names": ["E_p", "E_k", "theta"]}
```

**Response** — same structure as `GET /measurements`.

---

### `GET /cosim/measurements`

Returns current values of the measurements declared in the `configs` section
of the co-simulation JSON (populated via `app.state.cosim_config`).

Values are raw pydae h_dict values; apply `emec_scale` on the client if
field-unit conversion is needed.

Returns **501** if no co-simulation config was loaded.

**Response**

```json
{
  "data": {
    "U_POI":            20102.6,
    "p_line_POI_POIHV": 2987120.3,
    "p_s_SS1":          0.5,
    "p_s_SS2":          0.5
  },
  "t_sim": 197.5
}
```

---

### `POST /set_input`

Write a single runtime input.  The change is ramped over `ramp_chunks`
integration chunks.

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

Write several runtime inputs atomically.  All values are enqueued as a single
ramp so that the solver never sees a partially-updated input vector.

An optional `timestamp` field (wall-clock seconds from the sender) is echoed
back in the response and can be used by the co-simulation framework for
latency accounting.

**Request body**

```json
{
  "setpoints": {"f_x": 5.0, "K_d": 0.05},
  "timestamp": 1716500000.123
}
```

**Response**

```json
{
  "status": "ok",
  "setpoints": {"f_x": 5.0, "K_d": 0.05},
  "timestamp": 1716500000.123
}
```

---

### `POST /cosim/setpoints`

Write setpoints restricted to variables declared in `configs.*.setpoints`.
Unknown keys are rejected immediately with **422** so the co-simulation client
gets explicit feedback instead of silently writing into an unreachable
variable.

Values are in pydae units; apply `emec_scale` on the client side before
sending if your source is in field units.

Returns **501** if no co-simulation config was loaded.

**Request body** — same `SetpointsPayload` format as `POST /setpoints`:

```json
{
  "setpoints": {
    "p_s_ppc_SS1": 0.8,
    "p_s_ppc_SS2": 0.8
  },
  "timestamp": 1716500000.456
}
```

**422 response** when unknown keys are sent:

```json
{
  "detail": {
    "unknown_setpoints": ["bad_name"],
    "allowed": ["p_s_ppc_SS1", "p_s_ppc_SS2", "q_s_ppc_SS1", "q_s_ppc_SS2"]
  }
}
```

---

## `app.state` configuration reference

| Attribute | Type | Default | Description |
|---|---|---|---|
| `model` | `CasadiModel` | *required* | Initialised model (`ini()` already called) |
| `chunk_ms` | `float` | `50.0` | Integration chunk size in milliseconds |
| `ramp_chunks` | `int` | `20` | Number of chunks to spread each setpoint change over |
| `cosim_config` | `dict` | `{}` | Parsed `configs` section from the co-simulation JSON |

---

## Python API

```{eval-rst}
.. autoclass:: pydae.api.realtime_api.RealTimeSimulator
   :members: start, stop, set_input, setpoints, measurements,
             load_cosim_config, cosim_measurements
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
externally should be declared there.

### Why a fixed chunk size rather than variable step?

Variable-step integrators (like IDAS in its adaptive mode) produce steps of
unpredictable duration, making it impossible to target a wall-clock rate.
The fixed chunk maps directly to a fixed sleep duration, enabling the
drift-proof pacing scheme.  IDAS still uses adaptive sub-steps *within* the
chunk; the chunk is only the time interval handed to `ca.integrator()`.

### Tuning `chunk_ms`

| Value | Trade-off |
|---|---|
| **10–25 ms** | Finer time resolution, higher CPU overhead |
| **50 ms** (default) | Good balance for most power-system models |
| **100–200 ms** | Lower overhead for slow dynamics; coarser setpoint latency |

A chunk too small relative to the integrator's computation time will produce
continuous overrun warnings.  If that happens, increase `chunk_ms` or reduce
the model size.

### Why ramp setpoints?

IDAS computes consistent algebraic initial conditions (`calc_ic=True`) at the
start of each chunk.  A large instantaneous step in `u_run` makes the
Newton-based line search in `IDACalcIC` fail (`IDA_LINESEARCH_FAIL`) because
the algebraic variables from the previous step are far from the new equilibrium.
Spreading the change over many small increments keeps each step well within the
basin of convergence.
