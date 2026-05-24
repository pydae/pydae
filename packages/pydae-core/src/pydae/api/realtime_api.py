"""Soft real-time DAE simulation with a FastAPI REST interface.

Architecture
------------
A single background daemon thread runs the IDAS integrator in fixed-size
*chunks* (default 50 ms of simulation time per chunk).  After each chunk the
thread sleeps until the absolute wall-clock target for the next chunk, so
accumulated timing error stays bounded rather than compounding.

The thread never appends to the model's ``Time`` / ``X`` / ``Y`` / ``Z``
lists.  Only the two "hot" NumPy arrays ``model.x`` and ``model.y_run`` are
updated in place, keeping memory consumption flat for arbitrarily long runs.

All shared state between the background thread and the FastAPI handlers is
protected by a single :class:`threading.Lock`.  Handlers acquire it only for
the minimum critical section — the integrator step itself, or a single array
read / write.

Requires the optional ``api`` extra::

    pip install pydae[api]

Usage
-----
::

    from pydae.api.realtime_api import app
    app.state.model    = my_casadi_model   # CasadiModel, already ini()-ed
    app.state.chunk_ms = 50.0             # optional, default 50 ms

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

REST endpoints
--------------
``GET  /measurements``
    All h_dict outputs (browser-friendly).
``POST /measurements``
    Selective h_dict outputs: ``{"names": ["E_p", "E_k"]}``
``GET  /cosim/measurements``
    Only the measurements declared in ``configs`` (co-simulation subset).
``POST /set_input``
    Write a single setpoint: ``{"name": "f_x", "value": 0.5}``
``POST /setpoints``
    Write multiple setpoints: ``{"setpoints": {"f_x": 0.5}, "timestamp": 1.0}``
``POST /cosim/setpoints``
    Write setpoints restricted to those declared in ``configs``; rejects unknown keys.
``GET  /status``
    Returns ``{"is_running": bool, "t_sim": float}``
"""

import re
import time
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


def _resolve_emec_names(entry: dict, candidate_names: set[str]) -> list[str]:
    """Resolve one config entry to matching pydae variable names.

    Handles the three patterns used in the ``configs`` section of a pydae
    co-simulation JSON:

    ``emec_name``
        Exact variable name — returned as-is if present in *candidate_names*.
    ``emec_prefix``
        Prefix: returns all candidates matching ``{prefix}_{suffix}``.
    ``emec_template``
        Template with one or more ``<emec_id>`` tokens.  The first token
        becomes a regex capture group; subsequent tokens become backreferences
        so that only names where all tokens resolve to the *same* device id
        are matched (e.g. ``p_line_<emec_id>_<emec_id>LV`` matches
        ``p_line_SS1_SS1LV`` but not ``p_line_SS1_SS2LV``).
    """
    if "emec_name" in entry:
        name = entry["emec_name"]
        return [name] if name in candidate_names else []

    if "emec_prefix" in entry:
        prefix = entry["emec_prefix"] + "_"
        return sorted(n for n in candidate_names if n.startswith(prefix))

    if "emec_template" in entry:
        template = entry["emec_template"]
        parts = template.split("<emec_id>")
        n_tokens = len(parts) - 1
        if n_tokens == 0:
            return [template] if template in candidate_names else []
        # Build regex: first token → capture group, rest → backreference.
        pattern = re.escape(parts[0])
        for i in range(n_tokens):
            pattern += "([A-Za-z0-9_]+)" if i == 0 else r"\1"
            pattern += re.escape(parts[i + 1])
        rx = re.compile(pattern)
        return sorted(n for n in candidate_names if rx.fullmatch(n))

    return []


class RealTimeSimulator:
    r"""Drift-proof real-time wrapper around :class:`~pydae.core.model.casadi_model.CasadiModel`.

    The loop body is::

        acquire lock
          t_next = t_sim + chunk_sec
          run one IDAS step: x, y_run ← integrator(x, y_run, p)
          t_sim = t_next
        release lock
        sleep until wall_t0 + t_sim          ← absolute target, not relative

    Using an absolute target means a slow step causes a *one-shot* lag that is
    printed as an overrun warning, not a permanent offset that grows forever.

    Parameters
    ----------
    model:
        An initialised ``CasadiModel`` (``ini()`` already called).  The
        simulator reads and writes ``model.x``, ``model.y_run``,
        ``model.p_vals``, and ``model.u_run_vals`` directly.
    chunk_ms:
        Wall-clock chunk size in milliseconds.  Smaller values give finer
        time resolution but more OS scheduler overhead.
    """

    def __init__(self, model, chunk_ms: float = 50.0, ramp_chunks: int = 20):
        self.model = model
        self.chunk_sec = chunk_ms / 1000.0
        self.t_sim: float = 0.0
        self._lock = threading.Lock()
        self.is_running: bool = False
        self._thread: threading.Thread | None = None
        self._integrator = None
        self._ramp_chunks = ramp_chunks
        self._ramp_queue: list[dict[str, float]] = []
        # Co-simulation variable lists — populated by load_cosim_config().
        self._cosim_meas_names: list[str] = []
        self._cosim_setp_names: list[str] = []

    # ── integrator setup ───────────────────────────────────────────────────

    def _build_integrator(self):
        """Build and cache the CasADi IDAS integrator sized to ``chunk_sec``.

        This is separate from ``model.integrator`` because the model's own
        integrator may have a different step size (``model.Dt``).  Building
        here avoids mutating the model object from the background thread.
        """
        import casadi as ca

        opts = {
            "calc_ic": True,        # let IDAS correct algebraic initial values
            "print_stats": False,
            "max_num_steps": 5000,
            "reltol": getattr(self.model, "_integrator_reltol", 1e-6),
            "abstol": getattr(self.model, "_integrator_abstol", 1e-5),
            "linear_solver": "csparse",
        }
        # The integrator is created for the fixed interval [0, chunk_sec].
        # Passing t0=0 and tf=chunk_sec means each call advances exactly one
        # chunk regardless of absolute simulation time.
        self._integrator = ca.integrator(
            "idas_rt", "idas", self.model.dae_dict, 0.0, self.chunk_sec, opts
        )

    # ── single integration step ────────────────────────────────────────────

    def _step(self):
        """Advance ``model.x`` and ``model.y_run`` by one chunk.

        CasADi's IDAS integrator expects the parameter vector to be the
        concatenation of physical parameters and runtime inputs, matching the
        ``dae_dict['p']`` symbol list built by ``CasadiBuilder``.

        Nothing is appended to ``model.Time``, ``model.X``, ``model.Y``, or
        ``model.Z`` — memory consumption is O(1) over the lifetime of the run.
        """
        # CasADi convention: p = [p_vals | u_run_vals]
        p_run_vec = np.concatenate((self.model.p_vals, self.model.u_run_vals))
        res = self._integrator(x0=self.model.x, z0=self.model.y_run, p=p_run_vec)
        # Write results back in place; no copies kept.
        self.model.x = np.array(res["xf"]).flatten()
        self.model.y_run = np.array(res["zf"]).flatten()

    # ── background simulation loop ─────────────────────────────────────────

    def _apply_next_ramp_step(self):
        """Pop and apply one chunk's worth of ramped setpoint deltas.

        Must be called with ``self._lock`` already held.
        """
        if self._ramp_queue:
            deltas = self._ramp_queue.pop(0)
            for name, delta in deltas.items():
                u_run_names = self.model.u_run_names
                if name in u_run_names:
                    self.model.u_run_vals[u_run_names.index(name)] += delta
                else:
                    self.model.set_value(name, self.model.get_value(name) + delta)

    def _sim_loop(self):
        """Main loop executed in the background daemon thread."""
        self._build_integrator()
        wall_t0 = time.perf_counter()

        while self.is_running:
            with self._lock:
                self._apply_next_ramp_step()
                t_next = self.t_sim + self.chunk_sec
                try:
                    self._step()
                    self.t_sim = t_next
                except RuntimeError as exc:
                    print(f"[RT step failed at t_sim={self.t_sim:.3f} s] {exc}")

            target_wall_time = wall_t0 + self.t_sim
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                overrun_ms = -sleep_time * 1000.0
                print(
                    f"[RT overrun] {overrun_ms:.2f} ms behind schedule"
                    f" at t_sim={self.t_sim:.3f} s"
                )

    # ── co-simulation config ───────────────────────────────────────────────

    def load_cosim_config(self, cosim_config: dict) -> None:
        """Parse a ``configs`` dict and build co-simulation variable lists.

        Call this before :meth:`start`.  The *cosim_config* is the value of
        the ``configs`` key in the co-simulation JSON (already parsed — pass
        ``data["configs"]``, not the file path).

        Each top-level key in *cosim_config* is a device type (e.g. ``"poi"``,
        ``"inv"``).  Each device type has ``measurements`` and ``setpoints``
        lists whose entries use ``emec_name``, ``emec_prefix``, or
        ``emec_template`` to reference pydae variables.

        Measurement names are resolved against ``h_dict``; setpoint names
        against ``u_run_names``.  Duplicates are removed preserving first-seen
        order so the response is stable across calls.
        """
        h_names: set[str] = set(
            getattr(self.model, "sys_dict", {}).get("h_dict", {}).keys()
        )
        u_names: set[str] = set(getattr(self.model, "u_run_names", []))

        seen_m: set[str] = set()
        seen_s: set[str] = set()
        meas: list[str] = []
        setp: list[str] = []

        for device_cfg in cosim_config.values():
            for entry in device_cfg.get("measurements", []):
                for name in _resolve_emec_names(entry, h_names):
                    if name not in seen_m:
                        seen_m.add(name)
                        meas.append(name)
            for entry in device_cfg.get("setpoints", []):
                for name in _resolve_emec_names(entry, u_names):
                    if name not in seen_s:
                        seen_s.add(name)
                        setp.append(name)

        self._cosim_meas_names = meas
        self._cosim_setp_names = setp

    def cosim_measurements(self) -> dict[str, float | None]:
        """Return current values for co-simulation measurements, thread-safely.

        Returns only the variables declared in the ``configs`` section of the
        co-simulation JSON (populated by :meth:`load_cosim_config`).  The
        values are raw pydae h_dict values — unit conversion (``emec_scale``)
        is the responsibility of the co-simulation client.
        """
        return self.measurements(self._cosim_meas_names)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        """Spawn the background daemon thread and start simulating.

        The thread is a *daemon* so that it does not prevent the Python
        process from exiting if the main thread finishes or raises.
        """
        self.is_running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the loop to exit and block until the thread joins.

        A 5-second timeout prevents deadlock if the integrator hangs; after
        the timeout the thread is abandoned (it is a daemon, so the process
        can still exit cleanly).
        """
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── thread-safe setpoint writers ──────────────────────────────────────

    def _write_one(self, name: str, value: float) -> None:
        """Write one value to ``u_run_vals`` or via ``model.set_value``.

        The caller **must** already hold ``self._lock``.  Keeping this as a
        separate helper allows both :meth:`set_input` and :meth:`setpoints`
        to reuse the same write logic without nested lock acquisitions.

        ``u_run_names`` is checked first because the ``in`` test on a plain
        list is O(n) but avoids an exception path for the common case where
        the name is a runtime input.
        """
        u_run_names = self.model.u_run_names
        if name in u_run_names:
            # Direct array index write — no Python overhead from set_value.
            self.model.u_run_vals[u_run_names.index(name)] = value
        else:
            # Delegate to the model for parameters, states, and algebraics.
            self.model.set_value(name, value)

    def _enqueue_ramp(self, targets: dict[str, float]) -> None:
        """Replace any pending ramp with a new one toward ``targets``.

        Must be called with ``self._lock`` already held.  Reads each
        variable's *current* value (including any in-progress ramp) and
        spreads the remaining delta evenly over ``_ramp_chunks`` chunks.
        """
        def _current(name: str) -> float:
            u_run_names = self.model.u_run_names
            if name in u_run_names:
                return float(self.model.u_run_vals[u_run_names.index(name)])
            return float(self.model.get_value(name))

        n = self._ramp_chunks
        deltas_per_chunk = {
            name: (target - _current(name)) / n
            for name, target in targets.items()
        }
        self._ramp_queue = [deltas_per_chunk.copy() for _ in range(n)]

    def set_input(self, name: str, value: float) -> None:
        """Write a single runtime input via a smooth ramp."""
        with self._lock:
            self._enqueue_ramp({name: value})

    def setpoints(self, setpoints_dict: dict[str, float]) -> None:
        """Write one or more runtime inputs via a smooth ramp."""
        with self._lock:
            self._enqueue_ramp(setpoints_dict)

    # ── thread-safe measurement reader ────────────────────────────────────

    def measurements(self, names: list[str]) -> dict[str, float | None]:
        r"""Return current values of h_dict output variables, thread-safely.

        Only variables explicitly declared in ``sys_dict['h_dict']``
        (the output map :math:`z = h(x, y, u, p)`) are accessible here.
        States, algebraics, and inputs are intentionally excluded — if a
        variable must be observable from outside, declare it in ``h_dict``.

        ``_h_fn`` is evaluated **once** per call regardless of how many names
        are requested, so the CasADi function-call overhead is paid once even
        when reading a large number of outputs simultaneously.

        Parameters
        ----------
        names:
            Output variable names to read.  Names absent from ``h_dict``
            return ``None`` without raising.

        Returns
        -------
        dict
            ``{name: float}`` for names in ``h_dict``,
            ``{name: None}`` for names not found.
        """
        with self._lock:
            h_fn = getattr(self.model, "_h_fn", None)
            h_names: list[str] = []
            h_vals: np.ndarray | None = None

            if h_fn is not None and "h_dict" in getattr(self.model, "sys_dict", {}):
                h_names = list(self.model.sys_dict["h_dict"].keys())
                # Same concatenation convention as _step().
                p_run_vec = np.concatenate((self.model.p_vals, self.model.u_run_vals))
                h_vals = np.array(
                    h_fn(self.model.x, self.model.y_run, p_run_vec)
                ).flatten()

            return {
                name: (
                    float(h_vals[h_names.index(name)])
                    if h_vals is not None and name in h_names
                    else None
                )
                for name in names
            }


# ── Pydantic request / response schemas ────────────────────────────────────────

class InputPayload(BaseModel):
    """Payload for ``POST /set_input`` — a single name/value pair."""

    name: str
    value: float




class SetpointsPayload(BaseModel):
    """Payload for ``POST /setpoints`` — one or more name/value pairs."""

    setpoints: dict[str, float]
    timestamp: float | None = None  # optional wall-clock time from the sender


class MeasurementsRequest(BaseModel):
    """Request body for ``POST /measurements`` — a list of output names."""

    names: list[str]


# ── Module-level simulator reference ──────────────────────────────────────────

# Populated by the lifespan context manager on server startup.
# FastAPI route handlers read this reference; it is never written from the
# background thread, so no lock is needed for the reference itself.
simulator: RealTimeSimulator | None = None


# ── Lifespan context manager ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the simulator on startup and cleanly stop it on shutdown.

    The model must be attached to ``app.state`` *before* the ASGI server
    starts (i.e., before this context manager runs), because ``uvicorn``
    calls the lifespan on the first request or immediately at boot, not
    lazily.
    """
    global simulator

    model = getattr(app.state, "model", None)
    if model is None:
        raise RuntimeError(
            "No model attached to app.state.  "
            "Set  app.state.model = <CasadiModel>  before starting the server."
        )

    chunk_ms: float = getattr(app.state, "chunk_ms", 50.0)
    ramp_chunks: int = getattr(app.state, "ramp_chunks", 20)
    simulator = RealTimeSimulator(model, chunk_ms=chunk_ms, ramp_chunks=ramp_chunks)

    cosim_config = getattr(app.state, "cosim_config", None)
    if cosim_config:
        simulator.load_cosim_config(cosim_config)

    simulator.start()

    yield  # server is live here

    simulator.stop()


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="pydae Real-Time API",
    version="1.0.0",
    description=(
        "Soft real-time DAE simulation backed by CasADi IDAS.  "
        "The simulator runs in a background thread; endpoints provide "
        "lock-safe access to setpoints and measurements."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/set_input", summary="Write a single setpoint")
def set_input(payload: InputPayload):
    """Write one runtime input or parameter by name.

    Use ``/setpoints`` when you need to update several inputs atomically.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    simulator.set_input(payload.name, payload.value)
    return {"status": "ok", "name": payload.name, "value": payload.value}


@app.post("/setpoints", summary="Write multiple setpoints atomically")
def setpoints(payload: SetpointsPayload):
    """Write several runtime inputs in a single lock acquisition.

    All values are applied before the next integration step sees them,
    so the solver never encounters a partially-updated input vector.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    simulator.setpoints(payload.setpoints)
    return {"status": "ok", "setpoints": payload.setpoints, "timestamp": payload.timestamp}


@app.post("/measurements", summary="Read h_dict output variables")
def measurements(request: MeasurementsRequest):
    """Return current values of output variables declared in ``h_dict``.

    Names absent from ``h_dict`` are returned as ``null``.
    The response also includes ``t_sim`` (simulation time in seconds).
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    data = simulator.measurements(request.names)
    return {"data": data, "t_sim": simulator.t_sim}


@app.get("/measurements", summary="Read all h_dict output variables")
def measurements_get():
    """Return current values of all output variables declared in ``h_dict``.

    Equivalent to ``POST /measurements`` with all h_dict names, but callable
    from a browser or any GET client without a request body.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    h_names: list[str] = []
    if hasattr(simulator.model, "sys_dict") and "h_dict" in simulator.model.sys_dict:
        h_names = list(simulator.model.sys_dict["h_dict"].keys())
    data = simulator.measurements(h_names)
    return {"data": data, "t_sim": simulator.t_sim}


@app.get("/cosim/measurements", summary="Co-simulation measurements (configs subset)")
def cosim_measurements_get():
    """Return current values for the measurements declared in the co-simulation config.

    Only variables listed in the ``configs.*.measurements`` section of the
    co-simulation JSON are returned.  Values are raw pydae h_dict values;
    apply ``emec_scale`` on the client side if field-unit conversion is needed.

    Requires ``app.state.cosim_config`` to be set before server startup.
    Returns 501 if no co-simulation config was loaded.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    if not simulator._cosim_meas_names:
        raise HTTPException(status_code=501, detail="No co-simulation config loaded")
    data = simulator.cosim_measurements()
    return {"data": data, "t_sim": simulator.t_sim}


@app.post("/cosim/setpoints", summary="Co-simulation setpoints (configs subset)")
def cosim_setpoints(payload: SetpointsPayload):
    """Write setpoints restricted to variables declared in the co-simulation config.

    Only keys listed in ``configs.*.setpoints`` are accepted.  Unknown keys
    are rejected with 422 so the co-simulation client gets immediate feedback
    instead of silently writing into an unreachable variable.

    Values are in pydae units; apply ``emec_scale`` on the client side before
    sending if your source is in field units.
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not running")
    if not simulator._cosim_setp_names:
        raise HTTPException(status_code=501, detail="No co-simulation config loaded")
    allowed = set(simulator._cosim_setp_names)
    unknown = [k for k in payload.setpoints if k not in allowed]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"unknown_setpoints": unknown, "allowed": sorted(allowed)},
        )
    simulator.setpoints(payload.setpoints)
    return {"status": "ok", "setpoints": payload.setpoints, "timestamp": payload.timestamp}


@app.get("/status", summary="Simulator health check")
def status():
    """Return whether the simulator is running and the current simulation time."""
    if simulator is None:
        return {"is_running": False, "t_sim": 0.0}
    return {"is_running": simulator.is_running, "t_sim": simulator.t_sim}


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("pydae.api.realtime_api:app", host="0.0.0.0", port=8000, reload=False)
