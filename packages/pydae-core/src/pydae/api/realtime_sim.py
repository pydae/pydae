"""RealTimeSimulator — CasADi DAE soft real-time runner with no FastAPI dependency.

Wraps CasadiModel in a daemon thread paced to the wall clock.
Storage arrays (Time/X/Y/Z) are never accumulated, so memory stays flat for
days-long operation.
"""

import time
import threading

import numpy as np


class RealTimeSimulator:
    r"""Drift-proof real-time wrapper around CasadiModel.

    Runs the IDAS integrator in a daemon thread, stepping by ``chunk_sec``
    per iteration and sleeping to the absolute wall-clock target so that
    accumulated drift stays bounded (not compounded).

    Parameters
    ----------
    model:
        An initialised ``CasadiModel`` instance (``ini()`` already called).
    chunk_ms:
        Simulation chunk size in milliseconds (default 50 ms).
    """

    def __init__(self, model, chunk_ms: float = 50.0):
        self.model = model
        self.chunk_sec = chunk_ms / 1000.0
        self.t_sim = 0.0
        self._lock = threading.Lock()
        self.is_running = False
        self._thread: threading.Thread | None = None
        self._integrator = None

    # ── integrator setup ───────────────────────────────────────────────────

    def _build_integrator(self):
        """Build and cache the CasADi IDAS integrator sized to chunk_sec."""
        import casadi as ca

        opts = {
            "calc_ic": True,
            "print_stats": False,
            "max_num_steps": 5000,
            "reltol": getattr(self.model, "_integrator_reltol", 1e-6),
            "abstol": getattr(self.model, "_integrator_abstol", 1e-5),
            "linear_solver": "csparse",
        }
        self._integrator = ca.integrator(
            "idas_rt", "idas", self.model.dae_dict, 0.0, self.chunk_sec, opts
        )

    # ── simulation step ────────────────────────────────────────────────────

    def _step(self):
        """Advance the model by one chunk — no storage accumulation."""
        p_run_vec = np.concatenate((self.model.p_vals, self.model.u_run_vals))
        res = self._integrator(x0=self.model.x, z0=self.model.y_run, p=p_run_vec)
        self.model.x = np.array(res["xf"]).flatten()
        self.model.y_run = np.array(res["zf"]).flatten()

    # ── background loop ────────────────────────────────────────────────────

    def _sim_loop(self):
        self._build_integrator()
        wall_t0 = time.perf_counter()

        while self.is_running:
            with self._lock:
                t_next = self.t_sim + self.chunk_sec
                self._step()
                self.t_sim = t_next

            # Drift-proof pacing: target is absolute, not relative.
            target_wall_time = wall_t0 + self.t_sim
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                overrun_ms = -sleep_time * 1000.0
                print(
                    f"[RT overrun] {overrun_ms:.2f} ms behind schedule"
                    f" at t={self.t_sim:.3f} s"
                )

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        """Start the background simulation thread."""
        self.is_running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the loop to stop and wait for the thread to exit."""
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── data access (thread-safe) ──────────────────────────────────────────

    def set_input(self, name: str, value: float) -> None:
        """Set a run-time input by name, thread-safely.

        Targets ``u_run_vals`` directly for O(1) lookup; falls back to
        ``model.set_value`` for parameters, states, and algebraic variables.
        """
        with self._lock:
            u_run_names = self.model.u_run_names
            if name in u_run_names:
                self.model.u_run_vals[u_run_names.index(name)] = value
            else:
                self.model.set_value(name, value)

    def get_outputs(self, names: list[str]) -> dict[str, float | None]:
        """Return the current value of each requested variable, thread-safely.

        Output variables (defined in ``h_dict``) are evaluated via a single
        ``_h_fn`` call so CasADi overhead is paid once per request, not per
        variable.  States, algebraics, and inputs are read directly from the
        NumPy arrays.
        """
        with self._lock:
            result: dict[str, float | None] = {}

            # Evaluate all output (h_dict) variables in one shot.
            h_fn = getattr(self.model, "_h_fn", None)
            h_names: list[str] = []
            h_vals: np.ndarray | None = None
            if h_fn is not None and "h_dict" in getattr(self.model, "sys_dict", {}):
                h_names = list(self.model.sys_dict["h_dict"].keys())
                p_run_vec = np.concatenate((self.model.p_vals, self.model.u_run_vals))
                h_vals = np.array(
                    h_fn(self.model.x, self.model.y_run, p_run_vec)
                ).flatten()

            for name in names:
                if h_vals is not None and name in h_names:
                    result[name] = float(h_vals[h_names.index(name)])
                elif name in self.model.x_names:
                    result[name] = float(self.model.x[self.model.x_names.index(name)])
                elif name in self.model.y_run_names:
                    result[name] = float(
                        self.model.y_run[self.model.y_run_names.index(name)]
                    )
                elif name in self.model.u_run_names:
                    result[name] = float(
                        self.model.u_run_vals[self.model.u_run_names.index(name)]
                    )
                elif name in self.model.p_names:
                    result[name] = float(
                        self.model.p_vals[self.model.p_names.index(name)]
                    )
                else:
                    result[name] = None

            return result
