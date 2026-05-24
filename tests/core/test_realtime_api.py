# tests/core/test_realtime_api.py
"""Tests for the soft real-time simulation API using the pendulum CasADi model.

Run with:
    uv run pytest tests/core/test_realtime_api.py -v
    uv run pytest tests/core/test_realtime_api.py -v -m model
"""

import time

import casadi as ca
import numpy as np
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pendulum_sys_dict():
    """CasADi pendulum system dictionary (shared across all tests in this module)."""
    L, G, M, K_d, K_lam = [ca.SX.sym(n) for n in ["L", "G", "M", "K_d", "K_lam"]]
    p_x, p_y, v_x, v_y = [ca.SX.sym(n) for n in ["p_x", "p_y", "v_x", "v_y"]]
    lam, f_x, theta, u_dummy = [ca.SX.sym(n) for n in ["lam", "f_x", "theta", "u_dummy"]]

    return {
        "name": "rt_pendulum",
        "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3, "K_lam": 1e-2},
        "f_list": [
            v_x, v_y,
            (-2 * p_x * lam + f_x - K_d * v_x) / M,
            (-M * G - 2 * p_y * lam - K_d * v_y) / M,
        ],
        "g_list": [
            p_x ** 2 + p_y ** 2 - L ** 2 - lam * K_lam,
            -theta + ca.atan2(p_x, -p_y) + u_dummy,
        ],
        "x_list": [p_x, p_y, v_x, v_y],
        "y_ini_list": [lam, f_x],
        "y_run_list": [lam, theta],
        "u_ini_dict": {"theta": np.deg2rad(5.0), "u_dummy": 0.0},
        "u_run_dict": {"f_x": 0.0, "u_dummy": 0.0},
        "h_dict": {
            "E_p": M * G * (p_y + L),
            "E_k": 0.5 * M * (v_x ** 2 + v_y ** 2),
            "f_x_out": f_x,
            "lam_out": lam,
        },
    }


@pytest.fixture(scope="module")
def pendulum_builder(pendulum_sys_dict):
    """Build once per module — CasadiBuilder compilation is the expensive step."""
    from pydae.core.builder import CasadiBuilder
    return CasadiBuilder(pendulum_sys_dict).build()


@pytest.fixture
def pendulum_model(pendulum_builder):
    """Fresh initialized CasadiModel for every test (builder is reused)."""
    from pydae.core.builder import CasadiModel

    model = CasadiModel(pendulum_builder)
    L, deg = 5.21, 10
    model.ini(
        {"M": 10.0, "L": L, "K_lam": 1e-2, "theta": np.deg2rad(deg)},
        xy_0={
            "p_x": L * np.sin(np.deg2rad(deg)),
            "p_y": -L * np.cos(np.deg2rad(deg)),
            "lam": 50.0,
            "f_x": 0.0,
            "v_x": 0.0,
            "v_y": 0.0,
        },
    )
    return model


# ── RealTimeSimulator unit tests ───────────────────────────────────────────────

@pytest.mark.model
class TestRealTimeSimulator:

    def test_lifecycle_start_stop(self, pendulum_model):
        """Simulator starts a daemon thread and stops it cleanly."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        assert not sim.is_running
        assert sim._thread is None

        sim.start()
        assert sim.is_running
        assert sim._thread is not None
        assert sim._thread.is_alive()

        sim.stop()
        assert not sim.is_running
        assert not sim._thread.is_alive()

    def test_t_sim_advances(self, pendulum_model):
        """t_sim grows after the simulator runs for a short wall-clock interval."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.35)  # ~7 chunks of 50 ms
        t_mid = sim.t_sim
        sim.stop()

        assert t_mid > 0.0, "t_sim did not advance"
        assert t_mid < 2.0, "t_sim advanced far beyond wall time (drift problem)"

    def test_measurements_h_dict_outputs(self, pendulum_model):
        """measurements returns finite values for all h_dict outputs."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.15)
        out = sim.measurements(["E_p", "E_k", "f_x_out", "lam_out"])
        sim.stop()

        assert out["E_p"] is not None and np.isfinite(out["E_p"])
        assert out["E_k"] is not None and out["E_k"] >= 0.0
        assert out["f_x_out"] is not None and np.isfinite(out["f_x_out"])
        assert out["lam_out"] is not None and np.isfinite(out["lam_out"])

    def test_measurements_non_h_dict_names_return_none(self, pendulum_model):
        """Variables outside h_dict (states, algebraics, inputs) return None."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.05)
        out = sim.measurements(["p_x", "lam", "f_x", "does_not_exist"])
        sim.stop()

        for name in ["p_x", "lam", "f_x", "does_not_exist"]:
            assert out[name] is None, f"expected None for '{name}', got {out[name]}"

    def test_measurements_mixed_returns_none_for_non_outputs(self, pendulum_model):
        """h_dict names resolve correctly; non-h_dict names return None."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.15)
        out = sim.measurements(["E_k", "p_x"])
        sim.stop()

        assert out["E_k"] is not None and out["E_k"] >= 0.0
        assert out["p_x"] is None

    def test_set_input_updates_u_run_vals(self, pendulum_model):
        """set_input writes the new value into u_run_vals and it is visible to the integrator."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.1)
        sim.set_input("f_x", 5.0)
        time.sleep(0.05)
        idx = pendulum_model.u_run_names.index("f_x")
        val = pendulum_model.u_run_vals[idx]
        sim.stop()

        assert val == pytest.approx(5.0), f"u_run_vals['f_x'] = {val}, expected 5.0"

    def test_setpoints_updates_multiple_inputs(self, pendulum_model):
        """setpoints writes several inputs atomically in one lock acquisition."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.1)
        sim.setpoints({"f_x": 3.0, "u_dummy": 0.1})
        time.sleep(0.05)
        sim.stop()

        idx_fx = pendulum_model.u_run_names.index("f_x")
        idx_ud = pendulum_model.u_run_names.index("u_dummy")
        assert pendulum_model.u_run_vals[idx_fx] == pytest.approx(3.0)
        assert pendulum_model.u_run_vals[idx_ud] == pytest.approx(0.1)

    def test_setpoints_single_entry(self, pendulum_model):
        """setpoints with a single-entry dict behaves like set_input."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.1)
        sim.setpoints({"f_x": 7.5})
        time.sleep(0.05)
        sim.stop()

        idx = pendulum_model.u_run_names.index("f_x")
        assert pendulum_model.u_run_vals[idx] == pytest.approx(7.5)

    def test_set_input_effect_on_trajectory(self, pendulum_model):
        """Applying a lateral force excites the pendulum — kinetic energy changes."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.1)
        E_k_before = sim.measurements(["E_k"])["E_k"]

        sim.set_input("f_x", 20.0)   # large lateral force injects energy
        time.sleep(0.3)
        E_k_after = sim.measurements(["E_k"])["E_k"]
        sim.stop()

        assert E_k_after != pytest.approx(E_k_before, abs=1e-6), (
            "E_k did not change after applying lateral force"
        )

    def test_no_memory_accumulation(self, pendulum_model):
        """Storage lists (Time, X, Y) must not grow during real-time stepping."""
        from pydae.api.realtime_api import RealTimeSimulator

        # After ini(), lists have exactly 1 element each.
        initial_len = len(pendulum_model.Time)
        assert initial_len == 1

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        time.sleep(0.4)  # would be 8 appends if using model.run()
        sim.stop()

        assert len(pendulum_model.Time) == initial_len, (
            f"Time list grew from {initial_len} to {len(pendulum_model.Time)} — "
            "memory leak in RT loop"
        )
        assert len(pendulum_model.X) == initial_len
        assert len(pendulum_model.Y) == initial_len

    def test_thread_is_daemon(self, pendulum_model):
        """The background thread must be a daemon so it does not block process exit."""
        from pydae.api.realtime_api import RealTimeSimulator

        sim = RealTimeSimulator(pendulum_model, chunk_ms=50.0)
        sim.start()
        assert sim._thread.daemon is True
        sim.stop()


# ── FastAPI endpoint tests ─────────────────────────────────────────────────────

@pytest.mark.model
class TestRealTimeAPIEndpoints:

    @pytest.fixture(autouse=True)
    def _inject_model(self, pendulum_model):
        """Attach a fresh model to app.state before each test's lifespan starts."""
        from pydae.api.realtime_api import app
        app.state.model = pendulum_model
        app.state.chunk_ms = 50.0

    def test_status_reports_running(self, pendulum_model):
        """GET /status returns is_running=True while the lifespan is active."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_running"] is True
        assert "t_sim" in body

    def test_status_t_sim_advances_within_session(self, pendulum_model):
        """t_sim reported by /status grows between two calls spaced 200 ms apart."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            t0 = client.get("/status").json()["t_sim"]
            time.sleep(0.25)
            t1 = client.get("/status").json()["t_sim"]

        assert t1 > t0, f"t_sim did not advance: {t0} -> {t1}"

    def test_measurements_endpoint_returns_finite_values(self, pendulum_model):
        """POST /measurements returns finite floats for h_dict outputs."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            time.sleep(0.15)
            resp = client.post(
                "/measurements",
                json={"names": ["E_p", "E_k", "f_x_out", "lam_out"]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "t_sim" in body

        for name in ["E_p", "E_k", "f_x_out", "lam_out"]:
            val = body["data"][name]
            assert val is not None, f"{name} is None"
            assert np.isfinite(val), f"{name} is not finite: {val}"

    def test_measurements_non_output_is_null(self, pendulum_model):
        """POST /measurements returns null for names not in h_dict."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            resp = client.post(
                "/measurements",
                json={"names": ["E_p", "p_x", "no_such_var"]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["E_p"] is not None
        assert body["data"]["p_x"] is None
        assert body["data"]["no_such_var"] is None

    def test_set_input_endpoint_accepts_valid_payload(self, pendulum_model):
        """POST /set_input returns status=ok and echoes name/value."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            resp = client.post("/set_input", json={"name": "f_x", "value": 3.5})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["name"] == "f_x"
        assert body["value"] == pytest.approx(3.5)

    def test_setpoints_endpoint_writes_multiple(self, pendulum_model):
        """POST /setpoints accepts a dict and returns it back."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            resp = client.post(
                "/setpoints",
                json={"setpoints": {"f_x": 4.0, "u_dummy": 0.2}},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["setpoints"]["f_x"] == pytest.approx(4.0)
        assert body["setpoints"]["u_dummy"] == pytest.approx(0.2)

    def test_setpoints_then_measurements_reflects_change(self, pendulum_model):
        """Force set via /setpoints shows up in the h_dict output f_x_out."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            client.post("/setpoints", json={"setpoints": {"f_x": 7.0}})
            time.sleep(0.1)
            resp = client.post("/measurements", json={"names": ["f_x_out"]})

        assert resp.status_code == 200
        val = resp.json()["data"]["f_x_out"]
        assert val == pytest.approx(7.0)

    def test_measurements_endpoint_returns_t_sim(self, pendulum_model):
        """POST /measurements always returns t_sim alongside the data dict."""
        from fastapi.testclient import TestClient
        from pydae.api.realtime_api import app

        with TestClient(app) as client:
            time.sleep(0.15)
            resp = client.post("/measurements", json={"names": ["p_x"]})

        body = resp.json()
        assert "t_sim" in body
        assert body["t_sim"] >= 0.0
