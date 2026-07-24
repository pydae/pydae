"""Tests for CasadiModel.run(method=...) time-domain integration strategies.

Covers the four solver paths added on top of the legacy per-Dt IDAS loop:

- ``idas_grid``      one IDAS call over a grid of output times (default; ~30x
                     faster than ``idas`` with an identical trajectory).
- ``idas``           legacy stateless IDAS call once per Dt (BDF cold-start).
- ``trapezoidal``    fixed-step implicit theta-method, alpha=0.5 (2nd order).
- ``backward_euler`` fixed-step implicit theta-method, alpha=1.0 (1st order).

The benchmark is a single-machine-infinite-bus ``gencls`` grid — a small,
well-posed power-system DAE (unlike the pendulum, whose atan2 output has a
singular Jacobian at the origin).

Run selectively:
    uv run pytest tests/core/test_casadi_run_methods.py -v
"""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.bps


CLASSICAL_PARAMS = {
    "S_n": 100e6, "F_n": 50.0, "X1d": 0.30,
    "R_a": 0.0, "H": 5.0, "D": 1.0,
    "K_sec": 0.0, "K_delta": 0.0, "e1q": 1.2, "p_m": 0.5,
}


def _smib_grid():
    return {
        "system": {"name": "smib_run_methods", "S_base": 100e6},
        "buses": [{"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
                  {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0}],
        "lines": [{"bus_j": "1", "bus_k": "2", "X_pu": 0.05, "R_pu": 0.01,
                   "Bs_pu": 1e-6, "S_mva": 200}],
        "shunts": [{"bus": "1", "X_pu": 1e6, "R_pu": 0.0, "S_mva": 100}],
        "syns": [dict(CLASSICAL_PARAMS, bus="1", type="gencls")],
        "sources": [{"type": "vsource", "bus": "2"}],
    }


@pytest.fixture(scope="module")
def builder_xy0(tmp_path_factory):
    """Build the SMIB CasADi model once for the whole module."""
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder

    path = str(tmp_path_factory.mktemp("smib") / "smib.json")
    json.dump(_smib_grid(), open(path, "w"))
    grid = BpsBuilder(path, use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct("smib_run_methods")
    builder = CasadiBuilder(grid.sys_dict).build()
    return builder, grid.dae["xy_0_dict"]


def _model(builder_xy0, Dt=0.01):
    from pydae.core.builder import CasadiModel

    builder, xy_0 = builder_xy0
    m = CasadiModel(builder)
    m.Dt = Dt
    m.ini({}, xy_0=xy_0)
    return m


def _run_step(builder_xy0, method, t_end=5.0, p_m=0.7, **opts):
    """ini, apply a consistent p_m step, integrate to t_end with ``method``."""
    m = _model(builder_xy0)
    m.set_value("p_m_1", p_m)
    m.recalculate_algebraics(tol=1e-9)
    m.run(t_end, method=method, **opts)
    m.post()
    return m


# ─── 1. Correctness: fast methods match the reference trajectory ──────────

@pytest.mark.model
@pytest.mark.parametrize("method,tol", [
    ("idas", 5e-4),            # legacy per-Dt loop vs grid (both IDAS)
    ("trapezoidal", 5e-4),     # 2nd order
    ("backward_euler", 5e-3),  # 1st order -> looser
])
def test_method_matches_idas_grid(builder_xy0, method, tol):
    ref = _run_step(builder_xy0, "idas_grid")
    m = _run_step(builder_xy0, method)

    # identical storage layout (same number of output points)
    assert len(m.Time) == len(ref.Time)
    assert np.allclose(m.Time, ref.Time)

    dd = np.max(np.abs(m.get_values("delta_1") - ref.get_values("delta_1")))
    dw = np.max(np.abs(m.get_values("omega_1") - ref.get_values("omega_1")))
    assert dd < tol, f"{method}: max|d_delta|={dd:.2e}"
    assert dw < tol, f"{method}: max|d_omega|={dw:.2e}"


# ─── 2. idas_grid is the default and reproduces the legacy loop ───────────

@pytest.mark.model
def test_default_method_is_idas_grid(builder_xy0):
    m = _model(builder_xy0)
    assert m.run_method == "idas_grid"
    # default call (no method=) must produce the same trajectory as explicit grid
    m.set_value("p_m_1", 0.7)
    m.recalculate_algebraics(tol=1e-9)
    m.run(3.0)                       # default -> idas_grid
    m.post()
    ref = _run_step(builder_xy0, "idas_grid", t_end=3.0)
    assert np.allclose(m.get_values("delta_1"), ref.get_values("delta_1"))


@pytest.mark.model
def test_unknown_method_raises(builder_xy0):
    m = _model(builder_xy0)
    with pytest.raises(ValueError, match="unknown run method"):
        m.run(1.0, method="rk4")


# ─── 3. Discontinuity handling: step + recalc + refactor + continue ───────

def _run_segmented(builder_xy0, method, t_step=1.0, t_end=6.0, p_m=0.9):
    """Run flat, step p_m at t_step, recalc (set_value auto-invalidates the
    frozen LU), then continue — a full discontinuity-handling scenario."""
    m = _model(builder_xy0)
    m.run_method = method
    m.run(t_step)
    m.set_value("p_m_1", p_m)
    m.recalculate_algebraics(tol=1e-9)
    m.run(t_end)
    m.post()
    return m


@pytest.mark.model
@pytest.mark.parametrize("method,tol", [
    ("trapezoidal", 5e-4),
    ("backward_euler", 5e-3),
])
def test_discontinuity_across_segments(builder_xy0, method, tol):
    # Reference and candidate run the *identical* step-at-t_step scenario, so the
    # whole post-discontinuity trajectory must agree (not just the endpoint).
    ref = _run_segmented(builder_xy0, "idas_grid")
    m = _run_segmented(builder_xy0, method)

    assert len(m.Time) == len(ref.Time)
    dd = np.max(np.abs(m.get_values("delta_1") - ref.get_values("delta_1")))
    assert dd < tol, f"{method}: max|d_delta| after discontinuity = {dd:.2e}"


# ─── 4. Determinism/energy: undisturbed run stays at equilibrium ──────────

@pytest.mark.model
@pytest.mark.parametrize("method", ["idas_grid", "idas", "trapezoidal", "backward_euler"])
def test_undisturbed_stays_at_equilibrium(builder_xy0, method):
    m = _model(builder_xy0)
    x0 = m.x.copy()
    m.run(2.0, method=method)
    m.post()
    # a converged ini + no disturbance -> the state must not move
    assert np.max(np.abs(m.X - x0)) < 1e-8
    assert m.get_value("omega_1") == pytest.approx(1.0, abs=1e-8)


# ─── 5. Sequential runs chain and Dt changes are honoured ─────────────────

@pytest.mark.model
@pytest.mark.parametrize("method", ["idas_grid", "trapezoidal"])
def test_sequential_runs_chain(builder_xy0, method):
    m = _model(builder_xy0)
    m.set_value("p_m_1", 0.7)
    m.recalculate_algebraics(tol=1e-9)
    m.run(2.0, method=method)
    n_after_first = len(m.Time)
    m.run(4.0, method=method)
    m.post()
    # second run continues from t=2 to t=4, appending (not restarting)
    assert len(m.Time) > n_after_first
    assert m.Time[-1] == pytest.approx(4.0, abs=1e-9)
    assert m.Time[n_after_first - 1] == pytest.approx(2.0, abs=1e-9)


@pytest.mark.model
def test_zero_span_is_noop(builder_xy0):
    m = _model(builder_xy0)
    m.run(1.0, method="trapezoidal")
    n = len(m.Time)
    m.run(1.0, method="trapezoidal")   # t_end == current t -> nothing to do
    assert len(m.Time) == n
