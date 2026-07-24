"""Tests for the pluggable linear-solver backend of the fixed-step trapezoidal
solver in ``CasadiModel`` (``linsol=`` option).

Two backends are exercised:

- ``linsol='scipy'``  — scipy sparse LU (``splu``), the default.
- ``linsol='qr'``     — CasADi-native ``ca.Linsol`` with the sfact/nfact/solve
                        reuse trio (symbolic factorization once, numeric on
                        refresh, cheap back-substitution per Newton iteration).

They solve the same equations, so a transient must match to ~1e-9. An
unavailable plugin must fall back to SciPy without error.

Run selectively:
    uv run pytest tests/core/test_casadi_linsol.py -v
"""
import json

import numpy as np
import pytest

import pydae.core.model.casadi_model as cm

pytestmark = pytest.mark.bps


CLASSICAL_PARAMS = {
    "S_n": 100e6, "F_n": 50.0, "X1d": 0.30,
    "R_a": 0.0, "H": 5.0, "D": 1.0,
    "K_sec": 0.0, "K_delta": 0.0, "e1q": 1.2, "p_m": 0.5,
}


def _smib_grid():
    return {
        "system": {"name": "smib_linsol", "S_base": 100e6},
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
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder

    path = str(tmp_path_factory.mktemp("smib_linsol") / "smib.json")
    json.dump(_smib_grid(), open(path, "w"))
    grid = BpsBuilder(path, use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct("smib_linsol")
    return CasadiBuilder(grid.sys_dict).build(), grid.dae["xy_0_dict"]


def _run_transient(builder_xy0, linsol, t_end=5.0, p_m=0.7, Dt=0.01, **opts):
    from pydae.core.builder import CasadiModel

    builder, xy_0 = builder_xy0
    m = CasadiModel(builder)
    m.Dt = Dt
    m.ini({}, xy_0=xy_0)
    m.set_value("p_m_1", p_m)
    m.recalculate_algebraics(tol=1e-9)
    m.run(t_end, method="trapezoidal", linsol=linsol, **opts)
    m.post()
    return m


# ─── 1. Parity: scipy vs qr solve the same equations ──────────────────────

@pytest.mark.model
def test_scipy_qr_parity(builder_xy0):
    sc = _run_transient(builder_xy0, "scipy")
    qr = _run_transient(builder_xy0, "qr")
    assert isinstance(sc._lin_backend, cm._ScipyLU)
    assert isinstance(qr._lin_backend, cm._CasadiLinsol)
    for name in ("delta_1", "omega_1"):
        d = np.max(np.abs(np.array(qr.get_values(name)) - np.array(sc.get_values(name))))
        assert d < 1e-9, f"{name}: max|qr - scipy| = {d:.2e}"


# ─── 2. Reference accuracy: both within tol of an IDAS grid solve ─────────

@pytest.mark.model
@pytest.mark.parametrize("linsol", ["scipy", "qr"])
def test_matches_idas_grid(builder_xy0, linsol):
    from pydae.core.builder import CasadiModel

    builder, xy_0 = builder_xy0
    ref = CasadiModel(builder)
    ref.Dt = 0.01
    ref.ini({}, xy_0=xy_0)
    ref.set_value("p_m_1", 0.7)
    ref.recalculate_algebraics(tol=1e-9)
    ref.run(5.0, method="idas_grid")
    ref.post()

    m = _run_transient(builder_xy0, linsol)
    d = np.max(np.abs(np.array(m.get_values("delta_1")) - np.array(ref.get_values("delta_1"))))
    assert d < 5e-4, f"{linsol}: max|trap - idas_grid| = {d:.2e}"


# ─── 3. Reuse: the factorization is reused across solves; sfact once ──────

@pytest.mark.model
def test_casadi_linsol_reuses_factorization(builder_xy0, monkeypatch):
    counts = {"refresh": 0, "solve": 0}
    orig_refresh = cm._CasadiLinsol.refresh
    orig_solve = cm._CasadiLinsol.solve

    def counting_refresh(self, J_dm):
        counts["refresh"] += 1
        return orig_refresh(self, J_dm)

    def counting_solve(self, neg_R):
        counts["solve"] += 1
        return orig_solve(self, neg_R)

    monkeypatch.setattr(cm._CasadiLinsol, "refresh", counting_refresh)
    monkeypatch.setattr(cm._CasadiLinsol, "solve", counting_solve)

    m = _run_transient(builder_xy0, "qr", refactor_every=20)

    # symbolic factorization ran exactly once (bound to the fixed sparsity)
    assert m._lin_backend._sfact_done is True
    # more solves than (numeric) refreshes -> the factorization is genuinely
    # reused across Newton iterations and scheduled steps
    assert counts["solve"] > counts["refresh"] > 0


# ─── 4. Discontinuity: qr handles set_value + recalc + continue ───────────

@pytest.mark.model
def test_discontinuity_qr_matches_scipy(builder_xy0):
    from pydae.core.builder import CasadiModel

    def segmented(linsol):
        builder, xy_0 = builder_xy0
        m = CasadiModel(builder)
        m.Dt = 0.01
        m.ini({}, xy_0=xy_0)
        m.run_method = "trapezoidal"
        m.run(1.0, linsol=linsol)
        m.set_value("p_m_1", 0.9)          # auto-marks the backend stale
        m.recalculate_algebraics(tol=1e-9)
        m.run(5.0, linsol=linsol)
        m.post()
        return m

    sc = segmented("scipy")
    qr = segmented("qr")
    d = np.max(np.abs(np.array(qr.get_values("delta_1")) - np.array(sc.get_values("delta_1"))))
    assert d < 1e-9, f"discontinuity: max|qr - scipy| = {d:.2e}"


# ─── 5. Fallback: an unavailable plugin degrades to SciPy, no error ───────

@pytest.mark.model
def test_unavailable_plugin_falls_back_to_scipy(builder_xy0):
    with pytest.warns(RuntimeWarning, match="falling back to SciPy"):
        m = _run_transient(builder_xy0, "no_such_plugin_xyz")
    assert isinstance(m._lin_backend, cm._ScipyLU)
    # and it still produced a sensible trajectory
    ref = _run_transient(builder_xy0, "scipy")
    d = np.max(np.abs(np.array(m.get_values("delta_1")) - np.array(ref.get_values("delta_1"))))
    assert d < 1e-12
