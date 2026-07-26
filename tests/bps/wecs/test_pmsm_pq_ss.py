"""Tests for the pmsm_pq_ss simplified wind-turbine model.

pmsm_pq_ss keeps the wind/mechanical dynamics (aero + 2-mass drivetrain + pitch
+ MPPT) of pmsm_1 but reduces the electrical stack to an algebraic, current-
limited grid-following p/q injection (the pv_pq_ss simplification, wind side).

Build + ini need a C compiler; marked ``build``/``model``.
"""
import json

import numpy as np
import pytest

pytestmark = [pytest.mark.bps, pytest.mark.model]


GRID = {
    "system": {"name": "pmsm_pq_ss_test", "S_base": 100e6},
    "buses": [{"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0}],
    "wecs": [{"type": "pmsm_pq_ss", "bus": "1", "name": "WT1",
              "S_n": 3e6, "F_n": 50.0, "H_t": 4.0, "H_r": 0.8,
              "w_tr": 2.0, "d_tr": 0.1, "K_mppt3": 0.4, "T_mppt": 5.0,
              "I_max": 1.2}],
    "sources": [{"type": "vsource", "bus": "1"}],
}


@pytest.fixture(scope="module")
def model(tmp_path_factory, monkeypatch_module):
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    work = tmp_path_factory.mktemp("pmsm_pq_ss")
    monkeypatch_module.chdir(work)
    path = work / "grid.json"
    json.dump(GRID, open(path, "w"))

    grid = BpsBuilder(str(path))
    grid.checker()
    grid.uz_jacs = False
    grid.construct("pmsm_pq_ss_test")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()

    m = Model("pmsm_pq_ss_test")
    m.ini({}, grid.dae["xy_0_dict"])
    return m


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_mppt_steady_state(model):
    # p_s = K_mppt3 * omega_t^3, and aerodynamic power balances extraction
    p_s = float(model.get_value("p_s_WT1"))
    omega_t = float(model.get_value("omega_t_WT1"))
    assert p_s == pytest.approx(0.4 * omega_t**3, abs=1e-4)
    assert p_s == pytest.approx(float(model.get_value("p_w_WT1")), abs=1e-4)
    # turbine and rotor speeds coincide at steady state (no torsion rate)
    assert float(model.get_value("omega_r_WT1")) == pytest.approx(omega_t, abs=1e-4)
    # below rated -> no pitch action
    assert float(model.get_value("beta_WT1")) == pytest.approx(0.0, abs=1e-6)
    assert float(model.get_value("q_s_WT1")) == pytest.approx(0.0, abs=1e-6)


def test_undisturbed_run_holds_equilibrium(model):
    x0 = {v: float(model.get_value(v)) for v in model.x_list}
    model.run(2.0, {})
    model.post()
    drift = max(abs(float(model.get_value(v)) - x0[v]) for v in model.x_list)
    assert drift < 1e-6


def test_reactive_dispatch(model):
    model.run(model.t_start + 3.0, {"q_s_ppc_WT1": 0.2})
    model.post()
    assert float(model.get_value("q_s_WT1")) == pytest.approx(0.2, abs=1e-3)


def test_wind_step_increases_power(model):
    p0 = float(model.get_value("p_s_WT1"))
    w0 = float(model.get_value("omega_t_WT1"))
    model.run(model.t_start + 20.0, {"nu_w_0_WT1": 12.0})
    model.post()
    # higher wind -> rotor speeds up and more power is delivered
    assert float(model.get_value("omega_t_WT1")) > w0
    assert float(model.get_value("p_s_WT1")) > p0


def test_ppc_active_curtailment(model):
    # p_s_ppc sets the output = min(command, available wind power); commanding
    # below the available power curtails the injection to exactly the command.
    model.run(model.t_start + 40.0, {"p_s_ppc_WT1": 0.3})
    model.post()
    assert float(model.get_value("p_s_WT1")) == pytest.approx(0.3, abs=2e-2)


def test_powerflow_constant_power(tmp_path, monkeypatch):
    """Power-flow use case: high wind -> constant-power region -> output = p_s_ppc."""
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    monkeypatch.chdir(tmp_path)
    g = json.loads(json.dumps(GRID))            # deep copy
    g["system"]["name"] = "pmsm_pf"
    g["wecs"][0]["nu_w_0"] = 16.0               # high wind (constant-power region)
    g["wecs"][0]["p_s_ppc"] = 0.5              # command below rated
    path = tmp_path / "pf.json"
    json.dump(g, open(path, "w"))

    grid = BpsBuilder(str(path))
    grid.checker()
    grid.uz_jacs = False
    grid.construct("pmsm_pf")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()

    m = Model("pmsm_pf")
    assert m.ini({}, grid.dae["xy_0_dict"])
    # output equals the PPC command; speed is pitch-limited and pitch is spilling
    assert float(m.get_value("p_s_WT1")) == pytest.approx(0.5, abs=1e-3)
    assert float(m.get_value("omega_t_WT1")) == pytest.approx(1.2, abs=0.05)
    assert float(m.get_value("beta_WT1")) > 1.0
