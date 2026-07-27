"""Tests for the dfim_pq_ss simplified DFIG wind-turbine model.

dfim_pq_ss shares the wind/mechanical dynamics (aero + 2-mass drivetrain + pitch
+ MPPT) of pmsm_pq_ss but models a doubly-fed induction machine: the total grid
power p_g = min(p_s_ppc, available) splits by slip into the stator power
p_s = p_g/omega_r (direct to grid) and the rotor/converter power p_r = -slip*p_s.
The rotor (RSC) current, current-limited to the converter rating I_max_r, sets
p_s/q_s through the stator-flux-oriented (X_m/X_s) coupling.

Build + ini need a C compiler; marked ``build``/``model``.
"""
import json

import numpy as np
import pytest

pytestmark = [pytest.mark.bps, pytest.mark.model]


GRID = {
    "system": {"name": "dfim_pq_ss_test", "S_base": 100e6},
    "buses": [{"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0}],
    "wecs": [{"type": "dfim_pq_ss", "bus": "1", "name": "WT1",
              "S_n": 3e6, "F_n": 50.0, "H_t": 4.0, "H_r": 0.8,
              "w_tr": 2.0, "d_tr": 0.1, "K_mppt3": 0.4, "T_mppt": 5.0,
              "X_m": 3.0, "X_s": 3.05, "I_max_r": 1.2}],
    "sources": [{"type": "vsource", "bus": "1"}],
}


@pytest.fixture(scope="module")
def model(tmp_path_factory, monkeypatch_module):
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    work = tmp_path_factory.mktemp("dfim_pq_ss")
    monkeypatch_module.chdir(work)
    path = work / "grid.json"
    json.dump(GRID, open(path, "w"))

    grid = BpsBuilder(str(path))
    grid.checker()
    grid.uz_jacs = False
    grid.construct("dfim_pq_ss_test")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()

    m = Model("dfim_pq_ss_test")
    m.ini({}, grid.dae["xy_0_dict"])
    return m


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_mppt_steady_state(model):
    # total grid power tracks the MPPT cubic law and equals the aerodynamic power
    p_g = float(model.get_value("p_g_WT1"))
    omega_t = float(model.get_value("omega_t_WT1"))
    omega_r = float(model.get_value("omega_r_WT1"))
    assert p_g == pytest.approx(0.4 * omega_t**3, abs=1e-4)
    assert p_g == pytest.approx(float(model.get_value("p_w_WT1")), abs=1e-4)
    # turbine and rotor speeds coincide at steady state (no torsion rate)
    assert omega_r == pytest.approx(omega_t, abs=1e-4)
    # below rated -> no pitch action
    assert float(model.get_value("beta_WT1")) == pytest.approx(0.0, abs=1e-6)
    assert float(model.get_value("q_s_WT1")) == pytest.approx(0.0, abs=1e-6)


def test_slip_power_split(model):
    # p_g = p_s + p_r = p_s * omega_r ; stator p_s = p_g/omega_r ; p_r = -slip*p_s
    p_g = float(model.get_value("p_g_WT1"))
    p_s = float(model.get_value("p_s_WT1"))
    p_r = float(model.get_value("p_r_WT1"))
    slip = float(model.get_value("slip_WT1"))
    omega_r = float(model.get_value("omega_r_WT1"))
    assert slip == pytest.approx(1.0 - omega_r, abs=1e-9)
    assert p_s == pytest.approx(p_g / omega_r, abs=1e-4)
    assert p_r == pytest.approx(-slip * p_s, abs=1e-6)
    assert p_g == pytest.approx(p_s + p_r, abs=1e-6)
    # super-synchronous (omega_r > 1): converter delivers slip power
    assert p_r > 0.0


def test_undisturbed_run_holds_equilibrium(model):
    x0 = {v: float(model.get_value(v)) for v in model.x_list}
    model.run(2.0, {})
    model.post()
    drift = max(abs(float(model.get_value(v)) - x0[v]) for v in model.x_list)
    assert drift < 1e-6


def test_reactive_dispatch(model):
    # RSC controls the total reactive power; q_g tracks the command
    model.run(model.t_start + 3.0, {"q_s_ppc_WT1": 0.2})
    model.post()
    assert float(model.get_value("q_g_WT1")) == pytest.approx(0.2, abs=1e-3)


def test_wind_step_increases_power(model):
    p0 = float(model.get_value("p_g_WT1"))
    w0 = float(model.get_value("omega_t_WT1"))
    model.run(model.t_start + 20.0, {"nu_w_0_WT1": 12.0})
    model.post()
    # higher wind -> rotor speeds up and more total power is delivered
    assert float(model.get_value("omega_t_WT1")) > w0
    assert float(model.get_value("p_g_WT1")) > p0


def test_ppc_active_curtailment(model):
    # p_s_ppc sets the TOTAL output = min(command, available wind power);
    # commanding below the available power curtails p_g to exactly the command.
    model.run(model.t_start + 40.0, {"p_s_ppc_WT1": 0.3})
    model.post()
    assert float(model.get_value("p_g_WT1")) == pytest.approx(0.3, abs=2e-2)


def test_casadi_backend_parity(tmp_path, monkeypatch):
    """Dual-backend: the model builds and inis on the CasADi backend (no leftover
    SymPy expressions in the SX graph) and matches the SymPy operating point."""
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder, CasadiModel

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "grid_ca.json"
    json.dump(GRID, open(path, "w"))

    grid = BpsBuilder(str(path), use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct("dfim_pq_ss_ca")
    # every f/g entry must be a CasADi SX (a leftover sympy expr breaks vertcat)
    mc = CasadiModel(CasadiBuilder(grid.sys_dict).build())
    mc.ini({}, xy_0=grid.dae["xy_0_dict"])

    p_g = float(mc.get_value("p_g_WT1"))
    omega_t = float(mc.get_value("omega_t_WT1"))
    omega_r = float(mc.get_value("omega_r_WT1"))
    assert p_g == pytest.approx(0.4 * omega_t**3, abs=1e-4)   # MPPT steady state
    assert p_g == pytest.approx(0.5553, abs=2e-3)             # matches SymPy backend
    assert float(mc.get_value("p_s_WT1")) == pytest.approx(p_g / omega_r, abs=1e-4)
    assert float(mc.get_value("beta_WT1")) == pytest.approx(0.0, abs=1e-3)


def test_powerflow_constant_power(tmp_path, monkeypatch):
    """Power-flow use case: high wind -> constant-power region -> p_g = p_s_ppc."""
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    monkeypatch.chdir(tmp_path)
    g = json.loads(json.dumps(GRID))            # deep copy
    g["system"]["name"] = "dfim_pf"
    g["wecs"][0]["nu_w_0"] = 16.0               # high wind (constant-power region)
    g["wecs"][0]["p_s_ppc"] = 0.5               # command below rated
    path = tmp_path / "pf.json"
    json.dump(g, open(path, "w"))

    grid = BpsBuilder(str(path))
    grid.checker()
    grid.uz_jacs = False
    grid.construct("dfim_pf")
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()

    m = Model("dfim_pf")
    assert m.ini({}, grid.dae["xy_0_dict"])
    # total output equals the PPC command; speed is pitch-limited and spilling
    assert float(m.get_value("p_g_WT1")) == pytest.approx(0.5, abs=1e-3)
    assert float(m.get_value("omega_t_WT1")) == pytest.approx(1.2, abs=0.05)
    assert float(m.get_value("beta_WT1")) > 1.0
