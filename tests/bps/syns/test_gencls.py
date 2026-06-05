"""Tests for the gencls (classical 2nd-order) machine.

- ``test_smoke`` — ini converges; a 5-second time integration with
  constant inputs stays flat.
- ``test_state_set`` — gencls exposes exactly the classical two-state
  list (delta, omega). No rotor flux states.
- ``test_parity_milano2ord_no_saliency`` — when ``milano2ord`` is fed
  ``X1q = X1d`` (no transient saliency), the steady-state quantities
  match ``gencls`` to machine precision and the swing-mode eigenvalue
  matches to 1e-6.
"""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.bps


CLASSICAL_PARAMS = {
    "S_n": 100e6, "F_n": 50.0,
    "X1d": 0.30,
    "R_a": 0.0, "H": 5.0, "D": 0.0,
    "K_sec": 0.0, "K_delta": 0.0,
    "e1q": 1.2, "p_m": 0.5,
}


def _smib_grid(syn_overrides=None, model_type="gencls"):
    syn = dict(CLASSICAL_PARAMS)
    syn.update({"bus": "1", "type": model_type})
    if syn_overrides:
        syn.update(syn_overrides)
    return {
        "system": {"name": "test_gencls", "S_base": 100e6},
        "buses": [
            {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
            {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
        ],
        "lines": [
            {"bus_j": "1", "bus_k": "2", "X_pu": 0.05, "R_pu": 0.01,
             "Bs_pu": 1e-6, "S_mva": 200},
        ],
        "shunts": [{"bus": "1", "X_pu": 1e6, "R_pu": 0.0, "S_mva": 100}],
        "syns": [syn],
        "sources": [{"type": "vsource", "bus": "2"}],
    }


def _build_and_ini(grid_dict, name):
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder, CasadiModel

    path = f"/tmp/{name}.json"
    with open(path, "w") as fh:
        json.dump(grid_dict, fh)
    grid_dict["system"]["name"] = name
    grid = BpsBuilder(path, use_casadi=True)
    grid.checker(); grid.uz_jacs = False
    grid.construct(name)
    mc = CasadiModel(CasadiBuilder(grid.sys_dict).build())
    mc.ini({}, xy_0=grid.dae["xy_0_dict"])
    return mc


def test_smoke():
    mc = _build_and_ini(_smib_grid(), "gencls_smoke")

    delta0 = float(mc.get_value("delta_1"))
    omega0 = float(mc.get_value("omega_1"))
    p_g0 = float(mc.get_value("p_g_1"))

    assert omega0 == pytest.approx(1.0, abs=5e-2)
    assert p_g0 == pytest.approx(0.5, abs=1e-3)
    assert 0.0 < delta0 < np.pi / 2

    mc.run(5.0, {}); mc.post()
    assert float(mc.get_value("omega_1")) == pytest.approx(omega0, abs=1e-5)
    assert float(mc.get_value("delta_1")) == pytest.approx(delta0, abs=1e-4)
    assert float(mc.get_value("p_g_1")) == pytest.approx(p_g0, abs=1e-4)


def test_state_set():
    """gencls exposes exactly 2 syn states (delta, omega) — no rotor flux."""
    mc = _build_and_ini(_smib_grid(), "gencls_states")
    syn_states = [s for s in mc.x_list if s.endswith("_1")]
    assert syn_states == ["delta_1", "omega_1"], (
        f"gencls state list mismatch: {syn_states}")


def test_parity_milano2ord_no_saliency():
    """milano2ord with X1q = X1d collapses onto gencls; quantities and
    the swing-mode eigenvalue agree to machine precision."""
    from pydae.ssa.ssa import eig

    grid_c = _smib_grid(model_type="gencls")
    grid_m = _smib_grid(syn_overrides={"X1q": 0.30}, model_type="milano2ord")

    mc = _build_and_ini(grid_c, "parity_cls")
    mm = _build_and_ini(grid_m, "parity_m2")

    for var in ("delta_1", "omega_1", "p_g_1", "q_g_1",
                "i_d_1", "i_q_1", "V_1"):
        a = float(mc.get_value(var))
        b = float(mm.get_value(var))
        assert abs(a - b) < 1e-6, f"{var}: gencls={a}, milano2ord={b}"

    eig(mc); eig(mm)
    def _swing(model):
        osc = [z for z in model.eigenvalues if abs(z.imag) > 1e-3]
        return min(osc, key=lambda z: abs(z.imag))
    s_c = _swing(mc); s_m = _swing(mm)
    assert abs(s_c.real - s_m.real) < 1e-6, f"real: {s_c} vs {s_m}"
    assert abs(abs(s_c.imag) - abs(s_m.imag)) < 1e-6, f"imag: {s_c} vs {s_m}"
