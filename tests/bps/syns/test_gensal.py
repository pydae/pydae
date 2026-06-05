"""Tests for the gensal (salient-pole) machine.

- ``test_smoke`` — ini converges from cold seeds; a 5-second time
  integration with constant inputs stays flat at the operating point.
- ``test_state_set`` — gensal exposes exactly the IEEE 1110 Model 2.1
  state list (delta, omega, e1q, e2q, e2d) with no q-axis transient EMF
  (e1d). genrou has e1d as its 6th state; gensal does not.
- ``test_saturation_daxis_only`` — when saturation is active and the
  q-axis subtransient is nonzero, the saturation factor S_at depends
  only on e1q, not on e2d (PSS/E GENSAL convention).
"""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.bps


HYDRO_PARAMS = {
    "S_n": 500e6, "F_n": 50.0,
    "X_d": 1.10, "X_q": 0.70,
    "X1d": 0.25, "T1d0": 8.0,
    "X2d": 0.20, "T2d0": 0.03,
    "X2q": 0.20, "T2q0": 0.05,
    "R_a": 0.0, "H": 4.0, "D": 0.0,
    "S_10": 0.0, "S_12": 0.0,
    "K_sec": 0.0, "K_delta": 0.0,
    "v_f": 1.5, "p_m": 0.5,
}


def _smib_grid(syn_overrides=None, model_type="gensal"):
    syn = dict(HYDRO_PARAMS)
    syn.update({"bus": "1", "type": model_type})
    if syn_overrides:
        syn.update(syn_overrides)
    return {
        "system": {"name": "test_gensal", "S_base": 500e6},
        "buses": [
            {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 13.8},
            {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 13.8},
        ],
        "lines": [
            {"bus_j": "1", "bus_k": "2", "X_pu": 0.05, "R_pu": 0.01,
             "Bs_pu": 1e-6, "S_mva": 5000},
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
    grid.checker()
    grid.uz_jacs = False
    grid.construct(name)
    mc = CasadiModel(CasadiBuilder(grid.sys_dict).build())
    mc.ini({}, xy_0=grid.dae["xy_0_dict"])
    return mc


def test_smoke():
    """Ini converges; 5 s flat run with constant inputs."""
    mc = _build_and_ini(_smib_grid(), "gensal_smoke")

    delta0 = float(mc.get_value("delta_1"))
    omega0 = float(mc.get_value("omega_1"))
    p_g0 = float(mc.get_value("p_g_1"))

    assert omega0 == pytest.approx(1.0, abs=5e-2)
    assert p_g0 == pytest.approx(0.5, abs=1e-3)
    assert 0.0 < delta0 < np.pi / 2

    mc.run(5.0, {})
    mc.post()

    assert float(mc.get_value("omega_1")) == pytest.approx(omega0, abs=1e-5)
    assert float(mc.get_value("delta_1")) == pytest.approx(delta0, abs=1e-4)
    assert float(mc.get_value("p_g_1")) == pytest.approx(p_g0, abs=1e-4)


def test_state_set():
    """gensal exposes 5 syn states (no e1d). genrou has 6 (with e1d)."""
    mc_s = _build_and_ini(_smib_grid(), "gensal_states")

    # vsource adds a V_dummy state, so filter to gen-1 states only
    syn_states = [s for s in mc_s.x_list if s.endswith("_1")]
    assert syn_states == ["delta_1", "omega_1", "e1q_1", "e2q_1", "e2d_1"], (
        f"gensal state list mismatch: {syn_states}")

    # Cross-check that genrou on a similar grid carries e1d
    grid_gr = _smib_grid(syn_overrides={"X1q": 0.55, "T1q0": 0.4},
                         model_type="genrou")
    mc_r = _build_and_ini(grid_gr, "genrou_states")
    syn_states_r = [s for s in mc_r.x_list if s.endswith("_1")]
    assert "e1d_1" in syn_states_r, (
        f"genrou must carry e1d as a state: {syn_states_r}")


def test_saturation_daxis_only():
    """With saturation active, S_at depends only on e1q (not e1d / e2d).

    Practical check: build the model with saturation on, run, read S_at
    and confirm it equals B_sat * max(e1q - A_sat, 0)^2 / |e1q| within
    Newton tolerance. PSS/E GENSAL uses d-axis saturation only — the
    formula uses just e1q, not the air-gap flux magnitude that includes
    the q-axis EMF.
    """
    mc = _build_and_ini(_smib_grid(syn_overrides={
        "S_10": 0.10, "S_12": 0.30, "v_f": 2.0,
    }), "gensal_sat")

    e1q = float(mc.get_value("e1q_1"))
    S_at = float(mc.get_value("S_at_1"))
    # Saturation constants computed by gensal._saturation_constants
    from pydae.bps.syns.gensal import _saturation_constants
    A_sat, B_sat = _saturation_constants(0.10, 0.30)
    expected = B_sat * max(e1q - A_sat, 0.0) ** 2 / abs(e1q)
    assert S_at == pytest.approx(expected, rel=1e-6, abs=1e-9), (
        f"S_at={S_at} expected={expected} e1q={e1q} A={A_sat} B={B_sat}")
