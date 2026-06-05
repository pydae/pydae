"""Tests for the genrou (IEEE 1110 Model 2.2 / Anderson-Fouad) machine.

Three checks:

- ``test_smoke`` — ini converges from cold seeds, a 5-second time integration
  with constant inputs stays flat at the initial operating point.

- ``test_parity_milano6ord_no_xl`` — when milano6ord is fed ``X_l = 0``,
  ``T_AA = 0`` and saturation off, its **terminal-referred** steady-state
  quantities (``delta, omega, V, p_g, q_g, v_f, i_d, i_q, e2q, e2d``) and
  the electromechanical swing mode must agree with genrou to machine
  precision. The transient EMFs ``e1q, e1d`` and the fast subtransient
  eigenvalues differ structurally because milano6ord carries a Marconato
  cross-coupling term and genrou (clean Anderson-Fouad) does not.

- ``test_nts_damping`` — runs the migrated NTS benchmark
  (``benchmarks_public/nts/cases/base/nts_base.hjson``) end-to-end and
  asserts the dominant electromechanical-swing damping falls inside NTS
  Figura 18's reference band over the X_L sweep specified by NTS §5.9.2.
  Skipped when the benchmark dir is not present.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.bps

NTS_DIR = Path("/Users/jmmauricio/workspace/benchmarks_public/nts/cases/base")


NTS_GEN1_PARAMS = {
    # NTS Tabla 45 — Iberdrola 1500 MVA unit. X_l is intentionally omitted.
    "S_n": 1500e6, "F_n": 50.0,
    "X_d": 2.135, "X_q": 2.046,
    "X1d": 0.34,  "T1d0": 6.47,
    "X1q": 0.573, "T1q0": 0.61,
    "X2d": 0.269, "T2d0": 0.022,
    "X2q": 0.269, "T2q0": 0.034,
    "R_a": 0.0, "H": 6.3, "D": 0.0,
    "S_10": 0.1275, "S_12": 0.2706,
    "K_sec": 0.0, "K_delta": 0.0001,
}


def _smib_grid(syn_overrides, model_type="genrou", X_L_line=0.15):
    """Single-machine-infinite-bus dict for genrou / milano6ord parity tests."""
    syn = dict(NTS_GEN1_PARAMS)
    syn.update({"bus": "1", "type": model_type})
    syn.update(syn_overrides or {})
    return {
        "system": {"name": "test_genrou", "S_base": 1500e6},
        "buses": [
            {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
            {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
        ],
        "lines": [
            {"bus_j": "1", "bus_k": "2",
             "X_pu": X_L_line, "R_pu": 0.0, "S_mva": 20000},
        ],
        "shunts": [{"bus": "1", "X_pu": 1e6, "R_pu": 0.0, "S_mva": 100}],
        "syns": [syn],
        "sources": [{"type": "vsource", "bus": "2"}],
    }


def _build_and_ini(grid_dict, name, xy_overrides=None):
    """Build the model on the CasADi backend and run ini(); return the model."""
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
    seed = dict(grid.dae["xy_0_dict"])
    if xy_overrides:
        seed.update(xy_overrides)
    mc.ini({}, xy_0=seed)
    return mc


def test_smoke():
    """Ini converges; a 5-second run with constant inputs stays at OP."""
    grid_dict = _smib_grid(
        syn_overrides={"S_10": 0.0, "S_12": 0.0,
                       "v_f": 1.5, "p_m": 0.5},
        model_type="genrou",
    )
    mc = _build_and_ini(grid_dict, "genrou_smoke")

    delta0 = float(mc.get_value("delta_1"))
    omega0 = float(mc.get_value("omega_1"))
    p_g0   = float(mc.get_value("p_g_1"))

    # omega is mathematically free with K_delta small and no AGC / governor
    # pinning it (the only constraint is delta-equation + COI). Just check
    # it's near unity (no runaway) and that p_g matches the requested
    # operating point.
    assert omega0 == pytest.approx(1.0, abs=5e-2)
    assert p_g0   == pytest.approx(0.5, abs=1e-3)
    assert 0.0 < delta0 < np.pi / 2  # physically sensible load angle

    mc.run(5.0, {})
    mc.post()

    # Constant inputs → flat trajectory.
    assert float(mc.get_value("omega_1")) == pytest.approx(omega0, abs=1e-5)
    assert float(mc.get_value("delta_1")) == pytest.approx(delta0, abs=1e-4)
    assert float(mc.get_value("p_g_1"))   == pytest.approx(p_g0,   abs=1e-4)


def test_parity_milano6ord_no_xl():
    """milano6ord(X_l=0, T_AA=0, S=0) and genrou(S=0) agree on terminal
    quantities to machine precision and on the swing mode to 1e-3."""
    from pydae.ssa.ssa import eig

    syn_ovr_common = {"S_10": 0.0, "S_12": 0.0, "v_f": 1.5, "p_m": 0.5}
    syn_ovr_m6 = dict(syn_ovr_common); syn_ovr_m6.update({"X_l": 0.0, "T_AA": 0.0})

    grid_m6 = _smib_grid(syn_overrides=syn_ovr_m6,    model_type="milano6ord")
    grid_gr = _smib_grid(syn_overrides=syn_ovr_common, model_type="genrou")

    m_m6 = _build_and_ini(grid_m6, "parity_m6")
    m_gr = _build_and_ini(grid_gr, "parity_gr")

    # Terminal-referred quantities are insensitive to the Marconato cross-
    # coupling. They must agree to machine precision.
    for var in ("delta_1", "omega_1", "p_g_1", "q_g_1",
                "v_f_1", "i_d_1", "i_q_1",
                "e2q_1", "e2d_1", "V_1"):
        a = float(m_m6.get_value(var))
        b = float(m_gr.get_value(var))
        assert abs(a - b) < 1e-6, f"{var}: milano6ord={a}, genrou={b}"

    # Electromechanical swing mode (the slowest oscillatory pole) must agree
    # to ~1e-3. The subtransient poles differ structurally and are not
    # compared.
    eig(m_m6); eig(m_gr)
    def _swing(model):
        osc = [z for z in model.eigenvalues if abs(z.imag) > 1e-3]
        return min(osc, key=lambda z: abs(z.imag))
    s_m6 = _swing(m_m6)
    s_gr = _swing(m_gr)
    assert abs(s_m6.real - s_gr.real) < 1e-3, f"real: {s_m6} vs {s_gr}"
    assert abs(abs(s_m6.imag) - abs(s_gr.imag)) < 1e-3, f"imag: {s_m6} vs {s_gr}"


def _nts_smib_grid(X_L_line, with_pss=True):
    """NTS bus-1 generator + ST4B AVR (+ PSS2A) vs infinite bus.

    Drops the X_l override the legacy nts_base.hjson carried, takes NTS
    Tabla 45 / 46 parameters literally. Topology: gen at bus 1 → step-up
    high-side bus 2 → swept tie → infinite bus 3. AVR monitors bus 2
    (so it does not collide with the vsource pin at bus 3).
    """
    syn = dict(NTS_GEN1_PARAMS)
    syn.update({
        "bus": "1", "type": "genrou",
        "avr": {"type": "st4b", "bus": "2",
                "T_R": 0.02,
                "K_PR": 3.15, "K_IR": 3.15,
                "V_RMAX": 1.0, "V_RMIN": -0.87,
                "T_A": 0.02,
                "K_PM": 1.0, "K_IM": 0.0,
                "V_MMAX": 1.0, "V_MMIN": -0.87,
                "K_G": 0.0, "K_P": 6.5,
                "V_BMAX": 8.0,
                "v_ref": 1.0},
        "lc": {"K_i": 0.0001, "p_c_lc": 0.9},
    })
    if with_pss:
        syn["pss"] = {"type": "pss2a",
                      "T_w1": 2.0, "T_w2": 2.0,
                      "T_6": 0.0,
                      "T_w3": 2.0, "T_w4": 0.0,
                      "T_7": 2.0, "K_s2": 0.158,
                      "K_s3": 1.0,
                      "T_8": 0.0, "T_9": 0.1,
                      "M": 5, "N": 1,
                      "K_s1": 17.069,
                      "T_1": 0.28, "T_2": 0.04,
                      "T_3": 0.28, "T_4": 0.12,
                      "V_STmax": 0.1, "V_STmin": -0.1}
    return {
        "system": {"name": "nts_smib", "S_base": 1500e6},
        "buses": [
            {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 19.0},
            {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 230.0},
            {"name": "3", "P_W": 0.0, "Q_var": 0.0, "U_kV": 230.0},
        ],
        "lines": [
            {"bus_j": "1", "bus_k": "2",
             "X_pu": 0.15, "R_pu": 0.0, "S_mva": 1500},  # step-up
            {"bus_j": "2", "bus_k": "3",
             "X_pu": X_L_line, "R_pu": 0.0, "S_mva": 1500},  # swept tie
        ],
        "shunts": [{"bus": "1", "X_pu": 1e6, "R_pu": 0.0, "S_mva": 100}],
        "syns": [syn],
        "sources": [{"type": "vsource", "bus": "3"}],
    }


def _dominant_electromech(model):
    """Inter-area swing mode = oscillatory mode in [0.2, 30] rad/s with the
    highest delta participation. PSS-state coupling pollutes the omega
    participation, so we score on delta only."""
    from pydae.ssa.ssa import eig
    eig(model)
    eigs = np.asarray(model.eigenvalues)
    V = model.right_eigenvectors
    W = model.left_eigenvectors      # already inv(V)
    # Kundur participation: P[k,i] = |V[k,i]| * |W[i,k]| (column-normalised)
    P = np.abs(V) * np.abs(W.T)
    P = P / P.sum(axis=0, keepdims=True)
    idx_delta = list(model.x_list).index("delta_1")
    osc = [i for i, z in enumerate(eigs)
           if 0.2 < abs(z.imag) < 30.0 and z.imag > 0]
    if not osc:
        raise RuntimeError(f"no electromech mode in {eigs}")
    _, ib = max((P[idx_delta, i], i) for i in osc)
    return complex(eigs[ib])


@pytest.fixture(scope="module")
def nts_model():
    """Build the migrated NTS benchmark once for the X_L sweep."""
    if not (NTS_DIR / "nts_base.hjson").exists():
        pytest.skip(f"NTS benchmark dir not present: {NTS_DIR}")
    if not (NTS_DIR / "nts_base_xy_0.json").exists():
        pytest.skip("NTS xy_0 seed not present — run nts_base_ini_sympy.py once")

    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    cwd = os.getcwd()
    os.chdir(NTS_DIR)
    try:
        grid = BpsBuilder("nts_base.hjson")
        grid.checker(); grid.uz_jacs = False
        grid.construct("nts_base")
        Builder(grid.sys_dict, target="cffi", sparse=False).build()
        model = Model("nts_base")
        yield model
    finally:
        os.chdir(cwd)


@pytest.mark.slow
@pytest.mark.parametrize("X_L_line, zeta_lo, zeta_hi", [
    (0.30, 0.14, 0.18),
    (0.35, 0.13, 0.17),
    (0.40, 0.12, 0.16),
])
def test_nts_damping(nts_model, X_L_line, zeta_lo, zeta_hi):
    """NTS Figura 18 inter-area-mode damping sweep on the migrated benchmark.

    With genrou and NTS literal parameters (no X_l override) the dominant
    electromechanical mode damping decreases monotonically from ~16% at
    X_L=0.30 to ~14% at X_L=0.40 — matching NTS Figura 18. The legacy
    milano6ord with literal X_l = 0.234 sat several percentage points
    above this band.
    """
    from pydae.bps.lines import change_line

    cwd = os.getcwd()
    os.chdir(NTS_DIR)
    try:
        change_line(nts_model, {"bus_j": "2", "bus_k": "3",
                                "X_pu": X_L_line, "R_pu": 0.0,
                                "Bs_pu": 0.0, "S_mva": 100})
        assert nts_model.ini({}, "nts_base_xy_0.json"), (
            f"NTS ini failed at X_L={X_L_line}")
        nts_model.A_eval()
    finally:
        os.chdir(cwd)

    mode = _dominant_electromech(nts_model)
    zeta = -mode.real / np.sqrt(mode.real ** 2 + mode.imag ** 2)
    f = abs(mode.imag) / (2 * np.pi)
    print(f"X_L={X_L_line}: lambda={mode.real:+.4f}{mode.imag:+.4f}j,"
          f" f={f:.3f} Hz, zeta={zeta * 100:.2f}%")

    assert zeta_lo <= zeta <= zeta_hi, (
        f"dominant electromech damping zeta={zeta:.4f} outside"
        f" NTS Figura 18 band [{zeta_lo}, {zeta_hi}] at X_L={X_L_line}")
