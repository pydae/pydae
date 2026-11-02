# -*- coding: utf-8 -*-
"""pydae-emac mean-value DAE models on pydae's own solvers.

Every check is against something the code under test does not produce:
closed-form steady states, the eigenvalues the tuning rule predicts, the
first-order response it implies, Newton's second law, and the linear model
against the look-up-table model fed the same machine.
"""
import copy

import numpy as np
import pytest
from pydae.emac import EmacBuilder
from pydae.emac.luts import build_flux_tables, linear_fea_lut, linear_mean_torque

pytestmark = pytest.mark.emac

MACHINE = dict(type="pmsm_dq", name="1", R_s=0.12, L_d=2.0e-3, L_q=3.5e-3, Psi_pm=0.12, p=4,
               mech="imposed", omega_m=150.0)
CTRL = dict(type="foc_pi", name="1", machine="1", bandwidth=500.0, V_dc=540.0,
            i_d_ref=-5.0, i_q_ref=20.0)


def build(machine=None, ctrl=None, name="emac_test"):
    data = {"machines": [copy.deepcopy(machine or MACHINE)],
            "ctrls": [copy.deepcopy(ctrl or CTRL)]}
    drive = EmacBuilder(data)
    model = drive.build(name)
    model.ini({}, drive.sys_dict["xy_0_dict"])
    return drive, model


def test_steady_state_closed_form():
    _, m = build()
    i_d, i_q, w_e = CTRL["i_d_ref"], CTRL["i_q_ref"], MACHINE["p"] * MACHINE["omega_m"]
    R, L_d, L_q, Psi = (MACHINE[k] for k in ("R_s", "L_d", "L_q", "Psi_pm"))
    assert m.get_value("i_d_1") == pytest.approx(i_d, abs=1e-9)
    assert m.get_value("i_q_1") == pytest.approx(i_q, abs=1e-9)
    assert m.get_value("v_d_1") == pytest.approx(R * i_d - w_e * L_q * i_q, rel=1e-9)
    assert m.get_value("v_q_1") == pytest.approx(R * i_q + w_e * (L_d * i_d + Psi), rel=1e-9)
    assert m.get_value("T_e_1") == pytest.approx(linear_mean_torque(L_d, L_q, Psi, MACHINE["p"], i_d, i_q),
                                                 rel=1e-9)


def test_eigenvalues_are_the_tuned_poles():
    """K_p = w_bw L, K_i = K_p R/L with exact decoupling: the loop's poles are
    -w_bw on each axis, and the cancelled stator poles -R/L stay in the state
    space (unobservable from the reference)."""
    _, m = build()
    w_bw = 2 * np.pi * CTRL["bandwidth"]
    expected = np.sort([-w_bw, -w_bw, -MACHINE["R_s"] / MACHINE["L_d"], -MACHINE["R_s"] / MACHINE["L_q"]])
    eig = np.sort(np.real(m.eval_eigenvalues()))
    assert np.allclose(eig, expected, rtol=1e-6)


def test_step_response_first_order():
    """After a step in i_q_ref the current covers 1 - 1/e of it in 1/w_bw.

    The PI's proportional path jumps the algebraic voltage with the step, so
    the algebraics are re-solved at the discontinuity before the fixed-step
    run (as CasadiModel.run documents); without it the first step integrates
    the pre-step voltage and the response lags by ~Dt/2."""
    _, m = build()
    tau = 1 / (2 * np.pi * CTRL["bandwidth"])
    m.Dt = tau / 200
    m.set_value("i_q_ref_1", 30.0)
    m.recalculate_algebraics()
    m.run(tau, method="trapezoidal")
    m.post()
    assert m.get_values("i_q_1")[-1] == pytest.approx(20.0 + 10.0 * (1 - np.exp(-1)), rel=1e-4)
    assert m.get_values("i_d_1")[-1] == pytest.approx(CTRL["i_d_ref"], abs=1e-6)   # decoupled


def test_inverter_delay_keeps_the_steady_state():
    _, m0 = build()
    _, m1 = build(ctrl=dict(CTRL, T_delay=75e-6), name="emac_test_delay")
    for n in ("i_d_1", "i_q_1", "v_d_1", "v_q_1", "T_e_1"):
        assert m1.get_value(n) == pytest.approx(m0.get_value(n), rel=1e-9, abs=1e-9)
    assert len(m1.x_names) == len(m0.x_names) + 2               # the two lpf1 states


def test_free_rotor_ini_run_swap():
    """mech = free: ini solves the load torque that holds the given speed; run
    keeps it (the speed stays), and a load step decelerates at dT/J."""
    J, B = 0.05, 1e-3
    _, m = build(machine=dict(MACHINE, mech="free", J=J, B=B), name="emac_test_free")
    T_e = m.get_value("T_e_1")
    assert m.get_value("T_l_1") == pytest.approx(T_e - B * MACHINE["omega_m"], rel=1e-9)
    m.Dt = 1e-5
    m.run(0.002, method="trapezoidal")
    assert m.get_value("omega_m_1") == pytest.approx(MACHINE["omega_m"], rel=1e-9)
    dT, t = 2.0, 1e-3
    m.run(0.002 + t, {"T_l_1": m.get_value("T_l_1") + dT}, method="trapezoidal")
    # the current loop holds T_e; friction changes by B * d(omega), negligible here
    assert m.get_value("omega_m_1") - MACHINE["omega_m"] == pytest.approx(-dT / J * t, rel=1e-3)


def test_lut_machine_matches_linear_machine():
    """pmsm_lut fed the linear machine's (rippled) FEA-format table gives the
    linear machine's steady state: the inversion, the Fourier mean and the
    B-spline are checked against algebra."""
    lut = linear_fea_lut(MACHINE["L_d"], MACHINE["L_q"], MACHINE["Psi_pm"], MACHINE["p"], i_max=40.0,
                         torque_ripple=0.05, flux_ripple=0.02)
    tables = build_flux_tables(lut, n_lam=30)
    lut_machine = dict(type="pmsm_lut", name="1", R_s=MACHINE["R_s"], tables=tables,
                       mech="imposed", omega_m=MACHINE["omega_m"])
    _, m_lin = build()
    _, m_lut = build(machine=lut_machine, ctrl=dict(CTRL, L_d=MACHINE["L_d"], L_q=MACHINE["L_q"]),
                     name="emac_test_lut")
    for n in ("i_d_1", "i_q_1", "lam_d_1", "lam_q_1", "v_d_1", "v_q_1", "T_e_1"):
        assert m_lut.get_value(n) == pytest.approx(m_lin.get_value(n), rel=1e-5, abs=1e-6), n


def test_lut_machine_needs_casadi():
    lut = linear_fea_lut(1e-3, 1e-3, 0.1, 2, i_max=10.0)
    data = {"machines": [dict(type="pmsm_lut", name="1", tables=build_flux_tables(lut, n_lam=12))]}
    with pytest.raises(NotImplementedError):
        EmacBuilder(data, use_casadi=False).construct("x")


@pytest.mark.parametrize("kind", ["pmsm_dq", "pmsm_lut"])
def test_rk4_reduces_the_drive_to_an_ode(kind):
    """run(method='rk4') eliminates every algebraic variable of machine + FOC
    (pmsm_dq through its parameter coefficient L, pmsm_lut through the spline
    call) and follows idas_grid through a current step."""
    if kind == "pmsm_dq":
        machine, ctrl = MACHINE, CTRL
    else:
        lut = linear_fea_lut(MACHINE["L_d"], MACHINE["L_q"], MACHINE["Psi_pm"], MACHINE["p"], i_max=40.0)
        machine = dict(type="pmsm_lut", name="1", R_s=MACHINE["R_s"], tables=build_flux_tables(lut, n_lam=20),
                       mech="free", J=0.05, omega_m=MACHINE["omega_m"])
        ctrl = dict(CTRL, L_d=MACHINE["L_d"], L_q=MACHINE["L_q"])
    runs = {}
    for meth in ("idas_grid", "rk4"):
        _, m = build(machine=machine, ctrl=ctrl, name=f"emac_rk4_{kind}_{meth}")
        m._integrator_reltol, m._integrator_abstol = 1e-10, 1e-10
        m.Dt = 20e-6
        m.run(0.004, {"i_q_ref_1": 30.0}, method=meth)
        m.post()
        runs[meth] = m
    assert runs["rk4"]._explicit_runner.reduced.is_ode
    for n in ("i_q_1", "T_e_1"):
        assert np.allclose(runs["rk4"].get_values(n), runs["idas_grid"].get_values(n), rtol=1e-5, atol=1e-5), n
