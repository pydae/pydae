# -*- coding: utf-8 -*-
"""pydae-emac switched drive and FEA tables, on a synthetic machine whose
theta-mean is known in closed form."""
import numpy as np
import pytest
from pydae.emac.drive import (
    PIController,
    PredictiveController,
    PwmPlant,
    build_control_period,
    build_machine,
    build_window,
    simulate,
    time_average,
)
from pydae.emac.luts import build_flux_tables, linear_fea_lut, linear_mean_torque, load_fea_lut, save_fea_lut

pytestmark = pytest.mark.emac

L_D, L_Q, PSI, P, R, J = 2.0e-3, 3.5e-3, 0.12, 4, 0.12, 0.05
V_DC, T_S, OMEGA_M = 540.0, 50e-6, 150.0
I_REF = np.array([-5.0, 20.0])


@pytest.fixture(scope="module")
def lut():
    return linear_fea_lut(L_D, L_Q, PSI, P, i_max=40.0, torque_ripple=0.05, flux_ripple=0.02)


@pytest.fixture(scope="module")
def tables(lut):
    return build_flux_tables(lut, n_lam=30)


def test_table_file_round_trip(lut, tmp_path):
    save_fea_lut(tmp_path / "FluxMap_dqt.npz", lut)
    back = load_fea_lut(tmp_path / "FluxMap_dqt.npz")
    for k in ("i_d", "i_q", "theta_e", "lam_d", "lam_q", "T"):
        assert np.allclose(back[k], lut[k])
    assert back["p"] == lut["p"]


def test_tables_read_back_the_currents(lut, tables):
    """At every FEA angle, the model's currents at the FEA fluxes of I_REF are I_REF."""
    _, currents = build_machine(tables, R, J)
    for th in lut["theta_e"]:
        lam = [L_D * I_REF[0] + PSI + 0.02 * PSI * np.cos(6 * th), L_Q * I_REF[1] - 0.02 * PSI * np.sin(6 * th)]
        i_d, i_q, _ = np.array(currents([*lam, 0.0, th / P])).ravel()
        assert i_d == pytest.approx(I_REF[0], abs=1e-4)
        assert i_q == pytest.approx(I_REF[1], abs=1e-4)
    assert np.allclose(tables["lam_0"], [PSI, 0.0], atol=1e-9)


@pytest.mark.parametrize("mode", ["averaged", "switched"])
def test_rated_torque(tables, mode):
    """At the references, the mean torque is the closed-form one.

    Two things the measurement needs: the PI cancels the stator pole, so the
    loop rejects its decoupling error only at the machine's own L/R -- run
    5 L_q/R; and the ripple biases any mean not taken over whole electrical
    periods -- average over the last 4."""
    ode, currents = build_machine(tables, R, J)
    Kp = 2 * np.pi * 500.0 * np.array([L_D, L_Q])
    period = build_control_period(build_window(ode), currents, P, V_DC, T_S,
                                  PIController(Kp, Kp * R / np.array([L_D, L_Q])), mode=mode)
    t_end, f_e = 5 * L_Q / R + 0.05, P * OMEGA_M / (2 * np.pi)
    n = int(round(t_end / T_S))
    r = simulate(period, currents, np.array([PSI, 0.0, OMEGA_M, 0.0]), T_S, n, I_REF)
    last = r["t"] >= n * T_S - 4 / f_e
    sampled = last & np.isclose(np.mod(r["t"] + 1e-9, T_S), 1e-9, atol=1e-8)
    assert r["y"][0, sampled].mean() == pytest.approx(I_REF[0], abs=1e-3)   # what the PI regulates
    assert r["y"][1, sampled].mean() == pytest.approx(I_REF[1], abs=1e-3)
    _, _, T = time_average(r["t"][last], r["y"][:, last])
    assert T == pytest.approx(linear_mean_torque(L_D, L_Q, PSI, P, *I_REF), rel=1e-3)


def test_predictive_controller_reaches_the_reference(tables):
    ode, currents = build_machine(tables, R, J)
    period = build_control_period(build_window(ode), currents, P, V_DC, T_S,
                                  PredictiveController(R, L_D, L_Q), mode="averaged")
    r = simulate(period, currents, np.array([PSI, 0.0, OMEGA_M, 0.0]), T_S, int(0.01 / T_S), I_REF)
    last = r["t"] >= 0.005
    i_d, i_q, _ = time_average(r["t"][last], r["y"][:, last])
    assert i_q == pytest.approx(I_REF[1], rel=1e-2)


@pytest.mark.parametrize("t_dead, compensate", [(0.0, False), (2e-6, False), (1e-6, True)])
def test_pwm_plant_with_a_python_processor_reproduces_simulate(tables, t_dead, compensate):
    """PwmPlant driven one period at a time by a Python loop doing what
    PIController does (NumPy modulation helpers) gives the compiled
    simulate()'s trajectory: same switching instants, states to rounding.
    The loop measures as a drive does -- phase currents and the encoder
    angle -- and makes dq itself (Clarke + Park); only the fluxes come from
    inside the plant, because PIController's decoupling uses the true ones."""
    from pydae.emac.drive.modulation import (
        abc_to_dq,
        compensate_dead_time,
        dq_to_alpha_beta,
        limit_voltage,
        svpwm_duties,
    )
    Kp = 2 * np.pi * 500.0 * np.array([L_D, L_Q])
    Ki = Kp * R / np.array([L_D, L_Q])
    x0, n = np.array([PSI, 0.0, OMEGA_M, 0.0]), 1500
    ode, currents = build_machine(tables, R, J)
    period = build_control_period(build_window(ode), currents, P, V_DC, T_S, PIController(Kp, Ki),
                                  t_dead=t_dead, compensate=compensate)
    ref = simulate(period, currents, x0, T_S, n, I_REF)

    plant = PwmPlant(tables, R, J, V_DC, T_S, t_dead=t_dead, x0=x0)
    xc, d_wait = np.zeros(2), np.full(3, 0.5)
    for _ in range(n):
        m = plant.measurements()
        th, w = P * m["theta_m"], P * m["omega_m"]
        i_dq, lam = abc_to_dq(m["i_abc"], th), plant.x[:2]
        e = I_REF - i_dq
        v_dq = Kp * e + Ki * xc + np.array([-w * lam[1], w * lam[0]])
        v_ab, limited = limit_voltage(dq_to_alpha_beta(v_dq, th + 1.5 * T_S * w), V_DC / np.sqrt(3))
        xc = xc + (0.0 if limited else 1.0) * e * T_S
        d_new = svpwm_duties(v_ab, V_DC)
        if compensate:                          # for the NEXT period's ramp
            d_new = compensate_dead_time(d_new, m["i_abc"], 1.0 if (plant.k + 1) % 2 == 0 else 0.0, t_dead, T_S)
        plant.step(d_wait)
        d_wait = d_new
    r = plant.results()
    assert r["t"].shape == ref["t"].shape and np.allclose(r["t"], ref["t"], rtol=0, atol=1e-15)
    assert np.allclose(r["x"], ref["x"], rtol=0, atol=1e-12)
    assert np.allclose(r["y"], ref["y"], rtol=0, atol=1e-8)


def test_abc_to_dq_inverts_the_plant_convention(tables):
    from pydae.emac.drive.modulation import abc_to_dq
    plant = PwmPlant(tables, R, J, V_DC, T_S, x0=[PSI, 0.0, OMEGA_M, 0.7])
    plant.step([0.7, 0.2, 0.4])
    assert np.allclose(abc_to_dq(plant.i_abc, plant.theta_e), plant.i_dq, rtol=0, atol=1e-12)
    assert abs(plant.i_abc.sum()) < 1e-12


def test_pwm_plant_free_rotor_follows_the_load(tables):
    """step(d, T_load=...) frees the speed: J d(omega)/dt = T_e - T_load.

    The duties of zero voltage short-circuit the machine at speed, so it also
    brakes (T_e < 0): the speed change is the time-weighted mean of
    (T_e - T_load) / J over the run."""
    plant = PwmPlant(tables, R, J, V_DC, T_S, x0=[PSI, 0.0, OMEGA_M, 0.0])
    T_l, n, T_0 = 3.0, 20, plant.T_e
    for _ in range(n):
        plant.step(np.full(3, 0.5), T_load=T_l)
    r = plant.results()
    # the torque ramps fast here: trapezoidal mean over every sample, t = 0 included
    T_mean = np.trapezoid(np.r_[T_0, r["y"][2]], np.r_[0.0, r["t"]]) / r["t"][-1]
    assert T_mean < 0                                                  # it brakes
    assert plant.omega_m - OMEGA_M == pytest.approx((T_mean - T_l) / J * n * T_S, rel=0.02)


def test_sine_and_svpwm_duties_differ_only_by_a_common_shift():
    """Below their limits both modulators give the same line-to-line duties
    (what drives the currents of a star winding without neutral); svpwm adds a
    common-mode shift, and sine duties are the phase voltages themselves."""
    from pydae.emac.drive.modulation import sine_duties, svpwm_duties
    for ang in np.linspace(0.0, 2 * np.pi, 37):
        v_ab = 0.8 * V_DC / 2 * np.array([np.cos(ang), np.sin(ang)])   # inside both limits
        d_sin, d_sv = sine_duties(v_ab, V_DC), svpwm_duties(v_ab, V_DC)
        assert np.allclose(np.diff(d_sin), np.diff(d_sv), atol=1e-14)
        assert np.allclose(d_sin - d_sv, (d_sin - d_sv)[0], atol=1e-14)
        assert d_sin[0] == pytest.approx(0.5 + v_ab[0] / V_DC)


def test_skew_of_a_linear_machine_is_the_slice_average():
    """Skewing a linear salient machine (no ripple) by N slices at electrical
    angles a_k gives, in closed form, L_d' = L_d<cos^2> + L_q<sin^2>,
    L_q' = L_d<sin^2> + L_q<cos^2>, Psi' = Psi<cos>, no cross-coupling."""
    from pydae.emac.luts import skew_fea_lut
    lut = linear_fea_lut(L_D, L_Q, PSI, P, i_max=40.0)
    skew, n = 2 * np.pi / 48, 7                     # mechanical, slices
    a = P * skew * ((np.arange(n) + 0.5) / n - 0.5)
    c2, s2, cm = np.mean(np.cos(a) ** 2), np.mean(np.sin(a) ** 2), np.mean(np.cos(a))
    sk = skew_fea_lut(lut, skew, n)
    ID, IQ = np.meshgrid(lut["i_d"], lut["i_q"], indexing="ij")
    for j in (0, 7):                                # any angle: no ripple
        assert np.allclose(sk["lam_d"][:, :, j], (L_D * c2 + L_Q * s2) * ID + PSI * cm, atol=1e-9)
        assert np.allclose(sk["lam_q"][:, :, j], (L_D * s2 + L_Q * c2) * IQ, atol=1e-9)
