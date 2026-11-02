# -*- coding: utf-8 -*-
"""Reduced-DAE explicit runtime: ``pydae.core.builder.dae_reduce`` and
``CasadiModel.run(method='rk4')``.

Structure is checked on hand-made DAEs whose explicit/implicit split is
known; trajectories against ``idas_grid`` (adaptive, error-controlled) on
the same models.
"""
import casadi as ca
import numpy as np
import pytest
from pydae.core.builder.casadi_builder import CasadiBuilder
from pydae.core.builder.dae_reduce import reduce_explicit_algebraics
from pydae.core.model.casadi_model import CasadiModel

pytestmark = pytest.mark.core


def _syms(names):
    return [ca.SX.sym(n) for n in names.split()]


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
def test_chain_loop_parameter_and_implicit():
    """z1 needs z2 (chain), z2 has a parameter coefficient, z4/z5 form a linear
    loop, z3 is nonlinear in itself and must stay implicit."""
    x1, x2, z1, z2, z3, z4, z5, L, u = _syms("x1 x2 z1 z2 z3 z4 z5 L u")
    x, z, p = ca.vertcat(x1, x2), ca.vertcat(z1, z2, z3, z4, z5), ca.vertcat(L, u)
    g = ca.vertcat(z1 - (u - z2),
                   L * z2 - x1,
                   z3 + z3 ** 3 - x2,
                   z4 - z5 - x1,
                   z5 - 0.5 * z4)
    f = ca.vertcat(-x1 + z1 + z4, z2 - x2 * z3)
    r = reduce_explicit_algebraics(x, z, p, f, g)
    assert r.z_exp_idx == [0, 1, 3, 4]
    assert r.z_imp_idx == [2]
    # the solutions, numerically: z2 = x1/L, z1 = u - z2, z4 = 2 x1, z5 = x1
    zf = ca.Function("zf", [x, r.z_imp, p], [r.z_full])
    val = np.array(zf([0.6, 0.0], [0.7], [2.0, 1.5])).ravel()
    assert np.allclose(val, [1.5 - 0.3, 0.3, 0.7, 1.2, 0.6])
    assert 1 in [j for j, _ in r.coeffs]                   # z2's solution divides by L


def test_saturation_and_state_dependent_coefficient_stay_implicit():
    x1, z1, z2 = _syms("x1 z1 z2")
    x, z, p = x1, ca.vertcat(z1, z2), ca.SX(0, 1)
    g = ca.vertcat(ca.fmin(ca.fmax(z1, -1), 1) - x1,     # not affine in z1
                   x1 * z2 - 1.0)                        # coefficient depends on the state
    r = reduce_explicit_algebraics(x, z, p, -x1 + z1 + z2, g)
    assert r.z_imp_idx == [0, 1] and r.z_exp_idx == []


def test_clip_of_others_is_explicit():
    """z = clip(expression of x) is explicit even though the clip is nonlinear:
    it is affine in z itself (a PI block's saturated output)."""
    x1, z1 = _syms("x1 z1")
    r = reduce_explicit_algebraics(x1, z1, ca.SX(0, 1), -z1, z1 - ca.fmin(ca.fmax(3 * x1, -1), 1))
    assert r.is_ode


def test_interpolant_output_is_explicit():
    """i = I(lam) from a B-spline: CasADi's derivative of the call keeps a
    zero-seeded forward node (``1 - fwd_lut(..., 0)``), so a Jacobian-based
    test would call the coefficient state-dependent and leave i implicit.
    The structural split must eliminate it."""
    grid = np.linspace(0.0, 1.0, 6)
    lut = ca.interpolant("lut", "bspline", [grid, grid], np.arange(2 * 36.0), {"degree": [3, 3]})
    a, b, i_d, i_q = _syms("a b i_d i_q")
    I_d, I_q = ca.vertsplit(lut(ca.vertcat(a, b)))
    r = reduce_explicit_algebraics(ca.vertcat(a, b), ca.vertcat(i_d, i_q), ca.SX(0, 1),
                                   ca.vertcat(-i_d, -i_q), ca.vertcat(i_d - I_d, I_q - i_q))
    assert r.is_ode


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
def _model(L=2.0, u=1.5, name="reduce_rt"):
    x1, x2, z1, z2, z3 = _syms("x1 x2 z1 z2 z3")
    Ls, us = _syms("L u")
    sys_dict = {
        "name": name,
        "f_list": [-x1 + z1, z2 - x2 * z3 - 0.2 * x2],
        "x_list": [x1, x2],
        "g_list": [z1 - (us - z2), Ls * z2 - x1, z3 + z3 ** 3 - x2],
        "y_ini_list": [z1, z2, z3], "y_run_list": [z1, z2, z3],
        "u_ini_dict": {"u": u}, "u_run_dict": {"u": u},
        "params_dict": {"L": L},
        "h_dict": {"out": z1 + z3},
        "xy_0_dict": {"x1": 0.5, "x2": 0.5, "z1": 1.0, "z2": 0.2, "z3": 0.4},
    }
    m = CasadiModel(CasadiBuilder(sys_dict).build(), integrator_reltol=1e-10, integrator_abstol=1e-12)
    m.ini({}, sys_dict["xy_0_dict"])
    return m


def test_rk4_matches_idas_through_an_input_step():
    ref, rk = _model(name="reduce_ref"), _model(name="reduce_rk")
    for m, meth in ((ref, "idas_grid"), (rk, "rk4")):
        m.Dt = 1e-3
        m.run(1.0, {"u": 3.0}, method=meth)
        m.run(2.0, {"u": 0.5}, method=meth)
        m.post()
    assert np.allclose(rk.Time, ref.Time)
    for n in ("x1", "x2", "z1", "z2", "z3", "out"):
        assert np.allclose(rk.get_values(n), ref.get_values(n), atol=1e-7), n
    assert rk._explicit_runner.reduced.z_imp_idx == [2]


def test_rk4_algebraics_consistent_after_set_value_and_post():
    """No recalculate_algebraics: the algebraics come from the state. And a
    run after post() keeps appending."""
    m = _model()
    m.Dt = 1e-3
    m.run(0.1, method="rk4")
    m.post()
    m.set_value("u", 4.0)
    m.run(0.2, method="rk4")
    m.post()
    z1, z2, x1 = m.get_values("z1"), m.get_values("z2"), m.get_values("x1")
    assert len(m.Time) == 201
    k = 101                                                # first point after the change
    assert z2[k] == pytest.approx(x1[k] / 2.0, abs=1e-12)
    assert z1[k] == pytest.approx(4.0 - z2[k], abs=1e-12)


def test_rk4_divergence_raises_and_leaves_the_state():
    # one (explicit) algebraic variable: CasadiBuilder cannot build a model with none
    x1, z1 = _syms("x1 z1")
    sys_dict = {"name": "stiff", "f_list": [-1e5 * z1], "x_list": [x1], "g_list": [z1 - x1],
                "y_ini_list": [z1], "y_run_list": [z1], "u_ini_dict": {}, "u_run_dict": {},
                "params_dict": {}, "h_dict": {}, "xy_0_dict": {"x1": 1.0, "z1": 1.0}}
    m = CasadiModel(CasadiBuilder(sys_dict).build())
    m.ini({}, {"x1": 1.0, "z1": 1.0})
    m.x, m.y_run = np.array([1.0]), np.array([1.0])   # ini() of a pure decay finds 0; start at 1
    m.Dt = 1e-3                                            # |lambda| Dt = 100 >> 2.8
    with pytest.raises(RuntimeError, match="stiff"):
        m.run(1.0, method="rk4")
    assert m.t == 0.0 and m.x[0] == 1.0
    m.run(1e-3, method="rk4", substeps=100)                # 1e-5 per substep: stable
    assert m.x[0] == pytest.approx(np.exp(-100.0), rel=1e-3, abs=1e-12)


def test_rk4_zero_coefficient_raises():
    m = _model()
    m.set_value("L", 0.0)
    m.Dt = 1e-3
    with pytest.raises(ZeroDivisionError):
        m.run(0.01, method="rk4")


def test_rk4_on_a_control_block_loop():
    """A pydae.control PI in unity feedback (saturation, anti-windup): rk4
    against idas_grid through a step into saturation.

    This one is NOT reducible to an ODE, and correctly so: z = clip(K_p u +
    K_i xi) with u = r - z is an algebraic loop through the saturation (the
    proportional path feeds the output straight back), so whichever variable
    is eliminated the other is nonlinear in itself and stays implicit --
    Newton inside each RK4 stage."""
    from pydae.control.continuos.pi import _build_pi_loop_model
    ref = _build_pi_loop_model(name="r", z_min=-1.0, z_max=1.0)
    rk = _build_pi_loop_model(name="k", z_min=-1.0, z_max=1.0)
    ref._integrator_reltol, ref._integrator_abstol = 1e-10, 1e-12
    for m, meth, n in ((ref, "idas_grid", "r"), (rk, "rk4", "k")):
        m.Dt = 1e-3
        m.run(5.0, {f"r_pi_{n}": 2.0}, method=meth)
        m.post()
    assert len(rk._explicit_runner.reduced.z_imp_idx) == 1
    assert np.allclose(rk.get_values("xi_pi_k"), ref.get_values("xi_pi_r"), atol=1e-6)
    assert np.allclose(rk.get_values("z_pi_k"), ref.get_values("z_pi_r"), atol=1e-6)


def test_rk4_one_step_at_a_time_equals_one_span():
    """A user's control loop calls run(t + Dt) every period: that path (a
    FunctionBuffer evaluated in place) must give the batch path's trajectory,
    algebraics and outputs, including with an implicit remainder (z3)."""
    batch, loop = _model(name="reduce_batch"), _model(name="reduce_loop")
    for m in (batch, loop):
        m.Dt = 1e-3
    batch.run(0.2, {"u": 3.0}, method="rk4")
    loop.set_value("u", 3.0)
    for _ in range(200):
        loop.run(loop.t + loop.Dt, method="rk4")
    batch.post()
    loop.post()
    assert np.allclose(loop.Time, batch.Time)
    for n in ("x1", "x2", "z1", "z2", "z3", "out"):
        assert np.allclose(loop.get_values(n), batch.get_values(n), rtol=1e-12, atol=1e-9), n
