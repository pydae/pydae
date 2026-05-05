"""
Test: Compare ini() results between sympy/ctypes and CasADi backends.

This test builds the pendulum DAE with both backends,
calls ini() on each, and verifies they produce similar results.
"""

import numpy as np
import pytest

import sympy as sp
import casadi as ca


# ─── Helper: build pendulum sys_dict with SymPy or CasADi symbols ────────────────


def make_pendulum_sys(use_casadi=False):
    """Return pendulum system dictionary with SymPy or CasADi symbols."""
    if use_casadi:
        # Use CasADi SX symbols
        L, G, M, K_d, K_lam = ca.SX.sym("L"), ca.SX.sym("G"), ca.SX.sym("M"), ca.SX.sym("K_d"), ca.SX.sym("K_lam")
        p_x, p_y, v_x, v_y = ca.SX.sym("p_x"), ca.SX.sym("p_y"), ca.SX.sym("v_x"), ca.SX.sym("v_y")
        lam, f_x, theta, u_dummy = ca.SX.sym("lam"), ca.SX.sym("f_x"), ca.SX.sym("theta"), ca.SX.sym("u_dummy")
        atan2 = ca.atan2
    else:
        # Use SymPy symbols
        L, G, M, K_d, K_lam = sp.symbols("L,G,M,K_d,K_lam", real=True)
        p_x, p_y, v_x, v_y = sp.symbols("p_x,p_y,v_x,v_y", real=True)
        lam, f_x, theta, u_dummy = sp.symbols("lam,f_x,theta,u_dummy", real=True)
        atan2 = sp.atan2

    return {
        "name": "pendulum_cmp",
        "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3, "K_lam": 1e-6},
        "f_list": [
            v_x,
            v_y,
            (-2 * p_x * lam + f_x - K_d * v_x) / M,
            (-M * G - 2 * p_y * lam - K_d * v_y) / M,
        ],
        "g_list": [
            p_x**2 + p_y**2 - L**2 - lam * K_lam,
            -theta + atan2(p_x, -p_y) + u_dummy,
        ],
        "x_list": [p_x, p_y, v_x, v_y],
        "y_ini_list": [lam, f_x],
        "y_run_list": [lam, theta],
        "u_ini_dict": {"theta": np.deg2rad(5.0), "u_dummy": 0.0},
        "u_run_dict": {"f_x": 0, "u_dummy": 0.0},
        "h_dict": {
            "E_p": M * G * (p_y + L),
            "E_k": 0.5 * M * (v_x**2 + v_y**2),
        },
    }


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sympy_sys():
    return make_pendulum_sys(use_casadi=False)


@pytest.fixture
def casadi_sys():
    return make_pendulum_sys(use_casadi=True)


@pytest.fixture
def sympy_model(sympy_sys, tmp_path):
    """Build pendulum with sympy/ctypes and return initialized Model."""
    import os

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        from pydae.core import Builder, Model

        bld = Builder(sympy_sys, target="ctypes", sparse=False)
        bld.build()
        model = Model("pendulum_cmp")
        model.ini(
            {"theta": np.deg2rad(30), "K_d": 10.0},
            xy_0={"p_x": 5.21*np.sin(np.deg2rad(30)), "p_y": -5.21*np.cos(np.deg2rad(30)), "lam": 10.0, "f_x": 1.0},
        )
        yield model
    finally:
        os.chdir(orig)


@pytest.fixture
def casadi_model(casadi_sys, tmp_path):
    """Build pendulum with CasADi and return initialized CasadiModel."""
    import os

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        from pydae.core.builder.casadi_builder import CasadiBuilder
        from pydae.core.model.casadi_model import CasadiModel

        cb = CasadiBuilder(casadi_sys)
        cb.build()
        model = CasadiModel(cb)
        model.use_external_newton(True)
        model.ini(
            {"theta": np.deg2rad(30), "K_d": 10.0},
            xy_0={"p_x": 5.21*np.sin(np.deg2rad(30)), "p_y": -5.21*np.cos(np.deg2rad(30)), "lam": 10.0, "f_x": 1.0},
        )
        yield model
    finally:
        os.chdir(orig)


# ─── Comparison tests ────────────────────────────────────────────────


class TestPendulumIniComparison:
    """Verify sympy/ctypes and CasADi produce similar ini() results."""

    def test_states_match(self, sympy_model, casadi_model):
        """States x should match within tolerance."""
        for name in ["p_x", "p_y", "v_x", "v_y"]:
            v_sym = sympy_model.get_value(name)
            v_cas = casadi_model.get_value(name)
            # Use atol=1e-20 for near-zero values to handle floating-point zero
            atol = 1e-20 if abs(v_sym) < 1e-10 and abs(v_cas) < 1e-10 else 0.0
            np.testing.assert_allclose(
                v_sym,
                v_cas,
                rtol=1e-6,
                atol=atol,
                err_msg=f"State {name} differs: sympy={v_sym}, casadi={v_cas}",
            )

    def test_algebraic_match(self, sympy_model, casadi_model):
        """Algebraic variables should match within tolerance."""
        for name in ["lam"]:
            v_sym = sympy_model.get_value(name)
            v_cas = casadi_model.get_value(name)
            np.testing.assert_allclose(
                v_sym,
                v_cas,
                rtol=1e-6,
                err_msg=f"Algebraic {name} differs: sympy={v_sym}, casadi={v_cas}",
            )

    def test_output_match(self, sympy_model, casadi_model):
        """Output variables (theta) should match within tolerance."""
        for name in ["theta"]:
            v_sym = sympy_model.get_value(name)
            v_cas = casadi_model.get_value(name)
            np.testing.assert_allclose(
                v_sym,
                v_cas,
                rtol=1e-6,
                err_msg=f"Output {name} differs: sympy={v_sym}, casadi={v_cas}",
            )

    def test_input_match(self, sympy_model, casadi_model):
        """Input variables should match within tolerance."""
        for name in ["f_x"]:
            v_sym = sympy_model.get_value(name)
            v_cas = casadi_model.get_value(name)
            np.testing.assert_allclose(
                v_sym,
                v_cas,
                rtol=1e-6,
                err_msg=f"Input {name} differs: sympy={v_sym}, casadi={v_cas}",
            )

    def test_A_matrix_match(self, sympy_model, casadi_model):
        """State matrix A should match within tolerance."""
        A_sym = sympy_model.A_eval()
        A_cas = casadi_model.A_eval()
        np.testing.assert_allclose(
            A_sym,
            A_cas,
            rtol=1e-5,
            err_msg="State matrix A differs between sympy and CasADi",
        )

    def test_h_outputs_match(self, sympy_model, casadi_model):
        """Output expressions (E_p, E_k) should match within tolerance."""
        for name in ["E_p", "E_k"]:
            v_sym = sympy_model.get_value(name)
            v_cas = casadi_model.get_value(name)
            # Use atol for near-zero values like E_k (kinetic energy = 0 at equilibrium)
            atol = 1e-20 if abs(v_sym) < 1e-10 and abs(v_cas) < 1e-10 else 0.0
            np.testing.assert_allclose(
                v_sym,
                v_cas,
                rtol=1e-5,
                atol=atol,
                err_msg=f"Output {name} differs: sympy={v_sym}, casadi={v_cas}",
            )
