"""
Pendulum DAE example
====================

A damped pendulum modeled as a constrained mechanical system in (x, y)
coordinates with the rod length enforced by an algebraic constraint.

States (differential):
    p_x, p_y   position of the bob
    v_x, v_y   velocity of the bob

Algebraic variables:
    lam        Lagrange multiplier enforcing rod length L
    theta      angle of the rod from the downward vertical (output)

Inputs:
    f_x        horizontal external force on the bob

This is the example from the project README — kept as a runnable script
so users can verify their pydae installation works end-to-end.
"""

import numpy as np
import sympy as sym

from pydae.core import Builder, Model


def build():
    """Define the symbolic DAE and compile it to native C."""
    # Parameters
    L, G, M, K_d = sym.symbols("L,G,M,K_d", real=True)

    # States
    p_x, p_y, v_x, v_y = sym.symbols("p_x,p_y,v_x,v_y", real=True)

    # Algebraic variables / inputs
    lam, f_x, theta = sym.symbols("lam,f_x,theta", real=True)

    # Differential equations: dx/dt = f(x, y, u)
    dp_x = v_x
    dp_y = v_y
    dv_x = (-2 * p_x * lam + f_x - K_d * v_x) / M
    dv_y = (-M * G - 2 * p_y * lam - K_d * v_y) / M

    # Algebraic equations: 0 = g(x, y, u)
    # g_1: rod-length constraint (with a tiny lam regularization for numerics)
    # g_2: definition of the output angle theta
    g_1 = p_x ** 2 + p_y ** 2 - L ** 2 - lam * 1e-6
    g_2 = -theta + sym.atan2(p_x, -p_y)

    sys_dict = {
        "name": "pendulum",
        "params_dict": {"L": 5.21, "G": 9.81, "M": 10.0, "K_d": 1e-3},
        "f_list": [dp_x, dp_y, dv_x, dv_y],
        "g_list": [g_1, g_2],
        "x_list": [p_x, p_y, v_x, v_y],
        "y_ini_list": [lam, f_x],
        "y_run_list": [lam, theta],
        "u_ini_dict": {"theta": np.deg2rad(5.0)},
        "u_run_dict": {"f_x": 0},
        "h_dict": {
            "E_p": M * G * (p_y + L),
            "E_k": 0.5 * M * (v_x ** 2 + v_y ** 2),
        },
    }

    bld = Builder(sys_dict, target="ctypes")
    bld.build()
    return bld


def simulate():
    """Initialize, run the simulation, and post-process."""
    model = Model("pendulum")

    # Initial condition: solve the algebraic + initialization system.
    # We give a hint for p_x, p_y so the constraint solver lands on the
    # correct branch (negative p_y for a hanging pendulum).
    model.ini(
        {"theta": np.deg2rad(10)},
        xy_0={"p_x": 0.9, "p_y": -5.1, "lam": 0, "f_x": 1},
    )

    # Phase 1: hold the disturbance for 1 s
    model.run(1.0, {})

    # Phase 2: release the force and let the pendulum oscillate for 20 s
    model.run(20.0, {"f_x": 0.0})

    model.post()
    return model


if __name__ == "__main__":
    print("Building pendulum DAE (this compiles C code on first run)...")
    build()

    print("Simulating...")
    model = simulate()

    print("Done. Sample of trajectory:")
    # The exact API for accessing results may depend on the Model class;
    # adjust if your local pydae version exposes a different attribute.
    try:
        t = model.Time
        theta = np.degrees(model.get_values("theta"))
        print(f"  duration   : {t[0]:.2f} s -> {t[-1]:.2f} s ({len(t)} samples)")
        print(f"  theta range: {theta.min():+.2f} deg .. {theta.max():+.2f} deg")
    except AttributeError:
        print("  (Model has finished — inspect attributes interactively)")
