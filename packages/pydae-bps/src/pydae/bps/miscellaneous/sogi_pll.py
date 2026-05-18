# -*- coding: utf-8 -*-
r"""
Textbook SOGI-PLL block (canonical EMT formulation).

Implements the literature SOGI band-pass pre-filter combined with a
synchronous-reference-frame PI phase-locked loop — the input
:math:`v_\text{grid}` is treated as an instantaneous AC voltage sample,
*not* a phasor-frame quantity.

DAE
===

$$0 = v_\text{grid} - v_p - \varepsilon_v$$
$$0 = (v_p\cos\hat\theta + qv_p\sin\hat\theta) - v_d$$
$$0 = (-v_p\sin\hat\theta + qv_p\cos\hat\theta) - v_q$$
$$0 = (\omega_0 + k_p\,v_q + x_\text{pi}) - \hat\omega$$
$$\dot{v}_p       = \hat\omega\,(k_\text{sogi}\,\varepsilon_v - qv_p)$$
$$\dot{qv}_p      = \hat\omega\,v_p$$
$$\dot{x}_\text{pi} = k_i\,v_q - \epsilon_\text{reg}\,x_\text{pi}$$
$$\dot{\hat\theta}  = \hat\omega - \omega_\text{ref}
                     - \epsilon_\text{reg}\,\hat\theta$$

Steady-state notes
------------------

The textbook phase integrator :math:`\dot{\hat\theta} = \hat\omega` has no
equilibrium (``theta_est`` grows linearly at the grid rate); we work in
a frame rotating at ``omega_ref`` so a steady state exists. The trailing
:math:`\epsilon_\text{reg}` terms are leaky-integrator regularisations
that resolve the structural zero column in the algebraic Park transform
at ``v_p = qv_p = 0`` (``theta_est`` becomes unobservable) and pin the
PI integrator state at zero. Negligible effect on transient dynamics
for the default ``epsilon_reg = 1e-6``.

Configuration
=============

::

    plls: [{
        name:        "pll1",
        type:        "sogi_pll",
        k_sogi:      1.414,
        k_p:         100.0,
        k_i:         2000.0,
        omega_0:     314.159265358979,
        omega_ref:   314.159265358979,
        epsilon_reg: 1.0e-6,
        v_grid:      0.0
    }]

If a ``bus`` key is given instead of ``name``, the bus name is used as
the namespace identifier; ``v_grid`` is still a free input — it is *not*
bound to the bus voltage.

Phasor-framework caveat
=======================

Same caveat as :mod:`pydae.bps.miscellaneous.sogi_fll`: drive
``v_grid`` from the outer loop with a very small step to exercise the
textbook PLL dynamics. For a phasor-frame estimator that locks to a bus
phasor directly, use :mod:`pydae.bps.miscellaneous.sogi`
(``type: "sogi"``).
"""
import numpy as np


def descriptions():
    """Single source of truth for sogi_pll parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "k_{sogi}",
                           "data": "k_sogi", "model": "k_sogi_sogi_pll",
                           "default": float(np.sqrt(2)),
                           "description": "SOGI filter gain", "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "k_p",
                           "data": "k_p", "model": "k_p_sogi_pll",
                           "default": 100.0,
                           "description": "PLL PI proportional gain", "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "k_i",
                           "data": "k_i", "model": "k_i_sogi_pll",
                           "default": 2000.0,
                           "description": "PLL PI integral gain", "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\omega_0",
                           "data": "omega_0", "model": "omega_0_sogi_pll",
                           "default": 2 * np.pi * 50.0,
                           "description": "Nominal angular frequency",
                           "units": "rad/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\omega_\\text{ref}",
                           "data": "omega_ref", "model": "omega_ref_sogi_pll",
                           "default": 2 * np.pi * 50.0,
                           "description": "Reference-frame angular velocity",
                           "units": "rad/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\epsilon_\\text{reg}",
                           "data": "epsilon_reg", "model": "epsilon_reg_sogi_pll",
                           "default": 1e-6,
                           "description": "Leaky-integrator regularisation",
                           "units": "1/s"}]

    # Input
    descriptions_list += [{"type": "Input", "tex": "v_\\text{grid}",
                           "data": "v_grid", "model": "v_grid_sogi_pll",
                           "default": 0.0,
                           "description": "Instantaneous grid voltage sample",
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "v_p",
                           "data": "", "model": "v_p_sogi_pll", "default": "",
                           "description": "In-phase filtered signal", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "qv_p",
                           "data": "", "model": "qv_p_sogi_pll", "default": "",
                           "description": "Quadrature filtered signal", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_\\text{pi}",
                           "data": "", "model": "x_pi_sogi_pll", "default": "",
                           "description": "PI integrator state", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\hat\\theta",
                           "data": "", "model": "theta_est_sogi_pll", "default": "",
                           "description": "Estimated phase angle (in ref frame)",
                           "units": "rad"}]

    # Algebraic
    descriptions_list += [{"type": "Algebraic State", "tex": "\\varepsilon_v",
                           "data": "", "model": "err_v_sogi_pll", "default": "",
                           "description": "Voltage error", "units": "pu"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "v_d",
                           "data": "", "model": "v_d_sogi_pll", "default": "",
                           "description": "d-axis voltage", "units": "pu"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "v_q",
                           "data": "", "model": "v_q_sogi_pll", "default": "",
                           "description": "q-axis voltage", "units": "pu"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "\\hat\\omega",
                           "data": "", "model": "omega_est_sogi_pll", "default": "",
                           "description": "Instantaneous PLL frequency",
                           "units": "rad/s"}]

    return descriptions_list


def sogi_pll(dae, data, name, bus_name, backend=None):
    """Attach a textbook SOGI-PLL block to *dae*. sexs-style signature."""
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'use_casadi': False,
            'symbols': lambda _, n, **k: sym.symbols(n, real=True),
            'sin': sym.sin, 'cos': sym.cos,
        })()

    sin = backend.sin
    cos = backend.cos

    # Input
    v_grid = backend.symbols(f"v_grid_sogi_pll_{name}")

    # Differential states
    v_p, qv_p, x_pi, theta_est = backend.symbols(
        f"v_p_sogi_pll_{name}, qv_p_sogi_pll_{name}, "
        f"x_pi_sogi_pll_{name}, theta_est_sogi_pll_{name}"
    )

    # Algebraic
    err_v, v_d, v_q, omega_est = backend.symbols(
        f"err_v_sogi_pll_{name}, v_d_sogi_pll_{name}, "
        f"v_q_sogi_pll_{name}, omega_est_sogi_pll_{name}"
    )

    # Parameters
    k_sogi, k_p, k_i, omega_0, omega_ref, epsilon_reg = backend.symbols(
        f"k_sogi_sogi_pll_{name}, k_p_sogi_pll_{name}, "
        f"k_i_sogi_pll_{name}, omega_0_sogi_pll_{name}, "
        f"omega_ref_sogi_pll_{name}, epsilon_reg_sogi_pll_{name}"
    )

    # DAE: algebraic
    g_err   = v_grid - v_p - err_v
    g_v_d   = (v_p * cos(theta_est) + qv_p * sin(theta_est)) - v_d
    g_v_q   = (-v_p * sin(theta_est) + qv_p * cos(theta_est)) - v_q
    g_omega = (omega_0 + k_p * v_q + x_pi) - omega_est

    # DAE: differential. The leaky-integrator regularisation is applied
    # to every state whose textbook ODE has a zero block-diagonal at the
    # rest equilibrium (v_p, qv_p, x_pi, theta_est) — see module
    # docstring for the rationale.
    f_v_p       = omega_est * (k_sogi * err_v - qv_p) - epsilon_reg * v_p
    f_qv_p      = omega_est * v_p - epsilon_reg * qv_p
    f_x_pi      = k_i * v_q - epsilon_reg * x_pi
    f_theta_est = omega_est - omega_ref - epsilon_reg * theta_est

    # Assemble DAE
    dae['f']     += [f_v_p, f_qv_p, f_x_pi, f_theta_est]
    dae['x']     += [v_p, qv_p, x_pi, theta_est]
    dae['g']     += [g_err, g_v_d, g_v_q, g_omega]
    dae['y_ini'] += [err_v, v_d, v_q, omega_est]
    dae['y_run'] += [err_v, v_d, v_q, omega_est]

    omega_0_val   = data.get('omega_0',   2 * np.pi * 50.0)
    omega_ref_val = data.get('omega_ref', omega_0_val)
    dae['params_dict'].update({
        f"k_sogi_sogi_pll_{name}":      data.get('k_sogi', float(np.sqrt(2))),
        f"k_p_sogi_pll_{name}":         data.get('k_p', 100.0),
        f"k_i_sogi_pll_{name}":         data.get('k_i', 2000.0),
        f"omega_0_sogi_pll_{name}":     omega_0_val,
        f"omega_ref_sogi_pll_{name}":   omega_ref_val,
        f"epsilon_reg_sogi_pll_{name}": data.get('epsilon_reg', 1e-6),
    })

    v_grid_init = data.get('v_grid', 0.0)
    dae['u_ini_dict'][f"v_grid_sogi_pll_{name}"] = v_grid_init
    dae['u_run_dict'][f"v_grid_sogi_pll_{name}"] = v_grid_init

    dae['h_dict'].update({
        f"v_p_sogi_pll_{name}":       v_p,
        f"qv_p_sogi_pll_{name}":      qv_p,
        f"v_d_sogi_pll_{name}":       v_d,
        f"v_q_sogi_pll_{name}":       v_q,
        f"theta_est_sogi_pll_{name}": theta_est,
        f"omega_est_sogi_pll_{name}": omega_est,
        f"f_est_sogi_pll_{name}":     omega_est / (2 * np.pi),
        f"x_pi_sogi_pll_{name}":      x_pi,
        f"err_v_sogi_pll_{name}":     err_v,
    })

    dae['xy_0_dict'].update({
        f"v_p_sogi_pll_{name}":       0.0,
        f"qv_p_sogi_pll_{name}":      0.0,
        f"x_pi_sogi_pll_{name}":      0.0,
        f"theta_est_sogi_pll_{name}": 0.0,
        f"err_v_sogi_pll_{name}":     0.0,
        f"v_d_sogi_pll_{name}":       0.0,
        f"v_q_sogi_pll_{name}":       0.0,
        f"omega_est_sogi_pll_{name}": omega_0_val,
    })


def test():
    """In-module test using CasadiBuilder + CasadiModel."""
    import os
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder
    from pydae.core.builder import CasadiModel

    module_dir = os.path.dirname(__file__)
    hjson_path = os.path.join(module_dir, 'sogi_pll.hjson')

    grid = BpsBuilder(hjson_path, use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct('temp_sogi_pll')

    bld = CasadiBuilder(grid.sys_dict)
    bld.build()
    model = CasadiModel(builder=bld)

    model.ini({}, xy_0={})

    v_p       = float(model.get_value('v_p_sogi_pll_pll1'))
    qv_p      = float(model.get_value('qv_p_sogi_pll_pll1'))
    x_pi      = float(model.get_value('x_pi_sogi_pll_pll1'))
    theta_est = float(model.get_value('theta_est_sogi_pll_pll1'))
    omega_est = float(model.get_value('omega_est_sogi_pll_pll1'))
    f_est     = float(model.get_value('f_est_sogi_pll_pll1'))

    print(f"v_p       = {v_p:.6e}")
    print(f"qv_p      = {qv_p:.6e}")
    print(f"x_pi      = {x_pi:.6e}")
    print(f"theta_est = {theta_est:.6e}")
    print(f"omega_est = {omega_est:.6f} rad/s")
    print(f"f_est     = {f_est:.6f} Hz")

    # Rest equilibrium with v_grid = 0: all SOGI/PI states at zero,
    # omega_est = omega_0 (set by the algebraic constraint).
    assert abs(v_p)                       < 1e-6
    assert abs(qv_p)                      < 1e-6
    assert abs(x_pi)                      < 1e-6
    assert abs(theta_est)                 < 1e-6
    assert abs(omega_est - 2 * np.pi * 50) < 1e-3
    assert abs(f_est - 50.0)              < 1e-3


if __name__ == '__main__':
    test()
