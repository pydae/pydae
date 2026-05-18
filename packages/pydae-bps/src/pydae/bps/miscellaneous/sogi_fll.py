# -*- coding: utf-8 -*-
r"""
Textbook SOGI-FLL block (canonical EMT formulation).

Implements the literature SOGI-FLL exactly as in Rodriguez et al.
(2006 / 2011) — the input :math:`v_\text{grid}` is treated as an
instantaneous AC voltage sample, *not* a phasor-frame quantity.

DAE
===

$$0 = v_\text{grid} - v_p - \varepsilon_v$$
$$\dot{v}_p   = \hat\omega\,(k_\text{sogi}\,\varepsilon_v - q v_p)$$
$$\dot{q v_p} = \hat\omega\, v_p$$
$$\dot{\hat\omega} = -\Gamma\, \varepsilon_v\, q v_p
                    - \epsilon_\text{reg}\,(\hat\omega - \omega_0)$$

The trailing :math:`\epsilon_\text{reg}` term is a leaky-integrator
regularisation that gives ``omega_est`` a non-zero diagonal in the ini
Jacobian. At the trivial equilibrium ``err_v = qv_p = 0`` the textbook
FLL law has identically zero value and Jacobian, leaving ``omega_est``
unobservable to the Newton initialiser. With ``epsilon_reg > 0`` the
equilibrium uniquely pins ``omega_est = omega_0``; the term has
negligible effect on the transient FLL dynamics for the default
``epsilon_reg = 1e-6``.

Configuration
=============

::

    plls: [{
        name:        "fll1",
        type:        "sogi_fll",
        k_sogi:      1.414,
        Gamma:       50.0,
        omega_0:     314.159265358979,
        epsilon_reg: 1.0e-6,
        v_grid:      0.0
    }]

If a ``bus`` key is given instead of ``name``, the bus name is used as
the namespace identifier; ``v_grid`` is still a free input — it is *not*
bound to the bus voltage.

Phasor-framework caveat
=======================

To exercise the textbook dynamics, drive ``v_grid`` from the outer loop
with a small step ``Dt`` and per-call updates::

    omega_grid = 2 * np.pi * 50.0
    Dt = 1e-4
    for k in range(N):
        t = (k + 1) * Dt
        v = V_amp * np.sin(omega_grid * t)
        model.run(t, {f"v_grid_sogi_fll_{name}": v})

For a phasor-frame estimator that locks the bus phasor directly, use
:mod:`pydae.bps.miscellaneous.sogi` (``type: "sogi"``).
"""
import numpy as np


def descriptions():
    """Single source of truth for sogi_fll parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "k_{sogi}",
                           "data": "k_sogi", "model": "k_sogi_sogi_fll",
                           "default": float(np.sqrt(2)),
                           "description": "SOGI filter gain (sqrt(2) ~ critical)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\Gamma",
                           "data": "Gamma", "model": "Gamma_sogi_fll",
                           "default": 50.0,
                           "description": "FLL adaptation gain", "units": "pu/s/V^2"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\omega_0",
                           "data": "omega_0", "model": "omega_0_sogi_fll",
                           "default": 2 * np.pi * 50.0,
                           "description": "Nominal angular frequency",
                           "units": "rad/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\epsilon_\\text{reg}",
                           "data": "epsilon_reg", "model": "epsilon_reg_sogi_fll",
                           "default": 1e-6,
                           "description": "Leaky-integrator regularisation",
                           "units": "1/s"}]

    # Input
    descriptions_list += [{"type": "Input", "tex": "v_\\text{grid}",
                           "data": "v_grid", "model": "v_grid_sogi_fll",
                           "default": 0.0,
                           "description": "Instantaneous grid voltage sample",
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "v_p",
                           "data": "", "model": "v_p_sogi_fll", "default": "",
                           "description": "In-phase filtered signal", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "q v_p",
                           "data": "", "model": "qv_p_sogi_fll", "default": "",
                           "description": "Quadrature filtered signal",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\hat\\omega",
                           "data": "", "model": "omega_est_sogi_fll", "default": "",
                           "description": "FLL frequency estimate", "units": "rad/s"}]

    # Algebraic
    descriptions_list += [{"type": "Algebraic State", "tex": "\\varepsilon_v",
                           "data": "", "model": "err_v_sogi_fll", "default": "",
                           "description": "Voltage error", "units": "pu"}]

    return descriptions_list


def sogi_fll(dae, data, name, bus_name, backend=None):
    """Attach a textbook SOGI-FLL block to *dae*. sexs-style signature."""
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'use_casadi': False,
            'symbols': lambda _, n, **k: sym.symbols(n, real=True),
        })()

    # Input
    v_grid = backend.symbols(f"v_grid_sogi_fll_{name}")

    # Differential states
    v_p, qv_p, omega_est = backend.symbols(
        f"v_p_sogi_fll_{name}, qv_p_sogi_fll_{name}, omega_est_sogi_fll_{name}"
    )

    # Algebraic
    err_v = backend.symbols(f"err_v_sogi_fll_{name}")

    # Parameters
    k_sogi, Gamma_fll, omega_0, epsilon_reg = backend.symbols(
        f"k_sogi_sogi_fll_{name}, Gamma_sogi_fll_{name}, "
        f"omega_0_sogi_fll_{name}, epsilon_reg_sogi_fll_{name}"
    )

    # DAE
    g_err       = v_grid - v_p - err_v
    # Leaky-integrator regularisation on v_p and qv_p too: their textbook
    # ODEs have zero block-diagonals at the trivial equilibrium and trip
    # the dae ini diagnostic. The leak pins both states uniquely at zero
    # in the rest state and has negligible effect on the transient.
    f_v_p       = omega_est * (k_sogi * err_v - qv_p) - epsilon_reg * v_p
    f_qv_p      = omega_est * v_p - epsilon_reg * qv_p
    f_omega_est = -Gamma_fll * err_v * qv_p - epsilon_reg * (omega_est - omega_0)

    # Assemble DAE
    dae['f']     += [f_v_p, f_qv_p, f_omega_est]
    dae['x']     += [v_p, qv_p, omega_est]
    dae['g']     += [g_err]
    dae['y_ini'] += [err_v]
    dae['y_run'] += [err_v]

    omega_0_val = data.get('omega_0', 2 * np.pi * 50.0)
    dae['params_dict'].update({
        f"k_sogi_sogi_fll_{name}":      data.get('k_sogi', float(np.sqrt(2))),
        f"Gamma_sogi_fll_{name}":       data.get('Gamma', 50.0),
        f"omega_0_sogi_fll_{name}":     omega_0_val,
        f"epsilon_reg_sogi_fll_{name}": data.get('epsilon_reg', 1e-6),
    })

    v_grid_init = data.get('v_grid', 0.0)
    dae['u_ini_dict'][f"v_grid_sogi_fll_{name}"] = v_grid_init
    dae['u_run_dict'][f"v_grid_sogi_fll_{name}"] = v_grid_init

    dae['h_dict'].update({
        f"v_p_sogi_fll_{name}":       v_p,
        f"qv_p_sogi_fll_{name}":      qv_p,
        f"omega_est_sogi_fll_{name}": omega_est,
        f"f_est_sogi_fll_{name}":     omega_est / (2 * np.pi),
        f"err_v_sogi_fll_{name}":     err_v,
    })

    dae['xy_0_dict'].update({
        f"v_p_sogi_fll_{name}":       0.0,
        f"qv_p_sogi_fll_{name}":      0.0,
        f"omega_est_sogi_fll_{name}": omega_0_val,
        f"err_v_sogi_fll_{name}":     0.0,
    })


def test():
    """In-module test using CasadiBuilder + CasadiModel."""
    import os
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder
    from pydae.core.builder import CasadiModel

    module_dir = os.path.dirname(__file__)
    hjson_path = os.path.join(module_dir, 'sogi_fll.hjson')

    grid = BpsBuilder(hjson_path, use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct('temp_sogi_fll')

    bld = CasadiBuilder(grid.sys_dict)
    bld.build()
    model = CasadiModel(builder=bld)

    model.ini({}, xy_0={})

    v_p       = float(model.get_value('v_p_sogi_fll_fll1'))
    qv_p      = float(model.get_value('qv_p_sogi_fll_fll1'))
    omega_est = float(model.get_value('omega_est_sogi_fll_fll1'))
    f_est     = float(model.get_value('f_est_sogi_fll_fll1'))
    err_v     = float(model.get_value('err_v_sogi_fll_fll1'))

    print(f"v_p       = {v_p:.6e}")
    print(f"qv_p      = {qv_p:.6e}")
    print(f"omega_est = {omega_est:.6f} rad/s  (target {2*np.pi*50:.6f})")
    print(f"f_est     = {f_est:.6f} Hz       (target 50.0)")
    print(f"err_v     = {err_v:.6e}")

    # Rest equilibrium with v_grid = 0: trivial state, omega_est = omega_0.
    assert abs(v_p)                       < 1e-6
    assert abs(qv_p)                      < 1e-6
    assert abs(err_v)                     < 1e-6
    assert abs(omega_est - 2 * np.pi * 50) < 1e-3
    assert abs(f_est - 50.0)              < 1e-3


if __name__ == '__main__':
    test()
