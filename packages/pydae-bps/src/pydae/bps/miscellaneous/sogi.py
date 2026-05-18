# -*- coding: utf-8 -*-
r"""
SOGI Phase-Locked Loop / Frequency-Locked Loop (phasor-frame variant).

Implements a Second-Order Generalized Integrator (SOGI) tied to a
Frequency-Locked Loop (FLL) for estimating the phase and frequency of a
bus voltage *directly in the COI rotating-phasor frame*. Acts as a
drop-in alternative to the conventional PI-based
:mod:`pydae.bps.miscellaneous.pll`.

Topology
========

The block carries its own rotating reference :math:`\theta_\text{sogi}`
that is driven by an FLL-adapted frequency :math:`\omega_\text{sogi}`.
The SOGI filters the phase error obtained after a Park transform from
the bus angle :math:`\theta_\text{bus}` to the SOGI frame:

$$v_{\beta,\text{sogi}} =
   V_s \sin(\theta_\text{bus} - \theta_\text{sogi})$$

This signal is the input to the SOGI band-pass with in-phase output
:math:`x_v` and quadrature output :math:`x_{qv}`:

$$\varepsilon = v_{\beta,\text{sogi}} - x_v$$
$$\dot{x}_v   = \omega_n \,(k_\text{sogi}\, \varepsilon - x_{qv})$$
$$\dot{x}_{qv}= \omega_n \, x_v - \alpha_\text{leak}\, x_{qv}$$

The fixed :math:`\omega_n = 2\pi F_n` keeps the band-pass centred at
nominal rather than at the (initially unknown) estimated frequency,
which makes the initialisation well-posed even when the operating-point
slip is zero. The small leak :math:`\alpha_\text{leak}` resolves the
marginal stability of the quadrature integrator at the equilibrium
where the input is DC in the COI frame.

The FLL drives :math:`\omega_\text{sogi}` to zero out the filtered
phase error and the phase integrator closes the loop:

$$\dot{\omega}_\text{sogi} = -\gamma_\text{fll}\, x_v$$
$$\dot{\theta}_\text{sogi} = \Omega_b\,(\omega_\text{sogi} -
                                       \omega_\text{coi})$$

At steady state :math:`\theta_\text{sogi}` tracks
:math:`\theta_\text{bus}`, so :math:`v_{\beta,\text{sogi}} = 0`,
:math:`x_v = x_{qv} = 0` and
:math:`\omega_\text{sogi} = \omega_\text{coi}`.

Configuration
=============

::

    plls: [{
        bus:        "1",
        type:       "sogi",
        k_sogi:     1.414,
        gamma_fll:  50.0,
        alpha_leak: 0.5,
        T_f:        0.02,
        F_n:        50.0
    }]
"""
import numpy as np


def descriptions():
    """Single source of truth for sogi parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "k_{sogi}",
                           "data": "k_sogi", "model": "k_sogi",
                           "default": float(np.sqrt(2)),
                           "description": "SOGI damping", "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\gamma_{fll}",
                           "data": "gamma_fll", "model": "gamma_fll",
                           "default": 50.0,
                           "description": "FLL adaptation gain", "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\alpha_\\text{leak}",
                           "data": "alpha_leak", "model": "alpha_leak_sogi",
                           "default": 0.5,
                           "description": "Quadrature leakage for ini conditioning",
                           "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_f",
                           "data": "T_f", "model": "T_f_sogi", "default": 0.02,
                           "description": "Output filter time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\omega_n",
                           "data": "F_n", "model": "omega_n_sogi",
                           "default": 2 * np.pi * 50.0,
                           "description": "SOGI centre angular frequency",
                           "units": "rad/s"}]

    # Inputs (bus phasor, COI)
    descriptions_list += [{"type": "Input", "tex": "V_s", "data": "",
                           "model": "V_{bus}", "default": 1.0,
                           "description": "Bus voltage magnitude", "units": "pu"}]
    descriptions_list += [{"type": "Input", "tex": "\\theta_s", "data": "",
                           "model": "theta_{bus}", "default": 0.0,
                           "description": "Bus voltage angle (COI frame)",
                           "units": "rad"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "x_v",
                           "data": "", "model": "x_v_sogi", "default": "",
                           "description": "SOGI in-phase output", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{qv}",
                           "data": "", "model": "x_qv_sogi", "default": "",
                           "description": "SOGI quadrature output", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\omega_{sogi}",
                           "data": "", "model": "omega_sogi", "default": "",
                           "description": "FLL-adapted frequency", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\theta_{sogi}",
                           "data": "", "model": "theta_sogi", "default": "",
                           "description": "SOGI-tracked phase (COI frame)",
                           "units": "rad"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\omega_{sogi}^{f}",
                           "data": "", "model": "omega_sogi_f", "default": "",
                           "description": "Smoothed frequency estimate",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\dot\\omega_{sogi}^{f}",
                           "data": "", "model": "rocof_sogi_f", "default": "",
                           "description": "Smoothed RoCoF", "units": "pu/s"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "\\dot\\omega_{sogi}",
                           "data": "", "model": "rocof_sogi", "default": "",
                           "description": "Instantaneous RoCoF", "units": "pu/s"}]

    return descriptions_list


def sogi(dae, data, name, bus_name, backend=None):
    """Attach the phasor-frame SOGI-FLL block to *dae*. sexs-style signature."""
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'use_casadi': False,
            'symbols': lambda _, n, **k: sym.symbols(n, real=True),
            'sin': sym.sin, 'cos': sym.cos,
        })()

    sin = backend.sin
    cos = backend.cos

    # Inputs (bus phasor, COI)
    V_s       = backend.symbols(f"V_{bus_name}")
    theta_s   = backend.symbols(f"theta_{bus_name}")
    omega_coi = backend.symbols("omega_coi")

    # Dynamic states
    (x_v, x_qv, omega_sogi, theta_sogi,
     omega_sogi_f, rocof_sogi_f) = backend.symbols(
        f"x_v_sogi_{name}, x_qv_sogi_{name}, "
        f"omega_sogi_{name}, theta_sogi_{name}, "
        f"omega_sogi_f_{name}, rocof_sogi_f_{name}"
    )

    # Algebraic state
    rocof_sogi = backend.symbols(f"rocof_sogi_{name}")

    # Parameters
    (k_sogi, gamma_fll, alpha_leak,
     T_f, omega_n, Omega_b_sym) = backend.symbols(
        f"k_sogi_{name}, gamma_fll_{name}, alpha_leak_sogi_{name}, "
        f"T_f_sogi_{name}, omega_n_sogi_{name}, Omega_b_sogi_{name}"
    )

    # Park transform: bus angle into SOGI frame
    v_beta_sogi = V_s * sin(theta_s - theta_sogi)

    # SOGI band-pass on the phase error
    eps   = v_beta_sogi - x_v
    dx_v  = omega_n * (k_sogi * eps - x_qv)
    dx_qv = omega_n * x_v - alpha_leak * x_qv

    # FLL adapts omega_sogi to drive x_v -> 0
    domega_sogi = -gamma_fll * x_v

    # Phase integrator (COI frame)
    dtheta_sogi = Omega_b_sym * (omega_sogi - omega_coi)

    # Output smoothing
    domega_sogi_f = (omega_sogi - omega_sogi_f) / T_f
    drocof_sogi_f = (rocof_sogi - rocof_sogi_f) / T_f
    g_rocof = rocof_sogi - domega_sogi

    # Assemble DAE
    dae['f'] += [dx_v, dx_qv, domega_sogi, dtheta_sogi,
                 domega_sogi_f, drocof_sogi_f]
    dae['x'] += [x_v, x_qv, omega_sogi, theta_sogi,
                 omega_sogi_f, rocof_sogi_f]
    dae['g']     += [g_rocof]
    dae['y_ini'] += [rocof_sogi]
    dae['y_run'] += [rocof_sogi]

    F_n_val = data.get('F_n', 50.0)
    dae['params_dict'].update({
        f"k_sogi_{name}":          data.get('k_sogi', float(np.sqrt(2))),
        f"gamma_fll_{name}":       data.get('gamma_fll', 50.0),
        f"alpha_leak_sogi_{name}": data.get('alpha_leak', 0.5),
        f"T_f_sogi_{name}":        data.get('T_f', 0.02),
        f"omega_n_sogi_{name}":    2 * np.pi * F_n_val,
        f"Omega_b_sogi_{name}":    2 * np.pi * F_n_val,
    })

    dae['h_dict'].update({
        f"omega_sogi_{name}":     omega_sogi,
        f"omega_sogi_f_{name}":   omega_sogi_f,
        f"theta_sogi_{name}":     theta_sogi,
        f"rocof_sogi_{name}":     rocof_sogi,
        f"rocof_sogi_f_{name}":   rocof_sogi_f,
        f"frequency_sogi_{name}": F_n_val * omega_sogi_f,
        f"x_v_sogi_{name}":       x_v,
        f"x_qv_sogi_{name}":      x_qv,
    })

    dae['xy_0_dict'].update({
        f"x_v_sogi_{name}":       0.0,
        f"x_qv_sogi_{name}":      0.0,
        f"omega_sogi_{name}":     1.0,
        f"theta_sogi_{name}":     0.0,
        f"omega_sogi_f_{name}":   1.0,
        f"rocof_sogi_f_{name}":   0.0,
        f"rocof_sogi_{name}":     0.0,
    })


def test():
    """In-module test using CasadiBuilder + CasadiModel."""
    import os
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder
    from pydae.core.builder import CasadiModel

    module_dir = os.path.dirname(__file__)
    hjson_path = os.path.join(module_dir, 'sogi.hjson')

    grid = BpsBuilder(hjson_path, use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct('temp_sogi')

    bld = CasadiBuilder(grid.sys_dict)
    bld.build()
    model = CasadiModel(builder=bld)

    model.ini({}, xy_0={})

    omega_sogi_f = float(model.get_value('omega_sogi_f_1'))
    x_v          = float(model.get_value('x_v_sogi_1'))
    x_qv         = float(model.get_value('x_qv_sogi_1'))

    print(f"omega_sogi_f_1 = {omega_sogi_f:.6f} pu")
    print(f"x_v_sogi_1     = {x_v:.6e}")
    print(f"x_qv_sogi_1    = {x_qv:.6e}")

    # At rest the SOGI is locked to the COI frequency.
    assert abs(omega_sogi_f - 1.0) < 1e-4, "SOGI did not lock at omega_coi"
    assert abs(x_v)                < 1e-4, "SOGI in-phase output not at zero"
    assert abs(x_qv)               < 1e-4, "SOGI quadrature output not at zero"


if __name__ == '__main__':
    test()
