# -*- coding: utf-8 -*-
r"""
SRF Phase-Locked Loop (PI-PLL) bus-frequency estimator.

Estimates the phase and frequency of a bus voltage using a classical
synchronous-reference-frame phase-locked loop. The bus voltage phasor
:math:`V \angle \theta_s` is projected onto the PLL rotating reference
to yield a d-axis voltage that vanishes at lock. A PI regulator on the
d-axis voltage produces an angular-frequency correction, which is
integrated to obtain the phase estimate.

Signal path
-----------

Define the d-axis voltage in the PLL frame:

$$v_{sD} = V \sin\theta_s, \quad v_{sQ} = V \cos\theta_s$$
$$v_{sd}^{\mathrm{pll}} = v_{sD}\cos\theta_{\mathrm{pll}} -
                          v_{sQ}\sin\theta_{\mathrm{pll}}$$

The PI loop:

$$\omega_{\mathrm{pll}} = 1 + K_{p}\, v_{sd}^{\mathrm{pll}}
                            + K_{i}\, \xi_{\mathrm{pll}}$$
$$\dot{\xi}_{\mathrm{pll}} = v_{sd}^{\mathrm{pll}}$$

Phase integration in the COI rotating frame:

$$\dot{\theta}_{\mathrm{pll}} = 2\pi f_n \,
   K_{\theta}\,(\omega_{\mathrm{pll}} - \omega_{\mathrm{coi}})$$

Output smoothing and rate-of-change of frequency:

$$\dot{\omega}_{\mathrm{pll}}^{f} = (\omega_{\mathrm{pll}} -
                                    \omega_{\mathrm{pll}}^{f}) / T$$
$$\rho_f = \dot{\omega}_{\mathrm{pll}}^{f}$$
$$\dot{\rho}_f^{\mathrm{filt}} = (\rho_f - \rho_f^{\mathrm{filt}}) / T$$

At lock :math:`\omega_{\mathrm{pll}} = \omega_{\mathrm{coi}}` and
:math:`\theta_{\mathrm{pll}}` is constant in the COI frame.

Configuration
-------------

Attach to a bus via the top-level ``plls`` block::

    plls: [{
        bus:        "1",
        K_p_pll:    180.0,
        K_i_pll:    3200.0,
        T_pll:      0.02,
        K_theta_pll: 1.0
    }]
"""
import numpy as np


def descriptions():
    """Single source of truth for pll parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_{p,pll}", "data": "K_p_pll",
                           "model": "K_p_pll", "default": 180.0,
                           "description": "PLL proportional gain", "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{i,pll}", "data": "K_i_pll",
                           "model": "K_i_pll", "default": 3200.0,
                           "description": "PLL integral gain", "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_{pll}", "data": "T_pll",
                           "model": "T_pll", "default": 0.02,
                           "description": "Output filter time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{\\theta,pll}",
                           "data": "K_theta_pll", "model": "K_theta_pll", "default": 1.0,
                           "description": "Phase-integrator scaling (1.0 nominal)",
                           "units": "pu"}]

    # Inputs (bus phasor, system COI)
    descriptions_list += [{"type": "Input", "tex": "V_s", "data": "", "model": "V_{bus}",
                           "default": 1.0, "description": "Bus voltage magnitude",
                           "units": "pu"}]
    descriptions_list += [{"type": "Input", "tex": "\\theta_s", "data": "",
                           "model": "theta_{bus}", "default": 0.0,
                           "description": "Bus voltage angle (COI frame)",
                           "units": "rad"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "\\theta_{pll}",
                           "data": "", "model": "theta_pll", "default": "",
                           "description": "PLL phase estimate (COI frame)",
                           "units": "rad"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\xi_{pll}",
                           "data": "", "model": "xi_pll", "default": "",
                           "description": "PI integrator state", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\omega_{pll}^{f}",
                           "data": "", "model": "omega_pll_f", "default": "",
                           "description": "Smoothed angular-frequency estimate",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\dot\\omega_{pll}^{f}",
                           "data": "", "model": "rocof_pll_f", "default": "",
                           "description": "Smoothed rate-of-change of frequency",
                           "units": "pu/s"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "\\dot\\omega_{pll}",
                           "data": "", "model": "rocof_pll", "default": "",
                           "description": "Instantaneous RoCoF", "units": "pu/s"}]

    return descriptions_list


def pll(dae, data, name, bus_name, backend=None):
    """Attach a PI-PLL block to *dae*. sexs-style signature."""
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
    theta_pll, xi_pll, omega_pll_f, rocof_pll_f = backend.symbols(
        f"theta_pll_{name}, xi_pll_{name}, "
        f"omega_pll_f_{name}, rocof_pll_f_{name}"
    )

    # Algebraic state
    rocof_pll = backend.symbols(f"rocof_pll_{name}")

    # Parameters
    K_p_pll, K_i_pll, T_pll, K_theta_pll = backend.symbols(
        f"K_p_pll_{name}, K_i_pll_{name}, T_pll_{name}, K_theta_pll_{name}"
    )

    # Park projection onto PLL frame
    v_sD = V_s * sin(theta_s)
    v_sQ = V_s * cos(theta_s)
    v_sd_pll = v_sD * cos(theta_pll) - v_sQ * sin(theta_pll)

    # PI loop
    Domega_pll = K_p_pll * v_sd_pll + K_i_pll * xi_pll
    omega_pll  = Domega_pll + 1.0

    F_n = data.get('F_n', 50.0)
    Omega_b_val = 2 * np.pi * F_n

    dtheta_pll   = Omega_b_val * (omega_pll - omega_coi) * K_theta_pll
    dxi_pll      = v_sd_pll
    domega_pll_f = (omega_pll - omega_pll_f) / T_pll
    drocof_pll_f = (rocof_pll - rocof_pll_f) / T_pll

    g_rocof = rocof_pll - domega_pll_f

    # Assemble DAE
    dae['f']     += [dtheta_pll, dxi_pll, domega_pll_f, drocof_pll_f]
    dae['x']     += [theta_pll, xi_pll, omega_pll_f, rocof_pll_f]
    dae['g']     += [g_rocof]
    dae['y_ini'] += [rocof_pll]
    dae['y_run'] += [rocof_pll]

    dae['params_dict'].update({
        f"K_p_pll_{name}":     data.get('K_p_pll', 180.0),
        f"K_i_pll_{name}":     data.get('K_i_pll', 3200.0),
        f"T_pll_{name}":       data.get('T_pll', 0.02),
        f"K_theta_pll_{name}": data.get('K_theta_pll', 1.0),
    })

    dae['h_dict'].update({
        f"omega_pll_{name}":   omega_pll,
        f"omega_pll_f_{name}": omega_pll_f,
        f"rocof_pll_{name}":   rocof_pll,
        f"frequency_pll_{name}": F_n * omega_pll_f,
        f"theta_pll_{name}":   theta_pll,
    })

    dae['xy_0_dict'].update({
        f"theta_pll_{name}":   0.0,
        f"xi_pll_{name}":      0.0,
        f"omega_pll_f_{name}": 1.0,
        f"rocof_pll_f_{name}": 0.0,
        f"rocof_pll_{name}":   0.0,
    })


def add_pll(grid, data):
    """Legacy ``(grid, data)`` wrapper kept for backward-compat with
    callers that still use the old API (e.g. ``vscs.vscs.add_vscs``).
    """
    bus_name = data.get('bus')
    name = data.get('name', bus_name)
    pll(grid.dae, data, name, bus_name, grid.backend)


def test():
    """In-module test using CasadiBuilder + CasadiModel."""
    import os
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder
    from pydae.core.builder import CasadiModel

    module_dir = os.path.dirname(__file__)
    hjson_path = os.path.join(module_dir, 'pll.hjson')

    grid = BpsBuilder(hjson_path, use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct('temp_pll')

    bld = CasadiBuilder(grid.sys_dict)
    bld.build()
    model = CasadiModel(builder=bld)

    model.ini({}, xy_0={})

    omega_pll_f = float(model.get_value('omega_pll_f_1'))
    theta_pll   = float(model.get_value('theta_pll_1'))
    rocof_pll_f = float(model.get_value('rocof_pll_f_1'))

    print(f"omega_pll_f_1 = {omega_pll_f:.6f} pu")
    print(f"theta_pll_1   = {theta_pll:.6f} rad")
    print(f"rocof_pll_f_1 = {rocof_pll_f:.6e} pu/s")

    # At rest the PLL is locked to the COI frequency.
    assert abs(omega_pll_f - 1.0)  < 1e-6, "PLL did not lock at omega_coi"
    assert abs(rocof_pll_f)        < 1e-6, "RoCoF not zero at steady state"


if __name__ == '__main__':
    test()
