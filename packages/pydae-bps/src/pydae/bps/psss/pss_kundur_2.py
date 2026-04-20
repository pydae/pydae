# -*- coding: utf-8 -*-
r"""
Kundur two-stage power system stabilizer (Figure E12.9).

The canonical textbook PSS from Kundur's *Power System Stability and
Control*, Figure E12.9: speed deviation is passed through a washout,
two series lead-lag compensators, scaled by the stabiliser gain, and
saturated at the output. The two compensators provide the phase lead
needed over the ~0.1–2 Hz range of electromechanical modes.

**Signal path**

The input is the speed deviation from synchronism,

$$\Delta\omega = \omega - \omega_{ref}$$

with $\omega_{ref} = 1$ pu. It passes through a washout
$sT_w/(1 + sT_w)$ with state $x_{wo}$,

$$\frac{d x_{wo}}{dt} = \frac{\Delta\omega - x_{wo}}{T_w},\quad
  z_{wo} = \Delta\omega - x_{wo}$$

followed by two lead-lag compensators in series,

$$\frac{d x_{12}}{dt} = \frac{z_{wo} - x_{12}}{T_2},\quad
  z_{12} = (z_{wo} - x_{12})\frac{T_1}{T_2} + x_{12}$$
$$\frac{d x_{34}}{dt} = \frac{z_{12} - x_{34}}{T_4},\quad
  z_{34} = (z_{12} - x_{34})\frac{T_3}{T_4} + x_{34}$$

The main gain is applied to the final compensator output and the
result is clipped by the hard output limits,

$$0 = \mathrm{sat}(K_{stab}\, z_{34},\; V_{Smin},\; V_{Smax}) - v_{pss}$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**No ini/run swap**

The PSS output is an algebraic variable $v_{pss}$ that feeds the AVR
summing junction. The PSS does not pin a voltage setpoint, so the
``y_ini``/``y_run`` partitions are identical. At steady state
($\omega = 1$) every state settles to zero and the PSS output is
zero — the stabiliser only responds to transients.

**Configuration**

Example data entry::

    "pss": {"type": "pss_kundur_2",
            "K_stab": 20.0,
            "T_w": 10.0,
            "T_1": 0.05, "T_2": 0.02,
            "T_3": 3.0,  "T_4": 5.4,
            "V_Smax": 0.1, "V_Smin": -0.1}

Reference: Kundur, *Power System Stability and Control*, Example 12.6
(Figure E12.9).
"""


import sympy as sym


def descriptions():
    """Single source of truth for pss_kundur_2 parameters, inputs, states, outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_{stab}",
                           "data": "K_stab", "model": "K_stab_pss",
                           "default": 20.0,
                           "description": "Main stabiliser gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_w", "data": "T_w",
                           "model": "T_w_pss", "default": 10.0,
                           "description": "Washout time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_1", "data": "T_1",
                           "model": "T_1_pss", "default": 0.05,
                           "description": "Lead-lag 1 numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_2", "data": "T_2",
                           "model": "T_2_pss", "default": 0.02,
                           "description": "Lead-lag 1 denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_3", "data": "T_3",
                           "model": "T_3_pss", "default": 3.0,
                           "description": "Lead-lag 2 numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_4", "data": "T_4",
                           "model": "T_4_pss", "default": 5.4,
                           "description": "Lead-lag 2 denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{Smax}",
                           "data": "V_Smax", "model": "V_Smax_pss",
                           "default": 0.1,
                           "description": "Upper PSS output limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{Smin}",
                           "data": "V_Smin", "model": "V_Smin_pss",
                           "default": -0.1,
                           "description": "Lower PSS output limit",
                           "units": "pu"}]

    # Inputs (from the rest of the system)
    descriptions_list += [{"type": "Input", "tex": "\\omega",
                           "data": "", "model": "omega", "default": 1.0,
                           "description": ("Machine rotor speed (from the "
                                           "synchronous machine). Used as "
                                           "ω - 1."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{wo}",
                           "data": "", "model": "x_wo_pss", "default": "",
                           "description": "Washout state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{12}",
                           "data": "", "model": "x_12_pss", "default": "",
                           "description": "Lead-lag 1 internal state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{34}",
                           "data": "", "model": "x_34_pss", "default": "",
                           "description": "Lead-lag 2 internal state",
                           "units": "pu"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "v_{pss}",
                           "data": "", "model": "v_pss", "default": "",
                           "description": ("PSS output signal added to the "
                                           "AVR summing junction (saturated)."),
                           "units": "pu"}]

    return descriptions_list


def pss_kundur_2(dae, data, name, bus_name):
    """
    Example data entry::

        "pss": {"type": "pss_kundur_2",
                "K_stab": 20.0,
                "T_w": 10.0,
                "T_1": 0.05, "T_2": 0.02,
                "T_3": 3.0,  "T_4": 5.4,
                "V_Smax": 0.1, "V_Smin": -0.1}

    At steady state every state is zero and $v_{pss} = 0$; the
    stabiliser only produces a signal during speed transients.
    """

    pss_data = data['pss']

    # Inputs from the rest of the system.
    omega = sym.Symbol(f"omega_{name}", real=True)

    # Dynamic states.
    x_wo = sym.Symbol(f"x_wo_pss_{name}", real=True)
    x_12 = sym.Symbol(f"x_12_pss_{name}", real=True)
    x_34 = sym.Symbol(f"x_34_pss_{name}", real=True)

    # Algebraic state.
    v_pss = sym.Symbol(f"v_pss_{name}", real=True)

    # Parameters.
    K_stab = sym.Symbol(f"K_stab_pss_{name}", real=True)
    T_w = sym.Symbol(f"T_w_pss_{name}", real=True)
    T_1 = sym.Symbol(f"T_1_pss_{name}", real=True)
    T_2 = sym.Symbol(f"T_2_pss_{name}", real=True)
    T_3 = sym.Symbol(f"T_3_pss_{name}", real=True)
    T_4 = sym.Symbol(f"T_4_pss_{name}", real=True)
    V_Smax = sym.Symbol(f"V_Smax_pss_{name}", real=True)
    V_Smin = sym.Symbol(f"V_Smin_pss_{name}", real=True)

    # Speed deviation and washout.
    Domega = omega - 1.0
    dx_wo = (Domega - x_wo) / T_w
    z_wo = Domega - x_wo

    # Lead-lag 1.
    dx_12 = (z_wo - x_12) / T_2
    z_12 = (z_wo - x_12) * T_1 / T_2 + x_12

    # Lead-lag 2.
    dx_34 = (z_12 - x_34) / T_4
    z_34 = (z_12 - x_34) * T_3 / T_4 + x_34

    # Gain and output saturation.
    v_pss_nosat = K_stab * z_34
    v_pss_sat = sym.Piecewise((V_Smin, v_pss_nosat < V_Smin),
                              (V_Smax, v_pss_nosat > V_Smax),
                              (v_pss_nosat, True))
    g_v_pss = v_pss_sat - v_pss

    # --- ini/run variable partition (no swap) ---------------------------
    dae['f'] += [dx_wo, dx_12, dx_34]
    dae['x'] += [x_wo, x_12, x_34]
    dae['g'] += [g_v_pss]
    dae['y_ini'] += [v_pss]
    dae['y_run'] += [v_pss]

    dae['params_dict'].update({str(K_stab): pss_data['K_stab']})
    dae['params_dict'].update({str(T_w): pss_data['T_w']})
    dae['params_dict'].update({str(T_1): pss_data['T_1']})
    dae['params_dict'].update({str(T_2): pss_data['T_2']})
    dae['params_dict'].update({str(T_3): pss_data['T_3']})
    dae['params_dict'].update({str(T_4): pss_data['T_4']})
    dae['params_dict'].update({str(V_Smax): pss_data['V_Smax']})
    dae['params_dict'].update({str(V_Smin): pss_data['V_Smin']})

    # Steady-state seeds: all states zero, v_pss = 0.
    for state in [x_wo, x_12, x_34, v_pss]:
        dae['xy_0_dict'].update({str(state): 0.0})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('pss_kundur_2.hjson')
    grid.uz_jacs = False
    grid.construct('temp_pss_kundur_2')

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model('temp_pss_kundur_2')

    v_set = 1.02
    model.ini({'V_1': v_set, 'p_m_1': 0.8}, 'xy_0.json')
    model.report_x()
    model.report_y()
    model.report_u()

    assert model.get_value('V_1') == pytest.approx(v_set, rel=1e-3)
    assert model.get_value('v_pss_1') == pytest.approx(0.0, abs=1e-6)

    model.ini({'V_1': v_set, 'p_m_1': 0.8}, 'xy_0.json')
    model.run(1.0, {})
    model.run(5.0, {'v_ref_1': v_set + 0.05})
    model.post()

    string = f'{model.Time[0]:0.2f}, '
    string += f"{model.get_values('v_pss_1')[0]:0.4f}"
    print(string)

    string = f'{model.Time[-1]:0.2f}, '
    string += f"{model.get_values('v_pss_1')[-1]:0.4f}"
    print(string)


if __name__ == '__main__':
    test()
