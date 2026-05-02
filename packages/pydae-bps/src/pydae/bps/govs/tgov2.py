# -*- coding: utf-8 -*-
r"""
IEEE TGOV2 steam turbine-governor.

Extends TGOV1 by adding explicit rate limits on the valve actuator.
The valve change rate is clamped to $[V_{R,min},\, V_{R,max}]$ before
being integrated, so fast load steps or large droop signals cannot open
or close the valve faster than the physical actuator allows.  All other
signal blocks are identical to TGOV1.

**Signal path**

Speed error feeds the valve reference,

$$u_1 = p_c + \frac{1 - \omega}{R}$$

The unsaturated valve change rate is

$$v_1^{(0)} = \frac{u_1 - x_1}{T_1}$$

After the rate limiter,

$$v_1 = \mathrm{sat}\!\left(v_1^{(0)},\; V_{R,min},\; V_{R,max}\right)$$

The valve state $x_1$ is integrated with anti-windup for the position
limiter $[V_{min}, V_{max}]$.  Let $y_1 = \mathrm{sat}(x_1, V_{min},
V_{max})$:

$$\frac{d x_1}{dt} = v_1 + K_{awu}\,(y_1 - x_1)$$

The turbine lead-lag $(1 + T_2\,s)/(1 + T_3\,s)$ with state $x_2$:

$$\frac{d x_2}{dt} = \frac{y_1 - x_2}{T_3}$$

Mechanical power output with turbine damping:

$$0 = x_2 + \frac{T_2}{T_3}\,(y_1 - x_2) - D_t\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$, no saturation): $u_1 = p_c$,
$v_1^{(0)} = 0$, $x_1 = y_1 = p_c$, $x_2 = p_c$, $p_m = p_c$.

**TGOV1 vs TGOV2**

Set $V_{R,max} = \infty$, $V_{R,min} = -\infty$ (or a sufficiently
large value) to recover TGOV1 behaviour.

**Configuration**

Example data entry (typical defaults)::

    "gov": {"type": "tgov2",
            "R": 0.05,
            "T_1": 0.5,
            "V_max": 1.0,   "V_min": 0.0,
            "V_R_max": 0.2, "V_R_min": -0.2,
            "T_2": 2.1,     "T_3": 7.0,
            "D_t": 0.0,
            "p_c": 0.8}

The ``p_c`` field is the scheduled dispatch — equals the mechanical
power output at steady state when $\omega = 1$.
"""

from pydae import ssa
import sympy as sym


def descriptions():
    """Single source of truth for tgov2 parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "R", "data": "R",
                           "model": "R_gov", "default": 0.05,
                           "description": "Permanent droop (speed regulation)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_1", "data": "T_1",
                           "model": "T_1_gov", "default": 0.5,
                           "description": "Steam chest (valve actuator) time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{max}", "data": "V_max",
                           "model": "V_max_gov", "default": 1.0,
                           "description": "Maximum valve position limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{min}", "data": "V_min",
                           "model": "V_min_gov", "default": 0.0,
                           "description": "Minimum valve position limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{R,max}", "data": "V_R_max",
                           "model": "V_R_max_gov", "default": 0.2,
                           "description": "Maximum valve opening rate limit",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{R,min}", "data": "V_R_min",
                           "model": "V_R_min_gov", "default": -0.2,
                           "description": "Maximum valve closing rate limit (negative)",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_2", "data": "T_2",
                           "model": "T_2_gov", "default": 2.1,
                           "description": "Lead time constant of turbine lead-lag",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_3", "data": "T_3",
                           "model": "T_3_gov", "default": 7.0,
                           "description": "Lag time constant of turbine lead-lag",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "D_t", "data": "D_t",
                           "model": "D_t_gov", "default": 0.0,
                           "description": "Turbine damping coefficient",
                           "units": "pu"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "p_c", "data": "p_c",
                           "model": "p_c", "default": 0.8,
                           "description": ("Scheduled dispatch (load reference). "
                                           "Equals p_m at steady state when omega = 1."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "x_1",
                           "data": "", "model": "x_1_gov", "default": "",
                           "description": "Valve position integrator state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_2",
                           "data": "", "model": "x_2_gov", "default": "",
                           "description": "Lead-lag turbine lag state",
                           "units": "pu"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "p_m",
                           "data": "", "model": "p_m", "default": "",
                           "description": "Mechanical power delivered to the synchronous machine",
                           "units": "pu"}]

    return descriptions_list


def tgov2(dae, data, name, _bus_name, backend=None):
    r"""
    Example data entry::

        "gov": {"type": "tgov2",
                "R": 0.05,
                "T_1": 0.5,
                "V_max": 1.0,   "V_min": 0.0,
                "V_R_max": 0.2, "V_R_min": -0.2,
                "T_2": 2.1,     "T_3": 7.0,
                "D_t": 0.0,
                "p_c": 0.8}

    ``p_c`` is the scheduled dispatch; at steady state ($\omega = 1$)
    the mechanical power output equals ``p_c``.
    """
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'Piecewise': sym.Piecewise,
        })()

    gov_data = data['gov']

    # Inputs from the rest of the system.
    omega = backend.symbols(f"omega_{name}")

    # External input (dispatch setpoint).
    p_c = backend.symbols(f"p_c_{name}")

    # Dynamic states.
    x_1 = backend.symbols(f"x_1_gov_{name}")
    x_2 = backend.symbols(f"x_2_gov_{name}")

    # Algebraic state.
    p_m = backend.symbols(f"p_m_{name}")

    # Parameters.
    R       = backend.symbols(f"R_gov_{name}")
    T_1     = backend.symbols(f"T_1_gov_{name}")
    V_max   = backend.symbols(f"V_max_gov_{name}")
    V_min   = backend.symbols(f"V_min_gov_{name}")
    V_R_max = backend.symbols(f"V_R_max_gov_{name}")
    V_R_min = backend.symbols(f"V_R_min_gov_{name}")
    T_2     = backend.symbols(f"T_2_gov_{name}")
    T_3     = backend.symbols(f"T_3_gov_{name}")
    D_t     = backend.symbols(f"D_t_gov_{name}")
    K_awu   = backend.symbols(f"K_awu_gov_{name}")

    # Valve reference: load reference + speed droop.
    u_1 = p_c + (1.0 / R) * (1 - omega)

    # Rate limiter on valve change.
    v_1_nosat = (u_1 - x_1) / T_1
    v_1 = backend.Piecewise((V_R_min, v_1_nosat < V_R_min),
                            (V_R_max, v_1_nosat > V_R_max),
                            (v_1_nosat, True))

    # Valve position limit with anti-windup.
    y_1_nosat = x_1
    y_1 = backend.Piecewise((V_min, y_1_nosat < V_min),
                            (V_max, y_1_nosat > V_max),
                            (y_1_nosat, True))
    dx_1 = v_1 + K_awu * (y_1 - y_1_nosat)

    # Lead-lag turbine lag state.
    dx_2 = (y_1 - x_2) / T_3

    # Mechanical power: lead-lag output minus turbine damping.
    p_m_eq = x_2 + (T_2 / T_3) * (y_1 - x_2) - D_t * (omega - 1)
    g_p_m = p_m_eq - p_m

    # --- ini/run variable partition (no swap: governor does not pin V) ---
    dae['f'] += [dx_1, dx_2]
    dae['x'] += [x_1, x_2]
    dae['g'] += [g_p_m]
    dae['y_ini'] += [p_m]
    dae['y_run'] += [p_m]

    dae['params_dict'].update({str(R):       gov_data['R']})
    dae['params_dict'].update({str(T_1):     gov_data['T_1']})
    dae['params_dict'].update({str(V_max):   gov_data['V_max']})
    dae['params_dict'].update({str(V_min):   gov_data['V_min']})
    dae['params_dict'].update({str(V_R_max): gov_data['V_R_max']})
    dae['params_dict'].update({str(V_R_min): gov_data['V_R_min']})
    dae['params_dict'].update({str(T_2):     gov_data['T_2']})
    dae['params_dict'].update({str(T_3):     gov_data['T_3']})
    dae['params_dict'].update({str(D_t):     gov_data['D_t']})
    dae['params_dict'].update({str(K_awu):   1000.0})

    p_c_ini = gov_data.get('p_c', 0.5)
    dae['u_ini_dict'].update({str(p_c): p_c_ini})
    dae['u_run_dict'].update({str(p_c): p_c_ini})

    dae['h_dict'].update({str(p_c): p_c})

    # Steady-state seeds: u_1 = p_c → x_1 = p_c; x_2 = p_c; p_m = p_c.
    dae['xy_0_dict'].update({str(x_1): p_c_ini})
    dae['xy_0_dict'].update({str(x_2): p_c_ini})
    dae['xy_0_dict'].update({str(p_m): p_c_ini})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('tgov2.hjson')
    grid.uz_jacs = False
    grid.construct('temp_tgov2')

    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_tgov2')

    v_set = 1.02
    p_c_set = 0.8
    model.ini({'V_1': v_set, 'p_c_lc_1': p_c_set}, 'xy_0.json')

    print('\nx states:\n')
    model.report_x()
    print('\ny states:\n')
    model.report_y()
    print('\nu inputs:\n')
    model.report_u()

    assert model.get_value('p_g_1') == pytest.approx(p_c_set, rel=1e-3)
    assert model.get_value('p_m_1') > p_c_set  # p_m includes armature losses
    assert model.get_value('V_1') == pytest.approx(v_set, rel=1e-3)

    model.ini({'V_1': v_set, 'p_c_lc_1': p_c_set}, 'xy_0.json')

    model.A_eval()
    ssa.damp(model.A)

    model.run(1.0, {})
    model.run(60.0, {'p_c_lc_1': 0.6})
    model.post()

    string = f'{model.Time[0]:0.2f}, '
    string += f"{model.get_values('p_g_1')[0]:0.3f}"
    print(string)

    string = f'{model.Time[-1]:0.2f}, '
    string += f"{model.get_values('p_g_1')[-1]:0.3f}"
    print(string)

    assert model.get_values('p_g_1')[-1] == pytest.approx(0.6, rel=2e-2)


if __name__ == '__main__':
    test()
