# -*- coding: utf-8 -*-
r"""
IEEE HYGOV hydraulic turbine governor.

Four-state model combining a droop governor with a dashpot
(transient-droop) network, a rate- and position-limited gate
servomotor, an inelastic-water-column penstock, and an ideal hydraulic
turbine.

**Signal path**

Speed deviation:

$$\Delta\omega = \omega - 1$$

The dashpot state $c_d$ tracks the gate integrator $x_g$.  Its
time-derivative is the transient droop signal:

$$\frac{d c_d}{dt} = \frac{x_g - c_d}{T_r}, \qquad
  c_{d,out} = R_r \, \frac{x_g - c_d}{T_r}$$

The pilot valve demand sums the load reference, permanent droop, and
transient droop correction:

$$u_{pv} = p_c - \frac{\Delta\omega}{R} - c_{d,out}$$

A first-order filter with time constant $T_f$ smooths the demand:

$$\frac{d c_{pv}}{dt} = \frac{u_{pv} - c_{pv}}{T_f}$$

The gate servomotor is a rate-limited integrator with position limits
$[G_{min}, G_{max}]$.  Let $y_g = \mathrm{sat}(x_g, G_{min}, G_{max})$:

$$v_g = \mathrm{sat}\!\left(\frac{c_{pv} - x_g}{T_g},\;
         -V_{g,max},\; V_{g,max}\right)$$
$$\frac{d x_g}{dt} = v_g + K_{awu}\,(y_g - x_g)$$

The penstock obeys the inelastic water-column equation.  Hydraulic head
$h = (q / y_g)^2$, water starting time $T_w$:

$$\frac{d q}{dt} = \frac{1 - h}{T_w}$$

Mechanical power with turbine self-damping:

$$0 = A_t\,h\,(q - Q_{nl}) - D_{turb}\,\Delta\omega - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$, dashpot fully reset $c_d = x_g$):
$c_{d,out} = 0$, $u_{pv} = p_c$, $c_{pv} = x_g = y_g = p_c$,
$q = p_c$ (from $dq = 0 \Rightarrow h = 1$), and

$$p_m = A_t\,(p_c - Q_{nl})$$

``p_c`` is therefore the **gate position setpoint**.  For the test
fixture ($A_t = 1$, $Q_{nl} = 0$) this simplifies to $p_m = p_c$.

**Configuration**

Example data entry (typical defaults)::

    "gov": {"type": "hygov",
            "R": 0.05,
            "R_r": 0.3,
            "T_r": 5.0,
            "T_f": 0.05,
            "T_g": 0.5,
            "V_g_max": 0.2,
            "G_max": 1.0, "G_min": 0.01,
            "T_w": 1.0,
            "A_t": 1.0,
            "D_turb": 0.0,
            "Q_nl": 0.0,
            "p_c": 0.8}
"""

from pydae import ssa


def descriptions():
    """Single source of truth for hygov parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "R", "data": "R",
                           "model": "R_gov", "default": 0.05,
                           "description": "Permanent droop (speed regulation)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "R_r", "data": "R_r",
                           "model": "R_r_gov", "default": 0.3,
                           "description": "Transient (temporary) droop",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_r", "data": "T_r",
                           "model": "T_r_gov", "default": 5.0,
                           "description": "Dashpot reset (transient droop) time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_f", "data": "T_f",
                           "model": "T_f_gov", "default": 0.05,
                           "description": "Pilot valve filter time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_g", "data": "T_g",
                           "model": "T_g_gov", "default": 0.5,
                           "description": "Gate servomotor time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{g,max}", "data": "V_g_max",
                           "model": "V_g_max_gov", "default": 0.2,
                           "description": "Gate velocity (rate) limit",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "G_{max}", "data": "G_max",
                           "model": "G_max_gov", "default": 1.0,
                           "description": "Maximum gate position",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "G_{min}", "data": "G_min",
                           "model": "G_min_gov", "default": 0.01,
                           "description": "Minimum gate position (keep > 0 to avoid head singularity)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_w", "data": "T_w",
                           "model": "T_w_gov", "default": 1.0,
                           "description": "Water starting time (penstock inertia)",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "A_t", "data": "A_t",
                           "model": "A_t_gov", "default": 1.0,
                           "description": "Turbine gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "D_{turb}", "data": "D_turb",
                           "model": "D_turb_gov", "default": 0.0,
                           "description": "Turbine self-damping coefficient",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "Q_{nl}", "data": "Q_nl",
                           "model": "Q_nl_gov", "default": 0.0,
                           "description": "No-load water flow",
                           "units": "pu"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "p_c", "data": "p_c",
                           "model": "p_c", "default": 0.8,
                           "description": ("Gate position setpoint (load reference). "
                                           "At steady state: p_m = A_t*(p_c - Q_nl)."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "c_{pv}",
                           "data": "", "model": "c_pv_gov", "default": "",
                           "description": "Pilot valve filter output (gate demand)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "c_d",
                           "data": "", "model": "c_d_gov", "default": "",
                           "description": "Dashpot integrator state (tracks gate position)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_g",
                           "data": "", "model": "x_g_gov", "default": "",
                           "description": "Gate servomotor integrator state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "q",
                           "data": "", "model": "q_gov", "default": "",
                           "description": "Normalized water flow (penstock state)",
                           "units": "pu"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "p_m",
                           "data": "", "model": "p_m", "default": "",
                           "description": "Mechanical power delivered to the synchronous machine",
                           "units": "pu"}]

    return descriptions_list


def hygov(dae, data, name, _bus_name, backend=None):
    r"""
    Example data entry::

        "gov": {"type": "hygov",
                "R": 0.05,
                "R_r": 0.3,
                "T_r": 5.0,
                "T_f": 0.05,
                "T_g": 0.5,
                "V_g_max": 0.2,
                "G_max": 1.0, "G_min": 0.01,
                "T_w": 1.0,
                "A_t": 1.0,
                "D_turb": 0.0,
                "Q_nl": 0.0,
                "p_c": 0.8}

    ``p_c`` is the gate position setpoint.  At steady state
    ($\omega = 1$) the mechanical power is $p_m = A_t\,(p_c - Q_{nl})$.
    """
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'Piecewise': sym.Piecewise,
        })()

    gov_data = data['gov']

    # Input from the rest of the system.
    omega = backend.symbols(f"omega_{name}")

    # External input (gate/power setpoint).
    p_c = backend.symbols(f"p_c_{name}")

    # Dynamic states.
    c_pv = backend.symbols(f"c_pv_gov_{name}")
    c_d  = backend.symbols(f"c_d_gov_{name}")
    x_g  = backend.symbols(f"x_g_gov_{name}")
    q    = backend.symbols(f"q_gov_{name}")

    # Algebraic state.
    p_m = backend.symbols(f"p_m_{name}")

    # Parameters.
    R       = backend.symbols(f"R_gov_{name}")
    R_r     = backend.symbols(f"R_r_gov_{name}")
    T_r     = backend.symbols(f"T_r_gov_{name}")
    T_f     = backend.symbols(f"T_f_gov_{name}")
    T_g     = backend.symbols(f"T_g_gov_{name}")
    V_g_max = backend.symbols(f"V_g_max_gov_{name}")
    G_max   = backend.symbols(f"G_max_gov_{name}")
    G_min   = backend.symbols(f"G_min_gov_{name}")
    T_w     = backend.symbols(f"T_w_gov_{name}")
    A_t     = backend.symbols(f"A_t_gov_{name}")
    D_turb  = backend.symbols(f"D_turb_gov_{name}")
    Q_nl    = backend.symbols(f"Q_nl_gov_{name}")
    K_awu   = backend.symbols(f"K_awu_gov_{name}")

    # Speed deviation.
    delta_omega = omega - 1

    # Dashpot: tracks gate integrator, produces transient droop correction.
    dc_d    = (x_g - c_d) / T_r
    c_d_out = R_r * dc_d

    # Pilot valve demand: load reference + permanent droop + transient droop.
    u_pv = p_c - delta_omega / R - c_d_out

    # Pilot valve filter.
    dc_pv = (u_pv - c_pv) / T_f

    # Gate servomotor: rate-limited integrator with position limits.
    y_g_nosat = x_g
    y_g = backend.Piecewise((G_min, y_g_nosat < G_min),
                            (G_max, y_g_nosat > G_max),
                            (y_g_nosat, True))

    v_g_nosat = (c_pv - x_g) / T_g
    v_g = backend.Piecewise((-V_g_max, v_g_nosat < -V_g_max),
                            ( V_g_max, v_g_nosat >  V_g_max),
                            (v_g_nosat, True))

    dx_g = v_g + K_awu * (y_g - y_g_nosat)

    # Penstock: inelastic water column.
    h   = (q / y_g) ** 2
    dq  = (1 - h) / T_w

    # Mechanical power: turbine output minus self-damping.
    p_m_eq = A_t * h * (q - Q_nl) - D_turb * delta_omega
    g_p_m  = p_m_eq - p_m

    # --- ini/run variable partition ---
    dae['f'] += [dc_pv, dc_d, dx_g, dq]
    dae['x'] += [c_pv,  c_d,  x_g,  q]
    dae['g'] += [g_p_m]
    dae['y_ini'] += [p_m]
    dae['y_run'] += [p_m]

    dae['params_dict'].update({str(R):       gov_data['R']})
    dae['params_dict'].update({str(R_r):     gov_data['R_r']})
    dae['params_dict'].update({str(T_r):     gov_data['T_r']})
    dae['params_dict'].update({str(T_f):     gov_data['T_f']})
    dae['params_dict'].update({str(T_g):     gov_data['T_g']})
    dae['params_dict'].update({str(V_g_max): gov_data['V_g_max']})
    dae['params_dict'].update({str(G_max):   gov_data['G_max']})
    dae['params_dict'].update({str(G_min):   gov_data['G_min']})
    dae['params_dict'].update({str(T_w):     gov_data['T_w']})
    dae['params_dict'].update({str(A_t):     gov_data['A_t']})
    dae['params_dict'].update({str(D_turb):  gov_data['D_turb']})
    dae['params_dict'].update({str(Q_nl):    gov_data['Q_nl']})
    dae['params_dict'].update({str(K_awu):   1000.0})

    p_c_ini = gov_data.get('p_c', 0.5)
    dae['u_ini_dict'].update({str(p_c): p_c_ini})
    dae['u_run_dict'].update({str(p_c): p_c_ini})

    dae['h_dict'].update({str(p_c):                  p_c})
    dae['h_dict'].update({f'g_gov_{name}':           y_g})
    dae['h_dict'].update({f'h_gov_{name}':           h})
    dae['h_dict'].update({f'q_gov_{name}':           q})

    # Steady-state seeds:
    #   gate = p_c (dashpot reset → c_d = x_g = p_c);
    #   pilot valve = p_c; water flow = p_c (h=1 at ss);
    #   p_m = A_t * (p_c - Q_nl).
    p_m_ini = gov_data['A_t'] * (p_c_ini - gov_data['Q_nl'])
    dae['xy_0_dict'].update({str(c_pv): p_c_ini})
    dae['xy_0_dict'].update({str(c_d):  p_c_ini})
    dae['xy_0_dict'].update({str(x_g):  p_c_ini})
    dae['xy_0_dict'].update({str(q):    p_c_ini})
    dae['xy_0_dict'].update({str(p_m):  p_m_ini})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('hygov.hjson')
    grid.uz_jacs = False
    grid.construct('temp_hygov')

    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_hygov')

    v_set   = 1.02
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
    assert model.get_value('V_1')   == pytest.approx(v_set,   rel=1e-3)

    model.ini({'V_1': v_set, 'p_c_lc_1': p_c_set}, 'xy_0.json')

    model.A_eval()
    ssa.damp(model.A)

    model.run(1.0,  {})
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
