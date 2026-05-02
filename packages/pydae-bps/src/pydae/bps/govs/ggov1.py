# -*- coding: utf-8 -*-
r"""
IEEE / GE GGOV1 general-purpose governor.

Three-state model combining a PI speed governor with droop, a
rate-limited fuel valve actuator, and a lead-lag turbine block.
The model matches the PSS/E GGOV1 definition and is widely used for
gas and diesel turbine prime movers.

**Signal path**

Speed deviation and permanent-droop error:

$$e = \frac{1 - \omega}{R}$$

The governor PI produces the fuel demand correction:

$$\frac{d x_{gov}}{dt} = K_{igov}\, e$$
$$u_{gov} = K_{pgov}\, e + x_{gov}$$

The fuel demand is clamped to the valve position range:

$$u_{fuel} = \mathrm{sat}(p_c + u_{gov},\; V_{min},\; V_{max})$$

The fuel valve actuator is a rate-limited integrator.  Valve rate is
clamped to $[R_{close},\, R_{open}]$:

$$v_{act} = \mathrm{sat}\!\left(\frac{u_{fuel} - x_{fuel}}{T_{act}},\;
            R_{close},\; R_{open}\right)$$
$$\frac{d x_{fuel}}{dt} = v_{act}$$

The turbine is a lead-lag block $(1 + T_c\,s)/(1 + T_b\,s)$ driven
by the valve position $x_{fuel}$.  Lead-lag state $x_{tb}$:

$$\frac{d x_{tb}}{dt} = \frac{x_{fuel} - x_{tb}}{T_b}$$
$$y_{turb} = x_{tb} + \frac{T_c}{T_b}\,(x_{fuel} - x_{tb})$$

Mechanical power with no-load fuel offset and self-damping:

$$0 = K_{turb}\,(y_{turb} - W_{fnl}) - D_m\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$): $e = 0$, $x_{gov} = 0$,
$u_{fuel} = p_c$, $x_{fuel} = p_c$, $x_{tb} = p_c$,
$y_{turb} = p_c$, and

$$p_m = K_{turb}\,(p_c - W_{fnl})$$

The ``p_c`` field therefore represents the **fuel valve setpoint**.
For the test fixture ($K_{turb} = 1$, $W_{fnl} = 0$) this simplifies
to $p_m = p_c$.  For a typical gas turbine ($K_{turb} = 1.5$,
$W_{fnl} = 0.1$) set $p_c = p_m / K_{turb} + W_{fnl}$.

**Note on $T_b$**

$T_b$ must be $> 0$.  Setting $T_c = 0$ gives a pure lag (common for
combustion chambers); setting $T_c = T_b$ gives unity (direct
pass-through).

**Configuration**

Example data entry (typical gas-turbine defaults)::

    "gov": {"type": "ggov1",
            "R": 0.05,
            "K_pgov": 10.0,  "K_igov": 2.0,
            "T_act": 0.5,
            "R_open": 0.1,   "R_close": -0.1,
            "V_max": 1.0,    "V_min": 0.0,
            "K_turb": 1.5,   "W_fnl": 0.1,
            "T_b": 0.5,      "T_c": 0.0,
            "D_m": 0.0,
            "p_c": 0.633}

The ``p_c`` value seeds the valve and turbine states at ini() and sets
the steady-state operating point.
"""

from pydae import ssa


def descriptions():
    """Single source of truth for ggov1 parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "R", "data": "R",
                           "model": "R_gov", "default": 0.05,
                           "description": "Permanent droop (speed regulation)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{pgov}", "data": "K_pgov",
                           "model": "K_pgov_gov", "default": 10.0,
                           "description": "Governor proportional gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{igov}", "data": "K_igov",
                           "model": "K_igov_gov", "default": 2.0,
                           "description": "Governor integral gain",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_{act}", "data": "T_act",
                           "model": "T_act_gov", "default": 0.5,
                           "description": "Fuel valve actuator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "R_{open}", "data": "R_open",
                           "model": "R_open_gov", "default": 0.1,
                           "description": "Maximum valve opening rate",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "R_{close}", "data": "R_close",
                           "model": "R_close_gov", "default": -0.1,
                           "description": "Maximum valve closing rate (negative)",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{max}", "data": "V_max",
                           "model": "V_max_gov", "default": 1.0,
                           "description": "Maximum valve (fuel) position",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{min}", "data": "V_min",
                           "model": "V_min_gov", "default": 0.0,
                           "description": "Minimum valve (fuel) position",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{turb}", "data": "K_turb",
                           "model": "K_turb_gov", "default": 1.5,
                           "description": "Turbine gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "W_{fnl}", "data": "W_fnl",
                           "model": "W_fnl_gov", "default": 0.1,
                           "description": "No-load fuel flow",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_b", "data": "T_b",
                           "model": "T_b_gov", "default": 0.5,
                           "description": "Turbine lag time constant (must be > 0)",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_c", "data": "T_c",
                           "model": "T_c_gov", "default": 0.0,
                           "description": "Turbine lead time constant (0 = pure lag)",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "D_m", "data": "D_m",
                           "model": "D_m_gov", "default": 0.0,
                           "description": "Mechanical damping coefficient",
                           "units": "pu"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "p_c", "data": "p_c",
                           "model": "p_c", "default": 0.633,
                           "description": ("Fuel valve setpoint (load reference). "
                                           "At steady state: p_m = K_turb*(p_c - W_fnl)."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{gov}",
                           "data": "", "model": "x_gov_gov", "default": "",
                           "description": "Governor PI integral state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{fuel}",
                           "data": "", "model": "x_fuel_gov", "default": "",
                           "description": "Fuel valve actuator state (valve position)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{tb}",
                           "data": "", "model": "x_tb_gov", "default": "",
                           "description": "Turbine lead-lag lag state",
                           "units": "pu"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "p_m",
                           "data": "", "model": "p_m", "default": "",
                           "description": "Mechanical power delivered to the synchronous machine",
                           "units": "pu"}]

    return descriptions_list


def ggov1(dae, data, name, _bus_name, backend=None):
    r"""
    Example data entry::

        "gov": {"type": "ggov1",
                "R": 0.05,
                "K_pgov": 10.0, "K_igov": 2.0,
                "T_act": 0.5,
                "R_open": 0.1,  "R_close": -0.1,
                "V_max": 1.0,   "V_min": 0.0,
                "K_turb": 1.5,  "W_fnl": 0.1,
                "T_b": 0.5,     "T_c": 0.0,
                "D_m": 0.0,
                "p_c": 0.633}

    At steady state ($\omega = 1$):
    $p_m = K_{turb}\,(p_c - W_{fnl})$.
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

    # External input (fuel valve / load reference setpoint).
    p_c = backend.symbols(f"p_c_{name}")

    # Dynamic states.
    x_gov  = backend.symbols(f"x_gov_gov_{name}")
    x_fuel = backend.symbols(f"x_fuel_gov_{name}")
    x_tb   = backend.symbols(f"x_tb_gov_{name}")

    # Algebraic state.
    p_m = backend.symbols(f"p_m_{name}")

    # Parameters.
    R       = backend.symbols(f"R_gov_{name}")
    K_pgov  = backend.symbols(f"K_pgov_gov_{name}")
    K_igov  = backend.symbols(f"K_igov_gov_{name}")
    T_act   = backend.symbols(f"T_act_gov_{name}")
    R_open  = backend.symbols(f"R_open_gov_{name}")
    R_close = backend.symbols(f"R_close_gov_{name}")
    V_max   = backend.symbols(f"V_max_gov_{name}")
    V_min   = backend.symbols(f"V_min_gov_{name}")
    K_turb  = backend.symbols(f"K_turb_gov_{name}")
    W_fnl   = backend.symbols(f"W_fnl_gov_{name}")
    T_b     = backend.symbols(f"T_b_gov_{name}")
    T_c     = backend.symbols(f"T_c_gov_{name}")
    D_m     = backend.symbols(f"D_m_gov_{name}")

    # Speed error through permanent droop.
    e_speed = (1 - omega) / R

    # Governor PI: integral state + proportional correction.
    dx_gov = K_igov * e_speed
    u_gov  = K_pgov * e_speed + x_gov

    # Fuel demand: clamped to valve range.
    u_fuel_dem = p_c + u_gov
    u_fuel = backend.Piecewise((V_min, u_fuel_dem < V_min),
                               (V_max, u_fuel_dem > V_max),
                               (u_fuel_dem, True))

    # Fuel valve actuator: rate-limited integrator.
    v_act_nosat = (u_fuel - x_fuel) / T_act
    v_act = backend.Piecewise((R_close, v_act_nosat < R_close),
                              (R_open,  v_act_nosat > R_open),
                              (v_act_nosat, True))
    dx_fuel = v_act

    # Turbine lead-lag.
    dx_tb  = (x_fuel - x_tb) / T_b
    y_turb = x_tb + (T_c / T_b) * (x_fuel - x_tb)

    # Mechanical power: turbine output minus no-load fuel, minus damping.
    p_m_eq = K_turb * (y_turb - W_fnl) - D_m * (omega - 1)
    g_p_m  = p_m_eq - p_m

    # --- ini/run variable partition (no swap: governor does not pin V) ---
    dae['f'] += [dx_gov, dx_fuel, dx_tb]
    dae['x'] += [x_gov,  x_fuel,  x_tb]
    dae['g'] += [g_p_m]
    dae['y_ini'] += [p_m]
    dae['y_run'] += [p_m]

    dae['params_dict'].update({str(R):       gov_data['R']})
    dae['params_dict'].update({str(K_pgov):  gov_data['K_pgov']})
    dae['params_dict'].update({str(K_igov):  gov_data['K_igov']})
    dae['params_dict'].update({str(T_act):   gov_data['T_act']})
    dae['params_dict'].update({str(R_open):  gov_data['R_open']})
    dae['params_dict'].update({str(R_close): gov_data['R_close']})
    dae['params_dict'].update({str(V_max):   gov_data['V_max']})
    dae['params_dict'].update({str(V_min):   gov_data['V_min']})
    dae['params_dict'].update({str(K_turb):  gov_data['K_turb']})
    dae['params_dict'].update({str(W_fnl):   gov_data['W_fnl']})
    dae['params_dict'].update({str(T_b):     gov_data['T_b']})
    dae['params_dict'].update({str(T_c):     gov_data['T_c']})
    dae['params_dict'].update({str(D_m):     gov_data['D_m']})

    p_c_ini = gov_data.get('p_c', 0.5)
    dae['u_ini_dict'].update({str(p_c): p_c_ini})
    dae['u_run_dict'].update({str(p_c): p_c_ini})

    dae['h_dict'].update({str(p_c):              p_c})
    dae['h_dict'].update({f'x_fuel_gov_{name}':  x_fuel})

    # Steady-state seeds (omega=1 → e=0 → x_gov=0; valve and turbine at p_c):
    #   p_m = K_turb * (p_c - W_fnl).
    p_m_ini = gov_data['K_turb'] * (p_c_ini - gov_data['W_fnl'])
    dae['xy_0_dict'].update({str(x_gov):  0.0})
    dae['xy_0_dict'].update({str(x_fuel): p_c_ini})
    dae['xy_0_dict'].update({str(x_tb):   p_c_ini})
    dae['xy_0_dict'].update({str(p_m):    p_m_ini})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('ggov1.hjson')
    grid.uz_jacs = False
    grid.construct('temp_ggov1')

    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_ggov1')

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
