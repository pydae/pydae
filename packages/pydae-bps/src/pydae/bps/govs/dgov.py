# -*- coding: utf-8 -*-
r"""
DGOV diesel engine governor.

Three-state proportional governor for diesel prime movers.  Unlike
GGOV1 (which has PI integral action), DGOV uses a purely proportional
speed path filtered through a single governor lag, followed by a
rate-limited fuel actuator and a combustion delay.  This matches the
fast-response, droop-governed behaviour typical of diesel generator
sets.

**Signal path**

Speed error through permanent droop and governor gain:

$$e = \frac{K\,(1 - \omega)}{R}$$

The governor lag $T_1$ smooths the speed signal:

$$\frac{d x_{gov}}{dt} = \frac{e - x_{gov}}{T_1}$$

The fuel demand sums the load reference and the governor output,
clamped to the valve range $[V_{min}, V_{max}]$:

$$u_{fuel} = \mathrm{sat}(p_c + x_{gov},\; V_{min},\; V_{max})$$

The actuator is a rate-limited integrator.  Valve rate is clamped to
$[R_{close},\, R_{open}]$:

$$v_{act} = \mathrm{sat}\!\left(\frac{u_{fuel} - x_{act}}{T_2},\;
            R_{close},\; R_{open}\right)$$
$$\frac{d x_{act}}{dt} = v_{act}$$

The combustion/engine lag $T_3$ filters the fuel flow into shaft
torque:

$$\frac{d x_{eng}}{dt} = \frac{x_{act} - x_{eng}}{T_3}$$

Mechanical power with no-load fuel and self-damping:

$$0 = K_{turb}\,(x_{eng} - W_{fnl}) - D_m\,(\omega - 1) - p_m$$

**Steady-state relation**

At synchronism ($\omega = 1$): $e = 0$, $x_{gov} = 0$,
$u_{fuel} = p_c$, $x_{act} = p_c$, $x_{eng} = p_c$, and

$$p_m = K_{turb}\,(p_c - W_{fnl})$$

The test fixture uses $K_{turb} = 1$, $W_{fnl} = 0$ so $p_m = p_c$.

**DGOV vs GGOV1**

DGOV has no governor integral state: speed error correction vanishes
at $\omega = 1$ regardless of load, so non-zero steady-state speed
deviation is possible under load change.  GGOV1's integral forces
$\omega \to 1$ asymptotically.  DGOV is therefore appropriate when the
droop characteristic governs frequency in island or parallel operation
without isochronous correction.

**Configuration**

Example data entry (typical diesel-genset defaults)::

    "gov": {"type": "dgov",
            "R": 0.05,
            "K": 1.0,
            "T_1": 0.02,
            "T_2": 0.3,
            "R_open": 0.3,  "R_close": -0.3,
            "V_max": 1.0,   "V_min": 0.0,
            "T_3": 0.5,
            "K_turb": 1.0,  "W_fnl": 0.0,
            "D_m": 0.0,
            "p_c": 0.8}
"""

from pydae import ssa


def descriptions():
    """Single source of truth for dgov parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "R", "data": "R",
                           "model": "R_gov", "default": 0.05,
                           "description": "Permanent droop (speed regulation)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K", "data": "K",
                           "model": "K_gov", "default": 1.0,
                           "description": "Governor proportional gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_1", "data": "T_1",
                           "model": "T_1_gov", "default": 0.02,
                           "description": "Governor lag time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_2", "data": "T_2",
                           "model": "T_2_gov", "default": 0.3,
                           "description": "Fuel valve actuator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "R_{open}", "data": "R_open",
                           "model": "R_open_gov", "default": 0.3,
                           "description": "Maximum valve opening rate",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "R_{close}", "data": "R_close",
                           "model": "R_close_gov", "default": -0.3,
                           "description": "Maximum valve closing rate (negative)",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{max}", "data": "V_max",
                           "model": "V_max_gov", "default": 1.0,
                           "description": "Maximum fuel valve position",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{min}", "data": "V_min",
                           "model": "V_min_gov", "default": 0.0,
                           "description": "Minimum fuel valve position",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_3", "data": "T_3",
                           "model": "T_3_gov", "default": 0.5,
                           "description": "Engine combustion lag time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{turb}", "data": "K_turb",
                           "model": "K_turb_gov", "default": 1.0,
                           "description": "Engine/turbine gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "W_{fnl}", "data": "W_fnl",
                           "model": "W_fnl_gov", "default": 0.0,
                           "description": "No-load fuel flow",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "D_m", "data": "D_m",
                           "model": "D_m_gov", "default": 0.0,
                           "description": "Mechanical damping coefficient",
                           "units": "pu"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "p_c", "data": "p_c",
                           "model": "p_c", "default": 0.8,
                           "description": ("Fuel valve setpoint (load reference). "
                                           "At steady state: p_m = K_turb*(p_c - W_fnl)."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{gov}",
                           "data": "", "model": "x_gov_gov", "default": "",
                           "description": "Governor proportional lag state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{act}",
                           "data": "", "model": "x_act_gov", "default": "",
                           "description": "Fuel valve actuator state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{eng}",
                           "data": "", "model": "x_eng_gov", "default": "",
                           "description": "Engine combustion lag state",
                           "units": "pu"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "p_m",
                           "data": "", "model": "p_m", "default": "",
                           "description": "Mechanical power delivered to the synchronous machine",
                           "units": "pu"}]

    return descriptions_list


def dgov(dae, data, name, _bus_name, backend=None):
    r"""
    Example data entry::

        "gov": {"type": "dgov",
                "R": 0.05,
                "K": 1.0,
                "T_1": 0.02,
                "T_2": 0.3,
                "R_open": 0.3,  "R_close": -0.3,
                "V_max": 1.0,   "V_min": 0.0,
                "T_3": 0.5,
                "K_turb": 1.0,  "W_fnl": 0.0,
                "D_m": 0.0,
                "p_c": 0.8}

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
    x_gov = backend.symbols(f"x_gov_gov_{name}")
    x_act = backend.symbols(f"x_act_gov_{name}")
    x_eng = backend.symbols(f"x_eng_gov_{name}")

    # Algebraic state.
    p_m = backend.symbols(f"p_m_{name}")

    # Parameters.
    R       = backend.symbols(f"R_gov_{name}")
    K       = backend.symbols(f"K_gov_{name}")
    T_1     = backend.symbols(f"T_1_gov_{name}")
    T_2     = backend.symbols(f"T_2_gov_{name}")
    R_open  = backend.symbols(f"R_open_gov_{name}")
    R_close = backend.symbols(f"R_close_gov_{name}")
    V_max   = backend.symbols(f"V_max_gov_{name}")
    V_min   = backend.symbols(f"V_min_gov_{name}")
    T_3     = backend.symbols(f"T_3_gov_{name}")
    K_turb  = backend.symbols(f"K_turb_gov_{name}")
    W_fnl   = backend.symbols(f"W_fnl_gov_{name}")
    D_m     = backend.symbols(f"D_m_gov_{name}")

    # Speed error: proportional droop + governor gain.
    e_speed = K * (1 - omega) / R

    # Governor proportional lag.
    dx_gov = (e_speed - x_gov) / T_1

    # Fuel demand: governor correction + load reference, position-clamped.
    u_fuel_dem = p_c + x_gov
    u_fuel = backend.Piecewise((V_min, u_fuel_dem < V_min),
                               (V_max, u_fuel_dem > V_max),
                               (u_fuel_dem, True))

    # Fuel valve actuator: rate-limited integrator.
    v_act_nosat = (u_fuel - x_act) / T_2
    v_act = backend.Piecewise((R_close, v_act_nosat < R_close),
                              (R_open,  v_act_nosat > R_open),
                              (v_act_nosat, True))
    dx_act = v_act

    # Engine combustion lag.
    dx_eng = (x_act - x_eng) / T_3

    # Mechanical power: engine output minus no-load fuel, minus damping.
    p_m_eq = K_turb * (x_eng - W_fnl) - D_m * (omega - 1)
    g_p_m  = p_m_eq - p_m

    # --- ini/run variable partition (no swap: governor does not pin V) ---
    dae['f'] += [dx_gov, dx_act, dx_eng]
    dae['x'] += [x_gov,  x_act,  x_eng]
    dae['g'] += [g_p_m]
    dae['y_ini'] += [p_m]
    dae['y_run'] += [p_m]

    dae['params_dict'].update({str(R):       gov_data['R']})
    dae['params_dict'].update({str(K):       gov_data['K']})
    dae['params_dict'].update({str(T_1):     gov_data['T_1']})
    dae['params_dict'].update({str(T_2):     gov_data['T_2']})
    dae['params_dict'].update({str(R_open):  gov_data['R_open']})
    dae['params_dict'].update({str(R_close): gov_data['R_close']})
    dae['params_dict'].update({str(V_max):   gov_data['V_max']})
    dae['params_dict'].update({str(V_min):   gov_data['V_min']})
    dae['params_dict'].update({str(T_3):     gov_data['T_3']})
    dae['params_dict'].update({str(K_turb):  gov_data['K_turb']})
    dae['params_dict'].update({str(W_fnl):   gov_data['W_fnl']})
    dae['params_dict'].update({str(D_m):     gov_data['D_m']})

    p_c_ini = gov_data.get('p_c', 0.5)
    dae['u_ini_dict'].update({str(p_c): p_c_ini})
    dae['u_run_dict'].update({str(p_c): p_c_ini})

    dae['h_dict'].update({str(p_c):             p_c})
    dae['h_dict'].update({f'x_act_gov_{name}':  x_act})

    # Steady-state seeds (omega=1 → e=0 → x_gov=0; actuator and engine at p_c):
    #   p_m = K_turb * (p_c - W_fnl).
    p_m_ini = gov_data['K_turb'] * (p_c_ini - gov_data['W_fnl'])
    dae['xy_0_dict'].update({str(x_gov): 0.0})
    dae['xy_0_dict'].update({str(x_act): p_c_ini})
    dae['xy_0_dict'].update({str(x_eng): p_c_ini})
    dae['xy_0_dict'].update({str(p_m):   p_m_ini})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('dgov.hjson')
    grid.uz_jacs = False
    grid.construct('temp_dgov')

    bld = Builder(grid.sys_dict, target='cffi', sparse=False)
    bld.build()

    model = Model('temp_dgov')

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
