# -*- coding: utf-8 -*-
r"""
IEEE Type 1 (IEEEG1) steam turbine-governor (REE NTS parameter set).

The IEEEG1 model combines a speed-droop governor with a servo-actuated
main valve and a four-lag steam turbine whose output is tapped at four
points by coefficients $K_1 \dots K_8$. The HP fractions $K_1, K_3,
K_5, K_7$ drive the HP shaft and the LP fractions $K_2, K_4, K_6, K_8$
drive the LP shaft. For a single-mass generator (as in REE NTS) the
HP and LP taps are summed into a single mechanical power
$p_m = p_{m,HP} + p_{m,LP}$ that feeds the synchronous machine swing
equation.

**Signal path**

The speed deviation with respect to synchronism is

$$\Delta\omega = 1 - \omega$$

The summing junction combines droop, scheduled dispatch $p_c$, valve
feedback and AGC trim,

$$u_3 = K\,\Delta\omega + p_c - y_g + K_{sec}\, p_{agc}$$

The servo valve is a rate- and position-limited integrator with state
$x_3$,

$$u_g = \mathrm{sat}\!\left(u_3 / T_3,\; U_c,\; U_o\right)$$
$$\frac{d x_3}{dt} = u_g + K_{awu}\,(y_g - x_3)$$

where the anti-windup term drives the integrator state back into the
admissible range when the position limiter saturates,

$$y_g = \mathrm{sat}(x_3,\; P_{min},\; P_{max})$$

With the REE parameter set $T_1 = T_2 = 0$ so the optional lead-lag
block between the servo and the steam chest is a unity pass-through
and is not instantiated.

The steam turbine is a cascade of three first-order lags (the fourth
collapses because $T_7 = 0$ under REE; $K_7$ and $K_8$ coefficients,
if nonzero, are applied directly to $x_6$),

$$\frac{d x_4}{dt} = \frac{y_g - x_4}{T_4}$$
$$\frac{d x_5}{dt} = \frac{x_4 - x_5}{T_5}$$
$$\frac{d x_6}{dt} = \frac{x_5 - x_6}{T_6}$$

The mechanical output is the sum of HP and LP fractions,

$$p_{m,HP} = K_1 x_4 + K_3 x_5 + K_5 x_6 + K_7 x_6$$
$$p_{m,LP} = K_2 x_4 + K_4 x_5 + K_6 x_6 + K_8 x_6$$
$$0 = p_{m,HP} + p_{m,LP} - p_m$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**Steady-state relation**

At synchronism ($\omega = 1$, $\Delta\omega = 0$, $p_{agc} = 0$) the
servo integrator requires $u_g = 0 \Rightarrow u_3 = 0$, hence
$y_g = p_c$. The turbine cascade then gives
$x_4 = x_5 = x_6 = p_c$, and the mechanical output is
$p_m = (\sum K_i)\, p_c$ which equals $p_c$ when the HP+LP fractions
sum to unity — as in the REE parameter set
($K_1 + K_3 + K_5 = 1.0$, LP fractions zero).

**No ini/run swap**

Governors regulate mechanical power through speed feedback; they do
not pin a voltage setpoint, so the ``y_ini``/``y_run`` partitions are
identical. ``p_c`` is an input in both phases; ``p_m`` is solved
algebraically in both phases.

**Configuration**

Example data entry (REE NTS defaults)::

    "gov": {"type": "ieeeg1",
            "K": 20.0,
            "K_1": 0.3, "K_3": 0.3, "K_5": 0.4, "K_7": 0.0,
            "K_2": 0.0, "K_4": 0.0, "K_6": 0.0, "K_8": 0.0,
            "T_1": 0.0, "T_2": 0.0, "T_3": 0.1,
            "T_4": 0.3, "T_5": 7.0, "T_6": 0.6, "T_7": 0.0,
            "U_o": 0.5, "U_c": -0.5,
            "P_max": 1.0, "P_min": 0.0,
            "p_c": 0.8}

The ``p_c`` field is the scheduled dispatch — the steady-state
mechanical output when $\omega = 1$ and $p_{agc} = 0$. It is used as
both the ``u_ini`` and ``u_run`` value, and seeds the turbine-cascade
states.
"""


from pydae import ssa


def descriptions():
    """Single source of truth for ieeeg1 parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K", "data": "K",
                           "model": "K_gov", "default": 20.0,
                           "description": ("Speed-droop gain (1/R). REE NTS "
                                           "default corresponds to 5% droop."),
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_1", "data": "K_1",
                           "model": "K_1_gov", "default": 0.3,
                           "description": "HP fraction at steam-chest tap (after T_4)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_3", "data": "K_3",
                           "model": "K_3_gov", "default": 0.3,
                           "description": "HP fraction at reheater tap (after T_5)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_5", "data": "K_5",
                           "model": "K_5_gov", "default": 0.4,
                           "description": "HP fraction at crossover tap (after T_6)",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_7", "data": "K_7",
                           "model": "K_7_gov", "default": 0.0,
                           "description": ("HP fraction at LP-turbine tap "
                                           "(after T_7). Applied directly to "
                                           "x_6 when T_7 = 0."),
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_2", "data": "K_2",
                           "model": "K_2_gov", "default": 0.0,
                           "description": "LP fraction at steam-chest tap",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_4", "data": "K_4",
                           "model": "K_4_gov", "default": 0.0,
                           "description": "LP fraction at reheater tap",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_6", "data": "K_6",
                           "model": "K_6_gov", "default": 0.0,
                           "description": "LP fraction at crossover tap",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_8", "data": "K_8",
                           "model": "K_8_gov", "default": 0.0,
                           "description": "LP fraction at LP-turbine tap",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_1", "data": "T_1",
                           "model": "T_1_gov", "default": 0.0,
                           "description": ("Lead-lag numerator. Zero under REE "
                                           "so the lead-lag is unity and is "
                                           "not instantiated."),
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_2", "data": "T_2",
                           "model": "T_2_gov", "default": 0.0,
                           "description": ("Lead-lag denominator. Zero under "
                                           "REE — lead-lag not instantiated."),
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_3", "data": "T_3",
                           "model": "T_3_gov", "default": 0.1,
                           "description": "Servo (valve actuator) time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_4", "data": "T_4",
                           "model": "T_4_gov", "default": 0.3,
                           "description": "Steam-chest time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_5", "data": "T_5",
                           "model": "T_5_gov", "default": 7.0,
                           "description": "Reheater time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_6", "data": "T_6",
                           "model": "T_6_gov", "default": 0.6,
                           "description": "Crossover time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_7", "data": "T_7",
                           "model": "T_7_gov", "default": 0.0,
                           "description": ("LP-turbine time constant. Zero "
                                           "under REE — x_7 collapses into "
                                           "x_6."),
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "U_o", "data": "U_o",
                           "model": "U_o_gov", "default": 0.5,
                           "description": "Servo opening rate limit",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "U_c", "data": "U_c",
                           "model": "U_c_gov", "default": -0.5,
                           "description": "Servo closing rate limit",
                           "units": "pu/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "P_{max}", "data": "P_max",
                           "model": "P_max_gov", "default": 1.0,
                           "description": "Upper valve position limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "P_{min}", "data": "P_min",
                           "model": "P_min_gov", "default": 0.0,
                           "description": "Lower valve position limit",
                           "units": "pu"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "p_c", "data": "p_c",
                           "model": "p_c", "default": 0.8,
                           "description": ("Scheduled dispatch (load reference). "
                                           "Equals p_m at steady state when "
                                           "omega = 1 and p_agc = 0."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "x_3",
                           "data": "", "model": "x_3_gov", "default": "",
                           "description": "Servo (valve actuator) integrator state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_4",
                           "data": "", "model": "x_4_gov", "default": "",
                           "description": "Steam-chest lag state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_5",
                           "data": "", "model": "x_5_gov", "default": "",
                           "description": "Reheater lag state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_6",
                           "data": "", "model": "x_6_gov", "default": "",
                           "description": "Crossover lag state",
                           "units": "pu"}]

    # Algebraic state — solved in both ini and run phases.
    descriptions_list += [{"type": "Algebraic State", "tex": "p_m",
                           "data": "", "model": "p_m", "default": "",
                           "description": ("Mechanical power delivered to the "
                                           "synchronous machine swing equation "
                                           "(HP + LP tap sum)."),
                           "units": "pu"}]

    return descriptions_list


def ieeeg1(dae, data, name, bus_name, backend=None):
    """
    Example data entry (REE NTS parameter set)::

        "gov": {"type": "ieeeg1",
                "K": 20.0,
                "K_1": 0.3, "K_3": 0.3, "K_5": 0.4, "K_7": 0.0,
                "K_2": 0.0, "K_4": 0.0, "K_6": 0.0, "K_8": 0.0,
                "T_1": 0.0, "T_2": 0.0, "T_3": 0.1,
                "T_4": 0.3, "T_5": 7.0, "T_6": 0.6, "T_7": 0.0,
                "U_o": 0.5, "U_c": -0.5,
                "P_max": 1.0, "P_min": 0.0,
                "p_c": 0.8}

    The ``p_c`` value sets both the ``u_ini`` and ``u_run`` inputs and
    seeds the turbine-cascade initial guesses. At steady state the
    mechanical power output equals ``p_c`` when the HP+LP fractions sum
    to unity.
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
    p_agc = backend.symbols(f"p_agc")

    # External input (dispatch setpoint).
    p_c = backend.symbols(f"p_c_{name}")

    # Secondary-control gain carried by the synchronous machine.
    K_sec = backend.symbols(f"K_sec_{name}")

    # Dynamic states.
    x_3 = backend.symbols(f"x_3_gov_{name}")
    x_4 = backend.symbols(f"x_4_gov_{name}")
    x_5 = backend.symbols(f"x_5_gov_{name}")
    x_6 = backend.symbols(f"x_6_gov_{name}")

    # Algebraic state.
    p_m = backend.symbols(f"p_m_{name}")

    # Parameters.
    K = backend.symbols(f"K_gov_{name}")
    K_1 = backend.symbols(f"K_1_gov_{name}")
    K_3 = backend.symbols(f"K_3_gov_{name}")
    K_5 = backend.symbols(f"K_5_gov_{name}")
    K_7 = backend.symbols(f"K_7_gov_{name}")
    K_2 = backend.symbols(f"K_2_gov_{name}")
    K_4 = backend.symbols(f"K_4_gov_{name}")
    K_6 = backend.symbols(f"K_6_gov_{name}")
    K_8 = backend.symbols(f"K_8_gov_{name}")
    T_3 = backend.symbols(f"T_3_gov_{name}")
    T_4 = backend.symbols(f"T_4_gov_{name}")
    T_5 = backend.symbols(f"T_5_gov_{name}")
    T_6 = backend.symbols(f"T_6_gov_{name}")
    U_o = backend.symbols(f"U_o_gov_{name}")
    U_c = backend.symbols(f"U_c_gov_{name}")
    P_max = backend.symbols(f"P_max_gov_{name}")
    P_min = backend.symbols(f"P_min_gov_{name}")
    K_awu = backend.symbols(f"K_awu_gov_{name}")

    # Servo valve — position limiter with anti-windup feedback on the
    # integrator state.
    Domega = 1.0 - omega
    y_g_nosat = x_3
    y_g = backend.Piecewise((P_min, y_g_nosat < P_min),
                            (P_max, y_g_nosat > P_max),
                            (y_g_nosat, True))

    u_3 = K * Domega + p_c - y_g # + K_sec * p_agc
    u_g_nosat = u_3 / T_3
    u_g = backend.Piecewise((U_c, u_g_nosat < U_c),
                            (U_o, u_g_nosat > U_o),
                            (u_g_nosat, True))
    u_awu = K_awu * (y_g - y_g_nosat)

    dx_3 = u_g + u_awu

    # Steam turbine cascade — three lags (T_7 collapses under REE).
    dx_4 = (y_g - x_4) / T_4
    dx_5 = (x_4 - x_5) / T_5
    dx_6 = (x_5 - x_6) / T_6

    # Mechanical power — HP + LP tap sum. K_7, K_8 multiply x_6 because
    # T_7 = 0 collapses x_7 into x_6.
    p_m_hp = K_1 * x_4 + K_3 * x_5 + K_5 * x_6 + K_7 * x_6
    p_m_lp = K_2 * x_4 + K_4 * x_5 + K_6 * x_6 + K_8 * x_6
    g_p_m = p_m_hp + p_m_lp - p_m

    # --- ini/run variable partition (no swap: governor does not pin V) --
    dae['f'] += [dx_3, dx_4, dx_5, dx_6]
    dae['x'] += [x_3, x_4, x_5, x_6]
    dae['g'] += [g_p_m]
    dae['y_ini'] += [p_m]
    dae['y_run'] += [p_m]


    dae['params_dict'].update({str(K): gov_data['K']})
    dae['params_dict'].update({str(K_1): gov_data['K_1']})
    dae['params_dict'].update({str(K_3): gov_data['K_3']})
    dae['params_dict'].update({str(K_5): gov_data['K_5']})
    dae['params_dict'].update({str(K_7): gov_data['K_7']})
    dae['params_dict'].update({str(K_2): gov_data['K_2']})
    dae['params_dict'].update({str(K_4): gov_data['K_4']})
    dae['params_dict'].update({str(K_6): gov_data['K_6']})
    dae['params_dict'].update({str(K_8): gov_data['K_8']})
    dae['params_dict'].update({str(T_3): gov_data['T_3']})
    dae['params_dict'].update({str(T_4): gov_data['T_4']})
    dae['params_dict'].update({str(T_5): gov_data['T_5']})
    dae['params_dict'].update({str(T_6): gov_data['T_6']})
    dae['params_dict'].update({str(U_o): gov_data['U_o']})
    dae['params_dict'].update({str(U_c): gov_data['U_c']})
    dae['params_dict'].update({str(P_max): gov_data['P_max']})
    dae['params_dict'].update({str(P_min): gov_data['P_min']})
    dae['params_dict'].update({str(K_awu): 1000.0})

    p_c_ini = gov_data.get('p_c', 0.5)
    dae['u_ini_dict'].update({str(p_c): p_c_ini})
    dae['u_run_dict'].update({str(p_c): p_c_ini})

    dae['h_dict'].update({str(p_c): p_c})

    # Steady-state seeds: dx_3 = 0 → y_g = p_c (within limits) → x_3 = p_c;
    # turbine cascade settles to x_4 = x_5 = x_6 = p_c; p_m = (sum K_i) p_c.
    dae['xy_0_dict'].update({str(x_3): p_c_ini})
    dae['xy_0_dict'].update({str(x_4): p_c_ini})
    dae['xy_0_dict'].update({str(x_5): p_c_ini})
    dae['xy_0_dict'].update({str(x_6): p_c_ini})
    dae['xy_0_dict'].update({str(p_m): p_c_ini})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('ieeeg1.hjson')
    grid.uz_jacs = False
    grid.construct('temp_ieeeg1')

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model('temp_ieeeg1')

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
