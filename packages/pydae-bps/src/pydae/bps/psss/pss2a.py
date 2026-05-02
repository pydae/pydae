# -*- coding: utf-8 -*-
r"""
IEEE PSS2A dual-input power system stabilizer (REE NTS parameter set).

The PSS2A model is the IEEE 421.5 Type PSS2A dual-input stabilizer:
speed deviation is combined with an accelerating-power proxy derived
from electrical power to produce a voltage-regulator modulation
signal $v_{pss}$. The dual-input design cancels the synchronizing
torque component of the power signal so that only the damping
component remains, avoiding interaction with turbine torsional
modes.

**Signal path**

The speed-channel signal is the speed deviation from synchronism,

$$V_1 = \omega - \omega_{ref}$$

with $\omega_{ref} = 1$ pu. It passes through two cascaded washouts,

$$\frac{d x_{w1}}{dt} = \frac{V_1 - x_{w1}}{T_{w1}},\quad z_{w1} = V_1 - x_{w1}$$
$$\frac{d x_{w2}}{dt} = \frac{z_{w1} - x_{w2}}{T_{w2}},\quad z_{w2} = z_{w1} - x_{w2}$$

The transducer lag $1/(1 + s T_6)$ is a unity pass-through under the
REE spec ($T_6 = 0$), so $y_1 = z_{w2}$.

The power-channel signal $V_2 = p_g$ passes through a single washout
($T_{w4} = 0$ collapses the second),

$$\frac{d x_{w3}}{dt} = \frac{V_2 - x_{w3}}{T_{w3}},\quad z_{w3} = V_2 - x_{w3}$$

followed by the accelerating-power scale $K_{s2}$ and a lag with time
constant $T_7$ (typically chosen so that $K_{s2} T_7 = 1/(2H)$ for
cancellation of the synchronizing torque),

$$\frac{d x_p}{dt} = \frac{K_{s2}\, z_{w3} - x_p}{T_7},\quad y_3 = x_p$$

The ramp-tracking filter input combines the speed channel with the
weighted power proxy,

$$u_{rt} = y_1 + K_{s3}\, y_3$$

and the filter has transfer function
$F(s) = \left[(1 + s T_8)/(1 + s T_9)\right]^M \cdot 1/(1 + s T_9)^N$.
Under the REE spec $T_8 = 0$, $M = 5$, $N = 1$, so
$F(s) = 1/(1 + s T_9)^6$ — six cascaded first-order lags with state
variables $x_{r1} \dots x_{r6}$ and output $y_{rt} = x_{r6}$.

The dual-input cancellation applies the main gain,

$$u_{ll} = K_{s1}\,(y_{rt} - y_3)$$

followed by two series lead-lag compensators,

$$\frac{d x_{l1}}{dt} = \frac{u_{ll} - x_{l1}}{T_2},\quad
  z_{l1} = (u_{ll} - x_{l1})\frac{T_1}{T_2} + x_{l1}$$
$$\frac{d x_{l2}}{dt} = \frac{z_{l1} - x_{l2}}{T_4},\quad
  z_{l2} = (z_{l1} - x_{l2})\frac{T_3}{T_4} + x_{l2}$$

The output is saturated by the hard output limits,

$$0 = \mathrm{sat}(z_{l2},\; V_{STmin},\; V_{STmax}) - v_{pss}$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**Simplifications under the REE parameter set**

- $T_6 = 0$ — speed transducer is a unity pass-through; no state.
- $T_{w4} = 0$ — second power washout collapses; only $T_{w3}$ active.
- $T_8 = 0$, $M = 5$, $N = 1$ — ramp-tracker reduces to six cascaded
  lags $1/(1 + s T_9)^6$. Generalising to $T_8 \neq 0$ would require
  replacing each lag with a lead-lag, which is out of scope here.

**No ini/run swap**

The PSS output is an algebraic variable $v_{pss}$ that feeds the AVR
summing junction. The PSS does not pin a voltage setpoint, so the
``y_ini``/``y_run`` partitions are identical. At steady state
($\omega = 1$, $p_g$ constant) every state settles to zero and the
PSS output is zero — the stabiliser only responds to transients.

**Configuration**

Example data entry (REE NTS defaults)::

    "pss": {"type": "pss2a",
            "T_w1": 2.0, "T_w2": 2.0, "T_w3": 2.0, "T_w4": 0.0,
            "T_6": 0.0, "T_7": 2.0, "T_8": 0.0, "T_9": 0.1,
            "K_s1": 17.069, "K_s2": 0.158, "K_s3": 1.0,
            "T_1": 0.28, "T_2": 0.04, "T_3": 0.28, "T_4": 0.12,
            "V_STmax": 0.1, "V_STmin": -0.1,
            "M": 5, "N": 1}

``M`` and ``N`` are recorded in the parameter dict for documentation
but do not enter the symbolic equations — the ramp-tracker order is
baked in at six lags. Change the structure if a different order is
required.
"""


def descriptions():
    """Single source of truth for pss2a parameters, inputs, states, and outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "T_{w1}", "data": "T_w1",
                           "model": "T_w1_pss", "default": 2.0,
                           "description": "First speed washout time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_{w2}", "data": "T_w2",
                           "model": "T_w2_pss", "default": 2.0,
                           "description": "Second speed washout time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_{w3}", "data": "T_w3",
                           "model": "T_w3_pss", "default": 2.0,
                           "description": "First power washout time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_7", "data": "T_7",
                           "model": "T_7_pss", "default": 2.0,
                           "description": ("Power-channel lag (chosen so that "
                                           "K_s2 T_7 ≈ 1/(2H) for synchronizing "
                                           "torque cancellation)"),
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_9", "data": "T_9",
                           "model": "T_9_pss", "default": 0.1,
                           "description": "Ramp-tracker lag time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{s1}", "data": "K_s1",
                           "model": "K_s1_pss", "default": 17.069,
                           "description": "Main stabiliser gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{s2}", "data": "K_s2",
                           "model": "K_s2_pss", "default": 0.158,
                           "description": "Power-channel gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{s3}", "data": "K_s3",
                           "model": "K_s3_pss", "default": 1.0,
                           "description": "Cross-path gain at ramp-tracker input",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_1", "data": "T_1",
                           "model": "T_1_pss", "default": 0.28,
                           "description": "Lead-lag 1 numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_2", "data": "T_2",
                           "model": "T_2_pss", "default": 0.04,
                           "description": "Lead-lag 1 denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_3", "data": "T_3",
                           "model": "T_3_pss", "default": 0.28,
                           "description": "Lead-lag 2 numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_4", "data": "T_4",
                           "model": "T_4_pss", "default": 0.12,
                           "description": "Lead-lag 2 denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{STmax}",
                           "data": "V_STmax", "model": "V_STmax_pss",
                           "default": 0.1,
                           "description": "Upper PSS output limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{STmin}",
                           "data": "V_STmin", "model": "V_STmin_pss",
                           "default": -0.1,
                           "description": "Lower PSS output limit",
                           "units": "pu"}]

    # Inputs (from the rest of the system, not user-provided)
    descriptions_list += [{"type": "Input", "tex": "\\omega",
                           "data": "", "model": "omega", "ieee": "V_1",
                           "default": 1.0,
                           "description": ("Machine rotor speed (taken from "
                                           "the synchronous machine). Used as "
                                           "ω - 1 in the first channel."),
                           "units": "pu"}]
    descriptions_list += [{"type": "Input", "tex": "p_g",
                           "data": "", "model": "p_g", "ieee": "V_2",
                           "default": 0.0,
                           "description": ("Generator electrical power (taken "
                                           "from the synchronous machine's "
                                           "algebraic output). Second channel "
                                           "input."),
                           "units": "pu"}]

    # Dynamic states
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{w1}",
                           "data": "", "model": "x_w1_pss", "default": "",
                           "description": "First speed washout state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{w2}",
                           "data": "", "model": "x_w2_pss", "default": "",
                           "description": "Second speed washout state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{w3}",
                           "data": "", "model": "x_w3_pss", "default": "",
                           "description": "Power washout state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_p",
                           "data": "", "model": "x_p_pss", "default": "",
                           "description": ("Power-channel lag state "
                                           "(accelerating-power proxy)"),
                           "units": "pu"}]
    for i in range(1, 7):
        descriptions_list += [{"type": "Dynamic State",
                               "tex": f"x_{{r{i}}}",
                               "data": "",
                               "model": f"x_r{i}_pss",
                               "default": "",
                               "description": (f"Ramp-tracker lag state "
                                               f"{i}/6"),
                               "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{l1}",
                           "data": "", "model": "x_l1_pss", "default": "",
                           "description": "Lead-lag 1 internal state",
                           "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{l2}",
                           "data": "", "model": "x_l2_pss", "default": "",
                           "description": "Lead-lag 2 internal state",
                           "units": "pu"}]

    # Algebraic state
    descriptions_list += [{"type": "Algebraic State", "tex": "v_{pss}",
                           "data": "", "model": "v_pss", "default": "",
                           "description": ("PSS output signal added to the "
                                           "AVR summing junction (saturated)."),
                           "units": "pu"}]

    return descriptions_list


def pss2a(dae, data, name, bus_name, backend=None):
    """
    Example data entry (REE NTS parameter set)::

        "pss": {"type": "pss2a",
                 "T_w1": 2.0, "T_w2": 2.0, "T_w3": 2.0, "T_w4": 0.0,
                 "T_6": 0.0, "T_7": 2.0, "T_8": 0.0, "T_9": 0.1,
                 "K_s1": 17.069, "K_s2": 0.158, "K_s3": 1.0,
                 "T_1": 0.28, "T_2": 0.04, "T_3": 0.28, "T_4": 0.12,
                 "V_STmax": 0.1, "V_STmin": -0.1,
                 "M": 5, "N": 1}

    All states initialise to zero — at steady state the PSS output
    $v_{pss} = 0$ and only transients produce a non-zero signal.
    """
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'Piecewise': sym.Piecewise,
            'sin': sym.sin,
            'cos': sym.cos,
            'sqrt': sym.sqrt,
            'exp': sym.exp,
        })()

    pss_data = data['pss']

    # Inputs from the rest of the system.
    omega = backend.symbols(f"omega_{name}", real=True)
    p_g = backend.symbols(f"p_g_{name}", real=True)

    # Dynamic states — speed channel.
    x_w1 = backend.symbols(f"x_w1_pss_{name}", real=True)
    x_w2 = backend.symbols(f"x_w2_pss_{name}", real=True)

    # Dynamic states — power channel.
    x_w3 = backend.symbols(f"x_w3_pss_{name}", real=True)
    x_p = backend.symbols(f"x_p_pss_{name}", real=True)

    # Dynamic states — ramp-tracking filter (6 cascaded lags).
    x_r1 = backend.symbols(f"x_r1_pss_{name}", real=True)
    x_r2 = backend.symbols(f"x_r2_pss_{name}", real=True)
    x_r3 = backend.symbols(f"x_r3_pss_{name}", real=True)
    x_r4 = backend.symbols(f"x_r4_pss_{name}", real=True)
    x_r5 = backend.symbols(f"x_r5_pss_{name}", real=True)
    x_r6 = backend.symbols(f"x_r6_pss_{name}", real=True)

    # Dynamic states — lead-lag stages.
    x_l1 = backend.symbols(f"x_l1_pss_{name}", real=True)
    x_l2 = backend.symbols(f"x_l2_pss_{name}", real=True)

    # Algebraic state.
    v_pss = backend.symbols(f"v_pss_{name}", real=True)

    # Parameters.
    T_w1 = backend.symbols(f"T_w1_pss_{name}", real=True)
    T_w2 = backend.symbols(f"T_w2_pss_{name}", real=True)
    T_w3 = backend.symbols(f"T_w3_pss_{name}", real=True)
    T_7 = backend.symbols(f"T_7_pss_{name}", real=True)
    T_9 = backend.symbols(f"T_9_pss_{name}", real=True)
    K_s1 = backend.symbols(f"K_s1_pss_{name}", real=True)
    K_s2 = backend.symbols(f"K_s2_pss_{name}", real=True)
    K_s3 = backend.symbols(f"K_s3_pss_{name}", real=True)
    T_1 = backend.symbols(f"T_1_pss_{name}", real=True)
    T_2 = backend.symbols(f"T_2_pss_{name}", real=True)
    T_3 = backend.symbols(f"T_3_pss_{name}", real=True)
    T_4 = backend.symbols(f"T_4_pss_{name}", real=True)
    V_STmax = backend.symbols(f"V_STmax_pss_{name}", real=True)
    V_STmin = backend.symbols(f"V_STmin_pss_{name}", real=True)

    # Speed channel: V_1 = omega - 1; two washouts; T_6 = 0 skips transducer.
    V_1 = omega - 1.0
    dx_w1 = (V_1 - x_w1) / T_w1
    z_w1 = V_1 - x_w1
    dx_w2 = (z_w1 - x_w2) / T_w2
    z_w2 = z_w1 - x_w2
    y_1 = z_w2

    # Power channel: single washout (T_w4 = 0); K_s2 · 1/(1 + s T_7).
    V_2 = p_g
    dx_w3 = (V_2 - x_w3) / T_w3
    z_w3 = V_2 - x_w3
    dx_p = (K_s2 * z_w3 - x_p) / T_7
    y_3 = x_p

    # Ramp-tracking filter: 1/(1 + s T_9)^6 (T_8 = 0, M = 5, N = 1).
    u_rt = y_1 + K_s3 * y_3
    dx_r1 = (u_rt - x_r1) / T_9
    dx_r2 = (x_r1 - x_r2) / T_9
    dx_r3 = (x_r2 - x_r3) / T_9
    dx_r4 = (x_r3 - x_r4) / T_9
    dx_r5 = (x_r4 - x_r5) / T_9
    dx_r6 = (x_r5 - x_r6) / T_9
    y_rt = x_r6

    # Main gain with dual-input cancellation.
    u_ll = K_s1 * (y_rt - y_3)

    # Lead-lag 1.
    dx_l1 = (u_ll - x_l1) / T_2
    z_l1 = (u_ll - x_l1) * T_1 / T_2 + x_l1

    # Lead-lag 2.
    dx_l2 = (z_l1 - x_l2) / T_4
    z_l2 = (z_l1 - x_l2) * T_3 / T_4 + x_l2

    # Output saturation.
    v_pss_nosat = z_l2
    v_pss_sat = backend.Piecewise((V_STmin, v_pss_nosat < V_STmin),
                              (V_STmax, v_pss_nosat > V_STmax),
                              (v_pss_nosat, True))
    g_v_pss = v_pss_sat - v_pss

    # --- ini/run variable partition (no swap) ---------------------------
    dae['f'] += [dx_w1, dx_w2, dx_w3, dx_p,
                 dx_r1, dx_r2, dx_r3, dx_r4, dx_r5, dx_r6,
                 dx_l1, dx_l2]
    dae['x'] += [x_w1, x_w2, x_w3, x_p,
                 x_r1, x_r2, x_r3, x_r4, x_r5, x_r6,
                 x_l1, x_l2]
    dae['g'] += [g_v_pss]
    dae['y_ini'] += [v_pss]
    dae['y_run'] += [v_pss]

    dae['params_dict'].update({str(T_w1): pss_data['T_w1']})
    dae['params_dict'].update({str(T_w2): pss_data['T_w2']})
    dae['params_dict'].update({str(T_w3): pss_data['T_w3']})
    dae['params_dict'].update({str(T_7): pss_data['T_7']})
    dae['params_dict'].update({str(T_9): pss_data['T_9']})
    dae['params_dict'].update({str(K_s1): pss_data['K_s1']})
    dae['params_dict'].update({str(K_s2): pss_data['K_s2']})
    dae['params_dict'].update({str(K_s3): pss_data['K_s3']})
    dae['params_dict'].update({str(T_1): pss_data['T_1']})
    dae['params_dict'].update({str(T_2): pss_data['T_2']})
    dae['params_dict'].update({str(T_3): pss_data['T_3']})
    dae['params_dict'].update({str(T_4): pss_data['T_4']})
    dae['params_dict'].update({str(V_STmax): pss_data['V_STmax']})
    dae['params_dict'].update({str(V_STmin): pss_data['V_STmin']})

    # Steady-state seeds: all states zero, v_pss = 0 (PSS is silent at SS).
    for state in [x_w1, x_w2, x_w3, x_p,
                  x_r1, x_r2, x_r3, x_r4, x_r5, x_r6,
                  x_l1, x_l2, v_pss]:
        dae['xy_0_dict'].update({str(state): 0.0})


def test():
    from pydae.core import Builder, Model
    from pydae.bps import BpsBuilder
    import pytest

    grid = BpsBuilder('pss2a.hjson')
    grid.uz_jacs = False
    grid.construct('temp_pss2a')

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()

    model = Model('temp_pss2a')

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
