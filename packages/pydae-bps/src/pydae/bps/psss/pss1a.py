# -*- coding: utf-8 -*-
r"""
IEEE PSS1A power system stabilizer (single-input, two lead-lag).

Matches the IEEE Std 421.5 PSS1A topology: speed deviation (or power)
passes through a washout, two series lead-lag compensators, a gain stage,
and a hard output limiter.  An optional output low-pass filter with time
constant $T_6$ is supported but omitted when $T_6 = 0$ (most common case).

**Signal path**

Speed deviation:

$$\Delta\omega = \omega - 1$$

Washout $s T_5 / (1 + s T_5)$ with state $x_{wo}$:

$$\frac{d x_{wo}}{dt} = \frac{\Delta\omega - x_{wo}}{T_5}, \qquad
  z_{wo} = \Delta\omega - x_{wo}$$

First lead-lag $(1 + T_1 s)/(1 + T_2 s)$ with state $x_{12}$:

$$\frac{d x_{12}}{dt} = \frac{z_{wo} - x_{12}}{T_2}, \qquad
  z_{12} = (z_{wo} - x_{12})\frac{T_1}{T_2} + x_{12}$$

Second lead-lag $(1 + T_3 s)/(1 + T_4 s)$ with state $x_{34}$:

$$\frac{d x_{34}}{dt} = \frac{z_{12} - x_{34}}{T_4}, \qquad
  z_{34} = (z_{12} - x_{34})\frac{T_3}{T_4} + x_{34}$$

Gain and hard limits:

$$0 = \mathrm{sat}\!\left(K_s\, z_{34},\; V_{stmin},\; V_{stmax}\right) - v_{pss}$$

The output filter ($T_6 > 0$) is not yet modelled as a separate dynamic
state; set $T_6 = 0$ (default) to reproduce the standard two-lag PSS1A
behaviour used in most benchmark studies.

**Steady-state**

At $\omega = 1$: all states and $v_{pss}$ are zero.

**Configuration**

Example data entry (IEEE PES TR18 3MIB benchmark)::

    "pss": {"type": "pss1a",
            "K_s": 1.0, "T_5": 10.0,
            "T_1": 0.2, "T_2": 0.05,
            "T_3": 0.2, "T_4": 0.05,
            "T_6": 0.0,
            "V_stmax": 0.1, "V_stmin": -0.1}

Parameter names follow IEEE 421.5 Table D.15.  The ``pss_kundur_2``
model is structurally identical but uses the Kundur textbook names
(``K_stab``, ``T_w``, symmetric ``V_lim``).
"""

def descriptions():
    """Single source of truth for pss1a parameters, inputs, states, outputs."""
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "K_s",
                           "data": "K_s", "model": "K_s_pss",
                           "default": 1.0,
                           "description": "Main stabiliser gain",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_5",
                           "data": "T_5", "model": "T_5_pss",
                           "default": 10.0,
                           "description": "Washout time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_1",
                           "data": "T_1", "model": "T_1_pss",
                           "default": 0.2,
                           "description": "Lead-lag 1 numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_2",
                           "data": "T_2", "model": "T_2_pss",
                           "default": 0.05,
                           "description": "Lead-lag 1 denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_3",
                           "data": "T_3", "model": "T_3_pss",
                           "default": 0.2,
                           "description": "Lead-lag 2 numerator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_4",
                           "data": "T_4", "model": "T_4_pss",
                           "default": 0.05,
                           "description": "Lead-lag 2 denominator time constant",
                           "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{stmax}",
                           "data": "V_stmax", "model": "V_stmax_pss",
                           "default": 0.1,
                           "description": "Upper PSS output limit",
                           "units": "pu"}]
    descriptions_list += [{"type": "Parameter", "tex": "V_{stmin}",
                           "data": "V_stmin", "model": "V_stmin_pss",
                           "default": -0.1,
                           "description": "Lower PSS output limit",
                           "units": "pu"}]

    descriptions_list += [{"type": "Dynamic State", "tex": "x_{wo}",
                           "data": "", "model": "x_wo_pss", "default": "",
                           "description": "Washout state", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{12}",
                           "data": "", "model": "x_12_pss", "default": "",
                           "description": "Lead-lag 1 internal state", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{34}",
                           "data": "", "model": "x_34_pss", "default": "",
                           "description": "Lead-lag 2 internal state", "units": "pu"}]

    descriptions_list += [{"type": "Algebraic State", "tex": "v_{pss}",
                           "data": "", "model": "v_pss", "default": "",
                           "description": "PSS output sent to the AVR summing junction",
                           "units": "pu"}]

    return descriptions_list


def pss1a(dae, data, name, bus_name, backend=None):
    r"""
    Attach the PSS1A stabiliser to *dae* for generator *name*.

    T_6 (output filter) is read from data but not modelled as a state;
    set T_6 = 0 (default) for the standard two-lag configuration.
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

    omega  = backend.symbols(f"omega_{name}",      real=True)
    x_wo   = backend.symbols(f"x_wo_pss_{name}",   real=True)
    x_12   = backend.symbols(f"x_12_pss_{name}",   real=True)
    x_34   = backend.symbols(f"x_34_pss_{name}",   real=True)
    v_pss  = backend.symbols(f"v_pss_{name}",       real=True)

    K_s    = backend.symbols(f"K_s_pss_{name}",    real=True)
    T_5    = backend.symbols(f"T_5_pss_{name}",    real=True)
    T_1    = backend.symbols(f"T_1_pss_{name}",    real=True)
    T_2    = backend.symbols(f"T_2_pss_{name}",    real=True)
    T_3    = backend.symbols(f"T_3_pss_{name}",    real=True)
    T_4    = backend.symbols(f"T_4_pss_{name}",    real=True)
    V_stmax = backend.symbols(f"V_stmax_pss_{name}", real=True)
    V_stmin = backend.symbols(f"V_stmin_pss_{name}", real=True)

    Domega = omega - 1.0

    # Washout
    dx_wo = (Domega - x_wo) / T_5
    z_wo  = Domega - x_wo

    # Lead-lag 1
    dx_12 = (z_wo - x_12) / T_2
    z_12  = (z_wo - x_12) * T_1 / T_2 + x_12

    # Lead-lag 2
    dx_34 = (z_12 - x_34) / T_4
    z_34  = (z_12 - x_34) * T_3 / T_4 + x_34

    # Gain + output limiter
    v_nosat = K_s * z_34
    v_sat   = backend.Piecewise((V_stmin, v_nosat < V_stmin),
                            (V_stmax, v_nosat > V_stmax),
                            (v_nosat, True))
    g_v_pss = v_sat - v_pss

    dae['f']     += [dx_wo, dx_12, dx_34]
    dae['x']     += [x_wo,  x_12,  x_34]
    dae['g']     += [g_v_pss]
    dae['y_ini'] += [v_pss]
    dae['y_run'] += [v_pss]

    dae['params_dict'].update({
        str(K_s):     pss_data['K_s'],
        str(T_5):     pss_data['T_5'],
        str(T_1):     pss_data['T_1'],
        str(T_2):     pss_data['T_2'],
        str(T_3):     pss_data['T_3'],
        str(T_4):     pss_data['T_4'],
        str(V_stmax): pss_data['V_stmax'],
        str(V_stmin): pss_data['V_stmin'],
    })

    for state in [x_wo, x_12, x_34, v_pss]:
        dae['xy_0_dict'].update({str(state): 0.0})
