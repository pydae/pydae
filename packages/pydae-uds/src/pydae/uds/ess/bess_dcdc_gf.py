r"""
Battery-energy-storage system with DC-DC grid-forming converter.

Two stages: a low-voltage battery (with an open-circuit-voltage curve and
internal resistance) feeds a DC-DC converter that imposes the high-voltage
DC bus with a $P/V$ linear droop and a slow SOC-balancing loop.

**Battery stage** — OCV curve as a piecewise-linear interpolant of
`(socs, es)` HJSON samples (replacement for the legacy SymPy
`interpolating_spline`), plus internal resistance $R_{bat}$:

$$e_{ocv}(\text{soc}) = \text{PWL}(\text{soc};\, \text{socs}, \text{es})$$
$$0 = e_{ocv} - i_l R_{bat} - v_{bat}$$
$$\dot{\text{soc}} = -\frac{i_l \cdot e_{ocv}}{1000 \cdot 3600 \cdot E_{kWh}}$$
$$\dot{\xi_{soc}} = \text{soc}_{ref} - \text{soc}$$

The SOC-PI ($K_p, K_i$ both default to $10^{-6}$ for slow action) produces
a $p_{soc}$ signal that can modulate the HV-side voltage reference.

**DC-DC stage** — HV pole-to-pole voltage with linear current/power droop:

$$v_h = v_{hp} - v_{hn}, \qquad
  e_h = e_h^{ref} - D_i i_{hf} - D_p i_{hf} v_h + K_{soc}\,p_{soc}$$

Currents into the positive and negative HV poles are derived from a small
$R_h, R_g$ Thevenin equivalent:

$$i_{hp} = (v_{og} + e_h/2 - v_{hp})/R_h, \qquad
  i_{hn} = (v_{og} - e_h/2 - v_{hn})/R_h$$

with $v_{og} = i_g R_g, \; i_g = (v_{hn} + v_{hp})/(2 R_g + R_h)$.

**LV-side balance** — HV input power plus conduction loss equals LV battery
power:

$$p_l = p_h + A\,i_{hp}^2 + B\,|i_{hp}| + C, \qquad i_l = p_l/v_{bat} - i_{charger}$$

with $|i_{hp}|$ approximated by $\sqrt{i_{hp}^2 + 10^{-12}}$ for
differentiability.

**HJSON snippet**

```hjson
ess: [
    {bus: "D3", type: "bess_dcdc_gf", E_kWh: 1000, soc_0: 0.1, v_ref: 800,
     A: 0.0, B: 0.0, C: 0.0,
     socs: [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
     es:   [600, 650, 680, 700, 710, 750]}
]
```
"""

import numpy as np


def descriptions():
    return [
        # parameters
        {"type": "Parameter", "tex": "E_{kWh}",  "data": "E_kWh", "model": "E_kWh_{name}",  "default": "", "units": "kWh", "description": "Battery capacity"},
        {"type": "Parameter", "tex": "R_h",      "data": "",      "model": "R_h_{name}",    "default": 0.01,"units": r"\Omega", "description": "HV-side series resistance"},
        {"type": "Parameter", "tex": "R_g",      "data": "",      "model": "R_g_{name}",    "default": 3,  "units": r"\Omega", "description": "HV neutral-to-ground resistance"},
        {"type": "Parameter", "tex": "A",        "data": "A",     "model": "A_{name}",      "default": "", "units": r"\Omega", "description": "Quadratic conduction-loss coefficient"},
        {"type": "Parameter", "tex": "B",        "data": "B",     "model": "B_{name}",      "default": "", "units": "V",       "description": "Linear conduction-loss coefficient"},
        {"type": "Parameter", "tex": "C",        "data": "C",     "model": "C_{name}",      "default": "", "units": "W",       "description": "No-load loss"},
        {"type": "Parameter", "tex": "D_p",      "data": "",      "model": "Droop_p_{name}","default": 0.0,"units": r"\Omega^{-1}", "description": "Power droop gain"},
        {"type": "Parameter", "tex": "D_i",      "data": "",      "model": "Droop_i_{name}","default": 0.0,"units": r"\Omega",  "description": "Current droop gain"},
        {"type": "Parameter", "tex": "T_f",      "data": "",      "model": "T_f_{name}",    "default": 0.1,"units": "s",        "description": "Current LPF time constant"},
        {"type": "Parameter", "tex": "K_p",      "data": "",      "model": "K_p_{name}",    "default": 1e-6,"units": "-",       "description": "SOC-PI proportional gain"},
        {"type": "Parameter", "tex": "K_i",      "data": "",      "model": "K_i_{name}",    "default": 1e-6,"units": "1/s",     "description": "SOC-PI integral gain"},
        {"type": "Parameter", "tex": "K_{soc}",  "data": "",      "model": "K_soc_{name}",  "default": 0.001,"units": "V/W",    "description": "SOC-to-voltage coupling"},
        {"type": "Parameter", "tex": "K_{charger}","data": "",    "model": "K_charger_{name}","default": 0.0,"units": "-",      "description": "Charger injection gain"},
        {"type": "Parameter", "tex": "R_{bat}",  "data": "R_bat", "model": "R_bat_{name}",  "default": 0.0,"units": r"\Omega", "description": "Battery internal resistance"},
        # inputs
        {"type": "Input", "tex": "e_h^{ref}",    "data": "v_ref", "model": "e_h_ref_{name}","default": "v_ref","units": "V",   "description": "HV voltage reference"},
        {"type": "Input", "tex": "\\text{soc}_{ref}", "data": "", "model": "soc_ref_{name}", "default": 0.5, "units": "-",     "description": "SOC reference"},
        # dynamic states
        {"type": "Dynamic State", "tex": "i_{hf}", "data": "", "model": "i_h_f_{name}",  "default": "", "units": "A", "description": "HV-side filtered current"},
        {"type": "Dynamic State", "tex": "\\text{soc}", "data": "", "model": "soc_{name}", "default": "soc_0", "units": "-", "description": "Battery state of charge"},
        {"type": "Dynamic State", "tex": "\\xi_{soc}", "data": "", "model": "xi_soc_{name}", "default": 0.5, "units": "-·s", "description": "SOC-PI integrator state"},
        # algebraic states
        {"type": "Algebraic State", "tex": "v_{bat}", "data": "", "model": "v_bat_{name}", "default": "", "units": "V", "description": "Battery terminal voltage"},
        # outputs
        {"type": "Output", "tex": "i_l",   "data": "", "model": "i_l_{name}",        "default": "", "units": "A", "description": "Battery-side current"},
        {"type": "Output", "tex": "p_h",   "data": "", "model": "p_h_{name}",        "default": "", "units": "W", "description": "HV-side power"},
        {"type": "Output", "tex": "e_h",   "data": "", "model": "e_h_{name}",        "default": "", "units": "V", "description": "HV reference (after droop / SOC)"},
        {"type": "Output", "tex": "i_{charger}", "data": "", "model": "i_charger_{name}", "default": "", "units": "A", "description": "Charger current injection"},
    ]


def _piecewise_linear(bk, x, xs, ys):
    """Piecewise-linear interpolation y = f(x) given sample (xs, ys) pairs.
    For x >= xs[-1] returns 0 (matches the legacy SymPy spline-with-tail).
    Backend-agnostic via bk.Piecewise.
    """
    pieces = []
    for k in range(len(xs) - 1):
        slope = (ys[k+1] - ys[k]) / (xs[k+1] - xs[k])
        line = ys[k] + slope*(x - xs[k])
        pieces.append((line, x < xs[k+1]))
    pieces.append((0, True))
    return bk.Piecewise(*pieces)


def bess_dcdc_gf(grid, data, name, bus_name):
    '''
    BESS with DC/DC grid former.
    Battery on LV side; grid former on HV side with a P_h/V_h linear droop.
    Dual-backend (SymPy / CasADi) via grid.backend.
    '''
    bus_hv_name = data['bus']
    name = bus_hv_name
    bk = grid.backend

    A_value = data['A']
    B_value = data['B']
    C_value = data['C']

    # --- Common
    soc      = bk.symbols(f'soc_{name}')
    xi_soc   = bk.symbols(f'xi_soc_{name}')
    K_p      = bk.symbols(f'K_p_{name}')
    K_i      = bk.symbols(f'K_i_{name}')
    K_soc    = bk.symbols(f'K_soc_{name}')
    K_charger = bk.symbols(f'K_charger_{name}')

    # voltages
    v_hp   = bk.symbols(f'V_{bus_hv_name}_0_r')
    v_hn   = bk.symbols(f'V_{bus_hv_name}_1_r')
    v_hp_i = bk.symbols(f'V_{bus_hv_name}_0_i')
    v_hn_i = bk.symbols(f'V_{bus_hv_name}_1_i')
    v_bat  = bk.symbols(f'v_bat_{name}')

    # LV-side loss coefficients
    A = bk.symbols(f'A_{bus_hv_name}')
    B = bk.symbols(f'B_{bus_hv_name}')
    C = bk.symbols(f'C_{bus_hv_name}')

    # HV-side parameters
    R_h     = bk.symbols(f'R_h_{bus_hv_name}')
    e_h_ref = bk.symbols(f'e_h_ref_{bus_hv_name}')
    R_g     = bk.symbols(f'R_g_{bus_hv_name}')
    i_h_f   = bk.symbols(f'i_h_f_{bus_hv_name}')
    T_f     = bk.symbols(f'T_f_{bus_hv_name}')
    Droop_i = bk.symbols(f'Droop_i_{bus_hv_name}')
    Droop_p = bk.symbols(f'Droop_p_{bus_hv_name}')

    soc_ref = bk.symbols(f'soc_ref_{name}')

    epsilon_soc = soc_ref - soc
    p_soc = -(K_p*epsilon_soc + K_i*xi_soc)

    v_l = v_bat
    v_h = v_hp - v_hn
    m_h = v_l/e_h_ref
    e_h = e_h_ref - Droop_i*i_h_f - Droop_p*i_h_f*v_h + K_soc*p_soc
    i_g = (v_hn + v_hp)/(2*R_g + R_h)
    v_og = i_g*R_g
    i_hp = (v_og + e_h/2 - v_hp)/R_h
    i_hn = (v_og - e_h/2 - v_hn)/R_h
    p_h = i_hp*e_h
    # smooth |i_hp| so the Jacobian stays defined at zero current
    abs_i_hp = bk.sqrt(i_hp*i_hp + 1e-12)
    p_l = p_h + A*i_hp**2 + B*abs_i_hp + C
    i_charger = K_charger*p_soc/v_l
    i_l = p_l/v_l - i_charger

    di_h_f = 1/T_f*(i_hp - i_h_f)
    grid.dae['f'] += [di_h_f]
    grid.dae['x'] += [i_h_f]

    # current injections (HV DC side)
    idx_hp_r, idx_hp_i = grid.node2idx(bus_hv_name, 'a')
    idx_hn_r, idx_hn_i = grid.node2idx(bus_hv_name, 'b')
    grid.dae['g'][idx_hp_r] += -i_hp
    grid.dae['g'][idx_hn_r] += -i_hn
    grid.dae['g'][idx_hp_i] += -v_hp_i/1e3
    grid.dae['g'][idx_hn_i] += -v_hn_i/1e3

    grid.dae['u_ini_dict'].update({f'e_h_ref_{bus_hv_name}': data['v_ref']})
    grid.dae['u_run_dict'].update({f'e_h_ref_{bus_hv_name}': data['v_ref']})

    grid.dae['params_dict'].update({
        f'R_h_{bus_hv_name}': 0.01,
        f'R_g_{bus_hv_name}': 3,
        f'A_{bus_hv_name}': A_value,
        f'B_{bus_hv_name}': B_value,
        f'C_{bus_hv_name}': C_value,
        f'Droop_p_{bus_hv_name}': 0.0,
        f'Droop_i_{bus_hv_name}': 0.0,
        f'T_f_{bus_hv_name}': 0.1,
    })
    grid.dae['h_dict'][f'm_h_{bus_hv_name}'] = m_h

    # ------------------------- Battery cell stack -------------------------
    E_kWh   = bk.symbols(f'E_kWh_{name}')
    soc_min = bk.symbols(f'soc_min_{name}')
    soc_max = bk.symbols(f'soc_max_{name}')
    A_loss  = bk.symbols(f'A_loss_{name}')
    B_loss  = bk.symbols(f'B_loss_{name}')
    C_loss  = bk.symbols(f'C_loss_{name}')
    B_0     = bk.symbols(f'B_0_{name}')
    B_1     = bk.symbols(f'B_1_{name}')
    B_2     = bk.symbols(f'B_2_{name}')
    B_3     = bk.symbols(f'B_3_{name}')
    R_bat   = bk.symbols(f'R_bat_{name}')

    E_n = 1000*3600*E_kWh

    if 'socs' in data:
        # piecewise-linear OCV(soc) curve — backend-agnostic replacement for
        # the legacy SymPy interpolating_spline.
        socs = list(data['socs'])
        es   = list(data['es'])
        e_ocv = _piecewise_linear(bk, soc, socs, es)
    else:
        e_ocv = B_0 + B_1*soc + B_2*soc**2 + B_3*soc**3

    dsoc    = 1/E_n*(-i_l*e_ocv)
    dxi_soc = epsilon_soc
    g_v_bat = e_ocv - i_l*R_bat - v_bat

    grid.dae['f'] += [dsoc, dxi_soc]
    grid.dae['x'] += [soc,  xi_soc]
    grid.dae['g'] += [g_v_bat]
    grid.dae['y_ini'] += [v_bat]
    grid.dae['y_run'] += [v_bat]

    grid.dae['params_dict'].update({
        str(K_p): 1e-6, str(K_i): 1e-6,
        str(soc_min): 0.0, str(soc_max): 1.0,
        str(E_kWh): data['E_kWh'],
    })

    if 'B_0' in data:
        B_0_N = data['B_0_N']; B_1_N = data['B_1_N']
        B_2_N = data['B_2_N']; B_3_N = data['B_3_N']
    else:
        B_0_N = 600; B_1_N = 0.0; B_2_N = 0.0; B_3_N = 0.0
    grid.dae['params_dict'].update({str(B_0): B_0_N, str(B_1): B_1_N,
                                    str(B_2): B_2_N, str(B_3): B_3_N})

    R_bat_N = data.get('R_bat', 0.0)
    grid.dae['params_dict'].update({str(R_bat): R_bat_N})

    grid.dae['xy_0_dict'][f'v_bat_{name}'] = B_0_N

    grid.dae['u_ini_dict'].update({f'soc_ref_{name}': 0.5})
    grid.dae['u_run_dict'].update({f'soc_ref_{name}': 0.5})

    if 'A_loss' in data:
        A_loss_N = data['A_loss']; B_loss_N = data['B_loss']; C_loss_N = data['C_loss']
    else:
        A_loss_N = 0.0001; B_loss_N = 0.0; C_loss_N = 0.0001
    grid.dae['params_dict'].update({
        str(A_loss): A_loss_N, str(B_loss): B_loss_N, str(C_loss): C_loss_N,
        str(K_charger): 0.0, str(K_soc): 0.001,
    })

    grid.dae['xy_0_dict'].update({str(soc): 0.5, str(xi_soc): 0.5})

    # outputs
    grid.dae['h_dict'].update({
        f'i_l_{name}':   i_l,
        f'p_h_{name}':   p_h,
        f'v_bat_{name}': v_bat,
        f'e_h_{name}':   e_h,
        f'i_charger_{name}': i_charger,
    })

    grid.omega_coi_numerator   += 1.0
    grid.omega_coi_denominator += 1.0
