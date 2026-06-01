r"""
Bidirectional AC-DC converter, 3 phase + neutral on the AC side, two poles
on the DC side, controlled in **DC-bus voltage / per-phase reactive power**.
Used as a grid-forming converter on the DC link: it imposes the DC-pole
voltage and absorbs/injects whatever AC power balances it.

**AC side** — per-phase complex power on the converter terminals
($s_\varphi = (v_\varphi - v_n)\overline{i_\varphi}$, plus the neutral
$s_n = v_n \overline{i_n}$), expanded into real form to be backend-agnostic.

**Loss model** — per-wire conduction-loss polynomial in the rms current
(`A + B|i| + C|i|^2`), with a small smoothing constant inside the square
root so the Jacobian is well-defined at zero current:

$$|i_\varphi|_{rms} = \sqrt{(i_\varphi^r)^2 + (i_\varphi^i)^2 + 10^{-3}}$$
$$p_{loss,\varphi} = A_{loss} + B_{loss}|i|_{rms} + C_{loss}|i|^2_{rms}$$

**AC balance** — for each phase $\varphi$ the active power balance pulls
the DC active power $p_{dc}$ (split across phases by weights $C_\varphi$,
defaulting to 1/3 each) and adds the local + neutral conduction losses:

$$\Re(s_\varphi) = C_\varphi p_{dc} - p_{loss,\varphi} - p_{loss,n}/3$$
$$\Im(s_\varphi) = q_\varphi^{ref}$$

with $\sum_\varphi i_\varphi + i_n = 0$ closing the neutral KCL.

**DC side** — DC droop on the pole-to-pole voltage:

$$v_{dc} = v_{dc}^{ref} - K_{droop}\, p_{dc}$$

and a Thevenin-like coupling to the DC bus pole voltages through small
series resistances and a neutral-to-ground impedance $R_{gdc}$:

$$0 = v_{og} + v_{dc}/2 - R_{dc} i_{pos} - v_{pos}$$
$$0 = v_{og} - v_{dc}/2 - R_{dc} i_{neg} - v_{neg}$$
$$0 = -v_{og}/R_{gdc} - i_{pos} - i_{neg}$$
$$0 = -p_{dc} - v_{dc}/2 \cdot (i_{pos} - i_{neg})$$

**HJSON snippet**

```hjson
vscs: [
    {bus_ac: "A2", bus_dc: "D2", type: "acdc_3ph_4w_vdc_q",
     A: 0.0, B: 0.0, C: 0.0}
]
```
"""

import numpy as np


def descriptions():
    return [
        {"type": "Parameter", "tex": "A_{loss}",    "data": "A",   "model": "A_{bus_ac}",         "default": "",  "units": "W",   "description": "No-load loss"},
        {"type": "Parameter", "tex": "B_{loss}",    "data": "B",   "model": "B_{bus_ac}",         "default": "",  "units": "V",   "description": "Linear-loss coefficient"},
        {"type": "Parameter", "tex": "C_{loss}",    "data": "C",   "model": "C_{bus_ac}",         "default": "",  "units": r"\Omega", "description": "Quadratic-loss coefficient"},
        {"type": "Parameter", "tex": "C_\\varphi",  "data": "",    "model": "C_{ph}_{bus_ac}",    "default": 1/3, "units": "-",   "description": "DC-power share to phase ph (a/b/c)"},
        {"type": "Parameter", "tex": "K_{droop}",   "data": "",    "model": "K_droop_{bus_ac}",   "default": 0.0, "units": r"\Omega", "description": "DC voltage / power droop gain"},
        {"type": "Parameter", "tex": "R_{dc}",      "data": "",    "model": "R_dc_{bus_dc}",      "default": 1e-6, "units": r"\Omega", "description": "DC series resistance per pole"},
        {"type": "Parameter", "tex": "R_{gdc}",     "data": "",    "model": "R_gdc_{bus_dc}",     "default": 3.0, "units": r"\Omega", "description": "DC neutral-to-ground resistance"},
        {"type": "Input",     "tex": "v_{dc}^{ref}","data": "",    "model": "v_dc_{bus_dc}_ref",  "default": 800.0, "units": "V", "description": "DC voltage setpoint"},
        {"type": "Input",     "tex": "q_\\varphi^{ref}", "data": "", "model": "q_vsc_{ph}_{bus_ac}", "default": 0.0, "units": "var", "description": "Per-phase reactive-power setpoint"},
        {"type": "Algebraic State", "tex": r"i_\varphi^{r,i}", "data": "", "model": "i_vsc_{bus_ac}_{ph}_{r,i}", "default": "", "units": "A", "description": "Per-phase converter current (a/b/c/n)"},
        {"type": "Algebraic State", "tex": "p_{dc}",  "data": "", "model": "p_vsc_{bus_dc}",  "default": "", "units": "W", "description": "DC-side active power"},
        {"type": "Algebraic State", "tex": "i_{pos}", "data": "", "model": "i_vsc_pos_{bus_dc}_sp", "default": "", "units": "A", "description": "DC positive-pole current"},
        {"type": "Algebraic State", "tex": "i_{neg}", "data": "", "model": "i_vsc_{bus_dc}_sn", "default": "", "units": "A", "description": "DC negative-pole current"},
        {"type": "Algebraic State", "tex": "v_{og}",  "data": "", "model": "v_og_{bus_dc}",   "default": "", "units": "V", "description": "DC neutral-to-ground voltage"},
        {"type": "Output", "tex": "p_{vsc}",       "data": "", "model": "p_vsc_{bus_ac}",     "default": "", "units": "W", "description": "Total AC active power"},
        {"type": "Output", "tex": "p_{vsc}^{loss}","data": "", "model": "p_vsc_loss_{bus_ac}","default": "", "units": "W", "description": "Total conduction loss"},
    ]


def acdc_3ph_4w_vdc_q(grid, vsc_data):
    '''
    AC-DC converter, 3 phase 4 wire, v_dc and q_ac controlled.
    Dual-backend (SymPy / CasADi) via grid.backend; nodal algebra in real form.
    '''
    bus_ac_name = vsc_data['bus_ac']
    bus_dc_name = vsc_data['bus_dc']
    name = bus_ac_name
    bk = grid.backend

    A_value = vsc_data['A']
    B_value = vsc_data['B']
    C_value = vsc_data['C']

    # reactive setpoints per phase (inputs)
    q_a = bk.symbols(f'q_vsc_a_{bus_ac_name}')
    q_b = bk.symbols(f'q_vsc_b_{bus_ac_name}')
    q_c = bk.symbols(f'q_vsc_c_{bus_ac_name}')

    p_dc      = bk.symbols(f'p_vsc_{bus_dc_name}')
    p_a_d     = bk.symbols(f'p_a_d_{bus_ac_name}')
    p_b_d     = bk.symbols(f'p_b_d_{bus_ac_name}')
    p_c_d     = bk.symbols(f'p_c_d_{bus_ac_name}')
    p_n_d     = bk.symbols(f'p_n_d_{bus_ac_name}')
    C_a       = bk.symbols(f'C_a_{bus_ac_name}')
    C_b       = bk.symbols(f'C_b_{bus_ac_name}')
    C_c       = bk.symbols(f'C_c_{bus_ac_name}')
    K_droop   = bk.symbols(f'K_droop_{name}')

    # AC node voltages (rectangular)
    v_a_r = bk.symbols(f'V_{bus_ac_name}_0_r'); v_a_i = bk.symbols(f'V_{bus_ac_name}_0_i')
    v_b_r = bk.symbols(f'V_{bus_ac_name}_1_r'); v_b_i = bk.symbols(f'V_{bus_ac_name}_1_i')
    v_c_r = bk.symbols(f'V_{bus_ac_name}_2_r'); v_c_i = bk.symbols(f'V_{bus_ac_name}_2_i')
    v_n_r = bk.symbols(f'V_{bus_ac_name}_3_r'); v_n_i = bk.symbols(f'V_{bus_ac_name}_3_i')

    # AC currents (algebraic, rectangular)
    i_a_r = bk.symbols(f'i_vsc_{bus_ac_name}_a_r'); i_a_i = bk.symbols(f'i_vsc_{bus_ac_name}_a_i')
    i_b_r = bk.symbols(f'i_vsc_{bus_ac_name}_b_r'); i_b_i = bk.symbols(f'i_vsc_{bus_ac_name}_b_i')
    i_c_r = bk.symbols(f'i_vsc_{bus_ac_name}_c_r'); i_c_i = bk.symbols(f'i_vsc_{bus_ac_name}_c_i')
    i_n_r = bk.symbols(f'i_vsc_{bus_ac_name}_n_r'); i_n_i = bk.symbols(f'i_vsc_{bus_ac_name}_n_i')

    # DC node voltages and dc algebraic states
    v_pos = bk.symbols(f'V_{bus_dc_name}_0_r')
    v_neg = bk.symbols(f'V_{bus_dc_name}_1_r')
    v_posi = bk.symbols(f'V_{bus_dc_name}_0_i')
    v_negi = bk.symbols(f'V_{bus_dc_name}_1_i')

    i_pos = bk.symbols(f'i_vsc_pos_{bus_dc_name}_sp')
    i_neg = bk.symbols(f'i_vsc_{bus_dc_name}_sn')
    v_og  = bk.symbols(f'v_og_{bus_dc_name}')

    v_dc_ref = bk.symbols(f'v_dc_{bus_dc_name}_ref')
    v_dc = v_dc_ref - K_droop*p_dc

    A_loss = bk.symbols(f'A_{bus_ac_name}')
    B_loss = bk.symbols(f'B_{bus_ac_name}')
    C_loss = bk.symbols(f'C_{bus_ac_name}')
    R_dc   = bk.symbols(f'R_dc_{bus_dc_name}')
    R_gdc  = bk.symbols(f'R_gdc_{bus_dc_name}')

    # per-phase RMS currents (with smoothing) and conduction losses
    i_a_rms = bk.sqrt(i_a_r**2 + i_a_i**2 + 0.001)
    i_b_rms = bk.sqrt(i_b_r**2 + i_b_i**2 + 0.001)
    i_c_rms = bk.sqrt(i_c_r**2 + i_c_i**2 + 0.001)
    i_n_rms = bk.sqrt(i_n_r**2 + i_n_i**2 + 0.001)

    p_loss_a = A_loss + B_loss*i_a_rms + C_loss*i_a_rms*i_a_rms
    p_loss_b = A_loss + B_loss*i_b_rms + C_loss*i_b_rms*i_b_rms
    p_loss_c = A_loss + B_loss*i_c_rms + C_loss*i_c_rms*i_c_rms
    p_loss_n = A_loss + B_loss*i_n_rms + C_loss*i_n_rms*i_n_rms

    # phase apparent powers in real form: s = (v - v_n) * conj(i)
    #   re(s_x) = (v_x_r - v_n_r)*i_x_r + (v_x_i - v_n_i)*i_x_i
    #   im(s_x) = (v_x_i - v_n_i)*i_x_r - (v_x_r - v_n_r)*i_x_i
    def _pq(v_r, v_i, vn_r, vn_i, ir, ii):
        re = (v_r - vn_r)*ir + (v_i - vn_i)*ii
        im = (v_i - vn_i)*ir - (v_r - vn_r)*ii
        return re, im

    re_s_a, im_s_a = _pq(v_a_r, v_a_i, v_n_r, v_n_i, i_a_r, i_a_i)
    re_s_b, im_s_b = _pq(v_b_r, v_b_i, v_n_r, v_n_i, i_b_r, i_b_i)
    re_s_c, im_s_c = _pq(v_c_r, v_c_i, v_n_r, v_n_i, i_c_r, i_c_i)
    # neutral: s_n = v_n * conj(i_n)
    re_s_n = v_n_r*i_n_r + v_n_i*i_n_i
    im_s_n = v_n_i*i_n_r - v_n_r*i_n_i

    eq_p_a_d =  C_a*p_dc - p_a_d
    eq_p_b_d =  C_b*p_dc - p_b_d
    eq_p_c_d =  C_c*p_dc - p_c_d
    eq_p_n_d =  re_s_n - p_n_d

    eq_i_a_r =  re_s_a - p_a_d + p_loss_a + p_loss_n/3
    eq_i_b_r =  re_s_b - p_b_d + p_loss_b + p_loss_n/3
    eq_i_c_r =  re_s_c - p_c_d + p_loss_c + p_loss_n/3
    eq_i_a_i =  im_s_a - q_a
    eq_i_b_i =  im_s_b - q_b
    eq_i_c_i =  im_s_c - q_c

    eq_i_n_r = i_n_r + i_a_r + i_b_r + i_c_r
    eq_i_n_i = i_n_i + i_a_i + i_b_i + i_c_i

    eq_p_dc = -p_dc - (v_dc/2*i_pos - v_dc/2*i_neg)

    v_tp = v_dc/2.0
    v_tn = v_dc/2.0
    eq_i_pos = v_og + v_tp - R_dc*i_pos - v_pos
    eq_i_neg = v_og - v_tn - R_dc*i_neg - v_neg
    eq_v_og  = -v_og/R_gdc - i_pos - i_neg

    grid.dae['g'] += [eq_p_a_d, eq_p_b_d, eq_p_c_d, eq_p_n_d,
                      eq_i_a_r, eq_i_a_i, eq_i_b_r, eq_i_b_i,
                      eq_i_c_r, eq_i_c_i, eq_i_n_r, eq_i_n_i,
                      eq_i_pos, eq_i_neg, eq_v_og, eq_p_dc]
    ys = [p_a_d, p_b_d, p_c_d, p_n_d,
          i_a_r, i_a_i, i_b_r, i_b_i,
          i_c_r, i_c_i, i_n_r, i_n_i,
          i_pos, i_neg, v_og, p_dc]
    grid.dae['y_ini'] += ys
    grid.dae['y_run'] += ys

    # AC-side current injections + magnitudes
    for ph, ir, ii in [('a', i_a_r, i_a_i), ('b', i_b_r, i_b_i),
                       ('c', i_c_r, i_c_i), ('n', i_n_r, i_n_i)]:
        idx_r, idx_i = grid.node2idx(bus_ac_name, ph)
        grid.dae['g'][idx_r] += -ir
        grid.dae['g'][idx_i] += -ii
        grid.dae['h_dict'][f'i_vsc_{bus_ac_name}_{ph}_m'] = (ir**2 + ii**2)**0.5

    # DC-side current injections (DC bus uses node2idx 'a','b' → indices 0,1)
    idx_r, idx_i = grid.node2idx(bus_dc_name, 'a')
    grid.dae['g'][idx_r] += -i_pos
    grid.dae['g'][idx_i] += v_posi/1e3
    idx_r, idx_i = grid.node2idx(bus_dc_name, 'b')
    grid.dae['g'][idx_r] += -i_neg
    grid.dae['g'][idx_i] += v_negi/1e3

    grid.dae['u_ini_dict'].update({f'v_dc_{bus_dc_name}_ref': 800.0,
                                   str(q_a): 0.0, str(q_b): 0.0, str(q_c): 0.0})
    grid.dae['u_run_dict'].update({f'v_dc_{bus_dc_name}_ref': 800.0,
                                   str(q_a): 0.0, str(q_b): 0.0, str(q_c): 0.0})

    grid.dae['params_dict'].update({f'A_{bus_ac_name}': A_value,
                                    f'B_{bus_ac_name}': B_value,
                                    f'C_{bus_ac_name}': C_value})
    grid.dae['params_dict'].update({f'C_a_{bus_ac_name}': 1/3,
                                    f'C_b_{bus_ac_name}': 1/3,
                                    f'C_c_{bus_ac_name}': 1/3})
    grid.dae['params_dict'].update({f'R_dc_{bus_dc_name}': 1e-6})
    grid.dae['params_dict'].update({f'K_dc_{bus_dc_name}': 1e-6})
    grid.dae['params_dict'].update({f'R_gdc_{bus_dc_name}': 3.0})
    grid.dae['params_dict'].update({str(K_droop): 0.0})

    grid.dae['xy_0_dict'].update({f'v_{bus_dc_name}_a_r': 800.0,
                                  f'v_{bus_dc_name}_n_r': 1.0})

    grid.dae['h_dict'][f'p_vsc_{bus_ac_name}'] = re_s_a + re_s_b + re_s_c + re_s_n
    grid.dae['h_dict'][f'p_vsc_loss_{bus_ac_name}'] = p_loss_a + p_loss_b + p_loss_c + p_loss_n
