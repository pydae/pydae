r"""
Bidirectional AC-DC converter, 3 phase + neutral on the AC side, two poles
on the DC side, controlled in **per-phase active and reactive power**.

Unlike `acdc_3ph_4w_vdc_q` this model does **not** impose a DC voltage; it
tracks per-phase $p_{ac,\varphi}$, $q_{ac,\varphi}$ setpoints and the DC
side balances passively. The model pairs naturally with the
`ctrl_3ph_4w_droop` outer-loop control which sets the $p$ references from
an AC/DC voltage difference.

**AC balance** (one $p$ and one $q$ equation per phase, in real form):

$$\Re(s_\varphi) = p_\varphi^{ref}, \qquad \Im(s_\varphi) = q_\varphi^{ref}$$

with $s_\varphi = (v_\varphi - v_n)\overline{i_\varphi}$ and $\sum_\varphi i_\varphi + i_n = 0$.

**Loss model** — same conduction-loss polynomial as `acdc_3ph_4w_vdc_q`
(see its docstring), with a slightly larger smoothing constant
$|i|^2 \to |i|^2 + 10^{-2}$ to stabilise during start-up.

**DC balance**:

$$0 = -p_{dc} + p_{ac,total} + p_{loss,total}$$
$$i_{d} = \frac{p_{dc}}{v_{dc}} \quad \text{(injected into the DC poles)}$$

**HJSON snippet**

```hjson
vscs: [
    {bus_ac: "A4", bus_dc: "D4", type: "acdc_3ph_4w_pq",
     A: 350, B: 0, C: 0.03,
     vsc_ctrl: {type: "ctrl_3ph_4w_droop"}}
]
```
"""

import numpy as np


def descriptions():
    return [
        {"type": "Parameter", "tex": "A_{loss}",  "data": "A", "model": "A_loss_{bus_ac}",  "default": "",  "units": "W",       "description": "No-load loss"},
        {"type": "Parameter", "tex": "B_{loss}",  "data": "B", "model": "B_loss_{bus_ac}",  "default": "",  "units": "V",       "description": "Linear-loss coefficient"},
        {"type": "Parameter", "tex": "C_{loss}",  "data": "C", "model": "C_loss_{bus_ac}",  "default": "",  "units": r"\Omega", "description": "Quadratic-loss coefficient"},
        {"type": "Parameter", "tex": "R_{dc}",    "data": "",  "model": "R_dc_{bus_dc}",    "default": 1e-6, "units": r"\Omega", "description": "DC series resistance"},
        {"type": "Parameter", "tex": "R_{gdc}",   "data": "",  "model": "R_gdc_{bus_dc}",   "default": 3.0, "units": r"\Omega", "description": "DC neutral-to-ground resistance"},
        {"type": "Input",     "tex": "p_\\varphi^{ref}", "data": "", "model": "p_vsc_{ph}_{bus_ac}", "default": 0.0, "units": "W",   "description": "Per-phase active-power setpoint (replaced when a vsc_ctrl is attached)"},
        {"type": "Input",     "tex": "q_\\varphi^{ref}", "data": "", "model": "q_vsc_{ph}_{bus_ac}", "default": 0.0, "units": "var", "description": "Per-phase reactive-power setpoint"},
        {"type": "Algebraic State", "tex": r"i_\varphi^{r,i}", "data": "", "model": "i_vsc_{bus_ac}_{ph}_{r,i}", "default": "", "units": "A", "description": "Per-phase converter current (a/b/c/n)"},
        {"type": "Algebraic State", "tex": "p_{dc}", "data": "", "model": "p_vsc_{bus_dc}", "default": "", "units": "W", "description": "DC-side active power"},
        {"type": "Output", "tex": "p_{vsc}",       "data": "", "model": "p_vsc_{bus_ac}",      "default": "", "units": "W",   "description": "Total AC active power"},
        {"type": "Output", "tex": "q_{vsc}",       "data": "", "model": "q_vsc_{bus_ac}",      "default": "", "units": "var", "description": "Total AC reactive power"},
        {"type": "Output", "tex": "|s_{vsc}|",     "data": "", "model": "s_vsc_{bus_ac}",      "default": "", "units": "VA",  "description": "Total AC apparent power magnitude"},
        {"type": "Output", "tex": "v_{dc}",        "data": "", "model": "v_dc_{bus_dc}",       "default": "", "units": "V",   "description": "DC pole-to-pole voltage"},
        {"type": "Output", "tex": "|v_{an}|",      "data": "", "model": "v_anm_{bus_ac}",      "default": "", "units": "V",   "description": "Phase-a to neutral voltage magnitude"},
    ]


def acdc_3ph_4w_pq(grid, vsc_data):
    '''
    AC-DC converter, 3 phase 4 wire, per-phase p_ac and q_ac controlled.
    Dual-backend (SymPy / CasADi) via grid.backend; real-form nodal algebra.
    '''
    bus_ac_name = vsc_data['bus_ac']
    bus_dc_name = vsc_data['bus_dc']
    bk = grid.backend

    a_value = vsc_data['A']
    b_value = vsc_data['B']
    c_value = vsc_data['C']

    # per-phase p, q setpoints (may be replaced by a vsc_ctrl)
    p_a = bk.symbols(f'p_vsc_a_{bus_ac_name}')
    p_b = bk.symbols(f'p_vsc_b_{bus_ac_name}')
    p_c = bk.symbols(f'p_vsc_c_{bus_ac_name}')
    q_a = bk.symbols(f'q_vsc_a_{bus_ac_name}')
    q_b = bk.symbols(f'q_vsc_b_{bus_ac_name}')
    q_c = bk.symbols(f'q_vsc_c_{bus_ac_name}')

    p_dc = bk.symbols(f'p_vsc_{bus_dc_name}')

    # AC node voltages (rectangular)
    v_a_r = bk.symbols(f'V_{bus_ac_name}_0_r'); v_a_i = bk.symbols(f'V_{bus_ac_name}_0_i')
    v_b_r = bk.symbols(f'V_{bus_ac_name}_1_r'); v_b_i = bk.symbols(f'V_{bus_ac_name}_1_i')
    v_c_r = bk.symbols(f'V_{bus_ac_name}_2_r'); v_c_i = bk.symbols(f'V_{bus_ac_name}_2_i')
    v_n_r = bk.symbols(f'V_{bus_ac_name}_3_r'); v_n_i = bk.symbols(f'V_{bus_ac_name}_3_i')

    # AC algebraic currents (rectangular)
    i_a_r = bk.symbols(f'i_vsc_{bus_ac_name}_a_r'); i_a_i = bk.symbols(f'i_vsc_{bus_ac_name}_a_i')
    i_b_r = bk.symbols(f'i_vsc_{bus_ac_name}_b_r'); i_b_i = bk.symbols(f'i_vsc_{bus_ac_name}_b_i')
    i_c_r = bk.symbols(f'i_vsc_{bus_ac_name}_c_r'); i_c_i = bk.symbols(f'i_vsc_{bus_ac_name}_c_i')
    i_n_r = bk.symbols(f'i_vsc_{bus_ac_name}_n_r'); i_n_i = bk.symbols(f'i_vsc_{bus_ac_name}_n_i')

    # DC node voltages
    v_pos = bk.symbols(f'V_{bus_dc_name}_0_r')
    v_neg = bk.symbols(f'V_{bus_dc_name}_1_r')

    A_loss = bk.symbols(f'A_loss_{bus_ac_name}')
    B_loss = bk.symbols(f'B_loss_{bus_ac_name}')
    C_loss = bk.symbols(f'C_loss_{bus_ac_name}')

    # per-phase RMS currents (with smoothing)
    i_a_rms = bk.sqrt(i_a_r**2 + i_a_i**2 + 0.01)
    i_b_rms = bk.sqrt(i_b_r**2 + i_b_i**2 + 0.01)
    i_c_rms = bk.sqrt(i_c_r**2 + i_c_i**2 + 0.01)
    i_n_rms = bk.sqrt(i_n_r**2 + i_n_i**2 + 0.01)

    p_loss_a = A_loss + B_loss*i_a_rms + C_loss*i_a_rms*i_a_rms
    p_loss_b = A_loss + B_loss*i_b_rms + C_loss*i_b_rms*i_b_rms
    p_loss_c = A_loss + B_loss*i_c_rms + C_loss*i_c_rms*i_c_rms
    p_loss_n = A_loss + B_loss*i_n_rms + C_loss*i_n_rms*i_n_rms

    # phase apparent powers in real form: s = (v - v_n) * conj(i)
    def _pq(v_r, v_i, vn_r, vn_i, ir, ii):
        re = (v_r - vn_r)*ir + (v_i - vn_i)*ii
        im = (v_i - vn_i)*ir - (v_r - vn_r)*ii
        return re, im

    re_s_a, im_s_a = _pq(v_a_r, v_a_i, v_n_r, v_n_i, i_a_r, i_a_i)
    re_s_b, im_s_b = _pq(v_b_r, v_b_i, v_n_r, v_n_i, i_b_r, i_b_i)
    re_s_c, im_s_c = _pq(v_c_r, v_c_i, v_n_r, v_n_i, i_c_r, i_c_i)
    re_s_n = v_n_r*i_n_r + v_n_i*i_n_i
    im_s_n = v_n_i*i_n_r - v_n_r*i_n_i

    eq_i_a_r = re_s_a - p_a
    eq_i_b_r = re_s_b - p_b
    eq_i_c_r = re_s_c - p_c
    eq_i_a_i = im_s_a - q_a
    eq_i_b_i = im_s_b - q_b
    eq_i_c_i = im_s_c - q_c

    eq_i_n_r = i_n_r + i_a_r + i_b_r + i_c_r
    eq_i_n_i = i_n_i + i_a_i + i_b_i + i_c_i

    v_dc = v_pos - v_neg
    p_ac_total = p_a + p_b + p_c
    p_loss_total = p_loss_a + p_loss_b + p_loss_c + p_loss_n
    eq_p_dc = -p_dc + p_ac_total + p_loss_total
    i_d = p_dc/v_dc

    grid.dae['g'] += [eq_i_a_r, eq_i_a_i, eq_i_b_r, eq_i_b_i,
                      eq_i_c_r, eq_i_c_i, eq_i_n_r, eq_i_n_i,
                      eq_p_dc]
    ys = [i_a_r, i_a_i, i_b_r, i_b_i, i_c_r, i_c_i, i_n_r, i_n_i, p_dc]
    grid.dae['y_ini'] += ys
    grid.dae['y_run'] += ys

    # AC-side current injections + magnitudes
    for ph, ir, ii in [('a', i_a_r, i_a_i), ('b', i_b_r, i_b_i),
                       ('c', i_c_r, i_c_i), ('n', i_n_r, i_n_i)]:
        idx_r, idx_i = grid.node2idx(bus_ac_name, ph)
        grid.dae['g'][idx_r] += -ir
        grid.dae['g'][idx_i] += -ii
        grid.dae['h_dict'][f'i_vsc_{bus_ac_name}_{ph}_m'] = (ir**2 + ii**2)**0.5

    # DC-side current injections
    idx_r, _ = grid.node2idx(bus_dc_name, 'a')
    grid.dae['g'][idx_r] += i_d
    idx_r, _ = grid.node2idx(bus_dc_name, 'b')
    grid.dae['g'][idx_r] += -i_d

    grid.dae['u_ini_dict'].update({
        f'v_dc_{bus_dc_name}_ref': 800.0,
        str(p_a): 0.0, str(p_b): 0.0, str(p_c): 0.0,
        str(q_a): 0.0, str(q_b): 0.0, str(q_c): 0.0,
    })
    grid.dae['u_run_dict'].update({
        f'v_dc_{bus_dc_name}_ref': 800.0,
        str(p_a): 0.0, str(p_b): 0.0, str(p_c): 0.0,
        str(q_a): 0.0, str(q_b): 0.0, str(q_c): 0.0,
    })

    grid.dae['params_dict'].update({
        f'A_loss_{bus_ac_name}': a_value,
        f'B_loss_{bus_ac_name}': b_value,
        f'C_loss_{bus_ac_name}': c_value,
        f'R_dc_{bus_dc_name}': 1e-6,
        f'K_dc_{bus_dc_name}': 1e-6,
        f'R_gdc_{bus_dc_name}': 3.0,
    })

    # outputs: total apparent power, dc voltage, phase-to-neutral magnitudes
    p_vsc = re_s_a + re_s_b + re_s_c + re_s_n
    q_vsc = im_s_a + im_s_b + im_s_c + im_s_n
    grid.dae['h_dict'][f'p_vsc_{bus_ac_name}'] = p_vsc
    grid.dae['h_dict'][f'q_vsc_{bus_ac_name}'] = q_vsc
    grid.dae['h_dict'][f's_vsc_{bus_ac_name}'] = (p_vsc**2 + q_vsc**2)**0.5
    grid.dae['h_dict'][f'p_vsc_loss_{bus_ac_name}'] = p_loss_total
    grid.dae['h_dict'][f'v_dc_{bus_dc_name}'] = v_dc
    grid.dae['h_dict'][f'v_anm_{bus_ac_name}'] = ((v_a_r - v_n_r)**2 + (v_a_i - v_n_i)**2)**0.5
