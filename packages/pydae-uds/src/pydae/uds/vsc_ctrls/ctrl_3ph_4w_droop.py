r"""
AC/DC linear droop outer-loop control for a 4-wire AC-DC VSC.

The control couples the DC-bus per-unit voltage to the AC-bus per-unit
phase-neutral voltage through a per-phase droop gain, generating the
per-phase active-power command that drives the underlying VSC. With a zero
droop gain the controller passes the reference signals straight through.

**Per-unit voltages**

$$v_{dc}^{pu} = \frac{|v_{dc}|}{V_{dc,b}}, \qquad
  V_{phn,\varphi}^{pu} = \frac{|V_{\varphi n}|}{V_{ac,b}}$$

with $V_{ac,b} = U_{ac,b}/\sqrt{3}$ and $|v_{dc}|, |V_{\varphi n}|$ the
DC pole-to-pole and AC phase-to-neutral voltage magnitudes (already
emitted as 2-norms of the rectangular node voltages).

**Droop equation** (one per phase $\varphi \in \{a, b, c\}$):

$$p_{ac,\varphi} = K_{acdc}\, K_{acdc,\varphi}\, (v_{dc}^{pu} - V_{phn,\varphi}^{pu})
                  + p_{vsc,\varphi}^{ref}$$

**Algebraic balance**: the control turns the per-phase `p_vsc_{abc}_{bus_ac}`
variables (inputs of the host VSC) into algebraic states by adding the
equations $p_{ac,\varphi} - p_{vsc,\varphi} = 0$ to `g_list`, and pops the
corresponding entries from `u_ini_dict` / `u_run_dict`. The reference
channel `p_vsc_{abc}_ref_{bus_ac}` becomes the new input.

**HJSON snippet** (nested inside an AC-DC VSC entry):

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
        {"type": "Parameter", "tex": "K_{acdc}",       "data": "", "model": "K_acdc_{bus_ac}",    "default": 0.0, "units": "-", "description": "Common droop gain (set >0 to enable the AC/DC coupling)"},
        {"type": "Parameter", "tex": "K_{acdc,\\varphi}", "data": "", "model": "K_acdc_{ph}_{bus_ac}", "default": 1.0, "units": "-", "description": "Per-phase droop weight"},
        {"type": "Parameter", "tex": "U_{ac,b}",       "data": "", "model": "U_ac_b_{bus_ac}",    "default": "bus U_kV * 1e3", "units": "V", "description": "AC base voltage (auto-set from bus U_kV)"},
        {"type": "Parameter", "tex": "V_{dc,b}",       "data": "", "model": "V_dc_b_{bus_dc}",    "default": "bus U_kV * 1e3", "units": "V", "description": "DC base voltage"},
        {"type": "Input",     "tex": "p_{vsc,\\varphi}^{ref}", "data": "", "model": "p_vsc_{ph}_ref_{bus_ac}", "default": 0.0, "units": "W", "description": "Per-phase reference active-power injection"},
        {"type": "Algebraic State", "tex": "p_{vsc,\\varphi}", "data": "", "model": "p_vsc_{ph}_{bus_ac}", "default": "", "units": "W", "description": "Per-phase active-power injection commanded to the VSC"},
        {"type": "Output", "tex": "v_{ac,\\varphi}^{pu}", "data": "", "model": "v_ac_{ph}_pu_{bus_ac}", "default": "", "units": "pu", "description": "AC phase-neutral per-unit voltage (monitor)"},
        {"type": "Output", "tex": "v_{dc}^{pu}",       "data": "", "model": "v_dc_pu_{bus_dc}",   "default": "", "units": "pu", "description": "DC pole-to-pole per-unit voltage (monitor)"},
    ]


def ctrl_3ph_4w_droop(grid,vsc_data,ctrl_data,name,bus_name):

    bus_ac = vsc_data['bus_ac']
    bus_dc = vsc_data['bus_dc']
    bk = grid.backend

    buses_names = [bus['name'] for bus in grid.data['buses']]

    U_ac_b = bk.symbols(f'U_ac_b_{bus_ac}')
    V_dc_b = bk.symbols(f'V_dc_b_{bus_dc}')


    V_phn = []
    n2a = {0:'a',1:'b',2:'c'}
    # phase top neutral voltages:
    V_n_r = bk.symbols(f'V_{bus_ac}_{3}_r')
    V_n_i = bk.symbols(f'V_{bus_ac}_{3}_i')

    # phase-neutral voltage module
    for ph in [0,1,2]:
        V_ph_r = bk.symbols(f'V_{bus_ac}_{ph}_r')
        V_ph_i = bk.symbols(f'V_{bus_ac}_{ph}_i')
        z_value = ((V_ph_r-V_n_r)**2 + (V_ph_i-V_n_i)**2)**0.5
        V_phn += [z_value]

    # DC bus voltage magnitude (pole-to-pole)
    V_n_r = bk.symbols(f'V_{bus_dc}_{1}_r')
    V_n_i = bk.symbols(f'V_{bus_dc}_{1}_i')
    V_ph_r = bk.symbols(f'V_{bus_dc}_{0}_r')
    V_ph_i = bk.symbols(f'V_{bus_dc}_{0}_i')
    v_dc = ((V_ph_r-V_n_r)**2 + (V_ph_i-V_n_i)**2)**0.5

    K_acdc   = bk.symbols(f'K_acdc_{bus_ac}')
    K_acdc_a = bk.symbols(f'K_acdc_a_{bus_ac}')
    K_acdc_b = bk.symbols(f'K_acdc_b_{bus_ac}')
    K_acdc_c = bk.symbols(f'K_acdc_c_{bus_ac}')

    p_vsc_a = bk.symbols(f'p_vsc_a_{bus_ac}')
    p_vsc_b = bk.symbols(f'p_vsc_b_{bus_ac}')
    p_vsc_c = bk.symbols(f'p_vsc_c_{bus_ac}')
    p_vsc_a_ref = bk.symbols(f'p_vsc_a_ref_{bus_ac}')
    p_vsc_b_ref = bk.symbols(f'p_vsc_b_ref_{bus_ac}')
    p_vsc_c_ref = bk.symbols(f'p_vsc_c_ref_{bus_ac}')

    V_ac_b = U_ac_b/np.sqrt(3)

    v_dc_pu = v_dc/V_dc_b
    p_ac_a = K_acdc*K_acdc_a*(v_dc_pu - V_phn[0]/V_ac_b) + p_vsc_a_ref
    p_ac_b = K_acdc*K_acdc_b*(v_dc_pu - V_phn[1]/V_ac_b) + p_vsc_b_ref
    p_ac_c = K_acdc*K_acdc_c*(v_dc_pu - V_phn[2]/V_ac_b) + p_vsc_c_ref

    grid.dae['g'] += [p_ac_a - p_vsc_a]
    grid.dae['g'] += [p_ac_b - p_vsc_b]
    grid.dae['g'] += [p_ac_c - p_vsc_c]

    grid.dae['y_ini'] += [p_vsc_a]
    grid.dae['y_ini'] += [p_vsc_b]
    grid.dae['y_ini'] += [p_vsc_c]

    grid.dae['y_run'] += [p_vsc_a]
    grid.dae['y_run'] += [p_vsc_b]
    grid.dae['y_run'] += [p_vsc_c]

    grid.dae['params_dict'].update({f'K_acdc_{bus_ac}':0.0})
    grid.dae['params_dict'].update({f'K_acdc_a_{bus_ac}':1.0})
    grid.dae['params_dict'].update({f'K_acdc_b_{bus_ac}':1.0})
    grid.dae['params_dict'].update({f'K_acdc_c_{bus_ac}':1.0})

    idx = buses_names.index(bus_ac)
    U_ac_b = grid.data['buses'][idx]['U_kV']*1e3
    idx = buses_names.index(bus_dc)
    V_dc_b = grid.data['buses'][idx]['U_kV']*1e3

    grid.dae['params_dict'].update({f'U_ac_b_{bus_ac}':U_ac_b})
    grid.dae['params_dict'].update({f'V_dc_b_{bus_dc}':V_dc_b})

    grid.dae['u_ini_dict'].pop(f'p_vsc_a_{bus_ac}')
    grid.dae['u_ini_dict'].pop(f'p_vsc_b_{bus_ac}')
    grid.dae['u_ini_dict'].pop(f'p_vsc_c_{bus_ac}')
    grid.dae['u_run_dict'].pop(f'p_vsc_a_{bus_ac}')
    grid.dae['u_run_dict'].pop(f'p_vsc_b_{bus_ac}')
    grid.dae['u_run_dict'].pop(f'p_vsc_c_{bus_ac}')

    grid.dae['u_ini_dict'].update({f'p_vsc_a_ref_{bus_ac}':0.0})
    grid.dae['u_ini_dict'].update({f'p_vsc_b_ref_{bus_ac}':0.0})
    grid.dae['u_ini_dict'].update({f'p_vsc_c_ref_{bus_ac}':0.0})
    grid.dae['u_run_dict'].update({f'p_vsc_a_ref_{bus_ac}':0.0})
    grid.dae['u_run_dict'].update({f'p_vsc_b_ref_{bus_ac}':0.0})
    grid.dae['u_run_dict'].update({f'p_vsc_c_ref_{bus_ac}':0.0})

    grid.dae['h_dict'].update({f'v_ac_a_pu_{bus_ac}':V_phn[0]/V_ac_b})
    grid.dae['h_dict'].update({f'v_ac_b_pu_{bus_ac}':V_phn[1]/V_ac_b})
    grid.dae['h_dict'].update({f'v_ac_c_pu_{bus_ac}':V_phn[2]/V_ac_b})
    grid.dae['h_dict'].update({f'v_dc_pu_{bus_dc}':v_dc_pu})
