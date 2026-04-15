import numpy as np
import sympy as sym

def ctrl_3ph_4w_droop(grid,vsc_data,ctrl_data,name,bus_name):

    bus_ac = vsc_data['bus_ac']
    bus_dc = vsc_data['bus_dc']

    buses_names = [bus['name'] for bus in grid.data['buses']]

    U_ac_b = sym.Symbol(f'U_ac_b_{bus_ac}', real = True)
    V_dc_b = sym.Symbol(f'V_dc_b_{bus_dc}', real = True)


    V_phn = []
    n2a = {0:'a',1:'b',2:'c'}
    # phase top neutral voltages:
    V_n_r = sym.Symbol(f'V_{bus_ac}_{3}_r', real = True)
    V_n_i = sym.Symbol(f'V_{bus_ac}_{3}_i', real = True)

    # phase-neutral voltage module
    for ph in [0,1,2]:
        V_ph_r = sym.Symbol(f'V_{bus_ac}_{ph}_r', real = True)
        V_ph_i = sym.Symbol(f'V_{bus_ac}_{ph}_i', real = True)
        z_name = f'V_{bus_ac}_{n2a[ph]}n'
        z_value = ((V_ph_r-V_n_r)**2 + (V_ph_i-V_n_i)**2)**0.5
        V_phn += [z_value]

    # phase-neutral voltage module
    V_n_r = sym.Symbol(f'V_{bus_dc}_{1}_r', real = True)
    V_n_i = sym.Symbol(f'V_{bus_dc}_{1}_i', real = True)
    V_ph_r = sym.Symbol(f'V_{bus_dc}_{0}_r', real = True)
    V_ph_i = sym.Symbol(f'V_{bus_dc}_{0}_i', real = True)
    v_dc = ((V_ph_r-V_n_r)**2 + (V_ph_i-V_n_i)**2)**0.5

    K_acdc = sym.Symbol(f'K_acdc_{bus_ac}', real=True)
    K_acdc_a = sym.Symbol(f'K_acdc_a_{bus_ac}', real=True)
    K_acdc_b = sym.Symbol(f'K_acdc_b_{bus_ac}', real=True)
    K_acdc_c = sym.Symbol(f'K_acdc_c_{bus_ac}', real=True)

    p_vsc_a,p_vsc_b,p_vsc_c = sym.symbols(f'p_vsc_a_{bus_ac},p_vsc_b_{bus_ac},p_vsc_c_{bus_ac}',real=True)
    p_vsc_a_ref,p_vsc_b_ref,p_vsc_c_ref = sym.symbols(f'p_vsc_a_ref_{bus_ac},p_vsc_b_ref_{bus_ac},p_vsc_c_ref_{bus_ac}',real=True)

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
