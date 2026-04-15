
import numpy as np
import sympy as sym

def dc_ideal(grid,vsc_data):
    '''
    VSC dc working in open loop as a grid former.
    
    '''

    params_dict  = grid.dae['params_dict']
    g_list = grid.dae['g'] 
    y_ini_list = grid.dae['y_ini'] 
    u_ini_dict = grid.dae['u_ini_dict']
    y_run_list = grid.dae['y_run'] 
    u_run_dict = grid.dae['u_run_dict']
    h_dict = grid.dae['h_dict']

    #vscs = [
    #    {'bus':'B1','S_n':100e3,'R':0.01,'X':0.1,'R_n':0.01,'X_n':0.1,'R_ng':0.01,'X_ng':3.0,'K_f':0.1,'T_f':1.0,'K_sec':0.5,'K_delta':0.001},
    #    ]

    #for vsc in vsc_data:
        
    name = vsc_data['bus']

    # inputs
    v_dc = sym.Symbol(f'v_dc_{name}', real=True)
    v_sp,v_sn = sym.symbols(f'V_{name}_0_r,V_{name}_1_r', real=True)
    v_spi,v_sni = sym.symbols(f'V_{name}_0_i,V_{name}_1_i', real=True)

    # parameters
    R_s,R_g = sym.symbols(f'R_{name}_s,R_{name}_g', real=True)

    # dynamical states

    
    # algebraic states
    i_sp,i_sn,v_og = sym.symbols(f'i_vsc_{name}_sp,i_vsc_{name}_sn, v_og_{name}', real=True)

    v_tp = v_dc/2.0
    v_tn = v_dc/2.0

    eq_i_sp = v_og + v_tp - R_s*i_sp - v_sp
    eq_i_sn = v_og - v_tn - R_s*i_sn - v_sn
    eq_v_og = -v_og/R_g - i_sp - i_sn 


    g_list += [eq_i_sp,eq_i_sn,eq_v_og] 

    y_ini_list += [i_sp,i_sn,v_og]
    y_run_list += [i_sp,i_sn,v_og]

    # current injections dc side
    idx_r,idx_i = grid.node2idx(f'{name}','a')
    grid.dae['g'] [idx_r] += -i_sp 
    grid.dae['g'] [idx_i] += v_spi/1e3

    idx_r,idx_i = grid.node2idx(f'{name}','b')
    grid.dae['g'] [idx_r] += -i_sn 
    grid.dae['g'] [idx_i] += v_sni/1e3

    V_1 = vsc_data['U_n']

    u_ini_dict.update({f'v_dc_{name}':V_1})
    u_run_dict.update({f'v_dc_{name}':V_1})

    params_dict.update({f'R_{name}_s':vsc_data['R_s'],f'R_{name}_g':vsc_data['R_g']})

    HS_coi  = 1.0
    omega_coi_i = 1.0

    grid.omega_coi_numerator += omega_coi_i
    grid.omega_coi_denominator += HS_coi
