# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def load_zip(grid,name,bus_name,data_dict):
    """
    # auxiliar
    
    """

    V = sym.Symbol(f"V_{bus_name}", real=True)
    S_base = sym.Symbol(f"S_base", real=True)    

    T_pz = sym.Symbol(f"T_pz_{name}", real=True)
    T_qz = sym.Symbol(f"T_qz_{name}", real=True)
    
    # inputs and algebraic states
    p_z_f = sym.Symbol(f"p_z_f_{name}", real=True)
    q_z_f = sym.Symbol(f"q_z_f_{name}", real=True)
    p_z = sym.Symbol(f"p_z_{name}", real=True)
    q_z = sym.Symbol(f"q_z_{name}", real=True)
    p_i = sym.Symbol(f"p_i_{name}", real=True)
    q_i = sym.Symbol(f"q_i_{name}", real=True)
    p_p = sym.Symbol(f"p_p_{name}", real=True)
    q_p = sym.Symbol(f"q_p_{name}", real=True)

    i_p = sym.Symbol(f"i_p_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)
    g_load = sym.Symbol(f"g_load_{name}", real=True)
    b_load = sym.Symbol(f"b_load_{name}", real=True)

    # parameters
    
    # auxiliar

    # dynamic equations            
    dp_z_f = 1.0/T_pz*(p_z - p_z_f)
    dq_z_f = 1.0/T_qz*(q_z - q_z_f)

    # algebraic equations 
    eq_p_z = -p_z + g_load*V**2
    eq_q_z = -q_z + b_load*V**2
    eq_p_i = -p_i + i_p*V
    eq_q_i = -q_i + i_q*V 


    # dae 
    grid.dae['f'] += [dp_z_f,dq_z_f]
    grid.dae['x'] += [ p_z_f, q_z_f]
    grid.dae['g'] += [eq_p_i,eq_q_i,eq_p_z,eq_q_z]
    grid.dae['y_ini'] += [i_p,i_q,g_load,b_load]  
    grid.dae['y_run'] += [p_i,q_i,p_z,q_z]  

    K_zp_N,K_zq_N = 0.0,0.0
    K_ip_N,K_iq_N = 0.0,0.0

    if 'K_zp' in data_dict: K_zp_N = data_dict['K_zp']
    if 'K_zq' in data_dict: K_zq_N = data_dict['K_zq']
    if 'K_ip' in data_dict: K_ip_N = data_dict['K_ip']
    if 'K_iq' in data_dict: K_iq_N = data_dict['K_iq']

    K_pp_N = 1.0 - K_zp_N - K_ip_N
    K_pq_N = 1.0 - K_zq_N - K_iq_N

    S_base_N = grid.system['S_base']
    p_z_0 = K_zp_N*data_dict['p_mw']*1e6/S_base_N
    q_z_0 = K_zp_N*data_dict['q_mvar']*1e6/S_base_N
    p_i_0 = K_ip_N*data_dict['p_mw']*1e6/S_base_N
    q_i_0 = K_iq_N*data_dict['q_mvar']*1e6/S_base_N
    p_p_0 = K_pp_N*data_dict['p_mw']*1e6/S_base_N
    q_p_0 = K_pq_N*data_dict['q_mvar']*1e6/S_base_N

    grid.dae['u_ini_dict'].update({f'{str(p_z)}':p_z_0})
    grid.dae['u_ini_dict'].update({f'{str(q_z)}':q_z_0})
    grid.dae['u_ini_dict'].update({f'{str(p_i)}':p_i_0})
    grid.dae['u_ini_dict'].update({f'{str(q_i)}':q_i_0})    
    grid.dae['u_ini_dict'].update({f'{str(p_p)}':p_p_0})
    grid.dae['u_ini_dict'].update({f'{str(q_p)}':q_p_0})    
  
    grid.dae['u_run_dict'].update({f'{str(g_load)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(b_load)}':0.0})            
    grid.dae['u_run_dict'].update({f'{str(i_p)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(i_q)}':0.0})     
    grid.dae['u_run_dict'].update({f'{str(p_p)}':p_p_0})
    grid.dae['u_run_dict'].update({f'{str(q_p)}':q_p_0})  

    grid.dae['xy_0_dict'].update({str(p_z_f):p_z_0})
    grid.dae['xy_0_dict'].update({str(q_z_f):q_z_0})
    grid.dae['xy_0_dict'].update({str(g_load):p_z_0})
    grid.dae['xy_0_dict'].update({str(b_load):q_z_0})
    grid.dae['xy_0_dict'].update({str(i_p):p_i_0})
    grid.dae['xy_0_dict'].update({str(i_q):q_i_0})

    grid.dae['params_dict'].update({str(T_pz):0.1})  
    grid.dae['params_dict'].update({str(T_qz):0.1})  

    p_W   = (-p_z_f - p_i - p_p)*S_base
    q_var = (-q_z_f - q_i - q_p)*S_base

    # outputs
    grid.dae['h_dict'].update({f'p_load_{name}':p_W})     
    grid.dae['h_dict'].update({f'q_load_{name}':q_var})     

    return p_W,q_var


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import pytest

    grid = bmapu('zip.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.verbose = False
    grid.build('temp')

    import temp

    model = temp.model()

    model.ini({},'xy_0.json')
    
    model.report_u()
    model.report_y()
    model.report_z()

    # assert model.get_value('V_1') == pytest.approx(v_ref_1, rel=0.001)
    # # assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)

    # model.ini({'p_m_1':0.5,'v_ref_1':1.0},'xy_0.json')
    # model.run(1.0,{})
    # model.run(15.0,{'v_ref_1':1.05})
    # model.post()

    # import matplotlib.pyplot as plt

    # fig,axes = plt.subplots()
    # axes.plot(model.Time,model.get_values('V_1'))
    # fig.savefig('ntsst1_step.svg')


if __name__ == '__main__':

    #development()
    test()