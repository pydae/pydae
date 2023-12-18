# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def vsc_acq_dcgf(grid,name,bus_j,bus_ac,data_dict):
    """
    # auxiliar
    
    bus_j: bus DC
    bus_ac: bus AC

    """

    # inputs from grid
    V_s_m   = sym.Symbol(f"V_{bus_ac}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_ac}", real=True)
    V_d_m   = sym.Symbol(f"V_{bus_j}", real=True)
    theta_d = sym.Symbol(f"theta_{bus_j}", real=True)  # should be zero

    # inputs
    V_t_m = sym.Symbol(f"V_t_m_{bus_j}", real=True)     
    theta_t = sym.Symbol(f"theta_t_{bus_j}", real=True) # should be zero
    q_s_ref = sym.Symbol(f"q_s_ref_{bus_ac}", real=True) # should be zero
 
    # algebraic states
    i_d_r = sym.symbols(f"i_d_r_{bus_j}", real=True)
    i_d_i = sym.symbols(f"i_d_i_{bus_j}", real=True)


    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    R_d = sym.Symbol(f"R_d_{name}", real=True)
    X_d = sym.Symbol(f"X_d_{name}", real=True)
    Losses_s,Losses_p = sym.symbols(f"Losses_s_{name},Losses_p_{name}", real=True)

    # auxiliar
    v_t = V_t_m  # DC-side internal complex voltage (should be only real)
    v_d = V_d_m*sym.exp(sym.I*theta_d) # DC-side POI complex voltage (should be only real)
    v_s = V_s_m*sym.exp(sym.I*theta_s)  # AC-side POI complex voltage
    i_d = i_d_r + sym.I*i_d_i        # DC-side current (should be only real)
    Z_d = R_d               # DC-side impedance (should be only real)
    
    KVL = v_t - i_d*Z_d - v_d  # Kirchhoff's voltage law for the DC side (bus_j)
                
    # dynamic equations            


    # algebraic equations   
    g_i_d_r =  sym.re(KVL)
    g_i_d_i = i_d_i - sym.im(v_d)/1000.0

    # dae 
    grid.dae['f'] += []
    grid.dae['x'] += []
    grid.dae['g'] += [g_i_d_r,g_i_d_i]
    grid.dae['y_ini'] += [i_d_r,i_d_i]  
    grid.dae['y_run'] += [i_d_r,i_d_i]  
    
    grid.dae['u_ini_dict'].update({f'{V_t_m}':1.0})
    grid.dae['u_ini_dict'].update({f'{q_s_ref}':0.0})

    grid.dae['u_run_dict'].update({f'{V_t_m}':1.0})
    grid.dae['u_run_dict'].update({f'{q_s_ref}':0.0})         
               
    # outputs
    grid.dae['h_dict'].update({f"i_d_{name}":i_d_r})
    grid.dae['h_dict'].update({f"V_t_m_{name}":V_t_m})
    
    # parameters            
    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 
    grid.dae['params_dict'].update({f"{R_d}":data_dict['R_d']}) 
    grid.dae['params_dict'].update({f"{X_d}":data_dict['X_d']}) 

    Losses_p = 0.002 
    Losses_s = 0.01
    if 'Losses_p' in data_dict: Losses_p = data_dict['Losses_p']
    if 'Losses_s' in data_dict: Losses_s = data_dict['Losses_s']
    grid.dae['params_dict'].update({f"{Losses_p}":Losses_p}) 
    grid.dae['params_dict'].update({f"{Losses_s}":Losses_s}) 
    

    s_d = v_d*i_d
    p_d = sym.re(s_d)
    q_d = sym.im(s_d)

    s_s_m = sym.sqrt(p_d**2 + q_s_ref**2)

    p_s = -p_d + s_s_m**2*Losses_s + Losses_p
    q_s = q_s_ref

    grid.dae['h_dict'].update({f"P_dc_{name}":p_d})


    p_W_j   = -p_d*S_n  # DC side
    q_var_j = -q_d*S_n  # DC side

    p_W_k   = -p_s*S_n   # AC side
    q_var_k = -q_s*S_n

    return p_W_j,q_var_j,p_W_k,q_var_k


def test():
    
    from pydae.bmapu import bmapu_builder
    import pytest

    grid = bmapu_builder.bmapu('vsc_acq_dcgf.hjson')
    #grid.checker()
    grid.verbose = True 
    grid.build('temp')

    import temp

    model = temp.model()

    model.ini({'P_dc':-1000e6},'xy_0.json')

    model.report_y()
    model.report_z()

    # assert model.get_value('i_s_m_2') == pytest.approx(soc_ref, rel=0.001)

def test_hvdc():
    
    from pydae.bmapu import bmapu_builder
    import pytest

    grid = bmapu_builder.bmapu('hvdc.hjson')
    #grid.checker()
    grid.verbose = True 
    grid.build('temp')

    import temp

    model = temp.model()

    model.ini({'P_WFM':1500e6},'xy_0.json')

    model.report_y()
    model.report_z()

    p_GRI = model.get_value('p_s_GRI')*model.get_value('S_n_GRI')
    P_dc = model.get_value('p_dc_WFD_WFH')*model.get_value('S_n_WFD_WFH')

    print(f'P_GRI = {p_GRI/1e6:5.3f} MW')
    print(f'P_dc = {P_dc/1e6:5.3f} MW')

if __name__=='__main__':

    #test()
    test_hvdc()