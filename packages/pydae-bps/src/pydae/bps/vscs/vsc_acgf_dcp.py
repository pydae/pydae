# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def vsc_acgf_dcp(grid,name,bus_dc,bus_ac,data_dict):
    """
    # auxiliar
    
    """

    # inputs from grid
    V_s_m   = sym.Symbol(f"V_{bus_ac}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_ac}", real=True)

    # inputs
    V_t_m = sym.Symbol(f"V_t_m_{bus_ac}", real=True)
    theta_t = sym.Symbol(f"theta_t_{bus_ac}", real=True)
 
    # algebraic states
    i_s_r,i_s_i = sym.symbols(f"i_s_r_{bus_ac},i_s_i_{bus_ac}", real=True)


    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)
    X_s = sym.Symbol(f"X_s_{name}", real=True)
    Losses_s,Losses_p = sym.symbols(f"Losses_s_{name},Losses_p_{name}", real=True)

    # auxiliar
    v_t = V_t_m*sym.exp(sym.I*theta_t)
    v_s = V_s_m*sym.exp(sym.I*theta_s)
    i_s = i_s_r + sym.I*i_s_i
    Z_s = R_s + sym.I*X_s
    
    KVL = v_t - i_s*Z_s - v_s  # Kirchhoff's voltage law for the AC side (bus_ac)
                
    # dynamic equations            


    # algebraic equations   
    g_i_s_r = sym.re(KVL)
    g_i_s_i = sym.im(KVL)

    # dae 
    grid.dae['f'] += []
    grid.dae['x'] += []
    grid.dae['g'] += [g_i_s_r,g_i_s_i]
    grid.dae['y_ini'] += [i_s_r,i_s_i]  
    grid.dae['y_run'] += [i_s_r,i_s_i]  
    
    grid.dae['u_ini_dict'].update({f'{V_t_m}':1.0})
    grid.dae['u_ini_dict'].update({f'{theta_t}':0.0})
    grid.dae['u_run_dict'].update({f'{V_t_m}':1.0})
    grid.dae['u_run_dict'].update({f'{theta_t}':0.0})         

    # parameters            
    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 
    grid.dae['params_dict'].update({f"{R_s}":data_dict['R_s']}) 
    grid.dae['params_dict'].update({f"{X_s}":data_dict['X_s']}) 

    Losses_p = 0.002 
    Losses_s = 0.01
    if 'Losses_p' in data_dict: Losses_p = data_dict['Losses_p']
    if 'Losses_s' in data_dict: Losses_s = data_dict['Losses_s']
    grid.dae['params_dict'].update({f"{Losses_p}":Losses_p}) 
    grid.dae['params_dict'].update({f"{Losses_s}":Losses_s}) 

    s_s = v_s*sym.conjugate(i_s)
    p_s = sym.re(s_s)
    q_s = sym.im(s_s)

    p_dc = -p_s + sym.Abs(s_s)**2*Losses_s + Losses_p

               
    # outputs
    grid.dae['h_dict'].update({f"i_s_m_{name}":sym.Abs(i_s_r + sym.I*i_s_i)})
    grid.dae['h_dict'].update({f"V_t_m_{name}":V_t_m})
    grid.dae['h_dict'].update({f"p_dc_{name}":p_dc})
    

    p_W_j   =  -p_dc*S_n
    q_var_j =  0.0

    p_W_k   = -p_s*S_n
    q_var_k = -q_s*S_n

    return p_W_j,q_var_j,p_W_k,q_var_k


def test():
    
    from pydae.bmapu import bmapu_builder
    import pytest

    grid = bmapu_builder.bmapu('vsc_acgf_dcp.hjson')
    #grid.checker()
    grid.verbose = True 
    grid.build('temp')

    import temp

    model = temp.model()

    model.ini({'P_ac':-1000e6},'xy_0.json')

    model.report_y()
    model.report_z()

    # assert model.get_value('i_s_m_2') == pytest.approx(soc_ref, rel=0.001)



if __name__=='__main__':

    test()