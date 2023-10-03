# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def vsource(grid,name,bus_name,data_dict):
    '''

    parameters
    ----------



    inputs
    ------

    theta_ref: internal voltage angle reference
    v_ref: internal voltage magnitude reference

    example
    -------

    "vsource": [{"v_ref":1.0, "theta_ref":0.0}]

    
    '''

    sin = sym.sin
    cos = sym.cos

    
    # inputs
    V = sym.Symbol(f"V_{bus_name}", real=True)
    theta = sym.Symbol(f"theta_{bus_name}", real=True)
    v_ref = sym.Symbol(f"v_ref_{name}", real=True)
    theta_ref = sym.Symbol(f"theta_ref_{name}", real=True)
    V_dummy = sym.Symbol(f"V_dummy_{name}", real=True)

    
    # dynamic states

    # algebraic states
    # V = sym.Symbol(f"V_{name}", real=True)
    # theta = sym.Symbol(f"theta_{name}", real=True)            


    # parameters
    params_list = []
    
    # auxiliar

    # dynamic equations            
    grid.dae['f'] += [V_dummy - v_ref]
    grid.dae['x'] += [V_dummy]

    # algebraic equations   
    g_V = V - v_ref
    g_theta = theta - theta_ref
    
    
    # dae 

    H = 1e6
    grid.H_total += H
    grid.omega_coi_numerator += H
    grid.omega_coi_denominator += H

    idx_V = grid.dae['y_ini'].index(V)
    idx_theta = grid.dae['y_ini'].index(theta)

    grid.dae['g'][idx_V] = g_V
    grid.dae['g'][idx_theta] = g_theta

    # grid.dae['y_ini'] += [V, theta]  
    # grid.dae['y_run'] += [V, theta]  

    grid.dae['u_ini_dict'].update({f'{str(v_ref)}':1.0})
    grid.dae['u_run_dict'].update({f'{str(v_ref)}':1.0})

    grid.dae['u_ini_dict'].update({f'{str(theta_ref)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(theta_ref)}':0.0})

    grid.dae['xy_0_dict'].update({str(V):1.0})
    grid.dae['xy_0_dict'].update({str(theta):0.0})
       
    grid.dae['h_dict'].update({f"V_dummy_{name}":V_dummy})

    # outputs


if __name__=='__main__':

    from pydae.bmapu import bmapu_builder

    data = {
    "system":{"name":"smib","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":1e-6},       
    "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
    "vscs":[{"type":"vsc_pq","bus":"1","p_in":0.5,"S_n":10e6,"K_delta":0.0}],
    "sources":[{"type":"vsource","bus":"1"}]
    }
    
    grid = bmapu_builder.bmapu(data)
    grid.checker()
    grid.verbose = True 
    grid.build('smib_vsc_pq_inf')

    import smib_vsc_pq_inf

    model = smib_vsc_pq_inf.model()
    model.ini({},'xy_0.json')
    model.report_y()

