# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def bess_pq(grid,name,bus_name,data_dict):
    """
    # auxiliar
    
    """

    # inputs
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)
    sigma_ref = sym.Symbol(f"sigma_ref_{name}", real=True)
    p_s_ref = sym.Symbol(f"p_s_ref_{name}", real=True)
    q_s_ref = sym.Symbol(f"q_s_ref_{name}", real=True)

        
    # dynamic states
    sigma = sym.Symbol(f"sigma_{name}", real=True)
    xi_sigma = sym.Symbol(f"xi_sigma_{name}", real=True)

    # algebraic states
    p_dc = sym.Symbol(f"p_dc_{name}", real=True)
    i_dc = sym.Symbol(f"i_dc_{name}", real=True)
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    E_kWh = sym.Symbol(f"E_kWh_{name}", real=True)
    sigma_min = sym.Symbol(f"sigma_min_{name}", real=True)
    sigma_max = sym.Symbol(f"sigma_max_{name}", real=True)
    A_loss = sym.Symbol(f"A_loss_{name}", real=True)
    B_loss = sym.Symbol(f"B_loss_{name}", real=True)
    C_loss = sym.Symbol(f"C_loss_{name}", real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)
    K_i = sym.Symbol(f"K_i_{name}", real=True)
    B_0 = sym.Symbol(f"B_0_{name}", real=True)
    B_1 = sym.Symbol(f"B_1_{name}", real=True)
    B_2 = sym.Symbol(f"B_2_{name}", real=True)
    B_3 = sym.Symbol(f"B_3_{name}", real=True)
    R_bat = sym.Symbol(f"R_bat_{name}", real=True)

    # auxiliar
    H = E_kWh*1000.0*3600/S_n
    epsilon = sigma_ref - sigma
    p_sigma = -(K_p*epsilon + K_i*xi_sigma)
    p_s = sym.Piecewise((p_s_ref,(p_s_ref <= 0.0) & (sigma<sigma_max)),
                        (p_s_ref,(p_s_ref > 0.0) & (sigma>sigma_min)),
                        (0.0,True)) + p_sigma
    #p_s = p_s_ref + p_sigma
    q_s = q_s_ref
    s_s = (p_s**2 + q_s**2)**0.5
    i_s = s_s/V_s
    p_loss = A_loss*i_s**2 + B_loss*i_s + C_loss
    e = B_0 + B_1*sigma + B_2*sigma**2 + B_3*sigma**3

    # dynamic equations    
    dsigma = 1/H*(-i_dc*e)   
    dxi_sigma = epsilon     

    # algebraic equations   
    g_p_dc = p_s + p_loss - p_dc
    g_i_dc = v_dc*i_dc - p_dc
    g_v_dc = e - i_dc*R_bat - v_dc

    # dae 
    grid.dae['f'] += [dsigma, dxi_sigma]
    grid.dae['x'] += [ sigma,  xi_sigma]
    grid.dae['g'] += [g_p_dc,g_i_dc,g_v_dc]
    grid.dae['y_ini'] += [p_dc, i_dc, v_dc]  
    grid.dae['y_run'] += [p_dc, i_dc, v_dc]  
    
    grid.dae['u_ini_dict'].update({f'{p_s_ref}':0.0})
    grid.dae['u_run_dict'].update({f'{p_s_ref}':0.0})          
    grid.dae['u_ini_dict'].update({f'{q_s_ref}':0.0})
    grid.dae['u_run_dict'].update({f'{q_s_ref}':0.0})     
    grid.dae['u_ini_dict'].update({f'{sigma_ref}':0.5})
    grid.dae['u_run_dict'].update({f'{sigma_ref}':0.5})      

    # outputs
    grid.dae['h_dict'].update({f"p_loss_{name}":p_loss})
    grid.dae['h_dict'].update({f"i_s_{name}":i_s})

    # parameters  
    grid.dae['params_dict'].update({f"{K_p}":1e-6}) 
    grid.dae['params_dict'].update({f"{K_i}":1e-6})  
    grid.dae['params_dict'].update({f"{sigma_min}":0.0}) 
    grid.dae['params_dict'].update({f"{sigma_max}":1.0})           
    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 
    grid.dae['params_dict'].update({f"{E_kWh}":data_dict['E_kWh']}) 
    if 'A_loss' in data_dict:
        A_loss_N = data_dict['A_loss']
        B_loss_N = data_dict['B_loss']
        C_loss_N = data_dict['C_loss']
    else:
        A_loss_N = 0.0001
        B_loss_N = 0.0
        C_loss_N = 0.0001

    grid.dae['params_dict'].update({f"{A_loss}":A_loss_N}) 
    grid.dae['params_dict'].update({f"{B_loss}":B_loss_N}) 
    grid.dae['params_dict'].update({f"{C_loss}":C_loss_N}) 

    if 'A_loss' in data_dict:
        B_0_N = data_dict['B_0_N']
        B_1_N = data_dict['B_1_N']
        B_2_N = data_dict['B_2_N']
        B_3_N = data_dict['B_3_N']
    else:
        B_0_N = 1.2
        B_1_N = 0.0
        B_2_N = 0.0
        B_3_N = 0.0

    grid.dae['params_dict'].update({f"{B_0}":B_0_N}) 
    grid.dae['params_dict'].update({f"{B_1}":B_1_N}) 
    grid.dae['params_dict'].update({f"{B_2}":B_2_N}) 
    grid.dae['params_dict'].update({f"{B_3}":B_3_N}) 

    if 'R_bat' in data_dict:
        R_bat_N = data_dict['R_bat']
    else:
        R_bat_N = 0.0

    grid.dae['params_dict'].update({f"{R_bat}":R_bat_N}) 

    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 
    grid.dae['params_dict'].update({f"{E_kWh}":data_dict['E_kWh']}) 

    grid.dae['xy_0_dict'].update({str(v_dc):1.2})
    grid.dae['xy_0_dict'].update({str(sigma):0.5})
    grid.dae['xy_0_dict'].update({str(xi_sigma):0.5})


    p_W   = p_s*S_n
    q_var = q_s*S_n

    return p_W,q_var


if __name__=='__main__':

    from pydae.bmapu import bmapu_builder

    P_bess = 10e6
    E_bess = 20e6

    data = {
    "system":{"name":"smib","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":1e-6},       
    "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
    "vscs":[{"type":"bess_pq","bus":"1","E_kWh":E_bess/1e3,"S_n":P_bess,"K_delta":0.01}],
    "sources":[{"type":"vsource","bus":"1"}]
    }
    
    grid = bmapu_builder.bmapu(data)
    grid.checker()
    grid.verbose = True 
    grid.build('smib_bess_pq')

    import smib_bess_pq

    model = smib_bess_pq.model()
    model.ini({},'xy_0.json')
    model.report_x()
    model.report_y()
    model.report_z()    

    print('run:')
    model.Dt = 1.0
    model.run(1.0,{})
    model.run(0.5*3600,{'p_s_ref_1':1.0}) # half an hour discharging
  
    model.report_x()
    model.report_y()
    model.report_z()    


    model.run(1.0*3600,{'p_s_ref_1':-1.0}) # half an hour discharging
  
    model.report_x()
    model.report_y()
    model.report_z()    

    model.post()

