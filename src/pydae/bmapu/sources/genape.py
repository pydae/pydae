# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def genape(grid,name,bus_name,data_dict):
    '''

    parameters
    ----------

    S_n: nominal power in VA
    F_n: nominal frequency in Hz
    X_v: coupling reactance in pu (base machine S_n)
    R_v: coupling resistance in pu (base machine S_n)
    K_delta: if K_delta>0.0 current generator is converted to reference machine 
    K_alpha: alpha gain to obtain Domega integral 

    inputs
    ------

    alpha: RoCoF in pu if K_alpha = 1.0
    omega_ref: frequency in pu
    v_ref: internal voltage reference

    example
    -------

    "genapes": [{"S_n":1e9,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]

    S_n = sym.Symbol(f"S_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)            
    X_v = sym.Symbol(f"X_v_{name}", real=True)
    R_v = sym.Symbol(f"R_v_{name}", real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True)
    K_alpha = sym.Symbol(f"K_alpha_{name}", real=True)

    
    '''

    sin = sym.sin
    cos = sym.cos

    
    # inputs
    V_sm = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)
    omega_coi = sym.Symbol("omega_coi", real=True)   
    alpha = sym.Symbol(f"alpha_{name}", real=True)
    v_ref = sym.Symbol(f"v_ref_{name}", real=True)
    omega_ref = sym.Symbol(f"omega_ref_{name}", real=True) 
    phi = sym.Symbol(f"phi_{name}", real=True) 
    delta_ref = sym.Symbol(f"delta_ref_{name}", real=True) 
    rocov = sym.Symbol(f"rocov_{name}", real=True) 
    
    # dynamic states
    delta = sym.Symbol(f"delta_{name}", real=True)
    omega = sym.Symbol(f"omega_{name}", real=True)
    Domega = sym.Symbol(f"Domega_{name}", real=True)
    Dv = sym.Symbol(f"Dv_{name}", real=True)

    # algebraic states
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)            
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)
    
    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)            
    X_v = sym.Symbol(f"X_v_{name}", real=True)
    R_v = sym.Symbol(f"R_v_{name}", real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True)
    K_alpha = sym.Symbol(f"K_alpha_{name}", real=True)
    K_rocov = sym.Symbol(f"K_rocov_{name}", real=True)
    
    params_list = ['S_n','F_n','X_v','R_v','K_delta','K_alpha']
    
    # auxiliar
    theta_v = delta + phi 
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi 

    V_sr =  V_sm * sym.cos(theta_s)
    V_si =  V_sm * sym.sin(theta_s)

    V_vm = v_ref + Dv  
    V_vr =  V_vm * sym.cos(theta_v)
    V_vi =  V_vm * sym.sin(theta_v)

    p_pu = (R_v*V_si**2 - R_v*V_si*V_vi + R_v*V_sr**2 - R_v*V_sr*V_vr + V_si*V_vr*X_v - V_sr*V_vi*X_v)/(R_v**2 + X_v**2)
    q_pu = (-R_v*V_si*V_vr + R_v*V_sr*V_vi + V_si**2*X_v - V_si*V_vi*X_v + V_sr**2*X_v - V_sr*V_vr*X_v)/(R_v**2 + X_v**2)


    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*(delta - delta_ref)
    dDomega = alpha - K_alpha*Domega
    dDv = rocov - K_rocov*Dv

    # algebraic equations   
    g_omega = -omega + Domega + omega_ref

    
    # dae 
    f_vsg = [ddelta,dDomega,dDv]
    x_vsg = [ delta, Domega, Dv]
    g_vsg = [g_omega]
    y_vsg = [  omega]
    
    H = 1e6
    grid.H_total += H
    grid.omega_coi_numerator += omega*H*S_n
    grid.omega_coi_denominator += H*S_n

    grid.dae['f'] += f_vsg
    grid.dae['x'] += x_vsg
    grid.dae['g'] += g_vsg
    grid.dae['y_ini'] += y_vsg  
    grid.dae['y_run'] += y_vsg  
    
        
    grid.dae['u_ini_dict'].update({f'{str(alpha)}':0})
    grid.dae['u_run_dict'].update({f'{str(alpha)}':0})

    grid.dae['u_ini_dict'].update({f'{str(v_ref)}':1.0})
    grid.dae['u_run_dict'].update({f'{str(v_ref)}':1.0})

    grid.dae['u_ini_dict'].update({f'{str(omega_ref)}':1.0})
    grid.dae['u_run_dict'].update({f'{str(omega_ref)}':1.0})

    grid.dae['u_ini_dict'].update({f'{str(delta_ref)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(delta_ref)}':0.0})

    grid.dae['u_ini_dict'].update({f'{str(phi)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(phi)}':0.0})

    grid.dae['u_ini_dict'].update({f'{str(rocov)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(rocov)}':0.0})

    grid.dae['xy_0_dict'].update({str(omega):1.0})
       
    # outputs
    grid.dae['h_dict'].update({f"alpha_{name}":alpha})
    grid.dae['h_dict'].update({f"Dv_{name}":Dv})
    grid.dae['h_dict'].update({f"theta_v_{name}":theta_v})
    
    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]}) 

    grid.dae['params_dict'].update({f"K_rocov_{name}":1e-6}) 

    p_W   = p_pu * S_n
    q_var = q_pu * S_n

    return p_W,q_var

def development():
    bus_name = '1'
    name = bus_name

    # inputs
    V_sr = sym.Symbol(f"V_sr", real=True)
    V_si = sym.Symbol(f"V_si", real=True)
    V_vr = sym.Symbol(f"V_vr", real=True)
    V_vi = sym.Symbol(f"V_vi", real=True)

    V_s = V_sr + sym.I*V_si
    V_v = V_vr + sym.I*V_vi

    
    # parameters
    S_n = sym.Symbol(f"S_n", real=True)
    F_n = sym.Symbol(f"F_n", real=True)            
    X_v = sym.Symbol(f"X_v", real=True)
    R_v = sym.Symbol(f"R_v", real=True)

    Z_v = R_v + sym.I*X_v

    I = (V_s - V_v)/Z_v

    S = sym.expand(sym.simplify(V_s*np.conjugate(I)))

    print('p_pu =',sym.simplify((sym.re(S))))
    print('q_pu =',sym.simplify((sym.im(S))))


def test_build():

    from pydae.bmapu import bmapu_builder
    
    grid = bmapu_builder.bmapu('genape.hjson')
    grid.checker()
    grid.verbose = True 
    grid.build('temp')

def test_ini():

    import temp

    model = temp.model()
    model.ini({},'xy_0.json')
    model.report_x()
    model.report_y()


 
if __name__=='__main__':
    #development()
    test_build()
    test_ini()
