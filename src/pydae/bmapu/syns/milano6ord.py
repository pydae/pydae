# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
import pydae.ssa as ssa

def milano6ord(grid,name,bus_name,data_dict):
    """
    Model of order 6 as in PSAT Manual 2008
    
    """

    sin = sym.sin
    cos = sym.cos  

    # inputs
    V = sym.Symbol(f"V_{bus_name}", real=True)
    theta = sym.Symbol(f"theta_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    omega_coi = sym.Symbol("omega_coi", real=True)   
        
    # dynamic states
    delta = sym.Symbol(f"delta_{name}", real=True)
    omega = sym.Symbol(f"omega_{name}", real=True)
    e1q = sym.Symbol(f"e1q_{name}", real=True)
    e1d = sym.Symbol(f"e1d_{name}", real=True)
    e2d = sym.Symbol(f"e2d_{name}", real=True)
    e2q = sym.Symbol(f"e2q_{name}", real=True)

    # algebraic states
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)            
    p_g = sym.Symbol(f"p_g_{name}", real=True)
    q_g = sym.Symbol(f"q_g_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
    H = sym.Symbol(f"H_{name}", real=True)
    T1d0 = sym.Symbol(f"T1d0_{name}", real=True)
    T1q0 = sym.Symbol(f"T1q0_{name}", real=True)
    T2d0 = sym.Symbol(f"T2d0_{name}", real=True)
    T2q0 = sym.Symbol(f"T2q0_{name}", real=True)
    T_AA = sym.Symbol(f"T_AA_{name}", real=True)
    X_l = sym.Symbol(f"X_l_{name}", real=True)
    X_d = sym.Symbol(f"X_d_{name}", real=True)
    X_q = sym.Symbol(f"X_q_{name}", real=True)
    X1d = sym.Symbol(f"X1d_{name}", real=True)
    X1q = sym.Symbol(f"X1q_{name}", real=True)
    X2d = sym.Symbol(f"X2d_{name}", real=True)
    X2q = sym.Symbol(f"X2q_{name}", real=True)
    D = sym.Symbol(f"D_{name}", real=True)
    R_a = sym.Symbol(f"R_a_{name}", real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True)
    K_sat = sym.Symbol(f"K_sat_{name}", real=True)

    params_list = ['S_n','Omega_b','H','T1d0','T1q0','T2d0','T2q0']
    params_list+= ['X_l','X_d','X_q','X1d','X1q','X2d','X2q','D','R_a','K_delta','K_sec']
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     

    omega_s = omega_coi
    f_s_e1q = K_sat*e1q

    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
    de1q = (      -e1q - (X_d - X1d - T2d0/T1d0*X2d/X1d * (X_d - X1d))*i_d + (1 - T_AA/T1d0)*v_f)/T1d0
    de1d = (      -e1d + (X_q - X1q - T2q0/T1q0*X2q/X1q * (X_q - X1q))*i_q)/T1q0
    de2q = (-e2q + e1q - (X1d - X2d + T2d0/T1d0*X2d/X1d * (X_d - X1d))*i_d + T_AA/T1d0 * v_f)/T2d0
    de2d = (-e2d + e1d + (X1q - X2q + T2q0/T1q0*X2q/X1q * (X_q - X1q))*i_q)/T2q0

    # algebraic equations   
    g_i_d  = v_q + R_a*i_q - e2q + (X2d - X_l)*i_d
    g_i_q  = v_d + R_a*i_d - e2d - (X2q - X_l)*i_q
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    
    # dae 
    f_syn = [ddelta,domega,de1q,de1d,de2q,de2d]
    x_syn = [ delta, omega, e1q, e1d, e2q, e2d]
    g_syn = [g_i_d,g_i_q,g_p_g,g_q_g]
    y_syn = [  i_d,  i_q,  p_g,  q_g]
    
    grid.H_total += H
    grid.omega_coi_numerator += omega*H*S_n
    grid.omega_coi_denominator += H*S_n

    grid.dae['f'] += f_syn
    grid.dae['x'] += x_syn
    grid.dae['g'] += g_syn
    grid.dae['y_ini'] += y_syn  
    grid.dae['y_run'] += y_syn  
    
    if 'v_f' in data_dict:
        grid.dae['u_ini_dict'].update({f'{v_f}':{data_dict['v_f']}})
        grid.dae['u_run_dict'].update({f'{v_f}':{data_dict['v_f']}})
    else:
        grid.dae['u_ini_dict'].update({f'{v_f}':1.5})
        grid.dae['u_run_dict'].update({f'{v_f}':1.5})

    if 'p_m' in data_dict:
        grid.dae['u_ini_dict'].update({f'{p_m}':{data_dict['p_m']}})
        grid.dae['u_run_dict'].update({f'{p_m}':{data_dict['p_m']}})
    else:
        grid.dae['u_ini_dict'].update({f'{p_m}':0.5})
        grid.dae['u_run_dict'].update({f'{p_m}':0.5})

    grid.dae['xy_0_dict'].update({str(omega):1.0})
    grid.dae['xy_0_dict'].update({str(e1q):1.0})
    grid.dae['xy_0_dict'].update({str(e1d):1.0})
    grid.dae['xy_0_dict'].update({str(e2q):1.0})
    grid.dae['xy_0_dict'].update({str(e2d):1.0})
    grid.dae['xy_0_dict'].update({str(i_d):0.5})
    grid.dae['xy_0_dict'].update({str(i_q):0.5})

    # outputs
    grid.dae['h_dict'].update({f"p_e_{name}":p_e})
    grid.dae['h_dict'].update({f"v_f_{name}":v_f})
    grid.dae['h_dict'].update({f"v_d_{name}":v_d})
    grid.dae['h_dict'].update({f"v_q_{name}":v_q})

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]})

    grid.dae['params_dict'].update({f"{T_AA}":0.0})
    grid.dae['params_dict'].update({f"{K_sat}":1.0})


    p_W   = p_g * S_n
    q_var = q_g * S_n

    return p_W,q_var



def test():

    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu import bmapu_builder
    import pydae.build_cffi as db
    import pytest

    grid = bmapu_builder.bmapu('milano6ord.hjson')
    grid.uz_jacs = True
    grid.verbose = True
    grid.build('temp')

    import temp

    model = temp.model()
    
    model.ini({'v_f_1':2,'p_m_1':1},'xy_0.json')

    model.report_x()
    model.report_y()

if __name__ == '__main__':

    test()
 