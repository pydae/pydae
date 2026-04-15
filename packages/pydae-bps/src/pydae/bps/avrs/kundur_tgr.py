# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


import sympy as sym


def kundur_tgr(dae,syn_data,name,bus_name):
    '''
        "avr":{"type":"kundur_tgr","K_a":200,"T_r":0.01,"T_a":1,"T_b":10,"v_ref":1.0}
       
     '''

    avr_data = syn_data['avr']
    
    v_c = sym.Symbol(f"V_{name}", real=True)   
    v_r = sym.Symbol(f"v_r_{name}", real=True)  
    x_ab = sym.Symbol(f"x_ab_{name}", real=True)  
    xi_v  = sym.Symbol(f"xi_v_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    T_r = sym.Symbol(f"T_r_{name}", real=True) 
    T_a = sym.Symbol(f"T_a_{name}", real=True) 
    T_b = sym.Symbol(f"T_b_{name}", real=True) 
    K_a = sym.Symbol(f"K_a_{name}", real=True)
    K_ai = sym.Symbol(f"K_ai_{name}", real=True)
    
    # inputs
    v_ref = sym.Symbol(f"v_ref_{name}", real=True) 
    v_s   = sym.Symbol(f"v_pss_{name}", real=True) 

    # auxiliar
    epsilon_v = v_ref - v_r 
    v_i = K_ai*xi_v
    u_ab = K_a * (epsilon_v + v_s + v_i)
    z_ab = (u_ab - x_ab)*T_a/T_b + x_ab
    
        
    # differential equations
    dv_r =   (v_c - v_r)/T_r
    dx_ab =  (u_ab - x_ab)/T_b      # lead compensator state
    dxi_v =   epsilon_v 

    # algebraic equations   
    g_v_f  =   z_ab - v_f 
  
    dae['f'] += [dv_r,dx_ab,dxi_v]
    dae['x'] += [ v_r, x_ab, xi_v]
    dae['g'] += [g_v_f]
    dae['y_ini'] += [v_f] 
    dae['y_run'] += [v_f]  

    dae['params_dict'].update({str(K_a):avr_data['K_a']})
    dae['params_dict'].update({str(K_ai):1e-6})
    dae['params_dict'].update({str(T_r):avr_data['T_r']})  
    dae['params_dict'].update({str(T_a):avr_data['T_a']}) 
    dae['params_dict'].update({str(T_b):avr_data['T_b']}) 

    dae['u_ini_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_ini_dict'].update({str(v_s):0.0})

    dae['u_run_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_run_dict'].update({str(v_s):0.0})

    dae['xy_0_dict'].update({str(v_r):1.0})
    dae['xy_0_dict'].update({str(x_ab):1.0})
    dae['xy_0_dict'].update({str(xi_v):0.0})
    
    dae['h_dict'].update({str(v_ref):v_ref})
    dae['h_dict'].update({str(v_s):v_s})