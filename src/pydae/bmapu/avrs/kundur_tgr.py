# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


import sympy as sym


def kundur_tgr(dae,syn_data,name):
    '''
        "avr":{"type":"kundur_tgr","K_a":200,"T_r":0.01,"E_fmin":-5,"E_fmax":10.0,"T_a":1,"T_b":10,"v_ref":1.0}
       
     '''

    avr_data = syn_data['avr']
    
    v_t = sym.Symbol(f"V_{name}", real=True)   
    v_r = sym.Symbol(f"v_r_{name}", real=True)  
    x_ab = sym.Symbol(f"x_ab_{name}", real=True)  
    xi_v  = sym.Symbol(f"xi_v_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    T_r = sym.Symbol(f"T_r_{name}", real=True) 
    T_a = sym.Symbol(f"T_a_{name}", real=True) 
    T_b = sym.Symbol(f"T_b_{name}", real=True) 
    K_a = sym.Symbol(f"K_a_{name}", real=True)
    K_ai = sym.Symbol(f"K_ai_{name}", real=True)
    E_fmin = sym.Symbol(f"E_fmin_{name}", real=True)
    E_fmax = sym.Symbol(f"E_fmax_{name}", real=True)
    K_aw = sym.Symbol(f"K_aw_{name}", real=True)   
    
    # inputs
    v_ref = sym.Symbol(f"v_ref_{name}", real=True) 
    v_s   = sym.Symbol(f"v_pss_{name}", real=True) 

    # auxiliar
    epsilon_v = v_ref - v_r + v_s
    u_ab = K_a * epsilon_v
    z_ab = (u_ab - x_ab)*T_a/T_b + x_ab
    v_f_nosat = z_ab + K_ai*xi_v + 1.5
    
    # differential equations
    dv_r =   (v_t - v_r)/T_r
    dx_ab =  (u_ab - x_ab)/T_b      # lead compensator state
    dxi_v =   epsilon_v 

    # algebraic equations   
    g_v_f  =   sym.Piecewise((E_fmin, v_f_nosat<E_fmin),(E_fmax,v_f_nosat>E_fmax),(v_f_nosat,True)) - v_f 
  
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
    dae['params_dict'].update({str(E_fmin):avr_data['E_fmin']})  
    dae['params_dict'].update({str(E_fmax):avr_data['E_fmax']}) 

    dae['u_ini_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_ini_dict'].update({str(v_s):0.0})

    dae['u_run_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_run_dict'].update({str(v_s):0.0})

    dae['xy_0_dict'].update({str(v_r):1.0})
    dae['xy_0_dict'].update({str(x_ab):1.0})
    dae['xy_0_dict'].update({str(xi_v):0.0})
    
