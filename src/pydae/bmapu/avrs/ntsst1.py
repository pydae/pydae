# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


import sympy as sym


def ntsst1(dae,syn_data,name,bus_name):
    '''

    .. table:: Constants
        :widths: auto

    Example:
    
    ``"avr":{"type":"ntsst1","K_a":200.0,"T_c":1.0,"T_b":10.0,"v_ref":1.0},``

    '''


    avr_data = syn_data['avr']
    
    v_t = sym.Symbol(f"V_{bus_name}", real=True)  
    v_r = sym.Symbol(f"v_r_{name}", real=True)   
    v_c = sym.Symbol(f"v_c_{name}", real=True)  
    x_cb  = sym.Symbol(f"x_cb_{name}", real=True)
    xi_v  = sym.Symbol(f"xi_v_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    T_r = sym.Symbol(f"T_r_{name}", real=True) 
    T_b = sym.Symbol(f"T_b_{name}", real=True) 
    T_c = sym.Symbol(f"T_c_{name}", real=True) 

    K_a = sym.Symbol(f"K_a_{name}", real=True)
    K_ai = sym.Symbol(f"K_ai_{name}", real=True)

    V_f_min = sym.Symbol(f"V_f_min_{name}", real=True)
    V_f_max = sym.Symbol(f"V_f_max_{name}", real=True)

    v_ref = sym.Symbol(f"v_ref_{name}", real=True) 
    v_pss = sym.Symbol(f"v_pss_{name}", real=True) 

    v_s = v_pss # v_oel and v_uel are not considered
    v_ini = K_ai*xi_v

    v_c = v_t # no droop is considered
    v_1 = v_ref - v_c + v_s + v_ini # v_ini is added in pydae to force V = v_ref in the initialization
    
    epsilon_v = v_ref - v_c 
    
    dv_r = 1/T_r*(v_c - v_r)
    dx_cb = (v_1 - x_cb)/T_b;  
    z_cb  = (v_1 - x_cb)*T_c/T_b + x_cb 
    dxi_v = epsilon_v  # this integrator is added in pydae to force V = v_ref in the initialization
    
    v_f_nosat = K_a*z_cb
    v_f_sat = sym.Piecewise((V_f_max,v_f_nosat>V_f_max),(V_f_min,v_f_nosat<V_f_min),(v_f_nosat,True))
    #v_f_sat = v_f_nosat
    g_v_f  =   v_f_sat - v_f 
    dv_f = 1/T_r*(v_f_sat - v_f)
    
    dae['f'] += [dv_r,dx_cb,dxi_v,dv_f]
    dae['x'] += [ v_r, x_cb, xi_v, v_f]
    # dae['g'] += [g_v_f]
    # dae['y_ini'] += [v_f] 
    # dae['y_run'] += [v_f]  

    dae['params_dict'].update({str(K_a):avr_data['K_a']})
    dae['params_dict'].update({str(K_ai):1e-6})
    dae['params_dict'].update({str(T_r):avr_data['T_r']})  
    dae['params_dict'].update({str(T_c):avr_data['T_c']})  
    dae['params_dict'].update({str(T_b):avr_data['T_b']})  

    dae['params_dict'].update({str(V_f_max): 10.0})  
    dae['params_dict'].update({str(V_f_min):-10.0})  


    dae['u_ini_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_run_dict'].update({str(v_ref):avr_data['v_ref']})

    dae['u_run_dict'].update({str(v_pss):0.0})
    dae['u_ini_dict'].update({str(v_pss):0.0})

    dae['xy_0_dict'].update({str(v_f):2.0})
    dae['xy_0_dict'].update({str(v_r):1.0})
    dae['xy_0_dict'].update({str(xi_v):10.0})
    dae['xy_0_dict'].update({str(x_cb):2.0})

    dae['h_dict'].update({str(v_ref):v_ref})  


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import pytest

    grid = bmapu('ntsst1.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')

    import temp

    model = temp.model()

    v_ref_1 = 1.05
    model.ini({'p_m_1':0.5,'v_ref_1':v_ref_1},'xy_0.json')

    assert model.get_value('V_1') == pytest.approx(v_ref_1, rel=0.001)
    # assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)

    model.ini({'p_m_1':0.5,'v_ref_1':1.0},'xy_0.json')
    model.run(1.0,{})
    model.run(15.0,{'v_ref_1':1.05})
    model.post()

    import matplotlib.pyplot as plt

    fig,axes = plt.subplots()
    axes.plot(model.Time,model.get_values('V_1'))
    fig.savefig('ntsst1_step.svg')


if __name__ == '__main__':

    #development()
    test()
