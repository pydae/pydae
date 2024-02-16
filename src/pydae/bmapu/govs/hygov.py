# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


import sympy as sym


def hygov(dae,data,name,bus_name):
    '''

    .. table:: Constants
        :widths: auto

    Example:
    
    ``"avr":{"type":"ntsst1","K_a":200.0,"T_c":1.0,"T_b":10.0,"v_ref":1.0},``

    '''

    data_gov = data['gov']

    ## parameters
    R =      sym.Symbol(f"R_{name}", real=True) 
    T_w =    sym.Symbol(f"T_w_gov_{name}", real=True) 
    D_turb = sym.Symbol(f"D_turb_gov_{name}", real=True) 
    Q_nl =   sym.Symbol(f"Q_nl_gov_{name}", real=True) 
    A_t =    sym.Symbol(f"A_t_gov_{name}", real=True) 
    T_f =    sym.Symbol(f"T_f_gov_{name}", real=True) 
    T_r =    sym.Symbol(f"T_r_gov_{name}", real=True) 
    R_r =    sym.Symbol(f"R_r_gov_{name}", real=True) 
    T_g =    sym.Symbol(f"T_g_gov_{name}", real=True) 
    G_max =    sym.Symbol(f"G_max_gov_{name}", real=True) 
    G_min =    sym.Symbol(f"G_min_gov_{name}", real=True) 
    V_g_max =    sym.Symbol(f"V_g_max_gov_{name}", real=True) 
    K_sec = sym.Symbol(f"K_sec_{name}", real=True)  # 0.05

    ## input
    omega =    sym.Symbol(f"omega_{name}", real=True) 
    n_ref =    sym.Symbol(f"n_ref_gov_{name}", real=True) 
    p_c =    sym.Symbol(f"p_c_gov_{name}", real=True) 
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)    
    R_a = sym.Symbol(f"R_a_{name}", real=True)  
    p_agc = sym.Symbol(f"p_agc", real=True)

    ## dynamic states
    e =    sym.Symbol(f"e_gov_{name}", real=True) 
    c_i =    sym.Symbol(f"c_i_gov_{name}", real=True) 
    g =    sym.Symbol(f"g_gov_{name}", real=True) 
    q =    sym.Symbol(f"q_gov_{name}", real=True) 

    ## algebraic states
    p_m = sym.Symbol(f"p_m_{name}", real=True) 

    ## auxiliar
    c = c_i+e*R_r
    epsilon = n_ref - (omega + R*c)
    g_sat = g
    #g_sat = sym.Piecewise((G_max,g>G_max),(G_min,g>G_min),(g,True))
    h = (q/g_sat)**2 
    losses_p = R_a*(i_d**2 + i_q**2)
    p_r = K_sec*p_agc

    p_ref = p_c + losses_p + p_r

    de = 1/T_f*(p_ref + epsilon - e) 
    dc_i = 1/(R_r*T_r)*(e - 0.001*c_i)
    dg = 1/T_g*(c - g) 
    dq = 1/T_w*(1-h) 

    eq_p_m = -p_m + A_t*h*(q-Q_nl) - D_turb*omega*g

    dae['f'] += [de, dc_i, dg, dq]
    dae['x'] += [ e,  c_i,  g,  q]


    dae['g'] += [eq_p_m]
    dae['y_ini'] += [p_m] 
    dae['y_run'] += [p_m]  

    p_c_N = data_gov['p_c']
    dae['u_ini_dict'].update({str(p_c): p_c_N})  
    dae['u_run_dict'].update({str(p_c): p_c_N})  

    dae['params_dict'].update({str(R): data_gov['R']})  
    dae['params_dict'].update({str(R_r): data_gov['R_r']})  
    dae['params_dict'].update({str(T_r): data_gov['T_r']})  
    dae['params_dict'].update({str(T_f): data_gov['T_f']})  
    dae['params_dict'].update({str(T_g): data_gov['T_g']})  
    dae['params_dict'].update({str(V_g_max): data_gov['V_g_max']})  
    dae['params_dict'].update({str(G_max): data_gov['G_max']})  
    dae['params_dict'].update({str(G_min): data_gov['G_min']})  
    dae['params_dict'].update({str(T_w): data_gov['T_w']})  
    dae['params_dict'].update({str(A_t): data_gov['A_t']})  
    dae['params_dict'].update({str(D_turb): data_gov['D_turb']})  
    dae['params_dict'].update({str(Q_nl): data_gov['Q_nl']})  
    dae['params_dict'].update({str(K_sec):data_gov['K_sec']})

    dae['u_ini_dict'].update({str(n_ref):1.0})
    dae['u_run_dict'].update({str(n_ref):1.0})

    dae['xy_0_dict'].update({str(q):p_c_N})
    dae['xy_0_dict'].update({str(g):p_c_N})
    dae['xy_0_dict'].update({str(p_m):p_c_N})
    dae['xy_0_dict'].update({str(e):p_c_N})


    dae['h_dict'].update({f'h_gov_{name}':h})  
    dae['h_dict'].update({str(q):q})  
    dae['h_dict'].update({str(g):g})  


def test_build():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import pytest

    grid = bmapu('hygov.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')

def test_ini():

    import temp

    model = temp.model()

    v_ref_1 = 1.05
    model.ini({'P_2':-80e6},'xy_0.json')
    model.report_x()
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
    test_build()
    test_ini()
