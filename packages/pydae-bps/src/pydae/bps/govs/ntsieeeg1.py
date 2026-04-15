# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


import sympy as sym


def ntsieeeg1(dae,syn_data,name,bus_name):
    '''

    .. table:: Constants
        :widths: auto

        ================== =========== ============================================= =========== 
        Variable           Code        Description                                   Units
        ================== =========== ============================================= ===========
        :math:`T_A`        ``T_a``     Time Constant                                 s
        ================== =========== ============================================= ===========

    Example:
    
    ``"gov":{"type":"ntsst4","K":20,"K_1":0.3,"K_3":0.3,"K_5":0.4,"K_7":0,"T_1":0,"T_2":0.0,"T_3":0.1,"T_4":0.3,"T_5":7, "T_6":0.6, "T_7":0,
 "K_2":0,"K_4":0,"K_6":0,"K_8":0,"U_0":0.5, "U_c":-0.5,"p_c":0.1}
    '''

    data = syn_data['gov']
    
    omega = sym.Symbol(f"omega_{name}", real=True)   

    p_c = sym.Symbol(f"p_c_{name}", real=True)  
    p_agc = sym.Symbol(f"p_agc", real=True)  

    x_3 = sym.Symbol(f"x_3_gov_{name}", real=True);
    x_4 = sym.Symbol(f"x_4_gov_{name}", real=True);
    x_5 = sym.Symbol(f"x_5_gov_{name}", real=True);
    x_6 = sym.Symbol(f"x_6_gov_{name}", real=True);
    p_m = sym.Symbol(f"p_m_{name}", real=True)  

    K    = sym.Symbol(f"K_gov_{name}", real=True)
    K_1  = sym.Symbol(f"K_1_gov_{name}", real=True)
    K_3  = sym.Symbol(f"K_3_gov_{name}", real=True)
    K_5  = sym.Symbol(f"K_5_gov_{name}", real=True)
    K_7  = sym.Symbol(f"K_7_gov_{name}", real=True)
    T_1  = sym.Symbol(f"T_1_gov_{name}", real=True)
    T_2  = sym.Symbol(f"T_2_gov_{name}", real=True)
    T_3  = sym.Symbol(f"T_3_gov_{name}", real=True)
    T_4  = sym.Symbol(f"T_4_gov_{name}", real=True)
    T_5  = sym.Symbol(f"T_5_gov_{name}", real=True)
    T_6  = sym.Symbol(f"T_6_gov_{name}", real=True)
    T_7  = sym.Symbol(f"T_7_gov_{name}", real=True)
    K_2  = sym.Symbol(f"K_2_gov_{name}", real=True)
    K_4  = sym.Symbol(f"K_4_gov_{name}", real=True)
    K_6  = sym.Symbol(f"K_6_gov_{name}", real=True)
    K_8  = sym.Symbol(f"K_8_gov_{name}", real=True)
    U_0  = sym.Symbol(f"U_0_gov_{name}", real=True)
    U_c  = sym.Symbol(f"U_c_gov_{name}", real=True) 
    P_min  = sym.Symbol(f"P_min_gov_{name}", real=True)
    P_max  = sym.Symbol(f"P_max_gov_{name}", real=True) 
    K_awu  = sym.Symbol(f"K_awu_gov_{name}", real=True) 
    K_sec = sym.Symbol(f"K_sec_{name}", real=True)    


    Domega = 1.0 - omega

    y_g_nosat = x_3
    y_g = sym.Piecewise((P_max,y_g_nosat > P_max),(P_min,y_g_nosat < P_min),(y_g_nosat,True))
    # y_g = y_g_nosat

    u_3 = K*Domega + p_c - y_g + K_sec*p_agc
    u_g = sym.Piecewise((U_c,u_3/T_3<U_c),(U_0,u_3/T_3>U_0),(u_3/T_3,True))
    u_awu = K_awu*(y_g - y_g_nosat)

    dx_3 = u_g + u_awu
    dx_4 = 1/T_4*(y_g - x_4)
    dx_5 = 1/T_5*(x_4 - x_5)
    dx_6 = 1/T_6*(x_5 - x_6)

    g_p_m = -p_m + K_1*x_4 + K_3*x_5 + K_5*x_6

    dae['f'] += [dx_3,dx_4,dx_5,dx_6]
    dae['x'] += [ x_3, x_4, x_5, x_6]
    dae['g'] += [g_p_m]
    dae['y_ini'] += [p_m] 
    dae['y_run'] += [p_m]  

    dae['params_dict'].update({str(K):data['K']})
    dae['params_dict'].update({str(K_1):data['K_1']})
    dae['params_dict'].update({str(K_3):data['K_3']})
    dae['params_dict'].update({str(K_5):data['K_5']})
    dae['params_dict'].update({str(K_7):data['K_7']})
    dae['params_dict'].update({str(T_1):data['T_1']})
    dae['params_dict'].update({str(T_2):data['T_2']})
    dae['params_dict'].update({str(T_3):data['T_3']})
    dae['params_dict'].update({str(T_4):data['T_4']})
    dae['params_dict'].update({str(T_5):data['T_5']})
    dae['params_dict'].update({str(T_6):data['T_6']})
    dae['params_dict'].update({str(T_7):data['T_7']})
    dae['params_dict'].update({str(K_2):data['K_2']})
    dae['params_dict'].update({str(K_4):data['K_4']})
    dae['params_dict'].update({str(K_6):data['K_6']})
    dae['params_dict'].update({str(K_8):data['K_8']})
    dae['params_dict'].update({str(U_0):data['U_0']})
    dae['params_dict'].update({str(U_c):data['U_c']})
    dae['params_dict'].update({str(P_max):data['P_max']})
    dae['params_dict'].update({str(P_min):data['P_min']})
    dae['params_dict'].update({str(K_awu):1000.0}) 

    p_c_ini = data['p_c']

    dae['u_ini_dict'].update({str(p_c):p_c_ini})
    dae['u_run_dict'].update({str(p_c):p_c_ini})

    dae['xy_0_dict'].update({str(x_3):p_c_ini})
    dae['xy_0_dict'].update({str(x_4):p_c_ini})
    dae['xy_0_dict'].update({str(x_5):p_c_ini})
    dae['xy_0_dict'].update({str(x_6):p_c_ini})
    dae['xy_0_dict'].update({str(p_m):p_c_ini})    


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import matplotlib.pyplot as plt
    import pytest

    grid = bmapu('ieeeg1.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')

    import temp

    model = temp.model()

    model.ini({'P_2':-80e6},'xy_0.json')
    model.ini({})
    model.report_x()
    model.report_y()
    model.report_z()

    model.Dt = 0.01

    model.run(10.0,{})
    model.run(20.0,{'P_2':-100e6})
    model.post();

    fig,axes = plt.subplots()
    axes.plot(model.Time,model.get_values('omega_coi'))
    fig.savefig('ieeeg1.svg')




if __name__ == '__main__':

    #development()
    test()









 
