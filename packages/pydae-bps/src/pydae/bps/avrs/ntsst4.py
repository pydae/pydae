# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


def ntsst4(dae,data,name,bus_name,backend=None):
    '''

    .. table:: Constants
        :widths: auto

        ================== =========== ============================================= =========== 
        Variable           Code        Description                                   Units
        ================== =========== ============================================= ===========

        :math:`T_A`        ``T_a``     Time Constant                                 s
        ================== =========== ============================================= ===========

    Example:
    
    ``"avr":{"type":"ntsst4","K_pr":3.15,"K_ir":3.15,"V_rmax":1.0,"V_rmin":-0.87,"T_a":0.02,"K_pm":1.0,"K_im":0.0,"K_p": 6.5,"v_ref":1.0},``

    '''
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'Piecewise': sym.Piecewise,
            'sin': sym.sin,
            'cos': sym.cos,
            'sqrt': sym.sqrt,
            'exp': sym.exp,
        })()

    avr_data = data['avr']
    remote_bus_name = bus_name
    if 'bus' in avr_data:
        remote_bus_name = avr_data['bus']
    
    v_t = backend.symbols(f"V_{remote_bus_name}", real=True)   
    v_c = backend.symbols(f"v_c_{remote_bus_name}", real=True)  
    xi_v  = backend.symbols(f"xi_v_{name}", real=True)
    x_a   = backend.symbols(f"x_a_{name}", real=True)
    xi_m  = backend.symbols(f"xi_m_{name}", real=True)
    v_f = backend.symbols(f"v_f_{name}", real=True)  
    K_pr   = backend.symbols(f'K_pr_{name}', real=True) 
    K_ir   = backend.symbols(f'K_ir_{name}', real=True) 
    T_a    = backend.symbols(f'T_a_{name}', real=True) 
    K_pm   = backend.symbols(f'K_pm_{name}', real=True) 
    K_im   = backend.symbols(f'K_im_{name}', real=True) 
    K_p    = backend.symbols(f'K_p_{name}', real=True)  

    #V_rmax = backend.symbols('V_rmax ', real=True) 
    #V_rmin = backend.symbols('V_rmin ', real=True) 
    #V_mmax = backend.symbols('V_mmax ', real=True) 
    #V_mmin = backend.symbols('V_mmin ', real=True) 
    #K_g    = backend.symbols('K_g    ', real=True) 
    #K_i    = backend.symbols('K_i    ', real=True) 
    #T_r    = backend.symbols('T_r    ', real=True) 
    #V_bmax = backend.symbols('V_bmax ', real=True) 
    #K_c    = backend.symbols('K_c    ', real=True) 
    #X_l    = backend.symbols('X_l    ', real=True) 
    #theta_p= backend.symbols('theta_p', real=True) 
    
    v_ref = backend.symbols(f"v_ref_{name}", real=True) 
    v_pss = backend.symbols(f"v_pss_{name}", real=True) 

    v_s = v_pss # v_oel and v_uel are not considered
    #v_ini = K_ai*xi_v
    v_c = v_t

    epsilon_v = v_ref - v_c + v_s
    v_r = K_pr*epsilon_v + K_ir*xi_v
    v_e = K_p*v_c

    epsilon_m = x_a # -K_g*v_f;
    v_m = K_pm*epsilon_m + K_im*xi_m
 
    dxi_v = epsilon_v
    dx_a = 1.0/T_a*(v_r - x_a)
    dxi_m = epsilon_m
      
    g_v_f  =  v_m*v_e - v_f 
    
    dae['f'] += [dxi_v,dx_a,dxi_m]
    dae['x'] += [ xi_v, x_a, xi_m]
    dae['g'] += [g_v_f]
    dae['y_ini'] += [v_f] 
    dae['y_run'] += [v_f]  

    dae['params_dict'].update({str(K_pr):avr_data['K_pr']}) 
    dae['params_dict'].update({str(K_ir):avr_data['K_ir']}) 
    dae['params_dict'].update({str(T_a):avr_data['T_a']})  
    dae['params_dict'].update({str(K_pm):avr_data['K_pm']}) 
    dae['params_dict'].update({str(K_im):avr_data['K_im']}) 
    dae['params_dict'].update({str(K_p):avr_data['K_p']})  

    dae['u_ini_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_run_dict'].update({str(v_ref):avr_data['v_ref']})

    dae['u_run_dict'].update({str(v_pss):0.0})
    dae['u_ini_dict'].update({str(v_pss):0.0})

    dae['xy_0_dict'].update({str(xi_v):1.0})
    dae['xy_0_dict'].update({str(x_a):1.0})
    dae['xy_0_dict'].update({str(xi_m):1.0})
    
def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bps import BpsBuilder
    import pydae.build_cffi as db
    import pytest

    grid = BpsBuilder('ntsst4.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')

    import temp

    model = temp.model()

    v_ref_1 = 1.0
    model.ini({},'xy_0.json')

    model.report_x()
    model.report_y()

    assert model.get_value('V_2') == pytest.approx(v_ref_1, rel=0.001)


if __name__ == '__main__':

    #development()
    test()