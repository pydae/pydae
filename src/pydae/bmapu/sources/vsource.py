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

    H = 1e15
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


def test_mkl():

    import numpy as np
    from pydae.bmapu import bmapu_builder
    from pydae.build_v2 import builder
    from pydae.utils import read_data
    import json

    grid = bmapu_builder.bmapu('vsource.hjson')
    grid.construct(f'temp')
    grid.compile_mkl('temp')

    import temp

    model = temp.model()
    model.ini({},'xy_0.json')
    model.report_y()


def test():

    from pydae.bmapu import bmapu_builder
    
    grid = bmapu_builder.bmapu('vsource.hjson')
    grid.checker()
    grid.verbose = True 
    grid.build('temp')

    import temp

    model = temp.model()
    model.ini({},'xy_0.json')
    model.report_y()


 
if __name__=='__main__':
    test_mkl()





