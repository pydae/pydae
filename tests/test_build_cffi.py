# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 23:38:58 2021

@author: jmmau
"""

import pytest
import numpy as np
import sympy as sym
import pydae.build_cffi as db
 
 
def test_leon_vsg_ll():
    import pydae.build_cffi as db
    from pydae.bmapu import bmapu_builder
    from pydae.bmapu.vsgs.vsgs import add_vsgs
    import pydae.build_cffi as db
    import sympy as sym

    data = {
        "sys":{"name":"sys2buses","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0}],
        "vsgs":[
            {"bus":"1","type":"vsg_ll",'S_n':10e6,'F_n':50,'K_delta':0.0,
            'R_v':0.01,'X_v':0.1,'K_p':1.0,'K_i':0.1,'K_g':0.0,'K_q':20.0,
            'T_q':0.1,'K_p_v':1e-6,'K_i_v':1e-6}],
        "genapes":[{"bus":"2","S_n":100e6,"F_n":50.0,"R_v":0.0,"X_v":0.1,"K_delta":0.001,"K_alpha":1.0}],
        }

    grid = bmapu_builder.bmapu(data)

    add_vsgs(grid)
    omega_coi = sym.Symbol("omega_coi", real=True)  

    grid.dae['g'] += [ -omega_coi + grid.omega_coi_numerator/grid.omega_coi_denominator]
    grid.dae['y_ini'] += [ omega_coi]
    grid.dae['y_run'] += [ omega_coi]

    sys_dict = {'name':'prueba','uz_jacs':True,
        'params_dict':grid.dae['params_dict'],
        'f_list':grid.dae['f'],
        'g_list':grid.dae['g'],
        'x_list':grid.dae['x'],
        'y_ini_list':grid.dae['y_ini'],
        'y_run_list':grid.dae['y_run'],
        'u_run_dict':grid.dae['u_run_dict'],
        'u_ini_dict':grid.dae['u_ini_dict'],
        'h_dict':grid.dae['h_dict']}

    bldr = db.builder(sys_dict)
    bldr.build()

    import prueba 
    model = prueba.model()
    model.ini({'p_ref_1':0.8},'xy_0_2.json')
    model.report_y()





if __name__ == "__main__":
    
    #test_pendulum_builder()
    test_leon_vsg_ll()