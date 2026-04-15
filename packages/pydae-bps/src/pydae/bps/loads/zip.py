# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def load_zip(grid,name,bus_name,data_dict):
    """
    # auxiliar
    
    """

    V = sym.Symbol(f"V_{bus_name}", real=True)
        

    # algebraic states
    p_z = sym.Symbol(f"p_z_{name}", real=True)
    q_z = sym.Symbol(f"q_z_{name}", real=True)
    g_load = sym.Symbol(f"g_load_{name}", real=True)
    b_load = sym.Symbol(f"b_load_{name}", real=True)

    # parameters
    
    # auxiliar
       
    # dynamic equations            

    # algebraic equations   
    eq_p_z = -p_z + g_load*V**2
    eq_q_z = -q_z + b_load*V**2

    # dae 
    grid.dae['f'] += []
    grid.dae['x'] += []
    grid.dae['g'] += [eq_p_z,eq_q_z]
    grid.dae['y_ini'] += [g_load,b_load]  
    grid.dae['y_run'] += [p_z,q_z]  
    
    grid.dae['u_ini_dict'].update({f'{str(p_z)}':data_dict['p_mw']*1e6})
    grid.dae['u_ini_dict'].update({f'{str(q_z)}':data_dict['q_mvar']*1e6})
    grid.dae['u_run_dict'].update({f'{str(g_load)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(b_load)}':0.0})            
               
    # outputs

    
    # parameters            

    p_W   = p_z*S_n
    q_var = q_z*S_n

    return p_W,q_var


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import pytest

    grid = bmapu('zip.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')

    import temp

    model = temp.model()

    model.ini({},'xy_0.json')

    model.report_y()


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
    test()