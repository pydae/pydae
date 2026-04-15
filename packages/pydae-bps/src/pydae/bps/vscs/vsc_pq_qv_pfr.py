# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def vsc_pq_qv_pfr(grid,name,bus_name,data_dict):
    """
    VSC in p-q mode with possible Q-V droop and Primary Frequency Response 
    
    """

    # parameters 
    K_pfr = sym.Symbol(f"K_pfr_{name}", real=True)
    K_h = sym.Symbol(f"K_h_{name}", real=True)
    K_qv = sym.Symbol(f"K_qv_{name}", real=True)

    # inputs
    p_in = sym.Symbol(f"p_in_{name}", real=True)
    Dp_r = sym.Symbol(f"Dp_r_{name}", real=True)
    Dq_r = sym.Symbol(f"Dq_r_{name}", real=True)
    v_ref = sym.Symbol(f"v_ref_{name}", real=True)
    omega_pll_f = sym.Symbol(f"omega_pll_f_{name}", real=True)
    rocof_pll_f = sym.symbols(f'rocof_pll_f_{name}', real=True)
    omega_ref = 1.0
    V = sym.Symbol(f"V_{bus_name}", real=True)
        
    # dynamic states


    # algebraic states
    p_out = sym.Symbol(f"p_out_{name}", real=True)
    q_out = sym.Symbol(f"q_out_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    
    # auxiliar         
    Dp_pfr = K_pfr * (omega_ref - omega_pll_f)
    Dp_h = -K_h * rocof_pll_f
    Dq_qv  = K_qv * (v_ref - V)
    p_ref = p_in + Dp_r + Dp_pfr + Dp_h
    q_ref = Dq_qv + Dq_r
    
    # algebraic equations   
    p_out_sat = sym.Piecewise((0.0,p_ref<0.0),(p_in,p_ref>p_in),(p_ref,True))         
    q_out_max = (1**2 - p_out_sat**2)**0.5
    q_out_sat = sym.Piecewise((-q_out_max,q_ref<-q_out_max),(q_out_max,q_ref>q_out_max),(q_ref,True))     
    g_p_out = -p_out + p_out_sat
    g_q_out = -q_out + q_out_sat

    # dae 
    grid.dae['f'] += []
    grid.dae['x'] += []
    grid.dae['g'] += [g_p_out,g_q_out]
    grid.dae['y_ini'] += [p_out,q_out]  
    grid.dae['y_run'] += [p_out,q_out]  
    
    grid.dae['u_ini_dict'].update({f'{p_in}':data_dict['p_in']})
    grid.dae['u_run_dict'].update({f'{p_in}':data_dict['p_in']})
    grid.dae['u_ini_dict'].update({f'{Dp_r}':0.0})
    grid.dae['u_run_dict'].update({f'{Dp_r}':0.0})
    grid.dae['u_ini_dict'].update({f'{Dq_r}':0.0})
    grid.dae['u_run_dict'].update({f'{Dq_r}':0.0})            
    grid.dae['u_ini_dict'].update({f'{v_ref}':1.0})
    grid.dae['u_run_dict'].update({f'{v_ref}':1.0})                  
    # outputs
    grid.dae['h_dict'].update({f"{p_in}":p_in})
    grid.dae['h_dict'].update({f"{Dp_r}":Dp_r})
    grid.dae['h_dict'].update({f"{Dq_r}":Dq_r})
    
    # parameters            
    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 
    grid.dae['params_dict'].update({f"{K_pfr}":0.0}) 
    grid.dae['params_dict'].update({f"{K_h}":0.0}) 
    grid.dae['params_dict'].update({f"{K_qv}":0.0}) 

    p_W   = p_out*S_n
    q_var = q_out*S_n

    return p_W,q_var


def test_build():
    
    from pydae.bmapu import bmapu_builder
    import pytest

    grid = bmapu_builder.bmapu('vsc_pq_qv_pfr.hjson')
    #grid.checker()
    grid.verbose = False 
    grid.build('temp')

def test_ini():
    import pytest
    import temp

    model = temp.model()
    p_in = 0.9
    model.ini({'p_in_1':p_in},'xy_0.json')

    assert model.get_value('p_out_1') == pytest.approx(p_in, rel=0.001)
    assert model.get_value('q_out_1') == pytest.approx(0.0, rel=0.001)
    assert model.get_value('omega_pll_f_1') == pytest.approx(1.0, rel=0.001)


    # ### run:
    # model.Dt = 1.0
    # model.run(1.0,{})
    # model.run(0.5*3600,{'p_s_ref_1': 1.0}) # half an hour discharging
  
    # assert model.get_value('soc_1') == pytest.approx(0.25, rel=0.05)

    # model.run(1.0*3600,{'p_s_ref_1':-1.0}) # half an hour discharging
    # model.post()

    # assert model.get_value('soc_1') == pytest.approx(0.5, rel=0.05)



if __name__=='__main__':

    
    test_build()
    test_ini()





