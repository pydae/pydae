# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from sympy import interpolating_spline
from pydae.utils.ss_num2sym import ss_num2sym

def bess_pq_ss(grid,name,bus_name,data_dict):
    """
    # auxiliar
    
    """

    # inputs
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)
    soc_ref = sym.Symbol(f"soc_ref_{name}", real=True)
    p_s_ppc = sym.Symbol(f"p_s_ppc_{name}", real=True)
    q_s_ppc = sym.Symbol(f"q_s_ppc_{name}", real=True)

        
    # dynamic states
    soc = sym.Symbol(f"soc_{name}", real=True)
    xi_soc = sym.Symbol(f"xi_soc_{name}", real=True)

    # algebraic states
    p_dc = sym.Symbol(f"p_dc_{name}", real=True)
    i_dc = sym.Symbol(f"i_dc_{name}", real=True)
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    E_kWh = sym.Symbol(f"E_kWh_{name}", real=True)
    soc_min = sym.Symbol(f"soc_min_{name}", real=True)
    soc_max = sym.Symbol(f"soc_max_{name}", real=True)
    A_loss = sym.Symbol(f"A_loss_{name}", real=True)
    B_loss = sym.Symbol(f"B_loss_{name}", real=True)
    C_loss = sym.Symbol(f"C_loss_{name}", real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)
    K_i = sym.Symbol(f"K_i_{name}", real=True)

    if 'socs' in data_dict:
        socs = np.array(data_dict['socs'])
        es = np.array(data_dict['es'])
        e_min = np.min(es)  # minimun voltage
        e_max = np.max(es)  # maximum voltage
        e_soc_order = 1
        if 'e_soc_order' in data_dict:
            e_soc_order = data_dict['e_soc_order']
        interpolation = interpolating_spline(e_soc_order, soc, socs, es)
        interpolation._args = tuple(list(interpolation._args) + [sym.functions.elementary.piecewise.ExprCondPair(e_max,True)])
        e = interpolation
        soc_ref_N = data_dict['soc_ref']
        e_ini = np.interp(soc_ref_N,socs,es)
    else:
        e = B_0 + B_1*soc
        e_ini = data_dict['B_0']

    R_bat = sym.Symbol(f"R_bat_{name}", real=True)

    # auxiliar
    H = E_kWh*1000.0*3600/S_n
    epsilon = soc_ref - soc
    p_soc = -(K_p*epsilon + K_i*xi_soc)

    ## Begin state space model
    A = np.array([[-10.0,   0.0],
                  [  0.0, -10.0]])
    B = np.array([[ 10.0,   0.0],
                  [  0.0,  10.0]])
    C = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    D = np.array([[0.0, 0.0],
                  [0.0, 0.0]])

    if 'A' in data_dict:
        A = np.array(data_dict['A_pq'])
        B = np.array(data_dict['B_pq'])
        C = np.array(data_dict['C_pq'])
        D = np.array(data_dict['D_pq'])

    sys = ss_num2sym(f'{name}',A,B,C,D)
    print(sys)
    sys['dx']= sys['dx'].replace(sys['u'][0,0],p_s_ppc)
    sys['dx']= sys['dx'].replace(sys['u'][1,0],q_s_ppc)
    sys['z_evaluated']= sys['z_evaluated'].replace(sys['u'][0,0],p_s_ppc)
    sys['z_evaluated']= sys['z_evaluated'].replace(sys['u'][1,0],q_s_ppc)
    p_s_ref = sys['z_evaluated'][0,0]
    q_s_ref = sys['z_evaluated'][1,0]
    print(sys)

    # p_s_ref = sym.Piecewise((p_s_ppc_d,p_s_ppc_d<p_mp),(p_mp,p_s_ppc_d>=p_mp))
    # q_s_ref = q_s_ppc_d
    ## End state space model

    p_s = sym.Piecewise((p_s_ref,(p_s_ref <= 0.0) & (soc<soc_max)),
                        (p_s_ref,(p_s_ref > 0.0) & (soc>soc_min)),
                        (0.0,True)) + p_soc
    #p_s = p_s_ref + p_soc
    q_s = q_s_ref
    s_s = (p_s**2 + q_s**2)**0.5
    i_s = s_s/V_s
    p_loss = A_loss*i_s**2 + B_loss*i_s + C_loss


    # dynamic equations    
    dsoc = 1/(H)*(-i_dc*e)   
    dxi_soc = epsilon     

    # algebraic equations   
    g_p_dc = p_s + p_loss - p_dc
    g_i_dc = v_dc*i_dc - p_dc
    g_v_dc = e - i_dc*R_bat - v_dc

    # dae 

    grid.dae['f'] += [dsoc, dxi_soc] + list(sys['dx'])
    grid.dae['x'] += [ soc,  xi_soc] + list(sys['x'])
    grid.dae['g'] += [g_p_dc,g_i_dc,g_v_dc]
    grid.dae['y_ini'] += [p_dc, i_dc, v_dc]  
    grid.dae['y_run'] += [p_dc, i_dc, v_dc]  
    
    grid.dae['u_ini_dict'].update({f'{p_s_ppc}':0.0})
    grid.dae['u_run_dict'].update({f'{p_s_ppc}':0.0})          
    grid.dae['u_ini_dict'].update({f'{q_s_ppc}':0.0})
    grid.dae['u_run_dict'].update({f'{q_s_ppc}':0.0})     
    grid.dae['u_ini_dict'].update({f'{soc_ref}':0.5})
    grid.dae['u_run_dict'].update({f'{soc_ref}':0.5})      

    # outputs
    grid.dae['h_dict'].update({f"p_loss_{name}":p_loss})
    grid.dae['h_dict'].update({f"i_s_{name}":i_s})
    grid.dae['h_dict'].update({f"e_{name}":e})
    grid.dae['h_dict'].update({f"i_dc_{name}":i_dc})
    grid.dae['h_dict'].update({f"p_s_{name}":p_s})    
    grid.dae['h_dict'].update({f"q_s_{name}":q_s})
    grid.dae['h_dict'].update({f"p_s_ref_{name}":p_s_ref})    
    grid.dae['h_dict'].update({f"q_s_ref_{name}":q_s_ref})
    grid.dae['h_dict'].update({f"p_s_ppc_{name}":p_s_ppc})    
    grid.dae['h_dict'].update({f"q_s_ppc_{name}":q_s_ppc})

    # parameters  
    grid.dae['params_dict'].update({f"{K_p}":1e-6}) 
    grid.dae['params_dict'].update({f"{K_i}":1e-6})  
    grid.dae['params_dict'].update({f"{soc_min}":0.0}) 
    grid.dae['params_dict'].update({f"{soc_max}":1.0})           
    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 
    grid.dae['params_dict'].update({f"{E_kWh}":data_dict['E_kWh']}) 
    if 'A_loss' in data_dict:
        A_loss_N = data_dict['A_loss']
        B_loss_N = data_dict['B_loss']
        C_loss_N = data_dict['C_loss']
    else:
        A_loss_N = 0.0001
        B_loss_N = 0.0
        C_loss_N = 0.0001

    grid.dae['params_dict'].update({f"{A_loss}":A_loss_N}) 
    grid.dae['params_dict'].update({f"{B_loss}":B_loss_N}) 
    grid.dae['params_dict'].update({f"{C_loss}":C_loss_N}) 

    if 'R_bat' in data_dict:
        R_bat_N = data_dict['R_bat']
    else:
        R_bat_N = 0.0

    grid.dae['params_dict'].update({f"R_bat_{name}":R_bat_N}) 

    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 
    grid.dae['params_dict'].update({f"{E_kWh}":data_dict['E_kWh']}) 

    grid.dae['xy_0_dict'].update({str(v_dc):1.2})
    grid.dae['xy_0_dict'].update({str(soc):0.5})
    grid.dae['xy_0_dict'].update({str(xi_soc):0.5})

    grid.dae['params_dict'].update(sys['params_dict'])

    p_W   = p_s*S_n
    q_var = q_s*S_n

    return p_W,q_var


def test_build():

    from pydae.bmapu import bmapu_builder
    import pytest

    grid = bmapu_builder.bmapu('bess_pq_ss.hjson')
    #grid.checker()
    grid.verbose = False 
    grid.build('temp')


def test_run():
    

    import temp

    model = temp.model()
    soc_ref = 0.5
    model.ini({'soc_ref_1':soc_ref},'xy_0.json')
    print('x:')
    model.report_x()
    print('y:')
    model.report_y()
    print('z:')
    model.report_z()
    # assert model.get_value('soc_1') == pytest.approx(soc_ref, rel=0.001)
    # assert model.get_value('p_dc_1') == pytest.approx(0.0)


    ### run:
    model.Dt = 1.0
    model.run(1.0,{})
    model.run(0.5*3600,{'p_s_ppc_1': 1.0}) # half an hour discharging
  
    # assert model.get_value('soc_1') == pytest.approx(0.25, rel=0.05)

    # model.run(1.0*3600,{'p_s_ref_1':-1.0}) # half an hour discharging
    # model.post()

    # assert model.get_value('soc_1') == pytest.approx(0.5, rel=0.05)



if __name__=='__main__':

    test_build()
    test_run()





