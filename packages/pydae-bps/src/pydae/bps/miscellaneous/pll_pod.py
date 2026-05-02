# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np

def add_pll(grid,data):
    """
    # auxiliar
    
    """

    backend = grid.backend
    bus_name = data['bus']

    name = bus_name
    if 'name' in data:
        name = data['name']

    sin = backend.sin
    cos = backend.cos

    # inputs
    V_s = backend.symbols(f"V_{bus_name}")
    theta_s = backend.symbols(f"theta_{bus_name}")
    omega_coi = backend.symbols(f"omega_coi") 

    # PLL
    # dynamic states
    theta_pll,xi_pll,omega_pll_f = backend.symbols(f'theta_pll_{name}, xi_pll_{name}, omega_pll_f_{name}')

    # algebraic statate
    rocof = backend.symbols(f'rocof_{name}')


    # parameters
    T_pll,K_f = backend.symbols(f'T_pll_{name}, K_f_{name}')
    K_p_pll,K_i_pll,K_theta_pll = backend.symbols(f'K_p_pll_{name}, K_i_pll_{name}, K_theta_pll_{name}')

    delta = theta_s # ideal PLL
    v_sD = V_s*sin(theta_s)  # v_si   e^(-j)
    v_sQ = V_s*cos(theta_s)  # v_sr

    v_sd_pll = v_sD * cos(theta_pll) - v_sQ * sin(theta_pll)   
    Domega_pll = K_p_pll*v_sd_pll + K_i_pll*xi_pll 
    omega_pll = Domega_pll + 1.0
    dtheta_pll = 2*np.pi*50*(omega_pll - omega_coi)*K_theta_pll 
    dxi_pll = v_sd_pll 
    domega_pll_f = 1/T_pll*(omega_pll - omega_pll_f)
    
    eq_rocof = rocof - domega_pll_f


    grid.dae['f'] += [dtheta_pll,dxi_pll,domega_pll_f]
    grid.dae['x'] += [ theta_pll, xi_pll, omega_pll_f]
    grid.dae['g'] += [eq_rocof]
    grid.dae['y_ini'] += [rocof]  
    grid.dae['y_run'] += [rocof] 
    grid.dae['params_dict'].update({f'K_p_pll_{name}':data['K_p_pll']})
    grid.dae['params_dict'].update({f'K_i_pll_{name}':data['K_i_pll']})
    grid.dae['params_dict'].update({f'T_pll_{name}':data['T_pll']}) 
    grid.dae['params_dict'].update({f'K_theta_pll_{name}':1.0}) 

    grid.dae['h_dict'].update({f"omega_pll_{name}":omega_pll})
    grid.dae['h_dict'].update({f"omega_pll_f_{name}":omega_pll_f})
    grid.dae['h_dict'].update({f"rocof_{name}":rocof})

    # PSS/POD
    x_wo  = backend.symbols(f"x_wo_pss_{name}")
    x_lead  = backend.symbols(f"x_lead_pss_{name}")

    z_wo  = backend.symbols(f"z_wo_pss_{name}")
    x_12  = backend.symbols(f"x_12_pss_{name}")
    x_34  = backend.symbols(f"x_34_pss_{name}")  

    T_wo = backend.symbols(f"T_wo_pss_{name}")  
    T_1 = backend.symbols(f"T_1_pss_{name}") 
    T_2 = backend.symbols(f"T_2_pss_{name}")
    T_3 = backend.symbols(f"T_3_pss_{name}") 
    T_4 = backend.symbols(f"T_4_pss_{name}")
    K_stab = backend.symbols(f"K_stab_{name}")
    V_lim = backend.symbols(f"V_lim_pss_{name}")
    v_s = backend.symbols(f"v_pss_{name}") 
    u_pll_probe = backend.symbols(f"u_pll_probe_{name}") 
    print('u_pll_probe')

    # auxiliar
    omega = omega_pll_f
    u_wo = K_stab*(omega - 1.0) + u_pll_probe
    v_pss_nosat = K_stab*((z_wo - x_lead)*T_1/T_2 + x_lead)
    
    z_wo = u_wo - x_wo
    z_12 = (z_wo - x_12)*T_1/T_2 + x_12
    z_34 = (z_12 - x_34)*T_3/T_4 + x_34

    v_s_nosat = z_34

    dx_wo =  (u_wo - x_wo)/T_wo  # washout state
    dx_12 =  (z_wo - x_12)/T_2      # lead compensator state
    dx_34 =  (z_12 - x_34)/T_4      # lead compensator state

    g_v_s = -v_s + backend.Piecewise((-V_lim,v_s_nosat<-V_lim),(V_lim,v_s_nosat>V_lim),(v_s_nosat,True))  
    
    grid.dae['f'] += [dx_wo,dx_12,dx_34]
    grid.dae['x'] += [ x_wo, x_12, x_34]
    grid.dae['g'] += [g_v_s]
    grid.dae['y_ini'] += [v_s]  
    grid.dae['y_run'] += [v_s] 
    grid.dae['params_dict'].update({str(T_wo):data['T_w']})
    grid.dae['params_dict'].update({str(T_1):data['T_1']})
    grid.dae['params_dict'].update({str(T_2):data['T_2']})
    grid.dae['params_dict'].update({str(T_3):data['T_3']})
    grid.dae['params_dict'].update({str(T_4):data['T_4']})
    grid.dae['params_dict'].update({str(K_stab):data['K_stab']})
    grid.dae['params_dict'].update({str(V_lim):0.1})    

    grid.dae['u_ini_dict'].update({str(u_pll_probe):0.0})
    grid.dae['u_run_dict'].update({str(u_pll_probe):0.0})    

def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bps import BpsBuilder
    import pydae.build_cffi as db
    import pytest
    import matplotlib.pyplot as plt

    grid = BpsBuilder('pll.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')


    import temp

    model = temp.model()
    model.Dt = 0.001
    model.decimation = 1

    v_ref_1 = 1.0
    model.ini({'K_theta_pll_1':10},'xy_0.json')
    model.report_x()
    model.report_y()
    model.report_z()
    model.report_params()

    # assert model.get_value('V_1') == pytest.approx(v_ref_1, rel=0.001)

    model.run(1.0,{})
    model.run(10,{'v_ref_1':1.1})
    # model.run(1.4,{})
    # model.run(5,{'fault_g_ref_1':0})

    # # model.run(1.11,{'fault_g_ref_1':0})
    # # model.run(2,{'fault_g_ref_1':0})

    model.post()
    # # assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)

    # model.ini({'p_m_1':0.5,'v_ref_1':1.0},'xy_0.json')
    # model.run(1.0,{})
    # model.run(15.0,{'v_ref_1':1.05})
    # model.post()


    fig,axes = plt.subplots()
    axes.plot(model.Time,model.get_values('omega_1'))
    axes.plot(model.Time,model.get_values('omega_pll_1'))
    # axes.plot(model.Time,model.get_values('omega_pll_f_1'))
    # axes.plot(model.Time,model.get_values('v_pss_1')+1)

    axes.grid()
    fig.savefig('pll_omegas.svg')

if __name__ == '__main__':

    #development()
    test()