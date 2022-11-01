# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 23:38:58 2021

@author: jmmau
"""

import pytest
import numpy as np
from pydae.bmapu import bmapu_builder
import pydae.ssa as ssa
 
 
def test_vsc_l_vsg_ll_builder():

    zeta = 0.1 # 1.0/np.sqrt(2) 
    H_v = 4.0 
    WB = 2 *np.pi* 50
    R_v = 0.0
    X_v = 0.3

    Lt = X_v 
    P_max = 1/Lt
    fn = np.sqrt(WB*P_max/(2*H_v))/(2*np.pi)

    Dp = 0
    K_i = (2*np.pi*fn)**2/(WB*P_max)
    K_g = Dp*K_i
    K_p = (2*zeta*2*np.pi*fn - K_g)/(WB*P_max)

    T_q = 1.0/(2*np.pi*10/2)
    K_q = (1.0 - 0.0)/(1.05 - 0.95)
    K_i_q = 1e-6

    theta_red = 3.0/180*np.pi
    V = 1.0
    p_ref = 0.9
    q_ref = 0.434616
    v_ref = 1.0
    T_q = 1.0/(2*np.pi*10/2)


    data = {
    "system":{"name":"smib_vsc_l","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},       
    "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
            {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}
            ],
    "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.01,"Bs_pu":1e-6,"S_mva":100.0}],
    "vscs": [{"bus":"1","type":"vsc_l","S_n":10e6,"F_n":50.0,"X_s":0.05,"R_s":0.005,
          "ctrl":{"type":"leon_vsg_ll","K_delta":0.0,"F_n":50.0,
                  "R_v":R_v,"X_v":X_v,"K_p":K_p,"K_i":K_i,"K_g":K_g,"K_q":K_q,
                  "T_q":T_q,"K_p_v":1e-6,"K_i_v":1e-6}}],
    "genapes":[
        {"bus":"2","S_n":100e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]
    }
    
    grid = bmapu_builder.bmapu(data)
    #grid.checker()
    grid.uz_jacs = True
    grid.verbose = False
    grid.build('smib_vsc_l_vsg_ll')


def test_vsc_l_vsg_ll():

    import smib_vsc_l_vsg_ll

    model = smib_vsc_l_vsg_ll.model()
    zeta = 0.2 # 1.0/np.sqrt(2) 
    H_v = 4.0 
    WB = 2 *np.pi* 50;
    R_v = 0.0
    X_v = 0.3

    Lt = X_v 
    P_max = 1/Lt
    fn = np.sqrt(WB*P_max/(2*H_v))/(2*np.pi)

    Dp = 0;
    K_i = (2*np.pi*fn)**2/(WB*P_max);
    K_g = Dp*K_i;
    K_p = (2*zeta*2*np.pi*fn - K_g)/(WB*P_max);

    T_q = 1.0/(2*np.pi*10/2)
    K_q = (1.0 - 0.0)/(1.05 - 0.95)
    K_i_q = 1e-6

    theta_red = 3.0/180*np.pi
    V = 1.0
    p_ref = 0.9
    q_ref = 0.434616
    v_ref = 1.0
    T_q = 1.0/(2*np.pi*10/2)

    p_l_1 = 0.5
    v_ref_1 = 1.01
    X_s = 0.1
    R_s = 0.0
    params = {"S_n_1":10e6,'p_l_1':p_l_1,'v_ref_1':v_ref_1,'X_s_1':X_s,'R_s_1':R_s,
            'b_1_2':-2,
            'X_v_1': X_v-X_s, 'R_v_1':R_v,
            'T_q_1':T_q,'K_q_1':K_q,
            'K_i_1':K_i,'K_g_1':K_g,'K_p_1':K_p}

    model.ini(params,'xy_0.json')



    p_s_1 = model.get_value('p_s_1')
    q_s_1 = model.get_value('q_s_1')
    V_1 = model.get_value('V_1')

    # Check active and voltage references are achived
    tol = 1e-8
    assert np.abs(p_l_1 - p_s_1) < tol  # p_s
    assert np.abs(v_ref_1 - V_1) < tol  # V_s

    # check dq magnitudes are correct
    delta = model.get_value('delta_1')
    q_s = model.get_value('q_s_1')
    V_s = model.get_value('V_1')
    i_sr = model.get_value('i_sr_1')  
    i_si = model.get_value('i_si_1')  

    #v_sd = v_si * np.cos(delta) - v_sr * np.sin(delta)   
    #v_sq = v_si * np.sin(delta) + v_sr * np.cos(delta)
    i_sd_exp = i_si * np.cos(delta) - i_sr * np.sin(delta)   
    i_sq_exp = i_si * np.sin(delta) + i_sr * np.cos(delta)

    i_sd = model.get_value('i_sd_1')  
    i_sq = model.get_value('i_sq_1') 

    assert np.abs(i_sd_exp - i_sd) < tol  # i_sd
    assert np.abs(i_sq_exp - i_sq) < tol  # i_sq

    # check virtual impedance
    e_vd = 0.0
    e_vq = model.get_value('e_vq_1')
    R_v = model.get_value('R_v_1')
    X_v = model.get_value('X_v_1')
    v_td_exp = e_vd - R_v*i_sd_exp - X_v*i_sq_exp   
    v_tq_exp = e_vq - R_v*i_sq_exp + X_v*i_sd_exp   

    v_td = model.get_value('v_td_ref_1')
    v_tq = model.get_value('v_tq_ref_1')

    assert np.abs(v_td_exp - v_td) < tol  # v_td
    assert np.abs(v_tq_exp - v_tq) < tol  # v_tq

    # check control output
    v_dc = model.get_value('v_dc_1')
    m_exp = np.sqrt(v_td_exp**2 + v_tq_exp**2)/v_dc

    v_ti_exp =  v_td_exp * np.cos(delta) + v_tq_exp * np.sin(delta)   
    v_tr_exp = -v_td_exp * np.sin(delta) + v_tq_exp * np.cos(delta) 

    theta_t_exp = np.arctan2(v_ti_exp,v_tr_exp)
    theta_t = model.get_value('theta_t_1')

    assert np.abs(theta_t_exp - theta_t) < tol  # v_tq


    ssa.A_eval(model)
    damp = ssa.damp_report(model)
    assert  damp.sort_values('Damp').round(2)['Damp'][0] > 0.0
    assert  np.abs(damp.sort_values('Damp').round(2)['Damp'][0] - zeta) < 1e-2

if __name__ == "__main__":
    
    #test_pendulum_builder()
    pass 
    #test_leon_vsg_ll()