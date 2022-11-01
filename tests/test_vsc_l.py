# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 23:38:58 2021

@author: jmmau
"""

import pytest
import numpy as np
from pydae.bmapu import bmapu_builder
 
 
def test_vsc_l_pq_builder():
    data = {
    "system":{"name":"smib_vsc_l","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},       
    "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
            {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}
            ],
    "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.01,"Bs_pu":1e-6,"S_mva":100.0}],
    "vscs": [{"bus":"1","type":"vsc_l","S_n":1e6,"F_n":50.0,"X_s":0.1,"R_s":0.01,
            "ctrl":{"type":"pq","p_s_ref":0.2,"q_s_ref":0.1}}],
    "genapes":[
        {"bus":"2","S_n":100e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]
    }
    
    grid = bmapu_builder.bmapu(data)
    #grid.checker()
    grid.uz_jacs = True
    grid.verbose = False
    grid.build('smib_vsc_l_pq')


def test_vsc_l():

    import smib_vsc_l_pq

    model = smib_vsc_l_pq.model()
    model.ini({},'xy_0.json')

    X_s = model.get_value('X_s_1')
    R_s = model.get_value('R_s_1')
    V_s = model.get_value('V_1')
    theta_s = model.get_value('theta_1')
    i_sr = model.get_value('i_sr_1')
    i_si = model.get_value('i_si_1')

    v_s = V_s*np.exp(1j*theta_s)
    i_s = i_sr + 1j*i_si

    v_sr = v_s.real   # X_m*cos(X_a)
    v_si = v_s.imag   # X_m*sin(X_a)

    i_sr = i_s.real   # X_m*cos(X_a)
    i_si = i_s.imag   # X_m*sin(X_a)

    # test the impedance are propoperly implemented
    v_ti_exp =  R_s*i_si + X_s*i_sr + v_si  
    v_tr_exp =  R_s*i_sr - X_s*i_si + v_sr 


    m = model.get_value('m_1')
    theta_t = model.get_value('theta_t_1')
    v_dc = model.get_value('v_dc_1')

    v_t_m = m*v_dc
    v_tr =  v_t_m*np.cos(theta_t)
    v_ti =  v_t_m*np.sin(theta_t)

    tol = 1e-8 

    assert np.abs(v_tr_exp - v_tr) < tol  # v_tr
    assert np.abs(v_ti_exp - v_ti) < tol  # v_ti

    # check powers are computed properly
    s_s_exp = v_s*np.conj(i_s)

    p_s_ri = i_sr*v_sr + i_si*v_si
    q_s_ri = i_sr*v_si - i_si*v_sr

    assert np.abs(s_s_exp.real - p_s_ri) < tol  # p_s
    assert np.abs(s_s_exp.imag - q_s_ri) < tol  # q_s


def test_vsc_l_pq():

    import smib_vsc_l_pq

    model = smib_vsc_l_pq.model()
    model.ini({'p_s_ref_1':0.5, 'q_s_ref_1':0.3},'xy_0.json')

    p_s_ref_1 = model.get_value('p_s_ref_1')
    q_s_ref_1 = model.get_value('q_s_ref_1')
    p_s_1 = model.get_value('p_s_1')
    q_s_1 = model.get_value('q_s_1')

    tol = 1e-8
    assert np.abs(p_s_ref_1 - p_s_1) < tol  # p_s
    assert np.abs(q_s_ref_1 - q_s_1) < tol  # q_s

    #delta = -0.1
    #v_sd = v_si * np.cos(delta) - v_sr * np.sin(delta)   
    #v_sq = v_si * np.sin(delta) + v_sr * np.cos(delta)
    #i_sd = i_si * np.cos(delta) - i_sr * np.sin(delta)   
    #i_sq = i_si * np.sin(delta) + i_sr * np.cos(delta)
#
    #v_td =  R_s*i_sd + X_s*i_sq + v_sd  
    #v_tq =  R_s*i_sq - X_s*i_sd + v_sq 
#
    #v_ti =  v_td * np.cos(delta) + v_tq * np.sin(delta)   
    #v_tr = -v_td * np.sin(delta) + v_tq * np.cos(delta) 
#
    #print(f'v_tr = {v_tr:0.3f}, v_ti = {v_ti:0.3f}')
#
    #
#
    #p_s_ri = i_sr*v_sr + i_si*v_si
    #q_s_ri = i_sr*v_si - i_si*v_sr
#
    #p_s_dq = i_sd*v_sd + i_sq*v_sq
    #q_s_dq = i_sq*v_sd - i_sd*v_sq
#
    #print(f'p_s =    {s_s.real:0.3f},    q_s = {s_s.imag:0.3f}')
    #print(f'p_s_ri = {p_s_ri:0.3f}, q_s_ri = {q_s_ri:0.3f}')
    #print(f'p_s_dq = {p_s_dq:0.3f}, q_s_dq = {q_s_dq:0.3f}')


if __name__ == "__main__":
    
    #test_pendulum_builder()
    pass 
    #test_leon_vsg_ll()