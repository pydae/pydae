# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def pv_1(grid,name,bus_name,data_dict):
    '''

    parameters
    ----------

    S_n: nominal power in VA
    F_n: nominal frequency in Hz
    X_v: coupling reactance in pu (base machine S_n)
    R_v: coupling resistance in pu (base machine S_n)
    K_delta: if K_delta>0.0 current generator is converted to reference machine 
    K_alpha: alpha gain to obtain Domega integral 

    inputs
    ------

    alpha: RoCoF in pu if K_alpha = 1.0
    omega_ref: frequency in pu
    v_ref: internal voltage reference

    example
    -------

    "vscs": [{"type":"vsc_l","S_n":1e6,"F_n":50.0,"X_s":0.1,"R_s":0.01}]

    v_t = m*v_dc/sqrt(6)

    v_t_pu*v_b_ac = m*v_dc_pu*v_b_dc/sqrt(6)

    S_b = S_n
    U_b = U_n
    Omega_b = 2*np.pi*F_b
    Omega_rb = Omega_rn
    Tau_b = S_b/Omega_rb
    V_dc_b = sqrt(2)*U_b
    I_b = (np.sqrt(3)*U_b)
    I_dq_b = I_b*np.sqrt(2)
    V_dq_b = U_b*np.sqrt(2/3)

    V_dq_b and I_dq_b defined as below verify that: 
    3/2*I_dq_b*V_dq_b = 3/2*I_b*np.sqrt(2)*U_b*np.sqrt(2/3) = np.sqrt(3)*U_b*I_b

    Z_b = U_b**2/S_b
    Z_b = (V_dq_b/np.sqrt(2/3))**2/S_b = 3/2*V_dq_b**2/S_b
    L_m_b = Z_b/Omega_rb

    tau_r = 3/2*Phi*N_pp*i_mq

    tau_r_pu = (3/2*Phi*N_pp*i_mq_pu)*I_dq_b/Tau_b  = K_tau * i_mq_pu
    with K_tau = 3/2*Phi*N_pp*I_dq_b/Tau_b

    0 =               - R_s*i_md - omega_r*L_m*i_mq - v_md  
    0 = omega_r*Phi_m - R_s*i_mq + omega_r*L_m*i_md - v_mq 

    0 =               (- R_s*i_md_pu/V_dq_b - omega_r*L_m*i_mq_pu/V_dq_b - v_md_pu/I_dq_b)*I_dq_b*V_dq_b
    0 =               (- R_s_pu*i_md_pu/V_dq_b - omega_r*L_m_pu*i_mq_pu*Z_b/Omega_rb/(V_dq_b*Z_b) - v_md_pu/(I_dq_b*Z_b))*I_dq_b*V_dq_b*Z_b
    0 =               (- R_s_pu*i_md_pu/V_dq_b - omega_r_pu*L_m_pu*i_mq_pu/(V_dq_b) - v_md_pu/(I_dq_b*Z_b))*I_dq_b*V_dq_b*Z_b


    v_t_pu*v_b_ac = m*V_dc_b*sqrt(2)*U_b/sqrt(6)
    v_t_pu = m*V_dc_b*sqrt(2)/sqrt(6) = m*V_dc_b*sqrt(3)


    

    '''

    sin = sym.sin
    cos = sym.cos

    # inputs
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)
    m = sym.Symbol(f"m_{name}", real=True)
    theta_t = sym.Symbol(f"theta_t_{name}", real=True)
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)
    Dp_e_ref = sym.Symbol(f"Dp_e_ref_{name}", real=True)
    u_dummy = sym.Symbol(f"u_dummy_{name}", real=True)
      
    # dynamic states

    # algebraic states
    i_si = sym.Symbol(f"i_si_{name}", real=True)
    i_sr = sym.Symbol(f"i_sr_{name}", real=True)       
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)
    p_s_ref = sym.Symbol(f"p_s_ref_{name}", real=True)
    nu_w = sym.Symbol(f"nu_w_{name}", real=True)
    
    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)            
    X_s = sym.Symbol(f"X_s_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)
    A_l = sym.Symbol(f"A_l_{name}", real=True)
    B_l = sym.Symbol(f"B_l_{name}", real=True)
    C_l = sym.Symbol(f"C_l_{name}", real=True)

    # AUXILIAR
    Omega_b = 2*np.pi*F_n
    grid.dae['params_dict'].update({f"S_n_{name}":data_dict['S_n']})

    ## PV model
    I_sc,I_mpp,V_mpp,V_oc = sym.symbols(f'I_sc_{name},I_mpp_{name},V_mpp_{name},V_oc_{name}', real=True)
    N_s,K_vt,K_it = sym.symbols(f'N_s_{name},K_vt_{name},K_it_{name}', real=True)
    K_d,R_pv_s,R_pv_sh = sym.symbols(f'K_d_{name},R_pv_s_{name},R_pv_sh_{name}', real=True)

    temp_deg,irrad,i_pv,v_pv,p_pv = sym.symbols(f'temp_deg_{name},irrad_{name},i_pv_{name},v_pv_{name},p_pv_{name}', real=True)
    
    T_stc = 25 + 273.4
    E_c = 1.6022e-19 # Elementary charge
    Boltzmann = 1.3806e-23 # Boltzmann constant

    params_dict = {str(I_sc):data_dict['I_sc'],
                   str(I_mpp):data_dict['I_mpp'],
                   str(V_mpp):data_dict['V_mpp'],
                   str(V_oc):data_dict['V_oc'],
                   str(N_s):data_dict['N_s'],
                   str(K_vt):data_dict['K_vt'],
                   str(K_it):data_dict['K_it'],
                   str(R_pv_s):data_dict['R_pv_s'],
                   str(R_pv_sh):data_dict['R_pv_sh'],
                   str(K_d):data_dict['K_d'],
                   }
    
    temp_k = temp_deg +273.4

    N_ms,N_mp = sym.symbols(f'N_ms_{name},N_mp_{name}', real=True)

    I_rrad_sts = 1000

    S_n = data_dict['S_n']
    V_dc_b = data_dict['U_n']*np.sqrt(2)
    I_dc_b = S_n/V_dc_b
    # N_ms = data_dict['N_ms'] 
    # N_mp = data_dict['N_mp'] 
    v_pv = v_dc*V_dc_b/N_ms

    grid.dae['params_dict'].update({str(N_ms):data_dict['N_ms'],str(N_mp):data_dict['N_mp'] })


    # I_sc_t = I_sc*irrad/I_rrad_sts*(1 + K_it/100*(temp_k - T_stc))
    # I_0 = (I_sc - (V_oc_t - I_sc_t*R_pv_s)/R_pv_sh)*sym.exp(-V_oc_t/(N_s*V_t))

    # I_ph = (I_0*sym.exp(V_oc_t/(N_s*V_t)) + V_oc_t/R_pv_sh)*irrad/I_rrad_sts

    # eq_i_pv = -i_pv + I_ph - I_0 * (sym.exp((v_pv + i_pv*R_pv_s)/(N_s*V_t))-1)-(v_pv+i_pv*R_pv_s)/R_pv_sh 

    V_t = K_d*Boltzmann*T_stc/E_c
    V_oc_t = V_oc * (1+K_vt/100.0*( temp_k - T_stc))

    I_sc_t = I_sc*(1 + K_it/100*(temp_k - T_stc))
    I_0 = (I_sc_t - (V_oc_t - I_sc_t*R_pv_s)/R_pv_sh)*sym.exp(-V_oc_t/(N_s*V_t))
    I_d = I_0*(sym.exp((v_pv+i_pv*R_pv_s)/(V_t*N_s))-1)
    I_ph = I_sc_t*irrad/I_rrad_sts
    
    eq_i_pv = -i_pv + I_ph - I_d - (v_pv+i_pv*R_pv_s)/R_pv_sh 

    grid.dae['h_dict'].update({f"v_pv_{name}":v_pv})
    grid.dae['h_dict'].update({f"p_pv_{name}":v_pv*i_pv})

    grid.dae['u_ini_dict'].update({f'irrad_{name}':1000,f'temp_deg_{name}':25})  # input for the initialization problem
    grid.dae['u_run_dict'].update({f'irrad_{name}':1000,f'temp_deg_{name}':25})  # input for the running problem, its value is updated


    grid.dae['g'] += [eq_i_pv]
    grid.dae['y_ini'] += [  i_pv]  
    grid.dae['y_run'] += [  i_pv]  
    
    grid.dae['params_dict'].update(params_dict)

    grid.dae['xy_0_dict'].update({str(i_pv):data_dict['I_sc']})


    ## grid side VSC control  
    i_sd_pq_ref,i_sq_pq_ref,v_td_ref,v_tq_ref = sym.symbols(f'i_sd_pq_ref_{name},i_sq_pq_ref_{name},v_td_ref_{name},v_tq_ref_{name}', real=True)
    v_dc_ref,q_s_ref = sym.symbols(f'v_dc_ref_{name},q_s_ref_{name}', real=True)
    K_pdc,K_idc = sym.symbols(f'K_pdc_{name},K_idc_{name}', real=True)
    omega_coi = sym.symbols(f'omega_coi_{name}', real=True)
    mode,i_sd_i_ref,i_sq_i_ref = sym.symbols(f'mode_{name},i_sd_i_ref_{name},i_sq_i_ref_{name}', real=True)
    p_ppc_ref,q_ppc_ref = sym.symbols(f'p_ppc_ref_{name},q_ppc_ref_{name}', real=True)

    delta = theta_s # ideal PLL
    v_sD = V_s*sin(theta_s)  # v_si   e^(-j)
    v_sQ = V_s*cos(theta_s)  # v_sr
    v_sd = v_sD * cos(delta) - v_sQ * sin(delta)   
    v_sq = v_sD * sin(delta) + v_sQ * cos(delta)
    omega_s = omega_coi
    v_tD_ref = v_td_ref * cos(delta) + v_tq_ref * sin(delta)   
    v_tQ_ref =-v_td_ref * sin(delta) + v_tq_ref * cos(delta)    
    v_ti_ref = v_tD_ref
    v_tr_ref = v_tQ_ref   
    m_ref = sym.sqrt(v_tr_ref**2 + v_ti_ref**2)/v_dc
    theta_t_ref = sym.atan2(v_ti_ref,v_tr_ref) 
    
    i_sd_ref = i_sd_i_ref + i_sd_pq_ref
    i_sq_ref = i_sq_i_ref + i_sq_pq_ref

    p_s_vdc_ref = - K_pdc*(v_dc_ref - v_dc) 
    
    eq_p_s_ref = -p_s_ref + sym.Piecewise((p_ppc_ref,p_ppc_ref<p_s_vdc_ref),(p_s_vdc_ref,True))
    eq_i_sd_pq_ref  = i_sd_pq_ref*v_sd + i_sq_pq_ref*v_sq - p_s_ref  
    eq_i_sq_pq_ref  = i_sq_pq_ref*v_sd - i_sd_pq_ref*v_sq - q_s_ref - q_ppc_ref
    eq_v_td_ref  = v_td_ref - R_s*i_sd_ref - X_s*i_sq_ref - v_sd  
    eq_v_tq_ref  = v_tq_ref - R_s*i_sq_ref + X_s*i_sd_ref - v_sq 

    grid.dae['g'] += [eq_p_s_ref,eq_i_sd_pq_ref,eq_i_sq_pq_ref,eq_v_td_ref,eq_v_tq_ref]
    grid.dae['y_ini'] += [p_s_ref, i_sd_pq_ref, i_sq_pq_ref, v_td_ref, v_tq_ref]  
    grid.dae['y_run'] += [p_s_ref, i_sd_pq_ref, i_sq_pq_ref, v_td_ref, v_tq_ref] 
    grid.dae['u_ini_dict'].update({f'v_dc_ref_{name}':1.5})
    grid.dae['u_run_dict'].update({f'v_dc_ref_{name}':1.5})
    grid.dae['u_ini_dict'].update({f'q_s_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_s_ref_{name}':0.0})
    grid.dae['params_dict'].update({f'K_pdc_{name}':data_dict['K_pdc']}) 
    grid.dae['u_ini_dict'].update({f'mode_{name}':2})
    grid.dae['u_run_dict'].update({f'mode_{name}':2})
    grid.dae['u_ini_dict'].update({f'i_sd_i_ref_{name}':0})
    grid.dae['u_run_dict'].update({f'i_sd_i_ref_{name}':0})
    grid.dae['u_ini_dict'].update({f'i_sq_i_ref_{name}':0})
    grid.dae['u_run_dict'].update({f'i_sq_i_ref_{name}':0})

    grid.dae['u_ini_dict'].update({f'p_ppc_ref_{name}':2})
    grid.dae['u_run_dict'].update({f'p_ppc_ref_{name}':2})
    grid.dae['u_ini_dict'].update({f'q_ppc_ref_{name}':0})
    grid.dae['u_run_dict'].update({f'q_ppc_ref_{name}':0})

    grid.dae['xy_0_dict'].update({str(i_sr):0.1})
    grid.dae['h_dict'].update({f"i_sd_ref_{name}":i_sd_ref})
    grid.dae['h_dict'].update({f"i_sq_ref_{name}":i_sq_ref})
    grid.dae['h_dict'].update({f"p_s_vdc_ref_{name}":i_sq_ref})

    ## grid side VSC
    v_sr =  V_s*cos(theta_s)  # v_Q
    v_si =  V_s*sin(theta_s)  # v_D, e^(-j)
    i_tr = i_sr
    i_ti = i_si
    v_tr = v_tr_ref
    v_ti = v_ti_ref
    i_tdc = (i_ti*v_ti + i_tr*v_tr)/v_dc
    g_i_si = v_ti - R_s*i_si - X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr + X_s*i_si - v_sr 
    g_p_s  =  i_sr*v_sr + i_si*v_si - p_s  
    g_q_s  =  i_sr*v_si - i_si*v_sr - q_s 

    grid.dae['g'] += [g_i_si,g_i_sr,g_p_s,g_q_s]
    grid.dae['y_ini'] += [  i_si,  i_sr,  p_s,  q_s]  
    grid.dae['y_run'] += [  i_si,  i_sr,  p_s,  q_s]  
    
    grid.dae['u_ini_dict'].update({f'v_dc_ref_{name}':1.2})
    grid.dae['u_run_dict'].update({f'v_dc_ref_{name}':1.2})

    grid.dae['u_ini_dict'].update({f'{Dp_e_ref}':0.0})
    grid.dae['u_run_dict'].update({f'{Dp_e_ref}':0.0})

    grid.dae['u_ini_dict'].update({f'{u_dummy}':0.0})
    grid.dae['u_run_dict'].update({f'{u_dummy}':0.0})

    grid.dae['params_dict'].update({f"R_s_{name}":data_dict['R_s']}) 
    grid.dae['params_dict'].update({f"X_s_{name}":data_dict['X_s']}) 

    grid.dae['h_dict'].update({f"{str(p_s)}":p_s})
    grid.dae['h_dict'].update({f"{str(q_s)}":q_s}) 
    grid.dae['h_dict'].update({f"{str(i_si)}":i_si})
    grid.dae['h_dict'].update({f"{str(i_sr)}":i_sr})
    grid.dae['h_dict'].update({f"i_tdc_{name}":i_tdc})
    grid.dae['h_dict'].update({f"m_ref_{name}":m_ref})
    grid.dae['h_dict'].update({f"theta_t_ref_{name}":theta_t_ref}) 

    ## DC link   
    C_dc,i_dc = sym.symbols(f'C_dc_{name},i_dc_{name}', real=True)
    i_tdc = (i_ti*v_ti + i_tr*v_tr)/(v_dc + 1e-6)
    i_pv_pu = i_pv/I_dc_b*N_mp
    dv_dc = 0.5*(i_pv_pu - i_tdc)/(C_dc)

    grid.dae['f'] += [dv_dc]
    grid.dae['x'] += [ v_dc]
    grid.dae['params_dict'].update({f"C_dc_{name}":data_dict['C_dc']}) 
    grid.dae['xy_0_dict'].update({str(v_dc):1.2})
    grid.dae['h_dict'].update({f"i_pv_pu_{name}":i_pv_pu})
    grid.dae['h_dict'].update({f"i_pv_total_{name}":i_pv*N_mp})

    if 'A_l' in data_dict:
        grid.dae['params_dict'].update({f"A_l_{name}":data_dict['A_l']}) 
        grid.dae['params_dict'].update({f"B_l_{name}":data_dict['B_l']}) 
        grid.dae['params_dict'].update({f"C_l_{name}":data_dict['C_l']}) 
    else:
        grid.dae['params_dict'].update({f"A_l_{name}":0.005}) 
        grid.dae['params_dict'].update({f"B_l_{name}":0.005}) 
        grid.dae['params_dict'].update({f"C_l_{name}":0.005})         

    p_W   = p_s * S_n
    q_var = q_s * S_n
    return p_W,q_var


