# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def pmsm_1(grid,name,bus_name,data_dict):
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
    K_pow = sym.Symbol(f"K_pow_{name}", real=True)
    C_1,C_2,C_3 = sym.symbols(f"C_1_{name},C_2_{name},C_3_{name}", real=True)
    C_4,C_5,C_6 = sym.symbols(f"C_4_{name},C_5_{name},C_6_{name}", real=True)
    Nu_w_b,Lam_b = sym.symbols(f"Nu_w_b_{name},Lam_b_{name}", real=True)
    K_tr,D_tr,Omega_t_b = sym.symbols(f"K_tr_{name},D_tr_{name},Omega_t_b_{name}", real=True)
    C_p_b,T_beta,T_mppt,K_mppt = sym.symbols(f"C_p_b_{name},T_beta_{name},T_mppt_{name},K_mppt_{name}", real=True)
    K_p_beta = sym.Symbol(f"K_p_beta_{name}", real=True)
    K_i_beta = sym.Symbol(f"K_i_beta_{name}", real=True)
    H_t,H_r= sym.symbols(f"H_t_{name},H_r_{name}", real=True)
    K_tr,D_tr= sym.symbols(f"K_tr_{name},D_tr_{name}", real=True)
    params_list = ['S_n','F_n','X_s','R_s']
    
    mode = ['vsc_g','v_dc']
    mode = ['aero','pitch','mech','mppt','vsc_m','v_dc','vsc_g']

    # AUXILIAR
    Omega_b = 2*np.pi*F_n
    grid.dae['params_dict'].update({f"S_n_{name}":data_dict['S_n']})

    ## Aerodinamic
    theta_tr = sym.Symbol(f"theta_tr_{name}", real=True)
    omega_t  = sym.Symbol(f"omega_t_{name}", real=True)
    omega_r  = sym.Symbol(f"omega_r_{name}", real=True)
    beta     = sym.Symbol(f"beta_{name}", real=True)
    p_w_mppt_lpf = sym.Symbol(f"p_w_mppt_lpf_{name}", real=True)
    lam = Lam_b*(omega_t/Omega_t_b)/(nu_w/Nu_w_b) # pu
    inv_lam_i =  1/(lam + 0.08*beta) - 0.035/(beta**3 + 1.0)   
    c_p = C_1*(C_2*inv_lam_i - C_3*beta - C_4)*sym.exp(-C_5*inv_lam_i) + C_6*lam 
    c_p_pu = c_p/C_p_b # (pu)
    p_w_nosat = K_pow*c_p_pu*(nu_w/Nu_w_b)**3 
    p_w = sym.Piecewise((0.0,p_w_nosat<0.0), (p_w_nosat,True)) 

    ## MPPT
    K_mppt3  = sym.Symbol(f"K_mppt3_{name}", real=True)
    p_ref_ext = sym.Symbol(f"p_ref_ext_{name}", real=True)
    Lam_opt = Lam_b
    beta_mppt = 0.0
    nu_w_mppt = Nu_w_b*(omega_t/Omega_t_b)   
    inv_lam_i_mppt =  1/(Lam_opt) - 0.035   
    c_p_mppt = C_1*(C_2*inv_lam_i_mppt - C_4)*sym.exp(-C_5*inv_lam_i_mppt) + C_6*Lam_opt 
    c_p_mppt_pu = c_p_mppt/C_p_b # (pu)
    p_w_mppt_ref = K_pow*c_p_mppt_pu*(nu_w_mppt/Nu_w_b)**3
    omega_r_th = 0.2
    test_1 = (np.abs(Dp_e_ref)>0.01) & (omega_r>omega_r_th)
    p_w_mppt = sym.Piecewise((p_w_mppt_lpf,test_1), (p_w_mppt_ref,True))
    K_mppt = sym.Piecewise((0.0,test_1), (1.0,True))
    Dp_e = sym.Piecewise((Dp_e_ref,test_1), (Dp_e_ref * (1 + 50*(omega_r - omega_r_th)),True))
    p_w_mmpt_ref = p_w_mppt_lpf #+ Dp_e   test
    p_w_mmpt_ref = K_mppt3*omega_t**3 
    p_m_ref = sym.Piecewise((p_w_mmpt_ref+p_ref_ext,p_ref_ext<0), (p_w_mmpt_ref,True))
    dp_w_mppt_lpf  = K_mppt/T_mppt*(p_m_ref - p_w_mppt_lpf)
    if 'aero' in mode and 'mppt' in mode:
        grid.dae['f'] += [dp_w_mppt_lpf]
        grid.dae['x'] += [ p_w_mppt_lpf]
    grid.dae['params_dict'].update({f"K_mppt3_{name}":0.4})
    grid.dae['u_ini_dict'].update({f'p_ref_ext_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_ref_ext_{name}':0.0})  

    ## Pitch
    xi_beta  = sym.Symbol(f"xi_beta_{name}", real=True)
    beta_ext  = sym.Symbol(f"beta_ext_{name}", real=True)
    tau_r  = sym.Symbol(f"tau_r_{name}", real=True)
    Omega_r_max  = sym.Symbol(f"Omega_r_max_{name}", real=True)
    Omega_r_b = Omega_t_b
    epsilon_beta_nosat = omega_r - Omega_r_max
    epsilon_beta = sym.Piecewise((0.0,epsilon_beta_nosat<0),(epsilon_beta_nosat,True)) # test
    beta_ref = K_p_beta*epsilon_beta + K_i_beta*xi_beta
    p_m = tau_r*omega_r
    dxi_beta  = epsilon_beta - xi_beta*1e-8 # test
    #dxi_beta  = epsilon_beta_nosat - xi_beta*1e-8 # test
    dbeta     = 1.0/T_beta*(beta_ref + beta_ext - beta)

    grid.dae['u_ini_dict'].update({f'beta_ext_{name}':0.0})
    grid.dae['u_run_dict'].update({f'beta_ext_{name}':0.0})  
    
    grid.dae['params_dict'].update({f"Omega_r_max_{name}":1.2})

    if 'pitch' in mode:
        grid.dae['f'] += [dxi_beta,dbeta]
        grid.dae['x'] += [ xi_beta, beta]
    else:
        grid.dae['u_ini_dict'].update({f'beta_{name}':0.0})
        grid.dae['u_run_dict'].update({f'beta_{name}':0.0})          

    if not 'aero' in mode:
        p_w  = sym.Symbol(f"p_w_{name}", real=True)
        grid.dae['u_ini_dict'].update({f'p_w_{name}':0.0})
        grid.dae['u_run_dict'].update({f'p_w_{name}':0.0})   

    grid.dae['h_dict'].update({f"nu_w_{name}":nu_w})
    grid.dae['params_dict'].update({f"T_beta_{name}":2.0})
    grid.dae['params_dict'].update({f"K_p_beta_{name}":100.0})
    grid.dae['params_dict'].update({f"K_i_beta_{name}":1.0})

    ## Mechanical system: 2 mass equivalent    
    dtheta_tr = omega_t - omega_r - u_dummy
    domega_t  = 1.0/(2*H_t)*(p_w - K_tr*theta_tr - D_tr*(omega_t-omega_r))
    domega_r  = 1.0/(2*H_r)*(K_tr*theta_tr + D_tr*(omega_t-omega_r) - p_m)

    if 'mech' in mode:
        grid.dae['f'] += [dtheta_tr,domega_t,domega_r]
        grid.dae['x'] += [ theta_tr, omega_t, omega_r]
    else:
        grid.dae['u_ini_dict'].update({f'omega_r_{name}':1.2})
        grid.dae['u_run_dict'].update({f'omega_r_{name}':1.2})        

    p_w_0 = 0.8
    omega_t_0,omega_r_0,beta_0 = 1.2,1.2,0.0
    grid.dae['xy_0_dict'].update({str(theta_tr):p_w_0})
    grid.dae['xy_0_dict'].update({str(omega_t):omega_t_0})
    grid.dae['xy_0_dict'].update({str(omega_r):omega_r_0})
    grid.dae['xy_0_dict'].update({str(beta):beta_0})
    H_t, H_r = data_dict['H_t'],data_dict['H_r']
    grid.dae['params_dict'].update({f"H_t_{name}":H_t})
    grid.dae['params_dict'].update({f"H_r_{name}":H_r})
    w_tr = 2*np.pi*data_dict['w_tr']
    d_tr = 2*np.pi*data_dict['d_tr']
    K_tr = 2*H_r*H_t*w_tr**2/(H_r + H_t)
    D_tr =  4*np.pi*H_r*H_t*d_tr*w_tr/(H_r + H_t)
    grid.dae['params_dict'].update({f"K_tr_{name}":K_tr})
    grid.dae['params_dict'].update({f"D_tr_{name}":D_tr}) 
    beta_0 = 0.0  # initial input
    p_ref = 0.5*0.657/0.9 # initial input
    inv_lam_i =  1/(lam + 0.08*beta_0) - 0.035/(beta_0**3 + 1.0)   
    # c_p = C_1*(C_2*inv_lam_i - C_3*beta_0 - C_4)*np.exp(-C_5*inv_lam_i) + C_6*lam 
    # c_p_pu = c_p/C_p_b # (pu)
    # #p_m_ref = K_p*c_p*(nu_w/nu_w_b)**3
    # nu_w_0 = (p_ref/(K_pow*c_p_pu))**(1.0/3.0)*Nu_w_b
    # omega_t_0 = lam*(nu_w_0/Nu_w_b)/Lam_b*Omega_t_b


    ## machine + VSC control
    v_dc,p_m_ref = sym.symbols(f"v_dc_{name},p_m_ref_{name}", real=True)   
    i_md,i_mq,v_md,v_mq,tau_r = sym.symbols(f'i_md_{name},i_mq_{name},v_md_{name},v_mq_{name},tau_r_{name}', real=True)
    R_m,L_m,Phi_m = sym.symbols(f'R_m_{name},L_m_{name},Phi_m_{name}', real=True)
    omega_e = omega_r # from mechanical system
    

    i_mq_ref,i_md_ref,p_r = sym.symbols(f'i_mq_ref_{name},i_md_ref_{name},p_r_{name}', real=True)

    if 'mppt' in mode:
        # K_w_mppt = sym.Symbol(f'K_w_mppt_{name}', real=True)
        # p_m_ref = p_w_mmpt_ref # from mppt 
        # omega_r_ref = nu_w*K_w_mppt
        # grid.dae['params_dict'].update({f"K_w_mppt_{name}":8.0}) 
        p_m_ref = p_w_mppt_lpf + p_r
    else:
        if not 'aero' in mode:
            grid.dae['u_ini_dict'].update({f'p_w_{name}':0.0})
            grid.dae['u_run_dict'].update({f'p_w_{name}':0.0})   
        if not 'mech' in mode:
            grid.dae['u_ini_dict'].update({f'p_m_ref_{name}':0.0})
            grid.dae['u_run_dict'].update({f'p_m_ref_{name}':0.0})   
        else:
            omega_r_ref,K_omega_r = sym.symbols(f'omega_r_ref_{name},K_omega_r_{name}',real=True)
            K_w_mppt = sym.Symbol(f'K_w_mppt_{name}', real=True)
            #p_m_ref = p_w_mmpt_ref # from mppt 
            #omega_r_ref = nu_w*K_w_mppt
            grid.dae['params_dict'].update({f"K_w_mppt_{name}":8.0}) 
            #p_m_ref = K_omega_r*(omega_r_ref - omega_r)
            grid.dae['params_dict'].update({f"K_omega_r_{name}":1.0}) 
            grid.dae['u_ini_dict'].update({f'omega_r_ref_{name}':1.0})
            grid.dae['u_run_dict'].update({f'omega_r_ref_{name}':1.0})      
            grid.dae['u_ini_dict'].update({f'u_dummy_{name}':0.0})
            grid.dae['u_run_dict'].update({f'u_dummy_{name}':0.0})   


    g_i_mq_ref  = Phi_m*i_mq_ref*omega_r - p_m_ref  
    g_v_md = -L_m*i_mq_ref*omega_e - R_m*i_md_ref - v_md
    g_v_mq =  L_m*i_md_ref*omega_e + Phi_m*omega_e - R_m*i_mq_ref - v_mq

    if 'vsc_m' in mode:
        grid.dae['g'] += [g_i_mq_ref,g_v_md,g_v_mq]
        grid.dae['y_ini'] += [ i_mq_ref, v_md, v_mq]
        grid.dae['y_run'] += [ i_mq_ref, v_md, v_mq]
        grid.dae['u_ini_dict'].update({f'i_md_ref_{name}':0.0})
        grid.dae['u_run_dict'].update({f'i_md_ref_{name}':0.0})
        grid.dae['u_ini_dict'].update({f'p_r_{name}':0.0})
        grid.dae['u_run_dict'].update({f'p_r_{name}':0.0}) 
    ## machine + VSC in per unit
    i_md,i_mq,v_md,v_mq,tau_r = sym.symbols(f'i_md_{name},i_mq_{name},v_md_{name},v_mq_{name},tau_r_{name}', real=True)
    omega_e = omega_r # from mechanical system

    g_i_md = -L_m*i_mq*omega_e - R_m*i_md - v_md
    g_i_mq =  L_m*i_md*omega_e + Phi_m*omega_e - R_m*i_mq - v_mq
    g_tau_r = Phi_m*i_mq - tau_r

    grid.dae['h_dict'].update({f"p_m_{name}":i_md*v_md + i_mq*v_mq})
    grid.dae['h_dict'].update({f"q_m_{name}":i_mq*v_md - i_md*v_mq}) 


    if 'vsc_m' in mode:
        grid.dae['g'] += [g_i_md,g_i_mq,g_tau_r]
        grid.dae['y_ini'] += [i_md,i_mq,tau_r]
        grid.dae['y_run'] += [i_md,i_mq,tau_r]
        grid.dae['params_dict'].update({f"R_m_{name}":data_dict['R_m']}) 
        grid.dae['params_dict'].update({f"L_m_{name}":data_dict['L_m']}) 
        grid.dae['params_dict'].update({f"Phi_m_{name}":data_dict['Phi_m']}) 

    ## grid side VSC control  
    i_sd_ref,i_sq_ref,v_td_ref,v_tq_ref = sym.symbols(f'i_sd_ref_{name},i_sq_ref_{name},v_td_ref_{name},v_tq_ref_{name}', real=True)
    v_dc_ref,q_s_ref = sym.symbols(f'v_dc_ref_{name},q_s_ref_{name}', real=True)
    K_pdc,K_idc = sym.symbols(f'K_pdc_{name},K_idc_{name}', real=True)
    omega_coi = sym.symbols(f'omega_coi_{name}', real=True)
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
    
    eq_p_s_ref = -p_s_ref - K_pdc*(v_dc_ref - v_dc) + i_md*v_md + i_mq*v_mq
    eq_i_sd_ref  = i_sd_ref*v_sd + i_sq_ref*v_sq - p_s_ref  
    eq_i_sq_ref  = i_sq_ref*v_sd - i_sd_ref*v_sq - q_s_ref
    eq_v_td_ref  = v_td_ref - R_s*i_sd_ref - X_s*i_sq_ref - v_sd  
    eq_v_tq_ref  = v_tq_ref - R_s*i_sq_ref + X_s*i_sd_ref - v_sq 

    if 'vsc_g' in mode:
        grid.dae['g'] += [eq_p_s_ref,eq_i_sd_ref,eq_i_sq_ref,eq_v_td_ref,eq_v_tq_ref]
        grid.dae['y_ini'] += [p_s_ref, i_sd_ref, i_sq_ref, v_td_ref, v_tq_ref]  
        grid.dae['y_run'] += [p_s_ref, i_sd_ref, i_sq_ref, v_td_ref, v_tq_ref] 
        grid.dae['u_ini_dict'].update({f'v_dc_ref_{name}':1.5})
        grid.dae['u_run_dict'].update({f'v_dc_ref_{name}':1.5})
        grid.dae['u_ini_dict'].update({f'q_s_ref_{name}':0.0})
        grid.dae['u_run_dict'].update({f'q_s_ref_{name}':0.0})
        grid.dae['params_dict'].update({f'K_pdc_{name}':data_dict['K_pdc']}) 
        grid.dae['xy_0_dict'].update({str(i_sr):0.1})

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

    if 'vsc_g' in mode:

        grid.dae['g'] += [g_i_si,g_i_sr,g_p_s,g_q_s]
        grid.dae['y_ini'] += [  i_si,  i_sr,  p_s,  q_s]  
        grid.dae['y_run'] += [  i_si,  i_sr,  p_s,  q_s]  
        
        grid.dae['u_ini_dict'].update({f'v_dc_ref_{name}':1.2})
        grid.dae['u_run_dict'].update({f'v_dc_ref_{name}':1.2})

        grid.dae['u_ini_dict'].update({f'{Dp_e_ref}':0.0})
        grid.dae['u_run_dict'].update({f'{Dp_e_ref}':0.0})

        grid.dae['u_ini_dict'].update({f'{nu_w}':10.0})
        grid.dae['u_run_dict'].update({f'{nu_w}':10.0})

        grid.dae['u_ini_dict'].update({f'{u_dummy}':0.0})
        grid.dae['u_run_dict'].update({f'{u_dummy}':0.0})

        grid.dae['params_dict'].update({f"R_s_{name}":data_dict['R_s']}) 
        grid.dae['params_dict'].update({f"X_s_{name}":data_dict['X_s']}) 

        grid.dae['h_dict'].update({f"{str(p_s)}":p_s})
        grid.dae['h_dict'].update({f"{str(q_s)}":q_s}) 
        grid.dae['h_dict'].update({f"{str(i_si)}":i_si})
        grid.dae['h_dict'].update({f"{str(i_sr)}":i_sr})
        grid.dae['h_dict'].update({f"i_tdc_{name}":i_tdc})

    ## DC link   
    C_dc,i_dc = sym.symbols(f'C_dc_{name},i_dc_{name}', real=True)
    i_mdc = (i_md*v_md + i_mq*v_mq)/(v_dc + 1e-6)
    i_tdc = (i_ti*v_ti + i_tr*v_tr)/(v_dc + 1e-6)

    if 'vsc_m' in mode: 
        dv_dc = 0.5*(i_mdc - i_tdc)/(C_dc)
        grid.dae['h_dict'].update({f"i_mdc_{name}":i_mdc})
    if 'vsc_m' in mode and 'vsc_g' in mode: 
        dv_dc = 0.5*(i_mdc - i_tdc)/(C_dc)

    if not 'vsc_m' in mode and 'vsc_g' in mode: 
        dv_dc = 0.5*(i_dc - i_tdc)/(C_dc)
        grid.dae['u_ini_dict'].update({f'i_dc_{name}':0.0})
        grid.dae['u_run_dict'].update({f'i_dc_{name}':0.0})

    if 'v_dc' in mode:

        grid.dae['f'] += [dv_dc]
        grid.dae['x'] += [ v_dc]
        grid.dae['params_dict'].update({f"C_dc_{name}":data_dict['C_dc']}) 
        grid.dae['xy_0_dict'].update({str(v_dc):1.5})

    else:
        grid.dae['u_ini_dict'].update({f'v_dc_{name}':1.5})
        grid.dae['u_run_dict'].update({f'v_dc_{name}':1.5})



    # outputs


    # grid.dae['h_dict'].update({f"p_ac_{name}":p_ac})
    # grid.dae['h_dict'].update({f"p_dc_{name}":p_dc})
    # grid.dae['h_dict'].update({f"i_d_{name}":i_d})
    # grid.dae['h_dict'].update({f"u_dummy_{name}":u_dummy})
    # grid.dae['h_dict'].update({f"p_m_mppt_lpf_{name}":p_m_mppt_lpf})
    # grid.dae['h_dict'].update({f"p_t_{name}":p_m})
    # grid.dae['h_dict'].update({f"K_mppt_{name}":K_mppt})
    # grid.dae['h_dict'].update({f"lam_{name}":lam})
    # grid.dae['h_dict'].update({f"c_p_{name}":c_p})
    # grid.dae['h_dict'].update({f"c_p_pu_{name}":c_p_pu})
    # grid.dae['h_dict'].update({f"nu_w_mppt_{name}":nu_w_mppt})
    # grid.dae['h_dict'].update({f"p_m_{name}":p_m})
    

    # for item in params_list:       
    #     grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]}) 

    if 'A_l' in data_dict:
        grid.dae['params_dict'].update({f"A_l_{name}":data_dict['A_l']}) 
        grid.dae['params_dict'].update({f"B_l_{name}":data_dict['B_l']}) 
        grid.dae['params_dict'].update({f"C_l_{name}":data_dict['C_l']}) 
    else:
        grid.dae['params_dict'].update({f"A_l_{name}":0.005}) 
        grid.dae['params_dict'].update({f"B_l_{name}":0.005}) 
        grid.dae['params_dict'].update({f"C_l_{name}":0.005})         

    C_1 = 0.5176
    C_2 = 116
    C_3 = 0.4
    C_4 = 5.0
    C_5 = 21 
    C_6 = 0.0068
    Nu_w_b = 12
    Lam_b = 8.1
    Omega_t_b = 1.2
    K_pow = 1.0
    grid.dae['params_dict'].update({f"C_1_{name}":C_1})
    grid.dae['params_dict'].update({f"C_2_{name}":C_2})
    grid.dae['params_dict'].update({f"C_3_{name}":C_3})
    grid.dae['params_dict'].update({f"C_4_{name}":C_4})
    grid.dae['params_dict'].update({f"C_5_{name}":C_5})
    grid.dae['params_dict'].update({f"C_6_{name}":C_6})
    grid.dae['params_dict'].update({f"Nu_w_b_{name}":Nu_w_b})
    grid.dae['params_dict'].update({f"Lam_b_{name}":Lam_b})
    grid.dae['params_dict'].update({f"Omega_t_b_{name}":Omega_t_b})
    grid.dae['params_dict'].update({f"K_pow_{name}":K_pow})
    ## 2 mass mechanical model

    ## c_p_b:
    beta_b = 0.0
    lam = Lam_b
    inv_lam_i =  1/(lam + 0.08*beta_b) - 0.035/(beta_b**3 + 1.0)   
    C_p_b = C_1*(C_2*inv_lam_i - C_3*beta_b - C_4)*np.exp(-C_5*inv_lam_i) + C_6*lam 
    grid.dae['params_dict'].update({f"C_p_b_{name}":C_p_b})


    grid.dae['params_dict'].update({f"T_mppt_{name}":5})

    if 'vsc_g' in mode:
        p_W   = p_s * S_n
        q_var = q_s * S_n
    else:
        p_W   = 0
        q_var = 0      

    return p_W,q_var


def symmodel():

    S_b,U_b,F_b,Omega_eb,N_pp = sym.symbols('S_b,U_b,F_b,Omega_eb,N_pp', real = True)
    i_md_pu,i_mq_pu,v_md_pu,v_mq_pu,omega_e_pu = sym.symbols(f'i_md_pu,i_mq_pu,v_md_pu,v_mq_pu,omega_e_pu', real = True)
    R_m_pu,L_m_pu,Phi_m_pu,tau_r_pu,p_m_pu,q_m_pu= sym.symbols(f'R_m_pu,L_m_pu,Phi_m_pu,tau_r_pu,p_m_pu,q_m_pu', real = True)
    Omega_rb = Omega_eb/N_pp
    Tau_b = S_b/Omega_rb
    I_b = S_b/(sym.sqrt(3)*U_b)
    I_dq_b = I_b*sym.sqrt(2)
    V_dq_b = U_b*sym.sqrt(2)/sym.sqrt(3)
    V_dc_b = sym.sqrt(2)*U_b
    I_dc_b = S_b/V_dc_b

    # V_dq_b and I_dq_b defined as below verify that: 
    # 3/2*I_dq_b*V_dq_b = 3/2*I_b*sym.sqrt(2)*U_b*sym.sqrt(2/3) = np.sqrt(3)*U_b*I_b

    Z_b = U_b**2/S_b
    #Z_b = (V_dq_b/sym.sqrt(2/3))**2/S_b #= 3/2*V_dq_b**2/S_b
    L_m_b = Z_b/Omega_eb
    Phi_b = V_dq_b/Omega_eb

    #tau_r = 3/2*Phi*N_pp*i_mq

    #tau_r_pu = (3/2*Phi*N_pp*i_mq_pu)*I_dq_b/Tau_b  = K_tau * i_mq_pu
    #with K_tau = 3/2*Phi*N_pp*I_dq_b/Tau_b

    # 0 =               - R_s*i_md - omega_r*L_m*i_mq - v_md  
    # 0 = omega_r*Phi_m - R_s*i_mq + omega_r*L_m*i_md - v_mq 

    R_m = R_m_pu*Z_b
    i_md = i_md_pu*I_dq_b
    i_mq = i_mq_pu*I_dq_b
    v_md = v_md_pu*V_dq_b
    v_mq = v_mq_pu*V_dq_b
    L_m = L_m_pu*L_m_b
    omega_e = omega_e_pu*Omega_eb
    Phi_m = Phi_m_pu*Phi_b

    # PMSM electric equations
    eq_i_md =  - R_m*i_md - omega_e*L_m*i_mq - v_md  
    eq_i_mq = omega_e*Phi_m - R_m*i_mq + omega_e*L_m*i_md - v_mq 

    print(f"eq_i_md = {str(sym.simplify(eq_i_md/V_dq_b)).replace('_pu','')}")
    print(f"eq_i_mq = {str(sym.simplify(eq_i_mq/V_dq_b)).replace('_pu','')}")

    # Torque equations
    tau_r = tau_r_pu*Tau_b
    eq_tau_r = -tau_r + 3/2*Phi_m*N_pp*i_mq
    print(f"eq_tau_r = {str(sym.simplify(eq_tau_r/Tau_b)).replace('_pu','').replace('1.0*','')}")


    # PMSM Power equations
    p_m = p_m_pu*S_b
    q_m = q_m_pu*S_b

    eq_p_m = -p_m + 3/2*(v_md*i_md + v_mq*i_mq)
    eq_q_m = -q_m + 3/2*(v_md*i_mq - v_mq*i_md)

    print(f"eq_p_m = {str(sym.simplify(eq_p_m/S_b)).replace('_pu','').replace('1.0*','')}")
    print(f"eq_q_m = {str(sym.simplify(eq_q_m/S_b)).replace('_pu','').replace('1.0*','')}")

    # DC link
    v_dc_pu,C_dc_pu = sym.symbols('v_dc_pu,C_dc_pu', real=True)
    v_dc = v_dc_pu*V_dc_b
    p_mac = 3/2*(v_md*i_md + v_mq*i_mq)
    p_mdc = p_mac
    i_mdc = p_mdc/v_dc

    print(f"i_mdc = {str(sym.simplify(i_mdc/I_dc_b)).replace('_pu','').replace('1.0*','')}")

    v_tr_pu,v_ti_pu,i_tr_pu,i_ti_pu = sym.symbols('v_tr_pu,v_ti_pu,i_tr_pu,i_ti_pu', real=True)
    v_tr = v_tr_pu*V_dq_b
    v_ti = v_ti_pu*V_dq_b
    i_tr = i_tr_pu*I_dq_b
    i_ti = i_ti_pu*I_dq_b

    p_tac = 3/2*(v_tr*i_tr + v_ti*i_ti)
    p_tdc = p_tac
    i_tdc = p_tdc/v_dc
    
    print(f"i_tdc = {str(sym.simplify(i_tdc/I_dc_b)).replace('_pu','').replace('1.0*','')}")

    C_dc_b = 1/(Z_b)
    C_dc = C_dc_pu*C_dc_b

    dv_dc = 1/C_dc*(i_mdc - i_tdc)
    print(f"dv_dc = {str(sym.simplify(dv_dc/V_dc_b))}")





def mppt_omega_p(omega_t):
    
    C_1 = 0.5176
    C_2 = 116
    C_3 = 0.4
    C_4 = 5.0
    C_5 = 21 
    C_6 = 0.0068
    Nu_w_b = 12
    Lam_b = 8.1
    Omega_t_b = 1.2
    K_pow = 0.73

    ## MPPT
    Lam_opt = Lam_b
    beta_mppt = 0.0
    nu_w_mppt = Nu_w_b*(omega_t/Omega_t_b)   
    inv_lam_i_mppt =  1/(Lam_opt) - 0.035   
    c_p_mppt = C_1*(C_2*inv_lam_i_mppt - C_4)*sym.exp(-C_5*inv_lam_i_mppt) + C_6*Lam_opt 
    c_p_mppt_pu = c_p_mppt/C_p_b # (pu)
    p_w_mppt_ref = K_pow*c_p_mppt_pu*(nu_w_mppt/Nu_w_b)**3
    omega_r_th = 0.2
    # test_1 = (np.abs(Dp_e_ref)>0.01) & (omega_r>omega_r_th)
    # p_w_mppt = sym.Piecewise((p_w_mppt_lpf,test_1), (p_w_mppt_ref,True))
    # K_mppt = sym.Piecewise((0.0,test_1), (1.0,True))
    # Dp_e = sym.Piecewise((Dp_e_ref,test_1), (Dp_e_ref * (1 + 50*(omega_r - omega_r_th)),True))
    # p_w_mmpt_ref = p_w_mppt_lpf #+ Dp_e   test

    return p_w_mppt_ref


if __name__ == '__main__':
    #symmodel()

    from pydae.bmapu import bmapu_builder
    import sympy as sym

    data = {
    "system":{"name":"smib","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},       
    "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
             {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}
            ],
    "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.01,"Bs_pu":1e-6,"S_mva":100.0}],
    "wecs":[
        {"type":"pmsm_1","bus":"1","S_n":1e6,
            "H_t":4.0,"H_r":1.0, "w_tr":5.0, "d_tr":0.1,
            "R_m":0.01,"L_m":0.05,"Phi_m":1.0,
            "R_s":0.01,"X_s":0.05,
            "K_pdc":0.1,"C_dc":0.5}],
    "genapes":[{"bus":"2","S_n":1e9,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]
    }

    grid = bmapu_builder.bmapu(data)
    #grid.checker()
    grid.uz_jacs = True
    grid.verbose = False
    grid.build('pmsm_test')

    import pmsm_test

    model = pmsm_test.model()
    
    
    N = model.jac_ini.shape[0]
    for it in range(N):
        print(it,model.jac_ini[it,:])

    model.ini({},'xy_0.json')