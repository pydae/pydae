# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def full_converter(grid,name,bus_name,data_dict):
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

    v_dc_b = sqrt(2)*v_ac_b
    v_t_pu*v_b_ac = m*v_dc_pu*sqrt(2)*v_ac_b/sqrt(6)
    v_t_pu = m*v_dc_pu*sqrt(2)/sqrt(6) = m*v_dc_pu*sqrt(3)
    
    
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
    theta_tr = sym.Symbol(f"theta_tr_{name}", real=True)
    omega_t  = sym.Symbol(f"omega_t_{name}", real=True)
    omega_r  = sym.Symbol(f"omega_r_{name}", real=True)
    xi_beta  = sym.Symbol(f"xi_beta_{name}", real=True)
    beta     = sym.Symbol(f"beta_{name}", real=True)
    p_m_mppt_lpf = sym.Symbol(f"p_m_mppt_lpf_{name}", real=True)

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
    
    # auxiliar
    ## VSC
    v_sr =  V_s*cos(theta_s)  # v_Q
    v_si =  V_s*sin(theta_s)  # v_D, e^(-j)
    Omega_b = 2*np.pi*F_n
    v_t_m = m*v_dc
    v_tr =  v_t_m*cos(theta_t)
    v_ti =  v_t_m*sin(theta_t)
    i_m = sym.sqrt(i_sr**2 + i_si**2)
    p_loss = A_l + B_l*i_m + C_l*i_m**2
    p_ac = p_s + R_s*i_m**2
    p_dc = p_ac + p_loss
    i_d = p_dc/v_dc

    ## Aerodinamic
    lam = Lam_b*(omega_t/Omega_t_b)/(nu_w/Nu_w_b) # pu
    inv_lam_i =  1/(lam + 0.08*beta) - 0.035/(beta**3 + 1.0)   
    c_p = C_1*(C_2*inv_lam_i - C_3*beta - C_4)*sym.exp(-C_5*inv_lam_i) + C_6*lam 
    c_p_pu = c_p/C_p_b # (pu)
    p_m_nosat = K_pow*c_p_pu*(nu_w/Nu_w_b)**3
    p_m = sym.Piecewise((0.0,p_m_nosat<0.0), (p_m_nosat,True))

    ## MPPT
    Lam_opt = Lam_b
    beta_mppt = 0.0
    nu_w_mppt = Nu_w_b*(omega_t/Omega_t_b)   
    inv_lam_i_mppt =  1/(Lam_opt) - 0.035   
    c_p_mppt = C_1*(C_2*inv_lam_i_mppt - C_4)*sym.exp(-C_5*inv_lam_i_mppt) + C_6*Lam_opt 
    c_p_mppt_pu = c_p_mppt/C_p_b # (pu)
    p_m_mppt_ref = K_pow*c_p_mppt_pu*(nu_w_mppt/Nu_w_b)**3
    omega_r_th = 0.2
    test_1 = (np.abs(Dp_e_ref)>0.01) & (omega_r>omega_r_th)
    p_m_mppt = sym.Piecewise((p_m_mppt_lpf,test_1), (p_m_mppt_ref,True))
    K_mppt = sym.Piecewise((0.0,test_1), (1.0,True))
    Dp_e = sym.Piecewise((Dp_e_ref,test_1), (Dp_e_ref * (1 + 50*(omega_r - omega_r_th)),True))
    p_s_mmpt_ref = p_m_mppt_lpf + Dp_e    

    ## Pitch
    Omega_r_b = Omega_t_b
    omega_r_max = Omega_r_b
    epsilon_beta_nosat = omega_r - omega_r_max
    epsilon_beta = sym.Piecewise((0.0,epsilon_beta_nosat<0),(epsilon_beta_nosat,True))
    beta_ref = K_p_beta*epsilon_beta + K_i_beta*xi_beta

    # dynamic equations            
    dtheta_tr = omega_t-omega_r - u_dummy
    domega_t  = 1.0/(2*H_t)*(p_m - K_tr*theta_tr - D_tr*(omega_t-omega_r))
    domega_r  = 1.0/(2*H_r)*(K_tr*theta_tr + D_tr*(omega_t-omega_r) - p_dc)
    dxi_beta  = epsilon_beta - xi_beta*1e-8
    dbeta     = 1.0/T_beta*(beta_ref - beta)
    dp_m_mppt_lpf  = K_mppt/T_mppt*(p_m_mppt_ref - p_m_mppt_lpf)

    # algebraic equations   
    g_i_si = v_ti - R_s*i_si - X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr + X_s*i_si - v_sr 
    g_p_s  =  i_sr*v_sr + i_si*v_si - p_s  
    g_q_s  =  i_sr*v_si - i_si*v_sr - q_s 
    g_p_s_ref = -p_s_ref + p_s_mmpt_ref


    # dae 
    grid.dae['f'] += [dtheta_tr,domega_t,domega_r,dxi_beta,dbeta,dp_m_mppt_lpf]
    grid.dae['x'] += [ theta_tr, omega_t, omega_r, xi_beta, beta, p_m_mppt_lpf]
    grid.dae['g'] += [g_i_si,g_i_sr,g_p_s,g_q_s,g_p_s_ref]
    grid.dae['y_ini'] += [  i_si,  i_sr,  p_s,  q_s, p_s_ref]  
    grid.dae['y_run'] += [  i_si,  i_sr,  p_s,  q_s, p_s_ref]  
    
    grid.dae['u_ini_dict'].update({f'{m}':1.0})
    grid.dae['u_run_dict'].update({f'{m}':1.0})

    grid.dae['u_ini_dict'].update({f'{theta_t}':0.0})
    grid.dae['u_run_dict'].update({f'{theta_t}':0.0})

    grid.dae['u_ini_dict'].update({f'{v_dc}':1.2})
    grid.dae['u_run_dict'].update({f'{v_dc}':1.2})

    grid.dae['u_ini_dict'].update({f'{Dp_e_ref}':0.0})
    grid.dae['u_run_dict'].update({f'{Dp_e_ref}':0.0})

    grid.dae['u_ini_dict'].update({f'{nu_w}':10.0})
    grid.dae['u_run_dict'].update({f'{nu_w}':10.0})

    grid.dae['u_ini_dict'].update({f'{u_dummy}':0.0})
    grid.dae['u_run_dict'].update({f'{u_dummy}':0.0})
 


    # outputs
    grid.dae['h_dict'].update({f"{str(p_s)}":p_s})
    grid.dae['h_dict'].update({f"{str(q_s)}":q_s}) 
    grid.dae['h_dict'].update({f"{str(i_si)}":i_si})
    grid.dae['h_dict'].update({f"{str(i_sr)}":i_sr})
    grid.dae['h_dict'].update({f"p_ac_{name}":p_ac})
    grid.dae['h_dict'].update({f"p_dc_{name}":p_dc})
    grid.dae['h_dict'].update({f"i_d_{name}":i_d})
    grid.dae['h_dict'].update({f"u_dummy_{name}":u_dummy})
    grid.dae['h_dict'].update({f"p_m_mppt_lpf_{name}":p_m_mppt_lpf})
    grid.dae['h_dict'].update({f"p_t_{name}":p_m})
    grid.dae['h_dict'].update({f"K_mppt_{name}":K_mppt})
    grid.dae['h_dict'].update({f"lam_{name}":lam})
    grid.dae['h_dict'].update({f"c_p_{name}":c_p})
    grid.dae['h_dict'].update({f"c_p_pu_{name}":c_p_pu})
    grid.dae['h_dict'].update({f"nu_w_mppt_{name}":nu_w_mppt})
    grid.dae['h_dict'].update({f"p_m_{name}":p_m})
    

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]}) 

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
    K_pow = 0.73
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
    omega_ref = 2*np.pi*0.5
    zeta_ref = 0.05
    H_t = 3.0
    H_r = 1.0
    grid.dae['params_dict'].update({f"H_t_{name}":H_t})
    grid.dae['params_dict'].update({f"H_r_{name}":H_r})
    K_tr = 2*H_r*H_t*omega_ref**2/(H_r + H_t)
    D_tr =  4*np.pi*H_r*H_t*zeta_ref*omega_ref/(H_r + H_t)
    grid.dae['params_dict'].update({f"K_tr_{name}":K_tr})
    grid.dae['params_dict'].update({f"D_tr_{name}":D_tr})
    ## c_p_b:
    beta_b = 0.0
    lam = Lam_b
    inv_lam_i =  1/(lam + 0.08*beta_b) - 0.035/(beta_b**3 + 1.0)   
    C_p_b = C_1*(C_2*inv_lam_i - C_3*beta_b - C_4)*np.exp(-C_5*inv_lam_i) + C_6*lam 
    grid.dae['params_dict'].update({f"C_p_b_{name}":C_p_b})

    grid.dae['params_dict'].update({f"T_beta_{name}":2.0})
    grid.dae['params_dict'].update({f"K_p_beta_{name}":100.0})
    grid.dae['params_dict'].update({f"K_i_beta_{name}":1.0})
    grid.dae['params_dict'].update({f"T_mppt_{name}":5})

    
    # initial conditions
    beta_0 = 0.0  # initial input
    p_ref = 0.5*0.657/0.9 # initial input
    inv_lam_i =  1/(lam + 0.08*beta_0) - 0.035/(beta_0**3 + 1.0)   
    c_p = C_1*(C_2*inv_lam_i - C_3*beta_0 - C_4)*np.exp(-C_5*inv_lam_i) + C_6*lam 
    c_p_pu = c_p/C_p_b # (pu)
    #p_m_ref = K_p*c_p*(nu_w/nu_w_b)**3
    nu_w_0 = (p_ref/(K_pow*c_p_pu))**(1.0/3.0)*Nu_w_b
    omega_t_0 = lam*(nu_w_0/Nu_w_b)/Lam_b*Omega_t_b

    grid.dae['xy_0_dict'].update({str(p_s):p_ref})
    grid.dae['xy_0_dict'].update({str(theta_tr):p_ref/K_tr})
    grid.dae['xy_0_dict'].update({str(omega_t):omega_t_0})
    grid.dae['xy_0_dict'].update({str(omega_r):omega_t_0})
    grid.dae['xy_0_dict'].update({str(beta):beta_0})
    grid.dae['xy_0_dict'].update({str(p_s_ref):p_ref})
    grid.dae['xy_0_dict'].update({str(nu_w):nu_w_0})
    grid.dae['xy_0_dict'].update({str(i_sr):0.1})

    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var


