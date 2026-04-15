# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def uvsg(grid,name,bus_name,data_dict):
    """
    # auxiliar equations
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi
    v_D = V*sin(theta)  # e^(-j)
    v_Q = V*cos(theta) 
    v_d = v_D * cos(delta) - v_Q * sin(delta)   
    v_q = v_D * sin(delta) + v_Q * cos(delta)

    Domega = x_v + K_p * (p_ref - p)
    e_dv = 0.0
    epsilon_v = v_ref - V
    i_d = i_d_ref
    i_q = i_q_ref
    omega_v = Domega + 1.0
    q_ref_0 = K_p_v * epsilon_v + K_i_v * xi_v

    # dynamical equations
    ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
    dx_v = K_i*(p_ref - p) - K_g*(omega_v - 1.0)
    de_qm = 1.0/T_q * (q - q_ref_0 - q_ref - e_qm) 
    dxi_v = epsilon_v # PI agregado

    # algebraic equations
    g_i_d_ref  = e_dv - R_v * i_d_ref - X_v * i_q_ref - v_d 
    g_i_q_ref  = e_qv - R_v * i_q_ref + X_v * i_d_ref - v_q 
    g_p  = v_d*i_d + v_q*i_q - p  
    g_q  = v_d*i_q - v_q*i_d - q 
    g_e_qv = 1.0 - e_qv - K_q*e_qm 

    {"bus":"1","type":"vsg_ll",'S_n':10e6,'F_n':50,'K_delta':0.0,
    'R_v':0.01,'X_v':0.1,'K_p':1.0,'K_i':0.1,'K_g':0.0,'K_q':20.0,
    'T_q':0.1,'K_p_v':1e-6,'K_i_v':1e-6}
    
    """

    sin = sym.sin
    cos = sym.cos

    ctrl_data = data_dict['ctrl']

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]                  

    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1

    # inputs:
    q_l,q_r = sym.symbols(f'q_l_{name},q_r_{name}', real=True)
    p_l,p_r = sym.symbols(f'p_l_{name},p_r_{name}', real=True)
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)   
    Ddelta_ff,Domega_ff = sym.symbols(f'Ddelta_ff_{name},Domega_ff_{name}', real=True)
    v_u = sym.Symbol(f"v_u_{name}", real=True)  
    v_u_ref  = sym.Symbol(f"v_u_ref_{name}", real=True) 
    i_sact_ref = sym.Symbol(f"i_sact_ref_{name}", real=True) 
    i_srea_ref = sym.Symbol(f"i_srea_ref_{name}", real=True)

    # dynamical states:
    delta = sym.Symbol(f"delta_{name}", real=True)   
    xi_p  = sym.Symbol(f"xi_p_{name}", real=True)   
    xi_q  = sym.Symbol(f"xi_q_{name}", real=True)   
    xi_u  = sym.Symbol(f"xi_u{name}", real=True)   

    # algebraic states:
    p_s = sym.Symbol(f'p_s_{name}', real=True)
    q_s = sym.Symbol(f'q_s_{name}', real=True)
    e_qv = sym.Symbol(f"e_qv_{name}", real=True)
    v_td_ref = sym.Symbol(f"v_td_ref_{name}", real=True)
    v_tq_ref = sym.Symbol(f"v_tq_ref_{name}", real=True) 
    i_sd_mid = sym.Symbol(f"i_sd_mid_{name}", real=True)
    i_sq_mid = sym.Symbol(f"i_sq_mid_{name}", real=True) 
    i_sd_fault = sym.Symbol(f"i_sd_fault_{name}", real=True) 
    i_sq_fault = sym.Symbol(f"i_sq_fault_{name}", real=True)
    omega_v  = sym.Symbol(f"omega_v_{name}", real=True) 
    p_u = sym.Symbol(f"p_u_{name}", real=True) 

    # params:
    S_n,F_n = sym.symbols(f'S_n_{name},F_n_{name}', real=True)
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_p,T_p = sym.symbols(f'K_p_{name},T_p_{name}', real=True)
    K_q,T_q = sym.symbols(f'K_q_{name},T_q_{name}', real=True)
    K_u,T_u = sym.symbols(f'K_u_{name},T_u_{name}', real=True)
    R_s,X_s = sym.symbols(f'R_s_{name},X_s_{name}', real=True)
    K_u_max,K_ui,DV_th = sym.symbols(f'K_u_max_{name},K_ui_{name},DV_th_{name}', real=True)
    params_list = ['F_n','R_v','X_v','K_p','T_p','K_q','T_q','K_u','T_u']

    # auxiliar variables and constants
    omega_coi = sym.Symbol("omega_coi", real=True) # from global system
    V_s = sym.Symbol(f"V_{bus_name}", real=True) # from global system
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True) # from global system 
    m = sym.Symbol(f"m_{name}", real=True) 
    theta_t = sym.Symbol(f"theta_t_{name}", real=True) 

    # auxiliar equations
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi
    e_dv = 0
    delta_ff =  delta + Ddelta_ff

    park = 'easy-res'
    # easy-ref park
    if park == 'easy-res': 

        v_sr = V_s*cos(theta_s)  # v_sr
        v_si = V_s*sin(theta_s)  # v_si
        v_salpha =-v_si
        v_sbeta  =-v_sr

        v_sd = v_salpha * cos(delta_ff) + v_sbeta * sin(delta_ff)   
        v_sq =-v_salpha * sin(delta_ff) + v_sbeta * cos(delta_ff)

        v_talpha_ref =  v_td_ref * cos(delta_ff) - v_tq_ref * sin(delta_ff)   
        v_tbeta_ref  =  v_td_ref * sin(delta_ff) + v_tq_ref * cos(delta_ff) 

        v_ti_ref = -v_talpha_ref
        v_tr_ref = -v_tbeta_ref   

    v_m = (v_sd**2 + v_sq**2 + 0.0001)**0.5
    lvrt = sym.Piecewise((1.0,v_m<(0.8-DV_th)),(1.0*(0.8-v_m)/DV_th,v_m<0.8),(0.0,True))
    p_ref = p_l + p_r + p_u
    q_ref = q_l + q_r
    epsilon_p = (p_ref - p_s)*(1-lvrt)
    epsilon_q = q_ref - q_s
    epsilon_u = v_u_ref - v_u
    p_u_l_ref = -K_u*epsilon_u
    
    p_u_ref = sym.Piecewise((K_u_max*(v_u-0.1)+p_u_l_ref,v_u<0.1),
                            (K_u_max*(v_u-0.9)+p_u_l_ref,v_u>0.9),
                            (p_u_l_ref,True))


    #i_sd_ref = sym.Piecewise((i_sd_fault,lvrt>0.5),(i_sd_mid,True))
    #i_sq_ref = sym.Piecewise((i_sq_fault,lvrt>0.5),(i_sq_mid,True))
    i_sd_ref = (lvrt)*i_sd_fault + (lvrt-1)*i_sd_mid
    i_sq_ref = (lvrt)*i_sq_fault + (lvrt-1)*i_sq_mid

    # from controller to vsc hardware
    m_ref = sym.sqrt(v_tr_ref**2 + v_ti_ref**2)/v_dc
    theta_t_ref = sym.atan2(v_ti_ref,v_tr_ref) 


    # dynamic equations            
    ddelta = Omega_b*(omega_v - omega_s)
    dxi_p = epsilon_p
    dxi_q = epsilon_q
    dxi_u = epsilon_u



    # algebraic equations
    g_v_td_ref  = v_td_ref - R_s * i_sd_ref - X_s * i_sq_ref - v_sd   # ctrl1 i_sd control
    g_v_tq_ref  = v_tq_ref - R_s * i_sq_ref + X_s * i_sd_ref - v_sq   # ctrl1 i_sq control
    g_i_sd_mid  = e_dv - R_v * i_sd_mid - X_v * i_sq_mid - v_sd   # ctrl3 virtual impedance 
    g_i_sq_mid  = e_qv - R_v * i_sq_mid + X_v * i_sd_mid - v_sq   # ctrl3 virtual impedance
    g_i_sd_fault  = (i_sd_fault*v_sd + i_sq_fault*v_sq)/v_m - i_sact_ref  
    g_i_sq_fault  = (i_sq_fault*v_sd - i_sd_fault*v_sq)/v_m - i_srea_ref
    g_omega_v = -omega_v + K_p*(epsilon_p + xi_p/T_p)    # p PI -> omega_v
    g_e_qv = -e_qv   + K_q*(epsilon_q + xi_q/T_q)        # q PI -> e_qv
    g_p_u  = -p_u + p_u_ref - K_ui*xi_u
    g_m  = m-m_ref   # to the VSC
    g_theta_t  = theta_t-theta_t_ref # to the VSC

    # DAE system update
    grid.dae['f'] += [ddelta,dxi_p,dxi_q,dxi_u]
    grid.dae['x'] += [ delta, xi_p, xi_q, xi_u]
    grid.dae['g'] +=     [g_v_td_ref,g_v_tq_ref,g_i_sd_mid,g_i_sq_mid,g_i_sd_fault,g_i_sq_fault,g_omega_v,g_e_qv,g_p_u,g_m,g_theta_t]
    grid.dae['y_ini'] += [  v_td_ref,  v_tq_ref,  i_sd_mid,  i_sq_mid,  i_sd_fault,  i_sq_fault,  omega_v,  e_qv,  p_u,  m,  theta_t]
    grid.dae['y_run'] += [  v_td_ref,  v_tq_ref,  i_sd_mid,  i_sq_mid,  i_sd_fault,  i_sq_fault,  omega_v,  e_qv,  p_u,  m,  theta_t]
            
    # default inputs
    grid.dae['u_ini_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'Ddelta_ff_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'Domega_ff_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'v_u_ref_{name}':0.5})
    grid.dae['u_ini_dict'].update({f'i_sact_ref_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'i_srea_ref_{name}':1.0})

    grid.dae['u_run_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'Ddelta_ff_{name}':0.0})
    grid.dae['u_run_dict'].update({f'Domega_ff_{name}':0.0})
    grid.dae['u_run_dict'].update({f'v_u_ref_{name}':0.5})
    grid.dae['u_run_dict'].update({f'i_sact_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'i_srea_ref_{name}':1.0})

    # default parameters
    for item in params_list:
        grid.dae['params_dict'].update({item + f'_{name}':ctrl_data[item]})
    grid.dae['params_dict'].update({f'K_u_max_{name}':2.0})
    grid.dae['params_dict'].update({f'K_ui_{name}':0.001})
    grid.dae['params_dict'].update({f'DV_th_{name}':0.1})


    # add speed*H term to global for COI speed computing
    H = 4.0
    grid.H_total += H
    grid.omega_coi_numerator += omega_v*H*S_n
    grid.omega_coi_denominator += H*S_n

    # DAE outputs update
    grid.dae['h_dict'].update({f"omega_v_{name}":omega_v})
    grid.dae['h_dict'].update({f"p_ref_{name}":p_ref})
    grid.dae['h_dict'].update({f"q_ref_{name}":q_ref})
    grid.dae['h_dict'].update({f"i_sd_ref_{name}":i_sd_ref})
    grid.dae['h_dict'].update({f"i_sq_ref_{name}":i_sq_ref})
    grid.dae['h_dict'].update({f"delta_ff_{name}":delta_ff})
    grid.dae['h_dict'].update({f"lvrt_{name}":lvrt})
    

    grid.dae['xy_0_dict'].update({str(e_qv):1.0}) 
    grid.dae['xy_0_dict'].update({str(v_tq_ref):1.0}) 

    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var


if __name__ == "__main__":
    pass  