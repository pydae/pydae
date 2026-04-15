# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def regfm_b1(grid,name,bus_name,data_dict):
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
    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]                  

    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1

    # inputs:
    q_l,q_r,v_ref = sym.symbols(f'q_l_{name},q_r_{name},v_ref_{name}', real=True)
    p_l,p_r = sym.symbols(f'p_l_{name},p_r_{name}', real=True)

    # dynamical states:
    delta_v,Domega_m,x_d2,xi_v = sym.symbols(f'delta_v_{name},Domega_m_{name},x_d2_{name},xi_v_{name}', real=True)
    
    # algebraic states:
    i_d_ref,i_q_ref,p,q,e_qv = sym.symbols(f'i_d_ref_{name},i_q_ref_{name},p_{name},q_{name},e_qv_{name}', real=True)

    # params:
    S_base = sym.Symbol('S_base', real = True) # S_base = global power base, S_n = machine power base
    S_n,F_n,K_delta = sym.symbols(f'S_n_{name},F_n_{name},K_delta_{name}', real=True)
    H,D_1,D_2,Omega_d = sym.symbols(f'H_{name},D_1_{name},D_2_{name},Omega_d_{name}', real=True)
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_pv,K_iv = sym.symbols(f'K_pv_{name},K_iv_{name}', real=True)

    # auxiliar variables and constants
    omega_coi = sym.Symbol("omega_coi", real=True) # from global system
    V = sym.Symbol(f"V_{bus_name}", real=True) # from global system
    theta = sym.Symbol(f"theta_{bus_name}", real=True) # from global system
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)

    # auxiliar equations
    v_cmd = v_ref
    v_inv = V
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi
    v_d = V*sin(theta)  # e^(-j)
    v_q = V*cos(theta) 
    # v_d = v_D * cos(delta_v) - v_Q * sin(delta_v)   
    # v_q = v_D * sin(delta_v) + v_Q * cos(delta_v)

    e = K_pv*(v_cmd - v_inv) + xi_v + 1.0
    e_d = e*sin(delta_v)  # e^(-j)
    e_q = e*cos(delta_v) 

    p_ref = p_l + p_r
    q_ref = q_l + q_r
    i_d = i_d_ref
    i_q = i_q_ref
    omega_v = Domega_m + 1.0
    

    p_inv = p
    p_cmd = p_ref
    # damping
    p_d1 = D_1*Domega_m

    # transient damping
    D_1,D_2
    K_iv, K_pv
    p_d2 = D_2*(Domega_m - Omega_d*x_d2) 

    # dynamical equations
    ddelta_v   = Omega_b*(omega_v - omega_s)  
    dDomega_m = 1.0/(2*H)*(p_cmd - p_inv - p_d1 - p_d2)
    dx_d2 =  Domega_m - Omega_d*x_d2

    # voltage control   
    dxi_v = K_iv*(v_cmd - v_inv) ; 

    # algebraic equations
    g_i_d_ref  = e_d - R_v * i_d_ref - X_v * i_q_ref - v_d 
    g_i_q_ref  = e_q - R_v * i_q_ref + X_v * i_d_ref - v_q 
    
    g_p  = v_d*i_d + v_q*i_q - p  
    g_q  = v_d*i_q - v_q*i_d - q 
    

    # DAE system update
    grid.dae['f'] += [ddelta_v,dDomega_m,dx_d2,dxi_v]
    grid.dae['x'] += [ delta_v, Domega_m, x_d2, xi_v]
    grid.dae['g'] +=     [g_i_d_ref,g_i_q_ref,g_p,g_q]
    grid.dae['y_ini'] += [  i_d_ref,  i_q_ref,  p,  q]
    grid.dae['y_run'] += [  i_d_ref,  i_q_ref,  p,  q]
            
    # default inputs
    grid.dae['u_ini_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'v_ref_{name}':1.0})
    grid.dae['u_run_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'v_ref_{name}':1.0})

    # default parameters
    grid.dae['params_dict'].update({str(H):data_dict['H']})
    grid.dae['params_dict'].update({str(D_1):data_dict['D_1']})
    grid.dae['params_dict'].update({str(D_2):data_dict['D_2']})
    grid.dae['params_dict'].update({str(Omega_d):data_dict['Omega_d']})
    grid.dae['params_dict'].update({str(R_v):data_dict['R_v']})
    grid.dae['params_dict'].update({str(X_v):data_dict['X_v']})
    grid.dae['params_dict'].update({str(S_n):data_dict['S_n']})
    grid.dae['params_dict'].update({str(F_n):data_dict['F_n']})
    grid.dae['params_dict'].update({str(K_pv):data_dict['K_pv']})
    grid.dae['params_dict'].update({str(K_iv):data_dict['K_iv']})

    # add speed*H term to global for COI speed computing
    H = data_dict['H']
    grid.H_total += H
    grid.omega_coi_numerator += omega_v*H*S_n
    grid.omega_coi_denominator += H*S_n

    # DAE outputs update
    grid.dae['h_dict'].update({f"omega_v_{name}":omega_v})
    grid.dae['h_dict'].update({f"p_ref_{name}":p_ref})
    grid.dae['h_dict'].update({f"q_ref_{name}":q_ref})
    grid.dae['h_dict'].update({f"v_ref_{name}":v_ref})
    grid.dae['h_dict'].update({f"i_sd_{name}":i_d_ref})    
    grid.dae['h_dict'].update({f"i_sq_{name}":i_q_ref})

    grid.dae['xy_0_dict'].update({str(e_qv):1.0}) 

    p_W   = p * S_n
    q_var = q * S_n

    return p_W,q_var


def test_build():

    import pydae.build_cffi as db
    from pydae.bmapu import bmapu_builder
    import pydae.build_cffi as db

    data = {
        "system":{"name":"temp","S_base":1e6, "K_p_agc":0.0,"K_i_agc":0.01,"K_xif":1e-6},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.0, "S_mva":0.02}],
        "vsgs":[
            {"bus":"1","type":"regfm_b1",'S_n':20e3,'F_n':50,'K_delta':0.0,
            'H':5.0,'D_1':0.0,'D_2':100,'Omega_d':100,
            'R_v':0.01,'X_v':0.1,'K_pv':1.0,'K_iv':100}],
        "sources":[{"type":"genape","bus":"2","S_n":100e6,"F_n":50.0,"R_v":0.0,"X_v":0.1,"K_delta":0.001,"K_alpha":0.1}],
        }

    grid = bmapu_builder.bmapu(data)
    grid.uz_jacs = True
    grid.verbose = True
    grid.build('temp')


def test_ini():

    import temp 
    model = temp.model()
    model.ini({'p_l_1':0.5,'v_ref_1':1.02},'xy_0.json')

    print('u:')
    model.report_u()
    print('x:')
    model.report_x()
    print('y:')
    model.report_y()
    print('z:')
    model.report_z()

if __name__ == "__main__":

    test_build()
    test_ini()