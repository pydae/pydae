# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def uvsg_mid(grid,name,bus_name,data_dict):
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
    p_ref,q_ref,v_ref = sym.symbols(f'p_ref_{name},q_ref_{name},v_ref_{name}', real=True)

    # dynamical states:
    delta,x_v,e_qm,xi_v = sym.symbols(f'delta_{name},x_v_{name},e_qm_{name},xi_v_{name}', real=True)
    
    # algebraic states:
    i_d_ref,i_q_ref,p,q,e_qv = sym.symbols(f'i_d_ref_{name},i_q_ref_{name},p_{name},q_{name},e_qv_{name}', real=True)

    # params:
    S_base = sym.Symbol('S_base', real = True) # S_base = global power base, S_n = machine power base
    S_n,F_n,K_delta = sym.symbols(f'S_n_{name},F_n_{name},K_delta_{name}', real=True)
    K_p,K_i,K_g,K_i_q = sym.symbols(f'K_p_{name},K_i_{name},K_g_{name},K_i_q_{name}', real=True)
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_q,T_q = sym.symbols(f'K_q_{name},T_q_{name}', real=True)
    K_p_v,K_i_v = sym.symbols(f'K_p_v_{name},K_i_v_{name}', real=True)
    params_list = ['S_n','F_n','K_delta','K_p','K_i','K_g','R_v','X_v','K_q','T_q','K_p_v','K_i_v']

    # auxiliar variables and constants
    omega_coi = sym.Symbol("omega_coi", real=True) # from global system
    V = sym.Symbol(f"V_{bus_name}", real=True) # from global system
    theta = sym.Symbol(f"theta_{bus_name}", real=True) # from global system
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)

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
    
    # DAE system update
    grid.dae['f'] += [ddelta,dx_v,de_qm,dxi_v]
    grid.dae['x'] += [ delta, x_v, e_qm, xi_v]
    grid.dae['g'] +=     [g_i_d_ref,g_i_q_ref,g_p,g_q,g_e_qv]
    grid.dae['y_ini'] += [  i_d_ref,  i_q_ref,  p,  q,  e_qv]
    grid.dae['y_run'] += [  i_d_ref,  i_q_ref,  p,  q,  e_qv]
            
    # default inputs
    grid.dae['u_ini_dict'].update({f'p_ref_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_ref_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'v_ref_{name}':1.0})
    grid.dae['u_run_dict'].update({f'p_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'v_ref_{name}':1.0})

    # default parameters
    for item in params_list:
        grid.dae['params_dict'].update({item + f'_{name}':data_dict[item]})

    # add speed*H term to global for COI speed computing
    H = 4.0
    grid.H_total += H
    grid.omega_coi_numerator += omega_v*H*S_n
    grid.omega_coi_denominator += H*S_n

    # DAE outputs update
    grid.dae['h_dict'].update({f"omega_v_{name}":omega_v})
    grid.dae['h_dict'].update({f"p_ref_{name}":p_ref})
    grid.dae['h_dict'].update({f"q_ref_{name}":q_ref})
    grid.dae['h_dict'].update({f"v_ref_{name}":v_ref})

    grid.dae['xy_0_dict'].update({str(e_qv):1.0}) 

    p_W   = p * S_n
    q_var = q * S_n

    return p_W,q_var



def uvsg_mid(self,vsg_data):
    sin = sym.sin
    cos = sym.cos
    buses = self.data['buses']
    buses_list = [bus['name'] for bus in buses]
                
    bus_name = vsg_data['bus']
    
    if 'name' in vsg_data:
        name = vsg_data['name']
    else:
        name = bus_name
        
    for gen_id in range(100):
        if name not in self.generators_id_list:
            self.generators_id_list += [name]
            break
        else:
            name = name + f'_{gen_id}'
            
    vsg_data['name'] = name
                        
    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1
    
    # inputs
    V = sym.Symbol(f"V_{bus_name}", real=True)
    theta = sym.Symbol(f"theta_{bus_name}", real=True)
    omega_coi = sym.Symbol("omega_coi", real=True)   
    p_g = sym.Symbol(f"p_g_{name}", real=True)
    q_s_ref = sym.Symbol(f"q_s_ref_{name}", real=True)
    e_ref = sym.Symbol(f"e_ref_{name}", real=True) 
        
    # dynamic states
    delta = sym.Symbol(f"delta_{name}", real=True)
    e_qv = sym.Symbol(f"e_qv_{name}", real=True)
    xi_p = sym.Symbol(f"xi_p_{name}", real=True)
    xi_q = sym.Symbol(f"xi_q_{name}", real=True)
    e = sym.Symbol(f"e_{name}", real=True)

    # algebraic states
    omega = sym.Symbol(f"omega_{name}", real=True)
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)            
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)
    p_m = sym.Symbol(f"p_m_{name}", real=True)
    p_t = sym.Symbol(f"p_t_{name}", real=True)
    p_u = sym.Symbol(f"p_u_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
    K_p = sym.Symbol(f"K_p_{name}", real=True)        
    T_p = sym.Symbol(f"T_p_{name}", real=True)
    K_q = sym.Symbol(f"K_q_{name}", real=True)        
    T_q = sym.Symbol(f"T_q_{name}", real=True)
    X_v = sym.Symbol(f"X_v_{name}", real=True)
    R_v = sym.Symbol(f"R_v_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)
    T_e = sym.Symbol(f"T_e_{name}", real=True)
    K_e = sym.Symbol(f"K_e_{name}", real=True)
    params_list = ['S_n','Omega_b','K_p','T_p','K_q','T_q','X_v','R_v','R_s','T_e','K_e']
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    omega_s = omega_coi
    e_dv = 0
    epsilon_p = p_m - p_s
    epsilon_q = q_s_ref - q_s
                        
    
    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s)
    dxi_p = epsilon_p
    dxi_q = epsilon_q
    de = 1/T_e*(p_g-p_t)

    # algebraic equations   
    g_omega = -omega + K_p*(epsilon_p + xi_p/T_p)
    g_e_qv = -e_qv   + K_q*(epsilon_q + xi_q/T_q)
    g_i_d  = -R_v*i_d - X_v*i_q - v_d + e_dv 
    g_i_q  = -R_v*i_q + X_v*i_d - v_q + e_qv
    g_p_s  = i_d*v_d + i_q*v_q - p_s  
    g_q_s  = i_d*v_q - i_q*v_d - q_s 
    g_p_m  = -p_m + p_g + p_u
    g_p_t  = -p_t + i_d*(v_d + R_s*i_d) + i_q*(v_q + R_s*i_q)
    g_p_u  = -p_u + K_e*(e_ref - e)
    
    
    # dae 
    f_vsg = [ddelta,dxi_p,dxi_q,de]
    x_vsg = [ delta, xi_p, xi_q, e]
    g_vsg = [g_omega,g_e_qv,g_i_d,g_i_q,g_p_s,g_q_s,g_p_m,g_p_t,g_p_u]
    y_vsg = [  omega,  e_qv,  i_d,  i_q,  p_s,  q_s,  p_m,  p_t,  p_u]
    
    # T_p = K_p*2*H
    H = T_p/(2*K_p)
    self.H_total += H
    self.omega_coi_numerator += omega*H*S_n
    self.omega_coi_denominator += H*S_n

    self.dae['f'] += f_vsg
    self.dae['x'] += x_vsg
    self.dae['g'] += g_vsg
    self.dae['y_ini'] += y_vsg  
    self.dae['y_run'] += y_vsg  
    
    if 'q_s_ref' in vsg_data:
        self.dae['u_ini_dict'].update({f'{str(q_s_ref)}':vsg_data['q_s_ref']})
        self.dae['u_run_dict'].update({f'{str(q_s_ref)}':vsg_data['q_s_ref']})
    else:
        self.dae['u_ini_dict'].update({f'{str(q_s_ref)}':0.0})
        self.dae['u_run_dict'].update({f'{str(q_s_ref)}':0.0})
        
    self.dae['u_ini_dict'].update({f'{str(p_g)}':vsg_data['p_g']})
    self.dae['u_run_dict'].update({f'{str(p_g)}':vsg_data['p_g']})

    self.dae['u_ini_dict'].update({f'{str(e_ref)}':vsg_data['e_ref']})
    self.dae['u_run_dict'].update({f'{str(e_ref)}':vsg_data['e_ref']})
    
    self.dae['xy_0_dict'].update({str(omega):1.0})
    
    # grid power injection
    S_base = sym.Symbol('S_base', real = True)
    self.dae['g'][idx_bus*2]   += -p_s*S_n/S_base
    self.dae['g'][idx_bus*2+1] += -q_s*S_n/S_base
    
    # outputs
    #self.dae['h_dict'].update({f"p_e_{name}":p_e})
    
    for item in params_list:       
        self.dae['params_dict'].update({f"{item}_{name}":vsg_data[item]})


    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var

