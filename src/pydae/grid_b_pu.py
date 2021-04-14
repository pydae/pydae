# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:52:55 2021

@author: jmmau
"""

import numpy as np
import sympy as sym
import json

def bal_pu(data_input):
    '''
    

    Parameters
    ----------
    data_input : string or dict
        File path to the system data information or dictionar y with the information.

    Returns
    -------
    dict
        Dictionary with the equations for pydae. 
        
    {
     'sys':{'name':'pf_1','S_base':100e6},       
     'buses':[{'name':'GRI','P_W':0.0,'Q_var':0.0,'U_kV':66.0, 'type':'slack'},
              {'name':'POI','P_W':0.0,'Q_var':0.0,'U_kV':66.0},
              {'name':'PMV','P_W':0.0,'Q_var':0.0,'U_kV':20.0}],
     'lines':[{'bus_j':'GRI','bus_k':'POI','X_km':0.4,'R_km':0.12,'km':20},
              {'bus_j':'POI','bus_k':'PMV','X_pu':0.04,'R_pu':0.01, 'S_mva':50.0}]
    }
        

    '''

    if type(data_input) == str:
        with open(data_input,'r') as fobj:
            data = json.loads(fobj.read().replace("'",'"'))
    elif type(data_input) == dict:
        data = data_input

    sys = data['sys']
    buses = data['buses']
    lines = data['lines']
    x_grid = []
    f_grid = []

    params_grid = {'S_base':sys['S_base']}
    S_base = sym.Symbol("S_base", real=True) 
    N_bus = len(buses)
    N_branch = len(lines)
    A = sym.zeros(3*N_branch,N_bus)
    G_primitive = sym.zeros(3*N_branch,3*N_branch)
    B_primitive = sym.zeros(3*N_branch,3*N_branch)
    buses_list = [bus['name'] for bus in buses]
    it = 0
    for line in lines:

        bus_j = line['bus_j']
        bus_k = line['bus_k']

        idx_j = buses_list.index(bus_j)
        idx_k = buses_list.index(bus_k)    

        A[3*it,idx_j] = 1
        A[3*it,idx_k] =-1   
        A[3*it+1,idx_j] = 1
        A[3*it+2,idx_k] = 1   
        
        line_name = f"{bus_j}_{bus_k}"
        g_jk = sym.Symbol(f"g_{line_name}", real=True) 
        b_jk = sym.Symbol(f"b_{line_name}", real=True) 
        bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
        G_primitive[3*it,3*it] = g_jk
        B_primitive[3*it,3*it] = b_jk
        B_primitive[3*it+1,3*it+1] = bs_jk/2
        B_primitive[3*it+2,3*it+2] = bs_jk/2
        
        if 'X_pu' in line:
            if 'S_mva' in line: S_line = 1e6*line['S_mva']
            R = line['R_pu']*sys['S_base']/S_line  # in pu of the system base
            X = line['X_pu']*sys['S_base']/S_line  # in pu of the system base
            G =  R/(R**2+X**2)
            B = -X/(R**2+X**2)
            params_grid.update({f"g_{line_name}":G})
            params_grid.update({f'b_{line_name}':B})

        if 'X' in line:
            bus_idx = buses_list.index(line['bus_j'])
            U_base = buses[bus_idx]['U_kV']
            Z_base = U_base**2/sys['S_base']
            R = line['R']/Z_base  # in pu of the system base
            X = line['X']/Z_base  # in pu of the system base
            G =  R/(R**2+X**2)
            B = -X/(R**2+X**2)
            params_grid.update({f"g_{line_name}":G})
            params_grid.update({f'b_{line_name}':B})

        if 'X_km' in line:
            bus_idx = buses_list.index(line['bus_j'])
            U_base = buses[bus_idx]['U_kV']*1000
            Z_base = U_base**2/sys['S_base']
            R = line['R_km']*line['km']/Z_base  # in pu of the system base
            X = line['X_km']*line['km']/Z_base  # in pu of the system base
            G =  R/(R**2+X**2)
            B = -X/(R**2+X**2)
            params_grid.update({f"g_{line_name}":G})
            params_grid.update({f'b_{line_name}':B})        

        params_grid.update({f'bs_{line_name}':0.0})
        if 'Bs_pu' in line:
            if 'S_mva' in line: S_line = 1e6*line['S_mva']
            Bs = line['Bs_pu']*S_line/sys['S_base']  # in pu of the system base
            bs = -Bs/2.0
            params_grid[f'bs_{line_name}'] = bs
 
        if 'Bs_km' in line:
            bus_idx = buses_list.index(line['bus_j'])
            U_base = buses[bus_idx]['U_kV']*1000
            Z_base = U_base**2/sys['S_base']
            Y_base = 1.0/Z_base
            Bs = line['Bs_km']*line['km']/Y_base # in pu of the system base
            bs = Bs 
            params_grid[f'bs_{line_name}'] = bs
            
        it += 1


    G = A.T * G_primitive * A
    B = A.T * B_primitive * A    
    
    sin = sym.sin
    cos = sym.cos
    y_grid = []
    g = sym.zeros(2*N_bus,1)
    u_grid = {}
    h_grid = {}
    for j in range(N_bus):
        bus_j_name = buses_list[j]
        P_j = sym.Symbol(f"P_{bus_j_name}", real=True)
        Q_j = sym.Symbol(f"Q_{bus_j_name}", real=True)
        g[2*j]   = -P_j/S_base
        g[2*j+1] = -Q_j/S_base
        for k in range(N_bus): 

            bus_k_name = buses_list[k]
            V_j = sym.Symbol(f"V_{bus_j_name}", real=True) 
            V_k = sym.Symbol(f"V_{bus_k_name}", real=True) 
            theta_j = sym.Symbol(f"theta_{bus_j_name}", real=True) 
            theta_k = sym.Symbol(f"theta_{bus_k_name}", real=True) 
            g[2*j]   += V_j*V_k*(G[j,k]*cos(theta_j - theta_k) + B[j,k]*sin(theta_j - theta_k)) 
            g[2*j+1] += V_j*V_k*(G[j,k]*sin(theta_j - theta_k) - B[j,k]*cos(theta_j - theta_k))        
            h_grid.update({f"V_{bus_j_name}":V_j})
        bus = buses[j]
        bus_name = bus['name']
        if 'type' in bus:
            if bus['type'] == 'slack':
                y_grid += [P_j]
                y_grid += [Q_j]
                u_grid.update({f"V_{bus_name}":1.0})
                u_grid.update({f"theta_{bus_name}":0.0})  
        else:
            y_grid += [V_j]
            y_grid += [theta_j]        
            u_grid.update({f"P_{bus_name}":bus['P_W']})
            u_grid.update({f"Q_{bus_name}":bus['Q_var']})    
            
        params_grid.update({f'U_{bus_name}_n':bus['U_kV']*1000})
    g_grid = list(g)     

    if False:
        v_sym_list = []
        for bus in buses_list:
            V_m = sym.Symbol(f'V_{bus}',real=True)
            V_a = sym.Symbol(f'theta_{bus}',real=True)
            v_sym_list += [V_m*sym.exp(sym.I*V_a)]

        sym.Matrix(v_sym_list)

        I_lines = (G_primitive+1j*B_primitive) * A * sym.Matrix(v_sym_list)

        it = 0
        for line in lines:
            I_jk_r = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_r", real=True)
            I_jk_i = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_i", real=True)
            g_grid += [-I_jk_r + sym.re(I_lines[it])]
            g_grid += [-I_jk_i + sym.im(I_lines[it])]
            y_grid += [I_jk_r]
            y_grid += [I_jk_i]
            it += 1
            
    grid = {'g':g_grid,'y':y_grid,'u':u_grid,'h':h_grid, 'x':x_grid, 'f':f_grid,
            'params':params_grid, 'data':data, 'A':A, 'B_primitive':B_primitive} 
    
    N_syn = 0
    N_gformers = 0
    
    omega_coi_n = 0
    omega_coi = sym.Symbol("omega_coi", real=True)  
     
    if 'syns' in data:    
        syns_add(grid)
        
        # omega COI
        for syn in data['syns']:
            bus_name = syn['bus']
            omega = sym.Symbol(f"omega_{bus_name}", real=True)            
            omega_coi_n += omega 
            N_syn += 1

        # secondary frequency control
        xi_freq = sym.Symbol("xi_freq", real=True) 
        for syn in data['syns']:
            bus_name = syn['bus']
            p_r = sym.Symbol(f"p_r_{bus_name}", real=True)            
            K_sec = sym.Symbol(f"K_sec_{bus_name}", real=True) 
            
            y_grid += [  p_r]
            g_grid += [ -p_r + K_sec*xi_freq/N_syn]
            params_grid.update({str(K_sec):syn['K_sec']})
        x_grid += [ xi_freq]
        f_grid += [ 1-omega_coi]  
        
        
    if 'gformers' in data:    
        #syns_add(grid)
        
        # omega COI
        for gformer in data['gformers']:
            bus_name = gformer['bus']
            omega = sym.Symbol(f"omega_{bus_name}", real=True)            
            omega_coi_n += omega 
            N_gformers += 1
       
    y_grid += [ omega_coi]
    g_grid += [-omega_coi + omega_coi_n/(N_syn+N_gformers)]
        
        
    return grid


def syns_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for syn in grid['data']['syns']:

        bus_name = syn['bus']
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        p_g = sym.Symbol(f"p_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True)
        q_g = sym.Symbol(f"q_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True)
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        i_d = sym.Symbol(f"i_d_{bus_name}", real=True)
        i_q = sym.Symbol(f"i_q_{bus_name}", real=True)
        delta = sym.Symbol(f"delta_{bus_name}", real=True)
        omega = sym.Symbol(f"omega_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
        e1q = sym.Symbol(f"e1q_{bus_name}", real=True)
        e1d = sym.Symbol(f"e1d_{bus_name}", real=True)
        v_f = sym.Symbol(f"v_f_{bus_name}", real=True)
        
        omega_coi = sym.Symbol("omega_coi", real=True)

        p_m_ref  = sym.Symbol(f"p_m_ref_{bus_name}", real=True)
        v_ref  = sym.Symbol(f"v_ref_{bus_name}", real=True)
        xi_m = sym.Symbol(f"xi_m_{bus_name}", real=True)
        
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 

        for item in ['S_n','H','Omega_b','T1d0','T1q0','X_d','X_q','X1d','X1q',
                     'D','R_a','K_delta']:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())
            grid['params'].update({f'{item}_{bus_name}':syn[item]})

        p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q) 

        omega_s = omega_coi
        
        ddelta = Omega_b*(omega - omega_s) - K_delta*delta
        domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
        de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
        de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)

        
        dxi_m =   omega - 1

        g_id  = v_q + R_a*i_q + X1d*i_d - e1q
        g_iq  = v_d + R_a*i_d - X1q*i_q - e1d
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        
        f_syn = [ddelta,domega,de1q,de1d]
        x_syn = [ delta, omega, e1q, e1d]
        g_syn = [g_id,g_iq,g_p_g,g_q_g]
        y_syn = [i_d,i_q,p_g,q_g]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base
        
        #for item in syn:       
        #    grid['params'].update({f"{item}_{bus_name}":syn[item]})
        #grid['params'].pop(f"bus_{bus_name}")
        #
        
        if 'avr' in syn:
            add_avr(grid,syn)
        if 'gov' in syn:
            add_gov(grid,syn)            
        
        
def add_avr(grid,syn_data):
    
    if syn_data['avr']['type'] == 'sexs':
        sexs(grid,syn_data)

def add_gov(grid,syn_data):
    
    if syn_data['gov']['type'] == 'tgov1':
        tgov1(grid,syn_data)
    if syn_data['gov']['type'] == 'hygov':
        hygov(grid,syn_data)
        
def tgov1(grid,syn_data):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    omega = sym.Symbol(f"omega_{bus_name}", real=True)
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    p_c = sym.Symbol(f"p_c_{bus_name}", real=True)  
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{bus_name}", real=True)  
    T_m = sym.Symbol(f"T_m_{bus_name}", real=True) 
    Droop = sym.Symbol(f"Droop_{bus_name}", real=True) 
    
    dp_m =   (p_m_ref - p_m)/T_m
    g_p_m_ref  = -p_m_ref + p_c + p_r - 1/Droop*(omega - 1)
    
    grid['f'] += [dp_m]
    grid['x'] += [ p_m]
    grid['g'] += [g_p_m_ref]
    grid['y'] += [  p_m_ref]  
    grid['params'].update({str(Droop):gov_data['Droop']})
    grid['params'].update({str(T_m):gov_data['T_m']})
    grid['u'].update({str(p_c):gov_data['p_c']})

def hygov(grid,syn_data):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)
    omega = sym.Symbol(f"omega_{bus_name}", real=True)

    # dynamic states:
    xi_omega = sym.Symbol(f"xi_omega_{bus_name}", real=True)
    servo = sym.Symbol(f"servo_{bus_name}", real=True)
    pos = sym.Symbol(f"pos_{bus_name}", real=True)
    flow = sym.Symbol(f"flow_{bus_name}", real=True)
    # algebraic states:    
    servo_u = sym.Symbol(f"servo_u_{bus_name}", real=True)
    gate = sym.Symbol(f"gate_{bus_name}", real=True)
    head = sym.Symbol(f"head_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
    # parameters:
    Droop = sym.Symbol(f"Droop_{bus_name}", real=True) 
    K_p_gov = sym.Symbol(f"K_p_gov_{bus_name}", real=True)
    K_i_gov = sym.Symbol(f"K_i_gov_{bus_name}", real=True) 
    K_servo = sym.Symbol(f"K_servo_{bus_name}", real=True)  
    T_servo = sym.Symbol(f"T_servo_{bus_name}", real=True) 
    V_gate_max = sym.Symbol(f"V_gate_max_{bus_name}", real=True)
    Gate_max = sym.Symbol(f"Gate_max_{bus_name}", real=True)
    T_w = sym.Symbol(f"T_w_{bus_name}", real=True) 
    Flow_nl = sym.Symbol(f"Flow_nl_{bus_name}", real=True) 
    A_t = sym.Symbol(f"A_t_{bus_name}", real=True)     
    
    epsilon_omega = omega_ref - omega #- Droop*gate + p_r
    dxi_omega = epsilon_omega
    g_servo_u = -servo_u + K_p_gov*epsilon_omega + K_i_gov*xi_omega
    dservo = (K_servo*(servo_u - gate) - servo)/T_servo
    dpos = servo #sym.Piecewise((V_gate_max,servo>V_gate_max),(-V_gate_max,servo<-V_gate_max),(servo,True))
    g_gate = -gate + servo_u # + pos #sym.Piecewise((Gate_max,pos>Gate_max),(1e-6,pos<1e-6),(pos,True))
    dflow =   (gate - flow)/T_w
    g_head = -(gate - flow)/T_w + flow - head  
    g_p_m = -p_m + A_t*head




    grid['f'] += [dxi_omega, dflow] #[dxi_omega, dservo, dpos, dflow]
    grid['x'] += [ xi_omega,  flow]
    grid['g'] += [g_servo_u, g_gate, g_head, g_p_m] # [, g_gate, g_head, g_p_m]
    grid['y'] += [  servo_u,   gate,   head, p_m] # [  servo_u,   gate,   head,   p_m]
    grid['params'].update({str(Droop):gov_data['Droop'],str(K_p_gov):gov_data['K_p_gov'],str(K_i_gov):gov_data['K_i_gov']})
    grid['params'].update({str(K_servo):gov_data['K_servo'],str(T_servo):gov_data['T_servo']})
    grid['params'].update({str(V_gate_max):gov_data['V_gate_max'],str(Gate_max):gov_data['Gate_max']})
    grid['params'].update({str(T_w):gov_data['T_w'],str(Flow_nl):gov_data['Flow_nl']})
    grid['params'].update({str(A_t):gov_data['A_t']})
    grid['u'].update({str(omega_ref):gov_data['omega_ref']})
    

def sexs(grid,syn_data):
    bus_name = syn_data['bus']
    avr_data = syn_data['avr']
    
    v_t = sym.Symbol(f"V_{bus_name}", real=True)   
    v_c = sym.Symbol(f"v_c_{bus_name}", real=True)  
    xi_v  = sym.Symbol(f"xi_v_{bus_name}", real=True)
    v_f = sym.Symbol(f"v_f_{bus_name}", real=True)  
    T_r = sym.Symbol(f"T_r_{bus_name}", real=True) 
    K_a = sym.Symbol(f"K_a_{bus_name}", real=True)
    K_ai = sym.Symbol(f"K_ai_{bus_name}", real=True)
    v_ref = sym.Symbol(f"v_ref_{bus_name}", real=True) 
    v_pss = sym.Symbol(f"v_pss_{bus_name}", real=True) 
    
    dv_c =   (v_t - v_c)/T_r
    dxi_v =   v_ref - v_t  
    g_v_f  = K_a*(v_ref - v_c + v_pss) + K_ai*xi_v  - v_f 
    
    grid['f'] += [dv_c,dxi_v]
    grid['x'] += [ v_c, xi_v]
    grid['g'] += [g_v_f]
    grid['y'] += [v_f]  
    grid['params'].update({str(K_a):avr_data['K_a']})
    grid['params'].update({str(K_ai):avr_data['K_ai']})
    grid['params'].update({str(T_r):avr_data['T_r']})  
    grid['u'].update({str(v_ref):avr_data['v_ref']})
    grid['u'].update({str(v_pss):avr_data['v_pss']})
    
    
def hygov_original(grid,syn_data):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)
    omega = sym.Symbol(f"omega_{bus_name}", real=True)

    # dynamic states:
    xi_omega = sym.Symbol(f"xi_omega_{bus_name}", real=True)
    servo = sym.Symbol(f"servo_{bus_name}", real=True)
    pos = sym.Symbol(f"pos_{bus_name}", real=True)
    flow = sym.Symbol(f"flow_{bus_name}", real=True)
    # algebraic states:    
    servo_u = sym.Symbol(f"servo_u_{bus_name}", real=True)
    gate = sym.Symbol(f"gate_{bus_name}", real=True)
    head = sym.Symbol(f"head_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
    # parameters:
    Droop = sym.Symbol(f"Droop_{bus_name}", real=True) 
    K_p_gov = sym.Symbol(f"K_p_gov_{bus_name}", real=True)
    K_i_gov = sym.Symbol(f"K_i_gov_{bus_name}", real=True) 
    K_servo = sym.Symbol(f"K_servo_{bus_name}", real=True)  
    T_servo = sym.Symbol(f"T_servo_{bus_name}", real=True) 
    V_gate_max = sym.Symbol(f"V_gate_max_{bus_name}", real=True)
    Gate_max = sym.Symbol(f"Gate_max_{bus_name}", real=True)
    T_w = sym.Symbol(f"T_w_{bus_name}", real=True) 
    Flow_nl = sym.Symbol(f"Flow_nl_{bus_name}", real=True) 
    A_t = sym.Symbol(f"A_t_{bus_name}", real=True)     
    
    epsilon_omega = omega_ref - omega - Droop*gate + p_r
    dxi_omega = epsilon_omega
    g_servo_u = -servo_u + K_p_gov*epsilon_omega + K_i_gov*xi_omega
    dservo = (K_servo*(servo_u-gate) - servo)/T_servo
    dpos = servo #sym.Piecewise((V_gate_max,servo>V_gate_max),(-V_gate_max,servo<-V_gate_max),(servo,True))
    g_gate = -gate + K_p_gov*epsilon_omega # + pos #sym.Piecewise((Gate_max,pos>Gate_max),(1e-6,pos<1e-6),(pos,True))
    dflow =  (1-head)/T_w
    g_head = -head + (gate/(flow+1e-6))**2 
    g_p_m = -p_m + A_t*head*(flow - Flow_nl)

    grid['f'] += [dxi_omega, dservo, dpos, dflow] #[]
    grid['x'] += [ xi_omega,  servo,  pos,  flow]
    grid['g'] += [g_servo_u, g_gate, g_head, g_p_m] # [g_servo_u, g_gate, g_head, g_p_m]
    grid['y'] += [  servo_u,   gate,   head,   p_m] # [  servo_u,   gate,   head,   p_m]
    grid['params'].update({str(Droop):gov_data['Droop'],str(K_p_gov):gov_data['K_p_gov'],str(K_i_gov):gov_data['K_i_gov']})
    grid['params'].update({str(K_servo):gov_data['K_servo'],str(T_servo):gov_data['T_servo']})
    grid['params'].update({str(V_gate_max):gov_data['V_gate_max'],str(Gate_max):gov_data['Gate_max']})
    grid['params'].update({str(T_w):gov_data['T_w'],str(Flow_nl):gov_data['Flow_nl']})
    grid['params'].update({str(A_t):gov_data['A_t']})
    grid['u'].update({str(omega_ref):gov_data['omega_ref']})