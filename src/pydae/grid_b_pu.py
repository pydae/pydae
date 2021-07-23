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

    if 'vsgs' in data:   
        vsgs_add(grid)
        
    if 'uvsgs' in data:   
        uvsgs_add(grid)  
        
    if 'gformers_z' in data:           
        gformer_z_add(grid)
        
        # omega COI
        for gformer_z in data['gformers_z']:
            bus_name = gformer_z['bus']
            omega = sym.Symbol(f"omega_v_{bus_name}", real=True)
            omega_coi_n += omega 
            N_gformers += 1
        
    if 'gformers' in data:    
        #syns_add(grid)
        
        # omega COI
        for gformer in data['gformers']:
            bus_name = gformer['bus']
            omega = sym.Symbol(f"omega_{bus_name}", real=True)
            V = sym.Symbol(f"V_{bus_name}", real=True)
            V_ref = sym.Symbol(f"V_{bus_name}_ref", real=True)
            theta_ref = sym.Symbol(f"theta_{bus_name}_ref", real=True)
            params_grid.update({f"omega_{bus_name}":1.0}) 
            params_grid.update({f"V_{bus_name}_ref":1.0})  
            params_grid.update({f"theta_{bus_name}_ref":0.0}) 
            y_grid += [ V]
            g_grid += [-omega_coi + omega_coi_n/(N_syn+N_gformers)]
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
        grid['h'].update({f"p_e_{bus_name}":p_e})
        #for item in syn:       
        #    grid['params'].update({f"{item}_{bus_name}":syn[item]})
        #grid['params'].pop(f"bus_{bus_name}")
        #
        
        if 'avr' in syn:
            add_avr(grid,syn)
        if 'gov' in syn:
            add_gov(grid,syn,bus_i=buses[idx_bus]['idx_powers'])            
        if 'pss' in syn:
            add_pss(grid,syn)          

def add_avr(grid,syn_data,bus_i=''):
    
    if syn_data['avr']['type'] == 'sexs':
        sexs(grid,syn_data)

def add_gov(grid,syn_data,bus_i):
    
    if syn_data['gov']['type'] == 'tgov1':
        tgov1(grid,syn_data)

    if syn_data['gov']['type'] == 'agov1':
        agov1(grid,syn_data,bus_i=bus_i)

    if syn_data['gov']['type'] == 'hygov':
        hygov(grid,syn_data)

def add_pss(grid,syn_data):
    
    if syn_data['pss']['type'] == 'pss_kundur':
        pss_kundur(grid,syn_data)


def tgov1(grid,syn_data):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    # inpunts
    omega = sym.Symbol(f"omega_{bus_name}", real=True)
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    p_c = sym.Symbol(f"p_c_{bus_name}", real=True) 

    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{bus_name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{bus_name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{bus_name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{bus_name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{bus_name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{bus_name}", real=True)  # 10

    Droop = sym.Symbol(f"Droop_{bus_name}", real=True)  # 0.05
    
    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3

    g_p_m_ref  = -p_m_ref + p_c + p_r - 1/Droop*(omega - omega_ref)
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    
    grid['f'] += [dx_gov_1,dx_gov_2]
    grid['x'] += [ x_gov_1, x_gov_2]
    grid['g'] += [g_p_m_ref,g_p_m]
    grid['y'] += [  p_m_ref,  p_m]  
    grid['params'].update({str(Droop):gov_data['Droop']})
    grid['params'].update({str(T_1):gov_data['T_1']})
    grid['params'].update({str(T_2):gov_data['T_2']})
    grid['params'].update({str(T_3):gov_data['T_3']})
    grid['params'].update({str(omega_ref):1.0})

    grid['u'].update({str(p_c):gov_data['p_c']})


def agov1(grid,syn_data,bus_i = ''):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    # inpunts
    omega = sym.Symbol(f"omega_{bus_name}", real=True)
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    p_c = sym.Symbol(f"p_c_{bus_name}", real=True) 
    p_g = sym.Symbol(f"p_g_{bus_name}_{bus_i}", real=True)

    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{bus_name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{bus_name}", real=True)  
    xi_imw  = sym.Symbol(f"xi_imw_{bus_name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{bus_name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{bus_name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{bus_name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{bus_name}", real=True)  # 10

    Droop = sym.Symbol(f"Droop_{bus_name}", real=True)  # 0.05
    K_imw = sym.Symbol(f"K_imw_{bus_name}", real=True)  # 0.0

    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3
    dxi_imw =   K_imw*(p_c - p_g) - 1e-6*xi_imw

    g_p_m_ref  = -p_m_ref + p_c + xi_imw + p_r - 1/Droop*(omega - omega_ref) 

    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    
    grid['f'] += [dx_gov_1,dx_gov_2,dxi_imw]
    grid['x'] += [ x_gov_1, x_gov_2, xi_imw]
    grid['g'] += [g_p_m_ref,g_p_m]
    grid['y'] += [  p_m_ref,  p_m]  
    grid['params'].update({str(Droop):gov_data['Droop']})
    grid['params'].update({str(T_1):gov_data['T_1']})
    grid['params'].update({str(T_2):gov_data['T_2']})
    grid['params'].update({str(T_3):gov_data['T_3']})
    grid['params'].update({str(K_imw):gov_data['K_imw']})
    grid['params'].update({str(omega_ref):1.0})

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

def pss_kundur(grid,syn_data):
    bus_name = syn_data['bus']
    pss_data = syn_data['pss']
    
    omega = sym.Symbol(f"omega_{bus_name}", real=True)   
       
    x_wo  = sym.Symbol(f"x_wo_{bus_name}", real=True)
    x_lead  = sym.Symbol(f"x_lead_{bus_name}", real=True)

    z_wo  = sym.Symbol(f"z_wo_{bus_name}", real=True)
    x_lead  = sym.Symbol(f"x_lead_{bus_name}", real=True)
    
    T_wo = sym.Symbol(f"T_wo_{bus_name}", real=True)  
    T_1 = sym.Symbol(f"T_1_{bus_name}", real=True) 
    T_2 = sym.Symbol(f"T_2_{bus_name}", real=True)
    K_stab = sym.Symbol(f"K_stab_{bus_name}", real=True)

    v_pss = sym.Symbol(f"v_pss_{bus_name}", real=True) 
    
    
    u_wo = omega - 1.0
    
    dx_wo =   (u_wo - x_wo)/T_wo  # washout state
    dx_lead =  (z_wo - x_lead)/T_2      # lead compensator state
    
    g_z_wo =  (u_wo - x_wo) - z_wo  
    g_v_pss = K_stab*((z_wo - x_lead)*T_1/T_2 + x_lead) - v_pss  
    
    
    grid['f'] += [dx_wo,dx_lead]
    grid['x'] += [ x_wo, x_lead]
    grid['g'] += [g_z_wo,g_v_pss]
    grid['y'] += [  z_wo, v_pss]  
    grid['params'].update({str(T_wo):pss_data['T_wo']})
    grid['params'].update({str(T_1):pss_data['T_1']})
    grid['params'].update({str(T_2):pss_data['T_2']})
    grid['params'].update({str(K_stab):pss_data['K_stab']})



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


def vsgs_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for vsg in grid['data']['vsgs']:

        bus_name = vsg['bus']
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        p_g = sym.Symbol(f"p_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected active power (m-pu)
        q_g = sym.Symbol(f"q_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected reactive power (m-pu)
        V = sym.Symbol(f"V_{bus_name}", real=True)    # bus voltage module (pu)
        theta = sym.Symbol(f"theta_{bus_name}", real=True) # bus voltage angle (rad)
        i_d = sym.Symbol(f"i_d_{bus_name}", real=True)  # d-axe current (pu)
        i_q = sym.Symbol(f"i_q_{bus_name}", real=True)  # q-axe current (pu)
        delta = sym.Symbol(f"delta_{bus_name}", real=True)
        omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)
        x_wo = sym.Symbol(f"x_wo_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
        e_v = sym.Symbol(f"e_v_{bus_name}", real=True)
        p_d2 = sym.Symbol(f"p_d2_{bus_name}", real=True)
        i_d_ref = sym.Symbol(f"i_d_ref_{bus_name}", real=True)
        i_q_ref = sym.Symbol(f"i_q_ref_{bus_name}", real=True)
        q_ref = sym.Symbol(f"q_ref_{bus_name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{bus_name}", real=True)
        p_src = sym.Symbol(f"p_src_{bus_name}", real=True)
        soc_ref = sym.Symbol(f"soc_ref_{bus_name}", real=True)
        p_sto = sym.Symbol(f"p_sto_{bus_name}", real=True)
        Dp_ref = sym.Symbol(f"Dp_ref_{bus_name}", real=True)
        soc = sym.Symbol(f"soc_{bus_name}", real=True)
        xi_soc = sym.Symbol(f"xi_soc_{bus_name}", real=True)
        
        omega_coi = sym.Symbol("omega_coi", real=True)

        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        #S_n,H,Omega_b,R_v,X_v,D,K_delta = 0,0,0,0,0,0,0
        for item in ['S_n','R_s','H','Omega_b','R_v','X_v','D1','D2','D3','K_delta','T_wo','T_i','K_q','T_q','H_s','K_p_soc','K_i_soc']:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())
            grid['params'].update({f'{item}_{bus_name}':vsg[item]})

        p_t = i_d*(v_d + R_s*i_d) + i_q*(v_q + R_s*i_q) 

        omega_s = omega_coi
        
        u_wo = omega_v - 1.0
        epsilon_q = q_ref - q_g
        
        dx_wo  =  (u_wo - x_wo)/T_wo 
        g_p_d2 =  D2*(u_wo - x_wo) - p_d2  
        

        p_d1 = D1*(omega_v - 1)

        #  D3 = K_p*Omega_b
        ddelta   = Omega_b*(omega_v - omega_s) + D3*(p_m- p_g) - K_delta*delta
        domega_v = 1/(2*H)*(p_m - p_g - p_d1 - p_d2)
        di_d = 1/T_i*(i_d_ref - i_d)
        di_q = 1/T_i*(i_q_ref - i_q)
        dxi_q = epsilon_q
        dsoc = -p_sto/H_s
        dxi_soc = (soc_ref - soc)
        
        p_soc_ref = K_p_soc*(soc_ref - soc) + K_i_soc*xi_soc
        p_soc = sym.Piecewise((p_soc_ref - (Dp_ref + p_src),(soc<0.0) & (p_sto>0.0)),(p_soc_ref - (Dp_ref + p_src),(soc>1.0) & (p_sto<0.0)), (p_soc_ref, True))
        
        g_i_d_ref  = v_q + R_v*i_q_ref + X_v*i_d_ref - e_v
        g_i_q_ref  = v_d + R_v*i_d_ref - X_v*i_q_ref - 0
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        g_e_v = -e_v +  K_q*(epsilon_q + xi_q/T_q)
        g_p_sto = -p_t + p_sto + p_src 
        g_p_m = -p_m + p_soc + Dp_ref + p_src
        #dv_dc = 1/C_dc*(-i_t + i_src + i_sto)
        #dv_dc = 1/(v_dc*C_dc)*(-p_t + p_src + p_sto)
        
        f_syn = [ddelta,domega_v,dx_wo,di_d,di_q,dxi_q, dsoc,dxi_soc]
        x_syn = [ delta, omega_v, x_wo, i_d, i_q, xi_q,  soc, xi_soc]
        g_syn = [g_i_d_ref,g_i_q_ref,g_p_g,g_q_g,g_p_d2,g_e_v, g_p_sto,g_p_m]
        y_syn = [  i_d_ref,  i_q_ref,  p_g,  q_g,  p_d2,  e_v,   p_sto,  p_m]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['u'].update({f'p_in_{bus_name}':0.0})
        grid['u'].update({f'Dp_ref_{bus_name}':0.0})
        grid['u'].update({f'q_ref_{bus_name}':0.0})
        grid['u'].update({f'p_src_{bus_name}':vsg['p_src']})
        grid['u'].update({f'soc_ref_{bus_name}':vsg['soc_ref']})
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base
        grid['h'].update({f"p_t_{bus_name}":p_t})
        grid['h'].update({f"p_soc_{bus_name}":p_soc})


def uvsgs_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for uvsg in grid['data']['uvsgs']:

        bus_name = uvsg['bus']
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
        omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
        e_v = sym.Symbol(f"e_v_{bus_name}", real=True)
        i_d_ref = sym.Symbol(f"i_d_ref_{bus_name}", real=True)
        i_q_ref = sym.Symbol(f"i_q_ref_{bus_name}", real=True)
        q_ref = sym.Symbol(f"q_ref_{bus_name}", real=True)
        xi_p = sym.Symbol(f"xi_p_{bus_name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{bus_name}", real=True)
      
        omega_coi = sym.Symbol("omega_coi", real=True)

        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        #S_n,H,Omega_b,R_v,X_v,D,K_delta = 0,0,0,0,0,0,0
        for item in ['S_n','Omega_b','R_v','X_v','K_delta','T_i','K_q','T_q','K_p','K_i']:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())
            grid['params'].update({f'{item}_{bus_name}':uvsg[item]})

        p_e = i_d*(v_d + R_v*i_d) + i_q*(v_q + R_v*i_q) 

        omega_s = omega_coi
        
        epsilon_p = p_m - p_g
        epsilon_q = q_ref - q_g
        
        # digsilent: omega_v = K_p*(epsilon_p + xi_p/T_p) + 1.0  ! UVSG PI             
        # digsilent: phi_v. =  2.0*pi()*50.0*(omega_v-fref)  
   

        omega_v = (K_p*epsilon_p + K_i*xi_p)  + 1.0

        ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
        dxi_p = epsilon_p
        di_d = 1/T_i*(i_d_ref - i_d)
        di_q = 1/T_i*(i_q_ref - i_q)
        dxi_q = epsilon_q
        
        
        g_i_d_ref  = v_q + R_v*i_q_ref + X_v*i_d_ref - e_v
        g_i_q_ref  = v_d + R_v*i_d_ref - X_v*i_q_ref - 0
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        g_e_v = -e_v +  K_q*(epsilon_q + xi_q/T_q)
        
        f_syn = [ddelta,dxi_p,di_d,di_q,dxi_q]
        x_syn = [ delta, xi_p, i_d, i_q, xi_q]
        g_syn = [g_i_d_ref,g_i_q_ref,g_p_g,g_q_g,g_e_v]
        y_syn = [  i_d_ref,  i_q_ref,  p_g,  q_g,  e_v]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['u'].update({f'q_ref_{bus_name}':0.0})
        grid['u'].update({f'p_m_{bus_name}':uvsg['p_m']})
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base
        grid['h'].update({f"p_e_{bus_name}":p_e})
        grid['h'].update({f"omega_v_{bus_name}":omega_v})

def gformer_z_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for gformer_z in grid['data']['gformers_z']:

        bus_name = gformer_z['bus']
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        p_g = sym.Symbol(f"p_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected active power (m-pu)
        q_g = sym.Symbol(f"q_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected reactive power (m-pu)
        V = sym.Symbol(f"V_{bus_name}", real=True)    # bus voltage module (pu)
        theta   = sym.Symbol(f"theta_{bus_name}", real=True) # bus voltage angle (rad)
        theta_v_0 = sym.Symbol(f"theta_v_0_{bus_name}", real=True) # bus voltage angle (rad)

        i_d = sym.Symbol(f"i_d_{bus_name}", real=True)  # d-axe current (pu)
        i_q = sym.Symbol(f"i_q_{bus_name}", real=True)  # q-axe current (pu)
        delta = sym.Symbol(f"delta_{bus_name}", real=True)
        omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)
        x_wo = sym.Symbol(f"x_wo_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
        e_v = sym.Symbol(f"e_v_{bus_name}", real=True)
        p_d2 = sym.Symbol(f"p_d2_{bus_name}", real=True)
        i_d_ref = sym.Symbol(f"i_d_ref_{bus_name}", real=True)
        i_q_ref = sym.Symbol(f"i_q_ref_{bus_name}", real=True)
        q_ref = sym.Symbol(f"q_ref_{bus_name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{bus_name}", real=True)
        p_src = sym.Symbol(f"p_src_{bus_name}", real=True)
        soc_ref = sym.Symbol(f"soc_ref_{bus_name}", real=True)
        p_sto = sym.Symbol(f"p_sto_{bus_name}", real=True)
        Dp_ref = sym.Symbol(f"Dp_ref_{bus_name}", real=True)
        soc = sym.Symbol(f"soc_{bus_name}", real=True)
        xi_soc = sym.Symbol(f"xi_soc_{bus_name}", real=True)
        
        omega_coi = sym.Symbol("omega_coi", real=True)

        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        #S_n,H,Omega_b,R_v,X_v,D,K_delta = 0,0,0,0,0,0,0
        for item in ['S_n','Omega_b','R_v','X_v','T_i','K_delta']:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())
            grid['params'].update({f'{item}_{bus_name}':gformer_z[item]})

        omega_s = omega_coi
        
        ddelta   = Omega_b*(omega_v - omega_s)  - K_delta*delta
        di_d = 1/T_i*(i_d_ref - i_d)
        di_q = 1/T_i*(i_q_ref - i_q)
          
        g_i_d_ref  = v_q + R_v*i_q_ref + X_v*i_d_ref - e_v
        g_i_q_ref  = v_d + R_v*i_d_ref - X_v*i_q_ref - 0
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        
        f_syn = [ ddelta,di_d,di_q]
        x_syn = [  delta, i_d, i_q]
        g_syn = [g_i_d_ref,g_i_q_ref,g_p_g,g_q_g]
        y_syn = [  i_d_ref,  i_q_ref,  p_g,  q_g]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['u'].update({f'e_v_{bus_name}':1.0})
        grid['u'].update({f'omega_v_{bus_name}':1.0})
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base
 
        
        