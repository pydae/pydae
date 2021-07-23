#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:53:28 2018

@author: jmmauricio
hola
"""

import numpy as np
import sympy as sym
#from sympy.diffgeom import TensorProduct
from sympy.physics.quantum  import TensorProduct
import json



def data_processing(data):
    for line in data['lines']:
        if 'X' in line: 
            L = line['X']/(2*np.pi*data['system']['f_hz'])
            line.update({'L':L})
        if 'B' in line: 
            C = line['B']/(2*np.pi*data['system']['f_hz'])
            line.update({'C':C})  
            
        if 'X_km' in line: 
            L = line['km']*line['X_km']/(2*np.pi*data['system']['f_hz'])
            line.update({'L':L})

        if 'R_km' in line: 
            R = line['km']*line['R_km']
            line.update({'R':R})
            
        if 'B_muS_km' in line: 
            B = line['km'] * line['B_muS_km']*1e-6
            C = B/(2*np.pi*data['system']['f_hz'])
            line.update({'C':C})   
            
        if 'B_km' in line: 
            B = line['km'] * line['B_km']
            C = B/(2*np.pi*data['system']['f_hz'])
            line.update({'C':C})   
            
    for load in data['loads']:
        if 'I_max' not in load: 
            if 'kVA' in load:
                I_max = load['kVA']*1000/690
            load['I_max'] = I_max
        if 'T_i' not in line: 
            load['T_i'] = 0.01        
    return data

def grid2dae_dq(data_input, park_type='original',dq_name='DQ'):
    
    if dq_name == 'DQ':
        D_ = 'D'
        Q_ = 'Q'        

    if type(data_input) == str:
        json_file = data_input
        json_data = open(json_file).read().replace("'",'"')
        data = json.loads(json_data)
    elif type(data_input) == dict:
        data = data_input


    data_processing(data)
    model_type = data['system']['model_type']
    buses = data['buses']
    lines = data['lines']
    loads = data['loads']
    grid_formers = data['grid_formers']

    buses_list = [item['bus'] for item in buses]
    load_buses_list = [item['bus'] for item in loads]
    gformers_buses_list = [item['bus'] for item in grid_formers]

    params = {}
    u_grid = {}
    f_grid = []
    g_grid = []
    y_grid_list = []
    x_grid_list = []
    x_list = []
    y_list = []
    
    omega = sym.Symbol('omega', real=True)
    M = len(lines) # total number of branches
    N = len(buses) # total number of buses
    A = sym.Matrix.zeros(M,cols=N)

    i_l_list = []
    i_list = []
    v_list = []
    R_list = []
    L_list = []
    C_list = [0]*N

    itl = 0
    for line in lines:
        sub_name = f"{line['bus_j']}{line['bus_k']}"

        idx_bus_j = buses_list.index(line['bus_j'])
        idx_bus_k = buses_list.index(line['bus_k'])

        A[itl,idx_bus_j] =  1
        A[itl,idx_bus_k] = -1

        bus_j = line['bus_j']
        bus_k = line['bus_k']    

        R_ij = sym.Symbol(f'R_{sub_name}', real=True)
        L_ij = sym.Symbol(f'L_{sub_name}', real=True)

        R_list += [R_ij]*2
        L_list += [L_ij]*2

        i_l_d = sym.Symbol(f'i_l_{sub_name}_{D_}', real=True)
        i_l_q = sym.Symbol(f'i_l_{sub_name}_{Q_}', real=True)
        i_l_list += [i_l_d,i_l_q]

        C_ij = sym.Symbol(f'C_{sub_name}', real=True)
        C_list[idx_bus_j] += C_ij/2
        C_list[idx_bus_k] += C_ij/2

        # parameters
        R_name = f'R_{sub_name}'
        R_value = line['R']
        L_name = f'L_{sub_name}'
        L_value = line['L']
        C_name = f'C_{sub_name}'
        C_value = line['C']

        params.update({R_name:R_value,L_name:L_value,C_name:C_value})

        itl += 1


    C_e_list = []
    for item in C_list:
        C_e_list += [item]
        C_e_list += [item]

    for bus in buses:
        bus_name = bus['bus']
        v_d = sym.Symbol(f'v_{bus_name}_{D_}', real=True)
        v_q = sym.Symbol(f'v_{bus_name}_{Q_}', real=True)
        i_d = sym.Symbol(f'i_{bus_name}_{D_}', real=True)
        i_q = sym.Symbol(f'i_{bus_name}_{Q_}', real=True)

        v_list += [v_d,v_q]
        i_list += [i_d,i_q]


    i_l_dq = sym.Matrix(i_l_list)
    R_e = sym.Matrix.diag(R_list)
    L_e = sym.Matrix.diag(L_list)
    Omega_list = sym.Matrix([[0,omega],[-omega,0]])
    Omega_e_M = sym.Matrix.diag([sym.Matrix([[0,omega],[-omega,0]])]*M)
    Omega_e_N = sym.Matrix.diag([sym.Matrix([[0,omega],[-omega,0]])]*N)

    C_e = sym.Matrix.diag(C_e_list)

    v_dq = sym.Matrix(v_list)
    i_dq = sym.Matrix(i_list)
    def T(P):
        u = TensorProduct(sym.Matrix.eye(P),sym.Matrix([1,0]).T)
        l = TensorProduct(sym.Matrix.eye(P),sym.Matrix([0,1]).T)
        return sym.Matrix([u,l])

    A_e = T(M).inv() @ sym.Matrix.diag([A,A]) @ T(N)

    if park_type == 'fisix':
        di_l_dq =  (-(R_e + L_e @ Omega_e_M) @ i_l_dq + A_e @ v_dq)
        dv_dq =  (-C_e @ Omega_e_N @ v_dq - A_e.T @ i_l_dq + i_dq)
    if park_type == 'original':
        di_l_dq =  (-(R_e - L_e @ Omega_e_M) @ i_l_dq + A_e @ v_dq)
        dv_dq =  (C_e @ Omega_e_N @ v_dq - A_e.T @ i_l_dq + i_dq)

    if model_type == 'ode':
        f_grid += list(L_e.inv()*di_l_dq)
        f_grid += list(C_e.inv()*dv_dq)
        x_grid_list += list(i_l_dq)                    # items as sym.Symbol
        x_grid_list += list(v_dq)                      # items as sym.Symbol
        x_list = [str(item) for item in x_grid_list]   # items as str
        
        for gformer in grid_formers:
            bus = gformer['bus']
            idx_D = x_list.index(f'v_{bus}_D')
            f_grid.pop(idx_D)
            x_grid_list.pop(idx_D)
            x_list.pop(idx_D)
            u_grid.update({f'v_{bus}_D':gformer["V_phph"]*np.sqrt(2/3)*np.sin(np.deg2rad(gformer["deg"]))})
            idx_Q = x_list.index(f'v_{bus}_Q')
            f_grid.pop(idx_Q)
            x_grid_list.pop(idx_Q)
            x_list.pop(idx_Q)
            u_grid.update({f'v_{bus}_Q':gformer["V_phph"]*np.sqrt(2/3)*np.cos(np.deg2rad(gformer["deg"]))})
        
    if model_type == 'dae':
        f_grid += list(L_e.inv()*di_l_dq)
        g_grid += list(dv_dq)
        x_grid_list += list(i_l_dq)                   # items as sym.Symbol
        y_grid_list += list(v_dq)                     # items as sym.Symbol
        x_list = [str(item) for item in x_grid_list]  # items as str
        y_list = [str(item) for item in y_grid_list]  # items as str
        
        for gformer in grid_formers:
            bus = gformer['bus']
            idx_D = y_list.index(f'v_{bus}_D')
            g_grid.pop(idx_D)
            y_grid_list.pop(idx_D)
            y_list.pop(idx_D)
            u_grid.update({f'v_{bus}_D':gformer["V_phph"]*np.sqrt(2/3)*np.sin(np.deg2rad(gformer["deg"]))})
            idx_Q = y_list.index(f'v_{bus}_Q')
            g_grid.pop(idx_Q)
            y_grid_list.pop(idx_Q)
            y_list.pop(idx_Q)
            u_grid.update({f'v_{bus}_Q':gformer["V_phph"]*np.sqrt(2/3)*np.cos(np.deg2rad(gformer["deg"]))})
            
            
    if model_type == 'ae':
        g_grid += list(di_l_dq)
        g_grid += list(dv_dq)
        y_grid_list += list(i_l_dq)
        y_grid_list += list(v_dq)

        for gformer in grid_formers:
            N_i_branch = len(list(i_l_dq))
            idx_gformer = buses_list.index(gformer['bus'])
            y_grid_list[N_i_branch+2*idx_gformer] = i_list[2*idx_gformer]
            y_grid_list[N_i_branch+2*idx_gformer+1] = i_list[2*idx_gformer+1]

            bus_name = gformer['bus']
            phi = np.deg2rad(gformer['deg'])
            v_d = np.sin(phi)*gformer['V_phph']*np.sqrt(2/3)
            v_q = np.cos(phi)*gformer['V_phph']*np.sqrt(2/3)
            u_grid.update({f'v_{bus_name}_{D_}':v_d,f'v_{bus_name}_{Q_}':v_q})

    for load in loads:

        bus_name = load['bus']
        i_d_ref = sym.Symbol(f'i_{bus_name}_d_ref', real=True)
        i_q_ref = sym.Symbol(f'i_{bus_name}_q_ref', real=True)
        i_d = sym.Symbol(f'i_{bus_name}_{D_}', real=True)
        i_q = sym.Symbol(f'i_{bus_name}_{Q_}', real=True)
        T_i = sym.Symbol(f'T_i_{bus_name}', real=True)
        I_max = sym.Symbol(f'I_max_{bus_name}', real=True)

        p_ref = sym.Symbol(f'p_{bus_name}_ref', real=True)
        q_ref = sym.Symbol(f'q_{bus_name}_ref', real=True)
        v_d = sym.Symbol(f'v_{bus_name}_{D_}', real=True)
        v_q = sym.Symbol(f'v_{bus_name}_{Q_}', real=True)

        den = v_d**2 + v_q**2

        den_sat = sym.Piecewise((0.01,den<0.01),(1e12,den>1e12),(den,True))

        if park_type == 'original':
            g_d = -i_d_ref + 2/3*(-p_ref*v_d - q_ref*v_q)/den_sat 
            g_q = -i_q_ref - 2/3*( p_ref*v_q - q_ref*v_d)/den_sat
            y_d = i_d_ref
            y_q = i_q_ref    



        i_d_sat = sym.Piecewise((-I_max,i_d_ref<-I_max),(I_max,i_d_ref>I_max),(i_d_ref,True))
        i_q_sat = sym.Piecewise((-I_max,i_q_ref<-I_max),(I_max,i_q_ref>I_max),(i_q_ref,True))

        f_d = 1/0.01*(i_d_sat - i_d)
        f_q = 1/0.01*(i_q_sat - i_q)
        f_grid += [f_d,f_q]

        u_grid.update({f'T_i_{bus_name}':load['T_i']})
        u_grid.update({f'I_max_{bus_name}':load['I_max']})

        g_grid += [g_d,g_q]
        y_grid_list += [i_d_ref,i_q_ref]
        x_grid_list += [i_d,i_q]

        if "kVA" in load:
            phi = np.arccos(load["pf"])
            p = load['kVA']*1000*np.cos(phi)
            q = load['kVA']*1000*np.sin(phi)*np.sign(load["pf"])
        if "kW" in load:
            p = load['kW']*1000
            q = load['kvar']*1000
        u_grid.update({f'p_{bus_name}_ref':p,f'q_{bus_name}_ref':q})

    for bus in buses:
        if bus['bus'] not in load_buses_list+gformers_buses_list:
            bus_name = bus['bus']
            params.update({f'i_{bus_name}_{D_}':0.0,f'i_{bus_name}_{Q_}':0.0})

    f_hz = data['system']['f_hz']
    params.update({'omega':2*np.pi*f_hz})

    x_list = [str(item) for item in x_grid_list]
  
    return {'f':f_grid,'g':g_grid,
            'x':x_grid_list,'y':y_grid_list, 'x_list':x_list,
            'u':u_grid,'params':params,'v_list':v_list}

 

def dcgrid2dae(data_input):
    vscs = data_input['grid_formers']
    park_type='original'
    dq_name='DQ'
    
    xy_0_dict = {}

    if dq_name == 'DQ':
        D_ = 'D'
        Q_ = 'Q'        

    if type(data_input) == str:
        json_file = data_input
        json_data = open(json_file).read().replace("'",'"')
        data = json.loads(json_data)
    elif type(data_input) == dict:
        data = data_input

    data_processing(data)
    model_type = data['system']['model_type']
    buses = data['buses']
    lines = data['lines']
    loads = data['loads']
    grid_formers = data['grid_formers']

    buses_list = [item['bus'] for item in buses]
    load_buses_list = [item['bus'] for item in loads]
    gformers_buses_list = [item['bus'] for item in grid_formers]

    params = {}
    u_grid = {}
    f_grid = []
    g_grid = []
    y_grid_list = []
    x_grid_list = []
    x_list = []
    y_list = []
    h_dict = {}

    omega = sym.Symbol('omega', real=True)
    M = len(lines) # total number of branches
    N = len(buses) # total number of buses
    A_k = sym.Matrix.zeros(M,cols=N)

    i_line_list = []
    i_list = []
    v_list = []
    R_list = []
    L_list = []
    C_list = [0]*N

    itl = 0
    for line in lines:
        sub_name = f"{line['bus_j']}{line['bus_k']}"

        idx_bus_j = buses_list.index(line['bus_j'])
        idx_bus_k = buses_list.index(line['bus_k'])

        A_k[itl,idx_bus_j] =  1
        A_k[itl,idx_bus_k] = -1

        bus_j = line['bus_j']
        bus_k = line['bus_k']    

        R_ij = sym.Symbol(f'R_{sub_name}', real=True)
        L_ij = sym.Symbol(f'L_{sub_name}', real=True)

        R_list += [R_ij]
        L_list += [L_ij]

        i_line = sym.Symbol(f'i_l_{sub_name}', real=True)
        i_line_list += [i_line]

        C_ij = sym.Symbol(f'C_{sub_name}', real=True)
        C_list[idx_bus_j] += C_ij/2
        C_list[idx_bus_k] += C_ij/2

        # parameters
        R_name = f'R_{sub_name}'
        R_value = line['R']
        L_name = f'L_{sub_name}'
        L_value = line['L']
        C_name = f'C_{sub_name}'
        C_value = line['C']

        params.update({R_name:R_value,L_name:L_value,C_name:C_value})

        itl += 1


    C_e_list = []
    for item in C_list:
        C_e_list += [item]
        C_e_list += [item]

    for bus in buses:
        bus_name = bus['bus']
        v = sym.Symbol(f'v_{bus_name}', real=True)
        i = sym.Symbol(f'i_{bus_name}', real=True)

        v_list += [v]
        i_list += [i]

        xy_0_dict.update({f'v_{bus_name}':3000})

    i_line = sym.Matrix(i_line_list)
    R_e = sym.Matrix.diag(R_list)
    L_e = sym.Matrix.diag(L_list)

    v  = sym.Matrix(v_list)
    i  = sym.Matrix(i_list)
    def T(P):
        u = TensorProduct(sym.Matrix.eye(P),sym.Matrix([1,0]).T)
        l = TensorProduct(sym.Matrix.eye(P),sym.Matrix([0,1]).T)
        return sym.Matrix([u,l])

    A_e = sym.Matrix.diag([A_k])

    if park_type == 'fisix':
        di_l_dq =  (-R_e  @ i_line + A_e @ v)
        dv =  ( - A_e.T @ i_line + i)
    if park_type == 'original':
        di_line =  (-(R_e) @ i_line + A_e @ v)
        dv =  (- A_e.T @ i_line + i)

    if model_type == 'ae':
        g_grid += list(di_line)
        g_grid += list(dv)
        y_grid_list += list(i_line_list)
        y_grid_list += list(v)

        for gformer in grid_formers:
            N_i_branch = len(list(i_line_list))
            idx_gformer = buses_list.index(gformer['bus'])
            y_grid_list[N_i_branch+idx_gformer] = i_list[idx_gformer]

            bus_name = gformer['bus']
            phi = np.deg2rad(gformer['deg'])
            v_d = np.sin(phi)*gformer['V_phph']*np.sqrt(2/3)
            v_q = np.cos(phi)*gformer['V_phph']*np.sqrt(2/3)
            u_grid.update({f'v_{bus_name}_{D_}':v_d,f'v_{bus_name}_{Q_}':v_q})
        
    return {'f':f_grid,'g':g_grid,
            'x':x_grid_list,'y':y_grid_list, 'x_list':x_list,
            'u':u_grid,'params':params,'v_list':v_list,'xy_0_dict':xy_0_dict}

def vsg2dae(data,grid_dae):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    grid_dae : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    '''
    
    sin = sym.sin
    cos = sym.cos
    sqrt = sym.sqrt 
    
    
    vsgs = data['vsgs']
    N_vsg = len(vsgs)
    
    # secondary control
    p_sec = {}
    q_sec = {}
    xi_f_sec,xi_v_sec = sym.symbols('xi_f_sec,xi_v_sec', real=True)
    K_f_sec,K_v_sec = sym.symbols('K_f_sec,K_v_sec', real=True)
    
    omega_coi_h = 0
    H_total = 0
    N_voltage = 0
    v_prom = 0
    for vsg in vsgs:
        name = vsg['name']
        
        omega_v_i  = sym.Symbol(f'omega_v_{name}', real=True)
        H = sym.Symbol(f'H_{name}', real=True)
        
        omega_coi_h += omega_v_i*H
        H_total += H 
        
        v_s_filt_i = sym.symbols(f'v_s_filt_{name}', real=True)
        N_voltage += 1
        v_prom += v_s_filt_i/N_vsg
    
    omega_coi = omega_coi_h/H_total
    dxi_f_sec = 1 - omega_coi
    dxi_v_sec = 1 - v_prom
    
    for vsg in vsgs:
        name = vsg['name']
        p_sec.update({f'{name}':K_f_sec*xi_f_sec/N_vsg})
        q_sec.update({f'{name}':K_v_sec*xi_v_sec/N_vsg})

    f_ctrl_5 = [dxi_f_sec,dxi_v_sec]
    x_ctrl_5 = [ xi_f_sec, xi_v_sec]
    g_ctrl_5 = []
    y_ctrl_5 = []
    x_0_ctrl_5 = []
    y_0_ctrl_5 = []
    params_ctrl_5 = {'K_f_sec':0.001,'K_v_sec':0.01}
    u_ctrl_5 = {}
    h_ctrl_5 = {}

    
    f_vsg = []
    x_vsg = []
    g_vsg = []
    y_vsg = []
    x_0_vsg = []
    y_0_vsg = []
    params_vsg = {}
    u_vsg = {}
    h_vsg = {}

    for vsg in vsgs:
        name = vsg['name']
        bus = vsg['bus']

        U_b = vsg['U_b']
        S_b = vsg['S_b_kVA']*1000
        I_b = S_b/(np.sqrt(3)*U_b)

        U_bdq = U_b*(np.sqrt(2))
        V_bdq = U_bdq/np.sqrt(3)
        I_bdq = I_b*np.sqrt(2)




        ## Transformations #########################################################################
        ## feedbacks:
        feedbacks = ['i_tD','i_tQ'] + ['v_mD','v_mQ'] + ['i_sD','i_sQ'] + ['v_sD','v_sQ'] + ['phi']
        for item in feedbacks:
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())    



        v_md,v_mq = sym.symbols(f'v_md_{name},v_mq_{name}', real=True)
        v_sd,v_sq = sym.symbols(f'v_sd_{name},v_sq_{name}', real=True)
        i_sd,i_sq = sym.symbols(f'i_sd_{name},i_sq_{name}', real=True)
        i_td,i_tq = sym.symbols(f'i_td_{name},i_tq_{name}', real=True)
        phi,dum= sym.symbols(f'phi_{name},dum_{name}', real=True)

        eq_v_md = -v_md + v_mD*cos(phi) + v_mQ*sin(phi) # original park
        eq_v_mq = -v_mq - v_mD*sin(phi) + v_mQ*cos(phi) # original park

        eq_v_sd = -v_sd + v_sD*cos(phi) + v_sQ*sin(phi) # original park
        eq_v_sq = -v_sq - v_sD*sin(phi) + v_sQ*cos(phi) # original park

        eq_i_sd = -i_sd + i_sD*cos(phi) + i_sQ*sin(phi) # original park
        eq_i_sq = -i_sq - i_sD*sin(phi) + i_sQ*cos(phi) # original park

        # jmm: medimos i_t?
        eq_i_td = -i_td + i_tD*cos(phi) + i_tQ*sin(phi) # original park
        eq_i_tq = -i_tq - i_tD*sin(phi) + i_tQ*cos(phi) # original park

        g_aux = [eq_v_md,eq_v_mq,eq_v_sd,eq_v_sq,eq_i_td,eq_i_tq,eq_i_sd,eq_i_sq]    
        y_aux = [   v_md,   v_mq,   v_sd,   v_sq,   i_td,   i_tq,   i_sd,   i_sq]
        y_0_aux = [  0.0,  V_bdq,    0.0,  V_bdq,       0,      0]    


        #v_sd = v_md
        #v_sq = v_mq


       # S_b_kVA,U_b = sym.symbols(f'S_b_kVA_{name},U_b_{name}', real=True) # params



        ## per unit

        i_sd_pu = i_sd/I_bdq; # input in SI that is coverted to pu
        i_sq_pu = i_sq/I_bdq; # input in SI that is coverted to pu
        v_sd_pu = v_sd/V_bdq; # input in SI that is coverted to pu
        v_sq_pu = v_sq/V_bdq; # input in SI that is coverted to pu
        i_td_pu = i_td/I_bdq; # input in SI that is coverted to pu
        i_tq_pu = i_tq/I_bdq; # input in SI that is coverted to pu



        ## PLL #########################################################################


        # CTRL4 #########################################################################
        ## parameters:
        params_ctrl_4 = {}
        for item in ['T_vpoi','K_vpoi','T_f','K_f']:
            params_ctrl_4.update({f'{item}_{name}':vsg[item]})
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
        ## inputs:
        u_ctrl_4 = {}
        for item in ['v_s_ref','omega_ref','p_r','q_r']:
            u_ctrl_4.update({f'{item}_{name}':vsg[item]})
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
        ## dynamic states:
        x_list_ctrl_4 = ['omega_v_filt','v_s_filt']
        for item in x_list_ctrl_4:
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
        ## algebraic states
        y_list_ctrl_4 = ['p_m_ref','q_s_ref']
        for item in y_list_ctrl_4:
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())    
        ## feedbacks:
        feedbacks = ['omega_v']
        for item in feedbacks:
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())       

        domega_v_filt = 1/T_f*(omega_v - omega_v_filt)
        dv_s_filt = 1/T_vpoi*((v_sd_pu**2+v_sq_pu**2)**0.5 - v_s_filt)

        eq_p_m_ref = -p_m_ref + p_r + K_f*(omega_ref - omega_v_filt) + p_sec[name]   # PFR and secondary input 
        eq_q_s_ref = -q_s_ref + q_r + K_vpoi*(v_s_ref - v_s_filt)    + q_sec[name]


        # from derivatives to the integrator
        f_ctrl_4 = [domega_v_filt,dv_s_filt];
        x_ctrl_4 = [ omega_v_filt, v_s_filt];
        g_ctrl_4 = [ eq_p_m_ref, eq_q_s_ref] #eq_i_sd_ref, eq_i_sq_ref, ]
        y_ctrl_4 = [    p_m_ref,    q_s_ref] #   i_sd_ref,    i_sq_ref,    omega_v,    DV_sat];



        # CTRL3 #########################################################################


        if vsg['ctrl3'] == 'uvsg_i':

            ## parameters:
            params_ctrl_3 = {}
            for item in ['K_p','T_p','K_q','T_q','R_v','X_v','S_b_kVA','U_b','K_phi','H','D']:
                params_ctrl_3.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## inputs:
            u_ctrl_3 = {}
            for item in ['p_m_ref','q_s_ref']: # []
                u_ctrl_3.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## dynamic states:
            x_list_ctrl_3 = ['phi','omega_v','xi_q','omega_rads']
            for item in x_list_ctrl_3:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## algebraic states
            y_list_ctrl_3 = ['DV_sat','p_s_pu','q_s_pu']
            for item in y_list_ctrl_3:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())    
            ## feedbacks:   
            v_dc,dum = sym.symbols(f'v_dc_{name},dum_{name}', real=True)

            # equations:
            eq_omega_v = -omega_v + K_p*(epsilon_p + xi_p/T_p) + 1.0; 

            dphi =  Omega_b*(omega_v-1.0) - K_phi*phi;
            dxi_p = epsilon_p;
            dxi_q = epsilon_q; 
            domega_rads = 1.0/1.0*(Omega_b*omega_v - omega_rads);


        if vsg['ctrl3'] == 'sm2':

            ## parameters:
            params_ctrl_3 = {}
            for item in ['K_p','T_p','K_q','T_q','R_v','X_v','S_b_kVA','U_b','K_phi','H','D']:
                params_ctrl_3.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## inputs:
            u_ctrl_3 = {}
            for item in ['p_m_ref','q_s_ref']: # []
                u_ctrl_3.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## dynamic states:
            x_list_ctrl_3 = ['phi','omega_v','xi_q','omega_rads']
            for item in x_list_ctrl_3:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## algebraic states
            y_list_ctrl_3 = ['DV_sat','p_s_pu','q_s_pu']
            for item in y_list_ctrl_3:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())    
            ## feedbacks:   
            v_dc,dum = sym.symbols(f'v_dc_{name},dum_{name}', real=True)


            # equations:

            fault_flag = 0;
            e_0 = 1.0;    
            eq_p_s_pu = -p_s_pu + i_sd_pu*v_sd_pu + i_sq_pu*v_sq_pu; # pu
            eq_q_s_pu = -q_s_pu + i_sd_pu*v_sq_pu - i_sq_pu*v_sd_pu; # pu 

            # from the integrator to the states
            epsilon_p = (p_m_ref - p_s_pu)*(1.0-fault_flag);
            epsilon_q =  q_s_ref - q_s_pu;


            dphi =  Omega_b*(omega_v-omega_coi) - K_phi*phi;
            domega_v = 1/(2*H)*(p_m_ref - p_s_pu - D*(omega_v - 1.0))
            dxi_q = epsilon_q; 
            domega_rads = 1.0/1.0*(Omega_b*omega_v - omega_rads);

            DV =   K_q*(epsilon_q + xi_q/T_q); 

            eq_DV_sat = DV_sat - DV;
            #if DV_sat > 0.1 
            #    DV_sat = 0.1;
            #    dxi_q = 0.0;
            #end
            #if DV_sat < -0.1 
            #    DV_sat = -0.1;
            #    dxi_q = 0.0;
            #end   

            e = e_0 + DV_sat;   


            if (not 'ctrl1' in vsg) and (not 'ctrl2' in vsg):                      # CTRL3 over CTRL0
                v_t_d_pu = 0.0
                v_t_q_pu = e

                v_t_d = (v_t_d_pu - R_v*i_sd_pu + X_v*i_sq_pu)*V_bdq
                v_t_q = (v_t_q_pu - R_v*i_sq_pu - X_v*i_sd_pu)*V_bdq

                eta_d_ref = v_t_d/v_dc*2
                eta_q_ref = v_t_q/v_dc*2

                # from derivatives to the integrator
                f_ctrl_3 = [dphi,domega_v,dxi_q,domega_rads];
                x_ctrl_3 = [ phi, omega_v, xi_q, omega_rads];
                g_ctrl_3 = [ eq_DV_sat, eq_p_s_pu, eq_q_s_pu] #eq_i_sd_ref, eq_i_sq_ref, ]
                y_ctrl_3 = [    DV_sat,    p_s_pu,    q_s_pu] #   i_sd_ref,    i_sq_ref,    omega_v,    DV_sat];
                x_0_ctrl_3 = [ 0.0, 0.0, 0.0, 2*np.pi*50]
                y_0_ctrl_3 = [ ] #     0.0,       V_bdq,          1]  


            if ('ctrl1' in vsg) and (not 'ctrl2' in vsg):                          # CTRL3 over CTRL1
                i_sd_pu = -(X_v*e + R_v*v_sd_pu + X_v*v_sq_pu)/(R_v**2 + X_v**2)
                i_sq_pu = -(R_v*e + R_v*v_sq_pu - X_v*v_sd_pu)/(R_v**2 + X_v**2);
                eq_i_sd_ref = -i_sd_ref + i_sd_pu*I_bdq
                eq_i_sq_ref = -i_sq_ref + i_sq_pu*I_bdq

                # from derivatives to the integrator
                f_ctrl_3 = [dphi,domega_v,dxi_q,domega_rads];
                x_ctrl_3 = [ phi, omega_v, xi_q, omega_rads];
                g_ctrl_3 = [ eq_DV_sat, eq_p_s_pu, eq_q_s_pu, eq_i_sd_ref, eq_i_sq_ref]
                y_ctrl_3 = [    DV_sat,    p_s_pu,    q_s_pu,    i_sd_ref,    i_sq_ref];
                x_0_ctrl_3 = [ 0.0, 0.0, 0.0, 2*np.pi*50]
                y_0_ctrl_3 = [ ] #     0.0,       V_bdq,          1]     

            if ('ctrl1' in vsg) and ('ctrl2' in vsg):                          # CTRL3 over CTRL2
                v_sd_ref_pu =    - R_v*i_sd_pu + X_v*i_sq_pu; 
                v_sq_ref_pu = -e - R_v*i_sq_pu - X_v*i_sd_pu;
                eq_v_sd_ref = -v_sd_ref + v_sd_ref_pu*V_bdq
                eq_v_sq_ref = -v_sq_ref + v_sq_ref_pu*V_bdq

                # from derivatives to the integrator
                f_ctrl_3 = [dphi,domega_v,dxi_q,domega_rads];
                x_ctrl_3 = [ phi, omega_v, xi_q, omega_rads];
                g_ctrl_3 = [ eq_DV_sat, eq_p_s_pu, eq_q_s_pu, eq_v_sd_ref, eq_v_sq_ref]
                y_ctrl_3 = [    DV_sat,    p_s_pu,    q_s_pu,    v_sd_ref,    v_sq_ref];
                x_0_ctrl_3 = [ 0.0, 0.0, 0.0, 2*np.pi*50]
                y_0_ctrl_3 = [ ] #     0.0,       V_bdq,          1]              


        if vsg['ctrl3'] == 'droop_pq':

            ## parameters:
            params_ctrl_3 = {}
            for item in ['Omega_b','K_p','T_p','K_q','T_q','R_v','X_v','S_b_kVA','U_b','K_phi','H','D','K_omega','K_v','T_omega']:
                params_ctrl_3.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## inputs:
            u_ctrl_3 = {}
            for item in ['p_m_ref','q_s_ref','e_ref']: # []
                u_ctrl_3.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## dynamic states:
            x_list_ctrl_3 = ['phi','omega_v','xi_q','omega_rads']
            for item in x_list_ctrl_3:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## algebraic states
            y_list_ctrl_3 = ['p_s_pu','q_s_pu']
            for item in y_list_ctrl_3:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())    
            ## feedbacks:   
            v_dc,dum = sym.symbols(f'v_dc_{name},dum_{name}', real=True)


            # equations:

            fault_flag = 0;
            e_0 = 1.0;    
            eq_p_s_pu = -p_s_pu + i_sd_pu*v_sd_pu + i_sq_pu*v_sq_pu; # pu
            eq_q_s_pu = -q_s_pu + i_sd_pu*v_sq_pu - i_sq_pu*v_sd_pu; # pu 

            # from the integrator to the states
            epsilon_p = (p_m_ref - p_s_pu)*(1.0-fault_flag);
            epsilon_q =  q_s_ref - q_s_pu;

            omega_v_ref = omega_ref + K_omega*epsilon_p 
            e =   e_ref + K_v*epsilon_q

            dphi =  Omega_b*(omega_v-omega_coi) - K_phi*phi;
            domega_v = 1/T_omega*(omega_v_ref - omega_v)
            domega_rads = 1.0/1.0*(Omega_b*omega_v - omega_rads);


            if (not 'ctrl1' in vsg) and (not 'ctrl2' in vsg):                      # CTRL3 over CTRL0
                v_t_d_pu = 0.0
                v_t_q_pu = e

                v_t_d = (v_t_d_pu - R_v*i_sd_pu + X_v*i_sq_pu)*V_bdq
                v_t_q = (v_t_q_pu - R_v*i_sq_pu - X_v*i_sd_pu)*V_bdq

                eta_d_ref = v_t_d/v_dc*2
                eta_q_ref = v_t_q/v_dc*2

                # from derivatives to the integrator
                f_ctrl_3 = [dphi,domega_v,domega_rads];
                x_ctrl_3 = [ phi, omega_v, omega_rads];
                g_ctrl_3 = [ eq_p_s_pu, eq_q_s_pu] #eq_i_sd_ref, eq_i_sq_ref, ]
                y_ctrl_3 = [    p_s_pu,    q_s_pu] #   i_sd_ref,    i_sq_ref,    omega_v,    DV_sat];
                x_0_ctrl_3 = [ 0.0, 0.0, 0.0, 2*np.pi*50]
                y_0_ctrl_3 = [ ] #     0.0,       V_bdq,          1]  


            if ('ctrl1' in vsg) and (not 'ctrl2' in vsg):                          # CTRL3 over CTRL1
                i_sd_pu = -(X_v*e + R_v*v_sd_pu + X_v*v_sq_pu)/(R_v**2 + X_v**2)
                i_sq_pu = -(R_v*e + R_v*v_sq_pu - X_v*v_sd_pu)/(R_v**2 + X_v**2);
                eq_i_sd_ref = -i_sd_ref + i_sd_pu*I_bdq
                eq_i_sq_ref = -i_sq_ref + i_sq_pu*I_bdq

                # from derivatives to the integrator
                f_ctrl_3 = [dphi,domega_rads];
                x_ctrl_3 = [ phi, omega_rads];
                g_ctrl_3 = [ eq_DV_sat, eq_p_s_pu, eq_q_s_pu, eq_i_sd_ref, eq_i_sq_ref]
                y_ctrl_3 = [    DV_sat,    p_s_pu,    q_s_pu,    i_sd_ref,    i_sq_ref];
                x_0_ctrl_3 = [ 0.0, 0.0, 0.0, 2*np.pi*50]
                y_0_ctrl_3 = [ ] #     0.0,       V_bdq,          1]     

            if ('ctrl1' in vsg) and ('ctrl2' in vsg):                          # CTRL3 over CTRL2
                v_sd_ref_pu =    - R_v*i_sd_pu + X_v*i_sq_pu; 
                v_sq_ref_pu = -e - R_v*i_sq_pu - X_v*i_sd_pu;
                eq_v_sd_ref = -v_sd_ref + v_sd_ref_pu*V_bdq
                eq_v_sq_ref = -v_sq_ref + v_sq_ref_pu*V_bdq

                # from derivatives to the integrator
                f_ctrl_3 = [dphi,domega_v,dxi_q,domega_rads];
                x_ctrl_3 = [ phi, omega_v, xi_q, omega_rads];
                g_ctrl_3 = [ eq_DV_sat, eq_p_s_pu, eq_q_s_pu, eq_v_sd_ref, eq_v_sq_ref]
                y_ctrl_3 = [    DV_sat,    p_s_pu,    q_s_pu,    v_sd_ref,    v_sq_ref];
                x_0_ctrl_3 = [ 0.0, 0.0, 0.0, 2*np.pi*50]
                y_0_ctrl_3 = [ ] #     0.0,       V_bdq,          1]        

        if vsg['ctrl3'] == 'genape':
            dphi =  Omega_b*(omega_v-omega_coi) - K_phi*phi;
            domega_v = RoCoF/F_b;






        # CTRL0 #########################################################################

        ## inputs:
        params_ctrl_0 = {}
        u_ctrl_0 = {}
        for item in []: # ['eta_d_ref','eta_q_ref','phi']: #+['eta_D','eta_Q']:
            u_ctrl_0.update({f'{item}_{name}':vsg[item]})
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
        ## algebraic states
        y_list_ctrl_0 = ['eta_d','eta_q'] + ['eta_D','eta_Q'] 
        for item in y_list_ctrl_0:
            exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())        

        eq_eta_d = eta_d - eta_d_ref
        eq_eta_q = eta_q - eta_q_ref

        eq_eta_D = -eta_D + eta_d*cos(phi) - eta_q*sin(phi) # comment for test 1 
        eq_eta_Q = -eta_Q + eta_d*sin(phi) + eta_q*cos(phi) # comment for test 1

        # from derivatives to the integrator
        f_ctrl_0 = [];
        x_ctrl_0 = [];
        g_ctrl_0 = [eq_eta_d,eq_eta_q,eq_eta_D,eq_eta_Q]
        y_ctrl_0 = [   eta_d,   eta_q,   eta_D,   eta_Q];
        x_0_ctrl_0 = [ ]
        y_0_ctrl_0 = [0.0,  0.8, 0.0,  0.8]    



        ## VSC and Filters #########################################################################

        if vsg['filter'] == 'L':
            ## parameters:
            params_vsc_filter = {}
            for item in ['L_t','R_t','omega']:
                params_vsc_filter.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## inputs:
            u_vsc_filter= {}
            for item in ['v_dc']: #+['eta_D','eta_Q']:
                u_vsc_filter.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## dynamic states:
            x_list_vsc_filter = []
            for item in x_list_vsc_filter:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## algebraic states
            y_list_vsc_filter = ['i_tD','i_tQ'] + ['v_mD','v_mQ'] + ['i_sD','i_sQ'] + ['v_sD','v_sQ']
            for item in y_list_vsc_filter:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)")    
            ## feedbacks:
            v_poiD,v_poiQ = sym.symbols(f'v_{bus}_D,v_{bus}_Q', real=True)
            i_poiD,i_poiQ = sym.symbols(f'i_{bus}_D,i_{bus}_Q', real=True)

            #eta_D = eta_D_ref # - Gv_in*(i_tD - i_sD)
            #eta_Q = eta_Q_ref #- Gv_in*(i_tQ - i_sQ)

            # LCL filter
            di_tD = 1/L_t*(eta_D/2*v_dc - R_t*i_tD + omega*L_t*i_tQ - v_mD)  
            di_tQ = 1/L_t*(eta_Q/2*v_dc - R_t*i_tQ - omega*L_t*i_tD - v_mQ) 
            dv_mD = 1/C_m*(i_tD + C_m*omega*v_mQ - G_d*v_mD - i_sD) 
            dv_mQ = 1/C_m*(i_tQ - C_m*omega*v_mD - G_d*v_mQ - i_sQ) 
            di_sD = 1/L_s*(v_mD - R_s*i_sD + omega*L_s*i_sQ - v_sD)  
            di_sQ = 1/L_s*(v_mQ - R_s*i_sQ - omega*L_s*i_sD - v_sQ) 

            # Grid interaction
            eq_i_poiD =  i_sD - i_poiD
            eq_i_poiQ =  i_sQ - i_poiQ   

            eq_v_sD =  v_sD - v_poiD
            eq_v_sQ =  v_sQ - v_poiQ    

            grid_dae['params'].pop(f'i_{bus}_D')
            grid_dae['params'].pop(f'i_{bus}_Q')

            # DAE
            f_vsc_filter = []
            x_vsc_filter = []
            g_vsc_filter = [di_tD, di_tQ, eq_i_poiD, eq_i_poiQ, eq_v_sD, eq_v_sQ]
            y_vsc_filter = [ i_tD,  i_tQ,    i_poiD,    i_poiQ,    v_sD,    v_sQ]    
            x_0_vsc_filter = [ ]
            y_0_vsc_filter = [ 0.0,  0.0,       0.0,         0,       0,   V_bdq]

        if vsg['filter'] == 'LCL':
            ## parameters:
            params_vsc_filter = {}
            for item in ['L_t','R_t','C_m','L_s','R_s','omega','G_d']:
                params_vsc_filter.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## inputs:
            u_vsc_filter = {}
            for item in ['v_dc']: #+['eta_D','eta_Q']:
                u_vsc_filter.update({f'{item}_{name}':vsg[item]})
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())
            ## dynamic states:
            x_list_vsc_filter = []
            for item in x_list_vsc_filter:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)")
            ## algebraic states
            y_list_vsc_filter = ['i_tD','i_tQ'] + ['v_mD','v_mQ'] + ['i_sD','i_sQ'] + ['v_sD','v_sQ']
            for item in y_list_vsc_filter:
                exec(f"{item} = sym.Symbol('{item}_{name}', real=True)",globals())    
            ## feedbacks:
            v_poiD,v_poiQ = sym.symbols(f'v_{bus}_D,v_{bus}_Q', real=True)
            i_poiD,i_poiQ = sym.symbols(f'i_{bus}_D,i_{bus}_Q', real=True)

            #eta_D = eta_D_ref # - Gv_in*(i_tD - i_sD)
            #eta_Q = eta_Q_ref #- Gv_in*(i_tQ - i_sQ)

            # LCL filter
            di_tD = 1/L_t*(eta_D/2*v_dc - R_t*i_tD + omega*L_t*i_tQ - v_mD)  
            di_tQ = 1/L_t*(eta_Q/2*v_dc - R_t*i_tQ - omega*L_t*i_tD - v_mQ) 
            dv_mD = 1/C_m*(i_tD + C_m*omega*v_mQ - G_d*v_mD - i_sD) 
            dv_mQ = 1/C_m*(i_tQ - C_m*omega*v_mD - G_d*v_mQ - i_sQ) 
            di_sD = 1/L_s*(v_mD - R_s*i_sD + omega*L_s*i_sQ - v_sD)  
            di_sQ = 1/L_s*(v_mQ - R_s*i_sQ - omega*L_s*i_sD - v_sQ) 

            # Grid interaction
            eq_i_poiD =  i_sD - i_poiD
            eq_i_poiQ =  i_sQ - i_poiQ   

            eq_v_sD =  v_sD - v_poiD
            eq_v_sQ =  v_sQ - v_poiQ    

            grid_dae['params'].pop(f'i_{bus}_D')
            grid_dae['params'].pop(f'i_{bus}_Q')

            # DAE
            f_vsc_filter = []
            x_vsc_filter = []
            g_vsc_filter = [di_tD, di_tQ, dv_mD, dv_mQ, di_sD, di_sQ, eq_i_poiD, eq_i_poiQ, eq_v_sD, eq_v_sQ]
            y_vsc_filter = [ i_tD,  i_tQ,  v_mD,  v_mQ,  i_sD,  i_sQ,    i_poiD,    i_poiQ,    v_sD,    v_sQ]    
            x_0_vsc_filter = [ ]
            y_0_vsc_filter = [ 0.0,  0.0,   0.0, V_bdq,     0,     0,         0,         0,       0,   V_bdq]


        ## Model integration
        f_vsg += f_vsc_filter + f_ctrl_0 + f_ctrl_3 + f_ctrl_4
        x_vsg += x_vsc_filter + x_ctrl_0 + x_ctrl_3 + x_ctrl_4
        g_vsg += g_vsc_filter + g_ctrl_0 + g_aux + g_ctrl_3 + g_ctrl_4  
        y_vsg += y_vsc_filter + y_ctrl_0 + y_aux + y_ctrl_3 + y_ctrl_4
        params_vsg.update(params_vsc_filter)
        params_vsg.update(params_ctrl_0)
        params_vsg.update(params_ctrl_3)
        params_vsg.update(params_ctrl_4)
        u_vsg.update(u_vsc_filter)
        u_vsg.update(u_ctrl_0)
        u_vsg.update(u_ctrl_3)
        u_vsg.update(u_ctrl_4)    

        h_vsg.update({f'i_sD_{name}':i_sD,f'i_sQ_{name}':i_sQ})

       # x_0_vsg += x_0_vsc_lc 
       # y_0_vsg += y_0_vsc_lc 

    f_vsg += f_ctrl_5
    x_vsg += x_ctrl_5
    params_vsg.update(params_ctrl_5)
    u_vsg.update(u_ctrl_5)
    
    return {'f_list':f_vsg,'g_list':g_vsg,
            'x_list':x_vsg,'y_list':y_vsg,
            'u_run_dict':u_vsg,'params_dict':params_vsg,'h_dict':h_vsg,
            'omega_coi':omega_coi}

def dcrail2dae(data_input,dcgrid_dae):
    h_dict = {}
    sections = data_input['sections']
    g_grid = dcgrid_dae['g']
    y_grid_list = dcgrid_dae['y']
    f_grid = dcgrid_dae['f'] 
    x_grid_list  = dcgrid_dae['x'] 

    for section in sections:

        nodes_i = section['nodes_i']

        for node in nodes_i:
            v = sym.Symbol(f'v_{node}', real=True)
            i = sym.Symbol(f'i_{node}', real=True)
            p = sym.Symbol(f'p_{node}', real=True)
            g_grid += [-p + v*i]
            y_grid_list += [i]

    for section in sections:

        nodes_v = section['nodes_v']

        for node in nodes_v:
            v = sym.Symbol(f'v_{node}', real=True)
            i = sym.Symbol(f'i_{node}', real=True)
            p = sym.Symbol(f'p_{node}', real=True)
            h_dict.update({f'p_{node}':v*i})
            h_dict.update({f'v_{node}':v})

    for section in sections[1:]:

        nodes_v = section['nodes_v']

        node = nodes_v[0]
        v_nom = sym.Symbol(f'v_nom', real=True)
        v = sym.Symbol(f'v_{node}', real=True)
        i = sym.Symbol(f'i_{node}', real=True)
        v_ref = sym.Symbol(f'v_ref_{node}', real=True)
        T_v = sym.Symbol(f'T_v', real=True)
        K_r = sym.Symbol(f'K_r', real=True)
        Dv_r = sym.Symbol(f'Dv_r_{node}', real=True)
        p = v*i
        v_ref = v_nom - K_r*p - Dv_r    # v_nom = nominal voltage, K_r*p: power droop, Dv_r remote input
        f_grid += [1/T_v*(v_ref-v)]  # gracias por el cambio
        x_grid_list += [v]
        
    dcgrid_dae.update({'h_dict':h_dict})

def pf_network(file_path):
    '''
    

    Parameters
    ----------
    file_path : string
        File path to the system data information.

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
    
    with open(file_path,'r') as fobj:
        data = json.loads(fobj.read().replace("'",'"'))
    sys = data['sys']
    buses = data['buses']
    lines = data['lines']

    params_grid = {'S_base':sys['S_base']}
    S_base = sym.Symbol("S_base", real=True) 
    N_bus = len(buses)
    N_branch = len(lines)
    A = sym.zeros(N_branch,N_bus)
    G_primitive = sym.zeros(N_branch,N_branch)
    B_primitive = sym.zeros(N_branch,N_branch)
    buses_list = [bus['name'] for bus in buses]
    it = 0
    for line in lines:

        bus_j = line['bus_j']
        bus_k = line['bus_k']

        idx_j = buses_list.index(bus_j)
        idx_k = buses_list.index(bus_k)    

        A[it,idx_j] = 1
        A[it,idx_k] =-1   

        line_name = f"{bus_j}_{bus_k}"
        g_jk = sym.Symbol(f"g_{line_name}", real=True) 
        b_jk = sym.Symbol(f"b_{line_name}", real=True) 
        G_primitive[it,it] = g_jk
        B_primitive[it,it] = b_jk

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
            
    return {'g':g_grid,'y':y_grid,'u':u_grid,'h':h_grid, 'params':params_grid, 'data':data}




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

        P = sym.Symbol(f"P_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True)
        Q = sym.Symbol(f"Q_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True)
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
        v_c = sym.Symbol(f"v_c_{bus_name}", real=True)
        p_m_ref  = sym.Symbol(f"p_m_ref_{bus_name}", real=True)
        v_ref  = sym.Symbol(f"v_ref_{bus_name}", real=True)
        xi_m = sym.Symbol(f"xi_m_{bus_name}", real=True)
        
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 

        for item in syn:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())

        p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q) 


        ddelta = Omega_b*(omega - omega_s) - K_delta*delta
        domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
        de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
        de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)
        dv_c =   (V - v_c)/T_r
        dp_m =   (p_m_ref - p_m)/T_m
        dxi_m =   omega - 1

        g_id  = v_q + R_a*i_q + X1d*i_d - e1q
        g_iq  = v_d + R_a*i_d - X1q*i_q - e1d
        g_p  = i_d*v_d + i_q*v_q - P/S_n  
        g_q  = i_d*v_q - i_q*v_d - Q/S_n  
        g_vf  = K_a*(v_ref - v_c + v_pss) - v_f 
        g_pm  = -p_m_ref - K_sec*xi_m - 1/Droop*(omega - 1)

        f_syn = [ddelta,domega,de1q,de1d,dv_c,dp_m,dxi_m]
        x_syn = [ delta, omega, e1q, e1d, v_c, p_m, xi_m]
        g_syn = [g_id,g_iq,g_p,g_q,g_vf,g_pm]
        y_syn = [i_d,i_q,P,Q,v_f,p_m_ref]
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['g'][idx_bus*2]   += -P/S_base
        grid['g'][idx_bus*2+1] += -Q/S_base
        
        for item in syn:       
            grid['params'].update({f"{item}_{bus_name}":syn[item]})
        grid['params'].pop(f"bus_{bus_name}")
        grid['params'].update({f"v_ref_{bus_name}":1.0})




def psys_builder(file_path):
    
    grid = pf_network(file_path)
    syns_add(grid)
    
    return grid


if __name__ == "__main__":
    file_path = './data/sys2bus.json'    
    grid = pf_network_shunt(file_path)
