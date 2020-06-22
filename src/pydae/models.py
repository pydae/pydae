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
import os




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

        g_d = -i_d_ref + 2/3*(-p_ref*v_d + q_ref*v_q)/den_sat 
        g_q = -i_q_ref - 2/3*( p_ref*v_q + q_ref*v_d)/den_sat
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



def grid2dae(nodes_list,V_node,I_node,Y_iv,inv_Y_ii,load_buses):
    
    N_v = Y_iv.shape[1]   # number of nodes with known voltages
    I_node_sym_list = []
    V_node_sym_list = []
    v_list = []
    i_list = []
    v_list_str = []
    i_list_str = []
    i_node = []
    v_num_list = []
    i_num_list = []
    n2a = {'1':'a','2':'b','3':'c','4':'n'}
    a2n = {'a':'1','b':'2','c':'3','n':'4'}

    # every voltage bus and current bus injection is generted as sympy symbol
    # the voltages ar named as v_{bus_name}_{n2a[phase]}_r
    # the currents ar named as i_{bus_name}_{n2a[phase]}_r
    inode = 0
    for node in nodes_list:
        bus_name,phase = node.split('.')
        i_real = sym.Symbol(f"i_{bus_name}_{n2a[phase]}_r", real=True)
        i_imag = sym.Symbol(f"i_{bus_name}_{n2a[phase]}_i", real=True)
        v_real = sym.Symbol(f"v_{bus_name}_{n2a[phase]}_r", real=True)
        v_imag = sym.Symbol(f"v_{bus_name}_{n2a[phase]}_i", real=True)    

        v_list += [v_real,v_imag]  
        i_list += [i_real,i_imag]

        v_list_str += [str(v_real),str(v_imag)]
        i_list_str += [str(i_real),str(i_imag)]

        v_num_list += [V_node[inode].real[0],V_node[inode].imag[0]]
        i_num_list += [I_node[inode].real[0],I_node[inode].imag[0]]

        V_node_sym_list += [v_real+sym.I*v_imag]
        I_node_sym_list += [i_real+sym.I*i_imag]

        inode += 1
    
    # symbolic voltage and currents vectors (complex)
    V_known_sym = sym.Matrix(V_node_sym_list[:N_v])
    V_unknown_sym = sym.Matrix(V_node_sym_list[N_v:])
    I_known_sym = sym.Matrix(I_node_sym_list[N_v:])
    I_unknown_sym = sym.Matrix(I_node_sym_list[:N_v])
    
    inv_Y_ii_re = inv_Y_ii.real
    inv_Y_ii_im = inv_Y_ii.imag

    inv_Y_ii_re[np.abs(inv_Y_ii_re)<1e-8] = 0
    inv_Y_ii_im[np.abs(inv_Y_ii_im)<1e-8] = 0

    inv_Y_ii = inv_Y_ii_re+sym.I*inv_Y_ii_im

    I_aux = ( I_known_sym - Y_iv @ V_known_sym)  
    g_cplx = -V_unknown_sym + inv_Y_ii @ I_aux

    g_list = []
    for item in g_cplx:
        g_list += [sym.re(item)]
        g_list += [sym.im(item)]

    y_list = v_list[2*N_v:]
    y_0_list = v_num_list[2*N_v:]

    u_dict = dict(zip(v_list_str[:2*N_v],v_num_list[:2*N_v]))
    u_dict.update(dict(zip(i_list_str[2*N_v:],i_num_list[2*N_v:])))

    for bus in load_buses:

        v_a = V_node_sym_list[nodes_list.index(f'{bus}.1')]
        v_b = V_node_sym_list[nodes_list.index(f'{bus}.2')]
        v_c = V_node_sym_list[nodes_list.index(f'{bus}.3')]
        v_n = V_node_sym_list[nodes_list.index(f'{bus}.4')]

        i_a = I_node_sym_list[nodes_list.index(f'{bus}.1')]
        i_b = I_node_sym_list[nodes_list.index(f'{bus}.2')]
        i_c = I_node_sym_list[nodes_list.index(f'{bus}.3')]
        i_n = I_node_sym_list[nodes_list.index(f'{bus}.4')]


        v_an = v_a - v_n
        v_bn = v_b - v_n
        v_cn = v_c - v_n

        s_a = v_an*sym.conjugate(i_a)
        s_b = v_bn*sym.conjugate(i_b)
        s_c = v_cn*sym.conjugate(i_c)

        s = s_a + s_b + s_c
        p_a,p_b,p_c = sym.symbols(f'p_{bus}_a,p_{bus}_b,p_{bus}_c')
        q_a,q_b,q_c = sym.symbols(f'q_{bus}_a,q_{bus}_b,q_{bus}_c')
        g_list += [p_a + sym.re(s_a)]
        g_list += [p_b + sym.re(s_b)]
        g_list += [p_c + sym.re(s_c)]
        g_list += [q_a + sym.im(s_a)]
        g_list += [q_b + sym.im(s_b)]
        g_list += [q_c + sym.im(s_c)]

        g_list += [sym.re(i_a+i_b+i_c+i_n)]
        g_list += [sym.im(i_a+i_b+i_c+i_n)]

        buses_list = [bus['bus'] for bus in grid_1.buses]

        for phase in ['a','b','c']:
            i_real,i_imag = sym.symbols(f'i_{bus}_{phase}_r,i_{bus}_{phase}_i', real=True)
            y_list += [i_real,i_imag]
            i_cplx = I_node[grid_1.nodes.index(f'{bus}.{a2n[phase]}')][0]
            y_0_list += [i_cplx.real,i_cplx.imag]
            u_dict.pop(f'i_{bus}_{phase}_r')
            u_dict.pop(f'i_{bus}_{phase}_i')
            p_value = grid_1.buses[buses_list.index(bus)][f'p_{phase}']
            q_value = grid_1.buses[buses_list.index(bus)][f'q_{phase}']
            u_dict.update({f'p_{bus}_{phase}':p_value})
            u_dict.update({f'q_{bus}_{phase}':q_value})

        i_real,i_imag = sym.symbols(f'i_{bus}_n_r,i_{bus}_n_i', real=True)
        y_list += [i_real,i_imag]    
        i_cplx = I_node[grid_1.nodes.index(f'{bus}.{a2n["n"]}')][0]
        y_0_list += [i_cplx.real,i_cplx.imag]
        
    return y_list,g_list,y_0_list,u_dict