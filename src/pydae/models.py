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


def pydgrid2pydae(grid):
    
    nodes_list = grid.nodes
    I_node = grid.I_node
    V_node = grid.V_node
    Y_vv = grid.Y_vv
    Y_ii = grid.Y_ii.toarray()
    Y_iv = grid.Y_iv
    Y_vi = grid.Y_vi
    inv_Y_ii = np.linalg.inv(Y_ii)
    N_nz_nodes = grid.params_pf[0].N_nz_nodes
    N_v = grid.params_pf[0].N_nodes_v
    buses_list = [bus['bus'] for bus in grid.buses]
    
    N_v = Y_iv.shape[1]   # number of nodes with known voltages
    I_node_sym_list = []
    V_node_sym_list = []
    v_cplx_list = []
    v_list = []
    v_m_list = []
    i_list = []
    v_list_str = []
    i_list_str = []
    i_node = []
    v_num_list = []
    i_num_list = []
    h_v_m_dict = {}
    h_i_m_dict = {}
    
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
        v_cplx_list += [v_real+1j*v_imag]
        i_list += [i_real,i_imag]
        
        v_m = (v_real**2+v_imag**2)**0.5
        #i_m = (i_real**2+i_imag**2)**0.5

        
        h_v_m_dict.update({f"v_{bus_name}_{n2a[phase]}_m":v_m})
        #h_i_m_dict.update({f"i_{bus_name}_{n2a[phase]}_m":i_m})
    
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

    y_list   = v_list[2*N_v:]
    y_0_list = v_num_list[2*N_v:]

    u_dict = dict(zip(v_list_str[:2*N_v],v_num_list[:2*N_v]))
    u_dict.update(dict(zip(i_list_str[2*N_v:],i_num_list[2*N_v:])))

    for load in grid.loads:
        if load['type'] == '1P+N':
            bus_name = load['bus']
            phase_1 = str(load['bus_nodes'][0])
            i_real_1 = sym.Symbol(f"i_{bus_name}_{n2a[phase_1]}_r", real=True)
            i_imag_1 = sym.Symbol(f"i_{bus_name}_{n2a[phase_1]}_i", real=True)
            v_real_1 = sym.Symbol(f"v_{bus_name}_{n2a[phase_1]}_r", real=True)
            v_imag_1 = sym.Symbol(f"v_{bus_name}_{n2a[phase_1]}_i", real=True)          
            i_1 = i_real_1 +1j*i_imag_1
            v_1 = v_real_1 +1j*v_imag_1

            phase_2 = str(load['bus_nodes'][1])
            i_real_2 = sym.Symbol(f"i_{bus_name}_{n2a[phase_2]}_r", real=True)
            i_imag_2 = sym.Symbol(f"i_{bus_name}_{n2a[phase_2]}_i", real=True)
            v_real_2 = sym.Symbol(f"v_{bus_name}_{n2a[phase_2]}_r", real=True)
            v_imag_2 = sym.Symbol(f"v_{bus_name}_{n2a[phase_2]}_i", real=True)          
            i_2 = i_real_2 +1j*i_imag_2
            v_2 = v_real_2 +1j*v_imag_2

            v_12 = v_1 - v_2

            s_1 = v_12*sym.conjugate(i_1)

            p_1,p_2 = sym.symbols(f'p_{bus_name}_1,p_{bus_name}_2')
            q_1,q_2 = sym.symbols(f'q_{bus_name}_1,q_{bus_name}_2')

            g_list += [-p_1 + sym.re(s_1)]
            g_list += [-q_1 + sym.im(s_1)]

            y_list += [i_real_1,i_imag_1]
            
            g_list += [sym.re(i_1+i_2)]
            g_list += [sym.im(i_1+i_2)]

            y_list += [i_real_2,i_imag_2]
            
            i_real,i_imag = sym.symbols(f'i_{bus_name}_{phase}_r,i_{bus_name}_{phase}_i', real=True)

            i_cplx_1 = I_node[grid.nodes.index(f'{bus_name}.{phase_1}')][0]
            y_0_list += [i_cplx_1.real,i_cplx_1.imag]
            i_cplx_2 = I_node[grid.nodes.index(f'{bus_name}.{phase_2}')][0]
            y_0_list += [i_cplx_2.real,i_cplx_2.imag]
            
            u_dict.pop(f'i_{bus_name}_{n2a[phase_1]}_r')
            u_dict.pop(f'i_{bus_name}_{n2a[phase_1]}_i')
            u_dict.pop(f'i_{bus_name}_{n2a[phase_2]}_r')
            u_dict.pop(f'i_{bus_name}_{n2a[phase_2]}_i')            
            
            p_value = grid.buses[buses_list.index(bus_name)][f'p_{n2a[phase_1]}']
            q_value = grid.buses[buses_list.index(bus_name)][f'q_{n2a[phase_1]}']
            u_dict.update({f'p_{bus_name}_{phase_1}':p_value})
            u_dict.update({f'q_{bus_name}_{phase_1}':q_value})
                
        if load['type'] == '3P+N':
            bus_name = load['bus']
            v_a = V_node_sym_list[nodes_list.index(f'{bus_name}.1')]
            v_b = V_node_sym_list[nodes_list.index(f'{bus_name}.2')]
            v_c = V_node_sym_list[nodes_list.index(f'{bus_name}.3')]
            v_n = V_node_sym_list[nodes_list.index(f'{bus_name}.4')]

            i_a = I_node_sym_list[nodes_list.index(f'{bus_name}.1')]
            i_b = I_node_sym_list[nodes_list.index(f'{bus_name}.2')]
            i_c = I_node_sym_list[nodes_list.index(f'{bus_name}.3')]
            i_n = I_node_sym_list[nodes_list.index(f'{bus_name}.4')]


            v_an = v_a - v_n
            v_bn = v_b - v_n
            v_cn = v_c - v_n

            s_a = v_an*sym.conjugate(i_a)
            s_b = v_bn*sym.conjugate(i_b)
            s_c = v_cn*sym.conjugate(i_c)

            s = s_a + s_b + s_c
            p_a,p_b,p_c = sym.symbols(f'p_{bus_name}_a,p_{bus_name}_b,p_{bus_name}_c')
            q_a,q_b,q_c = sym.symbols(f'q_{bus_name}_a,q_{bus_name}_b,q_{bus_name}_c')
            g_list += [p_a + sym.re(s_a)]
            g_list += [p_b + sym.re(s_b)]
            g_list += [p_c + sym.re(s_c)]
            g_list += [q_a + sym.im(s_a)]
            g_list += [q_b + sym.im(s_b)]
            g_list += [q_c + sym.im(s_c)]

            g_list += [sym.re(i_a+i_b+i_c+i_n)]
            g_list += [sym.im(i_a+i_b+i_c+i_n)]

            

            for phase in ['a','b','c']:
                i_real,i_imag = sym.symbols(f'i_{bus_name}_{phase}_r,i_{bus_name}_{phase}_i', real=True)
                y_list += [i_real,i_imag]
                i_cplx = I_node[grid.nodes.index(f'{bus_name}.{a2n[phase]}')][0]
                y_0_list += [i_cplx.real,i_cplx.imag]
                u_dict.pop(f'i_{bus_name}_{phase}_r')
                u_dict.pop(f'i_{bus_name}_{phase}_i')
                p_value = grid.buses[buses_list.index(bus_name)][f'p_{phase}']
                q_value = grid.buses[buses_list.index(bus_name)][f'q_{phase}']
                u_dict.update({f'p_{bus_name}_{phase}':p_value})
                u_dict.update({f'q_{bus_name}_{phase}':q_value})

            i_real,i_imag = sym.symbols(f'i_{bus_name}_n_r,i_{bus_name}_n_i', real=True)
            y_list += [i_real,i_imag]    
            i_cplx = I_node[grid.nodes.index(f'{bus_name}.{a2n["n"]}')][0]
            y_0_list += [i_cplx.real,i_cplx.imag]

    f_list = []   
    x_list = []    
    return {'g':g_list,'y':y_list,'f':f_list,'x':x_list,
            'u':u_dict,'y_0_list':y_0_list,'v_list':v_list,'v_m_list':v_m_list,'v_cplx_list':v_cplx_list,
            'h_v_m_dict':h_v_m_dict}   


 
    
def dcgrid2dae(data_input):
    vscs = data_input['grid_formers']
    park_type='original'
    dq_name='DQ'

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
            'u':u_grid,'params':params,'v_list':v_list}


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


    