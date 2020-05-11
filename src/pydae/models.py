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

    for load in data['loads']:
        if 'I_max' not in load: 
            if 'kVA' in load:
                I_max = load['kVA']*1000/690
            load['I_max'] = I_max
        if 'T_i' not in line: 
            load['T_i'] = 0.01        
    return data

def grid_dq(data_input):

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

    omega = sym.Symbol('omega')
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

        i_l_d = sym.Symbol(f'i_l_{sub_name}_d', real=True)
        i_l_q = sym.Symbol(f'i_l_{sub_name}_q', real=True)
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
        v_d = sym.Symbol(f'v_{bus_name}_d', real=True)
        v_q = sym.Symbol(f'v_{bus_name}_q', real=True)
        i_d = sym.Symbol(f'i_{bus_name}_d', real=True)
        i_q = sym.Symbol(f'i_{bus_name}_q', real=True)

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

    di_l_dq =  (-(R_e + L_e @ Omega_e_M) @ i_l_dq + A_e @ v_dq)
    dv_dq =  (-C_e @ Omega_e_N @ v_dq - A_e.T @ i_l_dq + i_dq)

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
        u_grid.update({f'v_{bus_name}_d':v_d,f'v_{bus_name}_q':v_q})

    for load in loads:

        bus_name = load['bus']
        i_d_ref = sym.Symbol(f'i_{bus_name}_d_ref', real=True)
        i_q_ref = sym.Symbol(f'i_{bus_name}_q_ref', real=True)
        i_d = sym.Symbol(f'i_{bus_name}_d', real=True)
        i_q = sym.Symbol(f'i_{bus_name}_q', real=True)
        T_i = sym.Symbol(f'T_i_{bus_name}', real=True)
        I_max = sym.Symbol(f'I_max_{bus_name}', real=True)

        p_ref = sym.Symbol(f'p_{bus_name}_ref', real=True)
        q_ref = sym.Symbol(f'q_{bus_name}_ref', real=True)
        v_d = sym.Symbol(f'v_{bus_name}_d', real=True)
        v_q = sym.Symbol(f'v_{bus_name}_q', real=True)

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
            params.update({f'i_{bus_name}_d':0.0,f'i_{bus_name}_q':0.0})

    f_hz = data['system']['f_hz']
    params.update({'omega':2*np.pi*f_hz})

    return {'f':f_grid,'g':g_grid,
            'x':x_grid_list,'y':y_grid_list,
            'u':u_grid,'params':params,'v_list':v_list}