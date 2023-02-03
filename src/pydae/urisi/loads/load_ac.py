# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym



def load_ac(grid,data):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    self = grid

    name = data['bus']
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)

    # i_a = I_node_sym_list[nodes_list.index(f'{bus_name}.1')]
    # i_b = I_node_sym_list[nodes_list.index(f'{bus_name}.2')]
    # i_c = I_node_sym_list[nodes_list.index(f'{bus_name}.3')]
    # i_n = I_node_sym_list[nodes_list.index(f'{bus_name}.4')]

    i_a_r,i_a_i = sym.symbols(f'i_load_{name}_a_r,i_load_{name}_a_i', real=True)
    i_b_r,i_b_i = sym.symbols(f'i_load_{name}_b_r,i_load_{name}_b_i', real=True)
    i_c_r,i_c_i = sym.symbols(f'i_load_{name}_c_r,i_load_{name}_c_i', real=True)
    i_n_r,i_n_i = sym.symbols(f'i_load_{name}_n_r,i_load_{name}_n_i', real=True)
    
    i_a = i_a_r + sym.I*i_a_i
    i_b = i_b_r + sym.I*i_b_i
    i_c = i_c_r + sym.I*i_c_i
    i_n = i_n_r + sym.I*i_n_i
    
    v_a = v_sa_r +  sym.I*v_sa_i
    v_b = v_sb_r +  sym.I*v_sb_i
    v_c = v_sc_r +  sym.I*v_sc_i
    v_n = v_sn_r +  sym.I*v_sn_i

    v_an = v_a - v_n
    v_bn = v_b - v_n
    v_cn = v_c - v_n

    s_a = v_an*sym.conjugate(i_a)
    s_b = v_bn*sym.conjugate(i_b)
    s_c = v_cn*sym.conjugate(i_c)
    #s = s_a + s_b + s_c

    p_a,p_b,p_c = sym.symbols(f'p_load_{name}_a,p_load_{name}_b,p_load_{name}_c', real=True)
    q_a,q_b,q_c = sym.symbols(f'q_load_{name}_a,q_load_{name}_b,q_load_{name}_c', real=True)
    g_a,g_b,g_c = sym.symbols(f'g_load_{name}_a,g_load_{name}_b,g_load_{name}_c', real=True)
    b_a,b_b,b_c = sym.symbols(f'b_load_{name}_a,b_load_{name}_b,b_load_{name}_c', real=True)

    s_z_a = sym.conjugate((g_a + 1j*b_a)*v_an)*v_an
    s_z_b = sym.conjugate((g_b + 1j*b_b)*v_bn)*v_bn
    s_z_c = sym.conjugate((g_c + 1j*b_c)*v_cn)*v_cn
    
    self.dae['g'] += [p_a + sym.re(s_z_a) + sym.re(s_a)]
    self.dae['g'] += [p_b + sym.re(s_z_b) + sym.re(s_b)]
    self.dae['g'] += [p_c + sym.re(s_z_c) + sym.re(s_c)]
    self.dae['g'] += [q_a + sym.im(s_z_a) + sym.im(s_a)]
    self.dae['g'] += [q_b + sym.im(s_z_b) + sym.im(s_b)]
    self.dae['g'] += [q_c + sym.im(s_z_c) + sym.im(s_c)]

    self.dae['g'] += [sym.re(i_a+i_b+i_c+i_n)]
    self.dae['g'] += [sym.im(i_a+i_b+i_c+i_n)]

    self.dae['y_ini'] += [i_a_r]
    self.dae['y_ini'] += [i_a_i]
    self.dae['y_ini'] += [i_b_r]
    self.dae['y_ini'] += [i_b_i]
    self.dae['y_ini'] += [i_c_r]
    self.dae['y_ini'] += [i_c_i]
    self.dae['y_ini'] += [i_n_r]
    self.dae['y_ini'] += [i_n_i]

    self.dae['y_run'] += [i_a_r]
    self.dae['y_run'] += [i_a_i]
    self.dae['y_run'] += [i_b_r]
    self.dae['y_run'] += [i_b_i]
    self.dae['y_run'] += [i_c_r]
    self.dae['y_run'] += [i_c_i]
    self.dae['y_run'] += [i_n_r]
    self.dae['y_run'] += [i_n_i]


    for ph in ['a','b','c','n']:
        i_s_r,i_s_i = sym.symbols(f'i_load_{name}_{ph}_r,i_load_{name}_{ph}_i', real=True)
        idx_r,idx_i = self.node2idx(name,ph)
        self.dae['g'] [idx_r] += -i_s_r
        self.dae['g'] [idx_i] += -i_s_i



    for phase in ['a','b','c']:
        p_value,q_value = 1e3,0
        self.dae['u_ini_dict'].update({f'p_load_{name}_{phase}':p_value})
        self.dae['u_ini_dict'].update({f'q_load_{name}_{phase}':q_value})
        self.dae['u_run_dict'].update({f'p_load_{name}_{phase}':p_value})
        self.dae['u_run_dict'].update({f'q_load_{name}_{phase}':q_value})
        self.dae['u_ini_dict'].update({f'g_load_{name}_{phase}':0})
        self.dae['u_ini_dict'].update({f'b_load_{name}_{phase}':0})
        self.dae['u_run_dict'].update({f'g_load_{name}_{phase}':0})
        self.dae['u_run_dict'].update({f'b_load_{name}_{phase}':0})
