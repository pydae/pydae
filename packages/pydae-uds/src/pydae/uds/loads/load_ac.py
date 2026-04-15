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
    K_abc,V_th = sym.symbols(f'K_abc_{name},V_th_{name}',real=True)
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

    v_anm = (v_sa_r**2 + v_sa_i**2)**0.5 
    v_bnm = (v_sb_r**2 + v_sb_i**2)**0.5 
    v_cnm = (v_sc_r**2 + v_sc_i**2)**0.5 

    V_th = 0.7

    Kv_anm_lim = sym.Piecewise(((v_anm+0.3),v_anm<V_th),(1.0,v_anm>=V_th))
    Kv_bnm_lim = sym.Piecewise(((v_bnm+0.3),v_bnm<V_th),(1.0,v_bnm>=V_th))
    Kv_cnm_lim = sym.Piecewise(((v_cnm+0.3),v_cnm<V_th),(1.0,v_cnm>=V_th))

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
    
    self.dae['g'] += [K_abc*(p_a + sym.re(s_z_a) + sym.re(s_a)/Kv_anm_lim)]
    self.dae['g'] += [K_abc*(p_b + sym.re(s_z_b) + sym.re(s_b)/Kv_bnm_lim)]
    self.dae['g'] += [K_abc*(p_c + sym.re(s_z_c) + sym.re(s_c)/Kv_cnm_lim)]
    self.dae['g'] += [K_abc*(q_a + sym.im(s_z_a) + sym.im(s_a)/Kv_anm_lim)]
    self.dae['g'] += [K_abc*(q_b + sym.im(s_z_b) + sym.im(s_b)/Kv_bnm_lim)]
    self.dae['g'] += [K_abc*(q_c + sym.im(s_z_c) + sym.im(s_c)/Kv_cnm_lim)]

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


    p_load_N,q_load_N = 0.0,0.0
    if 'kVA' in data:
        if isinstance(data['kVA'], float) or isinstance(data['kVA'],int):
            S_N = np.array([data['kVA']]*3)*1000.0
            pf_N = np.array([data['pf']]*3)
        else: 
            S_N = np.array(data['kVA'])*1000.0
            pf_N = np.array(data['pf'])
        p_load_N = S_N*np.abs(pf_N)
        q_load_N = np.sqrt(S_N**2 - p_load_N**2)*np.sign(pf_N)
    if 'kW' in data:
        p_load_N =  data['kW']*1000
        q_load_N =  data['kvar']*1000

    it = 0
    for phase in ['a','b','c']:
        self.dae['u_ini_dict'].update({f'p_load_{name}_{phase}':p_load_N[it]/3})
        self.dae['u_ini_dict'].update({f'q_load_{name}_{phase}':q_load_N[it]/3})
        self.dae['u_run_dict'].update({f'p_load_{name}_{phase}':p_load_N[it]/3})
        self.dae['u_run_dict'].update({f'q_load_{name}_{phase}':q_load_N[it]/3})
        self.dae['u_ini_dict'].update({f'g_load_{name}_{phase}':0.0})
        self.dae['u_ini_dict'].update({f'b_load_{name}_{phase}':0.0})
        self.dae['u_run_dict'].update({f'g_load_{name}_{phase}':0.0})
        self.dae['u_run_dict'].update({f'b_load_{name}_{phase}':0.0})
        it += 1

    self.dae['params_dict'].update({f'K_abc_{name}':1.0})
    self.dae['h_dict'].update({f'v_anm_{name}':v_anm})
    self.dae['h_dict'].update({f'v_bnm_{name}':v_bnm})
    self.dae['h_dict'].update({f'v_cnm_{name}':v_cnm})