# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym



def load_ac_3w(grid,data):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    self = grid

    name = data['bus']
    v_sa_r,v_sb_r,v_sc_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r', real=True)
    v_sa_i,v_sb_i,v_sc_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i', real=True)
    K_abc,V_th = sym.symbols(f'K_abc_{name},V_th_{name}',real=True)
    # i_a = I_node_sym_list[nodes_list.index(f'{bus_name}.1')]
    # i_b = I_node_sym_list[nodes_list.index(f'{bus_name}.2')]
    # i_c = I_node_sym_list[nodes_list.index(f'{bus_name}.3')]
    # i_n = I_node_sym_list[nodes_list.index(f'{bus_name}.4')]

    i_a_r,i_a_i = sym.symbols(f'i_load_{name}_a_r,i_load_{name}_a_i', real=True)
    i_b_r,i_b_i = sym.symbols(f'i_load_{name}_b_r,i_load_{name}_b_i', real=True)
    i_c_r,i_c_i = sym.symbols(f'i_load_{name}_c_r,i_load_{name}_c_i', real=True)
    
    i_a = i_a_r + sym.I*i_a_i
    i_b = i_b_r + sym.I*i_b_i
    i_c = i_c_r + sym.I*i_c_i
    
    v_a = v_sa_r +  sym.I*v_sa_i
    v_b = v_sb_r +  sym.I*v_sb_i
    v_c = v_sc_r +  sym.I*v_sc_i

    s_a = v_a*sym.conjugate(i_a)
    s_b = v_b*sym.conjugate(i_b)
    s_c = v_c*sym.conjugate(i_c)
    #s = s_a + s_b + s_c

    p,q = sym.symbols(f'p_load_{name},q_load_{name}', real=True)
    
    self.dae['g'] += [p/3 + sym.re(s_a)]
    self.dae['g'] += [p/3 + sym.re(s_b)]
    self.dae['g'] += [q/3 + sym.im(s_a)]
    self.dae['g'] += [q/3 + sym.im(s_b)]

    self.dae['g'] += [sym.re(i_a+i_b+i_c)]
    self.dae['g'] += [sym.im(i_a+i_b+i_c)]

    self.dae['y_ini'] += [i_a_r]
    self.dae['y_ini'] += [i_a_i]
    self.dae['y_ini'] += [i_b_r]
    self.dae['y_ini'] += [i_b_i]
    self.dae['y_ini'] += [i_c_r]
    self.dae['y_ini'] += [i_c_i]

    self.dae['y_run'] += [i_a_r]
    self.dae['y_run'] += [i_a_i]
    self.dae['y_run'] += [i_b_r]
    self.dae['y_run'] += [i_b_i]
    self.dae['y_run'] += [i_c_r]
    self.dae['y_run'] += [i_c_i]


    for ph in ['a','b','c']:
        i_s_r,i_s_i = sym.symbols(f'i_load_{name}_{ph}_r,i_load_{name}_{ph}_i', real=True)
        idx_r,idx_i = self.node2idx(name,ph)
        self.dae['g'] [idx_r] += -i_s_r
        self.dae['g'] [idx_i] += -i_s_i


    p_load_N,q_load_N = 0.0,0.0
    if 'kVA' in data:
        S_N = data['kVA']*1000
        pf_N = data['pf']
        p_load_N = S_N*np.abs(pf_N)
        q_load_N = np.sqrt(S_N**2 - p_load_N**2)*np.sign(pf_N)
    if 'kW' in data:
        p_load_N =  data['kW']*1000
        q_load_N =  data['kvar']*1000
        
    self.dae['u_ini_dict'].update({f'p_load_{name}':p_load_N})
    self.dae['u_ini_dict'].update({f'q_load_{name}':q_load_N})
    self.dae['u_run_dict'].update({f'p_load_{name}':p_load_N})
    self.dae['u_run_dict'].update({f'q_load_{name}':q_load_N})


