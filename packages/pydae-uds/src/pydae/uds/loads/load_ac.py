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

    bk = grid.backend

    name = data['bus']
    v_sa_r,v_sb_r,v_sc_r,v_sn_r = bk.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r')
    v_sa_i,v_sb_i,v_sc_i,v_sn_i = bk.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i')
    K_abc = bk.symbols(f'K_abc_{name}')

    i_a_r,i_a_i = bk.symbols(f'i_load_{name}_a_r,i_load_{name}_a_i')
    i_b_r,i_b_i = bk.symbols(f'i_load_{name}_b_r,i_load_{name}_b_i')
    i_c_r,i_c_i = bk.symbols(f'i_load_{name}_c_r,i_load_{name}_c_i')
    i_n_r,i_n_i = bk.symbols(f'i_load_{name}_n_r,i_load_{name}_n_i')

    # phase-to-neutral voltages, real form
    van_r,van_i = v_sa_r - v_sn_r, v_sa_i - v_sn_i
    vbn_r,vbn_i = v_sb_r - v_sn_r, v_sb_i - v_sn_i
    vcn_r,vcn_i = v_sc_r - v_sn_r, v_sc_i - v_sn_i

    van_m2 = van_r**2 + van_i**2
    vbn_m2 = vbn_r**2 + vbn_i**2
    vcn_m2 = vcn_r**2 + vcn_i**2

    v_anm = (v_sa_r**2 + v_sa_i**2)**0.5
    v_bnm = (v_sb_r**2 + v_sb_i**2)**0.5
    v_cnm = (v_sc_r**2 + v_sc_i**2)**0.5

    V_th = 0.7

    Kv_anm_lim = bk.Piecewise(((v_anm+0.3),v_anm<V_th),(1.0,v_anm>=V_th))
    Kv_bnm_lim = bk.Piecewise(((v_bnm+0.3),v_bnm<V_th),(1.0,v_bnm>=V_th))
    Kv_cnm_lim = bk.Piecewise(((v_cnm+0.3),v_cnm<V_th),(1.0,v_cnm>=V_th))

    # constant-power part: s = v_pn * conj(i)  ->  re = vr*ir + vi*ii, im = vi*ir - vr*ii
    p_s_a = van_r*i_a_r + van_i*i_a_i
    p_s_b = vbn_r*i_b_r + vbn_i*i_b_i
    p_s_c = vcn_r*i_c_r + vcn_i*i_c_i
    q_s_a = van_i*i_a_r - van_r*i_a_i
    q_s_b = vbn_i*i_b_r - vbn_r*i_b_i
    q_s_c = vcn_i*i_c_r - vcn_r*i_c_i

    p_a,p_b,p_c = bk.symbols(f'p_load_{name}_a,p_load_{name}_b,p_load_{name}_c')
    q_a,q_b,q_c = bk.symbols(f'q_load_{name}_a,q_load_{name}_b,q_load_{name}_c')
    g_a,g_b,g_c = bk.symbols(f'g_load_{name}_a,g_load_{name}_b,g_load_{name}_c')
    b_a,b_b,b_c = bk.symbols(f'b_load_{name}_a,b_load_{name}_b,b_load_{name}_c')

    # constant-impedance part: s_z = conj((g+jb)*v_pn)*v_pn = conj(g+jb)*|v_pn|^2
    #   -> re(s_z) = g*|v_pn|^2 , im(s_z) = -b*|v_pn|^2
    p_z_a, q_z_a = g_a*van_m2, -b_a*van_m2
    p_z_b, q_z_b = g_b*vbn_m2, -b_b*vbn_m2
    p_z_c, q_z_c = g_c*vcn_m2, -b_c*vcn_m2

    self.dae['g'] += [K_abc*(p_a + p_z_a + p_s_a/Kv_anm_lim)]
    self.dae['g'] += [K_abc*(p_b + p_z_b + p_s_b/Kv_bnm_lim)]
    self.dae['g'] += [K_abc*(p_c + p_z_c + p_s_c/Kv_cnm_lim)]
    self.dae['g'] += [K_abc*(q_a + q_z_a + q_s_a/Kv_anm_lim)]
    self.dae['g'] += [K_abc*(q_b + q_z_b + q_s_b/Kv_bnm_lim)]
    self.dae['g'] += [K_abc*(q_c + q_z_c + q_s_c/Kv_cnm_lim)]

    self.dae['g'] += [i_a_r+i_b_r+i_c_r+i_n_r]
    self.dae['g'] += [i_a_i+i_b_i+i_c_i+i_n_i]

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
        i_s_r,i_s_i = bk.symbols(f'i_load_{name}_{ph}_r,i_load_{name}_{ph}_i')
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