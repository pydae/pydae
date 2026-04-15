# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def ac3ph4wgf(grid,vsc_data):
    '''
    VSC with 3 phase and 4 wire working in open loop as a grid former.

    
    '''

    params_dict  = grid.dae['params_dict']
    u_dict = {}
    h_dict = grid.dae['h_dict']


    alpha = np.exp(2.0/3*np.pi*1j)
    A_0a =  np.array([[1, 1, 1],
                    [1, alpha**2, alpha],
                    [1, alpha, alpha**2]])

    A_a0 = 1/3* np.array([[1, 1, 1],
                        [1, alpha, alpha**2],
                        [1, alpha**2, alpha]])

    omega_coi_i = 0
    HS_coi = 0

    omega_coi = sym.Symbol('omega_coi',real=True)
    xi_freq = sym.Symbol('xi_freq',real=True)
    K_agc = sym.Symbol('K_agc',real=True)

    #vscs = [
    #    {'bus':'B1','S_n':100e3,'R':0.01,'X':0.1,'R_n':0.01,'X_n':0.1,'R_ng':0.01,'X_ng':3.0,'K_f':0.1,'T_f':1.0,'K_sec':0.5,'K_delta':0.001},
    #    ]

    #for vsc in vsc_data:
        
    name = vsc_data['bus']

    # inputs
    e_am_m,e_bm_m,e_cm_m,e_om_m = sym.symbols(f'e_{name}_am_m,e_{name}_bm_m,e_{name}_cm_m,e_{name}_om_m', real=True)
    omega_ref,p_ref = sym.symbols(f'omega_{name}_ref,p_{name}_ref', real=True)
    
    # parameters
    S_n,H,K_f,T_f,K_sec,K_delta  = sym.symbols(f'S_n_{name},H_{name},K_f_{name},T_f_{name},K_sec_{name},K_delta_{name}', real=True)
    R_s,R_sn,R_ng = sym.symbols(f'R_{name}_s,R_{name}_sn,R_{name}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name}_s,X_{name}_sn,X_{name}_ng', real=True)
    
    # dynamical states
    phi = sym.Symbol(f'phi_{name}', real=True)
    phi_a = sym.Symbol(f'phi_a_{name}', real=True)
    phi_b = sym.Symbol(f'phi_b_{name}', real=True)
    phi_c = sym.Symbol(f'phi_c_{name}', real=True)

    omega = sym.Symbol(f'omega_{name}', real=True)
    
    # algebraic states
    #e_an_i,e_bn_i,e_cn_i,e_ng_i = sym.symbols(f'e_{name}_an_i,e_{name}_bn_i,e_{name}_cn_i,e_{name}_ng_i', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_ng_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_n_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_ng_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_n_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    v_mn_r,v_mn_i = sym.symbols(f'v_{name}_mn_r,v_{name}_mn_i', real=True)

    omega = sym.Symbol(f'omega_{name}', real=True)
    
    e_om_r,e_om_i = sym.symbols(f'e_{name}_om_r,e_{name}_om_i', real=True)
    
    Z_sa = R_s + 1j*X_s
    Z_sb = R_s + 1j*X_s
    Z_sc = R_s + 1j*X_s
    Z_sn = R_sn + 1j*X_sn
    Z_ng = R_ng + 1j*X_ng

    i_sa = i_sa_r + 1j*i_sa_i
    i_sb = i_sb_r + 1j*i_sb_i
    i_sc = i_sc_r + 1j*i_sc_i
    i_sn = i_sn_r + 1j*i_sn_i
    i_ng = i_ng_r + 1j*i_ng_i

    v_sa = v_sa_r + 1j*v_sa_i
    v_sb = v_sb_r + 1j*v_sb_i
    v_sc = v_sc_r + 1j*v_sc_i
    v_sn = v_sn_r + 1j*v_sn_i
    v_ng = v_ng_r + 1j*v_ng_i
    v_mn = v_mn_r + 1j*v_mn_i
    
    e_am_r = e_am_m*sym.cos(phi_a + phi) 
    e_am_i = e_am_m*sym.sin(phi_a + phi) 
    e_bm_r = e_bm_m*sym.cos(phi_b + phi-2/3*np.pi) 
    e_bm_i = e_bm_m*sym.sin(phi_b + phi-2/3*np.pi) 
    e_cm_r = e_cm_m*sym.cos(phi_c + phi-4/3*np.pi) 
    e_cm_i = e_cm_m*sym.sin(phi_c + phi-4/3*np.pi) 
    
    e_am_cplx = e_am_r + 1j*e_am_i
    e_bm_cplx = e_bm_r + 1j*e_bm_i
    e_cm_cplx = e_cm_r + 1j*e_cm_i
    e_om_cplx = e_om_r + 1j*e_om_i

    v_san = v_sa - v_sn
    v_sbn = v_sb - v_sn
    v_scn = v_sc - v_sn

    eq_i_sa_cplx = e_am_cplx - i_sa*Z_sa - v_san - v_mn
    eq_i_sb_cplx = e_bm_cplx - i_sb*Z_sb - v_sbn - v_mn
    eq_i_sc_cplx = e_cm_cplx - i_sc*Z_sc - v_scn - v_mn
    eq_v_nm_cplx = 0*e_om_cplx - i_sn*Z_sn - v_mn
    eq_i_sn_cplx = i_sa + i_sb + i_sc + i_sn
    #eq_i_sn_cplx = e_ng_cplx - i_sn*Z_sn - v_ng
    #eq_i_ng_cplx = i_ng + i_sa + i_sb + i_sc + i_sn
    #eq_e_ng_cplx  = -e_ng_cplx  + i_ng*Z_ng

    grid.dae['g'] += [sym.re(eq_i_sa_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sb_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sc_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_nm_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sn_cplx)] 
    
    #g_list += [sym.re(eq_i_ng_cplx)] 
    #g_list += [sym.re(eq_e_ng_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sa_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sb_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sc_cplx)] 
    grid.dae['g'] += [sym.im(eq_v_nm_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sn_cplx)] 
    #g_list += [sym.im(eq_i_ng_cplx)] 
    #g_list += [sym.im(eq_e_ng_cplx)]

    grid.dae['y_ini'] += [i_sa_r,i_sb_r,i_sc_r,v_mn_r,i_sn_r]
    grid.dae['y_ini'] += [i_sa_i,i_sb_i,i_sc_i,v_mn_i,i_sn_i]

    grid.dae['y_run'] += [i_sa_r,i_sb_r,i_sc_r,v_mn_r,i_sn_r]
    grid.dae['y_run'] += [i_sa_i,i_sb_i,i_sc_i,v_mn_i,i_sn_i]


    for ph in ['a','b','c','n']:
        i_s_r = sym.Symbol(f'i_vsc_{name}_{ph}_r', real=True)
        i_s_i = sym.Symbol(f'i_vsc_{name}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(name,ph)
        grid.dae['g'] [idx_r] += i_s_r
        grid.dae['g'] [idx_i] += i_s_i
        i_s = i_s_r + 1j*i_s_i
        i_s_m = np.abs(i_s)
        h_dict.update({f'i_vsc_{name}_{ph}_m':i_s_m})

        
    V_1 = 400/np.sqrt(3)
    #    V_1 = 400/np.sqrt(3)*np.exp(1j*np.deg2rad(0))
    # A_1toabc = np.array([1, alpha**2, alpha])
    #V_abc = V_1 * A_1toabc 
    #e_an_r,e_bn_r,e_cn_r = V_abc.real
    #e_an_i,e_bn_i,e_cn_i = V_abc.imag

    u_dict.update({f'e_{name}_am_m':V_1,f'e_{name}_bm_m':V_1,f'e_{name}_cm_m':V_1,f'e_{name}_om_m':0.0})
    #u_dict.update({f'phi_{name}':0.0})
    u_dict.update({f'phi_a_{name}':0.0})
    u_dict.update({f'phi_b_{name}':0.0})
    u_dict.update({f'phi_c_{name}':0.0})

    u_dict.update({f'p_{name}_ref':0.0})
    u_dict.update({f'omega_{name}_ref':1.0})

    grid.dae['u_ini_dict'].update(u_dict)
    grid.dae['u_run_dict'].update(u_dict)

    #for ph in ['a','b','c','n']:
    #    u_dict.pop(f'i_{name}_{ph}_r')
    #    u_dict.pop(f'i_{name}_{ph}_i')

    params_dict.update({f'X_{name}_s':vsc_data['X'],f'R_{name}_s':vsc_data['R']})
    params_dict.update({f'X_{name}_sn':vsc_data['X_n'],f'R_{name}_sn':vsc_data['R_n']})
    params_dict.update({f'X_{name}_ng':vsc_data['X_ng'],f'R_{name}_ng':vsc_data['R_ng']})
    
    params_dict.update({f'S_n_{name}':vsc_data['S_n']})

    params_dict.update({f'K_f_{name}':vsc_data['K_f']})
    params_dict.update({f'T_f_{name}':vsc_data['T_f']})
    params_dict.update({f'K_sec_{name}':vsc_data['K_sec']})
    params_dict.update({f'K_delta_{name}':vsc_data['K_delta']})
    
    
    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])
    
    v_szpn = A_a0*v_sabc
    i_szpn = A_a0*i_sabc
    
    s_pos = 3*v_szpn[1]*sym.conjugate(i_szpn[1])
    s_neg = 3*v_szpn[2]*sym.conjugate(i_szpn[2])
    s_zer = 3*v_szpn[0]*sym.conjugate(i_szpn[0])
    
    p_pos = sym.re(s_pos)
    
    dphi   = 2*np.pi*50*(omega - omega_coi) - K_delta*phi
    domega = 1/T_f*(omega_ref + K_f*(p_ref + K_sec*xi_freq - p_pos)/S_n - omega)
    
    grid.dae['f'] += [dphi,domega]
    grid.dae['x'] += [ phi, omega]
    
    h_dict.update({f'p_{name}_pos':sym.re(s_pos),f'p_{name}_neg':sym.re(s_neg),f'p_{name}_zer':sym.re(s_zer)})
    h_dict.update({str(e_am_m):e_am_m,str(e_bm_m):e_bm_m,str(e_cm_m):e_cm_m})
    h_dict.update({str(p_ref):p_ref,str(omega_ref):omega_ref})
    h_dict.update({str(omega):omega})
    HS_coi  = S_n
    omega_coi_i = S_n*omega

    grid.omega_coi_numerator += omega_coi_i
    grid.omega_coi_denominator += HS_coi