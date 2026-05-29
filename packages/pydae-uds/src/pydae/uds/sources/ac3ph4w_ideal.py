
import numpy as np
import sympy as sym

def ac3ph4w_ideal(grid,data):
    '''
    VSC with 3 phase and 4 wire working in open loop as a grid former.

    '''
    bk = grid.backend

    params_dict  = grid.dae['params_dict']
    g_list = grid.dae['g']
    y_ini_list = grid.dae['y_ini']
    u_ini_dict = grid.dae['u_ini_dict']
    y_run_list = grid.dae['y_run']
    u_run_dict = grid.dae['u_run_dict']
    h_dict = grid.dae['h_dict']

    name = data['bus']

    # inputs
    e_ao_m,e_bo_m,e_co_m = bk.symbols(f'e_ao_m_{name},e_bo_m_{name},e_co_m_{name}')
    v_sa_r,v_sb_r,v_sc_r,v_sn_r = bk.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r')
    v_sa_i,v_sb_i,v_sc_i,v_sn_i = bk.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i')
    phi =  bk.symbols(f'phi_{name}')

    # parameters
    R_s,R_sn = bk.symbols(f'R_s_{name},R_sn_{name}')
    X_s,X_sn = bk.symbols(f'X_s_{name},X_sn_{name}')

    # algebraic states (real/imaginary parts of the source branch currents)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r = bk.symbols(f'i_vsrc_{name}_a_r,i_vsrc_{name}_b_r,i_vsrc_{name}_c_r,i_vsrc_{name}_n_r')
    i_sa_i,i_sb_i,i_sc_i,i_sn_i = bk.symbols(f'i_vsrc_{name}_a_i,i_vsrc_{name}_b_i,i_vsrc_{name}_c_i,i_vsrc_{name}_n_i')

    # series impedances in real form (Z = R + jX)
    Z_s_re,  Z_s_im  = R_s,  X_s
    Z_sn_re, Z_sn_im = R_sn, X_sn

    # internal EMFs of the balanced three-phase set (real form)
    e_a_r = (e_ao_m)*bk.cos(phi)
    e_a_i = (e_ao_m)*bk.sin(phi)
    e_b_r = (e_bo_m)*bk.cos(phi-2/3*np.pi)
    e_b_i = (e_bo_m)*bk.sin(phi-2/3*np.pi)
    e_c_r = (e_co_m)*bk.cos(phi-4/3*np.pi)
    e_c_i = (e_co_m)*bk.sin(phi-4/3*np.pi)

    # branch equation  e - i_s*Z_s - v_s = 0, split into real/imaginary parts.
    # (a+jb)(c+jd) = (ac-bd) + j(ad+bc)
    eq_i_sa_re = e_a_r - (i_sa_r*Z_s_re  - i_sa_i*Z_s_im)  - v_sa_r
    eq_i_sb_re = e_b_r - (i_sb_r*Z_s_re  - i_sb_i*Z_s_im)  - v_sb_r
    eq_i_sc_re = e_c_r - (i_sc_r*Z_s_re  - i_sc_i*Z_s_im)  - v_sc_r
    eq_i_sn_re =       - (i_sn_r*Z_sn_re - i_sn_i*Z_sn_im) - v_sn_r
    eq_i_sa_im = e_a_i - (i_sa_r*Z_s_im  + i_sa_i*Z_s_re)  - v_sa_i
    eq_i_sb_im = e_b_i - (i_sb_r*Z_s_im  + i_sb_i*Z_s_re)  - v_sb_i
    eq_i_sc_im = e_c_i - (i_sc_r*Z_s_im  + i_sc_i*Z_s_re)  - v_sc_i
    eq_i_sn_im =       - (i_sn_r*Z_sn_im + i_sn_i*Z_sn_re) - v_sn_i

    g_list += [eq_i_sa_re]
    g_list += [eq_i_sb_re]
    g_list += [eq_i_sc_re]
    g_list += [eq_i_sn_re]
    g_list += [eq_i_sa_im]
    g_list += [eq_i_sb_im]
    g_list += [eq_i_sc_im]
    g_list += [eq_i_sn_im]

    y_ini_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r]
    y_ini_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i]
    y_run_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r]
    y_run_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i]

    i_s_r_by_ph = {'a':i_sa_r,'b':i_sb_r,'c':i_sc_r,'n':i_sn_r}
    i_s_i_by_ph = {'a':i_sa_i,'b':i_sb_i,'c':i_sc_i,'n':i_sn_i}
    for ph in ['a','b','c','n']:
        i_s_r = i_s_r_by_ph[ph]
        i_s_i = i_s_i_by_ph[ph]
        idx_r,idx_i = grid.node2idx(name,ph)
        grid.dae['g'] [idx_r] += -i_s_r
        grid.dae['g'] [idx_i] += -i_s_i
        i_s_m = (i_s_r**2 + i_s_i**2)**0.5
        h_dict.update({f'i_vsrc_{name}_{ph}_m':i_s_m})

    # injected apparent power per phase: s = v*conj(i)
    p_total = (v_sa_r*i_sa_r + v_sa_i*i_sa_i) \
            + (v_sb_r*i_sb_r + v_sb_i*i_sb_i) \
            + (v_sc_r*i_sc_r + v_sc_i*i_sc_i)
    q_total = (v_sa_i*i_sa_r - v_sa_r*i_sa_i) \
            + (v_sb_i*i_sb_r - v_sb_r*i_sb_i) \
            + (v_sc_i*i_sc_r - v_sc_r*i_sc_i)

    V_1 = 400/np.sqrt(3)

    u_ini_dict.update({f'e_ao_m_{name}':V_1,f'e_bo_m_{name}':V_1,f'e_co_m_{name}':V_1,f'e_no_m_{name}':0.0})
    u_run_dict.update({f'e_ao_m_{name}':V_1,f'e_bo_m_{name}':V_1,f'e_co_m_{name}':V_1,f'e_no_m_{name}':0.0})
    u_ini_dict.update({f'phi_{name}':0.0})
    u_run_dict.update({f'phi_{name}':0.0})

    X_s_N = 1e-4
    R_s_N = 0.0
    X_sn_N = 1e-4
    R_sn_N = 0.0
    X_ng_N = 1e-4
    R_ng_N = 1e-4
    
    if 'X_s' in data: X_s_N = data['X_s']  
    if 'R_s' in data: R_s_N = data['R_s']  
    if 'X_sn' in data: X_sn_N = data['X_sn'] 
    if 'R_sn' in data: R_sn_N = data['R_sn'] 

    if 'X_ng' in data: X_ng_N = data['X_ng'] 
    if 'R_ng' in data: R_ng_N = data['R_ng']  
        
    params_dict.update({f'X_s_{name}':X_s_N,f'R_s_{name}':R_s_N})
    params_dict.update({f'X_sn_{name}':X_sn_N,f'R_sn_{name}':R_sn_N})
    params_dict.update({f'X_ng_{name}':X_ng_N,f'R_ng_{name}':R_ng_N})

    grid.dae['xy_0_dict'].update({'omega':1.0})

    HS_coi  = 1.0
    omega_coi_i = 1.0

    grid.omega_coi_numerator += 1e6
    grid.omega_coi_denominator += 1e6

    grid.dae['h_dict'].update({f'p_{name}':p_total})
    grid.dae['h_dict'].update({f'q_{name}':q_total})

def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.uds import UdsBuilder
    import pydae.build_cffi as db

    grid = UdsBuilder('ac3ph4w_ideal.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()
    model.ini({'e_ao_m_A1':240,'e_bo_m_A1':240,'e_co_m_A1':240},'xy_0.json')
    model.report_u()
    model.report_y()
    model.report_z()



if __name__ == '__main__':

    #development()
    test()



