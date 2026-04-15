
import numpy as np
import sympy as sym

def ac3ph4w_ideal(grid,vsc_data):
    '''
    VSC with 3 phase and 4 wire working in open loop as a grid former.
    
    '''

    params_dict  = grid.dae['params_dict']
    g_list = grid.dae['g'] 
    y_ini_list = grid.dae['y_ini'] 
    u_ini_dict = grid.dae['u_ini_dict']
    y_run_list = grid.dae['y_run'] 
    u_run_dict = grid.dae['u_run_dict']
    h_dict = grid.dae['h_dict']

    #vscs = [
    #    {'bus':'B1','S_n':100e3,'R':0.01,'X':0.1,'R_n':0.01,'X_n':0.1,'R_ng':0.01,'X_ng':3.0,'K_f':0.1,'T_f':1.0,'K_sec':0.5,'K_delta':0.001},
    #    ]

    #for vsc in vsc_data:
        
    name = vsc_data['bus']

    # inputs
    e_ao_m,e_bo_m,e_co_m,e_no_m = sym.symbols(f'e_ao_m_{name},e_bo_m_{name},e_co_m_{name},e_no_m_{name}', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)
    phi =  sym.Symbol(f'phi_{name}', real=True)

    # parameters
    R_s,R_sn,R_ng = sym.symbols(f'R_{name}_s,R_{name}_sn,R_{name}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name}_s,X_{name}_sn,X_{name}_ng', real=True)

    # dynamical states

    
    # algebraic states
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    
    Z_sa = R_s + 1j*X_s
    Z_sb = R_s + 1j*X_s
    Z_sc = R_s + 1j*X_s
    Z_sn = R_sn + 1j*X_sn
    Z_ng = R_ng + 1j*X_ng

    i_sa = i_sa_r + 1j*i_sa_i
    i_sb = i_sb_r + 1j*i_sb_i
    i_sc = i_sc_r + 1j*i_sc_i
    i_sn = i_sn_r + 1j*i_sn_i

    v_sa = v_sa_r + 1j*v_sa_i
    v_sb = v_sb_r + 1j*v_sb_i
    v_sc = v_sc_r + 1j*v_sc_i
    v_sn = v_sn_r + 1j*v_sn_i
    
    e_a_r = (e_ao_m)*sym.cos(phi) 
    e_a_i = (e_ao_m)*sym.sin(phi) 
    e_b_r = (e_bo_m)*sym.cos(phi-2/3*np.pi) 
    e_b_i = (e_bo_m)*sym.sin(phi-2/3*np.pi) 
    e_c_r = (e_co_m)*sym.cos(phi-4/3*np.pi) 
    e_c_i = (e_co_m)*sym.sin(phi-4/3*np.pi) 

    e_a_cplx = e_a_r + 1j*e_a_i
    e_b_cplx = e_b_r + 1j*e_b_i
    e_c_cplx = e_c_r + 1j*e_c_i
    e_n_cplx = 0.0

    eq_i_sa_cplx = e_a_cplx - i_sa*Z_sa - v_sa   # v_sa = v_sag
    eq_i_sb_cplx = e_b_cplx - i_sb*Z_sb - v_sb
    eq_i_sc_cplx = e_c_cplx - i_sc*Z_sc - v_sc
    eq_i_sn_cplx = e_n_cplx - i_sn*Z_sn - v_sn

    g_list += [sym.re(eq_i_sa_cplx)] 
    g_list += [sym.re(eq_i_sb_cplx)] 
    g_list += [sym.re(eq_i_sc_cplx)] 
    g_list += [sym.re(eq_i_sn_cplx)] 
    g_list += [sym.im(eq_i_sa_cplx)] 
    g_list += [sym.im(eq_i_sb_cplx)] 
    g_list += [sym.im(eq_i_sc_cplx)] 
    g_list += [sym.im(eq_i_sn_cplx)] 

    y_ini_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r]
    y_ini_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i]
    y_run_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r]
    y_run_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i]

    #y_ini_str = [str(item) for item in y_list]

    for ph in ['a','b','c','n']:
        i_s_r = sym.Symbol(f'i_vsc_{name}_{ph}_r', real=True)
        i_s_i = sym.Symbol(f'i_vsc_{name}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(name,ph)
        grid.dae['g'] [idx_r] += -i_s_r
        grid.dae['g'] [idx_i] += -i_s_i
        i_s = i_s_r + 1j*i_s_i
        i_s_m = np.abs(i_s)
        h_dict.update({f'i_vsc_{name}_{ph}_m':i_s_m})

    V_1 = 400/np.sqrt(3)

    u_ini_dict.update({f'e_ao_m_{name}':V_1,f'e_bo_m_{name}':V_1,f'e_co_m_{name}':V_1,f'e_no_m_{name}':0.0})
    u_run_dict.update({f'e_ao_m_{name}':V_1,f'e_bo_m_{name}':V_1,f'e_co_m_{name}':V_1,f'e_no_m_{name}':0.0})
    u_ini_dict.update({f'phi_{name}':0.0})
    u_run_dict.update({f'phi_{name}':0.0})



    params_dict.update({f'X_{name}_s':vsc_data['X'],f'R_{name}_s':vsc_data['R']})
    params_dict.update({f'X_{name}_sn':vsc_data['X_n'],f'R_{name}_sn':vsc_data['R_n']})
    params_dict.update({f'X_{name}_ng':vsc_data['X_ng'],f'R_{name}_ng':vsc_data['R_ng']})

    grid.dae['xy_0_dict'].update({'omega':1.0})

    HS_coi  = 1.0
    omega_coi_i = 1.0

    grid.omega_coi_numerator += omega_coi_i
    grid.omega_coi_denominator += HS_coi
