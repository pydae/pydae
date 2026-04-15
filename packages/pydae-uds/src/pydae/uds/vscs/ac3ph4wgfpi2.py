
import numpy as np
import sympy as sym

def ac3ph4wgfpi2(grid,vsc_data):
    '''
    VSC with 3 phase and 4 wire working in open loop as a grid former.
    
    '''

    params_dict  = grid.dae['params_dict']
    f_list = grid.dae['f']
    x_list = grid.dae['x']
    g_list = grid.dae['g'] 
    y_ini_list = grid.dae['y_ini'] 
    u_ini_dict = grid.dae['u_ini_dict']
    y_run_list = grid.dae['y_run'] 
    u_run_dict = grid.dae['u_run_dict']
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

    #vscs = [
    #    {'bus':'B1','S_n':100e3,'R':0.01,'X':0.1,'R_n':0.01,'X_n':0.1,'R_ng':0.01,'X_ng':3.0,'K_f':0.1,'T_f':1.0,'K_sec':0.5,'K_delta':0.001},
    #    ]

    #for vsc in vsc_data:
        
    name = vsc_data['bus']

    # inputs
    e_ao_m,e_bo_m,e_co_m,e_no_m = sym.symbols(f'e_ao_m_{name},e_bo_m_{name},e_co_m_{name},e_no_m_{name}', real=True)
    omega_ref,p_c = sym.symbols(f'omega_{name}_ref,p_c_{name}', real=True)
    phi_a = sym.Symbol(f'phi_a_{name}', real=True)
    phi_b = sym.Symbol(f'phi_b_{name}', real=True)
    phi_c = sym.Symbol(f'phi_c_{name}', real=True)
    v_ra = sym.Symbol(f'v_ra_{name}', real=True)
    v_rb = sym.Symbol(f'v_rb_{name}', real=True)
    v_rc = sym.Symbol(f'v_rc_{name}', real=True)
    v_rn = sym.Symbol(f'v_rn_{name}', real=True)    

    # parameters
    S_n,H,K_f,T_f,K_agc,K_delta  = sym.symbols(f'S_n_{name},H_{name},K_f_{name},T_f_{name},K_agc_{name},K_delta_{name}', real=True)
    R_s,R_sn,R_ng = sym.symbols(f'R_{name}_s,R_{name}_sn,R_{name}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name}_s,X_{name}_sn,X_{name}_ng', real=True)
    T_e = sym.Symbol(f'T_e_{name}', real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)    
    T_p = sym.Symbol(f"T_p_{name}", real=True)   
    T_c = sym.Symbol(f"T_c_{name}", real=True)   
    T_v = sym.Symbol(f"T_v_{name}", real=True)   

    Droop = sym.Symbol(f"Droop_{name}", real=True)   
    
    # dynamical states
    phi = sym.Symbol(f'phi_{name}', real=True)
    xi_p = sym.Symbol(f'xi_p_{name}', real=True)
    p_ef = sym.Symbol(f'p_ef_{name}', real=True)
    De_ao_m,De_bo_m,De_co_m,De_no_m  = sym.symbols(f'De_ao_m_{name},De_bo_m_{name},De_co_m_{name},De_no_m_{name}', real=True)





    omega = sym.Symbol(f'omega_{name}', real=True)
    
    # algebraic states
    #e_an_i,e_bn_i,e_cn_i,e_ng_i = sym.symbols(f'e_{name}_an_i,e_{name}_bn_i,e_{name}_cn_i,e_{name}_ng_i', real=True)
    #v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'v_{name}_a_r,v_{name}_b_r,v_{name}_c_r,v_{name}_n_r,v_{name}_o_r', real=True)
    #v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'v_{name}_a_i,v_{name}_b_i,v_{name}_c_i,v_{name}_n_i,v_{name}_o_i', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    v_mn_r,v_mn_i = sym.symbols(f'v_{name}_mn_r,v_{name}_mn_i', real=True)

    omega = sym.Symbol(f'omega_{name}', real=True)
    p_ef  = sym.Symbol(f'p_ef_{name}', real=True)
    p_m = sym.Symbol(f'p_m_{name}', real=True) 
    p_c = sym.Symbol(f'p_c_{name}', real=True) 
    p_cf= sym.Symbol(f'p_cf_{name}', real=True)   

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

    v_sa = v_sa_r + 1j*v_sa_i
    v_sb = v_sb_r + 1j*v_sb_i
    v_sc = v_sc_r + 1j*v_sc_i
    v_sn = v_sn_r + 1j*v_sn_i
    v_og = v_og_r + 1j*v_og_i
    
    e_ao_r = (e_ao_m+De_ao_m)*sym.cos(phi_a + phi) 
    e_ao_i = (e_ao_m+De_ao_m)*sym.sin(phi_a + phi) 
    e_bo_r = (e_bo_m+De_bo_m)*sym.cos(phi_b + phi-2/3*np.pi) 
    e_bo_i = (e_bo_m+De_bo_m)*sym.sin(phi_b + phi-2/3*np.pi) 
    e_co_r = (e_co_m+De_co_m)*sym.cos(phi_c + phi-4/3*np.pi) 
    e_co_i = (e_co_m+De_co_m)*sym.sin(phi_c + phi-4/3*np.pi) 

    e_ao_cplx = e_ao_r + 1j*e_ao_i
    e_bo_cplx = e_bo_r + 1j*e_bo_i
    e_co_cplx = e_co_r + 1j*e_co_i
    e_no_cplx = 0.0

    v_san = v_sa - v_sn
    v_sbn = v_sb - v_sn
    v_scn = v_sc - v_sn

    eq_i_sa_cplx = v_og + e_ao_cplx - i_sa*Z_sa - v_sa   # v_sa = v_sag
    eq_i_sb_cplx = v_og + e_bo_cplx - i_sb*Z_sb - v_sb
    eq_i_sc_cplx = v_og + e_co_cplx - i_sc*Z_sc - v_sc
    eq_i_sn_cplx = v_og + e_no_cplx - i_sn*Z_sn - v_sn
    eq_v_og_cplx = i_sa + i_sb + i_sc + i_sn + v_og/Z_ng
    #eq_i_sn_cplx = e_ng_cplx - i_sn*Z_sn - v_ng
    #eq_i_ng_cplx = i_ng + i_sa + i_sb + i_sc + i_sn
    #eq_e_ng_cplx  = -e_ng_cplx  + i_ng*Z_ng

    g_list += [sym.re(eq_i_sa_cplx)] 
    g_list += [sym.re(eq_i_sb_cplx)] 
    g_list += [sym.re(eq_i_sc_cplx)] 
    g_list += [sym.re(eq_i_sn_cplx)] 
    g_list += [sym.re(eq_v_og_cplx)] 
    g_list += [sym.im(eq_i_sa_cplx)] 
    g_list += [sym.im(eq_i_sb_cplx)] 
    g_list += [sym.im(eq_i_sc_cplx)] 
    g_list += [sym.im(eq_i_sn_cplx)] 
    g_list += [sym.im(eq_v_og_cplx)] 

    y_ini_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r,v_og_r]
    y_ini_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i,v_og_i]
    y_run_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r,v_og_r]
    y_run_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i,v_og_i]

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
    V_b = V_1
    #    V_1 = 400/np.sqrt(3)*np.exp(1j*np.deg2rad(0))
    # A_1toabc = np.array([1, alpha**2, alpha])
    #V_abc = V_1 * A_1toabc 
    #e_an_r,e_bn_r,e_cn_r = V_abc.real
    #e_an_i,e_bn_i,e_cn_i = V_abc.imag

    u_ini_dict.update({f'e_ao_m_{name}':V_1,f'e_bo_m_{name}':V_1,f'e_co_m_{name}':V_1,f'e_no_m_{name}':0.0})
    u_ini_dict.update({f'v_ra_{name}':0.0,f'v_rb_{name}':0.0,f'v_rc_{name}':0.0,f'v_rn_{name}':0.0})
    u_run_dict.update({f'e_ao_m_{name}':V_1,f'e_bo_m_{name}':V_1,f'e_co_m_{name}':V_1,f'e_no_m_{name}':0.0})
    u_run_dict.update({f'v_ra_{name}':0.0,f'v_rb_{name}':0.0,f'v_rc_{name}':0.0,f'v_rn_{name}':0.0})

    #u_dict.update({f'phi_{name}':0.0})
    u_ini_dict.update({f'phi_a_{name}':0.0})
    u_ini_dict.update({f'phi_b_{name}':0.0})
    u_ini_dict.update({f'phi_c_{name}':0.0})

    u_ini_dict.update({f'p_c_{name}':0.0})
    u_ini_dict.update({f'omega_{name}_ref':1.0})

    u_run_dict.update({f'phi_a_{name}':0.0})
    u_run_dict.update({f'phi_b_{name}':0.0})
    u_run_dict.update({f'phi_c_{name}':0.0})

    u_run_dict.update({f'p_c_{name}':0.0})
    u_run_dict.update({f'omega_{name}_ref':1.0})

    #for ph in ['a','b','c','n']:
    #    u_dict.pop(f'i_{name}_{ph}_r')
    #    u_dict.pop(f'i_{name}_{ph}_i')

    params_dict.update({f'X_{name}_s':vsc_data['X'],f'R_{name}_s':vsc_data['R']})
    params_dict.update({f'X_{name}_sn':vsc_data['X_n'],f'R_{name}_sn':vsc_data['R_n']})
    params_dict.update({f'X_{name}_ng':vsc_data['X_ng'],f'R_{name}_ng':vsc_data['R_ng']})
    
    params_dict.update({f'S_n_{name}':vsc_data['S_n']})
    params_dict.update({f'T_e_{name}':vsc_data['T_e']})
    params_dict.update({f'T_c_{name}':vsc_data['T_c']})
    params_dict.update({f'T_v_{name}':vsc_data['T_v']})
    params_dict.update({f'Droop_{name}':vsc_data['Droop']})
    params_dict.update({f'K_p_{name}':vsc_data['K_p']})
    params_dict.update({f'T_p_{name}':vsc_data['T_p']})
    params_dict.update({f'K_agc_{name}':vsc_data['K_agc']})
    params_dict.update({f'K_delta_{name}':vsc_data['K_delta']})
    
    
    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])
    

    v_szpn = A_a0*v_sabc
    i_szpn = A_a0*i_sabc
    
    s_pos = 3*v_szpn[1]*sym.conjugate(i_szpn[1])
    s_neg = 3*v_szpn[2]*sym.conjugate(i_szpn[2])
    s_zer = 3*v_szpn[0]*sym.conjugate(i_szpn[0])
    
    p_pos = sym.re(s_pos)

    p_r = K_agc*xi_freq
    epsilon_p = p_m - p_ef

    dphi   = 2*np.pi*50*(omega - omega_coi) - K_delta*phi
    dxi_p = epsilon_p
    dp_ef = 1/T_e*(p_pos/S_n - p_ef)
    dp_cf = 1/T_c*(p_c - p_cf)
   
    dDe_ao_m = 1/T_v*(v_ra*V_b - De_ao_m)
    dDe_bo_m = 1/T_v*(v_rb*V_b - De_bo_m)
    dDe_co_m = 1/T_v*(v_rc*V_b - De_co_m)
    dDe_no_m = 1/T_v*(v_rn*V_b - De_no_m)


    # algebraic equations   
    g_omega = -omega + K_p*(epsilon_p + xi_p/T_p) + 1
    g_p_m  = -p_m + p_cf + p_r - 1/Droop*(omega - omega_ref)

    g_list += [g_omega, g_p_m] 
    y_ini_list += [  omega,   p_m] 
    y_run_list += [  omega,   p_m] 
 
    f_list += [dphi,dxi_p,dp_ef,dp_cf,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m] #,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m]
    x_list += [ phi, xi_p, p_ef, p_cf, De_ao_m, De_bo_m, De_co_m, De_no_m] #, De_ao_m, De_bo_m, De_co_m, De_no_m]
    
    h_dict.update({f'p_{name}_pos':sym.re(s_pos),f'p_{name}_neg':sym.re(s_neg),f'p_{name}_zer':sym.re(s_zer)})
    h_dict.update({str(e_ao_m):e_ao_m,str(e_bo_m):e_bo_m,str(e_co_m):e_co_m})
    h_dict.update({str(v_ra):v_ra,str(v_rb):v_rb,str(v_rc):v_rc})

    n2a = {0:'a',1:'b',2:'c'}
    for ph in [0,1,2]:
        s = v_sabc[ph]*sym.conjugate(i_sabc[ph])
        p = sym.re(s)
        q = sym.im(s)
        h_dict.update({f'p_vsc_{name}_{n2a[ph]}':p,f'q_vsc_{name}_{n2a[ph]}':q})

    h_dict.update({str(p_c):p_c,str(omega_ref):omega_ref})
    h_dict.update({str(phi):phi})

    grid.dae['xy_0_dict'].update({'omega':1.0})

    HS_coi  = S_n
    omega_coi_i = S_n*omega

    grid.omega_coi_numerator += omega_coi_i
    grid.omega_coi_denominator += HS_coi
