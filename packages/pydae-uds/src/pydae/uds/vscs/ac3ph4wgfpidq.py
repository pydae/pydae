



def ac3ph4wgfpidq(grid,vsc_data):
    '''
    VSC with 3 phase and 4 wire working in open loop as a grid former.
    
    '''

    params_dict  = grid.dae['params']
    f_list = grid.dae['f']
    x_list = grid.dae['x']
    g_list = grid.dae['g'] 
    y_list = grid.dae['y'] 
    u_dict = grid.dae['u']
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

    # secondary frequency control
    omega_coi = sym.Symbol('omega_coi',real=True)
    xi_freq = sym.Symbol('xi_freq',real=True)
    K_agc = sym.Symbol('K_agc',real=True)
       
    name = vsc_data['bus']

    # inputs
    e_d_ref, e_q_ref = sym.symbols(f'e_d_{name}_ref,e_q_{name}_ref', real=True)
    omega_ref,p_c = sym.symbols(f'omega_{name}_ref,p_c_{name}', real=True)
 

    # parameters
    S_n,U_n,H,K_f,T_f,K_sec,K_delta  = sym.symbols(f'S_n_{name},U_n_{name},H_{name},K_f_{name},T_f_{name},K_sec_{name},K_delta_{name}', real=True)
    R_s,R_sn,R_ng = sym.symbols(f'R_{name}_s,R_{name}_sn,R_{name}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name}_s,X_{name}_sn,X_{name}_ng', real=True)
    T_e = sym.Symbol(f'T_e_{name}', real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)    
    T_p = sym.Symbol(f"T_p_{name}", real=True)   
    T_c = sym.Symbol(f"T_c_{name}", real=True)  
    T_w = sym.Symbol(f"T_w_{name}", real=True)   
    R_v = sym.Symbol(f"R_v_{name}", real=True)   
    X_v = sym.Symbol(f"X_v_{name}", real=True)   
    Droop = sym.Symbol(f"Droop_{name}", real=True)   
    
    # dynamical states
    phi = sym.Symbol(f'phi_{name}', real=True)
    xi_p = sym.Symbol(f'xi_p_{name}', real=True)
    p_ef = sym.Symbol(f'p_ef_{name}', real=True)
    De_ao_m,De_bo_m,De_co_m,De_no_m  = sym.symbols(f'De_ao_m_{name},De_bo_m_{name},De_co_m_{name},De_no_m_{name}', real=True)

    omega = sym.Symbol(f'omega_{name}', real=True)
    
    # algebraic states
    #e_an_i,e_bn_i,e_cn_i,e_ng_i = sym.symbols(f'e_{name}_an_i,e_{name}_bn_i,e_{name}_cn_i,e_{name}_ng_i', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'v_{name}_a_r,v_{name}_b_r,v_{name}_c_r,v_{name}_n_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'v_{name}_a_i,v_{name}_b_i,v_{name}_c_i,v_{name}_n_i,v_{name}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)

    omega_f = sym.Symbol(f'omega_f_{name}', real=True)

    p_ef  = sym.Symbol(f'p_ef_{name}', real=True)
    p_m = sym.Symbol(f'p_m_{name}', real=True) 
    p_c = sym.Symbol(f'p_c_{name}', real=True) 
    p_cf= sym.Symbol(f'p_cf_{name}', real=True)   
   
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
    
    i_pos = 1/3*(i_sa + alpha*i_sb + alpha**2*i_sc)

    I_b = S_n/(np.sqrt(3)*U_n)

    i_dq = np.sqrt(2)*i_pos*sym.exp(1j*(phi-np.pi/2))/(np.sqrt(2)*I_b)

    i_sd = sym.re(i_dq)
    i_sq = sym.im(i_dq)

    e_d = -i_sd*(R_v) + (X_v)*i_sq + e_d_ref  
    e_q = -i_sq*(R_v) - (X_v)*i_sd + e_q_ref

    e_dq = e_d + 1j*e_q

    v_tdq = U_n*sym.sqrt(1/3)*sym.sqrt(2)*e_dq*sym.exp(1j*(-phi-np.pi/2))

    v_ta = -v_tdq/sym.sqrt(2)
    v_tb = -v_tdq*alpha**2/sym.sqrt(2)
    v_tc = -v_tdq*alpha/sym.sqrt(2) 

    e_no_cplx = 0.0

    eq_i_sa_cplx = v_og + v_ta - i_sa*Z_sa - v_sa   # v_sa = v_sag
    eq_i_sb_cplx = v_og + v_tb - i_sb*Z_sb - v_sb
    eq_i_sc_cplx = v_og + v_tc - i_sc*Z_sc - v_sc
    eq_i_sn_cplx = v_og + e_no_cplx - i_sn*Z_sn - v_sn
    eq_v_og_cplx = i_sa + i_sb + i_sc + i_sn + v_og/Z_ng

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

    y_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r,v_og_r]
    y_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i,v_og_i]

    y_ini_str = [str(item) for item in y_list]

    for ph in ['a','b','c','n']:
        i_s_r = sym.Symbol(f'i_vsc_{name}_{ph}_r', real=True)
        i_s_i = sym.Symbol(f'i_vsc_{name}_{ph}_i', real=True)  
        g_list[y_ini_str.index(f'v_{name}_{ph}_r')] += i_s_r
        g_list[y_ini_str.index(f'v_{name}_{ph}_i')] += i_s_i
        i_s = i_s_r + 1j*i_s_i
        i_s_m = np.abs(i_s)
        h_dict.update({f'i_vsc_{name}_{ph}_m':i_s_m})
    
    h_dict.update({f'v_ta_{name}_r':sym.re(v_ta),f'v_tb_{name}_r':sym.re(v_tb),f'v_tc_{name}_r':sym.re(v_tc)})
    h_dict.update({f'v_ta_{name}_i':sym.im(v_ta),f'v_tb_{name}_i':sym.im(v_tb),f'v_tc_{name}_i':sym.im(v_tc)})
    h_dict.update({f'i_sd_{name}':i_sd,f'i_sq_{name}':i_sq})

        
    V_1 = 400/np.sqrt(3)

    u_dict.update({f'e_{name}_ao_m':V_1,f'e_{name}_bo_m':V_1,f'e_{name}_co_m':V_1,f'e_{name}_no_m':0.0})
    u_dict.update({f'phi_a_{name}':0.0})
    u_dict.update({f'phi_b_{name}':0.0})
    u_dict.update({f'phi_c_{name}':0.0})

    u_dict.update({f'p_c_{name}':0.0})
    u_dict.update({f'omega_{name}_ref':1.0})
    u_dict.update({f'{str(e_d_ref)}': 0.0})
    u_dict.update({f'{str(e_q_ref)}':-1.0})

    params_dict.update({f'X_v_{name}':vsc_data['X_v'],f'R_v_{name}':vsc_data['R_v']})
    params_dict.update({f'X_{name}_s':vsc_data['X'],f'R_{name}_s':vsc_data['R']})
    params_dict.update({f'X_{name}_sn':vsc_data['X_n'],f'R_{name}_sn':vsc_data['R_n']})
    params_dict.update({f'X_{name}_ng':vsc_data['X_ng'],f'R_{name}_ng':vsc_data['R_ng']})
    
    params_dict.update({f'S_n_{name}':vsc_data['S_n'],f'U_n_{name}':vsc_data['U_n']})
    params_dict.update({f'T_e_{name}':vsc_data['T_e']})
    params_dict.update({f'T_c_{name}':vsc_data['T_c']})
    params_dict.update({f'Droop_{name}':vsc_data['Droop']})
    params_dict.update({f'K_p_{name}':vsc_data['K_p']})
    params_dict.update({f'T_p_{name}':vsc_data['T_p']})
    params_dict.update({f'T_w_{name}':vsc_data['T_w']})
    params_dict.update({f'K_sec_{name}':vsc_data['K_sec']})
    params_dict.update({f'K_delta_{name}':vsc_data['K_delta']})
    
    # VSG PI 
    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])
    
    v_szpn = A_a0*v_sabc
    i_szpn = A_a0*i_sabc
    
    s_pos = 3*v_szpn[1]*sym.conjugate(i_szpn[1])
    s_neg = 3*v_szpn[2]*sym.conjugate(i_szpn[2])
    s_zer = 3*v_szpn[0]*sym.conjugate(i_szpn[0])
    
    p_pos = sym.re(s_pos)

    p_agc = K_agc*xi_freq
    p_r = K_sec*p_agc
    epsilon_p = p_m - p_ef

    ## dynamic equations
    dphi   = 2*np.pi*50*(omega - omega_coi) - K_delta*phi
    dxi_p = epsilon_p
    dp_ef = 1/T_e*(p_pos/S_n - p_ef)  # simulink: p_s_fil, T_e_f
    dp_cf = 1/T_c*(p_c - p_cf)
    domega_f = 1/T_w*(omega - omega_f)
    dDe_ao_m = 1/T_v*(v_ra*V_1 - De_ao_m)
    dDe_bo_m = 1/T_v*(v_rb*V_1 - De_bo_m)
    dDe_co_m = 1/T_v*(v_rc*V_1 - De_co_m)
    dDe_no_m = 1/T_v*(v_rn*V_1 - De_co_m)


    p_pri = 1/Droop*(omega_f - omega_ref)

    ## algebraic equations   
    g_omega = -omega + K_p*(epsilon_p + xi_p/T_p) + 1
    g_p_m  = -p_m + p_cf + p_r - p_pri

    g_list += [g_omega, g_p_m] 
    y_list += [  omega,   p_m] 

    f_list += [dphi,dxi_p,dp_ef,dp_cf,domega_f,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m]
    x_list += [ phi, xi_p, p_ef, p_cf, omega_f, De_ao_m, De_bo_m, De_co_m, De_no_m]
    
    h_dict.update({f'p_{name}_pos':sym.re(s_pos),f'p_{name}_neg':sym.re(s_neg),f'p_{name}_zer':sym.re(s_zer)})
    h_dict.update({str(p_c):p_c,str(omega_ref):omega_ref})
    h_dict.update({f'p_pri_{name}':p_pri})


    # COI computation
    HS_coi  = S_n
    omega_coi_i = S_n*omega_f

    grid.omega_coi_h_i += omega_coi_i
    grid.hs_total += HS_coi



    