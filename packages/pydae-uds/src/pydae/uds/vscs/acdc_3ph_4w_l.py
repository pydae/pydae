
import numpy as np
import sympy as sym

def acdc_3ph_4w_l(grid,vsc_data):
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
        
    name_ac = vsc_data['ac_bus']
    name_dc = vsc_data['dc_bus']

    # inputs
    v_dc = sym.Symbol(f'v_dc_{name_ac}', real=True)
    m_a,m_b,m_c,m_n = sym.symbols(f'm_a_{name_ac},m_b_{name_ac},m_c_{name_ac},m_n_{name_ac}', real=True)
    phi = sym.Symbol(f'phi_{name_ac}', real=True)
    phi_a = sym.Symbol(f'phi_a_{name_ac}', real=True)
    phi_b = sym.Symbol(f'phi_b_{name_ac}', real=True)
    phi_c = sym.Symbol(f'phi_c_{name_ac}', real=True)
    phi_n = sym.Symbol(f'phi_n_{name_ac}', real=True)

    # parameters
    R_s,R_sn,R_ng = sym.symbols(f'R_{name_ac}_s,R_{name_ac}_sn,R_{name_ac}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name_ac}_s,X_{name_ac}_sn,X_{name_ac}_ng', real=True)
    
    # dynamical states
    De_ao_m,De_bo_m,De_co_m,De_no_m  = sym.symbols(f'De_ao_m_{name_ac},De_bo_m_{name_ac},De_co_m_{name_ac},De_no_m_{name_ac}', real=True)


    omega = sym.Symbol(f'omega_{name_ac}', real=True)
    
    # algebraic states

    ## ac side
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name_ac}_0_r,V_{name_ac}_1_r,V_{name_ac}_2_r,V_{name_ac}_3_r,v_{name_ac}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name_ac}_0_i,V_{name_ac}_1_i,V_{name_ac}_2_i,V_{name_ac}_3_i,v_{name_ac}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name_ac}_a_r,i_vsc_{name_ac}_b_r,i_vsc_{name_ac}_c_r,i_vsc_{name_ac}_n_r,i_vsc_{name_ac}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name_ac}_a_i,i_vsc_{name_ac}_b_i,i_vsc_{name_ac}_c_i,i_vsc_{name_ac}_n_i,i_vsc_{name_ac}_ng_i', real=True)
    p_dc,i_dc = sym.symbols(f'p_dc_{name_ac},i_dc_{name_ac}', real=True)

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

    s_sa = v_sa * sym.conjugate(i_sa)
    s_sb = v_sb * sym.conjugate(i_sb)
    s_sc = v_sc * sym.conjugate(i_sc)
    s_sn = v_sn * sym.conjugate(i_sc)



    sqrt6 = np.sqrt(6)
    e_ao_r = (m_a*v_dc/sqrt6)*sym.cos(phi_a + phi) 
    e_ao_i = (m_a*v_dc/sqrt6)*sym.sin(phi_a + phi) 
    e_bo_r = (m_b*v_dc/sqrt6)*sym.cos(phi_b + phi-2/3*np.pi) 
    e_bo_i = (m_b*v_dc/sqrt6)*sym.sin(phi_b + phi-2/3*np.pi) 
    e_co_r = (m_c*v_dc/sqrt6)*sym.cos(phi_c + phi-4/3*np.pi) 
    e_co_i = (m_c*v_dc/sqrt6)*sym.sin(phi_c + phi-4/3*np.pi) 
    e_no_r = (m_n*v_dc/sqrt6)*sym.cos(phi_n) 
    e_no_i = (m_n*v_dc/sqrt6)*sym.sin(phi_n) 

    e_ao_cplx = e_ao_r + 1j*e_ao_i
    e_bo_cplx = e_bo_r + 1j*e_bo_i
    e_co_cplx = e_co_r + 1j*e_co_i
    e_no_cplx = e_no_r + 1j*e_no_i

    s_ta = v_sa * sym.conjugate(i_sa)
    s_tb = v_sb * sym.conjugate(i_sb)
    s_tc = v_sc * sym.conjugate(i_sc)
    s_tn = v_sn * sym.conjugate(i_sc)

    p_ac_a = sym.re(s_ta)
    p_ac_b = sym.re(s_tb)
    p_ac_c = sym.re(s_tc)
    p_ac_n = sym.re(s_tn)

    v_san = v_sa - v_sn
    v_sbn = v_sb - v_sn
    v_scn = v_sc - v_sn

    eq_i_sa_cplx = v_og + e_ao_cplx - i_sa*Z_sa - v_sa   # v_sa = v_sag
    eq_i_sb_cplx = v_og + e_bo_cplx - i_sb*Z_sb - v_sb
    eq_i_sc_cplx = v_og + e_co_cplx - i_sc*Z_sc - v_sc
    eq_i_sn_cplx = v_og + e_no_cplx - i_sn*Z_sn - v_sn
    eq_v_og_cplx = i_sa + i_sb + i_sc + i_sn + v_og/Z_ng

    ### DC side
    v_sp_r,v_sm_r = sym.symbols(f'V_{name_dc}_0_r,V_{name_dc}_1_r', real=True)
    v_sp_i,v_sm_i = sym.symbols(f'V_{name_dc}_0_i,V_{name_dc}_1_i', real=True)

    A_loss,B_loss,C_loss = sym.symbols(f'A_loss_{name_ac},B_loss_{name_ac},C_loss_{name_ac}',real=True)
    i_rms_a = sym.sqrt(i_sa_r**2+i_sa_i**2+1e-6) 
    i_rms_b = sym.sqrt(i_sb_r**2+i_sb_i**2+1e-6) 
    i_rms_c = sym.sqrt(i_sc_r**2+i_sc_i**2+1e-6) 
    i_rms_n = sym.sqrt(i_sn_r**2+i_sn_i**2+1e-6) 

    p_loss_a = A_loss + B_loss*i_rms_a + C_loss*i_rms_a*i_rms_a
    p_loss_b = A_loss + B_loss*i_rms_b + C_loss*i_rms_b*i_rms_b
    p_loss_c = A_loss + B_loss*i_rms_c + C_loss*i_rms_c*i_rms_c
    p_loss_n = A_loss + B_loss*i_rms_n + C_loss*i_rms_n*i_rms_n

    p_vsc_loss = p_loss_a + p_loss_b + p_loss_c + p_loss_n
    p_ac = p_ac_a + p_ac_b + p_ac_c + p_ac_n

    eq_p_dc = -p_dc + p_ac + p_vsc_loss
    eq_i_dc = p_dc/v_dc - i_dc
 


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
    g_list += [eq_p_dc,eq_i_dc]

    y_ini_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r,v_og_r,p_dc,i_dc]
    y_ini_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i,v_og_i]
    y_run_list += [i_sa_r,i_sb_r,i_sc_r,i_sn_r,v_og_r,p_dc,i_dc]
    y_run_list += [i_sa_i,i_sb_i,i_sc_i,i_sn_i,v_og_i]


    for ph in ['a','b','c','n']:
        i_s_r = sym.Symbol(f'i_vsc_{name_ac}_{ph}_r', real=True)
        i_s_i = sym.Symbol(f'i_vsc_{name_ac}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(name_ac,ph)
        grid.dae['g'] [idx_r] += -i_s_r
        grid.dae['g'] [idx_i] += -i_s_i
        i_s = i_s_r + 1j*i_s_i
        i_s_m = np.abs(i_s)
        h_dict.update({f'i_vsc_{name_ac}_{ph}_m':i_s_m})

    idx_r = grid.node2idx(name_dc,'0')
    grid.dae['g'] [idx_r] += i_dc
    idx_r = grid.node2idx(name_dc,'1')
    grid.dae['g'] [idx_r] += -i_dc
    h_dict.update({f'i_vsc_{name_dc}':i_dc})

    #    V_1 = 400/np.sqrt(3)*np.exp(1j*np.deg2rad(0))
    # A_1toabc = np.array([1, alpha**2, alpha])
    #V_abc = V_1 * A_1toabc 
    #e_an_r,e_bn_r,e_cn_r = V_abc.real
    #e_an_i,e_bn_i,e_cn_i = V_abc.imag

    u_ini_dict.update({f'v_dc_{name_ac}':800.0})
    u_run_dict.update({f'v_dc_{name_ac}':800.0})
    m = 0.7071
    u_ini_dict.update({f'm_a_{name_ac}':m,f'm_b_{name_ac}':m,f'm_c_{name_ac}':m,f'm_n_{name_ac}':0.0})
    u_run_dict.update({f'm_a_{name_ac}':m,f'm_b_{name_ac}':m,f'm_c_{name_ac}':m,f'm_n_{name_ac}':0.0})

    u_ini_dict.update({f'phi_{name_ac}':0.0})
    u_ini_dict.update({f'phi_a_{name_ac}':0.0})
    u_ini_dict.update({f'phi_b_{name_ac}':0.0})
    u_ini_dict.update({f'phi_c_{name_ac}':0.0})
    u_ini_dict.update({f'phi_n_{name_ac}':0.0})

    u_run_dict.update({f'phi_{name_ac}':0.0})
    u_run_dict.update({f'phi_a_{name_ac}':0.0})
    u_run_dict.update({f'phi_b_{name_ac}':0.0})
    u_run_dict.update({f'phi_c_{name_ac}':0.0})
    u_run_dict.update({f'phi_n_{name_ac}':0.0})


    params_dict.update({f'X_{name_ac}_s':vsc_data['X'],f'R_{name_ac}_s':vsc_data['R']})
    params_dict.update({f'X_{name_ac}_sn':vsc_data['X_n'],f'R_{name_ac}_sn':vsc_data['R_n']})
    params_dict.update({f'X_{name_ac}_ng':vsc_data['X_ng'],f'R_{name_ac}_ng':vsc_data['R_ng']})
    
    params_dict.update({f'S_n_{name_ac}':vsc_data['S_n']})
   
    
    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])
    
    v_szpn = A_a0*v_sabc
    i_szpn = A_a0*i_sabc
    
    s_pos = 3*v_szpn[1]*sym.conjugate(i_szpn[1])
    s_neg = 3*v_szpn[2]*sym.conjugate(i_szpn[2])
    s_zer = 3*v_szpn[0]*sym.conjugate(i_szpn[0])

    h_dict.update({f'p_{name_ac}_pos':sym.re(s_pos),f'p_{name_ac}_neg':sym.re(s_neg),f'p_{name_ac}_zer':sym.re(s_zer)})
    h_dict.update({str(m_a):m_a,str(m_b):m_b,str(m_c):m_c,str(m_n):m_n})
    h_dict.update({str(phi):phi})

    S_n_num = vsc_data['S_n']
    U_n_num = vsc_data['U_n']
    I_n = S_n_num/(np.sqrt(3)*U_n_num)
    P_0 = 0.01*S_n_num
    A_loss_num = P_0/3
    P_cc = 0.01*S_n_num
    C_loss_num = P_cc/(I_n**2)/3

    params_dict.update({str(A_loss):A_loss_num})
    params_dict.update({str(B_loss):0.0})
    params_dict.update({str(C_loss):C_loss_num})


