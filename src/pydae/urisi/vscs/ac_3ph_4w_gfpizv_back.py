
import numpy as np
import sympy as sym
import pytest 

def ac_3ph_4w_gfpizv(grid,data):

    ac_3ph_4w(grid,data)
    gfpizv(grid,data)


def ac_3ph_4w(grid,data):
    '''
    VSC with 3 phase and 4 wire with a PI-VSG with PFR and Q control.
         
    '''
    name = data['bus']


    params_dict  = grid.dae['params_dict']
    f_list = grid.dae['f']
    x_list = grid.dae['x']
    g_list = grid.dae['g'] 
    y_ini_list = grid.dae['y_ini'] 
    u_ini_dict = grid.dae['u_ini_dict']
    y_run_list = grid.dae['y_run'] 
    u_run_dict = grid.dae['u_run_dict']
    h_dict = grid.dae['h_dict']


    # VSC

    ## VSC parameters
    S_n,U_n  = sym.symbols(f'S_n_{name},U_n_{name}', real=True)
    R_s,R_sn,R_ng = sym.symbols(f'R_{name}_s,R_{name}_sn,R_{name}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name}_s,X_{name}_sn,X_{name}_ng', real=True)
    A_loss = sym.symbols(f'A_loss_{name}',real=True)
    B_loss = sym.symbols(f'B_loss_{name}',real=True)
    C_loss = sym.symbols(f'C_loss_{name}',real=True)

    ## VSC inputs
    v_tao_r, v_tao_i = sym.symbols(f'v_tao_r_{name},v_tao_i_{name}', real=True)
    v_tbo_r, v_tbo_i = sym.symbols(f'v_tbo_r_{name},v_tbo_i_{name}', real=True)
    v_tco_r, v_tco_i = sym.symbols(f'v_tco_r_{name},v_tco_i_{name}', real=True)
    v_tno_r, v_tno_i = sym.symbols(f'v_tno_r_{name},v_tno_i_{name}', real=True)


    # algebraic states
    #e_an_i,e_bn_i,e_cn_i,e_ng_i = sym.symbols(f'e_{name}_an_i,e_{name}_bn_i,e_{name}_cn_i,e_{name}_ng_i', real=True)
    #v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'v_{name}_a_r,v_{name}_b_r,v_{name}_c_r,v_{name}_n_r,v_{name}_o_r', real=True)
    #v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'v_{name}_a_i,v_{name}_b_i,v_{name}_c_i,v_{name}_n_i,v_{name}_o_i', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    v_tao_r,v_tbo_r,v_tco_r,v_tno_r = sym.symbols(f'v_tao_r_{name},v_tbo_r_{name},v_tco_r_{name},v_tno_r_{name}', real=True)
    v_tao_i,v_tbo_i,v_tco_i,v_tno_i = sym.symbols(f'v_tao_i_{name},v_tbo_i_{name},v_tco_i_{name},v_tno_i_{name}', real=True)
    p_dc = sym.Symbol(f'p_dc_{name}', real=True)

    #e_om_r,e_om_i = sym.symbols(f'e_{name}_om_r,e_{name}_om_i', real=True)

    ## VSC impedances
    Z_sa = R_s + 1j*X_s 
    Z_sb = R_s + 1j*X_s
    Z_sc = R_s + 1j*X_s
    Z_sn = R_sn + 1j*X_sn
    Z_ng = R_ng + 1j*X_ng

    ## POI currents
    i_sa = i_sa_r + 1j*i_sa_i
    i_sb = i_sb_r + 1j*i_sb_i
    i_sc = i_sc_r + 1j*i_sc_i
    i_sn = i_sn_r + 1j*i_sn_i

    ## POI voltages
    v_sa = v_sa_r + 1j*v_sa_i
    v_sb = v_sb_r + 1j*v_sb_i
    v_sc = v_sc_r + 1j*v_sc_i
    v_sn = v_sn_r + 1j*v_sn_i
    v_og = v_og_r + 1j*v_og_i

    ## VSC AC side voltages
    v_tao = v_tao_r + 1j*v_tao_i
    v_tbo = v_tbo_r + 1j*v_tbo_i
    v_tco = v_tco_r + 1j*v_tco_i
    v_tno = v_tno_r + 1j*v_tno_i
    
    s_ta = v_tao * sym.conjugate(i_sa)
    s_tb = v_tbo * sym.conjugate(i_sb)
    s_tc = v_tco * sym.conjugate(i_sc)
    s_tn = v_tno * sym.conjugate(i_sc)

    p_ac_a = sym.re(s_ta)
    p_ac_b = sym.re(s_tb)
    p_ac_c = sym.re(s_tc)
    p_ac_n = sym.re(s_tn)

    ## VSC RMS currents
    i_rms_a = sym.sqrt(i_sa_r**2+i_sa_i**2+1e-6) 
    i_rms_b = sym.sqrt(i_sb_r**2+i_sb_i**2+1e-6) 
    i_rms_c = sym.sqrt(i_sc_r**2+i_sc_i**2+1e-6) 
    i_rms_n = sym.sqrt(i_sn_r**2+i_sn_i**2+1e-6) 

    ## VSC power losses
    p_loss_a = A_loss*i_rms_a*i_rms_a + B_loss*i_rms_a + C_loss
    p_loss_b = A_loss*i_rms_b*i_rms_b + B_loss*i_rms_b + C_loss
    p_loss_c = A_loss*i_rms_c*i_rms_c + B_loss*i_rms_c + C_loss
    p_loss_n = A_loss*i_rms_n*i_rms_n + B_loss*i_rms_n + C_loss

    p_vsc_loss = p_loss_a + p_loss_b + p_loss_c + p_loss_n
    p_ac = p_ac_a + p_ac_b + p_ac_c + p_ac_n

    eq_i_sa_cplx = v_og + v_tao - i_sa*Z_sa - v_sa   # v_sa = v_sag
    eq_i_sb_cplx = v_og + v_tbo - i_sb*Z_sb - v_sb
    eq_i_sc_cplx = v_og + v_tco - i_sc*Z_sc - v_sc
    eq_i_sn_cplx = v_og + v_tno - i_sn*Z_sn - v_sn
    eq_v_og_cplx = i_sa + i_sb + i_sc + i_sn + v_og/Z_ng
    eq_p_dc = -p_dc + p_ac + p_vsc_loss

    grid.dae['g'] += [sym.re(eq_i_sa_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sb_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sc_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sn_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_og_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sa_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sb_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sc_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sn_cplx)] 
    grid.dae['g'] += [sym.im(eq_v_og_cplx)] 
    grid.dae['g'] += [eq_p_dc] 

    grid.dae['y_ini'] += [i_sa_r,i_sb_r,i_sc_r,i_sn_r,v_og_r]
    grid.dae['y_ini'] += [i_sa_i,i_sb_i,i_sc_i,i_sn_i,v_og_i]
    grid.dae['y_ini'] += [p_dc]
    grid.dae['y_run'] += [i_sa_r,i_sb_r,i_sc_r,i_sn_r,v_og_r]
    grid.dae['y_run'] += [i_sa_i,i_sb_i,i_sc_i,i_sn_i,v_og_i]
    grid.dae['y_run'] += [p_dc]

    for ph in ['a','b','c','n']:
        i_s_r = sym.Symbol(f'i_vsc_{name}_{ph}_r', real=True)
        i_s_i = sym.Symbol(f'i_vsc_{name}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(name,ph)
        grid.dae['g'] [idx_r] += -i_s_r
        grid.dae['g'] [idx_i] += -i_s_i
        i_s = i_s_r + 1j*i_s_i
        i_s_m = np.abs(i_s)
        grid.dae['h_dict'].update({f'i_vsc_{name}_{ph}_m':i_s_m})

    ## outputs
    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])
    n2a = {0:'a',1:'b',2:'c'}
    for ph in [0,1,2]:
        s = v_sabc[ph]*sym.conjugate(i_sabc[ph])
        p = sym.re(s)
        q = sym.im(s)
        grid.dae['h_dict'].update({f'p_vsc_{name}_{n2a[ph]}':p,f'q_vsc_{name}_{n2a[ph]}':q})

    ## parameters default values
    grid.dae['params_dict'].update({f'X_{name}_s':data['X'],f'R_{name}_s':data['R']})
    grid.dae['params_dict'].update({f'X_{name}_sn':data['X_n'],f'R_{name}_sn':data['R_n']})
    grid.dae['params_dict'].update({f'X_{name}_ng':data['X_ng'],f'R_{name}_ng':data['R_ng']})

    grid.dae['xy_0_dict'].update({'omega':1.0})

    S_n_N = data['S_n']
    U_n_N = data['U_n']
    V_n_N = U_n_N/np.sqrt(3)
    phi_N = 0.0
    if 'phi' in data: phi_N = data['phi']

    v_tao_N = V_n_N*np.exp(1j*phi_N)
    v_tbo_N = V_n_N*np.exp(1j*(phi_N - np.pi*2.0/3.0))
    v_tco_N = V_n_N*np.exp(1j*(phi_N - np.pi*4.0/3.0))
    v_tno_N = 0.0

    I_n = S_n_N/(np.sqrt(3)*U_n_N)
    P_0 = 0.01*S_n_N
    C_loss_N = P_0/3
    P_cc = 0.01*S_n_N
    A_loss_N = P_cc/(I_n**2)/3
    B_loss_N = 1.0

    if 'A_loss' in data:
        A_loss_N = data['A_loss']
        B_loss_N = data['B_loss']
        C_loss_N = data['C_loss']

    params_dict.update({str(A_loss):A_loss_N})
    params_dict.update({str(B_loss):B_loss_N})
    params_dict.update({str(C_loss):C_loss_N})

    grid.dae['u_ini_dict'].update({str(v_tao_r):v_tao_N.real, str(v_tao_i):v_tao_N.imag})
    grid.dae['u_ini_dict'].update({str(v_tbo_r):v_tbo_N.real, str(v_tbo_i):v_tbo_N.imag})
    grid.dae['u_ini_dict'].update({str(v_tco_r):v_tco_N.real, str(v_tco_i):v_tco_N.imag})
    grid.dae['u_ini_dict'].update({str(v_tno_r):v_tno_N.real, str(v_tno_i):v_tno_N.imag})
    grid.dae['u_run_dict'].update({str(v_tao_r):v_tao_N.real, str(v_tao_i):v_tao_N.imag})
    grid.dae['u_run_dict'].update({str(v_tbo_r):v_tbo_N.real, str(v_tbo_i):v_tbo_N.imag})
    grid.dae['u_run_dict'].update({str(v_tco_r):v_tco_N.real, str(v_tco_i):v_tco_N.imag})
    grid.dae['u_run_dict'].update({str(v_tno_r):v_tno_N.real, str(v_tno_i):v_tno_N.imag})


def gfpizv(grid,data):

#######################################################################################
    # CTRL
    name = data['bus']

    # inputs
    p_c = sym.Symbol(f'p_c_{name}', real=True) 
    xi_freq = sym.Symbol('xi_freq',real=True)
    omega_coi = sym.Symbol('omega_coi',real=True)

    # parameters
    S_n,U_n = sym.symbols(f'S_n_{name},U_n_{name}', real=True)
    T_e = sym.Symbol(f'H_{name}', real=True)
    K_f = sym.Symbol(f'K_f_{name}', real=True)
    T_f = sym.Symbol(f'T_f_{name}', real=True)
    K_agc = sym.Symbol(f'K_agc_{name}', real=True)
    T_e = sym.Symbol(f'T_e_{name}', real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)    
    T_p = sym.Symbol(f"T_p_{name}", real=True)   
    T_c = sym.Symbol(f"T_c_{name}", real=True)   
    T_v = sym.Symbol(f"T_v_{name}", real=True)   
    K_qp = sym.Symbol(f"K_qp_{name}", real=True)   
    K_qi = sym.Symbol(f"K_qi_{name}", real=True) 
    Droop = sym.Symbol(f"Droop_{name}", real=True)   
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True) 

    ## inputs
    omega_ref = sym.Symbol(f'omega_{name}_ref', real=True)
    p_c = sym.Symbol(f'p_c_{name}', real=True)  
    q_ref = sym.Symbol(f'q_ref_{name}', real=True)    
    e_ao_m,e_bo_m,e_co_m,e_no_m = sym.symbols(f'e_ao_m_{name},e_bo_m_{name},e_co_m_{name},e_no_m_{name}', real=True)
    phi_a = sym.Symbol(f'phi_a_{name}', real=True)
    phi_b = sym.Symbol(f'phi_b_{name}', real=True)
    phi_c = sym.Symbol(f'phi_c_{name}', real=True)
    phi_n = sym.Symbol(f'phi_n_{name}', real=True)
    v_ra = sym.Symbol(f'v_ra_{name}', real=True)
    v_rb = sym.Symbol(f'v_rb_{name}', real=True)
    v_rc = sym.Symbol(f'v_rc_{name}', real=True)
    v_rn = sym.Symbol(f'v_rn_{name}', real=True)    
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    v_tao_r,v_tbo_r,v_tco_r,v_tno_r = sym.symbols(f'v_tao_r_{name},v_tbo_r_{name},v_tco_r_{name},v_tno_r_{name}', real=True)
    v_tao_i,v_tbo_i,v_tco_i,v_tno_i = sym.symbols(f'v_tao_i_{name},v_tbo_i_{name},v_tco_i_{name},v_tno_i_{name}', real=True)


    # dynamical states
    p_cf= sym.Symbol(f'p_cf_{name}', real=True)   
    phi = sym.Symbol(f'phi_{name}', real=True)
    xi_p = sym.Symbol(f'xi_p_{name}', real=True)
    xi_q = sym.Symbol(f'xi_q_{name}', real=True)
    p_ef = sym.Symbol(f'p_ef_{name}', real=True)
    De_ao_m,De_bo_m,De_co_m,De_no_m  = sym.symbols(f'De_ao_m_{name},De_bo_m_{name},De_co_m_{name},De_no_m_{name}', real=True)
    omega = sym.Symbol(f'omega_{name}', real=True)
    
    ## auxiliar variables
    omega = sym.Symbol(f'omega_{name}', real=True)
    p_m = sym.Symbol(f'p_m_{name}', real=True) 

    
    ## auxiliar equations
    alpha = np.exp(2.0/3*np.pi*1j)
    A_0a =  np.array([[1, 1, 1],
                    [1, alpha**2, alpha],
                    [1, alpha, alpha**2]])

    A_a0 = 1/3* np.array([[1, 1, 1],
                        [1, alpha, alpha**2],
                        [1, alpha**2, alpha]])

    omega_coi_i = 0
    HS_coi = 0

    ## POI currents
    i_sa = i_sa_r + 1j*i_sa_i
    i_sb = i_sb_r + 1j*i_sb_i
    i_sc = i_sc_r + 1j*i_sc_i
    i_sn = i_sn_r + 1j*i_sn_i

    ## POI voltages
    v_sa = v_sa_r + 1j*v_sa_i
    v_sb = v_sb_r + 1j*v_sb_i
    v_sc = v_sc_r + 1j*v_sc_i
    v_sn = v_sn_r + 1j*v_sn_i
    v_og = v_og_r + 1j*v_og_i

    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])

    v_szpn = A_a0*v_sabc
    i_szpn = A_a0*i_sabc
    
    s_pos = 3*v_szpn[1]*sym.conjugate(i_szpn[1])
    s_neg = 3*v_szpn[2]*sym.conjugate(i_szpn[2])
    s_zer = 3*v_szpn[0]*sym.conjugate(i_szpn[0])
    
    p_pos = sym.re(s_pos)
    q_pos = sym.im(s_pos)

    p_r = K_agc*xi_freq
    epsilon_p = p_m - p_ef
    epsilon_q = q_ref - q_pos/S_n
    De_q_nosat = K_qp*epsilon_q + K_qi*xi_q
    De_min = -U_n*0.05
    De_max =  U_n*0.05
    De_q  = sym.Piecewise((De_min,De_q_nosat<De_min),
                          (De_max,De_q_nosat>De_max),
                          (De_q_nosat,True))
    K_qaw = sym.Piecewise((0,De_q_nosat<De_min),
                          (0,De_q_nosat>De_max),
                          (1,True))

    Z_va = R_v + 1j*X_v # virtual impedance phase a
    Z_vb = R_v + 1j*X_v # virtual impedance phase b
    Z_vc = R_v + 1j*X_v # virtual impedance phase c

    e_ao_r = (e_ao_m+De_ao_m)*sym.cos(phi_a + phi) 
    e_ao_i = (e_ao_m+De_ao_m)*sym.sin(phi_a + phi) 
    e_bo_r = (e_bo_m+De_bo_m)*sym.cos(phi_b + phi-2/3*np.pi) 
    e_bo_i = (e_bo_m+De_bo_m)*sym.sin(phi_b + phi-2/3*np.pi) 
    e_co_r = (e_co_m+De_co_m)*sym.cos(phi_c + phi-4/3*np.pi) 
    e_co_i = (e_co_m+De_co_m)*sym.sin(phi_c + phi-4/3*np.pi) 
    e_no_r = (e_no_m+De_no_m)*sym.cos(phi_n + phi) 
    e_no_i = (e_no_m+De_no_m)*sym.sin(phi_n + phi)

    e_ao_cplx = e_ao_r + 1j*e_ao_i
    e_bo_cplx = e_bo_r + 1j*e_bo_i
    e_co_cplx = e_co_r + 1j*e_co_i
    e_no_cplx = e_no_r + 1j*e_no_i

    ## VSC AC side voltages
    v_tao = v_tao_r + 1j*v_tao_i
    v_tbo = v_tbo_r + 1j*v_tbo_i
    v_tco = v_tco_r + 1j*v_tco_i
    v_tno = v_tno_r + 1j*v_tno_i

    ## differential equations
    dphi   = 2*np.pi*50*(omega - omega_coi) - K_delta*phi
    dxi_p = epsilon_p
    dxi_q = K_qaw*epsilon_q - (1-K_qaw)*xi_q - 1e-6*xi_q
    dp_ef = 1/T_e*(p_pos/S_n - p_ef)
    dp_cf = 1/T_c*(p_c - p_cf)
    dDe_ao_m = 1/T_v*(v_ra + De_q - De_ao_m)
    dDe_bo_m = 1/T_v*(v_rb + De_q - De_bo_m)
    dDe_co_m = 1/T_v*(v_rc + De_q - De_co_m)
    dDe_no_m = 1/T_v*(v_rn - De_no_m)

    grid.dae['f'] += [dphi,dxi_p,dxi_q,dp_ef,dp_cf,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m] #,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m]
    grid.dae['x'] += [ phi, xi_p, xi_q, p_ef, p_cf, De_ao_m, De_bo_m, De_co_m, De_no_m] #, De_ao_m, De_bo_m, De_co_m, De_no_m]
    
    ## algebraic equations   
    g_omega = -omega + K_p*(epsilon_p + xi_p/T_p) + 1
    g_p_m  = -p_m + p_cf + p_r - 1/Droop*(omega - omega_ref)
    eq_v_tao_cplx =  e_ao_cplx - i_sa*Z_va - v_tao   # v_sa = v_sag
    eq_v_tbo_cplx =  e_bo_cplx - i_sb*Z_vb - v_tbo
    eq_v_tco_cplx =  e_co_cplx - i_sc*Z_vc - v_tco

    grid.dae['g'] +=     [g_omega, g_p_m] 
    grid.dae['y_ini'] += [  omega,   p_m] 
    grid.dae['y_run'] += [  omega,   p_m] 
 
    grid.dae['g'] += [sym.re(eq_v_tao_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_tbo_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_tco_cplx)] 
    grid.dae['g'] += [sym.im(eq_v_tao_cplx)]
    grid.dae['g'] += [sym.im(eq_v_tbo_cplx)]
    grid.dae['g'] += [sym.im(eq_v_tco_cplx)]
    grid.dae['y_ini'] += [v_tao_r,v_tbo_r,v_tco_r]
    grid.dae['y_ini'] += [v_tao_i,v_tbo_i,v_tco_i]
    grid.dae['y_run'] += [v_tao_r,v_tbo_r,v_tco_r]
    grid.dae['y_run'] += [v_tao_i,v_tbo_i,v_tco_i]

        
    V_1 = 400/np.sqrt(3)
    V_b = V_1
    #    V_1 = 400/np.sqrt(3)*np.exp(1j*np.deg2rad(0))
    # A_1toabc = np.array([1, alpha**2, alpha])
    #V_abc = V_1 * A_1toabc 
    #e_an_r,e_bn_r,e_cn_r = V_abc.real
    #e_an_i,e_bn_i,e_cn_i = V_abc.imag

    ## inputs default values
    grid.dae['u_ini_dict'].update({f'e_ao_m_{name}':V_1,
                       f'e_bo_m_{name}':V_1,
                       f'e_co_m_{name}':V_1,
                       f'e_no_m_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'v_ra_{name}':0.0,
                       f'v_rb_{name}':0.0,
                       f'v_rc_{name}':0.0,
                       f'v_rn_{name}':0.0})
    grid.dae['u_run_dict'].update({f'e_ao_m_{name}':V_1,
                       f'e_bo_m_{name}':V_1,
                       f'e_co_m_{name}':V_1,
                       f'e_no_m_{name}':0.0})
    grid.dae['u_run_dict'].update({f'v_ra_{name}':0.0,
                       f'v_rb_{name}':0.0,
                       f'v_rc_{name}':0.0,
                       f'v_rn_{name}':0.0})

    #u_dict.update({f'phi_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'phi_a_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'phi_b_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'phi_c_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'phi_n_{name}':0.0})

    grid.dae['u_ini_dict'].update({f'p_c_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'omega_{name}_ref':1.0})
    grid.dae['u_ini_dict'].update({f'q_ref_{name}':0.0})

    grid.dae['u_run_dict'].update({f'phi_a_{name}':0.0})
    grid.dae['u_run_dict'].update({f'phi_b_{name}':0.0})
    grid.dae['u_run_dict'].update({f'phi_c_{name}':0.0})
    grid.dae['u_run_dict'].update({f'phi_n_{name}':0.0})

    grid.dae['u_run_dict'].update({f'p_c_{name}':0.0})
    grid.dae['u_run_dict'].update({f'omega_{name}_ref':1.0})
    grid.dae['u_run_dict'].update({f'q_ref_{name}':0.0})

    #for ph in ['a','b','c','n']:
    #    u_dict.pop(f'i_{name}_{ph}_r')
    #    u_dict.pop(f'i_{name}_{ph}_i')

    ## parameters default values
    grid.dae['params_dict'].update({f'X_v_{name}':data['X_v'],f'R_v_{name}':data['R_v']})
    grid.dae['params_dict'].update({f'S_n_{name}':data['S_n']})
    grid.dae['params_dict'].update({f'U_n_{name}':data['U_n']})
    grid.dae['params_dict'].update({f'T_e_{name}':data['T_e']})
    grid.dae['params_dict'].update({f'T_c_{name}':data['T_c']})
    grid.dae['params_dict'].update({f'T_v_{name}':data['T_v']})
    grid.dae['params_dict'].update({f'Droop_{name}':data['Droop']})
    grid.dae['params_dict'].update({f'K_p_{name}':data['K_p']})
    grid.dae['params_dict'].update({f'T_p_{name}':data['T_p']})
    grid.dae['params_dict'].update({f'K_agc_{name}':data['K_agc']})
    grid.dae['params_dict'].update({f'K_delta_{name}':data['K_delta']})
    grid.dae['params_dict'].update({f'K_qp_{name}':data['K_qp']})
    grid.dae['params_dict'].update({f'K_qi_{name}':data['K_qi']})
    

    grid.dae['h_dict'].update({f'p_{name}_pos':sym.re(s_pos),
                   f'p_{name}_neg':sym.re(s_neg),
                   f'p_{name}_zer':sym.re(s_zer)})
    grid.dae['h_dict'].update({f'q_{name}_pos':sym.im(s_pos)})
    grid.dae['h_dict'].update({str(e_ao_m):e_ao_m,str(e_bo_m):e_bo_m,str(e_co_m):e_co_m})
    grid.dae['h_dict'].update({str(v_ra):v_ra,str(v_rb):v_rb,str(v_rc):v_rc})
    grid.dae['h_dict'].update({str(p_c):p_c,str(omega_ref):omega_ref})
    grid.dae['h_dict'].update({str(phi):phi})

    HS_coi  = S_n
    omega_coi_i = S_n*omega

    grid.omega_coi_numerator += omega_coi_i 
    grid.omega_coi_denominator += HS_coi


def test_ib_build():
    '''
    test grid former connected to infinite bus
    '''
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db
    import pytest

    grid = urisi('ac_3ph_4w_gfpizv_ib.hjson')
    grid.uz_jacs = True
    grid.build('temp')

def test_ib_ini():

    import temp

    model = temp.model()    
    p_c = 0.5
    q_ref = 0.1
    model.ini({'p_c_A1':p_c,'q_ref_A1':q_ref},'xy_0.json')
    
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()

    S_n = model.get_value('S_n_A1')
    assert model.get_value('omega_A1')  == pytest.approx(1)
    assert model.get_value('omega_coi') == pytest.approx(1)
    assert model.get_value('p_vsc_A1_a') == pytest.approx(p_c*S_n/3, rel=0.05)
    assert model.get_value('p_vsc_A1_b') == pytest.approx(p_c*S_n/3, rel=0.05)
    assert model.get_value('p_vsc_A1_c') == pytest.approx(p_c*S_n/3, rel=0.05)
    assert model.get_value('p_A2') == pytest.approx(-p_c*S_n, rel=0.05)
    assert model.get_value('q_A2') == pytest.approx(-q_ref*S_n, rel=0.05)

def test_iso_build():
    '''
    test isolated grid former feeding a load
    '''
    import numpy as np
    import sympy as sym
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db
    import pytest

    grid = urisi('ac_3ph_4w_gfpizv_iso.hjson')
    grid.uz_jacs = True
    grid.build('temp')

def test_iso_ini():

    import temp

    model = temp.model()
    p_load_ph =  50e3
    q_load_ph =  20e3
    model.ini({'p_load_A2_a':p_load_ph,'q_load_A2_a':q_load_ph,
               'p_load_A2_b':p_load_ph,'q_load_A2_b':q_load_ph,
               'p_load_A2_c':p_load_ph,'q_load_A2_c':q_load_ph,},'xy_0.json')

    model.report_x()  
    model.report_y()
    model.report_z()

    assert model.get_value('omega_A1')  == 1.0
    assert model.get_value('omega_coi') == 1.0
    assert model.get_value('p_vsc_A1_a') == pytest.approx(p_load_ph, rel=0.05)
    assert model.get_value('p_vsc_A1_b') == pytest.approx(p_load_ph, rel=0.05)
    assert model.get_value('p_vsc_A1_c') == pytest.approx(p_load_ph, rel=0.05)
    assert model.get_value('q_vsc_A1_a') == pytest.approx(q_load_ph, rel=0.05)
    assert model.get_value('q_vsc_A1_b') == pytest.approx(q_load_ph, rel=0.05)
    assert model.get_value('q_vsc_A1_c') == pytest.approx(q_load_ph, rel=0.05)


if __name__ == '__main__':

    #development()
    # test_iso_build()
    # test_iso_ini()
    test_ib_build()
    test_ib_ini()