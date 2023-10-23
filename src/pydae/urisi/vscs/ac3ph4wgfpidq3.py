import numpy as np
import sympy as sym



def ac3ph4wgfpidq3(grid,vsc_data):
    '''
    VSC with 3 phase and 4 wire working in open loop as a grid former.

    
    '''

    sin = sym.sin
    cos = sym.cos
    sqrt = sym.sqrt
    exp = sym.exp
    
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

    omega_coi = sym.Symbol('omega_coi',real=True)
    xi_freq = sym.Symbol('xi_freq',real=True)
    K_agc = sym.Symbol('K_agc',real=True)

    #vscs = [
    #    {'bus':'B1','S_n':100e3,'R':0.01,'X':0.1,'R_n':0.01,'X_n':0.1,'R_ng':0.01,'X_ng':3.0,'K_f':0.1,'T_f':1.0,'K_sec':0.5,'K_delta':0.001},
    #    ]

    #for vsc in vsc_data:
    name = vsc_data['bus']

    # inputs
    e_ref_a_m,e_ref_b_m,e_ref_c_m = sym.symbols(f'e_ref_a_m_{name},e_ref_b_m_{name},e_ref_c_m_{name}', real=True)
    phi_ref_a,phi_ref_b,phi_ref_c = sym.symbols(f'phi_ref_a_{name},phi_ref_b_{name},phi_ref_c_{name}', real=True)
    p_c = sym.Symbol(f'p_c_{name}', real=True) 

    # parameters
    S_n,U_n,H,K_f,T_f,K_sec,K_delta  = sym.symbols(f'S_n_{name},U_n_{name},H_{name},K_f_{name},T_f_{name},K_sec_{name},K_delta_{name}', real=True)
    R_s,R_sn,R_ng = sym.symbols(f'R_{name}_s,R_{name}_sn,R_{name}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name}_s,X_{name}_sn,X_{name}_ng', real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)    
    T_p = sym.Symbol(f"T_p_{name}", real=True)   
    R_v = sym.Symbol(f"R_v_{name}", real=True)   
    X_v = sym.Symbol(f"X_v_{name}", real=True)   
    Droop = sym.Symbol(f"Droop_{name}", real=True)  
    T_wf = sym.Symbol(f"T_wf_{name}", real=True)
    T_sf = sym.Symbol(f"T_sf_{name}", real=True)

    # basis for per unit
    S_b = S_n
    U_b = U_n
    I_b = S_b/(np.sqrt(3)*U_b)
    U_bdq = U_b*(np.sqrt(2/3))
    I_bdq = I_b*np.sqrt(2)  
    
    # dynamical states
    phi = sym.Symbol(f'phi_{name}', real=True)
    xi_p = sym.Symbol(f'xi_p_{name}', real=True)
    omega_f = sym.Symbol(f'omega_f_{name}', real=True)
    p_s_f = sym.Symbol(f'p_s_f_{name}', real=True) 


    # algebraic states
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'v_{name}_a_r,v_{name}_b_r,v_{name}_c_r,v_{name}_n_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'v_{name}_a_i,v_{name}_b_i,v_{name}_c_i,v_{name}_n_i,v_{name}_o_i', real=True)
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
    v_og = v_og_r + 1j*v_og_i
    

    # Controller 

    s_s = 1/S_b*(v_sa * sym.conjugate(i_sa) + 
                 v_sb * sym.conjugate(i_sb) + 
                 v_sc * sym.conjugate(i_sc))
    p_s = sym.re(s_s)

    
    #references pu
    p_pri = 1.0/Droop*(1.0 - omega_f)
    p_agc = K_agc*xi_freq

    p_sec = K_sec*p_agc

    p_m_ref = p_sec + p_pri + p_c

    #error
    #epsilon_p = (p_m_ref - p_s);
    #error. Low pass filter or Notch filter
    epsilon_p = (p_m_ref - p_s_f)

    #omega VSG
    omega = K_p*(epsilon_p + xi_p/T_p) + 1.0; 

    #derivatives
    dphi =  2.0*np.pi*50.0*(omega - omega_coi) - K_delta*phi
    dxi_p = epsilon_p
    dp_s_f = 1.0/T_sf*(p_s - p_s_f)    #Low pass filter for p_s
    domega_f = 1.0/T_wf*(omega - omega_f)

    omega_rads = 2.0*np.pi*50.0*omega
    # from derivatives to the integrator
    h_dict.update({f'omega_{name}':omega})

    phi_vsg = phi
    
    i_sd_a = 1.0/I_b*(i_sa_i*cos(phi_vsg) + i_sa_r*sin(phi_vsg))
    i_sq_a = 1.0/I_b*(i_sa_i*sin(phi_vsg) - i_sa_r*cos(phi_vsg))
    i_sd_b = 1.0/I_b*(i_sb_i*cos(phi_vsg) + i_sb_r*sin(phi_vsg))
    i_sq_b = 1.0/I_b*(i_sb_i*sin(phi_vsg) - i_sb_r*cos(phi_vsg))
    i_sd_c = 1.0/I_b*(i_sc_i*cos(phi_vsg) + i_sc_r*sin(phi_vsg))
    i_sq_c = 1.0/I_b*(i_sc_i*sin(phi_vsg) - i_sc_r*cos(phi_vsg))
    
    h_dict.update({f'i_sd_a_{name}':i_sd_a,f'i_sq_a_{name}':i_sq_a})
    h_dict.update({f'i_sd_b_{name}':i_sd_b,f'i_sq_b_{name}':i_sq_b})
    h_dict.update({f'i_sd_c_{name}':i_sd_c,f'i_sq_c_{name}':i_sq_c})


    #v_sd_a = sqrt(2.0)/U_bdq*(v_sa_i*cos(phi_vsg) + v_sa_r*sin(phi_vsg))
    #v_sq_a = sqrt(2.0)/U_bdq*(v_sa_i*sin(phi_vsg) - v_sa_r*cos(phi_vsg))
    #v_sd_b = sqrt(2.0)/U_bdq*(v_sb_i*cos(phi_vsg) + v_sb_r*sin(phi_vsg))
    #v_sq_b = sqrt(2.0)/U_bdq*(v_sb_i*sin(phi_vsg) - v_sb_r*cos(phi_vsg))
    #v_sd_c = sqrt(2.0)/U_bdq*(v_sc_i*cos(phi_vsg) + v_sc_r*sin(phi_vsg))
    #v_sq_c = sqrt(2.0)/U_bdq*(v_sc_i*sin(phi_vsg) - v_sc_r*cos(phi_vsg))

    e_ref_a = e_ref_a_m*exp(sym.I*(phi_ref_a))
    e_ref_b = e_ref_b_m*exp(sym.I*(phi_ref_b))
    e_ref_c = e_ref_c_m*exp(sym.I*(phi_ref_c))

    e_ref_a_r = sym.re(e_ref_a)
    e_ref_a_i = sym.im(e_ref_a)
    e_ref_b_r = sym.re(e_ref_b)
    e_ref_b_i = sym.im(e_ref_b)
    e_ref_c_r = sym.re(e_ref_c)
    e_ref_c_i = sym.im(e_ref_c)

    e_d_ref_a = (e_ref_a_i*cos(phi_vsg) + e_ref_a_r*sin(phi_vsg))
    e_q_ref_a = (e_ref_a_i*sin(phi_vsg) - e_ref_a_r*cos(phi_vsg))
    e_d_ref_b = (e_ref_b_i*cos(phi_vsg) + e_ref_b_r*sin(phi_vsg))
    e_q_ref_b = (e_ref_b_i*sin(phi_vsg) - e_ref_b_r*cos(phi_vsg))
    e_d_ref_c = (e_ref_c_i*cos(phi_vsg) + e_ref_c_r*sin(phi_vsg))
    e_q_ref_c = (e_ref_c_i*sin(phi_vsg) - e_ref_c_r*cos(phi_vsg))

    #Voltage at the VSC terminal phase a
    v_t_d_a = -(i_sd_a*(R_v) - (X_v)*i_sq_a) + e_d_ref_a;  #pu
    v_t_q_a = -(i_sq_a*(R_v) + (X_v)*i_sd_a) + e_q_ref_a;  #pu

    #Voltage at the VSC terminal phase b
    v_t_d_b = -(i_sd_b*(R_v) - (X_v)*i_sq_b) + e_d_ref_b;  #pu
    v_t_q_b = -(i_sq_b*(R_v) + (X_v)*i_sd_b) + e_q_ref_b;  #pu

    #Voltage at the VSC terminal phase c
    v_t_d_c = -(i_sd_c*(R_v) - (X_v)*i_sq_c) + e_d_ref_c;  #pu
    v_t_q_c = -(i_sq_c*(R_v) + (X_v)*i_sd_c) + e_q_ref_c;  #pu

    v_t_a_i = U_b/np.sqrt(3.0)*( v_t_d_a*cos(phi_vsg) - v_t_q_a*sin(phi_vsg))
    v_t_a_r = U_b/np.sqrt(3.0)*(-v_t_d_a*sin(phi_vsg) - v_t_q_a*cos(phi_vsg))
    v_t_b_i = U_b/np.sqrt(3.0)*( v_t_d_b*cos(phi_vsg) - v_t_q_b*sin(phi_vsg))
    v_t_b_r = U_b/np.sqrt(3.0)*(-v_t_d_b*sin(phi_vsg) - v_t_q_b*cos(phi_vsg))
    v_t_c_i = U_b/np.sqrt(3.0)*( v_t_d_c*cos(phi_vsg) - v_t_q_c*sin(phi_vsg))
    v_t_c_r = U_b/np.sqrt(3.0)*(-v_t_d_c*sin(phi_vsg) - v_t_q_c*cos(phi_vsg))

    v_t_a = v_t_a_r + 1j*v_t_a_i
    v_t_b = v_t_b_r + 1j*v_t_b_i
    v_t_c = v_t_c_r + 1j*v_t_c_i

    h_dict.update({f'v_t_a_r_{name}':v_t_a_r,f'v_t_a_i_{name}':v_t_a_i})
    h_dict.update({f'v_t_b_r_{name}':v_t_b_r,f'v_t_b_i_{name}':v_t_b_i})
    h_dict.update({f'v_t_c_r_{name}':v_t_c_r,f'v_t_c_i_{name}':v_t_c_i})
    
    h_dict.update({f'v_t_d_a_{name}':v_t_d_a,f'v_t_q_a_{name}':v_t_q_a})
    h_dict.update({f'v_t_d_b_{name}':v_t_d_b,f'v_t_q_b_{name}':v_t_q_b})
    h_dict.update({f'v_t_d_c_{name}':v_t_d_c,f'v_t_q_c_{name}':v_t_q_c})   
    
    h_dict.update({f'e_d_ref_a_{name}':e_d_ref_a,f'e_q_ref_a_{name}':e_q_ref_a})
    h_dict.update({f'e_d_ref_b_{name}':e_d_ref_b,f'e_q_ref_b_{name}':e_q_ref_b})
    h_dict.update({f'e_d_ref_c_{name}':e_d_ref_c,f'e_q_ref_c_{name}':e_q_ref_c})


    ### VSC 4 wire:
    e_no_cplx = 0.0
    eq_i_sa_cplx = v_og + v_t_a - i_sa*Z_sa - v_sa   # v_sa = v_sag
    eq_i_sb_cplx = v_og + v_t_b - i_sb*Z_sb - v_sb
    eq_i_sc_cplx = v_og + v_t_c - i_sc*Z_sc - v_sc
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
    ### end VSC 4 wire
    
    #h_dict.update({f'v_t_a_{name}_r':sym.re(v_t_a),f'v_tb_{name}_r':sym.re(v_t_b),f'v_t_c_{name}_r':sym.re(v_t_c)})
        

    u_dict.update({f'e_ref_a_m_{name}':1.0,
                   f'e_ref_b_m_{name}':1.0,
                   f'e_ref_c_m_{name}':1.0,
                   f'e_{name}_no_m':0.0,
                   f'p_c_{name}':0.0})
    #u_dict.update({f'phi_{name}':0.0})
    u_dict.update({f'phi_ref_a_{name}': 0.0})
    u_dict.update({f'phi_ref_b_{name}':-2./3*np.pi})
    u_dict.update({f'phi_ref_c_{name}':-4./3*np.pi})

    u_dict.update({f'omega_{name}_ref':1.0})



    #for ph in ['a','b','c','n']:
    #    u_dict.pop(f'i_{name}_{ph}_r')
    #    u_dict.pop(f'i_{name}_{ph}_i')
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
    params_dict.update({f'K_sec_{name}':vsc_data['K_sec']})
    params_dict.update({f'K_delta_{name}':vsc_data['K_delta']})
    params_dict.update({f'T_wf_{name}':0.1})
    params_dict.update({f'T_sf_{name}':0.1})
 
    
    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])
    
    v_szpn = A_a0*v_sabc
    i_szpn = A_a0*i_sabc
    
    s_pos = 3*v_szpn[1]*sym.conjugate(i_szpn[1])
    s_neg = 3*v_szpn[2]*sym.conjugate(i_szpn[2])
    s_zer = 3*v_szpn[0]*sym.conjugate(i_szpn[0])    

    f_list += [dphi,dxi_p,dp_s_f,domega_f]
    x_list += [ phi, xi_p, p_s_f, omega_f]
    
    #h_dict.update({f'p_{name}_pos':sym.re(s_pos),f'p_{name}_neg':sym.re(s_neg),f'p_{name}_zer':sym.re(s_zer)})
    #h_dict.update({str(p_c):p_c,str(omega_ref):omega_ref})
    #h_dict.update({f'p_pri_{name}':p_pri})

    HS_coi  = S_n
    omega_coi_i = S_n*omega

    grid.omega_coi_h_i += omega_coi_i
    grid.hs_total += HS_coi    
    
    u_dict.update({f'omega_{name}_ref':1.0})

    #for ph in ['a','b','c','n']:
    #    u_dict.pop(f'i_{name}_{ph}_r')
    #    u_dict.pop(f'i_{name}_{ph}_i')
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
    
    
    
def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('ac3ph4wgfpidq3_ib.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()
    # model.ini({'p_load_A2_a':20e3,'q_load_A2_a':-10e3,
    #            'p_load_A2_b':20e3,'q_load_A2_b':-10e3,
    #            'p_load_A2_c':20e3,'q_load_A2_c':-10e3,},'xy_0.json')
    
    model.ini({'p_c_A1':10e3},'xy_0.json')
    
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()


if __name__ == '__main__':

    #development()
    test()
