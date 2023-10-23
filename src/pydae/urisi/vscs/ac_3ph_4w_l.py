
import numpy as np
import sympy as sym

def ac_3ph_4w_l(grid,vsc_data):
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
    v_dc = sym.Symbol(f'v_dc_{name}', real=True)
    m_a,m_b,m_c,m_n = sym.symbols(f'm_a_{name},m_b_{name},m_c_{name},m_n_{name}', real=True)
    phi = sym.Symbol(f'phi_{name}', real=True)
    phi_a = sym.Symbol(f'phi_a_{name}', real=True)
    phi_b = sym.Symbol(f'phi_b_{name}', real=True)
    phi_c = sym.Symbol(f'phi_c_{name}', real=True)
    phi_n = sym.Symbol(f'phi_n_{name}', real=True)

    # parameters
    R_s,R_sn,R_ng = sym.symbols(f'R_{name}_s,R_{name}_sn,R_{name}_ng', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_{name}_s,X_{name}_sn,X_{name}_ng', real=True)
    
    # dynamical states
    De_ao_m,De_bo_m,De_co_m,De_no_m  = sym.symbols(f'De_ao_m_{name},De_bo_m_{name},De_co_m_{name},De_no_m_{name}', real=True)

    omega = sym.Symbol(f'omega_{name}', real=True)
    
    ## algebraic states
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    p_dc,i_dc = sym.symbols(f'p_dc_{name},i_dc_{name}', real=True)
    p_pos,q_pos = sym.symbols(f'p_pos_{name},q_pos_{name}', real=True)

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

    s_s_total = s_sa + s_sb + s_sc + s_sn
    p_s_total = sym.re(s_s_total) 

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

    s_ta = e_ao_cplx * sym.conjugate(i_sa)
    s_tb = e_bo_cplx * sym.conjugate(i_sb)
    s_tc = e_co_cplx * sym.conjugate(i_sc)
    s_tn = e_no_cplx * sym.conjugate(i_sc)

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
    A_loss = sym.symbols(f'A_loss_{name}',real=True)
    B_loss = sym.symbols(f'B_loss_{name}',real=True)
    C_loss = sym.symbols(f'C_loss_{name}',real=True)
    i_rms_a = sym.sqrt(i_sa_r**2+i_sa_i**2+1e-6) 
    i_rms_b = sym.sqrt(i_sb_r**2+i_sb_i**2+1e-6) 
    i_rms_c = sym.sqrt(i_sc_r**2+i_sc_i**2+1e-6) 
    i_rms_n = sym.sqrt(i_sn_r**2+i_sn_i**2+1e-6) 

    p_loss_a = A_loss*i_rms_a*i_rms_a + B_loss*i_rms_a + C_loss
    p_loss_b = A_loss*i_rms_b*i_rms_b + B_loss*i_rms_b + C_loss
    p_loss_c = A_loss*i_rms_c*i_rms_c + B_loss*i_rms_c + C_loss
    p_loss_n = A_loss*i_rms_n*i_rms_n + B_loss*i_rms_n + C_loss

    p_vsc_loss = p_loss_a + p_loss_b + p_loss_c + p_loss_n
    p_ac = p_ac_a + p_ac_b + p_ac_c + p_ac_n

    eq_p_dc = -p_dc + p_ac + p_vsc_loss
    eq_i_dc =  p_dc/v_dc - i_dc
 
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
        i_s_r = sym.Symbol(f'i_vsc_{name}_{ph}_r', real=True)
        i_s_i = sym.Symbol(f'i_vsc_{name}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(name,ph)
        grid.dae['g'] [idx_r] += -i_s_r
        grid.dae['g'] [idx_i] += -i_s_i
        i_s = i_s_r + 1j*i_s_i
        i_s_m = np.abs(i_s)
        h_dict.update({f'i_vsc_{name}_{ph}_m':i_s_m})

    u_ini_dict.update({f'v_dc_{name}':800.0})
    u_run_dict.update({f'v_dc_{name}':800.0})
    m = 0.7071
    u_ini_dict.update({f'm_a_{name}':m,
                       f'm_b_{name}':m,
                       f'm_c_{name}':m,
                       f'm_n_{name}':0.0})
    u_run_dict.update({f'm_a_{name}':m,
                       f'm_b_{name}':m,
                       f'm_c_{name}':m,
                       f'm_n_{name}':0.0})

    u_ini_dict.update({f'phi_{name}':0.0})
    u_ini_dict.update({f'phi_a_{name}':0.0})
    u_ini_dict.update({f'phi_b_{name}':0.0})
    u_ini_dict.update({f'phi_c_{name}':0.0})
    u_ini_dict.update({f'phi_n_{name}':0.0})

    u_run_dict.update({f'phi_{name}':0.0})
    u_run_dict.update({f'phi_a_{name}':0.0})
    u_run_dict.update({f'phi_b_{name}':0.0})
    u_run_dict.update({f'phi_c_{name}':0.0})
    u_run_dict.update({f'phi_n_{name}':0.0})

    params_dict.update({f'X_{name}_s':vsc_data['X'],f'R_{name}_s':vsc_data['R']})
    params_dict.update({f'X_{name}_sn':vsc_data['X_n'],f'R_{name}_sn':vsc_data['R_n']})
    params_dict.update({f'X_{name}_ng':vsc_data['X_ng'],f'R_{name}_ng':vsc_data['R_ng']})
    
    params_dict.update({f'S_n_{name}':vsc_data['S_n']})
   
    v_sabc = sym.Matrix([[v_sa],[v_sb],[v_sc]])
    i_sabc = sym.Matrix([[i_sa],[i_sb],[i_sc]])
    
    v_szpn = A_a0*v_sabc
    i_szpn = A_a0*i_sabc
    
    s_pos = 3*v_szpn[1]*sym.conjugate(i_szpn[1])
    s_neg = 3*v_szpn[2]*sym.conjugate(i_szpn[2])
    s_zer = 3*v_szpn[0]*sym.conjugate(i_szpn[0])

    g_list += [sym.re(s_pos) - p_pos]
    g_list += [sym.im(s_pos) - q_pos]

    y_ini_list += [p_pos,q_pos]
    y_run_list += [p_pos,q_pos]

    h_dict.update({f'p_{name}_pos':sym.re(s_pos),f'p_{name}_neg':sym.re(s_neg),f'p_{name}_zer':sym.re(s_zer)})
    h_dict.update({str(m_a):m_a,str(m_b):m_b,str(m_c):m_c,str(m_n):m_n})
    h_dict.update({str(phi):phi})
    h_dict.update({f'p_dc_{name}':p_dc})
    h_dict.update({f'p_s_total_{name}':p_s_total})
    
    
    S_n_num = vsc_data['S_n']
    U_n_num = vsc_data['U_n']
    I_n = S_n_num/(np.sqrt(3)*U_n_num)
    P_0 = 0.01*S_n_num
    C_loss_num = P_0/3
    P_cc = 0.01*S_n_num
    A_loss_num = P_cc/(I_n**2)/3
    B_loss_num = 1.0

    if 'A_loss' in vsc_data:
        A_loss_num = vsc_data['A_loss']
        B_loss_num = vsc_data['B_loss']
        C_loss_num = vsc_data['C_loss']

    params_dict.update({str(A_loss):A_loss_num})
    params_dict.update({str(B_loss):B_loss_num})
    params_dict.update({str(C_loss):C_loss_num})


def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('ac_3ph_4w_l.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 100e3
    V_n = 400
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    lossses = 1.0
    A = Conduction_losses/(I_n**2)/3*lossses
    B = 1/3*lossses
    C = 0.02*S_n/3*lossses
    model = temp.model()
    m = 0.75
    phi = 0.1
    model.ini({
               #'A_loss_A1':A,'B_loss_A1':B,'C_loss_A1':C,
               'm_a_A1':m,'m_b_A1':m,'m_c_A1':m,
               'phi_a_A1':phi,'phi_b_A1':phi,'phi_c_A1':phi,
               },'xy_0.json')
    model.report_y()
    model.report_z()


if __name__ == '__main__':

    #development()
    test()
