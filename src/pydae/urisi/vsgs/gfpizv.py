
import numpy as np
import sympy as sym

def gfpizv(grid,data,name,bus_name):

    # inputs
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
    K_soc = sym.Symbol(f"K_soc_{name}", real=True) 

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
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{bus_name}_0_r,V_{bus_name}_1_r,V_{bus_name}_2_r,V_{bus_name}_3_r,v_{bus_name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{bus_name}_0_i,V_{bus_name}_1_i,V_{bus_name}_2_i,V_{bus_name}_3_i,v_{bus_name}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    v_ta_r,v_tb_r,v_tc_r,v_tn_r = sym.symbols(f'v_ta_r_{name},v_tb_r_{name},v_tc_r_{name},v_tn_r_{name}', real=True)
    v_ta_i,v_tb_i,v_tc_i,v_tn_i = sym.symbols(f'v_ta_i_{name},v_tb_i_{name},v_tc_i_{name},v_tn_i_{name}', real=True)
    p_soc = sym.Symbol(f'p_soc_{name}',real=True)
    v_dc = sym.Symbol(f'v_dc_{name}',real=True)


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
    v_ta = v_ta_r + 1j*v_ta_i
    v_tb = v_tb_r + 1j*v_tb_i
    v_tc = v_tc_r + 1j*v_tc_i
    v_tn = v_tn_r + 1j*v_tn_i

    ## differential equations
    dphi   = 2*np.pi*50*(omega - omega_coi) - K_delta*phi
    dxi_p = epsilon_p
    dxi_q = K_qaw*epsilon_q - (1-K_qaw)*xi_q  - 1e-6*xi_q
    dp_ef = 1/T_e*(p_pos/S_n - p_ef)
    dp_cf = 1/T_c*(p_c - p_cf)
    dDe_ao_m = 1/T_v*(v_ra + De_q - De_ao_m)
    dDe_bo_m = 1/T_v*(v_rb + De_q - De_bo_m)
    dDe_co_m = 1/T_v*(v_rc + De_q - De_co_m)
    dDe_no_m = 1/T_v*(v_rn - De_no_m)

    grid.dae['f'] += [dphi,dxi_p,dxi_q,dp_ef,dp_cf,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m] #,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m]
    grid.dae['x'] += [ phi, xi_p, xi_q, p_ef, p_cf, De_ao_m, De_bo_m, De_co_m, De_no_m] #, De_ao_m, De_bo_m, De_co_m, De_no_m]
    # grid.dae['f'] += [dphi,dxi_p,dp_ef,dp_cf,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m] #,dDe_ao_m,dDe_bo_m,dDe_co_m,dDe_no_m]
    # grid.dae['x'] += [ phi, xi_p, p_ef, p_cf, De_ao_m, De_bo_m, De_co_m, De_no_m] #, De_ao_m, De_bo_m, De_co_m, De_no_m]
        
    ## algebraic equations   
    g_omega = -omega + K_p*(epsilon_p + xi_p/T_p) + 1
    g_p_m  = -p_m + p_cf + p_r - 1/Droop*(omega - omega_ref)  + K_soc*p_soc
    eq_v_ta_cplx =  e_ao_cplx - i_sa*Z_va - v_ta   # v_sa = v_sag
    eq_v_tb_cplx =  e_bo_cplx - i_sb*Z_vb - v_tb
    eq_v_tc_cplx =  e_co_cplx - i_sc*Z_vc - v_tc

    grid.dae['g'] +=     [g_omega, g_p_m] 
    grid.dae['y_ini'] += [  omega,   p_m] 
    grid.dae['y_run'] += [  omega,   p_m] 
 
    grid.dae['g'] += [sym.re(eq_v_ta_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_tb_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_tc_cplx)] 
    grid.dae['g'] += [sym.im(eq_v_ta_cplx)]
    grid.dae['g'] += [sym.im(eq_v_tb_cplx)]
    grid.dae['g'] += [sym.im(eq_v_tc_cplx)]
    grid.dae['y_ini'] += [v_ta_r,v_tb_r,v_tc_r]
    grid.dae['y_ini'] += [v_ta_i,v_tb_i,v_tc_i]
    grid.dae['y_run'] += [v_ta_r,v_tb_r,v_tc_r]
    grid.dae['y_run'] += [v_ta_i,v_tb_i,v_tc_i]

        
    V_1 = 400/np.sqrt(3)
    V_b = V_1
    #    V_1 = 400/np.sqrt(3)*np.exp(1j*np.deg2rad(0))
    # A_1toabc = np.array([1, alpha**2, alpha])
    #V_abc = V_1 * A_1toabc 
    #e_an_r,e_bn_r,e_cn_r = V_abc.real
    #e_an_i,e_bn_i,e_cn_i = V_abc.imag

    ## inputs default values
    grid.dae['u_ini_dict'].update(
                      {f'e_ao_m_{name}':V_1,
                       f'e_bo_m_{name}':V_1,
                       f'e_co_m_{name}':V_1,
                       f'e_no_m_{name}':0.0})
    grid.dae['u_ini_dict'].update({
                       f'v_ra_{name}':0.0,
                       f'v_rb_{name}':0.0,
                       f'v_rc_{name}':0.0,
                       f'v_rn_{name}':0.0})
    grid.dae['u_run_dict'].update({
                       f'e_ao_m_{name}':V_1,
                       f'e_bo_m_{name}':V_1,
                       f'e_co_m_{name}':V_1,
                       f'e_no_m_{name}':0.0})
    grid.dae['u_run_dict'].update(
                      {f'v_ra_{name}':0.0,
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

    grid.dae['u_ini_dict'].update({f'p_soc_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_soc_{name}':0.0})

    grid.dae['u_ini_dict'].update({f'v_dc_{name}':800.0})
    grid.dae['u_run_dict'].update({f'v_dc_{name}':800.0})

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
    grid.dae['params_dict'].update({f'K_soc_{name}':1.0})
        

    grid.dae['u_ini_dict'].pop(str(v_ta_r))
    grid.dae['u_ini_dict'].pop(str(v_tb_r))
    grid.dae['u_ini_dict'].pop(str(v_tc_r))
    grid.dae['u_run_dict'].pop(str(v_ta_r))
    grid.dae['u_run_dict'].pop(str(v_tb_r))
    grid.dae['u_run_dict'].pop(str(v_tc_r))
    grid.dae['u_ini_dict'].pop(str(v_ta_i))
    grid.dae['u_ini_dict'].pop(str(v_tb_i))
    grid.dae['u_ini_dict'].pop(str(v_tc_i))
    grid.dae['u_run_dict'].pop(str(v_ta_i))
    grid.dae['u_run_dict'].pop(str(v_tb_i))
    grid.dae['u_run_dict'].pop(str(v_tc_i))


    
    grid.dae['xy_0_dict'].update({f'omega_{name}':1.0})


    grid.dae['h_dict'].update({f'p_pos_{name}':sym.re(s_pos),
                   f'p_neg_{name}':sym.re(s_neg),
                   f'p_zer_{name}':sym.re(s_zer)})
    grid.dae['h_dict'].update({f'q_pos_{name}':sym.im(s_pos)})
    grid.dae['h_dict'].update({str(e_ao_m):e_ao_m,str(e_bo_m):e_bo_m,str(e_co_m):e_co_m})
    grid.dae['h_dict'].update({str(v_ra):v_ra,str(v_rb):v_rb,str(v_rc):v_rc})
    grid.dae['h_dict'].update({str(p_c):p_c,str(omega_ref):omega_ref})
    grid.dae['h_dict'].update({str(phi):phi})

    HS_coi  = S_n
    omega_coi_i = S_n*omega

    grid.omega_coi_numerator += omega_coi_i 
    grid.omega_coi_denominator += HS_coi


def test_ib():
    '''
    test grid former connected to infinite bus
    '''
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db
    import pytest

    grid = urisi('gfpizv_ib.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

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

def test_iso():
    '''
    test isolated grid former feeding a load
    '''
    import numpy as np
    import sympy as sym
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db
    import pytest

    grid = urisi('gfpizv_iso.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

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


def test_sharing():
    '''
    test isolated grid former feeding a load
    '''
    import numpy as np
    import sympy as sym
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db
    import pytest

    grid = urisi('gfpizv_sharing.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()
    p_load_ph =  50e3
    q_load_ph =  20e3
    model.ini({'p_load_A2_a':p_load_ph,'q_load_A2_a':q_load_ph,
               'p_load_A2_b':p_load_ph,'q_load_A2_b':q_load_ph,
               'p_load_A2_c':p_load_ph,'q_load_A2_c':q_load_ph,},'xy_0.json')

    # model.report_x()  
    # model.report_y()
    # model.report_z()

    assert model.get_value('omega_A1')  == 1.0
    assert model.get_value('omega_A3')  == 1.0
    assert model.get_value('omega_coi') == 1.0
    assert model.get_value('p_vsc_A1_a') == pytest.approx(p_load_ph/2, rel=0.05)
    assert model.get_value('p_vsc_A1_b') == pytest.approx(p_load_ph/2, rel=0.05)
    assert model.get_value('p_vsc_A1_c') == pytest.approx(p_load_ph/2, rel=0.05)
    assert model.get_value('p_vsc_A3_a') == pytest.approx(p_load_ph/2, rel=0.05)
    assert model.get_value('p_vsc_A3_b') == pytest.approx(p_load_ph/2, rel=0.05)
    assert model.get_value('p_vsc_A3_c') == pytest.approx(p_load_ph/2, rel=0.05)


if __name__ == '__main__':

    #development()
    test_iso()