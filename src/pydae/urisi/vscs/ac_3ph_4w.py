
import numpy as np
import sympy as sym

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
    R_s,R_sn,R_ng = sym.symbols(f'R_s_{name},R_sn_{name},R_ng_{name}', real=True)
    X_s,X_sn,X_ng = sym.symbols(f'X_s_{name},X_sn_{name},X_ng_{name}', real=True)
    A_loss = sym.symbols(f'A_loss_{name}',real=True)
    B_loss = sym.symbols(f'B_loss_{name}',real=True)
    C_loss = sym.symbols(f'C_loss_{name}',real=True)

    ## VSC inputs
    v_ta_r, v_ta_i = sym.symbols(f'v_ta_r_{name},v_ta_i_{name}', real=True)
    v_tb_r, v_tb_i = sym.symbols(f'v_tb_r_{name},v_tb_i_{name}', real=True)
    v_tc_r, v_tc_i = sym.symbols(f'v_tc_r_{name},v_tc_i_{name}', real=True)
    v_tn_r, v_tn_i = sym.symbols(f'v_tn_r_{name},v_tn_i_{name}', real=True)


    # algebraic states
    #e_an_i,e_bn_i,e_cn_i,e_ng_i = sym.symbols(f'e_{name}_an_i,e_{name}_bn_i,e_{name}_cn_i,e_{name}_ng_i', real=True)
    #v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'v_{name}_a_r,v_{name}_b_r,v_{name}_c_r,v_{name}_n_r,v_{name}_o_r', real=True)
    #v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'v_{name}_a_i,v_{name}_b_i,v_{name}_c_i,v_{name}_n_i,v_{name}_o_i', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)
    i_sa_r,i_sb_r,i_sc_r,i_sn_r,i_ng_r = sym.symbols(f'i_vsc_{name}_a_r,i_vsc_{name}_b_r,i_vsc_{name}_c_r,i_vsc_{name}_n_r,i_vsc_{name}_ng_r', real=True)
    i_sa_i,i_sb_i,i_sc_i,i_sn_i,i_ng_i = sym.symbols(f'i_vsc_{name}_a_i,i_vsc_{name}_b_i,i_vsc_{name}_c_i,i_vsc_{name}_n_i,i_vsc_{name}_ng_i', real=True)
    v_ta_r,v_tb_r,v_tc_r,v_tn_r = sym.symbols(f'v_ta_r_{name},v_tb_r_{name},v_tc_r_{name},v_tn_r_{name}', real=True)
    v_ta_i,v_tb_i,v_tc_i,v_tn_i = sym.symbols(f'v_ta_i_{name},v_tb_i_{name},v_tc_i_{name},v_tn_i_{name}', real=True)
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
    v_ta = v_ta_r + 1j*v_ta_i
    v_tb = v_tb_r + 1j*v_tb_i
    v_tc = v_tc_r + 1j*v_tc_i
    v_tn = v_tn_r + 1j*v_tn_i
    
    s_ta = v_ta * sym.conjugate(i_sa)
    s_tb = v_tb * sym.conjugate(i_sb)
    s_tc = v_tc * sym.conjugate(i_sc)
    s_tn = v_tn * sym.conjugate(i_sc)

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

    eq_i_sa_cplx = v_og + v_ta - i_sa*Z_sa - v_sa   # v_sa = v_sag
    eq_i_sb_cplx = v_og + v_tb - i_sb*Z_sb - v_sb
    eq_i_sc_cplx = v_og + v_tc - i_sc*Z_sc - v_sc
    eq_i_sn_cplx = v_og + v_tn - i_sn*Z_sn - v_sn
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
    grid.dae['params_dict'].update({f'X_s_{name}':data['X'],f'R_s_{name}':data['R']})
    grid.dae['params_dict'].update({f'X_sn_{name}':data['X_n'],f'R_sn_{name}':data['R_n']})
    grid.dae['params_dict'].update({f'X_ng_{name}':data['X_ng'],f'R_ng_{name}':data['R_ng']})

    V_n_N = data['U_n']/np.sqrt(3)
    phi_N = 0.0
    if 'phi' in data: phi_N = data['phi']

    v_ta_N = V_n_N*np.exp(1j*phi_N)
    v_tb_N = V_n_N*np.exp(1j*(phi_N - np.pi*2.0/3.0))
    v_tc_N = V_n_N*np.exp(1j*(phi_N - np.pi*4.0/3.0))
    v_tn_N = 0.0

    # the following are inputs in this models, however they should be states in the controllers, 
    # for reducing complexity at the controllers the initaializations are performed here:
    grid.dae['xy_0_dict'].update({str(v_ta_r):v_ta_N.real, str(v_ta_i):v_ta_N.imag})
    grid.dae['xy_0_dict'].update({str(v_tb_r):v_tb_N.real, str(v_tb_i):v_tb_N.imag})
    grid.dae['xy_0_dict'].update({str(v_tc_r):v_tc_N.real, str(v_tc_i):v_tc_N.imag})
    grid.dae['xy_0_dict'].update({str(v_tn_r):v_tn_N.real, str(v_tn_i):v_tn_N.imag})

    S_n_N = data['S_n']
    U_n_N = data['U_n']
    V_n_N = U_n_N/np.sqrt(3)
    phi_N = 0.0
    if 'phi' in data: phi_N = data['phi']

    v_ta_N = V_n_N*np.exp(1j*phi_N)
    v_tb_N = V_n_N*np.exp(1j*(phi_N - np.pi*2.0/3.0))
    v_tc_N = V_n_N*np.exp(1j*(phi_N - np.pi*4.0/3.0))
    v_tn_N = 0.0

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

    grid.dae['u_ini_dict'].update({str(v_ta_r):v_ta_N.real, str(v_ta_i):v_ta_N.imag})
    grid.dae['u_ini_dict'].update({str(v_tb_r):v_tb_N.real, str(v_tb_i):v_tb_N.imag})
    grid.dae['u_ini_dict'].update({str(v_tc_r):v_tc_N.real, str(v_tc_i):v_tc_N.imag})
    grid.dae['u_ini_dict'].update({str(v_tn_r):v_tn_N.real, str(v_tn_i):v_tn_N.imag})
    grid.dae['u_run_dict'].update({str(v_ta_r):v_ta_N.real, str(v_ta_i):v_ta_N.imag})
    grid.dae['u_run_dict'].update({str(v_tb_r):v_tb_N.real, str(v_tb_i):v_tb_N.imag})
    grid.dae['u_run_dict'].update({str(v_tc_r):v_tc_N.real, str(v_tc_i):v_tc_N.imag})
    grid.dae['u_run_dict'].update({str(v_tn_r):v_tn_N.real, str(v_tn_i):v_tn_N.imag})



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

    grid = urisi('ac_3ph_4w_gfpizv_ib.hjson')
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

    grid = urisi('ac_3ph_4w_gfpizv_iso.hjson')
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


if __name__ == '__main__':

    #development()
    test_ib()