
import numpy as np
import sympy as sym

def ctrl_3ph_4w_pq(grid,data,name,bus_name):


    # parameters
    S_n,U_n = sym.symbols(f'S_n_{name},U_n_{name}', real=True)
    R_s,X_s = sym.symbols(f'R_s_{name},X_s_{name}', real=True)

    ## inputs
    p_ref = sym.Symbol(f'p_ref_{name}', real=True)    
    q_ref = sym.Symbol(f'q_ref_{name}', real=True)    
    p_a_ref,q_a_ref = sym.symbols(f'p_a_ref_{name},q_a_ref_{name}', real=True)
    p_b_ref,q_b_ref = sym.symbols(f'p_b_ref_{name},q_b_ref_{name}', real=True)
    p_c_ref,q_c_ref = sym.symbols(f'p_c_ref_{name},q_c_ref_{name}', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{bus_name}_0_r,V_{bus_name}_1_r,V_{bus_name}_2_r,V_{bus_name}_3_r,v_{bus_name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{bus_name}_0_i,V_{bus_name}_1_i,V_{bus_name}_2_i,V_{bus_name}_3_i,v_{bus_name}_o_i', real=True)
    p_soc = sym.Symbol(f'p_soc_{name}',real=True)
    v_dc = sym.Symbol(f'v_dc_{name}',real=True)


    ## algebraic states
    i_sa_ref_r,i_sb_ref_r,i_sc_ref_r,i_sn_ref_r = sym.symbols(f'i_sa_ref_r_{name},i_sb_ref_r_{name},i_sc_ref_r_{name},i_sn_ref_r_{name}', real=True)
    i_sa_ref_i,i_sb_ref_i,i_sc_ref_i,i_sn_ref_i = sym.symbols(f'i_sa_ref_i_{name},i_sb_ref_i_{name},i_sc_ref_i_{name},i_sn_ref_i_{name}', real=True)
    v_ta_r,v_tb_r,v_tc_r,v_tn_r = sym.symbols(f'v_ta_r_{name},v_tb_r_{name},v_tc_r_{name},v_tn_r_{name}', real=True)
    v_ta_i,v_tb_i,v_tc_i,v_tn_i = sym.symbols(f'v_ta_i_{name},v_tb_i_{name},v_tc_i_{name},v_tn_i_{name}', real=True)


    ## dynamical states

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
    i_sa_ref = i_sa_ref_r + 1j*i_sa_ref_i
    i_sb_ref = i_sb_ref_r + 1j*i_sb_ref_i
    i_sc_ref = i_sc_ref_r + 1j*i_sc_ref_i

    ## POI voltages
    v_sa = v_sa_r + 1j*v_sa_i
    v_sb = v_sb_r + 1j*v_sb_i
    v_sc = v_sc_r + 1j*v_sc_i

    ## VSC AC side voltages
    v_ta = v_ta_r + 1j*v_ta_i
    v_tb = v_tb_r + 1j*v_tb_i
    v_tc = v_tc_r + 1j*v_tc_i
    v_tn = v_tn_r + 1j*v_tn_i

    ## powers
    s_a_ref = (p_ref/3+p_soc/3+p_a_ref) + sym.I*(q_ref/3+q_a_ref)
    s_b_ref = (p_ref/3+p_soc/3+p_b_ref) + sym.I*(q_ref/3+q_b_ref)
    s_c_ref = (p_ref/3+p_soc/3+p_c_ref) + sym.I*(q_ref/3+q_c_ref)

    Z_s = R_s + sym.I*X_s

    ## differential equations
     
    ## algebraic equations   
    eq_i_sa_ref_cplx =  s_a_ref - v_sa*np.conjugate(i_sa_ref)
    eq_i_sb_ref_cplx =  s_b_ref - v_sb*np.conjugate(i_sb_ref)
    eq_i_sc_ref_cplx =  s_c_ref - v_sc*np.conjugate(i_sc_ref)
    eq_v_ta_cplx =  v_ta - Z_s*i_sa_ref - v_sa 
    eq_v_tb_cplx =  v_tb - Z_s*i_sb_ref - v_sb 
    eq_v_tc_cplx =  v_tc - Z_s*i_sc_ref - v_sc 
 
    grid.dae['g'] += [sym.re(eq_i_sa_ref_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sb_ref_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_sc_ref_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_sa_ref_cplx)]
    grid.dae['g'] += [sym.im(eq_i_sb_ref_cplx)]
    grid.dae['g'] += [sym.im(eq_i_sc_ref_cplx)]

    grid.dae['g'] += [sym.re(eq_v_ta_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_tb_cplx)] 
    grid.dae['g'] += [sym.re(eq_v_tc_cplx)] 
    grid.dae['g'] += [sym.im(eq_v_ta_cplx)]
    grid.dae['g'] += [sym.im(eq_v_tb_cplx)]
    grid.dae['g'] += [sym.im(eq_v_tc_cplx)]

    grid.dae['y_ini'] += [i_sa_ref_r,i_sb_ref_r,i_sc_ref_r]
    grid.dae['y_ini'] += [i_sa_ref_i,i_sb_ref_i,i_sc_ref_i]
    grid.dae['y_run'] += [i_sa_ref_r,i_sb_ref_r,i_sc_ref_r]
    grid.dae['y_run'] += [i_sa_ref_i,i_sb_ref_i,i_sc_ref_i]

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
                      {f'p_a_ref_{name}':0.0,
                       f'p_b_ref_{name}':0.0,
                       f'p_c_ref_{name}':0.0,
                       f'p_soc_{name}':0.0})
    grid.dae['u_ini_dict'].update(
                      {f'q_a_ref_{name}':0.0,
                       f'q_b_ref_{name}':0.0,
                       f'q_c_ref_{name}':0.0,
                       f'q_ref_{name}':0.0})
    

    #u_dict.update({f'phi_{name}':0.0})
    grid.dae['u_run_dict'].update(
                      {f'p_a_ref_{name}':0.0,
                       f'p_b_ref_{name}':0.0,
                       f'p_c_ref_{name}':0.0,
                       f'p_soc_{name}':0.0})
    grid.dae['u_run_dict'].update(
                      {f'q_a_ref_{name}':0.0,
                       f'q_b_ref_{name}':0.0,
                       f'q_c_ref_{name}':0.0,
                       f'q_ref_{name}':0.0})

    if not p_ref in grid.dae['y_ini']:
        grid.dae['u_ini_dict'].update({f'p_ref_{name}':0.0})
        grid.dae['u_run_dict'].update({f'p_ref_{name}':0.0})


    ## parameters default values

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


    grid.dae['h_dict'].update({f'p_ref_{name}':p_ref})



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

    grid = urisi('ctrl_3ph_4w_pq_ib.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()    
    p_ref = 10e3
    q_ref = 0.0
    model.ini({'p_ref_A1':p_ref,'q_ref_A1':q_ref},'xy_0.json')
    
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()

    # S_n = model.get_value('S_n_A1')
    # assert model.get_value('omega_A1')  == pytest.approx(1)
    # assert model.get_value('omega_coi') == pytest.approx(1)
    # assert model.get_value('p_vsc_A1_a') == pytest.approx(p_c*S_n/3, rel=0.05)
    # assert model.get_value('p_vsc_A1_b') == pytest.approx(p_c*S_n/3, rel=0.05)
    # assert model.get_value('p_vsc_A1_c') == pytest.approx(p_c*S_n/3, rel=0.05)
    # assert model.get_value('p_A2') == pytest.approx(-p_c*S_n, rel=0.05)
    # assert model.get_value('q_A2') == pytest.approx(-q_ref*S_n, rel=0.05)

 

if __name__ == '__main__':

    #development()
    test_ib()