
import numpy as np
import sympy as sym

def add_breakers(grid,data):
    '''
    VSC with 3 phase and 4 wire with a PI-VSG with PFR and Q control.
         
    '''
    name1 = data['bus_1']
    name2 = data['bus_2']

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
    S_n,U_n  = sym.symbols(f'S_n_{name1},U_n_{name1}', real=True)
    R,X = sym.symbols(f'R_{name1},X_{name1}', real=True)

    ## inputs
    u_brk = sym.symbols(f'u_brk_{name1}', real=True)

    # algebraic states
    v_1a_r,v_1b_r,v_1c_r,v_1n_r = sym.symbols(f'V_{name1}_0_r,V_{name1}_1_r,V_{name1}_2_r,V_{name1}_3_r', real=True)
    v_1a_i,v_1b_i,v_1c_i,v_1n_i = sym.symbols(f'V_{name1}_0_i,V_{name1}_1_i,V_{name1}_2_i,V_{name1}_3_i', real=True)
    i_1a_r,i_1b_r,i_1c_r,i_1n_r = sym.symbols(f'i_brk_{name1}_a_r,i_brk_{name1}_b_r,i_brk_{name1}_c_r,i_brk_{name1}_n_r', real=True)
    i_1a_i,i_1b_i,i_1c_i,i_1n_i = sym.symbols(f'i_brk_{name1}_a_i,i_brk_{name1}_b_i,i_brk_{name1}_c_i,i_brk_{name1}_n_i', real=True)
    v_2a_r,v_2b_r,v_2c_r,v_2n_r = sym.symbols(f'V_{name2}_0_r,V_{name2}_1_r,V_{name2}_2_r,V_{name2}_3_r', real=True)
    v_2a_i,v_2b_i,v_2c_i,v_2n_i = sym.symbols(f'V_{name2}_0_i,V_{name2}_1_i,V_{name2}_2_i,V_{name2}_3_i', real=True)
    i_2a_r,i_2b_r,i_2c_r,i_2n_r = sym.symbols(f'i_brk_{name2}_a_r,i_brk_{name2}_b_r,i_brk_{name2}_c_r,i_brk_{name2}_n_r', real=True)
    i_2a_i,i_2b_i,i_2c_i,i_2n_i = sym.symbols(f'i_brk_{name2}_a_i,i_brk_{name2}_b_i,i_brk_{name2}_c_i,i_brk_{name2}_n_i', real=True)

    #e_om_r,e_om_i = sym.symbols(f'e_{name}_om_r,e_{name}_om_i', real=True)
    ## VSC impedances
    Z = R + 1j*X

    ## POI currents
    i_1a = i_1a_r + 1j*i_1a_i
    i_1b = i_1b_r + 1j*i_1b_i
    i_1c = i_1c_r + 1j*i_1c_i
    i_1n = i_1n_r + 1j*i_1n_i

    ## POI currents
    i_2a = i_2a_r + 1j*i_2a_i
    i_2b = i_2b_r + 1j*i_2b_i
    i_2c = i_2c_r + 1j*i_2c_i
    i_2n = i_2n_r + 1j*i_2n_i

    ## VSC AC side voltages
    v_1a = v_1a_r + 1j*v_1a_i
    v_1b = v_1b_r + 1j*v_1b_i
    v_1c = v_1c_r + 1j*v_1c_i
    v_1n = v_1n_r + 1j*v_1n_i
    
    ## VSC AC side voltages
    v_2a = v_2a_r + 1j*v_2a_i
    v_2b = v_2b_r + 1j*v_2b_i
    v_2c = v_2c_r + 1j*v_2c_i
    v_2n = v_2n_r + 1j*v_2n_i

    eq_i_1a_cplx = v_1a - i_1a*Z - v_2a   # v_sa = v_sag
    eq_i_1b_cplx = v_1b - i_1b*Z - v_2b
    eq_i_1c_cplx = v_1c - i_1c*Z - v_2c
    eq_i_1n_cplx = v_1n - i_1n*Z - v_2n

    grid.dae['g'] += [sym.re(eq_i_1a_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_1b_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_1c_cplx)] 
    grid.dae['g'] += [sym.re(eq_i_1n_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_1a_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_1b_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_1c_cplx)] 
    grid.dae['g'] += [sym.im(eq_i_1n_cplx)] 

    grid.dae['y_ini'] += [i_1a_r,i_1b_r,i_1c_r,i_1n_r]
    grid.dae['y_ini'] += [i_1a_i,i_1b_i,i_1c_i,i_1n_i]
    grid.dae['y_run'] += [i_1a_r,i_1b_r,i_1c_r,i_1n_r]
    grid.dae['y_run'] += [i_1a_i,i_1b_i,i_1c_i,i_1n_i]

    for ph in ['a','b','c','n']:  
        i_brk_r = sym.Symbol(f'i_brk_{name1}_{ph}_r', real=True)
        i_brk_i = sym.Symbol(f'i_brk_{name1}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(name1,ph)
        grid.dae['g'] [idx_r] += -i_brk_r*u_brk
        grid.dae['g'] [idx_i] += -i_brk_i*u_brk
        i_brk = i_brk_r + 1j*i_brk_i
        i_brk_m = np.abs(i_brk)
        grid.dae['h_dict'].update({f'i_brk_{name1}_{ph}_m':i_brk_m})

    for ph in ['a','b','c','n']:
        i_brk_r = sym.Symbol(f'i_brk_{name1}_{ph}_r', real=True)
        i_brk_i = sym.Symbol(f'i_brk_{name1}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(name2,ph)
        grid.dae['g'] [idx_r] +=  i_brk_r*u_brk
        grid.dae['g'] [idx_i] +=  i_brk_i*u_brk
        i_brk = i_brk_r + 1j*i_brk_i
        i_brk_m = np.abs(i_brk)
        grid.dae['h_dict'].update({f'i_brk_{name2}_{ph}_m':i_brk_m})

    grid.dae['u_ini_dict'].update({f'u_brk_{name1}':1.0})
    grid.dae['u_run_dict'].update({f'u_brk_{name1}':1.0})

    grid.dae['params_dict'].update({f'R_{name1}':1e-4})
    grid.dae['params_dict'].update({f'X_{name1}':1e-4})

def test_ib_2src():
    '''
    test grid former connected to infinite bus
    '''
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db
    import pytest

    grid = urisi('breaker.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()    
    model.ini({
                'p_load_A2_a':100e3,
                'q_load_A2_a':10e3,
                'p_load_A2_b':100e3,
                'q_load_A2_b':10e3,
                'p_load_A2_c':100e3,
                'q_load_A2_c':10e3,
                'u_brk_A2':0.0
                },'xy_0.json')
    
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
    test_ib_2src()