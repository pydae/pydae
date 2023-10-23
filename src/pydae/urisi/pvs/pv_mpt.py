import sympy as sym
import numpy as np

def pv_mpt(grid,data,name,bus_name): 
    '''

    PV+DCDC model MPPT control is implemented. 

    parameters
    ----------

    S_n: nominal power in VA
    U_n: nominal rms phase to phase voltage in V
    F_n: nominal frequency in Hz
    X_s: coupling reactance in pu (base machine S_n)
    R_s: coupling resistance in pu (base machine S_n)

    inputs
    ------

    p_s_ref: active power reference (pu, S_n base)
    q_s_ref: reactive power reference (pu, S_n base)
    v_dc: dc voltage in pu (when v_dc = 1 and m = 1, v_ac = 1)

    example
    -------

    "vscs": [{"bus":bus_name,"type":"pv_mpt_dcdc",
                 "S_n":1e6,"U_n":400.0,"F_n":50.0,
                 "X_s":0.1,"R_s":0.01,"monitor":True,
                 "I_sc":3.87,"V_oc":42.1,"I_mp":3.56,"V_mp":33.7,
                 "K_vt":-0.160,"K_it":0.065,
                 "N_pv_s":25,"N_pv_p":250}]
    
    '''
    
    ## inputs
    p_dc = sym.Symbol(f'p_dc_{name}',real=True)
    temp_deg,irrad = sym.symbols(f"temp_deg_{name},irrad_{name}", real=True)
    p_pv_ref  = sym.Symbol(f'p_pv_ref_{name}',real=True)
    p_r_ref   = sym.Symbol(f'p_r_ref_{name}',real=True)
    p_ref   = sym.Symbol(f'p_ref_{name}',real=True)

    ## parameters
    K_vt,K_it = sym.symbols(f"K_vt_{name},K_it_{name}", real=True)
    V_oc,V_mp,I_sc,I_mp = sym.symbols(f"V_oc_{name},V_mp_{name},I_sc_{name},I_mp_{name}", real=True)
    T_stc_k,i,v = sym.symbols(f"T_stc_k_{name},i_{name},v_{name}", real=True)
    v_dc,K_it = sym.symbols(f"v_dc_{name},K_it_{name}", real=True)
    N_pv_s,N_pv_p = sym.symbols(f"N_pv_s_{name},N_pv_p_{name}", real=True)
    K_mp_p,K_mp_i = sym.symbols(f"K_mp_p_{name},K_mp_i_{name}", real=True)
    T_mp,p_l_f = sym.symbols(f"T_mp_{name},p_l_f_{name}", real=True)
    p_l,xi_p = sym.symbols(f"p_l_{name},xi_p_{name}", real=True)

    T_stc_deg = 25.0

    V_oc_t = N_pv_s*V_oc * (1 + K_vt/100.0*(temp_deg - T_stc_deg))
    V_mp_t = N_pv_s*V_mp * (1 + K_vt/100.0*(temp_deg - T_stc_deg))
    I_sc_t = N_pv_p*I_sc * (1 + K_it/100.0*(temp_deg - T_stc_deg))
    I_mp_t = N_pv_p*I_mp * (1 + K_it/100.0*(temp_deg - T_stc_deg))
    I_mp_i = I_mp_t*irrad/1000.0

    v_1,i_1 = V_mp_t,I_mp_i
    v_2,i_2 = V_oc_t,0

    p_mp = V_mp_t*I_mp_i
    epsilon_p = p_mp - p_dc
    p_mppt_ref = K_mp_p*epsilon_p + K_mp_i*xi_p + p_mp
    mppt = sym.Piecewise((0.0,p_r_ref<p_mppt_ref),(1.0,p_r_ref>=p_mppt_ref))  
    
    dxi_p = epsilon_p*mppt - xi_p*(1-mppt)*1e-8  

    grid.dae['f'] += [dxi_p]
    grid.dae['x'] += [ xi_p]

    # (v_1 - v)/(v_1 - v_2) = (i_1 - i)/(i_1 - i_2)
    i_pv = p_dc/v_dc
    g_v_dc = -v_dc + v_1 - (i_1 - i_pv)*(v_1 - v_2)/(i_1 - i_2) 
    g_p_ref = - p_ref + mppt*p_mppt_ref + (1.0-mppt)*p_r_ref

    grid.dae['g'] +=     [g_v_dc, g_p_ref]
    grid.dae['y_ini'] += [v_dc, p_ref]  
    grid.dae['y_run'] += [v_dc, p_ref]  

    grid.dae['u_ini_dict'].update({f'{str(irrad)}':1000.0})
    grid.dae['u_run_dict'].update({f'{str(irrad)}':1000.0})

    grid.dae['u_ini_dict'].update({f'{str(temp_deg)}':25.0})
    grid.dae['u_run_dict'].update({f'{str(temp_deg)}':25.0})

    P_pv_n_N = data['N_pv_s']*data['V_mp']*data['N_pv_p']*data['I_mp']
    grid.dae['u_ini_dict'].update({f'{str(p_r_ref)}':1.2*P_pv_n_N})
    grid.dae['u_run_dict'].update({f'{str(p_r_ref)}':1.2*P_pv_n_N})

    grid.dae['params_dict'].update({
                   str(I_sc):data['I_sc'],
                   str(I_mp):data['I_mp'],
                   str(V_mp):data['V_mp'],
                   str(V_oc):data['V_oc'],
                   str(K_vt):data['K_vt'],
                   str(K_it):data['K_it'],
                   str(N_pv_s):data['N_pv_s'],
                   str(N_pv_p):data['N_pv_p'],
                   str(K_mp_p):data['K_mp_p'],
                   str(K_mp_i):data['K_mp_i']
                   })

    grid.dae['xy_0_dict'].update({f"v_dc_{name}":data['V_mp']*data['N_pv_s']})
    grid.dae['h_dict'].update({f"p_mp_{name}":p_mp})


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db
    import pytest

    grid = urisi('pv_mpt.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()

    with open('pv_mpt.hjson') as fobj:
        data = hjson.load(fobj)

    p_r_ref = 100e3
    q_ref =  50e3
    model.ini({'p_r_ref_A1':p_r_ref, 'q_ref_A1':q_ref,'N_pv_p_A1':100},'xy_0.json')
    
    assert model.get_value('p_A2') == pytest.approx(-p_r_ref, rel=0.05)
    assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)

    V_mp = data['vscs'][0]['pv']['V_mp']
    I_mp = data['vscs'][0]['pv']['I_mp']
    N_pv_s = 25
    N_pv_p = 100
    p_r_ref = 500e3
    q_ref = -50e3
    p_mp = N_pv_s*V_mp*N_pv_p*I_mp
    model.ini({'p_r_ref_A1':p_r_ref, 'q_ref_A1':q_ref,'N_pv_p_A1':N_pv_p},'xy_0.json')

    assert model.get_value('p_dc_A1') == pytest.approx(p_mp, rel=0.01)
    assert model.get_value('p_A2') == pytest.approx( -p_mp, rel=0.1)
    assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)


if __name__ == '__main__':

    #development()
    test()



