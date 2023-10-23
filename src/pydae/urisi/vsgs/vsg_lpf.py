
import numpy as np
import sympy as sym

def vsg_lpf(grid,data):
    '''
    VSG based on Low Pass Filter    
    '''

    name = data['bus']
    S_n, U_n = sym.symbols(f'S_n_{name}, U_n_{name}', real=True)
    i_dc, v_dc = sym.symbols(f'i_dc_{name}, v_dc_{name}', real=True)
    p_pos, q_pos = sym.symbols(f'p_pos_{name}, q_pos_{name}', real=True)
    v_sa_r,v_sb_r,v_sc_r,v_sn_r,v_og_r = sym.symbols(f'V_{name}_0_r,V_{name}_1_r,V_{name}_2_r,V_{name}_3_r,v_{name}_o_r', real=True)
    v_sa_i,v_sb_i,v_sc_i,v_sn_i,v_og_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i,V_{name}_2_i,V_{name}_3_i,v_{name}_o_i', real=True)

    omega_coi = sym.Symbol('omega_coi', real=True)

    phi_nosat, phi, omega = sym.symbols(f'phi_nosat_{name},phi_{name},omega_{name}', real=True)
    m,m_a,m_b,m_c = sym.symbols(f'm_{name},m_a_{name},m_b_{name},m_c_{name}', real=True)

    H,D = sym.symbols(f'H_{name},D_{name}', real=True)
    K_delta,K_f,K_sec = sym.symbols(f'K_delta_{name},K_f_{name},K_sec_{name}', real=True)
    p_agc = sym.Symbol(f'p_agc', real=True) 
    p_c = sym.Symbol(f'p_c_{name}', real=True) 
    q_ref = sym.Symbol(f'q_ref_{name}', real=True) 
    K_soc,p_soc = sym.symbols(f'K_soc_{name},p_soc_{name}', real=True)
    K_v = sym.Symbol(f'K_v_{name}', real=True)
    phi_v  = sym.Symbol(f'phi_v_{name}', real=True)
    xi_q  = sym.Symbol(f'xi_q_{name}', real=True)
    e_m = sym.Symbol(f'e_m_{name}', real=True) 

    v_sa = v_sa_r +  sym.I*v_sa_i
    s_pos = p_pos + sym.I*q_pos
    i_pos = sym.conjugate(s_pos/(3*v_sa))
    e = e_m*sym.exp(sym.I*phi_v)
    v_t = e #+ sym.I*1e-6*i_pos
    m_cplx = v_t/(400.0/np.sqrt(3))
    m_ref = sym.sqrt(sym.re(m_cplx)**2 + sym.im(m_cplx)**2)*0.715 
    m_phi = sym.atan2(sym.im(m_cplx),sym.re(m_cplx))

    p_m = K_f*S_n*(1 - omega) + K_sec*S_n*p_agc + K_soc*p_soc + p_c
    epsilon_q = q_ref - q_pos

    ## derivatives
    dphi_nosat =  2.0*np.pi*50.0*(omega - omega_coi) - K_delta*phi_nosat - 10*(phi_nosat - phi_v)
    domega = 1/(2*H*S_n)*(p_m - p_pos - D*(omega - 1.0))
    dxi_q = epsilon_q  - 1e-6*xi_q

    grid.dae['f'] += [dphi_nosat, domega, dxi_q]
    grid.dae['x'] += [ phi_nosat,  omega, xi_q]

    # 0.70 - K_v*q_pos/S_n
    grid.dae['g'] += [ m_ref - m]
    grid.dae['g'] += [m - m_a]
    grid.dae['g'] += [m - m_b]
    grid.dae['g'] += [m - m_c]
    grid.dae['g'] += [-phi_v + sym.Piecewise((phi_nosat,(phi_nosat>-2*np.pi)&(phi_nosat<2*np.pi)),(0,True))]
    grid.dae['g'] += [-phi + m_phi]
    grid.dae['g'] += [-e_m + 400.0/np.sqrt(3) - (0.0*epsilon_q/S_n + 0.0*xi_q/S_n) ]
    grid.dae['y_ini'] += [m,m_a,m_b,m_c,phi_v,phi,e_m]
    grid.dae['y_run'] += [m,m_a,m_b,m_c,phi_v,phi,e_m]

    for ph in ['a','b','c']:
        grid.dae['u_ini_dict'].pop(f'm_{ph}_{name}')
        grid.dae['u_run_dict'].pop(f'm_{ph}_{name}')

    grid.dae['u_ini_dict'].pop(f'phi_{name}')
    grid.dae['u_run_dict'].pop(f'phi_{name}')
    
    grid.dae['u_ini_dict'].update({f'p_soc_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_soc_{name}':0.0})

    grid.dae['u_ini_dict'].update({f'p_c_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_c_{name}':0.0})

    grid.dae['u_ini_dict'].update({f'q_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_ref_{name}':0.0})

    grid.dae['h_dict'].update({f'p_m_{name}':p_m})
    grid.dae['h_dict'].update({f'm_phi_{name}':m_phi})
    grid.dae['h_dict'].update({f'm_ref_{name}':m_ref})

    grid.dae['xy_0_dict'].update({f"v_dc_{name}":800})
    grid.dae['xy_0_dict'].update({f"omega_{name}":1.0})
    grid.dae['xy_0_dict'].update({f"e_m_{name}":231})

    for item in ['K_f','H','D','K_delta','K_sec','K_v']:
        grid.dae['params_dict'].update({f'{item}_{name}':data[item]})

    grid.dae['params_dict'].update({f'K_soc_{name}':0.0})

    omega_coi_i = S_n*omega
    HS_coi  = S_n
    

    grid.omega_coi_numerator += omega_coi_i
    grid.omega_coi_denominator += HS_coi  
    
def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('vsg_lpf.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()
    model.ini({'p_load_A2_a':20e3,'q_load_A2_a':-10e3,
               'p_load_A2_b':20e3,'q_load_A2_b':-10e3,
               'p_load_A2_c':20e3,'q_load_A2_c':-10e3,},'xy_0.json')
    
    #model.ini({'p_c_A1':0e3},'xy_0.json')
    
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()


if __name__ == '__main__':

    #development()
    test()
