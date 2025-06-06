import sympy as sym
import numpy as np

from pydae.urisi.vscs.ac_3ph_4w_l import ac_3ph_4w_l

def ac_3ph_4w_pq(grid,vsc_data):
    '''
    Converter type p_ac,q_ac 3 phase 4 wire without DC connection to the grid
    
    '''
    
    bus_ac_name = vsc_data['bus']
    bus_dc_name = vsc_data['bus']  
    name = bus_ac_name


    ac_3ph_4w_l(grid,vsc_data)

    v_san = grid.aux[f'ac_3ph_4w_l_{name}']['v_san']
    v_sbn = grid.aux[f'ac_3ph_4w_l_{name}']['v_sbn']
    v_scn = grid.aux[f'ac_3ph_4w_l_{name}']['v_scn']

    Z_sa = grid.aux[f'ac_3ph_4w_l_{name}']['Z_sa']
    Z_sb = grid.aux[f'ac_3ph_4w_l_{name}']['Z_sb']
    Z_sc = grid.aux[f'ac_3ph_4w_l_{name}']['Z_sc']

    v_dc = grid.aux[f'ac_3ph_4w_l_{name}']['v_dc']
    v_og = grid.aux[f'ac_3ph_4w_l_{name}']['v_og']

    i_sa_r_ref,i_sb_r_ref,i_sc_r_ref = sym.symbols(f'i_sa_r_ref_{name},i_sb_r_ref_{name},i_sc_r_ref_{name}', real=True)
    i_sa_i_ref,i_sb_i_ref,i_sc_i_ref = sym.symbols(f'i_sa_i_ref_{name},i_sb_i_ref_{name},i_sc_i_ref_{name}', real=True)
    m_a_ref,m_b_ref,m_c_ref = sym.symbols(f'm_a_{name},m_b_{name},m_c_{name}', real=True)
    phi_a_ref,phi_b_ref,phi_c_ref = sym.symbols(f'phi_a_{name},phi_b_{name},phi_c_{name}', real=True)

    p_sa_ref,p_sb_ref,p_sc_ref = sym.symbols(f'p_sa_ref_{name},p_sb_ref_{name},p_sc_ref_{name}', real=True)
    q_sa_ref,q_sb_ref,q_sc_ref = sym.symbols(f'q_sa_ref_{name},q_sb_ref_{name},q_sc_ref_{name}', real=True)


    sqrt6 = np.sqrt(6)
    e_ao_r = (m_a_ref*v_dc/sqrt6)*sym.cos(phi_a_ref) 
    e_ao_i = (m_a_ref*v_dc/sqrt6)*sym.sin(phi_a_ref) 
    e_bo_r = (m_b_ref*v_dc/sqrt6)*sym.cos(phi_b_ref-2/3*np.pi) 
    e_bo_i = (m_b_ref*v_dc/sqrt6)*sym.sin(phi_b_ref-2/3*np.pi) 
    e_co_r = (m_c_ref*v_dc/sqrt6)*sym.cos(phi_c_ref-4/3*np.pi) 
    e_co_i = (m_c_ref*v_dc/sqrt6)*sym.sin(phi_c_ref-4/3*np.pi) 

    e_ao_cplx = e_ao_r + 1j*e_ao_i
    e_bo_cplx = e_bo_r + 1j*e_bo_i
    e_co_cplx = e_co_r + 1j*e_co_i

    i_sa_ref = i_sa_r_ref + 1j*i_sa_i_ref
    i_sb_ref = i_sb_r_ref + 1j*i_sb_i_ref
    i_sc_ref = i_sc_r_ref + 1j*i_sc_i_ref

    s_sa_ref = p_sa_ref + 1j*q_sa_ref
    s_sb_ref = p_sb_ref + 1j*q_sb_ref
    s_sc_ref = p_sc_ref + 1j*q_sc_ref
 
    eq_i_sa_ref =  v_san * sym.conjugate(i_sa_ref) - s_sa_ref
    eq_i_sb_ref =  v_sbn * sym.conjugate(i_sb_ref) - s_sb_ref
    eq_i_sc_ref =  v_scn * sym.conjugate(i_sc_ref) - s_sc_ref
     

    eq_m_phi_a_cplx = v_og + e_ao_cplx - i_sa_ref*Z_sa - v_san   # v_sa = v_sag
    eq_m_phi_b_cplx = v_og + e_bo_cplx - i_sb_ref*Z_sb - v_sbn
    eq_m_phi_c_cplx = v_og + e_co_cplx - i_sc_ref*Z_sc - v_scn

    grid.dae['g'] += [sym.re(eq_i_sa_ref)] 
    grid.dae['g'] += [sym.re(eq_i_sb_ref)] 
    grid.dae['g'] += [sym.re(eq_i_sc_ref)] 

    grid.dae['g'] += [sym.im(eq_i_sa_ref)] 
    grid.dae['g'] += [sym.im(eq_i_sb_ref)] 
    grid.dae['g'] += [sym.im(eq_i_sc_ref)] 

    grid.dae['g'] += [sym.re(eq_m_phi_a_cplx)] 
    grid.dae['g'] += [sym.re(eq_m_phi_b_cplx)] 
    grid.dae['g'] += [sym.re(eq_m_phi_c_cplx)] 

    grid.dae['g'] += [sym.im(eq_m_phi_a_cplx)] 
    grid.dae['g'] += [sym.im(eq_m_phi_b_cplx)] 
    grid.dae['g'] += [sym.im(eq_m_phi_c_cplx)] 

    grid.dae['y_ini']  += [i_sa_r_ref,i_sb_r_ref,i_sc_r_ref ]
    grid.dae['y_ini']  += [i_sa_i_ref,i_sb_i_ref,i_sc_i_ref ]

    grid.dae['y_run']  += [i_sa_r_ref,i_sb_r_ref,i_sc_r_ref ]
    grid.dae['y_run']  += [i_sa_i_ref,i_sb_i_ref,i_sc_i_ref ]


    grid.dae['y_ini']  += [m_a_ref,m_b_ref,m_c_ref ]
    grid.dae['y_ini']  += [phi_a_ref,phi_b_ref,phi_c_ref ]

    grid.dae['y_run']  += [m_a_ref,m_b_ref,m_c_ref ]
    grid.dae['y_run']  += [phi_a_ref,phi_b_ref,phi_c_ref ]


    m = 0.7071
    grid.dae['u_ini_dict'].pop(f'm_a_{name}')
    grid.dae['u_ini_dict'].pop(f'm_b_{name}')
    grid.dae['u_ini_dict'].pop(f'm_c_{name}')

    grid.dae['u_run_dict'].pop(f'm_a_{name}')
    grid.dae['u_run_dict'].pop(f'm_b_{name}')
    grid.dae['u_run_dict'].pop(f'm_c_{name}')


    grid.dae['u_ini_dict'].pop(f'phi_a_{name}')
    grid.dae['u_ini_dict'].pop(f'phi_b_{name}')
    grid.dae['u_ini_dict'].pop(f'phi_c_{name}')

    grid.dae['u_run_dict'].pop(f'phi_a_{name}')
    grid.dae['u_run_dict'].pop(f'phi_b_{name}')
    grid.dae['u_run_dict'].pop(f'phi_c_{name}')


    grid.dae['u_ini_dict'].update({f'{str(p_sa_ref)}':0.0,f'{str(p_sb_ref)}':0.0,f'{str(p_sc_ref)}':0.0}) 
    grid.dae['u_run_dict'].update({f'{str(p_sa_ref)}':0.0,f'{str(p_sb_ref)}':0.0,f'{str(p_sc_ref)}':0.0}) 

    grid.dae['u_ini_dict'].update({f'{str(q_sa_ref)}':0.0,f'{str(q_sb_ref)}':0.0,f'{str(q_sc_ref)}':0.0}) 
    grid.dae['u_run_dict'].update({f'{str(q_sa_ref)}':0.0,f'{str(q_sb_ref)}':0.0,f'{str(q_sc_ref)}':0.0}) 

    grid.dae['xy_0_dict'].update({f'{m_a_ref}':0.75,f'{m_b_ref}':0.75,f'{m_c_ref}':0.75})
            

def test_build():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi

    grid = urisi('ac_3ph_4w_pq_ib.hjson')
    grid.uz_jacs = False
    grid.build('temp')

    print(grid.dae['y_ini'])


def test_ini():

    import temp

    # S_n = 100e3
    # V_n = 1000
    # I_n = S_n/V_n
    # Conduction_losses = 0.02*S_n # = A*I_n**2
    # A = Conduction_losses/(I_n**2)
    # B = 1
    # C = 0.02*S_n
    model = temp.model()
    m = 0.75
    phi = 0.1
    model.ini({
               #'A_loss_A1':A,'B_loss_A1':B,'C_loss_A1':C,
               #'m_a_A1':m,'m_b_A1':m,'m_c_A1':m,
               #'phi_a_A1':phi,'phi_b_A1':phi,'phi_c_A1':phi,
               'p_sa_ref_A1':50e3,'p_sb_ref_A1':0e3,'p_sc_ref_A1':0e3
               },'xy_0.json')
    model.report_y()
    model.report_z()



if __name__ == '__main__':

    #development()
    test_build()
    test_ini()