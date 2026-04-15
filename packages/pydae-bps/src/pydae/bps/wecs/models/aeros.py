# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def aero(grid,name,bus_name,data_dict):
    '''
    ## Aerodinamic


    parameters
    ----------

    S_n: nominal power in VA
    F_n: nominal frequency in Hz
    X_v: coupling reactance in pu (base machine S_n)
    R_v: coupling resistance in pu (base machine S_n)
    K_delta: if K_delta>0.0 current generator is converted to reference machine 
    K_alpha: alpha gain to obtain Domega integral 

    inputs
    ------

    alpha: RoCoF in pu if K_alpha = 1.0
    omega_ref: frequency in pu
    v_ref: internal voltage reference

    example
    -------

    "vscs": [{"type":"vsc_l","S_n":1e6,"F_n":50.0,"X_s":0.1,"R_s":0.01}]

    v_t = m*v_dc/sqrt(6)

    v_t_pu*v_b_ac = m*v_dc_pu*v_b_dc/sqrt(6)

    S_b = S_n
    U_b = U_n
    Omega_b = 2*np.pi*F_b
    Omega_rb = Omega_rn
    Tau_b = S_b/Omega_rb
    V_dc_b = sqrt(2)*U_b
    I_b = (np.sqrt(3)*U_b)
    I_dq_b = I_b*np.sqrt(2)
    V_dq_b = U_b*np.sqrt(2/3)

    V_dq_b and I_dq_b defined as below verify that: 
    3/2*I_dq_b*V_dq_b = 3/2*I_b*np.sqrt(2)*U_b*np.sqrt(2/3) = np.sqrt(3)*U_b*I_b

    Z_b = U_b**2/S_b
    Z_b = (V_dq_b/np.sqrt(2/3))**2/S_b = 3/2*V_dq_b**2/S_b
    L_m_b = Z_b/Omega_rb

    tau_r = 3/2*Phi*N_pp*i_mq

    tau_r_pu = (3/2*Phi*N_pp*i_mq_pu)*I_dq_b/Tau_b  = K_tau * i_mq_pu
    with K_tau = 3/2*Phi*N_pp*I_dq_b/Tau_b

    0 =               - R_s*i_md - omega_r*L_m*i_mq - v_md  
    0 = omega_r*Phi_m - R_s*i_mq + omega_r*L_m*i_md - v_mq 

    0 =               (- R_s*i_md_pu/V_dq_b - omega_r*L_m*i_mq_pu/V_dq_b - v_md_pu/I_dq_b)*I_dq_b*V_dq_b
    0 =               (- R_s_pu*i_md_pu/V_dq_b - omega_r*L_m_pu*i_mq_pu*Z_b/Omega_rb/(V_dq_b*Z_b) - v_md_pu/(I_dq_b*Z_b))*I_dq_b*V_dq_b*Z_b
    0 =               (- R_s_pu*i_md_pu/V_dq_b - omega_r_pu*L_m_pu*i_mq_pu/(V_dq_b) - v_md_pu/(I_dq_b*Z_b))*I_dq_b*V_dq_b*Z_b


    v_t_pu*v_b_ac = m*V_dc_b*sqrt(2)*U_b/sqrt(6)
    v_t_pu = m*V_dc_b*sqrt(2)/sqrt(6) = m*V_dc_b*sqrt(3)


    

    '''


    # inputs
    dnu_w_ramp_ref = sym.Symbol(f"dnu_w_ramp_ref_{name}", real=True)
    omega_t  = sym.Symbol(f"omega_t_{name}", real=True)
    beta     = sym.Symbol(f"beta_{name}", real=True)
    nu_w_ramp= sym.Symbol(f"nu_w_ramp_{name}", real=True) 
    grid.dae['u_ini_dict'].update({f'dnu_w_ramp_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'dnu_w_ramp_ref_{name}':0.0})  
    grid.dae['u_ini_dict'].update({f'nu_w_0_{name}':10.0})
    grid.dae['u_run_dict'].update({f'nu_w_0_{name}':10.0})  

    # algebraic states
    nu_w_0 = sym.Symbol(f"nu_w_0_{name}", real=True)
    
    # dynamic states
    nu_w_ramp = sym.Symbol(f"nu_w_ramp_{name}", real=True)

    # parameters
    C_1,C_2,C_3 = sym.symbols(f"C_1_{name},C_2_{name},C_3_{name}", real=True)
    C_4,C_5,C_6 = sym.symbols(f"C_4_{name},C_5_{name},C_6_{name}", real=True)
    Nu_w_b,Lam_b = sym.symbols(f"Nu_w_b_{name},Lam_b_{name}", real=True)
    Omega_t_b = sym.symbols(f"Omega_t_b_{name}", real=True)
    C_p_b,K_pow = sym.symbols(f"C_p_b_{name},K_pow_{name}", real=True)
    C_1 = 0.5176
    C_2 = 116
    C_3 = 0.4
    C_4 = 5.0
    C_5 = 21 
    C_6 = 0.0068
    Nu_w_b = 12
    Lam_b = 8.1
    Omega_t_b = 1.2
    K_pow = 1.0
    grid.dae['params_dict'].update({f"C_1_{name}":C_1})
    grid.dae['params_dict'].update({f"C_2_{name}":C_2})
    grid.dae['params_dict'].update({f"C_3_{name}":C_3})
    grid.dae['params_dict'].update({f"C_4_{name}":C_4})
    grid.dae['params_dict'].update({f"C_5_{name}":C_5})
    grid.dae['params_dict'].update({f"C_6_{name}":C_6})
    grid.dae['params_dict'].update({f"Nu_w_b_{name}":Nu_w_b})
    grid.dae['params_dict'].update({f"Lam_b_{name}":Lam_b})
    grid.dae['params_dict'].update({f"Omega_t_b_{name}":Omega_t_b})
    grid.dae['params_dict'].update({f"K_pow_{name}":K_pow})

    # auxiliar
    nu_w = nu_w_0 + nu_w_ramp
    lam = Lam_b*(omega_t/Omega_t_b)/(nu_w/Nu_w_b) # pu
    inv_lam_i =  1/(lam + 0.08*beta) - 0.035/(beta**3 + 1.0)   
    c_p = C_1*(C_2*inv_lam_i - C_3*beta - C_4)*sym.exp(-C_5*inv_lam_i) + C_6*lam 
    c_p_pu = c_p/C_p_b # (pu)
    p_w_nosat = K_pow*c_p_pu*(nu_w/Nu_w_b)**3 
    p_w = sym.Piecewise((0.0,p_w_nosat<0.0), (p_w_nosat,True)) 

    # dynamic
    dnu_w_ramp = dnu_w_ramp_ref - 0.0001*nu_w_ramp

    grid.dae['f'] += [dnu_w_ramp]
    grid.dae['x'] += [ nu_w_ramp]


    # outputs
    grid.dae['h_dict'].update({f"nu_w_{name}":nu_w})

    return p_w


def test_build():

    import sympy as sym
    from pydae.build_v2 import builder


    dae = {'f':[],'g':[],'x':[],'y_ini':[],'y_run':[],
                    'u_ini_dict':{},'u_run_dict':{},'params_dict':{},
                    'h_dict':{},'xy_0_dict':{},'name':'temp'}
    name = ''
    bus_name = ''
    data_dict = {}
    grid = type('grid_class', (object,), {'dae':dae})()

    
    p_w = aero(grid,name,bus_name,data_dict)

    sys_dict = {'name':'temp','uz_jacs':True,
                    'params_dict':grid.dae['params_dict'],
                    'f_list':grid.dae['f'],
                    'g_list':grid.dae['g'] ,
                    'x_list':grid.dae['x'],
                    'y_ini_list':grid.dae['y_ini'],
                    'y_run_list':grid.dae['y_run'],
                    'u_run_dict':grid.dae['u_run_dict'],
                    'u_ini_dict':grid.dae['u_ini_dict'],
                    'h_dict':grid.dae['h_dict']}
    b = builder(sys_dict,verbose=True)
    b.sparse = True
    b.mkl = True
    b.uz_jacs = True
    b.dict2system()
    b.functions()
    b.jacobians()
    b.cwrite()
    b.template()
    b.compile_mkl()  


if __name__ == '__main__':
    test_build()
    import temp
    model = temp.model()
    model.ini({})