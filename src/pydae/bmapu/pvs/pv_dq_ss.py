# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.utils.ss_num2sym import ss_num2sym

def pv_dq_ss(grid,name,bus_name,data_dict):
    '''

    VSC model with L filter coupling and purely algebraic.
    PQ control is implemented. 

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

    "vscs": [{"bus":bus_name,"type":"pv_pq",
                 "S_n":1e6,"U_n":400.0,"F_n":50.0,
                 "X_s":0.1,"R_s":0.01,"monitor":True,
                 "I_sc":3.87,"V_oc":42.1,"I_mp":3.56,"V_mp":33.7,
                 "K_vt":-0.160,"K_it":0.065,
                 "N_pv_s":25,"N_pv_p":250}]
    
    '''

    sin = sym.sin
    cos = sym.cos

    ## Common
    ### inputs
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)
    p_s_ppc = sym.Symbol(f"p_s_ppc_{name}", real=True)
    q_s_ppc = sym.Symbol(f"q_s_ppc_{name}", real=True)

    ### parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    U_n = sym.Symbol(f"U_n_{name}", real=True)
    V_dc_b = U_n*np.sqrt(2)
    X_s = sym.Symbol(f"X_s_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)

    ## PV
    K_vt,K_it = sym.symbols(f"K_vt_{name},K_it_{name}", real=True)
    V_oc,V_mp,I_sc,I_mp = sym.symbols(f"V_oc_{name},V_mp_{name},I_sc_{name},I_mp_{name}", real=True)
    temp_deg,irrad = sym.symbols(f"temp_deg_{name},irrad_{name}", real=True)
    T_stc_k,i,v = sym.symbols(f"T_stc_k_{name},i_{name},v_{name}", real=True)
    v_dc,v_dc_v,K_it = sym.symbols(f"v_dc_{name},v_dc_v_{name},K_it_{name}", real=True)
    N_pv_s,N_pv_p = sym.symbols(f"N_pv_s_{name},N_pv_p_{name}", real=True)

    T_stc_deg = 25.0

    V_oc_t = N_pv_s*V_oc * (1 + K_vt/100.0*(temp_deg - T_stc_deg))
    V_mp_t = N_pv_s*V_mp * (1 + K_vt/100.0*(temp_deg - T_stc_deg))
    I_sc_t = N_pv_p*I_sc * (1 + K_it/100.0*(temp_deg - T_stc_deg))
    I_mp_t = N_pv_p*I_mp * (1 + K_it/100.0*(temp_deg - T_stc_deg))
    I_mp_i = I_mp_t*irrad/1000.0

    v_1,i_1 = V_mp_t,I_mp_i
    v_2,i_2 = V_oc_t,0

    # (v_1 - v)/(v_1 - v_2) = (i_1 - i)/(i_1 - i_2)
    i_pv = p_s*S_n/(v_dc*V_dc_b)
    p_mp = (V_mp_t*I_mp_i)/S_n
    #v_dc_v = v_1 - (i_1 - i_pv)*(v_1 - v_2)/(i_1 - i_2) 
    v_dc_v =  v_1 - (i_1 - i_pv)*(v_1 - v_2)/(i_1 - i_2) 
    g_v_dc = -v_dc + v_dc_v/V_dc_b

    grid.dae['g'] +=     [g_v_dc]
    grid.dae['y_ini'] += [  v_dc]  
    grid.dae['y_run'] += [  v_dc]  

    grid.dae['u_ini_dict'].update({f'{str(irrad)}':1000.0})
    grid.dae['u_run_dict'].update({f'{str(irrad)}':1000.0})

    grid.dae['u_ini_dict'].update({f'{str(temp_deg)}':25.0})
    grid.dae['u_run_dict'].update({f'{str(temp_deg)}':25.0})

    grid.dae['params_dict'].update({
                   str(I_sc):data_dict['I_sc'],
                   str(I_mp):data_dict['I_mp'],
                   str(V_mp):data_dict['V_mp'],
                   str(V_oc):data_dict['V_oc'],
                   str(N_pv_s):data_dict['N_pv_s'],
                   str(N_pv_p):data_dict['N_pv_p'],
                   str(K_vt):data_dict['K_vt'],
                   str(K_it):data_dict['K_it']
                   })

   
    
    #grid.dae['xy_0_dict'].update({f"v_dc_v_{name}":data_dict['V_mp']*data_dict['N_pv_s']})
    grid.dae['xy_0_dict'].update({f"v_dc_{name}":1.5})

    ## VSC control 
    ### inputs
    p_s_ref = sym.Symbol(f"p_s_ref_{name}", real=True)
    q_s_ref = sym.Symbol(f"q_s_ref_{name}", real=True)
    i_sa_ref = sym.Symbol(f"i_sa_ref_{name}", real=True)
    i_sr_ref = sym.Symbol(f"i_sr_ref_{name}", real=True)

    ### dynamic states

    ### algebraic states
    i_sd_pq_ref,i_sq_pq_ref = sym.symbols(f'i_sd_pq_ref_{name},i_sq_pq_ref_{name}', real=True)
    i_sd_ar_ref,i_sq_ar_ref = sym.symbols(f'i_sd_ar_ref_{name},i_sq_ar_ref_{name}', real=True)
    i_sd_ref,i_sq_ref = sym.symbols(f'i_sd_ref_{name},i_sq_ref_{name}', real=True)
    i_sd_ref,i_sq_ref = sym.symbols(f'i_sd_ref_{name},i_sq_ref_{name}', real=True)

    v_td_ref,v_tq_ref = sym.symbols(f'v_td_ref_{name},v_tq_ref_{name}', real=True)
    v_lvrt,lvrt_ext = sym.symbols(f"v_lvrt_{name},lvrt_ext_{name}", real=True)

    x_p_1, x_p_2 = sym.symbols(f"x_p_1_{name},x_p_2_{name}", real=True)
    A_11p,A_12p,A_21p,A_22p = sym.symbols(f"A_11p_{name},A_12p_{name},A_21p_{name},A_22p_{name}", real=True) 
    B_11p,B_21p = sym.symbols(f"B_11p_{name},B_21p_{name}", real=True) 
    C_11p,C_12p = sym.symbols(f"C_11p_{name},C_12p_{name}", real=True) 
    D_11p = sym.symbols(f"D_11p_{name}", real=True) 

    x_q_1, x_q_2 = sym.symbols(f"x_q_1_{name},x_q_2_{name}", real=True) 
    A_11q,A_12q,A_21q,A_22q = sym.symbols(f"A_11q_{name},A_12q_{name},A_21q_{name},A_22q_{name}", real=True) 
    B_11q,B_21q = sym.symbols(f"B_11q_{name},B_12q_{name}", real=True) 
    C_11q,C_12q = sym.symbols(f"C_11q_{name},C_12q_{name}", real=True) 
    D_11q = sym.symbols(f"D_11q_{name}", real=True) 


    ### parameters
    
    ### auxiliar
    delta = theta_s # ideal PLL
    v_sD = V_s*sin(theta_s)  # v_si   e^(-j)
    v_sQ = V_s*cos(theta_s)  # v_sr
    v_sd = v_sD * cos(delta) - v_sQ * sin(delta)   
    v_sq = v_sD * sin(delta) + v_sQ * cos(delta)

    v_m = sym.sqrt(v_sd**2 + v_sq**2)
    lvrt = sym.Piecewise((0.0,v_m>=v_lvrt),(1.0,v_m<v_lvrt)) + lvrt_ext

    A_p = A_q = np.array([[-10.0]])
    B_p = B_q = np.array([[-10.0]])
    C_p = C_q = np.array([[1.0]])
    D_p = D_q = np.array([[0.0]])

    if 'A_p' in data_dict:
        A_p = np.array(data_dict['A_p'])
        B_p = np.array(data_dict['B_p'])
        C_p = np.array(data_dict['C_p'])
        D_p = np.array(data_dict['D_p'])

    if 'A_q' in data_dict:
        A_q = np.array(data_dict['A_q'])
        B_q = np.array(data_dict['B_q'])
        C_q = np.array(data_dict['C_q'])
        D_q = np.array(data_dict['D_q'])

    sys_p = ss_num2sym(f'p_{name}',A_p,B_p,C_p,D_p)
    sys_p['dx']= sys_p['dx'].replace(sys_p['u'][0],p_s_ppc)
    sys_p['z_evaluated']= sys_p['z_evaluated'].replace(sys_p['u'][0],p_s_ppc)

    sys_q = ss_num2sym(f'q_{name}',A_q,B_q,C_q,D_q)
    sys_q['dx']= sys_q['dx'].replace(sys_q['u'][0],q_s_ppc)
    sys_q['z_evaluated']= sys_q['z_evaluated'].replace(sys_q['u'][0],q_s_ppc)


    p_s_ppc_d = sys_p['z_evaluated'][0,0]
    q_s_ppc_d = sys_q['z_evaluated'][0,0]

    p_s_ref = sym.Piecewise((p_s_ppc_d,p_s_ppc_d<p_mp),(p_mp,p_s_ppc_d>=p_mp))
    q_s_ref = q_s_ppc_d


    ### dynamic equations            
    grid.dae['f'] += list(sys_p['dx']) + list(sys_q['dx'])
    grid.dae['x'] += list(sys_p['x']) + list(sys_q['x'])
 
 
    ### algebraic equations   
    #g_i_sd_pq_ref  = i_sd_pq_ref*v_sd + i_sq_pq_ref*v_sq - p_s_ref  
    #g_i_sq_pq_ref  =-i_sq_pq_ref*v_sd + i_sd_pq_ref*v_sq - q_s_ref 
    #g_i_sd_ar_ref  = i_sd_ar_ref*v_sd/v_m + i_sq_ar_ref*v_sq/v_m - i_sa_ref  
    #g_i_sq_ar_ref  =-i_sq_ar_ref*v_sd/v_m + i_sd_ar_ref*v_sq/v_m - i_sr_ref 
    #g_v_td_ref  = v_td_ref - R_s*i_sd_ref + X_s*i_sq_ref - v_sd  
    #g_v_tq_ref  = v_tq_ref - R_s*i_sq_ref - X_s*i_sd_ref - v_sq 

    i_sd_ar_ref = i_sa_ref*v_sd/sym.sqrt(v_sd**2 + v_sq**2) + i_sr_ref*v_sq/sym.sqrt(v_sd**2 + v_sq**2) 
    i_sq_ar_ref = i_sa_ref*v_sq/sym.sqrt(v_sd**2 + v_sq**2) - i_sr_ref*v_sd/sym.sqrt(v_sd**2 + v_sq**2)

    i_sd_pq_ref = (p_s_ref*v_sd + q_s_ref*v_sq)/(v_sd**2 + v_sq**2)
    i_sq_pq_ref = (p_s_ref*v_sq - q_s_ref*v_sd)/(v_sd**2 + v_sq**2)
    i_sd_ref_nosat = (1.0-lvrt)*i_sd_pq_ref + lvrt*i_sd_ar_ref
    i_sq_ref_nosat = (1.0-lvrt)*i_sq_pq_ref + lvrt*i_sq_ar_ref
    g_i_sd_ref = -i_sd_ref + sym.Piecewise((-1.2,i_sd_ref_nosat<-1.2),(1.2,i_sd_ref_nosat>1.2),(i_sd_ref_nosat,True))
    g_i_sq_ref = -i_sq_ref + sym.Piecewise((-1.2,i_sq_ref_nosat<-1.2),(1.2,i_sq_ref_nosat>1.2),(i_sq_ref_nosat,True))

    v_td_ref  =  R_s*i_sd_ref - X_s*i_sq_ref + v_sd  
    v_tq_ref  =  R_s*i_sq_ref + X_s*i_sd_ref + v_sq 

    v_tD_ref = v_td_ref * cos(delta) + v_tq_ref * sin(delta)   
    v_tQ_ref =-v_td_ref * sin(delta) + v_tq_ref * cos(delta)    
    v_ti_ref = v_tD_ref
    v_tr_ref = v_tQ_ref 
    m_ref = sym.sqrt(v_tr_ref**2 + v_ti_ref**2)/v_dc
    theta_t_ref = sym.atan2(v_ti_ref,v_tr_ref) 


    ### dae 
    grid.dae['g'] += [g_i_sd_ref, g_i_sq_ref]
    grid.dae['y_ini'] += [i_sq_ref, i_sd_ref]  
    grid.dae['y_run'] += [i_sq_ref, i_sd_ref]  

    grid.dae['u_ini_dict'].update({f'lvrt_ext_{name}':0.0})
    grid.dae['u_run_dict'].update({f'lvrt_ext_{name}':0.0})

    grid.dae['u_ini_dict'].update({f'p_s_ppc_{name}':1.5})
    grid.dae['u_run_dict'].update({f'p_s_ppc_{name}':1.5})

    grid.dae['u_ini_dict'].update({f'q_s_ppc_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_s_ppc_{name}':0.0})

    grid.dae['u_ini_dict'].update({f'{str(i_sa_ref)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(i_sa_ref)}':0.0})

    grid.dae['u_ini_dict'].update({f'{str(i_sr_ref)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(i_sr_ref)}':0.0})

    grid.dae['params_dict'].update({f'{str(v_lvrt)}':0.8})
    # grid.dae['params_dict'].update({f'{str(T_lp1p)}':0.1,f'{str(T_lp2p)}':0.1})
    # grid.dae['params_dict'].update({f'{str(T_lp1q)}':0.1,f'{str(T_lp2q)}':0.1})
    # grid.dae['params_dict'].update({f'{str(PRampUp)}':2.5,f'{str(PRampDown)}':-2.5})
    # grid.dae['params_dict'].update({f'{str(QRampUp)}':2.5,f'{str(QRampDown)}':-2.5})

    grid.dae['params_dict'].update(sys_p['params_dict'])
    grid.dae['params_dict'].update(sys_q['params_dict'])


    ### outputs
    grid.dae['h_dict'].update({f"m_ref_{name}":m_ref})
    grid.dae['h_dict'].update({f"v_sd_{name}":v_sd})
    grid.dae['h_dict'].update({f"v_sq_{name}":v_sq})
    grid.dae['h_dict'].update({f"lvrt_{name}":lvrt})
    grid.dae['h_dict'].update({f'p_s_ppc_{name}':p_s_ppc})
    grid.dae['h_dict'].update({f'q_s_ppc_{name}':q_s_ppc})

    ## VSC model 
    # m = sym.Symbol(f"m_{name}", real=True)
    # theta_t = sym.Symbol(f"theta_t_{name}", real=True)
      
    ### dynamic states
    #m_f = sym.Symbol(f"m_f_{name}", real=True)

    ### algebraic states
    i_si = sym.Symbol(f"i_si_{name}", real=True)
    i_sr = sym.Symbol(f"i_sr_{name}", real=True)       
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)

    ### parameters
    F_n = sym.Symbol(f"F_n_{name}", real=True)            

    
    params_list = ['S_n','F_n','U_n','X_s','R_s']
    
    ### auxiliar
    v_si = V_s*sin(theta_s)  # v_D, e^(-j)
    v_sr = V_s*cos(theta_s)  # v_Q
    Omega_b = 2*np.pi*F_n
    m = m_ref
    theta_t = theta_t_ref
    v_t_m = m*v_dc
    v_tr = v_t_m*cos(theta_t)
    v_ti = v_t_m*sin(theta_t)
    

    
    ### dynamic equations            

    ### algebraic equations   
    v_ti = v_ti_ref 
    v_tr = v_tr_ref 
    g_i_si = v_ti - R_s*i_si + X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr - X_s*i_si - v_sr 
    # i_sr = (-R_s*v_sr + R_s*v_tr + X_s*v_si - X_s*v_ti)/(R_s**2 + X_s**2)
    # i_si = (-R_s*v_si + R_s*v_ti - X_s*v_sr + X_s*v_tr)/(R_s**2 + X_s**2)
    g_p_s  = i_si*v_si + i_sr*v_sr - p_s  
    g_q_s  = i_si*v_sr - i_sr*v_si - q_s 

    ### dae 
    f_vsg = []
    x_vsg = []
    g_vsg = [g_i_si,g_i_sr,g_p_s,g_q_s]
    y_vsg = [  i_sr,  i_si,  p_s,  q_s]

    grid.dae['f'] += f_vsg
    grid.dae['x'] += x_vsg
    grid.dae['g'] += g_vsg
    grid.dae['y_ini'] += y_vsg  
    grid.dae['y_run'] += y_vsg  
    
    # grid.dae['u_ini_dict'].update({f'{m}':1.0})
    # grid.dae['u_run_dict'].update({f'{m}':1.0})

    # grid.dae['u_ini_dict'].update({f'{theta_t}':0.0})
    # grid.dae['u_run_dict'].update({f'{theta_t}':0.0})

    # grid.dae['u_ini_dict'].update({f'{v_dc}':1.2})
    # grid.dae['u_run_dict'].update({f'{v_dc}':1.2})

    grid.dae['xy_0_dict'].update({str(p_s):0.5})
       
    ### outputs

    

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]}) 

    if 'monitor' in data_dict:
        if data_dict['monitor'] == True:
            
            grid.dae['h_dict'].update({f"v_dc_v_{name}":v_dc*V_dc_b})
            grid.dae['h_dict'].update({f"v_ac_v_{name}":v_t_m*U_n})
            grid.dae['h_dict'].update({f"v_dc_v_{name}":v_dc_v})    
            grid.dae['h_dict'].update({f"p_mp_{name}":p_mp})    
            grid.dae['h_dict'].update({f"i_pv_{name}":i_pv})    
            grid.dae['h_dict'].update({f"v_dc_{name}":v_dc}) 
            grid.dae['h_dict'].update({f"p_s_{name}":p_s})
            grid.dae['h_dict'].update({f"q_s_{name}":q_s})
            grid.dae['h_dict'].update({f"v_ti_{name}":v_ti})
            grid.dae['h_dict'].update({f"v_tr_{name}":v_tr})    
            grid.dae['h_dict'].update({f"i_si_{name}":i_si})
            grid.dae['h_dict'].update({f"i_sr_{name}":i_sr})
            i_s = sym.sqrt(i_sr**2 + i_si**2) 
            grid.dae['h_dict'].update({f"i_s_{name}":i_s})            

    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var

def sym2model():
    v_sd = sym.Symbol(f"v_sd", real=True)
    v_sq = sym.Symbol(f"v_sq", real=True)
    p_s_ref = sym.Symbol(f"p_s_ref", real=True)
    q_s_ref = sym.Symbol(f"q_s_ref", real=True)
    i_sa_ref = sym.Symbol(f"i_sa_ref", real=True)
    i_sr_ref = sym.Symbol(f"i_sr_ref", real=True)
    i_sd_pq_ref,i_sq_pq_ref = sym.symbols(f'i_sd_pq_ref,i_sq_pq_ref', real=True)
    i_sd_ar_ref,i_sq_ar_ref = sym.symbols(f'i_sd_ar_ref,i_sq_ar_ref', real=True)


    g_i_sd_pq_ref  = i_sd_pq_ref*v_sd + i_sq_pq_ref*v_sq - p_s_ref  
    g_i_sq_pq_ref  =-i_sq_pq_ref*v_sd + i_sd_pq_ref*v_sq - q_s_ref 

    sol = sym.solve([g_i_sd_pq_ref,g_i_sq_pq_ref],[i_sd_pq_ref,i_sq_pq_ref])

    for item in sol:
        print(f"{item} = {sol[item]}")

    v_m = sym.sqrt(v_sd**2 + v_sq**2)
    g_i_sd_ar_ref  = i_sd_ar_ref*v_sd/v_m + i_sq_ar_ref*v_sq/v_m - i_sa_ref  
    g_i_sq_ar_ref  =-i_sq_ar_ref*v_sd/v_m + i_sd_ar_ref*v_sq/v_m - i_sr_ref 

    sol = sym.solve([g_i_sd_ar_ref,g_i_sq_ar_ref],[i_sd_ar_ref,i_sq_ar_ref])

    for item in sol:
        print(f"{item} = {sol[item]}")

    # coupling filter:
    v_tr,v_ti = sym.symbols(f"v_tr,v_ti", real=True)
    i_sr,i_si = sym.symbols(f"i_sr,i_si", real=True)
    v_sr,v_si = sym.symbols(f"v_sr,v_si", real=True)
    R_s,X_s =   sym.symbols(f"R_s,X_s", real=True)

    g_i_si = v_ti - R_s*i_si + X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr - X_s*i_si - v_sr 
    sol = sym.solve([g_i_sr,g_i_si],[i_sr,i_si])

    for item in sol:
        print(f"{item} = {sol[item]}")

    # simplified PV module


def change_ss(model,A,B,C,D):
    pass



def test_build():
    from pydae.bmapu import bmapu_builder
    
    data = {
    "system":{"name":"test_model","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},       
    "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
             {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}
            ],
    "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.0,"Bs_pu":0.0,"S_mva":10000.0}],
    "sources":[{"bus":"2","type":"genape", "S_n":10000e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}],
    "pvs":[{
      "bus": "1",
      "type": "pv_dq_ss",
      "S_n": 3000000.0,
      "U_n": 400.0,
      "F_n": 50.0,
      "X_s": 0.1,
      "R_s": 0.0001,
      "monitor": False,
      "I_sc": 8,
      "V_oc": 42.1,
      "I_mp": 3.56,
      "V_mp": 33.7,
      "K_vt": -0.16,
      "K_it": 0.065,
      "N_pv_s": 23,
      "N_pv_p": 1087,
    # Close loop
    #   "A_p": [[ 2.22745959,  2.89134367],[-5.98640302, -5.13853719]],
    #   "B_p":[[-2.62892537],[-3.35747765]],
    #   "C_p":[[-0.53183608 ,-0.28013641]],
    #   "D_p":[[0.]],
    #   "A_q": [[ 2.22745959,  2.89134367],[-5.98640302, -5.13853719]],
    #   "B_q":[[-2.62892537],[-3.35747765]],
    #   "C_q":[[-0.53183608 ,-0.28013641]],
    #   "D_q":[[0.]],
    # Open loop
    #   "A_p":[[ 0.33137699,  2.1000962 ], [-9.11825055, -6.20455623]],
    #   "B_p":[[-3.44429742], [-5.67395313]],
    #   "C_p":[[-0.53651084, -0.02498652]],
    #   "D_p":[[0.]],   
    #   "A_q":[[ 0.33137699,  2.1000962 ], [-9.11825055, -6.20455623]],
    #   "B_q":[[-3.44429742], [-5.67395313]],
    #   "C_q":[[-0.53651084, -0.02498652]],
    #   "D_q":[[0.]],   
    # Open loop high order
   "A_p":[[ 8.38271015, -1.7571524 ,  1.73425499, -1.30126187, -0.65661779, 0.19946719, -0.27419108],
          [ 9.21838154,  4.00099853, -7.81509434,  4.35694934,  1.93722067,-0.5469874 ,  0.71654726],
          [-2.40905211,  2.06691677,  1.45133758, -4.4215498 , -1.4626006 , 0.3642175 , -0.44442714],
          [ 1.49417659, -0.95112217,  3.64830378, -0.65345492,  2.62637562,-0.48697018,  0.52522081],
          [ 1.76580717, -0.98874696,  2.81999399, -6.13573245, -2.77497888,-1.4733826 ,  1.18610772],
          [-3.98772761,  2.0720406 , -5.20922579,  8.4373616 , 10.92677635,-5.37878648, -6.41515444],
          [ 2.15679302, -1.06505127,  2.48988235, -3.55911887, -3.43500894, 2.50096692, -9.87800368]],    
    "B_p":[[ 0.76019693],[-1.61641228],[ 0.79263982],[-0.70153677],[-1.0642704 ],[ 2.92345721],[-1.86444729]],
    "C_p":[[ 0.53471062, -0.22393912,  0.43508347, -0.49268284, -0.34211526,0.13852951, -0.2621668 ]],
    "D_p":[[0.]], 
    "A_q":[[ 8.38271015, -1.7571524 ,  1.73425499, -1.30126187, -0.65661779, 0.19946719, -0.27419108],
          [ 9.21838154,  4.00099853, -7.81509434,  4.35694934,  1.93722067,-0.5469874 ,  0.71654726],
          [-2.40905211,  2.06691677,  1.45133758, -4.4215498 , -1.4626006 , 0.3642175 , -0.44442714],
          [ 1.49417659, -0.95112217,  3.64830378, -0.65345492,  2.62637562,-0.48697018,  0.52522081],
          [ 1.76580717, -0.98874696,  2.81999399, -6.13573245, -2.77497888,-1.4733826 ,  1.18610772],
          [-3.98772761,  2.0720406 , -5.20922579,  8.4373616 , 10.92677635,-5.37878648, -6.41515444],
          [ 2.15679302, -1.06505127,  2.48988235, -3.55911887, -3.43500894, 2.50096692, -9.87800368]],    
    "B_q":[[ 0.76019693],[-1.61641228],[ 0.79263982],[-0.70153677],[-1.0642704 ],[ 2.92345721],[-1.86444729]],
    "C_q":[[ 0.53471062, -0.22393912,  0.43508347, -0.49268284, -0.34211526,0.13852951, -0.2621668 ]],
    "D_q":[[0.]],  
    }]
    }


             
    grid = bmapu_builder.bmapu(data)
    #grid.checker()
    grid.verbose = False 
    grid.build('temp')

def test_run():
    import temp

    model = temp.model()
    #model.ini({},'xy_0.json')
    model.ini({},1)

    model.report_x()
    model.report_y()
    model.report_z()

if __name__=='__main__':

    test_build()
    test_run()

# def test():
    
#     from pydae.bmapu import bmapu_builder
#     import pytest

#     grid = bmapu_builder.bmapu('bess_pq_Hithium.hjson')
#     #grid.checker()
#     grid.verbose = False 
#     grid.build('temp')

#     import temp

#     model = temp.model()
#     soc_ref = 0.5
#     model.ini({'soc_ref_1':soc_ref},'xy_0.json')

#     assert model.get_value('soc_1') == pytest.approx(soc_ref, rel=0.001)
#     assert model.get_value('p_dc_1') == pytest.approx(0.0)


#     ### run:
#     model.Dt = 1.0
#     model.run(1.0,{})
#     model.run(0.5*3600,{'p_s_ref_1': 1.0}) # half an hour discharging
  
#     assert model.get_value('soc_1') == pytest.approx(0.25, rel=0.05)

#     model.run(1.0*3600,{'p_s_ref_1':-1.0}) # half an hour discharging
#     model.post()

#     assert model.get_value('soc_1') == pytest.approx(0.5, rel=0.05)









# #sym2model()
# if __name__ == "__main__":



#     grid = bmapu_builder.bmapu(data)
#     #grid.checker()
#     grid.uz_jacs = True
#     grid.verbose = False
#     grid.construct('test_model')

#     bus_name = "1"
#     data_dict = {"bus":bus_name,"type":"pv_pq",
#                  "S_n":1e6,"U_n":400.0,"F_n":50.0,
#                  "X_s":0.1,"R_s":0.01,"monitor":True,
#                  "I_sc":8,"V_oc":42.1,"I_mp":3.56,"V_mp":33.7,
#                  "K_vt":-0.160,"K_it":0.065,
#                  "N_pv_s":25,"N_pv_p":250}

#     p_W,q_var = pv_dq(grid,'1','1',data_dict)

#     # grid power injection
#     idx_bus = [bus['name'] for bus in grid.data['buses']].index(bus_name) # get the number of the bus where the syn is connected

#     S_base = sym.Symbol('S_base', real = True)
#     grid.dae['g'][idx_bus*2]   += -p_W/S_base
#     grid.dae['g'][idx_bus*2+1] += -q_var/S_base
    
#     grid.compile()   

#     import test_model

#     xy_0 = {
#     "V_1": 1.0,
#     "theta_1": 0.0,
#     "V_2": 1.0,
#     "theta_2": 0.0,
#     "omega_coi": 1.0,
#     "omega_2": 1.0,
#     "v_dc_v_1":800
#     }
#     model = test_model.model()
#     model.ini({'p_s_ppc_1':0.9,'q_s_ppc_1':0.2,'irrad_1':100,'temp_deg_1':75,
#                'v_ref_2':1.0},xy_0)

#     #model.report_y()
#     model.report_z()
#     #model.report_u()
