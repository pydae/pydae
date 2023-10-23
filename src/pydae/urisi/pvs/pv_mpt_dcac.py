import sympy as sym
import numpy as np

def pv_mpt_dcac(grid,data,name,bus_name):
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
    
    bus_name = data['bus']
    name = bus_name
   
    A_value  = data['A']   
    B_value  = data['B']   
    C_value  = data['C']  

    #### voltages:
    v_hp, v_hn = sym.symbols(f'V_{bus_name}_0_r,V_{bus_name}_1_r', real=True)
    v_hp_i, v_hn_i = sym.symbols(f'V_{bus_name}_0_i,V_{bus_name}_1_i', real=True)
   
    ### LV-side
    A,B,C = sym.symbols(f'A_{name},B_{name},C_{name}',real=True)

    ### HV-side
    e_h,v_h,i_h,p_h = sym.symbols(f'e_h_{name},v_h_{name},i_h_{name},p_h_{name}',real=True)
    R_h = sym.Symbol(f'R_h_{name}',real=True)
    p_h_ref = sym.Symbol(f'p_ref_{name}',real=True)
    m_h = sym.Symbol(f'm_h_{name}',real=True)
    i_g = sym.Symbol(f'i_g_{name}',real=True) 
    R_g = sym.Symbol(f'R_g_{name}',real=True)


    ## PV
    K_vt,K_it = sym.symbols(f"K_vt_{name},K_it_{name}", real=True)
    V_oc,V_mp,I_sc,I_mp = sym.symbols(f"V_oc_{name},V_mp_{name},I_sc_{name},I_mp_{name}", real=True)
    temp_deg,irrad = sym.symbols(f"temp_deg_{name},irrad_{name}", real=True)
    T_stc_k,i,v = sym.symbols(f"T_stc_k_{name},i_{name},v_{name}", real=True)
    v_dc,K_it = sym.symbols(f"v_dc_{name},K_it_{name}", real=True)
    N_pv_s,N_pv_p = sym.symbols(f"N_pv_s_{name},N_pv_p_{name}", real=True)
    K_pp,K_pi = sym.symbols(f"K_pp_{name},K_pi_{name}", real=True)
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

    # (v_1 - v)/(v_1 - v_2) = (i_1 - i)/(i_1 - i_2)
    i_pv = p_l/v_dc
    p_mp = V_mp_t*I_mp_i
    #v_dc_v = v_1 - (i_1 - i_pv)*(v_1 - v_2)/(i_1 - i_2) 
    g_v_dc = -v_dc + v_1 - (i_1 - i_pv)*(v_1 - v_2)/(i_1 - i_2) 

    grid.dae['g'] +=     [g_v_dc]
    grid.dae['y_ini'] += [v_dc]  
    grid.dae['y_run'] += [v_dc]  

    grid.dae['u_ini_dict'].update({f'{str(irrad)}':1000.0})
    grid.dae['u_run_dict'].update({f'{str(irrad)}':1000.0})

    grid.dae['u_ini_dict'].update({f'{str(temp_deg)}':25.0})
    grid.dae['u_run_dict'].update({f'{str(temp_deg)}':25.0})

    grid.dae['params_dict'].update({
                   str(I_sc):data['I_sc'],
                   str(I_mp):data['I_mp'],
                   str(V_mp):data['V_mp'],
                   str(V_oc):data['V_oc'],
                   str(N_pv_s):data['N_pv_s'],
                   str(N_pv_p):data['N_pv_p'],
                   str(K_vt):data['K_vt'],
                   str(K_it):data['K_it']
                   })

    grid.dae['xy_0_dict'].update({f"v_dc_{name}":data['V_mp']*data['N_pv_s']})

    # VSC
    v_l = v_dc
    epsilon_p = p_mp - p_l 
    p_h_ref = K_pp*epsilon_p + K_pi*xi_p
    p_h = p_h_ref
    v_h = v_hp - v_hn
    i_h = p_h/v_h
    i_hp =  i_h
    i_hn = -i_h
    p_loss = A*i_hp**2 + B*sym.Abs(i_hp) + C

    eq_p_l = -p_l + p_h + p_loss

    dxi_p = epsilon_p
    dp_l_f = 1/T_mp*(p_l - p_l_f)

    grid.dae['f'] += [dxi_p, dp_l_f]
    grid.dae['x'] += [ xi_p,  p_l_f]
    grid.dae['g'] += [eq_p_l]
    grid.dae['y_ini'] += [ p_l]  
    grid.dae['y_run'] += [ p_l]  

    # current injections dc LV side
    idx_hp_r,idx_hp_i = grid.node2idx(f'{bus_name}','a')
    idx_hn_r,idx_hn_i = grid.node2idx(f'{bus_name}','b')
    grid.dae['g'] [idx_hp_r] += -i_hp 
    grid.dae['g'] [idx_hn_r] += -i_hn
    grid.dae['g'] [idx_hp_i] += -v_hp_i/1e3
    grid.dae['g'] [idx_hn_i] += -v_hn_i/1e3
   
    grid.dae['params_dict'].update({f'R_h_{name}':0.01})    
    grid.dae['params_dict'].update({f'R_g_{name}':3})   
    grid.dae['params_dict'].update({f'A_{name}':A_value})   
    grid.dae['params_dict'].update({f'B_{name}':B_value})   
    grid.dae['params_dict'].update({f'C_{name}':C_value})   
    grid.dae['params_dict'].update({f'K_pp_{name}':data['K_pp']})   
    grid.dae['params_dict'].update({f'K_pi_{name}':data['K_pi']})   
    grid.dae['params_dict'].update({f'T_mp_{name}':1.0})   

    grid.dae['h_dict'].update({f'p_loss_{name}':p_loss})
    grid.dae['h_dict'].update({f'i_pv_{name}':i_pv})
    grid.dae['h_dict'].update({f'p_mp_{name}':p_mp})

def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('pv_mpt_dcac.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 500e3
    V_n = 800
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    A = Conduction_losses/(I_n**2)*0
    B = 1*0
    C = 0.02*S_n*0
    model = temp.model()
    model.ini({'A_D1':A,'B_D1':B,'C_D1':C},'xy_0.json')
    model.report_y()
    model.report_z()



if __name__ == '__main__':

    #development()
    test()



