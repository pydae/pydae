import sympy as sym
import numpy as np
from sympy import interpolating_spline

import sympy as sym
import numpy as np

def sofc_dcdc_gf(grid,data,name,bus_name):

    #############################################################################
    ### Common
    #############################################################################
    v_dc = sym.Symbol(f'v_dc_{name}', real=True)  # DC voltage (V)


    #############################################################################
    ### DC/DC Converter
    #############################################################################

    bus_hv_name = data['bus']
    name = bus_hv_name
    
    A_value  = data['A']   
    B_value  = data['B']   
    C_value  = data['C']  

    #### voltages:
    v_hp, v_hn = sym.symbols(f'V_{bus_hv_name}_0_r,V_{bus_hv_name}_1_r', real=True)
    v_bat = sym.symbols(f'v_bat_{name}', real=True)
    v_hp_i, v_hn_i = sym.symbols(f'V_{bus_hv_name}_0_i,V_{bus_hv_name}_1_i', real=True)
   
    ### LV-side
    i_l,p_l = sym.symbols(f'i_l_{name},p_l_{name}',real=True)
    A,B,C = sym.symbols(f'A_{bus_hv_name},B_{bus_hv_name},C_{bus_hv_name}',real=True)

    ### HV-side
    e_h,v_h,i_h,p_h = sym.symbols(f'e_h_{bus_hv_name},v_h_{bus_hv_name},i_h_{bus_hv_name},p_h_{bus_hv_name}',real=True)
    R_h = sym.Symbol(f'R_h_{bus_hv_name}',real=True)
    e_h_ref = sym.Symbol(f'e_h_ref_{bus_hv_name}',real=True)
    m_h = sym.Symbol(f'm_h_{bus_hv_name}',real=True)
    i_g = sym.Symbol(f'i_g_{bus_hv_name}',real=True) 
    R_g = sym.Symbol(f'R_g_{bus_hv_name}',real=True)
    i_h_f = sym.Symbol(f'i_h_f_{bus_hv_name}',real=True)
    T_f,Droop_i,Droop_p = sym.symbols(f'T_f_{bus_hv_name},Droop_i_{bus_hv_name},Droop_p_{bus_hv_name}',real=True)

    A_loss = sym.Symbol(f"A_loss_{name}", real=True)
    B_loss = sym.Symbol(f"B_loss_{name}", real=True)
    C_loss = sym.Symbol(f"C_loss_{name}", real=True)

    v_l = v_dc
    v_h = v_hp-v_hn
    m_h = v_l/e_h_ref
    e_h = e_h_ref - Droop_i * i_h_f - Droop_p * i_h_f * v_h 
    # i_hp = (v_hn + e_h - v_hp)/R_h 
    # i_hn = -i_hp-v_hn/R_g
    i_g = (v_hn + v_hp)/(2*R_g + R_h)
    v_og = i_g*R_g
    i_hp = (v_og + e_h/2 - v_hp)/R_h
    i_hn = (v_og - e_h/2 - v_hn)/R_h
    p_h = i_hp*e_h
    p_l = p_h + A*i_hp**2 + B*sym.Abs(i_hp) + C
    i_l =  p_l/v_l 

    di_h_f = 1/T_f*(i_hp - i_h_f)

    grid.dae['f'] +=  [di_h_f]
    grid.dae['x'] +=  [ i_h_f]   

    # current injections dc HV side
    idx_hp_r,idx_hp_i = grid.node2idx(f'{bus_hv_name}','a')
    idx_hn_r,idx_hn_i = grid.node2idx(f'{bus_hv_name}','b')
    grid.dae['g'] [idx_hp_r] += -i_hp 
    grid.dae['g'] [idx_hn_r] += -i_hn
    grid.dae['g'] [idx_hp_i] += -v_hp_i/1e3
    grid.dae['g'] [idx_hn_i] += -v_hn_i/1e3

    grid.dae['u_ini_dict'].update({f'e_h_ref_{bus_hv_name}':data['v_ref']}) 
    grid.dae['u_run_dict'].update({f'e_h_ref_{bus_hv_name}':data['v_ref']}) 
   
    grid.dae['params_dict'].update({f'R_h_{bus_hv_name}':0.01})    
    grid.dae['params_dict'].update({f'R_g_{bus_hv_name}':3})   
    grid.dae['params_dict'].update({f'A_{bus_hv_name}':A_value})   
    grid.dae['params_dict'].update({f'B_{bus_hv_name}':B_value}) 
    grid.dae['params_dict'].update({f'C_{bus_hv_name}':C_value})     
    grid.dae['params_dict'].update({f'Droop_p_{bus_hv_name}':0.0})   
    grid.dae['params_dict'].update({f'Droop_i_{bus_hv_name}':0.0})   
    grid.dae['params_dict'].update({f'T_f_{bus_hv_name}':0.1})   

    grid.dae['h_dict'].update({f'm_h_{bus_hv_name}':m_h})

    ## inputs
    v_dc_h_ref = sym.Symbol(f"v_dc_h_ref_{name}", real=True)
    
    ## algebraic states
   
    if 'A_loss' in data:
        A_loss_N = data['A_loss']
        B_loss_N = data['B_loss']
        C_loss_N = data['C_loss']
    else:
        A_loss_N = 0.0001
        B_loss_N = 0.0
        C_loss_N = 0.0001

    grid.dae['params_dict'].update({f"{str(A_loss)}":A_loss_N}) 
    grid.dae['params_dict'].update({f"{str(B_loss)}":B_loss_N}) 
    grid.dae['params_dict'].update({f"{str(C_loss)}":C_loss_N}) 
    
    ## outputs
    #grid.dae['h_dict'].update({f'p_loss_{name}':p_loss})
    grid.dae['h_dict'].update({f'i_l_{name}':i_l})
    grid.dae['h_dict'].update({f'p_h_{name}':p_h})
    grid.dae['h_dict'].update({f'e_h_{name}':e_h})

    HS_coi  = 1.0
    omega_coi_i = 1.0

    grid.omega_coi_numerator += omega_coi_i
    grid.omega_coi_denominator += HS_coi


    #############################################################################
    ### SOFC
    #############################################################################

    # variables de entradas y dinamicas
    # inputs
    temp = sym.Symbol(f'temperature_{name}', real=True)   # Temperatura
    i_dc_ref = sym.Symbol(f'i_dc_ref_{name}', real=True)  # Reference current
    di_dc_ref  = sym.Symbol(f'di_dc_ref_{name}', real=True) # Reference current

    # dynamic states
    q_h2 = sym.Symbol(f'q_h2_{name}', real=True)   # H₂ flow (units?)
    p_h2 = sym.Symbol(f'p_h2_{name}', real=True)   # H₂ pressure (units?)
    p_h2o = sym.Symbol(f'p_h2o_{name}', real=True) # H₂O pressure (units?)
    p_o2 = sym.Symbol(f'p_o2_{name}', real=True)   # O₂ pressure (units?)
    Di_dc_ref  = sym.Symbol(f'Di_dc_ref_{name}', real=True)  # Current increment (A)

    # algebraic states

    # parameters
    K_r = sym.Symbol(f'K_r_{name}', real=True)    # ?? (units?)
    K_h2 = sym.Symbol(f'K_h2_{name}', real=True)  # Valve molar constant for hydrogen (kmol/(s atm))
    K_h2o = sym.Symbol(f'K_h2o_{name}', real=True) # Valve molar constant for water (kmol/(s atm))
    K_o2 = sym.Symbol(f'K_o2_{name}', real=True) # Valve molar constant for oxygen (kmol/(s atm))
    R_h01 = sym.Symbol(f'R_h01_{name}', real=True) # ?? (units?)
    U_opt = sym.Symbol(f'U_opt_{name}', real=True) # ?? (units?)
    Tau_f = sym.Symbol(f'Tau_f_{name}', real=True) # ?? (units?)
    Tau_h2 = sym.Symbol(f'Tau_h2_{name}', real=True) # ?? (units?)
    Tau_h2o = sym.Symbol(f'Tau_h2o_{name}', real=True) # ?? (units?)
    Tau_o2 = sym.Symbol(f'Tau_o2_{name}', real=True) # ?? (units?)
    N0 = sym.Symbol(f'N0_{name}', real=True) # ?? (units?)
    E0 = sym.Symbol(f'E0_{name}', real=True) # ?? (units?)
    R_ohm = sym.Symbol(f'R_ohm_{name}', real=True) # ?? (units?)

    F = 96487.3 # Faraday's constant (units?)
    R = 8.314 # Universal gas constant  (units?)
    i_dc = i_dc_ref + Di_dc_ref + i_l

    #formulacion del sistema de ecuaciones del modelo
    # dynamic equations
    dq_h2  = (2*K_r/U_opt*i_dc - q_h2)/Tau_f
    dp_h2  = (-p_h2 + (-2*K_r*i_dc + q_h2)/K_h2)/Tau_h2
    dp_h2o = (2*K_r*i_dc/K_h2o - p_h2o)/Tau_h2o
    dp_o2  = ((q_h2/R_h01 - K_r*i_dc)/K_o2 - p_o2)/Tau_o2

    dDi_dc_ref  = di_dc_ref - 1e-6*Di_dc_ref

    V_ernst = R*temp/(2*F)*sym.ln(p_h2*sym.sqrt(p_o2)/p_h2o)*N0
    P_dc = v_dc*i_dc

    # Algebraic equations
    g_v_dc = V_ernst + E0*N0 - R_ohm*N0*i_dc - v_dc


    grid.dae['f'] += [dq_h2,dp_h2,dp_h2o,dp_o2,dDi_dc_ref]
    grid.dae['x'] += [ q_h2, p_h2, p_h2o, p_o2, Di_dc_ref]
    grid.dae['g'] += [g_v_dc]
    grid.dae['y_ini'] += [v_dc]
    grid.dae['y_run'] += [v_dc]
    grid.dae['h_dict'].update({f'V_ernst_{name}':V_ernst,
                               f'P_dc_{name}': i_dc*v_dc,
                               f'i_dc_{name}':i_dc})

    # Inputs (esto es el estado inicial??)
    grid.dae['u_ini_dict'].update({f'temperature_{name}':1273})
    grid.dae['u_ini_dict'].update({f'i_dc_ref_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'di_dc_ref_{name}':0.0})

    grid.dae['u_run_dict'].update({f'temperature_{name}':1273})
    grid.dae['u_run_dict'].update({f'i_dc_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'di_dc_ref_{name}':0.0})

    # damos valores a los parámetros
    # Parameters
    grid.dae['params_dict'].update({f'K_r_{name}':1.166e-06})
    grid.dae['params_dict'].update({f'K_h2_{name}':0.000843})  # Valve molar constant for hydrogen
    grid.dae['params_dict'].update({f'K_h2o_{name}':0.000281}) # Valve molar constant for water
    grid.dae['params_dict'].update({f'K_o2_{name}':0.00252})   # Valve molar constant for oxygen
    grid.dae['params_dict'].update({f'R_h01_{name}':1.145})
    grid.dae['params_dict'].update({f'U_opt_{name}':0.85})
    grid.dae['params_dict'].update({f'Tau_f_{name}':5})
    grid.dae['params_dict'].update({f'Tau_h2_{name}':26.1})
    grid.dae['params_dict'].update({f'Tau_h2o_{name}':78.3})
    grid.dae['params_dict'].update({f'Tau_o2_{name}':2.91})
    grid.dae['params_dict'].update({f'N0_{name}':450})
    grid.dae['params_dict'].update({f'E0_{name}':1.18})
    grid.dae['params_dict'].update({f'R_ohm_{name}':3.2813e-004})

    grid.dae['xy_0_dict'].update({f"q_h2_{name}":1.0})
    grid.dae['xy_0_dict'].update({f"p_h2_{name}":1.0})
    grid.dae['xy_0_dict'].update({f"p_h2o_{name}":1.0})
    grid.dae['xy_0_dict'].update({f"p_o2_{name}":1.0})
    grid.dae['xy_0_dict'].update({f"v_dc_{name}":500.0})  


# # For documentation purposes (very experimental, do not use)
# doc_dict = {}
# doc_dict.update({'K_r':{'default':1.166e-06}, 'description':'', 'tex':''})


def development():
    pass

def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('sofc_dcdc_gf_iso.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()
    model.ini({},'xy_0.json')
    model.report_x()
    model.report_y()
    model.report_z()

def test_iso():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('sofc_dcdc_gf_iso.hjson')
    grid.uz_jacs = True
    grid.build('temp')

    import temp

    model = temp.model()
    model.ini({'p_load_D2':100e3, 'i_dc_ref_D1':0.0, 'U_opt_D1':0.7},'xy_0.json')
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()


if __name__ == '__main__':

    #development()
    test_iso()





