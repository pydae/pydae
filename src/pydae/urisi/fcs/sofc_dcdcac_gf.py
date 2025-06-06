import sympy as sym
import numpy as np
from sympy import interpolating_spline


import sympy as sym
import numpy as np

from pydae.urisi.vscs.ac_3ph_4w_gfpizv import ac_3ph_4w_gfpizv


def sofc_dcdcac_gf(grid,data,name,bus_name):

    #############################################################################
    ### DC/AC Converter
    #############################################################################

    ac_3ph_4w_gfpizv(grid,data)


    #############################################################################
    ### Common
    #############################################################################
    v_fc_dc =  sym.Symbol(f'v_fc_dc_{name}', real=True) # ?? (units?)
    p_dc = sym.Symbol(f'p_dc_{name}', real=True)  # DC/AC converter DC power
    v_dc = grid.aux[f'ac_3ph_4w_l_{name}']['v_dc']

    #############################################################################
    ### DC/DC Converter
    #############################################################################

    bus_hv_name = data['bus']
    name = bus_hv_name

  
    ## inputs
    v_dc_h_ref = sym.Symbol(f"v_dc_h_ref_{name}", real=True)


    ### LV-side
    i_l,p_l = sym.symbols(f'i_l_{name},p_l_{name}',real=True)
    A_dcdc,B_dcdc,C_dcdc = sym.symbols(f'A_dcdc_{bus_hv_name},B_dcdc_{bus_hv_name},C_dcdc_{bus_hv_name}',real=True)

    ### HV-side
    e_h,v_h,i_h,p_h = sym.symbols(f'e_h_{bus_hv_name},v_h_{bus_hv_name},i_h_{bus_hv_name},p_h_{bus_hv_name}',real=True)
    m_h = sym.Symbol(f'm_h_{bus_hv_name}',real=True)
    i_h_f = sym.Symbol(f'i_h_f_{bus_hv_name}',real=True)
    T_f,Droop_i,Droop_p = sym.symbols(f'T_f_{bus_hv_name},Droop_i_{bus_hv_name},Droop_p_{bus_hv_name}',real=True)

    A_dcdc = sym.Symbol(f"A_dcdc_{name}", real=True)
    B_dcdc = sym.Symbol(f"B_dcdc_{name}", real=True)
    C_dcdc = sym.Symbol(f"C_dcdc_{name}", real=True)

    v_l = v_fc_dc
    v_h = v_dc_h_ref
    m_h = v_l/v_h
    p_h = p_dc
    i_h = p_h/v_h
    p_l = p_h + A_dcdc*i_h**2 + B_dcdc*sym.Abs(i_h) + C_dcdc
    i_l =  p_l/v_l 

    # Algebraic equations
    g_v_dc = v_dc_h_ref - v_dc


    grid.dae['g'] += [g_v_dc]
    grid.dae['y_ini'] += [v_dc]
    grid.dae['y_run'] += [v_dc]


    grid.dae['u_ini_dict'].update({f'v_dc_h_ref_{name}':data['v_dc_h_ref']}) 
    grid.dae['u_run_dict'].update({f'v_dc_h_ref_{name}':data['v_dc_h_ref']}) 
   
    ## algebraic states
   
    if 'A_dcdc' in data:
        A_dcdc_N = data['A_dcdc']
        B_dcdc_N = data['B_dcdc']
        C_dcdc_N = data['C_dcdc']
    else:
        A_dcdc_N = 0.0001
        B_dcdc_N = 0.0
        C_dcdc_N = 0.0001

    grid.dae['params_dict'].update({f"{str(A_dcdc)}":A_dcdc_N}) 
    grid.dae['params_dict'].update({f"{str(B_dcdc)}":B_dcdc_N}) 
    grid.dae['params_dict'].update({f"{str(C_dcdc)}":C_dcdc_N}) 
    
    ## outputs
    #grid.dae['h_dict'].update({f'p_loss_{name}':p_loss})
    grid.dae['h_dict'].update({f'i_l_{name}':i_l})
    grid.dae['h_dict'].update({f'p_h_{name}':p_h})
    grid.dae['h_dict'].update({f'm_h_{bus_hv_name}':m_h})

    grid.dae['u_ini_dict'].pop(f'v_dc_{name}') 
    grid.dae['u_run_dict'].pop(f'v_dc_{name}') 



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
    i_dc = i_dc_ref + Di_dc_ref + sym.Piecewise((1.0,i_l<1.0),(250.0,i_l>250.0),(i_l,True))

    #formulacion del sistema de ecuaciones del modelo
    # dynamic equations
    dq_h2  = (2*K_r/U_opt*i_dc - q_h2)/Tau_f
    dp_h2  = (-p_h2 + (-2*K_r*i_dc + q_h2)/K_h2)/Tau_h2
    dp_h2o = (2*K_r*i_dc/K_h2o - p_h2o)/Tau_h2o
    dp_o2  = ((q_h2/R_h01 - K_r*i_dc)/K_o2 - p_o2)/Tau_o2

    dDi_dc_ref  = di_dc_ref - 1e-6*Di_dc_ref

    V_ernst = R*temp/(2*F)*sym.ln(p_h2*sym.sqrt(p_o2)/p_h2o)*N0
    P_dc = v_fc_dc*i_dc

    # Algebraic equations
    g_v_fc_dc = V_ernst + E0*N0 - R_ohm*N0*i_dc - v_fc_dc


    grid.dae['f'] += [dq_h2,dp_h2,dp_h2o,dp_o2,dDi_dc_ref]
    grid.dae['x'] += [ q_h2, p_h2, p_h2o, p_o2, Di_dc_ref]
    grid.dae['g'] += [g_v_fc_dc]
    grid.dae['y_ini'] += [v_fc_dc]
    grid.dae['y_run'] += [v_fc_dc]
    grid.dae['h_dict'].update({f'V_ernst_{name}':V_ernst,
                               f'P_dc_{name}': i_dc*v_fc_dc,
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
    grid.dae['xy_0_dict'].update({f"v_fc_dc_{name}":500.0})  
    grid.dae['xy_0_dict'].update({f"v_dc_{name}":800.0})  

# # For documentation purposes (very experimental, do not use)
# doc_dict = {}
# doc_dict.update({'K_r':{'default':1.166e-06}, 'description':'', 'tex':''})


def development():
    pass


def test_iso_build():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('sofc_dcdcac_gf_iso.hjson')
    grid.uz_jacs = True
    grid.build('temp')

def test_iso_ini():

    import temp

    model = temp.model()
    model.ini({'i_dc_ref_A1':5.0, 'U_opt_A1':0.7},'xy_0.json')
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()

def test_ib_build():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('sofc_dcdcac_gf_ib.hjson')
    grid.uz_jacs = True
    grid.build('temp')

def test_ib_ini():

    import temp

    model = temp.model()
    model.ini({'i_dc_ref_A1':5.0, 'U_opt_A1':0.7},'xy_0.json')
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()


if __name__ == '__main__':

    # test_iso_build()
    # test_iso_ini()
    test_ib_build()
    test_ib_ini()





