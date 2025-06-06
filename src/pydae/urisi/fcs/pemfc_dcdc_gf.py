import sympy as sym
import numpy as np
from sympy import interpolating_spline

import sympy as sym
import numpy as np

def pemfc_dcdc_gf(grid,data,name,bus_name):
    '''
    BESS with DC/DC grid former. 
    BESS in the low voltage side of the DC/DC
    Grid former is in the high voltage side.
    A P_h/V_h linear droop is implemented for the high voltage side.
    
    '''
    

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
    ### PEM
    #############################################################################

    # Constantes
    F = 96485.3
    R = 8.314510


    # Variables de entrada (símbolos)
    temp = sym.Symbol(f'temp_{name}', real=True)
    i_dc_ref = sym.Symbol(f'i_dc_ref_{name}', real=True)
    p_o2 = sym.Symbol(f'p_o2_{name}', real=True)
    p_h2 = sym.Symbol(f'p_h2_{name}', real=True)

    # Variables algebraicas y dinámicas
    v_dc = sym.Symbol(f'v_dc_{name}', real=True)
    v_d = sym.Symbol(f'v_d_{name}', real=True)

    # Parámetros simbólicos base
    #N0 = sym.Symbol('N0', real=True)
    n = sym.Symbol(f'n_{name}', real=True)
    l = sym.Symbol(f'l_{name}', real=True)
    A = sym.Symbol (f'A_{name}',real=True)
    Jn = sym.Symbol(f'Jn_{name}',real=True)
    N0 = sym.Symbol(f'N0_{name}',real=True)
    N_p = sym.Symbol(f'N_p_{name}',real=True)

    # Corriente
    i_dc = (i_dc_ref + i_l)/N_p

    # Expresiones simbólicas para parámetros derivados
    J = i_dc / A
    j0 = 2.752e-5 *(sym.exp(2.863e-3 * temp))
    jmax = 0.6284 - temp*0.0005
    alpha = 4.141e-5*(temp**1.642)
    Cd = 5.2532 * temp**(-1.3858)
    sigma =3.3e-3 *(sym.exp(5.5e-3 * temp))
    rom = 1 / sigma
    Rm = (rom * l) / A
    #RH = -1.2119 * sym.ln(temp) + 8.2553 la despreciamos al ser muy pequeña.
    RT = Rm #+RH
    B = -0.6387 * sym.ln(temp) + 3.8527
    V_ohm = (J + Jn) * RT
    V_cond = -B * sym.ln(1 - ((J + Jn) / jmax))
    V_act = ((2.3 * R * temp) / (alpha * n * F)) * sym.ln(((J + Jn) / j0))
    V_nerst = 1.229 - 0.0008456 * (temp - 298.25) + 0.00004308 * temp * (sym.ln(p_h2) + sym.ln(p_o2) / 2)
    Rd = (V_cond + V_act) / J
    tau_d = Rd * Cd

    # Ecuaciones diferenciales y algebraicas
    Dv_d = ((1 / Cd) *J) - ((1 / tau_d) * v_d)
    g_v_dc = N0*(V_nerst - V_ohm - v_d) - v_dc

    # Diccionario de entradas simbólicas
    u_dict = {
        f'i_dc_ref_{name}': 1e-6,
        f'temp_{name}': 0.0,
        f'p_h2_{name}': 0.0,
        f'p_o2_{name}': 0.0
    }


    # Diccionario de parámetros simbólicos
    params_dict = {
        f'Jn_{name}':3e-3,
        f'A_{name}': 75,
        f'N0_{name}': 300,
        f'n_{name}': 2,
        f'l_{name}': 183e-4,
        f'N_p_{name}': 20.0,   
    }

    # Sistema simbólico para PyDAE
  
    grid.dae['u_ini_dict'].update(u_dict)
    grid.dae['u_run_dict'].update(u_dict)
    grid.dae['params_dict'].update(params_dict)
    grid.dae['f'] += [Dv_d]
    grid.dae['x'] += [v_d]
    grid.dae['g'] += [g_v_dc]
    grid.dae['y_ini'] += [v_dc]
    grid.dae['y_run'] +=  [v_dc]
    grid.dae['h_dict'].update({
                f'V_nernst_{name}': V_nerst,
                f'P_dc_{name}': i_dc* N_p * v_dc,
                f'i_dc_{name}': i_dc})


    grid.dae['xy_0_dict'].update({f"v_d_{name}":0.5})
    grid.dae['xy_0_dict'].update({f"v_dc_{name}":250.0})
    

# # For documentation purposes (very experimental, do not use)
# doc_dict = {}
# doc_dict.update({'K_r':{'default':1.166e-06}, 'description':'', 'tex':''})


def development():
    pass

def test_build():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('pemfc_dcdc_gf_iso.hjson')
    grid.uz_jacs = True
    grid.build('temp')

def test_ini():
    import numpy as np
    import temp

    model = temp.model()
    model.ini({'i_dc_ref_D1':1,'temp_D1':323.15,'p_h2_D1':1,'p_o2_D1':0.21, 'p_load_D2':5000},'xy_0.json')  # model steady state computation
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()

if __name__ == '__main__':

    test_build()
    test_ini()





