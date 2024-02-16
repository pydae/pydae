import sympy as sym
import numpy as np
from sympy import interpolating_spline

import sympy as sym
import numpy as np

def bess_dcdc_gf(grid,data,name,bus_name):
    '''
    BESS with DC/DC grid former. 
    BESS in the low voltage side of the DC/DC
    Grid former is in the high voltage side.
    A P_h/V_h linear droop is implemented for the high voltage side.
    
    '''
    
    bus_hv_name = data['bus']
    name = bus_hv_name
    
    A_value  = data['A']   
    B_value  = data['B']   
    C_value  = data['C']  

    ### Common 
    soc = sym.Symbol(f"soc_{name}", real=True)
    xi_soc = sym.Symbol(f"xi_soc_{name}", real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)
    K_i = sym.Symbol(f"K_i_{name}", real=True)
    K_soc = sym.Symbol(f"K_soc_{name}", real=True) 
    K_charger = sym.Symbol(f"K_charger_{name}", real=True) 

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


    soc_ref = sym.Symbol(f"soc_ref_{name}", real=True) 

    epsilon_soc = soc_ref - soc
    p_soc = -(K_p*epsilon_soc + K_i*xi_soc)

    v_l = v_bat
    v_h = v_hp-v_hn
    m_h = v_l/e_h_ref
    e_h = e_h_ref - Droop_i * i_h_f - Droop_p * i_h_f * v_h + K_soc*p_soc
    # i_hp = (v_hn + e_h - v_hp)/R_h 
    # i_hn = -i_hp-v_hn/R_g
    i_g = (v_hn + v_hp)/(2*R_g + R_h)
    v_og = i_g*R_g
    i_hp = (v_og + e_h/2 - v_hp)/R_h
    i_hn = (v_og - e_h/2 - v_hn)/R_h
    p_h = i_hp*e_h
    p_l = p_h + A*i_hp**2 + B*sym.Abs(i_hp) + C
    i_charger = K_charger*p_soc/v_l
    i_l =  p_l/v_l - i_charger

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

    ############################################################################
    ## Battery
    ### dynamic states


    ### parameters
    E_kWh = sym.Symbol(f"E_kWh_{name}", real=True)
    soc_min = sym.Symbol(f"soc_min_{name}", real=True)
    soc_max = sym.Symbol(f"soc_max_{name}", real=True)
    A_loss = sym.Symbol(f"A_loss_{name}", real=True)
    B_loss = sym.Symbol(f"B_loss_{name}", real=True)
    C_loss = sym.Symbol(f"C_loss_{name}", real=True)
    B_0 = sym.Symbol(f"B_0_{name}", real=True)
    B_1 = sym.Symbol(f"B_1_{name}", real=True)
    B_2 = sym.Symbol(f"B_2_{name}", real=True)
    B_3 = sym.Symbol(f"B_3_{name}", real=True)
    R_bat = sym.Symbol(f"R_bat_{name}", real=True)

    E_n = 1000*3600*E_kWh

    if 'socs' in data:
        socs = np.array(data['socs'])
        es = np.array(data['es'])
        interpolation = interpolating_spline(1, soc, socs, es)
        interpolation._args = tuple(list(interpolation._args) + [sym.functions.elementary.piecewise.ExprCondPair(0,True)])
        e = interpolation
    else:
        e = B_0 + B_1*soc + B_2*soc**2 + B_3*soc**3


    ## dynamic equations    
    dsoc = 1/E_n*(-i_l*e)   
    dxi_soc = epsilon_soc     


    ## algebraic equations   
    g_v_bat = e - i_l*R_bat - v_bat

    grid.dae['f'] += [dsoc, dxi_soc]
    grid.dae['x'] += [ soc,  xi_soc] 

    grid.dae['g'] +=     [g_v_bat]
    grid.dae['y_ini'] += [  v_bat]  
    grid.dae['y_run'] += [  v_bat]  

    soc_ref_N = 0.5
    if 'soc_0' in data:
        soc_ref_N = data['soc_0']


    ## parameters  
    grid.dae['params_dict'].update({f"{K_p}":1e-6}) 
    grid.dae['params_dict'].update({f"{K_i}":1e-6})  
    grid.dae['params_dict'].update({f"{soc_min}":0.0}) 
    grid.dae['params_dict'].update({f"{soc_max}":1.0})           
    grid.dae['params_dict'].update({f"{E_kWh}":data['E_kWh']}) 

    if 'B_0' in data:
        B_0_N = data['B_0_N']
        B_1_N = data['B_1_N']
        B_2_N = data['B_2_N']
        B_3_N = data['B_3_N']
    else:
        B_0_N = 600
        B_1_N = 0.0
        B_2_N = 0.0
        B_3_N = 0.0

    grid.dae['params_dict'].update({f"{B_0}":B_0_N}) 
    grid.dae['params_dict'].update({f"{B_1}":B_1_N}) 
    grid.dae['params_dict'].update({f"{B_2}":B_2_N}) 
    grid.dae['params_dict'].update({f"{B_3}":B_3_N}) 

    if 'R_bat' in data:
        R_bat_N = data['R_bat']
    else:
        R_bat_N = 0.0

    grid.dae['params_dict'].update({f"{R_bat}":R_bat_N}) 

    grid.dae['xy_0_dict'].update({f"v_bat_{name}":B_0_N})


    # VSC

    ## inputs
    v_dc_h_ref = sym.Symbol(f"v_dc_h_ref_{name}", real=True)
    
    ## algebraic states
  

    grid.dae['u_ini_dict'].update({f'soc_ref_{name}':0.5}) 
    grid.dae['u_run_dict'].update({f'soc_ref_{name}':0.5}) 
   
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
    grid.dae['params_dict'].update({f"{K_charger}":0.0}) 
    grid.dae['params_dict'].update({f"{K_soc}":0.001}) 
    


    grid.dae['params_dict'].update({f"{E_kWh}":data['E_kWh']}) 

    grid.dae['xy_0_dict'].update({str(soc):0.5})
    grid.dae['xy_0_dict'].update({str(xi_soc):0.5})

    ## outputs
    #grid.dae['h_dict'].update({f'p_loss_{name}':p_loss})
    grid.dae['h_dict'].update({f'i_l_{name}':i_l})
    grid.dae['h_dict'].update({f'p_h_{name}':p_h})
    grid.dae['h_dict'].update({f'v_bat_{name}':v_bat})
    grid.dae['h_dict'].update({f'e_h_{name}':e_h})
    grid.dae['h_dict'].update({f'i_charger_{name}':i_charger})


    HS_coi  = 1.0
    omega_coi_i = 1.0

    grid.omega_coi_numerator += omega_coi_i
    grid.omega_coi_denominator += HS_coi


def development():
    pass

def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('bess_dcdc_gf.hjson')
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

    grid = urisi('bess_dcdc_gf_iso.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    model = temp.model()
    model.ini({'K_charger_D1':-1.0,'p_load_D2':100e3,'K_soc_D1':1e-6},'xy_0.json')
    model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()


if __name__ == '__main__':

    #development()
    test_iso()





