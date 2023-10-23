import sympy as sym
import numpy as np

def dcdc_gfl(grid,vsc_data):
    '''
    dc/dc converter grid former in the low voltage side
    
    '''
    
    bus_hv_name = vsc_data['bus_hv']
    bus_lv_name = vsc_data['bus_lv']  
    
    A_value  = vsc_data['A']   
    B_value  = vsc_data['B']   
    C_value  = vsc_data['C']   

    #### voltages:
    v_hp, v_hn = sym.symbols(f'V_{bus_hv_name}_0_r,V_{bus_hv_name}_1_r', real=True)
    v_lp, v_ln = sym.symbols(f'V_{bus_lv_name}_0_r,V_{bus_lv_name}_1_r', real=True)
    v_hp_i, v_hn_i = sym.symbols(f'V_{bus_hv_name}_0_i,V_{bus_hv_name}_1_i', real=True)
    v_lp_i, v_ln_i = sym.symbols(f'V_{bus_lv_name}_0_i,V_{bus_lv_name}_1_i', real=True)

    ### HV-side
    e_h,v_h,i_h,p_h = sym.symbols(f'e_h_{bus_hv_name},v_h_{bus_hv_name},i_h_{bus_hv_name},p_h_{bus_hv_name}',real=True)
    R_h = sym.Symbol(f'R_h_{bus_hv_name}',real=True)

    ### LV-side
    e_l,v_l,i_l,p_l = sym.symbols(f'e_l_{bus_lv_name},v_l_{bus_lv_name},i_l_{bus_lv_name},p_l_{bus_lv_name}',real=True)
    A,B,C = sym.symbols(f'A_{bus_lv_name},B_{bus_lv_name},C_{bus_lv_name}',real=True)
    R_l = sym.Symbol(f'R_l_{bus_lv_name}',real=True)
    e_l_ref = sym.Symbol(f'e_l_ref_{bus_lv_name}',real=True)
    m_l = sym.Symbol(f'm_l_{bus_lv_name}',real=True)
    R_g = sym.Symbol(f'R_g_{bus_lv_name}',real=True)

    v_h = v_hp-v_hn
    m_l = e_l_ref/v_h
    e_l = e_l_ref
    i_lp = (v_ln + e_l - v_lp)/R_l 
    i_ln = -i_lp-v_ln/R_g
    p_l = i_lp*e_l
    p_h = p_l + A*i_lp**2 + B*sym.Abs(i_lp) + C
    i_h =  p_h/v_h 

    # current injections dc HV side
    idx_hp_r,imag = grid.node2idx(f'{bus_hv_name}','a')
    idx_hn_r,imag = grid.node2idx(f'{bus_hv_name}','b')
    grid.dae['g'] [idx_hp_r] +=  i_h
    grid.dae['g'] [idx_hn_r] += -i_h

    # current injections dc LV side
    idx_lp_r,idx_lp_i = grid.node2idx(f'{bus_lv_name}','a')
    idx_ln_r,idx_ln_i = grid.node2idx(f'{bus_lv_name}','b')
    grid.dae['g'] [idx_lp_r] += -i_lp 
    grid.dae['g'] [idx_ln_r] += -i_ln
    grid.dae['g'] [idx_lp_i] += -v_lp_i/1e3
    grid.dae['g'] [idx_ln_i] += -v_ln_i/1e3

    grid.dae['u_ini_dict'].update({f'e_l_ref_{bus_lv_name}':400.0}) 
    grid.dae['u_run_dict'].update({f'e_l_ref_{bus_lv_name}':400.0}) 
   
    grid.dae['params_dict'].update({f'R_l_{bus_lv_name}':0.1})    
    grid.dae['params_dict'].update({f'R_g_{bus_lv_name}':3})   
    grid.dae['params_dict'].update({f'A_{bus_lv_name}':A_value})   
    grid.dae['params_dict'].update({f'B_{bus_lv_name}':B_value})   
    grid.dae['params_dict'].update({f'C_{bus_lv_name}':C_value})   

  #  grid.dae['xy_0_dict'].update({f'i_g_{bus_lv_name}':0.0})
    
    grid.dae['h_dict'].update({f'm_l_{bus_lv_name}':m_l})
   # grid.dae['h_dict'].update({f'i_g_{bus_lv_name}':i_g})


if __name__ == '__main__':

    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('dcdc_gfl.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 100e3
    V_n = 400
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    A = Conduction_losses/(I_n**2)
    B = 1
    C = 0.02*S_n
    model = temp.model()
    model.ini({'A_D2':A,'B_D2':B,'C_D2':C},'xy_0.json')
    model.report_y()
    model.report_z()
    #model.save_xy_0('xy_1.json')
