import sympy as sym
import numpy as np

def dcdc_ph(grid,vsc_data):
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
   
    ### LV-side
    i_l,p_l = sym.symbols(f'i_l_{bus_lv_name},p_l_{bus_lv_name}',real=True)
    A,B,C = sym.symbols(f'A_{bus_hv_name},B_{bus_hv_name},C_{bus_hv_name}',real=True)

    ### HV-side
    e_h,v_h,i_h,p_h = sym.symbols(f'e_h_{bus_hv_name},v_h_{bus_hv_name},i_h_{bus_hv_name},p_h_{bus_hv_name}',real=True)
    R_h = sym.Symbol(f'R_h_{bus_hv_name}',real=True)
    p_h_ref = sym.Symbol(f'p_ref_{bus_hv_name}',real=True)
    m_h = sym.Symbol(f'm_h_{bus_hv_name}',real=True)
    i_g = sym.Symbol(f'i_g_{bus_hv_name}',real=True) 
    R_g = sym.Symbol(f'R_g_{bus_hv_name}',real=True)



    v_l = v_lp-v_ln
    p_h = p_h_ref
    v_h = v_hp - v_hn
    i_h = p_h/v_h
    i_hp =  i_h
    i_hn = -i_h
    p_loss = A*i_hp**2 + B*sym.Abs(i_hp) + C
    p_l = p_h + p_loss
    i_l =  p_l/v_l  

    # current injections dc HV side
    idx_lp_r,imag = grid.node2idx(f'{bus_lv_name}','a')
    idx_ln_r,imag = grid.node2idx(f'{bus_lv_name}','b')
    grid.dae['g'] [idx_lp_r] +=  i_l
    grid.dae['g'] [idx_ln_r] += -i_l

    # current injections dc LV side
    idx_hp_r,idx_hp_i = grid.node2idx(f'{bus_hv_name}','a')
    idx_hn_r,idx_hn_i = grid.node2idx(f'{bus_hv_name}','b')
    grid.dae['g'] [idx_hp_r] += -i_hp 
    grid.dae['g'] [idx_hn_r] += -i_hn
    grid.dae['g'] [idx_hp_i] += -v_hp_i/1e3
    grid.dae['g'] [idx_hn_i] += -v_hn_i/1e3


    grid.dae['u_ini_dict'].update({f'p_ref_{bus_hv_name}':vsc_data['p_ref']}) 
    grid.dae['u_run_dict'].update({f'p_ref_{bus_hv_name}':vsc_data['p_ref']}) 
   
    grid.dae['params_dict'].update({f'R_h_{bus_hv_name}':0.01})    
    grid.dae['params_dict'].update({f'R_g_{bus_hv_name}':3})   
    grid.dae['params_dict'].update({f'A_{bus_hv_name}':A_value})   
    grid.dae['params_dict'].update({f'B_{bus_hv_name}':B_value})   
    grid.dae['params_dict'].update({f'C_{bus_hv_name}':C_value})   

    grid.dae['h_dict'].update({f'p_loss_{bus_hv_name}':p_loss})


def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('dcdc_ph.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 100e3
    V_n = 1000
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    A = Conduction_losses/(I_n**2)
    B = 1
    C = 0.02*S_n
    model = temp.model()
    model.ini({'A_D2':A,'B_D2':B,'C_D2':C},'xy_0.json')
    model.report_y()
    model.report_z()



if __name__ == '__main__':

    #development()
    test()



