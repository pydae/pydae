import sympy as sym
import numpy as np

def vdc_src(grid,data):
    '''
    dc/dc converter grid former in the low voltage side
    
    '''
    
    bus_name = data['bus']
    name = bus_name
    

    #### voltages:
    v_hp, v_hn = sym.symbols(f'V_{name}_0_r,V_{name}_1_r', real=True)
    v_hp_i, v_hn_i = sym.symbols(f'V_{name}_0_i,V_{name}_1_i', real=True)
   
    ### HV-side
    e_h,v_h,i_h,p_h = sym.symbols(f'e_h_{name},v_h_{name},i_h_{name},p_h_{name}',real=True)
    R_h = sym.Symbol(f'R_h_{name}',real=True)
    v_ref = sym.Symbol(f'v_ref_{name}',real=True)
    m_h = sym.Symbol(f'm_h_{name}',real=True)
    i_g = sym.Symbol(f'i_g_{name}',real=True) 
    R_g = sym.Symbol(f'R_g_{name}',real=True)

    i_g = (v_hn + v_hp)/(2*R_g + R_h)
    v_og = i_g*R_g
    i_hp = (v_og + v_ref/2 - v_hp)/R_h
    i_hn = (v_og - v_ref/2 - v_hn)/R_h
    p_h = i_hp*v_ref

    # current injections dc LV side
    idx_hp_r,idx_hp_i = grid.node2idx(f'{name}','a')
    idx_hn_r,idx_hn_i = grid.node2idx(f'{name}','b')
    grid.dae['g'][idx_hp_r] += -i_hp 
    grid.dae['g'][idx_hn_r] += -i_hn
    grid.dae['g'][idx_hp_i] += -v_hp_i
    grid.dae['g'][idx_hn_i] += -v_hn_i

    v_ref = 800.0
    if "v_ref" in data:
        v_ref = data['v_ref']

    grid.dae['u_ini_dict'].update({f'v_ref_{name}':v_ref}) 
    grid.dae['u_run_dict'].update({f'v_ref_{name}':v_ref}) 
   
    grid.dae['params_dict'].update({f'R_h_{name}':1e-5})    
    grid.dae['params_dict'].update({f'R_g_{name}':3})   
    grid.dae['h_dict'].update({f'p_h_{name}':p_h})   

    grid.omega_coi_denominator += 1e6

def development():
    v_hp, v_hn, v_ref = sym.symbols(f'v_hp,v_hn,v_ref', real=True)
    i_hp, i_hn, i_g = sym.symbols(f'i_hp,i_hn,i_g', real=True)
    R_h, R_g = sym.symbols(f'R_h,R_g', real=True)

    v_og = i_g*R_g
    eq_i_hp = v_og + v_ref/2 - i_hp*R_h - v_hp
    eq_i_hn = v_og - v_ref/2 - i_hn*R_h - v_hn
    eq_i_g  = i_g + i_hp + i_hn

    result = sym.solve([eq_i_hp,eq_i_hn,eq_i_g],[i_hp,i_hn,i_g])
    print(result)

def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('dc_src_load.hjson')
    grid.uz_jacs = True
    grid.construct('dae_module')
    grid.compile('dae_module')

    import dae_module

    model = dae_module.model()
    model.ini({},'xy_0.json')
    model.report_y()
    model.report_z()



if __name__ == '__main__':

    #development()
    test()



