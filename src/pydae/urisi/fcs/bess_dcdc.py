import sympy as sym
import numpy as np
from sympy import interpolating_spline


def bess_dcdc(grid,data):
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

    # commons
    ## voltages:
    v_hp, v_hn = sym.symbols(f'V_{bus_name}_0_r,V_{bus_name}_1_r', real=True)
    v_hp_i, v_hn_i = sym.symbols(f'V_{bus_name}_0_i,V_{bus_name}_1_i', real=True)
   
    ## algebraic states
    i_h = sym.Symbol(f"i_h_{name}", real=True)
    p_l = sym.Symbol(f"p_l_{name}", real=True)

    ### HV-side
    e_h,v_h,i_h,p_h = sym.symbols(f'e_h_{name},v_h_{name},i_h_{name},p_h_{name}',real=True)
    R_h = sym.Symbol(f'R_h_{name}',real=True)
    p_r_ref = sym.Symbol(f'p_r_ref_{name}',real=True)
    m_h = sym.Symbol(f'm_h_{name}',real=True)
    i_g = sym.Symbol(f'i_g_{name}',real=True) 
    R_g = sym.Symbol(f'R_g_{name}',real=True)

    # dynamic states
    soc = sym.Symbol(f"soc_{name}", real=True)
    xi_soc = sym.Symbol(f"xi_soc_{name}", real=True)

    # algebraic states
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)


    # Battery
    E_kWh = sym.Symbol(f"E_kWh_{name}", real=True)
    soc_min = sym.Symbol(f"soc_min_{name}", real=True)
    soc_max = sym.Symbol(f"soc_max_{name}", real=True)
    A_loss = sym.Symbol(f"A_loss_{name}", real=True)
    B_loss = sym.Symbol(f"B_loss_{name}", real=True)
    C_loss = sym.Symbol(f"C_loss_{name}", real=True)
    K_p = sym.Symbol(f"K_p_{name}", real=True)
    K_i = sym.Symbol(f"K_i_{name}", real=True)
    B_0 = sym.Symbol(f"B_0_{name}", real=True)
    B_1 = sym.Symbol(f"B_1_{name}", real=True)
    B_2 = sym.Symbol(f"B_2_{name}", real=True)
    B_3 = sym.Symbol(f"B_3_{name}", real=True)
    R_bat = sym.Symbol(f"R_bat_{name}", real=True)
    soc_ref = sym.Symbol(f"soc_ref_{name}", real=True) 

    epsilon_soc = soc_ref - soc
    p_soc = -(K_p*epsilon_soc + K_i*xi_soc)
    E_n = 1000*3600*E_kWh

    if 'socs' in data:
        socs = np.array(data['socs'])
        es = np.array(data['es'])
        interpolation = interpolating_spline(1, soc, socs, es)
        interpolation._args = tuple(list(interpolation._args) + [sym.functions.elementary.piecewise.ExprCondPair(0,True)])
        e = interpolation
    else:
        e = B_0 + B_1*soc + B_2*soc**2 + B_3*soc**3
    i_dc = p_l/v_dc

    ## dynamic equations    
    dsoc = 1/E_n*(-i_dc*e)   
    dxi_soc = epsilon_soc     

    ## algebraic equations   
    g_v_dc = e - i_dc*R_bat - v_dc

    grid.dae['f'] += [dsoc, dxi_soc]
    grid.dae['x'] += [ soc,  xi_soc] 

    grid.dae['g'] +=     [g_v_dc]
    grid.dae['y_ini'] += [  v_dc]  
    grid.dae['y_run'] += [  v_dc]  

    soc_ref_N = 0.5
    if 'soc_0' in data:
        soc_ref_N = data['soc_0']

    grid.dae['u_ini_dict'].update({f'{str(soc_ref)}':soc_ref_N})
    grid.dae['u_run_dict'].update({f'{str(soc_ref)}':soc_ref_N})

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

    grid.dae['xy_0_dict'].update({f"v_dc_{name}":B_0_N})


    # VSC

    ## inputs
    p_r_ref = sym.Symbol(f"p_r_ref_{name}", real=True)
    
    ## algebraic states
  

    p_ref = sym.Piecewise((p_r_ref,(p_r_ref <=0.0) & (soc<soc_max)),
                          (p_r_ref,(p_r_ref > 0.0) & (soc>soc_min)),
                        (0.0,True)) + p_soc
    
    v_l = v_dc
    p_h = p_ref
    v_h = v_hp - v_hn
    i_hp =  i_h
    i_hn = -i_h
    p_loss = A_loss*i_hp**2 + B_loss*sym.Abs(i_hp) + C_loss

    eq_i_h = -v_h*i_h + p_h
    eq_p_l = -p_l + p_h + p_loss

    grid.dae['g'] += [eq_i_h, eq_p_l]
    grid.dae['y_ini'] += [ i_h, p_l]  
    grid.dae['y_run'] += [ i_h, p_l]  

    grid.dae['u_ini_dict'].update({f'{str(p_r_ref)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(p_r_ref)}':0.0})

    ## current injections dc LV side
    idx_hp_r,idx_hp_i = grid.node2idx(f'{bus_name}','a')
    idx_hn_r,idx_hn_i = grid.node2idx(f'{bus_name}','b')
    grid.dae['g'] [idx_hp_r] += -i_hp 
    grid.dae['g'] [idx_hn_r] += -i_hn
    grid.dae['g'] [idx_hp_i] +=  v_hp_i/1e3
    grid.dae['g'] [idx_hn_i] +=  v_hn_i/1e3
   
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

    grid.dae['params_dict'].update({f"{E_kWh}":data['E_kWh']}) 

    grid.dae['xy_0_dict'].update({str(v_dc):B_0_N})
    grid.dae['xy_0_dict'].update({str(soc):0.5})
    grid.dae['xy_0_dict'].update({str(xi_soc):0.5})

    ## outputs
    grid.dae['h_dict'].update({f'p_loss_{name}':p_loss})
    grid.dae['h_dict'].update({f'i_dc_{name}':i_dc})
    grid.dae['h_dict'].update({f'p_h_{name}':p_h})

def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('bess_dcdc.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 500e3
    V_n = 800
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    losses = 1.0
    A = Conduction_losses/(I_n**2)*losses
    B = 1*losses
    C = 0.02*S_n*losses
    model = temp.model()
    model.ini({'A_loss_D1':A,'B_loss_D1':B,'C_loss_D1':C},'xy_0.json')
    #model.report_params()
    model.report_u()
    model.report_y()
    model.report_z()



if __name__ == '__main__':

    #development()
    test()



