import sympy as sym
import numpy as np
from sympy import interpolating_spline


def bess_r(grid,data,name,bus_name):
    '''

    Battery Energy Storage System modeled as a voltage source 
    behind a resistor. 

    parameters
    ----------

    S_n: nominal power in VA
    U_n: nominal rms phase to phase voltage in V
    F_n: nominal frequency in Hz
    X_s: coupling reactance in pu (base machine S_n)
    R_s: coupling resistance in pu (base machine S_n)

    ## inputs
    
    soc_ref: state of charge desired value (pu)
    p_dc: DC power (W)

    
    ## algebraic state
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

    # commons

    ## inputs
    p_dc = sym.Symbol(f"p_dc_{name}", real=True)
    k_ini = sym.Symbol(f"k_soc_ini_{name}", real=True)

    ## algebraic states
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)
    p_soc = sym.Symbol(f"p_soc_{name}", real=True)

    # Battery
    E_kWh = sym.Symbol(f"E_kWh_{name}", real=True)
    soc_min = sym.Symbol(f"soc_min_{name}", real=True)
    soc_max = sym.Symbol(f"soc_max_{name}", real=True)
    K_p = sym.Symbol(f"K_soc_p_{name}", real=True)
    K_i = sym.Symbol(f"K_soc_i_{name}", real=True)
    B_0 = sym.Symbol(f"B_0_{name}", real=True)
    B_1 = sym.Symbol(f"B_1_{name}", real=True)
    R_bat = sym.Symbol(f"R_bat_{name}", real=True)
    soc_ref = sym.Symbol(f"soc_ref_{name}", real=True) 
    soc = sym.Symbol(f"soc_{name}", real=True) 
    xi_soc = sym.Symbol(f"xi_soc_{name}", real=True) 
    p_soc = sym.Symbol(f"p_soc_{name}", real=True) 

    i_dc_nosat = (p_dc)/v_dc
    i_dc = sym.Piecewise((i_dc_nosat,(i_dc_nosat>-500) & (i_dc_nosat<500)),(-500,i_dc_nosat<-500),(500,i_dc_nosat>500),(i_dc_nosat,True))
    i_dc = i_dc_nosat
    epsilon_soc = soc_ref - soc
    E_n = 1000*3600*E_kWh

    if 'socs' in data:
        socs = np.array(data['socs'])
        es = np.array(data['es'])
        e_min = np.min(es)  # minimun voltage
        e_max = np.max(es)  # maximum voltage
        interpolation = interpolating_spline(1, soc, socs, es)
        interpolation._args = tuple(list(interpolation._args) + [sym.functions.elementary.piecewise.ExprCondPair(e_max,True)])
        e = interpolation
        soc_ref_N = data['soc_0']
        e_ini = np.interp(soc_ref_N,socs,es)
    else:
        e = B_0 + B_1*soc
        e_ini = data['B_0']

    ## dynamic equations    
    dsoc = 1.0/E_n*(-i_dc*e)  # state of charge
    dxi_soc = epsilon_soc # soc controller integrator

    ## algebraic equations 
    g_v_dc = e - i_dc*R_bat - v_dc   # DC voltage
    g_p_soc = -p_soc - (K_p*epsilon_soc  + K_i*xi_soc) # soc controller power reference

    grid.dae['f'] += [dsoc, dxi_soc]
    grid.dae['x'] += [ soc,  xi_soc]

    grid.dae['g'] +=     [g_v_dc, g_p_soc]
    grid.dae['y_ini'] += [  v_dc,   p_soc]  
    grid.dae['y_run'] += [  v_dc,   p_soc]  

    soc_ref_N = 0.5
    if 'soc_0' in data:
        soc_ref_N = data['soc_0']

    grid.dae['u_ini_dict'].update({f'{str(soc_ref)}':soc_ref_N})
    grid.dae['u_run_dict'].update({f'{str(soc_ref)}':soc_ref_N})

    ## parameters  
    grid.dae['params_dict'].update({f"{K_p}":5.0}) 
    grid.dae['params_dict'].update({f"{K_i}":0.01})  
    grid.dae['params_dict'].update({f"{soc_min}":0.0}) 
    grid.dae['params_dict'].update({f"{soc_max}":1.0})           
    grid.dae['params_dict'].update({f"{E_kWh}":data['E_kWh']}) 

    if 'R_bat' in data:
        R_bat_N = data['R_bat']
    else:
        R_bat_N = 0.0

    grid.dae['params_dict'].update({f"{R_bat}":R_bat_N}) 

    grid.dae['xy_0_dict'].update({f"v_dc_{name}":e_ini})
    grid.dae['xy_0_dict'].update({f"soc_{name}":data['soc_0']})
    grid.dae['xy_0_dict'].update({f"xi_soc_{name}":0.0})
    grid.dae['xy_0_dict'].update({f"p_soc_{name}":0.0})

    if f'p_soc_{name}' in grid.dae['u_ini_dict']:
        grid.dae['u_ini_dict'].pop(f'p_soc_{name}')
        grid.dae['u_run_dict'].pop(f'p_soc_{name}')
    if f'v_dc_{name}' in grid.dae['u_ini_dict']:
        grid.dae['u_ini_dict'].pop(f'v_dc_{name}')
        grid.dae['u_run_dict'].pop(f'v_dc_{name}')


def test_vsg():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('bess_r.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 100e3
    V_n = 800
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    losses = 1.0
    A = Conduction_losses/(I_n**2)*losses
    B = 1*losses
    C = 0.02*S_n*losses
    model = temp.model()
    model.Dt = 1.0
    params = {'A_loss_A1':A,'B_loss_A1':B,'C_loss_A1':C,'R_bat_A1':0.001,'p_c_A1':0.0,
            'K_soc_p_A1':0.5,'K_soc_i_A1':0.0001}
    model.ini(params,'xy_0.json')
    # model.save_xy_0('xy_1.json')
    # model.report_params()
    # model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()

def test_pq():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('bess_r_ctrl_3ph_4w_pq.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 100e3
    V_n = 800
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    losses = 1.0
    A = Conduction_losses/(I_n**2)*losses
    B = 1*losses
    C = 0.02*S_n*losses
    model = temp.model()
    model.Dt = 1.0
    params = {'A_loss_A1':A,'B_loss_A1':B,'C_loss_A1':C,'R_bat_A1':0.001,'q_ref_A1':50e3, 
            'K_soc_p_A1':0.5,'K_soc_i_A1':0.0001}
    model.ini(params,'xy_0.json')
    # model.save_xy_0('xy_1.json')
    # model.report_params()
    # model.report_u()
    model.report_x()
    model.report_y()
    model.report_z()

if __name__ == '__main__':

    #development()
    test_pq()



