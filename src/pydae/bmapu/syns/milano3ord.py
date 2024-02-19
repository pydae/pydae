"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def milano3ord(grid,name,bus_name,data_dict):
    r"""
    Synchronous machine model of order 4 from Federico Milano book.

    """

    sin = sym.sin
    cos = sym.cos  

    # inputs
    V = sym.Symbol(f"V_{bus_name}", real=True)
    theta = sym.Symbol(f"theta_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    omega_coi = sym.Symbol("omega_coi", real=True)   
        
    # dynamic states
    delta = sym.Symbol(f"delta_{name}", real=True)
    omega = sym.Symbol(f"omega_{name}", real=True)
    e1q = sym.Symbol(f"e1q_{name}", real=True)
    e1d = sym.Symbol(f"e1d_{name}", real=True)

    # algebraic states
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)            
    p_g = sym.Symbol(f"p_g_{name}", real=True)
    q_g = sym.Symbol(f"q_g_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
    H = sym.Symbol(f"H_{name}", real=True)
    T1d0 = sym.Symbol(f"T1d0_{name}", real=True)
    T1q0 = sym.Symbol(f"T1q0_{name}", real=True)
    X_d = sym.Symbol(f"X_d_{name}", real=True)
    X_q = sym.Symbol(f"X_q_{name}", real=True)
    X1d = sym.Symbol(f"X1d_{name}", real=True)
    X1q = sym.Symbol(f"X1q_{name}", real=True)
    D = sym.Symbol(f"D_{name}", real=True)
    R_a = sym.Symbol(f"R_a_{name}", real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True)
    K_sat = sym.Symbol(f"K_sat_{name}", real=True)
    params_list = ['S_n','H','T1d0','T1q0','X_d','X_q','X1d','X1q','D','R_a','K_delta','K_sec']
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     
    omega_s = omega_coi
                
    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
    de1q = 1/T1d0*(-e1q*K_sat - (X_d - X1d)*i_d + v_f)

    # algebraic equations   
    g_i_d  = v_q + R_a*i_q + X1d*i_d - e1q
    g_i_q  = v_d + R_a*i_d - X1q*i_q
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    
    # dae 
    f_syn = [ddelta,domega,de1q]
    x_syn = [ delta, omega, e1q]
    g_syn = [g_i_d,g_i_q,g_p_g,g_q_g]
    y_syn = [  i_d,  i_q,  p_g,  q_g]
    
    grid.H_total += H
    grid.omega_coi_numerator += omega*H*S_n
    grid.omega_coi_denominator += H*S_n

    grid.dae['f'] += f_syn
    grid.dae['x'] += x_syn
    grid.dae['g'] += g_syn
    grid.dae['y_ini'] += y_syn  
    grid.dae['y_run'] += y_syn  
    
    if 'v_f' in data_dict:
        grid.dae['u_ini_dict'].update({f'{v_f}':{data_dict['v_f']}})
        grid.dae['u_run_dict'].update({f'{v_f}':{data_dict['v_f']}})
    else:
        grid.dae['u_ini_dict'].update({f'{v_f}':1.0})
        grid.dae['u_run_dict'].update({f'{v_f}':1.0})

    if 'p_m' in data_dict:
        grid.dae['u_ini_dict'].update({f'{p_m}':{data_dict['p_m']}})
        grid.dae['u_run_dict'].update({f'{p_m}':{data_dict['p_m']}})
    else:
        grid.dae['u_ini_dict'].update({f'{p_m}':1.0})
        grid.dae['u_run_dict'].update({f'{p_m}':1.0})

    grid.dae['xy_0_dict'].update({str(omega):1.0})
    grid.dae['xy_0_dict'].update({str(e1q):1.0})
    grid.dae['xy_0_dict'].update({str(i_q):0.5})

    
    # outputs
    grid.dae['h_dict'].update({f"p_e_{name}":p_e})
    grid.dae['h_dict'].update({f"v_f_{name}":v_f})
    grid.dae['h_dict'].update({f"p_m_{name}":p_m})
    
    if 'F_n' in data_dict:
        grid.dae['params_dict'].update({f"Omega_b_{name}":2*np.pi*data_dict['F_n']})
    else:
        grid.dae['params_dict'].update({f"Omega_b_{name}":data_dict['Omega_b']})

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]})
    
    grid.dae['params_dict'].update({f"K_sat_{name}":1.0})

    # if 'avr' in syn_data:
    #     add_avr(grid.dae,syn_data)
    #     grid.dae['u_ini_dict'].pop(str(v_f))
    #     grid.dae['u_run_dict'].pop(str(v_f))
    #     grid.dae['xy_0_dict'].update({str(v_f):1.5})

    # if 'gov' in syn_data:
    #     add_gov(grid.dae,syn_data)  
    #     grid.dae['u_ini_dict'].pop(str(p_m))
    #     grid.dae['u_run_dict'].pop(str(p_m))
    #     grid.dae['xy_0_dict'].update({str(p_m):0.5})

    # if 'pss' in syn_data:
    #     add_pss(grid.dae,syn_data)  

    p_W   = p_g * S_n
    q_var = q_g * S_n

    return p_W,q_var


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import pytest

    grid = bmapu('milano3ord.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')

    import temp

    model = temp.model()

    v_ref_1 = 1.05
    model.ini({'p_m_1':0.5,'v_ref_1':v_ref_1},'xy_0.json')

    # assert model.get_value('V_1') == pytest.approx(v_ref_1, rel=0.001)

    model.run(1.0,{})
    model.run(1.1,{'b_shunt_1':-500})
    #model.run(2.0,{'b_shunt_1':0})

    model.post()
    # # assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)

    # model.ini({'p_m_1':0.5,'v_ref_1':1.0},'xy_0.json')
    # model.run(1.0,{})
    # model.run(15.0,{'v_ref_1':1.05})
    # model.post()

    import matplotlib.pyplot as plt

    fig,axes = plt.subplots()
    axes.plot(model.Time,model.get_values('V_1'))
    fig.savefig('milano4ord_fault.svg')


if __name__ == '__main__':

    #development()
    test()