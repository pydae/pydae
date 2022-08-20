"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def milano4ord(grid,name,bus_name,data_dict):
    r"""
 
    **Auxiliar equations**

    .. math::
        :nowrap:
        
        \begin{eqnarray}
            v_d &=& V \sin(\delta - \theta) \\
            v_q &=& V \cos(\delta - \theta) \\
            p_e &=& i_d \left(v_d + R_a i_d\right) + i_q \left(v_q + R_a i_q\right)  \\   
            \omega_s &=& \omega_{coi}
        \end{eqnarray} 


    **Dynamic equations**

    .. math::
        :nowrap:
        
        \begin{eqnarray}
            \frac{ d\delta}{dt} &=& \Omega_b \left(\omega - \omega_s \right) - K_{\delta} \delta  \\
            \frac{ d\omega}{dt} &=& \frac{1}{2H} \left(p_m - p_e - D  \left(\omega - \omega_s \right) \right)  \\
            \frac{ de'_q}{dt} &=& \frac{1}{T'_{d0}} \left(-e'_q - \left(X_d - X'_d\right) i_d + v_f\right)  \\
            \frac{ de'_d}{dt} &=& \frac{1}{T'_{q0}} \left(-e'_d + \left(X_q - X'_q\right) i_q\right)
        \end{eqnarray} 


    **Algebraic equations**
    

    .. math::
        :nowrap:
        
        \begin{eqnarray}
             0  &=& v_q + R_a i_q + X'_d i_d - e'_q \\
             0  &=& v_d + R_a i_d - X'_q i_q - e'_d \\
             0  &=& i_d v_d + i_q v_q - p_g  \\
             0  &=& i_d v_q - i_q v_d - q_g 
        \end{eqnarray} 

    .. table:: Constants
        :widths: auto

        ================== =========== ============================================= =========== 
        Variable           Code        Description                                   Units
        ================== =========== ============================================= ===========
        :math:`S_n`        ``S_n``     Nominal power                                  VA
        :math:`H`          ``H``       Inertia constaant                              s
        :math:`S_n`        ``S_n``     Nominal power                                  VA
        :math:`D`          ``D``       Damping coefficient                            s
        :math:`X_q`        ``X_q``     q-axis synchronous reactance                   pu-m
        :math:`X'_q`       ``X1q``     q-axis transient reactance                     pu-m
        :math:`T'_{q0}`    ``T1q0``    q-axis open circuit transient time constant    s  
        :math:`X_d`        ``X_d``     d-axis synchronous reactance                   pu-m  
        :math:`X'_d`       ``X1d``     d-axis transient reactance                     pu-m
        :math:`T'_{d0}`    ``T1d0``    d-axis open circuit transient time constant    s
        :math:`R_a`        ``R_a``     Armature resistance                            pu-m
        :math:`K_{\delta}` ``K_delta`` Reference machine constant                     -
        :math:`K_{sec}`    ``K_sec``   Secondary frequency control participation      -
        ================== =========== ============================================= ===========

    .. table:: Dynamic states
        :widths: auto

        ================= =========== ============================================= =========== 
        Variable          Code        Description                                   Units
        ================= =========== ============================================= ===========
        :math:`\delta`    ``delta``    Rotor angle                                  rad
        :math:`\omega`    ``omega``    Rotor speed                                  pu
        :math:`e'_q`      ``e1q``      q-axis transient voltage                     pu
        :math:`e'_d`      ``e1d``      d-axis transient voltage                     pu
        ================= =========== ============================================= ===========


    .. table:: Algebraic states
        :widths: auto

        ================= =========== ============================================= =========== 
        Variable          Code        Description                                   Units
        ================= =========== ============================================= ===========
        :math:`i_d`       ``i_d``      d-axis current                               pu-m
        :math:`i_q`       ``i_q``      q-axis current                               pu-m
        :math:`p_g`       ``p_g``      Active power                                 pu-m
        :math:`q_g`       ``q_g``      Reactive power                               pu-m
        ================= =========== ============================================= ===========

    .. table:: Inputs
        :widths: auto

        ================= =========== ============================================= =========== 
        Variable          Code        Description                                   Units
        ================= =========== ============================================= ===========
        :math:`v_f`       ``v_f``      Field voltage                                pu-m
        :math:`p_m`       ``p_m``      Mechanical power                             pu-m
        ================= =========== ============================================= ===========

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
    params_list = ['S_n','Omega_b','H','T1d0','T1q0','X_d','X_q','X1d','X1q','D','R_a','K_delta','K_sec']
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     
    omega_s = omega_coi
                
    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
    de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
    de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)

    # algebraic equations   
    g_i_d  = v_q + R_a*i_q + X1d*i_d - e1q
    g_i_q  = v_d + R_a*i_d - X1q*i_q - e1d
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    
    # dae 
    f_syn = [ddelta,domega,de1q,de1d]
    x_syn = [ delta, omega, e1q, e1d]
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
    
    # outputs
    grid.dae['h_dict'].update({f"p_e_{name}":p_e})
    
    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]})
    
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