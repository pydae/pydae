# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


import sympy as sym


def avr_kundur(dae,syn_data,name):
    '''
        {"type":"sexs","K_a":100, "T_r":0.02, "K_ai":1e-6, "v_pss":0.0, "v_ref":1.0,"V_min":-5,"V_max":5.0,"K_aw":2.0}




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
             0 &=& - v_{f } + \\begin{cases} V_{min } & \\text{for}\\: V_{min } > K_{a } \\left(- v_{c } + v_{pss } + v_{ref }\\right) + K_{ai } \\xi_{v } \\\\V_{max } & \\text{for}\\: V_{max } < K_{a } \\left(- v_{c } + v_{pss } + v_{ref }\\right) + K_{ai } \\xi_{v } \\\\K_{a } \\left(- v_{c } + v_{pss } + v_{ref }\\right) + K_{ai } \\xi_{v } & \\text{otherwise} \\end{cases} \\
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
    
        T_r K_a K_aw
        v_ref  v_pss
        
        v_c
        v_f_nosat
        
     '''

    avr_data = syn_data['avr']
    
    v_t = sym.Symbol(f"V_{name}", real=True)   
    v_c = sym.Symbol(f"v_c_{name}", real=True)  
    xi_v  = sym.Symbol(f"xi_v_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    T_r = sym.Symbol(f"T_r_{name}", real=True) 
    K_a = sym.Symbol(f"K_a_{name}", real=True)
    K_ai = sym.Symbol(f"K_ai_{name}", real=True)
    V_min = sym.Symbol(f"V_min_{name}", real=True)
    V_max = sym.Symbol(f"V_max_{name}", real=True)
    K_aw = sym.Symbol(f"K_aw_{name}", real=True)   
    
    v_ref = sym.Symbol(f"v_ref_{name}", real=True) 
    v_pss = sym.Symbol(f"v_pss_{name}", real=True) 
    
    epsilon_v = v_ref - v_c + v_pss
    v_f_nosat = K_a*epsilon_v + K_ai*xi_v

    
    dv_c =   (v_t - v_c)/T_r
    dxi_v =   epsilon_v  - K_aw*(v_f_nosat - v_f) 
    
    g_v_f  =   sym.Piecewise((V_min, v_f_nosat<V_min),(V_max,v_f_nosat>V_max),(v_f_nosat,True)) - v_f 
  #  g_v_f  =   v_f_nosat - v_f 
  
  
    
    dae['f'] += [dv_c,dxi_v]
    dae['x'] += [ v_c, xi_v]
    dae['g'] += [g_v_f]
    dae['y_ini'] += [v_f] 
    dae['y_run'] += [v_f]  
    dae['params_dict'].update({str(K_a):avr_data['K_a']})
    dae['params_dict'].update({str(K_ai):avr_data['K_ai']})
    dae['params_dict'].update({str(T_r):avr_data['T_r']})  
    dae['params_dict'].update({str(V_min):avr_data['V_min']})  
    dae['params_dict'].update({str(V_max):avr_data['V_max']})  
    dae['params_dict'].update({str(K_aw):avr_data['K_aw']}) 
    dae['u_ini_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_ini_dict'].update({str(v_pss):avr_data['v_pss']})
    dae['u_run_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_run_dict'].update({str(v_pss):avr_data['v_pss']})
    dae['xy_0_dict'].update({str(xi_v):1})
    
    
def avr_kundurq(dae,syn_data,name):
    '''

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
             0 &=& - v_{f } + \\begin{cases} V_{min } & \\text{for}\\: V_{min } > K_{a } \\left(- v_{c } + v_{pss } + v_{ref }\\right) + K_{ai } \\xi_{v } \\\\V_{max } & \\text{for}\\: V_{max } < K_{a } \\left(- v_{c } + v_{pss } + v_{ref }\\right) + K_{ai } \\xi_{v } \\\\K_{a } \\left(- v_{c } + v_{pss } + v_{ref }\\right) + K_{ai } \\xi_{v } & \\text{otherwise} \\end{cases} \\
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
    
        T_r K_a K_aw
        v_ref  v_pss
        
        v_c
        v_f_nosat
        
     '''


    avr_data = syn_data['avr']
    
    v_t = sym.Symbol(f"V_{name}", real=True)   
    q_g = sym.Symbol(f"q_g_{name}", real=True) 
    q = sym.Symbol(f"q_{name}", real=True)   


    v_c = sym.Symbol(f"v_c_{name}", real=True)  
    xi_vq  = sym.Symbol(f"xi_vq_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    T_r = sym.Symbol(f"T_r_{name}", real=True) 
    K_a = sym.Symbol(f"K_a_{name}", real=True)
    K_ai = sym.Symbol(f"K_ai_{name}", real=True)
    V_min = sym.Symbol(f"V_min_{name}", real=True)
    V_max = sym.Symbol(f"V_max_{name}", real=True)
    K_aw = sym.Symbol(f"K_aw_{name}", real=True)   
    K_qv = sym.Symbol(f"K_qv_{name}", real=True)   

    
    v_ref = sym.Symbol(f"v_ref_{name}", real=True) 
    q_ref = sym.Symbol(f"q_ref_{name}", real=True) 
    v_pss = sym.Symbol(f"v_pss_{name}", real=True) 
    
    epsilon_v = v_ref - v_c + v_pss
    epsilon_q = q_ref - q
    epsilon_vq = (1-K_qv) * epsilon_v + K_qv*epsilon_q

    v_f_nosat = K_a*epsilon_vq + K_ai*xi_vq

    
    dv_c =   (v_t - v_c)/T_r
    dq   = (q_g - q)/T_r
    dxi_vq =   epsilon_vq  - K_aw*(v_f_nosat - v_f) 
    
    g_v_f  =   sym.Piecewise((V_min, v_f_nosat<V_min),(V_max,v_f_nosat>V_max),(v_f_nosat,True)) - v_f 
  #  g_v_f  =   v_f_nosat - v_f 
  
  
    
    dae['f'] += [dv_c,dq,dxi_vq]
    dae['x'] += [ v_c,q, xi_vq]
    dae['g'] += [g_v_f]
    dae['y_ini'] += [v_f] 
    dae['y_run'] += [v_f]  
    dae['params_dict'].update({str(K_a):avr_data['K_a']})
    dae['params_dict'].update({str(K_ai):avr_data['K_ai']})
    dae['params_dict'].update({str(T_r):avr_data['T_r']})  
    dae['params_dict'].update({str(V_min):avr_data['V_min']})  
    dae['params_dict'].update({str(V_max):avr_data['V_max']})  
    dae['params_dict'].update({str(K_aw):avr_data['K_aw']}) 
    dae['u_ini_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_ini_dict'].update({str(q_ref):0.0})
    dae['u_ini_dict'].update({str(v_pss):avr_data['v_pss']})
    dae['u_run_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_run_dict'].update({str(q_ref):0.0})

    dae['u_run_dict'].update({str(v_pss):avr_data['v_pss']})
    dae['params_dict'].update({str(K_qv):0.0})

    dae['xy_0_dict'].update({str(xi_vq):1})
