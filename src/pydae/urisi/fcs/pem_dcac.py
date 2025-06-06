import sympy as sym
import numpy as np
from sympy import interpolating_spline


def bess_dcac(grid,data):
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
    bus_ac_name = bus_name

    # commons

    ## algebraic states
    p_dc = sym.Symbol(f"p_dc_{name}", real=True)

    ### HV-side
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
    p_soc = (K_p*epsilon_soc + K_i*xi_soc)
    E_n = 1000*3600*E_kWh

    if 'socs' in data:
        socs = np.array(data['socs'])
        es = np.array(data['es'])
        interpolation = interpolating_spline(1, soc, socs, es)
        interpolation._args = tuple(list(interpolation._args) + [sym.functions.elementary.piecewise.ExprCondPair(0,True)])
        e = interpolation
    else:
        e = B_0 + B_1*soc + B_2*soc**2 + B_3*soc**3
    i_dc = p_dc/v_dc

    ## dynamic equations    
    dsoc = 1/E_n*(-i_dc*e) #*0  + 1e-5*(soc_ref - soc)
    dxi_soc = epsilon_soc #-  1e-8*xi_soc  

    ## algebraic equations 
    #e = 867.0  
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
    grid.dae['params_dict'].update({f"{K_p}":1e-4}) 
    grid.dae['params_dict'].update({f"{K_i}":1e-4})  
    grid.dae['params_dict'].update({f"{soc_min}":0.0}) 
    grid.dae['params_dict'].update({f"{soc_max}":1.0})           
    grid.dae['params_dict'].update({f"{E_kWh}":data['E_kWh']}) 

    if 'B_0' in data:
        B_0_N = data['B_0']
        B_1_N = data['B_1']
        B_2_N = data['B_2']
        B_3_N = data['B_3']
    else:
        B_0_N = 800
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

    grid.dae['xy_0_dict'].update({f"v_dc_{name}":800})


    # VSC

    ## inputs
    p_r_ref = sym.Symbol(f"p_r_ref_{name}", real=True)
    
    ## algebraic states
  

    p_ref = sym.Piecewise((p_r_ref,(p_r_ref <=0.0) & (soc<soc_max)),
                        (p_r_ref,(p_r_ref > 0.0) & (soc>soc_min)),
                        (0.0,True)) + p_soc
    
    ### AC-side
    p_a_ref,p_b_ref,p_c_ref = sym.symbols(f'p_vsc_a_ref_{bus_ac_name},p_vsc_b_ref_{bus_ac_name},p_vsc_c_ref_{bus_ac_name}',real=True)
    q_a_ref,q_b_ref,q_c_ref = sym.symbols(f'q_vsc_a_ref_{bus_ac_name},q_vsc_b_ref_{bus_ac_name},q_vsc_c_ref_{bus_ac_name}',real=True)
    p_ref,q_ref = sym.symbols(f'p_vsc_ref_{bus_ac_name},q_vsc_ref_{bus_ac_name}',real=True)

    p_ac,q_ac,p_loss = sym.symbols(f'p_vsc_{bus_ac_name},q_vsc_{bus_ac_name},p_vsc_loss_{bus_ac_name}',real=True)
    p_a_d,p_b_d,p_c_d,p_n_d = sym.symbols(f'p_a_d_{bus_ac_name},p_b_d_{bus_ac_name},p_c_d_{bus_ac_name},p_n_d_{bus_ac_name}',real=True)
   
    #### AC voltages:
    v_a_r,v_b_r,v_c_r,v_n_r = sym.symbols(f'V_{bus_ac_name}_0_r,V_{bus_ac_name}_1_r,V_{bus_ac_name}_2_r,V_{bus_ac_name}_3_r', real=True)
    v_a_i,v_b_i,v_c_i,v_n_i = sym.symbols(f'V_{bus_ac_name}_0_i,V_{bus_ac_name}_1_i,V_{bus_ac_name}_2_i,V_{bus_ac_name}_3_i', real=True)
   
    #### AC currents:
    i_a_r,i_a_i = sym.symbols(f'i_vsc_{bus_ac_name}_a_r,i_vsc_{bus_ac_name}_a_i',real=True)
    i_b_r,i_b_i = sym.symbols(f'i_vsc_{bus_ac_name}_b_r,i_vsc_{bus_ac_name}_b_i',real=True)
    i_c_r,i_c_i = sym.symbols(f'i_vsc_{bus_ac_name}_c_r,i_vsc_{bus_ac_name}_c_i',real=True)
    i_n_r,i_n_i = sym.symbols(f'i_vsc_{bus_ac_name}_n_r,i_vsc_{bus_ac_name}_n_i',real=True)

    # algebraic states dc side
    A_loss,B_loss,C_loss = sym.symbols(f'A_loss_{bus_ac_name},B_loss_{bus_ac_name},C_loss_{bus_ac_name}',real=True)     

    i_a_rms = sym.sqrt(i_a_r**2+i_a_i**2 + 1e-2) 
    i_b_rms = sym.sqrt(i_b_r**2+i_b_i**2 + 1e-2) 
    i_c_rms = sym.sqrt(i_c_r**2+i_c_i**2 + 1e-2) 
    i_n_rms = sym.sqrt(i_n_r**2+i_n_i**2 + 1e-2) 

    p_loss_a = C_loss + B_loss*i_a_rms + A_loss*i_a_rms*i_a_rms
    p_loss_b = C_loss + B_loss*i_b_rms + A_loss*i_b_rms*i_b_rms
    p_loss_c = C_loss + B_loss*i_c_rms + A_loss*i_c_rms*i_c_rms
    p_loss_n = C_loss + B_loss*i_n_rms + A_loss*i_n_rms*i_n_rms

    v_a = v_a_r + 1j*v_a_i
    v_b = v_b_r + 1j*v_b_i
    v_c = v_c_r + 1j*v_c_i
    v_n = v_n_r + 1j*v_n_i

    i_a = i_a_r + 1j*i_a_i
    i_b = i_b_r + 1j*i_b_i
    i_c = i_c_r + 1j*i_c_i
    i_n = i_n_r + 1j*i_n_i

    s_a = (v_a - v_n) * sym.conjugate(i_a)
    s_b = (v_b - v_n) * sym.conjugate(i_b)
    s_c = (v_c - v_n) * sym.conjugate(i_c)
    s_n = (v_n) * sym.conjugate(i_n)   

    p_a = p_a_ref + p_ref/3 + p_soc/3
    p_b = p_b_ref + p_ref/3 + p_soc/3
    p_c = p_c_ref + p_ref/3 + p_soc/3
    q_a = q_a_ref + q_ref/3
    q_b = q_b_ref + q_ref/3
    q_c = q_c_ref + q_ref/3

    # from powers to currents:
    eq_i_a_r =  sym.re(s_a) - p_a
    eq_i_b_r =  sym.re(s_b) - p_b
    eq_i_c_r =  sym.re(s_c) - p_c
    eq_i_a_i =  sym.im(s_a) - q_a
    eq_i_b_i =  sym.im(s_b) - q_b
    eq_i_c_i =  sym.im(s_c) - q_c
    
    eq_i_n_r = i_n_r + i_a_r + i_b_r + i_c_r
    eq_i_n_i = i_n_i + i_a_i + i_b_i + i_c_i
    
    p_ac = p_a + p_b + p_c
    p_loss_total = p_loss_a+p_loss_b+p_loss_c+p_loss_n 

    eq_p_dc = -p_dc +  p_ac + p_loss_total

    # eq_i_pos = -i_pos + p_dc/v_dc
    # eq_i_neg = -i_neg - p_dc/v_dc
    #eq_v_og = -v_og/R_gdc - i_pos - i_neg 


    grid.dae['g'] += [eq_i_a_r,eq_i_a_i,
                      eq_i_b_r,eq_i_b_i,
                      eq_i_c_r,eq_i_c_i,
                      eq_i_n_r,eq_i_n_i,
                     # eq_i_pos,eq_i_neg,#eq_v_og,
                      eq_p_dc
                      ]
    
    grid.dae['y_ini'] += [i_a_r,   i_a_i,
                          i_b_r,   i_b_i,
                          i_c_r,   i_c_i,
                          i_n_r,   i_n_i,
                          p_dc]

    grid.dae['y_run'] += [i_a_r,   i_a_i,
                          i_b_r,   i_b_i,
                          i_c_r,   i_c_i,
                          i_n_r,   i_n_i,
                          p_dc]

    # current injections ac side
    for ph in ['a','b','c','n']:
        i_s_r = sym.Symbol(f'i_vsc_{bus_ac_name}_{ph}_r', real=True)
        i_s_i = sym.Symbol(f'i_vsc_{bus_ac_name}_{ph}_i', real=True)  
        idx_r,idx_i = grid.node2idx(bus_ac_name,ph)
        grid.dae['g'] [idx_r] += -i_s_r
        grid.dae['g'] [idx_i] += -i_s_i
        i_s = i_s_r + 1j*i_s_i
        i_s_m = np.abs(i_s)
        grid.dae['h_dict'].update({f'i_vsc_{bus_ac_name}_{ph}_m':i_s_m})
    
      
    grid.dae['u_ini_dict'].update({f'{str(p_a_ref)}':0.0,f'{str(p_b_ref)}':0.0,f'{str(p_c_ref)}':0.0}) 
    grid.dae['u_ini_dict'].update({f'{str(q_a_ref)}':0.0,f'{str(q_b_ref)}':0.0,f'{str(q_c_ref)}':0.0}) 
    grid.dae['u_ini_dict'].update({f'{str(p_ref)}':0.0}) 
    grid.dae['u_ini_dict'].update({f'{str(q_ref)}':0.0}) 

    grid.dae['u_run_dict'].update({f'{str(p_a_ref)}':0.0,f'{str(p_b_ref)}':0.0,f'{str(p_c_ref)}':0.0}) 
    grid.dae['u_run_dict'].update({f'{str(q_a_ref)}':0.0,f'{str(q_b_ref)}':0.0,f'{str(q_c_ref)}':0.0}) 
    grid.dae['u_run_dict'].update({f'{str(p_ref)}':0.0}) 
    grid.dae['u_run_dict'].update({f'{str(q_ref)}':0.0}) 


    #grid.dae['u'].pop(str(v_dc_a_r))
    grid.dae['params_dict'].update({f'A_loss_{bus_ac_name}':0.0,f'B_loss_{bus_ac_name}':0.0,f'C_loss_{bus_ac_name}':0.0})
    
    grid.dae['h_dict'].update({f'p_vsc_{bus_ac_name}':sym.re(s_a)+sym.re(s_b)+sym.re(s_c)+sym.re(s_n)})
    grid.dae['h_dict'].update({f'p_vsc_loss_{bus_ac_name}':(p_loss_total)})
    grid.dae['h_dict'].update({f'v_dc_{bus_ac_name}':v_dc})



def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    # grid = urisi('bess_dcac.hjson')
    # grid.uz_jacs = True
    # grid.construct('temp')
    # grid.compile('temp')

    import temp

    S_n = 100e3
    V_n = 400
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    losses = 1
    A = Conduction_losses/(I_n**2)*losses
    B = 1*losses
    C = 0.02*S_n*losses
    model = temp.model()
    model.ini({'A_loss_A1':A,'B_loss_A1':B,'C_loss_A1':C,'R_bat_A1':20e-3,'soc_ref_A1':1.0,
               'p_vsc_ref_A1':0e3},'xy_0.json')
    #model.report_params()
    model.report_u()
    model.report_y()
    model.report_z()
    model.Dt = 1.0
    model.run(3000, {'p_vsc_ref_A1': 100e3})
    model.run(6000, {'p_vsc_ref_A1':-100e3})

    model.post()
    model.report_x()
    model.report_z()

    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(nrows=2,ncols=2)

    # axes[0,0].plot(model.Time,model.get_values('p_h_D1'), label='p_h')
    axes[0,1].plot(model.Time,model.get_values('v_dc_A1'), label='v_dc')
    axes[1,0].plot(model.Time,model.get_values('soc_A1'), label='soc')

    fig.savefig('bess_dcac.svg')


    for ax in axes.flatten():
        ax.grid()
        ax.legend()
        ax.set_xlim(0,model.Time[-1])
    axes[1,0].set_xlabel('Time (s)')
    axes[1,1].set_xlabel('Time (s)')

if __name__ == '__main__':

    #development()
    test()



