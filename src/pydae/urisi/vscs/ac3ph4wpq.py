import numpy as np
import sympy as sym
  
    

def ac3ph4wpq(grid,vsc_data):
    '''
    Converter type p_ac,q_ac 3 phase 4 wire
    
    '''
    
    ## Model data
    bus_ac_name = vsc_data['bus_ac']
    bus_dc_name = vsc_data['bus_dc']  
    a_value  = vsc_data['a']   
    b_value  = vsc_data['b']   
    c_value  = vsc_data['c']   
    
    ## Voltages:
    v_a_r,v_a_i = sym.symbols(f'v_{bus_ac_name}_a_r,v_{bus_ac_name}_a_i',real=True)
    v_b_r,v_b_i = sym.symbols(f'v_{bus_ac_name}_b_r,v_{bus_ac_name}_b_i',real=True)
    v_c_r,v_c_i = sym.symbols(f'v_{bus_ac_name}_c_r,v_{bus_ac_name}_c_i',real=True)
    v_n_r,v_n_i = sym.symbols(f'v_{bus_ac_name}_n_r,v_{bus_ac_name}_n_i',real=True)
    v_dc_a_r,v_dc_n_r = sym.symbols(f'v_{bus_dc_name}_a_r,v_{bus_dc_name}_n_r',real=True)

    ## Currents:
    i_a_r,i_a_i = sym.symbols(f'i_vsc_{bus_ac_name}_a_r,i_vsc_{bus_ac_name}_a_i',real=True)
    i_b_r,i_b_i = sym.symbols(f'i_vsc_{bus_ac_name}_b_r,i_vsc_{bus_ac_name}_b_i',real=True)
    i_c_r,i_c_i = sym.symbols(f'i_vsc_{bus_ac_name}_c_r,i_vsc_{bus_ac_name}_c_i',real=True) 
    i_n_r,i_n_i = sym.symbols(f'i_vsc_{bus_ac_name}_n_r,i_vsc_{bus_ac_name}_n_i',real=True) 

    i_dc_a_r,i_dc_n_r = sym.symbols(f'i_vsc_{bus_dc_name}_a_r,i_vsc_{bus_dc_name}_n_r',real=True)


    
    ## Body:
        
    ### AC-side
    p_ac_a,p_ac_b,p_ac_c = sym.symbols(f'p_vsc_a_{bus_ac_name},p_vsc_b_{bus_ac_name},p_vsc_c_{bus_ac_name}',real=True)
    q_ac_a,q_ac_b,q_ac_c = sym.symbols(f'q_vsc_a_{bus_ac_name},q_vsc_b_{bus_ac_name},q_vsc_c_{bus_ac_name}',real=True)

    p_dc,p_loss = sym.symbols(f'p_vsc_{bus_dc_name},p_vsc_loss_{bus_ac_name}',real=True)
    #coef_a,coef_b,coef_c = sym.symbols(f'coef_a_{bus_ac_name},coef_b_{bus_ac_name},coef_c_{bus_ac_name}',real=True)
    
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

    eq_i_a_r =  sym.re(s_a) - p_ac_a
    eq_i_b_r =  sym.re(s_b) - p_ac_b
    eq_i_c_r =  sym.re(s_c) - p_ac_c
    eq_i_a_i =  sym.im(s_a) - q_ac_a
    eq_i_b_i =  sym.im(s_b) - q_ac_b
    eq_i_c_i =  sym.im(s_c) - q_ac_c
    eq_i_n_r =  sym.re(i_a+i_b+i_c+i_n)
    eq_i_n_i =  sym.im(i_a+i_b+i_c+i_n)

    i_abc_list  = [i_a_r,i_a_i,i_b_r,i_b_i,i_c_r,i_c_i,i_n_r,i_n_i]
    for itg in [1,2,3,4]:
        bus_idx = grid.nodes.index(f'{bus_ac_name}.{itg}')
        g_idx = bus_idx - grid.N_nodes_v
        grid.dae['g'][2*g_idx+0] += i_abc_list[2*(itg-1)  ]
        grid.dae['g'][2*g_idx+1] += i_abc_list[2*(itg-1)+1]

    ### DC side
    a,b,c = sym.symbols(f'a_{bus_ac_name},b_{bus_ac_name},c_{bus_ac_name}',real=True)
    i_rms_a = sym.sqrt(i_a_r**2+i_a_i**2+1e-6) 
    i_rms_b = sym.sqrt(i_b_r**2+i_b_i**2+1e-6) 
    i_rms_c = sym.sqrt(i_c_r**2+i_c_i**2+1e-6) 
    i_rms_n = sym.sqrt(i_n_r**2+i_n_i**2+1e-6) 

    p_loss_a = a + b*i_rms_a + c*i_rms_a*i_rms_a
    p_loss_b = a + b*i_rms_b + c*i_rms_b*i_rms_b
    p_loss_c = a + b*i_rms_c + c*i_rms_c*i_rms_c
    p_loss_n = a + b*i_rms_n + c*i_rms_n*i_rms_n

    p_vsc_loss = p_loss_a + p_loss_b + p_loss_c + p_loss_n
    p_ac = p_ac_a + p_ac_b + p_ac_c

    eq_p_loss = p_loss - p_vsc_loss
    eq_i_dc_a_r = i_dc_a_r + p_dc/(v_dc_a_r-v_dc_n_r+1e-8)
    eq_i_dc_n_r = i_dc_n_r + p_dc/(v_dc_n_r-v_dc_a_r+1e-8)
    eq_p_dc = -p_dc + p_ac + p_loss #- sym.Piecewise((p_loss, p_ac > 0), (-p_loss, p_ac < 0),(p_loss, True))


    ## DAE system update
    grid.dae['g'] += [eq_i_a_r,eq_i_a_i,
                      eq_i_b_r,eq_i_b_i,
                      eq_i_c_r,eq_i_c_i,
                      eq_i_n_r,eq_i_n_i]
    
    grid.dae['y'] += [   i_a_r,   i_a_i,
                         i_b_r,   i_b_i,
                         i_c_r,   i_c_i,
                         i_n_r,   i_n_i]

    grid.dae['g'] +=  [eq_i_dc_a_r,eq_i_dc_n_r,eq_p_dc,eq_p_loss]
    grid.dae['y'] +=  [   i_dc_a_r,   i_dc_n_r,   p_dc,   p_loss]  
    #grid.dae['u'].update({f'p_vsc_{bus_ac_name}':0.0,f'q_vsc_{bus_ac_name}':0.0}) 
    grid.dae['xy_0_dict'].update({f'v_{bus_dc_name}_a_r':800.0,f'v_{bus_dc_name}_n_r':0.01})
    
    #grid.dae['u'].pop(f'i_{bus_dc_name}_a_r')
    #grid.dae['u'].pop(f'i_{bus_dc_name}_n_r')
    grid.dae['params'].update({f'a_{bus_ac_name}':a_value,f'b_{bus_ac_name}':b_value,f'c_{bus_ac_name}':c_value})
    grid.dae['u'].update({f'{str(p_ac_a)}':0.0,f'{str(p_ac_b)}':0.0,f'{str(p_ac_c)}':0.0})
    grid.dae['u'].update({f'{str(q_ac_a)}':0.0,f'{str(q_ac_b)}':0.0,f'{str(q_ac_c)}':0.0})

    ## Add current injections to grid equations:
    bus_idx = grid.nodes.index(f'{bus_dc_name}.{1}')
    g_idx = bus_idx - grid.N_nodes_v
    grid.dae['g'][2*g_idx+0] += i_dc_a_r
    grid.dae['g'][2*g_idx+1] += 0.0   

    bus_idx = grid.nodes.index(f'{bus_dc_name}.{4}')
    g_idx = bus_idx - grid.N_nodes_v
    grid.dae['g'][2*g_idx+0] += i_dc_n_r
    grid.dae['g'][2*g_idx+1] += 0.0 

    grid.dae['h_dict'].update({f'p_vsc_{bus_ac_name}':sym.re(s_a)+sym.re(s_b)+sym.re(s_c)})
    grid.dae['h_dict'].update({f'p_vsc_loss_{bus_ac_name}':(p_loss_a+p_loss_b+p_loss_c+p_loss_n)})

