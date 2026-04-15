
import numpy as np
import sympy as sym

def ac1ph2wpq(grid,vsc_data):
    '''
    Converter type p_ac,q_ac 1 phase 2 ac wire
    
    '''
    
    bus_ac_name = vsc_data['bus_ac']
    bus_dc_name = vsc_data['bus_dc']  
    a_value  = vsc_data['a']   
    b_value  = vsc_data['b']   
    c_value  = vsc_data['c'] 
    
    
    ### AC-side
    p_ac,q_ac,p_dc = sym.symbols(f'p_{bus_ac_name},q_{bus_ac_name},p_{bus_dc_name}',real=True)

    #### AC voltages:
    v_a_r,v_a_i = sym.symbols(f'v_{bus_ac_name}_a_r,v_{bus_ac_name}_a_i',real=True)
    v_b_r,v_b_i = sym.symbols(f'v_{bus_ac_name}_b_r,v_{bus_ac_name}_b_i',real=True)
    v_c_r,v_c_i = sym.symbols(f'v_{bus_ac_name}_c_r,v_{bus_ac_name}_c_i',real=True)
    v_n_r,v_n_i = sym.symbols(f'v_{bus_ac_name}_n_r,v_{bus_ac_name}_n_i',real=True)
    #### AC currents:
    i_a_r,i_a_i = sym.symbols(f'i_vsc_{bus_ac_name}_a_r,i_vsc_{bus_ac_name}_a_i',real=True)
    i_b_r,i_b_i = sym.symbols(f'i_vsc_{bus_ac_name}_b_r,i_vsc_{bus_ac_name}_b_i',real=True)
    i_c_r,i_c_i = sym.symbols(f'i_vsc_{bus_ac_name}_c_r,i_vsc_{bus_ac_name}_c_i',real=True)
    i_n_r,i_n_i = sym.symbols(f'i_vsc_{bus_ac_name}_n_r,i_vsc_{bus_ac_name}_n_i',real=True)
    
    coef_a,coef_b,coef_c = sym.symbols(f'coef_a_{bus_ac_name},coef_b_{bus_ac_name},coef_c_{bus_ac_name}',real=True)
    
    v_a = v_a_r + 1j*v_a_i
    v_b = v_b_r + 1j*v_b_i
    v_c = v_c_r + 1j*v_c_i
    v_n = v_n_r + 1j*v_n_i

    i_a = i_a_r + 1j*i_a_i
    i_b = i_b_r + 1j*i_b_i
    i_c = i_c_r + 1j*i_c_i

    s_a_g = (v_a-v_n) * sym.conjugate(i_a)
    s_b_g = (v_b-v_n) * sym.conjugate(i_b)
    s_c_g = (v_c-v_n) * sym.conjugate(i_c)

    eq_i_a_r =  sym.re(s_a_g) - p_ac*coef_a
    eq_i_b_r =  sym.re(s_b_g) - p_ac*coef_b
    eq_i_c_r =  sym.re(s_c_g) - p_ac*coef_c
    eq_i_a_i =  sym.im(s_a_g) - q_ac*coef_a
    eq_i_b_i =  sym.im(s_b_g) - q_ac*coef_b
    eq_i_c_i =  sym.im(s_c_g) - q_ac*coef_c
    eq_i_n_r =  i_n_r + i_a_r + i_b_r + i_c_r
    eq_i_n_i =  i_n_i + i_a_i + i_b_i + i_c_i

    grid.dae['g'] += [eq_i_a_r,eq_i_a_i,
                      eq_i_b_r,eq_i_b_i,
                      eq_i_c_r,eq_i_c_i,
                      eq_i_n_r,eq_i_n_i,]
    grid.dae['y'] += [   i_a_r,   i_a_i,
                         i_b_r,   i_b_i,
                         i_c_r,   i_c_i,
                         i_n_r,   i_n_i]

    i_abc_list  = [i_a_r,i_a_i,i_b_r,i_b_i,i_c_r,i_c_i,i_n_r,i_n_i]
    for itg in [1,2,3,4]:
        bus_idx = grid.nodes.index(f'{bus_ac_name}.{itg}')
        g_idx = bus_idx - grid.N_nodes_v
        grid.dae['g'][2*g_idx+0] += i_abc_list[2*(itg-1)  ]
        grid.dae['g'][2*g_idx+1] += i_abc_list[2*(itg-1)+1]

    ### DC side
    a,b,c = sym.symbols(f'a_{bus_ac_name},b_{bus_ac_name},c_{bus_ac_name}',real=True)
    i_rms = sym.sqrt(i_a_r**2+i_a_i**2+0.1) 
    p_vsc_loss = a + b*i_rms + c*i_rms*i_rms

    v_dc_a_r,v_dc_n_r = sym.symbols(f'v_{bus_dc_name}_a_r,v_{bus_dc_name}_n_r',real=True)

    i_dc_a_r,i_dc_n_r = sym.symbols(f'i_vsc_{bus_dc_name}_a_r,i_vsc_{bus_dc_name}_n_r',real=True)

    eq_i_dc_a_r = i_dc_a_r + p_dc/(v_dc_a_r-v_dc_n_r+1e-8)
    eq_i_dc_n_r = i_dc_n_r + p_dc/(v_dc_n_r-v_dc_a_r+1e-8)
    eq_p_dc = p_dc - p_ac + sym.Piecewise((p_vsc_loss, p_dc < 0), (-p_vsc_loss, p_dc > 0),(p_vsc_loss, True))
   
    grid.dae['g'] +=  [eq_i_dc_a_r,eq_i_dc_n_r,eq_p_dc]
    grid.dae['y'] +=  [   i_dc_a_r,   i_dc_n_r,   p_dc]  
    grid.dae['u'].update({f'p_{bus_ac_name}':0.0,f'q_{bus_ac_name}':0.0}) 
    grid.dae['xy_0_dict'].update({f'v_{bus_dc_name}_a_r':800.0,f'v_{bus_dc_name}_n_r':10.0})
    
    grid.dae['u'].pop(f'i_{bus_dc_name}_a_r')
    grid.dae['u'].pop(f'i_{bus_dc_name}_n_r')
    grid.dae['params'].update({f'a_{bus_ac_name}':a_value,f'b_{bus_ac_name}':b_value,f'c_{bus_ac_name}':c_value})
    grid.dae['params'].update({f'coef_a_{bus_ac_name}':1/3,f'coef_b_{bus_ac_name}':1/3,f'coef_c_{bus_ac_name}':1/3})
    

    bus_idx = grid.nodes.index(f'{bus_dc_name}.{1}')
    g_idx = bus_idx - grid.N_nodes_v
    grid.dae['g'][2*g_idx+0] += i_dc_a_r
    grid.dae['g'][2*g_idx+1] += 0.0   

    bus_idx = grid.nodes.index(f'{bus_dc_name}.{4}')
    g_idx = bus_idx - grid.N_nodes_v
    grid.dae['g'][2*g_idx+0] += i_dc_n_r
    grid.dae['g'][2*g_idx+1] += 0.0   

