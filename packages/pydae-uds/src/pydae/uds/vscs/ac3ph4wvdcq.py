import numpy as np
import sympy as sym


def ac3ph4wvdcq(grid,vsc_data):
    '''
    Converter type v_dc,q_ac 3 phase 4 wire
    
    '''
    
    bus_ac_name = vsc_data['bus_ac']
    bus_dc_name = vsc_data['bus_dc']  
    to_bus_dc_name = vsc_data['to_bus_dc']  
    
    a_value  = vsc_data['a']   
    b_value  = vsc_data['b']   
    c_value  = vsc_data['c']   
    
   
    ### AC-side
    q_a,q_b,q_c = sym.symbols(f'q_vsc_a_{bus_name}_a,q_vsc_b_{bus_name}_b,q_vsc_c_{bus_name}_c')

    p_ac,q_ac,p_dc,p_loss = sym.symbols(f'p_vsc_{bus_ac_name},q_vsc_{bus_ac_name},p_vsc_{bus_dc_name},p_vsc_loss_{bus_ac_name}',real=True)
    p_a_d,p_b_d,p_c_d,p_n_d = sym.symbols(f'p_a_d_{bus_ac_name},p_b_d_{bus_ac_name},p_c_d_{bus_ac_name},p_n_d_{bus_ac_name}',real=True)
    C_a,C_b,C_c = sym.symbols(f'C_a_{bus_ac_name},C_b_{bus_ac_name},C_c_{bus_ac_name}',real=True)
   
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
    
    # DC line current for computing DC power
    i_l_a_r,i_l_n_r = sym.symbols(f'i_l_{bus_dc_name}_{to_bus_dc_name}_a_r,i_l_{bus_dc_name}_{to_bus_dc_name}_n_r', real = True)   
    v_dc_a_r,v_dc_n_r  = sym.symbols(f'v_{bus_dc_name}_a_r,v_{bus_dc_name}_n_r', real = True) 
    v_dc_ref  = sym.Symbol(f'v_dc_{bus_dc_name}', real = True) 

    a,b,c,K_dc = sym.symbols(f'a_{bus_ac_name},b_{bus_ac_name},c_{bus_ac_name},K_dc_{bus_dc_name}',real=True)
    
    i_a_rms = sym.sqrt(i_a_r**2+i_a_i**2 + 0.01) 
    i_b_rms = sym.sqrt(i_b_r**2+i_b_i**2+ 0.01) 
    i_c_rms = sym.sqrt(i_c_r**2+i_c_i**2+ 0.01) 
    i_n_rms = sym.sqrt(i_n_r**2+i_n_i**2+ 0.01) 


    p_loss_a_ = a + b*i_a_rms + c*i_a_rms*i_a_rms
    p_loss_b_ = a + b*i_b_rms + c*i_b_rms*i_b_rms
    p_loss_c_ = a + b*i_c_rms + c*i_c_rms*i_c_rms
    p_loss_n_ = a + b*i_n_rms + c*i_n_rms*i_n_rms
      
    

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
    s_n = (v_n)* sym.conjugate(i_n)
    
    
    eq_p_a_d =  C_a*p_dc - p_a_d 
    eq_p_b_d =  C_b*p_dc - p_b_d
    eq_p_c_d =  C_c*p_dc - p_c_d
    eq_p_n_d =  sym.re(s_n) - p_n_d
    
    #p_loss_a = sym.Piecewise((-p_loss_a_, p_a_d < 0), (p_loss_a_, p_a_d > 0),(p_loss_a_, True))
    #p_loss_b = sym.Piecewise((-p_loss_b_, p_b_d < 0), (p_loss_b_, p_b_d > 0),(p_loss_b_, True))
    #p_loss_c = sym.Piecewise((-p_loss_c_, p_c_d < 0), (p_loss_c_, p_c_d > 0),(p_loss_c_, True))
    #p_loss_n = sym.Piecewise((-p_loss_n_, p_n_d < 0), (p_loss_n_, p_n_d > 0),(p_loss_n_, True))

    eq_i_a_r =  sym.re(s_a) - p_a_d + p_loss_a_ + p_loss_n_
    eq_i_b_r =  sym.re(s_b) - p_b_d + p_loss_b_
    eq_i_c_r =  sym.re(s_c) - p_c_d + p_loss_c_
    eq_i_a_i =  sym.im(s_a) - q_c
    eq_i_b_i =  sym.im(s_b) - q_b
    eq_i_c_i =  sym.im(s_c) - q_c
    
    eq_i_n_r = -i_n_r + i_a_r + i_b_r + i_c_r
    eq_i_n_i = -i_n_i + i_a_i + i_b_i + i_c_i
   
    eq_p_dc = p_dc + i_l_a_r * v_dc_a_r + i_l_n_r * v_dc_n_r    
    eq_v_dc_a_r = -v_dc_a_r + v_dc_ref + v_dc_n_r - K_dc * p_dc


    grid.dae['g'] += [eq_p_a_d,
                      eq_p_b_d,
                      eq_p_c_d,
                      eq_p_n_d,
                      eq_i_a_r,eq_i_a_i,
                      eq_i_b_r,eq_i_b_i,
                      eq_i_c_r,eq_i_c_i,
                      eq_i_n_r,eq_i_n_i,
                      eq_p_dc,
                      eq_v_dc_a_r
                      ]
    
    grid.dae['y'] += [p_a_d,
                      p_b_d,
                      p_c_d,
                      p_n_d,
                      i_a_r,   i_a_i,
                      i_b_r,   i_b_i,
                      i_c_r,   i_c_i,
                      i_n_r,   i_n_i,
                      p_dc,
                      v_dc_a_r]
    
    i_abc_list  = [i_a_r,i_a_i,i_b_r,i_b_i,i_c_r,i_c_i]
    for itg in [1,2,3]:
        bus_idx = grid.nodes.index(f'{bus_ac_name}.{itg}')
        g_idx = bus_idx - grid.N_nodes_v
        grid.dae['g'][2*g_idx+0] += i_abc_list[2*(itg-1)  ]
        grid.dae['g'][2*g_idx+1] += i_abc_list[2*(itg-1)+1]

    bus_idx = grid.nodes.index(f'{bus_dc_name}.{4}')
    g_idx = bus_idx - grid.N_nodes_v
    grid.dae['g'][2*g_idx+0] += -i_l_a_r
    grid.dae['g'][2*g_idx+1] +=        0   

    
      
    grid.dae['u'].update({f'v_dc_{bus_dc_name}':800.0,f'{str(q_a)}':0.0,f'{str(q_b)}':0.0,f'{str(q_c)}':0.0}) 
    grid.dae['u'].pop(str(v_dc_a_r))
    grid.dae['params'].update({f'a_{bus_ac_name}':a_value,f'b_{bus_ac_name}':b_value,f'c_{bus_ac_name}':c_value})
    grid.dae['params'].update({f'C_a_{bus_ac_name}':1/3,f'C_b_{bus_ac_name}':1/3,f'C_c_{bus_ac_name}':1/3})
    grid.dae['params'].update({f'K_dc_{bus_dc_name}':1e-6})

    grid.dae['xy_0_dict'].update({f'v_{bus_dc_name}_a_r':800.0,f'v_{bus_dc_name}_n_r':1.0})
    
    grid.dae['h_dict'].update({f'p_vsc_{bus_ac_name}':sym.re(s_a)+sym.re(s_b)+sym.re(s_c)+sym.re(s_n)})
    grid.dae['h_dict'].update({f'p_vsc_loss_{bus_ac_name}':(p_loss_a_+p_loss_b_+p_loss_c_+p_loss_n_)})
    
