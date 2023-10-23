import sympy as sym
import numpy as np

def acdc_3ph_4w_vdc_q(grid,vsc_data):
    '''
    Converter type v_dc,q_ac 3 phase 4 wire
    
    '''
    
    bus_ac_name = vsc_data['bus_ac']
    bus_dc_name = vsc_data['bus_dc']  
    
    A_value  = vsc_data['A']   
    B_value  = vsc_data['B']   
    C_value  = vsc_data['C']   
    
    ### AC-side
    q_a,q_b,q_c = sym.symbols(f'q_vsc_a_{bus_ac_name},q_vsc_b_{bus_ac_name},q_vsc_c_{bus_ac_name}',real=True)

    p_ac,q_ac,p_dc,p_loss = sym.symbols(f'p_vsc_{bus_ac_name},q_vsc_{bus_ac_name},p_vsc_{bus_dc_name},p_vsc_loss_{bus_ac_name}',real=True)
    p_a_d,p_b_d,p_c_d,p_n_d = sym.symbols(f'p_a_d_{bus_ac_name},p_b_d_{bus_ac_name},p_c_d_{bus_ac_name},p_n_d_{bus_ac_name}',real=True)
    C_a,C_b,C_c = sym.symbols(f'C_a_{bus_ac_name},C_b_{bus_ac_name},C_c_{bus_ac_name}',real=True)
   
    #### AC voltages:
    v_a_r,v_b_r,v_c_r,v_n_r = sym.symbols(f'V_{bus_ac_name}_0_r,V_{bus_ac_name}_1_r,V_{bus_ac_name}_2_r,V_{bus_ac_name}_3_r', real=True)
    v_a_i,v_b_i,v_c_i,v_n_i = sym.symbols(f'V_{bus_ac_name}_0_i,V_{bus_ac_name}_1_i,V_{bus_ac_name}_2_i,V_{bus_ac_name}_3_i', real=True)
    
    #### AC currents:
    i_a_r,i_a_i = sym.symbols(f'i_vsc_{bus_ac_name}_a_r,i_vsc_{bus_ac_name}_a_i',real=True)
    i_b_r,i_b_i = sym.symbols(f'i_vsc_{bus_ac_name}_b_r,i_vsc_{bus_ac_name}_b_i',real=True)
    i_c_r,i_c_i = sym.symbols(f'i_vsc_{bus_ac_name}_c_r,i_vsc_{bus_ac_name}_c_i',real=True)
    i_n_r,i_n_i = sym.symbols(f'i_vsc_{bus_ac_name}_n_r,i_vsc_{bus_ac_name}_n_i',real=True)
    
    #### DC voltages:
    v_pos,v_neg = sym.symbols(f'V_{bus_dc_name}_0_r,V_{bus_dc_name}_1_r', real=True)
    v_posi,v_negi = sym.symbols(f'V_{bus_dc_name}_0_i,V_{bus_dc_name}_1_i', real=True)

    # algebraic states dc side
    i_pos,i_neg,v_og = sym.symbols(f'i_vsc_pos_{bus_dc_name}_sp,i_vsc_{bus_dc_name}_sn, v_og_{bus_dc_name}', real=True)

    v_dc_ref  = sym.Symbol(f'v_dc_{bus_dc_name}_ref', real = True) 

    v_dc = v_dc_ref


    A_loss,B_loss,C_loss = sym.symbols(f'A_{bus_ac_name},B_{bus_ac_name},C_{bus_ac_name}',real=True)
    R_dc,K_dc,R_gdc = sym.symbols(f'R_dc_{bus_dc_name},K_dc_{bus_dc_name},R_gdc_{bus_dc_name}',real=True)
     

    i_a_rms = sym.sqrt(i_a_r**2+i_a_i**2 + 0.001) 
    i_b_rms = sym.sqrt(i_b_r**2+i_b_i**2 + 0.001) 
    i_c_rms = sym.sqrt(i_c_r**2+i_c_i**2 + 0.001) 
    i_n_rms = sym.sqrt(i_n_r**2+i_n_i**2 + 0.001) 

    p_loss_a = A_loss*i_a_rms*i_a_rms + B_loss*i_a_rms + C_loss
    p_loss_b = A_loss*i_b_rms*i_b_rms + B_loss*i_b_rms + C_loss
    p_loss_c = A_loss*i_c_rms*i_c_rms + B_loss*i_c_rms + C_loss
    p_loss_n = A_loss*i_n_rms*i_n_rms + B_loss*i_n_rms + C_loss

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
    
    eq_p_a_d =  C_a*p_dc - p_a_d 
    eq_p_b_d =  C_b*p_dc - p_b_d
    eq_p_c_d =  C_c*p_dc - p_c_d
    eq_p_n_d =  sym.re(s_n) - p_n_d
    
    eq_i_a_r =  sym.re(s_a) - p_a_d + p_loss_a + p_loss_n/3
    eq_i_b_r =  sym.re(s_b) - p_b_d + p_loss_b + p_loss_n/3
    eq_i_c_r =  sym.re(s_c) - p_c_d + p_loss_c + p_loss_n/3
    eq_i_a_i =  sym.im(s_a) - q_a
    eq_i_b_i =  sym.im(s_b) - q_b
    eq_i_c_i =  sym.im(s_c) - q_c
    
    eq_i_n_r = i_n_r + i_a_r + i_b_r + i_c_r
    eq_i_n_i = i_n_i + i_a_i + i_b_i + i_c_i
   
    
    eq_p_dc = -p_dc - (v_dc/2*i_pos   - v_dc/2*i_neg)


    v_tp = v_dc/2.0
    v_tn = v_dc/2.0

    eq_i_pos = v_og + v_tp - R_dc*i_pos - v_pos
    eq_i_neg = v_og - v_tn - R_dc*i_neg - v_neg
    eq_v_og = -v_og/R_gdc - i_pos - i_neg 


    grid.dae['g'] += [eq_p_a_d,
                      eq_p_b_d,
                      eq_p_c_d,
                      eq_p_n_d,
                      eq_i_a_r,eq_i_a_i,
                      eq_i_b_r,eq_i_b_i,
                      eq_i_c_r,eq_i_c_i,
                      eq_i_n_r,eq_i_n_i,
                      eq_i_pos,eq_i_neg,eq_v_og,
                      eq_p_dc
                      ]
    
    grid.dae['y_ini'] += [p_a_d,
                      p_b_d,
                      p_c_d,
                      p_n_d,
                      i_a_r,   i_a_i,
                      i_b_r,   i_b_i,
                      i_c_r,   i_c_i,
                      i_n_r,   i_n_i,
                      i_pos,i_neg,v_og,
                      p_dc]

    grid.dae['y_run'] += [p_a_d,
                      p_b_d,
                      p_c_d,
                      p_n_d,
                      i_a_r,   i_a_i,
                      i_b_r,   i_b_i,
                      i_c_r,   i_c_i,
                      i_n_r,   i_n_i,
                      i_pos,i_neg,v_og,
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

    # current injections dc side
    idx_r,idx_i = grid.node2idx(f'{bus_dc_name}','a')
    grid.dae['g'] [idx_r] += -i_pos    
    grid.dae['g'] [idx_i] += v_posi/1e3

    idx_r,idx_i = grid.node2idx(f'{bus_dc_name}','b')
    grid.dae['g'] [idx_r] += -i_neg  
    grid.dae['g'] [idx_i] += v_negi/1e3  

    
      
    grid.dae['u_ini_dict'].update({f'v_dc_{bus_dc_name}_ref':800.0}) 
    grid.dae['u_ini_dict'].update({f'{str(q_a)}':0.0,f'{str(q_b)}':0.0,f'{str(q_c)}':0.0}) 

    grid.dae['u_run_dict'].update({f'v_dc_{bus_dc_name}_ref':800.0}) 
    grid.dae['u_run_dict'].update({f'{str(q_a)}':0.0,f'{str(q_b)}':0.0,f'{str(q_c)}':0.0}) 


    #grid.dae['u'].pop(str(v_dc_a_r))
    grid.dae['params_dict'].update({f'A_{bus_ac_name}':A_value,f'B_{bus_ac_name}':B_value,f'C_{bus_ac_name}':C_value})
    grid.dae['params_dict'].update({f'C_a_{bus_ac_name}':1/3,f'C_b_{bus_ac_name}':1/3,f'C_c_{bus_ac_name}':1/3})
    grid.dae['params_dict'].update({f'R_dc_{bus_dc_name}':1e-6})
    grid.dae['params_dict'].update({f'K_dc_{bus_dc_name}':1e-6})
    grid.dae['params_dict'].update({f'R_gdc_{bus_dc_name}':3.0})    

    grid.dae['xy_0_dict'].update({f'v_{bus_dc_name}_a_r':800.0,f'v_{bus_dc_name}_n_r':1.0})
    
    grid.dae['h_dict'].update({f'p_vsc_{bus_ac_name}':sym.re(s_a)+sym.re(s_b)+sym.re(s_c)+sym.re(s_n)})
    grid.dae['h_dict'].update({f'p_vsc_loss_{bus_ac_name}':(p_loss_a+p_loss_b+p_loss_c+p_loss_n)})

def test():
    import numpy as np
    import sympy as sym
    import json
    from pydae.urisi.urisi_builder import urisi
    import pydae.build_cffi as db

    grid = urisi('acdc_3ph_4w_vdc_q.hjson')
    grid.uz_jacs = True
    grid.construct('temp')
    grid.compile('temp')

    import temp

    S_n = 100e3
    V_n = 400
    I_n = S_n/V_n
    Conduction_losses = 0.02*S_n # = A*I_n**2
    lossses = 0.0
    A = Conduction_losses/(I_n**2)/3*lossses
    B = 1/3*lossses
    C = 0.02*S_n/3*lossses
    model = temp.model()
    model.ini({'A_A1':A,'B_A1':B,'C_A1':C},'xy_0.json')
    model.report_y()
    model.report_z()
    #model.save_xy_0('xy_1.json')
    print(800**2/10)



if __name__ == '__main__':

    #development()
    test()
