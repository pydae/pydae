import numpy as np
import sympy as sym


def add_Dyn11(grid,trafo):
    '''
    Transformer model for type Dyn11 connection as defined in:

    Álvaro Rodríguez del Nozal, Esther Romero-Ramos and Ángel Luis Trigo-García, 
    Accurate Assessment of Decoupled OLTC Transformers to Optimize the Operation of Low-Voltage Networks
    Energies 2019, 12, 2173; doi:10.3390/en12112173

    {"bus_j": "MV0",  "bus_k": "I01",  "S_n_kVA": 100, "U_j_kV":20, "U_k_kV":0.4,
     "R_cc_pu": 0.01, "X_cc_pu":0.04, "R_fe_pu": 1000, "X_mu_pu":100, 
     "connection": "Dyn11",   "conductors_j": 3, "conductors_k": 4,
     "monitor":true}
    
    '''

    # G_ta,B_ta,G_ma,B_ma,Ratio_a = sym.symbols('G_ta,B_ta,G_ma,B_ma,Ratio_a', real = True)
    # G_tb,B_tb,G_mb,B_mb,Ratio_b = sym.symbols('G_tb,B_tb,G_mb,B_mb,Ratio_b', real = True)
    # G_tc,B_tc,G_mc,B_mc,Ratio_c = sym.symbols('G_tc,B_tc,G_mc,B_mc,Ratio_c', real = True)

    bus_j_name = trafo['bus_j']
    bus_k_name = trafo['bus_k']
    name = f'{bus_j_name}_{bus_k_name}'
    
    G_t,B_t,G_m,B_m = sym.symbols(f'G_t_{name},B_t_{name},G_m_{name},B_m_{name}', real = True)
    Ratio_a,Ratio_b,Ratio_c = sym.symbols(f'Ratio_a_{name},Ratio_b_{name},Ratio_c_{name}', real = True)
    R_g,Ratio = sym.symbols(f'R_g_{name},Ratio_{name}', real = True)

    I = sym.I

    # Transformer Y primitive
    Y_prim = sym.Matrix([[2*I*B_m + 2*I*B_t + 2*G_m + 2*G_t, -I*B_m - I*B_t - G_m - G_t, -I*B_m - I*B_t - G_m - G_t, -Ratio_a*(I*B_t + G_t), 0, Ratio_c*(I*B_t + G_t), (Ratio_a - Ratio_c)*(I*B_t + G_t)], 
                         [-I*B_m - I*B_t - G_m - G_t, 2*I*B_m + 2*I*B_t + 2*G_m + 2*G_t, -I*B_m - I*B_t - G_m - G_t, Ratio_a*(I*B_t + G_t), -Ratio_b*(I*B_t + G_t), 0, (-Ratio_a + Ratio_b)*(I*B_t + G_t)], 
                         [-I*B_m - I*B_t - G_m - G_t, -I*B_m - I*B_t - G_m - G_t, 2*I*B_m + 2*I*B_t + 2*G_m + 2*G_t, 0, Ratio_b*(I*B_t + G_t), -Ratio_c*(I*B_t + G_t), (-Ratio_b + Ratio_c)*(I*B_t + G_t)], 
                         [-Ratio_a*(I*B_t + G_t), Ratio_a*(I*B_t + G_t), 0, Ratio_a**2*(I*B_t + G_t), 0, 0, Ratio_a**2*(-I*B_t - G_t)], 
                         [0, -Ratio_b*(I*B_t + G_t), Ratio_b*(I*B_t + G_t), 0, Ratio_b**2*(I*B_t + G_t), 0, Ratio_b**2*(-I*B_t - G_t)], 
                         [Ratio_c*(I*B_t + G_t), 0, -Ratio_c*(I*B_t + G_t), 0, 0, Ratio_c**2*(I*B_t + G_t), Ratio_c**2*(-I*B_t - G_t)], 
                         [(Ratio_a - Ratio_c)*(I*B_t + G_t), (-Ratio_a + Ratio_b)*(I*B_t + G_t), (-Ratio_b + Ratio_c)*(I*B_t + G_t), Ratio_a**2*(-I*B_t - G_t), Ratio_b**2*(-I*B_t - G_t), Ratio_c**2*(-I*B_t - G_t), (R_g*(I*B_t + G_t)*(Ratio_a**2 + Ratio_b**2 + Ratio_c**2) + 1)/R_g]])
    
    Y_prim = Y_prim.subs(Ratio_a,Ratio_a*Ratio)
    Y_prim = Y_prim.subs(Ratio_b,Ratio_b*Ratio)
    Y_prim = Y_prim.subs(Ratio_c,Ratio_c*Ratio)

    # Conductance and subseptance primitive
    G_primitive = sym.re(Y_prim)
    B_primitive = sym.im(Y_prim)

    # default nodes
    nodes_j = [0,1,2]
    nodes_k = [0,1,2,3]  

    # from trafo primitive to system global primitive
    rl = grid.it_branch
    rh = grid.it_branch + trafo['N_branches']
    grid.G_primitive[rl:rh,rl:rh] = G_primitive   
    grid.B_primitive[rl:rh,rl:rh] = B_primitive

    # j side 
    for item in trafo['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
        node_j = f"{trafo['bus_j']}.{item}"
        col = grid.nodes_list.index(node_j)
        row = grid.it_branch

        grid.A[row,col] = 1
        grid.it_branch +=1  

    # k side
    for item in  trafo['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
        node_k = f"{trafo['bus_k']}.{item}"
        col = grid.nodes_list.index(node_k)
        row = grid.it_branch
        grid.A[row,col] = 1
        grid.it_branch +=1  

    S_n = trafo['S_n_kVA']*1000.0
    U_jn = trafo['U_j_kV']*1000.0
    U_kn = trafo['U_k_kV']*1000.0
    Z_cc_pu = trafo['R_cc_pu'] +1j*trafo['X_cc_pu']

    Z_b = U_jn**2/S_n
    Z_cc = Z_cc_pu*Z_b

    Y_cc = 1/Z_cc
    G_t_N = Y_cc.real
    B_t_N = Y_cc.imag

    G_m_N = 0.0
    B_m_N = 0.0

    if 'R_fe_pu' in trafo:
        G_m_N = 1/(trafo['R_fe_pu']*Z_b) 
    if 'X_mu_pu' in trafo:
        B_m_N = 1/(trafo['X_mu_pu']*Z_b) 

    Ratio_N = U_jn/U_kn*np.sqrt(3)
    
    grid.dae['params_dict'].update({str(G_t):G_t_N,str(B_t):B_t_N,str(G_m):G_m_N,str(B_m):B_m_N})
    grid.dae['params_dict'].update({str(R_g):3.0,str(Ratio):Ratio_N})

    grid.dae['u_ini_dict'].update({str(Ratio_a):1,str(Ratio_b):1,str(Ratio_c):1})
    grid.dae['u_run_dict'].update({str(Ratio_a):1,str(Ratio_b):1,str(Ratio_c):1})

    # i_t_1_0_m = sym.Abs(i_t_MV0_I01_1_0_r + I*i_t_MV0_I01_1_0_i)
    # i_t_1_1_m = sym.Abs(i_t_MV0_I01_1_1_r + I*i_t_MV0_I01_1_1_i)
    # i_t_1_2_m = sym.Abs(i_t_MV0_I01_1_2_r + I*i_t_MV0_I01_1_2_i)
    # i_t_2_0_m = sym.Abs(i_t_MV0_I01_2_0_r + I*i_t_MV0_I01_2_0_i)
    # i_t_2_1_m = sym.Abs(i_t_MV0_I01_2_1_r + I*i_t_MV0_I01_2_1_i)
    # i_t_2_2_m = sym.Abs(i_t_MV0_I01_2_2_r + I*i_t_MV0_I01_2_1_r)
    # i_t_2_1_m = sym.Abs(i_t_MV0_I01_2_1_i + I*i_t_MV0_I01_2_2_r)
    # i_t_2_1_m = sym.Abs(i_t_MV0_I01_2_1_i + I*i_t_MV0_I01_2_2_r)
    # i_t_2_2_m = sym.Abs(i_t_MV0_I01_2_2_i + I*i_t_MV0_I01_2_2_r)
    # i_t_2_2_m = sym.Abs(i_t_MV0_I01_2_2_i + I*i_t_MV0_I01_2_2_i)
    # i_t_2_3_m = sym.Abs(i_t_MV0_I01_2_3_r + I*i_t_MV0_I01_2_3_i)

    return Y_prim,nodes_j,nodes_k




def dev_Dyn11():
    '''
    I1 = U1_to_I1 * U1
    U1 = N*UA
    IA = N.T * I1 + UnRG

    IA = Yp*UA

    IA =  (N.T * U1_to_I1 * N)*UA + UnRG 

    '''

    G_ta,B_ta,G_ma,B_ma,Ratio_a = sym.symbols('G_ta,B_ta,G_ma,B_ma,Ratio_a', real = True)
    G_tb,B_tb,G_mb,B_mb,Ratio_b = sym.symbols('G_tb,B_tb,G_mb,B_mb,Ratio_b', real = True)
    G_tc,B_tc,G_mc,B_mc,Ratio_c = sym.symbols('G_tc,B_tc,G_mc,B_mc,Ratio_c', real = True)

    G_ta,B_ta,G_ma,B_ma,Ratio_a = sym.symbols('G_t,B_t,G_m,B_m,Ratio_a', real = True)
    G_tb,B_tb,G_mb,B_mb,Ratio_b = sym.symbols('G_t,B_t,G_m,B_m,Ratio_b', real = True)
    G_tc,B_tc,G_mc,B_mc,Ratio_c = sym.symbols('G_t,B_t,G_m,B_m,Ratio_c', real = True)

    R_g = sym.symbols('R_g', real = True)


    Y_ta = G_ta + sym.I*B_ta
    Y_ma = G_ma + sym.I*B_ma
    Y_tb = G_tb + sym.I*B_tb
    Y_mb = G_mb + sym.I*B_mb   
    Y_tc = G_tc + sym.I*B_tc
    Y_mc = G_mc + sym.I*B_mc   


    U1_to_I1 = sym.Matrix([
        [  Y_ta + Y_ma,     -Ratio_a*Y_ta,            0,                 0,            0,                0 ],
        [-Ratio_a*Y_ta,   Ratio_a**2*Y_ta,            0,                 0,            0,                0 ],
        [            0,                 0,  Y_tb + Y_mb,     -Ratio_b*Y_tb,            0,                0 ],
        [            0,                 0,-Ratio_b*Y_tb,   Ratio_b**2*Y_tb,            0,                0 ],
        [            0,                 0,            0,                 0,  Y_tc + Y_mc,     -Ratio_c*Y_tc],
        [            0,                 0,            0,                 0,-Ratio_c*Y_tc,   Ratio_c**2*Y_tc]
    ])

    N = sym.Matrix([
        [ 1, -1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0,  0, -1],
        [ 0,  1, -1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0, -1],
        [-1,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  1, -1]
    ])

    Y_prim = N.T @ U1_to_I1 @ N
    Y_prim[-1,-1] += 1/R_g
    Y_prim = sym.simplify(Y_prim)

    sym.pprint(Y_prim)    


def test_Dyn11_build():
    from pydae.urisi.urisi_builder import urisi


    grid = urisi('Dyn11.json')
    grid.uz_jacs = False
    grid.build('cigre_eu_lv_ind')

def test_Dyn11_ini():
    from pydae.urisi.utils import report_v
    import cigre_eu_lv_ind
    model = cigre_eu_lv_ind.model()
    model.ini({'Ratio_a_MV0_I01':1.0,'Ratio_b_MV0_I01':1.0,'Ratio_c_MV0_I01':1.0,
               'R_g_MV0_I01':1000},'xy_0.json')
    model.report_y()
    report_v(model,'Dyn11.json')
    model.report_z()


if __name__ == "__main__":

    test_Dyn11_build()
    test_Dyn11_ini()

    #print(dev_Dyn11())