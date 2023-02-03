from pydae.build_v2 import builder
import numpy as np
import sympy as sym
from sympy.matrices.sparsetools import _doktocsr
from sympy import SparseMatrix

def smib_dae():
    r"""
    Synchronous machine model of order 4 from Federico Milano book.

    """

    sin = sym.sin
    cos = sym.cos  

    # inputs
    V = sym.Symbol(f"V", real=True)
    theta = sym.Symbol(f"theta", real=True)
    p_m = sym.Symbol(f"p_m", real=True)
    v_f = sym.Symbol(f"v_f", real=True)  
        
    # dynamic states
    delta = sym.Symbol(f"delta", real=True)
    omega = sym.Symbol(f"omega", real=True)
    e1q = sym.Symbol(f"e1q", real=True)
    e1d = sym.Symbol(f"e1d", real=True)

    # algebraic states
    i_d = sym.Symbol(f"i_d", real=True)
    i_q = sym.Symbol(f"i_q", real=True)            
    p_g = sym.Symbol(f"p_g", real=True)
    q_g = sym.Symbol(f"q_g", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n", real=True)
    Omega_b = sym.Symbol(f"Omega_b", real=True)            
    H = sym.Symbol(f"H", real=True)
    T1d0 = sym.Symbol(f"T1d0", real=True)
    T1q0 = sym.Symbol(f"T1q0", real=True)
    X_d = sym.Symbol(f"X_d", real=True)
    X_q = sym.Symbol(f"X_q", real=True)
    X1d = sym.Symbol(f"X1d", real=True)
    X1q = sym.Symbol(f"X1q", real=True)
    D = sym.Symbol(f"D", real=True)
    R_a = sym.Symbol(f"R_a", real=True)

    params_dict = {"S_n":100e6,
                   "X_d":1.8,"X1d":0.3, "T1d0":8.0,    
                   "X_q":1.7,"X1q":0.55,"T1q0":0.4,  
                   "R_a":0.01,"X_l": 0.2, 
                   "H":5.0,"D":1.0,
                   "Omega_b":2*np.pi*50}
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     

    aux_dict = {"v_d":V*sin(delta - theta),
                "v_q":V*cos(delta - theta),
                "p_e":i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)}

    # dynamic equations            
    ddelta = Omega_b*(omega - 1.0) 
    domega = 1/(2*H)*(p_m - p_e - D*(omega - 1.0))
    de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
    de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)

    # algebraic equations   
    g_i_d  = v_q + R_a*i_q + X1d*i_d - e1q
    g_i_q  = v_d + R_a*i_d - X1q*i_q - e1d
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    
    # dae 
    f_list = [ddelta,domega,de1q,de1d]
    x_list = [ delta, omega, e1q, e1d]
    g_list = [g_i_d,g_i_q,g_p_g,g_q_g]
    y_ini_list = [  i_d,  i_q,  p_m,  q_g]
    y_run_list = [  i_d,  i_q,  p_g,  q_g]
   
    u_ini_dict = {}
    u_ini_dict.update({'v_f':1.5})
    u_ini_dict.update({'p_g':0.5})
    u_ini_dict.update({'V':1.0})
    u_ini_dict.update({'theta':0.0})

    u_run_dict = {}
    u_run_dict.update({'v_f':1.5})
    u_run_dict.update({'p_m':0.5})
    u_run_dict.update({'V':1.0})
    u_run_dict.update({'theta':0.0})

    xy_0_dict = {}
    xy_0_dict.update({'omega':1.0})
    xy_0_dict.update({'e1q':1.0})
    xy_0_dict.update({'i_q':0.5})

    # outputs
    h_dict = {}
    h_dict.update({f"p_e":p_e})
    h_dict.update({f"v_f":v_f})
    h_dict.update({f"p_m":p_m})
    
    sys_dict = {'name':'dae1','uz_jacs':True,
                'params_dict':params_dict,
                'f_list':f_list,
                'g_list':g_list,
                'x_list':x_list,
                'y_ini_list':y_ini_list,
                'y_run_list':y_run_list,
                'u_ini_dict':u_ini_dict,
                'u_run_dict':u_run_dict,
                'h_dict':h_dict,
                'aux_dict':aux_dict}

    return sys_dict

def smib_build():

    sys_dict = smib_dae()
    b = builder(sys_dict)
    b.dict2system()
    b.functions()
    b.jacobians()
    b.cwrite()
    b.compile()
    b.template()

def expected_builder():

    sys_dict = smib_dae()

    f = sym.Matrix(sys_dict['f_list'])
    g = sym.Matrix(sys_dict['g_list'])
    h = sym.Matrix([item for item in sys_dict['h_dict']])

    x = sym.Matrix(sys_dict['x_list'])
    y_ini = sym.Matrix(sys_dict['y_ini_list'])
    u_ini = sym.Matrix([item for item in sys_dict['u_ini_dict']])
    y_run = sym.Matrix(sys_dict['y_run_list'])
    u_run = sym.Matrix([item for item in sys_dict['u_run_dict']])

    F_x = f.jacobian(x)
    F_y_ini = f.jacobian(y_ini)
    F_u_ini = f.jacobian(u_ini)
    F_y_run = f.jacobian(y_run)
    F_u_run = f.jacobian(u_run)

    G_x = g.jacobian(x)
    G_y_ini = g.jacobian(y_ini)
    G_u_ini = g.jacobian(u_ini)
    G_y_run = g.jacobian(y_run)
    G_u_run = g.jacobian(u_run)

    H_x = h.jacobian(x)
    H_y = h.jacobian(y_run)
    H_u = h.jacobian(u_run)

    jac_ini = sym.Matrix([[F_x,F_y_ini],[G_x,G_y_ini]])
    jac_run = sym.Matrix([[F_x,F_y_run],[G_x,G_y_run]])

    N_x = F_x.shape[0]
    eye = sym.eye(N_x, real=True)
    Dt = sym.Symbol('Dt',real=True)
    jac_trap = sym.Matrix([[eye - 0.5*Dt*F_x, -0.5*Dt*F_y_run],[G_x,G_y_run]])

    tab = 3*' '
    string = ''
    string += 'import numpy as np\n\n'   
    string += 'def eval_dae(model):\n\n'
    string += f'{tab} sin = np.sin\n'  
    string += f'{tab} cos = np.cos\n'  
    string += f'{tab} Dt = 0.001\n'  

    for item in sys_dict['x_list']:
        string += f'{tab} {item} = model.get_value("{item}")\n'
    string += f'{tab} x = np.array({str(list(x))})\n\n'
    string += f"\n"

    for item in sys_dict['y_ini_list']:
        string += f'{tab} {item} = model.get_value("{item}")\n'
    string += f'{tab} y = np.array({str(list(y_ini))})\n\n'
    string += f"\n"

    for item in sys_dict['u_ini_dict']:
        string += f"{tab} {item} = {sys_dict['u_ini_dict'][item]}\n"
    string += f"\n"

    for item in sys_dict['params_dict']:
        string += f"{tab} {item} = {sys_dict['params_dict'][item]}\n"
    string += f"\n"

    for item in sys_dict['aux_dict']:
        string += f"{tab} {item} = {sys_dict['aux_dict'][item]}\n"
    string += f"\n"


    string += f"{tab} f = {str(f).replace('Matrix','np.array')}\n\n"
    string += f"{tab} g = {str(g).replace('Matrix','np.array')}\n\n"
    string += f"{tab} h = {str(h).replace('Matrix','np.array')}\n\n"

    string += f"{tab} F_x = {str(F_x).replace('Matrix','np.array')}\n\n"
    string += f"{tab} F_y_ini = {str(F_y_ini).replace('Matrix','np.array')}\n\n"
    string += f"{tab} F_u_ini = {str(F_u_ini).replace('Matrix','np.array')}\n\n"
    string += f"{tab} G_x = {str(G_x).replace('Matrix','np.array')}\n\n"
    string += f"{tab} G_y_ini = {str(G_y_ini).replace('Matrix','np.array')}\n\n"
    string += f"{tab} G_u_ini = {str(G_u_ini).replace('Matrix','np.array')}\n\n"
    string += f"{tab} H_x = {str(H_x).replace('Matrix','np.array')}\n\n"
    string += f"{tab} H_y = {str(H_y).replace('Matrix','np.array')}\n\n"
    string += f"{tab} H_u = {str(H_u).replace('Matrix','np.array')}\n\n"
    string += f"{tab} jac_ini = {str(jac_ini).replace('Matrix','np.array')}\n\n"
    string += f"{tab} jac_run = {str(jac_run).replace('Matrix','np.array')}\n\n"
    string += f"{tab} jac_trap = {str(jac_trap).replace('Matrix','np.array')}\n\n"

    string += "    expected_dict = {'f':f,'g':g,'h':h}\n"
    string += "    expected_dict.update({'F_x':F_x,'F_y_ini':F_y_ini,'F_u_ini':F_u_ini,'G_x':G_x})\n"
    string += "    expected_dict.update({'G_y':G_y_ini,'G_u_ini':G_u_ini,'H_x':H_x,'H_y':H_y,'H_u':H_u})\n"
    string += "    expected_dict.update({'jac_ini':jac_ini,'jac_run':jac_run,'jac_trap':jac_trap})\n\n"

    string += "    return expected_dict\n"

    with open('smib_expected.py','w') as fobj:
        fobj.write(string)

def test_jacobians():

    expected_builder()
    tol = 1e-12

    from smib_expected import eval_dae
    import dae1
    model = dae1.model()
    model.ini({},1)

    expected_dict =eval_dae(model) 

    # jac_ini
    from dae1 import de_jac_ini_eval,sp_jac_ini_eval
    N_x = model.N_x
    N_y = model.N_y
    N_xy = N_x+N_y
    Dt = 0.001

    ## de_jac_ini
    de_jac_ini = np.zeros((N_xy,N_xy))
    de_jac_ini_eval(de_jac_ini,model.x,model.y_ini,model.u_ini,model.p,Dt)

    assert np.max(np.abs(de_jac_ini-expected_dict['jac_ini'])) < tol

    ## sp_jac_ini
    from dae1 import sp_jac_ini_vectors
    sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja = sp_jac_ini_vectors()

    from scipy.sparse import csr_array
    jac_ini_data = np.zeros(len(sp_jac_ini_ia))
    sp_jac_ini = csr_array((np.zeros(len(jac_ini_data)),sp_jac_ini_ia,sp_jac_ini_ja),shape=(sp_jac_ini_nia,sp_jac_ini_nja))
    sp_jac_ini_eval(sp_jac_ini.data,model.x,model.y_ini,model.u_ini,model.p,Dt)

    assert np.max(np.abs(sp_jac_ini-expected_dict['jac_ini'])) < tol

    # jac_run
    from dae1 import de_jac_run_eval,sp_jac_run_eval
    N_x = model.N_x
    N_y = model.N_y
    N_xy = N_x+N_y
    Dt = 0.001

    ## de_jac_run
    de_jac_run = np.zeros((N_xy,N_xy))
    de_jac_run_eval(de_jac_run,model.x,model.y_run,model.u_run,model.p,Dt)

    assert np.max(np.abs(de_jac_run-expected_dict['jac_run'])) < tol

    ## sp_jac_run
    from dae1 import sp_jac_run_vectors
    sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja = sp_jac_run_vectors()

    from scipy.sparse import csr_array
    jac_run_data = np.zeros(len(sp_jac_run_ia))
    sp_jac_run = csr_array((np.zeros(len(jac_run_data)),sp_jac_run_ia,sp_jac_run_ja),shape=(sp_jac_run_nia,sp_jac_run_nja))
    sp_jac_run_eval(sp_jac_run.data,model.x,model.y_run,model.u_run,model.p,Dt)

    assert np.max(np.abs(sp_jac_run-expected_dict['jac_run'])) < tol


    # jac_trap
    from dae1 import de_jac_trap_eval,sp_jac_trap_eval
    N_x = model.N_x
    N_y = model.N_y
    N_xy = N_x+N_y
    Dt = 0.001

    ## de_jac_trap
    de_jac_trap = np.zeros((N_xy,N_xy))

    de_jac_trap_eval(de_jac_trap,model.x,model.y_run,model.u_run,model.p,Dt)
    assert np.max(np.abs(de_jac_trap-expected_dict['jac_trap'])) < tol

    ## sp_jac_trap
    from dae1 import sp_jac_trap_vectors
    sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja = sp_jac_trap_vectors()

    from scipy.sparse import csr_array
    jac_trap_data = np.zeros(len(sp_jac_trap_ia))
    sp_jac_trap = csr_array((np.zeros(len(jac_trap_data)),sp_jac_trap_ia,sp_jac_trap_ja),shape=(sp_jac_trap_nia,sp_jac_trap_nja))
    sp_jac_trap_eval(sp_jac_trap.data,model.x,model.y_run,model.u_run,model.p,Dt)
    assert np.max(np.abs(sp_jac_trap-expected_dict['jac_trap'])) < tol

smib_build()
test_jacobians()

                # 'f_list':f_list,
                # 'g_list':g_list,
                # 'x_list':x_list,
                # 'y_ini_list':y_list,
                # 'y_run_list':y_list,
                # 'u_run_dict':u_dict,
                # 'u_ini_dict':u_dict,
                # 'h_dict':h_dict
# y = sym.Matrix([[y_1],   
#                 [y_2],   
#                 [y_3],   
#                 [y_4]])

# u = sym.Matrix([[u_1],   
#                 [u_2]])


# f = sym.Matrix([[-p_f1*x_1 + p_f4*y_1 + p_u1*u_1 + x_3*y_1], 
#                 [-p_f2*x_3], 
#                 [-p_f3*x_2 + u_2*y_1]])
                
# g = sym.Matrix([[p_g1*y_2 + p_u2*u_1], 
#                 [p_g2*y_1 + x_2*y_1], 
#                 [p_g3*y_3 + x_3*y_2], 
#                 [p_u3*u_2 + u_1*y_4 + x_2*y_2]])

# h = sym.Matrix([[p_h1*x_1 + p_h2*y_1 + p_h3*y_4 + p_h4*u_1], 
#                 [u_2*x_1 + x_1*y_2 + x_2*y_1]])

# z = h

# F_x = f.jacobian(x)
# F_y = f.jacobian(y)
# F_u = f.jacobian(u)

# G_x = g.jacobian(x)
# G_y = g.jacobian(y)
# G_u = g.jacobian(u)

# H_x = h.jacobian(x)
# H_y = h.jacobian(y)
# H_u = h.jacobian(u)

# jac_ini_sym = sym.Matrix([[F_x,F_y],[G_x,G_y]]) 
# #print(jac_ini_sym)
# jac_ini_data,jac_ini_indices,jac_ini_indptr,jac_ini_shape = _doktocsr(SparseMatrix(jac_ini_sym))

# jac_ini_sym = sym.Matrix([[F_x,F_y],[G_x,G_y]]) 
# #print(jac_ini_sym)
# jac_ini_data,jac_ini_indices,jac_ini_indptr,jac_ini_shape = _doktocsr(SparseMatrix(jac_ini_sym))

# N_x = F_x.shape[0]
# eye = sym.eye(N_x, real=True)
# Dt = sym.Symbol('Dt',real=True)
# jac_trap = sym.Matrix([[eye - 0.5*Dt*F_x, -0.5*Dt*F_y],[G_x,G_y]])    
# print('jac_trap = ',jac_trap)
# jac_trap_data,jac_trap_indices,jac_trap_indptr,jac_trap_shape = _doktocsr(SparseMatrix(jac_trap_sym))

# print('F_x = ',F_x)
# print('F_y = ',F_y)
# print('F_u = ',F_u)

# print('G_x = ',G_x)
# print('G_y = ',G_y)
# print('G_u = ',G_u)

# print('H_x = ',H_x)
# print('H_y = ',H_y)
# print('H_u = ',H_u)


# p_f1,p_f2,p_f3,p_f4= 1.0,2.0,3.0,4.0
# p_g1,p_g2,p_g3,p_g4= 1.0,2.0,3.0,4.0
# p_u1,p_u2,p_u3,p_u4= 1.0,2.0,3.0,4.0
# p_h1,p_h2,p_h3,p_h4= 1.0,2.0,3.0,4.0

# params_dict = {
# 'p_f1':p_f1,'p_f2':p_f2,'p_f3':p_f3,'p_f4':p_f4,
# 'p_g1':p_g1,'p_g2':p_g2,'p_g3':p_g3,'p_g4':p_g4,
# 'p_u1':p_u1,'p_u2':p_u2,'p_u3':p_u3,'p_u4':p_u4,
# 'p_h1':p_h1,'p_h2':p_h2,'p_h3':p_h3,'p_h4':p_h4,
# }

# u_1,u_2 = 1.0,2.0

# u_dict = {
# 'u_1':1.0,'u_2':2.0
# }

# h_dict ={
# 'z_1':z[0],'z_2':z[1]
# } 



# import dae1

# model = dae1.model()
# model.ini({},1)
# #model.report_x()
# #model.report_y()
# #model.report_u()
# #model.report_z()
# #model.report_params()
# #model.jac_ini

# x_1,x_2,x_3 = model.get_mvalue(['x_1','x_2','x_3'])
# y_1,y_2,y_3,y_4 = model.get_mvalue(['y_1','y_2','y_3','y_4'])

# x = np.array([[x_1],   
#               [x_2],   
#               [x_3]]).reshape(3,)

# y = np.array([[y_1],   
#               [y_2],   
#               [y_3],   
#               [y_4]]).reshape(4,)

# u = np.array([[u_1],   
#               [u_2]]).reshape(2,)


# p = np.array([p_f1,p_f2,p_f3,p_f4,
#               p_g1,p_g2,p_g3,p_g4,
#               p_u1,p_u2,p_u3,p_u4,
#               p_h1,p_h2,p_h3,p_h4]).reshape(16,)

# Dt = 1e-3

# F_x =  np.array([[-p_f1, 0, y_1], [0, 0, -p_f2], [0, -p_f3, 0]])
# F_y =  np.array([[p_f4 + x_3, 0, 0, 0], [0, 0, 0, 0], [u_2, 0, 0, 0]])
# F_u =  np.array([[p_u1, 0], [0, 0], [0, y_1]])
# G_x =  np.array([[0, 0, 0], [0, y_1, 0], [0, 0, y_2], [0, y_2, 0]])
# G_y =  np.array([[0, p_g1, 0, 0], [p_g2 + x_2, 0, 0, 0], [0, x_3, p_g3, 0], [0, x_2, 0, u_1]])
# G_u =  np.array([[p_u2, 0], [0, 0], [0, 0], [y_4, p_u3]])

# jac_ini = np.array([[-p_f1, 0, y_1, p_f4 + x_3, 0, 0, 0], 
#                     [0, 0, -p_f2, 0, 0, 0, 0], 
#                     [0, -p_f3, 0, u_2, 0, 0, 0], 
#                     [0, 0, 0, 0, p_g1, 0, 0], 
#                     [0, y_1, 0, p_g2 + x_2, 0, 0, 0], 
#                     [0, 0, y_2, 0, x_3, p_g3, 0], 
#                     [0, y_2, 0, 0, x_2, 0, u_1]])


# assert (np.max(np.abs(jac_ini-model.jac_ini))) < 1e-16

# from dae1 import de_jac_ini_eval,sp_jac_ini_eval

# N_x = F_x.shape[0]
# N_y = G_y.shape[0]
# N_xy = N_x+N_y
# de_jac_ini = np.zeros((N_xy,N_xy))
# de_jac_ini_eval(de_jac_ini,x,y,u,p,Dt)
# #print(np.max(np.abs(jac_ini-de_jac_ini)))

# from scipy.sparse import csr_array
# from scipy.sparse import csr_matrix

# sp_jac_ini = csr_matrix((np.zeros(len(jac_ini_data)),jac_ini_indices,jac_ini_indptr),shape=jac_ini_shape)
# sp_jac_ini_eval(sp_jac_ini.data,x,y,u,p,Dt)

# assert (np.max(np.abs(jac_ini-sp_jac_ini))) < 1e-16
# #print(de_jac_ini)
# #print(sp_jac_ini.toarray())
# #print(sp_jac_ini.indices)
# #print(sp_jac_ini.indptr)
# #print(sp_jac_ini.nnz)
# #
# #print(jac_ini_indices,jac_ini_indptr)

# jac_trap =  np.array([[0.5*Dt*p_f1 + 1, 0, -0.5*Dt*y_1, -0.5*Dt*(p_f4 + x_3), 0, 0, 0], 
#                       [0, 1, 0.5*Dt*p_f2, 0, 0, 0, 0], 
#                       [0, 0.5*Dt*p_f3, 1, -0.5*Dt*u_2, 0, 0, 0], 
#                       [0, 0, 0, 0, p_g1, 0, 0], 
#                       [0, y_1, 0, p_g2 + x_2, 0, 0, 0], 
#                       [0, 0, y_2, 0, x_3, p_g3, 0], 
#                       [0, y_2, 0, 0, x_2, 0, u_1]])