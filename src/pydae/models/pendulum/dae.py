import numpy as np
import sympy as sym


def dae(model_name):
    L,G,M,K_d = sym.symbols('L,G,M,K_d', real=True)
    p_x,p_y,v_x,v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True) 
    lam,f_x,theta,u_dummy = sym.symbols('lam,f_x,theta,u_dummy', real=True) 

    dp_x = v_x
    dp_y = v_y
    dv_x = (-2*p_x*lam + f_x - K_d*v_x)/M
    dv_y = (-M*G - 2*p_y*lam - K_d*v_y)/M   

    g_1 = p_x**2 + p_y**2 - L**2 -lam*1e-6
    g_2 = -theta + sym.atan2(p_x,-p_y) + u_dummy

    params_dict = {'L':5.21,'G':9.81,'M':10.0,'K_d':1e-3}  # parameters with default values

    u_ini_dict = {'theta':np.deg2rad(5.0),'u_dummy':0.0}  # input for the initialization problem
    u_run_dict = {'f_x':0,'u_dummy':0.0}                  # input for the running problem, its value is updated 

    sys_dict = {'name':model_name,
                'params_dict':params_dict,
                'f_list':[dp_x,dp_y,dv_x,dv_y],
                'g_list':[g_1,g_2],
                'x_list':[ p_x, p_y, v_x, v_y],
                'y_ini_list':[lam,f_x],
                'y_run_list':[lam,theta],
                'u_ini_dict':u_ini_dict,
                'u_run_dict':u_run_dict,
                'h_dict':{'E_p':M*G*(p_y+L),'E_k':0.5*M*(v_x**2+v_y**2),'f_x':f_x,'lam':lam}} 

    return sys_dict


def test_build_run():
    import pydae.build_cffi as db
    sys_dict = dae()
    bldr = db.builder(sys_dict)
    bldr.build()

    from pydae import ssa
    import pendulum 

    model = pendulum.model()

    M = 30.0  # mass of the bob (kg)
    L = 5.21  # length of the pendulum (m)
    model.ini({'M':M,'L':L,           # parameters setting
            'theta':np.deg2rad(0)  # initial desired angle = 0º
            },-1)                  # here -1 means that -1 is considered as initial gess for
                                    # dynamic and algebraic states

    model.report_x()  # obtained dynamic states
    model.report_y()  # obtained algebraic states
    model.report_z()  # obtained outputs
    model.report_u()  # obtained algebraic states (theta is both state and output; f_x is both input and output)
    model.report_params()  # considered parameters

if __name__ == '__main__':

    test_build_run()





