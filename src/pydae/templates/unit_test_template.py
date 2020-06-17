if __name__ == "__main__":

    
    params_dict = {'X_d':1.81,'X1d':0.3,'T1d0':8.0,
                   'X_q':1.76,'X1q':0.65,'T1q0':1.0,
                   'R_a':0.003,'X_l': 0.05, 
                   'H':3.5,'D':1.0,
                   'Omega_b':2*np.pi*50,'omega_s':1.0,
                   'v_0':0.9008,'theta_0':0.0}
    
    
    u_ini_dict = {'P_t':0.8, 'Q_t':0.2}  # for the initialization problem
    u_run_dict = {'p_m':0.8,'e1q':1.0}  # for the running problem (here initialization and running problem are the same)
    
    
    x_list = ['delta','omega']    # [inductor current, PI integrator]
    y_ini_list = ['i_d','i_q','v_1','theta_1','p_m','e1q'] # for the initialization problem
    y_run_list = ['i_d','i_q','v_1','theta_1','P_t','Q_t'] # for the running problem (here initialization and running problem are the same)
    
    sys_vars = {'params':params_dict,
                'u_list':u_run_dict,
                'x_list':x_list,
                'y_list':y_run_list}
    
    exec(sym_gen_str())  # exec to generate the required symbolic varables and constants
    


