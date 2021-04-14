import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 


class k12p6_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 29
        self.N_y = 51 
        self.N_z = 11 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_1_5', 'b_1_5', 'bs_1_5', 'g_2_6', 'b_2_6', 'bs_2_6', 'g_3_11', 'b_3_11', 'bs_3_11', 'g_4_10', 'b_4_10', 'bs_4_10', 'g_5_6', 'b_5_6', 'bs_5_6', 'g_6_7', 'b_6_7', 'bs_6_7', 'g_7_8', 'b_7_8', 'bs_7_8', 'g_8_9', 'b_8_9', 'bs_8_9', 'g_9_10', 'b_9_10', 'bs_9_10', 'g_10_11', 'b_10_11', 'bs_10_11', 'U_1_n', 'U_2_n', 'U_3_n', 'U_4_n', 'U_5_n', 'U_6_n', 'U_7_n', 'U_8_n', 'U_9_n', 'U_10_n', 'U_11_n', 'S_n_1', 'H_1', 'Omega_b_1', 'T1d0_1', 'T1q0_1', 'X_d_1', 'X_q_1', 'X1d_1', 'X1q_1', 'D_1', 'R_a_1', 'K_delta_1', 'K_a_1', 'K_ai_1', 'T_r_1', 'Droop_1', 'T_m_1', 'S_n_2', 'H_2', 'Omega_b_2', 'T1d0_2', 'T1q0_2', 'X_d_2', 'X_q_2', 'X1d_2', 'X1q_2', 'D_2', 'R_a_2', 'K_delta_2', 'K_a_2', 'K_ai_2', 'T_r_2', 'Droop_2', 'T_m_2', 'S_n_3', 'H_3', 'Omega_b_3', 'T1d0_3', 'T1q0_3', 'X_d_3', 'X_q_3', 'X1d_3', 'X1q_3', 'D_3', 'R_a_3', 'K_delta_3', 'K_a_3', 'K_ai_3', 'T_r_3', 'Droop_3', 'T_m_3', 'S_n_4', 'H_4', 'Omega_b_4', 'T1d0_4', 'T1q0_4', 'X_d_4', 'X_q_4', 'X1d_4', 'X1q_4', 'D_4', 'R_a_4', 'K_delta_4', 'K_a_4', 'K_ai_4', 'T_r_4', 'Droop_4', 'T_m_4', 'K_sec_1', 'K_sec_2', 'K_sec_3', 'K_sec_4'] 
        self.params_values_list  = [100000000.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 3.96039603960396, -39.603960396039604, 0.0437499994125, 9.900990099009901, -99.00990099009901, 0.017499999765000003, 0.9000900090008999, -9.000900090009, 0.192499997415, 0.9000900090008999, -9.000900090009, 0.192499997415, 9.900990099009901, -99.00990099009901, 0.017499999765000003, 3.96039603960396, -39.603960396039604, 0.0437499994125, 20000.0, 20000.0, 20000.0, 20000.0, 230000.0, 230000.0, 230000.0, 230000.0, 230000.0, 230000.0, 230000.0, 900000000.0, 6.5, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 0.0, 0.0025, 0.01, 100, 1e-06, 0.1, 0.05, 5.0, 900000000.0, 6.5, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 0.0, 0.0025, 0.01, 100, 1e-06, 0.1, 0.05, 5.0, 900000000.0, 6.175, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 0.0, 0.0025, 0.01, 100, 1e-06, 0.1, 0.05, 5.0, 900000000.0, 6.175, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 0.0, 0.0025, 0.01, 100, 1e-06, 0.1, 0.05, 5.0, 0.001, 0.001, 0.001, 0.001] 
        self.inputs_ini_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'P_4', 'Q_4', 'P_5', 'Q_5', 'P_6', 'Q_6', 'P_7', 'Q_7', 'P_8', 'Q_8', 'P_9', 'Q_9', 'P_10', 'Q_10', 'P_11', 'Q_11', 'v_ref_1', 'v_pss_1', 'p_c_1', 'v_ref_2', 'v_pss_2', 'p_c_2', 'v_ref_3', 'v_pss_3', 'p_c_3', 'v_ref_4', 'v_pss_4', 'p_c_4'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7] 
        self.inputs_run_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'P_4', 'Q_4', 'P_5', 'Q_5', 'P_6', 'Q_6', 'P_7', 'Q_7', 'P_8', 'Q_8', 'P_9', 'Q_9', 'P_10', 'Q_10', 'P_11', 'Q_11', 'v_ref_1', 'v_pss_1', 'p_c_1', 'v_ref_2', 'v_pss_2', 'p_c_2', 'v_ref_3', 'v_pss_3', 'p_c_3', 'v_ref_4', 'v_pss_4', 'p_c_4'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7] 
        self.outputs_list = ['V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9', 'V_10', 'V_11'] 
        self.x_list = ['delta_1', 'omega_1', 'e1q_1', 'e1d_1', 'v_c_1', 'xi_v_1', 'p_m_1', 'delta_2', 'omega_2', 'e1q_2', 'e1d_2', 'v_c_2', 'xi_v_2', 'p_m_2', 'delta_3', 'omega_3', 'e1q_3', 'e1d_3', 'v_c_3', 'xi_v_3', 'p_m_3', 'delta_4', 'omega_4', 'e1q_4', 'e1d_4', 'v_c_4', 'xi_v_4', 'p_m_4', 'xi_freq'] 
        self.y_run_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'V_4', 'theta_4', 'V_5', 'theta_5', 'V_6', 'theta_6', 'V_7', 'theta_7', 'V_8', 'theta_8', 'V_9', 'theta_9', 'V_10', 'theta_10', 'V_11', 'theta_11', 'i_d_1', 'i_q_1', 'p_g_1_1', 'q_g_1_1', 'v_f_1', 'p_m_ref_1', 'i_d_2', 'i_q_2', 'p_g_2_1', 'q_g_2_1', 'v_f_2', 'p_m_ref_2', 'i_d_3', 'i_q_3', 'p_g_3_1', 'q_g_3_1', 'v_f_3', 'p_m_ref_3', 'i_d_4', 'i_q_4', 'p_g_4_1', 'q_g_4_1', 'v_f_4', 'p_m_ref_4', 'p_r_1', 'p_r_2', 'p_r_3', 'p_r_4', 'omega_coi'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'V_4', 'theta_4', 'V_5', 'theta_5', 'V_6', 'theta_6', 'V_7', 'theta_7', 'V_8', 'theta_8', 'V_9', 'theta_9', 'V_10', 'theta_10', 'V_11', 'theta_11', 'i_d_1', 'i_q_1', 'p_g_1_1', 'q_g_1_1', 'v_f_1', 'p_m_ref_1', 'i_d_2', 'i_q_2', 'p_g_2_1', 'q_g_2_1', 'v_f_2', 'p_m_ref_2', 'i_d_3', 'i_q_3', 'p_g_3_1', 'q_g_3_1', 'v_f_3', 'p_m_ref_3', 'i_d_4', 'i_q_4', 'p_g_4_1', 'q_g_4_1', 'v_f_4', 'p_m_ref_4', 'p_r_1', 'p_r_2', 'p_r_3', 'p_r_4', 'omega_coi'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.t = 0.0
        self.it = 0
        self.it_store = 0
        self.xy_prev = np.zeros((self.N_x+self.N_y,1))
        self.initialization_tol = 1e-6
        self.N_u = len(self.inputs_run_list) 
        self.sopt_root_method='hybr'
        self.sopt_root_jac=True
        self.u_ini_list = self.inputs_ini_list
        self.u_ini_values_list = self.inputs_ini_values_list
        self.u_run_list = self.inputs_run_list
        self.u_run_values_list = self.inputs_run_values_list
        self.N_u = len(self.u_run_list)
        Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols = nonzeros()

        self.Fx_ini_rows = np.array(Fx_ini_rows) 
        if len(Fx_ini_rows) == 1: 
            self.Fx_ini_rows = np.array([[Fx_ini_rows]]).reshape(1,) 
            self.Fx_ini_cols = np.array([[Fx_ini_cols]]).reshape(1,)  
            
        self.Fx_ini_cols = np.array(Fx_ini_cols)
        self.Fy_ini_rows = np.array(Fy_ini_rows)        
        self.Fy_ini_cols = np.array(Fy_ini_cols)
        self.Gx_ini_rows = np.array(Gx_ini_rows)        
        self.Gx_ini_cols = np.array(Gx_ini_cols)
        self.Gy_ini_rows = np.array(Gy_ini_rows)        
        self.Gy_ini_cols = np.array(Gy_ini_cols)
        
        
        self.yini2urun = list(set(self.inputs_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.inputs_ini_list)))

        self.update() 
                
    def update(self): 

        self.N_steps = int(np.ceil(self.t_end/self.Dt)) 
        dt = [  
              ('t_end', np.float64),
              ('Dt', np.float64),
              ('decimation', np.float64),
              ('itol', np.float64),
              ('Dt_max', np.float64),
              ('Dt_min', np.float64),
              ('solvern', np.int64),
              ('imax', np.int64),
              ('N_steps', np.int64),
              ('N_store', np.int64),
              ('N_x', np.int64),
              ('N_y', np.int64),
              ('N_z', np.int64),
              ('t', np.float64),
              ('it', np.int64),
              ('it_store', np.int64),
              ('idx', np.int64),
              ('idy', np.int64),
              ('f', np.float64, (self.N_x,1)),
              ('x', np.float64, (self.N_x,1)),
              ('x_0', np.float64, (self.N_x,1)),
              ('g', np.float64, (self.N_y,1)),
              ('y_run', np.float64, (self.N_y,1)),
              ('y_ini', np.float64, (self.N_y,1)),
              ('u_run', np.float64, (self.N_u,1)),
              ('y_0', np.float64, (self.N_y,1)),
              ('h', np.float64, (self.N_z,1)),
              ('Fx', np.float64, (self.N_x,self.N_x)),
              ('Fy', np.float64, (self.N_x,self.N_y)),
              ('Gx', np.float64, (self.N_y,self.N_x)),
              ('Gy', np.float64, (self.N_y,self.N_y)),
              ('Fu', np.float64, (self.N_x,self.N_u)),
              ('Gu', np.float64, (self.N_y,self.N_u)),
              ('Hx', np.float64, (self.N_z,self.N_x)),
              ('Hy', np.float64, (self.N_z,self.N_y)),
              ('Hu', np.float64, (self.N_z,self.N_u)),
              ('Fx_ini', np.float64, (self.N_x,self.N_x)),
              ('Fy_ini', np.float64, (self.N_x,self.N_y)),
              ('Gx_ini', np.float64, (self.N_y,self.N_x)),
              ('Gy_ini', np.float64, (self.N_y,self.N_y)),
              ('T', np.float64, (self.N_store+1,1)),
              ('X', np.float64, (self.N_store+1,self.N_x)),
              ('Y', np.float64, (self.N_store+1,self.N_y)),
              ('Z', np.float64, (self.N_store+1,self.N_z)),
              ('iters', np.float64, (self.N_store+1,1)),
              ('store', np.int64),
              ('Fx_ini_rows', np.int64, self.Fx_ini_rows.shape),
              ('Fx_ini_cols', np.int64, self.Fx_ini_cols.shape),
              ('Fy_ini_rows', np.int64, self.Fy_ini_rows.shape),
              ('Fy_ini_cols', np.int64, self.Fy_ini_cols.shape),
              ('Gx_ini_rows', np.int64, self.Gx_ini_rows.shape),
              ('Gx_ini_cols', np.int64, self.Gx_ini_cols.shape),
              ('Gy_ini_rows', np.int64, self.Gy_ini_rows.shape),
              ('Gy_ini_cols', np.int64, self.Gy_ini_cols.shape),
              ('Ac_ini', np.float64, ((self.N_x+self.N_y,self.N_x+self.N_y))),   
              ('fg', np.float64, ((self.N_x+self.N_y,1))),  
             ]



        
        
        values = [
                self.t_end,                          
                self.Dt,
                self.decimation,
                self.itol,
                self.Dt_max,
                self.Dt_min,
                self.solvern,
                self.imax,
                self.N_steps,
                self.N_store,
                self.N_x,
                self.N_y,
                self.N_z,
                self.t,
                self.it,
                self.it_store,
                0,                                     # idx
                0,                                     # idy
                np.zeros((self.N_x,1)),                # f
                np.zeros((self.N_x,1)),                # x
                np.zeros((self.N_x,1)),                # x_0
                np.zeros((self.N_y,1)),                # g
                np.zeros((self.N_y,1)),                # y_run
                np.zeros((self.N_y,1)),                # y_ini
                np.zeros((self.N_u,1)),                # u_run
                np.zeros((self.N_y,1)),                # y_0
                np.zeros((self.N_z,1)),                # h
                np.zeros((self.N_x,self.N_x)),         # Fx   
                np.zeros((self.N_x,self.N_y)),         # Fy 
                np.zeros((self.N_y,self.N_x)),         # Gx 
                np.zeros((self.N_y,self.N_y)),         # Fy
                np.zeros((self.N_x,self.N_u)),         # Fu 
                np.zeros((self.N_y,self.N_u)),         # Gu 
                np.zeros((self.N_z,self.N_x)),         # Hx 
                np.zeros((self.N_z,self.N_y)),         # Hy 
                np.zeros((self.N_z,self.N_u)),         # Hu 
                np.zeros((self.N_x,self.N_x)),         # Fx_ini  
                np.zeros((self.N_x,self.N_y)),         # Fy_ini 
                np.zeros((self.N_y,self.N_x)),         # Gx_ini 
                np.zeros((self.N_y,self.N_y)),         # Fy_ini 
                np.zeros((self.N_store+1,1)),          # T
                np.zeros((self.N_store+1,self.N_x)),   # X
                np.zeros((self.N_store+1,self.N_y)),   # Y
                np.zeros((self.N_store+1,self.N_z)),   # Z
                np.zeros((self.N_store+1,1)),          # iters
                1,
                self.Fx_ini_rows,       
                self.Fx_ini_cols,
                self.Fy_ini_rows,       
                self.Fy_ini_cols,
                self.Gx_ini_rows,        
                self.Gx_ini_cols,
                self.Gy_ini_rows,       
                self.Gy_ini_cols,
                np.zeros((self.N_x+self.N_y,self.N_x+self.N_y)),  
                np.zeros((self.N_x+self.N_y,1)),
                ]  

        dt += [(item,np.float64) for item in self.params_list]
        values += [item for item in self.params_values_list]

        for item_id,item_val in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            if item_id in self.inputs_run_list: continue
            dt += [(item_id,np.float64)]
            values += [item_val]

        dt += [(item,np.float64) for item in self.inputs_run_list]
        values += [item for item in self.inputs_run_values_list]

        self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))
        
        xy0 = np.zeros((self.N_x+self.N_y,))
        self.ini_dae_jacobian_nn(xy0)
        self.run_dae_jacobian_nn(xy0)
        


    def load_params(self,data_input):

        if type(data_input) == str:
            json_file = data_input
            self.json_file = json_file
            self.json_data = open(json_file).read().replace("'",'"')
            data = json.loads(self.json_data)
        elif type(data_input) == dict:
            data = data_input

        self.data = data
        for item in self.data:
            self.struct[0][item] = self.data[item]
            self.params_values_list[self.params_list.index(item)] = self.data[item]



    def ini_problem(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        if self.compile:
            ini(self.struct,2)
            ini(self.struct,3)       
        else:
            ini.py_func(self.struct,2)
            ini.py_func(self.struct,3)                   
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg

    def run_problem(self,x):
        t = self.struct[0].t
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        
        if self.compile:
            run(t,self.struct,2)
            run(t,self.struct,3)
            run(t,self.struct,10)
            run(t,self.struct,11)
            run(t,self.struct,12)
            run(t,self.struct,13)
        else:
            run.py_func(t,self.struct,2)
            run.py_func(t,self.struct,3)
            run.py_func(t,self.struct,10)
            run.py_func(t,self.struct,11)
            run.py_func(t,self.struct,12)
            run.py_func(t,self.struct,13)            
        
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg
    

    def run_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(0.0,self.struct,10)
        run(0.0,self.struct,11)     
        run(0.0,self.struct,12)
        run(0.0,self.struct,13)
        A_c = np.block([[self.struct[0].Fx,self.struct[0].Fy],
                        [self.struct[0].Gx,self.struct[0].Gy]])
        return A_c

    def run_dae_jacobian_nn(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run_nn(0.0,self.struct,10)
        run_nn(0.0,self.struct,11)     
        run_nn(0.0,self.struct,12)
        run_nn(0.0,self.struct,13)
 

    
    def eval_jacobians(self):

        run(0.0,self.struct,10)
        run(0.0,self.struct,11)  
        run(0.0,self.struct,12) 

        return 1


    def ini_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        if self.compile:
            ini(self.struct,10)
            ini(self.struct,11) 
        else:
            ini.py_func(self.struct,10)
            ini.py_func(self.struct,11)             
        A_c = np.block([[self.struct[0].Fx_ini,self.struct[0].Fy_ini],
                        [self.struct[0].Gx_ini,self.struct[0].Gy_ini]])
        return A_c

    def ini_dae_jacobian_nn(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini_nn(self.struct,10)
        ini_nn(self.struct,11)       
 

    def f_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_odeint(self,x,t):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_ivp(self,t,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def Fx_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,10)
        return self.struct[0].Fx

    def eval_A(self):
        
        Fx = self.struct[0].Fx
        Fy = self.struct[0].Fy
        Gx = self.struct[0].Gx
        Gy = self.struct[0].Gy
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        self.A = A
        
        return A

    def eval_A_ini(self):
        
        Fx = self.struct[0].Fx_ini
        Fy = self.struct[0].Fy_ini
        Gx = self.struct[0].Gx_ini
        Gy = self.struct[0].Gy_ini
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        
        return A
    
    def reset(self):
        for param,param_value in zip(self.params_list,self.params_values_list):
            self.struct[0][param] = param_value
        for input_name,input_value in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            self.struct[0][input_name] = input_value   
        for input_name,input_value in zip(self.inputs_run_list,self.inputs_run_values_list):
            self.struct[0][input_name] = input_value  

    def simulate(self,events,xy0=0):
        
        # initialize both the ini and the run system
        self.initialize(events,xy0=xy0)
        
        # simulation run
        for event in events:  
            # make all the desired changes
            self.run([event]) 
            
        # post process
        T,X,Y,Z = self.post()
        
        return T,X,Y,Z
    

    
    def run(self,events):
        

        # simulation run
        for event in events:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        return 1
 
    def rtrun(self,events):
        

        # simulation run
        for event in events:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            self.struct[0].it_store = self.struct[0].N_store-1
            daesolver(self.struct)    # run until next event
            
            
        return 1
    
    def post(self):
        
        # post process result    
        T = self.struct[0]['T'][:self.struct[0].it_store]
        X = self.struct[0]['X'][:self.struct[0].it_store,:]
        Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
        Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
        iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
    
        self.T = T
        self.X = X
        self.Y = Y
        self.Z = Z
        self.iters = iters
        
        return T,X,Y,Z
        
    def save_0(self,file_name = 'xy_0.json'):
        xy_0_dict = {}
        for item in self.x_list:
            xy_0_dict.update({item:self.get_value(item)})
        for item in self.y_ini_list:
            xy_0_dict.update({item:self.get_value(item)})
    
        xy_0_str = json.dumps(xy_0_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(xy_0_str)

    def load_0(self,file_name = 'xy_0.json'):
        with open(file_name) as fobj:
            xy_0_str = fobj.read()
        xy_0_dict = json.loads(xy_0_str)
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_prev[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_prev[self.y_ini_list.index(item)+self.N_x] = xy_0_dict[item]
                
            
    def initialize(self,events=[{}],xy0=0,compile=True):
        '''
        

        Parameters
        ----------
        events : dictionary 
            Dictionary with at least 't_end' and all inputs and parameters 
            that need to be changed.
        xy0 : float or string, optional
            0 means all states should be zero as initial guess. 
            If not zero all the states initial guess are the given input.
            If 'prev' it uses the last known initialization result as initial guess.

        Returns
        -------
        T : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.
        Z : TYPE
            DESCRIPTION.

        '''
        
        self.compile = compile
        
        # simulation parameters
        self.struct[0].it = 0       # set time step to zero
        self.struct[0].it_store = 0 # set storage to zero
        self.struct[0].t = 0.0      # set time to zero
                    
        # initialization
        it_event = 0
        event = events[it_event]
        for item in event:
            self.struct[0][item] = event[item]
            
        
        ## compute initial conditions using x and y_ini 
        if type(xy0) == str:
            if xy0 == 'prev':
                xy0 = self.xy_prev
            else:
                self.load_0(xy0)
                xy0 = self.xy_prev
        elif type(xy0) == dict:
            with open('xy_0.json','w') as fobj:
                fobj.write(json.dumps(xy0))
            self.load_0('xy_0.json')
            xy0 = self.xy_prev            
        else:
            if xy0 == 0:
                xy0 = np.zeros(self.N_x+self.N_y)
            elif xy0 == 1:
                xy0 = np.ones(self.N_x+self.N_y)
            else:
                xy0 = xy0*np.ones(self.N_x+self.N_y)

        #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )

        
        if self.sopt_root_jac:
            sol = sopt.root(self.ini_problem, xy0, 
                            jac=self.ini_dae_jacobian, 
                            method=self.sopt_root_method, tol=self.initialization_tol)
        else:
            sol = sopt.root(self.ini_problem, xy0, method=self.sopt_root_method)

        self.initialization_ok = True
        if sol.success == False:
            print('initialization not found!')
            self.initialization_ok = False

            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]

        if self.initialization_ok:
            xy = sol.x
            self.xy_prev = xy
            self.struct[0].x[:,0] = xy[0:self.N_x]
            self.struct[0].y_run[:,0] = xy[self.N_x:]

            ## y_ini to u_run
            for item in self.inputs_run_list:
                if item in self.y_ini_list:
                    self.struct[0][item] = self.struct[0].y_ini[self.y_ini_list.index(item)]

            ## u_ini to y_run
            for item in self.inputs_ini_list:
                if item in self.y_run_list:
                    self.struct[0].y_run[self.y_run_list.index(item)] = self.struct[0][item]


            #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )
            if self.sopt_root_jac:
                sol = sopt.root(self.run_problem, xy0, 
                                jac=self.run_dae_jacobian, 
                                method=self.sopt_root_method, tol=self.initialization_tol)
            else:
                sol = sopt.root(self.run_problem, xy0, method=self.sopt_root_method)

            if self.compile:
                # evaluate f and g
                run(0.0,self.struct,2)
                run(0.0,self.struct,3)                
    
                # evaluate run jacobians 
                run(0.0,self.struct,10)
                run(0.0,self.struct,11)                
                run(0.0,self.struct,12) 
                run(0.0,self.struct,14) 
                
            else:
                # evaluate f and g
                run.py_func(0.0,self.struct,2)
                run.py_func(0.0,self.struct,3)                
    
                # evaluate run jacobians 
                run.py_func(0.0,self.struct,10)
                run.py_func(0.0,self.struct,11)                
                run.py_func(0.0,self.struct,12) 
                run.py_func(0.0,self.struct,14)                 
                
             
            # post process result    
            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
        
            self.T = T
            self.X = X
            self.Y = Y
            self.Z = Z
            self.iters = iters
            
        return self.initialization_ok
    
    
    def get_value(self,name):
        if name in self.inputs_run_list:
            value = self.struct[0][name]
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.struct[0].x[idx,0]
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.struct[0].y_run[idy,0]
        if name in self.params_list:
            value = self.struct[0][name]
        if name in self.outputs_list:
            value = self.struct[0].h[self.outputs_list.index(name),0] 

        return value
    
    def get_values(self,name):
        if name in self.x_list:
            values = self.X[:,self.x_list.index(name)]
        if name in self.y_run_list:
            values = self.Y[:,self.y_run_list.index(name)]
        if name in self.outputs_list:
            values = self.Z[:,self.outputs_list.index(name)]
                        
        return values

    def get_mvalue(self,names):
        '''

        Parameters
        ----------
        names : list
            list of variables names to return each value.

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        mvalue = []
        for name in names:
            mvalue += [self.get_value(name)]
                        
        return mvalue
    
    def set_value(self,name_,value):
        if name_ in self.inputs_run_list:
            self.struct[0][name_] = value
            return
        elif name_ in self.params_list:
            self.struct[0][name_] = value
            return
        elif name_ in self.inputs_ini_list:
            self.struct[0][name_] = value
            return 
        else:
            print(f'Input or parameter {name_} not found.')

    def set_values(self,dictionary):
        
        for item in dictionary:
            self.set_value(item,dictionary[item])
            
            
    def report_x(self,value_format='5.2f', decimals=2):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')

    def report_y(self,value_format='5.2f', decimals=2):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')
            
    def report_u(self,value_format='5.2f', decimals=2):
        for item in self.inputs_run_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')

    def report_z(self,value_format='5.2f', decimals=2):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')

    def report_params(self,value_format='5.2f', decimals=2):
        for item in self.params_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')
            
    def get_x(self):
        return self.struct[0].x
    
    def ss(self):
        
        ssate(self.struct,self.xy_prev.reshape(len(self.xy_prev),1))
        
        ## y_ini to y_run
        self.struct[0].y_run = self.struct[0].y_ini
        
        ## y_ini to u_run
        for item in self.yini2urun:
            self.struct[0][item] = self.struct[0].y_ini[self.y_ini_list.index(item)]
                
        ## u_ini to y_run
        for item in self.uini2yrun:
            self.struct[0].y_run[self.y_run_list.index(item)] = self.struct[0][item]






@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_5 = struct[0].g_1_5
    b_1_5 = struct[0].b_1_5
    bs_1_5 = struct[0].bs_1_5
    g_2_6 = struct[0].g_2_6
    b_2_6 = struct[0].b_2_6
    bs_2_6 = struct[0].bs_2_6
    g_3_11 = struct[0].g_3_11
    b_3_11 = struct[0].b_3_11
    bs_3_11 = struct[0].bs_3_11
    g_4_10 = struct[0].g_4_10
    b_4_10 = struct[0].b_4_10
    bs_4_10 = struct[0].bs_4_10
    g_5_6 = struct[0].g_5_6
    b_5_6 = struct[0].b_5_6
    bs_5_6 = struct[0].bs_5_6
    g_6_7 = struct[0].g_6_7
    b_6_7 = struct[0].b_6_7
    bs_6_7 = struct[0].bs_6_7
    g_7_8 = struct[0].g_7_8
    b_7_8 = struct[0].b_7_8
    bs_7_8 = struct[0].bs_7_8
    g_8_9 = struct[0].g_8_9
    b_8_9 = struct[0].b_8_9
    bs_8_9 = struct[0].bs_8_9
    g_9_10 = struct[0].g_9_10
    b_9_10 = struct[0].b_9_10
    bs_9_10 = struct[0].bs_9_10
    g_10_11 = struct[0].g_10_11
    b_10_11 = struct[0].b_10_11
    bs_10_11 = struct[0].bs_10_11
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    U_9_n = struct[0].U_9_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    S_n_1 = struct[0].S_n_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    T1d0_1 = struct[0].T1d0_1
    T1q0_1 = struct[0].T1q0_1
    X_d_1 = struct[0].X_d_1
    X_q_1 = struct[0].X_q_1
    X1d_1 = struct[0].X1d_1
    X1q_1 = struct[0].X1q_1
    D_1 = struct[0].D_1
    R_a_1 = struct[0].R_a_1
    K_delta_1 = struct[0].K_delta_1
    K_a_1 = struct[0].K_a_1
    K_ai_1 = struct[0].K_ai_1
    T_r_1 = struct[0].T_r_1
    Droop_1 = struct[0].Droop_1
    T_m_1 = struct[0].T_m_1
    S_n_2 = struct[0].S_n_2
    H_2 = struct[0].H_2
    Omega_b_2 = struct[0].Omega_b_2
    T1d0_2 = struct[0].T1d0_2
    T1q0_2 = struct[0].T1q0_2
    X_d_2 = struct[0].X_d_2
    X_q_2 = struct[0].X_q_2
    X1d_2 = struct[0].X1d_2
    X1q_2 = struct[0].X1q_2
    D_2 = struct[0].D_2
    R_a_2 = struct[0].R_a_2
    K_delta_2 = struct[0].K_delta_2
    K_a_2 = struct[0].K_a_2
    K_ai_2 = struct[0].K_ai_2
    T_r_2 = struct[0].T_r_2
    Droop_2 = struct[0].Droop_2
    T_m_2 = struct[0].T_m_2
    S_n_3 = struct[0].S_n_3
    H_3 = struct[0].H_3
    Omega_b_3 = struct[0].Omega_b_3
    T1d0_3 = struct[0].T1d0_3
    T1q0_3 = struct[0].T1q0_3
    X_d_3 = struct[0].X_d_3
    X_q_3 = struct[0].X_q_3
    X1d_3 = struct[0].X1d_3
    X1q_3 = struct[0].X1q_3
    D_3 = struct[0].D_3
    R_a_3 = struct[0].R_a_3
    K_delta_3 = struct[0].K_delta_3
    K_a_3 = struct[0].K_a_3
    K_ai_3 = struct[0].K_ai_3
    T_r_3 = struct[0].T_r_3
    Droop_3 = struct[0].Droop_3
    T_m_3 = struct[0].T_m_3
    S_n_4 = struct[0].S_n_4
    H_4 = struct[0].H_4
    Omega_b_4 = struct[0].Omega_b_4
    T1d0_4 = struct[0].T1d0_4
    T1q0_4 = struct[0].T1q0_4
    X_d_4 = struct[0].X_d_4
    X_q_4 = struct[0].X_q_4
    X1d_4 = struct[0].X1d_4
    X1q_4 = struct[0].X1q_4
    D_4 = struct[0].D_4
    R_a_4 = struct[0].R_a_4
    K_delta_4 = struct[0].K_delta_4
    K_a_4 = struct[0].K_a_4
    K_ai_4 = struct[0].K_ai_4
    T_r_4 = struct[0].T_r_4
    Droop_4 = struct[0].Droop_4
    T_m_4 = struct[0].T_m_4
    K_sec_1 = struct[0].K_sec_1
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    K_sec_4 = struct[0].K_sec_4
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    P_4 = struct[0].P_4
    Q_4 = struct[0].Q_4
    P_5 = struct[0].P_5
    Q_5 = struct[0].Q_5
    P_6 = struct[0].P_6
    Q_6 = struct[0].Q_6
    P_7 = struct[0].P_7
    Q_7 = struct[0].Q_7
    P_8 = struct[0].P_8
    Q_8 = struct[0].Q_8
    P_9 = struct[0].P_9
    Q_9 = struct[0].Q_9
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    v_ref_1 = struct[0].v_ref_1
    v_pss_1 = struct[0].v_pss_1
    p_c_1 = struct[0].p_c_1
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    v_ref_4 = struct[0].v_ref_4
    v_pss_4 = struct[0].v_pss_4
    p_c_4 = struct[0].p_c_4
    
    # Dynamical states:
    delta_1 = struct[0].x[0,0]
    omega_1 = struct[0].x[1,0]
    e1q_1 = struct[0].x[2,0]
    e1d_1 = struct[0].x[3,0]
    v_c_1 = struct[0].x[4,0]
    xi_v_1 = struct[0].x[5,0]
    p_m_1 = struct[0].x[6,0]
    delta_2 = struct[0].x[7,0]
    omega_2 = struct[0].x[8,0]
    e1q_2 = struct[0].x[9,0]
    e1d_2 = struct[0].x[10,0]
    v_c_2 = struct[0].x[11,0]
    xi_v_2 = struct[0].x[12,0]
    p_m_2 = struct[0].x[13,0]
    delta_3 = struct[0].x[14,0]
    omega_3 = struct[0].x[15,0]
    e1q_3 = struct[0].x[16,0]
    e1d_3 = struct[0].x[17,0]
    v_c_3 = struct[0].x[18,0]
    xi_v_3 = struct[0].x[19,0]
    p_m_3 = struct[0].x[20,0]
    delta_4 = struct[0].x[21,0]
    omega_4 = struct[0].x[22,0]
    e1q_4 = struct[0].x[23,0]
    e1d_4 = struct[0].x[24,0]
    v_c_4 = struct[0].x[25,0]
    xi_v_4 = struct[0].x[26,0]
    p_m_4 = struct[0].x[27,0]
    xi_freq = struct[0].x[28,0]
    
    # Algebraic states:
    V_1 = struct[0].y_ini[0,0]
    theta_1 = struct[0].y_ini[1,0]
    V_2 = struct[0].y_ini[2,0]
    theta_2 = struct[0].y_ini[3,0]
    V_3 = struct[0].y_ini[4,0]
    theta_3 = struct[0].y_ini[5,0]
    V_4 = struct[0].y_ini[6,0]
    theta_4 = struct[0].y_ini[7,0]
    V_5 = struct[0].y_ini[8,0]
    theta_5 = struct[0].y_ini[9,0]
    V_6 = struct[0].y_ini[10,0]
    theta_6 = struct[0].y_ini[11,0]
    V_7 = struct[0].y_ini[12,0]
    theta_7 = struct[0].y_ini[13,0]
    V_8 = struct[0].y_ini[14,0]
    theta_8 = struct[0].y_ini[15,0]
    V_9 = struct[0].y_ini[16,0]
    theta_9 = struct[0].y_ini[17,0]
    V_10 = struct[0].y_ini[18,0]
    theta_10 = struct[0].y_ini[19,0]
    V_11 = struct[0].y_ini[20,0]
    theta_11 = struct[0].y_ini[21,0]
    i_d_1 = struct[0].y_ini[22,0]
    i_q_1 = struct[0].y_ini[23,0]
    p_g_1_1 = struct[0].y_ini[24,0]
    q_g_1_1 = struct[0].y_ini[25,0]
    v_f_1 = struct[0].y_ini[26,0]
    p_m_ref_1 = struct[0].y_ini[27,0]
    i_d_2 = struct[0].y_ini[28,0]
    i_q_2 = struct[0].y_ini[29,0]
    p_g_2_1 = struct[0].y_ini[30,0]
    q_g_2_1 = struct[0].y_ini[31,0]
    v_f_2 = struct[0].y_ini[32,0]
    p_m_ref_2 = struct[0].y_ini[33,0]
    i_d_3 = struct[0].y_ini[34,0]
    i_q_3 = struct[0].y_ini[35,0]
    p_g_3_1 = struct[0].y_ini[36,0]
    q_g_3_1 = struct[0].y_ini[37,0]
    v_f_3 = struct[0].y_ini[38,0]
    p_m_ref_3 = struct[0].y_ini[39,0]
    i_d_4 = struct[0].y_ini[40,0]
    i_q_4 = struct[0].y_ini[41,0]
    p_g_4_1 = struct[0].y_ini[42,0]
    q_g_4_1 = struct[0].y_ini[43,0]
    v_f_4 = struct[0].y_ini[44,0]
    p_m_ref_4 = struct[0].y_ini[45,0]
    p_r_1 = struct[0].y_ini[46,0]
    p_r_2 = struct[0].y_ini[47,0]
    p_r_3 = struct[0].y_ini[48,0]
    p_r_4 = struct[0].y_ini[49,0]
    omega_coi = struct[0].y_ini[50,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_1*delta_1 + Omega_b_1*(omega_1 - omega_coi)
        struct[0].f[1,0] = (-D_1*(omega_1 - omega_coi) - i_d_1*(R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_m_1)/(2*H_1)
        struct[0].f[2,0] = (-e1q_1 - i_d_1*(-X1d_1 + X_d_1) + v_f_1)/T1d0_1
        struct[0].f[3,0] = (-e1d_1 + i_q_1*(-X1q_1 + X_q_1))/T1q0_1
        struct[0].f[4,0] = (V_1 - v_c_1)/T_r_1
        struct[0].f[5,0] = -V_1 + v_ref_1
        struct[0].f[6,0] = (-p_m_1 + p_m_ref_1)/T_m_1
        struct[0].f[7,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[8,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[9,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[10,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[11,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[12,0] = -V_2 + v_ref_2
        struct[0].f[13,0] = (-p_m_2 + p_m_ref_2)/T_m_2
        struct[0].f[14,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[15,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[16,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[17,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[18,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[19,0] = -V_3 + v_ref_3
        struct[0].f[20,0] = (-p_m_3 + p_m_ref_3)/T_m_3
        struct[0].f[21,0] = -K_delta_4*delta_4 + Omega_b_4*(omega_4 - omega_coi)
        struct[0].f[22,0] = (-D_4*(omega_4 - omega_coi) - i_d_4*(R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4)) - i_q_4*(R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4)) + p_m_4)/(2*H_4)
        struct[0].f[23,0] = (-e1q_4 - i_d_4*(-X1d_4 + X_d_4) + v_f_4)/T1d0_4
        struct[0].f[24,0] = (-e1d_4 + i_q_4*(-X1q_4 + X_q_4))/T1q0_4
        struct[0].f[25,0] = (V_4 - v_c_4)/T_r_4
        struct[0].f[26,0] = -V_4 + v_ref_4
        struct[0].f[27,0] = (-p_m_4 + p_m_ref_4)/T_m_4
        struct[0].f[28,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_5 + V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_5 - bs_1_5/2) + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_2**2*g_2_6 + V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_2**2*(-b_2_6 - bs_2_6/2) + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + V_3**2*g_3_11 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + V_3**2*(-b_3_11 - bs_3_11/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = -P_4/S_base + V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_4**2*g_4_10 - S_n_4*p_g_4_1/S_base
        struct[0].g[7,0] = -Q_4/S_base + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_4**2*(-b_4_10 - bs_4_10/2) - S_n_4*q_g_4_1/S_base
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5**2*(g_1_5 + g_5_6) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + V_5**2*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6**2*(g_2_6 + g_5_6 + g_6_7) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + V_6**2*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].g[12,0] = -P_7/S_base + V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7**2*(g_6_7 + 2*g_7_8) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].g[13,0] = -Q_7/S_base + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + V_7**2*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].g[14,0] = -P_8/S_base + V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8**2*(2*g_7_8 + 2*g_8_9) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].g[15,0] = -Q_8/S_base + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + V_8**2*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].g[16,0] = -P_9/S_base + V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + V_9**2*(2*g_8_9 + g_9_10)
        struct[0].g[17,0] = -Q_9/S_base + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + V_9**2*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].g[18,0] = -P_10/S_base + V_10**2*(g_10_11 + g_4_10 + g_9_10) + V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].g[19,0] = -Q_10/S_base + V_10**2*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].g[20,0] = -P_11/S_base + V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11**2*(g_10_11 + g_3_11) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].g[21,0] = -Q_11/S_base + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_11**2*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].g[22,0] = R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1) + X1d_1*i_d_1 - e1q_1
        struct[0].g[23,0] = R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1) - X1q_1*i_q_1 - e1d_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = K_a_1*(-v_c_1 + v_pss_1 + v_ref_1) + K_ai_1*xi_v_1 - v_f_1
        struct[0].g[27,0] = p_c_1 - p_m_ref_1 + p_r_1 - (omega_1 - 1)/Droop_1
        struct[0].g[28,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[29,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[30,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[31,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[32,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[33,0] = p_c_2 - p_m_ref_2 + p_r_2 - (omega_2 - 1)/Droop_2
        struct[0].g[34,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[35,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[36,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[37,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[38,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[39,0] = p_c_3 - p_m_ref_3 + p_r_3 - (omega_3 - 1)/Droop_3
        struct[0].g[40,0] = R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4) + X1d_4*i_d_4 - e1q_4
        struct[0].g[41,0] = R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4) - X1q_4*i_q_4 - e1d_4
        struct[0].g[42,0] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4) - p_g_4_1
        struct[0].g[43,0] = V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4) - q_g_4_1
        struct[0].g[44,0] = K_a_4*(-v_c_4 + v_pss_4 + v_ref_4) + K_ai_4*xi_v_4 - v_f_4
        struct[0].g[45,0] = p_c_4 - p_m_ref_4 + p_r_4 - (omega_4 - 1)/Droop_4
        struct[0].g[46,0] = K_sec_1*xi_freq/4 - p_r_1
        struct[0].g[47,0] = K_sec_2*xi_freq/4 - p_r_2
        struct[0].g[48,0] = K_sec_3*xi_freq/4 - p_r_3
        struct[0].g[49,0] = K_sec_4*xi_freq/4 - p_r_4
        struct[0].g[50,0] = omega_1/4 + omega_2/4 + omega_3/4 + omega_4/4 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = V_4
        struct[0].h[4,0] = V_5
        struct[0].h[5,0] = V_6
        struct[0].h[6,0] = V_7
        struct[0].h[7,0] = V_8
        struct[0].h[8,0] = V_9
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_1
        struct[0].Fx_ini[0,1] = Omega_b_1
        struct[0].Fx_ini[1,0] = (-V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fx_ini[1,1] = -D_1/(2*H_1)
        struct[0].Fx_ini[1,6] = 1/(2*H_1)
        struct[0].Fx_ini[2,2] = -1/T1d0_1
        struct[0].Fx_ini[3,3] = -1/T1q0_1
        struct[0].Fx_ini[4,4] = -1/T_r_1
        struct[0].Fx_ini[6,6] = -1/T_m_1
        struct[0].Fx_ini[7,7] = -K_delta_2
        struct[0].Fx_ini[7,8] = Omega_b_2
        struct[0].Fx_ini[8,7] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx_ini[8,8] = -D_2/(2*H_2)
        struct[0].Fx_ini[8,13] = 1/(2*H_2)
        struct[0].Fx_ini[9,9] = -1/T1d0_2
        struct[0].Fx_ini[10,10] = -1/T1q0_2
        struct[0].Fx_ini[11,11] = -1/T_r_2
        struct[0].Fx_ini[13,13] = -1/T_m_2
        struct[0].Fx_ini[14,14] = -K_delta_3
        struct[0].Fx_ini[14,15] = Omega_b_3
        struct[0].Fx_ini[15,14] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx_ini[15,15] = -D_3/(2*H_3)
        struct[0].Fx_ini[15,20] = 1/(2*H_3)
        struct[0].Fx_ini[16,16] = -1/T1d0_3
        struct[0].Fx_ini[17,17] = -1/T1q0_3
        struct[0].Fx_ini[18,18] = -1/T_r_3
        struct[0].Fx_ini[20,20] = -1/T_m_3
        struct[0].Fx_ini[21,21] = -K_delta_4
        struct[0].Fx_ini[21,22] = Omega_b_4
        struct[0].Fx_ini[22,21] = (-V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fx_ini[22,22] = -D_4/(2*H_4)
        struct[0].Fx_ini[22,27] = 1/(2*H_4)
        struct[0].Fx_ini[23,23] = -1/T1d0_4
        struct[0].Fx_ini[24,24] = -1/T1q0_4
        struct[0].Fx_ini[25,25] = -1/T_r_4
        struct[0].Fx_ini[27,27] = -1/T_m_4

    if mode == 11:

        struct[0].Fy_ini[0,50] = -Omega_b_1 
        struct[0].Fy_ini[1,0] = (-i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,1] = (V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,22] = (-2*R_a_1*i_d_1 - V_1*sin(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,23] = (-2*R_a_1*i_q_1 - V_1*cos(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,50] = D_1/(2*H_1) 
        struct[0].Fy_ini[2,22] = (X1d_1 - X_d_1)/T1d0_1 
        struct[0].Fy_ini[2,26] = 1/T1d0_1 
        struct[0].Fy_ini[3,23] = (-X1q_1 + X_q_1)/T1q0_1 
        struct[0].Fy_ini[4,0] = 1/T_r_1 
        struct[0].Fy_ini[5,0] = -1 
        struct[0].Fy_ini[6,27] = 1/T_m_1 
        struct[0].Fy_ini[7,50] = -Omega_b_2 
        struct[0].Fy_ini[8,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,28] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,29] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,50] = D_2/(2*H_2) 
        struct[0].Fy_ini[9,28] = (X1d_2 - X_d_2)/T1d0_2 
        struct[0].Fy_ini[9,32] = 1/T1d0_2 
        struct[0].Fy_ini[10,29] = (-X1q_2 + X_q_2)/T1q0_2 
        struct[0].Fy_ini[11,2] = 1/T_r_2 
        struct[0].Fy_ini[12,2] = -1 
        struct[0].Fy_ini[13,33] = 1/T_m_2 
        struct[0].Fy_ini[14,50] = -Omega_b_3 
        struct[0].Fy_ini[15,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,34] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,35] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,50] = D_3/(2*H_3) 
        struct[0].Fy_ini[16,34] = (X1d_3 - X_d_3)/T1d0_3 
        struct[0].Fy_ini[16,38] = 1/T1d0_3 
        struct[0].Fy_ini[17,35] = (-X1q_3 + X_q_3)/T1q0_3 
        struct[0].Fy_ini[18,4] = 1/T_r_3 
        struct[0].Fy_ini[19,4] = -1 
        struct[0].Fy_ini[20,39] = 1/T_m_3 
        struct[0].Fy_ini[21,50] = -Omega_b_4 
        struct[0].Fy_ini[22,6] = (-i_d_4*sin(delta_4 - theta_4) - i_q_4*cos(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,7] = (V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,40] = (-2*R_a_4*i_d_4 - V_4*sin(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,41] = (-2*R_a_4*i_q_4 - V_4*cos(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,50] = D_4/(2*H_4) 
        struct[0].Fy_ini[23,40] = (X1d_4 - X_d_4)/T1d0_4 
        struct[0].Fy_ini[23,44] = 1/T1d0_4 
        struct[0].Fy_ini[24,41] = (-X1q_4 + X_q_4)/T1q0_4 
        struct[0].Fy_ini[25,6] = 1/T_r_4 
        struct[0].Fy_ini[26,6] = -1 
        struct[0].Fy_ini[27,45] = 1/T_m_4 
        struct[0].Fy_ini[28,50] = -1 

        struct[0].Gx_ini[22,0] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[22,2] = -1
        struct[0].Gx_ini[23,0] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx_ini[23,3] = -1
        struct[0].Gx_ini[24,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[25,0] = -V_1*i_d_1*sin(delta_1 - theta_1) - V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gx_ini[26,4] = -K_a_1
        struct[0].Gx_ini[26,5] = K_ai_1
        struct[0].Gx_ini[27,1] = -1/Droop_1
        struct[0].Gx_ini[28,7] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gx_ini[28,9] = -1
        struct[0].Gx_ini[29,7] = V_2*cos(delta_2 - theta_2)
        struct[0].Gx_ini[29,10] = -1
        struct[0].Gx_ini[30,7] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gx_ini[31,7] = -V_2*i_d_2*sin(delta_2 - theta_2) - V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gx_ini[32,11] = -K_a_2
        struct[0].Gx_ini[32,12] = K_ai_2
        struct[0].Gx_ini[33,8] = -1/Droop_2
        struct[0].Gx_ini[34,14] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gx_ini[34,16] = -1
        struct[0].Gx_ini[35,14] = V_3*cos(delta_3 - theta_3)
        struct[0].Gx_ini[35,17] = -1
        struct[0].Gx_ini[36,14] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gx_ini[37,14] = -V_3*i_d_3*sin(delta_3 - theta_3) - V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gx_ini[38,18] = -K_a_3
        struct[0].Gx_ini[38,19] = K_ai_3
        struct[0].Gx_ini[39,15] = -1/Droop_3
        struct[0].Gx_ini[40,21] = -V_4*sin(delta_4 - theta_4)
        struct[0].Gx_ini[40,23] = -1
        struct[0].Gx_ini[41,21] = V_4*cos(delta_4 - theta_4)
        struct[0].Gx_ini[41,24] = -1
        struct[0].Gx_ini[42,21] = V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4)
        struct[0].Gx_ini[43,21] = -V_4*i_d_4*sin(delta_4 - theta_4) - V_4*i_q_4*cos(delta_4 - theta_4)
        struct[0].Gx_ini[44,25] = -K_a_4
        struct[0].Gx_ini[44,26] = K_ai_4
        struct[0].Gx_ini[45,22] = -1/Droop_4
        struct[0].Gx_ini[46,28] = K_sec_1/4
        struct[0].Gx_ini[47,28] = K_sec_2/4
        struct[0].Gx_ini[48,28] = K_sec_3/4
        struct[0].Gx_ini[49,28] = K_sec_4/4
        struct[0].Gx_ini[50,1] = 1/4
        struct[0].Gx_ini[50,8] = 1/4
        struct[0].Gx_ini[50,15] = 1/4
        struct[0].Gx_ini[50,22] = 1/4

        struct[0].Gy_ini[0,0] = 2*V_1*g_1_5 + V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,1] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[0,8] = V_1*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,9] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[0,24] = -S_n_1/S_base
        struct[0].Gy_ini[1,0] = 2*V_1*(-b_1_5 - bs_1_5/2) + V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[1,8] = V_1*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[1,25] = -S_n_1/S_base
        struct[0].Gy_ini[2,2] = 2*V_2*g_2_6 + V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,3] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[2,10] = V_2*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,11] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[2,30] = -S_n_2/S_base
        struct[0].Gy_ini[3,2] = 2*V_2*(-b_2_6 - bs_2_6/2) + V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[3,10] = V_2*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[3,31] = -S_n_2/S_base
        struct[0].Gy_ini[4,4] = V_11*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + 2*V_3*g_3_11
        struct[0].Gy_ini[4,5] = V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[4,20] = V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[4,21] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[4,36] = -S_n_3/S_base
        struct[0].Gy_ini[5,4] = V_11*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + 2*V_3*(-b_3_11 - bs_3_11/2)
        struct[0].Gy_ini[5,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[5,20] = V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[5,21] = V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[5,37] = -S_n_3/S_base
        struct[0].Gy_ini[6,6] = V_10*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + 2*V_4*g_4_10
        struct[0].Gy_ini[6,7] = V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[6,18] = V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[6,19] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[6,42] = -S_n_4/S_base
        struct[0].Gy_ini[7,6] = V_10*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + 2*V_4*(-b_4_10 - bs_4_10/2)
        struct[0].Gy_ini[7,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[7,18] = V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[7,19] = V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[7,43] = -S_n_4/S_base
        struct[0].Gy_ini[8,0] = V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[8,1] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[8,8] = V_1*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + 2*V_5*(g_1_5 + g_5_6) + V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[8,9] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[8,10] = V_5*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[8,11] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[9,0] = V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[9,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[9,8] = V_1*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + 2*V_5*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[9,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[9,10] = V_5*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[9,11] = V_5*V_6*(b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[10,2] = V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[10,3] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[10,8] = V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[10,9] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[10,10] = V_2*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + 2*V_6*(g_2_6 + g_5_6 + g_6_7) + V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[10,11] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[10,12] = V_6*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[10,13] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[11,2] = V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[11,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[11,8] = V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[11,9] = V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[11,10] = V_2*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + 2*V_6*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[11,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[11,12] = V_6*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[11,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[12,10] = V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[12,11] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[12,12] = V_6*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + 2*V_7*(g_6_7 + 2*g_7_8) + V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[12,13] = V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[12,14] = V_7*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[12,15] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[13,10] = V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[13,11] = V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[13,12] = V_6*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + 2*V_7*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[13,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[13,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[13,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[14,12] = V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[14,13] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[14,14] = V_7*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + 2*V_8*(2*g_7_8 + 2*g_8_9) + V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[14,15] = V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[14,16] = V_8*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[14,17] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[15,12] = V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[15,13] = V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[15,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + 2*V_8*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[15,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[15,16] = V_8*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[15,17] = V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[16,14] = V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[16,15] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[16,16] = V_10*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + 2*V_9*(2*g_8_9 + g_9_10)
        struct[0].Gy_ini[16,17] = V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[16,18] = V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[16,19] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[17,14] = V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[17,15] = V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[17,16] = V_10*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + 2*V_9*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].Gy_ini[17,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[17,18] = V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[17,19] = V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[18,6] = V_10*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[18,7] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[18,16] = V_10*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[18,17] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[18,18] = 2*V_10*(g_10_11 + g_4_10 + g_9_10) + V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[18,19] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[18,20] = V_10*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[18,21] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[19,6] = V_10*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[19,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[19,16] = V_10*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[19,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[19,18] = 2*V_10*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[19,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[19,20] = V_10*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[19,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[20,4] = V_11*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[20,5] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[20,18] = V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[20,19] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[20,20] = V_10*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + 2*V_11*(g_10_11 + g_3_11) + V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[20,21] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[21,4] = V_11*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[21,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[21,18] = V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[21,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[21,20] = V_10*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + 2*V_11*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[21,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy_ini[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[22,22] = X1d_1
        struct[0].Gy_ini[22,23] = R_a_1
        struct[0].Gy_ini[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy_ini[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[23,22] = R_a_1
        struct[0].Gy_ini[23,23] = -X1q_1
        struct[0].Gy_ini[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[24,22] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[24,23] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[25,22] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[25,23] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[28,2] = cos(delta_2 - theta_2)
        struct[0].Gy_ini[28,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[28,28] = X1d_2
        struct[0].Gy_ini[28,29] = R_a_2
        struct[0].Gy_ini[29,2] = sin(delta_2 - theta_2)
        struct[0].Gy_ini[29,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[29,28] = R_a_2
        struct[0].Gy_ini[29,29] = -X1q_2
        struct[0].Gy_ini[30,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[30,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[30,28] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[30,29] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[31,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[31,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[31,28] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[31,29] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[34,4] = cos(delta_3 - theta_3)
        struct[0].Gy_ini[34,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[34,34] = X1d_3
        struct[0].Gy_ini[34,35] = R_a_3
        struct[0].Gy_ini[35,4] = sin(delta_3 - theta_3)
        struct[0].Gy_ini[35,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[35,34] = R_a_3
        struct[0].Gy_ini[35,35] = -X1q_3
        struct[0].Gy_ini[36,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[36,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[36,34] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[36,35] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[37,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[37,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[37,34] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[37,35] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[40,6] = cos(delta_4 - theta_4)
        struct[0].Gy_ini[40,7] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[40,40] = X1d_4
        struct[0].Gy_ini[40,41] = R_a_4
        struct[0].Gy_ini[41,6] = sin(delta_4 - theta_4)
        struct[0].Gy_ini[41,7] = -V_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[41,40] = R_a_4
        struct[0].Gy_ini[41,41] = -X1q_4
        struct[0].Gy_ini[42,6] = i_d_4*sin(delta_4 - theta_4) + i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[42,7] = -V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[42,40] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[42,41] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[43,6] = i_d_4*cos(delta_4 - theta_4) - i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[43,7] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[43,40] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[43,41] = -V_4*sin(delta_4 - theta_4)



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_5 = struct[0].g_1_5
    b_1_5 = struct[0].b_1_5
    bs_1_5 = struct[0].bs_1_5
    g_2_6 = struct[0].g_2_6
    b_2_6 = struct[0].b_2_6
    bs_2_6 = struct[0].bs_2_6
    g_3_11 = struct[0].g_3_11
    b_3_11 = struct[0].b_3_11
    bs_3_11 = struct[0].bs_3_11
    g_4_10 = struct[0].g_4_10
    b_4_10 = struct[0].b_4_10
    bs_4_10 = struct[0].bs_4_10
    g_5_6 = struct[0].g_5_6
    b_5_6 = struct[0].b_5_6
    bs_5_6 = struct[0].bs_5_6
    g_6_7 = struct[0].g_6_7
    b_6_7 = struct[0].b_6_7
    bs_6_7 = struct[0].bs_6_7
    g_7_8 = struct[0].g_7_8
    b_7_8 = struct[0].b_7_8
    bs_7_8 = struct[0].bs_7_8
    g_8_9 = struct[0].g_8_9
    b_8_9 = struct[0].b_8_9
    bs_8_9 = struct[0].bs_8_9
    g_9_10 = struct[0].g_9_10
    b_9_10 = struct[0].b_9_10
    bs_9_10 = struct[0].bs_9_10
    g_10_11 = struct[0].g_10_11
    b_10_11 = struct[0].b_10_11
    bs_10_11 = struct[0].bs_10_11
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    U_9_n = struct[0].U_9_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    S_n_1 = struct[0].S_n_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    T1d0_1 = struct[0].T1d0_1
    T1q0_1 = struct[0].T1q0_1
    X_d_1 = struct[0].X_d_1
    X_q_1 = struct[0].X_q_1
    X1d_1 = struct[0].X1d_1
    X1q_1 = struct[0].X1q_1
    D_1 = struct[0].D_1
    R_a_1 = struct[0].R_a_1
    K_delta_1 = struct[0].K_delta_1
    K_a_1 = struct[0].K_a_1
    K_ai_1 = struct[0].K_ai_1
    T_r_1 = struct[0].T_r_1
    Droop_1 = struct[0].Droop_1
    T_m_1 = struct[0].T_m_1
    S_n_2 = struct[0].S_n_2
    H_2 = struct[0].H_2
    Omega_b_2 = struct[0].Omega_b_2
    T1d0_2 = struct[0].T1d0_2
    T1q0_2 = struct[0].T1q0_2
    X_d_2 = struct[0].X_d_2
    X_q_2 = struct[0].X_q_2
    X1d_2 = struct[0].X1d_2
    X1q_2 = struct[0].X1q_2
    D_2 = struct[0].D_2
    R_a_2 = struct[0].R_a_2
    K_delta_2 = struct[0].K_delta_2
    K_a_2 = struct[0].K_a_2
    K_ai_2 = struct[0].K_ai_2
    T_r_2 = struct[0].T_r_2
    Droop_2 = struct[0].Droop_2
    T_m_2 = struct[0].T_m_2
    S_n_3 = struct[0].S_n_3
    H_3 = struct[0].H_3
    Omega_b_3 = struct[0].Omega_b_3
    T1d0_3 = struct[0].T1d0_3
    T1q0_3 = struct[0].T1q0_3
    X_d_3 = struct[0].X_d_3
    X_q_3 = struct[0].X_q_3
    X1d_3 = struct[0].X1d_3
    X1q_3 = struct[0].X1q_3
    D_3 = struct[0].D_3
    R_a_3 = struct[0].R_a_3
    K_delta_3 = struct[0].K_delta_3
    K_a_3 = struct[0].K_a_3
    K_ai_3 = struct[0].K_ai_3
    T_r_3 = struct[0].T_r_3
    Droop_3 = struct[0].Droop_3
    T_m_3 = struct[0].T_m_3
    S_n_4 = struct[0].S_n_4
    H_4 = struct[0].H_4
    Omega_b_4 = struct[0].Omega_b_4
    T1d0_4 = struct[0].T1d0_4
    T1q0_4 = struct[0].T1q0_4
    X_d_4 = struct[0].X_d_4
    X_q_4 = struct[0].X_q_4
    X1d_4 = struct[0].X1d_4
    X1q_4 = struct[0].X1q_4
    D_4 = struct[0].D_4
    R_a_4 = struct[0].R_a_4
    K_delta_4 = struct[0].K_delta_4
    K_a_4 = struct[0].K_a_4
    K_ai_4 = struct[0].K_ai_4
    T_r_4 = struct[0].T_r_4
    Droop_4 = struct[0].Droop_4
    T_m_4 = struct[0].T_m_4
    K_sec_1 = struct[0].K_sec_1
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    K_sec_4 = struct[0].K_sec_4
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    P_4 = struct[0].P_4
    Q_4 = struct[0].Q_4
    P_5 = struct[0].P_5
    Q_5 = struct[0].Q_5
    P_6 = struct[0].P_6
    Q_6 = struct[0].Q_6
    P_7 = struct[0].P_7
    Q_7 = struct[0].Q_7
    P_8 = struct[0].P_8
    Q_8 = struct[0].Q_8
    P_9 = struct[0].P_9
    Q_9 = struct[0].Q_9
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    v_ref_1 = struct[0].v_ref_1
    v_pss_1 = struct[0].v_pss_1
    p_c_1 = struct[0].p_c_1
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    v_ref_4 = struct[0].v_ref_4
    v_pss_4 = struct[0].v_pss_4
    p_c_4 = struct[0].p_c_4
    
    # Dynamical states:
    delta_1 = struct[0].x[0,0]
    omega_1 = struct[0].x[1,0]
    e1q_1 = struct[0].x[2,0]
    e1d_1 = struct[0].x[3,0]
    v_c_1 = struct[0].x[4,0]
    xi_v_1 = struct[0].x[5,0]
    p_m_1 = struct[0].x[6,0]
    delta_2 = struct[0].x[7,0]
    omega_2 = struct[0].x[8,0]
    e1q_2 = struct[0].x[9,0]
    e1d_2 = struct[0].x[10,0]
    v_c_2 = struct[0].x[11,0]
    xi_v_2 = struct[0].x[12,0]
    p_m_2 = struct[0].x[13,0]
    delta_3 = struct[0].x[14,0]
    omega_3 = struct[0].x[15,0]
    e1q_3 = struct[0].x[16,0]
    e1d_3 = struct[0].x[17,0]
    v_c_3 = struct[0].x[18,0]
    xi_v_3 = struct[0].x[19,0]
    p_m_3 = struct[0].x[20,0]
    delta_4 = struct[0].x[21,0]
    omega_4 = struct[0].x[22,0]
    e1q_4 = struct[0].x[23,0]
    e1d_4 = struct[0].x[24,0]
    v_c_4 = struct[0].x[25,0]
    xi_v_4 = struct[0].x[26,0]
    p_m_4 = struct[0].x[27,0]
    xi_freq = struct[0].x[28,0]
    
    # Algebraic states:
    V_1 = struct[0].y_run[0,0]
    theta_1 = struct[0].y_run[1,0]
    V_2 = struct[0].y_run[2,0]
    theta_2 = struct[0].y_run[3,0]
    V_3 = struct[0].y_run[4,0]
    theta_3 = struct[0].y_run[5,0]
    V_4 = struct[0].y_run[6,0]
    theta_4 = struct[0].y_run[7,0]
    V_5 = struct[0].y_run[8,0]
    theta_5 = struct[0].y_run[9,0]
    V_6 = struct[0].y_run[10,0]
    theta_6 = struct[0].y_run[11,0]
    V_7 = struct[0].y_run[12,0]
    theta_7 = struct[0].y_run[13,0]
    V_8 = struct[0].y_run[14,0]
    theta_8 = struct[0].y_run[15,0]
    V_9 = struct[0].y_run[16,0]
    theta_9 = struct[0].y_run[17,0]
    V_10 = struct[0].y_run[18,0]
    theta_10 = struct[0].y_run[19,0]
    V_11 = struct[0].y_run[20,0]
    theta_11 = struct[0].y_run[21,0]
    i_d_1 = struct[0].y_run[22,0]
    i_q_1 = struct[0].y_run[23,0]
    p_g_1_1 = struct[0].y_run[24,0]
    q_g_1_1 = struct[0].y_run[25,0]
    v_f_1 = struct[0].y_run[26,0]
    p_m_ref_1 = struct[0].y_run[27,0]
    i_d_2 = struct[0].y_run[28,0]
    i_q_2 = struct[0].y_run[29,0]
    p_g_2_1 = struct[0].y_run[30,0]
    q_g_2_1 = struct[0].y_run[31,0]
    v_f_2 = struct[0].y_run[32,0]
    p_m_ref_2 = struct[0].y_run[33,0]
    i_d_3 = struct[0].y_run[34,0]
    i_q_3 = struct[0].y_run[35,0]
    p_g_3_1 = struct[0].y_run[36,0]
    q_g_3_1 = struct[0].y_run[37,0]
    v_f_3 = struct[0].y_run[38,0]
    p_m_ref_3 = struct[0].y_run[39,0]
    i_d_4 = struct[0].y_run[40,0]
    i_q_4 = struct[0].y_run[41,0]
    p_g_4_1 = struct[0].y_run[42,0]
    q_g_4_1 = struct[0].y_run[43,0]
    v_f_4 = struct[0].y_run[44,0]
    p_m_ref_4 = struct[0].y_run[45,0]
    p_r_1 = struct[0].y_run[46,0]
    p_r_2 = struct[0].y_run[47,0]
    p_r_3 = struct[0].y_run[48,0]
    p_r_4 = struct[0].y_run[49,0]
    omega_coi = struct[0].y_run[50,0]
    
    struct[0].u_run[0,0] = P_1
    struct[0].u_run[1,0] = Q_1
    struct[0].u_run[2,0] = P_2
    struct[0].u_run[3,0] = Q_2
    struct[0].u_run[4,0] = P_3
    struct[0].u_run[5,0] = Q_3
    struct[0].u_run[6,0] = P_4
    struct[0].u_run[7,0] = Q_4
    struct[0].u_run[8,0] = P_5
    struct[0].u_run[9,0] = Q_5
    struct[0].u_run[10,0] = P_6
    struct[0].u_run[11,0] = Q_6
    struct[0].u_run[12,0] = P_7
    struct[0].u_run[13,0] = Q_7
    struct[0].u_run[14,0] = P_8
    struct[0].u_run[15,0] = Q_8
    struct[0].u_run[16,0] = P_9
    struct[0].u_run[17,0] = Q_9
    struct[0].u_run[18,0] = P_10
    struct[0].u_run[19,0] = Q_10
    struct[0].u_run[20,0] = P_11
    struct[0].u_run[21,0] = Q_11
    struct[0].u_run[22,0] = v_ref_1
    struct[0].u_run[23,0] = v_pss_1
    struct[0].u_run[24,0] = p_c_1
    struct[0].u_run[25,0] = v_ref_2
    struct[0].u_run[26,0] = v_pss_2
    struct[0].u_run[27,0] = p_c_2
    struct[0].u_run[28,0] = v_ref_3
    struct[0].u_run[29,0] = v_pss_3
    struct[0].u_run[30,0] = p_c_3
    struct[0].u_run[31,0] = v_ref_4
    struct[0].u_run[32,0] = v_pss_4
    struct[0].u_run[33,0] = p_c_4
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_1*delta_1 + Omega_b_1*(omega_1 - omega_coi)
        struct[0].f[1,0] = (-D_1*(omega_1 - omega_coi) - i_d_1*(R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_m_1)/(2*H_1)
        struct[0].f[2,0] = (-e1q_1 - i_d_1*(-X1d_1 + X_d_1) + v_f_1)/T1d0_1
        struct[0].f[3,0] = (-e1d_1 + i_q_1*(-X1q_1 + X_q_1))/T1q0_1
        struct[0].f[4,0] = (V_1 - v_c_1)/T_r_1
        struct[0].f[5,0] = -V_1 + v_ref_1
        struct[0].f[6,0] = (-p_m_1 + p_m_ref_1)/T_m_1
        struct[0].f[7,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[8,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[9,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[10,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[11,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[12,0] = -V_2 + v_ref_2
        struct[0].f[13,0] = (-p_m_2 + p_m_ref_2)/T_m_2
        struct[0].f[14,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[15,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[16,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[17,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[18,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[19,0] = -V_3 + v_ref_3
        struct[0].f[20,0] = (-p_m_3 + p_m_ref_3)/T_m_3
        struct[0].f[21,0] = -K_delta_4*delta_4 + Omega_b_4*(omega_4 - omega_coi)
        struct[0].f[22,0] = (-D_4*(omega_4 - omega_coi) - i_d_4*(R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4)) - i_q_4*(R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4)) + p_m_4)/(2*H_4)
        struct[0].f[23,0] = (-e1q_4 - i_d_4*(-X1d_4 + X_d_4) + v_f_4)/T1d0_4
        struct[0].f[24,0] = (-e1d_4 + i_q_4*(-X1q_4 + X_q_4))/T1q0_4
        struct[0].f[25,0] = (V_4 - v_c_4)/T_r_4
        struct[0].f[26,0] = -V_4 + v_ref_4
        struct[0].f[27,0] = (-p_m_4 + p_m_ref_4)/T_m_4
        struct[0].f[28,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_5 + V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_5 - bs_1_5/2) + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_2**2*g_2_6 + V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_2**2*(-b_2_6 - bs_2_6/2) + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + V_3**2*g_3_11 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + V_3**2*(-b_3_11 - bs_3_11/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = -P_4/S_base + V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_4**2*g_4_10 - S_n_4*p_g_4_1/S_base
        struct[0].g[7,0] = -Q_4/S_base + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_4**2*(-b_4_10 - bs_4_10/2) - S_n_4*q_g_4_1/S_base
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5**2*(g_1_5 + g_5_6) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + V_5**2*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6**2*(g_2_6 + g_5_6 + g_6_7) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + V_6**2*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].g[12,0] = -P_7/S_base + V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7**2*(g_6_7 + 2*g_7_8) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].g[13,0] = -Q_7/S_base + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + V_7**2*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].g[14,0] = -P_8/S_base + V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8**2*(2*g_7_8 + 2*g_8_9) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].g[15,0] = -Q_8/S_base + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + V_8**2*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].g[16,0] = -P_9/S_base + V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + V_9**2*(2*g_8_9 + g_9_10)
        struct[0].g[17,0] = -Q_9/S_base + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + V_9**2*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].g[18,0] = -P_10/S_base + V_10**2*(g_10_11 + g_4_10 + g_9_10) + V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].g[19,0] = -Q_10/S_base + V_10**2*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].g[20,0] = -P_11/S_base + V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11**2*(g_10_11 + g_3_11) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].g[21,0] = -Q_11/S_base + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_11**2*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].g[22,0] = R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1) + X1d_1*i_d_1 - e1q_1
        struct[0].g[23,0] = R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1) - X1q_1*i_q_1 - e1d_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = K_a_1*(-v_c_1 + v_pss_1 + v_ref_1) + K_ai_1*xi_v_1 - v_f_1
        struct[0].g[27,0] = p_c_1 - p_m_ref_1 + p_r_1 - (omega_1 - 1)/Droop_1
        struct[0].g[28,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[29,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[30,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[31,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[32,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[33,0] = p_c_2 - p_m_ref_2 + p_r_2 - (omega_2 - 1)/Droop_2
        struct[0].g[34,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[35,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[36,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[37,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[38,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[39,0] = p_c_3 - p_m_ref_3 + p_r_3 - (omega_3 - 1)/Droop_3
        struct[0].g[40,0] = R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4) + X1d_4*i_d_4 - e1q_4
        struct[0].g[41,0] = R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4) - X1q_4*i_q_4 - e1d_4
        struct[0].g[42,0] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4) - p_g_4_1
        struct[0].g[43,0] = V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4) - q_g_4_1
        struct[0].g[44,0] = K_a_4*(-v_c_4 + v_pss_4 + v_ref_4) + K_ai_4*xi_v_4 - v_f_4
        struct[0].g[45,0] = p_c_4 - p_m_ref_4 + p_r_4 - (omega_4 - 1)/Droop_4
        struct[0].g[46,0] = K_sec_1*xi_freq/4 - p_r_1
        struct[0].g[47,0] = K_sec_2*xi_freq/4 - p_r_2
        struct[0].g[48,0] = K_sec_3*xi_freq/4 - p_r_3
        struct[0].g[49,0] = K_sec_4*xi_freq/4 - p_r_4
        struct[0].g[50,0] = omega_1/4 + omega_2/4 + omega_3/4 + omega_4/4 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = V_4
        struct[0].h[4,0] = V_5
        struct[0].h[5,0] = V_6
        struct[0].h[6,0] = V_7
        struct[0].h[7,0] = V_8
        struct[0].h[8,0] = V_9
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_1
        struct[0].Fx[0,1] = Omega_b_1
        struct[0].Fx[1,0] = (-V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fx[1,1] = -D_1/(2*H_1)
        struct[0].Fx[1,6] = 1/(2*H_1)
        struct[0].Fx[2,2] = -1/T1d0_1
        struct[0].Fx[3,3] = -1/T1q0_1
        struct[0].Fx[4,4] = -1/T_r_1
        struct[0].Fx[6,6] = -1/T_m_1
        struct[0].Fx[7,7] = -K_delta_2
        struct[0].Fx[7,8] = Omega_b_2
        struct[0].Fx[8,7] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx[8,8] = -D_2/(2*H_2)
        struct[0].Fx[8,13] = 1/(2*H_2)
        struct[0].Fx[9,9] = -1/T1d0_2
        struct[0].Fx[10,10] = -1/T1q0_2
        struct[0].Fx[11,11] = -1/T_r_2
        struct[0].Fx[13,13] = -1/T_m_2
        struct[0].Fx[14,14] = -K_delta_3
        struct[0].Fx[14,15] = Omega_b_3
        struct[0].Fx[15,14] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx[15,15] = -D_3/(2*H_3)
        struct[0].Fx[15,20] = 1/(2*H_3)
        struct[0].Fx[16,16] = -1/T1d0_3
        struct[0].Fx[17,17] = -1/T1q0_3
        struct[0].Fx[18,18] = -1/T_r_3
        struct[0].Fx[20,20] = -1/T_m_3
        struct[0].Fx[21,21] = -K_delta_4
        struct[0].Fx[21,22] = Omega_b_4
        struct[0].Fx[22,21] = (-V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fx[22,22] = -D_4/(2*H_4)
        struct[0].Fx[22,27] = 1/(2*H_4)
        struct[0].Fx[23,23] = -1/T1d0_4
        struct[0].Fx[24,24] = -1/T1q0_4
        struct[0].Fx[25,25] = -1/T_r_4
        struct[0].Fx[27,27] = -1/T_m_4

    if mode == 11:

        struct[0].Fy[0,50] = -Omega_b_1
        struct[0].Fy[1,0] = (-i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,1] = (V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,22] = (-2*R_a_1*i_d_1 - V_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,23] = (-2*R_a_1*i_q_1 - V_1*cos(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,50] = D_1/(2*H_1)
        struct[0].Fy[2,22] = (X1d_1 - X_d_1)/T1d0_1
        struct[0].Fy[2,26] = 1/T1d0_1
        struct[0].Fy[3,23] = (-X1q_1 + X_q_1)/T1q0_1
        struct[0].Fy[4,0] = 1/T_r_1
        struct[0].Fy[5,0] = -1
        struct[0].Fy[6,27] = 1/T_m_1
        struct[0].Fy[7,50] = -Omega_b_2
        struct[0].Fy[8,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,28] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,29] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,50] = D_2/(2*H_2)
        struct[0].Fy[9,28] = (X1d_2 - X_d_2)/T1d0_2
        struct[0].Fy[9,32] = 1/T1d0_2
        struct[0].Fy[10,29] = (-X1q_2 + X_q_2)/T1q0_2
        struct[0].Fy[11,2] = 1/T_r_2
        struct[0].Fy[12,2] = -1
        struct[0].Fy[13,33] = 1/T_m_2
        struct[0].Fy[14,50] = -Omega_b_3
        struct[0].Fy[15,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,34] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,35] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,50] = D_3/(2*H_3)
        struct[0].Fy[16,34] = (X1d_3 - X_d_3)/T1d0_3
        struct[0].Fy[16,38] = 1/T1d0_3
        struct[0].Fy[17,35] = (-X1q_3 + X_q_3)/T1q0_3
        struct[0].Fy[18,4] = 1/T_r_3
        struct[0].Fy[19,4] = -1
        struct[0].Fy[20,39] = 1/T_m_3
        struct[0].Fy[21,50] = -Omega_b_4
        struct[0].Fy[22,6] = (-i_d_4*sin(delta_4 - theta_4) - i_q_4*cos(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,7] = (V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,40] = (-2*R_a_4*i_d_4 - V_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,41] = (-2*R_a_4*i_q_4 - V_4*cos(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,50] = D_4/(2*H_4)
        struct[0].Fy[23,40] = (X1d_4 - X_d_4)/T1d0_4
        struct[0].Fy[23,44] = 1/T1d0_4
        struct[0].Fy[24,41] = (-X1q_4 + X_q_4)/T1q0_4
        struct[0].Fy[25,6] = 1/T_r_4
        struct[0].Fy[26,6] = -1
        struct[0].Fy[27,45] = 1/T_m_4
        struct[0].Fy[28,50] = -1

        struct[0].Gx[22,0] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gx[22,2] = -1
        struct[0].Gx[23,0] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx[23,3] = -1
        struct[0].Gx[24,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gx[25,0] = -V_1*i_d_1*sin(delta_1 - theta_1) - V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gx[26,4] = -K_a_1
        struct[0].Gx[26,5] = K_ai_1
        struct[0].Gx[27,1] = -1/Droop_1
        struct[0].Gx[28,7] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gx[28,9] = -1
        struct[0].Gx[29,7] = V_2*cos(delta_2 - theta_2)
        struct[0].Gx[29,10] = -1
        struct[0].Gx[30,7] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gx[31,7] = -V_2*i_d_2*sin(delta_2 - theta_2) - V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gx[32,11] = -K_a_2
        struct[0].Gx[32,12] = K_ai_2
        struct[0].Gx[33,8] = -1/Droop_2
        struct[0].Gx[34,14] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gx[34,16] = -1
        struct[0].Gx[35,14] = V_3*cos(delta_3 - theta_3)
        struct[0].Gx[35,17] = -1
        struct[0].Gx[36,14] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gx[37,14] = -V_3*i_d_3*sin(delta_3 - theta_3) - V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gx[38,18] = -K_a_3
        struct[0].Gx[38,19] = K_ai_3
        struct[0].Gx[39,15] = -1/Droop_3
        struct[0].Gx[40,21] = -V_4*sin(delta_4 - theta_4)
        struct[0].Gx[40,23] = -1
        struct[0].Gx[41,21] = V_4*cos(delta_4 - theta_4)
        struct[0].Gx[41,24] = -1
        struct[0].Gx[42,21] = V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4)
        struct[0].Gx[43,21] = -V_4*i_d_4*sin(delta_4 - theta_4) - V_4*i_q_4*cos(delta_4 - theta_4)
        struct[0].Gx[44,25] = -K_a_4
        struct[0].Gx[44,26] = K_ai_4
        struct[0].Gx[45,22] = -1/Droop_4
        struct[0].Gx[46,28] = K_sec_1/4
        struct[0].Gx[47,28] = K_sec_2/4
        struct[0].Gx[48,28] = K_sec_3/4
        struct[0].Gx[49,28] = K_sec_4/4
        struct[0].Gx[50,1] = 1/4
        struct[0].Gx[50,8] = 1/4
        struct[0].Gx[50,15] = 1/4
        struct[0].Gx[50,22] = 1/4

        struct[0].Gy[0,0] = 2*V_1*g_1_5 + V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[0,1] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[0,8] = V_1*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[0,9] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[0,24] = -S_n_1/S_base
        struct[0].Gy[1,0] = 2*V_1*(-b_1_5 - bs_1_5/2) + V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[1,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[1,8] = V_1*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[1,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[1,25] = -S_n_1/S_base
        struct[0].Gy[2,2] = 2*V_2*g_2_6 + V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[2,3] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[2,10] = V_2*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[2,11] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[2,30] = -S_n_2/S_base
        struct[0].Gy[3,2] = 2*V_2*(-b_2_6 - bs_2_6/2) + V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[3,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[3,10] = V_2*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[3,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[3,31] = -S_n_2/S_base
        struct[0].Gy[4,4] = V_11*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + 2*V_3*g_3_11
        struct[0].Gy[4,5] = V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[4,20] = V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[4,21] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[4,36] = -S_n_3/S_base
        struct[0].Gy[5,4] = V_11*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + 2*V_3*(-b_3_11 - bs_3_11/2)
        struct[0].Gy[5,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[5,20] = V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[5,21] = V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[5,37] = -S_n_3/S_base
        struct[0].Gy[6,6] = V_10*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + 2*V_4*g_4_10
        struct[0].Gy[6,7] = V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[6,18] = V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[6,19] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[6,42] = -S_n_4/S_base
        struct[0].Gy[7,6] = V_10*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + 2*V_4*(-b_4_10 - bs_4_10/2)
        struct[0].Gy[7,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[7,18] = V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[7,19] = V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[7,43] = -S_n_4/S_base
        struct[0].Gy[8,0] = V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[8,1] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[8,8] = V_1*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + 2*V_5*(g_1_5 + g_5_6) + V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[8,9] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[8,10] = V_5*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[8,11] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[9,0] = V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[9,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[9,8] = V_1*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + 2*V_5*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[9,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[9,10] = V_5*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[9,11] = V_5*V_6*(b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[10,2] = V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[10,3] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[10,8] = V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[10,9] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[10,10] = V_2*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + 2*V_6*(g_2_6 + g_5_6 + g_6_7) + V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[10,11] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[10,12] = V_6*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[10,13] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[11,2] = V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[11,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[11,8] = V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[11,9] = V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[11,10] = V_2*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + 2*V_6*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[11,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[11,12] = V_6*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[11,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[12,10] = V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[12,11] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[12,12] = V_6*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + 2*V_7*(g_6_7 + 2*g_7_8) + V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[12,13] = V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[12,14] = V_7*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[12,15] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[13,10] = V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[13,11] = V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[13,12] = V_6*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + 2*V_7*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[13,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[13,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[13,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[14,12] = V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[14,13] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[14,14] = V_7*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + 2*V_8*(2*g_7_8 + 2*g_8_9) + V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[14,15] = V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[14,16] = V_8*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[14,17] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[15,12] = V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[15,13] = V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[15,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + 2*V_8*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[15,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[15,16] = V_8*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[15,17] = V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[16,14] = V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[16,15] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[16,16] = V_10*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + 2*V_9*(2*g_8_9 + g_9_10)
        struct[0].Gy[16,17] = V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[16,18] = V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[16,19] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[17,14] = V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[17,15] = V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[17,16] = V_10*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + 2*V_9*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].Gy[17,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[17,18] = V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[17,19] = V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[18,6] = V_10*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[18,7] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[18,16] = V_10*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[18,17] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[18,18] = 2*V_10*(g_10_11 + g_4_10 + g_9_10) + V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[18,19] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[18,20] = V_10*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[18,21] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[19,6] = V_10*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[19,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[19,16] = V_10*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[19,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[19,18] = 2*V_10*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[19,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[19,20] = V_10*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[19,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[20,4] = V_11*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[20,5] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[20,18] = V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[20,19] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[20,20] = V_10*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + 2*V_11*(g_10_11 + g_3_11) + V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[20,21] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[21,4] = V_11*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[21,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[21,18] = V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[21,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[21,20] = V_10*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + 2*V_11*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[21,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy[22,22] = X1d_1
        struct[0].Gy[22,23] = R_a_1
        struct[0].Gy[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy[23,22] = R_a_1
        struct[0].Gy[23,23] = -X1q_1
        struct[0].Gy[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[24,22] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy[24,23] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[25,22] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy[25,23] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gy[28,2] = cos(delta_2 - theta_2)
        struct[0].Gy[28,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[28,28] = X1d_2
        struct[0].Gy[28,29] = R_a_2
        struct[0].Gy[29,2] = sin(delta_2 - theta_2)
        struct[0].Gy[29,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy[29,28] = R_a_2
        struct[0].Gy[29,29] = -X1q_2
        struct[0].Gy[30,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[30,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[30,28] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[30,29] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[31,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[31,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[31,28] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[31,29] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy[34,4] = cos(delta_3 - theta_3)
        struct[0].Gy[34,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[34,34] = X1d_3
        struct[0].Gy[34,35] = R_a_3
        struct[0].Gy[35,4] = sin(delta_3 - theta_3)
        struct[0].Gy[35,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy[35,34] = R_a_3
        struct[0].Gy[35,35] = -X1q_3
        struct[0].Gy[36,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[36,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[36,34] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[36,35] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[37,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[37,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[37,34] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[37,35] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy[40,6] = cos(delta_4 - theta_4)
        struct[0].Gy[40,7] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy[40,40] = X1d_4
        struct[0].Gy[40,41] = R_a_4
        struct[0].Gy[41,6] = sin(delta_4 - theta_4)
        struct[0].Gy[41,7] = -V_4*cos(delta_4 - theta_4)
        struct[0].Gy[41,40] = R_a_4
        struct[0].Gy[41,41] = -X1q_4
        struct[0].Gy[42,6] = i_d_4*sin(delta_4 - theta_4) + i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy[42,7] = -V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy[42,40] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy[42,41] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy[43,6] = i_d_4*cos(delta_4 - theta_4) - i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy[43,7] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy[43,40] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy[43,41] = -V_4*sin(delta_4 - theta_4)

    if mode > 12:

        struct[0].Fu[5,22] = 1
        struct[0].Fu[12,25] = 1
        struct[0].Fu[19,28] = 1
        struct[0].Fu[26,31] = 1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[6,6] = -1/S_base
        struct[0].Gu[7,7] = -1/S_base
        struct[0].Gu[8,8] = -1/S_base
        struct[0].Gu[9,9] = -1/S_base
        struct[0].Gu[10,10] = -1/S_base
        struct[0].Gu[11,11] = -1/S_base
        struct[0].Gu[12,12] = -1/S_base
        struct[0].Gu[13,13] = -1/S_base
        struct[0].Gu[14,14] = -1/S_base
        struct[0].Gu[15,15] = -1/S_base
        struct[0].Gu[16,16] = -1/S_base
        struct[0].Gu[17,17] = -1/S_base
        struct[0].Gu[18,18] = -1/S_base
        struct[0].Gu[19,19] = -1/S_base
        struct[0].Gu[20,20] = -1/S_base
        struct[0].Gu[21,21] = -1/S_base
        struct[0].Gu[26,22] = K_a_1
        struct[0].Gu[26,23] = K_a_1
        struct[0].Gu[32,25] = K_a_2
        struct[0].Gu[32,26] = K_a_2
        struct[0].Gu[38,28] = K_a_3
        struct[0].Gu[38,29] = K_a_3
        struct[0].Gu[44,31] = K_a_4
        struct[0].Gu[44,32] = K_a_4


        struct[0].Hy[0,0] = 1
        struct[0].Hy[1,2] = 1
        struct[0].Hy[2,4] = 1
        struct[0].Hy[3,6] = 1
        struct[0].Hy[4,8] = 1
        struct[0].Hy[5,10] = 1
        struct[0].Hy[6,12] = 1
        struct[0].Hy[7,14] = 1
        struct[0].Hy[8,16] = 1
        struct[0].Hy[9,18] = 1
        struct[0].Hy[10,20] = 1




def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_5 = struct[0].g_1_5
    b_1_5 = struct[0].b_1_5
    bs_1_5 = struct[0].bs_1_5
    g_2_6 = struct[0].g_2_6
    b_2_6 = struct[0].b_2_6
    bs_2_6 = struct[0].bs_2_6
    g_3_11 = struct[0].g_3_11
    b_3_11 = struct[0].b_3_11
    bs_3_11 = struct[0].bs_3_11
    g_4_10 = struct[0].g_4_10
    b_4_10 = struct[0].b_4_10
    bs_4_10 = struct[0].bs_4_10
    g_5_6 = struct[0].g_5_6
    b_5_6 = struct[0].b_5_6
    bs_5_6 = struct[0].bs_5_6
    g_6_7 = struct[0].g_6_7
    b_6_7 = struct[0].b_6_7
    bs_6_7 = struct[0].bs_6_7
    g_7_8 = struct[0].g_7_8
    b_7_8 = struct[0].b_7_8
    bs_7_8 = struct[0].bs_7_8
    g_8_9 = struct[0].g_8_9
    b_8_9 = struct[0].b_8_9
    bs_8_9 = struct[0].bs_8_9
    g_9_10 = struct[0].g_9_10
    b_9_10 = struct[0].b_9_10
    bs_9_10 = struct[0].bs_9_10
    g_10_11 = struct[0].g_10_11
    b_10_11 = struct[0].b_10_11
    bs_10_11 = struct[0].bs_10_11
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    U_9_n = struct[0].U_9_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    S_n_1 = struct[0].S_n_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    T1d0_1 = struct[0].T1d0_1
    T1q0_1 = struct[0].T1q0_1
    X_d_1 = struct[0].X_d_1
    X_q_1 = struct[0].X_q_1
    X1d_1 = struct[0].X1d_1
    X1q_1 = struct[0].X1q_1
    D_1 = struct[0].D_1
    R_a_1 = struct[0].R_a_1
    K_delta_1 = struct[0].K_delta_1
    K_a_1 = struct[0].K_a_1
    K_ai_1 = struct[0].K_ai_1
    T_r_1 = struct[0].T_r_1
    Droop_1 = struct[0].Droop_1
    T_m_1 = struct[0].T_m_1
    S_n_2 = struct[0].S_n_2
    H_2 = struct[0].H_2
    Omega_b_2 = struct[0].Omega_b_2
    T1d0_2 = struct[0].T1d0_2
    T1q0_2 = struct[0].T1q0_2
    X_d_2 = struct[0].X_d_2
    X_q_2 = struct[0].X_q_2
    X1d_2 = struct[0].X1d_2
    X1q_2 = struct[0].X1q_2
    D_2 = struct[0].D_2
    R_a_2 = struct[0].R_a_2
    K_delta_2 = struct[0].K_delta_2
    K_a_2 = struct[0].K_a_2
    K_ai_2 = struct[0].K_ai_2
    T_r_2 = struct[0].T_r_2
    Droop_2 = struct[0].Droop_2
    T_m_2 = struct[0].T_m_2
    S_n_3 = struct[0].S_n_3
    H_3 = struct[0].H_3
    Omega_b_3 = struct[0].Omega_b_3
    T1d0_3 = struct[0].T1d0_3
    T1q0_3 = struct[0].T1q0_3
    X_d_3 = struct[0].X_d_3
    X_q_3 = struct[0].X_q_3
    X1d_3 = struct[0].X1d_3
    X1q_3 = struct[0].X1q_3
    D_3 = struct[0].D_3
    R_a_3 = struct[0].R_a_3
    K_delta_3 = struct[0].K_delta_3
    K_a_3 = struct[0].K_a_3
    K_ai_3 = struct[0].K_ai_3
    T_r_3 = struct[0].T_r_3
    Droop_3 = struct[0].Droop_3
    T_m_3 = struct[0].T_m_3
    S_n_4 = struct[0].S_n_4
    H_4 = struct[0].H_4
    Omega_b_4 = struct[0].Omega_b_4
    T1d0_4 = struct[0].T1d0_4
    T1q0_4 = struct[0].T1q0_4
    X_d_4 = struct[0].X_d_4
    X_q_4 = struct[0].X_q_4
    X1d_4 = struct[0].X1d_4
    X1q_4 = struct[0].X1q_4
    D_4 = struct[0].D_4
    R_a_4 = struct[0].R_a_4
    K_delta_4 = struct[0].K_delta_4
    K_a_4 = struct[0].K_a_4
    K_ai_4 = struct[0].K_ai_4
    T_r_4 = struct[0].T_r_4
    Droop_4 = struct[0].Droop_4
    T_m_4 = struct[0].T_m_4
    K_sec_1 = struct[0].K_sec_1
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    K_sec_4 = struct[0].K_sec_4
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    P_4 = struct[0].P_4
    Q_4 = struct[0].Q_4
    P_5 = struct[0].P_5
    Q_5 = struct[0].Q_5
    P_6 = struct[0].P_6
    Q_6 = struct[0].Q_6
    P_7 = struct[0].P_7
    Q_7 = struct[0].Q_7
    P_8 = struct[0].P_8
    Q_8 = struct[0].Q_8
    P_9 = struct[0].P_9
    Q_9 = struct[0].Q_9
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    v_ref_1 = struct[0].v_ref_1
    v_pss_1 = struct[0].v_pss_1
    p_c_1 = struct[0].p_c_1
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    v_ref_4 = struct[0].v_ref_4
    v_pss_4 = struct[0].v_pss_4
    p_c_4 = struct[0].p_c_4
    
    # Dynamical states:
    delta_1 = struct[0].x[0,0]
    omega_1 = struct[0].x[1,0]
    e1q_1 = struct[0].x[2,0]
    e1d_1 = struct[0].x[3,0]
    v_c_1 = struct[0].x[4,0]
    xi_v_1 = struct[0].x[5,0]
    p_m_1 = struct[0].x[6,0]
    delta_2 = struct[0].x[7,0]
    omega_2 = struct[0].x[8,0]
    e1q_2 = struct[0].x[9,0]
    e1d_2 = struct[0].x[10,0]
    v_c_2 = struct[0].x[11,0]
    xi_v_2 = struct[0].x[12,0]
    p_m_2 = struct[0].x[13,0]
    delta_3 = struct[0].x[14,0]
    omega_3 = struct[0].x[15,0]
    e1q_3 = struct[0].x[16,0]
    e1d_3 = struct[0].x[17,0]
    v_c_3 = struct[0].x[18,0]
    xi_v_3 = struct[0].x[19,0]
    p_m_3 = struct[0].x[20,0]
    delta_4 = struct[0].x[21,0]
    omega_4 = struct[0].x[22,0]
    e1q_4 = struct[0].x[23,0]
    e1d_4 = struct[0].x[24,0]
    v_c_4 = struct[0].x[25,0]
    xi_v_4 = struct[0].x[26,0]
    p_m_4 = struct[0].x[27,0]
    xi_freq = struct[0].x[28,0]
    
    # Algebraic states:
    V_1 = struct[0].y_ini[0,0]
    theta_1 = struct[0].y_ini[1,0]
    V_2 = struct[0].y_ini[2,0]
    theta_2 = struct[0].y_ini[3,0]
    V_3 = struct[0].y_ini[4,0]
    theta_3 = struct[0].y_ini[5,0]
    V_4 = struct[0].y_ini[6,0]
    theta_4 = struct[0].y_ini[7,0]
    V_5 = struct[0].y_ini[8,0]
    theta_5 = struct[0].y_ini[9,0]
    V_6 = struct[0].y_ini[10,0]
    theta_6 = struct[0].y_ini[11,0]
    V_7 = struct[0].y_ini[12,0]
    theta_7 = struct[0].y_ini[13,0]
    V_8 = struct[0].y_ini[14,0]
    theta_8 = struct[0].y_ini[15,0]
    V_9 = struct[0].y_ini[16,0]
    theta_9 = struct[0].y_ini[17,0]
    V_10 = struct[0].y_ini[18,0]
    theta_10 = struct[0].y_ini[19,0]
    V_11 = struct[0].y_ini[20,0]
    theta_11 = struct[0].y_ini[21,0]
    i_d_1 = struct[0].y_ini[22,0]
    i_q_1 = struct[0].y_ini[23,0]
    p_g_1_1 = struct[0].y_ini[24,0]
    q_g_1_1 = struct[0].y_ini[25,0]
    v_f_1 = struct[0].y_ini[26,0]
    p_m_ref_1 = struct[0].y_ini[27,0]
    i_d_2 = struct[0].y_ini[28,0]
    i_q_2 = struct[0].y_ini[29,0]
    p_g_2_1 = struct[0].y_ini[30,0]
    q_g_2_1 = struct[0].y_ini[31,0]
    v_f_2 = struct[0].y_ini[32,0]
    p_m_ref_2 = struct[0].y_ini[33,0]
    i_d_3 = struct[0].y_ini[34,0]
    i_q_3 = struct[0].y_ini[35,0]
    p_g_3_1 = struct[0].y_ini[36,0]
    q_g_3_1 = struct[0].y_ini[37,0]
    v_f_3 = struct[0].y_ini[38,0]
    p_m_ref_3 = struct[0].y_ini[39,0]
    i_d_4 = struct[0].y_ini[40,0]
    i_q_4 = struct[0].y_ini[41,0]
    p_g_4_1 = struct[0].y_ini[42,0]
    q_g_4_1 = struct[0].y_ini[43,0]
    v_f_4 = struct[0].y_ini[44,0]
    p_m_ref_4 = struct[0].y_ini[45,0]
    p_r_1 = struct[0].y_ini[46,0]
    p_r_2 = struct[0].y_ini[47,0]
    p_r_3 = struct[0].y_ini[48,0]
    p_r_4 = struct[0].y_ini[49,0]
    omega_coi = struct[0].y_ini[50,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_1*delta_1 + Omega_b_1*(omega_1 - omega_coi)
        struct[0].f[1,0] = (-D_1*(omega_1 - omega_coi) - i_d_1*(R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_m_1)/(2*H_1)
        struct[0].f[2,0] = (-e1q_1 - i_d_1*(-X1d_1 + X_d_1) + v_f_1)/T1d0_1
        struct[0].f[3,0] = (-e1d_1 + i_q_1*(-X1q_1 + X_q_1))/T1q0_1
        struct[0].f[4,0] = (V_1 - v_c_1)/T_r_1
        struct[0].f[5,0] = -V_1 + v_ref_1
        struct[0].f[6,0] = (-p_m_1 + p_m_ref_1)/T_m_1
        struct[0].f[7,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[8,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[9,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[10,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[11,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[12,0] = -V_2 + v_ref_2
        struct[0].f[13,0] = (-p_m_2 + p_m_ref_2)/T_m_2
        struct[0].f[14,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[15,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[16,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[17,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[18,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[19,0] = -V_3 + v_ref_3
        struct[0].f[20,0] = (-p_m_3 + p_m_ref_3)/T_m_3
        struct[0].f[21,0] = -K_delta_4*delta_4 + Omega_b_4*(omega_4 - omega_coi)
        struct[0].f[22,0] = (-D_4*(omega_4 - omega_coi) - i_d_4*(R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4)) - i_q_4*(R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4)) + p_m_4)/(2*H_4)
        struct[0].f[23,0] = (-e1q_4 - i_d_4*(-X1d_4 + X_d_4) + v_f_4)/T1d0_4
        struct[0].f[24,0] = (-e1d_4 + i_q_4*(-X1q_4 + X_q_4))/T1q0_4
        struct[0].f[25,0] = (V_4 - v_c_4)/T_r_4
        struct[0].f[26,0] = -V_4 + v_ref_4
        struct[0].f[27,0] = (-p_m_4 + p_m_ref_4)/T_m_4
        struct[0].f[28,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_5 + V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_5 - bs_1_5/2) + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_2**2*g_2_6 + V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_2**2*(-b_2_6 - bs_2_6/2) + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + V_3**2*g_3_11 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + V_3**2*(-b_3_11 - bs_3_11/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = -P_4/S_base + V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_4**2*g_4_10 - S_n_4*p_g_4_1/S_base
        struct[0].g[7,0] = -Q_4/S_base + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_4**2*(-b_4_10 - bs_4_10/2) - S_n_4*q_g_4_1/S_base
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5**2*(g_1_5 + g_5_6) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + V_5**2*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6**2*(g_2_6 + g_5_6 + g_6_7) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + V_6**2*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].g[12,0] = -P_7/S_base + V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7**2*(g_6_7 + 2*g_7_8) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].g[13,0] = -Q_7/S_base + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + V_7**2*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].g[14,0] = -P_8/S_base + V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8**2*(2*g_7_8 + 2*g_8_9) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].g[15,0] = -Q_8/S_base + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + V_8**2*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].g[16,0] = -P_9/S_base + V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + V_9**2*(2*g_8_9 + g_9_10)
        struct[0].g[17,0] = -Q_9/S_base + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + V_9**2*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].g[18,0] = -P_10/S_base + V_10**2*(g_10_11 + g_4_10 + g_9_10) + V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].g[19,0] = -Q_10/S_base + V_10**2*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].g[20,0] = -P_11/S_base + V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11**2*(g_10_11 + g_3_11) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].g[21,0] = -Q_11/S_base + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_11**2*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].g[22,0] = R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1) + X1d_1*i_d_1 - e1q_1
        struct[0].g[23,0] = R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1) - X1q_1*i_q_1 - e1d_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = K_a_1*(-v_c_1 + v_pss_1 + v_ref_1) + K_ai_1*xi_v_1 - v_f_1
        struct[0].g[27,0] = p_c_1 - p_m_ref_1 + p_r_1 - (omega_1 - 1)/Droop_1
        struct[0].g[28,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[29,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[30,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[31,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[32,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[33,0] = p_c_2 - p_m_ref_2 + p_r_2 - (omega_2 - 1)/Droop_2
        struct[0].g[34,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[35,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[36,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[37,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[38,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[39,0] = p_c_3 - p_m_ref_3 + p_r_3 - (omega_3 - 1)/Droop_3
        struct[0].g[40,0] = R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4) + X1d_4*i_d_4 - e1q_4
        struct[0].g[41,0] = R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4) - X1q_4*i_q_4 - e1d_4
        struct[0].g[42,0] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4) - p_g_4_1
        struct[0].g[43,0] = V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4) - q_g_4_1
        struct[0].g[44,0] = K_a_4*(-v_c_4 + v_pss_4 + v_ref_4) + K_ai_4*xi_v_4 - v_f_4
        struct[0].g[45,0] = p_c_4 - p_m_ref_4 + p_r_4 - (omega_4 - 1)/Droop_4
        struct[0].g[46,0] = K_sec_1*xi_freq/4 - p_r_1
        struct[0].g[47,0] = K_sec_2*xi_freq/4 - p_r_2
        struct[0].g[48,0] = K_sec_3*xi_freq/4 - p_r_3
        struct[0].g[49,0] = K_sec_4*xi_freq/4 - p_r_4
        struct[0].g[50,0] = omega_1/4 + omega_2/4 + omega_3/4 + omega_4/4 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = V_4
        struct[0].h[4,0] = V_5
        struct[0].h[5,0] = V_6
        struct[0].h[6,0] = V_7
        struct[0].h[7,0] = V_8
        struct[0].h[8,0] = V_9
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_1
        struct[0].Fx_ini[0,1] = Omega_b_1
        struct[0].Fx_ini[1,0] = (-V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fx_ini[1,1] = -D_1/(2*H_1)
        struct[0].Fx_ini[1,6] = 1/(2*H_1)
        struct[0].Fx_ini[2,2] = -1/T1d0_1
        struct[0].Fx_ini[3,3] = -1/T1q0_1
        struct[0].Fx_ini[4,4] = -1/T_r_1
        struct[0].Fx_ini[6,6] = -1/T_m_1
        struct[0].Fx_ini[7,7] = -K_delta_2
        struct[0].Fx_ini[7,8] = Omega_b_2
        struct[0].Fx_ini[8,7] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx_ini[8,8] = -D_2/(2*H_2)
        struct[0].Fx_ini[8,13] = 1/(2*H_2)
        struct[0].Fx_ini[9,9] = -1/T1d0_2
        struct[0].Fx_ini[10,10] = -1/T1q0_2
        struct[0].Fx_ini[11,11] = -1/T_r_2
        struct[0].Fx_ini[13,13] = -1/T_m_2
        struct[0].Fx_ini[14,14] = -K_delta_3
        struct[0].Fx_ini[14,15] = Omega_b_3
        struct[0].Fx_ini[15,14] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx_ini[15,15] = -D_3/(2*H_3)
        struct[0].Fx_ini[15,20] = 1/(2*H_3)
        struct[0].Fx_ini[16,16] = -1/T1d0_3
        struct[0].Fx_ini[17,17] = -1/T1q0_3
        struct[0].Fx_ini[18,18] = -1/T_r_3
        struct[0].Fx_ini[20,20] = -1/T_m_3
        struct[0].Fx_ini[21,21] = -K_delta_4
        struct[0].Fx_ini[21,22] = Omega_b_4
        struct[0].Fx_ini[22,21] = (-V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fx_ini[22,22] = -D_4/(2*H_4)
        struct[0].Fx_ini[22,27] = 1/(2*H_4)
        struct[0].Fx_ini[23,23] = -1/T1d0_4
        struct[0].Fx_ini[24,24] = -1/T1q0_4
        struct[0].Fx_ini[25,25] = -1/T_r_4
        struct[0].Fx_ini[27,27] = -1/T_m_4

    if mode == 11:

        struct[0].Fy_ini[0,50] = -Omega_b_1 
        struct[0].Fy_ini[1,0] = (-i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,1] = (V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,22] = (-2*R_a_1*i_d_1 - V_1*sin(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,23] = (-2*R_a_1*i_q_1 - V_1*cos(delta_1 - theta_1))/(2*H_1) 
        struct[0].Fy_ini[1,50] = D_1/(2*H_1) 
        struct[0].Fy_ini[2,22] = (X1d_1 - X_d_1)/T1d0_1 
        struct[0].Fy_ini[2,26] = 1/T1d0_1 
        struct[0].Fy_ini[3,23] = (-X1q_1 + X_q_1)/T1q0_1 
        struct[0].Fy_ini[4,0] = 1/T_r_1 
        struct[0].Fy_ini[5,0] = -1 
        struct[0].Fy_ini[6,27] = 1/T_m_1 
        struct[0].Fy_ini[7,50] = -Omega_b_2 
        struct[0].Fy_ini[8,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,28] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,29] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[8,50] = D_2/(2*H_2) 
        struct[0].Fy_ini[9,28] = (X1d_2 - X_d_2)/T1d0_2 
        struct[0].Fy_ini[9,32] = 1/T1d0_2 
        struct[0].Fy_ini[10,29] = (-X1q_2 + X_q_2)/T1q0_2 
        struct[0].Fy_ini[11,2] = 1/T_r_2 
        struct[0].Fy_ini[12,2] = -1 
        struct[0].Fy_ini[13,33] = 1/T_m_2 
        struct[0].Fy_ini[14,50] = -Omega_b_3 
        struct[0].Fy_ini[15,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,34] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,35] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[15,50] = D_3/(2*H_3) 
        struct[0].Fy_ini[16,34] = (X1d_3 - X_d_3)/T1d0_3 
        struct[0].Fy_ini[16,38] = 1/T1d0_3 
        struct[0].Fy_ini[17,35] = (-X1q_3 + X_q_3)/T1q0_3 
        struct[0].Fy_ini[18,4] = 1/T_r_3 
        struct[0].Fy_ini[19,4] = -1 
        struct[0].Fy_ini[20,39] = 1/T_m_3 
        struct[0].Fy_ini[21,50] = -Omega_b_4 
        struct[0].Fy_ini[22,6] = (-i_d_4*sin(delta_4 - theta_4) - i_q_4*cos(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,7] = (V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,40] = (-2*R_a_4*i_d_4 - V_4*sin(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,41] = (-2*R_a_4*i_q_4 - V_4*cos(delta_4 - theta_4))/(2*H_4) 
        struct[0].Fy_ini[22,50] = D_4/(2*H_4) 
        struct[0].Fy_ini[23,40] = (X1d_4 - X_d_4)/T1d0_4 
        struct[0].Fy_ini[23,44] = 1/T1d0_4 
        struct[0].Fy_ini[24,41] = (-X1q_4 + X_q_4)/T1q0_4 
        struct[0].Fy_ini[25,6] = 1/T_r_4 
        struct[0].Fy_ini[26,6] = -1 
        struct[0].Fy_ini[27,45] = 1/T_m_4 
        struct[0].Fy_ini[28,50] = -1 

        struct[0].Gy_ini[0,0] = 2*V_1*g_1_5 + V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,1] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[0,8] = V_1*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,9] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[0,24] = -S_n_1/S_base
        struct[0].Gy_ini[1,0] = 2*V_1*(-b_1_5 - bs_1_5/2) + V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[1,8] = V_1*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[1,25] = -S_n_1/S_base
        struct[0].Gy_ini[2,2] = 2*V_2*g_2_6 + V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,3] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[2,10] = V_2*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,11] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[2,30] = -S_n_2/S_base
        struct[0].Gy_ini[3,2] = 2*V_2*(-b_2_6 - bs_2_6/2) + V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[3,10] = V_2*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[3,31] = -S_n_2/S_base
        struct[0].Gy_ini[4,4] = V_11*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + 2*V_3*g_3_11
        struct[0].Gy_ini[4,5] = V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[4,20] = V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[4,21] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[4,36] = -S_n_3/S_base
        struct[0].Gy_ini[5,4] = V_11*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + 2*V_3*(-b_3_11 - bs_3_11/2)
        struct[0].Gy_ini[5,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[5,20] = V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[5,21] = V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[5,37] = -S_n_3/S_base
        struct[0].Gy_ini[6,6] = V_10*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + 2*V_4*g_4_10
        struct[0].Gy_ini[6,7] = V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[6,18] = V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[6,19] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[6,42] = -S_n_4/S_base
        struct[0].Gy_ini[7,6] = V_10*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + 2*V_4*(-b_4_10 - bs_4_10/2)
        struct[0].Gy_ini[7,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[7,18] = V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[7,19] = V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[7,43] = -S_n_4/S_base
        struct[0].Gy_ini[8,0] = V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[8,1] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[8,8] = V_1*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + 2*V_5*(g_1_5 + g_5_6) + V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[8,9] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[8,10] = V_5*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[8,11] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[9,0] = V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy_ini[9,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy_ini[9,8] = V_1*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + 2*V_5*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[9,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[9,10] = V_5*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[9,11] = V_5*V_6*(b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[10,2] = V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[10,3] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[10,8] = V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[10,9] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[10,10] = V_2*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + 2*V_6*(g_2_6 + g_5_6 + g_6_7) + V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[10,11] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[10,12] = V_6*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[10,13] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[11,2] = V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy_ini[11,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy_ini[11,8] = V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy_ini[11,9] = V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy_ini[11,10] = V_2*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + 2*V_6*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[11,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[11,12] = V_6*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[11,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[12,10] = V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[12,11] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[12,12] = V_6*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + 2*V_7*(g_6_7 + 2*g_7_8) + V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[12,13] = V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[12,14] = V_7*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[12,15] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[13,10] = V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy_ini[13,11] = V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy_ini[13,12] = V_6*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + 2*V_7*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[13,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[13,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[13,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[14,12] = V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[14,13] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[14,14] = V_7*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + 2*V_8*(2*g_7_8 + 2*g_8_9) + V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[14,15] = V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[14,16] = V_8*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[14,17] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[15,12] = V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy_ini[15,13] = V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy_ini[15,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + 2*V_8*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[15,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[15,16] = V_8*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[15,17] = V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[16,14] = V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[16,15] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[16,16] = V_10*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + 2*V_9*(2*g_8_9 + g_9_10)
        struct[0].Gy_ini[16,17] = V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[16,18] = V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[16,19] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[17,14] = V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy_ini[17,15] = V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[17,16] = V_10*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + 2*V_9*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].Gy_ini[17,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy_ini[17,18] = V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[17,19] = V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[18,6] = V_10*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[18,7] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[18,16] = V_10*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[18,17] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[18,18] = 2*V_10*(g_10_11 + g_4_10 + g_9_10) + V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[18,19] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[18,20] = V_10*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[18,21] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[19,6] = V_10*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy_ini[19,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy_ini[19,16] = V_10*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[19,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[19,18] = 2*V_10*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy_ini[19,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy_ini[19,20] = V_10*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[19,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[20,4] = V_11*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[20,5] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[20,18] = V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[20,19] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[20,20] = V_10*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + 2*V_11*(g_10_11 + g_3_11) + V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[20,21] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[21,4] = V_11*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[21,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[21,18] = V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy_ini[21,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy_ini[21,20] = V_10*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + 2*V_11*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy_ini[21,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy_ini[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy_ini[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[22,22] = X1d_1
        struct[0].Gy_ini[22,23] = R_a_1
        struct[0].Gy_ini[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy_ini[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[23,22] = R_a_1
        struct[0].Gy_ini[23,23] = -X1q_1
        struct[0].Gy_ini[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[24,22] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[24,23] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[25,22] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[25,23] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[25,25] = -1
        struct[0].Gy_ini[26,26] = -1
        struct[0].Gy_ini[27,27] = -1
        struct[0].Gy_ini[27,46] = 1
        struct[0].Gy_ini[28,2] = cos(delta_2 - theta_2)
        struct[0].Gy_ini[28,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[28,28] = X1d_2
        struct[0].Gy_ini[28,29] = R_a_2
        struct[0].Gy_ini[29,2] = sin(delta_2 - theta_2)
        struct[0].Gy_ini[29,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[29,28] = R_a_2
        struct[0].Gy_ini[29,29] = -X1q_2
        struct[0].Gy_ini[30,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[30,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[30,28] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[30,29] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[30,30] = -1
        struct[0].Gy_ini[31,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[31,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[31,28] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[31,29] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[31,31] = -1
        struct[0].Gy_ini[32,32] = -1
        struct[0].Gy_ini[33,33] = -1
        struct[0].Gy_ini[33,47] = 1
        struct[0].Gy_ini[34,4] = cos(delta_3 - theta_3)
        struct[0].Gy_ini[34,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[34,34] = X1d_3
        struct[0].Gy_ini[34,35] = R_a_3
        struct[0].Gy_ini[35,4] = sin(delta_3 - theta_3)
        struct[0].Gy_ini[35,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[35,34] = R_a_3
        struct[0].Gy_ini[35,35] = -X1q_3
        struct[0].Gy_ini[36,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[36,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[36,34] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[36,35] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[36,36] = -1
        struct[0].Gy_ini[37,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[37,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[37,34] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[37,35] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[37,37] = -1
        struct[0].Gy_ini[38,38] = -1
        struct[0].Gy_ini[39,39] = -1
        struct[0].Gy_ini[39,48] = 1
        struct[0].Gy_ini[40,6] = cos(delta_4 - theta_4)
        struct[0].Gy_ini[40,7] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[40,40] = X1d_4
        struct[0].Gy_ini[40,41] = R_a_4
        struct[0].Gy_ini[41,6] = sin(delta_4 - theta_4)
        struct[0].Gy_ini[41,7] = -V_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[41,40] = R_a_4
        struct[0].Gy_ini[41,41] = -X1q_4
        struct[0].Gy_ini[42,6] = i_d_4*sin(delta_4 - theta_4) + i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[42,7] = -V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[42,40] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[42,41] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[42,42] = -1
        struct[0].Gy_ini[43,6] = i_d_4*cos(delta_4 - theta_4) - i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[43,7] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[43,40] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy_ini[43,41] = -V_4*sin(delta_4 - theta_4)
        struct[0].Gy_ini[43,43] = -1
        struct[0].Gy_ini[44,44] = -1
        struct[0].Gy_ini[45,45] = -1
        struct[0].Gy_ini[45,49] = 1
        struct[0].Gy_ini[46,46] = -1
        struct[0].Gy_ini[47,47] = -1
        struct[0].Gy_ini[48,48] = -1
        struct[0].Gy_ini[49,49] = -1
        struct[0].Gy_ini[50,50] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_5 = struct[0].g_1_5
    b_1_5 = struct[0].b_1_5
    bs_1_5 = struct[0].bs_1_5
    g_2_6 = struct[0].g_2_6
    b_2_6 = struct[0].b_2_6
    bs_2_6 = struct[0].bs_2_6
    g_3_11 = struct[0].g_3_11
    b_3_11 = struct[0].b_3_11
    bs_3_11 = struct[0].bs_3_11
    g_4_10 = struct[0].g_4_10
    b_4_10 = struct[0].b_4_10
    bs_4_10 = struct[0].bs_4_10
    g_5_6 = struct[0].g_5_6
    b_5_6 = struct[0].b_5_6
    bs_5_6 = struct[0].bs_5_6
    g_6_7 = struct[0].g_6_7
    b_6_7 = struct[0].b_6_7
    bs_6_7 = struct[0].bs_6_7
    g_7_8 = struct[0].g_7_8
    b_7_8 = struct[0].b_7_8
    bs_7_8 = struct[0].bs_7_8
    g_8_9 = struct[0].g_8_9
    b_8_9 = struct[0].b_8_9
    bs_8_9 = struct[0].bs_8_9
    g_9_10 = struct[0].g_9_10
    b_9_10 = struct[0].b_9_10
    bs_9_10 = struct[0].bs_9_10
    g_10_11 = struct[0].g_10_11
    b_10_11 = struct[0].b_10_11
    bs_10_11 = struct[0].bs_10_11
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    U_9_n = struct[0].U_9_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    S_n_1 = struct[0].S_n_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    T1d0_1 = struct[0].T1d0_1
    T1q0_1 = struct[0].T1q0_1
    X_d_1 = struct[0].X_d_1
    X_q_1 = struct[0].X_q_1
    X1d_1 = struct[0].X1d_1
    X1q_1 = struct[0].X1q_1
    D_1 = struct[0].D_1
    R_a_1 = struct[0].R_a_1
    K_delta_1 = struct[0].K_delta_1
    K_a_1 = struct[0].K_a_1
    K_ai_1 = struct[0].K_ai_1
    T_r_1 = struct[0].T_r_1
    Droop_1 = struct[0].Droop_1
    T_m_1 = struct[0].T_m_1
    S_n_2 = struct[0].S_n_2
    H_2 = struct[0].H_2
    Omega_b_2 = struct[0].Omega_b_2
    T1d0_2 = struct[0].T1d0_2
    T1q0_2 = struct[0].T1q0_2
    X_d_2 = struct[0].X_d_2
    X_q_2 = struct[0].X_q_2
    X1d_2 = struct[0].X1d_2
    X1q_2 = struct[0].X1q_2
    D_2 = struct[0].D_2
    R_a_2 = struct[0].R_a_2
    K_delta_2 = struct[0].K_delta_2
    K_a_2 = struct[0].K_a_2
    K_ai_2 = struct[0].K_ai_2
    T_r_2 = struct[0].T_r_2
    Droop_2 = struct[0].Droop_2
    T_m_2 = struct[0].T_m_2
    S_n_3 = struct[0].S_n_3
    H_3 = struct[0].H_3
    Omega_b_3 = struct[0].Omega_b_3
    T1d0_3 = struct[0].T1d0_3
    T1q0_3 = struct[0].T1q0_3
    X_d_3 = struct[0].X_d_3
    X_q_3 = struct[0].X_q_3
    X1d_3 = struct[0].X1d_3
    X1q_3 = struct[0].X1q_3
    D_3 = struct[0].D_3
    R_a_3 = struct[0].R_a_3
    K_delta_3 = struct[0].K_delta_3
    K_a_3 = struct[0].K_a_3
    K_ai_3 = struct[0].K_ai_3
    T_r_3 = struct[0].T_r_3
    Droop_3 = struct[0].Droop_3
    T_m_3 = struct[0].T_m_3
    S_n_4 = struct[0].S_n_4
    H_4 = struct[0].H_4
    Omega_b_4 = struct[0].Omega_b_4
    T1d0_4 = struct[0].T1d0_4
    T1q0_4 = struct[0].T1q0_4
    X_d_4 = struct[0].X_d_4
    X_q_4 = struct[0].X_q_4
    X1d_4 = struct[0].X1d_4
    X1q_4 = struct[0].X1q_4
    D_4 = struct[0].D_4
    R_a_4 = struct[0].R_a_4
    K_delta_4 = struct[0].K_delta_4
    K_a_4 = struct[0].K_a_4
    K_ai_4 = struct[0].K_ai_4
    T_r_4 = struct[0].T_r_4
    Droop_4 = struct[0].Droop_4
    T_m_4 = struct[0].T_m_4
    K_sec_1 = struct[0].K_sec_1
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    K_sec_4 = struct[0].K_sec_4
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    P_4 = struct[0].P_4
    Q_4 = struct[0].Q_4
    P_5 = struct[0].P_5
    Q_5 = struct[0].Q_5
    P_6 = struct[0].P_6
    Q_6 = struct[0].Q_6
    P_7 = struct[0].P_7
    Q_7 = struct[0].Q_7
    P_8 = struct[0].P_8
    Q_8 = struct[0].Q_8
    P_9 = struct[0].P_9
    Q_9 = struct[0].Q_9
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    v_ref_1 = struct[0].v_ref_1
    v_pss_1 = struct[0].v_pss_1
    p_c_1 = struct[0].p_c_1
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    v_ref_4 = struct[0].v_ref_4
    v_pss_4 = struct[0].v_pss_4
    p_c_4 = struct[0].p_c_4
    
    # Dynamical states:
    delta_1 = struct[0].x[0,0]
    omega_1 = struct[0].x[1,0]
    e1q_1 = struct[0].x[2,0]
    e1d_1 = struct[0].x[3,0]
    v_c_1 = struct[0].x[4,0]
    xi_v_1 = struct[0].x[5,0]
    p_m_1 = struct[0].x[6,0]
    delta_2 = struct[0].x[7,0]
    omega_2 = struct[0].x[8,0]
    e1q_2 = struct[0].x[9,0]
    e1d_2 = struct[0].x[10,0]
    v_c_2 = struct[0].x[11,0]
    xi_v_2 = struct[0].x[12,0]
    p_m_2 = struct[0].x[13,0]
    delta_3 = struct[0].x[14,0]
    omega_3 = struct[0].x[15,0]
    e1q_3 = struct[0].x[16,0]
    e1d_3 = struct[0].x[17,0]
    v_c_3 = struct[0].x[18,0]
    xi_v_3 = struct[0].x[19,0]
    p_m_3 = struct[0].x[20,0]
    delta_4 = struct[0].x[21,0]
    omega_4 = struct[0].x[22,0]
    e1q_4 = struct[0].x[23,0]
    e1d_4 = struct[0].x[24,0]
    v_c_4 = struct[0].x[25,0]
    xi_v_4 = struct[0].x[26,0]
    p_m_4 = struct[0].x[27,0]
    xi_freq = struct[0].x[28,0]
    
    # Algebraic states:
    V_1 = struct[0].y_run[0,0]
    theta_1 = struct[0].y_run[1,0]
    V_2 = struct[0].y_run[2,0]
    theta_2 = struct[0].y_run[3,0]
    V_3 = struct[0].y_run[4,0]
    theta_3 = struct[0].y_run[5,0]
    V_4 = struct[0].y_run[6,0]
    theta_4 = struct[0].y_run[7,0]
    V_5 = struct[0].y_run[8,0]
    theta_5 = struct[0].y_run[9,0]
    V_6 = struct[0].y_run[10,0]
    theta_6 = struct[0].y_run[11,0]
    V_7 = struct[0].y_run[12,0]
    theta_7 = struct[0].y_run[13,0]
    V_8 = struct[0].y_run[14,0]
    theta_8 = struct[0].y_run[15,0]
    V_9 = struct[0].y_run[16,0]
    theta_9 = struct[0].y_run[17,0]
    V_10 = struct[0].y_run[18,0]
    theta_10 = struct[0].y_run[19,0]
    V_11 = struct[0].y_run[20,0]
    theta_11 = struct[0].y_run[21,0]
    i_d_1 = struct[0].y_run[22,0]
    i_q_1 = struct[0].y_run[23,0]
    p_g_1_1 = struct[0].y_run[24,0]
    q_g_1_1 = struct[0].y_run[25,0]
    v_f_1 = struct[0].y_run[26,0]
    p_m_ref_1 = struct[0].y_run[27,0]
    i_d_2 = struct[0].y_run[28,0]
    i_q_2 = struct[0].y_run[29,0]
    p_g_2_1 = struct[0].y_run[30,0]
    q_g_2_1 = struct[0].y_run[31,0]
    v_f_2 = struct[0].y_run[32,0]
    p_m_ref_2 = struct[0].y_run[33,0]
    i_d_3 = struct[0].y_run[34,0]
    i_q_3 = struct[0].y_run[35,0]
    p_g_3_1 = struct[0].y_run[36,0]
    q_g_3_1 = struct[0].y_run[37,0]
    v_f_3 = struct[0].y_run[38,0]
    p_m_ref_3 = struct[0].y_run[39,0]
    i_d_4 = struct[0].y_run[40,0]
    i_q_4 = struct[0].y_run[41,0]
    p_g_4_1 = struct[0].y_run[42,0]
    q_g_4_1 = struct[0].y_run[43,0]
    v_f_4 = struct[0].y_run[44,0]
    p_m_ref_4 = struct[0].y_run[45,0]
    p_r_1 = struct[0].y_run[46,0]
    p_r_2 = struct[0].y_run[47,0]
    p_r_3 = struct[0].y_run[48,0]
    p_r_4 = struct[0].y_run[49,0]
    omega_coi = struct[0].y_run[50,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_1*delta_1 + Omega_b_1*(omega_1 - omega_coi)
        struct[0].f[1,0] = (-D_1*(omega_1 - omega_coi) - i_d_1*(R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_m_1)/(2*H_1)
        struct[0].f[2,0] = (-e1q_1 - i_d_1*(-X1d_1 + X_d_1) + v_f_1)/T1d0_1
        struct[0].f[3,0] = (-e1d_1 + i_q_1*(-X1q_1 + X_q_1))/T1q0_1
        struct[0].f[4,0] = (V_1 - v_c_1)/T_r_1
        struct[0].f[5,0] = -V_1 + v_ref_1
        struct[0].f[6,0] = (-p_m_1 + p_m_ref_1)/T_m_1
        struct[0].f[7,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[8,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[9,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[10,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[11,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[12,0] = -V_2 + v_ref_2
        struct[0].f[13,0] = (-p_m_2 + p_m_ref_2)/T_m_2
        struct[0].f[14,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[15,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[16,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[17,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[18,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[19,0] = -V_3 + v_ref_3
        struct[0].f[20,0] = (-p_m_3 + p_m_ref_3)/T_m_3
        struct[0].f[21,0] = -K_delta_4*delta_4 + Omega_b_4*(omega_4 - omega_coi)
        struct[0].f[22,0] = (-D_4*(omega_4 - omega_coi) - i_d_4*(R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4)) - i_q_4*(R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4)) + p_m_4)/(2*H_4)
        struct[0].f[23,0] = (-e1q_4 - i_d_4*(-X1d_4 + X_d_4) + v_f_4)/T1d0_4
        struct[0].f[24,0] = (-e1d_4 + i_q_4*(-X1q_4 + X_q_4))/T1q0_4
        struct[0].f[25,0] = (V_4 - v_c_4)/T_r_4
        struct[0].f[26,0] = -V_4 + v_ref_4
        struct[0].f[27,0] = (-p_m_4 + p_m_ref_4)/T_m_4
        struct[0].f[28,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_5 + V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_5 - bs_1_5/2) + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_2**2*g_2_6 + V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_2**2*(-b_2_6 - bs_2_6/2) + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + V_3**2*g_3_11 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + V_3**2*(-b_3_11 - bs_3_11/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = -P_4/S_base + V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_4**2*g_4_10 - S_n_4*p_g_4_1/S_base
        struct[0].g[7,0] = -Q_4/S_base + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_4**2*(-b_4_10 - bs_4_10/2) - S_n_4*q_g_4_1/S_base
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5**2*(g_1_5 + g_5_6) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + V_5**2*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6**2*(g_2_6 + g_5_6 + g_6_7) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + V_6**2*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].g[12,0] = -P_7/S_base + V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7**2*(g_6_7 + 2*g_7_8) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].g[13,0] = -Q_7/S_base + V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + V_7**2*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].g[14,0] = -P_8/S_base + V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8**2*(2*g_7_8 + 2*g_8_9) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].g[15,0] = -Q_8/S_base + V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + V_8**2*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].g[16,0] = -P_9/S_base + V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + V_9**2*(2*g_8_9 + g_9_10)
        struct[0].g[17,0] = -Q_9/S_base + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + V_9**2*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].g[18,0] = -P_10/S_base + V_10**2*(g_10_11 + g_4_10 + g_9_10) + V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].g[19,0] = -Q_10/S_base + V_10**2*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].g[20,0] = -P_11/S_base + V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11**2*(g_10_11 + g_3_11) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].g[21,0] = -Q_11/S_base + V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_11**2*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].g[22,0] = R_a_1*i_q_1 + V_1*cos(delta_1 - theta_1) + X1d_1*i_d_1 - e1q_1
        struct[0].g[23,0] = R_a_1*i_d_1 + V_1*sin(delta_1 - theta_1) - X1q_1*i_q_1 - e1d_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = K_a_1*(-v_c_1 + v_pss_1 + v_ref_1) + K_ai_1*xi_v_1 - v_f_1
        struct[0].g[27,0] = p_c_1 - p_m_ref_1 + p_r_1 - (omega_1 - 1)/Droop_1
        struct[0].g[28,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[29,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[30,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[31,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[32,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[33,0] = p_c_2 - p_m_ref_2 + p_r_2 - (omega_2 - 1)/Droop_2
        struct[0].g[34,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[35,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[36,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[37,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[38,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[39,0] = p_c_3 - p_m_ref_3 + p_r_3 - (omega_3 - 1)/Droop_3
        struct[0].g[40,0] = R_a_4*i_q_4 + V_4*cos(delta_4 - theta_4) + X1d_4*i_d_4 - e1q_4
        struct[0].g[41,0] = R_a_4*i_d_4 + V_4*sin(delta_4 - theta_4) - X1q_4*i_q_4 - e1d_4
        struct[0].g[42,0] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4) - p_g_4_1
        struct[0].g[43,0] = V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4) - q_g_4_1
        struct[0].g[44,0] = K_a_4*(-v_c_4 + v_pss_4 + v_ref_4) + K_ai_4*xi_v_4 - v_f_4
        struct[0].g[45,0] = p_c_4 - p_m_ref_4 + p_r_4 - (omega_4 - 1)/Droop_4
        struct[0].g[46,0] = K_sec_1*xi_freq/4 - p_r_1
        struct[0].g[47,0] = K_sec_2*xi_freq/4 - p_r_2
        struct[0].g[48,0] = K_sec_3*xi_freq/4 - p_r_3
        struct[0].g[49,0] = K_sec_4*xi_freq/4 - p_r_4
        struct[0].g[50,0] = omega_1/4 + omega_2/4 + omega_3/4 + omega_4/4 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = V_4
        struct[0].h[4,0] = V_5
        struct[0].h[5,0] = V_6
        struct[0].h[6,0] = V_7
        struct[0].h[7,0] = V_8
        struct[0].h[8,0] = V_9
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_1
        struct[0].Fx[0,1] = Omega_b_1
        struct[0].Fx[1,0] = (-V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fx[1,1] = -D_1/(2*H_1)
        struct[0].Fx[1,6] = 1/(2*H_1)
        struct[0].Fx[2,2] = -1/T1d0_1
        struct[0].Fx[3,3] = -1/T1q0_1
        struct[0].Fx[4,4] = -1/T_r_1
        struct[0].Fx[6,6] = -1/T_m_1
        struct[0].Fx[7,7] = -K_delta_2
        struct[0].Fx[7,8] = Omega_b_2
        struct[0].Fx[8,7] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx[8,8] = -D_2/(2*H_2)
        struct[0].Fx[8,13] = 1/(2*H_2)
        struct[0].Fx[9,9] = -1/T1d0_2
        struct[0].Fx[10,10] = -1/T1q0_2
        struct[0].Fx[11,11] = -1/T_r_2
        struct[0].Fx[13,13] = -1/T_m_2
        struct[0].Fx[14,14] = -K_delta_3
        struct[0].Fx[14,15] = Omega_b_3
        struct[0].Fx[15,14] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx[15,15] = -D_3/(2*H_3)
        struct[0].Fx[15,20] = 1/(2*H_3)
        struct[0].Fx[16,16] = -1/T1d0_3
        struct[0].Fx[17,17] = -1/T1q0_3
        struct[0].Fx[18,18] = -1/T_r_3
        struct[0].Fx[20,20] = -1/T_m_3
        struct[0].Fx[21,21] = -K_delta_4
        struct[0].Fx[21,22] = Omega_b_4
        struct[0].Fx[22,21] = (-V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fx[22,22] = -D_4/(2*H_4)
        struct[0].Fx[22,27] = 1/(2*H_4)
        struct[0].Fx[23,23] = -1/T1d0_4
        struct[0].Fx[24,24] = -1/T1q0_4
        struct[0].Fx[25,25] = -1/T_r_4
        struct[0].Fx[27,27] = -1/T_m_4

    if mode == 11:

        struct[0].Fy[0,50] = -Omega_b_1
        struct[0].Fy[1,0] = (-i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,1] = (V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,22] = (-2*R_a_1*i_d_1 - V_1*sin(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,23] = (-2*R_a_1*i_q_1 - V_1*cos(delta_1 - theta_1))/(2*H_1)
        struct[0].Fy[1,50] = D_1/(2*H_1)
        struct[0].Fy[2,22] = (X1d_1 - X_d_1)/T1d0_1
        struct[0].Fy[2,26] = 1/T1d0_1
        struct[0].Fy[3,23] = (-X1q_1 + X_q_1)/T1q0_1
        struct[0].Fy[4,0] = 1/T_r_1
        struct[0].Fy[5,0] = -1
        struct[0].Fy[6,27] = 1/T_m_1
        struct[0].Fy[7,50] = -Omega_b_2
        struct[0].Fy[8,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,28] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,29] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[8,50] = D_2/(2*H_2)
        struct[0].Fy[9,28] = (X1d_2 - X_d_2)/T1d0_2
        struct[0].Fy[9,32] = 1/T1d0_2
        struct[0].Fy[10,29] = (-X1q_2 + X_q_2)/T1q0_2
        struct[0].Fy[11,2] = 1/T_r_2
        struct[0].Fy[12,2] = -1
        struct[0].Fy[13,33] = 1/T_m_2
        struct[0].Fy[14,50] = -Omega_b_3
        struct[0].Fy[15,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,34] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,35] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[15,50] = D_3/(2*H_3)
        struct[0].Fy[16,34] = (X1d_3 - X_d_3)/T1d0_3
        struct[0].Fy[16,38] = 1/T1d0_3
        struct[0].Fy[17,35] = (-X1q_3 + X_q_3)/T1q0_3
        struct[0].Fy[18,4] = 1/T_r_3
        struct[0].Fy[19,4] = -1
        struct[0].Fy[20,39] = 1/T_m_3
        struct[0].Fy[21,50] = -Omega_b_4
        struct[0].Fy[22,6] = (-i_d_4*sin(delta_4 - theta_4) - i_q_4*cos(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,7] = (V_4*i_d_4*cos(delta_4 - theta_4) - V_4*i_q_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,40] = (-2*R_a_4*i_d_4 - V_4*sin(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,41] = (-2*R_a_4*i_q_4 - V_4*cos(delta_4 - theta_4))/(2*H_4)
        struct[0].Fy[22,50] = D_4/(2*H_4)
        struct[0].Fy[23,40] = (X1d_4 - X_d_4)/T1d0_4
        struct[0].Fy[23,44] = 1/T1d0_4
        struct[0].Fy[24,41] = (-X1q_4 + X_q_4)/T1q0_4
        struct[0].Fy[25,6] = 1/T_r_4
        struct[0].Fy[26,6] = -1
        struct[0].Fy[27,45] = 1/T_m_4
        struct[0].Fy[28,50] = -1

        struct[0].Gy[0,0] = 2*V_1*g_1_5 + V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[0,1] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[0,8] = V_1*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[0,9] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[0,24] = -S_n_1/S_base
        struct[0].Gy[1,0] = 2*V_1*(-b_1_5 - bs_1_5/2) + V_5*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[1,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[1,8] = V_1*(b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[1,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[1,25] = -S_n_1/S_base
        struct[0].Gy[2,2] = 2*V_2*g_2_6 + V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[2,3] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[2,10] = V_2*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[2,11] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[2,30] = -S_n_2/S_base
        struct[0].Gy[3,2] = 2*V_2*(-b_2_6 - bs_2_6/2) + V_6*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[3,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[3,10] = V_2*(b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[3,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[3,31] = -S_n_2/S_base
        struct[0].Gy[4,4] = V_11*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3)) + 2*V_3*g_3_11
        struct[0].Gy[4,5] = V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[4,20] = V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[4,21] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[4,36] = -S_n_3/S_base
        struct[0].Gy[5,4] = V_11*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3)) + 2*V_3*(-b_3_11 - bs_3_11/2)
        struct[0].Gy[5,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[5,20] = V_3*(b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[5,21] = V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[5,37] = -S_n_3/S_base
        struct[0].Gy[6,6] = V_10*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + 2*V_4*g_4_10
        struct[0].Gy[6,7] = V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[6,18] = V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[6,19] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[6,42] = -S_n_4/S_base
        struct[0].Gy[7,6] = V_10*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + 2*V_4*(-b_4_10 - bs_4_10/2)
        struct[0].Gy[7,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[7,18] = V_4*(b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[7,19] = V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[7,43] = -S_n_4/S_base
        struct[0].Gy[8,0] = V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[8,1] = V_1*V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[8,8] = V_1*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + 2*V_5*(g_1_5 + g_5_6) + V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[8,9] = V_1*V_5*(-b_1_5*cos(theta_1 - theta_5) - g_1_5*sin(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[8,10] = V_5*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[8,11] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[9,0] = V_5*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5))
        struct[0].Gy[9,1] = V_1*V_5*(-b_1_5*sin(theta_1 - theta_5) + g_1_5*cos(theta_1 - theta_5))
        struct[0].Gy[9,8] = V_1*(b_1_5*cos(theta_1 - theta_5) + g_1_5*sin(theta_1 - theta_5)) + 2*V_5*(-b_1_5 - b_5_6 - bs_1_5/2 - bs_5_6/2) + V_6*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[9,9] = V_1*V_5*(b_1_5*sin(theta_1 - theta_5) - g_1_5*cos(theta_1 - theta_5)) + V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[9,10] = V_5*(b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[9,11] = V_5*V_6*(b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[10,2] = V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[10,3] = V_2*V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[10,8] = V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[10,9] = V_5*V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[10,10] = V_2*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + 2*V_6*(g_2_6 + g_5_6 + g_6_7) + V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[10,11] = V_2*V_6*(-b_2_6*cos(theta_2 - theta_6) - g_2_6*sin(theta_2 - theta_6)) + V_5*V_6*(-b_5_6*cos(theta_5 - theta_6) - g_5_6*sin(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[10,12] = V_6*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[10,13] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[11,2] = V_6*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6))
        struct[0].Gy[11,3] = V_2*V_6*(-b_2_6*sin(theta_2 - theta_6) + g_2_6*cos(theta_2 - theta_6))
        struct[0].Gy[11,8] = V_6*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6))
        struct[0].Gy[11,9] = V_5*V_6*(-b_5_6*sin(theta_5 - theta_6) + g_5_6*cos(theta_5 - theta_6))
        struct[0].Gy[11,10] = V_2*(b_2_6*cos(theta_2 - theta_6) + g_2_6*sin(theta_2 - theta_6)) + V_5*(b_5_6*cos(theta_5 - theta_6) + g_5_6*sin(theta_5 - theta_6)) + 2*V_6*(-b_2_6 - b_5_6 - b_6_7 - bs_2_6/2 - bs_5_6/2 - bs_6_7/2) + V_7*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[11,11] = V_2*V_6*(b_2_6*sin(theta_2 - theta_6) - g_2_6*cos(theta_2 - theta_6)) + V_5*V_6*(b_5_6*sin(theta_5 - theta_6) - g_5_6*cos(theta_5 - theta_6)) + V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[11,12] = V_6*(b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[11,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[12,10] = V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[12,11] = V_6*V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[12,12] = V_6*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + 2*V_7*(g_6_7 + 2*g_7_8) + V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[12,13] = V_6*V_7*(-b_6_7*cos(theta_6 - theta_7) - g_6_7*sin(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[12,14] = V_7*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[12,15] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[13,10] = V_7*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7))
        struct[0].Gy[13,11] = V_6*V_7*(-b_6_7*sin(theta_6 - theta_7) + g_6_7*cos(theta_6 - theta_7))
        struct[0].Gy[13,12] = V_6*(b_6_7*cos(theta_6 - theta_7) + g_6_7*sin(theta_6 - theta_7)) + 2*V_7*(-b_6_7 - 2*b_7_8 - bs_6_7/2 - bs_7_8) + V_8*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[13,13] = V_6*V_7*(b_6_7*sin(theta_6 - theta_7) - g_6_7*cos(theta_6 - theta_7)) + V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[13,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[13,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[14,12] = V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[14,13] = V_7*V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[14,14] = V_7*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + 2*V_8*(2*g_7_8 + 2*g_8_9) + V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[14,15] = V_7*V_8*(-2*b_7_8*cos(theta_7 - theta_8) - 2*g_7_8*sin(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[14,16] = V_8*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[14,17] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[15,12] = V_8*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8))
        struct[0].Gy[15,13] = V_7*V_8*(-2*b_7_8*sin(theta_7 - theta_8) + 2*g_7_8*cos(theta_7 - theta_8))
        struct[0].Gy[15,14] = V_7*(2*b_7_8*cos(theta_7 - theta_8) + 2*g_7_8*sin(theta_7 - theta_8)) + 2*V_8*(-2*b_7_8 - 2*b_8_9 - bs_7_8 - bs_8_9) + V_9*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[15,15] = V_7*V_8*(2*b_7_8*sin(theta_7 - theta_8) - 2*g_7_8*cos(theta_7 - theta_8)) + V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[15,16] = V_8*(2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[15,17] = V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[16,14] = V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[16,15] = V_8*V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[16,16] = V_10*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9)) + 2*V_9*(2*g_8_9 + g_9_10)
        struct[0].Gy[16,17] = V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9)) + V_8*V_9*(-2*b_8_9*cos(theta_8 - theta_9) - 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[16,18] = V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[16,19] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[17,14] = V_9*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9))
        struct[0].Gy[17,15] = V_8*V_9*(-2*b_8_9*sin(theta_8 - theta_9) + 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[17,16] = V_10*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9)) + V_8*(2*b_8_9*cos(theta_8 - theta_9) + 2*g_8_9*sin(theta_8 - theta_9)) + 2*V_9*(-2*b_8_9 - b_9_10 - bs_8_9 - bs_9_10/2)
        struct[0].Gy[17,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9)) + V_8*V_9*(2*b_8_9*sin(theta_8 - theta_9) - 2*g_8_9*cos(theta_8 - theta_9))
        struct[0].Gy[17,18] = V_9*(b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[17,19] = V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[18,6] = V_10*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[18,7] = V_10*V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[18,16] = V_10*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[18,17] = V_10*V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[18,18] = 2*V_10*(g_10_11 + g_4_10 + g_9_10) + V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[18,19] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*cos(theta_10 - theta_4) + g_4_10*sin(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*cos(theta_10 - theta_9) + g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[18,20] = V_10*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[18,21] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[19,6] = V_10*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4))
        struct[0].Gy[19,7] = V_10*V_4*(b_4_10*sin(theta_10 - theta_4) + g_4_10*cos(theta_10 - theta_4))
        struct[0].Gy[19,16] = V_10*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[19,17] = V_10*V_9*(b_9_10*sin(theta_10 - theta_9) + g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[19,18] = 2*V_10*(-b_10_11 - b_4_10 - b_9_10 - bs_10_11/2 - bs_4_10/2 - bs_9_10/2) + V_11*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_4*(b_4_10*cos(theta_10 - theta_4) - g_4_10*sin(theta_10 - theta_4)) + V_9*(b_9_10*cos(theta_10 - theta_9) - g_9_10*sin(theta_10 - theta_9))
        struct[0].Gy[19,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_10*V_4*(-b_4_10*sin(theta_10 - theta_4) - g_4_10*cos(theta_10 - theta_4)) + V_10*V_9*(-b_9_10*sin(theta_10 - theta_9) - g_9_10*cos(theta_10 - theta_9))
        struct[0].Gy[19,20] = V_10*(b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[19,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[20,4] = V_11*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[20,5] = V_11*V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[20,18] = V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[20,19] = V_10*V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[20,20] = V_10*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + 2*V_11*(g_10_11 + g_3_11) + V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[20,21] = V_10*V_11*(-b_10_11*cos(theta_10 - theta_11) - g_10_11*sin(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*cos(theta_11 - theta_3) + g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[21,4] = V_11*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[21,5] = V_11*V_3*(b_3_11*sin(theta_11 - theta_3) + g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[21,18] = V_11*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11))
        struct[0].Gy[21,19] = V_10*V_11*(-b_10_11*sin(theta_10 - theta_11) + g_10_11*cos(theta_10 - theta_11))
        struct[0].Gy[21,20] = V_10*(b_10_11*cos(theta_10 - theta_11) + g_10_11*sin(theta_10 - theta_11)) + 2*V_11*(-b_10_11 - b_3_11 - bs_10_11/2 - bs_3_11/2) + V_3*(b_3_11*cos(theta_11 - theta_3) - g_3_11*sin(theta_11 - theta_3))
        struct[0].Gy[21,21] = V_10*V_11*(b_10_11*sin(theta_10 - theta_11) - g_10_11*cos(theta_10 - theta_11)) + V_11*V_3*(-b_3_11*sin(theta_11 - theta_3) - g_3_11*cos(theta_11 - theta_3))
        struct[0].Gy[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy[22,22] = X1d_1
        struct[0].Gy[22,23] = R_a_1
        struct[0].Gy[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy[23,22] = R_a_1
        struct[0].Gy[23,23] = -X1q_1
        struct[0].Gy[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[24,22] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy[24,23] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy[24,24] = -1
        struct[0].Gy[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[25,22] = V_1*cos(delta_1 - theta_1)
        struct[0].Gy[25,23] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gy[25,25] = -1
        struct[0].Gy[26,26] = -1
        struct[0].Gy[27,27] = -1
        struct[0].Gy[27,46] = 1
        struct[0].Gy[28,2] = cos(delta_2 - theta_2)
        struct[0].Gy[28,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[28,28] = X1d_2
        struct[0].Gy[28,29] = R_a_2
        struct[0].Gy[29,2] = sin(delta_2 - theta_2)
        struct[0].Gy[29,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy[29,28] = R_a_2
        struct[0].Gy[29,29] = -X1q_2
        struct[0].Gy[30,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[30,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[30,28] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[30,29] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[30,30] = -1
        struct[0].Gy[31,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[31,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[31,28] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[31,29] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy[31,31] = -1
        struct[0].Gy[32,32] = -1
        struct[0].Gy[33,33] = -1
        struct[0].Gy[33,47] = 1
        struct[0].Gy[34,4] = cos(delta_3 - theta_3)
        struct[0].Gy[34,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[34,34] = X1d_3
        struct[0].Gy[34,35] = R_a_3
        struct[0].Gy[35,4] = sin(delta_3 - theta_3)
        struct[0].Gy[35,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy[35,34] = R_a_3
        struct[0].Gy[35,35] = -X1q_3
        struct[0].Gy[36,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[36,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[36,34] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[36,35] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[36,36] = -1
        struct[0].Gy[37,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[37,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[37,34] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[37,35] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy[37,37] = -1
        struct[0].Gy[38,38] = -1
        struct[0].Gy[39,39] = -1
        struct[0].Gy[39,48] = 1
        struct[0].Gy[40,6] = cos(delta_4 - theta_4)
        struct[0].Gy[40,7] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy[40,40] = X1d_4
        struct[0].Gy[40,41] = R_a_4
        struct[0].Gy[41,6] = sin(delta_4 - theta_4)
        struct[0].Gy[41,7] = -V_4*cos(delta_4 - theta_4)
        struct[0].Gy[41,40] = R_a_4
        struct[0].Gy[41,41] = -X1q_4
        struct[0].Gy[42,6] = i_d_4*sin(delta_4 - theta_4) + i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy[42,7] = -V_4*i_d_4*cos(delta_4 - theta_4) + V_4*i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy[42,40] = V_4*sin(delta_4 - theta_4)
        struct[0].Gy[42,41] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy[42,42] = -1
        struct[0].Gy[43,6] = i_d_4*cos(delta_4 - theta_4) - i_q_4*sin(delta_4 - theta_4)
        struct[0].Gy[43,7] = V_4*i_d_4*sin(delta_4 - theta_4) + V_4*i_q_4*cos(delta_4 - theta_4)
        struct[0].Gy[43,40] = V_4*cos(delta_4 - theta_4)
        struct[0].Gy[43,41] = -V_4*sin(delta_4 - theta_4)
        struct[0].Gy[43,43] = -1
        struct[0].Gy[44,44] = -1
        struct[0].Gy[45,45] = -1
        struct[0].Gy[45,49] = 1
        struct[0].Gy[46,46] = -1
        struct[0].Gy[47,47] = -1
        struct[0].Gy[48,48] = -1
        struct[0].Gy[49,49] = -1
        struct[0].Gy[50,50] = -1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[6,6] = -1/S_base
        struct[0].Gu[7,7] = -1/S_base
        struct[0].Gu[8,8] = -1/S_base
        struct[0].Gu[9,9] = -1/S_base
        struct[0].Gu[10,10] = -1/S_base
        struct[0].Gu[11,11] = -1/S_base
        struct[0].Gu[12,12] = -1/S_base
        struct[0].Gu[13,13] = -1/S_base
        struct[0].Gu[14,14] = -1/S_base
        struct[0].Gu[15,15] = -1/S_base
        struct[0].Gu[16,16] = -1/S_base
        struct[0].Gu[17,17] = -1/S_base
        struct[0].Gu[18,18] = -1/S_base
        struct[0].Gu[19,19] = -1/S_base
        struct[0].Gu[20,20] = -1/S_base
        struct[0].Gu[21,21] = -1/S_base
        struct[0].Gu[26,22] = K_a_1
        struct[0].Gu[26,23] = K_a_1
        struct[0].Gu[27,24] = 1
        struct[0].Gu[32,25] = K_a_2
        struct[0].Gu[32,26] = K_a_2
        struct[0].Gu[33,27] = 1
        struct[0].Gu[38,28] = K_a_3
        struct[0].Gu[38,29] = K_a_3
        struct[0].Gu[39,30] = 1
        struct[0].Gu[44,31] = K_a_4
        struct[0].Gu[44,32] = K_a_4
        struct[0].Gu[45,33] = 1





@numba.njit(cache=True)
def Piecewise(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out

@numba.njit(cache=True)
def ITE(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out


@numba.njit(cache=True)
def Abs(x):
    return np.abs(x)


@numba.njit(cache=True)
def ini_dae_jacobian_numba(struct,x):
    N_x = struct[0].N_x
    N_y = struct[0].N_y
    struct[0].x[:,0] = x[0:N_x]
    struct[0].y_ini[:,0] = x[N_x:(N_x+N_y)]

    ini(struct,10)
    ini(struct,11) 

    for row,col in zip(struct[0].Fx_ini_rows,struct[0].Fx_ini_cols):
        struct[0].Ac_ini[row,col] = struct[0].Fx_ini[row,col]
    for row,col in zip(struct[0].Fy_ini_rows,struct[0].Fy_ini_cols):
        struct[0].Ac_ini[row,col+N_x] = struct[0].Fy_ini[row,col]
    for row,col in zip(struct[0].Gx_ini_rows,struct[0].Gx_ini_cols):
        struct[0].Ac_ini[row+N_x,col] = struct[0].Gx_ini[row,col]
    for row,col in zip(struct[0].Gy_ini_rows,struct[0].Gy_ini_cols):
        struct[0].Ac_ini[row+N_x,col+N_x] = struct[0].Gy_ini[row,col]


@numba.njit(cache=True)
def ini_dae_problem(struct,x):
    N_x = struct[0].N_x
    N_y = struct[0].N_y
    struct[0].x[:,0] = x[0:N_x]
    struct[0].y_ini[:,0] = x[N_x:(N_x+N_y)]

    ini(struct,2)
    ini(struct,3) 
    struct[0].fg[:N_x,:] = struct[0].f[:]
    struct[0].fg[N_x:,:] = struct[0].g[:]    

@numba.njit(cache=True)
def ssate(struct,xy):
    for it in range(100):
        ini_dae_jacobian_numba(struct,xy[:,0])
        ini_dae_problem(struct,xy[:,0])
        xy[:] += np.linalg.solve(struct[0].Ac_ini,-struct[0].fg)
        if np.max(np.abs(struct[0].fg[:,0]))<1e-8: break
    N_x = struct[0].N_x
    struct[0].x[:,0] = xy[:N_x,0]
    struct[0].y_ini[:,0] = xy[N_x:,0]
    return xy,it


@numba.njit(cache=True) 
def daesolver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 
    
    Dt = struct[i].Dt 

    N_x = struct[i].N_x
    N_y = struct[i].N_y
    N_z = struct[i].N_z

    decimation = struct[i].decimation 
    eye = np.eye(N_x)
    t = struct[i].t 
    t_end = struct[i].t_end 
    if struct[i].it == 0:
        run(t,struct, 1) 
        struct[i].it_store = 0  
        struct[i]['T'][0] = t 
        struct[i].X[0,:] = struct[i].x[:,0]  
        struct[i].Y[0,:] = struct[i].y_run[:,0]  
        struct[i].Z[0,:] = struct[i].h[:,0]  

    solver = struct[i].solvern 
    while t<t_end: 
        struct[i].it += 1
        struct[i].t += Dt
        
        t = struct[i].t


            
        if solver == 5: # Teapezoidal DAE as in Milano's book

            run(t,struct, 2) 
            run(t,struct, 3) 

            x = np.copy(struct[i].x[:]) 
            y = np.copy(struct[i].y_run[:]) 
            f = np.copy(struct[i].f[:]) 
            g = np.copy(struct[i].g[:]) 
            
            for iter in range(struct[i].imax):
                run(t,struct, 2) 
                run(t,struct, 3) 
                run(t,struct,10) 
                run(t,struct,11) 
                
                x_i = struct[i].x[:] 
                y_i = struct[i].y_run[:]  
                f_i = struct[i].f[:] 
                g_i = struct[i].g[:]                 
                F_x_i = struct[i].Fx[:,:]
                F_y_i = struct[i].Fy[:,:] 
                G_x_i = struct[i].Gx[:,:] 
                G_y_i = struct[i].Gy[:,:]                

                A_c_i = np.vstack((np.hstack((eye-0.5*Dt*F_x_i, -0.5*Dt*F_y_i)),
                                   np.hstack((G_x_i,         G_y_i))))
                     
                f_n_i = x_i - x - 0.5*Dt*(f_i+f) 
                # print(t,iter,g_i)
                Dxy_i = np.linalg.solve(-A_c_i,np.vstack((f_n_i,g_i))) 
                
                x_i = x_i + Dxy_i[0:N_x]
                y_i = y_i + Dxy_i[N_x:(N_x+N_y)]

                struct[i].x[:] = x_i
                struct[i].y_run[:] = y_i

        # [f_i,g_i,F_x_i,F_y_i,G_x_i,G_y_i] =  smib_transient(x_i,y_i,u);
        
        # A_c_i = [[eye(N_x)-0.5*Dt*F_x_i, -0.5*Dt*F_y_i],
        #          [                G_x_i,         G_y_i]];
             
        # f_n_i = x_i - x - 0.5*Dt*(f_i+f);
        
        # Dxy_i = -A_c_i\[f_n_i.',g_i.'].';
        
        # x_i = x_i + Dxy_i(1:N_x);
        # y_i = y_i + Dxy_i(N_x+1:N_x+N_y);
                
                xy = np.vstack((x_i,y_i))
                max_relative = 0.0
                for it_var in range(N_x+N_y):
                    abs_value = np.abs(xy[it_var,0])
                    if abs_value < 0.001:
                        abs_value = 0.001
                                             
                    relative_error = np.abs(Dxy_i[it_var,0])/abs_value
                    
                    if relative_error > max_relative: max_relative = relative_error
                    
                if max_relative<struct[i].itol:
                    
                    break
                
                # if iter>struct[i].imax-2:
                    
                #     print('Convergence problem')

            struct[i].x[:] = x_i
            struct[i].y_run[:] = y_i
                
        # channels 
        if struct[i].store == 1:
            it_store = struct[i].it_store
            if struct[i].it >= it_store*decimation: 
                struct[i]['T'][it_store+1] = t 
                struct[i].X[it_store+1,:] = struct[i].x[:,0] 
                struct[i].Y[it_store+1,:] = struct[i].y_run[:,0]
                struct[i].Z[it_store+1,:] = struct[i].h[:,0]
                struct[i].iters[it_store+1,0] = iter
                struct[i].it_store += 1 
            
    struct[i].t = t

    return t





def nonzeros():
    Fx_ini_rows = [0, 0, 1, 1, 1, 2, 3, 4, 6, 7, 7, 8, 8, 8, 9, 10, 11, 13, 14, 14, 15, 15, 15, 16, 17, 18, 20, 21, 21, 22, 22, 22, 23, 24, 25, 27]

    Fx_ini_cols = [0, 1, 0, 1, 6, 2, 3, 4, 6, 7, 8, 7, 8, 13, 9, 10, 11, 13, 14, 15, 14, 15, 20, 16, 17, 18, 20, 21, 22, 21, 22, 27, 23, 24, 25, 27]

    Fy_ini_rows = [0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 16, 16, 17, 18, 19, 20, 21, 22, 22, 22, 22, 22, 23, 23, 24, 25, 26, 27, 28]

    Fy_ini_cols = [50, 0, 1, 22, 23, 50, 22, 26, 23, 0, 0, 27, 50, 2, 3, 28, 29, 50, 28, 32, 29, 2, 2, 33, 50, 4, 5, 34, 35, 50, 34, 38, 35, 4, 4, 39, 50, 6, 7, 40, 41, 50, 40, 44, 41, 6, 6, 45, 50]

    Gx_ini_rows = [22, 22, 23, 23, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 37, 38, 38, 39, 40, 40, 41, 41, 42, 43, 44, 44, 45, 46, 47, 48, 49, 50, 50, 50, 50]

    Gx_ini_cols = [0, 2, 0, 3, 0, 0, 4, 5, 1, 7, 9, 7, 10, 7, 7, 11, 12, 8, 14, 16, 14, 17, 14, 14, 18, 19, 15, 21, 23, 21, 24, 21, 21, 25, 26, 22, 28, 28, 28, 28, 1, 8, 15, 22]

    Gy_ini_rows = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 44, 45, 45, 46, 47, 48, 49, 50]

    Gy_ini_cols = [0, 1, 8, 9, 24, 0, 1, 8, 9, 25, 2, 3, 10, 11, 30, 2, 3, 10, 11, 31, 4, 5, 20, 21, 36, 4, 5, 20, 21, 37, 6, 7, 18, 19, 42, 6, 7, 18, 19, 43, 0, 1, 8, 9, 10, 11, 0, 1, 8, 9, 10, 11, 2, 3, 8, 9, 10, 11, 12, 13, 2, 3, 8, 9, 10, 11, 12, 13, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 14, 15, 16, 17, 18, 19, 14, 15, 16, 17, 18, 19, 6, 7, 16, 17, 18, 19, 20, 21, 6, 7, 16, 17, 18, 19, 20, 21, 4, 5, 18, 19, 20, 21, 4, 5, 18, 19, 20, 21, 0, 1, 22, 23, 0, 1, 22, 23, 0, 1, 22, 23, 24, 0, 1, 22, 23, 25, 26, 27, 46, 2, 3, 28, 29, 2, 3, 28, 29, 2, 3, 28, 29, 30, 2, 3, 28, 29, 31, 32, 33, 47, 4, 5, 34, 35, 4, 5, 34, 35, 4, 5, 34, 35, 36, 4, 5, 34, 35, 37, 38, 39, 48, 6, 7, 40, 41, 6, 7, 40, 41, 6, 7, 40, 41, 42, 6, 7, 40, 41, 43, 44, 45, 49, 46, 47, 48, 49, 50]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols
