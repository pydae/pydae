import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class sys8bus4genpss_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 41
        self.N_y = 57 
        self.N_z = 12 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_1_2', 'b_1_2', 'bs_1_2', 'g_3_4', 'b_3_4', 'bs_3_4', 'g_1_3', 'b_1_3', 'bs_1_3', 'g_2_4', 'b_2_4', 'bs_2_4', 'g_1_4', 'b_1_4', 'bs_1_4', 'g_3_2', 'b_3_2', 'bs_3_2', 'g_5_1', 'b_5_1', 'bs_5_1', 'g_6_2', 'b_6_2', 'bs_6_2', 'g_7_3', 'b_7_3', 'bs_7_3', 'g_8_4', 'b_8_4', 'bs_8_4', 'U_1_n', 'U_2_n', 'U_3_n', 'U_4_n', 'U_5_n', 'U_6_n', 'U_7_n', 'U_8_n', 'S_n_5', 'H_5', 'Omega_b_5', 'T1d0_5', 'T1q0_5', 'X_d_5', 'X_q_5', 'X1d_5', 'X1q_5', 'D_5', 'R_a_5', 'K_delta_5', 'K_a_5', 'K_ai_5', 'T_r_5', 'Droop_5', 'T_gov_1_5', 'T_gov_2_5', 'T_gov_3_5', 'omega_ref_5', 'T_wo_5', 'T_1_5', 'T_2_5', 'K_stab_5', 'S_n_6', 'H_6', 'Omega_b_6', 'T1d0_6', 'T1q0_6', 'X_d_6', 'X_q_6', 'X1d_6', 'X1q_6', 'D_6', 'R_a_6', 'K_delta_6', 'K_a_6', 'K_ai_6', 'T_r_6', 'Droop_6', 'T_gov_1_6', 'T_gov_2_6', 'T_gov_3_6', 'omega_ref_6', 'T_wo_6', 'T_1_6', 'T_2_6', 'K_stab_6', 'S_n_7', 'H_7', 'Omega_b_7', 'T1d0_7', 'T1q0_7', 'X_d_7', 'X_q_7', 'X1d_7', 'X1q_7', 'D_7', 'R_a_7', 'K_delta_7', 'K_a_7', 'K_ai_7', 'T_r_7', 'Droop_7', 'T_gov_1_7', 'T_gov_2_7', 'T_gov_3_7', 'omega_ref_7', 'T_wo_7', 'T_1_7', 'T_2_7', 'K_stab_7', 'S_n_8', 'H_8', 'Omega_b_8', 'T1d0_8', 'T1q0_8', 'X_d_8', 'X_q_8', 'X1d_8', 'X1q_8', 'D_8', 'R_a_8', 'K_delta_8', 'K_a_8', 'K_ai_8', 'T_r_8', 'Droop_8', 'T_gov_1_8', 'T_gov_2_8', 'T_gov_3_8', 'omega_ref_8', 'T_wo_8', 'T_1_8', 'T_2_8', 'K_stab_8', 'K_sec_5', 'K_sec_6', 'K_sec_7', 'K_sec_8'] 
        self.params_values_list  = [100000000.0, 11.978513541335232, -119.78513541335231, 0.08399999999999999, 11.978513541335232, -119.78513541335231, 0.08399999999999999, 11.978513541335232, -119.78513541335231, 0.08399999999999999, 11.978513541335232, -119.78513541335231, 0.08399999999999999, 11.978513541335232, -119.78513541335231, 0.08399999999999999, 29.946283853338073, -299.4628385333807, 0.0336, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 400000.0, 400000.0, 400000.0, 400000.0, 20000.0, 20000.0, 20000.0, 20000.0, 900000000.0, 6.5, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.01, 100, 1e-06, 0.02, 0.05, 1.0, 2.0, 10.0, 1.0, 10.0, 0.1, 0.1, 0.0, 900000000.0, 6.5, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.01, 100, 1e-06, 0.02, 0.05, 1.0, 2.0, 10.0, 1.0, 10.0, 0.1, 0.1, 0.0, 900000000.0, 6.175, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.01, 100, 1e-06, 0.02, 0.05, 1.0, 2.0, 10.0, 1.0, 10.0, 0.1, 0.1, 0.0, 900000000.0, 6.175, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.01, 100, 1e-06, 0.02, 0.05, 1.0, 2.0, 10.0, 1.0, 10.0, 0.1, 0.1, 0.0, 0.001, 0.001, 0.001, 0.001] 
        self.inputs_ini_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'P_4', 'Q_4', 'P_5', 'Q_5', 'P_6', 'Q_6', 'P_7', 'Q_7', 'P_8', 'Q_8', 'v_ref_5', 'v_pss_5', 'p_c_5', 'v_ref_6', 'v_pss_6', 'p_c_6', 'v_ref_7', 'v_pss_7', 'p_c_7', 'v_ref_8', 'v_pss_8', 'p_c_8'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.03, 0.0, 0.7793, 1.01, 0.0, 0.7793, 1.03, 0.0, 0.7793, 1.01, 0.0, 0.7793] 
        self.inputs_run_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'P_4', 'Q_4', 'P_5', 'Q_5', 'P_6', 'Q_6', 'P_7', 'Q_7', 'P_8', 'Q_8', 'v_ref_5', 'v_pss_5', 'p_c_5', 'v_ref_6', 'v_pss_6', 'p_c_6', 'v_ref_7', 'v_pss_7', 'p_c_7', 'v_ref_8', 'v_pss_8', 'p_c_8'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.03, 0.0, 0.7793, 1.01, 0.0, 0.7793, 1.03, 0.0, 0.7793, 1.01, 0.0, 0.7793] 
        self.outputs_list = ['V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'p_e_5', 'p_e_6', 'p_e_7', 'p_e_8'] 
        self.x_list = ['delta_5', 'omega_5', 'e1q_5', 'e1d_5', 'v_c_5', 'xi_v_5', 'x_gov_1_5', 'x_gov_2_5', 'x_wo_5', 'x_lead_5', 'delta_6', 'omega_6', 'e1q_6', 'e1d_6', 'v_c_6', 'xi_v_6', 'x_gov_1_6', 'x_gov_2_6', 'x_wo_6', 'x_lead_6', 'delta_7', 'omega_7', 'e1q_7', 'e1d_7', 'v_c_7', 'xi_v_7', 'x_gov_1_7', 'x_gov_2_7', 'x_wo_7', 'x_lead_7', 'delta_8', 'omega_8', 'e1q_8', 'e1d_8', 'v_c_8', 'xi_v_8', 'x_gov_1_8', 'x_gov_2_8', 'x_wo_8', 'x_lead_8', 'xi_freq'] 
        self.y_run_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'V_4', 'theta_4', 'V_5', 'theta_5', 'V_6', 'theta_6', 'V_7', 'theta_7', 'V_8', 'theta_8', 'i_d_5', 'i_q_5', 'p_g_5_1', 'q_g_5_1', 'v_f_5', 'p_m_ref_5', 'p_m_5', 'z_wo_5', 'v_pss_5', 'i_d_6', 'i_q_6', 'p_g_6_1', 'q_g_6_1', 'v_f_6', 'p_m_ref_6', 'p_m_6', 'z_wo_6', 'v_pss_6', 'i_d_7', 'i_q_7', 'p_g_7_1', 'q_g_7_1', 'v_f_7', 'p_m_ref_7', 'p_m_7', 'z_wo_7', 'v_pss_7', 'i_d_8', 'i_q_8', 'p_g_8_1', 'q_g_8_1', 'v_f_8', 'p_m_ref_8', 'p_m_8', 'z_wo_8', 'v_pss_8', 'p_r_5', 'p_r_6', 'p_r_7', 'p_r_8', 'omega_coi'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'V_4', 'theta_4', 'V_5', 'theta_5', 'V_6', 'theta_6', 'V_7', 'theta_7', 'V_8', 'theta_8', 'i_d_5', 'i_q_5', 'p_g_5_1', 'q_g_5_1', 'v_f_5', 'p_m_ref_5', 'p_m_5', 'z_wo_5', 'v_pss_5', 'i_d_6', 'i_q_6', 'p_g_6_1', 'q_g_6_1', 'v_f_6', 'p_m_ref_6', 'p_m_6', 'z_wo_6', 'v_pss_6', 'i_d_7', 'i_q_7', 'p_g_7_1', 'q_g_7_1', 'v_f_7', 'p_m_ref_7', 'p_m_7', 'z_wo_7', 'v_pss_7', 'i_d_8', 'i_q_8', 'p_g_8_1', 'q_g_8_1', 'v_f_8', 'p_m_ref_8', 'p_m_8', 'z_wo_8', 'v_pss_8', 'p_r_5', 'p_r_6', 'p_r_7', 'p_r_8', 'omega_coi'] 
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
            if item in self.params_list:
                self.params_values_list[self.params_list.index(item)] = self.data[item]
            elif item in self.inputs_ini_list:
                self.inputs_ini_values_list[self.inputs_ini_list.index(item)] = self.data[item]
            elif item in self.inputs_run_list:
                self.inputs_run_values_list[self.inputs_run_list.index(item)] = self.data[item]
            else: 
                print(f'parameter or input {item} not found')



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
    g_1_2 = struct[0].g_1_2
    b_1_2 = struct[0].b_1_2
    bs_1_2 = struct[0].bs_1_2
    g_3_4 = struct[0].g_3_4
    b_3_4 = struct[0].b_3_4
    bs_3_4 = struct[0].bs_3_4
    g_1_3 = struct[0].g_1_3
    b_1_3 = struct[0].b_1_3
    bs_1_3 = struct[0].bs_1_3
    g_2_4 = struct[0].g_2_4
    b_2_4 = struct[0].b_2_4
    bs_2_4 = struct[0].bs_2_4
    g_1_4 = struct[0].g_1_4
    b_1_4 = struct[0].b_1_4
    bs_1_4 = struct[0].bs_1_4
    g_3_2 = struct[0].g_3_2
    b_3_2 = struct[0].b_3_2
    bs_3_2 = struct[0].bs_3_2
    g_5_1 = struct[0].g_5_1
    b_5_1 = struct[0].b_5_1
    bs_5_1 = struct[0].bs_5_1
    g_6_2 = struct[0].g_6_2
    b_6_2 = struct[0].b_6_2
    bs_6_2 = struct[0].bs_6_2
    g_7_3 = struct[0].g_7_3
    b_7_3 = struct[0].b_7_3
    bs_7_3 = struct[0].bs_7_3
    g_8_4 = struct[0].g_8_4
    b_8_4 = struct[0].b_8_4
    bs_8_4 = struct[0].bs_8_4
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    S_n_5 = struct[0].S_n_5
    H_5 = struct[0].H_5
    Omega_b_5 = struct[0].Omega_b_5
    T1d0_5 = struct[0].T1d0_5
    T1q0_5 = struct[0].T1q0_5
    X_d_5 = struct[0].X_d_5
    X_q_5 = struct[0].X_q_5
    X1d_5 = struct[0].X1d_5
    X1q_5 = struct[0].X1q_5
    D_5 = struct[0].D_5
    R_a_5 = struct[0].R_a_5
    K_delta_5 = struct[0].K_delta_5
    K_a_5 = struct[0].K_a_5
    K_ai_5 = struct[0].K_ai_5
    T_r_5 = struct[0].T_r_5
    Droop_5 = struct[0].Droop_5
    T_gov_1_5 = struct[0].T_gov_1_5
    T_gov_2_5 = struct[0].T_gov_2_5
    T_gov_3_5 = struct[0].T_gov_3_5
    omega_ref_5 = struct[0].omega_ref_5
    T_wo_5 = struct[0].T_wo_5
    T_1_5 = struct[0].T_1_5
    T_2_5 = struct[0].T_2_5
    K_stab_5 = struct[0].K_stab_5
    S_n_6 = struct[0].S_n_6
    H_6 = struct[0].H_6
    Omega_b_6 = struct[0].Omega_b_6
    T1d0_6 = struct[0].T1d0_6
    T1q0_6 = struct[0].T1q0_6
    X_d_6 = struct[0].X_d_6
    X_q_6 = struct[0].X_q_6
    X1d_6 = struct[0].X1d_6
    X1q_6 = struct[0].X1q_6
    D_6 = struct[0].D_6
    R_a_6 = struct[0].R_a_6
    K_delta_6 = struct[0].K_delta_6
    K_a_6 = struct[0].K_a_6
    K_ai_6 = struct[0].K_ai_6
    T_r_6 = struct[0].T_r_6
    Droop_6 = struct[0].Droop_6
    T_gov_1_6 = struct[0].T_gov_1_6
    T_gov_2_6 = struct[0].T_gov_2_6
    T_gov_3_6 = struct[0].T_gov_3_6
    omega_ref_6 = struct[0].omega_ref_6
    T_wo_6 = struct[0].T_wo_6
    T_1_6 = struct[0].T_1_6
    T_2_6 = struct[0].T_2_6
    K_stab_6 = struct[0].K_stab_6
    S_n_7 = struct[0].S_n_7
    H_7 = struct[0].H_7
    Omega_b_7 = struct[0].Omega_b_7
    T1d0_7 = struct[0].T1d0_7
    T1q0_7 = struct[0].T1q0_7
    X_d_7 = struct[0].X_d_7
    X_q_7 = struct[0].X_q_7
    X1d_7 = struct[0].X1d_7
    X1q_7 = struct[0].X1q_7
    D_7 = struct[0].D_7
    R_a_7 = struct[0].R_a_7
    K_delta_7 = struct[0].K_delta_7
    K_a_7 = struct[0].K_a_7
    K_ai_7 = struct[0].K_ai_7
    T_r_7 = struct[0].T_r_7
    Droop_7 = struct[0].Droop_7
    T_gov_1_7 = struct[0].T_gov_1_7
    T_gov_2_7 = struct[0].T_gov_2_7
    T_gov_3_7 = struct[0].T_gov_3_7
    omega_ref_7 = struct[0].omega_ref_7
    T_wo_7 = struct[0].T_wo_7
    T_1_7 = struct[0].T_1_7
    T_2_7 = struct[0].T_2_7
    K_stab_7 = struct[0].K_stab_7
    S_n_8 = struct[0].S_n_8
    H_8 = struct[0].H_8
    Omega_b_8 = struct[0].Omega_b_8
    T1d0_8 = struct[0].T1d0_8
    T1q0_8 = struct[0].T1q0_8
    X_d_8 = struct[0].X_d_8
    X_q_8 = struct[0].X_q_8
    X1d_8 = struct[0].X1d_8
    X1q_8 = struct[0].X1q_8
    D_8 = struct[0].D_8
    R_a_8 = struct[0].R_a_8
    K_delta_8 = struct[0].K_delta_8
    K_a_8 = struct[0].K_a_8
    K_ai_8 = struct[0].K_ai_8
    T_r_8 = struct[0].T_r_8
    Droop_8 = struct[0].Droop_8
    T_gov_1_8 = struct[0].T_gov_1_8
    T_gov_2_8 = struct[0].T_gov_2_8
    T_gov_3_8 = struct[0].T_gov_3_8
    omega_ref_8 = struct[0].omega_ref_8
    T_wo_8 = struct[0].T_wo_8
    T_1_8 = struct[0].T_1_8
    T_2_8 = struct[0].T_2_8
    K_stab_8 = struct[0].K_stab_8
    K_sec_5 = struct[0].K_sec_5
    K_sec_6 = struct[0].K_sec_6
    K_sec_7 = struct[0].K_sec_7
    K_sec_8 = struct[0].K_sec_8
    
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
    v_ref_5 = struct[0].v_ref_5
    v_pss_5 = struct[0].v_pss_5
    p_c_5 = struct[0].p_c_5
    v_ref_6 = struct[0].v_ref_6
    v_pss_6 = struct[0].v_pss_6
    p_c_6 = struct[0].p_c_6
    v_ref_7 = struct[0].v_ref_7
    v_pss_7 = struct[0].v_pss_7
    p_c_7 = struct[0].p_c_7
    v_ref_8 = struct[0].v_ref_8
    v_pss_8 = struct[0].v_pss_8
    p_c_8 = struct[0].p_c_8
    
    # Dynamical states:
    delta_5 = struct[0].x[0,0]
    omega_5 = struct[0].x[1,0]
    e1q_5 = struct[0].x[2,0]
    e1d_5 = struct[0].x[3,0]
    v_c_5 = struct[0].x[4,0]
    xi_v_5 = struct[0].x[5,0]
    x_gov_1_5 = struct[0].x[6,0]
    x_gov_2_5 = struct[0].x[7,0]
    x_wo_5 = struct[0].x[8,0]
    x_lead_5 = struct[0].x[9,0]
    delta_6 = struct[0].x[10,0]
    omega_6 = struct[0].x[11,0]
    e1q_6 = struct[0].x[12,0]
    e1d_6 = struct[0].x[13,0]
    v_c_6 = struct[0].x[14,0]
    xi_v_6 = struct[0].x[15,0]
    x_gov_1_6 = struct[0].x[16,0]
    x_gov_2_6 = struct[0].x[17,0]
    x_wo_6 = struct[0].x[18,0]
    x_lead_6 = struct[0].x[19,0]
    delta_7 = struct[0].x[20,0]
    omega_7 = struct[0].x[21,0]
    e1q_7 = struct[0].x[22,0]
    e1d_7 = struct[0].x[23,0]
    v_c_7 = struct[0].x[24,0]
    xi_v_7 = struct[0].x[25,0]
    x_gov_1_7 = struct[0].x[26,0]
    x_gov_2_7 = struct[0].x[27,0]
    x_wo_7 = struct[0].x[28,0]
    x_lead_7 = struct[0].x[29,0]
    delta_8 = struct[0].x[30,0]
    omega_8 = struct[0].x[31,0]
    e1q_8 = struct[0].x[32,0]
    e1d_8 = struct[0].x[33,0]
    v_c_8 = struct[0].x[34,0]
    xi_v_8 = struct[0].x[35,0]
    x_gov_1_8 = struct[0].x[36,0]
    x_gov_2_8 = struct[0].x[37,0]
    x_wo_8 = struct[0].x[38,0]
    x_lead_8 = struct[0].x[39,0]
    xi_freq = struct[0].x[40,0]
    
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
    i_d_5 = struct[0].y_ini[16,0]
    i_q_5 = struct[0].y_ini[17,0]
    p_g_5_1 = struct[0].y_ini[18,0]
    q_g_5_1 = struct[0].y_ini[19,0]
    v_f_5 = struct[0].y_ini[20,0]
    p_m_ref_5 = struct[0].y_ini[21,0]
    p_m_5 = struct[0].y_ini[22,0]
    z_wo_5 = struct[0].y_ini[23,0]
    v_pss_5 = struct[0].y_ini[24,0]
    i_d_6 = struct[0].y_ini[25,0]
    i_q_6 = struct[0].y_ini[26,0]
    p_g_6_1 = struct[0].y_ini[27,0]
    q_g_6_1 = struct[0].y_ini[28,0]
    v_f_6 = struct[0].y_ini[29,0]
    p_m_ref_6 = struct[0].y_ini[30,0]
    p_m_6 = struct[0].y_ini[31,0]
    z_wo_6 = struct[0].y_ini[32,0]
    v_pss_6 = struct[0].y_ini[33,0]
    i_d_7 = struct[0].y_ini[34,0]
    i_q_7 = struct[0].y_ini[35,0]
    p_g_7_1 = struct[0].y_ini[36,0]
    q_g_7_1 = struct[0].y_ini[37,0]
    v_f_7 = struct[0].y_ini[38,0]
    p_m_ref_7 = struct[0].y_ini[39,0]
    p_m_7 = struct[0].y_ini[40,0]
    z_wo_7 = struct[0].y_ini[41,0]
    v_pss_7 = struct[0].y_ini[42,0]
    i_d_8 = struct[0].y_ini[43,0]
    i_q_8 = struct[0].y_ini[44,0]
    p_g_8_1 = struct[0].y_ini[45,0]
    q_g_8_1 = struct[0].y_ini[46,0]
    v_f_8 = struct[0].y_ini[47,0]
    p_m_ref_8 = struct[0].y_ini[48,0]
    p_m_8 = struct[0].y_ini[49,0]
    z_wo_8 = struct[0].y_ini[50,0]
    v_pss_8 = struct[0].y_ini[51,0]
    p_r_5 = struct[0].y_ini[52,0]
    p_r_6 = struct[0].y_ini[53,0]
    p_r_7 = struct[0].y_ini[54,0]
    p_r_8 = struct[0].y_ini[55,0]
    omega_coi = struct[0].y_ini[56,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_5*delta_5 + Omega_b_5*(omega_5 - omega_coi)
        struct[0].f[1,0] = (-D_5*(omega_5 - omega_coi) - i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) - i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5)) + p_m_5)/(2*H_5)
        struct[0].f[2,0] = (-e1q_5 - i_d_5*(-X1d_5 + X_d_5) + v_f_5)/T1d0_5
        struct[0].f[3,0] = (-e1d_5 + i_q_5*(-X1q_5 + X_q_5))/T1q0_5
        struct[0].f[4,0] = (V_5 - v_c_5)/T_r_5
        struct[0].f[5,0] = -V_5 + v_ref_5
        struct[0].f[6,0] = (p_m_ref_5 - x_gov_1_5)/T_gov_1_5
        struct[0].f[7,0] = (x_gov_1_5 - x_gov_2_5)/T_gov_3_5
        struct[0].f[8,0] = (omega_5 - x_wo_5 - 1.0)/T_wo_5
        struct[0].f[9,0] = (-x_lead_5 + z_wo_5)/T_2_5
        struct[0].f[10,0] = -K_delta_6*delta_6 + Omega_b_6*(omega_6 - omega_coi)
        struct[0].f[11,0] = (-D_6*(omega_6 - omega_coi) - i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) - i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6)) + p_m_6)/(2*H_6)
        struct[0].f[12,0] = (-e1q_6 - i_d_6*(-X1d_6 + X_d_6) + v_f_6)/T1d0_6
        struct[0].f[13,0] = (-e1d_6 + i_q_6*(-X1q_6 + X_q_6))/T1q0_6
        struct[0].f[14,0] = (V_6 - v_c_6)/T_r_6
        struct[0].f[15,0] = -V_6 + v_ref_6
        struct[0].f[16,0] = (p_m_ref_6 - x_gov_1_6)/T_gov_1_6
        struct[0].f[17,0] = (x_gov_1_6 - x_gov_2_6)/T_gov_3_6
        struct[0].f[18,0] = (omega_6 - x_wo_6 - 1.0)/T_wo_6
        struct[0].f[19,0] = (-x_lead_6 + z_wo_6)/T_2_6
        struct[0].f[20,0] = -K_delta_7*delta_7 + Omega_b_7*(omega_7 - omega_coi)
        struct[0].f[21,0] = (-D_7*(omega_7 - omega_coi) - i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) - i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7)) + p_m_7)/(2*H_7)
        struct[0].f[22,0] = (-e1q_7 - i_d_7*(-X1d_7 + X_d_7) + v_f_7)/T1d0_7
        struct[0].f[23,0] = (-e1d_7 + i_q_7*(-X1q_7 + X_q_7))/T1q0_7
        struct[0].f[24,0] = (V_7 - v_c_7)/T_r_7
        struct[0].f[25,0] = -V_7 + v_ref_7
        struct[0].f[26,0] = (p_m_ref_7 - x_gov_1_7)/T_gov_1_7
        struct[0].f[27,0] = (x_gov_1_7 - x_gov_2_7)/T_gov_3_7
        struct[0].f[28,0] = (omega_7 - x_wo_7 - 1.0)/T_wo_7
        struct[0].f[29,0] = (-x_lead_7 + z_wo_7)/T_2_7
        struct[0].f[30,0] = -K_delta_8*delta_8 + Omega_b_8*(omega_8 - omega_coi)
        struct[0].f[31,0] = (-D_8*(omega_8 - omega_coi) - i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) - i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8)) + p_m_8)/(2*H_8)
        struct[0].f[32,0] = (-e1q_8 - i_d_8*(-X1d_8 + X_d_8) + v_f_8)/T1d0_8
        struct[0].f[33,0] = (-e1d_8 + i_q_8*(-X1q_8 + X_q_8))/T1q0_8
        struct[0].f[34,0] = (V_8 - v_c_8)/T_r_8
        struct[0].f[35,0] = -V_8 + v_ref_8
        struct[0].f[36,0] = (p_m_ref_8 - x_gov_1_8)/T_gov_1_8
        struct[0].f[37,0] = (x_gov_1_8 - x_gov_2_8)/T_gov_3_8
        struct[0].f[38,0] = (omega_8 - x_wo_8 - 1.0)/T_wo_8
        struct[0].f[39,0] = (-x_lead_8 + z_wo_8)/T_2_8
        struct[0].f[40,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_1/S_base + V_1**2*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].g[4,0] = -P_3/S_base + V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3**2*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].g[5,0] = -Q_3/S_base + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_3**2*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].g[6,0] = -P_4/S_base + V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4**2*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].g[7,0] = -Q_4/S_base + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_4**2*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + V_5**2*g_5_1 - S_n_5*p_g_5_1/S_base
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + V_5**2*(-b_5_1 - bs_5_1/2) - S_n_5*q_g_5_1/S_base
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + V_6**2*g_6_2 - S_n_6*p_g_6_1/S_base
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + V_6**2*(-b_6_2 - bs_6_2/2) - S_n_6*q_g_6_1/S_base
        struct[0].g[12,0] = -P_7/S_base + V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + V_7**2*g_7_3 - S_n_7*p_g_7_1/S_base
        struct[0].g[13,0] = -Q_7/S_base + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + V_7**2*(-b_7_3 - bs_7_3/2) - S_n_7*q_g_7_1/S_base
        struct[0].g[14,0] = -P_8/S_base + V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + V_8**2*g_8_4 - S_n_8*p_g_8_1/S_base
        struct[0].g[15,0] = -Q_8/S_base + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + V_8**2*(-b_8_4 - bs_8_4/2) - S_n_8*q_g_8_1/S_base
        struct[0].g[16,0] = R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5) + X1d_5*i_d_5 - e1q_5
        struct[0].g[17,0] = R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5) - X1q_5*i_q_5 - e1d_5
        struct[0].g[18,0] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5) - p_g_5_1
        struct[0].g[19,0] = V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5) - q_g_5_1
        struct[0].g[20,0] = K_a_5*(-v_c_5 + v_pss_5 + v_ref_5) + K_ai_5*xi_v_5 - v_f_5
        struct[0].g[21,0] = p_c_5 - p_m_ref_5 + p_r_5 - (omega_5 - omega_ref_5)/Droop_5
        struct[0].g[22,0] = T_gov_2_5*(x_gov_1_5 - x_gov_2_5)/T_gov_3_5 - p_m_5 + x_gov_2_5
        struct[0].g[23,0] = omega_5 - x_wo_5 - z_wo_5 - 1.0
        struct[0].g[24,0] = K_stab_5*(T_1_5*(-x_lead_5 + z_wo_5)/T_2_5 + x_lead_5) - v_pss_5
        struct[0].g[25,0] = R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6) + X1d_6*i_d_6 - e1q_6
        struct[0].g[26,0] = R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6) - X1q_6*i_q_6 - e1d_6
        struct[0].g[27,0] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6) - p_g_6_1
        struct[0].g[28,0] = V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6) - q_g_6_1
        struct[0].g[29,0] = K_a_6*(-v_c_6 + v_pss_6 + v_ref_6) + K_ai_6*xi_v_6 - v_f_6
        struct[0].g[30,0] = p_c_6 - p_m_ref_6 + p_r_6 - (omega_6 - omega_ref_6)/Droop_6
        struct[0].g[31,0] = T_gov_2_6*(x_gov_1_6 - x_gov_2_6)/T_gov_3_6 - p_m_6 + x_gov_2_6
        struct[0].g[32,0] = omega_6 - x_wo_6 - z_wo_6 - 1.0
        struct[0].g[33,0] = K_stab_6*(T_1_6*(-x_lead_6 + z_wo_6)/T_2_6 + x_lead_6) - v_pss_6
        struct[0].g[34,0] = R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7) + X1d_7*i_d_7 - e1q_7
        struct[0].g[35,0] = R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7) - X1q_7*i_q_7 - e1d_7
        struct[0].g[36,0] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7) - p_g_7_1
        struct[0].g[37,0] = V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7) - q_g_7_1
        struct[0].g[38,0] = K_a_7*(-v_c_7 + v_pss_7 + v_ref_7) + K_ai_7*xi_v_7 - v_f_7
        struct[0].g[39,0] = p_c_7 - p_m_ref_7 + p_r_7 - (omega_7 - omega_ref_7)/Droop_7
        struct[0].g[40,0] = T_gov_2_7*(x_gov_1_7 - x_gov_2_7)/T_gov_3_7 - p_m_7 + x_gov_2_7
        struct[0].g[41,0] = omega_7 - x_wo_7 - z_wo_7 - 1.0
        struct[0].g[42,0] = K_stab_7*(T_1_7*(-x_lead_7 + z_wo_7)/T_2_7 + x_lead_7) - v_pss_7
        struct[0].g[43,0] = R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8) + X1d_8*i_d_8 - e1q_8
        struct[0].g[44,0] = R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8) - X1q_8*i_q_8 - e1d_8
        struct[0].g[45,0] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8) - p_g_8_1
        struct[0].g[46,0] = V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8) - q_g_8_1
        struct[0].g[47,0] = K_a_8*(-v_c_8 + v_pss_8 + v_ref_8) + K_ai_8*xi_v_8 - v_f_8
        struct[0].g[48,0] = p_c_8 - p_m_ref_8 + p_r_8 - (omega_8 - omega_ref_8)/Droop_8
        struct[0].g[49,0] = T_gov_2_8*(x_gov_1_8 - x_gov_2_8)/T_gov_3_8 - p_m_8 + x_gov_2_8
        struct[0].g[50,0] = omega_8 - x_wo_8 - z_wo_8 - 1.0
        struct[0].g[51,0] = K_stab_8*(T_1_8*(-x_lead_8 + z_wo_8)/T_2_8 + x_lead_8) - v_pss_8
        struct[0].g[52,0] = K_sec_5*xi_freq/4 - p_r_5
        struct[0].g[53,0] = K_sec_6*xi_freq/4 - p_r_6
        struct[0].g[54,0] = K_sec_7*xi_freq/4 - p_r_7
        struct[0].g[55,0] = K_sec_8*xi_freq/4 - p_r_8
        struct[0].g[56,0] = omega_5/4 + omega_6/4 + omega_7/4 + omega_8/4 - omega_coi
    
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
        struct[0].h[8,0] = i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) + i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5))
        struct[0].h[9,0] = i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) + i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6))
        struct[0].h[10,0] = i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) + i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7))
        struct[0].h[11,0] = i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) + i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_5
        struct[0].Fx_ini[0,1] = Omega_b_5
        struct[0].Fx_ini[1,0] = (-V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fx_ini[1,1] = -D_5/(2*H_5)
        struct[0].Fx_ini[2,2] = -1/T1d0_5
        struct[0].Fx_ini[3,3] = -1/T1q0_5
        struct[0].Fx_ini[4,4] = -1/T_r_5
        struct[0].Fx_ini[6,6] = -1/T_gov_1_5
        struct[0].Fx_ini[7,6] = 1/T_gov_3_5
        struct[0].Fx_ini[7,7] = -1/T_gov_3_5
        struct[0].Fx_ini[8,1] = 1/T_wo_5
        struct[0].Fx_ini[8,8] = -1/T_wo_5
        struct[0].Fx_ini[9,9] = -1/T_2_5
        struct[0].Fx_ini[10,10] = -K_delta_6
        struct[0].Fx_ini[10,11] = Omega_b_6
        struct[0].Fx_ini[11,10] = (-V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fx_ini[11,11] = -D_6/(2*H_6)
        struct[0].Fx_ini[12,12] = -1/T1d0_6
        struct[0].Fx_ini[13,13] = -1/T1q0_6
        struct[0].Fx_ini[14,14] = -1/T_r_6
        struct[0].Fx_ini[16,16] = -1/T_gov_1_6
        struct[0].Fx_ini[17,16] = 1/T_gov_3_6
        struct[0].Fx_ini[17,17] = -1/T_gov_3_6
        struct[0].Fx_ini[18,11] = 1/T_wo_6
        struct[0].Fx_ini[18,18] = -1/T_wo_6
        struct[0].Fx_ini[19,19] = -1/T_2_6
        struct[0].Fx_ini[20,20] = -K_delta_7
        struct[0].Fx_ini[20,21] = Omega_b_7
        struct[0].Fx_ini[21,20] = (-V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fx_ini[21,21] = -D_7/(2*H_7)
        struct[0].Fx_ini[22,22] = -1/T1d0_7
        struct[0].Fx_ini[23,23] = -1/T1q0_7
        struct[0].Fx_ini[24,24] = -1/T_r_7
        struct[0].Fx_ini[26,26] = -1/T_gov_1_7
        struct[0].Fx_ini[27,26] = 1/T_gov_3_7
        struct[0].Fx_ini[27,27] = -1/T_gov_3_7
        struct[0].Fx_ini[28,21] = 1/T_wo_7
        struct[0].Fx_ini[28,28] = -1/T_wo_7
        struct[0].Fx_ini[29,29] = -1/T_2_7
        struct[0].Fx_ini[30,30] = -K_delta_8
        struct[0].Fx_ini[30,31] = Omega_b_8
        struct[0].Fx_ini[31,30] = (-V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fx_ini[31,31] = -D_8/(2*H_8)
        struct[0].Fx_ini[32,32] = -1/T1d0_8
        struct[0].Fx_ini[33,33] = -1/T1q0_8
        struct[0].Fx_ini[34,34] = -1/T_r_8
        struct[0].Fx_ini[36,36] = -1/T_gov_1_8
        struct[0].Fx_ini[37,36] = 1/T_gov_3_8
        struct[0].Fx_ini[37,37] = -1/T_gov_3_8
        struct[0].Fx_ini[38,31] = 1/T_wo_8
        struct[0].Fx_ini[38,38] = -1/T_wo_8
        struct[0].Fx_ini[39,39] = -1/T_2_8

    if mode == 11:

        struct[0].Fy_ini[0,56] = -Omega_b_5 
        struct[0].Fy_ini[1,8] = (-i_d_5*sin(delta_5 - theta_5) - i_q_5*cos(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,9] = (V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,16] = (-2*R_a_5*i_d_5 - V_5*sin(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,17] = (-2*R_a_5*i_q_5 - V_5*cos(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,22] = 1/(2*H_5) 
        struct[0].Fy_ini[1,56] = D_5/(2*H_5) 
        struct[0].Fy_ini[2,16] = (X1d_5 - X_d_5)/T1d0_5 
        struct[0].Fy_ini[2,20] = 1/T1d0_5 
        struct[0].Fy_ini[3,17] = (-X1q_5 + X_q_5)/T1q0_5 
        struct[0].Fy_ini[4,8] = 1/T_r_5 
        struct[0].Fy_ini[5,8] = -1 
        struct[0].Fy_ini[6,21] = 1/T_gov_1_5 
        struct[0].Fy_ini[9,23] = 1/T_2_5 
        struct[0].Fy_ini[10,56] = -Omega_b_6 
        struct[0].Fy_ini[11,10] = (-i_d_6*sin(delta_6 - theta_6) - i_q_6*cos(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,11] = (V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,25] = (-2*R_a_6*i_d_6 - V_6*sin(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,26] = (-2*R_a_6*i_q_6 - V_6*cos(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,31] = 1/(2*H_6) 
        struct[0].Fy_ini[11,56] = D_6/(2*H_6) 
        struct[0].Fy_ini[12,25] = (X1d_6 - X_d_6)/T1d0_6 
        struct[0].Fy_ini[12,29] = 1/T1d0_6 
        struct[0].Fy_ini[13,26] = (-X1q_6 + X_q_6)/T1q0_6 
        struct[0].Fy_ini[14,10] = 1/T_r_6 
        struct[0].Fy_ini[15,10] = -1 
        struct[0].Fy_ini[16,30] = 1/T_gov_1_6 
        struct[0].Fy_ini[19,32] = 1/T_2_6 
        struct[0].Fy_ini[20,56] = -Omega_b_7 
        struct[0].Fy_ini[21,12] = (-i_d_7*sin(delta_7 - theta_7) - i_q_7*cos(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,13] = (V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,34] = (-2*R_a_7*i_d_7 - V_7*sin(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,35] = (-2*R_a_7*i_q_7 - V_7*cos(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,40] = 1/(2*H_7) 
        struct[0].Fy_ini[21,56] = D_7/(2*H_7) 
        struct[0].Fy_ini[22,34] = (X1d_7 - X_d_7)/T1d0_7 
        struct[0].Fy_ini[22,38] = 1/T1d0_7 
        struct[0].Fy_ini[23,35] = (-X1q_7 + X_q_7)/T1q0_7 
        struct[0].Fy_ini[24,12] = 1/T_r_7 
        struct[0].Fy_ini[25,12] = -1 
        struct[0].Fy_ini[26,39] = 1/T_gov_1_7 
        struct[0].Fy_ini[29,41] = 1/T_2_7 
        struct[0].Fy_ini[30,56] = -Omega_b_8 
        struct[0].Fy_ini[31,14] = (-i_d_8*sin(delta_8 - theta_8) - i_q_8*cos(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,15] = (V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,43] = (-2*R_a_8*i_d_8 - V_8*sin(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,44] = (-2*R_a_8*i_q_8 - V_8*cos(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,49] = 1/(2*H_8) 
        struct[0].Fy_ini[31,56] = D_8/(2*H_8) 
        struct[0].Fy_ini[32,43] = (X1d_8 - X_d_8)/T1d0_8 
        struct[0].Fy_ini[32,47] = 1/T1d0_8 
        struct[0].Fy_ini[33,44] = (-X1q_8 + X_q_8)/T1q0_8 
        struct[0].Fy_ini[34,14] = 1/T_r_8 
        struct[0].Fy_ini[35,14] = -1 
        struct[0].Fy_ini[36,48] = 1/T_gov_1_8 
        struct[0].Fy_ini[39,50] = 1/T_2_8 
        struct[0].Fy_ini[40,56] = -1 

        struct[0].Gx_ini[16,0] = -V_5*sin(delta_5 - theta_5)
        struct[0].Gx_ini[16,2] = -1
        struct[0].Gx_ini[17,0] = V_5*cos(delta_5 - theta_5)
        struct[0].Gx_ini[17,3] = -1
        struct[0].Gx_ini[18,0] = V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Gx_ini[19,0] = -V_5*i_d_5*sin(delta_5 - theta_5) - V_5*i_q_5*cos(delta_5 - theta_5)
        struct[0].Gx_ini[20,4] = -K_a_5
        struct[0].Gx_ini[20,5] = K_ai_5
        struct[0].Gx_ini[21,1] = -1/Droop_5
        struct[0].Gx_ini[22,6] = T_gov_2_5/T_gov_3_5
        struct[0].Gx_ini[22,7] = -T_gov_2_5/T_gov_3_5 + 1
        struct[0].Gx_ini[23,1] = 1
        struct[0].Gx_ini[23,8] = -1
        struct[0].Gx_ini[24,9] = K_stab_5*(-T_1_5/T_2_5 + 1)
        struct[0].Gx_ini[25,10] = -V_6*sin(delta_6 - theta_6)
        struct[0].Gx_ini[25,12] = -1
        struct[0].Gx_ini[26,10] = V_6*cos(delta_6 - theta_6)
        struct[0].Gx_ini[26,13] = -1
        struct[0].Gx_ini[27,10] = V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Gx_ini[28,10] = -V_6*i_d_6*sin(delta_6 - theta_6) - V_6*i_q_6*cos(delta_6 - theta_6)
        struct[0].Gx_ini[29,14] = -K_a_6
        struct[0].Gx_ini[29,15] = K_ai_6
        struct[0].Gx_ini[30,11] = -1/Droop_6
        struct[0].Gx_ini[31,16] = T_gov_2_6/T_gov_3_6
        struct[0].Gx_ini[31,17] = -T_gov_2_6/T_gov_3_6 + 1
        struct[0].Gx_ini[32,11] = 1
        struct[0].Gx_ini[32,18] = -1
        struct[0].Gx_ini[33,19] = K_stab_6*(-T_1_6/T_2_6 + 1)
        struct[0].Gx_ini[34,20] = -V_7*sin(delta_7 - theta_7)
        struct[0].Gx_ini[34,22] = -1
        struct[0].Gx_ini[35,20] = V_7*cos(delta_7 - theta_7)
        struct[0].Gx_ini[35,23] = -1
        struct[0].Gx_ini[36,20] = V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Gx_ini[37,20] = -V_7*i_d_7*sin(delta_7 - theta_7) - V_7*i_q_7*cos(delta_7 - theta_7)
        struct[0].Gx_ini[38,24] = -K_a_7
        struct[0].Gx_ini[38,25] = K_ai_7
        struct[0].Gx_ini[39,21] = -1/Droop_7
        struct[0].Gx_ini[40,26] = T_gov_2_7/T_gov_3_7
        struct[0].Gx_ini[40,27] = -T_gov_2_7/T_gov_3_7 + 1
        struct[0].Gx_ini[41,21] = 1
        struct[0].Gx_ini[41,28] = -1
        struct[0].Gx_ini[42,29] = K_stab_7*(-T_1_7/T_2_7 + 1)
        struct[0].Gx_ini[43,30] = -V_8*sin(delta_8 - theta_8)
        struct[0].Gx_ini[43,32] = -1
        struct[0].Gx_ini[44,30] = V_8*cos(delta_8 - theta_8)
        struct[0].Gx_ini[44,33] = -1
        struct[0].Gx_ini[45,30] = V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8)
        struct[0].Gx_ini[46,30] = -V_8*i_d_8*sin(delta_8 - theta_8) - V_8*i_q_8*cos(delta_8 - theta_8)
        struct[0].Gx_ini[47,34] = -K_a_8
        struct[0].Gx_ini[47,35] = K_ai_8
        struct[0].Gx_ini[48,31] = -1/Droop_8
        struct[0].Gx_ini[49,36] = T_gov_2_8/T_gov_3_8
        struct[0].Gx_ini[49,37] = -T_gov_2_8/T_gov_3_8 + 1
        struct[0].Gx_ini[50,31] = 1
        struct[0].Gx_ini[50,38] = -1
        struct[0].Gx_ini[51,39] = K_stab_8*(-T_1_8/T_2_8 + 1)
        struct[0].Gx_ini[52,40] = K_sec_5/4
        struct[0].Gx_ini[53,40] = K_sec_6/4
        struct[0].Gx_ini[54,40] = K_sec_7/4
        struct[0].Gx_ini[55,40] = K_sec_8/4
        struct[0].Gx_ini[56,1] = 1/4
        struct[0].Gx_ini[56,11] = 1/4
        struct[0].Gx_ini[56,21] = 1/4
        struct[0].Gx_ini[56,31] = 1/4

        struct[0].Gy_ini[0,0] = 2*V_1*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[0,4] = V_1*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[0,5] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[0,6] = V_1*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[0,7] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[0,8] = V_1*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,9] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,0] = 2*V_1*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[1,4] = V_1*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[1,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[1,6] = V_1*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[1,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[1,8] = V_1*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[2,4] = V_2*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[2,5] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[2,6] = V_2*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[2,7] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[2,10] = V_2*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,11] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[3,4] = V_2*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[3,5] = V_2*V_3*(b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[3,6] = V_2*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[3,7] = V_2*V_4*(b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[3,10] = V_2*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[4,0] = V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[4,1] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[4,2] = V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[4,3] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[4,4] = V_1*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + 2*V_3*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[4,5] = V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[4,6] = V_3*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[4,7] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[4,12] = V_3*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[4,13] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[5,0] = V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[5,1] = V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[5,2] = V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[5,3] = V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[5,4] = V_1*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + 2*V_3*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[5,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[5,6] = V_3*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[5,7] = V_3*V_4*(b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[5,12] = V_3*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[5,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[6,0] = V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[6,1] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[6,2] = V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[6,3] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[6,4] = V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[6,5] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[6,6] = V_1*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + 2*V_4*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[6,7] = V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[6,14] = V_4*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[6,15] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[7,0] = V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[7,1] = V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[7,2] = V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[7,3] = V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[7,4] = V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[7,5] = V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[7,6] = V_1*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + 2*V_4*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[7,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[7,14] = V_4*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[7,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[8,0] = V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[8,1] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[8,8] = V_1*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + 2*V_5*g_5_1
        struct[0].Gy_ini[8,9] = V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[8,18] = -S_n_5/S_base
        struct[0].Gy_ini[9,0] = V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[9,1] = V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[9,8] = V_1*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + 2*V_5*(-b_5_1 - bs_5_1/2)
        struct[0].Gy_ini[9,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[9,19] = -S_n_5/S_base
        struct[0].Gy_ini[10,2] = V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[10,3] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[10,10] = V_2*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + 2*V_6*g_6_2
        struct[0].Gy_ini[10,11] = V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[10,27] = -S_n_6/S_base
        struct[0].Gy_ini[11,2] = V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[11,3] = V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[11,10] = V_2*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + 2*V_6*(-b_6_2 - bs_6_2/2)
        struct[0].Gy_ini[11,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[11,28] = -S_n_6/S_base
        struct[0].Gy_ini[12,4] = V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[12,5] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[12,12] = V_3*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + 2*V_7*g_7_3
        struct[0].Gy_ini[12,13] = V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[12,36] = -S_n_7/S_base
        struct[0].Gy_ini[13,4] = V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[13,5] = V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[13,12] = V_3*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + 2*V_7*(-b_7_3 - bs_7_3/2)
        struct[0].Gy_ini[13,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[13,37] = -S_n_7/S_base
        struct[0].Gy_ini[14,6] = V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[14,7] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[14,14] = V_4*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + 2*V_8*g_8_4
        struct[0].Gy_ini[14,15] = V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[14,45] = -S_n_8/S_base
        struct[0].Gy_ini[15,6] = V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[15,7] = V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[15,14] = V_4*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + 2*V_8*(-b_8_4 - bs_8_4/2)
        struct[0].Gy_ini[15,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[15,46] = -S_n_8/S_base
        struct[0].Gy_ini[16,8] = cos(delta_5 - theta_5)
        struct[0].Gy_ini[16,9] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[16,16] = X1d_5
        struct[0].Gy_ini[16,17] = R_a_5
        struct[0].Gy_ini[17,8] = sin(delta_5 - theta_5)
        struct[0].Gy_ini[17,9] = -V_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[17,16] = R_a_5
        struct[0].Gy_ini[17,17] = -X1q_5
        struct[0].Gy_ini[18,8] = i_d_5*sin(delta_5 - theta_5) + i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[18,9] = -V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[18,16] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[18,17] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[19,8] = i_d_5*cos(delta_5 - theta_5) - i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[19,9] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[19,16] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[19,17] = -V_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[20,24] = K_a_5
        struct[0].Gy_ini[24,23] = K_stab_5*T_1_5/T_2_5
        struct[0].Gy_ini[25,10] = cos(delta_6 - theta_6)
        struct[0].Gy_ini[25,11] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[25,25] = X1d_6
        struct[0].Gy_ini[25,26] = R_a_6
        struct[0].Gy_ini[26,10] = sin(delta_6 - theta_6)
        struct[0].Gy_ini[26,11] = -V_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[26,25] = R_a_6
        struct[0].Gy_ini[26,26] = -X1q_6
        struct[0].Gy_ini[27,10] = i_d_6*sin(delta_6 - theta_6) + i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[27,11] = -V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[27,25] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[27,26] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[28,10] = i_d_6*cos(delta_6 - theta_6) - i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[28,11] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[28,25] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[28,26] = -V_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[29,33] = K_a_6
        struct[0].Gy_ini[33,32] = K_stab_6*T_1_6/T_2_6
        struct[0].Gy_ini[34,12] = cos(delta_7 - theta_7)
        struct[0].Gy_ini[34,13] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[34,34] = X1d_7
        struct[0].Gy_ini[34,35] = R_a_7
        struct[0].Gy_ini[35,12] = sin(delta_7 - theta_7)
        struct[0].Gy_ini[35,13] = -V_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[35,34] = R_a_7
        struct[0].Gy_ini[35,35] = -X1q_7
        struct[0].Gy_ini[36,12] = i_d_7*sin(delta_7 - theta_7) + i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[36,13] = -V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[36,34] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[36,35] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[37,12] = i_d_7*cos(delta_7 - theta_7) - i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[37,13] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[37,34] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[37,35] = -V_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[38,42] = K_a_7
        struct[0].Gy_ini[42,41] = K_stab_7*T_1_7/T_2_7
        struct[0].Gy_ini[43,14] = cos(delta_8 - theta_8)
        struct[0].Gy_ini[43,15] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[43,43] = X1d_8
        struct[0].Gy_ini[43,44] = R_a_8
        struct[0].Gy_ini[44,14] = sin(delta_8 - theta_8)
        struct[0].Gy_ini[44,15] = -V_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[44,43] = R_a_8
        struct[0].Gy_ini[44,44] = -X1q_8
        struct[0].Gy_ini[45,14] = i_d_8*sin(delta_8 - theta_8) + i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[45,15] = -V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[45,43] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[45,44] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[46,14] = i_d_8*cos(delta_8 - theta_8) - i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[46,15] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[46,43] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[46,44] = -V_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[47,51] = K_a_8
        struct[0].Gy_ini[51,50] = K_stab_8*T_1_8/T_2_8



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_2 = struct[0].g_1_2
    b_1_2 = struct[0].b_1_2
    bs_1_2 = struct[0].bs_1_2
    g_3_4 = struct[0].g_3_4
    b_3_4 = struct[0].b_3_4
    bs_3_4 = struct[0].bs_3_4
    g_1_3 = struct[0].g_1_3
    b_1_3 = struct[0].b_1_3
    bs_1_3 = struct[0].bs_1_3
    g_2_4 = struct[0].g_2_4
    b_2_4 = struct[0].b_2_4
    bs_2_4 = struct[0].bs_2_4
    g_1_4 = struct[0].g_1_4
    b_1_4 = struct[0].b_1_4
    bs_1_4 = struct[0].bs_1_4
    g_3_2 = struct[0].g_3_2
    b_3_2 = struct[0].b_3_2
    bs_3_2 = struct[0].bs_3_2
    g_5_1 = struct[0].g_5_1
    b_5_1 = struct[0].b_5_1
    bs_5_1 = struct[0].bs_5_1
    g_6_2 = struct[0].g_6_2
    b_6_2 = struct[0].b_6_2
    bs_6_2 = struct[0].bs_6_2
    g_7_3 = struct[0].g_7_3
    b_7_3 = struct[0].b_7_3
    bs_7_3 = struct[0].bs_7_3
    g_8_4 = struct[0].g_8_4
    b_8_4 = struct[0].b_8_4
    bs_8_4 = struct[0].bs_8_4
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    S_n_5 = struct[0].S_n_5
    H_5 = struct[0].H_5
    Omega_b_5 = struct[0].Omega_b_5
    T1d0_5 = struct[0].T1d0_5
    T1q0_5 = struct[0].T1q0_5
    X_d_5 = struct[0].X_d_5
    X_q_5 = struct[0].X_q_5
    X1d_5 = struct[0].X1d_5
    X1q_5 = struct[0].X1q_5
    D_5 = struct[0].D_5
    R_a_5 = struct[0].R_a_5
    K_delta_5 = struct[0].K_delta_5
    K_a_5 = struct[0].K_a_5
    K_ai_5 = struct[0].K_ai_5
    T_r_5 = struct[0].T_r_5
    Droop_5 = struct[0].Droop_5
    T_gov_1_5 = struct[0].T_gov_1_5
    T_gov_2_5 = struct[0].T_gov_2_5
    T_gov_3_5 = struct[0].T_gov_3_5
    omega_ref_5 = struct[0].omega_ref_5
    T_wo_5 = struct[0].T_wo_5
    T_1_5 = struct[0].T_1_5
    T_2_5 = struct[0].T_2_5
    K_stab_5 = struct[0].K_stab_5
    S_n_6 = struct[0].S_n_6
    H_6 = struct[0].H_6
    Omega_b_6 = struct[0].Omega_b_6
    T1d0_6 = struct[0].T1d0_6
    T1q0_6 = struct[0].T1q0_6
    X_d_6 = struct[0].X_d_6
    X_q_6 = struct[0].X_q_6
    X1d_6 = struct[0].X1d_6
    X1q_6 = struct[0].X1q_6
    D_6 = struct[0].D_6
    R_a_6 = struct[0].R_a_6
    K_delta_6 = struct[0].K_delta_6
    K_a_6 = struct[0].K_a_6
    K_ai_6 = struct[0].K_ai_6
    T_r_6 = struct[0].T_r_6
    Droop_6 = struct[0].Droop_6
    T_gov_1_6 = struct[0].T_gov_1_6
    T_gov_2_6 = struct[0].T_gov_2_6
    T_gov_3_6 = struct[0].T_gov_3_6
    omega_ref_6 = struct[0].omega_ref_6
    T_wo_6 = struct[0].T_wo_6
    T_1_6 = struct[0].T_1_6
    T_2_6 = struct[0].T_2_6
    K_stab_6 = struct[0].K_stab_6
    S_n_7 = struct[0].S_n_7
    H_7 = struct[0].H_7
    Omega_b_7 = struct[0].Omega_b_7
    T1d0_7 = struct[0].T1d0_7
    T1q0_7 = struct[0].T1q0_7
    X_d_7 = struct[0].X_d_7
    X_q_7 = struct[0].X_q_7
    X1d_7 = struct[0].X1d_7
    X1q_7 = struct[0].X1q_7
    D_7 = struct[0].D_7
    R_a_7 = struct[0].R_a_7
    K_delta_7 = struct[0].K_delta_7
    K_a_7 = struct[0].K_a_7
    K_ai_7 = struct[0].K_ai_7
    T_r_7 = struct[0].T_r_7
    Droop_7 = struct[0].Droop_7
    T_gov_1_7 = struct[0].T_gov_1_7
    T_gov_2_7 = struct[0].T_gov_2_7
    T_gov_3_7 = struct[0].T_gov_3_7
    omega_ref_7 = struct[0].omega_ref_7
    T_wo_7 = struct[0].T_wo_7
    T_1_7 = struct[0].T_1_7
    T_2_7 = struct[0].T_2_7
    K_stab_7 = struct[0].K_stab_7
    S_n_8 = struct[0].S_n_8
    H_8 = struct[0].H_8
    Omega_b_8 = struct[0].Omega_b_8
    T1d0_8 = struct[0].T1d0_8
    T1q0_8 = struct[0].T1q0_8
    X_d_8 = struct[0].X_d_8
    X_q_8 = struct[0].X_q_8
    X1d_8 = struct[0].X1d_8
    X1q_8 = struct[0].X1q_8
    D_8 = struct[0].D_8
    R_a_8 = struct[0].R_a_8
    K_delta_8 = struct[0].K_delta_8
    K_a_8 = struct[0].K_a_8
    K_ai_8 = struct[0].K_ai_8
    T_r_8 = struct[0].T_r_8
    Droop_8 = struct[0].Droop_8
    T_gov_1_8 = struct[0].T_gov_1_8
    T_gov_2_8 = struct[0].T_gov_2_8
    T_gov_3_8 = struct[0].T_gov_3_8
    omega_ref_8 = struct[0].omega_ref_8
    T_wo_8 = struct[0].T_wo_8
    T_1_8 = struct[0].T_1_8
    T_2_8 = struct[0].T_2_8
    K_stab_8 = struct[0].K_stab_8
    K_sec_5 = struct[0].K_sec_5
    K_sec_6 = struct[0].K_sec_6
    K_sec_7 = struct[0].K_sec_7
    K_sec_8 = struct[0].K_sec_8
    
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
    v_ref_5 = struct[0].v_ref_5
    v_pss_5 = struct[0].v_pss_5
    p_c_5 = struct[0].p_c_5
    v_ref_6 = struct[0].v_ref_6
    v_pss_6 = struct[0].v_pss_6
    p_c_6 = struct[0].p_c_6
    v_ref_7 = struct[0].v_ref_7
    v_pss_7 = struct[0].v_pss_7
    p_c_7 = struct[0].p_c_7
    v_ref_8 = struct[0].v_ref_8
    v_pss_8 = struct[0].v_pss_8
    p_c_8 = struct[0].p_c_8
    
    # Dynamical states:
    delta_5 = struct[0].x[0,0]
    omega_5 = struct[0].x[1,0]
    e1q_5 = struct[0].x[2,0]
    e1d_5 = struct[0].x[3,0]
    v_c_5 = struct[0].x[4,0]
    xi_v_5 = struct[0].x[5,0]
    x_gov_1_5 = struct[0].x[6,0]
    x_gov_2_5 = struct[0].x[7,0]
    x_wo_5 = struct[0].x[8,0]
    x_lead_5 = struct[0].x[9,0]
    delta_6 = struct[0].x[10,0]
    omega_6 = struct[0].x[11,0]
    e1q_6 = struct[0].x[12,0]
    e1d_6 = struct[0].x[13,0]
    v_c_6 = struct[0].x[14,0]
    xi_v_6 = struct[0].x[15,0]
    x_gov_1_6 = struct[0].x[16,0]
    x_gov_2_6 = struct[0].x[17,0]
    x_wo_6 = struct[0].x[18,0]
    x_lead_6 = struct[0].x[19,0]
    delta_7 = struct[0].x[20,0]
    omega_7 = struct[0].x[21,0]
    e1q_7 = struct[0].x[22,0]
    e1d_7 = struct[0].x[23,0]
    v_c_7 = struct[0].x[24,0]
    xi_v_7 = struct[0].x[25,0]
    x_gov_1_7 = struct[0].x[26,0]
    x_gov_2_7 = struct[0].x[27,0]
    x_wo_7 = struct[0].x[28,0]
    x_lead_7 = struct[0].x[29,0]
    delta_8 = struct[0].x[30,0]
    omega_8 = struct[0].x[31,0]
    e1q_8 = struct[0].x[32,0]
    e1d_8 = struct[0].x[33,0]
    v_c_8 = struct[0].x[34,0]
    xi_v_8 = struct[0].x[35,0]
    x_gov_1_8 = struct[0].x[36,0]
    x_gov_2_8 = struct[0].x[37,0]
    x_wo_8 = struct[0].x[38,0]
    x_lead_8 = struct[0].x[39,0]
    xi_freq = struct[0].x[40,0]
    
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
    i_d_5 = struct[0].y_run[16,0]
    i_q_5 = struct[0].y_run[17,0]
    p_g_5_1 = struct[0].y_run[18,0]
    q_g_5_1 = struct[0].y_run[19,0]
    v_f_5 = struct[0].y_run[20,0]
    p_m_ref_5 = struct[0].y_run[21,0]
    p_m_5 = struct[0].y_run[22,0]
    z_wo_5 = struct[0].y_run[23,0]
    v_pss_5 = struct[0].y_run[24,0]
    i_d_6 = struct[0].y_run[25,0]
    i_q_6 = struct[0].y_run[26,0]
    p_g_6_1 = struct[0].y_run[27,0]
    q_g_6_1 = struct[0].y_run[28,0]
    v_f_6 = struct[0].y_run[29,0]
    p_m_ref_6 = struct[0].y_run[30,0]
    p_m_6 = struct[0].y_run[31,0]
    z_wo_6 = struct[0].y_run[32,0]
    v_pss_6 = struct[0].y_run[33,0]
    i_d_7 = struct[0].y_run[34,0]
    i_q_7 = struct[0].y_run[35,0]
    p_g_7_1 = struct[0].y_run[36,0]
    q_g_7_1 = struct[0].y_run[37,0]
    v_f_7 = struct[0].y_run[38,0]
    p_m_ref_7 = struct[0].y_run[39,0]
    p_m_7 = struct[0].y_run[40,0]
    z_wo_7 = struct[0].y_run[41,0]
    v_pss_7 = struct[0].y_run[42,0]
    i_d_8 = struct[0].y_run[43,0]
    i_q_8 = struct[0].y_run[44,0]
    p_g_8_1 = struct[0].y_run[45,0]
    q_g_8_1 = struct[0].y_run[46,0]
    v_f_8 = struct[0].y_run[47,0]
    p_m_ref_8 = struct[0].y_run[48,0]
    p_m_8 = struct[0].y_run[49,0]
    z_wo_8 = struct[0].y_run[50,0]
    v_pss_8 = struct[0].y_run[51,0]
    p_r_5 = struct[0].y_run[52,0]
    p_r_6 = struct[0].y_run[53,0]
    p_r_7 = struct[0].y_run[54,0]
    p_r_8 = struct[0].y_run[55,0]
    omega_coi = struct[0].y_run[56,0]
    
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
    struct[0].u_run[16,0] = v_ref_5
    struct[0].u_run[17,0] = v_pss_5
    struct[0].u_run[18,0] = p_c_5
    struct[0].u_run[19,0] = v_ref_6
    struct[0].u_run[20,0] = v_pss_6
    struct[0].u_run[21,0] = p_c_6
    struct[0].u_run[22,0] = v_ref_7
    struct[0].u_run[23,0] = v_pss_7
    struct[0].u_run[24,0] = p_c_7
    struct[0].u_run[25,0] = v_ref_8
    struct[0].u_run[26,0] = v_pss_8
    struct[0].u_run[27,0] = p_c_8
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_5*delta_5 + Omega_b_5*(omega_5 - omega_coi)
        struct[0].f[1,0] = (-D_5*(omega_5 - omega_coi) - i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) - i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5)) + p_m_5)/(2*H_5)
        struct[0].f[2,0] = (-e1q_5 - i_d_5*(-X1d_5 + X_d_5) + v_f_5)/T1d0_5
        struct[0].f[3,0] = (-e1d_5 + i_q_5*(-X1q_5 + X_q_5))/T1q0_5
        struct[0].f[4,0] = (V_5 - v_c_5)/T_r_5
        struct[0].f[5,0] = -V_5 + v_ref_5
        struct[0].f[6,0] = (p_m_ref_5 - x_gov_1_5)/T_gov_1_5
        struct[0].f[7,0] = (x_gov_1_5 - x_gov_2_5)/T_gov_3_5
        struct[0].f[8,0] = (omega_5 - x_wo_5 - 1.0)/T_wo_5
        struct[0].f[9,0] = (-x_lead_5 + z_wo_5)/T_2_5
        struct[0].f[10,0] = -K_delta_6*delta_6 + Omega_b_6*(omega_6 - omega_coi)
        struct[0].f[11,0] = (-D_6*(omega_6 - omega_coi) - i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) - i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6)) + p_m_6)/(2*H_6)
        struct[0].f[12,0] = (-e1q_6 - i_d_6*(-X1d_6 + X_d_6) + v_f_6)/T1d0_6
        struct[0].f[13,0] = (-e1d_6 + i_q_6*(-X1q_6 + X_q_6))/T1q0_6
        struct[0].f[14,0] = (V_6 - v_c_6)/T_r_6
        struct[0].f[15,0] = -V_6 + v_ref_6
        struct[0].f[16,0] = (p_m_ref_6 - x_gov_1_6)/T_gov_1_6
        struct[0].f[17,0] = (x_gov_1_6 - x_gov_2_6)/T_gov_3_6
        struct[0].f[18,0] = (omega_6 - x_wo_6 - 1.0)/T_wo_6
        struct[0].f[19,0] = (-x_lead_6 + z_wo_6)/T_2_6
        struct[0].f[20,0] = -K_delta_7*delta_7 + Omega_b_7*(omega_7 - omega_coi)
        struct[0].f[21,0] = (-D_7*(omega_7 - omega_coi) - i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) - i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7)) + p_m_7)/(2*H_7)
        struct[0].f[22,0] = (-e1q_7 - i_d_7*(-X1d_7 + X_d_7) + v_f_7)/T1d0_7
        struct[0].f[23,0] = (-e1d_7 + i_q_7*(-X1q_7 + X_q_7))/T1q0_7
        struct[0].f[24,0] = (V_7 - v_c_7)/T_r_7
        struct[0].f[25,0] = -V_7 + v_ref_7
        struct[0].f[26,0] = (p_m_ref_7 - x_gov_1_7)/T_gov_1_7
        struct[0].f[27,0] = (x_gov_1_7 - x_gov_2_7)/T_gov_3_7
        struct[0].f[28,0] = (omega_7 - x_wo_7 - 1.0)/T_wo_7
        struct[0].f[29,0] = (-x_lead_7 + z_wo_7)/T_2_7
        struct[0].f[30,0] = -K_delta_8*delta_8 + Omega_b_8*(omega_8 - omega_coi)
        struct[0].f[31,0] = (-D_8*(omega_8 - omega_coi) - i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) - i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8)) + p_m_8)/(2*H_8)
        struct[0].f[32,0] = (-e1q_8 - i_d_8*(-X1d_8 + X_d_8) + v_f_8)/T1d0_8
        struct[0].f[33,0] = (-e1d_8 + i_q_8*(-X1q_8 + X_q_8))/T1q0_8
        struct[0].f[34,0] = (V_8 - v_c_8)/T_r_8
        struct[0].f[35,0] = -V_8 + v_ref_8
        struct[0].f[36,0] = (p_m_ref_8 - x_gov_1_8)/T_gov_1_8
        struct[0].f[37,0] = (x_gov_1_8 - x_gov_2_8)/T_gov_3_8
        struct[0].f[38,0] = (omega_8 - x_wo_8 - 1.0)/T_wo_8
        struct[0].f[39,0] = (-x_lead_8 + z_wo_8)/T_2_8
        struct[0].f[40,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_1/S_base + V_1**2*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].g[4,0] = -P_3/S_base + V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3**2*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].g[5,0] = -Q_3/S_base + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_3**2*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].g[6,0] = -P_4/S_base + V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4**2*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].g[7,0] = -Q_4/S_base + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_4**2*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + V_5**2*g_5_1 - S_n_5*p_g_5_1/S_base
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + V_5**2*(-b_5_1 - bs_5_1/2) - S_n_5*q_g_5_1/S_base
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + V_6**2*g_6_2 - S_n_6*p_g_6_1/S_base
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + V_6**2*(-b_6_2 - bs_6_2/2) - S_n_6*q_g_6_1/S_base
        struct[0].g[12,0] = -P_7/S_base + V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + V_7**2*g_7_3 - S_n_7*p_g_7_1/S_base
        struct[0].g[13,0] = -Q_7/S_base + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + V_7**2*(-b_7_3 - bs_7_3/2) - S_n_7*q_g_7_1/S_base
        struct[0].g[14,0] = -P_8/S_base + V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + V_8**2*g_8_4 - S_n_8*p_g_8_1/S_base
        struct[0].g[15,0] = -Q_8/S_base + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + V_8**2*(-b_8_4 - bs_8_4/2) - S_n_8*q_g_8_1/S_base
        struct[0].g[16,0] = R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5) + X1d_5*i_d_5 - e1q_5
        struct[0].g[17,0] = R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5) - X1q_5*i_q_5 - e1d_5
        struct[0].g[18,0] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5) - p_g_5_1
        struct[0].g[19,0] = V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5) - q_g_5_1
        struct[0].g[20,0] = K_a_5*(-v_c_5 + v_pss_5 + v_ref_5) + K_ai_5*xi_v_5 - v_f_5
        struct[0].g[21,0] = p_c_5 - p_m_ref_5 + p_r_5 - (omega_5 - omega_ref_5)/Droop_5
        struct[0].g[22,0] = T_gov_2_5*(x_gov_1_5 - x_gov_2_5)/T_gov_3_5 - p_m_5 + x_gov_2_5
        struct[0].g[23,0] = omega_5 - x_wo_5 - z_wo_5 - 1.0
        struct[0].g[24,0] = K_stab_5*(T_1_5*(-x_lead_5 + z_wo_5)/T_2_5 + x_lead_5) - v_pss_5
        struct[0].g[25,0] = R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6) + X1d_6*i_d_6 - e1q_6
        struct[0].g[26,0] = R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6) - X1q_6*i_q_6 - e1d_6
        struct[0].g[27,0] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6) - p_g_6_1
        struct[0].g[28,0] = V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6) - q_g_6_1
        struct[0].g[29,0] = K_a_6*(-v_c_6 + v_pss_6 + v_ref_6) + K_ai_6*xi_v_6 - v_f_6
        struct[0].g[30,0] = p_c_6 - p_m_ref_6 + p_r_6 - (omega_6 - omega_ref_6)/Droop_6
        struct[0].g[31,0] = T_gov_2_6*(x_gov_1_6 - x_gov_2_6)/T_gov_3_6 - p_m_6 + x_gov_2_6
        struct[0].g[32,0] = omega_6 - x_wo_6 - z_wo_6 - 1.0
        struct[0].g[33,0] = K_stab_6*(T_1_6*(-x_lead_6 + z_wo_6)/T_2_6 + x_lead_6) - v_pss_6
        struct[0].g[34,0] = R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7) + X1d_7*i_d_7 - e1q_7
        struct[0].g[35,0] = R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7) - X1q_7*i_q_7 - e1d_7
        struct[0].g[36,0] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7) - p_g_7_1
        struct[0].g[37,0] = V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7) - q_g_7_1
        struct[0].g[38,0] = K_a_7*(-v_c_7 + v_pss_7 + v_ref_7) + K_ai_7*xi_v_7 - v_f_7
        struct[0].g[39,0] = p_c_7 - p_m_ref_7 + p_r_7 - (omega_7 - omega_ref_7)/Droop_7
        struct[0].g[40,0] = T_gov_2_7*(x_gov_1_7 - x_gov_2_7)/T_gov_3_7 - p_m_7 + x_gov_2_7
        struct[0].g[41,0] = omega_7 - x_wo_7 - z_wo_7 - 1.0
        struct[0].g[42,0] = K_stab_7*(T_1_7*(-x_lead_7 + z_wo_7)/T_2_7 + x_lead_7) - v_pss_7
        struct[0].g[43,0] = R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8) + X1d_8*i_d_8 - e1q_8
        struct[0].g[44,0] = R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8) - X1q_8*i_q_8 - e1d_8
        struct[0].g[45,0] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8) - p_g_8_1
        struct[0].g[46,0] = V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8) - q_g_8_1
        struct[0].g[47,0] = K_a_8*(-v_c_8 + v_pss_8 + v_ref_8) + K_ai_8*xi_v_8 - v_f_8
        struct[0].g[48,0] = p_c_8 - p_m_ref_8 + p_r_8 - (omega_8 - omega_ref_8)/Droop_8
        struct[0].g[49,0] = T_gov_2_8*(x_gov_1_8 - x_gov_2_8)/T_gov_3_8 - p_m_8 + x_gov_2_8
        struct[0].g[50,0] = omega_8 - x_wo_8 - z_wo_8 - 1.0
        struct[0].g[51,0] = K_stab_8*(T_1_8*(-x_lead_8 + z_wo_8)/T_2_8 + x_lead_8) - v_pss_8
        struct[0].g[52,0] = K_sec_5*xi_freq/4 - p_r_5
        struct[0].g[53,0] = K_sec_6*xi_freq/4 - p_r_6
        struct[0].g[54,0] = K_sec_7*xi_freq/4 - p_r_7
        struct[0].g[55,0] = K_sec_8*xi_freq/4 - p_r_8
        struct[0].g[56,0] = omega_5/4 + omega_6/4 + omega_7/4 + omega_8/4 - omega_coi
    
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
        struct[0].h[8,0] = i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) + i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5))
        struct[0].h[9,0] = i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) + i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6))
        struct[0].h[10,0] = i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) + i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7))
        struct[0].h[11,0] = i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) + i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_5
        struct[0].Fx[0,1] = Omega_b_5
        struct[0].Fx[1,0] = (-V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fx[1,1] = -D_5/(2*H_5)
        struct[0].Fx[2,2] = -1/T1d0_5
        struct[0].Fx[3,3] = -1/T1q0_5
        struct[0].Fx[4,4] = -1/T_r_5
        struct[0].Fx[6,6] = -1/T_gov_1_5
        struct[0].Fx[7,6] = 1/T_gov_3_5
        struct[0].Fx[7,7] = -1/T_gov_3_5
        struct[0].Fx[8,1] = 1/T_wo_5
        struct[0].Fx[8,8] = -1/T_wo_5
        struct[0].Fx[9,9] = -1/T_2_5
        struct[0].Fx[10,10] = -K_delta_6
        struct[0].Fx[10,11] = Omega_b_6
        struct[0].Fx[11,10] = (-V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fx[11,11] = -D_6/(2*H_6)
        struct[0].Fx[12,12] = -1/T1d0_6
        struct[0].Fx[13,13] = -1/T1q0_6
        struct[0].Fx[14,14] = -1/T_r_6
        struct[0].Fx[16,16] = -1/T_gov_1_6
        struct[0].Fx[17,16] = 1/T_gov_3_6
        struct[0].Fx[17,17] = -1/T_gov_3_6
        struct[0].Fx[18,11] = 1/T_wo_6
        struct[0].Fx[18,18] = -1/T_wo_6
        struct[0].Fx[19,19] = -1/T_2_6
        struct[0].Fx[20,20] = -K_delta_7
        struct[0].Fx[20,21] = Omega_b_7
        struct[0].Fx[21,20] = (-V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fx[21,21] = -D_7/(2*H_7)
        struct[0].Fx[22,22] = -1/T1d0_7
        struct[0].Fx[23,23] = -1/T1q0_7
        struct[0].Fx[24,24] = -1/T_r_7
        struct[0].Fx[26,26] = -1/T_gov_1_7
        struct[0].Fx[27,26] = 1/T_gov_3_7
        struct[0].Fx[27,27] = -1/T_gov_3_7
        struct[0].Fx[28,21] = 1/T_wo_7
        struct[0].Fx[28,28] = -1/T_wo_7
        struct[0].Fx[29,29] = -1/T_2_7
        struct[0].Fx[30,30] = -K_delta_8
        struct[0].Fx[30,31] = Omega_b_8
        struct[0].Fx[31,30] = (-V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fx[31,31] = -D_8/(2*H_8)
        struct[0].Fx[32,32] = -1/T1d0_8
        struct[0].Fx[33,33] = -1/T1q0_8
        struct[0].Fx[34,34] = -1/T_r_8
        struct[0].Fx[36,36] = -1/T_gov_1_8
        struct[0].Fx[37,36] = 1/T_gov_3_8
        struct[0].Fx[37,37] = -1/T_gov_3_8
        struct[0].Fx[38,31] = 1/T_wo_8
        struct[0].Fx[38,38] = -1/T_wo_8
        struct[0].Fx[39,39] = -1/T_2_8

    if mode == 11:

        struct[0].Fy[0,56] = -Omega_b_5
        struct[0].Fy[1,8] = (-i_d_5*sin(delta_5 - theta_5) - i_q_5*cos(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,9] = (V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,16] = (-2*R_a_5*i_d_5 - V_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,17] = (-2*R_a_5*i_q_5 - V_5*cos(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,22] = 1/(2*H_5)
        struct[0].Fy[1,56] = D_5/(2*H_5)
        struct[0].Fy[2,16] = (X1d_5 - X_d_5)/T1d0_5
        struct[0].Fy[2,20] = 1/T1d0_5
        struct[0].Fy[3,17] = (-X1q_5 + X_q_5)/T1q0_5
        struct[0].Fy[4,8] = 1/T_r_5
        struct[0].Fy[5,8] = -1
        struct[0].Fy[6,21] = 1/T_gov_1_5
        struct[0].Fy[9,23] = 1/T_2_5
        struct[0].Fy[10,56] = -Omega_b_6
        struct[0].Fy[11,10] = (-i_d_6*sin(delta_6 - theta_6) - i_q_6*cos(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,11] = (V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,25] = (-2*R_a_6*i_d_6 - V_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,26] = (-2*R_a_6*i_q_6 - V_6*cos(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,31] = 1/(2*H_6)
        struct[0].Fy[11,56] = D_6/(2*H_6)
        struct[0].Fy[12,25] = (X1d_6 - X_d_6)/T1d0_6
        struct[0].Fy[12,29] = 1/T1d0_6
        struct[0].Fy[13,26] = (-X1q_6 + X_q_6)/T1q0_6
        struct[0].Fy[14,10] = 1/T_r_6
        struct[0].Fy[15,10] = -1
        struct[0].Fy[16,30] = 1/T_gov_1_6
        struct[0].Fy[19,32] = 1/T_2_6
        struct[0].Fy[20,56] = -Omega_b_7
        struct[0].Fy[21,12] = (-i_d_7*sin(delta_7 - theta_7) - i_q_7*cos(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,13] = (V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,34] = (-2*R_a_7*i_d_7 - V_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,35] = (-2*R_a_7*i_q_7 - V_7*cos(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,40] = 1/(2*H_7)
        struct[0].Fy[21,56] = D_7/(2*H_7)
        struct[0].Fy[22,34] = (X1d_7 - X_d_7)/T1d0_7
        struct[0].Fy[22,38] = 1/T1d0_7
        struct[0].Fy[23,35] = (-X1q_7 + X_q_7)/T1q0_7
        struct[0].Fy[24,12] = 1/T_r_7
        struct[0].Fy[25,12] = -1
        struct[0].Fy[26,39] = 1/T_gov_1_7
        struct[0].Fy[29,41] = 1/T_2_7
        struct[0].Fy[30,56] = -Omega_b_8
        struct[0].Fy[31,14] = (-i_d_8*sin(delta_8 - theta_8) - i_q_8*cos(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,15] = (V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,43] = (-2*R_a_8*i_d_8 - V_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,44] = (-2*R_a_8*i_q_8 - V_8*cos(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,49] = 1/(2*H_8)
        struct[0].Fy[31,56] = D_8/(2*H_8)
        struct[0].Fy[32,43] = (X1d_8 - X_d_8)/T1d0_8
        struct[0].Fy[32,47] = 1/T1d0_8
        struct[0].Fy[33,44] = (-X1q_8 + X_q_8)/T1q0_8
        struct[0].Fy[34,14] = 1/T_r_8
        struct[0].Fy[35,14] = -1
        struct[0].Fy[36,48] = 1/T_gov_1_8
        struct[0].Fy[39,50] = 1/T_2_8
        struct[0].Fy[40,56] = -1

        struct[0].Gx[16,0] = -V_5*sin(delta_5 - theta_5)
        struct[0].Gx[16,2] = -1
        struct[0].Gx[17,0] = V_5*cos(delta_5 - theta_5)
        struct[0].Gx[17,3] = -1
        struct[0].Gx[18,0] = V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Gx[19,0] = -V_5*i_d_5*sin(delta_5 - theta_5) - V_5*i_q_5*cos(delta_5 - theta_5)
        struct[0].Gx[20,4] = -K_a_5
        struct[0].Gx[20,5] = K_ai_5
        struct[0].Gx[21,1] = -1/Droop_5
        struct[0].Gx[22,6] = T_gov_2_5/T_gov_3_5
        struct[0].Gx[22,7] = -T_gov_2_5/T_gov_3_5 + 1
        struct[0].Gx[23,1] = 1
        struct[0].Gx[23,8] = -1
        struct[0].Gx[24,9] = K_stab_5*(-T_1_5/T_2_5 + 1)
        struct[0].Gx[25,10] = -V_6*sin(delta_6 - theta_6)
        struct[0].Gx[25,12] = -1
        struct[0].Gx[26,10] = V_6*cos(delta_6 - theta_6)
        struct[0].Gx[26,13] = -1
        struct[0].Gx[27,10] = V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Gx[28,10] = -V_6*i_d_6*sin(delta_6 - theta_6) - V_6*i_q_6*cos(delta_6 - theta_6)
        struct[0].Gx[29,14] = -K_a_6
        struct[0].Gx[29,15] = K_ai_6
        struct[0].Gx[30,11] = -1/Droop_6
        struct[0].Gx[31,16] = T_gov_2_6/T_gov_3_6
        struct[0].Gx[31,17] = -T_gov_2_6/T_gov_3_6 + 1
        struct[0].Gx[32,11] = 1
        struct[0].Gx[32,18] = -1
        struct[0].Gx[33,19] = K_stab_6*(-T_1_6/T_2_6 + 1)
        struct[0].Gx[34,20] = -V_7*sin(delta_7 - theta_7)
        struct[0].Gx[34,22] = -1
        struct[0].Gx[35,20] = V_7*cos(delta_7 - theta_7)
        struct[0].Gx[35,23] = -1
        struct[0].Gx[36,20] = V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Gx[37,20] = -V_7*i_d_7*sin(delta_7 - theta_7) - V_7*i_q_7*cos(delta_7 - theta_7)
        struct[0].Gx[38,24] = -K_a_7
        struct[0].Gx[38,25] = K_ai_7
        struct[0].Gx[39,21] = -1/Droop_7
        struct[0].Gx[40,26] = T_gov_2_7/T_gov_3_7
        struct[0].Gx[40,27] = -T_gov_2_7/T_gov_3_7 + 1
        struct[0].Gx[41,21] = 1
        struct[0].Gx[41,28] = -1
        struct[0].Gx[42,29] = K_stab_7*(-T_1_7/T_2_7 + 1)
        struct[0].Gx[43,30] = -V_8*sin(delta_8 - theta_8)
        struct[0].Gx[43,32] = -1
        struct[0].Gx[44,30] = V_8*cos(delta_8 - theta_8)
        struct[0].Gx[44,33] = -1
        struct[0].Gx[45,30] = V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8)
        struct[0].Gx[46,30] = -V_8*i_d_8*sin(delta_8 - theta_8) - V_8*i_q_8*cos(delta_8 - theta_8)
        struct[0].Gx[47,34] = -K_a_8
        struct[0].Gx[47,35] = K_ai_8
        struct[0].Gx[48,31] = -1/Droop_8
        struct[0].Gx[49,36] = T_gov_2_8/T_gov_3_8
        struct[0].Gx[49,37] = -T_gov_2_8/T_gov_3_8 + 1
        struct[0].Gx[50,31] = 1
        struct[0].Gx[50,38] = -1
        struct[0].Gx[51,39] = K_stab_8*(-T_1_8/T_2_8 + 1)
        struct[0].Gx[52,40] = K_sec_5/4
        struct[0].Gx[53,40] = K_sec_6/4
        struct[0].Gx[54,40] = K_sec_7/4
        struct[0].Gx[55,40] = K_sec_8/4
        struct[0].Gx[56,1] = 1/4
        struct[0].Gx[56,11] = 1/4
        struct[0].Gx[56,21] = 1/4
        struct[0].Gx[56,31] = 1/4

        struct[0].Gy[0,0] = 2*V_1*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[0,4] = V_1*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[0,5] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[0,6] = V_1*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[0,7] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[0,8] = V_1*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[0,9] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[1,0] = 2*V_1*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[1,4] = V_1*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[1,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[1,6] = V_1*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[1,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[1,8] = V_1*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[1,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[2,4] = V_2*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[2,5] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[2,6] = V_2*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[2,7] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[2,10] = V_2*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[2,11] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[3,4] = V_2*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[3,5] = V_2*V_3*(b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[3,6] = V_2*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[3,7] = V_2*V_4*(b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[3,10] = V_2*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[3,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[4,0] = V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[4,1] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[4,2] = V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[4,3] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[4,4] = V_1*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + 2*V_3*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[4,5] = V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[4,6] = V_3*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[4,7] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[4,12] = V_3*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[4,13] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[5,0] = V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[5,1] = V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[5,2] = V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[5,3] = V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[5,4] = V_1*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + 2*V_3*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[5,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[5,6] = V_3*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[5,7] = V_3*V_4*(b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[5,12] = V_3*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[5,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[6,0] = V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[6,1] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[6,2] = V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[6,3] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[6,4] = V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[6,5] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[6,6] = V_1*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + 2*V_4*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[6,7] = V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[6,14] = V_4*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[6,15] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[7,0] = V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[7,1] = V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[7,2] = V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[7,3] = V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[7,4] = V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[7,5] = V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[7,6] = V_1*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + 2*V_4*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[7,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[7,14] = V_4*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[7,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[8,0] = V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[8,1] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[8,8] = V_1*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + 2*V_5*g_5_1
        struct[0].Gy[8,9] = V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[8,18] = -S_n_5/S_base
        struct[0].Gy[9,0] = V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[9,1] = V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[9,8] = V_1*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + 2*V_5*(-b_5_1 - bs_5_1/2)
        struct[0].Gy[9,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[9,19] = -S_n_5/S_base
        struct[0].Gy[10,2] = V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[10,3] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[10,10] = V_2*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + 2*V_6*g_6_2
        struct[0].Gy[10,11] = V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[10,27] = -S_n_6/S_base
        struct[0].Gy[11,2] = V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[11,3] = V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[11,10] = V_2*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + 2*V_6*(-b_6_2 - bs_6_2/2)
        struct[0].Gy[11,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[11,28] = -S_n_6/S_base
        struct[0].Gy[12,4] = V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[12,5] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[12,12] = V_3*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + 2*V_7*g_7_3
        struct[0].Gy[12,13] = V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[12,36] = -S_n_7/S_base
        struct[0].Gy[13,4] = V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[13,5] = V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[13,12] = V_3*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + 2*V_7*(-b_7_3 - bs_7_3/2)
        struct[0].Gy[13,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[13,37] = -S_n_7/S_base
        struct[0].Gy[14,6] = V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[14,7] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[14,14] = V_4*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + 2*V_8*g_8_4
        struct[0].Gy[14,15] = V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[14,45] = -S_n_8/S_base
        struct[0].Gy[15,6] = V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[15,7] = V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[15,14] = V_4*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + 2*V_8*(-b_8_4 - bs_8_4/2)
        struct[0].Gy[15,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[15,46] = -S_n_8/S_base
        struct[0].Gy[16,8] = cos(delta_5 - theta_5)
        struct[0].Gy[16,9] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy[16,16] = X1d_5
        struct[0].Gy[16,17] = R_a_5
        struct[0].Gy[17,8] = sin(delta_5 - theta_5)
        struct[0].Gy[17,9] = -V_5*cos(delta_5 - theta_5)
        struct[0].Gy[17,16] = R_a_5
        struct[0].Gy[17,17] = -X1q_5
        struct[0].Gy[18,8] = i_d_5*sin(delta_5 - theta_5) + i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy[18,9] = -V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy[18,16] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy[18,17] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy[19,8] = i_d_5*cos(delta_5 - theta_5) - i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy[19,9] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy[19,16] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy[19,17] = -V_5*sin(delta_5 - theta_5)
        struct[0].Gy[20,24] = K_a_5
        struct[0].Gy[24,23] = K_stab_5*T_1_5/T_2_5
        struct[0].Gy[25,10] = cos(delta_6 - theta_6)
        struct[0].Gy[25,11] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy[25,25] = X1d_6
        struct[0].Gy[25,26] = R_a_6
        struct[0].Gy[26,10] = sin(delta_6 - theta_6)
        struct[0].Gy[26,11] = -V_6*cos(delta_6 - theta_6)
        struct[0].Gy[26,25] = R_a_6
        struct[0].Gy[26,26] = -X1q_6
        struct[0].Gy[27,10] = i_d_6*sin(delta_6 - theta_6) + i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy[27,11] = -V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy[27,25] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy[27,26] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy[28,10] = i_d_6*cos(delta_6 - theta_6) - i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy[28,11] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy[28,25] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy[28,26] = -V_6*sin(delta_6 - theta_6)
        struct[0].Gy[29,33] = K_a_6
        struct[0].Gy[33,32] = K_stab_6*T_1_6/T_2_6
        struct[0].Gy[34,12] = cos(delta_7 - theta_7)
        struct[0].Gy[34,13] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy[34,34] = X1d_7
        struct[0].Gy[34,35] = R_a_7
        struct[0].Gy[35,12] = sin(delta_7 - theta_7)
        struct[0].Gy[35,13] = -V_7*cos(delta_7 - theta_7)
        struct[0].Gy[35,34] = R_a_7
        struct[0].Gy[35,35] = -X1q_7
        struct[0].Gy[36,12] = i_d_7*sin(delta_7 - theta_7) + i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy[36,13] = -V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy[36,34] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy[36,35] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy[37,12] = i_d_7*cos(delta_7 - theta_7) - i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy[37,13] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy[37,34] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy[37,35] = -V_7*sin(delta_7 - theta_7)
        struct[0].Gy[38,42] = K_a_7
        struct[0].Gy[42,41] = K_stab_7*T_1_7/T_2_7
        struct[0].Gy[43,14] = cos(delta_8 - theta_8)
        struct[0].Gy[43,15] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy[43,43] = X1d_8
        struct[0].Gy[43,44] = R_a_8
        struct[0].Gy[44,14] = sin(delta_8 - theta_8)
        struct[0].Gy[44,15] = -V_8*cos(delta_8 - theta_8)
        struct[0].Gy[44,43] = R_a_8
        struct[0].Gy[44,44] = -X1q_8
        struct[0].Gy[45,14] = i_d_8*sin(delta_8 - theta_8) + i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy[45,15] = -V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy[45,43] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy[45,44] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy[46,14] = i_d_8*cos(delta_8 - theta_8) - i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy[46,15] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy[46,43] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy[46,44] = -V_8*sin(delta_8 - theta_8)
        struct[0].Gy[47,51] = K_a_8
        struct[0].Gy[51,50] = K_stab_8*T_1_8/T_2_8

    if mode > 12:

        struct[0].Fu[5,16] = 1
        struct[0].Fu[15,19] = 1
        struct[0].Fu[25,22] = 1
        struct[0].Fu[35,25] = 1

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
        struct[0].Gu[20,16] = K_a_5
        struct[0].Gu[20,17] = K_a_5
        struct[0].Gu[29,19] = K_a_6
        struct[0].Gu[29,20] = K_a_6
        struct[0].Gu[38,22] = K_a_7
        struct[0].Gu[38,23] = K_a_7
        struct[0].Gu[47,25] = K_a_8
        struct[0].Gu[47,26] = K_a_8

        struct[0].Hx[8,0] = V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Hx[9,10] = V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Hx[10,20] = V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Hx[11,30] = V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8)

        struct[0].Hy[0,0] = 1
        struct[0].Hy[1,2] = 1
        struct[0].Hy[2,4] = 1
        struct[0].Hy[3,6] = 1
        struct[0].Hy[4,8] = 1
        struct[0].Hy[5,10] = 1
        struct[0].Hy[6,12] = 1
        struct[0].Hy[7,14] = 1
        struct[0].Hy[8,8] = i_d_5*sin(delta_5 - theta_5) + i_q_5*cos(delta_5 - theta_5)
        struct[0].Hy[8,9] = -V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Hy[8,16] = 2*R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)
        struct[0].Hy[8,17] = 2*R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5)
        struct[0].Hy[9,10] = i_d_6*sin(delta_6 - theta_6) + i_q_6*cos(delta_6 - theta_6)
        struct[0].Hy[9,11] = -V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Hy[9,25] = 2*R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)
        struct[0].Hy[9,26] = 2*R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6)
        struct[0].Hy[10,12] = i_d_7*sin(delta_7 - theta_7) + i_q_7*cos(delta_7 - theta_7)
        struct[0].Hy[10,13] = -V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Hy[10,34] = 2*R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)
        struct[0].Hy[10,35] = 2*R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7)
        struct[0].Hy[11,14] = i_d_8*sin(delta_8 - theta_8) + i_q_8*cos(delta_8 - theta_8)
        struct[0].Hy[11,15] = -V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8)
        struct[0].Hy[11,43] = 2*R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)
        struct[0].Hy[11,44] = 2*R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8)




def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_2 = struct[0].g_1_2
    b_1_2 = struct[0].b_1_2
    bs_1_2 = struct[0].bs_1_2
    g_3_4 = struct[0].g_3_4
    b_3_4 = struct[0].b_3_4
    bs_3_4 = struct[0].bs_3_4
    g_1_3 = struct[0].g_1_3
    b_1_3 = struct[0].b_1_3
    bs_1_3 = struct[0].bs_1_3
    g_2_4 = struct[0].g_2_4
    b_2_4 = struct[0].b_2_4
    bs_2_4 = struct[0].bs_2_4
    g_1_4 = struct[0].g_1_4
    b_1_4 = struct[0].b_1_4
    bs_1_4 = struct[0].bs_1_4
    g_3_2 = struct[0].g_3_2
    b_3_2 = struct[0].b_3_2
    bs_3_2 = struct[0].bs_3_2
    g_5_1 = struct[0].g_5_1
    b_5_1 = struct[0].b_5_1
    bs_5_1 = struct[0].bs_5_1
    g_6_2 = struct[0].g_6_2
    b_6_2 = struct[0].b_6_2
    bs_6_2 = struct[0].bs_6_2
    g_7_3 = struct[0].g_7_3
    b_7_3 = struct[0].b_7_3
    bs_7_3 = struct[0].bs_7_3
    g_8_4 = struct[0].g_8_4
    b_8_4 = struct[0].b_8_4
    bs_8_4 = struct[0].bs_8_4
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    S_n_5 = struct[0].S_n_5
    H_5 = struct[0].H_5
    Omega_b_5 = struct[0].Omega_b_5
    T1d0_5 = struct[0].T1d0_5
    T1q0_5 = struct[0].T1q0_5
    X_d_5 = struct[0].X_d_5
    X_q_5 = struct[0].X_q_5
    X1d_5 = struct[0].X1d_5
    X1q_5 = struct[0].X1q_5
    D_5 = struct[0].D_5
    R_a_5 = struct[0].R_a_5
    K_delta_5 = struct[0].K_delta_5
    K_a_5 = struct[0].K_a_5
    K_ai_5 = struct[0].K_ai_5
    T_r_5 = struct[0].T_r_5
    Droop_5 = struct[0].Droop_5
    T_gov_1_5 = struct[0].T_gov_1_5
    T_gov_2_5 = struct[0].T_gov_2_5
    T_gov_3_5 = struct[0].T_gov_3_5
    omega_ref_5 = struct[0].omega_ref_5
    T_wo_5 = struct[0].T_wo_5
    T_1_5 = struct[0].T_1_5
    T_2_5 = struct[0].T_2_5
    K_stab_5 = struct[0].K_stab_5
    S_n_6 = struct[0].S_n_6
    H_6 = struct[0].H_6
    Omega_b_6 = struct[0].Omega_b_6
    T1d0_6 = struct[0].T1d0_6
    T1q0_6 = struct[0].T1q0_6
    X_d_6 = struct[0].X_d_6
    X_q_6 = struct[0].X_q_6
    X1d_6 = struct[0].X1d_6
    X1q_6 = struct[0].X1q_6
    D_6 = struct[0].D_6
    R_a_6 = struct[0].R_a_6
    K_delta_6 = struct[0].K_delta_6
    K_a_6 = struct[0].K_a_6
    K_ai_6 = struct[0].K_ai_6
    T_r_6 = struct[0].T_r_6
    Droop_6 = struct[0].Droop_6
    T_gov_1_6 = struct[0].T_gov_1_6
    T_gov_2_6 = struct[0].T_gov_2_6
    T_gov_3_6 = struct[0].T_gov_3_6
    omega_ref_6 = struct[0].omega_ref_6
    T_wo_6 = struct[0].T_wo_6
    T_1_6 = struct[0].T_1_6
    T_2_6 = struct[0].T_2_6
    K_stab_6 = struct[0].K_stab_6
    S_n_7 = struct[0].S_n_7
    H_7 = struct[0].H_7
    Omega_b_7 = struct[0].Omega_b_7
    T1d0_7 = struct[0].T1d0_7
    T1q0_7 = struct[0].T1q0_7
    X_d_7 = struct[0].X_d_7
    X_q_7 = struct[0].X_q_7
    X1d_7 = struct[0].X1d_7
    X1q_7 = struct[0].X1q_7
    D_7 = struct[0].D_7
    R_a_7 = struct[0].R_a_7
    K_delta_7 = struct[0].K_delta_7
    K_a_7 = struct[0].K_a_7
    K_ai_7 = struct[0].K_ai_7
    T_r_7 = struct[0].T_r_7
    Droop_7 = struct[0].Droop_7
    T_gov_1_7 = struct[0].T_gov_1_7
    T_gov_2_7 = struct[0].T_gov_2_7
    T_gov_3_7 = struct[0].T_gov_3_7
    omega_ref_7 = struct[0].omega_ref_7
    T_wo_7 = struct[0].T_wo_7
    T_1_7 = struct[0].T_1_7
    T_2_7 = struct[0].T_2_7
    K_stab_7 = struct[0].K_stab_7
    S_n_8 = struct[0].S_n_8
    H_8 = struct[0].H_8
    Omega_b_8 = struct[0].Omega_b_8
    T1d0_8 = struct[0].T1d0_8
    T1q0_8 = struct[0].T1q0_8
    X_d_8 = struct[0].X_d_8
    X_q_8 = struct[0].X_q_8
    X1d_8 = struct[0].X1d_8
    X1q_8 = struct[0].X1q_8
    D_8 = struct[0].D_8
    R_a_8 = struct[0].R_a_8
    K_delta_8 = struct[0].K_delta_8
    K_a_8 = struct[0].K_a_8
    K_ai_8 = struct[0].K_ai_8
    T_r_8 = struct[0].T_r_8
    Droop_8 = struct[0].Droop_8
    T_gov_1_8 = struct[0].T_gov_1_8
    T_gov_2_8 = struct[0].T_gov_2_8
    T_gov_3_8 = struct[0].T_gov_3_8
    omega_ref_8 = struct[0].omega_ref_8
    T_wo_8 = struct[0].T_wo_8
    T_1_8 = struct[0].T_1_8
    T_2_8 = struct[0].T_2_8
    K_stab_8 = struct[0].K_stab_8
    K_sec_5 = struct[0].K_sec_5
    K_sec_6 = struct[0].K_sec_6
    K_sec_7 = struct[0].K_sec_7
    K_sec_8 = struct[0].K_sec_8
    
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
    v_ref_5 = struct[0].v_ref_5
    v_pss_5 = struct[0].v_pss_5
    p_c_5 = struct[0].p_c_5
    v_ref_6 = struct[0].v_ref_6
    v_pss_6 = struct[0].v_pss_6
    p_c_6 = struct[0].p_c_6
    v_ref_7 = struct[0].v_ref_7
    v_pss_7 = struct[0].v_pss_7
    p_c_7 = struct[0].p_c_7
    v_ref_8 = struct[0].v_ref_8
    v_pss_8 = struct[0].v_pss_8
    p_c_8 = struct[0].p_c_8
    
    # Dynamical states:
    delta_5 = struct[0].x[0,0]
    omega_5 = struct[0].x[1,0]
    e1q_5 = struct[0].x[2,0]
    e1d_5 = struct[0].x[3,0]
    v_c_5 = struct[0].x[4,0]
    xi_v_5 = struct[0].x[5,0]
    x_gov_1_5 = struct[0].x[6,0]
    x_gov_2_5 = struct[0].x[7,0]
    x_wo_5 = struct[0].x[8,0]
    x_lead_5 = struct[0].x[9,0]
    delta_6 = struct[0].x[10,0]
    omega_6 = struct[0].x[11,0]
    e1q_6 = struct[0].x[12,0]
    e1d_6 = struct[0].x[13,0]
    v_c_6 = struct[0].x[14,0]
    xi_v_6 = struct[0].x[15,0]
    x_gov_1_6 = struct[0].x[16,0]
    x_gov_2_6 = struct[0].x[17,0]
    x_wo_6 = struct[0].x[18,0]
    x_lead_6 = struct[0].x[19,0]
    delta_7 = struct[0].x[20,0]
    omega_7 = struct[0].x[21,0]
    e1q_7 = struct[0].x[22,0]
    e1d_7 = struct[0].x[23,0]
    v_c_7 = struct[0].x[24,0]
    xi_v_7 = struct[0].x[25,0]
    x_gov_1_7 = struct[0].x[26,0]
    x_gov_2_7 = struct[0].x[27,0]
    x_wo_7 = struct[0].x[28,0]
    x_lead_7 = struct[0].x[29,0]
    delta_8 = struct[0].x[30,0]
    omega_8 = struct[0].x[31,0]
    e1q_8 = struct[0].x[32,0]
    e1d_8 = struct[0].x[33,0]
    v_c_8 = struct[0].x[34,0]
    xi_v_8 = struct[0].x[35,0]
    x_gov_1_8 = struct[0].x[36,0]
    x_gov_2_8 = struct[0].x[37,0]
    x_wo_8 = struct[0].x[38,0]
    x_lead_8 = struct[0].x[39,0]
    xi_freq = struct[0].x[40,0]
    
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
    i_d_5 = struct[0].y_ini[16,0]
    i_q_5 = struct[0].y_ini[17,0]
    p_g_5_1 = struct[0].y_ini[18,0]
    q_g_5_1 = struct[0].y_ini[19,0]
    v_f_5 = struct[0].y_ini[20,0]
    p_m_ref_5 = struct[0].y_ini[21,0]
    p_m_5 = struct[0].y_ini[22,0]
    z_wo_5 = struct[0].y_ini[23,0]
    v_pss_5 = struct[0].y_ini[24,0]
    i_d_6 = struct[0].y_ini[25,0]
    i_q_6 = struct[0].y_ini[26,0]
    p_g_6_1 = struct[0].y_ini[27,0]
    q_g_6_1 = struct[0].y_ini[28,0]
    v_f_6 = struct[0].y_ini[29,0]
    p_m_ref_6 = struct[0].y_ini[30,0]
    p_m_6 = struct[0].y_ini[31,0]
    z_wo_6 = struct[0].y_ini[32,0]
    v_pss_6 = struct[0].y_ini[33,0]
    i_d_7 = struct[0].y_ini[34,0]
    i_q_7 = struct[0].y_ini[35,0]
    p_g_7_1 = struct[0].y_ini[36,0]
    q_g_7_1 = struct[0].y_ini[37,0]
    v_f_7 = struct[0].y_ini[38,0]
    p_m_ref_7 = struct[0].y_ini[39,0]
    p_m_7 = struct[0].y_ini[40,0]
    z_wo_7 = struct[0].y_ini[41,0]
    v_pss_7 = struct[0].y_ini[42,0]
    i_d_8 = struct[0].y_ini[43,0]
    i_q_8 = struct[0].y_ini[44,0]
    p_g_8_1 = struct[0].y_ini[45,0]
    q_g_8_1 = struct[0].y_ini[46,0]
    v_f_8 = struct[0].y_ini[47,0]
    p_m_ref_8 = struct[0].y_ini[48,0]
    p_m_8 = struct[0].y_ini[49,0]
    z_wo_8 = struct[0].y_ini[50,0]
    v_pss_8 = struct[0].y_ini[51,0]
    p_r_5 = struct[0].y_ini[52,0]
    p_r_6 = struct[0].y_ini[53,0]
    p_r_7 = struct[0].y_ini[54,0]
    p_r_8 = struct[0].y_ini[55,0]
    omega_coi = struct[0].y_ini[56,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_5*delta_5 + Omega_b_5*(omega_5 - omega_coi)
        struct[0].f[1,0] = (-D_5*(omega_5 - omega_coi) - i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) - i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5)) + p_m_5)/(2*H_5)
        struct[0].f[2,0] = (-e1q_5 - i_d_5*(-X1d_5 + X_d_5) + v_f_5)/T1d0_5
        struct[0].f[3,0] = (-e1d_5 + i_q_5*(-X1q_5 + X_q_5))/T1q0_5
        struct[0].f[4,0] = (V_5 - v_c_5)/T_r_5
        struct[0].f[5,0] = -V_5 + v_ref_5
        struct[0].f[6,0] = (p_m_ref_5 - x_gov_1_5)/T_gov_1_5
        struct[0].f[7,0] = (x_gov_1_5 - x_gov_2_5)/T_gov_3_5
        struct[0].f[8,0] = (omega_5 - x_wo_5 - 1.0)/T_wo_5
        struct[0].f[9,0] = (-x_lead_5 + z_wo_5)/T_2_5
        struct[0].f[10,0] = -K_delta_6*delta_6 + Omega_b_6*(omega_6 - omega_coi)
        struct[0].f[11,0] = (-D_6*(omega_6 - omega_coi) - i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) - i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6)) + p_m_6)/(2*H_6)
        struct[0].f[12,0] = (-e1q_6 - i_d_6*(-X1d_6 + X_d_6) + v_f_6)/T1d0_6
        struct[0].f[13,0] = (-e1d_6 + i_q_6*(-X1q_6 + X_q_6))/T1q0_6
        struct[0].f[14,0] = (V_6 - v_c_6)/T_r_6
        struct[0].f[15,0] = -V_6 + v_ref_6
        struct[0].f[16,0] = (p_m_ref_6 - x_gov_1_6)/T_gov_1_6
        struct[0].f[17,0] = (x_gov_1_6 - x_gov_2_6)/T_gov_3_6
        struct[0].f[18,0] = (omega_6 - x_wo_6 - 1.0)/T_wo_6
        struct[0].f[19,0] = (-x_lead_6 + z_wo_6)/T_2_6
        struct[0].f[20,0] = -K_delta_7*delta_7 + Omega_b_7*(omega_7 - omega_coi)
        struct[0].f[21,0] = (-D_7*(omega_7 - omega_coi) - i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) - i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7)) + p_m_7)/(2*H_7)
        struct[0].f[22,0] = (-e1q_7 - i_d_7*(-X1d_7 + X_d_7) + v_f_7)/T1d0_7
        struct[0].f[23,0] = (-e1d_7 + i_q_7*(-X1q_7 + X_q_7))/T1q0_7
        struct[0].f[24,0] = (V_7 - v_c_7)/T_r_7
        struct[0].f[25,0] = -V_7 + v_ref_7
        struct[0].f[26,0] = (p_m_ref_7 - x_gov_1_7)/T_gov_1_7
        struct[0].f[27,0] = (x_gov_1_7 - x_gov_2_7)/T_gov_3_7
        struct[0].f[28,0] = (omega_7 - x_wo_7 - 1.0)/T_wo_7
        struct[0].f[29,0] = (-x_lead_7 + z_wo_7)/T_2_7
        struct[0].f[30,0] = -K_delta_8*delta_8 + Omega_b_8*(omega_8 - omega_coi)
        struct[0].f[31,0] = (-D_8*(omega_8 - omega_coi) - i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) - i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8)) + p_m_8)/(2*H_8)
        struct[0].f[32,0] = (-e1q_8 - i_d_8*(-X1d_8 + X_d_8) + v_f_8)/T1d0_8
        struct[0].f[33,0] = (-e1d_8 + i_q_8*(-X1q_8 + X_q_8))/T1q0_8
        struct[0].f[34,0] = (V_8 - v_c_8)/T_r_8
        struct[0].f[35,0] = -V_8 + v_ref_8
        struct[0].f[36,0] = (p_m_ref_8 - x_gov_1_8)/T_gov_1_8
        struct[0].f[37,0] = (x_gov_1_8 - x_gov_2_8)/T_gov_3_8
        struct[0].f[38,0] = (omega_8 - x_wo_8 - 1.0)/T_wo_8
        struct[0].f[39,0] = (-x_lead_8 + z_wo_8)/T_2_8
        struct[0].f[40,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_1/S_base + V_1**2*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].g[4,0] = -P_3/S_base + V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3**2*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].g[5,0] = -Q_3/S_base + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_3**2*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].g[6,0] = -P_4/S_base + V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4**2*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].g[7,0] = -Q_4/S_base + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_4**2*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + V_5**2*g_5_1 - S_n_5*p_g_5_1/S_base
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + V_5**2*(-b_5_1 - bs_5_1/2) - S_n_5*q_g_5_1/S_base
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + V_6**2*g_6_2 - S_n_6*p_g_6_1/S_base
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + V_6**2*(-b_6_2 - bs_6_2/2) - S_n_6*q_g_6_1/S_base
        struct[0].g[12,0] = -P_7/S_base + V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + V_7**2*g_7_3 - S_n_7*p_g_7_1/S_base
        struct[0].g[13,0] = -Q_7/S_base + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + V_7**2*(-b_7_3 - bs_7_3/2) - S_n_7*q_g_7_1/S_base
        struct[0].g[14,0] = -P_8/S_base + V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + V_8**2*g_8_4 - S_n_8*p_g_8_1/S_base
        struct[0].g[15,0] = -Q_8/S_base + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + V_8**2*(-b_8_4 - bs_8_4/2) - S_n_8*q_g_8_1/S_base
        struct[0].g[16,0] = R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5) + X1d_5*i_d_5 - e1q_5
        struct[0].g[17,0] = R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5) - X1q_5*i_q_5 - e1d_5
        struct[0].g[18,0] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5) - p_g_5_1
        struct[0].g[19,0] = V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5) - q_g_5_1
        struct[0].g[20,0] = K_a_5*(-v_c_5 + v_pss_5 + v_ref_5) + K_ai_5*xi_v_5 - v_f_5
        struct[0].g[21,0] = p_c_5 - p_m_ref_5 + p_r_5 - (omega_5 - omega_ref_5)/Droop_5
        struct[0].g[22,0] = T_gov_2_5*(x_gov_1_5 - x_gov_2_5)/T_gov_3_5 - p_m_5 + x_gov_2_5
        struct[0].g[23,0] = omega_5 - x_wo_5 - z_wo_5 - 1.0
        struct[0].g[24,0] = K_stab_5*(T_1_5*(-x_lead_5 + z_wo_5)/T_2_5 + x_lead_5) - v_pss_5
        struct[0].g[25,0] = R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6) + X1d_6*i_d_6 - e1q_6
        struct[0].g[26,0] = R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6) - X1q_6*i_q_6 - e1d_6
        struct[0].g[27,0] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6) - p_g_6_1
        struct[0].g[28,0] = V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6) - q_g_6_1
        struct[0].g[29,0] = K_a_6*(-v_c_6 + v_pss_6 + v_ref_6) + K_ai_6*xi_v_6 - v_f_6
        struct[0].g[30,0] = p_c_6 - p_m_ref_6 + p_r_6 - (omega_6 - omega_ref_6)/Droop_6
        struct[0].g[31,0] = T_gov_2_6*(x_gov_1_6 - x_gov_2_6)/T_gov_3_6 - p_m_6 + x_gov_2_6
        struct[0].g[32,0] = omega_6 - x_wo_6 - z_wo_6 - 1.0
        struct[0].g[33,0] = K_stab_6*(T_1_6*(-x_lead_6 + z_wo_6)/T_2_6 + x_lead_6) - v_pss_6
        struct[0].g[34,0] = R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7) + X1d_7*i_d_7 - e1q_7
        struct[0].g[35,0] = R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7) - X1q_7*i_q_7 - e1d_7
        struct[0].g[36,0] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7) - p_g_7_1
        struct[0].g[37,0] = V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7) - q_g_7_1
        struct[0].g[38,0] = K_a_7*(-v_c_7 + v_pss_7 + v_ref_7) + K_ai_7*xi_v_7 - v_f_7
        struct[0].g[39,0] = p_c_7 - p_m_ref_7 + p_r_7 - (omega_7 - omega_ref_7)/Droop_7
        struct[0].g[40,0] = T_gov_2_7*(x_gov_1_7 - x_gov_2_7)/T_gov_3_7 - p_m_7 + x_gov_2_7
        struct[0].g[41,0] = omega_7 - x_wo_7 - z_wo_7 - 1.0
        struct[0].g[42,0] = K_stab_7*(T_1_7*(-x_lead_7 + z_wo_7)/T_2_7 + x_lead_7) - v_pss_7
        struct[0].g[43,0] = R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8) + X1d_8*i_d_8 - e1q_8
        struct[0].g[44,0] = R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8) - X1q_8*i_q_8 - e1d_8
        struct[0].g[45,0] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8) - p_g_8_1
        struct[0].g[46,0] = V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8) - q_g_8_1
        struct[0].g[47,0] = K_a_8*(-v_c_8 + v_pss_8 + v_ref_8) + K_ai_8*xi_v_8 - v_f_8
        struct[0].g[48,0] = p_c_8 - p_m_ref_8 + p_r_8 - (omega_8 - omega_ref_8)/Droop_8
        struct[0].g[49,0] = T_gov_2_8*(x_gov_1_8 - x_gov_2_8)/T_gov_3_8 - p_m_8 + x_gov_2_8
        struct[0].g[50,0] = omega_8 - x_wo_8 - z_wo_8 - 1.0
        struct[0].g[51,0] = K_stab_8*(T_1_8*(-x_lead_8 + z_wo_8)/T_2_8 + x_lead_8) - v_pss_8
        struct[0].g[52,0] = K_sec_5*xi_freq/4 - p_r_5
        struct[0].g[53,0] = K_sec_6*xi_freq/4 - p_r_6
        struct[0].g[54,0] = K_sec_7*xi_freq/4 - p_r_7
        struct[0].g[55,0] = K_sec_8*xi_freq/4 - p_r_8
        struct[0].g[56,0] = omega_5/4 + omega_6/4 + omega_7/4 + omega_8/4 - omega_coi
    
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
        struct[0].h[8,0] = i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) + i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5))
        struct[0].h[9,0] = i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) + i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6))
        struct[0].h[10,0] = i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) + i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7))
        struct[0].h[11,0] = i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) + i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_5
        struct[0].Fx_ini[0,1] = Omega_b_5
        struct[0].Fx_ini[1,0] = (-V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fx_ini[1,1] = -D_5/(2*H_5)
        struct[0].Fx_ini[2,2] = -1/T1d0_5
        struct[0].Fx_ini[3,3] = -1/T1q0_5
        struct[0].Fx_ini[4,4] = -1/T_r_5
        struct[0].Fx_ini[6,6] = -1/T_gov_1_5
        struct[0].Fx_ini[7,6] = 1/T_gov_3_5
        struct[0].Fx_ini[7,7] = -1/T_gov_3_5
        struct[0].Fx_ini[8,1] = 1/T_wo_5
        struct[0].Fx_ini[8,8] = -1/T_wo_5
        struct[0].Fx_ini[9,9] = -1/T_2_5
        struct[0].Fx_ini[10,10] = -K_delta_6
        struct[0].Fx_ini[10,11] = Omega_b_6
        struct[0].Fx_ini[11,10] = (-V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fx_ini[11,11] = -D_6/(2*H_6)
        struct[0].Fx_ini[12,12] = -1/T1d0_6
        struct[0].Fx_ini[13,13] = -1/T1q0_6
        struct[0].Fx_ini[14,14] = -1/T_r_6
        struct[0].Fx_ini[16,16] = -1/T_gov_1_6
        struct[0].Fx_ini[17,16] = 1/T_gov_3_6
        struct[0].Fx_ini[17,17] = -1/T_gov_3_6
        struct[0].Fx_ini[18,11] = 1/T_wo_6
        struct[0].Fx_ini[18,18] = -1/T_wo_6
        struct[0].Fx_ini[19,19] = -1/T_2_6
        struct[0].Fx_ini[20,20] = -K_delta_7
        struct[0].Fx_ini[20,21] = Omega_b_7
        struct[0].Fx_ini[21,20] = (-V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fx_ini[21,21] = -D_7/(2*H_7)
        struct[0].Fx_ini[22,22] = -1/T1d0_7
        struct[0].Fx_ini[23,23] = -1/T1q0_7
        struct[0].Fx_ini[24,24] = -1/T_r_7
        struct[0].Fx_ini[26,26] = -1/T_gov_1_7
        struct[0].Fx_ini[27,26] = 1/T_gov_3_7
        struct[0].Fx_ini[27,27] = -1/T_gov_3_7
        struct[0].Fx_ini[28,21] = 1/T_wo_7
        struct[0].Fx_ini[28,28] = -1/T_wo_7
        struct[0].Fx_ini[29,29] = -1/T_2_7
        struct[0].Fx_ini[30,30] = -K_delta_8
        struct[0].Fx_ini[30,31] = Omega_b_8
        struct[0].Fx_ini[31,30] = (-V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fx_ini[31,31] = -D_8/(2*H_8)
        struct[0].Fx_ini[32,32] = -1/T1d0_8
        struct[0].Fx_ini[33,33] = -1/T1q0_8
        struct[0].Fx_ini[34,34] = -1/T_r_8
        struct[0].Fx_ini[36,36] = -1/T_gov_1_8
        struct[0].Fx_ini[37,36] = 1/T_gov_3_8
        struct[0].Fx_ini[37,37] = -1/T_gov_3_8
        struct[0].Fx_ini[38,31] = 1/T_wo_8
        struct[0].Fx_ini[38,38] = -1/T_wo_8
        struct[0].Fx_ini[39,39] = -1/T_2_8

    if mode == 11:

        struct[0].Fy_ini[0,56] = -Omega_b_5 
        struct[0].Fy_ini[1,8] = (-i_d_5*sin(delta_5 - theta_5) - i_q_5*cos(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,9] = (V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,16] = (-2*R_a_5*i_d_5 - V_5*sin(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,17] = (-2*R_a_5*i_q_5 - V_5*cos(delta_5 - theta_5))/(2*H_5) 
        struct[0].Fy_ini[1,22] = 1/(2*H_5) 
        struct[0].Fy_ini[1,56] = D_5/(2*H_5) 
        struct[0].Fy_ini[2,16] = (X1d_5 - X_d_5)/T1d0_5 
        struct[0].Fy_ini[2,20] = 1/T1d0_5 
        struct[0].Fy_ini[3,17] = (-X1q_5 + X_q_5)/T1q0_5 
        struct[0].Fy_ini[4,8] = 1/T_r_5 
        struct[0].Fy_ini[5,8] = -1 
        struct[0].Fy_ini[6,21] = 1/T_gov_1_5 
        struct[0].Fy_ini[9,23] = 1/T_2_5 
        struct[0].Fy_ini[10,56] = -Omega_b_6 
        struct[0].Fy_ini[11,10] = (-i_d_6*sin(delta_6 - theta_6) - i_q_6*cos(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,11] = (V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,25] = (-2*R_a_6*i_d_6 - V_6*sin(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,26] = (-2*R_a_6*i_q_6 - V_6*cos(delta_6 - theta_6))/(2*H_6) 
        struct[0].Fy_ini[11,31] = 1/(2*H_6) 
        struct[0].Fy_ini[11,56] = D_6/(2*H_6) 
        struct[0].Fy_ini[12,25] = (X1d_6 - X_d_6)/T1d0_6 
        struct[0].Fy_ini[12,29] = 1/T1d0_6 
        struct[0].Fy_ini[13,26] = (-X1q_6 + X_q_6)/T1q0_6 
        struct[0].Fy_ini[14,10] = 1/T_r_6 
        struct[0].Fy_ini[15,10] = -1 
        struct[0].Fy_ini[16,30] = 1/T_gov_1_6 
        struct[0].Fy_ini[19,32] = 1/T_2_6 
        struct[0].Fy_ini[20,56] = -Omega_b_7 
        struct[0].Fy_ini[21,12] = (-i_d_7*sin(delta_7 - theta_7) - i_q_7*cos(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,13] = (V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,34] = (-2*R_a_7*i_d_7 - V_7*sin(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,35] = (-2*R_a_7*i_q_7 - V_7*cos(delta_7 - theta_7))/(2*H_7) 
        struct[0].Fy_ini[21,40] = 1/(2*H_7) 
        struct[0].Fy_ini[21,56] = D_7/(2*H_7) 
        struct[0].Fy_ini[22,34] = (X1d_7 - X_d_7)/T1d0_7 
        struct[0].Fy_ini[22,38] = 1/T1d0_7 
        struct[0].Fy_ini[23,35] = (-X1q_7 + X_q_7)/T1q0_7 
        struct[0].Fy_ini[24,12] = 1/T_r_7 
        struct[0].Fy_ini[25,12] = -1 
        struct[0].Fy_ini[26,39] = 1/T_gov_1_7 
        struct[0].Fy_ini[29,41] = 1/T_2_7 
        struct[0].Fy_ini[30,56] = -Omega_b_8 
        struct[0].Fy_ini[31,14] = (-i_d_8*sin(delta_8 - theta_8) - i_q_8*cos(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,15] = (V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,43] = (-2*R_a_8*i_d_8 - V_8*sin(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,44] = (-2*R_a_8*i_q_8 - V_8*cos(delta_8 - theta_8))/(2*H_8) 
        struct[0].Fy_ini[31,49] = 1/(2*H_8) 
        struct[0].Fy_ini[31,56] = D_8/(2*H_8) 
        struct[0].Fy_ini[32,43] = (X1d_8 - X_d_8)/T1d0_8 
        struct[0].Fy_ini[32,47] = 1/T1d0_8 
        struct[0].Fy_ini[33,44] = (-X1q_8 + X_q_8)/T1q0_8 
        struct[0].Fy_ini[34,14] = 1/T_r_8 
        struct[0].Fy_ini[35,14] = -1 
        struct[0].Fy_ini[36,48] = 1/T_gov_1_8 
        struct[0].Fy_ini[39,50] = 1/T_2_8 
        struct[0].Fy_ini[40,56] = -1 

        struct[0].Gy_ini[0,0] = 2*V_1*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[0,4] = V_1*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[0,5] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[0,6] = V_1*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[0,7] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[0,8] = V_1*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[0,9] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,0] = 2*V_1*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[1,4] = V_1*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[1,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[1,6] = V_1*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[1,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[1,8] = V_1*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[1,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[2,4] = V_2*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[2,5] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[2,6] = V_2*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[2,7] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[2,10] = V_2*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[2,11] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[3,4] = V_2*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[3,5] = V_2*V_3*(b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[3,6] = V_2*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[3,7] = V_2*V_4*(b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[3,10] = V_2*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[3,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[4,0] = V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[4,1] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[4,2] = V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[4,3] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[4,4] = V_1*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + 2*V_3*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[4,5] = V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[4,6] = V_3*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[4,7] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[4,12] = V_3*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[4,13] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[5,0] = V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy_ini[5,1] = V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy_ini[5,2] = V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy_ini[5,3] = V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy_ini[5,4] = V_1*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + 2*V_3*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[5,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[5,6] = V_3*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[5,7] = V_3*V_4*(b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[5,12] = V_3*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[5,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[6,0] = V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[6,1] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[6,2] = V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[6,3] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[6,4] = V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[6,5] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[6,6] = V_1*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + 2*V_4*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[6,7] = V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[6,14] = V_4*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[6,15] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[7,0] = V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy_ini[7,1] = V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy_ini[7,2] = V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy_ini[7,3] = V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy_ini[7,4] = V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy_ini[7,5] = V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy_ini[7,6] = V_1*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + 2*V_4*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[7,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[7,14] = V_4*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[7,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[8,0] = V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[8,1] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[8,8] = V_1*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + 2*V_5*g_5_1
        struct[0].Gy_ini[8,9] = V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[8,18] = -S_n_5/S_base
        struct[0].Gy_ini[9,0] = V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy_ini[9,1] = V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[9,8] = V_1*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + 2*V_5*(-b_5_1 - bs_5_1/2)
        struct[0].Gy_ini[9,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy_ini[9,19] = -S_n_5/S_base
        struct[0].Gy_ini[10,2] = V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[10,3] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[10,10] = V_2*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + 2*V_6*g_6_2
        struct[0].Gy_ini[10,11] = V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[10,27] = -S_n_6/S_base
        struct[0].Gy_ini[11,2] = V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy_ini[11,3] = V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[11,10] = V_2*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + 2*V_6*(-b_6_2 - bs_6_2/2)
        struct[0].Gy_ini[11,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy_ini[11,28] = -S_n_6/S_base
        struct[0].Gy_ini[12,4] = V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[12,5] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[12,12] = V_3*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + 2*V_7*g_7_3
        struct[0].Gy_ini[12,13] = V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[12,36] = -S_n_7/S_base
        struct[0].Gy_ini[13,4] = V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy_ini[13,5] = V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[13,12] = V_3*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + 2*V_7*(-b_7_3 - bs_7_3/2)
        struct[0].Gy_ini[13,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy_ini[13,37] = -S_n_7/S_base
        struct[0].Gy_ini[14,6] = V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[14,7] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[14,14] = V_4*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + 2*V_8*g_8_4
        struct[0].Gy_ini[14,15] = V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[14,45] = -S_n_8/S_base
        struct[0].Gy_ini[15,6] = V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy_ini[15,7] = V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[15,14] = V_4*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + 2*V_8*(-b_8_4 - bs_8_4/2)
        struct[0].Gy_ini[15,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy_ini[15,46] = -S_n_8/S_base
        struct[0].Gy_ini[16,8] = cos(delta_5 - theta_5)
        struct[0].Gy_ini[16,9] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[16,16] = X1d_5
        struct[0].Gy_ini[16,17] = R_a_5
        struct[0].Gy_ini[17,8] = sin(delta_5 - theta_5)
        struct[0].Gy_ini[17,9] = -V_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[17,16] = R_a_5
        struct[0].Gy_ini[17,17] = -X1q_5
        struct[0].Gy_ini[18,8] = i_d_5*sin(delta_5 - theta_5) + i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[18,9] = -V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[18,16] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[18,17] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[18,18] = -1
        struct[0].Gy_ini[19,8] = i_d_5*cos(delta_5 - theta_5) - i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[19,9] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[19,16] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy_ini[19,17] = -V_5*sin(delta_5 - theta_5)
        struct[0].Gy_ini[19,19] = -1
        struct[0].Gy_ini[20,20] = -1
        struct[0].Gy_ini[20,24] = K_a_5
        struct[0].Gy_ini[21,21] = -1
        struct[0].Gy_ini[21,52] = 1
        struct[0].Gy_ini[22,22] = -1
        struct[0].Gy_ini[23,23] = -1
        struct[0].Gy_ini[24,23] = K_stab_5*T_1_5/T_2_5
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[25,10] = cos(delta_6 - theta_6)
        struct[0].Gy_ini[25,11] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[25,25] = X1d_6
        struct[0].Gy_ini[25,26] = R_a_6
        struct[0].Gy_ini[26,10] = sin(delta_6 - theta_6)
        struct[0].Gy_ini[26,11] = -V_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[26,25] = R_a_6
        struct[0].Gy_ini[26,26] = -X1q_6
        struct[0].Gy_ini[27,10] = i_d_6*sin(delta_6 - theta_6) + i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[27,11] = -V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[27,25] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[27,26] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[27,27] = -1
        struct[0].Gy_ini[28,10] = i_d_6*cos(delta_6 - theta_6) - i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[28,11] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[28,25] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy_ini[28,26] = -V_6*sin(delta_6 - theta_6)
        struct[0].Gy_ini[28,28] = -1
        struct[0].Gy_ini[29,29] = -1
        struct[0].Gy_ini[29,33] = K_a_6
        struct[0].Gy_ini[30,30] = -1
        struct[0].Gy_ini[30,53] = 1
        struct[0].Gy_ini[31,31] = -1
        struct[0].Gy_ini[32,32] = -1
        struct[0].Gy_ini[33,32] = K_stab_6*T_1_6/T_2_6
        struct[0].Gy_ini[33,33] = -1
        struct[0].Gy_ini[34,12] = cos(delta_7 - theta_7)
        struct[0].Gy_ini[34,13] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[34,34] = X1d_7
        struct[0].Gy_ini[34,35] = R_a_7
        struct[0].Gy_ini[35,12] = sin(delta_7 - theta_7)
        struct[0].Gy_ini[35,13] = -V_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[35,34] = R_a_7
        struct[0].Gy_ini[35,35] = -X1q_7
        struct[0].Gy_ini[36,12] = i_d_7*sin(delta_7 - theta_7) + i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[36,13] = -V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[36,34] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[36,35] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[36,36] = -1
        struct[0].Gy_ini[37,12] = i_d_7*cos(delta_7 - theta_7) - i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[37,13] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[37,34] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy_ini[37,35] = -V_7*sin(delta_7 - theta_7)
        struct[0].Gy_ini[37,37] = -1
        struct[0].Gy_ini[38,38] = -1
        struct[0].Gy_ini[38,42] = K_a_7
        struct[0].Gy_ini[39,39] = -1
        struct[0].Gy_ini[39,54] = 1
        struct[0].Gy_ini[40,40] = -1
        struct[0].Gy_ini[41,41] = -1
        struct[0].Gy_ini[42,41] = K_stab_7*T_1_7/T_2_7
        struct[0].Gy_ini[42,42] = -1
        struct[0].Gy_ini[43,14] = cos(delta_8 - theta_8)
        struct[0].Gy_ini[43,15] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[43,43] = X1d_8
        struct[0].Gy_ini[43,44] = R_a_8
        struct[0].Gy_ini[44,14] = sin(delta_8 - theta_8)
        struct[0].Gy_ini[44,15] = -V_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[44,43] = R_a_8
        struct[0].Gy_ini[44,44] = -X1q_8
        struct[0].Gy_ini[45,14] = i_d_8*sin(delta_8 - theta_8) + i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[45,15] = -V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[45,43] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[45,44] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[45,45] = -1
        struct[0].Gy_ini[46,14] = i_d_8*cos(delta_8 - theta_8) - i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[46,15] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[46,43] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy_ini[46,44] = -V_8*sin(delta_8 - theta_8)
        struct[0].Gy_ini[46,46] = -1
        struct[0].Gy_ini[47,47] = -1
        struct[0].Gy_ini[47,51] = K_a_8
        struct[0].Gy_ini[48,48] = -1
        struct[0].Gy_ini[48,55] = 1
        struct[0].Gy_ini[49,49] = -1
        struct[0].Gy_ini[50,50] = -1
        struct[0].Gy_ini[51,50] = K_stab_8*T_1_8/T_2_8
        struct[0].Gy_ini[51,51] = -1
        struct[0].Gy_ini[52,52] = -1
        struct[0].Gy_ini[53,53] = -1
        struct[0].Gy_ini[54,54] = -1
        struct[0].Gy_ini[55,55] = -1
        struct[0].Gy_ini[56,56] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_2 = struct[0].g_1_2
    b_1_2 = struct[0].b_1_2
    bs_1_2 = struct[0].bs_1_2
    g_3_4 = struct[0].g_3_4
    b_3_4 = struct[0].b_3_4
    bs_3_4 = struct[0].bs_3_4
    g_1_3 = struct[0].g_1_3
    b_1_3 = struct[0].b_1_3
    bs_1_3 = struct[0].bs_1_3
    g_2_4 = struct[0].g_2_4
    b_2_4 = struct[0].b_2_4
    bs_2_4 = struct[0].bs_2_4
    g_1_4 = struct[0].g_1_4
    b_1_4 = struct[0].b_1_4
    bs_1_4 = struct[0].bs_1_4
    g_3_2 = struct[0].g_3_2
    b_3_2 = struct[0].b_3_2
    bs_3_2 = struct[0].bs_3_2
    g_5_1 = struct[0].g_5_1
    b_5_1 = struct[0].b_5_1
    bs_5_1 = struct[0].bs_5_1
    g_6_2 = struct[0].g_6_2
    b_6_2 = struct[0].b_6_2
    bs_6_2 = struct[0].bs_6_2
    g_7_3 = struct[0].g_7_3
    b_7_3 = struct[0].b_7_3
    bs_7_3 = struct[0].bs_7_3
    g_8_4 = struct[0].g_8_4
    b_8_4 = struct[0].b_8_4
    bs_8_4 = struct[0].bs_8_4
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
    U_4_n = struct[0].U_4_n
    U_5_n = struct[0].U_5_n
    U_6_n = struct[0].U_6_n
    U_7_n = struct[0].U_7_n
    U_8_n = struct[0].U_8_n
    S_n_5 = struct[0].S_n_5
    H_5 = struct[0].H_5
    Omega_b_5 = struct[0].Omega_b_5
    T1d0_5 = struct[0].T1d0_5
    T1q0_5 = struct[0].T1q0_5
    X_d_5 = struct[0].X_d_5
    X_q_5 = struct[0].X_q_5
    X1d_5 = struct[0].X1d_5
    X1q_5 = struct[0].X1q_5
    D_5 = struct[0].D_5
    R_a_5 = struct[0].R_a_5
    K_delta_5 = struct[0].K_delta_5
    K_a_5 = struct[0].K_a_5
    K_ai_5 = struct[0].K_ai_5
    T_r_5 = struct[0].T_r_5
    Droop_5 = struct[0].Droop_5
    T_gov_1_5 = struct[0].T_gov_1_5
    T_gov_2_5 = struct[0].T_gov_2_5
    T_gov_3_5 = struct[0].T_gov_3_5
    omega_ref_5 = struct[0].omega_ref_5
    T_wo_5 = struct[0].T_wo_5
    T_1_5 = struct[0].T_1_5
    T_2_5 = struct[0].T_2_5
    K_stab_5 = struct[0].K_stab_5
    S_n_6 = struct[0].S_n_6
    H_6 = struct[0].H_6
    Omega_b_6 = struct[0].Omega_b_6
    T1d0_6 = struct[0].T1d0_6
    T1q0_6 = struct[0].T1q0_6
    X_d_6 = struct[0].X_d_6
    X_q_6 = struct[0].X_q_6
    X1d_6 = struct[0].X1d_6
    X1q_6 = struct[0].X1q_6
    D_6 = struct[0].D_6
    R_a_6 = struct[0].R_a_6
    K_delta_6 = struct[0].K_delta_6
    K_a_6 = struct[0].K_a_6
    K_ai_6 = struct[0].K_ai_6
    T_r_6 = struct[0].T_r_6
    Droop_6 = struct[0].Droop_6
    T_gov_1_6 = struct[0].T_gov_1_6
    T_gov_2_6 = struct[0].T_gov_2_6
    T_gov_3_6 = struct[0].T_gov_3_6
    omega_ref_6 = struct[0].omega_ref_6
    T_wo_6 = struct[0].T_wo_6
    T_1_6 = struct[0].T_1_6
    T_2_6 = struct[0].T_2_6
    K_stab_6 = struct[0].K_stab_6
    S_n_7 = struct[0].S_n_7
    H_7 = struct[0].H_7
    Omega_b_7 = struct[0].Omega_b_7
    T1d0_7 = struct[0].T1d0_7
    T1q0_7 = struct[0].T1q0_7
    X_d_7 = struct[0].X_d_7
    X_q_7 = struct[0].X_q_7
    X1d_7 = struct[0].X1d_7
    X1q_7 = struct[0].X1q_7
    D_7 = struct[0].D_7
    R_a_7 = struct[0].R_a_7
    K_delta_7 = struct[0].K_delta_7
    K_a_7 = struct[0].K_a_7
    K_ai_7 = struct[0].K_ai_7
    T_r_7 = struct[0].T_r_7
    Droop_7 = struct[0].Droop_7
    T_gov_1_7 = struct[0].T_gov_1_7
    T_gov_2_7 = struct[0].T_gov_2_7
    T_gov_3_7 = struct[0].T_gov_3_7
    omega_ref_7 = struct[0].omega_ref_7
    T_wo_7 = struct[0].T_wo_7
    T_1_7 = struct[0].T_1_7
    T_2_7 = struct[0].T_2_7
    K_stab_7 = struct[0].K_stab_7
    S_n_8 = struct[0].S_n_8
    H_8 = struct[0].H_8
    Omega_b_8 = struct[0].Omega_b_8
    T1d0_8 = struct[0].T1d0_8
    T1q0_8 = struct[0].T1q0_8
    X_d_8 = struct[0].X_d_8
    X_q_8 = struct[0].X_q_8
    X1d_8 = struct[0].X1d_8
    X1q_8 = struct[0].X1q_8
    D_8 = struct[0].D_8
    R_a_8 = struct[0].R_a_8
    K_delta_8 = struct[0].K_delta_8
    K_a_8 = struct[0].K_a_8
    K_ai_8 = struct[0].K_ai_8
    T_r_8 = struct[0].T_r_8
    Droop_8 = struct[0].Droop_8
    T_gov_1_8 = struct[0].T_gov_1_8
    T_gov_2_8 = struct[0].T_gov_2_8
    T_gov_3_8 = struct[0].T_gov_3_8
    omega_ref_8 = struct[0].omega_ref_8
    T_wo_8 = struct[0].T_wo_8
    T_1_8 = struct[0].T_1_8
    T_2_8 = struct[0].T_2_8
    K_stab_8 = struct[0].K_stab_8
    K_sec_5 = struct[0].K_sec_5
    K_sec_6 = struct[0].K_sec_6
    K_sec_7 = struct[0].K_sec_7
    K_sec_8 = struct[0].K_sec_8
    
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
    v_ref_5 = struct[0].v_ref_5
    v_pss_5 = struct[0].v_pss_5
    p_c_5 = struct[0].p_c_5
    v_ref_6 = struct[0].v_ref_6
    v_pss_6 = struct[0].v_pss_6
    p_c_6 = struct[0].p_c_6
    v_ref_7 = struct[0].v_ref_7
    v_pss_7 = struct[0].v_pss_7
    p_c_7 = struct[0].p_c_7
    v_ref_8 = struct[0].v_ref_8
    v_pss_8 = struct[0].v_pss_8
    p_c_8 = struct[0].p_c_8
    
    # Dynamical states:
    delta_5 = struct[0].x[0,0]
    omega_5 = struct[0].x[1,0]
    e1q_5 = struct[0].x[2,0]
    e1d_5 = struct[0].x[3,0]
    v_c_5 = struct[0].x[4,0]
    xi_v_5 = struct[0].x[5,0]
    x_gov_1_5 = struct[0].x[6,0]
    x_gov_2_5 = struct[0].x[7,0]
    x_wo_5 = struct[0].x[8,0]
    x_lead_5 = struct[0].x[9,0]
    delta_6 = struct[0].x[10,0]
    omega_6 = struct[0].x[11,0]
    e1q_6 = struct[0].x[12,0]
    e1d_6 = struct[0].x[13,0]
    v_c_6 = struct[0].x[14,0]
    xi_v_6 = struct[0].x[15,0]
    x_gov_1_6 = struct[0].x[16,0]
    x_gov_2_6 = struct[0].x[17,0]
    x_wo_6 = struct[0].x[18,0]
    x_lead_6 = struct[0].x[19,0]
    delta_7 = struct[0].x[20,0]
    omega_7 = struct[0].x[21,0]
    e1q_7 = struct[0].x[22,0]
    e1d_7 = struct[0].x[23,0]
    v_c_7 = struct[0].x[24,0]
    xi_v_7 = struct[0].x[25,0]
    x_gov_1_7 = struct[0].x[26,0]
    x_gov_2_7 = struct[0].x[27,0]
    x_wo_7 = struct[0].x[28,0]
    x_lead_7 = struct[0].x[29,0]
    delta_8 = struct[0].x[30,0]
    omega_8 = struct[0].x[31,0]
    e1q_8 = struct[0].x[32,0]
    e1d_8 = struct[0].x[33,0]
    v_c_8 = struct[0].x[34,0]
    xi_v_8 = struct[0].x[35,0]
    x_gov_1_8 = struct[0].x[36,0]
    x_gov_2_8 = struct[0].x[37,0]
    x_wo_8 = struct[0].x[38,0]
    x_lead_8 = struct[0].x[39,0]
    xi_freq = struct[0].x[40,0]
    
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
    i_d_5 = struct[0].y_run[16,0]
    i_q_5 = struct[0].y_run[17,0]
    p_g_5_1 = struct[0].y_run[18,0]
    q_g_5_1 = struct[0].y_run[19,0]
    v_f_5 = struct[0].y_run[20,0]
    p_m_ref_5 = struct[0].y_run[21,0]
    p_m_5 = struct[0].y_run[22,0]
    z_wo_5 = struct[0].y_run[23,0]
    v_pss_5 = struct[0].y_run[24,0]
    i_d_6 = struct[0].y_run[25,0]
    i_q_6 = struct[0].y_run[26,0]
    p_g_6_1 = struct[0].y_run[27,0]
    q_g_6_1 = struct[0].y_run[28,0]
    v_f_6 = struct[0].y_run[29,0]
    p_m_ref_6 = struct[0].y_run[30,0]
    p_m_6 = struct[0].y_run[31,0]
    z_wo_6 = struct[0].y_run[32,0]
    v_pss_6 = struct[0].y_run[33,0]
    i_d_7 = struct[0].y_run[34,0]
    i_q_7 = struct[0].y_run[35,0]
    p_g_7_1 = struct[0].y_run[36,0]
    q_g_7_1 = struct[0].y_run[37,0]
    v_f_7 = struct[0].y_run[38,0]
    p_m_ref_7 = struct[0].y_run[39,0]
    p_m_7 = struct[0].y_run[40,0]
    z_wo_7 = struct[0].y_run[41,0]
    v_pss_7 = struct[0].y_run[42,0]
    i_d_8 = struct[0].y_run[43,0]
    i_q_8 = struct[0].y_run[44,0]
    p_g_8_1 = struct[0].y_run[45,0]
    q_g_8_1 = struct[0].y_run[46,0]
    v_f_8 = struct[0].y_run[47,0]
    p_m_ref_8 = struct[0].y_run[48,0]
    p_m_8 = struct[0].y_run[49,0]
    z_wo_8 = struct[0].y_run[50,0]
    v_pss_8 = struct[0].y_run[51,0]
    p_r_5 = struct[0].y_run[52,0]
    p_r_6 = struct[0].y_run[53,0]
    p_r_7 = struct[0].y_run[54,0]
    p_r_8 = struct[0].y_run[55,0]
    omega_coi = struct[0].y_run[56,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_5*delta_5 + Omega_b_5*(omega_5 - omega_coi)
        struct[0].f[1,0] = (-D_5*(omega_5 - omega_coi) - i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) - i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5)) + p_m_5)/(2*H_5)
        struct[0].f[2,0] = (-e1q_5 - i_d_5*(-X1d_5 + X_d_5) + v_f_5)/T1d0_5
        struct[0].f[3,0] = (-e1d_5 + i_q_5*(-X1q_5 + X_q_5))/T1q0_5
        struct[0].f[4,0] = (V_5 - v_c_5)/T_r_5
        struct[0].f[5,0] = -V_5 + v_ref_5
        struct[0].f[6,0] = (p_m_ref_5 - x_gov_1_5)/T_gov_1_5
        struct[0].f[7,0] = (x_gov_1_5 - x_gov_2_5)/T_gov_3_5
        struct[0].f[8,0] = (omega_5 - x_wo_5 - 1.0)/T_wo_5
        struct[0].f[9,0] = (-x_lead_5 + z_wo_5)/T_2_5
        struct[0].f[10,0] = -K_delta_6*delta_6 + Omega_b_6*(omega_6 - omega_coi)
        struct[0].f[11,0] = (-D_6*(omega_6 - omega_coi) - i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) - i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6)) + p_m_6)/(2*H_6)
        struct[0].f[12,0] = (-e1q_6 - i_d_6*(-X1d_6 + X_d_6) + v_f_6)/T1d0_6
        struct[0].f[13,0] = (-e1d_6 + i_q_6*(-X1q_6 + X_q_6))/T1q0_6
        struct[0].f[14,0] = (V_6 - v_c_6)/T_r_6
        struct[0].f[15,0] = -V_6 + v_ref_6
        struct[0].f[16,0] = (p_m_ref_6 - x_gov_1_6)/T_gov_1_6
        struct[0].f[17,0] = (x_gov_1_6 - x_gov_2_6)/T_gov_3_6
        struct[0].f[18,0] = (omega_6 - x_wo_6 - 1.0)/T_wo_6
        struct[0].f[19,0] = (-x_lead_6 + z_wo_6)/T_2_6
        struct[0].f[20,0] = -K_delta_7*delta_7 + Omega_b_7*(omega_7 - omega_coi)
        struct[0].f[21,0] = (-D_7*(omega_7 - omega_coi) - i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) - i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7)) + p_m_7)/(2*H_7)
        struct[0].f[22,0] = (-e1q_7 - i_d_7*(-X1d_7 + X_d_7) + v_f_7)/T1d0_7
        struct[0].f[23,0] = (-e1d_7 + i_q_7*(-X1q_7 + X_q_7))/T1q0_7
        struct[0].f[24,0] = (V_7 - v_c_7)/T_r_7
        struct[0].f[25,0] = -V_7 + v_ref_7
        struct[0].f[26,0] = (p_m_ref_7 - x_gov_1_7)/T_gov_1_7
        struct[0].f[27,0] = (x_gov_1_7 - x_gov_2_7)/T_gov_3_7
        struct[0].f[28,0] = (omega_7 - x_wo_7 - 1.0)/T_wo_7
        struct[0].f[29,0] = (-x_lead_7 + z_wo_7)/T_2_7
        struct[0].f[30,0] = -K_delta_8*delta_8 + Omega_b_8*(omega_8 - omega_coi)
        struct[0].f[31,0] = (-D_8*(omega_8 - omega_coi) - i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) - i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8)) + p_m_8)/(2*H_8)
        struct[0].f[32,0] = (-e1q_8 - i_d_8*(-X1d_8 + X_d_8) + v_f_8)/T1d0_8
        struct[0].f[33,0] = (-e1d_8 + i_q_8*(-X1q_8 + X_q_8))/T1q0_8
        struct[0].f[34,0] = (V_8 - v_c_8)/T_r_8
        struct[0].f[35,0] = -V_8 + v_ref_8
        struct[0].f[36,0] = (p_m_ref_8 - x_gov_1_8)/T_gov_1_8
        struct[0].f[37,0] = (x_gov_1_8 - x_gov_2_8)/T_gov_3_8
        struct[0].f[38,0] = (omega_8 - x_wo_8 - 1.0)/T_wo_8
        struct[0].f[39,0] = (-x_lead_8 + z_wo_8)/T_2_8
        struct[0].f[40,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_1/S_base + V_1**2*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].g[4,0] = -P_3/S_base + V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3**2*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].g[5,0] = -Q_3/S_base + V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_3**2*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].g[6,0] = -P_4/S_base + V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4**2*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].g[7,0] = -Q_4/S_base + V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_4**2*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].g[8,0] = -P_5/S_base + V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + V_5**2*g_5_1 - S_n_5*p_g_5_1/S_base
        struct[0].g[9,0] = -Q_5/S_base + V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + V_5**2*(-b_5_1 - bs_5_1/2) - S_n_5*q_g_5_1/S_base
        struct[0].g[10,0] = -P_6/S_base + V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + V_6**2*g_6_2 - S_n_6*p_g_6_1/S_base
        struct[0].g[11,0] = -Q_6/S_base + V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + V_6**2*(-b_6_2 - bs_6_2/2) - S_n_6*q_g_6_1/S_base
        struct[0].g[12,0] = -P_7/S_base + V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + V_7**2*g_7_3 - S_n_7*p_g_7_1/S_base
        struct[0].g[13,0] = -Q_7/S_base + V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + V_7**2*(-b_7_3 - bs_7_3/2) - S_n_7*q_g_7_1/S_base
        struct[0].g[14,0] = -P_8/S_base + V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + V_8**2*g_8_4 - S_n_8*p_g_8_1/S_base
        struct[0].g[15,0] = -Q_8/S_base + V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + V_8**2*(-b_8_4 - bs_8_4/2) - S_n_8*q_g_8_1/S_base
        struct[0].g[16,0] = R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5) + X1d_5*i_d_5 - e1q_5
        struct[0].g[17,0] = R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5) - X1q_5*i_q_5 - e1d_5
        struct[0].g[18,0] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5) - p_g_5_1
        struct[0].g[19,0] = V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5) - q_g_5_1
        struct[0].g[20,0] = K_a_5*(-v_c_5 + v_pss_5 + v_ref_5) + K_ai_5*xi_v_5 - v_f_5
        struct[0].g[21,0] = p_c_5 - p_m_ref_5 + p_r_5 - (omega_5 - omega_ref_5)/Droop_5
        struct[0].g[22,0] = T_gov_2_5*(x_gov_1_5 - x_gov_2_5)/T_gov_3_5 - p_m_5 + x_gov_2_5
        struct[0].g[23,0] = omega_5 - x_wo_5 - z_wo_5 - 1.0
        struct[0].g[24,0] = K_stab_5*(T_1_5*(-x_lead_5 + z_wo_5)/T_2_5 + x_lead_5) - v_pss_5
        struct[0].g[25,0] = R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6) + X1d_6*i_d_6 - e1q_6
        struct[0].g[26,0] = R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6) - X1q_6*i_q_6 - e1d_6
        struct[0].g[27,0] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6) - p_g_6_1
        struct[0].g[28,0] = V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6) - q_g_6_1
        struct[0].g[29,0] = K_a_6*(-v_c_6 + v_pss_6 + v_ref_6) + K_ai_6*xi_v_6 - v_f_6
        struct[0].g[30,0] = p_c_6 - p_m_ref_6 + p_r_6 - (omega_6 - omega_ref_6)/Droop_6
        struct[0].g[31,0] = T_gov_2_6*(x_gov_1_6 - x_gov_2_6)/T_gov_3_6 - p_m_6 + x_gov_2_6
        struct[0].g[32,0] = omega_6 - x_wo_6 - z_wo_6 - 1.0
        struct[0].g[33,0] = K_stab_6*(T_1_6*(-x_lead_6 + z_wo_6)/T_2_6 + x_lead_6) - v_pss_6
        struct[0].g[34,0] = R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7) + X1d_7*i_d_7 - e1q_7
        struct[0].g[35,0] = R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7) - X1q_7*i_q_7 - e1d_7
        struct[0].g[36,0] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7) - p_g_7_1
        struct[0].g[37,0] = V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7) - q_g_7_1
        struct[0].g[38,0] = K_a_7*(-v_c_7 + v_pss_7 + v_ref_7) + K_ai_7*xi_v_7 - v_f_7
        struct[0].g[39,0] = p_c_7 - p_m_ref_7 + p_r_7 - (omega_7 - omega_ref_7)/Droop_7
        struct[0].g[40,0] = T_gov_2_7*(x_gov_1_7 - x_gov_2_7)/T_gov_3_7 - p_m_7 + x_gov_2_7
        struct[0].g[41,0] = omega_7 - x_wo_7 - z_wo_7 - 1.0
        struct[0].g[42,0] = K_stab_7*(T_1_7*(-x_lead_7 + z_wo_7)/T_2_7 + x_lead_7) - v_pss_7
        struct[0].g[43,0] = R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8) + X1d_8*i_d_8 - e1q_8
        struct[0].g[44,0] = R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8) - X1q_8*i_q_8 - e1d_8
        struct[0].g[45,0] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8) - p_g_8_1
        struct[0].g[46,0] = V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8) - q_g_8_1
        struct[0].g[47,0] = K_a_8*(-v_c_8 + v_pss_8 + v_ref_8) + K_ai_8*xi_v_8 - v_f_8
        struct[0].g[48,0] = p_c_8 - p_m_ref_8 + p_r_8 - (omega_8 - omega_ref_8)/Droop_8
        struct[0].g[49,0] = T_gov_2_8*(x_gov_1_8 - x_gov_2_8)/T_gov_3_8 - p_m_8 + x_gov_2_8
        struct[0].g[50,0] = omega_8 - x_wo_8 - z_wo_8 - 1.0
        struct[0].g[51,0] = K_stab_8*(T_1_8*(-x_lead_8 + z_wo_8)/T_2_8 + x_lead_8) - v_pss_8
        struct[0].g[52,0] = K_sec_5*xi_freq/4 - p_r_5
        struct[0].g[53,0] = K_sec_6*xi_freq/4 - p_r_6
        struct[0].g[54,0] = K_sec_7*xi_freq/4 - p_r_7
        struct[0].g[55,0] = K_sec_8*xi_freq/4 - p_r_8
        struct[0].g[56,0] = omega_5/4 + omega_6/4 + omega_7/4 + omega_8/4 - omega_coi
    
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
        struct[0].h[8,0] = i_d_5*(R_a_5*i_d_5 + V_5*sin(delta_5 - theta_5)) + i_q_5*(R_a_5*i_q_5 + V_5*cos(delta_5 - theta_5))
        struct[0].h[9,0] = i_d_6*(R_a_6*i_d_6 + V_6*sin(delta_6 - theta_6)) + i_q_6*(R_a_6*i_q_6 + V_6*cos(delta_6 - theta_6))
        struct[0].h[10,0] = i_d_7*(R_a_7*i_d_7 + V_7*sin(delta_7 - theta_7)) + i_q_7*(R_a_7*i_q_7 + V_7*cos(delta_7 - theta_7))
        struct[0].h[11,0] = i_d_8*(R_a_8*i_d_8 + V_8*sin(delta_8 - theta_8)) + i_q_8*(R_a_8*i_q_8 + V_8*cos(delta_8 - theta_8))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_5
        struct[0].Fx[0,1] = Omega_b_5
        struct[0].Fx[1,0] = (-V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fx[1,1] = -D_5/(2*H_5)
        struct[0].Fx[2,2] = -1/T1d0_5
        struct[0].Fx[3,3] = -1/T1q0_5
        struct[0].Fx[4,4] = -1/T_r_5
        struct[0].Fx[6,6] = -1/T_gov_1_5
        struct[0].Fx[7,6] = 1/T_gov_3_5
        struct[0].Fx[7,7] = -1/T_gov_3_5
        struct[0].Fx[8,1] = 1/T_wo_5
        struct[0].Fx[8,8] = -1/T_wo_5
        struct[0].Fx[9,9] = -1/T_2_5
        struct[0].Fx[10,10] = -K_delta_6
        struct[0].Fx[10,11] = Omega_b_6
        struct[0].Fx[11,10] = (-V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fx[11,11] = -D_6/(2*H_6)
        struct[0].Fx[12,12] = -1/T1d0_6
        struct[0].Fx[13,13] = -1/T1q0_6
        struct[0].Fx[14,14] = -1/T_r_6
        struct[0].Fx[16,16] = -1/T_gov_1_6
        struct[0].Fx[17,16] = 1/T_gov_3_6
        struct[0].Fx[17,17] = -1/T_gov_3_6
        struct[0].Fx[18,11] = 1/T_wo_6
        struct[0].Fx[18,18] = -1/T_wo_6
        struct[0].Fx[19,19] = -1/T_2_6
        struct[0].Fx[20,20] = -K_delta_7
        struct[0].Fx[20,21] = Omega_b_7
        struct[0].Fx[21,20] = (-V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fx[21,21] = -D_7/(2*H_7)
        struct[0].Fx[22,22] = -1/T1d0_7
        struct[0].Fx[23,23] = -1/T1q0_7
        struct[0].Fx[24,24] = -1/T_r_7
        struct[0].Fx[26,26] = -1/T_gov_1_7
        struct[0].Fx[27,26] = 1/T_gov_3_7
        struct[0].Fx[27,27] = -1/T_gov_3_7
        struct[0].Fx[28,21] = 1/T_wo_7
        struct[0].Fx[28,28] = -1/T_wo_7
        struct[0].Fx[29,29] = -1/T_2_7
        struct[0].Fx[30,30] = -K_delta_8
        struct[0].Fx[30,31] = Omega_b_8
        struct[0].Fx[31,30] = (-V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fx[31,31] = -D_8/(2*H_8)
        struct[0].Fx[32,32] = -1/T1d0_8
        struct[0].Fx[33,33] = -1/T1q0_8
        struct[0].Fx[34,34] = -1/T_r_8
        struct[0].Fx[36,36] = -1/T_gov_1_8
        struct[0].Fx[37,36] = 1/T_gov_3_8
        struct[0].Fx[37,37] = -1/T_gov_3_8
        struct[0].Fx[38,31] = 1/T_wo_8
        struct[0].Fx[38,38] = -1/T_wo_8
        struct[0].Fx[39,39] = -1/T_2_8

    if mode == 11:

        struct[0].Fy[0,56] = -Omega_b_5
        struct[0].Fy[1,8] = (-i_d_5*sin(delta_5 - theta_5) - i_q_5*cos(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,9] = (V_5*i_d_5*cos(delta_5 - theta_5) - V_5*i_q_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,16] = (-2*R_a_5*i_d_5 - V_5*sin(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,17] = (-2*R_a_5*i_q_5 - V_5*cos(delta_5 - theta_5))/(2*H_5)
        struct[0].Fy[1,22] = 1/(2*H_5)
        struct[0].Fy[1,56] = D_5/(2*H_5)
        struct[0].Fy[2,16] = (X1d_5 - X_d_5)/T1d0_5
        struct[0].Fy[2,20] = 1/T1d0_5
        struct[0].Fy[3,17] = (-X1q_5 + X_q_5)/T1q0_5
        struct[0].Fy[4,8] = 1/T_r_5
        struct[0].Fy[5,8] = -1
        struct[0].Fy[6,21] = 1/T_gov_1_5
        struct[0].Fy[9,23] = 1/T_2_5
        struct[0].Fy[10,56] = -Omega_b_6
        struct[0].Fy[11,10] = (-i_d_6*sin(delta_6 - theta_6) - i_q_6*cos(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,11] = (V_6*i_d_6*cos(delta_6 - theta_6) - V_6*i_q_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,25] = (-2*R_a_6*i_d_6 - V_6*sin(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,26] = (-2*R_a_6*i_q_6 - V_6*cos(delta_6 - theta_6))/(2*H_6)
        struct[0].Fy[11,31] = 1/(2*H_6)
        struct[0].Fy[11,56] = D_6/(2*H_6)
        struct[0].Fy[12,25] = (X1d_6 - X_d_6)/T1d0_6
        struct[0].Fy[12,29] = 1/T1d0_6
        struct[0].Fy[13,26] = (-X1q_6 + X_q_6)/T1q0_6
        struct[0].Fy[14,10] = 1/T_r_6
        struct[0].Fy[15,10] = -1
        struct[0].Fy[16,30] = 1/T_gov_1_6
        struct[0].Fy[19,32] = 1/T_2_6
        struct[0].Fy[20,56] = -Omega_b_7
        struct[0].Fy[21,12] = (-i_d_7*sin(delta_7 - theta_7) - i_q_7*cos(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,13] = (V_7*i_d_7*cos(delta_7 - theta_7) - V_7*i_q_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,34] = (-2*R_a_7*i_d_7 - V_7*sin(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,35] = (-2*R_a_7*i_q_7 - V_7*cos(delta_7 - theta_7))/(2*H_7)
        struct[0].Fy[21,40] = 1/(2*H_7)
        struct[0].Fy[21,56] = D_7/(2*H_7)
        struct[0].Fy[22,34] = (X1d_7 - X_d_7)/T1d0_7
        struct[0].Fy[22,38] = 1/T1d0_7
        struct[0].Fy[23,35] = (-X1q_7 + X_q_7)/T1q0_7
        struct[0].Fy[24,12] = 1/T_r_7
        struct[0].Fy[25,12] = -1
        struct[0].Fy[26,39] = 1/T_gov_1_7
        struct[0].Fy[29,41] = 1/T_2_7
        struct[0].Fy[30,56] = -Omega_b_8
        struct[0].Fy[31,14] = (-i_d_8*sin(delta_8 - theta_8) - i_q_8*cos(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,15] = (V_8*i_d_8*cos(delta_8 - theta_8) - V_8*i_q_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,43] = (-2*R_a_8*i_d_8 - V_8*sin(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,44] = (-2*R_a_8*i_q_8 - V_8*cos(delta_8 - theta_8))/(2*H_8)
        struct[0].Fy[31,49] = 1/(2*H_8)
        struct[0].Fy[31,56] = D_8/(2*H_8)
        struct[0].Fy[32,43] = (X1d_8 - X_d_8)/T1d0_8
        struct[0].Fy[32,47] = 1/T1d0_8
        struct[0].Fy[33,44] = (-X1q_8 + X_q_8)/T1q0_8
        struct[0].Fy[34,14] = 1/T_r_8
        struct[0].Fy[35,14] = -1
        struct[0].Fy[36,48] = 1/T_gov_1_8
        struct[0].Fy[39,50] = 1/T_2_8
        struct[0].Fy[40,56] = -1

        struct[0].Gy[0,0] = 2*V_1*(g_1_2 + g_1_3 + g_1_4 + g_5_1) + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[0,4] = V_1*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[0,5] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[0,6] = V_1*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[0,7] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[0,8] = V_1*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[0,9] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[1,0] = 2*V_1*(-b_1_2 - b_1_3 - b_1_4 - b_5_1 - bs_1_2/2 - bs_1_3/2 - bs_1_4/2 - bs_5_1/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_3*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_4*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_5*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[1,4] = V_1*(b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[1,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[1,6] = V_1*(b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[1,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[1,8] = V_1*(b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[1,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_4 + g_3_2 + g_6_2) + V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[2,4] = V_2*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[2,5] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[2,6] = V_2*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[2,7] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[2,10] = V_2*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[2,11] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_4 - b_3_2 - b_6_2 - bs_1_2/2 - bs_2_4/2 - bs_3_2/2 - bs_6_2/2) + V_3*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_4*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_6*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[3,4] = V_2*(b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[3,5] = V_2*V_3*(b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[3,6] = V_2*(b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[3,7] = V_2*V_4*(b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[3,10] = V_2*(b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[3,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[4,0] = V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[4,1] = V_1*V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[4,2] = V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[4,3] = V_2*V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[4,4] = V_1*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + 2*V_3*(g_1_3 + g_3_2 + g_3_4 + g_7_3) + V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[4,5] = V_1*V_3*(-b_1_3*cos(theta_1 - theta_3) - g_1_3*sin(theta_1 - theta_3)) + V_2*V_3*(-b_3_2*cos(theta_2 - theta_3) - g_3_2*sin(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[4,6] = V_3*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[4,7] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[4,12] = V_3*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[4,13] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[5,0] = V_3*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3))
        struct[0].Gy[5,1] = V_1*V_3*(-b_1_3*sin(theta_1 - theta_3) + g_1_3*cos(theta_1 - theta_3))
        struct[0].Gy[5,2] = V_3*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3))
        struct[0].Gy[5,3] = V_2*V_3*(-b_3_2*sin(theta_2 - theta_3) + g_3_2*cos(theta_2 - theta_3))
        struct[0].Gy[5,4] = V_1*(b_1_3*cos(theta_1 - theta_3) + g_1_3*sin(theta_1 - theta_3)) + V_2*(b_3_2*cos(theta_2 - theta_3) + g_3_2*sin(theta_2 - theta_3)) + 2*V_3*(-b_1_3 - b_3_2 - b_3_4 - b_7_3 - bs_1_3/2 - bs_3_2/2 - bs_3_4/2 - bs_7_3/2) + V_4*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_7*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[5,5] = V_1*V_3*(b_1_3*sin(theta_1 - theta_3) - g_1_3*cos(theta_1 - theta_3)) + V_2*V_3*(b_3_2*sin(theta_2 - theta_3) - g_3_2*cos(theta_2 - theta_3)) + V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[5,6] = V_3*(b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[5,7] = V_3*V_4*(b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[5,12] = V_3*(b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[5,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[6,0] = V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[6,1] = V_1*V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[6,2] = V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[6,3] = V_2*V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[6,4] = V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[6,5] = V_3*V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[6,6] = V_1*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + 2*V_4*(g_1_4 + g_2_4 + g_3_4 + g_8_4) + V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[6,7] = V_1*V_4*(-b_1_4*cos(theta_1 - theta_4) - g_1_4*sin(theta_1 - theta_4)) + V_2*V_4*(-b_2_4*cos(theta_2 - theta_4) - g_2_4*sin(theta_2 - theta_4)) + V_3*V_4*(-b_3_4*cos(theta_3 - theta_4) - g_3_4*sin(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[6,14] = V_4*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[6,15] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[7,0] = V_4*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4))
        struct[0].Gy[7,1] = V_1*V_4*(-b_1_4*sin(theta_1 - theta_4) + g_1_4*cos(theta_1 - theta_4))
        struct[0].Gy[7,2] = V_4*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4))
        struct[0].Gy[7,3] = V_2*V_4*(-b_2_4*sin(theta_2 - theta_4) + g_2_4*cos(theta_2 - theta_4))
        struct[0].Gy[7,4] = V_4*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4))
        struct[0].Gy[7,5] = V_3*V_4*(-b_3_4*sin(theta_3 - theta_4) + g_3_4*cos(theta_3 - theta_4))
        struct[0].Gy[7,6] = V_1*(b_1_4*cos(theta_1 - theta_4) + g_1_4*sin(theta_1 - theta_4)) + V_2*(b_2_4*cos(theta_2 - theta_4) + g_2_4*sin(theta_2 - theta_4)) + V_3*(b_3_4*cos(theta_3 - theta_4) + g_3_4*sin(theta_3 - theta_4)) + 2*V_4*(-b_1_4 - b_2_4 - b_3_4 - b_8_4 - bs_1_4/2 - bs_2_4/2 - bs_3_4/2 - bs_8_4/2) + V_8*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[7,7] = V_1*V_4*(b_1_4*sin(theta_1 - theta_4) - g_1_4*cos(theta_1 - theta_4)) + V_2*V_4*(b_2_4*sin(theta_2 - theta_4) - g_2_4*cos(theta_2 - theta_4)) + V_3*V_4*(b_3_4*sin(theta_3 - theta_4) - g_3_4*cos(theta_3 - theta_4)) + V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[7,14] = V_4*(b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[7,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[8,0] = V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[8,1] = V_1*V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[8,8] = V_1*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5)) + 2*V_5*g_5_1
        struct[0].Gy[8,9] = V_1*V_5*(-b_5_1*cos(theta_1 - theta_5) - g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[8,18] = -S_n_5/S_base
        struct[0].Gy[9,0] = V_5*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5))
        struct[0].Gy[9,1] = V_1*V_5*(-b_5_1*sin(theta_1 - theta_5) + g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[9,8] = V_1*(b_5_1*cos(theta_1 - theta_5) + g_5_1*sin(theta_1 - theta_5)) + 2*V_5*(-b_5_1 - bs_5_1/2)
        struct[0].Gy[9,9] = V_1*V_5*(b_5_1*sin(theta_1 - theta_5) - g_5_1*cos(theta_1 - theta_5))
        struct[0].Gy[9,19] = -S_n_5/S_base
        struct[0].Gy[10,2] = V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[10,3] = V_2*V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[10,10] = V_2*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6)) + 2*V_6*g_6_2
        struct[0].Gy[10,11] = V_2*V_6*(-b_6_2*cos(theta_2 - theta_6) - g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[10,27] = -S_n_6/S_base
        struct[0].Gy[11,2] = V_6*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6))
        struct[0].Gy[11,3] = V_2*V_6*(-b_6_2*sin(theta_2 - theta_6) + g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[11,10] = V_2*(b_6_2*cos(theta_2 - theta_6) + g_6_2*sin(theta_2 - theta_6)) + 2*V_6*(-b_6_2 - bs_6_2/2)
        struct[0].Gy[11,11] = V_2*V_6*(b_6_2*sin(theta_2 - theta_6) - g_6_2*cos(theta_2 - theta_6))
        struct[0].Gy[11,28] = -S_n_6/S_base
        struct[0].Gy[12,4] = V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[12,5] = V_3*V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[12,12] = V_3*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7)) + 2*V_7*g_7_3
        struct[0].Gy[12,13] = V_3*V_7*(-b_7_3*cos(theta_3 - theta_7) - g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[12,36] = -S_n_7/S_base
        struct[0].Gy[13,4] = V_7*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7))
        struct[0].Gy[13,5] = V_3*V_7*(-b_7_3*sin(theta_3 - theta_7) + g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[13,12] = V_3*(b_7_3*cos(theta_3 - theta_7) + g_7_3*sin(theta_3 - theta_7)) + 2*V_7*(-b_7_3 - bs_7_3/2)
        struct[0].Gy[13,13] = V_3*V_7*(b_7_3*sin(theta_3 - theta_7) - g_7_3*cos(theta_3 - theta_7))
        struct[0].Gy[13,37] = -S_n_7/S_base
        struct[0].Gy[14,6] = V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[14,7] = V_4*V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[14,14] = V_4*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8)) + 2*V_8*g_8_4
        struct[0].Gy[14,15] = V_4*V_8*(-b_8_4*cos(theta_4 - theta_8) - g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[14,45] = -S_n_8/S_base
        struct[0].Gy[15,6] = V_8*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8))
        struct[0].Gy[15,7] = V_4*V_8*(-b_8_4*sin(theta_4 - theta_8) + g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[15,14] = V_4*(b_8_4*cos(theta_4 - theta_8) + g_8_4*sin(theta_4 - theta_8)) + 2*V_8*(-b_8_4 - bs_8_4/2)
        struct[0].Gy[15,15] = V_4*V_8*(b_8_4*sin(theta_4 - theta_8) - g_8_4*cos(theta_4 - theta_8))
        struct[0].Gy[15,46] = -S_n_8/S_base
        struct[0].Gy[16,8] = cos(delta_5 - theta_5)
        struct[0].Gy[16,9] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy[16,16] = X1d_5
        struct[0].Gy[16,17] = R_a_5
        struct[0].Gy[17,8] = sin(delta_5 - theta_5)
        struct[0].Gy[17,9] = -V_5*cos(delta_5 - theta_5)
        struct[0].Gy[17,16] = R_a_5
        struct[0].Gy[17,17] = -X1q_5
        struct[0].Gy[18,8] = i_d_5*sin(delta_5 - theta_5) + i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy[18,9] = -V_5*i_d_5*cos(delta_5 - theta_5) + V_5*i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy[18,16] = V_5*sin(delta_5 - theta_5)
        struct[0].Gy[18,17] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy[18,18] = -1
        struct[0].Gy[19,8] = i_d_5*cos(delta_5 - theta_5) - i_q_5*sin(delta_5 - theta_5)
        struct[0].Gy[19,9] = V_5*i_d_5*sin(delta_5 - theta_5) + V_5*i_q_5*cos(delta_5 - theta_5)
        struct[0].Gy[19,16] = V_5*cos(delta_5 - theta_5)
        struct[0].Gy[19,17] = -V_5*sin(delta_5 - theta_5)
        struct[0].Gy[19,19] = -1
        struct[0].Gy[20,20] = -1
        struct[0].Gy[20,24] = K_a_5
        struct[0].Gy[21,21] = -1
        struct[0].Gy[21,52] = 1
        struct[0].Gy[22,22] = -1
        struct[0].Gy[23,23] = -1
        struct[0].Gy[24,23] = K_stab_5*T_1_5/T_2_5
        struct[0].Gy[24,24] = -1
        struct[0].Gy[25,10] = cos(delta_6 - theta_6)
        struct[0].Gy[25,11] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy[25,25] = X1d_6
        struct[0].Gy[25,26] = R_a_6
        struct[0].Gy[26,10] = sin(delta_6 - theta_6)
        struct[0].Gy[26,11] = -V_6*cos(delta_6 - theta_6)
        struct[0].Gy[26,25] = R_a_6
        struct[0].Gy[26,26] = -X1q_6
        struct[0].Gy[27,10] = i_d_6*sin(delta_6 - theta_6) + i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy[27,11] = -V_6*i_d_6*cos(delta_6 - theta_6) + V_6*i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy[27,25] = V_6*sin(delta_6 - theta_6)
        struct[0].Gy[27,26] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy[27,27] = -1
        struct[0].Gy[28,10] = i_d_6*cos(delta_6 - theta_6) - i_q_6*sin(delta_6 - theta_6)
        struct[0].Gy[28,11] = V_6*i_d_6*sin(delta_6 - theta_6) + V_6*i_q_6*cos(delta_6 - theta_6)
        struct[0].Gy[28,25] = V_6*cos(delta_6 - theta_6)
        struct[0].Gy[28,26] = -V_6*sin(delta_6 - theta_6)
        struct[0].Gy[28,28] = -1
        struct[0].Gy[29,29] = -1
        struct[0].Gy[29,33] = K_a_6
        struct[0].Gy[30,30] = -1
        struct[0].Gy[30,53] = 1
        struct[0].Gy[31,31] = -1
        struct[0].Gy[32,32] = -1
        struct[0].Gy[33,32] = K_stab_6*T_1_6/T_2_6
        struct[0].Gy[33,33] = -1
        struct[0].Gy[34,12] = cos(delta_7 - theta_7)
        struct[0].Gy[34,13] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy[34,34] = X1d_7
        struct[0].Gy[34,35] = R_a_7
        struct[0].Gy[35,12] = sin(delta_7 - theta_7)
        struct[0].Gy[35,13] = -V_7*cos(delta_7 - theta_7)
        struct[0].Gy[35,34] = R_a_7
        struct[0].Gy[35,35] = -X1q_7
        struct[0].Gy[36,12] = i_d_7*sin(delta_7 - theta_7) + i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy[36,13] = -V_7*i_d_7*cos(delta_7 - theta_7) + V_7*i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy[36,34] = V_7*sin(delta_7 - theta_7)
        struct[0].Gy[36,35] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy[36,36] = -1
        struct[0].Gy[37,12] = i_d_7*cos(delta_7 - theta_7) - i_q_7*sin(delta_7 - theta_7)
        struct[0].Gy[37,13] = V_7*i_d_7*sin(delta_7 - theta_7) + V_7*i_q_7*cos(delta_7 - theta_7)
        struct[0].Gy[37,34] = V_7*cos(delta_7 - theta_7)
        struct[0].Gy[37,35] = -V_7*sin(delta_7 - theta_7)
        struct[0].Gy[37,37] = -1
        struct[0].Gy[38,38] = -1
        struct[0].Gy[38,42] = K_a_7
        struct[0].Gy[39,39] = -1
        struct[0].Gy[39,54] = 1
        struct[0].Gy[40,40] = -1
        struct[0].Gy[41,41] = -1
        struct[0].Gy[42,41] = K_stab_7*T_1_7/T_2_7
        struct[0].Gy[42,42] = -1
        struct[0].Gy[43,14] = cos(delta_8 - theta_8)
        struct[0].Gy[43,15] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy[43,43] = X1d_8
        struct[0].Gy[43,44] = R_a_8
        struct[0].Gy[44,14] = sin(delta_8 - theta_8)
        struct[0].Gy[44,15] = -V_8*cos(delta_8 - theta_8)
        struct[0].Gy[44,43] = R_a_8
        struct[0].Gy[44,44] = -X1q_8
        struct[0].Gy[45,14] = i_d_8*sin(delta_8 - theta_8) + i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy[45,15] = -V_8*i_d_8*cos(delta_8 - theta_8) + V_8*i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy[45,43] = V_8*sin(delta_8 - theta_8)
        struct[0].Gy[45,44] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy[45,45] = -1
        struct[0].Gy[46,14] = i_d_8*cos(delta_8 - theta_8) - i_q_8*sin(delta_8 - theta_8)
        struct[0].Gy[46,15] = V_8*i_d_8*sin(delta_8 - theta_8) + V_8*i_q_8*cos(delta_8 - theta_8)
        struct[0].Gy[46,43] = V_8*cos(delta_8 - theta_8)
        struct[0].Gy[46,44] = -V_8*sin(delta_8 - theta_8)
        struct[0].Gy[46,46] = -1
        struct[0].Gy[47,47] = -1
        struct[0].Gy[47,51] = K_a_8
        struct[0].Gy[48,48] = -1
        struct[0].Gy[48,55] = 1
        struct[0].Gy[49,49] = -1
        struct[0].Gy[50,50] = -1
        struct[0].Gy[51,50] = K_stab_8*T_1_8/T_2_8
        struct[0].Gy[51,51] = -1
        struct[0].Gy[52,52] = -1
        struct[0].Gy[53,53] = -1
        struct[0].Gy[54,54] = -1
        struct[0].Gy[55,55] = -1
        struct[0].Gy[56,56] = -1

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
        struct[0].Gu[20,16] = K_a_5
        struct[0].Gu[20,17] = K_a_5
        struct[0].Gu[21,18] = 1
        struct[0].Gu[24,17] = -1
        struct[0].Gu[29,19] = K_a_6
        struct[0].Gu[29,20] = K_a_6
        struct[0].Gu[30,21] = 1
        struct[0].Gu[33,20] = -1
        struct[0].Gu[38,22] = K_a_7
        struct[0].Gu[38,23] = K_a_7
        struct[0].Gu[39,24] = 1
        struct[0].Gu[42,23] = -1
        struct[0].Gu[47,25] = K_a_8
        struct[0].Gu[47,26] = K_a_8
        struct[0].Gu[48,27] = 1
        struct[0].Gu[51,26] = -1





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
    Fx_ini_rows = [0, 0, 1, 1, 2, 3, 4, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 13, 14, 16, 17, 17, 18, 18, 19, 20, 20, 21, 21, 22, 23, 24, 26, 27, 27, 28, 28, 29, 30, 30, 31, 31, 32, 33, 34, 36, 37, 37, 38, 38, 39]

    Fx_ini_cols = [0, 1, 0, 1, 2, 3, 4, 6, 6, 7, 1, 8, 9, 10, 11, 10, 11, 12, 13, 14, 16, 16, 17, 11, 18, 19, 20, 21, 20, 21, 22, 23, 24, 26, 26, 27, 21, 28, 29, 30, 31, 30, 31, 32, 33, 34, 36, 36, 37, 31, 38, 39]

    Fy_ini_rows = [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 9, 10, 11, 11, 11, 11, 11, 11, 12, 12, 13, 14, 15, 16, 19, 20, 21, 21, 21, 21, 21, 21, 22, 22, 23, 24, 25, 26, 29, 30, 31, 31, 31, 31, 31, 31, 32, 32, 33, 34, 35, 36, 39, 40]

    Fy_ini_cols = [56, 8, 9, 16, 17, 22, 56, 16, 20, 17, 8, 8, 21, 23, 56, 10, 11, 25, 26, 31, 56, 25, 29, 26, 10, 10, 30, 32, 56, 12, 13, 34, 35, 40, 56, 34, 38, 35, 12, 12, 39, 41, 56, 14, 15, 43, 44, 49, 56, 43, 47, 44, 14, 14, 48, 50, 56]

    Gx_ini_rows = [16, 16, 17, 17, 18, 19, 20, 20, 21, 22, 22, 23, 23, 24, 25, 25, 26, 26, 27, 28, 29, 29, 30, 31, 31, 32, 32, 33, 34, 34, 35, 35, 36, 37, 38, 38, 39, 40, 40, 41, 41, 42, 43, 43, 44, 44, 45, 46, 47, 47, 48, 49, 49, 50, 50, 51, 52, 53, 54, 55, 56, 56, 56, 56]

    Gx_ini_cols = [0, 2, 0, 3, 0, 0, 4, 5, 1, 6, 7, 1, 8, 9, 10, 12, 10, 13, 10, 10, 14, 15, 11, 16, 17, 11, 18, 19, 20, 22, 20, 23, 20, 20, 24, 25, 21, 26, 27, 21, 28, 29, 30, 32, 30, 33, 30, 30, 34, 35, 31, 36, 37, 31, 38, 39, 40, 40, 40, 40, 1, 11, 21, 31]

    Gy_ini_rows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 21, 21, 22, 23, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 30, 30, 31, 32, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 39, 39, 40, 41, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 47, 47, 48, 48, 49, 50, 51, 51, 52, 53, 54, 55, 56]

    Gy_ini_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 0, 1, 8, 9, 18, 0, 1, 8, 9, 19, 2, 3, 10, 11, 27, 2, 3, 10, 11, 28, 4, 5, 12, 13, 36, 4, 5, 12, 13, 37, 6, 7, 14, 15, 45, 6, 7, 14, 15, 46, 8, 9, 16, 17, 8, 9, 16, 17, 8, 9, 16, 17, 18, 8, 9, 16, 17, 19, 20, 24, 21, 52, 22, 23, 23, 24, 10, 11, 25, 26, 10, 11, 25, 26, 10, 11, 25, 26, 27, 10, 11, 25, 26, 28, 29, 33, 30, 53, 31, 32, 32, 33, 12, 13, 34, 35, 12, 13, 34, 35, 12, 13, 34, 35, 36, 12, 13, 34, 35, 37, 38, 42, 39, 54, 40, 41, 41, 42, 14, 15, 43, 44, 14, 15, 43, 44, 14, 15, 43, 44, 45, 14, 15, 43, 44, 46, 47, 51, 48, 55, 49, 50, 50, 51, 52, 53, 54, 55, 56]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols