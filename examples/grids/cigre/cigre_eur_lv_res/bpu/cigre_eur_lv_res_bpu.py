import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 


class cigre_eur_lv_res_bpu_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 15
        self.N_y = 51 
        self.N_z = 18 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_R01_R02', 'b_R01_R02', 'bs_R01_R02', 'g_R02_R03', 'b_R02_R03', 'bs_R02_R03', 'g_R03_R04', 'b_R03_R04', 'bs_R03_R04', 'g_R04_R05', 'b_R04_R05', 'bs_R04_R05', 'g_R05_R06', 'b_R05_R06', 'bs_R05_R06', 'g_R06_R07', 'b_R06_R07', 'bs_R06_R07', 'g_R07_R08', 'b_R07_R08', 'bs_R07_R08', 'g_R08_R09', 'b_R08_R09', 'bs_R08_R09', 'g_R09_R10', 'b_R09_R10', 'bs_R09_R10', 'g_R03_R11', 'b_R03_R11', 'bs_R03_R11', 'g_R04_R12', 'b_R04_R12', 'bs_R04_R12', 'g_R12_R13', 'b_R12_R13', 'bs_R12_R13', 'g_R13_R14', 'b_R13_R14', 'bs_R13_R14', 'g_R14_R15', 'b_R14_R15', 'bs_R14_R15', 'g_R06_R16', 'b_R06_R16', 'bs_R06_R16', 'g_R09_R17', 'b_R09_R17', 'bs_R09_R17', 'g_R10_R18', 'b_R10_R18', 'bs_R10_R18', 'U_R01_n', 'U_R02_n', 'U_R03_n', 'U_R04_n', 'U_R05_n', 'U_R06_n', 'U_R07_n', 'U_R08_n', 'U_R09_n', 'U_R10_n', 'U_R11_n', 'U_R12_n', 'U_R13_n', 'U_R14_n', 'U_R15_n', 'U_R16_n', 'U_R17_n', 'U_R18_n', 'S_n_R10', 'H_R10', 'Omega_b_R10', 'T1d0_R10', 'T1q0_R10', 'X_d_R10', 'X_q_R10', 'X1d_R10', 'X1q_R10', 'D_R10', 'R_a_R10', 'K_delta_R10', 'K_a_R10', 'K_ai_R10', 'T_r_R10', 'Droop_R10', 'T_m_R10', 'S_n_R14', 'H_R14', 'Omega_b_R14', 'T1d0_R14', 'T1q0_R14', 'X_d_R14', 'X_q_R14', 'X1d_R14', 'X1q_R14', 'D_R14', 'R_a_R14', 'K_delta_R14', 'K_a_R14', 'K_ai_R14', 'T_r_R14', 'Droop_R14', 'T_m_R14', 'K_sec_R10', 'K_sec_R14'] 
        self.params_values_list  = [1000000.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 644.4416190074766, -322.61846569219165, 0.0, 183.38422558864576, -19.0143186828563, 0.0, 157.1864790759821, -16.29798744244826, 0.0, 157.1864790759821, -16.29798744244826, 0.0, 157.1864790759821, -16.29798744244826, 0.0, 157.1864790759821, -16.29798744244826, 0.0, 183.38422558864576, -19.0143186828563, 0.0, 183.38422558864576, -19.0143186828563, 0.0, 183.38422558864576, -19.0143186828563, 0.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 1000000.0, 6.5, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 0.0, 0.0025, 0.01, 100, 1e-06, 0.1, 0.05, 5.0, 1000000.0, 6.5, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 0.0, 0.0025, 0.01, 100, 1e-06, 0.1, 0.05, 5.0, 0.001, 0.001] 
        self.inputs_ini_list = ['P_R01', 'Q_R01', 'P_R02', 'Q_R02', 'P_R03', 'Q_R03', 'P_R04', 'Q_R04', 'P_R05', 'Q_R05', 'P_R06', 'Q_R06', 'P_R07', 'Q_R07', 'P_R08', 'Q_R08', 'P_R09', 'Q_R09', 'P_R10', 'Q_R10', 'P_R11', 'Q_R11', 'P_R12', 'Q_R12', 'P_R13', 'Q_R13', 'P_R14', 'Q_R14', 'P_R15', 'Q_R15', 'P_R16', 'Q_R16', 'P_R17', 'Q_R17', 'P_R18', 'Q_R18', 'v_ref_R10', 'v_pss_R10', 'p_c_R10', 'v_ref_R14', 'v_pss_R14', 'p_c_R14'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -14250.0, -4683.748, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -49400.0, -16236.995, -52250.0, -17173.744, -33250.0, -10928.746, -44650.0, -14675.745, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7] 
        self.inputs_run_list = ['P_R01', 'Q_R01', 'P_R02', 'Q_R02', 'P_R03', 'Q_R03', 'P_R04', 'Q_R04', 'P_R05', 'Q_R05', 'P_R06', 'Q_R06', 'P_R07', 'Q_R07', 'P_R08', 'Q_R08', 'P_R09', 'Q_R09', 'P_R10', 'Q_R10', 'P_R11', 'Q_R11', 'P_R12', 'Q_R12', 'P_R13', 'Q_R13', 'P_R14', 'Q_R14', 'P_R15', 'Q_R15', 'P_R16', 'Q_R16', 'P_R17', 'Q_R17', 'P_R18', 'Q_R18', 'v_ref_R10', 'v_pss_R10', 'p_c_R10', 'v_ref_R14', 'v_pss_R14', 'p_c_R14'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -14250.0, -4683.748, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -49400.0, -16236.995, -52250.0, -17173.744, -33250.0, -10928.746, -44650.0, -14675.745, 1.0, 0.0, 0.7, 1.0, 0.0, 0.7] 
        self.outputs_list = ['V_R01', 'V_R02', 'V_R03', 'V_R04', 'V_R05', 'V_R06', 'V_R07', 'V_R08', 'V_R09', 'V_R10', 'V_R11', 'V_R12', 'V_R13', 'V_R14', 'V_R15', 'V_R16', 'V_R17', 'V_R18'] 
        self.x_list = ['delta_R10', 'omega_R10', 'e1q_R10', 'e1d_R10', 'v_c_R10', 'xi_v_R10', 'p_m_R10', 'delta_R14', 'omega_R14', 'e1q_R14', 'e1d_R14', 'v_c_R14', 'xi_v_R14', 'p_m_R14', 'xi_freq'] 
        self.y_run_list = ['V_R01', 'theta_R01', 'V_R02', 'theta_R02', 'V_R03', 'theta_R03', 'V_R04', 'theta_R04', 'V_R05', 'theta_R05', 'V_R06', 'theta_R06', 'V_R07', 'theta_R07', 'V_R08', 'theta_R08', 'V_R09', 'theta_R09', 'V_R10', 'theta_R10', 'V_R11', 'theta_R11', 'V_R12', 'theta_R12', 'V_R13', 'theta_R13', 'V_R14', 'theta_R14', 'V_R15', 'theta_R15', 'V_R16', 'theta_R16', 'V_R17', 'theta_R17', 'V_R18', 'theta_R18', 'i_d_R10', 'i_q_R10', 'p_g_R10_1', 'q_g_R10_1', 'v_f_R10', 'p_m_ref_R10', 'i_d_R14', 'i_q_R14', 'p_g_R14_1', 'q_g_R14_1', 'v_f_R14', 'p_m_ref_R14', 'omega_coi', 'p_r_R10', 'p_r_R14'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_R01', 'theta_R01', 'V_R02', 'theta_R02', 'V_R03', 'theta_R03', 'V_R04', 'theta_R04', 'V_R05', 'theta_R05', 'V_R06', 'theta_R06', 'V_R07', 'theta_R07', 'V_R08', 'theta_R08', 'V_R09', 'theta_R09', 'V_R10', 'theta_R10', 'V_R11', 'theta_R11', 'V_R12', 'theta_R12', 'V_R13', 'theta_R13', 'V_R14', 'theta_R14', 'V_R15', 'theta_R15', 'V_R16', 'theta_R16', 'V_R17', 'theta_R17', 'V_R18', 'theta_R18', 'i_d_R10', 'i_q_R10', 'p_g_R10_1', 'q_g_R10_1', 'v_f_R10', 'p_m_ref_R10', 'i_d_R14', 'i_q_R14', 'p_g_R14_1', 'q_g_R14_1', 'v_f_R14', 'p_m_ref_R14', 'omega_coi', 'p_r_R10', 'p_r_R14'] 
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
            
            
    def report_x(self,value_format='5.2f'):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_params(self,value_format='5.2f'):
        for item in self.params_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
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
    g_R01_R02 = struct[0].g_R01_R02
    b_R01_R02 = struct[0].b_R01_R02
    bs_R01_R02 = struct[0].bs_R01_R02
    g_R02_R03 = struct[0].g_R02_R03
    b_R02_R03 = struct[0].b_R02_R03
    bs_R02_R03 = struct[0].bs_R02_R03
    g_R03_R04 = struct[0].g_R03_R04
    b_R03_R04 = struct[0].b_R03_R04
    bs_R03_R04 = struct[0].bs_R03_R04
    g_R04_R05 = struct[0].g_R04_R05
    b_R04_R05 = struct[0].b_R04_R05
    bs_R04_R05 = struct[0].bs_R04_R05
    g_R05_R06 = struct[0].g_R05_R06
    b_R05_R06 = struct[0].b_R05_R06
    bs_R05_R06 = struct[0].bs_R05_R06
    g_R06_R07 = struct[0].g_R06_R07
    b_R06_R07 = struct[0].b_R06_R07
    bs_R06_R07 = struct[0].bs_R06_R07
    g_R07_R08 = struct[0].g_R07_R08
    b_R07_R08 = struct[0].b_R07_R08
    bs_R07_R08 = struct[0].bs_R07_R08
    g_R08_R09 = struct[0].g_R08_R09
    b_R08_R09 = struct[0].b_R08_R09
    bs_R08_R09 = struct[0].bs_R08_R09
    g_R09_R10 = struct[0].g_R09_R10
    b_R09_R10 = struct[0].b_R09_R10
    bs_R09_R10 = struct[0].bs_R09_R10
    g_R03_R11 = struct[0].g_R03_R11
    b_R03_R11 = struct[0].b_R03_R11
    bs_R03_R11 = struct[0].bs_R03_R11
    g_R04_R12 = struct[0].g_R04_R12
    b_R04_R12 = struct[0].b_R04_R12
    bs_R04_R12 = struct[0].bs_R04_R12
    g_R12_R13 = struct[0].g_R12_R13
    b_R12_R13 = struct[0].b_R12_R13
    bs_R12_R13 = struct[0].bs_R12_R13
    g_R13_R14 = struct[0].g_R13_R14
    b_R13_R14 = struct[0].b_R13_R14
    bs_R13_R14 = struct[0].bs_R13_R14
    g_R14_R15 = struct[0].g_R14_R15
    b_R14_R15 = struct[0].b_R14_R15
    bs_R14_R15 = struct[0].bs_R14_R15
    g_R06_R16 = struct[0].g_R06_R16
    b_R06_R16 = struct[0].b_R06_R16
    bs_R06_R16 = struct[0].bs_R06_R16
    g_R09_R17 = struct[0].g_R09_R17
    b_R09_R17 = struct[0].b_R09_R17
    bs_R09_R17 = struct[0].bs_R09_R17
    g_R10_R18 = struct[0].g_R10_R18
    b_R10_R18 = struct[0].b_R10_R18
    bs_R10_R18 = struct[0].bs_R10_R18
    U_R01_n = struct[0].U_R01_n
    U_R02_n = struct[0].U_R02_n
    U_R03_n = struct[0].U_R03_n
    U_R04_n = struct[0].U_R04_n
    U_R05_n = struct[0].U_R05_n
    U_R06_n = struct[0].U_R06_n
    U_R07_n = struct[0].U_R07_n
    U_R08_n = struct[0].U_R08_n
    U_R09_n = struct[0].U_R09_n
    U_R10_n = struct[0].U_R10_n
    U_R11_n = struct[0].U_R11_n
    U_R12_n = struct[0].U_R12_n
    U_R13_n = struct[0].U_R13_n
    U_R14_n = struct[0].U_R14_n
    U_R15_n = struct[0].U_R15_n
    U_R16_n = struct[0].U_R16_n
    U_R17_n = struct[0].U_R17_n
    U_R18_n = struct[0].U_R18_n
    S_n_R10 = struct[0].S_n_R10
    H_R10 = struct[0].H_R10
    Omega_b_R10 = struct[0].Omega_b_R10
    T1d0_R10 = struct[0].T1d0_R10
    T1q0_R10 = struct[0].T1q0_R10
    X_d_R10 = struct[0].X_d_R10
    X_q_R10 = struct[0].X_q_R10
    X1d_R10 = struct[0].X1d_R10
    X1q_R10 = struct[0].X1q_R10
    D_R10 = struct[0].D_R10
    R_a_R10 = struct[0].R_a_R10
    K_delta_R10 = struct[0].K_delta_R10
    K_a_R10 = struct[0].K_a_R10
    K_ai_R10 = struct[0].K_ai_R10
    T_r_R10 = struct[0].T_r_R10
    Droop_R10 = struct[0].Droop_R10
    T_m_R10 = struct[0].T_m_R10
    S_n_R14 = struct[0].S_n_R14
    H_R14 = struct[0].H_R14
    Omega_b_R14 = struct[0].Omega_b_R14
    T1d0_R14 = struct[0].T1d0_R14
    T1q0_R14 = struct[0].T1q0_R14
    X_d_R14 = struct[0].X_d_R14
    X_q_R14 = struct[0].X_q_R14
    X1d_R14 = struct[0].X1d_R14
    X1q_R14 = struct[0].X1q_R14
    D_R14 = struct[0].D_R14
    R_a_R14 = struct[0].R_a_R14
    K_delta_R14 = struct[0].K_delta_R14
    K_a_R14 = struct[0].K_a_R14
    K_ai_R14 = struct[0].K_ai_R14
    T_r_R14 = struct[0].T_r_R14
    Droop_R14 = struct[0].Droop_R14
    T_m_R14 = struct[0].T_m_R14
    K_sec_R10 = struct[0].K_sec_R10
    K_sec_R14 = struct[0].K_sec_R14
    
    # Inputs:
    P_R01 = struct[0].P_R01
    Q_R01 = struct[0].Q_R01
    P_R02 = struct[0].P_R02
    Q_R02 = struct[0].Q_R02
    P_R03 = struct[0].P_R03
    Q_R03 = struct[0].Q_R03
    P_R04 = struct[0].P_R04
    Q_R04 = struct[0].Q_R04
    P_R05 = struct[0].P_R05
    Q_R05 = struct[0].Q_R05
    P_R06 = struct[0].P_R06
    Q_R06 = struct[0].Q_R06
    P_R07 = struct[0].P_R07
    Q_R07 = struct[0].Q_R07
    P_R08 = struct[0].P_R08
    Q_R08 = struct[0].Q_R08
    P_R09 = struct[0].P_R09
    Q_R09 = struct[0].Q_R09
    P_R10 = struct[0].P_R10
    Q_R10 = struct[0].Q_R10
    P_R11 = struct[0].P_R11
    Q_R11 = struct[0].Q_R11
    P_R12 = struct[0].P_R12
    Q_R12 = struct[0].Q_R12
    P_R13 = struct[0].P_R13
    Q_R13 = struct[0].Q_R13
    P_R14 = struct[0].P_R14
    Q_R14 = struct[0].Q_R14
    P_R15 = struct[0].P_R15
    Q_R15 = struct[0].Q_R15
    P_R16 = struct[0].P_R16
    Q_R16 = struct[0].Q_R16
    P_R17 = struct[0].P_R17
    Q_R17 = struct[0].Q_R17
    P_R18 = struct[0].P_R18
    Q_R18 = struct[0].Q_R18
    v_ref_R10 = struct[0].v_ref_R10
    v_pss_R10 = struct[0].v_pss_R10
    p_c_R10 = struct[0].p_c_R10
    v_ref_R14 = struct[0].v_ref_R14
    v_pss_R14 = struct[0].v_pss_R14
    p_c_R14 = struct[0].p_c_R14
    
    # Dynamical states:
    delta_R10 = struct[0].x[0,0]
    omega_R10 = struct[0].x[1,0]
    e1q_R10 = struct[0].x[2,0]
    e1d_R10 = struct[0].x[3,0]
    v_c_R10 = struct[0].x[4,0]
    xi_v_R10 = struct[0].x[5,0]
    p_m_R10 = struct[0].x[6,0]
    delta_R14 = struct[0].x[7,0]
    omega_R14 = struct[0].x[8,0]
    e1q_R14 = struct[0].x[9,0]
    e1d_R14 = struct[0].x[10,0]
    v_c_R14 = struct[0].x[11,0]
    xi_v_R14 = struct[0].x[12,0]
    p_m_R14 = struct[0].x[13,0]
    xi_freq = struct[0].x[14,0]
    
    # Algebraic states:
    V_R01 = struct[0].y_ini[0,0]
    theta_R01 = struct[0].y_ini[1,0]
    V_R02 = struct[0].y_ini[2,0]
    theta_R02 = struct[0].y_ini[3,0]
    V_R03 = struct[0].y_ini[4,0]
    theta_R03 = struct[0].y_ini[5,0]
    V_R04 = struct[0].y_ini[6,0]
    theta_R04 = struct[0].y_ini[7,0]
    V_R05 = struct[0].y_ini[8,0]
    theta_R05 = struct[0].y_ini[9,0]
    V_R06 = struct[0].y_ini[10,0]
    theta_R06 = struct[0].y_ini[11,0]
    V_R07 = struct[0].y_ini[12,0]
    theta_R07 = struct[0].y_ini[13,0]
    V_R08 = struct[0].y_ini[14,0]
    theta_R08 = struct[0].y_ini[15,0]
    V_R09 = struct[0].y_ini[16,0]
    theta_R09 = struct[0].y_ini[17,0]
    V_R10 = struct[0].y_ini[18,0]
    theta_R10 = struct[0].y_ini[19,0]
    V_R11 = struct[0].y_ini[20,0]
    theta_R11 = struct[0].y_ini[21,0]
    V_R12 = struct[0].y_ini[22,0]
    theta_R12 = struct[0].y_ini[23,0]
    V_R13 = struct[0].y_ini[24,0]
    theta_R13 = struct[0].y_ini[25,0]
    V_R14 = struct[0].y_ini[26,0]
    theta_R14 = struct[0].y_ini[27,0]
    V_R15 = struct[0].y_ini[28,0]
    theta_R15 = struct[0].y_ini[29,0]
    V_R16 = struct[0].y_ini[30,0]
    theta_R16 = struct[0].y_ini[31,0]
    V_R17 = struct[0].y_ini[32,0]
    theta_R17 = struct[0].y_ini[33,0]
    V_R18 = struct[0].y_ini[34,0]
    theta_R18 = struct[0].y_ini[35,0]
    i_d_R10 = struct[0].y_ini[36,0]
    i_q_R10 = struct[0].y_ini[37,0]
    p_g_R10_1 = struct[0].y_ini[38,0]
    q_g_R10_1 = struct[0].y_ini[39,0]
    v_f_R10 = struct[0].y_ini[40,0]
    p_m_ref_R10 = struct[0].y_ini[41,0]
    i_d_R14 = struct[0].y_ini[42,0]
    i_q_R14 = struct[0].y_ini[43,0]
    p_g_R14_1 = struct[0].y_ini[44,0]
    q_g_R14_1 = struct[0].y_ini[45,0]
    v_f_R14 = struct[0].y_ini[46,0]
    p_m_ref_R14 = struct[0].y_ini[47,0]
    omega_coi = struct[0].y_ini[48,0]
    p_r_R10 = struct[0].y_ini[49,0]
    p_r_R14 = struct[0].y_ini[50,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_R10*delta_R10 + Omega_b_R10*(omega_R10 - omega_coi)
        struct[0].f[1,0] = (-D_R10*(omega_R10 - omega_coi) - i_d_R10*(R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10)) - i_q_R10*(R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10)) + p_m_R10)/(2*H_R10)
        struct[0].f[2,0] = (-e1q_R10 - i_d_R10*(-X1d_R10 + X_d_R10) + v_f_R10)/T1d0_R10
        struct[0].f[3,0] = (-e1d_R10 + i_q_R10*(-X1q_R10 + X_q_R10))/T1q0_R10
        struct[0].f[4,0] = (V_R10 - v_c_R10)/T_r_R10
        struct[0].f[5,0] = -V_R10 + v_ref_R10
        struct[0].f[6,0] = (-p_m_R10 + p_m_ref_R10)/T_m_R10
        struct[0].f[7,0] = -K_delta_R14*delta_R14 + Omega_b_R14*(omega_R14 - omega_coi)
        struct[0].f[8,0] = (-D_R14*(omega_R14 - omega_coi) - i_d_R14*(R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14)) - i_q_R14*(R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14)) + p_m_R14)/(2*H_R14)
        struct[0].f[9,0] = (-e1q_R14 - i_d_R14*(-X1d_R14 + X_d_R14) + v_f_R14)/T1d0_R14
        struct[0].f[10,0] = (-e1d_R14 + i_q_R14*(-X1q_R14 + X_q_R14))/T1q0_R14
        struct[0].f[11,0] = (V_R14 - v_c_R14)/T_r_R14
        struct[0].f[12,0] = -V_R14 + v_ref_R14
        struct[0].f[13,0] = (-p_m_R14 + p_m_ref_R14)/T_m_R14
        struct[0].f[14,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_R01/S_base + V_R01**2*g_R01_R02 + V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].g[1,0] = -Q_R01/S_base + V_R01**2*(-b_R01_R02 - bs_R01_R02/2) + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].g[2,0] = -P_R02/S_base + V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02**2*(g_R01_R02 + g_R02_R03) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].g[3,0] = -Q_R02/S_base + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02**2*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].g[4,0] = -P_R03/S_base + V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03**2*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].g[5,0] = -Q_R03/S_base + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03**2*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].g[6,0] = -P_R04/S_base + V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04**2*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].g[7,0] = -Q_R04/S_base + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04**2*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].g[8,0] = -P_R05/S_base + V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05**2*(g_R04_R05 + g_R05_R06) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].g[9,0] = -Q_R05/S_base + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05**2*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].g[10,0] = -P_R06/S_base + V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06**2*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].g[11,0] = -Q_R06/S_base + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06**2*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].g[12,0] = -P_R07/S_base + V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07**2*(g_R06_R07 + g_R07_R08) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].g[13,0] = -Q_R07/S_base + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07**2*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].g[14,0] = -P_R08/S_base + V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08**2*(g_R07_R08 + g_R08_R09) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].g[15,0] = -Q_R08/S_base + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08**2*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].g[16,0] = -P_R09/S_base + V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09**2*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].g[17,0] = -Q_R09/S_base + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09**2*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].g[18,0] = -P_R10/S_base + V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10**2*(g_R09_R10 + g_R10_R18) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) - S_n_R10*p_g_R10_1/S_base
        struct[0].g[19,0] = -Q_R10/S_base + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10**2*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18)) - S_n_R10*q_g_R10_1/S_base
        struct[0].g[20,0] = -P_R11/S_base + V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + V_R11**2*g_R03_R11
        struct[0].g[21,0] = -Q_R11/S_base + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + V_R11**2*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].g[22,0] = -P_R12/S_base + V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12**2*(g_R04_R12 + g_R12_R13) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].g[23,0] = -Q_R12/S_base + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12**2*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].g[24,0] = -P_R13/S_base + V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13**2*(g_R12_R13 + g_R13_R14) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].g[25,0] = -Q_R13/S_base + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13**2*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].g[26,0] = -P_R14/S_base + V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14**2*(g_R13_R14 + g_R14_R15) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) - S_n_R14*p_g_R14_1/S_base
        struct[0].g[27,0] = -Q_R14/S_base + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14**2*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15)) - S_n_R14*q_g_R14_1/S_base
        struct[0].g[28,0] = -P_R15/S_base + V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + V_R15**2*g_R14_R15
        struct[0].g[29,0] = -Q_R15/S_base + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + V_R15**2*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].g[30,0] = -P_R16/S_base + V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + V_R16**2*g_R06_R16
        struct[0].g[31,0] = -Q_R16/S_base + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + V_R16**2*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].g[32,0] = -P_R17/S_base + V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + V_R17**2*g_R09_R17
        struct[0].g[33,0] = -Q_R17/S_base + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + V_R17**2*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].g[34,0] = -P_R18/S_base + V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + V_R18**2*g_R10_R18
        struct[0].g[35,0] = -Q_R18/S_base + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + V_R18**2*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].g[36,0] = R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10) + X1d_R10*i_d_R10 - e1q_R10
        struct[0].g[37,0] = R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10) - X1q_R10*i_q_R10 - e1d_R10
        struct[0].g[38,0] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10) - p_g_R10_1
        struct[0].g[39,0] = V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10) - q_g_R10_1
        struct[0].g[40,0] = K_a_R10*(-v_c_R10 + v_pss_R10 + v_ref_R10) + K_ai_R10*xi_v_R10 - v_f_R10
        struct[0].g[41,0] = p_c_R10 - p_m_ref_R10 + p_r_R10 - (omega_R10 - 1)/Droop_R10
        struct[0].g[42,0] = R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14) + X1d_R14*i_d_R14 - e1q_R14
        struct[0].g[43,0] = R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14) - X1q_R14*i_q_R14 - e1d_R14
        struct[0].g[44,0] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14) - p_g_R14_1
        struct[0].g[45,0] = V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14) - q_g_R14_1
        struct[0].g[46,0] = K_a_R14*(-v_c_R14 + v_pss_R14 + v_ref_R14) + K_ai_R14*xi_v_R14 - v_f_R14
        struct[0].g[47,0] = p_c_R14 - p_m_ref_R14 + p_r_R14 - (omega_R14 - 1)/Droop_R14
        struct[0].g[48,0] = omega_R10/2 + omega_R14/2 - omega_coi
        struct[0].g[49,0] = K_sec_R10*xi_freq/2 - p_r_R10
        struct[0].g[50,0] = K_sec_R14*xi_freq/2 - p_r_R14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_R01
        struct[0].h[1,0] = V_R02
        struct[0].h[2,0] = V_R03
        struct[0].h[3,0] = V_R04
        struct[0].h[4,0] = V_R05
        struct[0].h[5,0] = V_R06
        struct[0].h[6,0] = V_R07
        struct[0].h[7,0] = V_R08
        struct[0].h[8,0] = V_R09
        struct[0].h[9,0] = V_R10
        struct[0].h[10,0] = V_R11
        struct[0].h[11,0] = V_R12
        struct[0].h[12,0] = V_R13
        struct[0].h[13,0] = V_R14
        struct[0].h[14,0] = V_R15
        struct[0].h[15,0] = V_R16
        struct[0].h[16,0] = V_R17
        struct[0].h[17,0] = V_R18
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_R10
        struct[0].Fx_ini[0,1] = Omega_b_R10
        struct[0].Fx_ini[1,0] = (-V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fx_ini[1,1] = -D_R10/(2*H_R10)
        struct[0].Fx_ini[1,6] = 1/(2*H_R10)
        struct[0].Fx_ini[2,2] = -1/T1d0_R10
        struct[0].Fx_ini[3,3] = -1/T1q0_R10
        struct[0].Fx_ini[4,4] = -1/T_r_R10
        struct[0].Fx_ini[6,6] = -1/T_m_R10
        struct[0].Fx_ini[7,7] = -K_delta_R14
        struct[0].Fx_ini[7,8] = Omega_b_R14
        struct[0].Fx_ini[8,7] = (-V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fx_ini[8,8] = -D_R14/(2*H_R14)
        struct[0].Fx_ini[8,13] = 1/(2*H_R14)
        struct[0].Fx_ini[9,9] = -1/T1d0_R14
        struct[0].Fx_ini[10,10] = -1/T1q0_R14
        struct[0].Fx_ini[11,11] = -1/T_r_R14
        struct[0].Fx_ini[13,13] = -1/T_m_R14

    if mode == 11:

        struct[0].Fy_ini[0,48] = -Omega_b_R10 
        struct[0].Fy_ini[1,18] = (-i_d_R10*sin(delta_R10 - theta_R10) - i_q_R10*cos(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,19] = (V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,36] = (-2*R_a_R10*i_d_R10 - V_R10*sin(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,37] = (-2*R_a_R10*i_q_R10 - V_R10*cos(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,48] = D_R10/(2*H_R10) 
        struct[0].Fy_ini[2,36] = (X1d_R10 - X_d_R10)/T1d0_R10 
        struct[0].Fy_ini[2,40] = 1/T1d0_R10 
        struct[0].Fy_ini[3,37] = (-X1q_R10 + X_q_R10)/T1q0_R10 
        struct[0].Fy_ini[4,18] = 1/T_r_R10 
        struct[0].Fy_ini[5,18] = -1 
        struct[0].Fy_ini[6,41] = 1/T_m_R10 
        struct[0].Fy_ini[7,48] = -Omega_b_R14 
        struct[0].Fy_ini[8,26] = (-i_d_R14*sin(delta_R14 - theta_R14) - i_q_R14*cos(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,27] = (V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,42] = (-2*R_a_R14*i_d_R14 - V_R14*sin(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,43] = (-2*R_a_R14*i_q_R14 - V_R14*cos(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,48] = D_R14/(2*H_R14) 
        struct[0].Fy_ini[9,42] = (X1d_R14 - X_d_R14)/T1d0_R14 
        struct[0].Fy_ini[9,46] = 1/T1d0_R14 
        struct[0].Fy_ini[10,43] = (-X1q_R14 + X_q_R14)/T1q0_R14 
        struct[0].Fy_ini[11,26] = 1/T_r_R14 
        struct[0].Fy_ini[12,26] = -1 
        struct[0].Fy_ini[13,47] = 1/T_m_R14 
        struct[0].Fy_ini[14,48] = -1 

        struct[0].Gx_ini[36,0] = -V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gx_ini[36,2] = -1
        struct[0].Gx_ini[37,0] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gx_ini[37,3] = -1
        struct[0].Gx_ini[38,0] = V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gx_ini[39,0] = -V_R10*i_d_R10*sin(delta_R10 - theta_R10) - V_R10*i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gx_ini[40,4] = -K_a_R10
        struct[0].Gx_ini[40,5] = K_ai_R10
        struct[0].Gx_ini[41,1] = -1/Droop_R10
        struct[0].Gx_ini[42,7] = -V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gx_ini[42,9] = -1
        struct[0].Gx_ini[43,7] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gx_ini[43,10] = -1
        struct[0].Gx_ini[44,7] = V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gx_ini[45,7] = -V_R14*i_d_R14*sin(delta_R14 - theta_R14) - V_R14*i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gx_ini[46,11] = -K_a_R14
        struct[0].Gx_ini[46,12] = K_ai_R14
        struct[0].Gx_ini[47,8] = -1/Droop_R14
        struct[0].Gx_ini[48,1] = 1/2
        struct[0].Gx_ini[48,8] = 1/2
        struct[0].Gx_ini[49,14] = K_sec_R10/2
        struct[0].Gx_ini[50,14] = K_sec_R14/2

        struct[0].Gy_ini[0,0] = 2*V_R01*g_R01_R02 + V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[0,1] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[0,2] = V_R01*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[0,3] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,0] = 2*V_R01*(-b_R01_R02 - bs_R01_R02/2) + V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[2,0] = V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[2,1] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[2,2] = V_R01*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + 2*V_R02*(g_R01_R02 + g_R02_R03) + V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[2,3] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[2,4] = V_R02*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[2,5] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,0] = V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[3,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[3,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + 2*V_R02*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[4,2] = V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[4,3] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[4,4] = V_R02*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + 2*V_R03*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[4,5] = V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[4,6] = V_R03*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[4,7] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[4,20] = V_R03*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[4,21] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,2] = V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[5,3] = V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[5,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + 2*V_R03*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[5,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[5,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[6,4] = V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[6,5] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[6,6] = V_R03*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + 2*V_R04*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[6,7] = V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[6,8] = V_R04*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[6,9] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[6,22] = V_R04*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[6,23] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,4] = V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[7,5] = V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[7,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + 2*V_R04*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[7,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[7,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[8,6] = V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[8,7] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[8,8] = V_R04*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + 2*V_R05*(g_R04_R05 + g_R05_R06) + V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[8,9] = V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[8,10] = V_R05*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[8,11] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,6] = V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[9,7] = V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[9,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + 2*V_R05*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[10,8] = V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[10,9] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[10,10] = V_R05*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + 2*V_R06*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[10,11] = V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[10,12] = V_R06*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[10,13] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[10,30] = V_R06*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[10,31] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,8] = V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[11,9] = V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[11,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + 2*V_R06*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[11,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[11,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[12,10] = V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[12,11] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[12,12] = V_R06*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + 2*V_R07*(g_R06_R07 + g_R07_R08) + V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[12,13] = V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[12,14] = V_R07*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[12,15] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,10] = V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[13,11] = V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[13,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + 2*V_R07*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[14,12] = V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[14,13] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[14,14] = V_R07*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + 2*V_R08*(g_R07_R08 + g_R08_R09) + V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[14,15] = V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[14,16] = V_R08*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[14,17] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,12] = V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[15,13] = V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[15,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + 2*V_R08*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[16,14] = V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[16,15] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[16,16] = V_R08*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + 2*V_R09*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[16,17] = V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[16,18] = V_R09*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[16,19] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[16,32] = V_R09*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[16,33] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,14] = V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[17,15] = V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[17,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + 2*V_R09*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[17,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[17,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[18,16] = V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[18,17] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[18,18] = V_R09*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + 2*V_R10*(g_R09_R10 + g_R10_R18) + V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,19] = V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,34] = V_R10*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,35] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,38] = -S_n_R10/S_base
        struct[0].Gy_ini[19,16] = V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[19,17] = V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[19,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + 2*V_R10*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,39] = -S_n_R10/S_base
        struct[0].Gy_ini[20,4] = V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[20,5] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[20,20] = V_R03*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + 2*V_R11*g_R03_R11
        struct[0].Gy_ini[20,21] = V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[21,4] = V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[21,5] = V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[21,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + 2*V_R11*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].Gy_ini[21,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[22,6] = V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[22,7] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[22,22] = V_R04*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + 2*V_R12*(g_R04_R12 + g_R12_R13) + V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[22,23] = V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[22,24] = V_R12*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[22,25] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,6] = V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[23,7] = V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[23,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + 2*V_R12*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[24,22] = V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[24,23] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[24,24] = V_R12*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + 2*V_R13*(g_R12_R13 + g_R13_R14) + V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[24,25] = V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[24,26] = V_R13*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[24,27] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,22] = V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[25,23] = V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[25,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + 2*V_R13*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[26,24] = V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[26,25] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[26,26] = V_R13*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + 2*V_R14*(g_R13_R14 + g_R14_R15) + V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,27] = V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,28] = V_R14*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,29] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,44] = -S_n_R14/S_base
        struct[0].Gy_ini[27,24] = V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[27,25] = V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[27,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + 2*V_R14*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,45] = -S_n_R14/S_base
        struct[0].Gy_ini[28,26] = V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[28,27] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[28,28] = V_R14*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + 2*V_R15*g_R14_R15
        struct[0].Gy_ini[28,29] = V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[29,26] = V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[29,27] = V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[29,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + 2*V_R15*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].Gy_ini[29,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[30,10] = V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[30,11] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[30,30] = V_R06*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + 2*V_R16*g_R06_R16
        struct[0].Gy_ini[30,31] = V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[31,10] = V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[31,11] = V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[31,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + 2*V_R16*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].Gy_ini[31,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[32,16] = V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[32,17] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[32,32] = V_R09*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + 2*V_R17*g_R09_R17
        struct[0].Gy_ini[32,33] = V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[33,16] = V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[33,17] = V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[33,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + 2*V_R17*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].Gy_ini[33,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[34,18] = V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[34,19] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[34,34] = V_R10*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + 2*V_R18*g_R10_R18
        struct[0].Gy_ini[34,35] = V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[35,18] = V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[35,19] = V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[35,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + 2*V_R18*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].Gy_ini[35,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[36,18] = cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[36,19] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[36,36] = X1d_R10
        struct[0].Gy_ini[36,37] = R_a_R10
        struct[0].Gy_ini[37,18] = sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[37,19] = -V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[37,36] = R_a_R10
        struct[0].Gy_ini[37,37] = -X1q_R10
        struct[0].Gy_ini[38,18] = i_d_R10*sin(delta_R10 - theta_R10) + i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[38,19] = -V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[38,36] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[38,37] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,18] = i_d_R10*cos(delta_R10 - theta_R10) - i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,19] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,36] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,37] = -V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[42,26] = cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[42,27] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[42,42] = X1d_R14
        struct[0].Gy_ini[42,43] = R_a_R14
        struct[0].Gy_ini[43,26] = sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[43,27] = -V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[43,42] = R_a_R14
        struct[0].Gy_ini[43,43] = -X1q_R14
        struct[0].Gy_ini[44,26] = i_d_R14*sin(delta_R14 - theta_R14) + i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[44,27] = -V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[44,42] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[44,43] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,26] = i_d_R14*cos(delta_R14 - theta_R14) - i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,27] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,42] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,43] = -V_R14*sin(delta_R14 - theta_R14)



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_R01_R02 = struct[0].g_R01_R02
    b_R01_R02 = struct[0].b_R01_R02
    bs_R01_R02 = struct[0].bs_R01_R02
    g_R02_R03 = struct[0].g_R02_R03
    b_R02_R03 = struct[0].b_R02_R03
    bs_R02_R03 = struct[0].bs_R02_R03
    g_R03_R04 = struct[0].g_R03_R04
    b_R03_R04 = struct[0].b_R03_R04
    bs_R03_R04 = struct[0].bs_R03_R04
    g_R04_R05 = struct[0].g_R04_R05
    b_R04_R05 = struct[0].b_R04_R05
    bs_R04_R05 = struct[0].bs_R04_R05
    g_R05_R06 = struct[0].g_R05_R06
    b_R05_R06 = struct[0].b_R05_R06
    bs_R05_R06 = struct[0].bs_R05_R06
    g_R06_R07 = struct[0].g_R06_R07
    b_R06_R07 = struct[0].b_R06_R07
    bs_R06_R07 = struct[0].bs_R06_R07
    g_R07_R08 = struct[0].g_R07_R08
    b_R07_R08 = struct[0].b_R07_R08
    bs_R07_R08 = struct[0].bs_R07_R08
    g_R08_R09 = struct[0].g_R08_R09
    b_R08_R09 = struct[0].b_R08_R09
    bs_R08_R09 = struct[0].bs_R08_R09
    g_R09_R10 = struct[0].g_R09_R10
    b_R09_R10 = struct[0].b_R09_R10
    bs_R09_R10 = struct[0].bs_R09_R10
    g_R03_R11 = struct[0].g_R03_R11
    b_R03_R11 = struct[0].b_R03_R11
    bs_R03_R11 = struct[0].bs_R03_R11
    g_R04_R12 = struct[0].g_R04_R12
    b_R04_R12 = struct[0].b_R04_R12
    bs_R04_R12 = struct[0].bs_R04_R12
    g_R12_R13 = struct[0].g_R12_R13
    b_R12_R13 = struct[0].b_R12_R13
    bs_R12_R13 = struct[0].bs_R12_R13
    g_R13_R14 = struct[0].g_R13_R14
    b_R13_R14 = struct[0].b_R13_R14
    bs_R13_R14 = struct[0].bs_R13_R14
    g_R14_R15 = struct[0].g_R14_R15
    b_R14_R15 = struct[0].b_R14_R15
    bs_R14_R15 = struct[0].bs_R14_R15
    g_R06_R16 = struct[0].g_R06_R16
    b_R06_R16 = struct[0].b_R06_R16
    bs_R06_R16 = struct[0].bs_R06_R16
    g_R09_R17 = struct[0].g_R09_R17
    b_R09_R17 = struct[0].b_R09_R17
    bs_R09_R17 = struct[0].bs_R09_R17
    g_R10_R18 = struct[0].g_R10_R18
    b_R10_R18 = struct[0].b_R10_R18
    bs_R10_R18 = struct[0].bs_R10_R18
    U_R01_n = struct[0].U_R01_n
    U_R02_n = struct[0].U_R02_n
    U_R03_n = struct[0].U_R03_n
    U_R04_n = struct[0].U_R04_n
    U_R05_n = struct[0].U_R05_n
    U_R06_n = struct[0].U_R06_n
    U_R07_n = struct[0].U_R07_n
    U_R08_n = struct[0].U_R08_n
    U_R09_n = struct[0].U_R09_n
    U_R10_n = struct[0].U_R10_n
    U_R11_n = struct[0].U_R11_n
    U_R12_n = struct[0].U_R12_n
    U_R13_n = struct[0].U_R13_n
    U_R14_n = struct[0].U_R14_n
    U_R15_n = struct[0].U_R15_n
    U_R16_n = struct[0].U_R16_n
    U_R17_n = struct[0].U_R17_n
    U_R18_n = struct[0].U_R18_n
    S_n_R10 = struct[0].S_n_R10
    H_R10 = struct[0].H_R10
    Omega_b_R10 = struct[0].Omega_b_R10
    T1d0_R10 = struct[0].T1d0_R10
    T1q0_R10 = struct[0].T1q0_R10
    X_d_R10 = struct[0].X_d_R10
    X_q_R10 = struct[0].X_q_R10
    X1d_R10 = struct[0].X1d_R10
    X1q_R10 = struct[0].X1q_R10
    D_R10 = struct[0].D_R10
    R_a_R10 = struct[0].R_a_R10
    K_delta_R10 = struct[0].K_delta_R10
    K_a_R10 = struct[0].K_a_R10
    K_ai_R10 = struct[0].K_ai_R10
    T_r_R10 = struct[0].T_r_R10
    Droop_R10 = struct[0].Droop_R10
    T_m_R10 = struct[0].T_m_R10
    S_n_R14 = struct[0].S_n_R14
    H_R14 = struct[0].H_R14
    Omega_b_R14 = struct[0].Omega_b_R14
    T1d0_R14 = struct[0].T1d0_R14
    T1q0_R14 = struct[0].T1q0_R14
    X_d_R14 = struct[0].X_d_R14
    X_q_R14 = struct[0].X_q_R14
    X1d_R14 = struct[0].X1d_R14
    X1q_R14 = struct[0].X1q_R14
    D_R14 = struct[0].D_R14
    R_a_R14 = struct[0].R_a_R14
    K_delta_R14 = struct[0].K_delta_R14
    K_a_R14 = struct[0].K_a_R14
    K_ai_R14 = struct[0].K_ai_R14
    T_r_R14 = struct[0].T_r_R14
    Droop_R14 = struct[0].Droop_R14
    T_m_R14 = struct[0].T_m_R14
    K_sec_R10 = struct[0].K_sec_R10
    K_sec_R14 = struct[0].K_sec_R14
    
    # Inputs:
    P_R01 = struct[0].P_R01
    Q_R01 = struct[0].Q_R01
    P_R02 = struct[0].P_R02
    Q_R02 = struct[0].Q_R02
    P_R03 = struct[0].P_R03
    Q_R03 = struct[0].Q_R03
    P_R04 = struct[0].P_R04
    Q_R04 = struct[0].Q_R04
    P_R05 = struct[0].P_R05
    Q_R05 = struct[0].Q_R05
    P_R06 = struct[0].P_R06
    Q_R06 = struct[0].Q_R06
    P_R07 = struct[0].P_R07
    Q_R07 = struct[0].Q_R07
    P_R08 = struct[0].P_R08
    Q_R08 = struct[0].Q_R08
    P_R09 = struct[0].P_R09
    Q_R09 = struct[0].Q_R09
    P_R10 = struct[0].P_R10
    Q_R10 = struct[0].Q_R10
    P_R11 = struct[0].P_R11
    Q_R11 = struct[0].Q_R11
    P_R12 = struct[0].P_R12
    Q_R12 = struct[0].Q_R12
    P_R13 = struct[0].P_R13
    Q_R13 = struct[0].Q_R13
    P_R14 = struct[0].P_R14
    Q_R14 = struct[0].Q_R14
    P_R15 = struct[0].P_R15
    Q_R15 = struct[0].Q_R15
    P_R16 = struct[0].P_R16
    Q_R16 = struct[0].Q_R16
    P_R17 = struct[0].P_R17
    Q_R17 = struct[0].Q_R17
    P_R18 = struct[0].P_R18
    Q_R18 = struct[0].Q_R18
    v_ref_R10 = struct[0].v_ref_R10
    v_pss_R10 = struct[0].v_pss_R10
    p_c_R10 = struct[0].p_c_R10
    v_ref_R14 = struct[0].v_ref_R14
    v_pss_R14 = struct[0].v_pss_R14
    p_c_R14 = struct[0].p_c_R14
    
    # Dynamical states:
    delta_R10 = struct[0].x[0,0]
    omega_R10 = struct[0].x[1,0]
    e1q_R10 = struct[0].x[2,0]
    e1d_R10 = struct[0].x[3,0]
    v_c_R10 = struct[0].x[4,0]
    xi_v_R10 = struct[0].x[5,0]
    p_m_R10 = struct[0].x[6,0]
    delta_R14 = struct[0].x[7,0]
    omega_R14 = struct[0].x[8,0]
    e1q_R14 = struct[0].x[9,0]
    e1d_R14 = struct[0].x[10,0]
    v_c_R14 = struct[0].x[11,0]
    xi_v_R14 = struct[0].x[12,0]
    p_m_R14 = struct[0].x[13,0]
    xi_freq = struct[0].x[14,0]
    
    # Algebraic states:
    V_R01 = struct[0].y_run[0,0]
    theta_R01 = struct[0].y_run[1,0]
    V_R02 = struct[0].y_run[2,0]
    theta_R02 = struct[0].y_run[3,0]
    V_R03 = struct[0].y_run[4,0]
    theta_R03 = struct[0].y_run[5,0]
    V_R04 = struct[0].y_run[6,0]
    theta_R04 = struct[0].y_run[7,0]
    V_R05 = struct[0].y_run[8,0]
    theta_R05 = struct[0].y_run[9,0]
    V_R06 = struct[0].y_run[10,0]
    theta_R06 = struct[0].y_run[11,0]
    V_R07 = struct[0].y_run[12,0]
    theta_R07 = struct[0].y_run[13,0]
    V_R08 = struct[0].y_run[14,0]
    theta_R08 = struct[0].y_run[15,0]
    V_R09 = struct[0].y_run[16,0]
    theta_R09 = struct[0].y_run[17,0]
    V_R10 = struct[0].y_run[18,0]
    theta_R10 = struct[0].y_run[19,0]
    V_R11 = struct[0].y_run[20,0]
    theta_R11 = struct[0].y_run[21,0]
    V_R12 = struct[0].y_run[22,0]
    theta_R12 = struct[0].y_run[23,0]
    V_R13 = struct[0].y_run[24,0]
    theta_R13 = struct[0].y_run[25,0]
    V_R14 = struct[0].y_run[26,0]
    theta_R14 = struct[0].y_run[27,0]
    V_R15 = struct[0].y_run[28,0]
    theta_R15 = struct[0].y_run[29,0]
    V_R16 = struct[0].y_run[30,0]
    theta_R16 = struct[0].y_run[31,0]
    V_R17 = struct[0].y_run[32,0]
    theta_R17 = struct[0].y_run[33,0]
    V_R18 = struct[0].y_run[34,0]
    theta_R18 = struct[0].y_run[35,0]
    i_d_R10 = struct[0].y_run[36,0]
    i_q_R10 = struct[0].y_run[37,0]
    p_g_R10_1 = struct[0].y_run[38,0]
    q_g_R10_1 = struct[0].y_run[39,0]
    v_f_R10 = struct[0].y_run[40,0]
    p_m_ref_R10 = struct[0].y_run[41,0]
    i_d_R14 = struct[0].y_run[42,0]
    i_q_R14 = struct[0].y_run[43,0]
    p_g_R14_1 = struct[0].y_run[44,0]
    q_g_R14_1 = struct[0].y_run[45,0]
    v_f_R14 = struct[0].y_run[46,0]
    p_m_ref_R14 = struct[0].y_run[47,0]
    omega_coi = struct[0].y_run[48,0]
    p_r_R10 = struct[0].y_run[49,0]
    p_r_R14 = struct[0].y_run[50,0]
    
    struct[0].u_run[0,0] = P_R01
    struct[0].u_run[1,0] = Q_R01
    struct[0].u_run[2,0] = P_R02
    struct[0].u_run[3,0] = Q_R02
    struct[0].u_run[4,0] = P_R03
    struct[0].u_run[5,0] = Q_R03
    struct[0].u_run[6,0] = P_R04
    struct[0].u_run[7,0] = Q_R04
    struct[0].u_run[8,0] = P_R05
    struct[0].u_run[9,0] = Q_R05
    struct[0].u_run[10,0] = P_R06
    struct[0].u_run[11,0] = Q_R06
    struct[0].u_run[12,0] = P_R07
    struct[0].u_run[13,0] = Q_R07
    struct[0].u_run[14,0] = P_R08
    struct[0].u_run[15,0] = Q_R08
    struct[0].u_run[16,0] = P_R09
    struct[0].u_run[17,0] = Q_R09
    struct[0].u_run[18,0] = P_R10
    struct[0].u_run[19,0] = Q_R10
    struct[0].u_run[20,0] = P_R11
    struct[0].u_run[21,0] = Q_R11
    struct[0].u_run[22,0] = P_R12
    struct[0].u_run[23,0] = Q_R12
    struct[0].u_run[24,0] = P_R13
    struct[0].u_run[25,0] = Q_R13
    struct[0].u_run[26,0] = P_R14
    struct[0].u_run[27,0] = Q_R14
    struct[0].u_run[28,0] = P_R15
    struct[0].u_run[29,0] = Q_R15
    struct[0].u_run[30,0] = P_R16
    struct[0].u_run[31,0] = Q_R16
    struct[0].u_run[32,0] = P_R17
    struct[0].u_run[33,0] = Q_R17
    struct[0].u_run[34,0] = P_R18
    struct[0].u_run[35,0] = Q_R18
    struct[0].u_run[36,0] = v_ref_R10
    struct[0].u_run[37,0] = v_pss_R10
    struct[0].u_run[38,0] = p_c_R10
    struct[0].u_run[39,0] = v_ref_R14
    struct[0].u_run[40,0] = v_pss_R14
    struct[0].u_run[41,0] = p_c_R14
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_R10*delta_R10 + Omega_b_R10*(omega_R10 - omega_coi)
        struct[0].f[1,0] = (-D_R10*(omega_R10 - omega_coi) - i_d_R10*(R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10)) - i_q_R10*(R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10)) + p_m_R10)/(2*H_R10)
        struct[0].f[2,0] = (-e1q_R10 - i_d_R10*(-X1d_R10 + X_d_R10) + v_f_R10)/T1d0_R10
        struct[0].f[3,0] = (-e1d_R10 + i_q_R10*(-X1q_R10 + X_q_R10))/T1q0_R10
        struct[0].f[4,0] = (V_R10 - v_c_R10)/T_r_R10
        struct[0].f[5,0] = -V_R10 + v_ref_R10
        struct[0].f[6,0] = (-p_m_R10 + p_m_ref_R10)/T_m_R10
        struct[0].f[7,0] = -K_delta_R14*delta_R14 + Omega_b_R14*(omega_R14 - omega_coi)
        struct[0].f[8,0] = (-D_R14*(omega_R14 - omega_coi) - i_d_R14*(R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14)) - i_q_R14*(R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14)) + p_m_R14)/(2*H_R14)
        struct[0].f[9,0] = (-e1q_R14 - i_d_R14*(-X1d_R14 + X_d_R14) + v_f_R14)/T1d0_R14
        struct[0].f[10,0] = (-e1d_R14 + i_q_R14*(-X1q_R14 + X_q_R14))/T1q0_R14
        struct[0].f[11,0] = (V_R14 - v_c_R14)/T_r_R14
        struct[0].f[12,0] = -V_R14 + v_ref_R14
        struct[0].f[13,0] = (-p_m_R14 + p_m_ref_R14)/T_m_R14
        struct[0].f[14,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_R01/S_base + V_R01**2*g_R01_R02 + V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].g[1,0] = -Q_R01/S_base + V_R01**2*(-b_R01_R02 - bs_R01_R02/2) + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].g[2,0] = -P_R02/S_base + V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02**2*(g_R01_R02 + g_R02_R03) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].g[3,0] = -Q_R02/S_base + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02**2*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].g[4,0] = -P_R03/S_base + V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03**2*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].g[5,0] = -Q_R03/S_base + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03**2*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].g[6,0] = -P_R04/S_base + V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04**2*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].g[7,0] = -Q_R04/S_base + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04**2*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].g[8,0] = -P_R05/S_base + V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05**2*(g_R04_R05 + g_R05_R06) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].g[9,0] = -Q_R05/S_base + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05**2*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].g[10,0] = -P_R06/S_base + V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06**2*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].g[11,0] = -Q_R06/S_base + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06**2*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].g[12,0] = -P_R07/S_base + V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07**2*(g_R06_R07 + g_R07_R08) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].g[13,0] = -Q_R07/S_base + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07**2*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].g[14,0] = -P_R08/S_base + V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08**2*(g_R07_R08 + g_R08_R09) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].g[15,0] = -Q_R08/S_base + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08**2*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].g[16,0] = -P_R09/S_base + V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09**2*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].g[17,0] = -Q_R09/S_base + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09**2*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].g[18,0] = -P_R10/S_base + V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10**2*(g_R09_R10 + g_R10_R18) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) - S_n_R10*p_g_R10_1/S_base
        struct[0].g[19,0] = -Q_R10/S_base + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10**2*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18)) - S_n_R10*q_g_R10_1/S_base
        struct[0].g[20,0] = -P_R11/S_base + V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + V_R11**2*g_R03_R11
        struct[0].g[21,0] = -Q_R11/S_base + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + V_R11**2*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].g[22,0] = -P_R12/S_base + V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12**2*(g_R04_R12 + g_R12_R13) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].g[23,0] = -Q_R12/S_base + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12**2*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].g[24,0] = -P_R13/S_base + V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13**2*(g_R12_R13 + g_R13_R14) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].g[25,0] = -Q_R13/S_base + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13**2*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].g[26,0] = -P_R14/S_base + V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14**2*(g_R13_R14 + g_R14_R15) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) - S_n_R14*p_g_R14_1/S_base
        struct[0].g[27,0] = -Q_R14/S_base + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14**2*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15)) - S_n_R14*q_g_R14_1/S_base
        struct[0].g[28,0] = -P_R15/S_base + V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + V_R15**2*g_R14_R15
        struct[0].g[29,0] = -Q_R15/S_base + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + V_R15**2*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].g[30,0] = -P_R16/S_base + V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + V_R16**2*g_R06_R16
        struct[0].g[31,0] = -Q_R16/S_base + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + V_R16**2*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].g[32,0] = -P_R17/S_base + V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + V_R17**2*g_R09_R17
        struct[0].g[33,0] = -Q_R17/S_base + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + V_R17**2*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].g[34,0] = -P_R18/S_base + V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + V_R18**2*g_R10_R18
        struct[0].g[35,0] = -Q_R18/S_base + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + V_R18**2*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].g[36,0] = R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10) + X1d_R10*i_d_R10 - e1q_R10
        struct[0].g[37,0] = R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10) - X1q_R10*i_q_R10 - e1d_R10
        struct[0].g[38,0] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10) - p_g_R10_1
        struct[0].g[39,0] = V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10) - q_g_R10_1
        struct[0].g[40,0] = K_a_R10*(-v_c_R10 + v_pss_R10 + v_ref_R10) + K_ai_R10*xi_v_R10 - v_f_R10
        struct[0].g[41,0] = p_c_R10 - p_m_ref_R10 + p_r_R10 - (omega_R10 - 1)/Droop_R10
        struct[0].g[42,0] = R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14) + X1d_R14*i_d_R14 - e1q_R14
        struct[0].g[43,0] = R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14) - X1q_R14*i_q_R14 - e1d_R14
        struct[0].g[44,0] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14) - p_g_R14_1
        struct[0].g[45,0] = V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14) - q_g_R14_1
        struct[0].g[46,0] = K_a_R14*(-v_c_R14 + v_pss_R14 + v_ref_R14) + K_ai_R14*xi_v_R14 - v_f_R14
        struct[0].g[47,0] = p_c_R14 - p_m_ref_R14 + p_r_R14 - (omega_R14 - 1)/Droop_R14
        struct[0].g[48,0] = omega_R10/2 + omega_R14/2 - omega_coi
        struct[0].g[49,0] = K_sec_R10*xi_freq/2 - p_r_R10
        struct[0].g[50,0] = K_sec_R14*xi_freq/2 - p_r_R14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_R01
        struct[0].h[1,0] = V_R02
        struct[0].h[2,0] = V_R03
        struct[0].h[3,0] = V_R04
        struct[0].h[4,0] = V_R05
        struct[0].h[5,0] = V_R06
        struct[0].h[6,0] = V_R07
        struct[0].h[7,0] = V_R08
        struct[0].h[8,0] = V_R09
        struct[0].h[9,0] = V_R10
        struct[0].h[10,0] = V_R11
        struct[0].h[11,0] = V_R12
        struct[0].h[12,0] = V_R13
        struct[0].h[13,0] = V_R14
        struct[0].h[14,0] = V_R15
        struct[0].h[15,0] = V_R16
        struct[0].h[16,0] = V_R17
        struct[0].h[17,0] = V_R18
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_R10
        struct[0].Fx[0,1] = Omega_b_R10
        struct[0].Fx[1,0] = (-V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fx[1,1] = -D_R10/(2*H_R10)
        struct[0].Fx[1,6] = 1/(2*H_R10)
        struct[0].Fx[2,2] = -1/T1d0_R10
        struct[0].Fx[3,3] = -1/T1q0_R10
        struct[0].Fx[4,4] = -1/T_r_R10
        struct[0].Fx[6,6] = -1/T_m_R10
        struct[0].Fx[7,7] = -K_delta_R14
        struct[0].Fx[7,8] = Omega_b_R14
        struct[0].Fx[8,7] = (-V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fx[8,8] = -D_R14/(2*H_R14)
        struct[0].Fx[8,13] = 1/(2*H_R14)
        struct[0].Fx[9,9] = -1/T1d0_R14
        struct[0].Fx[10,10] = -1/T1q0_R14
        struct[0].Fx[11,11] = -1/T_r_R14
        struct[0].Fx[13,13] = -1/T_m_R14

    if mode == 11:

        struct[0].Fy[0,48] = -Omega_b_R10
        struct[0].Fy[1,18] = (-i_d_R10*sin(delta_R10 - theta_R10) - i_q_R10*cos(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,19] = (V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,36] = (-2*R_a_R10*i_d_R10 - V_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,37] = (-2*R_a_R10*i_q_R10 - V_R10*cos(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,48] = D_R10/(2*H_R10)
        struct[0].Fy[2,36] = (X1d_R10 - X_d_R10)/T1d0_R10
        struct[0].Fy[2,40] = 1/T1d0_R10
        struct[0].Fy[3,37] = (-X1q_R10 + X_q_R10)/T1q0_R10
        struct[0].Fy[4,18] = 1/T_r_R10
        struct[0].Fy[5,18] = -1
        struct[0].Fy[6,41] = 1/T_m_R10
        struct[0].Fy[7,48] = -Omega_b_R14
        struct[0].Fy[8,26] = (-i_d_R14*sin(delta_R14 - theta_R14) - i_q_R14*cos(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,27] = (V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,42] = (-2*R_a_R14*i_d_R14 - V_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,43] = (-2*R_a_R14*i_q_R14 - V_R14*cos(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,48] = D_R14/(2*H_R14)
        struct[0].Fy[9,42] = (X1d_R14 - X_d_R14)/T1d0_R14
        struct[0].Fy[9,46] = 1/T1d0_R14
        struct[0].Fy[10,43] = (-X1q_R14 + X_q_R14)/T1q0_R14
        struct[0].Fy[11,26] = 1/T_r_R14
        struct[0].Fy[12,26] = -1
        struct[0].Fy[13,47] = 1/T_m_R14
        struct[0].Fy[14,48] = -1

        struct[0].Gx[36,0] = -V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gx[36,2] = -1
        struct[0].Gx[37,0] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gx[37,3] = -1
        struct[0].Gx[38,0] = V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gx[39,0] = -V_R10*i_d_R10*sin(delta_R10 - theta_R10) - V_R10*i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gx[40,4] = -K_a_R10
        struct[0].Gx[40,5] = K_ai_R10
        struct[0].Gx[41,1] = -1/Droop_R10
        struct[0].Gx[42,7] = -V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gx[42,9] = -1
        struct[0].Gx[43,7] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gx[43,10] = -1
        struct[0].Gx[44,7] = V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gx[45,7] = -V_R14*i_d_R14*sin(delta_R14 - theta_R14) - V_R14*i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gx[46,11] = -K_a_R14
        struct[0].Gx[46,12] = K_ai_R14
        struct[0].Gx[47,8] = -1/Droop_R14
        struct[0].Gx[48,1] = 1/2
        struct[0].Gx[48,8] = 1/2
        struct[0].Gx[49,14] = K_sec_R10/2
        struct[0].Gx[50,14] = K_sec_R14/2

        struct[0].Gy[0,0] = 2*V_R01*g_R01_R02 + V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[0,1] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[0,2] = V_R01*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[0,3] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[1,0] = 2*V_R01*(-b_R01_R02 - bs_R01_R02/2) + V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[1,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[1,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[1,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[2,0] = V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[2,1] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[2,2] = V_R01*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + 2*V_R02*(g_R01_R02 + g_R02_R03) + V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[2,3] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[2,4] = V_R02*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[2,5] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[3,0] = V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[3,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[3,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + 2*V_R02*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[3,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[3,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[3,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[4,2] = V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[4,3] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[4,4] = V_R02*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + 2*V_R03*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[4,5] = V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[4,6] = V_R03*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[4,7] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[4,20] = V_R03*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[4,21] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[5,2] = V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[5,3] = V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[5,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + 2*V_R03*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[5,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[5,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[5,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[5,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[5,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[6,4] = V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[6,5] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[6,6] = V_R03*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + 2*V_R04*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[6,7] = V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[6,8] = V_R04*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[6,9] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[6,22] = V_R04*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[6,23] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[7,4] = V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[7,5] = V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[7,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + 2*V_R04*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[7,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[7,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[7,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[7,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[7,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[8,6] = V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[8,7] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[8,8] = V_R04*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + 2*V_R05*(g_R04_R05 + g_R05_R06) + V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[8,9] = V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[8,10] = V_R05*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[8,11] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[9,6] = V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[9,7] = V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[9,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + 2*V_R05*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[9,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[9,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[9,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[10,8] = V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[10,9] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[10,10] = V_R05*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + 2*V_R06*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[10,11] = V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[10,12] = V_R06*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[10,13] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[10,30] = V_R06*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[10,31] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[11,8] = V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[11,9] = V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[11,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + 2*V_R06*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[11,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[11,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[11,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[11,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[11,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[12,10] = V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[12,11] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[12,12] = V_R06*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + 2*V_R07*(g_R06_R07 + g_R07_R08) + V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[12,13] = V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[12,14] = V_R07*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[12,15] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[13,10] = V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[13,11] = V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[13,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + 2*V_R07*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[13,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[13,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[13,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[14,12] = V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[14,13] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[14,14] = V_R07*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + 2*V_R08*(g_R07_R08 + g_R08_R09) + V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[14,15] = V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[14,16] = V_R08*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[14,17] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[15,12] = V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[15,13] = V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[15,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + 2*V_R08*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[15,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[15,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[15,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[16,14] = V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[16,15] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[16,16] = V_R08*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + 2*V_R09*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[16,17] = V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[16,18] = V_R09*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[16,19] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[16,32] = V_R09*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[16,33] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[17,14] = V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[17,15] = V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[17,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + 2*V_R09*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[17,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[17,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[17,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[17,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[17,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[18,16] = V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[18,17] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[18,18] = V_R09*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + 2*V_R10*(g_R09_R10 + g_R10_R18) + V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[18,19] = V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[18,34] = V_R10*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[18,35] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[18,38] = -S_n_R10/S_base
        struct[0].Gy[19,16] = V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[19,17] = V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[19,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + 2*V_R10*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[19,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[19,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[19,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[19,39] = -S_n_R10/S_base
        struct[0].Gy[20,4] = V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[20,5] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[20,20] = V_R03*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + 2*V_R11*g_R03_R11
        struct[0].Gy[20,21] = V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[21,4] = V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[21,5] = V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[21,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + 2*V_R11*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].Gy[21,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[22,6] = V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[22,7] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[22,22] = V_R04*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + 2*V_R12*(g_R04_R12 + g_R12_R13) + V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[22,23] = V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[22,24] = V_R12*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[22,25] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[23,6] = V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[23,7] = V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[23,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + 2*V_R12*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[23,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[23,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[23,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[24,22] = V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[24,23] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[24,24] = V_R12*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + 2*V_R13*(g_R12_R13 + g_R13_R14) + V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[24,25] = V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[24,26] = V_R13*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[24,27] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[25,22] = V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[25,23] = V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[25,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + 2*V_R13*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[25,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[25,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[25,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[26,24] = V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[26,25] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[26,26] = V_R13*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + 2*V_R14*(g_R13_R14 + g_R14_R15) + V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[26,27] = V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[26,28] = V_R14*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[26,29] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[26,44] = -S_n_R14/S_base
        struct[0].Gy[27,24] = V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[27,25] = V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[27,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + 2*V_R14*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[27,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[27,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[27,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[27,45] = -S_n_R14/S_base
        struct[0].Gy[28,26] = V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[28,27] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[28,28] = V_R14*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + 2*V_R15*g_R14_R15
        struct[0].Gy[28,29] = V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[29,26] = V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[29,27] = V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[29,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + 2*V_R15*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].Gy[29,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[30,10] = V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[30,11] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[30,30] = V_R06*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + 2*V_R16*g_R06_R16
        struct[0].Gy[30,31] = V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[31,10] = V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[31,11] = V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[31,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + 2*V_R16*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].Gy[31,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[32,16] = V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[32,17] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[32,32] = V_R09*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + 2*V_R17*g_R09_R17
        struct[0].Gy[32,33] = V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[33,16] = V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[33,17] = V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[33,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + 2*V_R17*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].Gy[33,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[34,18] = V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[34,19] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[34,34] = V_R10*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + 2*V_R18*g_R10_R18
        struct[0].Gy[34,35] = V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[35,18] = V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[35,19] = V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[35,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + 2*V_R18*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].Gy[35,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[36,18] = cos(delta_R10 - theta_R10)
        struct[0].Gy[36,19] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[36,36] = X1d_R10
        struct[0].Gy[36,37] = R_a_R10
        struct[0].Gy[37,18] = sin(delta_R10 - theta_R10)
        struct[0].Gy[37,19] = -V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[37,36] = R_a_R10
        struct[0].Gy[37,37] = -X1q_R10
        struct[0].Gy[38,18] = i_d_R10*sin(delta_R10 - theta_R10) + i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[38,19] = -V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[38,36] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[38,37] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[39,18] = i_d_R10*cos(delta_R10 - theta_R10) - i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[39,19] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[39,36] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[39,37] = -V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[42,26] = cos(delta_R14 - theta_R14)
        struct[0].Gy[42,27] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[42,42] = X1d_R14
        struct[0].Gy[42,43] = R_a_R14
        struct[0].Gy[43,26] = sin(delta_R14 - theta_R14)
        struct[0].Gy[43,27] = -V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[43,42] = R_a_R14
        struct[0].Gy[43,43] = -X1q_R14
        struct[0].Gy[44,26] = i_d_R14*sin(delta_R14 - theta_R14) + i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[44,27] = -V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[44,42] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[44,43] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[45,26] = i_d_R14*cos(delta_R14 - theta_R14) - i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[45,27] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[45,42] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[45,43] = -V_R14*sin(delta_R14 - theta_R14)

    if mode > 12:

        struct[0].Fu[5,36] = 1
        struct[0].Fu[12,39] = 1

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
        struct[0].Gu[22,22] = -1/S_base
        struct[0].Gu[23,23] = -1/S_base
        struct[0].Gu[24,24] = -1/S_base
        struct[0].Gu[25,25] = -1/S_base
        struct[0].Gu[26,26] = -1/S_base
        struct[0].Gu[27,27] = -1/S_base
        struct[0].Gu[28,28] = -1/S_base
        struct[0].Gu[29,29] = -1/S_base
        struct[0].Gu[30,30] = -1/S_base
        struct[0].Gu[31,31] = -1/S_base
        struct[0].Gu[32,32] = -1/S_base
        struct[0].Gu[33,33] = -1/S_base
        struct[0].Gu[34,34] = -1/S_base
        struct[0].Gu[35,35] = -1/S_base
        struct[0].Gu[40,36] = K_a_R10
        struct[0].Gu[40,37] = K_a_R10
        struct[0].Gu[46,39] = K_a_R14
        struct[0].Gu[46,40] = K_a_R14


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
        struct[0].Hy[11,22] = 1
        struct[0].Hy[12,24] = 1
        struct[0].Hy[13,26] = 1
        struct[0].Hy[14,28] = 1
        struct[0].Hy[15,30] = 1
        struct[0].Hy[16,32] = 1
        struct[0].Hy[17,34] = 1




def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_R01_R02 = struct[0].g_R01_R02
    b_R01_R02 = struct[0].b_R01_R02
    bs_R01_R02 = struct[0].bs_R01_R02
    g_R02_R03 = struct[0].g_R02_R03
    b_R02_R03 = struct[0].b_R02_R03
    bs_R02_R03 = struct[0].bs_R02_R03
    g_R03_R04 = struct[0].g_R03_R04
    b_R03_R04 = struct[0].b_R03_R04
    bs_R03_R04 = struct[0].bs_R03_R04
    g_R04_R05 = struct[0].g_R04_R05
    b_R04_R05 = struct[0].b_R04_R05
    bs_R04_R05 = struct[0].bs_R04_R05
    g_R05_R06 = struct[0].g_R05_R06
    b_R05_R06 = struct[0].b_R05_R06
    bs_R05_R06 = struct[0].bs_R05_R06
    g_R06_R07 = struct[0].g_R06_R07
    b_R06_R07 = struct[0].b_R06_R07
    bs_R06_R07 = struct[0].bs_R06_R07
    g_R07_R08 = struct[0].g_R07_R08
    b_R07_R08 = struct[0].b_R07_R08
    bs_R07_R08 = struct[0].bs_R07_R08
    g_R08_R09 = struct[0].g_R08_R09
    b_R08_R09 = struct[0].b_R08_R09
    bs_R08_R09 = struct[0].bs_R08_R09
    g_R09_R10 = struct[0].g_R09_R10
    b_R09_R10 = struct[0].b_R09_R10
    bs_R09_R10 = struct[0].bs_R09_R10
    g_R03_R11 = struct[0].g_R03_R11
    b_R03_R11 = struct[0].b_R03_R11
    bs_R03_R11 = struct[0].bs_R03_R11
    g_R04_R12 = struct[0].g_R04_R12
    b_R04_R12 = struct[0].b_R04_R12
    bs_R04_R12 = struct[0].bs_R04_R12
    g_R12_R13 = struct[0].g_R12_R13
    b_R12_R13 = struct[0].b_R12_R13
    bs_R12_R13 = struct[0].bs_R12_R13
    g_R13_R14 = struct[0].g_R13_R14
    b_R13_R14 = struct[0].b_R13_R14
    bs_R13_R14 = struct[0].bs_R13_R14
    g_R14_R15 = struct[0].g_R14_R15
    b_R14_R15 = struct[0].b_R14_R15
    bs_R14_R15 = struct[0].bs_R14_R15
    g_R06_R16 = struct[0].g_R06_R16
    b_R06_R16 = struct[0].b_R06_R16
    bs_R06_R16 = struct[0].bs_R06_R16
    g_R09_R17 = struct[0].g_R09_R17
    b_R09_R17 = struct[0].b_R09_R17
    bs_R09_R17 = struct[0].bs_R09_R17
    g_R10_R18 = struct[0].g_R10_R18
    b_R10_R18 = struct[0].b_R10_R18
    bs_R10_R18 = struct[0].bs_R10_R18
    U_R01_n = struct[0].U_R01_n
    U_R02_n = struct[0].U_R02_n
    U_R03_n = struct[0].U_R03_n
    U_R04_n = struct[0].U_R04_n
    U_R05_n = struct[0].U_R05_n
    U_R06_n = struct[0].U_R06_n
    U_R07_n = struct[0].U_R07_n
    U_R08_n = struct[0].U_R08_n
    U_R09_n = struct[0].U_R09_n
    U_R10_n = struct[0].U_R10_n
    U_R11_n = struct[0].U_R11_n
    U_R12_n = struct[0].U_R12_n
    U_R13_n = struct[0].U_R13_n
    U_R14_n = struct[0].U_R14_n
    U_R15_n = struct[0].U_R15_n
    U_R16_n = struct[0].U_R16_n
    U_R17_n = struct[0].U_R17_n
    U_R18_n = struct[0].U_R18_n
    S_n_R10 = struct[0].S_n_R10
    H_R10 = struct[0].H_R10
    Omega_b_R10 = struct[0].Omega_b_R10
    T1d0_R10 = struct[0].T1d0_R10
    T1q0_R10 = struct[0].T1q0_R10
    X_d_R10 = struct[0].X_d_R10
    X_q_R10 = struct[0].X_q_R10
    X1d_R10 = struct[0].X1d_R10
    X1q_R10 = struct[0].X1q_R10
    D_R10 = struct[0].D_R10
    R_a_R10 = struct[0].R_a_R10
    K_delta_R10 = struct[0].K_delta_R10
    K_a_R10 = struct[0].K_a_R10
    K_ai_R10 = struct[0].K_ai_R10
    T_r_R10 = struct[0].T_r_R10
    Droop_R10 = struct[0].Droop_R10
    T_m_R10 = struct[0].T_m_R10
    S_n_R14 = struct[0].S_n_R14
    H_R14 = struct[0].H_R14
    Omega_b_R14 = struct[0].Omega_b_R14
    T1d0_R14 = struct[0].T1d0_R14
    T1q0_R14 = struct[0].T1q0_R14
    X_d_R14 = struct[0].X_d_R14
    X_q_R14 = struct[0].X_q_R14
    X1d_R14 = struct[0].X1d_R14
    X1q_R14 = struct[0].X1q_R14
    D_R14 = struct[0].D_R14
    R_a_R14 = struct[0].R_a_R14
    K_delta_R14 = struct[0].K_delta_R14
    K_a_R14 = struct[0].K_a_R14
    K_ai_R14 = struct[0].K_ai_R14
    T_r_R14 = struct[0].T_r_R14
    Droop_R14 = struct[0].Droop_R14
    T_m_R14 = struct[0].T_m_R14
    K_sec_R10 = struct[0].K_sec_R10
    K_sec_R14 = struct[0].K_sec_R14
    
    # Inputs:
    P_R01 = struct[0].P_R01
    Q_R01 = struct[0].Q_R01
    P_R02 = struct[0].P_R02
    Q_R02 = struct[0].Q_R02
    P_R03 = struct[0].P_R03
    Q_R03 = struct[0].Q_R03
    P_R04 = struct[0].P_R04
    Q_R04 = struct[0].Q_R04
    P_R05 = struct[0].P_R05
    Q_R05 = struct[0].Q_R05
    P_R06 = struct[0].P_R06
    Q_R06 = struct[0].Q_R06
    P_R07 = struct[0].P_R07
    Q_R07 = struct[0].Q_R07
    P_R08 = struct[0].P_R08
    Q_R08 = struct[0].Q_R08
    P_R09 = struct[0].P_R09
    Q_R09 = struct[0].Q_R09
    P_R10 = struct[0].P_R10
    Q_R10 = struct[0].Q_R10
    P_R11 = struct[0].P_R11
    Q_R11 = struct[0].Q_R11
    P_R12 = struct[0].P_R12
    Q_R12 = struct[0].Q_R12
    P_R13 = struct[0].P_R13
    Q_R13 = struct[0].Q_R13
    P_R14 = struct[0].P_R14
    Q_R14 = struct[0].Q_R14
    P_R15 = struct[0].P_R15
    Q_R15 = struct[0].Q_R15
    P_R16 = struct[0].P_R16
    Q_R16 = struct[0].Q_R16
    P_R17 = struct[0].P_R17
    Q_R17 = struct[0].Q_R17
    P_R18 = struct[0].P_R18
    Q_R18 = struct[0].Q_R18
    v_ref_R10 = struct[0].v_ref_R10
    v_pss_R10 = struct[0].v_pss_R10
    p_c_R10 = struct[0].p_c_R10
    v_ref_R14 = struct[0].v_ref_R14
    v_pss_R14 = struct[0].v_pss_R14
    p_c_R14 = struct[0].p_c_R14
    
    # Dynamical states:
    delta_R10 = struct[0].x[0,0]
    omega_R10 = struct[0].x[1,0]
    e1q_R10 = struct[0].x[2,0]
    e1d_R10 = struct[0].x[3,0]
    v_c_R10 = struct[0].x[4,0]
    xi_v_R10 = struct[0].x[5,0]
    p_m_R10 = struct[0].x[6,0]
    delta_R14 = struct[0].x[7,0]
    omega_R14 = struct[0].x[8,0]
    e1q_R14 = struct[0].x[9,0]
    e1d_R14 = struct[0].x[10,0]
    v_c_R14 = struct[0].x[11,0]
    xi_v_R14 = struct[0].x[12,0]
    p_m_R14 = struct[0].x[13,0]
    xi_freq = struct[0].x[14,0]
    
    # Algebraic states:
    V_R01 = struct[0].y_ini[0,0]
    theta_R01 = struct[0].y_ini[1,0]
    V_R02 = struct[0].y_ini[2,0]
    theta_R02 = struct[0].y_ini[3,0]
    V_R03 = struct[0].y_ini[4,0]
    theta_R03 = struct[0].y_ini[5,0]
    V_R04 = struct[0].y_ini[6,0]
    theta_R04 = struct[0].y_ini[7,0]
    V_R05 = struct[0].y_ini[8,0]
    theta_R05 = struct[0].y_ini[9,0]
    V_R06 = struct[0].y_ini[10,0]
    theta_R06 = struct[0].y_ini[11,0]
    V_R07 = struct[0].y_ini[12,0]
    theta_R07 = struct[0].y_ini[13,0]
    V_R08 = struct[0].y_ini[14,0]
    theta_R08 = struct[0].y_ini[15,0]
    V_R09 = struct[0].y_ini[16,0]
    theta_R09 = struct[0].y_ini[17,0]
    V_R10 = struct[0].y_ini[18,0]
    theta_R10 = struct[0].y_ini[19,0]
    V_R11 = struct[0].y_ini[20,0]
    theta_R11 = struct[0].y_ini[21,0]
    V_R12 = struct[0].y_ini[22,0]
    theta_R12 = struct[0].y_ini[23,0]
    V_R13 = struct[0].y_ini[24,0]
    theta_R13 = struct[0].y_ini[25,0]
    V_R14 = struct[0].y_ini[26,0]
    theta_R14 = struct[0].y_ini[27,0]
    V_R15 = struct[0].y_ini[28,0]
    theta_R15 = struct[0].y_ini[29,0]
    V_R16 = struct[0].y_ini[30,0]
    theta_R16 = struct[0].y_ini[31,0]
    V_R17 = struct[0].y_ini[32,0]
    theta_R17 = struct[0].y_ini[33,0]
    V_R18 = struct[0].y_ini[34,0]
    theta_R18 = struct[0].y_ini[35,0]
    i_d_R10 = struct[0].y_ini[36,0]
    i_q_R10 = struct[0].y_ini[37,0]
    p_g_R10_1 = struct[0].y_ini[38,0]
    q_g_R10_1 = struct[0].y_ini[39,0]
    v_f_R10 = struct[0].y_ini[40,0]
    p_m_ref_R10 = struct[0].y_ini[41,0]
    i_d_R14 = struct[0].y_ini[42,0]
    i_q_R14 = struct[0].y_ini[43,0]
    p_g_R14_1 = struct[0].y_ini[44,0]
    q_g_R14_1 = struct[0].y_ini[45,0]
    v_f_R14 = struct[0].y_ini[46,0]
    p_m_ref_R14 = struct[0].y_ini[47,0]
    omega_coi = struct[0].y_ini[48,0]
    p_r_R10 = struct[0].y_ini[49,0]
    p_r_R14 = struct[0].y_ini[50,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_R10*delta_R10 + Omega_b_R10*(omega_R10 - omega_coi)
        struct[0].f[1,0] = (-D_R10*(omega_R10 - omega_coi) - i_d_R10*(R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10)) - i_q_R10*(R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10)) + p_m_R10)/(2*H_R10)
        struct[0].f[2,0] = (-e1q_R10 - i_d_R10*(-X1d_R10 + X_d_R10) + v_f_R10)/T1d0_R10
        struct[0].f[3,0] = (-e1d_R10 + i_q_R10*(-X1q_R10 + X_q_R10))/T1q0_R10
        struct[0].f[4,0] = (V_R10 - v_c_R10)/T_r_R10
        struct[0].f[5,0] = -V_R10 + v_ref_R10
        struct[0].f[6,0] = (-p_m_R10 + p_m_ref_R10)/T_m_R10
        struct[0].f[7,0] = -K_delta_R14*delta_R14 + Omega_b_R14*(omega_R14 - omega_coi)
        struct[0].f[8,0] = (-D_R14*(omega_R14 - omega_coi) - i_d_R14*(R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14)) - i_q_R14*(R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14)) + p_m_R14)/(2*H_R14)
        struct[0].f[9,0] = (-e1q_R14 - i_d_R14*(-X1d_R14 + X_d_R14) + v_f_R14)/T1d0_R14
        struct[0].f[10,0] = (-e1d_R14 + i_q_R14*(-X1q_R14 + X_q_R14))/T1q0_R14
        struct[0].f[11,0] = (V_R14 - v_c_R14)/T_r_R14
        struct[0].f[12,0] = -V_R14 + v_ref_R14
        struct[0].f[13,0] = (-p_m_R14 + p_m_ref_R14)/T_m_R14
        struct[0].f[14,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_R01/S_base + V_R01**2*g_R01_R02 + V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].g[1,0] = -Q_R01/S_base + V_R01**2*(-b_R01_R02 - bs_R01_R02/2) + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].g[2,0] = -P_R02/S_base + V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02**2*(g_R01_R02 + g_R02_R03) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].g[3,0] = -Q_R02/S_base + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02**2*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].g[4,0] = -P_R03/S_base + V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03**2*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].g[5,0] = -Q_R03/S_base + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03**2*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].g[6,0] = -P_R04/S_base + V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04**2*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].g[7,0] = -Q_R04/S_base + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04**2*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].g[8,0] = -P_R05/S_base + V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05**2*(g_R04_R05 + g_R05_R06) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].g[9,0] = -Q_R05/S_base + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05**2*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].g[10,0] = -P_R06/S_base + V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06**2*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].g[11,0] = -Q_R06/S_base + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06**2*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].g[12,0] = -P_R07/S_base + V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07**2*(g_R06_R07 + g_R07_R08) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].g[13,0] = -Q_R07/S_base + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07**2*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].g[14,0] = -P_R08/S_base + V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08**2*(g_R07_R08 + g_R08_R09) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].g[15,0] = -Q_R08/S_base + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08**2*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].g[16,0] = -P_R09/S_base + V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09**2*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].g[17,0] = -Q_R09/S_base + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09**2*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].g[18,0] = -P_R10/S_base + V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10**2*(g_R09_R10 + g_R10_R18) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) - S_n_R10*p_g_R10_1/S_base
        struct[0].g[19,0] = -Q_R10/S_base + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10**2*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18)) - S_n_R10*q_g_R10_1/S_base
        struct[0].g[20,0] = -P_R11/S_base + V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + V_R11**2*g_R03_R11
        struct[0].g[21,0] = -Q_R11/S_base + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + V_R11**2*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].g[22,0] = -P_R12/S_base + V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12**2*(g_R04_R12 + g_R12_R13) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].g[23,0] = -Q_R12/S_base + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12**2*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].g[24,0] = -P_R13/S_base + V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13**2*(g_R12_R13 + g_R13_R14) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].g[25,0] = -Q_R13/S_base + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13**2*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].g[26,0] = -P_R14/S_base + V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14**2*(g_R13_R14 + g_R14_R15) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) - S_n_R14*p_g_R14_1/S_base
        struct[0].g[27,0] = -Q_R14/S_base + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14**2*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15)) - S_n_R14*q_g_R14_1/S_base
        struct[0].g[28,0] = -P_R15/S_base + V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + V_R15**2*g_R14_R15
        struct[0].g[29,0] = -Q_R15/S_base + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + V_R15**2*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].g[30,0] = -P_R16/S_base + V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + V_R16**2*g_R06_R16
        struct[0].g[31,0] = -Q_R16/S_base + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + V_R16**2*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].g[32,0] = -P_R17/S_base + V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + V_R17**2*g_R09_R17
        struct[0].g[33,0] = -Q_R17/S_base + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + V_R17**2*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].g[34,0] = -P_R18/S_base + V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + V_R18**2*g_R10_R18
        struct[0].g[35,0] = -Q_R18/S_base + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + V_R18**2*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].g[36,0] = R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10) + X1d_R10*i_d_R10 - e1q_R10
        struct[0].g[37,0] = R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10) - X1q_R10*i_q_R10 - e1d_R10
        struct[0].g[38,0] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10) - p_g_R10_1
        struct[0].g[39,0] = V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10) - q_g_R10_1
        struct[0].g[40,0] = K_a_R10*(-v_c_R10 + v_pss_R10 + v_ref_R10) + K_ai_R10*xi_v_R10 - v_f_R10
        struct[0].g[41,0] = p_c_R10 - p_m_ref_R10 + p_r_R10 - (omega_R10 - 1)/Droop_R10
        struct[0].g[42,0] = R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14) + X1d_R14*i_d_R14 - e1q_R14
        struct[0].g[43,0] = R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14) - X1q_R14*i_q_R14 - e1d_R14
        struct[0].g[44,0] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14) - p_g_R14_1
        struct[0].g[45,0] = V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14) - q_g_R14_1
        struct[0].g[46,0] = K_a_R14*(-v_c_R14 + v_pss_R14 + v_ref_R14) + K_ai_R14*xi_v_R14 - v_f_R14
        struct[0].g[47,0] = p_c_R14 - p_m_ref_R14 + p_r_R14 - (omega_R14 - 1)/Droop_R14
        struct[0].g[48,0] = omega_R10/2 + omega_R14/2 - omega_coi
        struct[0].g[49,0] = K_sec_R10*xi_freq/2 - p_r_R10
        struct[0].g[50,0] = K_sec_R14*xi_freq/2 - p_r_R14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_R01
        struct[0].h[1,0] = V_R02
        struct[0].h[2,0] = V_R03
        struct[0].h[3,0] = V_R04
        struct[0].h[4,0] = V_R05
        struct[0].h[5,0] = V_R06
        struct[0].h[6,0] = V_R07
        struct[0].h[7,0] = V_R08
        struct[0].h[8,0] = V_R09
        struct[0].h[9,0] = V_R10
        struct[0].h[10,0] = V_R11
        struct[0].h[11,0] = V_R12
        struct[0].h[12,0] = V_R13
        struct[0].h[13,0] = V_R14
        struct[0].h[14,0] = V_R15
        struct[0].h[15,0] = V_R16
        struct[0].h[16,0] = V_R17
        struct[0].h[17,0] = V_R18
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_R10
        struct[0].Fx_ini[0,1] = Omega_b_R10
        struct[0].Fx_ini[1,0] = (-V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fx_ini[1,1] = -D_R10/(2*H_R10)
        struct[0].Fx_ini[1,6] = 1/(2*H_R10)
        struct[0].Fx_ini[2,2] = -1/T1d0_R10
        struct[0].Fx_ini[3,3] = -1/T1q0_R10
        struct[0].Fx_ini[4,4] = -1/T_r_R10
        struct[0].Fx_ini[6,6] = -1/T_m_R10
        struct[0].Fx_ini[7,7] = -K_delta_R14
        struct[0].Fx_ini[7,8] = Omega_b_R14
        struct[0].Fx_ini[8,7] = (-V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fx_ini[8,8] = -D_R14/(2*H_R14)
        struct[0].Fx_ini[8,13] = 1/(2*H_R14)
        struct[0].Fx_ini[9,9] = -1/T1d0_R14
        struct[0].Fx_ini[10,10] = -1/T1q0_R14
        struct[0].Fx_ini[11,11] = -1/T_r_R14
        struct[0].Fx_ini[13,13] = -1/T_m_R14

    if mode == 11:

        struct[0].Fy_ini[0,48] = -Omega_b_R10 
        struct[0].Fy_ini[1,18] = (-i_d_R10*sin(delta_R10 - theta_R10) - i_q_R10*cos(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,19] = (V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,36] = (-2*R_a_R10*i_d_R10 - V_R10*sin(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,37] = (-2*R_a_R10*i_q_R10 - V_R10*cos(delta_R10 - theta_R10))/(2*H_R10) 
        struct[0].Fy_ini[1,48] = D_R10/(2*H_R10) 
        struct[0].Fy_ini[2,36] = (X1d_R10 - X_d_R10)/T1d0_R10 
        struct[0].Fy_ini[2,40] = 1/T1d0_R10 
        struct[0].Fy_ini[3,37] = (-X1q_R10 + X_q_R10)/T1q0_R10 
        struct[0].Fy_ini[4,18] = 1/T_r_R10 
        struct[0].Fy_ini[5,18] = -1 
        struct[0].Fy_ini[6,41] = 1/T_m_R10 
        struct[0].Fy_ini[7,48] = -Omega_b_R14 
        struct[0].Fy_ini[8,26] = (-i_d_R14*sin(delta_R14 - theta_R14) - i_q_R14*cos(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,27] = (V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,42] = (-2*R_a_R14*i_d_R14 - V_R14*sin(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,43] = (-2*R_a_R14*i_q_R14 - V_R14*cos(delta_R14 - theta_R14))/(2*H_R14) 
        struct[0].Fy_ini[8,48] = D_R14/(2*H_R14) 
        struct[0].Fy_ini[9,42] = (X1d_R14 - X_d_R14)/T1d0_R14 
        struct[0].Fy_ini[9,46] = 1/T1d0_R14 
        struct[0].Fy_ini[10,43] = (-X1q_R14 + X_q_R14)/T1q0_R14 
        struct[0].Fy_ini[11,26] = 1/T_r_R14 
        struct[0].Fy_ini[12,26] = -1 
        struct[0].Fy_ini[13,47] = 1/T_m_R14 
        struct[0].Fy_ini[14,48] = -1 

        struct[0].Gy_ini[0,0] = 2*V_R01*g_R01_R02 + V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[0,1] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[0,2] = V_R01*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[0,3] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,0] = 2*V_R01*(-b_R01_R02 - bs_R01_R02/2) + V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[1,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[2,0] = V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[2,1] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[2,2] = V_R01*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + 2*V_R02*(g_R01_R02 + g_R02_R03) + V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[2,3] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[2,4] = V_R02*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[2,5] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,0] = V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy_ini[3,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy_ini[3,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + 2*V_R02*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[3,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[4,2] = V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[4,3] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[4,4] = V_R02*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + 2*V_R03*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[4,5] = V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[4,6] = V_R03*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[4,7] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[4,20] = V_R03*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[4,21] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,2] = V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy_ini[5,3] = V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy_ini[5,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + 2*V_R03*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[5,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[5,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[5,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[6,4] = V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[6,5] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[6,6] = V_R03*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + 2*V_R04*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[6,7] = V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[6,8] = V_R04*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[6,9] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[6,22] = V_R04*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[6,23] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,4] = V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy_ini[7,5] = V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy_ini[7,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + 2*V_R04*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[7,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[7,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[7,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[8,6] = V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[8,7] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[8,8] = V_R04*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + 2*V_R05*(g_R04_R05 + g_R05_R06) + V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[8,9] = V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[8,10] = V_R05*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[8,11] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,6] = V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy_ini[9,7] = V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy_ini[9,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + 2*V_R05*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[9,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[10,8] = V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[10,9] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[10,10] = V_R05*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + 2*V_R06*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[10,11] = V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[10,12] = V_R06*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[10,13] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[10,30] = V_R06*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[10,31] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,8] = V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy_ini[11,9] = V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy_ini[11,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + 2*V_R06*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[11,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[11,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[11,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[12,10] = V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[12,11] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[12,12] = V_R06*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + 2*V_R07*(g_R06_R07 + g_R07_R08) + V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[12,13] = V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[12,14] = V_R07*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[12,15] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,10] = V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy_ini[13,11] = V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy_ini[13,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + 2*V_R07*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[13,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[14,12] = V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[14,13] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[14,14] = V_R07*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + 2*V_R08*(g_R07_R08 + g_R08_R09) + V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[14,15] = V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[14,16] = V_R08*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[14,17] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,12] = V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy_ini[15,13] = V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy_ini[15,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + 2*V_R08*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[15,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[16,14] = V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[16,15] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[16,16] = V_R08*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + 2*V_R09*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[16,17] = V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[16,18] = V_R09*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[16,19] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[16,32] = V_R09*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[16,33] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,14] = V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy_ini[17,15] = V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy_ini[17,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + 2*V_R09*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[17,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[17,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[17,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[18,16] = V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[18,17] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[18,18] = V_R09*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + 2*V_R10*(g_R09_R10 + g_R10_R18) + V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,19] = V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,34] = V_R10*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,35] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[18,38] = -S_n_R10/S_base
        struct[0].Gy_ini[19,16] = V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy_ini[19,17] = V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy_ini[19,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + 2*V_R10*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[19,39] = -S_n_R10/S_base
        struct[0].Gy_ini[20,4] = V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[20,5] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[20,20] = V_R03*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + 2*V_R11*g_R03_R11
        struct[0].Gy_ini[20,21] = V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[21,4] = V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy_ini[21,5] = V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[21,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + 2*V_R11*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].Gy_ini[21,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy_ini[22,6] = V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[22,7] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[22,22] = V_R04*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + 2*V_R12*(g_R04_R12 + g_R12_R13) + V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[22,23] = V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[22,24] = V_R12*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[22,25] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,6] = V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy_ini[23,7] = V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy_ini[23,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + 2*V_R12*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[23,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[24,22] = V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[24,23] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[24,24] = V_R12*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + 2*V_R13*(g_R12_R13 + g_R13_R14) + V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[24,25] = V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[24,26] = V_R13*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[24,27] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,22] = V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy_ini[25,23] = V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy_ini[25,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + 2*V_R13*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[25,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[26,24] = V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[26,25] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[26,26] = V_R13*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + 2*V_R14*(g_R13_R14 + g_R14_R15) + V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,27] = V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,28] = V_R14*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,29] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[26,44] = -S_n_R14/S_base
        struct[0].Gy_ini[27,24] = V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy_ini[27,25] = V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy_ini[27,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + 2*V_R14*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[27,45] = -S_n_R14/S_base
        struct[0].Gy_ini[28,26] = V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[28,27] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[28,28] = V_R14*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + 2*V_R15*g_R14_R15
        struct[0].Gy_ini[28,29] = V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[29,26] = V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy_ini[29,27] = V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[29,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + 2*V_R15*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].Gy_ini[29,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy_ini[30,10] = V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[30,11] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[30,30] = V_R06*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + 2*V_R16*g_R06_R16
        struct[0].Gy_ini[30,31] = V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[31,10] = V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy_ini[31,11] = V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[31,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + 2*V_R16*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].Gy_ini[31,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy_ini[32,16] = V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[32,17] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[32,32] = V_R09*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + 2*V_R17*g_R09_R17
        struct[0].Gy_ini[32,33] = V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[33,16] = V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy_ini[33,17] = V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[33,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + 2*V_R17*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].Gy_ini[33,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy_ini[34,18] = V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[34,19] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[34,34] = V_R10*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + 2*V_R18*g_R10_R18
        struct[0].Gy_ini[34,35] = V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[35,18] = V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy_ini[35,19] = V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[35,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + 2*V_R18*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].Gy_ini[35,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy_ini[36,18] = cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[36,19] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[36,36] = X1d_R10
        struct[0].Gy_ini[36,37] = R_a_R10
        struct[0].Gy_ini[37,18] = sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[37,19] = -V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[37,36] = R_a_R10
        struct[0].Gy_ini[37,37] = -X1q_R10
        struct[0].Gy_ini[38,18] = i_d_R10*sin(delta_R10 - theta_R10) + i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[38,19] = -V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[38,36] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[38,37] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[38,38] = -1
        struct[0].Gy_ini[39,18] = i_d_R10*cos(delta_R10 - theta_R10) - i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,19] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,36] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,37] = -V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy_ini[39,39] = -1
        struct[0].Gy_ini[40,40] = -1
        struct[0].Gy_ini[41,41] = -1
        struct[0].Gy_ini[41,49] = 1
        struct[0].Gy_ini[42,26] = cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[42,27] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[42,42] = X1d_R14
        struct[0].Gy_ini[42,43] = R_a_R14
        struct[0].Gy_ini[43,26] = sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[43,27] = -V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[43,42] = R_a_R14
        struct[0].Gy_ini[43,43] = -X1q_R14
        struct[0].Gy_ini[44,26] = i_d_R14*sin(delta_R14 - theta_R14) + i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[44,27] = -V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[44,42] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[44,43] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[44,44] = -1
        struct[0].Gy_ini[45,26] = i_d_R14*cos(delta_R14 - theta_R14) - i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,27] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,42] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,43] = -V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy_ini[45,45] = -1
        struct[0].Gy_ini[46,46] = -1
        struct[0].Gy_ini[47,47] = -1
        struct[0].Gy_ini[47,50] = 1
        struct[0].Gy_ini[48,48] = -1
        struct[0].Gy_ini[49,49] = -1
        struct[0].Gy_ini[50,50] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_R01_R02 = struct[0].g_R01_R02
    b_R01_R02 = struct[0].b_R01_R02
    bs_R01_R02 = struct[0].bs_R01_R02
    g_R02_R03 = struct[0].g_R02_R03
    b_R02_R03 = struct[0].b_R02_R03
    bs_R02_R03 = struct[0].bs_R02_R03
    g_R03_R04 = struct[0].g_R03_R04
    b_R03_R04 = struct[0].b_R03_R04
    bs_R03_R04 = struct[0].bs_R03_R04
    g_R04_R05 = struct[0].g_R04_R05
    b_R04_R05 = struct[0].b_R04_R05
    bs_R04_R05 = struct[0].bs_R04_R05
    g_R05_R06 = struct[0].g_R05_R06
    b_R05_R06 = struct[0].b_R05_R06
    bs_R05_R06 = struct[0].bs_R05_R06
    g_R06_R07 = struct[0].g_R06_R07
    b_R06_R07 = struct[0].b_R06_R07
    bs_R06_R07 = struct[0].bs_R06_R07
    g_R07_R08 = struct[0].g_R07_R08
    b_R07_R08 = struct[0].b_R07_R08
    bs_R07_R08 = struct[0].bs_R07_R08
    g_R08_R09 = struct[0].g_R08_R09
    b_R08_R09 = struct[0].b_R08_R09
    bs_R08_R09 = struct[0].bs_R08_R09
    g_R09_R10 = struct[0].g_R09_R10
    b_R09_R10 = struct[0].b_R09_R10
    bs_R09_R10 = struct[0].bs_R09_R10
    g_R03_R11 = struct[0].g_R03_R11
    b_R03_R11 = struct[0].b_R03_R11
    bs_R03_R11 = struct[0].bs_R03_R11
    g_R04_R12 = struct[0].g_R04_R12
    b_R04_R12 = struct[0].b_R04_R12
    bs_R04_R12 = struct[0].bs_R04_R12
    g_R12_R13 = struct[0].g_R12_R13
    b_R12_R13 = struct[0].b_R12_R13
    bs_R12_R13 = struct[0].bs_R12_R13
    g_R13_R14 = struct[0].g_R13_R14
    b_R13_R14 = struct[0].b_R13_R14
    bs_R13_R14 = struct[0].bs_R13_R14
    g_R14_R15 = struct[0].g_R14_R15
    b_R14_R15 = struct[0].b_R14_R15
    bs_R14_R15 = struct[0].bs_R14_R15
    g_R06_R16 = struct[0].g_R06_R16
    b_R06_R16 = struct[0].b_R06_R16
    bs_R06_R16 = struct[0].bs_R06_R16
    g_R09_R17 = struct[0].g_R09_R17
    b_R09_R17 = struct[0].b_R09_R17
    bs_R09_R17 = struct[0].bs_R09_R17
    g_R10_R18 = struct[0].g_R10_R18
    b_R10_R18 = struct[0].b_R10_R18
    bs_R10_R18 = struct[0].bs_R10_R18
    U_R01_n = struct[0].U_R01_n
    U_R02_n = struct[0].U_R02_n
    U_R03_n = struct[0].U_R03_n
    U_R04_n = struct[0].U_R04_n
    U_R05_n = struct[0].U_R05_n
    U_R06_n = struct[0].U_R06_n
    U_R07_n = struct[0].U_R07_n
    U_R08_n = struct[0].U_R08_n
    U_R09_n = struct[0].U_R09_n
    U_R10_n = struct[0].U_R10_n
    U_R11_n = struct[0].U_R11_n
    U_R12_n = struct[0].U_R12_n
    U_R13_n = struct[0].U_R13_n
    U_R14_n = struct[0].U_R14_n
    U_R15_n = struct[0].U_R15_n
    U_R16_n = struct[0].U_R16_n
    U_R17_n = struct[0].U_R17_n
    U_R18_n = struct[0].U_R18_n
    S_n_R10 = struct[0].S_n_R10
    H_R10 = struct[0].H_R10
    Omega_b_R10 = struct[0].Omega_b_R10
    T1d0_R10 = struct[0].T1d0_R10
    T1q0_R10 = struct[0].T1q0_R10
    X_d_R10 = struct[0].X_d_R10
    X_q_R10 = struct[0].X_q_R10
    X1d_R10 = struct[0].X1d_R10
    X1q_R10 = struct[0].X1q_R10
    D_R10 = struct[0].D_R10
    R_a_R10 = struct[0].R_a_R10
    K_delta_R10 = struct[0].K_delta_R10
    K_a_R10 = struct[0].K_a_R10
    K_ai_R10 = struct[0].K_ai_R10
    T_r_R10 = struct[0].T_r_R10
    Droop_R10 = struct[0].Droop_R10
    T_m_R10 = struct[0].T_m_R10
    S_n_R14 = struct[0].S_n_R14
    H_R14 = struct[0].H_R14
    Omega_b_R14 = struct[0].Omega_b_R14
    T1d0_R14 = struct[0].T1d0_R14
    T1q0_R14 = struct[0].T1q0_R14
    X_d_R14 = struct[0].X_d_R14
    X_q_R14 = struct[0].X_q_R14
    X1d_R14 = struct[0].X1d_R14
    X1q_R14 = struct[0].X1q_R14
    D_R14 = struct[0].D_R14
    R_a_R14 = struct[0].R_a_R14
    K_delta_R14 = struct[0].K_delta_R14
    K_a_R14 = struct[0].K_a_R14
    K_ai_R14 = struct[0].K_ai_R14
    T_r_R14 = struct[0].T_r_R14
    Droop_R14 = struct[0].Droop_R14
    T_m_R14 = struct[0].T_m_R14
    K_sec_R10 = struct[0].K_sec_R10
    K_sec_R14 = struct[0].K_sec_R14
    
    # Inputs:
    P_R01 = struct[0].P_R01
    Q_R01 = struct[0].Q_R01
    P_R02 = struct[0].P_R02
    Q_R02 = struct[0].Q_R02
    P_R03 = struct[0].P_R03
    Q_R03 = struct[0].Q_R03
    P_R04 = struct[0].P_R04
    Q_R04 = struct[0].Q_R04
    P_R05 = struct[0].P_R05
    Q_R05 = struct[0].Q_R05
    P_R06 = struct[0].P_R06
    Q_R06 = struct[0].Q_R06
    P_R07 = struct[0].P_R07
    Q_R07 = struct[0].Q_R07
    P_R08 = struct[0].P_R08
    Q_R08 = struct[0].Q_R08
    P_R09 = struct[0].P_R09
    Q_R09 = struct[0].Q_R09
    P_R10 = struct[0].P_R10
    Q_R10 = struct[0].Q_R10
    P_R11 = struct[0].P_R11
    Q_R11 = struct[0].Q_R11
    P_R12 = struct[0].P_R12
    Q_R12 = struct[0].Q_R12
    P_R13 = struct[0].P_R13
    Q_R13 = struct[0].Q_R13
    P_R14 = struct[0].P_R14
    Q_R14 = struct[0].Q_R14
    P_R15 = struct[0].P_R15
    Q_R15 = struct[0].Q_R15
    P_R16 = struct[0].P_R16
    Q_R16 = struct[0].Q_R16
    P_R17 = struct[0].P_R17
    Q_R17 = struct[0].Q_R17
    P_R18 = struct[0].P_R18
    Q_R18 = struct[0].Q_R18
    v_ref_R10 = struct[0].v_ref_R10
    v_pss_R10 = struct[0].v_pss_R10
    p_c_R10 = struct[0].p_c_R10
    v_ref_R14 = struct[0].v_ref_R14
    v_pss_R14 = struct[0].v_pss_R14
    p_c_R14 = struct[0].p_c_R14
    
    # Dynamical states:
    delta_R10 = struct[0].x[0,0]
    omega_R10 = struct[0].x[1,0]
    e1q_R10 = struct[0].x[2,0]
    e1d_R10 = struct[0].x[3,0]
    v_c_R10 = struct[0].x[4,0]
    xi_v_R10 = struct[0].x[5,0]
    p_m_R10 = struct[0].x[6,0]
    delta_R14 = struct[0].x[7,0]
    omega_R14 = struct[0].x[8,0]
    e1q_R14 = struct[0].x[9,0]
    e1d_R14 = struct[0].x[10,0]
    v_c_R14 = struct[0].x[11,0]
    xi_v_R14 = struct[0].x[12,0]
    p_m_R14 = struct[0].x[13,0]
    xi_freq = struct[0].x[14,0]
    
    # Algebraic states:
    V_R01 = struct[0].y_run[0,0]
    theta_R01 = struct[0].y_run[1,0]
    V_R02 = struct[0].y_run[2,0]
    theta_R02 = struct[0].y_run[3,0]
    V_R03 = struct[0].y_run[4,0]
    theta_R03 = struct[0].y_run[5,0]
    V_R04 = struct[0].y_run[6,0]
    theta_R04 = struct[0].y_run[7,0]
    V_R05 = struct[0].y_run[8,0]
    theta_R05 = struct[0].y_run[9,0]
    V_R06 = struct[0].y_run[10,0]
    theta_R06 = struct[0].y_run[11,0]
    V_R07 = struct[0].y_run[12,0]
    theta_R07 = struct[0].y_run[13,0]
    V_R08 = struct[0].y_run[14,0]
    theta_R08 = struct[0].y_run[15,0]
    V_R09 = struct[0].y_run[16,0]
    theta_R09 = struct[0].y_run[17,0]
    V_R10 = struct[0].y_run[18,0]
    theta_R10 = struct[0].y_run[19,0]
    V_R11 = struct[0].y_run[20,0]
    theta_R11 = struct[0].y_run[21,0]
    V_R12 = struct[0].y_run[22,0]
    theta_R12 = struct[0].y_run[23,0]
    V_R13 = struct[0].y_run[24,0]
    theta_R13 = struct[0].y_run[25,0]
    V_R14 = struct[0].y_run[26,0]
    theta_R14 = struct[0].y_run[27,0]
    V_R15 = struct[0].y_run[28,0]
    theta_R15 = struct[0].y_run[29,0]
    V_R16 = struct[0].y_run[30,0]
    theta_R16 = struct[0].y_run[31,0]
    V_R17 = struct[0].y_run[32,0]
    theta_R17 = struct[0].y_run[33,0]
    V_R18 = struct[0].y_run[34,0]
    theta_R18 = struct[0].y_run[35,0]
    i_d_R10 = struct[0].y_run[36,0]
    i_q_R10 = struct[0].y_run[37,0]
    p_g_R10_1 = struct[0].y_run[38,0]
    q_g_R10_1 = struct[0].y_run[39,0]
    v_f_R10 = struct[0].y_run[40,0]
    p_m_ref_R10 = struct[0].y_run[41,0]
    i_d_R14 = struct[0].y_run[42,0]
    i_q_R14 = struct[0].y_run[43,0]
    p_g_R14_1 = struct[0].y_run[44,0]
    q_g_R14_1 = struct[0].y_run[45,0]
    v_f_R14 = struct[0].y_run[46,0]
    p_m_ref_R14 = struct[0].y_run[47,0]
    omega_coi = struct[0].y_run[48,0]
    p_r_R10 = struct[0].y_run[49,0]
    p_r_R14 = struct[0].y_run[50,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_R10*delta_R10 + Omega_b_R10*(omega_R10 - omega_coi)
        struct[0].f[1,0] = (-D_R10*(omega_R10 - omega_coi) - i_d_R10*(R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10)) - i_q_R10*(R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10)) + p_m_R10)/(2*H_R10)
        struct[0].f[2,0] = (-e1q_R10 - i_d_R10*(-X1d_R10 + X_d_R10) + v_f_R10)/T1d0_R10
        struct[0].f[3,0] = (-e1d_R10 + i_q_R10*(-X1q_R10 + X_q_R10))/T1q0_R10
        struct[0].f[4,0] = (V_R10 - v_c_R10)/T_r_R10
        struct[0].f[5,0] = -V_R10 + v_ref_R10
        struct[0].f[6,0] = (-p_m_R10 + p_m_ref_R10)/T_m_R10
        struct[0].f[7,0] = -K_delta_R14*delta_R14 + Omega_b_R14*(omega_R14 - omega_coi)
        struct[0].f[8,0] = (-D_R14*(omega_R14 - omega_coi) - i_d_R14*(R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14)) - i_q_R14*(R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14)) + p_m_R14)/(2*H_R14)
        struct[0].f[9,0] = (-e1q_R14 - i_d_R14*(-X1d_R14 + X_d_R14) + v_f_R14)/T1d0_R14
        struct[0].f[10,0] = (-e1d_R14 + i_q_R14*(-X1q_R14 + X_q_R14))/T1q0_R14
        struct[0].f[11,0] = (V_R14 - v_c_R14)/T_r_R14
        struct[0].f[12,0] = -V_R14 + v_ref_R14
        struct[0].f[13,0] = (-p_m_R14 + p_m_ref_R14)/T_m_R14
        struct[0].f[14,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_R01/S_base + V_R01**2*g_R01_R02 + V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].g[1,0] = -Q_R01/S_base + V_R01**2*(-b_R01_R02 - bs_R01_R02/2) + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].g[2,0] = -P_R02/S_base + V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02**2*(g_R01_R02 + g_R02_R03) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].g[3,0] = -Q_R02/S_base + V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02**2*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].g[4,0] = -P_R03/S_base + V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03**2*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].g[5,0] = -Q_R03/S_base + V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03**2*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].g[6,0] = -P_R04/S_base + V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04**2*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].g[7,0] = -Q_R04/S_base + V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04**2*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].g[8,0] = -P_R05/S_base + V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05**2*(g_R04_R05 + g_R05_R06) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].g[9,0] = -Q_R05/S_base + V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05**2*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].g[10,0] = -P_R06/S_base + V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06**2*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].g[11,0] = -Q_R06/S_base + V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06**2*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].g[12,0] = -P_R07/S_base + V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07**2*(g_R06_R07 + g_R07_R08) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].g[13,0] = -Q_R07/S_base + V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07**2*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].g[14,0] = -P_R08/S_base + V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08**2*(g_R07_R08 + g_R08_R09) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].g[15,0] = -Q_R08/S_base + V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08**2*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].g[16,0] = -P_R09/S_base + V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09**2*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].g[17,0] = -Q_R09/S_base + V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09**2*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].g[18,0] = -P_R10/S_base + V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10**2*(g_R09_R10 + g_R10_R18) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) - S_n_R10*p_g_R10_1/S_base
        struct[0].g[19,0] = -Q_R10/S_base + V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10**2*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18)) - S_n_R10*q_g_R10_1/S_base
        struct[0].g[20,0] = -P_R11/S_base + V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + V_R11**2*g_R03_R11
        struct[0].g[21,0] = -Q_R11/S_base + V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + V_R11**2*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].g[22,0] = -P_R12/S_base + V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12**2*(g_R04_R12 + g_R12_R13) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].g[23,0] = -Q_R12/S_base + V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12**2*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].g[24,0] = -P_R13/S_base + V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13**2*(g_R12_R13 + g_R13_R14) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].g[25,0] = -Q_R13/S_base + V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13**2*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].g[26,0] = -P_R14/S_base + V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14**2*(g_R13_R14 + g_R14_R15) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) - S_n_R14*p_g_R14_1/S_base
        struct[0].g[27,0] = -Q_R14/S_base + V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14**2*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15)) - S_n_R14*q_g_R14_1/S_base
        struct[0].g[28,0] = -P_R15/S_base + V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + V_R15**2*g_R14_R15
        struct[0].g[29,0] = -Q_R15/S_base + V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + V_R15**2*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].g[30,0] = -P_R16/S_base + V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + V_R16**2*g_R06_R16
        struct[0].g[31,0] = -Q_R16/S_base + V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + V_R16**2*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].g[32,0] = -P_R17/S_base + V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + V_R17**2*g_R09_R17
        struct[0].g[33,0] = -Q_R17/S_base + V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + V_R17**2*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].g[34,0] = -P_R18/S_base + V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + V_R18**2*g_R10_R18
        struct[0].g[35,0] = -Q_R18/S_base + V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + V_R18**2*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].g[36,0] = R_a_R10*i_q_R10 + V_R10*cos(delta_R10 - theta_R10) + X1d_R10*i_d_R10 - e1q_R10
        struct[0].g[37,0] = R_a_R10*i_d_R10 + V_R10*sin(delta_R10 - theta_R10) - X1q_R10*i_q_R10 - e1d_R10
        struct[0].g[38,0] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10) - p_g_R10_1
        struct[0].g[39,0] = V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10) - q_g_R10_1
        struct[0].g[40,0] = K_a_R10*(-v_c_R10 + v_pss_R10 + v_ref_R10) + K_ai_R10*xi_v_R10 - v_f_R10
        struct[0].g[41,0] = p_c_R10 - p_m_ref_R10 + p_r_R10 - (omega_R10 - 1)/Droop_R10
        struct[0].g[42,0] = R_a_R14*i_q_R14 + V_R14*cos(delta_R14 - theta_R14) + X1d_R14*i_d_R14 - e1q_R14
        struct[0].g[43,0] = R_a_R14*i_d_R14 + V_R14*sin(delta_R14 - theta_R14) - X1q_R14*i_q_R14 - e1d_R14
        struct[0].g[44,0] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14) - p_g_R14_1
        struct[0].g[45,0] = V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14) - q_g_R14_1
        struct[0].g[46,0] = K_a_R14*(-v_c_R14 + v_pss_R14 + v_ref_R14) + K_ai_R14*xi_v_R14 - v_f_R14
        struct[0].g[47,0] = p_c_R14 - p_m_ref_R14 + p_r_R14 - (omega_R14 - 1)/Droop_R14
        struct[0].g[48,0] = omega_R10/2 + omega_R14/2 - omega_coi
        struct[0].g[49,0] = K_sec_R10*xi_freq/2 - p_r_R10
        struct[0].g[50,0] = K_sec_R14*xi_freq/2 - p_r_R14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_R01
        struct[0].h[1,0] = V_R02
        struct[0].h[2,0] = V_R03
        struct[0].h[3,0] = V_R04
        struct[0].h[4,0] = V_R05
        struct[0].h[5,0] = V_R06
        struct[0].h[6,0] = V_R07
        struct[0].h[7,0] = V_R08
        struct[0].h[8,0] = V_R09
        struct[0].h[9,0] = V_R10
        struct[0].h[10,0] = V_R11
        struct[0].h[11,0] = V_R12
        struct[0].h[12,0] = V_R13
        struct[0].h[13,0] = V_R14
        struct[0].h[14,0] = V_R15
        struct[0].h[15,0] = V_R16
        struct[0].h[16,0] = V_R17
        struct[0].h[17,0] = V_R18
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_R10
        struct[0].Fx[0,1] = Omega_b_R10
        struct[0].Fx[1,0] = (-V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fx[1,1] = -D_R10/(2*H_R10)
        struct[0].Fx[1,6] = 1/(2*H_R10)
        struct[0].Fx[2,2] = -1/T1d0_R10
        struct[0].Fx[3,3] = -1/T1q0_R10
        struct[0].Fx[4,4] = -1/T_r_R10
        struct[0].Fx[6,6] = -1/T_m_R10
        struct[0].Fx[7,7] = -K_delta_R14
        struct[0].Fx[7,8] = Omega_b_R14
        struct[0].Fx[8,7] = (-V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fx[8,8] = -D_R14/(2*H_R14)
        struct[0].Fx[8,13] = 1/(2*H_R14)
        struct[0].Fx[9,9] = -1/T1d0_R14
        struct[0].Fx[10,10] = -1/T1q0_R14
        struct[0].Fx[11,11] = -1/T_r_R14
        struct[0].Fx[13,13] = -1/T_m_R14

    if mode == 11:

        struct[0].Fy[0,48] = -Omega_b_R10
        struct[0].Fy[1,18] = (-i_d_R10*sin(delta_R10 - theta_R10) - i_q_R10*cos(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,19] = (V_R10*i_d_R10*cos(delta_R10 - theta_R10) - V_R10*i_q_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,36] = (-2*R_a_R10*i_d_R10 - V_R10*sin(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,37] = (-2*R_a_R10*i_q_R10 - V_R10*cos(delta_R10 - theta_R10))/(2*H_R10)
        struct[0].Fy[1,48] = D_R10/(2*H_R10)
        struct[0].Fy[2,36] = (X1d_R10 - X_d_R10)/T1d0_R10
        struct[0].Fy[2,40] = 1/T1d0_R10
        struct[0].Fy[3,37] = (-X1q_R10 + X_q_R10)/T1q0_R10
        struct[0].Fy[4,18] = 1/T_r_R10
        struct[0].Fy[5,18] = -1
        struct[0].Fy[6,41] = 1/T_m_R10
        struct[0].Fy[7,48] = -Omega_b_R14
        struct[0].Fy[8,26] = (-i_d_R14*sin(delta_R14 - theta_R14) - i_q_R14*cos(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,27] = (V_R14*i_d_R14*cos(delta_R14 - theta_R14) - V_R14*i_q_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,42] = (-2*R_a_R14*i_d_R14 - V_R14*sin(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,43] = (-2*R_a_R14*i_q_R14 - V_R14*cos(delta_R14 - theta_R14))/(2*H_R14)
        struct[0].Fy[8,48] = D_R14/(2*H_R14)
        struct[0].Fy[9,42] = (X1d_R14 - X_d_R14)/T1d0_R14
        struct[0].Fy[9,46] = 1/T1d0_R14
        struct[0].Fy[10,43] = (-X1q_R14 + X_q_R14)/T1q0_R14
        struct[0].Fy[11,26] = 1/T_r_R14
        struct[0].Fy[12,26] = -1
        struct[0].Fy[13,47] = 1/T_m_R14
        struct[0].Fy[14,48] = -1

        struct[0].Gy[0,0] = 2*V_R01*g_R01_R02 + V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[0,1] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[0,2] = V_R01*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[0,3] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[1,0] = 2*V_R01*(-b_R01_R02 - bs_R01_R02/2) + V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[1,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[1,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[1,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[2,0] = V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[2,1] = V_R01*V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[2,2] = V_R01*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + 2*V_R02*(g_R01_R02 + g_R02_R03) + V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[2,3] = V_R01*V_R02*(-b_R01_R02*cos(theta_R01 - theta_R02) - g_R01_R02*sin(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[2,4] = V_R02*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[2,5] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[3,0] = V_R02*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02))
        struct[0].Gy[3,1] = V_R01*V_R02*(-b_R01_R02*sin(theta_R01 - theta_R02) + g_R01_R02*cos(theta_R01 - theta_R02))
        struct[0].Gy[3,2] = V_R01*(b_R01_R02*cos(theta_R01 - theta_R02) + g_R01_R02*sin(theta_R01 - theta_R02)) + 2*V_R02*(-b_R01_R02 - b_R02_R03 - bs_R01_R02/2 - bs_R02_R03/2) + V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[3,3] = V_R01*V_R02*(b_R01_R02*sin(theta_R01 - theta_R02) - g_R01_R02*cos(theta_R01 - theta_R02)) + V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[3,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[3,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[4,2] = V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[4,3] = V_R02*V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[4,4] = V_R02*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + 2*V_R03*(g_R02_R03 + g_R03_R04 + g_R03_R11) + V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[4,5] = V_R02*V_R03*(-b_R02_R03*cos(theta_R02 - theta_R03) - g_R02_R03*sin(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[4,6] = V_R03*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[4,7] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[4,20] = V_R03*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[4,21] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[5,2] = V_R03*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03))
        struct[0].Gy[5,3] = V_R02*V_R03*(-b_R02_R03*sin(theta_R02 - theta_R03) + g_R02_R03*cos(theta_R02 - theta_R03))
        struct[0].Gy[5,4] = V_R02*(b_R02_R03*cos(theta_R02 - theta_R03) + g_R02_R03*sin(theta_R02 - theta_R03)) + 2*V_R03*(-b_R02_R03 - b_R03_R04 - b_R03_R11 - bs_R02_R03/2 - bs_R03_R04/2 - bs_R03_R11/2) + V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[5,5] = V_R02*V_R03*(b_R02_R03*sin(theta_R02 - theta_R03) - g_R02_R03*cos(theta_R02 - theta_R03)) + V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[5,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[5,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[5,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[5,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[6,4] = V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[6,5] = V_R03*V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[6,6] = V_R03*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + 2*V_R04*(g_R03_R04 + g_R04_R05 + g_R04_R12) + V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[6,7] = V_R03*V_R04*(-b_R03_R04*cos(theta_R03 - theta_R04) - g_R03_R04*sin(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[6,8] = V_R04*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[6,9] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[6,22] = V_R04*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[6,23] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[7,4] = V_R04*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04))
        struct[0].Gy[7,5] = V_R03*V_R04*(-b_R03_R04*sin(theta_R03 - theta_R04) + g_R03_R04*cos(theta_R03 - theta_R04))
        struct[0].Gy[7,6] = V_R03*(b_R03_R04*cos(theta_R03 - theta_R04) + g_R03_R04*sin(theta_R03 - theta_R04)) + 2*V_R04*(-b_R03_R04 - b_R04_R05 - b_R04_R12 - bs_R03_R04/2 - bs_R04_R05/2 - bs_R04_R12/2) + V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[7,7] = V_R03*V_R04*(b_R03_R04*sin(theta_R03 - theta_R04) - g_R03_R04*cos(theta_R03 - theta_R04)) + V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[7,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[7,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[7,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[7,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[8,6] = V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[8,7] = V_R04*V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[8,8] = V_R04*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + 2*V_R05*(g_R04_R05 + g_R05_R06) + V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[8,9] = V_R04*V_R05*(-b_R04_R05*cos(theta_R04 - theta_R05) - g_R04_R05*sin(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[8,10] = V_R05*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[8,11] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[9,6] = V_R05*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05))
        struct[0].Gy[9,7] = V_R04*V_R05*(-b_R04_R05*sin(theta_R04 - theta_R05) + g_R04_R05*cos(theta_R04 - theta_R05))
        struct[0].Gy[9,8] = V_R04*(b_R04_R05*cos(theta_R04 - theta_R05) + g_R04_R05*sin(theta_R04 - theta_R05)) + 2*V_R05*(-b_R04_R05 - b_R05_R06 - bs_R04_R05/2 - bs_R05_R06/2) + V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[9,9] = V_R04*V_R05*(b_R04_R05*sin(theta_R04 - theta_R05) - g_R04_R05*cos(theta_R04 - theta_R05)) + V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[9,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[9,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[10,8] = V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[10,9] = V_R05*V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[10,10] = V_R05*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + 2*V_R06*(g_R05_R06 + g_R06_R07 + g_R06_R16) + V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[10,11] = V_R05*V_R06*(-b_R05_R06*cos(theta_R05 - theta_R06) - g_R05_R06*sin(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[10,12] = V_R06*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[10,13] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[10,30] = V_R06*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[10,31] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[11,8] = V_R06*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06))
        struct[0].Gy[11,9] = V_R05*V_R06*(-b_R05_R06*sin(theta_R05 - theta_R06) + g_R05_R06*cos(theta_R05 - theta_R06))
        struct[0].Gy[11,10] = V_R05*(b_R05_R06*cos(theta_R05 - theta_R06) + g_R05_R06*sin(theta_R05 - theta_R06)) + 2*V_R06*(-b_R05_R06 - b_R06_R07 - b_R06_R16 - bs_R05_R06/2 - bs_R06_R07/2 - bs_R06_R16/2) + V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[11,11] = V_R05*V_R06*(b_R05_R06*sin(theta_R05 - theta_R06) - g_R05_R06*cos(theta_R05 - theta_R06)) + V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[11,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[11,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[11,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[11,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[12,10] = V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[12,11] = V_R06*V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[12,12] = V_R06*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + 2*V_R07*(g_R06_R07 + g_R07_R08) + V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[12,13] = V_R06*V_R07*(-b_R06_R07*cos(theta_R06 - theta_R07) - g_R06_R07*sin(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[12,14] = V_R07*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[12,15] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[13,10] = V_R07*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07))
        struct[0].Gy[13,11] = V_R06*V_R07*(-b_R06_R07*sin(theta_R06 - theta_R07) + g_R06_R07*cos(theta_R06 - theta_R07))
        struct[0].Gy[13,12] = V_R06*(b_R06_R07*cos(theta_R06 - theta_R07) + g_R06_R07*sin(theta_R06 - theta_R07)) + 2*V_R07*(-b_R06_R07 - b_R07_R08 - bs_R06_R07/2 - bs_R07_R08/2) + V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[13,13] = V_R06*V_R07*(b_R06_R07*sin(theta_R06 - theta_R07) - g_R06_R07*cos(theta_R06 - theta_R07)) + V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[13,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[13,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[14,12] = V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[14,13] = V_R07*V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[14,14] = V_R07*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + 2*V_R08*(g_R07_R08 + g_R08_R09) + V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[14,15] = V_R07*V_R08*(-b_R07_R08*cos(theta_R07 - theta_R08) - g_R07_R08*sin(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[14,16] = V_R08*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[14,17] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[15,12] = V_R08*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08))
        struct[0].Gy[15,13] = V_R07*V_R08*(-b_R07_R08*sin(theta_R07 - theta_R08) + g_R07_R08*cos(theta_R07 - theta_R08))
        struct[0].Gy[15,14] = V_R07*(b_R07_R08*cos(theta_R07 - theta_R08) + g_R07_R08*sin(theta_R07 - theta_R08)) + 2*V_R08*(-b_R07_R08 - b_R08_R09 - bs_R07_R08/2 - bs_R08_R09/2) + V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[15,15] = V_R07*V_R08*(b_R07_R08*sin(theta_R07 - theta_R08) - g_R07_R08*cos(theta_R07 - theta_R08)) + V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[15,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[15,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[16,14] = V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[16,15] = V_R08*V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[16,16] = V_R08*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + 2*V_R09*(g_R08_R09 + g_R09_R10 + g_R09_R17) + V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[16,17] = V_R08*V_R09*(-b_R08_R09*cos(theta_R08 - theta_R09) - g_R08_R09*sin(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[16,18] = V_R09*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[16,19] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[16,32] = V_R09*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[16,33] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[17,14] = V_R09*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09))
        struct[0].Gy[17,15] = V_R08*V_R09*(-b_R08_R09*sin(theta_R08 - theta_R09) + g_R08_R09*cos(theta_R08 - theta_R09))
        struct[0].Gy[17,16] = V_R08*(b_R08_R09*cos(theta_R08 - theta_R09) + g_R08_R09*sin(theta_R08 - theta_R09)) + 2*V_R09*(-b_R08_R09 - b_R09_R10 - b_R09_R17 - bs_R08_R09/2 - bs_R09_R10/2 - bs_R09_R17/2) + V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[17,17] = V_R08*V_R09*(b_R08_R09*sin(theta_R08 - theta_R09) - g_R08_R09*cos(theta_R08 - theta_R09)) + V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[17,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[17,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[17,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[17,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[18,16] = V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[18,17] = V_R09*V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[18,18] = V_R09*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + 2*V_R10*(g_R09_R10 + g_R10_R18) + V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[18,19] = V_R09*V_R10*(-b_R09_R10*cos(theta_R09 - theta_R10) - g_R09_R10*sin(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[18,34] = V_R10*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[18,35] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[18,38] = -S_n_R10/S_base
        struct[0].Gy[19,16] = V_R10*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10))
        struct[0].Gy[19,17] = V_R09*V_R10*(-b_R09_R10*sin(theta_R09 - theta_R10) + g_R09_R10*cos(theta_R09 - theta_R10))
        struct[0].Gy[19,18] = V_R09*(b_R09_R10*cos(theta_R09 - theta_R10) + g_R09_R10*sin(theta_R09 - theta_R10)) + 2*V_R10*(-b_R09_R10 - b_R10_R18 - bs_R09_R10/2 - bs_R10_R18/2) + V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[19,19] = V_R09*V_R10*(b_R09_R10*sin(theta_R09 - theta_R10) - g_R09_R10*cos(theta_R09 - theta_R10)) + V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[19,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[19,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[19,39] = -S_n_R10/S_base
        struct[0].Gy[20,4] = V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[20,5] = V_R03*V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[20,20] = V_R03*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11)) + 2*V_R11*g_R03_R11
        struct[0].Gy[20,21] = V_R03*V_R11*(-b_R03_R11*cos(theta_R03 - theta_R11) - g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[21,4] = V_R11*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11))
        struct[0].Gy[21,5] = V_R03*V_R11*(-b_R03_R11*sin(theta_R03 - theta_R11) + g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[21,20] = V_R03*(b_R03_R11*cos(theta_R03 - theta_R11) + g_R03_R11*sin(theta_R03 - theta_R11)) + 2*V_R11*(-b_R03_R11 - bs_R03_R11/2)
        struct[0].Gy[21,21] = V_R03*V_R11*(b_R03_R11*sin(theta_R03 - theta_R11) - g_R03_R11*cos(theta_R03 - theta_R11))
        struct[0].Gy[22,6] = V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[22,7] = V_R04*V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[22,22] = V_R04*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + 2*V_R12*(g_R04_R12 + g_R12_R13) + V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[22,23] = V_R04*V_R12*(-b_R04_R12*cos(theta_R04 - theta_R12) - g_R04_R12*sin(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[22,24] = V_R12*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[22,25] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[23,6] = V_R12*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12))
        struct[0].Gy[23,7] = V_R04*V_R12*(-b_R04_R12*sin(theta_R04 - theta_R12) + g_R04_R12*cos(theta_R04 - theta_R12))
        struct[0].Gy[23,22] = V_R04*(b_R04_R12*cos(theta_R04 - theta_R12) + g_R04_R12*sin(theta_R04 - theta_R12)) + 2*V_R12*(-b_R04_R12 - b_R12_R13 - bs_R04_R12/2 - bs_R12_R13/2) + V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[23,23] = V_R04*V_R12*(b_R04_R12*sin(theta_R04 - theta_R12) - g_R04_R12*cos(theta_R04 - theta_R12)) + V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[23,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[23,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[24,22] = V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[24,23] = V_R12*V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[24,24] = V_R12*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + 2*V_R13*(g_R12_R13 + g_R13_R14) + V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[24,25] = V_R12*V_R13*(-b_R12_R13*cos(theta_R12 - theta_R13) - g_R12_R13*sin(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[24,26] = V_R13*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[24,27] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[25,22] = V_R13*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13))
        struct[0].Gy[25,23] = V_R12*V_R13*(-b_R12_R13*sin(theta_R12 - theta_R13) + g_R12_R13*cos(theta_R12 - theta_R13))
        struct[0].Gy[25,24] = V_R12*(b_R12_R13*cos(theta_R12 - theta_R13) + g_R12_R13*sin(theta_R12 - theta_R13)) + 2*V_R13*(-b_R12_R13 - b_R13_R14 - bs_R12_R13/2 - bs_R13_R14/2) + V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[25,25] = V_R12*V_R13*(b_R12_R13*sin(theta_R12 - theta_R13) - g_R12_R13*cos(theta_R12 - theta_R13)) + V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[25,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[25,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[26,24] = V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[26,25] = V_R13*V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[26,26] = V_R13*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + 2*V_R14*(g_R13_R14 + g_R14_R15) + V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[26,27] = V_R13*V_R14*(-b_R13_R14*cos(theta_R13 - theta_R14) - g_R13_R14*sin(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[26,28] = V_R14*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[26,29] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[26,44] = -S_n_R14/S_base
        struct[0].Gy[27,24] = V_R14*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14))
        struct[0].Gy[27,25] = V_R13*V_R14*(-b_R13_R14*sin(theta_R13 - theta_R14) + g_R13_R14*cos(theta_R13 - theta_R14))
        struct[0].Gy[27,26] = V_R13*(b_R13_R14*cos(theta_R13 - theta_R14) + g_R13_R14*sin(theta_R13 - theta_R14)) + 2*V_R14*(-b_R13_R14 - b_R14_R15 - bs_R13_R14/2 - bs_R14_R15/2) + V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[27,27] = V_R13*V_R14*(b_R13_R14*sin(theta_R13 - theta_R14) - g_R13_R14*cos(theta_R13 - theta_R14)) + V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[27,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[27,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[27,45] = -S_n_R14/S_base
        struct[0].Gy[28,26] = V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[28,27] = V_R14*V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[28,28] = V_R14*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15)) + 2*V_R15*g_R14_R15
        struct[0].Gy[28,29] = V_R14*V_R15*(-b_R14_R15*cos(theta_R14 - theta_R15) - g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[29,26] = V_R15*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15))
        struct[0].Gy[29,27] = V_R14*V_R15*(-b_R14_R15*sin(theta_R14 - theta_R15) + g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[29,28] = V_R14*(b_R14_R15*cos(theta_R14 - theta_R15) + g_R14_R15*sin(theta_R14 - theta_R15)) + 2*V_R15*(-b_R14_R15 - bs_R14_R15/2)
        struct[0].Gy[29,29] = V_R14*V_R15*(b_R14_R15*sin(theta_R14 - theta_R15) - g_R14_R15*cos(theta_R14 - theta_R15))
        struct[0].Gy[30,10] = V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[30,11] = V_R06*V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[30,30] = V_R06*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16)) + 2*V_R16*g_R06_R16
        struct[0].Gy[30,31] = V_R06*V_R16*(-b_R06_R16*cos(theta_R06 - theta_R16) - g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[31,10] = V_R16*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16))
        struct[0].Gy[31,11] = V_R06*V_R16*(-b_R06_R16*sin(theta_R06 - theta_R16) + g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[31,30] = V_R06*(b_R06_R16*cos(theta_R06 - theta_R16) + g_R06_R16*sin(theta_R06 - theta_R16)) + 2*V_R16*(-b_R06_R16 - bs_R06_R16/2)
        struct[0].Gy[31,31] = V_R06*V_R16*(b_R06_R16*sin(theta_R06 - theta_R16) - g_R06_R16*cos(theta_R06 - theta_R16))
        struct[0].Gy[32,16] = V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[32,17] = V_R09*V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[32,32] = V_R09*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17)) + 2*V_R17*g_R09_R17
        struct[0].Gy[32,33] = V_R09*V_R17*(-b_R09_R17*cos(theta_R09 - theta_R17) - g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[33,16] = V_R17*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17))
        struct[0].Gy[33,17] = V_R09*V_R17*(-b_R09_R17*sin(theta_R09 - theta_R17) + g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[33,32] = V_R09*(b_R09_R17*cos(theta_R09 - theta_R17) + g_R09_R17*sin(theta_R09 - theta_R17)) + 2*V_R17*(-b_R09_R17 - bs_R09_R17/2)
        struct[0].Gy[33,33] = V_R09*V_R17*(b_R09_R17*sin(theta_R09 - theta_R17) - g_R09_R17*cos(theta_R09 - theta_R17))
        struct[0].Gy[34,18] = V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[34,19] = V_R10*V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[34,34] = V_R10*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18)) + 2*V_R18*g_R10_R18
        struct[0].Gy[34,35] = V_R10*V_R18*(-b_R10_R18*cos(theta_R10 - theta_R18) - g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[35,18] = V_R18*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18))
        struct[0].Gy[35,19] = V_R10*V_R18*(-b_R10_R18*sin(theta_R10 - theta_R18) + g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[35,34] = V_R10*(b_R10_R18*cos(theta_R10 - theta_R18) + g_R10_R18*sin(theta_R10 - theta_R18)) + 2*V_R18*(-b_R10_R18 - bs_R10_R18/2)
        struct[0].Gy[35,35] = V_R10*V_R18*(b_R10_R18*sin(theta_R10 - theta_R18) - g_R10_R18*cos(theta_R10 - theta_R18))
        struct[0].Gy[36,18] = cos(delta_R10 - theta_R10)
        struct[0].Gy[36,19] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[36,36] = X1d_R10
        struct[0].Gy[36,37] = R_a_R10
        struct[0].Gy[37,18] = sin(delta_R10 - theta_R10)
        struct[0].Gy[37,19] = -V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[37,36] = R_a_R10
        struct[0].Gy[37,37] = -X1q_R10
        struct[0].Gy[38,18] = i_d_R10*sin(delta_R10 - theta_R10) + i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[38,19] = -V_R10*i_d_R10*cos(delta_R10 - theta_R10) + V_R10*i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[38,36] = V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[38,37] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[38,38] = -1
        struct[0].Gy[39,18] = i_d_R10*cos(delta_R10 - theta_R10) - i_q_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[39,19] = V_R10*i_d_R10*sin(delta_R10 - theta_R10) + V_R10*i_q_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[39,36] = V_R10*cos(delta_R10 - theta_R10)
        struct[0].Gy[39,37] = -V_R10*sin(delta_R10 - theta_R10)
        struct[0].Gy[39,39] = -1
        struct[0].Gy[40,40] = -1
        struct[0].Gy[41,41] = -1
        struct[0].Gy[41,49] = 1
        struct[0].Gy[42,26] = cos(delta_R14 - theta_R14)
        struct[0].Gy[42,27] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[42,42] = X1d_R14
        struct[0].Gy[42,43] = R_a_R14
        struct[0].Gy[43,26] = sin(delta_R14 - theta_R14)
        struct[0].Gy[43,27] = -V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[43,42] = R_a_R14
        struct[0].Gy[43,43] = -X1q_R14
        struct[0].Gy[44,26] = i_d_R14*sin(delta_R14 - theta_R14) + i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[44,27] = -V_R14*i_d_R14*cos(delta_R14 - theta_R14) + V_R14*i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[44,42] = V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[44,43] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[44,44] = -1
        struct[0].Gy[45,26] = i_d_R14*cos(delta_R14 - theta_R14) - i_q_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[45,27] = V_R14*i_d_R14*sin(delta_R14 - theta_R14) + V_R14*i_q_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[45,42] = V_R14*cos(delta_R14 - theta_R14)
        struct[0].Gy[45,43] = -V_R14*sin(delta_R14 - theta_R14)
        struct[0].Gy[45,45] = -1
        struct[0].Gy[46,46] = -1
        struct[0].Gy[47,47] = -1
        struct[0].Gy[47,50] = 1
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
        struct[0].Gu[22,22] = -1/S_base
        struct[0].Gu[23,23] = -1/S_base
        struct[0].Gu[24,24] = -1/S_base
        struct[0].Gu[25,25] = -1/S_base
        struct[0].Gu[26,26] = -1/S_base
        struct[0].Gu[27,27] = -1/S_base
        struct[0].Gu[28,28] = -1/S_base
        struct[0].Gu[29,29] = -1/S_base
        struct[0].Gu[30,30] = -1/S_base
        struct[0].Gu[31,31] = -1/S_base
        struct[0].Gu[32,32] = -1/S_base
        struct[0].Gu[33,33] = -1/S_base
        struct[0].Gu[34,34] = -1/S_base
        struct[0].Gu[35,35] = -1/S_base
        struct[0].Gu[40,36] = K_a_R10
        struct[0].Gu[40,37] = K_a_R10
        struct[0].Gu[41,38] = 1
        struct[0].Gu[46,39] = K_a_R14
        struct[0].Gu[46,40] = K_a_R14
        struct[0].Gu[47,41] = 1





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
    Fx_ini_rows = [0, 0, 1, 1, 1, 2, 3, 4, 6, 7, 7, 8, 8, 8, 9, 10, 11, 13]

    Fx_ini_cols = [0, 1, 0, 1, 6, 2, 3, 4, 6, 7, 8, 7, 8, 13, 9, 10, 11, 13]

    Fy_ini_rows = [0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 9, 9, 10, 11, 12, 13, 14]

    Fy_ini_cols = [48, 18, 19, 36, 37, 48, 36, 40, 37, 18, 18, 41, 48, 26, 27, 42, 43, 48, 42, 46, 43, 26, 26, 47, 48]

    Gx_ini_rows = [36, 36, 37, 37, 38, 39, 40, 40, 41, 42, 42, 43, 43, 44, 45, 46, 46, 47, 48, 48, 49, 50]

    Gx_ini_cols = [0, 2, 0, 3, 0, 0, 4, 5, 1, 7, 9, 7, 10, 7, 7, 11, 12, 8, 1, 8, 14, 14]

    Gy_ini_rows = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 40, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 47, 47, 48, 49, 50]

    Gy_ini_cols = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 20, 21, 2, 3, 4, 5, 6, 7, 20, 21, 4, 5, 6, 7, 8, 9, 22, 23, 4, 5, 6, 7, 8, 9, 22, 23, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 30, 31, 8, 9, 10, 11, 12, 13, 30, 31, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 14, 15, 16, 17, 18, 19, 32, 33, 14, 15, 16, 17, 18, 19, 32, 33, 16, 17, 18, 19, 34, 35, 38, 16, 17, 18, 19, 34, 35, 39, 4, 5, 20, 21, 4, 5, 20, 21, 6, 7, 22, 23, 24, 25, 6, 7, 22, 23, 24, 25, 22, 23, 24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 24, 25, 26, 27, 28, 29, 44, 24, 25, 26, 27, 28, 29, 45, 26, 27, 28, 29, 26, 27, 28, 29, 10, 11, 30, 31, 10, 11, 30, 31, 16, 17, 32, 33, 16, 17, 32, 33, 18, 19, 34, 35, 18, 19, 34, 35, 18, 19, 36, 37, 18, 19, 36, 37, 18, 19, 36, 37, 38, 18, 19, 36, 37, 39, 40, 41, 49, 26, 27, 42, 43, 26, 27, 42, 43, 26, 27, 42, 43, 44, 26, 27, 42, 43, 45, 46, 47, 50, 48, 49, 50]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols