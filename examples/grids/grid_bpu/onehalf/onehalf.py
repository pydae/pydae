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


class onehalf_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 21
        self.N_y = 33 
        self.N_z = 10 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_01_02', 'b_01_02', 'bs_01_02', 'g_02_03', 'b_02_03', 'bs_02_03', 'U_01_n', 'U_02_n', 'U_03_n', 'S_n_01', 'Omega_b_01', 'H_01', 'T1d0_01', 'T1q0_01', 'X_d_01', 'X_q_01', 'X1d_01', 'X1q_01', 'D_01', 'R_a_01', 'K_delta_01', 'K_sec_01', 'K_a_01', 'K_ai_01', 'T_r_01', 'V_min_01', 'V_max_01', 'K_aw_01', 'Droop_01', 'T_gov_1_01', 'T_gov_2_01', 'T_gov_3_01', 'K_imw_01', 'omega_ref_01', 'T_wo_01', 'T_1_01', 'T_2_01', 'K_stab_01', 'V_lim_01', 'S_n_03', 'Omega_b_03', 'K_p_03', 'T_p_03', 'K_q_03', 'T_q_03', 'X_v_03', 'R_v_03', 'R_s_03', 'C_u_03', 'K_u_0_03', 'K_u_max_03', 'V_u_min_03', 'V_u_max_03', 'R_uc_03', 'K_h_03', 'R_lim_03', 'V_u_lt_03', 'V_u_ht_03', 'Droop_03', 'DB_03', 'T_cur_03', 'R_lim_max_03', 'K_fpfr_03', 'P_f_min_03', 'P_f_max_03', 'K_p_pll_03', 'K_i_pll_03', 'K_speed_03', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [100000.0, 64.70588235294117, -258.8235294117647, 0.0, 12.131762250617438, -7.801776366956552, 0.0, 400.0, 400.0, 400.0, 10000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.001, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 20000.0, 314.1592653589793, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.02, 5.0, 0.005, 0.1, 80, 160, 0.1, 1.0, 0.2, 85, 155, 0.05, 0.001, 10.0, 100.0, 0.0, -1.0, 1.0, 126.0, 3948.0, 1.0, 0.01, 0.01] 
        self.inputs_ini_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'v_ref_01', 'v_pss_01', 'p_c_01', 'p_r_01', 'q_s_ref_03', 'v_u_ref_03', 'omega_ref_03', 'p_gin_0_03', 'p_g_ref_03', 'ramp_p_gin_03'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.03, 0.0, 0.778, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0] 
        self.inputs_run_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'v_ref_01', 'v_pss_01', 'p_c_01', 'p_r_01', 'q_s_ref_03', 'v_u_ref_03', 'omega_ref_03', 'p_gin_0_03', 'p_g_ref_03', 'ramp_p_gin_03'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.03, 0.0, 0.778, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0] 
        self.outputs_list = ['V_01', 'V_02', 'V_03', 'p_e_01', 'p_gin_03', 'p_g_ref_03', 'p_l_03', 'soc_03', 'p_fpfr_03', 'p_f_sat_03'] 
        self.x_list = ['delta_01', 'omega_01', 'e1q_01', 'e1d_01', 'v_c_01', 'xi_v_01', 'x_gov_1_01', 'x_gov_2_01', 'xi_imw_01', 'x_wo_01', 'x_lead_01', 'delta_03', 'xi_p_03', 'xi_q_03', 'e_u_03', 'p_ghr_03', 'k_cur_03', 'inc_p_gin_03', 'theta_pll_03', 'xi_pll_03', 'xi_freq'] 
        self.y_run_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'i_d_01', 'i_q_01', 'p_g_01', 'q_g_01', 'v_f_01', 'p_m_ref_01', 'p_m_01', 'z_wo_01', 'v_pss_01', 'omega_03', 'e_qv_03', 'i_d_03', 'i_q_03', 'p_s_03', 'q_s_03', 'p_m_03', 'p_t_03', 'p_u_03', 'v_u_03', 'k_u_03', 'k_cur_sat_03', 'p_gou_03', 'p_f_03', 'r_lim_03', 'omega_pll_03', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'i_d_01', 'i_q_01', 'p_g_01', 'q_g_01', 'v_f_01', 'p_m_ref_01', 'p_m_01', 'z_wo_01', 'v_pss_01', 'omega_03', 'e_qv_03', 'i_d_03', 'i_q_03', 'p_s_03', 'q_s_03', 'p_m_03', 'p_t_03', 'p_u_03', 'v_u_03', 'k_u_03', 'k_cur_sat_03', 'p_gou_03', 'p_f_03', 'r_lim_03', 'omega_pll_03', 'omega_coi', 'p_agc'] 
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

    def save_params(self,file_name = 'parameters.json'):
        params_dict = {}
        for item in self.params_list:
            params_dict.update({item:self.get_value(item)})

        params_dict_str = json.dumps(params_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(params_dict_str)

    def save_inputs_ini(self,file_name = 'inputs_ini.json'):
        inputs_ini_dict = {}
        for item in self.inputs_ini_list:
            inputs_ini_dict.update({item:self.get_value(item)})

        inputs_ini_dict_str = json.dumps(inputs_ini_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(inputs_ini_dict_str)

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
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    H_01 = struct[0].H_01
    T1d0_01 = struct[0].T1d0_01
    T1q0_01 = struct[0].T1q0_01
    X_d_01 = struct[0].X_d_01
    X_q_01 = struct[0].X_q_01
    X1d_01 = struct[0].X1d_01
    X1q_01 = struct[0].X1q_01
    D_01 = struct[0].D_01
    R_a_01 = struct[0].R_a_01
    K_delta_01 = struct[0].K_delta_01
    K_sec_01 = struct[0].K_sec_01
    K_a_01 = struct[0].K_a_01
    K_ai_01 = struct[0].K_ai_01
    T_r_01 = struct[0].T_r_01
    V_min_01 = struct[0].V_min_01
    V_max_01 = struct[0].V_max_01
    K_aw_01 = struct[0].K_aw_01
    Droop_01 = struct[0].Droop_01
    T_gov_1_01 = struct[0].T_gov_1_01
    T_gov_2_01 = struct[0].T_gov_2_01
    T_gov_3_01 = struct[0].T_gov_3_01
    K_imw_01 = struct[0].K_imw_01
    omega_ref_01 = struct[0].omega_ref_01
    T_wo_01 = struct[0].T_wo_01
    T_1_01 = struct[0].T_1_01
    T_2_01 = struct[0].T_2_01
    K_stab_01 = struct[0].K_stab_01
    V_lim_01 = struct[0].V_lim_01
    S_n_03 = struct[0].S_n_03
    Omega_b_03 = struct[0].Omega_b_03
    K_p_03 = struct[0].K_p_03
    T_p_03 = struct[0].T_p_03
    K_q_03 = struct[0].K_q_03
    T_q_03 = struct[0].T_q_03
    X_v_03 = struct[0].X_v_03
    R_v_03 = struct[0].R_v_03
    R_s_03 = struct[0].R_s_03
    C_u_03 = struct[0].C_u_03
    K_u_0_03 = struct[0].K_u_0_03
    K_u_max_03 = struct[0].K_u_max_03
    V_u_min_03 = struct[0].V_u_min_03
    V_u_max_03 = struct[0].V_u_max_03
    R_uc_03 = struct[0].R_uc_03
    K_h_03 = struct[0].K_h_03
    R_lim_03 = struct[0].R_lim_03
    V_u_lt_03 = struct[0].V_u_lt_03
    V_u_ht_03 = struct[0].V_u_ht_03
    Droop_03 = struct[0].Droop_03
    DB_03 = struct[0].DB_03
    T_cur_03 = struct[0].T_cur_03
    R_lim_max_03 = struct[0].R_lim_max_03
    K_fpfr_03 = struct[0].K_fpfr_03
    P_f_min_03 = struct[0].P_f_min_03
    P_f_max_03 = struct[0].P_f_max_03
    K_p_pll_03 = struct[0].K_p_pll_03
    K_i_pll_03 = struct[0].K_i_pll_03
    K_speed_03 = struct[0].K_speed_03
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    v_ref_01 = struct[0].v_ref_01
    v_pss_01 = struct[0].v_pss_01
    p_c_01 = struct[0].p_c_01
    p_r_01 = struct[0].p_r_01
    q_s_ref_03 = struct[0].q_s_ref_03
    v_u_ref_03 = struct[0].v_u_ref_03
    omega_ref_03 = struct[0].omega_ref_03
    p_gin_0_03 = struct[0].p_gin_0_03
    p_g_ref_03 = struct[0].p_g_ref_03
    ramp_p_gin_03 = struct[0].ramp_p_gin_03
    
    # Dynamical states:
    delta_01 = struct[0].x[0,0]
    omega_01 = struct[0].x[1,0]
    e1q_01 = struct[0].x[2,0]
    e1d_01 = struct[0].x[3,0]
    v_c_01 = struct[0].x[4,0]
    xi_v_01 = struct[0].x[5,0]
    x_gov_1_01 = struct[0].x[6,0]
    x_gov_2_01 = struct[0].x[7,0]
    xi_imw_01 = struct[0].x[8,0]
    x_wo_01 = struct[0].x[9,0]
    x_lead_01 = struct[0].x[10,0]
    delta_03 = struct[0].x[11,0]
    xi_p_03 = struct[0].x[12,0]
    xi_q_03 = struct[0].x[13,0]
    e_u_03 = struct[0].x[14,0]
    p_ghr_03 = struct[0].x[15,0]
    k_cur_03 = struct[0].x[16,0]
    inc_p_gin_03 = struct[0].x[17,0]
    theta_pll_03 = struct[0].x[18,0]
    xi_pll_03 = struct[0].x[19,0]
    xi_freq = struct[0].x[20,0]
    
    # Algebraic states:
    V_01 = struct[0].y_ini[0,0]
    theta_01 = struct[0].y_ini[1,0]
    V_02 = struct[0].y_ini[2,0]
    theta_02 = struct[0].y_ini[3,0]
    V_03 = struct[0].y_ini[4,0]
    theta_03 = struct[0].y_ini[5,0]
    i_d_01 = struct[0].y_ini[6,0]
    i_q_01 = struct[0].y_ini[7,0]
    p_g_01 = struct[0].y_ini[8,0]
    q_g_01 = struct[0].y_ini[9,0]
    v_f_01 = struct[0].y_ini[10,0]
    p_m_ref_01 = struct[0].y_ini[11,0]
    p_m_01 = struct[0].y_ini[12,0]
    z_wo_01 = struct[0].y_ini[13,0]
    v_pss_01 = struct[0].y_ini[14,0]
    omega_03 = struct[0].y_ini[15,0]
    e_qv_03 = struct[0].y_ini[16,0]
    i_d_03 = struct[0].y_ini[17,0]
    i_q_03 = struct[0].y_ini[18,0]
    p_s_03 = struct[0].y_ini[19,0]
    q_s_03 = struct[0].y_ini[20,0]
    p_m_03 = struct[0].y_ini[21,0]
    p_t_03 = struct[0].y_ini[22,0]
    p_u_03 = struct[0].y_ini[23,0]
    v_u_03 = struct[0].y_ini[24,0]
    k_u_03 = struct[0].y_ini[25,0]
    k_cur_sat_03 = struct[0].y_ini[26,0]
    p_gou_03 = struct[0].y_ini[27,0]
    p_f_03 = struct[0].y_ini[28,0]
    r_lim_03 = struct[0].y_ini[29,0]
    omega_pll_03 = struct[0].y_ini[30,0]
    omega_coi = struct[0].y_ini[31,0]
    p_agc = struct[0].y_ini[32,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[1,0] = (-D_01*(omega_01 - omega_coi) - i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) - i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01)) + p_m_01)/(2*H_01)
        struct[0].f[2,0] = (-e1q_01 - i_d_01*(-X1d_01 + X_d_01) + v_f_01)/T1d0_01
        struct[0].f[3,0] = (-e1d_01 + i_q_01*(-X1q_01 + X_q_01))/T1q0_01
        struct[0].f[4,0] = (V_01 - v_c_01)/T_r_01
        struct[0].f[5,0] = -K_aw_01*(K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01 - v_f_01) - v_c_01 + v_pss_01 + v_ref_01
        struct[0].f[6,0] = (p_m_ref_01 - x_gov_1_01)/T_gov_1_01
        struct[0].f[7,0] = (x_gov_1_01 - x_gov_2_01)/T_gov_3_01
        struct[0].f[8,0] = K_imw_01*(p_c_01 - p_g_01) - 1.0e-6*xi_imw_01
        struct[0].f[9,0] = (omega_01 - x_wo_01 - 1.0)/T_wo_01
        struct[0].f[10,0] = (-x_lead_01 + z_wo_01)/T_2_01
        struct[0].f[11,0] = Omega_b_03*(omega_03 - omega_coi)
        struct[0].f[12,0] = p_m_03 - p_s_03
        struct[0].f[13,0] = -q_s_03 + q_s_ref_03
        struct[0].f[14,0] = S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1))
        struct[0].f[15,0] = Piecewise(np.array([(-r_lim_03, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (r_lim_03, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (K_h_03*(-p_ghr_03 + p_gou_03), True)]))
        struct[0].f[16,0] = (-k_cur_03 + p_g_ref_03/(inc_p_gin_03 + p_gin_0_03) + Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03))/T_cur_03
        struct[0].f[17,0] = -0.001*inc_p_gin_03 + ramp_p_gin_03
        struct[0].f[18,0] = K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)) - Omega_b_03*omega_coi
        struct[0].f[19,0] = V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)
        struct[0].f[20,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_g_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_g_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*g_02_03 - S_n_03*p_s_03/S_base
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - bs_02_03/2) - S_n_03*q_s_03/S_base
        struct[0].g[6,0] = R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01) + X1d_01*i_d_01 - e1q_01
        struct[0].g[7,0] = R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01) - X1q_01*i_q_01 - e1d_01
        struct[0].g[8,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_g_01
        struct[0].g[9,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_g_01
        struct[0].g[10,0] = -v_f_01 + Piecewise(np.array([(V_min_01, V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (V_max_01, V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01, True)]))
        struct[0].g[11,0] = K_sec_01*p_agc - p_m_ref_01 + p_r_01 + xi_imw_01 - (omega_01 - omega_ref_01)/Droop_01
        struct[0].g[12,0] = T_gov_2_01*(x_gov_1_01 - x_gov_2_01)/T_gov_3_01 - p_m_01 + x_gov_2_01
        struct[0].g[13,0] = omega_01 - x_wo_01 - z_wo_01 - 1.0
        struct[0].g[14,0] = -v_pss_01 + Piecewise(np.array([(-V_lim_01, V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (V_lim_01, V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01), True)]))
        struct[0].g[15,0] = K_p_03*(p_m_03 - p_s_03 + xi_p_03/T_p_03) - omega_03
        struct[0].g[16,0] = K_q_03*(-q_s_03 + q_s_ref_03 + xi_q_03/T_q_03) - e_qv_03
        struct[0].g[17,0] = -R_v_03*i_d_03 - V_03*sin(delta_03 - theta_03) + X_v_03*i_q_03
        struct[0].g[18,0] = -R_v_03*i_q_03 - V_03*cos(delta_03 - theta_03) - X_v_03*i_d_03 + e_qv_03
        struct[0].g[19,0] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03) - p_s_03
        struct[0].g[20,0] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03) - q_s_03
        struct[0].g[21,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)])) + p_ghr_03 - p_m_03 + p_s_03 - p_t_03 + p_u_03
        struct[0].g[22,0] = i_d_03*(R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)) + i_q_03*(R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)) - p_t_03
        struct[0].g[23,0] = -p_u_03 - k_u_03*(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].g[24,0] = R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1) + e_u_03 - v_u_03
        struct[0].g[25,0] = -k_u_03 + Piecewise(np.array([(K_u_max_03, V_u_min_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (K_u_max_03, V_u_max_03 < v_u_03), (K_u_0_03, True)]))
        struct[0].g[26,0] = -k_cur_sat_03 + Piecewise(np.array([(0.0001, k_cur_03 < 0.0001), (1, k_cur_03 > 1), (k_cur_03, True)]))
        struct[0].g[28,0] = -p_f_03 - Piecewise(np.array([((0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03)), ((-0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)), (0.0, True)]))
        struct[0].g[29,0] = -r_lim_03 + Piecewise(np.array([(R_lim_max_03, (omega_03 > 0.5*DB_03 + omega_ref_03) | (omega_03 < -0.5*DB_03 + omega_ref_03)), (0.0, True)])) + Piecewise(np.array([(R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (R_lim_03, True)]))
        struct[0].g[30,0] = -omega_pll_03 + (K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)))/Omega_b_03
        struct[0].g[31,0] = -omega_coi + (H_01*S_n_01*omega_01 + S_n_03*T_p_03*omega_03/(2*K_p_03))/(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03))
        struct[0].g[32,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) + i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01))
        struct[0].h[4,0] = inc_p_gin_03 + p_gin_0_03
        struct[0].h[5,0] = p_g_ref_03
        struct[0].h[6,0] = -p_s_03 + p_t_03
        struct[0].h[7,0] = (-V_u_min_03**2 + e_u_03**2)/(V_u_max_03**2 - V_u_min_03**2)
        struct[0].h[8,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
        struct[0].h[9,0] = Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_01
        struct[0].Fx_ini[0,1] = Omega_b_01
        struct[0].Fx_ini[1,0] = (-V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fx_ini[1,1] = -D_01/(2*H_01)
        struct[0].Fx_ini[2,2] = -1/T1d0_01
        struct[0].Fx_ini[3,3] = -1/T1q0_01
        struct[0].Fx_ini[4,4] = -1/T_r_01
        struct[0].Fx_ini[5,4] = K_a_01*K_aw_01 - 1
        struct[0].Fx_ini[5,5] = -K_ai_01*K_aw_01
        struct[0].Fx_ini[6,6] = -1/T_gov_1_01
        struct[0].Fx_ini[7,6] = 1/T_gov_3_01
        struct[0].Fx_ini[7,7] = -1/T_gov_3_01
        struct[0].Fx_ini[9,1] = 1/T_wo_01
        struct[0].Fx_ini[9,9] = -1/T_wo_01
        struct[0].Fx_ini[10,10] = -1/T_2_01
        struct[0].Fx_ini[15,15] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (-K_h_03, True)]))
        struct[0].Fx_ini[16,16] = -1/T_cur_03
        struct[0].Fx_ini[16,17] = (-p_g_ref_03/(inc_p_gin_03 + p_gin_0_03)**2 - Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03)**2)/T_cur_03
        struct[0].Fx_ini[18,18] = K_p_pll_03*(-V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03))
        struct[0].Fx_ini[18,19] = K_i_pll_03
        struct[0].Fx_ini[19,18] = -V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03)

    if mode == 11:

        struct[0].Fy_ini[0,31] = -Omega_b_01 
        struct[0].Fy_ini[1,0] = (-i_d_01*sin(delta_01 - theta_01) - i_q_01*cos(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,1] = (V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,6] = (-2*R_a_01*i_d_01 - V_01*sin(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,7] = (-2*R_a_01*i_q_01 - V_01*cos(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,12] = 1/(2*H_01) 
        struct[0].Fy_ini[1,31] = D_01/(2*H_01) 
        struct[0].Fy_ini[2,6] = (X1d_01 - X_d_01)/T1d0_01 
        struct[0].Fy_ini[2,10] = 1/T1d0_01 
        struct[0].Fy_ini[3,7] = (-X1q_01 + X_q_01)/T1q0_01 
        struct[0].Fy_ini[4,0] = 1/T_r_01 
        struct[0].Fy_ini[5,10] = K_aw_01 
        struct[0].Fy_ini[5,14] = -K_a_01*K_aw_01 + 1 
        struct[0].Fy_ini[6,11] = 1/T_gov_1_01 
        struct[0].Fy_ini[8,8] = -K_imw_01 
        struct[0].Fy_ini[10,13] = 1/T_2_01 
        struct[0].Fy_ini[11,15] = Omega_b_03 
        struct[0].Fy_ini[11,31] = -Omega_b_03 
        struct[0].Fy_ini[12,19] = -1 
        struct[0].Fy_ini[12,21] = 1 
        struct[0].Fy_ini[13,20] = -1 
        struct[0].Fy_ini[14,22] = -S_n_03/(C_u_03*(v_u_03 + 0.1)) 
        struct[0].Fy_ini[14,24] = -S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1)**2) 
        struct[0].Fy_ini[14,27] = S_n_03/(C_u_03*(v_u_03 + 0.1)) 
        struct[0].Fy_ini[15,27] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (K_h_03, True)])) 
        struct[0].Fy_ini[15,29] = Piecewise(np.array([(-1, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (1, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (0, True)])) 
        struct[0].Fy_ini[16,28] = Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))/(T_cur_03*(inc_p_gin_03 + p_gin_0_03)) 
        struct[0].Fy_ini[18,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03)) 
        struct[0].Fy_ini[18,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03)) 
        struct[0].Fy_ini[18,31] = -Omega_b_03 
        struct[0].Fy_ini[19,4] = sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03) 
        struct[0].Fy_ini[19,5] = V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03) 
        struct[0].Fy_ini[20,31] = -1 

        struct[0].Gx_ini[6,0] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gx_ini[6,2] = -1
        struct[0].Gx_ini[7,0] = V_01*cos(delta_01 - theta_01)
        struct[0].Gx_ini[7,3] = -1
        struct[0].Gx_ini[8,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gx_ini[9,0] = -V_01*i_d_01*sin(delta_01 - theta_01) - V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gx_ini[10,4] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (-K_a_01, True)]))
        struct[0].Gx_ini[10,5] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_ai_01, True)]))
        struct[0].Gx_ini[11,1] = -1/Droop_01
        struct[0].Gx_ini[11,8] = 1
        struct[0].Gx_ini[12,6] = T_gov_2_01/T_gov_3_01
        struct[0].Gx_ini[12,7] = -T_gov_2_01/T_gov_3_01 + 1
        struct[0].Gx_ini[13,1] = 1
        struct[0].Gx_ini[13,9] = -1
        struct[0].Gx_ini[14,10] = Piecewise(np.array([(0, (V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)) | (V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01))), (K_stab_01*(-T_1_01/T_2_01 + 1), True)]))
        struct[0].Gx_ini[15,12] = K_p_03/T_p_03
        struct[0].Gx_ini[16,13] = K_q_03/T_q_03
        struct[0].Gx_ini[17,11] = -V_03*cos(delta_03 - theta_03)
        struct[0].Gx_ini[18,11] = V_03*sin(delta_03 - theta_03)
        struct[0].Gx_ini[19,11] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gx_ini[20,11] = -V_03*i_d_03*sin(delta_03 - theta_03) - V_03*i_q_03*cos(delta_03 - theta_03)
        struct[0].Gx_ini[21,15] = 1
        struct[0].Gx_ini[22,11] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gx_ini[24,14] = 1
        struct[0].Gx_ini[26,16] = Piecewise(np.array([(0, (k_cur_03 > 1) | (k_cur_03 < 0.0001)), (1, True)]))
        struct[0].Gx_ini[27,17] = k_cur_sat_03
        struct[0].Gx_ini[30,18] = K_p_pll_03*(-V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03))/Omega_b_03
        struct[0].Gx_ini[30,19] = K_i_pll_03/Omega_b_03
        struct[0].Gx_ini[31,1] = H_01*S_n_01/(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03))
        struct[0].Gx_ini[32,20] = K_i_agc

        struct[0].Gy_ini[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,8] = -S_n_01/S_base
        struct[0].Gy_ini[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,9] = -S_n_01/S_base
        struct[0].Gy_ini[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*g_02_03
        struct[0].Gy_ini[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[4,19] = -S_n_03/S_base
        struct[0].Gy_ini[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - bs_02_03/2)
        struct[0].Gy_ini[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[5,20] = -S_n_03/S_base
        struct[0].Gy_ini[6,0] = cos(delta_01 - theta_01)
        struct[0].Gy_ini[6,1] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[6,6] = X1d_01
        struct[0].Gy_ini[6,7] = R_a_01
        struct[0].Gy_ini[7,0] = sin(delta_01 - theta_01)
        struct[0].Gy_ini[7,1] = -V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[7,6] = R_a_01
        struct[0].Gy_ini[7,7] = -X1q_01
        struct[0].Gy_ini[8,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[8,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[8,6] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[8,7] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[9,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[9,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[9,6] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[9,7] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[10,14] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gy_ini[11,32] = K_sec_01
        struct[0].Gy_ini[14,13] = Piecewise(np.array([(0, (V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)) | (V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01))), (K_stab_01*T_1_01/T_2_01, True)]))
        struct[0].Gy_ini[15,19] = -K_p_03
        struct[0].Gy_ini[15,21] = K_p_03
        struct[0].Gy_ini[16,20] = -K_q_03
        struct[0].Gy_ini[17,4] = -sin(delta_03 - theta_03)
        struct[0].Gy_ini[17,5] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[17,17] = -R_v_03
        struct[0].Gy_ini[17,18] = X_v_03
        struct[0].Gy_ini[18,4] = -cos(delta_03 - theta_03)
        struct[0].Gy_ini[18,5] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[18,17] = -X_v_03
        struct[0].Gy_ini[18,18] = -R_v_03
        struct[0].Gy_ini[19,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[19,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[19,17] = V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[19,18] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[20,4] = i_d_03*cos(delta_03 - theta_03) - i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[20,5] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[20,17] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[20,18] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[21,28] = K_fpfr_03*Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))
        struct[0].Gy_ini[22,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[22,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[22,17] = 2*R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[22,18] = 2*R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[23,24] = 2*k_u_03*v_u_03/V_u_max_03**2
        struct[0].Gy_ini[23,25] = -(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].Gy_ini[24,22] = -R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy_ini[24,24] = -R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1)**2 - 1
        struct[0].Gy_ini[24,27] = R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy_ini[25,24] = Piecewise(np.array([(0, V_u_min_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy_ini[27,26] = inc_p_gin_03 + p_gin_0_03
        struct[0].Gy_ini[28,15] = -Piecewise(np.array([((1 - K_speed_03)/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy_ini[28,30] = -Piecewise(np.array([(K_speed_03/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy_ini[29,24] = Piecewise(np.array([((-R_lim_03 + R_lim_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-R_lim_03 + R_lim_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy_ini[30,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03))/Omega_b_03
        struct[0].Gy_ini[30,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03))/Omega_b_03
        struct[0].Gy_ini[31,15] = S_n_03*T_p_03/(2*K_p_03*(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03)))
        struct[0].Gy_ini[32,31] = -K_p_agc



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    H_01 = struct[0].H_01
    T1d0_01 = struct[0].T1d0_01
    T1q0_01 = struct[0].T1q0_01
    X_d_01 = struct[0].X_d_01
    X_q_01 = struct[0].X_q_01
    X1d_01 = struct[0].X1d_01
    X1q_01 = struct[0].X1q_01
    D_01 = struct[0].D_01
    R_a_01 = struct[0].R_a_01
    K_delta_01 = struct[0].K_delta_01
    K_sec_01 = struct[0].K_sec_01
    K_a_01 = struct[0].K_a_01
    K_ai_01 = struct[0].K_ai_01
    T_r_01 = struct[0].T_r_01
    V_min_01 = struct[0].V_min_01
    V_max_01 = struct[0].V_max_01
    K_aw_01 = struct[0].K_aw_01
    Droop_01 = struct[0].Droop_01
    T_gov_1_01 = struct[0].T_gov_1_01
    T_gov_2_01 = struct[0].T_gov_2_01
    T_gov_3_01 = struct[0].T_gov_3_01
    K_imw_01 = struct[0].K_imw_01
    omega_ref_01 = struct[0].omega_ref_01
    T_wo_01 = struct[0].T_wo_01
    T_1_01 = struct[0].T_1_01
    T_2_01 = struct[0].T_2_01
    K_stab_01 = struct[0].K_stab_01
    V_lim_01 = struct[0].V_lim_01
    S_n_03 = struct[0].S_n_03
    Omega_b_03 = struct[0].Omega_b_03
    K_p_03 = struct[0].K_p_03
    T_p_03 = struct[0].T_p_03
    K_q_03 = struct[0].K_q_03
    T_q_03 = struct[0].T_q_03
    X_v_03 = struct[0].X_v_03
    R_v_03 = struct[0].R_v_03
    R_s_03 = struct[0].R_s_03
    C_u_03 = struct[0].C_u_03
    K_u_0_03 = struct[0].K_u_0_03
    K_u_max_03 = struct[0].K_u_max_03
    V_u_min_03 = struct[0].V_u_min_03
    V_u_max_03 = struct[0].V_u_max_03
    R_uc_03 = struct[0].R_uc_03
    K_h_03 = struct[0].K_h_03
    R_lim_03 = struct[0].R_lim_03
    V_u_lt_03 = struct[0].V_u_lt_03
    V_u_ht_03 = struct[0].V_u_ht_03
    Droop_03 = struct[0].Droop_03
    DB_03 = struct[0].DB_03
    T_cur_03 = struct[0].T_cur_03
    R_lim_max_03 = struct[0].R_lim_max_03
    K_fpfr_03 = struct[0].K_fpfr_03
    P_f_min_03 = struct[0].P_f_min_03
    P_f_max_03 = struct[0].P_f_max_03
    K_p_pll_03 = struct[0].K_p_pll_03
    K_i_pll_03 = struct[0].K_i_pll_03
    K_speed_03 = struct[0].K_speed_03
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    v_ref_01 = struct[0].v_ref_01
    v_pss_01 = struct[0].v_pss_01
    p_c_01 = struct[0].p_c_01
    p_r_01 = struct[0].p_r_01
    q_s_ref_03 = struct[0].q_s_ref_03
    v_u_ref_03 = struct[0].v_u_ref_03
    omega_ref_03 = struct[0].omega_ref_03
    p_gin_0_03 = struct[0].p_gin_0_03
    p_g_ref_03 = struct[0].p_g_ref_03
    ramp_p_gin_03 = struct[0].ramp_p_gin_03
    
    # Dynamical states:
    delta_01 = struct[0].x[0,0]
    omega_01 = struct[0].x[1,0]
    e1q_01 = struct[0].x[2,0]
    e1d_01 = struct[0].x[3,0]
    v_c_01 = struct[0].x[4,0]
    xi_v_01 = struct[0].x[5,0]
    x_gov_1_01 = struct[0].x[6,0]
    x_gov_2_01 = struct[0].x[7,0]
    xi_imw_01 = struct[0].x[8,0]
    x_wo_01 = struct[0].x[9,0]
    x_lead_01 = struct[0].x[10,0]
    delta_03 = struct[0].x[11,0]
    xi_p_03 = struct[0].x[12,0]
    xi_q_03 = struct[0].x[13,0]
    e_u_03 = struct[0].x[14,0]
    p_ghr_03 = struct[0].x[15,0]
    k_cur_03 = struct[0].x[16,0]
    inc_p_gin_03 = struct[0].x[17,0]
    theta_pll_03 = struct[0].x[18,0]
    xi_pll_03 = struct[0].x[19,0]
    xi_freq = struct[0].x[20,0]
    
    # Algebraic states:
    V_01 = struct[0].y_run[0,0]
    theta_01 = struct[0].y_run[1,0]
    V_02 = struct[0].y_run[2,0]
    theta_02 = struct[0].y_run[3,0]
    V_03 = struct[0].y_run[4,0]
    theta_03 = struct[0].y_run[5,0]
    i_d_01 = struct[0].y_run[6,0]
    i_q_01 = struct[0].y_run[7,0]
    p_g_01 = struct[0].y_run[8,0]
    q_g_01 = struct[0].y_run[9,0]
    v_f_01 = struct[0].y_run[10,0]
    p_m_ref_01 = struct[0].y_run[11,0]
    p_m_01 = struct[0].y_run[12,0]
    z_wo_01 = struct[0].y_run[13,0]
    v_pss_01 = struct[0].y_run[14,0]
    omega_03 = struct[0].y_run[15,0]
    e_qv_03 = struct[0].y_run[16,0]
    i_d_03 = struct[0].y_run[17,0]
    i_q_03 = struct[0].y_run[18,0]
    p_s_03 = struct[0].y_run[19,0]
    q_s_03 = struct[0].y_run[20,0]
    p_m_03 = struct[0].y_run[21,0]
    p_t_03 = struct[0].y_run[22,0]
    p_u_03 = struct[0].y_run[23,0]
    v_u_03 = struct[0].y_run[24,0]
    k_u_03 = struct[0].y_run[25,0]
    k_cur_sat_03 = struct[0].y_run[26,0]
    p_gou_03 = struct[0].y_run[27,0]
    p_f_03 = struct[0].y_run[28,0]
    r_lim_03 = struct[0].y_run[29,0]
    omega_pll_03 = struct[0].y_run[30,0]
    omega_coi = struct[0].y_run[31,0]
    p_agc = struct[0].y_run[32,0]
    
    struct[0].u_run[0,0] = P_01
    struct[0].u_run[1,0] = Q_01
    struct[0].u_run[2,0] = P_02
    struct[0].u_run[3,0] = Q_02
    struct[0].u_run[4,0] = P_03
    struct[0].u_run[5,0] = Q_03
    struct[0].u_run[6,0] = v_ref_01
    struct[0].u_run[7,0] = v_pss_01
    struct[0].u_run[8,0] = p_c_01
    struct[0].u_run[9,0] = p_r_01
    struct[0].u_run[10,0] = q_s_ref_03
    struct[0].u_run[11,0] = v_u_ref_03
    struct[0].u_run[12,0] = omega_ref_03
    struct[0].u_run[13,0] = p_gin_0_03
    struct[0].u_run[14,0] = p_g_ref_03
    struct[0].u_run[15,0] = ramp_p_gin_03
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[1,0] = (-D_01*(omega_01 - omega_coi) - i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) - i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01)) + p_m_01)/(2*H_01)
        struct[0].f[2,0] = (-e1q_01 - i_d_01*(-X1d_01 + X_d_01) + v_f_01)/T1d0_01
        struct[0].f[3,0] = (-e1d_01 + i_q_01*(-X1q_01 + X_q_01))/T1q0_01
        struct[0].f[4,0] = (V_01 - v_c_01)/T_r_01
        struct[0].f[5,0] = -K_aw_01*(K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01 - v_f_01) - v_c_01 + v_pss_01 + v_ref_01
        struct[0].f[6,0] = (p_m_ref_01 - x_gov_1_01)/T_gov_1_01
        struct[0].f[7,0] = (x_gov_1_01 - x_gov_2_01)/T_gov_3_01
        struct[0].f[8,0] = K_imw_01*(p_c_01 - p_g_01) - 1.0e-6*xi_imw_01
        struct[0].f[9,0] = (omega_01 - x_wo_01 - 1.0)/T_wo_01
        struct[0].f[10,0] = (-x_lead_01 + z_wo_01)/T_2_01
        struct[0].f[11,0] = Omega_b_03*(omega_03 - omega_coi)
        struct[0].f[12,0] = p_m_03 - p_s_03
        struct[0].f[13,0] = -q_s_03 + q_s_ref_03
        struct[0].f[14,0] = S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1))
        struct[0].f[15,0] = Piecewise(np.array([(-r_lim_03, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (r_lim_03, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (K_h_03*(-p_ghr_03 + p_gou_03), True)]))
        struct[0].f[16,0] = (-k_cur_03 + p_g_ref_03/(inc_p_gin_03 + p_gin_0_03) + Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03))/T_cur_03
        struct[0].f[17,0] = -0.001*inc_p_gin_03 + ramp_p_gin_03
        struct[0].f[18,0] = K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)) - Omega_b_03*omega_coi
        struct[0].f[19,0] = V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)
        struct[0].f[20,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_g_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_g_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*g_02_03 - S_n_03*p_s_03/S_base
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - bs_02_03/2) - S_n_03*q_s_03/S_base
        struct[0].g[6,0] = R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01) + X1d_01*i_d_01 - e1q_01
        struct[0].g[7,0] = R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01) - X1q_01*i_q_01 - e1d_01
        struct[0].g[8,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_g_01
        struct[0].g[9,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_g_01
        struct[0].g[10,0] = -v_f_01 + Piecewise(np.array([(V_min_01, V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (V_max_01, V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01, True)]))
        struct[0].g[11,0] = K_sec_01*p_agc - p_m_ref_01 + p_r_01 + xi_imw_01 - (omega_01 - omega_ref_01)/Droop_01
        struct[0].g[12,0] = T_gov_2_01*(x_gov_1_01 - x_gov_2_01)/T_gov_3_01 - p_m_01 + x_gov_2_01
        struct[0].g[13,0] = omega_01 - x_wo_01 - z_wo_01 - 1.0
        struct[0].g[14,0] = -v_pss_01 + Piecewise(np.array([(-V_lim_01, V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (V_lim_01, V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01), True)]))
        struct[0].g[15,0] = K_p_03*(p_m_03 - p_s_03 + xi_p_03/T_p_03) - omega_03
        struct[0].g[16,0] = K_q_03*(-q_s_03 + q_s_ref_03 + xi_q_03/T_q_03) - e_qv_03
        struct[0].g[17,0] = -R_v_03*i_d_03 - V_03*sin(delta_03 - theta_03) + X_v_03*i_q_03
        struct[0].g[18,0] = -R_v_03*i_q_03 - V_03*cos(delta_03 - theta_03) - X_v_03*i_d_03 + e_qv_03
        struct[0].g[19,0] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03) - p_s_03
        struct[0].g[20,0] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03) - q_s_03
        struct[0].g[21,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)])) + p_ghr_03 - p_m_03 + p_s_03 - p_t_03 + p_u_03
        struct[0].g[22,0] = i_d_03*(R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)) + i_q_03*(R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)) - p_t_03
        struct[0].g[23,0] = -p_u_03 - k_u_03*(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].g[24,0] = R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1) + e_u_03 - v_u_03
        struct[0].g[25,0] = -k_u_03 + Piecewise(np.array([(K_u_max_03, V_u_min_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (K_u_max_03, V_u_max_03 < v_u_03), (K_u_0_03, True)]))
        struct[0].g[26,0] = -k_cur_sat_03 + Piecewise(np.array([(0.0001, k_cur_03 < 0.0001), (1, k_cur_03 > 1), (k_cur_03, True)]))
        struct[0].g[27,0] = k_cur_sat_03*(inc_p_gin_03 + p_gin_0_03) - p_gou_03
        struct[0].g[28,0] = -p_f_03 - Piecewise(np.array([((0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03)), ((-0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)), (0.0, True)]))
        struct[0].g[29,0] = -r_lim_03 + Piecewise(np.array([(R_lim_max_03, (omega_03 > 0.5*DB_03 + omega_ref_03) | (omega_03 < -0.5*DB_03 + omega_ref_03)), (0.0, True)])) + Piecewise(np.array([(R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (R_lim_03, True)]))
        struct[0].g[30,0] = -omega_pll_03 + (K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)))/Omega_b_03
        struct[0].g[31,0] = -omega_coi + (H_01*S_n_01*omega_01 + S_n_03*T_p_03*omega_03/(2*K_p_03))/(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03))
        struct[0].g[32,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) + i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01))
        struct[0].h[4,0] = inc_p_gin_03 + p_gin_0_03
        struct[0].h[5,0] = p_g_ref_03
        struct[0].h[6,0] = -p_s_03 + p_t_03
        struct[0].h[7,0] = (-V_u_min_03**2 + e_u_03**2)/(V_u_max_03**2 - V_u_min_03**2)
        struct[0].h[8,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
        struct[0].h[9,0] = Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_01
        struct[0].Fx[0,1] = Omega_b_01
        struct[0].Fx[1,0] = (-V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fx[1,1] = -D_01/(2*H_01)
        struct[0].Fx[2,2] = -1/T1d0_01
        struct[0].Fx[3,3] = -1/T1q0_01
        struct[0].Fx[4,4] = -1/T_r_01
        struct[0].Fx[5,4] = K_a_01*K_aw_01 - 1
        struct[0].Fx[5,5] = -K_ai_01*K_aw_01
        struct[0].Fx[6,6] = -1/T_gov_1_01
        struct[0].Fx[7,6] = 1/T_gov_3_01
        struct[0].Fx[7,7] = -1/T_gov_3_01
        struct[0].Fx[9,1] = 1/T_wo_01
        struct[0].Fx[9,9] = -1/T_wo_01
        struct[0].Fx[10,10] = -1/T_2_01
        struct[0].Fx[15,15] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (-K_h_03, True)]))
        struct[0].Fx[16,16] = -1/T_cur_03
        struct[0].Fx[16,17] = (-p_g_ref_03/(inc_p_gin_03 + p_gin_0_03)**2 - Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03)**2)/T_cur_03
        struct[0].Fx[18,18] = K_p_pll_03*(-V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03))
        struct[0].Fx[18,19] = K_i_pll_03
        struct[0].Fx[19,18] = -V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03)

    if mode == 11:

        struct[0].Fy[0,31] = -Omega_b_01
        struct[0].Fy[1,0] = (-i_d_01*sin(delta_01 - theta_01) - i_q_01*cos(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,1] = (V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,6] = (-2*R_a_01*i_d_01 - V_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,7] = (-2*R_a_01*i_q_01 - V_01*cos(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,12] = 1/(2*H_01)
        struct[0].Fy[1,31] = D_01/(2*H_01)
        struct[0].Fy[2,6] = (X1d_01 - X_d_01)/T1d0_01
        struct[0].Fy[2,10] = 1/T1d0_01
        struct[0].Fy[3,7] = (-X1q_01 + X_q_01)/T1q0_01
        struct[0].Fy[4,0] = 1/T_r_01
        struct[0].Fy[5,10] = K_aw_01
        struct[0].Fy[5,14] = -K_a_01*K_aw_01 + 1
        struct[0].Fy[6,11] = 1/T_gov_1_01
        struct[0].Fy[8,8] = -K_imw_01
        struct[0].Fy[10,13] = 1/T_2_01
        struct[0].Fy[11,15] = Omega_b_03
        struct[0].Fy[11,31] = -Omega_b_03
        struct[0].Fy[12,19] = -1
        struct[0].Fy[12,21] = 1
        struct[0].Fy[13,20] = -1
        struct[0].Fy[14,22] = -S_n_03/(C_u_03*(v_u_03 + 0.1))
        struct[0].Fy[14,24] = -S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1)**2)
        struct[0].Fy[14,27] = S_n_03/(C_u_03*(v_u_03 + 0.1))
        struct[0].Fy[15,27] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (K_h_03, True)]))
        struct[0].Fy[15,29] = Piecewise(np.array([(-1, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (1, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (0, True)]))
        struct[0].Fy[16,28] = Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))/(T_cur_03*(inc_p_gin_03 + p_gin_0_03))
        struct[0].Fy[18,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03))
        struct[0].Fy[18,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03))
        struct[0].Fy[18,31] = -Omega_b_03
        struct[0].Fy[19,4] = sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03)
        struct[0].Fy[19,5] = V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03)
        struct[0].Fy[20,31] = -1

        struct[0].Gx[6,0] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gx[6,2] = -1
        struct[0].Gx[7,0] = V_01*cos(delta_01 - theta_01)
        struct[0].Gx[7,3] = -1
        struct[0].Gx[8,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gx[9,0] = -V_01*i_d_01*sin(delta_01 - theta_01) - V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gx[10,4] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (-K_a_01, True)]))
        struct[0].Gx[10,5] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_ai_01, True)]))
        struct[0].Gx[11,1] = -1/Droop_01
        struct[0].Gx[11,8] = 1
        struct[0].Gx[12,6] = T_gov_2_01/T_gov_3_01
        struct[0].Gx[12,7] = -T_gov_2_01/T_gov_3_01 + 1
        struct[0].Gx[13,1] = 1
        struct[0].Gx[13,9] = -1
        struct[0].Gx[14,10] = Piecewise(np.array([(0, (V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)) | (V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01))), (K_stab_01*(-T_1_01/T_2_01 + 1), True)]))
        struct[0].Gx[15,12] = K_p_03/T_p_03
        struct[0].Gx[16,13] = K_q_03/T_q_03
        struct[0].Gx[17,11] = -V_03*cos(delta_03 - theta_03)
        struct[0].Gx[18,11] = V_03*sin(delta_03 - theta_03)
        struct[0].Gx[19,11] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gx[20,11] = -V_03*i_d_03*sin(delta_03 - theta_03) - V_03*i_q_03*cos(delta_03 - theta_03)
        struct[0].Gx[21,15] = 1
        struct[0].Gx[22,11] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gx[24,14] = 1
        struct[0].Gx[26,16] = Piecewise(np.array([(0, (k_cur_03 > 1) | (k_cur_03 < 0.0001)), (1, True)]))
        struct[0].Gx[27,17] = k_cur_sat_03
        struct[0].Gx[30,18] = K_p_pll_03*(-V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03))/Omega_b_03
        struct[0].Gx[30,19] = K_i_pll_03/Omega_b_03
        struct[0].Gx[31,1] = H_01*S_n_01/(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03))
        struct[0].Gx[32,20] = K_i_agc

        struct[0].Gy[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,8] = -S_n_01/S_base
        struct[0].Gy[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,9] = -S_n_01/S_base
        struct[0].Gy[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*g_02_03
        struct[0].Gy[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[4,19] = -S_n_03/S_base
        struct[0].Gy[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - bs_02_03/2)
        struct[0].Gy[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[5,20] = -S_n_03/S_base
        struct[0].Gy[6,0] = cos(delta_01 - theta_01)
        struct[0].Gy[6,1] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy[6,6] = X1d_01
        struct[0].Gy[6,7] = R_a_01
        struct[0].Gy[7,0] = sin(delta_01 - theta_01)
        struct[0].Gy[7,1] = -V_01*cos(delta_01 - theta_01)
        struct[0].Gy[7,6] = R_a_01
        struct[0].Gy[7,7] = -X1q_01
        struct[0].Gy[8,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[8,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[8,6] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy[8,7] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[9,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[9,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[9,6] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[9,7] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy[10,14] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gy[11,32] = K_sec_01
        struct[0].Gy[14,13] = Piecewise(np.array([(0, (V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)) | (V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01))), (K_stab_01*T_1_01/T_2_01, True)]))
        struct[0].Gy[15,19] = -K_p_03
        struct[0].Gy[15,21] = K_p_03
        struct[0].Gy[16,20] = -K_q_03
        struct[0].Gy[17,4] = -sin(delta_03 - theta_03)
        struct[0].Gy[17,5] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy[17,17] = -R_v_03
        struct[0].Gy[17,18] = X_v_03
        struct[0].Gy[18,4] = -cos(delta_03 - theta_03)
        struct[0].Gy[18,5] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy[18,17] = -X_v_03
        struct[0].Gy[18,18] = -R_v_03
        struct[0].Gy[19,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy[19,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy[19,17] = V_03*sin(delta_03 - theta_03)
        struct[0].Gy[19,18] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy[20,4] = i_d_03*cos(delta_03 - theta_03) - i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy[20,5] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy[20,17] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy[20,18] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy[21,28] = K_fpfr_03*Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))
        struct[0].Gy[22,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy[22,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy[22,17] = 2*R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)
        struct[0].Gy[22,18] = 2*R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)
        struct[0].Gy[23,24] = 2*k_u_03*v_u_03/V_u_max_03**2
        struct[0].Gy[23,25] = -(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].Gy[24,22] = -R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy[24,24] = -R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1)**2 - 1
        struct[0].Gy[24,27] = R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy[25,24] = Piecewise(np.array([(0, V_u_min_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy[27,26] = inc_p_gin_03 + p_gin_0_03
        struct[0].Gy[28,15] = -Piecewise(np.array([((1 - K_speed_03)/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy[28,30] = -Piecewise(np.array([(K_speed_03/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy[29,24] = Piecewise(np.array([((-R_lim_03 + R_lim_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-R_lim_03 + R_lim_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy[30,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03))/Omega_b_03
        struct[0].Gy[30,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03))/Omega_b_03
        struct[0].Gy[31,15] = S_n_03*T_p_03/(2*K_p_03*(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03)))
        struct[0].Gy[32,31] = -K_p_agc

    if mode > 12:

        struct[0].Fu[5,6] = -K_a_01*K_aw_01 + 1
        struct[0].Fu[5,7] = -K_a_01*K_aw_01 + 1
        struct[0].Fu[8,8] = K_imw_01
        struct[0].Fu[13,10] = 1
        struct[0].Fu[16,13] = (-p_g_ref_03/(inc_p_gin_03 + p_gin_0_03)**2 - Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03)**2)/T_cur_03
        struct[0].Fu[16,14] = 1/(T_cur_03*(inc_p_gin_03 + p_gin_0_03))
        struct[0].Fu[17,15] = 1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[10,6] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gu[10,7] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gu[16,10] = K_q_03
        struct[0].Gu[23,11] = -2*k_u_03*v_u_ref_03/V_u_max_03**2
        struct[0].Gu[27,13] = k_cur_sat_03
        struct[0].Gu[28,12] = -Piecewise(np.array([(-1/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))

        struct[0].Hx[3,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Hx[4,17] = 1
        struct[0].Hx[7,14] = 2*e_u_03/(V_u_max_03**2 - V_u_min_03**2)

        struct[0].Hy[0,0] = 1
        struct[0].Hy[1,2] = 1
        struct[0].Hy[2,4] = 1
        struct[0].Hy[3,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Hy[3,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Hy[3,6] = 2*R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)
        struct[0].Hy[3,7] = 2*R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01)
        struct[0].Hy[6,19] = -1
        struct[0].Hy[6,22] = 1
        struct[0].Hy[8,28] = K_fpfr_03*Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))
        struct[0].Hy[9,28] = Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))

        struct[0].Hu[4,13] = 1
        struct[0].Hu[5,14] = 1



def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    H_01 = struct[0].H_01
    T1d0_01 = struct[0].T1d0_01
    T1q0_01 = struct[0].T1q0_01
    X_d_01 = struct[0].X_d_01
    X_q_01 = struct[0].X_q_01
    X1d_01 = struct[0].X1d_01
    X1q_01 = struct[0].X1q_01
    D_01 = struct[0].D_01
    R_a_01 = struct[0].R_a_01
    K_delta_01 = struct[0].K_delta_01
    K_sec_01 = struct[0].K_sec_01
    K_a_01 = struct[0].K_a_01
    K_ai_01 = struct[0].K_ai_01
    T_r_01 = struct[0].T_r_01
    V_min_01 = struct[0].V_min_01
    V_max_01 = struct[0].V_max_01
    K_aw_01 = struct[0].K_aw_01
    Droop_01 = struct[0].Droop_01
    T_gov_1_01 = struct[0].T_gov_1_01
    T_gov_2_01 = struct[0].T_gov_2_01
    T_gov_3_01 = struct[0].T_gov_3_01
    K_imw_01 = struct[0].K_imw_01
    omega_ref_01 = struct[0].omega_ref_01
    T_wo_01 = struct[0].T_wo_01
    T_1_01 = struct[0].T_1_01
    T_2_01 = struct[0].T_2_01
    K_stab_01 = struct[0].K_stab_01
    V_lim_01 = struct[0].V_lim_01
    S_n_03 = struct[0].S_n_03
    Omega_b_03 = struct[0].Omega_b_03
    K_p_03 = struct[0].K_p_03
    T_p_03 = struct[0].T_p_03
    K_q_03 = struct[0].K_q_03
    T_q_03 = struct[0].T_q_03
    X_v_03 = struct[0].X_v_03
    R_v_03 = struct[0].R_v_03
    R_s_03 = struct[0].R_s_03
    C_u_03 = struct[0].C_u_03
    K_u_0_03 = struct[0].K_u_0_03
    K_u_max_03 = struct[0].K_u_max_03
    V_u_min_03 = struct[0].V_u_min_03
    V_u_max_03 = struct[0].V_u_max_03
    R_uc_03 = struct[0].R_uc_03
    K_h_03 = struct[0].K_h_03
    R_lim_03 = struct[0].R_lim_03
    V_u_lt_03 = struct[0].V_u_lt_03
    V_u_ht_03 = struct[0].V_u_ht_03
    Droop_03 = struct[0].Droop_03
    DB_03 = struct[0].DB_03
    T_cur_03 = struct[0].T_cur_03
    R_lim_max_03 = struct[0].R_lim_max_03
    K_fpfr_03 = struct[0].K_fpfr_03
    P_f_min_03 = struct[0].P_f_min_03
    P_f_max_03 = struct[0].P_f_max_03
    K_p_pll_03 = struct[0].K_p_pll_03
    K_i_pll_03 = struct[0].K_i_pll_03
    K_speed_03 = struct[0].K_speed_03
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    v_ref_01 = struct[0].v_ref_01
    v_pss_01 = struct[0].v_pss_01
    p_c_01 = struct[0].p_c_01
    p_r_01 = struct[0].p_r_01
    q_s_ref_03 = struct[0].q_s_ref_03
    v_u_ref_03 = struct[0].v_u_ref_03
    omega_ref_03 = struct[0].omega_ref_03
    p_gin_0_03 = struct[0].p_gin_0_03
    p_g_ref_03 = struct[0].p_g_ref_03
    ramp_p_gin_03 = struct[0].ramp_p_gin_03
    
    # Dynamical states:
    delta_01 = struct[0].x[0,0]
    omega_01 = struct[0].x[1,0]
    e1q_01 = struct[0].x[2,0]
    e1d_01 = struct[0].x[3,0]
    v_c_01 = struct[0].x[4,0]
    xi_v_01 = struct[0].x[5,0]
    x_gov_1_01 = struct[0].x[6,0]
    x_gov_2_01 = struct[0].x[7,0]
    xi_imw_01 = struct[0].x[8,0]
    x_wo_01 = struct[0].x[9,0]
    x_lead_01 = struct[0].x[10,0]
    delta_03 = struct[0].x[11,0]
    xi_p_03 = struct[0].x[12,0]
    xi_q_03 = struct[0].x[13,0]
    e_u_03 = struct[0].x[14,0]
    p_ghr_03 = struct[0].x[15,0]
    k_cur_03 = struct[0].x[16,0]
    inc_p_gin_03 = struct[0].x[17,0]
    theta_pll_03 = struct[0].x[18,0]
    xi_pll_03 = struct[0].x[19,0]
    xi_freq = struct[0].x[20,0]
    
    # Algebraic states:
    V_01 = struct[0].y_ini[0,0]
    theta_01 = struct[0].y_ini[1,0]
    V_02 = struct[0].y_ini[2,0]
    theta_02 = struct[0].y_ini[3,0]
    V_03 = struct[0].y_ini[4,0]
    theta_03 = struct[0].y_ini[5,0]
    i_d_01 = struct[0].y_ini[6,0]
    i_q_01 = struct[0].y_ini[7,0]
    p_g_01 = struct[0].y_ini[8,0]
    q_g_01 = struct[0].y_ini[9,0]
    v_f_01 = struct[0].y_ini[10,0]
    p_m_ref_01 = struct[0].y_ini[11,0]
    p_m_01 = struct[0].y_ini[12,0]
    z_wo_01 = struct[0].y_ini[13,0]
    v_pss_01 = struct[0].y_ini[14,0]
    omega_03 = struct[0].y_ini[15,0]
    e_qv_03 = struct[0].y_ini[16,0]
    i_d_03 = struct[0].y_ini[17,0]
    i_q_03 = struct[0].y_ini[18,0]
    p_s_03 = struct[0].y_ini[19,0]
    q_s_03 = struct[0].y_ini[20,0]
    p_m_03 = struct[0].y_ini[21,0]
    p_t_03 = struct[0].y_ini[22,0]
    p_u_03 = struct[0].y_ini[23,0]
    v_u_03 = struct[0].y_ini[24,0]
    k_u_03 = struct[0].y_ini[25,0]
    k_cur_sat_03 = struct[0].y_ini[26,0]
    p_gou_03 = struct[0].y_ini[27,0]
    p_f_03 = struct[0].y_ini[28,0]
    r_lim_03 = struct[0].y_ini[29,0]
    omega_pll_03 = struct[0].y_ini[30,0]
    omega_coi = struct[0].y_ini[31,0]
    p_agc = struct[0].y_ini[32,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[1,0] = (-D_01*(omega_01 - omega_coi) - i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) - i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01)) + p_m_01)/(2*H_01)
        struct[0].f[2,0] = (-e1q_01 - i_d_01*(-X1d_01 + X_d_01) + v_f_01)/T1d0_01
        struct[0].f[3,0] = (-e1d_01 + i_q_01*(-X1q_01 + X_q_01))/T1q0_01
        struct[0].f[4,0] = (V_01 - v_c_01)/T_r_01
        struct[0].f[5,0] = -K_aw_01*(K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01 - v_f_01) - v_c_01 + v_pss_01 + v_ref_01
        struct[0].f[6,0] = (p_m_ref_01 - x_gov_1_01)/T_gov_1_01
        struct[0].f[7,0] = (x_gov_1_01 - x_gov_2_01)/T_gov_3_01
        struct[0].f[8,0] = K_imw_01*(p_c_01 - p_g_01) - 1.0e-6*xi_imw_01
        struct[0].f[9,0] = (omega_01 - x_wo_01 - 1.0)/T_wo_01
        struct[0].f[10,0] = (-x_lead_01 + z_wo_01)/T_2_01
        struct[0].f[11,0] = Omega_b_03*(omega_03 - omega_coi)
        struct[0].f[12,0] = p_m_03 - p_s_03
        struct[0].f[13,0] = -q_s_03 + q_s_ref_03
        struct[0].f[14,0] = S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1))
        struct[0].f[15,0] = Piecewise(np.array([(-r_lim_03, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (r_lim_03, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (K_h_03*(-p_ghr_03 + p_gou_03), True)]))
        struct[0].f[16,0] = (-k_cur_03 + p_g_ref_03/(inc_p_gin_03 + p_gin_0_03) + Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03))/T_cur_03
        struct[0].f[17,0] = -0.001*inc_p_gin_03 + ramp_p_gin_03
        struct[0].f[18,0] = K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)) - Omega_b_03*omega_coi
        struct[0].f[19,0] = V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)
        struct[0].f[20,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_g_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_g_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*g_02_03 - S_n_03*p_s_03/S_base
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - bs_02_03/2) - S_n_03*q_s_03/S_base
        struct[0].g[6,0] = R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01) + X1d_01*i_d_01 - e1q_01
        struct[0].g[7,0] = R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01) - X1q_01*i_q_01 - e1d_01
        struct[0].g[8,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_g_01
        struct[0].g[9,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_g_01
        struct[0].g[10,0] = -v_f_01 + Piecewise(np.array([(V_min_01, V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (V_max_01, V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01, True)]))
        struct[0].g[11,0] = K_sec_01*p_agc - p_m_ref_01 + p_r_01 + xi_imw_01 - (omega_01 - omega_ref_01)/Droop_01
        struct[0].g[12,0] = T_gov_2_01*(x_gov_1_01 - x_gov_2_01)/T_gov_3_01 - p_m_01 + x_gov_2_01
        struct[0].g[13,0] = omega_01 - x_wo_01 - z_wo_01 - 1.0
        struct[0].g[14,0] = -v_pss_01 + Piecewise(np.array([(-V_lim_01, V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (V_lim_01, V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01), True)]))
        struct[0].g[15,0] = K_p_03*(p_m_03 - p_s_03 + xi_p_03/T_p_03) - omega_03
        struct[0].g[16,0] = K_q_03*(-q_s_03 + q_s_ref_03 + xi_q_03/T_q_03) - e_qv_03
        struct[0].g[17,0] = -R_v_03*i_d_03 - V_03*sin(delta_03 - theta_03) + X_v_03*i_q_03
        struct[0].g[18,0] = -R_v_03*i_q_03 - V_03*cos(delta_03 - theta_03) - X_v_03*i_d_03 + e_qv_03
        struct[0].g[19,0] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03) - p_s_03
        struct[0].g[20,0] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03) - q_s_03
        struct[0].g[21,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)])) + p_ghr_03 - p_m_03 + p_s_03 - p_t_03 + p_u_03
        struct[0].g[22,0] = i_d_03*(R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)) + i_q_03*(R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)) - p_t_03
        struct[0].g[23,0] = -p_u_03 - k_u_03*(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].g[24,0] = R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1) + e_u_03 - v_u_03
        struct[0].g[25,0] = -k_u_03 + Piecewise(np.array([(K_u_max_03, V_u_min_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (K_u_max_03, V_u_max_03 < v_u_03), (K_u_0_03, True)]))
        struct[0].g[26,0] = -k_cur_sat_03 + Piecewise(np.array([(0.0001, k_cur_03 < 0.0001), (1, k_cur_03 > 1), (k_cur_03, True)]))
        struct[0].g[27,0] = k_cur_sat_03*(inc_p_gin_03 + p_gin_0_03) - p_gou_03
        struct[0].g[28,0] = -p_f_03 - Piecewise(np.array([((0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03)), ((-0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)), (0.0, True)]))
        struct[0].g[29,0] = -r_lim_03 + Piecewise(np.array([(R_lim_max_03, (omega_03 > 0.5*DB_03 + omega_ref_03) | (omega_03 < -0.5*DB_03 + omega_ref_03)), (0.0, True)])) + Piecewise(np.array([(R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (R_lim_03, True)]))
        struct[0].g[30,0] = -omega_pll_03 + (K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)))/Omega_b_03
        struct[0].g[31,0] = -omega_coi + (H_01*S_n_01*omega_01 + S_n_03*T_p_03*omega_03/(2*K_p_03))/(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03))
        struct[0].g[32,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) + i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01))
        struct[0].h[4,0] = inc_p_gin_03 + p_gin_0_03
        struct[0].h[5,0] = p_g_ref_03
        struct[0].h[6,0] = -p_s_03 + p_t_03
        struct[0].h[7,0] = (-V_u_min_03**2 + e_u_03**2)/(V_u_max_03**2 - V_u_min_03**2)
        struct[0].h[8,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
        struct[0].h[9,0] = Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_01
        struct[0].Fx_ini[0,1] = Omega_b_01
        struct[0].Fx_ini[1,0] = (-V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fx_ini[1,1] = -D_01/(2*H_01)
        struct[0].Fx_ini[2,2] = -1/T1d0_01
        struct[0].Fx_ini[3,3] = -1/T1q0_01
        struct[0].Fx_ini[4,4] = -1/T_r_01
        struct[0].Fx_ini[5,4] = K_a_01*K_aw_01 - 1
        struct[0].Fx_ini[5,5] = -K_ai_01*K_aw_01
        struct[0].Fx_ini[6,6] = -1/T_gov_1_01
        struct[0].Fx_ini[7,6] = 1/T_gov_3_01
        struct[0].Fx_ini[7,7] = -1/T_gov_3_01
        struct[0].Fx_ini[8,8] = -0.00000100000000000000
        struct[0].Fx_ini[9,1] = 1/T_wo_01
        struct[0].Fx_ini[9,9] = -1/T_wo_01
        struct[0].Fx_ini[10,10] = -1/T_2_01
        struct[0].Fx_ini[15,15] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (-K_h_03, True)]))
        struct[0].Fx_ini[16,16] = -1/T_cur_03
        struct[0].Fx_ini[16,17] = (-p_g_ref_03/(inc_p_gin_03 + p_gin_0_03)**2 - Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03)**2)/T_cur_03
        struct[0].Fx_ini[17,17] = -0.00100000000000000
        struct[0].Fx_ini[18,18] = K_p_pll_03*(-V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03))
        struct[0].Fx_ini[18,19] = K_i_pll_03
        struct[0].Fx_ini[19,18] = -V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03)

    if mode == 11:

        struct[0].Fy_ini[0,31] = -Omega_b_01 
        struct[0].Fy_ini[1,0] = (-i_d_01*sin(delta_01 - theta_01) - i_q_01*cos(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,1] = (V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,6] = (-2*R_a_01*i_d_01 - V_01*sin(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,7] = (-2*R_a_01*i_q_01 - V_01*cos(delta_01 - theta_01))/(2*H_01) 
        struct[0].Fy_ini[1,12] = 1/(2*H_01) 
        struct[0].Fy_ini[1,31] = D_01/(2*H_01) 
        struct[0].Fy_ini[2,6] = (X1d_01 - X_d_01)/T1d0_01 
        struct[0].Fy_ini[2,10] = 1/T1d0_01 
        struct[0].Fy_ini[3,7] = (-X1q_01 + X_q_01)/T1q0_01 
        struct[0].Fy_ini[4,0] = 1/T_r_01 
        struct[0].Fy_ini[5,10] = K_aw_01 
        struct[0].Fy_ini[5,14] = -K_a_01*K_aw_01 + 1 
        struct[0].Fy_ini[6,11] = 1/T_gov_1_01 
        struct[0].Fy_ini[8,8] = -K_imw_01 
        struct[0].Fy_ini[10,13] = 1/T_2_01 
        struct[0].Fy_ini[11,15] = Omega_b_03 
        struct[0].Fy_ini[11,31] = -Omega_b_03 
        struct[0].Fy_ini[12,19] = -1 
        struct[0].Fy_ini[12,21] = 1 
        struct[0].Fy_ini[13,20] = -1 
        struct[0].Fy_ini[14,22] = -S_n_03/(C_u_03*(v_u_03 + 0.1)) 
        struct[0].Fy_ini[14,24] = -S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1)**2) 
        struct[0].Fy_ini[14,27] = S_n_03/(C_u_03*(v_u_03 + 0.1)) 
        struct[0].Fy_ini[15,27] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (K_h_03, True)])) 
        struct[0].Fy_ini[15,29] = Piecewise(np.array([(-1, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (1, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (0, True)])) 
        struct[0].Fy_ini[16,28] = Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))/(T_cur_03*(inc_p_gin_03 + p_gin_0_03)) 
        struct[0].Fy_ini[18,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03)) 
        struct[0].Fy_ini[18,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03)) 
        struct[0].Fy_ini[18,31] = -Omega_b_03 
        struct[0].Fy_ini[19,4] = sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03) 
        struct[0].Fy_ini[19,5] = V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03) 
        struct[0].Fy_ini[20,31] = -1 

        struct[0].Gy_ini[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,8] = -S_n_01/S_base
        struct[0].Gy_ini[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,9] = -S_n_01/S_base
        struct[0].Gy_ini[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*g_02_03
        struct[0].Gy_ini[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[4,19] = -S_n_03/S_base
        struct[0].Gy_ini[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - bs_02_03/2)
        struct[0].Gy_ini[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[5,20] = -S_n_03/S_base
        struct[0].Gy_ini[6,0] = cos(delta_01 - theta_01)
        struct[0].Gy_ini[6,1] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[6,6] = X1d_01
        struct[0].Gy_ini[6,7] = R_a_01
        struct[0].Gy_ini[7,0] = sin(delta_01 - theta_01)
        struct[0].Gy_ini[7,1] = -V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[7,6] = R_a_01
        struct[0].Gy_ini[7,7] = -X1q_01
        struct[0].Gy_ini[8,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[8,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[8,6] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[8,7] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[8,8] = -1
        struct[0].Gy_ini[9,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[9,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[9,6] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[9,7] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[9,9] = -1
        struct[0].Gy_ini[10,10] = -1
        struct[0].Gy_ini[10,14] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gy_ini[11,11] = -1
        struct[0].Gy_ini[11,32] = K_sec_01
        struct[0].Gy_ini[12,12] = -1
        struct[0].Gy_ini[13,13] = -1
        struct[0].Gy_ini[14,13] = Piecewise(np.array([(0, (V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)) | (V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01))), (K_stab_01*T_1_01/T_2_01, True)]))
        struct[0].Gy_ini[14,14] = -1
        struct[0].Gy_ini[15,15] = -1
        struct[0].Gy_ini[15,19] = -K_p_03
        struct[0].Gy_ini[15,21] = K_p_03
        struct[0].Gy_ini[16,16] = -1
        struct[0].Gy_ini[16,20] = -K_q_03
        struct[0].Gy_ini[17,4] = -sin(delta_03 - theta_03)
        struct[0].Gy_ini[17,5] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[17,17] = -R_v_03
        struct[0].Gy_ini[17,18] = X_v_03
        struct[0].Gy_ini[18,4] = -cos(delta_03 - theta_03)
        struct[0].Gy_ini[18,5] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[18,16] = 1
        struct[0].Gy_ini[18,17] = -X_v_03
        struct[0].Gy_ini[18,18] = -R_v_03
        struct[0].Gy_ini[19,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[19,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[19,17] = V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[19,18] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[19,19] = -1
        struct[0].Gy_ini[20,4] = i_d_03*cos(delta_03 - theta_03) - i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[20,5] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[20,17] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[20,18] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[20,20] = -1
        struct[0].Gy_ini[21,19] = 1
        struct[0].Gy_ini[21,21] = -1
        struct[0].Gy_ini[21,22] = -1
        struct[0].Gy_ini[21,23] = 1
        struct[0].Gy_ini[21,28] = K_fpfr_03*Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))
        struct[0].Gy_ini[22,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[22,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[22,17] = 2*R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)
        struct[0].Gy_ini[22,18] = 2*R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)
        struct[0].Gy_ini[22,22] = -1
        struct[0].Gy_ini[23,23] = -1
        struct[0].Gy_ini[23,24] = 2*k_u_03*v_u_03/V_u_max_03**2
        struct[0].Gy_ini[23,25] = -(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].Gy_ini[24,22] = -R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy_ini[24,24] = -R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1)**2 - 1
        struct[0].Gy_ini[24,27] = R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy_ini[25,24] = Piecewise(np.array([(0, V_u_min_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy_ini[25,25] = -1
        struct[0].Gy_ini[26,26] = -1
        struct[0].Gy_ini[27,26] = inc_p_gin_03 + p_gin_0_03
        struct[0].Gy_ini[27,27] = -1
        struct[0].Gy_ini[28,15] = -Piecewise(np.array([((1 - K_speed_03)/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy_ini[28,28] = -1
        struct[0].Gy_ini[28,30] = -Piecewise(np.array([(K_speed_03/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy_ini[29,24] = Piecewise(np.array([((-R_lim_03 + R_lim_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-R_lim_03 + R_lim_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy_ini[29,29] = -1
        struct[0].Gy_ini[30,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03))/Omega_b_03
        struct[0].Gy_ini[30,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03))/Omega_b_03
        struct[0].Gy_ini[30,30] = -1
        struct[0].Gy_ini[31,15] = S_n_03*T_p_03/(2*K_p_03*(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03)))
        struct[0].Gy_ini[31,31] = -1
        struct[0].Gy_ini[32,31] = -K_p_agc
        struct[0].Gy_ini[32,32] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    H_01 = struct[0].H_01
    T1d0_01 = struct[0].T1d0_01
    T1q0_01 = struct[0].T1q0_01
    X_d_01 = struct[0].X_d_01
    X_q_01 = struct[0].X_q_01
    X1d_01 = struct[0].X1d_01
    X1q_01 = struct[0].X1q_01
    D_01 = struct[0].D_01
    R_a_01 = struct[0].R_a_01
    K_delta_01 = struct[0].K_delta_01
    K_sec_01 = struct[0].K_sec_01
    K_a_01 = struct[0].K_a_01
    K_ai_01 = struct[0].K_ai_01
    T_r_01 = struct[0].T_r_01
    V_min_01 = struct[0].V_min_01
    V_max_01 = struct[0].V_max_01
    K_aw_01 = struct[0].K_aw_01
    Droop_01 = struct[0].Droop_01
    T_gov_1_01 = struct[0].T_gov_1_01
    T_gov_2_01 = struct[0].T_gov_2_01
    T_gov_3_01 = struct[0].T_gov_3_01
    K_imw_01 = struct[0].K_imw_01
    omega_ref_01 = struct[0].omega_ref_01
    T_wo_01 = struct[0].T_wo_01
    T_1_01 = struct[0].T_1_01
    T_2_01 = struct[0].T_2_01
    K_stab_01 = struct[0].K_stab_01
    V_lim_01 = struct[0].V_lim_01
    S_n_03 = struct[0].S_n_03
    Omega_b_03 = struct[0].Omega_b_03
    K_p_03 = struct[0].K_p_03
    T_p_03 = struct[0].T_p_03
    K_q_03 = struct[0].K_q_03
    T_q_03 = struct[0].T_q_03
    X_v_03 = struct[0].X_v_03
    R_v_03 = struct[0].R_v_03
    R_s_03 = struct[0].R_s_03
    C_u_03 = struct[0].C_u_03
    K_u_0_03 = struct[0].K_u_0_03
    K_u_max_03 = struct[0].K_u_max_03
    V_u_min_03 = struct[0].V_u_min_03
    V_u_max_03 = struct[0].V_u_max_03
    R_uc_03 = struct[0].R_uc_03
    K_h_03 = struct[0].K_h_03
    R_lim_03 = struct[0].R_lim_03
    V_u_lt_03 = struct[0].V_u_lt_03
    V_u_ht_03 = struct[0].V_u_ht_03
    Droop_03 = struct[0].Droop_03
    DB_03 = struct[0].DB_03
    T_cur_03 = struct[0].T_cur_03
    R_lim_max_03 = struct[0].R_lim_max_03
    K_fpfr_03 = struct[0].K_fpfr_03
    P_f_min_03 = struct[0].P_f_min_03
    P_f_max_03 = struct[0].P_f_max_03
    K_p_pll_03 = struct[0].K_p_pll_03
    K_i_pll_03 = struct[0].K_i_pll_03
    K_speed_03 = struct[0].K_speed_03
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    v_ref_01 = struct[0].v_ref_01
    v_pss_01 = struct[0].v_pss_01
    p_c_01 = struct[0].p_c_01
    p_r_01 = struct[0].p_r_01
    q_s_ref_03 = struct[0].q_s_ref_03
    v_u_ref_03 = struct[0].v_u_ref_03
    omega_ref_03 = struct[0].omega_ref_03
    p_gin_0_03 = struct[0].p_gin_0_03
    p_g_ref_03 = struct[0].p_g_ref_03
    ramp_p_gin_03 = struct[0].ramp_p_gin_03
    
    # Dynamical states:
    delta_01 = struct[0].x[0,0]
    omega_01 = struct[0].x[1,0]
    e1q_01 = struct[0].x[2,0]
    e1d_01 = struct[0].x[3,0]
    v_c_01 = struct[0].x[4,0]
    xi_v_01 = struct[0].x[5,0]
    x_gov_1_01 = struct[0].x[6,0]
    x_gov_2_01 = struct[0].x[7,0]
    xi_imw_01 = struct[0].x[8,0]
    x_wo_01 = struct[0].x[9,0]
    x_lead_01 = struct[0].x[10,0]
    delta_03 = struct[0].x[11,0]
    xi_p_03 = struct[0].x[12,0]
    xi_q_03 = struct[0].x[13,0]
    e_u_03 = struct[0].x[14,0]
    p_ghr_03 = struct[0].x[15,0]
    k_cur_03 = struct[0].x[16,0]
    inc_p_gin_03 = struct[0].x[17,0]
    theta_pll_03 = struct[0].x[18,0]
    xi_pll_03 = struct[0].x[19,0]
    xi_freq = struct[0].x[20,0]
    
    # Algebraic states:
    V_01 = struct[0].y_run[0,0]
    theta_01 = struct[0].y_run[1,0]
    V_02 = struct[0].y_run[2,0]
    theta_02 = struct[0].y_run[3,0]
    V_03 = struct[0].y_run[4,0]
    theta_03 = struct[0].y_run[5,0]
    i_d_01 = struct[0].y_run[6,0]
    i_q_01 = struct[0].y_run[7,0]
    p_g_01 = struct[0].y_run[8,0]
    q_g_01 = struct[0].y_run[9,0]
    v_f_01 = struct[0].y_run[10,0]
    p_m_ref_01 = struct[0].y_run[11,0]
    p_m_01 = struct[0].y_run[12,0]
    z_wo_01 = struct[0].y_run[13,0]
    v_pss_01 = struct[0].y_run[14,0]
    omega_03 = struct[0].y_run[15,0]
    e_qv_03 = struct[0].y_run[16,0]
    i_d_03 = struct[0].y_run[17,0]
    i_q_03 = struct[0].y_run[18,0]
    p_s_03 = struct[0].y_run[19,0]
    q_s_03 = struct[0].y_run[20,0]
    p_m_03 = struct[0].y_run[21,0]
    p_t_03 = struct[0].y_run[22,0]
    p_u_03 = struct[0].y_run[23,0]
    v_u_03 = struct[0].y_run[24,0]
    k_u_03 = struct[0].y_run[25,0]
    k_cur_sat_03 = struct[0].y_run[26,0]
    p_gou_03 = struct[0].y_run[27,0]
    p_f_03 = struct[0].y_run[28,0]
    r_lim_03 = struct[0].y_run[29,0]
    omega_pll_03 = struct[0].y_run[30,0]
    omega_coi = struct[0].y_run[31,0]
    p_agc = struct[0].y_run[32,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[1,0] = (-D_01*(omega_01 - omega_coi) - i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) - i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01)) + p_m_01)/(2*H_01)
        struct[0].f[2,0] = (-e1q_01 - i_d_01*(-X1d_01 + X_d_01) + v_f_01)/T1d0_01
        struct[0].f[3,0] = (-e1d_01 + i_q_01*(-X1q_01 + X_q_01))/T1q0_01
        struct[0].f[4,0] = (V_01 - v_c_01)/T_r_01
        struct[0].f[5,0] = -K_aw_01*(K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01 - v_f_01) - v_c_01 + v_pss_01 + v_ref_01
        struct[0].f[6,0] = (p_m_ref_01 - x_gov_1_01)/T_gov_1_01
        struct[0].f[7,0] = (x_gov_1_01 - x_gov_2_01)/T_gov_3_01
        struct[0].f[8,0] = K_imw_01*(p_c_01 - p_g_01) - 1.0e-6*xi_imw_01
        struct[0].f[9,0] = (omega_01 - x_wo_01 - 1.0)/T_wo_01
        struct[0].f[10,0] = (-x_lead_01 + z_wo_01)/T_2_01
        struct[0].f[11,0] = Omega_b_03*(omega_03 - omega_coi)
        struct[0].f[12,0] = p_m_03 - p_s_03
        struct[0].f[13,0] = -q_s_03 + q_s_ref_03
        struct[0].f[14,0] = S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1))
        struct[0].f[15,0] = Piecewise(np.array([(-r_lim_03, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (r_lim_03, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (K_h_03*(-p_ghr_03 + p_gou_03), True)]))
        struct[0].f[16,0] = (-k_cur_03 + p_g_ref_03/(inc_p_gin_03 + p_gin_0_03) + Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03))/T_cur_03
        struct[0].f[17,0] = -0.001*inc_p_gin_03 + ramp_p_gin_03
        struct[0].f[18,0] = K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)) - Omega_b_03*omega_coi
        struct[0].f[19,0] = V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)
        struct[0].f[20,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_g_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_g_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*g_02_03 - S_n_03*p_s_03/S_base
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - bs_02_03/2) - S_n_03*q_s_03/S_base
        struct[0].g[6,0] = R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01) + X1d_01*i_d_01 - e1q_01
        struct[0].g[7,0] = R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01) - X1q_01*i_q_01 - e1d_01
        struct[0].g[8,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_g_01
        struct[0].g[9,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_g_01
        struct[0].g[10,0] = -v_f_01 + Piecewise(np.array([(V_min_01, V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (V_max_01, V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01), (K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01, True)]))
        struct[0].g[11,0] = K_sec_01*p_agc - p_m_ref_01 + p_r_01 + xi_imw_01 - (omega_01 - omega_ref_01)/Droop_01
        struct[0].g[12,0] = T_gov_2_01*(x_gov_1_01 - x_gov_2_01)/T_gov_3_01 - p_m_01 + x_gov_2_01
        struct[0].g[13,0] = omega_01 - x_wo_01 - z_wo_01 - 1.0
        struct[0].g[14,0] = -v_pss_01 + Piecewise(np.array([(-V_lim_01, V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (V_lim_01, V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)), (K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01), True)]))
        struct[0].g[15,0] = K_p_03*(p_m_03 - p_s_03 + xi_p_03/T_p_03) - omega_03
        struct[0].g[16,0] = K_q_03*(-q_s_03 + q_s_ref_03 + xi_q_03/T_q_03) - e_qv_03
        struct[0].g[17,0] = -R_v_03*i_d_03 - V_03*sin(delta_03 - theta_03) + X_v_03*i_q_03
        struct[0].g[18,0] = -R_v_03*i_q_03 - V_03*cos(delta_03 - theta_03) - X_v_03*i_d_03 + e_qv_03
        struct[0].g[19,0] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03) - p_s_03
        struct[0].g[20,0] = V_03*i_d_03*cos(delta_03 - theta_03) - V_03*i_q_03*sin(delta_03 - theta_03) - q_s_03
        struct[0].g[21,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)])) + p_ghr_03 - p_m_03 + p_s_03 - p_t_03 + p_u_03
        struct[0].g[22,0] = i_d_03*(R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)) + i_q_03*(R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)) - p_t_03
        struct[0].g[23,0] = -p_u_03 - k_u_03*(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].g[24,0] = R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1) + e_u_03 - v_u_03
        struct[0].g[25,0] = -k_u_03 + Piecewise(np.array([(K_u_max_03, V_u_min_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (K_u_0_03 + (-K_u_0_03 + K_u_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (K_u_max_03, V_u_max_03 < v_u_03), (K_u_0_03, True)]))
        struct[0].g[26,0] = -k_cur_sat_03 + Piecewise(np.array([(0.0001, k_cur_03 < 0.0001), (1, k_cur_03 > 1), (k_cur_03, True)]))
        struct[0].g[27,0] = k_cur_sat_03*(inc_p_gin_03 + p_gin_0_03) - p_gou_03
        struct[0].g[28,0] = -p_f_03 - Piecewise(np.array([((0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03)), ((-0.5*DB_03 + K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03) - omega_ref_03)/Droop_03, 0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)), (0.0, True)]))
        struct[0].g[29,0] = -r_lim_03 + Piecewise(np.array([(R_lim_max_03, (omega_03 > 0.5*DB_03 + omega_ref_03) | (omega_03 < -0.5*DB_03 + omega_ref_03)), (0.0, True)])) + Piecewise(np.array([(R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_lt_03 + v_u_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), (R_lim_03 + (-R_lim_03 + R_lim_max_03)*(-V_u_ht_03 + v_u_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (R_lim_03, True)]))
        struct[0].g[30,0] = -omega_pll_03 + (K_i_pll_03*xi_pll_03 + K_p_pll_03*(V_03*sin(theta_03)*cos(theta_pll_03) - V_03*sin(theta_pll_03)*cos(theta_03)))/Omega_b_03
        struct[0].g[31,0] = -omega_coi + (H_01*S_n_01*omega_01 + S_n_03*T_p_03*omega_03/(2*K_p_03))/(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03))
        struct[0].g[32,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = i_d_01*(R_a_01*i_d_01 + V_01*sin(delta_01 - theta_01)) + i_q_01*(R_a_01*i_q_01 + V_01*cos(delta_01 - theta_01))
        struct[0].h[4,0] = inc_p_gin_03 + p_gin_0_03
        struct[0].h[5,0] = p_g_ref_03
        struct[0].h[6,0] = -p_s_03 + p_t_03
        struct[0].h[7,0] = (-V_u_min_03**2 + e_u_03**2)/(V_u_max_03**2 - V_u_min_03**2)
        struct[0].h[8,0] = K_fpfr_03*Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
        struct[0].h[9,0] = Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_01
        struct[0].Fx[0,1] = Omega_b_01
        struct[0].Fx[1,0] = (-V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fx[1,1] = -D_01/(2*H_01)
        struct[0].Fx[2,2] = -1/T1d0_01
        struct[0].Fx[3,3] = -1/T1q0_01
        struct[0].Fx[4,4] = -1/T_r_01
        struct[0].Fx[5,4] = K_a_01*K_aw_01 - 1
        struct[0].Fx[5,5] = -K_ai_01*K_aw_01
        struct[0].Fx[6,6] = -1/T_gov_1_01
        struct[0].Fx[7,6] = 1/T_gov_3_01
        struct[0].Fx[7,7] = -1/T_gov_3_01
        struct[0].Fx[8,8] = -0.00000100000000000000
        struct[0].Fx[9,1] = 1/T_wo_01
        struct[0].Fx[9,9] = -1/T_wo_01
        struct[0].Fx[10,10] = -1/T_2_01
        struct[0].Fx[15,15] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (-K_h_03, True)]))
        struct[0].Fx[16,16] = -1/T_cur_03
        struct[0].Fx[16,17] = (-p_g_ref_03/(inc_p_gin_03 + p_gin_0_03)**2 - Piecewise(np.array([(P_f_min_03, P_f_min_03 > p_f_03), (P_f_max_03, P_f_max_03 < p_f_03), (p_f_03, True)]))/(inc_p_gin_03 + p_gin_0_03)**2)/T_cur_03
        struct[0].Fx[17,17] = -0.00100000000000000
        struct[0].Fx[18,18] = K_p_pll_03*(-V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03))
        struct[0].Fx[18,19] = K_i_pll_03
        struct[0].Fx[19,18] = -V_03*sin(theta_03)*sin(theta_pll_03) - V_03*cos(theta_03)*cos(theta_pll_03)

    if mode == 11:

        struct[0].Fy[0,31] = -Omega_b_01
        struct[0].Fy[1,0] = (-i_d_01*sin(delta_01 - theta_01) - i_q_01*cos(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,1] = (V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,6] = (-2*R_a_01*i_d_01 - V_01*sin(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,7] = (-2*R_a_01*i_q_01 - V_01*cos(delta_01 - theta_01))/(2*H_01)
        struct[0].Fy[1,12] = 1/(2*H_01)
        struct[0].Fy[1,31] = D_01/(2*H_01)
        struct[0].Fy[2,6] = (X1d_01 - X_d_01)/T1d0_01
        struct[0].Fy[2,10] = 1/T1d0_01
        struct[0].Fy[3,7] = (-X1q_01 + X_q_01)/T1q0_01
        struct[0].Fy[4,0] = 1/T_r_01
        struct[0].Fy[5,10] = K_aw_01
        struct[0].Fy[5,14] = -K_a_01*K_aw_01 + 1
        struct[0].Fy[6,11] = 1/T_gov_1_01
        struct[0].Fy[8,8] = -K_imw_01
        struct[0].Fy[10,13] = 1/T_2_01
        struct[0].Fy[11,15] = Omega_b_03
        struct[0].Fy[11,31] = -Omega_b_03
        struct[0].Fy[12,19] = -1
        struct[0].Fy[12,21] = 1
        struct[0].Fy[13,20] = -1
        struct[0].Fy[14,22] = -S_n_03/(C_u_03*(v_u_03 + 0.1))
        struct[0].Fy[14,24] = -S_n_03*(p_gou_03 - p_t_03)/(C_u_03*(v_u_03 + 0.1)**2)
        struct[0].Fy[14,27] = S_n_03/(C_u_03*(v_u_03 + 0.1))
        struct[0].Fy[15,27] = Piecewise(np.array([(0, (r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)) | (r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03))), (K_h_03, True)]))
        struct[0].Fy[15,29] = Piecewise(np.array([(-1, r_lim_03 < -K_h_03*(-p_ghr_03 + p_gou_03)), (1, r_lim_03 < K_h_03*(-p_ghr_03 + p_gou_03)), (0, True)]))
        struct[0].Fy[16,28] = Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))/(T_cur_03*(inc_p_gin_03 + p_gin_0_03))
        struct[0].Fy[18,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03))
        struct[0].Fy[18,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03))
        struct[0].Fy[18,31] = -Omega_b_03
        struct[0].Fy[19,4] = sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03)
        struct[0].Fy[19,5] = V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03)
        struct[0].Fy[20,31] = -1

        struct[0].Gy[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,8] = -S_n_01/S_base
        struct[0].Gy[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,9] = -S_n_01/S_base
        struct[0].Gy[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*g_02_03
        struct[0].Gy[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[4,19] = -S_n_03/S_base
        struct[0].Gy[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - bs_02_03/2)
        struct[0].Gy[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[5,20] = -S_n_03/S_base
        struct[0].Gy[6,0] = cos(delta_01 - theta_01)
        struct[0].Gy[6,1] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy[6,6] = X1d_01
        struct[0].Gy[6,7] = R_a_01
        struct[0].Gy[7,0] = sin(delta_01 - theta_01)
        struct[0].Gy[7,1] = -V_01*cos(delta_01 - theta_01)
        struct[0].Gy[7,6] = R_a_01
        struct[0].Gy[7,7] = -X1q_01
        struct[0].Gy[8,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[8,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[8,6] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy[8,7] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[8,8] = -1
        struct[0].Gy[9,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[9,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[9,6] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[9,7] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy[9,9] = -1
        struct[0].Gy[10,10] = -1
        struct[0].Gy[10,14] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gy[11,11] = -1
        struct[0].Gy[11,32] = K_sec_01
        struct[0].Gy[12,12] = -1
        struct[0].Gy[13,13] = -1
        struct[0].Gy[14,13] = Piecewise(np.array([(0, (V_lim_01 < K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01)) | (V_lim_01 < -K_stab_01*(T_1_01*(-x_lead_01 + z_wo_01)/T_2_01 + x_lead_01))), (K_stab_01*T_1_01/T_2_01, True)]))
        struct[0].Gy[14,14] = -1
        struct[0].Gy[15,15] = -1
        struct[0].Gy[15,19] = -K_p_03
        struct[0].Gy[15,21] = K_p_03
        struct[0].Gy[16,16] = -1
        struct[0].Gy[16,20] = -K_q_03
        struct[0].Gy[17,4] = -sin(delta_03 - theta_03)
        struct[0].Gy[17,5] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy[17,17] = -R_v_03
        struct[0].Gy[17,18] = X_v_03
        struct[0].Gy[18,4] = -cos(delta_03 - theta_03)
        struct[0].Gy[18,5] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy[18,16] = 1
        struct[0].Gy[18,17] = -X_v_03
        struct[0].Gy[18,18] = -R_v_03
        struct[0].Gy[19,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy[19,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy[19,17] = V_03*sin(delta_03 - theta_03)
        struct[0].Gy[19,18] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy[19,19] = -1
        struct[0].Gy[20,4] = i_d_03*cos(delta_03 - theta_03) - i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy[20,5] = V_03*i_d_03*sin(delta_03 - theta_03) + V_03*i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy[20,17] = V_03*cos(delta_03 - theta_03)
        struct[0].Gy[20,18] = -V_03*sin(delta_03 - theta_03)
        struct[0].Gy[20,20] = -1
        struct[0].Gy[21,19] = 1
        struct[0].Gy[21,21] = -1
        struct[0].Gy[21,22] = -1
        struct[0].Gy[21,23] = 1
        struct[0].Gy[21,28] = K_fpfr_03*Piecewise(np.array([(0, (P_f_min_03 > p_f_03) | (P_f_max_03 < p_f_03)), (1, True)]))
        struct[0].Gy[22,4] = i_d_03*sin(delta_03 - theta_03) + i_q_03*cos(delta_03 - theta_03)
        struct[0].Gy[22,5] = -V_03*i_d_03*cos(delta_03 - theta_03) + V_03*i_q_03*sin(delta_03 - theta_03)
        struct[0].Gy[22,17] = 2*R_s_03*i_d_03 + V_03*sin(delta_03 - theta_03)
        struct[0].Gy[22,18] = 2*R_s_03*i_q_03 + V_03*cos(delta_03 - theta_03)
        struct[0].Gy[22,22] = -1
        struct[0].Gy[23,23] = -1
        struct[0].Gy[23,24] = 2*k_u_03*v_u_03/V_u_max_03**2
        struct[0].Gy[23,25] = -(-v_u_03**2 + v_u_ref_03**2)/V_u_max_03**2
        struct[0].Gy[24,22] = -R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy[24,24] = -R_uc_03*S_n_03*(p_gou_03 - p_t_03)/(v_u_03 + 0.1)**2 - 1
        struct[0].Gy[24,27] = R_uc_03*S_n_03/(v_u_03 + 0.1)
        struct[0].Gy[25,24] = Piecewise(np.array([(0, V_u_min_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-K_u_0_03 + K_u_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy[25,25] = -1
        struct[0].Gy[26,26] = -1
        struct[0].Gy[27,26] = inc_p_gin_03 + p_gin_0_03
        struct[0].Gy[27,27] = -1
        struct[0].Gy[28,15] = -Piecewise(np.array([((1 - K_speed_03)/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy[28,28] = -1
        struct[0].Gy[28,30] = -Piecewise(np.array([(K_speed_03/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))
        struct[0].Gy[29,24] = Piecewise(np.array([((-R_lim_03 + R_lim_max_03)/(-V_u_lt_03 + V_u_min_03), V_u_lt_03 > v_u_03), ((-R_lim_03 + R_lim_max_03)/(-V_u_ht_03 + V_u_max_03), V_u_ht_03 < v_u_03), (0, True)]))
        struct[0].Gy[29,29] = -1
        struct[0].Gy[30,4] = K_p_pll_03*(sin(theta_03)*cos(theta_pll_03) - sin(theta_pll_03)*cos(theta_03))/Omega_b_03
        struct[0].Gy[30,5] = K_p_pll_03*(V_03*sin(theta_03)*sin(theta_pll_03) + V_03*cos(theta_03)*cos(theta_pll_03))/Omega_b_03
        struct[0].Gy[30,30] = -1
        struct[0].Gy[31,15] = S_n_03*T_p_03/(2*K_p_03*(H_01*S_n_01 + S_n_03*T_p_03/(2*K_p_03)))
        struct[0].Gy[31,31] = -1
        struct[0].Gy[32,31] = -K_p_agc
        struct[0].Gy[32,32] = -1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[10,6] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gu[10,7] = Piecewise(np.array([(0, (V_min_01 > K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01) | (V_max_01 < K_a_01*(-v_c_01 + v_pss_01 + v_ref_01) + K_ai_01*xi_v_01)), (K_a_01, True)]))
        struct[0].Gu[11,9] = 1
        struct[0].Gu[14,7] = -1
        struct[0].Gu[16,10] = K_q_03
        struct[0].Gu[23,11] = -2*k_u_03*v_u_ref_03/V_u_max_03**2
        struct[0].Gu[27,13] = k_cur_sat_03
        struct[0].Gu[28,12] = -Piecewise(np.array([(-1/Droop_03, (0.5*DB_03 + omega_ref_03 < K_speed_03*omega_pll_03 + omega_03*(1 - K_speed_03)) | (0.5*DB_03 - omega_ref_03 < -K_speed_03*omega_pll_03 - omega_03*(1 - K_speed_03))), (0, True)]))





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
    Fx_ini_rows = [0, 0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 15, 16, 16, 17, 18, 18, 19]

    Fx_ini_cols = [0, 1, 0, 1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 1, 9, 10, 15, 16, 17, 17, 18, 19, 18]

    Fy_ini_rows = [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 5, 6, 8, 10, 11, 11, 12, 12, 13, 14, 14, 14, 15, 15, 16, 18, 18, 18, 19, 19, 20]

    Fy_ini_cols = [31, 0, 1, 6, 7, 12, 31, 6, 10, 7, 0, 10, 14, 11, 8, 13, 15, 31, 19, 21, 20, 22, 24, 27, 27, 29, 28, 4, 5, 31, 4, 5, 31]

    Gx_ini_rows = [6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 30, 30, 31, 32]

    Gx_ini_cols = [0, 2, 0, 3, 0, 0, 4, 5, 1, 8, 6, 7, 1, 9, 10, 12, 13, 11, 11, 11, 11, 15, 11, 14, 16, 17, 18, 19, 1, 20]

    Gy_ini_rows = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 26, 27, 27, 28, 28, 28, 29, 29, 30, 30, 30, 31, 31, 32, 32]

    Gy_ini_cols = [0, 1, 2, 3, 8, 0, 1, 2, 3, 9, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 19, 2, 3, 4, 5, 20, 0, 1, 6, 7, 0, 1, 6, 7, 0, 1, 6, 7, 8, 0, 1, 6, 7, 9, 10, 14, 11, 32, 12, 13, 13, 14, 15, 19, 21, 16, 20, 4, 5, 17, 18, 4, 5, 16, 17, 18, 4, 5, 17, 18, 19, 4, 5, 17, 18, 20, 19, 21, 22, 23, 28, 4, 5, 17, 18, 22, 23, 24, 25, 22, 24, 27, 24, 25, 26, 26, 27, 15, 28, 30, 24, 29, 4, 5, 30, 15, 31, 31, 32]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols