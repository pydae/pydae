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


class milano_8p1_vsg_1_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 27
        self.N_y = 31 
        self.N_z = 7 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_1_2', 'b_1_2', 'bs_1_2', 'g_2_3', 'b_2_3', 'bs_2_3', 'U_1_n', 'U_2_n', 'U_3_n', 'S_n_2', 'H_2', 'Omega_b_2', 'T1d0_2', 'T1q0_2', 'X_d_2', 'X_q_2', 'X1d_2', 'X1q_2', 'D_2', 'R_a_2', 'K_delta_2', 'K_a_2', 'K_ai_2', 'T_r_2', 'Droop_2', 'T_gov_1_2', 'T_gov_2_2', 'T_gov_3_2', 'K_imw_2', 'omega_ref_2', 'S_n_3', 'H_3', 'Omega_b_3', 'T1d0_3', 'T1q0_3', 'X_d_3', 'X_q_3', 'X1d_3', 'X1q_3', 'D_3', 'R_a_3', 'K_delta_3', 'K_a_3', 'K_ai_3', 'T_r_3', 'Droop_3', 'T_gov_1_3', 'T_gov_2_3', 'T_gov_3_3', 'K_imw_3', 'omega_ref_3', 'K_sec_2', 'K_sec_3', 'S_n_1', 'R_s_1', 'H_1', 'Omega_b_1', 'R_v_1', 'X_v_1', 'D1_1', 'D2_1', 'D3_1', 'K_delta_1', 'T_wo_1', 'T_i_1', 'K_q_1', 'T_q_1', 'H_s_1', 'K_p_soc_1', 'K_i_soc_1'] 
        self.params_values_list  = [100000000.0, 0.07486570963334518, -0.7486570963334518, 8.4e-05, 7.486570963334518, -74.86570963334518, 8.4e-07, 20000.0, 20000.0, 20000.0, 900000000.0, 5.0, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.01, 0.01, 100, 1e-06, 0.02, 0.05, 1.0, 2.0, 10.0, 0.0, 1.0, 900000000.0, 5.0, 314.1592653589793, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.01, 100, 1e-06, 0.02, 0.05, 1.0, 2.0, 10.0, 0.0, 1.0, 0.01, 0.01, 1000000.0, 0.01, 5.0, 314.1592653589793, 0.01, 0.1, 1.0, 0.0, 0.0, 0.01, 10.0, 0.01, 0.1, 0.1, 100.0, 1.0, 0.01] 
        self.inputs_ini_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'v_ref_2', 'v_pss_2', 'p_c_2', 'v_ref_3', 'v_pss_3', 'p_c_3', 'p_in_1', 'Dp_ref_1', 'q_ref_1', 'p_src_1', 'soc_ref_1'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.7793, 1.0, 0.0, 0.7793, 0.0, 0.0, 0.0, 0.8, 0.5] 
        self.inputs_run_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'v_ref_2', 'v_pss_2', 'p_c_2', 'v_ref_3', 'v_pss_3', 'p_c_3', 'p_in_1', 'Dp_ref_1', 'q_ref_1', 'p_src_1', 'soc_ref_1'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.7793, 1.0, 0.0, 0.7793, 0.0, 0.0, 0.0, 0.8, 0.5] 
        self.outputs_list = ['V_1', 'V_2', 'V_3', 'p_e_2', 'p_e_3', 'p_t_1', 'p_soc_1'] 
        self.x_list = ['delta_2', 'omega_2', 'e1q_2', 'e1d_2', 'v_c_2', 'xi_v_2', 'x_gov_1_2', 'x_gov_2_2', 'xi_imw_2', 'delta_3', 'omega_3', 'e1q_3', 'e1d_3', 'v_c_3', 'xi_v_3', 'x_gov_1_3', 'x_gov_2_3', 'xi_imw_3', 'xi_freq', 'delta_1', 'omega_v_1', 'x_wo_1', 'i_d_1', 'i_q_1', 'xi_q_1', 'soc_1', 'xi_soc_1'] 
        self.y_run_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'i_d_2', 'i_q_2', 'p_g_2_1', 'q_g_2_1', 'v_f_2', 'p_m_ref_2', 'p_m_2', 'i_d_3', 'i_q_3', 'p_g_3_1', 'q_g_3_1', 'v_f_3', 'p_m_ref_3', 'p_m_3', 'p_r_2', 'p_r_3', 'i_d_ref_1', 'i_q_ref_1', 'p_g_1_1', 'q_g_1_1', 'p_d2_1', 'e_v_1', 'p_sto_1', 'p_m_1', 'omega_coi'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'i_d_2', 'i_q_2', 'p_g_2_1', 'q_g_2_1', 'v_f_2', 'p_m_ref_2', 'p_m_2', 'i_d_3', 'i_q_3', 'p_g_3_1', 'q_g_3_1', 'v_f_3', 'p_m_ref_3', 'p_m_3', 'p_r_2', 'p_r_3', 'i_d_ref_1', 'i_q_ref_1', 'p_g_1_1', 'q_g_1_1', 'p_d2_1', 'e_v_1', 'p_sto_1', 'p_m_1', 'omega_coi'] 
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
    g_2_3 = struct[0].g_2_3
    b_2_3 = struct[0].b_2_3
    bs_2_3 = struct[0].bs_2_3
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
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
    T_gov_1_2 = struct[0].T_gov_1_2
    T_gov_2_2 = struct[0].T_gov_2_2
    T_gov_3_2 = struct[0].T_gov_3_2
    K_imw_2 = struct[0].K_imw_2
    omega_ref_2 = struct[0].omega_ref_2
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
    T_gov_1_3 = struct[0].T_gov_1_3
    T_gov_2_3 = struct[0].T_gov_2_3
    T_gov_3_3 = struct[0].T_gov_3_3
    K_imw_3 = struct[0].K_imw_3
    omega_ref_3 = struct[0].omega_ref_3
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    S_n_1 = struct[0].S_n_1
    R_s_1 = struct[0].R_s_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    R_v_1 = struct[0].R_v_1
    X_v_1 = struct[0].X_v_1
    D1_1 = struct[0].D1_1
    D2_1 = struct[0].D2_1
    D3_1 = struct[0].D3_1
    K_delta_1 = struct[0].K_delta_1
    T_wo_1 = struct[0].T_wo_1
    T_i_1 = struct[0].T_i_1
    K_q_1 = struct[0].K_q_1
    T_q_1 = struct[0].T_q_1
    H_s_1 = struct[0].H_s_1
    K_p_soc_1 = struct[0].K_p_soc_1
    K_i_soc_1 = struct[0].K_i_soc_1
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    p_in_1 = struct[0].p_in_1
    Dp_ref_1 = struct[0].Dp_ref_1
    q_ref_1 = struct[0].q_ref_1
    p_src_1 = struct[0].p_src_1
    soc_ref_1 = struct[0].soc_ref_1
    
    # Dynamical states:
    delta_2 = struct[0].x[0,0]
    omega_2 = struct[0].x[1,0]
    e1q_2 = struct[0].x[2,0]
    e1d_2 = struct[0].x[3,0]
    v_c_2 = struct[0].x[4,0]
    xi_v_2 = struct[0].x[5,0]
    x_gov_1_2 = struct[0].x[6,0]
    x_gov_2_2 = struct[0].x[7,0]
    xi_imw_2 = struct[0].x[8,0]
    delta_3 = struct[0].x[9,0]
    omega_3 = struct[0].x[10,0]
    e1q_3 = struct[0].x[11,0]
    e1d_3 = struct[0].x[12,0]
    v_c_3 = struct[0].x[13,0]
    xi_v_3 = struct[0].x[14,0]
    x_gov_1_3 = struct[0].x[15,0]
    x_gov_2_3 = struct[0].x[16,0]
    xi_imw_3 = struct[0].x[17,0]
    xi_freq = struct[0].x[18,0]
    delta_1 = struct[0].x[19,0]
    omega_v_1 = struct[0].x[20,0]
    x_wo_1 = struct[0].x[21,0]
    i_d_1 = struct[0].x[22,0]
    i_q_1 = struct[0].x[23,0]
    xi_q_1 = struct[0].x[24,0]
    soc_1 = struct[0].x[25,0]
    xi_soc_1 = struct[0].x[26,0]
    
    # Algebraic states:
    V_1 = struct[0].y_ini[0,0]
    theta_1 = struct[0].y_ini[1,0]
    V_2 = struct[0].y_ini[2,0]
    theta_2 = struct[0].y_ini[3,0]
    V_3 = struct[0].y_ini[4,0]
    theta_3 = struct[0].y_ini[5,0]
    i_d_2 = struct[0].y_ini[6,0]
    i_q_2 = struct[0].y_ini[7,0]
    p_g_2_1 = struct[0].y_ini[8,0]
    q_g_2_1 = struct[0].y_ini[9,0]
    v_f_2 = struct[0].y_ini[10,0]
    p_m_ref_2 = struct[0].y_ini[11,0]
    p_m_2 = struct[0].y_ini[12,0]
    i_d_3 = struct[0].y_ini[13,0]
    i_q_3 = struct[0].y_ini[14,0]
    p_g_3_1 = struct[0].y_ini[15,0]
    q_g_3_1 = struct[0].y_ini[16,0]
    v_f_3 = struct[0].y_ini[17,0]
    p_m_ref_3 = struct[0].y_ini[18,0]
    p_m_3 = struct[0].y_ini[19,0]
    p_r_2 = struct[0].y_ini[20,0]
    p_r_3 = struct[0].y_ini[21,0]
    i_d_ref_1 = struct[0].y_ini[22,0]
    i_q_ref_1 = struct[0].y_ini[23,0]
    p_g_1_1 = struct[0].y_ini[24,0]
    q_g_1_1 = struct[0].y_ini[25,0]
    p_d2_1 = struct[0].y_ini[26,0]
    e_v_1 = struct[0].y_ini[27,0]
    p_sto_1 = struct[0].y_ini[28,0]
    p_m_1 = struct[0].y_ini[29,0]
    omega_coi = struct[0].y_ini[30,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[1,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[2,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[3,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[4,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[5,0] = -V_2 + v_ref_2
        struct[0].f[6,0] = (p_m_ref_2 - x_gov_1_2)/T_gov_1_2
        struct[0].f[7,0] = (x_gov_1_2 - x_gov_2_2)/T_gov_3_2
        struct[0].f[8,0] = K_imw_2*(p_c_2 - p_g_2_1) - 1.0e-6*xi_imw_2
        struct[0].f[9,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[10,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[11,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[12,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[13,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[14,0] = -V_3 + v_ref_3
        struct[0].f[15,0] = (p_m_ref_3 - x_gov_1_3)/T_gov_1_3
        struct[0].f[16,0] = (x_gov_1_3 - x_gov_2_3)/T_gov_3_3
        struct[0].f[17,0] = K_imw_3*(p_c_3 - p_g_3_1) - 1.0e-6*xi_imw_3
        struct[0].f[18,0] = 1 - omega_coi
        struct[0].f[19,0] = D3_1*(-p_g_1_1 + p_m_1) - K_delta_1*delta_1 + Omega_b_1*(-omega_coi + omega_v_1)
        struct[0].f[20,0] = (-D1_1*(omega_v_1 - 1) - p_d2_1 - p_g_1_1 + p_m_1)/(2*H_1)
        struct[0].f[21,0] = (omega_v_1 - x_wo_1 - 1.0)/T_wo_1
        struct[0].f[22,0] = (-i_d_1 + i_d_ref_1)/T_i_1
        struct[0].f[23,0] = (-i_q_1 + i_q_ref_1)/T_i_1
        struct[0].f[24,0] = -q_g_1_1 + q_ref_1
        struct[0].f[25,0] = -p_sto_1/H_s_1
        struct[0].f[26,0] = -soc_1 + soc_ref_1
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_2 + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - bs_1_2/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_3) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + V_3**2*g_2_3 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + V_3**2*(-b_2_3 - bs_2_3/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[7,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[8,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[9,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[10,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[11,0] = p_c_2 - p_m_ref_2 + p_r_2 + xi_imw_2 - (omega_2 - omega_ref_2)/Droop_2
        struct[0].g[12,0] = T_gov_2_2*(x_gov_1_2 - x_gov_2_2)/T_gov_3_2 - p_m_2 + x_gov_2_2
        struct[0].g[13,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[14,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[15,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[16,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[17,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[18,0] = p_c_3 - p_m_ref_3 + p_r_3 + xi_imw_3 - (omega_3 - omega_ref_3)/Droop_3
        struct[0].g[19,0] = T_gov_2_3*(x_gov_1_3 - x_gov_2_3)/T_gov_3_3 - p_m_3 + x_gov_2_3
        struct[0].g[20,0] = K_sec_2*xi_freq/2 - p_r_2
        struct[0].g[21,0] = K_sec_3*xi_freq/2 - p_r_3
        struct[0].g[22,0] = R_v_1*i_q_ref_1 + V_1*cos(delta_1 - theta_1) + X_v_1*i_d_ref_1 - e_v_1
        struct[0].g[23,0] = R_v_1*i_d_ref_1 + V_1*sin(delta_1 - theta_1) - X_v_1*i_q_ref_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = D2_1*(omega_v_1 - x_wo_1 - 1.0) - p_d2_1
        struct[0].g[27,0] = K_q_1*(-q_g_1_1 + q_ref_1 + xi_q_1/T_q_1) - e_v_1
        struct[0].g[28,0] = -i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_src_1 + p_sto_1
        struct[0].g[29,0] = Dp_ref_1 - p_m_1 + p_src_1 + Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
        struct[0].g[30,0] = omega_2/2 + omega_3/2 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) + i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2))
        struct[0].h[4,0] = i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) + i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3))
        struct[0].h[5,0] = i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) + i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1))
        struct[0].h[6,0] = Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_2
        struct[0].Fx_ini[0,1] = Omega_b_2
        struct[0].Fx_ini[1,0] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx_ini[1,1] = -D_2/(2*H_2)
        struct[0].Fx_ini[2,2] = -1/T1d0_2
        struct[0].Fx_ini[3,3] = -1/T1q0_2
        struct[0].Fx_ini[4,4] = -1/T_r_2
        struct[0].Fx_ini[6,6] = -1/T_gov_1_2
        struct[0].Fx_ini[7,6] = 1/T_gov_3_2
        struct[0].Fx_ini[7,7] = -1/T_gov_3_2
        struct[0].Fx_ini[9,9] = -K_delta_3
        struct[0].Fx_ini[9,10] = Omega_b_3
        struct[0].Fx_ini[10,9] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx_ini[10,10] = -D_3/(2*H_3)
        struct[0].Fx_ini[11,11] = -1/T1d0_3
        struct[0].Fx_ini[12,12] = -1/T1q0_3
        struct[0].Fx_ini[13,13] = -1/T_r_3
        struct[0].Fx_ini[15,15] = -1/T_gov_1_3
        struct[0].Fx_ini[16,15] = 1/T_gov_3_3
        struct[0].Fx_ini[16,16] = -1/T_gov_3_3
        struct[0].Fx_ini[19,19] = -K_delta_1
        struct[0].Fx_ini[19,20] = Omega_b_1
        struct[0].Fx_ini[20,20] = -D1_1/(2*H_1)
        struct[0].Fx_ini[21,20] = 1/T_wo_1
        struct[0].Fx_ini[21,21] = -1/T_wo_1
        struct[0].Fx_ini[22,22] = -1/T_i_1
        struct[0].Fx_ini[23,23] = -1/T_i_1

    if mode == 11:

        struct[0].Fy_ini[0,30] = -Omega_b_2 
        struct[0].Fy_ini[1,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,6] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,7] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,12] = 1/(2*H_2) 
        struct[0].Fy_ini[1,30] = D_2/(2*H_2) 
        struct[0].Fy_ini[2,6] = (X1d_2 - X_d_2)/T1d0_2 
        struct[0].Fy_ini[2,10] = 1/T1d0_2 
        struct[0].Fy_ini[3,7] = (-X1q_2 + X_q_2)/T1q0_2 
        struct[0].Fy_ini[4,2] = 1/T_r_2 
        struct[0].Fy_ini[5,2] = -1 
        struct[0].Fy_ini[6,11] = 1/T_gov_1_2 
        struct[0].Fy_ini[8,8] = -K_imw_2 
        struct[0].Fy_ini[9,30] = -Omega_b_3 
        struct[0].Fy_ini[10,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,13] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,14] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,19] = 1/(2*H_3) 
        struct[0].Fy_ini[10,30] = D_3/(2*H_3) 
        struct[0].Fy_ini[11,13] = (X1d_3 - X_d_3)/T1d0_3 
        struct[0].Fy_ini[11,17] = 1/T1d0_3 
        struct[0].Fy_ini[12,14] = (-X1q_3 + X_q_3)/T1q0_3 
        struct[0].Fy_ini[13,4] = 1/T_r_3 
        struct[0].Fy_ini[14,4] = -1 
        struct[0].Fy_ini[15,18] = 1/T_gov_1_3 
        struct[0].Fy_ini[17,15] = -K_imw_3 
        struct[0].Fy_ini[18,30] = -1 
        struct[0].Fy_ini[19,24] = -D3_1 
        struct[0].Fy_ini[19,29] = D3_1 
        struct[0].Fy_ini[19,30] = -Omega_b_1 
        struct[0].Fy_ini[20,24] = -1/(2*H_1) 
        struct[0].Fy_ini[20,26] = -1/(2*H_1) 
        struct[0].Fy_ini[20,29] = 1/(2*H_1) 
        struct[0].Fy_ini[22,22] = 1/T_i_1 
        struct[0].Fy_ini[23,23] = 1/T_i_1 
        struct[0].Fy_ini[24,25] = -1 
        struct[0].Fy_ini[25,28] = -1/H_s_1 

        struct[0].Gx_ini[6,0] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gx_ini[6,2] = -1
        struct[0].Gx_ini[7,0] = V_2*cos(delta_2 - theta_2)
        struct[0].Gx_ini[7,3] = -1
        struct[0].Gx_ini[8,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gx_ini[9,0] = -V_2*i_d_2*sin(delta_2 - theta_2) - V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gx_ini[10,4] = -K_a_2
        struct[0].Gx_ini[10,5] = K_ai_2
        struct[0].Gx_ini[11,1] = -1/Droop_2
        struct[0].Gx_ini[11,8] = 1
        struct[0].Gx_ini[12,6] = T_gov_2_2/T_gov_3_2
        struct[0].Gx_ini[12,7] = -T_gov_2_2/T_gov_3_2 + 1
        struct[0].Gx_ini[13,9] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gx_ini[13,11] = -1
        struct[0].Gx_ini[14,9] = V_3*cos(delta_3 - theta_3)
        struct[0].Gx_ini[14,12] = -1
        struct[0].Gx_ini[15,9] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gx_ini[16,9] = -V_3*i_d_3*sin(delta_3 - theta_3) - V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gx_ini[17,13] = -K_a_3
        struct[0].Gx_ini[17,14] = K_ai_3
        struct[0].Gx_ini[18,10] = -1/Droop_3
        struct[0].Gx_ini[18,17] = 1
        struct[0].Gx_ini[19,15] = T_gov_2_3/T_gov_3_3
        struct[0].Gx_ini[19,16] = -T_gov_2_3/T_gov_3_3 + 1
        struct[0].Gx_ini[20,18] = K_sec_2/2
        struct[0].Gx_ini[21,18] = K_sec_3/2
        struct[0].Gx_ini[22,19] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[23,19] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx_ini[24,19] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[24,22] = V_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[24,23] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx_ini[25,19] = -V_1*i_d_1*sin(delta_1 - theta_1) - V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gx_ini[25,22] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx_ini[25,23] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[26,20] = D2_1
        struct[0].Gx_ini[26,21] = -D2_1
        struct[0].Gx_ini[27,24] = K_q_1/T_q_1
        struct[0].Gx_ini[28,19] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[28,22] = -2*R_s_1*i_d_1 - V_1*sin(delta_1 - theta_1)
        struct[0].Gx_ini[28,23] = -2*R_s_1*i_q_1 - V_1*cos(delta_1 - theta_1)
        struct[0].Gx_ini[29,25] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (-K_p_soc_1, True)]))
        struct[0].Gx_ini[29,26] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1, True)]))
        struct[0].Gx_ini[30,1] = 1/2
        struct[0].Gx_ini[30,10] = 1/2

        struct[0].Gy_ini[0,0] = 2*V_1*g_1_2 + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[0,24] = -S_n_1/S_base
        struct[0].Gy_ini[1,0] = 2*V_1*(-b_1_2 - bs_1_2/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[1,25] = -S_n_1/S_base
        struct[0].Gy_ini[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_3) + V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[2,4] = V_2*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[2,5] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[2,8] = -S_n_2/S_base
        struct[0].Gy_ini[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[3,4] = V_2*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[3,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[3,9] = -S_n_2/S_base
        struct[0].Gy_ini[4,2] = V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[4,3] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[4,4] = V_2*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + 2*V_3*g_2_3
        struct[0].Gy_ini[4,5] = V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[4,15] = -S_n_3/S_base
        struct[0].Gy_ini[5,2] = V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[5,3] = V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[5,4] = V_2*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + 2*V_3*(-b_2_3 - bs_2_3/2)
        struct[0].Gy_ini[5,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[5,16] = -S_n_3/S_base
        struct[0].Gy_ini[6,2] = cos(delta_2 - theta_2)
        struct[0].Gy_ini[6,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[6,6] = X1d_2
        struct[0].Gy_ini[6,7] = R_a_2
        struct[0].Gy_ini[7,2] = sin(delta_2 - theta_2)
        struct[0].Gy_ini[7,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[7,6] = R_a_2
        struct[0].Gy_ini[7,7] = -X1q_2
        struct[0].Gy_ini[8,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[8,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[8,6] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[8,7] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[9,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[9,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[9,6] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[9,7] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[13,4] = cos(delta_3 - theta_3)
        struct[0].Gy_ini[13,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[13,13] = X1d_3
        struct[0].Gy_ini[13,14] = R_a_3
        struct[0].Gy_ini[14,4] = sin(delta_3 - theta_3)
        struct[0].Gy_ini[14,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[14,13] = R_a_3
        struct[0].Gy_ini[14,14] = -X1q_3
        struct[0].Gy_ini[15,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[15,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[15,13] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[15,14] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[16,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[16,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[16,13] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[16,14] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy_ini[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[22,22] = X_v_1
        struct[0].Gy_ini[22,23] = R_v_1
        struct[0].Gy_ini[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy_ini[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[23,22] = R_v_1
        struct[0].Gy_ini[23,23] = -X_v_1
        struct[0].Gy_ini[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[27,25] = -K_q_1
        struct[0].Gy_ini[28,0] = -i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[28,1] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_2 = struct[0].g_1_2
    b_1_2 = struct[0].b_1_2
    bs_1_2 = struct[0].bs_1_2
    g_2_3 = struct[0].g_2_3
    b_2_3 = struct[0].b_2_3
    bs_2_3 = struct[0].bs_2_3
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
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
    T_gov_1_2 = struct[0].T_gov_1_2
    T_gov_2_2 = struct[0].T_gov_2_2
    T_gov_3_2 = struct[0].T_gov_3_2
    K_imw_2 = struct[0].K_imw_2
    omega_ref_2 = struct[0].omega_ref_2
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
    T_gov_1_3 = struct[0].T_gov_1_3
    T_gov_2_3 = struct[0].T_gov_2_3
    T_gov_3_3 = struct[0].T_gov_3_3
    K_imw_3 = struct[0].K_imw_3
    omega_ref_3 = struct[0].omega_ref_3
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    S_n_1 = struct[0].S_n_1
    R_s_1 = struct[0].R_s_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    R_v_1 = struct[0].R_v_1
    X_v_1 = struct[0].X_v_1
    D1_1 = struct[0].D1_1
    D2_1 = struct[0].D2_1
    D3_1 = struct[0].D3_1
    K_delta_1 = struct[0].K_delta_1
    T_wo_1 = struct[0].T_wo_1
    T_i_1 = struct[0].T_i_1
    K_q_1 = struct[0].K_q_1
    T_q_1 = struct[0].T_q_1
    H_s_1 = struct[0].H_s_1
    K_p_soc_1 = struct[0].K_p_soc_1
    K_i_soc_1 = struct[0].K_i_soc_1
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    p_in_1 = struct[0].p_in_1
    Dp_ref_1 = struct[0].Dp_ref_1
    q_ref_1 = struct[0].q_ref_1
    p_src_1 = struct[0].p_src_1
    soc_ref_1 = struct[0].soc_ref_1
    
    # Dynamical states:
    delta_2 = struct[0].x[0,0]
    omega_2 = struct[0].x[1,0]
    e1q_2 = struct[0].x[2,0]
    e1d_2 = struct[0].x[3,0]
    v_c_2 = struct[0].x[4,0]
    xi_v_2 = struct[0].x[5,0]
    x_gov_1_2 = struct[0].x[6,0]
    x_gov_2_2 = struct[0].x[7,0]
    xi_imw_2 = struct[0].x[8,0]
    delta_3 = struct[0].x[9,0]
    omega_3 = struct[0].x[10,0]
    e1q_3 = struct[0].x[11,0]
    e1d_3 = struct[0].x[12,0]
    v_c_3 = struct[0].x[13,0]
    xi_v_3 = struct[0].x[14,0]
    x_gov_1_3 = struct[0].x[15,0]
    x_gov_2_3 = struct[0].x[16,0]
    xi_imw_3 = struct[0].x[17,0]
    xi_freq = struct[0].x[18,0]
    delta_1 = struct[0].x[19,0]
    omega_v_1 = struct[0].x[20,0]
    x_wo_1 = struct[0].x[21,0]
    i_d_1 = struct[0].x[22,0]
    i_q_1 = struct[0].x[23,0]
    xi_q_1 = struct[0].x[24,0]
    soc_1 = struct[0].x[25,0]
    xi_soc_1 = struct[0].x[26,0]
    
    # Algebraic states:
    V_1 = struct[0].y_run[0,0]
    theta_1 = struct[0].y_run[1,0]
    V_2 = struct[0].y_run[2,0]
    theta_2 = struct[0].y_run[3,0]
    V_3 = struct[0].y_run[4,0]
    theta_3 = struct[0].y_run[5,0]
    i_d_2 = struct[0].y_run[6,0]
    i_q_2 = struct[0].y_run[7,0]
    p_g_2_1 = struct[0].y_run[8,0]
    q_g_2_1 = struct[0].y_run[9,0]
    v_f_2 = struct[0].y_run[10,0]
    p_m_ref_2 = struct[0].y_run[11,0]
    p_m_2 = struct[0].y_run[12,0]
    i_d_3 = struct[0].y_run[13,0]
    i_q_3 = struct[0].y_run[14,0]
    p_g_3_1 = struct[0].y_run[15,0]
    q_g_3_1 = struct[0].y_run[16,0]
    v_f_3 = struct[0].y_run[17,0]
    p_m_ref_3 = struct[0].y_run[18,0]
    p_m_3 = struct[0].y_run[19,0]
    p_r_2 = struct[0].y_run[20,0]
    p_r_3 = struct[0].y_run[21,0]
    i_d_ref_1 = struct[0].y_run[22,0]
    i_q_ref_1 = struct[0].y_run[23,0]
    p_g_1_1 = struct[0].y_run[24,0]
    q_g_1_1 = struct[0].y_run[25,0]
    p_d2_1 = struct[0].y_run[26,0]
    e_v_1 = struct[0].y_run[27,0]
    p_sto_1 = struct[0].y_run[28,0]
    p_m_1 = struct[0].y_run[29,0]
    omega_coi = struct[0].y_run[30,0]
    
    struct[0].u_run[0,0] = P_1
    struct[0].u_run[1,0] = Q_1
    struct[0].u_run[2,0] = P_2
    struct[0].u_run[3,0] = Q_2
    struct[0].u_run[4,0] = P_3
    struct[0].u_run[5,0] = Q_3
    struct[0].u_run[6,0] = v_ref_2
    struct[0].u_run[7,0] = v_pss_2
    struct[0].u_run[8,0] = p_c_2
    struct[0].u_run[9,0] = v_ref_3
    struct[0].u_run[10,0] = v_pss_3
    struct[0].u_run[11,0] = p_c_3
    struct[0].u_run[12,0] = p_in_1
    struct[0].u_run[13,0] = Dp_ref_1
    struct[0].u_run[14,0] = q_ref_1
    struct[0].u_run[15,0] = p_src_1
    struct[0].u_run[16,0] = soc_ref_1
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[1,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[2,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[3,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[4,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[5,0] = -V_2 + v_ref_2
        struct[0].f[6,0] = (p_m_ref_2 - x_gov_1_2)/T_gov_1_2
        struct[0].f[7,0] = (x_gov_1_2 - x_gov_2_2)/T_gov_3_2
        struct[0].f[8,0] = K_imw_2*(p_c_2 - p_g_2_1) - 1.0e-6*xi_imw_2
        struct[0].f[9,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[10,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[11,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[12,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[13,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[14,0] = -V_3 + v_ref_3
        struct[0].f[15,0] = (p_m_ref_3 - x_gov_1_3)/T_gov_1_3
        struct[0].f[16,0] = (x_gov_1_3 - x_gov_2_3)/T_gov_3_3
        struct[0].f[17,0] = K_imw_3*(p_c_3 - p_g_3_1) - 1.0e-6*xi_imw_3
        struct[0].f[18,0] = 1 - omega_coi
        struct[0].f[19,0] = D3_1*(-p_g_1_1 + p_m_1) - K_delta_1*delta_1 + Omega_b_1*(-omega_coi + omega_v_1)
        struct[0].f[20,0] = (-D1_1*(omega_v_1 - 1) - p_d2_1 - p_g_1_1 + p_m_1)/(2*H_1)
        struct[0].f[21,0] = (omega_v_1 - x_wo_1 - 1.0)/T_wo_1
        struct[0].f[22,0] = (-i_d_1 + i_d_ref_1)/T_i_1
        struct[0].f[23,0] = (-i_q_1 + i_q_ref_1)/T_i_1
        struct[0].f[24,0] = -q_g_1_1 + q_ref_1
        struct[0].f[25,0] = -p_sto_1/H_s_1
        struct[0].f[26,0] = -soc_1 + soc_ref_1
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_2 + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - bs_1_2/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_3) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + V_3**2*g_2_3 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + V_3**2*(-b_2_3 - bs_2_3/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[7,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[8,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[9,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[10,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[11,0] = p_c_2 - p_m_ref_2 + p_r_2 + xi_imw_2 - (omega_2 - omega_ref_2)/Droop_2
        struct[0].g[12,0] = T_gov_2_2*(x_gov_1_2 - x_gov_2_2)/T_gov_3_2 - p_m_2 + x_gov_2_2
        struct[0].g[13,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[14,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[15,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[16,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[17,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[18,0] = p_c_3 - p_m_ref_3 + p_r_3 + xi_imw_3 - (omega_3 - omega_ref_3)/Droop_3
        struct[0].g[19,0] = T_gov_2_3*(x_gov_1_3 - x_gov_2_3)/T_gov_3_3 - p_m_3 + x_gov_2_3
        struct[0].g[20,0] = K_sec_2*xi_freq/2 - p_r_2
        struct[0].g[21,0] = K_sec_3*xi_freq/2 - p_r_3
        struct[0].g[22,0] = R_v_1*i_q_ref_1 + V_1*cos(delta_1 - theta_1) + X_v_1*i_d_ref_1 - e_v_1
        struct[0].g[23,0] = R_v_1*i_d_ref_1 + V_1*sin(delta_1 - theta_1) - X_v_1*i_q_ref_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = D2_1*(omega_v_1 - x_wo_1 - 1.0) - p_d2_1
        struct[0].g[27,0] = K_q_1*(-q_g_1_1 + q_ref_1 + xi_q_1/T_q_1) - e_v_1
        struct[0].g[28,0] = -i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_src_1 + p_sto_1
        struct[0].g[29,0] = Dp_ref_1 - p_m_1 + p_src_1 + Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
        struct[0].g[30,0] = omega_2/2 + omega_3/2 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) + i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2))
        struct[0].h[4,0] = i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) + i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3))
        struct[0].h[5,0] = i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) + i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1))
        struct[0].h[6,0] = Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_2
        struct[0].Fx[0,1] = Omega_b_2
        struct[0].Fx[1,0] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx[1,1] = -D_2/(2*H_2)
        struct[0].Fx[2,2] = -1/T1d0_2
        struct[0].Fx[3,3] = -1/T1q0_2
        struct[0].Fx[4,4] = -1/T_r_2
        struct[0].Fx[6,6] = -1/T_gov_1_2
        struct[0].Fx[7,6] = 1/T_gov_3_2
        struct[0].Fx[7,7] = -1/T_gov_3_2
        struct[0].Fx[9,9] = -K_delta_3
        struct[0].Fx[9,10] = Omega_b_3
        struct[0].Fx[10,9] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx[10,10] = -D_3/(2*H_3)
        struct[0].Fx[11,11] = -1/T1d0_3
        struct[0].Fx[12,12] = -1/T1q0_3
        struct[0].Fx[13,13] = -1/T_r_3
        struct[0].Fx[15,15] = -1/T_gov_1_3
        struct[0].Fx[16,15] = 1/T_gov_3_3
        struct[0].Fx[16,16] = -1/T_gov_3_3
        struct[0].Fx[19,19] = -K_delta_1
        struct[0].Fx[19,20] = Omega_b_1
        struct[0].Fx[20,20] = -D1_1/(2*H_1)
        struct[0].Fx[21,20] = 1/T_wo_1
        struct[0].Fx[21,21] = -1/T_wo_1
        struct[0].Fx[22,22] = -1/T_i_1
        struct[0].Fx[23,23] = -1/T_i_1

    if mode == 11:

        struct[0].Fy[0,30] = -Omega_b_2
        struct[0].Fy[1,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,6] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,7] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,12] = 1/(2*H_2)
        struct[0].Fy[1,30] = D_2/(2*H_2)
        struct[0].Fy[2,6] = (X1d_2 - X_d_2)/T1d0_2
        struct[0].Fy[2,10] = 1/T1d0_2
        struct[0].Fy[3,7] = (-X1q_2 + X_q_2)/T1q0_2
        struct[0].Fy[4,2] = 1/T_r_2
        struct[0].Fy[5,2] = -1
        struct[0].Fy[6,11] = 1/T_gov_1_2
        struct[0].Fy[8,8] = -K_imw_2
        struct[0].Fy[9,30] = -Omega_b_3
        struct[0].Fy[10,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,13] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,14] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,19] = 1/(2*H_3)
        struct[0].Fy[10,30] = D_3/(2*H_3)
        struct[0].Fy[11,13] = (X1d_3 - X_d_3)/T1d0_3
        struct[0].Fy[11,17] = 1/T1d0_3
        struct[0].Fy[12,14] = (-X1q_3 + X_q_3)/T1q0_3
        struct[0].Fy[13,4] = 1/T_r_3
        struct[0].Fy[14,4] = -1
        struct[0].Fy[15,18] = 1/T_gov_1_3
        struct[0].Fy[17,15] = -K_imw_3
        struct[0].Fy[18,30] = -1
        struct[0].Fy[19,24] = -D3_1
        struct[0].Fy[19,29] = D3_1
        struct[0].Fy[19,30] = -Omega_b_1
        struct[0].Fy[20,24] = -1/(2*H_1)
        struct[0].Fy[20,26] = -1/(2*H_1)
        struct[0].Fy[20,29] = 1/(2*H_1)
        struct[0].Fy[22,22] = 1/T_i_1
        struct[0].Fy[23,23] = 1/T_i_1
        struct[0].Fy[24,25] = -1
        struct[0].Fy[25,28] = -1/H_s_1

        struct[0].Gx[6,0] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gx[6,2] = -1
        struct[0].Gx[7,0] = V_2*cos(delta_2 - theta_2)
        struct[0].Gx[7,3] = -1
        struct[0].Gx[8,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gx[9,0] = -V_2*i_d_2*sin(delta_2 - theta_2) - V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gx[10,4] = -K_a_2
        struct[0].Gx[10,5] = K_ai_2
        struct[0].Gx[11,1] = -1/Droop_2
        struct[0].Gx[11,8] = 1
        struct[0].Gx[12,6] = T_gov_2_2/T_gov_3_2
        struct[0].Gx[12,7] = -T_gov_2_2/T_gov_3_2 + 1
        struct[0].Gx[13,9] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gx[13,11] = -1
        struct[0].Gx[14,9] = V_3*cos(delta_3 - theta_3)
        struct[0].Gx[14,12] = -1
        struct[0].Gx[15,9] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gx[16,9] = -V_3*i_d_3*sin(delta_3 - theta_3) - V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gx[17,13] = -K_a_3
        struct[0].Gx[17,14] = K_ai_3
        struct[0].Gx[18,10] = -1/Droop_3
        struct[0].Gx[18,17] = 1
        struct[0].Gx[19,15] = T_gov_2_3/T_gov_3_3
        struct[0].Gx[19,16] = -T_gov_2_3/T_gov_3_3 + 1
        struct[0].Gx[20,18] = K_sec_2/2
        struct[0].Gx[21,18] = K_sec_3/2
        struct[0].Gx[22,19] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gx[23,19] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx[24,19] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gx[24,22] = V_1*sin(delta_1 - theta_1)
        struct[0].Gx[24,23] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx[25,19] = -V_1*i_d_1*sin(delta_1 - theta_1) - V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gx[25,22] = V_1*cos(delta_1 - theta_1)
        struct[0].Gx[25,23] = -V_1*sin(delta_1 - theta_1)
        struct[0].Gx[26,20] = D2_1
        struct[0].Gx[26,21] = -D2_1
        struct[0].Gx[27,24] = K_q_1/T_q_1
        struct[0].Gx[28,19] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gx[28,22] = -2*R_s_1*i_d_1 - V_1*sin(delta_1 - theta_1)
        struct[0].Gx[28,23] = -2*R_s_1*i_q_1 - V_1*cos(delta_1 - theta_1)
        struct[0].Gx[29,25] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (-K_p_soc_1, True)]))
        struct[0].Gx[29,26] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1, True)]))
        struct[0].Gx[30,1] = 1/2
        struct[0].Gx[30,10] = 1/2

        struct[0].Gy[0,0] = 2*V_1*g_1_2 + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[0,24] = -S_n_1/S_base
        struct[0].Gy[1,0] = 2*V_1*(-b_1_2 - bs_1_2/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[1,25] = -S_n_1/S_base
        struct[0].Gy[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_3) + V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[2,4] = V_2*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[2,5] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[2,8] = -S_n_2/S_base
        struct[0].Gy[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[3,4] = V_2*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[3,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[3,9] = -S_n_2/S_base
        struct[0].Gy[4,2] = V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[4,3] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[4,4] = V_2*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + 2*V_3*g_2_3
        struct[0].Gy[4,5] = V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[4,15] = -S_n_3/S_base
        struct[0].Gy[5,2] = V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[5,3] = V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[5,4] = V_2*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + 2*V_3*(-b_2_3 - bs_2_3/2)
        struct[0].Gy[5,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[5,16] = -S_n_3/S_base
        struct[0].Gy[6,2] = cos(delta_2 - theta_2)
        struct[0].Gy[6,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[6,6] = X1d_2
        struct[0].Gy[6,7] = R_a_2
        struct[0].Gy[7,2] = sin(delta_2 - theta_2)
        struct[0].Gy[7,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy[7,6] = R_a_2
        struct[0].Gy[7,7] = -X1q_2
        struct[0].Gy[8,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[8,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[8,6] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[8,7] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[9,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[9,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[9,6] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[9,7] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy[13,4] = cos(delta_3 - theta_3)
        struct[0].Gy[13,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[13,13] = X1d_3
        struct[0].Gy[13,14] = R_a_3
        struct[0].Gy[14,4] = sin(delta_3 - theta_3)
        struct[0].Gy[14,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy[14,13] = R_a_3
        struct[0].Gy[14,14] = -X1q_3
        struct[0].Gy[15,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[15,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[15,13] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[15,14] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[16,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[16,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[16,13] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[16,14] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy[22,22] = X_v_1
        struct[0].Gy[22,23] = R_v_1
        struct[0].Gy[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy[23,22] = R_v_1
        struct[0].Gy[23,23] = -X_v_1
        struct[0].Gy[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[27,25] = -K_q_1
        struct[0].Gy[28,0] = -i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[28,1] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)

    if mode > 12:

        struct[0].Fu[5,6] = 1
        struct[0].Fu[8,8] = K_imw_2
        struct[0].Fu[14,9] = 1
        struct[0].Fu[17,11] = K_imw_3
        struct[0].Fu[24,14] = 1
        struct[0].Fu[26,16] = 1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[10,6] = K_a_2
        struct[0].Gu[10,7] = K_a_2
        struct[0].Gu[17,9] = K_a_3
        struct[0].Gu[17,10] = K_a_3
        struct[0].Gu[27,14] = K_q_1
        struct[0].Gu[29,16] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_p_soc_1, True)]))

        struct[0].Hx[3,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Hx[4,9] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Hx[5,19] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Hx[5,22] = 2*R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)
        struct[0].Hx[5,23] = 2*R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1)
        struct[0].Hx[6,25] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (-K_p_soc_1, True)]))
        struct[0].Hx[6,26] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1, True)]))

        struct[0].Hy[0,0] = 1
        struct[0].Hy[1,2] = 1
        struct[0].Hy[2,4] = 1
        struct[0].Hy[3,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Hy[3,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Hy[3,6] = 2*R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)
        struct[0].Hy[3,7] = 2*R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)
        struct[0].Hy[4,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Hy[4,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Hy[4,13] = 2*R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)
        struct[0].Hy[4,14] = 2*R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)
        struct[0].Hy[5,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Hy[5,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)

        struct[0].Hu[6,16] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_p_soc_1, True)]))



def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_2 = struct[0].g_1_2
    b_1_2 = struct[0].b_1_2
    bs_1_2 = struct[0].bs_1_2
    g_2_3 = struct[0].g_2_3
    b_2_3 = struct[0].b_2_3
    bs_2_3 = struct[0].bs_2_3
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
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
    T_gov_1_2 = struct[0].T_gov_1_2
    T_gov_2_2 = struct[0].T_gov_2_2
    T_gov_3_2 = struct[0].T_gov_3_2
    K_imw_2 = struct[0].K_imw_2
    omega_ref_2 = struct[0].omega_ref_2
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
    T_gov_1_3 = struct[0].T_gov_1_3
    T_gov_2_3 = struct[0].T_gov_2_3
    T_gov_3_3 = struct[0].T_gov_3_3
    K_imw_3 = struct[0].K_imw_3
    omega_ref_3 = struct[0].omega_ref_3
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    S_n_1 = struct[0].S_n_1
    R_s_1 = struct[0].R_s_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    R_v_1 = struct[0].R_v_1
    X_v_1 = struct[0].X_v_1
    D1_1 = struct[0].D1_1
    D2_1 = struct[0].D2_1
    D3_1 = struct[0].D3_1
    K_delta_1 = struct[0].K_delta_1
    T_wo_1 = struct[0].T_wo_1
    T_i_1 = struct[0].T_i_1
    K_q_1 = struct[0].K_q_1
    T_q_1 = struct[0].T_q_1
    H_s_1 = struct[0].H_s_1
    K_p_soc_1 = struct[0].K_p_soc_1
    K_i_soc_1 = struct[0].K_i_soc_1
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    p_in_1 = struct[0].p_in_1
    Dp_ref_1 = struct[0].Dp_ref_1
    q_ref_1 = struct[0].q_ref_1
    p_src_1 = struct[0].p_src_1
    soc_ref_1 = struct[0].soc_ref_1
    
    # Dynamical states:
    delta_2 = struct[0].x[0,0]
    omega_2 = struct[0].x[1,0]
    e1q_2 = struct[0].x[2,0]
    e1d_2 = struct[0].x[3,0]
    v_c_2 = struct[0].x[4,0]
    xi_v_2 = struct[0].x[5,0]
    x_gov_1_2 = struct[0].x[6,0]
    x_gov_2_2 = struct[0].x[7,0]
    xi_imw_2 = struct[0].x[8,0]
    delta_3 = struct[0].x[9,0]
    omega_3 = struct[0].x[10,0]
    e1q_3 = struct[0].x[11,0]
    e1d_3 = struct[0].x[12,0]
    v_c_3 = struct[0].x[13,0]
    xi_v_3 = struct[0].x[14,0]
    x_gov_1_3 = struct[0].x[15,0]
    x_gov_2_3 = struct[0].x[16,0]
    xi_imw_3 = struct[0].x[17,0]
    xi_freq = struct[0].x[18,0]
    delta_1 = struct[0].x[19,0]
    omega_v_1 = struct[0].x[20,0]
    x_wo_1 = struct[0].x[21,0]
    i_d_1 = struct[0].x[22,0]
    i_q_1 = struct[0].x[23,0]
    xi_q_1 = struct[0].x[24,0]
    soc_1 = struct[0].x[25,0]
    xi_soc_1 = struct[0].x[26,0]
    
    # Algebraic states:
    V_1 = struct[0].y_ini[0,0]
    theta_1 = struct[0].y_ini[1,0]
    V_2 = struct[0].y_ini[2,0]
    theta_2 = struct[0].y_ini[3,0]
    V_3 = struct[0].y_ini[4,0]
    theta_3 = struct[0].y_ini[5,0]
    i_d_2 = struct[0].y_ini[6,0]
    i_q_2 = struct[0].y_ini[7,0]
    p_g_2_1 = struct[0].y_ini[8,0]
    q_g_2_1 = struct[0].y_ini[9,0]
    v_f_2 = struct[0].y_ini[10,0]
    p_m_ref_2 = struct[0].y_ini[11,0]
    p_m_2 = struct[0].y_ini[12,0]
    i_d_3 = struct[0].y_ini[13,0]
    i_q_3 = struct[0].y_ini[14,0]
    p_g_3_1 = struct[0].y_ini[15,0]
    q_g_3_1 = struct[0].y_ini[16,0]
    v_f_3 = struct[0].y_ini[17,0]
    p_m_ref_3 = struct[0].y_ini[18,0]
    p_m_3 = struct[0].y_ini[19,0]
    p_r_2 = struct[0].y_ini[20,0]
    p_r_3 = struct[0].y_ini[21,0]
    i_d_ref_1 = struct[0].y_ini[22,0]
    i_q_ref_1 = struct[0].y_ini[23,0]
    p_g_1_1 = struct[0].y_ini[24,0]
    q_g_1_1 = struct[0].y_ini[25,0]
    p_d2_1 = struct[0].y_ini[26,0]
    e_v_1 = struct[0].y_ini[27,0]
    p_sto_1 = struct[0].y_ini[28,0]
    p_m_1 = struct[0].y_ini[29,0]
    omega_coi = struct[0].y_ini[30,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[1,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[2,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[3,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[4,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[5,0] = -V_2 + v_ref_2
        struct[0].f[6,0] = (p_m_ref_2 - x_gov_1_2)/T_gov_1_2
        struct[0].f[7,0] = (x_gov_1_2 - x_gov_2_2)/T_gov_3_2
        struct[0].f[8,0] = K_imw_2*(p_c_2 - p_g_2_1) - 1.0e-6*xi_imw_2
        struct[0].f[9,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[10,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[11,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[12,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[13,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[14,0] = -V_3 + v_ref_3
        struct[0].f[15,0] = (p_m_ref_3 - x_gov_1_3)/T_gov_1_3
        struct[0].f[16,0] = (x_gov_1_3 - x_gov_2_3)/T_gov_3_3
        struct[0].f[17,0] = K_imw_3*(p_c_3 - p_g_3_1) - 1.0e-6*xi_imw_3
        struct[0].f[18,0] = 1 - omega_coi
        struct[0].f[19,0] = D3_1*(-p_g_1_1 + p_m_1) - K_delta_1*delta_1 + Omega_b_1*(-omega_coi + omega_v_1)
        struct[0].f[20,0] = (-D1_1*(omega_v_1 - 1) - p_d2_1 - p_g_1_1 + p_m_1)/(2*H_1)
        struct[0].f[21,0] = (omega_v_1 - x_wo_1 - 1.0)/T_wo_1
        struct[0].f[22,0] = (-i_d_1 + i_d_ref_1)/T_i_1
        struct[0].f[23,0] = (-i_q_1 + i_q_ref_1)/T_i_1
        struct[0].f[24,0] = -q_g_1_1 + q_ref_1
        struct[0].f[25,0] = -p_sto_1/H_s_1
        struct[0].f[26,0] = -soc_1 + soc_ref_1
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_2 + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - bs_1_2/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_3) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + V_3**2*g_2_3 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + V_3**2*(-b_2_3 - bs_2_3/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[7,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[8,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[9,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[10,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[11,0] = p_c_2 - p_m_ref_2 + p_r_2 + xi_imw_2 - (omega_2 - omega_ref_2)/Droop_2
        struct[0].g[12,0] = T_gov_2_2*(x_gov_1_2 - x_gov_2_2)/T_gov_3_2 - p_m_2 + x_gov_2_2
        struct[0].g[13,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[14,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[15,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[16,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[17,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[18,0] = p_c_3 - p_m_ref_3 + p_r_3 + xi_imw_3 - (omega_3 - omega_ref_3)/Droop_3
        struct[0].g[19,0] = T_gov_2_3*(x_gov_1_3 - x_gov_2_3)/T_gov_3_3 - p_m_3 + x_gov_2_3
        struct[0].g[20,0] = K_sec_2*xi_freq/2 - p_r_2
        struct[0].g[21,0] = K_sec_3*xi_freq/2 - p_r_3
        struct[0].g[22,0] = R_v_1*i_q_ref_1 + V_1*cos(delta_1 - theta_1) + X_v_1*i_d_ref_1 - e_v_1
        struct[0].g[23,0] = R_v_1*i_d_ref_1 + V_1*sin(delta_1 - theta_1) - X_v_1*i_q_ref_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = D2_1*(omega_v_1 - x_wo_1 - 1.0) - p_d2_1
        struct[0].g[27,0] = K_q_1*(-q_g_1_1 + q_ref_1 + xi_q_1/T_q_1) - e_v_1
        struct[0].g[28,0] = -i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_src_1 + p_sto_1
        struct[0].g[29,0] = Dp_ref_1 - p_m_1 + p_src_1 + Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
        struct[0].g[30,0] = omega_2/2 + omega_3/2 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) + i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2))
        struct[0].h[4,0] = i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) + i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3))
        struct[0].h[5,0] = i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) + i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1))
        struct[0].h[6,0] = Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_2
        struct[0].Fx_ini[0,1] = Omega_b_2
        struct[0].Fx_ini[1,0] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx_ini[1,1] = -D_2/(2*H_2)
        struct[0].Fx_ini[2,2] = -1/T1d0_2
        struct[0].Fx_ini[3,3] = -1/T1q0_2
        struct[0].Fx_ini[4,4] = -1/T_r_2
        struct[0].Fx_ini[6,6] = -1/T_gov_1_2
        struct[0].Fx_ini[7,6] = 1/T_gov_3_2
        struct[0].Fx_ini[7,7] = -1/T_gov_3_2
        struct[0].Fx_ini[8,8] = -0.00000100000000000000
        struct[0].Fx_ini[9,9] = -K_delta_3
        struct[0].Fx_ini[9,10] = Omega_b_3
        struct[0].Fx_ini[10,9] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx_ini[10,10] = -D_3/(2*H_3)
        struct[0].Fx_ini[11,11] = -1/T1d0_3
        struct[0].Fx_ini[12,12] = -1/T1q0_3
        struct[0].Fx_ini[13,13] = -1/T_r_3
        struct[0].Fx_ini[15,15] = -1/T_gov_1_3
        struct[0].Fx_ini[16,15] = 1/T_gov_3_3
        struct[0].Fx_ini[16,16] = -1/T_gov_3_3
        struct[0].Fx_ini[17,17] = -0.00000100000000000000
        struct[0].Fx_ini[19,19] = -K_delta_1
        struct[0].Fx_ini[19,20] = Omega_b_1
        struct[0].Fx_ini[20,20] = -D1_1/(2*H_1)
        struct[0].Fx_ini[21,20] = 1/T_wo_1
        struct[0].Fx_ini[21,21] = -1/T_wo_1
        struct[0].Fx_ini[22,22] = -1/T_i_1
        struct[0].Fx_ini[23,23] = -1/T_i_1
        struct[0].Fx_ini[26,25] = -1

    if mode == 11:

        struct[0].Fy_ini[0,30] = -Omega_b_2 
        struct[0].Fy_ini[1,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,6] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,7] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2) 
        struct[0].Fy_ini[1,12] = 1/(2*H_2) 
        struct[0].Fy_ini[1,30] = D_2/(2*H_2) 
        struct[0].Fy_ini[2,6] = (X1d_2 - X_d_2)/T1d0_2 
        struct[0].Fy_ini[2,10] = 1/T1d0_2 
        struct[0].Fy_ini[3,7] = (-X1q_2 + X_q_2)/T1q0_2 
        struct[0].Fy_ini[4,2] = 1/T_r_2 
        struct[0].Fy_ini[5,2] = -1 
        struct[0].Fy_ini[6,11] = 1/T_gov_1_2 
        struct[0].Fy_ini[8,8] = -K_imw_2 
        struct[0].Fy_ini[9,30] = -Omega_b_3 
        struct[0].Fy_ini[10,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,13] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,14] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3) 
        struct[0].Fy_ini[10,19] = 1/(2*H_3) 
        struct[0].Fy_ini[10,30] = D_3/(2*H_3) 
        struct[0].Fy_ini[11,13] = (X1d_3 - X_d_3)/T1d0_3 
        struct[0].Fy_ini[11,17] = 1/T1d0_3 
        struct[0].Fy_ini[12,14] = (-X1q_3 + X_q_3)/T1q0_3 
        struct[0].Fy_ini[13,4] = 1/T_r_3 
        struct[0].Fy_ini[14,4] = -1 
        struct[0].Fy_ini[15,18] = 1/T_gov_1_3 
        struct[0].Fy_ini[17,15] = -K_imw_3 
        struct[0].Fy_ini[18,30] = -1 
        struct[0].Fy_ini[19,24] = -D3_1 
        struct[0].Fy_ini[19,29] = D3_1 
        struct[0].Fy_ini[19,30] = -Omega_b_1 
        struct[0].Fy_ini[20,24] = -1/(2*H_1) 
        struct[0].Fy_ini[20,26] = -1/(2*H_1) 
        struct[0].Fy_ini[20,29] = 1/(2*H_1) 
        struct[0].Fy_ini[22,22] = 1/T_i_1 
        struct[0].Fy_ini[23,23] = 1/T_i_1 
        struct[0].Fy_ini[24,25] = -1 
        struct[0].Fy_ini[25,28] = -1/H_s_1 

        struct[0].Gy_ini[0,0] = 2*V_1*g_1_2 + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[0,24] = -S_n_1/S_base
        struct[0].Gy_ini[1,0] = 2*V_1*(-b_1_2 - bs_1_2/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[1,25] = -S_n_1/S_base
        struct[0].Gy_ini[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_3) + V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[2,4] = V_2*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[2,5] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[2,8] = -S_n_2/S_base
        struct[0].Gy_ini[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy_ini[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy_ini[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[3,4] = V_2*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[3,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[3,9] = -S_n_2/S_base
        struct[0].Gy_ini[4,2] = V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[4,3] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[4,4] = V_2*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + 2*V_3*g_2_3
        struct[0].Gy_ini[4,5] = V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[4,15] = -S_n_3/S_base
        struct[0].Gy_ini[5,2] = V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy_ini[5,3] = V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[5,4] = V_2*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + 2*V_3*(-b_2_3 - bs_2_3/2)
        struct[0].Gy_ini[5,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy_ini[5,16] = -S_n_3/S_base
        struct[0].Gy_ini[6,2] = cos(delta_2 - theta_2)
        struct[0].Gy_ini[6,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[6,6] = X1d_2
        struct[0].Gy_ini[6,7] = R_a_2
        struct[0].Gy_ini[7,2] = sin(delta_2 - theta_2)
        struct[0].Gy_ini[7,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[7,6] = R_a_2
        struct[0].Gy_ini[7,7] = -X1q_2
        struct[0].Gy_ini[8,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[8,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[8,6] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[8,7] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[8,8] = -1
        struct[0].Gy_ini[9,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[9,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[9,6] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy_ini[9,7] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy_ini[9,9] = -1
        struct[0].Gy_ini[10,10] = -1
        struct[0].Gy_ini[11,11] = -1
        struct[0].Gy_ini[11,20] = 1
        struct[0].Gy_ini[12,12] = -1
        struct[0].Gy_ini[13,4] = cos(delta_3 - theta_3)
        struct[0].Gy_ini[13,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[13,13] = X1d_3
        struct[0].Gy_ini[13,14] = R_a_3
        struct[0].Gy_ini[14,4] = sin(delta_3 - theta_3)
        struct[0].Gy_ini[14,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[14,13] = R_a_3
        struct[0].Gy_ini[14,14] = -X1q_3
        struct[0].Gy_ini[15,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[15,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[15,13] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[15,14] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[15,15] = -1
        struct[0].Gy_ini[16,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[16,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[16,13] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy_ini[16,14] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy_ini[16,16] = -1
        struct[0].Gy_ini[17,17] = -1
        struct[0].Gy_ini[18,18] = -1
        struct[0].Gy_ini[18,21] = 1
        struct[0].Gy_ini[19,19] = -1
        struct[0].Gy_ini[20,20] = -1
        struct[0].Gy_ini[21,21] = -1
        struct[0].Gy_ini[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy_ini[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[22,22] = X_v_1
        struct[0].Gy_ini[22,23] = R_v_1
        struct[0].Gy_ini[22,27] = -1
        struct[0].Gy_ini[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy_ini[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[23,22] = R_v_1
        struct[0].Gy_ini[23,23] = -X_v_1
        struct[0].Gy_ini[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[25,25] = -1
        struct[0].Gy_ini[26,26] = -1
        struct[0].Gy_ini[27,25] = -K_q_1
        struct[0].Gy_ini[27,27] = -1
        struct[0].Gy_ini[28,0] = -i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy_ini[28,1] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy_ini[28,28] = 1
        struct[0].Gy_ini[29,29] = -1
        struct[0].Gy_ini[30,30] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_1_2 = struct[0].g_1_2
    b_1_2 = struct[0].b_1_2
    bs_1_2 = struct[0].bs_1_2
    g_2_3 = struct[0].g_2_3
    b_2_3 = struct[0].b_2_3
    bs_2_3 = struct[0].bs_2_3
    U_1_n = struct[0].U_1_n
    U_2_n = struct[0].U_2_n
    U_3_n = struct[0].U_3_n
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
    T_gov_1_2 = struct[0].T_gov_1_2
    T_gov_2_2 = struct[0].T_gov_2_2
    T_gov_3_2 = struct[0].T_gov_3_2
    K_imw_2 = struct[0].K_imw_2
    omega_ref_2 = struct[0].omega_ref_2
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
    T_gov_1_3 = struct[0].T_gov_1_3
    T_gov_2_3 = struct[0].T_gov_2_3
    T_gov_3_3 = struct[0].T_gov_3_3
    K_imw_3 = struct[0].K_imw_3
    omega_ref_3 = struct[0].omega_ref_3
    K_sec_2 = struct[0].K_sec_2
    K_sec_3 = struct[0].K_sec_3
    S_n_1 = struct[0].S_n_1
    R_s_1 = struct[0].R_s_1
    H_1 = struct[0].H_1
    Omega_b_1 = struct[0].Omega_b_1
    R_v_1 = struct[0].R_v_1
    X_v_1 = struct[0].X_v_1
    D1_1 = struct[0].D1_1
    D2_1 = struct[0].D2_1
    D3_1 = struct[0].D3_1
    K_delta_1 = struct[0].K_delta_1
    T_wo_1 = struct[0].T_wo_1
    T_i_1 = struct[0].T_i_1
    K_q_1 = struct[0].K_q_1
    T_q_1 = struct[0].T_q_1
    H_s_1 = struct[0].H_s_1
    K_p_soc_1 = struct[0].K_p_soc_1
    K_i_soc_1 = struct[0].K_i_soc_1
    
    # Inputs:
    P_1 = struct[0].P_1
    Q_1 = struct[0].Q_1
    P_2 = struct[0].P_2
    Q_2 = struct[0].Q_2
    P_3 = struct[0].P_3
    Q_3 = struct[0].Q_3
    v_ref_2 = struct[0].v_ref_2
    v_pss_2 = struct[0].v_pss_2
    p_c_2 = struct[0].p_c_2
    v_ref_3 = struct[0].v_ref_3
    v_pss_3 = struct[0].v_pss_3
    p_c_3 = struct[0].p_c_3
    p_in_1 = struct[0].p_in_1
    Dp_ref_1 = struct[0].Dp_ref_1
    q_ref_1 = struct[0].q_ref_1
    p_src_1 = struct[0].p_src_1
    soc_ref_1 = struct[0].soc_ref_1
    
    # Dynamical states:
    delta_2 = struct[0].x[0,0]
    omega_2 = struct[0].x[1,0]
    e1q_2 = struct[0].x[2,0]
    e1d_2 = struct[0].x[3,0]
    v_c_2 = struct[0].x[4,0]
    xi_v_2 = struct[0].x[5,0]
    x_gov_1_2 = struct[0].x[6,0]
    x_gov_2_2 = struct[0].x[7,0]
    xi_imw_2 = struct[0].x[8,0]
    delta_3 = struct[0].x[9,0]
    omega_3 = struct[0].x[10,0]
    e1q_3 = struct[0].x[11,0]
    e1d_3 = struct[0].x[12,0]
    v_c_3 = struct[0].x[13,0]
    xi_v_3 = struct[0].x[14,0]
    x_gov_1_3 = struct[0].x[15,0]
    x_gov_2_3 = struct[0].x[16,0]
    xi_imw_3 = struct[0].x[17,0]
    xi_freq = struct[0].x[18,0]
    delta_1 = struct[0].x[19,0]
    omega_v_1 = struct[0].x[20,0]
    x_wo_1 = struct[0].x[21,0]
    i_d_1 = struct[0].x[22,0]
    i_q_1 = struct[0].x[23,0]
    xi_q_1 = struct[0].x[24,0]
    soc_1 = struct[0].x[25,0]
    xi_soc_1 = struct[0].x[26,0]
    
    # Algebraic states:
    V_1 = struct[0].y_run[0,0]
    theta_1 = struct[0].y_run[1,0]
    V_2 = struct[0].y_run[2,0]
    theta_2 = struct[0].y_run[3,0]
    V_3 = struct[0].y_run[4,0]
    theta_3 = struct[0].y_run[5,0]
    i_d_2 = struct[0].y_run[6,0]
    i_q_2 = struct[0].y_run[7,0]
    p_g_2_1 = struct[0].y_run[8,0]
    q_g_2_1 = struct[0].y_run[9,0]
    v_f_2 = struct[0].y_run[10,0]
    p_m_ref_2 = struct[0].y_run[11,0]
    p_m_2 = struct[0].y_run[12,0]
    i_d_3 = struct[0].y_run[13,0]
    i_q_3 = struct[0].y_run[14,0]
    p_g_3_1 = struct[0].y_run[15,0]
    q_g_3_1 = struct[0].y_run[16,0]
    v_f_3 = struct[0].y_run[17,0]
    p_m_ref_3 = struct[0].y_run[18,0]
    p_m_3 = struct[0].y_run[19,0]
    p_r_2 = struct[0].y_run[20,0]
    p_r_3 = struct[0].y_run[21,0]
    i_d_ref_1 = struct[0].y_run[22,0]
    i_q_ref_1 = struct[0].y_run[23,0]
    p_g_1_1 = struct[0].y_run[24,0]
    q_g_1_1 = struct[0].y_run[25,0]
    p_d2_1 = struct[0].y_run[26,0]
    e_v_1 = struct[0].y_run[27,0]
    p_sto_1 = struct[0].y_run[28,0]
    p_m_1 = struct[0].y_run[29,0]
    omega_coi = struct[0].y_run[30,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_2*delta_2 + Omega_b_2*(omega_2 - omega_coi)
        struct[0].f[1,0] = (-D_2*(omega_2 - omega_coi) - i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) - i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2)) + p_m_2)/(2*H_2)
        struct[0].f[2,0] = (-e1q_2 - i_d_2*(-X1d_2 + X_d_2) + v_f_2)/T1d0_2
        struct[0].f[3,0] = (-e1d_2 + i_q_2*(-X1q_2 + X_q_2))/T1q0_2
        struct[0].f[4,0] = (V_2 - v_c_2)/T_r_2
        struct[0].f[5,0] = -V_2 + v_ref_2
        struct[0].f[6,0] = (p_m_ref_2 - x_gov_1_2)/T_gov_1_2
        struct[0].f[7,0] = (x_gov_1_2 - x_gov_2_2)/T_gov_3_2
        struct[0].f[8,0] = K_imw_2*(p_c_2 - p_g_2_1) - 1.0e-6*xi_imw_2
        struct[0].f[9,0] = -K_delta_3*delta_3 + Omega_b_3*(omega_3 - omega_coi)
        struct[0].f[10,0] = (-D_3*(omega_3 - omega_coi) - i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) - i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3)) + p_m_3)/(2*H_3)
        struct[0].f[11,0] = (-e1q_3 - i_d_3*(-X1d_3 + X_d_3) + v_f_3)/T1d0_3
        struct[0].f[12,0] = (-e1d_3 + i_q_3*(-X1q_3 + X_q_3))/T1q0_3
        struct[0].f[13,0] = (V_3 - v_c_3)/T_r_3
        struct[0].f[14,0] = -V_3 + v_ref_3
        struct[0].f[15,0] = (p_m_ref_3 - x_gov_1_3)/T_gov_1_3
        struct[0].f[16,0] = (x_gov_1_3 - x_gov_2_3)/T_gov_3_3
        struct[0].f[17,0] = K_imw_3*(p_c_3 - p_g_3_1) - 1.0e-6*xi_imw_3
        struct[0].f[18,0] = 1 - omega_coi
        struct[0].f[19,0] = D3_1*(-p_g_1_1 + p_m_1) - K_delta_1*delta_1 + Omega_b_1*(-omega_coi + omega_v_1)
        struct[0].f[20,0] = (-D1_1*(omega_v_1 - 1) - p_d2_1 - p_g_1_1 + p_m_1)/(2*H_1)
        struct[0].f[21,0] = (omega_v_1 - x_wo_1 - 1.0)/T_wo_1
        struct[0].f[22,0] = (-i_d_1 + i_d_ref_1)/T_i_1
        struct[0].f[23,0] = (-i_q_1 + i_q_ref_1)/T_i_1
        struct[0].f[24,0] = -q_g_1_1 + q_ref_1
        struct[0].f[25,0] = -p_sto_1/H_s_1
        struct[0].f[26,0] = -soc_1 + soc_ref_1
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_1/S_base + V_1**2*g_1_2 + V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) - S_n_1*p_g_1_1/S_base
        struct[0].g[1,0] = -Q_1/S_base + V_1**2*(-b_1_2 - bs_1_2/2) + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) - S_n_1*q_g_1_1/S_base
        struct[0].g[2,0] = -P_2/S_base + V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2**2*(g_1_2 + g_2_3) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) - S_n_2*p_g_2_1/S_base
        struct[0].g[3,0] = -Q_2/S_base + V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + V_2**2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3)) - S_n_2*q_g_2_1/S_base
        struct[0].g[4,0] = -P_3/S_base + V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + V_3**2*g_2_3 - S_n_3*p_g_3_1/S_base
        struct[0].g[5,0] = -Q_3/S_base + V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + V_3**2*(-b_2_3 - bs_2_3/2) - S_n_3*q_g_3_1/S_base
        struct[0].g[6,0] = R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2) + X1d_2*i_d_2 - e1q_2
        struct[0].g[7,0] = R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2) - X1q_2*i_q_2 - e1d_2
        struct[0].g[8,0] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2) - p_g_2_1
        struct[0].g[9,0] = V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2) - q_g_2_1
        struct[0].g[10,0] = K_a_2*(-v_c_2 + v_pss_2 + v_ref_2) + K_ai_2*xi_v_2 - v_f_2
        struct[0].g[11,0] = p_c_2 - p_m_ref_2 + p_r_2 + xi_imw_2 - (omega_2 - omega_ref_2)/Droop_2
        struct[0].g[12,0] = T_gov_2_2*(x_gov_1_2 - x_gov_2_2)/T_gov_3_2 - p_m_2 + x_gov_2_2
        struct[0].g[13,0] = R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3) + X1d_3*i_d_3 - e1q_3
        struct[0].g[14,0] = R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3) - X1q_3*i_q_3 - e1d_3
        struct[0].g[15,0] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3) - p_g_3_1
        struct[0].g[16,0] = V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3) - q_g_3_1
        struct[0].g[17,0] = K_a_3*(-v_c_3 + v_pss_3 + v_ref_3) + K_ai_3*xi_v_3 - v_f_3
        struct[0].g[18,0] = p_c_3 - p_m_ref_3 + p_r_3 + xi_imw_3 - (omega_3 - omega_ref_3)/Droop_3
        struct[0].g[19,0] = T_gov_2_3*(x_gov_1_3 - x_gov_2_3)/T_gov_3_3 - p_m_3 + x_gov_2_3
        struct[0].g[20,0] = K_sec_2*xi_freq/2 - p_r_2
        struct[0].g[21,0] = K_sec_3*xi_freq/2 - p_r_3
        struct[0].g[22,0] = R_v_1*i_q_ref_1 + V_1*cos(delta_1 - theta_1) + X_v_1*i_d_ref_1 - e_v_1
        struct[0].g[23,0] = R_v_1*i_d_ref_1 + V_1*sin(delta_1 - theta_1) - X_v_1*i_q_ref_1
        struct[0].g[24,0] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1) - p_g_1_1
        struct[0].g[25,0] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1) - q_g_1_1
        struct[0].g[26,0] = D2_1*(omega_v_1 - x_wo_1 - 1.0) - p_d2_1
        struct[0].g[27,0] = K_q_1*(-q_g_1_1 + q_ref_1 + xi_q_1/T_q_1) - e_v_1
        struct[0].g[28,0] = -i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) - i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1)) + p_src_1 + p_sto_1
        struct[0].g[29,0] = Dp_ref_1 - p_m_1 + p_src_1 + Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
        struct[0].g[30,0] = omega_2/2 + omega_3/2 - omega_coi
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_1
        struct[0].h[1,0] = V_2
        struct[0].h[2,0] = V_3
        struct[0].h[3,0] = i_d_2*(R_a_2*i_d_2 + V_2*sin(delta_2 - theta_2)) + i_q_2*(R_a_2*i_q_2 + V_2*cos(delta_2 - theta_2))
        struct[0].h[4,0] = i_d_3*(R_a_3*i_d_3 + V_3*sin(delta_3 - theta_3)) + i_q_3*(R_a_3*i_q_3 + V_3*cos(delta_3 - theta_3))
        struct[0].h[5,0] = i_d_1*(R_s_1*i_d_1 + V_1*sin(delta_1 - theta_1)) + i_q_1*(R_s_1*i_q_1 + V_1*cos(delta_1 - theta_1))
        struct[0].h[6,0] = Piecewise(np.array([(0.0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_i_soc_1*xi_soc_1 + K_p_soc_1*(-soc_1 + soc_ref_1), True)]))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_2
        struct[0].Fx[0,1] = Omega_b_2
        struct[0].Fx[1,0] = (-V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fx[1,1] = -D_2/(2*H_2)
        struct[0].Fx[2,2] = -1/T1d0_2
        struct[0].Fx[3,3] = -1/T1q0_2
        struct[0].Fx[4,4] = -1/T_r_2
        struct[0].Fx[6,6] = -1/T_gov_1_2
        struct[0].Fx[7,6] = 1/T_gov_3_2
        struct[0].Fx[7,7] = -1/T_gov_3_2
        struct[0].Fx[8,8] = -0.00000100000000000000
        struct[0].Fx[9,9] = -K_delta_3
        struct[0].Fx[9,10] = Omega_b_3
        struct[0].Fx[10,9] = (-V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fx[10,10] = -D_3/(2*H_3)
        struct[0].Fx[11,11] = -1/T1d0_3
        struct[0].Fx[12,12] = -1/T1q0_3
        struct[0].Fx[13,13] = -1/T_r_3
        struct[0].Fx[15,15] = -1/T_gov_1_3
        struct[0].Fx[16,15] = 1/T_gov_3_3
        struct[0].Fx[16,16] = -1/T_gov_3_3
        struct[0].Fx[17,17] = -0.00000100000000000000
        struct[0].Fx[19,19] = -K_delta_1
        struct[0].Fx[19,20] = Omega_b_1
        struct[0].Fx[20,20] = -D1_1/(2*H_1)
        struct[0].Fx[21,20] = 1/T_wo_1
        struct[0].Fx[21,21] = -1/T_wo_1
        struct[0].Fx[22,22] = -1/T_i_1
        struct[0].Fx[23,23] = -1/T_i_1
        struct[0].Fx[26,25] = -1

    if mode == 11:

        struct[0].Fy[0,30] = -Omega_b_2
        struct[0].Fy[1,2] = (-i_d_2*sin(delta_2 - theta_2) - i_q_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,3] = (V_2*i_d_2*cos(delta_2 - theta_2) - V_2*i_q_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,6] = (-2*R_a_2*i_d_2 - V_2*sin(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,7] = (-2*R_a_2*i_q_2 - V_2*cos(delta_2 - theta_2))/(2*H_2)
        struct[0].Fy[1,12] = 1/(2*H_2)
        struct[0].Fy[1,30] = D_2/(2*H_2)
        struct[0].Fy[2,6] = (X1d_2 - X_d_2)/T1d0_2
        struct[0].Fy[2,10] = 1/T1d0_2
        struct[0].Fy[3,7] = (-X1q_2 + X_q_2)/T1q0_2
        struct[0].Fy[4,2] = 1/T_r_2
        struct[0].Fy[5,2] = -1
        struct[0].Fy[6,11] = 1/T_gov_1_2
        struct[0].Fy[8,8] = -K_imw_2
        struct[0].Fy[9,30] = -Omega_b_3
        struct[0].Fy[10,4] = (-i_d_3*sin(delta_3 - theta_3) - i_q_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,5] = (V_3*i_d_3*cos(delta_3 - theta_3) - V_3*i_q_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,13] = (-2*R_a_3*i_d_3 - V_3*sin(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,14] = (-2*R_a_3*i_q_3 - V_3*cos(delta_3 - theta_3))/(2*H_3)
        struct[0].Fy[10,19] = 1/(2*H_3)
        struct[0].Fy[10,30] = D_3/(2*H_3)
        struct[0].Fy[11,13] = (X1d_3 - X_d_3)/T1d0_3
        struct[0].Fy[11,17] = 1/T1d0_3
        struct[0].Fy[12,14] = (-X1q_3 + X_q_3)/T1q0_3
        struct[0].Fy[13,4] = 1/T_r_3
        struct[0].Fy[14,4] = -1
        struct[0].Fy[15,18] = 1/T_gov_1_3
        struct[0].Fy[17,15] = -K_imw_3
        struct[0].Fy[18,30] = -1
        struct[0].Fy[19,24] = -D3_1
        struct[0].Fy[19,29] = D3_1
        struct[0].Fy[19,30] = -Omega_b_1
        struct[0].Fy[20,24] = -1/(2*H_1)
        struct[0].Fy[20,26] = -1/(2*H_1)
        struct[0].Fy[20,29] = 1/(2*H_1)
        struct[0].Fy[22,22] = 1/T_i_1
        struct[0].Fy[23,23] = 1/T_i_1
        struct[0].Fy[24,25] = -1
        struct[0].Fy[25,28] = -1/H_s_1

        struct[0].Gy[0,0] = 2*V_1*g_1_2 + V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[0,1] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[0,2] = V_1*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[0,3] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[0,24] = -S_n_1/S_base
        struct[0].Gy[1,0] = 2*V_1*(-b_1_2 - bs_1_2/2) + V_2*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[1,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[1,2] = V_1*(b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[1,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[1,25] = -S_n_1/S_base
        struct[0].Gy[2,0] = V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[2,1] = V_1*V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[2,2] = V_1*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + 2*V_2*(g_1_2 + g_2_3) + V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[2,3] = V_1*V_2*(-b_1_2*cos(theta_1 - theta_2) - g_1_2*sin(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[2,4] = V_2*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[2,5] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[2,8] = -S_n_2/S_base
        struct[0].Gy[3,0] = V_2*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2))
        struct[0].Gy[3,1] = V_1*V_2*(-b_1_2*sin(theta_1 - theta_2) + g_1_2*cos(theta_1 - theta_2))
        struct[0].Gy[3,2] = V_1*(b_1_2*cos(theta_1 - theta_2) + g_1_2*sin(theta_1 - theta_2)) + 2*V_2*(-b_1_2 - b_2_3 - bs_1_2/2 - bs_2_3/2) + V_3*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[3,3] = V_1*V_2*(b_1_2*sin(theta_1 - theta_2) - g_1_2*cos(theta_1 - theta_2)) + V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[3,4] = V_2*(b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[3,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[3,9] = -S_n_2/S_base
        struct[0].Gy[4,2] = V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[4,3] = V_2*V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[4,4] = V_2*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3)) + 2*V_3*g_2_3
        struct[0].Gy[4,5] = V_2*V_3*(-b_2_3*cos(theta_2 - theta_3) - g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[4,15] = -S_n_3/S_base
        struct[0].Gy[5,2] = V_3*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3))
        struct[0].Gy[5,3] = V_2*V_3*(-b_2_3*sin(theta_2 - theta_3) + g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[5,4] = V_2*(b_2_3*cos(theta_2 - theta_3) + g_2_3*sin(theta_2 - theta_3)) + 2*V_3*(-b_2_3 - bs_2_3/2)
        struct[0].Gy[5,5] = V_2*V_3*(b_2_3*sin(theta_2 - theta_3) - g_2_3*cos(theta_2 - theta_3))
        struct[0].Gy[5,16] = -S_n_3/S_base
        struct[0].Gy[6,2] = cos(delta_2 - theta_2)
        struct[0].Gy[6,3] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[6,6] = X1d_2
        struct[0].Gy[6,7] = R_a_2
        struct[0].Gy[7,2] = sin(delta_2 - theta_2)
        struct[0].Gy[7,3] = -V_2*cos(delta_2 - theta_2)
        struct[0].Gy[7,6] = R_a_2
        struct[0].Gy[7,7] = -X1q_2
        struct[0].Gy[8,2] = i_d_2*sin(delta_2 - theta_2) + i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[8,3] = -V_2*i_d_2*cos(delta_2 - theta_2) + V_2*i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[8,6] = V_2*sin(delta_2 - theta_2)
        struct[0].Gy[8,7] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[8,8] = -1
        struct[0].Gy[9,2] = i_d_2*cos(delta_2 - theta_2) - i_q_2*sin(delta_2 - theta_2)
        struct[0].Gy[9,3] = V_2*i_d_2*sin(delta_2 - theta_2) + V_2*i_q_2*cos(delta_2 - theta_2)
        struct[0].Gy[9,6] = V_2*cos(delta_2 - theta_2)
        struct[0].Gy[9,7] = -V_2*sin(delta_2 - theta_2)
        struct[0].Gy[9,9] = -1
        struct[0].Gy[10,10] = -1
        struct[0].Gy[11,11] = -1
        struct[0].Gy[11,20] = 1
        struct[0].Gy[12,12] = -1
        struct[0].Gy[13,4] = cos(delta_3 - theta_3)
        struct[0].Gy[13,5] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[13,13] = X1d_3
        struct[0].Gy[13,14] = R_a_3
        struct[0].Gy[14,4] = sin(delta_3 - theta_3)
        struct[0].Gy[14,5] = -V_3*cos(delta_3 - theta_3)
        struct[0].Gy[14,13] = R_a_3
        struct[0].Gy[14,14] = -X1q_3
        struct[0].Gy[15,4] = i_d_3*sin(delta_3 - theta_3) + i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[15,5] = -V_3*i_d_3*cos(delta_3 - theta_3) + V_3*i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[15,13] = V_3*sin(delta_3 - theta_3)
        struct[0].Gy[15,14] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[15,15] = -1
        struct[0].Gy[16,4] = i_d_3*cos(delta_3 - theta_3) - i_q_3*sin(delta_3 - theta_3)
        struct[0].Gy[16,5] = V_3*i_d_3*sin(delta_3 - theta_3) + V_3*i_q_3*cos(delta_3 - theta_3)
        struct[0].Gy[16,13] = V_3*cos(delta_3 - theta_3)
        struct[0].Gy[16,14] = -V_3*sin(delta_3 - theta_3)
        struct[0].Gy[16,16] = -1
        struct[0].Gy[17,17] = -1
        struct[0].Gy[18,18] = -1
        struct[0].Gy[18,21] = 1
        struct[0].Gy[19,19] = -1
        struct[0].Gy[20,20] = -1
        struct[0].Gy[21,21] = -1
        struct[0].Gy[22,0] = cos(delta_1 - theta_1)
        struct[0].Gy[22,1] = V_1*sin(delta_1 - theta_1)
        struct[0].Gy[22,22] = X_v_1
        struct[0].Gy[22,23] = R_v_1
        struct[0].Gy[22,27] = -1
        struct[0].Gy[23,0] = sin(delta_1 - theta_1)
        struct[0].Gy[23,1] = -V_1*cos(delta_1 - theta_1)
        struct[0].Gy[23,22] = R_v_1
        struct[0].Gy[23,23] = -X_v_1
        struct[0].Gy[24,0] = i_d_1*sin(delta_1 - theta_1) + i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[24,1] = -V_1*i_d_1*cos(delta_1 - theta_1) + V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[24,24] = -1
        struct[0].Gy[25,0] = i_d_1*cos(delta_1 - theta_1) - i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[25,1] = V_1*i_d_1*sin(delta_1 - theta_1) + V_1*i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[25,25] = -1
        struct[0].Gy[26,26] = -1
        struct[0].Gy[27,25] = -K_q_1
        struct[0].Gy[27,27] = -1
        struct[0].Gy[28,0] = -i_d_1*sin(delta_1 - theta_1) - i_q_1*cos(delta_1 - theta_1)
        struct[0].Gy[28,1] = V_1*i_d_1*cos(delta_1 - theta_1) - V_1*i_q_1*sin(delta_1 - theta_1)
        struct[0].Gy[28,28] = 1
        struct[0].Gy[29,29] = -1
        struct[0].Gy[30,30] = -1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[10,6] = K_a_2
        struct[0].Gu[10,7] = K_a_2
        struct[0].Gu[11,8] = 1
        struct[0].Gu[17,9] = K_a_3
        struct[0].Gu[17,10] = K_a_3
        struct[0].Gu[18,11] = 1
        struct[0].Gu[27,14] = K_q_1
        struct[0].Gu[28,15] = 1
        struct[0].Gu[29,13] = 1
        struct[0].Gu[29,15] = 1
        struct[0].Gu[29,16] = Piecewise(np.array([(0, ((p_sto_1 > 0.0) | (soc_1 > 1.0)) & ((p_sto_1 > 0.0) | (p_sto_1 < 0.0)) & ((soc_1 > 1.0) | (soc_1 < 0.0)) & ((p_sto_1 < 0.0) | (soc_1 < 0.0))), (K_p_soc_1, True)]))





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
    Fx_ini_rows = [0, 0, 1, 1, 2, 3, 4, 6, 7, 7, 8, 9, 9, 10, 10, 11, 12, 13, 15, 16, 16, 17, 19, 19, 20, 21, 21, 22, 23, 26]

    Fx_ini_cols = [0, 1, 0, 1, 2, 3, 4, 6, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 15, 15, 16, 17, 19, 20, 20, 20, 21, 22, 23, 25]

    Fy_ini_rows = [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 9, 10, 10, 10, 10, 10, 10, 11, 11, 12, 13, 14, 15, 17, 18, 19, 19, 19, 20, 20, 20, 22, 23, 24, 25]

    Fy_ini_cols = [30, 2, 3, 6, 7, 12, 30, 6, 10, 7, 2, 2, 11, 8, 30, 4, 5, 13, 14, 19, 30, 13, 17, 14, 4, 4, 18, 15, 30, 24, 29, 30, 24, 26, 29, 22, 23, 25, 28]

    Gx_ini_rows = [6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 17, 17, 18, 18, 19, 19, 20, 21, 22, 23, 24, 24, 24, 25, 25, 25, 26, 26, 27, 28, 28, 28, 29, 29, 30, 30]

    Gx_ini_cols = [0, 2, 0, 3, 0, 0, 4, 5, 1, 8, 6, 7, 9, 11, 9, 12, 9, 9, 13, 14, 10, 17, 15, 16, 18, 18, 19, 19, 19, 22, 23, 19, 22, 23, 20, 21, 24, 19, 22, 23, 25, 26, 1, 10]

    Gy_ini_rows = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 11, 11, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 18, 18, 19, 20, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 27, 27, 28, 28, 28, 29, 30]

    Gy_ini_cols = [0, 1, 2, 3, 24, 0, 1, 2, 3, 25, 0, 1, 2, 3, 4, 5, 8, 0, 1, 2, 3, 4, 5, 9, 2, 3, 4, 5, 15, 2, 3, 4, 5, 16, 2, 3, 6, 7, 2, 3, 6, 7, 2, 3, 6, 7, 8, 2, 3, 6, 7, 9, 10, 11, 20, 12, 4, 5, 13, 14, 4, 5, 13, 14, 4, 5, 13, 14, 15, 4, 5, 13, 14, 16, 17, 18, 21, 19, 20, 21, 0, 1, 22, 23, 27, 0, 1, 22, 23, 0, 1, 24, 0, 1, 25, 26, 25, 27, 0, 1, 28, 29, 30]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols