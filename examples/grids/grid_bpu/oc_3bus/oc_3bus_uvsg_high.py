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


class oc_3bus_uvsg_high_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 9
        self.N_y = 27 
        self.N_z = 8 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_B1_B2', 'b_B1_B2', 'bs_B1_B2', 'g_B2_B3', 'b_B2_B3', 'bs_B2_B3', 'U_B1_n', 'U_B2_n', 'U_B3_n', 'S_n_B3', 'Omega_b_B3', 'K_p_B3', 'T_p_B3', 'K_q_B3', 'T_q_B3', 'X_v_B3', 'R_v_B3', 'R_s_B3', 'C_u_B3', 'K_u_0_B3', 'K_u_max_B3', 'V_u_min_B3', 'V_u_max_B3', 'R_uc_B3', 'K_h_B3', 'R_lim_B3', 'V_u_lt_B3', 'V_u_ht_B3', 'Droop_B3', 'DB_B3', 'T_cur_B3', 'S_n_B1', 'Omega_b_B1', 'X_v_B1', 'R_v_B1', 'K_delta_B1', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [100000.0, 38.46153846153846, -7.692307692307692, 0.0, 38.46153846153846, -7.692307692307692, 0.0, 400.0, 400.0, 400.0, 50000.0, 314.1592653589793, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.02, 5.0, 0.005, 0.1, 80, 160, 0.1, 1.0, 0.2, 85, 155, 0.05, 0.001, 10.0, 50000.0, 314.1592653589793, 0.001, 0.001, 0.001, 0.01, 0.01] 
        self.inputs_ini_list = ['P_B1', 'Q_B1', 'P_B2', 'Q_B2', 'P_B3', 'Q_B3', 'q_s_ref_B3', 'v_u_ref_B3', 'omega_ref_B3', 'p_gin_B3', 'p_g_ref_B3', 'alpha_B1', 'e_qv_B1', 'omega_ref_B1'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0, 1, 1] 
        self.inputs_run_list = ['P_B1', 'Q_B1', 'P_B2', 'Q_B2', 'P_B3', 'Q_B3', 'q_s_ref_B3', 'v_u_ref_B3', 'omega_ref_B3', 'p_gin_B3', 'p_g_ref_B3', 'alpha_B1', 'e_qv_B1', 'omega_ref_B1'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0, 1, 1] 
        self.outputs_list = ['V_B1', 'V_B2', 'V_B3', 'p_gin_B3', 'p_g_ref_B3', 'p_l_B3', 'soc_B3', 'alpha_B1'] 
        self.x_list = ['delta_B3', 'xi_p_B3', 'xi_q_B3', 'e_u_B3', 'p_ghr_B3', 'k_cur_B3', 'delta_B1', 'Domega_B1', 'xi_freq'] 
        self.y_run_list = ['V_B1', 'theta_B1', 'V_B2', 'theta_B2', 'V_B3', 'theta_B3', 'omega_B3', 'e_qv_B3', 'i_d_B3', 'i_q_B3', 'p_s_B3', 'q_s_B3', 'p_m_B3', 'p_t_B3', 'p_u_B3', 'v_u_B3', 'k_u_B3', 'k_cur_sat_B3', 'p_gou_B3', 'p_f_B3', 'omega_B1', 'i_d_B1', 'i_q_B1', 'p_s_B1', 'q_s_B1', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_B1', 'theta_B1', 'V_B2', 'theta_B2', 'V_B3', 'theta_B3', 'omega_B3', 'e_qv_B3', 'i_d_B3', 'i_q_B3', 'p_s_B3', 'q_s_B3', 'p_m_B3', 'p_t_B3', 'p_u_B3', 'v_u_B3', 'k_u_B3', 'k_cur_sat_B3', 'p_gou_B3', 'p_f_B3', 'omega_B1', 'i_d_B1', 'i_q_B1', 'p_s_B1', 'q_s_B1', 'omega_coi', 'p_agc'] 
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
    g_B1_B2 = struct[0].g_B1_B2
    b_B1_B2 = struct[0].b_B1_B2
    bs_B1_B2 = struct[0].bs_B1_B2
    g_B2_B3 = struct[0].g_B2_B3
    b_B2_B3 = struct[0].b_B2_B3
    bs_B2_B3 = struct[0].bs_B2_B3
    U_B1_n = struct[0].U_B1_n
    U_B2_n = struct[0].U_B2_n
    U_B3_n = struct[0].U_B3_n
    S_n_B3 = struct[0].S_n_B3
    Omega_b_B3 = struct[0].Omega_b_B3
    K_p_B3 = struct[0].K_p_B3
    T_p_B3 = struct[0].T_p_B3
    K_q_B3 = struct[0].K_q_B3
    T_q_B3 = struct[0].T_q_B3
    X_v_B3 = struct[0].X_v_B3
    R_v_B3 = struct[0].R_v_B3
    R_s_B3 = struct[0].R_s_B3
    C_u_B3 = struct[0].C_u_B3
    K_u_0_B3 = struct[0].K_u_0_B3
    K_u_max_B3 = struct[0].K_u_max_B3
    V_u_min_B3 = struct[0].V_u_min_B3
    V_u_max_B3 = struct[0].V_u_max_B3
    R_uc_B3 = struct[0].R_uc_B3
    K_h_B3 = struct[0].K_h_B3
    R_lim_B3 = struct[0].R_lim_B3
    V_u_lt_B3 = struct[0].V_u_lt_B3
    V_u_ht_B3 = struct[0].V_u_ht_B3
    Droop_B3 = struct[0].Droop_B3
    DB_B3 = struct[0].DB_B3
    T_cur_B3 = struct[0].T_cur_B3
    S_n_B1 = struct[0].S_n_B1
    Omega_b_B1 = struct[0].Omega_b_B1
    X_v_B1 = struct[0].X_v_B1
    R_v_B1 = struct[0].R_v_B1
    K_delta_B1 = struct[0].K_delta_B1
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_B1 = struct[0].P_B1
    Q_B1 = struct[0].Q_B1
    P_B2 = struct[0].P_B2
    Q_B2 = struct[0].Q_B2
    P_B3 = struct[0].P_B3
    Q_B3 = struct[0].Q_B3
    q_s_ref_B3 = struct[0].q_s_ref_B3
    v_u_ref_B3 = struct[0].v_u_ref_B3
    omega_ref_B3 = struct[0].omega_ref_B3
    p_gin_B3 = struct[0].p_gin_B3
    p_g_ref_B3 = struct[0].p_g_ref_B3
    alpha_B1 = struct[0].alpha_B1
    e_qv_B1 = struct[0].e_qv_B1
    omega_ref_B1 = struct[0].omega_ref_B1
    
    # Dynamical states:
    delta_B3 = struct[0].x[0,0]
    xi_p_B3 = struct[0].x[1,0]
    xi_q_B3 = struct[0].x[2,0]
    e_u_B3 = struct[0].x[3,0]
    p_ghr_B3 = struct[0].x[4,0]
    k_cur_B3 = struct[0].x[5,0]
    delta_B1 = struct[0].x[6,0]
    Domega_B1 = struct[0].x[7,0]
    xi_freq = struct[0].x[8,0]
    
    # Algebraic states:
    V_B1 = struct[0].y_ini[0,0]
    theta_B1 = struct[0].y_ini[1,0]
    V_B2 = struct[0].y_ini[2,0]
    theta_B2 = struct[0].y_ini[3,0]
    V_B3 = struct[0].y_ini[4,0]
    theta_B3 = struct[0].y_ini[5,0]
    omega_B3 = struct[0].y_ini[6,0]
    e_qv_B3 = struct[0].y_ini[7,0]
    i_d_B3 = struct[0].y_ini[8,0]
    i_q_B3 = struct[0].y_ini[9,0]
    p_s_B3 = struct[0].y_ini[10,0]
    q_s_B3 = struct[0].y_ini[11,0]
    p_m_B3 = struct[0].y_ini[12,0]
    p_t_B3 = struct[0].y_ini[13,0]
    p_u_B3 = struct[0].y_ini[14,0]
    v_u_B3 = struct[0].y_ini[15,0]
    k_u_B3 = struct[0].y_ini[16,0]
    k_cur_sat_B3 = struct[0].y_ini[17,0]
    p_gou_B3 = struct[0].y_ini[18,0]
    p_f_B3 = struct[0].y_ini[19,0]
    omega_B1 = struct[0].y_ini[20,0]
    i_d_B1 = struct[0].y_ini[21,0]
    i_q_B1 = struct[0].y_ini[22,0]
    p_s_B1 = struct[0].y_ini[23,0]
    q_s_B1 = struct[0].y_ini[24,0]
    omega_coi = struct[0].y_ini[25,0]
    p_agc = struct[0].y_ini[26,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_B3*(omega_B3 - omega_coi)
        struct[0].f[1,0] = p_m_B3 - p_s_B3
        struct[0].f[2,0] = -q_s_B3 + q_s_ref_B3
        struct[0].f[3,0] = S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-R_lim_B3, R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3)), (R_lim_B3, R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)), (K_h_B3*(-p_ghr_B3 + p_gou_B3), True)]))
        struct[0].f[5,0] = (-k_cur_B3 + p_f_B3/p_gin_B3 + p_g_ref_B3/p_gin_B3)/T_cur_B3
        struct[0].f[6,0] = -K_delta_B1*delta_B1 + Omega_b_B1*(omega_B1 - omega_coi)
        struct[0].f[7,0] = alpha_B1
        struct[0].f[8,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_B1/S_base + V_B1**2*g_B1_B2 + V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) - S_n_B1*p_s_B1/S_base
        struct[0].g[1,0] = -Q_B1/S_base + V_B1**2*(-b_B1_B2 - bs_B1_B2/2) + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) - S_n_B1*q_s_B1/S_base
        struct[0].g[2,0] = -P_B2/S_base + V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2**2*(g_B1_B2 + g_B2_B3) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].g[3,0] = -Q_B2/S_base + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2**2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].g[4,0] = -P_B3/S_base + V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + V_B3**2*g_B2_B3 - S_n_B3*p_s_B3/S_base
        struct[0].g[5,0] = -Q_B3/S_base + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + V_B3**2*(-b_B2_B3 - bs_B2_B3/2) - S_n_B3*q_s_B3/S_base
        struct[0].g[6,0] = K_p_B3*(p_m_B3 - p_s_B3 + xi_p_B3/T_p_B3) - omega_B3
        struct[0].g[7,0] = K_q_B3*(-q_s_B3 + q_s_ref_B3 + xi_q_B3/T_q_B3) - e_qv_B3
        struct[0].g[8,0] = -R_v_B3*i_d_B3 - V_B3*sin(delta_B3 - theta_B3) + X_v_B3*i_q_B3
        struct[0].g[9,0] = -R_v_B3*i_q_B3 - V_B3*cos(delta_B3 - theta_B3) - X_v_B3*i_d_B3 + e_qv_B3
        struct[0].g[10,0] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3) - p_s_B3
        struct[0].g[11,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3) - q_s_B3
        struct[0].g[12,0] = p_ghr_B3 - p_m_B3 + p_s_B3 - p_t_B3 + p_u_B3
        struct[0].g[13,0] = i_d_B3*(R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)) + i_q_B3*(R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)) - p_t_B3
        struct[0].g[14,0] = -p_u_B3 - k_u_B3*(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].g[15,0] = R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1) + e_u_B3 - v_u_B3
        struct[0].g[16,0] = -k_u_B3 + Piecewise(np.array([(K_u_max_B3, V_u_min_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_lt_B3 + v_u_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_ht_B3 + v_u_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (K_u_max_B3, V_u_max_B3 < v_u_B3), (K_u_0_B3, True)]))
        struct[0].g[17,0] = -k_cur_sat_B3 + Piecewise(np.array([(0.0001, k_cur_B3 < 0.0001), (1, k_cur_B3 > 1), (k_cur_B3, True)]))
        struct[0].g[19,0] = -p_f_B3 - Piecewise(np.array([((0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 < -0.5*DB_B3 + omega_ref_B3), ((-0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 > 0.5*DB_B3 + omega_ref_B3), (0.0, True)]))
        struct[0].g[20,0] = Domega_B1 - omega_B1 + omega_ref_B1
        struct[0].g[21,0] = -R_v_B1*i_d_B1 - V_B1*sin(delta_B1 - theta_B1) - X_v_B1*i_q_B1
        struct[0].g[22,0] = -R_v_B1*i_q_B1 - V_B1*cos(delta_B1 - theta_B1) + X_v_B1*i_d_B1 + e_qv_B1
        struct[0].g[23,0] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1) - p_s_B1
        struct[0].g[24,0] = V_B1*i_d_B1*cos(delta_B1 - theta_B1) - V_B1*i_q_B1*sin(delta_B1 - theta_B1) - q_s_B1
        struct[0].g[25,0] = -omega_coi + (1000000.0*S_n_B1*omega_B1 + S_n_B3*T_p_B3*omega_B3/(2*K_p_B3))/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].g[26,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_B1
        struct[0].h[1,0] = V_B2
        struct[0].h[2,0] = V_B3
        struct[0].h[3,0] = p_gin_B3
        struct[0].h[4,0] = p_g_ref_B3
        struct[0].h[5,0] = -p_s_B3 + p_t_B3
        struct[0].h[6,0] = (-V_u_min_B3**2 + e_u_B3**2)/(V_u_max_B3**2 - V_u_min_B3**2)
        struct[0].h[7,0] = alpha_B1
    

    if mode == 10:

        struct[0].Fx_ini[4,4] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (-K_h_B3, True)]))
        struct[0].Fx_ini[5,5] = -1/T_cur_B3
        struct[0].Fx_ini[6,6] = -K_delta_B1

    if mode == 11:

        struct[0].Fy_ini[0,6] = Omega_b_B3 
        struct[0].Fy_ini[0,25] = -Omega_b_B3 
        struct[0].Fy_ini[1,10] = -1 
        struct[0].Fy_ini[1,12] = 1 
        struct[0].Fy_ini[2,11] = -1 
        struct[0].Fy_ini[3,13] = -S_n_B3/(C_u_B3*(v_u_B3 + 0.1)) 
        struct[0].Fy_ini[3,15] = -S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1)**2) 
        struct[0].Fy_ini[3,18] = S_n_B3/(C_u_B3*(v_u_B3 + 0.1)) 
        struct[0].Fy_ini[4,18] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (K_h_B3, True)])) 
        struct[0].Fy_ini[5,19] = 1/(T_cur_B3*p_gin_B3) 
        struct[0].Fy_ini[6,20] = Omega_b_B1 
        struct[0].Fy_ini[6,25] = -Omega_b_B1 
        struct[0].Fy_ini[8,25] = -1 

        struct[0].Gx_ini[6,1] = K_p_B3/T_p_B3
        struct[0].Gx_ini[7,2] = K_q_B3/T_q_B3
        struct[0].Gx_ini[8,0] = -V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gx_ini[9,0] = V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gx_ini[10,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gx_ini[11,0] = -V_B3*i_d_B3*sin(delta_B3 - theta_B3) - V_B3*i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gx_ini[12,4] = 1
        struct[0].Gx_ini[13,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gx_ini[15,3] = 1
        struct[0].Gx_ini[17,5] = Piecewise(np.array([(0, (k_cur_B3 > 1) | (k_cur_B3 < 0.0001)), (1, True)]))
        struct[0].Gx_ini[20,7] = 1
        struct[0].Gx_ini[21,6] = -V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gx_ini[22,6] = V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gx_ini[23,6] = V_B1*i_d_B1*cos(delta_B1 - theta_B1) - V_B1*i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gx_ini[24,6] = -V_B1*i_d_B1*sin(delta_B1 - theta_B1) - V_B1*i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gx_ini[26,8] = K_i_agc

        struct[0].Gy_ini[0,0] = 2*V_B1*g_B1_B2 + V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,1] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,2] = V_B1*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,3] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,23] = -S_n_B1/S_base
        struct[0].Gy_ini[1,0] = 2*V_B1*(-b_B1_B2 - bs_B1_B2/2) + V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,24] = -S_n_B1/S_base
        struct[0].Gy_ini[2,0] = V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[2,1] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[2,2] = V_B1*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + 2*V_B2*(g_B1_B2 + g_B2_B3) + V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[2,3] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[2,4] = V_B2*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[2,5] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,0] = V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[3,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[3,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + 2*V_B2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,2] = V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,3] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,4] = V_B2*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + 2*V_B3*g_B2_B3
        struct[0].Gy_ini[4,5] = V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,10] = -S_n_B3/S_base
        struct[0].Gy_ini[5,2] = V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[5,3] = V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[5,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + 2*V_B3*(-b_B2_B3 - bs_B2_B3/2)
        struct[0].Gy_ini[5,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[5,11] = -S_n_B3/S_base
        struct[0].Gy_ini[6,10] = -K_p_B3
        struct[0].Gy_ini[6,12] = K_p_B3
        struct[0].Gy_ini[7,11] = -K_q_B3
        struct[0].Gy_ini[8,4] = -sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[8,5] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[8,8] = -R_v_B3
        struct[0].Gy_ini[8,9] = X_v_B3
        struct[0].Gy_ini[9,4] = -cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[9,5] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[9,8] = -X_v_B3
        struct[0].Gy_ini[9,9] = -R_v_B3
        struct[0].Gy_ini[10,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[10,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[10,8] = V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[10,9] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,4] = i_d_B3*cos(delta_B3 - theta_B3) - i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,5] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,8] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,9] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,8] = 2*R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,9] = 2*R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[14,15] = 2*k_u_B3*v_u_B3/V_u_max_B3**2
        struct[0].Gy_ini[14,16] = -(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].Gy_ini[15,13] = -R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy_ini[15,15] = -R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1)**2 - 1
        struct[0].Gy_ini[15,18] = R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy_ini[16,15] = Piecewise(np.array([(0, V_u_min_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (0, True)]))
        struct[0].Gy_ini[18,17] = p_gin_B3
        struct[0].Gy_ini[19,6] = -Piecewise(np.array([(1/Droop_B3, (omega_B3 > 0.5*DB_B3 + omega_ref_B3) | (omega_B3 < -0.5*DB_B3 + omega_ref_B3)), (0, True)]))
        struct[0].Gy_ini[21,0] = -sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[21,1] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[21,21] = -R_v_B1
        struct[0].Gy_ini[21,22] = -X_v_B1
        struct[0].Gy_ini[22,0] = -cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[22,1] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[22,21] = X_v_B1
        struct[0].Gy_ini[22,22] = -R_v_B1
        struct[0].Gy_ini[23,0] = i_d_B1*sin(delta_B1 - theta_B1) + i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[23,1] = -V_B1*i_d_B1*cos(delta_B1 - theta_B1) + V_B1*i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[23,21] = V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[23,22] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,0] = i_d_B1*cos(delta_B1 - theta_B1) - i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,1] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,21] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,22] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[25,6] = S_n_B3*T_p_B3/(2*K_p_B3*(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3)))
        struct[0].Gy_ini[25,20] = 1000000.0*S_n_B1/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].Gy_ini[26,25] = -K_p_agc



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_B1_B2 = struct[0].g_B1_B2
    b_B1_B2 = struct[0].b_B1_B2
    bs_B1_B2 = struct[0].bs_B1_B2
    g_B2_B3 = struct[0].g_B2_B3
    b_B2_B3 = struct[0].b_B2_B3
    bs_B2_B3 = struct[0].bs_B2_B3
    U_B1_n = struct[0].U_B1_n
    U_B2_n = struct[0].U_B2_n
    U_B3_n = struct[0].U_B3_n
    S_n_B3 = struct[0].S_n_B3
    Omega_b_B3 = struct[0].Omega_b_B3
    K_p_B3 = struct[0].K_p_B3
    T_p_B3 = struct[0].T_p_B3
    K_q_B3 = struct[0].K_q_B3
    T_q_B3 = struct[0].T_q_B3
    X_v_B3 = struct[0].X_v_B3
    R_v_B3 = struct[0].R_v_B3
    R_s_B3 = struct[0].R_s_B3
    C_u_B3 = struct[0].C_u_B3
    K_u_0_B3 = struct[0].K_u_0_B3
    K_u_max_B3 = struct[0].K_u_max_B3
    V_u_min_B3 = struct[0].V_u_min_B3
    V_u_max_B3 = struct[0].V_u_max_B3
    R_uc_B3 = struct[0].R_uc_B3
    K_h_B3 = struct[0].K_h_B3
    R_lim_B3 = struct[0].R_lim_B3
    V_u_lt_B3 = struct[0].V_u_lt_B3
    V_u_ht_B3 = struct[0].V_u_ht_B3
    Droop_B3 = struct[0].Droop_B3
    DB_B3 = struct[0].DB_B3
    T_cur_B3 = struct[0].T_cur_B3
    S_n_B1 = struct[0].S_n_B1
    Omega_b_B1 = struct[0].Omega_b_B1
    X_v_B1 = struct[0].X_v_B1
    R_v_B1 = struct[0].R_v_B1
    K_delta_B1 = struct[0].K_delta_B1
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_B1 = struct[0].P_B1
    Q_B1 = struct[0].Q_B1
    P_B2 = struct[0].P_B2
    Q_B2 = struct[0].Q_B2
    P_B3 = struct[0].P_B3
    Q_B3 = struct[0].Q_B3
    q_s_ref_B3 = struct[0].q_s_ref_B3
    v_u_ref_B3 = struct[0].v_u_ref_B3
    omega_ref_B3 = struct[0].omega_ref_B3
    p_gin_B3 = struct[0].p_gin_B3
    p_g_ref_B3 = struct[0].p_g_ref_B3
    alpha_B1 = struct[0].alpha_B1
    e_qv_B1 = struct[0].e_qv_B1
    omega_ref_B1 = struct[0].omega_ref_B1
    
    # Dynamical states:
    delta_B3 = struct[0].x[0,0]
    xi_p_B3 = struct[0].x[1,0]
    xi_q_B3 = struct[0].x[2,0]
    e_u_B3 = struct[0].x[3,0]
    p_ghr_B3 = struct[0].x[4,0]
    k_cur_B3 = struct[0].x[5,0]
    delta_B1 = struct[0].x[6,0]
    Domega_B1 = struct[0].x[7,0]
    xi_freq = struct[0].x[8,0]
    
    # Algebraic states:
    V_B1 = struct[0].y_run[0,0]
    theta_B1 = struct[0].y_run[1,0]
    V_B2 = struct[0].y_run[2,0]
    theta_B2 = struct[0].y_run[3,0]
    V_B3 = struct[0].y_run[4,0]
    theta_B3 = struct[0].y_run[5,0]
    omega_B3 = struct[0].y_run[6,0]
    e_qv_B3 = struct[0].y_run[7,0]
    i_d_B3 = struct[0].y_run[8,0]
    i_q_B3 = struct[0].y_run[9,0]
    p_s_B3 = struct[0].y_run[10,0]
    q_s_B3 = struct[0].y_run[11,0]
    p_m_B3 = struct[0].y_run[12,0]
    p_t_B3 = struct[0].y_run[13,0]
    p_u_B3 = struct[0].y_run[14,0]
    v_u_B3 = struct[0].y_run[15,0]
    k_u_B3 = struct[0].y_run[16,0]
    k_cur_sat_B3 = struct[0].y_run[17,0]
    p_gou_B3 = struct[0].y_run[18,0]
    p_f_B3 = struct[0].y_run[19,0]
    omega_B1 = struct[0].y_run[20,0]
    i_d_B1 = struct[0].y_run[21,0]
    i_q_B1 = struct[0].y_run[22,0]
    p_s_B1 = struct[0].y_run[23,0]
    q_s_B1 = struct[0].y_run[24,0]
    omega_coi = struct[0].y_run[25,0]
    p_agc = struct[0].y_run[26,0]
    
    struct[0].u_run[0,0] = P_B1
    struct[0].u_run[1,0] = Q_B1
    struct[0].u_run[2,0] = P_B2
    struct[0].u_run[3,0] = Q_B2
    struct[0].u_run[4,0] = P_B3
    struct[0].u_run[5,0] = Q_B3
    struct[0].u_run[6,0] = q_s_ref_B3
    struct[0].u_run[7,0] = v_u_ref_B3
    struct[0].u_run[8,0] = omega_ref_B3
    struct[0].u_run[9,0] = p_gin_B3
    struct[0].u_run[10,0] = p_g_ref_B3
    struct[0].u_run[11,0] = alpha_B1
    struct[0].u_run[12,0] = e_qv_B1
    struct[0].u_run[13,0] = omega_ref_B1
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_B3*(omega_B3 - omega_coi)
        struct[0].f[1,0] = p_m_B3 - p_s_B3
        struct[0].f[2,0] = -q_s_B3 + q_s_ref_B3
        struct[0].f[3,0] = S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-R_lim_B3, R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3)), (R_lim_B3, R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)), (K_h_B3*(-p_ghr_B3 + p_gou_B3), True)]))
        struct[0].f[5,0] = (-k_cur_B3 + p_f_B3/p_gin_B3 + p_g_ref_B3/p_gin_B3)/T_cur_B3
        struct[0].f[6,0] = -K_delta_B1*delta_B1 + Omega_b_B1*(omega_B1 - omega_coi)
        struct[0].f[7,0] = alpha_B1
        struct[0].f[8,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_B1/S_base + V_B1**2*g_B1_B2 + V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) - S_n_B1*p_s_B1/S_base
        struct[0].g[1,0] = -Q_B1/S_base + V_B1**2*(-b_B1_B2 - bs_B1_B2/2) + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) - S_n_B1*q_s_B1/S_base
        struct[0].g[2,0] = -P_B2/S_base + V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2**2*(g_B1_B2 + g_B2_B3) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].g[3,0] = -Q_B2/S_base + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2**2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].g[4,0] = -P_B3/S_base + V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + V_B3**2*g_B2_B3 - S_n_B3*p_s_B3/S_base
        struct[0].g[5,0] = -Q_B3/S_base + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + V_B3**2*(-b_B2_B3 - bs_B2_B3/2) - S_n_B3*q_s_B3/S_base
        struct[0].g[6,0] = K_p_B3*(p_m_B3 - p_s_B3 + xi_p_B3/T_p_B3) - omega_B3
        struct[0].g[7,0] = K_q_B3*(-q_s_B3 + q_s_ref_B3 + xi_q_B3/T_q_B3) - e_qv_B3
        struct[0].g[8,0] = -R_v_B3*i_d_B3 - V_B3*sin(delta_B3 - theta_B3) + X_v_B3*i_q_B3
        struct[0].g[9,0] = -R_v_B3*i_q_B3 - V_B3*cos(delta_B3 - theta_B3) - X_v_B3*i_d_B3 + e_qv_B3
        struct[0].g[10,0] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3) - p_s_B3
        struct[0].g[11,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3) - q_s_B3
        struct[0].g[12,0] = p_ghr_B3 - p_m_B3 + p_s_B3 - p_t_B3 + p_u_B3
        struct[0].g[13,0] = i_d_B3*(R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)) + i_q_B3*(R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)) - p_t_B3
        struct[0].g[14,0] = -p_u_B3 - k_u_B3*(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].g[15,0] = R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1) + e_u_B3 - v_u_B3
        struct[0].g[16,0] = -k_u_B3 + Piecewise(np.array([(K_u_max_B3, V_u_min_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_lt_B3 + v_u_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_ht_B3 + v_u_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (K_u_max_B3, V_u_max_B3 < v_u_B3), (K_u_0_B3, True)]))
        struct[0].g[17,0] = -k_cur_sat_B3 + Piecewise(np.array([(0.0001, k_cur_B3 < 0.0001), (1, k_cur_B3 > 1), (k_cur_B3, True)]))
        struct[0].g[18,0] = k_cur_sat_B3*p_gin_B3 - p_gou_B3
        struct[0].g[19,0] = -p_f_B3 - Piecewise(np.array([((0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 < -0.5*DB_B3 + omega_ref_B3), ((-0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 > 0.5*DB_B3 + omega_ref_B3), (0.0, True)]))
        struct[0].g[20,0] = Domega_B1 - omega_B1 + omega_ref_B1
        struct[0].g[21,0] = -R_v_B1*i_d_B1 - V_B1*sin(delta_B1 - theta_B1) - X_v_B1*i_q_B1
        struct[0].g[22,0] = -R_v_B1*i_q_B1 - V_B1*cos(delta_B1 - theta_B1) + X_v_B1*i_d_B1 + e_qv_B1
        struct[0].g[23,0] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1) - p_s_B1
        struct[0].g[24,0] = V_B1*i_d_B1*cos(delta_B1 - theta_B1) - V_B1*i_q_B1*sin(delta_B1 - theta_B1) - q_s_B1
        struct[0].g[25,0] = -omega_coi + (1000000.0*S_n_B1*omega_B1 + S_n_B3*T_p_B3*omega_B3/(2*K_p_B3))/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].g[26,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_B1
        struct[0].h[1,0] = V_B2
        struct[0].h[2,0] = V_B3
        struct[0].h[3,0] = p_gin_B3
        struct[0].h[4,0] = p_g_ref_B3
        struct[0].h[5,0] = -p_s_B3 + p_t_B3
        struct[0].h[6,0] = (-V_u_min_B3**2 + e_u_B3**2)/(V_u_max_B3**2 - V_u_min_B3**2)
        struct[0].h[7,0] = alpha_B1
    

    if mode == 10:

        struct[0].Fx[4,4] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (-K_h_B3, True)]))
        struct[0].Fx[5,5] = -1/T_cur_B3
        struct[0].Fx[6,6] = -K_delta_B1

    if mode == 11:

        struct[0].Fy[0,6] = Omega_b_B3
        struct[0].Fy[0,25] = -Omega_b_B3
        struct[0].Fy[1,10] = -1
        struct[0].Fy[1,12] = 1
        struct[0].Fy[2,11] = -1
        struct[0].Fy[3,13] = -S_n_B3/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].Fy[3,15] = -S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1)**2)
        struct[0].Fy[3,18] = S_n_B3/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].Fy[4,18] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (K_h_B3, True)]))
        struct[0].Fy[5,19] = 1/(T_cur_B3*p_gin_B3)
        struct[0].Fy[6,20] = Omega_b_B1
        struct[0].Fy[6,25] = -Omega_b_B1
        struct[0].Fy[8,25] = -1

        struct[0].Gx[6,1] = K_p_B3/T_p_B3
        struct[0].Gx[7,2] = K_q_B3/T_q_B3
        struct[0].Gx[8,0] = -V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gx[9,0] = V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gx[10,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gx[11,0] = -V_B3*i_d_B3*sin(delta_B3 - theta_B3) - V_B3*i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gx[12,4] = 1
        struct[0].Gx[13,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gx[15,3] = 1
        struct[0].Gx[17,5] = Piecewise(np.array([(0, (k_cur_B3 > 1) | (k_cur_B3 < 0.0001)), (1, True)]))
        struct[0].Gx[20,7] = 1
        struct[0].Gx[21,6] = -V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gx[22,6] = V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gx[23,6] = V_B1*i_d_B1*cos(delta_B1 - theta_B1) - V_B1*i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gx[24,6] = -V_B1*i_d_B1*sin(delta_B1 - theta_B1) - V_B1*i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gx[26,8] = K_i_agc

        struct[0].Gy[0,0] = 2*V_B1*g_B1_B2 + V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[0,1] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[0,2] = V_B1*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[0,3] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[0,23] = -S_n_B1/S_base
        struct[0].Gy[1,0] = 2*V_B1*(-b_B1_B2 - bs_B1_B2/2) + V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[1,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[1,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[1,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[1,24] = -S_n_B1/S_base
        struct[0].Gy[2,0] = V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[2,1] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[2,2] = V_B1*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + 2*V_B2*(g_B1_B2 + g_B2_B3) + V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[2,3] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[2,4] = V_B2*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[2,5] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[3,0] = V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[3,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[3,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + 2*V_B2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[3,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[3,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[3,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[4,2] = V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[4,3] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[4,4] = V_B2*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + 2*V_B3*g_B2_B3
        struct[0].Gy[4,5] = V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[4,10] = -S_n_B3/S_base
        struct[0].Gy[5,2] = V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[5,3] = V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[5,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + 2*V_B3*(-b_B2_B3 - bs_B2_B3/2)
        struct[0].Gy[5,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[5,11] = -S_n_B3/S_base
        struct[0].Gy[6,10] = -K_p_B3
        struct[0].Gy[6,12] = K_p_B3
        struct[0].Gy[7,11] = -K_q_B3
        struct[0].Gy[8,4] = -sin(delta_B3 - theta_B3)
        struct[0].Gy[8,5] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[8,8] = -R_v_B3
        struct[0].Gy[8,9] = X_v_B3
        struct[0].Gy[9,4] = -cos(delta_B3 - theta_B3)
        struct[0].Gy[9,5] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[9,8] = -X_v_B3
        struct[0].Gy[9,9] = -R_v_B3
        struct[0].Gy[10,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[10,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[10,8] = V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[10,9] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[11,4] = i_d_B3*cos(delta_B3 - theta_B3) - i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[11,5] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[11,8] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[11,9] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[13,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[13,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[13,8] = 2*R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[13,9] = 2*R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[14,15] = 2*k_u_B3*v_u_B3/V_u_max_B3**2
        struct[0].Gy[14,16] = -(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].Gy[15,13] = -R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy[15,15] = -R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1)**2 - 1
        struct[0].Gy[15,18] = R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy[16,15] = Piecewise(np.array([(0, V_u_min_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (0, True)]))
        struct[0].Gy[18,17] = p_gin_B3
        struct[0].Gy[19,6] = -Piecewise(np.array([(1/Droop_B3, (omega_B3 > 0.5*DB_B3 + omega_ref_B3) | (omega_B3 < -0.5*DB_B3 + omega_ref_B3)), (0, True)]))
        struct[0].Gy[21,0] = -sin(delta_B1 - theta_B1)
        struct[0].Gy[21,1] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[21,21] = -R_v_B1
        struct[0].Gy[21,22] = -X_v_B1
        struct[0].Gy[22,0] = -cos(delta_B1 - theta_B1)
        struct[0].Gy[22,1] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[22,21] = X_v_B1
        struct[0].Gy[22,22] = -R_v_B1
        struct[0].Gy[23,0] = i_d_B1*sin(delta_B1 - theta_B1) + i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[23,1] = -V_B1*i_d_B1*cos(delta_B1 - theta_B1) + V_B1*i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[23,21] = V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[23,22] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[24,0] = i_d_B1*cos(delta_B1 - theta_B1) - i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[24,1] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[24,21] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[24,22] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[25,6] = S_n_B3*T_p_B3/(2*K_p_B3*(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3)))
        struct[0].Gy[25,20] = 1000000.0*S_n_B1/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].Gy[26,25] = -K_p_agc

    if mode > 12:

        struct[0].Fu[2,6] = 1
        struct[0].Fu[5,9] = (-p_f_B3/p_gin_B3**2 - p_g_ref_B3/p_gin_B3**2)/T_cur_B3
        struct[0].Fu[5,10] = 1/(T_cur_B3*p_gin_B3)
        struct[0].Fu[7,11] = 1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[7,6] = K_q_B3
        struct[0].Gu[14,7] = -2*k_u_B3*v_u_ref_B3/V_u_max_B3**2
        struct[0].Gu[18,9] = k_cur_sat_B3
        struct[0].Gu[19,8] = -Piecewise(np.array([(-1/Droop_B3, (omega_B3 > 0.5*DB_B3 + omega_ref_B3) | (omega_B3 < -0.5*DB_B3 + omega_ref_B3)), (0, True)]))

        struct[0].Hx[6,3] = 2*e_u_B3/(V_u_max_B3**2 - V_u_min_B3**2)

        struct[0].Hy[0,0] = 1
        struct[0].Hy[1,2] = 1
        struct[0].Hy[2,4] = 1
        struct[0].Hy[5,10] = -1
        struct[0].Hy[5,13] = 1

        struct[0].Hu[3,9] = 1
        struct[0].Hu[4,10] = 1
        struct[0].Hu[7,11] = 1



def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_B1_B2 = struct[0].g_B1_B2
    b_B1_B2 = struct[0].b_B1_B2
    bs_B1_B2 = struct[0].bs_B1_B2
    g_B2_B3 = struct[0].g_B2_B3
    b_B2_B3 = struct[0].b_B2_B3
    bs_B2_B3 = struct[0].bs_B2_B3
    U_B1_n = struct[0].U_B1_n
    U_B2_n = struct[0].U_B2_n
    U_B3_n = struct[0].U_B3_n
    S_n_B3 = struct[0].S_n_B3
    Omega_b_B3 = struct[0].Omega_b_B3
    K_p_B3 = struct[0].K_p_B3
    T_p_B3 = struct[0].T_p_B3
    K_q_B3 = struct[0].K_q_B3
    T_q_B3 = struct[0].T_q_B3
    X_v_B3 = struct[0].X_v_B3
    R_v_B3 = struct[0].R_v_B3
    R_s_B3 = struct[0].R_s_B3
    C_u_B3 = struct[0].C_u_B3
    K_u_0_B3 = struct[0].K_u_0_B3
    K_u_max_B3 = struct[0].K_u_max_B3
    V_u_min_B3 = struct[0].V_u_min_B3
    V_u_max_B3 = struct[0].V_u_max_B3
    R_uc_B3 = struct[0].R_uc_B3
    K_h_B3 = struct[0].K_h_B3
    R_lim_B3 = struct[0].R_lim_B3
    V_u_lt_B3 = struct[0].V_u_lt_B3
    V_u_ht_B3 = struct[0].V_u_ht_B3
    Droop_B3 = struct[0].Droop_B3
    DB_B3 = struct[0].DB_B3
    T_cur_B3 = struct[0].T_cur_B3
    S_n_B1 = struct[0].S_n_B1
    Omega_b_B1 = struct[0].Omega_b_B1
    X_v_B1 = struct[0].X_v_B1
    R_v_B1 = struct[0].R_v_B1
    K_delta_B1 = struct[0].K_delta_B1
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_B1 = struct[0].P_B1
    Q_B1 = struct[0].Q_B1
    P_B2 = struct[0].P_B2
    Q_B2 = struct[0].Q_B2
    P_B3 = struct[0].P_B3
    Q_B3 = struct[0].Q_B3
    q_s_ref_B3 = struct[0].q_s_ref_B3
    v_u_ref_B3 = struct[0].v_u_ref_B3
    omega_ref_B3 = struct[0].omega_ref_B3
    p_gin_B3 = struct[0].p_gin_B3
    p_g_ref_B3 = struct[0].p_g_ref_B3
    alpha_B1 = struct[0].alpha_B1
    e_qv_B1 = struct[0].e_qv_B1
    omega_ref_B1 = struct[0].omega_ref_B1
    
    # Dynamical states:
    delta_B3 = struct[0].x[0,0]
    xi_p_B3 = struct[0].x[1,0]
    xi_q_B3 = struct[0].x[2,0]
    e_u_B3 = struct[0].x[3,0]
    p_ghr_B3 = struct[0].x[4,0]
    k_cur_B3 = struct[0].x[5,0]
    delta_B1 = struct[0].x[6,0]
    Domega_B1 = struct[0].x[7,0]
    xi_freq = struct[0].x[8,0]
    
    # Algebraic states:
    V_B1 = struct[0].y_ini[0,0]
    theta_B1 = struct[0].y_ini[1,0]
    V_B2 = struct[0].y_ini[2,0]
    theta_B2 = struct[0].y_ini[3,0]
    V_B3 = struct[0].y_ini[4,0]
    theta_B3 = struct[0].y_ini[5,0]
    omega_B3 = struct[0].y_ini[6,0]
    e_qv_B3 = struct[0].y_ini[7,0]
    i_d_B3 = struct[0].y_ini[8,0]
    i_q_B3 = struct[0].y_ini[9,0]
    p_s_B3 = struct[0].y_ini[10,0]
    q_s_B3 = struct[0].y_ini[11,0]
    p_m_B3 = struct[0].y_ini[12,0]
    p_t_B3 = struct[0].y_ini[13,0]
    p_u_B3 = struct[0].y_ini[14,0]
    v_u_B3 = struct[0].y_ini[15,0]
    k_u_B3 = struct[0].y_ini[16,0]
    k_cur_sat_B3 = struct[0].y_ini[17,0]
    p_gou_B3 = struct[0].y_ini[18,0]
    p_f_B3 = struct[0].y_ini[19,0]
    omega_B1 = struct[0].y_ini[20,0]
    i_d_B1 = struct[0].y_ini[21,0]
    i_q_B1 = struct[0].y_ini[22,0]
    p_s_B1 = struct[0].y_ini[23,0]
    q_s_B1 = struct[0].y_ini[24,0]
    omega_coi = struct[0].y_ini[25,0]
    p_agc = struct[0].y_ini[26,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_B3*(omega_B3 - omega_coi)
        struct[0].f[1,0] = p_m_B3 - p_s_B3
        struct[0].f[2,0] = -q_s_B3 + q_s_ref_B3
        struct[0].f[3,0] = S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-R_lim_B3, R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3)), (R_lim_B3, R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)), (K_h_B3*(-p_ghr_B3 + p_gou_B3), True)]))
        struct[0].f[5,0] = (-k_cur_B3 + p_f_B3/p_gin_B3 + p_g_ref_B3/p_gin_B3)/T_cur_B3
        struct[0].f[6,0] = -K_delta_B1*delta_B1 + Omega_b_B1*(omega_B1 - omega_coi)
        struct[0].f[7,0] = alpha_B1
        struct[0].f[8,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_B1/S_base + V_B1**2*g_B1_B2 + V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) - S_n_B1*p_s_B1/S_base
        struct[0].g[1,0] = -Q_B1/S_base + V_B1**2*(-b_B1_B2 - bs_B1_B2/2) + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) - S_n_B1*q_s_B1/S_base
        struct[0].g[2,0] = -P_B2/S_base + V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2**2*(g_B1_B2 + g_B2_B3) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].g[3,0] = -Q_B2/S_base + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2**2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].g[4,0] = -P_B3/S_base + V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + V_B3**2*g_B2_B3 - S_n_B3*p_s_B3/S_base
        struct[0].g[5,0] = -Q_B3/S_base + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + V_B3**2*(-b_B2_B3 - bs_B2_B3/2) - S_n_B3*q_s_B3/S_base
        struct[0].g[6,0] = K_p_B3*(p_m_B3 - p_s_B3 + xi_p_B3/T_p_B3) - omega_B3
        struct[0].g[7,0] = K_q_B3*(-q_s_B3 + q_s_ref_B3 + xi_q_B3/T_q_B3) - e_qv_B3
        struct[0].g[8,0] = -R_v_B3*i_d_B3 - V_B3*sin(delta_B3 - theta_B3) + X_v_B3*i_q_B3
        struct[0].g[9,0] = -R_v_B3*i_q_B3 - V_B3*cos(delta_B3 - theta_B3) - X_v_B3*i_d_B3 + e_qv_B3
        struct[0].g[10,0] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3) - p_s_B3
        struct[0].g[11,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3) - q_s_B3
        struct[0].g[12,0] = p_ghr_B3 - p_m_B3 + p_s_B3 - p_t_B3 + p_u_B3
        struct[0].g[13,0] = i_d_B3*(R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)) + i_q_B3*(R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)) - p_t_B3
        struct[0].g[14,0] = -p_u_B3 - k_u_B3*(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].g[15,0] = R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1) + e_u_B3 - v_u_B3
        struct[0].g[16,0] = -k_u_B3 + Piecewise(np.array([(K_u_max_B3, V_u_min_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_lt_B3 + v_u_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_ht_B3 + v_u_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (K_u_max_B3, V_u_max_B3 < v_u_B3), (K_u_0_B3, True)]))
        struct[0].g[17,0] = -k_cur_sat_B3 + Piecewise(np.array([(0.0001, k_cur_B3 < 0.0001), (1, k_cur_B3 > 1), (k_cur_B3, True)]))
        struct[0].g[18,0] = k_cur_sat_B3*p_gin_B3 - p_gou_B3
        struct[0].g[19,0] = -p_f_B3 - Piecewise(np.array([((0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 < -0.5*DB_B3 + omega_ref_B3), ((-0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 > 0.5*DB_B3 + omega_ref_B3), (0.0, True)]))
        struct[0].g[20,0] = Domega_B1 - omega_B1 + omega_ref_B1
        struct[0].g[21,0] = -R_v_B1*i_d_B1 - V_B1*sin(delta_B1 - theta_B1) - X_v_B1*i_q_B1
        struct[0].g[22,0] = -R_v_B1*i_q_B1 - V_B1*cos(delta_B1 - theta_B1) + X_v_B1*i_d_B1 + e_qv_B1
        struct[0].g[23,0] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1) - p_s_B1
        struct[0].g[24,0] = V_B1*i_d_B1*cos(delta_B1 - theta_B1) - V_B1*i_q_B1*sin(delta_B1 - theta_B1) - q_s_B1
        struct[0].g[25,0] = -omega_coi + (1000000.0*S_n_B1*omega_B1 + S_n_B3*T_p_B3*omega_B3/(2*K_p_B3))/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].g[26,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_B1
        struct[0].h[1,0] = V_B2
        struct[0].h[2,0] = V_B3
        struct[0].h[3,0] = p_gin_B3
        struct[0].h[4,0] = p_g_ref_B3
        struct[0].h[5,0] = -p_s_B3 + p_t_B3
        struct[0].h[6,0] = (-V_u_min_B3**2 + e_u_B3**2)/(V_u_max_B3**2 - V_u_min_B3**2)
        struct[0].h[7,0] = alpha_B1
    

    if mode == 10:

        struct[0].Fx_ini[4,4] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (-K_h_B3, True)]))
        struct[0].Fx_ini[5,5] = -1/T_cur_B3
        struct[0].Fx_ini[6,6] = -K_delta_B1

    if mode == 11:

        struct[0].Fy_ini[0,6] = Omega_b_B3 
        struct[0].Fy_ini[0,25] = -Omega_b_B3 
        struct[0].Fy_ini[1,10] = -1 
        struct[0].Fy_ini[1,12] = 1 
        struct[0].Fy_ini[2,11] = -1 
        struct[0].Fy_ini[3,13] = -S_n_B3/(C_u_B3*(v_u_B3 + 0.1)) 
        struct[0].Fy_ini[3,15] = -S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1)**2) 
        struct[0].Fy_ini[3,18] = S_n_B3/(C_u_B3*(v_u_B3 + 0.1)) 
        struct[0].Fy_ini[4,18] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (K_h_B3, True)])) 
        struct[0].Fy_ini[5,19] = 1/(T_cur_B3*p_gin_B3) 
        struct[0].Fy_ini[6,20] = Omega_b_B1 
        struct[0].Fy_ini[6,25] = -Omega_b_B1 
        struct[0].Fy_ini[8,25] = -1 

        struct[0].Gy_ini[0,0] = 2*V_B1*g_B1_B2 + V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,1] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,2] = V_B1*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,3] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[0,23] = -S_n_B1/S_base
        struct[0].Gy_ini[1,0] = 2*V_B1*(-b_B1_B2 - bs_B1_B2/2) + V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[1,24] = -S_n_B1/S_base
        struct[0].Gy_ini[2,0] = V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[2,1] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[2,2] = V_B1*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + 2*V_B2*(g_B1_B2 + g_B2_B3) + V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[2,3] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[2,4] = V_B2*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[2,5] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,0] = V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy_ini[3,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy_ini[3,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + 2*V_B2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[3,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,2] = V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,3] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,4] = V_B2*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + 2*V_B3*g_B2_B3
        struct[0].Gy_ini[4,5] = V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[4,10] = -S_n_B3/S_base
        struct[0].Gy_ini[5,2] = V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy_ini[5,3] = V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[5,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + 2*V_B3*(-b_B2_B3 - bs_B2_B3/2)
        struct[0].Gy_ini[5,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy_ini[5,11] = -S_n_B3/S_base
        struct[0].Gy_ini[6,6] = -1
        struct[0].Gy_ini[6,10] = -K_p_B3
        struct[0].Gy_ini[6,12] = K_p_B3
        struct[0].Gy_ini[7,7] = -1
        struct[0].Gy_ini[7,11] = -K_q_B3
        struct[0].Gy_ini[8,4] = -sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[8,5] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[8,8] = -R_v_B3
        struct[0].Gy_ini[8,9] = X_v_B3
        struct[0].Gy_ini[9,4] = -cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[9,5] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[9,7] = 1
        struct[0].Gy_ini[9,8] = -X_v_B3
        struct[0].Gy_ini[9,9] = -R_v_B3
        struct[0].Gy_ini[10,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[10,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[10,8] = V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[10,9] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[10,10] = -1
        struct[0].Gy_ini[11,4] = i_d_B3*cos(delta_B3 - theta_B3) - i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,5] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,8] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,9] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[11,11] = -1
        struct[0].Gy_ini[12,10] = 1
        struct[0].Gy_ini[12,12] = -1
        struct[0].Gy_ini[12,13] = -1
        struct[0].Gy_ini[12,14] = 1
        struct[0].Gy_ini[13,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,8] = 2*R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,9] = 2*R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy_ini[13,13] = -1
        struct[0].Gy_ini[14,14] = -1
        struct[0].Gy_ini[14,15] = 2*k_u_B3*v_u_B3/V_u_max_B3**2
        struct[0].Gy_ini[14,16] = -(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].Gy_ini[15,13] = -R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy_ini[15,15] = -R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1)**2 - 1
        struct[0].Gy_ini[15,18] = R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy_ini[16,15] = Piecewise(np.array([(0, V_u_min_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (0, True)]))
        struct[0].Gy_ini[16,16] = -1
        struct[0].Gy_ini[17,17] = -1
        struct[0].Gy_ini[18,17] = p_gin_B3
        struct[0].Gy_ini[18,18] = -1
        struct[0].Gy_ini[19,6] = -Piecewise(np.array([(1/Droop_B3, (omega_B3 > 0.5*DB_B3 + omega_ref_B3) | (omega_B3 < -0.5*DB_B3 + omega_ref_B3)), (0, True)]))
        struct[0].Gy_ini[19,19] = -1
        struct[0].Gy_ini[20,20] = -1
        struct[0].Gy_ini[21,0] = -sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[21,1] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[21,21] = -R_v_B1
        struct[0].Gy_ini[21,22] = -X_v_B1
        struct[0].Gy_ini[22,0] = -cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[22,1] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[22,21] = X_v_B1
        struct[0].Gy_ini[22,22] = -R_v_B1
        struct[0].Gy_ini[23,0] = i_d_B1*sin(delta_B1 - theta_B1) + i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[23,1] = -V_B1*i_d_B1*cos(delta_B1 - theta_B1) + V_B1*i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[23,21] = V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[23,22] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[23,23] = -1
        struct[0].Gy_ini[24,0] = i_d_B1*cos(delta_B1 - theta_B1) - i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,1] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,21] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,22] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[25,6] = S_n_B3*T_p_B3/(2*K_p_B3*(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3)))
        struct[0].Gy_ini[25,20] = 1000000.0*S_n_B1/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].Gy_ini[25,25] = -1
        struct[0].Gy_ini[26,25] = -K_p_agc
        struct[0].Gy_ini[26,26] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_B1_B2 = struct[0].g_B1_B2
    b_B1_B2 = struct[0].b_B1_B2
    bs_B1_B2 = struct[0].bs_B1_B2
    g_B2_B3 = struct[0].g_B2_B3
    b_B2_B3 = struct[0].b_B2_B3
    bs_B2_B3 = struct[0].bs_B2_B3
    U_B1_n = struct[0].U_B1_n
    U_B2_n = struct[0].U_B2_n
    U_B3_n = struct[0].U_B3_n
    S_n_B3 = struct[0].S_n_B3
    Omega_b_B3 = struct[0].Omega_b_B3
    K_p_B3 = struct[0].K_p_B3
    T_p_B3 = struct[0].T_p_B3
    K_q_B3 = struct[0].K_q_B3
    T_q_B3 = struct[0].T_q_B3
    X_v_B3 = struct[0].X_v_B3
    R_v_B3 = struct[0].R_v_B3
    R_s_B3 = struct[0].R_s_B3
    C_u_B3 = struct[0].C_u_B3
    K_u_0_B3 = struct[0].K_u_0_B3
    K_u_max_B3 = struct[0].K_u_max_B3
    V_u_min_B3 = struct[0].V_u_min_B3
    V_u_max_B3 = struct[0].V_u_max_B3
    R_uc_B3 = struct[0].R_uc_B3
    K_h_B3 = struct[0].K_h_B3
    R_lim_B3 = struct[0].R_lim_B3
    V_u_lt_B3 = struct[0].V_u_lt_B3
    V_u_ht_B3 = struct[0].V_u_ht_B3
    Droop_B3 = struct[0].Droop_B3
    DB_B3 = struct[0].DB_B3
    T_cur_B3 = struct[0].T_cur_B3
    S_n_B1 = struct[0].S_n_B1
    Omega_b_B1 = struct[0].Omega_b_B1
    X_v_B1 = struct[0].X_v_B1
    R_v_B1 = struct[0].R_v_B1
    K_delta_B1 = struct[0].K_delta_B1
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_B1 = struct[0].P_B1
    Q_B1 = struct[0].Q_B1
    P_B2 = struct[0].P_B2
    Q_B2 = struct[0].Q_B2
    P_B3 = struct[0].P_B3
    Q_B3 = struct[0].Q_B3
    q_s_ref_B3 = struct[0].q_s_ref_B3
    v_u_ref_B3 = struct[0].v_u_ref_B3
    omega_ref_B3 = struct[0].omega_ref_B3
    p_gin_B3 = struct[0].p_gin_B3
    p_g_ref_B3 = struct[0].p_g_ref_B3
    alpha_B1 = struct[0].alpha_B1
    e_qv_B1 = struct[0].e_qv_B1
    omega_ref_B1 = struct[0].omega_ref_B1
    
    # Dynamical states:
    delta_B3 = struct[0].x[0,0]
    xi_p_B3 = struct[0].x[1,0]
    xi_q_B3 = struct[0].x[2,0]
    e_u_B3 = struct[0].x[3,0]
    p_ghr_B3 = struct[0].x[4,0]
    k_cur_B3 = struct[0].x[5,0]
    delta_B1 = struct[0].x[6,0]
    Domega_B1 = struct[0].x[7,0]
    xi_freq = struct[0].x[8,0]
    
    # Algebraic states:
    V_B1 = struct[0].y_run[0,0]
    theta_B1 = struct[0].y_run[1,0]
    V_B2 = struct[0].y_run[2,0]
    theta_B2 = struct[0].y_run[3,0]
    V_B3 = struct[0].y_run[4,0]
    theta_B3 = struct[0].y_run[5,0]
    omega_B3 = struct[0].y_run[6,0]
    e_qv_B3 = struct[0].y_run[7,0]
    i_d_B3 = struct[0].y_run[8,0]
    i_q_B3 = struct[0].y_run[9,0]
    p_s_B3 = struct[0].y_run[10,0]
    q_s_B3 = struct[0].y_run[11,0]
    p_m_B3 = struct[0].y_run[12,0]
    p_t_B3 = struct[0].y_run[13,0]
    p_u_B3 = struct[0].y_run[14,0]
    v_u_B3 = struct[0].y_run[15,0]
    k_u_B3 = struct[0].y_run[16,0]
    k_cur_sat_B3 = struct[0].y_run[17,0]
    p_gou_B3 = struct[0].y_run[18,0]
    p_f_B3 = struct[0].y_run[19,0]
    omega_B1 = struct[0].y_run[20,0]
    i_d_B1 = struct[0].y_run[21,0]
    i_q_B1 = struct[0].y_run[22,0]
    p_s_B1 = struct[0].y_run[23,0]
    q_s_B1 = struct[0].y_run[24,0]
    omega_coi = struct[0].y_run[25,0]
    p_agc = struct[0].y_run[26,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_B3*(omega_B3 - omega_coi)
        struct[0].f[1,0] = p_m_B3 - p_s_B3
        struct[0].f[2,0] = -q_s_B3 + q_s_ref_B3
        struct[0].f[3,0] = S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-R_lim_B3, R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3)), (R_lim_B3, R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)), (K_h_B3*(-p_ghr_B3 + p_gou_B3), True)]))
        struct[0].f[5,0] = (-k_cur_B3 + p_f_B3/p_gin_B3 + p_g_ref_B3/p_gin_B3)/T_cur_B3
        struct[0].f[6,0] = -K_delta_B1*delta_B1 + Omega_b_B1*(omega_B1 - omega_coi)
        struct[0].f[7,0] = alpha_B1
        struct[0].f[8,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_B1/S_base + V_B1**2*g_B1_B2 + V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) - S_n_B1*p_s_B1/S_base
        struct[0].g[1,0] = -Q_B1/S_base + V_B1**2*(-b_B1_B2 - bs_B1_B2/2) + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) - S_n_B1*q_s_B1/S_base
        struct[0].g[2,0] = -P_B2/S_base + V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2**2*(g_B1_B2 + g_B2_B3) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].g[3,0] = -Q_B2/S_base + V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2**2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].g[4,0] = -P_B3/S_base + V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + V_B3**2*g_B2_B3 - S_n_B3*p_s_B3/S_base
        struct[0].g[5,0] = -Q_B3/S_base + V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + V_B3**2*(-b_B2_B3 - bs_B2_B3/2) - S_n_B3*q_s_B3/S_base
        struct[0].g[6,0] = K_p_B3*(p_m_B3 - p_s_B3 + xi_p_B3/T_p_B3) - omega_B3
        struct[0].g[7,0] = K_q_B3*(-q_s_B3 + q_s_ref_B3 + xi_q_B3/T_q_B3) - e_qv_B3
        struct[0].g[8,0] = -R_v_B3*i_d_B3 - V_B3*sin(delta_B3 - theta_B3) + X_v_B3*i_q_B3
        struct[0].g[9,0] = -R_v_B3*i_q_B3 - V_B3*cos(delta_B3 - theta_B3) - X_v_B3*i_d_B3 + e_qv_B3
        struct[0].g[10,0] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3) - p_s_B3
        struct[0].g[11,0] = V_B3*i_d_B3*cos(delta_B3 - theta_B3) - V_B3*i_q_B3*sin(delta_B3 - theta_B3) - q_s_B3
        struct[0].g[12,0] = p_ghr_B3 - p_m_B3 + p_s_B3 - p_t_B3 + p_u_B3
        struct[0].g[13,0] = i_d_B3*(R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)) + i_q_B3*(R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)) - p_t_B3
        struct[0].g[14,0] = -p_u_B3 - k_u_B3*(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].g[15,0] = R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1) + e_u_B3 - v_u_B3
        struct[0].g[16,0] = -k_u_B3 + Piecewise(np.array([(K_u_max_B3, V_u_min_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_lt_B3 + v_u_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), (K_u_0_B3 + (-K_u_0_B3 + K_u_max_B3)*(-V_u_ht_B3 + v_u_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (K_u_max_B3, V_u_max_B3 < v_u_B3), (K_u_0_B3, True)]))
        struct[0].g[17,0] = -k_cur_sat_B3 + Piecewise(np.array([(0.0001, k_cur_B3 < 0.0001), (1, k_cur_B3 > 1), (k_cur_B3, True)]))
        struct[0].g[18,0] = k_cur_sat_B3*p_gin_B3 - p_gou_B3
        struct[0].g[19,0] = -p_f_B3 - Piecewise(np.array([((0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 < -0.5*DB_B3 + omega_ref_B3), ((-0.5*DB_B3 + omega_B3 - omega_ref_B3)/Droop_B3, omega_B3 > 0.5*DB_B3 + omega_ref_B3), (0.0, True)]))
        struct[0].g[20,0] = Domega_B1 - omega_B1 + omega_ref_B1
        struct[0].g[21,0] = -R_v_B1*i_d_B1 - V_B1*sin(delta_B1 - theta_B1) - X_v_B1*i_q_B1
        struct[0].g[22,0] = -R_v_B1*i_q_B1 - V_B1*cos(delta_B1 - theta_B1) + X_v_B1*i_d_B1 + e_qv_B1
        struct[0].g[23,0] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1) - p_s_B1
        struct[0].g[24,0] = V_B1*i_d_B1*cos(delta_B1 - theta_B1) - V_B1*i_q_B1*sin(delta_B1 - theta_B1) - q_s_B1
        struct[0].g[25,0] = -omega_coi + (1000000.0*S_n_B1*omega_B1 + S_n_B3*T_p_B3*omega_B3/(2*K_p_B3))/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].g[26,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_B1
        struct[0].h[1,0] = V_B2
        struct[0].h[2,0] = V_B3
        struct[0].h[3,0] = p_gin_B3
        struct[0].h[4,0] = p_g_ref_B3
        struct[0].h[5,0] = -p_s_B3 + p_t_B3
        struct[0].h[6,0] = (-V_u_min_B3**2 + e_u_B3**2)/(V_u_max_B3**2 - V_u_min_B3**2)
        struct[0].h[7,0] = alpha_B1
    

    if mode == 10:

        struct[0].Fx[4,4] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (-K_h_B3, True)]))
        struct[0].Fx[5,5] = -1/T_cur_B3
        struct[0].Fx[6,6] = -K_delta_B1

    if mode == 11:

        struct[0].Fy[0,6] = Omega_b_B3
        struct[0].Fy[0,25] = -Omega_b_B3
        struct[0].Fy[1,10] = -1
        struct[0].Fy[1,12] = 1
        struct[0].Fy[2,11] = -1
        struct[0].Fy[3,13] = -S_n_B3/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].Fy[3,15] = -S_n_B3*(p_gou_B3 - p_t_B3)/(C_u_B3*(v_u_B3 + 0.1)**2)
        struct[0].Fy[3,18] = S_n_B3/(C_u_B3*(v_u_B3 + 0.1))
        struct[0].Fy[4,18] = Piecewise(np.array([(0, (R_lim_B3 < K_h_B3*(-p_ghr_B3 + p_gou_B3)) | (R_lim_B3 < -K_h_B3*(-p_ghr_B3 + p_gou_B3))), (K_h_B3, True)]))
        struct[0].Fy[5,19] = 1/(T_cur_B3*p_gin_B3)
        struct[0].Fy[6,20] = Omega_b_B1
        struct[0].Fy[6,25] = -Omega_b_B1
        struct[0].Fy[8,25] = -1

        struct[0].Gy[0,0] = 2*V_B1*g_B1_B2 + V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[0,1] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[0,2] = V_B1*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[0,3] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[0,23] = -S_n_B1/S_base
        struct[0].Gy[1,0] = 2*V_B1*(-b_B1_B2 - bs_B1_B2/2) + V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[1,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[1,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[1,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[1,24] = -S_n_B1/S_base
        struct[0].Gy[2,0] = V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[2,1] = V_B1*V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[2,2] = V_B1*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + 2*V_B2*(g_B1_B2 + g_B2_B3) + V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[2,3] = V_B1*V_B2*(-b_B1_B2*cos(theta_B1 - theta_B2) - g_B1_B2*sin(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[2,4] = V_B2*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[2,5] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[3,0] = V_B2*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2))
        struct[0].Gy[3,1] = V_B1*V_B2*(-b_B1_B2*sin(theta_B1 - theta_B2) + g_B1_B2*cos(theta_B1 - theta_B2))
        struct[0].Gy[3,2] = V_B1*(b_B1_B2*cos(theta_B1 - theta_B2) + g_B1_B2*sin(theta_B1 - theta_B2)) + 2*V_B2*(-b_B1_B2 - b_B2_B3 - bs_B1_B2/2 - bs_B2_B3/2) + V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[3,3] = V_B1*V_B2*(b_B1_B2*sin(theta_B1 - theta_B2) - g_B1_B2*cos(theta_B1 - theta_B2)) + V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[3,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[3,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[4,2] = V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[4,3] = V_B2*V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[4,4] = V_B2*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3)) + 2*V_B3*g_B2_B3
        struct[0].Gy[4,5] = V_B2*V_B3*(-b_B2_B3*cos(theta_B2 - theta_B3) - g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[4,10] = -S_n_B3/S_base
        struct[0].Gy[5,2] = V_B3*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3))
        struct[0].Gy[5,3] = V_B2*V_B3*(-b_B2_B3*sin(theta_B2 - theta_B3) + g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[5,4] = V_B2*(b_B2_B3*cos(theta_B2 - theta_B3) + g_B2_B3*sin(theta_B2 - theta_B3)) + 2*V_B3*(-b_B2_B3 - bs_B2_B3/2)
        struct[0].Gy[5,5] = V_B2*V_B3*(b_B2_B3*sin(theta_B2 - theta_B3) - g_B2_B3*cos(theta_B2 - theta_B3))
        struct[0].Gy[5,11] = -S_n_B3/S_base
        struct[0].Gy[6,6] = -1
        struct[0].Gy[6,10] = -K_p_B3
        struct[0].Gy[6,12] = K_p_B3
        struct[0].Gy[7,7] = -1
        struct[0].Gy[7,11] = -K_q_B3
        struct[0].Gy[8,4] = -sin(delta_B3 - theta_B3)
        struct[0].Gy[8,5] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[8,8] = -R_v_B3
        struct[0].Gy[8,9] = X_v_B3
        struct[0].Gy[9,4] = -cos(delta_B3 - theta_B3)
        struct[0].Gy[9,5] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[9,7] = 1
        struct[0].Gy[9,8] = -X_v_B3
        struct[0].Gy[9,9] = -R_v_B3
        struct[0].Gy[10,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[10,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[10,8] = V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[10,9] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[10,10] = -1
        struct[0].Gy[11,4] = i_d_B3*cos(delta_B3 - theta_B3) - i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[11,5] = V_B3*i_d_B3*sin(delta_B3 - theta_B3) + V_B3*i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[11,8] = V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[11,9] = -V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[11,11] = -1
        struct[0].Gy[12,10] = 1
        struct[0].Gy[12,12] = -1
        struct[0].Gy[12,13] = -1
        struct[0].Gy[12,14] = 1
        struct[0].Gy[13,4] = i_d_B3*sin(delta_B3 - theta_B3) + i_q_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[13,5] = -V_B3*i_d_B3*cos(delta_B3 - theta_B3) + V_B3*i_q_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[13,8] = 2*R_s_B3*i_d_B3 + V_B3*sin(delta_B3 - theta_B3)
        struct[0].Gy[13,9] = 2*R_s_B3*i_q_B3 + V_B3*cos(delta_B3 - theta_B3)
        struct[0].Gy[13,13] = -1
        struct[0].Gy[14,14] = -1
        struct[0].Gy[14,15] = 2*k_u_B3*v_u_B3/V_u_max_B3**2
        struct[0].Gy[14,16] = -(-v_u_B3**2 + v_u_ref_B3**2)/V_u_max_B3**2
        struct[0].Gy[15,13] = -R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy[15,15] = -R_uc_B3*S_n_B3*(p_gou_B3 - p_t_B3)/(v_u_B3 + 0.1)**2 - 1
        struct[0].Gy[15,18] = R_uc_B3*S_n_B3/(v_u_B3 + 0.1)
        struct[0].Gy[16,15] = Piecewise(np.array([(0, V_u_min_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_lt_B3 + V_u_min_B3), V_u_lt_B3 > v_u_B3), ((-K_u_0_B3 + K_u_max_B3)/(-V_u_ht_B3 + V_u_max_B3), V_u_ht_B3 < v_u_B3), (0, True)]))
        struct[0].Gy[16,16] = -1
        struct[0].Gy[17,17] = -1
        struct[0].Gy[18,17] = p_gin_B3
        struct[0].Gy[18,18] = -1
        struct[0].Gy[19,6] = -Piecewise(np.array([(1/Droop_B3, (omega_B3 > 0.5*DB_B3 + omega_ref_B3) | (omega_B3 < -0.5*DB_B3 + omega_ref_B3)), (0, True)]))
        struct[0].Gy[19,19] = -1
        struct[0].Gy[20,20] = -1
        struct[0].Gy[21,0] = -sin(delta_B1 - theta_B1)
        struct[0].Gy[21,1] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[21,21] = -R_v_B1
        struct[0].Gy[21,22] = -X_v_B1
        struct[0].Gy[22,0] = -cos(delta_B1 - theta_B1)
        struct[0].Gy[22,1] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[22,21] = X_v_B1
        struct[0].Gy[22,22] = -R_v_B1
        struct[0].Gy[23,0] = i_d_B1*sin(delta_B1 - theta_B1) + i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[23,1] = -V_B1*i_d_B1*cos(delta_B1 - theta_B1) + V_B1*i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[23,21] = V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[23,22] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[23,23] = -1
        struct[0].Gy[24,0] = i_d_B1*cos(delta_B1 - theta_B1) - i_q_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[24,1] = V_B1*i_d_B1*sin(delta_B1 - theta_B1) + V_B1*i_q_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[24,21] = V_B1*cos(delta_B1 - theta_B1)
        struct[0].Gy[24,22] = -V_B1*sin(delta_B1 - theta_B1)
        struct[0].Gy[24,24] = -1
        struct[0].Gy[25,6] = S_n_B3*T_p_B3/(2*K_p_B3*(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3)))
        struct[0].Gy[25,20] = 1000000.0*S_n_B1/(1000000.0*S_n_B1 + S_n_B3*T_p_B3/(2*K_p_B3))
        struct[0].Gy[25,25] = -1
        struct[0].Gy[26,25] = -K_p_agc
        struct[0].Gy[26,26] = -1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[7,6] = K_q_B3
        struct[0].Gu[14,7] = -2*k_u_B3*v_u_ref_B3/V_u_max_B3**2
        struct[0].Gu[18,9] = k_cur_sat_B3
        struct[0].Gu[19,8] = -Piecewise(np.array([(-1/Droop_B3, (omega_B3 > 0.5*DB_B3 + omega_ref_B3) | (omega_B3 < -0.5*DB_B3 + omega_ref_B3)), (0, True)]))
        struct[0].Gu[20,13] = 1
        struct[0].Gu[22,12] = 1





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
    Fx_ini_rows = [4, 5, 6]

    Fx_ini_cols = [4, 5, 6]

    Fy_ini_rows = [0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 8]

    Fy_ini_cols = [6, 25, 10, 12, 11, 13, 15, 18, 18, 19, 20, 25, 25]

    Gx_ini_rows = [6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 20, 21, 22, 23, 24, 26]

    Gx_ini_cols = [1, 2, 0, 0, 0, 0, 4, 0, 3, 5, 7, 6, 6, 6, 6, 8]

    Gy_ini_rows = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 17, 18, 18, 19, 19, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 26, 26]

    Gy_ini_cols = [0, 1, 2, 3, 23, 0, 1, 2, 3, 24, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 10, 2, 3, 4, 5, 11, 6, 10, 12, 7, 11, 4, 5, 8, 9, 4, 5, 7, 8, 9, 4, 5, 8, 9, 10, 4, 5, 8, 9, 11, 10, 12, 13, 14, 4, 5, 8, 9, 13, 14, 15, 16, 13, 15, 18, 15, 16, 17, 17, 18, 6, 19, 20, 0, 1, 21, 22, 0, 1, 21, 22, 0, 1, 21, 22, 23, 0, 1, 21, 22, 24, 6, 20, 25, 25, 26]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols