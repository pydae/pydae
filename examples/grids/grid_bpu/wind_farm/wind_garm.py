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


class wind_garm_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 4
        self.N_y = 32 
        self.N_z = 13 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_W1mv_W2mv', 'b_W1mv_W2mv', 'bs_W1mv_W2mv', 'g_W2mv_W3mv', 'b_W2mv_W3mv', 'bs_W2mv_W3mv', 'g_W3mv_POImv', 'b_W3mv_POImv', 'bs_W3mv_POImv', 'g_STmv_POImv', 'b_STmv_POImv', 'bs_STmv_POImv', 'g_POI_GRID', 'b_POI_GRID', 'bs_POI_GRID', 'g_POI_POImv', 'b_POI_POImv', 'bs_POI_POImv', 'g_W1mv_W1lv', 'b_W1mv_W1lv', 'bs_W1mv_W1lv', 'g_W2mv_W2lv', 'b_W2mv_W2lv', 'bs_W2mv_W2lv', 'g_W3mv_W3lv', 'b_W3mv_W3lv', 'bs_W3mv_W3lv', 'g_STmv_STlv', 'b_STmv_STlv', 'bs_STmv_STlv', 'U_W1lv_n', 'U_W2lv_n', 'U_W3lv_n', 'U_STlv_n', 'U_POIlv_n', 'U_W1mv_n', 'U_W2mv_n', 'U_W3mv_n', 'U_POImv_n', 'U_STmv_n', 'U_POI_n', 'U_GRID_n', 'S_n_GRID', 'Omega_b_GRID', 'K_p_GRID', 'T_p_GRID', 'K_q_GRID', 'T_v_GRID', 'X_v_GRID', 'R_v_GRID', 'K_delta_GRID', 'K_sec_GRID', 'Droop_GRID', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [100000000.0, 25.385137099118303, -11.433000678228852, 0.0, 25.385137099118303, -11.433000678228852, 0.0, 25.385137099118303, -11.433000678228852, 0.0, 25.385137099118303, -11.433000678228852, 0.0, 2.7644414300939832, -1.2450537738591219, 0.0, 1.923076923076923, -0.38461538461538464, 0.0, 0.4054054054054054, -0.06756756756756757, 0.0, 0.4054054054054054, -0.06756756756756757, 0.0, 0.4054054054054054, -0.06756756756756757, 0.0, 0.4054054054054054, -0.06756756756756757, 0.0, 690.0, 690.0, 690.0, 690.0, 20000.0, 20000.0, 20000.0, 20000.0, 20000.0, 20000.0, 66000.0, 66000.0, 50000000.0, 314.1592653589793, 0.01, 0.1, 0.01, 0.1, 0.1, 0.01, 0.001, 0.0, 0.05, 0.01, 0.01] 
        self.inputs_ini_list = ['P_W1lv', 'Q_W1lv', 'P_W2lv', 'Q_W2lv', 'P_W3lv', 'Q_W3lv', 'P_STlv', 'Q_STlv', 'P_POIlv', 'Q_POIlv', 'P_W1mv', 'Q_W1mv', 'P_W2mv', 'Q_W2mv', 'P_W3mv', 'Q_W3mv', 'P_POImv', 'Q_POImv', 'P_STmv', 'Q_STmv', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'v_ref_GRID', 'p_m_GRID', 'p_c_GRID', 'omega_ref_GRID', 'q_ref_GRID'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 1.0, 0.0, 1.0, 0.0] 
        self.inputs_run_list = ['P_W1lv', 'Q_W1lv', 'P_W2lv', 'Q_W2lv', 'P_W3lv', 'Q_W3lv', 'P_STlv', 'Q_STlv', 'P_POIlv', 'Q_POIlv', 'P_W1mv', 'Q_W1mv', 'P_W2mv', 'Q_W2mv', 'P_W3mv', 'Q_W3mv', 'P_POImv', 'Q_POImv', 'P_STmv', 'Q_STmv', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'v_ref_GRID', 'p_m_GRID', 'p_c_GRID', 'omega_ref_GRID', 'q_ref_GRID'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 1.0, 0.0, 1.0, 0.0] 
        self.outputs_list = ['V_W1lv', 'V_W2lv', 'V_W3lv', 'V_STlv', 'V_POIlv', 'V_W1mv', 'V_W2mv', 'V_W3mv', 'V_POImv', 'V_STmv', 'V_POI', 'V_GRID', 'p_e_GRID'] 
        self.x_list = ['delta_GRID', 'xi_p_GRID', 'e_qv_GRID', 'xi_freq'] 
        self.y_run_list = ['V_W1lv', 'theta_W1lv', 'V_W2lv', 'theta_W2lv', 'V_W3lv', 'theta_W3lv', 'V_STlv', 'theta_STlv', 'V_POIlv', 'theta_POIlv', 'V_W1mv', 'theta_W1mv', 'V_W2mv', 'theta_W2mv', 'V_W3mv', 'theta_W3mv', 'V_POImv', 'theta_POImv', 'V_STmv', 'theta_STmv', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_g_GRID', 'q_g_GRID', 'p_m_GRID', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_W1lv', 'theta_W1lv', 'V_W2lv', 'theta_W2lv', 'V_W3lv', 'theta_W3lv', 'V_STlv', 'theta_STlv', 'V_POIlv', 'theta_POIlv', 'V_W1mv', 'theta_W1mv', 'V_W2mv', 'theta_W2mv', 'V_W3mv', 'theta_W3mv', 'V_POImv', 'theta_POImv', 'V_STmv', 'theta_STmv', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_g_GRID', 'q_g_GRID', 'p_m_GRID', 'omega_coi', 'p_agc'] 
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
    g_W1mv_W2mv = struct[0].g_W1mv_W2mv
    b_W1mv_W2mv = struct[0].b_W1mv_W2mv
    bs_W1mv_W2mv = struct[0].bs_W1mv_W2mv
    g_W2mv_W3mv = struct[0].g_W2mv_W3mv
    b_W2mv_W3mv = struct[0].b_W2mv_W3mv
    bs_W2mv_W3mv = struct[0].bs_W2mv_W3mv
    g_W3mv_POImv = struct[0].g_W3mv_POImv
    b_W3mv_POImv = struct[0].b_W3mv_POImv
    bs_W3mv_POImv = struct[0].bs_W3mv_POImv
    g_STmv_POImv = struct[0].g_STmv_POImv
    b_STmv_POImv = struct[0].b_STmv_POImv
    bs_STmv_POImv = struct[0].bs_STmv_POImv
    g_POI_GRID = struct[0].g_POI_GRID
    b_POI_GRID = struct[0].b_POI_GRID
    bs_POI_GRID = struct[0].bs_POI_GRID
    g_POI_POImv = struct[0].g_POI_POImv
    b_POI_POImv = struct[0].b_POI_POImv
    bs_POI_POImv = struct[0].bs_POI_POImv
    g_W1mv_W1lv = struct[0].g_W1mv_W1lv
    b_W1mv_W1lv = struct[0].b_W1mv_W1lv
    bs_W1mv_W1lv = struct[0].bs_W1mv_W1lv
    g_W2mv_W2lv = struct[0].g_W2mv_W2lv
    b_W2mv_W2lv = struct[0].b_W2mv_W2lv
    bs_W2mv_W2lv = struct[0].bs_W2mv_W2lv
    g_W3mv_W3lv = struct[0].g_W3mv_W3lv
    b_W3mv_W3lv = struct[0].b_W3mv_W3lv
    bs_W3mv_W3lv = struct[0].bs_W3mv_W3lv
    g_STmv_STlv = struct[0].g_STmv_STlv
    b_STmv_STlv = struct[0].b_STmv_STlv
    bs_STmv_STlv = struct[0].bs_STmv_STlv
    U_W1lv_n = struct[0].U_W1lv_n
    U_W2lv_n = struct[0].U_W2lv_n
    U_W3lv_n = struct[0].U_W3lv_n
    U_STlv_n = struct[0].U_STlv_n
    U_POIlv_n = struct[0].U_POIlv_n
    U_W1mv_n = struct[0].U_W1mv_n
    U_W2mv_n = struct[0].U_W2mv_n
    U_W3mv_n = struct[0].U_W3mv_n
    U_POImv_n = struct[0].U_POImv_n
    U_STmv_n = struct[0].U_STmv_n
    U_POI_n = struct[0].U_POI_n
    U_GRID_n = struct[0].U_GRID_n
    S_n_GRID = struct[0].S_n_GRID
    Omega_b_GRID = struct[0].Omega_b_GRID
    K_p_GRID = struct[0].K_p_GRID
    T_p_GRID = struct[0].T_p_GRID
    K_q_GRID = struct[0].K_q_GRID
    T_v_GRID = struct[0].T_v_GRID
    X_v_GRID = struct[0].X_v_GRID
    R_v_GRID = struct[0].R_v_GRID
    K_delta_GRID = struct[0].K_delta_GRID
    K_sec_GRID = struct[0].K_sec_GRID
    Droop_GRID = struct[0].Droop_GRID
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_W1lv = struct[0].P_W1lv
    Q_W1lv = struct[0].Q_W1lv
    P_W2lv = struct[0].P_W2lv
    Q_W2lv = struct[0].Q_W2lv
    P_W3lv = struct[0].P_W3lv
    Q_W3lv = struct[0].Q_W3lv
    P_STlv = struct[0].P_STlv
    Q_STlv = struct[0].Q_STlv
    P_POIlv = struct[0].P_POIlv
    Q_POIlv = struct[0].Q_POIlv
    P_W1mv = struct[0].P_W1mv
    Q_W1mv = struct[0].Q_W1mv
    P_W2mv = struct[0].P_W2mv
    Q_W2mv = struct[0].Q_W2mv
    P_W3mv = struct[0].P_W3mv
    Q_W3mv = struct[0].Q_W3mv
    P_POImv = struct[0].P_POImv
    Q_POImv = struct[0].Q_POImv
    P_STmv = struct[0].P_STmv
    Q_STmv = struct[0].Q_STmv
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_GRID = struct[0].P_GRID
    Q_GRID = struct[0].Q_GRID
    v_ref_GRID = struct[0].v_ref_GRID
    p_m_GRID = struct[0].p_m_GRID
    p_c_GRID = struct[0].p_c_GRID
    omega_ref_GRID = struct[0].omega_ref_GRID
    q_ref_GRID = struct[0].q_ref_GRID
    
    # Dynamical states:
    delta_GRID = struct[0].x[0,0]
    xi_p_GRID = struct[0].x[1,0]
    e_qv_GRID = struct[0].x[2,0]
    xi_freq = struct[0].x[3,0]
    
    # Algebraic states:
    V_W1lv = struct[0].y_ini[0,0]
    theta_W1lv = struct[0].y_ini[1,0]
    V_W2lv = struct[0].y_ini[2,0]
    theta_W2lv = struct[0].y_ini[3,0]
    V_W3lv = struct[0].y_ini[4,0]
    theta_W3lv = struct[0].y_ini[5,0]
    V_STlv = struct[0].y_ini[6,0]
    theta_STlv = struct[0].y_ini[7,0]
    V_POIlv = struct[0].y_ini[8,0]
    theta_POIlv = struct[0].y_ini[9,0]
    V_W1mv = struct[0].y_ini[10,0]
    theta_W1mv = struct[0].y_ini[11,0]
    V_W2mv = struct[0].y_ini[12,0]
    theta_W2mv = struct[0].y_ini[13,0]
    V_W3mv = struct[0].y_ini[14,0]
    theta_W3mv = struct[0].y_ini[15,0]
    V_POImv = struct[0].y_ini[16,0]
    theta_POImv = struct[0].y_ini[17,0]
    V_STmv = struct[0].y_ini[18,0]
    theta_STmv = struct[0].y_ini[19,0]
    V_POI = struct[0].y_ini[20,0]
    theta_POI = struct[0].y_ini[21,0]
    V_GRID = struct[0].y_ini[22,0]
    theta_GRID = struct[0].y_ini[23,0]
    omega_GRID = struct[0].y_ini[24,0]
    i_d_GRID = struct[0].y_ini[25,0]
    i_q_GRID = struct[0].y_ini[26,0]
    p_g_GRID = struct[0].y_ini[27,0]
    q_g_GRID = struct[0].y_ini[28,0]
    p_m_GRID = struct[0].y_ini[29,0]
    omega_coi = struct[0].y_ini[30,0]
    p_agc = struct[0].y_ini[31,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRID*delta_GRID + Omega_b_GRID*(omega_GRID - omega_coi)
        struct[0].f[1,0] = -i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID
        struct[0].f[2,0] = (K_q_GRID*(-q_g_GRID + q_ref_GRID) - e_qv_GRID + v_ref_GRID)/T_v_GRID
        struct[0].f[3,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_W1lv/S_base + V_W1lv**2*g_W1mv_W1lv + V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].g[1,0] = -Q_W1lv/S_base + V_W1lv**2*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].g[2,0] = -P_W2lv/S_base + V_W2lv**2*g_W2mv_W2lv + V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].g[3,0] = -Q_W2lv/S_base + V_W2lv**2*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].g[4,0] = -P_W3lv/S_base + V_W3lv**2*g_W3mv_W3lv + V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].g[5,0] = -Q_W3lv/S_base + V_W3lv**2*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].g[6,0] = -P_STlv/S_base + V_STlv**2*g_STmv_STlv + V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].g[7,0] = -Q_STlv/S_base + V_STlv**2*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].g[8,0] = -P_POIlv/S_base
        struct[0].g[9,0] = -Q_POIlv/S_base
        struct[0].g[10,0] = -P_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv**2*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].g[11,0] = -Q_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv**2*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].g[12,0] = -P_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv**2*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].g[13,0] = -Q_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv**2*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].g[14,0] = -P_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + V_W3mv**2*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].g[15,0] = -Q_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + V_W3mv**2*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].g[16,0] = -P_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv**2*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].g[17,0] = -Q_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv**2*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].g[18,0] = -P_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + V_STmv**2*(g_STmv_POImv + g_STmv_STlv)
        struct[0].g[19,0] = -Q_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + V_STmv**2*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].g[20,0] = -P_POI/S_base + V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI**2*(g_POI_GRID + g_POI_POImv) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].g[21,0] = -Q_POI/S_base + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI**2*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].g[22,0] = -P_GRID/S_base + V_GRID**2*g_POI_GRID + V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) - S_n_GRID*p_g_GRID/S_base
        struct[0].g[23,0] = -Q_GRID/S_base + V_GRID**2*(-b_POI_GRID - bs_POI_GRID/2) + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) - S_n_GRID*q_g_GRID/S_base
        struct[0].g[24,0] = K_p_GRID*(-i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID + xi_p_GRID/T_p_GRID) - omega_GRID + 1
        struct[0].g[25,0] = -R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID) + X_v_GRID*i_q_GRID
        struct[0].g[26,0] = -R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID) - X_v_GRID*i_d_GRID + e_qv_GRID
        struct[0].g[27,0] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID) - p_g_GRID
        struct[0].g[28,0] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID) - q_g_GRID
        struct[0].g[29,0] = K_sec_GRID*p_agc + p_c_GRID - p_m_GRID - (omega_GRID - omega_ref_GRID)/Droop_GRID
        struct[0].g[31,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_W1lv
        struct[0].h[1,0] = V_W2lv
        struct[0].h[2,0] = V_W3lv
        struct[0].h[3,0] = V_STlv
        struct[0].h[4,0] = V_POIlv
        struct[0].h[5,0] = V_W1mv
        struct[0].h[6,0] = V_W2mv
        struct[0].h[7,0] = V_W3mv
        struct[0].h[8,0] = V_POImv
        struct[0].h[9,0] = V_STmv
        struct[0].h[10,0] = V_POI
        struct[0].h[11,0] = V_GRID
        struct[0].h[12,0] = i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) + i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_GRID
        struct[0].Fx_ini[1,0] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fx_ini[2,2] = -1/T_v_GRID

    if mode == 11:

        struct[0].Fy_ini[0,24] = Omega_b_GRID 
        struct[0].Fy_ini[0,30] = -Omega_b_GRID 
        struct[0].Fy_ini[1,22] = -i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,23] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,25] = -2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,26] = -2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,29] = 1 
        struct[0].Fy_ini[2,28] = -K_q_GRID/T_v_GRID 
        struct[0].Fy_ini[3,30] = -1 

        struct[0].Gx_ini[24,0] = K_p_GRID*(-V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gx_ini[24,1] = K_p_GRID/T_p_GRID
        struct[0].Gx_ini[25,0] = -V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gx_ini[26,0] = V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gx_ini[26,2] = 1
        struct[0].Gx_ini[27,0] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gx_ini[28,0] = -V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gx_ini[31,3] = K_i_agc

        struct[0].Gy_ini[0,0] = 2*V_W1lv*g_W1mv_W1lv + V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[0,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[0,10] = V_W1lv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[0,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,0] = 2*V_W1lv*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[2,2] = 2*V_W2lv*g_W2mv_W2lv + V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[2,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[2,12] = V_W2lv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[2,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,2] = 2*V_W2lv*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,12] = V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[4,4] = 2*V_W3lv*g_W3mv_W3lv + V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[4,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[4,14] = V_W3lv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[4,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,4] = 2*V_W3lv*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,14] = V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[6,6] = 2*V_STlv*g_STmv_STlv + V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[6,7] = V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[6,18] = V_STlv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[6,19] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,6] = 2*V_STlv*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,18] = V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,19] = V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[10,0] = V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[10,1] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[10,10] = V_W1lv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[10,11] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[10,12] = V_W1mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[10,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,0] = V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[11,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[11,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[12,2] = V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[12,3] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[12,10] = V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[12,11] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[12,12] = V_W1mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[12,13] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[12,14] = V_W2mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[12,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,2] = V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[13,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[13,10] = V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[13,11] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[13,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,14] = V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[14,4] = V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[14,5] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[14,12] = V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[14,13] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[14,14] = V_POImv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].Gy_ini[14,15] = V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[14,16] = V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[14,17] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[15,4] = V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[15,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[15,12] = V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[15,13] = V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[15,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].Gy_ini[15,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[15,16] = V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[15,17] = V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,14] = V_POImv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,15] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,16] = V_POI*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + 2*V_POImv*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,17] = V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,18] = V_POImv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[16,19] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[16,20] = V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[16,21] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[17,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + 2*V_POImv*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[17,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[17,20] = V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[17,21] = V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[18,6] = V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[18,7] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[18,16] = V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[18,17] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[18,18] = V_POImv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + 2*V_STmv*(g_STmv_POImv + g_STmv_STlv)
        struct[0].Gy_ini[18,19] = V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[19,6] = V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[19,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[19,16] = V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[19,17] = V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[19,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + 2*V_STmv*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].Gy_ini[19,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[20,16] = V_POI*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,17] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,20] = V_GRID*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + 2*V_POI*(g_POI_GRID + g_POI_POImv) + V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,21] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,22] = V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[20,23] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[21,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + 2*V_POI*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,22] = V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[21,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,20] = V_GRID*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,21] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,22] = 2*V_GRID*g_POI_GRID + V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,23] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,27] = -S_n_GRID/S_base
        struct[0].Gy_ini[23,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,22] = 2*V_GRID*(-b_POI_GRID - bs_POI_GRID/2) + V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,28] = -S_n_GRID/S_base
        struct[0].Gy_ini[24,22] = K_p_GRID*(-i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,23] = K_p_GRID*(V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,25] = K_p_GRID*(-2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,26] = K_p_GRID*(-2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,29] = K_p_GRID
        struct[0].Gy_ini[25,22] = -sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[25,23] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[25,25] = -R_v_GRID
        struct[0].Gy_ini[25,26] = X_v_GRID
        struct[0].Gy_ini[26,22] = -cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[26,23] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[26,25] = -X_v_GRID
        struct[0].Gy_ini[26,26] = -R_v_GRID
        struct[0].Gy_ini[27,22] = i_d_GRID*sin(delta_GRID - theta_GRID) + i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[27,23] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[27,25] = V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[27,26] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,22] = i_d_GRID*cos(delta_GRID - theta_GRID) - i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,23] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,25] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,26] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[29,24] = -1/Droop_GRID
        struct[0].Gy_ini[29,31] = K_sec_GRID
        struct[0].Gy_ini[31,30] = -K_p_agc



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_W1mv_W2mv = struct[0].g_W1mv_W2mv
    b_W1mv_W2mv = struct[0].b_W1mv_W2mv
    bs_W1mv_W2mv = struct[0].bs_W1mv_W2mv
    g_W2mv_W3mv = struct[0].g_W2mv_W3mv
    b_W2mv_W3mv = struct[0].b_W2mv_W3mv
    bs_W2mv_W3mv = struct[0].bs_W2mv_W3mv
    g_W3mv_POImv = struct[0].g_W3mv_POImv
    b_W3mv_POImv = struct[0].b_W3mv_POImv
    bs_W3mv_POImv = struct[0].bs_W3mv_POImv
    g_STmv_POImv = struct[0].g_STmv_POImv
    b_STmv_POImv = struct[0].b_STmv_POImv
    bs_STmv_POImv = struct[0].bs_STmv_POImv
    g_POI_GRID = struct[0].g_POI_GRID
    b_POI_GRID = struct[0].b_POI_GRID
    bs_POI_GRID = struct[0].bs_POI_GRID
    g_POI_POImv = struct[0].g_POI_POImv
    b_POI_POImv = struct[0].b_POI_POImv
    bs_POI_POImv = struct[0].bs_POI_POImv
    g_W1mv_W1lv = struct[0].g_W1mv_W1lv
    b_W1mv_W1lv = struct[0].b_W1mv_W1lv
    bs_W1mv_W1lv = struct[0].bs_W1mv_W1lv
    g_W2mv_W2lv = struct[0].g_W2mv_W2lv
    b_W2mv_W2lv = struct[0].b_W2mv_W2lv
    bs_W2mv_W2lv = struct[0].bs_W2mv_W2lv
    g_W3mv_W3lv = struct[0].g_W3mv_W3lv
    b_W3mv_W3lv = struct[0].b_W3mv_W3lv
    bs_W3mv_W3lv = struct[0].bs_W3mv_W3lv
    g_STmv_STlv = struct[0].g_STmv_STlv
    b_STmv_STlv = struct[0].b_STmv_STlv
    bs_STmv_STlv = struct[0].bs_STmv_STlv
    U_W1lv_n = struct[0].U_W1lv_n
    U_W2lv_n = struct[0].U_W2lv_n
    U_W3lv_n = struct[0].U_W3lv_n
    U_STlv_n = struct[0].U_STlv_n
    U_POIlv_n = struct[0].U_POIlv_n
    U_W1mv_n = struct[0].U_W1mv_n
    U_W2mv_n = struct[0].U_W2mv_n
    U_W3mv_n = struct[0].U_W3mv_n
    U_POImv_n = struct[0].U_POImv_n
    U_STmv_n = struct[0].U_STmv_n
    U_POI_n = struct[0].U_POI_n
    U_GRID_n = struct[0].U_GRID_n
    S_n_GRID = struct[0].S_n_GRID
    Omega_b_GRID = struct[0].Omega_b_GRID
    K_p_GRID = struct[0].K_p_GRID
    T_p_GRID = struct[0].T_p_GRID
    K_q_GRID = struct[0].K_q_GRID
    T_v_GRID = struct[0].T_v_GRID
    X_v_GRID = struct[0].X_v_GRID
    R_v_GRID = struct[0].R_v_GRID
    K_delta_GRID = struct[0].K_delta_GRID
    K_sec_GRID = struct[0].K_sec_GRID
    Droop_GRID = struct[0].Droop_GRID
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_W1lv = struct[0].P_W1lv
    Q_W1lv = struct[0].Q_W1lv
    P_W2lv = struct[0].P_W2lv
    Q_W2lv = struct[0].Q_W2lv
    P_W3lv = struct[0].P_W3lv
    Q_W3lv = struct[0].Q_W3lv
    P_STlv = struct[0].P_STlv
    Q_STlv = struct[0].Q_STlv
    P_POIlv = struct[0].P_POIlv
    Q_POIlv = struct[0].Q_POIlv
    P_W1mv = struct[0].P_W1mv
    Q_W1mv = struct[0].Q_W1mv
    P_W2mv = struct[0].P_W2mv
    Q_W2mv = struct[0].Q_W2mv
    P_W3mv = struct[0].P_W3mv
    Q_W3mv = struct[0].Q_W3mv
    P_POImv = struct[0].P_POImv
    Q_POImv = struct[0].Q_POImv
    P_STmv = struct[0].P_STmv
    Q_STmv = struct[0].Q_STmv
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_GRID = struct[0].P_GRID
    Q_GRID = struct[0].Q_GRID
    v_ref_GRID = struct[0].v_ref_GRID
    p_m_GRID = struct[0].p_m_GRID
    p_c_GRID = struct[0].p_c_GRID
    omega_ref_GRID = struct[0].omega_ref_GRID
    q_ref_GRID = struct[0].q_ref_GRID
    
    # Dynamical states:
    delta_GRID = struct[0].x[0,0]
    xi_p_GRID = struct[0].x[1,0]
    e_qv_GRID = struct[0].x[2,0]
    xi_freq = struct[0].x[3,0]
    
    # Algebraic states:
    V_W1lv = struct[0].y_run[0,0]
    theta_W1lv = struct[0].y_run[1,0]
    V_W2lv = struct[0].y_run[2,0]
    theta_W2lv = struct[0].y_run[3,0]
    V_W3lv = struct[0].y_run[4,0]
    theta_W3lv = struct[0].y_run[5,0]
    V_STlv = struct[0].y_run[6,0]
    theta_STlv = struct[0].y_run[7,0]
    V_POIlv = struct[0].y_run[8,0]
    theta_POIlv = struct[0].y_run[9,0]
    V_W1mv = struct[0].y_run[10,0]
    theta_W1mv = struct[0].y_run[11,0]
    V_W2mv = struct[0].y_run[12,0]
    theta_W2mv = struct[0].y_run[13,0]
    V_W3mv = struct[0].y_run[14,0]
    theta_W3mv = struct[0].y_run[15,0]
    V_POImv = struct[0].y_run[16,0]
    theta_POImv = struct[0].y_run[17,0]
    V_STmv = struct[0].y_run[18,0]
    theta_STmv = struct[0].y_run[19,0]
    V_POI = struct[0].y_run[20,0]
    theta_POI = struct[0].y_run[21,0]
    V_GRID = struct[0].y_run[22,0]
    theta_GRID = struct[0].y_run[23,0]
    omega_GRID = struct[0].y_run[24,0]
    i_d_GRID = struct[0].y_run[25,0]
    i_q_GRID = struct[0].y_run[26,0]
    p_g_GRID = struct[0].y_run[27,0]
    q_g_GRID = struct[0].y_run[28,0]
    p_m_GRID = struct[0].y_run[29,0]
    omega_coi = struct[0].y_run[30,0]
    p_agc = struct[0].y_run[31,0]
    
    struct[0].u_run[0,0] = P_W1lv
    struct[0].u_run[1,0] = Q_W1lv
    struct[0].u_run[2,0] = P_W2lv
    struct[0].u_run[3,0] = Q_W2lv
    struct[0].u_run[4,0] = P_W3lv
    struct[0].u_run[5,0] = Q_W3lv
    struct[0].u_run[6,0] = P_STlv
    struct[0].u_run[7,0] = Q_STlv
    struct[0].u_run[8,0] = P_POIlv
    struct[0].u_run[9,0] = Q_POIlv
    struct[0].u_run[10,0] = P_W1mv
    struct[0].u_run[11,0] = Q_W1mv
    struct[0].u_run[12,0] = P_W2mv
    struct[0].u_run[13,0] = Q_W2mv
    struct[0].u_run[14,0] = P_W3mv
    struct[0].u_run[15,0] = Q_W3mv
    struct[0].u_run[16,0] = P_POImv
    struct[0].u_run[17,0] = Q_POImv
    struct[0].u_run[18,0] = P_STmv
    struct[0].u_run[19,0] = Q_STmv
    struct[0].u_run[20,0] = P_POI
    struct[0].u_run[21,0] = Q_POI
    struct[0].u_run[22,0] = P_GRID
    struct[0].u_run[23,0] = Q_GRID
    struct[0].u_run[24,0] = v_ref_GRID
    struct[0].u_run[25,0] = p_m_GRID
    struct[0].u_run[26,0] = p_c_GRID
    struct[0].u_run[27,0] = omega_ref_GRID
    struct[0].u_run[28,0] = q_ref_GRID
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRID*delta_GRID + Omega_b_GRID*(omega_GRID - omega_coi)
        struct[0].f[1,0] = -i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID
        struct[0].f[2,0] = (K_q_GRID*(-q_g_GRID + q_ref_GRID) - e_qv_GRID + v_ref_GRID)/T_v_GRID
        struct[0].f[3,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_W1lv/S_base + V_W1lv**2*g_W1mv_W1lv + V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].g[1,0] = -Q_W1lv/S_base + V_W1lv**2*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].g[2,0] = -P_W2lv/S_base + V_W2lv**2*g_W2mv_W2lv + V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].g[3,0] = -Q_W2lv/S_base + V_W2lv**2*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].g[4,0] = -P_W3lv/S_base + V_W3lv**2*g_W3mv_W3lv + V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].g[5,0] = -Q_W3lv/S_base + V_W3lv**2*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].g[6,0] = -P_STlv/S_base + V_STlv**2*g_STmv_STlv + V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].g[7,0] = -Q_STlv/S_base + V_STlv**2*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].g[10,0] = -P_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv**2*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].g[11,0] = -Q_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv**2*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].g[12,0] = -P_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv**2*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].g[13,0] = -Q_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv**2*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].g[14,0] = -P_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + V_W3mv**2*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].g[15,0] = -Q_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + V_W3mv**2*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].g[16,0] = -P_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv**2*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].g[17,0] = -Q_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv**2*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].g[18,0] = -P_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + V_STmv**2*(g_STmv_POImv + g_STmv_STlv)
        struct[0].g[19,0] = -Q_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + V_STmv**2*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].g[20,0] = -P_POI/S_base + V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI**2*(g_POI_GRID + g_POI_POImv) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].g[21,0] = -Q_POI/S_base + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI**2*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].g[22,0] = -P_GRID/S_base + V_GRID**2*g_POI_GRID + V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) - S_n_GRID*p_g_GRID/S_base
        struct[0].g[23,0] = -Q_GRID/S_base + V_GRID**2*(-b_POI_GRID - bs_POI_GRID/2) + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) - S_n_GRID*q_g_GRID/S_base
        struct[0].g[24,0] = K_p_GRID*(-i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID + xi_p_GRID/T_p_GRID) - omega_GRID + 1
        struct[0].g[25,0] = -R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID) + X_v_GRID*i_q_GRID
        struct[0].g[26,0] = -R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID) - X_v_GRID*i_d_GRID + e_qv_GRID
        struct[0].g[27,0] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID) - p_g_GRID
        struct[0].g[28,0] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID) - q_g_GRID
        struct[0].g[29,0] = K_sec_GRID*p_agc + p_c_GRID - p_m_GRID - (omega_GRID - omega_ref_GRID)/Droop_GRID
        struct[0].g[31,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_W1lv
        struct[0].h[1,0] = V_W2lv
        struct[0].h[2,0] = V_W3lv
        struct[0].h[3,0] = V_STlv
        struct[0].h[4,0] = V_POIlv
        struct[0].h[5,0] = V_W1mv
        struct[0].h[6,0] = V_W2mv
        struct[0].h[7,0] = V_W3mv
        struct[0].h[8,0] = V_POImv
        struct[0].h[9,0] = V_STmv
        struct[0].h[10,0] = V_POI
        struct[0].h[11,0] = V_GRID
        struct[0].h[12,0] = i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) + i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_GRID
        struct[0].Fx[1,0] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fx[2,2] = -1/T_v_GRID

    if mode == 11:

        struct[0].Fy[0,24] = Omega_b_GRID
        struct[0].Fy[0,30] = -Omega_b_GRID
        struct[0].Fy[1,22] = -i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Fy[1,23] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fy[1,25] = -2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fy[1,26] = -2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Fy[1,29] = 1
        struct[0].Fy[2,28] = -K_q_GRID/T_v_GRID
        struct[0].Fy[3,30] = -1

        struct[0].Gx[24,0] = K_p_GRID*(-V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gx[24,1] = K_p_GRID/T_p_GRID
        struct[0].Gx[25,0] = -V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gx[26,0] = V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gx[26,2] = 1
        struct[0].Gx[27,0] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gx[28,0] = -V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gx[31,3] = K_i_agc

        struct[0].Gy[0,0] = 2*V_W1lv*g_W1mv_W1lv + V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[0,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[0,10] = V_W1lv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[0,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,0] = 2*V_W1lv*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[2,2] = 2*V_W2lv*g_W2mv_W2lv + V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[2,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[2,12] = V_W2lv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[2,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,2] = 2*V_W2lv*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,12] = V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[4,4] = 2*V_W3lv*g_W3mv_W3lv + V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[4,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[4,14] = V_W3lv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[4,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,4] = 2*V_W3lv*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,14] = V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[6,6] = 2*V_STlv*g_STmv_STlv + V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[6,7] = V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[6,18] = V_STlv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[6,19] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[7,6] = 2*V_STlv*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[7,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[7,18] = V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[7,19] = V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[10,0] = V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[10,1] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[10,10] = V_W1lv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[10,11] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[10,12] = V_W1mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[10,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,0] = V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[11,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[11,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[12,2] = V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[12,3] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[12,10] = V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[12,11] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[12,12] = V_W1mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[12,13] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[12,14] = V_W2mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[12,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,2] = V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[13,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[13,10] = V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[13,11] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[13,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,14] = V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[14,4] = V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[14,5] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[14,12] = V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[14,13] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[14,14] = V_POImv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].Gy[14,15] = V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[14,16] = V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[14,17] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[15,4] = V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[15,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[15,12] = V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[15,13] = V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[15,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].Gy[15,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[15,16] = V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[15,17] = V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[16,14] = V_POImv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[16,15] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[16,16] = V_POI*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + 2*V_POImv*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[16,17] = V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[16,18] = V_POImv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[16,19] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[16,20] = V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[16,21] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[17,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[17,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[17,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + 2*V_POImv*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[17,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[17,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[17,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[17,20] = V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[17,21] = V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[18,6] = V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[18,7] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[18,16] = V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[18,17] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[18,18] = V_POImv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + 2*V_STmv*(g_STmv_POImv + g_STmv_STlv)
        struct[0].Gy[18,19] = V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[19,6] = V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[19,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[19,16] = V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[19,17] = V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[19,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + 2*V_STmv*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].Gy[19,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[20,16] = V_POI*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[20,17] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[20,20] = V_GRID*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + 2*V_POI*(g_POI_GRID + g_POI_POImv) + V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[20,21] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[20,22] = V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[20,23] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[21,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[21,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[21,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + 2*V_POI*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[21,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[21,22] = V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[21,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[22,20] = V_GRID*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[22,21] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[22,22] = 2*V_GRID*g_POI_GRID + V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[22,23] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[22,27] = -S_n_GRID/S_base
        struct[0].Gy[23,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[23,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[23,22] = 2*V_GRID*(-b_POI_GRID - bs_POI_GRID/2) + V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[23,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[23,28] = -S_n_GRID/S_base
        struct[0].Gy[24,22] = K_p_GRID*(-i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy[24,23] = K_p_GRID*(V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy[24,25] = K_p_GRID*(-2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy[24,26] = K_p_GRID*(-2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy[24,29] = K_p_GRID
        struct[0].Gy[25,22] = -sin(delta_GRID - theta_GRID)
        struct[0].Gy[25,23] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[25,25] = -R_v_GRID
        struct[0].Gy[25,26] = X_v_GRID
        struct[0].Gy[26,22] = -cos(delta_GRID - theta_GRID)
        struct[0].Gy[26,23] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[26,25] = -X_v_GRID
        struct[0].Gy[26,26] = -R_v_GRID
        struct[0].Gy[27,22] = i_d_GRID*sin(delta_GRID - theta_GRID) + i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[27,23] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[27,25] = V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[27,26] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[28,22] = i_d_GRID*cos(delta_GRID - theta_GRID) - i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[28,23] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[28,25] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[28,26] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[29,24] = -1/Droop_GRID
        struct[0].Gy[29,31] = K_sec_GRID
        struct[0].Gy[31,30] = -K_p_agc

    if mode > 12:

        struct[0].Fu[1,25] = 1
        struct[0].Fu[2,24] = 1/T_v_GRID
        struct[0].Fu[2,28] = K_q_GRID/T_v_GRID

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
        struct[0].Gu[24,25] = K_p_GRID
        struct[0].Gu[29,27] = 1/Droop_GRID

        struct[0].Hx[12,0] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)

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
        struct[0].Hy[12,22] = i_d_GRID*sin(delta_GRID - theta_GRID) + i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Hy[12,23] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Hy[12,25] = 2*R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Hy[12,26] = 2*R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)




def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_W1mv_W2mv = struct[0].g_W1mv_W2mv
    b_W1mv_W2mv = struct[0].b_W1mv_W2mv
    bs_W1mv_W2mv = struct[0].bs_W1mv_W2mv
    g_W2mv_W3mv = struct[0].g_W2mv_W3mv
    b_W2mv_W3mv = struct[0].b_W2mv_W3mv
    bs_W2mv_W3mv = struct[0].bs_W2mv_W3mv
    g_W3mv_POImv = struct[0].g_W3mv_POImv
    b_W3mv_POImv = struct[0].b_W3mv_POImv
    bs_W3mv_POImv = struct[0].bs_W3mv_POImv
    g_STmv_POImv = struct[0].g_STmv_POImv
    b_STmv_POImv = struct[0].b_STmv_POImv
    bs_STmv_POImv = struct[0].bs_STmv_POImv
    g_POI_GRID = struct[0].g_POI_GRID
    b_POI_GRID = struct[0].b_POI_GRID
    bs_POI_GRID = struct[0].bs_POI_GRID
    g_POI_POImv = struct[0].g_POI_POImv
    b_POI_POImv = struct[0].b_POI_POImv
    bs_POI_POImv = struct[0].bs_POI_POImv
    g_W1mv_W1lv = struct[0].g_W1mv_W1lv
    b_W1mv_W1lv = struct[0].b_W1mv_W1lv
    bs_W1mv_W1lv = struct[0].bs_W1mv_W1lv
    g_W2mv_W2lv = struct[0].g_W2mv_W2lv
    b_W2mv_W2lv = struct[0].b_W2mv_W2lv
    bs_W2mv_W2lv = struct[0].bs_W2mv_W2lv
    g_W3mv_W3lv = struct[0].g_W3mv_W3lv
    b_W3mv_W3lv = struct[0].b_W3mv_W3lv
    bs_W3mv_W3lv = struct[0].bs_W3mv_W3lv
    g_STmv_STlv = struct[0].g_STmv_STlv
    b_STmv_STlv = struct[0].b_STmv_STlv
    bs_STmv_STlv = struct[0].bs_STmv_STlv
    U_W1lv_n = struct[0].U_W1lv_n
    U_W2lv_n = struct[0].U_W2lv_n
    U_W3lv_n = struct[0].U_W3lv_n
    U_STlv_n = struct[0].U_STlv_n
    U_POIlv_n = struct[0].U_POIlv_n
    U_W1mv_n = struct[0].U_W1mv_n
    U_W2mv_n = struct[0].U_W2mv_n
    U_W3mv_n = struct[0].U_W3mv_n
    U_POImv_n = struct[0].U_POImv_n
    U_STmv_n = struct[0].U_STmv_n
    U_POI_n = struct[0].U_POI_n
    U_GRID_n = struct[0].U_GRID_n
    S_n_GRID = struct[0].S_n_GRID
    Omega_b_GRID = struct[0].Omega_b_GRID
    K_p_GRID = struct[0].K_p_GRID
    T_p_GRID = struct[0].T_p_GRID
    K_q_GRID = struct[0].K_q_GRID
    T_v_GRID = struct[0].T_v_GRID
    X_v_GRID = struct[0].X_v_GRID
    R_v_GRID = struct[0].R_v_GRID
    K_delta_GRID = struct[0].K_delta_GRID
    K_sec_GRID = struct[0].K_sec_GRID
    Droop_GRID = struct[0].Droop_GRID
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_W1lv = struct[0].P_W1lv
    Q_W1lv = struct[0].Q_W1lv
    P_W2lv = struct[0].P_W2lv
    Q_W2lv = struct[0].Q_W2lv
    P_W3lv = struct[0].P_W3lv
    Q_W3lv = struct[0].Q_W3lv
    P_STlv = struct[0].P_STlv
    Q_STlv = struct[0].Q_STlv
    P_POIlv = struct[0].P_POIlv
    Q_POIlv = struct[0].Q_POIlv
    P_W1mv = struct[0].P_W1mv
    Q_W1mv = struct[0].Q_W1mv
    P_W2mv = struct[0].P_W2mv
    Q_W2mv = struct[0].Q_W2mv
    P_W3mv = struct[0].P_W3mv
    Q_W3mv = struct[0].Q_W3mv
    P_POImv = struct[0].P_POImv
    Q_POImv = struct[0].Q_POImv
    P_STmv = struct[0].P_STmv
    Q_STmv = struct[0].Q_STmv
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_GRID = struct[0].P_GRID
    Q_GRID = struct[0].Q_GRID
    v_ref_GRID = struct[0].v_ref_GRID
    p_m_GRID = struct[0].p_m_GRID
    p_c_GRID = struct[0].p_c_GRID
    omega_ref_GRID = struct[0].omega_ref_GRID
    q_ref_GRID = struct[0].q_ref_GRID
    
    # Dynamical states:
    delta_GRID = struct[0].x[0,0]
    xi_p_GRID = struct[0].x[1,0]
    e_qv_GRID = struct[0].x[2,0]
    xi_freq = struct[0].x[3,0]
    
    # Algebraic states:
    V_W1lv = struct[0].y_ini[0,0]
    theta_W1lv = struct[0].y_ini[1,0]
    V_W2lv = struct[0].y_ini[2,0]
    theta_W2lv = struct[0].y_ini[3,0]
    V_W3lv = struct[0].y_ini[4,0]
    theta_W3lv = struct[0].y_ini[5,0]
    V_STlv = struct[0].y_ini[6,0]
    theta_STlv = struct[0].y_ini[7,0]
    V_POIlv = struct[0].y_ini[8,0]
    theta_POIlv = struct[0].y_ini[9,0]
    V_W1mv = struct[0].y_ini[10,0]
    theta_W1mv = struct[0].y_ini[11,0]
    V_W2mv = struct[0].y_ini[12,0]
    theta_W2mv = struct[0].y_ini[13,0]
    V_W3mv = struct[0].y_ini[14,0]
    theta_W3mv = struct[0].y_ini[15,0]
    V_POImv = struct[0].y_ini[16,0]
    theta_POImv = struct[0].y_ini[17,0]
    V_STmv = struct[0].y_ini[18,0]
    theta_STmv = struct[0].y_ini[19,0]
    V_POI = struct[0].y_ini[20,0]
    theta_POI = struct[0].y_ini[21,0]
    V_GRID = struct[0].y_ini[22,0]
    theta_GRID = struct[0].y_ini[23,0]
    omega_GRID = struct[0].y_ini[24,0]
    i_d_GRID = struct[0].y_ini[25,0]
    i_q_GRID = struct[0].y_ini[26,0]
    p_g_GRID = struct[0].y_ini[27,0]
    q_g_GRID = struct[0].y_ini[28,0]
    p_m_GRID = struct[0].y_ini[29,0]
    omega_coi = struct[0].y_ini[30,0]
    p_agc = struct[0].y_ini[31,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRID*delta_GRID + Omega_b_GRID*(omega_GRID - omega_coi)
        struct[0].f[1,0] = -i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID
        struct[0].f[2,0] = (K_q_GRID*(-q_g_GRID + q_ref_GRID) - e_qv_GRID + v_ref_GRID)/T_v_GRID
        struct[0].f[3,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_W1lv/S_base + V_W1lv**2*g_W1mv_W1lv + V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].g[1,0] = -Q_W1lv/S_base + V_W1lv**2*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].g[2,0] = -P_W2lv/S_base + V_W2lv**2*g_W2mv_W2lv + V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].g[3,0] = -Q_W2lv/S_base + V_W2lv**2*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].g[4,0] = -P_W3lv/S_base + V_W3lv**2*g_W3mv_W3lv + V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].g[5,0] = -Q_W3lv/S_base + V_W3lv**2*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].g[6,0] = -P_STlv/S_base + V_STlv**2*g_STmv_STlv + V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].g[7,0] = -Q_STlv/S_base + V_STlv**2*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].g[8,0] = -P_POIlv/S_base
        struct[0].g[9,0] = -Q_POIlv/S_base
        struct[0].g[10,0] = -P_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv**2*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].g[11,0] = -Q_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv**2*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].g[12,0] = -P_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv**2*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].g[13,0] = -Q_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv**2*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].g[14,0] = -P_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + V_W3mv**2*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].g[15,0] = -Q_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + V_W3mv**2*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].g[16,0] = -P_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv**2*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].g[17,0] = -Q_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv**2*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].g[18,0] = -P_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + V_STmv**2*(g_STmv_POImv + g_STmv_STlv)
        struct[0].g[19,0] = -Q_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + V_STmv**2*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].g[20,0] = -P_POI/S_base + V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI**2*(g_POI_GRID + g_POI_POImv) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].g[21,0] = -Q_POI/S_base + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI**2*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].g[22,0] = -P_GRID/S_base + V_GRID**2*g_POI_GRID + V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) - S_n_GRID*p_g_GRID/S_base
        struct[0].g[23,0] = -Q_GRID/S_base + V_GRID**2*(-b_POI_GRID - bs_POI_GRID/2) + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) - S_n_GRID*q_g_GRID/S_base
        struct[0].g[24,0] = K_p_GRID*(-i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID + xi_p_GRID/T_p_GRID) - omega_GRID + 1
        struct[0].g[25,0] = -R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID) + X_v_GRID*i_q_GRID
        struct[0].g[26,0] = -R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID) - X_v_GRID*i_d_GRID + e_qv_GRID
        struct[0].g[27,0] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID) - p_g_GRID
        struct[0].g[28,0] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID) - q_g_GRID
        struct[0].g[29,0] = K_sec_GRID*p_agc + p_c_GRID - p_m_GRID - (omega_GRID - omega_ref_GRID)/Droop_GRID
        struct[0].g[30,0] = omega_GRID - omega_coi
        struct[0].g[31,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_W1lv
        struct[0].h[1,0] = V_W2lv
        struct[0].h[2,0] = V_W3lv
        struct[0].h[3,0] = V_STlv
        struct[0].h[4,0] = V_POIlv
        struct[0].h[5,0] = V_W1mv
        struct[0].h[6,0] = V_W2mv
        struct[0].h[7,0] = V_W3mv
        struct[0].h[8,0] = V_POImv
        struct[0].h[9,0] = V_STmv
        struct[0].h[10,0] = V_POI
        struct[0].h[11,0] = V_GRID
        struct[0].h[12,0] = i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) + i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID))
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_GRID
        struct[0].Fx_ini[1,0] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fx_ini[2,2] = -1/T_v_GRID

    if mode == 11:

        struct[0].Fy_ini[0,24] = Omega_b_GRID 
        struct[0].Fy_ini[0,30] = -Omega_b_GRID 
        struct[0].Fy_ini[1,22] = -i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,23] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,25] = -2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,26] = -2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID) 
        struct[0].Fy_ini[1,29] = 1 
        struct[0].Fy_ini[2,28] = -K_q_GRID/T_v_GRID 
        struct[0].Fy_ini[3,30] = -1 

        struct[0].Gy_ini[0,0] = 2*V_W1lv*g_W1mv_W1lv + V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[0,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[0,10] = V_W1lv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[0,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,0] = 2*V_W1lv*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[1,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[2,2] = 2*V_W2lv*g_W2mv_W2lv + V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[2,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[2,12] = V_W2lv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[2,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,2] = 2*V_W2lv*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,12] = V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[3,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[4,4] = 2*V_W3lv*g_W3mv_W3lv + V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[4,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[4,14] = V_W3lv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[4,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,4] = 2*V_W3lv*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,14] = V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[5,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[6,6] = 2*V_STlv*g_STmv_STlv + V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[6,7] = V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[6,18] = V_STlv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[6,19] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,6] = 2*V_STlv*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,18] = V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[7,19] = V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[10,0] = V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[10,1] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[10,10] = V_W1lv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[10,11] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[10,12] = V_W1mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[10,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,0] = V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[11,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy_ini[11,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[11,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[12,2] = V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[12,3] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[12,10] = V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[12,11] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[12,12] = V_W1mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[12,13] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[12,14] = V_W2mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[12,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,2] = V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[13,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy_ini[13,10] = V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[13,11] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy_ini[13,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,14] = V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[13,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[14,4] = V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[14,5] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[14,12] = V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[14,13] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[14,14] = V_POImv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].Gy_ini[14,15] = V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[14,16] = V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[14,17] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[15,4] = V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[15,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[15,12] = V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[15,13] = V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy_ini[15,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].Gy_ini[15,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy_ini[15,16] = V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[15,17] = V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,14] = V_POImv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,15] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,16] = V_POI*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + 2*V_POImv*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,17] = V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[16,18] = V_POImv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[16,19] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[16,20] = V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[16,21] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[17,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + 2*V_POImv*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy_ini[17,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[17,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[17,20] = V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[17,21] = V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[18,6] = V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[18,7] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[18,16] = V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[18,17] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[18,18] = V_POImv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + 2*V_STmv*(g_STmv_POImv + g_STmv_STlv)
        struct[0].Gy_ini[18,19] = V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[19,6] = V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy_ini[19,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[19,16] = V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy_ini[19,17] = V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy_ini[19,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + 2*V_STmv*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].Gy_ini[19,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy_ini[20,16] = V_POI*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,17] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,20] = V_GRID*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + 2*V_POI*(g_POI_GRID + g_POI_POImv) + V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,21] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[20,22] = V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[20,23] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[21,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + 2*V_POI*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy_ini[21,22] = V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[21,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,20] = V_GRID*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,21] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,22] = 2*V_GRID*g_POI_GRID + V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,23] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[22,27] = -S_n_GRID/S_base
        struct[0].Gy_ini[23,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,22] = 2*V_GRID*(-b_POI_GRID - bs_POI_GRID/2) + V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy_ini[23,28] = -S_n_GRID/S_base
        struct[0].Gy_ini[24,22] = K_p_GRID*(-i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,23] = K_p_GRID*(V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[24,25] = K_p_GRID*(-2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,26] = K_p_GRID*(-2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy_ini[24,29] = K_p_GRID
        struct[0].Gy_ini[25,22] = -sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[25,23] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[25,25] = -R_v_GRID
        struct[0].Gy_ini[25,26] = X_v_GRID
        struct[0].Gy_ini[26,22] = -cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[26,23] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[26,25] = -X_v_GRID
        struct[0].Gy_ini[26,26] = -R_v_GRID
        struct[0].Gy_ini[27,22] = i_d_GRID*sin(delta_GRID - theta_GRID) + i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[27,23] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[27,25] = V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[27,26] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[27,27] = -1
        struct[0].Gy_ini[28,22] = i_d_GRID*cos(delta_GRID - theta_GRID) - i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,23] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,25] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,26] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy_ini[28,28] = -1
        struct[0].Gy_ini[29,24] = -1/Droop_GRID
        struct[0].Gy_ini[29,29] = -1
        struct[0].Gy_ini[29,31] = K_sec_GRID
        struct[0].Gy_ini[30,24] = 1
        struct[0].Gy_ini[30,30] = -1
        struct[0].Gy_ini[31,30] = -K_p_agc
        struct[0].Gy_ini[31,31] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_W1mv_W2mv = struct[0].g_W1mv_W2mv
    b_W1mv_W2mv = struct[0].b_W1mv_W2mv
    bs_W1mv_W2mv = struct[0].bs_W1mv_W2mv
    g_W2mv_W3mv = struct[0].g_W2mv_W3mv
    b_W2mv_W3mv = struct[0].b_W2mv_W3mv
    bs_W2mv_W3mv = struct[0].bs_W2mv_W3mv
    g_W3mv_POImv = struct[0].g_W3mv_POImv
    b_W3mv_POImv = struct[0].b_W3mv_POImv
    bs_W3mv_POImv = struct[0].bs_W3mv_POImv
    g_STmv_POImv = struct[0].g_STmv_POImv
    b_STmv_POImv = struct[0].b_STmv_POImv
    bs_STmv_POImv = struct[0].bs_STmv_POImv
    g_POI_GRID = struct[0].g_POI_GRID
    b_POI_GRID = struct[0].b_POI_GRID
    bs_POI_GRID = struct[0].bs_POI_GRID
    g_POI_POImv = struct[0].g_POI_POImv
    b_POI_POImv = struct[0].b_POI_POImv
    bs_POI_POImv = struct[0].bs_POI_POImv
    g_W1mv_W1lv = struct[0].g_W1mv_W1lv
    b_W1mv_W1lv = struct[0].b_W1mv_W1lv
    bs_W1mv_W1lv = struct[0].bs_W1mv_W1lv
    g_W2mv_W2lv = struct[0].g_W2mv_W2lv
    b_W2mv_W2lv = struct[0].b_W2mv_W2lv
    bs_W2mv_W2lv = struct[0].bs_W2mv_W2lv
    g_W3mv_W3lv = struct[0].g_W3mv_W3lv
    b_W3mv_W3lv = struct[0].b_W3mv_W3lv
    bs_W3mv_W3lv = struct[0].bs_W3mv_W3lv
    g_STmv_STlv = struct[0].g_STmv_STlv
    b_STmv_STlv = struct[0].b_STmv_STlv
    bs_STmv_STlv = struct[0].bs_STmv_STlv
    U_W1lv_n = struct[0].U_W1lv_n
    U_W2lv_n = struct[0].U_W2lv_n
    U_W3lv_n = struct[0].U_W3lv_n
    U_STlv_n = struct[0].U_STlv_n
    U_POIlv_n = struct[0].U_POIlv_n
    U_W1mv_n = struct[0].U_W1mv_n
    U_W2mv_n = struct[0].U_W2mv_n
    U_W3mv_n = struct[0].U_W3mv_n
    U_POImv_n = struct[0].U_POImv_n
    U_STmv_n = struct[0].U_STmv_n
    U_POI_n = struct[0].U_POI_n
    U_GRID_n = struct[0].U_GRID_n
    S_n_GRID = struct[0].S_n_GRID
    Omega_b_GRID = struct[0].Omega_b_GRID
    K_p_GRID = struct[0].K_p_GRID
    T_p_GRID = struct[0].T_p_GRID
    K_q_GRID = struct[0].K_q_GRID
    T_v_GRID = struct[0].T_v_GRID
    X_v_GRID = struct[0].X_v_GRID
    R_v_GRID = struct[0].R_v_GRID
    K_delta_GRID = struct[0].K_delta_GRID
    K_sec_GRID = struct[0].K_sec_GRID
    Droop_GRID = struct[0].Droop_GRID
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_W1lv = struct[0].P_W1lv
    Q_W1lv = struct[0].Q_W1lv
    P_W2lv = struct[0].P_W2lv
    Q_W2lv = struct[0].Q_W2lv
    P_W3lv = struct[0].P_W3lv
    Q_W3lv = struct[0].Q_W3lv
    P_STlv = struct[0].P_STlv
    Q_STlv = struct[0].Q_STlv
    P_POIlv = struct[0].P_POIlv
    Q_POIlv = struct[0].Q_POIlv
    P_W1mv = struct[0].P_W1mv
    Q_W1mv = struct[0].Q_W1mv
    P_W2mv = struct[0].P_W2mv
    Q_W2mv = struct[0].Q_W2mv
    P_W3mv = struct[0].P_W3mv
    Q_W3mv = struct[0].Q_W3mv
    P_POImv = struct[0].P_POImv
    Q_POImv = struct[0].Q_POImv
    P_STmv = struct[0].P_STmv
    Q_STmv = struct[0].Q_STmv
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_GRID = struct[0].P_GRID
    Q_GRID = struct[0].Q_GRID
    v_ref_GRID = struct[0].v_ref_GRID
    p_m_GRID = struct[0].p_m_GRID
    p_c_GRID = struct[0].p_c_GRID
    omega_ref_GRID = struct[0].omega_ref_GRID
    q_ref_GRID = struct[0].q_ref_GRID
    
    # Dynamical states:
    delta_GRID = struct[0].x[0,0]
    xi_p_GRID = struct[0].x[1,0]
    e_qv_GRID = struct[0].x[2,0]
    xi_freq = struct[0].x[3,0]
    
    # Algebraic states:
    V_W1lv = struct[0].y_run[0,0]
    theta_W1lv = struct[0].y_run[1,0]
    V_W2lv = struct[0].y_run[2,0]
    theta_W2lv = struct[0].y_run[3,0]
    V_W3lv = struct[0].y_run[4,0]
    theta_W3lv = struct[0].y_run[5,0]
    V_STlv = struct[0].y_run[6,0]
    theta_STlv = struct[0].y_run[7,0]
    V_POIlv = struct[0].y_run[8,0]
    theta_POIlv = struct[0].y_run[9,0]
    V_W1mv = struct[0].y_run[10,0]
    theta_W1mv = struct[0].y_run[11,0]
    V_W2mv = struct[0].y_run[12,0]
    theta_W2mv = struct[0].y_run[13,0]
    V_W3mv = struct[0].y_run[14,0]
    theta_W3mv = struct[0].y_run[15,0]
    V_POImv = struct[0].y_run[16,0]
    theta_POImv = struct[0].y_run[17,0]
    V_STmv = struct[0].y_run[18,0]
    theta_STmv = struct[0].y_run[19,0]
    V_POI = struct[0].y_run[20,0]
    theta_POI = struct[0].y_run[21,0]
    V_GRID = struct[0].y_run[22,0]
    theta_GRID = struct[0].y_run[23,0]
    omega_GRID = struct[0].y_run[24,0]
    i_d_GRID = struct[0].y_run[25,0]
    i_q_GRID = struct[0].y_run[26,0]
    p_g_GRID = struct[0].y_run[27,0]
    q_g_GRID = struct[0].y_run[28,0]
    p_m_GRID = struct[0].y_run[29,0]
    omega_coi = struct[0].y_run[30,0]
    p_agc = struct[0].y_run[31,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRID*delta_GRID + Omega_b_GRID*(omega_GRID - omega_coi)
        struct[0].f[1,0] = -i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID
        struct[0].f[2,0] = (K_q_GRID*(-q_g_GRID + q_ref_GRID) - e_qv_GRID + v_ref_GRID)/T_v_GRID
        struct[0].f[3,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_W1lv/S_base + V_W1lv**2*g_W1mv_W1lv + V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].g[1,0] = -Q_W1lv/S_base + V_W1lv**2*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].g[2,0] = -P_W2lv/S_base + V_W2lv**2*g_W2mv_W2lv + V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].g[3,0] = -Q_W2lv/S_base + V_W2lv**2*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].g[4,0] = -P_W3lv/S_base + V_W3lv**2*g_W3mv_W3lv + V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].g[5,0] = -Q_W3lv/S_base + V_W3lv**2*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].g[6,0] = -P_STlv/S_base + V_STlv**2*g_STmv_STlv + V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].g[7,0] = -Q_STlv/S_base + V_STlv**2*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].g[8,0] = -P_POIlv/S_base
        struct[0].g[9,0] = -Q_POIlv/S_base
        struct[0].g[10,0] = -P_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv**2*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].g[11,0] = -Q_W1mv/S_base + V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv**2*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].g[12,0] = -P_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv**2*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].g[13,0] = -Q_W2mv/S_base + V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv**2*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].g[14,0] = -P_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + V_W3mv**2*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].g[15,0] = -Q_W3mv/S_base + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + V_W3mv**2*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].g[16,0] = -P_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv**2*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].g[17,0] = -Q_POImv/S_base + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv**2*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].g[18,0] = -P_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + V_STmv**2*(g_STmv_POImv + g_STmv_STlv)
        struct[0].g[19,0] = -Q_STmv/S_base + V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + V_STmv**2*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].g[20,0] = -P_POI/S_base + V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI**2*(g_POI_GRID + g_POI_POImv) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].g[21,0] = -Q_POI/S_base + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI**2*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].g[22,0] = -P_GRID/S_base + V_GRID**2*g_POI_GRID + V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) - S_n_GRID*p_g_GRID/S_base
        struct[0].g[23,0] = -Q_GRID/S_base + V_GRID**2*(-b_POI_GRID - bs_POI_GRID/2) + V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) - S_n_GRID*q_g_GRID/S_base
        struct[0].g[24,0] = K_p_GRID*(-i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) - i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID)) + p_m_GRID + xi_p_GRID/T_p_GRID) - omega_GRID + 1
        struct[0].g[25,0] = -R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID) + X_v_GRID*i_q_GRID
        struct[0].g[26,0] = -R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID) - X_v_GRID*i_d_GRID + e_qv_GRID
        struct[0].g[27,0] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID) - p_g_GRID
        struct[0].g[28,0] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID) - q_g_GRID
        struct[0].g[29,0] = K_sec_GRID*p_agc + p_c_GRID - p_m_GRID - (omega_GRID - omega_ref_GRID)/Droop_GRID
        struct[0].g[30,0] = omega_GRID - omega_coi
        struct[0].g[31,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_W1lv
        struct[0].h[1,0] = V_W2lv
        struct[0].h[2,0] = V_W3lv
        struct[0].h[3,0] = V_STlv
        struct[0].h[4,0] = V_POIlv
        struct[0].h[5,0] = V_W1mv
        struct[0].h[6,0] = V_W2mv
        struct[0].h[7,0] = V_W3mv
        struct[0].h[8,0] = V_POImv
        struct[0].h[9,0] = V_STmv
        struct[0].h[10,0] = V_POI
        struct[0].h[11,0] = V_GRID
        struct[0].h[12,0] = i_d_GRID*(R_v_GRID*i_d_GRID + V_GRID*sin(delta_GRID - theta_GRID)) + i_q_GRID*(R_v_GRID*i_q_GRID + V_GRID*cos(delta_GRID - theta_GRID))
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_GRID
        struct[0].Fx[1,0] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fx[2,2] = -1/T_v_GRID

    if mode == 11:

        struct[0].Fy[0,24] = Omega_b_GRID
        struct[0].Fy[0,30] = -Omega_b_GRID
        struct[0].Fy[1,22] = -i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Fy[1,23] = V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fy[1,25] = -2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Fy[1,26] = -2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Fy[1,29] = 1
        struct[0].Fy[2,28] = -K_q_GRID/T_v_GRID
        struct[0].Fy[3,30] = -1

        struct[0].Gy[0,0] = 2*V_W1lv*g_W1mv_W1lv + V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[0,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[0,10] = V_W1lv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[0,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,0] = 2*V_W1lv*(-b_W1mv_W1lv - bs_W1mv_W1lv/2) + V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[1,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[2,2] = 2*V_W2lv*g_W2mv_W2lv + V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[2,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[2,12] = V_W2lv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[2,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,2] = 2*V_W2lv*(-b_W2mv_W2lv - bs_W2mv_W2lv/2) + V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,12] = V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[3,13] = V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[4,4] = 2*V_W3lv*g_W3mv_W3lv + V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[4,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[4,14] = V_W3lv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[4,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,4] = 2*V_W3lv*(-b_W3mv_W3lv - bs_W3mv_W3lv/2) + V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,14] = V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[5,15] = V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[6,6] = 2*V_STlv*g_STmv_STlv + V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[6,7] = V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[6,18] = V_STlv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[6,19] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[7,6] = 2*V_STlv*(-b_STmv_STlv - bs_STmv_STlv/2) + V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[7,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[7,18] = V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[7,19] = V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[10,0] = V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[10,1] = V_W1lv*V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[10,10] = V_W1lv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(g_W1mv_W1lv + g_W1mv_W2mv) + V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[10,11] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[10,12] = V_W1mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[10,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,0] = V_W1mv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv))
        struct[0].Gy[11,1] = V_W1lv*V_W1mv*(-b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv))
        struct[0].Gy[11,10] = V_W1lv*(b_W1mv_W1lv*cos(theta_W1lv - theta_W1mv) + g_W1mv_W1lv*sin(theta_W1lv - theta_W1mv)) + 2*V_W1mv*(-b_W1mv_W1lv - b_W1mv_W2mv - bs_W1mv_W1lv/2 - bs_W1mv_W2mv/2) + V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,11] = V_W1lv*V_W1mv*(b_W1mv_W1lv*sin(theta_W1lv - theta_W1mv) - g_W1mv_W1lv*cos(theta_W1lv - theta_W1mv)) + V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[11,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[12,2] = V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[12,3] = V_W2lv*V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[12,10] = V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[12,11] = V_W1mv*V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[12,12] = V_W1mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(g_W1mv_W2mv + g_W2mv_W2lv + g_W2mv_W3mv) + V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[12,13] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(-b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[12,14] = V_W2mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[12,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,2] = V_W2mv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv))
        struct[0].Gy[13,3] = V_W2lv*V_W2mv*(-b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv))
        struct[0].Gy[13,10] = V_W2mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv))
        struct[0].Gy[13,11] = V_W1mv*V_W2mv*(-b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv))
        struct[0].Gy[13,12] = V_W1mv*(b_W1mv_W2mv*cos(theta_W1mv - theta_W2mv) + g_W1mv_W2mv*sin(theta_W1mv - theta_W2mv)) + V_W2lv*(b_W2mv_W2lv*cos(theta_W2lv - theta_W2mv) + g_W2mv_W2lv*sin(theta_W2lv - theta_W2mv)) + 2*V_W2mv*(-b_W1mv_W2mv - b_W2mv_W2lv - b_W2mv_W3mv - bs_W1mv_W2mv/2 - bs_W2mv_W2lv/2 - bs_W2mv_W3mv/2) + V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,13] = V_W1mv*V_W2mv*(b_W1mv_W2mv*sin(theta_W1mv - theta_W2mv) - g_W1mv_W2mv*cos(theta_W1mv - theta_W2mv)) + V_W2lv*V_W2mv*(b_W2mv_W2lv*sin(theta_W2lv - theta_W2mv) - g_W2mv_W2lv*cos(theta_W2lv - theta_W2mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,14] = V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[13,15] = V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[14,4] = V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[14,5] = V_W3lv*V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[14,12] = V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[14,13] = V_W2mv*V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[14,14] = V_POImv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(g_W2mv_W3mv + g_W3mv_POImv + g_W3mv_W3lv)
        struct[0].Gy[14,15] = V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(-b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(-b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[14,16] = V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[14,17] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[15,4] = V_W3mv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv))
        struct[0].Gy[15,5] = V_W3lv*V_W3mv*(-b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[15,12] = V_W3mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv))
        struct[0].Gy[15,13] = V_W2mv*V_W3mv*(-b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv))
        struct[0].Gy[15,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv)) + V_W2mv*(b_W2mv_W3mv*cos(theta_W2mv - theta_W3mv) + g_W2mv_W3mv*sin(theta_W2mv - theta_W3mv)) + V_W3lv*(b_W3mv_W3lv*cos(theta_W3lv - theta_W3mv) + g_W3mv_W3lv*sin(theta_W3lv - theta_W3mv)) + 2*V_W3mv*(-b_W2mv_W3mv - b_W3mv_POImv - b_W3mv_W3lv - bs_W2mv_W3mv/2 - bs_W3mv_POImv/2 - bs_W3mv_W3lv/2)
        struct[0].Gy[15,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv)) + V_W2mv*V_W3mv*(b_W2mv_W3mv*sin(theta_W2mv - theta_W3mv) - g_W2mv_W3mv*cos(theta_W2mv - theta_W3mv)) + V_W3lv*V_W3mv*(b_W3mv_W3lv*sin(theta_W3lv - theta_W3mv) - g_W3mv_W3lv*cos(theta_W3lv - theta_W3mv))
        struct[0].Gy[15,16] = V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[15,17] = V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[16,14] = V_POImv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[16,15] = V_POImv*V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[16,16] = V_POI*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + 2*V_POImv*(g_POI_POImv + g_STmv_POImv + g_W3mv_POImv) + V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[16,17] = V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*cos(theta_POImv - theta_W3mv) + g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[16,18] = V_POImv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[16,19] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[16,20] = V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[16,21] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[17,14] = V_POImv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[17,15] = V_POImv*V_W3mv*(b_W3mv_POImv*sin(theta_POImv - theta_W3mv) + g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[17,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv)) + 2*V_POImv*(-b_POI_POImv - b_STmv_POImv - b_W3mv_POImv - bs_POI_POImv/2 - bs_STmv_POImv/2 - bs_W3mv_POImv/2) + V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_W3mv*(b_W3mv_POImv*cos(theta_POImv - theta_W3mv) - g_W3mv_POImv*sin(theta_POImv - theta_W3mv))
        struct[0].Gy[17,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv)) + V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_POImv*V_W3mv*(-b_W3mv_POImv*sin(theta_POImv - theta_W3mv) - g_W3mv_POImv*cos(theta_POImv - theta_W3mv))
        struct[0].Gy[17,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[17,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[17,20] = V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[17,21] = V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[18,6] = V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[18,7] = V_STlv*V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[18,16] = V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[18,17] = V_POImv*V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[18,18] = V_POImv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv)) + 2*V_STmv*(g_STmv_POImv + g_STmv_STlv)
        struct[0].Gy[18,19] = V_POImv*V_STmv*(-b_STmv_POImv*cos(theta_POImv - theta_STmv) - g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(-b_STmv_STlv*cos(theta_STlv - theta_STmv) - g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[19,6] = V_STmv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv))
        struct[0].Gy[19,7] = V_STlv*V_STmv*(-b_STmv_STlv*sin(theta_STlv - theta_STmv) + g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[19,16] = V_STmv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv))
        struct[0].Gy[19,17] = V_POImv*V_STmv*(-b_STmv_POImv*sin(theta_POImv - theta_STmv) + g_STmv_POImv*cos(theta_POImv - theta_STmv))
        struct[0].Gy[19,18] = V_POImv*(b_STmv_POImv*cos(theta_POImv - theta_STmv) + g_STmv_POImv*sin(theta_POImv - theta_STmv)) + V_STlv*(b_STmv_STlv*cos(theta_STlv - theta_STmv) + g_STmv_STlv*sin(theta_STlv - theta_STmv)) + 2*V_STmv*(-b_STmv_POImv - b_STmv_STlv - bs_STmv_POImv/2 - bs_STmv_STlv/2)
        struct[0].Gy[19,19] = V_POImv*V_STmv*(b_STmv_POImv*sin(theta_POImv - theta_STmv) - g_STmv_POImv*cos(theta_POImv - theta_STmv)) + V_STlv*V_STmv*(b_STmv_STlv*sin(theta_STlv - theta_STmv) - g_STmv_STlv*cos(theta_STlv - theta_STmv))
        struct[0].Gy[20,16] = V_POI*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[20,17] = V_POI*V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[20,20] = V_GRID*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + 2*V_POI*(g_POI_GRID + g_POI_POImv) + V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[20,21] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*cos(theta_POI - theta_POImv) + g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[20,22] = V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[20,23] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[21,16] = V_POI*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[21,17] = V_POI*V_POImv*(b_POI_POImv*sin(theta_POI - theta_POImv) + g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[21,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI)) + 2*V_POI*(-b_POI_GRID - b_POI_POImv - bs_POI_GRID/2 - bs_POI_POImv/2) + V_POImv*(b_POI_POImv*cos(theta_POI - theta_POImv) - g_POI_POImv*sin(theta_POI - theta_POImv))
        struct[0].Gy[21,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI)) + V_POI*V_POImv*(-b_POI_POImv*sin(theta_POI - theta_POImv) - g_POI_POImv*cos(theta_POI - theta_POImv))
        struct[0].Gy[21,22] = V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[21,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[22,20] = V_GRID*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[22,21] = V_GRID*V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[22,22] = 2*V_GRID*g_POI_GRID + V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[22,23] = V_GRID*V_POI*(-b_POI_GRID*cos(theta_GRID - theta_POI) + g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[22,27] = -S_n_GRID/S_base
        struct[0].Gy[23,20] = V_GRID*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[23,21] = V_GRID*V_POI*(b_POI_GRID*sin(theta_GRID - theta_POI) + g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[23,22] = 2*V_GRID*(-b_POI_GRID - bs_POI_GRID/2) + V_POI*(b_POI_GRID*cos(theta_GRID - theta_POI) - g_POI_GRID*sin(theta_GRID - theta_POI))
        struct[0].Gy[23,23] = V_GRID*V_POI*(-b_POI_GRID*sin(theta_GRID - theta_POI) - g_POI_GRID*cos(theta_GRID - theta_POI))
        struct[0].Gy[23,28] = -S_n_GRID/S_base
        struct[0].Gy[24,22] = K_p_GRID*(-i_d_GRID*sin(delta_GRID - theta_GRID) - i_q_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy[24,23] = K_p_GRID*(V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) - V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy[24,24] = -1
        struct[0].Gy[24,25] = K_p_GRID*(-2*R_v_GRID*i_d_GRID - V_GRID*sin(delta_GRID - theta_GRID))
        struct[0].Gy[24,26] = K_p_GRID*(-2*R_v_GRID*i_q_GRID - V_GRID*cos(delta_GRID - theta_GRID))
        struct[0].Gy[24,29] = K_p_GRID
        struct[0].Gy[25,22] = -sin(delta_GRID - theta_GRID)
        struct[0].Gy[25,23] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[25,25] = -R_v_GRID
        struct[0].Gy[25,26] = X_v_GRID
        struct[0].Gy[26,22] = -cos(delta_GRID - theta_GRID)
        struct[0].Gy[26,23] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[26,25] = -X_v_GRID
        struct[0].Gy[26,26] = -R_v_GRID
        struct[0].Gy[27,22] = i_d_GRID*sin(delta_GRID - theta_GRID) + i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[27,23] = -V_GRID*i_d_GRID*cos(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[27,25] = V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[27,26] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[27,27] = -1
        struct[0].Gy[28,22] = i_d_GRID*cos(delta_GRID - theta_GRID) - i_q_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[28,23] = V_GRID*i_d_GRID*sin(delta_GRID - theta_GRID) + V_GRID*i_q_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[28,25] = V_GRID*cos(delta_GRID - theta_GRID)
        struct[0].Gy[28,26] = -V_GRID*sin(delta_GRID - theta_GRID)
        struct[0].Gy[28,28] = -1
        struct[0].Gy[29,24] = -1/Droop_GRID
        struct[0].Gy[29,29] = -1
        struct[0].Gy[29,31] = K_sec_GRID
        struct[0].Gy[30,24] = 1
        struct[0].Gy[30,30] = -1
        struct[0].Gy[31,30] = -K_p_agc
        struct[0].Gy[31,31] = -1

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
        struct[0].Gu[24,25] = K_p_GRID
        struct[0].Gu[29,25] = -1
        struct[0].Gu[29,26] = 1
        struct[0].Gu[29,27] = 1/Droop_GRID





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
    Fx_ini_rows = [0, 1, 2]

    Fx_ini_cols = [0, 0, 2]

    Fy_ini_rows = [0, 0, 1, 1, 1, 1, 1, 2, 3]

    Fy_ini_cols = [24, 30, 22, 23, 25, 26, 29, 28, 30]

    Gx_ini_rows = [24, 24, 25, 26, 26, 27, 28, 31]

    Gx_ini_cols = [0, 1, 0, 0, 2, 0, 0, 3]

    Gy_ini_rows = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 30, 30, 31, 31]

    Gy_ini_cols = [0, 1, 10, 11, 0, 1, 10, 11, 2, 3, 12, 13, 2, 3, 12, 13, 4, 5, 14, 15, 4, 5, 14, 15, 6, 7, 18, 19, 6, 7, 18, 19, 0, 1, 10, 11, 12, 13, 0, 1, 10, 11, 12, 13, 2, 3, 10, 11, 12, 13, 14, 15, 2, 3, 10, 11, 12, 13, 14, 15, 4, 5, 12, 13, 14, 15, 16, 17, 4, 5, 12, 13, 14, 15, 16, 17, 14, 15, 16, 17, 18, 19, 20, 21, 14, 15, 16, 17, 18, 19, 20, 21, 6, 7, 16, 17, 18, 19, 6, 7, 16, 17, 18, 19, 16, 17, 20, 21, 22, 23, 16, 17, 20, 21, 22, 23, 20, 21, 22, 23, 27, 20, 21, 22, 23, 28, 22, 23, 24, 25, 26, 29, 22, 23, 25, 26, 22, 23, 25, 26, 22, 23, 25, 26, 27, 22, 23, 25, 26, 28, 24, 29, 31, 24, 30, 30, 31]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols