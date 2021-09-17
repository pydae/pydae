import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 


class grid_4bus4wire_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 1
        self.N_y = 60 
        self.N_z = 16 
        self.N_store = 10000 
        self.params_list = [] 
        self.params_values_list  = [] 
        self.inputs_ini_list = ['v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'v_B4_a_r', 'v_B4_a_i', 'v_B4_b_r', 'v_B4_b_i', 'v_B4_c_r', 'v_B4_c_i', 'i_B2_n_r', 'i_B2_n_i', 'i_B3_n_r', 'i_B3_n_i', 'i_B1_n_r', 'i_B1_n_i', 'i_B4_n_r', 'i_B4_n_i', 'p_B2_a', 'q_B2_a', 'p_B2_b', 'q_B2_b', 'p_B2_c', 'q_B2_c', 'p_B3_a', 'q_B3_a', 'p_B3_b', 'q_B3_b', 'p_B3_c', 'q_B3_c', 'u_dummy'] 
        self.inputs_ini_values_list  = [231.0, 0.0, -115.49999999999994, -200.05186827420533, -115.5000000000001, 200.05186827420528, 231.0, 0.0, -115.49999999999994, -200.05186827420533, -115.5000000000001, 200.05186827420528, -0.006469882344603661, 0.07928596960876177, -130.1523812594316, 227.40721408266535, 0.0, 0.0, 0.0, 0.0, -30000.00000020362, 1.3036014934186824e-07, -30000.000000120897, -2.4661130737513304e-07, -29999.999999531643, 1.862445060396567e-07, -10000.000000250586, 1.7661484363884483e-07, -10000.000000070919, -3.0551382224075496e-07, -69999.99999612826, 1.7091988411266357e-06, 1.0] 
        self.inputs_run_list = ['v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'v_B4_a_r', 'v_B4_a_i', 'v_B4_b_r', 'v_B4_b_i', 'v_B4_c_r', 'v_B4_c_i', 'i_B2_n_r', 'i_B2_n_i', 'i_B3_n_r', 'i_B3_n_i', 'i_B1_n_r', 'i_B1_n_i', 'i_B4_n_r', 'i_B4_n_i', 'p_B2_a', 'q_B2_a', 'p_B2_b', 'q_B2_b', 'p_B2_c', 'q_B2_c', 'p_B3_a', 'q_B3_a', 'p_B3_b', 'q_B3_b', 'p_B3_c', 'q_B3_c', 'u_dummy'] 
        self.inputs_run_values_list = [231.0, 0.0, -115.49999999999994, -200.05186827420533, -115.5000000000001, 200.05186827420528, 231.0, 0.0, -115.49999999999994, -200.05186827420533, -115.5000000000001, 200.05186827420528, -0.006469882344603661, 0.07928596960876177, -130.1523812594316, 227.40721408266535, 0.0, 0.0, 0.0, 0.0, -30000.00000020362, 1.3036014934186824e-07, -30000.000000120897, -2.4661130737513304e-07, -29999.999999531643, 1.862445060396567e-07, -10000.000000250586, 1.7661484363884483e-07, -10000.000000070919, -3.0551382224075496e-07, -69999.99999612826, 1.7091988411266357e-06, 1.0] 
        self.outputs_list = ['v_B1_a_m', 'v_B1_b_m', 'v_B1_c_m', 'v_B4_a_m', 'v_B4_b_m', 'v_B4_c_m', 'v_B2_a_m', 'v_B2_b_m', 'v_B2_c_m', 'v_B2_n_m', 'v_B3_a_m', 'v_B3_b_m', 'v_B3_c_m', 'v_B3_n_m', 'v_B1_n_m', 'v_B4_n_m'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B2_n_r', 'v_B2_n_i', 'v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B3_n_r', 'v_B3_n_i', 'v_B1_n_r', 'v_B1_n_i', 'v_B4_n_r', 'v_B4_n_i', 'i_l_B1_B2_a_r', 'i_l_B1_B2_a_i', 'i_l_B1_B2_b_r', 'i_l_B1_B2_b_i', 'i_l_B1_B2_c_r', 'i_l_B1_B2_c_i', 'i_l_B1_B2_n_r', 'i_l_B1_B2_n_i', 'i_l_B2_B3_a_r', 'i_l_B2_B3_a_i', 'i_l_B2_B3_b_r', 'i_l_B2_B3_b_i', 'i_l_B2_B3_c_r', 'i_l_B2_B3_c_i', 'i_l_B2_B3_n_r', 'i_l_B2_B3_n_i', 'i_l_B3_B4_a_r', 'i_l_B3_B4_a_i', 'i_l_B3_B4_b_r', 'i_l_B3_B4_b_i', 'i_l_B3_B4_c_r', 'i_l_B3_B4_c_i', 'i_l_B3_B4_n_r', 'i_l_B3_B4_n_i', 'i_load_B2_a_r', 'i_load_B2_a_i', 'i_load_B2_b_r', 'i_load_B2_b_i', 'i_load_B2_c_r', 'i_load_B2_c_i', 'i_load_B2_n_r', 'i_load_B2_n_i', 'i_load_B3_a_r', 'i_load_B3_a_i', 'i_load_B3_b_r', 'i_load_B3_b_i', 'i_load_B3_c_r', 'i_load_B3_c_i', 'i_load_B3_n_r', 'i_load_B3_n_i'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B2_n_r', 'v_B2_n_i', 'v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B3_n_r', 'v_B3_n_i', 'v_B1_n_r', 'v_B1_n_i', 'v_B4_n_r', 'v_B4_n_i', 'i_l_B1_B2_a_r', 'i_l_B1_B2_a_i', 'i_l_B1_B2_b_r', 'i_l_B1_B2_b_i', 'i_l_B1_B2_c_r', 'i_l_B1_B2_c_i', 'i_l_B1_B2_n_r', 'i_l_B1_B2_n_i', 'i_l_B2_B3_a_r', 'i_l_B2_B3_a_i', 'i_l_B2_B3_b_r', 'i_l_B2_B3_b_i', 'i_l_B2_B3_c_r', 'i_l_B2_B3_c_i', 'i_l_B2_B3_n_r', 'i_l_B2_B3_n_i', 'i_l_B3_B4_a_r', 'i_l_B3_B4_a_i', 'i_l_B3_B4_b_r', 'i_l_B3_B4_b_i', 'i_l_B3_B4_c_r', 'i_l_B3_B4_c_i', 'i_l_B3_B4_n_r', 'i_l_B3_B4_n_i', 'i_load_B2_a_r', 'i_load_B2_a_i', 'i_load_B2_b_r', 'i_load_B2_b_i', 'i_load_B2_c_r', 'i_load_B2_c_i', 'i_load_B2_n_r', 'i_load_B2_n_i', 'i_load_B3_a_r', 'i_load_B3_a_i', 'i_load_B3_b_r', 'i_load_B3_b_i', 'i_load_B3_c_r', 'i_load_B3_c_i', 'i_load_B3_n_r', 'i_load_B3_n_i'] 
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
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B4_a_r = struct[0].v_B4_a_r
    v_B4_a_i = struct[0].v_B4_a_i
    v_B4_b_r = struct[0].v_B4_b_r
    v_B4_b_i = struct[0].v_B4_b_i
    v_B4_c_r = struct[0].v_B4_c_r
    v_B4_c_i = struct[0].v_B4_c_i
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
    i_B1_n_r = struct[0].i_B1_n_r
    i_B1_n_i = struct[0].i_B1_n_i
    i_B4_n_r = struct[0].i_B4_n_r
    i_B4_n_i = struct[0].i_B4_n_i
    p_B2_a = struct[0].p_B2_a
    q_B2_a = struct[0].q_B2_a
    p_B2_b = struct[0].p_B2_b
    q_B2_b = struct[0].q_B2_b
    p_B2_c = struct[0].p_B2_c
    q_B2_c = struct[0].q_B2_c
    p_B3_a = struct[0].p_B3_a
    q_B3_a = struct[0].q_B3_a
    p_B3_b = struct[0].p_B3_b
    q_B3_b = struct[0].q_B3_b
    p_B3_c = struct[0].p_B3_c
    q_B3_c = struct[0].q_B3_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2_a_r = struct[0].y_ini[0,0]
    v_B2_a_i = struct[0].y_ini[1,0]
    v_B2_b_r = struct[0].y_ini[2,0]
    v_B2_b_i = struct[0].y_ini[3,0]
    v_B2_c_r = struct[0].y_ini[4,0]
    v_B2_c_i = struct[0].y_ini[5,0]
    v_B2_n_r = struct[0].y_ini[6,0]
    v_B2_n_i = struct[0].y_ini[7,0]
    v_B3_a_r = struct[0].y_ini[8,0]
    v_B3_a_i = struct[0].y_ini[9,0]
    v_B3_b_r = struct[0].y_ini[10,0]
    v_B3_b_i = struct[0].y_ini[11,0]
    v_B3_c_r = struct[0].y_ini[12,0]
    v_B3_c_i = struct[0].y_ini[13,0]
    v_B3_n_r = struct[0].y_ini[14,0]
    v_B3_n_i = struct[0].y_ini[15,0]
    v_B1_n_r = struct[0].y_ini[16,0]
    v_B1_n_i = struct[0].y_ini[17,0]
    v_B4_n_r = struct[0].y_ini[18,0]
    v_B4_n_i = struct[0].y_ini[19,0]
    i_l_B1_B2_a_r = struct[0].y_ini[20,0]
    i_l_B1_B2_a_i = struct[0].y_ini[21,0]
    i_l_B1_B2_b_r = struct[0].y_ini[22,0]
    i_l_B1_B2_b_i = struct[0].y_ini[23,0]
    i_l_B1_B2_c_r = struct[0].y_ini[24,0]
    i_l_B1_B2_c_i = struct[0].y_ini[25,0]
    i_l_B1_B2_n_r = struct[0].y_ini[26,0]
    i_l_B1_B2_n_i = struct[0].y_ini[27,0]
    i_l_B2_B3_a_r = struct[0].y_ini[28,0]
    i_l_B2_B3_a_i = struct[0].y_ini[29,0]
    i_l_B2_B3_b_r = struct[0].y_ini[30,0]
    i_l_B2_B3_b_i = struct[0].y_ini[31,0]
    i_l_B2_B3_c_r = struct[0].y_ini[32,0]
    i_l_B2_B3_c_i = struct[0].y_ini[33,0]
    i_l_B2_B3_n_r = struct[0].y_ini[34,0]
    i_l_B2_B3_n_i = struct[0].y_ini[35,0]
    i_l_B3_B4_a_r = struct[0].y_ini[36,0]
    i_l_B3_B4_a_i = struct[0].y_ini[37,0]
    i_l_B3_B4_b_r = struct[0].y_ini[38,0]
    i_l_B3_B4_b_i = struct[0].y_ini[39,0]
    i_l_B3_B4_c_r = struct[0].y_ini[40,0]
    i_l_B3_B4_c_i = struct[0].y_ini[41,0]
    i_l_B3_B4_n_r = struct[0].y_ini[42,0]
    i_l_B3_B4_n_i = struct[0].y_ini[43,0]
    i_load_B2_a_r = struct[0].y_ini[44,0]
    i_load_B2_a_i = struct[0].y_ini[45,0]
    i_load_B2_b_r = struct[0].y_ini[46,0]
    i_load_B2_b_i = struct[0].y_ini[47,0]
    i_load_B2_c_r = struct[0].y_ini[48,0]
    i_load_B2_c_i = struct[0].y_ini[49,0]
    i_load_B2_n_r = struct[0].y_ini[50,0]
    i_load_B2_n_i = struct[0].y_ini[51,0]
    i_load_B3_a_r = struct[0].y_ini[52,0]
    i_load_B3_a_i = struct[0].y_ini[53,0]
    i_load_B3_b_r = struct[0].y_ini[54,0]
    i_load_B3_b_i = struct[0].y_ini[55,0]
    i_load_B3_c_r = struct[0].y_ini[56,0]
    i_load_B3_c_i = struct[0].y_ini[57,0]
    i_load_B3_n_r = struct[0].y_ini[58,0]
    i_load_B3_n_i = struct[0].y_ini[59,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = i_load_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 139.986584618974*v_B2_a_i - 292.221995392108*v_B2_a_r + 23.3310974364957*v_B3_a_i + 48.7036658986847*v_B3_a_r
        struct[0].g[1,0] = i_load_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 292.221995392108*v_B2_a_i + 139.986584618974*v_B2_a_r + 48.7036658986847*v_B3_a_i - 23.3310974364957*v_B3_a_r
        struct[0].g[2,0] = i_load_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 139.986584618974*v_B2_b_i - 292.221995392108*v_B2_b_r + 23.3310974364957*v_B3_b_i + 48.7036658986847*v_B3_b_r
        struct[0].g[3,0] = i_load_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 292.221995392108*v_B2_b_i + 139.986584618974*v_B2_b_r + 48.7036658986847*v_B3_b_i - 23.3310974364957*v_B3_b_r
        struct[0].g[4,0] = i_load_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 139.986584618974*v_B2_c_i - 292.221995392108*v_B2_c_r + 23.3310974364957*v_B3_c_i + 48.7036658986847*v_B3_c_r
        struct[0].g[5,0] = i_load_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 292.221995392108*v_B2_c_i + 139.986584618974*v_B2_c_r + 48.7036658986847*v_B3_c_i - 23.3310974364957*v_B3_c_r
        struct[0].g[8,0] = i_load_B3_a_r + 23.3310974364957*v_B2_a_i + 48.7036658986847*v_B2_a_r - 139.986584618974*v_B3_a_i - 292.221995392108*v_B3_a_r + 116.655487182478*v_B4_a_i + 243.518329493424*v_B4_a_r
        struct[0].g[9,0] = i_load_B3_a_i + 48.7036658986847*v_B2_a_i - 23.3310974364957*v_B2_a_r - 292.221995392108*v_B3_a_i + 139.986584618974*v_B3_a_r + 243.518329493424*v_B4_a_i - 116.655487182478*v_B4_a_r
        struct[0].g[10,0] = i_load_B3_b_r + 23.3310974364957*v_B2_b_i + 48.7036658986847*v_B2_b_r - 139.986584618974*v_B3_b_i - 292.221995392108*v_B3_b_r + 116.655487182478*v_B4_b_i + 243.518329493424*v_B4_b_r
        struct[0].g[11,0] = i_load_B3_b_i + 48.7036658986847*v_B2_b_i - 23.3310974364957*v_B2_b_r - 292.221995392108*v_B3_b_i + 139.986584618974*v_B3_b_r + 243.518329493424*v_B4_b_i - 116.655487182478*v_B4_b_r
        struct[0].g[12,0] = i_load_B3_c_r + 23.3310974364957*v_B2_c_i + 48.7036658986847*v_B2_c_r - 139.986584618974*v_B3_c_i - 292.221995392108*v_B3_c_r + 116.655487182478*v_B4_c_i + 243.518329493424*v_B4_c_r
        struct[0].g[13,0] = i_load_B3_c_i + 48.7036658986847*v_B2_c_i - 23.3310974364957*v_B2_c_r - 292.221995392108*v_B3_c_i + 139.986584618974*v_B3_c_r + 243.518329493424*v_B4_c_i - 116.655487182478*v_B4_c_r
        struct[0].g[20,0] = -i_l_B1_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 116.655487182478*v_B2_a_i - 243.518329493424*v_B2_a_r
        struct[0].g[21,0] = -i_l_B1_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 243.518329493424*v_B2_a_i + 116.655487182478*v_B2_a_r
        struct[0].g[22,0] = -i_l_B1_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 116.655487182478*v_B2_b_i - 243.518329493424*v_B2_b_r
        struct[0].g[23,0] = -i_l_B1_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 243.518329493424*v_B2_b_i + 116.655487182478*v_B2_b_r
        struct[0].g[24,0] = -i_l_B1_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 116.655487182478*v_B2_c_i - 243.518329493424*v_B2_c_r
        struct[0].g[25,0] = -i_l_B1_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 243.518329493424*v_B2_c_i + 116.655487182478*v_B2_c_r
        struct[0].g[36,0] = -i_l_B3_B4_a_r + 116.655487182478*v_B3_a_i + 243.518329493424*v_B3_a_r - 116.655487182478*v_B4_a_i - 243.518329493424*v_B4_a_r
        struct[0].g[37,0] = -i_l_B3_B4_a_i + 243.518329493424*v_B3_a_i - 116.655487182478*v_B3_a_r - 243.518329493424*v_B4_a_i + 116.655487182478*v_B4_a_r
        struct[0].g[38,0] = -i_l_B3_B4_b_r + 116.655487182478*v_B3_b_i + 243.518329493424*v_B3_b_r - 116.655487182478*v_B4_b_i - 243.518329493424*v_B4_b_r
        struct[0].g[39,0] = -i_l_B3_B4_b_i + 243.518329493424*v_B3_b_i - 116.655487182478*v_B3_b_r - 243.518329493424*v_B4_b_i + 116.655487182478*v_B4_b_r
        struct[0].g[40,0] = -i_l_B3_B4_c_r + 116.655487182478*v_B3_c_i + 243.518329493424*v_B3_c_r - 116.655487182478*v_B4_c_i - 243.518329493424*v_B4_c_r
        struct[0].g[41,0] = -i_l_B3_B4_c_i + 243.518329493424*v_B3_c_i - 116.655487182478*v_B3_c_r - 243.518329493424*v_B4_c_i + 116.655487182478*v_B4_c_r
        struct[0].g[44,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[45,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[46,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[47,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[48,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[49,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[52,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[53,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[54,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[55,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[56,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[57,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[4,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[5,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[14,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:



        struct[0].Gy_ini[44,0] = i_load_B2_a_r
        struct[0].Gy_ini[44,1] = i_load_B2_a_i
        struct[0].Gy_ini[44,6] = -i_load_B2_a_r
        struct[0].Gy_ini[44,7] = -i_load_B2_a_i
        struct[0].Gy_ini[44,44] = v_B2_a_r - v_B2_n_r
        struct[0].Gy_ini[44,45] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[45,2] = i_load_B2_b_r
        struct[0].Gy_ini[45,3] = i_load_B2_b_i
        struct[0].Gy_ini[45,6] = -i_load_B2_b_r
        struct[0].Gy_ini[45,7] = -i_load_B2_b_i
        struct[0].Gy_ini[45,46] = v_B2_b_r - v_B2_n_r
        struct[0].Gy_ini[45,47] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[46,4] = i_load_B2_c_r
        struct[0].Gy_ini[46,5] = i_load_B2_c_i
        struct[0].Gy_ini[46,6] = -i_load_B2_c_r
        struct[0].Gy_ini[46,7] = -i_load_B2_c_i
        struct[0].Gy_ini[46,48] = v_B2_c_r - v_B2_n_r
        struct[0].Gy_ini[46,49] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[47,0] = -i_load_B2_a_i
        struct[0].Gy_ini[47,1] = i_load_B2_a_r
        struct[0].Gy_ini[47,6] = i_load_B2_a_i
        struct[0].Gy_ini[47,7] = -i_load_B2_a_r
        struct[0].Gy_ini[47,44] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[47,45] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy_ini[48,2] = -i_load_B2_b_i
        struct[0].Gy_ini[48,3] = i_load_B2_b_r
        struct[0].Gy_ini[48,6] = i_load_B2_b_i
        struct[0].Gy_ini[48,7] = -i_load_B2_b_r
        struct[0].Gy_ini[48,46] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[48,47] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy_ini[49,4] = -i_load_B2_c_i
        struct[0].Gy_ini[49,5] = i_load_B2_c_r
        struct[0].Gy_ini[49,6] = i_load_B2_c_i
        struct[0].Gy_ini[49,7] = -i_load_B2_c_r
        struct[0].Gy_ini[49,48] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[49,49] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy_ini[52,8] = i_load_B3_a_r
        struct[0].Gy_ini[52,9] = i_load_B3_a_i
        struct[0].Gy_ini[52,14] = -i_load_B3_a_r
        struct[0].Gy_ini[52,15] = -i_load_B3_a_i
        struct[0].Gy_ini[52,52] = v_B3_a_r - v_B3_n_r
        struct[0].Gy_ini[52,53] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[53,10] = i_load_B3_b_r
        struct[0].Gy_ini[53,11] = i_load_B3_b_i
        struct[0].Gy_ini[53,14] = -i_load_B3_b_r
        struct[0].Gy_ini[53,15] = -i_load_B3_b_i
        struct[0].Gy_ini[53,54] = v_B3_b_r - v_B3_n_r
        struct[0].Gy_ini[53,55] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[54,12] = i_load_B3_c_r
        struct[0].Gy_ini[54,13] = i_load_B3_c_i
        struct[0].Gy_ini[54,14] = -i_load_B3_c_r
        struct[0].Gy_ini[54,15] = -i_load_B3_c_i
        struct[0].Gy_ini[54,56] = v_B3_c_r - v_B3_n_r
        struct[0].Gy_ini[54,57] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[55,8] = -i_load_B3_a_i
        struct[0].Gy_ini[55,9] = i_load_B3_a_r
        struct[0].Gy_ini[55,14] = i_load_B3_a_i
        struct[0].Gy_ini[55,15] = -i_load_B3_a_r
        struct[0].Gy_ini[55,52] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[55,53] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy_ini[56,10] = -i_load_B3_b_i
        struct[0].Gy_ini[56,11] = i_load_B3_b_r
        struct[0].Gy_ini[56,14] = i_load_B3_b_i
        struct[0].Gy_ini[56,15] = -i_load_B3_b_r
        struct[0].Gy_ini[56,54] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[56,55] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy_ini[57,12] = -i_load_B3_c_i
        struct[0].Gy_ini[57,13] = i_load_B3_c_r
        struct[0].Gy_ini[57,14] = i_load_B3_c_i
        struct[0].Gy_ini[57,15] = -i_load_B3_c_r
        struct[0].Gy_ini[57,56] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[57,57] = -v_B3_c_r + v_B3_n_r



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B4_a_r = struct[0].v_B4_a_r
    v_B4_a_i = struct[0].v_B4_a_i
    v_B4_b_r = struct[0].v_B4_b_r
    v_B4_b_i = struct[0].v_B4_b_i
    v_B4_c_r = struct[0].v_B4_c_r
    v_B4_c_i = struct[0].v_B4_c_i
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
    i_B1_n_r = struct[0].i_B1_n_r
    i_B1_n_i = struct[0].i_B1_n_i
    i_B4_n_r = struct[0].i_B4_n_r
    i_B4_n_i = struct[0].i_B4_n_i
    p_B2_a = struct[0].p_B2_a
    q_B2_a = struct[0].q_B2_a
    p_B2_b = struct[0].p_B2_b
    q_B2_b = struct[0].q_B2_b
    p_B2_c = struct[0].p_B2_c
    q_B2_c = struct[0].q_B2_c
    p_B3_a = struct[0].p_B3_a
    q_B3_a = struct[0].q_B3_a
    p_B3_b = struct[0].p_B3_b
    q_B3_b = struct[0].q_B3_b
    p_B3_c = struct[0].p_B3_c
    q_B3_c = struct[0].q_B3_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2_a_r = struct[0].y_run[0,0]
    v_B2_a_i = struct[0].y_run[1,0]
    v_B2_b_r = struct[0].y_run[2,0]
    v_B2_b_i = struct[0].y_run[3,0]
    v_B2_c_r = struct[0].y_run[4,0]
    v_B2_c_i = struct[0].y_run[5,0]
    v_B2_n_r = struct[0].y_run[6,0]
    v_B2_n_i = struct[0].y_run[7,0]
    v_B3_a_r = struct[0].y_run[8,0]
    v_B3_a_i = struct[0].y_run[9,0]
    v_B3_b_r = struct[0].y_run[10,0]
    v_B3_b_i = struct[0].y_run[11,0]
    v_B3_c_r = struct[0].y_run[12,0]
    v_B3_c_i = struct[0].y_run[13,0]
    v_B3_n_r = struct[0].y_run[14,0]
    v_B3_n_i = struct[0].y_run[15,0]
    v_B1_n_r = struct[0].y_run[16,0]
    v_B1_n_i = struct[0].y_run[17,0]
    v_B4_n_r = struct[0].y_run[18,0]
    v_B4_n_i = struct[0].y_run[19,0]
    i_l_B1_B2_a_r = struct[0].y_run[20,0]
    i_l_B1_B2_a_i = struct[0].y_run[21,0]
    i_l_B1_B2_b_r = struct[0].y_run[22,0]
    i_l_B1_B2_b_i = struct[0].y_run[23,0]
    i_l_B1_B2_c_r = struct[0].y_run[24,0]
    i_l_B1_B2_c_i = struct[0].y_run[25,0]
    i_l_B1_B2_n_r = struct[0].y_run[26,0]
    i_l_B1_B2_n_i = struct[0].y_run[27,0]
    i_l_B2_B3_a_r = struct[0].y_run[28,0]
    i_l_B2_B3_a_i = struct[0].y_run[29,0]
    i_l_B2_B3_b_r = struct[0].y_run[30,0]
    i_l_B2_B3_b_i = struct[0].y_run[31,0]
    i_l_B2_B3_c_r = struct[0].y_run[32,0]
    i_l_B2_B3_c_i = struct[0].y_run[33,0]
    i_l_B2_B3_n_r = struct[0].y_run[34,0]
    i_l_B2_B3_n_i = struct[0].y_run[35,0]
    i_l_B3_B4_a_r = struct[0].y_run[36,0]
    i_l_B3_B4_a_i = struct[0].y_run[37,0]
    i_l_B3_B4_b_r = struct[0].y_run[38,0]
    i_l_B3_B4_b_i = struct[0].y_run[39,0]
    i_l_B3_B4_c_r = struct[0].y_run[40,0]
    i_l_B3_B4_c_i = struct[0].y_run[41,0]
    i_l_B3_B4_n_r = struct[0].y_run[42,0]
    i_l_B3_B4_n_i = struct[0].y_run[43,0]
    i_load_B2_a_r = struct[0].y_run[44,0]
    i_load_B2_a_i = struct[0].y_run[45,0]
    i_load_B2_b_r = struct[0].y_run[46,0]
    i_load_B2_b_i = struct[0].y_run[47,0]
    i_load_B2_c_r = struct[0].y_run[48,0]
    i_load_B2_c_i = struct[0].y_run[49,0]
    i_load_B2_n_r = struct[0].y_run[50,0]
    i_load_B2_n_i = struct[0].y_run[51,0]
    i_load_B3_a_r = struct[0].y_run[52,0]
    i_load_B3_a_i = struct[0].y_run[53,0]
    i_load_B3_b_r = struct[0].y_run[54,0]
    i_load_B3_b_i = struct[0].y_run[55,0]
    i_load_B3_c_r = struct[0].y_run[56,0]
    i_load_B3_c_i = struct[0].y_run[57,0]
    i_load_B3_n_r = struct[0].y_run[58,0]
    i_load_B3_n_i = struct[0].y_run[59,0]
    
    struct[0].u_run[0,0] = v_B1_a_r
    struct[0].u_run[1,0] = v_B1_a_i
    struct[0].u_run[2,0] = v_B1_b_r
    struct[0].u_run[3,0] = v_B1_b_i
    struct[0].u_run[4,0] = v_B1_c_r
    struct[0].u_run[5,0] = v_B1_c_i
    struct[0].u_run[6,0] = v_B4_a_r
    struct[0].u_run[7,0] = v_B4_a_i
    struct[0].u_run[8,0] = v_B4_b_r
    struct[0].u_run[9,0] = v_B4_b_i
    struct[0].u_run[10,0] = v_B4_c_r
    struct[0].u_run[11,0] = v_B4_c_i
    struct[0].u_run[12,0] = i_B2_n_r
    struct[0].u_run[13,0] = i_B2_n_i
    struct[0].u_run[14,0] = i_B3_n_r
    struct[0].u_run[15,0] = i_B3_n_i
    struct[0].u_run[16,0] = i_B1_n_r
    struct[0].u_run[17,0] = i_B1_n_i
    struct[0].u_run[18,0] = i_B4_n_r
    struct[0].u_run[19,0] = i_B4_n_i
    struct[0].u_run[20,0] = p_B2_a
    struct[0].u_run[21,0] = q_B2_a
    struct[0].u_run[22,0] = p_B2_b
    struct[0].u_run[23,0] = q_B2_b
    struct[0].u_run[24,0] = p_B2_c
    struct[0].u_run[25,0] = q_B2_c
    struct[0].u_run[26,0] = p_B3_a
    struct[0].u_run[27,0] = q_B3_a
    struct[0].u_run[28,0] = p_B3_b
    struct[0].u_run[29,0] = q_B3_b
    struct[0].u_run[30,0] = p_B3_c
    struct[0].u_run[31,0] = q_B3_c
    struct[0].u_run[32,0] = u_dummy
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[44,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[45,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[46,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[47,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[48,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[49,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[52,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[53,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[54,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[55,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[56,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[57,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[4,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[5,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[14,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:



        struct[0].Gy[44,0] = i_load_B2_a_r
        struct[0].Gy[44,1] = i_load_B2_a_i
        struct[0].Gy[44,6] = -i_load_B2_a_r
        struct[0].Gy[44,7] = -i_load_B2_a_i
        struct[0].Gy[44,44] = v_B2_a_r - v_B2_n_r
        struct[0].Gy[44,45] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[45,2] = i_load_B2_b_r
        struct[0].Gy[45,3] = i_load_B2_b_i
        struct[0].Gy[45,6] = -i_load_B2_b_r
        struct[0].Gy[45,7] = -i_load_B2_b_i
        struct[0].Gy[45,46] = v_B2_b_r - v_B2_n_r
        struct[0].Gy[45,47] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[46,4] = i_load_B2_c_r
        struct[0].Gy[46,5] = i_load_B2_c_i
        struct[0].Gy[46,6] = -i_load_B2_c_r
        struct[0].Gy[46,7] = -i_load_B2_c_i
        struct[0].Gy[46,48] = v_B2_c_r - v_B2_n_r
        struct[0].Gy[46,49] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[47,0] = -i_load_B2_a_i
        struct[0].Gy[47,1] = i_load_B2_a_r
        struct[0].Gy[47,6] = i_load_B2_a_i
        struct[0].Gy[47,7] = -i_load_B2_a_r
        struct[0].Gy[47,44] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[47,45] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy[48,2] = -i_load_B2_b_i
        struct[0].Gy[48,3] = i_load_B2_b_r
        struct[0].Gy[48,6] = i_load_B2_b_i
        struct[0].Gy[48,7] = -i_load_B2_b_r
        struct[0].Gy[48,46] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[48,47] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy[49,4] = -i_load_B2_c_i
        struct[0].Gy[49,5] = i_load_B2_c_r
        struct[0].Gy[49,6] = i_load_B2_c_i
        struct[0].Gy[49,7] = -i_load_B2_c_r
        struct[0].Gy[49,48] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[49,49] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy[52,8] = i_load_B3_a_r
        struct[0].Gy[52,9] = i_load_B3_a_i
        struct[0].Gy[52,14] = -i_load_B3_a_r
        struct[0].Gy[52,15] = -i_load_B3_a_i
        struct[0].Gy[52,52] = v_B3_a_r - v_B3_n_r
        struct[0].Gy[52,53] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[53,10] = i_load_B3_b_r
        struct[0].Gy[53,11] = i_load_B3_b_i
        struct[0].Gy[53,14] = -i_load_B3_b_r
        struct[0].Gy[53,15] = -i_load_B3_b_i
        struct[0].Gy[53,54] = v_B3_b_r - v_B3_n_r
        struct[0].Gy[53,55] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[54,12] = i_load_B3_c_r
        struct[0].Gy[54,13] = i_load_B3_c_i
        struct[0].Gy[54,14] = -i_load_B3_c_r
        struct[0].Gy[54,15] = -i_load_B3_c_i
        struct[0].Gy[54,56] = v_B3_c_r - v_B3_n_r
        struct[0].Gy[54,57] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[55,8] = -i_load_B3_a_i
        struct[0].Gy[55,9] = i_load_B3_a_r
        struct[0].Gy[55,14] = i_load_B3_a_i
        struct[0].Gy[55,15] = -i_load_B3_a_r
        struct[0].Gy[55,52] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[55,53] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy[56,10] = -i_load_B3_b_i
        struct[0].Gy[56,11] = i_load_B3_b_r
        struct[0].Gy[56,14] = i_load_B3_b_i
        struct[0].Gy[56,15] = -i_load_B3_b_r
        struct[0].Gy[56,54] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[56,55] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy[57,12] = -i_load_B3_c_i
        struct[0].Gy[57,13] = i_load_B3_c_r
        struct[0].Gy[57,14] = i_load_B3_c_i
        struct[0].Gy[57,15] = -i_load_B3_c_r
        struct[0].Gy[57,56] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[57,57] = -v_B3_c_r + v_B3_n_r

    if mode > 12:




        struct[0].Hy[6,0] = 1.0*v_B2_a_r*(v_B2_a_i**2 + v_B2_a_r**2)**(-0.5)
        struct[0].Hy[6,1] = 1.0*v_B2_a_i*(v_B2_a_i**2 + v_B2_a_r**2)**(-0.5)
        struct[0].Hy[7,2] = 1.0*v_B2_b_r*(v_B2_b_i**2 + v_B2_b_r**2)**(-0.5)
        struct[0].Hy[7,3] = 1.0*v_B2_b_i*(v_B2_b_i**2 + v_B2_b_r**2)**(-0.5)
        struct[0].Hy[8,4] = 1.0*v_B2_c_r*(v_B2_c_i**2 + v_B2_c_r**2)**(-0.5)
        struct[0].Hy[8,5] = 1.0*v_B2_c_i*(v_B2_c_i**2 + v_B2_c_r**2)**(-0.5)
        struct[0].Hy[9,6] = 1.0*v_B2_n_r*(v_B2_n_i**2 + v_B2_n_r**2)**(-0.5)
        struct[0].Hy[9,7] = 1.0*v_B2_n_i*(v_B2_n_i**2 + v_B2_n_r**2)**(-0.5)
        struct[0].Hy[10,8] = 1.0*v_B3_a_r*(v_B3_a_i**2 + v_B3_a_r**2)**(-0.5)
        struct[0].Hy[10,9] = 1.0*v_B3_a_i*(v_B3_a_i**2 + v_B3_a_r**2)**(-0.5)
        struct[0].Hy[11,10] = 1.0*v_B3_b_r*(v_B3_b_i**2 + v_B3_b_r**2)**(-0.5)
        struct[0].Hy[11,11] = 1.0*v_B3_b_i*(v_B3_b_i**2 + v_B3_b_r**2)**(-0.5)
        struct[0].Hy[12,12] = 1.0*v_B3_c_r*(v_B3_c_i**2 + v_B3_c_r**2)**(-0.5)
        struct[0].Hy[12,13] = 1.0*v_B3_c_i*(v_B3_c_i**2 + v_B3_c_r**2)**(-0.5)
        struct[0].Hy[13,14] = 1.0*v_B3_n_r*(v_B3_n_i**2 + v_B3_n_r**2)**(-0.5)
        struct[0].Hy[13,15] = 1.0*v_B3_n_i*(v_B3_n_i**2 + v_B3_n_r**2)**(-0.5)
        struct[0].Hy[14,16] = 1.0*v_B1_n_r*(v_B1_n_i**2 + v_B1_n_r**2)**(-0.5)
        struct[0].Hy[14,17] = 1.0*v_B1_n_i*(v_B1_n_i**2 + v_B1_n_r**2)**(-0.5)
        struct[0].Hy[15,18] = 1.0*v_B4_n_r*(v_B4_n_i**2 + v_B4_n_r**2)**(-0.5)
        struct[0].Hy[15,19] = 1.0*v_B4_n_i*(v_B4_n_i**2 + v_B4_n_r**2)**(-0.5)

        struct[0].Hu[0,0] = 1.0*v_B1_a_r*(v_B1_a_i**2 + v_B1_a_r**2)**(-0.5)
        struct[0].Hu[0,1] = 1.0*v_B1_a_i*(v_B1_a_i**2 + v_B1_a_r**2)**(-0.5)
        struct[0].Hu[1,2] = 1.0*v_B1_b_r*(v_B1_b_i**2 + v_B1_b_r**2)**(-0.5)
        struct[0].Hu[1,3] = 1.0*v_B1_b_i*(v_B1_b_i**2 + v_B1_b_r**2)**(-0.5)
        struct[0].Hu[2,4] = 1.0*v_B1_c_r*(v_B1_c_i**2 + v_B1_c_r**2)**(-0.5)
        struct[0].Hu[2,5] = 1.0*v_B1_c_i*(v_B1_c_i**2 + v_B1_c_r**2)**(-0.5)
        struct[0].Hu[3,6] = 1.0*v_B4_a_r*(v_B4_a_i**2 + v_B4_a_r**2)**(-0.5)
        struct[0].Hu[3,7] = 1.0*v_B4_a_i*(v_B4_a_i**2 + v_B4_a_r**2)**(-0.5)
        struct[0].Hu[4,8] = 1.0*v_B4_b_r*(v_B4_b_i**2 + v_B4_b_r**2)**(-0.5)
        struct[0].Hu[4,9] = 1.0*v_B4_b_i*(v_B4_b_i**2 + v_B4_b_r**2)**(-0.5)
        struct[0].Hu[5,10] = 1.0*v_B4_c_r*(v_B4_c_i**2 + v_B4_c_r**2)**(-0.5)
        struct[0].Hu[5,11] = 1.0*v_B4_c_i*(v_B4_c_i**2 + v_B4_c_r**2)**(-0.5)



def ini_nn(struct,mode):

    # Parameters:
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B4_a_r = struct[0].v_B4_a_r
    v_B4_a_i = struct[0].v_B4_a_i
    v_B4_b_r = struct[0].v_B4_b_r
    v_B4_b_i = struct[0].v_B4_b_i
    v_B4_c_r = struct[0].v_B4_c_r
    v_B4_c_i = struct[0].v_B4_c_i
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
    i_B1_n_r = struct[0].i_B1_n_r
    i_B1_n_i = struct[0].i_B1_n_i
    i_B4_n_r = struct[0].i_B4_n_r
    i_B4_n_i = struct[0].i_B4_n_i
    p_B2_a = struct[0].p_B2_a
    q_B2_a = struct[0].q_B2_a
    p_B2_b = struct[0].p_B2_b
    q_B2_b = struct[0].q_B2_b
    p_B2_c = struct[0].p_B2_c
    q_B2_c = struct[0].q_B2_c
    p_B3_a = struct[0].p_B3_a
    q_B3_a = struct[0].q_B3_a
    p_B3_b = struct[0].p_B3_b
    q_B3_b = struct[0].q_B3_b
    p_B3_c = struct[0].p_B3_c
    q_B3_c = struct[0].q_B3_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2_a_r = struct[0].y_ini[0,0]
    v_B2_a_i = struct[0].y_ini[1,0]
    v_B2_b_r = struct[0].y_ini[2,0]
    v_B2_b_i = struct[0].y_ini[3,0]
    v_B2_c_r = struct[0].y_ini[4,0]
    v_B2_c_i = struct[0].y_ini[5,0]
    v_B2_n_r = struct[0].y_ini[6,0]
    v_B2_n_i = struct[0].y_ini[7,0]
    v_B3_a_r = struct[0].y_ini[8,0]
    v_B3_a_i = struct[0].y_ini[9,0]
    v_B3_b_r = struct[0].y_ini[10,0]
    v_B3_b_i = struct[0].y_ini[11,0]
    v_B3_c_r = struct[0].y_ini[12,0]
    v_B3_c_i = struct[0].y_ini[13,0]
    v_B3_n_r = struct[0].y_ini[14,0]
    v_B3_n_i = struct[0].y_ini[15,0]
    v_B1_n_r = struct[0].y_ini[16,0]
    v_B1_n_i = struct[0].y_ini[17,0]
    v_B4_n_r = struct[0].y_ini[18,0]
    v_B4_n_i = struct[0].y_ini[19,0]
    i_l_B1_B2_a_r = struct[0].y_ini[20,0]
    i_l_B1_B2_a_i = struct[0].y_ini[21,0]
    i_l_B1_B2_b_r = struct[0].y_ini[22,0]
    i_l_B1_B2_b_i = struct[0].y_ini[23,0]
    i_l_B1_B2_c_r = struct[0].y_ini[24,0]
    i_l_B1_B2_c_i = struct[0].y_ini[25,0]
    i_l_B1_B2_n_r = struct[0].y_ini[26,0]
    i_l_B1_B2_n_i = struct[0].y_ini[27,0]
    i_l_B2_B3_a_r = struct[0].y_ini[28,0]
    i_l_B2_B3_a_i = struct[0].y_ini[29,0]
    i_l_B2_B3_b_r = struct[0].y_ini[30,0]
    i_l_B2_B3_b_i = struct[0].y_ini[31,0]
    i_l_B2_B3_c_r = struct[0].y_ini[32,0]
    i_l_B2_B3_c_i = struct[0].y_ini[33,0]
    i_l_B2_B3_n_r = struct[0].y_ini[34,0]
    i_l_B2_B3_n_i = struct[0].y_ini[35,0]
    i_l_B3_B4_a_r = struct[0].y_ini[36,0]
    i_l_B3_B4_a_i = struct[0].y_ini[37,0]
    i_l_B3_B4_b_r = struct[0].y_ini[38,0]
    i_l_B3_B4_b_i = struct[0].y_ini[39,0]
    i_l_B3_B4_c_r = struct[0].y_ini[40,0]
    i_l_B3_B4_c_i = struct[0].y_ini[41,0]
    i_l_B3_B4_n_r = struct[0].y_ini[42,0]
    i_l_B3_B4_n_i = struct[0].y_ini[43,0]
    i_load_B2_a_r = struct[0].y_ini[44,0]
    i_load_B2_a_i = struct[0].y_ini[45,0]
    i_load_B2_b_r = struct[0].y_ini[46,0]
    i_load_B2_b_i = struct[0].y_ini[47,0]
    i_load_B2_c_r = struct[0].y_ini[48,0]
    i_load_B2_c_i = struct[0].y_ini[49,0]
    i_load_B2_n_r = struct[0].y_ini[50,0]
    i_load_B2_n_i = struct[0].y_ini[51,0]
    i_load_B3_a_r = struct[0].y_ini[52,0]
    i_load_B3_a_i = struct[0].y_ini[53,0]
    i_load_B3_b_r = struct[0].y_ini[54,0]
    i_load_B3_b_i = struct[0].y_ini[55,0]
    i_load_B3_c_r = struct[0].y_ini[56,0]
    i_load_B3_c_i = struct[0].y_ini[57,0]
    i_load_B3_n_r = struct[0].y_ini[58,0]
    i_load_B3_n_i = struct[0].y_ini[59,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_load_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 139.986584618974*v_B2_a_i - 292.221995392108*v_B2_a_r + 23.3310974364957*v_B3_a_i + 48.7036658986847*v_B3_a_r
        struct[0].g[1,0] = i_load_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 292.221995392108*v_B2_a_i + 139.986584618974*v_B2_a_r + 48.7036658986847*v_B3_a_i - 23.3310974364957*v_B3_a_r
        struct[0].g[2,0] = i_load_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 139.986584618974*v_B2_b_i - 292.221995392108*v_B2_b_r + 23.3310974364957*v_B3_b_i + 48.7036658986847*v_B3_b_r
        struct[0].g[3,0] = i_load_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 292.221995392108*v_B2_b_i + 139.986584618974*v_B2_b_r + 48.7036658986847*v_B3_b_i - 23.3310974364957*v_B3_b_r
        struct[0].g[4,0] = i_load_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 139.986584618974*v_B2_c_i - 292.221995392108*v_B2_c_r + 23.3310974364957*v_B3_c_i + 48.7036658986847*v_B3_c_r
        struct[0].g[5,0] = i_load_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 292.221995392108*v_B2_c_i + 139.986584618974*v_B2_c_r + 48.7036658986847*v_B3_c_i - 23.3310974364957*v_B3_c_r
        struct[0].g[6,0] = 116.655487182478*v_B1_n_i + 243.518329493424*v_B1_n_r - 139.986584618974*v_B2_n_i - 292.221995392108*v_B2_n_r + 23.3310974364957*v_B3_n_i + 48.7036658986847*v_B3_n_r
        struct[0].g[7,0] = 243.518329493424*v_B1_n_i - 116.655487182478*v_B1_n_r - 292.221995392108*v_B2_n_i + 139.986584618974*v_B2_n_r + 48.7036658986847*v_B3_n_i - 23.3310974364957*v_B3_n_r
        struct[0].g[8,0] = i_load_B3_a_r + 23.3310974364957*v_B2_a_i + 48.7036658986847*v_B2_a_r - 139.986584618974*v_B3_a_i - 292.221995392108*v_B3_a_r + 116.655487182478*v_B4_a_i + 243.518329493424*v_B4_a_r
        struct[0].g[9,0] = i_load_B3_a_i + 48.7036658986847*v_B2_a_i - 23.3310974364957*v_B2_a_r - 292.221995392108*v_B3_a_i + 139.986584618974*v_B3_a_r + 243.518329493424*v_B4_a_i - 116.655487182478*v_B4_a_r
        struct[0].g[10,0] = i_load_B3_b_r + 23.3310974364957*v_B2_b_i + 48.7036658986847*v_B2_b_r - 139.986584618974*v_B3_b_i - 292.221995392108*v_B3_b_r + 116.655487182478*v_B4_b_i + 243.518329493424*v_B4_b_r
        struct[0].g[11,0] = i_load_B3_b_i + 48.7036658986847*v_B2_b_i - 23.3310974364957*v_B2_b_r - 292.221995392108*v_B3_b_i + 139.986584618974*v_B3_b_r + 243.518329493424*v_B4_b_i - 116.655487182478*v_B4_b_r
        struct[0].g[12,0] = i_load_B3_c_r + 23.3310974364957*v_B2_c_i + 48.7036658986847*v_B2_c_r - 139.986584618974*v_B3_c_i - 292.221995392108*v_B3_c_r + 116.655487182478*v_B4_c_i + 243.518329493424*v_B4_c_r
        struct[0].g[13,0] = i_load_B3_c_i + 48.7036658986847*v_B2_c_i - 23.3310974364957*v_B2_c_r - 292.221995392108*v_B3_c_i + 139.986584618974*v_B3_c_r + 243.518329493424*v_B4_c_i - 116.655487182478*v_B4_c_r
        struct[0].g[14,0] = 23.3310974364957*v_B2_n_i + 48.7036658986847*v_B2_n_r - 139.986584618974*v_B3_n_i - 292.221995392108*v_B3_n_r + 116.655487182478*v_B4_n_i + 243.518329493424*v_B4_n_r
        struct[0].g[15,0] = 48.7036658986847*v_B2_n_i - 23.3310974364957*v_B2_n_r - 292.221995392108*v_B3_n_i + 139.986584618974*v_B3_n_r + 243.518329493424*v_B4_n_i - 116.655487182478*v_B4_n_r
        struct[0].g[16,0] = -116.655487182478*v_B1_n_i - 1243.51832949342*v_B1_n_r + 116.655487182478*v_B2_n_i + 243.518329493424*v_B2_n_r
        struct[0].g[17,0] = -1243.51832949342*v_B1_n_i + 116.655487182478*v_B1_n_r + 243.518329493424*v_B2_n_i - 116.655487182478*v_B2_n_r
        struct[0].g[18,0] = 116.655487182478*v_B3_n_i + 243.518329493424*v_B3_n_r - 116.655487182478*v_B4_n_i - 1243.51832949342*v_B4_n_r
        struct[0].g[19,0] = 243.518329493424*v_B3_n_i - 116.655487182478*v_B3_n_r - 1243.51832949342*v_B4_n_i + 116.655487182478*v_B4_n_r
        struct[0].g[20,0] = -i_l_B1_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 116.655487182478*v_B2_a_i - 243.518329493424*v_B2_a_r
        struct[0].g[21,0] = -i_l_B1_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 243.518329493424*v_B2_a_i + 116.655487182478*v_B2_a_r
        struct[0].g[22,0] = -i_l_B1_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 116.655487182478*v_B2_b_i - 243.518329493424*v_B2_b_r
        struct[0].g[23,0] = -i_l_B1_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 243.518329493424*v_B2_b_i + 116.655487182478*v_B2_b_r
        struct[0].g[24,0] = -i_l_B1_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 116.655487182478*v_B2_c_i - 243.518329493424*v_B2_c_r
        struct[0].g[25,0] = -i_l_B1_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 243.518329493424*v_B2_c_i + 116.655487182478*v_B2_c_r
        struct[0].g[26,0] = i_l_B1_B2_a_r + i_l_B1_B2_b_r + i_l_B1_B2_c_r - i_l_B1_B2_n_r
        struct[0].g[27,0] = i_l_B1_B2_a_i + i_l_B1_B2_b_i + i_l_B1_B2_c_i - i_l_B1_B2_n_i
        struct[0].g[28,0] = -i_l_B2_B3_a_r + 23.3310974364957*v_B2_a_i + 48.7036658986847*v_B2_a_r - 23.3310974364957*v_B3_a_i - 48.7036658986847*v_B3_a_r
        struct[0].g[29,0] = -i_l_B2_B3_a_i + 48.7036658986847*v_B2_a_i - 23.3310974364957*v_B2_a_r - 48.7036658986847*v_B3_a_i + 23.3310974364957*v_B3_a_r
        struct[0].g[30,0] = -i_l_B2_B3_b_r + 23.3310974364957*v_B2_b_i + 48.7036658986847*v_B2_b_r - 23.3310974364957*v_B3_b_i - 48.7036658986847*v_B3_b_r
        struct[0].g[31,0] = -i_l_B2_B3_b_i + 48.7036658986847*v_B2_b_i - 23.3310974364957*v_B2_b_r - 48.7036658986847*v_B3_b_i + 23.3310974364957*v_B3_b_r
        struct[0].g[32,0] = -i_l_B2_B3_c_r + 23.3310974364957*v_B2_c_i + 48.7036658986847*v_B2_c_r - 23.3310974364957*v_B3_c_i - 48.7036658986847*v_B3_c_r
        struct[0].g[33,0] = -i_l_B2_B3_c_i + 48.7036658986847*v_B2_c_i - 23.3310974364957*v_B2_c_r - 48.7036658986847*v_B3_c_i + 23.3310974364957*v_B3_c_r
        struct[0].g[34,0] = i_l_B2_B3_a_r + i_l_B2_B3_b_r + i_l_B2_B3_c_r - i_l_B2_B3_n_r
        struct[0].g[35,0] = i_l_B2_B3_a_i + i_l_B2_B3_b_i + i_l_B2_B3_c_i - i_l_B2_B3_n_i
        struct[0].g[36,0] = -i_l_B3_B4_a_r + 116.655487182478*v_B3_a_i + 243.518329493424*v_B3_a_r - 116.655487182478*v_B4_a_i - 243.518329493424*v_B4_a_r
        struct[0].g[37,0] = -i_l_B3_B4_a_i + 243.518329493424*v_B3_a_i - 116.655487182478*v_B3_a_r - 243.518329493424*v_B4_a_i + 116.655487182478*v_B4_a_r
        struct[0].g[38,0] = -i_l_B3_B4_b_r + 116.655487182478*v_B3_b_i + 243.518329493424*v_B3_b_r - 116.655487182478*v_B4_b_i - 243.518329493424*v_B4_b_r
        struct[0].g[39,0] = -i_l_B3_B4_b_i + 243.518329493424*v_B3_b_i - 116.655487182478*v_B3_b_r - 243.518329493424*v_B4_b_i + 116.655487182478*v_B4_b_r
        struct[0].g[40,0] = -i_l_B3_B4_c_r + 116.655487182478*v_B3_c_i + 243.518329493424*v_B3_c_r - 116.655487182478*v_B4_c_i - 243.518329493424*v_B4_c_r
        struct[0].g[41,0] = -i_l_B3_B4_c_i + 243.518329493424*v_B3_c_i - 116.655487182478*v_B3_c_r - 243.518329493424*v_B4_c_i + 116.655487182478*v_B4_c_r
        struct[0].g[42,0] = i_l_B3_B4_a_r + i_l_B3_B4_b_r + i_l_B3_B4_c_r - i_l_B3_B4_n_r
        struct[0].g[43,0] = i_l_B3_B4_a_i + i_l_B3_B4_b_i + i_l_B3_B4_c_i - i_l_B3_B4_n_i
        struct[0].g[44,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[45,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[46,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[47,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[48,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[49,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[50,0] = i_load_B2_a_r + i_load_B2_b_r + i_load_B2_c_r + i_load_B2_n_r
        struct[0].g[51,0] = i_load_B2_a_i + i_load_B2_b_i + i_load_B2_c_i + i_load_B2_n_i
        struct[0].g[52,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[53,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[54,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[55,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[56,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[57,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
        struct[0].g[58,0] = i_load_B3_a_r + i_load_B3_b_r + i_load_B3_c_r + i_load_B3_n_r
        struct[0].g[59,0] = i_load_B3_a_i + i_load_B3_b_i + i_load_B3_c_i + i_load_B3_n_i
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[4,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[5,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[14,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -1

    if mode == 11:


        struct[0].Gy_ini[0,0] = -292.221995392108
        struct[0].Gy_ini[0,1] = -139.986584618974
        struct[0].Gy_ini[0,8] = 48.7036658986847
        struct[0].Gy_ini[0,9] = 23.3310974364957
        struct[0].Gy_ini[0,44] = 1
        struct[0].Gy_ini[1,0] = 139.986584618974
        struct[0].Gy_ini[1,1] = -292.221995392108
        struct[0].Gy_ini[1,8] = -23.3310974364957
        struct[0].Gy_ini[1,9] = 48.7036658986847
        struct[0].Gy_ini[1,45] = 1
        struct[0].Gy_ini[2,2] = -292.221995392108
        struct[0].Gy_ini[2,3] = -139.986584618974
        struct[0].Gy_ini[2,10] = 48.7036658986847
        struct[0].Gy_ini[2,11] = 23.3310974364957
        struct[0].Gy_ini[2,46] = 1
        struct[0].Gy_ini[3,2] = 139.986584618974
        struct[0].Gy_ini[3,3] = -292.221995392108
        struct[0].Gy_ini[3,10] = -23.3310974364957
        struct[0].Gy_ini[3,11] = 48.7036658986847
        struct[0].Gy_ini[3,47] = 1
        struct[0].Gy_ini[4,4] = -292.221995392108
        struct[0].Gy_ini[4,5] = -139.986584618974
        struct[0].Gy_ini[4,12] = 48.7036658986847
        struct[0].Gy_ini[4,13] = 23.3310974364957
        struct[0].Gy_ini[4,48] = 1
        struct[0].Gy_ini[5,4] = 139.986584618974
        struct[0].Gy_ini[5,5] = -292.221995392108
        struct[0].Gy_ini[5,12] = -23.3310974364957
        struct[0].Gy_ini[5,13] = 48.7036658986847
        struct[0].Gy_ini[5,49] = 1
        struct[0].Gy_ini[6,6] = -292.221995392108
        struct[0].Gy_ini[6,7] = -139.986584618974
        struct[0].Gy_ini[6,14] = 48.7036658986847
        struct[0].Gy_ini[6,15] = 23.3310974364957
        struct[0].Gy_ini[6,16] = 243.518329493424
        struct[0].Gy_ini[6,17] = 116.655487182478
        struct[0].Gy_ini[7,6] = 139.986584618974
        struct[0].Gy_ini[7,7] = -292.221995392108
        struct[0].Gy_ini[7,14] = -23.3310974364957
        struct[0].Gy_ini[7,15] = 48.7036658986847
        struct[0].Gy_ini[7,16] = -116.655487182478
        struct[0].Gy_ini[7,17] = 243.518329493424
        struct[0].Gy_ini[8,0] = 48.7036658986847
        struct[0].Gy_ini[8,1] = 23.3310974364957
        struct[0].Gy_ini[8,8] = -292.221995392108
        struct[0].Gy_ini[8,9] = -139.986584618974
        struct[0].Gy_ini[8,52] = 1
        struct[0].Gy_ini[9,0] = -23.3310974364957
        struct[0].Gy_ini[9,1] = 48.7036658986847
        struct[0].Gy_ini[9,8] = 139.986584618974
        struct[0].Gy_ini[9,9] = -292.221995392108
        struct[0].Gy_ini[9,53] = 1
        struct[0].Gy_ini[10,2] = 48.7036658986847
        struct[0].Gy_ini[10,3] = 23.3310974364957
        struct[0].Gy_ini[10,10] = -292.221995392108
        struct[0].Gy_ini[10,11] = -139.986584618974
        struct[0].Gy_ini[10,54] = 1
        struct[0].Gy_ini[11,2] = -23.3310974364957
        struct[0].Gy_ini[11,3] = 48.7036658986847
        struct[0].Gy_ini[11,10] = 139.986584618974
        struct[0].Gy_ini[11,11] = -292.221995392108
        struct[0].Gy_ini[11,55] = 1
        struct[0].Gy_ini[12,4] = 48.7036658986847
        struct[0].Gy_ini[12,5] = 23.3310974364957
        struct[0].Gy_ini[12,12] = -292.221995392108
        struct[0].Gy_ini[12,13] = -139.986584618974
        struct[0].Gy_ini[12,56] = 1
        struct[0].Gy_ini[13,4] = -23.3310974364957
        struct[0].Gy_ini[13,5] = 48.7036658986847
        struct[0].Gy_ini[13,12] = 139.986584618974
        struct[0].Gy_ini[13,13] = -292.221995392108
        struct[0].Gy_ini[13,57] = 1
        struct[0].Gy_ini[14,6] = 48.7036658986847
        struct[0].Gy_ini[14,7] = 23.3310974364957
        struct[0].Gy_ini[14,14] = -292.221995392108
        struct[0].Gy_ini[14,15] = -139.986584618974
        struct[0].Gy_ini[14,18] = 243.518329493424
        struct[0].Gy_ini[14,19] = 116.655487182478
        struct[0].Gy_ini[15,6] = -23.3310974364957
        struct[0].Gy_ini[15,7] = 48.7036658986847
        struct[0].Gy_ini[15,14] = 139.986584618974
        struct[0].Gy_ini[15,15] = -292.221995392108
        struct[0].Gy_ini[15,18] = -116.655487182478
        struct[0].Gy_ini[15,19] = 243.518329493424
        struct[0].Gy_ini[16,6] = 243.518329493424
        struct[0].Gy_ini[16,7] = 116.655487182478
        struct[0].Gy_ini[16,16] = -1243.51832949342
        struct[0].Gy_ini[16,17] = -116.655487182478
        struct[0].Gy_ini[17,6] = -116.655487182478
        struct[0].Gy_ini[17,7] = 243.518329493424
        struct[0].Gy_ini[17,16] = 116.655487182478
        struct[0].Gy_ini[17,17] = -1243.51832949342
        struct[0].Gy_ini[18,14] = 243.518329493424
        struct[0].Gy_ini[18,15] = 116.655487182478
        struct[0].Gy_ini[18,18] = -1243.51832949342
        struct[0].Gy_ini[18,19] = -116.655487182478
        struct[0].Gy_ini[19,14] = -116.655487182478
        struct[0].Gy_ini[19,15] = 243.518329493424
        struct[0].Gy_ini[19,18] = 116.655487182478
        struct[0].Gy_ini[19,19] = -1243.51832949342
        struct[0].Gy_ini[20,0] = -243.518329493424
        struct[0].Gy_ini[20,1] = -116.655487182478
        struct[0].Gy_ini[20,20] = -1
        struct[0].Gy_ini[21,0] = 116.655487182478
        struct[0].Gy_ini[21,1] = -243.518329493424
        struct[0].Gy_ini[21,21] = -1
        struct[0].Gy_ini[22,2] = -243.518329493424
        struct[0].Gy_ini[22,3] = -116.655487182478
        struct[0].Gy_ini[22,22] = -1
        struct[0].Gy_ini[23,2] = 116.655487182478
        struct[0].Gy_ini[23,3] = -243.518329493424
        struct[0].Gy_ini[23,23] = -1
        struct[0].Gy_ini[24,4] = -243.518329493424
        struct[0].Gy_ini[24,5] = -116.655487182478
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[25,4] = 116.655487182478
        struct[0].Gy_ini[25,5] = -243.518329493424
        struct[0].Gy_ini[25,25] = -1
        struct[0].Gy_ini[26,20] = 1
        struct[0].Gy_ini[26,22] = 1
        struct[0].Gy_ini[26,24] = 1
        struct[0].Gy_ini[26,26] = -1
        struct[0].Gy_ini[27,21] = 1
        struct[0].Gy_ini[27,23] = 1
        struct[0].Gy_ini[27,25] = 1
        struct[0].Gy_ini[27,27] = -1
        struct[0].Gy_ini[28,0] = 48.7036658986847
        struct[0].Gy_ini[28,1] = 23.3310974364957
        struct[0].Gy_ini[28,8] = -48.7036658986847
        struct[0].Gy_ini[28,9] = -23.3310974364957
        struct[0].Gy_ini[28,28] = -1
        struct[0].Gy_ini[29,0] = -23.3310974364957
        struct[0].Gy_ini[29,1] = 48.7036658986847
        struct[0].Gy_ini[29,8] = 23.3310974364957
        struct[0].Gy_ini[29,9] = -48.7036658986847
        struct[0].Gy_ini[29,29] = -1
        struct[0].Gy_ini[30,2] = 48.7036658986847
        struct[0].Gy_ini[30,3] = 23.3310974364957
        struct[0].Gy_ini[30,10] = -48.7036658986847
        struct[0].Gy_ini[30,11] = -23.3310974364957
        struct[0].Gy_ini[30,30] = -1
        struct[0].Gy_ini[31,2] = -23.3310974364957
        struct[0].Gy_ini[31,3] = 48.7036658986847
        struct[0].Gy_ini[31,10] = 23.3310974364957
        struct[0].Gy_ini[31,11] = -48.7036658986847
        struct[0].Gy_ini[31,31] = -1
        struct[0].Gy_ini[32,4] = 48.7036658986847
        struct[0].Gy_ini[32,5] = 23.3310974364957
        struct[0].Gy_ini[32,12] = -48.7036658986847
        struct[0].Gy_ini[32,13] = -23.3310974364957
        struct[0].Gy_ini[32,32] = -1
        struct[0].Gy_ini[33,4] = -23.3310974364957
        struct[0].Gy_ini[33,5] = 48.7036658986847
        struct[0].Gy_ini[33,12] = 23.3310974364957
        struct[0].Gy_ini[33,13] = -48.7036658986847
        struct[0].Gy_ini[33,33] = -1
        struct[0].Gy_ini[34,28] = 1
        struct[0].Gy_ini[34,30] = 1
        struct[0].Gy_ini[34,32] = 1
        struct[0].Gy_ini[34,34] = -1
        struct[0].Gy_ini[35,29] = 1
        struct[0].Gy_ini[35,31] = 1
        struct[0].Gy_ini[35,33] = 1
        struct[0].Gy_ini[35,35] = -1
        struct[0].Gy_ini[36,8] = 243.518329493424
        struct[0].Gy_ini[36,9] = 116.655487182478
        struct[0].Gy_ini[36,36] = -1
        struct[0].Gy_ini[37,8] = -116.655487182478
        struct[0].Gy_ini[37,9] = 243.518329493424
        struct[0].Gy_ini[37,37] = -1
        struct[0].Gy_ini[38,10] = 243.518329493424
        struct[0].Gy_ini[38,11] = 116.655487182478
        struct[0].Gy_ini[38,38] = -1
        struct[0].Gy_ini[39,10] = -116.655487182478
        struct[0].Gy_ini[39,11] = 243.518329493424
        struct[0].Gy_ini[39,39] = -1
        struct[0].Gy_ini[40,12] = 243.518329493424
        struct[0].Gy_ini[40,13] = 116.655487182478
        struct[0].Gy_ini[40,40] = -1
        struct[0].Gy_ini[41,12] = -116.655487182478
        struct[0].Gy_ini[41,13] = 243.518329493424
        struct[0].Gy_ini[41,41] = -1
        struct[0].Gy_ini[42,36] = 1
        struct[0].Gy_ini[42,38] = 1
        struct[0].Gy_ini[42,40] = 1
        struct[0].Gy_ini[42,42] = -1
        struct[0].Gy_ini[43,37] = 1
        struct[0].Gy_ini[43,39] = 1
        struct[0].Gy_ini[43,41] = 1
        struct[0].Gy_ini[43,43] = -1
        struct[0].Gy_ini[44,0] = i_load_B2_a_r
        struct[0].Gy_ini[44,1] = i_load_B2_a_i
        struct[0].Gy_ini[44,6] = -i_load_B2_a_r
        struct[0].Gy_ini[44,7] = -i_load_B2_a_i
        struct[0].Gy_ini[44,44] = v_B2_a_r - v_B2_n_r
        struct[0].Gy_ini[44,45] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[45,2] = i_load_B2_b_r
        struct[0].Gy_ini[45,3] = i_load_B2_b_i
        struct[0].Gy_ini[45,6] = -i_load_B2_b_r
        struct[0].Gy_ini[45,7] = -i_load_B2_b_i
        struct[0].Gy_ini[45,46] = v_B2_b_r - v_B2_n_r
        struct[0].Gy_ini[45,47] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[46,4] = i_load_B2_c_r
        struct[0].Gy_ini[46,5] = i_load_B2_c_i
        struct[0].Gy_ini[46,6] = -i_load_B2_c_r
        struct[0].Gy_ini[46,7] = -i_load_B2_c_i
        struct[0].Gy_ini[46,48] = v_B2_c_r - v_B2_n_r
        struct[0].Gy_ini[46,49] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[47,0] = -i_load_B2_a_i
        struct[0].Gy_ini[47,1] = i_load_B2_a_r
        struct[0].Gy_ini[47,6] = i_load_B2_a_i
        struct[0].Gy_ini[47,7] = -i_load_B2_a_r
        struct[0].Gy_ini[47,44] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[47,45] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy_ini[48,2] = -i_load_B2_b_i
        struct[0].Gy_ini[48,3] = i_load_B2_b_r
        struct[0].Gy_ini[48,6] = i_load_B2_b_i
        struct[0].Gy_ini[48,7] = -i_load_B2_b_r
        struct[0].Gy_ini[48,46] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[48,47] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy_ini[49,4] = -i_load_B2_c_i
        struct[0].Gy_ini[49,5] = i_load_B2_c_r
        struct[0].Gy_ini[49,6] = i_load_B2_c_i
        struct[0].Gy_ini[49,7] = -i_load_B2_c_r
        struct[0].Gy_ini[49,48] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[49,49] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy_ini[50,44] = 1
        struct[0].Gy_ini[50,46] = 1
        struct[0].Gy_ini[50,48] = 1
        struct[0].Gy_ini[50,50] = 1
        struct[0].Gy_ini[51,45] = 1
        struct[0].Gy_ini[51,47] = 1
        struct[0].Gy_ini[51,49] = 1
        struct[0].Gy_ini[51,51] = 1
        struct[0].Gy_ini[52,8] = i_load_B3_a_r
        struct[0].Gy_ini[52,9] = i_load_B3_a_i
        struct[0].Gy_ini[52,14] = -i_load_B3_a_r
        struct[0].Gy_ini[52,15] = -i_load_B3_a_i
        struct[0].Gy_ini[52,52] = v_B3_a_r - v_B3_n_r
        struct[0].Gy_ini[52,53] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[53,10] = i_load_B3_b_r
        struct[0].Gy_ini[53,11] = i_load_B3_b_i
        struct[0].Gy_ini[53,14] = -i_load_B3_b_r
        struct[0].Gy_ini[53,15] = -i_load_B3_b_i
        struct[0].Gy_ini[53,54] = v_B3_b_r - v_B3_n_r
        struct[0].Gy_ini[53,55] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[54,12] = i_load_B3_c_r
        struct[0].Gy_ini[54,13] = i_load_B3_c_i
        struct[0].Gy_ini[54,14] = -i_load_B3_c_r
        struct[0].Gy_ini[54,15] = -i_load_B3_c_i
        struct[0].Gy_ini[54,56] = v_B3_c_r - v_B3_n_r
        struct[0].Gy_ini[54,57] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[55,8] = -i_load_B3_a_i
        struct[0].Gy_ini[55,9] = i_load_B3_a_r
        struct[0].Gy_ini[55,14] = i_load_B3_a_i
        struct[0].Gy_ini[55,15] = -i_load_B3_a_r
        struct[0].Gy_ini[55,52] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[55,53] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy_ini[56,10] = -i_load_B3_b_i
        struct[0].Gy_ini[56,11] = i_load_B3_b_r
        struct[0].Gy_ini[56,14] = i_load_B3_b_i
        struct[0].Gy_ini[56,15] = -i_load_B3_b_r
        struct[0].Gy_ini[56,54] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[56,55] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy_ini[57,12] = -i_load_B3_c_i
        struct[0].Gy_ini[57,13] = i_load_B3_c_r
        struct[0].Gy_ini[57,14] = i_load_B3_c_i
        struct[0].Gy_ini[57,15] = -i_load_B3_c_r
        struct[0].Gy_ini[57,56] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[57,57] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy_ini[58,52] = 1
        struct[0].Gy_ini[58,54] = 1
        struct[0].Gy_ini[58,56] = 1
        struct[0].Gy_ini[58,58] = 1
        struct[0].Gy_ini[59,53] = 1
        struct[0].Gy_ini[59,55] = 1
        struct[0].Gy_ini[59,57] = 1
        struct[0].Gy_ini[59,59] = 1



def run_nn(t,struct,mode):

    # Parameters:
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B4_a_r = struct[0].v_B4_a_r
    v_B4_a_i = struct[0].v_B4_a_i
    v_B4_b_r = struct[0].v_B4_b_r
    v_B4_b_i = struct[0].v_B4_b_i
    v_B4_c_r = struct[0].v_B4_c_r
    v_B4_c_i = struct[0].v_B4_c_i
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
    i_B1_n_r = struct[0].i_B1_n_r
    i_B1_n_i = struct[0].i_B1_n_i
    i_B4_n_r = struct[0].i_B4_n_r
    i_B4_n_i = struct[0].i_B4_n_i
    p_B2_a = struct[0].p_B2_a
    q_B2_a = struct[0].q_B2_a
    p_B2_b = struct[0].p_B2_b
    q_B2_b = struct[0].q_B2_b
    p_B2_c = struct[0].p_B2_c
    q_B2_c = struct[0].q_B2_c
    p_B3_a = struct[0].p_B3_a
    q_B3_a = struct[0].q_B3_a
    p_B3_b = struct[0].p_B3_b
    q_B3_b = struct[0].q_B3_b
    p_B3_c = struct[0].p_B3_c
    q_B3_c = struct[0].q_B3_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2_a_r = struct[0].y_run[0,0]
    v_B2_a_i = struct[0].y_run[1,0]
    v_B2_b_r = struct[0].y_run[2,0]
    v_B2_b_i = struct[0].y_run[3,0]
    v_B2_c_r = struct[0].y_run[4,0]
    v_B2_c_i = struct[0].y_run[5,0]
    v_B2_n_r = struct[0].y_run[6,0]
    v_B2_n_i = struct[0].y_run[7,0]
    v_B3_a_r = struct[0].y_run[8,0]
    v_B3_a_i = struct[0].y_run[9,0]
    v_B3_b_r = struct[0].y_run[10,0]
    v_B3_b_i = struct[0].y_run[11,0]
    v_B3_c_r = struct[0].y_run[12,0]
    v_B3_c_i = struct[0].y_run[13,0]
    v_B3_n_r = struct[0].y_run[14,0]
    v_B3_n_i = struct[0].y_run[15,0]
    v_B1_n_r = struct[0].y_run[16,0]
    v_B1_n_i = struct[0].y_run[17,0]
    v_B4_n_r = struct[0].y_run[18,0]
    v_B4_n_i = struct[0].y_run[19,0]
    i_l_B1_B2_a_r = struct[0].y_run[20,0]
    i_l_B1_B2_a_i = struct[0].y_run[21,0]
    i_l_B1_B2_b_r = struct[0].y_run[22,0]
    i_l_B1_B2_b_i = struct[0].y_run[23,0]
    i_l_B1_B2_c_r = struct[0].y_run[24,0]
    i_l_B1_B2_c_i = struct[0].y_run[25,0]
    i_l_B1_B2_n_r = struct[0].y_run[26,0]
    i_l_B1_B2_n_i = struct[0].y_run[27,0]
    i_l_B2_B3_a_r = struct[0].y_run[28,0]
    i_l_B2_B3_a_i = struct[0].y_run[29,0]
    i_l_B2_B3_b_r = struct[0].y_run[30,0]
    i_l_B2_B3_b_i = struct[0].y_run[31,0]
    i_l_B2_B3_c_r = struct[0].y_run[32,0]
    i_l_B2_B3_c_i = struct[0].y_run[33,0]
    i_l_B2_B3_n_r = struct[0].y_run[34,0]
    i_l_B2_B3_n_i = struct[0].y_run[35,0]
    i_l_B3_B4_a_r = struct[0].y_run[36,0]
    i_l_B3_B4_a_i = struct[0].y_run[37,0]
    i_l_B3_B4_b_r = struct[0].y_run[38,0]
    i_l_B3_B4_b_i = struct[0].y_run[39,0]
    i_l_B3_B4_c_r = struct[0].y_run[40,0]
    i_l_B3_B4_c_i = struct[0].y_run[41,0]
    i_l_B3_B4_n_r = struct[0].y_run[42,0]
    i_l_B3_B4_n_i = struct[0].y_run[43,0]
    i_load_B2_a_r = struct[0].y_run[44,0]
    i_load_B2_a_i = struct[0].y_run[45,0]
    i_load_B2_b_r = struct[0].y_run[46,0]
    i_load_B2_b_i = struct[0].y_run[47,0]
    i_load_B2_c_r = struct[0].y_run[48,0]
    i_load_B2_c_i = struct[0].y_run[49,0]
    i_load_B2_n_r = struct[0].y_run[50,0]
    i_load_B2_n_i = struct[0].y_run[51,0]
    i_load_B3_a_r = struct[0].y_run[52,0]
    i_load_B3_a_i = struct[0].y_run[53,0]
    i_load_B3_b_r = struct[0].y_run[54,0]
    i_load_B3_b_i = struct[0].y_run[55,0]
    i_load_B3_c_r = struct[0].y_run[56,0]
    i_load_B3_c_i = struct[0].y_run[57,0]
    i_load_B3_n_r = struct[0].y_run[58,0]
    i_load_B3_n_i = struct[0].y_run[59,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_load_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 139.986584618974*v_B2_a_i - 292.221995392108*v_B2_a_r + 23.3310974364957*v_B3_a_i + 48.7036658986847*v_B3_a_r
        struct[0].g[1,0] = i_load_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 292.221995392108*v_B2_a_i + 139.986584618974*v_B2_a_r + 48.7036658986847*v_B3_a_i - 23.3310974364957*v_B3_a_r
        struct[0].g[2,0] = i_load_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 139.986584618974*v_B2_b_i - 292.221995392108*v_B2_b_r + 23.3310974364957*v_B3_b_i + 48.7036658986847*v_B3_b_r
        struct[0].g[3,0] = i_load_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 292.221995392108*v_B2_b_i + 139.986584618974*v_B2_b_r + 48.7036658986847*v_B3_b_i - 23.3310974364957*v_B3_b_r
        struct[0].g[4,0] = i_load_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 139.986584618974*v_B2_c_i - 292.221995392108*v_B2_c_r + 23.3310974364957*v_B3_c_i + 48.7036658986847*v_B3_c_r
        struct[0].g[5,0] = i_load_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 292.221995392108*v_B2_c_i + 139.986584618974*v_B2_c_r + 48.7036658986847*v_B3_c_i - 23.3310974364957*v_B3_c_r
        struct[0].g[6,0] = 116.655487182478*v_B1_n_i + 243.518329493424*v_B1_n_r - 139.986584618974*v_B2_n_i - 292.221995392108*v_B2_n_r + 23.3310974364957*v_B3_n_i + 48.7036658986847*v_B3_n_r
        struct[0].g[7,0] = 243.518329493424*v_B1_n_i - 116.655487182478*v_B1_n_r - 292.221995392108*v_B2_n_i + 139.986584618974*v_B2_n_r + 48.7036658986847*v_B3_n_i - 23.3310974364957*v_B3_n_r
        struct[0].g[8,0] = i_load_B3_a_r + 23.3310974364957*v_B2_a_i + 48.7036658986847*v_B2_a_r - 139.986584618974*v_B3_a_i - 292.221995392108*v_B3_a_r + 116.655487182478*v_B4_a_i + 243.518329493424*v_B4_a_r
        struct[0].g[9,0] = i_load_B3_a_i + 48.7036658986847*v_B2_a_i - 23.3310974364957*v_B2_a_r - 292.221995392108*v_B3_a_i + 139.986584618974*v_B3_a_r + 243.518329493424*v_B4_a_i - 116.655487182478*v_B4_a_r
        struct[0].g[10,0] = i_load_B3_b_r + 23.3310974364957*v_B2_b_i + 48.7036658986847*v_B2_b_r - 139.986584618974*v_B3_b_i - 292.221995392108*v_B3_b_r + 116.655487182478*v_B4_b_i + 243.518329493424*v_B4_b_r
        struct[0].g[11,0] = i_load_B3_b_i + 48.7036658986847*v_B2_b_i - 23.3310974364957*v_B2_b_r - 292.221995392108*v_B3_b_i + 139.986584618974*v_B3_b_r + 243.518329493424*v_B4_b_i - 116.655487182478*v_B4_b_r
        struct[0].g[12,0] = i_load_B3_c_r + 23.3310974364957*v_B2_c_i + 48.7036658986847*v_B2_c_r - 139.986584618974*v_B3_c_i - 292.221995392108*v_B3_c_r + 116.655487182478*v_B4_c_i + 243.518329493424*v_B4_c_r
        struct[0].g[13,0] = i_load_B3_c_i + 48.7036658986847*v_B2_c_i - 23.3310974364957*v_B2_c_r - 292.221995392108*v_B3_c_i + 139.986584618974*v_B3_c_r + 243.518329493424*v_B4_c_i - 116.655487182478*v_B4_c_r
        struct[0].g[14,0] = 23.3310974364957*v_B2_n_i + 48.7036658986847*v_B2_n_r - 139.986584618974*v_B3_n_i - 292.221995392108*v_B3_n_r + 116.655487182478*v_B4_n_i + 243.518329493424*v_B4_n_r
        struct[0].g[15,0] = 48.7036658986847*v_B2_n_i - 23.3310974364957*v_B2_n_r - 292.221995392108*v_B3_n_i + 139.986584618974*v_B3_n_r + 243.518329493424*v_B4_n_i - 116.655487182478*v_B4_n_r
        struct[0].g[16,0] = -116.655487182478*v_B1_n_i - 1243.51832949342*v_B1_n_r + 116.655487182478*v_B2_n_i + 243.518329493424*v_B2_n_r
        struct[0].g[17,0] = -1243.51832949342*v_B1_n_i + 116.655487182478*v_B1_n_r + 243.518329493424*v_B2_n_i - 116.655487182478*v_B2_n_r
        struct[0].g[18,0] = 116.655487182478*v_B3_n_i + 243.518329493424*v_B3_n_r - 116.655487182478*v_B4_n_i - 1243.51832949342*v_B4_n_r
        struct[0].g[19,0] = 243.518329493424*v_B3_n_i - 116.655487182478*v_B3_n_r - 1243.51832949342*v_B4_n_i + 116.655487182478*v_B4_n_r
        struct[0].g[20,0] = -i_l_B1_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 116.655487182478*v_B2_a_i - 243.518329493424*v_B2_a_r
        struct[0].g[21,0] = -i_l_B1_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 243.518329493424*v_B2_a_i + 116.655487182478*v_B2_a_r
        struct[0].g[22,0] = -i_l_B1_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 116.655487182478*v_B2_b_i - 243.518329493424*v_B2_b_r
        struct[0].g[23,0] = -i_l_B1_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 243.518329493424*v_B2_b_i + 116.655487182478*v_B2_b_r
        struct[0].g[24,0] = -i_l_B1_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 116.655487182478*v_B2_c_i - 243.518329493424*v_B2_c_r
        struct[0].g[25,0] = -i_l_B1_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 243.518329493424*v_B2_c_i + 116.655487182478*v_B2_c_r
        struct[0].g[26,0] = i_l_B1_B2_a_r + i_l_B1_B2_b_r + i_l_B1_B2_c_r - i_l_B1_B2_n_r
        struct[0].g[27,0] = i_l_B1_B2_a_i + i_l_B1_B2_b_i + i_l_B1_B2_c_i - i_l_B1_B2_n_i
        struct[0].g[28,0] = -i_l_B2_B3_a_r + 23.3310974364957*v_B2_a_i + 48.7036658986847*v_B2_a_r - 23.3310974364957*v_B3_a_i - 48.7036658986847*v_B3_a_r
        struct[0].g[29,0] = -i_l_B2_B3_a_i + 48.7036658986847*v_B2_a_i - 23.3310974364957*v_B2_a_r - 48.7036658986847*v_B3_a_i + 23.3310974364957*v_B3_a_r
        struct[0].g[30,0] = -i_l_B2_B3_b_r + 23.3310974364957*v_B2_b_i + 48.7036658986847*v_B2_b_r - 23.3310974364957*v_B3_b_i - 48.7036658986847*v_B3_b_r
        struct[0].g[31,0] = -i_l_B2_B3_b_i + 48.7036658986847*v_B2_b_i - 23.3310974364957*v_B2_b_r - 48.7036658986847*v_B3_b_i + 23.3310974364957*v_B3_b_r
        struct[0].g[32,0] = -i_l_B2_B3_c_r + 23.3310974364957*v_B2_c_i + 48.7036658986847*v_B2_c_r - 23.3310974364957*v_B3_c_i - 48.7036658986847*v_B3_c_r
        struct[0].g[33,0] = -i_l_B2_B3_c_i + 48.7036658986847*v_B2_c_i - 23.3310974364957*v_B2_c_r - 48.7036658986847*v_B3_c_i + 23.3310974364957*v_B3_c_r
        struct[0].g[34,0] = i_l_B2_B3_a_r + i_l_B2_B3_b_r + i_l_B2_B3_c_r - i_l_B2_B3_n_r
        struct[0].g[35,0] = i_l_B2_B3_a_i + i_l_B2_B3_b_i + i_l_B2_B3_c_i - i_l_B2_B3_n_i
        struct[0].g[36,0] = -i_l_B3_B4_a_r + 116.655487182478*v_B3_a_i + 243.518329493424*v_B3_a_r - 116.655487182478*v_B4_a_i - 243.518329493424*v_B4_a_r
        struct[0].g[37,0] = -i_l_B3_B4_a_i + 243.518329493424*v_B3_a_i - 116.655487182478*v_B3_a_r - 243.518329493424*v_B4_a_i + 116.655487182478*v_B4_a_r
        struct[0].g[38,0] = -i_l_B3_B4_b_r + 116.655487182478*v_B3_b_i + 243.518329493424*v_B3_b_r - 116.655487182478*v_B4_b_i - 243.518329493424*v_B4_b_r
        struct[0].g[39,0] = -i_l_B3_B4_b_i + 243.518329493424*v_B3_b_i - 116.655487182478*v_B3_b_r - 243.518329493424*v_B4_b_i + 116.655487182478*v_B4_b_r
        struct[0].g[40,0] = -i_l_B3_B4_c_r + 116.655487182478*v_B3_c_i + 243.518329493424*v_B3_c_r - 116.655487182478*v_B4_c_i - 243.518329493424*v_B4_c_r
        struct[0].g[41,0] = -i_l_B3_B4_c_i + 243.518329493424*v_B3_c_i - 116.655487182478*v_B3_c_r - 243.518329493424*v_B4_c_i + 116.655487182478*v_B4_c_r
        struct[0].g[42,0] = i_l_B3_B4_a_r + i_l_B3_B4_b_r + i_l_B3_B4_c_r - i_l_B3_B4_n_r
        struct[0].g[43,0] = i_l_B3_B4_a_i + i_l_B3_B4_b_i + i_l_B3_B4_c_i - i_l_B3_B4_n_i
        struct[0].g[44,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[45,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[46,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[47,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[48,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[49,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[50,0] = i_load_B2_a_r + i_load_B2_b_r + i_load_B2_c_r + i_load_B2_n_r
        struct[0].g[51,0] = i_load_B2_a_i + i_load_B2_b_i + i_load_B2_c_i + i_load_B2_n_i
        struct[0].g[52,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[53,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[54,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[55,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[56,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[57,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
        struct[0].g[58,0] = i_load_B3_a_r + i_load_B3_b_r + i_load_B3_c_r + i_load_B3_n_r
        struct[0].g[59,0] = i_load_B3_a_i + i_load_B3_b_i + i_load_B3_c_i + i_load_B3_n_i
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[4,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[5,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[14,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx[0,0] = -1

    if mode == 11:


        struct[0].Gy[0,0] = -292.221995392108
        struct[0].Gy[0,1] = -139.986584618974
        struct[0].Gy[0,8] = 48.7036658986847
        struct[0].Gy[0,9] = 23.3310974364957
        struct[0].Gy[0,44] = 1
        struct[0].Gy[1,0] = 139.986584618974
        struct[0].Gy[1,1] = -292.221995392108
        struct[0].Gy[1,8] = -23.3310974364957
        struct[0].Gy[1,9] = 48.7036658986847
        struct[0].Gy[1,45] = 1
        struct[0].Gy[2,2] = -292.221995392108
        struct[0].Gy[2,3] = -139.986584618974
        struct[0].Gy[2,10] = 48.7036658986847
        struct[0].Gy[2,11] = 23.3310974364957
        struct[0].Gy[2,46] = 1
        struct[0].Gy[3,2] = 139.986584618974
        struct[0].Gy[3,3] = -292.221995392108
        struct[0].Gy[3,10] = -23.3310974364957
        struct[0].Gy[3,11] = 48.7036658986847
        struct[0].Gy[3,47] = 1
        struct[0].Gy[4,4] = -292.221995392108
        struct[0].Gy[4,5] = -139.986584618974
        struct[0].Gy[4,12] = 48.7036658986847
        struct[0].Gy[4,13] = 23.3310974364957
        struct[0].Gy[4,48] = 1
        struct[0].Gy[5,4] = 139.986584618974
        struct[0].Gy[5,5] = -292.221995392108
        struct[0].Gy[5,12] = -23.3310974364957
        struct[0].Gy[5,13] = 48.7036658986847
        struct[0].Gy[5,49] = 1
        struct[0].Gy[6,6] = -292.221995392108
        struct[0].Gy[6,7] = -139.986584618974
        struct[0].Gy[6,14] = 48.7036658986847
        struct[0].Gy[6,15] = 23.3310974364957
        struct[0].Gy[6,16] = 243.518329493424
        struct[0].Gy[6,17] = 116.655487182478
        struct[0].Gy[7,6] = 139.986584618974
        struct[0].Gy[7,7] = -292.221995392108
        struct[0].Gy[7,14] = -23.3310974364957
        struct[0].Gy[7,15] = 48.7036658986847
        struct[0].Gy[7,16] = -116.655487182478
        struct[0].Gy[7,17] = 243.518329493424
        struct[0].Gy[8,0] = 48.7036658986847
        struct[0].Gy[8,1] = 23.3310974364957
        struct[0].Gy[8,8] = -292.221995392108
        struct[0].Gy[8,9] = -139.986584618974
        struct[0].Gy[8,52] = 1
        struct[0].Gy[9,0] = -23.3310974364957
        struct[0].Gy[9,1] = 48.7036658986847
        struct[0].Gy[9,8] = 139.986584618974
        struct[0].Gy[9,9] = -292.221995392108
        struct[0].Gy[9,53] = 1
        struct[0].Gy[10,2] = 48.7036658986847
        struct[0].Gy[10,3] = 23.3310974364957
        struct[0].Gy[10,10] = -292.221995392108
        struct[0].Gy[10,11] = -139.986584618974
        struct[0].Gy[10,54] = 1
        struct[0].Gy[11,2] = -23.3310974364957
        struct[0].Gy[11,3] = 48.7036658986847
        struct[0].Gy[11,10] = 139.986584618974
        struct[0].Gy[11,11] = -292.221995392108
        struct[0].Gy[11,55] = 1
        struct[0].Gy[12,4] = 48.7036658986847
        struct[0].Gy[12,5] = 23.3310974364957
        struct[0].Gy[12,12] = -292.221995392108
        struct[0].Gy[12,13] = -139.986584618974
        struct[0].Gy[12,56] = 1
        struct[0].Gy[13,4] = -23.3310974364957
        struct[0].Gy[13,5] = 48.7036658986847
        struct[0].Gy[13,12] = 139.986584618974
        struct[0].Gy[13,13] = -292.221995392108
        struct[0].Gy[13,57] = 1
        struct[0].Gy[14,6] = 48.7036658986847
        struct[0].Gy[14,7] = 23.3310974364957
        struct[0].Gy[14,14] = -292.221995392108
        struct[0].Gy[14,15] = -139.986584618974
        struct[0].Gy[14,18] = 243.518329493424
        struct[0].Gy[14,19] = 116.655487182478
        struct[0].Gy[15,6] = -23.3310974364957
        struct[0].Gy[15,7] = 48.7036658986847
        struct[0].Gy[15,14] = 139.986584618974
        struct[0].Gy[15,15] = -292.221995392108
        struct[0].Gy[15,18] = -116.655487182478
        struct[0].Gy[15,19] = 243.518329493424
        struct[0].Gy[16,6] = 243.518329493424
        struct[0].Gy[16,7] = 116.655487182478
        struct[0].Gy[16,16] = -1243.51832949342
        struct[0].Gy[16,17] = -116.655487182478
        struct[0].Gy[17,6] = -116.655487182478
        struct[0].Gy[17,7] = 243.518329493424
        struct[0].Gy[17,16] = 116.655487182478
        struct[0].Gy[17,17] = -1243.51832949342
        struct[0].Gy[18,14] = 243.518329493424
        struct[0].Gy[18,15] = 116.655487182478
        struct[0].Gy[18,18] = -1243.51832949342
        struct[0].Gy[18,19] = -116.655487182478
        struct[0].Gy[19,14] = -116.655487182478
        struct[0].Gy[19,15] = 243.518329493424
        struct[0].Gy[19,18] = 116.655487182478
        struct[0].Gy[19,19] = -1243.51832949342
        struct[0].Gy[20,0] = -243.518329493424
        struct[0].Gy[20,1] = -116.655487182478
        struct[0].Gy[20,20] = -1
        struct[0].Gy[21,0] = 116.655487182478
        struct[0].Gy[21,1] = -243.518329493424
        struct[0].Gy[21,21] = -1
        struct[0].Gy[22,2] = -243.518329493424
        struct[0].Gy[22,3] = -116.655487182478
        struct[0].Gy[22,22] = -1
        struct[0].Gy[23,2] = 116.655487182478
        struct[0].Gy[23,3] = -243.518329493424
        struct[0].Gy[23,23] = -1
        struct[0].Gy[24,4] = -243.518329493424
        struct[0].Gy[24,5] = -116.655487182478
        struct[0].Gy[24,24] = -1
        struct[0].Gy[25,4] = 116.655487182478
        struct[0].Gy[25,5] = -243.518329493424
        struct[0].Gy[25,25] = -1
        struct[0].Gy[26,20] = 1
        struct[0].Gy[26,22] = 1
        struct[0].Gy[26,24] = 1
        struct[0].Gy[26,26] = -1
        struct[0].Gy[27,21] = 1
        struct[0].Gy[27,23] = 1
        struct[0].Gy[27,25] = 1
        struct[0].Gy[27,27] = -1
        struct[0].Gy[28,0] = 48.7036658986847
        struct[0].Gy[28,1] = 23.3310974364957
        struct[0].Gy[28,8] = -48.7036658986847
        struct[0].Gy[28,9] = -23.3310974364957
        struct[0].Gy[28,28] = -1
        struct[0].Gy[29,0] = -23.3310974364957
        struct[0].Gy[29,1] = 48.7036658986847
        struct[0].Gy[29,8] = 23.3310974364957
        struct[0].Gy[29,9] = -48.7036658986847
        struct[0].Gy[29,29] = -1
        struct[0].Gy[30,2] = 48.7036658986847
        struct[0].Gy[30,3] = 23.3310974364957
        struct[0].Gy[30,10] = -48.7036658986847
        struct[0].Gy[30,11] = -23.3310974364957
        struct[0].Gy[30,30] = -1
        struct[0].Gy[31,2] = -23.3310974364957
        struct[0].Gy[31,3] = 48.7036658986847
        struct[0].Gy[31,10] = 23.3310974364957
        struct[0].Gy[31,11] = -48.7036658986847
        struct[0].Gy[31,31] = -1
        struct[0].Gy[32,4] = 48.7036658986847
        struct[0].Gy[32,5] = 23.3310974364957
        struct[0].Gy[32,12] = -48.7036658986847
        struct[0].Gy[32,13] = -23.3310974364957
        struct[0].Gy[32,32] = -1
        struct[0].Gy[33,4] = -23.3310974364957
        struct[0].Gy[33,5] = 48.7036658986847
        struct[0].Gy[33,12] = 23.3310974364957
        struct[0].Gy[33,13] = -48.7036658986847
        struct[0].Gy[33,33] = -1
        struct[0].Gy[34,28] = 1
        struct[0].Gy[34,30] = 1
        struct[0].Gy[34,32] = 1
        struct[0].Gy[34,34] = -1
        struct[0].Gy[35,29] = 1
        struct[0].Gy[35,31] = 1
        struct[0].Gy[35,33] = 1
        struct[0].Gy[35,35] = -1
        struct[0].Gy[36,8] = 243.518329493424
        struct[0].Gy[36,9] = 116.655487182478
        struct[0].Gy[36,36] = -1
        struct[0].Gy[37,8] = -116.655487182478
        struct[0].Gy[37,9] = 243.518329493424
        struct[0].Gy[37,37] = -1
        struct[0].Gy[38,10] = 243.518329493424
        struct[0].Gy[38,11] = 116.655487182478
        struct[0].Gy[38,38] = -1
        struct[0].Gy[39,10] = -116.655487182478
        struct[0].Gy[39,11] = 243.518329493424
        struct[0].Gy[39,39] = -1
        struct[0].Gy[40,12] = 243.518329493424
        struct[0].Gy[40,13] = 116.655487182478
        struct[0].Gy[40,40] = -1
        struct[0].Gy[41,12] = -116.655487182478
        struct[0].Gy[41,13] = 243.518329493424
        struct[0].Gy[41,41] = -1
        struct[0].Gy[42,36] = 1
        struct[0].Gy[42,38] = 1
        struct[0].Gy[42,40] = 1
        struct[0].Gy[42,42] = -1
        struct[0].Gy[43,37] = 1
        struct[0].Gy[43,39] = 1
        struct[0].Gy[43,41] = 1
        struct[0].Gy[43,43] = -1
        struct[0].Gy[44,0] = i_load_B2_a_r
        struct[0].Gy[44,1] = i_load_B2_a_i
        struct[0].Gy[44,6] = -i_load_B2_a_r
        struct[0].Gy[44,7] = -i_load_B2_a_i
        struct[0].Gy[44,44] = v_B2_a_r - v_B2_n_r
        struct[0].Gy[44,45] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[45,2] = i_load_B2_b_r
        struct[0].Gy[45,3] = i_load_B2_b_i
        struct[0].Gy[45,6] = -i_load_B2_b_r
        struct[0].Gy[45,7] = -i_load_B2_b_i
        struct[0].Gy[45,46] = v_B2_b_r - v_B2_n_r
        struct[0].Gy[45,47] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[46,4] = i_load_B2_c_r
        struct[0].Gy[46,5] = i_load_B2_c_i
        struct[0].Gy[46,6] = -i_load_B2_c_r
        struct[0].Gy[46,7] = -i_load_B2_c_i
        struct[0].Gy[46,48] = v_B2_c_r - v_B2_n_r
        struct[0].Gy[46,49] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[47,0] = -i_load_B2_a_i
        struct[0].Gy[47,1] = i_load_B2_a_r
        struct[0].Gy[47,6] = i_load_B2_a_i
        struct[0].Gy[47,7] = -i_load_B2_a_r
        struct[0].Gy[47,44] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[47,45] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy[48,2] = -i_load_B2_b_i
        struct[0].Gy[48,3] = i_load_B2_b_r
        struct[0].Gy[48,6] = i_load_B2_b_i
        struct[0].Gy[48,7] = -i_load_B2_b_r
        struct[0].Gy[48,46] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[48,47] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy[49,4] = -i_load_B2_c_i
        struct[0].Gy[49,5] = i_load_B2_c_r
        struct[0].Gy[49,6] = i_load_B2_c_i
        struct[0].Gy[49,7] = -i_load_B2_c_r
        struct[0].Gy[49,48] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[49,49] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy[50,44] = 1
        struct[0].Gy[50,46] = 1
        struct[0].Gy[50,48] = 1
        struct[0].Gy[50,50] = 1
        struct[0].Gy[51,45] = 1
        struct[0].Gy[51,47] = 1
        struct[0].Gy[51,49] = 1
        struct[0].Gy[51,51] = 1
        struct[0].Gy[52,8] = i_load_B3_a_r
        struct[0].Gy[52,9] = i_load_B3_a_i
        struct[0].Gy[52,14] = -i_load_B3_a_r
        struct[0].Gy[52,15] = -i_load_B3_a_i
        struct[0].Gy[52,52] = v_B3_a_r - v_B3_n_r
        struct[0].Gy[52,53] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[53,10] = i_load_B3_b_r
        struct[0].Gy[53,11] = i_load_B3_b_i
        struct[0].Gy[53,14] = -i_load_B3_b_r
        struct[0].Gy[53,15] = -i_load_B3_b_i
        struct[0].Gy[53,54] = v_B3_b_r - v_B3_n_r
        struct[0].Gy[53,55] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[54,12] = i_load_B3_c_r
        struct[0].Gy[54,13] = i_load_B3_c_i
        struct[0].Gy[54,14] = -i_load_B3_c_r
        struct[0].Gy[54,15] = -i_load_B3_c_i
        struct[0].Gy[54,56] = v_B3_c_r - v_B3_n_r
        struct[0].Gy[54,57] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[55,8] = -i_load_B3_a_i
        struct[0].Gy[55,9] = i_load_B3_a_r
        struct[0].Gy[55,14] = i_load_B3_a_i
        struct[0].Gy[55,15] = -i_load_B3_a_r
        struct[0].Gy[55,52] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[55,53] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy[56,10] = -i_load_B3_b_i
        struct[0].Gy[56,11] = i_load_B3_b_r
        struct[0].Gy[56,14] = i_load_B3_b_i
        struct[0].Gy[56,15] = -i_load_B3_b_r
        struct[0].Gy[56,54] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[56,55] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy[57,12] = -i_load_B3_c_i
        struct[0].Gy[57,13] = i_load_B3_c_r
        struct[0].Gy[57,14] = i_load_B3_c_i
        struct[0].Gy[57,15] = -i_load_B3_c_r
        struct[0].Gy[57,56] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[57,57] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy[58,52] = 1
        struct[0].Gy[58,54] = 1
        struct[0].Gy[58,56] = 1
        struct[0].Gy[58,58] = 1
        struct[0].Gy[59,53] = 1
        struct[0].Gy[59,55] = 1
        struct[0].Gy[59,57] = 1
        struct[0].Gy[59,59] = 1

        struct[0].Gu[0,0] = 243.518329493424
        struct[0].Gu[0,1] = 116.655487182478
        struct[0].Gu[1,0] = -116.655487182478
        struct[0].Gu[1,1] = 243.518329493424
        struct[0].Gu[2,2] = 243.518329493424
        struct[0].Gu[2,3] = 116.655487182478
        struct[0].Gu[3,2] = -116.655487182478
        struct[0].Gu[3,3] = 243.518329493424
        struct[0].Gu[4,4] = 243.518329493424
        struct[0].Gu[4,5] = 116.655487182478
        struct[0].Gu[5,4] = -116.655487182478
        struct[0].Gu[5,5] = 243.518329493424
        struct[0].Gu[8,6] = 243.518329493424
        struct[0].Gu[8,7] = 116.655487182478
        struct[0].Gu[9,6] = -116.655487182478
        struct[0].Gu[9,7] = 243.518329493424
        struct[0].Gu[10,8] = 243.518329493424
        struct[0].Gu[10,9] = 116.655487182478
        struct[0].Gu[11,8] = -116.655487182478
        struct[0].Gu[11,9] = 243.518329493424
        struct[0].Gu[12,10] = 243.518329493424
        struct[0].Gu[12,11] = 116.655487182478
        struct[0].Gu[13,10] = -116.655487182478
        struct[0].Gu[13,11] = 243.518329493424
        struct[0].Gu[20,0] = 243.518329493424
        struct[0].Gu[20,1] = 116.655487182478
        struct[0].Gu[21,0] = -116.655487182478
        struct[0].Gu[21,1] = 243.518329493424
        struct[0].Gu[22,2] = 243.518329493424
        struct[0].Gu[22,3] = 116.655487182478
        struct[0].Gu[23,2] = -116.655487182478
        struct[0].Gu[23,3] = 243.518329493424
        struct[0].Gu[24,4] = 243.518329493424
        struct[0].Gu[24,5] = 116.655487182478
        struct[0].Gu[25,4] = -116.655487182478
        struct[0].Gu[25,5] = 243.518329493424
        struct[0].Gu[36,6] = -243.518329493424
        struct[0].Gu[36,7] = -116.655487182478
        struct[0].Gu[37,6] = 116.655487182478
        struct[0].Gu[37,7] = -243.518329493424
        struct[0].Gu[38,8] = -243.518329493424
        struct[0].Gu[38,9] = -116.655487182478
        struct[0].Gu[39,8] = 116.655487182478
        struct[0].Gu[39,9] = -243.518329493424
        struct[0].Gu[40,10] = -243.518329493424
        struct[0].Gu[40,11] = -116.655487182478
        struct[0].Gu[41,10] = 116.655487182478
        struct[0].Gu[41,11] = -243.518329493424
        struct[0].Gu[44,20] = -1
        struct[0].Gu[45,22] = -1
        struct[0].Gu[46,24] = -1
        struct[0].Gu[47,21] = -1
        struct[0].Gu[48,23] = -1
        struct[0].Gu[49,25] = -1
        struct[0].Gu[52,26] = -1
        struct[0].Gu[53,28] = -1
        struct[0].Gu[54,30] = -1
        struct[0].Gu[55,27] = -1
        struct[0].Gu[56,29] = -1
        struct[0].Gu[57,31] = -1





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
    Fx_ini_rows = [0]

    Fx_ini_cols = [0]

    Fy_ini_rows = []

    Fy_ini_cols = []

    Gx_ini_rows = []

    Gx_ini_cols = []

    Gy_ini_rows = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59]

    Gy_ini_cols = [0, 1, 8, 9, 44, 0, 1, 8, 9, 45, 2, 3, 10, 11, 46, 2, 3, 10, 11, 47, 4, 5, 12, 13, 48, 4, 5, 12, 13, 49, 6, 7, 14, 15, 16, 17, 6, 7, 14, 15, 16, 17, 0, 1, 8, 9, 52, 0, 1, 8, 9, 53, 2, 3, 10, 11, 54, 2, 3, 10, 11, 55, 4, 5, 12, 13, 56, 4, 5, 12, 13, 57, 6, 7, 14, 15, 18, 19, 6, 7, 14, 15, 18, 19, 6, 7, 16, 17, 6, 7, 16, 17, 14, 15, 18, 19, 14, 15, 18, 19, 0, 1, 20, 0, 1, 21, 2, 3, 22, 2, 3, 23, 4, 5, 24, 4, 5, 25, 20, 22, 24, 26, 21, 23, 25, 27, 0, 1, 8, 9, 28, 0, 1, 8, 9, 29, 2, 3, 10, 11, 30, 2, 3, 10, 11, 31, 4, 5, 12, 13, 32, 4, 5, 12, 13, 33, 28, 30, 32, 34, 29, 31, 33, 35, 8, 9, 36, 8, 9, 37, 10, 11, 38, 10, 11, 39, 12, 13, 40, 12, 13, 41, 36, 38, 40, 42, 37, 39, 41, 43, 0, 1, 6, 7, 44, 45, 2, 3, 6, 7, 46, 47, 4, 5, 6, 7, 48, 49, 0, 1, 6, 7, 44, 45, 2, 3, 6, 7, 46, 47, 4, 5, 6, 7, 48, 49, 44, 46, 48, 50, 45, 47, 49, 51, 8, 9, 14, 15, 52, 53, 10, 11, 14, 15, 54, 55, 12, 13, 14, 15, 56, 57, 8, 9, 14, 15, 52, 53, 10, 11, 14, 15, 54, 55, 12, 13, 14, 15, 56, 57, 52, 54, 56, 58, 53, 55, 57, 59]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols