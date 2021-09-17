import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 


class smib_milano_ex8p1_4ord_avr_pss_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 7
        self.N_y = 12 
        self.N_z = 1 
        self.N_store = 10000 
        self.params_list = ['X_d', 'X1d', 'T1d0', 'X_q', 'X1q', 'T1q0', 'R_a', 'X_l', 'H', 'D', 'Omega_b', 'omega_s', 'v_0', 'theta_0', 'K_a', 'T_e', 'T_wo', 'T_1', 'T_2', 'K_stab'] 
        self.params_values_list  = [1.81, 0.3, 8.0, 1.76, 0.65, 1.0, 0.003, 0.1, 3.5, 0.0, 314.1592653589793, 1.0, 1.0, 0.0, 100, 0.1, 10.0, 0.1, 0.1, 0.0] 
        self.inputs_ini_list = ['p_t', 'v_t'] 
        self.inputs_ini_values_list  = [0.8, 1.0] 
        self.inputs_run_list = ['p_m', 'v_ref'] 
        self.inputs_run_values_list = [0.8, 1.0] 
        self.outputs_list = ['p_m'] 
        self.x_list = ['delta', 'omega', 'e1q', 'e1d', 'v_c', 'x_wo', 'x_l'] 
        self.y_run_list = ['v_d', 'v_q', 'i_d', 'i_q', 'p_e', 'p_t', 'q_t', 'v_t', 'theta_t', 'v_f', 'z_wo', 'v_pss'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_d', 'v_q', 'i_d', 'i_q', 'p_e', 'p_m', 'q_t', 'v_ref', 'theta_t', 'v_f', 'z_wo', 'v_pss'] 
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
    X_d = struct[0].X_d
    X1d = struct[0].X1d
    T1d0 = struct[0].T1d0
    X_q = struct[0].X_q
    X1q = struct[0].X1q
    T1q0 = struct[0].T1q0
    R_a = struct[0].R_a
    X_l = struct[0].X_l
    H = struct[0].H
    D = struct[0].D
    Omega_b = struct[0].Omega_b
    omega_s = struct[0].omega_s
    v_0 = struct[0].v_0
    theta_0 = struct[0].theta_0
    K_a = struct[0].K_a
    T_e = struct[0].T_e
    T_wo = struct[0].T_wo
    T_1 = struct[0].T_1
    T_2 = struct[0].T_2
    K_stab = struct[0].K_stab
    
    # Inputs:
    p_t = struct[0].p_t
    v_t = struct[0].v_t
    
    # Dynamical states:
    delta = struct[0].x[0,0]
    omega = struct[0].x[1,0]
    e1q = struct[0].x[2,0]
    e1d = struct[0].x[3,0]
    v_c = struct[0].x[4,0]
    x_wo = struct[0].x[5,0]
    x_l = struct[0].x[6,0]
    
    # Algebraic states:
    v_d = struct[0].y_ini[0,0]
    v_q = struct[0].y_ini[1,0]
    i_d = struct[0].y_ini[2,0]
    i_q = struct[0].y_ini[3,0]
    p_e = struct[0].y_ini[4,0]
    p_m = struct[0].y_ini[5,0]
    q_t = struct[0].y_ini[6,0]
    v_ref = struct[0].y_ini[7,0]
    theta_t = struct[0].y_ini[8,0]
    v_f = struct[0].y_ini[9,0]
    z_wo = struct[0].y_ini[10,0]
    v_pss = struct[0].y_ini[11,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b*(omega - omega_s)
        struct[0].f[1,0] = (-D*(omega - omega_s) - p_e + p_m)/(2*H)
        struct[0].f[2,0] = (-e1q - i_d*(-X1d + X_d) + v_f)/T1d0
        struct[0].f[3,0] = (-e1d + i_q*(-X1q + X_q))/T1q0
        struct[0].f[4,0] = (-v_c + v_t)/T_e
        struct[0].f[5,0] = (omega - omega_s - x_wo)/T_wo
        struct[0].f[6,0] = (-x_l + z_wo)/T_2
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -v_d + v_t*sin(delta - theta_t)
        struct[0].g[1,0] = -v_q + v_t*cos(delta - theta_t)
        struct[0].g[2,0] = R_a*i_q + X1d*i_d - e1q + v_q
        struct[0].g[3,0] = R_a*i_d - X1q*i_q - e1d + v_d
        struct[0].g[4,0] = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e
        struct[0].g[5,0] = i_d*v_d + i_q*v_q - p_t
        struct[0].g[6,0] = i_d*v_q - i_q*v_d - q_t
        struct[0].g[7,0] = p_t + v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].g[8,0] = q_t + v_0*v_t*cos(theta_0 - theta_t)/X_l - v_t**2/X_l
        struct[0].g[9,0] = K_a*(-v_c + v_pss + v_ref) - v_f
        struct[0].g[10,0] = omega - omega_s - x_wo - z_wo
        struct[0].g[11,0] = K_stab*(T_1*(-x_l + z_wo)/T_2 + x_l) - v_pss
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = p_m
    

    if mode == 10:

        struct[0].Fx_ini[0,1] = Omega_b
        struct[0].Fx_ini[1,1] = -D/(2*H)
        struct[0].Fx_ini[2,2] = -1/T1d0
        struct[0].Fx_ini[3,3] = -1/T1q0
        struct[0].Fx_ini[4,4] = -1/T_e
        struct[0].Fx_ini[5,1] = 1/T_wo
        struct[0].Fx_ini[5,5] = -1/T_wo
        struct[0].Fx_ini[6,6] = -1/T_2

    if mode == 11:

        struct[0].Fy_ini[1,4] = -1/(2*H) 
        struct[0].Fy_ini[1,5] = 1/(2*H) 
        struct[0].Fy_ini[2,2] = (X1d - X_d)/T1d0 
        struct[0].Fy_ini[2,9] = 1/T1d0 
        struct[0].Fy_ini[3,3] = (-X1q + X_q)/T1q0 
        struct[0].Fy_ini[6,10] = 1/T_2 

        struct[0].Gx_ini[0,0] = v_t*cos(delta - theta_t)
        struct[0].Gx_ini[1,0] = -v_t*sin(delta - theta_t)
        struct[0].Gx_ini[2,2] = -1
        struct[0].Gx_ini[3,3] = -1
        struct[0].Gx_ini[9,4] = -K_a
        struct[0].Gx_ini[10,1] = 1
        struct[0].Gx_ini[10,5] = -1
        struct[0].Gx_ini[11,6] = K_stab*(-T_1/T_2 + 1)

        struct[0].Gy_ini[0,8] = -v_t*cos(delta - theta_t)
        struct[0].Gy_ini[1,8] = v_t*sin(delta - theta_t)
        struct[0].Gy_ini[2,2] = X1d
        struct[0].Gy_ini[2,3] = R_a
        struct[0].Gy_ini[3,2] = R_a
        struct[0].Gy_ini[3,3] = -X1q
        struct[0].Gy_ini[4,0] = i_d
        struct[0].Gy_ini[4,1] = i_q
        struct[0].Gy_ini[4,2] = 2*R_a*i_d + v_d
        struct[0].Gy_ini[4,3] = 2*R_a*i_q + v_q
        struct[0].Gy_ini[5,0] = i_d
        struct[0].Gy_ini[5,1] = i_q
        struct[0].Gy_ini[5,2] = v_d
        struct[0].Gy_ini[5,3] = v_q
        struct[0].Gy_ini[6,0] = -i_q
        struct[0].Gy_ini[6,1] = i_d
        struct[0].Gy_ini[6,2] = v_q
        struct[0].Gy_ini[6,3] = -v_d
        struct[0].Gy_ini[7,8] = -v_0*v_t*cos(theta_0 - theta_t)/X_l
        struct[0].Gy_ini[8,8] = v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].Gy_ini[9,7] = K_a
        struct[0].Gy_ini[9,11] = K_a
        struct[0].Gy_ini[11,10] = K_stab*T_1/T_2



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    X_d = struct[0].X_d
    X1d = struct[0].X1d
    T1d0 = struct[0].T1d0
    X_q = struct[0].X_q
    X1q = struct[0].X1q
    T1q0 = struct[0].T1q0
    R_a = struct[0].R_a
    X_l = struct[0].X_l
    H = struct[0].H
    D = struct[0].D
    Omega_b = struct[0].Omega_b
    omega_s = struct[0].omega_s
    v_0 = struct[0].v_0
    theta_0 = struct[0].theta_0
    K_a = struct[0].K_a
    T_e = struct[0].T_e
    T_wo = struct[0].T_wo
    T_1 = struct[0].T_1
    T_2 = struct[0].T_2
    K_stab = struct[0].K_stab
    
    # Inputs:
    p_m = struct[0].p_m
    v_ref = struct[0].v_ref
    
    # Dynamical states:
    delta = struct[0].x[0,0]
    omega = struct[0].x[1,0]
    e1q = struct[0].x[2,0]
    e1d = struct[0].x[3,0]
    v_c = struct[0].x[4,0]
    x_wo = struct[0].x[5,0]
    x_l = struct[0].x[6,0]
    
    # Algebraic states:
    v_d = struct[0].y_run[0,0]
    v_q = struct[0].y_run[1,0]
    i_d = struct[0].y_run[2,0]
    i_q = struct[0].y_run[3,0]
    p_e = struct[0].y_run[4,0]
    p_t = struct[0].y_run[5,0]
    q_t = struct[0].y_run[6,0]
    v_t = struct[0].y_run[7,0]
    theta_t = struct[0].y_run[8,0]
    v_f = struct[0].y_run[9,0]
    z_wo = struct[0].y_run[10,0]
    v_pss = struct[0].y_run[11,0]
    
    struct[0].u_run[0,0] = p_m
    struct[0].u_run[1,0] = v_ref
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b*(omega - omega_s)
        struct[0].f[1,0] = (-D*(omega - omega_s) - p_e + p_m)/(2*H)
        struct[0].f[2,0] = (-e1q - i_d*(-X1d + X_d) + v_f)/T1d0
        struct[0].f[3,0] = (-e1d + i_q*(-X1q + X_q))/T1q0
        struct[0].f[4,0] = (-v_c + v_t)/T_e
        struct[0].f[5,0] = (omega - omega_s - x_wo)/T_wo
        struct[0].f[6,0] = (-x_l + z_wo)/T_2
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -v_d + v_t*sin(delta - theta_t)
        struct[0].g[1,0] = -v_q + v_t*cos(delta - theta_t)
        struct[0].g[2,0] = R_a*i_q + X1d*i_d - e1q + v_q
        struct[0].g[3,0] = R_a*i_d - X1q*i_q - e1d + v_d
        struct[0].g[4,0] = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e
        struct[0].g[5,0] = i_d*v_d + i_q*v_q - p_t
        struct[0].g[6,0] = i_d*v_q - i_q*v_d - q_t
        struct[0].g[7,0] = p_t + v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].g[8,0] = q_t + v_0*v_t*cos(theta_0 - theta_t)/X_l - v_t**2/X_l
        struct[0].g[9,0] = K_a*(-v_c + v_pss + v_ref) - v_f
        struct[0].g[10,0] = omega - omega_s - x_wo - z_wo
        struct[0].g[11,0] = K_stab*(T_1*(-x_l + z_wo)/T_2 + x_l) - v_pss
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = p_m
    

    if mode == 10:

        struct[0].Fx[0,1] = Omega_b
        struct[0].Fx[1,1] = -D/(2*H)
        struct[0].Fx[2,2] = -1/T1d0
        struct[0].Fx[3,3] = -1/T1q0
        struct[0].Fx[4,4] = -1/T_e
        struct[0].Fx[5,1] = 1/T_wo
        struct[0].Fx[5,5] = -1/T_wo
        struct[0].Fx[6,6] = -1/T_2

    if mode == 11:

        struct[0].Fy[1,4] = -1/(2*H)
        struct[0].Fy[2,2] = (X1d - X_d)/T1d0
        struct[0].Fy[2,9] = 1/T1d0
        struct[0].Fy[3,3] = (-X1q + X_q)/T1q0
        struct[0].Fy[4,7] = 1/T_e
        struct[0].Fy[6,10] = 1/T_2

        struct[0].Gx[0,0] = v_t*cos(delta - theta_t)
        struct[0].Gx[1,0] = -v_t*sin(delta - theta_t)
        struct[0].Gx[2,2] = -1
        struct[0].Gx[3,3] = -1
        struct[0].Gx[9,4] = -K_a
        struct[0].Gx[10,1] = 1
        struct[0].Gx[10,5] = -1
        struct[0].Gx[11,6] = K_stab*(-T_1/T_2 + 1)

        struct[0].Gy[0,7] = sin(delta - theta_t)
        struct[0].Gy[0,8] = -v_t*cos(delta - theta_t)
        struct[0].Gy[1,7] = cos(delta - theta_t)
        struct[0].Gy[1,8] = v_t*sin(delta - theta_t)
        struct[0].Gy[2,2] = X1d
        struct[0].Gy[2,3] = R_a
        struct[0].Gy[3,2] = R_a
        struct[0].Gy[3,3] = -X1q
        struct[0].Gy[4,0] = i_d
        struct[0].Gy[4,1] = i_q
        struct[0].Gy[4,2] = 2*R_a*i_d + v_d
        struct[0].Gy[4,3] = 2*R_a*i_q + v_q
        struct[0].Gy[5,0] = i_d
        struct[0].Gy[5,1] = i_q
        struct[0].Gy[5,2] = v_d
        struct[0].Gy[5,3] = v_q
        struct[0].Gy[6,0] = -i_q
        struct[0].Gy[6,1] = i_d
        struct[0].Gy[6,2] = v_q
        struct[0].Gy[6,3] = -v_d
        struct[0].Gy[7,7] = v_0*sin(theta_0 - theta_t)/X_l
        struct[0].Gy[7,8] = -v_0*v_t*cos(theta_0 - theta_t)/X_l
        struct[0].Gy[8,7] = v_0*cos(theta_0 - theta_t)/X_l - 2*v_t/X_l
        struct[0].Gy[8,8] = v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].Gy[9,11] = K_a
        struct[0].Gy[11,10] = K_stab*T_1/T_2

    if mode > 12:





        pass



def ini_nn(struct,mode):

    # Parameters:
    X_d = struct[0].X_d
    X1d = struct[0].X1d
    T1d0 = struct[0].T1d0
    X_q = struct[0].X_q
    X1q = struct[0].X1q
    T1q0 = struct[0].T1q0
    R_a = struct[0].R_a
    X_l = struct[0].X_l
    H = struct[0].H
    D = struct[0].D
    Omega_b = struct[0].Omega_b
    omega_s = struct[0].omega_s
    v_0 = struct[0].v_0
    theta_0 = struct[0].theta_0
    K_a = struct[0].K_a
    T_e = struct[0].T_e
    T_wo = struct[0].T_wo
    T_1 = struct[0].T_1
    T_2 = struct[0].T_2
    K_stab = struct[0].K_stab
    
    # Inputs:
    p_t = struct[0].p_t
    v_t = struct[0].v_t
    
    # Dynamical states:
    delta = struct[0].x[0,0]
    omega = struct[0].x[1,0]
    e1q = struct[0].x[2,0]
    e1d = struct[0].x[3,0]
    v_c = struct[0].x[4,0]
    x_wo = struct[0].x[5,0]
    x_l = struct[0].x[6,0]
    
    # Algebraic states:
    v_d = struct[0].y_ini[0,0]
    v_q = struct[0].y_ini[1,0]
    i_d = struct[0].y_ini[2,0]
    i_q = struct[0].y_ini[3,0]
    p_e = struct[0].y_ini[4,0]
    p_m = struct[0].y_ini[5,0]
    q_t = struct[0].y_ini[6,0]
    v_ref = struct[0].y_ini[7,0]
    theta_t = struct[0].y_ini[8,0]
    v_f = struct[0].y_ini[9,0]
    z_wo = struct[0].y_ini[10,0]
    v_pss = struct[0].y_ini[11,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b*(omega - omega_s)
        struct[0].f[1,0] = (-D*(omega - omega_s) - p_e + p_m)/(2*H)
        struct[0].f[2,0] = (-e1q - i_d*(-X1d + X_d) + v_f)/T1d0
        struct[0].f[3,0] = (-e1d + i_q*(-X1q + X_q))/T1q0
        struct[0].f[4,0] = (-v_c + v_t)/T_e
        struct[0].f[5,0] = (omega - omega_s - x_wo)/T_wo
        struct[0].f[6,0] = (-x_l + z_wo)/T_2
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -v_d + v_t*sin(delta - theta_t)
        struct[0].g[1,0] = -v_q + v_t*cos(delta - theta_t)
        struct[0].g[2,0] = R_a*i_q + X1d*i_d - e1q + v_q
        struct[0].g[3,0] = R_a*i_d - X1q*i_q - e1d + v_d
        struct[0].g[4,0] = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e
        struct[0].g[5,0] = i_d*v_d + i_q*v_q - p_t
        struct[0].g[6,0] = i_d*v_q - i_q*v_d - q_t
        struct[0].g[7,0] = p_t + v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].g[8,0] = q_t + v_0*v_t*cos(theta_0 - theta_t)/X_l - v_t**2/X_l
        struct[0].g[9,0] = K_a*(-v_c + v_pss + v_ref) - v_f
        struct[0].g[10,0] = omega - omega_s - x_wo - z_wo
        struct[0].g[11,0] = K_stab*(T_1*(-x_l + z_wo)/T_2 + x_l) - v_pss
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = p_m
    

    if mode == 10:

        struct[0].Fx_ini[0,1] = Omega_b
        struct[0].Fx_ini[1,1] = -D/(2*H)
        struct[0].Fx_ini[2,2] = -1/T1d0
        struct[0].Fx_ini[3,3] = -1/T1q0
        struct[0].Fx_ini[4,4] = -1/T_e
        struct[0].Fx_ini[5,1] = 1/T_wo
        struct[0].Fx_ini[5,5] = -1/T_wo
        struct[0].Fx_ini[6,6] = -1/T_2

    if mode == 11:

        struct[0].Fy_ini[1,4] = -1/(2*H) 
        struct[0].Fy_ini[1,5] = 1/(2*H) 
        struct[0].Fy_ini[2,2] = (X1d - X_d)/T1d0 
        struct[0].Fy_ini[2,9] = 1/T1d0 
        struct[0].Fy_ini[3,3] = (-X1q + X_q)/T1q0 
        struct[0].Fy_ini[6,10] = 1/T_2 

        struct[0].Gy_ini[0,0] = -1
        struct[0].Gy_ini[0,8] = -v_t*cos(delta - theta_t)
        struct[0].Gy_ini[1,1] = -1
        struct[0].Gy_ini[1,8] = v_t*sin(delta - theta_t)
        struct[0].Gy_ini[2,1] = 1
        struct[0].Gy_ini[2,2] = X1d
        struct[0].Gy_ini[2,3] = R_a
        struct[0].Gy_ini[3,0] = 1
        struct[0].Gy_ini[3,2] = R_a
        struct[0].Gy_ini[3,3] = -X1q
        struct[0].Gy_ini[4,0] = i_d
        struct[0].Gy_ini[4,1] = i_q
        struct[0].Gy_ini[4,2] = 2*R_a*i_d + v_d
        struct[0].Gy_ini[4,3] = 2*R_a*i_q + v_q
        struct[0].Gy_ini[4,4] = -1
        struct[0].Gy_ini[5,0] = i_d
        struct[0].Gy_ini[5,1] = i_q
        struct[0].Gy_ini[5,2] = v_d
        struct[0].Gy_ini[5,3] = v_q
        struct[0].Gy_ini[6,0] = -i_q
        struct[0].Gy_ini[6,1] = i_d
        struct[0].Gy_ini[6,2] = v_q
        struct[0].Gy_ini[6,3] = -v_d
        struct[0].Gy_ini[6,6] = -1
        struct[0].Gy_ini[7,8] = -v_0*v_t*cos(theta_0 - theta_t)/X_l
        struct[0].Gy_ini[8,6] = 1
        struct[0].Gy_ini[8,8] = v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].Gy_ini[9,7] = K_a
        struct[0].Gy_ini[9,9] = -1
        struct[0].Gy_ini[9,11] = K_a
        struct[0].Gy_ini[10,10] = -1
        struct[0].Gy_ini[11,10] = K_stab*T_1/T_2
        struct[0].Gy_ini[11,11] = -1



def run_nn(t,struct,mode):

    # Parameters:
    X_d = struct[0].X_d
    X1d = struct[0].X1d
    T1d0 = struct[0].T1d0
    X_q = struct[0].X_q
    X1q = struct[0].X1q
    T1q0 = struct[0].T1q0
    R_a = struct[0].R_a
    X_l = struct[0].X_l
    H = struct[0].H
    D = struct[0].D
    Omega_b = struct[0].Omega_b
    omega_s = struct[0].omega_s
    v_0 = struct[0].v_0
    theta_0 = struct[0].theta_0
    K_a = struct[0].K_a
    T_e = struct[0].T_e
    T_wo = struct[0].T_wo
    T_1 = struct[0].T_1
    T_2 = struct[0].T_2
    K_stab = struct[0].K_stab
    
    # Inputs:
    p_m = struct[0].p_m
    v_ref = struct[0].v_ref
    
    # Dynamical states:
    delta = struct[0].x[0,0]
    omega = struct[0].x[1,0]
    e1q = struct[0].x[2,0]
    e1d = struct[0].x[3,0]
    v_c = struct[0].x[4,0]
    x_wo = struct[0].x[5,0]
    x_l = struct[0].x[6,0]
    
    # Algebraic states:
    v_d = struct[0].y_run[0,0]
    v_q = struct[0].y_run[1,0]
    i_d = struct[0].y_run[2,0]
    i_q = struct[0].y_run[3,0]
    p_e = struct[0].y_run[4,0]
    p_t = struct[0].y_run[5,0]
    q_t = struct[0].y_run[6,0]
    v_t = struct[0].y_run[7,0]
    theta_t = struct[0].y_run[8,0]
    v_f = struct[0].y_run[9,0]
    z_wo = struct[0].y_run[10,0]
    v_pss = struct[0].y_run[11,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b*(omega - omega_s)
        struct[0].f[1,0] = (-D*(omega - omega_s) - p_e + p_m)/(2*H)
        struct[0].f[2,0] = (-e1q - i_d*(-X1d + X_d) + v_f)/T1d0
        struct[0].f[3,0] = (-e1d + i_q*(-X1q + X_q))/T1q0
        struct[0].f[4,0] = (-v_c + v_t)/T_e
        struct[0].f[5,0] = (omega - omega_s - x_wo)/T_wo
        struct[0].f[6,0] = (-x_l + z_wo)/T_2
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -v_d + v_t*sin(delta - theta_t)
        struct[0].g[1,0] = -v_q + v_t*cos(delta - theta_t)
        struct[0].g[2,0] = R_a*i_q + X1d*i_d - e1q + v_q
        struct[0].g[3,0] = R_a*i_d - X1q*i_q - e1d + v_d
        struct[0].g[4,0] = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e
        struct[0].g[5,0] = i_d*v_d + i_q*v_q - p_t
        struct[0].g[6,0] = i_d*v_q - i_q*v_d - q_t
        struct[0].g[7,0] = p_t + v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].g[8,0] = q_t + v_0*v_t*cos(theta_0 - theta_t)/X_l - v_t**2/X_l
        struct[0].g[9,0] = K_a*(-v_c + v_pss + v_ref) - v_f
        struct[0].g[10,0] = omega - omega_s - x_wo - z_wo
        struct[0].g[11,0] = K_stab*(T_1*(-x_l + z_wo)/T_2 + x_l) - v_pss
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = p_m
    

    if mode == 10:

        struct[0].Fx[0,1] = Omega_b
        struct[0].Fx[1,1] = -D/(2*H)
        struct[0].Fx[2,2] = -1/T1d0
        struct[0].Fx[3,3] = -1/T1q0
        struct[0].Fx[4,4] = -1/T_e
        struct[0].Fx[5,1] = 1/T_wo
        struct[0].Fx[5,5] = -1/T_wo
        struct[0].Fx[6,6] = -1/T_2

    if mode == 11:

        struct[0].Fy[1,4] = -1/(2*H)
        struct[0].Fy[2,2] = (X1d - X_d)/T1d0
        struct[0].Fy[2,9] = 1/T1d0
        struct[0].Fy[3,3] = (-X1q + X_q)/T1q0
        struct[0].Fy[4,7] = 1/T_e
        struct[0].Fy[6,10] = 1/T_2

        struct[0].Gy[0,0] = -1
        struct[0].Gy[0,7] = sin(delta - theta_t)
        struct[0].Gy[0,8] = -v_t*cos(delta - theta_t)
        struct[0].Gy[1,1] = -1
        struct[0].Gy[1,7] = cos(delta - theta_t)
        struct[0].Gy[1,8] = v_t*sin(delta - theta_t)
        struct[0].Gy[2,1] = 1
        struct[0].Gy[2,2] = X1d
        struct[0].Gy[2,3] = R_a
        struct[0].Gy[3,0] = 1
        struct[0].Gy[3,2] = R_a
        struct[0].Gy[3,3] = -X1q
        struct[0].Gy[4,0] = i_d
        struct[0].Gy[4,1] = i_q
        struct[0].Gy[4,2] = 2*R_a*i_d + v_d
        struct[0].Gy[4,3] = 2*R_a*i_q + v_q
        struct[0].Gy[4,4] = -1
        struct[0].Gy[5,0] = i_d
        struct[0].Gy[5,1] = i_q
        struct[0].Gy[5,2] = v_d
        struct[0].Gy[5,3] = v_q
        struct[0].Gy[5,5] = -1
        struct[0].Gy[6,0] = -i_q
        struct[0].Gy[6,1] = i_d
        struct[0].Gy[6,2] = v_q
        struct[0].Gy[6,3] = -v_d
        struct[0].Gy[6,6] = -1
        struct[0].Gy[7,5] = 1
        struct[0].Gy[7,7] = v_0*sin(theta_0 - theta_t)/X_l
        struct[0].Gy[7,8] = -v_0*v_t*cos(theta_0 - theta_t)/X_l
        struct[0].Gy[8,6] = 1
        struct[0].Gy[8,7] = v_0*cos(theta_0 - theta_t)/X_l - 2*v_t/X_l
        struct[0].Gy[8,8] = v_0*v_t*sin(theta_0 - theta_t)/X_l
        struct[0].Gy[9,9] = -1
        struct[0].Gy[9,11] = K_a
        struct[0].Gy[10,10] = -1
        struct[0].Gy[11,10] = K_stab*T_1/T_2
        struct[0].Gy[11,11] = -1






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
    Fx_ini_rows = [0, 1, 2, 3, 4, 5, 5, 6]

    Fx_ini_cols = [1, 1, 2, 3, 4, 1, 5, 6]

    Fy_ini_rows = [1, 1, 2, 2, 3, 6]

    Fy_ini_cols = [4, 5, 2, 9, 3, 10]

    Gx_ini_rows = [0, 1, 2, 3, 9, 10, 10, 11]

    Gx_ini_cols = [0, 0, 2, 3, 4, 1, 5, 6]

    Gy_ini_rows = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 8, 8, 9, 9, 9, 10, 11, 11]

    Gy_ini_cols = [0, 8, 1, 8, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3, 6, 8, 6, 8, 7, 9, 11, 10, 10, 11]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols
