import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 


class grid_3bus4wire_class: 

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
        self.N_y = 32 
        self.N_z = 1 
        self.N_store = 10000 
        self.params_list = ['a'] 
        self.params_values_list  = [1] 
        self.inputs_ini_list = ['v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'i_B3_n_r', 'i_B3_n_i', 'i_B2_n_r', 'i_B2_n_i', 'p_B2_a', 'q_B2_a', 'p_B2_b', 'q_B2_b', 'p_B2_c', 'q_B2_c', 'p_B3_a', 'q_B3_a', 'p_B3_b', 'q_B3_b', 'p_B3_c', 'q_B3_c', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -2.2737367544323206e-13, 1.1368683772161603e-13, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -85000.00011979532, -52678.26838479351, -85000.00011979532, -52678.2683847935, -85000.00011979532, -52678.26838479348, 1.0] 
        self.inputs_run_list = ['v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'i_B3_n_r', 'i_B3_n_i', 'i_B2_n_r', 'i_B2_n_i', 'p_B2_a', 'q_B2_a', 'p_B2_b', 'q_B2_b', 'p_B2_c', 'q_B2_c', 'p_B3_a', 'q_B3_a', 'p_B3_b', 'q_B3_b', 'p_B3_c', 'q_B3_c', 'u_dummy'] 
        self.inputs_run_values_list = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -2.2737367544323206e-13, 1.1368683772161603e-13, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -85000.00011979532, -52678.26838479351, -85000.00011979532, -52678.2683847935, -85000.00011979532, -52678.26838479348, 1.0] 
        self.outputs_list = ['x_dummy'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B3_n_r', 'v_B3_n_i', 'v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B2_n_r', 'v_B2_n_i', 'i_B2_a_r', 'i_B2_a_i', 'i_B2_b_r', 'i_B2_b_i', 'i_B2_c_r', 'i_B2_c_i', 'i_B2_n_r', 'i_B2_n_i', 'i_B3_a_r', 'i_B3_a_i', 'i_B3_b_r', 'i_B3_b_i', 'i_B3_c_r', 'i_B3_c_i', 'i_B3_n_r', 'i_B3_n_i'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B3_n_r', 'v_B3_n_i', 'v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B2_n_r', 'v_B2_n_i', 'i_B2_a_r', 'i_B2_a_i', 'i_B2_b_r', 'i_B2_b_i', 'i_B2_c_r', 'i_B2_c_i', 'i_B2_n_r', 'i_B2_n_i', 'i_B3_a_r', 'i_B3_a_i', 'i_B3_b_r', 'i_B3_b_i', 'i_B3_c_r', 'i_B3_c_i', 'i_B3_n_r', 'i_B3_n_i'] 
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
        ini(self.struct,2)
        ini(self.struct,3)       
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg

    def run_problem(self,x):
        t = self.struct[0].t
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(t,self.struct,2)
        run(t,self.struct,3)
        run(t,self.struct,10)
        run(t,self.struct,11)
        run(t,self.struct,12)
        run(t,self.struct,13)
        
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
    
    def eval_jacobians(self):

        run(0.0,self.struct,10)
        run(0.0,self.struct,11)  
        run(0.0,self.struct,12) 

        return 1


    def ini_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini(self.struct,10)
        ini(self.struct,11)       
        A_c = np.block([[self.struct[0].Fx_ini,self.struct[0].Fy_ini],
                        [self.struct[0].Gx_ini,self.struct[0].Gy_ini]])
        return A_c



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
        
        ## solve 
        #daesolver(self.struct)    # run until first event

        # simulation run
        for event in events[1:]:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        
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
        
        
    def initialize(self,events,xy0=0):
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
        if xy0 == 0:
            xy0 = np.zeros(self.N_x+self.N_y)
        elif xy0 == 1:
            xy0 = np.ones(self.N_x+self.N_y)
        elif xy0 == 'prev':
            xy0 = self.xy_prev
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

            # evaluate f and g
            run(0.0,self.struct,2)
            run(0.0,self.struct,3)                

            
            # evaluate run jacobians 
            run(0.0,self.struct,10)
            run(0.0,self.struct,11)                
            run(0.0,self.struct,12) 
            run(0.0,self.struct,14) 
             
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
    
    def set_value(self,name,value):
        if name in self.inputs_run_list:
            self.struct[0][name] = value
        if name in self.params_list:
            self.struct[0][name] = value


@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    a = struct[0].a
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
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
    v_B3_a_r = struct[0].y_run[0,0]
    v_B3_a_i = struct[0].y_run[1,0]
    v_B3_b_r = struct[0].y_run[2,0]
    v_B3_b_i = struct[0].y_run[3,0]
    v_B3_c_r = struct[0].y_run[4,0]
    v_B3_c_i = struct[0].y_run[5,0]
    v_B3_n_r = struct[0].y_run[6,0]
    v_B3_n_i = struct[0].y_run[7,0]
    v_B2_a_r = struct[0].y_run[8,0]
    v_B2_a_i = struct[0].y_run[9,0]
    v_B2_b_r = struct[0].y_run[10,0]
    v_B2_b_i = struct[0].y_run[11,0]
    v_B2_c_r = struct[0].y_run[12,0]
    v_B2_c_i = struct[0].y_run[13,0]
    v_B2_n_r = struct[0].y_run[14,0]
    v_B2_n_i = struct[0].y_run[15,0]
    i_B2_a_r = struct[0].y_run[16,0]
    i_B2_a_i = struct[0].y_run[17,0]
    i_B2_b_r = struct[0].y_run[18,0]
    i_B2_b_i = struct[0].y_run[19,0]
    i_B2_c_r = struct[0].y_run[20,0]
    i_B2_c_i = struct[0].y_run[21,0]
    i_B2_n_r = struct[0].y_run[22,0]
    i_B2_n_i = struct[0].y_run[23,0]
    i_B3_a_r = struct[0].y_run[24,0]
    i_B3_a_i = struct[0].y_run[25,0]
    i_B3_b_r = struct[0].y_run[26,0]
    i_B3_b_i = struct[0].y_run[27,0]
    i_B3_c_r = struct[0].y_run[28,0]
    i_B3_c_i = struct[0].y_run[29,0]
    i_B3_n_r = struct[0].y_run[30,0]
    i_B3_n_i = struct[0].y_run[31,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -0.00706562359705999*i_B2_a_i + 0.00273067829517978*i_B2_a_r - 9.62359705999139e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 9.62359705999139e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999124e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 0.01506562359706*i_B3_a_i + 0.0194306782951798*i_B3_a_r - 9.62359705999132e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 9.62359705999132e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r + 6.50521303491303e-19*v_B1_a_i + 0.0121243556529821*v_B1_a_r - 6.50521303491303e-19*v_B1_c_i - 0.0121243556529821*v_B1_c_r - v_B3_a_r
        struct[0].g[1,0] = 0.00273067829517978*i_B2_a_i + 0.00706562359705999*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999139e-6*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999139e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999124e-6*i_B2_n_r + 0.0194306782951798*i_B3_a_i + 0.01506562359706*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999132e-6*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999132e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r + 0.0121243556529821*v_B1_a_i - 6.50521303491303e-19*v_B1_a_r - 0.0121243556529821*v_B1_c_i + 6.50521303491303e-19*v_B1_c_r - v_B3_a_i
        struct[0].g[2,0] = -9.62359705999139e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 0.00706562359705999*i_B2_b_i + 0.00273067829517978*i_B2_b_r - 9.62359705999139e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999124e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999132e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 0.01506562359706*i_B3_b_i + 0.0194306782951798*i_B3_b_r - 9.62359705999132e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - 6.50521303491303e-19*v_B1_a_i - 0.0121243556529821*v_B1_a_r + 6.50521303491303e-19*v_B1_b_i + 0.0121243556529821*v_B1_b_r - v_B3_b_r
        struct[0].g[3,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999139e-6*i_B2_a_r + 0.00273067829517978*i_B2_b_i + 0.00706562359705999*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999139e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999124e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999132e-6*i_B3_a_r + 0.0194306782951798*i_B3_b_i + 0.01506562359706*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999132e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_a_i + 6.50521303491303e-19*v_B1_a_r + 0.0121243556529821*v_B1_b_i - 6.50521303491303e-19*v_B1_b_r - v_B3_b_i
        struct[0].g[4,0] = -9.62359705999139e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 9.62359705999139e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 0.00706562359705999*i_B2_c_i + 0.00273067829517978*i_B2_c_r - 9.62359705999124e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999132e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 9.62359705999132e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 0.01506562359706*i_B3_c_i + 0.0194306782951798*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - 6.50521303491303e-19*v_B1_b_i - 0.0121243556529821*v_B1_b_r + 6.50521303491303e-19*v_B1_c_i + 0.0121243556529821*v_B1_c_r - v_B3_c_r
        struct[0].g[5,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999139e-6*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999139e-6*i_B2_b_r + 0.00273067829517978*i_B2_c_i + 0.00706562359705999*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999124e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999132e-6*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999132e-6*i_B3_b_r + 0.0194306782951798*i_B3_c_i + 0.01506562359706*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_b_i + 6.50521303491303e-19*v_B1_b_r + 0.0121243556529821*v_B1_c_i - 6.50521303491303e-19*v_B1_c_r - v_B3_c_i
        struct[0].g[6,0] = 9.62359705999109e-5*i_B2_a_i + 0.000333217048202192*i_B2_a_r + 9.62359705999109e-5*i_B2_b_i + 0.000333217048202192*i_B2_b_r + 9.62359705999109e-5*i_B2_c_i + 0.000333217048202192*i_B2_c_r + 9.6235970599911e-5*i_B2_n_i + 0.000333217048202192*i_B2_n_r + 9.62359705999109e-5*i_B3_a_i + 0.000333217048202192*i_B3_a_r + 9.62359705999109e-5*i_B3_b_i + 0.000333217048202192*i_B3_b_r + 9.62359705999109e-5*i_B3_c_i + 0.000333217048202192*i_B3_c_r - 0.00096235970599911*i_B3_n_i + 0.00666782951797808*i_B3_n_r - v_B3_n_r
        struct[0].g[7,0] = 0.000333217048202192*i_B2_a_i - 9.62359705999109e-5*i_B2_a_r + 0.000333217048202192*i_B2_b_i - 9.62359705999109e-5*i_B2_b_r + 0.000333217048202192*i_B2_c_i - 9.62359705999109e-5*i_B2_c_r + 0.000333217048202192*i_B2_n_i - 9.6235970599911e-5*i_B2_n_r + 0.000333217048202192*i_B3_a_i - 9.62359705999109e-5*i_B3_a_r + 0.000333217048202192*i_B3_b_i - 9.62359705999109e-5*i_B3_b_r + 0.000333217048202192*i_B3_c_i - 9.62359705999109e-5*i_B3_c_r + 0.00666782951797808*i_B3_n_i + 0.00096235970599911*i_B3_n_r - v_B3_n_i
        struct[0].g[8,0] = -0.00706562359705999*i_B2_a_i + 0.00273067829517978*i_B2_a_r - 9.62359705999132e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 9.62359705999132e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999127e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 0.00706562359705999*i_B3_a_i + 0.00273067829517978*i_B3_a_r - 9.62359705999127e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 9.62359705999127e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - 1.0842021724855e-18*v_B1_a_i + 0.0121243556529821*v_B1_a_r + 1.0842021724855e-18*v_B1_c_i - 0.0121243556529821*v_B1_c_r - v_B2_a_r
        struct[0].g[9,0] = 0.00273067829517978*i_B2_a_i + 0.00706562359705999*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999132e-6*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999132e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999127e-6*i_B2_n_r + 0.00273067829517978*i_B3_a_i + 0.00706562359705999*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999127e-6*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999127e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r + 0.0121243556529821*v_B1_a_i + 1.0842021724855e-18*v_B1_a_r - 0.0121243556529821*v_B1_c_i - 1.0842021724855e-18*v_B1_c_r - v_B2_a_i
        struct[0].g[10,0] = -9.62359705999132e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 0.00706562359705999*i_B2_b_i + 0.00273067829517978*i_B2_b_r - 9.62359705999132e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999127e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999127e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 0.00706562359705999*i_B3_b_i + 0.00273067829517978*i_B3_b_r - 9.62359705999127e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r + 1.0842021724855e-18*v_B1_a_i - 0.0121243556529821*v_B1_a_r - 1.0842021724855e-18*v_B1_b_i + 0.0121243556529821*v_B1_b_r - v_B2_b_r
        struct[0].g[11,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999132e-6*i_B2_a_r + 0.00273067829517978*i_B2_b_i + 0.00706562359705999*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999132e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999127e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999127e-6*i_B3_a_r + 0.00273067829517978*i_B3_b_i + 0.00706562359705999*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999127e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_a_i - 1.0842021724855e-18*v_B1_a_r + 0.0121243556529821*v_B1_b_i + 1.0842021724855e-18*v_B1_b_r - v_B2_b_i
        struct[0].g[12,0] = -9.62359705999132e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 9.62359705999132e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 0.00706562359705999*i_B2_c_i + 0.00273067829517978*i_B2_c_r - 9.62359705999127e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999127e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 9.62359705999127e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 0.00706562359705999*i_B3_c_i + 0.00273067829517978*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r + 1.0842021724855e-18*v_B1_b_i - 0.0121243556529821*v_B1_b_r - 1.0842021724855e-18*v_B1_c_i + 0.0121243556529821*v_B1_c_r - v_B2_c_r
        struct[0].g[13,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999132e-6*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999132e-6*i_B2_b_r + 0.00273067829517978*i_B2_c_i + 0.00706562359705999*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999127e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999127e-6*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999127e-6*i_B3_b_r + 0.00273067829517978*i_B3_c_i + 0.00706562359705999*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_b_i - 1.0842021724855e-18*v_B1_b_r + 0.0121243556529821*v_B1_c_i + 1.0842021724855e-18*v_B1_c_r - v_B2_c_i
        struct[0].g[14,0] = -9.62359705999126e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 9.62359705999126e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 9.62359705999126e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.6235970599912e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999122e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 9.62359705999122e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 9.62359705999122e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.6235970599911e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - v_B2_n_r
        struct[0].g[15,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999126e-6*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999126e-6*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999126e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.6235970599912e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999122e-6*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999122e-6*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999122e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.6235970599911e-5*i_B3_n_r - v_B2_n_i
        struct[0].g[16,0] = i_B2_a_i*v_B2_a_i - i_B2_a_i*v_B2_n_i + i_B2_a_r*v_B2_a_r - i_B2_a_r*v_B2_n_r + p_B2_a
        struct[0].g[17,0] = i_B2_b_i*v_B2_b_i - i_B2_b_i*v_B2_n_i + i_B2_b_r*v_B2_b_r - i_B2_b_r*v_B2_n_r + p_B2_b
        struct[0].g[18,0] = i_B2_c_i*v_B2_c_i - i_B2_c_i*v_B2_n_i + i_B2_c_r*v_B2_c_r - i_B2_c_r*v_B2_n_r + p_B2_c
        struct[0].g[19,0] = -i_B2_a_i*v_B2_a_r + i_B2_a_i*v_B2_n_r + i_B2_a_r*v_B2_a_i - i_B2_a_r*v_B2_n_i + q_B2_a
        struct[0].g[20,0] = -i_B2_b_i*v_B2_b_r + i_B2_b_i*v_B2_n_r + i_B2_b_r*v_B2_b_i - i_B2_b_r*v_B2_n_i + q_B2_b
        struct[0].g[21,0] = -i_B2_c_i*v_B2_c_r + i_B2_c_i*v_B2_n_r + i_B2_c_r*v_B2_c_i - i_B2_c_r*v_B2_n_i + q_B2_c
        struct[0].g[22,0] = i_B2_a_r + i_B2_b_r + i_B2_c_r + i_B2_n_r
        struct[0].g[23,0] = i_B2_a_i + i_B2_b_i + i_B2_c_i + i_B2_n_i
        struct[0].g[24,0] = i_B3_a_i*v_B3_a_i - i_B3_a_i*v_B3_n_i + i_B3_a_r*v_B3_a_r - i_B3_a_r*v_B3_n_r + p_B3_a
        struct[0].g[25,0] = i_B3_b_i*v_B3_b_i - i_B3_b_i*v_B3_n_i + i_B3_b_r*v_B3_b_r - i_B3_b_r*v_B3_n_r + p_B3_b
        struct[0].g[26,0] = i_B3_c_i*v_B3_c_i - i_B3_c_i*v_B3_n_i + i_B3_c_r*v_B3_c_r - i_B3_c_r*v_B3_n_r + p_B3_c
        struct[0].g[27,0] = -i_B3_a_i*v_B3_a_r + i_B3_a_i*v_B3_n_r + i_B3_a_r*v_B3_a_i - i_B3_a_r*v_B3_n_i + q_B3_a
        struct[0].g[28,0] = -i_B3_b_i*v_B3_b_r + i_B3_b_i*v_B3_n_r + i_B3_b_r*v_B3_b_i - i_B3_b_r*v_B3_n_i + q_B3_b
        struct[0].g[29,0] = -i_B3_c_i*v_B3_c_r + i_B3_c_i*v_B3_n_r + i_B3_c_r*v_B3_c_i - i_B3_c_r*v_B3_n_i + q_B3_c
        struct[0].g[30,0] = i_B3_a_r + i_B3_b_r + i_B3_c_r + i_B3_n_r
        struct[0].g[31,0] = i_B3_a_i + i_B3_b_i + i_B3_c_i + i_B3_n_i
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = x_dummy
    

    if mode == 10:

        struct[0].Fx[0,0] = -1

    if mode == 11:



        struct[0].Gy[0,0] = -1
        struct[0].Gy[0,16] = 0.00273067829517978
        struct[0].Gy[0,17] = -0.00706562359705999
        struct[0].Gy[0,18] = 0.000966678295179781
        struct[0].Gy[0,19] = -0.00000962359705999139
        struct[0].Gy[0,20] = 0.000966678295179781
        struct[0].Gy[0,21] = -0.00000962359705999139
        struct[0].Gy[0,22] = 0.000966678295179781
        struct[0].Gy[0,23] = -0.00000962359705999124
        struct[0].Gy[0,24] = 0.0194306782951798
        struct[0].Gy[0,25] = -0.0150656235970600
        struct[0].Gy[0,26] = 0.000966678295179781
        struct[0].Gy[0,27] = -0.00000962359705999132
        struct[0].Gy[0,28] = 0.000966678295179781
        struct[0].Gy[0,29] = -0.00000962359705999132
        struct[0].Gy[0,30] = 0.000333217048202192
        struct[0].Gy[0,31] = 0.0000962359705999109
        struct[0].Gy[1,1] = -1
        struct[0].Gy[1,16] = 0.00706562359705999
        struct[0].Gy[1,17] = 0.00273067829517978
        struct[0].Gy[1,18] = 0.00000962359705999139
        struct[0].Gy[1,19] = 0.000966678295179781
        struct[0].Gy[1,20] = 0.00000962359705999139
        struct[0].Gy[1,21] = 0.000966678295179781
        struct[0].Gy[1,22] = 0.00000962359705999124
        struct[0].Gy[1,23] = 0.000966678295179781
        struct[0].Gy[1,24] = 0.0150656235970600
        struct[0].Gy[1,25] = 0.0194306782951798
        struct[0].Gy[1,26] = 0.00000962359705999132
        struct[0].Gy[1,27] = 0.000966678295179781
        struct[0].Gy[1,28] = 0.00000962359705999132
        struct[0].Gy[1,29] = 0.000966678295179781
        struct[0].Gy[1,30] = -0.0000962359705999109
        struct[0].Gy[1,31] = 0.000333217048202192
        struct[0].Gy[2,2] = -1
        struct[0].Gy[2,16] = 0.000966678295179781
        struct[0].Gy[2,17] = -0.00000962359705999139
        struct[0].Gy[2,18] = 0.00273067829517978
        struct[0].Gy[2,19] = -0.00706562359705999
        struct[0].Gy[2,20] = 0.000966678295179781
        struct[0].Gy[2,21] = -0.00000962359705999139
        struct[0].Gy[2,22] = 0.000966678295179781
        struct[0].Gy[2,23] = -0.00000962359705999124
        struct[0].Gy[2,24] = 0.000966678295179781
        struct[0].Gy[2,25] = -0.00000962359705999132
        struct[0].Gy[2,26] = 0.0194306782951798
        struct[0].Gy[2,27] = -0.0150656235970600
        struct[0].Gy[2,28] = 0.000966678295179781
        struct[0].Gy[2,29] = -0.00000962359705999132
        struct[0].Gy[2,30] = 0.000333217048202192
        struct[0].Gy[2,31] = 0.0000962359705999109
        struct[0].Gy[3,3] = -1
        struct[0].Gy[3,16] = 0.00000962359705999139
        struct[0].Gy[3,17] = 0.000966678295179781
        struct[0].Gy[3,18] = 0.00706562359705999
        struct[0].Gy[3,19] = 0.00273067829517978
        struct[0].Gy[3,20] = 0.00000962359705999139
        struct[0].Gy[3,21] = 0.000966678295179781
        struct[0].Gy[3,22] = 0.00000962359705999124
        struct[0].Gy[3,23] = 0.000966678295179781
        struct[0].Gy[3,24] = 0.00000962359705999132
        struct[0].Gy[3,25] = 0.000966678295179781
        struct[0].Gy[3,26] = 0.0150656235970600
        struct[0].Gy[3,27] = 0.0194306782951798
        struct[0].Gy[3,28] = 0.00000962359705999132
        struct[0].Gy[3,29] = 0.000966678295179781
        struct[0].Gy[3,30] = -0.0000962359705999109
        struct[0].Gy[3,31] = 0.000333217048202192
        struct[0].Gy[4,4] = -1
        struct[0].Gy[4,16] = 0.000966678295179781
        struct[0].Gy[4,17] = -0.00000962359705999139
        struct[0].Gy[4,18] = 0.000966678295179781
        struct[0].Gy[4,19] = -0.00000962359705999139
        struct[0].Gy[4,20] = 0.00273067829517978
        struct[0].Gy[4,21] = -0.00706562359705999
        struct[0].Gy[4,22] = 0.000966678295179781
        struct[0].Gy[4,23] = -0.00000962359705999124
        struct[0].Gy[4,24] = 0.000966678295179781
        struct[0].Gy[4,25] = -0.00000962359705999132
        struct[0].Gy[4,26] = 0.000966678295179781
        struct[0].Gy[4,27] = -0.00000962359705999132
        struct[0].Gy[4,28] = 0.0194306782951798
        struct[0].Gy[4,29] = -0.0150656235970600
        struct[0].Gy[4,30] = 0.000333217048202192
        struct[0].Gy[4,31] = 0.0000962359705999109
        struct[0].Gy[5,5] = -1
        struct[0].Gy[5,16] = 0.00000962359705999139
        struct[0].Gy[5,17] = 0.000966678295179781
        struct[0].Gy[5,18] = 0.00000962359705999139
        struct[0].Gy[5,19] = 0.000966678295179781
        struct[0].Gy[5,20] = 0.00706562359705999
        struct[0].Gy[5,21] = 0.00273067829517978
        struct[0].Gy[5,22] = 0.00000962359705999124
        struct[0].Gy[5,23] = 0.000966678295179781
        struct[0].Gy[5,24] = 0.00000962359705999132
        struct[0].Gy[5,25] = 0.000966678295179781
        struct[0].Gy[5,26] = 0.00000962359705999132
        struct[0].Gy[5,27] = 0.000966678295179781
        struct[0].Gy[5,28] = 0.0150656235970600
        struct[0].Gy[5,29] = 0.0194306782951798
        struct[0].Gy[5,30] = -0.0000962359705999109
        struct[0].Gy[5,31] = 0.000333217048202192
        struct[0].Gy[6,6] = -1
        struct[0].Gy[6,16] = 0.000333217048202192
        struct[0].Gy[6,17] = 0.0000962359705999109
        struct[0].Gy[6,18] = 0.000333217048202192
        struct[0].Gy[6,19] = 0.0000962359705999109
        struct[0].Gy[6,20] = 0.000333217048202192
        struct[0].Gy[6,21] = 0.0000962359705999109
        struct[0].Gy[6,22] = 0.000333217048202192
        struct[0].Gy[6,23] = 0.0000962359705999110
        struct[0].Gy[6,24] = 0.000333217048202192
        struct[0].Gy[6,25] = 0.0000962359705999109
        struct[0].Gy[6,26] = 0.000333217048202192
        struct[0].Gy[6,27] = 0.0000962359705999109
        struct[0].Gy[6,28] = 0.000333217048202192
        struct[0].Gy[6,29] = 0.0000962359705999109
        struct[0].Gy[6,30] = 0.00666782951797808
        struct[0].Gy[6,31] = -0.000962359705999110
        struct[0].Gy[7,7] = -1
        struct[0].Gy[7,16] = -0.0000962359705999109
        struct[0].Gy[7,17] = 0.000333217048202192
        struct[0].Gy[7,18] = -0.0000962359705999109
        struct[0].Gy[7,19] = 0.000333217048202192
        struct[0].Gy[7,20] = -0.0000962359705999109
        struct[0].Gy[7,21] = 0.000333217048202192
        struct[0].Gy[7,22] = -0.0000962359705999110
        struct[0].Gy[7,23] = 0.000333217048202192
        struct[0].Gy[7,24] = -0.0000962359705999109
        struct[0].Gy[7,25] = 0.000333217048202192
        struct[0].Gy[7,26] = -0.0000962359705999109
        struct[0].Gy[7,27] = 0.000333217048202192
        struct[0].Gy[7,28] = -0.0000962359705999109
        struct[0].Gy[7,29] = 0.000333217048202192
        struct[0].Gy[7,30] = 0.000962359705999110
        struct[0].Gy[7,31] = 0.00666782951797808
        struct[0].Gy[8,8] = -1
        struct[0].Gy[8,16] = 0.00273067829517978
        struct[0].Gy[8,17] = -0.00706562359705999
        struct[0].Gy[8,18] = 0.000966678295179781
        struct[0].Gy[8,19] = -0.00000962359705999132
        struct[0].Gy[8,20] = 0.000966678295179781
        struct[0].Gy[8,21] = -0.00000962359705999132
        struct[0].Gy[8,22] = 0.000966678295179781
        struct[0].Gy[8,23] = -0.00000962359705999127
        struct[0].Gy[8,24] = 0.00273067829517978
        struct[0].Gy[8,25] = -0.00706562359705999
        struct[0].Gy[8,26] = 0.000966678295179781
        struct[0].Gy[8,27] = -0.00000962359705999127
        struct[0].Gy[8,28] = 0.000966678295179781
        struct[0].Gy[8,29] = -0.00000962359705999127
        struct[0].Gy[8,30] = 0.000333217048202192
        struct[0].Gy[8,31] = 0.0000962359705999109
        struct[0].Gy[9,9] = -1
        struct[0].Gy[9,16] = 0.00706562359705999
        struct[0].Gy[9,17] = 0.00273067829517978
        struct[0].Gy[9,18] = 0.00000962359705999132
        struct[0].Gy[9,19] = 0.000966678295179781
        struct[0].Gy[9,20] = 0.00000962359705999132
        struct[0].Gy[9,21] = 0.000966678295179781
        struct[0].Gy[9,22] = 0.00000962359705999127
        struct[0].Gy[9,23] = 0.000966678295179781
        struct[0].Gy[9,24] = 0.00706562359705999
        struct[0].Gy[9,25] = 0.00273067829517978
        struct[0].Gy[9,26] = 0.00000962359705999127
        struct[0].Gy[9,27] = 0.000966678295179781
        struct[0].Gy[9,28] = 0.00000962359705999127
        struct[0].Gy[9,29] = 0.000966678295179781
        struct[0].Gy[9,30] = -0.0000962359705999109
        struct[0].Gy[9,31] = 0.000333217048202192
        struct[0].Gy[10,10] = -1
        struct[0].Gy[10,16] = 0.000966678295179781
        struct[0].Gy[10,17] = -0.00000962359705999132
        struct[0].Gy[10,18] = 0.00273067829517978
        struct[0].Gy[10,19] = -0.00706562359705999
        struct[0].Gy[10,20] = 0.000966678295179781
        struct[0].Gy[10,21] = -0.00000962359705999132
        struct[0].Gy[10,22] = 0.000966678295179781
        struct[0].Gy[10,23] = -0.00000962359705999127
        struct[0].Gy[10,24] = 0.000966678295179781
        struct[0].Gy[10,25] = -0.00000962359705999127
        struct[0].Gy[10,26] = 0.00273067829517978
        struct[0].Gy[10,27] = -0.00706562359705999
        struct[0].Gy[10,28] = 0.000966678295179781
        struct[0].Gy[10,29] = -0.00000962359705999127
        struct[0].Gy[10,30] = 0.000333217048202192
        struct[0].Gy[10,31] = 0.0000962359705999109
        struct[0].Gy[11,11] = -1
        struct[0].Gy[11,16] = 0.00000962359705999132
        struct[0].Gy[11,17] = 0.000966678295179781
        struct[0].Gy[11,18] = 0.00706562359705999
        struct[0].Gy[11,19] = 0.00273067829517978
        struct[0].Gy[11,20] = 0.00000962359705999132
        struct[0].Gy[11,21] = 0.000966678295179781
        struct[0].Gy[11,22] = 0.00000962359705999127
        struct[0].Gy[11,23] = 0.000966678295179781
        struct[0].Gy[11,24] = 0.00000962359705999127
        struct[0].Gy[11,25] = 0.000966678295179781
        struct[0].Gy[11,26] = 0.00706562359705999
        struct[0].Gy[11,27] = 0.00273067829517978
        struct[0].Gy[11,28] = 0.00000962359705999127
        struct[0].Gy[11,29] = 0.000966678295179781
        struct[0].Gy[11,30] = -0.0000962359705999109
        struct[0].Gy[11,31] = 0.000333217048202192
        struct[0].Gy[12,12] = -1
        struct[0].Gy[12,16] = 0.000966678295179781
        struct[0].Gy[12,17] = -0.00000962359705999132
        struct[0].Gy[12,18] = 0.000966678295179781
        struct[0].Gy[12,19] = -0.00000962359705999132
        struct[0].Gy[12,20] = 0.00273067829517978
        struct[0].Gy[12,21] = -0.00706562359705999
        struct[0].Gy[12,22] = 0.000966678295179781
        struct[0].Gy[12,23] = -0.00000962359705999127
        struct[0].Gy[12,24] = 0.000966678295179781
        struct[0].Gy[12,25] = -0.00000962359705999127
        struct[0].Gy[12,26] = 0.000966678295179781
        struct[0].Gy[12,27] = -0.00000962359705999127
        struct[0].Gy[12,28] = 0.00273067829517978
        struct[0].Gy[12,29] = -0.00706562359705999
        struct[0].Gy[12,30] = 0.000333217048202192
        struct[0].Gy[12,31] = 0.0000962359705999109
        struct[0].Gy[13,13] = -1
        struct[0].Gy[13,16] = 0.00000962359705999132
        struct[0].Gy[13,17] = 0.000966678295179781
        struct[0].Gy[13,18] = 0.00000962359705999132
        struct[0].Gy[13,19] = 0.000966678295179781
        struct[0].Gy[13,20] = 0.00706562359705999
        struct[0].Gy[13,21] = 0.00273067829517978
        struct[0].Gy[13,22] = 0.00000962359705999127
        struct[0].Gy[13,23] = 0.000966678295179781
        struct[0].Gy[13,24] = 0.00000962359705999127
        struct[0].Gy[13,25] = 0.000966678295179781
        struct[0].Gy[13,26] = 0.00000962359705999127
        struct[0].Gy[13,27] = 0.000966678295179781
        struct[0].Gy[13,28] = 0.00706562359705999
        struct[0].Gy[13,29] = 0.00273067829517978
        struct[0].Gy[13,30] = -0.0000962359705999109
        struct[0].Gy[13,31] = 0.000333217048202192
        struct[0].Gy[14,14] = -1
        struct[0].Gy[14,16] = 0.000966678295179781
        struct[0].Gy[14,17] = -0.00000962359705999126
        struct[0].Gy[14,18] = 0.000966678295179781
        struct[0].Gy[14,19] = -0.00000962359705999126
        struct[0].Gy[14,20] = 0.000966678295179781
        struct[0].Gy[14,21] = -0.00000962359705999126
        struct[0].Gy[14,22] = 0.000966678295179781
        struct[0].Gy[14,23] = -0.00000962359705999120
        struct[0].Gy[14,24] = 0.000966678295179781
        struct[0].Gy[14,25] = -0.00000962359705999122
        struct[0].Gy[14,26] = 0.000966678295179781
        struct[0].Gy[14,27] = -0.00000962359705999122
        struct[0].Gy[14,28] = 0.000966678295179781
        struct[0].Gy[14,29] = -0.00000962359705999122
        struct[0].Gy[14,30] = 0.000333217048202192
        struct[0].Gy[14,31] = 0.0000962359705999110
        struct[0].Gy[15,15] = -1
        struct[0].Gy[15,16] = 0.00000962359705999126
        struct[0].Gy[15,17] = 0.000966678295179781
        struct[0].Gy[15,18] = 0.00000962359705999126
        struct[0].Gy[15,19] = 0.000966678295179781
        struct[0].Gy[15,20] = 0.00000962359705999126
        struct[0].Gy[15,21] = 0.000966678295179781
        struct[0].Gy[15,22] = 0.00000962359705999120
        struct[0].Gy[15,23] = 0.000966678295179781
        struct[0].Gy[15,24] = 0.00000962359705999122
        struct[0].Gy[15,25] = 0.000966678295179781
        struct[0].Gy[15,26] = 0.00000962359705999122
        struct[0].Gy[15,27] = 0.000966678295179781
        struct[0].Gy[15,28] = 0.00000962359705999122
        struct[0].Gy[15,29] = 0.000966678295179781
        struct[0].Gy[15,30] = -0.0000962359705999110
        struct[0].Gy[15,31] = 0.000333217048202192
        struct[0].Gy[16,8] = i_B2_a_r
        struct[0].Gy[16,9] = i_B2_a_i
        struct[0].Gy[16,14] = -i_B2_a_r
        struct[0].Gy[16,15] = -i_B2_a_i
        struct[0].Gy[16,16] = v_B2_a_r - v_B2_n_r
        struct[0].Gy[16,17] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[17,10] = i_B2_b_r
        struct[0].Gy[17,11] = i_B2_b_i
        struct[0].Gy[17,14] = -i_B2_b_r
        struct[0].Gy[17,15] = -i_B2_b_i
        struct[0].Gy[17,18] = v_B2_b_r - v_B2_n_r
        struct[0].Gy[17,19] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[18,12] = i_B2_c_r
        struct[0].Gy[18,13] = i_B2_c_i
        struct[0].Gy[18,14] = -i_B2_c_r
        struct[0].Gy[18,15] = -i_B2_c_i
        struct[0].Gy[18,20] = v_B2_c_r - v_B2_n_r
        struct[0].Gy[18,21] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[19,8] = -i_B2_a_i
        struct[0].Gy[19,9] = i_B2_a_r
        struct[0].Gy[19,14] = i_B2_a_i
        struct[0].Gy[19,15] = -i_B2_a_r
        struct[0].Gy[19,16] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[19,17] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy[20,10] = -i_B2_b_i
        struct[0].Gy[20,11] = i_B2_b_r
        struct[0].Gy[20,14] = i_B2_b_i
        struct[0].Gy[20,15] = -i_B2_b_r
        struct[0].Gy[20,18] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[20,19] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy[21,12] = -i_B2_c_i
        struct[0].Gy[21,13] = i_B2_c_r
        struct[0].Gy[21,14] = i_B2_c_i
        struct[0].Gy[21,15] = -i_B2_c_r
        struct[0].Gy[21,20] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[21,21] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy[22,16] = 1
        struct[0].Gy[22,18] = 1
        struct[0].Gy[22,20] = 1
        struct[0].Gy[22,22] = 1
        struct[0].Gy[23,17] = 1
        struct[0].Gy[23,19] = 1
        struct[0].Gy[23,21] = 1
        struct[0].Gy[23,23] = 1
        struct[0].Gy[24,0] = i_B3_a_r
        struct[0].Gy[24,1] = i_B3_a_i
        struct[0].Gy[24,6] = -i_B3_a_r
        struct[0].Gy[24,7] = -i_B3_a_i
        struct[0].Gy[24,24] = v_B3_a_r - v_B3_n_r
        struct[0].Gy[24,25] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[25,2] = i_B3_b_r
        struct[0].Gy[25,3] = i_B3_b_i
        struct[0].Gy[25,6] = -i_B3_b_r
        struct[0].Gy[25,7] = -i_B3_b_i
        struct[0].Gy[25,26] = v_B3_b_r - v_B3_n_r
        struct[0].Gy[25,27] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[26,4] = i_B3_c_r
        struct[0].Gy[26,5] = i_B3_c_i
        struct[0].Gy[26,6] = -i_B3_c_r
        struct[0].Gy[26,7] = -i_B3_c_i
        struct[0].Gy[26,28] = v_B3_c_r - v_B3_n_r
        struct[0].Gy[26,29] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[27,0] = -i_B3_a_i
        struct[0].Gy[27,1] = i_B3_a_r
        struct[0].Gy[27,6] = i_B3_a_i
        struct[0].Gy[27,7] = -i_B3_a_r
        struct[0].Gy[27,24] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[27,25] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy[28,2] = -i_B3_b_i
        struct[0].Gy[28,3] = i_B3_b_r
        struct[0].Gy[28,6] = i_B3_b_i
        struct[0].Gy[28,7] = -i_B3_b_r
        struct[0].Gy[28,26] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[28,27] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy[29,4] = -i_B3_c_i
        struct[0].Gy[29,5] = i_B3_c_r
        struct[0].Gy[29,6] = i_B3_c_i
        struct[0].Gy[29,7] = -i_B3_c_r
        struct[0].Gy[29,28] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[29,29] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy[30,24] = 1
        struct[0].Gy[30,26] = 1
        struct[0].Gy[30,28] = 1
        struct[0].Gy[30,30] = 1
        struct[0].Gy[31,25] = 1
        struct[0].Gy[31,27] = 1
        struct[0].Gy[31,29] = 1
        struct[0].Gy[31,31] = 1

    if mode > 12:

        struct[0].Fu[0,22] = 1

        struct[0].Gu[16,10] = 1
        struct[0].Gu[17,12] = 1
        struct[0].Gu[18,14] = 1
        struct[0].Gu[19,11] = 1
        struct[0].Gu[20,13] = 1
        struct[0].Gu[21,15] = 1
        struct[0].Gu[24,16] = 1
        struct[0].Gu[25,18] = 1
        struct[0].Gu[26,20] = 1
        struct[0].Gu[27,17] = 1
        struct[0].Gu[28,19] = 1
        struct[0].Gu[29,21] = 1

        struct[0].Hx[0,0] = 1





@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    a = struct[0].a
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
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
    v_B3_a_r = struct[0].y_ini[0,0]
    v_B3_a_i = struct[0].y_ini[1,0]
    v_B3_b_r = struct[0].y_ini[2,0]
    v_B3_b_i = struct[0].y_ini[3,0]
    v_B3_c_r = struct[0].y_ini[4,0]
    v_B3_c_i = struct[0].y_ini[5,0]
    v_B3_n_r = struct[0].y_ini[6,0]
    v_B3_n_i = struct[0].y_ini[7,0]
    v_B2_a_r = struct[0].y_ini[8,0]
    v_B2_a_i = struct[0].y_ini[9,0]
    v_B2_b_r = struct[0].y_ini[10,0]
    v_B2_b_i = struct[0].y_ini[11,0]
    v_B2_c_r = struct[0].y_ini[12,0]
    v_B2_c_i = struct[0].y_ini[13,0]
    v_B2_n_r = struct[0].y_ini[14,0]
    v_B2_n_i = struct[0].y_ini[15,0]
    i_B2_a_r = struct[0].y_ini[16,0]
    i_B2_a_i = struct[0].y_ini[17,0]
    i_B2_b_r = struct[0].y_ini[18,0]
    i_B2_b_i = struct[0].y_ini[19,0]
    i_B2_c_r = struct[0].y_ini[20,0]
    i_B2_c_i = struct[0].y_ini[21,0]
    i_B2_n_r = struct[0].y_ini[22,0]
    i_B2_n_i = struct[0].y_ini[23,0]
    i_B3_a_r = struct[0].y_ini[24,0]
    i_B3_a_i = struct[0].y_ini[25,0]
    i_B3_b_r = struct[0].y_ini[26,0]
    i_B3_b_i = struct[0].y_ini[27,0]
    i_B3_c_r = struct[0].y_ini[28,0]
    i_B3_c_i = struct[0].y_ini[29,0]
    i_B3_n_r = struct[0].y_ini[30,0]
    i_B3_n_i = struct[0].y_ini[31,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -0.00706562359705999*i_B2_a_i + 0.00273067829517978*i_B2_a_r - 9.62359705999139e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 9.62359705999139e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999124e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 0.01506562359706*i_B3_a_i + 0.0194306782951798*i_B3_a_r - 9.62359705999132e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 9.62359705999132e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r + 6.50521303491303e-19*v_B1_a_i + 0.0121243556529821*v_B1_a_r - 6.50521303491303e-19*v_B1_c_i - 0.0121243556529821*v_B1_c_r - v_B3_a_r
        struct[0].g[1,0] = 0.00273067829517978*i_B2_a_i + 0.00706562359705999*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999139e-6*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999139e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999124e-6*i_B2_n_r + 0.0194306782951798*i_B3_a_i + 0.01506562359706*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999132e-6*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999132e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r + 0.0121243556529821*v_B1_a_i - 6.50521303491303e-19*v_B1_a_r - 0.0121243556529821*v_B1_c_i + 6.50521303491303e-19*v_B1_c_r - v_B3_a_i
        struct[0].g[2,0] = -9.62359705999139e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 0.00706562359705999*i_B2_b_i + 0.00273067829517978*i_B2_b_r - 9.62359705999139e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999124e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999132e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 0.01506562359706*i_B3_b_i + 0.0194306782951798*i_B3_b_r - 9.62359705999132e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - 6.50521303491303e-19*v_B1_a_i - 0.0121243556529821*v_B1_a_r + 6.50521303491303e-19*v_B1_b_i + 0.0121243556529821*v_B1_b_r - v_B3_b_r
        struct[0].g[3,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999139e-6*i_B2_a_r + 0.00273067829517978*i_B2_b_i + 0.00706562359705999*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999139e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999124e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999132e-6*i_B3_a_r + 0.0194306782951798*i_B3_b_i + 0.01506562359706*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999132e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_a_i + 6.50521303491303e-19*v_B1_a_r + 0.0121243556529821*v_B1_b_i - 6.50521303491303e-19*v_B1_b_r - v_B3_b_i
        struct[0].g[4,0] = -9.62359705999139e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 9.62359705999139e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 0.00706562359705999*i_B2_c_i + 0.00273067829517978*i_B2_c_r - 9.62359705999124e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999132e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 9.62359705999132e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 0.01506562359706*i_B3_c_i + 0.0194306782951798*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - 6.50521303491303e-19*v_B1_b_i - 0.0121243556529821*v_B1_b_r + 6.50521303491303e-19*v_B1_c_i + 0.0121243556529821*v_B1_c_r - v_B3_c_r
        struct[0].g[5,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999139e-6*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999139e-6*i_B2_b_r + 0.00273067829517978*i_B2_c_i + 0.00706562359705999*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999124e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999132e-6*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999132e-6*i_B3_b_r + 0.0194306782951798*i_B3_c_i + 0.01506562359706*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_b_i + 6.50521303491303e-19*v_B1_b_r + 0.0121243556529821*v_B1_c_i - 6.50521303491303e-19*v_B1_c_r - v_B3_c_i
        struct[0].g[6,0] = 9.62359705999109e-5*i_B2_a_i + 0.000333217048202192*i_B2_a_r + 9.62359705999109e-5*i_B2_b_i + 0.000333217048202192*i_B2_b_r + 9.62359705999109e-5*i_B2_c_i + 0.000333217048202192*i_B2_c_r + 9.6235970599911e-5*i_B2_n_i + 0.000333217048202192*i_B2_n_r + 9.62359705999109e-5*i_B3_a_i + 0.000333217048202192*i_B3_a_r + 9.62359705999109e-5*i_B3_b_i + 0.000333217048202192*i_B3_b_r + 9.62359705999109e-5*i_B3_c_i + 0.000333217048202192*i_B3_c_r - 0.00096235970599911*i_B3_n_i + 0.00666782951797808*i_B3_n_r - v_B3_n_r
        struct[0].g[7,0] = 0.000333217048202192*i_B2_a_i - 9.62359705999109e-5*i_B2_a_r + 0.000333217048202192*i_B2_b_i - 9.62359705999109e-5*i_B2_b_r + 0.000333217048202192*i_B2_c_i - 9.62359705999109e-5*i_B2_c_r + 0.000333217048202192*i_B2_n_i - 9.6235970599911e-5*i_B2_n_r + 0.000333217048202192*i_B3_a_i - 9.62359705999109e-5*i_B3_a_r + 0.000333217048202192*i_B3_b_i - 9.62359705999109e-5*i_B3_b_r + 0.000333217048202192*i_B3_c_i - 9.62359705999109e-5*i_B3_c_r + 0.00666782951797808*i_B3_n_i + 0.00096235970599911*i_B3_n_r - v_B3_n_i
        struct[0].g[8,0] = -0.00706562359705999*i_B2_a_i + 0.00273067829517978*i_B2_a_r - 9.62359705999132e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 9.62359705999132e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999127e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 0.00706562359705999*i_B3_a_i + 0.00273067829517978*i_B3_a_r - 9.62359705999127e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 9.62359705999127e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - 1.0842021724855e-18*v_B1_a_i + 0.0121243556529821*v_B1_a_r + 1.0842021724855e-18*v_B1_c_i - 0.0121243556529821*v_B1_c_r - v_B2_a_r
        struct[0].g[9,0] = 0.00273067829517978*i_B2_a_i + 0.00706562359705999*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999132e-6*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999132e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999127e-6*i_B2_n_r + 0.00273067829517978*i_B3_a_i + 0.00706562359705999*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999127e-6*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999127e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r + 0.0121243556529821*v_B1_a_i + 1.0842021724855e-18*v_B1_a_r - 0.0121243556529821*v_B1_c_i - 1.0842021724855e-18*v_B1_c_r - v_B2_a_i
        struct[0].g[10,0] = -9.62359705999132e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 0.00706562359705999*i_B2_b_i + 0.00273067829517978*i_B2_b_r - 9.62359705999132e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.62359705999127e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999127e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 0.00706562359705999*i_B3_b_i + 0.00273067829517978*i_B3_b_r - 9.62359705999127e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r + 1.0842021724855e-18*v_B1_a_i - 0.0121243556529821*v_B1_a_r - 1.0842021724855e-18*v_B1_b_i + 0.0121243556529821*v_B1_b_r - v_B2_b_r
        struct[0].g[11,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999132e-6*i_B2_a_r + 0.00273067829517978*i_B2_b_i + 0.00706562359705999*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999132e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999127e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999127e-6*i_B3_a_r + 0.00273067829517978*i_B3_b_i + 0.00706562359705999*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999127e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_a_i - 1.0842021724855e-18*v_B1_a_r + 0.0121243556529821*v_B1_b_i + 1.0842021724855e-18*v_B1_b_r - v_B2_b_i
        struct[0].g[12,0] = -9.62359705999132e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 9.62359705999132e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 0.00706562359705999*i_B2_c_i + 0.00273067829517978*i_B2_c_r - 9.62359705999127e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999127e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 9.62359705999127e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 0.00706562359705999*i_B3_c_i + 0.00273067829517978*i_B3_c_r + 9.62359705999109e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r + 1.0842021724855e-18*v_B1_b_i - 0.0121243556529821*v_B1_b_r - 1.0842021724855e-18*v_B1_c_i + 0.0121243556529821*v_B1_c_r - v_B2_c_r
        struct[0].g[13,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999132e-6*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999132e-6*i_B2_b_r + 0.00273067829517978*i_B2_c_i + 0.00706562359705999*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.62359705999127e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999127e-6*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999127e-6*i_B3_b_r + 0.00273067829517978*i_B3_c_i + 0.00706562359705999*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.62359705999109e-5*i_B3_n_r - 0.0121243556529821*v_B1_b_i - 1.0842021724855e-18*v_B1_b_r + 0.0121243556529821*v_B1_c_i + 1.0842021724855e-18*v_B1_c_r - v_B2_c_i
        struct[0].g[14,0] = -9.62359705999126e-6*i_B2_a_i + 0.000966678295179781*i_B2_a_r - 9.62359705999126e-6*i_B2_b_i + 0.000966678295179781*i_B2_b_r - 9.62359705999126e-6*i_B2_c_i + 0.000966678295179781*i_B2_c_r - 9.6235970599912e-6*i_B2_n_i + 0.000966678295179781*i_B2_n_r - 9.62359705999122e-6*i_B3_a_i + 0.000966678295179781*i_B3_a_r - 9.62359705999122e-6*i_B3_b_i + 0.000966678295179781*i_B3_b_r - 9.62359705999122e-6*i_B3_c_i + 0.000966678295179781*i_B3_c_r + 9.6235970599911e-5*i_B3_n_i + 0.000333217048202192*i_B3_n_r - v_B2_n_r
        struct[0].g[15,0] = 0.000966678295179781*i_B2_a_i + 9.62359705999126e-6*i_B2_a_r + 0.000966678295179781*i_B2_b_i + 9.62359705999126e-6*i_B2_b_r + 0.000966678295179781*i_B2_c_i + 9.62359705999126e-6*i_B2_c_r + 0.000966678295179781*i_B2_n_i + 9.6235970599912e-6*i_B2_n_r + 0.000966678295179781*i_B3_a_i + 9.62359705999122e-6*i_B3_a_r + 0.000966678295179781*i_B3_b_i + 9.62359705999122e-6*i_B3_b_r + 0.000966678295179781*i_B3_c_i + 9.62359705999122e-6*i_B3_c_r + 0.000333217048202192*i_B3_n_i - 9.6235970599911e-5*i_B3_n_r - v_B2_n_i
        struct[0].g[16,0] = i_B2_a_i*v_B2_a_i - i_B2_a_i*v_B2_n_i + i_B2_a_r*v_B2_a_r - i_B2_a_r*v_B2_n_r + p_B2_a
        struct[0].g[17,0] = i_B2_b_i*v_B2_b_i - i_B2_b_i*v_B2_n_i + i_B2_b_r*v_B2_b_r - i_B2_b_r*v_B2_n_r + p_B2_b
        struct[0].g[18,0] = i_B2_c_i*v_B2_c_i - i_B2_c_i*v_B2_n_i + i_B2_c_r*v_B2_c_r - i_B2_c_r*v_B2_n_r + p_B2_c
        struct[0].g[19,0] = -i_B2_a_i*v_B2_a_r + i_B2_a_i*v_B2_n_r + i_B2_a_r*v_B2_a_i - i_B2_a_r*v_B2_n_i + q_B2_a
        struct[0].g[20,0] = -i_B2_b_i*v_B2_b_r + i_B2_b_i*v_B2_n_r + i_B2_b_r*v_B2_b_i - i_B2_b_r*v_B2_n_i + q_B2_b
        struct[0].g[21,0] = -i_B2_c_i*v_B2_c_r + i_B2_c_i*v_B2_n_r + i_B2_c_r*v_B2_c_i - i_B2_c_r*v_B2_n_i + q_B2_c
        struct[0].g[22,0] = i_B2_a_r + i_B2_b_r + i_B2_c_r + i_B2_n_r
        struct[0].g[23,0] = i_B2_a_i + i_B2_b_i + i_B2_c_i + i_B2_n_i
        struct[0].g[24,0] = i_B3_a_i*v_B3_a_i - i_B3_a_i*v_B3_n_i + i_B3_a_r*v_B3_a_r - i_B3_a_r*v_B3_n_r + p_B3_a
        struct[0].g[25,0] = i_B3_b_i*v_B3_b_i - i_B3_b_i*v_B3_n_i + i_B3_b_r*v_B3_b_r - i_B3_b_r*v_B3_n_r + p_B3_b
        struct[0].g[26,0] = i_B3_c_i*v_B3_c_i - i_B3_c_i*v_B3_n_i + i_B3_c_r*v_B3_c_r - i_B3_c_r*v_B3_n_r + p_B3_c
        struct[0].g[27,0] = -i_B3_a_i*v_B3_a_r + i_B3_a_i*v_B3_n_r + i_B3_a_r*v_B3_a_i - i_B3_a_r*v_B3_n_i + q_B3_a
        struct[0].g[28,0] = -i_B3_b_i*v_B3_b_r + i_B3_b_i*v_B3_n_r + i_B3_b_r*v_B3_b_i - i_B3_b_r*v_B3_n_i + q_B3_b
        struct[0].g[29,0] = -i_B3_c_i*v_B3_c_r + i_B3_c_i*v_B3_n_r + i_B3_c_r*v_B3_c_i - i_B3_c_r*v_B3_n_i + q_B3_c
        struct[0].g[30,0] = i_B3_a_r + i_B3_b_r + i_B3_c_r + i_B3_n_r
        struct[0].g[31,0] = i_B3_a_i + i_B3_b_i + i_B3_c_i + i_B3_n_i
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = x_dummy
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -1

    if mode == 11:



        struct[0].Gy_ini[0,0] = -1
        struct[0].Gy_ini[0,16] = 0.00273067829517978
        struct[0].Gy_ini[0,17] = -0.00706562359705999
        struct[0].Gy_ini[0,18] = 0.000966678295179781
        struct[0].Gy_ini[0,19] = -0.00000962359705999139
        struct[0].Gy_ini[0,20] = 0.000966678295179781
        struct[0].Gy_ini[0,21] = -0.00000962359705999139
        struct[0].Gy_ini[0,22] = 0.000966678295179781
        struct[0].Gy_ini[0,23] = -0.00000962359705999124
        struct[0].Gy_ini[0,24] = 0.0194306782951798
        struct[0].Gy_ini[0,25] = -0.0150656235970600
        struct[0].Gy_ini[0,26] = 0.000966678295179781
        struct[0].Gy_ini[0,27] = -0.00000962359705999132
        struct[0].Gy_ini[0,28] = 0.000966678295179781
        struct[0].Gy_ini[0,29] = -0.00000962359705999132
        struct[0].Gy_ini[0,30] = 0.000333217048202192
        struct[0].Gy_ini[0,31] = 0.0000962359705999109
        struct[0].Gy_ini[1,1] = -1
        struct[0].Gy_ini[1,16] = 0.00706562359705999
        struct[0].Gy_ini[1,17] = 0.00273067829517978
        struct[0].Gy_ini[1,18] = 0.00000962359705999139
        struct[0].Gy_ini[1,19] = 0.000966678295179781
        struct[0].Gy_ini[1,20] = 0.00000962359705999139
        struct[0].Gy_ini[1,21] = 0.000966678295179781
        struct[0].Gy_ini[1,22] = 0.00000962359705999124
        struct[0].Gy_ini[1,23] = 0.000966678295179781
        struct[0].Gy_ini[1,24] = 0.0150656235970600
        struct[0].Gy_ini[1,25] = 0.0194306782951798
        struct[0].Gy_ini[1,26] = 0.00000962359705999132
        struct[0].Gy_ini[1,27] = 0.000966678295179781
        struct[0].Gy_ini[1,28] = 0.00000962359705999132
        struct[0].Gy_ini[1,29] = 0.000966678295179781
        struct[0].Gy_ini[1,30] = -0.0000962359705999109
        struct[0].Gy_ini[1,31] = 0.000333217048202192
        struct[0].Gy_ini[2,2] = -1
        struct[0].Gy_ini[2,16] = 0.000966678295179781
        struct[0].Gy_ini[2,17] = -0.00000962359705999139
        struct[0].Gy_ini[2,18] = 0.00273067829517978
        struct[0].Gy_ini[2,19] = -0.00706562359705999
        struct[0].Gy_ini[2,20] = 0.000966678295179781
        struct[0].Gy_ini[2,21] = -0.00000962359705999139
        struct[0].Gy_ini[2,22] = 0.000966678295179781
        struct[0].Gy_ini[2,23] = -0.00000962359705999124
        struct[0].Gy_ini[2,24] = 0.000966678295179781
        struct[0].Gy_ini[2,25] = -0.00000962359705999132
        struct[0].Gy_ini[2,26] = 0.0194306782951798
        struct[0].Gy_ini[2,27] = -0.0150656235970600
        struct[0].Gy_ini[2,28] = 0.000966678295179781
        struct[0].Gy_ini[2,29] = -0.00000962359705999132
        struct[0].Gy_ini[2,30] = 0.000333217048202192
        struct[0].Gy_ini[2,31] = 0.0000962359705999109
        struct[0].Gy_ini[3,3] = -1
        struct[0].Gy_ini[3,16] = 0.00000962359705999139
        struct[0].Gy_ini[3,17] = 0.000966678295179781
        struct[0].Gy_ini[3,18] = 0.00706562359705999
        struct[0].Gy_ini[3,19] = 0.00273067829517978
        struct[0].Gy_ini[3,20] = 0.00000962359705999139
        struct[0].Gy_ini[3,21] = 0.000966678295179781
        struct[0].Gy_ini[3,22] = 0.00000962359705999124
        struct[0].Gy_ini[3,23] = 0.000966678295179781
        struct[0].Gy_ini[3,24] = 0.00000962359705999132
        struct[0].Gy_ini[3,25] = 0.000966678295179781
        struct[0].Gy_ini[3,26] = 0.0150656235970600
        struct[0].Gy_ini[3,27] = 0.0194306782951798
        struct[0].Gy_ini[3,28] = 0.00000962359705999132
        struct[0].Gy_ini[3,29] = 0.000966678295179781
        struct[0].Gy_ini[3,30] = -0.0000962359705999109
        struct[0].Gy_ini[3,31] = 0.000333217048202192
        struct[0].Gy_ini[4,4] = -1
        struct[0].Gy_ini[4,16] = 0.000966678295179781
        struct[0].Gy_ini[4,17] = -0.00000962359705999139
        struct[0].Gy_ini[4,18] = 0.000966678295179781
        struct[0].Gy_ini[4,19] = -0.00000962359705999139
        struct[0].Gy_ini[4,20] = 0.00273067829517978
        struct[0].Gy_ini[4,21] = -0.00706562359705999
        struct[0].Gy_ini[4,22] = 0.000966678295179781
        struct[0].Gy_ini[4,23] = -0.00000962359705999124
        struct[0].Gy_ini[4,24] = 0.000966678295179781
        struct[0].Gy_ini[4,25] = -0.00000962359705999132
        struct[0].Gy_ini[4,26] = 0.000966678295179781
        struct[0].Gy_ini[4,27] = -0.00000962359705999132
        struct[0].Gy_ini[4,28] = 0.0194306782951798
        struct[0].Gy_ini[4,29] = -0.0150656235970600
        struct[0].Gy_ini[4,30] = 0.000333217048202192
        struct[0].Gy_ini[4,31] = 0.0000962359705999109
        struct[0].Gy_ini[5,5] = -1
        struct[0].Gy_ini[5,16] = 0.00000962359705999139
        struct[0].Gy_ini[5,17] = 0.000966678295179781
        struct[0].Gy_ini[5,18] = 0.00000962359705999139
        struct[0].Gy_ini[5,19] = 0.000966678295179781
        struct[0].Gy_ini[5,20] = 0.00706562359705999
        struct[0].Gy_ini[5,21] = 0.00273067829517978
        struct[0].Gy_ini[5,22] = 0.00000962359705999124
        struct[0].Gy_ini[5,23] = 0.000966678295179781
        struct[0].Gy_ini[5,24] = 0.00000962359705999132
        struct[0].Gy_ini[5,25] = 0.000966678295179781
        struct[0].Gy_ini[5,26] = 0.00000962359705999132
        struct[0].Gy_ini[5,27] = 0.000966678295179781
        struct[0].Gy_ini[5,28] = 0.0150656235970600
        struct[0].Gy_ini[5,29] = 0.0194306782951798
        struct[0].Gy_ini[5,30] = -0.0000962359705999109
        struct[0].Gy_ini[5,31] = 0.000333217048202192
        struct[0].Gy_ini[6,6] = -1
        struct[0].Gy_ini[6,16] = 0.000333217048202192
        struct[0].Gy_ini[6,17] = 0.0000962359705999109
        struct[0].Gy_ini[6,18] = 0.000333217048202192
        struct[0].Gy_ini[6,19] = 0.0000962359705999109
        struct[0].Gy_ini[6,20] = 0.000333217048202192
        struct[0].Gy_ini[6,21] = 0.0000962359705999109
        struct[0].Gy_ini[6,22] = 0.000333217048202192
        struct[0].Gy_ini[6,23] = 0.0000962359705999110
        struct[0].Gy_ini[6,24] = 0.000333217048202192
        struct[0].Gy_ini[6,25] = 0.0000962359705999109
        struct[0].Gy_ini[6,26] = 0.000333217048202192
        struct[0].Gy_ini[6,27] = 0.0000962359705999109
        struct[0].Gy_ini[6,28] = 0.000333217048202192
        struct[0].Gy_ini[6,29] = 0.0000962359705999109
        struct[0].Gy_ini[6,30] = 0.00666782951797808
        struct[0].Gy_ini[6,31] = -0.000962359705999110
        struct[0].Gy_ini[7,7] = -1
        struct[0].Gy_ini[7,16] = -0.0000962359705999109
        struct[0].Gy_ini[7,17] = 0.000333217048202192
        struct[0].Gy_ini[7,18] = -0.0000962359705999109
        struct[0].Gy_ini[7,19] = 0.000333217048202192
        struct[0].Gy_ini[7,20] = -0.0000962359705999109
        struct[0].Gy_ini[7,21] = 0.000333217048202192
        struct[0].Gy_ini[7,22] = -0.0000962359705999110
        struct[0].Gy_ini[7,23] = 0.000333217048202192
        struct[0].Gy_ini[7,24] = -0.0000962359705999109
        struct[0].Gy_ini[7,25] = 0.000333217048202192
        struct[0].Gy_ini[7,26] = -0.0000962359705999109
        struct[0].Gy_ini[7,27] = 0.000333217048202192
        struct[0].Gy_ini[7,28] = -0.0000962359705999109
        struct[0].Gy_ini[7,29] = 0.000333217048202192
        struct[0].Gy_ini[7,30] = 0.000962359705999110
        struct[0].Gy_ini[7,31] = 0.00666782951797808
        struct[0].Gy_ini[8,8] = -1
        struct[0].Gy_ini[8,16] = 0.00273067829517978
        struct[0].Gy_ini[8,17] = -0.00706562359705999
        struct[0].Gy_ini[8,18] = 0.000966678295179781
        struct[0].Gy_ini[8,19] = -0.00000962359705999132
        struct[0].Gy_ini[8,20] = 0.000966678295179781
        struct[0].Gy_ini[8,21] = -0.00000962359705999132
        struct[0].Gy_ini[8,22] = 0.000966678295179781
        struct[0].Gy_ini[8,23] = -0.00000962359705999127
        struct[0].Gy_ini[8,24] = 0.00273067829517978
        struct[0].Gy_ini[8,25] = -0.00706562359705999
        struct[0].Gy_ini[8,26] = 0.000966678295179781
        struct[0].Gy_ini[8,27] = -0.00000962359705999127
        struct[0].Gy_ini[8,28] = 0.000966678295179781
        struct[0].Gy_ini[8,29] = -0.00000962359705999127
        struct[0].Gy_ini[8,30] = 0.000333217048202192
        struct[0].Gy_ini[8,31] = 0.0000962359705999109
        struct[0].Gy_ini[9,9] = -1
        struct[0].Gy_ini[9,16] = 0.00706562359705999
        struct[0].Gy_ini[9,17] = 0.00273067829517978
        struct[0].Gy_ini[9,18] = 0.00000962359705999132
        struct[0].Gy_ini[9,19] = 0.000966678295179781
        struct[0].Gy_ini[9,20] = 0.00000962359705999132
        struct[0].Gy_ini[9,21] = 0.000966678295179781
        struct[0].Gy_ini[9,22] = 0.00000962359705999127
        struct[0].Gy_ini[9,23] = 0.000966678295179781
        struct[0].Gy_ini[9,24] = 0.00706562359705999
        struct[0].Gy_ini[9,25] = 0.00273067829517978
        struct[0].Gy_ini[9,26] = 0.00000962359705999127
        struct[0].Gy_ini[9,27] = 0.000966678295179781
        struct[0].Gy_ini[9,28] = 0.00000962359705999127
        struct[0].Gy_ini[9,29] = 0.000966678295179781
        struct[0].Gy_ini[9,30] = -0.0000962359705999109
        struct[0].Gy_ini[9,31] = 0.000333217048202192
        struct[0].Gy_ini[10,10] = -1
        struct[0].Gy_ini[10,16] = 0.000966678295179781
        struct[0].Gy_ini[10,17] = -0.00000962359705999132
        struct[0].Gy_ini[10,18] = 0.00273067829517978
        struct[0].Gy_ini[10,19] = -0.00706562359705999
        struct[0].Gy_ini[10,20] = 0.000966678295179781
        struct[0].Gy_ini[10,21] = -0.00000962359705999132
        struct[0].Gy_ini[10,22] = 0.000966678295179781
        struct[0].Gy_ini[10,23] = -0.00000962359705999127
        struct[0].Gy_ini[10,24] = 0.000966678295179781
        struct[0].Gy_ini[10,25] = -0.00000962359705999127
        struct[0].Gy_ini[10,26] = 0.00273067829517978
        struct[0].Gy_ini[10,27] = -0.00706562359705999
        struct[0].Gy_ini[10,28] = 0.000966678295179781
        struct[0].Gy_ini[10,29] = -0.00000962359705999127
        struct[0].Gy_ini[10,30] = 0.000333217048202192
        struct[0].Gy_ini[10,31] = 0.0000962359705999109
        struct[0].Gy_ini[11,11] = -1
        struct[0].Gy_ini[11,16] = 0.00000962359705999132
        struct[0].Gy_ini[11,17] = 0.000966678295179781
        struct[0].Gy_ini[11,18] = 0.00706562359705999
        struct[0].Gy_ini[11,19] = 0.00273067829517978
        struct[0].Gy_ini[11,20] = 0.00000962359705999132
        struct[0].Gy_ini[11,21] = 0.000966678295179781
        struct[0].Gy_ini[11,22] = 0.00000962359705999127
        struct[0].Gy_ini[11,23] = 0.000966678295179781
        struct[0].Gy_ini[11,24] = 0.00000962359705999127
        struct[0].Gy_ini[11,25] = 0.000966678295179781
        struct[0].Gy_ini[11,26] = 0.00706562359705999
        struct[0].Gy_ini[11,27] = 0.00273067829517978
        struct[0].Gy_ini[11,28] = 0.00000962359705999127
        struct[0].Gy_ini[11,29] = 0.000966678295179781
        struct[0].Gy_ini[11,30] = -0.0000962359705999109
        struct[0].Gy_ini[11,31] = 0.000333217048202192
        struct[0].Gy_ini[12,12] = -1
        struct[0].Gy_ini[12,16] = 0.000966678295179781
        struct[0].Gy_ini[12,17] = -0.00000962359705999132
        struct[0].Gy_ini[12,18] = 0.000966678295179781
        struct[0].Gy_ini[12,19] = -0.00000962359705999132
        struct[0].Gy_ini[12,20] = 0.00273067829517978
        struct[0].Gy_ini[12,21] = -0.00706562359705999
        struct[0].Gy_ini[12,22] = 0.000966678295179781
        struct[0].Gy_ini[12,23] = -0.00000962359705999127
        struct[0].Gy_ini[12,24] = 0.000966678295179781
        struct[0].Gy_ini[12,25] = -0.00000962359705999127
        struct[0].Gy_ini[12,26] = 0.000966678295179781
        struct[0].Gy_ini[12,27] = -0.00000962359705999127
        struct[0].Gy_ini[12,28] = 0.00273067829517978
        struct[0].Gy_ini[12,29] = -0.00706562359705999
        struct[0].Gy_ini[12,30] = 0.000333217048202192
        struct[0].Gy_ini[12,31] = 0.0000962359705999109
        struct[0].Gy_ini[13,13] = -1
        struct[0].Gy_ini[13,16] = 0.00000962359705999132
        struct[0].Gy_ini[13,17] = 0.000966678295179781
        struct[0].Gy_ini[13,18] = 0.00000962359705999132
        struct[0].Gy_ini[13,19] = 0.000966678295179781
        struct[0].Gy_ini[13,20] = 0.00706562359705999
        struct[0].Gy_ini[13,21] = 0.00273067829517978
        struct[0].Gy_ini[13,22] = 0.00000962359705999127
        struct[0].Gy_ini[13,23] = 0.000966678295179781
        struct[0].Gy_ini[13,24] = 0.00000962359705999127
        struct[0].Gy_ini[13,25] = 0.000966678295179781
        struct[0].Gy_ini[13,26] = 0.00000962359705999127
        struct[0].Gy_ini[13,27] = 0.000966678295179781
        struct[0].Gy_ini[13,28] = 0.00706562359705999
        struct[0].Gy_ini[13,29] = 0.00273067829517978
        struct[0].Gy_ini[13,30] = -0.0000962359705999109
        struct[0].Gy_ini[13,31] = 0.000333217048202192
        struct[0].Gy_ini[14,14] = -1
        struct[0].Gy_ini[14,16] = 0.000966678295179781
        struct[0].Gy_ini[14,17] = -0.00000962359705999126
        struct[0].Gy_ini[14,18] = 0.000966678295179781
        struct[0].Gy_ini[14,19] = -0.00000962359705999126
        struct[0].Gy_ini[14,20] = 0.000966678295179781
        struct[0].Gy_ini[14,21] = -0.00000962359705999126
        struct[0].Gy_ini[14,22] = 0.000966678295179781
        struct[0].Gy_ini[14,23] = -0.00000962359705999120
        struct[0].Gy_ini[14,24] = 0.000966678295179781
        struct[0].Gy_ini[14,25] = -0.00000962359705999122
        struct[0].Gy_ini[14,26] = 0.000966678295179781
        struct[0].Gy_ini[14,27] = -0.00000962359705999122
        struct[0].Gy_ini[14,28] = 0.000966678295179781
        struct[0].Gy_ini[14,29] = -0.00000962359705999122
        struct[0].Gy_ini[14,30] = 0.000333217048202192
        struct[0].Gy_ini[14,31] = 0.0000962359705999110
        struct[0].Gy_ini[15,15] = -1
        struct[0].Gy_ini[15,16] = 0.00000962359705999126
        struct[0].Gy_ini[15,17] = 0.000966678295179781
        struct[0].Gy_ini[15,18] = 0.00000962359705999126
        struct[0].Gy_ini[15,19] = 0.000966678295179781
        struct[0].Gy_ini[15,20] = 0.00000962359705999126
        struct[0].Gy_ini[15,21] = 0.000966678295179781
        struct[0].Gy_ini[15,22] = 0.00000962359705999120
        struct[0].Gy_ini[15,23] = 0.000966678295179781
        struct[0].Gy_ini[15,24] = 0.00000962359705999122
        struct[0].Gy_ini[15,25] = 0.000966678295179781
        struct[0].Gy_ini[15,26] = 0.00000962359705999122
        struct[0].Gy_ini[15,27] = 0.000966678295179781
        struct[0].Gy_ini[15,28] = 0.00000962359705999122
        struct[0].Gy_ini[15,29] = 0.000966678295179781
        struct[0].Gy_ini[15,30] = -0.0000962359705999110
        struct[0].Gy_ini[15,31] = 0.000333217048202192
        struct[0].Gy_ini[16,8] = i_B2_a_r
        struct[0].Gy_ini[16,9] = i_B2_a_i
        struct[0].Gy_ini[16,14] = -i_B2_a_r
        struct[0].Gy_ini[16,15] = -i_B2_a_i
        struct[0].Gy_ini[16,16] = v_B2_a_r - v_B2_n_r
        struct[0].Gy_ini[16,17] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[17,10] = i_B2_b_r
        struct[0].Gy_ini[17,11] = i_B2_b_i
        struct[0].Gy_ini[17,14] = -i_B2_b_r
        struct[0].Gy_ini[17,15] = -i_B2_b_i
        struct[0].Gy_ini[17,18] = v_B2_b_r - v_B2_n_r
        struct[0].Gy_ini[17,19] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[18,12] = i_B2_c_r
        struct[0].Gy_ini[18,13] = i_B2_c_i
        struct[0].Gy_ini[18,14] = -i_B2_c_r
        struct[0].Gy_ini[18,15] = -i_B2_c_i
        struct[0].Gy_ini[18,20] = v_B2_c_r - v_B2_n_r
        struct[0].Gy_ini[18,21] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[19,8] = -i_B2_a_i
        struct[0].Gy_ini[19,9] = i_B2_a_r
        struct[0].Gy_ini[19,14] = i_B2_a_i
        struct[0].Gy_ini[19,15] = -i_B2_a_r
        struct[0].Gy_ini[19,16] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[19,17] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy_ini[20,10] = -i_B2_b_i
        struct[0].Gy_ini[20,11] = i_B2_b_r
        struct[0].Gy_ini[20,14] = i_B2_b_i
        struct[0].Gy_ini[20,15] = -i_B2_b_r
        struct[0].Gy_ini[20,18] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[20,19] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy_ini[21,12] = -i_B2_c_i
        struct[0].Gy_ini[21,13] = i_B2_c_r
        struct[0].Gy_ini[21,14] = i_B2_c_i
        struct[0].Gy_ini[21,15] = -i_B2_c_r
        struct[0].Gy_ini[21,20] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[21,21] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy_ini[22,16] = 1
        struct[0].Gy_ini[22,18] = 1
        struct[0].Gy_ini[22,20] = 1
        struct[0].Gy_ini[22,22] = 1
        struct[0].Gy_ini[23,17] = 1
        struct[0].Gy_ini[23,19] = 1
        struct[0].Gy_ini[23,21] = 1
        struct[0].Gy_ini[23,23] = 1
        struct[0].Gy_ini[24,0] = i_B3_a_r
        struct[0].Gy_ini[24,1] = i_B3_a_i
        struct[0].Gy_ini[24,6] = -i_B3_a_r
        struct[0].Gy_ini[24,7] = -i_B3_a_i
        struct[0].Gy_ini[24,24] = v_B3_a_r - v_B3_n_r
        struct[0].Gy_ini[24,25] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[25,2] = i_B3_b_r
        struct[0].Gy_ini[25,3] = i_B3_b_i
        struct[0].Gy_ini[25,6] = -i_B3_b_r
        struct[0].Gy_ini[25,7] = -i_B3_b_i
        struct[0].Gy_ini[25,26] = v_B3_b_r - v_B3_n_r
        struct[0].Gy_ini[25,27] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[26,4] = i_B3_c_r
        struct[0].Gy_ini[26,5] = i_B3_c_i
        struct[0].Gy_ini[26,6] = -i_B3_c_r
        struct[0].Gy_ini[26,7] = -i_B3_c_i
        struct[0].Gy_ini[26,28] = v_B3_c_r - v_B3_n_r
        struct[0].Gy_ini[26,29] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[27,0] = -i_B3_a_i
        struct[0].Gy_ini[27,1] = i_B3_a_r
        struct[0].Gy_ini[27,6] = i_B3_a_i
        struct[0].Gy_ini[27,7] = -i_B3_a_r
        struct[0].Gy_ini[27,24] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[27,25] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy_ini[28,2] = -i_B3_b_i
        struct[0].Gy_ini[28,3] = i_B3_b_r
        struct[0].Gy_ini[28,6] = i_B3_b_i
        struct[0].Gy_ini[28,7] = -i_B3_b_r
        struct[0].Gy_ini[28,26] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[28,27] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy_ini[29,4] = -i_B3_c_i
        struct[0].Gy_ini[29,5] = i_B3_c_r
        struct[0].Gy_ini[29,6] = i_B3_c_i
        struct[0].Gy_ini[29,7] = -i_B3_c_r
        struct[0].Gy_ini[29,28] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[29,29] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy_ini[30,24] = 1
        struct[0].Gy_ini[30,26] = 1
        struct[0].Gy_ini[30,28] = 1
        struct[0].Gy_ini[30,30] = 1
        struct[0].Gy_ini[31,25] = 1
        struct[0].Gy_ini[31,27] = 1
        struct[0].Gy_ini[31,29] = 1
        struct[0].Gy_ini[31,31] = 1





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


