import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 


class cigre_eu_mv_dae_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 22
        self.N_y = 20 
        self.N_z = 1 
        self.N_store = 10000 
        self.params_list = ['R_0102', 'L_0102', 'C_0102', 'R_0203', 'L_0203', 'C_0203', 'R_0304', 'L_0304', 'C_0304', 'R_0308', 'L_0308', 'C_0308', 'R_0405', 'L_0405', 'C_0405', 'R_0506', 'L_0506', 'C_0506', 'R_0607', 'L_0607', 'C_0607', 'R_0708', 'L_0708', 'C_0708', 'R_0809', 'L_0809', 'C_0809', 'R_0910', 'L_0910', 'C_0910', 'R_1011', 'L_1011', 'C_1011', 'i_02_D', 'i_02_Q', 'i_03_D', 'i_03_Q', 'i_04_D', 'i_04_Q', 'i_05_D', 'i_05_Q', 'i_06_D', 'i_06_Q', 'i_07_D', 'i_07_Q', 'i_08_D', 'i_08_Q', 'i_09_D', 'i_09_Q', 'i_10_D', 'i_10_Q', 'i_11_D', 'i_11_Q', 'omega'] 
        self.params_values_list  = [1.41282, 0.0064270585739141526, 4.2631325817165496e-07, 2.21442, 0.01007361663003566, 6.681931209640832e-07, 0.30561, 0.001390250258896324, 9.22166976896133e-08, 0.6513, 0.0029628284205987236, 1.96527388518848e-07, 0.28056000000000003, 0.0012762953196425273, 8.465795197734993e-08, 0.77154, 0.0035098121290169496, 2.3280936793771228e-07, 0.12024, 0.0005469837084182259, 3.628197941886425e-08, 0.8366699999999999, 0.0038060949710768213, 2.5246210678959706e-07, 0.16032, 0.0007293116112243012, 4.837597255848566e-08, 0.38577, 0.0017549060645084748, 1.1640468396885614e-07, 0.16533, 0.0007521025990750605, 4.988772170093834e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 314.1592653589793] 
        self.inputs_ini_list = ['v_01_D', 'v_01_Q'] 
        self.inputs_ini_values_list  = [0.0, 16329.931618554521] 
        self.inputs_run_list = ['v_01_D', 'v_01_Q'] 
        self.inputs_run_values_list = [0.0, 16329.931618554521] 
        self.outputs_list = ['v_01_D'] 
        self.x_list = ['i_l_0102_D', 'i_l_0102_Q', 'i_l_0203_D', 'i_l_0203_Q', 'i_l_0304_D', 'i_l_0304_Q', 'i_l_0308_D', 'i_l_0308_Q', 'i_l_0405_D', 'i_l_0405_Q', 'i_l_0506_D', 'i_l_0506_Q', 'i_l_0607_D', 'i_l_0607_Q', 'i_l_0708_D', 'i_l_0708_Q', 'i_l_0809_D', 'i_l_0809_Q', 'i_l_0910_D', 'i_l_0910_Q', 'i_l_1011_D', 'i_l_1011_Q'] 
        self.y_run_list = ['v_02_D', 'v_02_Q', 'v_03_D', 'v_03_Q', 'v_04_D', 'v_04_Q', 'v_05_D', 'v_05_Q', 'v_06_D', 'v_06_Q', 'v_07_D', 'v_07_Q', 'v_08_D', 'v_08_Q', 'v_09_D', 'v_09_Q', 'v_10_D', 'v_10_Q', 'v_11_D', 'v_11_Q'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_02_D', 'v_02_Q', 'v_03_D', 'v_03_Q', 'v_04_D', 'v_04_Q', 'v_05_D', 'v_05_Q', 'v_06_D', 'v_06_Q', 'v_07_D', 'v_07_Q', 'v_08_D', 'v_08_Q', 'v_09_D', 'v_09_Q', 'v_10_D', 'v_10_Q', 'v_11_D', 'v_11_Q'] 
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
    R_0102 = struct[0].R_0102
    L_0102 = struct[0].L_0102
    C_0102 = struct[0].C_0102
    R_0203 = struct[0].R_0203
    L_0203 = struct[0].L_0203
    C_0203 = struct[0].C_0203
    R_0304 = struct[0].R_0304
    L_0304 = struct[0].L_0304
    C_0304 = struct[0].C_0304
    R_0308 = struct[0].R_0308
    L_0308 = struct[0].L_0308
    C_0308 = struct[0].C_0308
    R_0405 = struct[0].R_0405
    L_0405 = struct[0].L_0405
    C_0405 = struct[0].C_0405
    R_0506 = struct[0].R_0506
    L_0506 = struct[0].L_0506
    C_0506 = struct[0].C_0506
    R_0607 = struct[0].R_0607
    L_0607 = struct[0].L_0607
    C_0607 = struct[0].C_0607
    R_0708 = struct[0].R_0708
    L_0708 = struct[0].L_0708
    C_0708 = struct[0].C_0708
    R_0809 = struct[0].R_0809
    L_0809 = struct[0].L_0809
    C_0809 = struct[0].C_0809
    R_0910 = struct[0].R_0910
    L_0910 = struct[0].L_0910
    C_0910 = struct[0].C_0910
    R_1011 = struct[0].R_1011
    L_1011 = struct[0].L_1011
    C_1011 = struct[0].C_1011
    i_02_D = struct[0].i_02_D
    i_02_Q = struct[0].i_02_Q
    i_03_D = struct[0].i_03_D
    i_03_Q = struct[0].i_03_Q
    i_04_D = struct[0].i_04_D
    i_04_Q = struct[0].i_04_Q
    i_05_D = struct[0].i_05_D
    i_05_Q = struct[0].i_05_Q
    i_06_D = struct[0].i_06_D
    i_06_Q = struct[0].i_06_Q
    i_07_D = struct[0].i_07_D
    i_07_Q = struct[0].i_07_Q
    i_08_D = struct[0].i_08_D
    i_08_Q = struct[0].i_08_Q
    i_09_D = struct[0].i_09_D
    i_09_Q = struct[0].i_09_Q
    i_10_D = struct[0].i_10_D
    i_10_Q = struct[0].i_10_Q
    i_11_D = struct[0].i_11_D
    i_11_Q = struct[0].i_11_Q
    omega = struct[0].omega
    
    # Inputs:
    v_01_D = struct[0].v_01_D
    v_01_Q = struct[0].v_01_Q
    
    # Dynamical states:
    i_l_0102_D = struct[0].x[0,0]
    i_l_0102_Q = struct[0].x[1,0]
    i_l_0203_D = struct[0].x[2,0]
    i_l_0203_Q = struct[0].x[3,0]
    i_l_0304_D = struct[0].x[4,0]
    i_l_0304_Q = struct[0].x[5,0]
    i_l_0308_D = struct[0].x[6,0]
    i_l_0308_Q = struct[0].x[7,0]
    i_l_0405_D = struct[0].x[8,0]
    i_l_0405_Q = struct[0].x[9,0]
    i_l_0506_D = struct[0].x[10,0]
    i_l_0506_Q = struct[0].x[11,0]
    i_l_0607_D = struct[0].x[12,0]
    i_l_0607_Q = struct[0].x[13,0]
    i_l_0708_D = struct[0].x[14,0]
    i_l_0708_Q = struct[0].x[15,0]
    i_l_0809_D = struct[0].x[16,0]
    i_l_0809_Q = struct[0].x[17,0]
    i_l_0910_D = struct[0].x[18,0]
    i_l_0910_Q = struct[0].x[19,0]
    i_l_1011_D = struct[0].x[20,0]
    i_l_1011_Q = struct[0].x[21,0]
    
    # Algebraic states:
    v_02_D = struct[0].y_run[0,0]
    v_02_Q = struct[0].y_run[1,0]
    v_03_D = struct[0].y_run[2,0]
    v_03_Q = struct[0].y_run[3,0]
    v_04_D = struct[0].y_run[4,0]
    v_04_Q = struct[0].y_run[5,0]
    v_05_D = struct[0].y_run[6,0]
    v_05_Q = struct[0].y_run[7,0]
    v_06_D = struct[0].y_run[8,0]
    v_06_Q = struct[0].y_run[9,0]
    v_07_D = struct[0].y_run[10,0]
    v_07_Q = struct[0].y_run[11,0]
    v_08_D = struct[0].y_run[12,0]
    v_08_Q = struct[0].y_run[13,0]
    v_09_D = struct[0].y_run[14,0]
    v_09_Q = struct[0].y_run[15,0]
    v_10_D = struct[0].y_run[16,0]
    v_10_Q = struct[0].y_run[17,0]
    v_11_D = struct[0].y_run[18,0]
    v_11_Q = struct[0].y_run[19,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (L_0102*i_l_0102_Q*omega - R_0102*i_l_0102_D + v_01_D - v_02_D)/L_0102
        struct[0].f[1,0] = (-L_0102*i_l_0102_D*omega - R_0102*i_l_0102_Q + v_01_Q - v_02_Q)/L_0102
        struct[0].f[2,0] = (L_0203*i_l_0203_Q*omega - R_0203*i_l_0203_D + v_02_D - v_03_D)/L_0203
        struct[0].f[3,0] = (-L_0203*i_l_0203_D*omega - R_0203*i_l_0203_Q + v_02_Q - v_03_Q)/L_0203
        struct[0].f[4,0] = (L_0304*i_l_0304_Q*omega - R_0304*i_l_0304_D + v_03_D - v_04_D)/L_0304
        struct[0].f[5,0] = (-L_0304*i_l_0304_D*omega - R_0304*i_l_0304_Q + v_03_Q - v_04_Q)/L_0304
        struct[0].f[6,0] = (L_0308*i_l_0308_Q*omega - R_0308*i_l_0308_D + v_03_D - v_08_D)/L_0308
        struct[0].f[7,0] = (-L_0308*i_l_0308_D*omega - R_0308*i_l_0308_Q + v_03_Q - v_08_Q)/L_0308
        struct[0].f[8,0] = (L_0405*i_l_0405_Q*omega - R_0405*i_l_0405_D + v_04_D - v_05_D)/L_0405
        struct[0].f[9,0] = (-L_0405*i_l_0405_D*omega - R_0405*i_l_0405_Q + v_04_Q - v_05_Q)/L_0405
        struct[0].f[10,0] = (L_0506*i_l_0506_Q*omega - R_0506*i_l_0506_D + v_05_D - v_06_D)/L_0506
        struct[0].f[11,0] = (-L_0506*i_l_0506_D*omega - R_0506*i_l_0506_Q + v_05_Q - v_06_Q)/L_0506
        struct[0].f[12,0] = (L_0607*i_l_0607_Q*omega - R_0607*i_l_0607_D + v_06_D - v_07_D)/L_0607
        struct[0].f[13,0] = (-L_0607*i_l_0607_D*omega - R_0607*i_l_0607_Q + v_06_Q - v_07_Q)/L_0607
        struct[0].f[14,0] = (L_0708*i_l_0708_Q*omega - R_0708*i_l_0708_D + v_07_D - v_08_D)/L_0708
        struct[0].f[15,0] = (-L_0708*i_l_0708_D*omega - R_0708*i_l_0708_Q + v_07_Q - v_08_Q)/L_0708
        struct[0].f[16,0] = (L_0809*i_l_0809_Q*omega - R_0809*i_l_0809_D + v_08_D - v_09_D)/L_0809
        struct[0].f[17,0] = (-L_0809*i_l_0809_D*omega - R_0809*i_l_0809_Q + v_08_Q - v_09_Q)/L_0809
        struct[0].f[18,0] = (L_0910*i_l_0910_Q*omega - R_0910*i_l_0910_D + v_09_D - v_10_D)/L_0910
        struct[0].f[19,0] = (-L_0910*i_l_0910_D*omega - R_0910*i_l_0910_Q + v_09_Q - v_10_Q)/L_0910
        struct[0].f[20,0] = (L_1011*i_l_1011_Q*omega - R_1011*i_l_1011_D + v_10_D - v_11_D)/L_1011
        struct[0].f[21,0] = (-L_1011*i_l_1011_D*omega - R_1011*i_l_1011_Q + v_10_Q - v_11_Q)/L_1011
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_02_D + i_l_0102_D - i_l_0203_D + omega*v_02_Q*(C_0102/2 + C_0203/2)
        struct[0].g[1,0] = i_02_Q + i_l_0102_Q - i_l_0203_Q - omega*v_02_D*(C_0102/2 + C_0203/2)
        struct[0].g[2,0] = i_03_D + i_l_0203_D - i_l_0304_D - i_l_0308_D + omega*v_03_Q*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].g[3,0] = i_03_Q + i_l_0203_Q - i_l_0304_Q - i_l_0308_Q - omega*v_03_D*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].g[4,0] = i_04_D + i_l_0304_D - i_l_0405_D + omega*v_04_Q*(C_0304/2 + C_0405/2)
        struct[0].g[5,0] = i_04_Q + i_l_0304_Q - i_l_0405_Q - omega*v_04_D*(C_0304/2 + C_0405/2)
        struct[0].g[6,0] = i_05_D + i_l_0405_D - i_l_0506_D + omega*v_05_Q*(C_0405/2 + C_0506/2)
        struct[0].g[7,0] = i_05_Q + i_l_0405_Q - i_l_0506_Q - omega*v_05_D*(C_0405/2 + C_0506/2)
        struct[0].g[8,0] = i_06_D + i_l_0506_D - i_l_0607_D + omega*v_06_Q*(C_0506/2 + C_0607/2)
        struct[0].g[9,0] = i_06_Q + i_l_0506_Q - i_l_0607_Q - omega*v_06_D*(C_0506/2 + C_0607/2)
        struct[0].g[10,0] = i_07_D + i_l_0607_D - i_l_0708_D + omega*v_07_Q*(C_0607/2 + C_0708/2)
        struct[0].g[11,0] = i_07_Q + i_l_0607_Q - i_l_0708_Q - omega*v_07_D*(C_0607/2 + C_0708/2)
        struct[0].g[12,0] = i_08_D + i_l_0308_D + i_l_0708_D - i_l_0809_D + omega*v_08_Q*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].g[13,0] = i_08_Q + i_l_0308_Q + i_l_0708_Q - i_l_0809_Q - omega*v_08_D*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].g[14,0] = i_09_D + i_l_0809_D - i_l_0910_D + omega*v_09_Q*(C_0809/2 + C_0910/2)
        struct[0].g[15,0] = i_09_Q + i_l_0809_Q - i_l_0910_Q - omega*v_09_D*(C_0809/2 + C_0910/2)
        struct[0].g[16,0] = i_10_D + i_l_0910_D - i_l_1011_D + omega*v_10_Q*(C_0910/2 + C_1011/2)
        struct[0].g[17,0] = i_10_Q + i_l_0910_Q - i_l_1011_Q - omega*v_10_D*(C_0910/2 + C_1011/2)
        struct[0].g[18,0] = C_1011*omega*v_11_Q/2 + i_11_D + i_l_1011_D
        struct[0].g[19,0] = -C_1011*omega*v_11_D/2 + i_11_Q + i_l_1011_Q
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = v_01_D
    

    if mode == 10:

        struct[0].Fx[0,0] = -R_0102/L_0102
        struct[0].Fx[0,1] = omega
        struct[0].Fx[1,0] = -omega
        struct[0].Fx[1,1] = -R_0102/L_0102
        struct[0].Fx[2,2] = -R_0203/L_0203
        struct[0].Fx[2,3] = omega
        struct[0].Fx[3,2] = -omega
        struct[0].Fx[3,3] = -R_0203/L_0203
        struct[0].Fx[4,4] = -R_0304/L_0304
        struct[0].Fx[4,5] = omega
        struct[0].Fx[5,4] = -omega
        struct[0].Fx[5,5] = -R_0304/L_0304
        struct[0].Fx[6,6] = -R_0308/L_0308
        struct[0].Fx[6,7] = omega
        struct[0].Fx[7,6] = -omega
        struct[0].Fx[7,7] = -R_0308/L_0308
        struct[0].Fx[8,8] = -R_0405/L_0405
        struct[0].Fx[8,9] = omega
        struct[0].Fx[9,8] = -omega
        struct[0].Fx[9,9] = -R_0405/L_0405
        struct[0].Fx[10,10] = -R_0506/L_0506
        struct[0].Fx[10,11] = omega
        struct[0].Fx[11,10] = -omega
        struct[0].Fx[11,11] = -R_0506/L_0506
        struct[0].Fx[12,12] = -R_0607/L_0607
        struct[0].Fx[12,13] = omega
        struct[0].Fx[13,12] = -omega
        struct[0].Fx[13,13] = -R_0607/L_0607
        struct[0].Fx[14,14] = -R_0708/L_0708
        struct[0].Fx[14,15] = omega
        struct[0].Fx[15,14] = -omega
        struct[0].Fx[15,15] = -R_0708/L_0708
        struct[0].Fx[16,16] = -R_0809/L_0809
        struct[0].Fx[16,17] = omega
        struct[0].Fx[17,16] = -omega
        struct[0].Fx[17,17] = -R_0809/L_0809
        struct[0].Fx[18,18] = -R_0910/L_0910
        struct[0].Fx[18,19] = omega
        struct[0].Fx[19,18] = -omega
        struct[0].Fx[19,19] = -R_0910/L_0910
        struct[0].Fx[20,20] = -R_1011/L_1011
        struct[0].Fx[20,21] = omega
        struct[0].Fx[21,20] = -omega
        struct[0].Fx[21,21] = -R_1011/L_1011

    if mode == 11:

        struct[0].Fy[0,0] = -1/L_0102
        struct[0].Fy[1,1] = -1/L_0102
        struct[0].Fy[2,0] = 1/L_0203
        struct[0].Fy[2,2] = -1/L_0203
        struct[0].Fy[3,1] = 1/L_0203
        struct[0].Fy[3,3] = -1/L_0203
        struct[0].Fy[4,2] = 1/L_0304
        struct[0].Fy[4,4] = -1/L_0304
        struct[0].Fy[5,3] = 1/L_0304
        struct[0].Fy[5,5] = -1/L_0304
        struct[0].Fy[6,2] = 1/L_0308
        struct[0].Fy[6,12] = -1/L_0308
        struct[0].Fy[7,3] = 1/L_0308
        struct[0].Fy[7,13] = -1/L_0308
        struct[0].Fy[8,4] = 1/L_0405
        struct[0].Fy[8,6] = -1/L_0405
        struct[0].Fy[9,5] = 1/L_0405
        struct[0].Fy[9,7] = -1/L_0405
        struct[0].Fy[10,6] = 1/L_0506
        struct[0].Fy[10,8] = -1/L_0506
        struct[0].Fy[11,7] = 1/L_0506
        struct[0].Fy[11,9] = -1/L_0506
        struct[0].Fy[12,8] = 1/L_0607
        struct[0].Fy[12,10] = -1/L_0607
        struct[0].Fy[13,9] = 1/L_0607
        struct[0].Fy[13,11] = -1/L_0607
        struct[0].Fy[14,10] = 1/L_0708
        struct[0].Fy[14,12] = -1/L_0708
        struct[0].Fy[15,11] = 1/L_0708
        struct[0].Fy[15,13] = -1/L_0708
        struct[0].Fy[16,12] = 1/L_0809
        struct[0].Fy[16,14] = -1/L_0809
        struct[0].Fy[17,13] = 1/L_0809
        struct[0].Fy[17,15] = -1/L_0809
        struct[0].Fy[18,14] = 1/L_0910
        struct[0].Fy[18,16] = -1/L_0910
        struct[0].Fy[19,15] = 1/L_0910
        struct[0].Fy[19,17] = -1/L_0910
        struct[0].Fy[20,16] = 1/L_1011
        struct[0].Fy[20,18] = -1/L_1011
        struct[0].Fy[21,17] = 1/L_1011
        struct[0].Fy[21,19] = -1/L_1011

        struct[0].Gx[0,0] = 1
        struct[0].Gx[0,2] = -1
        struct[0].Gx[1,1] = 1
        struct[0].Gx[1,3] = -1
        struct[0].Gx[2,2] = 1
        struct[0].Gx[2,4] = -1
        struct[0].Gx[2,6] = -1
        struct[0].Gx[3,3] = 1
        struct[0].Gx[3,5] = -1
        struct[0].Gx[3,7] = -1
        struct[0].Gx[4,4] = 1
        struct[0].Gx[4,8] = -1
        struct[0].Gx[5,5] = 1
        struct[0].Gx[5,9] = -1
        struct[0].Gx[6,8] = 1
        struct[0].Gx[6,10] = -1
        struct[0].Gx[7,9] = 1
        struct[0].Gx[7,11] = -1
        struct[0].Gx[8,10] = 1
        struct[0].Gx[8,12] = -1
        struct[0].Gx[9,11] = 1
        struct[0].Gx[9,13] = -1
        struct[0].Gx[10,12] = 1
        struct[0].Gx[10,14] = -1
        struct[0].Gx[11,13] = 1
        struct[0].Gx[11,15] = -1
        struct[0].Gx[12,6] = 1
        struct[0].Gx[12,14] = 1
        struct[0].Gx[12,16] = -1
        struct[0].Gx[13,7] = 1
        struct[0].Gx[13,15] = 1
        struct[0].Gx[13,17] = -1
        struct[0].Gx[14,16] = 1
        struct[0].Gx[14,18] = -1
        struct[0].Gx[15,17] = 1
        struct[0].Gx[15,19] = -1
        struct[0].Gx[16,18] = 1
        struct[0].Gx[16,20] = -1
        struct[0].Gx[17,19] = 1
        struct[0].Gx[17,21] = -1
        struct[0].Gx[18,20] = 1
        struct[0].Gx[19,21] = 1

        struct[0].Gy[0,1] = omega*(C_0102/2 + C_0203/2)
        struct[0].Gy[1,0] = -omega*(C_0102/2 + C_0203/2)
        struct[0].Gy[2,3] = omega*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].Gy[3,2] = -omega*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].Gy[4,5] = omega*(C_0304/2 + C_0405/2)
        struct[0].Gy[5,4] = -omega*(C_0304/2 + C_0405/2)
        struct[0].Gy[6,7] = omega*(C_0405/2 + C_0506/2)
        struct[0].Gy[7,6] = -omega*(C_0405/2 + C_0506/2)
        struct[0].Gy[8,9] = omega*(C_0506/2 + C_0607/2)
        struct[0].Gy[9,8] = -omega*(C_0506/2 + C_0607/2)
        struct[0].Gy[10,11] = omega*(C_0607/2 + C_0708/2)
        struct[0].Gy[11,10] = -omega*(C_0607/2 + C_0708/2)
        struct[0].Gy[12,13] = omega*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].Gy[13,12] = -omega*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].Gy[14,15] = omega*(C_0809/2 + C_0910/2)
        struct[0].Gy[15,14] = -omega*(C_0809/2 + C_0910/2)
        struct[0].Gy[16,17] = omega*(C_0910/2 + C_1011/2)
        struct[0].Gy[17,16] = -omega*(C_0910/2 + C_1011/2)
        struct[0].Gy[18,19] = C_1011*omega/2
        struct[0].Gy[19,18] = -C_1011*omega/2

    if mode > 12:





        pass



@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    R_0102 = struct[0].R_0102
    L_0102 = struct[0].L_0102
    C_0102 = struct[0].C_0102
    R_0203 = struct[0].R_0203
    L_0203 = struct[0].L_0203
    C_0203 = struct[0].C_0203
    R_0304 = struct[0].R_0304
    L_0304 = struct[0].L_0304
    C_0304 = struct[0].C_0304
    R_0308 = struct[0].R_0308
    L_0308 = struct[0].L_0308
    C_0308 = struct[0].C_0308
    R_0405 = struct[0].R_0405
    L_0405 = struct[0].L_0405
    C_0405 = struct[0].C_0405
    R_0506 = struct[0].R_0506
    L_0506 = struct[0].L_0506
    C_0506 = struct[0].C_0506
    R_0607 = struct[0].R_0607
    L_0607 = struct[0].L_0607
    C_0607 = struct[0].C_0607
    R_0708 = struct[0].R_0708
    L_0708 = struct[0].L_0708
    C_0708 = struct[0].C_0708
    R_0809 = struct[0].R_0809
    L_0809 = struct[0].L_0809
    C_0809 = struct[0].C_0809
    R_0910 = struct[0].R_0910
    L_0910 = struct[0].L_0910
    C_0910 = struct[0].C_0910
    R_1011 = struct[0].R_1011
    L_1011 = struct[0].L_1011
    C_1011 = struct[0].C_1011
    i_02_D = struct[0].i_02_D
    i_02_Q = struct[0].i_02_Q
    i_03_D = struct[0].i_03_D
    i_03_Q = struct[0].i_03_Q
    i_04_D = struct[0].i_04_D
    i_04_Q = struct[0].i_04_Q
    i_05_D = struct[0].i_05_D
    i_05_Q = struct[0].i_05_Q
    i_06_D = struct[0].i_06_D
    i_06_Q = struct[0].i_06_Q
    i_07_D = struct[0].i_07_D
    i_07_Q = struct[0].i_07_Q
    i_08_D = struct[0].i_08_D
    i_08_Q = struct[0].i_08_Q
    i_09_D = struct[0].i_09_D
    i_09_Q = struct[0].i_09_Q
    i_10_D = struct[0].i_10_D
    i_10_Q = struct[0].i_10_Q
    i_11_D = struct[0].i_11_D
    i_11_Q = struct[0].i_11_Q
    omega = struct[0].omega
    
    # Inputs:
    v_01_D = struct[0].v_01_D
    v_01_Q = struct[0].v_01_Q
    
    # Dynamical states:
    i_l_0102_D = struct[0].x[0,0]
    i_l_0102_Q = struct[0].x[1,0]
    i_l_0203_D = struct[0].x[2,0]
    i_l_0203_Q = struct[0].x[3,0]
    i_l_0304_D = struct[0].x[4,0]
    i_l_0304_Q = struct[0].x[5,0]
    i_l_0308_D = struct[0].x[6,0]
    i_l_0308_Q = struct[0].x[7,0]
    i_l_0405_D = struct[0].x[8,0]
    i_l_0405_Q = struct[0].x[9,0]
    i_l_0506_D = struct[0].x[10,0]
    i_l_0506_Q = struct[0].x[11,0]
    i_l_0607_D = struct[0].x[12,0]
    i_l_0607_Q = struct[0].x[13,0]
    i_l_0708_D = struct[0].x[14,0]
    i_l_0708_Q = struct[0].x[15,0]
    i_l_0809_D = struct[0].x[16,0]
    i_l_0809_Q = struct[0].x[17,0]
    i_l_0910_D = struct[0].x[18,0]
    i_l_0910_Q = struct[0].x[19,0]
    i_l_1011_D = struct[0].x[20,0]
    i_l_1011_Q = struct[0].x[21,0]
    
    # Algebraic states:
    v_02_D = struct[0].y_ini[0,0]
    v_02_Q = struct[0].y_ini[1,0]
    v_03_D = struct[0].y_ini[2,0]
    v_03_Q = struct[0].y_ini[3,0]
    v_04_D = struct[0].y_ini[4,0]
    v_04_Q = struct[0].y_ini[5,0]
    v_05_D = struct[0].y_ini[6,0]
    v_05_Q = struct[0].y_ini[7,0]
    v_06_D = struct[0].y_ini[8,0]
    v_06_Q = struct[0].y_ini[9,0]
    v_07_D = struct[0].y_ini[10,0]
    v_07_Q = struct[0].y_ini[11,0]
    v_08_D = struct[0].y_ini[12,0]
    v_08_Q = struct[0].y_ini[13,0]
    v_09_D = struct[0].y_ini[14,0]
    v_09_Q = struct[0].y_ini[15,0]
    v_10_D = struct[0].y_ini[16,0]
    v_10_Q = struct[0].y_ini[17,0]
    v_11_D = struct[0].y_ini[18,0]
    v_11_Q = struct[0].y_ini[19,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (L_0102*i_l_0102_Q*omega - R_0102*i_l_0102_D + v_01_D - v_02_D)/L_0102
        struct[0].f[1,0] = (-L_0102*i_l_0102_D*omega - R_0102*i_l_0102_Q + v_01_Q - v_02_Q)/L_0102
        struct[0].f[2,0] = (L_0203*i_l_0203_Q*omega - R_0203*i_l_0203_D + v_02_D - v_03_D)/L_0203
        struct[0].f[3,0] = (-L_0203*i_l_0203_D*omega - R_0203*i_l_0203_Q + v_02_Q - v_03_Q)/L_0203
        struct[0].f[4,0] = (L_0304*i_l_0304_Q*omega - R_0304*i_l_0304_D + v_03_D - v_04_D)/L_0304
        struct[0].f[5,0] = (-L_0304*i_l_0304_D*omega - R_0304*i_l_0304_Q + v_03_Q - v_04_Q)/L_0304
        struct[0].f[6,0] = (L_0308*i_l_0308_Q*omega - R_0308*i_l_0308_D + v_03_D - v_08_D)/L_0308
        struct[0].f[7,0] = (-L_0308*i_l_0308_D*omega - R_0308*i_l_0308_Q + v_03_Q - v_08_Q)/L_0308
        struct[0].f[8,0] = (L_0405*i_l_0405_Q*omega - R_0405*i_l_0405_D + v_04_D - v_05_D)/L_0405
        struct[0].f[9,0] = (-L_0405*i_l_0405_D*omega - R_0405*i_l_0405_Q + v_04_Q - v_05_Q)/L_0405
        struct[0].f[10,0] = (L_0506*i_l_0506_Q*omega - R_0506*i_l_0506_D + v_05_D - v_06_D)/L_0506
        struct[0].f[11,0] = (-L_0506*i_l_0506_D*omega - R_0506*i_l_0506_Q + v_05_Q - v_06_Q)/L_0506
        struct[0].f[12,0] = (L_0607*i_l_0607_Q*omega - R_0607*i_l_0607_D + v_06_D - v_07_D)/L_0607
        struct[0].f[13,0] = (-L_0607*i_l_0607_D*omega - R_0607*i_l_0607_Q + v_06_Q - v_07_Q)/L_0607
        struct[0].f[14,0] = (L_0708*i_l_0708_Q*omega - R_0708*i_l_0708_D + v_07_D - v_08_D)/L_0708
        struct[0].f[15,0] = (-L_0708*i_l_0708_D*omega - R_0708*i_l_0708_Q + v_07_Q - v_08_Q)/L_0708
        struct[0].f[16,0] = (L_0809*i_l_0809_Q*omega - R_0809*i_l_0809_D + v_08_D - v_09_D)/L_0809
        struct[0].f[17,0] = (-L_0809*i_l_0809_D*omega - R_0809*i_l_0809_Q + v_08_Q - v_09_Q)/L_0809
        struct[0].f[18,0] = (L_0910*i_l_0910_Q*omega - R_0910*i_l_0910_D + v_09_D - v_10_D)/L_0910
        struct[0].f[19,0] = (-L_0910*i_l_0910_D*omega - R_0910*i_l_0910_Q + v_09_Q - v_10_Q)/L_0910
        struct[0].f[20,0] = (L_1011*i_l_1011_Q*omega - R_1011*i_l_1011_D + v_10_D - v_11_D)/L_1011
        struct[0].f[21,0] = (-L_1011*i_l_1011_D*omega - R_1011*i_l_1011_Q + v_10_Q - v_11_Q)/L_1011
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_02_D + i_l_0102_D - i_l_0203_D + omega*v_02_Q*(C_0102/2 + C_0203/2)
        struct[0].g[1,0] = i_02_Q + i_l_0102_Q - i_l_0203_Q - omega*v_02_D*(C_0102/2 + C_0203/2)
        struct[0].g[2,0] = i_03_D + i_l_0203_D - i_l_0304_D - i_l_0308_D + omega*v_03_Q*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].g[3,0] = i_03_Q + i_l_0203_Q - i_l_0304_Q - i_l_0308_Q - omega*v_03_D*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].g[4,0] = i_04_D + i_l_0304_D - i_l_0405_D + omega*v_04_Q*(C_0304/2 + C_0405/2)
        struct[0].g[5,0] = i_04_Q + i_l_0304_Q - i_l_0405_Q - omega*v_04_D*(C_0304/2 + C_0405/2)
        struct[0].g[6,0] = i_05_D + i_l_0405_D - i_l_0506_D + omega*v_05_Q*(C_0405/2 + C_0506/2)
        struct[0].g[7,0] = i_05_Q + i_l_0405_Q - i_l_0506_Q - omega*v_05_D*(C_0405/2 + C_0506/2)
        struct[0].g[8,0] = i_06_D + i_l_0506_D - i_l_0607_D + omega*v_06_Q*(C_0506/2 + C_0607/2)
        struct[0].g[9,0] = i_06_Q + i_l_0506_Q - i_l_0607_Q - omega*v_06_D*(C_0506/2 + C_0607/2)
        struct[0].g[10,0] = i_07_D + i_l_0607_D - i_l_0708_D + omega*v_07_Q*(C_0607/2 + C_0708/2)
        struct[0].g[11,0] = i_07_Q + i_l_0607_Q - i_l_0708_Q - omega*v_07_D*(C_0607/2 + C_0708/2)
        struct[0].g[12,0] = i_08_D + i_l_0308_D + i_l_0708_D - i_l_0809_D + omega*v_08_Q*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].g[13,0] = i_08_Q + i_l_0308_Q + i_l_0708_Q - i_l_0809_Q - omega*v_08_D*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].g[14,0] = i_09_D + i_l_0809_D - i_l_0910_D + omega*v_09_Q*(C_0809/2 + C_0910/2)
        struct[0].g[15,0] = i_09_Q + i_l_0809_Q - i_l_0910_Q - omega*v_09_D*(C_0809/2 + C_0910/2)
        struct[0].g[16,0] = i_10_D + i_l_0910_D - i_l_1011_D + omega*v_10_Q*(C_0910/2 + C_1011/2)
        struct[0].g[17,0] = i_10_Q + i_l_0910_Q - i_l_1011_Q - omega*v_10_D*(C_0910/2 + C_1011/2)
        struct[0].g[18,0] = C_1011*omega*v_11_Q/2 + i_11_D + i_l_1011_D
        struct[0].g[19,0] = -C_1011*omega*v_11_D/2 + i_11_Q + i_l_1011_Q
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = v_01_D
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -R_0102/L_0102
        struct[0].Fx_ini[0,1] = omega
        struct[0].Fx_ini[1,0] = -omega
        struct[0].Fx_ini[1,1] = -R_0102/L_0102
        struct[0].Fx_ini[2,2] = -R_0203/L_0203
        struct[0].Fx_ini[2,3] = omega
        struct[0].Fx_ini[3,2] = -omega
        struct[0].Fx_ini[3,3] = -R_0203/L_0203
        struct[0].Fx_ini[4,4] = -R_0304/L_0304
        struct[0].Fx_ini[4,5] = omega
        struct[0].Fx_ini[5,4] = -omega
        struct[0].Fx_ini[5,5] = -R_0304/L_0304
        struct[0].Fx_ini[6,6] = -R_0308/L_0308
        struct[0].Fx_ini[6,7] = omega
        struct[0].Fx_ini[7,6] = -omega
        struct[0].Fx_ini[7,7] = -R_0308/L_0308
        struct[0].Fx_ini[8,8] = -R_0405/L_0405
        struct[0].Fx_ini[8,9] = omega
        struct[0].Fx_ini[9,8] = -omega
        struct[0].Fx_ini[9,9] = -R_0405/L_0405
        struct[0].Fx_ini[10,10] = -R_0506/L_0506
        struct[0].Fx_ini[10,11] = omega
        struct[0].Fx_ini[11,10] = -omega
        struct[0].Fx_ini[11,11] = -R_0506/L_0506
        struct[0].Fx_ini[12,12] = -R_0607/L_0607
        struct[0].Fx_ini[12,13] = omega
        struct[0].Fx_ini[13,12] = -omega
        struct[0].Fx_ini[13,13] = -R_0607/L_0607
        struct[0].Fx_ini[14,14] = -R_0708/L_0708
        struct[0].Fx_ini[14,15] = omega
        struct[0].Fx_ini[15,14] = -omega
        struct[0].Fx_ini[15,15] = -R_0708/L_0708
        struct[0].Fx_ini[16,16] = -R_0809/L_0809
        struct[0].Fx_ini[16,17] = omega
        struct[0].Fx_ini[17,16] = -omega
        struct[0].Fx_ini[17,17] = -R_0809/L_0809
        struct[0].Fx_ini[18,18] = -R_0910/L_0910
        struct[0].Fx_ini[18,19] = omega
        struct[0].Fx_ini[19,18] = -omega
        struct[0].Fx_ini[19,19] = -R_0910/L_0910
        struct[0].Fx_ini[20,20] = -R_1011/L_1011
        struct[0].Fx_ini[20,21] = omega
        struct[0].Fx_ini[21,20] = -omega
        struct[0].Fx_ini[21,21] = -R_1011/L_1011

    if mode == 11:

        struct[0].Fy_ini[0,0] = -1/L_0102 
        struct[0].Fy_ini[1,1] = -1/L_0102 
        struct[0].Fy_ini[2,0] = 1/L_0203 
        struct[0].Fy_ini[2,2] = -1/L_0203 
        struct[0].Fy_ini[3,1] = 1/L_0203 
        struct[0].Fy_ini[3,3] = -1/L_0203 
        struct[0].Fy_ini[4,2] = 1/L_0304 
        struct[0].Fy_ini[4,4] = -1/L_0304 
        struct[0].Fy_ini[5,3] = 1/L_0304 
        struct[0].Fy_ini[5,5] = -1/L_0304 
        struct[0].Fy_ini[6,2] = 1/L_0308 
        struct[0].Fy_ini[6,12] = -1/L_0308 
        struct[0].Fy_ini[7,3] = 1/L_0308 
        struct[0].Fy_ini[7,13] = -1/L_0308 
        struct[0].Fy_ini[8,4] = 1/L_0405 
        struct[0].Fy_ini[8,6] = -1/L_0405 
        struct[0].Fy_ini[9,5] = 1/L_0405 
        struct[0].Fy_ini[9,7] = -1/L_0405 
        struct[0].Fy_ini[10,6] = 1/L_0506 
        struct[0].Fy_ini[10,8] = -1/L_0506 
        struct[0].Fy_ini[11,7] = 1/L_0506 
        struct[0].Fy_ini[11,9] = -1/L_0506 
        struct[0].Fy_ini[12,8] = 1/L_0607 
        struct[0].Fy_ini[12,10] = -1/L_0607 
        struct[0].Fy_ini[13,9] = 1/L_0607 
        struct[0].Fy_ini[13,11] = -1/L_0607 
        struct[0].Fy_ini[14,10] = 1/L_0708 
        struct[0].Fy_ini[14,12] = -1/L_0708 
        struct[0].Fy_ini[15,11] = 1/L_0708 
        struct[0].Fy_ini[15,13] = -1/L_0708 
        struct[0].Fy_ini[16,12] = 1/L_0809 
        struct[0].Fy_ini[16,14] = -1/L_0809 
        struct[0].Fy_ini[17,13] = 1/L_0809 
        struct[0].Fy_ini[17,15] = -1/L_0809 
        struct[0].Fy_ini[18,14] = 1/L_0910 
        struct[0].Fy_ini[18,16] = -1/L_0910 
        struct[0].Fy_ini[19,15] = 1/L_0910 
        struct[0].Fy_ini[19,17] = -1/L_0910 
        struct[0].Fy_ini[20,16] = 1/L_1011 
        struct[0].Fy_ini[20,18] = -1/L_1011 
        struct[0].Fy_ini[21,17] = 1/L_1011 
        struct[0].Fy_ini[21,19] = -1/L_1011 

        struct[0].Gx_ini[0,0] = 1
        struct[0].Gx_ini[0,2] = -1
        struct[0].Gx_ini[1,1] = 1
        struct[0].Gx_ini[1,3] = -1
        struct[0].Gx_ini[2,2] = 1
        struct[0].Gx_ini[2,4] = -1
        struct[0].Gx_ini[2,6] = -1
        struct[0].Gx_ini[3,3] = 1
        struct[0].Gx_ini[3,5] = -1
        struct[0].Gx_ini[3,7] = -1
        struct[0].Gx_ini[4,4] = 1
        struct[0].Gx_ini[4,8] = -1
        struct[0].Gx_ini[5,5] = 1
        struct[0].Gx_ini[5,9] = -1
        struct[0].Gx_ini[6,8] = 1
        struct[0].Gx_ini[6,10] = -1
        struct[0].Gx_ini[7,9] = 1
        struct[0].Gx_ini[7,11] = -1
        struct[0].Gx_ini[8,10] = 1
        struct[0].Gx_ini[8,12] = -1
        struct[0].Gx_ini[9,11] = 1
        struct[0].Gx_ini[9,13] = -1
        struct[0].Gx_ini[10,12] = 1
        struct[0].Gx_ini[10,14] = -1
        struct[0].Gx_ini[11,13] = 1
        struct[0].Gx_ini[11,15] = -1
        struct[0].Gx_ini[12,6] = 1
        struct[0].Gx_ini[12,14] = 1
        struct[0].Gx_ini[12,16] = -1
        struct[0].Gx_ini[13,7] = 1
        struct[0].Gx_ini[13,15] = 1
        struct[0].Gx_ini[13,17] = -1
        struct[0].Gx_ini[14,16] = 1
        struct[0].Gx_ini[14,18] = -1
        struct[0].Gx_ini[15,17] = 1
        struct[0].Gx_ini[15,19] = -1
        struct[0].Gx_ini[16,18] = 1
        struct[0].Gx_ini[16,20] = -1
        struct[0].Gx_ini[17,19] = 1
        struct[0].Gx_ini[17,21] = -1
        struct[0].Gx_ini[18,20] = 1
        struct[0].Gx_ini[19,21] = 1

        struct[0].Gy_ini[0,1] = omega*(C_0102/2 + C_0203/2)
        struct[0].Gy_ini[1,0] = -omega*(C_0102/2 + C_0203/2)
        struct[0].Gy_ini[2,3] = omega*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].Gy_ini[3,2] = -omega*(C_0203/2 + C_0304/2 + C_0308/2)
        struct[0].Gy_ini[4,5] = omega*(C_0304/2 + C_0405/2)
        struct[0].Gy_ini[5,4] = -omega*(C_0304/2 + C_0405/2)
        struct[0].Gy_ini[6,7] = omega*(C_0405/2 + C_0506/2)
        struct[0].Gy_ini[7,6] = -omega*(C_0405/2 + C_0506/2)
        struct[0].Gy_ini[8,9] = omega*(C_0506/2 + C_0607/2)
        struct[0].Gy_ini[9,8] = -omega*(C_0506/2 + C_0607/2)
        struct[0].Gy_ini[10,11] = omega*(C_0607/2 + C_0708/2)
        struct[0].Gy_ini[11,10] = -omega*(C_0607/2 + C_0708/2)
        struct[0].Gy_ini[12,13] = omega*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].Gy_ini[13,12] = -omega*(C_0308/2 + C_0708/2 + C_0809/2)
        struct[0].Gy_ini[14,15] = omega*(C_0809/2 + C_0910/2)
        struct[0].Gy_ini[15,14] = -omega*(C_0809/2 + C_0910/2)
        struct[0].Gy_ini[16,17] = omega*(C_0910/2 + C_1011/2)
        struct[0].Gy_ini[17,16] = -omega*(C_0910/2 + C_1011/2)
        struct[0].Gy_ini[18,19] = C_1011*omega/2
        struct[0].Gy_ini[19,18] = -C_1011*omega/2





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


