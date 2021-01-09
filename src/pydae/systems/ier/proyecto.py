import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 


class proyecto_class: 

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
        self.N_y = 20 
        self.N_z = 7 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_GRI_POI', 'b_GRI_POI', 'g_POI_PMV', 'b_POI_PMV', 'g_PMV_GR1', 'b_PMV_GR1', 'g_GR1_GR2', 'b_GR1_GR2', 'g_PMV_GR3', 'b_PMV_GR3', 'g_GR3_GR4', 'b_GR3_GR4', 'U_GRI_n', 'U_POI_n', 'U_PMV_n', 'U_GR1_n', 'U_GR2_n', 'U_GR3_n', 'U_GR4_n', 'S_n_GRI', 'X_d_GRI', 'X1d_GRI', 'T1d0_GRI', 'X_q_GRI', 'X1q_GRI', 'T1q0_GRI', 'R_a_GRI', 'X_l_GRI', 'H_GRI', 'D_GRI', 'Omega_b_GRI', 'omega_s_GRI', 'K_a_GRI', 'T_r_GRI', 'v_pss_GRI', 'Droop_GRI', 'T_m_GRI', 'K_sec_GRI', 'K_delta_GRI', 'v_ref_GRI'] 
        self.params_values_list  = [100000000.0, 1.4986238532110094, -4.995412844036698, 2.941176470588235, -11.76470588235294, 24.742268041237114, -10.996563573883162, 24.742268041237114, -10.996563573883162, 24.742268041237114, -10.996563573883162, 24.742268041237114, -10.996563573883162, 66000.0, 66000.0, 20000.0, 20000.0, 20000.0, 20000.0, 20000.0, 100000000.0, 1.81, 0.3, 8.0, 1.76, 0.65, 1.0, 0.003, 0.05, 6.0, 1.0, 314.1592653589793, 1.0, 100, 0.1, 0.0, 0.05, 5.0, 0.001, 0.01, 1.0] 
        self.inputs_ini_list = ['P_GRI', 'Q_GRI', 'P_POI', 'Q_POI', 'P_PMV', 'Q_PMV', 'P_GR1', 'Q_GR1', 'P_GR2', 'Q_GR2', 'P_GR3', 'Q_GR3', 'P_GR4', 'Q_GR4'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0, 1000000.0, 0.0, 1000000.0, 0.0, 1000000.0, 0.0] 
        self.inputs_run_list = ['P_GRI', 'Q_GRI', 'P_POI', 'Q_POI', 'P_PMV', 'Q_PMV', 'P_GR1', 'Q_GR1', 'P_GR2', 'Q_GR2', 'P_GR3', 'Q_GR3', 'P_GR4', 'Q_GR4'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0, 1000000.0, 0.0, 1000000.0, 0.0, 1000000.0, 0.0] 
        self.outputs_list = ['V_GRI', 'V_POI', 'V_PMV', 'V_GR1', 'V_GR2', 'V_GR3', 'V_GR4'] 
        self.x_list = ['delta_GRI', 'omega_GRI', 'e1q_GRI', 'e1d_GRI', 'v_c_GRI', 'p_m_GRI', 'xi_m_GRI'] 
        self.y_run_list = ['V_GRI', 'theta_GRI', 'V_POI', 'theta_POI', 'V_PMV', 'theta_PMV', 'V_GR1', 'theta_GR1', 'V_GR2', 'theta_GR2', 'V_GR3', 'theta_GR3', 'V_GR4', 'theta_GR4', 'i_d_GRI', 'i_q_GRI', 'P_GRI_1', 'Q_GRI_1', 'v_f_GRI', 'p_m_ref_GRI'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_GRI', 'theta_GRI', 'V_POI', 'theta_POI', 'V_PMV', 'theta_PMV', 'V_GR1', 'theta_GR1', 'V_GR2', 'theta_GR2', 'V_GR3', 'theta_GR3', 'V_GR4', 'theta_GR4', 'i_d_GRI', 'i_q_GRI', 'P_GRI_1', 'Q_GRI_1', 'v_f_GRI', 'p_m_ref_GRI'] 
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
        ini(self.struct,10)
        ini(self.struct,11)       
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
                
            
    def initialize(self,events=[{}],xy0=0):
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
    
    def set_value(self,name,value):
        if name in self.inputs_run_list:
            self.struct[0][name] = value
        if name in self.params_list:
            self.struct[0][name] = value
            
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


@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_GRI_POI = struct[0].g_GRI_POI
    b_GRI_POI = struct[0].b_GRI_POI
    g_POI_PMV = struct[0].g_POI_PMV
    b_POI_PMV = struct[0].b_POI_PMV
    g_PMV_GR1 = struct[0].g_PMV_GR1
    b_PMV_GR1 = struct[0].b_PMV_GR1
    g_GR1_GR2 = struct[0].g_GR1_GR2
    b_GR1_GR2 = struct[0].b_GR1_GR2
    g_PMV_GR3 = struct[0].g_PMV_GR3
    b_PMV_GR3 = struct[0].b_PMV_GR3
    g_GR3_GR4 = struct[0].g_GR3_GR4
    b_GR3_GR4 = struct[0].b_GR3_GR4
    U_GRI_n = struct[0].U_GRI_n
    U_POI_n = struct[0].U_POI_n
    U_PMV_n = struct[0].U_PMV_n
    U_GR1_n = struct[0].U_GR1_n
    U_GR2_n = struct[0].U_GR2_n
    U_GR3_n = struct[0].U_GR3_n
    U_GR4_n = struct[0].U_GR4_n
    S_n_GRI = struct[0].S_n_GRI
    X_d_GRI = struct[0].X_d_GRI
    X1d_GRI = struct[0].X1d_GRI
    T1d0_GRI = struct[0].T1d0_GRI
    X_q_GRI = struct[0].X_q_GRI
    X1q_GRI = struct[0].X1q_GRI
    T1q0_GRI = struct[0].T1q0_GRI
    R_a_GRI = struct[0].R_a_GRI
    X_l_GRI = struct[0].X_l_GRI
    H_GRI = struct[0].H_GRI
    D_GRI = struct[0].D_GRI
    Omega_b_GRI = struct[0].Omega_b_GRI
    omega_s_GRI = struct[0].omega_s_GRI
    K_a_GRI = struct[0].K_a_GRI
    T_r_GRI = struct[0].T_r_GRI
    v_pss_GRI = struct[0].v_pss_GRI
    Droop_GRI = struct[0].Droop_GRI
    T_m_GRI = struct[0].T_m_GRI
    K_sec_GRI = struct[0].K_sec_GRI
    K_delta_GRI = struct[0].K_delta_GRI
    v_ref_GRI = struct[0].v_ref_GRI
    
    # Inputs:
    P_GRI = struct[0].P_GRI
    Q_GRI = struct[0].Q_GRI
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_PMV = struct[0].P_PMV
    Q_PMV = struct[0].Q_PMV
    P_GR1 = struct[0].P_GR1
    Q_GR1 = struct[0].Q_GR1
    P_GR2 = struct[0].P_GR2
    Q_GR2 = struct[0].Q_GR2
    P_GR3 = struct[0].P_GR3
    Q_GR3 = struct[0].Q_GR3
    P_GR4 = struct[0].P_GR4
    Q_GR4 = struct[0].Q_GR4
    
    # Dynamical states:
    delta_GRI = struct[0].x[0,0]
    omega_GRI = struct[0].x[1,0]
    e1q_GRI = struct[0].x[2,0]
    e1d_GRI = struct[0].x[3,0]
    v_c_GRI = struct[0].x[4,0]
    p_m_GRI = struct[0].x[5,0]
    xi_m_GRI = struct[0].x[6,0]
    
    # Algebraic states:
    V_GRI = struct[0].y_ini[0,0]
    theta_GRI = struct[0].y_ini[1,0]
    V_POI = struct[0].y_ini[2,0]
    theta_POI = struct[0].y_ini[3,0]
    V_PMV = struct[0].y_ini[4,0]
    theta_PMV = struct[0].y_ini[5,0]
    V_GR1 = struct[0].y_ini[6,0]
    theta_GR1 = struct[0].y_ini[7,0]
    V_GR2 = struct[0].y_ini[8,0]
    theta_GR2 = struct[0].y_ini[9,0]
    V_GR3 = struct[0].y_ini[10,0]
    theta_GR3 = struct[0].y_ini[11,0]
    V_GR4 = struct[0].y_ini[12,0]
    theta_GR4 = struct[0].y_ini[13,0]
    i_d_GRI = struct[0].y_ini[14,0]
    i_q_GRI = struct[0].y_ini[15,0]
    P_GRI_1 = struct[0].y_ini[16,0]
    Q_GRI_1 = struct[0].y_ini[17,0]
    v_f_GRI = struct[0].y_ini[18,0]
    p_m_ref_GRI = struct[0].y_ini[19,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRI*delta_GRI + Omega_b_GRI*(omega_GRI - omega_s_GRI)
        struct[0].f[1,0] = (-D_GRI*(omega_GRI - omega_s_GRI) - i_d_GRI*(R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI)) - i_q_GRI*(R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI)) + p_m_GRI)/(2*H_GRI)
        struct[0].f[2,0] = (-e1q_GRI - i_d_GRI*(-X1d_GRI + X_d_GRI) + v_f_GRI)/T1d0_GRI
        struct[0].f[3,0] = (-e1d_GRI + i_q_GRI*(-X1q_GRI + X_q_GRI))/T1q0_GRI
        struct[0].f[4,0] = (V_GRI - v_c_GRI)/T_r_GRI
        struct[0].f[5,0] = (-p_m_GRI + p_m_ref_GRI)/T_m_GRI
        struct[0].f[6,0] = omega_GRI - 1
    
    # Algebraic equations:
    if mode == 3:

        g_n = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_GRI/S_base - P_GRI_1/S_base + V_GRI**2*g_GRI_POI + V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].g[1,0] = -Q_GRI/S_base - Q_GRI_1/S_base - V_GRI**2*b_GRI_POI + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].g[2,0] = -P_POI/S_base + V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + V_POI**2*(g_GRI_POI + g_POI_PMV)
        struct[0].g[3,0] = -Q_POI/S_base + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + V_POI**2*(-b_GRI_POI - b_POI_PMV)
        struct[0].g[4,0] = -P_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV**2*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].g[5,0] = -Q_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV**2*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].g[6,0] = -P_GR1/S_base + V_GR1**2*(g_GR1_GR2 + g_PMV_GR1) + V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].g[7,0] = -Q_GR1/S_base + V_GR1**2*(-b_GR1_GR2 - b_PMV_GR1) + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].g[8,0] = -P_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR2**2*g_GR1_GR2
        struct[0].g[9,0] = -Q_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - V_GR2**2*b_GR1_GR2
        struct[0].g[10,0] = -P_GR3/S_base + V_GR3**2*(g_GR3_GR4 + g_PMV_GR3) + V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].g[11,0] = -Q_GR3/S_base + V_GR3**2*(-b_GR3_GR4 - b_PMV_GR3) + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].g[12,0] = -P_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR4**2*g_GR3_GR4
        struct[0].g[13,0] = -Q_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - V_GR4**2*b_GR3_GR4
        struct[0].g[14,0] = R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI) + X1d_GRI*i_d_GRI - e1q_GRI
        struct[0].g[15,0] = R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI) - X1q_GRI*i_q_GRI - e1d_GRI
        struct[0].g[16,0] = -P_GRI_1/S_n_GRI + V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].g[17,0] = -Q_GRI_1/S_n_GRI + V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].g[18,0] = K_a_GRI*(-v_c_GRI + v_pss_GRI + v_ref_GRI) - v_f_GRI
        struct[0].g[19,0] = -K_sec_GRI*xi_m_GRI - p_m_ref_GRI - (omega_GRI - 1)/Droop_GRI
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_GRI
        struct[0].h[1,0] = V_POI
        struct[0].h[2,0] = V_PMV
        struct[0].h[3,0] = V_GR1
        struct[0].h[4,0] = V_GR2
        struct[0].h[5,0] = V_GR3
        struct[0].h[6,0] = V_GR4
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_GRI
        struct[0].Fx_ini[0,1] = Omega_b_GRI
        struct[0].Fx_ini[1,0] = (-V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fx_ini[1,1] = -D_GRI/(2*H_GRI)
        struct[0].Fx_ini[1,5] = 1/(2*H_GRI)
        struct[0].Fx_ini[2,2] = -1/T1d0_GRI
        struct[0].Fx_ini[3,3] = -1/T1q0_GRI
        struct[0].Fx_ini[4,4] = -1/T_r_GRI
        struct[0].Fx_ini[5,5] = -1/T_m_GRI

    if mode == 11:

        struct[0].Fy_ini[1,0] = (-i_d_GRI*sin(delta_GRI - theta_GRI) - i_q_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[1,1] = (V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[1,14] = (-2*R_a_GRI*i_d_GRI - V_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[1,15] = (-2*R_a_GRI*i_q_GRI - V_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[2,14] = (X1d_GRI - X_d_GRI)/T1d0_GRI 
        struct[0].Fy_ini[2,18] = 1/T1d0_GRI 
        struct[0].Fy_ini[3,15] = (-X1q_GRI + X_q_GRI)/T1q0_GRI 
        struct[0].Fy_ini[4,0] = 1/T_r_GRI 
        struct[0].Fy_ini[5,19] = 1/T_m_GRI 

        struct[0].Gx_ini[14,0] = -V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gx_ini[14,2] = -1
        struct[0].Gx_ini[15,0] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gx_ini[15,3] = -1
        struct[0].Gx_ini[16,0] = V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gx_ini[17,0] = -V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gx_ini[18,4] = -K_a_GRI
        struct[0].Gx_ini[19,1] = -1/Droop_GRI
        struct[0].Gx_ini[19,6] = -K_sec_GRI

        struct[0].Gy_ini[0,0] = 2*V_GRI*g_GRI_POI + V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,1] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,2] = V_GRI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,3] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,16] = -1/S_base
        struct[0].Gy_ini[1,0] = -2*V_GRI*b_GRI_POI + V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,17] = -1/S_base
        struct[0].Gy_ini[2,0] = V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[2,1] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[2,2] = V_GRI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + 2*V_POI*(g_GRI_POI + g_POI_PMV)
        struct[0].Gy_ini[2,3] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[2,4] = V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[2,5] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[3,0] = V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[3,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[3,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + 2*V_POI*(-b_GRI_POI - b_POI_PMV)
        struct[0].Gy_ini[3,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[3,4] = V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[3,5] = V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,2] = V_PMV*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,3] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,4] = V_GR1*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + 2*V_PMV*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,5] = V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,6] = V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[4,7] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[4,10] = V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[4,11] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[5,2] = V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,3] = V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + 2*V_PMV*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,6] = V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[5,7] = V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[5,10] = V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[5,11] = V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[6,4] = V_GR1*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,5] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,6] = 2*V_GR1*(g_GR1_GR2 + g_PMV_GR1) + V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,7] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,8] = V_GR1*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[6,9] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[7,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,6] = 2*V_GR1*(-b_GR1_GR2 - b_PMV_GR1) + V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[7,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[8,6] = V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[8,7] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[8,8] = V_GR1*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + 2*V_GR2*g_GR1_GR2
        struct[0].Gy_ini[8,9] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[9,6] = V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[9,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[9,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - 2*V_GR2*b_GR1_GR2
        struct[0].Gy_ini[9,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[10,4] = V_GR3*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,5] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,10] = 2*V_GR3*(g_GR3_GR4 + g_PMV_GR3) + V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,11] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,12] = V_GR3*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[10,13] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[11,4] = V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,5] = V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,10] = 2*V_GR3*(-b_GR3_GR4 - b_PMV_GR3) + V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[11,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[12,10] = V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[12,11] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[12,12] = V_GR3*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + 2*V_GR4*g_GR3_GR4
        struct[0].Gy_ini[12,13] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[13,10] = V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[13,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[13,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - 2*V_GR4*b_GR3_GR4
        struct[0].Gy_ini[13,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[14,0] = cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[14,1] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[14,14] = X1d_GRI
        struct[0].Gy_ini[14,15] = R_a_GRI
        struct[0].Gy_ini[15,0] = sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[15,1] = -V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[15,14] = R_a_GRI
        struct[0].Gy_ini[15,15] = -X1q_GRI
        struct[0].Gy_ini[16,0] = i_d_GRI*sin(delta_GRI - theta_GRI) + i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,1] = -V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,14] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,15] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,16] = -1/S_n_GRI
        struct[0].Gy_ini[17,0] = i_d_GRI*cos(delta_GRI - theta_GRI) - i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,1] = V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,14] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,15] = -V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,17] = -1/S_n_GRI



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_GRI_POI = struct[0].g_GRI_POI
    b_GRI_POI = struct[0].b_GRI_POI
    g_POI_PMV = struct[0].g_POI_PMV
    b_POI_PMV = struct[0].b_POI_PMV
    g_PMV_GR1 = struct[0].g_PMV_GR1
    b_PMV_GR1 = struct[0].b_PMV_GR1
    g_GR1_GR2 = struct[0].g_GR1_GR2
    b_GR1_GR2 = struct[0].b_GR1_GR2
    g_PMV_GR3 = struct[0].g_PMV_GR3
    b_PMV_GR3 = struct[0].b_PMV_GR3
    g_GR3_GR4 = struct[0].g_GR3_GR4
    b_GR3_GR4 = struct[0].b_GR3_GR4
    U_GRI_n = struct[0].U_GRI_n
    U_POI_n = struct[0].U_POI_n
    U_PMV_n = struct[0].U_PMV_n
    U_GR1_n = struct[0].U_GR1_n
    U_GR2_n = struct[0].U_GR2_n
    U_GR3_n = struct[0].U_GR3_n
    U_GR4_n = struct[0].U_GR4_n
    S_n_GRI = struct[0].S_n_GRI
    X_d_GRI = struct[0].X_d_GRI
    X1d_GRI = struct[0].X1d_GRI
    T1d0_GRI = struct[0].T1d0_GRI
    X_q_GRI = struct[0].X_q_GRI
    X1q_GRI = struct[0].X1q_GRI
    T1q0_GRI = struct[0].T1q0_GRI
    R_a_GRI = struct[0].R_a_GRI
    X_l_GRI = struct[0].X_l_GRI
    H_GRI = struct[0].H_GRI
    D_GRI = struct[0].D_GRI
    Omega_b_GRI = struct[0].Omega_b_GRI
    omega_s_GRI = struct[0].omega_s_GRI
    K_a_GRI = struct[0].K_a_GRI
    T_r_GRI = struct[0].T_r_GRI
    v_pss_GRI = struct[0].v_pss_GRI
    Droop_GRI = struct[0].Droop_GRI
    T_m_GRI = struct[0].T_m_GRI
    K_sec_GRI = struct[0].K_sec_GRI
    K_delta_GRI = struct[0].K_delta_GRI
    v_ref_GRI = struct[0].v_ref_GRI
    
    # Inputs:
    P_GRI = struct[0].P_GRI
    Q_GRI = struct[0].Q_GRI
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_PMV = struct[0].P_PMV
    Q_PMV = struct[0].Q_PMV
    P_GR1 = struct[0].P_GR1
    Q_GR1 = struct[0].Q_GR1
    P_GR2 = struct[0].P_GR2
    Q_GR2 = struct[0].Q_GR2
    P_GR3 = struct[0].P_GR3
    Q_GR3 = struct[0].Q_GR3
    P_GR4 = struct[0].P_GR4
    Q_GR4 = struct[0].Q_GR4
    
    # Dynamical states:
    delta_GRI = struct[0].x[0,0]
    omega_GRI = struct[0].x[1,0]
    e1q_GRI = struct[0].x[2,0]
    e1d_GRI = struct[0].x[3,0]
    v_c_GRI = struct[0].x[4,0]
    p_m_GRI = struct[0].x[5,0]
    xi_m_GRI = struct[0].x[6,0]
    
    # Algebraic states:
    V_GRI = struct[0].y_run[0,0]
    theta_GRI = struct[0].y_run[1,0]
    V_POI = struct[0].y_run[2,0]
    theta_POI = struct[0].y_run[3,0]
    V_PMV = struct[0].y_run[4,0]
    theta_PMV = struct[0].y_run[5,0]
    V_GR1 = struct[0].y_run[6,0]
    theta_GR1 = struct[0].y_run[7,0]
    V_GR2 = struct[0].y_run[8,0]
    theta_GR2 = struct[0].y_run[9,0]
    V_GR3 = struct[0].y_run[10,0]
    theta_GR3 = struct[0].y_run[11,0]
    V_GR4 = struct[0].y_run[12,0]
    theta_GR4 = struct[0].y_run[13,0]
    i_d_GRI = struct[0].y_run[14,0]
    i_q_GRI = struct[0].y_run[15,0]
    P_GRI_1 = struct[0].y_run[16,0]
    Q_GRI_1 = struct[0].y_run[17,0]
    v_f_GRI = struct[0].y_run[18,0]
    p_m_ref_GRI = struct[0].y_run[19,0]
    
    struct[0].u_run[0,0] = P_GRI
    struct[0].u_run[1,0] = Q_GRI
    struct[0].u_run[2,0] = P_POI
    struct[0].u_run[3,0] = Q_POI
    struct[0].u_run[4,0] = P_PMV
    struct[0].u_run[5,0] = Q_PMV
    struct[0].u_run[6,0] = P_GR1
    struct[0].u_run[7,0] = Q_GR1
    struct[0].u_run[8,0] = P_GR2
    struct[0].u_run[9,0] = Q_GR2
    struct[0].u_run[10,0] = P_GR3
    struct[0].u_run[11,0] = Q_GR3
    struct[0].u_run[12,0] = P_GR4
    struct[0].u_run[13,0] = Q_GR4
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRI*delta_GRI + Omega_b_GRI*(omega_GRI - omega_s_GRI)
        struct[0].f[1,0] = (-D_GRI*(omega_GRI - omega_s_GRI) - i_d_GRI*(R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI)) - i_q_GRI*(R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI)) + p_m_GRI)/(2*H_GRI)
        struct[0].f[2,0] = (-e1q_GRI - i_d_GRI*(-X1d_GRI + X_d_GRI) + v_f_GRI)/T1d0_GRI
        struct[0].f[3,0] = (-e1d_GRI + i_q_GRI*(-X1q_GRI + X_q_GRI))/T1q0_GRI
        struct[0].f[4,0] = (V_GRI - v_c_GRI)/T_r_GRI
        struct[0].f[5,0] = (-p_m_GRI + p_m_ref_GRI)/T_m_GRI
        struct[0].f[6,0] = omega_GRI - 1
    
    # Algebraic equations:
    if mode == 3:

        g_n = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_GRI/S_base - P_GRI_1/S_base + V_GRI**2*g_GRI_POI + V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].g[1,0] = -Q_GRI/S_base - Q_GRI_1/S_base - V_GRI**2*b_GRI_POI + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].g[2,0] = -P_POI/S_base + V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + V_POI**2*(g_GRI_POI + g_POI_PMV)
        struct[0].g[3,0] = -Q_POI/S_base + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + V_POI**2*(-b_GRI_POI - b_POI_PMV)
        struct[0].g[4,0] = -P_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV**2*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].g[5,0] = -Q_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV**2*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].g[6,0] = -P_GR1/S_base + V_GR1**2*(g_GR1_GR2 + g_PMV_GR1) + V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].g[7,0] = -Q_GR1/S_base + V_GR1**2*(-b_GR1_GR2 - b_PMV_GR1) + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].g[8,0] = -P_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR2**2*g_GR1_GR2
        struct[0].g[9,0] = -Q_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - V_GR2**2*b_GR1_GR2
        struct[0].g[10,0] = -P_GR3/S_base + V_GR3**2*(g_GR3_GR4 + g_PMV_GR3) + V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].g[11,0] = -Q_GR3/S_base + V_GR3**2*(-b_GR3_GR4 - b_PMV_GR3) + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].g[12,0] = -P_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR4**2*g_GR3_GR4
        struct[0].g[13,0] = -Q_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - V_GR4**2*b_GR3_GR4
        struct[0].g[14,0] = R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI) + X1d_GRI*i_d_GRI - e1q_GRI
        struct[0].g[15,0] = R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI) - X1q_GRI*i_q_GRI - e1d_GRI
        struct[0].g[16,0] = -P_GRI_1/S_n_GRI + V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].g[17,0] = -Q_GRI_1/S_n_GRI + V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].g[18,0] = K_a_GRI*(-v_c_GRI + v_pss_GRI + v_ref_GRI) - v_f_GRI
        struct[0].g[19,0] = -K_sec_GRI*xi_m_GRI - p_m_ref_GRI - (omega_GRI - 1)/Droop_GRI
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_GRI
        struct[0].h[1,0] = V_POI
        struct[0].h[2,0] = V_PMV
        struct[0].h[3,0] = V_GR1
        struct[0].h[4,0] = V_GR2
        struct[0].h[5,0] = V_GR3
        struct[0].h[6,0] = V_GR4
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_GRI
        struct[0].Fx[0,1] = Omega_b_GRI
        struct[0].Fx[1,0] = (-V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fx[1,1] = -D_GRI/(2*H_GRI)
        struct[0].Fx[1,5] = 1/(2*H_GRI)
        struct[0].Fx[2,2] = -1/T1d0_GRI
        struct[0].Fx[3,3] = -1/T1q0_GRI
        struct[0].Fx[4,4] = -1/T_r_GRI
        struct[0].Fx[5,5] = -1/T_m_GRI

    if mode == 11:

        struct[0].Fy[1,0] = (-i_d_GRI*sin(delta_GRI - theta_GRI) - i_q_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[1,1] = (V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[1,14] = (-2*R_a_GRI*i_d_GRI - V_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[1,15] = (-2*R_a_GRI*i_q_GRI - V_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[2,14] = (X1d_GRI - X_d_GRI)/T1d0_GRI
        struct[0].Fy[2,18] = 1/T1d0_GRI
        struct[0].Fy[3,15] = (-X1q_GRI + X_q_GRI)/T1q0_GRI
        struct[0].Fy[4,0] = 1/T_r_GRI
        struct[0].Fy[5,19] = 1/T_m_GRI

        struct[0].Gx[14,0] = -V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gx[14,2] = -1
        struct[0].Gx[15,0] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gx[15,3] = -1
        struct[0].Gx[16,0] = V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gx[17,0] = -V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gx[18,4] = -K_a_GRI
        struct[0].Gx[19,1] = -1/Droop_GRI
        struct[0].Gx[19,6] = -K_sec_GRI

        struct[0].Gy[0,0] = 2*V_GRI*g_GRI_POI + V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[0,1] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[0,2] = V_GRI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[0,3] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[0,16] = -1/S_base
        struct[0].Gy[1,0] = -2*V_GRI*b_GRI_POI + V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[1,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[1,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[1,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[1,17] = -1/S_base
        struct[0].Gy[2,0] = V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[2,1] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[2,2] = V_GRI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + 2*V_POI*(g_GRI_POI + g_POI_PMV)
        struct[0].Gy[2,3] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[2,4] = V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[2,5] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[3,0] = V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[3,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[3,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + 2*V_POI*(-b_GRI_POI - b_POI_PMV)
        struct[0].Gy[3,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[3,4] = V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[3,5] = V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[4,2] = V_PMV*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[4,3] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[4,4] = V_GR1*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + 2*V_PMV*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[4,5] = V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[4,6] = V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[4,7] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[4,10] = V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[4,11] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[5,2] = V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[5,3] = V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[5,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + 2*V_PMV*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[5,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[5,6] = V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[5,7] = V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[5,10] = V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[5,11] = V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[6,4] = V_GR1*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[6,5] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[6,6] = 2*V_GR1*(g_GR1_GR2 + g_PMV_GR1) + V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[6,7] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[6,8] = V_GR1*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[6,9] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[7,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[7,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[7,6] = 2*V_GR1*(-b_GR1_GR2 - b_PMV_GR1) + V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[7,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[7,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[7,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[8,6] = V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[8,7] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[8,8] = V_GR1*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + 2*V_GR2*g_GR1_GR2
        struct[0].Gy[8,9] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[9,6] = V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[9,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[9,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - 2*V_GR2*b_GR1_GR2
        struct[0].Gy[9,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[10,4] = V_GR3*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[10,5] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[10,10] = 2*V_GR3*(g_GR3_GR4 + g_PMV_GR3) + V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[10,11] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[10,12] = V_GR3*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[10,13] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[11,4] = V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[11,5] = V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[11,10] = 2*V_GR3*(-b_GR3_GR4 - b_PMV_GR3) + V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[11,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[11,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[11,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[12,10] = V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[12,11] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[12,12] = V_GR3*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + 2*V_GR4*g_GR3_GR4
        struct[0].Gy[12,13] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[13,10] = V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[13,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[13,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - 2*V_GR4*b_GR3_GR4
        struct[0].Gy[13,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[14,0] = cos(delta_GRI - theta_GRI)
        struct[0].Gy[14,1] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[14,14] = X1d_GRI
        struct[0].Gy[14,15] = R_a_GRI
        struct[0].Gy[15,0] = sin(delta_GRI - theta_GRI)
        struct[0].Gy[15,1] = -V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[15,14] = R_a_GRI
        struct[0].Gy[15,15] = -X1q_GRI
        struct[0].Gy[16,0] = i_d_GRI*sin(delta_GRI - theta_GRI) + i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[16,1] = -V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[16,14] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[16,15] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[16,16] = -1/S_n_GRI
        struct[0].Gy[17,0] = i_d_GRI*cos(delta_GRI - theta_GRI) - i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[17,1] = V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[17,14] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[17,15] = -V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[17,17] = -1/S_n_GRI

    if mode > 12:


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


        struct[0].Hy[0,0] = 1
        struct[0].Hy[1,2] = 1
        struct[0].Hy[2,4] = 1
        struct[0].Hy[3,6] = 1
        struct[0].Hy[4,8] = 1
        struct[0].Hy[5,10] = 1
        struct[0].Hy[6,12] = 1




def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_GRI_POI = struct[0].g_GRI_POI
    b_GRI_POI = struct[0].b_GRI_POI
    g_POI_PMV = struct[0].g_POI_PMV
    b_POI_PMV = struct[0].b_POI_PMV
    g_PMV_GR1 = struct[0].g_PMV_GR1
    b_PMV_GR1 = struct[0].b_PMV_GR1
    g_GR1_GR2 = struct[0].g_GR1_GR2
    b_GR1_GR2 = struct[0].b_GR1_GR2
    g_PMV_GR3 = struct[0].g_PMV_GR3
    b_PMV_GR3 = struct[0].b_PMV_GR3
    g_GR3_GR4 = struct[0].g_GR3_GR4
    b_GR3_GR4 = struct[0].b_GR3_GR4
    U_GRI_n = struct[0].U_GRI_n
    U_POI_n = struct[0].U_POI_n
    U_PMV_n = struct[0].U_PMV_n
    U_GR1_n = struct[0].U_GR1_n
    U_GR2_n = struct[0].U_GR2_n
    U_GR3_n = struct[0].U_GR3_n
    U_GR4_n = struct[0].U_GR4_n
    S_n_GRI = struct[0].S_n_GRI
    X_d_GRI = struct[0].X_d_GRI
    X1d_GRI = struct[0].X1d_GRI
    T1d0_GRI = struct[0].T1d0_GRI
    X_q_GRI = struct[0].X_q_GRI
    X1q_GRI = struct[0].X1q_GRI
    T1q0_GRI = struct[0].T1q0_GRI
    R_a_GRI = struct[0].R_a_GRI
    X_l_GRI = struct[0].X_l_GRI
    H_GRI = struct[0].H_GRI
    D_GRI = struct[0].D_GRI
    Omega_b_GRI = struct[0].Omega_b_GRI
    omega_s_GRI = struct[0].omega_s_GRI
    K_a_GRI = struct[0].K_a_GRI
    T_r_GRI = struct[0].T_r_GRI
    v_pss_GRI = struct[0].v_pss_GRI
    Droop_GRI = struct[0].Droop_GRI
    T_m_GRI = struct[0].T_m_GRI
    K_sec_GRI = struct[0].K_sec_GRI
    K_delta_GRI = struct[0].K_delta_GRI
    v_ref_GRI = struct[0].v_ref_GRI
    
    # Inputs:
    P_GRI = struct[0].P_GRI
    Q_GRI = struct[0].Q_GRI
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_PMV = struct[0].P_PMV
    Q_PMV = struct[0].Q_PMV
    P_GR1 = struct[0].P_GR1
    Q_GR1 = struct[0].Q_GR1
    P_GR2 = struct[0].P_GR2
    Q_GR2 = struct[0].Q_GR2
    P_GR3 = struct[0].P_GR3
    Q_GR3 = struct[0].Q_GR3
    P_GR4 = struct[0].P_GR4
    Q_GR4 = struct[0].Q_GR4
    
    # Dynamical states:
    delta_GRI = struct[0].x[0,0]
    omega_GRI = struct[0].x[1,0]
    e1q_GRI = struct[0].x[2,0]
    e1d_GRI = struct[0].x[3,0]
    v_c_GRI = struct[0].x[4,0]
    p_m_GRI = struct[0].x[5,0]
    xi_m_GRI = struct[0].x[6,0]
    
    # Algebraic states:
    V_GRI = struct[0].y_ini[0,0]
    theta_GRI = struct[0].y_ini[1,0]
    V_POI = struct[0].y_ini[2,0]
    theta_POI = struct[0].y_ini[3,0]
    V_PMV = struct[0].y_ini[4,0]
    theta_PMV = struct[0].y_ini[5,0]
    V_GR1 = struct[0].y_ini[6,0]
    theta_GR1 = struct[0].y_ini[7,0]
    V_GR2 = struct[0].y_ini[8,0]
    theta_GR2 = struct[0].y_ini[9,0]
    V_GR3 = struct[0].y_ini[10,0]
    theta_GR3 = struct[0].y_ini[11,0]
    V_GR4 = struct[0].y_ini[12,0]
    theta_GR4 = struct[0].y_ini[13,0]
    i_d_GRI = struct[0].y_ini[14,0]
    i_q_GRI = struct[0].y_ini[15,0]
    P_GRI_1 = struct[0].y_ini[16,0]
    Q_GRI_1 = struct[0].y_ini[17,0]
    v_f_GRI = struct[0].y_ini[18,0]
    p_m_ref_GRI = struct[0].y_ini[19,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRI*delta_GRI + Omega_b_GRI*(omega_GRI - omega_s_GRI)
        struct[0].f[1,0] = (-D_GRI*(omega_GRI - omega_s_GRI) - i_d_GRI*(R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI)) - i_q_GRI*(R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI)) + p_m_GRI)/(2*H_GRI)
        struct[0].f[2,0] = (-e1q_GRI - i_d_GRI*(-X1d_GRI + X_d_GRI) + v_f_GRI)/T1d0_GRI
        struct[0].f[3,0] = (-e1d_GRI + i_q_GRI*(-X1q_GRI + X_q_GRI))/T1q0_GRI
        struct[0].f[4,0] = (V_GRI - v_c_GRI)/T_r_GRI
        struct[0].f[5,0] = (-p_m_GRI + p_m_ref_GRI)/T_m_GRI
        struct[0].f[6,0] = omega_GRI - 1
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_GRI/S_base - P_GRI_1/S_base + V_GRI**2*g_GRI_POI + V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].g[1,0] = -Q_GRI/S_base - Q_GRI_1/S_base - V_GRI**2*b_GRI_POI + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].g[2,0] = -P_POI/S_base + V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + V_POI**2*(g_GRI_POI + g_POI_PMV)
        struct[0].g[3,0] = -Q_POI/S_base + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + V_POI**2*(-b_GRI_POI - b_POI_PMV)
        struct[0].g[4,0] = -P_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV**2*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].g[5,0] = -Q_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV**2*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].g[6,0] = -P_GR1/S_base + V_GR1**2*(g_GR1_GR2 + g_PMV_GR1) + V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].g[7,0] = -Q_GR1/S_base + V_GR1**2*(-b_GR1_GR2 - b_PMV_GR1) + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].g[8,0] = -P_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR2**2*g_GR1_GR2
        struct[0].g[9,0] = -Q_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - V_GR2**2*b_GR1_GR2
        struct[0].g[10,0] = -P_GR3/S_base + V_GR3**2*(g_GR3_GR4 + g_PMV_GR3) + V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].g[11,0] = -Q_GR3/S_base + V_GR3**2*(-b_GR3_GR4 - b_PMV_GR3) + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].g[12,0] = -P_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR4**2*g_GR3_GR4
        struct[0].g[13,0] = -Q_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - V_GR4**2*b_GR3_GR4
        struct[0].g[14,0] = R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI) + X1d_GRI*i_d_GRI - e1q_GRI
        struct[0].g[15,0] = R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI) - X1q_GRI*i_q_GRI - e1d_GRI
        struct[0].g[16,0] = -P_GRI_1/S_n_GRI + V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].g[17,0] = -Q_GRI_1/S_n_GRI + V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].g[18,0] = K_a_GRI*(-v_c_GRI + v_pss_GRI + v_ref_GRI) - v_f_GRI
        struct[0].g[19,0] = -K_sec_GRI*xi_m_GRI - p_m_ref_GRI - (omega_GRI - 1)/Droop_GRI
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_GRI
        struct[0].h[1,0] = V_POI
        struct[0].h[2,0] = V_PMV
        struct[0].h[3,0] = V_GR1
        struct[0].h[4,0] = V_GR2
        struct[0].h[5,0] = V_GR3
        struct[0].h[6,0] = V_GR4
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_GRI
        struct[0].Fx_ini[0,1] = Omega_b_GRI
        struct[0].Fx_ini[1,0] = (-V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fx_ini[1,1] = -D_GRI/(2*H_GRI)
        struct[0].Fx_ini[1,5] = 1/(2*H_GRI)
        struct[0].Fx_ini[2,2] = -1/T1d0_GRI
        struct[0].Fx_ini[3,3] = -1/T1q0_GRI
        struct[0].Fx_ini[4,4] = -1/T_r_GRI
        struct[0].Fx_ini[5,5] = -1/T_m_GRI
        struct[0].Fx_ini[6,1] = 1

    if mode == 11:

        struct[0].Fy_ini[1,0] = (-i_d_GRI*sin(delta_GRI - theta_GRI) - i_q_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[1,1] = (V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[1,14] = (-2*R_a_GRI*i_d_GRI - V_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[1,15] = (-2*R_a_GRI*i_q_GRI - V_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI) 
        struct[0].Fy_ini[2,14] = (X1d_GRI - X_d_GRI)/T1d0_GRI 
        struct[0].Fy_ini[2,18] = 1/T1d0_GRI 
        struct[0].Fy_ini[3,15] = (-X1q_GRI + X_q_GRI)/T1q0_GRI 
        struct[0].Fy_ini[4,0] = 1/T_r_GRI 
        struct[0].Fy_ini[5,19] = 1/T_m_GRI 

        struct[0].Gy_ini[0,0] = 2*V_GRI*g_GRI_POI + V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,1] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,2] = V_GRI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,3] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[0,16] = -1/S_base
        struct[0].Gy_ini[1,0] = -2*V_GRI*b_GRI_POI + V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[1,17] = -1/S_base
        struct[0].Gy_ini[2,0] = V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[2,1] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[2,2] = V_GRI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + 2*V_POI*(g_GRI_POI + g_POI_PMV)
        struct[0].Gy_ini[2,3] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[2,4] = V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[2,5] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[3,0] = V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy_ini[3,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy_ini[3,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + 2*V_POI*(-b_GRI_POI - b_POI_PMV)
        struct[0].Gy_ini[3,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[3,4] = V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[3,5] = V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,2] = V_PMV*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,3] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,4] = V_GR1*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + 2*V_PMV*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,5] = V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[4,6] = V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[4,7] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[4,10] = V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[4,11] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[5,2] = V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,3] = V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + 2*V_PMV*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy_ini[5,6] = V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[5,7] = V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[5,10] = V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[5,11] = V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[6,4] = V_GR1*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,5] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,6] = 2*V_GR1*(g_GR1_GR2 + g_PMV_GR1) + V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,7] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[6,8] = V_GR1*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[6,9] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[7,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,6] = 2*V_GR1*(-b_GR1_GR2 - b_PMV_GR1) + V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy_ini[7,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[7,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[8,6] = V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[8,7] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[8,8] = V_GR1*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + 2*V_GR2*g_GR1_GR2
        struct[0].Gy_ini[8,9] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[9,6] = V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[9,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[9,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - 2*V_GR2*b_GR1_GR2
        struct[0].Gy_ini[9,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy_ini[10,4] = V_GR3*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,5] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,10] = 2*V_GR3*(g_GR3_GR4 + g_PMV_GR3) + V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,11] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[10,12] = V_GR3*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[10,13] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[11,4] = V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,5] = V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,10] = 2*V_GR3*(-b_GR3_GR4 - b_PMV_GR3) + V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy_ini[11,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[11,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[12,10] = V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[12,11] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[12,12] = V_GR3*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + 2*V_GR4*g_GR3_GR4
        struct[0].Gy_ini[12,13] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[13,10] = V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[13,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[13,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - 2*V_GR4*b_GR3_GR4
        struct[0].Gy_ini[13,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy_ini[14,0] = cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[14,1] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[14,14] = X1d_GRI
        struct[0].Gy_ini[14,15] = R_a_GRI
        struct[0].Gy_ini[15,0] = sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[15,1] = -V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[15,14] = R_a_GRI
        struct[0].Gy_ini[15,15] = -X1q_GRI
        struct[0].Gy_ini[16,0] = i_d_GRI*sin(delta_GRI - theta_GRI) + i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,1] = -V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,14] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,15] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[16,16] = -1/S_n_GRI
        struct[0].Gy_ini[17,0] = i_d_GRI*cos(delta_GRI - theta_GRI) - i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,1] = V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,14] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,15] = -V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy_ini[17,17] = -1/S_n_GRI
        struct[0].Gy_ini[18,18] = -1
        struct[0].Gy_ini[19,19] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_GRI_POI = struct[0].g_GRI_POI
    b_GRI_POI = struct[0].b_GRI_POI
    g_POI_PMV = struct[0].g_POI_PMV
    b_POI_PMV = struct[0].b_POI_PMV
    g_PMV_GR1 = struct[0].g_PMV_GR1
    b_PMV_GR1 = struct[0].b_PMV_GR1
    g_GR1_GR2 = struct[0].g_GR1_GR2
    b_GR1_GR2 = struct[0].b_GR1_GR2
    g_PMV_GR3 = struct[0].g_PMV_GR3
    b_PMV_GR3 = struct[0].b_PMV_GR3
    g_GR3_GR4 = struct[0].g_GR3_GR4
    b_GR3_GR4 = struct[0].b_GR3_GR4
    U_GRI_n = struct[0].U_GRI_n
    U_POI_n = struct[0].U_POI_n
    U_PMV_n = struct[0].U_PMV_n
    U_GR1_n = struct[0].U_GR1_n
    U_GR2_n = struct[0].U_GR2_n
    U_GR3_n = struct[0].U_GR3_n
    U_GR4_n = struct[0].U_GR4_n
    S_n_GRI = struct[0].S_n_GRI
    X_d_GRI = struct[0].X_d_GRI
    X1d_GRI = struct[0].X1d_GRI
    T1d0_GRI = struct[0].T1d0_GRI
    X_q_GRI = struct[0].X_q_GRI
    X1q_GRI = struct[0].X1q_GRI
    T1q0_GRI = struct[0].T1q0_GRI
    R_a_GRI = struct[0].R_a_GRI
    X_l_GRI = struct[0].X_l_GRI
    H_GRI = struct[0].H_GRI
    D_GRI = struct[0].D_GRI
    Omega_b_GRI = struct[0].Omega_b_GRI
    omega_s_GRI = struct[0].omega_s_GRI
    K_a_GRI = struct[0].K_a_GRI
    T_r_GRI = struct[0].T_r_GRI
    v_pss_GRI = struct[0].v_pss_GRI
    Droop_GRI = struct[0].Droop_GRI
    T_m_GRI = struct[0].T_m_GRI
    K_sec_GRI = struct[0].K_sec_GRI
    K_delta_GRI = struct[0].K_delta_GRI
    v_ref_GRI = struct[0].v_ref_GRI
    
    # Inputs:
    P_GRI = struct[0].P_GRI
    Q_GRI = struct[0].Q_GRI
    P_POI = struct[0].P_POI
    Q_POI = struct[0].Q_POI
    P_PMV = struct[0].P_PMV
    Q_PMV = struct[0].Q_PMV
    P_GR1 = struct[0].P_GR1
    Q_GR1 = struct[0].Q_GR1
    P_GR2 = struct[0].P_GR2
    Q_GR2 = struct[0].Q_GR2
    P_GR3 = struct[0].P_GR3
    Q_GR3 = struct[0].Q_GR3
    P_GR4 = struct[0].P_GR4
    Q_GR4 = struct[0].Q_GR4
    
    # Dynamical states:
    delta_GRI = struct[0].x[0,0]
    omega_GRI = struct[0].x[1,0]
    e1q_GRI = struct[0].x[2,0]
    e1d_GRI = struct[0].x[3,0]
    v_c_GRI = struct[0].x[4,0]
    p_m_GRI = struct[0].x[5,0]
    xi_m_GRI = struct[0].x[6,0]
    
    # Algebraic states:
    V_GRI = struct[0].y_run[0,0]
    theta_GRI = struct[0].y_run[1,0]
    V_POI = struct[0].y_run[2,0]
    theta_POI = struct[0].y_run[3,0]
    V_PMV = struct[0].y_run[4,0]
    theta_PMV = struct[0].y_run[5,0]
    V_GR1 = struct[0].y_run[6,0]
    theta_GR1 = struct[0].y_run[7,0]
    V_GR2 = struct[0].y_run[8,0]
    theta_GR2 = struct[0].y_run[9,0]
    V_GR3 = struct[0].y_run[10,0]
    theta_GR3 = struct[0].y_run[11,0]
    V_GR4 = struct[0].y_run[12,0]
    theta_GR4 = struct[0].y_run[13,0]
    i_d_GRI = struct[0].y_run[14,0]
    i_q_GRI = struct[0].y_run[15,0]
    P_GRI_1 = struct[0].y_run[16,0]
    Q_GRI_1 = struct[0].y_run[17,0]
    v_f_GRI = struct[0].y_run[18,0]
    p_m_ref_GRI = struct[0].y_run[19,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_GRI*delta_GRI + Omega_b_GRI*(omega_GRI - omega_s_GRI)
        struct[0].f[1,0] = (-D_GRI*(omega_GRI - omega_s_GRI) - i_d_GRI*(R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI)) - i_q_GRI*(R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI)) + p_m_GRI)/(2*H_GRI)
        struct[0].f[2,0] = (-e1q_GRI - i_d_GRI*(-X1d_GRI + X_d_GRI) + v_f_GRI)/T1d0_GRI
        struct[0].f[3,0] = (-e1d_GRI + i_q_GRI*(-X1q_GRI + X_q_GRI))/T1q0_GRI
        struct[0].f[4,0] = (V_GRI - v_c_GRI)/T_r_GRI
        struct[0].f[5,0] = (-p_m_GRI + p_m_ref_GRI)/T_m_GRI
        struct[0].f[6,0] = omega_GRI - 1
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_GRI/S_base - P_GRI_1/S_base + V_GRI**2*g_GRI_POI + V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].g[1,0] = -Q_GRI/S_base - Q_GRI_1/S_base - V_GRI**2*b_GRI_POI + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].g[2,0] = -P_POI/S_base + V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + V_POI**2*(g_GRI_POI + g_POI_PMV)
        struct[0].g[3,0] = -Q_POI/S_base + V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + V_POI**2*(-b_GRI_POI - b_POI_PMV)
        struct[0].g[4,0] = -P_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV**2*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].g[5,0] = -Q_PMV/S_base + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV**2*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].g[6,0] = -P_GR1/S_base + V_GR1**2*(g_GR1_GR2 + g_PMV_GR1) + V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].g[7,0] = -Q_GR1/S_base + V_GR1**2*(-b_GR1_GR2 - b_PMV_GR1) + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].g[8,0] = -P_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR2**2*g_GR1_GR2
        struct[0].g[9,0] = -Q_GR2/S_base + V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - V_GR2**2*b_GR1_GR2
        struct[0].g[10,0] = -P_GR3/S_base + V_GR3**2*(g_GR3_GR4 + g_PMV_GR3) + V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].g[11,0] = -Q_GR3/S_base + V_GR3**2*(-b_GR3_GR4 - b_PMV_GR3) + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].g[12,0] = -P_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR4**2*g_GR3_GR4
        struct[0].g[13,0] = -Q_GR4/S_base + V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - V_GR4**2*b_GR3_GR4
        struct[0].g[14,0] = R_a_GRI*i_q_GRI + V_GRI*cos(delta_GRI - theta_GRI) + X1d_GRI*i_d_GRI - e1q_GRI
        struct[0].g[15,0] = R_a_GRI*i_d_GRI + V_GRI*sin(delta_GRI - theta_GRI) - X1q_GRI*i_q_GRI - e1d_GRI
        struct[0].g[16,0] = -P_GRI_1/S_n_GRI + V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].g[17,0] = -Q_GRI_1/S_n_GRI + V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].g[18,0] = K_a_GRI*(-v_c_GRI + v_pss_GRI + v_ref_GRI) - v_f_GRI
        struct[0].g[19,0] = -K_sec_GRI*xi_m_GRI - p_m_ref_GRI - (omega_GRI - 1)/Droop_GRI
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_GRI
        struct[0].h[1,0] = V_POI
        struct[0].h[2,0] = V_PMV
        struct[0].h[3,0] = V_GR1
        struct[0].h[4,0] = V_GR2
        struct[0].h[5,0] = V_GR3
        struct[0].h[6,0] = V_GR4
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_GRI
        struct[0].Fx[0,1] = Omega_b_GRI
        struct[0].Fx[1,0] = (-V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fx[1,1] = -D_GRI/(2*H_GRI)
        struct[0].Fx[1,5] = 1/(2*H_GRI)
        struct[0].Fx[2,2] = -1/T1d0_GRI
        struct[0].Fx[3,3] = -1/T1q0_GRI
        struct[0].Fx[4,4] = -1/T_r_GRI
        struct[0].Fx[5,5] = -1/T_m_GRI
        struct[0].Fx[6,1] = 1

    if mode == 11:

        struct[0].Fy[1,0] = (-i_d_GRI*sin(delta_GRI - theta_GRI) - i_q_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[1,1] = (V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) - V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[1,14] = (-2*R_a_GRI*i_d_GRI - V_GRI*sin(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[1,15] = (-2*R_a_GRI*i_q_GRI - V_GRI*cos(delta_GRI - theta_GRI))/(2*H_GRI)
        struct[0].Fy[2,14] = (X1d_GRI - X_d_GRI)/T1d0_GRI
        struct[0].Fy[2,18] = 1/T1d0_GRI
        struct[0].Fy[3,15] = (-X1q_GRI + X_q_GRI)/T1q0_GRI
        struct[0].Fy[4,0] = 1/T_r_GRI
        struct[0].Fy[5,19] = 1/T_m_GRI

        struct[0].Gy[0,0] = 2*V_GRI*g_GRI_POI + V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[0,1] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[0,2] = V_GRI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[0,3] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[0,16] = -1/S_base
        struct[0].Gy[1,0] = -2*V_GRI*b_GRI_POI + V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[1,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[1,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[1,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[1,17] = -1/S_base
        struct[0].Gy[2,0] = V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[2,1] = V_GRI*V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[2,2] = V_GRI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI)) + 2*V_POI*(g_GRI_POI + g_POI_PMV)
        struct[0].Gy[2,3] = V_GRI*V_POI*(-b_GRI_POI*cos(theta_GRI - theta_POI) - g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[2,4] = V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[2,5] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[3,0] = V_POI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI))
        struct[0].Gy[3,1] = V_GRI*V_POI*(-b_GRI_POI*sin(theta_GRI - theta_POI) + g_GRI_POI*cos(theta_GRI - theta_POI))
        struct[0].Gy[3,2] = V_GRI*(b_GRI_POI*cos(theta_GRI - theta_POI) + g_GRI_POI*sin(theta_GRI - theta_POI)) + V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI)) + 2*V_POI*(-b_GRI_POI - b_POI_PMV)
        struct[0].Gy[3,3] = V_GRI*V_POI*(b_GRI_POI*sin(theta_GRI - theta_POI) - g_GRI_POI*cos(theta_GRI - theta_POI)) + V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[3,4] = V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[3,5] = V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[4,2] = V_PMV*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[4,3] = V_PMV*V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[4,4] = V_GR1*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + 2*V_PMV*(g_PMV_GR1 + g_PMV_GR3 + g_POI_PMV) + V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[4,5] = V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*cos(theta_PMV - theta_POI) + g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[4,6] = V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[4,7] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[4,10] = V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[4,11] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[5,2] = V_PMV*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[5,3] = V_PMV*V_POI*(b_POI_PMV*sin(theta_PMV - theta_POI) + g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[5,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV)) + V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV)) + 2*V_PMV*(-b_PMV_GR1 - b_PMV_GR3 - b_POI_PMV) + V_POI*(b_POI_PMV*cos(theta_PMV - theta_POI) - g_POI_PMV*sin(theta_PMV - theta_POI))
        struct[0].Gy[5,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV)) + V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV)) + V_PMV*V_POI*(-b_POI_PMV*sin(theta_PMV - theta_POI) - g_POI_PMV*cos(theta_PMV - theta_POI))
        struct[0].Gy[5,6] = V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[5,7] = V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[5,10] = V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[5,11] = V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[6,4] = V_GR1*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[6,5] = V_GR1*V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[6,6] = 2*V_GR1*(g_GR1_GR2 + g_PMV_GR1) + V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[6,7] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*cos(theta_GR1 - theta_PMV) + g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[6,8] = V_GR1*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[6,9] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[7,4] = V_GR1*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[7,5] = V_GR1*V_PMV*(b_PMV_GR1*sin(theta_GR1 - theta_PMV) + g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[7,6] = 2*V_GR1*(-b_GR1_GR2 - b_PMV_GR1) + V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2)) + V_PMV*(b_PMV_GR1*cos(theta_GR1 - theta_PMV) - g_PMV_GR1*sin(theta_GR1 - theta_PMV))
        struct[0].Gy[7,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + V_GR1*V_PMV*(-b_PMV_GR1*sin(theta_GR1 - theta_PMV) - g_PMV_GR1*cos(theta_GR1 - theta_PMV))
        struct[0].Gy[7,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[7,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[8,6] = V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[8,7] = V_GR1*V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[8,8] = V_GR1*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2)) + 2*V_GR2*g_GR1_GR2
        struct[0].Gy[8,9] = V_GR1*V_GR2*(-b_GR1_GR2*cos(theta_GR1 - theta_GR2) - g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[9,6] = V_GR2*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2))
        struct[0].Gy[9,7] = V_GR1*V_GR2*(-b_GR1_GR2*sin(theta_GR1 - theta_GR2) + g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[9,8] = V_GR1*(b_GR1_GR2*cos(theta_GR1 - theta_GR2) + g_GR1_GR2*sin(theta_GR1 - theta_GR2)) - 2*V_GR2*b_GR1_GR2
        struct[0].Gy[9,9] = V_GR1*V_GR2*(b_GR1_GR2*sin(theta_GR1 - theta_GR2) - g_GR1_GR2*cos(theta_GR1 - theta_GR2))
        struct[0].Gy[10,4] = V_GR3*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[10,5] = V_GR3*V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[10,10] = 2*V_GR3*(g_GR3_GR4 + g_PMV_GR3) + V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[10,11] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*cos(theta_GR3 - theta_PMV) + g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[10,12] = V_GR3*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[10,13] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[11,4] = V_GR3*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[11,5] = V_GR3*V_PMV*(b_PMV_GR3*sin(theta_GR3 - theta_PMV) + g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[11,10] = 2*V_GR3*(-b_GR3_GR4 - b_PMV_GR3) + V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4)) + V_PMV*(b_PMV_GR3*cos(theta_GR3 - theta_PMV) - g_PMV_GR3*sin(theta_GR3 - theta_PMV))
        struct[0].Gy[11,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + V_GR3*V_PMV*(-b_PMV_GR3*sin(theta_GR3 - theta_PMV) - g_PMV_GR3*cos(theta_GR3 - theta_PMV))
        struct[0].Gy[11,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[11,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[12,10] = V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[12,11] = V_GR3*V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[12,12] = V_GR3*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4)) + 2*V_GR4*g_GR3_GR4
        struct[0].Gy[12,13] = V_GR3*V_GR4*(-b_GR3_GR4*cos(theta_GR3 - theta_GR4) - g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[13,10] = V_GR4*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4))
        struct[0].Gy[13,11] = V_GR3*V_GR4*(-b_GR3_GR4*sin(theta_GR3 - theta_GR4) + g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[13,12] = V_GR3*(b_GR3_GR4*cos(theta_GR3 - theta_GR4) + g_GR3_GR4*sin(theta_GR3 - theta_GR4)) - 2*V_GR4*b_GR3_GR4
        struct[0].Gy[13,13] = V_GR3*V_GR4*(b_GR3_GR4*sin(theta_GR3 - theta_GR4) - g_GR3_GR4*cos(theta_GR3 - theta_GR4))
        struct[0].Gy[14,0] = cos(delta_GRI - theta_GRI)
        struct[0].Gy[14,1] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[14,14] = X1d_GRI
        struct[0].Gy[14,15] = R_a_GRI
        struct[0].Gy[15,0] = sin(delta_GRI - theta_GRI)
        struct[0].Gy[15,1] = -V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[15,14] = R_a_GRI
        struct[0].Gy[15,15] = -X1q_GRI
        struct[0].Gy[16,0] = i_d_GRI*sin(delta_GRI - theta_GRI) + i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[16,1] = -V_GRI*i_d_GRI*cos(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[16,14] = V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[16,15] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[16,16] = -1/S_n_GRI
        struct[0].Gy[17,0] = i_d_GRI*cos(delta_GRI - theta_GRI) - i_q_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[17,1] = V_GRI*i_d_GRI*sin(delta_GRI - theta_GRI) + V_GRI*i_q_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[17,14] = V_GRI*cos(delta_GRI - theta_GRI)
        struct[0].Gy[17,15] = -V_GRI*sin(delta_GRI - theta_GRI)
        struct[0].Gy[17,17] = -1/S_n_GRI
        struct[0].Gy[18,18] = -1
        struct[0].Gy[19,19] = -1

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


