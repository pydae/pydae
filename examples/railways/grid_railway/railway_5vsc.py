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


class railway_5vsc_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 5
        self.N_y = 69 
        self.N_z = 10 
        self.N_store = 10000 
        self.params_list = ['R_1112', 'R_1213', 'R_1314', 'R_1415', 'R_1521', 'R_2122', 'R_2223', 'R_2324', 'R_2425', 'R_2531', 'R_3132', 'R_3233', 'R_3334', 'R_3435', 'R_3541', 'R_4142', 'R_4243', 'R_4344', 'R_4445', 'R_4551', 'R_5152', 'R_5253', 'R_5354', 'R_5455', 'p_11', 'p_12', 'p_14', 'p_15', 'p_21', 'p_22', 'p_24', 'p_25', 'p_31', 'p_32', 'p_34', 'p_35', 'p_41', 'p_42', 'p_44', 'p_45', 'p_51', 'p_52', 'p_54', 'p_55'] 
        self.params_values_list  = [0.06306666666666667, 0.06306666666666667, 0.07961686626133334, 0.008762450101333334, 0.008762450101333334, 0.008762450101333334, 0.008762450101333334, 0.018346666666666667, 0.018346666666666667, 0.018346666666666667, 0.018346666666666667, 0.018346666666666667, 0.029813333333333334, 0.029813333333333334, 0.029813333333333334, 0.029813333333333334, 0.029813333333333334, 0.07803063134933337, 0.02922567549599999, 0.02922567549599999, 0.02922567549599999, 0.02922567549599999, 0.0344, 0.0344, 0.0, 0.0, -1932995.075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1438308.138, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.inputs_ini_list = ['Dv_r_13', 'Dv_r_23', 'Dv_r_33', 'Dv_r_43', 'Dv_r_53', 'v_nom', 'T_v', 'K_r'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.02, 0.0003] 
        self.inputs_run_list = ['Dv_r_13', 'Dv_r_23', 'Dv_r_33', 'Dv_r_43', 'Dv_r_53', 'v_nom', 'T_v', 'K_r'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.02, 0.0003] 
        self.outputs_list = ['p_13', 'v_13', 'p_23', 'v_23', 'p_33', 'v_33', 'p_43', 'v_43', 'p_53', 'v_53'] 
        self.x_list = ['v_13', 'v_23', 'v_33', 'v_43', 'v_53'] 
        self.y_run_list = ['i_l_1112', 'i_l_1213', 'i_l_1314', 'i_l_1415', 'i_l_2122', 'i_l_2223', 'i_l_2324', 'i_l_2425', 'i_l_3132', 'i_l_3233', 'i_l_3334', 'i_l_3435', 'i_l_4142', 'i_l_4243', 'i_l_4344', 'i_l_4445', 'i_l_5152', 'i_l_5253', 'i_l_5354', 'i_l_5455', 'i_l_1521', 'i_l_2531', 'i_l_3541', 'i_l_4551', 'v_11', 'v_12', 'i_13', 'v_14', 'v_15', 'v_21', 'v_22', 'i_23', 'v_24', 'v_25', 'v_31', 'v_32', 'i_33', 'v_34', 'v_35', 'v_41', 'v_42', 'i_43', 'v_44', 'v_45', 'v_51', 'v_52', 'i_53', 'v_54', 'v_55', 'i_11', 'i_12', 'i_14', 'i_15', 'i_21', 'i_22', 'i_24', 'i_25', 'i_31', 'i_32', 'i_34', 'i_35', 'i_41', 'i_42', 'i_44', 'i_45', 'i_51', 'i_52', 'i_54', 'i_55'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['i_l_1112', 'i_l_1213', 'i_l_1314', 'i_l_1415', 'i_l_2122', 'i_l_2223', 'i_l_2324', 'i_l_2425', 'i_l_3132', 'i_l_3233', 'i_l_3334', 'i_l_3435', 'i_l_4142', 'i_l_4243', 'i_l_4344', 'i_l_4445', 'i_l_5152', 'i_l_5253', 'i_l_5354', 'i_l_5455', 'i_l_1521', 'i_l_2531', 'i_l_3541', 'i_l_4551', 'v_11', 'v_12', 'i_13', 'v_14', 'v_15', 'v_21', 'v_22', 'i_23', 'v_24', 'v_25', 'v_31', 'v_32', 'i_33', 'v_34', 'v_35', 'v_41', 'v_42', 'i_43', 'v_44', 'v_45', 'v_51', 'v_52', 'i_53', 'v_54', 'v_55', 'i_11', 'i_12', 'i_14', 'i_15', 'i_21', 'i_22', 'i_24', 'i_25', 'i_31', 'i_32', 'i_34', 'i_35', 'i_41', 'i_42', 'i_44', 'i_45', 'i_51', 'i_52', 'i_54', 'i_55'] 
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
    R_1112 = struct[0].R_1112
    R_1213 = struct[0].R_1213
    R_1314 = struct[0].R_1314
    R_1415 = struct[0].R_1415
    R_1521 = struct[0].R_1521
    R_2122 = struct[0].R_2122
    R_2223 = struct[0].R_2223
    R_2324 = struct[0].R_2324
    R_2425 = struct[0].R_2425
    R_2531 = struct[0].R_2531
    R_3132 = struct[0].R_3132
    R_3233 = struct[0].R_3233
    R_3334 = struct[0].R_3334
    R_3435 = struct[0].R_3435
    R_3541 = struct[0].R_3541
    R_4142 = struct[0].R_4142
    R_4243 = struct[0].R_4243
    R_4344 = struct[0].R_4344
    R_4445 = struct[0].R_4445
    R_4551 = struct[0].R_4551
    R_5152 = struct[0].R_5152
    R_5253 = struct[0].R_5253
    R_5354 = struct[0].R_5354
    R_5455 = struct[0].R_5455
    p_11 = struct[0].p_11
    p_12 = struct[0].p_12
    p_14 = struct[0].p_14
    p_15 = struct[0].p_15
    p_21 = struct[0].p_21
    p_22 = struct[0].p_22
    p_24 = struct[0].p_24
    p_25 = struct[0].p_25
    p_31 = struct[0].p_31
    p_32 = struct[0].p_32
    p_34 = struct[0].p_34
    p_35 = struct[0].p_35
    p_41 = struct[0].p_41
    p_42 = struct[0].p_42
    p_44 = struct[0].p_44
    p_45 = struct[0].p_45
    p_51 = struct[0].p_51
    p_52 = struct[0].p_52
    p_54 = struct[0].p_54
    p_55 = struct[0].p_55
    
    # Inputs:
    Dv_r_13 = struct[0].Dv_r_13
    Dv_r_23 = struct[0].Dv_r_23
    Dv_r_33 = struct[0].Dv_r_33
    Dv_r_43 = struct[0].Dv_r_43
    Dv_r_53 = struct[0].Dv_r_53
    v_nom = struct[0].v_nom
    T_v = struct[0].T_v
    K_r = struct[0].K_r
    
    # Dynamical states:
    v_13 = struct[0].x[0,0]
    v_23 = struct[0].x[1,0]
    v_33 = struct[0].x[2,0]
    v_43 = struct[0].x[3,0]
    v_53 = struct[0].x[4,0]
    
    # Algebraic states:
    i_l_1112 = struct[0].y_ini[0,0]
    i_l_1213 = struct[0].y_ini[1,0]
    i_l_1314 = struct[0].y_ini[2,0]
    i_l_1415 = struct[0].y_ini[3,0]
    i_l_2122 = struct[0].y_ini[4,0]
    i_l_2223 = struct[0].y_ini[5,0]
    i_l_2324 = struct[0].y_ini[6,0]
    i_l_2425 = struct[0].y_ini[7,0]
    i_l_3132 = struct[0].y_ini[8,0]
    i_l_3233 = struct[0].y_ini[9,0]
    i_l_3334 = struct[0].y_ini[10,0]
    i_l_3435 = struct[0].y_ini[11,0]
    i_l_4142 = struct[0].y_ini[12,0]
    i_l_4243 = struct[0].y_ini[13,0]
    i_l_4344 = struct[0].y_ini[14,0]
    i_l_4445 = struct[0].y_ini[15,0]
    i_l_5152 = struct[0].y_ini[16,0]
    i_l_5253 = struct[0].y_ini[17,0]
    i_l_5354 = struct[0].y_ini[18,0]
    i_l_5455 = struct[0].y_ini[19,0]
    i_l_1521 = struct[0].y_ini[20,0]
    i_l_2531 = struct[0].y_ini[21,0]
    i_l_3541 = struct[0].y_ini[22,0]
    i_l_4551 = struct[0].y_ini[23,0]
    v_11 = struct[0].y_ini[24,0]
    v_12 = struct[0].y_ini[25,0]
    i_13 = struct[0].y_ini[26,0]
    v_14 = struct[0].y_ini[27,0]
    v_15 = struct[0].y_ini[28,0]
    v_21 = struct[0].y_ini[29,0]
    v_22 = struct[0].y_ini[30,0]
    i_23 = struct[0].y_ini[31,0]
    v_24 = struct[0].y_ini[32,0]
    v_25 = struct[0].y_ini[33,0]
    v_31 = struct[0].y_ini[34,0]
    v_32 = struct[0].y_ini[35,0]
    i_33 = struct[0].y_ini[36,0]
    v_34 = struct[0].y_ini[37,0]
    v_35 = struct[0].y_ini[38,0]
    v_41 = struct[0].y_ini[39,0]
    v_42 = struct[0].y_ini[40,0]
    i_43 = struct[0].y_ini[41,0]
    v_44 = struct[0].y_ini[42,0]
    v_45 = struct[0].y_ini[43,0]
    v_51 = struct[0].y_ini[44,0]
    v_52 = struct[0].y_ini[45,0]
    i_53 = struct[0].y_ini[46,0]
    v_54 = struct[0].y_ini[47,0]
    v_55 = struct[0].y_ini[48,0]
    i_11 = struct[0].y_ini[49,0]
    i_12 = struct[0].y_ini[50,0]
    i_14 = struct[0].y_ini[51,0]
    i_15 = struct[0].y_ini[52,0]
    i_21 = struct[0].y_ini[53,0]
    i_22 = struct[0].y_ini[54,0]
    i_24 = struct[0].y_ini[55,0]
    i_25 = struct[0].y_ini[56,0]
    i_31 = struct[0].y_ini[57,0]
    i_32 = struct[0].y_ini[58,0]
    i_34 = struct[0].y_ini[59,0]
    i_35 = struct[0].y_ini[60,0]
    i_41 = struct[0].y_ini[61,0]
    i_42 = struct[0].y_ini[62,0]
    i_44 = struct[0].y_ini[63,0]
    i_45 = struct[0].y_ini[64,0]
    i_51 = struct[0].y_ini[65,0]
    i_52 = struct[0].y_ini[66,0]
    i_54 = struct[0].y_ini[67,0]
    i_55 = struct[0].y_ini[68,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-Dv_r_13 - K_r*i_13*v_13 - v_13 + v_nom)/T_v
        struct[0].f[1,0] = (-Dv_r_23 - K_r*i_23*v_23 - v_23 + v_nom)/T_v
        struct[0].f[2,0] = (-Dv_r_33 - K_r*i_33*v_33 - v_33 + v_nom)/T_v
        struct[0].f[3,0] = (-Dv_r_43 - K_r*i_43*v_43 - v_43 + v_nom)/T_v
        struct[0].f[4,0] = (-Dv_r_53 - K_r*i_53*v_53 - v_53 + v_nom)/T_v
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[1,0] = -R_1213*i_l_1213 + v_12 - v_13
        struct[0].g[2,0] = -R_1314*i_l_1314 + v_13 - v_14
        struct[0].g[5,0] = -R_2223*i_l_2223 + v_22 - v_23
        struct[0].g[6,0] = -R_2324*i_l_2324 + v_23 - v_24
        struct[0].g[9,0] = -R_3233*i_l_3233 + v_32 - v_33
        struct[0].g[10,0] = -R_3334*i_l_3334 + v_33 - v_34
        struct[0].g[13,0] = -R_4243*i_l_4243 + v_42 - v_43
        struct[0].g[14,0] = -R_4344*i_l_4344 + v_43 - v_44
        struct[0].g[17,0] = -R_5253*i_l_5253 + v_52 - v_53
        struct[0].g[18,0] = -R_5354*i_l_5354 + v_53 - v_54
        struct[0].g[49,0] = i_11*v_11 - p_11
        struct[0].g[50,0] = i_12*v_12 - p_12
        struct[0].g[51,0] = i_14*v_14 - p_14
        struct[0].g[52,0] = i_15*v_15 - p_15
        struct[0].g[53,0] = i_21*v_21 - p_21
        struct[0].g[54,0] = i_22*v_22 - p_22
        struct[0].g[55,0] = i_24*v_24 - p_24
        struct[0].g[56,0] = i_25*v_25 - p_25
        struct[0].g[57,0] = i_31*v_31 - p_31
        struct[0].g[58,0] = i_32*v_32 - p_32
        struct[0].g[59,0] = i_34*v_34 - p_34
        struct[0].g[60,0] = i_35*v_35 - p_35
        struct[0].g[61,0] = i_41*v_41 - p_41
        struct[0].g[62,0] = i_42*v_42 - p_42
        struct[0].g[63,0] = i_44*v_44 - p_44
        struct[0].g[64,0] = i_45*v_45 - p_45
        struct[0].g[65,0] = i_51*v_51 - p_51
        struct[0].g[66,0] = i_52*v_52 - p_52
        struct[0].g[67,0] = i_54*v_54 - p_54
        struct[0].g[68,0] = i_55*v_55 - p_55
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_13*v_13
        struct[0].h[1,0] = v_13
        struct[0].h[2,0] = i_23*v_23
        struct[0].h[3,0] = v_23
        struct[0].h[4,0] = i_33*v_33
        struct[0].h[5,0] = v_33
        struct[0].h[6,0] = i_43*v_43
        struct[0].h[7,0] = v_43
        struct[0].h[8,0] = i_53*v_53
        struct[0].h[9,0] = v_53
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = (-K_r*i_13 - 1)/T_v
        struct[0].Fx_ini[1,1] = (-K_r*i_23 - 1)/T_v
        struct[0].Fx_ini[2,2] = (-K_r*i_33 - 1)/T_v
        struct[0].Fx_ini[3,3] = (-K_r*i_43 - 1)/T_v
        struct[0].Fx_ini[4,4] = (-K_r*i_53 - 1)/T_v

    if mode == 11:

        struct[0].Fy_ini[0,26] = -K_r*v_13/T_v 
        struct[0].Fy_ini[1,31] = -K_r*v_23/T_v 
        struct[0].Fy_ini[2,36] = -K_r*v_33/T_v 
        struct[0].Fy_ini[3,41] = -K_r*v_43/T_v 
        struct[0].Fy_ini[4,46] = -K_r*v_53/T_v 

        struct[0].Gx_ini[1,0] = -1
        struct[0].Gx_ini[2,0] = 1
        struct[0].Gx_ini[5,1] = -1
        struct[0].Gx_ini[6,1] = 1
        struct[0].Gx_ini[9,2] = -1
        struct[0].Gx_ini[10,2] = 1
        struct[0].Gx_ini[13,3] = -1
        struct[0].Gx_ini[14,3] = 1
        struct[0].Gx_ini[17,4] = -1
        struct[0].Gx_ini[18,4] = 1

        struct[0].Gy_ini[0,0] = -R_1112
        struct[0].Gy_ini[1,1] = -R_1213
        struct[0].Gy_ini[2,2] = -R_1314
        struct[0].Gy_ini[3,3] = -R_1415
        struct[0].Gy_ini[4,4] = -R_2122
        struct[0].Gy_ini[5,5] = -R_2223
        struct[0].Gy_ini[6,6] = -R_2324
        struct[0].Gy_ini[7,7] = -R_2425
        struct[0].Gy_ini[8,8] = -R_3132
        struct[0].Gy_ini[9,9] = -R_3233
        struct[0].Gy_ini[10,10] = -R_3334
        struct[0].Gy_ini[11,11] = -R_3435
        struct[0].Gy_ini[12,12] = -R_4142
        struct[0].Gy_ini[13,13] = -R_4243
        struct[0].Gy_ini[14,14] = -R_4344
        struct[0].Gy_ini[15,15] = -R_4445
        struct[0].Gy_ini[16,16] = -R_5152
        struct[0].Gy_ini[17,17] = -R_5253
        struct[0].Gy_ini[18,18] = -R_5354
        struct[0].Gy_ini[19,19] = -R_5455
        struct[0].Gy_ini[20,20] = -R_1521
        struct[0].Gy_ini[21,21] = -R_2531
        struct[0].Gy_ini[22,22] = -R_3541
        struct[0].Gy_ini[23,23] = -R_4551
        struct[0].Gy_ini[49,24] = i_11
        struct[0].Gy_ini[49,49] = v_11
        struct[0].Gy_ini[50,25] = i_12
        struct[0].Gy_ini[50,50] = v_12
        struct[0].Gy_ini[51,27] = i_14
        struct[0].Gy_ini[51,51] = v_14
        struct[0].Gy_ini[52,28] = i_15
        struct[0].Gy_ini[52,52] = v_15
        struct[0].Gy_ini[53,29] = i_21
        struct[0].Gy_ini[53,53] = v_21
        struct[0].Gy_ini[54,30] = i_22
        struct[0].Gy_ini[54,54] = v_22
        struct[0].Gy_ini[55,32] = i_24
        struct[0].Gy_ini[55,55] = v_24
        struct[0].Gy_ini[56,33] = i_25
        struct[0].Gy_ini[56,56] = v_25
        struct[0].Gy_ini[57,34] = i_31
        struct[0].Gy_ini[57,57] = v_31
        struct[0].Gy_ini[58,35] = i_32
        struct[0].Gy_ini[58,58] = v_32
        struct[0].Gy_ini[59,37] = i_34
        struct[0].Gy_ini[59,59] = v_34
        struct[0].Gy_ini[60,38] = i_35
        struct[0].Gy_ini[60,60] = v_35
        struct[0].Gy_ini[61,39] = i_41
        struct[0].Gy_ini[61,61] = v_41
        struct[0].Gy_ini[62,40] = i_42
        struct[0].Gy_ini[62,62] = v_42
        struct[0].Gy_ini[63,42] = i_44
        struct[0].Gy_ini[63,63] = v_44
        struct[0].Gy_ini[64,43] = i_45
        struct[0].Gy_ini[64,64] = v_45
        struct[0].Gy_ini[65,44] = i_51
        struct[0].Gy_ini[65,65] = v_51
        struct[0].Gy_ini[66,45] = i_52
        struct[0].Gy_ini[66,66] = v_52
        struct[0].Gy_ini[67,47] = i_54
        struct[0].Gy_ini[67,67] = v_54
        struct[0].Gy_ini[68,48] = i_55
        struct[0].Gy_ini[68,68] = v_55



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    R_1112 = struct[0].R_1112
    R_1213 = struct[0].R_1213
    R_1314 = struct[0].R_1314
    R_1415 = struct[0].R_1415
    R_1521 = struct[0].R_1521
    R_2122 = struct[0].R_2122
    R_2223 = struct[0].R_2223
    R_2324 = struct[0].R_2324
    R_2425 = struct[0].R_2425
    R_2531 = struct[0].R_2531
    R_3132 = struct[0].R_3132
    R_3233 = struct[0].R_3233
    R_3334 = struct[0].R_3334
    R_3435 = struct[0].R_3435
    R_3541 = struct[0].R_3541
    R_4142 = struct[0].R_4142
    R_4243 = struct[0].R_4243
    R_4344 = struct[0].R_4344
    R_4445 = struct[0].R_4445
    R_4551 = struct[0].R_4551
    R_5152 = struct[0].R_5152
    R_5253 = struct[0].R_5253
    R_5354 = struct[0].R_5354
    R_5455 = struct[0].R_5455
    p_11 = struct[0].p_11
    p_12 = struct[0].p_12
    p_14 = struct[0].p_14
    p_15 = struct[0].p_15
    p_21 = struct[0].p_21
    p_22 = struct[0].p_22
    p_24 = struct[0].p_24
    p_25 = struct[0].p_25
    p_31 = struct[0].p_31
    p_32 = struct[0].p_32
    p_34 = struct[0].p_34
    p_35 = struct[0].p_35
    p_41 = struct[0].p_41
    p_42 = struct[0].p_42
    p_44 = struct[0].p_44
    p_45 = struct[0].p_45
    p_51 = struct[0].p_51
    p_52 = struct[0].p_52
    p_54 = struct[0].p_54
    p_55 = struct[0].p_55
    
    # Inputs:
    Dv_r_13 = struct[0].Dv_r_13
    Dv_r_23 = struct[0].Dv_r_23
    Dv_r_33 = struct[0].Dv_r_33
    Dv_r_43 = struct[0].Dv_r_43
    Dv_r_53 = struct[0].Dv_r_53
    v_nom = struct[0].v_nom
    T_v = struct[0].T_v
    K_r = struct[0].K_r
    
    # Dynamical states:
    v_13 = struct[0].x[0,0]
    v_23 = struct[0].x[1,0]
    v_33 = struct[0].x[2,0]
    v_43 = struct[0].x[3,0]
    v_53 = struct[0].x[4,0]
    
    # Algebraic states:
    i_l_1112 = struct[0].y_run[0,0]
    i_l_1213 = struct[0].y_run[1,0]
    i_l_1314 = struct[0].y_run[2,0]
    i_l_1415 = struct[0].y_run[3,0]
    i_l_2122 = struct[0].y_run[4,0]
    i_l_2223 = struct[0].y_run[5,0]
    i_l_2324 = struct[0].y_run[6,0]
    i_l_2425 = struct[0].y_run[7,0]
    i_l_3132 = struct[0].y_run[8,0]
    i_l_3233 = struct[0].y_run[9,0]
    i_l_3334 = struct[0].y_run[10,0]
    i_l_3435 = struct[0].y_run[11,0]
    i_l_4142 = struct[0].y_run[12,0]
    i_l_4243 = struct[0].y_run[13,0]
    i_l_4344 = struct[0].y_run[14,0]
    i_l_4445 = struct[0].y_run[15,0]
    i_l_5152 = struct[0].y_run[16,0]
    i_l_5253 = struct[0].y_run[17,0]
    i_l_5354 = struct[0].y_run[18,0]
    i_l_5455 = struct[0].y_run[19,0]
    i_l_1521 = struct[0].y_run[20,0]
    i_l_2531 = struct[0].y_run[21,0]
    i_l_3541 = struct[0].y_run[22,0]
    i_l_4551 = struct[0].y_run[23,0]
    v_11 = struct[0].y_run[24,0]
    v_12 = struct[0].y_run[25,0]
    i_13 = struct[0].y_run[26,0]
    v_14 = struct[0].y_run[27,0]
    v_15 = struct[0].y_run[28,0]
    v_21 = struct[0].y_run[29,0]
    v_22 = struct[0].y_run[30,0]
    i_23 = struct[0].y_run[31,0]
    v_24 = struct[0].y_run[32,0]
    v_25 = struct[0].y_run[33,0]
    v_31 = struct[0].y_run[34,0]
    v_32 = struct[0].y_run[35,0]
    i_33 = struct[0].y_run[36,0]
    v_34 = struct[0].y_run[37,0]
    v_35 = struct[0].y_run[38,0]
    v_41 = struct[0].y_run[39,0]
    v_42 = struct[0].y_run[40,0]
    i_43 = struct[0].y_run[41,0]
    v_44 = struct[0].y_run[42,0]
    v_45 = struct[0].y_run[43,0]
    v_51 = struct[0].y_run[44,0]
    v_52 = struct[0].y_run[45,0]
    i_53 = struct[0].y_run[46,0]
    v_54 = struct[0].y_run[47,0]
    v_55 = struct[0].y_run[48,0]
    i_11 = struct[0].y_run[49,0]
    i_12 = struct[0].y_run[50,0]
    i_14 = struct[0].y_run[51,0]
    i_15 = struct[0].y_run[52,0]
    i_21 = struct[0].y_run[53,0]
    i_22 = struct[0].y_run[54,0]
    i_24 = struct[0].y_run[55,0]
    i_25 = struct[0].y_run[56,0]
    i_31 = struct[0].y_run[57,0]
    i_32 = struct[0].y_run[58,0]
    i_34 = struct[0].y_run[59,0]
    i_35 = struct[0].y_run[60,0]
    i_41 = struct[0].y_run[61,0]
    i_42 = struct[0].y_run[62,0]
    i_44 = struct[0].y_run[63,0]
    i_45 = struct[0].y_run[64,0]
    i_51 = struct[0].y_run[65,0]
    i_52 = struct[0].y_run[66,0]
    i_54 = struct[0].y_run[67,0]
    i_55 = struct[0].y_run[68,0]
    
    struct[0].u_run[0,0] = Dv_r_13
    struct[0].u_run[1,0] = Dv_r_23
    struct[0].u_run[2,0] = Dv_r_33
    struct[0].u_run[3,0] = Dv_r_43
    struct[0].u_run[4,0] = Dv_r_53
    struct[0].u_run[5,0] = v_nom
    struct[0].u_run[6,0] = T_v
    struct[0].u_run[7,0] = K_r
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-Dv_r_13 - K_r*i_13*v_13 - v_13 + v_nom)/T_v
        struct[0].f[1,0] = (-Dv_r_23 - K_r*i_23*v_23 - v_23 + v_nom)/T_v
        struct[0].f[2,0] = (-Dv_r_33 - K_r*i_33*v_33 - v_33 + v_nom)/T_v
        struct[0].f[3,0] = (-Dv_r_43 - K_r*i_43*v_43 - v_43 + v_nom)/T_v
        struct[0].f[4,0] = (-Dv_r_53 - K_r*i_53*v_53 - v_53 + v_nom)/T_v
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[1,0] = -R_1213*i_l_1213 + v_12 - v_13
        struct[0].g[2,0] = -R_1314*i_l_1314 + v_13 - v_14
        struct[0].g[5,0] = -R_2223*i_l_2223 + v_22 - v_23
        struct[0].g[6,0] = -R_2324*i_l_2324 + v_23 - v_24
        struct[0].g[9,0] = -R_3233*i_l_3233 + v_32 - v_33
        struct[0].g[10,0] = -R_3334*i_l_3334 + v_33 - v_34
        struct[0].g[13,0] = -R_4243*i_l_4243 + v_42 - v_43
        struct[0].g[14,0] = -R_4344*i_l_4344 + v_43 - v_44
        struct[0].g[17,0] = -R_5253*i_l_5253 + v_52 - v_53
        struct[0].g[18,0] = -R_5354*i_l_5354 + v_53 - v_54
        struct[0].g[49,0] = i_11*v_11 - p_11
        struct[0].g[50,0] = i_12*v_12 - p_12
        struct[0].g[51,0] = i_14*v_14 - p_14
        struct[0].g[52,0] = i_15*v_15 - p_15
        struct[0].g[53,0] = i_21*v_21 - p_21
        struct[0].g[54,0] = i_22*v_22 - p_22
        struct[0].g[55,0] = i_24*v_24 - p_24
        struct[0].g[56,0] = i_25*v_25 - p_25
        struct[0].g[57,0] = i_31*v_31 - p_31
        struct[0].g[58,0] = i_32*v_32 - p_32
        struct[0].g[59,0] = i_34*v_34 - p_34
        struct[0].g[60,0] = i_35*v_35 - p_35
        struct[0].g[61,0] = i_41*v_41 - p_41
        struct[0].g[62,0] = i_42*v_42 - p_42
        struct[0].g[63,0] = i_44*v_44 - p_44
        struct[0].g[64,0] = i_45*v_45 - p_45
        struct[0].g[65,0] = i_51*v_51 - p_51
        struct[0].g[66,0] = i_52*v_52 - p_52
        struct[0].g[67,0] = i_54*v_54 - p_54
        struct[0].g[68,0] = i_55*v_55 - p_55
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_13*v_13
        struct[0].h[1,0] = v_13
        struct[0].h[2,0] = i_23*v_23
        struct[0].h[3,0] = v_23
        struct[0].h[4,0] = i_33*v_33
        struct[0].h[5,0] = v_33
        struct[0].h[6,0] = i_43*v_43
        struct[0].h[7,0] = v_43
        struct[0].h[8,0] = i_53*v_53
        struct[0].h[9,0] = v_53
    

    if mode == 10:

        struct[0].Fx[0,0] = (-K_r*i_13 - 1)/T_v
        struct[0].Fx[1,1] = (-K_r*i_23 - 1)/T_v
        struct[0].Fx[2,2] = (-K_r*i_33 - 1)/T_v
        struct[0].Fx[3,3] = (-K_r*i_43 - 1)/T_v
        struct[0].Fx[4,4] = (-K_r*i_53 - 1)/T_v

    if mode == 11:

        struct[0].Fy[0,26] = -K_r*v_13/T_v
        struct[0].Fy[1,31] = -K_r*v_23/T_v
        struct[0].Fy[2,36] = -K_r*v_33/T_v
        struct[0].Fy[3,41] = -K_r*v_43/T_v
        struct[0].Fy[4,46] = -K_r*v_53/T_v

        struct[0].Gx[1,0] = -1
        struct[0].Gx[2,0] = 1
        struct[0].Gx[5,1] = -1
        struct[0].Gx[6,1] = 1
        struct[0].Gx[9,2] = -1
        struct[0].Gx[10,2] = 1
        struct[0].Gx[13,3] = -1
        struct[0].Gx[14,3] = 1
        struct[0].Gx[17,4] = -1
        struct[0].Gx[18,4] = 1

        struct[0].Gy[0,0] = -R_1112
        struct[0].Gy[1,1] = -R_1213
        struct[0].Gy[2,2] = -R_1314
        struct[0].Gy[3,3] = -R_1415
        struct[0].Gy[4,4] = -R_2122
        struct[0].Gy[5,5] = -R_2223
        struct[0].Gy[6,6] = -R_2324
        struct[0].Gy[7,7] = -R_2425
        struct[0].Gy[8,8] = -R_3132
        struct[0].Gy[9,9] = -R_3233
        struct[0].Gy[10,10] = -R_3334
        struct[0].Gy[11,11] = -R_3435
        struct[0].Gy[12,12] = -R_4142
        struct[0].Gy[13,13] = -R_4243
        struct[0].Gy[14,14] = -R_4344
        struct[0].Gy[15,15] = -R_4445
        struct[0].Gy[16,16] = -R_5152
        struct[0].Gy[17,17] = -R_5253
        struct[0].Gy[18,18] = -R_5354
        struct[0].Gy[19,19] = -R_5455
        struct[0].Gy[20,20] = -R_1521
        struct[0].Gy[21,21] = -R_2531
        struct[0].Gy[22,22] = -R_3541
        struct[0].Gy[23,23] = -R_4551
        struct[0].Gy[49,24] = i_11
        struct[0].Gy[49,49] = v_11
        struct[0].Gy[50,25] = i_12
        struct[0].Gy[50,50] = v_12
        struct[0].Gy[51,27] = i_14
        struct[0].Gy[51,51] = v_14
        struct[0].Gy[52,28] = i_15
        struct[0].Gy[52,52] = v_15
        struct[0].Gy[53,29] = i_21
        struct[0].Gy[53,53] = v_21
        struct[0].Gy[54,30] = i_22
        struct[0].Gy[54,54] = v_22
        struct[0].Gy[55,32] = i_24
        struct[0].Gy[55,55] = v_24
        struct[0].Gy[56,33] = i_25
        struct[0].Gy[56,56] = v_25
        struct[0].Gy[57,34] = i_31
        struct[0].Gy[57,57] = v_31
        struct[0].Gy[58,35] = i_32
        struct[0].Gy[58,58] = v_32
        struct[0].Gy[59,37] = i_34
        struct[0].Gy[59,59] = v_34
        struct[0].Gy[60,38] = i_35
        struct[0].Gy[60,60] = v_35
        struct[0].Gy[61,39] = i_41
        struct[0].Gy[61,61] = v_41
        struct[0].Gy[62,40] = i_42
        struct[0].Gy[62,62] = v_42
        struct[0].Gy[63,42] = i_44
        struct[0].Gy[63,63] = v_44
        struct[0].Gy[64,43] = i_45
        struct[0].Gy[64,64] = v_45
        struct[0].Gy[65,44] = i_51
        struct[0].Gy[65,65] = v_51
        struct[0].Gy[66,45] = i_52
        struct[0].Gy[66,66] = v_52
        struct[0].Gy[67,47] = i_54
        struct[0].Gy[67,67] = v_54
        struct[0].Gy[68,48] = i_55
        struct[0].Gy[68,68] = v_55

    if mode > 12:

        struct[0].Fu[0,0] = -1/T_v
        struct[0].Fu[0,5] = 1/T_v
        struct[0].Fu[0,6] = -(-Dv_r_13 - K_r*i_13*v_13 - v_13 + v_nom)/T_v**2
        struct[0].Fu[0,7] = -i_13*v_13/T_v
        struct[0].Fu[1,1] = -1/T_v
        struct[0].Fu[1,5] = 1/T_v
        struct[0].Fu[1,6] = -(-Dv_r_23 - K_r*i_23*v_23 - v_23 + v_nom)/T_v**2
        struct[0].Fu[1,7] = -i_23*v_23/T_v
        struct[0].Fu[2,2] = -1/T_v
        struct[0].Fu[2,5] = 1/T_v
        struct[0].Fu[2,6] = -(-Dv_r_33 - K_r*i_33*v_33 - v_33 + v_nom)/T_v**2
        struct[0].Fu[2,7] = -i_33*v_33/T_v
        struct[0].Fu[3,3] = -1/T_v
        struct[0].Fu[3,5] = 1/T_v
        struct[0].Fu[3,6] = -(-Dv_r_43 - K_r*i_43*v_43 - v_43 + v_nom)/T_v**2
        struct[0].Fu[3,7] = -i_43*v_43/T_v
        struct[0].Fu[4,4] = -1/T_v
        struct[0].Fu[4,5] = 1/T_v
        struct[0].Fu[4,6] = -(-Dv_r_53 - K_r*i_53*v_53 - v_53 + v_nom)/T_v**2
        struct[0].Fu[4,7] = -i_53*v_53/T_v


        struct[0].Hx[0,0] = i_13
        struct[0].Hx[1,0] = 1
        struct[0].Hx[2,1] = i_23
        struct[0].Hx[3,1] = 1
        struct[0].Hx[4,2] = i_33
        struct[0].Hx[5,2] = 1
        struct[0].Hx[6,3] = i_43
        struct[0].Hx[7,3] = 1
        struct[0].Hx[8,4] = i_53
        struct[0].Hx[9,4] = 1

        struct[0].Hy[0,26] = v_13
        struct[0].Hy[2,31] = v_23
        struct[0].Hy[4,36] = v_33
        struct[0].Hy[6,41] = v_43
        struct[0].Hy[8,46] = v_53




def ini_nn(struct,mode):

    # Parameters:
    R_1112 = struct[0].R_1112
    R_1213 = struct[0].R_1213
    R_1314 = struct[0].R_1314
    R_1415 = struct[0].R_1415
    R_1521 = struct[0].R_1521
    R_2122 = struct[0].R_2122
    R_2223 = struct[0].R_2223
    R_2324 = struct[0].R_2324
    R_2425 = struct[0].R_2425
    R_2531 = struct[0].R_2531
    R_3132 = struct[0].R_3132
    R_3233 = struct[0].R_3233
    R_3334 = struct[0].R_3334
    R_3435 = struct[0].R_3435
    R_3541 = struct[0].R_3541
    R_4142 = struct[0].R_4142
    R_4243 = struct[0].R_4243
    R_4344 = struct[0].R_4344
    R_4445 = struct[0].R_4445
    R_4551 = struct[0].R_4551
    R_5152 = struct[0].R_5152
    R_5253 = struct[0].R_5253
    R_5354 = struct[0].R_5354
    R_5455 = struct[0].R_5455
    p_11 = struct[0].p_11
    p_12 = struct[0].p_12
    p_14 = struct[0].p_14
    p_15 = struct[0].p_15
    p_21 = struct[0].p_21
    p_22 = struct[0].p_22
    p_24 = struct[0].p_24
    p_25 = struct[0].p_25
    p_31 = struct[0].p_31
    p_32 = struct[0].p_32
    p_34 = struct[0].p_34
    p_35 = struct[0].p_35
    p_41 = struct[0].p_41
    p_42 = struct[0].p_42
    p_44 = struct[0].p_44
    p_45 = struct[0].p_45
    p_51 = struct[0].p_51
    p_52 = struct[0].p_52
    p_54 = struct[0].p_54
    p_55 = struct[0].p_55
    
    # Inputs:
    Dv_r_13 = struct[0].Dv_r_13
    Dv_r_23 = struct[0].Dv_r_23
    Dv_r_33 = struct[0].Dv_r_33
    Dv_r_43 = struct[0].Dv_r_43
    Dv_r_53 = struct[0].Dv_r_53
    v_nom = struct[0].v_nom
    T_v = struct[0].T_v
    K_r = struct[0].K_r
    
    # Dynamical states:
    v_13 = struct[0].x[0,0]
    v_23 = struct[0].x[1,0]
    v_33 = struct[0].x[2,0]
    v_43 = struct[0].x[3,0]
    v_53 = struct[0].x[4,0]
    
    # Algebraic states:
    i_l_1112 = struct[0].y_ini[0,0]
    i_l_1213 = struct[0].y_ini[1,0]
    i_l_1314 = struct[0].y_ini[2,0]
    i_l_1415 = struct[0].y_ini[3,0]
    i_l_2122 = struct[0].y_ini[4,0]
    i_l_2223 = struct[0].y_ini[5,0]
    i_l_2324 = struct[0].y_ini[6,0]
    i_l_2425 = struct[0].y_ini[7,0]
    i_l_3132 = struct[0].y_ini[8,0]
    i_l_3233 = struct[0].y_ini[9,0]
    i_l_3334 = struct[0].y_ini[10,0]
    i_l_3435 = struct[0].y_ini[11,0]
    i_l_4142 = struct[0].y_ini[12,0]
    i_l_4243 = struct[0].y_ini[13,0]
    i_l_4344 = struct[0].y_ini[14,0]
    i_l_4445 = struct[0].y_ini[15,0]
    i_l_5152 = struct[0].y_ini[16,0]
    i_l_5253 = struct[0].y_ini[17,0]
    i_l_5354 = struct[0].y_ini[18,0]
    i_l_5455 = struct[0].y_ini[19,0]
    i_l_1521 = struct[0].y_ini[20,0]
    i_l_2531 = struct[0].y_ini[21,0]
    i_l_3541 = struct[0].y_ini[22,0]
    i_l_4551 = struct[0].y_ini[23,0]
    v_11 = struct[0].y_ini[24,0]
    v_12 = struct[0].y_ini[25,0]
    i_13 = struct[0].y_ini[26,0]
    v_14 = struct[0].y_ini[27,0]
    v_15 = struct[0].y_ini[28,0]
    v_21 = struct[0].y_ini[29,0]
    v_22 = struct[0].y_ini[30,0]
    i_23 = struct[0].y_ini[31,0]
    v_24 = struct[0].y_ini[32,0]
    v_25 = struct[0].y_ini[33,0]
    v_31 = struct[0].y_ini[34,0]
    v_32 = struct[0].y_ini[35,0]
    i_33 = struct[0].y_ini[36,0]
    v_34 = struct[0].y_ini[37,0]
    v_35 = struct[0].y_ini[38,0]
    v_41 = struct[0].y_ini[39,0]
    v_42 = struct[0].y_ini[40,0]
    i_43 = struct[0].y_ini[41,0]
    v_44 = struct[0].y_ini[42,0]
    v_45 = struct[0].y_ini[43,0]
    v_51 = struct[0].y_ini[44,0]
    v_52 = struct[0].y_ini[45,0]
    i_53 = struct[0].y_ini[46,0]
    v_54 = struct[0].y_ini[47,0]
    v_55 = struct[0].y_ini[48,0]
    i_11 = struct[0].y_ini[49,0]
    i_12 = struct[0].y_ini[50,0]
    i_14 = struct[0].y_ini[51,0]
    i_15 = struct[0].y_ini[52,0]
    i_21 = struct[0].y_ini[53,0]
    i_22 = struct[0].y_ini[54,0]
    i_24 = struct[0].y_ini[55,0]
    i_25 = struct[0].y_ini[56,0]
    i_31 = struct[0].y_ini[57,0]
    i_32 = struct[0].y_ini[58,0]
    i_34 = struct[0].y_ini[59,0]
    i_35 = struct[0].y_ini[60,0]
    i_41 = struct[0].y_ini[61,0]
    i_42 = struct[0].y_ini[62,0]
    i_44 = struct[0].y_ini[63,0]
    i_45 = struct[0].y_ini[64,0]
    i_51 = struct[0].y_ini[65,0]
    i_52 = struct[0].y_ini[66,0]
    i_54 = struct[0].y_ini[67,0]
    i_55 = struct[0].y_ini[68,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-Dv_r_13 - K_r*i_13*v_13 - v_13 + v_nom)/T_v
        struct[0].f[1,0] = (-Dv_r_23 - K_r*i_23*v_23 - v_23 + v_nom)/T_v
        struct[0].f[2,0] = (-Dv_r_33 - K_r*i_33*v_33 - v_33 + v_nom)/T_v
        struct[0].f[3,0] = (-Dv_r_43 - K_r*i_43*v_43 - v_43 + v_nom)/T_v
        struct[0].f[4,0] = (-Dv_r_53 - K_r*i_53*v_53 - v_53 + v_nom)/T_v
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -R_1112*i_l_1112 + v_11 - v_12
        struct[0].g[1,0] = -R_1213*i_l_1213 + v_12 - v_13
        struct[0].g[2,0] = -R_1314*i_l_1314 + v_13 - v_14
        struct[0].g[3,0] = -R_1415*i_l_1415 + v_14 - v_15
        struct[0].g[4,0] = -R_2122*i_l_2122 + v_21 - v_22
        struct[0].g[5,0] = -R_2223*i_l_2223 + v_22 - v_23
        struct[0].g[6,0] = -R_2324*i_l_2324 + v_23 - v_24
        struct[0].g[7,0] = -R_2425*i_l_2425 + v_24 - v_25
        struct[0].g[8,0] = -R_3132*i_l_3132 + v_31 - v_32
        struct[0].g[9,0] = -R_3233*i_l_3233 + v_32 - v_33
        struct[0].g[10,0] = -R_3334*i_l_3334 + v_33 - v_34
        struct[0].g[11,0] = -R_3435*i_l_3435 + v_34 - v_35
        struct[0].g[12,0] = -R_4142*i_l_4142 + v_41 - v_42
        struct[0].g[13,0] = -R_4243*i_l_4243 + v_42 - v_43
        struct[0].g[14,0] = -R_4344*i_l_4344 + v_43 - v_44
        struct[0].g[15,0] = -R_4445*i_l_4445 + v_44 - v_45
        struct[0].g[16,0] = -R_5152*i_l_5152 + v_51 - v_52
        struct[0].g[17,0] = -R_5253*i_l_5253 + v_52 - v_53
        struct[0].g[18,0] = -R_5354*i_l_5354 + v_53 - v_54
        struct[0].g[19,0] = -R_5455*i_l_5455 + v_54 - v_55
        struct[0].g[20,0] = -R_1521*i_l_1521 + v_15 - v_21
        struct[0].g[21,0] = -R_2531*i_l_2531 + v_25 - v_31
        struct[0].g[22,0] = -R_3541*i_l_3541 + v_35 - v_41
        struct[0].g[23,0] = -R_4551*i_l_4551 + v_45 - v_51
        struct[0].g[24,0] = i_11 - i_l_1112
        struct[0].g[25,0] = i_12 + i_l_1112 - i_l_1213
        struct[0].g[26,0] = i_13 + i_l_1213 - i_l_1314
        struct[0].g[27,0] = i_14 + i_l_1314 - i_l_1415
        struct[0].g[28,0] = i_15 + i_l_1415 - i_l_1521
        struct[0].g[29,0] = i_21 + i_l_1521 - i_l_2122
        struct[0].g[30,0] = i_22 + i_l_2122 - i_l_2223
        struct[0].g[31,0] = i_23 + i_l_2223 - i_l_2324
        struct[0].g[32,0] = i_24 + i_l_2324 - i_l_2425
        struct[0].g[33,0] = i_25 + i_l_2425 - i_l_2531
        struct[0].g[34,0] = i_31 + i_l_2531 - i_l_3132
        struct[0].g[35,0] = i_32 + i_l_3132 - i_l_3233
        struct[0].g[36,0] = i_33 + i_l_3233 - i_l_3334
        struct[0].g[37,0] = i_34 + i_l_3334 - i_l_3435
        struct[0].g[38,0] = i_35 + i_l_3435 - i_l_3541
        struct[0].g[39,0] = i_41 + i_l_3541 - i_l_4142
        struct[0].g[40,0] = i_42 + i_l_4142 - i_l_4243
        struct[0].g[41,0] = i_43 + i_l_4243 - i_l_4344
        struct[0].g[42,0] = i_44 + i_l_4344 - i_l_4445
        struct[0].g[43,0] = i_45 + i_l_4445 - i_l_4551
        struct[0].g[44,0] = i_51 + i_l_4551 - i_l_5152
        struct[0].g[45,0] = i_52 + i_l_5152 - i_l_5253
        struct[0].g[46,0] = i_53 + i_l_5253 - i_l_5354
        struct[0].g[47,0] = i_54 + i_l_5354 - i_l_5455
        struct[0].g[48,0] = i_55 + i_l_5455
        struct[0].g[49,0] = i_11*v_11 - p_11
        struct[0].g[50,0] = i_12*v_12 - p_12
        struct[0].g[51,0] = i_14*v_14 - p_14
        struct[0].g[52,0] = i_15*v_15 - p_15
        struct[0].g[53,0] = i_21*v_21 - p_21
        struct[0].g[54,0] = i_22*v_22 - p_22
        struct[0].g[55,0] = i_24*v_24 - p_24
        struct[0].g[56,0] = i_25*v_25 - p_25
        struct[0].g[57,0] = i_31*v_31 - p_31
        struct[0].g[58,0] = i_32*v_32 - p_32
        struct[0].g[59,0] = i_34*v_34 - p_34
        struct[0].g[60,0] = i_35*v_35 - p_35
        struct[0].g[61,0] = i_41*v_41 - p_41
        struct[0].g[62,0] = i_42*v_42 - p_42
        struct[0].g[63,0] = i_44*v_44 - p_44
        struct[0].g[64,0] = i_45*v_45 - p_45
        struct[0].g[65,0] = i_51*v_51 - p_51
        struct[0].g[66,0] = i_52*v_52 - p_52
        struct[0].g[67,0] = i_54*v_54 - p_54
        struct[0].g[68,0] = i_55*v_55 - p_55
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_13*v_13
        struct[0].h[1,0] = v_13
        struct[0].h[2,0] = i_23*v_23
        struct[0].h[3,0] = v_23
        struct[0].h[4,0] = i_33*v_33
        struct[0].h[5,0] = v_33
        struct[0].h[6,0] = i_43*v_43
        struct[0].h[7,0] = v_43
        struct[0].h[8,0] = i_53*v_53
        struct[0].h[9,0] = v_53
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = (-K_r*i_13 - 1)/T_v
        struct[0].Fx_ini[1,1] = (-K_r*i_23 - 1)/T_v
        struct[0].Fx_ini[2,2] = (-K_r*i_33 - 1)/T_v
        struct[0].Fx_ini[3,3] = (-K_r*i_43 - 1)/T_v
        struct[0].Fx_ini[4,4] = (-K_r*i_53 - 1)/T_v

    if mode == 11:

        struct[0].Fy_ini[0,26] = -K_r*v_13/T_v 
        struct[0].Fy_ini[1,31] = -K_r*v_23/T_v 
        struct[0].Fy_ini[2,36] = -K_r*v_33/T_v 
        struct[0].Fy_ini[3,41] = -K_r*v_43/T_v 
        struct[0].Fy_ini[4,46] = -K_r*v_53/T_v 

        struct[0].Gy_ini[0,0] = -R_1112
        struct[0].Gy_ini[0,24] = 1
        struct[0].Gy_ini[0,25] = -1
        struct[0].Gy_ini[1,1] = -R_1213
        struct[0].Gy_ini[1,25] = 1
        struct[0].Gy_ini[2,2] = -R_1314
        struct[0].Gy_ini[2,27] = -1
        struct[0].Gy_ini[3,3] = -R_1415
        struct[0].Gy_ini[3,27] = 1
        struct[0].Gy_ini[3,28] = -1
        struct[0].Gy_ini[4,4] = -R_2122
        struct[0].Gy_ini[4,29] = 1
        struct[0].Gy_ini[4,30] = -1
        struct[0].Gy_ini[5,5] = -R_2223
        struct[0].Gy_ini[5,30] = 1
        struct[0].Gy_ini[6,6] = -R_2324
        struct[0].Gy_ini[6,32] = -1
        struct[0].Gy_ini[7,7] = -R_2425
        struct[0].Gy_ini[7,32] = 1
        struct[0].Gy_ini[7,33] = -1
        struct[0].Gy_ini[8,8] = -R_3132
        struct[0].Gy_ini[8,34] = 1
        struct[0].Gy_ini[8,35] = -1
        struct[0].Gy_ini[9,9] = -R_3233
        struct[0].Gy_ini[9,35] = 1
        struct[0].Gy_ini[10,10] = -R_3334
        struct[0].Gy_ini[10,37] = -1
        struct[0].Gy_ini[11,11] = -R_3435
        struct[0].Gy_ini[11,37] = 1
        struct[0].Gy_ini[11,38] = -1
        struct[0].Gy_ini[12,12] = -R_4142
        struct[0].Gy_ini[12,39] = 1
        struct[0].Gy_ini[12,40] = -1
        struct[0].Gy_ini[13,13] = -R_4243
        struct[0].Gy_ini[13,40] = 1
        struct[0].Gy_ini[14,14] = -R_4344
        struct[0].Gy_ini[14,42] = -1
        struct[0].Gy_ini[15,15] = -R_4445
        struct[0].Gy_ini[15,42] = 1
        struct[0].Gy_ini[15,43] = -1
        struct[0].Gy_ini[16,16] = -R_5152
        struct[0].Gy_ini[16,44] = 1
        struct[0].Gy_ini[16,45] = -1
        struct[0].Gy_ini[17,17] = -R_5253
        struct[0].Gy_ini[17,45] = 1
        struct[0].Gy_ini[18,18] = -R_5354
        struct[0].Gy_ini[18,47] = -1
        struct[0].Gy_ini[19,19] = -R_5455
        struct[0].Gy_ini[19,47] = 1
        struct[0].Gy_ini[19,48] = -1
        struct[0].Gy_ini[20,20] = -R_1521
        struct[0].Gy_ini[20,28] = 1
        struct[0].Gy_ini[20,29] = -1
        struct[0].Gy_ini[21,21] = -R_2531
        struct[0].Gy_ini[21,33] = 1
        struct[0].Gy_ini[21,34] = -1
        struct[0].Gy_ini[22,22] = -R_3541
        struct[0].Gy_ini[22,38] = 1
        struct[0].Gy_ini[22,39] = -1
        struct[0].Gy_ini[23,23] = -R_4551
        struct[0].Gy_ini[23,43] = 1
        struct[0].Gy_ini[23,44] = -1
        struct[0].Gy_ini[24,0] = -1
        struct[0].Gy_ini[24,49] = 1
        struct[0].Gy_ini[25,0] = 1
        struct[0].Gy_ini[25,1] = -1
        struct[0].Gy_ini[25,50] = 1
        struct[0].Gy_ini[26,1] = 1
        struct[0].Gy_ini[26,2] = -1
        struct[0].Gy_ini[26,26] = 1
        struct[0].Gy_ini[27,2] = 1
        struct[0].Gy_ini[27,3] = -1
        struct[0].Gy_ini[27,51] = 1
        struct[0].Gy_ini[28,3] = 1
        struct[0].Gy_ini[28,20] = -1
        struct[0].Gy_ini[28,52] = 1
        struct[0].Gy_ini[29,4] = -1
        struct[0].Gy_ini[29,20] = 1
        struct[0].Gy_ini[29,53] = 1
        struct[0].Gy_ini[30,4] = 1
        struct[0].Gy_ini[30,5] = -1
        struct[0].Gy_ini[30,54] = 1
        struct[0].Gy_ini[31,5] = 1
        struct[0].Gy_ini[31,6] = -1
        struct[0].Gy_ini[31,31] = 1
        struct[0].Gy_ini[32,6] = 1
        struct[0].Gy_ini[32,7] = -1
        struct[0].Gy_ini[32,55] = 1
        struct[0].Gy_ini[33,7] = 1
        struct[0].Gy_ini[33,21] = -1
        struct[0].Gy_ini[33,56] = 1
        struct[0].Gy_ini[34,8] = -1
        struct[0].Gy_ini[34,21] = 1
        struct[0].Gy_ini[34,57] = 1
        struct[0].Gy_ini[35,8] = 1
        struct[0].Gy_ini[35,9] = -1
        struct[0].Gy_ini[35,58] = 1
        struct[0].Gy_ini[36,9] = 1
        struct[0].Gy_ini[36,10] = -1
        struct[0].Gy_ini[36,36] = 1
        struct[0].Gy_ini[37,10] = 1
        struct[0].Gy_ini[37,11] = -1
        struct[0].Gy_ini[37,59] = 1
        struct[0].Gy_ini[38,11] = 1
        struct[0].Gy_ini[38,22] = -1
        struct[0].Gy_ini[38,60] = 1
        struct[0].Gy_ini[39,12] = -1
        struct[0].Gy_ini[39,22] = 1
        struct[0].Gy_ini[39,61] = 1
        struct[0].Gy_ini[40,12] = 1
        struct[0].Gy_ini[40,13] = -1
        struct[0].Gy_ini[40,62] = 1
        struct[0].Gy_ini[41,13] = 1
        struct[0].Gy_ini[41,14] = -1
        struct[0].Gy_ini[41,41] = 1
        struct[0].Gy_ini[42,14] = 1
        struct[0].Gy_ini[42,15] = -1
        struct[0].Gy_ini[42,63] = 1
        struct[0].Gy_ini[43,15] = 1
        struct[0].Gy_ini[43,23] = -1
        struct[0].Gy_ini[43,64] = 1
        struct[0].Gy_ini[44,16] = -1
        struct[0].Gy_ini[44,23] = 1
        struct[0].Gy_ini[44,65] = 1
        struct[0].Gy_ini[45,16] = 1
        struct[0].Gy_ini[45,17] = -1
        struct[0].Gy_ini[45,66] = 1
        struct[0].Gy_ini[46,17] = 1
        struct[0].Gy_ini[46,18] = -1
        struct[0].Gy_ini[46,46] = 1
        struct[0].Gy_ini[47,18] = 1
        struct[0].Gy_ini[47,19] = -1
        struct[0].Gy_ini[47,67] = 1
        struct[0].Gy_ini[48,19] = 1
        struct[0].Gy_ini[48,68] = 1
        struct[0].Gy_ini[49,24] = i_11
        struct[0].Gy_ini[49,49] = v_11
        struct[0].Gy_ini[50,25] = i_12
        struct[0].Gy_ini[50,50] = v_12
        struct[0].Gy_ini[51,27] = i_14
        struct[0].Gy_ini[51,51] = v_14
        struct[0].Gy_ini[52,28] = i_15
        struct[0].Gy_ini[52,52] = v_15
        struct[0].Gy_ini[53,29] = i_21
        struct[0].Gy_ini[53,53] = v_21
        struct[0].Gy_ini[54,30] = i_22
        struct[0].Gy_ini[54,54] = v_22
        struct[0].Gy_ini[55,32] = i_24
        struct[0].Gy_ini[55,55] = v_24
        struct[0].Gy_ini[56,33] = i_25
        struct[0].Gy_ini[56,56] = v_25
        struct[0].Gy_ini[57,34] = i_31
        struct[0].Gy_ini[57,57] = v_31
        struct[0].Gy_ini[58,35] = i_32
        struct[0].Gy_ini[58,58] = v_32
        struct[0].Gy_ini[59,37] = i_34
        struct[0].Gy_ini[59,59] = v_34
        struct[0].Gy_ini[60,38] = i_35
        struct[0].Gy_ini[60,60] = v_35
        struct[0].Gy_ini[61,39] = i_41
        struct[0].Gy_ini[61,61] = v_41
        struct[0].Gy_ini[62,40] = i_42
        struct[0].Gy_ini[62,62] = v_42
        struct[0].Gy_ini[63,42] = i_44
        struct[0].Gy_ini[63,63] = v_44
        struct[0].Gy_ini[64,43] = i_45
        struct[0].Gy_ini[64,64] = v_45
        struct[0].Gy_ini[65,44] = i_51
        struct[0].Gy_ini[65,65] = v_51
        struct[0].Gy_ini[66,45] = i_52
        struct[0].Gy_ini[66,66] = v_52
        struct[0].Gy_ini[67,47] = i_54
        struct[0].Gy_ini[67,67] = v_54
        struct[0].Gy_ini[68,48] = i_55
        struct[0].Gy_ini[68,68] = v_55



def run_nn(t,struct,mode):

    # Parameters:
    R_1112 = struct[0].R_1112
    R_1213 = struct[0].R_1213
    R_1314 = struct[0].R_1314
    R_1415 = struct[0].R_1415
    R_1521 = struct[0].R_1521
    R_2122 = struct[0].R_2122
    R_2223 = struct[0].R_2223
    R_2324 = struct[0].R_2324
    R_2425 = struct[0].R_2425
    R_2531 = struct[0].R_2531
    R_3132 = struct[0].R_3132
    R_3233 = struct[0].R_3233
    R_3334 = struct[0].R_3334
    R_3435 = struct[0].R_3435
    R_3541 = struct[0].R_3541
    R_4142 = struct[0].R_4142
    R_4243 = struct[0].R_4243
    R_4344 = struct[0].R_4344
    R_4445 = struct[0].R_4445
    R_4551 = struct[0].R_4551
    R_5152 = struct[0].R_5152
    R_5253 = struct[0].R_5253
    R_5354 = struct[0].R_5354
    R_5455 = struct[0].R_5455
    p_11 = struct[0].p_11
    p_12 = struct[0].p_12
    p_14 = struct[0].p_14
    p_15 = struct[0].p_15
    p_21 = struct[0].p_21
    p_22 = struct[0].p_22
    p_24 = struct[0].p_24
    p_25 = struct[0].p_25
    p_31 = struct[0].p_31
    p_32 = struct[0].p_32
    p_34 = struct[0].p_34
    p_35 = struct[0].p_35
    p_41 = struct[0].p_41
    p_42 = struct[0].p_42
    p_44 = struct[0].p_44
    p_45 = struct[0].p_45
    p_51 = struct[0].p_51
    p_52 = struct[0].p_52
    p_54 = struct[0].p_54
    p_55 = struct[0].p_55
    
    # Inputs:
    Dv_r_13 = struct[0].Dv_r_13
    Dv_r_23 = struct[0].Dv_r_23
    Dv_r_33 = struct[0].Dv_r_33
    Dv_r_43 = struct[0].Dv_r_43
    Dv_r_53 = struct[0].Dv_r_53
    v_nom = struct[0].v_nom
    T_v = struct[0].T_v
    K_r = struct[0].K_r
    
    # Dynamical states:
    v_13 = struct[0].x[0,0]
    v_23 = struct[0].x[1,0]
    v_33 = struct[0].x[2,0]
    v_43 = struct[0].x[3,0]
    v_53 = struct[0].x[4,0]
    
    # Algebraic states:
    i_l_1112 = struct[0].y_run[0,0]
    i_l_1213 = struct[0].y_run[1,0]
    i_l_1314 = struct[0].y_run[2,0]
    i_l_1415 = struct[0].y_run[3,0]
    i_l_2122 = struct[0].y_run[4,0]
    i_l_2223 = struct[0].y_run[5,0]
    i_l_2324 = struct[0].y_run[6,0]
    i_l_2425 = struct[0].y_run[7,0]
    i_l_3132 = struct[0].y_run[8,0]
    i_l_3233 = struct[0].y_run[9,0]
    i_l_3334 = struct[0].y_run[10,0]
    i_l_3435 = struct[0].y_run[11,0]
    i_l_4142 = struct[0].y_run[12,0]
    i_l_4243 = struct[0].y_run[13,0]
    i_l_4344 = struct[0].y_run[14,0]
    i_l_4445 = struct[0].y_run[15,0]
    i_l_5152 = struct[0].y_run[16,0]
    i_l_5253 = struct[0].y_run[17,0]
    i_l_5354 = struct[0].y_run[18,0]
    i_l_5455 = struct[0].y_run[19,0]
    i_l_1521 = struct[0].y_run[20,0]
    i_l_2531 = struct[0].y_run[21,0]
    i_l_3541 = struct[0].y_run[22,0]
    i_l_4551 = struct[0].y_run[23,0]
    v_11 = struct[0].y_run[24,0]
    v_12 = struct[0].y_run[25,0]
    i_13 = struct[0].y_run[26,0]
    v_14 = struct[0].y_run[27,0]
    v_15 = struct[0].y_run[28,0]
    v_21 = struct[0].y_run[29,0]
    v_22 = struct[0].y_run[30,0]
    i_23 = struct[0].y_run[31,0]
    v_24 = struct[0].y_run[32,0]
    v_25 = struct[0].y_run[33,0]
    v_31 = struct[0].y_run[34,0]
    v_32 = struct[0].y_run[35,0]
    i_33 = struct[0].y_run[36,0]
    v_34 = struct[0].y_run[37,0]
    v_35 = struct[0].y_run[38,0]
    v_41 = struct[0].y_run[39,0]
    v_42 = struct[0].y_run[40,0]
    i_43 = struct[0].y_run[41,0]
    v_44 = struct[0].y_run[42,0]
    v_45 = struct[0].y_run[43,0]
    v_51 = struct[0].y_run[44,0]
    v_52 = struct[0].y_run[45,0]
    i_53 = struct[0].y_run[46,0]
    v_54 = struct[0].y_run[47,0]
    v_55 = struct[0].y_run[48,0]
    i_11 = struct[0].y_run[49,0]
    i_12 = struct[0].y_run[50,0]
    i_14 = struct[0].y_run[51,0]
    i_15 = struct[0].y_run[52,0]
    i_21 = struct[0].y_run[53,0]
    i_22 = struct[0].y_run[54,0]
    i_24 = struct[0].y_run[55,0]
    i_25 = struct[0].y_run[56,0]
    i_31 = struct[0].y_run[57,0]
    i_32 = struct[0].y_run[58,0]
    i_34 = struct[0].y_run[59,0]
    i_35 = struct[0].y_run[60,0]
    i_41 = struct[0].y_run[61,0]
    i_42 = struct[0].y_run[62,0]
    i_44 = struct[0].y_run[63,0]
    i_45 = struct[0].y_run[64,0]
    i_51 = struct[0].y_run[65,0]
    i_52 = struct[0].y_run[66,0]
    i_54 = struct[0].y_run[67,0]
    i_55 = struct[0].y_run[68,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-Dv_r_13 - K_r*i_13*v_13 - v_13 + v_nom)/T_v
        struct[0].f[1,0] = (-Dv_r_23 - K_r*i_23*v_23 - v_23 + v_nom)/T_v
        struct[0].f[2,0] = (-Dv_r_33 - K_r*i_33*v_33 - v_33 + v_nom)/T_v
        struct[0].f[3,0] = (-Dv_r_43 - K_r*i_43*v_43 - v_43 + v_nom)/T_v
        struct[0].f[4,0] = (-Dv_r_53 - K_r*i_53*v_53 - v_53 + v_nom)/T_v
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -R_1112*i_l_1112 + v_11 - v_12
        struct[0].g[1,0] = -R_1213*i_l_1213 + v_12 - v_13
        struct[0].g[2,0] = -R_1314*i_l_1314 + v_13 - v_14
        struct[0].g[3,0] = -R_1415*i_l_1415 + v_14 - v_15
        struct[0].g[4,0] = -R_2122*i_l_2122 + v_21 - v_22
        struct[0].g[5,0] = -R_2223*i_l_2223 + v_22 - v_23
        struct[0].g[6,0] = -R_2324*i_l_2324 + v_23 - v_24
        struct[0].g[7,0] = -R_2425*i_l_2425 + v_24 - v_25
        struct[0].g[8,0] = -R_3132*i_l_3132 + v_31 - v_32
        struct[0].g[9,0] = -R_3233*i_l_3233 + v_32 - v_33
        struct[0].g[10,0] = -R_3334*i_l_3334 + v_33 - v_34
        struct[0].g[11,0] = -R_3435*i_l_3435 + v_34 - v_35
        struct[0].g[12,0] = -R_4142*i_l_4142 + v_41 - v_42
        struct[0].g[13,0] = -R_4243*i_l_4243 + v_42 - v_43
        struct[0].g[14,0] = -R_4344*i_l_4344 + v_43 - v_44
        struct[0].g[15,0] = -R_4445*i_l_4445 + v_44 - v_45
        struct[0].g[16,0] = -R_5152*i_l_5152 + v_51 - v_52
        struct[0].g[17,0] = -R_5253*i_l_5253 + v_52 - v_53
        struct[0].g[18,0] = -R_5354*i_l_5354 + v_53 - v_54
        struct[0].g[19,0] = -R_5455*i_l_5455 + v_54 - v_55
        struct[0].g[20,0] = -R_1521*i_l_1521 + v_15 - v_21
        struct[0].g[21,0] = -R_2531*i_l_2531 + v_25 - v_31
        struct[0].g[22,0] = -R_3541*i_l_3541 + v_35 - v_41
        struct[0].g[23,0] = -R_4551*i_l_4551 + v_45 - v_51
        struct[0].g[24,0] = i_11 - i_l_1112
        struct[0].g[25,0] = i_12 + i_l_1112 - i_l_1213
        struct[0].g[26,0] = i_13 + i_l_1213 - i_l_1314
        struct[0].g[27,0] = i_14 + i_l_1314 - i_l_1415
        struct[0].g[28,0] = i_15 + i_l_1415 - i_l_1521
        struct[0].g[29,0] = i_21 + i_l_1521 - i_l_2122
        struct[0].g[30,0] = i_22 + i_l_2122 - i_l_2223
        struct[0].g[31,0] = i_23 + i_l_2223 - i_l_2324
        struct[0].g[32,0] = i_24 + i_l_2324 - i_l_2425
        struct[0].g[33,0] = i_25 + i_l_2425 - i_l_2531
        struct[0].g[34,0] = i_31 + i_l_2531 - i_l_3132
        struct[0].g[35,0] = i_32 + i_l_3132 - i_l_3233
        struct[0].g[36,0] = i_33 + i_l_3233 - i_l_3334
        struct[0].g[37,0] = i_34 + i_l_3334 - i_l_3435
        struct[0].g[38,0] = i_35 + i_l_3435 - i_l_3541
        struct[0].g[39,0] = i_41 + i_l_3541 - i_l_4142
        struct[0].g[40,0] = i_42 + i_l_4142 - i_l_4243
        struct[0].g[41,0] = i_43 + i_l_4243 - i_l_4344
        struct[0].g[42,0] = i_44 + i_l_4344 - i_l_4445
        struct[0].g[43,0] = i_45 + i_l_4445 - i_l_4551
        struct[0].g[44,0] = i_51 + i_l_4551 - i_l_5152
        struct[0].g[45,0] = i_52 + i_l_5152 - i_l_5253
        struct[0].g[46,0] = i_53 + i_l_5253 - i_l_5354
        struct[0].g[47,0] = i_54 + i_l_5354 - i_l_5455
        struct[0].g[48,0] = i_55 + i_l_5455
        struct[0].g[49,0] = i_11*v_11 - p_11
        struct[0].g[50,0] = i_12*v_12 - p_12
        struct[0].g[51,0] = i_14*v_14 - p_14
        struct[0].g[52,0] = i_15*v_15 - p_15
        struct[0].g[53,0] = i_21*v_21 - p_21
        struct[0].g[54,0] = i_22*v_22 - p_22
        struct[0].g[55,0] = i_24*v_24 - p_24
        struct[0].g[56,0] = i_25*v_25 - p_25
        struct[0].g[57,0] = i_31*v_31 - p_31
        struct[0].g[58,0] = i_32*v_32 - p_32
        struct[0].g[59,0] = i_34*v_34 - p_34
        struct[0].g[60,0] = i_35*v_35 - p_35
        struct[0].g[61,0] = i_41*v_41 - p_41
        struct[0].g[62,0] = i_42*v_42 - p_42
        struct[0].g[63,0] = i_44*v_44 - p_44
        struct[0].g[64,0] = i_45*v_45 - p_45
        struct[0].g[65,0] = i_51*v_51 - p_51
        struct[0].g[66,0] = i_52*v_52 - p_52
        struct[0].g[67,0] = i_54*v_54 - p_54
        struct[0].g[68,0] = i_55*v_55 - p_55
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_13*v_13
        struct[0].h[1,0] = v_13
        struct[0].h[2,0] = i_23*v_23
        struct[0].h[3,0] = v_23
        struct[0].h[4,0] = i_33*v_33
        struct[0].h[5,0] = v_33
        struct[0].h[6,0] = i_43*v_43
        struct[0].h[7,0] = v_43
        struct[0].h[8,0] = i_53*v_53
        struct[0].h[9,0] = v_53
    

    if mode == 10:

        struct[0].Fx[0,0] = (-K_r*i_13 - 1)/T_v
        struct[0].Fx[1,1] = (-K_r*i_23 - 1)/T_v
        struct[0].Fx[2,2] = (-K_r*i_33 - 1)/T_v
        struct[0].Fx[3,3] = (-K_r*i_43 - 1)/T_v
        struct[0].Fx[4,4] = (-K_r*i_53 - 1)/T_v

    if mode == 11:

        struct[0].Fy[0,26] = -K_r*v_13/T_v
        struct[0].Fy[1,31] = -K_r*v_23/T_v
        struct[0].Fy[2,36] = -K_r*v_33/T_v
        struct[0].Fy[3,41] = -K_r*v_43/T_v
        struct[0].Fy[4,46] = -K_r*v_53/T_v

        struct[0].Gy[0,0] = -R_1112
        struct[0].Gy[0,24] = 1
        struct[0].Gy[0,25] = -1
        struct[0].Gy[1,1] = -R_1213
        struct[0].Gy[1,25] = 1
        struct[0].Gy[2,2] = -R_1314
        struct[0].Gy[2,27] = -1
        struct[0].Gy[3,3] = -R_1415
        struct[0].Gy[3,27] = 1
        struct[0].Gy[3,28] = -1
        struct[0].Gy[4,4] = -R_2122
        struct[0].Gy[4,29] = 1
        struct[0].Gy[4,30] = -1
        struct[0].Gy[5,5] = -R_2223
        struct[0].Gy[5,30] = 1
        struct[0].Gy[6,6] = -R_2324
        struct[0].Gy[6,32] = -1
        struct[0].Gy[7,7] = -R_2425
        struct[0].Gy[7,32] = 1
        struct[0].Gy[7,33] = -1
        struct[0].Gy[8,8] = -R_3132
        struct[0].Gy[8,34] = 1
        struct[0].Gy[8,35] = -1
        struct[0].Gy[9,9] = -R_3233
        struct[0].Gy[9,35] = 1
        struct[0].Gy[10,10] = -R_3334
        struct[0].Gy[10,37] = -1
        struct[0].Gy[11,11] = -R_3435
        struct[0].Gy[11,37] = 1
        struct[0].Gy[11,38] = -1
        struct[0].Gy[12,12] = -R_4142
        struct[0].Gy[12,39] = 1
        struct[0].Gy[12,40] = -1
        struct[0].Gy[13,13] = -R_4243
        struct[0].Gy[13,40] = 1
        struct[0].Gy[14,14] = -R_4344
        struct[0].Gy[14,42] = -1
        struct[0].Gy[15,15] = -R_4445
        struct[0].Gy[15,42] = 1
        struct[0].Gy[15,43] = -1
        struct[0].Gy[16,16] = -R_5152
        struct[0].Gy[16,44] = 1
        struct[0].Gy[16,45] = -1
        struct[0].Gy[17,17] = -R_5253
        struct[0].Gy[17,45] = 1
        struct[0].Gy[18,18] = -R_5354
        struct[0].Gy[18,47] = -1
        struct[0].Gy[19,19] = -R_5455
        struct[0].Gy[19,47] = 1
        struct[0].Gy[19,48] = -1
        struct[0].Gy[20,20] = -R_1521
        struct[0].Gy[20,28] = 1
        struct[0].Gy[20,29] = -1
        struct[0].Gy[21,21] = -R_2531
        struct[0].Gy[21,33] = 1
        struct[0].Gy[21,34] = -1
        struct[0].Gy[22,22] = -R_3541
        struct[0].Gy[22,38] = 1
        struct[0].Gy[22,39] = -1
        struct[0].Gy[23,23] = -R_4551
        struct[0].Gy[23,43] = 1
        struct[0].Gy[23,44] = -1
        struct[0].Gy[24,0] = -1
        struct[0].Gy[24,49] = 1
        struct[0].Gy[25,0] = 1
        struct[0].Gy[25,1] = -1
        struct[0].Gy[25,50] = 1
        struct[0].Gy[26,1] = 1
        struct[0].Gy[26,2] = -1
        struct[0].Gy[26,26] = 1
        struct[0].Gy[27,2] = 1
        struct[0].Gy[27,3] = -1
        struct[0].Gy[27,51] = 1
        struct[0].Gy[28,3] = 1
        struct[0].Gy[28,20] = -1
        struct[0].Gy[28,52] = 1
        struct[0].Gy[29,4] = -1
        struct[0].Gy[29,20] = 1
        struct[0].Gy[29,53] = 1
        struct[0].Gy[30,4] = 1
        struct[0].Gy[30,5] = -1
        struct[0].Gy[30,54] = 1
        struct[0].Gy[31,5] = 1
        struct[0].Gy[31,6] = -1
        struct[0].Gy[31,31] = 1
        struct[0].Gy[32,6] = 1
        struct[0].Gy[32,7] = -1
        struct[0].Gy[32,55] = 1
        struct[0].Gy[33,7] = 1
        struct[0].Gy[33,21] = -1
        struct[0].Gy[33,56] = 1
        struct[0].Gy[34,8] = -1
        struct[0].Gy[34,21] = 1
        struct[0].Gy[34,57] = 1
        struct[0].Gy[35,8] = 1
        struct[0].Gy[35,9] = -1
        struct[0].Gy[35,58] = 1
        struct[0].Gy[36,9] = 1
        struct[0].Gy[36,10] = -1
        struct[0].Gy[36,36] = 1
        struct[0].Gy[37,10] = 1
        struct[0].Gy[37,11] = -1
        struct[0].Gy[37,59] = 1
        struct[0].Gy[38,11] = 1
        struct[0].Gy[38,22] = -1
        struct[0].Gy[38,60] = 1
        struct[0].Gy[39,12] = -1
        struct[0].Gy[39,22] = 1
        struct[0].Gy[39,61] = 1
        struct[0].Gy[40,12] = 1
        struct[0].Gy[40,13] = -1
        struct[0].Gy[40,62] = 1
        struct[0].Gy[41,13] = 1
        struct[0].Gy[41,14] = -1
        struct[0].Gy[41,41] = 1
        struct[0].Gy[42,14] = 1
        struct[0].Gy[42,15] = -1
        struct[0].Gy[42,63] = 1
        struct[0].Gy[43,15] = 1
        struct[0].Gy[43,23] = -1
        struct[0].Gy[43,64] = 1
        struct[0].Gy[44,16] = -1
        struct[0].Gy[44,23] = 1
        struct[0].Gy[44,65] = 1
        struct[0].Gy[45,16] = 1
        struct[0].Gy[45,17] = -1
        struct[0].Gy[45,66] = 1
        struct[0].Gy[46,17] = 1
        struct[0].Gy[46,18] = -1
        struct[0].Gy[46,46] = 1
        struct[0].Gy[47,18] = 1
        struct[0].Gy[47,19] = -1
        struct[0].Gy[47,67] = 1
        struct[0].Gy[48,19] = 1
        struct[0].Gy[48,68] = 1
        struct[0].Gy[49,24] = i_11
        struct[0].Gy[49,49] = v_11
        struct[0].Gy[50,25] = i_12
        struct[0].Gy[50,50] = v_12
        struct[0].Gy[51,27] = i_14
        struct[0].Gy[51,51] = v_14
        struct[0].Gy[52,28] = i_15
        struct[0].Gy[52,52] = v_15
        struct[0].Gy[53,29] = i_21
        struct[0].Gy[53,53] = v_21
        struct[0].Gy[54,30] = i_22
        struct[0].Gy[54,54] = v_22
        struct[0].Gy[55,32] = i_24
        struct[0].Gy[55,55] = v_24
        struct[0].Gy[56,33] = i_25
        struct[0].Gy[56,56] = v_25
        struct[0].Gy[57,34] = i_31
        struct[0].Gy[57,57] = v_31
        struct[0].Gy[58,35] = i_32
        struct[0].Gy[58,58] = v_32
        struct[0].Gy[59,37] = i_34
        struct[0].Gy[59,59] = v_34
        struct[0].Gy[60,38] = i_35
        struct[0].Gy[60,60] = v_35
        struct[0].Gy[61,39] = i_41
        struct[0].Gy[61,61] = v_41
        struct[0].Gy[62,40] = i_42
        struct[0].Gy[62,62] = v_42
        struct[0].Gy[63,42] = i_44
        struct[0].Gy[63,63] = v_44
        struct[0].Gy[64,43] = i_45
        struct[0].Gy[64,64] = v_45
        struct[0].Gy[65,44] = i_51
        struct[0].Gy[65,65] = v_51
        struct[0].Gy[66,45] = i_52
        struct[0].Gy[66,66] = v_52
        struct[0].Gy[67,47] = i_54
        struct[0].Gy[67,67] = v_54
        struct[0].Gy[68,48] = i_55
        struct[0].Gy[68,68] = v_55






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
    Fx_ini_rows = [0, 1, 2, 3, 4]

    Fx_ini_cols = [0, 1, 2, 3, 4]

    Fy_ini_rows = [0, 1, 2, 3, 4]

    Fy_ini_cols = [26, 31, 36, 41, 46]

    Gx_ini_rows = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18]

    Gx_ini_cols = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    Gy_ini_rows = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68]

    Gy_ini_cols = [0, 24, 25, 1, 25, 2, 27, 3, 27, 28, 4, 29, 30, 5, 30, 6, 32, 7, 32, 33, 8, 34, 35, 9, 35, 10, 37, 11, 37, 38, 12, 39, 40, 13, 40, 14, 42, 15, 42, 43, 16, 44, 45, 17, 45, 18, 47, 19, 47, 48, 20, 28, 29, 21, 33, 34, 22, 38, 39, 23, 43, 44, 0, 49, 0, 1, 50, 1, 2, 26, 2, 3, 51, 3, 20, 52, 4, 20, 53, 4, 5, 54, 5, 6, 31, 6, 7, 55, 7, 21, 56, 8, 21, 57, 8, 9, 58, 9, 10, 36, 10, 11, 59, 11, 22, 60, 12, 22, 61, 12, 13, 62, 13, 14, 41, 14, 15, 63, 15, 23, 64, 16, 23, 65, 16, 17, 66, 17, 18, 46, 18, 19, 67, 19, 68, 24, 49, 25, 50, 27, 51, 28, 52, 29, 53, 30, 54, 32, 55, 33, 56, 34, 57, 35, 58, 37, 59, 38, 60, 39, 61, 40, 62, 42, 63, 43, 64, 44, 65, 45, 66, 47, 67, 48, 68]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols