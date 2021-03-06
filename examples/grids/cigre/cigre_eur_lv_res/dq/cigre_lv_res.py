import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 


class cigre_lv_res_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 12
        self.N_y = 86 
        self.N_z = 1 
        self.N_store = 10000 
        self.params_list = ['R_R00R01', 'L_R00R01', 'C_R00R01', 'R_R02R01', 'L_R02R01', 'C_R02R01', 'R_R02R03', 'L_R02R03', 'C_R02R03', 'R_R03R04', 'L_R03R04', 'C_R03R04', 'R_R04R05', 'L_R04R05', 'C_R04R05', 'R_R04R12', 'L_R04R12', 'C_R04R12', 'R_R05R06', 'L_R05R06', 'C_R05R06', 'R_R06R07', 'L_R06R07', 'C_R06R07', 'R_R07R08', 'L_R07R08', 'C_R07R08', 'R_R08R09', 'L_R08R09', 'C_R08R09', 'R_R09R10', 'L_R09R10', 'C_R09R10', 'R_R09R17', 'L_R09R17', 'C_R09R17', 'R_R11R03', 'L_R11R03', 'C_R11R03', 'R_R12R13', 'L_R12R13', 'C_R12R13', 'R_R13R14', 'L_R13R14', 'C_R13R14', 'R_R14R15', 'L_R14R15', 'C_R14R15', 'R_R16R06', 'L_R16R06', 'C_R16R06', 'R_R18R10', 'L_R18R10', 'C_R18R10', 'i_R02_D', 'i_R02_Q', 'i_R03_D', 'i_R03_Q', 'i_R04_D', 'i_R04_Q', 'i_R05_D', 'i_R05_Q', 'i_R06_D', 'i_R06_Q', 'i_R07_D', 'i_R07_Q', 'i_R08_D', 'i_R08_Q', 'i_R09_D', 'i_R09_Q', 'i_R10_D', 'i_R10_Q', 'i_R12_D', 'i_R12_Q', 'i_R13_D', 'i_R13_Q', 'i_R14_D', 'i_R14_Q', 'omega'] 
        self.params_values_list  = [0.0032, 4.074366543152521e-05, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 314.1592653589793] 
        self.inputs_ini_list = ['v_R00_D', 'v_R00_Q', 'T_i_R01', 'I_max_R01', 'p_R01_ref', 'q_R01_ref', 'T_i_R11', 'I_max_R11', 'p_R11_ref', 'q_R11_ref', 'T_i_R15', 'I_max_R15', 'p_R15_ref', 'q_R15_ref', 'T_i_R16', 'I_max_R16', 'p_R16_ref', 'q_R16_ref', 'T_i_R17', 'I_max_R17', 'p_R17_ref', 'q_R17_ref', 'T_i_R18', 'I_max_R18', 'p_R18_ref', 'q_R18_ref'] 
        self.inputs_ini_values_list  = [0.0, 326.5986323710904, 0.01, 434.78, 950.0, 312.24989991992004, 0.01, 32.6, 14250.0, 4683.7484987988, 0.01, 120, 49400.0, 16236.99479583584, 0.01, 120, 52250.0, 17173.7444955956, 0.01, 120, 33250.0, 10928.7464971972, 0.01, 120, 44650.0, 14675.74529623624] 
        self.inputs_run_list = ['v_R00_D', 'v_R00_Q', 'T_i_R01', 'I_max_R01', 'p_R01_ref', 'q_R01_ref', 'T_i_R11', 'I_max_R11', 'p_R11_ref', 'q_R11_ref', 'T_i_R15', 'I_max_R15', 'p_R15_ref', 'q_R15_ref', 'T_i_R16', 'I_max_R16', 'p_R16_ref', 'q_R16_ref', 'T_i_R17', 'I_max_R17', 'p_R17_ref', 'q_R17_ref', 'T_i_R18', 'I_max_R18', 'p_R18_ref', 'q_R18_ref'] 
        self.inputs_run_values_list = [0.0, 326.5986323710904, 0.01, 434.78, 950.0, 312.24989991992004, 0.01, 32.6, 14250.0, 4683.7484987988, 0.01, 120, 49400.0, 16236.99479583584, 0.01, 120, 52250.0, 17173.7444955956, 0.01, 120, 33250.0, 10928.7464971972, 0.01, 120, 44650.0, 14675.74529623624] 
        self.outputs_list = ['i_R01'] 
        self.x_list = ['i_R01_D', 'i_R01_Q', 'i_R11_D', 'i_R11_Q', 'i_R15_D', 'i_R15_Q', 'i_R16_D', 'i_R16_Q', 'i_R17_D', 'i_R17_Q', 'i_R18_D', 'i_R18_Q'] 
        self.y_run_list = ['i_l_R00R01_D', 'i_l_R00R01_Q', 'i_l_R02R01_D', 'i_l_R02R01_Q', 'i_l_R02R03_D', 'i_l_R02R03_Q', 'i_l_R03R04_D', 'i_l_R03R04_Q', 'i_l_R04R05_D', 'i_l_R04R05_Q', 'i_l_R04R12_D', 'i_l_R04R12_Q', 'i_l_R05R06_D', 'i_l_R05R06_Q', 'i_l_R06R07_D', 'i_l_R06R07_Q', 'i_l_R07R08_D', 'i_l_R07R08_Q', 'i_l_R08R09_D', 'i_l_R08R09_Q', 'i_l_R09R10_D', 'i_l_R09R10_Q', 'i_l_R09R17_D', 'i_l_R09R17_Q', 'i_l_R11R03_D', 'i_l_R11R03_Q', 'i_l_R12R13_D', 'i_l_R12R13_Q', 'i_l_R13R14_D', 'i_l_R13R14_Q', 'i_l_R14R15_D', 'i_l_R14R15_Q', 'i_l_R16R06_D', 'i_l_R16R06_Q', 'i_l_R18R10_D', 'i_l_R18R10_Q', 'i_R00_D', 'i_R00_Q', 'v_R01_D', 'v_R01_Q', 'v_R02_D', 'v_R02_Q', 'v_R03_D', 'v_R03_Q', 'v_R04_D', 'v_R04_Q', 'v_R05_D', 'v_R05_Q', 'v_R06_D', 'v_R06_Q', 'v_R07_D', 'v_R07_Q', 'v_R08_D', 'v_R08_Q', 'v_R09_D', 'v_R09_Q', 'v_R10_D', 'v_R10_Q', 'v_R11_D', 'v_R11_Q', 'v_R12_D', 'v_R12_Q', 'v_R13_D', 'v_R13_Q', 'v_R14_D', 'v_R14_Q', 'v_R15_D', 'v_R15_Q', 'v_R16_D', 'v_R16_Q', 'v_R17_D', 'v_R17_Q', 'v_R18_D', 'v_R18_Q', 'i_R01_d_ref', 'i_R01_q_ref', 'i_R11_d_ref', 'i_R11_q_ref', 'i_R15_d_ref', 'i_R15_q_ref', 'i_R16_d_ref', 'i_R16_q_ref', 'i_R17_d_ref', 'i_R17_q_ref', 'i_R18_d_ref', 'i_R18_q_ref'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['i_l_R00R01_D', 'i_l_R00R01_Q', 'i_l_R02R01_D', 'i_l_R02R01_Q', 'i_l_R02R03_D', 'i_l_R02R03_Q', 'i_l_R03R04_D', 'i_l_R03R04_Q', 'i_l_R04R05_D', 'i_l_R04R05_Q', 'i_l_R04R12_D', 'i_l_R04R12_Q', 'i_l_R05R06_D', 'i_l_R05R06_Q', 'i_l_R06R07_D', 'i_l_R06R07_Q', 'i_l_R07R08_D', 'i_l_R07R08_Q', 'i_l_R08R09_D', 'i_l_R08R09_Q', 'i_l_R09R10_D', 'i_l_R09R10_Q', 'i_l_R09R17_D', 'i_l_R09R17_Q', 'i_l_R11R03_D', 'i_l_R11R03_Q', 'i_l_R12R13_D', 'i_l_R12R13_Q', 'i_l_R13R14_D', 'i_l_R13R14_Q', 'i_l_R14R15_D', 'i_l_R14R15_Q', 'i_l_R16R06_D', 'i_l_R16R06_Q', 'i_l_R18R10_D', 'i_l_R18R10_Q', 'i_R00_D', 'i_R00_Q', 'v_R01_D', 'v_R01_Q', 'v_R02_D', 'v_R02_Q', 'v_R03_D', 'v_R03_Q', 'v_R04_D', 'v_R04_Q', 'v_R05_D', 'v_R05_Q', 'v_R06_D', 'v_R06_Q', 'v_R07_D', 'v_R07_Q', 'v_R08_D', 'v_R08_Q', 'v_R09_D', 'v_R09_Q', 'v_R10_D', 'v_R10_Q', 'v_R11_D', 'v_R11_Q', 'v_R12_D', 'v_R12_Q', 'v_R13_D', 'v_R13_Q', 'v_R14_D', 'v_R14_Q', 'v_R15_D', 'v_R15_Q', 'v_R16_D', 'v_R16_Q', 'v_R17_D', 'v_R17_Q', 'v_R18_D', 'v_R18_Q', 'i_R01_d_ref', 'i_R01_q_ref', 'i_R11_d_ref', 'i_R11_q_ref', 'i_R15_d_ref', 'i_R15_q_ref', 'i_R16_d_ref', 'i_R16_q_ref', 'i_R17_d_ref', 'i_R17_q_ref', 'i_R18_d_ref', 'i_R18_q_ref'] 
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
            
    def get_x(self):
        return self.struct[0].x


@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R10_D = struct[0].i_R10_D
    i_R10_Q = struct[0].i_R10_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    i_R14_D = struct[0].i_R14_D
    i_R14_Q = struct[0].i_R14_Q
    omega = struct[0].omega
    
    # Inputs:
    v_R00_D = struct[0].v_R00_D
    v_R00_Q = struct[0].v_R00_Q
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_ini[0,0]
    i_l_R00R01_Q = struct[0].y_ini[1,0]
    i_l_R02R01_D = struct[0].y_ini[2,0]
    i_l_R02R01_Q = struct[0].y_ini[3,0]
    i_l_R02R03_D = struct[0].y_ini[4,0]
    i_l_R02R03_Q = struct[0].y_ini[5,0]
    i_l_R03R04_D = struct[0].y_ini[6,0]
    i_l_R03R04_Q = struct[0].y_ini[7,0]
    i_l_R04R05_D = struct[0].y_ini[8,0]
    i_l_R04R05_Q = struct[0].y_ini[9,0]
    i_l_R04R12_D = struct[0].y_ini[10,0]
    i_l_R04R12_Q = struct[0].y_ini[11,0]
    i_l_R05R06_D = struct[0].y_ini[12,0]
    i_l_R05R06_Q = struct[0].y_ini[13,0]
    i_l_R06R07_D = struct[0].y_ini[14,0]
    i_l_R06R07_Q = struct[0].y_ini[15,0]
    i_l_R07R08_D = struct[0].y_ini[16,0]
    i_l_R07R08_Q = struct[0].y_ini[17,0]
    i_l_R08R09_D = struct[0].y_ini[18,0]
    i_l_R08R09_Q = struct[0].y_ini[19,0]
    i_l_R09R10_D = struct[0].y_ini[20,0]
    i_l_R09R10_Q = struct[0].y_ini[21,0]
    i_l_R09R17_D = struct[0].y_ini[22,0]
    i_l_R09R17_Q = struct[0].y_ini[23,0]
    i_l_R11R03_D = struct[0].y_ini[24,0]
    i_l_R11R03_Q = struct[0].y_ini[25,0]
    i_l_R12R13_D = struct[0].y_ini[26,0]
    i_l_R12R13_Q = struct[0].y_ini[27,0]
    i_l_R13R14_D = struct[0].y_ini[28,0]
    i_l_R13R14_Q = struct[0].y_ini[29,0]
    i_l_R14R15_D = struct[0].y_ini[30,0]
    i_l_R14R15_Q = struct[0].y_ini[31,0]
    i_l_R16R06_D = struct[0].y_ini[32,0]
    i_l_R16R06_Q = struct[0].y_ini[33,0]
    i_l_R18R10_D = struct[0].y_ini[34,0]
    i_l_R18R10_Q = struct[0].y_ini[35,0]
    i_R00_D = struct[0].y_ini[36,0]
    i_R00_Q = struct[0].y_ini[37,0]
    v_R01_D = struct[0].y_ini[38,0]
    v_R01_Q = struct[0].y_ini[39,0]
    v_R02_D = struct[0].y_ini[40,0]
    v_R02_Q = struct[0].y_ini[41,0]
    v_R03_D = struct[0].y_ini[42,0]
    v_R03_Q = struct[0].y_ini[43,0]
    v_R04_D = struct[0].y_ini[44,0]
    v_R04_Q = struct[0].y_ini[45,0]
    v_R05_D = struct[0].y_ini[46,0]
    v_R05_Q = struct[0].y_ini[47,0]
    v_R06_D = struct[0].y_ini[48,0]
    v_R06_Q = struct[0].y_ini[49,0]
    v_R07_D = struct[0].y_ini[50,0]
    v_R07_Q = struct[0].y_ini[51,0]
    v_R08_D = struct[0].y_ini[52,0]
    v_R08_Q = struct[0].y_ini[53,0]
    v_R09_D = struct[0].y_ini[54,0]
    v_R09_Q = struct[0].y_ini[55,0]
    v_R10_D = struct[0].y_ini[56,0]
    v_R10_Q = struct[0].y_ini[57,0]
    v_R11_D = struct[0].y_ini[58,0]
    v_R11_Q = struct[0].y_ini[59,0]
    v_R12_D = struct[0].y_ini[60,0]
    v_R12_Q = struct[0].y_ini[61,0]
    v_R13_D = struct[0].y_ini[62,0]
    v_R13_Q = struct[0].y_ini[63,0]
    v_R14_D = struct[0].y_ini[64,0]
    v_R14_Q = struct[0].y_ini[65,0]
    v_R15_D = struct[0].y_ini[66,0]
    v_R15_Q = struct[0].y_ini[67,0]
    v_R16_D = struct[0].y_ini[68,0]
    v_R16_Q = struct[0].y_ini[69,0]
    v_R17_D = struct[0].y_ini[70,0]
    v_R17_Q = struct[0].y_ini[71,0]
    v_R18_D = struct[0].y_ini[72,0]
    v_R18_Q = struct[0].y_ini[73,0]
    i_R01_d_ref = struct[0].y_ini[74,0]
    i_R01_q_ref = struct[0].y_ini[75,0]
    i_R11_d_ref = struct[0].y_ini[76,0]
    i_R11_q_ref = struct[0].y_ini[77,0]
    i_R15_d_ref = struct[0].y_ini[78,0]
    i_R15_q_ref = struct[0].y_ini[79,0]
    i_R16_d_ref = struct[0].y_ini[80,0]
    i_R16_q_ref = struct[0].y_ini[81,0]
    i_R17_d_ref = struct[0].y_ini[82,0]
    i_R17_q_ref = struct[0].y_ini[83,0]
    i_R18_d_ref = struct[0].y_ini[84,0]
    i_R18_q_ref = struct[0].y_ini[85,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
    
    # Algebraic equations:
    if mode == 3:

        g_n = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = L_R00R01*i_l_R00R01_Q*omega - R_R00R01*i_l_R00R01_D + v_R00_D - v_R01_D
        struct[0].g[1,0] = -L_R00R01*i_l_R00R01_D*omega - R_R00R01*i_l_R00R01_Q + v_R00_Q - v_R01_Q
        struct[0].g[2,0] = g_n[2,0]
        struct[0].g[3,0] = g_n[3,0]
        struct[0].g[4,0] = g_n[4,0]
        struct[0].g[5,0] = g_n[5,0]
        struct[0].g[6,0] = g_n[6,0]
        struct[0].g[7,0] = g_n[7,0]
        struct[0].g[8,0] = g_n[8,0]
        struct[0].g[9,0] = g_n[9,0]
        struct[0].g[10,0] = g_n[10,0]
        struct[0].g[11,0] = g_n[11,0]
        struct[0].g[12,0] = g_n[12,0]
        struct[0].g[13,0] = g_n[13,0]
        struct[0].g[14,0] = g_n[14,0]
        struct[0].g[15,0] = g_n[15,0]
        struct[0].g[16,0] = g_n[16,0]
        struct[0].g[17,0] = g_n[17,0]
        struct[0].g[18,0] = g_n[18,0]
        struct[0].g[19,0] = g_n[19,0]
        struct[0].g[20,0] = g_n[20,0]
        struct[0].g[21,0] = g_n[21,0]
        struct[0].g[22,0] = g_n[22,0]
        struct[0].g[23,0] = g_n[23,0]
        struct[0].g[24,0] = g_n[24,0]
        struct[0].g[25,0] = g_n[25,0]
        struct[0].g[26,0] = g_n[26,0]
        struct[0].g[27,0] = g_n[27,0]
        struct[0].g[28,0] = g_n[28,0]
        struct[0].g[29,0] = g_n[29,0]
        struct[0].g[30,0] = g_n[30,0]
        struct[0].g[31,0] = g_n[31,0]
        struct[0].g[32,0] = g_n[32,0]
        struct[0].g[33,0] = g_n[33,0]
        struct[0].g[34,0] = g_n[34,0]
        struct[0].g[35,0] = g_n[35,0]
        struct[0].g[36,0] = C_R00R01*omega*v_R00_Q/2 + i_R00_D - i_l_R00R01_D
        struct[0].g[37,0] = -C_R00R01*omega*v_R00_D/2 + i_R00_Q - i_l_R00R01_Q
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[56,0] = i_R10_D + i_l_R09R10_D + i_l_R18R10_D + omega*v_R10_Q*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[57,0] = i_R10_Q + i_l_R09R10_Q + i_l_R18R10_Q - omega*v_R10_D*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[64,0] = i_R14_D + i_l_R13R14_D - i_l_R14R15_D + omega*v_R14_Q*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[65,0] = i_R14_Q + i_l_R13R14_Q - i_l_R14R15_Q - omega*v_R14_D*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise((100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise((100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise((100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise((100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise((100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise((100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise((100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise((100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise((100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise((100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise((100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise((100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True))
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (i_R01_D**2 + i_R01_Q**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:

        struct[0].Fy_ini[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)])) 
        struct[0].Fy_ini[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)])) 
        struct[0].Fy_ini[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)])) 
        struct[0].Fy_ini[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)])) 
        struct[0].Fy_ini[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)])) 
        struct[0].Fy_ini[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)])) 
        struct[0].Fy_ini[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)])) 
        struct[0].Fy_ini[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)])) 
        struct[0].Fy_ini[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)])) 
        struct[0].Fy_ini[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)])) 
        struct[0].Fy_ini[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)])) 
        struct[0].Fy_ini[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)])) 

        struct[0].Gx_ini[38,0] = 1
        struct[0].Gx_ini[39,1] = 1
        struct[0].Gx_ini[58,2] = 1
        struct[0].Gx_ini[59,3] = 1
        struct[0].Gx_ini[66,4] = 1
        struct[0].Gx_ini[67,5] = 1
        struct[0].Gx_ini[68,6] = 1
        struct[0].Gx_ini[69,7] = 1
        struct[0].Gx_ini[70,8] = 1
        struct[0].Gx_ini[71,9] = 1
        struct[0].Gx_ini[72,10] = 1
        struct[0].Gx_ini[73,11] = 1

        struct[0].Gy_ini[0,0] = -R_R00R01
        struct[0].Gy_ini[0,1] = L_R00R01*omega
        struct[0].Gy_ini[1,0] = -L_R00R01*omega
        struct[0].Gy_ini[1,1] = -R_R00R01
        struct[0].Gy_ini[2,2] = -R_R02R01
        struct[0].Gy_ini[2,3] = L_R02R01*omega
        struct[0].Gy_ini[3,2] = -L_R02R01*omega
        struct[0].Gy_ini[3,3] = -R_R02R01
        struct[0].Gy_ini[4,4] = -R_R02R03
        struct[0].Gy_ini[4,5] = L_R02R03*omega
        struct[0].Gy_ini[5,4] = -L_R02R03*omega
        struct[0].Gy_ini[5,5] = -R_R02R03
        struct[0].Gy_ini[6,6] = -R_R03R04
        struct[0].Gy_ini[6,7] = L_R03R04*omega
        struct[0].Gy_ini[7,6] = -L_R03R04*omega
        struct[0].Gy_ini[7,7] = -R_R03R04
        struct[0].Gy_ini[8,8] = -R_R04R05
        struct[0].Gy_ini[8,9] = L_R04R05*omega
        struct[0].Gy_ini[9,8] = -L_R04R05*omega
        struct[0].Gy_ini[9,9] = -R_R04R05
        struct[0].Gy_ini[10,10] = -R_R04R12
        struct[0].Gy_ini[10,11] = L_R04R12*omega
        struct[0].Gy_ini[11,10] = -L_R04R12*omega
        struct[0].Gy_ini[11,11] = -R_R04R12
        struct[0].Gy_ini[12,12] = -R_R05R06
        struct[0].Gy_ini[12,13] = L_R05R06*omega
        struct[0].Gy_ini[13,12] = -L_R05R06*omega
        struct[0].Gy_ini[13,13] = -R_R05R06
        struct[0].Gy_ini[14,14] = -R_R06R07
        struct[0].Gy_ini[14,15] = L_R06R07*omega
        struct[0].Gy_ini[15,14] = -L_R06R07*omega
        struct[0].Gy_ini[15,15] = -R_R06R07
        struct[0].Gy_ini[16,16] = -R_R07R08
        struct[0].Gy_ini[16,17] = L_R07R08*omega
        struct[0].Gy_ini[17,16] = -L_R07R08*omega
        struct[0].Gy_ini[17,17] = -R_R07R08
        struct[0].Gy_ini[18,18] = -R_R08R09
        struct[0].Gy_ini[18,19] = L_R08R09*omega
        struct[0].Gy_ini[19,18] = -L_R08R09*omega
        struct[0].Gy_ini[19,19] = -R_R08R09
        struct[0].Gy_ini[20,20] = -R_R09R10
        struct[0].Gy_ini[20,21] = L_R09R10*omega
        struct[0].Gy_ini[21,20] = -L_R09R10*omega
        struct[0].Gy_ini[21,21] = -R_R09R10
        struct[0].Gy_ini[22,22] = -R_R09R17
        struct[0].Gy_ini[22,23] = L_R09R17*omega
        struct[0].Gy_ini[23,22] = -L_R09R17*omega
        struct[0].Gy_ini[23,23] = -R_R09R17
        struct[0].Gy_ini[24,24] = -R_R11R03
        struct[0].Gy_ini[24,25] = L_R11R03*omega
        struct[0].Gy_ini[25,24] = -L_R11R03*omega
        struct[0].Gy_ini[25,25] = -R_R11R03
        struct[0].Gy_ini[26,26] = -R_R12R13
        struct[0].Gy_ini[26,27] = L_R12R13*omega
        struct[0].Gy_ini[27,26] = -L_R12R13*omega
        struct[0].Gy_ini[27,27] = -R_R12R13
        struct[0].Gy_ini[28,28] = -R_R13R14
        struct[0].Gy_ini[28,29] = L_R13R14*omega
        struct[0].Gy_ini[29,28] = -L_R13R14*omega
        struct[0].Gy_ini[29,29] = -R_R13R14
        struct[0].Gy_ini[30,30] = -R_R14R15
        struct[0].Gy_ini[30,31] = L_R14R15*omega
        struct[0].Gy_ini[31,30] = -L_R14R15*omega
        struct[0].Gy_ini[31,31] = -R_R14R15
        struct[0].Gy_ini[32,32] = -R_R16R06
        struct[0].Gy_ini[32,33] = L_R16R06*omega
        struct[0].Gy_ini[33,32] = -L_R16R06*omega
        struct[0].Gy_ini[33,33] = -R_R16R06
        struct[0].Gy_ini[34,34] = -R_R18R10
        struct[0].Gy_ini[34,35] = L_R18R10*omega
        struct[0].Gy_ini[35,34] = -L_R18R10*omega
        struct[0].Gy_ini[35,35] = -R_R18R10
        struct[0].Gy_ini[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[58,59] = C_R11R03*omega/2
        struct[0].Gy_ini[59,58] = -C_R11R03*omega/2
        struct[0].Gy_ini[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[66,67] = C_R14R15*omega/2
        struct[0].Gy_ini[67,66] = -C_R14R15*omega/2
        struct[0].Gy_ini[68,69] = C_R16R06*omega/2
        struct[0].Gy_ini[69,68] = -C_R16R06*omega/2
        struct[0].Gy_ini[70,71] = C_R09R17*omega/2
        struct[0].Gy_ini[71,70] = -C_R09R17*omega/2
        struct[0].Gy_ini[72,73] = C_R18R10*omega/2
        struct[0].Gy_ini[73,72] = -C_R18R10*omega/2
        struct[0].Gy_ini[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R10_D = struct[0].i_R10_D
    i_R10_Q = struct[0].i_R10_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    i_R14_D = struct[0].i_R14_D
    i_R14_Q = struct[0].i_R14_Q
    omega = struct[0].omega
    
    # Inputs:
    v_R00_D = struct[0].v_R00_D
    v_R00_Q = struct[0].v_R00_Q
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_run[0,0]
    i_l_R00R01_Q = struct[0].y_run[1,0]
    i_l_R02R01_D = struct[0].y_run[2,0]
    i_l_R02R01_Q = struct[0].y_run[3,0]
    i_l_R02R03_D = struct[0].y_run[4,0]
    i_l_R02R03_Q = struct[0].y_run[5,0]
    i_l_R03R04_D = struct[0].y_run[6,0]
    i_l_R03R04_Q = struct[0].y_run[7,0]
    i_l_R04R05_D = struct[0].y_run[8,0]
    i_l_R04R05_Q = struct[0].y_run[9,0]
    i_l_R04R12_D = struct[0].y_run[10,0]
    i_l_R04R12_Q = struct[0].y_run[11,0]
    i_l_R05R06_D = struct[0].y_run[12,0]
    i_l_R05R06_Q = struct[0].y_run[13,0]
    i_l_R06R07_D = struct[0].y_run[14,0]
    i_l_R06R07_Q = struct[0].y_run[15,0]
    i_l_R07R08_D = struct[0].y_run[16,0]
    i_l_R07R08_Q = struct[0].y_run[17,0]
    i_l_R08R09_D = struct[0].y_run[18,0]
    i_l_R08R09_Q = struct[0].y_run[19,0]
    i_l_R09R10_D = struct[0].y_run[20,0]
    i_l_R09R10_Q = struct[0].y_run[21,0]
    i_l_R09R17_D = struct[0].y_run[22,0]
    i_l_R09R17_Q = struct[0].y_run[23,0]
    i_l_R11R03_D = struct[0].y_run[24,0]
    i_l_R11R03_Q = struct[0].y_run[25,0]
    i_l_R12R13_D = struct[0].y_run[26,0]
    i_l_R12R13_Q = struct[0].y_run[27,0]
    i_l_R13R14_D = struct[0].y_run[28,0]
    i_l_R13R14_Q = struct[0].y_run[29,0]
    i_l_R14R15_D = struct[0].y_run[30,0]
    i_l_R14R15_Q = struct[0].y_run[31,0]
    i_l_R16R06_D = struct[0].y_run[32,0]
    i_l_R16R06_Q = struct[0].y_run[33,0]
    i_l_R18R10_D = struct[0].y_run[34,0]
    i_l_R18R10_Q = struct[0].y_run[35,0]
    i_R00_D = struct[0].y_run[36,0]
    i_R00_Q = struct[0].y_run[37,0]
    v_R01_D = struct[0].y_run[38,0]
    v_R01_Q = struct[0].y_run[39,0]
    v_R02_D = struct[0].y_run[40,0]
    v_R02_Q = struct[0].y_run[41,0]
    v_R03_D = struct[0].y_run[42,0]
    v_R03_Q = struct[0].y_run[43,0]
    v_R04_D = struct[0].y_run[44,0]
    v_R04_Q = struct[0].y_run[45,0]
    v_R05_D = struct[0].y_run[46,0]
    v_R05_Q = struct[0].y_run[47,0]
    v_R06_D = struct[0].y_run[48,0]
    v_R06_Q = struct[0].y_run[49,0]
    v_R07_D = struct[0].y_run[50,0]
    v_R07_Q = struct[0].y_run[51,0]
    v_R08_D = struct[0].y_run[52,0]
    v_R08_Q = struct[0].y_run[53,0]
    v_R09_D = struct[0].y_run[54,0]
    v_R09_Q = struct[0].y_run[55,0]
    v_R10_D = struct[0].y_run[56,0]
    v_R10_Q = struct[0].y_run[57,0]
    v_R11_D = struct[0].y_run[58,0]
    v_R11_Q = struct[0].y_run[59,0]
    v_R12_D = struct[0].y_run[60,0]
    v_R12_Q = struct[0].y_run[61,0]
    v_R13_D = struct[0].y_run[62,0]
    v_R13_Q = struct[0].y_run[63,0]
    v_R14_D = struct[0].y_run[64,0]
    v_R14_Q = struct[0].y_run[65,0]
    v_R15_D = struct[0].y_run[66,0]
    v_R15_Q = struct[0].y_run[67,0]
    v_R16_D = struct[0].y_run[68,0]
    v_R16_Q = struct[0].y_run[69,0]
    v_R17_D = struct[0].y_run[70,0]
    v_R17_Q = struct[0].y_run[71,0]
    v_R18_D = struct[0].y_run[72,0]
    v_R18_Q = struct[0].y_run[73,0]
    i_R01_d_ref = struct[0].y_run[74,0]
    i_R01_q_ref = struct[0].y_run[75,0]
    i_R11_d_ref = struct[0].y_run[76,0]
    i_R11_q_ref = struct[0].y_run[77,0]
    i_R15_d_ref = struct[0].y_run[78,0]
    i_R15_q_ref = struct[0].y_run[79,0]
    i_R16_d_ref = struct[0].y_run[80,0]
    i_R16_q_ref = struct[0].y_run[81,0]
    i_R17_d_ref = struct[0].y_run[82,0]
    i_R17_q_ref = struct[0].y_run[83,0]
    i_R18_d_ref = struct[0].y_run[84,0]
    i_R18_q_ref = struct[0].y_run[85,0]
    
    struct[0].u_run[0,0] = v_R00_D
    struct[0].u_run[1,0] = v_R00_Q
    struct[0].u_run[2,0] = T_i_R01
    struct[0].u_run[3,0] = I_max_R01
    struct[0].u_run[4,0] = p_R01_ref
    struct[0].u_run[5,0] = q_R01_ref
    struct[0].u_run[6,0] = T_i_R11
    struct[0].u_run[7,0] = I_max_R11
    struct[0].u_run[8,0] = p_R11_ref
    struct[0].u_run[9,0] = q_R11_ref
    struct[0].u_run[10,0] = T_i_R15
    struct[0].u_run[11,0] = I_max_R15
    struct[0].u_run[12,0] = p_R15_ref
    struct[0].u_run[13,0] = q_R15_ref
    struct[0].u_run[14,0] = T_i_R16
    struct[0].u_run[15,0] = I_max_R16
    struct[0].u_run[16,0] = p_R16_ref
    struct[0].u_run[17,0] = q_R16_ref
    struct[0].u_run[18,0] = T_i_R17
    struct[0].u_run[19,0] = I_max_R17
    struct[0].u_run[20,0] = p_R17_ref
    struct[0].u_run[21,0] = q_R17_ref
    struct[0].u_run[22,0] = T_i_R18
    struct[0].u_run[23,0] = I_max_R18
    struct[0].u_run[24,0] = p_R18_ref
    struct[0].u_run[25,0] = q_R18_ref
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
    
    # Algebraic equations:
    if mode == 3:

        g_n = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = g_n[0,0]
        struct[0].g[1,0] = g_n[1,0]
        struct[0].g[2,0] = g_n[2,0]
        struct[0].g[3,0] = g_n[3,0]
        struct[0].g[4,0] = g_n[4,0]
        struct[0].g[5,0] = g_n[5,0]
        struct[0].g[6,0] = g_n[6,0]
        struct[0].g[7,0] = g_n[7,0]
        struct[0].g[8,0] = g_n[8,0]
        struct[0].g[9,0] = g_n[9,0]
        struct[0].g[10,0] = g_n[10,0]
        struct[0].g[11,0] = g_n[11,0]
        struct[0].g[12,0] = g_n[12,0]
        struct[0].g[13,0] = g_n[13,0]
        struct[0].g[14,0] = g_n[14,0]
        struct[0].g[15,0] = g_n[15,0]
        struct[0].g[16,0] = g_n[16,0]
        struct[0].g[17,0] = g_n[17,0]
        struct[0].g[18,0] = g_n[18,0]
        struct[0].g[19,0] = g_n[19,0]
        struct[0].g[20,0] = g_n[20,0]
        struct[0].g[21,0] = g_n[21,0]
        struct[0].g[22,0] = g_n[22,0]
        struct[0].g[23,0] = g_n[23,0]
        struct[0].g[24,0] = g_n[24,0]
        struct[0].g[25,0] = g_n[25,0]
        struct[0].g[26,0] = g_n[26,0]
        struct[0].g[27,0] = g_n[27,0]
        struct[0].g[28,0] = g_n[28,0]
        struct[0].g[29,0] = g_n[29,0]
        struct[0].g[30,0] = g_n[30,0]
        struct[0].g[31,0] = g_n[31,0]
        struct[0].g[32,0] = g_n[32,0]
        struct[0].g[33,0] = g_n[33,0]
        struct[0].g[34,0] = g_n[34,0]
        struct[0].g[35,0] = g_n[35,0]
        struct[0].g[36,0] = g_n[36,0]
        struct[0].g[37,0] = g_n[37,0]
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[56,0] = i_R10_D + i_l_R09R10_D + i_l_R18R10_D + omega*v_R10_Q*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[57,0] = i_R10_Q + i_l_R09R10_Q + i_l_R18R10_Q - omega*v_R10_D*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[64,0] = i_R14_D + i_l_R13R14_D - i_l_R14R15_D + omega*v_R14_Q*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[65,0] = i_R14_Q + i_l_R13R14_Q - i_l_R14R15_Q - omega*v_R14_D*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise((100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise((100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise((100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise((100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise((100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise((100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise((100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise((100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise((100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise((100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise((100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise((100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True))
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (i_R01_D**2 + i_R01_Q**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:

        struct[0].Fy[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)]))
        struct[0].Fy[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)]))
        struct[0].Fy[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)]))
        struct[0].Fy[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)]))
        struct[0].Fy[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)]))
        struct[0].Fy[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)]))
        struct[0].Fy[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)]))
        struct[0].Fy[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)]))
        struct[0].Fy[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)]))
        struct[0].Fy[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)]))
        struct[0].Fy[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)]))
        struct[0].Fy[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)]))

        struct[0].Gx[38,0] = 1
        struct[0].Gx[39,1] = 1
        struct[0].Gx[58,2] = 1
        struct[0].Gx[59,3] = 1
        struct[0].Gx[66,4] = 1
        struct[0].Gx[67,5] = 1
        struct[0].Gx[68,6] = 1
        struct[0].Gx[69,7] = 1
        struct[0].Gx[70,8] = 1
        struct[0].Gx[71,9] = 1
        struct[0].Gx[72,10] = 1
        struct[0].Gx[73,11] = 1

        struct[0].Gy[0,0] = -R_R00R01
        struct[0].Gy[0,1] = L_R00R01*omega
        struct[0].Gy[1,0] = -L_R00R01*omega
        struct[0].Gy[1,1] = -R_R00R01
        struct[0].Gy[2,2] = -R_R02R01
        struct[0].Gy[2,3] = L_R02R01*omega
        struct[0].Gy[3,2] = -L_R02R01*omega
        struct[0].Gy[3,3] = -R_R02R01
        struct[0].Gy[4,4] = -R_R02R03
        struct[0].Gy[4,5] = L_R02R03*omega
        struct[0].Gy[5,4] = -L_R02R03*omega
        struct[0].Gy[5,5] = -R_R02R03
        struct[0].Gy[6,6] = -R_R03R04
        struct[0].Gy[6,7] = L_R03R04*omega
        struct[0].Gy[7,6] = -L_R03R04*omega
        struct[0].Gy[7,7] = -R_R03R04
        struct[0].Gy[8,8] = -R_R04R05
        struct[0].Gy[8,9] = L_R04R05*omega
        struct[0].Gy[9,8] = -L_R04R05*omega
        struct[0].Gy[9,9] = -R_R04R05
        struct[0].Gy[10,10] = -R_R04R12
        struct[0].Gy[10,11] = L_R04R12*omega
        struct[0].Gy[11,10] = -L_R04R12*omega
        struct[0].Gy[11,11] = -R_R04R12
        struct[0].Gy[12,12] = -R_R05R06
        struct[0].Gy[12,13] = L_R05R06*omega
        struct[0].Gy[13,12] = -L_R05R06*omega
        struct[0].Gy[13,13] = -R_R05R06
        struct[0].Gy[14,14] = -R_R06R07
        struct[0].Gy[14,15] = L_R06R07*omega
        struct[0].Gy[15,14] = -L_R06R07*omega
        struct[0].Gy[15,15] = -R_R06R07
        struct[0].Gy[16,16] = -R_R07R08
        struct[0].Gy[16,17] = L_R07R08*omega
        struct[0].Gy[17,16] = -L_R07R08*omega
        struct[0].Gy[17,17] = -R_R07R08
        struct[0].Gy[18,18] = -R_R08R09
        struct[0].Gy[18,19] = L_R08R09*omega
        struct[0].Gy[19,18] = -L_R08R09*omega
        struct[0].Gy[19,19] = -R_R08R09
        struct[0].Gy[20,20] = -R_R09R10
        struct[0].Gy[20,21] = L_R09R10*omega
        struct[0].Gy[21,20] = -L_R09R10*omega
        struct[0].Gy[21,21] = -R_R09R10
        struct[0].Gy[22,22] = -R_R09R17
        struct[0].Gy[22,23] = L_R09R17*omega
        struct[0].Gy[23,22] = -L_R09R17*omega
        struct[0].Gy[23,23] = -R_R09R17
        struct[0].Gy[24,24] = -R_R11R03
        struct[0].Gy[24,25] = L_R11R03*omega
        struct[0].Gy[25,24] = -L_R11R03*omega
        struct[0].Gy[25,25] = -R_R11R03
        struct[0].Gy[26,26] = -R_R12R13
        struct[0].Gy[26,27] = L_R12R13*omega
        struct[0].Gy[27,26] = -L_R12R13*omega
        struct[0].Gy[27,27] = -R_R12R13
        struct[0].Gy[28,28] = -R_R13R14
        struct[0].Gy[28,29] = L_R13R14*omega
        struct[0].Gy[29,28] = -L_R13R14*omega
        struct[0].Gy[29,29] = -R_R13R14
        struct[0].Gy[30,30] = -R_R14R15
        struct[0].Gy[30,31] = L_R14R15*omega
        struct[0].Gy[31,30] = -L_R14R15*omega
        struct[0].Gy[31,31] = -R_R14R15
        struct[0].Gy[32,32] = -R_R16R06
        struct[0].Gy[32,33] = L_R16R06*omega
        struct[0].Gy[33,32] = -L_R16R06*omega
        struct[0].Gy[33,33] = -R_R16R06
        struct[0].Gy[34,34] = -R_R18R10
        struct[0].Gy[34,35] = L_R18R10*omega
        struct[0].Gy[35,34] = -L_R18R10*omega
        struct[0].Gy[35,35] = -R_R18R10
        struct[0].Gy[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[58,59] = C_R11R03*omega/2
        struct[0].Gy[59,58] = -C_R11R03*omega/2
        struct[0].Gy[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[66,67] = C_R14R15*omega/2
        struct[0].Gy[67,66] = -C_R14R15*omega/2
        struct[0].Gy[68,69] = C_R16R06*omega/2
        struct[0].Gy[69,68] = -C_R16R06*omega/2
        struct[0].Gy[70,71] = C_R09R17*omega/2
        struct[0].Gy[71,70] = -C_R09R17*omega/2
        struct[0].Gy[72,73] = C_R18R10*omega/2
        struct[0].Gy[73,72] = -C_R18R10*omega/2
        struct[0].Gy[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))

    if mode > 12:

        struct[0].Fu[0,3] = 100.0*Piecewise(np.array([(-1, I_max_R01 < -i_R01_d_ref), (1, I_max_R01 < i_R01_d_ref), (0, True)]))
        struct[0].Fu[1,3] = 100.0*Piecewise(np.array([(-1, I_max_R01 < -i_R01_q_ref), (1, I_max_R01 < i_R01_q_ref), (0, True)]))
        struct[0].Fu[2,7] = 100.0*Piecewise(np.array([(-1, I_max_R11 < -i_R11_d_ref), (1, I_max_R11 < i_R11_d_ref), (0, True)]))
        struct[0].Fu[3,7] = 100.0*Piecewise(np.array([(-1, I_max_R11 < -i_R11_q_ref), (1, I_max_R11 < i_R11_q_ref), (0, True)]))
        struct[0].Fu[4,11] = 100.0*Piecewise(np.array([(-1, I_max_R15 < -i_R15_d_ref), (1, I_max_R15 < i_R15_d_ref), (0, True)]))
        struct[0].Fu[5,11] = 100.0*Piecewise(np.array([(-1, I_max_R15 < -i_R15_q_ref), (1, I_max_R15 < i_R15_q_ref), (0, True)]))
        struct[0].Fu[6,15] = 100.0*Piecewise(np.array([(-1, I_max_R16 < -i_R16_d_ref), (1, I_max_R16 < i_R16_d_ref), (0, True)]))
        struct[0].Fu[7,15] = 100.0*Piecewise(np.array([(-1, I_max_R16 < -i_R16_q_ref), (1, I_max_R16 < i_R16_q_ref), (0, True)]))
        struct[0].Fu[8,19] = 100.0*Piecewise(np.array([(-1, I_max_R17 < -i_R17_d_ref), (1, I_max_R17 < i_R17_d_ref), (0, True)]))
        struct[0].Fu[9,19] = 100.0*Piecewise(np.array([(-1, I_max_R17 < -i_R17_q_ref), (1, I_max_R17 < i_R17_q_ref), (0, True)]))
        struct[0].Fu[10,23] = 100.0*Piecewise(np.array([(-1, I_max_R18 < -i_R18_d_ref), (1, I_max_R18 < i_R18_d_ref), (0, True)]))
        struct[0].Fu[11,23] = 100.0*Piecewise(np.array([(-1, I_max_R18 < -i_R18_q_ref), (1, I_max_R18 < i_R18_q_ref), (0, True)]))

        struct[0].Gu[36,1] = C_R00R01*omega/2
        struct[0].Gu[37,0] = -C_R00R01*omega/2
        struct[0].Gu[74,4] = -0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[74,5] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,4] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,5] = 0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[76,8] = -0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[76,9] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,8] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,9] = 0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[78,12] = -0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[78,13] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,12] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,13] = 0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[80,16] = -0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[80,17] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,16] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,17] = 0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[82,20] = -0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[82,21] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,20] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,21] = 0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[84,24] = -0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[84,25] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,24] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,25] = 0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))

        struct[0].Hx[0,0] = 1.0*i_R01_D*(i_R01_D**2 + i_R01_Q**2)**(-0.5)
        struct[0].Hx[0,1] = 1.0*i_R01_Q*(i_R01_D**2 + i_R01_Q**2)**(-0.5)





def ini_nn(struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R10_D = struct[0].i_R10_D
    i_R10_Q = struct[0].i_R10_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    i_R14_D = struct[0].i_R14_D
    i_R14_Q = struct[0].i_R14_Q
    omega = struct[0].omega
    
    # Inputs:
    v_R00_D = struct[0].v_R00_D
    v_R00_Q = struct[0].v_R00_Q
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_ini[0,0]
    i_l_R00R01_Q = struct[0].y_ini[1,0]
    i_l_R02R01_D = struct[0].y_ini[2,0]
    i_l_R02R01_Q = struct[0].y_ini[3,0]
    i_l_R02R03_D = struct[0].y_ini[4,0]
    i_l_R02R03_Q = struct[0].y_ini[5,0]
    i_l_R03R04_D = struct[0].y_ini[6,0]
    i_l_R03R04_Q = struct[0].y_ini[7,0]
    i_l_R04R05_D = struct[0].y_ini[8,0]
    i_l_R04R05_Q = struct[0].y_ini[9,0]
    i_l_R04R12_D = struct[0].y_ini[10,0]
    i_l_R04R12_Q = struct[0].y_ini[11,0]
    i_l_R05R06_D = struct[0].y_ini[12,0]
    i_l_R05R06_Q = struct[0].y_ini[13,0]
    i_l_R06R07_D = struct[0].y_ini[14,0]
    i_l_R06R07_Q = struct[0].y_ini[15,0]
    i_l_R07R08_D = struct[0].y_ini[16,0]
    i_l_R07R08_Q = struct[0].y_ini[17,0]
    i_l_R08R09_D = struct[0].y_ini[18,0]
    i_l_R08R09_Q = struct[0].y_ini[19,0]
    i_l_R09R10_D = struct[0].y_ini[20,0]
    i_l_R09R10_Q = struct[0].y_ini[21,0]
    i_l_R09R17_D = struct[0].y_ini[22,0]
    i_l_R09R17_Q = struct[0].y_ini[23,0]
    i_l_R11R03_D = struct[0].y_ini[24,0]
    i_l_R11R03_Q = struct[0].y_ini[25,0]
    i_l_R12R13_D = struct[0].y_ini[26,0]
    i_l_R12R13_Q = struct[0].y_ini[27,0]
    i_l_R13R14_D = struct[0].y_ini[28,0]
    i_l_R13R14_Q = struct[0].y_ini[29,0]
    i_l_R14R15_D = struct[0].y_ini[30,0]
    i_l_R14R15_Q = struct[0].y_ini[31,0]
    i_l_R16R06_D = struct[0].y_ini[32,0]
    i_l_R16R06_Q = struct[0].y_ini[33,0]
    i_l_R18R10_D = struct[0].y_ini[34,0]
    i_l_R18R10_Q = struct[0].y_ini[35,0]
    i_R00_D = struct[0].y_ini[36,0]
    i_R00_Q = struct[0].y_ini[37,0]
    v_R01_D = struct[0].y_ini[38,0]
    v_R01_Q = struct[0].y_ini[39,0]
    v_R02_D = struct[0].y_ini[40,0]
    v_R02_Q = struct[0].y_ini[41,0]
    v_R03_D = struct[0].y_ini[42,0]
    v_R03_Q = struct[0].y_ini[43,0]
    v_R04_D = struct[0].y_ini[44,0]
    v_R04_Q = struct[0].y_ini[45,0]
    v_R05_D = struct[0].y_ini[46,0]
    v_R05_Q = struct[0].y_ini[47,0]
    v_R06_D = struct[0].y_ini[48,0]
    v_R06_Q = struct[0].y_ini[49,0]
    v_R07_D = struct[0].y_ini[50,0]
    v_R07_Q = struct[0].y_ini[51,0]
    v_R08_D = struct[0].y_ini[52,0]
    v_R08_Q = struct[0].y_ini[53,0]
    v_R09_D = struct[0].y_ini[54,0]
    v_R09_Q = struct[0].y_ini[55,0]
    v_R10_D = struct[0].y_ini[56,0]
    v_R10_Q = struct[0].y_ini[57,0]
    v_R11_D = struct[0].y_ini[58,0]
    v_R11_Q = struct[0].y_ini[59,0]
    v_R12_D = struct[0].y_ini[60,0]
    v_R12_Q = struct[0].y_ini[61,0]
    v_R13_D = struct[0].y_ini[62,0]
    v_R13_Q = struct[0].y_ini[63,0]
    v_R14_D = struct[0].y_ini[64,0]
    v_R14_Q = struct[0].y_ini[65,0]
    v_R15_D = struct[0].y_ini[66,0]
    v_R15_Q = struct[0].y_ini[67,0]
    v_R16_D = struct[0].y_ini[68,0]
    v_R16_Q = struct[0].y_ini[69,0]
    v_R17_D = struct[0].y_ini[70,0]
    v_R17_Q = struct[0].y_ini[71,0]
    v_R18_D = struct[0].y_ini[72,0]
    v_R18_Q = struct[0].y_ini[73,0]
    i_R01_d_ref = struct[0].y_ini[74,0]
    i_R01_q_ref = struct[0].y_ini[75,0]
    i_R11_d_ref = struct[0].y_ini[76,0]
    i_R11_q_ref = struct[0].y_ini[77,0]
    i_R15_d_ref = struct[0].y_ini[78,0]
    i_R15_q_ref = struct[0].y_ini[79,0]
    i_R16_d_ref = struct[0].y_ini[80,0]
    i_R16_q_ref = struct[0].y_ini[81,0]
    i_R17_d_ref = struct[0].y_ini[82,0]
    i_R17_q_ref = struct[0].y_ini[83,0]
    i_R18_d_ref = struct[0].y_ini[84,0]
    i_R18_q_ref = struct[0].y_ini[85,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = L_R00R01*i_l_R00R01_Q*omega - R_R00R01*i_l_R00R01_D + v_R00_D - v_R01_D
        struct[0].g[1,0] = -L_R00R01*i_l_R00R01_D*omega - R_R00R01*i_l_R00R01_Q + v_R00_Q - v_R01_Q
        struct[0].g[2,0] = L_R02R01*i_l_R02R01_Q*omega - R_R02R01*i_l_R02R01_D - v_R01_D + v_R02_D
        struct[0].g[3,0] = -L_R02R01*i_l_R02R01_D*omega - R_R02R01*i_l_R02R01_Q - v_R01_Q + v_R02_Q
        struct[0].g[4,0] = L_R02R03*i_l_R02R03_Q*omega - R_R02R03*i_l_R02R03_D + v_R02_D - v_R03_D
        struct[0].g[5,0] = -L_R02R03*i_l_R02R03_D*omega - R_R02R03*i_l_R02R03_Q + v_R02_Q - v_R03_Q
        struct[0].g[6,0] = L_R03R04*i_l_R03R04_Q*omega - R_R03R04*i_l_R03R04_D + v_R03_D - v_R04_D
        struct[0].g[7,0] = -L_R03R04*i_l_R03R04_D*omega - R_R03R04*i_l_R03R04_Q + v_R03_Q - v_R04_Q
        struct[0].g[8,0] = L_R04R05*i_l_R04R05_Q*omega - R_R04R05*i_l_R04R05_D + v_R04_D - v_R05_D
        struct[0].g[9,0] = -L_R04R05*i_l_R04R05_D*omega - R_R04R05*i_l_R04R05_Q + v_R04_Q - v_R05_Q
        struct[0].g[10,0] = L_R04R12*i_l_R04R12_Q*omega - R_R04R12*i_l_R04R12_D + v_R04_D - v_R12_D
        struct[0].g[11,0] = -L_R04R12*i_l_R04R12_D*omega - R_R04R12*i_l_R04R12_Q + v_R04_Q - v_R12_Q
        struct[0].g[12,0] = L_R05R06*i_l_R05R06_Q*omega - R_R05R06*i_l_R05R06_D + v_R05_D - v_R06_D
        struct[0].g[13,0] = -L_R05R06*i_l_R05R06_D*omega - R_R05R06*i_l_R05R06_Q + v_R05_Q - v_R06_Q
        struct[0].g[14,0] = L_R06R07*i_l_R06R07_Q*omega - R_R06R07*i_l_R06R07_D + v_R06_D - v_R07_D
        struct[0].g[15,0] = -L_R06R07*i_l_R06R07_D*omega - R_R06R07*i_l_R06R07_Q + v_R06_Q - v_R07_Q
        struct[0].g[16,0] = L_R07R08*i_l_R07R08_Q*omega - R_R07R08*i_l_R07R08_D + v_R07_D - v_R08_D
        struct[0].g[17,0] = -L_R07R08*i_l_R07R08_D*omega - R_R07R08*i_l_R07R08_Q + v_R07_Q - v_R08_Q
        struct[0].g[18,0] = L_R08R09*i_l_R08R09_Q*omega - R_R08R09*i_l_R08R09_D + v_R08_D - v_R09_D
        struct[0].g[19,0] = -L_R08R09*i_l_R08R09_D*omega - R_R08R09*i_l_R08R09_Q + v_R08_Q - v_R09_Q
        struct[0].g[20,0] = L_R09R10*i_l_R09R10_Q*omega - R_R09R10*i_l_R09R10_D + v_R09_D - v_R10_D
        struct[0].g[21,0] = -L_R09R10*i_l_R09R10_D*omega - R_R09R10*i_l_R09R10_Q + v_R09_Q - v_R10_Q
        struct[0].g[22,0] = L_R09R17*i_l_R09R17_Q*omega - R_R09R17*i_l_R09R17_D + v_R09_D - v_R17_D
        struct[0].g[23,0] = -L_R09R17*i_l_R09R17_D*omega - R_R09R17*i_l_R09R17_Q + v_R09_Q - v_R17_Q
        struct[0].g[24,0] = L_R11R03*i_l_R11R03_Q*omega - R_R11R03*i_l_R11R03_D - v_R03_D + v_R11_D
        struct[0].g[25,0] = -L_R11R03*i_l_R11R03_D*omega - R_R11R03*i_l_R11R03_Q - v_R03_Q + v_R11_Q
        struct[0].g[26,0] = L_R12R13*i_l_R12R13_Q*omega - R_R12R13*i_l_R12R13_D + v_R12_D - v_R13_D
        struct[0].g[27,0] = -L_R12R13*i_l_R12R13_D*omega - R_R12R13*i_l_R12R13_Q + v_R12_Q - v_R13_Q
        struct[0].g[28,0] = L_R13R14*i_l_R13R14_Q*omega - R_R13R14*i_l_R13R14_D + v_R13_D - v_R14_D
        struct[0].g[29,0] = -L_R13R14*i_l_R13R14_D*omega - R_R13R14*i_l_R13R14_Q + v_R13_Q - v_R14_Q
        struct[0].g[30,0] = L_R14R15*i_l_R14R15_Q*omega - R_R14R15*i_l_R14R15_D + v_R14_D - v_R15_D
        struct[0].g[31,0] = -L_R14R15*i_l_R14R15_D*omega - R_R14R15*i_l_R14R15_Q + v_R14_Q - v_R15_Q
        struct[0].g[32,0] = L_R16R06*i_l_R16R06_Q*omega - R_R16R06*i_l_R16R06_D - v_R06_D + v_R16_D
        struct[0].g[33,0] = -L_R16R06*i_l_R16R06_D*omega - R_R16R06*i_l_R16R06_Q - v_R06_Q + v_R16_Q
        struct[0].g[34,0] = L_R18R10*i_l_R18R10_Q*omega - R_R18R10*i_l_R18R10_D - v_R10_D + v_R18_D
        struct[0].g[35,0] = -L_R18R10*i_l_R18R10_D*omega - R_R18R10*i_l_R18R10_Q - v_R10_Q + v_R18_Q
        struct[0].g[36,0] = C_R00R01*omega*v_R00_Q/2 + i_R00_D - i_l_R00R01_D
        struct[0].g[37,0] = -C_R00R01*omega*v_R00_D/2 + i_R00_Q - i_l_R00R01_Q
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[56,0] = i_R10_D + i_l_R09R10_D + i_l_R18R10_D + omega*v_R10_Q*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[57,0] = i_R10_Q + i_l_R09R10_Q + i_l_R18R10_Q - omega*v_R10_D*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[64,0] = i_R14_D + i_l_R13R14_D - i_l_R14R15_D + omega*v_R14_Q*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[65,0] = i_R14_Q + i_l_R13R14_Q - i_l_R14R15_Q - omega*v_R14_D*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (i_R01_D**2 + i_R01_Q**2)**0.5
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -100.000000000000
        struct[0].Fx_ini[1,1] = -100.000000000000
        struct[0].Fx_ini[2,2] = -100.000000000000
        struct[0].Fx_ini[3,3] = -100.000000000000
        struct[0].Fx_ini[4,4] = -100.000000000000
        struct[0].Fx_ini[5,5] = -100.000000000000
        struct[0].Fx_ini[6,6] = -100.000000000000
        struct[0].Fx_ini[7,7] = -100.000000000000
        struct[0].Fx_ini[8,8] = -100.000000000000
        struct[0].Fx_ini[9,9] = -100.000000000000
        struct[0].Fx_ini[10,10] = -100.000000000000
        struct[0].Fx_ini[11,11] = -100.000000000000

    if mode == 11:

        struct[0].Fy_ini[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)])) 
        struct[0].Fy_ini[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)])) 
        struct[0].Fy_ini[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)])) 
        struct[0].Fy_ini[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)])) 
        struct[0].Fy_ini[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)])) 
        struct[0].Fy_ini[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)])) 
        struct[0].Fy_ini[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)])) 
        struct[0].Fy_ini[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)])) 
        struct[0].Fy_ini[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)])) 
        struct[0].Fy_ini[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)])) 
        struct[0].Fy_ini[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)])) 
        struct[0].Fy_ini[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)])) 

        struct[0].Gy_ini[0,0] = -R_R00R01
        struct[0].Gy_ini[0,1] = L_R00R01*omega
        struct[0].Gy_ini[0,38] = -1
        struct[0].Gy_ini[1,0] = -L_R00R01*omega
        struct[0].Gy_ini[1,1] = -R_R00R01
        struct[0].Gy_ini[1,39] = -1
        struct[0].Gy_ini[2,2] = -R_R02R01
        struct[0].Gy_ini[2,3] = L_R02R01*omega
        struct[0].Gy_ini[2,38] = -1
        struct[0].Gy_ini[2,40] = 1
        struct[0].Gy_ini[3,2] = -L_R02R01*omega
        struct[0].Gy_ini[3,3] = -R_R02R01
        struct[0].Gy_ini[3,39] = -1
        struct[0].Gy_ini[3,41] = 1
        struct[0].Gy_ini[4,4] = -R_R02R03
        struct[0].Gy_ini[4,5] = L_R02R03*omega
        struct[0].Gy_ini[4,40] = 1
        struct[0].Gy_ini[4,42] = -1
        struct[0].Gy_ini[5,4] = -L_R02R03*omega
        struct[0].Gy_ini[5,5] = -R_R02R03
        struct[0].Gy_ini[5,41] = 1
        struct[0].Gy_ini[5,43] = -1
        struct[0].Gy_ini[6,6] = -R_R03R04
        struct[0].Gy_ini[6,7] = L_R03R04*omega
        struct[0].Gy_ini[6,42] = 1
        struct[0].Gy_ini[6,44] = -1
        struct[0].Gy_ini[7,6] = -L_R03R04*omega
        struct[0].Gy_ini[7,7] = -R_R03R04
        struct[0].Gy_ini[7,43] = 1
        struct[0].Gy_ini[7,45] = -1
        struct[0].Gy_ini[8,8] = -R_R04R05
        struct[0].Gy_ini[8,9] = L_R04R05*omega
        struct[0].Gy_ini[8,44] = 1
        struct[0].Gy_ini[8,46] = -1
        struct[0].Gy_ini[9,8] = -L_R04R05*omega
        struct[0].Gy_ini[9,9] = -R_R04R05
        struct[0].Gy_ini[9,45] = 1
        struct[0].Gy_ini[9,47] = -1
        struct[0].Gy_ini[10,10] = -R_R04R12
        struct[0].Gy_ini[10,11] = L_R04R12*omega
        struct[0].Gy_ini[10,44] = 1
        struct[0].Gy_ini[10,60] = -1
        struct[0].Gy_ini[11,10] = -L_R04R12*omega
        struct[0].Gy_ini[11,11] = -R_R04R12
        struct[0].Gy_ini[11,45] = 1
        struct[0].Gy_ini[11,61] = -1
        struct[0].Gy_ini[12,12] = -R_R05R06
        struct[0].Gy_ini[12,13] = L_R05R06*omega
        struct[0].Gy_ini[12,46] = 1
        struct[0].Gy_ini[12,48] = -1
        struct[0].Gy_ini[13,12] = -L_R05R06*omega
        struct[0].Gy_ini[13,13] = -R_R05R06
        struct[0].Gy_ini[13,47] = 1
        struct[0].Gy_ini[13,49] = -1
        struct[0].Gy_ini[14,14] = -R_R06R07
        struct[0].Gy_ini[14,15] = L_R06R07*omega
        struct[0].Gy_ini[14,48] = 1
        struct[0].Gy_ini[14,50] = -1
        struct[0].Gy_ini[15,14] = -L_R06R07*omega
        struct[0].Gy_ini[15,15] = -R_R06R07
        struct[0].Gy_ini[15,49] = 1
        struct[0].Gy_ini[15,51] = -1
        struct[0].Gy_ini[16,16] = -R_R07R08
        struct[0].Gy_ini[16,17] = L_R07R08*omega
        struct[0].Gy_ini[16,50] = 1
        struct[0].Gy_ini[16,52] = -1
        struct[0].Gy_ini[17,16] = -L_R07R08*omega
        struct[0].Gy_ini[17,17] = -R_R07R08
        struct[0].Gy_ini[17,51] = 1
        struct[0].Gy_ini[17,53] = -1
        struct[0].Gy_ini[18,18] = -R_R08R09
        struct[0].Gy_ini[18,19] = L_R08R09*omega
        struct[0].Gy_ini[18,52] = 1
        struct[0].Gy_ini[18,54] = -1
        struct[0].Gy_ini[19,18] = -L_R08R09*omega
        struct[0].Gy_ini[19,19] = -R_R08R09
        struct[0].Gy_ini[19,53] = 1
        struct[0].Gy_ini[19,55] = -1
        struct[0].Gy_ini[20,20] = -R_R09R10
        struct[0].Gy_ini[20,21] = L_R09R10*omega
        struct[0].Gy_ini[20,54] = 1
        struct[0].Gy_ini[20,56] = -1
        struct[0].Gy_ini[21,20] = -L_R09R10*omega
        struct[0].Gy_ini[21,21] = -R_R09R10
        struct[0].Gy_ini[21,55] = 1
        struct[0].Gy_ini[21,57] = -1
        struct[0].Gy_ini[22,22] = -R_R09R17
        struct[0].Gy_ini[22,23] = L_R09R17*omega
        struct[0].Gy_ini[22,54] = 1
        struct[0].Gy_ini[22,70] = -1
        struct[0].Gy_ini[23,22] = -L_R09R17*omega
        struct[0].Gy_ini[23,23] = -R_R09R17
        struct[0].Gy_ini[23,55] = 1
        struct[0].Gy_ini[23,71] = -1
        struct[0].Gy_ini[24,24] = -R_R11R03
        struct[0].Gy_ini[24,25] = L_R11R03*omega
        struct[0].Gy_ini[24,42] = -1
        struct[0].Gy_ini[24,58] = 1
        struct[0].Gy_ini[25,24] = -L_R11R03*omega
        struct[0].Gy_ini[25,25] = -R_R11R03
        struct[0].Gy_ini[25,43] = -1
        struct[0].Gy_ini[25,59] = 1
        struct[0].Gy_ini[26,26] = -R_R12R13
        struct[0].Gy_ini[26,27] = L_R12R13*omega
        struct[0].Gy_ini[26,60] = 1
        struct[0].Gy_ini[26,62] = -1
        struct[0].Gy_ini[27,26] = -L_R12R13*omega
        struct[0].Gy_ini[27,27] = -R_R12R13
        struct[0].Gy_ini[27,61] = 1
        struct[0].Gy_ini[27,63] = -1
        struct[0].Gy_ini[28,28] = -R_R13R14
        struct[0].Gy_ini[28,29] = L_R13R14*omega
        struct[0].Gy_ini[28,62] = 1
        struct[0].Gy_ini[28,64] = -1
        struct[0].Gy_ini[29,28] = -L_R13R14*omega
        struct[0].Gy_ini[29,29] = -R_R13R14
        struct[0].Gy_ini[29,63] = 1
        struct[0].Gy_ini[29,65] = -1
        struct[0].Gy_ini[30,30] = -R_R14R15
        struct[0].Gy_ini[30,31] = L_R14R15*omega
        struct[0].Gy_ini[30,64] = 1
        struct[0].Gy_ini[30,66] = -1
        struct[0].Gy_ini[31,30] = -L_R14R15*omega
        struct[0].Gy_ini[31,31] = -R_R14R15
        struct[0].Gy_ini[31,65] = 1
        struct[0].Gy_ini[31,67] = -1
        struct[0].Gy_ini[32,32] = -R_R16R06
        struct[0].Gy_ini[32,33] = L_R16R06*omega
        struct[0].Gy_ini[32,48] = -1
        struct[0].Gy_ini[32,68] = 1
        struct[0].Gy_ini[33,32] = -L_R16R06*omega
        struct[0].Gy_ini[33,33] = -R_R16R06
        struct[0].Gy_ini[33,49] = -1
        struct[0].Gy_ini[33,69] = 1
        struct[0].Gy_ini[34,34] = -R_R18R10
        struct[0].Gy_ini[34,35] = L_R18R10*omega
        struct[0].Gy_ini[34,56] = -1
        struct[0].Gy_ini[34,72] = 1
        struct[0].Gy_ini[35,34] = -L_R18R10*omega
        struct[0].Gy_ini[35,35] = -R_R18R10
        struct[0].Gy_ini[35,57] = -1
        struct[0].Gy_ini[35,73] = 1
        struct[0].Gy_ini[36,0] = -1
        struct[0].Gy_ini[36,36] = 1
        struct[0].Gy_ini[37,1] = -1
        struct[0].Gy_ini[37,37] = 1
        struct[0].Gy_ini[38,0] = 1
        struct[0].Gy_ini[38,2] = 1
        struct[0].Gy_ini[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[39,1] = 1
        struct[0].Gy_ini[39,3] = 1
        struct[0].Gy_ini[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[40,2] = -1
        struct[0].Gy_ini[40,4] = -1
        struct[0].Gy_ini[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[41,3] = -1
        struct[0].Gy_ini[41,5] = -1
        struct[0].Gy_ini[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[42,4] = 1
        struct[0].Gy_ini[42,6] = -1
        struct[0].Gy_ini[42,24] = 1
        struct[0].Gy_ini[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[43,5] = 1
        struct[0].Gy_ini[43,7] = -1
        struct[0].Gy_ini[43,25] = 1
        struct[0].Gy_ini[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[44,6] = 1
        struct[0].Gy_ini[44,8] = -1
        struct[0].Gy_ini[44,10] = -1
        struct[0].Gy_ini[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[45,7] = 1
        struct[0].Gy_ini[45,9] = -1
        struct[0].Gy_ini[45,11] = -1
        struct[0].Gy_ini[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[46,8] = 1
        struct[0].Gy_ini[46,12] = -1
        struct[0].Gy_ini[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[47,9] = 1
        struct[0].Gy_ini[47,13] = -1
        struct[0].Gy_ini[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[48,12] = 1
        struct[0].Gy_ini[48,14] = -1
        struct[0].Gy_ini[48,32] = 1
        struct[0].Gy_ini[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[49,13] = 1
        struct[0].Gy_ini[49,15] = -1
        struct[0].Gy_ini[49,33] = 1
        struct[0].Gy_ini[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[50,14] = 1
        struct[0].Gy_ini[50,16] = -1
        struct[0].Gy_ini[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[51,15] = 1
        struct[0].Gy_ini[51,17] = -1
        struct[0].Gy_ini[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[52,16] = 1
        struct[0].Gy_ini[52,18] = -1
        struct[0].Gy_ini[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[53,17] = 1
        struct[0].Gy_ini[53,19] = -1
        struct[0].Gy_ini[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[54,18] = 1
        struct[0].Gy_ini[54,20] = -1
        struct[0].Gy_ini[54,22] = -1
        struct[0].Gy_ini[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[55,19] = 1
        struct[0].Gy_ini[55,21] = -1
        struct[0].Gy_ini[55,23] = -1
        struct[0].Gy_ini[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[56,20] = 1
        struct[0].Gy_ini[56,34] = 1
        struct[0].Gy_ini[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[57,21] = 1
        struct[0].Gy_ini[57,35] = 1
        struct[0].Gy_ini[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[58,24] = -1
        struct[0].Gy_ini[58,59] = C_R11R03*omega/2
        struct[0].Gy_ini[59,25] = -1
        struct[0].Gy_ini[59,58] = -C_R11R03*omega/2
        struct[0].Gy_ini[60,10] = 1
        struct[0].Gy_ini[60,26] = -1
        struct[0].Gy_ini[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[61,11] = 1
        struct[0].Gy_ini[61,27] = -1
        struct[0].Gy_ini[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[62,26] = 1
        struct[0].Gy_ini[62,28] = -1
        struct[0].Gy_ini[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[63,27] = 1
        struct[0].Gy_ini[63,29] = -1
        struct[0].Gy_ini[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[64,28] = 1
        struct[0].Gy_ini[64,30] = -1
        struct[0].Gy_ini[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[65,29] = 1
        struct[0].Gy_ini[65,31] = -1
        struct[0].Gy_ini[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[66,30] = 1
        struct[0].Gy_ini[66,67] = C_R14R15*omega/2
        struct[0].Gy_ini[67,31] = 1
        struct[0].Gy_ini[67,66] = -C_R14R15*omega/2
        struct[0].Gy_ini[68,32] = -1
        struct[0].Gy_ini[68,69] = C_R16R06*omega/2
        struct[0].Gy_ini[69,33] = -1
        struct[0].Gy_ini[69,68] = -C_R16R06*omega/2
        struct[0].Gy_ini[70,22] = 1
        struct[0].Gy_ini[70,71] = C_R09R17*omega/2
        struct[0].Gy_ini[71,23] = 1
        struct[0].Gy_ini[71,70] = -C_R09R17*omega/2
        struct[0].Gy_ini[72,34] = -1
        struct[0].Gy_ini[72,73] = C_R18R10*omega/2
        struct[0].Gy_ini[73,35] = -1
        struct[0].Gy_ini[73,72] = -C_R18R10*omega/2
        struct[0].Gy_ini[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[74,74] = -1
        struct[0].Gy_ini[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,75] = -1
        struct[0].Gy_ini[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[76,76] = -1
        struct[0].Gy_ini[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,77] = -1
        struct[0].Gy_ini[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[78,78] = -1
        struct[0].Gy_ini[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,79] = -1
        struct[0].Gy_ini[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[80,80] = -1
        struct[0].Gy_ini[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,81] = -1
        struct[0].Gy_ini[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[82,82] = -1
        struct[0].Gy_ini[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,83] = -1
        struct[0].Gy_ini[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[84,84] = -1
        struct[0].Gy_ini[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,85] = -1



def run_nn(t,struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R10_D = struct[0].i_R10_D
    i_R10_Q = struct[0].i_R10_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    i_R14_D = struct[0].i_R14_D
    i_R14_Q = struct[0].i_R14_Q
    omega = struct[0].omega
    
    # Inputs:
    v_R00_D = struct[0].v_R00_D
    v_R00_Q = struct[0].v_R00_Q
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_run[0,0]
    i_l_R00R01_Q = struct[0].y_run[1,0]
    i_l_R02R01_D = struct[0].y_run[2,0]
    i_l_R02R01_Q = struct[0].y_run[3,0]
    i_l_R02R03_D = struct[0].y_run[4,0]
    i_l_R02R03_Q = struct[0].y_run[5,0]
    i_l_R03R04_D = struct[0].y_run[6,0]
    i_l_R03R04_Q = struct[0].y_run[7,0]
    i_l_R04R05_D = struct[0].y_run[8,0]
    i_l_R04R05_Q = struct[0].y_run[9,0]
    i_l_R04R12_D = struct[0].y_run[10,0]
    i_l_R04R12_Q = struct[0].y_run[11,0]
    i_l_R05R06_D = struct[0].y_run[12,0]
    i_l_R05R06_Q = struct[0].y_run[13,0]
    i_l_R06R07_D = struct[0].y_run[14,0]
    i_l_R06R07_Q = struct[0].y_run[15,0]
    i_l_R07R08_D = struct[0].y_run[16,0]
    i_l_R07R08_Q = struct[0].y_run[17,0]
    i_l_R08R09_D = struct[0].y_run[18,0]
    i_l_R08R09_Q = struct[0].y_run[19,0]
    i_l_R09R10_D = struct[0].y_run[20,0]
    i_l_R09R10_Q = struct[0].y_run[21,0]
    i_l_R09R17_D = struct[0].y_run[22,0]
    i_l_R09R17_Q = struct[0].y_run[23,0]
    i_l_R11R03_D = struct[0].y_run[24,0]
    i_l_R11R03_Q = struct[0].y_run[25,0]
    i_l_R12R13_D = struct[0].y_run[26,0]
    i_l_R12R13_Q = struct[0].y_run[27,0]
    i_l_R13R14_D = struct[0].y_run[28,0]
    i_l_R13R14_Q = struct[0].y_run[29,0]
    i_l_R14R15_D = struct[0].y_run[30,0]
    i_l_R14R15_Q = struct[0].y_run[31,0]
    i_l_R16R06_D = struct[0].y_run[32,0]
    i_l_R16R06_Q = struct[0].y_run[33,0]
    i_l_R18R10_D = struct[0].y_run[34,0]
    i_l_R18R10_Q = struct[0].y_run[35,0]
    i_R00_D = struct[0].y_run[36,0]
    i_R00_Q = struct[0].y_run[37,0]
    v_R01_D = struct[0].y_run[38,0]
    v_R01_Q = struct[0].y_run[39,0]
    v_R02_D = struct[0].y_run[40,0]
    v_R02_Q = struct[0].y_run[41,0]
    v_R03_D = struct[0].y_run[42,0]
    v_R03_Q = struct[0].y_run[43,0]
    v_R04_D = struct[0].y_run[44,0]
    v_R04_Q = struct[0].y_run[45,0]
    v_R05_D = struct[0].y_run[46,0]
    v_R05_Q = struct[0].y_run[47,0]
    v_R06_D = struct[0].y_run[48,0]
    v_R06_Q = struct[0].y_run[49,0]
    v_R07_D = struct[0].y_run[50,0]
    v_R07_Q = struct[0].y_run[51,0]
    v_R08_D = struct[0].y_run[52,0]
    v_R08_Q = struct[0].y_run[53,0]
    v_R09_D = struct[0].y_run[54,0]
    v_R09_Q = struct[0].y_run[55,0]
    v_R10_D = struct[0].y_run[56,0]
    v_R10_Q = struct[0].y_run[57,0]
    v_R11_D = struct[0].y_run[58,0]
    v_R11_Q = struct[0].y_run[59,0]
    v_R12_D = struct[0].y_run[60,0]
    v_R12_Q = struct[0].y_run[61,0]
    v_R13_D = struct[0].y_run[62,0]
    v_R13_Q = struct[0].y_run[63,0]
    v_R14_D = struct[0].y_run[64,0]
    v_R14_Q = struct[0].y_run[65,0]
    v_R15_D = struct[0].y_run[66,0]
    v_R15_Q = struct[0].y_run[67,0]
    v_R16_D = struct[0].y_run[68,0]
    v_R16_Q = struct[0].y_run[69,0]
    v_R17_D = struct[0].y_run[70,0]
    v_R17_Q = struct[0].y_run[71,0]
    v_R18_D = struct[0].y_run[72,0]
    v_R18_Q = struct[0].y_run[73,0]
    i_R01_d_ref = struct[0].y_run[74,0]
    i_R01_q_ref = struct[0].y_run[75,0]
    i_R11_d_ref = struct[0].y_run[76,0]
    i_R11_q_ref = struct[0].y_run[77,0]
    i_R15_d_ref = struct[0].y_run[78,0]
    i_R15_q_ref = struct[0].y_run[79,0]
    i_R16_d_ref = struct[0].y_run[80,0]
    i_R16_q_ref = struct[0].y_run[81,0]
    i_R17_d_ref = struct[0].y_run[82,0]
    i_R17_q_ref = struct[0].y_run[83,0]
    i_R18_d_ref = struct[0].y_run[84,0]
    i_R18_q_ref = struct[0].y_run[85,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = L_R00R01*i_l_R00R01_Q*omega - R_R00R01*i_l_R00R01_D + v_R00_D - v_R01_D
        struct[0].g[1,0] = -L_R00R01*i_l_R00R01_D*omega - R_R00R01*i_l_R00R01_Q + v_R00_Q - v_R01_Q
        struct[0].g[2,0] = L_R02R01*i_l_R02R01_Q*omega - R_R02R01*i_l_R02R01_D - v_R01_D + v_R02_D
        struct[0].g[3,0] = -L_R02R01*i_l_R02R01_D*omega - R_R02R01*i_l_R02R01_Q - v_R01_Q + v_R02_Q
        struct[0].g[4,0] = L_R02R03*i_l_R02R03_Q*omega - R_R02R03*i_l_R02R03_D + v_R02_D - v_R03_D
        struct[0].g[5,0] = -L_R02R03*i_l_R02R03_D*omega - R_R02R03*i_l_R02R03_Q + v_R02_Q - v_R03_Q
        struct[0].g[6,0] = L_R03R04*i_l_R03R04_Q*omega - R_R03R04*i_l_R03R04_D + v_R03_D - v_R04_D
        struct[0].g[7,0] = -L_R03R04*i_l_R03R04_D*omega - R_R03R04*i_l_R03R04_Q + v_R03_Q - v_R04_Q
        struct[0].g[8,0] = L_R04R05*i_l_R04R05_Q*omega - R_R04R05*i_l_R04R05_D + v_R04_D - v_R05_D
        struct[0].g[9,0] = -L_R04R05*i_l_R04R05_D*omega - R_R04R05*i_l_R04R05_Q + v_R04_Q - v_R05_Q
        struct[0].g[10,0] = L_R04R12*i_l_R04R12_Q*omega - R_R04R12*i_l_R04R12_D + v_R04_D - v_R12_D
        struct[0].g[11,0] = -L_R04R12*i_l_R04R12_D*omega - R_R04R12*i_l_R04R12_Q + v_R04_Q - v_R12_Q
        struct[0].g[12,0] = L_R05R06*i_l_R05R06_Q*omega - R_R05R06*i_l_R05R06_D + v_R05_D - v_R06_D
        struct[0].g[13,0] = -L_R05R06*i_l_R05R06_D*omega - R_R05R06*i_l_R05R06_Q + v_R05_Q - v_R06_Q
        struct[0].g[14,0] = L_R06R07*i_l_R06R07_Q*omega - R_R06R07*i_l_R06R07_D + v_R06_D - v_R07_D
        struct[0].g[15,0] = -L_R06R07*i_l_R06R07_D*omega - R_R06R07*i_l_R06R07_Q + v_R06_Q - v_R07_Q
        struct[0].g[16,0] = L_R07R08*i_l_R07R08_Q*omega - R_R07R08*i_l_R07R08_D + v_R07_D - v_R08_D
        struct[0].g[17,0] = -L_R07R08*i_l_R07R08_D*omega - R_R07R08*i_l_R07R08_Q + v_R07_Q - v_R08_Q
        struct[0].g[18,0] = L_R08R09*i_l_R08R09_Q*omega - R_R08R09*i_l_R08R09_D + v_R08_D - v_R09_D
        struct[0].g[19,0] = -L_R08R09*i_l_R08R09_D*omega - R_R08R09*i_l_R08R09_Q + v_R08_Q - v_R09_Q
        struct[0].g[20,0] = L_R09R10*i_l_R09R10_Q*omega - R_R09R10*i_l_R09R10_D + v_R09_D - v_R10_D
        struct[0].g[21,0] = -L_R09R10*i_l_R09R10_D*omega - R_R09R10*i_l_R09R10_Q + v_R09_Q - v_R10_Q
        struct[0].g[22,0] = L_R09R17*i_l_R09R17_Q*omega - R_R09R17*i_l_R09R17_D + v_R09_D - v_R17_D
        struct[0].g[23,0] = -L_R09R17*i_l_R09R17_D*omega - R_R09R17*i_l_R09R17_Q + v_R09_Q - v_R17_Q
        struct[0].g[24,0] = L_R11R03*i_l_R11R03_Q*omega - R_R11R03*i_l_R11R03_D - v_R03_D + v_R11_D
        struct[0].g[25,0] = -L_R11R03*i_l_R11R03_D*omega - R_R11R03*i_l_R11R03_Q - v_R03_Q + v_R11_Q
        struct[0].g[26,0] = L_R12R13*i_l_R12R13_Q*omega - R_R12R13*i_l_R12R13_D + v_R12_D - v_R13_D
        struct[0].g[27,0] = -L_R12R13*i_l_R12R13_D*omega - R_R12R13*i_l_R12R13_Q + v_R12_Q - v_R13_Q
        struct[0].g[28,0] = L_R13R14*i_l_R13R14_Q*omega - R_R13R14*i_l_R13R14_D + v_R13_D - v_R14_D
        struct[0].g[29,0] = -L_R13R14*i_l_R13R14_D*omega - R_R13R14*i_l_R13R14_Q + v_R13_Q - v_R14_Q
        struct[0].g[30,0] = L_R14R15*i_l_R14R15_Q*omega - R_R14R15*i_l_R14R15_D + v_R14_D - v_R15_D
        struct[0].g[31,0] = -L_R14R15*i_l_R14R15_D*omega - R_R14R15*i_l_R14R15_Q + v_R14_Q - v_R15_Q
        struct[0].g[32,0] = L_R16R06*i_l_R16R06_Q*omega - R_R16R06*i_l_R16R06_D - v_R06_D + v_R16_D
        struct[0].g[33,0] = -L_R16R06*i_l_R16R06_D*omega - R_R16R06*i_l_R16R06_Q - v_R06_Q + v_R16_Q
        struct[0].g[34,0] = L_R18R10*i_l_R18R10_Q*omega - R_R18R10*i_l_R18R10_D - v_R10_D + v_R18_D
        struct[0].g[35,0] = -L_R18R10*i_l_R18R10_D*omega - R_R18R10*i_l_R18R10_Q - v_R10_Q + v_R18_Q
        struct[0].g[36,0] = C_R00R01*omega*v_R00_Q/2 + i_R00_D - i_l_R00R01_D
        struct[0].g[37,0] = -C_R00R01*omega*v_R00_D/2 + i_R00_Q - i_l_R00R01_Q
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[56,0] = i_R10_D + i_l_R09R10_D + i_l_R18R10_D + omega*v_R10_Q*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[57,0] = i_R10_Q + i_l_R09R10_Q + i_l_R18R10_Q - omega*v_R10_D*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[64,0] = i_R14_D + i_l_R13R14_D - i_l_R14R15_D + omega*v_R14_Q*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[65,0] = i_R14_Q + i_l_R13R14_Q - i_l_R14R15_Q - omega*v_R14_D*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (i_R01_D**2 + i_R01_Q**2)**0.5
    

    if mode == 10:

        struct[0].Fx[0,0] = -100.000000000000
        struct[0].Fx[1,1] = -100.000000000000
        struct[0].Fx[2,2] = -100.000000000000
        struct[0].Fx[3,3] = -100.000000000000
        struct[0].Fx[4,4] = -100.000000000000
        struct[0].Fx[5,5] = -100.000000000000
        struct[0].Fx[6,6] = -100.000000000000
        struct[0].Fx[7,7] = -100.000000000000
        struct[0].Fx[8,8] = -100.000000000000
        struct[0].Fx[9,9] = -100.000000000000
        struct[0].Fx[10,10] = -100.000000000000
        struct[0].Fx[11,11] = -100.000000000000

    if mode == 11:

        struct[0].Fy[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)]))
        struct[0].Fy[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)]))
        struct[0].Fy[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)]))
        struct[0].Fy[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)]))
        struct[0].Fy[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)]))
        struct[0].Fy[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)]))
        struct[0].Fy[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)]))
        struct[0].Fy[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)]))
        struct[0].Fy[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)]))
        struct[0].Fy[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)]))
        struct[0].Fy[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)]))
        struct[0].Fy[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)]))

        struct[0].Gy[0,0] = -R_R00R01
        struct[0].Gy[0,1] = L_R00R01*omega
        struct[0].Gy[0,38] = -1
        struct[0].Gy[1,0] = -L_R00R01*omega
        struct[0].Gy[1,1] = -R_R00R01
        struct[0].Gy[1,39] = -1
        struct[0].Gy[2,2] = -R_R02R01
        struct[0].Gy[2,3] = L_R02R01*omega
        struct[0].Gy[2,38] = -1
        struct[0].Gy[2,40] = 1
        struct[0].Gy[3,2] = -L_R02R01*omega
        struct[0].Gy[3,3] = -R_R02R01
        struct[0].Gy[3,39] = -1
        struct[0].Gy[3,41] = 1
        struct[0].Gy[4,4] = -R_R02R03
        struct[0].Gy[4,5] = L_R02R03*omega
        struct[0].Gy[4,40] = 1
        struct[0].Gy[4,42] = -1
        struct[0].Gy[5,4] = -L_R02R03*omega
        struct[0].Gy[5,5] = -R_R02R03
        struct[0].Gy[5,41] = 1
        struct[0].Gy[5,43] = -1
        struct[0].Gy[6,6] = -R_R03R04
        struct[0].Gy[6,7] = L_R03R04*omega
        struct[0].Gy[6,42] = 1
        struct[0].Gy[6,44] = -1
        struct[0].Gy[7,6] = -L_R03R04*omega
        struct[0].Gy[7,7] = -R_R03R04
        struct[0].Gy[7,43] = 1
        struct[0].Gy[7,45] = -1
        struct[0].Gy[8,8] = -R_R04R05
        struct[0].Gy[8,9] = L_R04R05*omega
        struct[0].Gy[8,44] = 1
        struct[0].Gy[8,46] = -1
        struct[0].Gy[9,8] = -L_R04R05*omega
        struct[0].Gy[9,9] = -R_R04R05
        struct[0].Gy[9,45] = 1
        struct[0].Gy[9,47] = -1
        struct[0].Gy[10,10] = -R_R04R12
        struct[0].Gy[10,11] = L_R04R12*omega
        struct[0].Gy[10,44] = 1
        struct[0].Gy[10,60] = -1
        struct[0].Gy[11,10] = -L_R04R12*omega
        struct[0].Gy[11,11] = -R_R04R12
        struct[0].Gy[11,45] = 1
        struct[0].Gy[11,61] = -1
        struct[0].Gy[12,12] = -R_R05R06
        struct[0].Gy[12,13] = L_R05R06*omega
        struct[0].Gy[12,46] = 1
        struct[0].Gy[12,48] = -1
        struct[0].Gy[13,12] = -L_R05R06*omega
        struct[0].Gy[13,13] = -R_R05R06
        struct[0].Gy[13,47] = 1
        struct[0].Gy[13,49] = -1
        struct[0].Gy[14,14] = -R_R06R07
        struct[0].Gy[14,15] = L_R06R07*omega
        struct[0].Gy[14,48] = 1
        struct[0].Gy[14,50] = -1
        struct[0].Gy[15,14] = -L_R06R07*omega
        struct[0].Gy[15,15] = -R_R06R07
        struct[0].Gy[15,49] = 1
        struct[0].Gy[15,51] = -1
        struct[0].Gy[16,16] = -R_R07R08
        struct[0].Gy[16,17] = L_R07R08*omega
        struct[0].Gy[16,50] = 1
        struct[0].Gy[16,52] = -1
        struct[0].Gy[17,16] = -L_R07R08*omega
        struct[0].Gy[17,17] = -R_R07R08
        struct[0].Gy[17,51] = 1
        struct[0].Gy[17,53] = -1
        struct[0].Gy[18,18] = -R_R08R09
        struct[0].Gy[18,19] = L_R08R09*omega
        struct[0].Gy[18,52] = 1
        struct[0].Gy[18,54] = -1
        struct[0].Gy[19,18] = -L_R08R09*omega
        struct[0].Gy[19,19] = -R_R08R09
        struct[0].Gy[19,53] = 1
        struct[0].Gy[19,55] = -1
        struct[0].Gy[20,20] = -R_R09R10
        struct[0].Gy[20,21] = L_R09R10*omega
        struct[0].Gy[20,54] = 1
        struct[0].Gy[20,56] = -1
        struct[0].Gy[21,20] = -L_R09R10*omega
        struct[0].Gy[21,21] = -R_R09R10
        struct[0].Gy[21,55] = 1
        struct[0].Gy[21,57] = -1
        struct[0].Gy[22,22] = -R_R09R17
        struct[0].Gy[22,23] = L_R09R17*omega
        struct[0].Gy[22,54] = 1
        struct[0].Gy[22,70] = -1
        struct[0].Gy[23,22] = -L_R09R17*omega
        struct[0].Gy[23,23] = -R_R09R17
        struct[0].Gy[23,55] = 1
        struct[0].Gy[23,71] = -1
        struct[0].Gy[24,24] = -R_R11R03
        struct[0].Gy[24,25] = L_R11R03*omega
        struct[0].Gy[24,42] = -1
        struct[0].Gy[24,58] = 1
        struct[0].Gy[25,24] = -L_R11R03*omega
        struct[0].Gy[25,25] = -R_R11R03
        struct[0].Gy[25,43] = -1
        struct[0].Gy[25,59] = 1
        struct[0].Gy[26,26] = -R_R12R13
        struct[0].Gy[26,27] = L_R12R13*omega
        struct[0].Gy[26,60] = 1
        struct[0].Gy[26,62] = -1
        struct[0].Gy[27,26] = -L_R12R13*omega
        struct[0].Gy[27,27] = -R_R12R13
        struct[0].Gy[27,61] = 1
        struct[0].Gy[27,63] = -1
        struct[0].Gy[28,28] = -R_R13R14
        struct[0].Gy[28,29] = L_R13R14*omega
        struct[0].Gy[28,62] = 1
        struct[0].Gy[28,64] = -1
        struct[0].Gy[29,28] = -L_R13R14*omega
        struct[0].Gy[29,29] = -R_R13R14
        struct[0].Gy[29,63] = 1
        struct[0].Gy[29,65] = -1
        struct[0].Gy[30,30] = -R_R14R15
        struct[0].Gy[30,31] = L_R14R15*omega
        struct[0].Gy[30,64] = 1
        struct[0].Gy[30,66] = -1
        struct[0].Gy[31,30] = -L_R14R15*omega
        struct[0].Gy[31,31] = -R_R14R15
        struct[0].Gy[31,65] = 1
        struct[0].Gy[31,67] = -1
        struct[0].Gy[32,32] = -R_R16R06
        struct[0].Gy[32,33] = L_R16R06*omega
        struct[0].Gy[32,48] = -1
        struct[0].Gy[32,68] = 1
        struct[0].Gy[33,32] = -L_R16R06*omega
        struct[0].Gy[33,33] = -R_R16R06
        struct[0].Gy[33,49] = -1
        struct[0].Gy[33,69] = 1
        struct[0].Gy[34,34] = -R_R18R10
        struct[0].Gy[34,35] = L_R18R10*omega
        struct[0].Gy[34,56] = -1
        struct[0].Gy[34,72] = 1
        struct[0].Gy[35,34] = -L_R18R10*omega
        struct[0].Gy[35,35] = -R_R18R10
        struct[0].Gy[35,57] = -1
        struct[0].Gy[35,73] = 1
        struct[0].Gy[36,0] = -1
        struct[0].Gy[36,36] = 1
        struct[0].Gy[37,1] = -1
        struct[0].Gy[37,37] = 1
        struct[0].Gy[38,0] = 1
        struct[0].Gy[38,2] = 1
        struct[0].Gy[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[39,1] = 1
        struct[0].Gy[39,3] = 1
        struct[0].Gy[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[40,2] = -1
        struct[0].Gy[40,4] = -1
        struct[0].Gy[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[41,3] = -1
        struct[0].Gy[41,5] = -1
        struct[0].Gy[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[42,4] = 1
        struct[0].Gy[42,6] = -1
        struct[0].Gy[42,24] = 1
        struct[0].Gy[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[43,5] = 1
        struct[0].Gy[43,7] = -1
        struct[0].Gy[43,25] = 1
        struct[0].Gy[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[44,6] = 1
        struct[0].Gy[44,8] = -1
        struct[0].Gy[44,10] = -1
        struct[0].Gy[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[45,7] = 1
        struct[0].Gy[45,9] = -1
        struct[0].Gy[45,11] = -1
        struct[0].Gy[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[46,8] = 1
        struct[0].Gy[46,12] = -1
        struct[0].Gy[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[47,9] = 1
        struct[0].Gy[47,13] = -1
        struct[0].Gy[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[48,12] = 1
        struct[0].Gy[48,14] = -1
        struct[0].Gy[48,32] = 1
        struct[0].Gy[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[49,13] = 1
        struct[0].Gy[49,15] = -1
        struct[0].Gy[49,33] = 1
        struct[0].Gy[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[50,14] = 1
        struct[0].Gy[50,16] = -1
        struct[0].Gy[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[51,15] = 1
        struct[0].Gy[51,17] = -1
        struct[0].Gy[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[52,16] = 1
        struct[0].Gy[52,18] = -1
        struct[0].Gy[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[53,17] = 1
        struct[0].Gy[53,19] = -1
        struct[0].Gy[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[54,18] = 1
        struct[0].Gy[54,20] = -1
        struct[0].Gy[54,22] = -1
        struct[0].Gy[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[55,19] = 1
        struct[0].Gy[55,21] = -1
        struct[0].Gy[55,23] = -1
        struct[0].Gy[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[56,20] = 1
        struct[0].Gy[56,34] = 1
        struct[0].Gy[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[57,21] = 1
        struct[0].Gy[57,35] = 1
        struct[0].Gy[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[58,24] = -1
        struct[0].Gy[58,59] = C_R11R03*omega/2
        struct[0].Gy[59,25] = -1
        struct[0].Gy[59,58] = -C_R11R03*omega/2
        struct[0].Gy[60,10] = 1
        struct[0].Gy[60,26] = -1
        struct[0].Gy[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[61,11] = 1
        struct[0].Gy[61,27] = -1
        struct[0].Gy[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[62,26] = 1
        struct[0].Gy[62,28] = -1
        struct[0].Gy[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[63,27] = 1
        struct[0].Gy[63,29] = -1
        struct[0].Gy[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[64,28] = 1
        struct[0].Gy[64,30] = -1
        struct[0].Gy[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[65,29] = 1
        struct[0].Gy[65,31] = -1
        struct[0].Gy[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[66,30] = 1
        struct[0].Gy[66,67] = C_R14R15*omega/2
        struct[0].Gy[67,31] = 1
        struct[0].Gy[67,66] = -C_R14R15*omega/2
        struct[0].Gy[68,32] = -1
        struct[0].Gy[68,69] = C_R16R06*omega/2
        struct[0].Gy[69,33] = -1
        struct[0].Gy[69,68] = -C_R16R06*omega/2
        struct[0].Gy[70,22] = 1
        struct[0].Gy[70,71] = C_R09R17*omega/2
        struct[0].Gy[71,23] = 1
        struct[0].Gy[71,70] = -C_R09R17*omega/2
        struct[0].Gy[72,34] = -1
        struct[0].Gy[72,73] = C_R18R10*omega/2
        struct[0].Gy[73,35] = -1
        struct[0].Gy[73,72] = -C_R18R10*omega/2
        struct[0].Gy[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[74,74] = -1
        struct[0].Gy[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,75] = -1
        struct[0].Gy[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[76,76] = -1
        struct[0].Gy[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,77] = -1
        struct[0].Gy[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[78,78] = -1
        struct[0].Gy[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,79] = -1
        struct[0].Gy[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[80,80] = -1
        struct[0].Gy[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,81] = -1
        struct[0].Gy[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[82,82] = -1
        struct[0].Gy[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,83] = -1
        struct[0].Gy[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[84,84] = -1
        struct[0].Gy[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,85] = -1

        struct[0].Gu[0,0] = 1
        struct[0].Gu[1,1] = 1
        struct[0].Gu[36,1] = C_R00R01*omega/2
        struct[0].Gu[37,0] = -C_R00R01*omega/2
        struct[0].Gu[74,4] = -0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[74,5] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,4] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,5] = 0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[76,8] = -0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[76,9] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,8] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,9] = 0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[78,12] = -0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[78,13] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,12] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,13] = 0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[80,16] = -0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[80,17] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,16] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,17] = 0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[82,20] = -0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[82,21] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,20] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,21] = 0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[84,24] = -0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[84,25] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,24] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,25] = 0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))





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


