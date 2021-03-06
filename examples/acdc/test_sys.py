import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 


class test_sys_class: 

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
        self.N_y = 107 
        self.N_z = 27 
        self.N_store = 10000 
        self.params_list = ['a_R1', 'b_R1', 'c_R1', 'a_R10', 'b_R10', 'c_R10', 'coef_a_R10', 'coef_b_R10', 'coef_c_R10'] 
        self.params_values_list  = [2.92, 0.45, 0.027, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333] 
        self.inputs_ini_list = ['v_R0_a_r', 'v_R0_a_i', 'v_R0_b_r', 'v_R0_b_i', 'v_R0_c_r', 'v_R0_c_i', 'v_D1_a_r', 'v_D1_a_i', 'v_D1_b_r', 'v_D1_b_i', 'v_D1_c_r', 'v_D1_c_i', 'i_R1_n_r', 'i_R1_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R18_b_r', 'i_R18_b_i', 'i_R18_c_r', 'i_R18_c_i', 'i_D1_n_r', 'i_D1_n_i', 'i_D10_a_i', 'i_D10_b_r', 'i_D10_b_i', 'i_D10_c_r', 'i_D10_c_i', 'i_D10_n_i', 'i_D18_b_r', 'i_D18_b_i', 'i_D18_c_r', 'i_D18_c_i', 'p_R1_a', 'q_R1_a', 'p_R1_b', 'q_R1_b', 'p_R1_c', 'q_R1_c', 'p_R18_1', 'q_R18_1', 'p_D18_1', 'q_D18_1', 'v_dc_D1', 'q_R1', 'p_R10', 'q_R10', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, 800.0, 0.0, 0.0, -0.0, -0.0, 0.0, -1.1964607142191, -4.231459684193851, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.333338834665, -20816.659986990373, -63333.33333333188, -20816.659994659458, -63333.33333333703, -20816.659994660364, -44649.99997286533, -14675.745191641749, -1000.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
        self.inputs_run_list = ['v_R0_a_r', 'v_R0_a_i', 'v_R0_b_r', 'v_R0_b_i', 'v_R0_c_r', 'v_R0_c_i', 'v_D1_a_r', 'v_D1_a_i', 'v_D1_b_r', 'v_D1_b_i', 'v_D1_c_r', 'v_D1_c_i', 'i_R1_n_r', 'i_R1_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R18_b_r', 'i_R18_b_i', 'i_R18_c_r', 'i_R18_c_i', 'i_D1_n_r', 'i_D1_n_i', 'i_D10_a_i', 'i_D10_b_r', 'i_D10_b_i', 'i_D10_c_r', 'i_D10_c_i', 'i_D10_n_i', 'i_D18_b_r', 'i_D18_b_i', 'i_D18_c_r', 'i_D18_c_i', 'p_R1_a', 'q_R1_a', 'p_R1_b', 'q_R1_b', 'p_R1_c', 'q_R1_c', 'p_R18_1', 'q_R18_1', 'p_D18_1', 'q_D18_1', 'v_dc_D1', 'q_R1', 'p_R10', 'q_R10', 'u_dummy'] 
        self.inputs_run_values_list = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, 800.0, 0.0, 0.0, -0.0, -0.0, 0.0, -1.1964607142191, -4.231459684193851, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.333338834665, -20816.659986990373, -63333.33333333188, -20816.659994659458, -63333.33333333703, -20816.659994660364, -44649.99997286533, -14675.745191641749, -1000.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
        self.outputs_list = ['v_R0_a_m', 'v_R0_b_m', 'v_R0_c_m', 'v_D1_a_m', 'v_D1_b_m', 'v_D1_c_m', 'v_R1_a_m', 'v_R1_b_m', 'v_R1_c_m', 'v_R1_n_m', 'v_R18_a_m', 'v_R18_n_m', 'v_D18_a_m', 'v_D18_n_m', 'v_R10_a_m', 'v_R10_b_m', 'v_R10_c_m', 'v_R10_n_m', 'v_R18_b_m', 'v_R18_c_m', 'v_D1_n_m', 'v_D10_a_m', 'v_D10_b_m', 'v_D10_c_m', 'v_D10_n_m', 'v_D18_b_m', 'v_D18_c_m'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_R1_a_r', 'v_R1_a_i', 'v_R1_b_r', 'v_R1_b_i', 'v_R1_c_r', 'v_R1_c_i', 'v_R1_n_r', 'v_R1_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_n_r', 'v_R18_n_i', 'v_D18_a_r', 'v_D18_a_i', 'v_D18_n_r', 'v_D18_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_D1_n_r', 'v_D1_n_i', 'v_D10_a_r', 'v_D10_a_i', 'v_D10_b_r', 'v_D10_b_i', 'v_D10_c_r', 'v_D10_c_i', 'v_D10_n_r', 'v_D10_n_i', 'v_D18_b_r', 'v_D18_b_i', 'v_D18_c_r', 'v_D18_c_i', 'i_t_R0_R1_a_r', 'i_t_R0_R1_a_i', 'i_t_R0_R1_b_r', 'i_t_R0_R1_b_i', 'i_t_R0_R1_c_r', 'i_t_R0_R1_c_i', 'i_l_R1_R10_a_r', 'i_l_R1_R10_a_i', 'i_l_R1_R10_b_r', 'i_l_R1_R10_b_i', 'i_l_R1_R10_c_r', 'i_l_R1_R10_c_i', 'i_l_R1_R10_n_r', 'i_l_R1_R10_n_i', 'i_l_D1_D10_a_r', 'i_l_D1_D10_a_i', 'i_l_D1_D10_b_r', 'i_l_D1_D10_b_i', 'i_l_D1_D10_c_r', 'i_l_D1_D10_c_i', 'i_l_D1_D10_n_r', 'i_l_D1_D10_n_i', 'i_l_D10_D18_a_r', 'i_l_D10_D18_a_i', 'i_l_D10_D18_b_r', 'i_l_D10_D18_b_i', 'i_l_D10_D18_c_r', 'i_l_D10_D18_c_i', 'i_l_D10_D18_n_r', 'i_l_D10_D18_n_i', 'i_load_R1_a_r', 'i_load_R1_a_i', 'i_load_R1_b_r', 'i_load_R1_b_i', 'i_load_R1_c_r', 'i_load_R1_c_i', 'i_load_R1_n_r', 'i_load_R1_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_D18_a_r', 'i_load_D18_a_i', 'i_load_D18_n_r', 'i_load_D18_n_i', 'i_vsc_R1_a_r', 'i_vsc_R1_a_i', 'i_vsc_R1_b_r', 'i_vsc_R1_b_i', 'i_vsc_R1_c_r', 'i_vsc_R1_c_i', 'p_R1', 'p_D1', 'p_loss_R1', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_D10_a_r', 'i_vsc_D10_n_r', 'p_D10', 'p_loss_R10'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_R1_a_r', 'v_R1_a_i', 'v_R1_b_r', 'v_R1_b_i', 'v_R1_c_r', 'v_R1_c_i', 'v_R1_n_r', 'v_R1_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_n_r', 'v_R18_n_i', 'v_D18_a_r', 'v_D18_a_i', 'v_D18_n_r', 'v_D18_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_D1_n_r', 'v_D1_n_i', 'v_D10_a_r', 'v_D10_a_i', 'v_D10_b_r', 'v_D10_b_i', 'v_D10_c_r', 'v_D10_c_i', 'v_D10_n_r', 'v_D10_n_i', 'v_D18_b_r', 'v_D18_b_i', 'v_D18_c_r', 'v_D18_c_i', 'i_t_R0_R1_a_r', 'i_t_R0_R1_a_i', 'i_t_R0_R1_b_r', 'i_t_R0_R1_b_i', 'i_t_R0_R1_c_r', 'i_t_R0_R1_c_i', 'i_l_R1_R10_a_r', 'i_l_R1_R10_a_i', 'i_l_R1_R10_b_r', 'i_l_R1_R10_b_i', 'i_l_R1_R10_c_r', 'i_l_R1_R10_c_i', 'i_l_R1_R10_n_r', 'i_l_R1_R10_n_i', 'i_l_D1_D10_a_r', 'i_l_D1_D10_a_i', 'i_l_D1_D10_b_r', 'i_l_D1_D10_b_i', 'i_l_D1_D10_c_r', 'i_l_D1_D10_c_i', 'i_l_D1_D10_n_r', 'i_l_D1_D10_n_i', 'i_l_D10_D18_a_r', 'i_l_D10_D18_a_i', 'i_l_D10_D18_b_r', 'i_l_D10_D18_b_i', 'i_l_D10_D18_c_r', 'i_l_D10_D18_c_i', 'i_l_D10_D18_n_r', 'i_l_D10_D18_n_i', 'i_load_R1_a_r', 'i_load_R1_a_i', 'i_load_R1_b_r', 'i_load_R1_b_i', 'i_load_R1_c_r', 'i_load_R1_c_i', 'i_load_R1_n_r', 'i_load_R1_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_D18_a_r', 'i_load_D18_a_i', 'i_load_D18_n_r', 'i_load_D18_n_i', 'i_vsc_R1_a_r', 'i_vsc_R1_a_i', 'i_vsc_R1_b_r', 'i_vsc_R1_b_i', 'i_vsc_R1_c_r', 'i_vsc_R1_c_i', 'p_R1', 'p_D1', 'p_loss_R1', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_D10_a_r', 'i_vsc_D10_n_r', 'p_D10', 'p_loss_R10'] 
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
    a_R1 = struct[0].a_R1
    b_R1 = struct[0].b_R1
    c_R1 = struct[0].c_R1
    a_R10 = struct[0].a_R10
    b_R10 = struct[0].b_R10
    c_R10 = struct[0].c_R10
    coef_a_R10 = struct[0].coef_a_R10
    coef_b_R10 = struct[0].coef_b_R10
    coef_c_R10 = struct[0].coef_c_R10
    
    # Inputs:
    v_R0_a_r = struct[0].v_R0_a_r
    v_R0_a_i = struct[0].v_R0_a_i
    v_R0_b_r = struct[0].v_R0_b_r
    v_R0_b_i = struct[0].v_R0_b_i
    v_R0_c_r = struct[0].v_R0_c_r
    v_R0_c_i = struct[0].v_R0_c_i
    v_D1_a_r = struct[0].v_D1_a_r
    v_D1_a_i = struct[0].v_D1_a_i
    v_D1_b_r = struct[0].v_D1_b_r
    v_D1_b_i = struct[0].v_D1_b_i
    v_D1_c_r = struct[0].v_D1_c_r
    v_D1_c_i = struct[0].v_D1_c_i
    i_R1_n_r = struct[0].i_R1_n_r
    i_R1_n_i = struct[0].i_R1_n_i
    i_R10_a_r = struct[0].i_R10_a_r
    i_R10_a_i = struct[0].i_R10_a_i
    i_R10_b_r = struct[0].i_R10_b_r
    i_R10_b_i = struct[0].i_R10_b_i
    i_R10_c_r = struct[0].i_R10_c_r
    i_R10_c_i = struct[0].i_R10_c_i
    i_R10_n_r = struct[0].i_R10_n_r
    i_R10_n_i = struct[0].i_R10_n_i
    i_R18_b_r = struct[0].i_R18_b_r
    i_R18_b_i = struct[0].i_R18_b_i
    i_R18_c_r = struct[0].i_R18_c_r
    i_R18_c_i = struct[0].i_R18_c_i
    i_D1_n_r = struct[0].i_D1_n_r
    i_D1_n_i = struct[0].i_D1_n_i
    i_D10_a_i = struct[0].i_D10_a_i
    i_D10_b_r = struct[0].i_D10_b_r
    i_D10_b_i = struct[0].i_D10_b_i
    i_D10_c_r = struct[0].i_D10_c_r
    i_D10_c_i = struct[0].i_D10_c_i
    i_D10_n_i = struct[0].i_D10_n_i
    i_D18_b_r = struct[0].i_D18_b_r
    i_D18_b_i = struct[0].i_D18_b_i
    i_D18_c_r = struct[0].i_D18_c_r
    i_D18_c_i = struct[0].i_D18_c_i
    p_R1_a = struct[0].p_R1_a
    q_R1_a = struct[0].q_R1_a
    p_R1_b = struct[0].p_R1_b
    q_R1_b = struct[0].q_R1_b
    p_R1_c = struct[0].p_R1_c
    q_R1_c = struct[0].q_R1_c
    p_R18_1 = struct[0].p_R18_1
    q_R18_1 = struct[0].q_R18_1
    p_D18_1 = struct[0].p_D18_1
    q_D18_1 = struct[0].q_D18_1
    v_dc_D1 = struct[0].v_dc_D1
    q_R1 = struct[0].q_R1
    p_R10 = struct[0].p_R10
    q_R10 = struct[0].q_R10
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_R1_a_r = struct[0].y_ini[0,0]
    v_R1_a_i = struct[0].y_ini[1,0]
    v_R1_b_r = struct[0].y_ini[2,0]
    v_R1_b_i = struct[0].y_ini[3,0]
    v_R1_c_r = struct[0].y_ini[4,0]
    v_R1_c_i = struct[0].y_ini[5,0]
    v_R1_n_r = struct[0].y_ini[6,0]
    v_R1_n_i = struct[0].y_ini[7,0]
    v_R18_a_r = struct[0].y_ini[8,0]
    v_R18_a_i = struct[0].y_ini[9,0]
    v_R18_n_r = struct[0].y_ini[10,0]
    v_R18_n_i = struct[0].y_ini[11,0]
    v_D18_a_r = struct[0].y_ini[12,0]
    v_D18_a_i = struct[0].y_ini[13,0]
    v_D18_n_r = struct[0].y_ini[14,0]
    v_D18_n_i = struct[0].y_ini[15,0]
    v_R10_a_r = struct[0].y_ini[16,0]
    v_R10_a_i = struct[0].y_ini[17,0]
    v_R10_b_r = struct[0].y_ini[18,0]
    v_R10_b_i = struct[0].y_ini[19,0]
    v_R10_c_r = struct[0].y_ini[20,0]
    v_R10_c_i = struct[0].y_ini[21,0]
    v_R10_n_r = struct[0].y_ini[22,0]
    v_R10_n_i = struct[0].y_ini[23,0]
    v_R18_b_r = struct[0].y_ini[24,0]
    v_R18_b_i = struct[0].y_ini[25,0]
    v_R18_c_r = struct[0].y_ini[26,0]
    v_R18_c_i = struct[0].y_ini[27,0]
    v_D1_n_r = struct[0].y_ini[28,0]
    v_D1_n_i = struct[0].y_ini[29,0]
    v_D10_a_r = struct[0].y_ini[30,0]
    v_D10_a_i = struct[0].y_ini[31,0]
    v_D10_b_r = struct[0].y_ini[32,0]
    v_D10_b_i = struct[0].y_ini[33,0]
    v_D10_c_r = struct[0].y_ini[34,0]
    v_D10_c_i = struct[0].y_ini[35,0]
    v_D10_n_r = struct[0].y_ini[36,0]
    v_D10_n_i = struct[0].y_ini[37,0]
    v_D18_b_r = struct[0].y_ini[38,0]
    v_D18_b_i = struct[0].y_ini[39,0]
    v_D18_c_r = struct[0].y_ini[40,0]
    v_D18_c_i = struct[0].y_ini[41,0]
    i_t_R0_R1_a_r = struct[0].y_ini[42,0]
    i_t_R0_R1_a_i = struct[0].y_ini[43,0]
    i_t_R0_R1_b_r = struct[0].y_ini[44,0]
    i_t_R0_R1_b_i = struct[0].y_ini[45,0]
    i_t_R0_R1_c_r = struct[0].y_ini[46,0]
    i_t_R0_R1_c_i = struct[0].y_ini[47,0]
    i_l_R1_R10_a_r = struct[0].y_ini[48,0]
    i_l_R1_R10_a_i = struct[0].y_ini[49,0]
    i_l_R1_R10_b_r = struct[0].y_ini[50,0]
    i_l_R1_R10_b_i = struct[0].y_ini[51,0]
    i_l_R1_R10_c_r = struct[0].y_ini[52,0]
    i_l_R1_R10_c_i = struct[0].y_ini[53,0]
    i_l_R1_R10_n_r = struct[0].y_ini[54,0]
    i_l_R1_R10_n_i = struct[0].y_ini[55,0]
    i_l_D1_D10_a_r = struct[0].y_ini[56,0]
    i_l_D1_D10_a_i = struct[0].y_ini[57,0]
    i_l_D1_D10_b_r = struct[0].y_ini[58,0]
    i_l_D1_D10_b_i = struct[0].y_ini[59,0]
    i_l_D1_D10_c_r = struct[0].y_ini[60,0]
    i_l_D1_D10_c_i = struct[0].y_ini[61,0]
    i_l_D1_D10_n_r = struct[0].y_ini[62,0]
    i_l_D1_D10_n_i = struct[0].y_ini[63,0]
    i_l_D10_D18_a_r = struct[0].y_ini[64,0]
    i_l_D10_D18_a_i = struct[0].y_ini[65,0]
    i_l_D10_D18_b_r = struct[0].y_ini[66,0]
    i_l_D10_D18_b_i = struct[0].y_ini[67,0]
    i_l_D10_D18_c_r = struct[0].y_ini[68,0]
    i_l_D10_D18_c_i = struct[0].y_ini[69,0]
    i_l_D10_D18_n_r = struct[0].y_ini[70,0]
    i_l_D10_D18_n_i = struct[0].y_ini[71,0]
    i_load_R1_a_r = struct[0].y_ini[72,0]
    i_load_R1_a_i = struct[0].y_ini[73,0]
    i_load_R1_b_r = struct[0].y_ini[74,0]
    i_load_R1_b_i = struct[0].y_ini[75,0]
    i_load_R1_c_r = struct[0].y_ini[76,0]
    i_load_R1_c_i = struct[0].y_ini[77,0]
    i_load_R1_n_r = struct[0].y_ini[78,0]
    i_load_R1_n_i = struct[0].y_ini[79,0]
    i_load_R18_a_r = struct[0].y_ini[80,0]
    i_load_R18_a_i = struct[0].y_ini[81,0]
    i_load_R18_n_r = struct[0].y_ini[82,0]
    i_load_R18_n_i = struct[0].y_ini[83,0]
    i_load_D18_a_r = struct[0].y_ini[84,0]
    i_load_D18_a_i = struct[0].y_ini[85,0]
    i_load_D18_n_r = struct[0].y_ini[86,0]
    i_load_D18_n_i = struct[0].y_ini[87,0]
    i_vsc_R1_a_r = struct[0].y_ini[88,0]
    i_vsc_R1_a_i = struct[0].y_ini[89,0]
    i_vsc_R1_b_r = struct[0].y_ini[90,0]
    i_vsc_R1_b_i = struct[0].y_ini[91,0]
    i_vsc_R1_c_r = struct[0].y_ini[92,0]
    i_vsc_R1_c_i = struct[0].y_ini[93,0]
    p_R1 = struct[0].y_ini[94,0]
    p_D1 = struct[0].y_ini[95,0]
    p_loss_R1 = struct[0].y_ini[96,0]
    i_vsc_R10_a_r = struct[0].y_ini[97,0]
    i_vsc_R10_a_i = struct[0].y_ini[98,0]
    i_vsc_R10_b_r = struct[0].y_ini[99,0]
    i_vsc_R10_b_i = struct[0].y_ini[100,0]
    i_vsc_R10_c_r = struct[0].y_ini[101,0]
    i_vsc_R10_c_i = struct[0].y_ini[102,0]
    i_vsc_D10_a_r = struct[0].y_ini[103,0]
    i_vsc_D10_n_r = struct[0].y_ini[104,0]
    p_D10 = struct[0].y_ini[105,0]
    p_loss_R10 = struct[0].y_ini[106,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = i_load_R1_a_r + i_vsc_R1_a_r + 0.849044513514155*v_R0_a_i + 0.212261128378539*v_R0_a_r - 0.849044513514155*v_R0_c_i - 0.212261128378539*v_R0_c_r + 5.40657727682604*v_R10_a_i + 10.557176931318*v_R10_a_r - 1.02713736253513*v_R10_b_i - 3.96392229058202*v_R10_b_r - 2.3284964480954*v_R10_c_i - 2.49575997948692*v_R10_c_r - 1.02713736253513*v_R10_n_i - 3.96392229058202*v_R10_n_r - 78.9359890415319*v_R1_a_i - 28.9395298724945*v_R1_a_r + 1.02713736253513*v_R1_b_i + 3.96392229058202*v_R1_b_r + 2.3284964480954*v_R1_c_i + 2.49575997948692*v_R1_c_r + 74.556549127241*v_R1_n_i + 22.3462752317585*v_R1_n_r
        struct[0].g[1,0] = i_load_R1_a_i + i_vsc_R1_a_i + 0.212261128378539*v_R0_a_i - 0.849044513514155*v_R0_a_r - 0.212261128378539*v_R0_c_i + 0.849044513514155*v_R0_c_r + 10.557176931318*v_R10_a_i - 5.40657727682604*v_R10_a_r - 3.96392229058202*v_R10_b_i + 1.02713736253513*v_R10_b_r - 2.49575997948692*v_R10_c_i + 2.3284964480954*v_R10_c_r - 3.96392229058202*v_R10_n_i + 1.02713736253513*v_R10_n_r - 28.9395298724945*v_R1_a_i + 78.9359890415319*v_R1_a_r + 3.96392229058202*v_R1_b_i - 1.02713736253513*v_R1_b_r + 2.49575997948692*v_R1_c_i - 2.3284964480954*v_R1_c_r + 22.3462752317585*v_R1_n_i - 74.556549127241*v_R1_n_r
        struct[0].g[2,0] = i_load_R1_b_r + i_vsc_R1_b_r - 0.849044513514155*v_R0_a_i - 0.212261128378539*v_R0_a_r + 0.849044513514155*v_R0_b_i + 0.212261128378539*v_R0_b_r - 1.02713736253513*v_R10_a_i - 3.96392229058202*v_R10_a_r + 5.40657727682604*v_R10_b_i + 10.557176931318*v_R10_b_r - 1.02713736253513*v_R10_c_i - 3.96392229058202*v_R10_c_r - 2.3284964480954*v_R10_n_i - 2.49575997948692*v_R10_n_r + 1.02713736253513*v_R1_a_i + 3.96392229058202*v_R1_a_r - 78.9359890415319*v_R1_b_i - 28.9395298724945*v_R1_b_r + 1.02713736253513*v_R1_c_i + 3.96392229058202*v_R1_c_r + 75.8579082128012*v_R1_n_i + 20.8781129206634*v_R1_n_r
        struct[0].g[3,0] = i_load_R1_b_i + i_vsc_R1_b_i - 0.212261128378539*v_R0_a_i + 0.849044513514155*v_R0_a_r + 0.212261128378539*v_R0_b_i - 0.849044513514155*v_R0_b_r - 3.96392229058202*v_R10_a_i + 1.02713736253513*v_R10_a_r + 10.557176931318*v_R10_b_i - 5.40657727682604*v_R10_b_r - 3.96392229058202*v_R10_c_i + 1.02713736253513*v_R10_c_r - 2.49575997948692*v_R10_n_i + 2.3284964480954*v_R10_n_r + 3.96392229058202*v_R1_a_i - 1.02713736253513*v_R1_a_r - 28.9395298724945*v_R1_b_i + 78.9359890415319*v_R1_b_r + 3.96392229058202*v_R1_c_i - 1.02713736253513*v_R1_c_r + 20.8781129206634*v_R1_n_i - 75.8579082128012*v_R1_n_r
        struct[0].g[4,0] = i_load_R1_c_r + i_vsc_R1_c_r - 0.849044513514155*v_R0_b_i - 0.212261128378539*v_R0_b_r + 0.849044513514155*v_R0_c_i + 0.212261128378539*v_R0_c_r - 2.3284964480954*v_R10_a_i - 2.49575997948692*v_R10_a_r - 1.02713736253513*v_R10_b_i - 3.96392229058202*v_R10_b_r + 5.40657727682604*v_R10_c_i + 10.557176931318*v_R10_c_r - 1.02713736253513*v_R10_n_i - 3.96392229058202*v_R10_n_r + 2.3284964480954*v_R1_a_i + 2.49575997948692*v_R1_a_r + 1.02713736253513*v_R1_b_i + 3.96392229058202*v_R1_b_r - 78.9359890415319*v_R1_c_i - 28.9395298724945*v_R1_c_r + 74.556549127241*v_R1_n_i + 22.3462752317585*v_R1_n_r
        struct[0].g[5,0] = i_load_R1_c_i + i_vsc_R1_c_i - 0.212261128378539*v_R0_b_i + 0.849044513514155*v_R0_b_r + 0.212261128378539*v_R0_c_i - 0.849044513514155*v_R0_c_r - 2.49575997948692*v_R10_a_i + 2.3284964480954*v_R10_a_r - 3.96392229058202*v_R10_b_i + 1.02713736253513*v_R10_b_r + 10.557176931318*v_R10_c_i - 5.40657727682604*v_R10_c_r - 3.96392229058202*v_R10_n_i + 1.02713736253513*v_R10_n_r + 2.49575997948692*v_R1_a_i - 2.3284964480954*v_R1_a_r + 3.96392229058202*v_R1_b_i - 1.02713736253513*v_R1_b_r - 28.9395298724945*v_R1_c_i + 78.9359890415319*v_R1_c_r + 22.3462752317585*v_R1_n_i - 74.556549127241*v_R1_n_r
        struct[0].g[30,0] = i_vsc_D10_a_r - 225.682690137666*v_D10_a_r + 157.977883096366*v_D18_a_r + 67.7048070412999*v_D1_a_r
        struct[0].g[31,0] = -225.682690137666*v_D10_a_i + 157.977883096366*v_D18_a_i + 67.7048070412999*v_D1_a_i
        struct[0].g[32,0] = -225.682690137666*v_D10_b_r + 157.977883096366*v_D18_b_r + 67.7048070412999*v_D1_b_r
        struct[0].g[33,0] = -225.682690137666*v_D10_b_i + 157.977883096366*v_D18_b_i + 67.7048070412999*v_D1_b_i
        struct[0].g[34,0] = -225.682690137666*v_D10_c_r + 157.977883096366*v_D18_c_r + 67.7048070412999*v_D1_c_r
        struct[0].g[35,0] = -225.682690137666*v_D10_c_i + 157.977883096366*v_D18_c_i + 67.7048070412999*v_D1_c_i
        struct[0].g[42,0] = -i_t_R0_R1_a_r + 0.0196078431372549*v_R0_a_i + 0.00490196078431373*v_R0_a_r - 0.00980392156862745*v_R0_b_i - 0.00245098039215686*v_R0_b_r - 0.00980392156862745*v_R0_c_i - 0.00245098039215686*v_R0_c_r - 0.849044513514155*v_R1_a_i - 0.212261128378539*v_R1_a_r + 0.849044513514155*v_R1_b_i + 0.212261128378539*v_R1_b_r
        struct[0].g[43,0] = -i_t_R0_R1_a_i + 0.00490196078431373*v_R0_a_i - 0.0196078431372549*v_R0_a_r - 0.00245098039215686*v_R0_b_i + 0.00980392156862745*v_R0_b_r - 0.00245098039215686*v_R0_c_i + 0.00980392156862745*v_R0_c_r - 0.212261128378539*v_R1_a_i + 0.849044513514155*v_R1_a_r + 0.212261128378539*v_R1_b_i - 0.849044513514155*v_R1_b_r
        struct[0].g[44,0] = -i_t_R0_R1_b_r - 0.00980392156862745*v_R0_a_i - 0.00245098039215686*v_R0_a_r + 0.0196078431372549*v_R0_b_i + 0.00490196078431373*v_R0_b_r - 0.00980392156862745*v_R0_c_i - 0.00245098039215686*v_R0_c_r - 0.849044513514155*v_R1_b_i - 0.212261128378539*v_R1_b_r + 0.849044513514155*v_R1_c_i + 0.212261128378539*v_R1_c_r
        struct[0].g[45,0] = -i_t_R0_R1_b_i - 0.00245098039215686*v_R0_a_i + 0.00980392156862745*v_R0_a_r + 0.00490196078431373*v_R0_b_i - 0.0196078431372549*v_R0_b_r - 0.00245098039215686*v_R0_c_i + 0.00980392156862745*v_R0_c_r - 0.212261128378539*v_R1_b_i + 0.849044513514155*v_R1_b_r + 0.212261128378539*v_R1_c_i - 0.849044513514155*v_R1_c_r
        struct[0].g[46,0] = -i_t_R0_R1_c_r - 0.00980392156862745*v_R0_a_i - 0.00245098039215686*v_R0_a_r - 0.00980392156862745*v_R0_b_i - 0.00245098039215686*v_R0_b_r + 0.0196078431372549*v_R0_c_i + 0.00490196078431373*v_R0_c_r + 0.849044513514155*v_R1_a_i + 0.212261128378539*v_R1_a_r - 0.849044513514155*v_R1_c_i - 0.212261128378539*v_R1_c_r
        struct[0].g[47,0] = -i_t_R0_R1_c_i - 0.00245098039215686*v_R0_a_i + 0.00980392156862745*v_R0_a_r - 0.00245098039215686*v_R0_b_i + 0.00980392156862745*v_R0_b_r + 0.00490196078431373*v_R0_c_i - 0.0196078431372549*v_R0_c_r + 0.212261128378539*v_R1_a_i - 0.849044513514155*v_R1_a_r - 0.212261128378539*v_R1_c_i + 0.849044513514155*v_R1_c_r
        struct[0].g[56,0] = -i_l_D1_D10_a_r - 67.7048070412999*v_D10_a_r + 67.7048070412999*v_D1_a_r
        struct[0].g[57,0] = -i_l_D1_D10_a_i - 67.7048070412999*v_D10_a_i + 67.7048070412999*v_D1_a_i
        struct[0].g[58,0] = -i_l_D1_D10_b_r - 67.7048070412999*v_D10_b_r + 67.7048070412999*v_D1_b_r
        struct[0].g[59,0] = -i_l_D1_D10_b_i - 67.7048070412999*v_D10_b_i + 67.7048070412999*v_D1_b_i
        struct[0].g[60,0] = -i_l_D1_D10_c_r - 67.7048070412999*v_D10_c_r + 67.7048070412999*v_D1_c_r
        struct[0].g[61,0] = -i_l_D1_D10_c_i - 67.7048070412999*v_D10_c_i + 67.7048070412999*v_D1_c_i
        struct[0].g[72,0] = i_load_R1_a_i*v_R1_a_i - i_load_R1_a_i*v_R1_n_i + i_load_R1_a_r*v_R1_a_r - i_load_R1_a_r*v_R1_n_r - p_R1_a
        struct[0].g[73,0] = i_load_R1_b_i*v_R1_b_i - i_load_R1_b_i*v_R1_n_i + i_load_R1_b_r*v_R1_b_r - i_load_R1_b_r*v_R1_n_r - p_R1_b
        struct[0].g[74,0] = i_load_R1_c_i*v_R1_c_i - i_load_R1_c_i*v_R1_n_i + i_load_R1_c_r*v_R1_c_r - i_load_R1_c_r*v_R1_n_r - p_R1_c
        struct[0].g[75,0] = -i_load_R1_a_i*v_R1_a_r + i_load_R1_a_i*v_R1_n_r + i_load_R1_a_r*v_R1_a_i - i_load_R1_a_r*v_R1_n_i - q_R1_a
        struct[0].g[76,0] = -i_load_R1_b_i*v_R1_b_r + i_load_R1_b_i*v_R1_n_r + i_load_R1_b_r*v_R1_b_i - i_load_R1_b_r*v_R1_n_i - q_R1_b
        struct[0].g[77,0] = -i_load_R1_c_i*v_R1_c_r + i_load_R1_c_i*v_R1_n_r + i_load_R1_c_r*v_R1_c_i - i_load_R1_c_r*v_R1_n_i - q_R1_c
        struct[0].g[80,0] = 1.0*i_load_R18_a_i*v_R18_a_i - 1.0*i_load_R18_a_i*v_R18_n_i + i_load_R18_a_r*v_R18_a_r - i_load_R18_a_r*v_R18_n_r - p_R18_1
        struct[0].g[81,0] = -1.0*i_load_R18_a_i*v_R18_a_r + 1.0*i_load_R18_a_i*v_R18_n_r + 1.0*i_load_R18_a_r*v_R18_a_i - 1.0*i_load_R18_a_r*v_R18_n_i - q_R18_1
        struct[0].g[84,0] = 1.0*i_load_D18_a_i*v_D18_a_i - 1.0*i_load_D18_a_i*v_D18_n_i + i_load_D18_a_r*v_D18_a_r - i_load_D18_a_r*v_D18_n_r - p_D18_1
        struct[0].g[85,0] = -1.0*i_load_D18_a_i*v_D18_a_r + 1.0*i_load_D18_a_i*v_D18_n_r + 1.0*i_load_D18_a_r*v_D18_a_i - 1.0*i_load_D18_a_r*v_D18_n_i - q_D18_1
        struct[0].g[88,0] = 1.0*i_vsc_R1_a_i*v_R1_a_i - 1.0*i_vsc_R1_a_i*v_R1_n_i + i_vsc_R1_a_r*v_R1_a_r - i_vsc_R1_a_r*v_R1_n_r - p_R1/3
        struct[0].g[89,0] = -1.0*i_vsc_R1_a_i*v_R1_a_r + 1.0*i_vsc_R1_a_i*v_R1_n_r + 1.0*i_vsc_R1_a_r*v_R1_a_i - 1.0*i_vsc_R1_a_r*v_R1_n_i - q_R1/3
        struct[0].g[90,0] = 1.0*i_vsc_R1_b_i*v_R1_b_i - 1.0*i_vsc_R1_b_i*v_R1_n_i + i_vsc_R1_b_r*v_R1_b_r - i_vsc_R1_b_r*v_R1_n_r - p_R1/3
        struct[0].g[91,0] = -1.0*i_vsc_R1_b_i*v_R1_b_r + 1.0*i_vsc_R1_b_i*v_R1_n_r + 1.0*i_vsc_R1_b_r*v_R1_b_i - 1.0*i_vsc_R1_b_r*v_R1_n_i - q_R1/3
        struct[0].g[92,0] = 1.0*i_vsc_R1_c_i*v_R1_c_i - 1.0*i_vsc_R1_c_i*v_R1_n_i + i_vsc_R1_c_r*v_R1_c_r - i_vsc_R1_c_r*v_R1_n_r - p_R1/3
        struct[0].g[93,0] = -1.0*i_vsc_R1_c_i*v_R1_c_r + 1.0*i_vsc_R1_c_i*v_R1_n_r + 1.0*i_vsc_R1_c_r*v_R1_c_i - 1.0*i_vsc_R1_c_r*v_R1_n_i - q_R1/3
        struct[0].g[94,0] = p_D1 + p_R1 + Piecewise(np.array([(-p_loss_R1, p_D1 < 0), (p_loss_R1, True)]))
        struct[0].g[96,0] = -a_R1 - b_R1*sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - c_R1*(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) + p_loss_R1
        struct[0].g[97,0] = -coef_a_R10*p_R10 + 1.0*i_vsc_R10_a_i*v_R10_a_i - 1.0*i_vsc_R10_a_i*v_R10_n_i + i_vsc_R10_a_r*v_R10_a_r - i_vsc_R10_a_r*v_R10_n_r
        struct[0].g[98,0] = -coef_a_R10*q_R10 - 1.0*i_vsc_R10_a_i*v_R10_a_r + 1.0*i_vsc_R10_a_i*v_R10_n_r + 1.0*i_vsc_R10_a_r*v_R10_a_i - 1.0*i_vsc_R10_a_r*v_R10_n_i
        struct[0].g[99,0] = -coef_b_R10*p_R10 + 1.0*i_vsc_R10_b_i*v_R10_b_i - 1.0*i_vsc_R10_b_i*v_R10_n_i + i_vsc_R10_b_r*v_R10_b_r - i_vsc_R10_b_r*v_R10_n_r
        struct[0].g[100,0] = -coef_b_R10*q_R10 - 1.0*i_vsc_R10_b_i*v_R10_b_r + 1.0*i_vsc_R10_b_i*v_R10_n_r + 1.0*i_vsc_R10_b_r*v_R10_b_i - 1.0*i_vsc_R10_b_r*v_R10_n_i
        struct[0].g[101,0] = -coef_c_R10*p_R10 + 1.0*i_vsc_R10_c_i*v_R10_c_i - 1.0*i_vsc_R10_c_i*v_R10_n_i + i_vsc_R10_c_r*v_R10_c_r - i_vsc_R10_c_r*v_R10_n_r
        struct[0].g[102,0] = -coef_c_R10*q_R10 - 1.0*i_vsc_R10_c_i*v_R10_c_r + 1.0*i_vsc_R10_c_i*v_R10_n_r + 1.0*i_vsc_R10_c_r*v_R10_c_i - 1.0*i_vsc_R10_c_r*v_R10_n_i
        struct[0].g[103,0] = i_vsc_D10_a_r + p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].g[104,0] = i_vsc_D10_n_r + p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].g[105,0] = p_D10 - p_R10 - Piecewise(np.array([(-p_loss_R10, p_D10 < 0), (p_loss_R10, True)]))
        struct[0].g[106,0] = -a_R10 - b_R10*sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - c_R10*(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) + p_loss_R10
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_R0_a_i**2 + v_R0_a_r**2)**0.5
        struct[0].h[1,0] = (v_R0_b_i**2 + v_R0_b_r**2)**0.5
        struct[0].h[2,0] = (v_R0_c_i**2 + v_R0_c_r**2)**0.5
        struct[0].h[3,0] = (v_D1_a_i**2 + v_D1_a_r**2)**0.5
        struct[0].h[4,0] = (v_D1_b_i**2 + v_D1_b_r**2)**0.5
        struct[0].h[5,0] = (v_D1_c_i**2 + v_D1_c_r**2)**0.5
        struct[0].h[6,0] = (v_R1_a_i**2 + v_R1_a_r**2)**0.5
        struct[0].h[7,0] = (v_R1_b_i**2 + v_R1_b_r**2)**0.5
        struct[0].h[8,0] = (v_R1_c_i**2 + v_R1_c_r**2)**0.5
        struct[0].h[9,0] = (v_R1_n_i**2 + v_R1_n_r**2)**0.5
        struct[0].h[10,0] = (v_R18_a_i**2 + v_R18_a_r**2)**0.5
        struct[0].h[11,0] = (v_R18_n_i**2 + v_R18_n_r**2)**0.5
        struct[0].h[12,0] = (v_D18_a_i**2 + v_D18_a_r**2)**0.5
        struct[0].h[13,0] = (v_D18_n_i**2 + v_D18_n_r**2)**0.5
        struct[0].h[14,0] = (v_R10_a_i**2 + v_R10_a_r**2)**0.5
        struct[0].h[15,0] = (v_R10_b_i**2 + v_R10_b_r**2)**0.5
        struct[0].h[16,0] = (v_R10_c_i**2 + v_R10_c_r**2)**0.5
        struct[0].h[17,0] = (v_R10_n_i**2 + v_R10_n_r**2)**0.5
        struct[0].h[18,0] = (v_R18_b_i**2 + v_R18_b_r**2)**0.5
        struct[0].h[19,0] = (v_R18_c_i**2 + v_R18_c_r**2)**0.5
        struct[0].h[20,0] = (v_D1_n_i**2 + v_D1_n_r**2)**0.5
        struct[0].h[21,0] = (v_D10_a_i**2 + v_D10_a_r**2)**0.5
        struct[0].h[22,0] = (v_D10_b_i**2 + v_D10_b_r**2)**0.5
        struct[0].h[23,0] = (v_D10_c_i**2 + v_D10_c_r**2)**0.5
        struct[0].h[24,0] = (v_D10_n_i**2 + v_D10_n_r**2)**0.5
        struct[0].h[25,0] = (v_D18_b_i**2 + v_D18_b_r**2)**0.5
        struct[0].h[26,0] = (v_D18_c_i**2 + v_D18_c_r**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:



        struct[0].Gy_ini[72,0] = i_load_R1_a_r
        struct[0].Gy_ini[72,1] = i_load_R1_a_i
        struct[0].Gy_ini[72,6] = -i_load_R1_a_r
        struct[0].Gy_ini[72,7] = -i_load_R1_a_i
        struct[0].Gy_ini[72,72] = v_R1_a_r - v_R1_n_r
        struct[0].Gy_ini[72,73] = v_R1_a_i - v_R1_n_i
        struct[0].Gy_ini[73,2] = i_load_R1_b_r
        struct[0].Gy_ini[73,3] = i_load_R1_b_i
        struct[0].Gy_ini[73,6] = -i_load_R1_b_r
        struct[0].Gy_ini[73,7] = -i_load_R1_b_i
        struct[0].Gy_ini[73,74] = v_R1_b_r - v_R1_n_r
        struct[0].Gy_ini[73,75] = v_R1_b_i - v_R1_n_i
        struct[0].Gy_ini[74,4] = i_load_R1_c_r
        struct[0].Gy_ini[74,5] = i_load_R1_c_i
        struct[0].Gy_ini[74,6] = -i_load_R1_c_r
        struct[0].Gy_ini[74,7] = -i_load_R1_c_i
        struct[0].Gy_ini[74,76] = v_R1_c_r - v_R1_n_r
        struct[0].Gy_ini[74,77] = v_R1_c_i - v_R1_n_i
        struct[0].Gy_ini[75,0] = -i_load_R1_a_i
        struct[0].Gy_ini[75,1] = i_load_R1_a_r
        struct[0].Gy_ini[75,6] = i_load_R1_a_i
        struct[0].Gy_ini[75,7] = -i_load_R1_a_r
        struct[0].Gy_ini[75,72] = v_R1_a_i - v_R1_n_i
        struct[0].Gy_ini[75,73] = -v_R1_a_r + v_R1_n_r
        struct[0].Gy_ini[76,2] = -i_load_R1_b_i
        struct[0].Gy_ini[76,3] = i_load_R1_b_r
        struct[0].Gy_ini[76,6] = i_load_R1_b_i
        struct[0].Gy_ini[76,7] = -i_load_R1_b_r
        struct[0].Gy_ini[76,74] = v_R1_b_i - v_R1_n_i
        struct[0].Gy_ini[76,75] = -v_R1_b_r + v_R1_n_r
        struct[0].Gy_ini[77,4] = -i_load_R1_c_i
        struct[0].Gy_ini[77,5] = i_load_R1_c_r
        struct[0].Gy_ini[77,6] = i_load_R1_c_i
        struct[0].Gy_ini[77,7] = -i_load_R1_c_r
        struct[0].Gy_ini[77,76] = v_R1_c_i - v_R1_n_i
        struct[0].Gy_ini[77,77] = -v_R1_c_r + v_R1_n_r
        struct[0].Gy_ini[80,8] = i_load_R18_a_r
        struct[0].Gy_ini[80,9] = 1.0*i_load_R18_a_i
        struct[0].Gy_ini[80,10] = -i_load_R18_a_r
        struct[0].Gy_ini[80,11] = -1.0*i_load_R18_a_i
        struct[0].Gy_ini[80,80] = v_R18_a_r - v_R18_n_r
        struct[0].Gy_ini[80,81] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy_ini[81,8] = -1.0*i_load_R18_a_i
        struct[0].Gy_ini[81,9] = 1.0*i_load_R18_a_r
        struct[0].Gy_ini[81,10] = 1.0*i_load_R18_a_i
        struct[0].Gy_ini[81,11] = -1.0*i_load_R18_a_r
        struct[0].Gy_ini[81,80] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy_ini[81,81] = -1.0*v_R18_a_r + 1.0*v_R18_n_r
        struct[0].Gy_ini[84,12] = i_load_D18_a_r
        struct[0].Gy_ini[84,13] = 1.0*i_load_D18_a_i
        struct[0].Gy_ini[84,14] = -i_load_D18_a_r
        struct[0].Gy_ini[84,15] = -1.0*i_load_D18_a_i
        struct[0].Gy_ini[84,84] = v_D18_a_r - v_D18_n_r
        struct[0].Gy_ini[84,85] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy_ini[85,12] = -1.0*i_load_D18_a_i
        struct[0].Gy_ini[85,13] = 1.0*i_load_D18_a_r
        struct[0].Gy_ini[85,14] = 1.0*i_load_D18_a_i
        struct[0].Gy_ini[85,15] = -1.0*i_load_D18_a_r
        struct[0].Gy_ini[85,84] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy_ini[85,85] = -1.0*v_D18_a_r + 1.0*v_D18_n_r
        struct[0].Gy_ini[88,0] = i_vsc_R1_a_r
        struct[0].Gy_ini[88,1] = 1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[88,6] = -i_vsc_R1_a_r
        struct[0].Gy_ini[88,7] = -1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[88,88] = v_R1_a_r - v_R1_n_r
        struct[0].Gy_ini[88,89] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[89,0] = -1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[89,1] = 1.0*i_vsc_R1_a_r
        struct[0].Gy_ini[89,6] = 1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[89,7] = -1.0*i_vsc_R1_a_r
        struct[0].Gy_ini[89,88] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[89,89] = -1.0*v_R1_a_r + 1.0*v_R1_n_r
        struct[0].Gy_ini[90,2] = i_vsc_R1_b_r
        struct[0].Gy_ini[90,3] = 1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[90,6] = -i_vsc_R1_b_r
        struct[0].Gy_ini[90,7] = -1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[90,90] = v_R1_b_r - v_R1_n_r
        struct[0].Gy_ini[90,91] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[91,2] = -1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[91,3] = 1.0*i_vsc_R1_b_r
        struct[0].Gy_ini[91,6] = 1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[91,7] = -1.0*i_vsc_R1_b_r
        struct[0].Gy_ini[91,90] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[91,91] = -1.0*v_R1_b_r + 1.0*v_R1_n_r
        struct[0].Gy_ini[92,4] = i_vsc_R1_c_r
        struct[0].Gy_ini[92,5] = 1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[92,6] = -i_vsc_R1_c_r
        struct[0].Gy_ini[92,7] = -1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[92,92] = v_R1_c_r - v_R1_n_r
        struct[0].Gy_ini[92,93] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[93,4] = -1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[93,5] = 1.0*i_vsc_R1_c_r
        struct[0].Gy_ini[93,6] = 1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[93,7] = -1.0*i_vsc_R1_c_r
        struct[0].Gy_ini[93,92] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[93,93] = -1.0*v_R1_c_r + 1.0*v_R1_n_r
        struct[0].Gy_ini[94,96] = Piecewise(np.array([(-1, p_D1 < 0), (1, True)]))
        struct[0].Gy_ini[95,56] = v_D1_a_r
        struct[0].Gy_ini[95,62] = v_D1_n_r
        struct[0].Gy_ini[96,88] = -b_R1*i_vsc_R1_a_r/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_r
        struct[0].Gy_ini[96,89] = -b_R1*i_vsc_R1_a_i/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_i
        struct[0].Gy_ini[97,16] = i_vsc_R10_a_r
        struct[0].Gy_ini[97,17] = 1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[97,22] = -i_vsc_R10_a_r
        struct[0].Gy_ini[97,23] = -1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[97,97] = v_R10_a_r - v_R10_n_r
        struct[0].Gy_ini[97,98] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[98,16] = -1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[98,17] = 1.0*i_vsc_R10_a_r
        struct[0].Gy_ini[98,22] = 1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[98,23] = -1.0*i_vsc_R10_a_r
        struct[0].Gy_ini[98,97] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[98,98] = -1.0*v_R10_a_r + 1.0*v_R10_n_r
        struct[0].Gy_ini[99,18] = i_vsc_R10_b_r
        struct[0].Gy_ini[99,19] = 1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[99,22] = -i_vsc_R10_b_r
        struct[0].Gy_ini[99,23] = -1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[99,99] = v_R10_b_r - v_R10_n_r
        struct[0].Gy_ini[99,100] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[100,18] = -1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[100,19] = 1.0*i_vsc_R10_b_r
        struct[0].Gy_ini[100,22] = 1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[100,23] = -1.0*i_vsc_R10_b_r
        struct[0].Gy_ini[100,99] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[100,100] = -1.0*v_R10_b_r + 1.0*v_R10_n_r
        struct[0].Gy_ini[101,20] = i_vsc_R10_c_r
        struct[0].Gy_ini[101,21] = 1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[101,22] = -i_vsc_R10_c_r
        struct[0].Gy_ini[101,23] = -1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[101,101] = v_R10_c_r - v_R10_n_r
        struct[0].Gy_ini[101,102] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[102,20] = -1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[102,21] = 1.0*i_vsc_R10_c_r
        struct[0].Gy_ini[102,22] = 1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[102,23] = -1.0*i_vsc_R10_c_r
        struct[0].Gy_ini[102,101] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[102,102] = -1.0*v_R10_c_r + 1.0*v_R10_n_r
        struct[0].Gy_ini[103,30] = -p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[103,36] = p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[103,105] = 1/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].Gy_ini[104,30] = p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[104,36] = -p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[104,105] = 1/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].Gy_ini[105,106] = -Piecewise(np.array([(-1, p_D10 < 0), (1, True)]))
        struct[0].Gy_ini[106,97] = -b_R10*i_vsc_R10_a_r/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_r
        struct[0].Gy_ini[106,98] = -b_R10*i_vsc_R10_a_i/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_i



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    a_R1 = struct[0].a_R1
    b_R1 = struct[0].b_R1
    c_R1 = struct[0].c_R1
    a_R10 = struct[0].a_R10
    b_R10 = struct[0].b_R10
    c_R10 = struct[0].c_R10
    coef_a_R10 = struct[0].coef_a_R10
    coef_b_R10 = struct[0].coef_b_R10
    coef_c_R10 = struct[0].coef_c_R10
    
    # Inputs:
    v_R0_a_r = struct[0].v_R0_a_r
    v_R0_a_i = struct[0].v_R0_a_i
    v_R0_b_r = struct[0].v_R0_b_r
    v_R0_b_i = struct[0].v_R0_b_i
    v_R0_c_r = struct[0].v_R0_c_r
    v_R0_c_i = struct[0].v_R0_c_i
    v_D1_a_r = struct[0].v_D1_a_r
    v_D1_a_i = struct[0].v_D1_a_i
    v_D1_b_r = struct[0].v_D1_b_r
    v_D1_b_i = struct[0].v_D1_b_i
    v_D1_c_r = struct[0].v_D1_c_r
    v_D1_c_i = struct[0].v_D1_c_i
    i_R1_n_r = struct[0].i_R1_n_r
    i_R1_n_i = struct[0].i_R1_n_i
    i_R10_a_r = struct[0].i_R10_a_r
    i_R10_a_i = struct[0].i_R10_a_i
    i_R10_b_r = struct[0].i_R10_b_r
    i_R10_b_i = struct[0].i_R10_b_i
    i_R10_c_r = struct[0].i_R10_c_r
    i_R10_c_i = struct[0].i_R10_c_i
    i_R10_n_r = struct[0].i_R10_n_r
    i_R10_n_i = struct[0].i_R10_n_i
    i_R18_b_r = struct[0].i_R18_b_r
    i_R18_b_i = struct[0].i_R18_b_i
    i_R18_c_r = struct[0].i_R18_c_r
    i_R18_c_i = struct[0].i_R18_c_i
    i_D1_n_r = struct[0].i_D1_n_r
    i_D1_n_i = struct[0].i_D1_n_i
    i_D10_a_i = struct[0].i_D10_a_i
    i_D10_b_r = struct[0].i_D10_b_r
    i_D10_b_i = struct[0].i_D10_b_i
    i_D10_c_r = struct[0].i_D10_c_r
    i_D10_c_i = struct[0].i_D10_c_i
    i_D10_n_i = struct[0].i_D10_n_i
    i_D18_b_r = struct[0].i_D18_b_r
    i_D18_b_i = struct[0].i_D18_b_i
    i_D18_c_r = struct[0].i_D18_c_r
    i_D18_c_i = struct[0].i_D18_c_i
    p_R1_a = struct[0].p_R1_a
    q_R1_a = struct[0].q_R1_a
    p_R1_b = struct[0].p_R1_b
    q_R1_b = struct[0].q_R1_b
    p_R1_c = struct[0].p_R1_c
    q_R1_c = struct[0].q_R1_c
    p_R18_1 = struct[0].p_R18_1
    q_R18_1 = struct[0].q_R18_1
    p_D18_1 = struct[0].p_D18_1
    q_D18_1 = struct[0].q_D18_1
    v_dc_D1 = struct[0].v_dc_D1
    q_R1 = struct[0].q_R1
    p_R10 = struct[0].p_R10
    q_R10 = struct[0].q_R10
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_R1_a_r = struct[0].y_run[0,0]
    v_R1_a_i = struct[0].y_run[1,0]
    v_R1_b_r = struct[0].y_run[2,0]
    v_R1_b_i = struct[0].y_run[3,0]
    v_R1_c_r = struct[0].y_run[4,0]
    v_R1_c_i = struct[0].y_run[5,0]
    v_R1_n_r = struct[0].y_run[6,0]
    v_R1_n_i = struct[0].y_run[7,0]
    v_R18_a_r = struct[0].y_run[8,0]
    v_R18_a_i = struct[0].y_run[9,0]
    v_R18_n_r = struct[0].y_run[10,0]
    v_R18_n_i = struct[0].y_run[11,0]
    v_D18_a_r = struct[0].y_run[12,0]
    v_D18_a_i = struct[0].y_run[13,0]
    v_D18_n_r = struct[0].y_run[14,0]
    v_D18_n_i = struct[0].y_run[15,0]
    v_R10_a_r = struct[0].y_run[16,0]
    v_R10_a_i = struct[0].y_run[17,0]
    v_R10_b_r = struct[0].y_run[18,0]
    v_R10_b_i = struct[0].y_run[19,0]
    v_R10_c_r = struct[0].y_run[20,0]
    v_R10_c_i = struct[0].y_run[21,0]
    v_R10_n_r = struct[0].y_run[22,0]
    v_R10_n_i = struct[0].y_run[23,0]
    v_R18_b_r = struct[0].y_run[24,0]
    v_R18_b_i = struct[0].y_run[25,0]
    v_R18_c_r = struct[0].y_run[26,0]
    v_R18_c_i = struct[0].y_run[27,0]
    v_D1_n_r = struct[0].y_run[28,0]
    v_D1_n_i = struct[0].y_run[29,0]
    v_D10_a_r = struct[0].y_run[30,0]
    v_D10_a_i = struct[0].y_run[31,0]
    v_D10_b_r = struct[0].y_run[32,0]
    v_D10_b_i = struct[0].y_run[33,0]
    v_D10_c_r = struct[0].y_run[34,0]
    v_D10_c_i = struct[0].y_run[35,0]
    v_D10_n_r = struct[0].y_run[36,0]
    v_D10_n_i = struct[0].y_run[37,0]
    v_D18_b_r = struct[0].y_run[38,0]
    v_D18_b_i = struct[0].y_run[39,0]
    v_D18_c_r = struct[0].y_run[40,0]
    v_D18_c_i = struct[0].y_run[41,0]
    i_t_R0_R1_a_r = struct[0].y_run[42,0]
    i_t_R0_R1_a_i = struct[0].y_run[43,0]
    i_t_R0_R1_b_r = struct[0].y_run[44,0]
    i_t_R0_R1_b_i = struct[0].y_run[45,0]
    i_t_R0_R1_c_r = struct[0].y_run[46,0]
    i_t_R0_R1_c_i = struct[0].y_run[47,0]
    i_l_R1_R10_a_r = struct[0].y_run[48,0]
    i_l_R1_R10_a_i = struct[0].y_run[49,0]
    i_l_R1_R10_b_r = struct[0].y_run[50,0]
    i_l_R1_R10_b_i = struct[0].y_run[51,0]
    i_l_R1_R10_c_r = struct[0].y_run[52,0]
    i_l_R1_R10_c_i = struct[0].y_run[53,0]
    i_l_R1_R10_n_r = struct[0].y_run[54,0]
    i_l_R1_R10_n_i = struct[0].y_run[55,0]
    i_l_D1_D10_a_r = struct[0].y_run[56,0]
    i_l_D1_D10_a_i = struct[0].y_run[57,0]
    i_l_D1_D10_b_r = struct[0].y_run[58,0]
    i_l_D1_D10_b_i = struct[0].y_run[59,0]
    i_l_D1_D10_c_r = struct[0].y_run[60,0]
    i_l_D1_D10_c_i = struct[0].y_run[61,0]
    i_l_D1_D10_n_r = struct[0].y_run[62,0]
    i_l_D1_D10_n_i = struct[0].y_run[63,0]
    i_l_D10_D18_a_r = struct[0].y_run[64,0]
    i_l_D10_D18_a_i = struct[0].y_run[65,0]
    i_l_D10_D18_b_r = struct[0].y_run[66,0]
    i_l_D10_D18_b_i = struct[0].y_run[67,0]
    i_l_D10_D18_c_r = struct[0].y_run[68,0]
    i_l_D10_D18_c_i = struct[0].y_run[69,0]
    i_l_D10_D18_n_r = struct[0].y_run[70,0]
    i_l_D10_D18_n_i = struct[0].y_run[71,0]
    i_load_R1_a_r = struct[0].y_run[72,0]
    i_load_R1_a_i = struct[0].y_run[73,0]
    i_load_R1_b_r = struct[0].y_run[74,0]
    i_load_R1_b_i = struct[0].y_run[75,0]
    i_load_R1_c_r = struct[0].y_run[76,0]
    i_load_R1_c_i = struct[0].y_run[77,0]
    i_load_R1_n_r = struct[0].y_run[78,0]
    i_load_R1_n_i = struct[0].y_run[79,0]
    i_load_R18_a_r = struct[0].y_run[80,0]
    i_load_R18_a_i = struct[0].y_run[81,0]
    i_load_R18_n_r = struct[0].y_run[82,0]
    i_load_R18_n_i = struct[0].y_run[83,0]
    i_load_D18_a_r = struct[0].y_run[84,0]
    i_load_D18_a_i = struct[0].y_run[85,0]
    i_load_D18_n_r = struct[0].y_run[86,0]
    i_load_D18_n_i = struct[0].y_run[87,0]
    i_vsc_R1_a_r = struct[0].y_run[88,0]
    i_vsc_R1_a_i = struct[0].y_run[89,0]
    i_vsc_R1_b_r = struct[0].y_run[90,0]
    i_vsc_R1_b_i = struct[0].y_run[91,0]
    i_vsc_R1_c_r = struct[0].y_run[92,0]
    i_vsc_R1_c_i = struct[0].y_run[93,0]
    p_R1 = struct[0].y_run[94,0]
    p_D1 = struct[0].y_run[95,0]
    p_loss_R1 = struct[0].y_run[96,0]
    i_vsc_R10_a_r = struct[0].y_run[97,0]
    i_vsc_R10_a_i = struct[0].y_run[98,0]
    i_vsc_R10_b_r = struct[0].y_run[99,0]
    i_vsc_R10_b_i = struct[0].y_run[100,0]
    i_vsc_R10_c_r = struct[0].y_run[101,0]
    i_vsc_R10_c_i = struct[0].y_run[102,0]
    i_vsc_D10_a_r = struct[0].y_run[103,0]
    i_vsc_D10_n_r = struct[0].y_run[104,0]
    p_D10 = struct[0].y_run[105,0]
    p_loss_R10 = struct[0].y_run[106,0]
    
    struct[0].u_run[0,0] = v_R0_a_r
    struct[0].u_run[1,0] = v_R0_a_i
    struct[0].u_run[2,0] = v_R0_b_r
    struct[0].u_run[3,0] = v_R0_b_i
    struct[0].u_run[4,0] = v_R0_c_r
    struct[0].u_run[5,0] = v_R0_c_i
    struct[0].u_run[6,0] = v_D1_a_r
    struct[0].u_run[7,0] = v_D1_a_i
    struct[0].u_run[8,0] = v_D1_b_r
    struct[0].u_run[9,0] = v_D1_b_i
    struct[0].u_run[10,0] = v_D1_c_r
    struct[0].u_run[11,0] = v_D1_c_i
    struct[0].u_run[12,0] = i_R1_n_r
    struct[0].u_run[13,0] = i_R1_n_i
    struct[0].u_run[14,0] = i_R10_a_r
    struct[0].u_run[15,0] = i_R10_a_i
    struct[0].u_run[16,0] = i_R10_b_r
    struct[0].u_run[17,0] = i_R10_b_i
    struct[0].u_run[18,0] = i_R10_c_r
    struct[0].u_run[19,0] = i_R10_c_i
    struct[0].u_run[20,0] = i_R10_n_r
    struct[0].u_run[21,0] = i_R10_n_i
    struct[0].u_run[22,0] = i_R18_b_r
    struct[0].u_run[23,0] = i_R18_b_i
    struct[0].u_run[24,0] = i_R18_c_r
    struct[0].u_run[25,0] = i_R18_c_i
    struct[0].u_run[26,0] = i_D1_n_r
    struct[0].u_run[27,0] = i_D1_n_i
    struct[0].u_run[28,0] = i_D10_a_i
    struct[0].u_run[29,0] = i_D10_b_r
    struct[0].u_run[30,0] = i_D10_b_i
    struct[0].u_run[31,0] = i_D10_c_r
    struct[0].u_run[32,0] = i_D10_c_i
    struct[0].u_run[33,0] = i_D10_n_i
    struct[0].u_run[34,0] = i_D18_b_r
    struct[0].u_run[35,0] = i_D18_b_i
    struct[0].u_run[36,0] = i_D18_c_r
    struct[0].u_run[37,0] = i_D18_c_i
    struct[0].u_run[38,0] = p_R1_a
    struct[0].u_run[39,0] = q_R1_a
    struct[0].u_run[40,0] = p_R1_b
    struct[0].u_run[41,0] = q_R1_b
    struct[0].u_run[42,0] = p_R1_c
    struct[0].u_run[43,0] = q_R1_c
    struct[0].u_run[44,0] = p_R18_1
    struct[0].u_run[45,0] = q_R18_1
    struct[0].u_run[46,0] = p_D18_1
    struct[0].u_run[47,0] = q_D18_1
    struct[0].u_run[48,0] = v_dc_D1
    struct[0].u_run[49,0] = q_R1
    struct[0].u_run[50,0] = p_R10
    struct[0].u_run[51,0] = q_R10
    struct[0].u_run[52,0] = u_dummy
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[72,0] = i_load_R1_a_i*v_R1_a_i - i_load_R1_a_i*v_R1_n_i + i_load_R1_a_r*v_R1_a_r - i_load_R1_a_r*v_R1_n_r - p_R1_a
        struct[0].g[73,0] = i_load_R1_b_i*v_R1_b_i - i_load_R1_b_i*v_R1_n_i + i_load_R1_b_r*v_R1_b_r - i_load_R1_b_r*v_R1_n_r - p_R1_b
        struct[0].g[74,0] = i_load_R1_c_i*v_R1_c_i - i_load_R1_c_i*v_R1_n_i + i_load_R1_c_r*v_R1_c_r - i_load_R1_c_r*v_R1_n_r - p_R1_c
        struct[0].g[75,0] = -i_load_R1_a_i*v_R1_a_r + i_load_R1_a_i*v_R1_n_r + i_load_R1_a_r*v_R1_a_i - i_load_R1_a_r*v_R1_n_i - q_R1_a
        struct[0].g[76,0] = -i_load_R1_b_i*v_R1_b_r + i_load_R1_b_i*v_R1_n_r + i_load_R1_b_r*v_R1_b_i - i_load_R1_b_r*v_R1_n_i - q_R1_b
        struct[0].g[77,0] = -i_load_R1_c_i*v_R1_c_r + i_load_R1_c_i*v_R1_n_r + i_load_R1_c_r*v_R1_c_i - i_load_R1_c_r*v_R1_n_i - q_R1_c
        struct[0].g[80,0] = 1.0*i_load_R18_a_i*v_R18_a_i - 1.0*i_load_R18_a_i*v_R18_n_i + i_load_R18_a_r*v_R18_a_r - i_load_R18_a_r*v_R18_n_r - p_R18_1
        struct[0].g[81,0] = -1.0*i_load_R18_a_i*v_R18_a_r + 1.0*i_load_R18_a_i*v_R18_n_r + 1.0*i_load_R18_a_r*v_R18_a_i - 1.0*i_load_R18_a_r*v_R18_n_i - q_R18_1
        struct[0].g[84,0] = 1.0*i_load_D18_a_i*v_D18_a_i - 1.0*i_load_D18_a_i*v_D18_n_i + i_load_D18_a_r*v_D18_a_r - i_load_D18_a_r*v_D18_n_r - p_D18_1
        struct[0].g[85,0] = -1.0*i_load_D18_a_i*v_D18_a_r + 1.0*i_load_D18_a_i*v_D18_n_r + 1.0*i_load_D18_a_r*v_D18_a_i - 1.0*i_load_D18_a_r*v_D18_n_i - q_D18_1
        struct[0].g[88,0] = 1.0*i_vsc_R1_a_i*v_R1_a_i - 1.0*i_vsc_R1_a_i*v_R1_n_i + i_vsc_R1_a_r*v_R1_a_r - i_vsc_R1_a_r*v_R1_n_r - p_R1/3
        struct[0].g[89,0] = -1.0*i_vsc_R1_a_i*v_R1_a_r + 1.0*i_vsc_R1_a_i*v_R1_n_r + 1.0*i_vsc_R1_a_r*v_R1_a_i - 1.0*i_vsc_R1_a_r*v_R1_n_i - q_R1/3
        struct[0].g[90,0] = 1.0*i_vsc_R1_b_i*v_R1_b_i - 1.0*i_vsc_R1_b_i*v_R1_n_i + i_vsc_R1_b_r*v_R1_b_r - i_vsc_R1_b_r*v_R1_n_r - p_R1/3
        struct[0].g[91,0] = -1.0*i_vsc_R1_b_i*v_R1_b_r + 1.0*i_vsc_R1_b_i*v_R1_n_r + 1.0*i_vsc_R1_b_r*v_R1_b_i - 1.0*i_vsc_R1_b_r*v_R1_n_i - q_R1/3
        struct[0].g[92,0] = 1.0*i_vsc_R1_c_i*v_R1_c_i - 1.0*i_vsc_R1_c_i*v_R1_n_i + i_vsc_R1_c_r*v_R1_c_r - i_vsc_R1_c_r*v_R1_n_r - p_R1/3
        struct[0].g[93,0] = -1.0*i_vsc_R1_c_i*v_R1_c_r + 1.0*i_vsc_R1_c_i*v_R1_n_r + 1.0*i_vsc_R1_c_r*v_R1_c_i - 1.0*i_vsc_R1_c_r*v_R1_n_i - q_R1/3
        struct[0].g[94,0] = p_D1 + p_R1 + Piecewise(np.array([(-p_loss_R1, p_D1 < 0), (p_loss_R1, True)]))
        struct[0].g[96,0] = -a_R1 - b_R1*sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - c_R1*(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) + p_loss_R1
        struct[0].g[97,0] = -coef_a_R10*p_R10 + 1.0*i_vsc_R10_a_i*v_R10_a_i - 1.0*i_vsc_R10_a_i*v_R10_n_i + i_vsc_R10_a_r*v_R10_a_r - i_vsc_R10_a_r*v_R10_n_r
        struct[0].g[98,0] = -coef_a_R10*q_R10 - 1.0*i_vsc_R10_a_i*v_R10_a_r + 1.0*i_vsc_R10_a_i*v_R10_n_r + 1.0*i_vsc_R10_a_r*v_R10_a_i - 1.0*i_vsc_R10_a_r*v_R10_n_i
        struct[0].g[99,0] = -coef_b_R10*p_R10 + 1.0*i_vsc_R10_b_i*v_R10_b_i - 1.0*i_vsc_R10_b_i*v_R10_n_i + i_vsc_R10_b_r*v_R10_b_r - i_vsc_R10_b_r*v_R10_n_r
        struct[0].g[100,0] = -coef_b_R10*q_R10 - 1.0*i_vsc_R10_b_i*v_R10_b_r + 1.0*i_vsc_R10_b_i*v_R10_n_r + 1.0*i_vsc_R10_b_r*v_R10_b_i - 1.0*i_vsc_R10_b_r*v_R10_n_i
        struct[0].g[101,0] = -coef_c_R10*p_R10 + 1.0*i_vsc_R10_c_i*v_R10_c_i - 1.0*i_vsc_R10_c_i*v_R10_n_i + i_vsc_R10_c_r*v_R10_c_r - i_vsc_R10_c_r*v_R10_n_r
        struct[0].g[102,0] = -coef_c_R10*q_R10 - 1.0*i_vsc_R10_c_i*v_R10_c_r + 1.0*i_vsc_R10_c_i*v_R10_n_r + 1.0*i_vsc_R10_c_r*v_R10_c_i - 1.0*i_vsc_R10_c_r*v_R10_n_i
        struct[0].g[103,0] = i_vsc_D10_a_r + p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].g[104,0] = i_vsc_D10_n_r + p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].g[105,0] = p_D10 - p_R10 - Piecewise(np.array([(-p_loss_R10, p_D10 < 0), (p_loss_R10, True)]))
        struct[0].g[106,0] = -a_R10 - b_R10*sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - c_R10*(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) + p_loss_R10
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_R0_a_i**2 + v_R0_a_r**2)**0.5
        struct[0].h[1,0] = (v_R0_b_i**2 + v_R0_b_r**2)**0.5
        struct[0].h[2,0] = (v_R0_c_i**2 + v_R0_c_r**2)**0.5
        struct[0].h[3,0] = (v_D1_a_i**2 + v_D1_a_r**2)**0.5
        struct[0].h[4,0] = (v_D1_b_i**2 + v_D1_b_r**2)**0.5
        struct[0].h[5,0] = (v_D1_c_i**2 + v_D1_c_r**2)**0.5
        struct[0].h[6,0] = (v_R1_a_i**2 + v_R1_a_r**2)**0.5
        struct[0].h[7,0] = (v_R1_b_i**2 + v_R1_b_r**2)**0.5
        struct[0].h[8,0] = (v_R1_c_i**2 + v_R1_c_r**2)**0.5
        struct[0].h[9,0] = (v_R1_n_i**2 + v_R1_n_r**2)**0.5
        struct[0].h[10,0] = (v_R18_a_i**2 + v_R18_a_r**2)**0.5
        struct[0].h[11,0] = (v_R18_n_i**2 + v_R18_n_r**2)**0.5
        struct[0].h[12,0] = (v_D18_a_i**2 + v_D18_a_r**2)**0.5
        struct[0].h[13,0] = (v_D18_n_i**2 + v_D18_n_r**2)**0.5
        struct[0].h[14,0] = (v_R10_a_i**2 + v_R10_a_r**2)**0.5
        struct[0].h[15,0] = (v_R10_b_i**2 + v_R10_b_r**2)**0.5
        struct[0].h[16,0] = (v_R10_c_i**2 + v_R10_c_r**2)**0.5
        struct[0].h[17,0] = (v_R10_n_i**2 + v_R10_n_r**2)**0.5
        struct[0].h[18,0] = (v_R18_b_i**2 + v_R18_b_r**2)**0.5
        struct[0].h[19,0] = (v_R18_c_i**2 + v_R18_c_r**2)**0.5
        struct[0].h[20,0] = (v_D1_n_i**2 + v_D1_n_r**2)**0.5
        struct[0].h[21,0] = (v_D10_a_i**2 + v_D10_a_r**2)**0.5
        struct[0].h[22,0] = (v_D10_b_i**2 + v_D10_b_r**2)**0.5
        struct[0].h[23,0] = (v_D10_c_i**2 + v_D10_c_r**2)**0.5
        struct[0].h[24,0] = (v_D10_n_i**2 + v_D10_n_r**2)**0.5
        struct[0].h[25,0] = (v_D18_b_i**2 + v_D18_b_r**2)**0.5
        struct[0].h[26,0] = (v_D18_c_i**2 + v_D18_c_r**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:



        struct[0].Gy[72,0] = i_load_R1_a_r
        struct[0].Gy[72,1] = i_load_R1_a_i
        struct[0].Gy[72,6] = -i_load_R1_a_r
        struct[0].Gy[72,7] = -i_load_R1_a_i
        struct[0].Gy[72,72] = v_R1_a_r - v_R1_n_r
        struct[0].Gy[72,73] = v_R1_a_i - v_R1_n_i
        struct[0].Gy[73,2] = i_load_R1_b_r
        struct[0].Gy[73,3] = i_load_R1_b_i
        struct[0].Gy[73,6] = -i_load_R1_b_r
        struct[0].Gy[73,7] = -i_load_R1_b_i
        struct[0].Gy[73,74] = v_R1_b_r - v_R1_n_r
        struct[0].Gy[73,75] = v_R1_b_i - v_R1_n_i
        struct[0].Gy[74,4] = i_load_R1_c_r
        struct[0].Gy[74,5] = i_load_R1_c_i
        struct[0].Gy[74,6] = -i_load_R1_c_r
        struct[0].Gy[74,7] = -i_load_R1_c_i
        struct[0].Gy[74,76] = v_R1_c_r - v_R1_n_r
        struct[0].Gy[74,77] = v_R1_c_i - v_R1_n_i
        struct[0].Gy[75,0] = -i_load_R1_a_i
        struct[0].Gy[75,1] = i_load_R1_a_r
        struct[0].Gy[75,6] = i_load_R1_a_i
        struct[0].Gy[75,7] = -i_load_R1_a_r
        struct[0].Gy[75,72] = v_R1_a_i - v_R1_n_i
        struct[0].Gy[75,73] = -v_R1_a_r + v_R1_n_r
        struct[0].Gy[76,2] = -i_load_R1_b_i
        struct[0].Gy[76,3] = i_load_R1_b_r
        struct[0].Gy[76,6] = i_load_R1_b_i
        struct[0].Gy[76,7] = -i_load_R1_b_r
        struct[0].Gy[76,74] = v_R1_b_i - v_R1_n_i
        struct[0].Gy[76,75] = -v_R1_b_r + v_R1_n_r
        struct[0].Gy[77,4] = -i_load_R1_c_i
        struct[0].Gy[77,5] = i_load_R1_c_r
        struct[0].Gy[77,6] = i_load_R1_c_i
        struct[0].Gy[77,7] = -i_load_R1_c_r
        struct[0].Gy[77,76] = v_R1_c_i - v_R1_n_i
        struct[0].Gy[77,77] = -v_R1_c_r + v_R1_n_r
        struct[0].Gy[80,8] = i_load_R18_a_r
        struct[0].Gy[80,9] = 1.0*i_load_R18_a_i
        struct[0].Gy[80,10] = -i_load_R18_a_r
        struct[0].Gy[80,11] = -1.0*i_load_R18_a_i
        struct[0].Gy[80,80] = v_R18_a_r - v_R18_n_r
        struct[0].Gy[80,81] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy[81,8] = -1.0*i_load_R18_a_i
        struct[0].Gy[81,9] = 1.0*i_load_R18_a_r
        struct[0].Gy[81,10] = 1.0*i_load_R18_a_i
        struct[0].Gy[81,11] = -1.0*i_load_R18_a_r
        struct[0].Gy[81,80] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy[81,81] = -1.0*v_R18_a_r + 1.0*v_R18_n_r
        struct[0].Gy[84,12] = i_load_D18_a_r
        struct[0].Gy[84,13] = 1.0*i_load_D18_a_i
        struct[0].Gy[84,14] = -i_load_D18_a_r
        struct[0].Gy[84,15] = -1.0*i_load_D18_a_i
        struct[0].Gy[84,84] = v_D18_a_r - v_D18_n_r
        struct[0].Gy[84,85] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy[85,12] = -1.0*i_load_D18_a_i
        struct[0].Gy[85,13] = 1.0*i_load_D18_a_r
        struct[0].Gy[85,14] = 1.0*i_load_D18_a_i
        struct[0].Gy[85,15] = -1.0*i_load_D18_a_r
        struct[0].Gy[85,84] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy[85,85] = -1.0*v_D18_a_r + 1.0*v_D18_n_r
        struct[0].Gy[88,0] = i_vsc_R1_a_r
        struct[0].Gy[88,1] = 1.0*i_vsc_R1_a_i
        struct[0].Gy[88,6] = -i_vsc_R1_a_r
        struct[0].Gy[88,7] = -1.0*i_vsc_R1_a_i
        struct[0].Gy[88,88] = v_R1_a_r - v_R1_n_r
        struct[0].Gy[88,89] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy[89,0] = -1.0*i_vsc_R1_a_i
        struct[0].Gy[89,1] = 1.0*i_vsc_R1_a_r
        struct[0].Gy[89,6] = 1.0*i_vsc_R1_a_i
        struct[0].Gy[89,7] = -1.0*i_vsc_R1_a_r
        struct[0].Gy[89,88] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy[89,89] = -1.0*v_R1_a_r + 1.0*v_R1_n_r
        struct[0].Gy[90,2] = i_vsc_R1_b_r
        struct[0].Gy[90,3] = 1.0*i_vsc_R1_b_i
        struct[0].Gy[90,6] = -i_vsc_R1_b_r
        struct[0].Gy[90,7] = -1.0*i_vsc_R1_b_i
        struct[0].Gy[90,90] = v_R1_b_r - v_R1_n_r
        struct[0].Gy[90,91] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy[91,2] = -1.0*i_vsc_R1_b_i
        struct[0].Gy[91,3] = 1.0*i_vsc_R1_b_r
        struct[0].Gy[91,6] = 1.0*i_vsc_R1_b_i
        struct[0].Gy[91,7] = -1.0*i_vsc_R1_b_r
        struct[0].Gy[91,90] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy[91,91] = -1.0*v_R1_b_r + 1.0*v_R1_n_r
        struct[0].Gy[92,4] = i_vsc_R1_c_r
        struct[0].Gy[92,5] = 1.0*i_vsc_R1_c_i
        struct[0].Gy[92,6] = -i_vsc_R1_c_r
        struct[0].Gy[92,7] = -1.0*i_vsc_R1_c_i
        struct[0].Gy[92,92] = v_R1_c_r - v_R1_n_r
        struct[0].Gy[92,93] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy[93,4] = -1.0*i_vsc_R1_c_i
        struct[0].Gy[93,5] = 1.0*i_vsc_R1_c_r
        struct[0].Gy[93,6] = 1.0*i_vsc_R1_c_i
        struct[0].Gy[93,7] = -1.0*i_vsc_R1_c_r
        struct[0].Gy[93,92] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy[93,93] = -1.0*v_R1_c_r + 1.0*v_R1_n_r
        struct[0].Gy[94,96] = Piecewise(np.array([(-1, p_D1 < 0), (1, True)]))
        struct[0].Gy[95,56] = v_D1_a_r
        struct[0].Gy[95,62] = v_D1_n_r
        struct[0].Gy[96,88] = -b_R1*i_vsc_R1_a_r/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_r
        struct[0].Gy[96,89] = -b_R1*i_vsc_R1_a_i/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_i
        struct[0].Gy[97,16] = i_vsc_R10_a_r
        struct[0].Gy[97,17] = 1.0*i_vsc_R10_a_i
        struct[0].Gy[97,22] = -i_vsc_R10_a_r
        struct[0].Gy[97,23] = -1.0*i_vsc_R10_a_i
        struct[0].Gy[97,97] = v_R10_a_r - v_R10_n_r
        struct[0].Gy[97,98] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy[98,16] = -1.0*i_vsc_R10_a_i
        struct[0].Gy[98,17] = 1.0*i_vsc_R10_a_r
        struct[0].Gy[98,22] = 1.0*i_vsc_R10_a_i
        struct[0].Gy[98,23] = -1.0*i_vsc_R10_a_r
        struct[0].Gy[98,97] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy[98,98] = -1.0*v_R10_a_r + 1.0*v_R10_n_r
        struct[0].Gy[99,18] = i_vsc_R10_b_r
        struct[0].Gy[99,19] = 1.0*i_vsc_R10_b_i
        struct[0].Gy[99,22] = -i_vsc_R10_b_r
        struct[0].Gy[99,23] = -1.0*i_vsc_R10_b_i
        struct[0].Gy[99,99] = v_R10_b_r - v_R10_n_r
        struct[0].Gy[99,100] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy[100,18] = -1.0*i_vsc_R10_b_i
        struct[0].Gy[100,19] = 1.0*i_vsc_R10_b_r
        struct[0].Gy[100,22] = 1.0*i_vsc_R10_b_i
        struct[0].Gy[100,23] = -1.0*i_vsc_R10_b_r
        struct[0].Gy[100,99] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy[100,100] = -1.0*v_R10_b_r + 1.0*v_R10_n_r
        struct[0].Gy[101,20] = i_vsc_R10_c_r
        struct[0].Gy[101,21] = 1.0*i_vsc_R10_c_i
        struct[0].Gy[101,22] = -i_vsc_R10_c_r
        struct[0].Gy[101,23] = -1.0*i_vsc_R10_c_i
        struct[0].Gy[101,101] = v_R10_c_r - v_R10_n_r
        struct[0].Gy[101,102] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy[102,20] = -1.0*i_vsc_R10_c_i
        struct[0].Gy[102,21] = 1.0*i_vsc_R10_c_r
        struct[0].Gy[102,22] = 1.0*i_vsc_R10_c_i
        struct[0].Gy[102,23] = -1.0*i_vsc_R10_c_r
        struct[0].Gy[102,101] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy[102,102] = -1.0*v_R10_c_r + 1.0*v_R10_n_r
        struct[0].Gy[103,30] = -p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[103,36] = p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[103,105] = 1/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].Gy[104,30] = p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[104,36] = -p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[104,105] = 1/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].Gy[105,106] = -Piecewise(np.array([(-1, p_D10 < 0), (1, True)]))
        struct[0].Gy[106,97] = -b_R10*i_vsc_R10_a_r/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_r
        struct[0].Gy[106,98] = -b_R10*i_vsc_R10_a_i/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_i

    if mode > 12:


        struct[0].Gu[97,50] = -coef_a_R10
        struct[0].Gu[98,51] = -coef_a_R10
        struct[0].Gu[99,50] = -coef_b_R10
        struct[0].Gu[100,51] = -coef_b_R10
        struct[0].Gu[101,50] = -coef_c_R10
        struct[0].Gu[102,51] = -coef_c_R10


        struct[0].Hy[6,0] = 1.0*v_R1_a_r*(v_R1_a_i**2 + v_R1_a_r**2)**(-0.5)
        struct[0].Hy[6,1] = 1.0*v_R1_a_i*(v_R1_a_i**2 + v_R1_a_r**2)**(-0.5)
        struct[0].Hy[7,2] = 1.0*v_R1_b_r*(v_R1_b_i**2 + v_R1_b_r**2)**(-0.5)
        struct[0].Hy[7,3] = 1.0*v_R1_b_i*(v_R1_b_i**2 + v_R1_b_r**2)**(-0.5)
        struct[0].Hy[8,4] = 1.0*v_R1_c_r*(v_R1_c_i**2 + v_R1_c_r**2)**(-0.5)
        struct[0].Hy[8,5] = 1.0*v_R1_c_i*(v_R1_c_i**2 + v_R1_c_r**2)**(-0.5)
        struct[0].Hy[9,6] = 1.0*v_R1_n_r*(v_R1_n_i**2 + v_R1_n_r**2)**(-0.5)
        struct[0].Hy[9,7] = 1.0*v_R1_n_i*(v_R1_n_i**2 + v_R1_n_r**2)**(-0.5)
        struct[0].Hy[10,8] = 1.0*v_R18_a_r*(v_R18_a_i**2 + v_R18_a_r**2)**(-0.5)
        struct[0].Hy[10,9] = 1.0*v_R18_a_i*(v_R18_a_i**2 + v_R18_a_r**2)**(-0.5)
        struct[0].Hy[11,10] = 1.0*v_R18_n_r*(v_R18_n_i**2 + v_R18_n_r**2)**(-0.5)
        struct[0].Hy[11,11] = 1.0*v_R18_n_i*(v_R18_n_i**2 + v_R18_n_r**2)**(-0.5)
        struct[0].Hy[12,12] = 1.0*v_D18_a_r*(v_D18_a_i**2 + v_D18_a_r**2)**(-0.5)
        struct[0].Hy[12,13] = 1.0*v_D18_a_i*(v_D18_a_i**2 + v_D18_a_r**2)**(-0.5)
        struct[0].Hy[13,14] = 1.0*v_D18_n_r*(v_D18_n_i**2 + v_D18_n_r**2)**(-0.5)
        struct[0].Hy[13,15] = 1.0*v_D18_n_i*(v_D18_n_i**2 + v_D18_n_r**2)**(-0.5)
        struct[0].Hy[14,16] = 1.0*v_R10_a_r*(v_R10_a_i**2 + v_R10_a_r**2)**(-0.5)
        struct[0].Hy[14,17] = 1.0*v_R10_a_i*(v_R10_a_i**2 + v_R10_a_r**2)**(-0.5)
        struct[0].Hy[15,18] = 1.0*v_R10_b_r*(v_R10_b_i**2 + v_R10_b_r**2)**(-0.5)
        struct[0].Hy[15,19] = 1.0*v_R10_b_i*(v_R10_b_i**2 + v_R10_b_r**2)**(-0.5)
        struct[0].Hy[16,20] = 1.0*v_R10_c_r*(v_R10_c_i**2 + v_R10_c_r**2)**(-0.5)
        struct[0].Hy[16,21] = 1.0*v_R10_c_i*(v_R10_c_i**2 + v_R10_c_r**2)**(-0.5)
        struct[0].Hy[17,22] = 1.0*v_R10_n_r*(v_R10_n_i**2 + v_R10_n_r**2)**(-0.5)
        struct[0].Hy[17,23] = 1.0*v_R10_n_i*(v_R10_n_i**2 + v_R10_n_r**2)**(-0.5)
        struct[0].Hy[18,24] = 1.0*v_R18_b_r*(v_R18_b_i**2 + v_R18_b_r**2)**(-0.5)
        struct[0].Hy[18,25] = 1.0*v_R18_b_i*(v_R18_b_i**2 + v_R18_b_r**2)**(-0.5)
        struct[0].Hy[19,26] = 1.0*v_R18_c_r*(v_R18_c_i**2 + v_R18_c_r**2)**(-0.5)
        struct[0].Hy[19,27] = 1.0*v_R18_c_i*(v_R18_c_i**2 + v_R18_c_r**2)**(-0.5)
        struct[0].Hy[20,28] = 1.0*v_D1_n_r*(v_D1_n_i**2 + v_D1_n_r**2)**(-0.5)
        struct[0].Hy[20,29] = 1.0*v_D1_n_i*(v_D1_n_i**2 + v_D1_n_r**2)**(-0.5)
        struct[0].Hy[21,30] = 1.0*v_D10_a_r*(v_D10_a_i**2 + v_D10_a_r**2)**(-0.5)
        struct[0].Hy[21,31] = 1.0*v_D10_a_i*(v_D10_a_i**2 + v_D10_a_r**2)**(-0.5)
        struct[0].Hy[22,32] = 1.0*v_D10_b_r*(v_D10_b_i**2 + v_D10_b_r**2)**(-0.5)
        struct[0].Hy[22,33] = 1.0*v_D10_b_i*(v_D10_b_i**2 + v_D10_b_r**2)**(-0.5)
        struct[0].Hy[23,34] = 1.0*v_D10_c_r*(v_D10_c_i**2 + v_D10_c_r**2)**(-0.5)
        struct[0].Hy[23,35] = 1.0*v_D10_c_i*(v_D10_c_i**2 + v_D10_c_r**2)**(-0.5)
        struct[0].Hy[24,36] = 1.0*v_D10_n_r*(v_D10_n_i**2 + v_D10_n_r**2)**(-0.5)
        struct[0].Hy[24,37] = 1.0*v_D10_n_i*(v_D10_n_i**2 + v_D10_n_r**2)**(-0.5)
        struct[0].Hy[25,38] = 1.0*v_D18_b_r*(v_D18_b_i**2 + v_D18_b_r**2)**(-0.5)
        struct[0].Hy[25,39] = 1.0*v_D18_b_i*(v_D18_b_i**2 + v_D18_b_r**2)**(-0.5)
        struct[0].Hy[26,40] = 1.0*v_D18_c_r*(v_D18_c_i**2 + v_D18_c_r**2)**(-0.5)
        struct[0].Hy[26,41] = 1.0*v_D18_c_i*(v_D18_c_i**2 + v_D18_c_r**2)**(-0.5)

        struct[0].Hu[0,0] = 1.0*v_R0_a_r*(v_R0_a_i**2 + v_R0_a_r**2)**(-0.5)
        struct[0].Hu[0,1] = 1.0*v_R0_a_i*(v_R0_a_i**2 + v_R0_a_r**2)**(-0.5)
        struct[0].Hu[1,2] = 1.0*v_R0_b_r*(v_R0_b_i**2 + v_R0_b_r**2)**(-0.5)
        struct[0].Hu[1,3] = 1.0*v_R0_b_i*(v_R0_b_i**2 + v_R0_b_r**2)**(-0.5)
        struct[0].Hu[2,4] = 1.0*v_R0_c_r*(v_R0_c_i**2 + v_R0_c_r**2)**(-0.5)
        struct[0].Hu[2,5] = 1.0*v_R0_c_i*(v_R0_c_i**2 + v_R0_c_r**2)**(-0.5)
        struct[0].Hu[3,6] = 1.0*v_D1_a_r*(v_D1_a_i**2 + v_D1_a_r**2)**(-0.5)
        struct[0].Hu[3,7] = 1.0*v_D1_a_i*(v_D1_a_i**2 + v_D1_a_r**2)**(-0.5)
        struct[0].Hu[4,8] = 1.0*v_D1_b_r*(v_D1_b_i**2 + v_D1_b_r**2)**(-0.5)
        struct[0].Hu[4,9] = 1.0*v_D1_b_i*(v_D1_b_i**2 + v_D1_b_r**2)**(-0.5)
        struct[0].Hu[5,10] = 1.0*v_D1_c_r*(v_D1_c_i**2 + v_D1_c_r**2)**(-0.5)
        struct[0].Hu[5,11] = 1.0*v_D1_c_i*(v_D1_c_i**2 + v_D1_c_r**2)**(-0.5)



def ini_nn(struct,mode):

    # Parameters:
    a_R1 = struct[0].a_R1
    b_R1 = struct[0].b_R1
    c_R1 = struct[0].c_R1
    a_R10 = struct[0].a_R10
    b_R10 = struct[0].b_R10
    c_R10 = struct[0].c_R10
    coef_a_R10 = struct[0].coef_a_R10
    coef_b_R10 = struct[0].coef_b_R10
    coef_c_R10 = struct[0].coef_c_R10
    
    # Inputs:
    v_R0_a_r = struct[0].v_R0_a_r
    v_R0_a_i = struct[0].v_R0_a_i
    v_R0_b_r = struct[0].v_R0_b_r
    v_R0_b_i = struct[0].v_R0_b_i
    v_R0_c_r = struct[0].v_R0_c_r
    v_R0_c_i = struct[0].v_R0_c_i
    v_D1_a_r = struct[0].v_D1_a_r
    v_D1_a_i = struct[0].v_D1_a_i
    v_D1_b_r = struct[0].v_D1_b_r
    v_D1_b_i = struct[0].v_D1_b_i
    v_D1_c_r = struct[0].v_D1_c_r
    v_D1_c_i = struct[0].v_D1_c_i
    i_R1_n_r = struct[0].i_R1_n_r
    i_R1_n_i = struct[0].i_R1_n_i
    i_R10_a_r = struct[0].i_R10_a_r
    i_R10_a_i = struct[0].i_R10_a_i
    i_R10_b_r = struct[0].i_R10_b_r
    i_R10_b_i = struct[0].i_R10_b_i
    i_R10_c_r = struct[0].i_R10_c_r
    i_R10_c_i = struct[0].i_R10_c_i
    i_R10_n_r = struct[0].i_R10_n_r
    i_R10_n_i = struct[0].i_R10_n_i
    i_R18_b_r = struct[0].i_R18_b_r
    i_R18_b_i = struct[0].i_R18_b_i
    i_R18_c_r = struct[0].i_R18_c_r
    i_R18_c_i = struct[0].i_R18_c_i
    i_D1_n_r = struct[0].i_D1_n_r
    i_D1_n_i = struct[0].i_D1_n_i
    i_D10_a_i = struct[0].i_D10_a_i
    i_D10_b_r = struct[0].i_D10_b_r
    i_D10_b_i = struct[0].i_D10_b_i
    i_D10_c_r = struct[0].i_D10_c_r
    i_D10_c_i = struct[0].i_D10_c_i
    i_D10_n_i = struct[0].i_D10_n_i
    i_D18_b_r = struct[0].i_D18_b_r
    i_D18_b_i = struct[0].i_D18_b_i
    i_D18_c_r = struct[0].i_D18_c_r
    i_D18_c_i = struct[0].i_D18_c_i
    p_R1_a = struct[0].p_R1_a
    q_R1_a = struct[0].q_R1_a
    p_R1_b = struct[0].p_R1_b
    q_R1_b = struct[0].q_R1_b
    p_R1_c = struct[0].p_R1_c
    q_R1_c = struct[0].q_R1_c
    p_R18_1 = struct[0].p_R18_1
    q_R18_1 = struct[0].q_R18_1
    p_D18_1 = struct[0].p_D18_1
    q_D18_1 = struct[0].q_D18_1
    v_dc_D1 = struct[0].v_dc_D1
    q_R1 = struct[0].q_R1
    p_R10 = struct[0].p_R10
    q_R10 = struct[0].q_R10
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_R1_a_r = struct[0].y_ini[0,0]
    v_R1_a_i = struct[0].y_ini[1,0]
    v_R1_b_r = struct[0].y_ini[2,0]
    v_R1_b_i = struct[0].y_ini[3,0]
    v_R1_c_r = struct[0].y_ini[4,0]
    v_R1_c_i = struct[0].y_ini[5,0]
    v_R1_n_r = struct[0].y_ini[6,0]
    v_R1_n_i = struct[0].y_ini[7,0]
    v_R18_a_r = struct[0].y_ini[8,0]
    v_R18_a_i = struct[0].y_ini[9,0]
    v_R18_n_r = struct[0].y_ini[10,0]
    v_R18_n_i = struct[0].y_ini[11,0]
    v_D18_a_r = struct[0].y_ini[12,0]
    v_D18_a_i = struct[0].y_ini[13,0]
    v_D18_n_r = struct[0].y_ini[14,0]
    v_D18_n_i = struct[0].y_ini[15,0]
    v_R10_a_r = struct[0].y_ini[16,0]
    v_R10_a_i = struct[0].y_ini[17,0]
    v_R10_b_r = struct[0].y_ini[18,0]
    v_R10_b_i = struct[0].y_ini[19,0]
    v_R10_c_r = struct[0].y_ini[20,0]
    v_R10_c_i = struct[0].y_ini[21,0]
    v_R10_n_r = struct[0].y_ini[22,0]
    v_R10_n_i = struct[0].y_ini[23,0]
    v_R18_b_r = struct[0].y_ini[24,0]
    v_R18_b_i = struct[0].y_ini[25,0]
    v_R18_c_r = struct[0].y_ini[26,0]
    v_R18_c_i = struct[0].y_ini[27,0]
    v_D1_n_r = struct[0].y_ini[28,0]
    v_D1_n_i = struct[0].y_ini[29,0]
    v_D10_a_r = struct[0].y_ini[30,0]
    v_D10_a_i = struct[0].y_ini[31,0]
    v_D10_b_r = struct[0].y_ini[32,0]
    v_D10_b_i = struct[0].y_ini[33,0]
    v_D10_c_r = struct[0].y_ini[34,0]
    v_D10_c_i = struct[0].y_ini[35,0]
    v_D10_n_r = struct[0].y_ini[36,0]
    v_D10_n_i = struct[0].y_ini[37,0]
    v_D18_b_r = struct[0].y_ini[38,0]
    v_D18_b_i = struct[0].y_ini[39,0]
    v_D18_c_r = struct[0].y_ini[40,0]
    v_D18_c_i = struct[0].y_ini[41,0]
    i_t_R0_R1_a_r = struct[0].y_ini[42,0]
    i_t_R0_R1_a_i = struct[0].y_ini[43,0]
    i_t_R0_R1_b_r = struct[0].y_ini[44,0]
    i_t_R0_R1_b_i = struct[0].y_ini[45,0]
    i_t_R0_R1_c_r = struct[0].y_ini[46,0]
    i_t_R0_R1_c_i = struct[0].y_ini[47,0]
    i_l_R1_R10_a_r = struct[0].y_ini[48,0]
    i_l_R1_R10_a_i = struct[0].y_ini[49,0]
    i_l_R1_R10_b_r = struct[0].y_ini[50,0]
    i_l_R1_R10_b_i = struct[0].y_ini[51,0]
    i_l_R1_R10_c_r = struct[0].y_ini[52,0]
    i_l_R1_R10_c_i = struct[0].y_ini[53,0]
    i_l_R1_R10_n_r = struct[0].y_ini[54,0]
    i_l_R1_R10_n_i = struct[0].y_ini[55,0]
    i_l_D1_D10_a_r = struct[0].y_ini[56,0]
    i_l_D1_D10_a_i = struct[0].y_ini[57,0]
    i_l_D1_D10_b_r = struct[0].y_ini[58,0]
    i_l_D1_D10_b_i = struct[0].y_ini[59,0]
    i_l_D1_D10_c_r = struct[0].y_ini[60,0]
    i_l_D1_D10_c_i = struct[0].y_ini[61,0]
    i_l_D1_D10_n_r = struct[0].y_ini[62,0]
    i_l_D1_D10_n_i = struct[0].y_ini[63,0]
    i_l_D10_D18_a_r = struct[0].y_ini[64,0]
    i_l_D10_D18_a_i = struct[0].y_ini[65,0]
    i_l_D10_D18_b_r = struct[0].y_ini[66,0]
    i_l_D10_D18_b_i = struct[0].y_ini[67,0]
    i_l_D10_D18_c_r = struct[0].y_ini[68,0]
    i_l_D10_D18_c_i = struct[0].y_ini[69,0]
    i_l_D10_D18_n_r = struct[0].y_ini[70,0]
    i_l_D10_D18_n_i = struct[0].y_ini[71,0]
    i_load_R1_a_r = struct[0].y_ini[72,0]
    i_load_R1_a_i = struct[0].y_ini[73,0]
    i_load_R1_b_r = struct[0].y_ini[74,0]
    i_load_R1_b_i = struct[0].y_ini[75,0]
    i_load_R1_c_r = struct[0].y_ini[76,0]
    i_load_R1_c_i = struct[0].y_ini[77,0]
    i_load_R1_n_r = struct[0].y_ini[78,0]
    i_load_R1_n_i = struct[0].y_ini[79,0]
    i_load_R18_a_r = struct[0].y_ini[80,0]
    i_load_R18_a_i = struct[0].y_ini[81,0]
    i_load_R18_n_r = struct[0].y_ini[82,0]
    i_load_R18_n_i = struct[0].y_ini[83,0]
    i_load_D18_a_r = struct[0].y_ini[84,0]
    i_load_D18_a_i = struct[0].y_ini[85,0]
    i_load_D18_n_r = struct[0].y_ini[86,0]
    i_load_D18_n_i = struct[0].y_ini[87,0]
    i_vsc_R1_a_r = struct[0].y_ini[88,0]
    i_vsc_R1_a_i = struct[0].y_ini[89,0]
    i_vsc_R1_b_r = struct[0].y_ini[90,0]
    i_vsc_R1_b_i = struct[0].y_ini[91,0]
    i_vsc_R1_c_r = struct[0].y_ini[92,0]
    i_vsc_R1_c_i = struct[0].y_ini[93,0]
    p_R1 = struct[0].y_ini[94,0]
    p_D1 = struct[0].y_ini[95,0]
    p_loss_R1 = struct[0].y_ini[96,0]
    i_vsc_R10_a_r = struct[0].y_ini[97,0]
    i_vsc_R10_a_i = struct[0].y_ini[98,0]
    i_vsc_R10_b_r = struct[0].y_ini[99,0]
    i_vsc_R10_b_i = struct[0].y_ini[100,0]
    i_vsc_R10_c_r = struct[0].y_ini[101,0]
    i_vsc_R10_c_i = struct[0].y_ini[102,0]
    i_vsc_D10_a_r = struct[0].y_ini[103,0]
    i_vsc_D10_n_r = struct[0].y_ini[104,0]
    p_D10 = struct[0].y_ini[105,0]
    p_loss_R10 = struct[0].y_ini[106,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_load_R1_a_r + i_vsc_R1_a_r + 0.849044513514155*v_R0_a_i + 0.212261128378539*v_R0_a_r - 0.849044513514155*v_R0_c_i - 0.212261128378539*v_R0_c_r + 5.40657727682604*v_R10_a_i + 10.557176931318*v_R10_a_r - 1.02713736253513*v_R10_b_i - 3.96392229058202*v_R10_b_r - 2.3284964480954*v_R10_c_i - 2.49575997948692*v_R10_c_r - 1.02713736253513*v_R10_n_i - 3.96392229058202*v_R10_n_r - 78.9359890415319*v_R1_a_i - 28.9395298724945*v_R1_a_r + 1.02713736253513*v_R1_b_i + 3.96392229058202*v_R1_b_r + 2.3284964480954*v_R1_c_i + 2.49575997948692*v_R1_c_r + 74.556549127241*v_R1_n_i + 22.3462752317585*v_R1_n_r
        struct[0].g[1,0] = i_load_R1_a_i + i_vsc_R1_a_i + 0.212261128378539*v_R0_a_i - 0.849044513514155*v_R0_a_r - 0.212261128378539*v_R0_c_i + 0.849044513514155*v_R0_c_r + 10.557176931318*v_R10_a_i - 5.40657727682604*v_R10_a_r - 3.96392229058202*v_R10_b_i + 1.02713736253513*v_R10_b_r - 2.49575997948692*v_R10_c_i + 2.3284964480954*v_R10_c_r - 3.96392229058202*v_R10_n_i + 1.02713736253513*v_R10_n_r - 28.9395298724945*v_R1_a_i + 78.9359890415319*v_R1_a_r + 3.96392229058202*v_R1_b_i - 1.02713736253513*v_R1_b_r + 2.49575997948692*v_R1_c_i - 2.3284964480954*v_R1_c_r + 22.3462752317585*v_R1_n_i - 74.556549127241*v_R1_n_r
        struct[0].g[2,0] = i_load_R1_b_r + i_vsc_R1_b_r - 0.849044513514155*v_R0_a_i - 0.212261128378539*v_R0_a_r + 0.849044513514155*v_R0_b_i + 0.212261128378539*v_R0_b_r - 1.02713736253513*v_R10_a_i - 3.96392229058202*v_R10_a_r + 5.40657727682604*v_R10_b_i + 10.557176931318*v_R10_b_r - 1.02713736253513*v_R10_c_i - 3.96392229058202*v_R10_c_r - 2.3284964480954*v_R10_n_i - 2.49575997948692*v_R10_n_r + 1.02713736253513*v_R1_a_i + 3.96392229058202*v_R1_a_r - 78.9359890415319*v_R1_b_i - 28.9395298724945*v_R1_b_r + 1.02713736253513*v_R1_c_i + 3.96392229058202*v_R1_c_r + 75.8579082128012*v_R1_n_i + 20.8781129206634*v_R1_n_r
        struct[0].g[3,0] = i_load_R1_b_i + i_vsc_R1_b_i - 0.212261128378539*v_R0_a_i + 0.849044513514155*v_R0_a_r + 0.212261128378539*v_R0_b_i - 0.849044513514155*v_R0_b_r - 3.96392229058202*v_R10_a_i + 1.02713736253513*v_R10_a_r + 10.557176931318*v_R10_b_i - 5.40657727682604*v_R10_b_r - 3.96392229058202*v_R10_c_i + 1.02713736253513*v_R10_c_r - 2.49575997948692*v_R10_n_i + 2.3284964480954*v_R10_n_r + 3.96392229058202*v_R1_a_i - 1.02713736253513*v_R1_a_r - 28.9395298724945*v_R1_b_i + 78.9359890415319*v_R1_b_r + 3.96392229058202*v_R1_c_i - 1.02713736253513*v_R1_c_r + 20.8781129206634*v_R1_n_i - 75.8579082128012*v_R1_n_r
        struct[0].g[4,0] = i_load_R1_c_r + i_vsc_R1_c_r - 0.849044513514155*v_R0_b_i - 0.212261128378539*v_R0_b_r + 0.849044513514155*v_R0_c_i + 0.212261128378539*v_R0_c_r - 2.3284964480954*v_R10_a_i - 2.49575997948692*v_R10_a_r - 1.02713736253513*v_R10_b_i - 3.96392229058202*v_R10_b_r + 5.40657727682604*v_R10_c_i + 10.557176931318*v_R10_c_r - 1.02713736253513*v_R10_n_i - 3.96392229058202*v_R10_n_r + 2.3284964480954*v_R1_a_i + 2.49575997948692*v_R1_a_r + 1.02713736253513*v_R1_b_i + 3.96392229058202*v_R1_b_r - 78.9359890415319*v_R1_c_i - 28.9395298724945*v_R1_c_r + 74.556549127241*v_R1_n_i + 22.3462752317585*v_R1_n_r
        struct[0].g[5,0] = i_load_R1_c_i + i_vsc_R1_c_i - 0.212261128378539*v_R0_b_i + 0.849044513514155*v_R0_b_r + 0.212261128378539*v_R0_c_i - 0.849044513514155*v_R0_c_r - 2.49575997948692*v_R10_a_i + 2.3284964480954*v_R10_a_r - 3.96392229058202*v_R10_b_i + 1.02713736253513*v_R10_b_r + 10.557176931318*v_R10_c_i - 5.40657727682604*v_R10_c_r - 3.96392229058202*v_R10_n_i + 1.02713736253513*v_R10_n_r + 2.49575997948692*v_R1_a_i - 2.3284964480954*v_R1_a_r + 3.96392229058202*v_R1_b_i - 1.02713736253513*v_R1_b_r - 28.9395298724945*v_R1_c_i + 78.9359890415319*v_R1_c_r + 22.3462752317585*v_R1_n_i - 74.556549127241*v_R1_n_r
        struct[0].g[6,0] = -1.02713736253513*v_R10_a_i - 3.96392229058202*v_R10_a_r - 2.3284964480954*v_R10_b_i - 2.49575997948692*v_R10_b_r - 1.02713736253513*v_R10_c_i - 3.96392229058202*v_R10_c_r + 5.40657727682604*v_R10_n_i + 10.557176931318*v_R10_n_r + 74.556549127241*v_R1_a_i + 22.3462752317585*v_R1_a_r + 75.8579082128012*v_R1_b_i + 20.8781129206634*v_R1_b_r + 74.556549127241*v_R1_c_i + 22.3462752317585*v_R1_c_r - 225.994812570944*v_R1_n_i - 66.0375690881807*v_R1_n_r
        struct[0].g[7,0] = -3.96392229058202*v_R10_a_i + 1.02713736253513*v_R10_a_r - 2.49575997948692*v_R10_b_i + 2.3284964480954*v_R10_b_r - 3.96392229058202*v_R10_c_i + 1.02713736253513*v_R10_c_r + 10.557176931318*v_R10_n_i - 5.40657727682604*v_R10_n_r + 22.3462752317585*v_R1_a_i - 74.556549127241*v_R1_a_r + 20.8781129206634*v_R1_b_i - 75.8579082128012*v_R1_b_r + 22.3462752317585*v_R1_c_i - 74.556549127241*v_R1_c_r - 66.0375690881807*v_R1_n_i + 225.994812570944*v_R1_n_r
        struct[0].g[8,0] = i_load_R18_a_r + 5.65456401516768*v_R10_a_i + 30.9517475172273*v_R10_a_r + 1.84896616921897*v_R10_b_i - 9.21038227100566*v_R10_b_r + 0.793238195499529*v_R10_c_i - 9.00835072044485*v_R10_c_r + 1.84896616921897*v_R10_n_i - 9.21038227100566*v_R10_n_r - 5.65456401516768*v_R18_a_i - 30.9517475172273*v_R18_a_r - 1.84896616921897*v_R18_b_i + 9.21038227100566*v_R18_b_r - 0.793238195499529*v_R18_c_i + 9.00835072044485*v_R18_c_r - 1.84896616921897*v_R18_n_i + 9.21038227100566*v_R18_n_r
        struct[0].g[9,0] = i_load_R18_a_i + 30.9517475172273*v_R10_a_i - 5.65456401516768*v_R10_a_r - 9.21038227100566*v_R10_b_i - 1.84896616921897*v_R10_b_r - 9.00835072044485*v_R10_c_i - 0.793238195499529*v_R10_c_r - 9.21038227100566*v_R10_n_i - 1.84896616921897*v_R10_n_r - 30.9517475172273*v_R18_a_i + 5.65456401516768*v_R18_a_r + 9.21038227100566*v_R18_b_i + 1.84896616921897*v_R18_b_r + 9.00835072044485*v_R18_c_i + 0.793238195499529*v_R18_c_r + 9.21038227100566*v_R18_n_i + 1.84896616921897*v_R18_n_r
        struct[0].g[10,0] = i_load_R18_n_r + 1.84896616921897*v_R10_a_i - 9.21038227100566*v_R10_a_r + 0.793238195499527*v_R10_b_i - 9.00835072044485*v_R10_b_r + 1.84896616921897*v_R10_c_i - 9.21038227100566*v_R10_c_r + 5.65456401516768*v_R10_n_i + 30.9517475172273*v_R10_n_r - 1.84896616921897*v_R18_a_i + 9.21038227100566*v_R18_a_r - 0.793238195499527*v_R18_b_i + 9.00835072044485*v_R18_b_r - 1.84896616921897*v_R18_c_i + 9.21038227100566*v_R18_c_r - 5.65456401516768*v_R18_n_i - 30.9767475172273*v_R18_n_r
        struct[0].g[11,0] = i_load_R18_n_i - 9.21038227100566*v_R10_a_i - 1.84896616921897*v_R10_a_r - 9.00835072044485*v_R10_b_i - 0.793238195499527*v_R10_b_r - 9.21038227100566*v_R10_c_i - 1.84896616921897*v_R10_c_r + 30.9517475172273*v_R10_n_i - 5.65456401516768*v_R10_n_r + 9.21038227100566*v_R18_a_i + 1.84896616921897*v_R18_a_r + 9.00835072044485*v_R18_b_i + 0.793238195499527*v_R18_b_r + 9.21038227100566*v_R18_c_i + 1.84896616921897*v_R18_c_r - 30.9767475172273*v_R18_n_i + 5.65456401516768*v_R18_n_r
        struct[0].g[12,0] = i_load_D18_a_r + 157.977883096366*v_D10_a_r - 157.977883096366*v_D18_a_r
        struct[0].g[13,0] = i_load_D18_a_i + 157.977883096366*v_D10_a_i - 157.977883096366*v_D18_a_i
        struct[0].g[14,0] = i_load_D18_n_r + 157.977883096366*v_D10_n_r - 157.977883096366*v_D18_n_r
        struct[0].g[15,0] = i_load_D18_n_i + 157.977883096366*v_D10_n_i - 157.977883096366*v_D18_n_i
        struct[0].g[16,0] = i_vsc_R10_a_r - 11.0611412919937*v_R10_a_i - 41.5089244485453*v_R10_a_r - 0.821828806683838*v_R10_b_i + 13.1743045615877*v_R10_b_r + 1.53525825259587*v_R10_c_i + 11.5041106999318*v_R10_c_r - 0.82182880668384*v_R10_n_i + 13.1743045615877*v_R10_n_r + 5.65456401516768*v_R18_a_i + 30.9517475172273*v_R18_a_r + 1.84896616921897*v_R18_b_i - 9.21038227100566*v_R18_b_r + 0.793238195499529*v_R18_c_i - 9.00835072044485*v_R18_c_r + 1.84896616921897*v_R18_n_i - 9.21038227100566*v_R18_n_r + 5.40657727682604*v_R1_a_i + 10.557176931318*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r - 2.3284964480954*v_R1_c_i - 2.49575997948692*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[17,0] = i_vsc_R10_a_i - 41.5089244485453*v_R10_a_i + 11.0611412919937*v_R10_a_r + 13.1743045615877*v_R10_b_i + 0.821828806683838*v_R10_b_r + 11.5041106999318*v_R10_c_i - 1.53525825259587*v_R10_c_r + 13.1743045615877*v_R10_n_i + 0.82182880668384*v_R10_n_r + 30.9517475172273*v_R18_a_i - 5.65456401516768*v_R18_a_r - 9.21038227100566*v_R18_b_i - 1.84896616921897*v_R18_b_r - 9.00835072044485*v_R18_c_i - 0.793238195499529*v_R18_c_r - 9.21038227100566*v_R18_n_i - 1.84896616921897*v_R18_n_r + 10.557176931318*v_R1_a_i - 5.40657727682604*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r - 2.49575997948692*v_R1_c_i + 2.3284964480954*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[18,0] = i_vsc_R10_b_r - 0.821828806683841*v_R10_a_i + 13.1743045615877*v_R10_a_r - 11.0611412919937*v_R10_b_i - 41.5089244485453*v_R10_b_r - 0.821828806683839*v_R10_c_i + 13.1743045615877*v_R10_c_r + 1.53525825259588*v_R10_n_i + 11.5041106999318*v_R10_n_r + 1.84896616921897*v_R18_a_i - 9.21038227100566*v_R18_a_r + 5.65456401516768*v_R18_b_i + 30.9517475172273*v_R18_b_r + 1.84896616921897*v_R18_c_i - 9.21038227100566*v_R18_c_r + 0.793238195499528*v_R18_n_i - 9.00835072044485*v_R18_n_r - 1.02713736253513*v_R1_a_i - 3.96392229058202*v_R1_a_r + 5.40657727682604*v_R1_b_i + 10.557176931318*v_R1_b_r - 1.02713736253513*v_R1_c_i - 3.96392229058202*v_R1_c_r - 2.3284964480954*v_R1_n_i - 2.49575997948692*v_R1_n_r
        struct[0].g[19,0] = i_vsc_R10_b_i + 13.1743045615877*v_R10_a_i + 0.821828806683841*v_R10_a_r - 41.5089244485453*v_R10_b_i + 11.0611412919937*v_R10_b_r + 13.1743045615877*v_R10_c_i + 0.821828806683839*v_R10_c_r + 11.5041106999318*v_R10_n_i - 1.53525825259588*v_R10_n_r - 9.21038227100566*v_R18_a_i - 1.84896616921897*v_R18_a_r + 30.9517475172273*v_R18_b_i - 5.65456401516768*v_R18_b_r - 9.21038227100566*v_R18_c_i - 1.84896616921897*v_R18_c_r - 9.00835072044485*v_R18_n_i - 0.793238195499528*v_R18_n_r - 3.96392229058202*v_R1_a_i + 1.02713736253513*v_R1_a_r + 10.557176931318*v_R1_b_i - 5.40657727682604*v_R1_b_r - 3.96392229058202*v_R1_c_i + 1.02713736253513*v_R1_c_r - 2.49575997948692*v_R1_n_i + 2.3284964480954*v_R1_n_r
        struct[0].g[20,0] = i_vsc_R10_c_r + 1.53525825259588*v_R10_a_i + 11.5041106999318*v_R10_a_r - 0.82182880668384*v_R10_b_i + 13.1743045615877*v_R10_b_r - 11.0611412919937*v_R10_c_i - 41.5089244485453*v_R10_c_r - 0.821828806683838*v_R10_n_i + 13.1743045615877*v_R10_n_r + 0.793238195499527*v_R18_a_i - 9.00835072044484*v_R18_a_r + 1.84896616921897*v_R18_b_i - 9.21038227100566*v_R18_b_r + 5.65456401516768*v_R18_c_i + 30.9517475172273*v_R18_c_r + 1.84896616921897*v_R18_n_i - 9.21038227100566*v_R18_n_r - 2.3284964480954*v_R1_a_i - 2.49575997948692*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r + 5.40657727682604*v_R1_c_i + 10.557176931318*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[21,0] = i_vsc_R10_c_i + 11.5041106999318*v_R10_a_i - 1.53525825259588*v_R10_a_r + 13.1743045615877*v_R10_b_i + 0.82182880668384*v_R10_b_r - 41.5089244485453*v_R10_c_i + 11.0611412919937*v_R10_c_r + 13.1743045615877*v_R10_n_i + 0.821828806683838*v_R10_n_r - 9.00835072044484*v_R18_a_i - 0.793238195499527*v_R18_a_r - 9.21038227100566*v_R18_b_i - 1.84896616921897*v_R18_b_r + 30.9517475172273*v_R18_c_i - 5.65456401516768*v_R18_c_r - 9.21038227100566*v_R18_n_i - 1.84896616921897*v_R18_n_r - 2.49575997948692*v_R1_a_i + 2.3284964480954*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r + 10.557176931318*v_R1_c_i - 5.40657727682604*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[22,0] = -0.82182880668384*v_R10_a_i + 13.1743045615877*v_R10_a_r + 1.53525825259588*v_R10_b_i + 11.5041106999318*v_R10_b_r - 0.821828806683837*v_R10_c_i + 13.1743045615877*v_R10_c_r - 11.0611412919937*v_R10_n_i - 41.5339244485453*v_R10_n_r + 1.84896616921897*v_R18_a_i - 9.21038227100566*v_R18_a_r + 0.793238195499527*v_R18_b_i - 9.00835072044485*v_R18_b_r + 1.84896616921897*v_R18_c_i - 9.21038227100566*v_R18_c_r + 5.65456401516768*v_R18_n_i + 30.9517475172273*v_R18_n_r - 1.02713736253513*v_R1_a_i - 3.96392229058202*v_R1_a_r - 2.3284964480954*v_R1_b_i - 2.49575997948692*v_R1_b_r - 1.02713736253513*v_R1_c_i - 3.96392229058202*v_R1_c_r + 5.40657727682604*v_R1_n_i + 10.557176931318*v_R1_n_r
        struct[0].g[23,0] = 13.1743045615877*v_R10_a_i + 0.82182880668384*v_R10_a_r + 11.5041106999318*v_R10_b_i - 1.53525825259588*v_R10_b_r + 13.1743045615877*v_R10_c_i + 0.821828806683837*v_R10_c_r - 41.5339244485453*v_R10_n_i + 11.0611412919937*v_R10_n_r - 9.21038227100566*v_R18_a_i - 1.84896616921897*v_R18_a_r - 9.00835072044485*v_R18_b_i - 0.793238195499527*v_R18_b_r - 9.21038227100566*v_R18_c_i - 1.84896616921897*v_R18_c_r + 30.9517475172273*v_R18_n_i - 5.65456401516768*v_R18_n_r - 3.96392229058202*v_R1_a_i + 1.02713736253513*v_R1_a_r - 2.49575997948692*v_R1_b_i + 2.3284964480954*v_R1_b_r - 3.96392229058202*v_R1_c_i + 1.02713736253513*v_R1_c_r + 10.557176931318*v_R1_n_i - 5.40657727682604*v_R1_n_r
        struct[0].g[24,0] = 1.84896616921897*v_R10_a_i - 9.21038227100566*v_R10_a_r + 5.65456401516768*v_R10_b_i + 30.9517475172273*v_R10_b_r + 1.84896616921897*v_R10_c_i - 9.21038227100566*v_R10_c_r + 0.793238195499528*v_R10_n_i - 9.00835072044485*v_R10_n_r - 1.84896616921897*v_R18_a_i + 9.21038227100566*v_R18_a_r - 5.65456401516768*v_R18_b_i - 30.9517475172273*v_R18_b_r - 1.84896616921897*v_R18_c_i + 9.21038227100566*v_R18_c_r - 0.793238195499528*v_R18_n_i + 9.00835072044485*v_R18_n_r
        struct[0].g[25,0] = -9.21038227100566*v_R10_a_i - 1.84896616921897*v_R10_a_r + 30.9517475172273*v_R10_b_i - 5.65456401516768*v_R10_b_r - 9.21038227100566*v_R10_c_i - 1.84896616921897*v_R10_c_r - 9.00835072044485*v_R10_n_i - 0.793238195499528*v_R10_n_r + 9.21038227100566*v_R18_a_i + 1.84896616921897*v_R18_a_r - 30.9517475172273*v_R18_b_i + 5.65456401516768*v_R18_b_r + 9.21038227100566*v_R18_c_i + 1.84896616921897*v_R18_c_r + 9.00835072044485*v_R18_n_i + 0.793238195499528*v_R18_n_r
        struct[0].g[26,0] = 0.793238195499527*v_R10_a_i - 9.00835072044484*v_R10_a_r + 1.84896616921897*v_R10_b_i - 9.21038227100566*v_R10_b_r + 5.65456401516768*v_R10_c_i + 30.9517475172273*v_R10_c_r + 1.84896616921897*v_R10_n_i - 9.21038227100566*v_R10_n_r - 0.793238195499527*v_R18_a_i + 9.00835072044484*v_R18_a_r - 1.84896616921897*v_R18_b_i + 9.21038227100566*v_R18_b_r - 5.65456401516768*v_R18_c_i - 30.9517475172273*v_R18_c_r - 1.84896616921897*v_R18_n_i + 9.21038227100566*v_R18_n_r
        struct[0].g[27,0] = -9.00835072044484*v_R10_a_i - 0.793238195499527*v_R10_a_r - 9.21038227100566*v_R10_b_i - 1.84896616921897*v_R10_b_r + 30.9517475172273*v_R10_c_i - 5.65456401516768*v_R10_c_r - 9.21038227100566*v_R10_n_i - 1.84896616921897*v_R10_n_r + 9.00835072044484*v_R18_a_i + 0.793238195499527*v_R18_a_r + 9.21038227100566*v_R18_b_i + 1.84896616921897*v_R18_b_r - 30.9517475172273*v_R18_c_i + 5.65456401516768*v_R18_c_r + 9.21038227100566*v_R18_n_i + 1.84896616921897*v_R18_n_r
        struct[0].g[28,0] = 67.7048070412999*v_D10_n_r - 1067.7048070413*v_D1_n_r
        struct[0].g[29,0] = 67.7048070412999*v_D10_n_i - 1067.7048070413*v_D1_n_i
        struct[0].g[30,0] = i_vsc_D10_a_r - 225.682690137666*v_D10_a_r + 157.977883096366*v_D18_a_r + 67.7048070412999*v_D1_a_r
        struct[0].g[31,0] = -225.682690137666*v_D10_a_i + 157.977883096366*v_D18_a_i + 67.7048070412999*v_D1_a_i
        struct[0].g[32,0] = -225.682690137666*v_D10_b_r + 157.977883096366*v_D18_b_r + 67.7048070412999*v_D1_b_r
        struct[0].g[33,0] = -225.682690137666*v_D10_b_i + 157.977883096366*v_D18_b_i + 67.7048070412999*v_D1_b_i
        struct[0].g[34,0] = -225.682690137666*v_D10_c_r + 157.977883096366*v_D18_c_r + 67.7048070412999*v_D1_c_r
        struct[0].g[35,0] = -225.682690137666*v_D10_c_i + 157.977883096366*v_D18_c_i + 67.7048070412999*v_D1_c_i
        struct[0].g[36,0] = i_vsc_D10_n_r - 225.682690137666*v_D10_n_r + 157.977883096366*v_D18_n_r + 67.7048070412999*v_D1_n_r
        struct[0].g[37,0] = -225.682690137666*v_D10_n_i + 157.977883096366*v_D18_n_i + 67.7048070412999*v_D1_n_i
        struct[0].g[38,0] = 157.977883096366*v_D10_b_r - 157.977883096366*v_D18_b_r
        struct[0].g[39,0] = 157.977883096366*v_D10_b_i - 157.977883096366*v_D18_b_i
        struct[0].g[40,0] = 157.977883096366*v_D10_c_r - 157.977883096366*v_D18_c_r
        struct[0].g[41,0] = 157.977883096366*v_D10_c_i - 157.977883096366*v_D18_c_i
        struct[0].g[42,0] = -i_t_R0_R1_a_r + 0.0196078431372549*v_R0_a_i + 0.00490196078431373*v_R0_a_r - 0.00980392156862745*v_R0_b_i - 0.00245098039215686*v_R0_b_r - 0.00980392156862745*v_R0_c_i - 0.00245098039215686*v_R0_c_r - 0.849044513514155*v_R1_a_i - 0.212261128378539*v_R1_a_r + 0.849044513514155*v_R1_b_i + 0.212261128378539*v_R1_b_r
        struct[0].g[43,0] = -i_t_R0_R1_a_i + 0.00490196078431373*v_R0_a_i - 0.0196078431372549*v_R0_a_r - 0.00245098039215686*v_R0_b_i + 0.00980392156862745*v_R0_b_r - 0.00245098039215686*v_R0_c_i + 0.00980392156862745*v_R0_c_r - 0.212261128378539*v_R1_a_i + 0.849044513514155*v_R1_a_r + 0.212261128378539*v_R1_b_i - 0.849044513514155*v_R1_b_r
        struct[0].g[44,0] = -i_t_R0_R1_b_r - 0.00980392156862745*v_R0_a_i - 0.00245098039215686*v_R0_a_r + 0.0196078431372549*v_R0_b_i + 0.00490196078431373*v_R0_b_r - 0.00980392156862745*v_R0_c_i - 0.00245098039215686*v_R0_c_r - 0.849044513514155*v_R1_b_i - 0.212261128378539*v_R1_b_r + 0.849044513514155*v_R1_c_i + 0.212261128378539*v_R1_c_r
        struct[0].g[45,0] = -i_t_R0_R1_b_i - 0.00245098039215686*v_R0_a_i + 0.00980392156862745*v_R0_a_r + 0.00490196078431373*v_R0_b_i - 0.0196078431372549*v_R0_b_r - 0.00245098039215686*v_R0_c_i + 0.00980392156862745*v_R0_c_r - 0.212261128378539*v_R1_b_i + 0.849044513514155*v_R1_b_r + 0.212261128378539*v_R1_c_i - 0.849044513514155*v_R1_c_r
        struct[0].g[46,0] = -i_t_R0_R1_c_r - 0.00980392156862745*v_R0_a_i - 0.00245098039215686*v_R0_a_r - 0.00980392156862745*v_R0_b_i - 0.00245098039215686*v_R0_b_r + 0.0196078431372549*v_R0_c_i + 0.00490196078431373*v_R0_c_r + 0.849044513514155*v_R1_a_i + 0.212261128378539*v_R1_a_r - 0.849044513514155*v_R1_c_i - 0.212261128378539*v_R1_c_r
        struct[0].g[47,0] = -i_t_R0_R1_c_i - 0.00245098039215686*v_R0_a_i + 0.00980392156862745*v_R0_a_r - 0.00245098039215686*v_R0_b_i + 0.00980392156862745*v_R0_b_r + 0.00490196078431373*v_R0_c_i - 0.0196078431372549*v_R0_c_r + 0.212261128378539*v_R1_a_i - 0.849044513514155*v_R1_a_r - 0.212261128378539*v_R1_c_i + 0.849044513514155*v_R1_c_r
        struct[0].g[48,0] = -i_l_R1_R10_a_r - 5.40657727682604*v_R10_a_i - 10.557176931318*v_R10_a_r + 1.02713736253513*v_R10_b_i + 3.96392229058202*v_R10_b_r + 2.3284964480954*v_R10_c_i + 2.49575997948692*v_R10_c_r + 1.02713736253513*v_R10_n_i + 3.96392229058202*v_R10_n_r + 5.40657727682604*v_R1_a_i + 10.557176931318*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r - 2.3284964480954*v_R1_c_i - 2.49575997948692*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[49,0] = -i_l_R1_R10_a_i - 10.557176931318*v_R10_a_i + 5.40657727682604*v_R10_a_r + 3.96392229058202*v_R10_b_i - 1.02713736253513*v_R10_b_r + 2.49575997948692*v_R10_c_i - 2.3284964480954*v_R10_c_r + 3.96392229058202*v_R10_n_i - 1.02713736253513*v_R10_n_r + 10.557176931318*v_R1_a_i - 5.40657727682604*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r - 2.49575997948692*v_R1_c_i + 2.3284964480954*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[50,0] = -i_l_R1_R10_b_r + 1.02713736253513*v_R10_a_i + 3.96392229058202*v_R10_a_r - 5.40657727682604*v_R10_b_i - 10.557176931318*v_R10_b_r + 1.02713736253513*v_R10_c_i + 3.96392229058202*v_R10_c_r + 2.3284964480954*v_R10_n_i + 2.49575997948692*v_R10_n_r - 1.02713736253513*v_R1_a_i - 3.96392229058202*v_R1_a_r + 5.40657727682604*v_R1_b_i + 10.557176931318*v_R1_b_r - 1.02713736253513*v_R1_c_i - 3.96392229058202*v_R1_c_r - 2.3284964480954*v_R1_n_i - 2.49575997948692*v_R1_n_r
        struct[0].g[51,0] = -i_l_R1_R10_b_i + 3.96392229058202*v_R10_a_i - 1.02713736253513*v_R10_a_r - 10.557176931318*v_R10_b_i + 5.40657727682604*v_R10_b_r + 3.96392229058202*v_R10_c_i - 1.02713736253513*v_R10_c_r + 2.49575997948692*v_R10_n_i - 2.3284964480954*v_R10_n_r - 3.96392229058202*v_R1_a_i + 1.02713736253513*v_R1_a_r + 10.557176931318*v_R1_b_i - 5.40657727682604*v_R1_b_r - 3.96392229058202*v_R1_c_i + 1.02713736253513*v_R1_c_r - 2.49575997948692*v_R1_n_i + 2.3284964480954*v_R1_n_r
        struct[0].g[52,0] = -i_l_R1_R10_c_r + 2.3284964480954*v_R10_a_i + 2.49575997948692*v_R10_a_r + 1.02713736253513*v_R10_b_i + 3.96392229058202*v_R10_b_r - 5.40657727682604*v_R10_c_i - 10.557176931318*v_R10_c_r + 1.02713736253513*v_R10_n_i + 3.96392229058202*v_R10_n_r - 2.3284964480954*v_R1_a_i - 2.49575997948692*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r + 5.40657727682604*v_R1_c_i + 10.557176931318*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[53,0] = -i_l_R1_R10_c_i + 2.49575997948692*v_R10_a_i - 2.3284964480954*v_R10_a_r + 3.96392229058202*v_R10_b_i - 1.02713736253513*v_R10_b_r - 10.557176931318*v_R10_c_i + 5.40657727682604*v_R10_c_r + 3.96392229058202*v_R10_n_i - 1.02713736253513*v_R10_n_r - 2.49575997948692*v_R1_a_i + 2.3284964480954*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r + 10.557176931318*v_R1_c_i - 5.40657727682604*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[54,0] = i_l_R1_R10_a_r + i_l_R1_R10_b_r + i_l_R1_R10_c_r - i_l_R1_R10_n_r
        struct[0].g[55,0] = i_l_R1_R10_a_i + i_l_R1_R10_b_i + i_l_R1_R10_c_i - i_l_R1_R10_n_i
        struct[0].g[56,0] = -i_l_D1_D10_a_r - 67.7048070412999*v_D10_a_r + 67.7048070412999*v_D1_a_r
        struct[0].g[57,0] = -i_l_D1_D10_a_i - 67.7048070412999*v_D10_a_i + 67.7048070412999*v_D1_a_i
        struct[0].g[58,0] = -i_l_D1_D10_b_r - 67.7048070412999*v_D10_b_r + 67.7048070412999*v_D1_b_r
        struct[0].g[59,0] = -i_l_D1_D10_b_i - 67.7048070412999*v_D10_b_i + 67.7048070412999*v_D1_b_i
        struct[0].g[60,0] = -i_l_D1_D10_c_r - 67.7048070412999*v_D10_c_r + 67.7048070412999*v_D1_c_r
        struct[0].g[61,0] = -i_l_D1_D10_c_i - 67.7048070412999*v_D10_c_i + 67.7048070412999*v_D1_c_i
        struct[0].g[62,0] = i_l_D1_D10_a_r + i_l_D1_D10_b_r + i_l_D1_D10_c_r - i_l_D1_D10_n_r
        struct[0].g[63,0] = i_l_D1_D10_a_i + i_l_D1_D10_b_i + i_l_D1_D10_c_i - i_l_D1_D10_n_i
        struct[0].g[64,0] = -i_l_D10_D18_a_r + 157.977883096366*v_D10_a_r - 157.977883096366*v_D18_a_r
        struct[0].g[65,0] = -i_l_D10_D18_a_i + 157.977883096366*v_D10_a_i - 157.977883096366*v_D18_a_i
        struct[0].g[66,0] = -i_l_D10_D18_b_r + 157.977883096366*v_D10_b_r - 157.977883096366*v_D18_b_r
        struct[0].g[67,0] = -i_l_D10_D18_b_i + 157.977883096366*v_D10_b_i - 157.977883096366*v_D18_b_i
        struct[0].g[68,0] = -i_l_D10_D18_c_r + 157.977883096366*v_D10_c_r - 157.977883096366*v_D18_c_r
        struct[0].g[69,0] = -i_l_D10_D18_c_i + 157.977883096366*v_D10_c_i - 157.977883096366*v_D18_c_i
        struct[0].g[70,0] = i_l_D10_D18_a_r + i_l_D10_D18_b_r + i_l_D10_D18_c_r - i_l_D10_D18_n_r
        struct[0].g[71,0] = i_l_D10_D18_a_i + i_l_D10_D18_b_i + i_l_D10_D18_c_i - i_l_D10_D18_n_i
        struct[0].g[72,0] = i_load_R1_a_i*v_R1_a_i - i_load_R1_a_i*v_R1_n_i + i_load_R1_a_r*v_R1_a_r - i_load_R1_a_r*v_R1_n_r - p_R1_a
        struct[0].g[73,0] = i_load_R1_b_i*v_R1_b_i - i_load_R1_b_i*v_R1_n_i + i_load_R1_b_r*v_R1_b_r - i_load_R1_b_r*v_R1_n_r - p_R1_b
        struct[0].g[74,0] = i_load_R1_c_i*v_R1_c_i - i_load_R1_c_i*v_R1_n_i + i_load_R1_c_r*v_R1_c_r - i_load_R1_c_r*v_R1_n_r - p_R1_c
        struct[0].g[75,0] = -i_load_R1_a_i*v_R1_a_r + i_load_R1_a_i*v_R1_n_r + i_load_R1_a_r*v_R1_a_i - i_load_R1_a_r*v_R1_n_i - q_R1_a
        struct[0].g[76,0] = -i_load_R1_b_i*v_R1_b_r + i_load_R1_b_i*v_R1_n_r + i_load_R1_b_r*v_R1_b_i - i_load_R1_b_r*v_R1_n_i - q_R1_b
        struct[0].g[77,0] = -i_load_R1_c_i*v_R1_c_r + i_load_R1_c_i*v_R1_n_r + i_load_R1_c_r*v_R1_c_i - i_load_R1_c_r*v_R1_n_i - q_R1_c
        struct[0].g[78,0] = i_load_R1_a_r + i_load_R1_b_r + i_load_R1_c_r + i_load_R1_n_r
        struct[0].g[79,0] = i_load_R1_a_i + i_load_R1_b_i + i_load_R1_c_i + i_load_R1_n_i
        struct[0].g[80,0] = 1.0*i_load_R18_a_i*v_R18_a_i - 1.0*i_load_R18_a_i*v_R18_n_i + i_load_R18_a_r*v_R18_a_r - i_load_R18_a_r*v_R18_n_r - p_R18_1
        struct[0].g[81,0] = -1.0*i_load_R18_a_i*v_R18_a_r + 1.0*i_load_R18_a_i*v_R18_n_r + 1.0*i_load_R18_a_r*v_R18_a_i - 1.0*i_load_R18_a_r*v_R18_n_i - q_R18_1
        struct[0].g[82,0] = i_load_R18_a_r + i_load_R18_n_r
        struct[0].g[83,0] = 1.0*i_load_R18_a_i + 1.0*i_load_R18_n_i
        struct[0].g[84,0] = 1.0*i_load_D18_a_i*v_D18_a_i - 1.0*i_load_D18_a_i*v_D18_n_i + i_load_D18_a_r*v_D18_a_r - i_load_D18_a_r*v_D18_n_r - p_D18_1
        struct[0].g[85,0] = -1.0*i_load_D18_a_i*v_D18_a_r + 1.0*i_load_D18_a_i*v_D18_n_r + 1.0*i_load_D18_a_r*v_D18_a_i - 1.0*i_load_D18_a_r*v_D18_n_i - q_D18_1
        struct[0].g[86,0] = i_load_D18_a_r + i_load_D18_n_r
        struct[0].g[87,0] = 1.0*i_load_D18_a_i + 1.0*i_load_D18_n_i
        struct[0].g[88,0] = 1.0*i_vsc_R1_a_i*v_R1_a_i - 1.0*i_vsc_R1_a_i*v_R1_n_i + i_vsc_R1_a_r*v_R1_a_r - i_vsc_R1_a_r*v_R1_n_r - p_R1/3
        struct[0].g[89,0] = -1.0*i_vsc_R1_a_i*v_R1_a_r + 1.0*i_vsc_R1_a_i*v_R1_n_r + 1.0*i_vsc_R1_a_r*v_R1_a_i - 1.0*i_vsc_R1_a_r*v_R1_n_i - q_R1/3
        struct[0].g[90,0] = 1.0*i_vsc_R1_b_i*v_R1_b_i - 1.0*i_vsc_R1_b_i*v_R1_n_i + i_vsc_R1_b_r*v_R1_b_r - i_vsc_R1_b_r*v_R1_n_r - p_R1/3
        struct[0].g[91,0] = -1.0*i_vsc_R1_b_i*v_R1_b_r + 1.0*i_vsc_R1_b_i*v_R1_n_r + 1.0*i_vsc_R1_b_r*v_R1_b_i - 1.0*i_vsc_R1_b_r*v_R1_n_i - q_R1/3
        struct[0].g[92,0] = 1.0*i_vsc_R1_c_i*v_R1_c_i - 1.0*i_vsc_R1_c_i*v_R1_n_i + i_vsc_R1_c_r*v_R1_c_r - i_vsc_R1_c_r*v_R1_n_r - p_R1/3
        struct[0].g[93,0] = -1.0*i_vsc_R1_c_i*v_R1_c_r + 1.0*i_vsc_R1_c_i*v_R1_n_r + 1.0*i_vsc_R1_c_r*v_R1_c_i - 1.0*i_vsc_R1_c_r*v_R1_n_i - q_R1/3
        struct[0].g[94,0] = p_D1 + p_R1 + Piecewise(np.array([(-p_loss_R1, p_D1 < 0), (p_loss_R1, True)]))
        struct[0].g[95,0] = i_l_D1_D10_a_r*v_D1_a_r + i_l_D1_D10_n_r*v_D1_n_r - p_D1
        struct[0].g[96,0] = -a_R1 - b_R1*sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - c_R1*(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) + p_loss_R1
        struct[0].g[97,0] = -coef_a_R10*p_R10 + 1.0*i_vsc_R10_a_i*v_R10_a_i - 1.0*i_vsc_R10_a_i*v_R10_n_i + i_vsc_R10_a_r*v_R10_a_r - i_vsc_R10_a_r*v_R10_n_r
        struct[0].g[98,0] = -coef_a_R10*q_R10 - 1.0*i_vsc_R10_a_i*v_R10_a_r + 1.0*i_vsc_R10_a_i*v_R10_n_r + 1.0*i_vsc_R10_a_r*v_R10_a_i - 1.0*i_vsc_R10_a_r*v_R10_n_i
        struct[0].g[99,0] = -coef_b_R10*p_R10 + 1.0*i_vsc_R10_b_i*v_R10_b_i - 1.0*i_vsc_R10_b_i*v_R10_n_i + i_vsc_R10_b_r*v_R10_b_r - i_vsc_R10_b_r*v_R10_n_r
        struct[0].g[100,0] = -coef_b_R10*q_R10 - 1.0*i_vsc_R10_b_i*v_R10_b_r + 1.0*i_vsc_R10_b_i*v_R10_n_r + 1.0*i_vsc_R10_b_r*v_R10_b_i - 1.0*i_vsc_R10_b_r*v_R10_n_i
        struct[0].g[101,0] = -coef_c_R10*p_R10 + 1.0*i_vsc_R10_c_i*v_R10_c_i - 1.0*i_vsc_R10_c_i*v_R10_n_i + i_vsc_R10_c_r*v_R10_c_r - i_vsc_R10_c_r*v_R10_n_r
        struct[0].g[102,0] = -coef_c_R10*q_R10 - 1.0*i_vsc_R10_c_i*v_R10_c_r + 1.0*i_vsc_R10_c_i*v_R10_n_r + 1.0*i_vsc_R10_c_r*v_R10_c_i - 1.0*i_vsc_R10_c_r*v_R10_n_i
        struct[0].g[103,0] = i_vsc_D10_a_r + p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].g[104,0] = i_vsc_D10_n_r + p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].g[105,0] = p_D10 - p_R10 - Piecewise(np.array([(-p_loss_R10, p_D10 < 0), (p_loss_R10, True)]))
        struct[0].g[106,0] = -a_R10 - b_R10*sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - c_R10*(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) + p_loss_R10
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_R0_a_i**2 + v_R0_a_r**2)**0.5
        struct[0].h[1,0] = (v_R0_b_i**2 + v_R0_b_r**2)**0.5
        struct[0].h[2,0] = (v_R0_c_i**2 + v_R0_c_r**2)**0.5
        struct[0].h[3,0] = (v_D1_a_i**2 + v_D1_a_r**2)**0.5
        struct[0].h[4,0] = (v_D1_b_i**2 + v_D1_b_r**2)**0.5
        struct[0].h[5,0] = (v_D1_c_i**2 + v_D1_c_r**2)**0.5
        struct[0].h[6,0] = (v_R1_a_i**2 + v_R1_a_r**2)**0.5
        struct[0].h[7,0] = (v_R1_b_i**2 + v_R1_b_r**2)**0.5
        struct[0].h[8,0] = (v_R1_c_i**2 + v_R1_c_r**2)**0.5
        struct[0].h[9,0] = (v_R1_n_i**2 + v_R1_n_r**2)**0.5
        struct[0].h[10,0] = (v_R18_a_i**2 + v_R18_a_r**2)**0.5
        struct[0].h[11,0] = (v_R18_n_i**2 + v_R18_n_r**2)**0.5
        struct[0].h[12,0] = (v_D18_a_i**2 + v_D18_a_r**2)**0.5
        struct[0].h[13,0] = (v_D18_n_i**2 + v_D18_n_r**2)**0.5
        struct[0].h[14,0] = (v_R10_a_i**2 + v_R10_a_r**2)**0.5
        struct[0].h[15,0] = (v_R10_b_i**2 + v_R10_b_r**2)**0.5
        struct[0].h[16,0] = (v_R10_c_i**2 + v_R10_c_r**2)**0.5
        struct[0].h[17,0] = (v_R10_n_i**2 + v_R10_n_r**2)**0.5
        struct[0].h[18,0] = (v_R18_b_i**2 + v_R18_b_r**2)**0.5
        struct[0].h[19,0] = (v_R18_c_i**2 + v_R18_c_r**2)**0.5
        struct[0].h[20,0] = (v_D1_n_i**2 + v_D1_n_r**2)**0.5
        struct[0].h[21,0] = (v_D10_a_i**2 + v_D10_a_r**2)**0.5
        struct[0].h[22,0] = (v_D10_b_i**2 + v_D10_b_r**2)**0.5
        struct[0].h[23,0] = (v_D10_c_i**2 + v_D10_c_r**2)**0.5
        struct[0].h[24,0] = (v_D10_n_i**2 + v_D10_n_r**2)**0.5
        struct[0].h[25,0] = (v_D18_b_i**2 + v_D18_b_r**2)**0.5
        struct[0].h[26,0] = (v_D18_c_i**2 + v_D18_c_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -1

    if mode == 11:


        struct[0].Gy_ini[0,0] = -28.9395298724945
        struct[0].Gy_ini[0,1] = -78.9359890415319
        struct[0].Gy_ini[0,2] = 3.96392229058202
        struct[0].Gy_ini[0,3] = 1.02713736253513
        struct[0].Gy_ini[0,4] = 2.49575997948692
        struct[0].Gy_ini[0,5] = 2.32849644809540
        struct[0].Gy_ini[0,6] = 22.3462752317585
        struct[0].Gy_ini[0,7] = 74.5565491272410
        struct[0].Gy_ini[0,16] = 10.5571769313180
        struct[0].Gy_ini[0,17] = 5.40657727682604
        struct[0].Gy_ini[0,18] = -3.96392229058202
        struct[0].Gy_ini[0,19] = -1.02713736253513
        struct[0].Gy_ini[0,20] = -2.49575997948692
        struct[0].Gy_ini[0,21] = -2.32849644809540
        struct[0].Gy_ini[0,22] = -3.96392229058202
        struct[0].Gy_ini[0,23] = -1.02713736253513
        struct[0].Gy_ini[0,72] = 1
        struct[0].Gy_ini[0,88] = 1
        struct[0].Gy_ini[1,0] = 78.9359890415319
        struct[0].Gy_ini[1,1] = -28.9395298724945
        struct[0].Gy_ini[1,2] = -1.02713736253513
        struct[0].Gy_ini[1,3] = 3.96392229058202
        struct[0].Gy_ini[1,4] = -2.32849644809540
        struct[0].Gy_ini[1,5] = 2.49575997948692
        struct[0].Gy_ini[1,6] = -74.5565491272410
        struct[0].Gy_ini[1,7] = 22.3462752317585
        struct[0].Gy_ini[1,16] = -5.40657727682604
        struct[0].Gy_ini[1,17] = 10.5571769313180
        struct[0].Gy_ini[1,18] = 1.02713736253513
        struct[0].Gy_ini[1,19] = -3.96392229058202
        struct[0].Gy_ini[1,20] = 2.32849644809540
        struct[0].Gy_ini[1,21] = -2.49575997948692
        struct[0].Gy_ini[1,22] = 1.02713736253513
        struct[0].Gy_ini[1,23] = -3.96392229058202
        struct[0].Gy_ini[1,73] = 1
        struct[0].Gy_ini[1,89] = 1
        struct[0].Gy_ini[2,0] = 3.96392229058202
        struct[0].Gy_ini[2,1] = 1.02713736253513
        struct[0].Gy_ini[2,2] = -28.9395298724945
        struct[0].Gy_ini[2,3] = -78.9359890415319
        struct[0].Gy_ini[2,4] = 3.96392229058202
        struct[0].Gy_ini[2,5] = 1.02713736253513
        struct[0].Gy_ini[2,6] = 20.8781129206634
        struct[0].Gy_ini[2,7] = 75.8579082128012
        struct[0].Gy_ini[2,16] = -3.96392229058202
        struct[0].Gy_ini[2,17] = -1.02713736253513
        struct[0].Gy_ini[2,18] = 10.5571769313180
        struct[0].Gy_ini[2,19] = 5.40657727682604
        struct[0].Gy_ini[2,20] = -3.96392229058202
        struct[0].Gy_ini[2,21] = -1.02713736253513
        struct[0].Gy_ini[2,22] = -2.49575997948692
        struct[0].Gy_ini[2,23] = -2.32849644809540
        struct[0].Gy_ini[2,74] = 1
        struct[0].Gy_ini[2,90] = 1
        struct[0].Gy_ini[3,0] = -1.02713736253513
        struct[0].Gy_ini[3,1] = 3.96392229058202
        struct[0].Gy_ini[3,2] = 78.9359890415319
        struct[0].Gy_ini[3,3] = -28.9395298724945
        struct[0].Gy_ini[3,4] = -1.02713736253513
        struct[0].Gy_ini[3,5] = 3.96392229058202
        struct[0].Gy_ini[3,6] = -75.8579082128012
        struct[0].Gy_ini[3,7] = 20.8781129206634
        struct[0].Gy_ini[3,16] = 1.02713736253513
        struct[0].Gy_ini[3,17] = -3.96392229058202
        struct[0].Gy_ini[3,18] = -5.40657727682604
        struct[0].Gy_ini[3,19] = 10.5571769313180
        struct[0].Gy_ini[3,20] = 1.02713736253513
        struct[0].Gy_ini[3,21] = -3.96392229058202
        struct[0].Gy_ini[3,22] = 2.32849644809540
        struct[0].Gy_ini[3,23] = -2.49575997948692
        struct[0].Gy_ini[3,75] = 1
        struct[0].Gy_ini[3,91] = 1
        struct[0].Gy_ini[4,0] = 2.49575997948692
        struct[0].Gy_ini[4,1] = 2.32849644809540
        struct[0].Gy_ini[4,2] = 3.96392229058202
        struct[0].Gy_ini[4,3] = 1.02713736253513
        struct[0].Gy_ini[4,4] = -28.9395298724945
        struct[0].Gy_ini[4,5] = -78.9359890415319
        struct[0].Gy_ini[4,6] = 22.3462752317585
        struct[0].Gy_ini[4,7] = 74.5565491272410
        struct[0].Gy_ini[4,16] = -2.49575997948692
        struct[0].Gy_ini[4,17] = -2.32849644809540
        struct[0].Gy_ini[4,18] = -3.96392229058202
        struct[0].Gy_ini[4,19] = -1.02713736253513
        struct[0].Gy_ini[4,20] = 10.5571769313180
        struct[0].Gy_ini[4,21] = 5.40657727682604
        struct[0].Gy_ini[4,22] = -3.96392229058202
        struct[0].Gy_ini[4,23] = -1.02713736253513
        struct[0].Gy_ini[4,76] = 1
        struct[0].Gy_ini[4,92] = 1
        struct[0].Gy_ini[5,0] = -2.32849644809540
        struct[0].Gy_ini[5,1] = 2.49575997948692
        struct[0].Gy_ini[5,2] = -1.02713736253513
        struct[0].Gy_ini[5,3] = 3.96392229058202
        struct[0].Gy_ini[5,4] = 78.9359890415319
        struct[0].Gy_ini[5,5] = -28.9395298724945
        struct[0].Gy_ini[5,6] = -74.5565491272410
        struct[0].Gy_ini[5,7] = 22.3462752317585
        struct[0].Gy_ini[5,16] = 2.32849644809540
        struct[0].Gy_ini[5,17] = -2.49575997948692
        struct[0].Gy_ini[5,18] = 1.02713736253513
        struct[0].Gy_ini[5,19] = -3.96392229058202
        struct[0].Gy_ini[5,20] = -5.40657727682604
        struct[0].Gy_ini[5,21] = 10.5571769313180
        struct[0].Gy_ini[5,22] = 1.02713736253513
        struct[0].Gy_ini[5,23] = -3.96392229058202
        struct[0].Gy_ini[5,77] = 1
        struct[0].Gy_ini[5,93] = 1
        struct[0].Gy_ini[6,0] = 22.3462752317585
        struct[0].Gy_ini[6,1] = 74.5565491272410
        struct[0].Gy_ini[6,2] = 20.8781129206634
        struct[0].Gy_ini[6,3] = 75.8579082128012
        struct[0].Gy_ini[6,4] = 22.3462752317585
        struct[0].Gy_ini[6,5] = 74.5565491272410
        struct[0].Gy_ini[6,6] = -66.0375690881807
        struct[0].Gy_ini[6,7] = -225.994812570944
        struct[0].Gy_ini[6,16] = -3.96392229058202
        struct[0].Gy_ini[6,17] = -1.02713736253513
        struct[0].Gy_ini[6,18] = -2.49575997948692
        struct[0].Gy_ini[6,19] = -2.32849644809540
        struct[0].Gy_ini[6,20] = -3.96392229058202
        struct[0].Gy_ini[6,21] = -1.02713736253513
        struct[0].Gy_ini[6,22] = 10.5571769313180
        struct[0].Gy_ini[6,23] = 5.40657727682604
        struct[0].Gy_ini[7,0] = -74.5565491272410
        struct[0].Gy_ini[7,1] = 22.3462752317585
        struct[0].Gy_ini[7,2] = -75.8579082128012
        struct[0].Gy_ini[7,3] = 20.8781129206634
        struct[0].Gy_ini[7,4] = -74.5565491272410
        struct[0].Gy_ini[7,5] = 22.3462752317585
        struct[0].Gy_ini[7,6] = 225.994812570944
        struct[0].Gy_ini[7,7] = -66.0375690881807
        struct[0].Gy_ini[7,16] = 1.02713736253513
        struct[0].Gy_ini[7,17] = -3.96392229058202
        struct[0].Gy_ini[7,18] = 2.32849644809540
        struct[0].Gy_ini[7,19] = -2.49575997948692
        struct[0].Gy_ini[7,20] = 1.02713736253513
        struct[0].Gy_ini[7,21] = -3.96392229058202
        struct[0].Gy_ini[7,22] = -5.40657727682604
        struct[0].Gy_ini[7,23] = 10.5571769313180
        struct[0].Gy_ini[8,8] = -30.9517475172273
        struct[0].Gy_ini[8,9] = -5.65456401516768
        struct[0].Gy_ini[8,10] = 9.21038227100566
        struct[0].Gy_ini[8,11] = -1.84896616921897
        struct[0].Gy_ini[8,16] = 30.9517475172273
        struct[0].Gy_ini[8,17] = 5.65456401516768
        struct[0].Gy_ini[8,18] = -9.21038227100566
        struct[0].Gy_ini[8,19] = 1.84896616921897
        struct[0].Gy_ini[8,20] = -9.00835072044485
        struct[0].Gy_ini[8,21] = 0.793238195499529
        struct[0].Gy_ini[8,22] = -9.21038227100566
        struct[0].Gy_ini[8,23] = 1.84896616921897
        struct[0].Gy_ini[8,24] = 9.21038227100566
        struct[0].Gy_ini[8,25] = -1.84896616921897
        struct[0].Gy_ini[8,26] = 9.00835072044485
        struct[0].Gy_ini[8,27] = -0.793238195499529
        struct[0].Gy_ini[8,80] = 1
        struct[0].Gy_ini[9,8] = 5.65456401516768
        struct[0].Gy_ini[9,9] = -30.9517475172273
        struct[0].Gy_ini[9,10] = 1.84896616921897
        struct[0].Gy_ini[9,11] = 9.21038227100566
        struct[0].Gy_ini[9,16] = -5.65456401516768
        struct[0].Gy_ini[9,17] = 30.9517475172273
        struct[0].Gy_ini[9,18] = -1.84896616921897
        struct[0].Gy_ini[9,19] = -9.21038227100566
        struct[0].Gy_ini[9,20] = -0.793238195499529
        struct[0].Gy_ini[9,21] = -9.00835072044485
        struct[0].Gy_ini[9,22] = -1.84896616921897
        struct[0].Gy_ini[9,23] = -9.21038227100566
        struct[0].Gy_ini[9,24] = 1.84896616921897
        struct[0].Gy_ini[9,25] = 9.21038227100566
        struct[0].Gy_ini[9,26] = 0.793238195499529
        struct[0].Gy_ini[9,27] = 9.00835072044485
        struct[0].Gy_ini[9,81] = 1
        struct[0].Gy_ini[10,8] = 9.21038227100566
        struct[0].Gy_ini[10,9] = -1.84896616921897
        struct[0].Gy_ini[10,10] = -30.9767475172273
        struct[0].Gy_ini[10,11] = -5.65456401516768
        struct[0].Gy_ini[10,16] = -9.21038227100566
        struct[0].Gy_ini[10,17] = 1.84896616921897
        struct[0].Gy_ini[10,18] = -9.00835072044485
        struct[0].Gy_ini[10,19] = 0.793238195499527
        struct[0].Gy_ini[10,20] = -9.21038227100566
        struct[0].Gy_ini[10,21] = 1.84896616921897
        struct[0].Gy_ini[10,22] = 30.9517475172273
        struct[0].Gy_ini[10,23] = 5.65456401516768
        struct[0].Gy_ini[10,24] = 9.00835072044485
        struct[0].Gy_ini[10,25] = -0.793238195499527
        struct[0].Gy_ini[10,26] = 9.21038227100566
        struct[0].Gy_ini[10,27] = -1.84896616921897
        struct[0].Gy_ini[10,82] = 1
        struct[0].Gy_ini[11,8] = 1.84896616921897
        struct[0].Gy_ini[11,9] = 9.21038227100566
        struct[0].Gy_ini[11,10] = 5.65456401516768
        struct[0].Gy_ini[11,11] = -30.9767475172273
        struct[0].Gy_ini[11,16] = -1.84896616921897
        struct[0].Gy_ini[11,17] = -9.21038227100566
        struct[0].Gy_ini[11,18] = -0.793238195499527
        struct[0].Gy_ini[11,19] = -9.00835072044485
        struct[0].Gy_ini[11,20] = -1.84896616921897
        struct[0].Gy_ini[11,21] = -9.21038227100566
        struct[0].Gy_ini[11,22] = -5.65456401516768
        struct[0].Gy_ini[11,23] = 30.9517475172273
        struct[0].Gy_ini[11,24] = 0.793238195499527
        struct[0].Gy_ini[11,25] = 9.00835072044485
        struct[0].Gy_ini[11,26] = 1.84896616921897
        struct[0].Gy_ini[11,27] = 9.21038227100566
        struct[0].Gy_ini[11,83] = 1
        struct[0].Gy_ini[12,12] = -157.977883096366
        struct[0].Gy_ini[12,30] = 157.977883096366
        struct[0].Gy_ini[12,84] = 1
        struct[0].Gy_ini[13,13] = -157.977883096366
        struct[0].Gy_ini[13,31] = 157.977883096366
        struct[0].Gy_ini[13,85] = 1
        struct[0].Gy_ini[14,14] = -157.977883096366
        struct[0].Gy_ini[14,36] = 157.977883096366
        struct[0].Gy_ini[14,86] = 1
        struct[0].Gy_ini[15,15] = -157.977883096366
        struct[0].Gy_ini[15,37] = 157.977883096366
        struct[0].Gy_ini[15,87] = 1
        struct[0].Gy_ini[16,0] = 10.5571769313180
        struct[0].Gy_ini[16,1] = 5.40657727682604
        struct[0].Gy_ini[16,2] = -3.96392229058202
        struct[0].Gy_ini[16,3] = -1.02713736253513
        struct[0].Gy_ini[16,4] = -2.49575997948692
        struct[0].Gy_ini[16,5] = -2.32849644809540
        struct[0].Gy_ini[16,6] = -3.96392229058202
        struct[0].Gy_ini[16,7] = -1.02713736253513
        struct[0].Gy_ini[16,8] = 30.9517475172273
        struct[0].Gy_ini[16,9] = 5.65456401516768
        struct[0].Gy_ini[16,10] = -9.21038227100566
        struct[0].Gy_ini[16,11] = 1.84896616921897
        struct[0].Gy_ini[16,16] = -41.5089244485453
        struct[0].Gy_ini[16,17] = -11.0611412919937
        struct[0].Gy_ini[16,18] = 13.1743045615877
        struct[0].Gy_ini[16,19] = -0.821828806683838
        struct[0].Gy_ini[16,20] = 11.5041106999318
        struct[0].Gy_ini[16,21] = 1.53525825259587
        struct[0].Gy_ini[16,22] = 13.1743045615877
        struct[0].Gy_ini[16,23] = -0.821828806683840
        struct[0].Gy_ini[16,24] = -9.21038227100566
        struct[0].Gy_ini[16,25] = 1.84896616921897
        struct[0].Gy_ini[16,26] = -9.00835072044485
        struct[0].Gy_ini[16,27] = 0.793238195499529
        struct[0].Gy_ini[16,97] = 1
        struct[0].Gy_ini[17,0] = -5.40657727682604
        struct[0].Gy_ini[17,1] = 10.5571769313180
        struct[0].Gy_ini[17,2] = 1.02713736253513
        struct[0].Gy_ini[17,3] = -3.96392229058202
        struct[0].Gy_ini[17,4] = 2.32849644809540
        struct[0].Gy_ini[17,5] = -2.49575997948692
        struct[0].Gy_ini[17,6] = 1.02713736253513
        struct[0].Gy_ini[17,7] = -3.96392229058202
        struct[0].Gy_ini[17,8] = -5.65456401516768
        struct[0].Gy_ini[17,9] = 30.9517475172273
        struct[0].Gy_ini[17,10] = -1.84896616921897
        struct[0].Gy_ini[17,11] = -9.21038227100566
        struct[0].Gy_ini[17,16] = 11.0611412919937
        struct[0].Gy_ini[17,17] = -41.5089244485453
        struct[0].Gy_ini[17,18] = 0.821828806683838
        struct[0].Gy_ini[17,19] = 13.1743045615877
        struct[0].Gy_ini[17,20] = -1.53525825259587
        struct[0].Gy_ini[17,21] = 11.5041106999318
        struct[0].Gy_ini[17,22] = 0.821828806683840
        struct[0].Gy_ini[17,23] = 13.1743045615877
        struct[0].Gy_ini[17,24] = -1.84896616921897
        struct[0].Gy_ini[17,25] = -9.21038227100566
        struct[0].Gy_ini[17,26] = -0.793238195499529
        struct[0].Gy_ini[17,27] = -9.00835072044485
        struct[0].Gy_ini[17,98] = 1
        struct[0].Gy_ini[18,0] = -3.96392229058202
        struct[0].Gy_ini[18,1] = -1.02713736253513
        struct[0].Gy_ini[18,2] = 10.5571769313180
        struct[0].Gy_ini[18,3] = 5.40657727682604
        struct[0].Gy_ini[18,4] = -3.96392229058202
        struct[0].Gy_ini[18,5] = -1.02713736253513
        struct[0].Gy_ini[18,6] = -2.49575997948692
        struct[0].Gy_ini[18,7] = -2.32849644809540
        struct[0].Gy_ini[18,8] = -9.21038227100566
        struct[0].Gy_ini[18,9] = 1.84896616921897
        struct[0].Gy_ini[18,10] = -9.00835072044485
        struct[0].Gy_ini[18,11] = 0.793238195499528
        struct[0].Gy_ini[18,16] = 13.1743045615877
        struct[0].Gy_ini[18,17] = -0.821828806683841
        struct[0].Gy_ini[18,18] = -41.5089244485453
        struct[0].Gy_ini[18,19] = -11.0611412919937
        struct[0].Gy_ini[18,20] = 13.1743045615877
        struct[0].Gy_ini[18,21] = -0.821828806683839
        struct[0].Gy_ini[18,22] = 11.5041106999318
        struct[0].Gy_ini[18,23] = 1.53525825259588
        struct[0].Gy_ini[18,24] = 30.9517475172273
        struct[0].Gy_ini[18,25] = 5.65456401516768
        struct[0].Gy_ini[18,26] = -9.21038227100566
        struct[0].Gy_ini[18,27] = 1.84896616921897
        struct[0].Gy_ini[18,99] = 1
        struct[0].Gy_ini[19,0] = 1.02713736253513
        struct[0].Gy_ini[19,1] = -3.96392229058202
        struct[0].Gy_ini[19,2] = -5.40657727682604
        struct[0].Gy_ini[19,3] = 10.5571769313180
        struct[0].Gy_ini[19,4] = 1.02713736253513
        struct[0].Gy_ini[19,5] = -3.96392229058202
        struct[0].Gy_ini[19,6] = 2.32849644809540
        struct[0].Gy_ini[19,7] = -2.49575997948692
        struct[0].Gy_ini[19,8] = -1.84896616921897
        struct[0].Gy_ini[19,9] = -9.21038227100566
        struct[0].Gy_ini[19,10] = -0.793238195499528
        struct[0].Gy_ini[19,11] = -9.00835072044485
        struct[0].Gy_ini[19,16] = 0.821828806683841
        struct[0].Gy_ini[19,17] = 13.1743045615877
        struct[0].Gy_ini[19,18] = 11.0611412919937
        struct[0].Gy_ini[19,19] = -41.5089244485453
        struct[0].Gy_ini[19,20] = 0.821828806683839
        struct[0].Gy_ini[19,21] = 13.1743045615877
        struct[0].Gy_ini[19,22] = -1.53525825259588
        struct[0].Gy_ini[19,23] = 11.5041106999318
        struct[0].Gy_ini[19,24] = -5.65456401516768
        struct[0].Gy_ini[19,25] = 30.9517475172273
        struct[0].Gy_ini[19,26] = -1.84896616921897
        struct[0].Gy_ini[19,27] = -9.21038227100566
        struct[0].Gy_ini[19,100] = 1
        struct[0].Gy_ini[20,0] = -2.49575997948692
        struct[0].Gy_ini[20,1] = -2.32849644809540
        struct[0].Gy_ini[20,2] = -3.96392229058202
        struct[0].Gy_ini[20,3] = -1.02713736253513
        struct[0].Gy_ini[20,4] = 10.5571769313180
        struct[0].Gy_ini[20,5] = 5.40657727682604
        struct[0].Gy_ini[20,6] = -3.96392229058202
        struct[0].Gy_ini[20,7] = -1.02713736253513
        struct[0].Gy_ini[20,8] = -9.00835072044484
        struct[0].Gy_ini[20,9] = 0.793238195499527
        struct[0].Gy_ini[20,10] = -9.21038227100566
        struct[0].Gy_ini[20,11] = 1.84896616921897
        struct[0].Gy_ini[20,16] = 11.5041106999318
        struct[0].Gy_ini[20,17] = 1.53525825259588
        struct[0].Gy_ini[20,18] = 13.1743045615877
        struct[0].Gy_ini[20,19] = -0.821828806683840
        struct[0].Gy_ini[20,20] = -41.5089244485453
        struct[0].Gy_ini[20,21] = -11.0611412919937
        struct[0].Gy_ini[20,22] = 13.1743045615877
        struct[0].Gy_ini[20,23] = -0.821828806683838
        struct[0].Gy_ini[20,24] = -9.21038227100566
        struct[0].Gy_ini[20,25] = 1.84896616921897
        struct[0].Gy_ini[20,26] = 30.9517475172273
        struct[0].Gy_ini[20,27] = 5.65456401516768
        struct[0].Gy_ini[20,101] = 1
        struct[0].Gy_ini[21,0] = 2.32849644809540
        struct[0].Gy_ini[21,1] = -2.49575997948692
        struct[0].Gy_ini[21,2] = 1.02713736253513
        struct[0].Gy_ini[21,3] = -3.96392229058202
        struct[0].Gy_ini[21,4] = -5.40657727682604
        struct[0].Gy_ini[21,5] = 10.5571769313180
        struct[0].Gy_ini[21,6] = 1.02713736253513
        struct[0].Gy_ini[21,7] = -3.96392229058202
        struct[0].Gy_ini[21,8] = -0.793238195499527
        struct[0].Gy_ini[21,9] = -9.00835072044484
        struct[0].Gy_ini[21,10] = -1.84896616921897
        struct[0].Gy_ini[21,11] = -9.21038227100566
        struct[0].Gy_ini[21,16] = -1.53525825259588
        struct[0].Gy_ini[21,17] = 11.5041106999318
        struct[0].Gy_ini[21,18] = 0.821828806683840
        struct[0].Gy_ini[21,19] = 13.1743045615877
        struct[0].Gy_ini[21,20] = 11.0611412919937
        struct[0].Gy_ini[21,21] = -41.5089244485453
        struct[0].Gy_ini[21,22] = 0.821828806683838
        struct[0].Gy_ini[21,23] = 13.1743045615877
        struct[0].Gy_ini[21,24] = -1.84896616921897
        struct[0].Gy_ini[21,25] = -9.21038227100566
        struct[0].Gy_ini[21,26] = -5.65456401516768
        struct[0].Gy_ini[21,27] = 30.9517475172273
        struct[0].Gy_ini[21,102] = 1
        struct[0].Gy_ini[22,0] = -3.96392229058202
        struct[0].Gy_ini[22,1] = -1.02713736253513
        struct[0].Gy_ini[22,2] = -2.49575997948692
        struct[0].Gy_ini[22,3] = -2.32849644809540
        struct[0].Gy_ini[22,4] = -3.96392229058202
        struct[0].Gy_ini[22,5] = -1.02713736253513
        struct[0].Gy_ini[22,6] = 10.5571769313180
        struct[0].Gy_ini[22,7] = 5.40657727682604
        struct[0].Gy_ini[22,8] = -9.21038227100566
        struct[0].Gy_ini[22,9] = 1.84896616921897
        struct[0].Gy_ini[22,10] = 30.9517475172273
        struct[0].Gy_ini[22,11] = 5.65456401516768
        struct[0].Gy_ini[22,16] = 13.1743045615877
        struct[0].Gy_ini[22,17] = -0.821828806683840
        struct[0].Gy_ini[22,18] = 11.5041106999318
        struct[0].Gy_ini[22,19] = 1.53525825259588
        struct[0].Gy_ini[22,20] = 13.1743045615877
        struct[0].Gy_ini[22,21] = -0.821828806683837
        struct[0].Gy_ini[22,22] = -41.5339244485453
        struct[0].Gy_ini[22,23] = -11.0611412919937
        struct[0].Gy_ini[22,24] = -9.00835072044485
        struct[0].Gy_ini[22,25] = 0.793238195499527
        struct[0].Gy_ini[22,26] = -9.21038227100566
        struct[0].Gy_ini[22,27] = 1.84896616921897
        struct[0].Gy_ini[23,0] = 1.02713736253513
        struct[0].Gy_ini[23,1] = -3.96392229058202
        struct[0].Gy_ini[23,2] = 2.32849644809540
        struct[0].Gy_ini[23,3] = -2.49575997948692
        struct[0].Gy_ini[23,4] = 1.02713736253513
        struct[0].Gy_ini[23,5] = -3.96392229058202
        struct[0].Gy_ini[23,6] = -5.40657727682604
        struct[0].Gy_ini[23,7] = 10.5571769313180
        struct[0].Gy_ini[23,8] = -1.84896616921897
        struct[0].Gy_ini[23,9] = -9.21038227100566
        struct[0].Gy_ini[23,10] = -5.65456401516768
        struct[0].Gy_ini[23,11] = 30.9517475172273
        struct[0].Gy_ini[23,16] = 0.821828806683840
        struct[0].Gy_ini[23,17] = 13.1743045615877
        struct[0].Gy_ini[23,18] = -1.53525825259588
        struct[0].Gy_ini[23,19] = 11.5041106999318
        struct[0].Gy_ini[23,20] = 0.821828806683837
        struct[0].Gy_ini[23,21] = 13.1743045615877
        struct[0].Gy_ini[23,22] = 11.0611412919937
        struct[0].Gy_ini[23,23] = -41.5339244485453
        struct[0].Gy_ini[23,24] = -0.793238195499527
        struct[0].Gy_ini[23,25] = -9.00835072044485
        struct[0].Gy_ini[23,26] = -1.84896616921897
        struct[0].Gy_ini[23,27] = -9.21038227100566
        struct[0].Gy_ini[24,8] = 9.21038227100566
        struct[0].Gy_ini[24,9] = -1.84896616921897
        struct[0].Gy_ini[24,10] = 9.00835072044485
        struct[0].Gy_ini[24,11] = -0.793238195499528
        struct[0].Gy_ini[24,16] = -9.21038227100566
        struct[0].Gy_ini[24,17] = 1.84896616921897
        struct[0].Gy_ini[24,18] = 30.9517475172273
        struct[0].Gy_ini[24,19] = 5.65456401516768
        struct[0].Gy_ini[24,20] = -9.21038227100566
        struct[0].Gy_ini[24,21] = 1.84896616921897
        struct[0].Gy_ini[24,22] = -9.00835072044485
        struct[0].Gy_ini[24,23] = 0.793238195499528
        struct[0].Gy_ini[24,24] = -30.9517475172273
        struct[0].Gy_ini[24,25] = -5.65456401516768
        struct[0].Gy_ini[24,26] = 9.21038227100566
        struct[0].Gy_ini[24,27] = -1.84896616921897
        struct[0].Gy_ini[25,8] = 1.84896616921897
        struct[0].Gy_ini[25,9] = 9.21038227100566
        struct[0].Gy_ini[25,10] = 0.793238195499528
        struct[0].Gy_ini[25,11] = 9.00835072044485
        struct[0].Gy_ini[25,16] = -1.84896616921897
        struct[0].Gy_ini[25,17] = -9.21038227100566
        struct[0].Gy_ini[25,18] = -5.65456401516768
        struct[0].Gy_ini[25,19] = 30.9517475172273
        struct[0].Gy_ini[25,20] = -1.84896616921897
        struct[0].Gy_ini[25,21] = -9.21038227100566
        struct[0].Gy_ini[25,22] = -0.793238195499528
        struct[0].Gy_ini[25,23] = -9.00835072044485
        struct[0].Gy_ini[25,24] = 5.65456401516768
        struct[0].Gy_ini[25,25] = -30.9517475172273
        struct[0].Gy_ini[25,26] = 1.84896616921897
        struct[0].Gy_ini[25,27] = 9.21038227100566
        struct[0].Gy_ini[26,8] = 9.00835072044484
        struct[0].Gy_ini[26,9] = -0.793238195499527
        struct[0].Gy_ini[26,10] = 9.21038227100566
        struct[0].Gy_ini[26,11] = -1.84896616921897
        struct[0].Gy_ini[26,16] = -9.00835072044484
        struct[0].Gy_ini[26,17] = 0.793238195499527
        struct[0].Gy_ini[26,18] = -9.21038227100566
        struct[0].Gy_ini[26,19] = 1.84896616921897
        struct[0].Gy_ini[26,20] = 30.9517475172273
        struct[0].Gy_ini[26,21] = 5.65456401516768
        struct[0].Gy_ini[26,22] = -9.21038227100566
        struct[0].Gy_ini[26,23] = 1.84896616921897
        struct[0].Gy_ini[26,24] = 9.21038227100566
        struct[0].Gy_ini[26,25] = -1.84896616921897
        struct[0].Gy_ini[26,26] = -30.9517475172273
        struct[0].Gy_ini[26,27] = -5.65456401516768
        struct[0].Gy_ini[27,8] = 0.793238195499527
        struct[0].Gy_ini[27,9] = 9.00835072044484
        struct[0].Gy_ini[27,10] = 1.84896616921897
        struct[0].Gy_ini[27,11] = 9.21038227100566
        struct[0].Gy_ini[27,16] = -0.793238195499527
        struct[0].Gy_ini[27,17] = -9.00835072044484
        struct[0].Gy_ini[27,18] = -1.84896616921897
        struct[0].Gy_ini[27,19] = -9.21038227100566
        struct[0].Gy_ini[27,20] = -5.65456401516768
        struct[0].Gy_ini[27,21] = 30.9517475172273
        struct[0].Gy_ini[27,22] = -1.84896616921897
        struct[0].Gy_ini[27,23] = -9.21038227100566
        struct[0].Gy_ini[27,24] = 1.84896616921897
        struct[0].Gy_ini[27,25] = 9.21038227100566
        struct[0].Gy_ini[27,26] = 5.65456401516768
        struct[0].Gy_ini[27,27] = -30.9517475172273
        struct[0].Gy_ini[28,28] = -1067.70480704130
        struct[0].Gy_ini[28,36] = 67.7048070412999
        struct[0].Gy_ini[29,29] = -1067.70480704130
        struct[0].Gy_ini[29,37] = 67.7048070412999
        struct[0].Gy_ini[30,12] = 157.977883096366
        struct[0].Gy_ini[30,30] = -225.682690137666
        struct[0].Gy_ini[30,103] = 1
        struct[0].Gy_ini[31,13] = 157.977883096366
        struct[0].Gy_ini[31,31] = -225.682690137666
        struct[0].Gy_ini[32,32] = -225.682690137666
        struct[0].Gy_ini[32,38] = 157.977883096366
        struct[0].Gy_ini[33,33] = -225.682690137666
        struct[0].Gy_ini[33,39] = 157.977883096366
        struct[0].Gy_ini[34,34] = -225.682690137666
        struct[0].Gy_ini[34,40] = 157.977883096366
        struct[0].Gy_ini[35,35] = -225.682690137666
        struct[0].Gy_ini[35,41] = 157.977883096366
        struct[0].Gy_ini[36,14] = 157.977883096366
        struct[0].Gy_ini[36,28] = 67.7048070412999
        struct[0].Gy_ini[36,36] = -225.682690137666
        struct[0].Gy_ini[36,104] = 1
        struct[0].Gy_ini[37,15] = 157.977883096366
        struct[0].Gy_ini[37,29] = 67.7048070412999
        struct[0].Gy_ini[37,37] = -225.682690137666
        struct[0].Gy_ini[38,32] = 157.977883096366
        struct[0].Gy_ini[38,38] = -157.977883096366
        struct[0].Gy_ini[39,33] = 157.977883096366
        struct[0].Gy_ini[39,39] = -157.977883096366
        struct[0].Gy_ini[40,34] = 157.977883096366
        struct[0].Gy_ini[40,40] = -157.977883096366
        struct[0].Gy_ini[41,35] = 157.977883096366
        struct[0].Gy_ini[41,41] = -157.977883096366
        struct[0].Gy_ini[42,0] = -0.212261128378539
        struct[0].Gy_ini[42,1] = -0.849044513514155
        struct[0].Gy_ini[42,2] = 0.212261128378539
        struct[0].Gy_ini[42,3] = 0.849044513514155
        struct[0].Gy_ini[42,42] = -1
        struct[0].Gy_ini[43,0] = 0.849044513514155
        struct[0].Gy_ini[43,1] = -0.212261128378539
        struct[0].Gy_ini[43,2] = -0.849044513514155
        struct[0].Gy_ini[43,3] = 0.212261128378539
        struct[0].Gy_ini[43,43] = -1
        struct[0].Gy_ini[44,2] = -0.212261128378539
        struct[0].Gy_ini[44,3] = -0.849044513514155
        struct[0].Gy_ini[44,4] = 0.212261128378539
        struct[0].Gy_ini[44,5] = 0.849044513514155
        struct[0].Gy_ini[44,44] = -1
        struct[0].Gy_ini[45,2] = 0.849044513514155
        struct[0].Gy_ini[45,3] = -0.212261128378539
        struct[0].Gy_ini[45,4] = -0.849044513514155
        struct[0].Gy_ini[45,5] = 0.212261128378539
        struct[0].Gy_ini[45,45] = -1
        struct[0].Gy_ini[46,0] = 0.212261128378539
        struct[0].Gy_ini[46,1] = 0.849044513514155
        struct[0].Gy_ini[46,4] = -0.212261128378539
        struct[0].Gy_ini[46,5] = -0.849044513514155
        struct[0].Gy_ini[46,46] = -1
        struct[0].Gy_ini[47,0] = -0.849044513514155
        struct[0].Gy_ini[47,1] = 0.212261128378539
        struct[0].Gy_ini[47,4] = 0.849044513514155
        struct[0].Gy_ini[47,5] = -0.212261128378539
        struct[0].Gy_ini[47,47] = -1
        struct[0].Gy_ini[48,0] = 10.5571769313180
        struct[0].Gy_ini[48,1] = 5.40657727682604
        struct[0].Gy_ini[48,2] = -3.96392229058202
        struct[0].Gy_ini[48,3] = -1.02713736253513
        struct[0].Gy_ini[48,4] = -2.49575997948692
        struct[0].Gy_ini[48,5] = -2.32849644809540
        struct[0].Gy_ini[48,6] = -3.96392229058202
        struct[0].Gy_ini[48,7] = -1.02713736253513
        struct[0].Gy_ini[48,16] = -10.5571769313180
        struct[0].Gy_ini[48,17] = -5.40657727682604
        struct[0].Gy_ini[48,18] = 3.96392229058202
        struct[0].Gy_ini[48,19] = 1.02713736253513
        struct[0].Gy_ini[48,20] = 2.49575997948692
        struct[0].Gy_ini[48,21] = 2.32849644809540
        struct[0].Gy_ini[48,22] = 3.96392229058202
        struct[0].Gy_ini[48,23] = 1.02713736253513
        struct[0].Gy_ini[48,48] = -1
        struct[0].Gy_ini[49,0] = -5.40657727682604
        struct[0].Gy_ini[49,1] = 10.5571769313180
        struct[0].Gy_ini[49,2] = 1.02713736253513
        struct[0].Gy_ini[49,3] = -3.96392229058202
        struct[0].Gy_ini[49,4] = 2.32849644809540
        struct[0].Gy_ini[49,5] = -2.49575997948692
        struct[0].Gy_ini[49,6] = 1.02713736253513
        struct[0].Gy_ini[49,7] = -3.96392229058202
        struct[0].Gy_ini[49,16] = 5.40657727682604
        struct[0].Gy_ini[49,17] = -10.5571769313180
        struct[0].Gy_ini[49,18] = -1.02713736253513
        struct[0].Gy_ini[49,19] = 3.96392229058202
        struct[0].Gy_ini[49,20] = -2.32849644809540
        struct[0].Gy_ini[49,21] = 2.49575997948692
        struct[0].Gy_ini[49,22] = -1.02713736253513
        struct[0].Gy_ini[49,23] = 3.96392229058202
        struct[0].Gy_ini[49,49] = -1
        struct[0].Gy_ini[50,0] = -3.96392229058202
        struct[0].Gy_ini[50,1] = -1.02713736253513
        struct[0].Gy_ini[50,2] = 10.5571769313180
        struct[0].Gy_ini[50,3] = 5.40657727682604
        struct[0].Gy_ini[50,4] = -3.96392229058202
        struct[0].Gy_ini[50,5] = -1.02713736253513
        struct[0].Gy_ini[50,6] = -2.49575997948692
        struct[0].Gy_ini[50,7] = -2.32849644809540
        struct[0].Gy_ini[50,16] = 3.96392229058202
        struct[0].Gy_ini[50,17] = 1.02713736253513
        struct[0].Gy_ini[50,18] = -10.5571769313180
        struct[0].Gy_ini[50,19] = -5.40657727682604
        struct[0].Gy_ini[50,20] = 3.96392229058202
        struct[0].Gy_ini[50,21] = 1.02713736253513
        struct[0].Gy_ini[50,22] = 2.49575997948692
        struct[0].Gy_ini[50,23] = 2.32849644809540
        struct[0].Gy_ini[50,50] = -1
        struct[0].Gy_ini[51,0] = 1.02713736253513
        struct[0].Gy_ini[51,1] = -3.96392229058202
        struct[0].Gy_ini[51,2] = -5.40657727682604
        struct[0].Gy_ini[51,3] = 10.5571769313180
        struct[0].Gy_ini[51,4] = 1.02713736253513
        struct[0].Gy_ini[51,5] = -3.96392229058202
        struct[0].Gy_ini[51,6] = 2.32849644809540
        struct[0].Gy_ini[51,7] = -2.49575997948692
        struct[0].Gy_ini[51,16] = -1.02713736253513
        struct[0].Gy_ini[51,17] = 3.96392229058202
        struct[0].Gy_ini[51,18] = 5.40657727682604
        struct[0].Gy_ini[51,19] = -10.5571769313180
        struct[0].Gy_ini[51,20] = -1.02713736253513
        struct[0].Gy_ini[51,21] = 3.96392229058202
        struct[0].Gy_ini[51,22] = -2.32849644809540
        struct[0].Gy_ini[51,23] = 2.49575997948692
        struct[0].Gy_ini[51,51] = -1
        struct[0].Gy_ini[52,0] = -2.49575997948692
        struct[0].Gy_ini[52,1] = -2.32849644809540
        struct[0].Gy_ini[52,2] = -3.96392229058202
        struct[0].Gy_ini[52,3] = -1.02713736253513
        struct[0].Gy_ini[52,4] = 10.5571769313180
        struct[0].Gy_ini[52,5] = 5.40657727682604
        struct[0].Gy_ini[52,6] = -3.96392229058202
        struct[0].Gy_ini[52,7] = -1.02713736253513
        struct[0].Gy_ini[52,16] = 2.49575997948692
        struct[0].Gy_ini[52,17] = 2.32849644809540
        struct[0].Gy_ini[52,18] = 3.96392229058202
        struct[0].Gy_ini[52,19] = 1.02713736253513
        struct[0].Gy_ini[52,20] = -10.5571769313180
        struct[0].Gy_ini[52,21] = -5.40657727682604
        struct[0].Gy_ini[52,22] = 3.96392229058202
        struct[0].Gy_ini[52,23] = 1.02713736253513
        struct[0].Gy_ini[52,52] = -1
        struct[0].Gy_ini[53,0] = 2.32849644809540
        struct[0].Gy_ini[53,1] = -2.49575997948692
        struct[0].Gy_ini[53,2] = 1.02713736253513
        struct[0].Gy_ini[53,3] = -3.96392229058202
        struct[0].Gy_ini[53,4] = -5.40657727682604
        struct[0].Gy_ini[53,5] = 10.5571769313180
        struct[0].Gy_ini[53,6] = 1.02713736253513
        struct[0].Gy_ini[53,7] = -3.96392229058202
        struct[0].Gy_ini[53,16] = -2.32849644809540
        struct[0].Gy_ini[53,17] = 2.49575997948692
        struct[0].Gy_ini[53,18] = -1.02713736253513
        struct[0].Gy_ini[53,19] = 3.96392229058202
        struct[0].Gy_ini[53,20] = 5.40657727682604
        struct[0].Gy_ini[53,21] = -10.5571769313180
        struct[0].Gy_ini[53,22] = -1.02713736253513
        struct[0].Gy_ini[53,23] = 3.96392229058202
        struct[0].Gy_ini[53,53] = -1
        struct[0].Gy_ini[54,48] = 1
        struct[0].Gy_ini[54,50] = 1
        struct[0].Gy_ini[54,52] = 1
        struct[0].Gy_ini[54,54] = -1
        struct[0].Gy_ini[55,49] = 1
        struct[0].Gy_ini[55,51] = 1
        struct[0].Gy_ini[55,53] = 1
        struct[0].Gy_ini[55,55] = -1
        struct[0].Gy_ini[56,30] = -67.7048070412999
        struct[0].Gy_ini[56,56] = -1
        struct[0].Gy_ini[57,31] = -67.7048070412999
        struct[0].Gy_ini[57,57] = -1
        struct[0].Gy_ini[58,32] = -67.7048070412999
        struct[0].Gy_ini[58,58] = -1
        struct[0].Gy_ini[59,33] = -67.7048070412999
        struct[0].Gy_ini[59,59] = -1
        struct[0].Gy_ini[60,34] = -67.7048070412999
        struct[0].Gy_ini[60,60] = -1
        struct[0].Gy_ini[61,35] = -67.7048070412999
        struct[0].Gy_ini[61,61] = -1
        struct[0].Gy_ini[62,56] = 1
        struct[0].Gy_ini[62,58] = 1
        struct[0].Gy_ini[62,60] = 1
        struct[0].Gy_ini[62,62] = -1
        struct[0].Gy_ini[63,57] = 1
        struct[0].Gy_ini[63,59] = 1
        struct[0].Gy_ini[63,61] = 1
        struct[0].Gy_ini[63,63] = -1
        struct[0].Gy_ini[64,12] = -157.977883096366
        struct[0].Gy_ini[64,30] = 157.977883096366
        struct[0].Gy_ini[64,64] = -1
        struct[0].Gy_ini[65,13] = -157.977883096366
        struct[0].Gy_ini[65,31] = 157.977883096366
        struct[0].Gy_ini[65,65] = -1
        struct[0].Gy_ini[66,32] = 157.977883096366
        struct[0].Gy_ini[66,38] = -157.977883096366
        struct[0].Gy_ini[66,66] = -1
        struct[0].Gy_ini[67,33] = 157.977883096366
        struct[0].Gy_ini[67,39] = -157.977883096366
        struct[0].Gy_ini[67,67] = -1
        struct[0].Gy_ini[68,34] = 157.977883096366
        struct[0].Gy_ini[68,40] = -157.977883096366
        struct[0].Gy_ini[68,68] = -1
        struct[0].Gy_ini[69,35] = 157.977883096366
        struct[0].Gy_ini[69,41] = -157.977883096366
        struct[0].Gy_ini[69,69] = -1
        struct[0].Gy_ini[70,64] = 1
        struct[0].Gy_ini[70,66] = 1
        struct[0].Gy_ini[70,68] = 1
        struct[0].Gy_ini[70,70] = -1
        struct[0].Gy_ini[71,65] = 1
        struct[0].Gy_ini[71,67] = 1
        struct[0].Gy_ini[71,69] = 1
        struct[0].Gy_ini[71,71] = -1
        struct[0].Gy_ini[72,0] = i_load_R1_a_r
        struct[0].Gy_ini[72,1] = i_load_R1_a_i
        struct[0].Gy_ini[72,6] = -i_load_R1_a_r
        struct[0].Gy_ini[72,7] = -i_load_R1_a_i
        struct[0].Gy_ini[72,72] = v_R1_a_r - v_R1_n_r
        struct[0].Gy_ini[72,73] = v_R1_a_i - v_R1_n_i
        struct[0].Gy_ini[73,2] = i_load_R1_b_r
        struct[0].Gy_ini[73,3] = i_load_R1_b_i
        struct[0].Gy_ini[73,6] = -i_load_R1_b_r
        struct[0].Gy_ini[73,7] = -i_load_R1_b_i
        struct[0].Gy_ini[73,74] = v_R1_b_r - v_R1_n_r
        struct[0].Gy_ini[73,75] = v_R1_b_i - v_R1_n_i
        struct[0].Gy_ini[74,4] = i_load_R1_c_r
        struct[0].Gy_ini[74,5] = i_load_R1_c_i
        struct[0].Gy_ini[74,6] = -i_load_R1_c_r
        struct[0].Gy_ini[74,7] = -i_load_R1_c_i
        struct[0].Gy_ini[74,76] = v_R1_c_r - v_R1_n_r
        struct[0].Gy_ini[74,77] = v_R1_c_i - v_R1_n_i
        struct[0].Gy_ini[75,0] = -i_load_R1_a_i
        struct[0].Gy_ini[75,1] = i_load_R1_a_r
        struct[0].Gy_ini[75,6] = i_load_R1_a_i
        struct[0].Gy_ini[75,7] = -i_load_R1_a_r
        struct[0].Gy_ini[75,72] = v_R1_a_i - v_R1_n_i
        struct[0].Gy_ini[75,73] = -v_R1_a_r + v_R1_n_r
        struct[0].Gy_ini[76,2] = -i_load_R1_b_i
        struct[0].Gy_ini[76,3] = i_load_R1_b_r
        struct[0].Gy_ini[76,6] = i_load_R1_b_i
        struct[0].Gy_ini[76,7] = -i_load_R1_b_r
        struct[0].Gy_ini[76,74] = v_R1_b_i - v_R1_n_i
        struct[0].Gy_ini[76,75] = -v_R1_b_r + v_R1_n_r
        struct[0].Gy_ini[77,4] = -i_load_R1_c_i
        struct[0].Gy_ini[77,5] = i_load_R1_c_r
        struct[0].Gy_ini[77,6] = i_load_R1_c_i
        struct[0].Gy_ini[77,7] = -i_load_R1_c_r
        struct[0].Gy_ini[77,76] = v_R1_c_i - v_R1_n_i
        struct[0].Gy_ini[77,77] = -v_R1_c_r + v_R1_n_r
        struct[0].Gy_ini[78,72] = 1
        struct[0].Gy_ini[78,74] = 1
        struct[0].Gy_ini[78,76] = 1
        struct[0].Gy_ini[78,78] = 1
        struct[0].Gy_ini[79,73] = 1
        struct[0].Gy_ini[79,75] = 1
        struct[0].Gy_ini[79,77] = 1
        struct[0].Gy_ini[79,79] = 1
        struct[0].Gy_ini[80,8] = i_load_R18_a_r
        struct[0].Gy_ini[80,9] = 1.0*i_load_R18_a_i
        struct[0].Gy_ini[80,10] = -i_load_R18_a_r
        struct[0].Gy_ini[80,11] = -1.0*i_load_R18_a_i
        struct[0].Gy_ini[80,80] = v_R18_a_r - v_R18_n_r
        struct[0].Gy_ini[80,81] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy_ini[81,8] = -1.0*i_load_R18_a_i
        struct[0].Gy_ini[81,9] = 1.0*i_load_R18_a_r
        struct[0].Gy_ini[81,10] = 1.0*i_load_R18_a_i
        struct[0].Gy_ini[81,11] = -1.0*i_load_R18_a_r
        struct[0].Gy_ini[81,80] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy_ini[81,81] = -1.0*v_R18_a_r + 1.0*v_R18_n_r
        struct[0].Gy_ini[82,80] = 1
        struct[0].Gy_ini[82,82] = 1
        struct[0].Gy_ini[83,81] = 1.00000000000000
        struct[0].Gy_ini[83,83] = 1.00000000000000
        struct[0].Gy_ini[84,12] = i_load_D18_a_r
        struct[0].Gy_ini[84,13] = 1.0*i_load_D18_a_i
        struct[0].Gy_ini[84,14] = -i_load_D18_a_r
        struct[0].Gy_ini[84,15] = -1.0*i_load_D18_a_i
        struct[0].Gy_ini[84,84] = v_D18_a_r - v_D18_n_r
        struct[0].Gy_ini[84,85] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy_ini[85,12] = -1.0*i_load_D18_a_i
        struct[0].Gy_ini[85,13] = 1.0*i_load_D18_a_r
        struct[0].Gy_ini[85,14] = 1.0*i_load_D18_a_i
        struct[0].Gy_ini[85,15] = -1.0*i_load_D18_a_r
        struct[0].Gy_ini[85,84] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy_ini[85,85] = -1.0*v_D18_a_r + 1.0*v_D18_n_r
        struct[0].Gy_ini[86,84] = 1
        struct[0].Gy_ini[86,86] = 1
        struct[0].Gy_ini[87,85] = 1.00000000000000
        struct[0].Gy_ini[87,87] = 1.00000000000000
        struct[0].Gy_ini[88,0] = i_vsc_R1_a_r
        struct[0].Gy_ini[88,1] = 1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[88,6] = -i_vsc_R1_a_r
        struct[0].Gy_ini[88,7] = -1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[88,88] = v_R1_a_r - v_R1_n_r
        struct[0].Gy_ini[88,89] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[88,94] = -1/3
        struct[0].Gy_ini[89,0] = -1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[89,1] = 1.0*i_vsc_R1_a_r
        struct[0].Gy_ini[89,6] = 1.0*i_vsc_R1_a_i
        struct[0].Gy_ini[89,7] = -1.0*i_vsc_R1_a_r
        struct[0].Gy_ini[89,88] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[89,89] = -1.0*v_R1_a_r + 1.0*v_R1_n_r
        struct[0].Gy_ini[90,2] = i_vsc_R1_b_r
        struct[0].Gy_ini[90,3] = 1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[90,6] = -i_vsc_R1_b_r
        struct[0].Gy_ini[90,7] = -1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[90,90] = v_R1_b_r - v_R1_n_r
        struct[0].Gy_ini[90,91] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[90,94] = -1/3
        struct[0].Gy_ini[91,2] = -1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[91,3] = 1.0*i_vsc_R1_b_r
        struct[0].Gy_ini[91,6] = 1.0*i_vsc_R1_b_i
        struct[0].Gy_ini[91,7] = -1.0*i_vsc_R1_b_r
        struct[0].Gy_ini[91,90] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[91,91] = -1.0*v_R1_b_r + 1.0*v_R1_n_r
        struct[0].Gy_ini[92,4] = i_vsc_R1_c_r
        struct[0].Gy_ini[92,5] = 1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[92,6] = -i_vsc_R1_c_r
        struct[0].Gy_ini[92,7] = -1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[92,92] = v_R1_c_r - v_R1_n_r
        struct[0].Gy_ini[92,93] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[92,94] = -1/3
        struct[0].Gy_ini[93,4] = -1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[93,5] = 1.0*i_vsc_R1_c_r
        struct[0].Gy_ini[93,6] = 1.0*i_vsc_R1_c_i
        struct[0].Gy_ini[93,7] = -1.0*i_vsc_R1_c_r
        struct[0].Gy_ini[93,92] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy_ini[93,93] = -1.0*v_R1_c_r + 1.0*v_R1_n_r
        struct[0].Gy_ini[94,94] = 1
        struct[0].Gy_ini[94,95] = 1
        struct[0].Gy_ini[94,96] = Piecewise(np.array([(-1, p_D1 < 0), (1, True)]))
        struct[0].Gy_ini[95,56] = v_D1_a_r
        struct[0].Gy_ini[95,62] = v_D1_n_r
        struct[0].Gy_ini[95,95] = -1
        struct[0].Gy_ini[96,88] = -b_R1*i_vsc_R1_a_r/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_r
        struct[0].Gy_ini[96,89] = -b_R1*i_vsc_R1_a_i/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_i
        struct[0].Gy_ini[96,96] = 1
        struct[0].Gy_ini[97,16] = i_vsc_R10_a_r
        struct[0].Gy_ini[97,17] = 1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[97,22] = -i_vsc_R10_a_r
        struct[0].Gy_ini[97,23] = -1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[97,97] = v_R10_a_r - v_R10_n_r
        struct[0].Gy_ini[97,98] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[98,16] = -1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[98,17] = 1.0*i_vsc_R10_a_r
        struct[0].Gy_ini[98,22] = 1.0*i_vsc_R10_a_i
        struct[0].Gy_ini[98,23] = -1.0*i_vsc_R10_a_r
        struct[0].Gy_ini[98,97] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[98,98] = -1.0*v_R10_a_r + 1.0*v_R10_n_r
        struct[0].Gy_ini[99,18] = i_vsc_R10_b_r
        struct[0].Gy_ini[99,19] = 1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[99,22] = -i_vsc_R10_b_r
        struct[0].Gy_ini[99,23] = -1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[99,99] = v_R10_b_r - v_R10_n_r
        struct[0].Gy_ini[99,100] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[100,18] = -1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[100,19] = 1.0*i_vsc_R10_b_r
        struct[0].Gy_ini[100,22] = 1.0*i_vsc_R10_b_i
        struct[0].Gy_ini[100,23] = -1.0*i_vsc_R10_b_r
        struct[0].Gy_ini[100,99] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[100,100] = -1.0*v_R10_b_r + 1.0*v_R10_n_r
        struct[0].Gy_ini[101,20] = i_vsc_R10_c_r
        struct[0].Gy_ini[101,21] = 1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[101,22] = -i_vsc_R10_c_r
        struct[0].Gy_ini[101,23] = -1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[101,101] = v_R10_c_r - v_R10_n_r
        struct[0].Gy_ini[101,102] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[102,20] = -1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[102,21] = 1.0*i_vsc_R10_c_r
        struct[0].Gy_ini[102,22] = 1.0*i_vsc_R10_c_i
        struct[0].Gy_ini[102,23] = -1.0*i_vsc_R10_c_r
        struct[0].Gy_ini[102,101] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy_ini[102,102] = -1.0*v_R10_c_r + 1.0*v_R10_n_r
        struct[0].Gy_ini[103,30] = -p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[103,36] = p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[103,103] = 1
        struct[0].Gy_ini[103,105] = 1/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].Gy_ini[104,30] = p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[104,36] = -p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy_ini[104,104] = 1
        struct[0].Gy_ini[104,105] = 1/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].Gy_ini[105,105] = 1
        struct[0].Gy_ini[105,106] = -Piecewise(np.array([(-1, p_D10 < 0), (1, True)]))
        struct[0].Gy_ini[106,97] = -b_R10*i_vsc_R10_a_r/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_r
        struct[0].Gy_ini[106,98] = -b_R10*i_vsc_R10_a_i/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_i
        struct[0].Gy_ini[106,106] = 1



def run_nn(t,struct,mode):

    # Parameters:
    a_R1 = struct[0].a_R1
    b_R1 = struct[0].b_R1
    c_R1 = struct[0].c_R1
    a_R10 = struct[0].a_R10
    b_R10 = struct[0].b_R10
    c_R10 = struct[0].c_R10
    coef_a_R10 = struct[0].coef_a_R10
    coef_b_R10 = struct[0].coef_b_R10
    coef_c_R10 = struct[0].coef_c_R10
    
    # Inputs:
    v_R0_a_r = struct[0].v_R0_a_r
    v_R0_a_i = struct[0].v_R0_a_i
    v_R0_b_r = struct[0].v_R0_b_r
    v_R0_b_i = struct[0].v_R0_b_i
    v_R0_c_r = struct[0].v_R0_c_r
    v_R0_c_i = struct[0].v_R0_c_i
    v_D1_a_r = struct[0].v_D1_a_r
    v_D1_a_i = struct[0].v_D1_a_i
    v_D1_b_r = struct[0].v_D1_b_r
    v_D1_b_i = struct[0].v_D1_b_i
    v_D1_c_r = struct[0].v_D1_c_r
    v_D1_c_i = struct[0].v_D1_c_i
    i_R1_n_r = struct[0].i_R1_n_r
    i_R1_n_i = struct[0].i_R1_n_i
    i_R10_a_r = struct[0].i_R10_a_r
    i_R10_a_i = struct[0].i_R10_a_i
    i_R10_b_r = struct[0].i_R10_b_r
    i_R10_b_i = struct[0].i_R10_b_i
    i_R10_c_r = struct[0].i_R10_c_r
    i_R10_c_i = struct[0].i_R10_c_i
    i_R10_n_r = struct[0].i_R10_n_r
    i_R10_n_i = struct[0].i_R10_n_i
    i_R18_b_r = struct[0].i_R18_b_r
    i_R18_b_i = struct[0].i_R18_b_i
    i_R18_c_r = struct[0].i_R18_c_r
    i_R18_c_i = struct[0].i_R18_c_i
    i_D1_n_r = struct[0].i_D1_n_r
    i_D1_n_i = struct[0].i_D1_n_i
    i_D10_a_i = struct[0].i_D10_a_i
    i_D10_b_r = struct[0].i_D10_b_r
    i_D10_b_i = struct[0].i_D10_b_i
    i_D10_c_r = struct[0].i_D10_c_r
    i_D10_c_i = struct[0].i_D10_c_i
    i_D10_n_i = struct[0].i_D10_n_i
    i_D18_b_r = struct[0].i_D18_b_r
    i_D18_b_i = struct[0].i_D18_b_i
    i_D18_c_r = struct[0].i_D18_c_r
    i_D18_c_i = struct[0].i_D18_c_i
    p_R1_a = struct[0].p_R1_a
    q_R1_a = struct[0].q_R1_a
    p_R1_b = struct[0].p_R1_b
    q_R1_b = struct[0].q_R1_b
    p_R1_c = struct[0].p_R1_c
    q_R1_c = struct[0].q_R1_c
    p_R18_1 = struct[0].p_R18_1
    q_R18_1 = struct[0].q_R18_1
    p_D18_1 = struct[0].p_D18_1
    q_D18_1 = struct[0].q_D18_1
    v_dc_D1 = struct[0].v_dc_D1
    q_R1 = struct[0].q_R1
    p_R10 = struct[0].p_R10
    q_R10 = struct[0].q_R10
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_R1_a_r = struct[0].y_run[0,0]
    v_R1_a_i = struct[0].y_run[1,0]
    v_R1_b_r = struct[0].y_run[2,0]
    v_R1_b_i = struct[0].y_run[3,0]
    v_R1_c_r = struct[0].y_run[4,0]
    v_R1_c_i = struct[0].y_run[5,0]
    v_R1_n_r = struct[0].y_run[6,0]
    v_R1_n_i = struct[0].y_run[7,0]
    v_R18_a_r = struct[0].y_run[8,0]
    v_R18_a_i = struct[0].y_run[9,0]
    v_R18_n_r = struct[0].y_run[10,0]
    v_R18_n_i = struct[0].y_run[11,0]
    v_D18_a_r = struct[0].y_run[12,0]
    v_D18_a_i = struct[0].y_run[13,0]
    v_D18_n_r = struct[0].y_run[14,0]
    v_D18_n_i = struct[0].y_run[15,0]
    v_R10_a_r = struct[0].y_run[16,0]
    v_R10_a_i = struct[0].y_run[17,0]
    v_R10_b_r = struct[0].y_run[18,0]
    v_R10_b_i = struct[0].y_run[19,0]
    v_R10_c_r = struct[0].y_run[20,0]
    v_R10_c_i = struct[0].y_run[21,0]
    v_R10_n_r = struct[0].y_run[22,0]
    v_R10_n_i = struct[0].y_run[23,0]
    v_R18_b_r = struct[0].y_run[24,0]
    v_R18_b_i = struct[0].y_run[25,0]
    v_R18_c_r = struct[0].y_run[26,0]
    v_R18_c_i = struct[0].y_run[27,0]
    v_D1_n_r = struct[0].y_run[28,0]
    v_D1_n_i = struct[0].y_run[29,0]
    v_D10_a_r = struct[0].y_run[30,0]
    v_D10_a_i = struct[0].y_run[31,0]
    v_D10_b_r = struct[0].y_run[32,0]
    v_D10_b_i = struct[0].y_run[33,0]
    v_D10_c_r = struct[0].y_run[34,0]
    v_D10_c_i = struct[0].y_run[35,0]
    v_D10_n_r = struct[0].y_run[36,0]
    v_D10_n_i = struct[0].y_run[37,0]
    v_D18_b_r = struct[0].y_run[38,0]
    v_D18_b_i = struct[0].y_run[39,0]
    v_D18_c_r = struct[0].y_run[40,0]
    v_D18_c_i = struct[0].y_run[41,0]
    i_t_R0_R1_a_r = struct[0].y_run[42,0]
    i_t_R0_R1_a_i = struct[0].y_run[43,0]
    i_t_R0_R1_b_r = struct[0].y_run[44,0]
    i_t_R0_R1_b_i = struct[0].y_run[45,0]
    i_t_R0_R1_c_r = struct[0].y_run[46,0]
    i_t_R0_R1_c_i = struct[0].y_run[47,0]
    i_l_R1_R10_a_r = struct[0].y_run[48,0]
    i_l_R1_R10_a_i = struct[0].y_run[49,0]
    i_l_R1_R10_b_r = struct[0].y_run[50,0]
    i_l_R1_R10_b_i = struct[0].y_run[51,0]
    i_l_R1_R10_c_r = struct[0].y_run[52,0]
    i_l_R1_R10_c_i = struct[0].y_run[53,0]
    i_l_R1_R10_n_r = struct[0].y_run[54,0]
    i_l_R1_R10_n_i = struct[0].y_run[55,0]
    i_l_D1_D10_a_r = struct[0].y_run[56,0]
    i_l_D1_D10_a_i = struct[0].y_run[57,0]
    i_l_D1_D10_b_r = struct[0].y_run[58,0]
    i_l_D1_D10_b_i = struct[0].y_run[59,0]
    i_l_D1_D10_c_r = struct[0].y_run[60,0]
    i_l_D1_D10_c_i = struct[0].y_run[61,0]
    i_l_D1_D10_n_r = struct[0].y_run[62,0]
    i_l_D1_D10_n_i = struct[0].y_run[63,0]
    i_l_D10_D18_a_r = struct[0].y_run[64,0]
    i_l_D10_D18_a_i = struct[0].y_run[65,0]
    i_l_D10_D18_b_r = struct[0].y_run[66,0]
    i_l_D10_D18_b_i = struct[0].y_run[67,0]
    i_l_D10_D18_c_r = struct[0].y_run[68,0]
    i_l_D10_D18_c_i = struct[0].y_run[69,0]
    i_l_D10_D18_n_r = struct[0].y_run[70,0]
    i_l_D10_D18_n_i = struct[0].y_run[71,0]
    i_load_R1_a_r = struct[0].y_run[72,0]
    i_load_R1_a_i = struct[0].y_run[73,0]
    i_load_R1_b_r = struct[0].y_run[74,0]
    i_load_R1_b_i = struct[0].y_run[75,0]
    i_load_R1_c_r = struct[0].y_run[76,0]
    i_load_R1_c_i = struct[0].y_run[77,0]
    i_load_R1_n_r = struct[0].y_run[78,0]
    i_load_R1_n_i = struct[0].y_run[79,0]
    i_load_R18_a_r = struct[0].y_run[80,0]
    i_load_R18_a_i = struct[0].y_run[81,0]
    i_load_R18_n_r = struct[0].y_run[82,0]
    i_load_R18_n_i = struct[0].y_run[83,0]
    i_load_D18_a_r = struct[0].y_run[84,0]
    i_load_D18_a_i = struct[0].y_run[85,0]
    i_load_D18_n_r = struct[0].y_run[86,0]
    i_load_D18_n_i = struct[0].y_run[87,0]
    i_vsc_R1_a_r = struct[0].y_run[88,0]
    i_vsc_R1_a_i = struct[0].y_run[89,0]
    i_vsc_R1_b_r = struct[0].y_run[90,0]
    i_vsc_R1_b_i = struct[0].y_run[91,0]
    i_vsc_R1_c_r = struct[0].y_run[92,0]
    i_vsc_R1_c_i = struct[0].y_run[93,0]
    p_R1 = struct[0].y_run[94,0]
    p_D1 = struct[0].y_run[95,0]
    p_loss_R1 = struct[0].y_run[96,0]
    i_vsc_R10_a_r = struct[0].y_run[97,0]
    i_vsc_R10_a_i = struct[0].y_run[98,0]
    i_vsc_R10_b_r = struct[0].y_run[99,0]
    i_vsc_R10_b_i = struct[0].y_run[100,0]
    i_vsc_R10_c_r = struct[0].y_run[101,0]
    i_vsc_R10_c_i = struct[0].y_run[102,0]
    i_vsc_D10_a_r = struct[0].y_run[103,0]
    i_vsc_D10_n_r = struct[0].y_run[104,0]
    p_D10 = struct[0].y_run[105,0]
    p_loss_R10 = struct[0].y_run[106,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_load_R1_a_r + i_vsc_R1_a_r + 0.849044513514155*v_R0_a_i + 0.212261128378539*v_R0_a_r - 0.849044513514155*v_R0_c_i - 0.212261128378539*v_R0_c_r + 5.40657727682604*v_R10_a_i + 10.557176931318*v_R10_a_r - 1.02713736253513*v_R10_b_i - 3.96392229058202*v_R10_b_r - 2.3284964480954*v_R10_c_i - 2.49575997948692*v_R10_c_r - 1.02713736253513*v_R10_n_i - 3.96392229058202*v_R10_n_r - 78.9359890415319*v_R1_a_i - 28.9395298724945*v_R1_a_r + 1.02713736253513*v_R1_b_i + 3.96392229058202*v_R1_b_r + 2.3284964480954*v_R1_c_i + 2.49575997948692*v_R1_c_r + 74.556549127241*v_R1_n_i + 22.3462752317585*v_R1_n_r
        struct[0].g[1,0] = i_load_R1_a_i + i_vsc_R1_a_i + 0.212261128378539*v_R0_a_i - 0.849044513514155*v_R0_a_r - 0.212261128378539*v_R0_c_i + 0.849044513514155*v_R0_c_r + 10.557176931318*v_R10_a_i - 5.40657727682604*v_R10_a_r - 3.96392229058202*v_R10_b_i + 1.02713736253513*v_R10_b_r - 2.49575997948692*v_R10_c_i + 2.3284964480954*v_R10_c_r - 3.96392229058202*v_R10_n_i + 1.02713736253513*v_R10_n_r - 28.9395298724945*v_R1_a_i + 78.9359890415319*v_R1_a_r + 3.96392229058202*v_R1_b_i - 1.02713736253513*v_R1_b_r + 2.49575997948692*v_R1_c_i - 2.3284964480954*v_R1_c_r + 22.3462752317585*v_R1_n_i - 74.556549127241*v_R1_n_r
        struct[0].g[2,0] = i_load_R1_b_r + i_vsc_R1_b_r - 0.849044513514155*v_R0_a_i - 0.212261128378539*v_R0_a_r + 0.849044513514155*v_R0_b_i + 0.212261128378539*v_R0_b_r - 1.02713736253513*v_R10_a_i - 3.96392229058202*v_R10_a_r + 5.40657727682604*v_R10_b_i + 10.557176931318*v_R10_b_r - 1.02713736253513*v_R10_c_i - 3.96392229058202*v_R10_c_r - 2.3284964480954*v_R10_n_i - 2.49575997948692*v_R10_n_r + 1.02713736253513*v_R1_a_i + 3.96392229058202*v_R1_a_r - 78.9359890415319*v_R1_b_i - 28.9395298724945*v_R1_b_r + 1.02713736253513*v_R1_c_i + 3.96392229058202*v_R1_c_r + 75.8579082128012*v_R1_n_i + 20.8781129206634*v_R1_n_r
        struct[0].g[3,0] = i_load_R1_b_i + i_vsc_R1_b_i - 0.212261128378539*v_R0_a_i + 0.849044513514155*v_R0_a_r + 0.212261128378539*v_R0_b_i - 0.849044513514155*v_R0_b_r - 3.96392229058202*v_R10_a_i + 1.02713736253513*v_R10_a_r + 10.557176931318*v_R10_b_i - 5.40657727682604*v_R10_b_r - 3.96392229058202*v_R10_c_i + 1.02713736253513*v_R10_c_r - 2.49575997948692*v_R10_n_i + 2.3284964480954*v_R10_n_r + 3.96392229058202*v_R1_a_i - 1.02713736253513*v_R1_a_r - 28.9395298724945*v_R1_b_i + 78.9359890415319*v_R1_b_r + 3.96392229058202*v_R1_c_i - 1.02713736253513*v_R1_c_r + 20.8781129206634*v_R1_n_i - 75.8579082128012*v_R1_n_r
        struct[0].g[4,0] = i_load_R1_c_r + i_vsc_R1_c_r - 0.849044513514155*v_R0_b_i - 0.212261128378539*v_R0_b_r + 0.849044513514155*v_R0_c_i + 0.212261128378539*v_R0_c_r - 2.3284964480954*v_R10_a_i - 2.49575997948692*v_R10_a_r - 1.02713736253513*v_R10_b_i - 3.96392229058202*v_R10_b_r + 5.40657727682604*v_R10_c_i + 10.557176931318*v_R10_c_r - 1.02713736253513*v_R10_n_i - 3.96392229058202*v_R10_n_r + 2.3284964480954*v_R1_a_i + 2.49575997948692*v_R1_a_r + 1.02713736253513*v_R1_b_i + 3.96392229058202*v_R1_b_r - 78.9359890415319*v_R1_c_i - 28.9395298724945*v_R1_c_r + 74.556549127241*v_R1_n_i + 22.3462752317585*v_R1_n_r
        struct[0].g[5,0] = i_load_R1_c_i + i_vsc_R1_c_i - 0.212261128378539*v_R0_b_i + 0.849044513514155*v_R0_b_r + 0.212261128378539*v_R0_c_i - 0.849044513514155*v_R0_c_r - 2.49575997948692*v_R10_a_i + 2.3284964480954*v_R10_a_r - 3.96392229058202*v_R10_b_i + 1.02713736253513*v_R10_b_r + 10.557176931318*v_R10_c_i - 5.40657727682604*v_R10_c_r - 3.96392229058202*v_R10_n_i + 1.02713736253513*v_R10_n_r + 2.49575997948692*v_R1_a_i - 2.3284964480954*v_R1_a_r + 3.96392229058202*v_R1_b_i - 1.02713736253513*v_R1_b_r - 28.9395298724945*v_R1_c_i + 78.9359890415319*v_R1_c_r + 22.3462752317585*v_R1_n_i - 74.556549127241*v_R1_n_r
        struct[0].g[6,0] = -1.02713736253513*v_R10_a_i - 3.96392229058202*v_R10_a_r - 2.3284964480954*v_R10_b_i - 2.49575997948692*v_R10_b_r - 1.02713736253513*v_R10_c_i - 3.96392229058202*v_R10_c_r + 5.40657727682604*v_R10_n_i + 10.557176931318*v_R10_n_r + 74.556549127241*v_R1_a_i + 22.3462752317585*v_R1_a_r + 75.8579082128012*v_R1_b_i + 20.8781129206634*v_R1_b_r + 74.556549127241*v_R1_c_i + 22.3462752317585*v_R1_c_r - 225.994812570944*v_R1_n_i - 66.0375690881807*v_R1_n_r
        struct[0].g[7,0] = -3.96392229058202*v_R10_a_i + 1.02713736253513*v_R10_a_r - 2.49575997948692*v_R10_b_i + 2.3284964480954*v_R10_b_r - 3.96392229058202*v_R10_c_i + 1.02713736253513*v_R10_c_r + 10.557176931318*v_R10_n_i - 5.40657727682604*v_R10_n_r + 22.3462752317585*v_R1_a_i - 74.556549127241*v_R1_a_r + 20.8781129206634*v_R1_b_i - 75.8579082128012*v_R1_b_r + 22.3462752317585*v_R1_c_i - 74.556549127241*v_R1_c_r - 66.0375690881807*v_R1_n_i + 225.994812570944*v_R1_n_r
        struct[0].g[8,0] = i_load_R18_a_r + 5.65456401516768*v_R10_a_i + 30.9517475172273*v_R10_a_r + 1.84896616921897*v_R10_b_i - 9.21038227100566*v_R10_b_r + 0.793238195499529*v_R10_c_i - 9.00835072044485*v_R10_c_r + 1.84896616921897*v_R10_n_i - 9.21038227100566*v_R10_n_r - 5.65456401516768*v_R18_a_i - 30.9517475172273*v_R18_a_r - 1.84896616921897*v_R18_b_i + 9.21038227100566*v_R18_b_r - 0.793238195499529*v_R18_c_i + 9.00835072044485*v_R18_c_r - 1.84896616921897*v_R18_n_i + 9.21038227100566*v_R18_n_r
        struct[0].g[9,0] = i_load_R18_a_i + 30.9517475172273*v_R10_a_i - 5.65456401516768*v_R10_a_r - 9.21038227100566*v_R10_b_i - 1.84896616921897*v_R10_b_r - 9.00835072044485*v_R10_c_i - 0.793238195499529*v_R10_c_r - 9.21038227100566*v_R10_n_i - 1.84896616921897*v_R10_n_r - 30.9517475172273*v_R18_a_i + 5.65456401516768*v_R18_a_r + 9.21038227100566*v_R18_b_i + 1.84896616921897*v_R18_b_r + 9.00835072044485*v_R18_c_i + 0.793238195499529*v_R18_c_r + 9.21038227100566*v_R18_n_i + 1.84896616921897*v_R18_n_r
        struct[0].g[10,0] = i_load_R18_n_r + 1.84896616921897*v_R10_a_i - 9.21038227100566*v_R10_a_r + 0.793238195499527*v_R10_b_i - 9.00835072044485*v_R10_b_r + 1.84896616921897*v_R10_c_i - 9.21038227100566*v_R10_c_r + 5.65456401516768*v_R10_n_i + 30.9517475172273*v_R10_n_r - 1.84896616921897*v_R18_a_i + 9.21038227100566*v_R18_a_r - 0.793238195499527*v_R18_b_i + 9.00835072044485*v_R18_b_r - 1.84896616921897*v_R18_c_i + 9.21038227100566*v_R18_c_r - 5.65456401516768*v_R18_n_i - 30.9767475172273*v_R18_n_r
        struct[0].g[11,0] = i_load_R18_n_i - 9.21038227100566*v_R10_a_i - 1.84896616921897*v_R10_a_r - 9.00835072044485*v_R10_b_i - 0.793238195499527*v_R10_b_r - 9.21038227100566*v_R10_c_i - 1.84896616921897*v_R10_c_r + 30.9517475172273*v_R10_n_i - 5.65456401516768*v_R10_n_r + 9.21038227100566*v_R18_a_i + 1.84896616921897*v_R18_a_r + 9.00835072044485*v_R18_b_i + 0.793238195499527*v_R18_b_r + 9.21038227100566*v_R18_c_i + 1.84896616921897*v_R18_c_r - 30.9767475172273*v_R18_n_i + 5.65456401516768*v_R18_n_r
        struct[0].g[12,0] = i_load_D18_a_r + 157.977883096366*v_D10_a_r - 157.977883096366*v_D18_a_r
        struct[0].g[13,0] = i_load_D18_a_i + 157.977883096366*v_D10_a_i - 157.977883096366*v_D18_a_i
        struct[0].g[14,0] = i_load_D18_n_r + 157.977883096366*v_D10_n_r - 157.977883096366*v_D18_n_r
        struct[0].g[15,0] = i_load_D18_n_i + 157.977883096366*v_D10_n_i - 157.977883096366*v_D18_n_i
        struct[0].g[16,0] = i_vsc_R10_a_r - 11.0611412919937*v_R10_a_i - 41.5089244485453*v_R10_a_r - 0.821828806683838*v_R10_b_i + 13.1743045615877*v_R10_b_r + 1.53525825259587*v_R10_c_i + 11.5041106999318*v_R10_c_r - 0.82182880668384*v_R10_n_i + 13.1743045615877*v_R10_n_r + 5.65456401516768*v_R18_a_i + 30.9517475172273*v_R18_a_r + 1.84896616921897*v_R18_b_i - 9.21038227100566*v_R18_b_r + 0.793238195499529*v_R18_c_i - 9.00835072044485*v_R18_c_r + 1.84896616921897*v_R18_n_i - 9.21038227100566*v_R18_n_r + 5.40657727682604*v_R1_a_i + 10.557176931318*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r - 2.3284964480954*v_R1_c_i - 2.49575997948692*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[17,0] = i_vsc_R10_a_i - 41.5089244485453*v_R10_a_i + 11.0611412919937*v_R10_a_r + 13.1743045615877*v_R10_b_i + 0.821828806683838*v_R10_b_r + 11.5041106999318*v_R10_c_i - 1.53525825259587*v_R10_c_r + 13.1743045615877*v_R10_n_i + 0.82182880668384*v_R10_n_r + 30.9517475172273*v_R18_a_i - 5.65456401516768*v_R18_a_r - 9.21038227100566*v_R18_b_i - 1.84896616921897*v_R18_b_r - 9.00835072044485*v_R18_c_i - 0.793238195499529*v_R18_c_r - 9.21038227100566*v_R18_n_i - 1.84896616921897*v_R18_n_r + 10.557176931318*v_R1_a_i - 5.40657727682604*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r - 2.49575997948692*v_R1_c_i + 2.3284964480954*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[18,0] = i_vsc_R10_b_r - 0.821828806683841*v_R10_a_i + 13.1743045615877*v_R10_a_r - 11.0611412919937*v_R10_b_i - 41.5089244485453*v_R10_b_r - 0.821828806683839*v_R10_c_i + 13.1743045615877*v_R10_c_r + 1.53525825259588*v_R10_n_i + 11.5041106999318*v_R10_n_r + 1.84896616921897*v_R18_a_i - 9.21038227100566*v_R18_a_r + 5.65456401516768*v_R18_b_i + 30.9517475172273*v_R18_b_r + 1.84896616921897*v_R18_c_i - 9.21038227100566*v_R18_c_r + 0.793238195499528*v_R18_n_i - 9.00835072044485*v_R18_n_r - 1.02713736253513*v_R1_a_i - 3.96392229058202*v_R1_a_r + 5.40657727682604*v_R1_b_i + 10.557176931318*v_R1_b_r - 1.02713736253513*v_R1_c_i - 3.96392229058202*v_R1_c_r - 2.3284964480954*v_R1_n_i - 2.49575997948692*v_R1_n_r
        struct[0].g[19,0] = i_vsc_R10_b_i + 13.1743045615877*v_R10_a_i + 0.821828806683841*v_R10_a_r - 41.5089244485453*v_R10_b_i + 11.0611412919937*v_R10_b_r + 13.1743045615877*v_R10_c_i + 0.821828806683839*v_R10_c_r + 11.5041106999318*v_R10_n_i - 1.53525825259588*v_R10_n_r - 9.21038227100566*v_R18_a_i - 1.84896616921897*v_R18_a_r + 30.9517475172273*v_R18_b_i - 5.65456401516768*v_R18_b_r - 9.21038227100566*v_R18_c_i - 1.84896616921897*v_R18_c_r - 9.00835072044485*v_R18_n_i - 0.793238195499528*v_R18_n_r - 3.96392229058202*v_R1_a_i + 1.02713736253513*v_R1_a_r + 10.557176931318*v_R1_b_i - 5.40657727682604*v_R1_b_r - 3.96392229058202*v_R1_c_i + 1.02713736253513*v_R1_c_r - 2.49575997948692*v_R1_n_i + 2.3284964480954*v_R1_n_r
        struct[0].g[20,0] = i_vsc_R10_c_r + 1.53525825259588*v_R10_a_i + 11.5041106999318*v_R10_a_r - 0.82182880668384*v_R10_b_i + 13.1743045615877*v_R10_b_r - 11.0611412919937*v_R10_c_i - 41.5089244485453*v_R10_c_r - 0.821828806683838*v_R10_n_i + 13.1743045615877*v_R10_n_r + 0.793238195499527*v_R18_a_i - 9.00835072044484*v_R18_a_r + 1.84896616921897*v_R18_b_i - 9.21038227100566*v_R18_b_r + 5.65456401516768*v_R18_c_i + 30.9517475172273*v_R18_c_r + 1.84896616921897*v_R18_n_i - 9.21038227100566*v_R18_n_r - 2.3284964480954*v_R1_a_i - 2.49575997948692*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r + 5.40657727682604*v_R1_c_i + 10.557176931318*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[21,0] = i_vsc_R10_c_i + 11.5041106999318*v_R10_a_i - 1.53525825259588*v_R10_a_r + 13.1743045615877*v_R10_b_i + 0.82182880668384*v_R10_b_r - 41.5089244485453*v_R10_c_i + 11.0611412919937*v_R10_c_r + 13.1743045615877*v_R10_n_i + 0.821828806683838*v_R10_n_r - 9.00835072044484*v_R18_a_i - 0.793238195499527*v_R18_a_r - 9.21038227100566*v_R18_b_i - 1.84896616921897*v_R18_b_r + 30.9517475172273*v_R18_c_i - 5.65456401516768*v_R18_c_r - 9.21038227100566*v_R18_n_i - 1.84896616921897*v_R18_n_r - 2.49575997948692*v_R1_a_i + 2.3284964480954*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r + 10.557176931318*v_R1_c_i - 5.40657727682604*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[22,0] = -0.82182880668384*v_R10_a_i + 13.1743045615877*v_R10_a_r + 1.53525825259588*v_R10_b_i + 11.5041106999318*v_R10_b_r - 0.821828806683837*v_R10_c_i + 13.1743045615877*v_R10_c_r - 11.0611412919937*v_R10_n_i - 41.5339244485453*v_R10_n_r + 1.84896616921897*v_R18_a_i - 9.21038227100566*v_R18_a_r + 0.793238195499527*v_R18_b_i - 9.00835072044485*v_R18_b_r + 1.84896616921897*v_R18_c_i - 9.21038227100566*v_R18_c_r + 5.65456401516768*v_R18_n_i + 30.9517475172273*v_R18_n_r - 1.02713736253513*v_R1_a_i - 3.96392229058202*v_R1_a_r - 2.3284964480954*v_R1_b_i - 2.49575997948692*v_R1_b_r - 1.02713736253513*v_R1_c_i - 3.96392229058202*v_R1_c_r + 5.40657727682604*v_R1_n_i + 10.557176931318*v_R1_n_r
        struct[0].g[23,0] = 13.1743045615877*v_R10_a_i + 0.82182880668384*v_R10_a_r + 11.5041106999318*v_R10_b_i - 1.53525825259588*v_R10_b_r + 13.1743045615877*v_R10_c_i + 0.821828806683837*v_R10_c_r - 41.5339244485453*v_R10_n_i + 11.0611412919937*v_R10_n_r - 9.21038227100566*v_R18_a_i - 1.84896616921897*v_R18_a_r - 9.00835072044485*v_R18_b_i - 0.793238195499527*v_R18_b_r - 9.21038227100566*v_R18_c_i - 1.84896616921897*v_R18_c_r + 30.9517475172273*v_R18_n_i - 5.65456401516768*v_R18_n_r - 3.96392229058202*v_R1_a_i + 1.02713736253513*v_R1_a_r - 2.49575997948692*v_R1_b_i + 2.3284964480954*v_R1_b_r - 3.96392229058202*v_R1_c_i + 1.02713736253513*v_R1_c_r + 10.557176931318*v_R1_n_i - 5.40657727682604*v_R1_n_r
        struct[0].g[24,0] = 1.84896616921897*v_R10_a_i - 9.21038227100566*v_R10_a_r + 5.65456401516768*v_R10_b_i + 30.9517475172273*v_R10_b_r + 1.84896616921897*v_R10_c_i - 9.21038227100566*v_R10_c_r + 0.793238195499528*v_R10_n_i - 9.00835072044485*v_R10_n_r - 1.84896616921897*v_R18_a_i + 9.21038227100566*v_R18_a_r - 5.65456401516768*v_R18_b_i - 30.9517475172273*v_R18_b_r - 1.84896616921897*v_R18_c_i + 9.21038227100566*v_R18_c_r - 0.793238195499528*v_R18_n_i + 9.00835072044485*v_R18_n_r
        struct[0].g[25,0] = -9.21038227100566*v_R10_a_i - 1.84896616921897*v_R10_a_r + 30.9517475172273*v_R10_b_i - 5.65456401516768*v_R10_b_r - 9.21038227100566*v_R10_c_i - 1.84896616921897*v_R10_c_r - 9.00835072044485*v_R10_n_i - 0.793238195499528*v_R10_n_r + 9.21038227100566*v_R18_a_i + 1.84896616921897*v_R18_a_r - 30.9517475172273*v_R18_b_i + 5.65456401516768*v_R18_b_r + 9.21038227100566*v_R18_c_i + 1.84896616921897*v_R18_c_r + 9.00835072044485*v_R18_n_i + 0.793238195499528*v_R18_n_r
        struct[0].g[26,0] = 0.793238195499527*v_R10_a_i - 9.00835072044484*v_R10_a_r + 1.84896616921897*v_R10_b_i - 9.21038227100566*v_R10_b_r + 5.65456401516768*v_R10_c_i + 30.9517475172273*v_R10_c_r + 1.84896616921897*v_R10_n_i - 9.21038227100566*v_R10_n_r - 0.793238195499527*v_R18_a_i + 9.00835072044484*v_R18_a_r - 1.84896616921897*v_R18_b_i + 9.21038227100566*v_R18_b_r - 5.65456401516768*v_R18_c_i - 30.9517475172273*v_R18_c_r - 1.84896616921897*v_R18_n_i + 9.21038227100566*v_R18_n_r
        struct[0].g[27,0] = -9.00835072044484*v_R10_a_i - 0.793238195499527*v_R10_a_r - 9.21038227100566*v_R10_b_i - 1.84896616921897*v_R10_b_r + 30.9517475172273*v_R10_c_i - 5.65456401516768*v_R10_c_r - 9.21038227100566*v_R10_n_i - 1.84896616921897*v_R10_n_r + 9.00835072044484*v_R18_a_i + 0.793238195499527*v_R18_a_r + 9.21038227100566*v_R18_b_i + 1.84896616921897*v_R18_b_r - 30.9517475172273*v_R18_c_i + 5.65456401516768*v_R18_c_r + 9.21038227100566*v_R18_n_i + 1.84896616921897*v_R18_n_r
        struct[0].g[28,0] = 67.7048070412999*v_D10_n_r - 1067.7048070413*v_D1_n_r
        struct[0].g[29,0] = 67.7048070412999*v_D10_n_i - 1067.7048070413*v_D1_n_i
        struct[0].g[30,0] = i_vsc_D10_a_r - 225.682690137666*v_D10_a_r + 157.977883096366*v_D18_a_r + 67.7048070412999*v_D1_a_r
        struct[0].g[31,0] = -225.682690137666*v_D10_a_i + 157.977883096366*v_D18_a_i + 67.7048070412999*v_D1_a_i
        struct[0].g[32,0] = -225.682690137666*v_D10_b_r + 157.977883096366*v_D18_b_r + 67.7048070412999*v_D1_b_r
        struct[0].g[33,0] = -225.682690137666*v_D10_b_i + 157.977883096366*v_D18_b_i + 67.7048070412999*v_D1_b_i
        struct[0].g[34,0] = -225.682690137666*v_D10_c_r + 157.977883096366*v_D18_c_r + 67.7048070412999*v_D1_c_r
        struct[0].g[35,0] = -225.682690137666*v_D10_c_i + 157.977883096366*v_D18_c_i + 67.7048070412999*v_D1_c_i
        struct[0].g[36,0] = i_vsc_D10_n_r - 225.682690137666*v_D10_n_r + 157.977883096366*v_D18_n_r + 67.7048070412999*v_D1_n_r
        struct[0].g[37,0] = -225.682690137666*v_D10_n_i + 157.977883096366*v_D18_n_i + 67.7048070412999*v_D1_n_i
        struct[0].g[38,0] = 157.977883096366*v_D10_b_r - 157.977883096366*v_D18_b_r
        struct[0].g[39,0] = 157.977883096366*v_D10_b_i - 157.977883096366*v_D18_b_i
        struct[0].g[40,0] = 157.977883096366*v_D10_c_r - 157.977883096366*v_D18_c_r
        struct[0].g[41,0] = 157.977883096366*v_D10_c_i - 157.977883096366*v_D18_c_i
        struct[0].g[42,0] = -i_t_R0_R1_a_r + 0.0196078431372549*v_R0_a_i + 0.00490196078431373*v_R0_a_r - 0.00980392156862745*v_R0_b_i - 0.00245098039215686*v_R0_b_r - 0.00980392156862745*v_R0_c_i - 0.00245098039215686*v_R0_c_r - 0.849044513514155*v_R1_a_i - 0.212261128378539*v_R1_a_r + 0.849044513514155*v_R1_b_i + 0.212261128378539*v_R1_b_r
        struct[0].g[43,0] = -i_t_R0_R1_a_i + 0.00490196078431373*v_R0_a_i - 0.0196078431372549*v_R0_a_r - 0.00245098039215686*v_R0_b_i + 0.00980392156862745*v_R0_b_r - 0.00245098039215686*v_R0_c_i + 0.00980392156862745*v_R0_c_r - 0.212261128378539*v_R1_a_i + 0.849044513514155*v_R1_a_r + 0.212261128378539*v_R1_b_i - 0.849044513514155*v_R1_b_r
        struct[0].g[44,0] = -i_t_R0_R1_b_r - 0.00980392156862745*v_R0_a_i - 0.00245098039215686*v_R0_a_r + 0.0196078431372549*v_R0_b_i + 0.00490196078431373*v_R0_b_r - 0.00980392156862745*v_R0_c_i - 0.00245098039215686*v_R0_c_r - 0.849044513514155*v_R1_b_i - 0.212261128378539*v_R1_b_r + 0.849044513514155*v_R1_c_i + 0.212261128378539*v_R1_c_r
        struct[0].g[45,0] = -i_t_R0_R1_b_i - 0.00245098039215686*v_R0_a_i + 0.00980392156862745*v_R0_a_r + 0.00490196078431373*v_R0_b_i - 0.0196078431372549*v_R0_b_r - 0.00245098039215686*v_R0_c_i + 0.00980392156862745*v_R0_c_r - 0.212261128378539*v_R1_b_i + 0.849044513514155*v_R1_b_r + 0.212261128378539*v_R1_c_i - 0.849044513514155*v_R1_c_r
        struct[0].g[46,0] = -i_t_R0_R1_c_r - 0.00980392156862745*v_R0_a_i - 0.00245098039215686*v_R0_a_r - 0.00980392156862745*v_R0_b_i - 0.00245098039215686*v_R0_b_r + 0.0196078431372549*v_R0_c_i + 0.00490196078431373*v_R0_c_r + 0.849044513514155*v_R1_a_i + 0.212261128378539*v_R1_a_r - 0.849044513514155*v_R1_c_i - 0.212261128378539*v_R1_c_r
        struct[0].g[47,0] = -i_t_R0_R1_c_i - 0.00245098039215686*v_R0_a_i + 0.00980392156862745*v_R0_a_r - 0.00245098039215686*v_R0_b_i + 0.00980392156862745*v_R0_b_r + 0.00490196078431373*v_R0_c_i - 0.0196078431372549*v_R0_c_r + 0.212261128378539*v_R1_a_i - 0.849044513514155*v_R1_a_r - 0.212261128378539*v_R1_c_i + 0.849044513514155*v_R1_c_r
        struct[0].g[48,0] = -i_l_R1_R10_a_r - 5.40657727682604*v_R10_a_i - 10.557176931318*v_R10_a_r + 1.02713736253513*v_R10_b_i + 3.96392229058202*v_R10_b_r + 2.3284964480954*v_R10_c_i + 2.49575997948692*v_R10_c_r + 1.02713736253513*v_R10_n_i + 3.96392229058202*v_R10_n_r + 5.40657727682604*v_R1_a_i + 10.557176931318*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r - 2.3284964480954*v_R1_c_i - 2.49575997948692*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[49,0] = -i_l_R1_R10_a_i - 10.557176931318*v_R10_a_i + 5.40657727682604*v_R10_a_r + 3.96392229058202*v_R10_b_i - 1.02713736253513*v_R10_b_r + 2.49575997948692*v_R10_c_i - 2.3284964480954*v_R10_c_r + 3.96392229058202*v_R10_n_i - 1.02713736253513*v_R10_n_r + 10.557176931318*v_R1_a_i - 5.40657727682604*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r - 2.49575997948692*v_R1_c_i + 2.3284964480954*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[50,0] = -i_l_R1_R10_b_r + 1.02713736253513*v_R10_a_i + 3.96392229058202*v_R10_a_r - 5.40657727682604*v_R10_b_i - 10.557176931318*v_R10_b_r + 1.02713736253513*v_R10_c_i + 3.96392229058202*v_R10_c_r + 2.3284964480954*v_R10_n_i + 2.49575997948692*v_R10_n_r - 1.02713736253513*v_R1_a_i - 3.96392229058202*v_R1_a_r + 5.40657727682604*v_R1_b_i + 10.557176931318*v_R1_b_r - 1.02713736253513*v_R1_c_i - 3.96392229058202*v_R1_c_r - 2.3284964480954*v_R1_n_i - 2.49575997948692*v_R1_n_r
        struct[0].g[51,0] = -i_l_R1_R10_b_i + 3.96392229058202*v_R10_a_i - 1.02713736253513*v_R10_a_r - 10.557176931318*v_R10_b_i + 5.40657727682604*v_R10_b_r + 3.96392229058202*v_R10_c_i - 1.02713736253513*v_R10_c_r + 2.49575997948692*v_R10_n_i - 2.3284964480954*v_R10_n_r - 3.96392229058202*v_R1_a_i + 1.02713736253513*v_R1_a_r + 10.557176931318*v_R1_b_i - 5.40657727682604*v_R1_b_r - 3.96392229058202*v_R1_c_i + 1.02713736253513*v_R1_c_r - 2.49575997948692*v_R1_n_i + 2.3284964480954*v_R1_n_r
        struct[0].g[52,0] = -i_l_R1_R10_c_r + 2.3284964480954*v_R10_a_i + 2.49575997948692*v_R10_a_r + 1.02713736253513*v_R10_b_i + 3.96392229058202*v_R10_b_r - 5.40657727682604*v_R10_c_i - 10.557176931318*v_R10_c_r + 1.02713736253513*v_R10_n_i + 3.96392229058202*v_R10_n_r - 2.3284964480954*v_R1_a_i - 2.49575997948692*v_R1_a_r - 1.02713736253513*v_R1_b_i - 3.96392229058202*v_R1_b_r + 5.40657727682604*v_R1_c_i + 10.557176931318*v_R1_c_r - 1.02713736253513*v_R1_n_i - 3.96392229058202*v_R1_n_r
        struct[0].g[53,0] = -i_l_R1_R10_c_i + 2.49575997948692*v_R10_a_i - 2.3284964480954*v_R10_a_r + 3.96392229058202*v_R10_b_i - 1.02713736253513*v_R10_b_r - 10.557176931318*v_R10_c_i + 5.40657727682604*v_R10_c_r + 3.96392229058202*v_R10_n_i - 1.02713736253513*v_R10_n_r - 2.49575997948692*v_R1_a_i + 2.3284964480954*v_R1_a_r - 3.96392229058202*v_R1_b_i + 1.02713736253513*v_R1_b_r + 10.557176931318*v_R1_c_i - 5.40657727682604*v_R1_c_r - 3.96392229058202*v_R1_n_i + 1.02713736253513*v_R1_n_r
        struct[0].g[54,0] = i_l_R1_R10_a_r + i_l_R1_R10_b_r + i_l_R1_R10_c_r - i_l_R1_R10_n_r
        struct[0].g[55,0] = i_l_R1_R10_a_i + i_l_R1_R10_b_i + i_l_R1_R10_c_i - i_l_R1_R10_n_i
        struct[0].g[56,0] = -i_l_D1_D10_a_r - 67.7048070412999*v_D10_a_r + 67.7048070412999*v_D1_a_r
        struct[0].g[57,0] = -i_l_D1_D10_a_i - 67.7048070412999*v_D10_a_i + 67.7048070412999*v_D1_a_i
        struct[0].g[58,0] = -i_l_D1_D10_b_r - 67.7048070412999*v_D10_b_r + 67.7048070412999*v_D1_b_r
        struct[0].g[59,0] = -i_l_D1_D10_b_i - 67.7048070412999*v_D10_b_i + 67.7048070412999*v_D1_b_i
        struct[0].g[60,0] = -i_l_D1_D10_c_r - 67.7048070412999*v_D10_c_r + 67.7048070412999*v_D1_c_r
        struct[0].g[61,0] = -i_l_D1_D10_c_i - 67.7048070412999*v_D10_c_i + 67.7048070412999*v_D1_c_i
        struct[0].g[62,0] = i_l_D1_D10_a_r + i_l_D1_D10_b_r + i_l_D1_D10_c_r - i_l_D1_D10_n_r
        struct[0].g[63,0] = i_l_D1_D10_a_i + i_l_D1_D10_b_i + i_l_D1_D10_c_i - i_l_D1_D10_n_i
        struct[0].g[64,0] = -i_l_D10_D18_a_r + 157.977883096366*v_D10_a_r - 157.977883096366*v_D18_a_r
        struct[0].g[65,0] = -i_l_D10_D18_a_i + 157.977883096366*v_D10_a_i - 157.977883096366*v_D18_a_i
        struct[0].g[66,0] = -i_l_D10_D18_b_r + 157.977883096366*v_D10_b_r - 157.977883096366*v_D18_b_r
        struct[0].g[67,0] = -i_l_D10_D18_b_i + 157.977883096366*v_D10_b_i - 157.977883096366*v_D18_b_i
        struct[0].g[68,0] = -i_l_D10_D18_c_r + 157.977883096366*v_D10_c_r - 157.977883096366*v_D18_c_r
        struct[0].g[69,0] = -i_l_D10_D18_c_i + 157.977883096366*v_D10_c_i - 157.977883096366*v_D18_c_i
        struct[0].g[70,0] = i_l_D10_D18_a_r + i_l_D10_D18_b_r + i_l_D10_D18_c_r - i_l_D10_D18_n_r
        struct[0].g[71,0] = i_l_D10_D18_a_i + i_l_D10_D18_b_i + i_l_D10_D18_c_i - i_l_D10_D18_n_i
        struct[0].g[72,0] = i_load_R1_a_i*v_R1_a_i - i_load_R1_a_i*v_R1_n_i + i_load_R1_a_r*v_R1_a_r - i_load_R1_a_r*v_R1_n_r - p_R1_a
        struct[0].g[73,0] = i_load_R1_b_i*v_R1_b_i - i_load_R1_b_i*v_R1_n_i + i_load_R1_b_r*v_R1_b_r - i_load_R1_b_r*v_R1_n_r - p_R1_b
        struct[0].g[74,0] = i_load_R1_c_i*v_R1_c_i - i_load_R1_c_i*v_R1_n_i + i_load_R1_c_r*v_R1_c_r - i_load_R1_c_r*v_R1_n_r - p_R1_c
        struct[0].g[75,0] = -i_load_R1_a_i*v_R1_a_r + i_load_R1_a_i*v_R1_n_r + i_load_R1_a_r*v_R1_a_i - i_load_R1_a_r*v_R1_n_i - q_R1_a
        struct[0].g[76,0] = -i_load_R1_b_i*v_R1_b_r + i_load_R1_b_i*v_R1_n_r + i_load_R1_b_r*v_R1_b_i - i_load_R1_b_r*v_R1_n_i - q_R1_b
        struct[0].g[77,0] = -i_load_R1_c_i*v_R1_c_r + i_load_R1_c_i*v_R1_n_r + i_load_R1_c_r*v_R1_c_i - i_load_R1_c_r*v_R1_n_i - q_R1_c
        struct[0].g[78,0] = i_load_R1_a_r + i_load_R1_b_r + i_load_R1_c_r + i_load_R1_n_r
        struct[0].g[79,0] = i_load_R1_a_i + i_load_R1_b_i + i_load_R1_c_i + i_load_R1_n_i
        struct[0].g[80,0] = 1.0*i_load_R18_a_i*v_R18_a_i - 1.0*i_load_R18_a_i*v_R18_n_i + i_load_R18_a_r*v_R18_a_r - i_load_R18_a_r*v_R18_n_r - p_R18_1
        struct[0].g[81,0] = -1.0*i_load_R18_a_i*v_R18_a_r + 1.0*i_load_R18_a_i*v_R18_n_r + 1.0*i_load_R18_a_r*v_R18_a_i - 1.0*i_load_R18_a_r*v_R18_n_i - q_R18_1
        struct[0].g[82,0] = i_load_R18_a_r + i_load_R18_n_r
        struct[0].g[83,0] = 1.0*i_load_R18_a_i + 1.0*i_load_R18_n_i
        struct[0].g[84,0] = 1.0*i_load_D18_a_i*v_D18_a_i - 1.0*i_load_D18_a_i*v_D18_n_i + i_load_D18_a_r*v_D18_a_r - i_load_D18_a_r*v_D18_n_r - p_D18_1
        struct[0].g[85,0] = -1.0*i_load_D18_a_i*v_D18_a_r + 1.0*i_load_D18_a_i*v_D18_n_r + 1.0*i_load_D18_a_r*v_D18_a_i - 1.0*i_load_D18_a_r*v_D18_n_i - q_D18_1
        struct[0].g[86,0] = i_load_D18_a_r + i_load_D18_n_r
        struct[0].g[87,0] = 1.0*i_load_D18_a_i + 1.0*i_load_D18_n_i
        struct[0].g[88,0] = 1.0*i_vsc_R1_a_i*v_R1_a_i - 1.0*i_vsc_R1_a_i*v_R1_n_i + i_vsc_R1_a_r*v_R1_a_r - i_vsc_R1_a_r*v_R1_n_r - p_R1/3
        struct[0].g[89,0] = -1.0*i_vsc_R1_a_i*v_R1_a_r + 1.0*i_vsc_R1_a_i*v_R1_n_r + 1.0*i_vsc_R1_a_r*v_R1_a_i - 1.0*i_vsc_R1_a_r*v_R1_n_i - q_R1/3
        struct[0].g[90,0] = 1.0*i_vsc_R1_b_i*v_R1_b_i - 1.0*i_vsc_R1_b_i*v_R1_n_i + i_vsc_R1_b_r*v_R1_b_r - i_vsc_R1_b_r*v_R1_n_r - p_R1/3
        struct[0].g[91,0] = -1.0*i_vsc_R1_b_i*v_R1_b_r + 1.0*i_vsc_R1_b_i*v_R1_n_r + 1.0*i_vsc_R1_b_r*v_R1_b_i - 1.0*i_vsc_R1_b_r*v_R1_n_i - q_R1/3
        struct[0].g[92,0] = 1.0*i_vsc_R1_c_i*v_R1_c_i - 1.0*i_vsc_R1_c_i*v_R1_n_i + i_vsc_R1_c_r*v_R1_c_r - i_vsc_R1_c_r*v_R1_n_r - p_R1/3
        struct[0].g[93,0] = -1.0*i_vsc_R1_c_i*v_R1_c_r + 1.0*i_vsc_R1_c_i*v_R1_n_r + 1.0*i_vsc_R1_c_r*v_R1_c_i - 1.0*i_vsc_R1_c_r*v_R1_n_i - q_R1/3
        struct[0].g[94,0] = p_D1 + p_R1 + Piecewise(np.array([(-p_loss_R1, p_D1 < 0), (p_loss_R1, True)]))
        struct[0].g[95,0] = i_l_D1_D10_a_r*v_D1_a_r + i_l_D1_D10_n_r*v_D1_n_r - p_D1
        struct[0].g[96,0] = -a_R1 - b_R1*sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - c_R1*(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) + p_loss_R1
        struct[0].g[97,0] = -coef_a_R10*p_R10 + 1.0*i_vsc_R10_a_i*v_R10_a_i - 1.0*i_vsc_R10_a_i*v_R10_n_i + i_vsc_R10_a_r*v_R10_a_r - i_vsc_R10_a_r*v_R10_n_r
        struct[0].g[98,0] = -coef_a_R10*q_R10 - 1.0*i_vsc_R10_a_i*v_R10_a_r + 1.0*i_vsc_R10_a_i*v_R10_n_r + 1.0*i_vsc_R10_a_r*v_R10_a_i - 1.0*i_vsc_R10_a_r*v_R10_n_i
        struct[0].g[99,0] = -coef_b_R10*p_R10 + 1.0*i_vsc_R10_b_i*v_R10_b_i - 1.0*i_vsc_R10_b_i*v_R10_n_i + i_vsc_R10_b_r*v_R10_b_r - i_vsc_R10_b_r*v_R10_n_r
        struct[0].g[100,0] = -coef_b_R10*q_R10 - 1.0*i_vsc_R10_b_i*v_R10_b_r + 1.0*i_vsc_R10_b_i*v_R10_n_r + 1.0*i_vsc_R10_b_r*v_R10_b_i - 1.0*i_vsc_R10_b_r*v_R10_n_i
        struct[0].g[101,0] = -coef_c_R10*p_R10 + 1.0*i_vsc_R10_c_i*v_R10_c_i - 1.0*i_vsc_R10_c_i*v_R10_n_i + i_vsc_R10_c_r*v_R10_c_r - i_vsc_R10_c_r*v_R10_n_r
        struct[0].g[102,0] = -coef_c_R10*q_R10 - 1.0*i_vsc_R10_c_i*v_R10_c_r + 1.0*i_vsc_R10_c_i*v_R10_n_r + 1.0*i_vsc_R10_c_r*v_R10_c_i - 1.0*i_vsc_R10_c_r*v_R10_n_i
        struct[0].g[103,0] = i_vsc_D10_a_r + p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].g[104,0] = i_vsc_D10_n_r + p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].g[105,0] = p_D10 - p_R10 - Piecewise(np.array([(-p_loss_R10, p_D10 < 0), (p_loss_R10, True)]))
        struct[0].g[106,0] = -a_R10 - b_R10*sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - c_R10*(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) + p_loss_R10
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_R0_a_i**2 + v_R0_a_r**2)**0.5
        struct[0].h[1,0] = (v_R0_b_i**2 + v_R0_b_r**2)**0.5
        struct[0].h[2,0] = (v_R0_c_i**2 + v_R0_c_r**2)**0.5
        struct[0].h[3,0] = (v_D1_a_i**2 + v_D1_a_r**2)**0.5
        struct[0].h[4,0] = (v_D1_b_i**2 + v_D1_b_r**2)**0.5
        struct[0].h[5,0] = (v_D1_c_i**2 + v_D1_c_r**2)**0.5
        struct[0].h[6,0] = (v_R1_a_i**2 + v_R1_a_r**2)**0.5
        struct[0].h[7,0] = (v_R1_b_i**2 + v_R1_b_r**2)**0.5
        struct[0].h[8,0] = (v_R1_c_i**2 + v_R1_c_r**2)**0.5
        struct[0].h[9,0] = (v_R1_n_i**2 + v_R1_n_r**2)**0.5
        struct[0].h[10,0] = (v_R18_a_i**2 + v_R18_a_r**2)**0.5
        struct[0].h[11,0] = (v_R18_n_i**2 + v_R18_n_r**2)**0.5
        struct[0].h[12,0] = (v_D18_a_i**2 + v_D18_a_r**2)**0.5
        struct[0].h[13,0] = (v_D18_n_i**2 + v_D18_n_r**2)**0.5
        struct[0].h[14,0] = (v_R10_a_i**2 + v_R10_a_r**2)**0.5
        struct[0].h[15,0] = (v_R10_b_i**2 + v_R10_b_r**2)**0.5
        struct[0].h[16,0] = (v_R10_c_i**2 + v_R10_c_r**2)**0.5
        struct[0].h[17,0] = (v_R10_n_i**2 + v_R10_n_r**2)**0.5
        struct[0].h[18,0] = (v_R18_b_i**2 + v_R18_b_r**2)**0.5
        struct[0].h[19,0] = (v_R18_c_i**2 + v_R18_c_r**2)**0.5
        struct[0].h[20,0] = (v_D1_n_i**2 + v_D1_n_r**2)**0.5
        struct[0].h[21,0] = (v_D10_a_i**2 + v_D10_a_r**2)**0.5
        struct[0].h[22,0] = (v_D10_b_i**2 + v_D10_b_r**2)**0.5
        struct[0].h[23,0] = (v_D10_c_i**2 + v_D10_c_r**2)**0.5
        struct[0].h[24,0] = (v_D10_n_i**2 + v_D10_n_r**2)**0.5
        struct[0].h[25,0] = (v_D18_b_i**2 + v_D18_b_r**2)**0.5
        struct[0].h[26,0] = (v_D18_c_i**2 + v_D18_c_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx[0,0] = -1

    if mode == 11:


        struct[0].Gy[0,0] = -28.9395298724945
        struct[0].Gy[0,1] = -78.9359890415319
        struct[0].Gy[0,2] = 3.96392229058202
        struct[0].Gy[0,3] = 1.02713736253513
        struct[0].Gy[0,4] = 2.49575997948692
        struct[0].Gy[0,5] = 2.32849644809540
        struct[0].Gy[0,6] = 22.3462752317585
        struct[0].Gy[0,7] = 74.5565491272410
        struct[0].Gy[0,16] = 10.5571769313180
        struct[0].Gy[0,17] = 5.40657727682604
        struct[0].Gy[0,18] = -3.96392229058202
        struct[0].Gy[0,19] = -1.02713736253513
        struct[0].Gy[0,20] = -2.49575997948692
        struct[0].Gy[0,21] = -2.32849644809540
        struct[0].Gy[0,22] = -3.96392229058202
        struct[0].Gy[0,23] = -1.02713736253513
        struct[0].Gy[0,72] = 1
        struct[0].Gy[0,88] = 1
        struct[0].Gy[1,0] = 78.9359890415319
        struct[0].Gy[1,1] = -28.9395298724945
        struct[0].Gy[1,2] = -1.02713736253513
        struct[0].Gy[1,3] = 3.96392229058202
        struct[0].Gy[1,4] = -2.32849644809540
        struct[0].Gy[1,5] = 2.49575997948692
        struct[0].Gy[1,6] = -74.5565491272410
        struct[0].Gy[1,7] = 22.3462752317585
        struct[0].Gy[1,16] = -5.40657727682604
        struct[0].Gy[1,17] = 10.5571769313180
        struct[0].Gy[1,18] = 1.02713736253513
        struct[0].Gy[1,19] = -3.96392229058202
        struct[0].Gy[1,20] = 2.32849644809540
        struct[0].Gy[1,21] = -2.49575997948692
        struct[0].Gy[1,22] = 1.02713736253513
        struct[0].Gy[1,23] = -3.96392229058202
        struct[0].Gy[1,73] = 1
        struct[0].Gy[1,89] = 1
        struct[0].Gy[2,0] = 3.96392229058202
        struct[0].Gy[2,1] = 1.02713736253513
        struct[0].Gy[2,2] = -28.9395298724945
        struct[0].Gy[2,3] = -78.9359890415319
        struct[0].Gy[2,4] = 3.96392229058202
        struct[0].Gy[2,5] = 1.02713736253513
        struct[0].Gy[2,6] = 20.8781129206634
        struct[0].Gy[2,7] = 75.8579082128012
        struct[0].Gy[2,16] = -3.96392229058202
        struct[0].Gy[2,17] = -1.02713736253513
        struct[0].Gy[2,18] = 10.5571769313180
        struct[0].Gy[2,19] = 5.40657727682604
        struct[0].Gy[2,20] = -3.96392229058202
        struct[0].Gy[2,21] = -1.02713736253513
        struct[0].Gy[2,22] = -2.49575997948692
        struct[0].Gy[2,23] = -2.32849644809540
        struct[0].Gy[2,74] = 1
        struct[0].Gy[2,90] = 1
        struct[0].Gy[3,0] = -1.02713736253513
        struct[0].Gy[3,1] = 3.96392229058202
        struct[0].Gy[3,2] = 78.9359890415319
        struct[0].Gy[3,3] = -28.9395298724945
        struct[0].Gy[3,4] = -1.02713736253513
        struct[0].Gy[3,5] = 3.96392229058202
        struct[0].Gy[3,6] = -75.8579082128012
        struct[0].Gy[3,7] = 20.8781129206634
        struct[0].Gy[3,16] = 1.02713736253513
        struct[0].Gy[3,17] = -3.96392229058202
        struct[0].Gy[3,18] = -5.40657727682604
        struct[0].Gy[3,19] = 10.5571769313180
        struct[0].Gy[3,20] = 1.02713736253513
        struct[0].Gy[3,21] = -3.96392229058202
        struct[0].Gy[3,22] = 2.32849644809540
        struct[0].Gy[3,23] = -2.49575997948692
        struct[0].Gy[3,75] = 1
        struct[0].Gy[3,91] = 1
        struct[0].Gy[4,0] = 2.49575997948692
        struct[0].Gy[4,1] = 2.32849644809540
        struct[0].Gy[4,2] = 3.96392229058202
        struct[0].Gy[4,3] = 1.02713736253513
        struct[0].Gy[4,4] = -28.9395298724945
        struct[0].Gy[4,5] = -78.9359890415319
        struct[0].Gy[4,6] = 22.3462752317585
        struct[0].Gy[4,7] = 74.5565491272410
        struct[0].Gy[4,16] = -2.49575997948692
        struct[0].Gy[4,17] = -2.32849644809540
        struct[0].Gy[4,18] = -3.96392229058202
        struct[0].Gy[4,19] = -1.02713736253513
        struct[0].Gy[4,20] = 10.5571769313180
        struct[0].Gy[4,21] = 5.40657727682604
        struct[0].Gy[4,22] = -3.96392229058202
        struct[0].Gy[4,23] = -1.02713736253513
        struct[0].Gy[4,76] = 1
        struct[0].Gy[4,92] = 1
        struct[0].Gy[5,0] = -2.32849644809540
        struct[0].Gy[5,1] = 2.49575997948692
        struct[0].Gy[5,2] = -1.02713736253513
        struct[0].Gy[5,3] = 3.96392229058202
        struct[0].Gy[5,4] = 78.9359890415319
        struct[0].Gy[5,5] = -28.9395298724945
        struct[0].Gy[5,6] = -74.5565491272410
        struct[0].Gy[5,7] = 22.3462752317585
        struct[0].Gy[5,16] = 2.32849644809540
        struct[0].Gy[5,17] = -2.49575997948692
        struct[0].Gy[5,18] = 1.02713736253513
        struct[0].Gy[5,19] = -3.96392229058202
        struct[0].Gy[5,20] = -5.40657727682604
        struct[0].Gy[5,21] = 10.5571769313180
        struct[0].Gy[5,22] = 1.02713736253513
        struct[0].Gy[5,23] = -3.96392229058202
        struct[0].Gy[5,77] = 1
        struct[0].Gy[5,93] = 1
        struct[0].Gy[6,0] = 22.3462752317585
        struct[0].Gy[6,1] = 74.5565491272410
        struct[0].Gy[6,2] = 20.8781129206634
        struct[0].Gy[6,3] = 75.8579082128012
        struct[0].Gy[6,4] = 22.3462752317585
        struct[0].Gy[6,5] = 74.5565491272410
        struct[0].Gy[6,6] = -66.0375690881807
        struct[0].Gy[6,7] = -225.994812570944
        struct[0].Gy[6,16] = -3.96392229058202
        struct[0].Gy[6,17] = -1.02713736253513
        struct[0].Gy[6,18] = -2.49575997948692
        struct[0].Gy[6,19] = -2.32849644809540
        struct[0].Gy[6,20] = -3.96392229058202
        struct[0].Gy[6,21] = -1.02713736253513
        struct[0].Gy[6,22] = 10.5571769313180
        struct[0].Gy[6,23] = 5.40657727682604
        struct[0].Gy[7,0] = -74.5565491272410
        struct[0].Gy[7,1] = 22.3462752317585
        struct[0].Gy[7,2] = -75.8579082128012
        struct[0].Gy[7,3] = 20.8781129206634
        struct[0].Gy[7,4] = -74.5565491272410
        struct[0].Gy[7,5] = 22.3462752317585
        struct[0].Gy[7,6] = 225.994812570944
        struct[0].Gy[7,7] = -66.0375690881807
        struct[0].Gy[7,16] = 1.02713736253513
        struct[0].Gy[7,17] = -3.96392229058202
        struct[0].Gy[7,18] = 2.32849644809540
        struct[0].Gy[7,19] = -2.49575997948692
        struct[0].Gy[7,20] = 1.02713736253513
        struct[0].Gy[7,21] = -3.96392229058202
        struct[0].Gy[7,22] = -5.40657727682604
        struct[0].Gy[7,23] = 10.5571769313180
        struct[0].Gy[8,8] = -30.9517475172273
        struct[0].Gy[8,9] = -5.65456401516768
        struct[0].Gy[8,10] = 9.21038227100566
        struct[0].Gy[8,11] = -1.84896616921897
        struct[0].Gy[8,16] = 30.9517475172273
        struct[0].Gy[8,17] = 5.65456401516768
        struct[0].Gy[8,18] = -9.21038227100566
        struct[0].Gy[8,19] = 1.84896616921897
        struct[0].Gy[8,20] = -9.00835072044485
        struct[0].Gy[8,21] = 0.793238195499529
        struct[0].Gy[8,22] = -9.21038227100566
        struct[0].Gy[8,23] = 1.84896616921897
        struct[0].Gy[8,24] = 9.21038227100566
        struct[0].Gy[8,25] = -1.84896616921897
        struct[0].Gy[8,26] = 9.00835072044485
        struct[0].Gy[8,27] = -0.793238195499529
        struct[0].Gy[8,80] = 1
        struct[0].Gy[9,8] = 5.65456401516768
        struct[0].Gy[9,9] = -30.9517475172273
        struct[0].Gy[9,10] = 1.84896616921897
        struct[0].Gy[9,11] = 9.21038227100566
        struct[0].Gy[9,16] = -5.65456401516768
        struct[0].Gy[9,17] = 30.9517475172273
        struct[0].Gy[9,18] = -1.84896616921897
        struct[0].Gy[9,19] = -9.21038227100566
        struct[0].Gy[9,20] = -0.793238195499529
        struct[0].Gy[9,21] = -9.00835072044485
        struct[0].Gy[9,22] = -1.84896616921897
        struct[0].Gy[9,23] = -9.21038227100566
        struct[0].Gy[9,24] = 1.84896616921897
        struct[0].Gy[9,25] = 9.21038227100566
        struct[0].Gy[9,26] = 0.793238195499529
        struct[0].Gy[9,27] = 9.00835072044485
        struct[0].Gy[9,81] = 1
        struct[0].Gy[10,8] = 9.21038227100566
        struct[0].Gy[10,9] = -1.84896616921897
        struct[0].Gy[10,10] = -30.9767475172273
        struct[0].Gy[10,11] = -5.65456401516768
        struct[0].Gy[10,16] = -9.21038227100566
        struct[0].Gy[10,17] = 1.84896616921897
        struct[0].Gy[10,18] = -9.00835072044485
        struct[0].Gy[10,19] = 0.793238195499527
        struct[0].Gy[10,20] = -9.21038227100566
        struct[0].Gy[10,21] = 1.84896616921897
        struct[0].Gy[10,22] = 30.9517475172273
        struct[0].Gy[10,23] = 5.65456401516768
        struct[0].Gy[10,24] = 9.00835072044485
        struct[0].Gy[10,25] = -0.793238195499527
        struct[0].Gy[10,26] = 9.21038227100566
        struct[0].Gy[10,27] = -1.84896616921897
        struct[0].Gy[10,82] = 1
        struct[0].Gy[11,8] = 1.84896616921897
        struct[0].Gy[11,9] = 9.21038227100566
        struct[0].Gy[11,10] = 5.65456401516768
        struct[0].Gy[11,11] = -30.9767475172273
        struct[0].Gy[11,16] = -1.84896616921897
        struct[0].Gy[11,17] = -9.21038227100566
        struct[0].Gy[11,18] = -0.793238195499527
        struct[0].Gy[11,19] = -9.00835072044485
        struct[0].Gy[11,20] = -1.84896616921897
        struct[0].Gy[11,21] = -9.21038227100566
        struct[0].Gy[11,22] = -5.65456401516768
        struct[0].Gy[11,23] = 30.9517475172273
        struct[0].Gy[11,24] = 0.793238195499527
        struct[0].Gy[11,25] = 9.00835072044485
        struct[0].Gy[11,26] = 1.84896616921897
        struct[0].Gy[11,27] = 9.21038227100566
        struct[0].Gy[11,83] = 1
        struct[0].Gy[12,12] = -157.977883096366
        struct[0].Gy[12,30] = 157.977883096366
        struct[0].Gy[12,84] = 1
        struct[0].Gy[13,13] = -157.977883096366
        struct[0].Gy[13,31] = 157.977883096366
        struct[0].Gy[13,85] = 1
        struct[0].Gy[14,14] = -157.977883096366
        struct[0].Gy[14,36] = 157.977883096366
        struct[0].Gy[14,86] = 1
        struct[0].Gy[15,15] = -157.977883096366
        struct[0].Gy[15,37] = 157.977883096366
        struct[0].Gy[15,87] = 1
        struct[0].Gy[16,0] = 10.5571769313180
        struct[0].Gy[16,1] = 5.40657727682604
        struct[0].Gy[16,2] = -3.96392229058202
        struct[0].Gy[16,3] = -1.02713736253513
        struct[0].Gy[16,4] = -2.49575997948692
        struct[0].Gy[16,5] = -2.32849644809540
        struct[0].Gy[16,6] = -3.96392229058202
        struct[0].Gy[16,7] = -1.02713736253513
        struct[0].Gy[16,8] = 30.9517475172273
        struct[0].Gy[16,9] = 5.65456401516768
        struct[0].Gy[16,10] = -9.21038227100566
        struct[0].Gy[16,11] = 1.84896616921897
        struct[0].Gy[16,16] = -41.5089244485453
        struct[0].Gy[16,17] = -11.0611412919937
        struct[0].Gy[16,18] = 13.1743045615877
        struct[0].Gy[16,19] = -0.821828806683838
        struct[0].Gy[16,20] = 11.5041106999318
        struct[0].Gy[16,21] = 1.53525825259587
        struct[0].Gy[16,22] = 13.1743045615877
        struct[0].Gy[16,23] = -0.821828806683840
        struct[0].Gy[16,24] = -9.21038227100566
        struct[0].Gy[16,25] = 1.84896616921897
        struct[0].Gy[16,26] = -9.00835072044485
        struct[0].Gy[16,27] = 0.793238195499529
        struct[0].Gy[16,97] = 1
        struct[0].Gy[17,0] = -5.40657727682604
        struct[0].Gy[17,1] = 10.5571769313180
        struct[0].Gy[17,2] = 1.02713736253513
        struct[0].Gy[17,3] = -3.96392229058202
        struct[0].Gy[17,4] = 2.32849644809540
        struct[0].Gy[17,5] = -2.49575997948692
        struct[0].Gy[17,6] = 1.02713736253513
        struct[0].Gy[17,7] = -3.96392229058202
        struct[0].Gy[17,8] = -5.65456401516768
        struct[0].Gy[17,9] = 30.9517475172273
        struct[0].Gy[17,10] = -1.84896616921897
        struct[0].Gy[17,11] = -9.21038227100566
        struct[0].Gy[17,16] = 11.0611412919937
        struct[0].Gy[17,17] = -41.5089244485453
        struct[0].Gy[17,18] = 0.821828806683838
        struct[0].Gy[17,19] = 13.1743045615877
        struct[0].Gy[17,20] = -1.53525825259587
        struct[0].Gy[17,21] = 11.5041106999318
        struct[0].Gy[17,22] = 0.821828806683840
        struct[0].Gy[17,23] = 13.1743045615877
        struct[0].Gy[17,24] = -1.84896616921897
        struct[0].Gy[17,25] = -9.21038227100566
        struct[0].Gy[17,26] = -0.793238195499529
        struct[0].Gy[17,27] = -9.00835072044485
        struct[0].Gy[17,98] = 1
        struct[0].Gy[18,0] = -3.96392229058202
        struct[0].Gy[18,1] = -1.02713736253513
        struct[0].Gy[18,2] = 10.5571769313180
        struct[0].Gy[18,3] = 5.40657727682604
        struct[0].Gy[18,4] = -3.96392229058202
        struct[0].Gy[18,5] = -1.02713736253513
        struct[0].Gy[18,6] = -2.49575997948692
        struct[0].Gy[18,7] = -2.32849644809540
        struct[0].Gy[18,8] = -9.21038227100566
        struct[0].Gy[18,9] = 1.84896616921897
        struct[0].Gy[18,10] = -9.00835072044485
        struct[0].Gy[18,11] = 0.793238195499528
        struct[0].Gy[18,16] = 13.1743045615877
        struct[0].Gy[18,17] = -0.821828806683841
        struct[0].Gy[18,18] = -41.5089244485453
        struct[0].Gy[18,19] = -11.0611412919937
        struct[0].Gy[18,20] = 13.1743045615877
        struct[0].Gy[18,21] = -0.821828806683839
        struct[0].Gy[18,22] = 11.5041106999318
        struct[0].Gy[18,23] = 1.53525825259588
        struct[0].Gy[18,24] = 30.9517475172273
        struct[0].Gy[18,25] = 5.65456401516768
        struct[0].Gy[18,26] = -9.21038227100566
        struct[0].Gy[18,27] = 1.84896616921897
        struct[0].Gy[18,99] = 1
        struct[0].Gy[19,0] = 1.02713736253513
        struct[0].Gy[19,1] = -3.96392229058202
        struct[0].Gy[19,2] = -5.40657727682604
        struct[0].Gy[19,3] = 10.5571769313180
        struct[0].Gy[19,4] = 1.02713736253513
        struct[0].Gy[19,5] = -3.96392229058202
        struct[0].Gy[19,6] = 2.32849644809540
        struct[0].Gy[19,7] = -2.49575997948692
        struct[0].Gy[19,8] = -1.84896616921897
        struct[0].Gy[19,9] = -9.21038227100566
        struct[0].Gy[19,10] = -0.793238195499528
        struct[0].Gy[19,11] = -9.00835072044485
        struct[0].Gy[19,16] = 0.821828806683841
        struct[0].Gy[19,17] = 13.1743045615877
        struct[0].Gy[19,18] = 11.0611412919937
        struct[0].Gy[19,19] = -41.5089244485453
        struct[0].Gy[19,20] = 0.821828806683839
        struct[0].Gy[19,21] = 13.1743045615877
        struct[0].Gy[19,22] = -1.53525825259588
        struct[0].Gy[19,23] = 11.5041106999318
        struct[0].Gy[19,24] = -5.65456401516768
        struct[0].Gy[19,25] = 30.9517475172273
        struct[0].Gy[19,26] = -1.84896616921897
        struct[0].Gy[19,27] = -9.21038227100566
        struct[0].Gy[19,100] = 1
        struct[0].Gy[20,0] = -2.49575997948692
        struct[0].Gy[20,1] = -2.32849644809540
        struct[0].Gy[20,2] = -3.96392229058202
        struct[0].Gy[20,3] = -1.02713736253513
        struct[0].Gy[20,4] = 10.5571769313180
        struct[0].Gy[20,5] = 5.40657727682604
        struct[0].Gy[20,6] = -3.96392229058202
        struct[0].Gy[20,7] = -1.02713736253513
        struct[0].Gy[20,8] = -9.00835072044484
        struct[0].Gy[20,9] = 0.793238195499527
        struct[0].Gy[20,10] = -9.21038227100566
        struct[0].Gy[20,11] = 1.84896616921897
        struct[0].Gy[20,16] = 11.5041106999318
        struct[0].Gy[20,17] = 1.53525825259588
        struct[0].Gy[20,18] = 13.1743045615877
        struct[0].Gy[20,19] = -0.821828806683840
        struct[0].Gy[20,20] = -41.5089244485453
        struct[0].Gy[20,21] = -11.0611412919937
        struct[0].Gy[20,22] = 13.1743045615877
        struct[0].Gy[20,23] = -0.821828806683838
        struct[0].Gy[20,24] = -9.21038227100566
        struct[0].Gy[20,25] = 1.84896616921897
        struct[0].Gy[20,26] = 30.9517475172273
        struct[0].Gy[20,27] = 5.65456401516768
        struct[0].Gy[20,101] = 1
        struct[0].Gy[21,0] = 2.32849644809540
        struct[0].Gy[21,1] = -2.49575997948692
        struct[0].Gy[21,2] = 1.02713736253513
        struct[0].Gy[21,3] = -3.96392229058202
        struct[0].Gy[21,4] = -5.40657727682604
        struct[0].Gy[21,5] = 10.5571769313180
        struct[0].Gy[21,6] = 1.02713736253513
        struct[0].Gy[21,7] = -3.96392229058202
        struct[0].Gy[21,8] = -0.793238195499527
        struct[0].Gy[21,9] = -9.00835072044484
        struct[0].Gy[21,10] = -1.84896616921897
        struct[0].Gy[21,11] = -9.21038227100566
        struct[0].Gy[21,16] = -1.53525825259588
        struct[0].Gy[21,17] = 11.5041106999318
        struct[0].Gy[21,18] = 0.821828806683840
        struct[0].Gy[21,19] = 13.1743045615877
        struct[0].Gy[21,20] = 11.0611412919937
        struct[0].Gy[21,21] = -41.5089244485453
        struct[0].Gy[21,22] = 0.821828806683838
        struct[0].Gy[21,23] = 13.1743045615877
        struct[0].Gy[21,24] = -1.84896616921897
        struct[0].Gy[21,25] = -9.21038227100566
        struct[0].Gy[21,26] = -5.65456401516768
        struct[0].Gy[21,27] = 30.9517475172273
        struct[0].Gy[21,102] = 1
        struct[0].Gy[22,0] = -3.96392229058202
        struct[0].Gy[22,1] = -1.02713736253513
        struct[0].Gy[22,2] = -2.49575997948692
        struct[0].Gy[22,3] = -2.32849644809540
        struct[0].Gy[22,4] = -3.96392229058202
        struct[0].Gy[22,5] = -1.02713736253513
        struct[0].Gy[22,6] = 10.5571769313180
        struct[0].Gy[22,7] = 5.40657727682604
        struct[0].Gy[22,8] = -9.21038227100566
        struct[0].Gy[22,9] = 1.84896616921897
        struct[0].Gy[22,10] = 30.9517475172273
        struct[0].Gy[22,11] = 5.65456401516768
        struct[0].Gy[22,16] = 13.1743045615877
        struct[0].Gy[22,17] = -0.821828806683840
        struct[0].Gy[22,18] = 11.5041106999318
        struct[0].Gy[22,19] = 1.53525825259588
        struct[0].Gy[22,20] = 13.1743045615877
        struct[0].Gy[22,21] = -0.821828806683837
        struct[0].Gy[22,22] = -41.5339244485453
        struct[0].Gy[22,23] = -11.0611412919937
        struct[0].Gy[22,24] = -9.00835072044485
        struct[0].Gy[22,25] = 0.793238195499527
        struct[0].Gy[22,26] = -9.21038227100566
        struct[0].Gy[22,27] = 1.84896616921897
        struct[0].Gy[23,0] = 1.02713736253513
        struct[0].Gy[23,1] = -3.96392229058202
        struct[0].Gy[23,2] = 2.32849644809540
        struct[0].Gy[23,3] = -2.49575997948692
        struct[0].Gy[23,4] = 1.02713736253513
        struct[0].Gy[23,5] = -3.96392229058202
        struct[0].Gy[23,6] = -5.40657727682604
        struct[0].Gy[23,7] = 10.5571769313180
        struct[0].Gy[23,8] = -1.84896616921897
        struct[0].Gy[23,9] = -9.21038227100566
        struct[0].Gy[23,10] = -5.65456401516768
        struct[0].Gy[23,11] = 30.9517475172273
        struct[0].Gy[23,16] = 0.821828806683840
        struct[0].Gy[23,17] = 13.1743045615877
        struct[0].Gy[23,18] = -1.53525825259588
        struct[0].Gy[23,19] = 11.5041106999318
        struct[0].Gy[23,20] = 0.821828806683837
        struct[0].Gy[23,21] = 13.1743045615877
        struct[0].Gy[23,22] = 11.0611412919937
        struct[0].Gy[23,23] = -41.5339244485453
        struct[0].Gy[23,24] = -0.793238195499527
        struct[0].Gy[23,25] = -9.00835072044485
        struct[0].Gy[23,26] = -1.84896616921897
        struct[0].Gy[23,27] = -9.21038227100566
        struct[0].Gy[24,8] = 9.21038227100566
        struct[0].Gy[24,9] = -1.84896616921897
        struct[0].Gy[24,10] = 9.00835072044485
        struct[0].Gy[24,11] = -0.793238195499528
        struct[0].Gy[24,16] = -9.21038227100566
        struct[0].Gy[24,17] = 1.84896616921897
        struct[0].Gy[24,18] = 30.9517475172273
        struct[0].Gy[24,19] = 5.65456401516768
        struct[0].Gy[24,20] = -9.21038227100566
        struct[0].Gy[24,21] = 1.84896616921897
        struct[0].Gy[24,22] = -9.00835072044485
        struct[0].Gy[24,23] = 0.793238195499528
        struct[0].Gy[24,24] = -30.9517475172273
        struct[0].Gy[24,25] = -5.65456401516768
        struct[0].Gy[24,26] = 9.21038227100566
        struct[0].Gy[24,27] = -1.84896616921897
        struct[0].Gy[25,8] = 1.84896616921897
        struct[0].Gy[25,9] = 9.21038227100566
        struct[0].Gy[25,10] = 0.793238195499528
        struct[0].Gy[25,11] = 9.00835072044485
        struct[0].Gy[25,16] = -1.84896616921897
        struct[0].Gy[25,17] = -9.21038227100566
        struct[0].Gy[25,18] = -5.65456401516768
        struct[0].Gy[25,19] = 30.9517475172273
        struct[0].Gy[25,20] = -1.84896616921897
        struct[0].Gy[25,21] = -9.21038227100566
        struct[0].Gy[25,22] = -0.793238195499528
        struct[0].Gy[25,23] = -9.00835072044485
        struct[0].Gy[25,24] = 5.65456401516768
        struct[0].Gy[25,25] = -30.9517475172273
        struct[0].Gy[25,26] = 1.84896616921897
        struct[0].Gy[25,27] = 9.21038227100566
        struct[0].Gy[26,8] = 9.00835072044484
        struct[0].Gy[26,9] = -0.793238195499527
        struct[0].Gy[26,10] = 9.21038227100566
        struct[0].Gy[26,11] = -1.84896616921897
        struct[0].Gy[26,16] = -9.00835072044484
        struct[0].Gy[26,17] = 0.793238195499527
        struct[0].Gy[26,18] = -9.21038227100566
        struct[0].Gy[26,19] = 1.84896616921897
        struct[0].Gy[26,20] = 30.9517475172273
        struct[0].Gy[26,21] = 5.65456401516768
        struct[0].Gy[26,22] = -9.21038227100566
        struct[0].Gy[26,23] = 1.84896616921897
        struct[0].Gy[26,24] = 9.21038227100566
        struct[0].Gy[26,25] = -1.84896616921897
        struct[0].Gy[26,26] = -30.9517475172273
        struct[0].Gy[26,27] = -5.65456401516768
        struct[0].Gy[27,8] = 0.793238195499527
        struct[0].Gy[27,9] = 9.00835072044484
        struct[0].Gy[27,10] = 1.84896616921897
        struct[0].Gy[27,11] = 9.21038227100566
        struct[0].Gy[27,16] = -0.793238195499527
        struct[0].Gy[27,17] = -9.00835072044484
        struct[0].Gy[27,18] = -1.84896616921897
        struct[0].Gy[27,19] = -9.21038227100566
        struct[0].Gy[27,20] = -5.65456401516768
        struct[0].Gy[27,21] = 30.9517475172273
        struct[0].Gy[27,22] = -1.84896616921897
        struct[0].Gy[27,23] = -9.21038227100566
        struct[0].Gy[27,24] = 1.84896616921897
        struct[0].Gy[27,25] = 9.21038227100566
        struct[0].Gy[27,26] = 5.65456401516768
        struct[0].Gy[27,27] = -30.9517475172273
        struct[0].Gy[28,28] = -1067.70480704130
        struct[0].Gy[28,36] = 67.7048070412999
        struct[0].Gy[29,29] = -1067.70480704130
        struct[0].Gy[29,37] = 67.7048070412999
        struct[0].Gy[30,12] = 157.977883096366
        struct[0].Gy[30,30] = -225.682690137666
        struct[0].Gy[30,103] = 1
        struct[0].Gy[31,13] = 157.977883096366
        struct[0].Gy[31,31] = -225.682690137666
        struct[0].Gy[32,32] = -225.682690137666
        struct[0].Gy[32,38] = 157.977883096366
        struct[0].Gy[33,33] = -225.682690137666
        struct[0].Gy[33,39] = 157.977883096366
        struct[0].Gy[34,34] = -225.682690137666
        struct[0].Gy[34,40] = 157.977883096366
        struct[0].Gy[35,35] = -225.682690137666
        struct[0].Gy[35,41] = 157.977883096366
        struct[0].Gy[36,14] = 157.977883096366
        struct[0].Gy[36,28] = 67.7048070412999
        struct[0].Gy[36,36] = -225.682690137666
        struct[0].Gy[36,104] = 1
        struct[0].Gy[37,15] = 157.977883096366
        struct[0].Gy[37,29] = 67.7048070412999
        struct[0].Gy[37,37] = -225.682690137666
        struct[0].Gy[38,32] = 157.977883096366
        struct[0].Gy[38,38] = -157.977883096366
        struct[0].Gy[39,33] = 157.977883096366
        struct[0].Gy[39,39] = -157.977883096366
        struct[0].Gy[40,34] = 157.977883096366
        struct[0].Gy[40,40] = -157.977883096366
        struct[0].Gy[41,35] = 157.977883096366
        struct[0].Gy[41,41] = -157.977883096366
        struct[0].Gy[42,0] = -0.212261128378539
        struct[0].Gy[42,1] = -0.849044513514155
        struct[0].Gy[42,2] = 0.212261128378539
        struct[0].Gy[42,3] = 0.849044513514155
        struct[0].Gy[42,42] = -1
        struct[0].Gy[43,0] = 0.849044513514155
        struct[0].Gy[43,1] = -0.212261128378539
        struct[0].Gy[43,2] = -0.849044513514155
        struct[0].Gy[43,3] = 0.212261128378539
        struct[0].Gy[43,43] = -1
        struct[0].Gy[44,2] = -0.212261128378539
        struct[0].Gy[44,3] = -0.849044513514155
        struct[0].Gy[44,4] = 0.212261128378539
        struct[0].Gy[44,5] = 0.849044513514155
        struct[0].Gy[44,44] = -1
        struct[0].Gy[45,2] = 0.849044513514155
        struct[0].Gy[45,3] = -0.212261128378539
        struct[0].Gy[45,4] = -0.849044513514155
        struct[0].Gy[45,5] = 0.212261128378539
        struct[0].Gy[45,45] = -1
        struct[0].Gy[46,0] = 0.212261128378539
        struct[0].Gy[46,1] = 0.849044513514155
        struct[0].Gy[46,4] = -0.212261128378539
        struct[0].Gy[46,5] = -0.849044513514155
        struct[0].Gy[46,46] = -1
        struct[0].Gy[47,0] = -0.849044513514155
        struct[0].Gy[47,1] = 0.212261128378539
        struct[0].Gy[47,4] = 0.849044513514155
        struct[0].Gy[47,5] = -0.212261128378539
        struct[0].Gy[47,47] = -1
        struct[0].Gy[48,0] = 10.5571769313180
        struct[0].Gy[48,1] = 5.40657727682604
        struct[0].Gy[48,2] = -3.96392229058202
        struct[0].Gy[48,3] = -1.02713736253513
        struct[0].Gy[48,4] = -2.49575997948692
        struct[0].Gy[48,5] = -2.32849644809540
        struct[0].Gy[48,6] = -3.96392229058202
        struct[0].Gy[48,7] = -1.02713736253513
        struct[0].Gy[48,16] = -10.5571769313180
        struct[0].Gy[48,17] = -5.40657727682604
        struct[0].Gy[48,18] = 3.96392229058202
        struct[0].Gy[48,19] = 1.02713736253513
        struct[0].Gy[48,20] = 2.49575997948692
        struct[0].Gy[48,21] = 2.32849644809540
        struct[0].Gy[48,22] = 3.96392229058202
        struct[0].Gy[48,23] = 1.02713736253513
        struct[0].Gy[48,48] = -1
        struct[0].Gy[49,0] = -5.40657727682604
        struct[0].Gy[49,1] = 10.5571769313180
        struct[0].Gy[49,2] = 1.02713736253513
        struct[0].Gy[49,3] = -3.96392229058202
        struct[0].Gy[49,4] = 2.32849644809540
        struct[0].Gy[49,5] = -2.49575997948692
        struct[0].Gy[49,6] = 1.02713736253513
        struct[0].Gy[49,7] = -3.96392229058202
        struct[0].Gy[49,16] = 5.40657727682604
        struct[0].Gy[49,17] = -10.5571769313180
        struct[0].Gy[49,18] = -1.02713736253513
        struct[0].Gy[49,19] = 3.96392229058202
        struct[0].Gy[49,20] = -2.32849644809540
        struct[0].Gy[49,21] = 2.49575997948692
        struct[0].Gy[49,22] = -1.02713736253513
        struct[0].Gy[49,23] = 3.96392229058202
        struct[0].Gy[49,49] = -1
        struct[0].Gy[50,0] = -3.96392229058202
        struct[0].Gy[50,1] = -1.02713736253513
        struct[0].Gy[50,2] = 10.5571769313180
        struct[0].Gy[50,3] = 5.40657727682604
        struct[0].Gy[50,4] = -3.96392229058202
        struct[0].Gy[50,5] = -1.02713736253513
        struct[0].Gy[50,6] = -2.49575997948692
        struct[0].Gy[50,7] = -2.32849644809540
        struct[0].Gy[50,16] = 3.96392229058202
        struct[0].Gy[50,17] = 1.02713736253513
        struct[0].Gy[50,18] = -10.5571769313180
        struct[0].Gy[50,19] = -5.40657727682604
        struct[0].Gy[50,20] = 3.96392229058202
        struct[0].Gy[50,21] = 1.02713736253513
        struct[0].Gy[50,22] = 2.49575997948692
        struct[0].Gy[50,23] = 2.32849644809540
        struct[0].Gy[50,50] = -1
        struct[0].Gy[51,0] = 1.02713736253513
        struct[0].Gy[51,1] = -3.96392229058202
        struct[0].Gy[51,2] = -5.40657727682604
        struct[0].Gy[51,3] = 10.5571769313180
        struct[0].Gy[51,4] = 1.02713736253513
        struct[0].Gy[51,5] = -3.96392229058202
        struct[0].Gy[51,6] = 2.32849644809540
        struct[0].Gy[51,7] = -2.49575997948692
        struct[0].Gy[51,16] = -1.02713736253513
        struct[0].Gy[51,17] = 3.96392229058202
        struct[0].Gy[51,18] = 5.40657727682604
        struct[0].Gy[51,19] = -10.5571769313180
        struct[0].Gy[51,20] = -1.02713736253513
        struct[0].Gy[51,21] = 3.96392229058202
        struct[0].Gy[51,22] = -2.32849644809540
        struct[0].Gy[51,23] = 2.49575997948692
        struct[0].Gy[51,51] = -1
        struct[0].Gy[52,0] = -2.49575997948692
        struct[0].Gy[52,1] = -2.32849644809540
        struct[0].Gy[52,2] = -3.96392229058202
        struct[0].Gy[52,3] = -1.02713736253513
        struct[0].Gy[52,4] = 10.5571769313180
        struct[0].Gy[52,5] = 5.40657727682604
        struct[0].Gy[52,6] = -3.96392229058202
        struct[0].Gy[52,7] = -1.02713736253513
        struct[0].Gy[52,16] = 2.49575997948692
        struct[0].Gy[52,17] = 2.32849644809540
        struct[0].Gy[52,18] = 3.96392229058202
        struct[0].Gy[52,19] = 1.02713736253513
        struct[0].Gy[52,20] = -10.5571769313180
        struct[0].Gy[52,21] = -5.40657727682604
        struct[0].Gy[52,22] = 3.96392229058202
        struct[0].Gy[52,23] = 1.02713736253513
        struct[0].Gy[52,52] = -1
        struct[0].Gy[53,0] = 2.32849644809540
        struct[0].Gy[53,1] = -2.49575997948692
        struct[0].Gy[53,2] = 1.02713736253513
        struct[0].Gy[53,3] = -3.96392229058202
        struct[0].Gy[53,4] = -5.40657727682604
        struct[0].Gy[53,5] = 10.5571769313180
        struct[0].Gy[53,6] = 1.02713736253513
        struct[0].Gy[53,7] = -3.96392229058202
        struct[0].Gy[53,16] = -2.32849644809540
        struct[0].Gy[53,17] = 2.49575997948692
        struct[0].Gy[53,18] = -1.02713736253513
        struct[0].Gy[53,19] = 3.96392229058202
        struct[0].Gy[53,20] = 5.40657727682604
        struct[0].Gy[53,21] = -10.5571769313180
        struct[0].Gy[53,22] = -1.02713736253513
        struct[0].Gy[53,23] = 3.96392229058202
        struct[0].Gy[53,53] = -1
        struct[0].Gy[54,48] = 1
        struct[0].Gy[54,50] = 1
        struct[0].Gy[54,52] = 1
        struct[0].Gy[54,54] = -1
        struct[0].Gy[55,49] = 1
        struct[0].Gy[55,51] = 1
        struct[0].Gy[55,53] = 1
        struct[0].Gy[55,55] = -1
        struct[0].Gy[56,30] = -67.7048070412999
        struct[0].Gy[56,56] = -1
        struct[0].Gy[57,31] = -67.7048070412999
        struct[0].Gy[57,57] = -1
        struct[0].Gy[58,32] = -67.7048070412999
        struct[0].Gy[58,58] = -1
        struct[0].Gy[59,33] = -67.7048070412999
        struct[0].Gy[59,59] = -1
        struct[0].Gy[60,34] = -67.7048070412999
        struct[0].Gy[60,60] = -1
        struct[0].Gy[61,35] = -67.7048070412999
        struct[0].Gy[61,61] = -1
        struct[0].Gy[62,56] = 1
        struct[0].Gy[62,58] = 1
        struct[0].Gy[62,60] = 1
        struct[0].Gy[62,62] = -1
        struct[0].Gy[63,57] = 1
        struct[0].Gy[63,59] = 1
        struct[0].Gy[63,61] = 1
        struct[0].Gy[63,63] = -1
        struct[0].Gy[64,12] = -157.977883096366
        struct[0].Gy[64,30] = 157.977883096366
        struct[0].Gy[64,64] = -1
        struct[0].Gy[65,13] = -157.977883096366
        struct[0].Gy[65,31] = 157.977883096366
        struct[0].Gy[65,65] = -1
        struct[0].Gy[66,32] = 157.977883096366
        struct[0].Gy[66,38] = -157.977883096366
        struct[0].Gy[66,66] = -1
        struct[0].Gy[67,33] = 157.977883096366
        struct[0].Gy[67,39] = -157.977883096366
        struct[0].Gy[67,67] = -1
        struct[0].Gy[68,34] = 157.977883096366
        struct[0].Gy[68,40] = -157.977883096366
        struct[0].Gy[68,68] = -1
        struct[0].Gy[69,35] = 157.977883096366
        struct[0].Gy[69,41] = -157.977883096366
        struct[0].Gy[69,69] = -1
        struct[0].Gy[70,64] = 1
        struct[0].Gy[70,66] = 1
        struct[0].Gy[70,68] = 1
        struct[0].Gy[70,70] = -1
        struct[0].Gy[71,65] = 1
        struct[0].Gy[71,67] = 1
        struct[0].Gy[71,69] = 1
        struct[0].Gy[71,71] = -1
        struct[0].Gy[72,0] = i_load_R1_a_r
        struct[0].Gy[72,1] = i_load_R1_a_i
        struct[0].Gy[72,6] = -i_load_R1_a_r
        struct[0].Gy[72,7] = -i_load_R1_a_i
        struct[0].Gy[72,72] = v_R1_a_r - v_R1_n_r
        struct[0].Gy[72,73] = v_R1_a_i - v_R1_n_i
        struct[0].Gy[73,2] = i_load_R1_b_r
        struct[0].Gy[73,3] = i_load_R1_b_i
        struct[0].Gy[73,6] = -i_load_R1_b_r
        struct[0].Gy[73,7] = -i_load_R1_b_i
        struct[0].Gy[73,74] = v_R1_b_r - v_R1_n_r
        struct[0].Gy[73,75] = v_R1_b_i - v_R1_n_i
        struct[0].Gy[74,4] = i_load_R1_c_r
        struct[0].Gy[74,5] = i_load_R1_c_i
        struct[0].Gy[74,6] = -i_load_R1_c_r
        struct[0].Gy[74,7] = -i_load_R1_c_i
        struct[0].Gy[74,76] = v_R1_c_r - v_R1_n_r
        struct[0].Gy[74,77] = v_R1_c_i - v_R1_n_i
        struct[0].Gy[75,0] = -i_load_R1_a_i
        struct[0].Gy[75,1] = i_load_R1_a_r
        struct[0].Gy[75,6] = i_load_R1_a_i
        struct[0].Gy[75,7] = -i_load_R1_a_r
        struct[0].Gy[75,72] = v_R1_a_i - v_R1_n_i
        struct[0].Gy[75,73] = -v_R1_a_r + v_R1_n_r
        struct[0].Gy[76,2] = -i_load_R1_b_i
        struct[0].Gy[76,3] = i_load_R1_b_r
        struct[0].Gy[76,6] = i_load_R1_b_i
        struct[0].Gy[76,7] = -i_load_R1_b_r
        struct[0].Gy[76,74] = v_R1_b_i - v_R1_n_i
        struct[0].Gy[76,75] = -v_R1_b_r + v_R1_n_r
        struct[0].Gy[77,4] = -i_load_R1_c_i
        struct[0].Gy[77,5] = i_load_R1_c_r
        struct[0].Gy[77,6] = i_load_R1_c_i
        struct[0].Gy[77,7] = -i_load_R1_c_r
        struct[0].Gy[77,76] = v_R1_c_i - v_R1_n_i
        struct[0].Gy[77,77] = -v_R1_c_r + v_R1_n_r
        struct[0].Gy[78,72] = 1
        struct[0].Gy[78,74] = 1
        struct[0].Gy[78,76] = 1
        struct[0].Gy[78,78] = 1
        struct[0].Gy[79,73] = 1
        struct[0].Gy[79,75] = 1
        struct[0].Gy[79,77] = 1
        struct[0].Gy[79,79] = 1
        struct[0].Gy[80,8] = i_load_R18_a_r
        struct[0].Gy[80,9] = 1.0*i_load_R18_a_i
        struct[0].Gy[80,10] = -i_load_R18_a_r
        struct[0].Gy[80,11] = -1.0*i_load_R18_a_i
        struct[0].Gy[80,80] = v_R18_a_r - v_R18_n_r
        struct[0].Gy[80,81] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy[81,8] = -1.0*i_load_R18_a_i
        struct[0].Gy[81,9] = 1.0*i_load_R18_a_r
        struct[0].Gy[81,10] = 1.0*i_load_R18_a_i
        struct[0].Gy[81,11] = -1.0*i_load_R18_a_r
        struct[0].Gy[81,80] = 1.0*v_R18_a_i - 1.0*v_R18_n_i
        struct[0].Gy[81,81] = -1.0*v_R18_a_r + 1.0*v_R18_n_r
        struct[0].Gy[82,80] = 1
        struct[0].Gy[82,82] = 1
        struct[0].Gy[83,81] = 1.00000000000000
        struct[0].Gy[83,83] = 1.00000000000000
        struct[0].Gy[84,12] = i_load_D18_a_r
        struct[0].Gy[84,13] = 1.0*i_load_D18_a_i
        struct[0].Gy[84,14] = -i_load_D18_a_r
        struct[0].Gy[84,15] = -1.0*i_load_D18_a_i
        struct[0].Gy[84,84] = v_D18_a_r - v_D18_n_r
        struct[0].Gy[84,85] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy[85,12] = -1.0*i_load_D18_a_i
        struct[0].Gy[85,13] = 1.0*i_load_D18_a_r
        struct[0].Gy[85,14] = 1.0*i_load_D18_a_i
        struct[0].Gy[85,15] = -1.0*i_load_D18_a_r
        struct[0].Gy[85,84] = 1.0*v_D18_a_i - 1.0*v_D18_n_i
        struct[0].Gy[85,85] = -1.0*v_D18_a_r + 1.0*v_D18_n_r
        struct[0].Gy[86,84] = 1
        struct[0].Gy[86,86] = 1
        struct[0].Gy[87,85] = 1.00000000000000
        struct[0].Gy[87,87] = 1.00000000000000
        struct[0].Gy[88,0] = i_vsc_R1_a_r
        struct[0].Gy[88,1] = 1.0*i_vsc_R1_a_i
        struct[0].Gy[88,6] = -i_vsc_R1_a_r
        struct[0].Gy[88,7] = -1.0*i_vsc_R1_a_i
        struct[0].Gy[88,88] = v_R1_a_r - v_R1_n_r
        struct[0].Gy[88,89] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy[88,94] = -1/3
        struct[0].Gy[89,0] = -1.0*i_vsc_R1_a_i
        struct[0].Gy[89,1] = 1.0*i_vsc_R1_a_r
        struct[0].Gy[89,6] = 1.0*i_vsc_R1_a_i
        struct[0].Gy[89,7] = -1.0*i_vsc_R1_a_r
        struct[0].Gy[89,88] = 1.0*v_R1_a_i - 1.0*v_R1_n_i
        struct[0].Gy[89,89] = -1.0*v_R1_a_r + 1.0*v_R1_n_r
        struct[0].Gy[90,2] = i_vsc_R1_b_r
        struct[0].Gy[90,3] = 1.0*i_vsc_R1_b_i
        struct[0].Gy[90,6] = -i_vsc_R1_b_r
        struct[0].Gy[90,7] = -1.0*i_vsc_R1_b_i
        struct[0].Gy[90,90] = v_R1_b_r - v_R1_n_r
        struct[0].Gy[90,91] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy[90,94] = -1/3
        struct[0].Gy[91,2] = -1.0*i_vsc_R1_b_i
        struct[0].Gy[91,3] = 1.0*i_vsc_R1_b_r
        struct[0].Gy[91,6] = 1.0*i_vsc_R1_b_i
        struct[0].Gy[91,7] = -1.0*i_vsc_R1_b_r
        struct[0].Gy[91,90] = 1.0*v_R1_b_i - 1.0*v_R1_n_i
        struct[0].Gy[91,91] = -1.0*v_R1_b_r + 1.0*v_R1_n_r
        struct[0].Gy[92,4] = i_vsc_R1_c_r
        struct[0].Gy[92,5] = 1.0*i_vsc_R1_c_i
        struct[0].Gy[92,6] = -i_vsc_R1_c_r
        struct[0].Gy[92,7] = -1.0*i_vsc_R1_c_i
        struct[0].Gy[92,92] = v_R1_c_r - v_R1_n_r
        struct[0].Gy[92,93] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy[92,94] = -1/3
        struct[0].Gy[93,4] = -1.0*i_vsc_R1_c_i
        struct[0].Gy[93,5] = 1.0*i_vsc_R1_c_r
        struct[0].Gy[93,6] = 1.0*i_vsc_R1_c_i
        struct[0].Gy[93,7] = -1.0*i_vsc_R1_c_r
        struct[0].Gy[93,92] = 1.0*v_R1_c_i - 1.0*v_R1_n_i
        struct[0].Gy[93,93] = -1.0*v_R1_c_r + 1.0*v_R1_n_r
        struct[0].Gy[94,94] = 1
        struct[0].Gy[94,95] = 1
        struct[0].Gy[94,96] = Piecewise(np.array([(-1, p_D1 < 0), (1, True)]))
        struct[0].Gy[95,56] = v_D1_a_r
        struct[0].Gy[95,62] = v_D1_n_r
        struct[0].Gy[95,95] = -1
        struct[0].Gy[96,88] = -b_R1*i_vsc_R1_a_r/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_r
        struct[0].Gy[96,89] = -b_R1*i_vsc_R1_a_i/sqrt(i_vsc_R1_a_i**2 + i_vsc_R1_a_r**2 + 0.1) - 2*c_R1*i_vsc_R1_a_i
        struct[0].Gy[96,96] = 1
        struct[0].Gy[97,16] = i_vsc_R10_a_r
        struct[0].Gy[97,17] = 1.0*i_vsc_R10_a_i
        struct[0].Gy[97,22] = -i_vsc_R10_a_r
        struct[0].Gy[97,23] = -1.0*i_vsc_R10_a_i
        struct[0].Gy[97,97] = v_R10_a_r - v_R10_n_r
        struct[0].Gy[97,98] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy[98,16] = -1.0*i_vsc_R10_a_i
        struct[0].Gy[98,17] = 1.0*i_vsc_R10_a_r
        struct[0].Gy[98,22] = 1.0*i_vsc_R10_a_i
        struct[0].Gy[98,23] = -1.0*i_vsc_R10_a_r
        struct[0].Gy[98,97] = 1.0*v_R10_a_i - 1.0*v_R10_n_i
        struct[0].Gy[98,98] = -1.0*v_R10_a_r + 1.0*v_R10_n_r
        struct[0].Gy[99,18] = i_vsc_R10_b_r
        struct[0].Gy[99,19] = 1.0*i_vsc_R10_b_i
        struct[0].Gy[99,22] = -i_vsc_R10_b_r
        struct[0].Gy[99,23] = -1.0*i_vsc_R10_b_i
        struct[0].Gy[99,99] = v_R10_b_r - v_R10_n_r
        struct[0].Gy[99,100] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy[100,18] = -1.0*i_vsc_R10_b_i
        struct[0].Gy[100,19] = 1.0*i_vsc_R10_b_r
        struct[0].Gy[100,22] = 1.0*i_vsc_R10_b_i
        struct[0].Gy[100,23] = -1.0*i_vsc_R10_b_r
        struct[0].Gy[100,99] = 1.0*v_R10_b_i - 1.0*v_R10_n_i
        struct[0].Gy[100,100] = -1.0*v_R10_b_r + 1.0*v_R10_n_r
        struct[0].Gy[101,20] = i_vsc_R10_c_r
        struct[0].Gy[101,21] = 1.0*i_vsc_R10_c_i
        struct[0].Gy[101,22] = -i_vsc_R10_c_r
        struct[0].Gy[101,23] = -1.0*i_vsc_R10_c_i
        struct[0].Gy[101,101] = v_R10_c_r - v_R10_n_r
        struct[0].Gy[101,102] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy[102,20] = -1.0*i_vsc_R10_c_i
        struct[0].Gy[102,21] = 1.0*i_vsc_R10_c_r
        struct[0].Gy[102,22] = 1.0*i_vsc_R10_c_i
        struct[0].Gy[102,23] = -1.0*i_vsc_R10_c_r
        struct[0].Gy[102,101] = 1.0*v_R10_c_i - 1.0*v_R10_n_i
        struct[0].Gy[102,102] = -1.0*v_R10_c_r + 1.0*v_R10_n_r
        struct[0].Gy[103,30] = -p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[103,36] = p_D10/(v_D10_a_r - v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[103,103] = 1
        struct[0].Gy[103,105] = 1/(v_D10_a_r - v_D10_n_r + 1.0e-8)
        struct[0].Gy[104,30] = p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[104,36] = -p_D10/(-v_D10_a_r + v_D10_n_r + 1.0e-8)**2
        struct[0].Gy[104,104] = 1
        struct[0].Gy[104,105] = 1/(-v_D10_a_r + v_D10_n_r + 1.0e-8)
        struct[0].Gy[105,105] = 1
        struct[0].Gy[105,106] = -Piecewise(np.array([(-1, p_D10 < 0), (1, True)]))
        struct[0].Gy[106,97] = -b_R10*i_vsc_R10_a_r/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_r
        struct[0].Gy[106,98] = -b_R10*i_vsc_R10_a_i/sqrt(i_vsc_R10_a_i**2 + i_vsc_R10_a_r**2 + 0.1) - 2*c_R10*i_vsc_R10_a_i
        struct[0].Gy[106,106] = 1

        struct[0].Gu[0,0] = 0.212261128378539
        struct[0].Gu[0,1] = 0.849044513514155
        struct[0].Gu[0,4] = -0.212261128378539
        struct[0].Gu[0,5] = -0.849044513514155
        struct[0].Gu[1,0] = -0.849044513514155
        struct[0].Gu[1,1] = 0.212261128378539
        struct[0].Gu[1,4] = 0.849044513514155
        struct[0].Gu[1,5] = -0.212261128378539
        struct[0].Gu[2,0] = -0.212261128378539
        struct[0].Gu[2,1] = -0.849044513514155
        struct[0].Gu[2,2] = 0.212261128378539
        struct[0].Gu[2,3] = 0.849044513514155
        struct[0].Gu[3,0] = 0.849044513514155
        struct[0].Gu[3,1] = -0.212261128378539
        struct[0].Gu[3,2] = -0.849044513514155
        struct[0].Gu[3,3] = 0.212261128378539
        struct[0].Gu[4,2] = -0.212261128378539
        struct[0].Gu[4,3] = -0.849044513514155
        struct[0].Gu[4,4] = 0.212261128378539
        struct[0].Gu[4,5] = 0.849044513514155
        struct[0].Gu[5,2] = 0.849044513514155
        struct[0].Gu[5,3] = -0.212261128378539
        struct[0].Gu[5,4] = -0.849044513514155
        struct[0].Gu[5,5] = 0.212261128378539
        struct[0].Gu[30,6] = 67.7048070412999
        struct[0].Gu[31,7] = 67.7048070412999
        struct[0].Gu[32,8] = 67.7048070412999
        struct[0].Gu[33,9] = 67.7048070412999
        struct[0].Gu[34,10] = 67.7048070412999
        struct[0].Gu[35,11] = 67.7048070412999
        struct[0].Gu[42,0] = 0.00490196078431373
        struct[0].Gu[42,1] = 0.0196078431372549
        struct[0].Gu[42,2] = -0.00245098039215686
        struct[0].Gu[42,3] = -0.00980392156862745
        struct[0].Gu[42,4] = -0.00245098039215686
        struct[0].Gu[42,5] = -0.00980392156862745
        struct[0].Gu[43,0] = -0.0196078431372549
        struct[0].Gu[43,1] = 0.00490196078431373
        struct[0].Gu[43,2] = 0.00980392156862745
        struct[0].Gu[43,3] = -0.00245098039215686
        struct[0].Gu[43,4] = 0.00980392156862745
        struct[0].Gu[43,5] = -0.00245098039215686
        struct[0].Gu[44,0] = -0.00245098039215686
        struct[0].Gu[44,1] = -0.00980392156862745
        struct[0].Gu[44,2] = 0.00490196078431373
        struct[0].Gu[44,3] = 0.0196078431372549
        struct[0].Gu[44,4] = -0.00245098039215686
        struct[0].Gu[44,5] = -0.00980392156862745
        struct[0].Gu[45,0] = 0.00980392156862745
        struct[0].Gu[45,1] = -0.00245098039215686
        struct[0].Gu[45,2] = -0.0196078431372549
        struct[0].Gu[45,3] = 0.00490196078431373
        struct[0].Gu[45,4] = 0.00980392156862745
        struct[0].Gu[45,5] = -0.00245098039215686
        struct[0].Gu[46,0] = -0.00245098039215686
        struct[0].Gu[46,1] = -0.00980392156862745
        struct[0].Gu[46,2] = -0.00245098039215686
        struct[0].Gu[46,3] = -0.00980392156862745
        struct[0].Gu[46,4] = 0.00490196078431373
        struct[0].Gu[46,5] = 0.0196078431372549
        struct[0].Gu[47,0] = 0.00980392156862745
        struct[0].Gu[47,1] = -0.00245098039215686
        struct[0].Gu[47,2] = 0.00980392156862745
        struct[0].Gu[47,3] = -0.00245098039215686
        struct[0].Gu[47,4] = -0.0196078431372549
        struct[0].Gu[47,5] = 0.00490196078431373
        struct[0].Gu[56,6] = 67.7048070412999
        struct[0].Gu[57,7] = 67.7048070412999
        struct[0].Gu[58,8] = 67.7048070412999
        struct[0].Gu[59,9] = 67.7048070412999
        struct[0].Gu[60,10] = 67.7048070412999
        struct[0].Gu[61,11] = 67.7048070412999
        struct[0].Gu[72,38] = -1
        struct[0].Gu[73,40] = -1
        struct[0].Gu[74,42] = -1
        struct[0].Gu[75,39] = -1
        struct[0].Gu[76,41] = -1
        struct[0].Gu[77,43] = -1
        struct[0].Gu[80,44] = -1
        struct[0].Gu[81,45] = -1
        struct[0].Gu[84,46] = -1
        struct[0].Gu[85,47] = -1
        struct[0].Gu[89,49] = -1/3
        struct[0].Gu[91,49] = -1/3
        struct[0].Gu[93,49] = -1/3
        struct[0].Gu[97,50] = -coef_a_R10
        struct[0].Gu[98,51] = -coef_a_R10
        struct[0].Gu[99,50] = -coef_b_R10
        struct[0].Gu[100,51] = -coef_b_R10
        struct[0].Gu[101,50] = -coef_c_R10
        struct[0].Gu[102,51] = -coef_c_R10
        struct[0].Gu[105,50] = -1





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

    Gy_ini_rows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 29, 29, 30, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 79, 79, 79, 79, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 82, 82, 83, 83, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 86, 86, 87, 87, 88, 88, 88, 88, 88, 88, 88, 89, 89, 89, 89, 89, 89, 90, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 91, 92, 92, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103, 104, 104, 104, 104, 105, 105, 106, 106, 106]

    Gy_ini_cols = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 72, 88, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 73, 89, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 74, 90, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 75, 91, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 76, 92, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 77, 93, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 80, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 81, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 82, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 83, 12, 30, 84, 13, 31, 85, 14, 36, 86, 15, 37, 87, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 97, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 98, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 100, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 101, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 102, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 36, 29, 37, 12, 30, 103, 13, 31, 32, 38, 33, 39, 34, 40, 35, 41, 14, 28, 36, 104, 15, 29, 37, 32, 38, 33, 39, 34, 40, 35, 41, 0, 1, 2, 3, 42, 0, 1, 2, 3, 43, 2, 3, 4, 5, 44, 2, 3, 4, 5, 45, 0, 1, 4, 5, 46, 0, 1, 4, 5, 47, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 48, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 49, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 50, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 51, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 52, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 53, 48, 50, 52, 54, 49, 51, 53, 55, 30, 56, 31, 57, 32, 58, 33, 59, 34, 60, 35, 61, 56, 58, 60, 62, 57, 59, 61, 63, 12, 30, 64, 13, 31, 65, 32, 38, 66, 33, 39, 67, 34, 40, 68, 35, 41, 69, 64, 66, 68, 70, 65, 67, 69, 71, 0, 1, 6, 7, 72, 73, 2, 3, 6, 7, 74, 75, 4, 5, 6, 7, 76, 77, 0, 1, 6, 7, 72, 73, 2, 3, 6, 7, 74, 75, 4, 5, 6, 7, 76, 77, 72, 74, 76, 78, 73, 75, 77, 79, 8, 9, 10, 11, 80, 81, 8, 9, 10, 11, 80, 81, 80, 82, 81, 83, 12, 13, 14, 15, 84, 85, 12, 13, 14, 15, 84, 85, 84, 86, 85, 87, 0, 1, 6, 7, 88, 89, 94, 0, 1, 6, 7, 88, 89, 2, 3, 6, 7, 90, 91, 94, 2, 3, 6, 7, 90, 91, 4, 5, 6, 7, 92, 93, 94, 4, 5, 6, 7, 92, 93, 94, 95, 96, 56, 62, 95, 88, 89, 96, 16, 17, 22, 23, 97, 98, 16, 17, 22, 23, 97, 98, 18, 19, 22, 23, 99, 100, 18, 19, 22, 23, 99, 100, 20, 21, 22, 23, 101, 102, 20, 21, 22, 23, 101, 102, 30, 36, 103, 105, 30, 36, 104, 105, 105, 106, 97, 98, 106]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols