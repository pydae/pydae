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


class oc_4bus4wire2src_class: 

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
        self.N_y = 97 
        self.N_z = 32 
        self.N_store = 10000 
        self.params_list = ['X_B1_sa', 'R_B1_sa', 'X_B1_sb', 'R_B1_sb', 'X_B1_sc', 'R_B1_sc', 'X_B1_sn', 'R_B1_sn', 'S_n_B1', 'X_B1_ng', 'R_B1_ng', 'K_f_B1', 'T_f_B1', 'K_sec_B1', 'K_delta_B1', 'X_B4_sa', 'R_B4_sa', 'X_B4_sb', 'R_B4_sb', 'X_B4_sc', 'R_B4_sc', 'X_B4_sn', 'R_B4_sn', 'S_n_B4', 'X_B4_ng', 'R_B4_ng', 'K_f_B4', 'T_f_B4', 'K_sec_B4', 'K_delta_B4', 'K_agc'] 
        self.params_values_list  = [0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 100000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.001, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 100000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.0, 0.001] 
        self.inputs_ini_list = ['i_B2_n_r', 'i_B2_n_i', 'i_B3_n_r', 'i_B3_n_i', 'p_B2_a', 'q_B2_a', 'p_B2_b', 'q_B2_b', 'p_B2_c', 'q_B2_c', 'p_B3_a', 'q_B3_a', 'p_B3_b', 'q_B3_b', 'p_B3_c', 'q_B3_c', 'e_B1_an', 'e_B1_bn', 'e_B1_cn', 'phi_B1', 'p_B1_ref', 'omega_B1_ref', 'e_B4_an', 'e_B4_bn', 'e_B4_cn', 'phi_B4', 'p_B4_ref', 'omega_B4_ref'] 
        self.inputs_ini_values_list  = [-8.224533812913847e-12, -5.952463449725065e-12, -8.224533812913847e-12, -5.952463449725065e-12, 1.2485570081018256e-09, 6.381276858813941e-10, 1.2485570081018256e-09, 6.381276858813941e-10, 1.2485570081018256e-09, 6.381276858813941e-10, 1.2485570081018254e-09, 6.381276858813937e-10, 1.2485570081018254e-09, 6.381276858813937e-10, 1.2485570081018254e-09, 6.381276858813937e-10, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0] 
        self.inputs_run_list = ['i_B2_n_r', 'i_B2_n_i', 'i_B3_n_r', 'i_B3_n_i', 'p_B2_a', 'q_B2_a', 'p_B2_b', 'q_B2_b', 'p_B2_c', 'q_B2_c', 'p_B3_a', 'q_B3_a', 'p_B3_b', 'q_B3_b', 'p_B3_c', 'q_B3_c', 'e_B1_an', 'e_B1_bn', 'e_B1_cn', 'phi_B1', 'p_B1_ref', 'omega_B1_ref', 'e_B4_an', 'e_B4_bn', 'e_B4_cn', 'phi_B4', 'p_B4_ref', 'omega_B4_ref'] 
        self.inputs_run_values_list = [-8.224533812913847e-12, -5.952463449725065e-12, -8.224533812913847e-12, -5.952463449725065e-12, 1.2485570081018256e-09, 6.381276858813941e-10, 1.2485570081018256e-09, 6.381276858813941e-10, 1.2485570081018256e-09, 6.381276858813941e-10, 1.2485570081018254e-09, 6.381276858813937e-10, 1.2485570081018254e-09, 6.381276858813937e-10, 1.2485570081018254e-09, 6.381276858813937e-10, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0] 
        self.outputs_list = ['v_B2_a_m', 'v_B2_b_m', 'v_B2_c_m', 'v_B2_n_m', 'v_B3_a_m', 'v_B3_b_m', 'v_B3_c_m', 'v_B3_n_m', 'v_B1_a_m', 'v_B1_b_m', 'v_B1_c_m', 'v_B1_n_m', 'v_B4_a_m', 'v_B4_b_m', 'v_B4_c_m', 'v_B4_n_m', 'p_B1_pos', 'p_B1_neg', 'p_B1_zer', 'e_B1_an', 'e_B1_bn', 'e_B1_cn', 'p_B1_ref', 'omega_B1_ref', 'p_B4_pos', 'p_B4_neg', 'p_B4_zer', 'e_B4_an', 'e_B4_bn', 'e_B4_cn', 'p_B4_ref', 'omega_B4_ref'] 
        self.x_list = ['phi_B1', 'omega_B1', 'phi_B4', 'omega_B4', 'xi_freq'] 
        self.y_run_list = ['v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B2_n_r', 'v_B2_n_i', 'v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B3_n_r', 'v_B3_n_i', 'v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'v_B1_n_r', 'v_B1_n_i', 'v_B4_a_r', 'v_B4_a_i', 'v_B4_b_r', 'v_B4_b_i', 'v_B4_c_r', 'v_B4_c_i', 'v_B4_n_r', 'v_B4_n_i', 'i_l_B1_B2_a_r', 'i_l_B1_B2_a_i', 'i_l_B1_B2_b_r', 'i_l_B1_B2_b_i', 'i_l_B1_B2_c_r', 'i_l_B1_B2_c_i', 'i_l_B1_B2_n_r', 'i_l_B1_B2_n_i', 'i_l_B2_B3_a_r', 'i_l_B2_B3_a_i', 'i_l_B2_B3_b_r', 'i_l_B2_B3_b_i', 'i_l_B2_B3_c_r', 'i_l_B2_B3_c_i', 'i_l_B2_B3_n_r', 'i_l_B2_B3_n_i', 'i_l_B3_B4_a_r', 'i_l_B3_B4_a_i', 'i_l_B3_B4_b_r', 'i_l_B3_B4_b_i', 'i_l_B3_B4_c_r', 'i_l_B3_B4_c_i', 'i_l_B3_B4_n_r', 'i_l_B3_B4_n_i', 'i_load_B2_a_r', 'i_load_B2_a_i', 'i_load_B2_b_r', 'i_load_B2_b_i', 'i_load_B2_c_r', 'i_load_B2_c_i', 'i_load_B2_n_r', 'i_load_B2_n_i', 'i_load_B3_a_r', 'i_load_B3_a_i', 'i_load_B3_b_r', 'i_load_B3_b_i', 'i_load_B3_c_r', 'i_load_B3_c_i', 'i_load_B3_n_r', 'i_load_B3_n_i', 'i_B1_a_r', 'i_B1_b_r', 'i_B1_c_r', 'i_B1_n_r', 'i_B1_ng_r', 'e_B1_ng_r', 'i_B1_a_i', 'i_B1_b_i', 'i_B1_c_i', 'i_B1_n_i', 'i_B1_ng_i', 'e_B1_ng_i', 'i_B4_a_r', 'i_B4_b_r', 'i_B4_c_r', 'i_B4_n_r', 'i_B4_ng_r', 'e_B4_ng_r', 'i_B4_a_i', 'i_B4_b_i', 'i_B4_c_i', 'i_B4_n_i', 'i_B4_ng_i', 'e_B4_ng_i', 'omega_coi'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B2_n_r', 'v_B2_n_i', 'v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B3_n_r', 'v_B3_n_i', 'v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'v_B1_n_r', 'v_B1_n_i', 'v_B4_a_r', 'v_B4_a_i', 'v_B4_b_r', 'v_B4_b_i', 'v_B4_c_r', 'v_B4_c_i', 'v_B4_n_r', 'v_B4_n_i', 'i_l_B1_B2_a_r', 'i_l_B1_B2_a_i', 'i_l_B1_B2_b_r', 'i_l_B1_B2_b_i', 'i_l_B1_B2_c_r', 'i_l_B1_B2_c_i', 'i_l_B1_B2_n_r', 'i_l_B1_B2_n_i', 'i_l_B2_B3_a_r', 'i_l_B2_B3_a_i', 'i_l_B2_B3_b_r', 'i_l_B2_B3_b_i', 'i_l_B2_B3_c_r', 'i_l_B2_B3_c_i', 'i_l_B2_B3_n_r', 'i_l_B2_B3_n_i', 'i_l_B3_B4_a_r', 'i_l_B3_B4_a_i', 'i_l_B3_B4_b_r', 'i_l_B3_B4_b_i', 'i_l_B3_B4_c_r', 'i_l_B3_B4_c_i', 'i_l_B3_B4_n_r', 'i_l_B3_B4_n_i', 'i_load_B2_a_r', 'i_load_B2_a_i', 'i_load_B2_b_r', 'i_load_B2_b_i', 'i_load_B2_c_r', 'i_load_B2_c_i', 'i_load_B2_n_r', 'i_load_B2_n_i', 'i_load_B3_a_r', 'i_load_B3_a_i', 'i_load_B3_b_r', 'i_load_B3_b_i', 'i_load_B3_c_r', 'i_load_B3_c_i', 'i_load_B3_n_r', 'i_load_B3_n_i', 'i_B1_a_r', 'i_B1_b_r', 'i_B1_c_r', 'i_B1_n_r', 'i_B1_ng_r', 'e_B1_ng_r', 'i_B1_a_i', 'i_B1_b_i', 'i_B1_c_i', 'i_B1_n_i', 'i_B1_ng_i', 'e_B1_ng_i', 'i_B4_a_r', 'i_B4_b_r', 'i_B4_c_r', 'i_B4_n_r', 'i_B4_ng_r', 'e_B4_ng_r', 'i_B4_a_i', 'i_B4_b_i', 'i_B4_c_i', 'i_B4_n_i', 'i_B4_ng_i', 'e_B4_ng_i', 'omega_coi'] 
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
    X_B1_sa = struct[0].X_B1_sa
    R_B1_sa = struct[0].R_B1_sa
    X_B1_sb = struct[0].X_B1_sb
    R_B1_sb = struct[0].R_B1_sb
    X_B1_sc = struct[0].X_B1_sc
    R_B1_sc = struct[0].R_B1_sc
    X_B1_sn = struct[0].X_B1_sn
    R_B1_sn = struct[0].R_B1_sn
    S_n_B1 = struct[0].S_n_B1
    X_B1_ng = struct[0].X_B1_ng
    R_B1_ng = struct[0].R_B1_ng
    K_f_B1 = struct[0].K_f_B1
    T_f_B1 = struct[0].T_f_B1
    K_sec_B1 = struct[0].K_sec_B1
    K_delta_B1 = struct[0].K_delta_B1
    X_B4_sa = struct[0].X_B4_sa
    R_B4_sa = struct[0].R_B4_sa
    X_B4_sb = struct[0].X_B4_sb
    R_B4_sb = struct[0].R_B4_sb
    X_B4_sc = struct[0].X_B4_sc
    R_B4_sc = struct[0].R_B4_sc
    X_B4_sn = struct[0].X_B4_sn
    R_B4_sn = struct[0].R_B4_sn
    S_n_B4 = struct[0].S_n_B4
    X_B4_ng = struct[0].X_B4_ng
    R_B4_ng = struct[0].R_B4_ng
    K_f_B4 = struct[0].K_f_B4
    T_f_B4 = struct[0].T_f_B4
    K_sec_B4 = struct[0].K_sec_B4
    K_delta_B4 = struct[0].K_delta_B4
    K_agc = struct[0].K_agc
    
    # Inputs:
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
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
    e_B1_an = struct[0].e_B1_an
    e_B1_bn = struct[0].e_B1_bn
    e_B1_cn = struct[0].e_B1_cn
    phi_B1 = struct[0].phi_B1
    p_B1_ref = struct[0].p_B1_ref
    omega_B1_ref = struct[0].omega_B1_ref
    e_B4_an = struct[0].e_B4_an
    e_B4_bn = struct[0].e_B4_bn
    e_B4_cn = struct[0].e_B4_cn
    phi_B4 = struct[0].phi_B4
    p_B4_ref = struct[0].p_B4_ref
    omega_B4_ref = struct[0].omega_B4_ref
    
    # Dynamical states:
    phi_B1 = struct[0].x[0,0]
    omega_B1 = struct[0].x[1,0]
    phi_B4 = struct[0].x[2,0]
    omega_B4 = struct[0].x[3,0]
    xi_freq = struct[0].x[4,0]
    
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
    v_B1_a_r = struct[0].y_ini[16,0]
    v_B1_a_i = struct[0].y_ini[17,0]
    v_B1_b_r = struct[0].y_ini[18,0]
    v_B1_b_i = struct[0].y_ini[19,0]
    v_B1_c_r = struct[0].y_ini[20,0]
    v_B1_c_i = struct[0].y_ini[21,0]
    v_B1_n_r = struct[0].y_ini[22,0]
    v_B1_n_i = struct[0].y_ini[23,0]
    v_B4_a_r = struct[0].y_ini[24,0]
    v_B4_a_i = struct[0].y_ini[25,0]
    v_B4_b_r = struct[0].y_ini[26,0]
    v_B4_b_i = struct[0].y_ini[27,0]
    v_B4_c_r = struct[0].y_ini[28,0]
    v_B4_c_i = struct[0].y_ini[29,0]
    v_B4_n_r = struct[0].y_ini[30,0]
    v_B4_n_i = struct[0].y_ini[31,0]
    i_l_B1_B2_a_r = struct[0].y_ini[32,0]
    i_l_B1_B2_a_i = struct[0].y_ini[33,0]
    i_l_B1_B2_b_r = struct[0].y_ini[34,0]
    i_l_B1_B2_b_i = struct[0].y_ini[35,0]
    i_l_B1_B2_c_r = struct[0].y_ini[36,0]
    i_l_B1_B2_c_i = struct[0].y_ini[37,0]
    i_l_B1_B2_n_r = struct[0].y_ini[38,0]
    i_l_B1_B2_n_i = struct[0].y_ini[39,0]
    i_l_B2_B3_a_r = struct[0].y_ini[40,0]
    i_l_B2_B3_a_i = struct[0].y_ini[41,0]
    i_l_B2_B3_b_r = struct[0].y_ini[42,0]
    i_l_B2_B3_b_i = struct[0].y_ini[43,0]
    i_l_B2_B3_c_r = struct[0].y_ini[44,0]
    i_l_B2_B3_c_i = struct[0].y_ini[45,0]
    i_l_B2_B3_n_r = struct[0].y_ini[46,0]
    i_l_B2_B3_n_i = struct[0].y_ini[47,0]
    i_l_B3_B4_a_r = struct[0].y_ini[48,0]
    i_l_B3_B4_a_i = struct[0].y_ini[49,0]
    i_l_B3_B4_b_r = struct[0].y_ini[50,0]
    i_l_B3_B4_b_i = struct[0].y_ini[51,0]
    i_l_B3_B4_c_r = struct[0].y_ini[52,0]
    i_l_B3_B4_c_i = struct[0].y_ini[53,0]
    i_l_B3_B4_n_r = struct[0].y_ini[54,0]
    i_l_B3_B4_n_i = struct[0].y_ini[55,0]
    i_load_B2_a_r = struct[0].y_ini[56,0]
    i_load_B2_a_i = struct[0].y_ini[57,0]
    i_load_B2_b_r = struct[0].y_ini[58,0]
    i_load_B2_b_i = struct[0].y_ini[59,0]
    i_load_B2_c_r = struct[0].y_ini[60,0]
    i_load_B2_c_i = struct[0].y_ini[61,0]
    i_load_B2_n_r = struct[0].y_ini[62,0]
    i_load_B2_n_i = struct[0].y_ini[63,0]
    i_load_B3_a_r = struct[0].y_ini[64,0]
    i_load_B3_a_i = struct[0].y_ini[65,0]
    i_load_B3_b_r = struct[0].y_ini[66,0]
    i_load_B3_b_i = struct[0].y_ini[67,0]
    i_load_B3_c_r = struct[0].y_ini[68,0]
    i_load_B3_c_i = struct[0].y_ini[69,0]
    i_load_B3_n_r = struct[0].y_ini[70,0]
    i_load_B3_n_i = struct[0].y_ini[71,0]
    i_B1_a_r = struct[0].y_ini[72,0]
    i_B1_b_r = struct[0].y_ini[73,0]
    i_B1_c_r = struct[0].y_ini[74,0]
    i_B1_n_r = struct[0].y_ini[75,0]
    i_B1_ng_r = struct[0].y_ini[76,0]
    e_B1_ng_r = struct[0].y_ini[77,0]
    i_B1_a_i = struct[0].y_ini[78,0]
    i_B1_b_i = struct[0].y_ini[79,0]
    i_B1_c_i = struct[0].y_ini[80,0]
    i_B1_n_i = struct[0].y_ini[81,0]
    i_B1_ng_i = struct[0].y_ini[82,0]
    e_B1_ng_i = struct[0].y_ini[83,0]
    i_B4_a_r = struct[0].y_ini[84,0]
    i_B4_b_r = struct[0].y_ini[85,0]
    i_B4_c_r = struct[0].y_ini[86,0]
    i_B4_n_r = struct[0].y_ini[87,0]
    i_B4_ng_r = struct[0].y_ini[88,0]
    e_B4_ng_r = struct[0].y_ini[89,0]
    i_B4_a_i = struct[0].y_ini[90,0]
    i_B4_b_i = struct[0].y_ini[91,0]
    i_B4_c_i = struct[0].y_ini[92,0]
    i_B4_n_i = struct[0].y_ini[93,0]
    i_B4_ng_i = struct[0].y_ini[94,0]
    e_B4_ng_i = struct[0].y_ini[95,0]
    omega_coi = struct[0].y_ini[96,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_B1*phi_B1 + 314.159265358979*omega_B1 - 314.159265358979*omega_coi
        struct[0].f[1,0] = (-K_f_B1*(K_sec_B1*xi_freq - 0.333333333333333*i_B1_a_i*v_B1_a_i - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) - 0.333333333333333*i_B1_a_r*v_B1_a_r - 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) - 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) - 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r - 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i + 0.166666666666667*i_B1_b_r*v_B1_c_r + 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r - 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i + 0.166666666666667*i_B1_c_r*v_B1_b_r - 0.333333333333333*i_B1_c_r*v_B1_c_r + p_B1_ref + v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) + v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) - v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) - v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r))/S_n_B1 - omega_B1 + omega_B1_ref)/T_f_B1
        struct[0].f[2,0] = -K_delta_B4*phi_B4 + 314.159265358979*omega_B4 - 314.159265358979*omega_coi
        struct[0].f[3,0] = (-K_f_B4*(K_sec_B4*xi_freq - 0.333333333333333*i_B4_a_i*v_B4_a_i - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) - 0.333333333333333*i_B4_a_r*v_B4_a_r - 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) - 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) - 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r - 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i + 0.166666666666667*i_B4_b_r*v_B4_c_r + 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r - 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i + 0.166666666666667*i_B4_c_r*v_B4_b_r - 0.333333333333333*i_B4_c_r*v_B4_c_r + p_B4_ref + v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) + v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) - v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) - v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r))/S_n_B4 - omega_B4 + omega_B4_ref)/T_f_B4
        struct[0].f[4,0] = K_agc*(1 - omega_coi)
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[56,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[57,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[58,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[59,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[60,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[61,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[64,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[65,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[66,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[67,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[68,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[69,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
        struct[0].g[72,0] = -R_B1_sa*i_B1_a_r + 1.0*X_B1_sa*i_B1_a_i + e_B1_an*cos(phi_B1) - v_B1_a_r + v_B1_n_r
        struct[0].g[73,0] = -R_B1_sb*i_B1_b_r + 1.0*X_B1_sb*i_B1_b_i + e_B1_bn*cos(phi_B1 - 2.0943951023932) - v_B1_b_r + v_B1_n_r
        struct[0].g[74,0] = -R_B1_sc*i_B1_c_r + 1.0*X_B1_sc*i_B1_c_i + e_B1_cn*cos(phi_B1 - 4.18879020478639) - v_B1_c_r + v_B1_n_r
        struct[0].g[78,0] = -1.0*R_B1_sa*i_B1_a_i - 1.0*X_B1_sa*i_B1_a_r + 1.0*e_B1_an*sin(phi_B1) - 1.0*v_B1_a_i + 1.0*v_B1_n_i
        struct[0].g[79,0] = -1.0*R_B1_sb*i_B1_b_i - 1.0*X_B1_sb*i_B1_b_r + 1.0*e_B1_bn*sin(phi_B1 - 2.0943951023932) - 1.0*v_B1_b_i + 1.0*v_B1_n_i
        struct[0].g[80,0] = -1.0*R_B1_sc*i_B1_c_i - 1.0*X_B1_sc*i_B1_c_r + 1.0*e_B1_cn*sin(phi_B1 - 4.18879020478639) - 1.0*v_B1_c_i + 1.0*v_B1_n_i
        struct[0].g[84,0] = -R_B4_sa*i_B4_a_r + 1.0*X_B4_sa*i_B4_a_i + e_B4_an*cos(phi_B4) - v_B4_a_r + v_B4_n_r
        struct[0].g[85,0] = -R_B4_sb*i_B4_b_r + 1.0*X_B4_sb*i_B4_b_i + e_B4_bn*cos(phi_B4 - 2.0943951023932) - v_B4_b_r + v_B4_n_r
        struct[0].g[86,0] = -R_B4_sc*i_B4_c_r + 1.0*X_B4_sc*i_B4_c_i + e_B4_cn*cos(phi_B4 - 4.18879020478639) - v_B4_c_r + v_B4_n_r
        struct[0].g[90,0] = -1.0*R_B4_sa*i_B4_a_i - 1.0*X_B4_sa*i_B4_a_r + 1.0*e_B4_an*sin(phi_B4) - 1.0*v_B4_a_i + 1.0*v_B4_n_i
        struct[0].g[91,0] = -1.0*R_B4_sb*i_B4_b_i - 1.0*X_B4_sb*i_B4_b_r + 1.0*e_B4_bn*sin(phi_B4 - 2.0943951023932) - 1.0*v_B4_b_i + 1.0*v_B4_n_i
        struct[0].g[92,0] = -1.0*R_B4_sc*i_B4_c_i - 1.0*X_B4_sc*i_B4_c_r + 1.0*e_B4_cn*sin(phi_B4 - 4.18879020478639) - 1.0*v_B4_c_i + 1.0*v_B4_n_i
        struct[0].g[96,0] = omega_coi - (S_n_B1*omega_B1 + S_n_B4*omega_B4)/(S_n_B1 + S_n_B4)
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[1,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[2,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[3,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[4,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[5,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[6,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[7,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[8,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[9,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[10,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[11,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[12,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[13,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[14,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
        struct[0].h[16,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i + 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r - 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i - 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) + v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[17,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r) + v_B1_a_r*(0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(-0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[18,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 0.333333333333333*i_B1_a_i*v_B1_b_i + 0.333333333333333*i_B1_a_i*v_B1_c_i + 0.333333333333333*i_B1_a_r*v_B1_a_r + 0.333333333333333*i_B1_a_r*v_B1_b_r + 0.333333333333333*i_B1_a_r*v_B1_c_r + 0.333333333333333*i_B1_b_i*v_B1_a_i + 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.333333333333333*i_B1_b_i*v_B1_c_i + 0.333333333333333*i_B1_b_r*v_B1_a_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.333333333333333*i_B1_b_r*v_B1_c_r + 0.333333333333333*i_B1_c_i*v_B1_a_i + 0.333333333333333*i_B1_c_i*v_B1_b_i + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.333333333333333*i_B1_c_r*v_B1_a_r + 0.333333333333333*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r
        struct[0].h[19,0] = e_B1_an
        struct[0].h[20,0] = e_B1_bn
        struct[0].h[21,0] = e_B1_cn
        struct[0].h[22,0] = p_B1_ref
        struct[0].h[23,0] = omega_B1_ref
        struct[0].h[24,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i + 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r - 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i - 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) + v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[25,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r) + v_B4_a_r*(0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(-0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[26,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 0.333333333333333*i_B4_a_i*v_B4_b_i + 0.333333333333333*i_B4_a_i*v_B4_c_i + 0.333333333333333*i_B4_a_r*v_B4_a_r + 0.333333333333333*i_B4_a_r*v_B4_b_r + 0.333333333333333*i_B4_a_r*v_B4_c_r + 0.333333333333333*i_B4_b_i*v_B4_a_i + 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.333333333333333*i_B4_b_i*v_B4_c_i + 0.333333333333333*i_B4_b_r*v_B4_a_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.333333333333333*i_B4_b_r*v_B4_c_r + 0.333333333333333*i_B4_c_i*v_B4_a_i + 0.333333333333333*i_B4_c_i*v_B4_b_i + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.333333333333333*i_B4_c_r*v_B4_a_r + 0.333333333333333*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r
        struct[0].h[27,0] = e_B4_an
        struct[0].h[28,0] = e_B4_bn
        struct[0].h[29,0] = e_B4_cn
        struct[0].h[30,0] = p_B4_ref
        struct[0].h[31,0] = omega_B4_ref
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_B1
        struct[0].Fx_ini[1,1] = -1/T_f_B1
        struct[0].Fx_ini[1,4] = -K_f_B1*K_sec_B1/(S_n_B1*T_f_B1)
        struct[0].Fx_ini[2,2] = -K_delta_B4
        struct[0].Fx_ini[3,3] = -1/T_f_B4
        struct[0].Fx_ini[3,4] = -K_f_B4*K_sec_B4/(S_n_B4*T_f_B4)

    if mode == 11:

        struct[0].Fy_ini[0,96] = -314.159265358979 
        struct[0].Fy_ini[1,16] = -K_f_B1*(-0.333333333333333*i_B1_a_r + 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,17] = -K_f_B1*(-0.333333333333333*i_B1_a_i + 0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r + 0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,18] = -K_f_B1*(-0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.333333333333333*i_B1_b_r + 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,19] = -K_f_B1*(0.166666666666667*i_B1_a_i + 0.288675134594813*i_B1_a_r - 0.333333333333333*i_B1_b_i + 0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,20] = -K_f_B1*(0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.333333333333333*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,21] = -K_f_B1*(0.166666666666667*i_B1_a_i - 0.288675134594813*i_B1_a_r + 0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r - 0.333333333333333*i_B1_c_i)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,72] = -K_f_B1*(-0.333333333333333*v_B1_a_r + 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,73] = -K_f_B1*(-0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.333333333333333*v_B1_b_r + 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,74] = -K_f_B1*(0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.333333333333333*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,78] = -K_f_B1*(-0.333333333333333*v_B1_a_i + 0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r + 0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,79] = -K_f_B1*(0.166666666666667*v_B1_a_i + 0.288675134594813*v_B1_a_r - 0.333333333333333*v_B1_b_i + 0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,80] = -K_f_B1*(0.166666666666667*v_B1_a_i - 0.288675134594813*v_B1_a_r + 0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r - 0.333333333333333*v_B1_c_i)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[2,96] = -314.159265358979 
        struct[0].Fy_ini[3,24] = -K_f_B4*(-0.333333333333333*i_B4_a_r + 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,25] = -K_f_B4*(-0.333333333333333*i_B4_a_i + 0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r + 0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,26] = -K_f_B4*(-0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.333333333333333*i_B4_b_r + 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,27] = -K_f_B4*(0.166666666666667*i_B4_a_i + 0.288675134594813*i_B4_a_r - 0.333333333333333*i_B4_b_i + 0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,28] = -K_f_B4*(0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.333333333333333*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,29] = -K_f_B4*(0.166666666666667*i_B4_a_i - 0.288675134594813*i_B4_a_r + 0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r - 0.333333333333333*i_B4_c_i)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,84] = -K_f_B4*(-0.333333333333333*v_B4_a_r + 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,85] = -K_f_B4*(-0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.333333333333333*v_B4_b_r + 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,86] = -K_f_B4*(0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.333333333333333*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,90] = -K_f_B4*(-0.333333333333333*v_B4_a_i + 0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r + 0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,91] = -K_f_B4*(0.166666666666667*v_B4_a_i + 0.288675134594813*v_B4_a_r - 0.333333333333333*v_B4_b_i + 0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,92] = -K_f_B4*(0.166666666666667*v_B4_a_i - 0.288675134594813*v_B4_a_r + 0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r - 0.333333333333333*v_B4_c_i)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[4,96] = -K_agc 

        struct[0].Gx_ini[72,0] = -e_B1_an*sin(phi_B1)
        struct[0].Gx_ini[73,0] = -e_B1_bn*sin(phi_B1 - 2.0943951023932)
        struct[0].Gx_ini[74,0] = -e_B1_cn*sin(phi_B1 - 4.18879020478639)
        struct[0].Gx_ini[78,0] = 1.0*e_B1_an*cos(phi_B1)
        struct[0].Gx_ini[79,0] = 1.0*e_B1_bn*cos(phi_B1 - 2.0943951023932)
        struct[0].Gx_ini[80,0] = 1.0*e_B1_cn*cos(phi_B1 - 4.18879020478639)
        struct[0].Gx_ini[84,2] = -e_B4_an*sin(phi_B4)
        struct[0].Gx_ini[85,2] = -e_B4_bn*sin(phi_B4 - 2.0943951023932)
        struct[0].Gx_ini[86,2] = -e_B4_cn*sin(phi_B4 - 4.18879020478639)
        struct[0].Gx_ini[90,2] = 1.0*e_B4_an*cos(phi_B4)
        struct[0].Gx_ini[91,2] = 1.0*e_B4_bn*cos(phi_B4 - 2.0943951023932)
        struct[0].Gx_ini[92,2] = 1.0*e_B4_cn*cos(phi_B4 - 4.18879020478639)
        struct[0].Gx_ini[96,1] = -S_n_B1/(S_n_B1 + S_n_B4)
        struct[0].Gx_ini[96,3] = -S_n_B4/(S_n_B1 + S_n_B4)

        struct[0].Gy_ini[56,0] = i_load_B2_a_r
        struct[0].Gy_ini[56,1] = i_load_B2_a_i
        struct[0].Gy_ini[56,6] = -i_load_B2_a_r
        struct[0].Gy_ini[56,7] = -i_load_B2_a_i
        struct[0].Gy_ini[56,56] = v_B2_a_r - v_B2_n_r
        struct[0].Gy_ini[56,57] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[57,2] = i_load_B2_b_r
        struct[0].Gy_ini[57,3] = i_load_B2_b_i
        struct[0].Gy_ini[57,6] = -i_load_B2_b_r
        struct[0].Gy_ini[57,7] = -i_load_B2_b_i
        struct[0].Gy_ini[57,58] = v_B2_b_r - v_B2_n_r
        struct[0].Gy_ini[57,59] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[58,4] = i_load_B2_c_r
        struct[0].Gy_ini[58,5] = i_load_B2_c_i
        struct[0].Gy_ini[58,6] = -i_load_B2_c_r
        struct[0].Gy_ini[58,7] = -i_load_B2_c_i
        struct[0].Gy_ini[58,60] = v_B2_c_r - v_B2_n_r
        struct[0].Gy_ini[58,61] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[59,0] = -i_load_B2_a_i
        struct[0].Gy_ini[59,1] = i_load_B2_a_r
        struct[0].Gy_ini[59,6] = i_load_B2_a_i
        struct[0].Gy_ini[59,7] = -i_load_B2_a_r
        struct[0].Gy_ini[59,56] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[59,57] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy_ini[60,2] = -i_load_B2_b_i
        struct[0].Gy_ini[60,3] = i_load_B2_b_r
        struct[0].Gy_ini[60,6] = i_load_B2_b_i
        struct[0].Gy_ini[60,7] = -i_load_B2_b_r
        struct[0].Gy_ini[60,58] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[60,59] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy_ini[61,4] = -i_load_B2_c_i
        struct[0].Gy_ini[61,5] = i_load_B2_c_r
        struct[0].Gy_ini[61,6] = i_load_B2_c_i
        struct[0].Gy_ini[61,7] = -i_load_B2_c_r
        struct[0].Gy_ini[61,60] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[61,61] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy_ini[64,8] = i_load_B3_a_r
        struct[0].Gy_ini[64,9] = i_load_B3_a_i
        struct[0].Gy_ini[64,14] = -i_load_B3_a_r
        struct[0].Gy_ini[64,15] = -i_load_B3_a_i
        struct[0].Gy_ini[64,64] = v_B3_a_r - v_B3_n_r
        struct[0].Gy_ini[64,65] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[65,10] = i_load_B3_b_r
        struct[0].Gy_ini[65,11] = i_load_B3_b_i
        struct[0].Gy_ini[65,14] = -i_load_B3_b_r
        struct[0].Gy_ini[65,15] = -i_load_B3_b_i
        struct[0].Gy_ini[65,66] = v_B3_b_r - v_B3_n_r
        struct[0].Gy_ini[65,67] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[66,12] = i_load_B3_c_r
        struct[0].Gy_ini[66,13] = i_load_B3_c_i
        struct[0].Gy_ini[66,14] = -i_load_B3_c_r
        struct[0].Gy_ini[66,15] = -i_load_B3_c_i
        struct[0].Gy_ini[66,68] = v_B3_c_r - v_B3_n_r
        struct[0].Gy_ini[66,69] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[67,8] = -i_load_B3_a_i
        struct[0].Gy_ini[67,9] = i_load_B3_a_r
        struct[0].Gy_ini[67,14] = i_load_B3_a_i
        struct[0].Gy_ini[67,15] = -i_load_B3_a_r
        struct[0].Gy_ini[67,64] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[67,65] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy_ini[68,10] = -i_load_B3_b_i
        struct[0].Gy_ini[68,11] = i_load_B3_b_r
        struct[0].Gy_ini[68,14] = i_load_B3_b_i
        struct[0].Gy_ini[68,15] = -i_load_B3_b_r
        struct[0].Gy_ini[68,66] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[68,67] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy_ini[69,12] = -i_load_B3_c_i
        struct[0].Gy_ini[69,13] = i_load_B3_c_r
        struct[0].Gy_ini[69,14] = i_load_B3_c_i
        struct[0].Gy_ini[69,15] = -i_load_B3_c_r
        struct[0].Gy_ini[69,68] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[69,69] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy_ini[72,72] = -R_B1_sa
        struct[0].Gy_ini[72,78] = 1.0*X_B1_sa
        struct[0].Gy_ini[73,73] = -R_B1_sb
        struct[0].Gy_ini[73,79] = 1.0*X_B1_sb
        struct[0].Gy_ini[74,74] = -R_B1_sc
        struct[0].Gy_ini[74,80] = 1.0*X_B1_sc
        struct[0].Gy_ini[75,75] = -R_B1_sn
        struct[0].Gy_ini[75,81] = 1.0*X_B1_sn
        struct[0].Gy_ini[77,76] = R_B1_ng
        struct[0].Gy_ini[77,82] = -1.0*X_B1_ng
        struct[0].Gy_ini[78,72] = -1.0*X_B1_sa
        struct[0].Gy_ini[78,78] = -1.0*R_B1_sa
        struct[0].Gy_ini[79,73] = -1.0*X_B1_sb
        struct[0].Gy_ini[79,79] = -1.0*R_B1_sb
        struct[0].Gy_ini[80,74] = -1.0*X_B1_sc
        struct[0].Gy_ini[80,80] = -1.0*R_B1_sc
        struct[0].Gy_ini[81,75] = -1.0*X_B1_sn
        struct[0].Gy_ini[81,81] = -1.0*R_B1_sn
        struct[0].Gy_ini[83,76] = 1.0*X_B1_ng
        struct[0].Gy_ini[83,82] = 1.0*R_B1_ng
        struct[0].Gy_ini[84,84] = -R_B4_sa
        struct[0].Gy_ini[84,90] = 1.0*X_B4_sa
        struct[0].Gy_ini[85,85] = -R_B4_sb
        struct[0].Gy_ini[85,91] = 1.0*X_B4_sb
        struct[0].Gy_ini[86,86] = -R_B4_sc
        struct[0].Gy_ini[86,92] = 1.0*X_B4_sc
        struct[0].Gy_ini[87,87] = -R_B4_sn
        struct[0].Gy_ini[87,93] = 1.0*X_B4_sn
        struct[0].Gy_ini[89,88] = R_B4_ng
        struct[0].Gy_ini[89,94] = -1.0*X_B4_ng
        struct[0].Gy_ini[90,84] = -1.0*X_B4_sa
        struct[0].Gy_ini[90,90] = -1.0*R_B4_sa
        struct[0].Gy_ini[91,85] = -1.0*X_B4_sb
        struct[0].Gy_ini[91,91] = -1.0*R_B4_sb
        struct[0].Gy_ini[92,86] = -1.0*X_B4_sc
        struct[0].Gy_ini[92,92] = -1.0*R_B4_sc
        struct[0].Gy_ini[93,87] = -1.0*X_B4_sn
        struct[0].Gy_ini[93,93] = -1.0*R_B4_sn
        struct[0].Gy_ini[95,88] = 1.0*X_B4_ng
        struct[0].Gy_ini[95,94] = 1.0*R_B4_ng



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    X_B1_sa = struct[0].X_B1_sa
    R_B1_sa = struct[0].R_B1_sa
    X_B1_sb = struct[0].X_B1_sb
    R_B1_sb = struct[0].R_B1_sb
    X_B1_sc = struct[0].X_B1_sc
    R_B1_sc = struct[0].R_B1_sc
    X_B1_sn = struct[0].X_B1_sn
    R_B1_sn = struct[0].R_B1_sn
    S_n_B1 = struct[0].S_n_B1
    X_B1_ng = struct[0].X_B1_ng
    R_B1_ng = struct[0].R_B1_ng
    K_f_B1 = struct[0].K_f_B1
    T_f_B1 = struct[0].T_f_B1
    K_sec_B1 = struct[0].K_sec_B1
    K_delta_B1 = struct[0].K_delta_B1
    X_B4_sa = struct[0].X_B4_sa
    R_B4_sa = struct[0].R_B4_sa
    X_B4_sb = struct[0].X_B4_sb
    R_B4_sb = struct[0].R_B4_sb
    X_B4_sc = struct[0].X_B4_sc
    R_B4_sc = struct[0].R_B4_sc
    X_B4_sn = struct[0].X_B4_sn
    R_B4_sn = struct[0].R_B4_sn
    S_n_B4 = struct[0].S_n_B4
    X_B4_ng = struct[0].X_B4_ng
    R_B4_ng = struct[0].R_B4_ng
    K_f_B4 = struct[0].K_f_B4
    T_f_B4 = struct[0].T_f_B4
    K_sec_B4 = struct[0].K_sec_B4
    K_delta_B4 = struct[0].K_delta_B4
    K_agc = struct[0].K_agc
    
    # Inputs:
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
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
    e_B1_an = struct[0].e_B1_an
    e_B1_bn = struct[0].e_B1_bn
    e_B1_cn = struct[0].e_B1_cn
    phi_B1 = struct[0].phi_B1
    p_B1_ref = struct[0].p_B1_ref
    omega_B1_ref = struct[0].omega_B1_ref
    e_B4_an = struct[0].e_B4_an
    e_B4_bn = struct[0].e_B4_bn
    e_B4_cn = struct[0].e_B4_cn
    phi_B4 = struct[0].phi_B4
    p_B4_ref = struct[0].p_B4_ref
    omega_B4_ref = struct[0].omega_B4_ref
    
    # Dynamical states:
    phi_B1 = struct[0].x[0,0]
    omega_B1 = struct[0].x[1,0]
    phi_B4 = struct[0].x[2,0]
    omega_B4 = struct[0].x[3,0]
    xi_freq = struct[0].x[4,0]
    
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
    v_B1_a_r = struct[0].y_run[16,0]
    v_B1_a_i = struct[0].y_run[17,0]
    v_B1_b_r = struct[0].y_run[18,0]
    v_B1_b_i = struct[0].y_run[19,0]
    v_B1_c_r = struct[0].y_run[20,0]
    v_B1_c_i = struct[0].y_run[21,0]
    v_B1_n_r = struct[0].y_run[22,0]
    v_B1_n_i = struct[0].y_run[23,0]
    v_B4_a_r = struct[0].y_run[24,0]
    v_B4_a_i = struct[0].y_run[25,0]
    v_B4_b_r = struct[0].y_run[26,0]
    v_B4_b_i = struct[0].y_run[27,0]
    v_B4_c_r = struct[0].y_run[28,0]
    v_B4_c_i = struct[0].y_run[29,0]
    v_B4_n_r = struct[0].y_run[30,0]
    v_B4_n_i = struct[0].y_run[31,0]
    i_l_B1_B2_a_r = struct[0].y_run[32,0]
    i_l_B1_B2_a_i = struct[0].y_run[33,0]
    i_l_B1_B2_b_r = struct[0].y_run[34,0]
    i_l_B1_B2_b_i = struct[0].y_run[35,0]
    i_l_B1_B2_c_r = struct[0].y_run[36,0]
    i_l_B1_B2_c_i = struct[0].y_run[37,0]
    i_l_B1_B2_n_r = struct[0].y_run[38,0]
    i_l_B1_B2_n_i = struct[0].y_run[39,0]
    i_l_B2_B3_a_r = struct[0].y_run[40,0]
    i_l_B2_B3_a_i = struct[0].y_run[41,0]
    i_l_B2_B3_b_r = struct[0].y_run[42,0]
    i_l_B2_B3_b_i = struct[0].y_run[43,0]
    i_l_B2_B3_c_r = struct[0].y_run[44,0]
    i_l_B2_B3_c_i = struct[0].y_run[45,0]
    i_l_B2_B3_n_r = struct[0].y_run[46,0]
    i_l_B2_B3_n_i = struct[0].y_run[47,0]
    i_l_B3_B4_a_r = struct[0].y_run[48,0]
    i_l_B3_B4_a_i = struct[0].y_run[49,0]
    i_l_B3_B4_b_r = struct[0].y_run[50,0]
    i_l_B3_B4_b_i = struct[0].y_run[51,0]
    i_l_B3_B4_c_r = struct[0].y_run[52,0]
    i_l_B3_B4_c_i = struct[0].y_run[53,0]
    i_l_B3_B4_n_r = struct[0].y_run[54,0]
    i_l_B3_B4_n_i = struct[0].y_run[55,0]
    i_load_B2_a_r = struct[0].y_run[56,0]
    i_load_B2_a_i = struct[0].y_run[57,0]
    i_load_B2_b_r = struct[0].y_run[58,0]
    i_load_B2_b_i = struct[0].y_run[59,0]
    i_load_B2_c_r = struct[0].y_run[60,0]
    i_load_B2_c_i = struct[0].y_run[61,0]
    i_load_B2_n_r = struct[0].y_run[62,0]
    i_load_B2_n_i = struct[0].y_run[63,0]
    i_load_B3_a_r = struct[0].y_run[64,0]
    i_load_B3_a_i = struct[0].y_run[65,0]
    i_load_B3_b_r = struct[0].y_run[66,0]
    i_load_B3_b_i = struct[0].y_run[67,0]
    i_load_B3_c_r = struct[0].y_run[68,0]
    i_load_B3_c_i = struct[0].y_run[69,0]
    i_load_B3_n_r = struct[0].y_run[70,0]
    i_load_B3_n_i = struct[0].y_run[71,0]
    i_B1_a_r = struct[0].y_run[72,0]
    i_B1_b_r = struct[0].y_run[73,0]
    i_B1_c_r = struct[0].y_run[74,0]
    i_B1_n_r = struct[0].y_run[75,0]
    i_B1_ng_r = struct[0].y_run[76,0]
    e_B1_ng_r = struct[0].y_run[77,0]
    i_B1_a_i = struct[0].y_run[78,0]
    i_B1_b_i = struct[0].y_run[79,0]
    i_B1_c_i = struct[0].y_run[80,0]
    i_B1_n_i = struct[0].y_run[81,0]
    i_B1_ng_i = struct[0].y_run[82,0]
    e_B1_ng_i = struct[0].y_run[83,0]
    i_B4_a_r = struct[0].y_run[84,0]
    i_B4_b_r = struct[0].y_run[85,0]
    i_B4_c_r = struct[0].y_run[86,0]
    i_B4_n_r = struct[0].y_run[87,0]
    i_B4_ng_r = struct[0].y_run[88,0]
    e_B4_ng_r = struct[0].y_run[89,0]
    i_B4_a_i = struct[0].y_run[90,0]
    i_B4_b_i = struct[0].y_run[91,0]
    i_B4_c_i = struct[0].y_run[92,0]
    i_B4_n_i = struct[0].y_run[93,0]
    i_B4_ng_i = struct[0].y_run[94,0]
    e_B4_ng_i = struct[0].y_run[95,0]
    omega_coi = struct[0].y_run[96,0]
    
    struct[0].u_run[0,0] = i_B2_n_r
    struct[0].u_run[1,0] = i_B2_n_i
    struct[0].u_run[2,0] = i_B3_n_r
    struct[0].u_run[3,0] = i_B3_n_i
    struct[0].u_run[4,0] = p_B2_a
    struct[0].u_run[5,0] = q_B2_a
    struct[0].u_run[6,0] = p_B2_b
    struct[0].u_run[7,0] = q_B2_b
    struct[0].u_run[8,0] = p_B2_c
    struct[0].u_run[9,0] = q_B2_c
    struct[0].u_run[10,0] = p_B3_a
    struct[0].u_run[11,0] = q_B3_a
    struct[0].u_run[12,0] = p_B3_b
    struct[0].u_run[13,0] = q_B3_b
    struct[0].u_run[14,0] = p_B3_c
    struct[0].u_run[15,0] = q_B3_c
    struct[0].u_run[16,0] = e_B1_an
    struct[0].u_run[17,0] = e_B1_bn
    struct[0].u_run[18,0] = e_B1_cn
    struct[0].u_run[19,0] = phi_B1
    struct[0].u_run[20,0] = p_B1_ref
    struct[0].u_run[21,0] = omega_B1_ref
    struct[0].u_run[22,0] = e_B4_an
    struct[0].u_run[23,0] = e_B4_bn
    struct[0].u_run[24,0] = e_B4_cn
    struct[0].u_run[25,0] = phi_B4
    struct[0].u_run[26,0] = p_B4_ref
    struct[0].u_run[27,0] = omega_B4_ref
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_B1*phi_B1 + 314.159265358979*omega_B1 - 314.159265358979*omega_coi
        struct[0].f[1,0] = (-K_f_B1*(K_sec_B1*xi_freq - 0.333333333333333*i_B1_a_i*v_B1_a_i - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) - 0.333333333333333*i_B1_a_r*v_B1_a_r - 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) - 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) - 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r - 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i + 0.166666666666667*i_B1_b_r*v_B1_c_r + 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r - 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i + 0.166666666666667*i_B1_c_r*v_B1_b_r - 0.333333333333333*i_B1_c_r*v_B1_c_r + p_B1_ref + v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) + v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) - v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) - v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r))/S_n_B1 - omega_B1 + omega_B1_ref)/T_f_B1
        struct[0].f[2,0] = -K_delta_B4*phi_B4 + 314.159265358979*omega_B4 - 314.159265358979*omega_coi
        struct[0].f[3,0] = (-K_f_B4*(K_sec_B4*xi_freq - 0.333333333333333*i_B4_a_i*v_B4_a_i - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) - 0.333333333333333*i_B4_a_r*v_B4_a_r - 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) - 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) - 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r - 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i + 0.166666666666667*i_B4_b_r*v_B4_c_r + 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r - 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i + 0.166666666666667*i_B4_c_r*v_B4_b_r - 0.333333333333333*i_B4_c_r*v_B4_c_r + p_B4_ref + v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) + v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) - v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) - v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r))/S_n_B4 - omega_B4 + omega_B4_ref)/T_f_B4
        struct[0].f[4,0] = K_agc*(1 - omega_coi)
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[56,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[57,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[58,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[59,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[60,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[61,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[64,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[65,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[66,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[67,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[68,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[69,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
        struct[0].g[72,0] = -R_B1_sa*i_B1_a_r + 1.0*X_B1_sa*i_B1_a_i + e_B1_an*cos(phi_B1) - v_B1_a_r + v_B1_n_r
        struct[0].g[73,0] = -R_B1_sb*i_B1_b_r + 1.0*X_B1_sb*i_B1_b_i + e_B1_bn*cos(phi_B1 - 2.0943951023932) - v_B1_b_r + v_B1_n_r
        struct[0].g[74,0] = -R_B1_sc*i_B1_c_r + 1.0*X_B1_sc*i_B1_c_i + e_B1_cn*cos(phi_B1 - 4.18879020478639) - v_B1_c_r + v_B1_n_r
        struct[0].g[78,0] = -1.0*R_B1_sa*i_B1_a_i - 1.0*X_B1_sa*i_B1_a_r + 1.0*e_B1_an*sin(phi_B1) - 1.0*v_B1_a_i + 1.0*v_B1_n_i
        struct[0].g[79,0] = -1.0*R_B1_sb*i_B1_b_i - 1.0*X_B1_sb*i_B1_b_r + 1.0*e_B1_bn*sin(phi_B1 - 2.0943951023932) - 1.0*v_B1_b_i + 1.0*v_B1_n_i
        struct[0].g[80,0] = -1.0*R_B1_sc*i_B1_c_i - 1.0*X_B1_sc*i_B1_c_r + 1.0*e_B1_cn*sin(phi_B1 - 4.18879020478639) - 1.0*v_B1_c_i + 1.0*v_B1_n_i
        struct[0].g[84,0] = -R_B4_sa*i_B4_a_r + 1.0*X_B4_sa*i_B4_a_i + e_B4_an*cos(phi_B4) - v_B4_a_r + v_B4_n_r
        struct[0].g[85,0] = -R_B4_sb*i_B4_b_r + 1.0*X_B4_sb*i_B4_b_i + e_B4_bn*cos(phi_B4 - 2.0943951023932) - v_B4_b_r + v_B4_n_r
        struct[0].g[86,0] = -R_B4_sc*i_B4_c_r + 1.0*X_B4_sc*i_B4_c_i + e_B4_cn*cos(phi_B4 - 4.18879020478639) - v_B4_c_r + v_B4_n_r
        struct[0].g[90,0] = -1.0*R_B4_sa*i_B4_a_i - 1.0*X_B4_sa*i_B4_a_r + 1.0*e_B4_an*sin(phi_B4) - 1.0*v_B4_a_i + 1.0*v_B4_n_i
        struct[0].g[91,0] = -1.0*R_B4_sb*i_B4_b_i - 1.0*X_B4_sb*i_B4_b_r + 1.0*e_B4_bn*sin(phi_B4 - 2.0943951023932) - 1.0*v_B4_b_i + 1.0*v_B4_n_i
        struct[0].g[92,0] = -1.0*R_B4_sc*i_B4_c_i - 1.0*X_B4_sc*i_B4_c_r + 1.0*e_B4_cn*sin(phi_B4 - 4.18879020478639) - 1.0*v_B4_c_i + 1.0*v_B4_n_i
        struct[0].g[96,0] = omega_coi - (S_n_B1*omega_B1 + S_n_B4*omega_B4)/(S_n_B1 + S_n_B4)
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[1,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[2,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[3,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[4,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[5,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[6,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[7,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[8,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[9,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[10,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[11,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[12,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[13,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[14,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
        struct[0].h[16,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i + 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r - 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i - 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) + v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[17,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r) + v_B1_a_r*(0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(-0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[18,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 0.333333333333333*i_B1_a_i*v_B1_b_i + 0.333333333333333*i_B1_a_i*v_B1_c_i + 0.333333333333333*i_B1_a_r*v_B1_a_r + 0.333333333333333*i_B1_a_r*v_B1_b_r + 0.333333333333333*i_B1_a_r*v_B1_c_r + 0.333333333333333*i_B1_b_i*v_B1_a_i + 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.333333333333333*i_B1_b_i*v_B1_c_i + 0.333333333333333*i_B1_b_r*v_B1_a_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.333333333333333*i_B1_b_r*v_B1_c_r + 0.333333333333333*i_B1_c_i*v_B1_a_i + 0.333333333333333*i_B1_c_i*v_B1_b_i + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.333333333333333*i_B1_c_r*v_B1_a_r + 0.333333333333333*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r
        struct[0].h[19,0] = e_B1_an
        struct[0].h[20,0] = e_B1_bn
        struct[0].h[21,0] = e_B1_cn
        struct[0].h[22,0] = p_B1_ref
        struct[0].h[23,0] = omega_B1_ref
        struct[0].h[24,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i + 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r - 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i - 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) + v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[25,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r) + v_B4_a_r*(0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(-0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[26,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 0.333333333333333*i_B4_a_i*v_B4_b_i + 0.333333333333333*i_B4_a_i*v_B4_c_i + 0.333333333333333*i_B4_a_r*v_B4_a_r + 0.333333333333333*i_B4_a_r*v_B4_b_r + 0.333333333333333*i_B4_a_r*v_B4_c_r + 0.333333333333333*i_B4_b_i*v_B4_a_i + 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.333333333333333*i_B4_b_i*v_B4_c_i + 0.333333333333333*i_B4_b_r*v_B4_a_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.333333333333333*i_B4_b_r*v_B4_c_r + 0.333333333333333*i_B4_c_i*v_B4_a_i + 0.333333333333333*i_B4_c_i*v_B4_b_i + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.333333333333333*i_B4_c_r*v_B4_a_r + 0.333333333333333*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r
        struct[0].h[27,0] = e_B4_an
        struct[0].h[28,0] = e_B4_bn
        struct[0].h[29,0] = e_B4_cn
        struct[0].h[30,0] = p_B4_ref
        struct[0].h[31,0] = omega_B4_ref
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_B1
        struct[0].Fx[1,1] = -1/T_f_B1
        struct[0].Fx[1,4] = -K_f_B1*K_sec_B1/(S_n_B1*T_f_B1)
        struct[0].Fx[2,2] = -K_delta_B4
        struct[0].Fx[3,3] = -1/T_f_B4
        struct[0].Fx[3,4] = -K_f_B4*K_sec_B4/(S_n_B4*T_f_B4)

    if mode == 11:

        struct[0].Fy[0,96] = -314.159265358979
        struct[0].Fy[1,16] = -K_f_B1*(-0.333333333333333*i_B1_a_r + 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,17] = -K_f_B1*(-0.333333333333333*i_B1_a_i + 0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r + 0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,18] = -K_f_B1*(-0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.333333333333333*i_B1_b_r + 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,19] = -K_f_B1*(0.166666666666667*i_B1_a_i + 0.288675134594813*i_B1_a_r - 0.333333333333333*i_B1_b_i + 0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,20] = -K_f_B1*(0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.333333333333333*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,21] = -K_f_B1*(0.166666666666667*i_B1_a_i - 0.288675134594813*i_B1_a_r + 0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r - 0.333333333333333*i_B1_c_i)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,72] = -K_f_B1*(-0.333333333333333*v_B1_a_r + 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,73] = -K_f_B1*(-0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.333333333333333*v_B1_b_r + 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,74] = -K_f_B1*(0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.333333333333333*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,78] = -K_f_B1*(-0.333333333333333*v_B1_a_i + 0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r + 0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,79] = -K_f_B1*(0.166666666666667*v_B1_a_i + 0.288675134594813*v_B1_a_r - 0.333333333333333*v_B1_b_i + 0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,80] = -K_f_B1*(0.166666666666667*v_B1_a_i - 0.288675134594813*v_B1_a_r + 0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r - 0.333333333333333*v_B1_c_i)/(S_n_B1*T_f_B1)
        struct[0].Fy[2,96] = -314.159265358979
        struct[0].Fy[3,24] = -K_f_B4*(-0.333333333333333*i_B4_a_r + 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,25] = -K_f_B4*(-0.333333333333333*i_B4_a_i + 0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r + 0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,26] = -K_f_B4*(-0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.333333333333333*i_B4_b_r + 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,27] = -K_f_B4*(0.166666666666667*i_B4_a_i + 0.288675134594813*i_B4_a_r - 0.333333333333333*i_B4_b_i + 0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,28] = -K_f_B4*(0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.333333333333333*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,29] = -K_f_B4*(0.166666666666667*i_B4_a_i - 0.288675134594813*i_B4_a_r + 0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r - 0.333333333333333*i_B4_c_i)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,84] = -K_f_B4*(-0.333333333333333*v_B4_a_r + 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,85] = -K_f_B4*(-0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.333333333333333*v_B4_b_r + 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,86] = -K_f_B4*(0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.333333333333333*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,90] = -K_f_B4*(-0.333333333333333*v_B4_a_i + 0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r + 0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,91] = -K_f_B4*(0.166666666666667*v_B4_a_i + 0.288675134594813*v_B4_a_r - 0.333333333333333*v_B4_b_i + 0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,92] = -K_f_B4*(0.166666666666667*v_B4_a_i - 0.288675134594813*v_B4_a_r + 0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r - 0.333333333333333*v_B4_c_i)/(S_n_B4*T_f_B4)
        struct[0].Fy[4,96] = -K_agc

        struct[0].Gx[72,0] = -e_B1_an*sin(phi_B1)
        struct[0].Gx[73,0] = -e_B1_bn*sin(phi_B1 - 2.0943951023932)
        struct[0].Gx[74,0] = -e_B1_cn*sin(phi_B1 - 4.18879020478639)
        struct[0].Gx[78,0] = 1.0*e_B1_an*cos(phi_B1)
        struct[0].Gx[79,0] = 1.0*e_B1_bn*cos(phi_B1 - 2.0943951023932)
        struct[0].Gx[80,0] = 1.0*e_B1_cn*cos(phi_B1 - 4.18879020478639)
        struct[0].Gx[84,2] = -e_B4_an*sin(phi_B4)
        struct[0].Gx[85,2] = -e_B4_bn*sin(phi_B4 - 2.0943951023932)
        struct[0].Gx[86,2] = -e_B4_cn*sin(phi_B4 - 4.18879020478639)
        struct[0].Gx[90,2] = 1.0*e_B4_an*cos(phi_B4)
        struct[0].Gx[91,2] = 1.0*e_B4_bn*cos(phi_B4 - 2.0943951023932)
        struct[0].Gx[92,2] = 1.0*e_B4_cn*cos(phi_B4 - 4.18879020478639)
        struct[0].Gx[96,1] = -S_n_B1/(S_n_B1 + S_n_B4)
        struct[0].Gx[96,3] = -S_n_B4/(S_n_B1 + S_n_B4)

        struct[0].Gy[56,0] = i_load_B2_a_r
        struct[0].Gy[56,1] = i_load_B2_a_i
        struct[0].Gy[56,6] = -i_load_B2_a_r
        struct[0].Gy[56,7] = -i_load_B2_a_i
        struct[0].Gy[56,56] = v_B2_a_r - v_B2_n_r
        struct[0].Gy[56,57] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[57,2] = i_load_B2_b_r
        struct[0].Gy[57,3] = i_load_B2_b_i
        struct[0].Gy[57,6] = -i_load_B2_b_r
        struct[0].Gy[57,7] = -i_load_B2_b_i
        struct[0].Gy[57,58] = v_B2_b_r - v_B2_n_r
        struct[0].Gy[57,59] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[58,4] = i_load_B2_c_r
        struct[0].Gy[58,5] = i_load_B2_c_i
        struct[0].Gy[58,6] = -i_load_B2_c_r
        struct[0].Gy[58,7] = -i_load_B2_c_i
        struct[0].Gy[58,60] = v_B2_c_r - v_B2_n_r
        struct[0].Gy[58,61] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[59,0] = -i_load_B2_a_i
        struct[0].Gy[59,1] = i_load_B2_a_r
        struct[0].Gy[59,6] = i_load_B2_a_i
        struct[0].Gy[59,7] = -i_load_B2_a_r
        struct[0].Gy[59,56] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[59,57] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy[60,2] = -i_load_B2_b_i
        struct[0].Gy[60,3] = i_load_B2_b_r
        struct[0].Gy[60,6] = i_load_B2_b_i
        struct[0].Gy[60,7] = -i_load_B2_b_r
        struct[0].Gy[60,58] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[60,59] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy[61,4] = -i_load_B2_c_i
        struct[0].Gy[61,5] = i_load_B2_c_r
        struct[0].Gy[61,6] = i_load_B2_c_i
        struct[0].Gy[61,7] = -i_load_B2_c_r
        struct[0].Gy[61,60] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[61,61] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy[64,8] = i_load_B3_a_r
        struct[0].Gy[64,9] = i_load_B3_a_i
        struct[0].Gy[64,14] = -i_load_B3_a_r
        struct[0].Gy[64,15] = -i_load_B3_a_i
        struct[0].Gy[64,64] = v_B3_a_r - v_B3_n_r
        struct[0].Gy[64,65] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[65,10] = i_load_B3_b_r
        struct[0].Gy[65,11] = i_load_B3_b_i
        struct[0].Gy[65,14] = -i_load_B3_b_r
        struct[0].Gy[65,15] = -i_load_B3_b_i
        struct[0].Gy[65,66] = v_B3_b_r - v_B3_n_r
        struct[0].Gy[65,67] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[66,12] = i_load_B3_c_r
        struct[0].Gy[66,13] = i_load_B3_c_i
        struct[0].Gy[66,14] = -i_load_B3_c_r
        struct[0].Gy[66,15] = -i_load_B3_c_i
        struct[0].Gy[66,68] = v_B3_c_r - v_B3_n_r
        struct[0].Gy[66,69] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[67,8] = -i_load_B3_a_i
        struct[0].Gy[67,9] = i_load_B3_a_r
        struct[0].Gy[67,14] = i_load_B3_a_i
        struct[0].Gy[67,15] = -i_load_B3_a_r
        struct[0].Gy[67,64] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[67,65] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy[68,10] = -i_load_B3_b_i
        struct[0].Gy[68,11] = i_load_B3_b_r
        struct[0].Gy[68,14] = i_load_B3_b_i
        struct[0].Gy[68,15] = -i_load_B3_b_r
        struct[0].Gy[68,66] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[68,67] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy[69,12] = -i_load_B3_c_i
        struct[0].Gy[69,13] = i_load_B3_c_r
        struct[0].Gy[69,14] = i_load_B3_c_i
        struct[0].Gy[69,15] = -i_load_B3_c_r
        struct[0].Gy[69,68] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[69,69] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy[72,72] = -R_B1_sa
        struct[0].Gy[72,78] = 1.0*X_B1_sa
        struct[0].Gy[73,73] = -R_B1_sb
        struct[0].Gy[73,79] = 1.0*X_B1_sb
        struct[0].Gy[74,74] = -R_B1_sc
        struct[0].Gy[74,80] = 1.0*X_B1_sc
        struct[0].Gy[75,75] = -R_B1_sn
        struct[0].Gy[75,81] = 1.0*X_B1_sn
        struct[0].Gy[77,76] = R_B1_ng
        struct[0].Gy[77,82] = -1.0*X_B1_ng
        struct[0].Gy[78,72] = -1.0*X_B1_sa
        struct[0].Gy[78,78] = -1.0*R_B1_sa
        struct[0].Gy[79,73] = -1.0*X_B1_sb
        struct[0].Gy[79,79] = -1.0*R_B1_sb
        struct[0].Gy[80,74] = -1.0*X_B1_sc
        struct[0].Gy[80,80] = -1.0*R_B1_sc
        struct[0].Gy[81,75] = -1.0*X_B1_sn
        struct[0].Gy[81,81] = -1.0*R_B1_sn
        struct[0].Gy[83,76] = 1.0*X_B1_ng
        struct[0].Gy[83,82] = 1.0*R_B1_ng
        struct[0].Gy[84,84] = -R_B4_sa
        struct[0].Gy[84,90] = 1.0*X_B4_sa
        struct[0].Gy[85,85] = -R_B4_sb
        struct[0].Gy[85,91] = 1.0*X_B4_sb
        struct[0].Gy[86,86] = -R_B4_sc
        struct[0].Gy[86,92] = 1.0*X_B4_sc
        struct[0].Gy[87,87] = -R_B4_sn
        struct[0].Gy[87,93] = 1.0*X_B4_sn
        struct[0].Gy[89,88] = R_B4_ng
        struct[0].Gy[89,94] = -1.0*X_B4_ng
        struct[0].Gy[90,84] = -1.0*X_B4_sa
        struct[0].Gy[90,90] = -1.0*R_B4_sa
        struct[0].Gy[91,85] = -1.0*X_B4_sb
        struct[0].Gy[91,91] = -1.0*R_B4_sb
        struct[0].Gy[92,86] = -1.0*X_B4_sc
        struct[0].Gy[92,92] = -1.0*R_B4_sc
        struct[0].Gy[93,87] = -1.0*X_B4_sn
        struct[0].Gy[93,93] = -1.0*R_B4_sn
        struct[0].Gy[95,88] = 1.0*X_B4_ng
        struct[0].Gy[95,94] = 1.0*R_B4_ng

    if mode > 12:

        struct[0].Fu[0,19] = -K_delta_B1
        struct[0].Fu[1,20] = -K_f_B1/(S_n_B1*T_f_B1)
        struct[0].Fu[1,21] = 1/T_f_B1
        struct[0].Fu[2,25] = -K_delta_B4
        struct[0].Fu[3,26] = -K_f_B4/(S_n_B4*T_f_B4)
        struct[0].Fu[3,27] = 1/T_f_B4

        struct[0].Gu[72,16] = cos(phi_B1)
        struct[0].Gu[72,19] = -e_B1_an*sin(phi_B1)
        struct[0].Gu[73,17] = cos(phi_B1 - 2.0943951023932)
        struct[0].Gu[73,19] = -e_B1_bn*sin(phi_B1 - 2.0943951023932)
        struct[0].Gu[74,18] = cos(phi_B1 - 4.18879020478639)
        struct[0].Gu[74,19] = -e_B1_cn*sin(phi_B1 - 4.18879020478639)
        struct[0].Gu[78,16] = 1.0*sin(phi_B1)
        struct[0].Gu[78,19] = 1.0*e_B1_an*cos(phi_B1)
        struct[0].Gu[79,17] = 1.0*sin(phi_B1 - 2.0943951023932)
        struct[0].Gu[79,19] = 1.0*e_B1_bn*cos(phi_B1 - 2.0943951023932)
        struct[0].Gu[80,18] = 1.0*sin(phi_B1 - 4.18879020478639)
        struct[0].Gu[80,19] = 1.0*e_B1_cn*cos(phi_B1 - 4.18879020478639)
        struct[0].Gu[84,22] = cos(phi_B4)
        struct[0].Gu[84,25] = -e_B4_an*sin(phi_B4)
        struct[0].Gu[85,23] = cos(phi_B4 - 2.0943951023932)
        struct[0].Gu[85,25] = -e_B4_bn*sin(phi_B4 - 2.0943951023932)
        struct[0].Gu[86,24] = cos(phi_B4 - 4.18879020478639)
        struct[0].Gu[86,25] = -e_B4_cn*sin(phi_B4 - 4.18879020478639)
        struct[0].Gu[90,22] = 1.0*sin(phi_B4)
        struct[0].Gu[90,25] = 1.0*e_B4_an*cos(phi_B4)
        struct[0].Gu[91,23] = 1.0*sin(phi_B4 - 2.0943951023932)
        struct[0].Gu[91,25] = 1.0*e_B4_bn*cos(phi_B4 - 2.0943951023932)
        struct[0].Gu[92,24] = 1.0*sin(phi_B4 - 4.18879020478639)
        struct[0].Gu[92,25] = 1.0*e_B4_cn*cos(phi_B4 - 4.18879020478639)


        struct[0].Hy[0,0] = 1.0*v_B2_a_r*(v_B2_a_i**2 + v_B2_a_r**2)**(-0.5)
        struct[0].Hy[0,1] = 1.0*v_B2_a_i*(v_B2_a_i**2 + v_B2_a_r**2)**(-0.5)
        struct[0].Hy[1,2] = 1.0*v_B2_b_r*(v_B2_b_i**2 + v_B2_b_r**2)**(-0.5)
        struct[0].Hy[1,3] = 1.0*v_B2_b_i*(v_B2_b_i**2 + v_B2_b_r**2)**(-0.5)
        struct[0].Hy[2,4] = 1.0*v_B2_c_r*(v_B2_c_i**2 + v_B2_c_r**2)**(-0.5)
        struct[0].Hy[2,5] = 1.0*v_B2_c_i*(v_B2_c_i**2 + v_B2_c_r**2)**(-0.5)
        struct[0].Hy[3,6] = 1.0*v_B2_n_r*(v_B2_n_i**2 + v_B2_n_r**2)**(-0.5)
        struct[0].Hy[3,7] = 1.0*v_B2_n_i*(v_B2_n_i**2 + v_B2_n_r**2)**(-0.5)
        struct[0].Hy[4,8] = 1.0*v_B3_a_r*(v_B3_a_i**2 + v_B3_a_r**2)**(-0.5)
        struct[0].Hy[4,9] = 1.0*v_B3_a_i*(v_B3_a_i**2 + v_B3_a_r**2)**(-0.5)
        struct[0].Hy[5,10] = 1.0*v_B3_b_r*(v_B3_b_i**2 + v_B3_b_r**2)**(-0.5)
        struct[0].Hy[5,11] = 1.0*v_B3_b_i*(v_B3_b_i**2 + v_B3_b_r**2)**(-0.5)
        struct[0].Hy[6,12] = 1.0*v_B3_c_r*(v_B3_c_i**2 + v_B3_c_r**2)**(-0.5)
        struct[0].Hy[6,13] = 1.0*v_B3_c_i*(v_B3_c_i**2 + v_B3_c_r**2)**(-0.5)
        struct[0].Hy[7,14] = 1.0*v_B3_n_r*(v_B3_n_i**2 + v_B3_n_r**2)**(-0.5)
        struct[0].Hy[7,15] = 1.0*v_B3_n_i*(v_B3_n_i**2 + v_B3_n_r**2)**(-0.5)
        struct[0].Hy[8,16] = 1.0*v_B1_a_r*(v_B1_a_i**2 + v_B1_a_r**2)**(-0.5)
        struct[0].Hy[8,17] = 1.0*v_B1_a_i*(v_B1_a_i**2 + v_B1_a_r**2)**(-0.5)
        struct[0].Hy[9,18] = 1.0*v_B1_b_r*(v_B1_b_i**2 + v_B1_b_r**2)**(-0.5)
        struct[0].Hy[9,19] = 1.0*v_B1_b_i*(v_B1_b_i**2 + v_B1_b_r**2)**(-0.5)
        struct[0].Hy[10,20] = 1.0*v_B1_c_r*(v_B1_c_i**2 + v_B1_c_r**2)**(-0.5)
        struct[0].Hy[10,21] = 1.0*v_B1_c_i*(v_B1_c_i**2 + v_B1_c_r**2)**(-0.5)
        struct[0].Hy[11,22] = 1.0*v_B1_n_r*(v_B1_n_i**2 + v_B1_n_r**2)**(-0.5)
        struct[0].Hy[11,23] = 1.0*v_B1_n_i*(v_B1_n_i**2 + v_B1_n_r**2)**(-0.5)
        struct[0].Hy[12,24] = 1.0*v_B4_a_r*(v_B4_a_i**2 + v_B4_a_r**2)**(-0.5)
        struct[0].Hy[12,25] = 1.0*v_B4_a_i*(v_B4_a_i**2 + v_B4_a_r**2)**(-0.5)
        struct[0].Hy[13,26] = 1.0*v_B4_b_r*(v_B4_b_i**2 + v_B4_b_r**2)**(-0.5)
        struct[0].Hy[13,27] = 1.0*v_B4_b_i*(v_B4_b_i**2 + v_B4_b_r**2)**(-0.5)
        struct[0].Hy[14,28] = 1.0*v_B4_c_r*(v_B4_c_i**2 + v_B4_c_r**2)**(-0.5)
        struct[0].Hy[14,29] = 1.0*v_B4_c_i*(v_B4_c_i**2 + v_B4_c_r**2)**(-0.5)
        struct[0].Hy[15,30] = 1.0*v_B4_n_r*(v_B4_n_i**2 + v_B4_n_r**2)**(-0.5)
        struct[0].Hy[15,31] = 1.0*v_B4_n_i*(v_B4_n_i**2 + v_B4_n_r**2)**(-0.5)
        struct[0].Hy[16,16] = 0.333333333333333*i_B1_a_r - 0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r + 0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r
        struct[0].Hy[16,17] = 0.333333333333333*i_B1_a_i - 0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r - 0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r
        struct[0].Hy[16,18] = 0.288675134594813*i_B1_a_i - 0.166666666666667*i_B1_a_r + 0.333333333333333*i_B1_b_r - 0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r
        struct[0].Hy[16,19] = -0.166666666666667*i_B1_a_i - 0.288675134594813*i_B1_a_r + 0.333333333333333*i_B1_b_i - 0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r
        struct[0].Hy[16,20] = -0.288675134594813*i_B1_a_i - 0.166666666666667*i_B1_a_r + 0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r + 0.333333333333333*i_B1_c_r
        struct[0].Hy[16,21] = -0.166666666666667*i_B1_a_i + 0.288675134594813*i_B1_a_r - 0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r + 0.333333333333333*i_B1_c_i
        struct[0].Hy[16,72] = 0.333333333333333*v_B1_a_r - 0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r + 0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r
        struct[0].Hy[16,73] = 0.288675134594813*v_B1_a_i - 0.166666666666667*v_B1_a_r + 0.333333333333333*v_B1_b_r - 0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r
        struct[0].Hy[16,74] = -0.288675134594813*v_B1_a_i - 0.166666666666667*v_B1_a_r + 0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r + 0.333333333333333*v_B1_c_r
        struct[0].Hy[16,78] = 0.333333333333333*v_B1_a_i - 0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r - 0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r
        struct[0].Hy[16,79] = -0.166666666666667*v_B1_a_i - 0.288675134594813*v_B1_a_r + 0.333333333333333*v_B1_b_i - 0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r
        struct[0].Hy[16,80] = -0.166666666666667*v_B1_a_i + 0.288675134594813*v_B1_a_r - 0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r + 0.333333333333333*v_B1_c_i
        struct[0].Hy[17,16] = 0.333333333333333*i_B1_a_r + 0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r - 0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r
        struct[0].Hy[17,17] = 0.333333333333333*i_B1_a_i - 0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r - 0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r
        struct[0].Hy[17,18] = -0.288675134594813*i_B1_a_i - 0.166666666666667*i_B1_a_r + 0.333333333333333*i_B1_b_r + 0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r
        struct[0].Hy[17,19] = -0.166666666666667*i_B1_a_i + 0.288675134594813*i_B1_a_r + 0.333333333333333*i_B1_b_i - 0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r
        struct[0].Hy[17,20] = 0.288675134594813*i_B1_a_i - 0.166666666666667*i_B1_a_r - 0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r + 0.333333333333333*i_B1_c_r
        struct[0].Hy[17,21] = -0.166666666666667*i_B1_a_i - 0.288675134594813*i_B1_a_r - 0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r + 0.333333333333333*i_B1_c_i
        struct[0].Hy[17,72] = 0.333333333333333*v_B1_a_r + 0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r - 0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r
        struct[0].Hy[17,73] = -0.288675134594813*v_B1_a_i - 0.166666666666667*v_B1_a_r + 0.333333333333333*v_B1_b_r + 0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r
        struct[0].Hy[17,74] = 0.288675134594813*v_B1_a_i - 0.166666666666667*v_B1_a_r - 0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r + 0.333333333333333*v_B1_c_r
        struct[0].Hy[17,78] = 0.333333333333333*v_B1_a_i - 0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r - 0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r
        struct[0].Hy[17,79] = -0.166666666666667*v_B1_a_i + 0.288675134594813*v_B1_a_r + 0.333333333333333*v_B1_b_i - 0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r
        struct[0].Hy[17,80] = -0.166666666666667*v_B1_a_i - 0.288675134594813*v_B1_a_r - 0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r + 0.333333333333333*v_B1_c_i
        struct[0].Hy[18,16] = 0.333333333333333*i_B1_a_r + 0.333333333333333*i_B1_b_r + 0.333333333333333*i_B1_c_r
        struct[0].Hy[18,17] = 0.333333333333333*i_B1_a_i + 0.333333333333333*i_B1_b_i + 0.333333333333333*i_B1_c_i
        struct[0].Hy[18,18] = 0.333333333333333*i_B1_a_r + 0.333333333333333*i_B1_b_r + 0.333333333333333*i_B1_c_r
        struct[0].Hy[18,19] = 0.333333333333333*i_B1_a_i + 0.333333333333333*i_B1_b_i + 0.333333333333333*i_B1_c_i
        struct[0].Hy[18,20] = 0.333333333333333*i_B1_a_r + 0.333333333333333*i_B1_b_r + 0.333333333333333*i_B1_c_r
        struct[0].Hy[18,21] = 0.333333333333333*i_B1_a_i + 0.333333333333333*i_B1_b_i + 0.333333333333333*i_B1_c_i
        struct[0].Hy[18,72] = 0.333333333333333*v_B1_a_r + 0.333333333333333*v_B1_b_r + 0.333333333333333*v_B1_c_r
        struct[0].Hy[18,73] = 0.333333333333333*v_B1_a_r + 0.333333333333333*v_B1_b_r + 0.333333333333333*v_B1_c_r
        struct[0].Hy[18,74] = 0.333333333333333*v_B1_a_r + 0.333333333333333*v_B1_b_r + 0.333333333333333*v_B1_c_r
        struct[0].Hy[18,78] = 0.333333333333333*v_B1_a_i + 0.333333333333333*v_B1_b_i + 0.333333333333333*v_B1_c_i
        struct[0].Hy[18,79] = 0.333333333333333*v_B1_a_i + 0.333333333333333*v_B1_b_i + 0.333333333333333*v_B1_c_i
        struct[0].Hy[18,80] = 0.333333333333333*v_B1_a_i + 0.333333333333333*v_B1_b_i + 0.333333333333333*v_B1_c_i
        struct[0].Hy[24,24] = 0.333333333333333*i_B4_a_r - 0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r + 0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r
        struct[0].Hy[24,25] = 0.333333333333333*i_B4_a_i - 0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r - 0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r
        struct[0].Hy[24,26] = 0.288675134594813*i_B4_a_i - 0.166666666666667*i_B4_a_r + 0.333333333333333*i_B4_b_r - 0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r
        struct[0].Hy[24,27] = -0.166666666666667*i_B4_a_i - 0.288675134594813*i_B4_a_r + 0.333333333333333*i_B4_b_i - 0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r
        struct[0].Hy[24,28] = -0.288675134594813*i_B4_a_i - 0.166666666666667*i_B4_a_r + 0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r + 0.333333333333333*i_B4_c_r
        struct[0].Hy[24,29] = -0.166666666666667*i_B4_a_i + 0.288675134594813*i_B4_a_r - 0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r + 0.333333333333333*i_B4_c_i
        struct[0].Hy[24,84] = 0.333333333333333*v_B4_a_r - 0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r + 0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r
        struct[0].Hy[24,85] = 0.288675134594813*v_B4_a_i - 0.166666666666667*v_B4_a_r + 0.333333333333333*v_B4_b_r - 0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r
        struct[0].Hy[24,86] = -0.288675134594813*v_B4_a_i - 0.166666666666667*v_B4_a_r + 0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r + 0.333333333333333*v_B4_c_r
        struct[0].Hy[24,90] = 0.333333333333333*v_B4_a_i - 0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r - 0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r
        struct[0].Hy[24,91] = -0.166666666666667*v_B4_a_i - 0.288675134594813*v_B4_a_r + 0.333333333333333*v_B4_b_i - 0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r
        struct[0].Hy[24,92] = -0.166666666666667*v_B4_a_i + 0.288675134594813*v_B4_a_r - 0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r + 0.333333333333333*v_B4_c_i
        struct[0].Hy[25,24] = 0.333333333333333*i_B4_a_r + 0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r - 0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r
        struct[0].Hy[25,25] = 0.333333333333333*i_B4_a_i - 0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r - 0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r
        struct[0].Hy[25,26] = -0.288675134594813*i_B4_a_i - 0.166666666666667*i_B4_a_r + 0.333333333333333*i_B4_b_r + 0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r
        struct[0].Hy[25,27] = -0.166666666666667*i_B4_a_i + 0.288675134594813*i_B4_a_r + 0.333333333333333*i_B4_b_i - 0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r
        struct[0].Hy[25,28] = 0.288675134594813*i_B4_a_i - 0.166666666666667*i_B4_a_r - 0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r + 0.333333333333333*i_B4_c_r
        struct[0].Hy[25,29] = -0.166666666666667*i_B4_a_i - 0.288675134594813*i_B4_a_r - 0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r + 0.333333333333333*i_B4_c_i
        struct[0].Hy[25,84] = 0.333333333333333*v_B4_a_r + 0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r - 0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r
        struct[0].Hy[25,85] = -0.288675134594813*v_B4_a_i - 0.166666666666667*v_B4_a_r + 0.333333333333333*v_B4_b_r + 0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r
        struct[0].Hy[25,86] = 0.288675134594813*v_B4_a_i - 0.166666666666667*v_B4_a_r - 0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r + 0.333333333333333*v_B4_c_r
        struct[0].Hy[25,90] = 0.333333333333333*v_B4_a_i - 0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r - 0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r
        struct[0].Hy[25,91] = -0.166666666666667*v_B4_a_i + 0.288675134594813*v_B4_a_r + 0.333333333333333*v_B4_b_i - 0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r
        struct[0].Hy[25,92] = -0.166666666666667*v_B4_a_i - 0.288675134594813*v_B4_a_r - 0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r + 0.333333333333333*v_B4_c_i
        struct[0].Hy[26,24] = 0.333333333333333*i_B4_a_r + 0.333333333333333*i_B4_b_r + 0.333333333333333*i_B4_c_r
        struct[0].Hy[26,25] = 0.333333333333333*i_B4_a_i + 0.333333333333333*i_B4_b_i + 0.333333333333333*i_B4_c_i
        struct[0].Hy[26,26] = 0.333333333333333*i_B4_a_r + 0.333333333333333*i_B4_b_r + 0.333333333333333*i_B4_c_r
        struct[0].Hy[26,27] = 0.333333333333333*i_B4_a_i + 0.333333333333333*i_B4_b_i + 0.333333333333333*i_B4_c_i
        struct[0].Hy[26,28] = 0.333333333333333*i_B4_a_r + 0.333333333333333*i_B4_b_r + 0.333333333333333*i_B4_c_r
        struct[0].Hy[26,29] = 0.333333333333333*i_B4_a_i + 0.333333333333333*i_B4_b_i + 0.333333333333333*i_B4_c_i
        struct[0].Hy[26,84] = 0.333333333333333*v_B4_a_r + 0.333333333333333*v_B4_b_r + 0.333333333333333*v_B4_c_r
        struct[0].Hy[26,85] = 0.333333333333333*v_B4_a_r + 0.333333333333333*v_B4_b_r + 0.333333333333333*v_B4_c_r
        struct[0].Hy[26,86] = 0.333333333333333*v_B4_a_r + 0.333333333333333*v_B4_b_r + 0.333333333333333*v_B4_c_r
        struct[0].Hy[26,90] = 0.333333333333333*v_B4_a_i + 0.333333333333333*v_B4_b_i + 0.333333333333333*v_B4_c_i
        struct[0].Hy[26,91] = 0.333333333333333*v_B4_a_i + 0.333333333333333*v_B4_b_i + 0.333333333333333*v_B4_c_i
        struct[0].Hy[26,92] = 0.333333333333333*v_B4_a_i + 0.333333333333333*v_B4_b_i + 0.333333333333333*v_B4_c_i

        struct[0].Hu[19,16] = 1
        struct[0].Hu[20,17] = 1
        struct[0].Hu[21,18] = 1
        struct[0].Hu[22,20] = 1
        struct[0].Hu[23,21] = 1
        struct[0].Hu[27,22] = 1
        struct[0].Hu[28,23] = 1
        struct[0].Hu[29,24] = 1
        struct[0].Hu[30,26] = 1
        struct[0].Hu[31,27] = 1



def ini_nn(struct,mode):

    # Parameters:
    X_B1_sa = struct[0].X_B1_sa
    R_B1_sa = struct[0].R_B1_sa
    X_B1_sb = struct[0].X_B1_sb
    R_B1_sb = struct[0].R_B1_sb
    X_B1_sc = struct[0].X_B1_sc
    R_B1_sc = struct[0].R_B1_sc
    X_B1_sn = struct[0].X_B1_sn
    R_B1_sn = struct[0].R_B1_sn
    S_n_B1 = struct[0].S_n_B1
    X_B1_ng = struct[0].X_B1_ng
    R_B1_ng = struct[0].R_B1_ng
    K_f_B1 = struct[0].K_f_B1
    T_f_B1 = struct[0].T_f_B1
    K_sec_B1 = struct[0].K_sec_B1
    K_delta_B1 = struct[0].K_delta_B1
    X_B4_sa = struct[0].X_B4_sa
    R_B4_sa = struct[0].R_B4_sa
    X_B4_sb = struct[0].X_B4_sb
    R_B4_sb = struct[0].R_B4_sb
    X_B4_sc = struct[0].X_B4_sc
    R_B4_sc = struct[0].R_B4_sc
    X_B4_sn = struct[0].X_B4_sn
    R_B4_sn = struct[0].R_B4_sn
    S_n_B4 = struct[0].S_n_B4
    X_B4_ng = struct[0].X_B4_ng
    R_B4_ng = struct[0].R_B4_ng
    K_f_B4 = struct[0].K_f_B4
    T_f_B4 = struct[0].T_f_B4
    K_sec_B4 = struct[0].K_sec_B4
    K_delta_B4 = struct[0].K_delta_B4
    K_agc = struct[0].K_agc
    
    # Inputs:
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
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
    e_B1_an = struct[0].e_B1_an
    e_B1_bn = struct[0].e_B1_bn
    e_B1_cn = struct[0].e_B1_cn
    phi_B1 = struct[0].phi_B1
    p_B1_ref = struct[0].p_B1_ref
    omega_B1_ref = struct[0].omega_B1_ref
    e_B4_an = struct[0].e_B4_an
    e_B4_bn = struct[0].e_B4_bn
    e_B4_cn = struct[0].e_B4_cn
    phi_B4 = struct[0].phi_B4
    p_B4_ref = struct[0].p_B4_ref
    omega_B4_ref = struct[0].omega_B4_ref
    
    # Dynamical states:
    phi_B1 = struct[0].x[0,0]
    omega_B1 = struct[0].x[1,0]
    phi_B4 = struct[0].x[2,0]
    omega_B4 = struct[0].x[3,0]
    xi_freq = struct[0].x[4,0]
    
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
    v_B1_a_r = struct[0].y_ini[16,0]
    v_B1_a_i = struct[0].y_ini[17,0]
    v_B1_b_r = struct[0].y_ini[18,0]
    v_B1_b_i = struct[0].y_ini[19,0]
    v_B1_c_r = struct[0].y_ini[20,0]
    v_B1_c_i = struct[0].y_ini[21,0]
    v_B1_n_r = struct[0].y_ini[22,0]
    v_B1_n_i = struct[0].y_ini[23,0]
    v_B4_a_r = struct[0].y_ini[24,0]
    v_B4_a_i = struct[0].y_ini[25,0]
    v_B4_b_r = struct[0].y_ini[26,0]
    v_B4_b_i = struct[0].y_ini[27,0]
    v_B4_c_r = struct[0].y_ini[28,0]
    v_B4_c_i = struct[0].y_ini[29,0]
    v_B4_n_r = struct[0].y_ini[30,0]
    v_B4_n_i = struct[0].y_ini[31,0]
    i_l_B1_B2_a_r = struct[0].y_ini[32,0]
    i_l_B1_B2_a_i = struct[0].y_ini[33,0]
    i_l_B1_B2_b_r = struct[0].y_ini[34,0]
    i_l_B1_B2_b_i = struct[0].y_ini[35,0]
    i_l_B1_B2_c_r = struct[0].y_ini[36,0]
    i_l_B1_B2_c_i = struct[0].y_ini[37,0]
    i_l_B1_B2_n_r = struct[0].y_ini[38,0]
    i_l_B1_B2_n_i = struct[0].y_ini[39,0]
    i_l_B2_B3_a_r = struct[0].y_ini[40,0]
    i_l_B2_B3_a_i = struct[0].y_ini[41,0]
    i_l_B2_B3_b_r = struct[0].y_ini[42,0]
    i_l_B2_B3_b_i = struct[0].y_ini[43,0]
    i_l_B2_B3_c_r = struct[0].y_ini[44,0]
    i_l_B2_B3_c_i = struct[0].y_ini[45,0]
    i_l_B2_B3_n_r = struct[0].y_ini[46,0]
    i_l_B2_B3_n_i = struct[0].y_ini[47,0]
    i_l_B3_B4_a_r = struct[0].y_ini[48,0]
    i_l_B3_B4_a_i = struct[0].y_ini[49,0]
    i_l_B3_B4_b_r = struct[0].y_ini[50,0]
    i_l_B3_B4_b_i = struct[0].y_ini[51,0]
    i_l_B3_B4_c_r = struct[0].y_ini[52,0]
    i_l_B3_B4_c_i = struct[0].y_ini[53,0]
    i_l_B3_B4_n_r = struct[0].y_ini[54,0]
    i_l_B3_B4_n_i = struct[0].y_ini[55,0]
    i_load_B2_a_r = struct[0].y_ini[56,0]
    i_load_B2_a_i = struct[0].y_ini[57,0]
    i_load_B2_b_r = struct[0].y_ini[58,0]
    i_load_B2_b_i = struct[0].y_ini[59,0]
    i_load_B2_c_r = struct[0].y_ini[60,0]
    i_load_B2_c_i = struct[0].y_ini[61,0]
    i_load_B2_n_r = struct[0].y_ini[62,0]
    i_load_B2_n_i = struct[0].y_ini[63,0]
    i_load_B3_a_r = struct[0].y_ini[64,0]
    i_load_B3_a_i = struct[0].y_ini[65,0]
    i_load_B3_b_r = struct[0].y_ini[66,0]
    i_load_B3_b_i = struct[0].y_ini[67,0]
    i_load_B3_c_r = struct[0].y_ini[68,0]
    i_load_B3_c_i = struct[0].y_ini[69,0]
    i_load_B3_n_r = struct[0].y_ini[70,0]
    i_load_B3_n_i = struct[0].y_ini[71,0]
    i_B1_a_r = struct[0].y_ini[72,0]
    i_B1_b_r = struct[0].y_ini[73,0]
    i_B1_c_r = struct[0].y_ini[74,0]
    i_B1_n_r = struct[0].y_ini[75,0]
    i_B1_ng_r = struct[0].y_ini[76,0]
    e_B1_ng_r = struct[0].y_ini[77,0]
    i_B1_a_i = struct[0].y_ini[78,0]
    i_B1_b_i = struct[0].y_ini[79,0]
    i_B1_c_i = struct[0].y_ini[80,0]
    i_B1_n_i = struct[0].y_ini[81,0]
    i_B1_ng_i = struct[0].y_ini[82,0]
    e_B1_ng_i = struct[0].y_ini[83,0]
    i_B4_a_r = struct[0].y_ini[84,0]
    i_B4_b_r = struct[0].y_ini[85,0]
    i_B4_c_r = struct[0].y_ini[86,0]
    i_B4_n_r = struct[0].y_ini[87,0]
    i_B4_ng_r = struct[0].y_ini[88,0]
    e_B4_ng_r = struct[0].y_ini[89,0]
    i_B4_a_i = struct[0].y_ini[90,0]
    i_B4_b_i = struct[0].y_ini[91,0]
    i_B4_c_i = struct[0].y_ini[92,0]
    i_B4_n_i = struct[0].y_ini[93,0]
    i_B4_ng_i = struct[0].y_ini[94,0]
    e_B4_ng_i = struct[0].y_ini[95,0]
    omega_coi = struct[0].y_ini[96,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_B1*phi_B1 + 314.159265358979*omega_B1 - 314.159265358979*omega_coi
        struct[0].f[1,0] = (-K_f_B1*(K_sec_B1*xi_freq - 0.333333333333333*i_B1_a_i*v_B1_a_i - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) - 0.333333333333333*i_B1_a_r*v_B1_a_r - 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) - 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) - 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r - 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i + 0.166666666666667*i_B1_b_r*v_B1_c_r + 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r - 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i + 0.166666666666667*i_B1_c_r*v_B1_b_r - 0.333333333333333*i_B1_c_r*v_B1_c_r + p_B1_ref + v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) + v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) - v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) - v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r))/S_n_B1 - omega_B1 + omega_B1_ref)/T_f_B1
        struct[0].f[2,0] = -K_delta_B4*phi_B4 + 314.159265358979*omega_B4 - 314.159265358979*omega_coi
        struct[0].f[3,0] = (-K_f_B4*(K_sec_B4*xi_freq - 0.333333333333333*i_B4_a_i*v_B4_a_i - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) - 0.333333333333333*i_B4_a_r*v_B4_a_r - 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) - 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) - 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r - 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i + 0.166666666666667*i_B4_b_r*v_B4_c_r + 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r - 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i + 0.166666666666667*i_B4_c_r*v_B4_b_r - 0.333333333333333*i_B4_c_r*v_B4_c_r + p_B4_ref + v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) + v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) - v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) - v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r))/S_n_B4 - omega_B4 + omega_B4_ref)/T_f_B4
        struct[0].f[4,0] = K_agc*(1 - omega_coi)
    
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
        struct[0].g[16,0] = i_B1_a_r - 116.655487182478*v_B1_a_i - 243.518329493424*v_B1_a_r + 116.655487182478*v_B2_a_i + 243.518329493424*v_B2_a_r
        struct[0].g[17,0] = i_B1_a_i - 243.518329493424*v_B1_a_i + 116.655487182478*v_B1_a_r + 243.518329493424*v_B2_a_i - 116.655487182478*v_B2_a_r
        struct[0].g[18,0] = i_B1_b_r - 116.655487182478*v_B1_b_i - 243.518329493424*v_B1_b_r + 116.655487182478*v_B2_b_i + 243.518329493424*v_B2_b_r
        struct[0].g[19,0] = i_B1_b_i - 243.518329493424*v_B1_b_i + 116.655487182478*v_B1_b_r + 243.518329493424*v_B2_b_i - 116.655487182478*v_B2_b_r
        struct[0].g[20,0] = i_B1_c_r - 116.655487182478*v_B1_c_i - 243.518329493424*v_B1_c_r + 116.655487182478*v_B2_c_i + 243.518329493424*v_B2_c_r
        struct[0].g[21,0] = i_B1_c_i - 243.518329493424*v_B1_c_i + 116.655487182478*v_B1_c_r + 243.518329493424*v_B2_c_i - 116.655487182478*v_B2_c_r
        struct[0].g[22,0] = i_B1_n_r - 116.655487182478*v_B1_n_i - 243.518329493424*v_B1_n_r + 116.655487182478*v_B2_n_i + 243.518329493424*v_B2_n_r
        struct[0].g[23,0] = i_B1_n_i - 243.518329493424*v_B1_n_i + 116.655487182478*v_B1_n_r + 243.518329493424*v_B2_n_i - 116.655487182478*v_B2_n_r
        struct[0].g[24,0] = i_B4_a_r + 116.655487182478*v_B3_a_i + 243.518329493424*v_B3_a_r - 116.655487182478*v_B4_a_i - 243.518329493424*v_B4_a_r
        struct[0].g[25,0] = i_B4_a_i + 243.518329493424*v_B3_a_i - 116.655487182478*v_B3_a_r - 243.518329493424*v_B4_a_i + 116.655487182478*v_B4_a_r
        struct[0].g[26,0] = i_B4_b_r + 116.655487182478*v_B3_b_i + 243.518329493424*v_B3_b_r - 116.655487182478*v_B4_b_i - 243.518329493424*v_B4_b_r
        struct[0].g[27,0] = i_B4_b_i + 243.518329493424*v_B3_b_i - 116.655487182478*v_B3_b_r - 243.518329493424*v_B4_b_i + 116.655487182478*v_B4_b_r
        struct[0].g[28,0] = i_B4_c_r + 116.655487182478*v_B3_c_i + 243.518329493424*v_B3_c_r - 116.655487182478*v_B4_c_i - 243.518329493424*v_B4_c_r
        struct[0].g[29,0] = i_B4_c_i + 243.518329493424*v_B3_c_i - 116.655487182478*v_B3_c_r - 243.518329493424*v_B4_c_i + 116.655487182478*v_B4_c_r
        struct[0].g[30,0] = i_B4_n_r + 116.655487182478*v_B3_n_i + 243.518329493424*v_B3_n_r - 116.655487182478*v_B4_n_i - 243.518329493424*v_B4_n_r
        struct[0].g[31,0] = i_B4_n_i + 243.518329493424*v_B3_n_i - 116.655487182478*v_B3_n_r - 243.518329493424*v_B4_n_i + 116.655487182478*v_B4_n_r
        struct[0].g[32,0] = -i_l_B1_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 116.655487182478*v_B2_a_i - 243.518329493424*v_B2_a_r
        struct[0].g[33,0] = -i_l_B1_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 243.518329493424*v_B2_a_i + 116.655487182478*v_B2_a_r
        struct[0].g[34,0] = -i_l_B1_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 116.655487182478*v_B2_b_i - 243.518329493424*v_B2_b_r
        struct[0].g[35,0] = -i_l_B1_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 243.518329493424*v_B2_b_i + 116.655487182478*v_B2_b_r
        struct[0].g[36,0] = -i_l_B1_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 116.655487182478*v_B2_c_i - 243.518329493424*v_B2_c_r
        struct[0].g[37,0] = -i_l_B1_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 243.518329493424*v_B2_c_i + 116.655487182478*v_B2_c_r
        struct[0].g[38,0] = i_l_B1_B2_a_r + i_l_B1_B2_b_r + i_l_B1_B2_c_r - i_l_B1_B2_n_r
        struct[0].g[39,0] = i_l_B1_B2_a_i + i_l_B1_B2_b_i + i_l_B1_B2_c_i - i_l_B1_B2_n_i
        struct[0].g[40,0] = -i_l_B2_B3_a_r + 23.3310974364957*v_B2_a_i + 48.7036658986847*v_B2_a_r - 23.3310974364957*v_B3_a_i - 48.7036658986847*v_B3_a_r
        struct[0].g[41,0] = -i_l_B2_B3_a_i + 48.7036658986847*v_B2_a_i - 23.3310974364957*v_B2_a_r - 48.7036658986847*v_B3_a_i + 23.3310974364957*v_B3_a_r
        struct[0].g[42,0] = -i_l_B2_B3_b_r + 23.3310974364957*v_B2_b_i + 48.7036658986847*v_B2_b_r - 23.3310974364957*v_B3_b_i - 48.7036658986847*v_B3_b_r
        struct[0].g[43,0] = -i_l_B2_B3_b_i + 48.7036658986847*v_B2_b_i - 23.3310974364957*v_B2_b_r - 48.7036658986847*v_B3_b_i + 23.3310974364957*v_B3_b_r
        struct[0].g[44,0] = -i_l_B2_B3_c_r + 23.3310974364957*v_B2_c_i + 48.7036658986847*v_B2_c_r - 23.3310974364957*v_B3_c_i - 48.7036658986847*v_B3_c_r
        struct[0].g[45,0] = -i_l_B2_B3_c_i + 48.7036658986847*v_B2_c_i - 23.3310974364957*v_B2_c_r - 48.7036658986847*v_B3_c_i + 23.3310974364957*v_B3_c_r
        struct[0].g[46,0] = i_l_B2_B3_a_r + i_l_B2_B3_b_r + i_l_B2_B3_c_r - i_l_B2_B3_n_r
        struct[0].g[47,0] = i_l_B2_B3_a_i + i_l_B2_B3_b_i + i_l_B2_B3_c_i - i_l_B2_B3_n_i
        struct[0].g[48,0] = -i_l_B3_B4_a_r + 116.655487182478*v_B3_a_i + 243.518329493424*v_B3_a_r - 116.655487182478*v_B4_a_i - 243.518329493424*v_B4_a_r
        struct[0].g[49,0] = -i_l_B3_B4_a_i + 243.518329493424*v_B3_a_i - 116.655487182478*v_B3_a_r - 243.518329493424*v_B4_a_i + 116.655487182478*v_B4_a_r
        struct[0].g[50,0] = -i_l_B3_B4_b_r + 116.655487182478*v_B3_b_i + 243.518329493424*v_B3_b_r - 116.655487182478*v_B4_b_i - 243.518329493424*v_B4_b_r
        struct[0].g[51,0] = -i_l_B3_B4_b_i + 243.518329493424*v_B3_b_i - 116.655487182478*v_B3_b_r - 243.518329493424*v_B4_b_i + 116.655487182478*v_B4_b_r
        struct[0].g[52,0] = -i_l_B3_B4_c_r + 116.655487182478*v_B3_c_i + 243.518329493424*v_B3_c_r - 116.655487182478*v_B4_c_i - 243.518329493424*v_B4_c_r
        struct[0].g[53,0] = -i_l_B3_B4_c_i + 243.518329493424*v_B3_c_i - 116.655487182478*v_B3_c_r - 243.518329493424*v_B4_c_i + 116.655487182478*v_B4_c_r
        struct[0].g[54,0] = i_l_B3_B4_a_r + i_l_B3_B4_b_r + i_l_B3_B4_c_r - i_l_B3_B4_n_r
        struct[0].g[55,0] = i_l_B3_B4_a_i + i_l_B3_B4_b_i + i_l_B3_B4_c_i - i_l_B3_B4_n_i
        struct[0].g[56,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[57,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[58,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[59,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[60,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[61,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[62,0] = i_load_B2_a_r + i_load_B2_b_r + i_load_B2_c_r + i_load_B2_n_r
        struct[0].g[63,0] = i_load_B2_a_i + i_load_B2_b_i + i_load_B2_c_i + i_load_B2_n_i
        struct[0].g[64,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[65,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[66,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[67,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[68,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[69,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
        struct[0].g[70,0] = i_load_B3_a_r + i_load_B3_b_r + i_load_B3_c_r + i_load_B3_n_r
        struct[0].g[71,0] = i_load_B3_a_i + i_load_B3_b_i + i_load_B3_c_i + i_load_B3_n_i
        struct[0].g[72,0] = -R_B1_sa*i_B1_a_r + 1.0*X_B1_sa*i_B1_a_i + e_B1_an*cos(phi_B1) - v_B1_a_r + v_B1_n_r
        struct[0].g[73,0] = -R_B1_sb*i_B1_b_r + 1.0*X_B1_sb*i_B1_b_i + e_B1_bn*cos(phi_B1 - 2.0943951023932) - v_B1_b_r + v_B1_n_r
        struct[0].g[74,0] = -R_B1_sc*i_B1_c_r + 1.0*X_B1_sc*i_B1_c_i + e_B1_cn*cos(phi_B1 - 4.18879020478639) - v_B1_c_r + v_B1_n_r
        struct[0].g[75,0] = -R_B1_sn*i_B1_n_r + 1.0*X_B1_sn*i_B1_n_i + e_B1_ng_r - v_B1_n_r
        struct[0].g[76,0] = i_B1_a_r + i_B1_b_r + i_B1_c_r + i_B1_n_r - i_B1_ng_r
        struct[0].g[77,0] = R_B1_ng*i_B1_ng_r - 1.0*X_B1_ng*i_B1_ng_i - e_B1_ng_r
        struct[0].g[78,0] = -1.0*R_B1_sa*i_B1_a_i - 1.0*X_B1_sa*i_B1_a_r + 1.0*e_B1_an*sin(phi_B1) - 1.0*v_B1_a_i + 1.0*v_B1_n_i
        struct[0].g[79,0] = -1.0*R_B1_sb*i_B1_b_i - 1.0*X_B1_sb*i_B1_b_r + 1.0*e_B1_bn*sin(phi_B1 - 2.0943951023932) - 1.0*v_B1_b_i + 1.0*v_B1_n_i
        struct[0].g[80,0] = -1.0*R_B1_sc*i_B1_c_i - 1.0*X_B1_sc*i_B1_c_r + 1.0*e_B1_cn*sin(phi_B1 - 4.18879020478639) - 1.0*v_B1_c_i + 1.0*v_B1_n_i
        struct[0].g[81,0] = -1.0*R_B1_sn*i_B1_n_i - 1.0*X_B1_sn*i_B1_n_r + 1.0*e_B1_ng_i - 1.0*v_B1_n_i
        struct[0].g[82,0] = 1.0*i_B1_a_i + 1.0*i_B1_b_i + 1.0*i_B1_c_i + 1.0*i_B1_n_i - 1.0*i_B1_ng_i
        struct[0].g[83,0] = 1.0*R_B1_ng*i_B1_ng_i + 1.0*X_B1_ng*i_B1_ng_r - 1.0*e_B1_ng_i
        struct[0].g[84,0] = -R_B4_sa*i_B4_a_r + 1.0*X_B4_sa*i_B4_a_i + e_B4_an*cos(phi_B4) - v_B4_a_r + v_B4_n_r
        struct[0].g[85,0] = -R_B4_sb*i_B4_b_r + 1.0*X_B4_sb*i_B4_b_i + e_B4_bn*cos(phi_B4 - 2.0943951023932) - v_B4_b_r + v_B4_n_r
        struct[0].g[86,0] = -R_B4_sc*i_B4_c_r + 1.0*X_B4_sc*i_B4_c_i + e_B4_cn*cos(phi_B4 - 4.18879020478639) - v_B4_c_r + v_B4_n_r
        struct[0].g[87,0] = -R_B4_sn*i_B4_n_r + 1.0*X_B4_sn*i_B4_n_i + e_B4_ng_r - v_B4_n_r
        struct[0].g[88,0] = i_B4_a_r + i_B4_b_r + i_B4_c_r + i_B4_n_r - i_B4_ng_r
        struct[0].g[89,0] = R_B4_ng*i_B4_ng_r - 1.0*X_B4_ng*i_B4_ng_i - e_B4_ng_r
        struct[0].g[90,0] = -1.0*R_B4_sa*i_B4_a_i - 1.0*X_B4_sa*i_B4_a_r + 1.0*e_B4_an*sin(phi_B4) - 1.0*v_B4_a_i + 1.0*v_B4_n_i
        struct[0].g[91,0] = -1.0*R_B4_sb*i_B4_b_i - 1.0*X_B4_sb*i_B4_b_r + 1.0*e_B4_bn*sin(phi_B4 - 2.0943951023932) - 1.0*v_B4_b_i + 1.0*v_B4_n_i
        struct[0].g[92,0] = -1.0*R_B4_sc*i_B4_c_i - 1.0*X_B4_sc*i_B4_c_r + 1.0*e_B4_cn*sin(phi_B4 - 4.18879020478639) - 1.0*v_B4_c_i + 1.0*v_B4_n_i
        struct[0].g[93,0] = -1.0*R_B4_sn*i_B4_n_i - 1.0*X_B4_sn*i_B4_n_r + 1.0*e_B4_ng_i - 1.0*v_B4_n_i
        struct[0].g[94,0] = 1.0*i_B4_a_i + 1.0*i_B4_b_i + 1.0*i_B4_c_i + 1.0*i_B4_n_i - 1.0*i_B4_ng_i
        struct[0].g[95,0] = 1.0*R_B4_ng*i_B4_ng_i + 1.0*X_B4_ng*i_B4_ng_r - 1.0*e_B4_ng_i
        struct[0].g[96,0] = omega_coi - (S_n_B1*omega_B1 + S_n_B4*omega_B4)/(S_n_B1 + S_n_B4)
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[1,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[2,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[3,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[4,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[5,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[6,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[7,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[8,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[9,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[10,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[11,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[12,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[13,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[14,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
        struct[0].h[16,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i + 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r - 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i - 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) + v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[17,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r) + v_B1_a_r*(0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(-0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[18,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 0.333333333333333*i_B1_a_i*v_B1_b_i + 0.333333333333333*i_B1_a_i*v_B1_c_i + 0.333333333333333*i_B1_a_r*v_B1_a_r + 0.333333333333333*i_B1_a_r*v_B1_b_r + 0.333333333333333*i_B1_a_r*v_B1_c_r + 0.333333333333333*i_B1_b_i*v_B1_a_i + 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.333333333333333*i_B1_b_i*v_B1_c_i + 0.333333333333333*i_B1_b_r*v_B1_a_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.333333333333333*i_B1_b_r*v_B1_c_r + 0.333333333333333*i_B1_c_i*v_B1_a_i + 0.333333333333333*i_B1_c_i*v_B1_b_i + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.333333333333333*i_B1_c_r*v_B1_a_r + 0.333333333333333*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r
        struct[0].h[19,0] = e_B1_an
        struct[0].h[20,0] = e_B1_bn
        struct[0].h[21,0] = e_B1_cn
        struct[0].h[22,0] = p_B1_ref
        struct[0].h[23,0] = omega_B1_ref
        struct[0].h[24,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i + 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r - 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i - 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) + v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[25,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r) + v_B4_a_r*(0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(-0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[26,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 0.333333333333333*i_B4_a_i*v_B4_b_i + 0.333333333333333*i_B4_a_i*v_B4_c_i + 0.333333333333333*i_B4_a_r*v_B4_a_r + 0.333333333333333*i_B4_a_r*v_B4_b_r + 0.333333333333333*i_B4_a_r*v_B4_c_r + 0.333333333333333*i_B4_b_i*v_B4_a_i + 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.333333333333333*i_B4_b_i*v_B4_c_i + 0.333333333333333*i_B4_b_r*v_B4_a_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.333333333333333*i_B4_b_r*v_B4_c_r + 0.333333333333333*i_B4_c_i*v_B4_a_i + 0.333333333333333*i_B4_c_i*v_B4_b_i + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.333333333333333*i_B4_c_r*v_B4_a_r + 0.333333333333333*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r
        struct[0].h[27,0] = e_B4_an
        struct[0].h[28,0] = e_B4_bn
        struct[0].h[29,0] = e_B4_cn
        struct[0].h[30,0] = p_B4_ref
        struct[0].h[31,0] = omega_B4_ref
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -K_delta_B1
        struct[0].Fx_ini[0,1] = 314.159265358979
        struct[0].Fx_ini[1,1] = -1/T_f_B1
        struct[0].Fx_ini[1,4] = -K_f_B1*K_sec_B1/(S_n_B1*T_f_B1)
        struct[0].Fx_ini[2,2] = -K_delta_B4
        struct[0].Fx_ini[2,3] = 314.159265358979
        struct[0].Fx_ini[3,3] = -1/T_f_B4
        struct[0].Fx_ini[3,4] = -K_f_B4*K_sec_B4/(S_n_B4*T_f_B4)

    if mode == 11:

        struct[0].Fy_ini[0,96] = -314.159265358979 
        struct[0].Fy_ini[1,16] = -K_f_B1*(-0.333333333333333*i_B1_a_r + 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,17] = -K_f_B1*(-0.333333333333333*i_B1_a_i + 0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r + 0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,18] = -K_f_B1*(-0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.333333333333333*i_B1_b_r + 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,19] = -K_f_B1*(0.166666666666667*i_B1_a_i + 0.288675134594813*i_B1_a_r - 0.333333333333333*i_B1_b_i + 0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,20] = -K_f_B1*(0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.333333333333333*i_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,21] = -K_f_B1*(0.166666666666667*i_B1_a_i - 0.288675134594813*i_B1_a_r + 0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r - 0.333333333333333*i_B1_c_i)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,72] = -K_f_B1*(-0.333333333333333*v_B1_a_r + 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,73] = -K_f_B1*(-0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.333333333333333*v_B1_b_r + 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,74] = -K_f_B1*(0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.333333333333333*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,78] = -K_f_B1*(-0.333333333333333*v_B1_a_i + 0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r + 0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,79] = -K_f_B1*(0.166666666666667*v_B1_a_i + 0.288675134594813*v_B1_a_r - 0.333333333333333*v_B1_b_i + 0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[1,80] = -K_f_B1*(0.166666666666667*v_B1_a_i - 0.288675134594813*v_B1_a_r + 0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r - 0.333333333333333*v_B1_c_i)/(S_n_B1*T_f_B1) 
        struct[0].Fy_ini[2,96] = -314.159265358979 
        struct[0].Fy_ini[3,24] = -K_f_B4*(-0.333333333333333*i_B4_a_r + 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,25] = -K_f_B4*(-0.333333333333333*i_B4_a_i + 0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r + 0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,26] = -K_f_B4*(-0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.333333333333333*i_B4_b_r + 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,27] = -K_f_B4*(0.166666666666667*i_B4_a_i + 0.288675134594813*i_B4_a_r - 0.333333333333333*i_B4_b_i + 0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,28] = -K_f_B4*(0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.333333333333333*i_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,29] = -K_f_B4*(0.166666666666667*i_B4_a_i - 0.288675134594813*i_B4_a_r + 0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r - 0.333333333333333*i_B4_c_i)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,84] = -K_f_B4*(-0.333333333333333*v_B4_a_r + 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,85] = -K_f_B4*(-0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.333333333333333*v_B4_b_r + 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,86] = -K_f_B4*(0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.333333333333333*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,90] = -K_f_B4*(-0.333333333333333*v_B4_a_i + 0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r + 0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,91] = -K_f_B4*(0.166666666666667*v_B4_a_i + 0.288675134594813*v_B4_a_r - 0.333333333333333*v_B4_b_i + 0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[3,92] = -K_f_B4*(0.166666666666667*v_B4_a_i - 0.288675134594813*v_B4_a_r + 0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r - 0.333333333333333*v_B4_c_i)/(S_n_B4*T_f_B4) 
        struct[0].Fy_ini[4,96] = -K_agc 

        struct[0].Gy_ini[0,0] = -292.221995392108
        struct[0].Gy_ini[0,1] = -139.986584618974
        struct[0].Gy_ini[0,8] = 48.7036658986847
        struct[0].Gy_ini[0,9] = 23.3310974364957
        struct[0].Gy_ini[0,16] = 243.518329493424
        struct[0].Gy_ini[0,17] = 116.655487182478
        struct[0].Gy_ini[0,56] = 1
        struct[0].Gy_ini[1,0] = 139.986584618974
        struct[0].Gy_ini[1,1] = -292.221995392108
        struct[0].Gy_ini[1,8] = -23.3310974364957
        struct[0].Gy_ini[1,9] = 48.7036658986847
        struct[0].Gy_ini[1,16] = -116.655487182478
        struct[0].Gy_ini[1,17] = 243.518329493424
        struct[0].Gy_ini[1,57] = 1
        struct[0].Gy_ini[2,2] = -292.221995392108
        struct[0].Gy_ini[2,3] = -139.986584618974
        struct[0].Gy_ini[2,10] = 48.7036658986847
        struct[0].Gy_ini[2,11] = 23.3310974364957
        struct[0].Gy_ini[2,18] = 243.518329493424
        struct[0].Gy_ini[2,19] = 116.655487182478
        struct[0].Gy_ini[2,58] = 1
        struct[0].Gy_ini[3,2] = 139.986584618974
        struct[0].Gy_ini[3,3] = -292.221995392108
        struct[0].Gy_ini[3,10] = -23.3310974364957
        struct[0].Gy_ini[3,11] = 48.7036658986847
        struct[0].Gy_ini[3,18] = -116.655487182478
        struct[0].Gy_ini[3,19] = 243.518329493424
        struct[0].Gy_ini[3,59] = 1
        struct[0].Gy_ini[4,4] = -292.221995392108
        struct[0].Gy_ini[4,5] = -139.986584618974
        struct[0].Gy_ini[4,12] = 48.7036658986847
        struct[0].Gy_ini[4,13] = 23.3310974364957
        struct[0].Gy_ini[4,20] = 243.518329493424
        struct[0].Gy_ini[4,21] = 116.655487182478
        struct[0].Gy_ini[4,60] = 1
        struct[0].Gy_ini[5,4] = 139.986584618974
        struct[0].Gy_ini[5,5] = -292.221995392108
        struct[0].Gy_ini[5,12] = -23.3310974364957
        struct[0].Gy_ini[5,13] = 48.7036658986847
        struct[0].Gy_ini[5,20] = -116.655487182478
        struct[0].Gy_ini[5,21] = 243.518329493424
        struct[0].Gy_ini[5,61] = 1
        struct[0].Gy_ini[6,6] = -292.221995392108
        struct[0].Gy_ini[6,7] = -139.986584618974
        struct[0].Gy_ini[6,14] = 48.7036658986847
        struct[0].Gy_ini[6,15] = 23.3310974364957
        struct[0].Gy_ini[6,22] = 243.518329493424
        struct[0].Gy_ini[6,23] = 116.655487182478
        struct[0].Gy_ini[7,6] = 139.986584618974
        struct[0].Gy_ini[7,7] = -292.221995392108
        struct[0].Gy_ini[7,14] = -23.3310974364957
        struct[0].Gy_ini[7,15] = 48.7036658986847
        struct[0].Gy_ini[7,22] = -116.655487182478
        struct[0].Gy_ini[7,23] = 243.518329493424
        struct[0].Gy_ini[8,0] = 48.7036658986847
        struct[0].Gy_ini[8,1] = 23.3310974364957
        struct[0].Gy_ini[8,8] = -292.221995392108
        struct[0].Gy_ini[8,9] = -139.986584618974
        struct[0].Gy_ini[8,24] = 243.518329493424
        struct[0].Gy_ini[8,25] = 116.655487182478
        struct[0].Gy_ini[8,64] = 1
        struct[0].Gy_ini[9,0] = -23.3310974364957
        struct[0].Gy_ini[9,1] = 48.7036658986847
        struct[0].Gy_ini[9,8] = 139.986584618974
        struct[0].Gy_ini[9,9] = -292.221995392108
        struct[0].Gy_ini[9,24] = -116.655487182478
        struct[0].Gy_ini[9,25] = 243.518329493424
        struct[0].Gy_ini[9,65] = 1
        struct[0].Gy_ini[10,2] = 48.7036658986847
        struct[0].Gy_ini[10,3] = 23.3310974364957
        struct[0].Gy_ini[10,10] = -292.221995392108
        struct[0].Gy_ini[10,11] = -139.986584618974
        struct[0].Gy_ini[10,26] = 243.518329493424
        struct[0].Gy_ini[10,27] = 116.655487182478
        struct[0].Gy_ini[10,66] = 1
        struct[0].Gy_ini[11,2] = -23.3310974364957
        struct[0].Gy_ini[11,3] = 48.7036658986847
        struct[0].Gy_ini[11,10] = 139.986584618974
        struct[0].Gy_ini[11,11] = -292.221995392108
        struct[0].Gy_ini[11,26] = -116.655487182478
        struct[0].Gy_ini[11,27] = 243.518329493424
        struct[0].Gy_ini[11,67] = 1
        struct[0].Gy_ini[12,4] = 48.7036658986847
        struct[0].Gy_ini[12,5] = 23.3310974364957
        struct[0].Gy_ini[12,12] = -292.221995392108
        struct[0].Gy_ini[12,13] = -139.986584618974
        struct[0].Gy_ini[12,28] = 243.518329493424
        struct[0].Gy_ini[12,29] = 116.655487182478
        struct[0].Gy_ini[12,68] = 1
        struct[0].Gy_ini[13,4] = -23.3310974364957
        struct[0].Gy_ini[13,5] = 48.7036658986847
        struct[0].Gy_ini[13,12] = 139.986584618974
        struct[0].Gy_ini[13,13] = -292.221995392108
        struct[0].Gy_ini[13,28] = -116.655487182478
        struct[0].Gy_ini[13,29] = 243.518329493424
        struct[0].Gy_ini[13,69] = 1
        struct[0].Gy_ini[14,6] = 48.7036658986847
        struct[0].Gy_ini[14,7] = 23.3310974364957
        struct[0].Gy_ini[14,14] = -292.221995392108
        struct[0].Gy_ini[14,15] = -139.986584618974
        struct[0].Gy_ini[14,30] = 243.518329493424
        struct[0].Gy_ini[14,31] = 116.655487182478
        struct[0].Gy_ini[15,6] = -23.3310974364957
        struct[0].Gy_ini[15,7] = 48.7036658986847
        struct[0].Gy_ini[15,14] = 139.986584618974
        struct[0].Gy_ini[15,15] = -292.221995392108
        struct[0].Gy_ini[15,30] = -116.655487182478
        struct[0].Gy_ini[15,31] = 243.518329493424
        struct[0].Gy_ini[16,0] = 243.518329493424
        struct[0].Gy_ini[16,1] = 116.655487182478
        struct[0].Gy_ini[16,16] = -243.518329493424
        struct[0].Gy_ini[16,17] = -116.655487182478
        struct[0].Gy_ini[16,72] = 1
        struct[0].Gy_ini[17,0] = -116.655487182478
        struct[0].Gy_ini[17,1] = 243.518329493424
        struct[0].Gy_ini[17,16] = 116.655487182478
        struct[0].Gy_ini[17,17] = -243.518329493424
        struct[0].Gy_ini[17,78] = 1
        struct[0].Gy_ini[18,2] = 243.518329493424
        struct[0].Gy_ini[18,3] = 116.655487182478
        struct[0].Gy_ini[18,18] = -243.518329493424
        struct[0].Gy_ini[18,19] = -116.655487182478
        struct[0].Gy_ini[18,73] = 1
        struct[0].Gy_ini[19,2] = -116.655487182478
        struct[0].Gy_ini[19,3] = 243.518329493424
        struct[0].Gy_ini[19,18] = 116.655487182478
        struct[0].Gy_ini[19,19] = -243.518329493424
        struct[0].Gy_ini[19,79] = 1
        struct[0].Gy_ini[20,4] = 243.518329493424
        struct[0].Gy_ini[20,5] = 116.655487182478
        struct[0].Gy_ini[20,20] = -243.518329493424
        struct[0].Gy_ini[20,21] = -116.655487182478
        struct[0].Gy_ini[20,74] = 1
        struct[0].Gy_ini[21,4] = -116.655487182478
        struct[0].Gy_ini[21,5] = 243.518329493424
        struct[0].Gy_ini[21,20] = 116.655487182478
        struct[0].Gy_ini[21,21] = -243.518329493424
        struct[0].Gy_ini[21,80] = 1
        struct[0].Gy_ini[22,6] = 243.518329493424
        struct[0].Gy_ini[22,7] = 116.655487182478
        struct[0].Gy_ini[22,22] = -243.518329493424
        struct[0].Gy_ini[22,23] = -116.655487182478
        struct[0].Gy_ini[22,75] = 1
        struct[0].Gy_ini[23,6] = -116.655487182478
        struct[0].Gy_ini[23,7] = 243.518329493424
        struct[0].Gy_ini[23,22] = 116.655487182478
        struct[0].Gy_ini[23,23] = -243.518329493424
        struct[0].Gy_ini[23,81] = 1
        struct[0].Gy_ini[24,8] = 243.518329493424
        struct[0].Gy_ini[24,9] = 116.655487182478
        struct[0].Gy_ini[24,24] = -243.518329493424
        struct[0].Gy_ini[24,25] = -116.655487182478
        struct[0].Gy_ini[24,84] = 1
        struct[0].Gy_ini[25,8] = -116.655487182478
        struct[0].Gy_ini[25,9] = 243.518329493424
        struct[0].Gy_ini[25,24] = 116.655487182478
        struct[0].Gy_ini[25,25] = -243.518329493424
        struct[0].Gy_ini[25,90] = 1
        struct[0].Gy_ini[26,10] = 243.518329493424
        struct[0].Gy_ini[26,11] = 116.655487182478
        struct[0].Gy_ini[26,26] = -243.518329493424
        struct[0].Gy_ini[26,27] = -116.655487182478
        struct[0].Gy_ini[26,85] = 1
        struct[0].Gy_ini[27,10] = -116.655487182478
        struct[0].Gy_ini[27,11] = 243.518329493424
        struct[0].Gy_ini[27,26] = 116.655487182478
        struct[0].Gy_ini[27,27] = -243.518329493424
        struct[0].Gy_ini[27,91] = 1
        struct[0].Gy_ini[28,12] = 243.518329493424
        struct[0].Gy_ini[28,13] = 116.655487182478
        struct[0].Gy_ini[28,28] = -243.518329493424
        struct[0].Gy_ini[28,29] = -116.655487182478
        struct[0].Gy_ini[28,86] = 1
        struct[0].Gy_ini[29,12] = -116.655487182478
        struct[0].Gy_ini[29,13] = 243.518329493424
        struct[0].Gy_ini[29,28] = 116.655487182478
        struct[0].Gy_ini[29,29] = -243.518329493424
        struct[0].Gy_ini[29,92] = 1
        struct[0].Gy_ini[30,14] = 243.518329493424
        struct[0].Gy_ini[30,15] = 116.655487182478
        struct[0].Gy_ini[30,30] = -243.518329493424
        struct[0].Gy_ini[30,31] = -116.655487182478
        struct[0].Gy_ini[30,87] = 1
        struct[0].Gy_ini[31,14] = -116.655487182478
        struct[0].Gy_ini[31,15] = 243.518329493424
        struct[0].Gy_ini[31,30] = 116.655487182478
        struct[0].Gy_ini[31,31] = -243.518329493424
        struct[0].Gy_ini[31,93] = 1
        struct[0].Gy_ini[32,0] = -243.518329493424
        struct[0].Gy_ini[32,1] = -116.655487182478
        struct[0].Gy_ini[32,16] = 243.518329493424
        struct[0].Gy_ini[32,17] = 116.655487182478
        struct[0].Gy_ini[32,32] = -1
        struct[0].Gy_ini[33,0] = 116.655487182478
        struct[0].Gy_ini[33,1] = -243.518329493424
        struct[0].Gy_ini[33,16] = -116.655487182478
        struct[0].Gy_ini[33,17] = 243.518329493424
        struct[0].Gy_ini[33,33] = -1
        struct[0].Gy_ini[34,2] = -243.518329493424
        struct[0].Gy_ini[34,3] = -116.655487182478
        struct[0].Gy_ini[34,18] = 243.518329493424
        struct[0].Gy_ini[34,19] = 116.655487182478
        struct[0].Gy_ini[34,34] = -1
        struct[0].Gy_ini[35,2] = 116.655487182478
        struct[0].Gy_ini[35,3] = -243.518329493424
        struct[0].Gy_ini[35,18] = -116.655487182478
        struct[0].Gy_ini[35,19] = 243.518329493424
        struct[0].Gy_ini[35,35] = -1
        struct[0].Gy_ini[36,4] = -243.518329493424
        struct[0].Gy_ini[36,5] = -116.655487182478
        struct[0].Gy_ini[36,20] = 243.518329493424
        struct[0].Gy_ini[36,21] = 116.655487182478
        struct[0].Gy_ini[36,36] = -1
        struct[0].Gy_ini[37,4] = 116.655487182478
        struct[0].Gy_ini[37,5] = -243.518329493424
        struct[0].Gy_ini[37,20] = -116.655487182478
        struct[0].Gy_ini[37,21] = 243.518329493424
        struct[0].Gy_ini[37,37] = -1
        struct[0].Gy_ini[38,32] = 1
        struct[0].Gy_ini[38,34] = 1
        struct[0].Gy_ini[38,36] = 1
        struct[0].Gy_ini[38,38] = -1
        struct[0].Gy_ini[39,33] = 1
        struct[0].Gy_ini[39,35] = 1
        struct[0].Gy_ini[39,37] = 1
        struct[0].Gy_ini[39,39] = -1
        struct[0].Gy_ini[40,0] = 48.7036658986847
        struct[0].Gy_ini[40,1] = 23.3310974364957
        struct[0].Gy_ini[40,8] = -48.7036658986847
        struct[0].Gy_ini[40,9] = -23.3310974364957
        struct[0].Gy_ini[40,40] = -1
        struct[0].Gy_ini[41,0] = -23.3310974364957
        struct[0].Gy_ini[41,1] = 48.7036658986847
        struct[0].Gy_ini[41,8] = 23.3310974364957
        struct[0].Gy_ini[41,9] = -48.7036658986847
        struct[0].Gy_ini[41,41] = -1
        struct[0].Gy_ini[42,2] = 48.7036658986847
        struct[0].Gy_ini[42,3] = 23.3310974364957
        struct[0].Gy_ini[42,10] = -48.7036658986847
        struct[0].Gy_ini[42,11] = -23.3310974364957
        struct[0].Gy_ini[42,42] = -1
        struct[0].Gy_ini[43,2] = -23.3310974364957
        struct[0].Gy_ini[43,3] = 48.7036658986847
        struct[0].Gy_ini[43,10] = 23.3310974364957
        struct[0].Gy_ini[43,11] = -48.7036658986847
        struct[0].Gy_ini[43,43] = -1
        struct[0].Gy_ini[44,4] = 48.7036658986847
        struct[0].Gy_ini[44,5] = 23.3310974364957
        struct[0].Gy_ini[44,12] = -48.7036658986847
        struct[0].Gy_ini[44,13] = -23.3310974364957
        struct[0].Gy_ini[44,44] = -1
        struct[0].Gy_ini[45,4] = -23.3310974364957
        struct[0].Gy_ini[45,5] = 48.7036658986847
        struct[0].Gy_ini[45,12] = 23.3310974364957
        struct[0].Gy_ini[45,13] = -48.7036658986847
        struct[0].Gy_ini[45,45] = -1
        struct[0].Gy_ini[46,40] = 1
        struct[0].Gy_ini[46,42] = 1
        struct[0].Gy_ini[46,44] = 1
        struct[0].Gy_ini[46,46] = -1
        struct[0].Gy_ini[47,41] = 1
        struct[0].Gy_ini[47,43] = 1
        struct[0].Gy_ini[47,45] = 1
        struct[0].Gy_ini[47,47] = -1
        struct[0].Gy_ini[48,8] = 243.518329493424
        struct[0].Gy_ini[48,9] = 116.655487182478
        struct[0].Gy_ini[48,24] = -243.518329493424
        struct[0].Gy_ini[48,25] = -116.655487182478
        struct[0].Gy_ini[48,48] = -1
        struct[0].Gy_ini[49,8] = -116.655487182478
        struct[0].Gy_ini[49,9] = 243.518329493424
        struct[0].Gy_ini[49,24] = 116.655487182478
        struct[0].Gy_ini[49,25] = -243.518329493424
        struct[0].Gy_ini[49,49] = -1
        struct[0].Gy_ini[50,10] = 243.518329493424
        struct[0].Gy_ini[50,11] = 116.655487182478
        struct[0].Gy_ini[50,26] = -243.518329493424
        struct[0].Gy_ini[50,27] = -116.655487182478
        struct[0].Gy_ini[50,50] = -1
        struct[0].Gy_ini[51,10] = -116.655487182478
        struct[0].Gy_ini[51,11] = 243.518329493424
        struct[0].Gy_ini[51,26] = 116.655487182478
        struct[0].Gy_ini[51,27] = -243.518329493424
        struct[0].Gy_ini[51,51] = -1
        struct[0].Gy_ini[52,12] = 243.518329493424
        struct[0].Gy_ini[52,13] = 116.655487182478
        struct[0].Gy_ini[52,28] = -243.518329493424
        struct[0].Gy_ini[52,29] = -116.655487182478
        struct[0].Gy_ini[52,52] = -1
        struct[0].Gy_ini[53,12] = -116.655487182478
        struct[0].Gy_ini[53,13] = 243.518329493424
        struct[0].Gy_ini[53,28] = 116.655487182478
        struct[0].Gy_ini[53,29] = -243.518329493424
        struct[0].Gy_ini[53,53] = -1
        struct[0].Gy_ini[54,48] = 1
        struct[0].Gy_ini[54,50] = 1
        struct[0].Gy_ini[54,52] = 1
        struct[0].Gy_ini[54,54] = -1
        struct[0].Gy_ini[55,49] = 1
        struct[0].Gy_ini[55,51] = 1
        struct[0].Gy_ini[55,53] = 1
        struct[0].Gy_ini[55,55] = -1
        struct[0].Gy_ini[56,0] = i_load_B2_a_r
        struct[0].Gy_ini[56,1] = i_load_B2_a_i
        struct[0].Gy_ini[56,6] = -i_load_B2_a_r
        struct[0].Gy_ini[56,7] = -i_load_B2_a_i
        struct[0].Gy_ini[56,56] = v_B2_a_r - v_B2_n_r
        struct[0].Gy_ini[56,57] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[57,2] = i_load_B2_b_r
        struct[0].Gy_ini[57,3] = i_load_B2_b_i
        struct[0].Gy_ini[57,6] = -i_load_B2_b_r
        struct[0].Gy_ini[57,7] = -i_load_B2_b_i
        struct[0].Gy_ini[57,58] = v_B2_b_r - v_B2_n_r
        struct[0].Gy_ini[57,59] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[58,4] = i_load_B2_c_r
        struct[0].Gy_ini[58,5] = i_load_B2_c_i
        struct[0].Gy_ini[58,6] = -i_load_B2_c_r
        struct[0].Gy_ini[58,7] = -i_load_B2_c_i
        struct[0].Gy_ini[58,60] = v_B2_c_r - v_B2_n_r
        struct[0].Gy_ini[58,61] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[59,0] = -i_load_B2_a_i
        struct[0].Gy_ini[59,1] = i_load_B2_a_r
        struct[0].Gy_ini[59,6] = i_load_B2_a_i
        struct[0].Gy_ini[59,7] = -i_load_B2_a_r
        struct[0].Gy_ini[59,56] = v_B2_a_i - v_B2_n_i
        struct[0].Gy_ini[59,57] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy_ini[60,2] = -i_load_B2_b_i
        struct[0].Gy_ini[60,3] = i_load_B2_b_r
        struct[0].Gy_ini[60,6] = i_load_B2_b_i
        struct[0].Gy_ini[60,7] = -i_load_B2_b_r
        struct[0].Gy_ini[60,58] = v_B2_b_i - v_B2_n_i
        struct[0].Gy_ini[60,59] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy_ini[61,4] = -i_load_B2_c_i
        struct[0].Gy_ini[61,5] = i_load_B2_c_r
        struct[0].Gy_ini[61,6] = i_load_B2_c_i
        struct[0].Gy_ini[61,7] = -i_load_B2_c_r
        struct[0].Gy_ini[61,60] = v_B2_c_i - v_B2_n_i
        struct[0].Gy_ini[61,61] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy_ini[62,56] = 1
        struct[0].Gy_ini[62,58] = 1
        struct[0].Gy_ini[62,60] = 1
        struct[0].Gy_ini[62,62] = 1
        struct[0].Gy_ini[63,57] = 1
        struct[0].Gy_ini[63,59] = 1
        struct[0].Gy_ini[63,61] = 1
        struct[0].Gy_ini[63,63] = 1
        struct[0].Gy_ini[64,8] = i_load_B3_a_r
        struct[0].Gy_ini[64,9] = i_load_B3_a_i
        struct[0].Gy_ini[64,14] = -i_load_B3_a_r
        struct[0].Gy_ini[64,15] = -i_load_B3_a_i
        struct[0].Gy_ini[64,64] = v_B3_a_r - v_B3_n_r
        struct[0].Gy_ini[64,65] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[65,10] = i_load_B3_b_r
        struct[0].Gy_ini[65,11] = i_load_B3_b_i
        struct[0].Gy_ini[65,14] = -i_load_B3_b_r
        struct[0].Gy_ini[65,15] = -i_load_B3_b_i
        struct[0].Gy_ini[65,66] = v_B3_b_r - v_B3_n_r
        struct[0].Gy_ini[65,67] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[66,12] = i_load_B3_c_r
        struct[0].Gy_ini[66,13] = i_load_B3_c_i
        struct[0].Gy_ini[66,14] = -i_load_B3_c_r
        struct[0].Gy_ini[66,15] = -i_load_B3_c_i
        struct[0].Gy_ini[66,68] = v_B3_c_r - v_B3_n_r
        struct[0].Gy_ini[66,69] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[67,8] = -i_load_B3_a_i
        struct[0].Gy_ini[67,9] = i_load_B3_a_r
        struct[0].Gy_ini[67,14] = i_load_B3_a_i
        struct[0].Gy_ini[67,15] = -i_load_B3_a_r
        struct[0].Gy_ini[67,64] = v_B3_a_i - v_B3_n_i
        struct[0].Gy_ini[67,65] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy_ini[68,10] = -i_load_B3_b_i
        struct[0].Gy_ini[68,11] = i_load_B3_b_r
        struct[0].Gy_ini[68,14] = i_load_B3_b_i
        struct[0].Gy_ini[68,15] = -i_load_B3_b_r
        struct[0].Gy_ini[68,66] = v_B3_b_i - v_B3_n_i
        struct[0].Gy_ini[68,67] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy_ini[69,12] = -i_load_B3_c_i
        struct[0].Gy_ini[69,13] = i_load_B3_c_r
        struct[0].Gy_ini[69,14] = i_load_B3_c_i
        struct[0].Gy_ini[69,15] = -i_load_B3_c_r
        struct[0].Gy_ini[69,68] = v_B3_c_i - v_B3_n_i
        struct[0].Gy_ini[69,69] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy_ini[70,64] = 1
        struct[0].Gy_ini[70,66] = 1
        struct[0].Gy_ini[70,68] = 1
        struct[0].Gy_ini[70,70] = 1
        struct[0].Gy_ini[71,65] = 1
        struct[0].Gy_ini[71,67] = 1
        struct[0].Gy_ini[71,69] = 1
        struct[0].Gy_ini[71,71] = 1
        struct[0].Gy_ini[72,16] = -1
        struct[0].Gy_ini[72,22] = 1
        struct[0].Gy_ini[72,72] = -R_B1_sa
        struct[0].Gy_ini[72,78] = 1.0*X_B1_sa
        struct[0].Gy_ini[73,18] = -1
        struct[0].Gy_ini[73,22] = 1
        struct[0].Gy_ini[73,73] = -R_B1_sb
        struct[0].Gy_ini[73,79] = 1.0*X_B1_sb
        struct[0].Gy_ini[74,20] = -1
        struct[0].Gy_ini[74,22] = 1
        struct[0].Gy_ini[74,74] = -R_B1_sc
        struct[0].Gy_ini[74,80] = 1.0*X_B1_sc
        struct[0].Gy_ini[75,22] = -1
        struct[0].Gy_ini[75,75] = -R_B1_sn
        struct[0].Gy_ini[75,77] = 1
        struct[0].Gy_ini[75,81] = 1.0*X_B1_sn
        struct[0].Gy_ini[76,72] = 1
        struct[0].Gy_ini[76,73] = 1
        struct[0].Gy_ini[76,74] = 1
        struct[0].Gy_ini[76,75] = 1
        struct[0].Gy_ini[76,76] = -1
        struct[0].Gy_ini[77,76] = R_B1_ng
        struct[0].Gy_ini[77,77] = -1
        struct[0].Gy_ini[77,82] = -1.0*X_B1_ng
        struct[0].Gy_ini[78,17] = -1.00000000000000
        struct[0].Gy_ini[78,23] = 1.00000000000000
        struct[0].Gy_ini[78,72] = -1.0*X_B1_sa
        struct[0].Gy_ini[78,78] = -1.0*R_B1_sa
        struct[0].Gy_ini[79,19] = -1.00000000000000
        struct[0].Gy_ini[79,23] = 1.00000000000000
        struct[0].Gy_ini[79,73] = -1.0*X_B1_sb
        struct[0].Gy_ini[79,79] = -1.0*R_B1_sb
        struct[0].Gy_ini[80,21] = -1.00000000000000
        struct[0].Gy_ini[80,23] = 1.00000000000000
        struct[0].Gy_ini[80,74] = -1.0*X_B1_sc
        struct[0].Gy_ini[80,80] = -1.0*R_B1_sc
        struct[0].Gy_ini[81,23] = -1.00000000000000
        struct[0].Gy_ini[81,75] = -1.0*X_B1_sn
        struct[0].Gy_ini[81,81] = -1.0*R_B1_sn
        struct[0].Gy_ini[81,83] = 1.00000000000000
        struct[0].Gy_ini[82,78] = 1.00000000000000
        struct[0].Gy_ini[82,79] = 1.00000000000000
        struct[0].Gy_ini[82,80] = 1.00000000000000
        struct[0].Gy_ini[82,81] = 1.00000000000000
        struct[0].Gy_ini[82,82] = -1.00000000000000
        struct[0].Gy_ini[83,76] = 1.0*X_B1_ng
        struct[0].Gy_ini[83,82] = 1.0*R_B1_ng
        struct[0].Gy_ini[83,83] = -1.00000000000000
        struct[0].Gy_ini[84,24] = -1
        struct[0].Gy_ini[84,30] = 1
        struct[0].Gy_ini[84,84] = -R_B4_sa
        struct[0].Gy_ini[84,90] = 1.0*X_B4_sa
        struct[0].Gy_ini[85,26] = -1
        struct[0].Gy_ini[85,30] = 1
        struct[0].Gy_ini[85,85] = -R_B4_sb
        struct[0].Gy_ini[85,91] = 1.0*X_B4_sb
        struct[0].Gy_ini[86,28] = -1
        struct[0].Gy_ini[86,30] = 1
        struct[0].Gy_ini[86,86] = -R_B4_sc
        struct[0].Gy_ini[86,92] = 1.0*X_B4_sc
        struct[0].Gy_ini[87,30] = -1
        struct[0].Gy_ini[87,87] = -R_B4_sn
        struct[0].Gy_ini[87,89] = 1
        struct[0].Gy_ini[87,93] = 1.0*X_B4_sn
        struct[0].Gy_ini[88,84] = 1
        struct[0].Gy_ini[88,85] = 1
        struct[0].Gy_ini[88,86] = 1
        struct[0].Gy_ini[88,87] = 1
        struct[0].Gy_ini[88,88] = -1
        struct[0].Gy_ini[89,88] = R_B4_ng
        struct[0].Gy_ini[89,89] = -1
        struct[0].Gy_ini[89,94] = -1.0*X_B4_ng
        struct[0].Gy_ini[90,25] = -1.00000000000000
        struct[0].Gy_ini[90,31] = 1.00000000000000
        struct[0].Gy_ini[90,84] = -1.0*X_B4_sa
        struct[0].Gy_ini[90,90] = -1.0*R_B4_sa
        struct[0].Gy_ini[91,27] = -1.00000000000000
        struct[0].Gy_ini[91,31] = 1.00000000000000
        struct[0].Gy_ini[91,85] = -1.0*X_B4_sb
        struct[0].Gy_ini[91,91] = -1.0*R_B4_sb
        struct[0].Gy_ini[92,29] = -1.00000000000000
        struct[0].Gy_ini[92,31] = 1.00000000000000
        struct[0].Gy_ini[92,86] = -1.0*X_B4_sc
        struct[0].Gy_ini[92,92] = -1.0*R_B4_sc
        struct[0].Gy_ini[93,31] = -1.00000000000000
        struct[0].Gy_ini[93,87] = -1.0*X_B4_sn
        struct[0].Gy_ini[93,93] = -1.0*R_B4_sn
        struct[0].Gy_ini[93,95] = 1.00000000000000
        struct[0].Gy_ini[94,90] = 1.00000000000000
        struct[0].Gy_ini[94,91] = 1.00000000000000
        struct[0].Gy_ini[94,92] = 1.00000000000000
        struct[0].Gy_ini[94,93] = 1.00000000000000
        struct[0].Gy_ini[94,94] = -1.00000000000000
        struct[0].Gy_ini[95,88] = 1.0*X_B4_ng
        struct[0].Gy_ini[95,94] = 1.0*R_B4_ng
        struct[0].Gy_ini[95,95] = -1.00000000000000
        struct[0].Gy_ini[96,96] = 1



def run_nn(t,struct,mode):

    # Parameters:
    X_B1_sa = struct[0].X_B1_sa
    R_B1_sa = struct[0].R_B1_sa
    X_B1_sb = struct[0].X_B1_sb
    R_B1_sb = struct[0].R_B1_sb
    X_B1_sc = struct[0].X_B1_sc
    R_B1_sc = struct[0].R_B1_sc
    X_B1_sn = struct[0].X_B1_sn
    R_B1_sn = struct[0].R_B1_sn
    S_n_B1 = struct[0].S_n_B1
    X_B1_ng = struct[0].X_B1_ng
    R_B1_ng = struct[0].R_B1_ng
    K_f_B1 = struct[0].K_f_B1
    T_f_B1 = struct[0].T_f_B1
    K_sec_B1 = struct[0].K_sec_B1
    K_delta_B1 = struct[0].K_delta_B1
    X_B4_sa = struct[0].X_B4_sa
    R_B4_sa = struct[0].R_B4_sa
    X_B4_sb = struct[0].X_B4_sb
    R_B4_sb = struct[0].R_B4_sb
    X_B4_sc = struct[0].X_B4_sc
    R_B4_sc = struct[0].R_B4_sc
    X_B4_sn = struct[0].X_B4_sn
    R_B4_sn = struct[0].R_B4_sn
    S_n_B4 = struct[0].S_n_B4
    X_B4_ng = struct[0].X_B4_ng
    R_B4_ng = struct[0].R_B4_ng
    K_f_B4 = struct[0].K_f_B4
    T_f_B4 = struct[0].T_f_B4
    K_sec_B4 = struct[0].K_sec_B4
    K_delta_B4 = struct[0].K_delta_B4
    K_agc = struct[0].K_agc
    
    # Inputs:
    i_B2_n_r = struct[0].i_B2_n_r
    i_B2_n_i = struct[0].i_B2_n_i
    i_B3_n_r = struct[0].i_B3_n_r
    i_B3_n_i = struct[0].i_B3_n_i
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
    e_B1_an = struct[0].e_B1_an
    e_B1_bn = struct[0].e_B1_bn
    e_B1_cn = struct[0].e_B1_cn
    phi_B1 = struct[0].phi_B1
    p_B1_ref = struct[0].p_B1_ref
    omega_B1_ref = struct[0].omega_B1_ref
    e_B4_an = struct[0].e_B4_an
    e_B4_bn = struct[0].e_B4_bn
    e_B4_cn = struct[0].e_B4_cn
    phi_B4 = struct[0].phi_B4
    p_B4_ref = struct[0].p_B4_ref
    omega_B4_ref = struct[0].omega_B4_ref
    
    # Dynamical states:
    phi_B1 = struct[0].x[0,0]
    omega_B1 = struct[0].x[1,0]
    phi_B4 = struct[0].x[2,0]
    omega_B4 = struct[0].x[3,0]
    xi_freq = struct[0].x[4,0]
    
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
    v_B1_a_r = struct[0].y_run[16,0]
    v_B1_a_i = struct[0].y_run[17,0]
    v_B1_b_r = struct[0].y_run[18,0]
    v_B1_b_i = struct[0].y_run[19,0]
    v_B1_c_r = struct[0].y_run[20,0]
    v_B1_c_i = struct[0].y_run[21,0]
    v_B1_n_r = struct[0].y_run[22,0]
    v_B1_n_i = struct[0].y_run[23,0]
    v_B4_a_r = struct[0].y_run[24,0]
    v_B4_a_i = struct[0].y_run[25,0]
    v_B4_b_r = struct[0].y_run[26,0]
    v_B4_b_i = struct[0].y_run[27,0]
    v_B4_c_r = struct[0].y_run[28,0]
    v_B4_c_i = struct[0].y_run[29,0]
    v_B4_n_r = struct[0].y_run[30,0]
    v_B4_n_i = struct[0].y_run[31,0]
    i_l_B1_B2_a_r = struct[0].y_run[32,0]
    i_l_B1_B2_a_i = struct[0].y_run[33,0]
    i_l_B1_B2_b_r = struct[0].y_run[34,0]
    i_l_B1_B2_b_i = struct[0].y_run[35,0]
    i_l_B1_B2_c_r = struct[0].y_run[36,0]
    i_l_B1_B2_c_i = struct[0].y_run[37,0]
    i_l_B1_B2_n_r = struct[0].y_run[38,0]
    i_l_B1_B2_n_i = struct[0].y_run[39,0]
    i_l_B2_B3_a_r = struct[0].y_run[40,0]
    i_l_B2_B3_a_i = struct[0].y_run[41,0]
    i_l_B2_B3_b_r = struct[0].y_run[42,0]
    i_l_B2_B3_b_i = struct[0].y_run[43,0]
    i_l_B2_B3_c_r = struct[0].y_run[44,0]
    i_l_B2_B3_c_i = struct[0].y_run[45,0]
    i_l_B2_B3_n_r = struct[0].y_run[46,0]
    i_l_B2_B3_n_i = struct[0].y_run[47,0]
    i_l_B3_B4_a_r = struct[0].y_run[48,0]
    i_l_B3_B4_a_i = struct[0].y_run[49,0]
    i_l_B3_B4_b_r = struct[0].y_run[50,0]
    i_l_B3_B4_b_i = struct[0].y_run[51,0]
    i_l_B3_B4_c_r = struct[0].y_run[52,0]
    i_l_B3_B4_c_i = struct[0].y_run[53,0]
    i_l_B3_B4_n_r = struct[0].y_run[54,0]
    i_l_B3_B4_n_i = struct[0].y_run[55,0]
    i_load_B2_a_r = struct[0].y_run[56,0]
    i_load_B2_a_i = struct[0].y_run[57,0]
    i_load_B2_b_r = struct[0].y_run[58,0]
    i_load_B2_b_i = struct[0].y_run[59,0]
    i_load_B2_c_r = struct[0].y_run[60,0]
    i_load_B2_c_i = struct[0].y_run[61,0]
    i_load_B2_n_r = struct[0].y_run[62,0]
    i_load_B2_n_i = struct[0].y_run[63,0]
    i_load_B3_a_r = struct[0].y_run[64,0]
    i_load_B3_a_i = struct[0].y_run[65,0]
    i_load_B3_b_r = struct[0].y_run[66,0]
    i_load_B3_b_i = struct[0].y_run[67,0]
    i_load_B3_c_r = struct[0].y_run[68,0]
    i_load_B3_c_i = struct[0].y_run[69,0]
    i_load_B3_n_r = struct[0].y_run[70,0]
    i_load_B3_n_i = struct[0].y_run[71,0]
    i_B1_a_r = struct[0].y_run[72,0]
    i_B1_b_r = struct[0].y_run[73,0]
    i_B1_c_r = struct[0].y_run[74,0]
    i_B1_n_r = struct[0].y_run[75,0]
    i_B1_ng_r = struct[0].y_run[76,0]
    e_B1_ng_r = struct[0].y_run[77,0]
    i_B1_a_i = struct[0].y_run[78,0]
    i_B1_b_i = struct[0].y_run[79,0]
    i_B1_c_i = struct[0].y_run[80,0]
    i_B1_n_i = struct[0].y_run[81,0]
    i_B1_ng_i = struct[0].y_run[82,0]
    e_B1_ng_i = struct[0].y_run[83,0]
    i_B4_a_r = struct[0].y_run[84,0]
    i_B4_b_r = struct[0].y_run[85,0]
    i_B4_c_r = struct[0].y_run[86,0]
    i_B4_n_r = struct[0].y_run[87,0]
    i_B4_ng_r = struct[0].y_run[88,0]
    e_B4_ng_r = struct[0].y_run[89,0]
    i_B4_a_i = struct[0].y_run[90,0]
    i_B4_b_i = struct[0].y_run[91,0]
    i_B4_c_i = struct[0].y_run[92,0]
    i_B4_n_i = struct[0].y_run[93,0]
    i_B4_ng_i = struct[0].y_run[94,0]
    e_B4_ng_i = struct[0].y_run[95,0]
    omega_coi = struct[0].y_run[96,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -K_delta_B1*phi_B1 + 314.159265358979*omega_B1 - 314.159265358979*omega_coi
        struct[0].f[1,0] = (-K_f_B1*(K_sec_B1*xi_freq - 0.333333333333333*i_B1_a_i*v_B1_a_i - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) - 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) - 0.333333333333333*i_B1_a_r*v_B1_a_r - 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) - 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) - 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r - 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i + 0.166666666666667*i_B1_b_r*v_B1_c_r + 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r - 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i + 0.166666666666667*i_B1_c_r*v_B1_b_r - 0.333333333333333*i_B1_c_r*v_B1_c_r + p_B1_ref + v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) + v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) - v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) - v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r))/S_n_B1 - omega_B1 + omega_B1_ref)/T_f_B1
        struct[0].f[2,0] = -K_delta_B4*phi_B4 + 314.159265358979*omega_B4 - 314.159265358979*omega_coi
        struct[0].f[3,0] = (-K_f_B4*(K_sec_B4*xi_freq - 0.333333333333333*i_B4_a_i*v_B4_a_i - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) - 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) - 0.333333333333333*i_B4_a_r*v_B4_a_r - 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) - 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) - 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r - 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i + 0.166666666666667*i_B4_b_r*v_B4_c_r + 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r - 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i + 0.166666666666667*i_B4_c_r*v_B4_b_r - 0.333333333333333*i_B4_c_r*v_B4_c_r + p_B4_ref + v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) + v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) - v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) - v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r))/S_n_B4 - omega_B4 + omega_B4_ref)/T_f_B4
        struct[0].f[4,0] = K_agc*(1 - omega_coi)
    
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
        struct[0].g[16,0] = i_B1_a_r - 116.655487182478*v_B1_a_i - 243.518329493424*v_B1_a_r + 116.655487182478*v_B2_a_i + 243.518329493424*v_B2_a_r
        struct[0].g[17,0] = i_B1_a_i - 243.518329493424*v_B1_a_i + 116.655487182478*v_B1_a_r + 243.518329493424*v_B2_a_i - 116.655487182478*v_B2_a_r
        struct[0].g[18,0] = i_B1_b_r - 116.655487182478*v_B1_b_i - 243.518329493424*v_B1_b_r + 116.655487182478*v_B2_b_i + 243.518329493424*v_B2_b_r
        struct[0].g[19,0] = i_B1_b_i - 243.518329493424*v_B1_b_i + 116.655487182478*v_B1_b_r + 243.518329493424*v_B2_b_i - 116.655487182478*v_B2_b_r
        struct[0].g[20,0] = i_B1_c_r - 116.655487182478*v_B1_c_i - 243.518329493424*v_B1_c_r + 116.655487182478*v_B2_c_i + 243.518329493424*v_B2_c_r
        struct[0].g[21,0] = i_B1_c_i - 243.518329493424*v_B1_c_i + 116.655487182478*v_B1_c_r + 243.518329493424*v_B2_c_i - 116.655487182478*v_B2_c_r
        struct[0].g[22,0] = i_B1_n_r - 116.655487182478*v_B1_n_i - 243.518329493424*v_B1_n_r + 116.655487182478*v_B2_n_i + 243.518329493424*v_B2_n_r
        struct[0].g[23,0] = i_B1_n_i - 243.518329493424*v_B1_n_i + 116.655487182478*v_B1_n_r + 243.518329493424*v_B2_n_i - 116.655487182478*v_B2_n_r
        struct[0].g[24,0] = i_B4_a_r + 116.655487182478*v_B3_a_i + 243.518329493424*v_B3_a_r - 116.655487182478*v_B4_a_i - 243.518329493424*v_B4_a_r
        struct[0].g[25,0] = i_B4_a_i + 243.518329493424*v_B3_a_i - 116.655487182478*v_B3_a_r - 243.518329493424*v_B4_a_i + 116.655487182478*v_B4_a_r
        struct[0].g[26,0] = i_B4_b_r + 116.655487182478*v_B3_b_i + 243.518329493424*v_B3_b_r - 116.655487182478*v_B4_b_i - 243.518329493424*v_B4_b_r
        struct[0].g[27,0] = i_B4_b_i + 243.518329493424*v_B3_b_i - 116.655487182478*v_B3_b_r - 243.518329493424*v_B4_b_i + 116.655487182478*v_B4_b_r
        struct[0].g[28,0] = i_B4_c_r + 116.655487182478*v_B3_c_i + 243.518329493424*v_B3_c_r - 116.655487182478*v_B4_c_i - 243.518329493424*v_B4_c_r
        struct[0].g[29,0] = i_B4_c_i + 243.518329493424*v_B3_c_i - 116.655487182478*v_B3_c_r - 243.518329493424*v_B4_c_i + 116.655487182478*v_B4_c_r
        struct[0].g[30,0] = i_B4_n_r + 116.655487182478*v_B3_n_i + 243.518329493424*v_B3_n_r - 116.655487182478*v_B4_n_i - 243.518329493424*v_B4_n_r
        struct[0].g[31,0] = i_B4_n_i + 243.518329493424*v_B3_n_i - 116.655487182478*v_B3_n_r - 243.518329493424*v_B4_n_i + 116.655487182478*v_B4_n_r
        struct[0].g[32,0] = -i_l_B1_B2_a_r + 116.655487182478*v_B1_a_i + 243.518329493424*v_B1_a_r - 116.655487182478*v_B2_a_i - 243.518329493424*v_B2_a_r
        struct[0].g[33,0] = -i_l_B1_B2_a_i + 243.518329493424*v_B1_a_i - 116.655487182478*v_B1_a_r - 243.518329493424*v_B2_a_i + 116.655487182478*v_B2_a_r
        struct[0].g[34,0] = -i_l_B1_B2_b_r + 116.655487182478*v_B1_b_i + 243.518329493424*v_B1_b_r - 116.655487182478*v_B2_b_i - 243.518329493424*v_B2_b_r
        struct[0].g[35,0] = -i_l_B1_B2_b_i + 243.518329493424*v_B1_b_i - 116.655487182478*v_B1_b_r - 243.518329493424*v_B2_b_i + 116.655487182478*v_B2_b_r
        struct[0].g[36,0] = -i_l_B1_B2_c_r + 116.655487182478*v_B1_c_i + 243.518329493424*v_B1_c_r - 116.655487182478*v_B2_c_i - 243.518329493424*v_B2_c_r
        struct[0].g[37,0] = -i_l_B1_B2_c_i + 243.518329493424*v_B1_c_i - 116.655487182478*v_B1_c_r - 243.518329493424*v_B2_c_i + 116.655487182478*v_B2_c_r
        struct[0].g[38,0] = i_l_B1_B2_a_r + i_l_B1_B2_b_r + i_l_B1_B2_c_r - i_l_B1_B2_n_r
        struct[0].g[39,0] = i_l_B1_B2_a_i + i_l_B1_B2_b_i + i_l_B1_B2_c_i - i_l_B1_B2_n_i
        struct[0].g[40,0] = -i_l_B2_B3_a_r + 23.3310974364957*v_B2_a_i + 48.7036658986847*v_B2_a_r - 23.3310974364957*v_B3_a_i - 48.7036658986847*v_B3_a_r
        struct[0].g[41,0] = -i_l_B2_B3_a_i + 48.7036658986847*v_B2_a_i - 23.3310974364957*v_B2_a_r - 48.7036658986847*v_B3_a_i + 23.3310974364957*v_B3_a_r
        struct[0].g[42,0] = -i_l_B2_B3_b_r + 23.3310974364957*v_B2_b_i + 48.7036658986847*v_B2_b_r - 23.3310974364957*v_B3_b_i - 48.7036658986847*v_B3_b_r
        struct[0].g[43,0] = -i_l_B2_B3_b_i + 48.7036658986847*v_B2_b_i - 23.3310974364957*v_B2_b_r - 48.7036658986847*v_B3_b_i + 23.3310974364957*v_B3_b_r
        struct[0].g[44,0] = -i_l_B2_B3_c_r + 23.3310974364957*v_B2_c_i + 48.7036658986847*v_B2_c_r - 23.3310974364957*v_B3_c_i - 48.7036658986847*v_B3_c_r
        struct[0].g[45,0] = -i_l_B2_B3_c_i + 48.7036658986847*v_B2_c_i - 23.3310974364957*v_B2_c_r - 48.7036658986847*v_B3_c_i + 23.3310974364957*v_B3_c_r
        struct[0].g[46,0] = i_l_B2_B3_a_r + i_l_B2_B3_b_r + i_l_B2_B3_c_r - i_l_B2_B3_n_r
        struct[0].g[47,0] = i_l_B2_B3_a_i + i_l_B2_B3_b_i + i_l_B2_B3_c_i - i_l_B2_B3_n_i
        struct[0].g[48,0] = -i_l_B3_B4_a_r + 116.655487182478*v_B3_a_i + 243.518329493424*v_B3_a_r - 116.655487182478*v_B4_a_i - 243.518329493424*v_B4_a_r
        struct[0].g[49,0] = -i_l_B3_B4_a_i + 243.518329493424*v_B3_a_i - 116.655487182478*v_B3_a_r - 243.518329493424*v_B4_a_i + 116.655487182478*v_B4_a_r
        struct[0].g[50,0] = -i_l_B3_B4_b_r + 116.655487182478*v_B3_b_i + 243.518329493424*v_B3_b_r - 116.655487182478*v_B4_b_i - 243.518329493424*v_B4_b_r
        struct[0].g[51,0] = -i_l_B3_B4_b_i + 243.518329493424*v_B3_b_i - 116.655487182478*v_B3_b_r - 243.518329493424*v_B4_b_i + 116.655487182478*v_B4_b_r
        struct[0].g[52,0] = -i_l_B3_B4_c_r + 116.655487182478*v_B3_c_i + 243.518329493424*v_B3_c_r - 116.655487182478*v_B4_c_i - 243.518329493424*v_B4_c_r
        struct[0].g[53,0] = -i_l_B3_B4_c_i + 243.518329493424*v_B3_c_i - 116.655487182478*v_B3_c_r - 243.518329493424*v_B4_c_i + 116.655487182478*v_B4_c_r
        struct[0].g[54,0] = i_l_B3_B4_a_r + i_l_B3_B4_b_r + i_l_B3_B4_c_r - i_l_B3_B4_n_r
        struct[0].g[55,0] = i_l_B3_B4_a_i + i_l_B3_B4_b_i + i_l_B3_B4_c_i - i_l_B3_B4_n_i
        struct[0].g[56,0] = i_load_B2_a_i*v_B2_a_i - i_load_B2_a_i*v_B2_n_i + i_load_B2_a_r*v_B2_a_r - i_load_B2_a_r*v_B2_n_r - p_B2_a
        struct[0].g[57,0] = i_load_B2_b_i*v_B2_b_i - i_load_B2_b_i*v_B2_n_i + i_load_B2_b_r*v_B2_b_r - i_load_B2_b_r*v_B2_n_r - p_B2_b
        struct[0].g[58,0] = i_load_B2_c_i*v_B2_c_i - i_load_B2_c_i*v_B2_n_i + i_load_B2_c_r*v_B2_c_r - i_load_B2_c_r*v_B2_n_r - p_B2_c
        struct[0].g[59,0] = -i_load_B2_a_i*v_B2_a_r + i_load_B2_a_i*v_B2_n_r + i_load_B2_a_r*v_B2_a_i - i_load_B2_a_r*v_B2_n_i - q_B2_a
        struct[0].g[60,0] = -i_load_B2_b_i*v_B2_b_r + i_load_B2_b_i*v_B2_n_r + i_load_B2_b_r*v_B2_b_i - i_load_B2_b_r*v_B2_n_i - q_B2_b
        struct[0].g[61,0] = -i_load_B2_c_i*v_B2_c_r + i_load_B2_c_i*v_B2_n_r + i_load_B2_c_r*v_B2_c_i - i_load_B2_c_r*v_B2_n_i - q_B2_c
        struct[0].g[62,0] = i_load_B2_a_r + i_load_B2_b_r + i_load_B2_c_r + i_load_B2_n_r
        struct[0].g[63,0] = i_load_B2_a_i + i_load_B2_b_i + i_load_B2_c_i + i_load_B2_n_i
        struct[0].g[64,0] = i_load_B3_a_i*v_B3_a_i - i_load_B3_a_i*v_B3_n_i + i_load_B3_a_r*v_B3_a_r - i_load_B3_a_r*v_B3_n_r - p_B3_a
        struct[0].g[65,0] = i_load_B3_b_i*v_B3_b_i - i_load_B3_b_i*v_B3_n_i + i_load_B3_b_r*v_B3_b_r - i_load_B3_b_r*v_B3_n_r - p_B3_b
        struct[0].g[66,0] = i_load_B3_c_i*v_B3_c_i - i_load_B3_c_i*v_B3_n_i + i_load_B3_c_r*v_B3_c_r - i_load_B3_c_r*v_B3_n_r - p_B3_c
        struct[0].g[67,0] = -i_load_B3_a_i*v_B3_a_r + i_load_B3_a_i*v_B3_n_r + i_load_B3_a_r*v_B3_a_i - i_load_B3_a_r*v_B3_n_i - q_B3_a
        struct[0].g[68,0] = -i_load_B3_b_i*v_B3_b_r + i_load_B3_b_i*v_B3_n_r + i_load_B3_b_r*v_B3_b_i - i_load_B3_b_r*v_B3_n_i - q_B3_b
        struct[0].g[69,0] = -i_load_B3_c_i*v_B3_c_r + i_load_B3_c_i*v_B3_n_r + i_load_B3_c_r*v_B3_c_i - i_load_B3_c_r*v_B3_n_i - q_B3_c
        struct[0].g[70,0] = i_load_B3_a_r + i_load_B3_b_r + i_load_B3_c_r + i_load_B3_n_r
        struct[0].g[71,0] = i_load_B3_a_i + i_load_B3_b_i + i_load_B3_c_i + i_load_B3_n_i
        struct[0].g[72,0] = -R_B1_sa*i_B1_a_r + 1.0*X_B1_sa*i_B1_a_i + e_B1_an*cos(phi_B1) - v_B1_a_r + v_B1_n_r
        struct[0].g[73,0] = -R_B1_sb*i_B1_b_r + 1.0*X_B1_sb*i_B1_b_i + e_B1_bn*cos(phi_B1 - 2.0943951023932) - v_B1_b_r + v_B1_n_r
        struct[0].g[74,0] = -R_B1_sc*i_B1_c_r + 1.0*X_B1_sc*i_B1_c_i + e_B1_cn*cos(phi_B1 - 4.18879020478639) - v_B1_c_r + v_B1_n_r
        struct[0].g[75,0] = -R_B1_sn*i_B1_n_r + 1.0*X_B1_sn*i_B1_n_i + e_B1_ng_r - v_B1_n_r
        struct[0].g[76,0] = i_B1_a_r + i_B1_b_r + i_B1_c_r + i_B1_n_r - i_B1_ng_r
        struct[0].g[77,0] = R_B1_ng*i_B1_ng_r - 1.0*X_B1_ng*i_B1_ng_i - e_B1_ng_r
        struct[0].g[78,0] = -1.0*R_B1_sa*i_B1_a_i - 1.0*X_B1_sa*i_B1_a_r + 1.0*e_B1_an*sin(phi_B1) - 1.0*v_B1_a_i + 1.0*v_B1_n_i
        struct[0].g[79,0] = -1.0*R_B1_sb*i_B1_b_i - 1.0*X_B1_sb*i_B1_b_r + 1.0*e_B1_bn*sin(phi_B1 - 2.0943951023932) - 1.0*v_B1_b_i + 1.0*v_B1_n_i
        struct[0].g[80,0] = -1.0*R_B1_sc*i_B1_c_i - 1.0*X_B1_sc*i_B1_c_r + 1.0*e_B1_cn*sin(phi_B1 - 4.18879020478639) - 1.0*v_B1_c_i + 1.0*v_B1_n_i
        struct[0].g[81,0] = -1.0*R_B1_sn*i_B1_n_i - 1.0*X_B1_sn*i_B1_n_r + 1.0*e_B1_ng_i - 1.0*v_B1_n_i
        struct[0].g[82,0] = 1.0*i_B1_a_i + 1.0*i_B1_b_i + 1.0*i_B1_c_i + 1.0*i_B1_n_i - 1.0*i_B1_ng_i
        struct[0].g[83,0] = 1.0*R_B1_ng*i_B1_ng_i + 1.0*X_B1_ng*i_B1_ng_r - 1.0*e_B1_ng_i
        struct[0].g[84,0] = -R_B4_sa*i_B4_a_r + 1.0*X_B4_sa*i_B4_a_i + e_B4_an*cos(phi_B4) - v_B4_a_r + v_B4_n_r
        struct[0].g[85,0] = -R_B4_sb*i_B4_b_r + 1.0*X_B4_sb*i_B4_b_i + e_B4_bn*cos(phi_B4 - 2.0943951023932) - v_B4_b_r + v_B4_n_r
        struct[0].g[86,0] = -R_B4_sc*i_B4_c_r + 1.0*X_B4_sc*i_B4_c_i + e_B4_cn*cos(phi_B4 - 4.18879020478639) - v_B4_c_r + v_B4_n_r
        struct[0].g[87,0] = -R_B4_sn*i_B4_n_r + 1.0*X_B4_sn*i_B4_n_i + e_B4_ng_r - v_B4_n_r
        struct[0].g[88,0] = i_B4_a_r + i_B4_b_r + i_B4_c_r + i_B4_n_r - i_B4_ng_r
        struct[0].g[89,0] = R_B4_ng*i_B4_ng_r - 1.0*X_B4_ng*i_B4_ng_i - e_B4_ng_r
        struct[0].g[90,0] = -1.0*R_B4_sa*i_B4_a_i - 1.0*X_B4_sa*i_B4_a_r + 1.0*e_B4_an*sin(phi_B4) - 1.0*v_B4_a_i + 1.0*v_B4_n_i
        struct[0].g[91,0] = -1.0*R_B4_sb*i_B4_b_i - 1.0*X_B4_sb*i_B4_b_r + 1.0*e_B4_bn*sin(phi_B4 - 2.0943951023932) - 1.0*v_B4_b_i + 1.0*v_B4_n_i
        struct[0].g[92,0] = -1.0*R_B4_sc*i_B4_c_i - 1.0*X_B4_sc*i_B4_c_r + 1.0*e_B4_cn*sin(phi_B4 - 4.18879020478639) - 1.0*v_B4_c_i + 1.0*v_B4_n_i
        struct[0].g[93,0] = -1.0*R_B4_sn*i_B4_n_i - 1.0*X_B4_sn*i_B4_n_r + 1.0*e_B4_ng_i - 1.0*v_B4_n_i
        struct[0].g[94,0] = 1.0*i_B4_a_i + 1.0*i_B4_b_i + 1.0*i_B4_c_i + 1.0*i_B4_n_i - 1.0*i_B4_ng_i
        struct[0].g[95,0] = 1.0*R_B4_ng*i_B4_ng_i + 1.0*X_B4_ng*i_B4_ng_r - 1.0*e_B4_ng_i
        struct[0].g[96,0] = omega_coi - (S_n_B1*omega_B1 + S_n_B4*omega_B4)/(S_n_B1 + S_n_B4)
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[1,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[2,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[3,0] = (v_B2_n_i**2 + v_B2_n_r**2)**0.5
        struct[0].h[4,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[5,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[6,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[7,0] = (v_B3_n_i**2 + v_B3_n_r**2)**0.5
        struct[0].h[8,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[9,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[10,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[11,0] = (v_B1_n_i**2 + v_B1_n_r**2)**0.5
        struct[0].h[12,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[13,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[14,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[15,0] = (v_B4_n_i**2 + v_B4_n_r**2)**0.5
        struct[0].h[16,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i + 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r - 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i - 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r) + v_B1_a_r*(-0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[17,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r) + 1.0*i_B1_a_i*(-0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r) + 0.333333333333333*i_B1_a_r*v_B1_a_r + 1.0*i_B1_a_r*(0.288675134594813*v_B1_b_i - 0.166666666666667*v_B1_b_r) + 1.0*i_B1_a_r*(-0.288675134594813*v_B1_c_i - 0.166666666666667*v_B1_c_r) + 0.333333333333333*i_B1_b_i*v_B1_b_i - 0.166666666666667*i_B1_b_i*v_B1_c_i - 0.288675134594813*i_B1_b_i*v_B1_c_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.288675134594813*i_B1_b_r*v_B1_c_i - 0.166666666666667*i_B1_b_r*v_B1_c_r - 0.166666666666667*i_B1_c_i*v_B1_b_i + 0.288675134594813*i_B1_c_i*v_B1_b_r + 0.333333333333333*i_B1_c_i*v_B1_c_i - 0.288675134594813*i_B1_c_r*v_B1_b_i - 0.166666666666667*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r - v_B1_a_i*(0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r) - v_B1_a_i*(0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r) + v_B1_a_r*(0.288675134594813*i_B1_b_i - 0.166666666666667*i_B1_b_r) + v_B1_a_r*(-0.288675134594813*i_B1_c_i - 0.166666666666667*i_B1_c_r)
        struct[0].h[18,0] = 0.333333333333333*i_B1_a_i*v_B1_a_i + 0.333333333333333*i_B1_a_i*v_B1_b_i + 0.333333333333333*i_B1_a_i*v_B1_c_i + 0.333333333333333*i_B1_a_r*v_B1_a_r + 0.333333333333333*i_B1_a_r*v_B1_b_r + 0.333333333333333*i_B1_a_r*v_B1_c_r + 0.333333333333333*i_B1_b_i*v_B1_a_i + 0.333333333333333*i_B1_b_i*v_B1_b_i + 0.333333333333333*i_B1_b_i*v_B1_c_i + 0.333333333333333*i_B1_b_r*v_B1_a_r + 0.333333333333333*i_B1_b_r*v_B1_b_r + 0.333333333333333*i_B1_b_r*v_B1_c_r + 0.333333333333333*i_B1_c_i*v_B1_a_i + 0.333333333333333*i_B1_c_i*v_B1_b_i + 0.333333333333333*i_B1_c_i*v_B1_c_i + 0.333333333333333*i_B1_c_r*v_B1_a_r + 0.333333333333333*i_B1_c_r*v_B1_b_r + 0.333333333333333*i_B1_c_r*v_B1_c_r
        struct[0].h[19,0] = e_B1_an
        struct[0].h[20,0] = e_B1_bn
        struct[0].h[21,0] = e_B1_cn
        struct[0].h[22,0] = p_B1_ref
        struct[0].h[23,0] = omega_B1_ref
        struct[0].h[24,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i + 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r - 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i - 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r) + v_B4_a_r*(-0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[25,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r) + 1.0*i_B4_a_i*(-0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r) + 0.333333333333333*i_B4_a_r*v_B4_a_r + 1.0*i_B4_a_r*(0.288675134594813*v_B4_b_i - 0.166666666666667*v_B4_b_r) + 1.0*i_B4_a_r*(-0.288675134594813*v_B4_c_i - 0.166666666666667*v_B4_c_r) + 0.333333333333333*i_B4_b_i*v_B4_b_i - 0.166666666666667*i_B4_b_i*v_B4_c_i - 0.288675134594813*i_B4_b_i*v_B4_c_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.288675134594813*i_B4_b_r*v_B4_c_i - 0.166666666666667*i_B4_b_r*v_B4_c_r - 0.166666666666667*i_B4_c_i*v_B4_b_i + 0.288675134594813*i_B4_c_i*v_B4_b_r + 0.333333333333333*i_B4_c_i*v_B4_c_i - 0.288675134594813*i_B4_c_r*v_B4_b_i - 0.166666666666667*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r - v_B4_a_i*(0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r) - v_B4_a_i*(0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r) + v_B4_a_r*(0.288675134594813*i_B4_b_i - 0.166666666666667*i_B4_b_r) + v_B4_a_r*(-0.288675134594813*i_B4_c_i - 0.166666666666667*i_B4_c_r)
        struct[0].h[26,0] = 0.333333333333333*i_B4_a_i*v_B4_a_i + 0.333333333333333*i_B4_a_i*v_B4_b_i + 0.333333333333333*i_B4_a_i*v_B4_c_i + 0.333333333333333*i_B4_a_r*v_B4_a_r + 0.333333333333333*i_B4_a_r*v_B4_b_r + 0.333333333333333*i_B4_a_r*v_B4_c_r + 0.333333333333333*i_B4_b_i*v_B4_a_i + 0.333333333333333*i_B4_b_i*v_B4_b_i + 0.333333333333333*i_B4_b_i*v_B4_c_i + 0.333333333333333*i_B4_b_r*v_B4_a_r + 0.333333333333333*i_B4_b_r*v_B4_b_r + 0.333333333333333*i_B4_b_r*v_B4_c_r + 0.333333333333333*i_B4_c_i*v_B4_a_i + 0.333333333333333*i_B4_c_i*v_B4_b_i + 0.333333333333333*i_B4_c_i*v_B4_c_i + 0.333333333333333*i_B4_c_r*v_B4_a_r + 0.333333333333333*i_B4_c_r*v_B4_b_r + 0.333333333333333*i_B4_c_r*v_B4_c_r
        struct[0].h[27,0] = e_B4_an
        struct[0].h[28,0] = e_B4_bn
        struct[0].h[29,0] = e_B4_cn
        struct[0].h[30,0] = p_B4_ref
        struct[0].h[31,0] = omega_B4_ref
    

    if mode == 10:

        struct[0].Fx[0,0] = -K_delta_B1
        struct[0].Fx[0,1] = 314.159265358979
        struct[0].Fx[1,1] = -1/T_f_B1
        struct[0].Fx[1,4] = -K_f_B1*K_sec_B1/(S_n_B1*T_f_B1)
        struct[0].Fx[2,2] = -K_delta_B4
        struct[0].Fx[2,3] = 314.159265358979
        struct[0].Fx[3,3] = -1/T_f_B4
        struct[0].Fx[3,4] = -K_f_B4*K_sec_B4/(S_n_B4*T_f_B4)

    if mode == 11:

        struct[0].Fy[0,96] = -314.159265358979
        struct[0].Fy[1,16] = -K_f_B1*(-0.333333333333333*i_B1_a_r + 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,17] = -K_f_B1*(-0.333333333333333*i_B1_a_i + 0.166666666666667*i_B1_b_i - 0.288675134594813*i_B1_b_r + 0.166666666666667*i_B1_c_i + 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,18] = -K_f_B1*(-0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.333333333333333*i_B1_b_r + 0.288675134594813*i_B1_c_i + 0.166666666666667*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,19] = -K_f_B1*(0.166666666666667*i_B1_a_i + 0.288675134594813*i_B1_a_r - 0.333333333333333*i_B1_b_i + 0.166666666666667*i_B1_c_i - 0.288675134594813*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,20] = -K_f_B1*(0.288675134594813*i_B1_a_i + 0.166666666666667*i_B1_a_r - 0.288675134594813*i_B1_b_i + 0.166666666666667*i_B1_b_r - 0.333333333333333*i_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,21] = -K_f_B1*(0.166666666666667*i_B1_a_i - 0.288675134594813*i_B1_a_r + 0.166666666666667*i_B1_b_i + 0.288675134594813*i_B1_b_r - 0.333333333333333*i_B1_c_i)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,72] = -K_f_B1*(-0.333333333333333*v_B1_a_r + 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,73] = -K_f_B1*(-0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.333333333333333*v_B1_b_r + 0.288675134594813*v_B1_c_i + 0.166666666666667*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,74] = -K_f_B1*(0.288675134594813*v_B1_a_i + 0.166666666666667*v_B1_a_r - 0.288675134594813*v_B1_b_i + 0.166666666666667*v_B1_b_r - 0.333333333333333*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,78] = -K_f_B1*(-0.333333333333333*v_B1_a_i + 0.166666666666667*v_B1_b_i - 0.288675134594813*v_B1_b_r + 0.166666666666667*v_B1_c_i + 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,79] = -K_f_B1*(0.166666666666667*v_B1_a_i + 0.288675134594813*v_B1_a_r - 0.333333333333333*v_B1_b_i + 0.166666666666667*v_B1_c_i - 0.288675134594813*v_B1_c_r)/(S_n_B1*T_f_B1)
        struct[0].Fy[1,80] = -K_f_B1*(0.166666666666667*v_B1_a_i - 0.288675134594813*v_B1_a_r + 0.166666666666667*v_B1_b_i + 0.288675134594813*v_B1_b_r - 0.333333333333333*v_B1_c_i)/(S_n_B1*T_f_B1)
        struct[0].Fy[2,96] = -314.159265358979
        struct[0].Fy[3,24] = -K_f_B4*(-0.333333333333333*i_B4_a_r + 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,25] = -K_f_B4*(-0.333333333333333*i_B4_a_i + 0.166666666666667*i_B4_b_i - 0.288675134594813*i_B4_b_r + 0.166666666666667*i_B4_c_i + 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,26] = -K_f_B4*(-0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.333333333333333*i_B4_b_r + 0.288675134594813*i_B4_c_i + 0.166666666666667*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,27] = -K_f_B4*(0.166666666666667*i_B4_a_i + 0.288675134594813*i_B4_a_r - 0.333333333333333*i_B4_b_i + 0.166666666666667*i_B4_c_i - 0.288675134594813*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,28] = -K_f_B4*(0.288675134594813*i_B4_a_i + 0.166666666666667*i_B4_a_r - 0.288675134594813*i_B4_b_i + 0.166666666666667*i_B4_b_r - 0.333333333333333*i_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,29] = -K_f_B4*(0.166666666666667*i_B4_a_i - 0.288675134594813*i_B4_a_r + 0.166666666666667*i_B4_b_i + 0.288675134594813*i_B4_b_r - 0.333333333333333*i_B4_c_i)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,84] = -K_f_B4*(-0.333333333333333*v_B4_a_r + 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,85] = -K_f_B4*(-0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.333333333333333*v_B4_b_r + 0.288675134594813*v_B4_c_i + 0.166666666666667*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,86] = -K_f_B4*(0.288675134594813*v_B4_a_i + 0.166666666666667*v_B4_a_r - 0.288675134594813*v_B4_b_i + 0.166666666666667*v_B4_b_r - 0.333333333333333*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,90] = -K_f_B4*(-0.333333333333333*v_B4_a_i + 0.166666666666667*v_B4_b_i - 0.288675134594813*v_B4_b_r + 0.166666666666667*v_B4_c_i + 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,91] = -K_f_B4*(0.166666666666667*v_B4_a_i + 0.288675134594813*v_B4_a_r - 0.333333333333333*v_B4_b_i + 0.166666666666667*v_B4_c_i - 0.288675134594813*v_B4_c_r)/(S_n_B4*T_f_B4)
        struct[0].Fy[3,92] = -K_f_B4*(0.166666666666667*v_B4_a_i - 0.288675134594813*v_B4_a_r + 0.166666666666667*v_B4_b_i + 0.288675134594813*v_B4_b_r - 0.333333333333333*v_B4_c_i)/(S_n_B4*T_f_B4)
        struct[0].Fy[4,96] = -K_agc

        struct[0].Gy[0,0] = -292.221995392108
        struct[0].Gy[0,1] = -139.986584618974
        struct[0].Gy[0,8] = 48.7036658986847
        struct[0].Gy[0,9] = 23.3310974364957
        struct[0].Gy[0,16] = 243.518329493424
        struct[0].Gy[0,17] = 116.655487182478
        struct[0].Gy[0,56] = 1
        struct[0].Gy[1,0] = 139.986584618974
        struct[0].Gy[1,1] = -292.221995392108
        struct[0].Gy[1,8] = -23.3310974364957
        struct[0].Gy[1,9] = 48.7036658986847
        struct[0].Gy[1,16] = -116.655487182478
        struct[0].Gy[1,17] = 243.518329493424
        struct[0].Gy[1,57] = 1
        struct[0].Gy[2,2] = -292.221995392108
        struct[0].Gy[2,3] = -139.986584618974
        struct[0].Gy[2,10] = 48.7036658986847
        struct[0].Gy[2,11] = 23.3310974364957
        struct[0].Gy[2,18] = 243.518329493424
        struct[0].Gy[2,19] = 116.655487182478
        struct[0].Gy[2,58] = 1
        struct[0].Gy[3,2] = 139.986584618974
        struct[0].Gy[3,3] = -292.221995392108
        struct[0].Gy[3,10] = -23.3310974364957
        struct[0].Gy[3,11] = 48.7036658986847
        struct[0].Gy[3,18] = -116.655487182478
        struct[0].Gy[3,19] = 243.518329493424
        struct[0].Gy[3,59] = 1
        struct[0].Gy[4,4] = -292.221995392108
        struct[0].Gy[4,5] = -139.986584618974
        struct[0].Gy[4,12] = 48.7036658986847
        struct[0].Gy[4,13] = 23.3310974364957
        struct[0].Gy[4,20] = 243.518329493424
        struct[0].Gy[4,21] = 116.655487182478
        struct[0].Gy[4,60] = 1
        struct[0].Gy[5,4] = 139.986584618974
        struct[0].Gy[5,5] = -292.221995392108
        struct[0].Gy[5,12] = -23.3310974364957
        struct[0].Gy[5,13] = 48.7036658986847
        struct[0].Gy[5,20] = -116.655487182478
        struct[0].Gy[5,21] = 243.518329493424
        struct[0].Gy[5,61] = 1
        struct[0].Gy[6,6] = -292.221995392108
        struct[0].Gy[6,7] = -139.986584618974
        struct[0].Gy[6,14] = 48.7036658986847
        struct[0].Gy[6,15] = 23.3310974364957
        struct[0].Gy[6,22] = 243.518329493424
        struct[0].Gy[6,23] = 116.655487182478
        struct[0].Gy[7,6] = 139.986584618974
        struct[0].Gy[7,7] = -292.221995392108
        struct[0].Gy[7,14] = -23.3310974364957
        struct[0].Gy[7,15] = 48.7036658986847
        struct[0].Gy[7,22] = -116.655487182478
        struct[0].Gy[7,23] = 243.518329493424
        struct[0].Gy[8,0] = 48.7036658986847
        struct[0].Gy[8,1] = 23.3310974364957
        struct[0].Gy[8,8] = -292.221995392108
        struct[0].Gy[8,9] = -139.986584618974
        struct[0].Gy[8,24] = 243.518329493424
        struct[0].Gy[8,25] = 116.655487182478
        struct[0].Gy[8,64] = 1
        struct[0].Gy[9,0] = -23.3310974364957
        struct[0].Gy[9,1] = 48.7036658986847
        struct[0].Gy[9,8] = 139.986584618974
        struct[0].Gy[9,9] = -292.221995392108
        struct[0].Gy[9,24] = -116.655487182478
        struct[0].Gy[9,25] = 243.518329493424
        struct[0].Gy[9,65] = 1
        struct[0].Gy[10,2] = 48.7036658986847
        struct[0].Gy[10,3] = 23.3310974364957
        struct[0].Gy[10,10] = -292.221995392108
        struct[0].Gy[10,11] = -139.986584618974
        struct[0].Gy[10,26] = 243.518329493424
        struct[0].Gy[10,27] = 116.655487182478
        struct[0].Gy[10,66] = 1
        struct[0].Gy[11,2] = -23.3310974364957
        struct[0].Gy[11,3] = 48.7036658986847
        struct[0].Gy[11,10] = 139.986584618974
        struct[0].Gy[11,11] = -292.221995392108
        struct[0].Gy[11,26] = -116.655487182478
        struct[0].Gy[11,27] = 243.518329493424
        struct[0].Gy[11,67] = 1
        struct[0].Gy[12,4] = 48.7036658986847
        struct[0].Gy[12,5] = 23.3310974364957
        struct[0].Gy[12,12] = -292.221995392108
        struct[0].Gy[12,13] = -139.986584618974
        struct[0].Gy[12,28] = 243.518329493424
        struct[0].Gy[12,29] = 116.655487182478
        struct[0].Gy[12,68] = 1
        struct[0].Gy[13,4] = -23.3310974364957
        struct[0].Gy[13,5] = 48.7036658986847
        struct[0].Gy[13,12] = 139.986584618974
        struct[0].Gy[13,13] = -292.221995392108
        struct[0].Gy[13,28] = -116.655487182478
        struct[0].Gy[13,29] = 243.518329493424
        struct[0].Gy[13,69] = 1
        struct[0].Gy[14,6] = 48.7036658986847
        struct[0].Gy[14,7] = 23.3310974364957
        struct[0].Gy[14,14] = -292.221995392108
        struct[0].Gy[14,15] = -139.986584618974
        struct[0].Gy[14,30] = 243.518329493424
        struct[0].Gy[14,31] = 116.655487182478
        struct[0].Gy[15,6] = -23.3310974364957
        struct[0].Gy[15,7] = 48.7036658986847
        struct[0].Gy[15,14] = 139.986584618974
        struct[0].Gy[15,15] = -292.221995392108
        struct[0].Gy[15,30] = -116.655487182478
        struct[0].Gy[15,31] = 243.518329493424
        struct[0].Gy[16,0] = 243.518329493424
        struct[0].Gy[16,1] = 116.655487182478
        struct[0].Gy[16,16] = -243.518329493424
        struct[0].Gy[16,17] = -116.655487182478
        struct[0].Gy[16,72] = 1
        struct[0].Gy[17,0] = -116.655487182478
        struct[0].Gy[17,1] = 243.518329493424
        struct[0].Gy[17,16] = 116.655487182478
        struct[0].Gy[17,17] = -243.518329493424
        struct[0].Gy[17,78] = 1
        struct[0].Gy[18,2] = 243.518329493424
        struct[0].Gy[18,3] = 116.655487182478
        struct[0].Gy[18,18] = -243.518329493424
        struct[0].Gy[18,19] = -116.655487182478
        struct[0].Gy[18,73] = 1
        struct[0].Gy[19,2] = -116.655487182478
        struct[0].Gy[19,3] = 243.518329493424
        struct[0].Gy[19,18] = 116.655487182478
        struct[0].Gy[19,19] = -243.518329493424
        struct[0].Gy[19,79] = 1
        struct[0].Gy[20,4] = 243.518329493424
        struct[0].Gy[20,5] = 116.655487182478
        struct[0].Gy[20,20] = -243.518329493424
        struct[0].Gy[20,21] = -116.655487182478
        struct[0].Gy[20,74] = 1
        struct[0].Gy[21,4] = -116.655487182478
        struct[0].Gy[21,5] = 243.518329493424
        struct[0].Gy[21,20] = 116.655487182478
        struct[0].Gy[21,21] = -243.518329493424
        struct[0].Gy[21,80] = 1
        struct[0].Gy[22,6] = 243.518329493424
        struct[0].Gy[22,7] = 116.655487182478
        struct[0].Gy[22,22] = -243.518329493424
        struct[0].Gy[22,23] = -116.655487182478
        struct[0].Gy[22,75] = 1
        struct[0].Gy[23,6] = -116.655487182478
        struct[0].Gy[23,7] = 243.518329493424
        struct[0].Gy[23,22] = 116.655487182478
        struct[0].Gy[23,23] = -243.518329493424
        struct[0].Gy[23,81] = 1
        struct[0].Gy[24,8] = 243.518329493424
        struct[0].Gy[24,9] = 116.655487182478
        struct[0].Gy[24,24] = -243.518329493424
        struct[0].Gy[24,25] = -116.655487182478
        struct[0].Gy[24,84] = 1
        struct[0].Gy[25,8] = -116.655487182478
        struct[0].Gy[25,9] = 243.518329493424
        struct[0].Gy[25,24] = 116.655487182478
        struct[0].Gy[25,25] = -243.518329493424
        struct[0].Gy[25,90] = 1
        struct[0].Gy[26,10] = 243.518329493424
        struct[0].Gy[26,11] = 116.655487182478
        struct[0].Gy[26,26] = -243.518329493424
        struct[0].Gy[26,27] = -116.655487182478
        struct[0].Gy[26,85] = 1
        struct[0].Gy[27,10] = -116.655487182478
        struct[0].Gy[27,11] = 243.518329493424
        struct[0].Gy[27,26] = 116.655487182478
        struct[0].Gy[27,27] = -243.518329493424
        struct[0].Gy[27,91] = 1
        struct[0].Gy[28,12] = 243.518329493424
        struct[0].Gy[28,13] = 116.655487182478
        struct[0].Gy[28,28] = -243.518329493424
        struct[0].Gy[28,29] = -116.655487182478
        struct[0].Gy[28,86] = 1
        struct[0].Gy[29,12] = -116.655487182478
        struct[0].Gy[29,13] = 243.518329493424
        struct[0].Gy[29,28] = 116.655487182478
        struct[0].Gy[29,29] = -243.518329493424
        struct[0].Gy[29,92] = 1
        struct[0].Gy[30,14] = 243.518329493424
        struct[0].Gy[30,15] = 116.655487182478
        struct[0].Gy[30,30] = -243.518329493424
        struct[0].Gy[30,31] = -116.655487182478
        struct[0].Gy[30,87] = 1
        struct[0].Gy[31,14] = -116.655487182478
        struct[0].Gy[31,15] = 243.518329493424
        struct[0].Gy[31,30] = 116.655487182478
        struct[0].Gy[31,31] = -243.518329493424
        struct[0].Gy[31,93] = 1
        struct[0].Gy[32,0] = -243.518329493424
        struct[0].Gy[32,1] = -116.655487182478
        struct[0].Gy[32,16] = 243.518329493424
        struct[0].Gy[32,17] = 116.655487182478
        struct[0].Gy[32,32] = -1
        struct[0].Gy[33,0] = 116.655487182478
        struct[0].Gy[33,1] = -243.518329493424
        struct[0].Gy[33,16] = -116.655487182478
        struct[0].Gy[33,17] = 243.518329493424
        struct[0].Gy[33,33] = -1
        struct[0].Gy[34,2] = -243.518329493424
        struct[0].Gy[34,3] = -116.655487182478
        struct[0].Gy[34,18] = 243.518329493424
        struct[0].Gy[34,19] = 116.655487182478
        struct[0].Gy[34,34] = -1
        struct[0].Gy[35,2] = 116.655487182478
        struct[0].Gy[35,3] = -243.518329493424
        struct[0].Gy[35,18] = -116.655487182478
        struct[0].Gy[35,19] = 243.518329493424
        struct[0].Gy[35,35] = -1
        struct[0].Gy[36,4] = -243.518329493424
        struct[0].Gy[36,5] = -116.655487182478
        struct[0].Gy[36,20] = 243.518329493424
        struct[0].Gy[36,21] = 116.655487182478
        struct[0].Gy[36,36] = -1
        struct[0].Gy[37,4] = 116.655487182478
        struct[0].Gy[37,5] = -243.518329493424
        struct[0].Gy[37,20] = -116.655487182478
        struct[0].Gy[37,21] = 243.518329493424
        struct[0].Gy[37,37] = -1
        struct[0].Gy[38,32] = 1
        struct[0].Gy[38,34] = 1
        struct[0].Gy[38,36] = 1
        struct[0].Gy[38,38] = -1
        struct[0].Gy[39,33] = 1
        struct[0].Gy[39,35] = 1
        struct[0].Gy[39,37] = 1
        struct[0].Gy[39,39] = -1
        struct[0].Gy[40,0] = 48.7036658986847
        struct[0].Gy[40,1] = 23.3310974364957
        struct[0].Gy[40,8] = -48.7036658986847
        struct[0].Gy[40,9] = -23.3310974364957
        struct[0].Gy[40,40] = -1
        struct[0].Gy[41,0] = -23.3310974364957
        struct[0].Gy[41,1] = 48.7036658986847
        struct[0].Gy[41,8] = 23.3310974364957
        struct[0].Gy[41,9] = -48.7036658986847
        struct[0].Gy[41,41] = -1
        struct[0].Gy[42,2] = 48.7036658986847
        struct[0].Gy[42,3] = 23.3310974364957
        struct[0].Gy[42,10] = -48.7036658986847
        struct[0].Gy[42,11] = -23.3310974364957
        struct[0].Gy[42,42] = -1
        struct[0].Gy[43,2] = -23.3310974364957
        struct[0].Gy[43,3] = 48.7036658986847
        struct[0].Gy[43,10] = 23.3310974364957
        struct[0].Gy[43,11] = -48.7036658986847
        struct[0].Gy[43,43] = -1
        struct[0].Gy[44,4] = 48.7036658986847
        struct[0].Gy[44,5] = 23.3310974364957
        struct[0].Gy[44,12] = -48.7036658986847
        struct[0].Gy[44,13] = -23.3310974364957
        struct[0].Gy[44,44] = -1
        struct[0].Gy[45,4] = -23.3310974364957
        struct[0].Gy[45,5] = 48.7036658986847
        struct[0].Gy[45,12] = 23.3310974364957
        struct[0].Gy[45,13] = -48.7036658986847
        struct[0].Gy[45,45] = -1
        struct[0].Gy[46,40] = 1
        struct[0].Gy[46,42] = 1
        struct[0].Gy[46,44] = 1
        struct[0].Gy[46,46] = -1
        struct[0].Gy[47,41] = 1
        struct[0].Gy[47,43] = 1
        struct[0].Gy[47,45] = 1
        struct[0].Gy[47,47] = -1
        struct[0].Gy[48,8] = 243.518329493424
        struct[0].Gy[48,9] = 116.655487182478
        struct[0].Gy[48,24] = -243.518329493424
        struct[0].Gy[48,25] = -116.655487182478
        struct[0].Gy[48,48] = -1
        struct[0].Gy[49,8] = -116.655487182478
        struct[0].Gy[49,9] = 243.518329493424
        struct[0].Gy[49,24] = 116.655487182478
        struct[0].Gy[49,25] = -243.518329493424
        struct[0].Gy[49,49] = -1
        struct[0].Gy[50,10] = 243.518329493424
        struct[0].Gy[50,11] = 116.655487182478
        struct[0].Gy[50,26] = -243.518329493424
        struct[0].Gy[50,27] = -116.655487182478
        struct[0].Gy[50,50] = -1
        struct[0].Gy[51,10] = -116.655487182478
        struct[0].Gy[51,11] = 243.518329493424
        struct[0].Gy[51,26] = 116.655487182478
        struct[0].Gy[51,27] = -243.518329493424
        struct[0].Gy[51,51] = -1
        struct[0].Gy[52,12] = 243.518329493424
        struct[0].Gy[52,13] = 116.655487182478
        struct[0].Gy[52,28] = -243.518329493424
        struct[0].Gy[52,29] = -116.655487182478
        struct[0].Gy[52,52] = -1
        struct[0].Gy[53,12] = -116.655487182478
        struct[0].Gy[53,13] = 243.518329493424
        struct[0].Gy[53,28] = 116.655487182478
        struct[0].Gy[53,29] = -243.518329493424
        struct[0].Gy[53,53] = -1
        struct[0].Gy[54,48] = 1
        struct[0].Gy[54,50] = 1
        struct[0].Gy[54,52] = 1
        struct[0].Gy[54,54] = -1
        struct[0].Gy[55,49] = 1
        struct[0].Gy[55,51] = 1
        struct[0].Gy[55,53] = 1
        struct[0].Gy[55,55] = -1
        struct[0].Gy[56,0] = i_load_B2_a_r
        struct[0].Gy[56,1] = i_load_B2_a_i
        struct[0].Gy[56,6] = -i_load_B2_a_r
        struct[0].Gy[56,7] = -i_load_B2_a_i
        struct[0].Gy[56,56] = v_B2_a_r - v_B2_n_r
        struct[0].Gy[56,57] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[57,2] = i_load_B2_b_r
        struct[0].Gy[57,3] = i_load_B2_b_i
        struct[0].Gy[57,6] = -i_load_B2_b_r
        struct[0].Gy[57,7] = -i_load_B2_b_i
        struct[0].Gy[57,58] = v_B2_b_r - v_B2_n_r
        struct[0].Gy[57,59] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[58,4] = i_load_B2_c_r
        struct[0].Gy[58,5] = i_load_B2_c_i
        struct[0].Gy[58,6] = -i_load_B2_c_r
        struct[0].Gy[58,7] = -i_load_B2_c_i
        struct[0].Gy[58,60] = v_B2_c_r - v_B2_n_r
        struct[0].Gy[58,61] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[59,0] = -i_load_B2_a_i
        struct[0].Gy[59,1] = i_load_B2_a_r
        struct[0].Gy[59,6] = i_load_B2_a_i
        struct[0].Gy[59,7] = -i_load_B2_a_r
        struct[0].Gy[59,56] = v_B2_a_i - v_B2_n_i
        struct[0].Gy[59,57] = -v_B2_a_r + v_B2_n_r
        struct[0].Gy[60,2] = -i_load_B2_b_i
        struct[0].Gy[60,3] = i_load_B2_b_r
        struct[0].Gy[60,6] = i_load_B2_b_i
        struct[0].Gy[60,7] = -i_load_B2_b_r
        struct[0].Gy[60,58] = v_B2_b_i - v_B2_n_i
        struct[0].Gy[60,59] = -v_B2_b_r + v_B2_n_r
        struct[0].Gy[61,4] = -i_load_B2_c_i
        struct[0].Gy[61,5] = i_load_B2_c_r
        struct[0].Gy[61,6] = i_load_B2_c_i
        struct[0].Gy[61,7] = -i_load_B2_c_r
        struct[0].Gy[61,60] = v_B2_c_i - v_B2_n_i
        struct[0].Gy[61,61] = -v_B2_c_r + v_B2_n_r
        struct[0].Gy[62,56] = 1
        struct[0].Gy[62,58] = 1
        struct[0].Gy[62,60] = 1
        struct[0].Gy[62,62] = 1
        struct[0].Gy[63,57] = 1
        struct[0].Gy[63,59] = 1
        struct[0].Gy[63,61] = 1
        struct[0].Gy[63,63] = 1
        struct[0].Gy[64,8] = i_load_B3_a_r
        struct[0].Gy[64,9] = i_load_B3_a_i
        struct[0].Gy[64,14] = -i_load_B3_a_r
        struct[0].Gy[64,15] = -i_load_B3_a_i
        struct[0].Gy[64,64] = v_B3_a_r - v_B3_n_r
        struct[0].Gy[64,65] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[65,10] = i_load_B3_b_r
        struct[0].Gy[65,11] = i_load_B3_b_i
        struct[0].Gy[65,14] = -i_load_B3_b_r
        struct[0].Gy[65,15] = -i_load_B3_b_i
        struct[0].Gy[65,66] = v_B3_b_r - v_B3_n_r
        struct[0].Gy[65,67] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[66,12] = i_load_B3_c_r
        struct[0].Gy[66,13] = i_load_B3_c_i
        struct[0].Gy[66,14] = -i_load_B3_c_r
        struct[0].Gy[66,15] = -i_load_B3_c_i
        struct[0].Gy[66,68] = v_B3_c_r - v_B3_n_r
        struct[0].Gy[66,69] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[67,8] = -i_load_B3_a_i
        struct[0].Gy[67,9] = i_load_B3_a_r
        struct[0].Gy[67,14] = i_load_B3_a_i
        struct[0].Gy[67,15] = -i_load_B3_a_r
        struct[0].Gy[67,64] = v_B3_a_i - v_B3_n_i
        struct[0].Gy[67,65] = -v_B3_a_r + v_B3_n_r
        struct[0].Gy[68,10] = -i_load_B3_b_i
        struct[0].Gy[68,11] = i_load_B3_b_r
        struct[0].Gy[68,14] = i_load_B3_b_i
        struct[0].Gy[68,15] = -i_load_B3_b_r
        struct[0].Gy[68,66] = v_B3_b_i - v_B3_n_i
        struct[0].Gy[68,67] = -v_B3_b_r + v_B3_n_r
        struct[0].Gy[69,12] = -i_load_B3_c_i
        struct[0].Gy[69,13] = i_load_B3_c_r
        struct[0].Gy[69,14] = i_load_B3_c_i
        struct[0].Gy[69,15] = -i_load_B3_c_r
        struct[0].Gy[69,68] = v_B3_c_i - v_B3_n_i
        struct[0].Gy[69,69] = -v_B3_c_r + v_B3_n_r
        struct[0].Gy[70,64] = 1
        struct[0].Gy[70,66] = 1
        struct[0].Gy[70,68] = 1
        struct[0].Gy[70,70] = 1
        struct[0].Gy[71,65] = 1
        struct[0].Gy[71,67] = 1
        struct[0].Gy[71,69] = 1
        struct[0].Gy[71,71] = 1
        struct[0].Gy[72,16] = -1
        struct[0].Gy[72,22] = 1
        struct[0].Gy[72,72] = -R_B1_sa
        struct[0].Gy[72,78] = 1.0*X_B1_sa
        struct[0].Gy[73,18] = -1
        struct[0].Gy[73,22] = 1
        struct[0].Gy[73,73] = -R_B1_sb
        struct[0].Gy[73,79] = 1.0*X_B1_sb
        struct[0].Gy[74,20] = -1
        struct[0].Gy[74,22] = 1
        struct[0].Gy[74,74] = -R_B1_sc
        struct[0].Gy[74,80] = 1.0*X_B1_sc
        struct[0].Gy[75,22] = -1
        struct[0].Gy[75,75] = -R_B1_sn
        struct[0].Gy[75,77] = 1
        struct[0].Gy[75,81] = 1.0*X_B1_sn
        struct[0].Gy[76,72] = 1
        struct[0].Gy[76,73] = 1
        struct[0].Gy[76,74] = 1
        struct[0].Gy[76,75] = 1
        struct[0].Gy[76,76] = -1
        struct[0].Gy[77,76] = R_B1_ng
        struct[0].Gy[77,77] = -1
        struct[0].Gy[77,82] = -1.0*X_B1_ng
        struct[0].Gy[78,17] = -1.00000000000000
        struct[0].Gy[78,23] = 1.00000000000000
        struct[0].Gy[78,72] = -1.0*X_B1_sa
        struct[0].Gy[78,78] = -1.0*R_B1_sa
        struct[0].Gy[79,19] = -1.00000000000000
        struct[0].Gy[79,23] = 1.00000000000000
        struct[0].Gy[79,73] = -1.0*X_B1_sb
        struct[0].Gy[79,79] = -1.0*R_B1_sb
        struct[0].Gy[80,21] = -1.00000000000000
        struct[0].Gy[80,23] = 1.00000000000000
        struct[0].Gy[80,74] = -1.0*X_B1_sc
        struct[0].Gy[80,80] = -1.0*R_B1_sc
        struct[0].Gy[81,23] = -1.00000000000000
        struct[0].Gy[81,75] = -1.0*X_B1_sn
        struct[0].Gy[81,81] = -1.0*R_B1_sn
        struct[0].Gy[81,83] = 1.00000000000000
        struct[0].Gy[82,78] = 1.00000000000000
        struct[0].Gy[82,79] = 1.00000000000000
        struct[0].Gy[82,80] = 1.00000000000000
        struct[0].Gy[82,81] = 1.00000000000000
        struct[0].Gy[82,82] = -1.00000000000000
        struct[0].Gy[83,76] = 1.0*X_B1_ng
        struct[0].Gy[83,82] = 1.0*R_B1_ng
        struct[0].Gy[83,83] = -1.00000000000000
        struct[0].Gy[84,24] = -1
        struct[0].Gy[84,30] = 1
        struct[0].Gy[84,84] = -R_B4_sa
        struct[0].Gy[84,90] = 1.0*X_B4_sa
        struct[0].Gy[85,26] = -1
        struct[0].Gy[85,30] = 1
        struct[0].Gy[85,85] = -R_B4_sb
        struct[0].Gy[85,91] = 1.0*X_B4_sb
        struct[0].Gy[86,28] = -1
        struct[0].Gy[86,30] = 1
        struct[0].Gy[86,86] = -R_B4_sc
        struct[0].Gy[86,92] = 1.0*X_B4_sc
        struct[0].Gy[87,30] = -1
        struct[0].Gy[87,87] = -R_B4_sn
        struct[0].Gy[87,89] = 1
        struct[0].Gy[87,93] = 1.0*X_B4_sn
        struct[0].Gy[88,84] = 1
        struct[0].Gy[88,85] = 1
        struct[0].Gy[88,86] = 1
        struct[0].Gy[88,87] = 1
        struct[0].Gy[88,88] = -1
        struct[0].Gy[89,88] = R_B4_ng
        struct[0].Gy[89,89] = -1
        struct[0].Gy[89,94] = -1.0*X_B4_ng
        struct[0].Gy[90,25] = -1.00000000000000
        struct[0].Gy[90,31] = 1.00000000000000
        struct[0].Gy[90,84] = -1.0*X_B4_sa
        struct[0].Gy[90,90] = -1.0*R_B4_sa
        struct[0].Gy[91,27] = -1.00000000000000
        struct[0].Gy[91,31] = 1.00000000000000
        struct[0].Gy[91,85] = -1.0*X_B4_sb
        struct[0].Gy[91,91] = -1.0*R_B4_sb
        struct[0].Gy[92,29] = -1.00000000000000
        struct[0].Gy[92,31] = 1.00000000000000
        struct[0].Gy[92,86] = -1.0*X_B4_sc
        struct[0].Gy[92,92] = -1.0*R_B4_sc
        struct[0].Gy[93,31] = -1.00000000000000
        struct[0].Gy[93,87] = -1.0*X_B4_sn
        struct[0].Gy[93,93] = -1.0*R_B4_sn
        struct[0].Gy[93,95] = 1.00000000000000
        struct[0].Gy[94,90] = 1.00000000000000
        struct[0].Gy[94,91] = 1.00000000000000
        struct[0].Gy[94,92] = 1.00000000000000
        struct[0].Gy[94,93] = 1.00000000000000
        struct[0].Gy[94,94] = -1.00000000000000
        struct[0].Gy[95,88] = 1.0*X_B4_ng
        struct[0].Gy[95,94] = 1.0*R_B4_ng
        struct[0].Gy[95,95] = -1.00000000000000
        struct[0].Gy[96,96] = 1

        struct[0].Gu[56,4] = -1
        struct[0].Gu[57,6] = -1
        struct[0].Gu[58,8] = -1
        struct[0].Gu[59,5] = -1
        struct[0].Gu[60,7] = -1
        struct[0].Gu[61,9] = -1
        struct[0].Gu[64,10] = -1
        struct[0].Gu[65,12] = -1
        struct[0].Gu[66,14] = -1
        struct[0].Gu[67,11] = -1
        struct[0].Gu[68,13] = -1
        struct[0].Gu[69,15] = -1
        struct[0].Gu[72,16] = cos(phi_B1)
        struct[0].Gu[72,19] = -e_B1_an*sin(phi_B1)
        struct[0].Gu[73,17] = cos(phi_B1 - 2.0943951023932)
        struct[0].Gu[73,19] = -e_B1_bn*sin(phi_B1 - 2.0943951023932)
        struct[0].Gu[74,18] = cos(phi_B1 - 4.18879020478639)
        struct[0].Gu[74,19] = -e_B1_cn*sin(phi_B1 - 4.18879020478639)
        struct[0].Gu[78,16] = 1.0*sin(phi_B1)
        struct[0].Gu[78,19] = 1.0*e_B1_an*cos(phi_B1)
        struct[0].Gu[79,17] = 1.0*sin(phi_B1 - 2.0943951023932)
        struct[0].Gu[79,19] = 1.0*e_B1_bn*cos(phi_B1 - 2.0943951023932)
        struct[0].Gu[80,18] = 1.0*sin(phi_B1 - 4.18879020478639)
        struct[0].Gu[80,19] = 1.0*e_B1_cn*cos(phi_B1 - 4.18879020478639)
        struct[0].Gu[84,22] = cos(phi_B4)
        struct[0].Gu[84,25] = -e_B4_an*sin(phi_B4)
        struct[0].Gu[85,23] = cos(phi_B4 - 2.0943951023932)
        struct[0].Gu[85,25] = -e_B4_bn*sin(phi_B4 - 2.0943951023932)
        struct[0].Gu[86,24] = cos(phi_B4 - 4.18879020478639)
        struct[0].Gu[86,25] = -e_B4_cn*sin(phi_B4 - 4.18879020478639)
        struct[0].Gu[90,22] = 1.0*sin(phi_B4)
        struct[0].Gu[90,25] = 1.0*e_B4_an*cos(phi_B4)
        struct[0].Gu[91,23] = 1.0*sin(phi_B4 - 2.0943951023932)
        struct[0].Gu[91,25] = 1.0*e_B4_bn*cos(phi_B4 - 2.0943951023932)
        struct[0].Gu[92,24] = 1.0*sin(phi_B4 - 4.18879020478639)
        struct[0].Gu[92,25] = 1.0*e_B4_cn*cos(phi_B4 - 4.18879020478639)





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
    Fx_ini_rows = [0, 0, 1, 1, 2, 2, 3, 3]

    Fx_ini_cols = [0, 1, 1, 4, 2, 3, 3, 4]

    Fy_ini_rows = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]

    Fy_ini_cols = [96, 16, 17, 18, 19, 20, 21, 72, 73, 74, 78, 79, 80, 96, 24, 25, 26, 27, 28, 29, 84, 85, 86, 90, 91, 92, 96]

    Gx_ini_rows = [72, 73, 74, 78, 79, 80, 84, 85, 86, 90, 91, 92, 96, 96]

    Gx_ini_cols = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 3]

    Gy_ini_rows = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 74, 74, 74, 74, 75, 75, 75, 75, 76, 76, 76, 76, 76, 77, 77, 77, 78, 78, 78, 78, 79, 79, 79, 79, 80, 80, 80, 80, 81, 81, 81, 81, 82, 82, 82, 82, 82, 83, 83, 83, 84, 84, 84, 84, 85, 85, 85, 85, 86, 86, 86, 86, 87, 87, 87, 87, 88, 88, 88, 88, 88, 89, 89, 89, 90, 90, 90, 90, 91, 91, 91, 91, 92, 92, 92, 92, 93, 93, 93, 93, 94, 94, 94, 94, 94, 95, 95, 95, 96]

    Gy_ini_cols = [0, 1, 8, 9, 16, 17, 56, 0, 1, 8, 9, 16, 17, 57, 2, 3, 10, 11, 18, 19, 58, 2, 3, 10, 11, 18, 19, 59, 4, 5, 12, 13, 20, 21, 60, 4, 5, 12, 13, 20, 21, 61, 6, 7, 14, 15, 22, 23, 6, 7, 14, 15, 22, 23, 0, 1, 8, 9, 24, 25, 64, 0, 1, 8, 9, 24, 25, 65, 2, 3, 10, 11, 26, 27, 66, 2, 3, 10, 11, 26, 27, 67, 4, 5, 12, 13, 28, 29, 68, 4, 5, 12, 13, 28, 29, 69, 6, 7, 14, 15, 30, 31, 6, 7, 14, 15, 30, 31, 0, 1, 16, 17, 72, 0, 1, 16, 17, 78, 2, 3, 18, 19, 73, 2, 3, 18, 19, 79, 4, 5, 20, 21, 74, 4, 5, 20, 21, 80, 6, 7, 22, 23, 75, 6, 7, 22, 23, 81, 8, 9, 24, 25, 84, 8, 9, 24, 25, 90, 10, 11, 26, 27, 85, 10, 11, 26, 27, 91, 12, 13, 28, 29, 86, 12, 13, 28, 29, 92, 14, 15, 30, 31, 87, 14, 15, 30, 31, 93, 0, 1, 16, 17, 32, 0, 1, 16, 17, 33, 2, 3, 18, 19, 34, 2, 3, 18, 19, 35, 4, 5, 20, 21, 36, 4, 5, 20, 21, 37, 32, 34, 36, 38, 33, 35, 37, 39, 0, 1, 8, 9, 40, 0, 1, 8, 9, 41, 2, 3, 10, 11, 42, 2, 3, 10, 11, 43, 4, 5, 12, 13, 44, 4, 5, 12, 13, 45, 40, 42, 44, 46, 41, 43, 45, 47, 8, 9, 24, 25, 48, 8, 9, 24, 25, 49, 10, 11, 26, 27, 50, 10, 11, 26, 27, 51, 12, 13, 28, 29, 52, 12, 13, 28, 29, 53, 48, 50, 52, 54, 49, 51, 53, 55, 0, 1, 6, 7, 56, 57, 2, 3, 6, 7, 58, 59, 4, 5, 6, 7, 60, 61, 0, 1, 6, 7, 56, 57, 2, 3, 6, 7, 58, 59, 4, 5, 6, 7, 60, 61, 56, 58, 60, 62, 57, 59, 61, 63, 8, 9, 14, 15, 64, 65, 10, 11, 14, 15, 66, 67, 12, 13, 14, 15, 68, 69, 8, 9, 14, 15, 64, 65, 10, 11, 14, 15, 66, 67, 12, 13, 14, 15, 68, 69, 64, 66, 68, 70, 65, 67, 69, 71, 16, 22, 72, 78, 18, 22, 73, 79, 20, 22, 74, 80, 22, 75, 77, 81, 72, 73, 74, 75, 76, 76, 77, 82, 17, 23, 72, 78, 19, 23, 73, 79, 21, 23, 74, 80, 23, 75, 81, 83, 78, 79, 80, 81, 82, 76, 82, 83, 24, 30, 84, 90, 26, 30, 85, 91, 28, 30, 86, 92, 30, 87, 89, 93, 84, 85, 86, 87, 88, 88, 89, 94, 25, 31, 84, 90, 27, 31, 85, 91, 29, 31, 86, 92, 31, 87, 93, 95, 90, 91, 92, 93, 94, 88, 94, 95, 96]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols