import numpy as np
import numba

class {name}_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 3 
        self.imax = 100 
        self.N_x = {N_x}
        self.N_y = {N_y} 
        self.N_z = {N_z} 
        self.N_store = 10000 
        self.params_list = {params_list} 
        self.params_values_list  = {params_values_list} 
        self.inputs_ini_list = {inputs_ini_list} 
        self.inputs_ini_values_list  = {inputs_ini_values_list} 
        self.inputs_run_list = {inputs_run_list} 
        self.inputs_run_values_list = {inputs_run_values_list} 
        self.x_list = {x_list} 
        self.y_list = {y_list} 
        self.xy_list = self.x_list + self.y_list 
        self.y_ini_list = {y_ini_list} 
        self.xy_ini_list = self.x_list + self.y_ini_list 
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
              ('idx', np.int64),
              ('f', np.float64, (self.N_x,1)),
              ('x', np.float64, (self.N_x,1)),
              ('x_0', np.float64, (self.N_x,1)),
              ('h', np.float64, (self.N_z,1)),
              ('Fx', np.float64, (self.N_x,self.N_x)),
              ('T', np.float64, (self.N_store+1,1)),
              ('X', np.float64, (self.N_store+1,{N_x})),
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
                0,
                np.zeros((self.N_x,1)),
                np.zeros((self.N_x,1)),
                np.zeros((self.N_x,1)),
                np.zeros((self.N_z,1)),
                np.zeros((self.N_x,self.N_x)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,self.N_x)),
                ]  

        dt += [(item,np.float64) for item in self.params_list]
        values += [item for item in self.params_values_list]

        dt += [(item,np.float64) for item in self.inputs_list]
        values += [item for item in self.inputs_values_list]

        self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))


    def ini_problem(self,x):
        self.struct[0].x_ini[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        initialization(self.struct)
        fg = np.vstack((self.struct[0].f_ini,self.struct[0].g_ini))[:,0]
        return fg

    def run_problem(self,x):
        t = self.struct[0].t
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(t,self.struct,2)
        run(t,self.struct,3)
        run(t,self.struct,10)
        run(t,self.struct,11)
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg

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