import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos

class imib_fisix_3rd_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 3
        self.N_y = 6 
        self.N_z = 3 
        self.N_store = 10000 
        self.params_list = ['S_b', 'U_b', 'I_b', 'R_s', 'R_r', 'L_ls', 'L_lr', 'L_m', 'H_m', 'Omega_b', 'D', 'v_0', 'theta_0', 'X_l', 'omega_s', 'v_rd', 'v_rq', 'v_sd', 'v_sq'] 
        self.params_values_list  = [90000.0, 400.0, 129.9038105676658, 0.012937500000000001, 0.013500000000000002, 0.048375, 0.11025000000000001, 2.0812500000000003, 3.5, 314.1592653589793, 0.1, 1, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0, -1.0] 
        self.inputs_ini_list = ['tau_m', 'Q_c'] 
        self.inputs_ini_values_list  = [0.1, 0.0] 
        self.inputs_run_list = ['tau_m', 'Q_c'] 
        self.inputs_run_values_list = [0.1, 0.0] 
        self.x_list = ['omega_e', 'psi_rd', 'psi_rq'] 
        self.y_list = ['psi_sd', 'psi_sq', 'i_sd', 'i_sq', 'i_rd', 'i_rq'] 
        self.xy_list = self.x_list + self.y_list 
        self.y_ini_list = ['psi_sd', 'psi_sq', 'i_sd', 'i_sq', 'i_rd', 'i_rq'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.t = 0.0
        self.it = 0
        self.it_store = 0
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
              ('f_ini', np.float64, (self.N_x,1)),
              ('x', np.float64, (self.N_x,1)),
              ('x_ini', np.float64, (self.N_x,1)),
              ('x_0', np.float64, (self.N_x,1)),
              ('g', np.float64, (self.N_y,1)),
              ('g_ini', np.float64, (self.N_y,1)),
              ('y', np.float64, (self.N_y,1)),
              ('y_ini', np.float64, (self.N_y,1)),
              ('y_0', np.float64, (self.N_y,1)),
              ('h', np.float64, (self.N_z,1)),
              ('Fx', np.float64, (self.N_x,self.N_x)),
              ('Fy', np.float64, (self.N_x,self.N_y)),
              ('Gx', np.float64, (self.N_y,self.N_x)),
              ('Gy', np.float64, (self.N_y,self.N_y)),
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
                np.zeros((self.N_x,1)),                # f_ini
                np.zeros((self.N_x,1)),                # x
                np.zeros((self.N_x,1)),                # x_ini
                np.zeros((self.N_x,1)),                # x_0
                np.zeros((self.N_y,1)),                # g
                np.zeros((self.N_y,1)),                # g_ini
                np.zeros((self.N_y,1)),                # y
                np.zeros((self.N_y,1)),                # y_ini
                np.zeros((self.N_y,1)),                # y_0
                np.zeros((self.N_z,1)),                # h
                np.zeros((self.N_x,self.N_x)),         # Fx   
                np.zeros((self.N_x,self.N_y)),         # Fy 
                np.zeros((self.N_y,self.N_x)),         # Gx 
                np.zeros((self.N_y,self.N_y)),         # Fy 
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
        self.struct[0].x_ini[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini(self.struct,2)
        ini(self.struct,3)       
        fg = np.vstack((self.struct[0].f_ini,self.struct[0].g_ini))[:,0]
        return fg

    def dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(0.0,self.struct,10)
        run(0.0,self.struct,11)       
        A_c = np.block([[self.struct[0].Fx,self.struct[0].Fy],
                        [self.struct[0].Gx,self.struct[0].Gy]])
        return A_c

    def ini_dae_jacobian(self,x):
        self.struct[0].x_ini[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini(self.struct,10)
        ini(self.struct,11)       
        A_c = np.block([[self.struct[0].Fx_ini,self.struct[0].Fy_ini],
                        [self.struct[0].Gx_ini,self.struct[0].Gy_ini]])
        return A_c

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

    def simulate(self,events):
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
        xy0 = np.ones(self.N_x+self.N_y)
        xy = sopt.fsolve(self.ini_problem,xy0 )
        self.struct[0].x[:,0] = xy[0:self.N_x]
        self.struct[0].y[:,0] = xy[self.N_x:]

        ## y_ini to u_run
        for item in self.inputs_run_list:
            if item in self.y_ini_list:
                self.struct[0][item] = self.struct[0].y_ini[self.y_ini_list.index(item)]

        ## u_ini to y_run
        for item in self.inputs_ini_list:
            if item in self.y_list:
                self.struct[0].y[self.y_list.index(item)] = self.struct[0][item]
    
        ## solve selfem
        daesolver(self.struct)    # run until first event

        # simulation run
        for event in events[1:]:  
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        
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


@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_b = struct[0].S_b
    U_b = struct[0].U_b
    I_b = struct[0].I_b
    R_s = struct[0].R_s
    R_r = struct[0].R_r
    L_ls = struct[0].L_ls
    L_lr = struct[0].L_lr
    L_m = struct[0].L_m
    H_m = struct[0].H_m
    Omega_b = struct[0].Omega_b
    D = struct[0].D
    v_0 = struct[0].v_0
    theta_0 = struct[0].theta_0
    X_l = struct[0].X_l
    omega_s = struct[0].omega_s
    v_rd = struct[0].v_rd
    v_rq = struct[0].v_rq
    v_sd = struct[0].v_sd
    v_sq = struct[0].v_sq
    
    # Inputs:
    tau_m = struct[0].tau_m
    Q_c = struct[0].Q_c
    
    # Dynamical states:
    omega_e = struct[0].x[0,0]
    psi_rd = struct[0].x[1,0]
    psi_rq = struct[0].x[2,0]
    
    # Algebraic states:
    psi_sd = struct[0].y[0,0]
    psi_sq = struct[0].y[1,0]
    i_sd = struct[0].y[2,0]
    i_sq = struct[0].y[3,0]
    i_rd = struct[0].y[4,0]
    i_rq = struct[0].y[5,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-D*omega_e + i_sd*psi_sq - i_sq*psi_sd + tau_m)/(2*H_m)
        struct[0].f[1,0] = Omega_b*(-R_s*i_sd - omega_s*psi_sq - v_sd)
        struct[0].f[2,0] = Omega_b*(-R_s*i_sq + omega_s*psi_sd - v_sq)
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = Omega_b*(-R_s*i_sd - omega_s*psi_sq - v_sd)
        struct[0].g[1,0] = Omega_b*(-R_s*i_sq + omega_s*psi_sd - v_sq)
        struct[0].g[2,0] = L_m*i_rd + i_sd*(L_ls + L_m) - psi_sd
        struct[0].g[3,0] = L_m*i_rq + i_sq*(L_ls + L_m) - psi_sq
        struct[0].g[4,0] = L_m*i_sd + i_rd*(L_lr + L_m) - psi_rd
        struct[0].g[5,0] = L_m*i_sq + i_rq*(L_lr + L_m) - psi_rq
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = I_b*(i_sd**2 + i_sq**2)**0.5
        struct[0].h[1,0] = i_sd*v_sd + i_sq*v_sq
        struct[0].h[2,0] = -i_sd*v_sq + i_sq*v_sd
    

    if mode == 10:

        struct[0].Fx[0,0] = -D/(2*H_m)

    if mode == 11:

        struct[0].Fy[0,0] = -i_sq/(2*H_m)
        struct[0].Fy[0,1] = i_sd/(2*H_m)
        struct[0].Fy[0,2] = psi_sq/(2*H_m)
        struct[0].Fy[0,3] = -psi_sd/(2*H_m)
        struct[0].Fy[1,1] = -Omega_b*omega_s
        struct[0].Fy[1,2] = -Omega_b*R_s
        struct[0].Fy[2,0] = Omega_b*omega_s
        struct[0].Fy[2,3] = -Omega_b*R_s

        struct[0].Gx[4,1] = -1
        struct[0].Gx[5,2] = -1

        struct[0].Gy[0,1] = -Omega_b*omega_s
        struct[0].Gy[0,2] = -Omega_b*R_s
        struct[0].Gy[1,0] = Omega_b*omega_s
        struct[0].Gy[1,3] = -Omega_b*R_s
        struct[0].Gy[2,0] = -1
        struct[0].Gy[2,2] = L_ls + L_m
        struct[0].Gy[2,4] = L_m
        struct[0].Gy[3,1] = -1
        struct[0].Gy[3,3] = L_ls + L_m
        struct[0].Gy[3,5] = L_m
        struct[0].Gy[4,2] = L_m
        struct[0].Gy[4,4] = L_lr + L_m
        struct[0].Gy[5,3] = L_m
        struct[0].Gy[5,5] = L_lr + L_m



@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    S_b = struct[0].S_b
    U_b = struct[0].U_b
    I_b = struct[0].I_b
    R_s = struct[0].R_s
    R_r = struct[0].R_r
    L_ls = struct[0].L_ls
    L_lr = struct[0].L_lr
    L_m = struct[0].L_m
    H_m = struct[0].H_m
    Omega_b = struct[0].Omega_b
    D = struct[0].D
    v_0 = struct[0].v_0
    theta_0 = struct[0].theta_0
    X_l = struct[0].X_l
    omega_s = struct[0].omega_s
    v_rd = struct[0].v_rd
    v_rq = struct[0].v_rq
    v_sd = struct[0].v_sd
    v_sq = struct[0].v_sq
    
    # Inputs:
    tau_m = struct[0].tau_m
    Q_c = struct[0].Q_c
    
    # Dynamical states:
    omega_e = struct[0].x_ini[0,0]
    psi_rd = struct[0].x_ini[1,0]
    psi_rq = struct[0].x_ini[2,0]
    
    # Algebraic states:
    psi_sd = struct[0].y_ini[0,0]
    psi_sq = struct[0].y_ini[1,0]
    i_sd = struct[0].y_ini[2,0]
    i_sq = struct[0].y_ini[3,0]
    i_rd = struct[0].y_ini[4,0]
    i_rq = struct[0].y_ini[5,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f_ini[0,0] = (-D*omega_e + i_sd*psi_sq - i_sq*psi_sd + tau_m)/(2*H_m)
        struct[0].f_ini[1,0] = Omega_b*(-R_s*i_sd - omega_s*psi_sq - v_sd)
        struct[0].f_ini[2,0] = Omega_b*(-R_s*i_sq + omega_s*psi_sd - v_sq)
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g_ini[0,0] = Omega_b*(-R_s*i_sd - omega_s*psi_sq - v_sd)
        struct[0].g_ini[1,0] = Omega_b*(-R_s*i_sq + omega_s*psi_sd - v_sq)
        struct[0].g_ini[2,0] = L_m*i_rd + i_sd*(L_ls + L_m) - psi_sd
        struct[0].g_ini[3,0] = L_m*i_rq + i_sq*(L_ls + L_m) - psi_sq
        struct[0].g_ini[4,0] = L_m*i_sd + i_rd*(L_lr + L_m) - psi_rd
        struct[0].g_ini[5,0] = L_m*i_sq + i_rq*(L_lr + L_m) - psi_rq
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = I_b*(i_sd**2 + i_sq**2)**0.5
        struct[0].h[1,0] = i_sd*v_sd + i_sq*v_sq
        struct[0].h[2,0] = -i_sd*v_sq + i_sq*v_sd
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -D/(2*H_m)

    if mode == 11:

        struct[0].Fy_ini[0,0] = -i_sq/(2*H_m)
        struct[0].Fy_ini[0,1] = i_sd/(2*H_m)
        struct[0].Fy_ini[0,2] = psi_sq/(2*H_m)
        struct[0].Fy_ini[0,3] = -psi_sd/(2*H_m)
        struct[0].Fy_ini[1,1] = -Omega_b*omega_s
        struct[0].Fy_ini[1,2] = -Omega_b*R_s
        struct[0].Fy_ini[2,0] = Omega_b*omega_s
        struct[0].Fy_ini[2,3] = -Omega_b*R_s

        struct[0].Gx_ini[4,1] = -1
        struct[0].Gx_ini[5,2] = -1

        struct[0].Gy_ini[0,1] = -Omega_b*omega_s
        struct[0].Gy_ini[0,2] = -Omega_b*R_s
        struct[0].Gy_ini[1,0] = Omega_b*omega_s
        struct[0].Gy_ini[1,3] = -Omega_b*R_s
        struct[0].Gy_ini[2,0] = -1
        struct[0].Gy_ini[2,2] = L_ls + L_m
        struct[0].Gy_ini[2,4] = L_m
        struct[0].Gy_ini[3,1] = -1
        struct[0].Gy_ini[3,3] = L_ls + L_m
        struct[0].Gy_ini[3,5] = L_m
        struct[0].Gy_ini[4,2] = L_m
        struct[0].Gy_ini[4,4] = L_lr + L_m
        struct[0].Gy_ini[5,3] = L_m
        struct[0].Gy_ini[5,5] = L_lr + L_m





@numba.njit(cache=True)
def Piecewise(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out


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
        struct[i].Y[0,:] = struct[i].y[:,0]  
        struct[i].Z[0,:] = struct[i].h[:,0]  

    solver = struct[i].solvern 
    while t<t_end: 
        struct[i].it += 1
        struct[i].t += Dt
        
        it = struct[i].it
        t = struct[i].t
        it_store = struct[i].it_store
        
        #perturbations(t,struct) 
        
        if solver == 1: 
            # forward euler solver  
            run(t,struct, 2)  
            struct[i].x[:] += Dt*struct[i].f  
 
        if solver == 2: 
            
            # bacward euler solver
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                run(t,struct, 2) 
                run(t,struct, 3) 
                run(t,struct, 10)  
                phi =  x_0 + Dt*struct[i].f - struct[i].x 
                Dx = np.linalg.solve(-(Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                if np.max(np.abs(Dx)) < struct[i].itol: break 
            print(struct[i].f)
 
        if solver == 3: 
            # trapezoidal solver
            run(t,struct, 2) 
            f_0 = np.copy(struct[i].f[:]) 
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                run(t,struct, 10)  
                phi =  x_0 + 0.5*Dt*(f_0 + struct[i].f) - struct[i].x 
                Dx = np.linalg.solve(-(0.5*Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                run(t,struct, 2) 
                if np.max(np.abs(Dx)) < struct[i].itol: break 

        if solver == 4: # Teapezoidal DAE as in Milano's book

            run(t,struct, 2) 
            run(t,struct, 3) 

            x = np.copy(struct[i].x[:]) 
            y = np.copy(struct[i].y[:]) 
            f = np.copy(struct[i].f[:]) 
            g = np.copy(struct[i].g[:]) 
            
            for iter in range(struct[i].imax):
                run(t,struct, 2) 
                run(t,struct, 3) 
                run(t,struct,10) 
                run(t,struct,11) 
                
                x_i = struct[i].x[:] 
                y_i = struct[i].y[:]  
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
                struct[i].y[:] = y_i

                if np.max(np.abs(Dxy_i[:,0]))<struct[i].itol:
                    
                    break
                
                # if iter>struct[i].imax-2:
                    
                #     print('Convergence problem')

            struct[i].x[:] = x_i
            struct[i].y[:] = y_i
            
        if solver == 5: # Teapezoidal DAE as in Milano's book

            run(t,struct, 2) 
            run(t,struct, 3) 

            x = np.copy(struct[i].x[:]) 
            y = np.copy(struct[i].y[:]) 
            f = np.copy(struct[i].f[:]) 
            g = np.copy(struct[i].g[:]) 
            
            for iter in range(struct[i].imax):
                run(t,struct, 2) 
                run(t,struct, 3) 
                run(t,struct,10) 
                run(t,struct,11) 
                
                x_i = struct[i].x[:] 
                y_i = struct[i].y[:]  
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
                struct[i].y[:] = y_i

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
            struct[i].y[:] = y_i
                
        # channels 
        if it >= it_store*decimation: 
            struct[i]['T'][it_store+1] = t 
            struct[i].X[it_store+1,:] = struct[i].x[:,0] 
            struct[i].Y[it_store+1,:] = struct[i].y[:,0]
            struct[i].Z[it_store+1,:] = struct[i].h[:,0]
            struct[i].iters[it_store+1,0] = iter
            struct[i].it_store += 1 
            
    struct[i].t = t
    struct[i].it_store = it_store
    return t