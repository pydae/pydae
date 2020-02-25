import numpy as np
import numba
from pydae.nummath import interp


class smib_milano_ex8p1_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.001000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 3 
        self.imax = 100 
        self.N_x = 2 
        self.N_y = 9 
        self.N_store = 10000 
        self.x_list = ['delta', 'omega'] 
        self.y_list = ['p_e', 'i_d', 'i_q', 'v_d', 'v_q', 'v_1', 'theta_1', 'P_t', 'Q_t'] 
        self.xy_list = self.x_list + self.y_list 
        self.y_ini_list = ['p_e', 'i_d', 'i_q', 'v_d', 'v_q', 'v_1', 'theta_1', 'p_m', 'e1q'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.update() 

    def update(self): 

        self.N_steps = int(np.ceil(self.t_end/self.Dt)) 
        dt =  [  
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
                ('X_d', np.float64),
                ('X1d', np.float64),
                ('T1d0', np.float64),
                ('X_q', np.float64),
                ('X1q', np.float64),
                ('T1q0', np.float64),
                ('R_a', np.float64),
                ('X_l', np.float64),
                ('H', np.float64),
                ('D', np.float64),
                ('Omega_b', np.float64),
                ('omega_s', np.float64),
                ('v_0', np.float64),
                ('theta_0', np.float64),
                ('p_m', np.float64),
                ('e1q', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (2,1)),
                    ('x', np.float64, (2,1)),
                    ('x_0', np.float64, (2,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (2,2)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,2)),
                    ('Y', np.float64, (self.N_store+1,9)),
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
                1.81,   # X_d 
                 0.3,   # X1d 
                 8.0,   # T1d0 
                 1.76,   # X_q 
                 0.65,   # X1q 
                 1.0,   # T1q0 
                 0.003,   # R_a 
                 0.05,   # X_l 
                 3.5,   # H 
                 1.0,   # D 
                 314.1592653589793,   # Omega_b 
                 1.0,   # omega_s 
                 0.9008,   # v_0 
                 0.0,   # theta_0 
                 0.2,   # p_m 
                 1.2,   # e1q 
                 2,
                0,
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((1,1)),
                                np.zeros((2,2)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,2)),
                np.zeros((self.N_store+1,9)),
                ]  
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]
        dt +=     [('it', np.int64)]
        values += [0]
        dt +=     [('it_store', np.int64)]
        values += [0]
        dt +=     [('N_y', np.int64)]
        values += [self.N_y]

        dt +=     [('g', np.float64, (9,1))]
        values += [np.zeros((9,1))]
        dt +=     [('y', np.float64, (9,1))]
        values += [np.zeros((9,1))]
        dt +=     [('Fy', np.float64, (2,9))]
        values += [np.zeros((2,9))]
        dt +=     [('Gx', np.float64, (9,2))]
        values += [np.zeros((9,2))]
        dt +=     [('Gy', np.float64, (9,9))]
        values += [np.zeros((9,9))]




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


    def ss_run(self):
        t=0.0
        run(t,self.struct,2)
        run(t,self.struct,3)
        run(t,self.struct,10)
        run(t,self.struct,11)
        if np.max(np.abs(self.struct[0].f))>1e-4:
            print('the system is not in steady state!')

@numba.jit(nopython=True, cache=True)
def run(t,struct, mode):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    it = 0

    # parameters 
    X_d = struct[it].X_d
    X1d = struct[it].X1d
    T1d0 = struct[it].T1d0
    X_q = struct[it].X_q
    X1q = struct[it].X1q
    T1q0 = struct[it].T1q0
    R_a = struct[it].R_a
    X_l = struct[it].X_l
    H = struct[it].H
    D = struct[it].D
    Omega_b = struct[it].Omega_b
    omega_s = struct[it].omega_s
    v_0 = struct[it].v_0
    theta_0 = struct[it].theta_0

    # inputs 
    p_m = struct[it].p_m
    e1q = struct[it].e1q

    # states 
    delta  = struct[it].x[0,0] 
    omega  = struct[it].x[1,0] 


    # algebraic states 
    p_e = struct[it].y[0,0] 
    i_d = struct[it].y[1,0] 
    i_q = struct[it].y[2,0] 
    v_d = struct[it].y[3,0] 
    v_q = struct[it].y[4,0] 
    v_1 = struct[it].y[5,0] 
    theta_1 = struct[it].y[6,0] 
    P_t = struct[it].y[7,0] 
    Q_t = struct[it].y[8,0] 


    if mode==2: # derivatives 

        ddelta = Omega_b*(omega - omega_s) 
        domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s)) 

        struct[it].f[0,0] = ddelta   
        struct[it].f[1,0] = domega   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e  
        struct[it].g[1,0] = R_a*i_q + X1d*i_d - e1q + v_q  
        struct[it].g[2,0] = R_a*i_d - X1q*i_q + v_d  
        struct[it].g[3,0] = v_1*sin(delta - theta_1) - v_d  
        struct[it].g[4,0] = v_1*cos(delta - theta_1) - v_q  
        struct[it].g[5,0] = P_t + v_0*v_1*sin(theta_0 - theta_1)/X_l  
        struct[it].g[6,0] = Q_t + v_0*v_1*cos(theta_0 - theta_1)/X_l - v_1**2/X_l  
        struct[it].g[7,0] = -P_t + i_d*v_d + i_q*v_q  
        struct[it].g[8,0] = -Q_t + i_d*v_q - i_q*v_d  

    if mode==4: # outputs 

        struct[it].h[0,0] = omega  
    

    if mode==10: # Fx 

        struct[it].Fx[0,1] = Omega_b 
        struct[it].Fx[1,1] = -D/(2*H) 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[1,0] = -1/(2*H) 
    

        struct[it].Gx[3,0] = v_1*cos(delta - theta_1) 
        struct[it].Gx[4,0] = -v_1*sin(delta - theta_1) 
    

        struct[it].Gy[0,0] = -1 
        struct[it].Gy[0,1] = 2*R_a*i_d + v_d 
        struct[it].Gy[0,2] = 2*R_a*i_q + v_q 
        struct[it].Gy[0,3] = i_d 
        struct[it].Gy[0,4] = i_q 
        struct[it].Gy[1,1] = X1d 
        struct[it].Gy[1,2] = R_a 
        struct[it].Gy[1,4] = 1 
        struct[it].Gy[2,1] = R_a 
        struct[it].Gy[2,2] = -X1q 
        struct[it].Gy[2,3] = 1 
        struct[it].Gy[3,3] = -1 
        struct[it].Gy[3,5] = sin(delta - theta_1) 
        struct[it].Gy[3,6] = -v_1*cos(delta - theta_1) 
        struct[it].Gy[4,4] = -1 
        struct[it].Gy[4,5] = cos(delta - theta_1) 
        struct[it].Gy[4,6] = v_1*sin(delta - theta_1) 
        struct[it].Gy[5,5] = v_0*sin(theta_0 - theta_1)/X_l 
        struct[it].Gy[5,6] = -v_0*v_1*cos(theta_0 - theta_1)/X_l 
        struct[it].Gy[5,7] = 1 
        struct[it].Gy[6,5] = v_0*cos(theta_0 - theta_1)/X_l - 2*v_1/X_l 
        struct[it].Gy[6,6] = v_0*v_1*sin(theta_0 - theta_1)/X_l 
        struct[it].Gy[6,8] = 1 
        struct[it].Gy[7,1] = v_d 
        struct[it].Gy[7,2] = v_q 
        struct[it].Gy[7,3] = i_d 
        struct[it].Gy[7,4] = i_q 
        struct[it].Gy[7,7] = -1 
        struct[it].Gy[8,1] = v_q 
        struct[it].Gy[8,2] = -v_d 
        struct[it].Gy[8,3] = -i_q 
        struct[it].Gy[8,4] = i_d 
        struct[it].Gy[8,8] = -1 


@numba.njit(cache=True)
def initialization(struct):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
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
    p_m = struct[0].p_m 
    e1q = struct[0].e1q 
    P_t = struct[0].P_t 
    Q_t = struct[0].Q_t 
    delta = struct[0].x_ini[0,0] 
    omega = struct[0].x_ini[1,0] 
    p_e = struct[0].y_ini[0,0] 
    i_d = struct[0].y_ini[1,0] 
    i_q = struct[0].y_ini[2,0] 
    v_d = struct[0].y_ini[3,0] 
    v_q = struct[0].y_ini[4,0] 
    v_1 = struct[0].y_ini[5,0] 
    theta_1 = struct[0].y_ini[6,0] 
    p_m = struct[0].y_ini[7,0] 
    e1q = struct[0].y_ini[8,0] 
    struct[0].f_ini[0,0] = Omega_b*(omega - omega_s) 
    struct[0].f_ini[1,0] = (-D*(omega - omega_s) - p_e + p_m)/(2*H) 
    struct[0].g_ini[0,0]  = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e  
    struct[0].g_ini[1,0]  = R_a*i_q + X1d*i_d - e1q + v_q  
    struct[0].g_ini[2,0]  = R_a*i_d - X1q*i_q + v_d  
    struct[0].g_ini[3,0]  = v_1*sin(delta - theta_1) - v_d  
    struct[0].g_ini[4,0]  = v_1*cos(delta - theta_1) - v_q  
    struct[0].g_ini[5,0]  = P_t + v_0*v_1*sin(theta_0 - theta_1)/X_l  
    struct[0].g_ini[6,0]  = Q_t + v_0*v_1*cos(theta_0 - theta_1)/X_l - v_1**2/X_l  
    struct[0].g_ini[7,0]  = -P_t + i_d*v_d + i_q*v_q  
    struct[0].g_ini[8,0]  = -Q_t + i_d*v_q - i_q*v_d  
    struct[0].Fx_ini[0,1] = Omega_b 
    struct[0].Fx_ini[1,1] = -D/(2*H) 
    struct[0].Fy_ini[1,0] = -1/(2*H) 
    struct[0].Fy_ini[1,7] = 1/(2*H) 
    struct[0].Gx_ini[3,0] = v_1*cos(delta - theta_1) 
    struct[0].Gx_ini[4,0] = -v_1*sin(delta - theta_1) 
    struct[0].Gy_ini[0,0] = -1 
    struct[0].Gy_ini[0,1] = 2*R_a*i_d + v_d 
    struct[0].Gy_ini[0,2] = 2*R_a*i_q + v_q 
    struct[0].Gy_ini[0,3] = i_d 
    struct[0].Gy_ini[0,4] = i_q 
    struct[0].Gy_ini[1,1] = X1d 
    struct[0].Gy_ini[1,2] = R_a 
    struct[0].Gy_ini[1,4] = 1 
    struct[0].Gy_ini[1,8] = -1 
    struct[0].Gy_ini[2,1] = R_a 
    struct[0].Gy_ini[2,2] = -X1q 
    struct[0].Gy_ini[2,3] = 1 
    struct[0].Gy_ini[3,3] = -1 
    struct[0].Gy_ini[3,5] = sin(delta - theta_1) 
    struct[0].Gy_ini[3,6] = -v_1*cos(delta - theta_1) 
    struct[0].Gy_ini[4,4] = -1 
    struct[0].Gy_ini[4,5] = cos(delta - theta_1) 
    struct[0].Gy_ini[4,6] = v_1*sin(delta - theta_1) 
    struct[0].Gy_ini[5,5] = v_0*sin(theta_0 - theta_1)/X_l 
    struct[0].Gy_ini[5,6] = -v_0*v_1*cos(theta_0 - theta_1)/X_l 
    struct[0].Gy_ini[6,5] = v_0*cos(theta_0 - theta_1)/X_l - 2*v_1/X_l 
    struct[0].Gy_ini[6,6] = v_0*v_1*sin(theta_0 - theta_1)/X_l 
    struct[0].Gy_ini[7,1] = v_d 
    struct[0].Gy_ini[7,2] = v_q 
    struct[0].Gy_ini[7,3] = i_d 
    struct[0].Gy_ini[7,4] = i_q 
    struct[0].Gy_ini[8,1] = v_q 
    struct[0].Gy_ini[8,2] = -v_d 
    struct[0].Gy_ini[8,3] = -i_q 
    struct[0].Gy_ini[8,4] = i_d 


def ini_struct(dt,values):

    dt += [('P_t', np.float64)] 
    values += [0.500000] 
    dt += [('Q_t', np.float64)] 
    values += [0.100000] 
    dt +=     [('x_ini', np.float64, (2,1))]
    values += [np.zeros((2,1))]
    dt +=     [('y_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('f_ini', np.float64, (2,1))]
    values += [np.zeros((2,1))]
    dt +=     [('g_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('Fx_ini', np.float64, (2,2))]
    values += [np.zeros((2,2))]
    dt +=     [('Fy_ini', np.float64, (2,9))]
    values += [np.zeros((2,9))]
    dt +=     [('Gx_ini', np.float64, (9,2))]
    values += [np.zeros((9,2))]
    dt +=     [('Gy_ini', np.float64, (9,9))]
    values += [np.zeros((9,9))]
@numba.njit(cache=True) 
def daesolver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 
    
    Dt = struct[i].Dt 

    N_x = struct[i].N_x
    N_y = struct[i].N_y

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


    solver = struct[i].solvern 
    while t<t_end: 
        struct[i].it += 1
        struct[i].t += Dt
        
        it = struct[i].it
        t = struct[i].t
        it_store = struct[i].it_store
        
        perturbations(t,struct) 
        
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
            struct[i].it_store += 1 
    struct[i].t = t
    struct[i].it_store = it_store
    return struct[i]['T'][:], struct[i].X[:]

@numba.njit(cache=True) 
def perturbations(t,struct): 
    if t>1.000000: struct[0].p_m = 0.600000



if __name__ == "__main__":
    sys = {'t_end': 20.0, 'Dt': 0.001, 'solver': 'trapezoidal', 'decimation': 10, 'name': 'smib_milano_ex8p1', 'models': [{'params': {'X_d': 1.81, 'X1d': 0.3, 'T1d0': 8.0, 'X_q': 1.76, 'X1q': 0.65, 'T1q0': 1.0, 'R_a': 0.003, 'X_l': 0.05, 'H': 3.5, 'D': 1.0, 'Omega_b': 314.1592653589793, 'omega_s': 1.0, 'v_0': 0.9008, 'theta_0': 0.0}, 'f': ['ddelta = Omega_b*(omega - omega_s)', 'domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))'], 'g': ['p_e@ i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q) - p_e', 'i_d@ v_q + R_a*i_q + X1d*i_d - e1q', 'i_q@ v_d + R_a*i_d - X1q*i_q', 'v_d@ v_1*sin(delta - theta_1) - v_d', 'v_q@ v_1*cos(delta - theta_1) - v_q', 'v_1@ P_t - (v_1*v_0*sin(theta_1 - theta_0))/X_l ', 'theta_1@ Q_t + (v_1*v_0*cos(theta_1 - theta_0))/X_l - v_1**2/X_l', 'P_t@ i_d*v_d + i_q*v_q - P_t', 'Q_t@ i_d*v_q - i_q*v_d - Q_t'], 'u': {'p_m': 0.2, 'e1q': 1.2}, 'u_ini': {'P_t': 0.5, 'Q_t': 0.1}, 'y_ini': ['p_e', 'i_d', 'i_q', 'v_d', 'v_q', 'v_1', 'theta_1', 'p_m', 'e1q'], 'h': ['omega']}], 'perturbations': [{'type': 'step', 'time': 1.0, 'var': 'p_m', 'final': 0.6}], 'itol': 1e-08, 'imax': 100, 'Dt_min': 0.001, 'Dt_max': 0.001, 'solvern': 3}
    syst =  smib_milano_ex8p1_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)