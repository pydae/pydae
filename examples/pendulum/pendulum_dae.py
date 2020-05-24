import numpy as np
import numba
from pydae.nummath import interp


class pendulum_dae_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 3 
        self.imax = 100 
        self.N_x = 4 
        self.N_y = 1 
        self.N_store = 10000 
        self.x_list = ['x_pos', 'y_pos', 'v', 'w'] 
        self.y_list = ['lam'] 
        self.xy_list = self.x_list + self.y_list 
        self.y_ini_list = ['lam'] 
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
                ('m', np.float64),
                ('G', np.float64),
                ('s', np.float64),
                ('p', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (4,1)),
                    ('x', np.float64, (4,1)),
                    ('x_0', np.float64, (4,1)),
                    ('h', np.float64, (3,1)),
                    ('Fx', np.float64, (4,4)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,4)),
                    ('Y', np.float64, (self.N_store+1,1)),
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
                1,   # m 
                 9.81,   # G 
                 1,   # s 
                 0.0,   # p 
                 4,
                0,
                np.zeros((4,1)),
                np.zeros((4,1)),
                np.zeros((4,1)),
                np.zeros((3,1)),
                                np.zeros((4,4)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,4)),
                np.zeros((self.N_store+1,1)),
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

        dt +=     [('g', np.float64, (1,1))]
        values += [np.zeros((1,1))]
        dt +=     [('y', np.float64, (1,1))]
        values += [np.zeros((1,1))]
        dt +=     [('Fy', np.float64, (4,1))]
        values += [np.zeros((4,1))]
        dt +=     [('Gx', np.float64, (1,4))]
        values += [np.zeros((1,4))]
        dt +=     [('Gy', np.float64, (1,1))]
        values += [np.zeros((1,1))]




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
@numba.jit(nopython=True, cache=True)
def run(t,struct, mode):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    it = 0

    # parameters 
    m = struct[it].m
    G = struct[it].G
    s = struct[it].s

    # inputs 
    p = struct[it].p

    # states 
    x_pos  = struct[it].x[0,0] 
    y_pos  = struct[it].x[1,0] 
    v  = struct[it].x[2,0] 
    w  = struct[it].x[3,0] 


    # algebraic states 
    lam = struct[it].y[0,0] 


    if mode==2: # derivatives 

        dx_pos = v 
        dy_pos = w 
        dv = (-2*x_pos*lam )/m 
        dw = (-m*G - 2*y_pos*lam)/m 

        struct[it].f[0,0] = dx_pos   
        struct[it].f[1,0] = dy_pos   
        struct[it].f[2,0] = dv   
        struct[it].f[3,0] = dw   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = -s**2 + x_pos**2 + y_pos**2  

    if mode==4: # outputs 

        struct[it].h[0,0] = x_pos  
        struct[it].h[1,0] = y_pos  
        struct[it].h[2,0] = lam  
    

    if mode==10: # Fx 

        struct[it].Fx[0,2] = 1 
        struct[it].Fx[1,3] = 1 
        struct[it].Fx[2,0] = -2*lam/m 
        struct[it].Fx[3,1] = -2*lam/m 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[2,0] = -2*x_pos/m 
        struct[it].Fy[3,0] = -2*y_pos/m 
    

        struct[it].Gx[0,0] = 2*x_pos 
        struct[it].Gx[0,1] = 2*y_pos 
    



@numba.njit(cache=True)
def initialization(struct):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    m = struct[0].m 
    G = struct[0].G 
    s = struct[0].s 
    p = struct[0].p 
    x_pos = struct[0].x_ini[0,0] 
    y_pos = struct[0].x_ini[1,0] 
    v = struct[0].x_ini[2,0] 
    w = struct[0].x_ini[3,0] 
    lam = struct[0].y_ini[0,0] 
    struct[0].f_ini[0,0] = v 
    struct[0].f_ini[1,0] = w 
    struct[0].f_ini[2,0] = -2*lam*x_pos/m 
    struct[0].f_ini[3,0] = (-G*m - 2*lam*y_pos)/m 
    struct[0].g_ini[0,0]  = -s**2 + x_pos**2 + y_pos**2  
    struct[0].Fx_ini[0,2] = 1 
    struct[0].Fx_ini[1,3] = 1 
    struct[0].Fx_ini[2,0] = -2*lam/m 
    struct[0].Fx_ini[3,1] = -2*lam/m 
    struct[0].Fy_ini[2,0] = -2*x_pos/m 
    struct[0].Fy_ini[3,0] = -2*y_pos/m 
    struct[0].Gx_ini[0,0] = 2*x_pos 
    struct[0].Gx_ini[0,1] = 2*y_pos 


def ini_struct(dt,values):

    dt +=     [('x_ini', np.float64, (4,1))]
    values += [np.zeros((4,1))]
    dt +=     [('y_ini', np.float64, (1,1))]
    values += [np.zeros((1,1))]
    dt +=     [('f_ini', np.float64, (4,1))]
    values += [np.zeros((4,1))]
    dt +=     [('g_ini', np.float64, (1,1))]
    values += [np.zeros((1,1))]
    dt +=     [('Fx_ini', np.float64, (4,4))]
    values += [np.zeros((4,4))]
    dt +=     [('Fy_ini', np.float64, (4,1))]
    values += [np.zeros((4,1))]
    dt +=     [('Gx_ini', np.float64, (1,4))]
    values += [np.zeros((1,4))]
    dt +=     [('Gy_ini', np.float64, (1,1))]
    values += [np.zeros((1,1))]
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
    if t>1.000000: struct[0].p = 0.900000



if __name__ == "__main__":
    sys = {'t_end': 20.0, 'Dt': 0.01, 'solver': 'trapezoidal', 'decimation': 10, 'name': 'pendulum_dae', 'models': [{'params': {'m': 1, 'G': 9.81, 's': 1}, 'f': ['dx_pos = v', 'dy_pos = w', 'dv = (-2*x_pos*lam )/m', 'dw = (-m*G - 2*y_pos*lam)/m'], 'g': ['lam@x_pos**2 + y_pos**2 - s**2'], 'u': {'p': 0.0}, 'y': ['lam'], 'y_ini': ['lam'], 'h': ['x_pos', 'y_pos', 'lam']}], 'perturbations': [{'type': 'step', 'time': 1.0, 'var': 'p', 'final': 0.9}], 'itol': 1e-08, 'imax': 100, 'Dt_min': 0.001, 'Dt_max': 0.001, 'solvern': 3}
    syst =  pendulum_dae_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)