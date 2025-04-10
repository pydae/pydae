# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:54:52 2020

@author: jmmau
"""

        #perturbations(t,struct) 
        
        # if solver == 1: 
        #     # forward euler solver  
        #     run(t,struct, 2)  
        #     struct[i].x[:] += Dt*struct[i].f  
 
        # if solver == 2: 
            
        #     # bacward euler solver
        #     x_0 = np.copy(struct[i].x[:]) 
        #     for j in range(struct[i].imax): 
        #         run(t,struct, 2) 
        #         run(t,struct, 3) 
        #         run(t,struct, 10)  
        #         phi =  x_0 + Dt*struct[i].f - struct[i].x 
        #         Dx = np.linalg.solve(-(Dt*struct[i].Fx - np.eye(N_x)), phi) 
        #         struct[i].x[:] += Dx[:] 
        #         if np.max(np.abs(Dx)) < struct[i].itol: break 
        #     print(struct[i].f)
 
        # if solver == 3: 
        #     # trapezoidal solver
        #     run(t,struct, 2) 
        #     f_0 = np.copy(struct[i].f[:]) 
        #     x_0 = np.copy(struct[i].x[:]) 
        #     for j in range(struct[i].imax): 
        #         run(t,struct, 10)  
        #         phi =  x_0 + 0.5*Dt*(f_0 + struct[i].f) - struct[i].x 
        #         Dx = np.linalg.solve(-(0.5*Dt*struct[i].Fx - np.eye(N_x)), phi) 
        #         struct[i].x[:] += Dx[:] 
        #         run(t,struct, 2) 
        #         if np.max(np.abs(Dx)) < struct[i].itol: break 

        # if solver == 4: # Teapezoidal DAE as in Milano's book

        #     run(t,struct, 2) 
        #     run(t,struct, 3) 

        #     x = np.copy(struct[i].x[:]) 
        #     y = np.copy(struct[i].y[:]) 
        #     f = np.copy(struct[i].f[:]) 
        #     g = np.copy(struct[i].g[:]) 
            
        #     for iter in range(struct[i].imax):
        #         run(t,struct, 2) 
        #         run(t,struct, 3) 
        #         run(t,struct,10) 
        #         run(t,struct,11) 
                
        #         x_i = struct[i].x[:] 
        #         y_i = struct[i].y[:]  
        #         f_i = struct[i].f[:] 
        #         g_i = struct[i].g[:]                 
        #         F_x_i = struct[i].Fx[:,:]
        #         F_y_i = struct[i].Fy[:,:] 
        #         G_x_i = struct[i].Gx[:,:] 
        #         G_y_i = struct[i].Gy[:,:]                

        #         A_c_i = np.vstack((np.hstack((eye-0.5*Dt*F_x_i, -0.5*Dt*F_y_i)),
        #                            np.hstack((G_x_i,         G_y_i))))
                     
        #         f_n_i = x_i - x - 0.5*Dt*(f_i+f) 
        #         # print(t,iter,g_i)
        #         Dxy_i = np.linalg.solve(-A_c_i,np.vstack((f_n_i,g_i))) 
                
        #         x_i = x_i + Dxy_i[0:N_x]
        #         y_i = y_i + Dxy_i[N_x:(N_x+N_y)]

        #         struct[i].x[:] = x_i
        #         struct[i].y[:] = y_i

        #         if np.max(np.abs(Dxy_i[:,0]))<struct[i].itol:
                    
        #             break
                
        #         # if iter>struct[i].imax-2:
                    
        #         #     print('Convergence problem')

        #     struct[i].x[:] = x_i
        #     struct[i].y[:] = y_i