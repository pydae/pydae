'''
    File name: ofo.py
    Author: Álvaro Rodriguez del Nozal
    Date created: 05/07/2024

'''

import numpy as np

class ofo:
    '''
    Online Feedback Optmization

    minimize q'.u + u'.Q.u + r'.z + z'.R.z

    subject to: 
                -u < -u_min 
                 u <  u_max
                 C.z < bounds

    Dimesions:
    
    - q = [Nu,1] 
    - Q = [Nu,Nu]
    - r = [Nz,1]
    - R = [Nz,Nz]

    '''

    def __init__(self, in_bnds, C, d, alpha, rho, q, Q, r, R, model):
        self.in_bnds = in_bnds                  # list of tuples (min, max) with the bounds of the "u" elements  
        self.C = C                              # Output "z" constraints: Cz \leq d
        self.d = d
        self.lmb = np.zeros((self.C.shape[0],1))    # Lagrangian dual variables
        self.s = np.zeros((self.C.shape[0],1))      # Slack variables: Cz \leq d -> Cz - d + s = 0 with s \geq 0
        self.nin = len(in_bnds)                 # Array "u" dimension
        self.nout = C.shape[1]                  # Array "z" dimension
        self.alpha = alpha                      # Step size
        self.rho = rho                          # Augmented Lagrangian penalty term
        
        # The cost function has the shape: q'*u + u'*Q*u + r'*z + z'*R*z
        # Introduce a vector/matrix of zeros to the non used terms
        self.q = q
        self.Q = Q
        self.r = r
        self.R = R
        self.model = model
        self.u = np.zeros((q.shape[0],1))
        self.delta_u = 0.05
       
    def h(self, u):
        y = self.model.h_eval(u)
        return y
       
    # def h(self, u):
    #     u_bounds = self.model.u_bounds(u)
    #     y = self.model.h_eval(u_bounds)
    #     return y
    

    def compute_sens(self, delta_u):
        '''
        Method to compute the sensitivty matrix.
        From a perturbation vector "delta_u", the inputs to the system are
        perturbed one by one and the influence on the output vector is cuantified

        '''
        
        u = self.u
        self.H = np.array([-np.array(self.h(u)) for _ in u])
        for index_in in range(self.nin):
            u[index_in] += delta_u
            self.H[index_in, :] = (self.H[index_in, :] + self.model.h_eval(u))/delta_u # If the values in H are tiny, forget about divide by "delta_u"
            u[index_in] -= delta_u
        self.H = self.H.T[0]
        return self.H
           
    def run(self, niter, pr = False):
        # Define at the  beginning self.u and self.z
        update_H = 1 # Every "update_H" time steps the sensitivity matrix is updated
        self.compute_sens(self.delta_u)
        self.z = self.h(self.u)

        self.Z = np.zeros((niter,self.nout))
        self.U = np.zeros((niter,self.nin))


        for it in range(niter):
            if pr:
                print(f'Iteration {it+1}...')
            if it == update_H:
                self.compute_sens(self.delta_u) # Sensitivity matrix update
                update_H += 10

            ########################################### Step 1: Update slack variables
            self.s = -(1/self.rho)*self.lmb - self.C.dot(self.z) + self.d
            for index, item in enumerate(self.s):
                self.s[index] = np.clip(item, 0, np.inf)

            ########################################### Step 2: Update primal variables
            dL = self.q + 2*self.Q.T.dot(self.u) + self.H.T.dot(self.r) + 2*self.H.T.dot(self.R).dot(self.z) # dJ/du
            dL = dL + self.H.T.dot(self.C.T).dot(self.lmb + self.rho*(self.C.dot(self.z) - self.d + self.s)) # dL/du
            u_unconstrained = self.u - self.alpha*(dL)   
            for index, item in enumerate(self.u):
                self.u[index] = np.clip(u_unconstrained[index], self.in_bnds[index][0], self.in_bnds[index][1])
            # self.u = u_unconstrained.clip([item[0] for item in self.in_bnds], [item[1] for item in self.in_bnds])

            ########################################### Step 3: Dispacth new inputs "u" and gather outputs "z"
            self.z = self.h(self.u)

            ########################################### Step 4: Update dual variables
            self.lmb = self.lmb + self.rho*(self.C.dot(self.z) - self.d + self.s)  

            self.U[it,:] = self.u[:,0]
            self.Z[it,:] = self.z[:,0]

        return niter
           
    # def opf(self):
    #     opf_const = lambda u : - self.C.dot(self.h(u)) + self.d
    #     of = lambda u : self.qin.dot(u)
    #     cons = NonlinearConstraint(opf_const, 0, np.inf)
    #     sol = minimize(of,
    #                    self.u[-1],
    #                    bounds = self.in_bnds,
    #                    constraints = (cons,))        
    #     return sol