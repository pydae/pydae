from pydae.build_v2 import builder
import numpy as np
import sympy as sym
from sympy.matrices.sparsetools import _doktocsr
from sympy import SparseMatrix

x_1,x_2,x_3 = sym.symbols('x_1,x_2,x_3', real=True)
y_1,y_2,y_3,y_4 = sym.symbols('y_1,y_2,y_3,y_4', real=True)
u_1,u_2 = sym.symbols('u_1,u_2', real=True)

p_f1,p_f2,p_f3,p_f4 = sym.symbols('p_f1,p_f2,p_f3,p_f4', real=True)
p_g1,p_g2,p_g3,p_g4 = sym.symbols('p_g1,p_g2,p_g3,p_g4', real=True)
p_u1,p_u2,p_u3,p_u4 = sym.symbols('p_u1,p_u2,p_u3,p_u4', real=True)
p_h1,p_h2,p_h3,p_h4 = sym.symbols('p_h1,p_h2,p_h3,p_h4', real=True)




x = sym.Matrix([[x_1],   
                [x_2],   
                [x_3]])

y = sym.Matrix([[y_1],   
                [y_2],   
                [y_3],   
                [y_4]])

u = sym.Matrix([[u_1],   
                [u_2]])


f = sym.Matrix([[-p_f1*x_1 + p_f4*y_1 + p_u1*u_1 + x_3*y_1], 
                [-p_f2*x_3], 
                [-p_f3*x_2 + u_2*y_1]])
                
g = sym.Matrix([[p_g1*y_2 + p_u2*u_1], 
                [p_g2*y_1 + x_2*y_1], 
                [p_g3*y_3 + x_3*y_2], 
                [p_u3*u_2 + u_1*y_4 + x_2*y_2]])

h = sym.Matrix([[p_h1*x_1 + p_h2*y_1 + p_h3*y_4 + p_h4*u_1], 
                [u_2*x_1 + x_1*y_2 + x_2*y_1]])

z = h

F_x = f.jacobian(x)
F_y = f.jacobian(y)
F_u = f.jacobian(u)

G_x = g.jacobian(x)
G_y = g.jacobian(y)
G_u = g.jacobian(u)

H_x = h.jacobian(x)
H_y = h.jacobian(y)
H_u = h.jacobian(u)

jac_ini_sym = sym.Matrix([[F_x,F_y],[G_x,G_y]]) 
#print(jac_ini_sym)
jac_ini_data,jac_ini_indices,jac_ini_indptr,jac_ini_shape = _doktocsr(SparseMatrix(jac_ini_sym))

jac_ini_sym = sym.Matrix([[F_x,F_y],[G_x,G_y]]) 
#print(jac_ini_sym)
jac_ini_data,jac_ini_indices,jac_ini_indptr,jac_ini_shape = _doktocsr(SparseMatrix(jac_ini_sym))

N_x = F_x.shape[0]
eye = sym.eye(N_x, real=True)
Dt = sym.Symbol('Dt',real=True)
jac_trap = sym.Matrix([[eye - 0.5*Dt*F_x, -0.5*Dt*F_y],[G_x,G_y]])    
print('jac_trap = ',jac_trap)
jac_trap_data,jac_trap_indices,jac_trap_indptr,jac_trap_shape = _doktocsr(SparseMatrix(jac_trap_sym))

print('F_x = ',F_x)
print('F_y = ',F_y)
print('F_u = ',F_u)

print('G_x = ',G_x)
print('G_y = ',G_y)
print('G_u = ',G_u)

print('H_x = ',H_x)
print('H_y = ',H_y)
print('H_u = ',H_u)


p_f1,p_f2,p_f3,p_f4= 1.0,2.0,3.0,4.0
p_g1,p_g2,p_g3,p_g4= 1.0,2.0,3.0,4.0
p_u1,p_u2,p_u3,p_u4= 1.0,2.0,3.0,4.0
p_h1,p_h2,p_h3,p_h4= 1.0,2.0,3.0,4.0

params_dict = {
'p_f1':p_f1,'p_f2':p_f2,'p_f3':p_f3,'p_f4':p_f4,
'p_g1':p_g1,'p_g2':p_g2,'p_g3':p_g3,'p_g4':p_g4,
'p_u1':p_u1,'p_u2':p_u2,'p_u3':p_u3,'p_u4':p_u4,
'p_h1':p_h1,'p_h2':p_h2,'p_h3':p_h3,'p_h4':p_h4,
}

u_1,u_2 = 1.0,2.0

u_dict = {
'u_1':1.0,'u_2':2.0
}

h_dict ={
'z_1':z[0],'z_2':z[1]
} 

sys_dict = {'name':'dae1','uz_jacs':True,
            'params_dict':params_dict,
            'f_list':list(f),
            'g_list':list(g),
            'x_list':list(x),
            'y_ini_list':list(y),
            'y_run_list':list(y),
            'u_run_dict':u_dict,
            'u_ini_dict':u_dict,
            'h_dict':h_dict}

b = builder(sys_dict)
b.dict2system()
b.functions()
b.jacobians()
b.cwrite()
b.compile()
b.template()

import dae1

model = dae1.model()
model.ini({},1)
#model.report_x()
#model.report_y()
#model.report_u()
#model.report_z()
#model.report_params()
#model.jac_ini

x_1,x_2,x_3 = model.get_mvalue(['x_1','x_2','x_3'])
y_1,y_2,y_3,y_4 = model.get_mvalue(['y_1','y_2','y_3','y_4'])

x = np.array([[x_1],   
              [x_2],   
              [x_3]]).reshape(3,)

y = np.array([[y_1],   
              [y_2],   
              [y_3],   
              [y_4]]).reshape(4,)

u = np.array([[u_1],   
              [u_2]]).reshape(2,)


p = np.array([p_f1,p_f2,p_f3,p_f4,
              p_g1,p_g2,p_g3,p_g4,
              p_u1,p_u2,p_u3,p_u4,
              p_h1,p_h2,p_h3,p_h4]).reshape(16,)

Dt = 1e-3

F_x =  np.array([[-p_f1, 0, y_1], [0, 0, -p_f2], [0, -p_f3, 0]])
F_y =  np.array([[p_f4 + x_3, 0, 0, 0], [0, 0, 0, 0], [u_2, 0, 0, 0]])
F_u =  np.array([[p_u1, 0], [0, 0], [0, y_1]])
G_x =  np.array([[0, 0, 0], [0, y_1, 0], [0, 0, y_2], [0, y_2, 0]])
G_y =  np.array([[0, p_g1, 0, 0], [p_g2 + x_2, 0, 0, 0], [0, x_3, p_g3, 0], [0, x_2, 0, u_1]])
G_u =  np.array([[p_u2, 0], [0, 0], [0, 0], [y_4, p_u3]])

jac_ini = np.array([[-p_f1, 0, y_1, p_f4 + x_3, 0, 0, 0], 
                    [0, 0, -p_f2, 0, 0, 0, 0], 
                    [0, -p_f3, 0, u_2, 0, 0, 0], 
                    [0, 0, 0, 0, p_g1, 0, 0], 
                    [0, y_1, 0, p_g2 + x_2, 0, 0, 0], 
                    [0, 0, y_2, 0, x_3, p_g3, 0], 
                    [0, y_2, 0, 0, x_2, 0, u_1]])


assert (np.max(np.abs(jac_ini-model.jac_ini))) < 1e-16

from dae1 import de_jac_ini_eval,sp_jac_ini_eval

N_x = F_x.shape[0]
N_y = G_y.shape[0]
N_xy = N_x+N_y
de_jac_ini = np.zeros((N_xy,N_xy))
de_jac_ini_eval(de_jac_ini,x,y,u,p,Dt)
#print(np.max(np.abs(jac_ini-de_jac_ini)))

from scipy.sparse import csr_array
from scipy.sparse import csr_matrix

sp_jac_ini = csr_matrix((np.zeros(len(jac_ini_data)),jac_ini_indices,jac_ini_indptr),shape=jac_ini_shape)
sp_jac_ini_eval(sp_jac_ini.data,x,y,u,p,Dt)

assert (np.max(np.abs(jac_ini-sp_jac_ini))) < 1e-16
#print(de_jac_ini)
#print(sp_jac_ini.toarray())
#print(sp_jac_ini.indices)
#print(sp_jac_ini.indptr)
#print(sp_jac_ini.nnz)
#
#print(jac_ini_indices,jac_ini_indptr)

jac_trap =  np.array([[0.5*Dt*p_f1 + 1, 0, -0.5*Dt*y_1, -0.5*Dt*(p_f4 + x_3), 0, 0, 0], 
                      [0, 1, 0.5*Dt*p_f2, 0, 0, 0, 0], 
                      [0, 0.5*Dt*p_f3, 1, -0.5*Dt*u_2, 0, 0, 0], 
                      [0, 0, 0, 0, p_g1, 0, 0], 
                      [0, y_1, 0, p_g2 + x_2, 0, 0, 0], 
                      [0, 0, y_2, 0, x_3, p_g3, 0], 
                      [0, y_2, 0, 0, x_2, 0, u_1]])