import numpy as np
import sympy as sym


def ss_num2sym(name,A_num,B_num,C_num,D_num):
    '''
    Converts numeric state space model to symbolic

    Parameters
    ----------
    name : string
        subfix of the parameters and variables of the system
    A_num : numpy array_like
        Matrix A.
    B_num : numpy array_like
        Matrix B.
    C_num : numpy array_like
        Matrix C.
    D_num : numpy array_like
        Matrix D.

    Returns
    -------

    describe : dict
        Dictionary with the symbolic model.


    Example
    -------

    >>> name = 'B1'
    >>> A_num = np.array([[1,2],[2.,3]])
    >>> B_num = np.array([[1],[3.]])
    >>> C_num = np.array([[1.,0]])
    >>> D_num = np.array([[0.]])

    
    
    '''
    N_x = B_num.shape[0]
    N_u = B_num.shape[1]
    N_z = C_num.shape[0]

    u = sym.Matrix.zeros(N_u,1)
    x = sym.Matrix.zeros(N_x,1)
    z = sym.Matrix.zeros(N_z,1)

    A = sym.Matrix.zeros(N_x,N_x)
    B = sym.Matrix.zeros(N_x,N_u)
    C = sym.Matrix.zeros(N_z,N_x)
    D = sym.Matrix.zeros(N_z,N_u)

    params = {}

    if name == '':
        subfix = f'{name}'
    else:
        subfix = f'_{name}'        

    for row in range(N_u):
        con_str = f'u_{row}{subfix}'
        u_i = sym.Symbol(con_str)
        u[row,0] = u_i

    for row in range(N_x):
        con_str = f'x_{row}{subfix}'
        x_i = sym.Symbol(con_str)
        x[row,0] = x_i

    for row in range(N_x):
        for col in range(N_x):
            con_str = f'A_{row}{col}{subfix}'
            A_ii = sym.Symbol(con_str)
            A[row,col] = A_ii
            params.update({f'A_{row}{col}{subfix}':A_num[row,col]})

    for row in range(N_x):
        for col in range(N_u):
            con_str = f'B_{row}{col}{subfix}'
            B_ii = sym.Symbol(con_str)
            B[row,col] = B_ii
            params.update({f'B_{row}{col}{subfix}':B_num[row,col]})

    for row in range(N_z):
        for col in range(N_x):
            con_str = f'C_{row}{col}{subfix}'
            C_ii = sym.Symbol(con_str)
            C[row,col] = C_ii
            params.update({f'C_{row}{col}{subfix}':C_num[row,col]})

    for row in range(N_z):
        for col in range(N_u):
            con_str = f'D_{row}{col}{subfix}'
            D_ii = sym.Symbol(con_str)
            D[row,col] = D_ii
            params.update({f'D_{row}{col}{subfix}':D_num[row,col]})

    dx = A @ x + B @ u
    z_evaluated  = C @ x + D @ u

    return {'x':x,'u':u,'z':z,
            'A':A,'B':B,'C':C,'D':D, 
            'dx':dx,'z_evaluated':z_evaluated,
            'params_dict':params}

if __name__ == "__main__":
    
    from pydae.utils.ss_num2sym import ss_num2sym

    A = np.array( [[ 2.22745959,  2.89134367],
                   [-5.98640302, -5.13853719]])
    B = np.array( [[-2.62892537],
                   [-3.35747765]])
    C = np.array( [[-0.53183608 ,-0.28013641]])
    D = np.array( [[0.]])

    sys = ss_num2sym('',A,B,C,D)

    p_ppc = sym.Symbol('p_ppc', real=True)

    sys['dx']= sys['dx'].replace(sys['u'][0],p_ppc)
    f_list = list(sys['dx'])
    print(f_list)
