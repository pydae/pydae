import numpy as np


def ss_num2sym(name, A_num, B_num, C_num, D_num, backend=None):
    '''
    Converts numeric state space model to symbolic.

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
    backend : object, optional
        Math backend (SymPy or CasADi). If None, defaults to SymPy for
        backward compatibility.

    Returns
    -------
    dict
        Dictionary with the symbolic model: x, u, z, A, B, C, D, dx,
        z_evaluated, params_dict.
    '''
    import sympy as sym

    use_casadi = backend is not None and getattr(backend, 'use_casadi', False)

    N_x = B_num.shape[0]
    N_u = B_num.shape[1]
    N_z = C_num.shape[0]

    if use_casadi:
        import casadi as ca

        def make_sym(label):
            return ca.SX.sym(label)

        def zeros(r, c):
            return ca.SX.zeros(r, c)
    else:
        def make_sym(label):
            return sym.Symbol(label)

        def zeros(r, c):
            return sym.zeros(r, c)

    subfix = f'{name}' if name == '' else f'_{name}'

    u = zeros(N_u, 1)
    x = zeros(N_x, 1)
    z = zeros(N_z, 1)

    A = zeros(N_x, N_x)
    B = zeros(N_x, N_u)
    C = zeros(N_z, N_x)
    D = zeros(N_z, N_u)

    params = {}

    for row in range(N_u):
        con_str = f'u_{row}{subfix}'
        u[row, 0] = make_sym(con_str)

    for row in range(N_x):
        con_str = f'x_{row}{subfix}'
        x[row, 0] = make_sym(con_str)

    for row in range(N_x):
        for col in range(N_x):
            con_str = f'A_{row}{col}{subfix}'
            A[row, col] = make_sym(con_str)
            params.update({con_str: A_num[row, col]})

    for row in range(N_x):
        for col in range(N_u):
            con_str = f'B_{row}{col}{subfix}'
            B[row, col] = make_sym(con_str)
            params.update({con_str: B_num[row, col]})

    for row in range(N_z):
        for col in range(N_x):
            con_str = f'C_{row}{col}{subfix}'
            C[row, col] = make_sym(con_str)
            params.update({con_str: C_num[row, col]})

    for row in range(N_z):
        for col in range(N_u):
            con_str = f'D_{row}{col}{subfix}'
            D[row, col] = make_sym(con_str)
            params.update({con_str: D_num[row, col]})

    dx = A @ x + B @ u
    z_evaluated = C @ x + D @ u

    return {
        'x': x, 'u': u, 'z': z,
        'A': A, 'B': B, 'C': C, 'D': D,
        'dx': dx, 'z_evaluated': z_evaluated,
        'params_dict': params,
    }


if __name__ == "__main__":
    import sympy as sym

    A = np.array([[2.22745959, 2.89134367],
                  [-5.98640302, -5.13853719]])
    B = np.array([[-2.62892537],
                  [-3.35747765]])
    C = np.array([[-0.53183608, -0.28013641]])
    D = np.array([[0.]])

    sys = ss_num2sym('', A, B, C, D)

    p_ppc = sym.Symbol('p_ppc', real=True)

    sys['dx'] = sys['dx'].replace(sys['u'][0], p_ppc)
    f_list = list(sys['dx'])
    print(f_list)
