# pydae/core/common/parser.py
import logging

import sympy as sym


def check_system(sys):
    """
    Validates the user-provided system dictionary.
    Adds dummy equations if dynamic or algebraic equations are missing,
    checks for duplicates, and determines if a separate initialization 
    run ('inirun') is required.
    
    Parameters
    ----------
    sys : dict
        The raw system dictionary provided by the user.
        
    Returns
    -------
    sys : dict
        The validated and potentially patched system dictionary.
    inirun : bool
        Flag indicating if the initialization and run stages are different.
    """
    logging.debug('Checking system dictionary structure...')
    inirun = True

    # 1. Check for dynamic equations
    if len(sys.get('f_list', [])) == 0:
        logging.warning('System without dynamic equations. Adding dummy dynamic equation.')
        x_dummy, u_dummy = sym.symbols('x_dummy, u_dummy')
        sys.setdefault('x_list', []).append('x_dummy')
        sys['f_list'] = [u_dummy - x_dummy]
        sys.setdefault('u_ini_dict', {})['u_dummy'] = 1.0
        sys.setdefault('u_run_dict', {})['u_dummy'] = 1.0

    # 2. Check for algebraic equations
    if len(sys.get('g_list', [])) == 0:
        logging.warning('System without algebraic equations. Adding dummy algebraic equations.')
        y_dummy, u_dummy = sym.symbols('y_dummy, u_dummy')
        y_dummy2, u_dummy2 = sym.symbols('y_dummy2, u_dummy2')

        sys.setdefault('g_list', []).extend([u_dummy - y_dummy, u_dummy2 - y_dummy2])
        sys.setdefault('y_ini_list', []).extend(['y_dummy', 'y_dummy2'])
        sys.setdefault('y_run_list', []).extend(['y_dummy', 'y_dummy2'])

        sys.setdefault('u_ini_dict', {}).update({'u_dummy': 1.0, 'u_dummy2': 1.0})
        sys.setdefault('u_run_dict', {}).update({'u_dummy': 1.0, 'u_dummy2': 1.0})

    # 3. Check for duplicated variables
    if len(sys['y_ini_list']) != len(set(sys['y_ini_list'])):
        logging.error('Error: y_ini_list contains duplicate variables.')

    if len(sys['y_run_list']) != len(set(sys['y_run_list'])):
        logging.error('Error: y_run_list contains duplicate variables.')

    # 4. Check if initialization variables are identical to run variables
    if sys['y_run_list'] == sys['y_ini_list']:
        inirun = False

    return sys, inirun


def process_system_dict(sys):
    """
    Parses the raw user dictionary. It handles cases where the variables 
    in the lists are either strings OR already defined SymPy symbolic objects.
    It extracts the EXACT SymPy symbols from the expressions to ensure 
    derivatives compute correctly.
    """
    logging.debug('Parsing dictionary: converting lists to SymPy matrices and vectors')

    # 1. Collect all equations to extract their original symbols
    all_exprs = sys.get('f_list', []) + sys.get('g_list', [])
    if 'h_dict' in sys:
        all_exprs += list(sys['h_dict'].values())
    elif 'h_list' in sys:
        all_exprs += sys['h_list']

    # Dictionary with the exact symbols used in the expressions (keyed by their string name)
    exact_symbols = {}
    for expr in all_exprs:
        if hasattr(expr, 'free_symbols'):
            for s in expr.free_symbols:
                exact_symbols[s.name] = s

    def get_sym(item):
        """
        If the item is already a SymPy Symbol, return it directly.
        If it is a string, retrieve the exact symbol from the equations, 
        or create a new real symbol if it doesn't exist.
        """
        if isinstance(item, sym.Symbol):
            return item
        elif isinstance(item, str):
            return exact_symbols.get(item, sym.Symbol(item, real=True))
        else:
            return sym.Symbol(str(item), real=True)

    # 2. Convert equations to row matrices (.T)
    sys['f'] = sym.Matrix(sys['f_list']).T
    sys['g'] = sym.Matrix(sys['g_list']).T

    # 3. Build state and input vectors dynamically
    sys['x'] = sym.Matrix([get_sym(item) for item in sys['x_list']]).T
    sys['y_ini'] = sym.Matrix([get_sym(item) for item in sys['y_ini_list']]).T
    sys['y_run'] = sym.Matrix([get_sym(item) for item in sys['y_run_list']]).T

    sys['u_ini'] = sym.Matrix([get_sym(item) for item in sys['u_ini_dict'].keys()]).T
    sys['u_run'] = sym.Matrix([get_sym(item) for item in sys['u_run_dict'].keys()]).T

    # 4. Outputs (h)
    if 'h_dict' in sys:
        sys['h'] = sym.Matrix(list(sys['h_dict'].values())).T
    elif 'h_list' in sys:
        sys['h'] = sym.Matrix(sys['h_list']).T
    else:
        sys['h'] = sym.Matrix([get_sym(item) for item in sys['y_run_list']]).T

    # 5. Store dimensions
    sys['N_x'] = len(sys['x'])
    sys['N_y'] = len(sys['y_run'])
    sys['N_u'] = len(sys['u_run'])
    sys['N_z'] = len(sys['h'])

    return sys
