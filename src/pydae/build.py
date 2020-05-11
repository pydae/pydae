#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:53:28 2018

@author: jmmauricio
hola
"""
import numpy as np
import sympy as sym
import os
from collections import deque 
import pkgutil

def sym_gen_str():

    str = '''\
params = sys_vars['params']
u_list = sys_vars['u_list']
x_list = sys_vars['x_list']
y_list = sys_vars['y_list']
for item in params:
    exec(f"{item} = sym.Symbol('{item}')", globals())
for item in u_list:
    exec(f"{item} = sym.Symbol('{item}')", globals())
for item in x_list:
    exec(f"{item} = sym.Symbol('{item}')", globals())
for item in y_list:
    exec(f"{item} = sym.Symbol('{item}')", globals())  
sym_func_list = ['sin','cos']
for item in sym_func_list:
    exec(f"{item} = sym.{item}", globals()) 
    '''
    return str

def sym_gen(sys_vars):
    """Generates global sympy symbolic variables from a dictionary.
    
    Parameters
    ----------
    sys_vars : A ``dict`` with the following keys:
    
    params: ``dict``: Dictionary of the parameters and their values.
    u_list: ``list(str)``: List of input variables as strings for the running problem. 
    x_list: ``list(str)``: List of dynamical states as strings for the running problem.
    y_list: ``list(str)``: List of algebraic states as strings for the running problem.   

    R = 0.1
    L = 0.01
    T_pi = L/R
    K_p = L/0.01

    params= {'R' : 0.1, 'L':0.01, 'V_max':10.0, 'V_min':0.0, 'K_p':1.0, 'T_pi':0.01, 'K_aw':1.0}
    u_ini_dict = { 'i_ref':1.0}  # for the initialization problem
    u_run_dict = { 'i_ref':1.0}  # for the running problem (here initialization and running problem are the same)

    x_list = ['i','xi']
    y_ini_list = ['y','v']
    y_run_list = ['y','v']

    sys_vars = {'params':params,
                'u_list':u_run_dict,
                'x_list':x_list,
                'y_run_list':y_run_list}   
    """    

    params = sys_vars['params']
    u_list = sys_vars['u_list']
    x_list = sys_vars['x_list']
    y_list = sys_vars['y_list']

    for item in params:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())
    for item in u_list:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())
    for item in x_list:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())
    for item in y_list:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())  
    sym_func_list = ['sin','cos']
    for item in sym_func_list:
        exec(f"{item} = sym.{item}", globals()) 

    prueba_a = sym.Symbol('prueba_a')
    prueba_b = sym.Symbol('prueba_b')
    return [prueba_a,prueba_b]

    
        
def system(sys):
    y_ini = sym.Matrix(sys['y_ini'])
    y = sym.Matrix(sys['y'])
    x = sym.Matrix(sys['x'])
    u_ini = sym.Matrix(list(sys['u_ini_dict'].keys()))
    u = sym.Matrix(list(sys['u_run_dict'].keys()))

    f = sym.Matrix(sys['f'])
    g = sym.Matrix(sys['g'])
    g_ini = sym.Matrix(sys['g_ini'])
    
    h =  sym.Matrix(sys['h'])    

    Fx = f.jacobian(x)
    Fy = f.jacobian(y)
    Gx = g.jacobian(x)
    Gy = g.jacobian(y)

    Fx_ini = f.jacobian(x)
    Fy_ini = f.jacobian(y_ini)
    Gx_ini = g.jacobian(x)
    Gy_ini = g.jacobian(y_ini)

    sys['u_ini_dict'] = sys['u_ini_dict']
    sys['u_run_dict'] = sys['u_run_dict']

    sys['g']= g    
    sys['g_ini']= g_ini
    sys['f']= f
    sys['y']= y    
    sys['x']= x
    sys['g']= g    
    sys['g_ini']= g_ini
    sys['f']= f
    sys['y_ini']= y_ini    
    sys['x_ini']= x    
    sys['u_ini']= u_ini    
    sys['u']= u     
    sys['h']= h 
    
    sys['Fx'] = Fx
    sys['Fy'] = Fy
    sys['Gx'] = Gx
    sys['Gy'] = Gy

    sys['Fx_ini'] = Fx_ini
    sys['Fy_ini'] = Fy_ini
    sys['Gx_ini'] = Gx_ini
    sys['Gy_ini'] = Gy_ini
    
    return sys


def getIndex(s, i): 
  
    # If input is invalid. 
    if s[i] != '(':
        return -1
  
    # Create a deque to use it as a stack. 
    d = deque() 
  
    # Traverse through all elements 
    # starting from i. 
    for k in range(i, len(s)): 
  
        # Pop a starting bracket 
        # for every closing bracket 
        if s[k] == ')': 
            d.popleft() 
  
        # Push all starting brackets 
        elif s[k] == '(': 
            d.append(s[i]) 
  
        # If deque becomes empty 
        if not d: 
            return k 
  
    return -1
  
def arg2np(string,function_name):
    idx_end = 0
    for it in range(3):
        if function_name in string[idx_end:]:
            #print(string[idx_end:])
            idx_ini = string.find(f'{function_name}(',idx_end)+len(f'{function_name}(')
            idx_end = getIndex(string, idx_ini-1)
            string = string.replace(string[idx_ini:idx_end],f'np.array([{string[idx_ini:idx_end]}])')
    else: pass
    return string    

def sys2num(sys):
    
    params = sys['params']
    y_ini = sys['y_ini']
    y = sys['y']
    x = sys['x']
    u_ini = sys['u_ini']
    u_run = sys['u']

    f = sys['f']
    g = sys['g']
    g_ini = sys['g_ini']
    
    h =  sys['h']    

    Fx = sys['Fx']
    Fy = sys['Fy']
    Gx = sys['Gx']
    Gy = sys['Gy']

    Fx_ini = sys['Fx_ini']
    Fy_ini = sys['Fy_ini']
    Gx_ini = sys['Gx_ini']
    Gy_ini = sys['Gy_ini']
    
    
    N_x = len(x)
    N_y = len(y)
    N_z = len(h)

    N_params = len(params)
    N_inputs = len(u_run)

    name = sys['name']

    run_fun = ''
    ini_fun = ''

    numba_enable = True
    tab = '    '

    if numba_enable: run_fun += f'@numba.njit(cache=True)\n'
    run_fun += f'def run(t,struct,mode):\n\n'

    if numba_enable: ini_fun += f'@numba.njit(cache=True)\n'
    ini_fun += f'def ini(struct,mode):\n\n'

    run_fun += f'{tab}# Parameters:\n'
    ini_fun += f'{tab}# Parameters:\n'
    for item in params:
        run_fun += f'{tab}{item} = struct[0].{item}\n'
        ini_fun += f'{tab}{item} = struct[0].{item}\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    run_fun += f'{tab}# Inputs:\n'
    ini_fun += f'{tab}# Inputs:\n'
    for item in u_run:
        run_fun += f'{tab}{item} = struct[0].{item}\n'

    for item in u_ini:    
        ini_fun += f'{tab}{item} = struct[0].{item}\n'    
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    run_fun += f'{tab}# Dynamical states:\n'
    ini_fun += f'{tab}# Dynamical states:\n'    
    for irow in range(N_x):
        run_fun += f'{tab}{x[irow]} = struct[0].x[{irow},0]\n'
        ini_fun += f'{tab}{x[irow]} = struct[0].x_ini[{irow},0]\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'
    
    run_fun += f'{tab}# Algebraic states:\n'
    ini_fun += f'{tab}# Algebraic states:\n' 
    for irow in range(N_y):
        run_fun += f'{tab}{y[irow]} = struct[0].y[{irow},0]\n'
        ini_fun += f'{tab}{y_ini[irow]} = struct[0].y_ini[{irow},0]\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    run_fun += f'{tab}# Differential equations:\n'
    run_fun += f'{tab}if mode == 2:\n\n'
    run_fun += f'\n'
    ini_fun += f'{tab}# Differential equations:\n'
    ini_fun += f'{tab}if mode == 2:\n\n'
    ini_fun += f'\n'
    for irow in range(N_x):
        string = f'{f[irow]}'
        run_fun += f'{2*tab}struct[0].f[{irow},0] = {arg2np(string,"Piecewise")}\n'
        ini_fun += f'{2*tab}struct[0].f_ini[{irow},0] = {arg2np(string,"Piecewise")}\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    ## g
    run_fun += f'{tab}# Algebraic equations:\n'
    run_fun += f'{tab}if mode == 3:\n\n'
    run_fun += f'\n'
    ini_fun += f'{tab}# Algebraic equations:\n'
    ini_fun += f'{tab}if mode == 3:\n\n'
    ini_fun += f'\n'
    for irow in range(N_y):
        string = f'{g[irow]}'
        run_fun += f'{2*tab}struct[0].g[{irow},0] = {arg2np(string,"Piecewise")}\n'
        string = f'{g[irow]}'
        ini_fun += f'{2*tab}struct[0].g_ini[{irow},0] = {arg2np(string,"Piecewise")}\n'    
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    ## outputs
    run_fun += f'{tab}# Outputs:\n'
    run_fun += f'{tab}if mode == 3:\n'
    run_fun += f'\n'
    ini_fun += f'{tab}# Outputs:\n'
    ini_fun += f'{tab}if mode == 3:\n'
    ini_fun += f'\n'
    for irow in range(N_z):
        string = f'{h[irow]}'
        run_fun += f'{2*tab}struct[0].h[{irow},0] = {arg2np(string,"Piecewise")}\n'
        string = f'{h[irow]}'
        ini_fun += f'{2*tab}struct[0].h[{irow},0] = {arg2np(string,"Piecewise")}\n'    
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    run_fun += f'\n'
    run_fun += f'{tab}if mode == 10:\n\n'
    ini_fun += f'\n'
    ini_fun += f'{tab}if mode == 10:\n\n'
    for irow in range(N_x):
        for icol in range(N_x):
            if not Fx[irow,icol]==0:
                string = arg2np(f'{Fx[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Fx[{irow},{icol}] = {string}\n'
                ini_fun += f'{2*tab}struct[0].Fx_ini[{irow},{icol}] = {string}\n'

    run_fun += f'\n'
    run_fun += f'{tab}if mode == 11:\n\n'
    ini_fun += f'\n'
    ini_fun += f'{tab}if mode == 11:\n\n'
    for irow in range(N_x):
        for icol in range(N_y):
            if not Fy[irow,icol]==0:
                string = arg2np(f'{Fy[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Fy[{irow},{icol}] = {string}\n'
                string = arg2np(f'{Fy_ini[irow,icol]}',"Piecewise")
                ini_fun += f'{2*tab}struct[0].Fy_ini[{irow},{icol}] = {string} \n'

    run_fun += f'\n'
    ini_fun += f'\n'
    for irow in range(N_y):
        for icol in range(N_x):
            if not Gx[irow,icol]==0:
                string = arg2np(f'{Gx[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Gx[{irow},{icol}] = {string}\n'
                string = arg2np(f'{Gx[irow,icol]}',"Piecewise")
                ini_fun += f'{2*tab}struct[0].Gx_ini[{irow},{icol}] = {string}\n'

    run_fun += f'\n'
    ini_fun += f'\n'
    for irow in range(N_y):
        for icol in range(N_y):
            if not Gy[irow,icol]==0:
                string = f'{Gy[irow,icol]}'
                run_fun += f'{2*tab}struct[0].Gy[{irow},{icol}] = {arg2np(string,"Piecewise")}\n'
            if not Gy_ini[irow,icol]==0:
                string = f'{Gy_ini[irow,icol]}'
                ini_fun += f'{2*tab}struct[0].Gy_ini[{irow},{icol}] = {arg2np(string,"Piecewise")}\n'


    #with open('./class_dae_template.py','r') as fobj:
    #    class_template = fobj.read()
    class_template = pkgutil.get_data(__name__, "templates/class_dae_template.py").decode().replace('\r\n','\n') 
    functions_template = pkgutil.get_data(__name__, "templates/functions_template.py").decode().replace('\r\n','\n') 
    solver_template = pkgutil.get_data(__name__, "templates/solver_template_v2.py").decode().replace('\r\n','\n') 

    class_template = class_template.replace('{name}',str(name))
    class_template = class_template.replace('{N_x}',str(N_x)).replace('{N_y}',str(N_y)).replace('{N_z}',str(N_z))
    class_template = class_template.replace('{x_list}', str([str(item) for item in x]))
    class_template = class_template.replace('{y_list}', str([str(item) for item in y]))
    class_template = class_template.replace('{y_ini_list}', str([str(item) for item in y_ini]))
    class_template = class_template.replace('{params_list}', str([str(item) for item in params]))
    class_template = class_template.replace('{params_values_list}', str([(params[item]) for item in params]))
    class_template = class_template.replace('{inputs_ini_list}', str([str(item) for item in u_ini]))
    class_template = class_template.replace('{inputs_run_list}', str([str(item) for item in u_run]))
    class_template = class_template.replace('{inputs_ini_values_list}', str([(sys['u_ini_dict'][item]) for item in sys['u_ini_dict']]))
    class_template = class_template.replace('{inputs_run_values_list}', str([(sys['u_run_dict'][item]) for item in sys['u_run_dict']]))





    module = class_template
    module += '\n'*3
    module += run_fun
    module += '\n'*3
    module += ini_fun
    module += '\n'*3
    module += functions_template
    module += '\n'*3
    module += solver_template

    with open(f'{name}.py','w') as fobj:
        fobj.write(module)


#sys = {'name':'freq_1',
#       'params':[
#                {'T_g':[0.1, 0.6]},
#                {'H':[6.5, 0.6]},
#                {'D':[1.0, 0.6]},
#                {'Droop':[0.05, 0.6]},
#                {'p_l':[0.05, 0.6]}
#                ],
#       'equations':
#               ['dDf  = 1/(2*H)*(p_g - p_l - D*Df)',
#                'dp_g = 1/T_g*(-Df/Droop - p_g)'
#                ]
#       }
        


functions_np = ['sin','cos','sqrt']

def dic2sys(sys):
    
    sin = sym.sin
    cos = sym.cos
    pi = sym.pi
    sqrt = sym.sqrt
    
    tab4 = ' '*4
    solver_dict = {'forward-euler':1,
                   'backward-euler':2,
                   'trapezoidal':3,
                   'dae-backward-euler':4,
                   'dae-trapezoidal':5,
                   'dae-trap-varstep':7,
                   }
    system_params_float = ['t_end','Dt','decimation','itol','Dt_max','Dt_min']
    system_params_int   = [ 'solvern','imax']
        
        
    string_imports = ''
    string_imports += 'import numpy as np\n'
    string_imports += 'import numba\n'
    string_imports += 'from pydae.nummath import interp\n'


    defaults = {'t_end':1.0,
                'Dt':1.0e-3,'decimation':1,'itol':1.0e-8,'solver':'dae-trapezoidal',
                'imax':100,
                'name':'noname',
                'Dt_min':1.0e-3,
                'Dt_max':1.0e-3,                
                }
    
    for item in defaults:
        if item in sys: 
            continue
        else:
            sys.update({item:defaults[item]})
        
    ## symbolic model
    string_sym = ''
    string_x_sym = ''
    string_u_sym = ''
    string_f_sym = ''
    string_g_sym = ''
    string_y_sym = ''
    string_h_sym = ''
    x_sym_list = []
    f_sym_list = []
    x_list = []
    y_list = []
    y_ini_list = []
    
    algebraic_eqs = False

    for model in sys['models']:
        if 'params' in model:
            for item in model['params']:
                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(item.strip(),item.strip())
        if 'u' in model:
            for item in model['u']:
                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(item.strip(),item.strip())
                string_u_sym +=  '{:s}, '.format(item.strip())
        if 'g' in model:
            for item in model['g']:
                
                alg_state = item.split('@')[0] 

                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(alg_state.strip(),alg_state.strip())
 
                string_y_sym += '{:s}, '.format(alg_state.strip())   
                  
                
            algebraic_eqs = True  

            exec(string_sym)    
        istate = 0
        if 'f' in model:
            for item in model['f']:
                if item[0] == 'd':
                    state = item[1:].split('=')[0]
                    istate += 1
                    string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(state.strip(),state.strip())
                    string_x_sym += '{:s}, '.format(state.strip())   
                    string_f_sym += '{:s}, '.format(item[1:].split('=')[1]) 
                    
                    x_list += [state.strip()]
                    
        if 'g' in model:
            for item in model['g']:
                
                alg_state = item.split('@')[0] 
                alg_eq = item.split('@')[1]    
                 
                istate += 1
                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(alg_state.strip(),alg_state.strip())
 
   
                string_g_sym += '{:s}, '.format(alg_eq)                     
                algebraic_eqs = True  
                
                y_list += [alg_state.strip()]


    exec(string_sym)       
    string_f_sym.replace('sin','sym.sin')
    x = eval('sym.Matrix([{:s}])'.format(string_x_sym))
    f = eval('sym.Matrix([{:s}])'.format(string_f_sym))
    Fx = f.jacobian(x)
    sys_sym = {'x':x,'f':f,'Fx':Fx}
    if algebraic_eqs:
        y = eval('sym.Matrix([{:s}])'.format(string_y_sym))
        g = eval('sym.Matrix([{:s}])'.format(string_g_sym[:-2]))
        Fy = f.jacobian(y)
        Gx = g.jacobian(x)
        Gy = g.jacobian(y)
        N_y = len(y)
        sys_sym.update({'y':y,'g':g,'Fy':Fy,'Gx':Gx,'Gy':Gy})
    
   
    
    N_x = len(x)
    
    x_dict = {}
    istate = 0
    for x_ in x:
        x_dict.update({x_:istate})
        istate += 1
    
    
    
    ini_dict = sym.solve(f,x)
    
    string = ''  
    
    for model in sys['models']:
        if 'params' in model:
        
            string += '\n'
            string += '@numba.jit(nopython=True, cache=True)\n'
            string += 'def run(t,struct, mode):\n\n' 
            for item in functions_np:
                 string += tab4*1 +  '{:s} = np.{:s}\n'.format(item,item)                
            string += tab4 + 'it = 0\n\n' 
            string += tab4 + '# parameters \n'
            for item in model['params']:
                
                string += tab4
                string += '{:s} = struct[it].{:s}\n'.format(item.strip(),item.strip()) 

        ### inputs  
        if 'u' in model:
        
            string += '\n'
                  
            string += tab4 + '# inputs \n'
            for item in model['u']:
                string += tab4
                string += '{:s} = struct[it].{:s}\n'.format(item,item) 


        ### dyn states       
        if 'f' in model:                
            string += '\n'  
            string += tab4 + '# states \n'
            istate = 0
            for item in model['f']:
                if item[0] == 'd':
                    state = item[1:].split('=')[0]
                    string += tab4
                    string +=  '{:s} = struct[it].x[{:d},0] \n'.format(state,istate)
                    istate += 1
            string += '\n'

        ### alg states       
        if 'g' in model:                
            string += '\n'  
            string += tab4 + '# algebraic states \n'
            for y_i,y_idx in zip(y,range(N_y)):
                string += tab4
                string +=  '{:s} = struct[it].y[{:d},0] \n'.format(str(y_i),y_idx)
            string += '\n'


#        if 'f' in model:                
#            string += '\n' 
#            string += tab4 + 'if mode==1: # initialization \n\n' 
#            for item in ini_dict:
#                string += tab4*2
#                string += '{:s} = {:s}\n'.format(str(item), str(ini_dict[item]))
#
#            for item,istate in zip(ini_dict,range(N_x)):
#                string += tab4*2
#                string += 'struct[it].x[{:d},0] = {:s}  \n'.format(x_dict[item], str(item))
                  
            string += '\n' 
            string += tab4 + 'if mode==2: # derivatives \n\n' 
            for item in model['f']:
                if item[0] == 'd':
                    string += tab4*2
                    string += '{:s} \n'.format(item)
            string += '\n'  
            #string += tab4*2 + '# Derivatives \n'
            istate = 0
            for item in model['f']:
                if item[0] == 'd':
                    state = item[1:].split('=')[0]
                    string += tab4*2
                    string +=  'struct[it].f[{:d},0] = d{:s}  \n'.format(istate, state)
                    istate += 1

        if 'g' in model:   
            string += '\n' 
            string += tab4 + 'if mode==3: # algebraic equations \n\n' 
            for g_i,g_idx in zip(g,range(N_y)):
                string +=  tab4*2 + 'struct[it].g[{:d},0] = {:s}  \n'.format(g_idx,str(g_i))


        ### outputs       
        if 'h' in model:    
            ioutput = 0
            
            string += '\n'  
            string += tab4 + 'if mode==4: # outputs \n' 
            string += '\n'  
            #string += tab4*2 + '# Derivatives \n'
            for item in model['h']:

                string += tab4*2
                string +=  'struct[it].h[{:d},0] = {:s}  \n'.format(ioutput, item)
                ioutput += 1

    ## F_x
    string += tab4 + '\n'*2 
    string += tab4 + 'if mode==10: # Fx \n\n' 
    for irow in range(N_x):
        for icol in range(N_x):
            if str(Fx[irow,icol]) != '0':
                string += tab4*2 + 'struct[it].Fx[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Fx[irow,icol]))
                    
    ## F_y
    if 'g' in model:
        string += tab4 + '\n'*2 
        string += tab4 + 'if mode==11: # Fy,Gx,Gy \n\n' 
        for irow in range(N_x):
            for icol in range(N_y):
                if str(Fy[irow,icol]) != '0':
                    string += tab4*2 + 'struct[it].Fy[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Fy[irow,icol]))


    if 'g' in model:
        ## G_x
        string += tab4 + '\n'*2 
        for irow in range(N_y):
            for icol in range(N_x):
                if str(Gx[irow,icol]) != '0':
                    string += tab4*2 + 'struct[it].Gx[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Gx[irow,icol]))
                        
        ## G_y
        if 'g' in model:
            string += tab4 + '\n'*2 
            for irow in range(N_y):
                
                for icol in range(N_y):
                    if str(Gy[irow,icol]) != '0':
                        string += tab4*2 + 'struct[it].Gy[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Gy[irow,icol]))
                    

        
        
        
    string_class = '\n'*2
    string_class += 'class {:s}_class: \n'.format(sys['name'])

    string_class += tab4*1 + 'def __init__(self): \n\n'
    
    
    sys.update({'solvern': solver_dict[sys['solver']]})
    

    
    for item in system_params_float:            
        string_class += tab4*2 + "self.{:s} = {:f} \n".format(item.strip(),sys[item]) 
  
    for item in system_params_int:            
        string_class += tab4*2 + "self.{:s} = {:d} \n".format(item.strip(),sys[item])

    string_class += tab4*2 + 'self.N_x = {:d} \n'.format(N_x)
    string_class += tab4*2 + 'self.N_y = {:d} \n'.format(N_y)
    string_class += tab4*2 + 'self.N_store = {:d} \n'.format(10000)
    string_class += tab4*2 + 'self.x_list = {:s} \n'.format(str(x_list))
    string_class += tab4*2 + 'self.y_list = {:s} \n'.format(str(y_list))

    string_class += tab4*2 + 'self.xy_list = self.x_list + self.y_list \n'  
    
    if 'y_ini' in model:
        y_ini_list = []
        for item in model['y_ini']:
            y_ini_list += [item.strip()]
            
        string_class += tab4*2 + 'self.y_ini_list = {:s} \n'.format(str(y_ini_list))
        string_class += tab4*2 + 'self.xy_ini_list = self.x_list + self.y_ini_list \n'  

        
#    string_class += tab4*2 + 'self.y_list = ' + y

        
    string_class += tab4*2 + 'self.update() \n\n'
    
    
    string_class += tab4*1 + 'def update(self): \n\n'     

    string_class += tab4*2 + "self.N_steps = int(np.ceil(self.t_end/self.Dt)) \n"
    #string_class += tab4*2 + "self.N_store = int(np.ceil(self.N_steps/self.decimation))\n"
       
    string_class += tab4*2 +'dt =  [  \n'
    
    for item in system_params_float:            
        string_class += tab4*2 + "('{:s}', np.float64),\n".format(item.strip())
  
    for item in system_params_int:            
        string_class += tab4*2 + "('{:s}', np.int64),\n".format(item.strip())
        

    string_class += tab4*5 + "('N_steps', np.int64),\n" 
    string_class += tab4*5 + "('N_store', np.int64),\n" 
    for item in model['params']:
        #var = list(item.keys())[0]
        values = model['params'][item]
        string_class += tab4*4
        string_class += "('{:s}', np.float64),\n".format(item,item)
    for item in model['u']:
        #var = list(item.keys())[0]
        values = model['u'][item]
        string_class += tab4*4
        string_class += "('{:s}', np.float64),\n".format(item,item)
    string_class += tab4*5 + "('N_x', np.int64),\n"
    string_class += tab4*5 + "('idx', np.int64),\n" 
    string_class += tab4*5 + "('f', np.float64, ({:d},1)),\n".format(N_x) 
    string_class += tab4*5 + "('x', np.float64, ({:d},1)),\n".format(N_x) 
    string_class += tab4*5 + "('x_0', np.float64, ({:d},1)),\n".format(N_x) 
    string_class += tab4*5 + "('h', np.float64, ({:d},1)),\n".format(ioutput) 
    string_class += tab4*5 + "('Fx', np.float64, ({:d},{:d})),\n".format(N_x,N_x) 
    string_class += tab4*5 + "('T', np.float64, (self.N_store+1,{:d})),\n".format(1) 
    string_class += tab4*5 + "('X', np.float64, (self.N_store+1,{:d})),\n".format(N_x) 
    string_class += tab4*5 + "('Y', np.float64, (self.N_store+1,{:d})),\n".format(N_y) 

    string_class += tab4*4 + ']  \n\n'
    

    string_class += tab4*2 + 'values = [\n'
    
    
    
    
    N = 1
    for it in range(N):
 
        for item in system_params_float:            
            string_class += tab4*4 + "self.{:s},\n".format(item)
  
        for item in system_params_int:            
            string_class += tab4*4 + "self.{:s},\n".format(item)
            
        string_class += tab4*4 + "self.N_steps,\n" 
        string_class += tab4*4 + "self.N_store,\n" 
        for item in sys['models'][0]['params']:
            var = item
            values = sys['models'][0]['params'][item]
            string_class += tab4*4
            string_class += "{},   # {:s} \n ".format(values,var)
        for item in sys['models'][0]['u']:
            var = item
            values = sys['models'][0]['u'][item]
            string_class += tab4*4
            string_class += "{},   # {:s} \n ".format(values,var)        
        string_class += tab4*4
        string_class += "{:d},\n".format(N_x)    
        string_class += tab4*4
        string_class += "{:d},\n".format(N_x*it)   
        string_class += tab4*4
        string_class += "np.zeros(({:d},1)),\n".format(N_x)
        string_class += tab4*4
        string_class += "np.zeros(({:d},1)),\n".format(N_x)
        string_class += tab4*4
        string_class += "np.zeros(({:d},1)),\n".format(N_x)
        string_class += tab4*4
        string_class += "np.zeros(({:d},1)),\n".format(ioutput)
        string_class += tab4*4  
        string_class += tab4*4  + "np.zeros(({:d},{:d})),\n".format(N_x,N_x)
        string_class += tab4*4  +"np.zeros((self.N_store+1,{:d})),\n".format(1)
        string_class += tab4*4  +"np.zeros((self.N_store+1,{:d})),\n".format(N_x)
        string_class += tab4*4  +"np.zeros((self.N_store+1,{:d})),\n".format(N_y)
 
        string_class += tab4*4 + ']  \n'                

        string_class += tab4*2 + 'ini_struct(dt,values)\n\n'
        
        string_class += tab4*2 + "dt +=     [('t', np.float64)]\n"
        string_class += tab4*2 + "values += [0.0]\n"
        string_class += tab4*2 + "dt +=     [('it', np.int64)]\n"
        string_class += tab4*2 + "values += [0]\n"        
        string_class += tab4*2 + "dt +=     [('it_store', np.int64)]\n"
        string_class += tab4*2 + "values += [0]\n"         
    if 'g' in model:
        
        string_class += tab4*2 + "dt +=     [('N_y', np.int64)]\n"
        string_class += tab4*2 + "values += [self.N_y]\n\n"
        
        string_class += tab4*2 + "dt +=     [('g', np.float64, ({:d},1))]\n".format(N_y) 
        string_class += tab4*2 + "values += [np.zeros(({:d},1))]\n".format(N_y)   
          
        string_class += tab4*2 + "dt +=     [('y', np.float64, ({:d},1))]\n".format(N_y) 
        string_class += tab4*2 + "values += [np.zeros(({:d},1))]\n".format(N_y)   
          
        string_class += tab4*2 + "dt +=     [('Fy', np.float64, ({:d},{:d}))]\n".format(N_x,N_y) 
        string_class += tab4*2 + "values += [np.zeros(({:d},{:d}))]\n".format(N_x,N_y)
             
        string_class += tab4*2 + "dt +=     [('Gx', np.float64, ({:d},{:d}))]\n".format(N_y,N_x) 
        string_class += tab4*2 + "values += [np.zeros(({:d},{:d}))]\n".format(N_y,N_x)            
        
        string_class += tab4*2 + "dt +=     [('Gy', np.float64, ({:d},{:d}))]\n".format(N_y,N_y) 
        string_class += tab4*2 + "values += [np.zeros(({:d},{:d}))]\n".format(N_y,N_y)                



        
    string_class += '\n'*2 
    exec(string_sym) 


    
    import pkgutil
    data = pkgutil.get_data(__package__, 'solver_template_v2.py')    

    # with open('solver_template.py', 'r') as f:
    #     string_solver = f.read()
    
    string_solver =   str(data,'utf-8').replace('\r\n','\n') 


    
    
    string_perturbation = '\n'*2    
    string_perturbation += '@numba.njit(cache=True) \n'
    string_perturbation += 'def perturbations(t,struct): \n'
    if len(sys['perturbations'])==0:
            string_perturbation += tab4 + 'pass\n\n' 

    for item in sys['perturbations']:
        if item['type'] == 'step':

            string_perturbation += tab4 + 'if t>{:f}:'.format(item['time']) 
            string_perturbation += ' struct[0].{:s} = {:f}'.format(item['var'],item['final'])  
            string_perturbation += '\n'

        iinterp = 0
        if item['type'] == 'interp':
            iinterp += 1

            #string_class += tab4*2 + "Val_pert1 = np.array({:s}) \n".format(str(item['values']))
            string_class += tab4*2 + "dt += [('T_pert_{:d}', np.float64, ({:d}))] \n".format(iinterp, len(item['times']))
            string_class += tab4*2 + "values += [{:s}] \n".format( str(item['times']))
            string_class += tab4*2 + "dt += [('Values_pert_{:d}', np.float64, ({:d}))] \n".format(iinterp, len(item['times']))
            string_class += tab4*2 + "values += [{:s}] \n".format( str(item['values']))

            string_perturbation += tab4*1+ "struct[0].{:s} = interp(t,struct[0].T_pert_{:d},struct[0].Values_pert_{:d})".format(item['var'],iinterp,iinterp)  
            string_perturbation += '\n'   

        if item['type'] == 'func':

            string_perturbation += tab4*1+ "struct[0].{:s} = {:s}\n".format(item['var'],item['func'])  
            string_perturbation += '\n'  
            
    #string += tab4*2 + 'Nx = {:d} \n'.format(Nx)
    #string += tab4*2 + 'params[it].Fx = np.zeros((Nx,Nx)) \n\n'
    

    string_test = '\n'*2 

    string_test += '\n' + 'if __name__ == "__main__":'
    string_test += '\n' + tab4 +  'sys = {:s}'.format(str(sys))
    string_test += '\n' + tab4 + 'syst =  {:s}_class()'.format(sys['name'])
    string_test += '\n' + tab4 + 'T,X = solver(syst.struct)'    
    
#    string_test += '\n' + tab4 + 'import matplotlib.pyplot as plt'
#    
#    string_test += '\n' + tab4 + 'fig, (ax0,ax1) = plt.subplots(nrows=2)   # creates a figure with one axe'
#    string_test += '\n' + tab4 + 'ax0.plot(T,X[:,0])'
#    string_test += '\n' + tab4 + 'ax1.plot(T,X[:,1])'
#
#    string_test += '\n' + tab4 + "ax0.set_ylabel('x1')"
#    string_test += '\n' + tab4 + "ax1.set_ylabel('x2')"
#    string_test += '\n' + tab4 + "ax1.set_xlabel('Time (s)')"
#
#    string_test += '\n' + tab4 + 'plt.show()'

    
    string_test += '\n' + tab4 + "from scipy.optimize import fsolve"
    #string_test += '\n' + tab4 + "syst.struct[0].p_l = 0.1"
    string_test += '\n' + tab4 + "x0 = np.ones(syst.N_x+syst.N_y)"
    string_test += '\n' + tab4 + "s = fsolve(syst.ini_problem,x0 )"
    string_test += '\n' + tab4 + "print(s)"    
        
    string_class += '\n'*2 +  tab4*2 + "self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))\n\n"
        
    string_class += '\n' +  tab4*1 + "def ini_problem(self,x):"
    string_class += '\n' +  tab4*1 + "    self.struct[0].x_ini[:,0] = x[0:self.N_x]"
    string_class += '\n' +  tab4*1 + "    self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]"
    string_class += '\n' +  tab4*1 + "    initialization(self.struct)"
    string_class += '\n' +  tab4*1 + "    fg = np.vstack((self.struct[0].f_ini,self.struct[0].g_ini))[:,0]"
    string_class += '\n' +  tab4*1 + "    return fg"
    string_class += '\n'

    string_class += '\n' +  tab4*1 + "def run_problem(self,x):"
    string_class += '\n' +  tab4*2 + "t = self.struct[0].t"
    string_class += '\n' +  tab4*2 + "self.struct[0].x[:,0] = x[0:self.N_x]"
    string_class += '\n' +  tab4*2 + "self.struct[0].y[:,0] = x[self.N_x:(self.N_x+self.N_y)]"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,2)"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,3)"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,10)"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,11)"
    string_class += '\n' +  tab4*2 + "fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]"
    string_class += '\n' +  tab4*2 + "return fg\n\n"
    
    string_class += '\n' +  tab4*1 + "def ss_run(self):"
    string_class += '\n' +  tab4*2 + "t=0.0"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,2)"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,3)"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,10)"
    string_class += '\n' +  tab4*2 + "run(t,self.struct,11)"
    string_class += '\n' +  tab4*2 + "if np.max(np.abs(self.struct[0].f))>1e-4:"
    string_class += '\n' +  tab4*2 + "    print('the system is not in steady state!')\n"



        
    fobj = open(sys['name'] + '.py', 'w')
    fobj.write(string_imports)
    fobj.write(string_class)
    fobj.write(string)
    fobj.write(write_initialization(sys))
    fobj.write(string_solver)
    fobj.write(string_perturbation)
    fobj.write(string_test)

    fobj.close()
    
 
        #('Droop', np.float64),
        #                    ('K_pgov', np.float64),
        #                    ('K_igov', np.float64), 
        #                    ('T_b', np.float64), 
        #                    ('T_c', np.float64),
        #                    ('K_imw', np.float64),
        #                    ('f', np.float64, (3,1)),
        #                    ('x', np.float64, (3,1)),
        #                    ('p_out_ref', np.float64),
        #                    ('p_out_meas', np.float64),
        #                    ('speed', np.float64),
        #                    ('gov_ref', np.float64),
        #                    ('p_mech', np.float64),
        #                    ('Dt', np.float64) 
        #                   ]) 
        
        #im = 'ode_' + sys['name']
        #eval('    from {:s} import params'.format('ode_' + sys['name']))
    #    exec(string_d)
    #    print(string_d)
    #
    #
    #    return params    

    
    return x,f   

def write_initialization(sys):
    
    
    ## symbolic model

    string_sym = ''
    string_x_sym = ''
    string_u_sym = ''
    string_f_sym = ''
    string_g_sym = ''
    string_y_sym = ''
    string_h_sym = ''
    string_u_ini_sym = ''
    x_sym_list = []
    f_sym_list = []
    tab4 = ' '*4
    algebraic_eqs = False
    y_ini_list = []

    for model in sys['models']:
        if 'params' in model:
            for item in model['params']:
                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(item,item)
        if 'u' in model:
            for item in model['u']:
                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(item,item)
                string_u_sym += '{:s}, '.format(item.strip()) 
            exec(string_sym)
        if 'u_ini' in model:
            for item in model['u_ini']:
                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(item,item)
                string_u_ini_sym += '{:s}, '.format(item.strip()) 
                exec(string_sym)
        istate = 0
        if 'f' in model:
            for item in model['f']:
                if item[0] == 'd':
                    state = item[1:].split('=')[0]
                    istate += 1
                    string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(state.strip(),state.strip())
                    string_x_sym += '{:s}, '.format(state.strip())   
                    string_f_sym += '{:s}, '.format(item[1:].split('=')[1]) 
        if 'g' in model:
            for item in model['g']:
                
                alg_eq = item.split('@')[1]
 
#                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(alg_state.strip(),alg_state.strip())
                string_g_sym += '{:s}, '.format(alg_eq)                     
                algebraic_eqs = True     

        if 'y_ini' in model:
            for item in model['y_ini']:
                string_sym += '{:s} = sym.Symbol("{:s}")\n'.format(item.strip(),item.strip())
                string_y_sym += '{:s}, '.format(item)                     
                algebraic_eqs = True 
                y_ini_list += [item]
                
        print('y_ini_list')
        print(y_ini_list) 
    
    string_g_sym = string_g_sym.replace('sin(','sym.sin(')
    string_g_sym = string_g_sym.replace('cos(','sym.cos(')
    string_g_sym = string_g_sym.replace('pi','sym.pi')
    string_g_sym = string_g_sym.replace('sqrt(','sym.sqrt(')

    
    exec(string_sym)               
    x = eval('sym.Matrix([{:s}])'.format(string_x_sym))
    f = eval('sym.Matrix([{:s}])'.format(string_f_sym))
    u = eval('sym.Matrix([{:s}])'.format(string_u_sym)) 
    u_ini = eval('sym.Matrix([{:s}])'.format(string_u_ini_sym))     
    N_u = len(u)
    N_u_ini = len(u_ini)
    N_x = len(x)
    Fx = f.jacobian(x)
    sys_sym = {'x':x,'f':f,'Fx':Fx}
    
    
    
    if algebraic_eqs:
        y = eval('sym.Matrix([{:s}])'.format(string_y_sym))
        g = eval('sym.Matrix([{:s}])'.format(string_g_sym))
        Fy = f.jacobian(y)
        Gx = g.jacobian(x)
        Gy = g.jacobian(y)
        N_y = len(y)
        sys_sym.update({'y':y,'g':g,'Fy':Fy,'Gx':Gx,'Gy':Gy, 'u':u})
        
    struct_update = '\n\n'    
    string = '\n\n'
    
    struct_update += 'def ini_struct(dt,values):\n\n' 

    string += '@numba.njit(cache=True)\n'    
    string += 'def initialization(struct):\n\n'
    
    for item in functions_np:
         string += tab4*1 +  '{:s} = np.{:s}\n'.format(item,item)
        
        
    
    for param in model['params']:
        string += tab4*1 + '{:s} = struct[0].{:s} \n'.format(param,param)
       
    if 'u' in model:
        for irow in range(N_u):
            string += tab4*1 + '{:s} = struct[0].{:s} \n'.format(str(u[irow]),str(u[irow]))
             
    if 'u_ini' in model:
        for irow in range(N_u_ini):
            string += tab4*1 + '{:s} = struct[0].{:s} \n'.format(str(u_ini[irow]),str(u_ini[irow]))
            struct_update += tab4*1 + "dt += [('{:s}', np.float64)] \n".format(str(u_ini[irow]))
            struct_update += tab4*1 + "values += [{:f}] \n".format(model['u_ini'][str(u_ini[irow])])
            
    if 'f' in model:
        for irow in range(N_x):
            string += tab4*1 + '{:s} = struct[0].x_ini[{:d},0] \n'.format(str(x[irow]),irow)
        struct_update += tab4*1 + "dt +=     [('x_ini', np.float64, ({:d},1))]\n".format(N_x) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},1))]\n".format(N_x)   
          
            
    if 'g' in model:
        for irow in range(N_y):
            string += tab4*1 + '{:s} = struct[0].y_ini[{:d},0] \n'.format(str(y[irow]),irow)
        struct_update += tab4*1 + "dt +=     [('y_ini', np.float64, ({:d},1))]\n".format(N_y) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},1))]\n".format(N_y)   
        
    if 'f' in model:
        for irow in range(N_x):
            string += tab4*1 + 'struct[0].f_ini[{:d},0] = {:s} \n'.format(irow, str(f[irow]))
        struct_update += tab4*1 + "dt +=     [('f_ini', np.float64, ({:d},1))]\n".format(N_x) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},1))]\n".format(N_x)    
        
        
    if 'g' in model:
        for irow in range(N_y):
            string += tab4*1 + 'struct[0].g_ini[{:d},0]  = {:s}  \n'.format(irow, str(g[irow]))
        struct_update += tab4*1 + "dt +=     [('g_ini', np.float64, ({:d},1))]\n".format(N_y) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},1))]\n".format(N_y)   
        
    if 'f' in model:
        for irow in range(N_x):
            for icol in range(N_x):
                if str(Fx[irow,icol])!='0':
                    string += tab4*1 + 'struct[0].Fx_ini[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Fx[irow,icol]))
        struct_update += tab4*1 + "dt +=     [('Fx_ini', np.float64, ({:d},{:d}))]\n".format(N_x,N_x) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},{:d}))]\n".format(N_x,N_x)   
            
    if 'g' in model:
        for irow in range(N_x):
            for icol in range(N_y):
                if str(Fy[irow,icol])!='0':
                    string += tab4*1 + 'struct[0].Fy_ini[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Fy[irow,icol]))
        struct_update += tab4*1 + "dt +=     [('Fy_ini', np.float64, ({:d},{:d}))]\n".format(N_x,N_y) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},{:d}))]\n".format(N_x,N_y)  
        for irow in range(N_y):
            for icol in range(N_x):
                if str(Gx[irow,icol])!='0':
                
                    string += tab4*1 + 'struct[0].Gx_ini[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Gx[irow,icol]))
        struct_update += tab4*1 + "dt +=     [('Gx_ini', np.float64, ({:d},{:d}))]\n".format(N_y,N_x) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},{:d}))]\n".format(N_y,N_x)          
        
        for irow in range(N_y):
            for icol in range(N_y):
                if str(Gy[irow,icol])!='0':
                    string += tab4*1 + 'struct[0].Gy_ini[{:d},{:d}] = {:s} \n'.format(irow,icol,str(Gy[irow,icol]))       
        struct_update += tab4*1 + "dt +=     [('Gy_ini', np.float64, ({:d},{:d}))]\n".format(N_y,N_y) 
        struct_update += tab4*1 + "values += [np.zeros(({:d},{:d}))]\n".format(N_y,N_y)  
                
    return string + struct_update
    
    
if __name__ == "__main__":

    
    sys = { 't_end':20.0,'Dt':0.01,'solver':'forward-euler', 'decimation':10, 'name':'smib_1',
       'models':[{'params':
                       {'X_d' : 1.81,
                        'X1d'  : 0.3,
                        'T1d0'  : 8.0,
                        'X_q'  : 1.76,
                        'X1q'  : 0.65,
                        'T1q0'  : 1.0 ,
                        'R_a'  :  0.003, 
                        'X_l'  : 0.15 , 
                        'H'  : 3.5,   
                        'Omega_b' : 2*np.pi*60,
                        'B_t0':0.0,
                        'G_t_inf':0.0,
                        'T_r' : 0.05,
                        'theta_inf': 0.0,  
                        'T_r':0.05,  
                        'T_pss_1' : 1.281,
                        'T_pss_2' : 0.013,
                        'T_w' : 5.0,
                        'D':0.0,
                        'K_a':200.0,
                        'K_stab':0.5,
                        'B_t_inf':-1.0/(0.15+1.0/(1.0/0.5+1.0/0.93)),
                        'G_t0':0.01,
                        'V_inf':0.9008},
                  'f':[
                        'ddelta=Omega_b*(omega - 1)',
                        'domega = -(p_e - p_m + D*(omega - 1))/(2*H)',
                        'de1q =(v_f - e1q + i_d*(X1d - X_d))/T1d0',
                        'de1d = -(e1d + i_q*(X1q - X_q))/T1q0',
                        'dv_c =   (V_t - v_c)/T_r',
                        'dx_pss = (omega_w - x_pss)/T_pss_2',
                        'dx_w =  (omega - x_w)/T_w'
                       ],
                  'g':[  'i_d@    v_q - e1q + R_a*i_q + i_d*(X1d - X_l)',
                         'i_q@    v_d - e1d + R_a*i_d - i_q*(X1q - X_l)',
                         'p_e@   p_e - i_d*(v_d + R_a*i_d) - i_q*(v_q + R_a*i_q) ',
                         'v_d@    v_d - V_t*sin(delta - theta_t)   ',
                         'v_q@    v_q - V_t*cos(delta - theta_t)',
                         'P_t@    i_d*v_d - P_t + i_q*v_q   ',
                         'Q_t@ i_d*v_q - Q_t - i_q*v_d ',
                         'theta_t @(G_t0 + G_t_inf)*V_t**2 - V_inf*(G_t_inf*cos(theta_t - theta_inf) + B_t_inf*sin(theta_t - theta_inf))*V_t - P_t    ',
                         'V_t@ (- B_t0 - B_t_inf)*V_t**2 + V_inf*(B_t_inf*cos(theta_t - theta_inf) - G_t_inf*sin(theta_t - theta_inf))*V_t - Q_t',
                         'v_f@K_a*(V_ref - v_c + v_pss) - v_f        ',
                         'v_pss@v_pss + K_stab*(x_pss*(1/T_pss_2 - 1) - (T_pss_1*omega_w)/T_pss_2)    ',
                         'omega_w@ omega_w - omega + x_w '],
                  'u':{'p_m':0.9,'V_ref':1.0}, 
                  'u_ini':{},
                  'y_ini':[  'i_d',  'i_q',  'p_e',  'v_d', 'v_q', 'P_t', 'Q_t', 'theta_t','V_t' , 'v_f','v_pss', 'omega_w'],
                  'h':[
                       'omega'
                       ]}
                  ],
        'perturbations':[{'type':'step','time':1.0,'var':'V_ref','final':1.01} ]
        }
 
    sys = { 't_end':20.0,'Dt':0.01,'solver':'forward-euler', 'decimation':10, 'name':'vsg_cl',
       'models':[{'params':
                       {
                        'S_b':20.0e3,                       
                        'U_b':400.0,  
                        'F_b':50.0,   
                        'X': 0.3,
                        'R': 0.1,
                        'H'  : 1.0,
                        'D'  : 10.0,
                        'theta_0' : 0.0,
                        'e_0': 1.0,
                        'X_l': 0.1,
                        'R_l': 0.0,
                        'V_inf':1.0,
                        'theta_inf':0.0,
                        'Omega_b':2*np.pi*50.0,
                       },
                  'f':[
    'ddelta  = Omega_b*(omega-omega_ref)',
    'domega  = 1/(2*H)*(p_m - p_e - D*(omega-omega_ref))',
    'dtheta_p  = Omega_b*omega_ref',
    'dxi_q  = eps_q'
                       ],
                  'g':[
    'v_sd @ -v_d + cos(delta)*v_sd + cos(delta - pi/2)*v_sq',
    'v_sq @ -v_q - sin(delta)*v_sd - sin(delta - pi/2)*v_sq',
    'i_d @ -i_d + cos(delta)*i_sd  + cos(delta - pi/2)*i_sq',
    'i_q @ -i_q - sin(delta)*i_sd  - sin(delta - pi/2)*i_sq',
    'p_e @ -p_e + (v_d*i_d + v_q*i_q)',
    'p_s @ -p_s + (v_d*i_d + v_q*i_q)',
    'q_s @ -q_s + (v_d*i_q - v_q*i_d)',
    'eps_q @ - eps_q + q_ref - q_s',
    'e @  -e + e_0 + 1*eps_q + 5 *xi_q',
    'i_sd @ v_sd - R_l*i_sd + X_l*i_sq - V_inf*cos(theta_inf)',
    'i_sq @ v_sd - R_l*i_sd + X_l*i_sq + V_inf*sin(theta_inf)',                 
    'v_d @ -v_d + e - R*i_d + X * i_q',
    'v_q @ -v_q     - R*i_q - X * i_d'
                          ],
                  'u':{'p_m':0.8,'q_ref':0.1,'omega_ref':1.0},
                  'y':['v_sd','v_sq','i_sd','i_sq','v_d','v_q','i_d','i_q','p_e','p_s','q_s','eps_q','e'],
                  'y_ini':['v_sd','v_sq','i_sd','i_sq','v_d','v_q','i_d','i_q','p_e','p_s','q_s','eps_q','e'],
                  'h':[
                       'omega'
                       ]}
                  ],
        'perturbations':[{'type':'step','time':1.0,'var':'p_m_0','final':0.9} ]
        }
    
    x,f = dic2sys(sys)  ;

           
#    sys = { 't_end':20.0,'Dt':0.01,'solver':'forward-euler', 'decimation':10,
#       'models':[{'params':
#                      {'T_g':2.0, 'H':6.5, 'D':1.0, 'Droop':0.05},
#                  'f':[
#                      'dDf  = 1/(2*H)*(p_g + p_nc - p_l - D*Df)',
#                      'dp_g = 1/T_g*(p_c - Df/Droop - p_g)'
#                       ],
#                  'g':['p_c@p_c_0-p_c'],
#                  'u':{'p_l':0.0, 'p_nc':0.0, 'p_c_0':0.0}, 
#                  'u_ini':{'p_c':0.3},
#                  'y_ini':['p_c_0'],
#                  'h':[
#                       'Df'
#                       ]}
#                  ],
#        'perturbations':[{'type':'step','time':1.0,'var':'p_l','final':0.1},
#                         {'type':'interp','times':[0,1,2,3,4,5,6,7,8],'values':[0,2,1,3,2,5,2,7,1],'var':'p_nc'},
#                         {'type':'func','func':'np.sin(10*t)','var':'p_l'}]
#        }
#        
#    x,f = system(sys)
#
#    
#    from ode_freq_1 import freq_1, freq_1_class, solver
#    fr1 = freq_1_class()
#    x1 = np.array(np.arange(0,10), dtype = [('x1', float)])
#
#    freq_1(0.0,fr1.struct,2)
#    freq_1(0.0,fr1.struct,3)
#    freq_1(0.0,fr1.struct,10)
#    T,X = solver(fr1.struct)
#    
    
    
    
    
    
    
    
#    sys = { 't_end':20.0,'Dt':0.01,'solver':'forward-euler', 'decimation':10,
#           'models':[{'params':
#                          {'T_g':2.0, 'H':6.5, 'D':1.0, 'Droop':0.05},
#                      'f':[
#                          'dDf  = 1/(2*H)*(p_g + p_nc - p_l - D*Df)',
#                          'dp_g = 1/T_g*(p_c - Df/Droop - p_g)'
#                           ],
#                      'g':['p_c@p_c_0-p_c&p_c_0|p_c=p_l'],
#                      'u':{'p_l':0.0, 'p_nc':0.0, 'p_c':0.0},                          
#                      'h':[
#                           'Df'
#                           ]}
#                      ],
#            'perturbations':[{'type':'step','time':1.0,'var':'p_l','final':0.1},
#                             {'type':'interp','times':[0,1,2,3,4,5,6,7,8],'values':[0,2,1,3,2,5,2,7,1],'var':'p_nc'},
#                             {'type':'func','func':'np.sin(10*t)','var':'p_l'}]
#            }
#        
#    x,f = system(sys)
#
#    
#    from ode_freq_1 import freq_1, freq_1_class, solver
#    fr1 = freq_1_class()
#    x1 = np.array(np.arange(0,10), dtype = [('x1', float)])
#
#    freq_1(0.0,fr1.struct,2)
#    freq_1(0.0,fr1.struct,3)
#    freq_1(0.0,fr1.struct,10)
#    T,X = solver(fr1.struct)
    


