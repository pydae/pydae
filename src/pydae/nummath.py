import numpy as np
import numba



@numba.njit(cache=True) 
def interp(t,T,X):
    
    if t<=T[-1]:
        t_idx = np.argmax(T>t)    
        t_1 = T[t_idx-1] 
        t_2 = T[t_idx] 
        x_1 = X[t_idx-1] 
        x_2 = X[t_idx] 
    
        x = (x_2-x_1)/(t_2-t_1)*(t-t_1)+x_1
    else:
        x = X[-1]
    return x