# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:45:19 2020

@author: jmmau
"""
import numpy as np
from pydae.build import dic2sys




m = 1
G = 9.81
s= 1
sys = { 't_end':20.0,'Dt':0.01,'solver':'trapezoidal', 'decimation':10, 'name':'pendulum_dae',
   'models':[{'params':
                   {
                    'm':m,                       
                    'G':G,  
                    's':s,  
                   },
              'f':[
     'dx_pos = v',
     'dy_pos = w',
     'dv = (-2*x_pos*lam )/m',
     'dw = (-m*G - 2*y_pos*lam)/m',                
                   ],
              'g':[
'lam@x_pos**2 + y_pos**2 - s**2'
                      ],
              'u':{'p':0.0},
              'y':['lam'],
              'y_ini':['lam'],
              'h':[
                   'x_pos','y_pos','lam'
                   ]}
              ],
    'perturbations':[{'type':'step','time':1.0,'var':'p','final':0.9} ]
    }

x,f = dic2sys(sys)  




from pendulum_dae import pendulum_dae_class,run,daesolver



syst = pendulum_dae_class()
syst.struct.x[0,0] = 1
syst.struct.y[0,0] = m*(1+s*G)/2*s**2
syst.struct.solvern = 5
T,X = daesolver(syst.struct)
