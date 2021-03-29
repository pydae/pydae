# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 23:36:15 2020

@author: jmmau
"""

import numpy as np
import pandas as pd 
import numba
import scipy.optimize as sopt
class yoko:
    
    def __init__(self):
        
        self.names = ['error','i_a','v_a','i_b','v_b','i_c','v_c']
        self.t_offset = 0.0
        self.Dt_pll = 0.04
        
    def read(self,file_path):  
        
        
        with open(file_path,'r') as fobj:
            lines = fobj.readlines()

        self.meta_data = {}
        for it in range(14):
            parameter_value = lines[it].split(',')
            key = parameter_value[0].replace('"','').strip()
            value = parameter_value[1].replace('"','').replace('\n','').strip()
            #print(value)
            if value.isdecimal(): value = float(value)
            try: 
                value = float(value)
            except ValueError:
                pass

            self.meta_data.update({key:value})
        df_exp = pd.read_csv(file_path,skiprows=14,header=0,names=self.names,index_col=False)

        time_interval = self.meta_data['BlockSize']*self.meta_data['HResolution']
        df_exp['Time'] = np.linspace(self.t_offset,self.t_offset+time_interval,int(self.meta_data['BlockSize']))
        df_exp.drop('error',axis='columns', inplace=True)
        self.df  = df_exp

    def abc2pq(self,omega=2*np.pi*50,theta_0=0.0):

        times = self.df.Time.values 
        freq = 50.0
        omega = 2*np.pi*freq
        v_a = self.df.v_a.values
        v_b = self.df.v_b.values
        v_c = self.df.v_c.values
        i_a = self.df.i_a.values
        i_b = self.df.i_b.values
        i_c = self.df.i_c.values




        Dt = times[1]-times[0] 
        p = times*0.0
        q = times*0.0
        for it in range(len(times)):

            theta = Dt*it*omega + theta_0
            v_abc = np.array([[v_a[it]],[v_b[it]],[v_c[it]]])
            T_p = 2.0/3.0*np.array([[ np.cos(theta), np.cos(theta-2.0/3.0*np.pi), np.cos(theta+2.0/3.0*np.pi)],
                                    [-np.sin(theta),-np.sin(theta-2.0/3.0*np.pi),-np.sin(theta+2.0/3.0*np.pi)]])

            dq=T_p@v_abc;

            v_d = dq[0]
            v_q = dq[1]

            theta = Dt*it*omega + theta_0
            i_abc = np.array([[i_a[it]],[i_b[it]],[i_c[it]]])
            T_p = 2.0/3.0*np.array([[ np.cos(theta), np.cos(theta-2.0/3.0*np.pi), np.cos(theta+2.0/3.0*np.pi)],
                                    [-np.sin(theta),-np.sin(theta-2.0/3.0*np.pi),-np.sin(theta+2.0/3.0*np.pi)]])

            i_dq=T_p@i_abc;

            i_d = i_dq[0]
            i_q = i_dq[1]

            p[it] = 3/2*(v_d*i_d + v_q*i_q)
            q[it] = 3/2*(v_d*i_q - v_q*i_d)
            
        self.p = p
        self.q = q


    def pll(self):
        
        df_time = self.df['Time'].values
        times = np.linspace(df_time[0],df_time[-1]-self.Dt_pll,1000)
        theta = np.zeros(len(times))
        omega = np.zeros(len(times))
        it = 0
        a = np.max(self.df['v_a'].values)
        b = 2*np.pi*50
        c = 0.0
        d = 0.0
        for t in times:
            t_0 = t
            idx_1 = np.searchsorted(self.df['Time'],t_0)
            idx_2 = np.searchsorted(self.df['Time'],t_0 + self.Dt_pll)
            decimation = 10
            xdata = self.df['Time'][idx_1:idx_2:decimation].values
            ydata = self.df['v_a'][idx_1:idx_2:decimation].values
    
    
            popt, pcov = sopt.curve_fit(func, xdata, ydata, p0=[a,b,c,d])
            a,b,c,d = popt
            theta[it] = b*t+c
            omega[it] = b
            it += 1
    
        self.df['theta_pll'] = np.interp(df_time,times,theta)
        self.df['omega_pll'] = np.interp(df_time,times,omega)



@numba.njit(cache=True)
def func(x, a, b, c, d):
    y = np.copy(x)
    y = a * np.sin(b * x + c) + d
    return y

@numba.njit(cache=True)
def abc2dq(times,a,b,c,Theta):
    N = len(times)
    d = np.zeros((N,1))
    q = np.zeros((N,1))
    for it in range(len(times)):

        abc = np.array([[a[it]],[b[it]],[c[it]]])
        theta = Theta[it]
        T_p = 2.0/3.0*np.array([[ np.cos(theta), np.cos(theta-2.0/3.0*np.pi), np.cos(theta+2.0/3.0*np.pi)],
                                [-np.sin(theta),-np.sin(theta-2.0/3.0*np.pi),-np.sin(theta+2.0/3.0*np.pi)]])

        dq=T_p@abc;
        
        d[it] = dq[0]
        q[it] = dq[1]
    
    return d,q
