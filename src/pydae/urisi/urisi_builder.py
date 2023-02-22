import numpy as np
import sympy as sym
import json
import hjson
import os
from pydae.urisi.genapes.genapes import add_genapes
from pydae.urisi.vscs.vscs import add_vscs
from pydae.urisi.loads.loads import add_loads
from pydae.urisi.lines.lines import lines_preprocess,add_lines,add_line_monitors
from pydae.urisi.transformers.transformers import trafos_preprocess,add_trafos,add_trafo_monitors
from pydae.urisi.shunts.shunts import add_shunts,shunts_preprocess

import pydae.build_cffi as db

import requests


class urisi:
    '''
    

    Parameters
    ----------
    data_input : string or dict
        File path to the system data information or dictionary with the information.

    Returns
    -------
    dict
        Dictionary with the equations for pydae. 
        
    {
     'sys':{'name':'pf_1','S_base':100e6},       
     'buses':[{'name':'GRI','P_W':0.0,'Q_var':0.0,'U_kV':66.0, 'type':'slack'},
              {'name':'POI','P_W':0.0,'Q_var':0.0,'U_kV':66.0},
              {'name':'PMV','P_W':0.0,'Q_var':0.0,'U_kV':20.0}],
     'lines':[{'bus_j':'GRI','bus_k':'POI','X_km':0.4,'R_km':0.12,'km':20},
              {'bus_j':'POI','bus_k':'PMV','X_pu':0.04,'R_pu':0.01, 'S_mva':50.0}]
    }
        

    '''

    def __init__(self,data_input=''):
        
        if type(data_input) == str:
            if 'http' in data_input:
                url = data_input
                resp = requests.get(url)
                data = json.loads(resp.text)
            else:
                if os.path.splitext(data_input)[1] == '.json':
                    with open(data_input,'r') as fobj:
                        data = json.loads(fobj.read().replace("'",'"'))
                if os.path.splitext(data_input)[1] == '.hjson':
                    with open(data_input,'r') as fobj:
                        data = hjson.loads(fobj.read().replace("'",'"'))
        elif type(data_input) == dict:
            data = data_input
            
        self.data = data
        self.N_branches = 0
        self.preprocess()

        self.dae = {'f':[],'g':[],'x':[],'y_ini':[],'y_run':[],
                    'u_ini_dict':{},'u_run_dict':{},'params_dict':{},
                    'h_dict':{},'xy_0_dict':{}}

        self.uz_jacs = True     
        self.verbose = False 

        self.omega_coi_numerator = 0.0
        self.omega_coi_denominator = 0.0

        self.A = sym.zeros(self.N_branches,self.N_nodes)
        self.At = sym.zeros(self.N_branches,self.N_nodes)

        self.G_primitive = sym.zeros(self.N_branches,self.N_branches)
        self.B_primitive = sym.zeros(self.N_branches,self.N_branches)

    def preprocess(self):

        if not 'lines' in self.data:
            self.data['lines'] = []
        if not 'shunts' in self.data:
            self.data['shunts'] = []
        if not 'transformers' in self.data:
            self.data['transformers'] = []

        self.system = self.data['system']
        self.buses = self.data['buses']
        self.lines = self.data['lines']
        self.shunts = self.data['shunts']
        self.transformers = self.data['transformers']
        
    
        self.params_grid = {'S_base':self.system['S_base']}
        self.S_base = sym.Symbol("S_base", real=True) 
        self.N_bus = len(self.buses)

        self.nodes_list = []
        self.N_nodes = 0
        for bus in self.buses:
            if 'nodes' in bus:
                bus['N_nodes'] = len(bus['nodes'])
            elif 'N_nodes' in bus:
                bus['nodes'] = list(range(bus['N_nodes']))
            else:
                bus['N_nodes'] = 4
                bus['nodes'] = list(range(bus['N_nodes']))

            self.N_nodes += bus['N_nodes']

            for node in bus['nodes']:
                self.nodes_list += [f"{bus['name']}.{node}"]

        self.it_branch = 0 # current branch
        lines_preprocess(self)
        trafos_preprocess(self)
        shunts_preprocess(self)



        #self.N_branch = 3*len(self.lines) + len(self.shunts) + 2*len(self.transformers)


    def contruct_grid(self):

        self.it_branch = 0 # current branch

        add_lines(self)
        add_trafos(self)
        add_shunts(self)

        self.G = (self.A.T @ self.G_primitive) @ self.A
        self.B = (self.A.T @ self.B_primitive) @ self.A    
        
 
        # vector of unknown voltages 
        V_list = []
        for item in self.nodes_list:
            name = item.replace('.','_')
            V_r = sym.Symbol(f'V_{name}_r', real = True)
            V_i = sym.Symbol(f'V_{name}_i', real = True)
            V_list += [V_r + sym.I*V_i]

        V = sym.Matrix(V_list)
        self.V = V

        self.I_lines = (self.G_primitive + sym.I*self.B_primitive) @ self.A @ V

        self.it_branch = 0
        add_line_monitors(self)
        add_trafo_monitors(self)

        # main complex equations 
        g_cplx = (self.G + sym.I*self.B)@V
        self.g_cplx = g_cplx
        
        # from complex equations to DAE g and y
        self.dae['g'] = []
        self.dae['y_ini']= []
        self.dae['y_run']= []
        for it in range(len(g_cplx)):
            self.dae['g'] += [sym.re(g_cplx[it])]
            self.dae['g'] += [sym.im(g_cplx[it])]

            self.dae['y_ini'] += [sym.re(V[it])]
            self.dae['y_ini'] += [sym.im(V[it])]
            self.dae['y_run'] += [sym.re(V[it])]
            self.dae['y_run'] += [sym.im(V[it])]


        # monitor voltages
        for bus in self.buses:
            if not 'monitor' in bus:
                bus.update({'monitor':False})  # do not monitor by default
            
            if bus['monitor']:
                n2a = {0:'a',1:'b',2:'c'}
                # phase top neutral voltages:
                name = bus['name']
                V_n_r = sym.Symbol(f'V_{name}_{3}_r', real = True)
                V_n_i = sym.Symbol(f'V_{name}_{3}_i', real = True)

                # phase-neutral voltage module
                for ph in [0,1,2]:
                    V_ph_r = sym.Symbol(f'V_{name}_{ph}_r', real = True)
                    V_ph_i = sym.Symbol(f'V_{name}_{ph}_i', real = True)
                    z_name = f'V_{name}_{n2a[ph]}n'
                    z_value = ((V_ph_r-V_n_r)**2 + (V_ph_i-V_n_i)**2)**0.5
                    self.dae['h_dict'].update({z_name:z_value})
                # neutral-ground voltage module
                z_name = f'V_{name}_ng'
                z_value = ((V_n_r)**2 + (V_n_i)**2)**0.5
                self.dae['h_dict'].update({z_name:z_value})

                # phase-phase voltage module
                for phj,phk in [(0,1),(1,2),(2,0)]:
                    V_phj_r = sym.Symbol(f'V_{name}_{phj}_r', real = True)
                    V_phj_i = sym.Symbol(f'V_{name}_{phj}_i', real = True)
                    V_phk_r = sym.Symbol(f'V_{name}_{phk}_r', real = True)
                    V_phk_i = sym.Symbol(f'V_{name}_{phk}_i', real = True)

                    z_name = f'V_{name}_{n2a[phj]}{n2a[phk]}'
                    z_value = ((V_phj_r-V_phk_r)**2 + (V_phj_i-V_phk_i)**2)**0.5
                    self.dae['h_dict'].update({z_name:z_value})               

    
        # voltages initial guess
        for bus in self.buses:
            
            V_phg = bus['U_kV']*1000.0/np.sqrt(3)
            if 'phi_deg_0' in bus:
                phi_0 = np.deg2rad(-30 + bus['phi_deg_0'])
            else: phi_0 = np.deg2rad(-30)

            for node in [0,1,2,3]:
                name = f"{bus['name']}_{node}"
                if node == 0:
                    phi = phi_0
                    V_ini = V_phg*np.exp(1j*phi)
                if node == 1:
                    phi = phi_0 - 2/3*np.pi
                    V_ini = V_phg*np.exp(1j*phi)
                if node == 2:
                    phi = phi_0 + 2/3*np.pi
                    V_ini = V_phg*np.exp(1j*phi)
                if node == 3:
                    V_ini = 0.0
        
                self.dae['xy_0_dict'].update({f'V_{name}_r':V_ini.real,f'V_{name}_i':V_ini.imag})


    def node2idx(self,bus,phase):
        '''
        Function to obtain the indexes for real and imaginary equations 
        '''
        a2n = {'a':0,'b':1,'c':2,'n':3}
        idx_r = 2*self.nodes_list.index(f'{bus}.{a2n[phase]}')
        idx_i = idx_r+1

        return idx_r,idx_i


    def construct(self, name):
        
        self.contruct_grid()          

        add_loads(self)

        #if 'syns' in self.data:
        #    add_syns(self)
        if 'vscs' in self.data:
            add_vscs(self)
        #if 'vsgs' in self.data:
        #    add_vsgs(self)
        if 'genapes' in  self.data:
            add_genapes(self)
        #if 'wecs' in  self.data:
        #    add_wecs(self)
    
        omega_coi = sym.Symbol("omega_coi", real=True)  

        # if self.omega_coi_denominator <1e-6:
        #     self.omega_coi_denominator = 1e-6
        #     self.omega_coi_numerator = 1e-6

        self.dae['g'] += [ -omega_coi + self.omega_coi_numerator/self.omega_coi_denominator]
        self.dae['y_ini'] += [ omega_coi]
        self.dae['y_run'] += [ omega_coi]
        self.dae['xy_0_dict'].update({'omega_coi':1.0})

        # secondary frequency control
        xi_freq = sym.Symbol("xi_freq", real=True) 
        p_agc = sym.Symbol("p_agc", real=True)  
        K_p_agc = sym.Symbol("K_p_agc", real=True) 
        K_i_agc = sym.Symbol("K_i_agc", real=True) 
        K_xif  = sym.Symbol("K_xif", real=True)
        u_freq   = sym.Symbol("u_freq", real=True)
        epsilon_freq = 1-omega_coi
        g_agc = [ -p_agc + K_p_agc*epsilon_freq + K_i_agc*xi_freq ]
        y_agc = [  p_agc]
        x_agc = [ xi_freq]
        f_agc = [epsilon_freq - K_xif*xi_freq + u_freq]

        self.dae['g'] += g_agc
        self.dae['y_ini'] += y_agc
        self.dae['y_run'] += y_agc
        self.dae['f'] += f_agc
        self.dae['x'] += x_agc
        self.dae['params_dict'].update({'K_p_agc':self.system['K_p_agc'],'K_i_agc':self.system['K_i_agc']})
        self.dae['h_dict'].update({'xi_freq':xi_freq}) 
        self.dae['u_ini_dict'].update({'u_freq':0.0}) 
        self.dae['u_run_dict'].update({'u_freq':0.0}) 
        self.dae['h_dict'].update({'u_freq':u_freq}) 
        self.dae['h_dict'].update({'u_freq':u_freq}) 
        
        if 'K_xif' in self.system:
            self.dae['params_dict'].update({'K_xif':self.system['K_xif']})
        else:
            self.dae['params_dict'].update({'K_xif':0.0})
            
        with open('xy_0.json','w') as fobj:
            fobj.write(json.dumps(self.dae['xy_0_dict'],indent=4))


    def compile(self, name):

        sys_dict = {'name':name,'uz_jacs':self.uz_jacs,
                'params_dict':self.dae['params_dict'],
                'f_list':self.dae['f'],
                'g_list':self.dae['g'] ,
                'x_list':self.dae['x'],
                'y_ini_list':self.dae['y_ini'],
                'y_run_list':self.dae['y_run'],
                'u_run_dict':self.dae['u_run_dict'],
                'u_ini_dict':self.dae['u_ini_dict'],
                'h_dict':self.dae['h_dict']}

        self.sys_dict = sys_dict 
        
        bldr = db.builder(sys_dict,verbose=self.verbose);
        bldr.build()       


    def build(self, name=''):
        if name == '':
            print('Error: name is not provided.')
        self.construct(name)    
        self.compile(name)  