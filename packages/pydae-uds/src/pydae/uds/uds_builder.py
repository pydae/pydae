import numpy as np
import sympy as sym
import json
import hjson
import os
from pydae.core.builder.casadi_builder import MathBackend
from pydae.uds.genapes.genapes import add_genapes
from pydae.uds.vscs.vscs import add_vscs
from pydae.uds.loads.loads import add_loads
from pydae.uds.lines.lines import lines_preprocess,add_lines,add_line_monitors
from pydae.uds.transformers.transformers import trafos_preprocess,add_trafos,add_trafo_monitors
from pydae.uds.shunts.shunts import add_shunts,shunts_preprocess
from pydae.uds.sources.sources import add_sources
from pydae.uds.ess.ess import add_ess
from pydae.uds.miscellaneous.breaker import add_breakers
from pydae.uds.fcs.fcs import add_fcs

class UdsBuilder:
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

    def __init__(self,data_input='', use_casadi=False):

        if type(data_input) == str:
            if 'http' in data_input:
                url = data_input
                resp = requests.get(url)
                data = json.loads(resp.text)
            else:
                if os.path.splitext(data_input)[1] == '.json':
                    with open(data_input,'r', encoding='utf-8') as fobj:
                        data = json.loads(fobj.read().replace("'",'"'))
                if os.path.splitext(data_input)[1] == '.hjson':
                    with open(data_input,'r', encoding='utf-8') as fobj:
                        data = hjson.loads(fobj.read().replace("'",'"'))
        elif type(data_input) == dict:
            data = data_input

        self.data = data

        self.use_casadi = use_casadi
        self.backend = MathBackend(use_casadi)

        self.N_branches = 0
        self.preprocess()

        self.dae = {'f':[],'g':[],'x':[],'y_ini':[],'y_run':[],
                    'u_ini_dict':{},'u_run_dict':{},'params_dict':{},
                    'h_dict':{},'xy_0_dict':{}}

        self.aux = {}

        self.uz_jacs = False
        self.verbose = False

        self.omega_coi_numerator = 0.0
        self.omega_coi_denominator = 0.0

        self.A = self.backend.zeros(self.N_branches,self.N_nodes)
        self.At = self.backend.zeros(self.N_branches,self.N_nodes)

        self.G_primitive = self.backend.zeros(self.N_branches,self.N_branches)
        self.B_primitive = self.backend.zeros(self.N_branches,self.N_branches)

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
        self.S_base = self.backend.symbols("S_base")
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

        bk = self.backend

        self.G = (self.A.T @ self.G_primitive) @ self.A
        self.B = (self.A.T @ self.B_primitive) @ self.A


        # real and imaginary parts of the unknown node voltages.
        # CasADi SX has no complex type, so the complex nodal algebra
        # (G + jB)(V_r + jV_i) is carried out explicitly in real form.
        V_r_list = []
        V_i_list = []
        for item in self.nodes_list:
            name = item.replace('.','_')
            V_r_list += [bk.symbols(f'V_{name}_r')]
            V_i_list += [bk.symbols(f'V_{name}_i')]

        V_r = bk.Matrix([[v] for v in V_r_list])
        V_i = bk.Matrix([[v] for v in V_i_list])
        self.V_r = V_r
        self.V_i = V_i

        # line currents in real form: I = (G_p + jB_p) A (V_r + jV_i)
        A_V_r = self.A @ V_r
        A_V_i = self.A @ V_i
        self.I_lines_re = self.G_primitive @ A_V_r - self.B_primitive @ A_V_i
        self.I_lines_im = self.G_primitive @ A_V_i + self.B_primitive @ A_V_r

        if not self.use_casadi:
            # keep the complex voltage vector / line currents available for
            # components not yet ported to the real-form backend (e.g. some
            # transformer monitors). Only meaningful with the SymPy backend.
            V = sym.Matrix([V_r_list[k] + sym.I*V_i_list[k] for k in range(len(V_r_list))])
            self.V = V
            self.I_lines = (self.G_primitive + sym.I*self.B_primitive) @ self.A @ V

        self.it_branch = 0
        add_line_monitors(self)
        add_trafo_monitors(self)

        # main nodal equations in real form: (G + jB)(V_r + jV_i)
        g_re = self.G @ V_r - self.B @ V_i
        g_im = self.G @ V_i + self.B @ V_r
        self.g_re = g_re
        self.g_im = g_im

        # from complex equations to DAE g and y (interleaved re/im per node,
        # so node2idx() keeps targeting the correct equation)
        self.dae['g'] = []
        self.dae['y_ini']= []
        self.dae['y_run']= []
        for it in range(len(self.nodes_list)):
            self.dae['g'] += [g_re[it]]
            self.dae['g'] += [g_im[it]]

            self.dae['y_ini'] += [V_r_list[it]]
            self.dae['y_ini'] += [V_i_list[it]]
            self.dae['y_run'] += [V_r_list[it]]
            self.dae['y_run'] += [V_i_list[it]]


        # monitor voltages
        for bus in self.buses:
            if not 'monitor' in bus:
                bus.update({'monitor':False})  # do not monitor by default
            
            if bus['monitor']:
                n2a = {0:'a',1:'b',2:'c'}
                # phase top neutral voltages:
                name = bus['name']
                V_n_r = bk.symbols(f'V_{name}_{3}_r')
                V_n_i = bk.symbols(f'V_{name}_{3}_i')

                # phase-neutral voltage module
                for ph in [0,1,2]:
                    V_ph_r = bk.symbols(f'V_{name}_{ph}_r')
                    V_ph_i = bk.symbols(f'V_{name}_{ph}_i')
                    z_name = f'V_{name}_{n2a[ph]}n'
                    z_value = ((V_ph_r-V_n_r)**2 + (V_ph_i-V_n_i)**2)**0.5
                    self.dae['h_dict'].update({z_name:z_value})
                # neutral-ground voltage module
                z_name = f'V_{name}_ng'
                z_value = ((V_n_r)**2 + (V_n_i)**2)**0.5
                self.dae['h_dict'].update({z_name:z_value})

                # phase-phase voltage module
                for phj,phk in [(0,1),(1,2),(2,0)]:
                    V_phj_r = bk.symbols(f'V_{name}_{phj}_r')
                    V_phj_i = bk.symbols(f'V_{name}_{phj}_i')
                    V_phk_r = bk.symbols(f'V_{name}_{phk}_r')
                    V_phk_i = bk.symbols(f'V_{name}_{phk}_i')

                    z_name = f'V_{name}_{n2a[phj]}{n2a[phk]}'
                    z_value = ((V_phj_r-V_phk_r)**2 + (V_phj_i-V_phk_i)**2)**0.5
                    self.dae['h_dict'].update({z_name:z_value})               

    
        # voltages initial guess
        phi_deg_default = -30.0
        if 'phi_deg_default' in self.system:
            phi_deg_default = self.system['phi_deg_default']
        phi_default = np.deg2rad(phi_deg_default)


        for bus in self.buses:
            
            if not 'acdc' in bus:
                bus.update({'acdc':'AC'})
            
            if bus['acdc'] == 'AC':
                V_phg = bus['U_kV']*1000.0/np.sqrt(3)
            
                if 'phi_deg_0' in bus:
                    phi_0 = np.deg2rad(phi_default + bus['phi_deg_0'])
                else: phi_0 = np.deg2rad(phi_deg_default)

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
            
            if bus['acdc'] == 'DC':

                if not 'nodes' in bus:
                    bus.update({'nodes':[0,1]})

                V_dc = bus['U_kV']*1000.0

                node = bus['nodes'][0]
                name = f"{bus['name']}_{node}"
                self.dae['xy_0_dict'].update({f'V_{name}_r':V_dc/2,f'V_{name}_i':0.0})
                node = bus['nodes'][1]
                name = f"{bus['name']}_{node}"
                self.dae['xy_0_dict'].update({f'V_{name}_r':-V_dc/2,f'V_{name}_i':0.0})


    def node2idx(self,bus,phase):
        '''
        Function to obtain the indexes for real and imaginary equations 
        '''
        a2n = {'a':0,'b':1,'c':2,'n':3}
        idx_r = 2*self.nodes_list.index(f'{bus}.{a2n[phase]}')
        idx_i = idx_r+1

        return idx_r,idx_i


    def add_branch_monitor(self, name, branch_idx, magnitude=True):
        '''
        Emit real/imag (and optionally magnitude) current outputs for a single
        branch into h_dict, reading the backend-agnostic real-form line-current
        arrays. Shared by line and transformer monitors so both work on the
        SymPy and CasADi backends.
        '''
        i_re = self.I_lines_re[branch_idx]
        i_im = self.I_lines_im[branch_idx]
        self.dae['h_dict'][f"{name}_r"] = i_re
        self.dae['h_dict'][f"{name}_i"] = i_im
        if magnitude:
            self.dae['h_dict'][f"{name}_m"] = (i_re**2 + i_im**2)**0.5


    def construct(self, name):
        
        self.contruct_grid()          

        if 'loads' in self.data:
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
        if 'sources' in  self.data:
            add_sources(self)
        if 'pvs' in  self.data:
            for item in self.data['pvs']:
                add_pv(self)
        if 'ess' in  self.data:
            for item in self.data['ess']:
                add_ess(self,item)
        if 'breakers' in  self.data:
            for item in self.data['breakers']:
                add_breakers(self,item)   
        if 'fcs' in  self.data:
            for item in self.data['fcs']:
                add_fcs(self,item) 

        bk = self.backend

        # Center Of Inertia (COI)
        omega_coi = bk.symbols("omega_coi")

        # the COI accumulators stay numeric (0.0) only when no component
        # contributed; once a backend symbol has been added, comparing to 0.0
        # would evaluate as a symbolic expression (and SX can't be truthy).
        if isinstance(self.omega_coi_denominator, (int, float)) and self.omega_coi_denominator == 0.0:
            self.omega_coi_denominator = 1e-6
            self.omega_coi_numerator = 1e-6

        self.dae['g'] += [ -omega_coi + self.omega_coi_numerator/self.omega_coi_denominator]
        self.dae['y_ini'] += [ omega_coi]
        self.dae['y_run'] += [ omega_coi]
        self.dae['xy_0_dict'].update({'omega_coi':1.0})

        # secondary frequency control
        xi_freq = bk.symbols("xi_freq")
        p_agc = bk.symbols("p_agc")
        K_p_agc = bk.symbols("K_p_agc")
        K_i_agc = bk.symbols("K_i_agc")
        K_xif  = bk.symbols("K_xif")
        u_freq   = bk.symbols("u_freq")
        epsilon_freq = 1-omega_coi
        
        f_agc = [epsilon_freq - K_xif*xi_freq + u_freq]
        x_agc = [ xi_freq]

        g_agc = [ -p_agc + K_p_agc*epsilon_freq + K_i_agc*xi_freq ]
        y_agc = [  p_agc]
        

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

        sys_dict = {'name':name,'uz_jacs':self.uz_jacs,
                'params_dict':self.dae['params_dict'],
                'f_list':self.dae['f'],
                'g_list':self.dae['g'] ,
                'x_list':self.dae['x'],
                'y_ini_list':self.dae['y_ini'],
                'y_run_list':self.dae['y_run'],
                'u_run_dict':self.dae['u_run_dict'],
                'u_ini_dict':self.dae['u_ini_dict'],
                'h_dict':self.dae['h_dict'],
                'xy_0_dict':self.dae['xy_0_dict']}

        self.sys_dict = sys_dict

    def compile_numba(self, name):

        build_numba(self.sys_dict,verbose=self.verbose) 

    def compile_mkl(self, name):

        b = build_mkl(self.sys_dict,verbose=self.verbose)
        # b.sparse = True
        # b.mkl = True
        # b.uz_jacs = self.uz_jacs
        # b.dict2system()
        # b.functions()
        # b.jacobians()
        # b.cwrite()
        # b.template()
        # b.compile_mkl()  

    def build(self, name=''):
        if name == '':
            print('Error: name is not provided.')
        self.construct(name)    
        self.compile_numba(name)  

    def build_mkl(self, name=''):
        if name == '':
            print('Error: name is not provided.')
        self.construct(name)  
        self.uz_jacs = False  
        self.compile_mkl(name)  

    