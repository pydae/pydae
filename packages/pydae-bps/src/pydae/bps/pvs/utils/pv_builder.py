import numpy as np
import sympy as sym
import pydae.build_cffi as db
import json 
import hjson
import requests

class pv_model:

    def __init__(self,name='pv'):

        self.name = name

        self.I_sc = 3.87
        self.V_oc = 42.1
        self.I_mpp = 3.56
        self.V_mpp = 33.7
        self.N_s = 72
        self.K_vt = -0.160
        self.K_it = 0.065
        self.R_pv_s = 0.5602
        self.R_pv_sh = 1862
        self.K_d = 1.3433

    def model_params(self,data_input):
        '''
        Input
        =====

        data_input: dict
            {"I_sc":3.87,"V_oc":42.1,"I_mpp":3.56,"V_mpp":33.7,"N_s":72,
            "K_vt":-0.160,"K_it":0.065,"R_s": 0.5602, "R_sh": 1862, "K_d": 1.3433}
                       
            I_sc: short circuit current # A
            I_mpp: maximum power point current # A
            V_mpp: maximum power point voltage # V
            V_oc:  open circuit voltage # V
            N_s: number of cells in series
            P_mpp: 120  # W 
            K_vt = -0.160 # Temperature Coefficient of V_oc %/Cº
            K_it = 0.065  # Temperature Coefficient of I_sc %/Cº

        Example
        =======

           http://www.posharp.com/msx120-solar-panel-from-bp-solar_p857364851d.aspx

        # from manufacturer
        I_sc = 3.87 # short circuit current
        I_mpp = 3.56 # maximum power point current
        V_mpp = 33.7 # maximum power point voltage
        V_oc = 42.1  # open circuit voltage
        N_s = 72 # number of cells in series
        P_mpp = 120 
        K_vt = -0.160 # Temperature Coefficient of V_oc %/Cº
        K_it = 0.065  # Temperature Coefficient of I_sc %/Cº	       
        '''
        from photovoltaic_modeling.parameter.parameter_extraction import ParameterExtraction

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

        self.I_sc = self.data['I_sc'] 
        self.V_oc = self.data['V_oc']  
        self.I_mpp= self.data['I_mpp']  
        self.V_mpp= self.data['V_mpp']  
        self.N_s = self.data['N_s']  

        # k_p = -0.5
        T_stc = 25 + 273.4

        E_c = 1.6022e-19 # Elementary charge
        Boltzmann = 1.3806e-23 # Boltzmann constant

        parameter_extraction = ParameterExtraction(self.I_sc, 
                                                   self.V_oc, 
                                                   self.I_mpp, 
                                                   self.V_mpp, 
                                                   number_of_cells_in_series = self.N_s)
        
        initial_guess = [1,1000,1] # [series_resistance_estimate,shunt_resistance_estimate,diode_quality_factor_estimate]
        parameter_extraction.calculate(initial_guess) 

        self.R_pv_s = parameter_extraction.series_resistance
        self.R_pv_sh = parameter_extraction.shunt_resistance
        self.K_d = parameter_extraction.diode_quality_factor


    def build(self):
        I_sc,I_mpp,V_mpp,V_oc = sym.symbols(f'I_sc,I_mpp,V_mpp,V_oc', real=True)
        N_s,K_vt,K_it = sym.symbols(f'N_s,K_vt,K_it', real=True)
        K_d,R_pv_s,R_pv_sh,v_mpp,i_pv_mpp = sym.symbols(f'K_d,R_pv_s,R_pv_sh,v_mpp,i_pv_mpp', real=True)

        temp_deg,irrad,i_pv,v_pv,p = sym.symbols(f'temp_deg,irrad,i_pv,v_pv,p', real=True)

        T_stc = 25 + 273.4
        E_c = 1.6022e-19 # Elementary charge
        Boltzmann = 1.3806e-23 # Boltzmann constant

        params_dict = {str(I_sc): self.I_sc,  # 3.87,
                       str(I_mpp):self.I_mpp,  # 42.1,
                       str(V_mpp):self.V_mpp, # 3.56,
                       str(V_oc): self.V_oc, # 33.7,
                       str(N_s):  self.N_s,   # 72,
                       str(K_vt): self.K_vt,  # -0.160,
                       str(K_it): self.K_it,  # 0.065
                       str(R_pv_s):  self.R_pv_s,   # 0.5602
                       str(R_pv_sh): self.R_pv_sh,  # 1862
                       str(K_d):  self.K_d}   # 1.3433


        temp_k = temp_deg + 273.4

        # V_t = K_d*Boltzmann*T_stc/E_c
        # V_oc_t = V_oc + K_vt*( temp_k - T_stc)

        I_rrad_sts = 1000
  
        # I_sc_t = I_sc*irrad/I_rrad_sts*(1 + K_it/100*(temp_k - T_stc))

        # I_0 = (I_sc_t - (V_oc_t - I_sc_t*R_pv_s)/R_pv_sh)*sym.exp(-V_oc_t/(N_s*V_t))

        # I_ph = (I_0*sym.exp(V_oc_t/(N_s*V_t)) + V_oc_t/R_pv_sh)*irrad/I_rrad_sts

        # eq_i_pv = -i_pv + I_ph - I_0 * (sym.exp((v_pv + i_pv*R_pv_s)/(N_s*V_t))-1)-(v_pv+i_pv*R_pv_s)/R_pv_sh 
        #eq_p = -p + i*v

        V_t = K_d*Boltzmann*T_stc/E_c
        V_oc_t = V_oc * (1+K_vt/100.0*( temp_k - T_stc))
        
        I_sc_t = I_sc*(1 + K_it/100*(temp_k - T_stc))
        I_0 = (I_sc_t - (V_oc_t - I_sc_t*R_pv_s)/R_pv_sh)*sym.exp(-V_oc_t/(N_s*V_t))
        I_d = I_0*(sym.exp((v_pv+i_pv*R_pv_s)/(V_t*N_s))-1)
        I_ph = I_sc_t*irrad/I_rrad_sts
        
        eq_i_pv = -i_pv + I_ph - I_d - (v_pv+i_pv*R_pv_s)/R_pv_sh 

        I_d = I_0*(sym.exp((v_mpp+i_pv_mpp*R_pv_s)/(V_t*N_s))-1)
        I_ph = I_sc_t*irrad/I_rrad_sts
        eq_i_pv_mpp  = -i_pv_mpp + I_ph - I_d - (v_mpp+i_pv_mpp*R_pv_s)/R_pv_sh

        i_mpp = I_ph - I_d - (v_mpp+i_pv_mpp*R_pv_s)/R_pv_sh
        p = i_mpp*v_mpp
        dPdv = sym.diff(p,v_mpp)
        eq_v_mpp = dPdv 

        i = sym.Symbol(f'i', real=True)

        #dp_0 = sym.diff((I_ph - I_0 * (sym.exp((v + i*R_pv_s)/(N_s*V_t))-1)-(v+i*R_pv_s)/R_pv_sh)*v,v)
        #v_mpp = sym.Symbol(f'v_mpp', real = True)
        #eq_v_mpp = dp_0.subs(v,v_mpp)

        u_ini_dict = {f'v_pv':30,f'irrad':1000,f'temp_deg':25}  # input for the initialization problem
        u_run_dict = {f'v_pv':30,f'irrad':1000,f'temp_deg':25}  # input for the running problem, its value is updated

        h_dict = {'I_ph':I_ph}
        h_dict = {'v':v_pv}
        h_dict = {'p':v_pv*i_pv}

        self.sys_dict = {'name':self.name,
                    'params_dict':params_dict,
                    'f_list':[],
                    'g_list':[eq_i_pv,eq_i_pv_mpp,eq_v_mpp],
                    'x_list':[ ],
                    'y_ini_list':[i_pv,i_pv_mpp,v_mpp],
                    'y_run_list':[i_pv,i_pv_mpp,v_mpp],
                    'u_ini_dict':u_ini_dict,
                    'u_run_dict':u_run_dict,
                    'h_dict':h_dict}

        bldr = db.builder(self.sys_dict)
        bldr.build()


if __name__ == "__main__":

    data = {"I_sc":3.87,"V_oc":42.1,"I_mpp":3.56,"V_mpp":33.7,"N_s":72,
            "K_vt":-0.160,"K_it":0.065,"R_pv_s": 0.5602, "R_pv_sh": 1862, "K_d": 1.3433}
    
    pv = pv_model('pv_test')
    pv.model_params(data)
    pv.build()

    import pv_test

    model = pv_test.model()
    model.ini({},1)
    model.report_y()
