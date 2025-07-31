import numpy as np
import json,hjson
from pydae.utils import read_data


class Desing:

    def __init__(self, data):

        self.data = read_data(data)
        self.z_decimals = 5
        self.lines_impedance_format = 'Z_km' # 'Z_km', 'pu-s', 'Z'
        

        self.S_base    = self.data['S_base']  
        self.P_inv_max = self.data['P_inv_max']  
        self.S_inv_max = self.data['S_inv_max']  
        self.U_lv      = self.data['U_lv']  
        self.U_mv      = self.data['U_mv']  
        self.U_hv      = self.data['U_hv'] 
        self.F         = self.data['F']  
        self.M         = self.data['M'] 
        self.N         = self.data['N']  
        self.S_bess_mva         = self.data['S_bess_mva'] 
        self.S_bess_storage_kWh = self.data['S_bess_storage_kWh'] 
        self.Irrad_max          = self.data['Irrad_max'] 
        self.Area_form_factor   = self.data['Area_form_factor'] 
        self.PV_efficiency      = self.data['PV_efficiency'] 
        self.Z_trafo_poi_pu     = self.data['Z_trafo_poi_pu'] 
        self.I_mp = data["I_mp"]
        self.V_mp = data["V_mp"]
        self.V_dc_n = data["V_dc_n"] 

        self.cables = [
        {'name':  '1X95', 'R_dc_km': 0.320, 'R_ac_km': 0.403, 'X_km': 0.128, 'muC_km': 0.187, 'R_0_km': 1.050, 'X_0_km': 0.391, 'muC_0_km': 0.187,  'I_max_aire': 255, 'I_max_direct': 205, 'I_max_tube': 190, 'I_cc_max':  8930, 'I_cc_0_max': 3140},
        {'name': '1X150', 'R_dc_km': 0.206, 'R_ac_km': 0.262, 'X_km': 0.119, 'muC_km': 0.216, 'R_0_km': 0.890, 'X_0_km': 0.341, 'muC_0_km': 0.216,  'I_max_aire': 335, 'I_max_direct': 260, 'I_max_tube': 245, 'I_cc_max': 14100, 'I_cc_0_max': 3470},
        {'name': '1X240', 'R_dc_km': 0.125, 'R_ac_km': 0.161, 'X_km': 0.109, 'muC_km': 0.260, 'R_0_km': 0.768, 'X_0_km': 0.297, 'muC_0_km': 0.260,  'I_max_aire': 455, 'I_max_direct': 345, 'I_max_tube': 320, 'I_cc_max': 22600, 'I_cc_0_max': 3810},
        {'name': '1X400', 'R_dc_km':0.0778, 'R_ac_km': 0.102, 'X_km': 0.103, 'muC_km': 0.313, 'R_0_km': 0.650, 'X_0_km': 0.237, 'muC_0_km': 0.313,  'I_max_aire': 610, 'I_max_direct': 445, 'I_max_tube': 415, 'I_cc_max': 37600, 'I_cc_0_max': 4300},
        {'name': '1X500', 'R_dc_km':0.0605, 'R_ac_km': 0.084, 'X_km': 0.099, 'muC_km': 0.329, 'R_0_km': 0.618, 'X_0_km': 0.225, 'muC_0_km': 0.329,  'I_max_aire': 715, 'I_max_direct': 505, 'I_max_tube': 480, 'I_cc_max': 47000, 'I_cc_0_max': 4810},
        {'name': '1X630', 'R_dc_km':0.0469, 'R_ac_km':0.0636, 'X_km': 0.095, 'muC_km': 0.396, 'R_0_km': 0.561, 'X_0_km': 0.195, 'muC_0_km': 0.396,  'I_max_aire': 830, 'I_max_direct': 575, 'I_max_tube': 545, 'I_cc_max': 59200, 'I_cc_0_max': 5140}
        ]


    def design(self):

        M = self.M
        N = self.N
        self.S_plant = self.S_inv_max*N*M + 1.2*self.S_bess_mva*1e6
        S_base = self.S_base

        self.LV_MV_trafo_X_pu = 0.05
        self.LV_MV_trafo_R_pu = 0.01

        S_trafo_poi = M*N*self.S_inv_max*1.1

        # Area_inv = Long_n * Long_m    
        # Long_n =  Area_form_factor * Long_m
        # Area_inv = Area_form_factor * Long_m**2

        Area_inv = self.P_inv_max/(self.Irrad_max*self.PV_efficiency)
        self.Long_m = np.sqrt( Area_inv/self.Area_form_factor )
        self.Long_n =  self.Area_form_factor * self.Long_m

        S_grid_line = self.S_plant*2.0

        self.base_data = {
            "system":{"name":f"pv_{M}_{N}","S_base":S_base,"K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},
            "buses":[
                {"name": "POIMV","P_W":0.0,"Q_var":0.0,"U_kV":self.U_mv/1e3},
                {"name":   "POI","P_W":0.0,"Q_var":0.0,"U_kV":self.U_hv/1e3},
                {"name":  "GRID","P_W":0.0,"Q_var":0.0,"U_kV":self.U_hv/1e3},
                {"name":  "BESS","P_W":0.0,"Q_var":0.0,"U_kV":self.U_lv/1e3},
            ],
            "lines":[
                {"bus_j":"POI","bus_k":"GRID","X_pu":0.001*S_base/S_grid_line,"R_pu":0.0,"Bs_pu":0.0,"S_mva":S_base/1e6, 'sym':True, 'monitor':True},
                {"bus_j":"BESS","bus_k": "POIMV","X_pu":0.01,"R_pu":0.0,"Bs_pu":0.0,"S_mva":S_base/1e6, 'sym':True, 'monitor':True},
                ],
            "transformers":[{"bus_j":"POIMV","bus_k": "POI","X_pu":0.1,"R_pu":0.0,"Bs_pu":0.0,"S_mva":self.S_plant/1e6}],
            "pvs":[],
            "sources":[{"type":"genape","bus":"GRID",
                        "S_n":1000e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,
                        "K_delta":0.001,"K_alpha":1e-6}],
            "vscs":[{"type":"bess_pq","bus":"BESS","E_kWh":self.S_bess_storage_kWh,"S_n":self.S_bess_mva*1e6,
                    "soc_ref":0.5,
                    "socs":[0.0, 0.1, 0.2, 0.8,0.9,1.0],
                    "es":[1, 1.08, 1.13, 1.17, 1.18,1.25]}
                ],
            }


        Z_base_trafo_poi = self.U_mv**2/S_trafo_poi 
        Z_trafo_poi = Z_base_trafo_poi*self.Z_trafo_poi_pu
        I_cc_max = self.U_mv/(np.sqrt(3)*Z_trafo_poi)

        for it, item in enumerate(self.cables):
            if item['I_cc_max'] > I_cc_max:
                print(f"Short circuit designed cable ({I_cc_max/1e3:0.2f} kA): {item['name']}")
                idx_cc = it
                break

        I_max = 1.0*N*self.S_inv_max/(np.sqrt(3)*self.U_mv)

        for it, item in enumerate(self.cables):
            if item['I_max_tube'] > I_max:
                print(f"Nominal current designed cable  ({I_max:0.2f} A): {item['name']}")
                idx_n = it
                break

        if idx_cc > idx_n:
            idx_cable = idx_cc
        else:
            idx_cable = idx_n

        print(f"Cable: {self.cables[idx_cable]['name']}")

        self.section_cable =  float(self.cables[idx_cable]['name'].split('X')[-1])
        self.R_km_cable =  self.cables[idx_cable]['R_ac_km']
        self.X_km_cable =  self.cables[idx_cable]['X_km']
        self.B_km_cable =  2*np.pi*50*self.cables[idx_cable]['muC_km']*1e-6
    



        P_mp = self.I_mp*self.V_mp
        N_pv_s = int(self.V_dc_n/self.V_mp)
        N_pv_p = int(self.P_inv_max/(P_mp*N_pv_s))

        S_feeder = N*self.S_inv_max
        I_feeder = S_feeder/(np.sqrt(3)*self.U_mv)

        pos_mv_poi_m_pu = 0.0
        pos_mv_poi_m = pos_mv_poi_m_pu * (M-1)*self.Long_m
        pos_mv_poi_n = 0.0

        for i_m in range(1,M+1):
            name_j = "POIMV"
            monitor = True
            pos_m =  (i_m-1) * self.Long_m


            for i_n in range(1,N+1):
                name = f"{i_m}".zfill(2) + f"{i_n}".zfill(2)
                name_k = 'MV' + name

                self.base_data['buses'].append({"name":f"LV{name}","P_W":0.0,"Q_var":0.0,"U_kV":self.U_lv/1e3})
                self.base_data['buses'].append({"name":f"MV{name}","P_W":0.0,"Q_var":0.0,"U_kV":self.U_mv/1e3})


                if i_n == 1:
                    Long = np.abs(pos_mv_poi_m - pos_m) + self.Long_n*0.5
                else:
                    Long = self.Long_n

                pos_n =  (i_n-1) * self.Long_n + self.Long_n/2
                #print(f'pos_m = {pos_m:5.1f}, pos_n = {pos_n:5.1f}')

                # LV-MV Trafos
                S_trafo_n = 1.2*self.S_inv_max
                X_pu = self.LV_MV_trafo_X_pu*self.S_base/S_trafo_n
                R_pu = self.LV_MV_trafo_R_pu*self.S_base/S_trafo_n



                self.base_data['lines'].append({"bus_j":f"LV{name}","bus_k":f"MV{name}","X_pu":np.round(X_pu,self.z_decimals),"R_pu":np.round(R_pu,self.z_decimals),"Bs_pu":0.0,"S_mva":S_base/1e6,"monitor":False})
      
            
                # MV Cables:
                S_b = self.S_base
                Z_b = self.U_mv**2/S_b
                R_cable_pu = self.R_km_cable*Long/1e3/Z_b
                X_cable_pu = self.X_km_cable*Long/1e3/Z_b
                B_cable_pu = self.B_km_cable*Long/1e3*Z_b

                print(f"{name_k} - {name_j}, Cable: {self.cables[idx_cable]['name']}, R_km = {self.R_km_cable:3.4f}, X_km = {self.X_km_cable:3.4f}, B_km = {self.B_km_cable:3.4f}, Long_km = {Long/1000:2.5f}")
                
                # LV/MV Transformers
 
                # MV Feeders lines
                if self.lines_impedance_format == 'pu-s':
                    self.base_data['lines'].append({"bus_j":f"{name_k}","bus_k":f"{name_j}","X_pu":X_cable_pu,"R_pu":R_cable_pu,"Bs_pu":B_cable_pu,"S_mva":S_base/1e6,"monitor":monitor})

                if self.lines_impedance_format == 'Z_km':
                    X_km_cable = np.round(self.X_km_cable,self.z_decimals) 
                    R_km_cable = np.round(self.R_km_cable,self.z_decimals)
                    B_km_cable = np.round(self.B_km_cable,self.z_decimals)

                    self.base_data['lines'].append({"bus_j":f"{name_k}","bus_k":f"{name_j}","X_km": X_km_cable,"R_km": R_km_cable,"Bs_km": B_km_cable,"km":np.round(Long/1e3,4),"monitor":monitor})

                
                name_j = name_k
                self.base_data['pvs'].append({"bus":f"LV{name}","type":"pv_dq_d","S_n":self.S_inv_max,"U_n":self.U_lv,"F_n":50.0,"X_s":0.1,"R_s":0.0001,"monitor":False,
                                    "I_sc":8,"V_oc":42.1,"I_mp":self.I_mp,"V_mp":self.V_mp,"K_vt":-0.160,"K_it":0.065,"N_pv_s":N_pv_s,"N_pv_p":N_pv_p})
            
                monitor = False

        with open(f'pv_{M}_{N}.json','w') as fobj:
            fobj.write(json.dumps(self.base_data, indent=2))

    def report(self):

        M = self.M
        N = self.N

        latex = r'''
        \begin{table}
            \centering
            \caption{Nominal characteristics of PV inverters and transformers of the benchmark 2$\times$3 PV park.}
            \begin{tabular}{lr}
            \toprule
                \multicolumn{2}{c}{PV inverters} \\ \midrule
                Rated voltage & inverter_rated_voltage V \\ \hline
                Rated power & inverter_rated_mva MVA \\ \hline
                Rated power factor & inverter_pf (ind/cap)\\ \midrule
                \multicolumn{2}{c}{MV/LV PV transformers} \\ \midrule
                Rated voltages & U_mv_kV/U_lv_kV kV \\ \hline
                Rated power & MV_LV_rated_mva MVA \\ \hline
                Short-circuit impedance & 0.06 p.u. \\ \hline
                $X_{sc}/R_{sc}$ & 10 \\  \midrule
                \multicolumn{2}{c}{HV/MV POI transformer} \\ \midrule
                Rated voltages & U_hv_kV/U_mv_kV kV \\ \hline
                Rated power & poi_trafo_rated_power MVA \\ \hline 
                Short-circuit impedance & 0.13 p.u. \\ \hline
                $X_{sc}/R_{sc}$ & 10\\ \midrule
                \multicolumn{2}{c}{MV branches} \\
                \midrule
first_feeder_segment_string
                Length MV\#\# - MV\#\# & length_segments m \\ \hline
                Cross section &  section_cable mm$^2$\\ \hline
                Resistance & R_km_cable $\Omega$/km \\ \hline
                Reactance  & X_km_cable $\Omega$/km \\ %\hline
        %        Susceptance & B_km_cable S/km \\ 
                \bottomrule
            \end{tabular}
            \label{tab:PV_MxN}
        \end{table}        
        '''

        latex = latex.replace('inverter_rated_voltage', f'{self.U_lv:0.0f}')
        latex = latex.replace('inverter_rated_mva', f'{self.S_inv_max/1e6:0.1f}')
        latex = latex.replace('inverter_pf', f'{self.P_inv_max/self.S_inv_max:0.2f}')
        latex = latex.replace('U_mv_kV', f'{self.U_mv/1e3:0.0f}').replace('U_lv_kV', f'{self.U_lv/1e3}')
        latex = latex.replace('inverter_pf', f'{self.P_inv_max/self.S_inv_max:0.2f}')
        latex = latex.replace('U_mv_kV', f'{self.U_mv/1e3:0.0f}').replace('U_hv_kV', f'{self.U_hv/1e3:0.0f}')
        latex = latex.replace('poi_trafo_rated_power', f'{self.S_plant/1e6:0.0f}') 


        pos_mv_poi_m_pu = 0.0
        pos_mv_poi_m = pos_mv_poi_m_pu * (M-1)*self.Long_m
        pos_mv_poi_n = 0.0

        first_feeder_segment_string = ''
        for i_m in range(1,M+1):
            name_j = "POIMV"
            monitor = True
            pos_m =  (i_m-1) * self.Long_m


            for i_n in range(1,N+1):
                name = f"{i_m}".zfill(2) + f"{i_n}".zfill(2)
                name_k = 'MV' + name

                self.base_data['buses'].append({"name":f"LV{name}","P_W":0.0,"Q_var":0.0,"U_kV":self.U_lv/1e3})
                self.base_data['buses'].append({"name":f"MV{name}","P_W":0.0,"Q_var":0.0,"U_kV":self.U_mv/1e3})


                if i_n == 1:
                    Long = np.abs(pos_mv_poi_m - pos_m) + self.Long_n*0.5
                else:
                    Long = self.Long_n

                if i_n == 1:
                    first_feeder_segment_string += f'                Length {name_k}-{name_j}  & {Long:0.0f} m' + r'\\ \hline' 
                    if i_m < M:
                        first_feeder_segment_string +=  '\n'
                else:
                    self.length_segments = Long
 
        latex = latex.replace('first_feeder_segment_string', first_feeder_segment_string) 
        latex = latex.replace('length_segments', f'{self.length_segments:0.0f}') 

        latex = latex.replace('length_segments', f'{self.length_segments:0.0f}') 
        latex = latex.replace('R_km_cable', f'{self.R_km_cable:0.4f}') 
        latex = latex.replace('X_km_cable', f'{self.X_km_cable:0.4f}') 

        if self.B_km_cable > 0.001:
            latex = latex.replace('B_km_cable', f'{self.B_km_cable:0.4f}') 
        else:
            latex = latex.replace('B_km_cable', '0') 

        latex = latex.replace('section_cable', f'{self.section_cable:0.0f}')  
        latex = latex.replace('PV_MxN',  f'PV_{M}x{N}')  

        return latex


if __name__ == "__main__": 

    data = {
    "M": 2,
    "N": 3,
    "S_base": 10e6,
    "P_inv_max": 3e6, # W
    "S_inv_max": 3e6/0.85,
    "U_lv": 800,
    "U_mv": 20e3,
    "U_hv": 132e3,
    "F": 50, #Nominal Frequency (Hz)
    "S_bess_mva": 1,
    "S_bess_storage_kWh": 250,
    "Irrad_max": 1000, # W/m_2
    "Area_form_factor": 1.0,
    "PV_efficiency": 0.1,
    "Z_trafo_poi_pu": 0.1, 
    "I_mp" : 3.56,  # PV module current at MP
    "V_mp" : 33.7,  # PV module voltage at MP
    "V_dc_n" : 1200  # DC nominal voltage 
}

    d = Desing(data)
    d.design()
    print(d.report())

    #print(d.base_data)

    
