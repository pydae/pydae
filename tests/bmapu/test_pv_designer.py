from pydae.bmapu import bmapu_builder
from pydae.etools.pv_designer import Desing


def build():
    data = {
        "M": 2,
        "N": 3,
        "S_base": 10e6,
        "P_inv_max": 3e6, # W
        "S_inv_max": 3e6,
        "U_lv": 400,
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
        "V_dc_n" : 800  # DC nominal voltage 
        }

    d = Desing(data)
    d.design()

    grid = bmapu_builder.bmapu(d.base_data)
    #grid.checker()
    grid.uz_jacs = True
    grid.verbose = False
    grid.construct('temp')     
    grid.compile()   

def ini():

    import temp


    model = temp.model()
    model.ini({},xy_0='xy_0.json')

    model.report_z()

if __name__ == "__main__":
    build()
    ini()



    




