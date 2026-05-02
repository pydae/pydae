# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

from pydae.bps.avrs.avrs import add_avr
from pydae.bps.govs.govs import add_gov
from pydae.bps.miscellaneous.load_controller import add_lc
from pydae.bps.psss.psss import add_pss
from pydae.bps.syns.milano2ord import milano2ord
from pydae.bps.syns.milano3ord import milano3ord
from pydae.bps.syns.milano4ord import milano4ord
from pydae.bps.syns.milano6ord import milano6ord
from pydae.bps.syns.pai6ord import pai6


def add_syns(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['syns']:

        data_dict = item

        bus_name = item['bus']

        if 'name' in item:
            name = item['name']
        else:
            name = bus_name

        for gen_id in range(100):
            if name not in grid.generators_id_list:
                grid.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'

        item['name'] = name

        if 'type' in item:
            if item['type'] == 'pai6':
                p_W, q_var = pai6(grid, name, bus_name, data_dict)
            elif item['type'] == 'milano6ord':
                p_W, q_var = milano6ord(grid, name, bus_name, data_dict)
            elif item['type'] == 'milano6':
                p_W, q_var = milano6ord(grid, name, bus_name, data_dict)
            elif item['type'] == 'milano4ord':
                p_W, q_var = milano4ord(grid, name, bus_name, data_dict)
            elif item['type'] == 'milano3ord':
                p_W, q_var = milano3ord(grid, name, bus_name, data_dict)
            elif item['type'] == 'milano2ord':
                p_W, q_var = milano2ord(grid, name, bus_name, data_dict)
            else:
                print(f"Synchrnous machine model type {item['type']} not found")
        else:
            p_W, q_var = milano4ord(grid, name, bus_name, data_dict)

        # grid power injection
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if 'idx_powers' not in buses[idx_bus]: buses[idx_bus].update({'idx_powers': 0})
        buses[idx_bus]['idx_powers'] += 1

        S_base = grid.backend.symbols('S_base')
        grid.dae['g'][idx_bus*2]   += -p_W / S_base
        grid.dae['g'][idx_bus*2+1] += -q_var / S_base

        v_f = f'v_f_{name}'
        p_m = f'p_m_{name}'
        v_pss = f'v_pss_{name}'

        if 'avr' in item:
            add_avr(grid.dae, item, name, bus_name, grid.backend)
            grid.dae['u_ini_dict'].pop(str(v_f))
            grid.dae['u_run_dict'].pop(str(v_f))
            grid.dae['xy_0_dict'].update({str(v_f): 1.5})
        if 'gov' in item:
            add_gov(grid.dae, item, name, bus_name, grid.backend)
            grid.dae['u_ini_dict'].pop(str(p_m))
            grid.dae['u_run_dict'].pop(str(p_m))
            grid.dae['xy_0_dict'].update({str(p_m): 0.5})
        if 'pss' in item:
            add_pss(grid.dae, item, name, bus_name, grid.backend)
            grid.dae['u_ini_dict'].pop(str(v_pss))
            grid.dae['u_run_dict'].pop(str(v_pss))
            grid.dae['xy_0_dict'].update({str(v_pss): 0.0})
        # ── MW → pu normalisation (done once, before LC trigger checks) ──────
        # Top-level: p_c_lc_mw → p_c_lc (pu on machine base)
        s_n = item.get('S_n', 100e6)
        if 'p_c_lc_mw' in item and 'p_c_lc' not in item:
            item['p_c_lc'] = item['p_c_lc_mw'] * 1e6 / s_n
        #print(f"Added syn {name} at bus {bus_name} with S_n={s_n/1e6:.2f} MVA, p_W={p_W}, q_var={q_var}, p_c_lc={item.get('p_c_lc', 'N/A')}")
        # Explicit lc sub-dict: lc.p_c_lc_mw → lc.p_c_lc
        if 'lc' in item and 'p_c_lc_mw' in item['lc'] and 'p_c_lc' not in item['lc']:
            item['lc']['p_c_lc'] = item['lc']['p_c_lc_mw'] * 1e6 / s_n

        if 'lc' in item:
            add_lc(grid.dae, item, name, bus_name, grid.backend)
        elif 'p_c_lc' in item and 'p_m' not in item:
            # Bare p_c_lc at the syn level means "desired p_g" → auto-wrap with LC.
            item['lc'] = {'K_i': 0.001, 'p_c_lc': item['p_c_lc']}
            add_lc(grid.dae, item, name, bus_name, grid.backend)


# from pydae.utils import read_data

# def load_params(model,data_input):

#     data = read_data(data_input)

#     if 'syns' in data:
#         for syn_data in data['syns']:
#             name = syn_data['bus']
#             if 'name' in syn_data: name = syn_data['name']
#             for item in syn_data:
#                 if item not in ['avr','pss','gov','bus']:
#                     model.set_value(f'{item}_{name}',syn_data[item])
#                 if item == 'gov':
#                     for item_gov in syn_data['gov']:
#                         param_name = f'{item_gov}_{name}'
#                         if param_name in model.params_list:
#                             model.set_value(f'{item_gov}_{name}',syn_data['gov'][item_gov])

#     if 'vscs' in data:
#         for vsc_data in data['vscs']:
#             name = vsc_data['bus']
#             if 'name' in vsc_data: name = vsc_data['name']
#             for item in vsc_data:
#                 if item in ['S_n']:
#                     model.set_value(f'{item}_{name}',vsc_data[item])

#     if 'buses' in data:
#         for bus_data in data['buses']:
#             name = bus_data['name']

#             for item in bus_data:
#                 if item == 'P_W':
#                     model.set_value(f'P_{name}',bus_data[item])





