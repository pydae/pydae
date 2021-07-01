# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:57:25 2020

@author: jmmauricio
"""
import numpy as np


def get_v(syst,nodes_dict):
    
    for node in nodes_dict:
        v = syst.get_value(f'v_{node}')
        nodes_dict[node].update({'v':v})

def get_vp(syst,nodes_dict):
    
    for node in nodes_dict:
        v = syst.get_value(f'v_{node}')
        p = syst.get_value(f'p_{node}')
        nodes_dict[node].update({'v':v,'p':p})




def trains_update(t,trains,trips):
    '''
    trips = {
        'right':{'times':t_right,'positions':x_right,'powers':p_right},
        'left': {'times': t_left,'positions': x_left,'powers': p_left},
         }
    '''
    # train positions and powers updates
    train_positions = []
    train_powers = []
    train_idxs = []
    train_idts = []
    for train in trains:
        if t>train['t_ini']*60:  # the train started the trip
            trip = train['trip']
            t_trip = trips[trip]['times']
            if t<(train['t_ini']*60+t_trip[-1]):   # the train if still travelling 
                x_trip = trips[trip]['positions']
                p_trip = trips[trip]['powers']
                train_idt = np.argmin(np.array(t_trip)<(t-train['t_ini']*60)) 
                train_positions += [x_trip[train_idt]]
                train_powers    += [p_trip[train_idt]]
                train_idts  += [train_idt]   
    
    return train_positions,train_powers


def trains2params(sections,train_positions_list,train_powers_list,r_m):
    
    train_positions = np.array(train_positions_list) 
    train_powers = np.array(train_powers_list) 
    sort_index = np.argsort(train_positions) 

    train_positions_sorted =  train_positions[sort_index] 
    train_powers_sorted =  train_powers[sort_index] 

    for section in sections:
        section['N_trains'] = 0
        section['T_power'] = []
        section['T_pos'] = []
        
        
    # trains per section
    for train_position,train_power in zip(train_positions_sorted,train_powers_sorted):
        for section in sections:
            if section["pos_ini"] < train_position < section["pos_end"]:
                section['N_trains']+=1
                section['T_power']+=[train_power]
                section['T_pos']+=[train_position]    

    params_dict = {}
    for section in sections:
        nodes = section['nodes_i'] + section['nodes_v']
        N_sections = len(nodes)-1
        section['length'] =  section["pos_end"] - section["pos_ini"] 
        section['N_sections'] =  N_sections
        section['sub_length'] = []

        if section['N_trains'] == 0:
            for it in range(section['N_sections']):
                section['sub_length'] += [section['length']/section['N_sections']]

        if section['N_trains'] == 1:
            length = section['T_pos'][0] - section["pos_ini"]
            section['sub_length'] +=  [length]
            for it in range(1,section['N_sections']):
                section['sub_length'] += [(section['length']-length)/(section['N_sections']-1)]  

        if section['N_trains'] == 2:
            length_1 = section['T_pos'][0] - section["pos_ini"]
            section['sub_length'] +=  [length_1]
            length_2 = section['T_pos'][1] - section['T_pos'][0]
            section['sub_length'] +=  [length_2]
            for it in range(2,section['N_sections']):
                section['sub_length'] += [(section['length']-(length_1+length_2))/(section['N_sections']-2)]     

        if section['N_trains'] == 3:
            length_1 = section['T_pos'][0] - section["pos_ini"]
            section['sub_length'] +=  [length_1]
            length_2 = section['T_pos'][1] - section['T_pos'][0]
            section['sub_length'] +=  [length_2]
            length_3 = section['T_pos'][2] - section['T_pos'][1]
            section['sub_length'] +=  [length_3]
            for it in range(3,section['N_sections']):
                section['sub_length'] += [(section['length']-(length_1+length_2+length_3))/(section['N_sections']-3)] 

        if section['N_trains'] == 4:
            length_1 = section['T_pos'][0] - section["pos_ini"]
            section['sub_length'] +=  [length_1]
            length_2 = section['T_pos'][1] - section['T_pos'][0]
            section['sub_length'] +=  [length_2]
            length_3 = section['T_pos'][2] - section['T_pos'][1]
            section['sub_length'] +=  [length_3]
            length_4 = section['T_pos'][3] - section['T_pos'][2]
            section['sub_length'] +=  [length_4]
            for it in range(3,section['N_sections']):
                section['sub_length'] += [(section['length']-(length_1+length_2+length_3+length_4))/(section['N_sections']-4)] 



    # impedance between nodes
    for section in sections:
        nodes = sorted(section['nodes_i']+section['nodes_v'])
        for i_sub in range(len(nodes)-1):           
            j = nodes[i_sub]
            k = nodes[i_sub+1]
            params_dict.update({f'R_{j}{k}':r_m*section['sub_length'][i_sub]})

    # train powers to nodes_i
    i_sec = 0
    for section in sections:
        nodes_i = section['nodes_i']
        T_powers = section['T_power']
        for it in range(section['N_tnodes']):           
            params_dict.update({f'p_{nodes_i[it]}':0.0})
        for it in range(section['N_trains']):       
            if i_sec == 0: params_dict.update({f'p_{nodes_i[it+1]}':-T_powers[it]})
            if i_sec >  0: params_dict.update({f'p_{nodes_i[it]}':-T_powers[it]})
        i_sec += 1

    nodes_dict = {}
    abs_length = 0.0
    for section in sections:
        nodes_i = section['nodes_i']
        nodes_v = section['nodes_v']
        section_nodes = sorted(nodes_i + nodes_v)
        for length,node in zip(section['sub_length'],section_nodes[:-1]):
            nodes_dict.update({node:{'pos':abs_length}})
            abs_length += length
    nodes_dict.update({section_nodes[-1]:{'pos':abs_length}})


    return params_dict,nodes_dict
