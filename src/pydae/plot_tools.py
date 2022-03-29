#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020

@author: jmmauricio
"""

import numpy as np
from cycler import cycler


def set_style(plt):

    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#d9524f', '#5cb85c', '#337ab7', '#f0ad4e', '#5bc0de','#5e4485'])
    #cycler('color',['#d9524f', '5cb85c', '337ab7', 'f0ad4e', '5bc0de','5e4485']) # jmcolo 
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Set default figure size
    plt.rcParams['figure.figsize'] = (3.3, 2.5)
    #plt.rcParams['figure.dpi'] : 600

    # Font sizes
    plt.rcParams['font.size'] =  10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm' #'stix'
    #plt.rcParams['text.usetex'] = True
    plt.rcParams['legend.handlelength'] = 1.

    return colors


