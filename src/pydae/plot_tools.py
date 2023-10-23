#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020

@author: jmmauricio
"""

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

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


def plot(model,layout,file_name=''):
    '''
    layout = [[['v_ref_1'],['V_1','V_2']],
              [['omega_1'],['theta_1']]]

    layout = [['v_ref_1'],['V_1','V_2']]

    layout = [[['v_ref_1'],
               ['omega_1']]]

    layout = ['v_ref_1','omega_1']
    
    '''


    
    colors = set_style(plt)
    nrows = len(layout)
    if type(layout[0]) == list:
        ncols = len(layout[0])
        figsize = (3.3*ncols,3*nrows)
        fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
    else:
        nrows = 1
        ncols = 0
        figsize = (3.3,3)
        fig,axes = plt.subplots(figsize=figsize)


    fig.tight_layout()
    
    if ncols>1 and nrows>1:
        for irow in range(nrows):
            for icol in range(ncols):
                for item in layout[irow][icol]:
                    axes[irow,icol].plot(model.Time,model.get_values(item),label=item)
                axes[irow,icol].grid()
                axes[irow,icol].legend()
                axes[irow,icol].set_xlim([model.Time[0],model.Time[-1]])
        for icol in range(ncols):
            axes[-1,icol].set_xlabel('Time (s)')
        
    if nrows>1 and ncols==1:
        for irow in range(nrows):
            for item in layout[irow]:
                axes[irow].plot(model.Time,model.get_values(item),label=item)
            axes[irow].grid()
            axes[irow].legend()
            axes[irow].set_xlim([model.Time[0],model.Time[-1]])
            axes[-1].set_xlabel('Time (s)')

    if nrows==1 and ncols>1:
        for icol in range(ncols):
            for item in layout[0][icol]:
                axes[icol].plot(model.Time,model.get_values(item),label=item)
            axes[icol].grid()
            axes[icol].legend()
            axes[icol].set_xlim([model.Time[0],model.Time[-1]])
            axes[icol].set_xlabel('Time (s)')

    if nrows==1 and ncols==0:
        for item in layout:
            axes.plot(model.Time,model.get_values(item),label=item)
        axes.grid()
        axes.legend()
        axes.set_xlim([model.Time[0],model.Time[-1]])
        axes.set_xlabel('Time (s)')

    if file_name != '':
        fig.savefig(file_name)

    return fig

