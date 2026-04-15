# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


from pydae.bmapu.pods.pod_2wo_3ll import add_pod_2wo_3ll


def add_pods(grid):

    for item in grid.data['pods']:
        if 'type' in item:
            if item['type'] == 'pod_2wo_3ll':
                add_pod_2wo_3ll(grid,item)


