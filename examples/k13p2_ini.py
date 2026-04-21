import numpy as np
from pydae.core import Builder, Model
from pydae.bps import BpsBuilder  


grid = BpsBuilder('k13p2.hjson')
grid.construct('k13p2')
bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
bld.build()