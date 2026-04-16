from pydae.bps import BpsBuilder
from pydae.core import Builder, Model

grid = BpsBuilder("smib_k13p2_4ord.hjson")
grid.checker()
grid.uz_jacs = True
grid.construct("smib_k13p2_4ord")
bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
bld.build()
