# pydae/bps/builder.py
"""
Balanced Power Systems Builder.

This module will contain the code currently in pydae.bmapu.bmapu_builder.
It reads JSON/HJSON network descriptions and constructs the symbolic
DAE system dictionary (sys_dict) for the pydae.core.Builder.

Migration checklist:
  1. Copy bmapu_builder.py content here
  2. Rename class bmapu -> BpsBuilder  
  3. Update internal imports from pydae.bmapu.* -> pydae.bps.*
  4. Move component models (generators, loads, lines, etc.) into
     pydae/bps/components/
  5. Move grid utility functions (lines.py, etc.) into pydae/bps/utils/
"""

# Placeholder - replace with actual bmapu code
# class BpsBuilder:
#     def __init__(self, network_file):
#         ...
#     def construct(self, name):
#         ...
#         return self.sys_dict
