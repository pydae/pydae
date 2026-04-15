# pydae/uds/builder.py
"""
Unbalanced Distribution Systems Builder.

This module will contain the code currently in pydae.urisi.urisi_builder.
It reads 3-phase unbalanced network descriptions and constructs the
symbolic DAE system dictionary for pydae.core.Builder.

Migration checklist:
  1. Copy urisi_builder.py content here
  2. Rename class urisi -> UdsBuilder
  3. Update internal imports from pydae.urisi.* -> pydae.uds.*
  4. Move component models into pydae/uds/components/
"""

# Placeholder - replace with actual urisi code
# class UdsBuilder:
#     def __init__(self, network_file):
#         ...
#     def construct(self, name):
#         ...
#         return self.sys_dict
