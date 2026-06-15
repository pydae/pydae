# -*- coding: utf-8 -*-
r"""
Created on Thu August 10 23:52:55 2026

@author: jmmauricio

Control system builder (CONTROL_BUILDER).

Assembles a DAE system (`sys_dict`) from a list of continuous control
blocks (``pydae.control.continuos``) described in a ``"control"``
section of a JSON/HJSON data dictionary. Plays the same role for
standalone control systems that
:class:`pydae.bps.bps_builder.BpsBuilder` plays for power-system
networks: it reads a data description, dispatches each item to the
matching component builder, and concatenates the resulting equations
into a single ``dae`` dict ready for ``CasadiBuilder``/``Builder``.

**Data format**

```python
data = {
    "control": [
        {"type": "lpf1", "name": "meas_1", "T": 0.1, "u": 0.0},
        {"type": "pi",   "name": "avr_1",  "K_p": 5.0, "K_i": 2.0},
        {"type": "stab_1lpf_2wo_3ll", "name": "pss_1", "T_wo1": 5.0},
    ]
}
```

Each item is a dictionary with at least a ``"type"`` key (selecting
the block builder, one of the modules in
``pydae.control.continuos``: ``pi``, ``pid``, ``lpf1``, ``lpf2``,
``leadlag``, ``washout``, ``stab_1lpf_2wo_3ll``) and a ``"name"`` key
(the instance name, used to build the symbol prefixes, e.g.
``u_lpf1_meas_1``). Any other key is forwarded as-is to the block's
``data_dict`` (parameters not given fall back to the block's
``descriptions()`` defaults).

Each block is added as an independent set of equations/variables --
``ControlBuilder`` does not wire blocks together. Cross-block wiring
(as in ``stab_1lpf_2wo_3ll``) is the responsibility of the block
builder itself; combining independently-listed blocks into a loop can
be done afterwards on the returned ``dae``/``sys_dict``, following the
pattern in ``tests/core/test_control_pi_lpf1_loop.py``.
"""

import json
import os

import hjson

from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
from pydae.core.builder.sympy_builder import Builder as db

from pydae.control.continuos.pi import pi
from pydae.control.continuos.pid import pid
from pydae.control.continuos.lpf1 import lpf1
from pydae.control.continuos.lpf2 import lpf2
from pydae.control.continuos.leadlag import leadlag
from pydae.control.continuos.washout import washout
from pydae.control.continuos.summation import summation
from pydae.control.continuos.matrix_gain import matrix_gain
from pydae.control.continuos.tf import tf
from pydae.control.continuos.lti import lti
from pydae.control.continuos.luenberger import luenberger
from pydae.control.continuos.kalman import kalman
from pydae.control.continuos.stab_1lpf_2wo_3ll import stab_1lpf_2wo_3ll

BUILDERS = {
    'pi': pi,
    'pid': pid,
    'lpf1': lpf1,
    'lpf2': lpf2,
    'leadlag': leadlag,
    'washout': washout,
    'summation': summation,
    'matrix_gain': matrix_gain,
    'tf': tf,
    'lti': lti,
    'luenberger': luenberger,
    'kalman': kalman,
    'stab_1lpf_2wo_3ll': stab_1lpf_2wo_3ll,
}


class ControlBuilder:
    '''

    Parameters
    ----------
    data_input : str or dict
        File path to a ``.json`` or ``.hjson`` file holding the control
        system description, or a dict with the same structure.
    use_casadi : bool
        If True (default), use CasADi SX symbols (no C compiler
        needed). If False, use SymPy symbols for the SymPy/C-codegen
        backend.

    Returns
    -------
    ControlBuilder
        Object exposing ``.dae`` (equations/variables accumulated from
        every ``control`` item) and, after :meth:`construct`,
        ``.sys_dict`` (ready for ``CasadiBuilder`` or
        ``pydae.core.builder.sympy_builder.Builder``).

    {
     'control': [
        {'type': 'lpf1', 'name': 'meas_1', 'T': 0.1, 'u': 0.0},
        {'type': 'pi',   'name': 'avr_1',  'K_p': 5.0, 'K_i': 2.0},
     ]
    }

    '''

    def __init__(self, data_input='', use_casadi=True, testing=False):

        if isinstance(data_input, str):
            ext = os.path.splitext(data_input)[1]
            if ext == '.json':
                with open(data_input, 'r', encoding='utf-8') as fobj:
                    data = json.loads(fobj.read().replace("'", '"'))
            elif ext == '.hjson':
                with open(data_input, 'r', encoding='utf-8') as fobj:
                    data = hjson.loads(fobj.read().replace("'", '"'))
            else:
                raise ValueError(f"data_input must have a .json or .hjson extension, got '{data_input}'")
        elif isinstance(data_input, dict):
            data = data_input
        else:
            raise ValueError(f"data_input must be a str (file path) or dict, got {type(data_input)}")

        self.data = data
        self.control = data.get('control', [])

        self.use_casadi = use_casadi
        self.backend = MathBackend(use_casadi)

        self.dae = {'f_list': [], 'x_list': [],
                     'g_list': [], 'y_ini_list': [], 'y_run_list': [],
                     'u_ini_dict': {}, 'u_run_dict': {},
                     'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}

        self.testing = testing

    def construct(self, name):

        for item in self.control:

            if 'type' not in item:
                raise ValueError(f"Control item is missing the 'type' key: {item}")
            if 'name' not in item:
                raise ValueError(f"Control item is missing the 'name' key: {item}")

            block_type = item['type']
            if block_type not in BUILDERS:
                raise ValueError(
                    f"Unknown control block type '{block_type}'. "
                    f"Available types: {sorted(BUILDERS)}")

            builder = BUILDERS[block_type]
            # Uniform signature across the library:
            #   (dae, name, backend, data_dict, input=None, output=None)
            # see pydae.control.continuos._common for input/output semantics.
            builder(self.dae, item['name'], self.backend, item,
                    item.get('input'), item.get('output'))

        self.sys_dict = {'name': name,
                         'params_dict': self.dae['params_dict'],
                         'f_list': self.dae['f_list'],
                         'g_list': self.dae['g_list'],
                         'x_list': self.dae['x_list'],
                         'y_ini_list': self.dae['y_ini_list'],
                         'y_run_list': self.dae['y_run_list'],
                         'u_ini_dict': self.dae['u_ini_dict'],
                         'u_run_dict': self.dae['u_run_dict'],
                         'h_dict': self.dae['h_dict'],
                         'xy_0_dict': self.dae['xy_0_dict']}

        if self.testing:
            self.sys_dict.update({'testing': True})

    def compile(self):

        if self.use_casadi:
            return CasadiBuilder(self.sys_dict).build()

        bldr = db(self.sys_dict)
        bldr.build()
        return self.sys_dict

    def build(self, name=''):
        if name == '':
            print('Error: name is not provided.')
        self.construct(name)
        return self.compile()


if __name__ == "__main__":

    data = {
        "control": [
            {"type": "lpf1", "name": "meas_1", "T": 0.1, "u": 0.0},
            {"type": "leadlag", "name": "comp_1", "T_num": 1.0, "T_den": 0.5, "u": 0.0},
            {"type": "washout", "name": "pss_1", "T_wo": 5.0, "u": 0.0},
        ]
    }

    from pydae.core.model.casadi_model import CasadiModel

    cb = ControlBuilder(data, use_casadi=True)
    cb.construct('control_demo')

    model = CasadiModel(CasadiBuilder(cb.sys_dict).build())
    model.ini({}, cb.dae['xy_0_dict'])

    model.report_x()
    model.report_z()
    model.report_params()
