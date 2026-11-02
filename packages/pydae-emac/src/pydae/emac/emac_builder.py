# -*- coding: utf-8 -*-
"""EmacBuilder: electric machines and their controllers -> a pydae ``sys_dict``.

Plays the role ``BpsBuilder`` plays for power systems: it reads a data
description, dispatches each item to its component model, and concatenates
the equations into one ``sys_dict`` for pydae-core's builders.

**Data format**::

    {
      machines: [
        {type: "pmsm_dq", name: "1", R_s: 0.1, L_d: 1e-3, L_q: 2e-3, Psi_pm: 0.1,
         p: 4, mech: "imposed", omega_m: 150.0},
      ],
      ctrls: [
        {type: "foc_pi", name: "1", machine: "1", bandwidth: 500.0, V_dc: 540.0,
         i_d_ref: 0.0, i_q_ref: 10.0},
      ],
    }

Machine models live in ``pydae.emac.machines`` (``pmsm_dq``, ``pmsm_lut``),
controllers in ``pydae.emac.ctrls`` (``foc_pi``).  Each documents its keys
in its ``descriptions()``.

The ``dae`` dict uses the ``sys_dict`` keys (``f_list``, ``x_list``, ...),
the same the ``pydae.control.continuos`` blocks append into, so any of those
blocks can be wired onto a machine model directly (``foc_pi`` is built that
way, from two ``pi`` blocks and optionally two ``lpf1``).

CasADi is the default backend: the look-up-table machine is a CasADi
B-spline.  ``use_casadi=False`` (SymPy -> C) works for the linear models.
"""

import json
import os

import hjson
from pydae.core.builder.casadi_builder import MathBackend
from pydae.emac.ctrls.ctrls import add_ctrls
from pydae.emac.machines.machines import add_machines


class EmacBuilder:
    """Assemble a drive (machines + controllers) into a pydae ``sys_dict``.

    Parameters
    ----------
    data_input : str or dict
        Path to a ``.json`` / ``.hjson`` description, or the dict itself.
    use_casadi : bool
        True (default): CasADi SX expressions, for ``CasadiBuilder``.
        False: SymPy, for the C builder (linear models only).
    """

    def __init__(self, data_input, use_casadi=True):
        if isinstance(data_input, dict):
            data = data_input
        else:
            with open(data_input, "r", encoding="utf-8") as fobj:
                text = fobj.read()
            data = hjson.loads(text) if os.path.splitext(data_input)[1] == ".hjson" else json.loads(text)
            # relative table paths are relative to the description file
            base = os.path.dirname(os.path.abspath(data_input))
            for item in data.get("machines", []):
                if "lut_file" in item and not os.path.isabs(item["lut_file"]):
                    item["lut_file"] = os.path.join(base, item["lut_file"])
        self.data = data
        self.use_casadi = use_casadi
        self.backend = MathBackend(use_casadi)
        self.machines = {}      # name -> what controllers need (omega_e, L_dq, R_s, ...)
        self.ctrls = {}
        self.dae = {"f_list": [], "x_list": [], "g_list": [], "y_ini_list": [], "y_run_list": [],
                    "u_ini_dict": {}, "u_run_dict": {}, "params_dict": {}, "h_dict": {},
                    "xy_0_dict": {}}

    def construct(self, name):
        """Build ``self.sys_dict`` (and ``self.dae``) for a model called ``name``."""
        add_machines(self)
        add_ctrls(self)
        self.sys_dict = dict(name=name, **self.dae)
        return self.sys_dict

    def build(self, name):
        """``construct(name)`` and build it.

        CasADi backend: returns a ready ``CasadiModel`` (nothing is compiled).
        SymPy backend: compiles to C with pydae's ``Builder`` and returns the
        ``Model``.
        """
        self.construct(name)
        if self.use_casadi:
            from pydae.core.builder.casadi_builder import CasadiBuilder
            from pydae.core.model.casadi_model import CasadiModel
            return CasadiModel(CasadiBuilder(self.sys_dict).build())
        from pydae.core import Builder, Model
        Builder(self.sys_dict).build()
        return Model(name)
