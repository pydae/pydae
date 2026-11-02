"""
pydae.emac - Electric Machines and their Controllers
=====================================================

Two ways to simulate a machine and its drive, sharing one set of tables:

* **Mean-value DAE models**, built like any other pydae system and solved by
  pydae's own solvers (``ini``, ``run``, ``A_eval``/eigenvalues).  Machines
  in the rotor's dq frame -- linear (``pmsm_dq``) or from FEA look-up tables
  with saturation and cross-saturation (``pmsm_lut``) -- plus their
  controllers (``foc_pi``)::

      from pydae.emac import EmacBuilder

      drive = EmacBuilder('drive.hjson')
      model = drive.build('drive')          # CasadiModel
      model.ini({}, drive.sys_dict['xy_0_dict'])
      model.run(0.05, {'i_q_ref_1': 20.0}, method='trapezoidal')
      model.post()

* **The switched drive** (``pydae.emac.drive``), a multirate solver for what
  a mean-value model cannot show: the DSP sampling every T_s, the
  computation delay, the triangular carrier, each leg's switching edge, dead
  time and the FEA's spatial ripple.  One control period is one CasADi
  function and ``mapaccum`` chains thousands of them in one call.

The tables both use come from ``pydae.emac.luts``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pydae-emac")
except PackageNotFoundError:
    __version__ = "unknown"

from pydae.emac.emac_builder import EmacBuilder

__all__ = ["EmacBuilder", "__version__"]
