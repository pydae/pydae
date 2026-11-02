"""FEA look-up tables: load, invert to fluxes -> currents, Fourier in theta,
pack for CasADi's B-spline.  Shared by the DAE machines and the switched drive."""

from pydae.emac.luts.flux_tables import (
    ForwardMap,
    build_flux_tables,
    load_fea_lut,
    mean_tables,
    save_fea_lut,
    zero_current_flux,
)
from pydae.emac.luts.skew import skew_fea_lut
from pydae.emac.luts.synthetic import linear_fea_lut, linear_mean_torque

__all__ = ["ForwardMap", "build_flux_tables", "load_fea_lut", "mean_tables",
           "save_fea_lut", "skew_fea_lut", "zero_current_flux", "linear_fea_lut", "linear_mean_torque"]
