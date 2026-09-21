# -*- coding: utf-8 -*-
r"""
Finite-element thermal characterisation of underground cables → pydae-bps DTR.

This module is the *identification* counterpart of
:mod:`pydae.bps.lines.line_dtr_cable`. That module integrates a two-node
IEC 60853 ladder at run time; this one works out **what its parameters should
be** for a real, geometrically-resolved cable circuit, by solving the 2D heat
equation over the cable cross-section and the surrounding soil and reducing the
result to a lumped RC ladder.

The four stages of the suite
----------------------------

1. **Geometry and materials** — :class:`Layer`, :class:`CableDesign`,
   :class:`Installation`, :class:`Backfill` and :class:`CableCase` describe a
   buried circuit as a concentric layer stack (conductor → screens →
   insulation → shield → bedding → jacket) laid flat or in trefoil in native
   soil or a backfilled trench. :data:`CABLE_LIBRARY` ships the CIGRE TB 575
   European-MV benchmark cable.

2. **FEM simulation** — :class:`ThermalFEM` meshes the cross-section (graded
   tensor grid + local adaptive refinement, no external mesher), assembles

   .. math:: -\nabla\!\cdot\!(k\,\nabla\theta) = q_v

   with a Robin (convective) ground surface and Dirichlet undisturbed-earth
   far field, and solves the steady field. The same assembly gives the
   transient descriptor system :math:`M\dot\theta + K\theta = b\,P` used by
   stage 3.

3. **Ladder model** — :func:`identify_ladder` reduces the driving-point /
   transfer thermal impedance :math:`Z(s) = c^{T}(K + sM)^{-1}b` by rational
   Krylov projection, diagonalises the reduced pencil to get the Foster form
   (positive residues ⇒ realisable), and converts it to a **Cauer ladder** by
   continued-fraction expansion. The result, :class:`LadderModel`, is an
   n-stage RC ladder whose step response matches the FEM over six decades and
   whose DC resistance is *exact*.

4. **pydae-bps parameters** — :func:`fit_two_node` collapses the ladder to the
   two-node form ``line_dtr_cable`` implements: ``T1`` analytically from the
   resolved layer stack, ``T4`` from the FEM conductor temperature (so it
   *embeds the mutual heating* of the neighbouring phases, which the library's
   isolated-cable formula cannot), and ``Q1``/``Q2`` by least squares against
   the FEM step response with the DC behaviour pinned.
   :meth:`TwoNodeFit.as_dtr` emits the ``dtr`` block to drop into a
   ``lines[]`` entry, and :func:`verify_with_pydae` builds the real pydae-bps
   model with it and checks the temperature against the FEM.

Why the single-phase equivalent is legitimate
---------------------------------------------
``pydae.bps`` is a balanced, positive-sequence simulator: one current magnitude
per line. Electrically that is exact under balance. *Thermally* the three
cables are coupled — mutual heating is ~30 % of the conductor rise — but under
balanced loading superposition collapses the 3×3 mutual matrix into one number,

.. math:: \Delta\theta_{\rm centre} = W\,(Z_{\rm self} + 2Z_{\rm mutual})
                                    = W\,Z_{\rm hot},

so a single ladder is still exact *provided* ``T4`` carries that sum. It is the
``T4`` that the analytic Kennelly formula gets wrong, not the model structure.
The self and mutual resistances are reported separately
(:attr:`LadderModel.z_self`, :attr:`LadderModel.z_mutual`) because the collapse
is invalid the moment the phases are unbalanced.

Requirements
------------
``scikit-fem >= 8`` and ``scipy`` for stages 2–4 — imported lazily, inside the
functions that need them, so importing this module (or the rest of
``pydae.bps``) never pays for them. ``matplotlib`` is needed only for the
optional plots, and a CasADi-capable ``pydae`` install only for
:func:`verify_with_pydae`. Stage 4 does import ``line_dtr_cable`` itself, to
compute the library's own geometry-derived network rather than re-implementing
its formulas here.

Quick start
-----------
.. code-block:: python

    from pydae.bps.lines.cable_fem import run_suite
    result = run_suite()                 # CIGRE TB 575 European MV, 200 A
    print(result['fit'].as_dtr())        # paste into a lines[] entry

or from a shell::

    python -m pydae.bps.lines.cable_fem --current 250 --verify
    python -m pydae.bps.lines.cable_fem --test

A different cable, layout or soil is a matter of building the case yourself:

.. code-block:: python

    from pydae.bps.lines.cable_fem import (CableCase, Installation,
                                           NA2XS2Y_120_AL, run_suite)

    case = CableCase(design=NA2XS2Y_120_AL,
                     installation=Installation(formation='trefoil', depth=1.2,
                                               k_soil=0.8, t_deep=18.0),
                     current=260.0)
    fit = run_suite(case)['fit']

For the whole workflow narrated end to end -- geometry, field, ladder, block,
ampacity, ``ini()`` and a daily load cycle through the real simulator -- see
:mod:`pydae.bps.lines.cable_fem_example`::

    python -m pydae.bps.lines.cable_fem_example --plot .

and for the same path expressed as assertions, ``tests/bps/test_cable_fem_dtr.py``.

Data provenance
---------------
Only the cable dimensions, resistances and burial layout of the default case
come from CIGRE TB 575 (Tables 6.9, 6.11, 6.12; Fig. 6.6). TB 575 is an
*electrical* benchmark: it specifies no thermal conductivity, no heat capacity,
no soil resistivity, no ambient temperature and no surface convection
coefficient. Every such value here is literature/typical data, carried on the
dataclasses so it is explicit at the call site rather than buried in the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

__all__ = [
    # stage 1 — geometry and materials
    'Layer', 'CableDesign', 'Backfill', 'Installation', 'CableCase',
    'BACKFILL_CASES', 'CABLE_LIBRARY', 'NA2XS2Y_120_AL', 'cigre_mv_eu_case',
    # stage 2 — FEM
    'ThermalFEM', 'SteadyResult',
    # stage 3 — ladder
    'LadderModel', 'identify_ladder', 'foster_to_cauer', 'transient_fem_step',
    'truncation_cost', 'DEFAULT_TAU_RANGE', 'REFERENCE_ORDER',
    # stage 4 — pydae-bps export
    'TwoNodeFit', 'fit_two_node', 'ladder_dtr', 'geometry_dtr',
    'library_defaults',
    'compare_with_library', 'bps_data', 'verify_with_pydae',
    # suite
    'run_suite', 'plot_field', 'test', 'main',
]


# =============================================================================
# 0. LAZY OPTIONAL IMPORTS
# =============================================================================

_SKFEM_HINT = (
    "cable_fem needs scikit-fem (>= 8) and scipy for the FEM stages.\n"
    '    pip install "pydae-bps[fem]"        (or: uv sync --extra fem)\n'
    "The rest of pydae.bps does not depend on them, which is why the import "
    "is deferred to here."
)


def _require_skfem():
    """Import scikit-fem on first use, with an actionable message if missing."""
    try:
        import skfem  # noqa: F401
        import scipy.sparse  # noqa: F401
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(_SKFEM_HINT) from exc
    return skfem


# =============================================================================
# 1. GEOMETRY AND MATERIALS
# =============================================================================
# A cable is a stack of concentric annuli. Each carries its own conductivity
# and volumetric heat capacity, may be a *loss source*, and may play a
# structural *role* that the rest of the suite keys off:
#
#   role 'conductor'   the rated region — where the 90 degC limit applies and
#                      where the hot-spot output functional is integrated.
#   role 'insulation'  the main dielectric — its outer radius is the default
#                      T1/T4 split node (pydae's theta_e).
#   heat 'conductor'   Joule loss W_c is injected uniformly here.
#   heat 'screen'      induced screen/sheath loss W_s is injected here. Spread
#                      over a composite when the metallic tape is thinner than
#                      the mesh (see CIGRE default below).

_ROLES = ('conductor', 'insulation', 'passive')
_HEATS = (None, 'conductor', 'screen')


@dataclass(frozen=True)
class Layer:
    """
    One concentric annulus of a cable.

    :param name: label used in reports.
    :param thickness: radial thickness [m]. Measured outward from the previous
        layer, so the first layer's thickness is the conductor *radius*.
    :param k: thermal conductivity [W/(m·K)].
    :param rho_c: volumetric heat capacity ρ·c [J/(m³·K)].
    :param role: ``'conductor'``, ``'insulation'`` or ``'passive'`` — see the
        module-level note.
    :param heat: ``'conductor'``, ``'screen'`` or ``None`` — which loss
        component is dissipated in this layer.
    """

    name: str
    thickness: float
    k: float
    rho_c: float
    role: str = 'passive'
    heat: str | None = None

    def __post_init__(self):
        if self.thickness <= 0.0:
            raise ValueError(f"layer {self.name!r}: thickness must be > 0")
        if self.k <= 0.0 or self.rho_c <= 0.0:
            raise ValueError(f"layer {self.name!r}: k and rho_c must be > 0")
        if self.role not in _ROLES:
            raise ValueError(f"layer {self.name!r}: role must be one of {_ROLES}")
        if self.heat not in _HEATS:
            raise ValueError(f"layer {self.name!r}: heat must be one of {_HEATS}")


@dataclass(frozen=True)
class CableDesign:
    """
    A single-core cable: its layer stack plus the electrical data that turns a
    current into the two loss components.

    :param name: cable designation.
    :param layers: the stack, innermost first.
    :param R_ac_90: conductor a.c. resistance at 90 °C [Ω/m].
    :param R_ph: positive-sequence resistance [Ω/m], i.e. conductor *plus*
        induced screen losses. ``R_ph / R_ac_90 - 1`` is the IEC 60287 sheath
        loss factor λ₁.
    :param alpha_R: resistance temperature coefficient [1/K].
    :param d_over: nameplate overall diameter [m]. When given, the stack is
        checked to close on it — a stack that does not close means a
        transcription error, so this is verified rather than assumed.
    :param Wd: dielectric loss [W/m]. Negligible at MV; non-zero only at HV.
    """

    name: str
    layers: tuple[Layer, ...]
    R_ac_90: float
    R_ph: float
    alpha_R: float = 0.004
    d_over: float | None = None
    Wd: float = 0.0

    # --- validation ---------------------------------------------------------
    def __post_init__(self):
        object.__setattr__(self, 'layers', tuple(self.layers))
        if not self.layers:
            raise ValueError("a cable design needs at least one layer")
        if self.layers[0].role != 'conductor':
            raise ValueError("the innermost layer must have role='conductor'")
        if self.R_ph < self.R_ac_90:
            raise ValueError("R_ph (conductor + screen) cannot be below R_ac_90")
        # The loss regions must each be a single contiguous radial band,
        # otherwise the volumetric source density is not well defined.
        for kind in ('conductor', 'screen'):
            idx = [i for i, L in enumerate(self.layers) if L.heat == kind]
            if idx and idx != list(range(idx[0], idx[-1] + 1)):
                raise ValueError(f"{kind} loss layers are not contiguous: {idx}")
        if self.d_over is not None:
            closure = abs(self.r_outer - 0.5 * self.d_over)
            if closure > 5.0e-5:                    # 0.05 mm
                raise ValueError(
                    f"{self.name}: layer stack closes on "
                    f"{2e3 * self.r_outer:.3f} mm but d_over is "
                    f"{1e3 * self.d_over:.3f} mm ({1e3 * closure:.3f} mm out). "
                    "Adjust the screen/bedding allocation until it closes.")

    # --- derived geometry ---------------------------------------------------
    @property
    def r_bounds(self) -> np.ndarray:
        """Cumulative outer radius of every layer [m], shape ``(n_layers,)``."""
        return np.cumsum([L.thickness for L in self.layers])

    @property
    def r_outer(self) -> float:
        """Cable outer radius [m]."""
        return float(self.r_bounds[-1])

    @property
    def r_conductor(self) -> float:
        """Outer radius of the conductor [m]."""
        return self._role_r_out('conductor')

    @property
    def r_insulation(self) -> float:
        """Outer radius of the main dielectric [m] — the default θ_e node."""
        return self._role_r_out('insulation')

    def _role_r_out(self, role: str) -> float:
        r = self.r_bounds
        hits = [i for i, L in enumerate(self.layers) if L.role == role]
        if not hits:
            raise ValueError(f"{self.name}: no layer with role={role!r}")
        return float(r[hits[-1]])

    def heat_band(self, kind: str) -> tuple[float, float] | None:
        """Inner/outer radius of the ``kind`` loss region [m], or ``None``."""
        r = self.r_bounds
        idx = [i for i, L in enumerate(self.layers) if L.heat == kind]
        if not idx:
            return None
        r_in = 0.0 if idx[0] == 0 else float(r[idx[0] - 1])
        return r_in, float(r[idx[-1]])

    # --- derived electrical -------------------------------------------------
    @property
    def R_ac_20(self) -> float:
        """Conductor a.c. resistance referred back to 20 °C [Ω/m]."""
        return self.R_ac_90 / (1.0 + self.alpha_R * 70.0)

    @property
    def lambda1(self) -> float:
        """IEC 60287 sheath loss factor λ₁ = W_s / W_c [-]."""
        return self.R_ph / self.R_ac_90 - 1.0

    def R_ac(self, theta: float) -> float:
        """Conductor a.c. resistance at conductor temperature *theta* [Ω/m]."""
        return self.R_ac_20 * (1.0 + self.alpha_R * (theta - 20.0))

    # --- reporting ----------------------------------------------------------
    def table(self) -> str:
        """Human-readable layer table, for the suite report."""
        rows = [f" {'layer':22s}{'r_out [mm]':>11s}{'k [W/mK]':>11s}"
                f"{'rho·c [MJ/m3K]':>17s}  role / heat"]
        for L, r in zip(self.layers, self.r_bounds):
            tag = L.role if L.role != 'passive' else '-'
            if L.heat:
                tag += f" / {L.heat}"
            rows.append(f" {L.name:22s}{1e3 * r:11.2f}{L.k:11.3f}"
                        f"{1e-6 * L.rho_c:17.2f}  {tag}")
        return "\n".join(rows)


# --- Reference design: CIGRE TB 575 European MV benchmark --------------------
# Table 6.11 (p.35), Conductor ID 2: NA2XS2Y, 19 strands, 120 mm2 Al,
#   d_c = 12.4 mm, t_i = 5.5 mm, t_ts = 0.2 mm, t_j = 2.5 mm, d_ov = 34.2 mm,
#   R'_ac(90 degC) = 0.338 Ohm/km.
# Table 6.12 (p.36): R'_ph = 0.501 Ohm/km, positive sequence, screen losses in.
#
# The 2.70 mm that the nominal layers leave unallocated between d_c/2 and
# d_ov/2 is the semiconducting screens plus the bedding/swelling tape — real
# cable construction, not a data error. The split below is typical European MV
# practice; every k and rho·c is literature data, NOT from TB 575.
#
# The 0.2 mm copper tape shield is thinner than the mesh can resolve, and its
# own thermal resistance is negligible, so the SCREEN LOSS is injected over the
# insulation-screen → bedding composite (12.50 → 14.60 mm) instead of refining
# to 0.1 mm. The source radius is then right to ~0.6 mm, and W_s never crosses
# T1, so the effect on the conductor is negligible.
NA2XS2Y_120_AL = CableDesign(
    name='NA2XS2Y 120 mm2 Al (CIGRE TB 575 cond. ID 2)',
    layers=(
        Layer('conductor',        6.20e-3, 237.0,  2.43e6, 'conductor', 'conductor'),
        Layer('conductor screen', 0.80e-3,   0.40, 2.40e6),
        Layer('insulation XLPE',  5.50e-3,   0.286, 2.40e6, 'insulation'),
        Layer('insulation screen',0.80e-3,   0.40, 2.40e6, heat='screen'),
        Layer('tape shield Cu',   0.20e-3, 400.0,  3.45e6, heat='screen'),
        Layer('bedding',          1.10e-3,   0.30, 2.40e6, heat='screen'),
        Layer('jacket PE',        2.50e-3,   0.286, 2.30e6),
    ),
    R_ac_90=0.338e-3,
    R_ph=0.501e-3,
    alpha_R=0.004,
    d_over=34.2e-3,
    Wd=0.0,
)

CABLE_LIBRARY: dict[str, CableDesign] = {
    'NA2XS2Y_120_AL': NA2XS2Y_120_AL,
}


@dataclass(frozen=True)
class Backfill:
    """
    Backfilled trench and protective plate — an axis-aligned rectangle of
    substitute soil, optionally capped by a plate.

    TB 575 §6.2.2 says the cables are "buried in back-filled trenches with a
    protective plate" but gives no width, no cover dimensions and no backfill
    properties, so every value here is an assumption.

    The trench half-width is *derived* from the resolved cable layout plus
    :attr:`margin`, never hard-coded: a fixed half-width silently leaves the
    outer cables sitting in native soil while only the centre one sees the
    backfill, which corrupts the whole comparison.

    :param k: backfill thermal conductivity [W/(m·K)].
    :param rho_c: backfill volumetric heat capacity [J/(m³·K)].
    :param margin: clearance from the outer cable surface to the trench wall [m].
    :param y_top: y-coordinate of the top of the backfill [m] (negative, below
        the ground surface at y = 0).
    :param plate_k: plate conductivity [W/(m·K)], or ``None`` for no plate.
    :param plate_rho_c: plate volumetric heat capacity [J/(m³·K)].
    :param plate_thickness: plate thickness [m].
    :param plate_y_top: y-coordinate of the top of the plate [m].
    """

    k: float = 0.83
    rho_c: float = 1.8e6
    margin: float = 0.15
    y_top: float = -0.50
    plate_k: float | None = 1.40
    plate_rho_c: float = 2.0e6
    plate_thickness: float = 0.06
    plate_y_top: float = -0.52


#: Named backfill cases for sensitivity studies (all assumed, not TB 575).
BACKFILL_CASES: dict[str, Backfill] = {
    'sand_concrete':       Backfill(k=0.83, rho_c=1.8e6, plate_k=1.40),
    'stabilised_concrete': Backfill(k=1.43, rho_c=2.2e6, plate_k=1.40),
    'native_pvc':          Backfill(k=1.00, rho_c=2.0e6, plate_k=0.19,
                                    plate_rho_c=1.7e6),
    'dry_sand_concrete':   Backfill(k=0.40, rho_c=1.3e6, plate_k=1.40),
}


@dataclass(frozen=True)
class Installation:
    """
    How the circuit is laid and what surrounds it.

    :param formation: ``'flat'`` (cables in a horizontal row, ``spacing``
        centre-to-centre) or ``'trefoil'`` (three touching cables, one on top).
    :param depth: depth of the cable axes below the ground surface [m], > 0.
        For trefoil this is the depth of the group centroid.
    :param spacing: centre-to-centre spacing [m], flat formation only.
    :param n_cables: number of cables. ``'trefoil'`` requires 3.
    :param k_soil: native soil thermal conductivity [W/(m·K)]. A soil thermal
        *resistivity* of 1.0 K·m/W is ``k_soil = 1.0``.
    :param rho_c_soil: soil volumetric heat capacity [J/(m³·K)].
    :param t_deep: undisturbed deep-earth temperature [°C] — Dirichlet data on
        the sides and bottom of the domain.
    :param t_air: air temperature above the ground [°C].
    :param h_conv: ground-surface convection coefficient [W/(m²·K)].
    :param x_half: half-width of the soil domain [m].
    :param y_deep: depth of the bottom of the soil domain [m] (negative).
    :param backfill: trench description, or ``None`` for uniform native soil.
    """

    formation: str = 'flat'
    depth: float = 0.7
    spacing: float = 0.3
    n_cables: int = 3
    k_soil: float = 1.0
    rho_c_soil: float = 2.0e6
    t_deep: float = 15.0
    t_air: float = 20.0
    h_conv: float = 10.0
    x_half: float = 4.0
    y_deep: float = -5.0
    backfill: Backfill | None = None

    def __post_init__(self):
        if self.formation not in ('flat', 'trefoil'):
            raise ValueError("formation must be 'flat' or 'trefoil'")
        if self.formation == 'trefoil' and self.n_cables != 3:
            raise ValueError("a trefoil is three cables")
        if self.depth <= 0.0:
            raise ValueError("depth is measured downward and must be > 0")
        if self.y_deep >= -self.depth:
            raise ValueError("the domain bottom must lie below the cables")

    def centers(self, r_cable: float) -> np.ndarray:
        """
        Cable axis coordinates ``(n_cables, 2)`` [m], with y = 0 the ground
        surface (so all y are negative).

        Three touching circles of radius r have their axes on an equilateral
        triangle of side 2r, whose circumradius is 2r/√3 — that is the trefoil
        layout, one cable on top, two on a flat trench bottom.
        """
        if self.formation == 'trefoil':
            rt = 2.0 * r_cable / math.sqrt(3.0)
            yc = -self.depth
            return np.array([[0.0, yc + rt],
                             [-r_cable, yc - 0.5 * rt],
                             [+r_cable, yc - 0.5 * rt]])
        n = self.n_cables
        xs = (np.arange(n) - 0.5 * (n - 1)) * self.spacing
        return np.column_stack([xs, np.full(n, -self.depth)])


@dataclass
class CableCase:
    """
    A complete, self-contained thermal case: *what* cable, laid *how*, carrying
    *what* current. Everything the FEM needs is reachable from here — there is
    no module-level mutable state, so several cases can coexist in one process.

    :param design: the cable.
    :param installation: the layout and surroundings.
    :param current: phase current [A].
    :param loss_reference_temperature: conductor temperature [°C] at which the
        losses fed to the FEM are evaluated. The thermal problem is *linear*,
        so ``T1``, ``T4``, ``Q1``, ``Q2`` and λ₁ are independent of this choice;
        it only scales the temperatures that get reported. 90 °C matches the
        temperature at which ``R_ac_90`` and ``R_ph`` are tabulated.
    :param label: optional report title.
    """

    design: CableDesign
    installation: Installation
    current: float = 200.0
    loss_reference_temperature: float = 90.0
    label: str = ''

    def __post_init__(self):
        if self.current < 0.0:
            raise ValueError("current must be >= 0")
        if not self.label:
            self.label = (f"{self.design.name}, {self.installation.formation}, "
                          f"depth {self.installation.depth} m, "
                          f"I = {self.current:.0f} A")

    # --- layout -------------------------------------------------------------
    @property
    def r_cable(self) -> float:
        """Cable outer radius [m]."""
        return self.design.r_outer

    @property
    def centers(self) -> np.ndarray:
        """Cable axis coordinates ``(n, 2)`` [m]."""
        return self.installation.centers(self.r_cable)

    @property
    def n_cables(self) -> int:
        return self.centers.shape[0]

    def kennelly_matrix(self) -> np.ndarray:
        r"""
        IEC 60287 / Kennelly external thermal-resistance matrix [K·m/W],
        ``Z[i, j]`` = surface rise of cable *i* per unit loss of cable *j*.

        The image method puts a mirror heat *sink* above the ground plane, so
        for a line source at :math:`P_j` observed at :math:`P_i`

        .. math:: Z_{ij} = \frac{\rho_{\rm soil}}{2\pi}
                           \ln\frac{d'_{ij}}{d_{ij}},

        with :math:`d_{ij}` the axis separation (the cable's own outer radius
        when :math:`i = j`) and :math:`d'_{ij}` the distance to the image of
        *j*. The diagonal reduces to the familiar
        :math:`\ln(4L/D_e)/2\pi` — which is exactly the isolated-cable formula
        ``line_dtr_cable`` uses for ``T4``, i.e. this matrix's diagonal *only*.

        Two approximations separate it from the FEM: the ground surface is
        taken isothermal (no convective film), and the cable is a bare line
        source (no jacket). Both are accounted for in
        :meth:`ThermalFEM.analytic_T4`, which is why that comparison agrees to
        a couple of percent and is a genuine check on the FEM.
        """
        c = self.centers
        n = len(c)
        rho = 1.0 / self.installation.k_soil
        if np.abs(c[:, 1]).min() <= self.r_cable:
            raise ValueError("a cable breaks the ground surface; the image "
                             "method does not apply")
        Z = np.zeros((n, n))
        for i in range(n):
            xi, yi = c[i]
            for j in range(n):
                xj, yj = c[j]
                if i == j:
                    d, dp = self.r_cable, 2.0 * abs(yi)
                else:
                    d = math.hypot(xi - xj, yi - yj)
                    dp = math.hypot(xi - xj, yi + yj)   # image of j at (xj,-yj)
                Z[i, j] = rho / (2.0 * math.pi) * math.log(dp / d)
        return Z

    @property
    def hot_index(self) -> int:
        """
        Index of the hottest cable — the one with the largest total external
        thermal resistance under balanced loading, i.e. the largest row sum of
        :meth:`kennelly_matrix`. This is the cable the circuit is rated on.

        A geometric "closest to the centroid" rule looks equivalent but is not:
        in a trefoil every cable is equidistant from the centroid, and it is
        the two *deeper* ones that run hot. The Kennelly row sum gets the
        depth and the mutual coupling right in one number, and it is free.
        """
        return int(np.argmax(self.kennelly_matrix().sum(axis=1)))

    # --- losses -------------------------------------------------------------
    def losses(self, theta: float | None = None) -> dict:
        """
        Loss split at conductor temperature *theta* (default
        :attr:`loss_reference_temperature`).

        Only the CONDUCTOR loss crosses the insulation resistance T1, so a
        layered model has to place the two components separately::

            W_c = I² · R_ac(θ)                 conductor
            W_s = λ₁ · W_c                     screen / sheath
            λ₁  = R_ph / R_ac(90 °C) − 1       IEC 60287 loss factor

        λ₁ is taken as the tabulated ratio and held constant; it is a weak
        function of temperature compared with ``R_ac`` itself.
        """
        theta = self.loss_reference_temperature if theta is None else theta
        w_c = self.current ** 2 * self.design.R_ac(theta)
        lam = self.design.lambda1
        w_s = lam * w_c
        return {'W_c': w_c, 'W_s': w_s, 'lambda1': lam,
                'W_total': w_c + w_s, 'R_ac': self.design.R_ac(theta),
                'theta': theta}

    def self_consistent_conductor_temperature(self, z_eq: float,
                                              t_soil: float) -> float:
        r"""
        Steady conductor temperature with the loss re-evaluated at that same
        temperature — the fixed point the pydae-bps model converges to.

        With :math:`Z_{\rm eq} = T_1 + (1+\lambda_1)T_4` and
        :math:`W_c = I^2 R_{ac,20}(1 + \alpha(\theta_c - 20))`,

        .. math::
            \theta_c = \frac{\theta_{\rm soil}
                             + I^2 R_{ac,20}(1 - 20\alpha) Z_{\rm eq}}
                            {1 - I^2 R_{ac,20}\,\alpha\,Z_{\rm eq}}

        A non-positive denominator means thermal runaway: the loss grows with
        temperature faster than the soil can carry it away, and no steady state
        exists at this current.
        """
        a = self.design.alpha_R
        g = self.current ** 2 * self.design.R_ac_20 * z_eq
        den = 1.0 - g * a
        if den <= 0.0:
            raise ValueError(
                f"no steady state at I = {self.current:.0f} A: the "
                f"resistance-temperature feedback loop gain is {g * a:.3f} "
                ">= 1 (thermal runaway)")
        return (t_soil + g * (1.0 - 20.0 * a)) / den

    # --- material and source fields ----------------------------------------
    # All of these are fully vectorised and accept arrays of ANY shape: nodes,
    # element centroids, or the (n_elements, n_quadrature_points) arrays that
    # scikit-fem hands to a form as ``w.x``. Evaluating the material at the
    # QUADRATURE POINTS rather than per element gives a sub-element-accurate
    # weighting of the elements that the cable circles cut through.

    def distance_to_axes(self, x, y) -> np.ndarray:
        """Distance from each point to the nearest cable axis [m]."""
        d = np.full(np.shape(x), np.inf, dtype=float)
        for cx, cy in self.centers:
            d = np.minimum(d, np.hypot(x - cx, y - cy))
        return d

    def inside_cable(self, x, y) -> np.ndarray:
        """Boolean mask: point lies inside any cable."""
        return self.distance_to_axes(x, y) <= self.r_cable

    def conductor_mask(self, x, y, cable: int | None = None) -> np.ndarray:
        """Boolean mask: point lies inside a conductor (the rated region)."""
        r = self.design.r_conductor
        if cable is None:
            return self.distance_to_axes(x, y) <= r
        cx, cy = self.centers[cable]
        return np.hypot(x - cx, y - cy) <= r

    def _in_trench(self, x, y) -> np.ndarray:
        bf = self.installation.backfill
        return ((np.abs(x) <= self.trench_x_half)
                & (y <= bf.y_top) & (y >= self.trench_y_bottom))

    def _in_plate(self, x, y) -> np.ndarray:
        bf = self.installation.backfill
        return ((np.abs(x) <= self.trench_x_half)
                & (y <= bf.plate_y_top)
                & (y >= bf.plate_y_top - bf.plate_thickness))

    @property
    def trench_x_half(self) -> float:
        """Trench half-width [m], derived from the layout plus the margin."""
        bf = self.installation.backfill
        c = self.centers
        return float(np.abs(c[:, 0]).max() + self.r_cable + bf.margin)

    @property
    def trench_y_bottom(self) -> float:
        """Trench floor [m], derived from the layout plus the margin."""
        bf = self.installation.backfill
        c = self.centers
        return float(c[:, 1].min() - self.r_cable - bf.margin)

    def _material_field(self, x, y, layer_values: np.ndarray,
                        soil_value: float, fill_value: float | None,
                        plate_value: float | None) -> np.ndarray:
        """
        Vectorised material lookup. Precedence, outermost first: native soil →
        trench backfill → protective plate → cable layers. The cable always
        wins, so a cable sitting inside the trench is still resolved layer by
        layer.
        """
        out = np.full(np.shape(x), soil_value, dtype=float)
        if self.installation.backfill is not None:
            bf = self.installation.backfill
            if fill_value is not None:
                out = np.where(self._in_trench(x, y), fill_value, out)
            if plate_value is not None and bf.plate_k is not None:
                out = np.where(self._in_plate(x, y), plate_value, out)
        d = self.distance_to_axes(x, y)
        r = self.design.r_bounds
        # Walk the stack from the OUTSIDE in, so the innermost layer that
        # contains the point is the one finally written.
        for i in range(len(r) - 1, -1, -1):
            out = np.where(d <= r[i], layer_values[i], out)
        return out

    def k_field(self, x, y) -> np.ndarray:
        """Thermal conductivity field [W/(m·K)]."""
        bf = self.installation.backfill
        return self._material_field(
            x, y,
            np.array([L.k for L in self.design.layers]),
            self.installation.k_soil,
            None if bf is None else bf.k,
            None if bf is None else bf.plate_k)

    def rho_c_field(self, x, y) -> np.ndarray:
        """Volumetric heat capacity field [J/(m³·K)] — transient model only."""
        bf = self.installation.backfill
        return self._material_field(
            x, y,
            np.array([L.rho_c for L in self.design.layers]),
            self.installation.rho_c_soil,
            None if bf is None else bf.rho_c,
            None if bf is None else bf.plate_rho_c)

    def source_densities(self, theta: float | None = None) -> dict:
        """
        Volumetric source densities [W/m³]: the FEM needs W/m³, the loss tables
        give W/m, and the two loss bands have different areas.
        """
        w = self.losses(theta)
        out = {}
        for kind, key in (('conductor', 'W_c'), ('screen', 'W_s')):
            band = self.design.heat_band(kind)
            if band is None:
                out[kind] = (0.0, 0.0, 0.0)
                continue
            r_in, r_out = band
            area = math.pi * (r_out ** 2 - r_in ** 2)
            out[kind] = (w[key] / area, r_in, r_out)
        return out

    def q_field(self, x, y, cable: int | None = None,
                theta: float | None = None) -> np.ndarray:
        """
        Volumetric heat source field [W/m³]. *cable* restricts the source to a
        single cable, which is what the self/mutual decomposition needs.
        """
        dens = self.source_densities(theta)
        centers = self.centers if cable is None else self.centers[[cable]]
        q = np.zeros(np.shape(x), dtype=float)
        for cx, cy in centers:
            d = np.hypot(x - cx, y - cy)
            for qv, r_in, r_out in dens.values():
                if qv == 0.0:
                    continue
                band = (d <= r_out) if r_in <= 0.0 else ((d > r_in) & (d <= r_out))
                q = np.where(band, q + qv, q)
        return q

    # --- reporting ----------------------------------------------------------
    def report(self) -> str:
        """Multi-line description of the resolved geometry and loading."""
        w = self.losses()
        inst, des = self.installation, self.design
        c = self.centers
        L = [f" case      : {self.label}",
             f" cable     : {des.name}",
             f"             r_outer {1e3 * self.r_cable:.2f} mm, "
             f"r_conductor {1e3 * des.r_conductor:.2f} mm, "
             f"r_insulation {1e3 * des.r_insulation:.2f} mm",
             des.table(),
             f" layout    : {inst.formation}, {self.n_cables} cables, "
             f"depth {inst.depth} m"
             + (f", spacing {inst.spacing} m" if inst.formation == 'flat' else ""),
             "             " + "  ".join(f"L{i+1}({x:+.3f},{y:+.3f})"
                                          for i, (x, y) in enumerate(c)),
             f" soil      : k {inst.k_soil:.3g} W/(m·K) "
             f"(rho {1.0 / inst.k_soil:.3g} K·m/W), "
             f"rho·c {1e-6 * inst.rho_c_soil:.2f} MJ/(m³·K)",
             f" boundaries: deep earth {inst.t_deep:.1f} °C, air "
             f"{inst.t_air:.1f} °C, h {inst.h_conv:.1f} W/(m²·K)",
             f" domain    : x ±{inst.x_half} m, y {inst.y_deep} … 0 m"]
        if inst.backfill is not None:
            bf = inst.backfill
            L.append(f" trench    : k {bf.k:.2f} W/(m·K) over x ±"
                     f"{self.trench_x_half:.3f} m, y {self.trench_y_bottom:.3f} "
                     f"… {bf.y_top:.3f} m"
                     + ("" if bf.plate_k is None
                        else f"; plate k {bf.plate_k:.2f}, "
                             f"{1e3 * bf.plate_thickness:.0f} mm at {bf.plate_y_top} m"))
        else:
            L.append(" trench    : none (uniform native soil to the cable surface)")
        L += [f" loading   : I = {self.current:.1f} A, losses evaluated at "
              f"{w['theta']:.0f} °C",
              f"             W_c {w['W_c']:.3f} W/m, W_s {w['W_s']:.3f} W/m, "
              f"lambda1 {w['lambda1']:.4f}, W_total {w['W_total']:.3f} W/m",
              f"             R_ac(20 °C) {1e3 * des.R_ac_20:.4f} Ω/km, "
              f"alpha {des.alpha_R:.4f} 1/K"]
        return "\n".join(L)


def cigre_mv_eu_case(current: float = 200.0,
                     backfill: Backfill | str | None = None,
                     **installation_kw) -> CableCase:
    """
    The default case: CIGRE TB 575 European-MV benchmark underground feeder.

    Three NA2XS2Y 120 mm² Al cables laid **flat**, axis depth a = 0.7 m,
    spacing b = 0.3 m (Fig. 6.6, Table 6.9), on a 20 kV / 50 Hz feeder.

    :param current: phase current [A].
    :param backfill: a :class:`Backfill`, a key of :data:`BACKFILL_CASES`, or
        ``None`` for uniform native soil.
    :param installation_kw: overrides forwarded to :class:`Installation`
        (``k_soil``, ``t_deep``, ``h_conv``, …).
    """
    if isinstance(backfill, str):
        backfill = BACKFILL_CASES[backfill]
    kw = {'formation': 'flat', 'depth': 0.7, 'spacing': 0.3, 'n_cables': 3}
    kw.update(installation_kw)
    inst = Installation(backfill=backfill, **kw)
    benchmark = (kw['depth'] == 0.7 and kw['spacing'] == 0.3
                 and kw['formation'] == 'flat')
    return CableCase(
        design=NA2XS2Y_120_AL, installation=inst, current=current,
        label=("CIGRE TB 575 European MV benchmark — NA2XS2Y 120 mm², "
               f"{kw['formation']} a = {kw['depth']} m, b = {kw['spacing']} m"
               + ("" if benchmark else "  [layout CHANGED from the benchmark]")
               + (f", {backfill_label(backfill)}" if backfill else "")
               + f", I = {current:.0f} A"))


def backfill_label(bf: Backfill) -> str:
    """Short description of a backfill, for case labels."""
    for name, cand in BACKFILL_CASES.items():
        if cand == bf:
            return f"trench '{name}'"
    return f"trench k = {bf.k:.2f} W/(m·K)"


# =============================================================================
# 2. FINITE-ELEMENT THERMAL SIMULATION
# =============================================================================
# Governing equation (steady-state heat conduction over the cross-section):
#
#       -div( k(x,y) grad(theta) ) = q_v(x,y)                    in Omega
#
# Weak form: find theta in V such that for all v in V
#
#       INT_Omega k grad(theta).grad(v) dOmega
#     + INT_Gamma_top h theta v dGamma                    <- Robin (convection)
#     = INT_Omega q_v v dOmega + INT_Gamma_top h t_air v dGamma
#
# Boundary conditions
#   ground surface y = 0            Robin / Newton convection to air
#   sides and bottom                Dirichlet, undisturbed deep-earth temperature
#
# Discretisation  linear triangles (ElementTriP1)
# Linear solver   sparse LU (scipy splu / spsolve)


@dataclass
class SteadyResult:
    """
    Outcome of :meth:`ThermalFEM.steady_metrics`.

    :param t_conductor: area-mean conductor temperature of every cable [°C].
    :param t_node_e: area-mean temperature on a thin annulus at the θ_e split
        radius, hottest cable [°C] — a direct FEM measurement of pydae's
        external node.
    :param t_soil: temperature the hottest cable's conductor region would sit
        at with the circuit de-energised [°C]. This is the ``t_soil`` input of
        ``line_dtr_cable``, *not* the deep-earth boundary value: it already
        contains the surface-convection gradient down to burial depth.
    :param t_surface_max: peak ground-surface temperature [°C].
    :param z_hot: conductor rise per unit total per-cable loss [K·m/W] on the
        hottest cable, all cables energised.
    :param losses: the loss split used (see :meth:`CableCase.losses`).
    :param energy_balance: (surface outflow + deep-earth outflow − injected)
        / injected [-]. Guards the assembly and the linear solve.
    :param source_error: relative error of the assembled load vector against
        the analytic total power [-]. Measures how well the mesh resolves the
        thin loss bands, and is the check that actually moves with refinement.
    :param surface_fraction: share of the injected power that leaves through
        the ground surface rather than the deep-earth boundary [-].
    """

    t_conductor: np.ndarray
    t_node_e: float
    t_soil: float
    t_surface_max: float
    z_hot: float
    losses: dict
    energy_balance: float
    source_error: float = 0.0
    surface_fraction: float = 0.0
    hot_index: int = 0

    def __str__(self) -> str:
        w = self.losses
        rows = [f" conductor temperatures : "
                + ", ".join(f"L{i+1} {t:.2f} °C" for i, t in
                            enumerate(self.t_conductor))
                + f"   (hottest: L{self.hot_index + 1})",
                f" theta_e (split node)   : {self.t_node_e:.2f} °C",
                f" no-load soil at cable  : {self.t_soil:.2f} °C",
                f" peak ground surface    : {self.t_surface_max:.2f} °C",
                f" conductor rise         : "
                f"{self.t_conductor[self.hot_index] - self.t_soil:.2f} K at "
                f"{w['W_total']:.2f} W/m per cable",
                f" Z_hot                  : {self.z_hot:.5f} K·m/W",
                f" heat out via surface   : {1e2 * self.surface_fraction:.1f} % "
                f"(rest through the deep-earth boundary)",
                f" source quadrature err  : {1e2 * self.source_error:+.4f} %",
                f" energy balance err     : {1e2 * self.energy_balance:+.3e} %"]
        return "\n".join(rows)


class ThermalFEM:
    """
    2D thermal finite-element model of a :class:`CableCase`.

    The mesh and every assembled matrix are built once, on first use, and
    cached: the ladder identification factorises the stiffness matrix a dozen
    times over, so re-meshing per call would dominate the runtime.

    :param case: the case to solve.
    :param in_cable_passes: extra adaptive refinement passes confined to the
        cable interior. The semiconducting screens (0.8 mm) and the bedding
        (1.1 mm) are far thinner than the insulation; at the base resolution
        they are under-resolved, which under-predicts the conductor by ~0.5 K.
        Two passes take the cable interior to ~0.11 mm. Use 1 when the extra
        0.1 K does not justify roughly tripling the element count.
    :param refine_factors: radii of the adaptive refinement passes, as
        multiples of the cable radius. Every element whose centroid falls
        inside a radius is split in that pass.
    :param intorder: quadrature order. 4 (six Gauss points per triangle) makes
        the ``np.where`` material mapping resolve the curved cable boundary
        well inside the elements the circles cut.
    """

    #: Default refinement schedule, in multiples of the cable radius. Scaled to
    #: the cable so a compact trefoil and a 0.3 m-spaced flat row end up
    #: equally well resolved (final element size ~ r/40 inside the cables).
    REFINE_FACTORS: tuple[float, ...] = (35., 20., 11., 6.5, 4.0, 2.6)
    IN_CABLE_FACTORS: tuple[float, ...] = (1.25, 1.15, 1.08)

    def __init__(self, case: CableCase, in_cable_passes: int = 2,
                 refine_factors: Sequence[float] | None = None,
                 intorder: int = 4):
        self.case = case
        self.in_cable_passes = int(in_cable_passes)
        self.refine_factors = tuple(self.REFINE_FACTORS if refine_factors is None
                                    else refine_factors)
        self.intorder = int(intorder)
        self._cache: dict = {}

    # --- 2.1 mesh -----------------------------------------------------------
    @property
    def mesh(self):
        """The triangular mesh (built and cached on first access)."""
        if 'mesh' not in self._cache:
            self._cache['mesh'] = self._build_mesh()
        return self._cache['mesh']

    def _build_mesh(self):
        """
        Coarse in the far field, very fine around the cables, with no external
        mesher:

        (a) start from a graded structured tensor grid — a fine band spanning
            the cable zone, stretched towards the far-field boundaries;
        (b) apply several passes of *local* adaptive refinement, each splitting
            only the elements whose centroid lies within a shrinking radius of
            the cable axes. scikit-fem's ``refined(elements)`` is conforming
            (red-green), so no hanging nodes appear.

        Trench and plate edges are inserted straight into the tensor grid so
        element edges lie exactly on the material interfaces — far cheaper and
        more accurate than refining until the jump is resolved.
        """
        skfem = _require_skfem()
        case, inst = self.case, self.case.installation
        c, r = case.centers, case.r_cable

        # (a) graded tensor grid, sized from the actual cable bounding box so
        #     the same code serves a flat row and a compact trefoil.
        bx = max(0.8, 2.0 * np.abs(c[:, 0]).max() + 8.0 * r)
        by = c[:, 1].min() - max(0.8, 8.0 * r)
        h_band = max(0.02, 1.5 * r)

        xs_parts = [np.linspace(inst.x_half * -1.0, -bx, 13),
                    np.linspace(-bx, bx, int(2 * bx / h_band) + 1),
                    np.linspace(bx, inst.x_half, 13)]
        ys_parts = [np.linspace(inst.y_deep, by, 13),
                    np.linspace(by, 0.0, int(-by / h_band) + 1)]

        if inst.backfill is not None:
            bf = inst.backfill
            xh = case.trench_x_half
            xs_parts.append(np.array([-xh, xh]))
            ys_parts.append(np.array([bf.y_top, case.trench_y_bottom]))
            if bf.plate_k is not None:
                ys_parts.append(np.array([bf.plate_y_top,
                                          bf.plate_y_top - bf.plate_thickness]))

        xs = np.unique(np.concatenate(xs_parts))
        ys = np.unique(np.concatenate(ys_parts))
        mesh = skfem.MeshTri.init_tensor(xs, ys)

        # (b) local adaptive refinement
        radii = [f * r for f in self.refine_factors]
        radii += [f * r for f in self.IN_CABLE_FACTORS][:self.in_cable_passes]
        for radius in radii:
            cx = mesh.p[0, mesh.t].mean(axis=0)      # element centroids
            cy = mesh.p[1, mesh.t].mean(axis=0)
            targets = np.nonzero(case.distance_to_axes(cx, cy) < radius)[0]
            if targets.size:
                mesh = mesh.refined(targets)
        return mesh

    # --- 2.2 assembly -------------------------------------------------------
    @property
    def assembly(self) -> dict:
        """
        Assembled operators, cached. Keys:

        ``basis``, ``top_basis``, ``K`` (conduction **plus** surface
        convection), ``M`` (heat capacity), ``f_source`` (Joule load),
        ``f_conv`` (Robin load), ``dirichlet``, ``interior``.

        The surface convection belongs in ``K``: it is a conductance to
        ambient, not a source.
        """
        if 'asm' not in self._cache:
            self._cache['asm'] = self._assemble()
        return self._cache['asm']

    def _assemble(self) -> dict:
        skfem = _require_skfem()
        from skfem import (Basis, FacetBasis, ElementTriP1, BilinearForm,
                           LinearForm, asm)
        from skfem.helpers import dot, grad
        from scipy.sparse import csc_matrix

        case, inst = self.case, self.case.installation
        mesh = self.mesh
        element = ElementTriP1()
        basis = Basis(mesh, element, intorder=self.intorder)

        @BilinearForm
        def conduction(u, v, w):
            return case.k_field(w.x[0], w.x[1]) * dot(grad(u), grad(v))

        @BilinearForm
        def heat_capacity(u, v, w):
            return case.rho_c_field(w.x[0], w.x[1]) * u * v

        @LinearForm
        def joule_source(v, w):
            return case.q_field(w.x[0], w.x[1]) * v

        @BilinearForm
        def convection_matrix(u, v, w):
            return inst.h_conv * u * v

        @LinearForm
        def convection_load(v, w):
            return inst.h_conv * inst.t_air * v

        K = asm(conduction, basis)
        M = asm(heat_capacity, basis)
        f_source = asm(joule_source, basis)

        top = mesh.facets_satisfying(lambda x: np.isclose(x[1], 0.0))
        top_basis = FacetBasis(mesh, element, facets=top)
        K = K + asm(convection_matrix, top_basis)
        f_conv = asm(convection_load, top_basis)

        dirichlet = basis.get_dofs(
            lambda x: (np.isclose(x[0], -inst.x_half)
                       | np.isclose(x[0], inst.x_half)
                       | np.isclose(x[1], inst.y_deep)))
        interior = np.setdiff1d(np.arange(basis.N), dirichlet.flatten())

        return {'basis': basis, 'top_basis': top_basis,
                'K': csc_matrix(K), 'M': csc_matrix(M),
                'f_source': f_source, 'f_conv': f_conv,
                'dirichlet': dirichlet, 'interior': interior}

    @property
    def basis(self):
        """The P1 finite-element basis."""
        return self.assembly['basis']

    @property
    def n_elements(self) -> int:
        return int(self.mesh.t.shape[1])

    @property
    def n_dofs(self) -> int:
        return int(self.basis.N)

    # --- 2.3 steady solves --------------------------------------------------
    def steady(self, energised: bool = True) -> np.ndarray:
        """
        Nodal steady temperature field [°C].

        :param energised: with ``False`` the cables carry no current, giving
            the *no-load* field — soil sitting between the deep Dirichlet
            boundary and the convective surface. That field is the datum every
            temperature rise in this module is measured from, and its value at
            burial depth is ``line_dtr_cable``'s ``t_soil`` input.
        """
        key = f'steady_{bool(energised)}'
        if key in self._cache:
            return self._cache[key]
        from scipy.sparse.linalg import spsolve
        from skfem import condense

        a = self.assembly
        f = a['f_conv'] + (a['f_source'] if energised else 0.0)
        T = np.zeros(self.n_dofs)
        T[a['dirichlet']] = self.case.installation.t_deep
        K_c, f_c, _, interior = condense(a['K'], f, x=T, D=a['dirichlet'])
        T[interior] = spsolve(K_c.tocsr(), f_c)
        self._cache[key] = T
        return T

    # --- 2.4 functionals ----------------------------------------------------
    def region_functional(self, r_in: float, r_out: float,
                          cable: int | None = None,
                          restrict_interior: bool = False) -> np.ndarray:
        r"""
        Unit-sum load vector over the annulus ``r_in < r <= r_out`` of one
        cable (or every cable).

        Because the vector sums to one, ``vec @ T`` is exactly the
        **area-averaged temperature** of that region — so the same vector
        serves as a unit-power input *and* as an output functional. That
        collocation is what guarantees the identified impedance is a Stieltjes
        function, hence realisable as a passive ladder.

        :param restrict_interior: return only the interior (non-Dirichlet)
            entries, as the reduced-order machinery needs.
        """
        from skfem import LinearForm, asm
        centers = self.case.centers if cable is None else self.case.centers[[cable]]

        @LinearForm
        def indicator(v, w):
            inside = np.zeros(np.shape(w.x[0]), dtype=bool)
            for cx, cy in centers:
                d = np.hypot(w.x[0] - cx, w.x[1] - cy)
                inside |= (d <= r_out) & (d > r_in)
            return np.where(inside, 1.0, 0.0) * v

        vec = asm(indicator, self.basis)
        total = vec.sum()
        if total <= 0.0:
            raise ValueError(
                f"the annulus {1e3*r_in:.3f}–{1e3*r_out:.3f} mm captured no "
                "quadrature weight; widen it or refine the mesh "
                "(in_cable_passes)")
        vec = vec / total
        return vec[self.assembly['interior']] if restrict_interior else vec

    def conductor_functional(self, cable: int | None = None,
                             restrict_interior: bool = False) -> np.ndarray:
        """Unit-sum functional over a conductor — the rated hot-spot output."""
        return self.region_functional(0.0, self.case.design.r_conductor,
                                      cable, restrict_interior)

    def source_functional(self, cable: int | None = None,
                          restrict_interior: bool = False) -> np.ndarray:
        """
        The **actual** loss distribution of one cable, normalised to unit total
        power: W_c in the conductor plus W_s at the screen.

        This is what determines how much heat has to cross the insulation. A
        uniformly-smeared vector would get the internal drop wrong.
        """
        from skfem import LinearForm, asm
        case = self.case

        @LinearForm
        def density(v, w):
            return case.q_field(w.x[0], w.x[1], cable=cable) * v

        vec = asm(density, self.basis)
        vec = vec / vec.sum()
        return vec[self.assembly['interior']] if restrict_interior else vec

    def rise_system(self):
        r"""
        The transient descriptor system for the temperature **rise** above the
        no-load field, restricted to the free degrees of freedom:

        .. math:: M\,\dot\theta + K\,\theta = b\,P(t)

        Superposition splits the field into (no-load base) + (rise due to
        losses); the rise problem therefore has *homogeneous* Dirichlet data on
        the sides and bottom, and the driving-point impedance

        .. math:: Z(s) = c^{T}(K + sM)^{-1}b

        is well defined. Returns ``(K, M, b, interior)`` with ``b`` normalised
        so that ``b.sum() == 1`` (unit total injected power) and the total
        physical power returned alongside.
        """
        if 'rise' not in self._cache:
            a = self.assembly
            i = a['interior']
            K = a['K'][i][:, i]
            M = a['M'][i][:, i]
            b_full = a['f_source'][i]
            self._cache['rise'] = (K, M, b_full, i)
        return self._cache['rise']

    # --- 2.5 engineering metrics -------------------------------------------
    def steady_metrics(self, node_e_radius: float | None = None,
                       node_e_band: float = 3.0e-4) -> SteadyResult:
        """
        Reduce the two steady solves to the numbers the export needs.

        :param node_e_radius: radius [m] at which pydae's external node θ_e is
            measured. Defaults to the outer radius of the main dielectric —
            the physically meaningful place for "the insulation thermal
            resistance" to end. Note that ``line_dtr_cable``'s own geometric
            route puts it at ``(dc + 2·t_ins)/2``, which ignores the
            conductor screen; where θ_e sits changes only the T1/T4 *split*,
            never their sum, and hence never the conductor temperature.
        :param node_e_band: half-width [m] of the sampling annulus.
        """
        case = self.case
        T = self.steady(True)
        T0 = self.steady(False)
        losses = case.losses()
        r_e = case.design.r_insulation if node_e_radius is None else node_e_radius
        hot = case.hot_index

        t_cond = np.array([self.conductor_functional(i) @ T
                           for i in range(case.n_cables)])
        c_e = self.region_functional(max(r_e - node_e_band, 0.0),
                                     r_e + node_e_band, cable=hot)
        t_e = float(c_e @ T)
        t_soil = float(self.conductor_functional(hot) @ T0)
        surf = np.isclose(self.mesh.p[1], 0.0)
        t_surface_max = float(T[surf].max())

        w_total = losses['W_total']
        z_hot = (t_cond[hot] - t_soil) / w_total if w_total > 0.0 else 0.0

        # --- discretisation checks ------------------------------------------
        # (a) source quadrature: the assembled load vector must integrate to
        #     the analytic total power. This is the one that actually moves --
        #     it measures how well the mesh resolves the thin loss bands.
        # (b) global balance: injected power = convected out of the ground
        #     surface + conducted out through the deep-earth boundary. Exact
        #     by construction for a converged direct solve, so it guards the
        #     assembly and the solve rather than the mesh.
        a = self.assembly
        p_in = float(a['f_source'].sum())
        p_exact = case.n_cables * w_total
        source_error = ((p_in - p_exact) / p_exact) if p_exact > 0.0 else 0.0

        surface_out = float((a['K'] @ T - a['f_conv']).sum())
        residual = a['K'] @ T - a['f_source'] - a['f_conv']
        deep_out = -float(residual[a['dirichlet'].flatten()].sum())
        balance = ((surface_out + deep_out - p_in) / p_in) if p_in > 0.0 else 0.0

        return SteadyResult(t_conductor=t_cond, t_node_e=t_e, t_soil=t_soil,
                            t_surface_max=t_surface_max, z_hot=z_hot,
                            losses=losses, energy_balance=balance,
                            source_error=source_error,
                            surface_fraction=(surface_out / p_in
                                              if p_in > 0.0 else 0.0),
                            hot_index=hot)

    def analytic_T1(self, node_e_radius: float | None = None) -> float:
        r"""
        Insulation thermal resistance from the resolved layer stack [K·m/W]:

        .. math:: T_1 = \sum_{\rm layers} \frac{1}{2\pi k_i}
                        \ln\frac{r_{i,\rm out}}{r_{i,\rm in}}

        summed over the layers between the conductor surface and the θ_e node.

        This is **exact** for radial conduction through concentric annuli, so
        there is nothing for the FEM to add: the conductor is nearly isothermal
        (its internal rise is ``W_c/(4πk) ≈ 0.007 K`` for aluminium), and the
        field between the conductor and θ_e is radial to within the small
        asymmetry that mutual heating introduces. Taking T1 from here and
        leaving *all* the FEM information in T4 avoids sampling a 0.6 mm-wide
        annulus on a 0.1 mm mesh, which is the numerically fragile part.
        ``SteadyResult.t_node_e`` provides the FEM measurement as a cross-check.
        """
        return self._radial_resistance(self.case.design.r_conductor,
                                       self.case.design.r_insulation
                                       if node_e_radius is None
                                       else node_e_radius)

    def analytic_T4(self, node_e_radius: float | None = None,
                    cable: int | None = None) -> dict:
        r"""
        Independent analytic estimate of ``T4`` [K·m/W], for cross-checking the
        FEM — built from textbook pieces that share no code with stage 2:

        ``outer_layers``
            radial conduction from the θ_e node out to the cable surface,
            :math:`\sum \ln(r_{\rm out}/r_{\rm in})/2\pi k`.
        ``soil_self`` / ``soil_mutual``
            the Kennelly image-method row of :meth:`CableCase.kennelly_matrix`,
            so the mutual heating of the other phases is included.

        The surface convective film does **not** appear, even though the FEM
        imposes a Robin boundary while Kennelly assumes an isothermal plane:
        both quantities are a *rise above the undisturbed no-load field*, and
        that datum already carries the film drop. (The cancellation weakens at
        very small ``h``, where the film distorts the shape of the loaded
        field and not merely its offset.)

        For the CIGRE default the total lands within ~2 % of the FEM ``T4`` --
        a strong independent check, since nothing here touches the mesh, the
        assembly or the reduction. What remains is mostly the finite domain
        (Dirichlet at 5 m rather than infinity) and the fact that the two-node
        model drives ALL of W_total across T4 whereas the FEM injects the
        screen loss part-way out.

        :returns: a dict of the pieces plus ``'total'``.
        """
        case = self.case
        design = case.design
        hot = case.hot_index if cable is None else int(cable)
        r_e = design.r_insulation if node_e_radius is None else node_e_radius

        outer = self._radial_resistance(r_e, design.r_outer)
        Z = case.kennelly_matrix()
        self_z = float(Z[hot, hot])
        mutual = float(Z[hot].sum() - Z[hot, hot])

        return {'outer_layers': outer, 'soil_self': self_z,
                'soil_mutual': mutual,
                'total': outer + self_z + mutual}

    def _radial_resistance(self, r_from: float, r_to: float) -> float:
        """Series radial thermal resistance of the layers between two radii."""
        design = self.case.design
        total = 0.0
        r_prev = r_from
        r_e = r_to
        for L, r_out in zip(design.layers, design.r_bounds):
            if r_out <= r_from + 1e-15:
                continue
            r_hi = min(r_out, r_e)
            if r_hi <= r_prev + 1e-15:
                continue
            total += math.log(r_hi / r_prev) / (2.0 * math.pi * L.k)
            r_prev = r_hi
            if r_hi >= r_e - 1e-15:
                break
        return total


# =============================================================================
# 3. LADDER (CAUER) MODEL IDENTIFICATION
# =============================================================================
# The semi-discrete heat equation from stage 2 is the descriptor system
#
#       M dT/dt + K T = b P(t)
#
# so the thermal impedance seen from the loss distribution b, observed through
# the functional c, is
#
#       Z(s) = c^T (K + s M)^-1 b .
#
# COLLOCATED case (c = b): K and M are symmetric positive definite, which makes
# Z(s) a Stieltjes function -- exactly the class realisable as an RC ladder with
# strictly positive elements. A valid Cauer network is then guaranteed to exist,
# with no curve fitting anywhere in the derivation.
#
# HOT-SPOT case (c = the centre cable's conductor, b = all three cables): this
# is a NON-collocated transfer impedance and carries no a-priori guarantee of
# positive residues, hence no guarantee that a passive ladder exists. The code
# therefore CHECKS positivity numerically and reports it, rather than assuming
# it. When the check fails the Foster form is still a valid linear model -- use
# it as a state-space block rather than as a physical network.
#
# Reduction is by RATIONAL KRYLOV projection. Plain Lanczos about s = 0 matches
# moments only at DC, which is accurate at long times but poor across the six
# decades a buried cable actually spans. Interpolating at a spread of real
# shifts s_j = 1/tau_j cuts the order-4 error from ~14 % to under 4 %. Including
# the s = 0 vector K^-1 b in the basis makes the steady-state resistance EXACT
# regardless of where the other shifts are placed. Projecting a symmetric
# positive definite pencil keeps (K_r, M_r) symmetric positive definite, so the
# reduced Z(s) is still Stieltjes in the collocated case.


#: Default interpolation time constants: the loading cycles a cable engineer
#: cares about, ~10 s (insulation) to ~116 days (soil). The fast end has to
#: reach 10 s for the LAYERED model — resolving the insulation adds a ~190 s
#: stage that a window starting at 100 s cannot interpolate.
DEFAULT_TAU_RANGE = (1.0e1, 1.0e7)

#: Order used for the "near-exact" reference model the two-node fit is matched
#: against. High enough that its own truncation error is well below the
#: two-node error it is used to measure.
REFERENCE_ORDER = 13


def foster_to_cauer(R_f: np.ndarray, tau_f: np.ndarray):
    r"""
    Convert a Foster network :math:`\sum R_i/(1+s\tau_i)` into the Cauer ladder

    .. math:: Z(s) = \cfrac{1}{sC_1 + \cfrac{1}{R_1 + \cfrac{1}{sC_2 + \dots}}}

    by repeatedly removing the pole at infinity (extract a shunt C) and the
    residual constant (extract a series R).

    Time is rescaled by the dominant constant so the polynomial arithmetic
    stays conditioned. At each removal the leading coefficient cancels
    *exactly* by construction, so it is sliced away rather than trusted to
    floating-point cancellation.

    :returns: ``(R, C)`` ordered from the cable outwards — ``R[-1]`` is the
        resistor that returns to ambient, and ``R.sum()`` is the DC thermal
        resistance.
    """
    R_f, tau_f = np.asarray(R_f, float), np.asarray(tau_f, float)
    scale = tau_f.max()
    tau = tau_f / scale
    n = len(R_f)

    # Z(s) = N(s)/D(s) from the partial fractions, descending coefficients.
    # deg D = n and deg N = n-1, so Z is strictly proper, as an RC impedance
    # driven by a capacitor must be.
    D = np.array([1.0])
    for t in tau:
        D = np.convolve(D, np.array([t, 1.0]))
    N = np.zeros(n)
    for i in range(n):
        Di = np.array([1.0])
        for j in range(n):
            if j != i:
                Di = np.convolve(Di, np.array([tau[j], 1.0]))
        N = N + R_f[i] * Di

    R_c, C_c = [], []
    for _ in range(n):
        # remove the pole at infinity of Y = D/N  ->  shunt capacitance
        c = D[0] / N[0]
        D = (D - c * np.concatenate([N, [0.0]]))[1:]
        C_c.append(c * scale)                       # undo the time rescaling
        # remove the constant of Z = N/D  ->  series resistance
        r = N[0] / D[0]
        N = (N - r * D)[1:]
        R_c.append(r)

    return np.array(R_c), np.array(C_c)


@dataclass
class LadderModel:
    r"""
    An n-stage RC ladder identified from the FEM, plus the Foster form it came
    from and the steady-state self/mutual decomposition.

    Topology::

        P --+--[R1]--+--[R2]--+-- ... --+--[R_{n-1}]--+--[Rn]--+
            |        |        |         |             |        |
           C1       C2       C3        ...           Cn        |
            |        |        |         |             |        |
            +--------+--------+---------+-------------+--------+  ambient

    State :math:`T_i` is the temperature **rise** of node i above the no-load
    field [K]; the input is the loss driving the ladder [W/m]; the output is
    :math:`T_1`, the conductor rise.

    :param R: series thermal resistances [K·m/W], cable outwards.
    :param C: shunt thermal capacitances [J/(m·K)], cable outwards.
    :param R_foster: Foster residues [K·m/W].
    :param tau_foster: Foster time constants [s], slowest first.
    :param passive: whether every Foster residue came out positive, so the
        ladder is physically realisable. Checked, never assumed.
    :param z_self: hot cable's own steady rise per unit of its own loss [K·m/W].
    :param z_mutual: rise contributed by each *other* cable [K·m/W].
    :param z_outer: steady rise of a non-hottest cable, for reference [K·m/W].
    :param hot_index: which cable the model observes.
    :param power_basis: ``'per_cable'`` — the input is the loss of **one**
        cable, applied to all of them simultaneously (the balanced case).
    """

    R: np.ndarray
    C: np.ndarray
    R_foster: np.ndarray
    tau_foster: np.ndarray
    passive: bool
    z_self: float = 0.0
    z_mutual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z_outer: float = 0.0
    hot_index: int = 0
    power_basis: str = 'per_cable'

    # --- basic properties ---------------------------------------------------
    @property
    def n_stages(self) -> int:
        return len(self.R)

    @property
    def dc_resistance(self) -> float:
        """Steady thermal resistance [K·m/W] — exact, and equal to ``R.sum()``."""
        return float(np.sum(self.R_foster))

    @property
    def z_balanced(self) -> float:
        """``z_self + sum(z_mutual)`` — what a single ``T4`` has to carry."""
        return float(self.z_self + np.sum(self.z_mutual))

    @property
    def mutual_share(self) -> float:
        """Fraction of the hot cable's steady rise due to its neighbours [-]."""
        z = self.z_balanced
        return float(np.sum(self.z_mutual) / z) if z else 0.0

    # --- state space --------------------------------------------------------
    def matrices(self):
        r"""
        ``(A, b, c)`` of :math:`\dot T = A T + b P`, :math:`y = c^{T} T`.

        ``G`` is assembled exactly as a nodal-analysis conductance matrix:
        ``R[i]`` couples node i to node i+1 for i = 0 … n−2, and ``R[n−1]``
        grounds node n−1 to ambient. That last term — with no
        :math:`(T_n - T_{n+1})` partner, because ambient's rise is zero by
        definition — is what makes the steady state finite, and it is why
        ``sum(R)`` is the DC thermal resistance.
        """
        R, C = self.R, self.C
        if R.size == 0:
            raise ValueError("no passive ladder was realised; use the Foster "
                             "form (foster_step) instead")
        n = len(R)
        G = np.zeros((n, n))
        for i in range(n - 1):
            g = 1.0 / R[i]
            G[i, i] += g
            G[i + 1, i + 1] += g
            G[i, i + 1] -= g
            G[i + 1, i] -= g
        G[n - 1, n - 1] += 1.0 / R[n - 1]
        A = -np.diag(1.0 / C) @ G
        b = np.zeros(n)
        b[0] = 1.0 / C[0]
        c = np.zeros(n)
        c[0] = 1.0
        return A, b, c

    @property
    def time_constants(self) -> np.ndarray:
        """Eigenvalue time constants of the ladder [s], slowest first."""
        A, _, _ = self.matrices()
        tau = -1.0 / np.linalg.eigvals(A).real
        return np.sort(tau)[::-1]

    # --- responses ----------------------------------------------------------
    def impedance(self, omega) -> np.ndarray:
        r"""
        Thermal impedance :math:`Z(j\omega) = \sum_i R_i/(1+j\omega\tau_i)`
        [K·m/W per unit per-cable loss], from the Foster form.

        This is where the difference between model orders becomes legible. A
        pole responds to an excitation only while :math:`\omega\tau \ll 1`, so
        at a given period the model's *usable* thermal resistance is whatever
        sits behind poles fast enough to follow. A daily cycle
        (:math:`\omega = 2\pi/86400`) reaches roughly 40 % of this cable's DC
        resistance; a two-pole reduction, forced to put its poles at the
        extremes, reaches about half that.
        """
        omega = np.atleast_1d(np.asarray(omega, float))
        return (self.R_foster[None, :]
                / (1.0 + 1j * omega[:, None] * self.tau_foster[None, :])).sum(1)

    def foster_step(self, t) -> np.ndarray:
        """Exact step response of the Foster form [K·m/W] — the reference."""
        t = np.atleast_1d(np.asarray(t, float))
        return (self.R_foster[None, :]
                * (1.0 - np.exp(-t[:, None] / self.tau_foster[None, :]))).sum(1)

    def step(self, t) -> np.ndarray:
        """
        Step response of the **ladder**, integrated from its own state-space
        form [K·m/W per unit input power].

        This deliberately does not reuse the Foster poles: agreement with
        :meth:`foster_step` validates that the extracted ladder really is the
        same system, rather than restating the same numbers.
        """
        from scipy.linalg import expm
        t = np.atleast_1d(np.asarray(t, float))
        A, b, c = self.matrices()
        Ainv_b = np.linalg.solve(A, b)
        out = np.empty_like(t)
        for k, tk in enumerate(t):
            out[k] = c @ (expm(A * tk) @ Ainv_b - Ainv_b)
        return out

    def simulate(self, t, power: Callable[[float], float] | np.ndarray,
                 y0=None) -> np.ndarray:
        r"""
        Integrate :math:`\dot T = A T + b P(t)` for an arbitrary load profile,
        by zero-order hold on P over each step — exact for piecewise-constant
        P, which is how measured load profiles come.

        For dynamic rating the input is a current, not a power:
        :math:`P(t) = I(t)^2 R'_{\rm ph}`, so the system is linear in P but
        quadratic in I.

        :param t: time grid [s].
        :param power: callable ``P(t)`` [W/m], or an array aligned with *t*.
        :param y0: initial node rises [K]; default all zero (cold start).
        :returns: conductor rise at each time [K].
        """
        from scipy.linalg import expm
        t = np.asarray(t, float)
        A, b, c = self.matrices()
        n = A.shape[0]
        T = np.zeros(n) if y0 is None else np.asarray(y0, float).copy()
        p_arr = None if callable(power) else np.asarray(power, float)
        out = np.empty_like(t)
        out[0] = c @ T
        eye = np.eye(n)
        for k in range(1, len(t)):
            dt = t[k] - t[k - 1]
            Ad = expm(A * dt)
            Bd = np.linalg.solve(A, (Ad - eye) @ b)
            p = power(t[k - 1]) if p_arr is None else p_arr[k - 1]
            T = Ad @ T + Bd * p
            out[k] = c @ T
        return out

    # --- reporting ----------------------------------------------------------
    def table(self) -> str:
        """Stage-by-stage table of the ladder."""
        if self.R.size == 0:
            rows = [" Foster form only (no passive ladder realised):",
                    "  stage       R [K·m/W]        tau [s]"]
            for i, (r, tt) in enumerate(zip(self.R_foster, self.tau_foster), 1):
                rows.append(f"   {i:<6d}{r:+14.5f}   {tt:14.4e}")
            return "\n".join(rows)
        rows = ["  stage      R [K·m/W]      C [J/(m·K)]      R·C [s]"
                "     R·C [days]"]
        for i, (r, c) in enumerate(zip(self.R, self.C), 1):
            rows.append(f"   {i:<6d}{r:12.5f}    {c:12.4e}   {r * c:11.4e}  "
                        f"{r * c / 86400.0:9.3f}")
        rows.append(f"   SUM   {self.R.sum():12.5f}    {self.C.sum():12.4e}")
        return "\n".join(rows)

    def equations(self) -> str:
        """
        The scalar ODEs of the ladder, symbolically and with the identified
        numbers substituted — the model as an engineer would write it down.
        """
        n = self.n_stages
        R, C = self.R, self.C
        out = [" Symbolic:",
               "   C1 dT1/dt = P - (T1 - T2)/R1"]
        for i in range(2, n):
            out.append(f"   C{i} dT{i}/dt = (T{i-1} - T{i})/R{i-1} "
                       f"- (T{i} - T{i+1})/R{i}")
        out += [f"   C{n} dT{n}/dt = (T{n-1} - T{n})/R{n-1} - T{n}/R{n}",
                "   y = T1",
                "",
                " With the identified values (T in K, P in W/m, t in s):"]
        for i in range(n):
            terms = ["P"] if i == 0 else [f"(T{i} - T{i+1})/{R[i-1]:.5f}"]
            terms.append(f"- (T{i+1} - T{i+2})/{R[i]:.5f}" if i < n - 1
                         else f"- T{n}/{R[n-1]:.5f}")
            out.append(f"   {C[i]:.4e} dT{i+1}/dt = " + " ".join(terms))
        return "\n".join(out)

    def __str__(self) -> str:
        head = (f" {self.n_stages}-stage Cauer ladder, cable L"
                f"{self.hot_index + 1} conductor, input = loss "
                f"{self.power_basis.replace('_', ' ')}")
        tail = [f" sum R = {self.dc_resistance:.5f} K·m/W = DC thermal resistance",
                f" self {self.z_self:.5f} + mutual "
                f"{' + '.join(f'{v:.5f}' for v in self.z_mutual)} = "
                f"{self.z_balanced:.5f} K·m/W "
                f"({1e2 * self.mutual_share:.1f} % mutual)",
                f" other-cable rise {self.z_outer:.5f} K·m/W",
                " passive: " + ("yes (residues checked positive)" if self.passive
                                else "NO — Foster residues change sign")]
        return "\n".join([head, self.table()] + tail)


def identify_ladder(fem: ThermalFEM, n_stages: int = 5,
                    tau_range: tuple[float, float] = DEFAULT_TAU_RANGE,
                    cable: int | None = None,
                    collocated: bool = False) -> LadderModel:
    r"""
    Identify an n-stage Cauer ladder for the hot-spot temperature of one cable.

    * **input** — the real loss distribution (W_c in the conductor, W_s at the
      screen), unit total power per cable, applied to **every** cable. That is
      the balanced operating case, and it makes the resulting ladder absorb the
      mutual heating.
    * **output** — the area-mean **conductor** temperature of the observed
      cable, which is the quantity rated against the 90 °C XLPE limit, not the
      whole-cable mean and not the surface.

    :param fem: a built :class:`ThermalFEM`.
    :param n_stages: ladder order. The Krylov basis has ``n_stages`` vectors
        (one at s = 0 plus ``n_stages − 1`` shifts).
    :param tau_range: ``(tau_min, tau_max)`` [s] spanned by the interpolation
        shifts, log-spaced. The reduced model is near-exact at these times and
        degrades outside them; the steady state is exact regardless.
    :param cable: which cable to observe; defaults to
        :attr:`CableCase.hot_index`.
    :param collocated: observe the power-weighted mean over all cables instead
        of one conductor. Gives up the hot spot in exchange for a *guaranteed*
        passive realisation — useful as a sanity check on the method itself.
    """
    from scipy.sparse.linalg import splu

    if n_stages < 2:
        raise ValueError("n_stages must be at least 2")
    K, M, _b_full, interior = fem.rise_system()
    case = fem.case
    hot = case.hot_index if cable is None else int(cable)
    others = [i for i in range(case.n_cables) if i != hot]

    # per-cable source vectors, each of unit total power
    b_cab = [fem.source_functional(i, restrict_interior=True)
             for i in range(case.n_cables)]
    b_eq = np.sum(b_cab, axis=0)              # unit power in EACH cable
    c_hot = fem.conductor_functional(hot, restrict_interior=True)
    c_out = (fem.conductor_functional(others[0], restrict_interior=True)
             if others else c_hot)
    c_obs = b_eq / case.n_cables if collocated else c_hot

    # --- one-sided rational Krylov basis, seeded by the input ---------------
    taus = np.logspace(math.log10(tau_range[0]), math.log10(tau_range[1]),
                       n_stages - 1)
    lu0 = splu(K.tocsc())
    V = [lu0.solve(b_eq)]                      # s = 0  ->  exact DC
    for tau in taus:
        V.append(splu((K + (1.0 / tau) * M).tocsc()).solve(b_eq))
    Q, _ = np.linalg.qr(np.array(V).T)

    Kr = Q.T @ (K @ Q)
    Mr = Q.T @ (M @ Q)
    from scipy.linalg import eigh
    lam, X = eigh((Kr + Kr.T) / 2.0, (Mr + Mr.T) / 2.0)

    # Residues of a (possibly non-collocated) transfer: (c^T phi)(phi^T b).
    # For c = b this is a square, hence positive; otherwise the sign is not
    # fixed a priori, which is exactly why `passive` is checked below.
    y_c = X.T @ (Q.T @ c_obs)
    y_b = X.T @ (Q.T @ b_eq)
    R_f = y_c * y_b / lam
    tau_f = 1.0 / lam
    order = np.argsort(tau_f)[::-1]            # slowest stage first
    R_f, tau_f = R_f[order], tau_f[order]

    passive = bool(np.all(R_f > 0.0))
    if passive:
        try:
            R_c, C_c = foster_to_cauer(R_f, tau_f)
            if not (np.all(R_c > 0.0) and np.all(C_c > 0.0)):
                R_c = C_c = np.zeros(0)
                passive = False
        except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
            R_c = C_c = np.zeros(0)
            passive = False
    else:
        R_c = C_c = np.zeros(0)

    # --- exact steady self / mutual split (a few extra direct solves) -------
    z_self = float(c_hot @ lu0.solve(b_cab[hot]))
    z_mut = np.array([float(c_hot @ lu0.solve(b_cab[i])) for i in others])
    z_outer = float(c_out @ lu0.solve(b_eq))

    return LadderModel(R=R_c, C=C_c, R_foster=R_f, tau_foster=tau_f,
                       passive=passive, z_self=z_self, z_mutual=z_mut,
                       z_outer=z_outer, hot_index=hot)


def transient_fem_step(fem: ThermalFEM, targets, n_steps: int = 400,
                       cable: int | None = None) -> np.ndarray:
    r"""
    Independent ground truth: integrate :math:`M\dot T + KT = b` from
    :math:`T(0)=0` with uniform backward Euler, and return the hot cable's mean
    conductor rise at each target time [K·m/W per unit per-cable loss].

    This shares no machinery with the Krylov/Cauer path — no projection, no
    eigendecomposition, no continued fraction — so agreement is a genuine check
    rather than a restatement of the same approximation. One LU factorisation
    per target time, ``n_steps`` uniform steps each.
    """
    from scipy.sparse.linalg import splu
    K, M, _b, _i = fem.rise_system()
    case = fem.case
    hot = case.hot_index if cable is None else int(cable)
    b_eq = np.sum([fem.source_functional(i, restrict_interior=True)
                   for i in range(case.n_cables)], axis=0)
    c_hot = fem.conductor_functional(hot, restrict_interior=True)

    out = []
    for t_end in np.atleast_1d(np.asarray(targets, float)):
        dt = t_end / n_steps
        lu = splu((M + dt * K).tocsc())
        T = np.zeros(K.shape[0])
        rhs = dt * b_eq
        for _ in range(n_steps):
            T = lu.solve(M @ T + rhs)
        out.append(float(c_hot @ T))
    return np.array(out)


# =============================================================================
# 4. PARAMETERS FOR pydae.bps.lines.line_dtr_cable
# =============================================================================
# line_dtr_cable integrates the IEC 60853 TWO-NODE ladder
#
#     Q1 dTc/dt = (W_c + p W_d)          - (Tc - Te)/T1
#     Q2 dTe/dt = (Tc - Te)/T1 + (1-p) W_d + lambda1 W_c - (Te - Tsoil)/T4
#
# which is structurally the SAME ladder stage 3 derives, truncated to n = 2.
# So the export is a PARAMETER-FITTING problem, not a remodelling one:
#
#   T1  analytic, from the resolved layer stack (exact for radial conduction
#       through concentric annuli -- there is nothing for the FEM to add).
#   T4  from the FEM conductor temperature, so it embeds the mutual heating of
#       the neighbouring phases. This is precisely what the library's
#       geometry-derived default does NOT do: its T4 is the Kennelly result for
#       an ISOLATED cable.
#   Q1,Q2  least squares against the FEM step response. T1 and T4 are NOT
#       fitted -- they are pinned by the steady state, which the FEM gives
#       exactly, so the fit cannot drift away from the correct DC behaviour.
#   lambda1  from the tabulated resistance ratio, R_ph/R_ac - 1.
#   t_soil   the no-load field at burial depth, NOT the deep-earth boundary
#       value: it already contains the surface-convection gradient.
#
# Setting T1/T4/Q1/Q2 explicitly in the dtr block OVERRIDES the library's
# geometry route. That is deliberate: the geometry route cannot represent three
# thermally coupled cables.

#: How the two-node steady state relates the parameters (used by the fit and
#: by the checks): theta_c - theta_soil = W_c·T1 + W_total·T4, with
#: W_total = (1+lambda1)·W_c. Hence Z_hot = T1/(1+lambda1) + T4.


def _two_node_matrices(T1: float, T4: float, Q1: float, Q2: float,
                       lambda1: float, w_c_frac: float):
    r"""
    ``(A, b)`` of the two-node model, driven by **unit total per-cable loss**
    so its step response is directly comparable with the FEM reference.

    *w_c_frac* is the conductor share of the total loss, ``1/(1+lambda1)``.
    Dielectric loss is excluded here: it is a constant offset, not part of the
    current-driven step response.
    """
    A = np.array([[-1.0 / (Q1 * T1), 1.0 / (Q1 * T1)],
                  [1.0 / (Q2 * T1), -(1.0 / T1 + 1.0 / T4) / Q2]])
    b = np.array([w_c_frac / Q1, lambda1 * w_c_frac / Q2])
    return A, b


def _two_node_step(T1, T4, Q1, Q2, lambda1, w_c_frac, t) -> np.ndarray:
    """Conductor-rise step response of the two-node model [K·m/W]."""
    from scipy.linalg import expm
    A, b = _two_node_matrices(T1, T4, Q1, Q2, lambda1, w_c_frac)
    Ainv_b = np.linalg.solve(A, b)
    t = np.atleast_1d(np.asarray(t, float))
    out = np.empty_like(t)
    for k, tk in enumerate(t):
        out[k] = (expm(A * tk) @ Ainv_b - Ainv_b)[0]
    return out


@dataclass
class TwoNodeFit:
    """
    The complete parameter set for :mod:`pydae.bps.lines.line_dtr_cable`,
    together with everything needed to judge how good it is.

    :param T1: insulation thermal resistance [K·m/W].
    :param T4: external/soil thermal resistance [K·m/W], mutual heating of the
        neighbouring phases included.
    :param Q1: conductor-node thermal capacitance [J/(m·K)].
    :param Q2: external-node thermal capacitance [J/(m·K)].
    :param p: Van Wormer capacitance-split factor [-]. Inert while
        ``Wd == 0``, because it only ever multiplies the dielectric loss; kept
        so the block is complete and physically meaningful at HV.
    :param lambda1: sheath loss factor [-].
    :param Wd: dielectric loss [W/m].
    :param R_ac_20: conductor a.c. resistance at 20 °C [Ω/m].
    :param alpha_R: resistance temperature coefficient [1/K].
    :param K_dtr: network-current → conductor-current scaling [-]. 1.0 for one
        cable per phase; set it to ``1/n`` for *n* cables per phase sharing the
        line current.
    :param t_soil: ambient soil temperature at burial depth [°C].
    :param t_conductor_fem: FEM conductor temperature at the loss reference
        temperature [°C] — what the fitted model must reproduce.
    :param t_conductor_self_consistent: conductor temperature with the loss
        re-evaluated at that same temperature [°C] — what the pydae-bps model
        will actually converge to.
    :param err_step_max: largest absolute difference between the two-node and
        the reference FEM step response [K·m/W].
    :param err_kelvin: the same error expressed in kelvin at the case loading.
    :param z_hot: FEM steady conductor rise per unit per-cable loss [K·m/W].
    :param T1_fem_check: T1 as measured by sampling the FEM at the θ_e radius
        [K·m/W] — a cross-check on the analytic value, not the value used.
    :param dc_rescale: factor applied to the transient reference so its DC
        matches the (possibly finer-mesh) steady solve [-]. Departures from 1
        are the mesh-refinement difference between the two solves.
    :param T4_analytic: the Kennelly/layer-stack estimate of ``T4`` and its
        pieces, from :meth:`ThermalFEM.analytic_T4` — an independent check that
        shares no code with the FEM.
    :param T1_pinned: whether ``T1`` was held at its analytic value. With
        ``False`` it was fitted too (subject to the exact DC constraint), which
        roughly halves the transient error at the cost of a ``T1`` that no
        longer corresponds to a physical radius — see :func:`fit_two_node`.
    """

    T1: float
    T4: float
    Q1: float
    Q2: float
    p: float
    lambda1: float
    Wd: float
    R_ac_20: float
    alpha_R: float
    K_dtr: float
    t_soil: float
    t_conductor_fem: float = float('nan')
    t_conductor_self_consistent: float = float('nan')
    err_step_max: float = float('nan')
    err_kelvin: float = float('nan')
    z_hot: float = float('nan')
    T1_fem_check: float = float('nan')
    dc_rescale: float = 1.0
    T1_pinned: bool = True
    T4_analytic: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)

    # --- derived ------------------------------------------------------------
    @property
    def z_equivalent(self) -> float:
        """``T1 + (1+λ₁)·T4`` — steady rise per unit *conductor* loss [K·m/W]."""
        return self.T1 + (1.0 + self.lambda1) * self.T4

    @property
    def time_constants(self) -> np.ndarray:
        """The two eigenvalue time constants [s], slowest first."""
        A, _ = _two_node_matrices(self.T1, self.T4, self.Q1, self.Q2,
                                  self.lambda1, 1.0 / (1.0 + self.lambda1))
        return np.sort(-1.0 / np.linalg.eigvals(A).real)[::-1]

    def step(self, t) -> np.ndarray:
        """Conductor-rise step response per unit per-cable loss [K·m/W]."""
        return _two_node_step(self.T1, self.T4, self.Q1, self.Q2, self.lambda1,
                              1.0 / (1.0 + self.lambda1), t)

    def impedance(self, omega) -> np.ndarray:
        r"""
        Thermal impedance [K·m/W per unit per-cable loss] at angular frequency
        *omega*, for comparison with :meth:`LadderModel.impedance`.

        Driven by the total loss, so the two are directly comparable: the
        conductor share W_c enters node 1 and lambda1*W_c enters node 2, as
        the model specifies.
        """
        A, b = _two_node_matrices(self.T1, self.T4, self.Q1, self.Q2,
                                  self.lambda1, 1.0 / (1.0 + self.lambda1))
        omega = np.atleast_1d(np.asarray(omega, float))
        eye = np.eye(2)
        return np.array([np.linalg.solve(1j * w * eye - A, b)[0] for w in omega])

    def steady_conductor(self, w_c: float, t_soil: float | None = None) -> float:
        """Two-node steady conductor temperature [°C] at conductor loss *w_c*."""
        t_soil = self.t_soil if t_soil is None else t_soil
        return t_soil + w_c * self.T1 + (w_c * (1.0 + self.lambda1)
                                         + self.Wd) * self.T4

    # --- export -------------------------------------------------------------
    def as_dtr(self, mode: str = 'fitted', round_to: bool = True) -> dict:
        """
        The ``dtr`` block to drop into a pydae-bps ``lines[]`` entry (or into a
        ``dtr_codes`` entry, so one cable type can be reused by many lines).

        :param mode: ``'fitted'`` writes ``T1``/``T4``/``Q1``/``Q2``/``p``
            explicitly, which overrides the library's geometry route — the
            point of the whole exercise. ``'geometry'`` omits them, leaving
            ``line_dtr_cable`` to derive them analytically from the cable
            dimensions (isolated-cable T4, so hotter cables will read low).
        :param round_to: round to a sensible number of digits for pasting into
            a JSON/HJSON file.
        """
        if mode not in ('fitted', 'geometry'):
            raise ValueError("mode must be 'fitted' or 'geometry'")

        def r(v, nd):
            return round(float(v), nd) if round_to else float(v)

        dtr: dict = {'model': 'cable'}
        if mode == 'fitted':
            dtr.update({'T1': r(self.T1, 6), 'T4': r(self.T4, 6),
                        'Q1': r(self.Q1, 3), 'Q2': r(self.Q2, 1),
                        'p': r(self.p, 4)})
        dtr.update({'lambda1': r(self.lambda1, 4),
                    'R_ac_20': float(f"{self.R_ac_20:.4e}"),
                    'alpha_R': r(self.alpha_R, 5),
                    'Wd': r(self.Wd, 4),
                    'K_dtr': r(self.K_dtr, 4),
                    't_soil': r(self.t_soil, 2)})
        return dtr

    def __str__(self) -> str:
        tau = self.time_constants
        return "\n".join([
            f"   T1      = {self.T1:.6f} K·m/W    "
            + ("(insulation, analytic from the layer stack)" if self.T1_pinned
               else "(FITTED — a lumped split, not a physical radius)"),
            f"   T4      = {self.T4:.6f} K·m/W    (external, MUTUAL HEATING "
            f"INCLUDED)",
            f"   Q1      = {self.Q1:.4e} J/(m·K)",
            f"   Q2      = {self.Q2:.4e} J/(m·K)",
            f"   p       = {self.p:.4f}            (Van Wormer; inert while "
            f"Wd = {self.Wd:g})",
            f"   lambda1 = {self.lambda1:.4f}",
            f"   R_ac_20 = {self.R_ac_20:.4e} Ω/m   alpha_R = {self.alpha_R:g} 1/K",
            f"   K_dtr   = {self.K_dtr:g}",
            f"   t_soil  = {self.t_soil:.2f} °C",
            f"   time constants {tau[0]:.4g} s ({tau[0] / 86400:.2f} d) and "
            f"{tau[1]:.4g} s",
            f"   steady check : {self.t_soil:.2f} + "
            f"{self.losses.get('W_c', float('nan')):.2f}·{self.T1:.4f} + "
            f"{self.losses.get('W_total', float('nan')):.2f}·{self.T4:.4f} = "
            f"{self.steady_conductor(self.losses.get('W_c', float('nan'))):.2f} °C "
            f"(FEM {self.t_conductor_fem:.2f} °C)",
            f"   transient fit: max |two-node − FEM| = {self.err_step_max:.4f} "
            f"K·m/W = {self.err_kelvin:.2f} K at the case loading",
        ])


def fit_two_node(fem: ThermalFEM, steady: SteadyResult | None = None,
                 reference: LadderModel | None = None,
                 node_e_radius: float | None = None,
                 t_grid: np.ndarray | None = None,
                 K_dtr: float = 1.0, pin_T1: bool = True) -> TwoNodeFit:
    """
    Fit the ``line_dtr_cable`` two-node model to the FEM.

    Steady state first: ``T1`` analytically from the layer stack and ``T4``
    from the FEM conductor temperature, which makes the two-node steady state
    reproduce the FEM *exactly*. Then ``Q1`` and ``Q2`` by bounded least
    squares against the reference step response, with ``T1``/``T4`` held fixed
    so the DC behaviour cannot drift.

    The search runs in ``log(Q)``: an unbounded linear search drives ``Q1 → 0``
    and the state matrix blows up, so the starting point is the physical
    estimate (conductor + insulation for ``Q1``, a soil shell for ``Q2``) and
    the bounds are wide but finite.

    :param steady: reuse an existing :meth:`ThermalFEM.steady_metrics`.
    :param reference: reuse an existing high-order :class:`LadderModel` as the
        transient reference; one is identified at :data:`REFERENCE_ORDER` if
        not given.
    :param node_e_radius: where θ_e sits [m]; see
        :meth:`ThermalFEM.analytic_T1`.
    :param t_grid: times [s] the transient fit is evaluated on. Default 160
        points log-spaced over 1 s … 10⁸ s (~3 years), which covers everything
        from the insulation stage to full soil saturation.
    :param K_dtr: network- to conductor-current scaling written into the block.
    :param pin_T1: keep ``T1`` at its analytic value (the default), so that
        ``T1`` and ``T4`` mean what their names say and θ_e is the temperature
        at a real radius. With ``False``, ``T1`` becomes a third free parameter
        — still subject to ``T1/(1+λ₁) + T4 = Z_hot``, so the steady state
        stays exact — and the transient error roughly halves. The trade is
        honest but real: the internal node of a two-node reduction is a lumped
        fiction either way, and insisting it sit at the insulation boundary is
        what costs the transient accuracy. Use ``False`` when the block is
        wanted for dynamic rating and ``True`` when the parameters are to be
        read as IEC 60287 resistances.
    """
    from scipy.optimize import least_squares

    case = fem.case
    st = fem.steady_metrics(node_e_radius=node_e_radius) if steady is None else steady
    ref = (identify_ladder(fem, n_stages=REFERENCE_ORDER,
                           tau_range=(10 ** 0.5, 10 ** 7.5))
           if reference is None else reference)

    losses = st.losses
    lam = losses['lambda1']
    w_c, w_tot = losses['W_c'], losses['W_total']

    # --- steady state: T1 analytic, T4 carries all the FEM information ------
    T1_analytic = fem.analytic_T1(node_e_radius=node_e_radius)
    T1_fem = ((st.t_conductor[st.hot_index] - st.t_node_e) / w_c
              if w_c > 0.0 else float('nan'))
    T1 = T1_analytic
    T4 = st.z_hot - T1 / (1.0 + lam)
    if T4 <= 0.0:
        raise ValueError(
            f"T4 came out non-positive ({T4:.4g} K·m/W): the analytic T1 "
            f"({T1:.4g}) already exceeds the whole FEM conductor-to-soil drop "
            f"({st.z_hot:.4g}). Check the layer conductivities and the θ_e "
            "radius.")

    # --- transient: fit Q1, Q2 ----------------------------------------------
    t_ref = (np.logspace(0.0, 8.0, 160) if t_grid is None
             else np.asarray(t_grid, float))
    z_ref = ref.foster_step(t_ref)

    # The steady numbers may come from a finer mesh than the transient
    # reference (the ladder factorises the stiffness matrix a dozen times, so
    # it is usually run one refinement pass coarser). Rescale the reference
    # SHAPE onto the fine-mesh DC value rather than letting a constant offset
    # contaminate the capacitance fit.
    dc_ref = ref.dc_resistance
    dc_scale = (st.z_hot / dc_ref) if dc_ref > 0.0 else 1.0
    z_ref = z_ref * dc_scale

    w_c_frac = 1.0 / (1.0 + lam)

    # T1 and T4 are tied by the DC constraint T1/(1+lambda1) + T4 = Z_hot,
    # so the steady state is exact whatever the search does with them.
    def _split(t1):
        return t1, st.z_hot - t1 / (1.0 + lam)

    def residual(par):
        if pin_T1:
            q1, q2 = np.exp(par)
            t1, t4 = T1, T4
        else:
            t1, q1, q2 = np.exp(par)
            t1, t4 = _split(t1)
        return _two_node_step(t1, t4, q1, q2, lam, w_c_frac, t_ref) - z_ref

    design = case.design
    q1_0 = (math.pi * design.r_conductor ** 2 * design.layers[0].rho_c
            + math.pi * (design.r_insulation ** 2 - design.r_conductor ** 2)
            * 0.5 * design.layers[2].rho_c if len(design.layers) > 2
            else math.pi * design.r_conductor ** 2 * design.layers[0].rho_c)
    q2_0 = 3.0e6
    if pin_T1:
        x0 = np.log([q1_0, q2_0])
        lo, hi = np.log([1e1, 1e3]), np.log([1e7, 1e10])
    else:
        # T1 stays below (1+lambda1)·Z_hot so that T4 remains positive.
        t1_max = 0.999 * (1.0 + lam) * st.z_hot
        x0 = np.log([min(T1, t1_max), q1_0, q2_0])
        lo = np.log([1e-4, 1e1, 1e3])
        hi = np.log([t1_max, 1e7, 1e10])
    sol = least_squares(residual, x0, method='trf', bounds=(lo, hi),
                        xtol=1e-12, ftol=1e-12)
    if pin_T1:
        Q1, Q2 = np.exp(sol.x)
    else:
        T1, Q1, Q2 = np.exp(sol.x)
        T1, T4 = _split(T1)

    err = float(np.max(np.abs(residual(sol.x))))

    # --- Van Wormer factor, from the library's own geometry route ----------
    p_vw = library_defaults(case)['p']

    fit = TwoNodeFit(
        T1=float(T1), T4=float(T4), Q1=float(Q1), Q2=float(Q2), p=float(p_vw),
        lambda1=float(lam), Wd=float(design.Wd), R_ac_20=float(design.R_ac_20),
        alpha_R=float(design.alpha_R), K_dtr=float(K_dtr),
        t_soil=float(st.t_soil),
        t_conductor_fem=float(st.t_conductor[st.hot_index]),
        err_step_max=err, err_kelvin=err * w_tot, z_hot=float(st.z_hot),
        T1_fem_check=float(T1_fem), dc_rescale=float(dc_scale),
        T1_pinned=bool(pin_T1),
        T4_analytic=fem.analytic_T4(node_e_radius=node_e_radius),
        losses=dict(losses))
    fit.t_conductor_self_consistent = case.self_consistent_conductor_temperature(
        fit.z_equivalent, fit.t_soil)
    return fit


def ladder_dtr(ladder: LadderModel, fit: TwoNodeFit,
               round_to: bool = True) -> dict:
    """
    The **n-stage** ``dtr`` block for :mod:`pydae.bps.lines.line_dtr_cable`.

    Where :meth:`TwoNodeFit.as_dtr` writes ``T1``/``T4``/``Q1``/``Q2``, this
    writes the full ``R``/``C`` lists, which select the ladder form. The
    electrical data (``lambda1``, ``R_ac_20``, ``alpha_R``, ``Wd``, ``K_dtr``)
    and the soil datum come from the same fit, because they describe the cable
    rather than the thermal reduction.

    Use this when the transient matters: the two-node form reproduces the
    steady state exactly but flattens cyclic swings, and on the CIGRE MV
    benchmark it halves the daily conductor swing. The ladder does not.

    Note the ladder is driven by the **total** per-cable loss at node 1, and
    ``lambda1`` is carried only to compute that total — the identified R and C
    already contain the spatial loss split, because the FEM was driven with it.
    See the ``line_dtr_cable`` module docstring.

    :param round_to: round to eight significant digits, which keeps
        ``sum(R)`` — the DC thermal resistance — accurate to ~1e-8 K·m/W while
        staying readable in a JSON/HJSON file.
    :raises ValueError: if no passive ladder was realised, since a block with a
        negative R or C would not be a physical network.
    """
    if ladder.R.size == 0:
        raise ValueError(
            "this LadderModel has no passive realisation (the Foster residues "
            "changed sign), so there is no R/C block to export. Use the Foster "
            "form as a state-space block, or re-identify at a different order.")

    def fmt(v):
        return float(f"{float(v):.8g}") if round_to else float(v)

    def r(v, nd):
        return round(float(v), nd) if round_to else float(v)

    return {
        'model': 'cable',
        'R': [fmt(v) for v in ladder.R],
        'C': [fmt(v) for v in ladder.C],
        'lambda1': r(fit.lambda1, 4),
        'R_ac_20': float(f"{fit.R_ac_20:.4e}"),
        'alpha_R': r(fit.alpha_R, 5),
        'Wd': r(fit.Wd, 4),
        'K_dtr': r(fit.K_dtr, 4),
        't_soil': r(fit.t_soil, 2),
    }


def geometry_dtr(case: CableCase) -> dict:
    """
    Map the resolved FEM geometry onto ``line_dtr_cable``'s *geometry* keys, so
    the library's own analytic route can be driven from the same cable.

    The mapping is deliberately literal, which is what makes the comparison in
    :func:`library_defaults` meaningful: ``t_ins`` is the nominal insulation
    thickness (the library's ``De_ins = dc + 2·t_ins`` therefore ignores the
    semiconducting screens, exactly as IEC 60287 tables do), and ``rho_ins`` is
    the reciprocal conductivity of the main dielectric.
    """
    d, inst = case.design, case.installation
    ins = [L for L in d.layers if L.role == 'insulation'][0]
    cond = d.layers[0]
    return {
        'dc': 2.0 * d.r_conductor,
        't_ins': ins.thickness,
        'De': 2.0 * d.r_outer,
        'rho_ins': 1.0 / ins.k,
        'rho_soil': 1.0 / inst.k_soil,
        'L_depth': inst.depth,
        'cv_cond': cond.rho_c,
        'cv_ins': ins.rho_c,
        'cv_soil': inst.rho_c_soil,
        'R_ac_20': d.R_ac_20,
        'alpha_R': d.alpha_R,
        'lambda1': d.lambda1,
        'Wd': d.Wd,
    }


def library_defaults(case: CableCase) -> dict:
    """
    What ``line_dtr_cable``'s geometry route gives for this cable — computed by
    calling the library's own
    :func:`~pydae.bps.lines.line_dtr_cable._cable_thermal_network`, not by
    re-implementing its formulas, so the comparison cannot drift out of date.

    Returns ``T1``, ``T4``, ``Q1``, ``Q2``, ``p``.
    """
    from pydae.bps.lines.line_dtr_cable import _cable_thermal_network
    return _cable_thermal_network(geometry_dtr(case))


def compare_with_library(case: CableCase, fit: TwoNodeFit) -> str:
    """
    Side-by-side report of the fitted parameters against the library's
    geometry-derived defaults, and what the difference costs in kelvin.
    """
    lib = library_defaults(case)
    w = fit.losses
    t_lib = (15.0 + w['W_c'] * lib['T1']
             + w['W_c'] * lib['T4'])            # library default lambda1 = 0
    rows = [
        f"   T1      {lib['T1']:10.5f}  vs fitted {fit.T1:10.5f}   "
        "— XLPE only, no semiconducting screens",
        f"   T4      {lib['T4']:10.5f}  vs fitted {fit.T4:10.5f}   "
        "— ISOLATED cable: no mutual heating from the other two phases",
        f"   Q1      {lib['Q1']:10.4e}  vs fitted {fit.Q1:10.4e}",
        f"   Q2      {lib['Q2']:10.4e}  vs fitted {fit.Q2:10.4e}   "
        "— a 3·De soil shell, not the real semi-infinite response",
        f"   lambda1 {0.0:10.4f}  vs fitted {fit.lambda1:10.4f}   "
        "— screen losses dropped",
        f"   t_soil  {15.0:10.2f}  vs fitted {fit.t_soil:10.2f}   "
        "— fixed default, no surface-convection gradient",
        f"   => conductor {t_lib:.2f} °C instead of {fit.t_conductor_fem:.2f} °C "
        f"({fit.t_conductor_fem - t_lib:+.2f} K error)",
    ]
    return "\n".join(rows)


def bps_data(case: CableCase, fit: TwoNodeFit, km: float = 2.0,
             name: str = 'cable_dtr_fem', X_km: float = 0.12,
             Bs_km: float = 4.7e-5, S_base: float = 25e6,
             P_load_W: float = 10e6, Q_load_var: float = 2e6,
             U_kV: float = 20.0, mode: str = 'fitted',
             dtr: dict | None = None) -> dict:
    """
    A complete, minimal pydae-bps system data dict that exercises the fitted
    cable: two buses, one DTR line carrying the load, one slack source.

    The line's ``R_km`` is taken from the cable's own a.c. resistance so the
    electrical and thermal models describe the same conductor.

    :param km: line length [km].
    :param mode: forwarded to :meth:`TwoNodeFit.as_dtr`.
    :param dtr: use this ``dtr`` block verbatim instead of deriving one from
        *fit* — pass :func:`ladder_dtr` output here to build the n-stage form.
    """
    dtr = fit.as_dtr(mode=mode) if dtr is None else dict(dtr)
    return {
        'system': {'name': name, 'S_base': S_base, 'K_xif': 0.01},
        'buses': [{'name': '1', 'P_W': 0.0, 'Q_var': 0.0, 'U_kV': U_kV},
                  {'name': '2', 'P_W': P_load_W, 'Q_var': Q_load_var,
                   'U_kV': U_kV}],
        'lines': [{'bus_j': '1', 'bus_k': '2',
                   'R_km': 1e3 * case.design.R_ph, 'X_km': X_km,
                   'Bs_km': Bs_km, 'km': km, 'monitor': True,
                   'dtr': dtr}],
        'sources': [{'type': 'genape', 'bus': '1', 'S_n': 1e9, 'F_n': 50.0,
                     'X_v': 0.001, 'R_v': 0.0, 'K_delta': 0.001,
                     'K_alpha': 1e-6}],
    }


def verify_with_pydae(case: CableCase, fit: TwoNodeFit,
                      current: float | None = None,
                      run_seconds: float = 0.0) -> dict:
    """
    Close the loop: build the **real** pydae-bps model with the exported ``dtr``
    block, initialise it, force the line current to the case value, and check
    that the conductor temperature lands where the FEM says it should.

    The line current in the test system is set by the load flow, so it will not
    generally equal the case current. Rather than tune the load, the check is
    done on the *model's own* current: the two-node steady relation is
    evaluated at whatever ``I_cond`` the model settled on and compared with the
    temperature the model reports. That verifies the parameters were
    transported into the model correctly, which is what can actually go wrong.
    A second check compares the FEM-referenced temperature at the case current
    with :attr:`TwoNodeFit.t_conductor_self_consistent`.

    :param current: override the current used for the FEM-referenced check [A].
    :param run_seconds: if > 0, also integrate a soil-temperature step for this
        long and confirm the conductor follows it.
    :returns: a dict of the compared quantities, plus ``'ok'``.
    :raises ImportError: if pydae-bps or CasADi is unavailable.
    """
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    data = bps_data(case, fit)
    name = data['system']['name']
    grid = BpsBuilder(data, use_casadi=True)
    grid.construct(name)
    model = CasadiModel(CasadiBuilder(grid.sys_dict).build())
    model.ini({}, newton_tol=1e-10)

    g = lambda k: float(model.get_value(f'{k}_1_2'))
    t_c, t_e, t_soil = g('t_conductor'), g('t_sheath'), g('t_soil')
    w_c, i_cond = g('W_c'), g('I_cond')
    T1, T4, lam = g('T1'), g('T4'), g('lambda1')

    # (a) the model reproduces the two-node steady relation with OUR numbers
    t_c_expect = t_soil + w_c * T1 + w_c * (1.0 + lam) * T4
    err_relation = abs(t_c - t_c_expect)

    # (b) at the case current the model agrees with the self-consistent
    #     temperature derived from the FEM parameters
    i_case = case.current if current is None else current
    z_eq = T1 + (1.0 + lam) * T4
    t_c_case = replace(case, current=i_case).self_consistent_conductor_temperature(
        z_eq, t_soil)
    err_case = abs(t_c_case - fit.t_conductor_self_consistent)

    out = {'t_conductor': t_c, 't_sheath': t_e, 't_soil': t_soil,
           'W_c': w_c, 'I_cond': i_cond,
           'T1': T1, 'T4': T4, 'lambda1': lam,
           't_conductor_expected': t_c_expect,
           'err_steady_relation': err_relation,
           't_conductor_at_case_current': t_c_case,
           'err_vs_fem_self_consistent': err_case,
           'monotonic': bool(t_c > t_e > t_soil)}

    if run_seconds > 0.0:
        model.Dt = max(1.0, run_seconds / 360.0)
        model.run(run_seconds, {'t_soil_1_2': t_soil + 10.0})
        model.post()
        out['t_conductor_after_soil_step'] = float(
            model.get_value('t_conductor_1_2'))
        out['responds_to_soil_step'] = bool(
            out['t_conductor_after_soil_step'] > t_c)

    # err_case is dominated by the ROUNDING of the exported block (T1/T4 to
    # 6 decimals, t_soil to 2, lambda1 to 4), which together move the
    # conductor by a few thousandths of a kelvin. That is what this check is
    # for: the block as actually written must still reproduce the FEM to far
    # better than any thermometer.
    out['ok'] = (err_relation < 1e-3 and err_case < 0.05 and out['monotonic']
                 and out.get('responds_to_soil_step', True))
    return out


# =============================================================================
# 5. THE SUITE
# =============================================================================

_RULE = "=" * 78
_THIN = "-" * 78


def truncation_cost(fem: ThermalFEM, reference: LadderModel,
                    orders: Sequence[int] = (2, 3, 4, 5, 6),
                    tau_range: tuple[float, float] = DEFAULT_TAU_RANGE,
                    t_grid: np.ndarray | None = None) -> dict:
    """
    How much accuracy each ladder order costs, measured against *reference*.

    The two-node error is **structural**, not a fitting-window artefact:
    the FEM step response has a distinct insulation stage and a distinct soil
    stage, and two poles cannot trace both. It therefore vanishes at both ends
    (below ~10 min and above ~4 months the two-node model is right for the
    trivial reasons — nothing has happened yet, or everything has) and peaks in
    between, over roughly 20 min to two weeks. That is precisely the band
    dynamic rating cares about, which is why stage 3 exists. The steady state
    is exact at every order — only the transient suffers.

    :returns: ``{order: {'ladder': LadderModel, 'err': float}}``.
    """
    t = np.logspace(0.0, 8.0, 160) if t_grid is None else np.asarray(t_grid, float)
    z_ref = reference.foster_step(t)
    out = {}
    for n in orders:
        lad = identify_ladder(fem, n_stages=n, tau_range=tau_range)
        err = float(np.max(np.abs(lad.foster_step(t) - z_ref)))
        out[n] = {'ladder': lad, 'err': err}
    return out


def run_suite(case: CableCase | None = None,
              n_stages: int = 5,
              in_cable_passes: int = 1,
              steady_passes: int | None = 2,
              refine_factors: Sequence[float] | None = None,
              tau_range: tuple[float, float] = DEFAULT_TAU_RANGE,
              orders: Sequence[int] = (2, 4, 5, 6),
              node_e_radius: float | None = None,
              pin_T1: bool = True,
              validate: bool = True,
              verify: bool = False,
              plot: str | None = None,
              json_path: str | None = None,
              verbose: bool = True) -> dict:
    """
    Run the whole pipeline — geometry → FEM → ladder → pydae-bps parameters —
    and report it.

    :param case: the case to characterise; defaults to
        :func:`cigre_mv_eu_case`.
    :param n_stages: order of the ladder reported in stage 3.
    :param in_cable_passes: in-cable refinement passes for the **transient**
        mesh. The reduction factorises the stiffness matrix once per
        interpolation point, so this mesh is deliberately one pass coarser than
        the steady one.
    :param steady_passes: in-cable refinement passes for the **steady** mesh,
        which sets ``T1``/``T4``. ``None`` reuses the transient mesh for
        everything (faster, ~0.5 K coarser on the conductor).
    :param refine_factors: adaptive-refinement radii as multiples of the cable
        radius; see :class:`ThermalFEM`. Dropping the outermost passes is the
        cheapest way to trade accuracy for speed in a test run.
    :param orders: ladder orders to price in the truncation-cost table.
    :param node_e_radius: where θ_e sits [m]; see
        :meth:`ThermalFEM.analytic_T1`.
    :param pin_T1: keep ``T1`` physical in the exported block; see
        :func:`fit_two_node`. Both variants are reported either way.
    :param validate: cross-check the ladder against a direct backward-Euler
        transient FEM solve (a few extra factorisations).
    :param verify: build the real pydae-bps model with the exported block and
        check it (needs ``pydae.bps`` and CasADi).
    :param plot: path for a validation figure, or ``None``.
    :param json_path: path to write the ``dtr`` block and the n-stage ladder to.
    :param verbose: print the report.
    :returns: a dict with ``case``, ``fem``, ``fem_steady``, ``steady``,
        ``ladder``, ``reference``, ``fit``, ``orders``, ``validation`` and
        ``verification``.
    """
    case = cigre_mv_eu_case() if case is None else case
    say = print if verbose else (lambda *a, **k: None)

    # ---- stage 1: geometry and materials ----------------------------------
    say(_RULE)
    say(" 1. GEOMETRY AND MATERIALS")
    say(_RULE)
    say(case.report())

    # ---- stage 2: FEM ------------------------------------------------------
    fem = ThermalFEM(case, in_cable_passes=in_cable_passes,
                     refine_factors=refine_factors)
    fem_steady = (fem if steady_passes is None
                  else ThermalFEM(case, in_cable_passes=steady_passes,
                                  refine_factors=refine_factors))
    say("\n" + _RULE)
    say(" 2. FINITE-ELEMENT THERMAL SIMULATION")
    say(_RULE)
    say(f" steady mesh    : {fem_steady.n_elements} elements, "
        f"{fem_steady.n_dofs} dofs ({fem_steady.in_cable_passes} in-cable "
        "refinement passes)")
    if fem_steady is not fem:
        say(f" transient mesh : {fem.n_elements} elements, {fem.n_dofs} dofs "
            f"({fem.in_cable_passes} passes — the reduction factorises K once "
            "per interpolation point)")
    steady = fem_steady.steady_metrics(node_e_radius=node_e_radius)
    say(str(steady))
    say("\n" + _THIN)
    say(" INDEPENDENT ANALYTIC CROSS-CHECKS  (no mesh, no assembly, no Krylov)")
    say(_THIN)
    t1_a = fem_steady.analytic_T1(node_e_radius)
    t1_s = ((steady.t_conductor[steady.hot_index] - steady.t_node_e)
            / steady.losses['W_c'])
    say(f"   T1  layer-stack log formula {t1_a:.5f} vs FEM θ_e sample "
        f"{t1_s:.5f} K·m/W  ({1e2 * (t1_s - t1_a) / t1_a:+.2f} %)")
    a4 = fem_steady.analytic_T4(node_e_radius)
    t4_fem = steady.z_hot - t1_a / (1.0 + steady.losses['lambda1'])
    say(f"   T4  Kennelly images + outer layers {a4['total']:.5f} vs FEM "
        f"{t4_fem:.5f} K·m/W  "
        f"({1e2 * (t4_fem - a4['total']) / a4['total']:+.2f} %)")
    say(f"       = outer layers {a4['outer_layers']:.5f} + soil self "
        f"{a4['soil_self']:.5f} + soil mutual {a4['soil_mutual']:.5f}")
    say(f"   the soil-self term {a4['soil_self']:.5f} is EXACTLY what "
        "line_dtr_cable's")
    say("   geometry route computes on its own — the mutual term is what it "
        "misses.")

    # ---- stage 3: ladder ---------------------------------------------------
    say("\n" + _RULE)
    say(" 3. LADDER (CAUER) MODEL")
    say(_RULE)
    ladder = identify_ladder(fem, n_stages=n_stages, tau_range=tau_range)
    reference = identify_ladder(fem, n_stages=REFERENCE_ORDER,
                                tau_range=(10 ** 0.5, 10 ** 7.5))
    say(str(ladder))
    say("\n Eigenvalue time constants [s]: "
        + ", ".join(f"{t:.4g}" for t in ladder.time_constants))
    say("\n" + _THIN)
    say(" DIFFERENTIAL EQUATIONS OF THE IDENTIFIED LADDER")
    say(_THIN)
    say(ladder.equations())

    validation = None
    if validate:
        t_chk = np.array([1e3, 1e5, 1e7])
        z_fem = transient_fem_step(fem, t_chk)
        z_lad = ladder.step(t_chk)
        z_fos = ladder.foster_step(t_chk)
        validation = {'t': t_chk, 'fem': z_fem, 'ladder': z_lad,
                      'foster': z_fos,
                      'err_ladder_vs_fem': np.max(np.abs(z_lad - z_fem)),
                      'err_ladder_vs_foster': np.max(np.abs(z_lad - z_fos))}
        say("\n" + _THIN)
        say(" VALIDATION — the ladder is re-evaluated from its own state space,")
        say(" and compared with a direct backward-Euler transient FEM solve that")
        say(" shares no machinery with the Krylov/Cauer path.")
        say(_THIN)
        say("        t [s]    FEM [K·m/W]  ladder [K·m/W]   Foster        diff")
        for tt, a_, b_, c_ in zip(t_chk, z_fem, z_lad, z_fos):
            say(f"    {tt:9.1e}   {a_:11.5f}    {b_:11.5f}  {c_:11.5f}  "
                f"{b_ - a_:+10.5f}")
        A_l, b_l, c_l = ladder.matrices()
        dc = float(-c_l @ np.linalg.solve(A_l, b_l))
        validation['dc_gain'] = dc
        say(f"    state-space DC gain {dc:.6f} vs sum R "
            f"{ladder.R.sum():.6f} (diff {abs(dc - ladder.R.sum()):.2e})")

    order_costs = truncation_cost(fem, reference, orders=orders,
                                  tau_range=tau_range)
    p_cable = steady.losses['W_total']
    say("\n" + _THIN)
    say(" COST OF TRUNCATING THE LADDER  (max transient error vs the FEM)")
    say(_THIN)
    for n, d in sorted(order_costs.items()):
        tag = "  <- the order line_dtr_cable implements" if n == 2 else ""
        say(f"   n = {n}  : {d['err']:.4f} K·m/W = {d['err'] * p_cable:5.2f} K"
            + tag)
    say("   These are MOMENT-MATCHED at the interpolation points. Stage 4 does")
    say("   better at n = 2 by least-squares fitting the capacitances over the")
    say("   whole window instead — at the cost of interpolating nothing exactly.")
    say("   The n = 2 error is not spread evenly: it is near zero below ~10 min")
    say("   and above ~4 months, and peaks in between — roughly 20 min to two")
    say("   weeks, which is exactly the range dynamic rating operates in. That")
    say("   is STRUCTURAL, not a fitting-window choice: the FEM response has a")
    say("   distinct insulation stage and a distinct soil stage, and two poles")
    say("   cannot trace both. Steady state is exact at every order — only the")
    say("   transient suffers.")

    # ---- stage 4: pydae-bps parameters ------------------------------------
    say("\n" + _RULE)
    say(" 4. PARAMETERS FOR pydae.bps.lines.line_dtr_cable")
    say(_RULE)
    fit = fit_two_node(fem, steady=steady, reference=reference,
                       node_e_radius=node_e_radius, pin_T1=pin_T1)
    fit_alt = fit_two_node(fem, steady=steady, reference=reference,
                           node_e_radius=node_e_radius, pin_T1=not pin_T1)
    say(str(fit))
    if abs(fit.dc_rescale - 1.0) > 1e-9:
        say(f"   (transient reference rescaled by {fit.dc_rescale:.5f} onto the "
            "finer steady-mesh DC value)")
    say(f"   self-consistent conductor temperature at I = {case.current:.0f} A, "
        f"with R_ac re-evaluated at that temperature: "
        f"{fit.t_conductor_self_consistent:.2f} °C")
    rise = steady.t_conductor[steady.hot_index] - steady.t_soil
    say(f"   the {fit.err_kelvin:.2f} K transient error is "
        f"{1e2 * fit.err_kelvin / rise:.0f} % of the {rise:.1f} K steady rise, "
        "and it is the price of TWO nodes, not a bad fit:")
    say(f"     T1 {'pinned to its analytic value' if pin_T1 else 'fitted'} "
        f"(as exported)     : {fit.err_kelvin:5.2f} K")
    say(f"     T1 {'fitted' if pin_T1 else 'pinned to its analytic value'} "
        f"(pin_T1={not pin_T1})       : {fit_alt.err_kelvin:5.2f} K  "
        f"(T1 {fit_alt.T1:.4f}, T4 {fit_alt.T4:.4f})")
    say("   Both keep the steady state EXACT — only the internal split moves.")
    say("   Stage 3's ladder is the way out if the transient matters more than")
    say("   the IEC parameter names do.")

    say("\n" + _THIN)
    say(" WHAT THE LIBRARY'S GEOMETRY ROUTE WOULD GIVE, AND WHY IT IS WRONG HERE")
    say(_THIN)
    say(compare_with_library(case, fit))

    say("\n" + _THIN)
    say(" IF THE PHASES ARE EVER UNBALANCED")
    say(_THIN)
    say(f"   self          Z_self = {ladder.z_self:.5f} K·m/W")
    for i, zm in enumerate(ladder.z_mutual, 1):
        say(f"   mutual {i}             = {zm:.5f} K·m/W")
    say(f"   balanced total        = {ladder.z_balanced:.5f} K·m/W  -> the "
        "single T4 above")
    say("   Under imbalance the collapse to one number is invalid: each cable")
    say("   needs its own W and the full matrix  dT_i = SUM_j Z_ij W_j  comes")
    say("   back. Use the entries above, not T4.")

    dtr = fit.as_dtr()
    say("\n" + _RULE)
    say(" READY TO PASTE INTO A pydae-bps lines[] ENTRY")
    say(_RULE)
    say('   "dtr": ' + json.dumps(dtr, indent=8).replace("\n", "\n   "))
    say("\n   T1/T4/Q1/Q2 are given EXPLICITLY, which overrides the library's")
    say("   geometry-derived values — deliberately: the geometry route cannot")
    say("   represent three thermally coupled cables.")

    verification = None
    if verify:
        say("\n" + _THIN)
        say(" VERIFICATION — building the real pydae-bps model with this block")
        say(_THIN)
        try:
            verification = verify_with_pydae(case, fit, run_seconds=3600.0)
        except ImportError as exc:
            say(f"   skipped: {exc}")
        else:
            v = verification
            say(f"   model ini: I_cond {v['I_cond']:.1f} A -> W_c "
                f"{v['W_c']:.3f} W/m")
            say(f"   t_conductor {v['t_conductor']:.2f} °C, t_sheath "
                f"{v['t_sheath']:.2f} °C, t_soil {v['t_soil']:.2f} °C")
            say(f"   two-node steady relation residual "
                f"{v['err_steady_relation']:.2e} K")
            say(f"   at the case current the model gives "
                f"{v['t_conductor_at_case_current']:.2f} °C vs the FEM-derived "
                f"{fit.t_conductor_self_consistent:.2f} °C "
                f"(diff {v['err_vs_fem_self_consistent']:.2e} K)")
            if 'responds_to_soil_step' in v:
                say(f"   +10 K soil step for 1 h -> "
                    f"{v['t_conductor_after_soil_step']:.2f} °C "
                    f"({'follows' if v['responds_to_soil_step'] else 'NO RESPONSE'})")
            say(f"   => {'PASS' if v['ok'] else 'FAIL'}")

    result = {'case': case, 'fem': fem, 'fem_steady': fem_steady,
              'steady': steady, 'ladder': ladder, 'reference': reference,
              'fit': fit, 'fit_alternative': fit_alt, 'orders': order_costs,
              'validation': validation, 'verification': verification}

    if json_path:
        payload = {
            'case': case.label,
            'two_node': dtr,
            'two_node_alternative_split': fit_alt.as_dtr(),
            'two_node_geometry_only': fit.as_dtr(mode='geometry'),
            'ladder': {'n_stages': ladder.n_stages,
                       'R': [float(f"{v:.6f}") for v in ladder.R],
                       'C': [float(f"{v:.4e}") for v in ladder.C],
                       'note': ("Cauer ladder, cable outwards; needs "
                                "line_dtr_cable extended beyond two nodes")},
            'steady': {'t_conductor': float(steady.t_conductor[steady.hot_index]),
                       't_soil': float(steady.t_soil),
                       'z_hot': float(steady.z_hot),
                       'z_self': float(ladder.z_self),
                       'z_mutual': [float(v) for v in ladder.z_mutual]},
            'errors': {'two_node_K': float(fit.err_kelvin),
                       f'ladder_n{ladder.n_stages}_Km_per_W':
                           float(order_costs.get(ladder.n_stages, {})
                                 .get('err', float('nan')))},
        }
        with open(json_path, 'w') as fh:
            json.dump(payload, fh, indent=2)
        say(f"\n Written: {json_path}")

    if plot:
        _plot_validation(result, plot)
        say(f" Figure : {plot}")

    return result


def _plot_validation(result: dict, path: str) -> None:
    """Two-panel figure: model vs FEM, and the error that remains."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ladder, fit, ref = result['ladder'], result['fit'], result['reference']
    p = result['steady'].losses['W_total']
    t = np.logspace(0.0, 8.0, 200)
    z_ref = ref.foster_step(t) * fit.dc_rescale
    z_2n = fit.step(t)
    z_n = ladder.foster_step(t) * fit.dc_rescale

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.semilogx(t, z_ref * p, lw=3, color='0.75',
                label=f"FEM ({result['case'].n_cables} coupled cables)")
    a1.semilogx(t, z_2n * p, lw=1.8, ls='--', color='crimson',
                label=f"two-node (IEC 60853), err {fit.err_kelvin:.1f} K")
    a1.semilogx(t, z_n * p, lw=1.8, ls='-.', color='seagreen',
                label=f"{ladder.n_stages}-stage ladder, err "
                      f"{result['orders'].get(ladder.n_stages, {}).get('err', 0.0) * p:.2f} K")
    for d, lab in ((3600, '1 h'), (86400, '1 d'), (30 * 86400, '1 mo')):
        a1.axvline(d, color='0.9', lw=0.8, zorder=0)
        a1.text(d, 0.5, f" {lab}", fontsize=7, color='0.5', rotation=90)
    a1.set_xlabel('time [s]')
    a1.set_ylabel('conductor rise [K]')
    a1.set_title('Single-phase-equivalent thermal model vs FEM', fontsize=11)
    a1.legend(fontsize=9, loc='upper left')
    a1.grid(alpha=0.3)

    a2.semilogx(t, (z_2n - z_ref) * p, color='crimson', lw=1.8,
                label='two-node error')
    a2.semilogx(t, (z_n - z_ref) * p, color='seagreen', lw=1.8,
                label=f'{ladder.n_stages}-stage error')
    a2.axhline(0, color='k', lw=0.8)
    a2.set_xlabel('time [s]')
    a2.set_ylabel('error [K]')
    a2.set_title('Two-node error peaks over 20 min … 2 weeks — the DTR range',
                 fontsize=11)
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_field(result_or_fem, path: str, zoom: float | None = None) -> None:
    """
    Filled-contour plot of the steady temperature field with the cable outlines
    overlaid — the picture that makes a wrong geometry obvious at a glance.

    :param result_or_fem: a :func:`run_suite` result dict or a
        :class:`ThermalFEM`.
    :param zoom: half-width of the plotted window [m]; auto-sized from the
        layout when omitted.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.patches import Circle

    fem = (result_or_fem['fem_steady'] if isinstance(result_or_fem, dict)
           else result_or_fem)
    case = fem.case
    T = fem.steady(True)
    mesh = fem.mesh
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)

    c = case.centers
    half = np.abs(c[:, 0]).max()
    depth = abs(c[:, 1].min())
    pad = zoom if zoom else max(0.45, 2.5 * half, 0.60 * depth)

    fig, ax = plt.subplots(figsize=(9, 6))
    cs = ax.tricontourf(tri, T, levels=40, cmap='inferno')
    ax.tricontour(tri, T, levels=14, colors='k', linewidths=0.35, alpha=0.5)
    for cx, cy in c:
        ax.add_patch(Circle((cx, cy), case.r_cable, fill=False, ec='w', lw=1.2))
        ax.add_patch(Circle((cx, cy), case.design.r_conductor, fill=False,
                            ec='w', lw=0.8, ls=':'))
    ax.axhline(0.0, color='k', lw=1.5)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-(depth + 0.8 * pad), 0.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]  (0 = ground surface)')
    ax.set_title(case.label, fontsize=10)
    fig.colorbar(cs, ax=ax, label='temperature [°C]')
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# =============================================================================
# 6. SELF-TESTS
# =============================================================================
# Ordered cheapest first. test_geometry needs nothing beyond numpy; the FEM
# tests are marked slow in spirit -- each builds a mesh -- so they share one
# module-level fixture rather than re-meshing per assertion.

_FIXTURE: dict = {}


def _fixture(in_cable_passes: int = 1) -> dict:
    """Build (once) a small FEM + ladder + fit for the test functions."""
    key = f'f{in_cable_passes}'
    if key not in _FIXTURE:
        case = cigre_mv_eu_case(200.0)
        fem = ThermalFEM(case, in_cable_passes=in_cable_passes)
        steady = fem.steady_metrics()
        ladder = identify_ladder(fem, n_stages=5)
        reference = identify_ladder(fem, n_stages=REFERENCE_ORDER,
                                    tau_range=(10 ** 0.5, 10 ** 7.5))
        fit = fit_two_node(fem, steady=steady, reference=reference)
        _FIXTURE[key] = {'case': case, 'fem': fem, 'steady': steady,
                         'ladder': ladder, 'reference': reference, 'fit': fit}
    return _FIXTURE[key]


def test_geometry():
    """Stage 1: the layer stack, the layout and the loss split."""
    d = NA2XS2Y_120_AL
    # Table 6.11 dimensions must be recoverable from the stack.
    assert abs(2.0 * d.r_outer - 34.2e-3) < 5e-5, "stack does not close on d_ov"
    assert abs(2.0 * d.r_conductor - 12.4e-3) < 1e-6
    assert abs(d.r_insulation - (6.2e-3 + 0.8e-3 + 5.5e-3)) < 1e-9
    # pi*(d_c/2)^2 must give back the 120 mm2 nameplate area.
    area = math.pi * d.r_conductor ** 2
    assert abs(area * 1e6 - 120.0) < 1.5, f"conductor area {area*1e6:.1f} mm2"
    # lambda1 from the tabulated resistance ratio.
    assert abs(d.lambda1 - (0.501 / 0.338 - 1.0)) < 1e-9
    assert abs(d.R_ac(90.0) - d.R_ac_90) < 1e-15
    # The screen loss band must be the insulation-screen -> bedding composite.
    band = d.heat_band('screen')
    assert abs(band[0] - 12.5e-3) < 1e-9 and abs(band[1] - 14.6e-3) < 1e-9

    case = cigre_mv_eu_case(200.0)
    c = case.centers
    assert c.shape == (3, 2)
    assert np.allclose(c[:, 1], -0.7), "all three cables sit at the axis depth"
    assert abs(np.linalg.norm(c[1] - c[0]) - 0.3) < 1e-12
    assert case.hot_index == 1, "the centre cable is the hot spot"

    w = case.losses()
    assert abs(w['W_c'] - 200.0 ** 2 * 0.338e-3) < 1e-9
    assert abs(w['W_total'] - 200.0 ** 2 * 0.501e-3) < 1e-9

    # A stack that does not close on the nameplate diameter must be rejected.
    try:
        CableDesign(name='bad', layers=(Layer('c', 6e-3, 237.0, 2.4e6,
                                              'conductor', 'conductor'),),
                    R_ac_90=1e-4, R_ph=1e-4, d_over=34.2e-3)
    except ValueError:
        pass
    else:                                                # pragma: no cover
        raise AssertionError("a non-closing stack was accepted")

    # The Kennelly diagonal must be the textbook isolated-cable formula.
    Z = case.kennelly_matrix()
    assert np.allclose(Z, Z.T), "the mutual resistance matrix is not symmetric"
    expect = math.log(4.0 * 0.7 / (2.0 * case.r_cable)) / (2.0 * math.pi)
    assert abs(Z[0, 0] - expect) < 1e-12
    assert Z.sum(1)[1] > Z.sum(1)[0], "the centre cable must see more mutual"

    # Trefoil: three touching cables, and the hot spot is one of the DEEPER
    # two -- not the centroid-nearest cable, since all three tie on that.
    tre = CableCase(NA2XS2Y_120_AL,
                    Installation(formation='trefoil', depth=1.2), 200.0)
    ct = tre.centers
    for a, b in ((0, 1), (1, 2), (2, 0)):
        assert abs(np.linalg.norm(ct[a] - ct[b]) - 2 * tre.r_cable) < 1e-12, \
            "trefoil cables are not tangent"
    assert np.allclose(np.linalg.norm(ct - ct.mean(0), axis=1),
                       2 * tre.r_cable / math.sqrt(3.0)), \
        "trefoil cables should be equidistant from the centroid"
    assert tre.hot_index in (1, 2), \
        "the hot spot in a trefoil is one of the two lower cables"
    print("test_geometry passed")


def test_fem_steady():
    """Stage 2: the steady field, its energy balance and its ordering."""
    f = _fixture()
    st, case = f['steady'], f['case']
    hot = st.hot_index

    assert abs(st.energy_balance) < 1e-8, \
        f"global energy balance off by {st.energy_balance:.2e}"
    assert abs(st.source_error) < 2e-3, \
        f"source quadrature off by {1e2*st.source_error:.3f} %"

    # Physical ordering: conductor hotter than the split node, hotter than the
    # no-load soil, and the centre cable hotter than its neighbours.
    assert st.t_conductor[hot] > st.t_node_e > st.t_soil
    others = [i for i in range(case.n_cables) if i != hot]
    for i in others:
        assert st.t_conductor[hot] > st.t_conductor[i], "centre is not hottest"
    assert abs(st.t_conductor[others[0]] - st.t_conductor[others[1]]) < 0.05, \
        "the symmetric outer cables disagree — the mesh is not symmetric"

    # The no-load soil at burial depth must sit strictly between the deep-earth
    # boundary and the air temperature: that gradient is the whole reason
    # t_soil is not simply t_deep.
    assert case.installation.t_deep < st.t_soil < case.installation.t_air

    # The two ways of getting T1 must agree to a few percent.
    t1_a = f['fem'].analytic_T1()
    t1_f = (st.t_conductor[hot] - st.t_node_e) / st.losses['W_c']
    assert abs(t1_a - t1_f) / t1_a < 0.05, \
        f"analytic T1 {t1_a:.5f} vs FEM sample {t1_f:.5f}"

    # T4 against the Kennelly image method plus the outer layer stack. This
    # shares no code at all with the FEM, so agreement validates the mesh, the
    # material mapping, the boundary conditions and the loss split at once.
    a4 = f['fem'].analytic_T4()
    t4_fem = st.z_hot - t1_a / (1.0 + st.losses['lambda1'])
    assert abs(t4_fem - a4['total']) / a4['total'] < 0.08, \
        f"FEM T4 {t4_fem:.5f} vs analytic {a4['total']:.5f}"
    # The analytic diagonal must reproduce the library's isolated-cable T4,
    # which is the whole basis of the "T4 is too small" claim.
    lib = library_defaults(case)
    assert abs(a4['soil_self'] - lib['T4']) / lib['T4'] < 1e-3, \
        "the Kennelly diagonal and line_dtr_cable's T4 formula disagree"
    assert a4['soil_mutual'] > 0.0

    # A de-energised circuit must be strictly cooler everywhere.
    assert np.all(f['fem'].steady(False) <= f['fem'].steady(True) + 1e-9)
    print("test_fem_steady passed")


def test_ladder():
    """Stage 3: realisability, DC exactness and agreement with a direct solve."""
    f = _fixture()
    lad, st = f['ladder'], f['steady']

    assert lad.passive, "no passive ladder was realised for the hot spot"
    assert np.all(lad.R > 0.0) and np.all(lad.C > 0.0)
    assert np.all(lad.R_foster > 0.0)

    # sum(R) is the DC thermal resistance, and it must reproduce the steady
    # solve -- the s = 0 vector is in the Krylov basis, so this is exact up to
    # the mesh difference between the two solves.
    assert abs(lad.dc_resistance - lad.R.sum()) < 1e-9 * lad.R.sum()

    # The state-space form and the Foster form must be the same system.
    t = np.logspace(0.0, 8.0, 60)
    assert np.max(np.abs(lad.step(t) - lad.foster_step(t))) < 1e-6, \
        "the extracted ladder is not the system it was extracted from"

    # DC gain straight out of (A, b, c).
    A, b, c = lad.matrices()
    assert abs(float(-c @ np.linalg.solve(A, b)) - lad.R.sum()) < 1e-8

    # Independent ground truth: backward Euler on the full FEM system.
    t_chk = np.array([1e3, 1e5, 1e7])
    z_fem = transient_fem_step(f['fem'], t_chk)
    z_lad = lad.step(t_chk)
    assert np.max(np.abs(z_lad - z_fem)) < 0.02, \
        f"ladder vs direct transient FEM: {np.max(np.abs(z_lad - z_fem)):.4f}"

    # Mutual heating is a large, positive share of the centre cable's rise.
    assert np.all(lad.z_mutual > 0.0)
    assert 0.1 < lad.mutual_share < 0.6, \
        f"implausible mutual share {lad.mutual_share:.3f}"
    assert lad.z_balanced > lad.z_outer > 0.0, \
        "the centre cable must run hotter than the outer ones"

    # Time constants must span the insulation-to-soil range.
    tau = lad.time_constants
    assert tau[0] > 1e5 and tau[-1] < 1e4, f"time constants {tau}"

    # Higher order must not be worse than lower order.
    costs = truncation_cost(f['fem'], f['reference'], orders=(2, 5))
    assert costs[5]['err'] < costs[2]['err'], \
        "the five-stage ladder is not better than the two-node one"

    # The collocated variant is guaranteed realisable -- if it is not, the
    # method itself is broken rather than the hot-spot transfer being awkward.
    col = identify_ladder(f['fem'], n_stages=4, collocated=True)
    assert col.passive, "the collocated (Stieltjes) case failed to realise"
    print("test_ladder passed")


def test_two_node_fit():
    """Stage 4: the exported block reproduces the FEM."""
    f = _fixture()
    fit, st, case = f['fit'], f['steady'], f['case']
    hot = st.hot_index

    assert fit.T1 > 0.0 and fit.T4 > 0.0
    assert fit.Q1 > 0.0 and fit.Q2 > fit.Q1, \
        "the soil node must hold far more heat than the conductor node"
    assert 0.0 < fit.p <= 0.5, f"Van Wormer factor out of range: {fit.p}"

    # Steady state must be reproduced EXACTLY -- T1 and T4 are pinned by it,
    # never fitted, so anything above round-off is a bookkeeping error.
    t_c = fit.steady_conductor(st.losses['W_c'])
    assert abs(t_c - st.t_conductor[hot]) < 1e-6, \
        f"two-node steady {t_c:.4f} vs FEM {st.t_conductor[hot]:.4f}"
    # ... and it must equal the FEM Z_hot decomposition.
    assert abs(fit.T1 / (1.0 + fit.lambda1) + fit.T4 - st.z_hot) < 1e-9

    # The transient truncation error is real and large -- roughly 20 % of the
    # steady rise -- because two nodes cannot span the insulation and soil time
    # scales at once. The bound below only guards against a *broken* fit; the
    # honest number is reported, not asserted away.
    rise = st.t_conductor[hot] - st.t_soil
    assert 0.0 < fit.err_kelvin < 0.45 * rise, \
        f"two-node transient error {fit.err_kelvin:.2f} K against a " \
        f"{rise:.1f} K rise is beyond the known truncation cost"

    # Letting T1 float (DC still pinned) must do better on the transient and
    # must not disturb the steady state.
    free = fit_two_node(f['fem'], steady=st, reference=f['reference'],
                        pin_T1=False)
    assert free.err_kelvin < fit.err_kelvin, \
        "freeing T1 did not improve the transient — the search is stuck"
    assert abs(free.T1 / (1.0 + free.lambda1) + free.T4 - st.z_hot) < 1e-9, \
        "the free-T1 fit broke the exact DC constraint"
    assert abs(free.steady_conductor(st.losses['W_c'])
               - st.t_conductor[hot]) < 1e-6

    # Two well-separated time constants: insulation-ish and soil-ish.
    tau = fit.time_constants
    assert tau[0] > 1e4 > tau[1] > 0.0, f"two-node time constants {tau}"

    # The exported block must carry every key line_dtr_cable reads.
    dtr = fit.as_dtr()
    for key in ('model', 'T1', 'T4', 'Q1', 'Q2', 'p', 'lambda1', 'Wd',
                'R_ac_20', 'alpha_R', 'K_dtr', 't_soil'):
        assert key in dtr, f"missing dtr key {key!r}"
    assert dtr['model'] == 'cable'
    assert json.loads(json.dumps(dtr)) == dtr, "the block is not JSON-clean"
    # 'geometry' mode must NOT pin the network, so the library derives it.
    geo = fit.as_dtr(mode='geometry')
    assert not {'T1', 'T4', 'Q1', 'Q2'} & set(geo)

    # The whole point: the library's isolated-cable T4 under-predicts.
    lib = library_defaults(case)
    assert fit.T4 > lib['T4'], \
        "the fitted T4 should exceed the isolated-cable value — it carries " \
        "the mutual heating of the other two phases"

    # Self-consistent temperature must exceed the fixed-loss one, because
    # R_ac rises with temperature.
    assert fit.t_conductor_self_consistent > fit.t_soil
    print("test_two_node_fit passed")


def test_suite():
    """End-to-end: the driver runs and returns a coherent result."""
    case = cigre_mv_eu_case(200.0)
    res = run_suite(case, n_stages=4, in_cable_passes=1, steady_passes=None,
                    orders=(2, 4), validate=False, verbose=False)
    assert set(res) >= {'case', 'fem', 'steady', 'ladder', 'fit', 'orders'}
    assert res['fit'].as_dtr()['model'] == 'cable'
    assert res['orders'][4]['err'] < res['orders'][2]['err']
    print("test_suite passed")


def test():
    """Run every self-test in order."""
    test_geometry()
    test_fem_steady()
    test_ladder()
    test_two_node_fit()
    test_suite()
    print("all cable_fem tests passed")


def main(argv: Sequence[str] | None = None) -> dict:
    """
    Command-line entry point::

        python -m pydae.bps.lines.cable_fem --current 200 --plot field.png
        python -m pydae.bps.lines.cable_fem --test

    :returns: the :func:`run_suite` result dict (empty when ``--test`` is given).
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="FEM thermal characterisation of an underground cable "
                    "circuit, exported as line_dtr_cable parameters.")
    ap.add_argument('--current', type=float, default=200.0,
                    help="phase current [A] (default 200)")
    ap.add_argument('--stages', type=int, default=5,
                    help="ladder order to report (default 5)")
    ap.add_argument('--passes', type=int, default=1,
                    help="in-cable refinement passes for the transient mesh")
    ap.add_argument('--steady-passes', type=int, default=2,
                    help="in-cable refinement passes for the steady mesh; "
                         "-1 reuses the transient mesh")
    ap.add_argument('--backfill', default=None,
                    choices=sorted(BACKFILL_CASES),
                    help="use a backfilled trench instead of native soil")
    ap.add_argument('--k-soil', type=float, default=1.0,
                    help="soil thermal conductivity [W/(m·K)] (default 1.0)")
    ap.add_argument('--depth', type=float, default=0.7,
                    help="axis burial depth [m] (default 0.7)")
    ap.add_argument('--spacing', type=float, default=0.3,
                    help="centre-to-centre spacing [m] (default 0.3)")
    ap.add_argument('--json', default=None, help="write the dtr block here")
    ap.add_argument('--plot', default=None, help="write the validation figure here")
    ap.add_argument('--field', default=None, help="write the field plot here")
    ap.add_argument('--verify', action='store_true',
                    help="build the real pydae-bps model and check it")
    ap.add_argument('--test', action='store_true', help="run the self-tests")
    args = ap.parse_args(argv)

    if args.test:
        test()
        return {}

    case = cigre_mv_eu_case(args.current, backfill=args.backfill,
                            k_soil=args.k_soil, depth=args.depth,
                            spacing=args.spacing)
    res = run_suite(case, n_stages=args.stages, in_cable_passes=args.passes,
                    steady_passes=(None if args.steady_passes < 0
                                   else args.steady_passes),
                    verify=args.verify, plot=args.plot, json_path=args.json)
    if args.field:
        plot_field(res, args.field)
        print(f" Field  : {args.field}")
    return res


if __name__ == "__main__":
    main()
