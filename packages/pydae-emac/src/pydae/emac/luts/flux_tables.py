# -*- coding: utf-8 -*-
r"""FEA look-up tables -> the tables the dynamic models use.

The FEA gives, at every rotor angle theta, the fluxes and torque as functions
of the CURRENTS:      lam_d, lam_q, T  =  f(i_d, i_q, theta)

The dynamic model has the FLUXES as states (a voltage step then changes only
d(lambda)/dt, never jumps a state), so it needs the opposite direction:
                      i_d, i_q, T  =  g(lam_d, lam_q, theta)

This file builds g on a regular (lam_d, lam_q) grid, in three steps:
  1. invert f at every FEA angle (Newton's method on a smooth version of f);
  2. write the theta-dependence as a Fourier series, so the model needs only
     2-D tables of Fourier coefficients instead of a 3-D table;
  3. pack all coefficient tables into one flat array, the layout CasADi's
     B-spline interpolant wants.

Both solvers read the result: the switched drive (``pydae.emac.drive``) uses
every Fourier term, the mean-value DAE machine (``machines.pmsm_lut``) only
the mean, through :func:`mean_tables`.

The FEA table format (``FluxMap_dqt.npz``) holds, indexed [i_d, i_q, theta]:
``i_d``, ``i_q`` (A peak), ``th_elec`` (deg electrical, uniform over one
period of the dq quantities: 60 deg for a 3-phase machine), ``Fd``, ``Fq``
(Wb peak), ``T`` (N.m) and ``p`` (pole pairs).  :func:`save_fea_lut` writes
it, :func:`load_fea_lut` reads it.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata


def load_fea_lut(path):
    """Read a position-resolved FEA table (``FluxMap_dqt.npz`` layout).

    Arrays are indexed [i_d, i_q, theta].  Currents are peak amperes in the
    machine's dq frame (amplitude-invariant), fluxes in Wb, torque in N.m.
    The table's angle ``th_elec`` is in electrical degrees; the models work
    in electrical radians, so it is converted here.
    """
    z = np.load(path)
    return dict(i_d=z["i_d"],                       # (n_id,)  A peak
                i_q=z["i_q"],                       # (n_iq,)  A peak
                theta_e=np.radians(z["th_elec"]),   # (n_th,)  rad electrical
                lam_d=z["Fd"],                      # (n_id, n_iq, n_th) Wb
                lam_q=z["Fq"],                      # (n_id, n_iq, n_th) Wb
                T=z["T"],                           # (n_id, n_iq, n_th) N.m
                p=int(z["p"]))                      # pole pairs


class ForwardMap:
    """One FEA table at one angle, f(i_d, i_q), usable OUTSIDE the table too.

    Inside the swept currents: cubic interpolation (smooth, accurate).
    Outside: continued LINEARLY from the edge,
        f(x) = cubic(clip(x)) + [linear(x) - linear(clip(x))]
    Why: the regular flux grid always reaches flux values the FEA never swept
    (its corners, and here all of i_d < 0).  Continuing the edge CUBICS there
    diverges wildly, and one bad value spoils the whole B-spline fitted later.
    A linear continuation simply keeps the edge's inductance.
    """

    def __init__(self, i_d, i_q, F):
        self.lo = np.array([i_d[0], i_q[0]])        # lower corner of the table
        self.hi = np.array([i_d[-1], i_q[-1]])      # upper corner of the table
        self.cubic = RegularGridInterpolator((i_d, i_q), F, method="cubic")
        self.linear = RegularGridInterpolator((i_d, i_q), F, method="linear",
                                              bounds_error=False,
                                              fill_value=None)  # extrapolates

    def __call__(self, x):
        x = np.atleast_2d(x)                        # rows of (i_d, i_q)
        xc = np.clip(x, self.lo, self.hi)           # nearest point inside
        return self.cubic(xc) + (self.linear(x) - self.linear(xc))


def invert_one_angle(lut, k, LD, LQ, n_iter=30, tol=1e-10):
    """Currents and torque on the flux grid (LD, LQ), at FEA angle number k.

    For every grid node we look for the currents (i_d, i_q) whose fluxes are
    the node's fluxes: two equations, two unknowns, solved by Newton's method.
    tol is RELATIVE to the table's flux scale, and the finite-difference step
    to its current scale: the same code serves 1 kW (0.1 Wb, 5 A) and 2 MW
    (20 Wb, 5000 A).
    """
    f_d = ForwardMap(lut["i_d"], lut["i_q"], lut["lam_d"][:, :, k])
    f_q = ForwardMap(lut["i_d"], lut["i_q"], lut["lam_q"][:, :, k])
    f_T = ForwardMap(lut["i_d"], lut["i_q"], lut["T"][:, :, k])
    target = np.c_[LD.ravel(), LQ.ravel()]          # one row per grid node

    # Starting guess: interpolate the FEA points "backwards" (scattered data).
    # Nodes outside the FEA data get the nearest FEA point instead.
    ID, IQ = np.meshgrid(lut["i_d"], lut["i_q"], indexing="ij")
    points = np.c_[lut["lam_d"][:, :, k].ravel(), lut["lam_q"][:, :, k].ravel()]
    x = np.empty_like(target)
    for j, values in enumerate((ID.ravel(), IQ.ravel())):
        guess = griddata(points, values, target, method="linear")
        nearest = griddata(points, values, target, method="nearest")
        x[:, j] = np.where(np.isnan(guess), nearest, guess)

    def residual(x):                                # fluxes(x) - target fluxes
        return np.c_[f_d(x), f_q(x)] - target

    tol = tol * flux_scale(lut)                     # [Wb]
    r = residual(x)
    h = 1e-6 * np.max(np.abs(np.r_[lut["i_d"], lut["i_q"]]))   # step [A]
    for _ in range(n_iter):
        if np.max(np.abs(r)) < tol:
            break
        # Jacobian d(flux)/d(current), 2x2 per node, by central differences
        J = np.empty((len(x), 2, 2))
        for j in range(2):
            dx = np.zeros(2)
            dx[j] = h
            J[:, :, j] = (residual(x + dx) - residual(x - dx)) / (2 * h)
        step = np.linalg.solve(J, r[:, :, None])[:, :, 0]
        # Damped Newton: halve the step, node by node, until the residual falls
        size = np.linalg.norm(r, axis=1)
        alpha = np.ones(len(x))
        for _ in range(12):
            x_new = x - alpha[:, None] * step
            r_new = residual(x_new)
            worse = np.linalg.norm(r_new, axis=1) >= size
            if not worse.any():
                break
            alpha[worse] *= 0.5
        x, r = x_new, r_new

    # Nodes the FEA map never reaches at this angle (corners of the
    # rectangular flux box outside the map's image) have no exact inverse:
    # their least-squares current -- the nearest reachable flux -- is kept,
    # and only REACHABLE nodes count toward the convergence test.  Without
    # that, a strongly saturated map leaves such corners unconverged and
    # stops the build although every node the machine can reach is exact.
    from matplotlib.path import Path
    Ld, Lq = lut["lam_d"][:, :, k], lut["lam_q"][:, :, k]
    ring = np.concatenate([np.c_[Ld[0, :], Lq[0, :]], np.c_[Ld[:, -1], Lq[:, -1]],
                           np.c_[Ld[-1, ::-1], Lq[-1, ::-1]], np.c_[Ld[::-1, 0], Lq[::-1, 0]]])
    reach = Path(ring).contains_points(target)
    err = np.abs(r).max(axis=1)
    shape = LD.shape
    return (x[:, 0].reshape(shape),                 # i_d on the flux grid
            x[:, 1].reshape(shape),                 # i_q on the flux grid
            f_T(x).reshape(shape),                  # torque at those currents
            float(err[reach].max()) if reach.any() else 0.0,   # worst REACHABLE flux error [Wb]
            int((~reach & (err > tol)).sum()))      # unreachable nodes kept at least squares


def flux_scale(lut):
    """The table's flux magnitude [Wb]: what relative tolerances refer to."""
    return float(max(np.abs(lut["lam_d"]).max(), np.abs(lut["lam_q"]).max()))


def build_flux_tables(lut, n_lam=40, theta_offset_e=0.0):
    """All three steps.  Returns the dict the dynamic model is built from.

    theta_offset_e [rad]: where the machine's d axis sits, electrically, at
    FEM angle 0, relative to phase a.  pyleecan aligns ITS d axis with phase
    a at t = 0 (MachineSync.comp_angle_rotor_initial; checked by FEA: phase
    a's no-load flux peaks at t = 0).  So a PM map ("pm", d = pyleecan d)
    has offset 0, and a reluctance map ("syr", d = pyleecan q) has +pi/2.
    The model's Park angle is p theta; the tables are shifted here so that
    the Park angle and the ripple sit where they do in the machine: with the
    wrong offset the 6th, 18th, 30th ... harmonics come out inverted against
    the stator.  Means are unchanged by it.
    """
    th = lut["theta_e"]
    n_th = len(th)
    # The FEA angles cover exactly ONE period of the dq quantities, uniformly.
    # For a 3-phase machine that period is 60 deg electrical, so the Fourier
    # orders are multiples of base = 2 pi / period = 6.
    period = n_th * (th[1] - th[0])
    base = int(round(2 * np.pi / period))
    harmonics = [base * m for m in range(1, n_th // 2)]   # every order resolved

    # The regular flux grid: the box the FEA fluxes span.
    grid_d = np.linspace(lut["lam_d"].min(), lut["lam_d"].max(), n_lam)
    grid_q = np.linspace(lut["lam_q"].min(), lut["lam_q"].max(), n_lam)
    LD, LQ = np.meshgrid(grid_d, grid_q, indexing="ij")

    # ---- step 1: invert at every angle -------------------------------------
    i_d = np.empty((n_lam, n_lam, n_th))
    i_q = np.empty_like(i_d)
    T = np.empty_like(i_d)
    worst, n_out = 0.0, 0
    for k in range(n_th):
        i_d[:, :, k], i_q[:, :, k], T[:, :, k], err, out = invert_one_angle(lut, k, LD, LQ)
        worst, n_out = max(worst, err), n_out + out
    if n_out:
        print("flux tables: %d of %d nodes outside the FEA map's image, kept at least squares"
              % (n_out, LD.size * n_th))
    if worst > 1e-7 * flux_scale(lut):
        # a node that did not converge would ring through the whole B-spline
        raise RuntimeError("flux inversion did not converge (%.2g Wb)" % worst)

    # ---- step 2: Fourier series in theta, at every flux node ----------------
    # f(theta) = mean + sum_h [ A_h cos(h theta) + B_h sin(h theta) ]
    # With uniform samples over one period, order h = m*base is FFT bin m.
    def fourier(F):
        X = np.fft.rfft(F, axis=2) / n_th
        # f(theta - offset): bin m of base-order harmonic rotates by
        # exp(-j m base offset); the samples sit at theta_k = th[0] + k dth
        m_all = np.arange(X.shape[2])
        X = X * np.exp(-1j * m_all * base * theta_offset_e)
        tables = [X[:, :, 0].real]                  # the mean
        for h in harmonics:
            m = h // base
            tables.append(2 * X[:, :, m].real)      # A_h (cosine)
            tables.append(-2 * X[:, :, m].imag)     # B_h (sine)
        return tables                               # list of (n_lam, n_lam)

    coeffs = {"i_d": fourier(i_d), "i_q": fourier(i_q), "T": fourier(T)}

    # ---- step 3: pack everything for ONE vector-valued B-spline ------------
    # Output number  j*3 + q  holds basis function j (0 = mean, 1 = cos h1,
    # 2 = sin h1, 3 = cos h2, ...) of quantity q (0 = i_d, 1 = i_q, 2 = T).
    # CasADi wants the output index running fastest, then lam_d, then lam_q:
    # that is a Fortran-order flattening of an (n_out, n_lam, n_lam) array.
    n_basis = 1 + 2 * len(harmonics)
    stack = np.empty((3 * n_basis, n_lam, n_lam))
    for j in range(n_basis):
        for q, name in enumerate(("i_d", "i_q", "T")):
            stack[j * 3 + q] = coeffs[name][j]

    return dict(grid_d=grid_d, grid_q=grid_q,       # the flux grid [Wb]
                harmonics=harmonics,                # Fourier orders (electrical)
                n_basis=n_basis,                    # 1 + 2 * len(harmonics)
                data=stack.ravel(order="F"),        # the packed coefficients
                p=lut["p"],                         # pole pairs
                theta_offset_e=theta_offset_e,      # d axis at FEM angle 0
                inversion_error=worst,              # worst flux error [Wb]
                lam_0=zero_current_flux(lut))       # theta-mean flux at i = 0 [Wb]


def zero_current_flux(lut):
    """The theta-averaged (lam_d, lam_q) at zero current [Wb]: the magnets'
    flux, and the natural starting point of a simulation."""
    return np.array([ForwardMap(lut["i_d"], lut["i_q"], lut[k].mean(axis=2))([0.0, 0.0])[0]
                     for k in ("lam_d", "lam_q")])


def mean_tables(tables):
    """Only the theta-mean of :func:`build_flux_tables`' output: one B-spline
    data block with the three outputs (i_d, i_q, T) on the same flux grid.

    The mean-value DAE machine uses it: in the rotor's dq frame and without
    the spatial ripple, currents and torque depend on the fluxes alone.
    """
    n_d, n_q = len(tables["grid_d"]), len(tables["grid_q"])
    stack = tables["data"].reshape((3 * tables["n_basis"], n_d, n_q), order="F")
    return dict(grid_d=tables["grid_d"], grid_q=tables["grid_q"],
                data=stack[0:3].ravel(order="F"),   # basis 0 (the mean) of i_d, i_q, T
                p=tables["p"], lam_0=tables["lam_0"])


def save_fea_lut(path, lut):
    """Write a table in the layout :func:`load_fea_lut` reads (the inverse
    of it: theta goes back to electrical degrees)."""
    np.savez_compressed(path, i_d=lut["i_d"], i_q=lut["i_q"],
                        th_elec=np.degrees(lut["theta_e"]),
                        Fd=lut["lam_d"], Fq=lut["lam_q"], T=lut["T"], p=lut["p"])
