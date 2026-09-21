# -*- coding: utf-8 -*-
r"""
Residue-based PSS tuning for pydae-bps systems.

Takes a system whose stabiliser gains are all zero and works down the
electromechanical modes, worst damped first, designing one PSS at a time until
every reachable mode clears a target damping ratio.

**The loop**

For the worst damped mode :math:`\lambda`, the machine with the largest residue

.. math::

    R_k = (C_i v_k)(w_k^T B_j)

on the loop it would close --- actuator :math:`v_{ref}` (the AVR summing junction,
where :math:`v_{pss}` injects), sensor :math:`\omega` (what the stabiliser
measures) --- is the one with authority over it.  The closed-loop shift for a
small gain is :math:`\Delta\lambda_k \approx R_k H(\lambda_k)`, so pushing the
mode straight left requires

.. math::

    \arg R_k + \arg H(\lambda_k) = 180^\circ

and the lead-lag chain has to supply what the washout does not:

.. math::

    \phi = 180^\circ - \arg R_k
           - \arg\!\left(\frac{\lambda T_w}{1 + \lambda T_w}\right)

**Why a band and not an eigenvalue**

Solving that for one :math:`\lambda` gives a network that is the right shape at
exactly one frequency.  A stabiliser centred on a 1 Hz local mode is the wrong
shape at 0.34 Hz, which is how a local design leaks damping out of the inter-area
mode -- the coupling NTS-SEPE has in mind when it requires the gain *and* the
phase network to be determined *considerando varios modos*, so that the design
damps the inter-area mode *sin deteriorar los modos locales*.

So the compensator is fitted across every mode the machine has authority over
(:func:`authority_modes`), weighted by :math:`|R|`, with :math:`\alpha` bounded by
what the stages can deliver and the lag time constant bounded to stay realisable.

**Why the objective is the worst mode**

The gain is chosen to maximise the damping of the worst electromechanical mode in
the system, not of the mode being targeted.  Optimising the target mode and
vetoing afterwards cannot trade a local mode against an inter-area one -- it can
only refuse to move, and a sequence of individually acceptable refusals still
walks downhill.  Selecting on the floor makes the trade-off the criterion.

**Things that look right and are not**

Several of these produce plausible numbers while being wrong, so they are guarded
rather than left to inspection:

- The sensor is :math:`\omega`, not the terminal voltage.  Using ``V`` gives a
  residue angle that can be 150 deg off.
- The compensator supplies :math:`180 - \arg R`, not :math:`\arg R`.  The latter
  turns the lead network into a lag network.
- Modes are filtered by damping *before* being ranked by ``|R|``.  A well damped
  mode can carry a large residue, and the ``T_2 = 1/(w*sqrt(alpha))`` seed quietly
  stops meaning anything once the eigenvalue leaves the neighbourhood of the jw
  axis.
- A stabiliser is kept only if it raises the *worst* mode.  Judged on its own
  mode, a machine with no authority can always be handed a large enough gain to
  gain a fraction of a point while giving it back elsewhere; doing that for
  machine after machine is how a run ends up behind where it started.
- The host is chosen by ``|R|``, not by participation.  A large machine can
  dominate an inter-area mode it has no authority over, and participation will
  happily point the design at it.
- ``ini()`` is called once.  A second call on the same model converges somewhere
  else entirely; and it is not needed, since ``v_pss = 0`` at equilibrium means no
  PSS parameter moves the operating point.
- Gain-sweep rows where the tracked mode was lost are discarded.  Otherwise a mode
  swap reads as a spectacular damping result.

**Usage**::

    from pydae.bps.utils.pss_tuner import tune_psss

    report = tune_psss(model, 'system.hjson', zeta_target=0.07)
    print(report.markdown())

Reference: the residue method as in Pagola/Rouco/Perez-Arriaga; the
damping-torque equivalent is the P-Vr / GEP formulation of PES-TR18 section 6.3,
cross-checkable here with :func:`gep`.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np

from pydae.ssa import eig, lead_design
from pydae.bps.utils.reporter import _load_data, _md_table


# ─────────────────────────────────────────────────────────────────────────────
# PSS model adapters
# ─────────────────────────────────────────────────────────────────────────────
#
# The supported stabilisers share one topology -- speed in, washout, lead-lag
# chain, gain -- and differ only in parameter naming and stage count.  Keys are
# the ``pss.type`` strings dispatched by pydae.bps.psss.psss.add_pss.
#
# Base names serve twice: the model parameter is "<base>_pss_<machine>", and the
# field inside the machine's "pss" block in the hjson is "<base>".

PSS_MODELS = {
    'pss_kundur_2': {'gain': 'K_stab', 'washout': 'T_w',
                     'stages': [('T_1', 'T_2'), ('T_3', 'T_4')]},
    'pss_kundur_1': {'gain': 'K_stab', 'washout': 'T_w',
                     'stages': [('T_1', 'T_2')]},
    'pss1a':        {'gain': 'K_s',    'washout': 'T_5',
                     'stages': [('T_1', 'T_2'), ('T_3', 'T_4')]},
}


def _spec(pss_type):
    if pss_type not in PSS_MODELS:
        raise ValueError(
            f"pss type '{pss_type}' is not supported by the tuner. "
            f"Supported: {', '.join(sorted(PSS_MODELS))}. "
            f"Dual-input models (pss2a, pss2) need a different phase model.")
    return PSS_MODELS[pss_type]


def _param(base, machine):
    """Model parameter name for a pss field, e.g. K_stab -> K_stab_pss_G1."""
    return f'{base}_pss_{machine}'


def machines_with_pss(data):
    """[(machine name, pss type)] for every synchronous machine carrying a pss block."""
    d = _load_data(data)
    return [(s['name'], s['pss']['type']) for s in d.get('syns', []) if 'pss' in s]


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Design:
    """One tuned stabiliser."""
    machine: str
    pss_type: str
    f_hz: float
    zeta_before: float
    zeta_after: float
    lam_before: complex
    lam_after: complex
    R_abs: float
    R_deg: float
    participation: float
    phi_deg: float                    # total phase asked of the lead-lag chain
    alpha: float
    T_1: float
    T_2: float
    gain: float
    gain_param: str
    n_band: int = 1                   # modes the compensator was fitted across
    phase_err_deg: float = 0.0        # |R|-weighted RMS phase error over that band
    warning: str = ''

    def hjson_fields(self):
        """The pss-block fields this design sets, as {field: value}."""
        spec = _spec(self.pss_type)
        out = {spec['gain']: self.gain}
        for num, den in spec['stages']:
            out[num] = self.T_1
            out[den] = self.T_2
        return out


@dataclass
class Skipped:
    """A mode left untuned, and why."""
    f_hz: float
    zeta: float
    lam: complex
    reason: str
    machine: str = ''
    detail: str = ''


@dataclass
class TuningReport:
    designs: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    zeta_target: float = 0.07
    rounds: int = 0
    floor_before: float = float('nan')     # worst electromechanical damping, open loop
    floor_after: float = float('nan')      # ... and once every design is in
    modes_before: list = field(default_factory=list)   # open-loop band eigenvalues
    reverted: bool = False

    @property
    def improved(self):
        return self.floor_after >= self.floor_before - 1e-9

    def markdown(self):
        lines = [f'Worst electromechanical damping: '
                 f'{100*self.floor_before:.2f} % -> {100*self.floor_after:.2f} % '
                 f'(target {100*self.zeta_target:.0f} %)\n']
        if self.reverted:
            lines.append('**Reverted**: the run left the system worse than it '
                         'started, so every gain was returned to zero.\n')
        if self.designs:
            rows = [[d.machine, d.f_hz, 100*d.zeta_before, 100*d.zeta_after,
                     d.R_abs, d.R_deg, d.phi_deg, d.alpha, d.T_1, d.T_2, d.gain,
                     d.n_band, d.phase_err_deg, d.warning or '']
                    for d in self.designs]
            lines.append('### PSS designs\n')
            lines.append(_md_table(
                ['machine', 'f [Hz]', 'zeta0 [%]', 'zeta [%]', '|R|', 'arg R [deg]',
                 'phi [deg]', 'alpha', 'T_1', 'T_2', 'gain', 'band', 'err [deg]',
                 'note'],
                rows,
                ['{}', '{:.3f}', '{:.2f}', '{:.2f}', '{:.4f}', '{:.1f}', '{:.1f}',
                 '{:.2f}', '{:.4f}', '{:.4f}', '{:.4f}', '{}', '{:.1f}', '{}']))
        else:
            lines.append('### PSS designs\n\n_none_')

        if self.skipped:
            rows = [[s.f_hz, 100*s.zeta, s.machine or '-', s.reason, s.detail]
                    for s in self.skipped]
            lines.append('\n### Modes left untuned\n')
            lines.append(_md_table(
                ['f [Hz]', 'zeta [%]', 'best host', 'reason', 'detail'],
                rows, ['{:.3f}', '{:.2f}', '{}', '{}', '{}']))
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Modal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _zeta(lam):
    return -lam.real / abs(lam)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _osc_modes(model, f_band):
    """Indices of one-per-conjugate-pair oscillatory modes inside the band."""
    ev = model.eigenvalues
    f_lo, f_hi = f_band
    return [k for k in range(len(ev))
            if ev[k].imag > 0.05 and f_lo <= ev[k].imag/(2*np.pi) <= f_hi]


def _is_dismissed(model, k, dismissed):
    ev, V = model.eigenvalues, model.right_eigenvectors
    for vec, imag in dismissed:
        if (abs(np.vdot(vec, _unit(V[:, k]))) >= 0.8
                and abs(ev[k].imag - imag) <= 0.3 * imag):
            return True
    return False


def _tried_against(model, k, attempts):
    """Machines already tried against this mode, matched across re-linearisations."""
    for vec, imag, machines in attempts:
        if (abs(np.vdot(vec, _unit(model.right_eigenvectors[:, k]))) >= 0.8
                and abs(model.eigenvalues[k].imag - imag) <= 0.3 * imag):
            return machines
    return set()


def _record_attempt(model, k, machine, attempts):
    machines = _tried_against(model, k, attempts)
    if machines:
        machines.add(machine)
    else:
        attempts.append((_unit(model.right_eigenvectors[:, k]),
                         model.eigenvalues[k].imag, {machine}))


def _candidate_modes(model, f_band, zeta_target, dismissed):
    """Under-damped electromechanical modes that have not been ruled out."""
    ev = model.eigenvalues
    return [k for k in _osc_modes(model, f_band)
            if _zeta(ev[k]) < zeta_target and not _is_dismissed(model, k, dismissed)]


def _global_floor(model, f_band):
    """Lowest damping among *all* electromechanical modes.

    Deliberately not filtered by the dismissed list.  A mode no stabiliser can fix
    still must not be allowed to get worse, and excluding it here would let every
    later machine degrade it for free.
    """
    ev = model.eigenvalues
    return min((_zeta(ev[k]) for k in _osc_modes(model, f_band)), default=1.0)


def _track(model, ref_vec, ref_imag, tol_corr=0.8, tol_f=0.3):
    """Follow a mode after a parameter change.

    Returns ``(eigenvalue, correlation, held)``.  ``held`` is False when the best
    match has drifted too far to still be the same mode -- at high gain the
    tracker can jump to an unrelated eigenvalue, and believing it turns a mode
    swap into an apparently excellent damping result.
    """
    ev, V = model.eigenvalues, model.right_eigenvectors
    osc = [k for k in range(len(ev)) if ev[k].imag > 0.05]
    corr = [abs(np.vdot(ref_vec, _unit(V[:, k]))) for k in osc]
    j = osc[int(np.argmax(corr))]
    best = max(corr)
    held = best >= tol_corr and abs(ev[j].imag - ref_imag) <= tol_f * ref_imag
    return ev[j], best, held


# ─────────────────────────────────────────────────────────────────────────────
# Residue and design
# ─────────────────────────────────────────────────────────────────────────────

def loop_residue(model, machine, mode_idx):
    r"""Residue of the loop the stabiliser closes: ``v_ref_<m>`` -> ``omega_<m>``.

    The PSS output enters the AVR summing junction as ``v_2 = v_ref - V + v_pss``,
    i.e. through the same path and sign as ``v_ref``, so the ``v_ref`` column of B
    is the actuator.  The sensor is the rotor speed, which is what every supported
    stabiliser measures.
    """
    u_name, x_name = f'v_ref_{machine}', f'omega_{machine}'
    if u_name not in model.u_run_list:
        raise ValueError(f'{u_name} is not a run input of this model')
    if x_name not in model.x_list:
        raise ValueError(f'{x_name} is not a state of this model')

    B = np.array(model.B)[:, model.u_run_list.index(u_name)].reshape(-1, 1)
    C = np.zeros((1, model.N_x))
    C[0, model.x_list.index(x_name)] = 1.0

    V, W = model.right_eigenvectors, model.left_eigenvectors
    return ((C @ V[:, [mode_idx]]) * (W[[mode_idx], :] @ B))[0, 0]


def required_phase(lam, R, T_w):
    """Phase the lead-lag chain must supply, plus the washout evaluated at ``lam``.

    Everything is taken at ``s = lam`` rather than on the jw axis, so the answer
    stays right for modes that are not especially lightly damped.
    """
    G_w = (lam * T_w) / (1.0 + lam * T_w)
    return _wrap(180.0 - np.angle(R, deg=True) - np.angle(G_w, deg=True)), G_w


def design_compensator(lam, R, T_w, n_stages, phi_max_per_stage=70.0, tol=1e-9):
    """Lead-lag time constants for one machine.

    Returns ``((T_1, T_2, phi, alpha, H_unit), '')`` where ``H_unit`` is the
    washout and compensator at ``lam`` with unit gain, so ``dlambda/dK = R*H_unit``.

    Returns ``(None, reason)`` when the required phase exceeds what the available
    stages can deliver.  In practice that means the mode sits above the band where
    the excitation system can produce damping torque at all -- confirm with
    :func:`gep`, whose lag will be approaching -180 deg there.

    The textbook construction centres the maximum phase at ``omega``, so it
    delivers the requested angle on the jw axis -- but the residue formula
    evaluates ``H`` at ``s = lam``, which sits off that axis by the mode's real
    part.  The gap is a degree or two for a lightly damped mode and grows with
    damping.  Rather than accept it, the requested angle is corrected until the
    compensator hits the target *at lam*, which is where it has to be right.
    """
    phi, G_w = required_phase(lam, R, T_w)

    ask = phi
    T_1 = T_2 = None
    for _ in range(40):
        per_stage = ask / n_stages
        if abs(per_stage) > phi_max_per_stage:
            return None, (f'needs {phi:.1f} deg, i.e. {phi/n_stages:.1f} deg/stage '
                          f'over {n_stages} stage(s); limit {phi_max_per_stage:.0f}')
        T_1, T_2 = lead_design(np.deg2rad(per_stage), lam.imag)
        got = np.angle(((1.0 + T_1 * lam) / (1.0 + T_2 * lam)) ** n_stages, deg=True)
        err = _wrap(phi - got)
        if abs(err) < tol:
            break
        ask += err
    else:
        return None, (f'phase correction at s = lambda did not converge '
                      f'(target {phi:.1f} deg, reached {got:.1f} deg)')

    alpha = T_1 / T_2
    G_c = ((1.0 + T_1 * lam) / (1.0 + T_2 * lam)) ** n_stages
    return (T_1, T_2, phi, alpha, G_w * G_c), ''


def authority_modes(model, machine, f_band, zeta_max, frac=0.1):
    """``[(k, R_k)]`` for the lightly damped modes this machine can actually move.

    Everything within *frac* of the machine's own best residue counts.  This is the
    set the compensator has to be right for -- typically its local mode plus any
    inter-area mode it participates in.
    """
    ks = [k for k in _osc_modes(model, f_band)
          if _zeta(model.eigenvalues[k]) <= zeta_max]
    if not ks:
        return []
    residues = {k: loop_residue(model, machine, k) for k in ks}
    best = max(abs(v) for v in residues.values())
    if best <= 0.0:
        return []
    return [(k, residues[k]) for k in ks if abs(residues[k]) >= frac * best]


def design_compensator_band(lams, Rs, T_w, n_stages, phi_max_per_stage=70.0,
                            t2_bounds=(0.01, 10.0)):
    """Fit one lead-lag chain to the phase demanded at several modes at once.

    Centring the maximum phase on a single eigenvalue makes the network the right
    shape at exactly one frequency.  A stabiliser tuned that way for a 1 Hz local
    mode is simply the wrong shape at 0.34 Hz, which is how a local design leaks
    damping out of the inter-area mode -- the coupling the NTS warns about when it
    requires the gain and the phase network to be determined *considerando varios
    modos*.

    Fits ``(T_2, alpha)`` in logs, weighting each mode by ``|R|`` so the effort
    goes where the machine has authority, and bounding ``alpha`` by what
    *phi_max_per_stage* permits.  With one mode in the set this reproduces
    :func:`design_compensator`.

    ``t2_bounds`` keeps the lag time constant realisable.  Fitting a wide band
    otherwise drives ``T_2`` into the milliseconds -- a corner at tens of Hz, with
    the compensator's full high-frequency gain sitting on top of the speed
    measurement noise.  A stabiliser like that fits the phase and cannot be built.

    Returns ``((T_1, T_2, phi_dominant, alpha, H_unit_at_dominant, err_deg), '')``
    or ``(None, reason)``.
    """
    from scipy.optimize import least_squares

    targets, weights, G_ws = [], [], []
    for lam, R in zip(lams, Rs):
        phi, G_w = required_phase(lam, R, T_w)
        targets.append(phi)
        weights.append(abs(R))
        G_ws.append(G_w)

    j = int(np.argmax(weights))                   # the mode that matters most
    seed, reason = design_compensator(lams[j], Rs[j], T_w, n_stages,
                                      phi_max_per_stage)
    if seed is None:
        return None, reason
    T_1s, T_2s = seed[0], seed[1]

    s = np.sin(np.deg2rad(phi_max_per_stage))
    alpha_max = (1.0 + s) / (1.0 - s)
    log_a_max = np.log(alpha_max)

    def _phase(T_1, T_2, lam):
        return np.angle(((1.0 + T_1 * lam) / (1.0 + T_2 * lam)) ** n_stages, deg=True)

    def residuals(p):
        T_2 = np.exp(p[0])
        T_1 = T_2 * np.exp(p[1])
        return [np.sqrt(w) * _wrap(_phase(T_1, T_2, lam) - phi)
                for lam, phi, w in zip(lams, targets, weights)]

    lo, hi = np.log(t2_bounds[0]), np.log(t2_bounds[1])
    p0 = [np.clip(np.log(T_2s), lo, hi),
          np.clip(np.log(T_1s / T_2s), -log_a_max, log_a_max)]
    sol = least_squares(residuals, p0,
                        bounds=([lo, -log_a_max], [hi, log_a_max]))

    T_2 = float(np.exp(sol.x[0]))
    alpha = float(np.exp(sol.x[1]))
    T_1 = alpha * T_2

    errs = [abs(_wrap(_phase(T_1, T_2, lam) - phi))
            for lam, phi in zip(lams, targets)]
    err_deg = float(np.sqrt(np.average(np.square(errs), weights=weights)))

    # The dominant mode is the one the gain will be swept against, so it is the
    # one that has to be compensated properly.
    if errs[j] > 45.0:
        return None, (f'no lead-lag fits the band: {errs[j]:.0f} deg short at the '
                      f'dominant mode ({lams[j].imag/(2*np.pi):.2f} Hz) over '
                      f'{n_stages} stage(s)')

    G_c = ((1.0 + T_1 * lams[j]) / (1.0 + T_2 * lams[j])) ** n_stages
    return (T_1, T_2, targets[j], alpha, G_ws[j] * G_c, err_deg), ''


# ─────────────────────────────────────────────────────────────────────────────
# Gain sweep
# ─────────────────────────────────────────────────────────────────────────────

def _gain_grid(lam, R, H_unit, zeta_target, n=12):
    """Gains bracketing the first-order estimate of what reaches ``zeta_target``.

    ``dlambda/dK = R*H_unit`` is known analytically, so rather than guess a fixed
    grid we solve for the gain that would place the eigenvalue at the target
    damping and spread points around it.  The estimate is only first order, but it
    gets the decade right, which no fixed grid can do across machines whose useful
    gains span 0.05 to 20.
    """
    sigma_t = -zeta_target * abs(lam) / math.sqrt(1.0 - zeta_target**2)
    slope = float(np.real(R * H_unit))
    if slope >= -1e-12:                    # not heading left; fall back to a broad sweep
        K_est = 1.0
    else:
        K_est = max((sigma_t - lam.real) / slope, 1e-9)
    return [0.0] + list(K_est * np.logspace(-1.5, 1.0, n))


def sweep_gain(model, gain_param, mode_idx, R, H_unit, zeta_target,
               f_band=(0.1, 2.5), gains=None, verbose=False):
    """Move the gain and record where the mode actually goes.

    No ``ini()``: every supported stabiliser has ``v_pss = 0`` at equilibrium, so
    the operating point depends on neither the gain nor the time constants -- only
    the Jacobian does.  (Re-running ``ini()`` would also not be reproducible.)

    Each row also carries the worst damping in the whole electromechanical band at
    that gain, which is what the gain is actually chosen on: damping the target
    mode at the expense of another is not an improvement.

    Returns ``[(gain, lam, zeta, floor, corr, held)]``, first row the open loop.
    """
    lam0 = model.eigenvalues[mode_idx]
    ref_vec = _unit(model.right_eigenvectors[:, mode_idx])
    if gains is None:
        gains = _gain_grid(lam0, R, H_unit, zeta_target)

    trace = []
    for K in gains:
        model.set_value(gain_param, K)
        eig(model)
        lam, corr, held = _track(model, ref_vec, lam0.imag)
        floor = _global_floor(model, f_band)
        trace.append((K, lam, _zeta(lam), floor, corr, held))
        if verbose:
            pred = lam0 + K * R * H_unit
            print(f'    K={K:10.5f}  lam={lam.real:8.4f}{lam.imag:+8.4f}j  '
                  f'zeta={100*_zeta(lam):6.2f}%  floor={100*floor:6.2f}%   '
                  f'pred={pred.real:8.4f}{pred.imag:+8.4f}j  corr={corr:.3f}'
                  f'{"" if held else "   LOST"}')
    return trace


def pick_gain(trace, zeta_target, gain_max=float('inf')):
    """Choose the gain that leaves the *whole system* best damped.

    The selection is on the global floor, not on the target mode.  Optimising the
    target mode and vetoing afterwards -- the earlier design -- cannot trade a
    local mode against an inter-area one; it can only refuse to move, and a
    sequence of individually acceptable refusals still walks downhill.  Damping a
    local mode by half a point while handing the inter-area mode a point back is
    not an improvement, and choosing on the floor is what makes that explicit.

    Rows where the tracker lost the mode are discarded first -- they are the ones
    that look best.  Gains above *gain_max* are discarded too: a stabiliser needing
    a gain of several hundred is not a design, it is a machine with no authority
    over the mode being asked to supply it anyway.

    Returns ``(gain, zeta, lam, floor)`` or ``(None, None, None, None)``.
    """
    kept = [(K, lam, z, fl) for K, lam, z, fl, _c, held in trace
            if held and 0 < K <= gain_max]
    if not kept:
        return None, None, None, None

    reaching = [row for row in kept if row[3] >= zeta_target]
    if reaching:
        pool = reaching                      # target met: take the cheapest gain
    else:
        best = max(row[3] for row in kept)
        pool = [row for row in kept if row[3] >= best - 0.005]

    K, lam, z, fl = min(pool, key=lambda row: row[0])
    return K, z, lam, fl


def _row_at(trace, gain):
    for K, lam, z, fl, _c, _h in trace:
        if K == gain:
            return lam, z, fl
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# GEP diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def gep(model, machine, freqs_hz):
    r"""``arg(GEP)`` and ``|GEP|`` over frequency, with shaft dynamics disabled.

    GEP is the transfer function from the AVR summing junction to electrical power
    with the rotor held still -- Generator x Exciter x Power system.  Deleting the
    machine's ``delta`` and ``omega`` rows and columns from A is exactly the
    "shaft dynamics are disabled" condition of the P-Vr and GEP methods.

    Use it when the tuner skips a machine on required phase.  The classical rule is
    ``arg(H) = -arg(GEP)``, so a GEP lag approaching -180 deg means no lead-lag
    chain can compensate it and the mode is simply above the band where the
    excitation system can produce damping torque.  Note the lag is a property of
    the frequency far more than of the machine.

    Returns ``(angles_deg, magnitudes)`` as arrays.
    """
    A = np.array(model.A)
    B = np.array(model.B)[:, model.u_run_list.index(f'v_ref_{machine}')].reshape(-1, 1)
    i_om = model.x_list.index(f'omega_{machine}')
    i_de = model.x_list.index(f'delta_{machine}')
    keep = [i for i in range(model.N_x) if i not in (i_om, i_de)]

    pe = f'p_e_{machine}'
    if pe not in model.z_list:
        raise ValueError(f'{pe} is not an output of this model; cannot form GEP')
    C = np.array(model.C)[model.z_list.index(pe), :].reshape(1, -1)[:, keep]

    A_f, B_f = A[np.ix_(keep, keep)], B[keep, :]
    eye = np.eye(len(keep))
    ang, mag = [], []
    for f in np.atleast_1d(freqs_hz):
        G = (C @ np.linalg.solve(1j*2*np.pi*f*eye - A_f, B_f))[0, 0]
        ang.append(np.angle(G, deg=True))
        mag.append(abs(G))
    return np.array(ang), np.array(mag)


# ─────────────────────────────────────────────────────────────────────────────
# The tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def tune_psss(model, data, zeta_target=0.07, f_band=(0.1, 2.5),
              phi_max_per_stage=70.0, r_min=1e-3, gain_max=50.0,
              min_improvement=0.002, authority_frac=0.1, max_rounds=None,
              second_pass=True,
              revert_if_worse=True, xy_0=None, write=False, verbose=True):
    """Tune stabilisers one at a time until every reachable mode clears the target.

    Parameters
    ----------
    model : CasadiModel or Model
        Built but **not yet initialised** -- ``ini()`` is called here, exactly
        once, and calling it beforehand yourself lands the tuner on a different
        operating point than the one you measured.
    data : str | dict | BpsBuilder
        The system description, for the list of machines carrying a pss block.
    zeta_target : float
        Damping ratio every electromechanical mode should reach.
    f_band : (float, float)
        Frequency window searched for electromechanical modes, in Hz.
    phi_max_per_stage : float
        Most phase one lead-lag stage may be asked for.  Past roughly 70 deg the
        spread ``alpha`` explodes and the network amplifies measurement noise; a
        machine needing more is skipped and reported rather than tuned.
    r_min : float
        Residue magnitude below which no machine is considered to have authority
        over the mode.
    authority_frac : float
        A mode joins a machine's fitting band when its residue is at least this
        fraction of that machine's largest.  The compensator is then fitted across
        the whole band instead of being centred on one eigenvalue, which is what
        keeps a local-mode design from leaking damping out of the inter-area mode.
    gain_max : float
        Largest stabiliser gain the tuner will propose.  Without a cap the sweep
        happily reaches into the hundreds for a machine that barely sees the mode.
    min_improvement : float
        How much a stabiliser must raise the *worst* electromechanical mode (in
        ratio, so 0.002 is a fifth of a point) for it to be kept.  It can be small
        because the quantity being improved is the floor, so every accepted design
        provably leaves the system better damped than it found it; on a large
        network real progress genuinely arrives a fraction of a point at a time
        from many machines.
    second_pass : bool
        Re-tune machines whose mode is still short of target once the others are
        in.  Not cosmetic: a machine designed against the untuned plant can do
        markedly better after its neighbours are stabilised.
    revert_if_worse : bool
        If the completed run leaves the worst electromechanical mode below where
        it started, put every gain back to zero and say so.  The per-machine guard
        only compares one step at a time, so a sequence of individually acceptable
        moves can still end up behind the starting point.
    xy_0 : str or None
        Optional initialisation seed file passed through to ``model.ini()``.
    write : bool or str
        Write the resulting parameters into the pss blocks of *data*, which must
        then be a path.  Pass a path instead of ``True`` to write a new file and
        leave the source alone.  Stabilisers the run did not design are commented
        out, because that is the state the reported result was measured in.  Off
        by default: a tuning run is a proposal until someone looks at it.

    Returns
    -------
    TuningReport
    """
    pss_list = machines_with_pss(data)
    if not pss_list:
        raise ValueError('no synchronous machine in this system carries a pss block')
    types = dict(pss_list)
    for _m, t in pss_list:
        _spec(t)                                   # fail early on unsupported types

    # Open every stabiliser loop, then initialise -- once.
    for m, t in pss_list:
        model.set_value(_param(_spec(t)['gain'], m), 0.0)
    model.ini({}, xy_0=xy_0) if xy_0 else model.ini({})
    eig(model)

    # Recorded here because it cannot be recovered afterwards: re-running ini() on
    # the same model converges to a different operating point, so a caller cannot
    # measure the open loop for itself without invalidating the comparison.
    report = TuningReport(
        zeta_target=zeta_target,
        floor_before=_global_floor(model, f_band),
        modes_before=[model.eigenvalues[k] for k in _osc_modes(model, f_band)])
    tuned = {}                                     # machine -> Design
    dismissed = []                                 # (eigenvector, imag) ruled out
    attempts = []                                  # (eigenvector, imag, {machines})
    limit = max_rounds if max_rounds is not None else 4 * len(pss_list) + 8

    for _ in range(limit):
        cands = _candidate_modes(model, f_band, zeta_target, dismissed)
        if not cands:
            break
        k = min(cands, key=lambda k: _zeta(model.eigenvalues[k]))
        lam = model.eigenvalues[k]
        report.rounds += 1

        # An inter-area mode has no single host by definition -- on the Spanish
        # case the best residue for the 0.34 Hz mode is 0.027 against 0.13-0.15
        # for local modes.  So a mode is retired only once every machine has been
        # tried against it, not when the first one fails.
        already = _tried_against(model, k, attempts)
        free = [m for m, _t in pss_list if m not in tuned and m not in already]
        if not free:
            report.skipped.append(Skipped(
                lam.imag/(2*np.pi), _zeta(lam), lam, 'no machine left to try',
                '', f'{len(already)} machine(s) tried, none improved it'))
            dismissed.append((_unit(model.right_eigenvectors[:, k]), lam.imag))
            continue

        residues = {m: loop_residue(model, m, k) for m in free}
        host = max(residues, key=lambda m: abs(residues[m]))
        R = residues[host]

        if abs(R) < r_min:
            report.skipped.append(Skipped(
                lam.imag/(2*np.pi), _zeta(lam), lam, 'no host with authority',
                host, f'best untried |R| = {abs(R):.2e} < r_min = {r_min:g}'))
            dismissed.append((_unit(model.right_eigenvectors[:, k]), lam.imag))
            continue

        _record_attempt(model, k, host, attempts)
        outcome = _tune_one(model, host, types[host], k, R, zeta_target,
                            phi_max_per_stage, f_band, gain_max, min_improvement,
                            authority_frac, verbose)
        if isinstance(outcome, Skipped):
            report.skipped.append(outcome)      # mode stays alive for the next machine
            continue

        tuned[host] = outcome

    if second_pass and tuned:
        _second_pass(model, types, tuned, zeta_target, f_band, phi_max_per_stage,
                     r_min, gain_max, min_improvement, authority_frac, dismissed,
                     verbose)

    report.designs = list(tuned.values())
    report.floor_after = _global_floor(model, f_band)

    # Last line of defence.  The per-machine guard only ever compares one step at
    # a time, so a sequence of individually acceptable moves can still end up
    # behind where it started -- and a tuning run that degrades the system is
    # never what was wanted.
    if revert_if_worse and report.floor_after < report.floor_before - 1e-9:
        for m, t in pss_list:
            model.set_value(_param(_spec(t)['gain'], m), 0.0)
        eig(model)
        report.reverted = True
        report.designs = []
        report.floor_after = _global_floor(model, f_band)

    if write and report.reverted:
        raise RuntimeError('refusing to write: the tuning run left the system worse '
                           'than it started and was reverted')

    if write:
        if not isinstance(data, str):
            raise TypeError('write=True needs `data` to be a path to the hjson file')
        designed = {d.machine for d in report.designs}
        untuned = [m for m, _t in pss_list if m not in designed]
        target = write_pss_params(data, report.designs,
                                  out_path=(write if isinstance(write, str) else None),
                                  disable=untuned)
        if verbose:
            print(f'\nwrote {len(report.designs)} design(s) and commented out '
                  f'{len(untuned)} unused stabiliser(s) in {target}')

    if verbose:
        print()
        print(report.markdown())
    return report


def _tune_one(model, host, pss_type, k, R, zeta_target, phi_max_per_stage,
              f_band, gain_max, min_improvement, authority_frac, verbose):
    """Design, sweep and commit one stabiliser.  Returns a Design or a Skipped."""
    spec = _spec(pss_type)
    lam = model.eigenvalues[k]
    T_w = model.get_value(_param(spec['washout'], host))
    n_stages = len(spec['stages'])
    gain_param = _param(spec['gain'], host)
    part = float(model.participation[model.x_list.index(f'omega_{host}'), k])

    # Fit the compensator to every mode this machine has authority over, with the
    # target mode forced into the set so it is always represented.
    band = authority_modes(model, host, f_band, zeta_target, authority_frac)
    if k not in [kk for kk, _r in band]:
        band = [(k, R)] + band
    lams = [model.eigenvalues[kk] for kk, _r in band]
    Rs = [r for _kk, r in band]

    out, reason = design_compensator_band(lams, Rs, T_w, n_stages, phi_max_per_stage)
    if out is None:
        return Skipped(lam.imag/(2*np.pi), _zeta(lam), lam,
                       'required phase out of reach', host, reason)

    T_1, T_2, phi, alpha, H_unit, err_deg = out

    # H_unit came back at the band's dominant mode; the sweep predicts the target
    # mode, so re-evaluate there.
    G_w = (lam * T_w) / (1.0 + lam * T_w)
    H_unit = G_w * ((1.0 + T_1 * lam) / (1.0 + T_2 * lam)) ** n_stages

    if verbose:
        print(f'\n{host}: mode {lam.imag/(2*np.pi):.3f} Hz, zeta {100*_zeta(lam):.2f}% '
              f'-> |R| {abs(R):.4f} < {np.angle(R, deg=True):.1f} deg, '
              f'phi {phi:.1f} deg, alpha {alpha:.2f}, '
              f'T_1 {T_1:.4f}, T_2 {T_2:.4f}, '
              f'{len(band)} mode(s) in band, phase err {err_deg:.1f} deg')

    for num, den in spec['stages']:
        model.set_value(_param(num, host), T_1)
        model.set_value(_param(den, host), T_2)

    floor_before = _global_floor(model, f_band)
    trace = sweep_gain(model, gain_param, k, R, H_unit, zeta_target, f_band,
                       verbose=verbose)
    gain, zeta_after, lam_after, _fl = pick_gain(trace, zeta_target, gain_max)

    if gain is None:
        model.set_value(gain_param, 0.0)
        eig(model)
        return Skipped(lam.imag/(2*np.pi), _zeta(lam), lam,
                       'no usable gain', host,
                       f'every gain within the cap ({gain_max:g}) either lost the '
                       f'mode or did nothing')

    gain, zeta_after, lam_after, warning = _commit(
        model, gain_param, gain, zeta_after, lam_after, trace,
        floor_before, f_band)

    # Judged on the worst mode in the system, not on the target mode.  A machine
    # with no real authority can always be handed a large enough gain to nudge its
    # own mode a fraction of a point while giving it back somewhere else; doing
    # that for machine after machine is how a run ends up behind where it started.
    gained = _global_floor(model, f_band) - floor_before
    if gained < min_improvement:
        model.set_value(gain_param, 0.0)
        eig(model)
        return Skipped(lam.imag/(2*np.pi), _zeta(lam), lam,
                       'improvement not worth a stabiliser', host,
                       f'best gain {gain:.3g} moved the worst mode by only '
                       f'{100*gained:.2f} points')

    return Design(machine=host, pss_type=pss_type, f_hz=lam.imag/(2*np.pi),
                  zeta_before=_zeta(lam), zeta_after=zeta_after,
                  lam_before=lam, lam_after=lam_after,
                  R_abs=abs(R), R_deg=float(np.angle(R, deg=True)),
                  participation=part, phi_deg=float(phi), alpha=float(alpha),
                  T_1=float(T_1), T_2=float(T_2), gain=float(gain),
                  gain_param=gain_param, n_band=len(band),
                  phase_err_deg=float(err_deg), warning=warning)


def _commit(model, gain_param, gain, zeta_after, lam_after, trace,
            floor_before, f_band, tol=1e-6):
    """Commit the chosen gain and re-check the floor on the real model.

    ``pick_gain`` already optimises the floor, so this is a backstop rather than
    the mechanism: it catches the case where the swept row and the committed model
    disagree because the tracker slipped.
    """
    model.set_value(gain_param, gain)
    eig(model)
    if _global_floor(model, f_band) >= floor_before - tol:
        return gain, zeta_after, lam_after, ''

    model.set_value(gain_param, 0.0)
    eig(model)
    lam0, z0, _fl0 = _row_at(trace, 0.0)
    return 0.0, z0, lam0, 'left at zero: the chosen gain degraded another mode'


def _second_pass(model, types, tuned, zeta_target, f_band, phi_max_per_stage,
                 r_min, gain_max, min_improvement, authority_frac, dismissed,
                 verbose, tol=0.005):
    """Re-tune machines now that the plant around them has changed.

    Judged on the *global* worst damping, never on a per-machine record.  A design
    recorded earlier carries the damping its mode had at the time, and later
    stabilisers may since have lifted that very mode -- acting on the stale number
    re-points the machine at a different mode and quietly gives up the
    contribution it was making.

    For the same reason the regression baseline is taken before this machine's
    loop is opened.  Measuring it afterwards makes the damage look like the
    starting point, and the guard then approves anything.
    """
    for host, old in list(tuned.items()):
        floor_before = _global_floor(model, f_band)
        if floor_before >= zeta_target:
            return                                  # every mode already clears it

        keep_gain = model.get_value(old.gain_param)
        model.set_value(old.gain_param, 0.0)        # open this loop only
        eig(model)

        cands = _candidate_modes(model, f_band, zeta_target, dismissed)
        if cands:
            residues = {k: loop_residue(model, host, k) for k in cands}
            k = max(residues, key=lambda k: abs(residues[k]))
            R = residues[k]
        else:
            k, R = None, 0.0

        if k is None or abs(R) < r_min:
            _restore(model, old)
            continue

        if verbose:
            print(f'\nsecond pass: {host}')
        outcome = _tune_one(model, host, types[host], k, R, zeta_target,
                            phi_max_per_stage, f_band, gain_max, min_improvement,
                            authority_frac, verbose)

        if isinstance(outcome, Skipped):
            _restore(model, old)
            continue

        if _global_floor(model, f_band) > floor_before + tol:
            tuned[host] = outcome
        else:
            _restore(model, old)


def _restore(model, design):
    """Put a machine's parameters back to a previously recorded design."""
    spec = _spec(design.pss_type)
    for num, den in spec['stages']:
        model.set_value(_param(num, design.machine), design.T_1)
        model.set_value(_param(den, design.machine), design.T_2)
    model.set_value(design.gain_param, design.gain)
    eig(model)


# ─────────────────────────────────────────────────────────────────────────────
# Writing designs back into the hjson
# ─────────────────────────────────────────────────────────────────────────────

def _pss_block_span(text, machine):
    """(start, end) of the braces of *machine*'s pss block in the raw hjson text.

    Works on the text rather than a parsed structure because round-tripping
    through ``hjson.dump`` reformats the whole file.  Both layouts in use are
    handled: expanded blocks with quoted keys, and single-line blocks with bare
    keys.
    """
    name_pat = re.compile(rf'"?name"?\s*:\s*"{re.escape(machine)}"')
    hit = name_pat.search(text)
    if hit is None:
        raise ValueError(f'machine "{machine}" not found in the data file')

    pss_pat = re.compile(r'"?pss"?\s*:\s*\{')
    pss_hit = pss_pat.search(text, hit.end())
    if pss_hit is None:
        raise ValueError(f'no pss block found after machine "{machine}"')

    # Guard against walking past this machine into the next one's block.
    other = re.compile(r'"?name"?\s*:\s*"[^"]*"').search(text, hit.end(), pss_hit.start())
    if other is not None:
        raise ValueError(f'machine "{machine}" appears to carry no pss block')

    start = pss_hit.end() - 1                     # at the '{'
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise ValueError(f'unterminated pss block for machine "{machine}"')


def _fmt(value):
    """Compact but exact enough for a time constant or a gain."""
    return f'{value:.6g}'


def _comment_out_pss(text, machine):
    """Comment out a machine's whole ``pss`` entry, key and trailing comma included.

    Preferred over writing a zero gain: a commented-out block means the machine
    genuinely has no stabiliser, so ``add_pss`` never runs and the model carries no
    dead PSS states.  A zero gain leaves the washout and lead-lag states in the
    system, decoupled but present.
    """
    start, end = _pss_block_span(text, machine)

    key = re.compile(r'"?pss"?\s*:\s*$')
    line_start = text.rfind('\n', 0, start) + 1
    if not key.search(text[line_start:start + 1].rstrip().rstrip('{').rstrip()):
        # the key sits on the same line as the brace; fall back to the line start
        pass
    entry_start = line_start

    entry_end = end
    tail = re.match(r'\s*,', text[end:])
    if tail:
        entry_end = end + tail.end()

    body = text[entry_start:entry_end]
    indent = re.match(r'[ \t]*', body).group(0)
    commented = '\n'.join(
        (f'{indent}// {ln[len(indent):]}' if ln.startswith(indent) else f'// {ln}')
        if ln.strip() else ln
        for ln in body.split('\n'))
    return text[:entry_start] + commented + text[entry_end:]


def write_pss_params(path, designs, out_path=None, disable=()):
    """Write tuned parameters into the pss blocks of an hjson file.

    Only the numeric fields each design sets are touched; layout, comments and
    every other field are preserved byte for byte.  Fields the stabiliser has but
    the tuner does not set (output limits, for instance) are left alone.

    *disable* is a list of machine names whose ``pss`` entry should be commented
    out.  A tuning run opens every stabiliser loop and designs only some of them,
    so the machines it did not design carried no stabiliser when the result was
    measured; writing the designs without disabling the rest produces a file that
    does not reproduce the run.

    Returns the path written.
    """
    with open(path, encoding='utf-8') as fh:
        text = fh.read()

    for machine in disable:
        text = _comment_out_pss(text, machine)

    for machine, fields in [(d.machine, d.hjson_fields()) for d in designs]:
        start, end = _pss_block_span(text, machine)
        block = text[start:end]
        for fieldname, value in fields.items():
            pat = re.compile(rf'("?{re.escape(fieldname)}"?\s*:\s*)'
                             r'(-?\d+\.?\d*(?:[eE][-+]?\d+)?)')
            block, n = pat.subn(lambda m: m.group(1) + _fmt(value), block, count=1)
            if n == 0:
                raise ValueError(f'field "{fieldname}" not found in the pss block '
                                 f'of machine "{machine}"')
        text = text[:start] + block + text[end:]

    target = out_path or path
    with open(target, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return target
