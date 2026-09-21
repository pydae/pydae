"""
pydae.utils.pod_wph — lead-lag phase compensation design for a POD / PSS chain.

Residue-based damper design gives, for every operating point, the phase the damper must add at
the target mode for the eigenvalue to move left (pure damping)::

    phi_req = 180 deg - angle(R)          R = residue of  u_damper -> y_measurement

`pod_wph` (Washout-Phase design) fits the lead-lag stages of a washout + lead-lag damper so its
full transfer function

    H(s) = K_stab * ---------  * prod_j  ---------  * prod_i  ----------------
                    T_lpf*s+1            T_woj*s+1            T_lag_i*s + 1

                     1                    T_woj*s               T_lead_i*s + 1

supplies `phi_req` at every target frequency, in a weighted least-squares sense. The washouts and
the input low-pass are taken as given (they are usually fixed by the measurement chain); only the
lead-lag stages are designed. The structure matches the pydae `pod_2wo_3ll` device, whose
lead-lag parameters are `T_1..T_6` = (T_lead_1, T_lag_1, T_lead_2, T_lag_2, T_lead_3, T_lag_3).

Each stage i is parameterised by its centre frequency `wc_i` and its maximum phase lead `phim_i`::

    alpha_i = (1 + sin(phim_i)) / (1 - sin(phim_i))
    T_lead_i = sqrt(alpha_i) / wc_i        T_lag_i = 1 / (wc_i * sqrt(alpha_i))

which keeps the stage well conditioned (`alpha_i` bounded by `alpha_max`) and makes the bound on
the fastest time constant a simple linear constraint.

The NLP is solved with CasADi/IPOPT; casadi is imported lazily so this module can be imported
without it.

Typical use (design phase compensation for six operating points)::

    from pydae.utils import pod_wph

    r = pod_wph(omega_target, phi_req_deg, T_lpf=0.02, T_wo=(5.0, 5.0))
    print(r['T_lead'], r['T_lag'], r['rms_deg'])

Sizing the gain afterwards is case specific: with the phase compensated the eigenvalue shift is
(nearly) real, `Dsigma = |R| * K_stab * |H1|`, and `r['gain']` returns `|H1|` (the chain magnitude
with K_stab = 1) at each target frequency.
"""

import numpy as np

__all__ = ["pod_wph", "pod_chain_tf", "pod_chain_phase_deg"]


def pod_chain_tf(s, T_lead, T_lag, T_lpf, T_wo, K_stab=1.0):
    """Complex transfer function of the full damper chain at ``s`` (scalar or array).

    Parameters
    ----------
    s : complex or ndarray
        Laplace variable, typically ``1j*omega``.
    T_lead, T_lag : sequence of float
        Lead-lag time constants, one pair per stage.
    T_lpf : float
        Input low-pass time constant (``0`` disables it).
    T_wo : sequence of float
        Washout time constants (one entry per washout block).
    K_stab : float
        Overall gain (does not affect the phase).
    """
    H = K_stab * np.ones_like(np.asarray(s, dtype=complex))
    if T_lpf:
        H = H / (T_lpf * s + 1.0)
    for T in T_wo:
        H = H * (T * s) / (T * s + 1.0)
    for T_l, T_g in zip(T_lead, T_lag):
        H = H * (T_l * s + 1.0) / (T_g * s + 1.0)
    return H


def pod_chain_phase_deg(omega, T_lead, T_lag, T_lpf, T_wo):
    """Phase [deg] added by the damper chain at ``omega`` [rad/s], unwrapped by construction.

    Summing the per-block phases (instead of taking ``angle`` of the product) keeps the result
    continuous past +-180 deg, which is what a design target must be compared against.
    """
    omega = np.asarray(omega, dtype=float)
    phase = np.zeros_like(omega)
    if T_lpf:
        phase -= np.arctan(omega * T_lpf)
    for T in T_wo:
        phase += np.pi / 2.0 - np.arctan(omega * T)
    for T_l, T_g in zip(T_lead, T_lag):
        phase += np.arctan(omega * T_l) - np.arctan(omega * T_g)
    return np.degrees(phase)


def pod_wph(omega_target, phi_req_deg, *, T_lpf=0.0, T_wo=(), n_stages=3, weights=None,
            alpha_max=12.0, T_min=0.015, T_max=None, wc_min=None, wc_max=None, x0=None,
            gain_weight=0.0, ipopt_tol=1e-10, verbose=True):
    """Fit the lead-lag stages of a POD/PSS so the chain supplies ``phi_req_deg``.

    Parameters
    ----------
    omega_target : array_like
        Target frequencies [rad/s], one per operating point / mode.
    phi_req_deg : array_like
        Phase required from the damper at each target [deg], normally ``180 - angle(residue)``.
    T_lpf : float
        Input low-pass time constant of the damper (fixed).
    T_wo : sequence of float
        Washout time constants (fixed), e.g. ``(5.0, 5.0)`` for ``pod_2wo_3ll``.
    n_stages : int
        Number of lead-lag stages to design (3 for ``pod_2wo_3ll``).
    weights : array_like, optional
        Per-target weights of the squared phase error. Defaults to 1.
    alpha_max : float
        Maximum lead ratio per stage; caps the phase per stage at
        ``arcsin((alpha_max-1)/(alpha_max+1))`` (57.8 deg for 12).
    T_min : float
        Lower bound on every designed time constant [s] (keeps the fastest lag realisable).
    T_max : float, optional
        Upper bound on every designed time constant [s]. Unbounded by default. Use it to keep the
        compensator *faster than the mode it damps*: a chain whose slowest pole approaches the
        target eigenvalue under-delivers against its own first-order estimate
        ``Dlambda = R*H(lambda)``, and a phase-only objective cannot see that -- it will happily
        buy phase with slow stages. ``T_max = 1/(k*abs(lam.real))`` with ``k`` around 3 puts the
        slowest pole a factor ``k`` to the left of the mode.
    wc_min, wc_max : float, optional
        Bounds on each stage centre frequency [rad/s]. Default: 0.1x the lowest target and
        10x the highest.
    x0 : array_like, optional
        Initial guess ``[wc..., phim...]``. Default spreads the stages over the target band.
    gain_weight : float
        Weight of a penalty on ``ln|H_ll(jw)|**2``, the squared log-magnitude of the *lead-lag*
        product at each target (the fixed ``T_lpf``/``T_wo`` terms are excluded -- they cannot be
        traded). 0.0 (default) reproduces the pure phase fit.

        The phase objective sees only the *sum* of the stage phases, so whenever the required
        phase is reachable the solution is massively degenerate and the solver returns an
        arbitrary member of that set -- often one that spends every stage and throws the loop gain
        away. A lag target is the worst case: three stages each at ``alpha ~ 0.26`` deliver the
        phase while attenuating ~50x, where one lag stage and two pass-throughs would give the
        same phase at ~4x. This term breaks the tie toward unity chain gain, so ``K_stab`` is not
        spent undoing the compensator's own attenuation. Start around 1.0 and raise it until the
        phase ``rms_deg`` starts to suffer.
    verbose : bool
        Print the design and the phase validation table.

    Returns
    -------
    dict with keys
        ``T_lead``, ``T_lag`` : ndarray, the designed time constants (``T_1..T_2n`` interleaved
        in ``T_pod``); ``wc``, ``phim``, ``alpha`` : the stage parameters; ``phase_deg`` : phase
        obtained at each target; ``error_deg`` : obtained - required; ``rms_deg``, ``max_abs_deg``;
        ``gain`` : chain magnitude at each target with ``K_stab = 1``; ``T_pod`` : the interleaved
        ``[T_lead_1, T_lag_1, ...]`` list, i.e. ``T_1..T_6`` of ``pod_2wo_3ll``.
    """
    import casadi as ca

    omega_target = np.atleast_1d(np.asarray(omega_target, dtype=float))
    phi_req_deg = np.atleast_1d(np.asarray(phi_req_deg, dtype=float))
    if omega_target.size != phi_req_deg.size:
        raise ValueError("omega_target and phi_req_deg must have the same length")

    n_t = omega_target.size
    phi_req_rad = np.deg2rad(phi_req_deg)
    weights = np.ones(n_t) if weights is None else np.asarray(weights, dtype=float)
    wc_min = 0.1 * omega_target.min() if wc_min is None else wc_min
    wc_max = 10.0 * omega_target.max() if wc_max is None else wc_max
    phi_max_rad = np.arcsin((alpha_max - 1.0) / (alpha_max + 1.0))

    # decision variables: stage centre frequency and stage maximum phase
    wc = ca.SX.sym('wc', n_stages)
    phim = ca.SX.sym('phim', n_stages)

    alpha = (1 + ca.sin(phim)) / (1 - ca.sin(phim))
    T_lead = ca.sqrt(alpha) / wc
    T_lag = 1 / (wc * ca.sqrt(alpha))

    J = 0
    for k in range(n_t):
        w = omega_target[k]
        phase = -ca.atan(w * T_lpf) if T_lpf else 0
        for T in T_wo:
            phase += ca.pi / 2.0 - ca.atan(w * T)
        for i in range(n_stages):
            phase += ca.atan(w * T_lead[i]) - ca.atan(w * T_lag[i])
        J += weights[k] * (phase - phi_req_rad[k])**2

        if gain_weight:
            # ln|H_ll| = sum_i 0.5*(ln(1+(w T_lead_i)^2) - ln(1+(w T_lag_i)^2)); pull it
            # toward 0 (unity gain) to pick a cheap member of the degenerate phase solution.
            log_mag = 0
            for i in range(n_stages):
                log_mag += 0.5 * (ca.log(1 + (w * T_lead[i])**2)
                                  - ca.log(1 + (w * T_lag[i])**2))
            J += gain_weight * weights[k] * log_mag**2

    nlp = {'x': ca.vertcat(wc, phim), 'f': J, 'g': ca.vertcat(T_lead, T_lag)}
    solver = ca.nlpsol('solver', 'ipopt', nlp,
                       {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': ipopt_tol})

    if x0 is None:
        x0 = np.concatenate([np.geomspace(max(omega_target.min(), wc_min),
                                          min(omega_target.max(), wc_max), n_stages),
                             np.full(n_stages, np.deg2rad(np.mean(phi_req_deg) / n_stages))])

    sol = solver(x0=x0,
                 lbx=[wc_min] * n_stages + [-phi_max_rad] * n_stages,
                 ubx=[wc_max] * n_stages + [phi_max_rad] * n_stages,
                 lbg=[T_min] * (2 * n_stages),
                 ubg=[ca.inf if T_max is None else T_max] * (2 * n_stages))

    x_opt = np.array(sol['x']).flatten()
    wc_opt, phim_opt = x_opt[:n_stages], x_opt[n_stages:]
    alpha_opt = (1 + np.sin(phim_opt)) / (1 - np.sin(phim_opt))
    T_lead_opt = np.sqrt(alpha_opt) / wc_opt
    T_lag_opt = 1 / (wc_opt * np.sqrt(alpha_opt))

    phase_deg = pod_chain_phase_deg(omega_target, T_lead_opt, T_lag_opt, T_lpf, T_wo)
    error_deg = phase_deg - phi_req_deg
    gain = np.abs(pod_chain_tf(1j * omega_target, T_lead_opt, T_lag_opt, T_lpf, T_wo))

    T_pod = [t for pair in zip(T_lead_opt, T_lag_opt) for t in pair]

    result = {'T_lead': T_lead_opt, 'T_lag': T_lag_opt, 'T_pod': T_pod,
              'wc': wc_opt, 'phim': phim_opt, 'alpha': alpha_opt,
              'phase_deg': phase_deg, 'error_deg': error_deg,
              'rms_deg': float(np.sqrt(np.mean(error_deg**2))),
              'max_abs_deg': float(np.max(np.abs(error_deg))),
              'gain': gain}

    if verbose:
        print("\n" + "=" * 64)
        print(f" POD LEAD-LAG DESIGN  ({n_stages} stages, {n_t} targets, T_min = {T_min} s)")
        print("=" * 64)
        for i in range(n_stages):
            print(f"Stage {i+1}: T_lead = {T_lead_opt[i]:8.4f} s, T_lag = {T_lag_opt[i]:8.4f} s "
                  f"(alpha = {max(alpha_opt[i], 1/alpha_opt[i]):5.2f}, wc = {wc_opt[i]:6.3f} rad/s)")
        print("\nT_1..T_%d = %s" % (2 * n_stages, ", ".join(f"{t:.4f}" for t in T_pod)))

        print("\n  w[rad/s]  phi_req  phi_got     err   |H|/K_stab")
        for k in range(n_t):
            print(f"  {omega_target[k]:8.3f} {phi_req_deg[k]:8.2f} {phase_deg[k]:8.2f} "
                  f"{error_deg[k]:+7.2f}   {gain[k]:8.2f}")
        print(f"\nmax |error| = {result['max_abs_deg']:.2f} deg, "
              f"rms = {result['rms_deg']:.2f} deg")

    return result


if __name__ == '__main__':
    # original demo: three modes, three stages, plotted with plotly
    omega_demo = np.array([0.1, 0.3, 0.7]) * 2 * np.pi
    phi_demo = np.array([45.0, 60.0, 75.0])

    r = pod_wph(omega_demo, phi_demo, T_lpf=0.2, T_wo=(10.0, 10.0), n_stages=3,
                wc_min=0.05 * 2 * np.pi, wc_max=1.5 * 2 * np.pi)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    w_plot = np.logspace(-1, 2, 1000)
    H = pod_chain_tf(1j * w_plot, r['T_lead'], r['T_lag'], 0.2, (10.0, 10.0), K_stab=10.0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Magnitude (dB)", "Phase (Degrees)"))
    fig.add_trace(go.Scatter(x=w_plot, y=20 * np.log10(np.abs(H)), mode='lines',
                             name='Total POD Magnitude', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=w_plot, y=np.angle(H, deg=True), mode='lines',
                             name='Total POD Phase', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=omega_demo, y=(phi_demo + 180) % 360 - 180, mode='markers',
                             name='Optimization Targets',
                             marker=dict(color='red', size=10, symbol='x',
                                         line=dict(width=2, color='DarkSlateGrey'))), row=2, col=1)
    fig.update_layout(title="Bode Plot of Full POD Evaluated via s = jw", height=700,
                      showlegend=True, hovermode="x unified", template="plotly_white")
    fig.update_xaxes(type="log", title_text="Frequency (rad/s)", row=2, col=1)
    fig.update_xaxes(type="log", showticklabels=True, row=1, col=1)
    fig.show()
