# -*- coding: utf-8 -*-
"""
Kundur two-area system (k12p6) — build, initialise and simulate.

Four generators, 11 buses, 230 kV transmission ring.
Generators 1, 2, 4 dispatch at 700 MW via LC (p_c_lc = 0.7778 pu on 900 MVA).
Generator 3 is the AGC reference and angle reference (K_delta = 0.01).
"""

import os
import matplotlib.pyplot as plt
from pydae.bps import BpsBuilder
from pydae.core import Builder, Model


HJSON = os.path.join(os.path.dirname(__file__), 'k12p6.hjson')
NAME  = 'k12p6'


def build():
    grid = BpsBuilder(HJSON)
    grid.uz_jacs = False
    grid.construct(NAME)

    bld = Builder(grid.sys_dict, target='ctypes', sparse=False)
    bld.build()
    print('Build OK')


def run_sim():
    model = Model(NAME)

    # ── initialise ──────────────────────────────────────────────────────
    model.ini({
        'p_c_lc_1': 0.7778,
        'p_c_lc_2': 0.7778,
        'p_c_lc_4': 0.7778,
    }, 'xy_0.json')

    print('\n── Power injections at ini ──')
    for g in ('1', '2', '3', '4'):
        pg = model.get_value(f'p_g_{g}')
        pm = model.get_value(f'p_m_{g}')
        print(f'  gen {g}:  p_g = {pg*900:.1f} MW   p_m = {pm*900:.1f} MW')

    print('\n── Bus voltages at ini ──')
    for b in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'):
        v = model.get_value(f'V_{b}')
        print(f'  bus {b:>2s}:  V = {v:.4f} pu')

    # ── simulate ─────────────────────────────────────────────────────────
    # Steady state
    model.run(1.0, {})

    # Three-phase fault on bus 7 at t=1 s: apply large shunt admittance
    # (approximate short-circuit by removing bus 7 loads and adding shunt)
    # Here we simply step the load at bus 9 by +200 MW to study AGC response
    model.run(30.0, {'P_9': -1967e6})   # +200 MW load step
    model.run(300.0, {})                # let AGC and LC settle
    model.post()

    # ── plots ─────────────────────────────────────────────────────────────
    t = model.Time

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Rotor speeds
    ax = axes[0]
    for g in ('1', '2', '3', '4'):
        ax.plot(t, model.get_values(f'omega_{g}'), label=f'$\\omega_{{{g}}}$')
    ax.axhline(1.0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Speed (pu)')
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Active power injections
    ax = axes[1]
    for g in ('1', '2', '3', '4'):
        ax.plot(t, model.get_values(f'p_g_{g}') * 900,
                label=f'$p_{{g,{g}}}$')
    ax.set_ylabel('$p_g$ (MW on 900 MVA base)')
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # AGC output (dp_lc_3) and LC integrator (x_lc_3)
    ax = axes[2]
    ax.plot(t, model.get_values('xi_agc'),   label='$\\xi_{agc}$')
    ax.plot(t, model.get_values('x_lc_3'),   label='$x_{lc,3}$')
    ax.plot(t, model.get_values('p_c_lc_3'), label='$p_{c,lc,3}$', ls='--')
    ax.set_ylabel('Power (pu, machine base)')
    ax.set_xlabel('Time (s)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Kundur two-area system — 200 MW load step at bus 9', y=1.01)
    fig.tight_layout()
    fig.savefig(f'{NAME}_response.svg', bbox_inches='tight')
    print(f'\nPlot saved to {NAME}_response.svg')

    return model


if __name__ == '__main__':
    build()
    model = run_sim()
