# -*- coding: utf-8 -*-
"""
6-bus 3-generator system benchmark.

Run via:
    cd /Users/jmmauricio/workspace/benchmarks/sys8bus4gen
    python test_sys8bus4gen.py          # full build + ini + run + SSA
    uv run pytest test_sys8bus4gen.py   # pytest mode
"""
from pathlib import Path
import numpy as np
from pydae.bps.utils.reporter import report_buses, report_gens, report_lines

def ini():
    from pydae import ssa
    from pydae.bps import BpsBuilder
    from pydae.core.builder import CasadiBuilder
    from pydae.core.model import CasadiModel

    grid = BpsBuilder('copv.json', use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct('copv')

    bld = CasadiBuilder(grid.sys_dict)
    bld.build()


    model = CasadiModel(bld)
    model.ini({}, xy_0='xy_0.json')
    
    report_buses(model, 'copv.json')
    report_lines(model, 'copv.json')
    # # Small-signal analysis
    # model.A_eval()
    # sstudy = ssa.damp(model.A, model=model, sort='damp', part_threshold=0.5)

    # fig = ssa.plot_eig(sstudy['eigvalues'], x_min=-2.5, x_max=0, y_min=0, y_max=8,
    #                    fig='', mark='o', color='blue', label='base case')
    # fig.savefig('sys8bus4gen_eig.png')

    # ssa.eig(model)

    # eigenvalues, right_eigenvectors = np.linalg.eig(model.A)
    # left_eigenvectors = np.linalg.inv(right_eigenvectors)
    # participation = right_eigenvectors * left_eigenvectors.T

    # print(eigenvalues)

    # # Kundur normalisation: |p_ki| / Σ_k|p_ki| → each column sums to 1
    # participation = np.abs(participation)
    # participation = participation / participation.sum(axis=0)

    # out_folder = '.'

    # def build_eigvec_table(data, fmt, x_list):
    #     N_rows, N_cols = data.shape
    #     header = ' & '
    #     for icol in range(N_cols):
    #         n = icol + 1
    #         subscript = f'{{{n}}}' if n >= 10 else f'{n}'
    #         header += r'$\lambda_' + subscript + r' $ & '
    #     lines = [r'\begin{tabular}{l|' + 'r'*N_cols + '}',
    #              header[:-2] + r'\\ \hline']
    #     for irow in range(N_rows):
    #         row = r'\verb|' + x_list[irow] + '| & '
    #         for icol in range(N_cols):
    #             row += fmt.format(data[irow, icol]) + ' & '
    #         lines.append(row[:-2] + r'\\')
    #     lines.append(r'\end{tabular}')
    #     return '\n'.join(lines)

    # # angle table (degrees)
    # angles = np.angle(right_eigenvectors, deg=True)
    # angles[angles < 0] += 360
    # table_deg = build_eigvec_table(angles, '{:4.0f}', model.x_list)
    # with open(out_folder + 'rvectors_deg.tex', 'w') as f:
    #     f.write(table_deg)

    # # module table
    # mods = np.abs(right_eigenvectors)
    # table_mod = build_eigvec_table(mods, '{:5.3f}', model.x_list)
    # with open(out_folder + 'rvectors_mod.tex', 'w') as f:
    #     f.write(table_mod)

    # # participation matrix table — Kundur-normalised (column sums = 1)
    # table_part = build_eigvec_table(participation, '{:4.2f}', model.x_list)
    # with open(out_folder + 'participation.tex', 'w') as f:
    #     f.write(table_part)

    # # eigenvalues table (same sort order as all other tables)
    # lines = [r'\begin{tabular}{r|r}',
    #          r'Modo & $\lambda$ \\ \hline']
    # for i, ev in enumerate(eigenvalues):
    #     sign = '+' if ev.imag >= 0 else '-'
    #     lines.append(f'{i+1} & ${ev.real:.3f} {sign} {abs(ev.imag):.3f}j$ \\\\')
    # lines.append(r'\end{tabular}')
    # table_eig = '\n'.join(lines)
    # with open(out_folder + 'eigenvalues.tex', 'w') as f:
    #     f.write(table_eig)






def test_run():
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pydae.core import Model

    model = Model('sys6bus3gen.hjson')
    model.ini({}, )

    # Steady state for 1 s
    model.run(1.0, {})

    # Load step at bus 1: add 50 MW extra load
    S_base = 100e6
    P_step = -50e6 / S_base  # -0.5 pu (additional load)
    model.run(10.0, {'P_1': -350e6 / S_base + P_step})

    # Clear the step — run long enough for AGC integral to unwind
    model.run(60.0, {'P_1': -350e6 / S_base})

    model.post()

    t       = model.Time
    omega_G1 = model.get_values('omega_G1')
    omega_G2 = model.get_values('omega_G2')
    omega_G3 = model.get_values('omega_G3')
    omega_G4 = model.get_values('omega_G4')

    # Frequency must return to 1.0 pu after disturbance (allow AGC settling)
    assert abs(omega_G1[-1] - 1.0) < 5e-3, \
        f'omega_G1 did not recover: {omega_G1[-1]:.6f}'

    # --- Plotly ---
    p_g_G1 = model.get_values('p_g_G1')
    p_g_G2 = model.get_values('p_g_G2')
    p_g_G3 = model.get_values('p_g_G3')
    p_g_G4 = model.get_values('p_g_G4')

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=('Rotor speed (pu)', 'Active power injection (pu machine base)'),
    )

    for label, omega in (('G1', omega_G1), ('G2', omega_G2),
                         ('G3', omega_G3), ('G4', omega_G4)):
        fig.add_trace(go.Scatter(x=t, y=omega, name=f'ω_{label}'), row=1, col=1)

    for label, pg in (('G1', p_g_G1), ('G2', p_g_G2),
                      ('G3', p_g_G3), ('G4', p_g_G4)):
        fig.add_trace(go.Scatter(x=t, y=pg, name=f'p_g_{label}'), row=2, col=1)

    fig.update_yaxes(title_text='ω (pu)',   row=1, col=1)
    fig.update_yaxes(title_text='p_g (pu)', row=2, col=1)
    fig.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig.update_layout(title='sys8bus4gen — 50 MW load step at bus 1 (t=1–10 s)', height=600)

    out = str(Path(__file__).parent / 'sys8bus4gen_run.html')
    fig.write_html(out)
    print(f'Saved {out}')


if __name__ == '__main__':
    #build()

    from pydae.bps.utils.visualizer import PowerSystemVisualizer


    visualizer = PowerSystemVisualizer("copv.json")
    visualizer.draw_diagram("copv.png")
    ini()
    # test_run()
    # print('All sys8bus4gen tests passed.')
