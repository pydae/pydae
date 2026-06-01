r"""
Three-phase Dyn11 transformer (delta primary, wye-neutral secondary, +30° group).

The primitive admittance is built in real form so it lives on either the
SymPy or CasADi backend. The construction follows Rodríguez del Nozal,
Romero-Ramos & Trigo-García, *Accurate Assessment of Decoupled OLTC
Transformers to Optimize the Operation of Low-Voltage Networks*, Energies
12, 2173 (2019).

**Per-winding admittance** (one short-circuit branch + one magnetising
branch per phase, all the same value by default):

$$Y_t = G_t + j B_t = \frac{1}{R_{cc} + j X_{cc}} \cdot \frac{1}{Z_b},
  \qquad Z_b = U_{jn}^2 / S_n$$

$$Y_m = G_m + j B_m = \frac{1}{R_{fe} \cdot Z_b} - j \frac{1}{X_{mu} \cdot Z_b}$$

**Winding-to-terminal incidence** $N \in \mathbb{R}^{6 \times 7}$ encodes
the Dyn11 vector group (columns = $[A, B, C, a, b, c, n]$):

$$N = \begin{bmatrix}
 1 & -1 &  0 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 &  1 &  0 &  0 & -1 \\
 0 &  1 & -1 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 &  0 &  1 &  0 & -1 \\
-1 &  0 &  1 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 &  0 &  0 &  1 & -1
\end{bmatrix}$$

**Per-winding primitive** (with turns ratio $r$ applied symmetrically):

$$Y_p = \begin{bmatrix}
 Y_t + Y_m & -r\, Y_t \\
-r\, Y_t & r^2\, Y_t
\end{bmatrix} \text{ (one 2×2 block per phase, block-diagonal in 6×6)}$$

**Terminal primitive**:

$$Y_\text{primitive} = N^T Y_p N$$

with the LV neutral node closed to ground through $R_g$:
$Y_\text{primitive}[6, 6] \mathrel{{+}{=}} 1/R_g$.

The real and imaginary parts $G_\text{primitive}, B_\text{primitive}$ are
written directly into the grid's branch-admittance matrices at this
transformer's branch slice; the standard nodal assembly in
`UdsBuilder.contruct_grid()` picks them up.

**Per-phase tap ratios** `Ratio_{a,b,c}` enter the primitive as multipliers
on $Y_t$, so the model supports independent per-phase OLTC.

**HJSON snippet**

```hjson
transformers: [
    {bus_j: "MV0", bus_k: "I01", S_n_kVA: 100, U_j_kV: 20, U_k_kV: 0.4,
     R_cc_pu: 0.01, X_cc_pu: 0.04,
     connection: "Dyn11", conductors_j: 3, conductors_k: 4, monitor: true}
]
```
"""

import numpy as np
import sympy as sym


def descriptions():
    return [
        {"type": "Parameter", "tex": "S_n",       "data": "S_n_kVA",   "model": "",                  "default": "",    "units": "kVA",  "description": "Nominal apparent power"},
        {"type": "Parameter", "tex": "U_j",       "data": "U_j_kV",    "model": "",                  "default": "",    "units": "kV",   "description": "Primary (delta) line-to-line voltage"},
        {"type": "Parameter", "tex": "U_k",       "data": "U_k_kV",    "model": "",                  "default": "",    "units": "kV",   "description": "Secondary (wye) line-to-line voltage"},
        {"type": "Parameter", "tex": "R_{cc}",    "data": "R_cc_pu",   "model": "",                  "default": "",    "units": "pu",   "description": "Short-circuit resistance"},
        {"type": "Parameter", "tex": "X_{cc}",    "data": "X_cc_pu",   "model": "",                  "default": "",    "units": "pu",   "description": "Short-circuit reactance"},
        {"type": "Parameter", "tex": "R_{fe}",    "data": "R_fe_pu",   "model": "",                  "default": r"\infty", "units": "pu", "description": "Iron-loss resistance (optional)"},
        {"type": "Parameter", "tex": "X_\\mu",    "data": "X_mu_pu",   "model": "",                  "default": r"\infty", "units": "pu", "description": "Magnetising reactance (optional)"},
        {"type": "Parameter", "tex": "R_g",       "data": "",          "model": "R_g_{bus_j}_{bus_k}", "default": 3.0, "units": r"\Omega", "description": "Neutral-to-ground resistance"},
        {"type": "Parameter", "tex": "G_t",       "data": "",          "model": "G_t_{bus_j}_{bus_k}", "default": "",  "units": "S",    "description": "Per-winding conductance (derived)"},
        {"type": "Parameter", "tex": "B_t",       "data": "",          "model": "B_t_{bus_j}_{bus_k}", "default": "",  "units": "S",    "description": "Per-winding susceptance (derived)"},
        {"type": "Input",     "tex": "r_a",       "data": "",          "model": "Ratio_a_{bus_j}_{bus_k}", "default": 1.0, "units": "-", "description": "Phase-a OLTC tap ratio"},
        {"type": "Input",     "tex": "r_b",       "data": "",          "model": "Ratio_b_{bus_j}_{bus_k}", "default": 1.0, "units": "-", "description": "Phase-b OLTC tap ratio"},
        {"type": "Input",     "tex": "r_c",       "data": "",          "model": "Ratio_c_{bus_j}_{bus_k}", "default": 1.0, "units": "-", "description": "Phase-c OLTC tap ratio"},
        {"type": "Output",    "tex": r"i_{t,j}^{\varphi r/i}", "data": "", "model": "i_t_{bus_j}_{bus_k}_1_{ph}_{r,i,m}", "default": "", "units": "A", "description": "Primary-side branch currents (with `monitor:true`)"},
        {"type": "Output",    "tex": r"i_{t,k}^{\varphi r/i}", "data": "", "model": "i_t_{bus_j}_{bus_k}_2_{ph}_{r,i,m}", "default": "", "units": "A", "description": "Secondary-side branch currents"},
    ]


def add_Dyn11(grid,trafo):
    '''
    Transformer model for type Dyn11 connection as defined in:

    Álvaro Rodríguez del Nozal, Esther Romero-Ramos and Ángel Luis Trigo-García, 
    Accurate Assessment of Decoupled OLTC Transformers to Optimize the Operation of Low-Voltage Networks
    Energies 2019, 12, 2173; doi:10.3390/en12112173

    {"bus_j": "MV0",  "bus_k": "I01",  "S_n_kVA": 100, "U_j_kV":20, "U_k_kV":0.4,
     "R_cc_pu": 0.01, "X_cc_pu":0.04, "R_fe_pu": 1000, "X_mu_pu":100, 
     "connection": "Dyn11",   "conductors_j": 3, "conductors_k": 4,
     "monitor":true}
    
    '''

    # G_ta,B_ta,G_ma,B_ma,Ratio_a = sym.symbols('G_ta,B_ta,G_ma,B_ma,Ratio_a', real = True)
    # G_tb,B_tb,G_mb,B_mb,Ratio_b = sym.symbols('G_tb,B_tb,G_mb,B_mb,Ratio_b', real = True)
    # G_tc,B_tc,G_mc,B_mc,Ratio_c = sym.symbols('G_tc,B_tc,G_mc,B_mc,Ratio_c', real = True)

    bk = grid.backend

    bus_j_name = trafo['bus_j']
    bus_k_name = trafo['bus_k']
    name = f'{bus_j_name}_{bus_k_name}'

    G_t,B_t,G_m,B_m = bk.symbols(f'G_t_{name},B_t_{name},G_m_{name},B_m_{name}')
    Ratio_a,Ratio_b,Ratio_c = bk.symbols(f'Ratio_a_{name},Ratio_b_{name},Ratio_c_{name}')
    R_g,Ratio = bk.symbols(f'R_g_{name},Ratio_{name}')

    # Transformer primitive admittance built in real form (CasADi SX has no
    # complex type). Derived as G+jB = N^T U1 N with the neutral grounded
    # through R_g, where U1 is the block-diagonal per-phase winding admittance
    # and N the winding-to-terminal incidence (see dev_Dyn11). This reproduces
    # the closed-form Y_prim of del Nozal et al. (2019) exactly.
    ra, rb, rc = Ratio_a*Ratio, Ratio_b*Ratio, Ratio_c*Ratio
    Gp1 = bk.zeros(6, 6)
    Bp1 = bk.zeros(6, 6)
    for blk, r in [(0, ra), (2, rb), (4, rc)]:
        Gp1[blk, blk]     = G_t + G_m
        Bp1[blk, blk]     = B_t + B_m
        Gp1[blk, blk+1]   = -r*G_t
        Bp1[blk, blk+1]   = -r*B_t
        Gp1[blk+1, blk]   = -r*G_t
        Bp1[blk+1, blk]   = -r*B_t
        Gp1[blk+1, blk+1] = r**2*G_t
        Bp1[blk+1, blk+1] = r**2*B_t

    # winding -> terminal incidence, columns = [A,B,C,a,b,c,n]
    N = bk.zeros(6, 7)
    N[0, 0] =  1; N[0, 1] = -1
    N[1, 3] =  1; N[1, 6] = -1
    N[2, 1] =  1; N[2, 2] = -1
    N[3, 4] =  1; N[3, 6] = -1
    N[4, 0] = -1; N[4, 2] =  1
    N[5, 5] =  1; N[5, 6] = -1

    G_primitive = N.T @ Gp1 @ N
    B_primitive = N.T @ Bp1 @ N
    # neutral-to-ground conductance closes the LV neutral node
    G_primitive[6, 6] = G_primitive[6, 6] + 1/R_g

    if grid.use_casadi:
        Y_prim = None
    else:
        Y_prim = G_primitive + sym.I*B_primitive

    # default nodes
    nodes_j = [0,1,2]
    nodes_k = [0,1,2,3]  

    # from trafo primitive to system global primitive
    rl = grid.it_branch
    rh = grid.it_branch + trafo['N_branches']
    grid.G_primitive[rl:rh,rl:rh] = G_primitive   
    grid.B_primitive[rl:rh,rl:rh] = B_primitive

    # j side 
    for item in trafo['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
        node_j = f"{trafo['bus_j']}.{item}"
        col = grid.nodes_list.index(node_j)
        row = grid.it_branch

        grid.A[row,col] = 1
        grid.it_branch +=1  

    # k side
    for item in  trafo['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
        node_k = f"{trafo['bus_k']}.{item}"
        col = grid.nodes_list.index(node_k)
        row = grid.it_branch
        grid.A[row,col] = 1
        grid.it_branch +=1  

    S_n = trafo['S_n_kVA']*1000.0
    U_jn = trafo['U_j_kV']*1000.0
    U_kn = trafo['U_k_kV']*1000.0
    Z_cc_pu = trafo['R_cc_pu'] +1j*trafo['X_cc_pu']

    Z_b = U_jn**2/S_n
    Z_cc = Z_cc_pu*Z_b

    Y_cc = 1/Z_cc
    G_t_N = Y_cc.real
    B_t_N = Y_cc.imag

    G_m_N = 0.0
    B_m_N = 0.0

    if 'R_fe_pu' in trafo:
        G_m_N = 1/(trafo['R_fe_pu']*Z_b) 
    if 'X_mu_pu' in trafo:
        B_m_N = 1/(trafo['X_mu_pu']*Z_b) 

    Ratio_N = U_jn/U_kn*np.sqrt(3)
    
    grid.dae['params_dict'].update({str(G_t):G_t_N,str(B_t):B_t_N,str(G_m):G_m_N,str(B_m):B_m_N})
    grid.dae['params_dict'].update({str(R_g):3.0,str(Ratio):Ratio_N})

    grid.dae['u_ini_dict'].update({str(Ratio_a):1,str(Ratio_b):1,str(Ratio_c):1})
    grid.dae['u_run_dict'].update({str(Ratio_a):1,str(Ratio_b):1,str(Ratio_c):1})

    # i_t_1_0_m = sym.Abs(i_t_MV0_I01_1_0_r + I*i_t_MV0_I01_1_0_i)
    # i_t_1_1_m = sym.Abs(i_t_MV0_I01_1_1_r + I*i_t_MV0_I01_1_1_i)
    # i_t_1_2_m = sym.Abs(i_t_MV0_I01_1_2_r + I*i_t_MV0_I01_1_2_i)
    # i_t_2_0_m = sym.Abs(i_t_MV0_I01_2_0_r + I*i_t_MV0_I01_2_0_i)
    # i_t_2_1_m = sym.Abs(i_t_MV0_I01_2_1_r + I*i_t_MV0_I01_2_1_i)
    # i_t_2_2_m = sym.Abs(i_t_MV0_I01_2_2_r + I*i_t_MV0_I01_2_1_r)
    # i_t_2_1_m = sym.Abs(i_t_MV0_I01_2_1_i + I*i_t_MV0_I01_2_2_r)
    # i_t_2_1_m = sym.Abs(i_t_MV0_I01_2_1_i + I*i_t_MV0_I01_2_2_r)
    # i_t_2_2_m = sym.Abs(i_t_MV0_I01_2_2_i + I*i_t_MV0_I01_2_2_r)
    # i_t_2_2_m = sym.Abs(i_t_MV0_I01_2_2_i + I*i_t_MV0_I01_2_2_i)
    # i_t_2_3_m = sym.Abs(i_t_MV0_I01_2_3_r + I*i_t_MV0_I01_2_3_i)

    return Y_prim,nodes_j,nodes_k




def dev_Dyn11():
    '''
    I1 = U1_to_I1 * U1
    U1 = N*UA
    IA = N.T * I1 + UnRG

    IA = Yp*UA

    IA =  (N.T * U1_to_I1 * N)*UA + UnRG 

    '''

    G_ta,B_ta,G_ma,B_ma,Ratio_a = sym.symbols('G_ta,B_ta,G_ma,B_ma,Ratio_a', real = True)
    G_tb,B_tb,G_mb,B_mb,Ratio_b = sym.symbols('G_tb,B_tb,G_mb,B_mb,Ratio_b', real = True)
    G_tc,B_tc,G_mc,B_mc,Ratio_c = sym.symbols('G_tc,B_tc,G_mc,B_mc,Ratio_c', real = True)

    G_ta,B_ta,G_ma,B_ma,Ratio_a = sym.symbols('G_t,B_t,G_m,B_m,Ratio_a', real = True)
    G_tb,B_tb,G_mb,B_mb,Ratio_b = sym.symbols('G_t,B_t,G_m,B_m,Ratio_b', real = True)
    G_tc,B_tc,G_mc,B_mc,Ratio_c = sym.symbols('G_t,B_t,G_m,B_m,Ratio_c', real = True)

    R_g = sym.symbols('R_g', real = True)


    Y_ta = G_ta + sym.I*B_ta
    Y_ma = G_ma + sym.I*B_ma
    Y_tb = G_tb + sym.I*B_tb
    Y_mb = G_mb + sym.I*B_mb   
    Y_tc = G_tc + sym.I*B_tc
    Y_mc = G_mc + sym.I*B_mc   


    U1_to_I1 = sym.Matrix([
        [  Y_ta + Y_ma,     -Ratio_a*Y_ta,            0,                 0,            0,                0 ],
        [-Ratio_a*Y_ta,   Ratio_a**2*Y_ta,            0,                 0,            0,                0 ],
        [            0,                 0,  Y_tb + Y_mb,     -Ratio_b*Y_tb,            0,                0 ],
        [            0,                 0,-Ratio_b*Y_tb,   Ratio_b**2*Y_tb,            0,                0 ],
        [            0,                 0,            0,                 0,  Y_tc + Y_mc,     -Ratio_c*Y_tc],
        [            0,                 0,            0,                 0,-Ratio_c*Y_tc,   Ratio_c**2*Y_tc]
    ])

    N = sym.Matrix([# A   B   C   a   b   c   n
                    [ 1, -1,  0,  0,  0,  0,  0], # 1 
                    [ 0,  0,  0,  1,  0,  0, -1], # 2
                    [ 0,  1, -1,  0,  0,  0,  0], # 3
                    [ 0,  0,  0,  0,  1,  0, -1], # 4
                    [-1,  0,  1,  0,  0,  0,  0], # 5
                    [ 0,  0,  0,  0,  0,  1, -1]  # 6
    ])
    
    Y_prim = N.T @ U1_to_I1 @ N
    Y_prim[-1,-1] += 1/R_g
    Y_prim = sym.simplify(Y_prim)

    sym.pprint(Y_prim)    


def test_Dyn11_build():
    from pydae.uds import UdsBuilder


    grid = UdsBuilder('Dyn11.json')
    grid.uz_jacs = False
    grid.build('cigre_eu_lv_ind')

def test_Dyn11_ini():
    from pydae.uds.utils import report_v
    import cigre_eu_lv_ind
    model = cigre_eu_lv_ind.model()
    model.ini({'Ratio_a_MV0_I01':1.0,'Ratio_b_MV0_I01':1.0,'Ratio_c_MV0_I01':1.0,
               'R_g_MV0_I01':1000},'xy_0.json')
    model.report_y()
    report_v(model,'Dyn11.json')
    model.report_z()


if __name__ == "__main__":

    test_Dyn11_build()
    test_Dyn11_ini()

    #print(dev_Dyn11())