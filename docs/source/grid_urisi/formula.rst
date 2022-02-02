Formulation
===========


Grid
----

.. math::
     \left[ {\begin{array}{cc}
       \mathbf Y_{vv} &  \mathbf Y_{vi} \\
       \mathbf Y_{iv} &  \mathbf Y_{ii} \\
    \end{array} } \right]
    \left[ {\begin{array}{c}
       \mathbf{\underline  v}_k \\
       \mathbf{\underline  v}_u \\
    \end{array} } \right]
    =
    \left[ {\begin{array}{c}
       \mathbf{\underline  i}_u \\
       \mathbf{\underline  i}_k \\
    \end{array} } \right]



.. math::
     \mathbf g_{cplx} = -\mathbf Y_{ii}  \mathbf{\underline  v}_u  - \mathbf Y_{iv} \mathbf{\underline  v}_k




Loads
-----

Phasor voltages and neutral voltage are defined as:

.. math::
    \underline v_p = v_{lp}^r + jv_{lp}^i  \;\;\;\;  p = a,\,b,\,c,\,n
    
    
Phasor currents and neutral currents are defined as:

.. math::
    \underline i_p = i_{lp}^r + ji_{lp}^i  \;\;\;\;  p = a,\,b,\,c,\,n


.. math::
    \mathbf g_l =
    \left[
    \begin{array}{c}
     p_a + \Re \left(\left(\underline v_a - \underline v_n\right) \underline i_a^*\right) \\ 
     p_b + \Re \left(\left(\underline v_b - \underline v_n\right) \underline i_b^*\right) \\
     p_c + \Re \left(\left(\underline v_c - \underline v_n\right) \underline i_c^*\right) \\
     q_a + \Im \left(\left(\underline v_a - \underline v_n\right) \underline i_a^*\right) \\
     q_b + \Im \left(\left(\underline v_b - \underline v_n\right) \underline i_b^*\right) \\
     q_c + \Im \left(\left(\underline v_c - \underline v_n\right) \underline i_c^*\right) \\
     \Re \left(\underline i_a+ \underline i_b+ \underline i_c+ \underline i_n\right) \\
     \Im \left(\underline i_a+ \underline i_b+ \underline i_c+ \underline i_n\right) \\
    \end{array} 
    \right]
    \;\;\;\;\;\;\;
    \mathbf y_l =
    \left[
    \begin{array}{c}
     i_{la}^r \\ 
     i_{lb}^r \\
     i_{lc}^r \\
     i_{la}^i \\
     i_{lb}^i \\
     i_{lc}^i \\
     i_{ln}^r \\
     i_{ln}^i \\
    \end{array} 
    \right]
    
$p_a$ are the inputs.

.. math::
    \mathbf g_l =
    \left[
    \begin{array}{c}
     p_a + \Re \left(\left(\underline v_a - \underline v_n\right) \underline i_a^*\right) \\ 
     p_b + \Re \left(\left(\underline v_b - \underline v_n\right) \underline i_b^*\right) \\
     p_c + \Re \left(\left(\underline v_c - \underline v_n\right) \underline i_c^*\right) \\
     q_a + \Im \left(\left(\underline v_a - \underline v_n\right) \underline i_a^*\right) \\
     q_b + \Im \left(\left(\underline v_b - \underline v_n\right) \underline i_b^*\right) \\
     q_c + \Im \left(\left(\underline v_c - \underline v_n\right) \underline i_c^*\right) \\
     \Re \left(\underline i_a+ \underline i_b+ \underline i_c+ \underline i_n\right) \\
     \Im \left(\underline i_a+ \underline i_b+ \underline i_c+ \underline i_n\right) \\
    \end{array} 
    \right]
    \;\;\;\;\;\;\;
    \mathbf y_l =
    \left[
    \begin{array}{c}
     i_{la}^r \\ 
     i_{lb}^r \\
     i_{lc}^r \\
     i_{la}^i \\
     i_{lb}^i \\
     i_{lc}^i \\
     i_{ln}^r \\
     i_{ln}^i \\
    \end{array} 
    \right]    

