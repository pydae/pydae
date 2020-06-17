Small signal analysis and control
=================================

.. math::

	\mathbf {\dot x}  & \sf =  \mathbf {f (x,y,u) } \\
	\mathbf 0 & \sf =  \mathbf {g (x,y,u) }  \\
    \mathbf z & \sf =  \mathbf {h (x,y,u) }  \\

.. math::

    \mathbf {F_x} = \left[
    \begin{array}{cccc} 
    \sf \frac{\partial f_1}{\partial x_1} & \sf \frac{\partial f_1}{\partial x_2} &  \cdots & \sf  \frac{\partial f_1}{\partial x_{N_x}} \\ 
    \sf \frac{\partial f_2}{\partial x_1} & \sf \frac{\partial f_2}{\partial x_2} &  \cdots &  \sf \frac{\partial f_2}{\partial x_{N_x}} \\ 
    \vdots & \vdots &  \cdots &  \vdots \\ 
    \sf \frac{\partial f_{N_x}}{\partial x_1} & \sf \frac{\partial f_{N_x}}{\partial x_2} &  \cdots & \sf  \frac{\partial f_{N_x}}{\partial x_{N_x}} 
    \end{array} \right] \;\;\;
    \mathbf {F_y} = \left[
    \begin{array}{cccc} 
    \sf \frac{\partial f_1}{\partial y_1} & \sf \frac{\partial f_1}{\partial y_2} &  \cdots & \sf  \frac{\partial f_1}{\partial y_{N_y}} \\ 
    \sf \frac{\partial f_2}{\partial y_1} & \sf \frac{\partial f_2}{\partial y_2} &  \cdots &  \sf \frac{\partial f_2}{\partial y_{N_y}} \\ 
    \vdots & \vdots &  \cdots &  \vdots \\ 
    \sf \frac{\partial f_{N_x}}{\partial y_1} & \sf \frac{\partial f_{N_x}}{\partial y_2} &  \cdots & \sf  \frac{\partial f_{N_x}}{\partial y_{N_y}} 
    \end{array} \right]

.. math::

    \sf  \nonumber
	\mathbf {G_x} = \left[
	\begin{array}{cccc} 
	\sf \frac{\partial g_1}{\partial x_1} & \sf \frac{\partial g_1}{\partial x_2} &  \cdots & \sf  \frac{\partial g_1}{\partial x_{N_x}} \\ 
	\sf \frac{\partial g_2}{\partial x_1} & \sf \frac{\partial g_2}{\partial x_2} &  \cdots &  \sf \frac{\partial g_2}{\partial x_{N_x}} \\ 
	\vdots & \vdots &  \cdots &  \vdots \\ 
	\sf \frac{\partial g_{N_y}}{\partial x_1} & \sf \frac{\partial g_{N_y}}{\partial x_2} &  \cdots & \sf  \frac{\partial g_{N_y}}{\partial x_{N_x}} 
	\end{array} \right] \;\;\;
	\mathbf {G_y} = \left[
	\begin{array}{cccc} 
	\sf \frac{\partial g_1}{\partial y_1} & \sf \frac{\partial g_1}{\partial y_2} &  \cdots & \sf  \frac{\partial g_1}{\partial y_{N_y}} \\ 
	\sf \frac{\partial g_2}{\partial y_1} & \sf \frac{\partial g_2}{\partial y_2} &  \cdots &  \sf \frac{\partial g_2}{\partial y_{N_y}} \\ 
	\vdots & \vdots &  \cdots &  \vdots \\ 
	\sf \frac{\partial g_{N_y}}{\partial y_1} & \sf \frac{\partial g_{N_y}}{\partial y_2} &  \cdots & \sf  \frac{\partial g_{N_y}}{\partial y_{N_y}} 
	\end{array} \right]



Linealized DAE system
---------------------

.. math::

	\mathbf {\Delta \dot x }&= \mathbf {A \Delta x + B \Delta u } \\
	\mathbf {\Delta z      }&= \mathbf {C \Delta x + D \Delta u}  

.. math::

	\mathbf A &= \mathbf {F_x - F_y G_y^{-1} G_x} \\
	\mathbf B &= \mathbf {F_u - F_y G_y^{-1} G_u} \\
	\mathbf C &= \mathbf {H_x - H_y G_y^{-1} G_x} \\
	\mathbf D &= \mathbf {H_u - H_y G_y^{-1} G_u}

