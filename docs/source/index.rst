pydae - DAE equations in Python
===============================

.. image:: _static/logo_text.png
   :width: 675px
   :align: center

**pydae** is an open source (MIT) collection of Python functions useful
for analysis and simulation of Diferential Algebraic Equations (DAE) systems.

View `source code`_ of pydae!

.. _`source code`: https://github.com/pydae/pydae



Some of its features are:

* State space symbolic definition of the DAE system using sympy `sympy`_.
* Jacobians of the DAE system obtention.
* DAE solver for time domain simulation.
* Small signal analysis tools.
* Numba implemented functions.
* A collection of DAE models for power systems.


Contents
--------

.. math::

	\begin{equation}
	\begin{split} \sf \nonumber
	\mathbf {\dot x}  & \sf =  \mathbf {f (x,y^{ini},u^{ini}) } \\
	\mathbf 0 & \sf =  \mathbf {g (x,y^{ini},u^{ini}) }  
	\end{split}
	\end{equation}

.. math::

	\begin{equation}
	\begin{split} \sf \nonumber
	\mathbf {\dot x}  & \sf =  \mathbf {f (x,y^{run},u^{run}) } \\
	\mathbf 0 & \sf =  \mathbf {g (x,y^{run},u^{run}) }  
	\end{split}
	\end{equation}



.. toctree::
    :maxdepth: 2

    about
    getting_started
    user_guide   
