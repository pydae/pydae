User guide
==========

pydae consist of two main parts:

* the system builder
* the analysis and simulation collection of tools

Basically, the system builder converts a symbolic defined DAE system in a module that contains the main class and part of the tools collection. 

The analysis and simulation tools are functions that allows the user to solve the DAE system in different ways using the build module. 

Two DAE system approach
-----------------------

The DAE problem can be formulated as follows:

.. math::

	\mathbf {\dot x}  & =  \sf  \mathbf {f (x,y,u) } \\
	\mathbf 0 & =  \sf \mathbf {g (x,y,u) }  


where:

* :math:`\sf \mathbf x` is a vector with the dynamic states
* :math:`\sf \mathbf y` is a vector with the algebraic states
* :math:`\sf \mathbf u` is a vector of known inputs
* :math:`\sf \mathbf x` is a vector with the differential equations
* :math:`\sf \mathbf y` is a vector with the algebraic equations


In pydae it is considered that the DAE time domain solution should start from a steady state point. This can be made by making:

.. math::

	\mathbf 0 & \sf =  \mathbf {f (x,y,u) } \\
	\mathbf 0 & \sf =  \mathbf {g (x,y,u) }  


and solving for :math:`\sf \mathbf x` and :math:`\sf \mathbf y` considering :math:`\sf \mathbf u` known.
However there are cases where the time domain solution requires inputs thar are not known. As an example, in power systems it is common to define some inputs in the so called power flow problem (i.e. the active and reactive power output from a synchronous machine) while in the time solution other inputs have to be considered (i.e. mechanical power and excitation voltage) that depends on  the previous inputs. This gives place to two DAE prblems un pydae that gives:

* Backward solution 
* Foreward solution

In pydae the only difference between both solutions is related with the vectors :math:`\sf \mathbf u` and :math:`\sf \mathbf y`. This gives the following  DAE systems:

Backward solution DAE
'''''''''''''''''''''
.. math::

	\mathbf {\dot x}  & \sf =  \mathbf {f (x,y^{ini},u^{ini}) } \\
	\mathbf 0 & \sf =  \mathbf {g (x,y^{ini},u^{ini}) }  


Forward solution DAE
''''''''''''''''''''
.. math::

	\mathbf {\dot x}  & \sf =  \mathbf {f (x,y^{run},u^{run}) } \\
	\mathbf 0 & \sf =  \mathbf {g (x,y^{run},u^{run}) }  

poner figura de esto

System builder
--------------

In the system building step (etapa) the full DAE system have to be provided to the pydae builder. For this porpose the following dictionary have to be crated:

.. code::
    
    sys_dict = {'name':sys_name,
                'params_dict':params_dict,
                'f_list':f_list,
                'g_list':g_list,
                'x_list':x_list,
                'y_ini_list':y_ini_list,
                'y_run_list':y_run_list,
                'u_run_dict':u_run_dict,
                'u_ini_dict':u_ini_dict,
                'h_dict':h_dict}

where:

* :name:`name` is a vector with the dynamic states


In case the backward solution is not required the ini can be ignored and the system that have to be provided can be simplified as follows:

.. code-block:: python
    
    sys_dict = {'name':sys_name,
                'params_dict':params_dict,
                'f_list':f_list,
                'g_list':g_list,
                'x_list':x_list,
                'y_list':y_list,
                'u_dict':u_dict,
                'h_dict':h_dict}


