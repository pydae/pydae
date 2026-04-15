# =============================================================================
# Testing Block
# =============================================================================
def test_build():
    from pydae.bmapu.bmapu_builder import bmapu
    from pydae.builder.core import Builder
    import pytest

    grid = bmapu('config_planta_scib.json')
    grid.checker()
    grid.uz_jacs = False
    grid.construct('scib')
    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()
 
def test_run():
    import matplotlib.pyplot as plt
    from pydae.builder.model_class import Model
    
    model = Model('scib')

    # Initialization and Fault Simulation
    model.ini({'p_m_1': 0.5}, 'xy_0.json')

    model.run(1.0, {})
    model.run(10.0, {'p_m_1': 1.0, 'D_1':20.0})    # Fault cleared

    model.post()

    fig, axes = plt.subplots()
    axes.plot(model.Time, model.get_values('p_s_1'), label='p_s_1')
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('omega_1')
    axes.legend()
    fig.savefig('scib.svg')
    print("Test completed. Plot saved as 'scib.svg'.")

if __name__ == '__main__':
    test_build()
    test_run()