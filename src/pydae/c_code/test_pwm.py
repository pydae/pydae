import numpy as np
import pwm_cffi
import matplotlib.pyplot as plt

Dt = 100e-6
freq = 2e3

times = np.arange(0,0.02,Dt)
outputs = {'t':[],'eta':[],'s':[]}
for it,t in enumerate(times):
    eta = 1*np.sin(2*np.pi*50*t)
    s = pwm_cffi.lib.PWM(t,eta,freq,Dt)

    #print(f't = {t*1000:0.3f}, {s:2.0f}')

    outputs['t'] += [t]
    outputs['eta'] += [eta]
    outputs['s'] += [s]



fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(6, 3), dpi=100)

axes.plot(outputs['t'], outputs['s'], label=f's')
axes.plot(outputs['t'], outputs['eta'], label=f'eta')

axes.grid()
axes.legend()
axes.set_xlabel('Time (s)')
fig.tight_layout()
fig.savefig('pwm.svg')
