# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:40:54 2022

@author: Zilong
"""
import brainpy as bp
import brainpy.math as bm

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'plasma'

wc = bp.dyn.WilsonCowanModel(2,
                             wEE=16., wIE=15., wEI=12., wII=3.,
                             E_a=1.5, I_a=1.5, E_theta=3., I_theta=3.,
                             method='exp_euler_auto')
wc.x[:] = [-0.2, 1.]
wc.y[:] = [0.0, 1.]

runner = bp.dyn.DSRunner(wc, monitors=['x', 'y'], inputs=['input', -0.5])
runner.run(10.)

bp.visualize.line_plot(runner.mon.ts, runner.mon.x,
                       plot_ids=[0, 1], legend='e', show=True)