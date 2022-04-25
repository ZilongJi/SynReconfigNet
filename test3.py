# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:44:51 2022

@author: Zilong
"""

import brainpy as bp

bp.math.set_platform('cpu')

E = bp.dyn.LIF(3200, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5., method='exp_auto')
I = bp.dyn.LIF(800, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5., method='exp_auto')

E.V[:] = bp.math.random.randn(3200) * 2 - 60.
I.V[:] = bp.math.random.randn(800) * 2 - 60.

E2E = bp.dyn.ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), E=0., g_max=0.6, tau=5., method='exp_auto')
E2I = bp.dyn.ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), E=0., g_max=0.6, tau=5., method='exp_auto')
I2E = bp.dyn.ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), E=-80., g_max=6.7, tau=10., method='exp_auto')
I2I = bp.dyn.ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), E=-80., g_max=6.7, tau=10., method='exp_auto')

net = bp.dyn.Network(E2E, E2I, I2E, I2I, E=E, I=I)

runner = bp.dyn.DSRunner(net,
                         monitors=['E.spike', 'I.spike'],
                         inputs=[('E.input', 20.), ('I.input', 20.)],
                         dt=0.1)

runner(100)