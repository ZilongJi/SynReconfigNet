# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:13:07 2021

A rate-based two compartment micro-circuit model of PCs, PVs, SSTs and VIPs.

Modeling for Li Yao's work on:
Temporal Reconfiguration ofCortical Microcircuits for Neural Activity by 
Visual Deprivation

@author: Zilong Ji
Acknowledgement: Brainpy developer: Chaoming Wang
"""

import brainpy as bp
import numpy as np
import csv
from utils import SetConnectivity
bp.backend.set('numpy', dt=0.1)
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron 

def build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre):
    #%%initialize the hyper-parameters  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
    noise_strength = 0.
    lambda_s = 0.31; lambda_d = 0.27
    c = 7; theta_c = 28
    theta_s = 14
    
    #total number of neurons (nparray)
    NC = np.asarray([num_pc, num_pv, num_sst, num_vip])
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Prob = np.array([[0.09,0.80,0.08,0.005],
                         [0.60,0.65,0.38,0.13],
                         [0.45,0.22,0.0,0.10],
                         [0.21,0.24,0.23,0.0]])
    
    #normalize the synaptic strength for stability
    Con_Stre = Con_Stre/80.0

    #%% generate the synaptic connections    
    Weights = SetConnectivity(Con_Prob, Con_Stre, NC) 
    
    W_pc_pc, W_pc_pv, W_pc_sst \
        = Weights['pc_pc'], Weights['pc_pv'], Weights['pc_sst']
    
    W_pv_pc, W_pv_pv, W_pv_sst, W_pv_vip \
        = Weights['pv_pc'], Weights['pv_pv'], Weights['pv_sst'], Weights['pv_vip']
    
    W_sst_pc, W_sst_pv, W_sst_sst, W_sst_vip \
        = Weights['sst_pc'], Weights['sst_pv'], Weights['sst_sst'], Weights['sst_vip']
    
    W_vip_pc, W_vip_pv, W_vip_sst, W_vip_vip \
        = Weights['vip_pc'], Weights['vip_pv'], Weights['vip_sst'], Weights['vip_vip']

    #%% initialize the neuron class and build the network
    pcs = PCNeuron(num_pc, tau_pc, noise_strength, lambda_s, lambda_d, x_s, x_d, c, theta_s, 
                   theta_c, W_pc_pv, W_pc_pc, W_pc_sst, monitors=['r_pc', 'I_0'])
    pvs = PVNeuron(num_pv, tau_pv, noise_strength, x_i_pv, W_pv_pc, W_pv_pv, W_pv_sst, W_pv_vip,
                   monitors=['r_pv'])   
    
    ssts = SSTNeuron(num_sst, tau_sst, noise_strength, x_i_sst, W_sst_pc, W_sst_pv, W_sst_sst, 
                     W_sst_vip, monitors=['r_sst'])
    
    vips = VIPNeuron(num_vip, tau_vip, noise_strength, x_i_vip, W_vip_pc, W_vip_pv, W_vip_sst, 
                     W_vip_vip, monitors=['r_vip']) 
    
    pcs.PV = pvs; pcs.SST = ssts 
    pvs.PC = pcs; pvs.SST = ssts; pvs.VIP = vips
    ssts.PC = pcs; ssts.PV = pvs; ssts.VIP = vips
    vips.PC = pcs; vips.PV = pvs; vips.SST = ssts
    
    #build the network
    micro_net = bp.Network(pcs, pvs, ssts, vips)
    
    return micro_net, pcs, pvs, ssts, vips

def search_spont():
    
    with open("./paramsearch/results_spont.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index",\
                     "control_pc","control_pv","control_sst", "control_vip",\
                     "md1_pc", "md1_pv", "md1_sst", "md1_vip",\
                     "md4_pc", "md4_pv", "md4_sst", "md4_vip",\
                     "x_s", "x_d", "x_i_pv", "x_i_sst", "x_i_vip",\
                     "count1", "count2", "cond1", "cond2", "cond3", "cond4", "cond5",\
                     "cond6", "cond7", "cond8", "cond9"])
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_Ctrl = np.array([[10., -80., -10., 0.],
                              [25, -70, -10, -8],
                              [5, -40, 0, -5],
                              [10, -25, -10, 0]])   
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md1 = np.array([[10., -80., -10., 0.],
                             [34, -70, -10, -8],
                             [5, -40, 0, -5],
                             [22, -25, -10, 0]])  
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md4 = np.array([[10., -38., -10., 0.],
                             [25, -70, -28, -8],
                             [5, -40, 0, -5],
                             [10, -25, -18, 0]])       
    search_times = 3000
    
    for st in range(search_times):
        print('-'*50)
        print('Current search count {}'.format(st))
        #random sample the external input in a range
        x_s = np.random.uniform(low=15, high=20, size=1)[0]
        x_d = np.random.uniform(low=10, high=20, size=1)[0]
        x_i_pv = np.random.uniform(low=0, high=3, size=1)[0]
        x_i_sst = np.random.uniform(low=0, high=3, size=1)[0]
        x_i_vip = np.random.uniform(low=0, high=3, size=1)[0]
        
        
        micro_net_ctrl, pcs_ctrl, pvs_ctrl, ssts_ctrl, vips_ctrl = \
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_Ctrl)
        #run the network control
        micro_net_ctrl.run(500, report=False)
        fpc_ctrl = np.mean(pcs_ctrl.mon.r_pc[-1,:])
        fpv_ctrl = np.mean(pvs_ctrl.mon.r_pv[-1,:])
        fsst_ctrl = np.mean(ssts_ctrl.mon.r_sst[-1,:])
        fvip_ctrl = np.mean(vips_ctrl.mon.r_vip[-1,:])
        print('control pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl))
        
        micro_net_md1, pcs_md1, pvs_md1, ssts_md1, vips_md1 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md1)
        #run the network md1
        micro_net_md1.run(500, report=False)
        fpc_md1 = np.mean(pcs_md1.mon.r_pc[-1,:])
        fpv_md1 = np.mean(pvs_md1.mon.r_pv[-1,:])
        fsst_md1 = np.mean(ssts_md1.mon.r_sst[-1,:])
        fvip_md1 = np.mean(vips_md1.mon.r_vip[-1,:])
        print('md1 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md1, fpv_md1, fsst_md1, fvip_md1))
            
        micro_net_md4, pcs_md4, pvs_md4, ssts_md4, vips_md4 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md4)    
        #run the network md4
        micro_net_md4.run(500, report=False)        
        fpc_md4 = np.mean(pcs_md4.mon.r_pc[-1,:])
        fpv_md4 = np.mean(pvs_md4.mon.r_pv[-1,:])
        fsst_md4 = np.mean(ssts_md4.mon.r_sst[-1,:])
        fvip_md4 = np.mean(vips_md4.mon.r_vip[-1,:])   
        print('md4 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md4, fpv_md4, fsst_md4, fvip_md4))
     
        cond1 = 1.5<fpv_ctrl<1.8
        cond2 = 0.8<fsst_ctrl<1.2
        cond3 = 0.4<fvip_ctrl<0.7
        cond4 = 1.1<fpv_md1<1.5
        cond5 = 0.7<fsst_md1<1.0
        cond6 = 0.7<fvip_md1<1.1
        cond7 = 1.0<fpv_md4<1.4
        cond8 = 0.8<fsst_md4<1.2
        cond9 = 0.0<fvip_md4<0.4
        
        count1 = int(cond1)+int(cond2)+int(cond3)
        count2 = int(cond1)+int(cond2)+int(cond3)+int(cond4)\
            +int(cond5)+int(cond6)+int(cond7)+int(cond8)+int(cond9)        
        print('total meeted conditions {} and {}'.format(count1, count2))
        
        with open("./paramsearch/results_spont.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile) 
            writer.writerow([st, fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl,
                             fpc_md1, fpv_md1, fsst_md1, fvip_md1,
                             fpc_md4, fpv_md4, fsst_md4, fvip_md4,
                             x_s, x_d, x_i_pv, x_i_sst, x_i_vip, count1, count2,
                             cond1, cond2, cond3, cond4, cond5, cond6,
                             cond7, cond8, cond9])

def search_spont_topdown():
    
    with open("./paramsearch/results_spont_topdown.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index",\
                     "control_pc","control_pv","control_sst", "control_vip",\
                     "md1_pc", "md1_pv", "md1_sst", "md1_vip",\
                     "md4_pc", "md4_pv", "md4_sst", "md4_vip",\
                     "x_s", "x_d", "x_i_pv", "x_i_sst", "x_i_vip",\
                     "count1", "count2", "cond1", "cond2", "cond3", "cond4", "cond5",\
                     "cond6", "cond7", "cond8", "cond9"])
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_Ctrl = np.array([[10., -80., -10., 0.],
                              [25, -70, -10, -8],
                              [5, -40, 0, -5],
                              [10, -25, -10, 0]])   
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md1 = np.array([[10., -80., -10., 0.],
                             [34, -70, -10, -8],
                             [5, -40, 0, -5],
                             [22, -25, -10, 0]])  
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md4 = np.array([[10., -38., -10., 0.],
                             [25, -70, -28, -8],
                             [5, -40, 0, -5],
                             [10, -25, -18, 0]])       
    search_times = 3000
    
    for st in range(search_times):
        print('-'*50)
        print('Current search count {}'.format(st))
        #random sample the external input in a range
        x_s = 18
        x_d = np.random.uniform(low=15, high=20, size=1)[0]
        x_i_pv = 1.5
        x_i_sst = 1.5
        x_i_vip = 0.5
        
        
        micro_net_ctrl, pcs_ctrl, pvs_ctrl, ssts_ctrl, vips_ctrl = \
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_Ctrl)
        #run the network control
        micro_net_ctrl.run(500, report=False)
        fpc_ctrl = np.mean(pcs_ctrl.mon.r_pc[-1,:])
        fpv_ctrl = np.mean(pvs_ctrl.mon.r_pv[-1,:])
        fsst_ctrl = np.mean(ssts_ctrl.mon.r_sst[-1,:])
        fvip_ctrl = np.mean(vips_ctrl.mon.r_vip[-1,:])
        print('control pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl))
        
        micro_net_md1, pcs_md1, pvs_md1, ssts_md1, vips_md1 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md1)
        #run the network md1
        micro_net_md1.run(500, report=False)
        fpc_md1 = np.mean(pcs_md1.mon.r_pc[-1,:])
        fpv_md1 = np.mean(pvs_md1.mon.r_pv[-1,:])
        fsst_md1 = np.mean(ssts_md1.mon.r_sst[-1,:])
        fvip_md1 = np.mean(vips_md1.mon.r_vip[-1,:])
        print('md1 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md1, fpv_md1, fsst_md1, fvip_md1))
            
        micro_net_md4, pcs_md4, pvs_md4, ssts_md4, vips_md4 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md4)    
        #run the network md4
        micro_net_md4.run(500, report=False)        
        fpc_md4 = np.mean(pcs_md4.mon.r_pc[-1,:])
        fpv_md4 = np.mean(pvs_md4.mon.r_pv[-1,:])
        fsst_md4 = np.mean(ssts_md4.mon.r_sst[-1,:])
        fvip_md4 = np.mean(vips_md4.mon.r_vip[-1,:])   
        print('md4 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md4, fpv_md4, fsst_md4, fvip_md4))
     
        cond1 = 1.5<fpv_ctrl<1.8
        cond2 = 0.8<fsst_ctrl<1.2
        cond3 = 0.4<fvip_ctrl<0.7
        cond4 = 1.1<fpv_md1<1.5
        cond5 = 0.7<fsst_md1<1.0
        cond6 = 0.7<fvip_md1<1.1
        cond7 = 1.0<fpv_md4<1.4
        cond8 = 0.8<fsst_md4<1.2
        cond9 = 0.0<fvip_md4<0.4
        
        count1 = int(cond1)+int(cond2)+int(cond3)
        count2 = int(cond1)+int(cond2)+int(cond3)+int(cond4)\
            +int(cond5)+int(cond6)+int(cond7)+int(cond8)+int(cond9)        
        print('total meeted conditions {} and {}'.format(count1, count2))
        
        with open("./paramsearch/results_spont_topdown.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile) 
            writer.writerow([st, fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl,
                             fpc_md1, fpv_md1, fsst_md1, fvip_md1,
                             fpc_md4, fpv_md4, fsst_md4, fvip_md4,
                             x_s, x_d, x_i_pv, x_i_sst, x_i_vip, count1, count2,
                             cond1, cond2, cond3, cond4, cond5, cond6,
                             cond7, cond8, cond9])

def search_evoked():

    with open("./paramsearch/results_evoked.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index",\
                     "control_pc","control_pv","control_sst", "control_vip",\
                     "md1_pc", "md1_pv", "md1_sst", "md1_vip",\
                     "md4_pc", "md4_pv", "md4_sst", "md4_vip",\
                     "x_s", "x_d", "x_i_pv", "x_i_sst", "x_i_vip",\
                     "count1", "count2", "cond1", "cond2", "cond3", "cond4", "cond5",\
                     "cond6", "cond7", "cond8", "cond9"])
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_Ctrl = np.array([[10., -80., -10., 0.],
                              [25, -70, -10, -8],
                              [5, -40, 0, -5],
                              [10, -25, -10, 0]])   
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md1 = np.array([[10., -80., -10., 0.],
                             [34, -70, -10, -8],
                             [5, -40, 0, -5],
                             [22, -25, -10, 0]])  
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md4 = np.array([[10., -38., -10., 0.],
                             [25, -70, -28, -8],
                             [5, -40, 0, -5],
                             [10, -25, -18, 0]])       
    
    search_times = 3000
    
    for st in range(search_times):
        print('-'*50)
        print('Current search count {}'.format(st))
        #random sample the external input in a range
        x_s = np.random.uniform(low=10, high=40, size=1)[0]
        x_d = np.random.uniform(low=10, high=40, size=1)[0]
        x_i_pv = np.random.uniform(low=0, high=5, size=1)[0]
        x_i_sst = np.random.uniform(low=0, high=5, size=1)[0]
        x_i_vip = np.random.uniform(low=0, high=5, size=1)[0]
        
        
        micro_net_ctrl, pcs_ctrl, pvs_ctrl, ssts_ctrl, vips_ctrl = \
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_Ctrl)
        #run the network control
        micro_net_ctrl.run(500, report=False)
        fpc_ctrl = np.mean(pcs_ctrl.mon.r_pc[-1,:])
        fpv_ctrl = np.mean(pvs_ctrl.mon.r_pv[-1,:])
        fsst_ctrl = np.mean(ssts_ctrl.mon.r_sst[-1,:])
        fvip_ctrl = np.mean(vips_ctrl.mon.r_vip[-1,:])
        print('control pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl))
        
        micro_net_md1, pcs_md1, pvs_md1, ssts_md1, vips_md1 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md1)
        #run the network md1
        micro_net_md1.run(500, report=False)
        fpc_md1 = np.mean(pcs_md1.mon.r_pc[-1,:])
        fpv_md1 = np.mean(pvs_md1.mon.r_pv[-1,:])
        fsst_md1 = np.mean(ssts_md1.mon.r_sst[-1,:])
        fvip_md1 = np.mean(vips_md1.mon.r_vip[-1,:])
        print('md1 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md1, fpv_md1, fsst_md1, fvip_md1))
            
        micro_net_md4, pcs_md4, pvs_md4, ssts_md4, vips_md4 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md4)    
        #run the network md4
        micro_net_md4.run(500, report=False)        
        fpc_md4 = np.mean(pcs_md4.mon.r_pc[-1,:])
        fpv_md4 = np.mean(pvs_md4.mon.r_pv[-1,:])
        fsst_md4 = np.mean(ssts_md4.mon.r_sst[-1,:])
        fvip_md4 = np.mean(vips_md4.mon.r_vip[-1,:])   
        print('md4 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md4, fpv_md4, fsst_md4, fvip_md4)) 
 
        cond1 = 4.0<fpv_ctrl<6.0
        cond2 = 3.0<fsst_ctrl<4.0
        cond3 = 2.0<fvip_ctrl<3.0
        cond4 = 6.0<fpv_md1<10.0
        cond5 = 3.0<fsst_md1<4.0
        cond6 = 3.0<fvip_md1<4.0
        cond7 = 4.0<fpv_md4<6.0
        cond8 = 3.0<fsst_md4<4.0
        cond9 = 2.0<fvip_md4<3.0        
 
        count1 = int(cond1)+int(cond2)+int(cond3)
        count2 = int(cond1)+int(cond2)+int(cond3)+int(cond4)\
            +int(cond5)+int(cond6)+int(cond7)+int(cond8)+int(cond9)
        print('total meeted conditions {} and {}'.format(count1, count2))
        
        
        with open("./paramsearch/results_evoked.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile) 
            
            writer.writerow([st, fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl,
                             fpc_md1, fpv_md1, fsst_md1, fvip_md1,
                             fpc_md4, fpv_md4, fsst_md4, fvip_md4,
                             x_s, x_d, x_i_pv, x_i_sst, x_i_vip, count1, count2,
                             cond1, cond2, cond3, cond4, cond5, cond6,
                             cond7, cond8, cond9])

def search_evoked_bottomup():

    with open("./paramsearch/results_evoked_bottomup.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index",\
                     "control_pc","control_pv","control_sst", "control_vip",\
                     "md1_pc", "md1_pv", "md1_sst", "md1_vip",\
                     "md4_pc", "md4_pv", "md4_sst", "md4_vip",\
                     "x_s", "x_d", "x_i_pv", "x_i_sst", "x_i_vip",\
                     "count1", "count2", "cond1", "cond2", "cond3", "cond4", "cond5",\
                     "cond6", "cond7", "cond8", "cond9"])
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_Ctrl = np.array([[10., -80., -10., 0.],
                              [25, -70, -10, -8],
                              [5, -40, 0, -5],
                              [10, -25, -10, 0]])   
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md1 = np.array([[10., -80., -10., 0.],
                             [34, -70, -10, -8],
                             [5, -40, 0, -5],
                             [22, -25, -10, 0]])  
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md4 = np.array([[10., -38., -10., 0.],
                             [25, -70, -28, -8],
                             [5, -40, 0, -5],
                             [10, -25, -18, 0]])       
    
    search_times = 3000
    
    for st in range(search_times):
        print('-'*50)
        print('Current search count {}'.format(st))
        #random sample the external input in a range
        x_s = np.random.uniform(low=30, high=40, size=1)[0]
        x_d = 18
        x_i_pv = 4.5
        x_i_sst = 4.5
        x_i_vip = np.random.uniform(low=0, high=5, size=1)[0]
        
        
        micro_net_ctrl, pcs_ctrl, pvs_ctrl, ssts_ctrl, vips_ctrl = \
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_Ctrl)
        #run the network control
        micro_net_ctrl.run(500, report=False)
        fpc_ctrl = np.mean(pcs_ctrl.mon.r_pc[-1,:])
        fpv_ctrl = np.mean(pvs_ctrl.mon.r_pv[-1,:])
        fsst_ctrl = np.mean(ssts_ctrl.mon.r_sst[-1,:])
        fvip_ctrl = np.mean(vips_ctrl.mon.r_vip[-1,:])
        print('control pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl))
        
        micro_net_md1, pcs_md1, pvs_md1, ssts_md1, vips_md1 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md1)
        #run the network md1
        micro_net_md1.run(500, report=False)
        fpc_md1 = np.mean(pcs_md1.mon.r_pc[-1,:])
        fpv_md1 = np.mean(pvs_md1.mon.r_pv[-1,:])
        fsst_md1 = np.mean(ssts_md1.mon.r_sst[-1,:])
        fvip_md1 = np.mean(vips_md1.mon.r_vip[-1,:])
        print('md1 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md1, fpv_md1, fsst_md1, fvip_md1))
            
        micro_net_md4, pcs_md4, pvs_md4, ssts_md4, vips_md4 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md4)    
        #run the network md4
        micro_net_md4.run(500, report=False)        
        fpc_md4 = np.mean(pcs_md4.mon.r_pc[-1,:])
        fpv_md4 = np.mean(pvs_md4.mon.r_pv[-1,:])
        fsst_md4 = np.mean(ssts_md4.mon.r_sst[-1,:])
        fvip_md4 = np.mean(vips_md4.mon.r_vip[-1,:])   
        print('md4 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md4, fpv_md4, fsst_md4, fvip_md4)) 
 
        cond1 = 4.5<fpv_ctrl<6.0
        cond2 = 3.0<fsst_ctrl<4.0
        cond3 = 2.0<fvip_ctrl<3.0
        cond4 = 4.0<fpv_md1<6.0
        cond5 = 2.0<fsst_md1<3.0
        cond6 = 3.0<fvip_md1<4.0
        cond7 = 4.0<fpv_md4<6.0
        cond8 = 1.0<fsst_md4<2.0
        cond9 = 1.0<fvip_md4<1.5        
 
        count1 = int(cond1)+int(cond2)+int(cond3)
        count2 = int(cond1)+int(cond2)+int(cond3)+int(cond4)\
            +int(cond5)+int(cond6)+int(cond7)+int(cond8)+int(cond9)
        print('total meeted conditions {} and {}'.format(count1, count2))
        
        
        with open("./paramsearch/results_evoked_bottomup.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile) 
            
            writer.writerow([st, fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl,
                             fpc_md1, fpv_md1, fsst_md1, fvip_md1,
                             fpc_md4, fpv_md4, fsst_md4, fvip_md4,
                             x_s, x_d, x_i_pv, x_i_sst, x_i_vip, count1, count2,
                             cond1, cond2, cond3, cond4, cond5, cond6,
                             cond7, cond8, cond9])
        
def search_evoked_vip():

    with open("./paramsearch/results_evoked_vip.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index",\
                         "control_pc", "md1_pc", "md4_pc", \
                         "control_pv", "md1_pv", "md4_pv", \
                         "control_sst", "md1_sst", "md4_sst", \
                         "control_vip", "md1_vip", "md4_vip",\
                         "x_s", "x_d", "x_i_pv", "x_i_sst", "x_i_vip",\
                         "count", "cond1", "cond2", "cond3", "cond4", "cond5",\
                         "cond6", "cond7", "cond8", "cond9"])
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_Ctrl = np.array([[10., -80., -10., 0.],
                              [25, -70, -10, -8],
                              [5, -40, 0, -5],
                              [10, -25, -10, 0]])   
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md1 = np.array([[10., -80., -10., 0.],
                             [34, -70, -10, -8],
                             [5, -40, 0, -5],
                             [22, -25, -10, 0]])  
    
    #Connection Probability (Data from Li Yao's experiment)
    Con_Stre_md4 = np.array([[10., -38., -10., 0.],
                             [25, -70, -28, -8],
                             [5, -40, 0, -5],
                             [10, -25, -18, 0]])       
    
    search_times = 3000
    
    for st in range(search_times):
        print('-'*50)
        print('Current search count {}'.format(st))
        #random sample the external input in a range
        x_s = 34
        x_d = 24
        x_i_pv = 3.8
        x_i_sst = 4.8
        x_i_vip = np.random.uniform(low=1.0, high=3.0, size=1)[0]
        
        
        micro_net_ctrl, pcs_ctrl, pvs_ctrl, ssts_ctrl, vips_ctrl = \
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_Ctrl)
        #run the network control
        micro_net_ctrl.run(500, report=False)
        fpc_ctrl = np.mean(pcs_ctrl.mon.r_pc[-1,:])
        fpv_ctrl = np.mean(pvs_ctrl.mon.r_pv[-1,:])
        fsst_ctrl = np.mean(ssts_ctrl.mon.r_sst[-1,:])
        fvip_ctrl = np.mean(vips_ctrl.mon.r_vip[-1,:])
        print('control pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_ctrl, fpv_ctrl, fsst_ctrl, fvip_ctrl))
        
        micro_net_md1, pcs_md1, pvs_md1, ssts_md1, vips_md1 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md1)
        #run the network md1
        micro_net_md1.run(500, report=False)
        fpc_md1 = np.mean(pcs_md1.mon.r_pc[-1,:])
        fpv_md1 = np.mean(pvs_md1.mon.r_pv[-1,:])
        fsst_md1 = np.mean(ssts_md1.mon.r_sst[-1,:])
        fvip_md1 = np.mean(vips_md1.mon.r_vip[-1,:])
        print('md1 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md1, fpv_md1, fsst_md1, fvip_md1))
            
        micro_net_md4, pcs_md4, pvs_md4, ssts_md4, vips_md4 =\
            build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, Con_Stre_md4)    
        #run the network md4
        micro_net_md4.run(500, report=False)        
        fpc_md4 = np.mean(pcs_md4.mon.r_pc[-1,:])
        fpv_md4 = np.mean(pvs_md4.mon.r_pv[-1,:])
        fsst_md4 = np.mean(ssts_md4.mon.r_sst[-1,:])
        fvip_md4 = np.mean(vips_md4.mon.r_vip[-1,:])   
        print('md4 pc {:.4f} pv {:.4f} sst {:.4f} vip {:.4f}'.format(fpc_md4, fpv_md4, fsst_md4, fvip_md4))
        
        cond1 = 4.0<fpv_ctrl<6.0
        cond2 = 3.0<fsst_ctrl<4.0
        cond3 = 2.0<fvip_ctrl<3.0
        cond4 = 6.0<fpv_md1<10.0
        cond5 = 3.0<fsst_md1<4.0
        cond6 = 3.0<fvip_md1<4.0
        cond7 = 4.0<fpv_md4<6.0
        cond8 = 3.0<fsst_md4<4.0
        cond9 = 2.0<fvip_md4<3.0         
        
        count = int(cond1)+int(cond2)+int(cond3)+int(cond4)\
            +int(cond5)+int(cond6)+int(cond7)+int(cond8)+int(cond9)
        print('total meeted conditions {}'.format(count))
        
        
        with open("./paramsearch/results_evoked_vip.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile) 

            writer.writerow([st, fpc_ctrl, fpc_md1, fpc_md4, 
                             fpv_ctrl, fpv_md1, fpv_md4, 
                             fsst_ctrl, fsst_md1, fsst_md4, 
                             fvip_ctrl, fvip_md1, fvip_md4,
                             x_s, x_d, x_i_pv, x_i_sst, x_i_vip, count,
                             cond1, cond2, cond3, cond4, cond5, cond6,
                             cond7, cond8, cond9])

if __name__=='__main__':
    search_evoked_vip()
    #search_spont()
    #search_spont_topdown()
    #search_evoked()
    #search_evoked_bottomup()