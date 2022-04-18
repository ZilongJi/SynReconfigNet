# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:36:45 2021

A rate-based two compartment micro-circuit model of PCs, PVs, SSTs and VIPs.

Modeling for Li Yao's work on:
Temporal Reconfiguration ofCortical Microcircuits for Neural Activity by 
Visual Deprivation


@author: Zilong Ji
Acknowledgement: Brainpy developer: Chaoming Wang
"""

import brainpy as bp
import numpy as np
from utils import SetConnectivity, joint_varying
bp.backend.set('numba', dt=0.1)
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron 

def build_model(Con_Stre, cond='spont'):
    #%%initialize the hyper-parameters  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
    lambda_s = 0.31; lambda_d = 0.27
    c = 7; theta_c = 28
    theta_s = 14
    #x_s = 17.5; x_d = 21 # 0-40?
    if cond == 'spont':
        noise_strength = 0.
        x_s = 18.0; x_d = 18.0
        x_i_pv = 1.5; x_i_sst = 1.5; xi_i_vip = 0.5
    elif cond == 'evoked':
        noise_strength = 0.
        x_s = 30.0; x_d = 18.0
        x_i_pv = 4.5; x_i_sst = 4.5; xi_i_vip = 1.3        
    
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
    
    vips = VIPNeuron(num_vip, tau_vip, noise_strength, xi_i_vip, W_vip_pc, W_vip_pv, W_vip_sst, 
                     W_vip_vip, monitors=['r_vip']) 
    
    pcs.PV = pvs; pcs.SST = ssts 
    pvs.PC = pcs; pvs.SST = ssts; pvs.VIP = vips
    ssts.PC = pcs; ssts.PV = pvs; ssts.VIP = vips
    vips.PC = pcs; vips.PV = pvs; vips.SST = ssts
    
    #build the network
    micro_net = bp.Network(pcs, pvs, ssts, vips)
    
    return micro_net, pcs, pvs, ssts, vips

#%% 
def vary_syn(cond):
    
    Con_Stre = np.array([[10., -80., -10., 0.],
                         [25, -70, -10, -8],
                         [5, -40, 0, -5],
                         [10, -25, -10, 0]])
    
    Synap1 = 'pc_pv'
    Synap2 = 'pv_sst'
    Synap_Stre_1 = np.arange(-38, -81, -4)
    Synap_Stre_2 = np.arange(-10, -29, -2)
    
    N = len(Synap_Stre_1); M=len(Synap_Stre_2)
    
    Results_PC = np.zeros((N,M)); Results_PV = np.zeros((N,M));
    Results_SST = np.zeros((N,M)); Results_VIP = np.zeros((N,M));  

    for i, syn1 in enumerate(Synap_Stre_1):
        for j, syn2 in enumerate(Synap_Stre_2):
            print('*'*50)
            print('pc_pv strength {}, pv_sst strength {}'.format(syn1, syn2))
            Con_Stre[0,1] = syn1
            Con_Stre[1,2] = syn2
            
            micro_net, pcs, pvs, ssts, vips = build_model(Con_Stre, cond)    
            #run the network
            micro_net.run(500, report=False)
            fpc = np.mean(pcs.mon.r_pc[-1,:])
            fpv = np.mean(pvs.mon.r_pv[-1,:])
            fsst = np.mean(ssts.mon.r_sst[-1,:])
            fvip = np.mean(vips.mon.r_vip[-1,:])  
            
            Results_PC[i,j] = fpc
            Results_PV[i,j] = fpv
            Results_SST[i,j] = fsst
            Results_VIP[i,j]= fvip    
    
    return Synap_Stre_1, Synap_Stre_2, Results_PC, Results_PV, Results_SST, Results_VIP

if __name__=='__main__':
    cond='spont'
    #cond='evoked'
    #Results_PC, Results_PV, Results_SST, Results_VIP = vary_syn(cond)
    X, Y, Results_PC, Results_PV, Results_SST, Results_VIP = vary_syn(cond)
    
    Synap_Stre_1 = np.arange(-38, -81, -4)
    Synap_Stre_2 = np.arange(-10, -29, -2)
    X, Y = np.meshgrid(Synap_Stre_1, Synap_Stre_2, indexing='ij')
    #%%plot imshow

    joint_varying(X, Y, Results_PC, Results_PV, Results_SST, Results_VIP, cond)

    