# -*- coding: utf-8 -*-
"""
A rate-based two compartment micro-circuit model of PCs, PVs, SSTs and VIPs.

Modeling for Li Yao's work on:
Temporal Reconfiguration ofCortical Microcircuits for Neural Activity by 
Visual Deprivation


Created on Sun Aug 15 13:41:45 2021

@author: Zilong Ji
Acknowledgement: Brainpy developer: Chaoming Wang
"""
import brainpy as bp
import numpy as np
bp.backend.set('numpy', dt=0.1) 
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron
from utils import SetConnectivity, grid_plot, grid_plot_diff

def build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, noise_strength, state='control'):
    #%%initialize the hyper-parameters  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
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
        
    #Connection Strength (Data from Li Yao's experiment)
    if state == 'control':
        Con_Stre = np.array([[10., -80., -10., 0.],
                             [25, -70, -10, -8],
                             [5, -40, 0, -5],
                             [10, -25, -10, 0]])
        #normalize the synaptic strength for stability
        Con_Stre = Con_Stre/80.0 
    elif state == 'md1':
        Con_Stre = np.array([[10., -80., -10., 0.],
                             [25, -70, -10, -8],
                             [5, -40, 0, -5],
                             [22, -25, -10, 0]])
        #normalize the synaptic strength for stability
        Con_Stre = Con_Stre/80.0
    elif state == 'md4':
        Con_Stre = np.array([[10., -38., -10., 0.],
                             [25, -70, -28, -8],
                             [5, -40, 0, -5],
                             [10, -25, -18, 0]])
        #normalize the synaptic strength for stability
        Con_Stre = Con_Stre/80.0 
    else:
        raise ValueError("Wrong State Name, Check It.")

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

#%% function for running and ploting the model when varying bottom up input and top-down input 
def grid_xs_xd(XS, XD, x_i_pv, x_i_sst, x_i_vip, noise_strength, state):
    """
    run the model when varying bottom up input and top-down input
    Input：
        micro_net: the network class
        pcs, pvs, ssts, vips: cell type class
        XS：bottom-up input grid
        XD: top-down input grid
    Ouput: 
        the fring rate array of four cell types: FPC, FPV, FSST, FVIP
        
    """
    FPC = np.zeros((len(XS),len(XD))); FPV = np.zeros((len(XS),len(XD)))
    FSST = np.zeros((len(XS),len(XD))); FVIP = np.zeros((len(XS),len(XD)))
    for i, x_si in enumerate(XS):
        for j, x_di in enumerate(XD):
            micro_net, pcs, pvs, ssts, vips = build_model(x_si, x_di, x_i_pv, x_i_sst, x_i_vip, noise_strength, state)       
            #run the model
            micro_net.run(1000, report=True)
            f_pc = np.mean(pcs.mon.r_pc, axis=1)[-1]; FPC[i,j] = f_pc
            f_pv = np.mean(pvs.mon.r_pv, axis=1)[-1]; FPV[i,j] = f_pv
            f_sst = np.mean(ssts.mon.r_sst, axis=1)[-1]; FSST[i,j] = f_sst
            f_vip = np.mean(vips.mon.r_vip, axis=1)[-1]; FVIP[i,j] = f_vip  
            print('Bottom-up input is {} Top down input is {}, Mean firing rate of\
                  PC {:.5f}, PV {:.5f}, SST {:.5f}, VIP {:.5f}'.
                  format(x_si, x_di, np.mean(pcs.mon.r_pc, axis=1)[-1],
                         np.mean(pvs.mon.r_pv, axis=1)[-1],
                         np.mean(ssts.mon.r_sst, axis=1)[-1],
                         np.mean(vips.mon.r_vip, axis=1)[-1])) 
    
    #plot
    grid_plot(FPC, FPV, FSST, FVIP, XS, XD, name='grid_'+state) 
    
    return FPC, FPV, FSST, FVIP

#%% comparison of 'control' and 'MD1'  

XS = np.arange(0,41,5)
XD = np.arange(0,41,5)
x_i_pv=1.5; x_i_sst=1.5; x_i_vip=0.5 #spontaneous
noise_strength=5.0
#
state='control'
FPC_ctrl, FPV_ctrl, FSST_ctrl, FVIP_ctrl = grid_xs_xd(XS, XD, x_i_pv, x_i_sst, 
                                                      x_i_vip, noise_strength, state)
#
state='md1'
FPC_md1, FPV_md1, FSST_md1, FVIP_md1 = grid_xs_xd(XS, XD, x_i_pv, x_i_sst, 
                                                  x_i_vip, noise_strength, state)    
#plot
FPC_diff = FPC_ctrl - FPC_md1; FPV_diff = FPV_ctrl - FPV_md1
FSST_diff = FSST_ctrl - FSST_md1; FVIP_diff = FVIP_ctrl - FVIP_md1
grid_plot_diff(FPC_diff, FPV_diff, FSST_diff, FVIP_diff, XS, XD, name='grid_diff_md1')

#%% comparison of 'control' and 'MD4'   

state='control'
FPC_ctrl, FPV_ctrl, FSST_ctrl, FVIP_ctrl = grid_xs_xd(XS, XD, x_i_pv, x_i_sst, 
                                                      x_i_vip, noise_strength, state)
#
state='md4'
FPC_md4, FPV_md4, FSST_md4, FVIP_md4 = grid_xs_xd(XS, XD, x_i_pv, x_i_sst, 
                                                  x_i_vip, noise_strength, state)    
#plot
FPC_diff = FPC_ctrl - FPC_md4; FPV_diff = FPV_ctrl - FPV_md4
FSST_diff = FSST_ctrl - FSST_md4; FVIP_diff = FVIP_ctrl - FVIP_md4
grid_plot_diff(FPC_diff, FPV_diff, FSST_diff, FVIP_diff, XS, XD, name='grid_diff_md4')
