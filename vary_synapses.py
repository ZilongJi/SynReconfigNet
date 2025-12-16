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
import pickle
import brainpy as bp
import numpy as np
from utils import SetConnectivity, slope_plot, percentage_plot, slopewithshadow_plot, slope_rank_plot, synaptic_contribution_plot

from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron 

seed = 0

def build_model(Con_Stre, cond='spont'):
    #%%initialize the hyper-parameters  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
    lambda_s = 0.31; lambda_d = 0.27
    c = 7; theta_c = 28
    theta_s = 14
    #x_s = 17.5; x_d = 21 # 0-40?
    if cond == 'spont':
        noise_strength = 5.
        x_s = 18.0; x_d = 18.0
        x_i_pv = 1.5; x_i_sst = 1.5; xi_i_vip = 0.5
    elif cond == 'evoked':
        noise_strength = 5.
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
                   theta_c, W_pc_pv, W_pc_pc, W_pc_sst, seed) #, monitors=['r_pc', 'I_0']
    pvs = PVNeuron(num_pv, tau_pv, noise_strength, x_i_pv, W_pv_pc, W_pv_pv, W_pv_sst, W_pv_vip,
                   seed)   #, monitors=['r_pv']
    
    ssts = SSTNeuron(num_sst, tau_sst, noise_strength, x_i_sst, W_sst_pc, W_sst_pv, W_sst_sst, 
                     seed, W_sst_vip) #, monitors=['r_sst']
    
    vips = VIPNeuron(num_vip, tau_vip, noise_strength, xi_i_vip, W_vip_pc, W_vip_pv, W_vip_sst, 
                     W_vip_vip, seed) #, monitors=['r_vip']
    
    pcs.PV = pvs; pcs.SST = ssts 
    pvs.PC = pcs; pvs.SST = ssts; pvs.VIP = vips
    ssts.PC = pcs; ssts.PV = pvs; ssts.VIP = vips
    vips.PC = pcs; vips.PV = pvs; vips.SST = ssts
    
    #build the network
    micro_net = bp.Network(pcs, pvs, ssts, vips)
    
    return micro_net, pcs, pvs, ssts, vips

def vary_strength_and_run(pre_i, post_j, synap_name, Con_Stre, lowerbound, upperbound, cond, interval=1):
    
    orig_synap_strength = Con_Stre[post_j,pre_i]
    #extract the lower boundary and the upper boundary
    low = lowerbound[post_j,pre_i]; up = upperbound[post_j,pre_i]
    Syn_Strength = np.arange(np.abs(low), np.abs(up)+1, interval) * np.sign(orig_synap_strength)
    
    #for store the firing rate at each synaptic strength
    FPC = []; FPV = []; FSST = []; FVIP = []
    for syn_strength in Syn_Strength:
        print('Synapse name {}; Set synaptic strength to {:.2f}'.format(synap_name, syn_strength))
        
        #copy the connection matrix
        Con_Stre_copy = Con_Stre.copy()
        
        #change the prototype strength
        Con_Stre_copy[post_j,pre_i] = syn_strength
        
        micro_net, pcs, pvs, ssts, vips = build_model(Con_Stre_copy, cond)    
        
        #run the network
        micro_net.run(500, report=False)
        
        fpc = np.mean(pcs.mon.r_pc[-1,:])
        FPC.append(fpc)
        
        fpv = np.mean(pvs.mon.r_pv[-1,:])
        FPV.append(fpv)
        
        fsst = np.mean(ssts.mon.r_sst[-1,:])
        FSST.append(fsst)

        fvip = np.mean(vips.mon.r_vip[-1,:])
        FVIP.append(fvip)        
        
    return FPC, FPV, FSST, FVIP, Syn_Strength

#%% 
def vary_syn(cond):
    
    Con_Stre = np.array([[10., -80., -10., 0.],
                         [25, -70, -10, -8],
                         [5, -40, 0, -5],
                         [10, -25, -10, 0]])
    
    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']
    
    lowerbound = np.array([[0., -30., 0., 0.],
                    [15, -60, 0, 0],
                    [0, -30, 0, 0],
                    [0, -15, 0, 0]])
    upperbound = np.array([[20, -100, -20, 0.],
                   [35, -80, -30, -20],
                   [10, -50, 0, -10],
                   [30, -35, -20, 0]])
    
    Results_PC = {}; Results_PV = {}
    Results_SST = {}; Results_VIP = {}
    for pre_i, pre in enumerate(PreSynap):
        for post_j, post in enumerate(PostSynap):
            synap_name = post+'_'+pre
            
            if synap_name in ['pc_vip', 'sst_sst', 'vip_vip']:
                continue
            
            print('*'*50)
            FPC, FPV, FSST, FVIP, Syn_Strength = vary_strength_and_run(pre_i, post_j, synap_name, Con_Stre, lowerbound, upperbound, cond)
            Results_PC[synap_name] = FPC
            Results_PV[synap_name] = FPV
            Results_SST[synap_name] = FSST
            Results_VIP[synap_name] = FVIP
            #----------------------
            Results_PC[synap_name+'_input'] = Syn_Strength
            Results_PV[synap_name+'_input'] = Syn_Strength
            Results_SST[synap_name+'_input'] = Syn_Strength
            Results_VIP[synap_name+'_input'] = Syn_Strength        
            
    #%%slope plot 
    slope_plot(Results_PC, Con_Stre, lowerbound, upperbound, cond, celltype='pc')
    slope_plot(Results_PV, Con_Stre, lowerbound, upperbound, cond, celltype='pv')
    slope_plot(Results_SST, Con_Stre, lowerbound, upperbound, cond, celltype='sst')
    slope_plot(Results_VIP, Con_Stre, lowerbound, upperbound, cond, celltype='vip')
    
    #%%percetange plot
    percentage_plot(Results_PC, cond, celltype='pc')
    percentage_plot(Results_PV, cond, celltype='pv')
    percentage_plot(Results_SST, cond, celltype='sst')
    percentage_plot(Results_VIP, cond, celltype='vip')
    
    return Results_PC, Results_PV, Results_SST, Results_VIP

def vary_one_synapse(presyn_name, postsyn_name, cond, ntrial=10, interval=2):

    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']
    
    pre_i = PreSynap.index(presyn_name)
    post_j = PostSynap.index(postsyn_name)
    
    Con_Stre = np.array([[10., -80., -10., 0.],
                         [25, -70, -10, -8],
                         [5, -40, 0, -5],
                         [10, -25, -10, 0]])
    
    lowerbound = np.array([[0., -30., 0., 0.],
                    [15, -60, 0, 0],
                    [0, -30, 0, 0],
                    [0, -15, 0, 0]])
    upperbound = np.array([[20, -100, -20, 0.],
                   [35, -80, -30, -20],
                   [10, -50, 0, -10],
                   [30, -35, -20, 0]])
    
    Results_PC = {}; Results_PV = {}
    Results_SST = {}; Results_VIP = {}

    synap_name = postsyn_name+'_'+presyn_name
    
    for i in range(ntrial):
        print('trial {}'.format(i))
        FPC, FPV, FSST, FVIP, Syn_Strength = vary_strength_and_run(pre_i, post_j, synap_name, Con_Stre, lowerbound, upperbound, cond, interval)
        Results_PC[i] = FPC
        Results_PV[i] = FPV
        Results_SST[i] = FSST
        Results_VIP[i] = FVIP
    
    return Results_PC, Results_PV, Results_SST, Results_VIP, Syn_Strength   

def vary_strength_at_twoends_run(synap_name, Con_Stre, cond):
    
    if synap_name == 'pc_pv':
        Md_Stre = np.array([[10., -38., -10., 0.],
                             [25, -70, -10, -8],
                             [5, -40, 0, -5],
                             [10, -25, -10, 0]])
    elif synap_name == 'pv_sst': 
        Md_Stre = np.array([[10., -80., -10., 0.],
                             [25, -70, -28, -8],
                             [5, -40, 0, -5],
                             [10, -25, -10, 0]])
    elif synap_name == 'vip_sst': 
        Md_Stre = np.array([[10., -80., -10., 0.],
                             [25, -70, -10, -8],
                             [5, -40, 0, -5],
                             [10, -25, -18, 0]])   
        
        
    #for store the firing rate at each synaptic strength
    FPC = []; FPV = []; FSST = []; FVIP = []   
    
    #run the network at control condition
    micro_net, pcs, pvs, ssts, vips = build_model(Con_Stre, cond) 
    micro_net.run(500, report=False) 
    fpc = np.mean(pcs.mon.r_pc[-1,:]); FPC.append(fpc)
    fpv = np.mean(pvs.mon.r_pv[-1,:]); FPV.append(fpv)
    fsst = np.mean(ssts.mon.r_sst[-1,:]); FSST.append(fsst)
    fvip = np.mean(vips.mon.r_vip[-1,:]); FVIP.append(fvip)    
    
    #run the network at md4 condition
    micro_net, pcs, pvs, ssts, vips = build_model(Md_Stre, cond)     
    micro_net.run(500, report=False) 
    fpc = np.mean(pcs.mon.r_pc[-1,:]); FPC.append(fpc)
    fpv = np.mean(pvs.mon.r_pv[-1,:]); FPV.append(fpv)
    fsst = np.mean(ssts.mon.r_sst[-1,:]); FSST.append(fsst)
    fvip = np.mean(vips.mon.r_vip[-1,:]); FVIP.append(fvip) 

    return FPC, FPV, FSST, FVIP     # list=[control_result,md_result]

def vary_one_synapse_only_start_and_end(cond, ntrial=10):

    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']
    
    Con_Stre = np.array([[10., -80., -10., 0.],
                           [25, -70, -10, -8],
                           [5, -40, 0, -5],
                           [10, -25, -10, 0]])

    Results_PC = {}; Results_PV = {}
    Results_SST = {}; Results_VIP = {}

    for pre_i, pre in enumerate(PreSynap):
        for post_j, post in enumerate(PostSynap):    

            synap_name = post+'_'+pre
            
            if synap_name in ['pc_pv', 'pv_sst', 'vip_sst']:
                ALL_PC = []; ALL_PV = []; ALL_SST = []; ALL_VIP = []
                for i in range(ntrial):
                    print('trial {}'.format(i))
                    FPC, FPV, FSST, FVIP = vary_strength_at_twoends_run(synap_name, Con_Stre, cond)
                    
                    ALL_PC.append(FPC);ALL_PV.append(FPV)
                    ALL_SST.append(FSST);ALL_VIP.append(FVIP)
                    
                Results_PC[synap_name] = ALL_PC
                Results_PV[synap_name] = ALL_PV
                Results_SST[synap_name] = ALL_SST
                Results_VIP[synap_name] = ALL_VIP
    
    return Results_PC, Results_PV, Results_SST, Results_VIP

def vary_run(pre_i, post_j, synap_name, Con_Stre, lowerbound, upperbound, cond, interval=1):
    
    orig_synap_strength = Con_Stre[post_j,pre_i]
    #extract the lower boundary and the upper boundary
    low = lowerbound[post_j,pre_i]; up = upperbound[post_j,pre_i]
    Syn_Strength = np.arange(np.abs(low), np.abs(up)+1, interval) * np.sign(orig_synap_strength)
    
    #for store the firing rate at each synaptic strength
    FPC = []
    for syn_strength in Syn_Strength:
        print('Synapse name {}; Set synaptic strength to {:.2f}'.format(synap_name, syn_strength))
        
        #copy the connection matrix
        Con_Stre_copy = Con_Stre.copy()
        
        #change the prototype strength
        Con_Stre_copy[post_j,pre_i] = syn_strength
        
        micro_net, pcs, pvs, ssts, vips = build_model(Con_Stre_copy, cond)    
        
        #run the network
        micro_net.run(500, report=False)
        
        fpc = np.mean(pcs.mon.r_pc[-1,:])
        FPC.append(fpc)
              
        
    return FPC, Syn_Strength

def vary_syn_with_trails(cond):
    
    Con_Stre = np.array([[10., -80., -10., 0.],
                         [25, -70, -10, -8],
                         [5, -40, 0, -5],
                         [10, -25, -10, 0]])
    
    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']
    
    lowerbound = np.array([[0., -30., 0., 0.],
                    [15, -60, 0, 0],
                    [0, -30, 0, 0],
                    [0, -15, 0, 0]])
    upperbound = np.array([[20, -100, -20, 0.],
                   [35, -80, -30, -20],
                   [10, -50, 0, -10],
                   [30, -35, -20, 0]])
    
    ntrial=10; interval=10
    Results_PC = {}

    for pre_i, pre in enumerate(PreSynap):
        for post_j, post in enumerate(PostSynap):
            synap_name = post+'_'+pre
            
            if synap_name in ['pc_vip', 'sst_sst', 'vip_vip']:
                continue
            
            print('*'*50)
            
            ALL_PC = []
            
            for i in range(ntrial):
                print('trial {}'.format(i))
                FPC, Syn_Strength = vary_run(pre_i, 
                                             post_j, 
                                             synap_name, 
                                             Con_Stre, 
                                             lowerbound, 
                                             upperbound, 
                                             cond, 
                                             interval)
                ALL_PC.append(FPC) #FPC: [streng1, strenth2, ...]
                                   #ALL_PC: 嵌套的list，每一个小list表示一个trial 
            print(synap_name)    
            Results_PC[synap_name] = ALL_PC
            #----------------------
            Results_PC[synap_name+'_input'] = Syn_Strength                 
    
    #%%percetange plot
    slope_rank_plot(Results_PC, Con_Stre, lowerbound, upperbound, cond, interval)

    return Results_PC


if __name__=='__main__':
    
    Results_PC, Results_PV, Results_SST, Results_VIP = vary_one_synapse_only_start_and_end(cond='evoked', ntrial=10)
    synaptic_contribution_plot(Results_PC, celltype='pc')
    synaptic_contribution_plot(Results_PV, celltype='pv')
    synaptic_contribution_plot(Results_SST, celltype='sst')
    synaptic_contribution_plot(Results_VIP, celltype='vip')
    """
    #all synapses with many trials
    Results_PC = vary_syn_with_trails(cond='evoked')
    #slope rank plot
    """
    
    
    """
    #all synapse
    cond='spont'
    Results_PC, Results_PV, Results_SST, Results_VIP = vary_syn(cond)

    cond='evoked'
    Results_PC, Results_PV, Results_SST, Results_VIP = vary_syn(cond)
    
    #%%percetange plot
    percentage_plot(Results_PC, cond, celltype='pc')
    percentage_plot(Results_PV, cond, celltype='pv')
    percentage_plot(Results_SST, cond, celltype='sst')
    percentage_plot(Results_VIP, cond, celltype='vip')
    """
    
    """
    #specific synapse
    cond='evoked'
    ntrial=10   
    
    presyn_name='pc'; postsyn_name='vip'
    syn_name = postsyn_name+'_'+presyn_name
    Results_PC, Results_PV, Results_SST, Results_VIP, Syn_Strength = vary_one_synapse(
        presyn_name, postsyn_name, cond, ntrial, interval=2)
    slopewithshadow_plot(Results_PC, Syn_Strength, syn_name, ntrial) 

    presyn_name='pv'; postsyn_name='pc'
    syn_name = postsyn_name+'_'+presyn_name
    Results_PC, Results_PV, Results_SST, Results_VIP, Syn_Strength = vary_one_synapse(
        presyn_name, postsyn_name, cond, ntrial, interval=4)
    slopewithshadow_plot(Results_PC, Syn_Strength, syn_name, ntrial)

    presyn_name='sst'; postsyn_name='pv'
    syn_name = postsyn_name+'_'+presyn_name
    Results_PC, Results_PV, Results_SST, Results_VIP, Syn_Strength = vary_one_synapse(
        presyn_name, postsyn_name, cond, ntrial, interval=2)
    slopewithshadow_plot(Results_PC, Syn_Strength, syn_name, ntrial)

    presyn_name='sst'; postsyn_name='vip'
    syn_name = postsyn_name+'_'+presyn_name
    Results_PC, Results_PV, Results_SST, Results_VIP, Syn_Strength = vary_one_synapse(
        presyn_name, postsyn_name, cond, ntrial, interval=2)
    slopewithshadow_plot(Results_PC, Syn_Strength, syn_name, ntrial)    
    """