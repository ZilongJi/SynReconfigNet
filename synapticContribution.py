# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 09:26:18 2022

Synaptic contribution (synapse ranking)

@author: Zilong
"""

import brainpy as bp
import numpy as np
import brainpy.math as bm
from utils import SetConnectivity, synaptic_contribution_plot, normed_synaptic_contribution_plot
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron 

bp.math.set_platform('cpu')
seed=1234
np.random.seed(seed)

def build_model(Con_Stre, Con_Prob, cond):
    #%%initialize neuron numbers, time constant, etc  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
    lambda_s = 0.31; lambda_d = 0.27
    c = 7; theta_c = 28
    theta_s = 14
    noise_strength = 0
    
    #bottom-up input and top-down input
    if cond=='Spont.':
        # homogenous input 
        x_s     =   18.8*bm.ones(num_pc)
        x_d     =   10.0*bm.ones(num_pc)
        x_i_pv  =   3.1*bm.ones(num_pv)
        x_i_sst =   1.9*bm.ones(num_sst)
        x_i_vip =   1.4*bm.ones(num_vip)
    elif cond=='Evoked':  
        # heterogeneous input
        x_s     =   bm.concatenate((30.0*bm.ones(int(num_pc/4)), 20.4*bm.ones(num_pc-int(num_pc/4)))) #22.8
        x_d     =   10*bm.ones(num_pc)
        
        x_i_pv  =   bm.concatenate((10.0*bm.ones(int(num_pv/4)), 6.1*bm.ones(num_pv-int(num_pv/4))))  #7.1
        x_i_sst =   bm.concatenate((6.0*bm.ones(int(num_sst/4)), 2.4*bm.ones(num_sst-int(num_sst/4))))  #3.3
        x_i_vip =   bm.concatenate((5.0*bm.ones(int(num_vip/4)), 2.1*bm.ones(num_vip-int(num_vip/4))))  #2.8   
    else:
        raise ValueError('Choose correct condition!')       
    
    #total number of neurons (nparray)
    NC = np.array([num_pc, num_pv, num_sst, num_vip])
        
    #Normalize the Connection Strength 
    Con_Stre = Con_Stre/79.0

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
    pcs = PCNeuron(num_pc, tau_pc, noise_strength, lambda_s, lambda_d, 
                   x_s, x_d, c, theta_s, theta_c, 
                   W_pc_pv, W_pc_pc, W_pc_sst, seed)   #monitors=['r_pc', 'I_0']
    pvs = PVNeuron(num_pv, tau_pv, noise_strength, x_i_pv, 
                   W_pv_pc, W_pv_pv, W_pv_sst, W_pv_vip, seed)   #monitors=['r_pv'] 
    
    ssts = SSTNeuron(num_sst, tau_sst, noise_strength, x_i_sst, 
                     W_sst_pc, W_sst_pv, W_sst_sst, W_sst_vip, seed)  #monitors=['r_sst']
    
    vips = VIPNeuron(num_vip, tau_vip, noise_strength, x_i_vip, 
                     W_vip_pc, W_vip_pv, W_vip_sst, W_vip_vip, seed)  #monitors=['r_vip']
    
    pcs.PV = pvs; pcs.SST = ssts 
    pvs.PC = pcs; pvs.SST = ssts; pvs.VIP = vips
    ssts.PC = pcs; ssts.PV = pvs; ssts.VIP = vips
    vips.PC = pcs; vips.PV = pvs; vips.SST = ssts
    
    #build the network
    micro_net = bp.dyn.Network(pcs, pvs, ssts, vips)
    
    return micro_net, pcs, pvs, ssts, vips

def get_mean_fr(Con_Stre, Con_Prob, cond):
    '''
    Get the mean firing rate of different neurons by running the built model
    Args:
        Con_Stre: connection strength of the synapse
        Con_Prob: connection probability of the synapse
        cond: 'Spont.' or 'Evoked'
    Output:
        the mean firing rate of 4 types of neurons: fpc, fpv, fsst, fvip
        Note: calculate the mean firing rate of the neurons with preferred stimulus
    '''
    bp.base.clear_name_cache()
    print('simulating trail...')
    micro_net, pcs, pvs, ssts, vips = build_model(Con_Stre, Con_Prob, cond) 
    #reset the firing rates of different cell types
    pcs.r_pc[:] = 0.; pvs.r_pv[:] = 0.; ssts.r_sst[:] = 0.; vips.r_vip[:] = 0. 
    runner = bp.dyn.DSRunner(micro_net,
                             monitors=['PC.r_pc', 'PC.I_0',
                                       'PV.r_pv', 'SST.r_sst',
                                       'VIP.r_vip'],
                             dt=0.1,
                             numpy_mon_after_run=False,
                             progress_bar=True)
    
    runner.run(duration=1000)    
    fpc = np.mean(runner.mon['PC.r_pc'][-1,:int(pcs.size/4)])  #int(pcs.size/4) calculate the mean fr for the prefered neurons
    fpv = np.mean(runner.mon['PV.r_pv'][-1,:int(pvs.size/4)])
    fsst = np.mean(runner.mon['SST.r_sst'][-1,:int(ssts.size/4)])
    fvip = np.mean(runner.mon['VIP.r_vip'][-1,:int(vips.size/4)])
    
    return fpc, fpv, fsst, fvip

def generate_strength_prob_matrix(status):
    '''
    Generate the strength matrix and the probability matrix
    Args:
        status: 'MD1' or 'MD4'
    Output:
        a cell array with each element the changed synaptic strength matrix 
        and probability matrix
    '''
    Strength = {}
    Probability = {}
    
    Ctrl_Stre = np.array([[8.,  -79.,  -8.,  0.],
                          [22., -70.,  -12., -14.],
                          [4.,  -41.,   0.,  -5.],
                          [10., -35.,  -7.,  0.]])
    
    Ctrl_Prob = np.array([[0.096,0.776,0.08,0.007],
                          [0.622,0.643,0.317,0.088],
                          [0.460,0.176,0.000,0.119],
                          [0.245,0.239,0.237,0.000]])   
    
    Strength['Ctrl'] = Ctrl_Stre
    Probability['Ctrl'] = Ctrl_Prob
    
    if status=='MD1':
        #pc_pv
        PcPv_Stre = Ctrl_Stre.copy()
        PcPv_Prob = Ctrl_Prob.copy(); PcPv_Prob[0,1] = 0.905
        
        Strength['pc_pv'] = PcPv_Stre
        Probability['pc_pv'] = PcPv_Prob
        
        #pv_pc
        PvPc_Stre = Ctrl_Stre.copy(); PvPc_Stre[1,0] = 34
        PvPc_Prob = Ctrl_Prob.copy()
        
        Strength['pv_pc'] = PvPc_Stre
        Probability['pv_pc'] = PvPc_Prob   
        
        #vip_pc
        VipPc_Stre = Ctrl_Stre.copy(); VipPc_Stre[3,0] = 22
        VipPc_Prob = Ctrl_Prob.copy()
        
        Strength['vip_pc'] = VipPc_Stre
        Probability['vip_pc'] = VipPc_Prob
    else: #MD4
        #pc_pv
        PcPv_Stre = Ctrl_Stre.copy(); PcPv_Stre[0,1] = -38
        PcPv_Prob = Ctrl_Prob.copy();       
        
        Strength['pc_pv'] = PcPv_Stre
        Probability['pc_pv'] = PcPv_Prob    
        
        #pv_pv
        PvPv_Stre = Ctrl_Stre.copy()
        PvPv_Prob = Ctrl_Prob.copy(); PvPv_Prob[1,1] = 0.426
        
        Strength['pv_pv'] = PvPv_Stre
        Probability['pv_pv'] = PvPv_Prob 

        #sst_pv
        SstPv_Stre = Ctrl_Stre.copy()
        SstPv_Prob = Ctrl_Prob.copy(); SstPv_Prob[2,1] = 0.367
        
        Strength['sst_pv'] = SstPv_Stre
        Probability['sst_pv'] = SstPv_Prob 
        
        #pv_sst
        PvSst_Stre = Ctrl_Stre.copy(); PvSst_Stre[1,2] = -28
        PvSst_Prob =  Ctrl_Prob.copy(); PvSst_Prob[1,2] = 0.533

        Strength['pv_sst'] = PvSst_Stre
        Probability['pv_sst'] = PvSst_Prob 
        
        #vip_sst
        VipSst_Stre = Ctrl_Stre.copy(); VipSst_Stre[3,2] = -19
        VipSst_Prob =  Ctrl_Prob.copy()
        
        Strength['vip_sst'] = VipSst_Stre
        Probability['vip_sst'] = VipSst_Prob

    return Strength, Probability

def vary_per_synapse(cond, status, ntrial=10):

    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']

    if status == 'MD1':
        synapList = ['pc_pv', 'pv_pc', 'vip_pc']
    else: #MD4
        synapList = ['pc_pv', 'pv_pv', 'sst_pv', 'pv_sst', 'vip_sst']
        
    Strength, Probability = generate_strength_prob_matrix(status)

    Results_PC = {}; Results_PV = {}
    Results_SST = {}; Results_VIP = {}
    
    for pre_i, pre in enumerate(PreSynap):
        for post_j, post in enumerate(PostSynap):    
            
            synap_name = post+'_'+pre
            print('simulating synapse '+synap_name)
            
            if synap_name in synapList:
                ALL_PC = []; ALL_PV = []; ALL_SST = []; ALL_VIP = []
                for i in range(ntrial):
                    print('trial {} control run'.format(i))
                    #get the mean firing rate in control group
                    Ctrl_Stre = Strength['Ctrl']
                    Ctrl_Prob = Probability['Ctrl']
                    Ctrl_fpc, Ctrl_fpv, Ctrl_fsst, Ctrl_fvip =  get_mean_fr(Ctrl_Stre, Ctrl_Prob, cond)
                    
                    print('trial {} target run'.format(i))
                    #vary one synapse and get the mean firing rate
                    target_Stre = Strength[synap_name]
                    target_Prob = Probability[synap_name]
                    target_fpc, target_fpv, target_fsst, target_fvip = get_mean_fr(target_Stre, target_Prob, cond)
                    
                    ALL_PC.append([Ctrl_fpc,target_fpc])
                    ALL_PV.append([Ctrl_fpv,target_fpv])
                    ALL_SST.append([Ctrl_fsst,target_fsst])
                    ALL_VIP.append([Ctrl_fvip,target_fvip])
                    
                Results_PC[synap_name] = ALL_PC
                Results_PV[synap_name] = ALL_PV
                Results_SST[synap_name] = ALL_SST
                Results_VIP[synap_name] = ALL_VIP
    
    return Results_PC, Results_PV, Results_SST, Results_VIP

if __name__=='__main__':
    cond = 'Spont.'
    status='MD4'
    Results_PC, Results_PV, Results_SST, Results_VIP = vary_per_synapse(cond, status, 1) #number of trials
    #%%
    normed_synaptic_contribution_plot(Results_PC, cond, status, celltype='pc')
    normed_synaptic_contribution_plot(Results_PV, cond, status, celltype='pv')
    normed_synaptic_contribution_plot(Results_SST, cond, status, celltype='sst')
    normed_synaptic_contribution_plot(Results_VIP, cond, status, celltype='vip')

    
