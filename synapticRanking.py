# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 14:49:34 2022

Synapse rankingof all synapse, each time vary one synapse

@author: Zilong
"""

import brainpy as bp
import numpy as np
import brainpy.math as bm
from utils import SetConnectivity
from utils import synaptic_ranking_plot
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron 

bp.math.set_platform('cpu')
seed=1234
np.random.seed(seed)

#%%
def build_model(Con_Stre, Con_Prob, cond):
    #%%initialize neuron numbers, time constant, etc  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
    lambda_s = 0.31; lambda_d = 0.27
    c = 7; theta_c = 28
    theta_s = 14
    noise_strength = 0.5
    
    #bottom-up input and top-down input
    if cond=='Spont.':
        # homogenous input 
        x_s     =   18.0*bm.ones(num_pc)
        x_d     =   5.0*bm.ones(num_pc)
        x_i_pv  =   3.1*bm.ones(num_pv)
        x_i_sst =   2.0*bm.ones(num_sst)
        x_i_vip =   1.4*bm.ones(num_vip)
    elif cond=='Evoked':  
        x_s     =   bm.concatenate((20.8*bm.ones(int(num_pc/4)), 13.5*bm.ones(num_pc-int(num_pc/4)))) #22.8
        x_d     =   5*bm.ones(num_pc)
        
        x_i_pv  =   bm.concatenate((4.0*bm.ones(int(num_pv/4)), 1.2*bm.ones(num_pv-int(num_pv/4))))  #7.1
        x_i_sst =   bm.concatenate((3.0*bm.ones(int(num_sst/4)), 0.6*bm.ones(num_sst-int(num_sst/4))))  #3.3
        x_i_vip =   bm.concatenate((2.0*bm.ones(int(num_vip/4)), 0.1*bm.ones(num_vip-int(num_vip/4))))  #2.8  
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
    micro_net = bp.Network(pcs, pvs, ssts, vips)
    
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
    bp.math.clear_name_cache()
    print('simulating trail...')
    micro_net, pcs, pvs, ssts, vips = build_model(Con_Stre, Con_Prob, cond) 
    #reset the firing rates of different cell types
    pcs.r_pc[:] = 0.; pvs.r_pv[:] = 0.; ssts.r_sst[:] = 0.; vips.r_vip[:] = 0. 
    runner = bp.DSRunner(micro_net,
                             monitors=['PC.r_pc', 'PC.I_0',
                                       'PV.r_pv', 'SST.r_sst',
                                       'VIP.r_vip'],
                             dt=0.1,
                             numpy_mon_after_run=False,
                             progress_bar=True)
    
    runner.run(duration=1000)    
    
    #concatenate
    fr_vector = np.concatenate([
        runner.mon['PC.r_pc'][-1,:int(pcs.size[0]/4)],
        runner.mon['PV.r_pv'][-1,:int(pvs.size[0]/4)],
        runner.mon['SST.r_sst'][-1,:int(ssts.size[0]/4)],
        runner.mon['VIP.r_vip'][-1,:int(vips.size[0]/4)]
        ])
    
    return fr_vector

def main(cond, delta_stre = 10, ntrial=10):

    #Connection Probability (Data from Li Yao's experiment)
    Ctrl_Prob = np.array([[0.096,0.776,0.08,0.007],
                         [0.622,0.643,0.317,0.088],
                         [0.460,0.176,0.000,0.119],
                         [0.245,0.239,0.237,0.000]])        
    
    Ctrl_Stre = np.array([[8.,  -79.,  -8.,  0.],
                         [22., -70.,  -12., -14.],
                         [4.,  -41.,   0.,  -5.],
                         [10., -35.,  -7.,  0.]])
    
    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']

    CorrCoef = np.zeros((13, ntrial))
    DiffPerChange = np.zeros((13, ntrial))  
    
    SynapName = []
    
    synap_idx = 0
    
    for pre_i, pre in enumerate(PreSynap):
        for post_j, post in enumerate(PostSynap): 
            
            synap_name = post+'_'+pre
            
            if synap_name in ['pc_vip', 'sst_sst', 'vip_vip']:
                continue
            
            synap_name = pre+'_'+post
            #capitalisze
            synap_name = synap_name.upper()
            SynapName.append(synap_name)
            print('simulating synapse '+synap_name)
 
            Target_Prob = Ctrl_Prob.copy()
            Target_Stre = Ctrl_Stre.copy()
            
            stre = Target_Stre[post_j, pre_i]
            if delta_stre<1:
                #increase of a percenatge
                new_stre = stre+delta_stre*np.abs(stre)*np.sign(stre)
            else:
                #absolute increase
                new_stre = stre+delta_stre*np.sign(stre)
                
            Target_Stre[post_j, pre_i] = new_stre

            for i in range(ntrial):
                #get the mean firing rate in control group
                Ctrl_frvector =  get_mean_fr(Ctrl_Stre, Ctrl_Prob, cond)
                
                #get the mean firing rate in target group
                Target_frvector = get_mean_fr(Target_Stre, Target_Prob, cond)

                cc = np.corrcoef(Ctrl_frvector,Target_frvector)[0,1]
                
                CorrCoef[synap_idx, i] = cc
                if delta_stre<1:
                    #increase of a percenatge
                    DiffPerChange[synap_idx, i] = (1-cc)/(delta_stre*np.abs(stre))
                else:
                    #absolute increase
                    DiffPerChange[synap_idx, i] = (1-cc)/delta_stre
            
            synap_idx += 1
            
    return  CorrCoef, DiffPerChange, SynapName

if __name__=='__main__':
    cond = 'Spont.'
    # cond = 'Evoked'
    
    #absolute increase
    #CorrCoef, DiffPerChange, SynapName = main(cond, delta_stre = 5, ntrial= 5) #number of trials
    
    #increase a percetage
    CorrCoef, DiffPerChange, SynapName = main(cond, delta_stre = 0.1, ntrial= 3) #number of trials
    
    #%% 16 synapse ranking plot
    synaptic_ranking_plot(DiffPerChange, SynapName, cond)


    