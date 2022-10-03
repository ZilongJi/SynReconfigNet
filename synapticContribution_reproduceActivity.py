# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:28:58 2022

Synaptic contribution. Synapse ranking by varying all except one synapse

@author: Zilong
"""

import brainpy as bp
import numpy as np
import brainpy.math as bm
from utils import SetConnectivity
from utils import plot_activity_reproduce_index
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron 

from itertools import combinations

bp.math.set_platform('cpu')
seed=1234

def build_model(Con_Stre, Con_Prob, cond, status):
    
    np.random.seed(seed)
    
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
        if status == 'MD1' or status == 'Ctrl':
            x_s =   18.0*bm.ones(num_pc)
        else:
            x_s =   16.0*bm.ones(num_pc)
            
        x_d     =   5.0*bm.ones(num_pc)
        x_i_pv  =   3.1*bm.ones(num_pv)
        x_i_sst =   2.0*bm.ones(num_sst)
        x_i_vip =   1.4*bm.ones(num_vip)
    elif cond=='Evoked':  
        if status == 'MD1' or status == 'Ctrl':
            x_s =   bm.concatenate((20.8*bm.ones(int(num_pc/4)), 13.5*bm.ones(num_pc-int(num_pc/4)))) #22.8
        else:
            x_s =   bm.concatenate((20.3*bm.ones(int(num_pc/4)), 13.0*bm.ones(num_pc-int(num_pc/4)))) #22.8     
        
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
    micro_net = bp.dyn.Network(pcs, pvs, ssts, vips)
    
    return micro_net, pcs, pvs, ssts, vips

def get_meanfr(Con_Stre, Con_Prob, cond, status):
    '''
    Get the mean firing rate of different neurons
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
    micro_net, pcs, pvs, ssts, vips = build_model(Con_Stre, Con_Prob, cond, status) 
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
    
    mean_fr = np.asarray([fpc,fpv,fsst,fvip])

    return mean_fr

def find_idx(synap_name):
    
    pre, post = synap_name.split('_')
    
    cellname = ['pc', 'pv', 'sst', 'vip']
    row = cellname.index(pre)
    column = cellname.index(post)
    
    return row, column

def main(cond, status):
    
    #Connection Probability (Data from Li Yao's experiment)
    Ctrl_Prob = np.array([[0.096,0.776,0.08,0.007],
                         [0.622,0.643,0.317,0.088],
                         [0.460,0.176,0.000,0.119],
                         [0.245,0.239,0.237,0.000]])        
    
    Ctrl_Stre = np.array([[8.,  -79.,  -8.,  0.],
                         [22., -70.,  -12., -14.],
                         [4.,  -41.,   0.,  -5.],
                         [10., -35.,  -7.,  0.]])
    
    #write down connection strength and probility
    if status=='MD1':
        #Connection Probability (Data from Li Yao's experiment)
        MD_Prob = np.array([[0.096,0.905,0.08,0.007],
                            [0.622,0.643,0.317,0.088],
                            [0.460,0.176,0.000,0.119],
                            [0.245,0.239,0.237,0.000]])  
        
        MD_Stre = np.array([[8.,  -79.,  -8.,  0.],
                            [34., -70.,  -12., -14.],
                            [4.,  -41.,  0.,   -5.],
                            [22., -35,   -7.,  0.]])  
          
    elif status=='MD4':
        
        MD_Stre = np.array([[8.,  -38.,  -8.,  0.],
                            [22., -70.,  -28., -14.],
                            [4.,  -41.,  0.,   -5.],
                            [10., -35.,  -19,  0.]])
        
        MD_Prob = np.array([[0.096,0.776,0.08,0.007],
                            [0.622,0.426,0.533,0.088],
                            [0.460,0.367,0.000,0.119],
                            [0.245,0.239,0.237,0.000]])          
    else:
        raise ValueError('choose correct status, either MD1 or MD4!')
        
        
    #get activity in control group
    Ctrl_meanfr = get_meanfr(Ctrl_Stre, Ctrl_Prob, cond, 'Ctrl')
    
    #get activity in MD group
    MD_meanfr = get_meanfr(MD_Stre, MD_Prob, cond, status)
    
    #get activity in sub-change group 
    if status == 'MD1':
        #synapList=['pc_pv', 'pv_pc', 'vip_pc']
        synapList=['pv_pc', 'vip_pc']
    elif status == 'MD4':
        #synapList=['pc_pv', 'pv_pv', 'sst_pv', 'pv_sst', 'vip_sst']
        synapList=['pc_pv',  'pv_sst', 'vip_sst']
        
    numSynapses = len(synapList)
    
    all_rp_name = []
    all_rp_index = []
    
    for r in np.arange(1,numSynapses+1,1):
        for cb in combinations(synapList, r):
            print('current combination is', cb)
            
            target_Prob = Ctrl_Prob.copy()
            target_Stre = Ctrl_Stre.copy()
            #for each element in cb, change the control value to MD value
            string = ' ' 
            for synap_name in cb:
                row, column = find_idx(synap_name)
                
                target_Stre[row, column] = MD_Stre[row, column]
                target_Prob[row, column] = MD_Prob[row, column]
                
                string += synap_name+'\n '
            #get activity based on target_Stre and target_Prob
            target_meanfr = get_meanfr(target_Stre, target_Prob, cond, status)
                
            #calculate the reproducing index
            #rp_index = (target_meanfr-Ctrl_meanfr)/(MD_meanfr-Ctrl_meanfr)
            #rp_index = (target_meanfr-MD_meanfr)/(Ctrl_meanfr-MD_meanfr)
            rp_index = target_meanfr-Ctrl_meanfr
            
            all_rp_name.append(string)
            all_rp_index.append(rp_index)
    
    #vertically stack all_rp_index
    all_rp_index = np.vstack(all_rp_index)
    
    return all_rp_name, all_rp_index

if __name__=='__main__':
    cond = 'Spont.'
    #cond = 'Evoked'
    status='MD4'
    
    all_rp_name, all_rp_index = main(cond, status)
    
    #plot
    plot_activity_reproduce_index(all_rp_name, all_rp_index, status)
    
   


    