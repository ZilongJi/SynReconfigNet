# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:27:33 2022

@author: zji

Modeling for Li Yao's work on:
Temporal Reconfiguration ofCortical Microcircuits for Neural Activity by 
Visual Deprivation

@author: Zilong Ji
Acknowledgement: Brainpy developer: Chaoming Wang
"""
import brainpy as bp
import numpy as np
import brainpy.math as bm
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron
from utils import SetConnectivity, violoin_plot, trial_plot_pc

bp.math.set_platform('cpu')
rngseed=123
np.random.seed(rngseed)

def build_model(noise_strength, state='control', cond='Spont.'):
    #%%initialize neuron numbers, time constant, etc  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
    lambda_s = 0.31; lambda_d = 0.27
    c = 7; theta_c = 28
    theta_s = 14
    seed=np.random.randint(1000)
    
    #bottom-up input and top-down input
    if cond=='Spont.':
        # homogenous input 

        x_s     =   18.0*bm.ones(num_pc)
        x_d     =   5.0*bm.ones(num_pc)
        x_i_pv  =   3.1*bm.ones(num_pv)
        x_i_sst =   2.0*bm.ones(num_sst)
        x_i_vip =   1.4*bm.ones(num_vip)

    elif cond=='Evoked':  
        # heterogeneous input
        x_s     =   bm.concatenate((20.8*bm.ones(int(num_pc/4)), 13.5*bm.ones(num_pc-int(num_pc/4)))) #22.8
        x_d     =   5*bm.ones(num_pc)
        
        x_i_pv  =   bm.concatenate((4.0*bm.ones(int(num_pv/4)), 1.2*bm.ones(num_pv-int(num_pv/4))))  #7.1
        x_i_sst =   bm.concatenate((3.0*bm.ones(int(num_sst/4)), 0.6*bm.ones(num_sst-int(num_sst/4))))  #3.3
        x_i_vip =   bm.concatenate((2.0*bm.ones(int(num_vip/4)), 0.1*bm.ones(num_vip-int(num_vip/4))))  #2.8   
    else:
        raise ValueError('Choose correct condition!')       
    
    #total number of neurons (nparray)
    NC = np.array([num_pc, num_pv, num_sst, num_vip])
        
    #Connection Strength (Data from Li Yao's experiment)
    if state == 'control':
        #Connection Probability (Data from Li Yao's experiment)
        Con_Prob = np.array([[0.096,0.776,0.08,0.007],
                             [0.622,0.643,0.317,0.088],
                             [0.460,0.176,0.000,0.119],
                             [0.245,0.239,0.237,0.000]])        
        
        Con_Stre = np.array([[8.,  -79.,  -8.,  0.],
                             [22., -70.,  -12., -14.],
                             [4.,  -41.,   0.,  -5.],
                             [10., -35.,  -7.,  0.]])
        #normalize the synaptic strength for stability
        Con_Stre = Con_Stre/79.0 
    elif state == 'md1':
        #Connection Probability (Data from Li Yao's experiment)
        Con_Prob = np.array([[0.096,0.905,0.08,0.007],
                             [0.622,0.643,0.317,0.088],
                             [0.460,0.176,0.000,0.119],
                             [0.245,0.239,0.237,0.000]])     
        
        Con_Stre = np.array([[8.,  -79.,  -8.,  0.],
                             [34., -70.,  -12., -14.],
                             [4.,  -41.,  0.,   -5.],
                             [22., -35,   -7.,  0.]])
        #normalize the synaptic strength for stability
        Con_Stre = Con_Stre/79.0
    elif state == 'md4':
        if cond=='Spont.':
            x_s = 16.0
            #aaa=1
            #x_s = 19.0*bm.ones(num_pc) # spontaneous: bottom up input decreased with MD 4 days 
        elif cond=='Evoked':
            #x_s = 22.8
            #x_s = 20.7*bm.ones(num_pc) # evoked: bottom up input decreased with MD 4 days 
            x_s     =   bm.concatenate((20.3*bm.ones(int(num_pc/4)), 13.0*bm.ones(num_pc-int(num_pc/4)))) #22.8     
            #aaa=1
        else:
            raise ValueError('Choose correct condition!')
        #Connection Probability (Data from Li Yao's experiment)
        Con_Prob = np.array([[0.096,0.776,0.08,0.007],
                             [0.622,0.426,0.533,0.088],
                             [0.460,0.367,0.000,0.119],
                             [0.245,0.239,0.237,0.000]])    
        
        Con_Stre = np.array([[8.,  -38.,  -8.,  0.],
                             [22., -70.,  -28., -14.],
                             [4.,  -41.,  0.,   -5.],
                             [10., -35.,  -19,  0.]])
        #normalize the synaptic strength for stability
        Con_Stre = Con_Stre/79.0
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

#%% do statistics
def visualize_dynamics(cond):
    state='control'
    noise_strength = 0.5
    bp.base.clear_name_cache()
    
    micro_net, pcs, pvs, ssts, vips = build_model(noise_strength, state, cond)

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
    
    #for each trial, random sampling 5 neurons
    pcs = runner.mon['PC.r_pc']
    
    trial_plot_pc(state, pcs, numsamples=5)  
    '''
    idx = np.random.choice(int(pvs.size/4), n_cells, replace=False)
    pv_samples = runner.mon['PV.r_pv'][-1,idx]

    idx = np.random.choice(int(ssts.size/4), n_cells, replace=False)
    sst_samples = runner.mon['SST.r_sst'][-1,idx]

    idx = np.random.choice(int(vips.size/4), n_cells, replace=False)
    vip_samples = runner.mon['VIP.r_vip'][-1,idx]
    '''

if __name__=='__main__':
    visualize_dynamics(cond = 'Spont.')