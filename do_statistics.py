# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 19:18:16 2021

@author: Zilong

A rate-based two compartment micro-circuit model of PCs, PVs, SSTs and VIPs.

Modeling for Li Yao's work on:
Temporal Reconfiguration ofCortical Microcircuits for Neural Activity by 
Visual Deprivation

@author: Zilong Ji
Acknowledgement: Brainpy developer: Chaoming Wang
"""
import brainpy as bp
import numpy as np
from NeuronZoo import PCNeuron, PVNeuron, SSTNeuron, VIPNeuron
from utils import SetConnectivity, violoin_plot

bp.math.set_platform('cpu')

def build_model(x_s, x_d, x_i_pv, x_i_sst, x_i_vip, noise_strength, state='control'):
    #%%initialize the hyper-parameters  
    num_pc = 700; num_pv = 100; num_sst = 100; num_vip = 100
    tau_pc = 10; tau_pv = 10; tau_sst = 10; tau_vip = 10
    lambda_s = 0.31; lambda_d = 0.27
    c = 7; theta_c = 28
    theta_s = 14
    
    #total number of neurons (nparray)
    NC = np.asarray([num_pc, num_pv, num_sst, num_vip])
        
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
        #x_s = 10.0 # spontaneous: bottom up input decreased with MD 4 days 
        x_s = 16.0 # spontaneous: bottom up input decreased with MD 4 days 
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
                   W_pc_pv, W_pc_pc, W_pc_sst)   #monitors=['r_pc', 'I_0']
    pvs = PVNeuron(num_pv, tau_pv, noise_strength, x_i_pv, 
                   W_pv_pc, W_pv_pv, W_pv_sst, W_pv_vip)   #monitors=['r_pv'] 
    
    ssts = SSTNeuron(num_sst, tau_sst, noise_strength, x_i_sst, 
                     W_sst_pc, W_sst_pv, W_sst_sst, W_sst_vip)  #monitors=['r_sst']
    
    vips = VIPNeuron(num_vip, tau_vip, noise_strength, x_i_vip, 
                     W_vip_pc, W_vip_pv, W_vip_sst, W_vip_vip)  #monitors=['r_vip']
    
    pcs.PV = pvs; pcs.SST = ssts 
    pvs.PC = pcs; pvs.SST = ssts; pvs.VIP = vips
    ssts.PC = pcs; ssts.PV = pvs; ssts.VIP = vips
    vips.PC = pcs; vips.PV = pvs; vips.SST = ssts
    
    #build the network
    micro_net = bp.dyn.Network(pcs, pvs, ssts, vips)
    
    return micro_net, pcs, pvs, ssts, vips

#%%
def run_trials(n_trials, x_s, x_d, x_i_pv, x_i_sst, x_i_vip, noise_strength, state):
    PC_Sam = []; PV_Sam = []; SST_Sam=[]; VIP_Sam=[]
    for i in range(n_trials):
        bp.base.clear_name_cache()
        print('simulating trail {:.0f}'.format(i)) 
        micro_net, pcs, pvs, ssts, vips = build_model(x_s, x_d, x_i_pv, x_i_sst, 
                                                      x_i_vip, noise_strength, state)  
        #reset the firing rates of different cell types
        pcs.r_pc[:] = 0.; pvs.r_pv[:] = 0.; ssts.r_sst[:] = 0.; vips.r_vip[:] = 0.  
        runner = bp.dyn.DSRunner(micro_net, 
                                 monitors=['PC.r_pc', 'PC.I_0',
                                           'PV.r_pv', 'SST.r_sst',
                                           'VIP.r_vip'],
                                 dt=0.1)
        runner.run(duration=1000)
        
        #for each trial, random sampling 5 neurons
        idx = np.random.choice(700, 1, replace=False)
        pc_samples = runner.mon['PC.r_pc'][-1,idx]; PC_Sam.append(pc_samples)
        
        idx = np.random.choice(100, 1, replace=False)
        pv_samples = runner.mon['PV.r_pv'][-1,idx]; PV_Sam.append(pv_samples)
        # pv_samples = pvs.mon.r_pv[-1,idx]; PV_Sam.append(pv_samples)

        idx = np.random.choice(100, 1, replace=False)
        sst_samples = runner.mon['SST.r_sst'][-1,idx]; SST_Sam.append(sst_samples)
        # sst_samples = ssts.mon.r_sst[-1,idx]; SST_Sam.append(sst_samples)

        idx = np.random.choice(100, 1, replace=False)
        vip_samples = runner.mon['VIP.r_vip'][-1,idx]; VIP_Sam.append(vip_samples)
        # vip_samples = vips.mon.r_vip[-1,idx]; VIP_Sam.append(vip_samples)

    PC_Samples = np.concatenate(PC_Sam); PV_Samples = np.concatenate(PV_Sam)
    SST_Samples= np.concatenate(SST_Sam); VIP_Samples = np.concatenate(VIP_Sam)
    return PC_Samples, PV_Samples, SST_Samples, VIP_Samples

#%% do statistics
def do_stats(cond):
    
    if cond == 'Spont.':
        n_trials = 20; noise_strength = 3. #noise level
        x_s = 14; x_d = 14
        x_i_pv = 4; x_i_sst=3; x_i_vip=2
        print('Modeling Spontaneous...') 
    elif cond == 'Evoked':
        n_trials = 1; noise_strength = 0. #noise level
        x_s = 18; x_d = 14
        x_i_pv = 6; x_i_sst=3; x_i_vip=2
        print('Modeling Evoked...') 
    else:
        raise ValueError("Wrong Condition Name, Check It.")
        
    PC_Samples_ctrl1, PV_Samples_ctrl1, SST_Samples_ctrl1, VIP_Samples_ctrl1 \
        = run_trials(n_trials, x_s, x_d, x_i_pv, x_i_sst, x_i_vip, noise_strength, state='control')
    PC_Samples_md1, PV_Samples_md1, SST_Samples_md1, VIP_Samples_md1 \
        = run_trials(n_trials, x_s, x_d, x_i_pv, x_i_sst, x_i_vip, noise_strength, state='md1')
    PC_Samples_ctrl2, PV_Samples_ctrl2, SST_Samples_ctrl2, VIP_Samples_ctrl2 \
        = run_trials(n_trials, x_s, x_d, x_i_pv, x_i_sst, x_i_vip, noise_strength, state='control')
    PC_Samples_md4, PV_Samples_md4, SST_Samples_md4, VIP_Samples_md4 \
        = run_trials(n_trials, x_s, x_d, x_i_pv, x_i_sst, x_i_vip, noise_strength, state='md4')
    
    # ttest on PCs ctrl vs. md1 & ctrl vs. md4
    violoin_plot(PC_Samples_ctrl1, PC_Samples_md1, PC_Samples_ctrl2, 
               PC_Samples_md4, cond, celltype='PC')
    
    violoin_plot(PV_Samples_ctrl1, PV_Samples_md1, PV_Samples_ctrl2, 
               PV_Samples_md4, cond, celltype='PV')
    
    violoin_plot(SST_Samples_ctrl1, SST_Samples_md1, SST_Samples_ctrl2, 
               SST_Samples_md4, cond, celltype='SST')
    
    violoin_plot(VIP_Samples_ctrl1, VIP_Samples_md1, VIP_Samples_ctrl2, 
               VIP_Samples_md4, cond, celltype='VIP')

if __name__=='__main__':
    #do_stats(cond = 'Spont.')
    do_stats(cond = 'Evoked')