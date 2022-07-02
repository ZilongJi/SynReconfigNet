# -*- coding: utf-8 -*-
"""
A rate-based two compartment micro-circuit model of PCs, PVs, SSTs and VIPs.

Modeling for Li Yao's work on:
Temporal Reconfiguration ofCortical Microcircuits for Neural Activity by 
Visual Deprivation

Created on Sun Aug 15 15:39:34 2021

@author: Zilong Ji
Acknowledgement: Brainpy developer: Chaoming Wang
"""
import brainpy.math as bm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def SetConnectivity(Con_Prob, Con_Stre, NC):
    '''
    Con_Prob: connection probability of all cell types
    Con_Stre: connection strength between two cells
    NC: nparray of the number of PCs, PVs, SSTs, VIPs
    '''
    #
    NCon = np.round(Con_Prob*NC).astype(np.int)
    
    NameList = np.asarray([['pc_pc', 'pc_pv', 'pc_sst', 'pc_vip'],
                      ['pv_pc', 'pv_pv', 'pv_sst', 'pv_vip'],
                      ['sst_pc', 'sst_pv', 'sst_sst', 'sst_vip'],
                      ['vip_pc', 'vip_pv', 'vip_sst', 'vip_vip']])
    
    WeightDic = {} #store all the weight information
    
    for i in range(16):
        m,n = np.unravel_index(i,(4,4))
        weightname = NameList[m,n]
        Mtx = bm.zeros((NC[m], NC[n]))
        if NCon[m,n]>0: #if there are connections, go to the next step
            if m==n: #connection between the neurons in same type, omit the autapse
                for l in range(NC[m]):
                    weight = Con_Stre[m,n]*bm.array([0] * (NC[n]-1-NCon[m,n]) + [1] * NCon[m,n])/NCon[m,n]
                    weight = bm.asarray(weight)
                    bm.random.shuffle(weight)
                    weight = bm.insert(weight,l,0)
                    Mtx[l,:] = weight
            else: #connection between the neurons in different types
                for l in range(NC[m]):
                    weight = Con_Stre[m,n]*bm.array([0] * (NC[n]-NCon[m,n]) + [1] * NCon[m,n])/NCon[m,n]
                    weight = bm.asarray(weight)
                    bm.random.shuffle(weight)
                    Mtx[l,:] = weight
        WeightDic[weightname] = Mtx
    
    return WeightDic

def grid_plot(FPC, FPV, FSST, FVIP, XS, XD, name):
    """
    imshow the mean firing rate (or differences between conditions) of each cell type
    under different values of top-down and bottom-up
    """
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    
    pos0 = axs[0].imshow(FPC, cmap='Blues', aspect='auto', vmin=0,\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[0].set_xlabel('bottom-up stregth'); axs[0].set_ylabel('top-down strength')
    fig.colorbar(pos0, ax=axs[0])
    
    pos1 = axs[1].imshow(FPV, cmap='Blues', aspect='auto', vmin=0,\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[1].set_xlabel('bottom-up stregth'); axs[1].set_ylabel('top-down strength')
    fig.colorbar(pos1, ax=axs[1]) 
    
    pos2 = axs[2].imshow(FSST, cmap='Blues', aspect='auto', vmin=0,\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[2].set_xlabel('bottom-up stregth'); axs[2].set_ylabel('top-down strength') 
    fig.colorbar(pos2, ax=axs[2])
    
    pos3 = axs[3].imshow(FVIP, cmap='Blues', aspect='auto', vmin=0,\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[3].set_xlabel('bottom-up stregth'); axs[3].set_ylabel('top-down strength') 
    fig.colorbar(pos3, ax=axs[3]) 
    
    plt.tight_layout()   
    
    plt.savefig('./figures/'+name+'.png')   
    plt.savefig('./figures/EPS/'+name+'.eps') 

def grid_plot_diff(FPC_diff, FPV_diff, FSST_diff, FVIP_diff, XS, XD, name):
    """
    imshow the mean firing rate (or differences between conditions) of each cell type
    under different values of top-down and bottom-up
    """
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    
    pos0 = axs[0].imshow(FPC_diff, cmap='RdBu', aspect='auto', vmin=-np.max(FPC_diff),\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[0].set_xlabel('bottom-up stregth'); axs[0].set_ylabel('top-down strength')
    fig.colorbar(pos0, ax=axs[0])
    
    pos1 = axs[1].imshow(FPV_diff, cmap='RdBu', aspect='auto', vmin=-np.max(FPV_diff),\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[1].set_xlabel('bottom-up stregth'); axs[1].set_ylabel('top-down strength')
    fig.colorbar(pos1, ax=axs[1]) 
    
    pos2 = axs[2].imshow(FSST_diff, cmap='RdBu', aspect='auto', vmin=-np.max(FSST_diff),\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[2].set_xlabel('bottom-up stregth'); axs[2].set_ylabel('top-down strength') 
    fig.colorbar(pos2, ax=axs[2])
    
    pos3 = axs[3].imshow(FVIP_diff, cmap='RdBu', aspect='auto', vmin=-np.max(FVIP_diff),\
                         extent=[min(XS),max(XS),max(XD),min(XD)]); 
    axs[3].set_xlabel('bottom-up stregth'); axs[3].set_ylabel('top-down strength') 
    fig.colorbar(pos3, ax=axs[3]) 
    
    plt.tight_layout()   
    
    plt.savefig('./figures/'+name+'.png')
    plt.savefig('./figures/EPS/'+name+'.eps') 

def trial_plot(name, pcs, pvs, ssts, vips, numsamples=20):
    """
    plot some trial of the simulation process 
    """
    fig, axs = plt.subplots(2, 4, figsize=(10,4))
    
    #PC
    for k in range(numsamples):
        idx = np.random.choice(700, 1)[0]
        axs[0,0].plot(pcs.mon.r_pc[::10,idx], alpha=0.1, color='k')  
    axs[0,0].set_xlabel('Simulation time (ms)');
    axs[0,0].set_ylabel('Firing rate')
    axs[0,0].plot(np.mean(pcs.mon.r_pc[::10,:], axis=1), color='r') 
    
    axs[1,0].hist(pcs.mon.r_pc[-1,:], bins=20, facecolor='k', alpha=0.5, density=True)
    axs[1,0].set_xlabel('Final firing rate')
    axs[1,0].set_ylabel('frequency distribution')
    axs[1,0].set_title('mean firing rate {:.2f}'.format(np.mean(pcs.mon.r_pc[-1,:])))
    
    #PV
    for k in range(numsamples):
        idx = np.random.choice(100, 1)[0]
        axs[0,1].plot(pvs.mon.r_pv[::10,idx], alpha=0.1, color='k')  
    axs[0,1].set_xlabel('Simulation time (ms)')
    axs[0,1].set_ylabel('Firing rate')
    axs[0,1].plot(np.mean(pvs.mon.r_pv[::10,:], axis=1), color='r') 
    
    axs[1,1].hist(pvs.mon.r_pv[-1,:], bins=20, facecolor='k', alpha=0.5, density=True)
    axs[1,1].set_xlabel('Final firing rate')
    axs[1,1].set_ylabel('frequency distribution')
    axs[1,1].set_title('mean firing rate {:.2f}'.format(np.mean(pvs.mon.r_pv[-1,:])))    
    
    #SST
    for k in range(numsamples):
        idx = np.random.choice(100, 1)[0]
        axs[0,2].plot(ssts.mon.r_sst[::10,idx], alpha=0.1, color='k')  
    axs[0,2].set_xlabel('Simulation time (ms)');
    axs[0,2].set_ylabel('Firing rate')
    axs[0,2].plot(np.mean(ssts.mon.r_sst[::10,:], axis=1), color='r') 
    
    axs[1,2].hist(ssts.mon.r_sst[-1,:], bins=20, facecolor='k', alpha=0.5, density=True)
    axs[1,2].set_xlabel('Final firing rate')
    axs[1,2].set_ylabel('frequency distribution')
    axs[1,2].set_title('mean firing rate {:.2f}'.format(np.mean(ssts.mon.r_sst[-1,:])))        
    
    #VIP
    for k in range(numsamples):
        idx = np.random.choice(100, 1)[0]
        axs[0,3].plot(vips.mon.r_vip[::10,idx], alpha=0.1, color='k')  
    axs[0,3].set_xlabel('Simulation time (ms)');
    axs[0,3].set_ylabel('Firing rate')
    axs[0,3].plot(np.mean(vips.mon.r_vip[::10,:], axis=1), color='r') 
    
    axs[1,3].hist(vips.mon.r_vip[-1,:], bins=20, facecolor='k', alpha=0.5, density=True)
    axs[1,3].set_xlabel('Final firing rate')
    axs[1,3].set_ylabel('frequency distribution')
    axs[1,3].set_title('mean firing rate {:.2f}'.format(np.mean(vips.mon.r_vip[-1,:])))    
    
    plt.tight_layout()
    plt.savefig('./figures/'+name+'.png')
    plt.savefig('./figures/EPS/'+name+'.eps')

def trial_plot_pc(name, pcs, numsamples=20):
    """
    plot some trial of the simulation process 
    """
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    
    #PC
    for k in range(numsamples):
        idx = np.random.choice(700, 1)[0]
        axs[0].plot(pcs.mon.r_pc[:,idx], alpha=0.1, color='k')  
    axs[0].set_xlabel('Simulation time (ms)');
    axs[0].set_ylabel('Firing rate')
    axs[0].plot(np.mean(pcs.mon.r_pc[:,:], axis=1), color='r') 
    
    axs[1].hist(pcs.mon.r_pc[-1,:], bins=20, facecolor='k', alpha=0.5, density=True)
    axs[1].set_xlabel('Final firing rate')
    axs[1].set_ylabel('frequency distribution')
    axs[1].set_title('mean firing rate {:.2f}'.format(np.mean(pcs.mon.r_pc[-1,:])))
    
    plt.tight_layout()
    plt.savefig('./figures/'+name+'_pc.png')
    plt.savefig('./figures/EPS/'+name+'_pc.eps')

def violoin_plot(ctrl1, md1, ctrl2, md4, cond, celltype):
    """
    violinplot with seaborn
    """
    ctrl1 = bm.as_numpy(ctrl1).flatten()
    md1 = bm.as_numpy(md1).flatten()
    ctrl2 = bm.as_numpy(ctrl2).flatten()
    md4 = bm.as_numpy(md4).flatten()

    #regroup the data into dataframe
    data = np.concatenate(
        [[ctrl1, len(ctrl1)*['control vs. md1'], len(ctrl1)*['control']],
         [md1, len(md1)*['control vs. md1'], len(md1)*['md']],
         [ctrl2, len(ctrl2)*['control vs. md4'], len(ctrl2)*['control']],
         [md4, len(md4)*['control vs. md4'], len(md4)*['md']]],
        axis=1)
    
    df = pd.DataFrame(columns=['value', 'comparison', 'state'], data=data.T)
    df['value'] = df['value'].astype(float)

    plt.figure(figsize=(10,6))
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.color_palette("flare", as_cmap=True)
    bp = sns.violinplot(x="comparison", y="value", hue="state",
                    data=df, split=True, inner='stick')
    handles, labels = bp.get_legend_handles_labels()
    plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(1.01, 0.7), loc='upper left', fontsize=20)
    name = cond +' ('+ celltype+ ')'
    plt.xlabel(name, fontname="Arial", size=20)
    plt.ylabel('Firing rate of '+celltype + ' (Hz)', fontname="Arial", size=20)
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)        
    
    #Perform the Mann-Whitney U rank test on two independent samples.
    _, P_ctrl_md1 = stats.mannwhitneyu(ctrl1, md1)
    _, P_ctrl_md4 = stats.mannwhitneyu(ctrl2, md4)
    
    
    plt.title('P value of Ctrl vs. MD1 {:.3f}, Ctrl vs. MD4 {:.3f}'\
              .format(P_ctrl_md1, P_ctrl_md4), fontname="Arial", size=20)    
    
    plt.tight_layout()
    
    plt.savefig('./figures/'+name+'_violin.png')
    plt.savefig('./figures/EPS/'+name+'_violin.eps')       


def bar_plot(ctrl1, md1, ctrl2, md4, cond, celltype):
    """
    boxplot with standard error of the mean (SEM) 
    """    
     
    #regroup the data into dataframe
    data = np.concatenate(
        [[ctrl1, len(ctrl1)*['control vs. md1'], len(ctrl1)*['control1']],
         [md1, len(md1)*['control vs. md1'], len(md1)*['md1']],
         [ctrl2, len(ctrl2)*['control vs. md4'], len(ctrl2)*['control4']],
         [md4, len(md4)*['control vs. md4'], len(md4)*['md4']]],
        axis=1)
    
    df = pd.DataFrame(columns=['value', 'comparison', 'state'], data=data.T)
    df['value'] = df['value'].astype(float)
    
    plt.figure(figsize=(10,4))
    sns.set_style('white')
    bp = sns.barplot(x='comparison', y='value', hue='state', data=df, 
                palette='colorblind', ci=68, capsize=.15, linewidth=3)   
    bp = sns.stripplot(x='comparison', y='value', hue='state', data=df, jitter=0.25, 
                 size=10, alpha=0.5, edgecolor=sns.color_palette("hls", 4), linewidth=1, dodge=True)
    handles, labels = bp.get_legend_handles_labels()
    plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(1.01, 0.7), loc='upper left', fontsize=20)    
    name = cond +' ('+ celltype+ ')'
    plt.xlabel(name, size=20)
    plt.ylabel('Firing rate of '+celltype + ' (Hz)', size=20)
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)
    
    #Perform the Mann-Whitney U rank test on two independent samples.
    _, P_ctrl_md1 = stats.mannwhitneyu(ctrl1, md1)
    _, P_ctrl_md4 = stats.mannwhitneyu(ctrl2, md4)
    
    
    plt.title('P value of Ctrl vs. MD1 {:.3f}, Ctrl vs. MD4 {:.3f}'\
              .format(P_ctrl_md1, P_ctrl_md4), size=20)    
    
    plt.tight_layout()
    
    plt.savefig('./figures/'+name+'.png')
    plt.savefig('./figures/EPS/'+name+'.eps')        

def box_plot(ctrl1, md1, ctrl2, md4, cond, celltype):
    """
    ttest plot of ctrl vs. md1 & ctrl vs. md4
    """
    
    #regroup the data into dataframe
    data = np.concatenate(
        [[ctrl1, len(ctrl1)*['control vs. md1'], len(ctrl1)*['control1']],
         [md1, len(md1)*['control vs. md1'], len(md1)*['md1']],
         [ctrl2, len(ctrl2)*['control vs. md4'], len(ctrl2)*['control4']],
         [md4, len(md4)*['control vs. md4'], len(md4)*['md4']]],
        axis=1)
    
    df = pd.DataFrame(columns=['value', 'comparison', 'state'], data=data.T)
    df['value'] = df['value'].astype(float)
    
    plt.figure(figsize=(10,8))
    
    sns.set_style('white')
    
    bp = sns.boxplot(x='comparison', y='value', hue='state', data=df, 
                palette='colorblind', fliersize=0, linewidth=3)
    bp = sns.stripplot(x='comparison', y='value', hue='state', data=df, jitter=True, 
                 marker='o', alpha=0.9, color='grey', dodge=True)
    handles, labels = bp.get_legend_handles_labels()
    plt.legend(handles[0:4], labels[0:4], loc='upper center', fontsize=20)
    name = cond +' '+ celltype
    plt.xlabel(name, size=30)
    plt.ylabel('Firing rate of '+celltype + ' (Hz)', size=30)
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)
    
    #doing ttest This is a two-sided test for the null hypothesis that 2 
    #independent samples have identical average (expected) values. 
    #This test assumes that the populations have identical variances by default.
    P_ctrl_md1 = stats.ttest_ind(ctrl1, md1)[1]
    P_ctrl_md4 = stats.ttest_ind(ctrl2, md4)[1]
    
    plt.title('P value of Ctrl vs. MD1 {:.3f}, Ctrl vs. MD4 {:.3f}'\
              .format(P_ctrl_md1, P_ctrl_md4), size=20)    
    
    plt.tight_layout()
    
    plt.savefig('./figures/'+name+'.png')
    plt.savefig('./figures/EPS/'+name+'.eps')

def slopewithshadow_plot(Results, Syn_Strength, syn_name, ntrial):
    """
    plot the slope of neuron activity by varying specific synapse
    """
    plt.figure(figsize=(10,8), dpi=100)
    
    abs_syn_stre = np.abs(Syn_Strength)
    FR = np.zeros((ntrial, len(Results[0])))
    SLOPE = np.zeros(ntrial)
    Z_list = []
    for i in range(ntrial):
        fr = Results[i]
        plt.plot(abs_syn_stre, fr, 'o', alpha=0.7, markersize=10)
        
        #linear fitting
        z = np.polyfit(abs_syn_stre, fr, deg=1)
        
        FR[i] = fr
        SLOPE[i] = z[0]
        Z_list.append(z)
        
    mean_fr = np.mean(FR, axis=0)
    #linear fitting of the mean slope and plot
    z = np.polyfit(abs_syn_stre, mean_fr, deg=1)  
    p = np.poly1d(z)
    plt.plot(abs_syn_stre, p(abs_syn_stre), '-', color='b', linewidth=5)
    
    #fill between the min slope and the max slope
    min_indx = np.argmin(np.abs(SLOPE))
    min_z = Z_list[min_indx]
    min_p = np.poly1d(min_z)
    min_y = min_p(abs_syn_stre)
    
    max_indx = np.argmax(np.abs(SLOPE))
    max_z = Z_list[max_indx]
    max_p = np.poly1d(max_z)
    max_y = max_p(abs_syn_stre) 
    
    plt.fill_between(abs_syn_stre, min_y, max_y, color='b', alpha=0.3)
    
    plt.title('slope ' + str(np.round(z[0],4)))
    plt.xlabel('Synaptic strength', fontname="Arial", size=20)
    plt.ylabel('Evoked activity (Hz)', fontname="Arial", size=20)    
    plt.tight_layout()
    plt.savefig('./figures20/varying_'+ syn_name +'.png')
    plt.savefig('./figures20/EPS/varying_'+ syn_name +'.eps')                          

def slope_rank_plot(Results, Con_Stre, lowerbound, upperbound, cond, interval):
    """
    plot the slope rank of varying all synapses
    """
    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']

    Slope = {}; mean_slope={}
    for pre_i, pre in enumerate(PreSynap):
        for post_j, post in enumerate(PostSynap):
            synap_name = post+'_'+pre
            if synap_name in ['pc_vip', 'sst_sst', 'vip_vip']:
                continue
            else:    
                orig_synap_strength = Con_Stre[post_j,pre_i]
                #extract the lower boundary and the upper boundary
                low = lowerbound[post_j,pre_i]; up = upperbound[post_j,pre_i]
                vary_sign = np.sign(orig_synap_strength)
                X = np.arange(np.abs(low), np.abs(up)+1, interval) * vary_sign
                
                slopes = []
                for ii in range(len(Results[synap_name])): #a list
                    F_Neuron = Results[synap_name][ii]
                    #linear fitting
                    z = np.polyfit(X, F_Neuron, deg=1)
                    
                    slopes.append(z[0] * vary_sign) #*vary_sign means when increase the strength, see how the activity changes
                
                Slope[synap_name] = slopes  
                mean_slope[synap_name] = np.mean(slopes)

    #%% rank slope
    name = sorted(mean_slope, key=mean_slope.get, reverse=True)
    Value = []; Names = []
    for n in name:
        slopes = Slope[n]
        Value += slopes
        ns = [n]*len(slopes)
        Names += ns
    
    data = np.column_stack((Value, Names))
    #regroup the data into dataframe
    df = pd.DataFrame(data=data, columns=['value', 'name'])
    df['value'] = df['value'].astype(float)    

    plt.figure(figsize=(10,6), dpi=100)
    sns.set_style('white')    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    #sns.set_palette(sns.color_palette("vlag"))
    bp = sns.boxplot(x='name', y='value', data=df, fliersize=0, linewidth=1)
    bp = sns.stripplot(x='name', y='value', data=df, jitter=True, marker='o', alpha=0.9, dodge=True)
    handles, labels = bp.get_legend_handles_labels()
    plt.legend(handles[0:4], labels[0:4], loc='upper center', fontsize=20)
    plt.ylabel('Synaptic contribution (slope value)')
    plt.xticks(rotation=45)     
    
    plt.tight_layout()
    plt.savefig('./figures/sloperank.png')
    plt.savefig('./figures/EPS/sloperank.eps')        

def slope_plot(Results, Con_Stre, lowerbound, upperbound, cond, celltype):
    """
    plot the slope of varying the synapse, monitoring specific cell type
    Input: 
        Results: a dictionary 
        Con_Stre: 
        celltype:
        strength_width
    """
    PreSynap = ['pc', 'pv', 'sst', 'vip']
    PostSynap = ['pc', 'pv', 'sst', 'vip']

    fig, axs = plt.subplots(4, 4, figsize=(10,8), dpi=100)
    Slope = {}
    for pre_i, pre in enumerate(PreSynap):
        for post_j, post in enumerate(PostSynap):
            synap_name = post+'_'+pre
            if synap_name in ['pc_vip', 'sst_sst', 'vip_vip']:
                axs[post_j, pre_i].set_xticks([])
                axs[post_j, pre_i].set_yticks([])
                continue
            else:
                F_Neuron = Results[synap_name]
                
                orig_synap_strength = Con_Stre[post_j,pre_i]
                #extract the lower boundary and the upper boundary
                low = lowerbound[post_j,pre_i]; up = upperbound[post_j,pre_i]
                X = np.arange(np.abs(low), np.abs(up)+1) * np.sign(orig_synap_strength)
                axs[post_j, pre_i].plot(X, F_Neuron, 'o-', color='k', markersize=4, linewidth=2)
                axs[post_j, pre_i].set_xlabel('Varying '+synap_name)
                axs[post_j, pre_i].set_ylabel('Mean FR of '+celltype)
                
                #linear fitting
                z = np.polyfit(X, F_Neuron, deg=1)
                p = np.poly1d(z)
                axs[post_j, pre_i].plot(X, p(X), '--', linewidth=2)
                axs[post_j, pre_i].set_title('slope ' + str(np.round(z[0],4)))
                Slope[synap_name] = z[0]
                
                #add the start point and the end point of the varied synapses
                #MD4
                if synap_name == 'pc_pv':
                    synap_strength = Results[synap_name+'_input']
                    start_idx = np.where(synap_strength==-80)[0][0]
                    start_fr = F_Neuron[start_idx]
                    end_idx = np.where(synap_strength==-38)[0][0]
                    end_fr = F_Neuron[end_idx]
                    axs[post_j, pre_i].plot(-80, start_fr, 'go', markersize=10)
                    axs[post_j, pre_i].plot(-38, end_fr, 'ro', markersize=10)
                    
                elif synap_name == 'pv_sst':
                    synap_strength = Results[synap_name+'_input']
                    start_idx = np.where(synap_strength==-10)[0][0]
                    start_fr = F_Neuron[start_idx]
                    end_idx = np.where(synap_strength==-28)[0][0]
                    end_fr = F_Neuron[end_idx]   
                    axs[post_j, pre_i].plot(-10, start_fr, 'go', markersize=10)
                    axs[post_j, pre_i].plot(-28, end_fr, 'ro', markersize=10)                                    
                    
                elif synap_name == 'vip_sst':
                    synap_strength = Results[synap_name+'_input']
                    start_idx = np.where(synap_strength==-10)[0][0]
                    start_fr = F_Neuron[start_idx]
                    end_idx = np.where(synap_strength==-18)[0][0]
                    end_fr = F_Neuron[end_idx]                 
                    axs[post_j, pre_i].plot(-10, start_fr, 'go', markersize=10)
                    axs[post_j, pre_i].plot(-18, end_fr, 'ro', markersize=10)                    
                
                #MD1
                elif synap_name == 'vip_pc':
                    synap_strength = Results[synap_name+'_input']
                    start_idx = np.where(synap_strength==10)[0][0]
                    start_fr = F_Neuron[start_idx]
                    end_idx = np.where(synap_strength==22)[0][0]
                    end_fr = F_Neuron[end_idx]                 
                    axs[post_j, pre_i].plot(10, start_fr, 'g^', markersize=10)
                    axs[post_j, pre_i].plot(22, end_fr, 'r^', markersize=10)  
                else:
                    ValueError("Wrong Synapse Name, Check It.")
                
                #Invert the xaxis when the synaptic strength is negative,
                #so that we can see the varying of firing rate when 
                #increase of the strength
                '''
                if np.sign(orig_synap_strength) < 0:
                    #axs[post_j, pre_i].set_ylim(axs[post_j, pre_i].get_xlim()[::-1])
                    axs[post_j, pre_i].invert_xaxis()
                '''
                    
    plt.tight_layout()
    plt.savefig('./figures/varying_synpase_'+cond+'_'+celltype+'.png')
    plt.savefig('./figures/EPS/varying_synpase_'+cond+'_'+celltype+'.eps')

    #%% rank slope
    name = sorted(Slope, key=Slope.get, reverse=True)
    Value = []
    for n in name:
        value = Slope[n]
        Value.append(value)
    plt.figure(figsize=(10,4), dpi=100)
    plt.bar(x=name, height=Value, color='k', alpha=0.7)
    plt.ylabel('slope')
    plt.xticks(rotation=45)
    plt.savefig('./figures/slope_'+cond+'_'+celltype+'.png')
    plt.savefig('./figures/EPS/slope_'+cond+'_'+celltype+'.eps')
    
    abs_Slope = {}
    for n in name:
        abs_Slope[n] = np.abs(Slope[n])
    name = sorted(abs_Slope, key=abs_Slope.get, reverse=True)
    Value = []
    for n in name:
        value = abs_Slope[n]
        Value.append(value)
    plt.figure(figsize=(10,4), dpi=100)
    plt.bar(x=name, height=Value, color='k', alpha=0.7)
    plt.ylabel('absolute slope')
    plt.xticks(rotation=45)
    plt.savefig('./figures/abs_slope_'+cond+'_'+celltype+'.png')
    plt.savefig('./figures/EPS/abs_slope_'+cond+'_'+celltype+'.eps')


def percentage_plot(Results, cond, celltype):
    """
    plot the percentage of contribution of each synapses
    Input:
        Results: storing the varying-synapse firing rate results of one cell type
        condition:
    """
    varied_synapses = ['pc_pv', 'pv_sst', 'vip_sst']
    
    abs_diff = []
    norm_diff = []
    
    for synap in varied_synapses:
        F_Neuron = Results[synap] 
        synap_strength = Results[synap+'_input']
        
        if synap == 'pc_pv':
            start_idx = np.where(synap_strength==-80)[0][0]
            start_fr = F_Neuron[start_idx]
            end_idx = np.where(synap_strength==-38)[0][0]
            end_fr = F_Neuron[end_idx]     
            
            absolute_diff = end_fr-start_fr
            normalize_diff = (end_fr-start_fr)/np.abs((end_idx-start_idx)) 
            #divide by np.abs cuz we only want to koow the change per strength unit
        elif synap == 'pv_sst':
            start_idx = np.where(synap_strength==-10)[0][0]
            start_fr = F_Neuron[start_idx]
            end_idx = np.where(synap_strength==-28)[0][0]
            end_fr = F_Neuron[end_idx]     

            absolute_diff = end_fr-start_fr
            normalize_diff = (end_fr-start_fr)/np.abs((end_idx-start_idx)) 
            #divide by np.abs cuz we only want to koow the change per strength unit
        elif synap == 'vip_sst':
            start_idx = np.where(synap_strength==-10)[0][0]
            start_fr = F_Neuron[start_idx]
            end_idx = np.where(synap_strength==-18)[0][0]
            end_fr = F_Neuron[end_idx]     
            
            absolute_diff = end_fr-start_fr
            normalize_diff = (end_fr-start_fr)/np.abs((end_idx-start_idx)) 
            #divide by np.abs cuz we only want to koow the change per strength unit
        else:
            ValueError("Wrong Synapse Name, Check It.")
        
        abs_diff.append(absolute_diff)
        norm_diff.append(normalize_diff)
    
    if cond=='spont':
        fig, axs = plt.subplots(2, 1, figsize=(5,10), dpi=100)
        
        axs[0].bar(x=varied_synapses, height=abs_diff, color='k', alpha=0.7)
        axs[0].set_ylabel('Synaptic contribution to '+celltype)
        if celltype == 'pv':
            axs[0].set_ylim([-0.25, 0.02])
        elif celltype == 'sst':
            axs[0].set_ylim([-0.02, 0.25])
        elif celltype == 'vip':
            axs[0].set_ylim([-0.25, 0.02])
            
        plt.xticks(rotation=45)
        axs[1].bar(x=varied_synapses, height=norm_diff, color='k', alpha=0.7)
        axs[1].set_ylabel('Normalized synaptic contribution to '+celltype)
        if celltype == 'pv':
            axs[1].set_ylim([-0.014, 0.001])
        elif celltype == 'sst':
            axs[1].set_ylim([-0.001, 0.014])
        elif celltype == 'vip':
            axs[1].set_ylim([-0.014, 0.001])    
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./figures/percentage_'+cond+'_'+celltype+'.png')
        plt.savefig('./figures/EPS/percentage_'+cond+'_'+celltype+'.eps')
    else:
        fig, axs = plt.subplots(2, 1, figsize=(5,10), dpi=100)
        
        axs[0].bar(x=varied_synapses, height=abs_diff, color='k', alpha=0.7)
        axs[0].set_ylabel('Synaptic contribution to '+celltype)
        if celltype == 'pv':
            axs[0].set_ylim([-0.8, 0.1])
        elif celltype == 'sst':
            axs[0].set_ylim([-0.1, 0.8])
        elif celltype == 'vip':
            axs[0].set_ylim([-0.8, 0.1])
            
        plt.xticks(rotation=45)
        axs[1].bar(x=varied_synapses, height=norm_diff, color='k', alpha=0.7)
        axs[1].set_ylabel('Normalized synaptic contribution to '+celltype)
        if celltype == 'pv':
            axs[1].set_ylim([-0.04, 0.005])
        elif celltype == 'sst':
            axs[1].set_ylim([-0.005, 0.04])
        elif celltype == 'vip':
            axs[1].set_ylim([-0.04, 0.005])    
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./figures/percentage_'+cond+'_'+celltype+'.png')
        plt.savefig('./figures/EPS/percentage_'+cond+'_'+celltype+'.eps')   

def synaptic_contribution_plot(Results, celltype):
    """
    plot the percentage of contribution of each varying synapses during MD4
    Input:
        Results: storing the varying-synapse firing rate results of one cell type
        condition:
    """
    varied_synapses = ['pc_pv', 'pv_sst', 'vip_sst']
    
    norm_diff = []
    name = []
    
    for synap in varied_synapses:
        F_Neuron = Results[synap] 

        if synap == 'pc_pv':
            ND = []; Name = []
            for i in range(len(F_Neuron)):
                start_fr = F_Neuron[i][0]
                end_fr = F_Neuron[i][1]
                normalize_diff = (end_fr-start_fr)/(80-38)
                ND.append(normalize_diff)
                Name.append(synap)
        elif synap == 'pv_sst':
            ND = []; Name = []
            for i in range(len(F_Neuron)):
                start_fr = F_Neuron[i][0]
                end_fr = F_Neuron[i][1]
                normalize_diff = (end_fr-start_fr)/(28-10)
                ND.append(normalize_diff)  
                Name.append(synap)
        elif synap == 'vip_sst':
            ND = []; Name = []
            for i in range(len(F_Neuron)):
                start_fr = F_Neuron[i][0]
                end_fr = F_Neuron[i][1]
                normalize_diff = (end_fr-start_fr)/(18-10)
                ND.append(normalize_diff) 
                Name.append(synap)
        else:
            ValueError("Wrong Synapse Name, Check It.")
        
        norm_diff+=ND
        name+=Name
   
    data = np.column_stack((norm_diff, name))
    #regroup the data into dataframe
    df = pd.DataFrame(data=data, columns=['value', 'syn_name'])
    df['value'] = df['value'].astype(float)   
          

    plt.figure(figsize=(6,10), dpi=100)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)    
    bp = sns.barplot(x='syn_name', y='value', data=df, 
                palette='colorblind', ci=68, capsize=.35, linewidth=1)   
    bp = sns.stripplot(x='syn_name', y='value', data=df, jitter=0.25, 
                 size=10, alpha=0.5, linewidth=0.5, dodge=True)
    plt.ylabel('Evoked activity of '+celltype + ' (Hz)', size=20)
    plt.xticks(rotation=45)

    if celltype == 'pc':
        plt.ylim([-0.05, 0.01])    
    elif celltype == 'pv':
        plt.ylim([-0.05, 0.01])
    elif celltype == 'sst':
        plt.ylim([-0.01, 0.05])
    elif celltype == 'vip':
        plt.ylim([-0.05, 0.01])      
    
    plt.tight_layout()
    plt.savefig('./figures/percentage_'+celltype+'.png')
    plt.savefig('./figures/EPS/percentage_'+celltype+'.eps')        
        
def joint_varying(X, Y, Results_PC, Results_PV, Results_SST, Results_VIP, cond):
    """
    plot the joint varying synapses on firing rate
    Input:
        Results: storing the varying-synapse firing rate results of one cell type
        condition:
    """    
    fig = plt.figure(figsize=(20,5), dpi=100)
    
    ax = fig.add_subplot(141, projection='3d')
    pnt3d = ax.plot_surface(X, Y, Results_PC, rstride=1, cstride=1, cmap='cool', edgecolor='none')
    ax.set_xlabel('pc_pv strength', fontsize=20)
    ax.set_ylabel('pv_sst strength', fontsize=20)
    ax.set_zlabel('Mean FR of pcs', fontsize=20)
    plt.colorbar(pnt3d)

    
    ax = fig.add_subplot(142, projection='3d')
    pnt3d = ax.plot_surface(X, Y, Results_PV, rstride=1, cstride=1, cmap='cool', edgecolor='none')
    ax.set_xlabel('pc_pv strength', fontsize=20)
    ax.set_ylabel('pv_sst strength', fontsize=20)
    ax.set_zlabel('Mean FR of pvs', fontsize=20)    
    plt.colorbar(pnt3d)
        
    ax = fig.add_subplot(143, projection='3d')
    pnt3d = ax.plot_surface(X, Y, Results_SST, rstride=1, cstride=1, cmap='cool', edgecolor='none')
    ax.set_xlabel('pc_pv strength', fontsize=20)
    ax.set_ylabel('pv_sst strength', fontsize=20)
    ax.set_zlabel('Mean FR of ssts', fontsize=20)
    plt.colorbar(pnt3d)
    
    ax = fig.add_subplot(144, projection='3d')
    pnt3d = ax.plot_surface(X, Y, Results_VIP, rstride=1, cstride=1, cmap='cool', edgecolor='none')
    ax.set_xlabel('pc_pv strength', fontsize=20)
    ax.set_ylabel('pv_sst strength', fontsize=20)
    ax.set_zlabel('Mean FR of cips', fontsize=20)
    plt.colorbar(pnt3d)
    
    plt.tight_layout()
    
    plt.savefig('./figures/joint_varying_synapses_'+cond+'.png')
    plt.savefig('./figures/EPS/joint_varying_synapses_'+cond+'.eps')
    
        
        
    