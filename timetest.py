# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 13:27:16 2022

@author: Zilong
"""

import numpy as np
import brainpy.math as bm
import time 

def SetConnectivityNP(Con_Prob, Con_Stre, NC):
    '''
    Con_Prob: connection probability of all cell types
    Con_Stre: connection strength between two cells
    NC: nparray of the number of PCs, PVs, SSTs, VIPs
    '''
    #
    NCon = np.round(Con_Prob*NC)
    NCon = np.asarray(NCon, dtype=np.int32)
    
    NameList = np.asarray([['pc_pc', 'pc_pv', 'pc_sst', 'pc_vip'],
                      ['pv_pc', 'pv_pv', 'pv_sst', 'pv_vip'],
                      ['sst_pc', 'sst_pv', 'sst_sst', 'sst_vip'],
                      ['vip_pc', 'vip_pv', 'vip_sst', 'vip_vip']])
    
    WeightDic = {} #store all the weight information
    
    for i in range(16):
        m,n = np.unravel_index(i,(4,4))
        weightname = NameList[m,n]
        Mtx = np.zeros((NC[m], NC[n]))
        if NCon[m,n]>0: #if there are connections, go to the next step
            if m==n: #connection between the neurons in same type, omit the autapse
                for l in range(NC[m]):
                    weight = Con_Stre[m,n]*np.array([0] * (NC[n]-1-NCon[m,n]) + [1] * NCon[m,n])/NCon[m,n]
                    np.random.shuffle(weight)
                    weight = np.insert(weight,l,0)
                    Mtx[l,:] = weight
            else: #connection between the neurons in different types
                for l in range(NC[m]):
                    weight = Con_Stre[m,n]*np.array([0] * (NC[n]-NCon[m,n]) + [1] * NCon[m,n])/NCon[m,n]
                    np.random.shuffle(weight)
                    Mtx[l,:] = weight
        WeightDic[weightname] = Mtx
    
    return WeightDic


def SetConnectivityBM(Con_Prob, Con_Stre, NC):
    '''
    Con_Prob: connection probability of all cell types
    Con_Stre: connection strength between two cells
    NC: nparray of the number of PCs, PVs, SSTs, VIPs
    '''
    #
    NCon = bm.round(Con_Prob*NC)
    NCon= bm.asarray(NCon, dtype=int)
    
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

#%%
np.random.seed(123)
Con_Prob = bm.array([[0.096,0.776,0.08,0.007],
                     [0.622,0.643,0.317,0.088],
                     [0.460,0.176,0.000,0.119],
                     [0.245,0.239,0.237,0.000]])   

Con_Stre = bm.array([[8.,  -79.,  -8.,  0.],
                     [22., -70.,  -12., -14.],
                     [4.,  -41.,   0.,  -5.],
                     [10., -35.,  -7.,  0.]])
#normalize the synaptic strength for stability
Con_Stre = Con_Stre/79.0 

#total number of neurons (nparray)
NC = bm.array([700, 100, 100, 100])

start = time.time()
WeightsBM = SetConnectivityBM(Con_Prob, Con_Stre, NC) 
print('processing time is {:.2f}s'.format(time.time()-start))

#%%
np.random.seed(123)
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

#total number of neurons (nparray)
NC = np.array([700, 100, 100, 100])

start = time.time()
WeightsNP = SetConnectivityNP(Con_Prob, Con_Stre, NC) 
print('processing time is {:.2f}s'.format(time.time()-start))
