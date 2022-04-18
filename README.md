# SynReconfigNet
The Network Model for Systematic Reconfiguration of Cortical Microcircuits

## Required packages and dependencies
1, brainpy: https://github.com/PKU-NIP-Lab/BrainPy <br />
2, jaxlib: see brainy installation instructions <br />
2, pandas <br />
3, seaborn <br />

## Short introduction
The network is composed by rate-based point neurons of excitatory pyramidal cells and inhibitory PV, SST and VIP cells. All neurons are connected randomly with probabilities and strengths constrained by the experimental findings
![alt text](https://github.com/Lenakeiz/PathIntegrationDataEstimationModelling/blob/main/ReadmeImages/DistributionsAtTheEndOfEachSegment.png "Fitted Distributions at each segment")

## The network model is described as follows:
$$
 \tau_{PC} \frac{\mathrm{d} r_{PC,i}}{\mathrm{d} t} =-r_{PC,i}+f(I_{i}-\theta)+\sigma\mathrm{d}W
$$
