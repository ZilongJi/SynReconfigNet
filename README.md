# SynReconfigNet
The Network Model for Systematic Reconfiguration of Cortical Microcircuits

## Required packages and dependencies
Implemented using Python 3.8 but later versions should work as well
The folllowing packages are also required
1, brainpy: https://github.com/PKU-NIP-Lab/BrainPy <br />
2, jaxlib: see brainy installation instructions <br />
3, pandas <br />
4, seaborn <br />

## Short introduction

### Synaptic connection probability and strength
The network is composed by rate-based point neurons of excitatory pyramidal cells and inhibitory PV, SST and VIP cells. All neurons are connected randomly with probabilities and strengths constrained by the experimental findings:
<img src="https://github.com/ZilongJi/SynReconfigNet/blob/main/ReadmeImages/ProbAndStrength.jpg" width=60% height=60%>

### The network model:
<img src="https://github.com/ZilongJi/SynReconfigNet/blob/main/ReadmeImages/equations.jpg" width=60% height=60%>
