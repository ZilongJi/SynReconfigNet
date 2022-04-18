# SynReconfigNet
The Network Model for Systematic Reconfiguration of Cortical Microcircuits

## To run this code, you need the following python packages:
1, brainpy: https://github.com/PKU-NIP-Lab/BrainPy <br />
2, jaxlib: see brainy installation instructions <br />
2, pandas <br />
3, seaborn <br />

## The network model is described as follows:
$$
\tau_{PC} \frac{\mathrm{d} r_{PC,i}}{\mathrm{d} t} =-r_{PC,i}+f(I_{i}-\theta)+\sigma\mathrm{d}W
$$
