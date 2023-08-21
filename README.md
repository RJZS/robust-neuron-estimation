# robust-neuron-estimation
Code for the CDC 2023 submission 'Robust online estimation of biophysical neural networks'

## Neuron Parameters & Dynamics

The maximal conductances are the parameters learned by the adaptive observer. Four of the maximal conductances are constant:

gleak=0.3
gNa=100
gK=65
gCaT=0.5

Two of the maximal conductances, gCaL and gKCa are modulated. They are defined respectively by the functions `gCaL_fn` and `gKCa_fn`.

The reversal potentials are as follows:

ENa = 40.; # Sodium reversal potential
EK = -90.; # Potassium reversal potential
ECa = 120.; # Calcium reversal potential
Eleak = -50.; # Reversal potential of leak channels

The neuron's membrane capacitance c = 0.1

The activation and time-constant functions for the gating variables are given at the top of `GD_odes.jl`

## Observer Parameters

The observer parameters are the gain γ and the forgetting rate α (not to be confused with the variable αCa, which is a parameter of the neuron).
