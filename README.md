# sofenn: Self-Organizing Fuzzy Neural Network

Welcome to sofenn (sounds like soften)! 

This package is a Keras/TensorFlow implementation of a Self-Organizing Fuzzy Neural Network. The sofenn framework 
consists of two models:
<br /> (1) Fuzzy Network - Underlying fuzzy network whose parameters are optimized during training
<br /> (2) Self-Organizer - Meta-model that optimizes the architecture of the Fuzzy Network

## FuzzyNetwork
Underlying neural network model that contains parameters to be optimized during training

## SelfOrganizer
Meta-model to optimize architecture of underlying fuzzy network

## Model Description
The model is implemented per the description in:
<br />

'An on-line algorithm for creating self-organizing fuzzy neural networks\'
<br /> Leng, Prasad, McGinnity (2004)

## Layers

### Inputs

### Fuzzy Layer (2) of SOFNN
- Radial (Ellipsoidal) Basis Function Layer
- each neuron represents "if-part" or premise of a fuzzy rule
- individual Membership Functions (MF) are applied to each feature for each neuron
- output is product of Membership Functions
- each MF is Gaussian function:

    - mu(i,j) = exp{- [x(i) - c(i,j)]^2 / [2 * sigma(i,j)^2]}

    - for i features and  j neurons:

    - mu(i,j)    = ith MF of jth neuron

    - c(i,j)     = center of ith MF of jth neuron

    - sigma(i,j) = width of ith MF of jth neuron

- output for Fuzzy Layer is:
    phi(j) = exp{-sum[i=1,r;
            [x(i) - c(i,j)]^2 / [2 * sigma(i,j)^2]]}
            
### Normalized Layer (3) of SOFNN
- Normalization Layer

- output of each neuron is normalized by total output from previous layer
- number of outputs equal to previous layer (# of neurons)
- output for Normalized Layer is:

    psi(j) = phi(j) / sum[k=1, u; phi(k)]
                for u neurons
        - with:

    psi(j) = output of Fuzzy Layer neuron j
    
### Weighted Layer (4) of SOFNN
- Weighting of ith MF of each feature

- yields the "consequence" of the jth fuzzy rule of fuzzy model
- each neuron has two inputs:
    - output of previous related neuron j
    - weighted bias w2j
- with:
    r      = number of original input features

    B      = [1, x1, x2, ... xr]
    Aj     = [aj0, aj1, ... ajr]

    w2j    = Aj * B =
            aj0 + aj1x1 + aj2x2 + ... ajrxr

    psi(j) = output of jth neuron from
            normalized layer

-output for weighted layer is:
    fj     = w2j psi(j)
    
###     Output Layer (5) of SOFNN
- Unweighted sum of each output of previous layer (f)

- output for fuzzy layer is:
    sum[k=1, u; f(k)]
            for u neurons
    - shape: (samples,)