# sofenn: Self-Organizing Fuzzy Neural Network

Welcome to sofenn (sounds like soften)! 

This package is a Keras/TensorFlow implementation of a Self-Organizing Fuzzy Neural Network. The **sofenn** framework 
consists of two models:
<br /> (1) **Fuzzy Network** - Underlying fuzzy network whose parameters are optimized during training
<br /> (2) **Self-Organizer** - Metamodel that optimizes the architecture of the Fuzzy Network

## FuzzyNetwork
Underlying neural network model that contains parameters to be optimized during training

## FuzzySelfOrganizer
Metamodel to optimize the architecture of underlying fuzzy network

## Installation

You can install the package using pip:

```bash
pip install sofenn
```

## Usage
[Demo notebooks](https://github.com/andrewre23/sofenn/tree/master/demos) available on Github.

Importing the model and self-organizer:
```python
from sofenn import FuzzyNetwork, FuzzySelfOrganizer

# initialize model separate, and attach to self-organizer
model = FuzzyNetwork(input_shape, **_init_params)
sofnn = FuzzySelfOrganizer(model=model)
sofnn.self_organize(x, y)

# initialize sofnn directly
sofnn = FuzzySelfOrganizer(input_shape, **_init_params)
sofnn.self_organize(x, y)
```

## Model Description
The model is implemented per the description in:
<br />

**'An on-line algorithm for creating self-organizing fuzzy neural networks\'**
<br /> Leng, Prasad, McGinnity (2004)

![alt text](https://raw.githubusercontent.com/andrewre23/sofenn/images/images/sofnn_structure.png)

*Fuzzy Neural Network Architecture*

Credit: Leng, Prasad, McGinnity (2004)



## Layers

### Inputs Layer (0)
**Input layer of network**

- **input** : <a href="https://www.codecogs.com/eqnedit.php?latex=input" target="_blank"><img src="https://latex.codecogs.com/gif.latex?input" title="input" /></a>
    - shape: (*, features)

### Fuzzy Layer (1)
**Radial (Ellipsoidal) Basis Function Layer**
- Each neuron represents "if-part" or premise of a fuzzy rule
- Individual Membership Functions (MF) are applied to each feature for each neuron
- Output is product of Membership Functions
- Each MF is a Gaussian function:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\mu_{ij}&space;=&space;\exp([-\frac{(x_i&space;-&space;c_{ij})^2}{2\sigma^2_{ij}}])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu_{ij}&space;=&space;\exp([-\frac{(x_i&space;-&space;c_{ij})^2}{2\sigma^2_{ij}}])" title="\mu_{ij} = \exp([-\frac{(x_i - c_{ij})^2}{2\sigma^2_{ij}}])" /></a>
    - for i features and j neurons:

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\mu_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu_{ij}" title="\mu_{ij}" /></a>    = ith MF of jth neuron

    - <a href="https://www.codecogs.com/eqnedit.php?latex=c_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{ij}" title="c_{ij}" /></a> = center of ith MF of jth neuron

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_{ij}" title="\sigma_{ij}" /></a> = width of ith MF of jth neuron

- output for Fuzzy Layer is:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\Phi_j&space;=&space;\exp(\sum_{i=1}^{r}&space;\frac{(x_i&space;-&space;c_{ij})^2}{2\sigma^2_{ij}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Phi_j&space;=&space;\exp(\sum_{i=1}^{r}&space;\frac{(x_i&space;-&space;c_{ij})^2}{2\sigma^2_{ij}})" title="\Phi_j = \exp(\sum_{i=1}^{r} \frac{(x_i - c_{ij})^2}{2\sigma^2_{ij}})" /></a>
    
- **input** : <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a>
    - shape: (*, features)
- **output** : <a href="https://www.codecogs.com/eqnedit.php?latex=\Phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Phi" title="\Phi" /></a>
    - shape: (*, neurons)

![alt text](https://raw.githubusercontent.com/andrewre23/sofenn/images/images/neuron.png)

*Information flow of* **r** *features within neuron* **j**

Credit: Leng, Prasad, McGinnity (2004)

### Normalize Layer (2)
**Normalization Layer**

- Output of each neuron is normalized by total output from the previous layer
- Number of outputs equal to the previous layer (# of neurons)
- Output for Normalize Layer is:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\Psi_j&space;=&space;\frac{\Phi_j}{\sum_{k=1}^{u}&space;\Phi_k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Psi_j&space;=&space;\frac{\Phi_j}{\sum_{k=1}^{u}&space;\Phi_k}" title="\Psi_j = \frac{\Phi_j}{\sum_{k=1}^{u} \Phi_k}" /></a>

    <a href="https://www.codecogs.com/eqnedit.php?latex=\Psi_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Psi_j" title="\Psi_j" /></a> = output of Fuzzy Layer neuron j

- **input** : <a href="https://www.codecogs.com/eqnedit.php?latex=\Phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Phi" title="\Phi" /></a>
    - shape  : (*, neurons)
- **output** : <a href="https://www.codecogs.com/eqnedit.php?latex=\Psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Psi" title="\Psi" /></a>
    - shape : (*, neurons)

### Weighted Layer (3)
**Weighting of ith MF of each feature**

- Yields the "consequence" of the *j*th fuzzy rule of the fuzzy model
- Each neuron has two inputs:
    - <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a> = output of previous related neuron
    - <a href="https://www.codecogs.com/eqnedit.php?latex=w_2j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_2j" title="w_2j" /></a> = weighted bias 
- with:
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r" title="r" /></a>      = number of original input features

    <a href="https://www.codecogs.com/eqnedit.php?latex=B&space;=&space;[1,&space;x_1,&space;x_2,&space;...&space;,&space;x_r]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B&space;=&space;[1,&space;x_1,&space;x_2,&space;...&space;,&space;x_r]" title="B = [1, x_1, x_2, ... , x_r]" /></a>
    <a href="https://www.codecogs.com/eqnedit.php?latex=A_j&space;=&space;[a_{j0},&space;a_{j1},&space;a_{j2},&space;...,&space;a_{jr}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A_j&space;=&space;[a_{j0},&space;a_{j1},&space;a_{j2},&space;...,&space;a_{jr}]" title="A_j = [a_{j0}, a_{j1}, a_{j2}, ..., a_{jr}]" /></a>

    <a href="https://www.codecogs.com/eqnedit.php?latex=w_2j&space;=&space;A_j&space;*&space;B&space;=&space;a_{j0}&space;&plus;&space;a_{j1x_1}&space;&plus;&space;a_{j2x_2}&space;&plus;&space;...&space;&plus;&space;a_{jr}x_r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_2j&space;=&space;A_j&space;*&space;B&space;=&space;a_{j0}&space;&plus;&space;a_{j1x_1}&space;&plus;&space;a_{j2x_2}&space;&plus;&space;...&space;&plus;&space;a_{jr}x_r" title="w_2j = A_j * B = a_{j0} + a_{j1x_1} + a_{j2x_2} + ... + a_{jr}x_r" /></a>

    <a href="https://www.codecogs.com/eqnedit.php?latex=\Psi_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Psi_j" title="\Psi_j" /></a> = output of jth neuron from
            normalize layer

- output for weighted layer is:
    <a href="https://www.codecogs.com/eqnedit.php?latex=f_j&space;=&space;w_{2j}\Psi_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_j&space;=&space;w_{2j}\Psi_j" title="f_j = w_{2j}\Psi_j" /></a>

- **inputs** : <a href="https://www.codecogs.com/eqnedit.php?latex=[x,&space;\Psi]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[x,&space;\Psi]" title="[x, \Psi]" /></a>
    - shape: \[(\*, 1+features), (*, neurons)]
- **output** : <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>
    - shape: (*, neurons)


### Output Layer (4)
**Final Output**

- Unweighted sum of each output of the previous layer (<a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>)
- Provide activation function to layer
- Function choice determines output shape (e.g., linear vs. softmax)
- Output for fuzzy layer is:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{k=1}^{u}&space;f(k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{k=1}^{u}&space;f(k)" title="\sum_{k=1}^{u} f(k)" /></a>
            
    for *u* neurons
    
- Provide activation function to layer (default: linear)
- Activation function determines output dimensions

### Examples

Regression output:
- **input** : <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>
    - shape: (*, neurons)
- **output** : <a href="https://www.codecogs.com/eqnedit.php?latex=output" target="_blank"><img src="https://latex.codecogs.com/gif.latex?output" title="output" /></a>
    - shape: (*,)


Softmax classification output:
- **input** : <a href="https://www.codecogs.com/eqnedit.php?latex=output" target="_blank"><img src="https://latex.codecogs.com/gif.latex?output" title="output" /></a>
    - shape: (*, )
- **output** : <a href="https://www.codecogs.com/eqnedit.php?latex=softmax" target="_blank"><img src="https://latex.codecogs.com/gif.latex?softmax" title="softmax" /></a>
    - shape: (*, classes)
