# -*- coding: utf-8 -*-
"""
BMI 3002 HW 2 Question 4. Toy Multilayer Perceptron capable of classifying 3D 
vectors into binary classes.

Created on Tue Mar 30 15:59:44 2021

@author: Jonathan Chien
"""

#%% Import standard libraries

import numpy as np
import matplotlib.pyplot as plt


#%% Write helper functions and MultilayerPerceptron class 

def sigmoid(z):
        sigma = 1 / (1 + np.exp(-z))
        return sigma
    
    
def calc_loss(yhat, y_true):
        loss = 0.5*(yhat - y_true)**2
        return loss 
    
    
class MultilayerPerceptron:
    def __init__(self, weights, biases):
        self.weights = weights # {1: w1, 2: w2, 3: w3} index according to rightmost layer
        self.biases = biases # {1: b1, 2: b2, 3: b3}
        self.activations = np.nan
        self.inputs = np.nan
        self.loss = np.nan
        self.gradient = np.nan
        self.i_epoch = 0
    
    def forward_pass(self, x, y_true):
        # Calculate inputs and activation for layer 1 (hidden 1)
        z1 = self.weights[1] @ x + self.biases[1]
        a1 = sigmoid(z1)
        
        # Calculate inputs and activation for layer 2 (hidden 2)
        z2 = self.weights[2] @ a1 + self.biases[2]
        a2 = sigmoid(z2)
        
        # Calculate inputs and activation for layer 3 (hidden 3)
        z3 = self.weights[3] @ a2 + self.biases[3]
        a3 = sigmoid(z3) 
        
        # Store all inputs and activations for all three layers.
        self.activations = {1: a1, 2: a2, 3: a3}
        self.inputs = {1: z1, 2: z2, 3: z3}
        
        # Calculate and store loss. 
        if self.i_epoch == 0:
            self.loss = calc_loss(a3, y_true)
        elif self.i_epoch > 0:
            self.loss = np.append(self.loss, calc_loss(a3, y_true))
            
    def backward_pass(self, x, y_true, alpha=1.):
        # Calculate bias and weight gradients for layer 3 (output).
        delta3 = (self.activations[3] - y_true) \
            * (sigmoid(self.inputs[3]) * (1 - sigmoid(self.inputs[3])))
        delCdelB3 = delta3*alpha
        delCdelW3 = np.outer(delta3, self.activations[2])*alpha
        
        # Calculate bias and weight gradients for layer 2 (hidden 2).
        delta2 = np.squeeze(self.weights[3].T * delta3) \
            * (sigmoid(self.inputs[2]) * (1 - sigmoid(self.inputs[2])))
        delCdelB2 = delta2*alpha       
        delCdelW2 = np.outer(delta2, self.activations[1])*alpha
        
        # Calculate bias and weight gradients for layer 1 (hidden 1).
        delta1 = self.weights[2].T @ delta2 \
            * (sigmoid(self.inputs[1]) * (1 - sigmoid(self.inputs[1])))
        delCdelB1 = delta1*alpha  
        delCdelW1 = np.outer(delta1, x)*alpha
        
        # Store current gradient.
        self.gradient = {"delCdelW1": delCdelW1, "delCdelB1": delCdelB1,
                         "delCdelW2": delCdelW2, "delCdelB2": delCdelB2,
                         "delCdelW3": delCdelW3, "delCdelB3": delCdelB3}
        
        # Step in direction of negative gradient.
        self.weights[3] = self.weights[3] - delCdelW3
        self.biases[3] = self.biases[3] - delCdelB3
        self.weights[2] = self.weights[2] - delCdelW2
        self.biases[2] = self.biases[2] - delCdelB2
        self.weights[1] = self.weights[1] - delCdelW1
        self.biases[1] = self.biases[1] - delCdelB1
        
        # Update epoch index.
        self.i_epoch += 1
        
            
#%% Load data and instantiate MultilayerPerceptron object

# Load weights and biases.
w1 = np.genfromtxt('w1.csv', delimiter=',')   
w2 = np.genfromtxt('w2.csv', delimiter=',') 
w3 = np.genfromtxt('w3.csv', delimiter=',') 
b1 = np.genfromtxt('b1.csv', delimiter=',') 
b2 = np.genfromtxt('b2.csv', delimiter=',') 
b3 = np.genfromtxt('b3.csv', delimiter=',') 

# Store weights and biases in respective dictionaries.
weights = {1: w1, 2: w2, 3: w3} # index according to layers with input as layer 0
biases = {1: b1, 2: b2, 3: b3}

# Create input and ground truth output.
x = np.array([-1, 0, 1], dtype=float).T
y_true = np.array([1], dtype = float)

# Instantiate neural network as object of MultilayerPerceptron class.
nn = MultilayerPerceptron(weights, biases)


#%% Run training epochs manually (make sure to instantiate new object first)

alpha = 1. # learning rate
nn.forward_pass(x, y_true)
nn.backward_pass(x, y_true, alpha)


#%% Run multiple epochs at once (make sure to instantiate new object first)

# Set learning rate and number of training epochs.
alpha = 1. 
n_epochs = 15 

# Run all training epochs.
for i in range(n_epochs):
    nn.forward_pass(x, y_true)
    nn.backward_pass(x, y_true, alpha)       

# Plot epoch index against loss.
plt.plot(np.arange(1,n_epochs+1), nn.loss)     
plt.title("Multilayer perceptron loss across training epochs")
plt.xlabel("Epoch index")      
plt.ylabel("Loss")  
