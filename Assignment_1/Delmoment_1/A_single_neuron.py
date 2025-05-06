#!/usr/bin/env python3

import numpy as np
import time
from typing import List, Callable, Union, Optional

# Part A: Neuron implementation with NumPy
class NeuronWithNumPy:
    def __init__(self, num_inputs: int, activation_function: str = 'sigmoid'):
        """
        Initialize a neuron with random weights and bias using NumPy
        
        Args:
            num_inputs: Number of input features
            activation_function: Type of activation function to use ('sigmoid', 'relu', 'leaky_relu', or 'tanh')
        """
        # Initialize weights and bias as NumPy arrays
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = 0
        
        # Set the activation function
        self.activation_function = activation_function
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function"""
        return np.maximum(alpha * x, x)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        """Apply the selected activation function"""
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return self.relu(x)
        elif self.activation_function == 'leaky_relu':
            return self.leaky_relu(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")

    def forward(self, inputs: np.ndarray) -> float:
        """
        Forward pass through the neuron using NumPy
        
        Args:
            inputs: NumPy array of input values
            
        Returns:
            Output of the neuron after activation
        """
        # Ensure inputs is numpy array
        inputs = np.array(inputs)
        
        # Check input shape
        if inputs.shape[0] != self.weights.shape[0]:
            raise ValueError(f"Expected {self.weights.shape[0]} inputs, got {inputs.shape[0]}")
        
        # Calculate weighted sum using dot product
        pre_activation = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        output = self.activate(pre_activation)
        
        return output

# Om filen körs direkt, utför demonstration
if __name__ == "__main__":
    print("\n=== Del A: NumPy Neuron Implementation ===")
    
    # Skapa sample input (28x28 flattened MNIST image)
    sample_input = np.random.random(784)
    
    # Tidtagning för NumPy-implementationen
    start = time.time()
    neuron = NeuronWithNumPy(num_inputs=784, activation_function='sigmoid')
    output = neuron.forward(sample_input)
    end = time.time()
    time_numpy = end - start
    
    # Skriv ut resultat
    print(f"NumPy implementation output: {output:.6f}")
    print(f"Time for NumPy implementation: {time_numpy:.6f} seconds")
    print(f"This implementation uses NumPy's vectorized operations for efficiency")
    
    # Example for testing the neuron
    sample_inputs = np.array([0.5, -0.2, 0.1, 0.8])
    
    # Test NumPy implementation
    neuron = NeuronWithNumPy(num_inputs=4, activation_function='sigmoid')
    output = neuron.forward(sample_inputs)
    print(f"NumPy neuron output: {output}")
    
    # Test different activation functions
    for activation in ['sigmoid', 'relu', 'leaky_relu', 'tanh']:
        neuron = NeuronWithNumPy(num_inputs=4, activation_function=activation)
        output = neuron.forward(sample_inputs)
        print(f"{activation} output: {output}") 