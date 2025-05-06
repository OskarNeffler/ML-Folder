#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List, Callable, Optional
import time

class NeuralLayerNumPy:
    def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid'):
        """
        Initialize a neural network layer using NumPy
        
        Args:
            input_size: Number of input features
            output_size: Number of neurons in the layer (output size)
            activation: Activation function to use ('sigmoid', 'relu', 'leaky_relu', 'tanh')
        """
        # Initialize weights with shape (input_size, output_size)
        # Using He initialization for better convergence
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(output_size)
        self.activation = activation
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer
        
        Args:
            inputs: Input data with shape (batch_size, input_size)
            
        Returns:
            Layer output after activation with shape (batch_size, output_size)
        """
        # Ensure inputs is properly shaped
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
            
        # Matrix multiplication: (batch_size, input_size) @ (input_size, output_size) = (batch_size, output_size)
        z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'leaky_relu':
            return self._leaky_relu(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")


class SimpleNeuralNetworkNumPy:
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        """
        Initialize a multi-layer neural network using NumPy
        
        Args:
            layer_sizes: List of layer sizes (including input size)
            activations: List of activation functions for each layer
        """
        if len(layer_sizes) < 2:
            raise ValueError("At least two layers (input + output) are required")
        
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must match number of layers - 1")
        
        # Initialize layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = NeuralLayerNumPy(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire network
        
        Args:
            x: Input data
            
        Returns:
            Network output
        """
        # Pass input through each layer sequentially
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, X):
        """
        Gör en prediktion baserat på input-data
        
        Args:
            X: Input-data, shape [batch_size, input_size]
        
        Returns:
            Predikterade klasser, shape [batch_size]
        """
        # Kör forward pass
        outputs = self.forward(X)
        
        # Returnera den klass med högst sannolikhet
        return np.argmax(outputs, axis=1)


# Om filen körs direkt
if __name__ == "__main__":
    print("\n=== Del B: NumPy Neural Network Implementation ===")
    
    # Skapa en enkel modell för MNIST-klassificering
    model = SimpleNeuralNetworkNumPy(
        layer_sizes=[784, 128, 10],  # 784 inputs, 128 hidden, 10 outputs
        activations=['relu', 'sigmoid']
    )
    
    # Generera slumpmässig indata (batch med 100 MNIST-liknande bilder)
    batch_size = 100
    sample_input = np.random.rand(batch_size, 784)
    
    # Tidtagning för forward pass
    start = time.time()
    output = model.forward(sample_input)
    predictions = model.predict(sample_input)
    end = time.time()
    forward_time = end - start
    
    # Skriv ut resultat
    print(f"Network architecture: 784 -> 128 -> 10")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Time for forward pass with batch size {batch_size}: {forward_time:.6f} seconds")
    print(f"Average time per sample: {forward_time/batch_size:.6f} seconds")
    
    # Example with a single sample
    single_input = np.random.rand(784)
    single_output = model.forward(single_input)
    single_prediction = model.predict(single_input)
    print(f"Single sample prediction: {single_prediction[0]}")
    
    # Note: In a real implementation, we would need to:
    # 1. Load the MNIST dataset
    # 2. Implement training procedure with backpropagation
    # 3. Evaluate model performance
    # These steps are implemented in PyTorch version (Part C) 