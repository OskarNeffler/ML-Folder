#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# MNIST data loader implementation
def get_mnist_data_loaders(batch_size=64):
    # Skapa en transform för att normalisera data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Ladda MNIST train och test dataset
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Skapa data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class MNISTNet(nn.Module):
    def __init__(self, hidden_size=128):
        """
        Initialize a simple neural network for MNIST digit classification
        
        Args:
            hidden_size: Size of the hidden layer
        """
        super(MNISTNet, self).__init__()
        
        # Define the network architecture
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input data with shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor with shape (batch_size, 10)
        """
        # Flatten the input: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = self.flatten(x)
        
        # First fully connected layer with ReLU activation
        x = self.fc1(x)
        x = self.relu(x)
        
        # Output layer with sigmoid activation
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x


def train_model(model, train_loader, test_loader, device, epochs=5, learning_rate=0.001):
    """
    Train the PyTorch model
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to train on (GPU or CPU)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model, training history, and execution time
    """
    start_time = time.time()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar for training
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                # Move data to the device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update running loss
                running_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({"loss": running_loss / (i + 1)})
        
        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    execution_time = time.time() - start_time
    
    return model, {"losses": train_losses, "accuracies": test_accuracies}, execution_time


def evaluate_model(model, test_loader, device):
    """Evaluate a PyTorch model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def plot_results(history, gpu_time=None, cpu_time=None):
    """
    Plot training loss and test accuracy
    
    Args:
        history: Dictionary with training losses and test accuracies
        gpu_time: Execution time on GPU (if available)
        cpu_time: Execution time on CPU
    """
    # Create a figure with subplots
    if gpu_time is not None and cpu_time is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(history["losses"])
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    # Plot test accuracy
    ax2.plot(history["accuracies"])
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    
    # Plot execution time comparison if both GPU and CPU times are available
    if gpu_time is not None and cpu_time is not None:
        times = [cpu_time, gpu_time]
        labels = ["CPU", "GPU"]
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        ax3.bar(labels, times)
        ax3.set_title(f"Execution Time (Speed-up: {speedup:.2f}x)")
        ax3.set_ylabel("Time (seconds)")
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    
    return fig  # Return the figure object for display in Jupyter


def check_cuda_availability():
    """Check if CUDA is available and return device"""
    if torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        print(f"CUDA is available! Found {cuda_device_count} GPU(s).")
        for i in range(cuda_device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return True, torch.device("cuda:0")
    else:
        print("CUDA is not available. Using CPU instead.")
        return False, torch.device("cpu")


if __name__ == "__main__":
    print("\n=== Del C: PyTorch Implementation ===")
    
    # Kontrollera CUDA-tillgänglighet
    cuda_available, device = check_cuda_availability()
    
    # Ladda MNIST-data
    train_loader, test_loader = get_mnist_data_loaders(batch_size=64)
    
    # Skapa modellen och flytta till tillgänglig enhet (GPU eller CPU)
    model = MNISTNet(hidden_size=128)
    model.to(device)
    
    # Träna i 3 epoker på GPU
    device_name = "GPU" if cuda_available else "CPU"
    print(f"Training PyTorch model on {device_name} (3 epochs)...")
    trained_model, history, execution_time = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=3,
        learning_rate=0.001
    )
    
    # Slutlig utvärdering
    accuracy = evaluate_model(trained_model, test_loader, device)
    print(f"Test accuracy after 3 epochs: {accuracy:.4f}")
    print(f"Execution time on {device_name}: {execution_time:.2f} seconds")
    
    # Om GPU är tillgänglig, kör också på CPU för jämförelse
    cpu_history = None
    cpu_time = None
    
    if cuda_available:
        print("\nRunning the same model on CPU for comparison...")
        device_cpu = torch.device("cpu")
        model_cpu = MNISTNet(hidden_size=128)
        model_cpu.to(device_cpu)
        
        trained_model_cpu, cpu_history, cpu_time = train_model(
            model=model_cpu,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device_cpu,
            epochs=3,
            learning_rate=0.001
        )
        
        # Rapport för speedup
        speedup = cpu_time / execution_time
        print(f"\nGPU vs CPU Speed-up: {speedup:.2f}x")
        print(f"GPU training time: {execution_time:.2f} seconds")
        print(f"CPU training time: {cpu_time:.2f} seconds")
        
        # Plotta loss function för både GPU och CPU
        plt.figure(figsize=(10, 6))
        plt.plot(history["losses"], label=f'GPU loss')
        plt.plot(cpu_history["losses"], label=f'CPU loss')
        plt.title('Loss Function')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_comparison.png')
        
        # Visa grafen direkt i Colab
        print("Grafen är sparad som 'loss_comparison.png'")
        print("För att visa grafen i Colab, kör följande kod i en ny cell:")
        print("from IPython.display import Image")
        print("Image('loss_comparison.png')")
        
        plt.close()
    
    # Save the model
    model_path = "mnist_model_gpu.pth" if cuda_available else "mnist_model.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}") 