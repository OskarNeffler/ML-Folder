#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Förbättrad MNIST data loader med augmentation
def get_mnist_data_loaders(batch_size=128):
    # Förbättrad transform med data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotera slumpmässigt upp till 10 grader
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slumpmässig förskjutning
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Standard transform för testdata (ingen augmentation på testdata)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST tränings- och testdataset
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Skapa data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# CNN-modell för MNIST
class MNISTNetCNN(nn.Module):
    def __init__(self, hidden_size=256):
        super(MNISTNetCNN, self).__init__()
        
        # CNN-lager för förbättrad feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected lager
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        # Omforma platta bilder (784) till 2D (28x28) om det behövs
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            
        # CNN-lager
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected lager
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

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

def train_model_gpu(model, train_loader, test_loader, device, epochs=20, learning_rate=0.001):
    """Train a PyTorch model on GPU with optimizations"""
    start_time = time.time()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler för bättre konvergens
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # För att spara träningshistorik
    history = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Använd tqdm för progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, data in enumerate(progress_bar):
            # Get the inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/(i+1))
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        accuracy = evaluate_model_gpu(model, test_loader, device)
        
        # Get current learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate based on loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print if learning rate changed
        if old_lr != new_lr:
            print(f"Learning rate reduced from {old_lr} to {new_lr}")
        
        # Save history
        history.append((epoch_loss, accuracy))
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    return model, history, execution_time

def evaluate_model_gpu(model, test_loader, device):
    """Evaluate a PyTorch model on GPU"""
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

# Om filen körs direkt
if __name__ == "__main__":
    print("\n=== Optimized CNN on CUDA GPU ===")
    
    # Kontrollera CUDA-tillgänglighet
    cuda_available, device = check_cuda_availability()
    
    if not cuda_available:
        print("CUDA is not available. Optimized CNN requires GPU acceleration.")
        exit()
    
    # Ladda MNIST-data med augmentation
    train_loader, test_loader = get_mnist_data_loaders(batch_size=128)
    
    # Skapa CNN-modellen och flytta till GPU
    model = MNISTNetCNN(hidden_size=256)
    model.to(device)
    
    # Träna modellen med optimeringar
    print(f"Training optimized CNN model on {device} (5 epochs)...")
    trained_model, history, execution_time = train_model_gpu(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        learning_rate=0.001
    )
    
    # Final evaluation
    accuracy = evaluate_model_gpu(trained_model, test_loader, device)
    print(f"Test accuracy with optimized CNN: {accuracy:.4f}")
    print(f"Execution time on GPU: {execution_time:.2f} seconds") 