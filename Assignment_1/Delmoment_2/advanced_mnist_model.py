#!/usr/bin/env python3

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime
import uuid

# ===========================
# 1. MLOps Implementation
# ===========================

class MLOpsManager:
    def __init__(self, base_dir="models"):
        """
        Initialize a manager for MLOps best practices
        
        Args:
            base_dir: Directory to save model checkpoints and logs
        """
        self.base_dir = base_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.model_dir = os.path.join(base_dir, self.run_id)
        self.log_file = os.path.join(self.model_dir, "training_log.json")
        self.config_file = os.path.join(self.model_dir, "config.json")
        self.metrics = []
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def save_checkpoint(self, model, epoch, optimizer=None, metrics=None):
        """
        Save a model checkpoint
        
        Args:
            model: PyTorch model to save
            epoch: Current epoch number
            optimizer: Optional optimizer state
            metrics: Optional dictionary of metrics
        """
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_epoch_{epoch}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def save_best_model(self, model, metrics):
        """Save the best model based on validation accuracy"""
        best_model_path = os.path.join(self.model_dir, "best_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, best_model_path)
        print(f"Best model saved to {best_model_path}")
    
    def log_metrics(self, epoch, train_loss, test_accuracy, execution_time=None):
        """Log training metrics"""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_accuracy': test_accuracy,
            'timestamp': datetime.now().isoformat(),
        }
        
        if execution_time is not None:
            metrics['execution_time'] = execution_time
            
        self.metrics.append(metrics)
        
        # Write to log file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def save_config(self, config):
        """Save model configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {self.config_file}")


# ===========================
# 2. Data Augmentation
# ===========================

def get_augmented_mnist_data_loaders(batch_size=64, apply_augmentation=True):
    """
    Get MNIST data loaders with optional data augmentation
    
    Args:
        batch_size: Batch size for data loaders
        apply_augmentation: Whether to apply data augmentation
        
    Returns:
        train_loader, test_loader: Data loaders for training and testing
    """
    # Basic transformations for both train and test
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Advanced transformations for training (with augmentation)
    augmentation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),  # Random rotation up to 10 degrees
        transforms.RandomAffine(
            degrees=0,  # No additional rotation
            translate=(0.1, 0.1),  # Random translation up to 10%
            scale=(0.9, 1.1),  # Random scaling between 90% and 110%
        ),
        # Add some random noise
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        # Clip values to valid range [0, 1]
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Choose which transform to use for training
    train_transform = augmentation_transform if apply_augmentation else basic_transform
    
    # Load MNIST train and test datasets
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=basic_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ===========================
# 3. CNN Architecture
# ===========================

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.25, use_batch_norm=True):
        """
        Initialize a CNN for MNIST digit classification
        
        Args:
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third convolutional layer (additional layer)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate size after convolutions and pooling
        # Original: 28x28 -> Conv1 -> 28x28 -> Pool1 -> 14x14
        # -> Conv2 -> 14x14 -> Pool2 -> 7x7
        # -> Conv3 -> 7x7 -> Pool3 -> 3x3
        self.fc_input_size = 128 * 3 * 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(256, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# Define alternative CNN architectures for experimentation

class BasicCNN(nn.Module):
    """A more basic CNN with just two convolutional layers"""
    def __init__(self, dropout_rate=0.25, use_batch_norm=True):
        super(BasicCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2)
        
        # Size after convolutions: 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc1 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 64 * 7 * 7)
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class DeepCNN(nn.Module):
    """A deeper CNN with four convolutional layers"""
    def __init__(self, dropout_rate=0.25, use_batch_norm=True):
        super(DeepCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Fourth convolutional block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Fully connected layers
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# ===========================
# 4. Training function with regularization
# ===========================

def train_model(
    model, 
    train_loader, 
    test_loader, 
    device, 
    mlops_manager=None,
    epochs=10, 
    learning_rate=0.001,
    weight_decay=1e-5,  # L2 regularization
    checkpoint_interval=5
):
    """
    Train a PyTorch model with regularization and MLOps best practices
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to train on (GPU or CPU)
        mlops_manager: MLOps manager for logging and checkpoints
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay parameter for L2 regularization
        checkpoint_interval: Frequency of saving checkpoints (in epochs)
        
    Returns:
        Trained model, training history, and execution time
    """
    start_time = time.time()
    
    # Define loss function and optimizer with weight decay (L2 regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training history
    train_losses = []
    test_accuracies = []
    
    # Track best performance for model saving
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
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
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Time: {epoch_time:.2f}s")
        
        # Save checkpoint at specified intervals
        if mlops_manager and (epoch + 1) % checkpoint_interval == 0:
            metrics = {
                'train_loss': epoch_loss,
                'test_accuracy': test_accuracy
            }
            mlops_manager.save_checkpoint(model, epoch + 1, optimizer, metrics)
            mlops_manager.log_metrics(epoch + 1, epoch_loss, test_accuracy, epoch_time)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if mlops_manager:
                mlops_manager.save_best_model(model, {
                    'epoch': epoch + 1,
                    'accuracy': test_accuracy,
                    'loss': epoch_loss
                })
                print(f"New best model at epoch {epoch+1} with accuracy: {test_accuracy:.4f}")
    
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


# ===========================
# 5. Visualization utilities
# ===========================

def plot_training_results(history, model_name="Model", save_path=None):
    """
    Plot training loss and test accuracy
    
    Args:
        history: Dictionary with training losses and test accuracies
        model_name: Name of the model for the plot title
        save_path: Path to save the figure
    """
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(history["losses"])
    ax1.set_title(f"{model_name} - Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    # Plot test accuracy
    ax2.plot(history["accuracies"])
    ax2.set_title(f"{model_name} - Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training results plot saved to {save_path}")
    
    return fig


def visualize_filters(model, layer_idx=0, num_filters=16, save_path=None):
    """
    Visualize convolutional filters
    
    Args:
        model: Trained PyTorch model
        layer_idx: Index of the convolutional layer to visualize
        num_filters: Number of filters to display
        save_path: Path to save the figure
    """
    # Get the weights of the specified convolutional layer
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    
    if layer_idx >= len(conv_layers):
        print(f"Layer index {layer_idx} out of range. Model has {len(conv_layers)} convolutional layers.")
        return None
    
    weights = conv_layers[layer_idx].weight.data.cpu().numpy()
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    # Create a figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    # Plot filters
    for i in range(min(num_filters, weights.shape[0])):
        row, col = i // grid_size, i % grid_size
        
        # For the first layer, filters are 2D (in_channels=1)
        if weights.shape[1] == 1:
            img = weights[i, 0]
        else:
            # For deeper layers, take the mean across input channels
            img = np.mean(weights[i], axis=0)
        
        if grid_size > 1:
            ax = axes[row, col]
        else:
            ax = axes
        
        im = ax.imshow(img, cmap='viridis')
        ax.set_title(f"Filter {i+1}")
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_filters, grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        if grid_size > 1:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Filter visualization saved to {save_path}")
    
    return fig


def compare_models(model_results, save_path=None):
    """
    Compare performance of multiple models
    
    Args:
        model_results: Dictionary mapping model names to their results
        save_path: Path to save the figure
    """
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot training loss for each model
    for model_name, results in model_results.items():
        ax1.plot(results["history"]["losses"], label=model_name)
    
    ax1.set_title("Training Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # Plot test accuracy for each model
    for model_name, results in model_results.items():
        ax2.plot(results["history"]["accuracies"], label=model_name)
    
    ax2.set_title("Test Accuracy Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")
    
    return fig


# ===========================
# 6. Hyperparameter Tuning
# ===========================

def hyperparameter_tuning(
    model_class,
    train_loader,
    test_loader,
    device,
    param_grid,
    epochs=5,
    base_dir="models/hyperparameter_tuning"
):
    """
    Simple grid search for hyperparameter tuning
    
    Args:
        model_class: PyTorch model class to tune
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to train on (GPU or CPU)
        param_grid: Dictionary of hyperparameter combinations to try
        epochs: Number of epochs for each run
        base_dir: Directory to save results
        
    Returns:
        Best model and performance metrics
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Results tracking
    results = []
    best_accuracy = 0.0
    best_model = None
    best_params = None
    
    # Create parameter combinations
    import itertools
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Create experiment log file
    exp_log_path = os.path.join(base_dir, "experiment_log.json")
    
    # Run each combination
    for i, combination in enumerate(param_combinations):
        params = dict(zip(param_keys, combination))
        print(f"\nHyperparameter Combination {i+1}/{len(param_combinations)}:")
        print(json.dumps(params, indent=2))
        
        # Create model with current parameters
        model = model_class(**{k: v for k, v in params.items() 
                              if k in ['dropout_rate', 'use_batch_norm']})
        model.to(device)
        
        # Create MLOps manager for this run
        run_dir = os.path.join(base_dir, f"run_{i+1}")
        mlops_manager = MLOpsManager(base_dir=run_dir)
        mlops_manager.save_config(params)
        
        # Train model
        trained_model, history, execution_time = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            mlops_manager=mlops_manager,
            epochs=epochs,
            learning_rate=params.get('learning_rate', 0.001),
            weight_decay=params.get('weight_decay', 1e-5)
        )
        
        # Evaluate final performance
        final_accuracy = history["accuracies"][-1]
        
        # Save results
        run_result = {
            "params": params,
            "accuracy": final_accuracy,
            "train_loss": history["losses"][-1],
            "execution_time": execution_time,
            "run_id": mlops_manager.run_id
        }
        results.append(run_result)
        
        # Update experiment log
        with open(exp_log_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Check if this is the best model so far
        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_model = trained_model
            best_params = params
            print(f"New best model! Accuracy: {best_accuracy:.4f}")
        
        # Create and save performance plot
        plot_training_results(
            history,
            model_name=f"Run {i+1}",
            save_path=os.path.join(run_dir, "training_plot.png")
        )
    
    # Save best model
    best_model_path = os.path.join(base_dir, "best_model.pt")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'params': best_params,
        'accuracy': best_accuracy
    }, best_model_path)
    
    print(f"\nHyperparameter tuning complete!")
    print(f"Best model accuracy: {best_accuracy:.4f}")
    print(f"Best parameters: {json.dumps(best_params, indent=2)}")
    print(f"Best model saved to {best_model_path}")
    
    return best_model, best_params, results


# ===========================
# 7. Main function
# ===========================

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


def main():
    """Main function to run the entire pipeline"""
    print("\n=== MNIST CNN with MLOps, Data Augmentation, and Regularization ===")
    
    # Check CUDA availability
    cuda_available, device = check_cuda_availability()
    
    # Define configuration
    config = {
        "models_dir": "models",
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "dropout_rate": 0.25,
        "use_batch_norm": True,
        "use_data_augmentation": True,
        "device": str(device)
    }
    
    # Create MLOps manager
    mlops_manager = MLOpsManager(base_dir=config["models_dir"])
    mlops_manager.save_config(config)
    
    print("\n1. Loading data with augmentation...")
    train_loader, test_loader = get_augmented_mnist_data_loaders(
        batch_size=config["batch_size"],
        apply_augmentation=config["use_data_augmentation"]
    )
    
    # First experiment: Compare basic MLP vs CNN
    print("\n2. Comparing MLP vs CNN architectures...")
    
    # Original MLP model
    class MNISTNet(nn.Module):
        def __init__(self, hidden_size=128):
            super(MNISTNet, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28 * 28, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, 10)
            
        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # Train MLP model
    mlp_model = MNISTNet(hidden_size=128).to(device)
    mlp_model, mlp_history, mlp_time = train_model(
        model=mlp_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5  # Fewer epochs for comparison
    )
    
    # Train basic CNN model
    cnn_model = BasicCNN(
        dropout_rate=config["dropout_rate"],
        use_batch_norm=config["use_batch_norm"]
    ).to(device)
    
    cnn_model, cnn_history, cnn_time = train_model(
        model=cnn_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5  # Fewer epochs for comparison
    )
    
    # Compare MLP vs CNN
    model_results = {
        "MLP": {"history": mlp_history, "time": mlp_time},
        "CNN": {"history": cnn_history, "time": cnn_time}
    }
    
    compare_models(model_results, save_path=os.path.join(mlops_manager.model_dir, "mlp_vs_cnn.png"))
    
    # Second experiment: Train the full CNN model with regularization and MLOps
    print("\n3. Training CNN with regularization and MLOps...")
    
    # Create and train the full CNN model
    full_cnn_model = SimpleCNN(
        dropout_rate=config["dropout_rate"],
        use_batch_norm=config["use_batch_norm"]
    ).to(device)
    
    full_cnn_model, full_history, full_time = train_model(
        model=full_cnn_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        mlops_manager=mlops_manager,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        checkpoint_interval=2
    )
    
    # Plot training results
    plot_training_results(
        full_history,
        model_name="CNN with Regularization",
        save_path=os.path.join(mlops_manager.model_dir, "training_results.png")
    )
    
    # Third experiment (optional): Hyperparameter tuning
    if config.get("run_hyperparameter_tuning", False):
        print("\n4. Running hyperparameter tuning...")
        
        param_grid = {
            "dropout_rate": [0.1, 0.25, 0.5],
            "learning_rate": [0.0001, 0.001, 0.01],
            "weight_decay": [1e-6, 1e-5, 1e-4],
            "use_batch_norm": [True, False]
        }
        
        best_model, best_params, tuning_results = hyperparameter_tuning(
            model_class=SimpleCNN,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            param_grid=param_grid,
            epochs=5,
            base_dir=os.path.join(config["models_dir"], "hyperparameter_tuning")
        )
    
    # Visualize convolutional filters
    print("\n5. Visualizing convolutional filters...")
    visualize_filters(
        full_cnn_model,
        layer_idx=0,  # First convolutional layer
        num_filters=16,
        save_path=os.path.join(mlops_manager.model_dir, "conv1_filters.png")
    )
    
    visualize_filters(
        full_cnn_model,
        layer_idx=1,  # Second convolutional layer
        num_filters=16,
        save_path=os.path.join(mlops_manager.model_dir, "conv2_filters.png")
    )
    
    # Final evaluation
    final_accuracy = evaluate_model(full_cnn_model, test_loader, device)
    print(f"\nFinal model accuracy: {final_accuracy:.4f}")
    print(f"Training time: {full_time:.2f} seconds")
    print(f"Model and results saved to {mlops_manager.model_dir}")
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()