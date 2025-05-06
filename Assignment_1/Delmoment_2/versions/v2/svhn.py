#!/usr/bin/env python3

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime
import uuid

# ===========================
# Colab-specifika konfigurationer
# ===========================

def setup_colab():
    """
    Konfigurerar skriptet för Google Colab-miljön
    
    Returns:
        bool: True om kör på Colab, False annars
        str: Sökväg för att spara modeller
        str: Sökväg för TensorBoard loggar
    """
    try:
        import google.colab
        is_colab = True
        print("Kör i Google Colab-miljö!")
        
        # Konfigurera för att använda GPU
        print("\nKontrollerar GPU-tillgänglighet...")
        try:
            # Kör nvidia-smi för att visa GPU-information
            !nvidia-smi
        except:
            print("Kunde inte köra nvidia-smi. GPU kanske inte är aktiverad.")
        
        # Montera Google Drive för att spara resultat
        from google.colab import drive
        print("\nMonterar Google Drive för att spara modeller och resultat...")
        drive.mount('/content/drive')
        models_dir = "/content/drive/MyDrive/svhn_models_resnet"
        os.makedirs(models_dir, exist_ok=True)
        print(f"Resultat kommer att sparas till: {models_dir}")
            
        # Konfigurera TensorBoard för visualisering
        try:
            print("\nKonfigurerar TensorBoard...")
            %load_ext tensorboard
            import torch.utils.tensorboard as tb
            tb_logdir = f"{models_dir}/tensorboard"
            os.makedirs(tb_logdir, exist_ok=True)
            print(f"TensorBoard loggar sparas i: {tb_logdir}")
            
            # Visa TensorBoard i Colab
            %tensorboard --logdir={tb_logdir}
            
            return True, models_dir, tb_logdir
        except:
            print("Kunde inte konfigurera TensorBoard, fortsätter utan det.")
            return True, models_dir, None
        
    except ImportError:
        print("Kör inte i Google Colab-miljö")
        return False, "models/svhn_resnet", None


# ===========================
# 1. MLOps Implementation
# ===========================

class MLOpsManager:
    def __init__(self, base_dir="models", tb_writer=None):
        """
        Initialize a manager for MLOps best practices
        
        Args:
            base_dir: Directory to save model checkpoints and logs
            tb_writer: TensorBoard SummaryWriter (optional)
        """
        self.base_dir = base_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.model_dir = os.path.join(base_dir, self.run_id)
        self.log_file = os.path.join(self.model_dir, "training_log.json")
        self.config_file = os.path.join(self.model_dir, "config.json")
        self.metrics = []
        self.tb_writer = tb_writer
        
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
        
        # Log to TensorBoard if available
        if self.tb_writer:
            self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
            self.tb_writer.add_scalar('Accuracy/val', test_accuracy, epoch)
            if execution_time:
                self.tb_writer.add_scalar('Time/epoch', execution_time, epoch)
    
    def save_config(self, config):
        """Save model configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {self.config_file}")


# ===========================
# 2. Data Augmentation and Loading for SVHN
# ===========================

def get_augmented_svhn_data_loaders(batch_size=64, apply_augmentation=True, val_split=0.1):
    """
    Get SVHN data loaders with optional data augmentation
    
    Args:
        batch_size: Batch size for data loaders
        apply_augmentation: Whether to apply data augmentation
        val_split: Proportion of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader: Data loaders
    """
    # Basic transformations for both train and test
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])
    
    # Advanced transformations for training (with augmentation)
    augmentation_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,  # No additional rotation
            translate=(0.1, 0.1),  # Random translation up to 10%
            scale=(0.9, 1.1),  # Random scaling between 90% and 110%
        ),
        transforms.ToTensor(),
        # Add some random noise
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        # Clip values to valid range [-1, 1]
        transforms.Lambda(lambda x: torch.clamp(x, -1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Choose which transform to use for training
    train_transform = augmentation_transform if apply_augmentation else basic_transform
    
    # Load SVHN datasets
    print("Downloading and loading SVHN dataset...")
    train_dataset = datasets.SVHN(
        root='./data', split='train', download=True, transform=train_transform
    )
    
    test_dataset = datasets.SVHN(
        root='./data', split='test', download=True, transform=basic_transform
    )
    
    # Create validation split from training data
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Override the transform for validation set
    val_dataset.dataset.transform = basic_transform
    
    # Create data loaders
    # For Colab: decrease num_workers if there are issues
    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def show_svhn_samples(dataloader, num_samples=8, save_path=None):
    """
    Display sample images from the SVHN dataset
    
    Args:
        dataloader: DataLoader containing SVHN data
        num_samples: Number of samples to display
        save_path: Path to save the figure
    """
    # Get a batch of training data
    images, labels = next(iter(dataloader))
    
    # Convert tensors to numpy arrays
    images = images.numpy()
    
    # Create a grid of images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        img = np.transpose(images[i], (1, 2, 0))  # CHW -> HWC
        img = (img * 0.5) + 0.5  # Unnormalize to [0, 1]
        img = np.clip(img, 0, 1)  # Ensure values are in valid range
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Sample images saved to {save_path}")
    
    # För Colab: visa direkt i notebook
    try:
        from IPython.display import display
        display(fig)
    except:
        pass
    
    return fig


# ===========================
# 3. ResNet Architecture for SVHN
# ===========================

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity  # This is the residual connection
        out = self.relu(out)
        
        return out


class SVHNResNet(nn.Module):
    """ResNet architecture for SVHN classification"""
    def __init__(self, block, layers, num_classes=10, dropout_rate=0.2):
        super(SVHNResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


def resnet20(dropout_rate=0.2):
    """ResNet-20 model for SVHN"""
    return SVHNResNet(ResidualBlock, [3, 3, 3], dropout_rate=dropout_rate)


def resnet32(dropout_rate=0.2):
    """ResNet-32 model for SVHN"""
    return SVHNResNet(ResidualBlock, [5, 5, 5], dropout_rate=dropout_rate)


def resnet44(dropout_rate=0.2):
    """ResNet-44 model for SVHN"""
    return SVHNResNet(ResidualBlock, [7, 7, 7], dropout_rate=dropout_rate)


# ===========================
# 4. Training function with regularization and recovery
# ===========================

def load_checkpoint_and_resume(checkpoint_path, model, optimizer=None, device=None):
    """
    Ladda en sparad checkpoint och återuppta träning
    
    Args:
        checkpoint_path: Sökväg till checkpointen
        model: Modell att ladda vikter till
        optimizer: Valfri optimizer att återställa tillstånd för
        device: Enhet att ladda modellen på
        
    Returns:
        epoch: Senast avslutade epoch
        model: Modell med laddade vikter
        optimizer: Optimizer med återställt tillstånd (om tillhandahållen)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Ingen checkpoint hittades på {checkpoint_path}")
        return 0, model, optimizer
    
    print(f"Laddar checkpoint från {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Återupptar från epoch {epoch}")
    return epoch, model, optimizer


def train_model(
    model, 
    train_loader, 
    val_loader, 
    device, 
    mlops_manager=None,
    epochs=10, 
    learning_rate=0.001,
    weight_decay=1e-5,  # L2 regularization
    checkpoint_interval=2,
    resume_from=None  # Möjlighet att återuppta från specifik checkpoint
):
    """
    Train a PyTorch model with regularization and MLOps best practices
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (GPU or CPU)
        mlops_manager: MLOps manager for logging and checkpoints
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay parameter for L2 regularization
        checkpoint_interval: Frequency of saving checkpoints (in epochs)
        resume_from: Path to checkpoint to resume from (optional)
        
    Returns:
        Trained model, training history, and execution time
    """
    start_time = time.time()
    
    # Define loss function and optimizer with weight decay (L2 regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        start_epoch, model, optimizer = load_checkpoint_and_resume(
            resume_from, model, optimizer, device
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    train_losses = []
    val_accuracies = []
    learning_rates = []
    
    # Track best performance for model saving
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(start_epoch, epochs):
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
        
        # Evaluate on validation set
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate scheduler
        scheduler.step(val_accuracy)
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Save checkpoint at specified intervals (always in Colab due to potential timeouts)
        if mlops_manager:
            if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
                metrics = {
                    'train_loss': epoch_loss,
                    'val_accuracy': val_accuracy,
                    'learning_rate': current_lr
                }
                mlops_manager.save_checkpoint(model, epoch + 1, optimizer, metrics)
                mlops_manager.log_metrics(epoch + 1, epoch_loss, val_accuracy, epoch_time)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if mlops_manager:
                mlops_manager.save_best_model(model, {
                    'epoch': epoch + 1,
                    'accuracy': val_accuracy,
                    'loss': epoch_loss
                })
                print(f"New best model at epoch {epoch+1} with accuracy: {val_accuracy:.4f}")
    
    execution_time = time.time() - start_time
    
    return model, {
        "losses": train_losses, 
        "accuracies": val_accuracies,
        "learning_rates": learning_rates
    }, execution_time


def evaluate_model(model, data_loader, device):
    """Evaluate a PyTorch model on validation or test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in data_loader:
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
    Plot training loss, validation accuracy, and learning rate
    
    Args:
        history: Dictionary with training metrics
        model_name: Name of the model for the plot title
        save_path: Path to save the figure
    """
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot training loss
    ax1.plot(history["losses"])
    ax1.set_title(f"{model_name} - Training Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(history["accuracies"])
    ax2.set_title(f"{model_name} - Validation Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    
    # Plot learning rate
    if "learning_rates" in history:
        ax3.plot(history["learning_rates"])
        ax3.set_title(f"{model_name} - Learning Rate")
        ax3.set_ylabel("Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_yscale("log")
        ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training results plot saved to {save_path}")
    
    # För Colab: visa direkt i notebook
    try:
        from IPython.display import display
        display(fig)
    except:
        pass
    
    return fig


def visualize_feature_maps(model, sample_image, layer_names=None, save_path=None):
    """
    Visualize feature maps from different layers of the model
    
    Args:
        model: Trained PyTorch model
        sample_image: Single image tensor to visualize (should be unsqueezed to batch dim)
        layer_names: List of layer names to visualize (if None, use defaults)
        save_path: Path to save the figure
    """
    # Default layer names if none provided
    if layer_names is None:
        layer_names = ['layer1', 'layer2', 'layer3']
    
    # Create hooks to extract feature maps
    feature_maps = {}
    
    def get_features(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    for name in layer_names:
        layer = getattr(model, name, None)
        if layer is not None:
            hooks.append(layer.register_forward_hook(get_features(name)))
    
    # Forward pass with the sample image
    model.eval()
    with torch.no_grad():
        _ = model(sample_image.unsqueeze(0).cuda())
    
    # Remove the hooks
    for hook in hooks:
        hook.remove()
    
    # Create figure to display feature maps
    fig = plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(layer_names):
        if name not in feature_maps:
            continue
            
        # Get the feature maps for this layer
        fmap = feature_maps[name]
        
        # For each layer, show up to 8 feature maps
        num_maps = min(8, fmap.shape[1])
        
        for j in range(num_maps):
            # Calculate subplot position
            plt.subplot(len(layer_names), num_maps, i * num_maps + j + 1)
            
            # Get the feature map
            feat = fmap[0, j].numpy()
            
            # Normalize for better visualization
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
            
            # Display feature map
            plt.imshow(feat, cmap='viridis')
            plt.title(f"{name} - Filter {j+1}")
            plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature maps visualization saved to {save_path}")
    
    # För Colab: visa direkt i notebook
    try:
        from IPython.display import display
        display(fig)
    except:
        pass
    
    return fig


def visualize_misclassifications(model, test_loader, device, num_examples=10, save_path=None):
    """
    Visualize examples of misclassified images
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run inference on
        num_examples: Number of misclassified examples to show
        save_path: Path to save the figure
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples
            incorrect_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in incorrect_idx:
                misclassified_images.append(images[idx].cpu())
                misclassified_labels.append(labels[idx].item())
                predicted_labels.append(preds[idx].item())
                
                if len(misclassified_images) >= num_examples:
                    break
            
            if len(misclassified_images) >= num_examples:
                break
    
    # Convert to numpy arrays for plotting
    misclassified_images = [img.numpy() for img in misclassified_images[:num_examples]]
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (img, true_label, pred_label) in enumerate(zip(
        misclassified_images, misclassified_labels[:num_examples], predicted_labels[:num_examples]
    )):
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        img = (img * 0.5) + 0.5  # Unnormalize
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Misclassification examples saved to {save_path}")
    
    # För Colab: visa direkt i notebook
    try:
        from IPython.display import display
        display(fig)
    except:
        pass
    
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
    ax1.grid(True)
    
    # Plot validation accuracy for each model
    for model_name, results in model_results.items():
        ax2.plot(results["history"]["accuracies"], label=model_name)
    
    ax2.set_title("Validation Accuracy Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")
    
    # För Colab: visa direkt i notebook
    try:
        from IPython.display import display
        display(fig)
    except:
        pass
    
    return fig


# ===========================
# 6. Hyperparameter Tuning
# ===========================

def hyperparameter_tuning(
    model_classes,
    train_loader,
    val_loader,
    device,
    param_grid,
    epochs=5,
    base_dir="models/hyperparameter_tuning",
    tb_writer=None
):
    """
    Simple grid search for hyperparameter tuning across different model architectures
    
    Args:
        model_classes: Dictionary mapping model names to their class constructors
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (GPU or CPU)
        param_grid: Dictionary of hyperparameter combinations to try
        epochs: Number of epochs for each run
        base_dir: Directory to save results
        tb_writer: TensorBoard SummaryWriter (optional)
        
    Returns:
        Best model, model name, and performance metrics
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Results tracking
    results = []
    best_accuracy = 0.0
    best_model = None
    best_model_name = None
    best_params = None
    
    # Create parameter combinations - common parameters
    import itertools
    shared_param_keys = [k for k in param_grid.keys() if k != 'model_specific']
    shared_param_values = [param_grid[k] for k in shared_param_keys]
    shared_param_combinations = list(itertools.product(*shared_param_values))
    
    # Create experiment log file
    exp_log_path = os.path.join(base_dir, "experiment_log.json")
    
    # Run for each model type
    for model_name, model_class in model_classes.items():
        print(f"\n===== Testing {model_name} architecture =====")
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Get model-specific parameters if any
        model_specific_params = param_grid.get('model_specific', {}).get(model_name, {})
        model_param_keys = list(model_specific_params.keys())
        
        if model_param_keys:
            model_param_values = [model_specific_params[k] for k in model_param_keys]
            model_param_combinations = list(itertools.product(*model_param_values))
        else:
            # If no model-specific params, create a dummy combination
            model_param_combinations = [tuple()]
            model_param_keys = []
            
        # Iterate through all combinations of shared and model-specific parameters
        for shared_idx, shared_combo in enumerate(shared_param_combinations):
            shared_params = dict(zip(shared_param_keys, shared_combo))
            
            for model_idx, model_combo in enumerate(model_param_combinations):
                model_params = dict(zip(model_param_keys, model_combo)) if model_param_keys else {}
                
                # Combine parameters
                combined_params = {**shared_params, **model_params}
                run_id = f"{shared_idx+1}_{model_idx+1}"
                
                print(f"\nTesting {model_name} with parameters (Run {run_id}):")
                print(json.dumps(combined_params, indent=2))
                
                # Create model with current parameters
                model_kwargs = {k: v for k, v in combined_params.items() 
                               if k in ['dropout_rate']}  # Only pass relevant parameters to model
                
                model = model_class(**model_kwargs)
                model.to(device)
                
                # Create MLOps manager for this run
                run_dir = os.path.join(model_dir, f"run_{run_id}")
                mlops_manager = MLOpsManager(base_dir=run_dir, tb_writer=tb_writer)
                mlops_manager.save_config(combined_params)
                
                # Train model
                trained_model, history, execution_time = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    mlops_manager=mlops_manager,
                    epochs=epochs,
                    learning_rate=combined_params.get('learning_rate', 0.001),
                    weight_decay=combined_params.get('weight_decay', 1e-5),
                    checkpoint_interval=2  # För Colab: spara oftare
                )
                
                # Evaluate final performance
                final_accuracy = history["accuracies"][-1]
                
                # Save results
                run_result = {
                    "model": model_name,
                    "params": combined_params,
                    "accuracy": final_accuracy,
                    "train_loss": history["losses"][-1],
                    "execution_time": execution_time,
                    "run_id": f"{model_name}/{run_id}"
                }
                results.append(run_result)
                
                # Update experiment log
                with open(exp_log_path, 'w') as f:
                    json.dump(results, f, indent=4)
                
                # Check if this is the best model so far
                if final_accuracy > best_accuracy:
                    best_accuracy = final_accuracy
                    best_model = trained_model
                    best_model_name = model_name
                    best_params = combined_params
                    print(f"New best model! {model_name} with accuracy: {best_accuracy:.4f}")
                
                # Create and save performance plot
                plot_training_results(
                    history,
                    model_name=f"{model_name} - Run {run_id}",
                    save_path=os.path.join(run_dir, "training_plot.png")
                )
    
    # Save best model
    best_model_path = os.path.join(base_dir, "best_model.pt")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_name': best_model_name,
        'params': best_params,
        'accuracy': best_accuracy
    }, best_model_path)
    
    # Create a summary results visualization
    create_hyperparameter_tuning_summary(results, save_path=os.path.join(base_dir, "tuning_summary.png"))
    
    print(f"\nHyperparameter tuning complete!")
    print(f"Best model: {best_model_name}")
    print(f"Best model accuracy: {best_accuracy:.4f}")
    print(f"Best parameters: {json.dumps(best_params, indent=2)}")
    print(f"Best model saved to {best_model_path}")
    
    return best_model, best_model_name, best_params, results


def create_hyperparameter_tuning_summary(results, save_path=None):
    """
    Create a visualization summarizing hyperparameter tuning results
    
    Args:
        results: List of results from hyperparameter tuning
        save_path: Path to save the figure
    """
    # Extract model names and sort results
    model_names = list(set(r["model"] for r in results))
    model_results = {name: [] for name in model_names}
    
    for result in results:
        model_results[result["model"]].append(result)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot accuracy by model
    model_accuracies = []
    model_labels = []
    for model_name, model_runs in model_results.items():
        accuracies = [run["accuracy"] for run in model_runs]
        model_accuracies.append(accuracies)
        model_labels.append(model_name)
    
    # Box plot of model accuracies
    axes[0].boxplot(model_accuracies, labels=model_labels)
    axes[0].set_title("Validation Accuracy by Model Architecture")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot execution times
    model_times = []
    for model_name, model_runs in model_results.items():
        times = [run["execution_time"] for run in model_runs]
        model_times.append(times)
    
    # Box plot of execution times
    axes[1].boxplot(model_times, labels=model_labels)
    axes[1].set_title("Training Time by Model Architecture")
    axes[1].set_ylabel("Time (seconds)")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Display best results as text
    best_result = max(results, key=lambda x: x["accuracy"])
    best_text = (f"Best Model: {best_result['model']}\n"
                f"Accuracy: {best_result['accuracy']:.4f}\n"
                f"Parameters: {json.dumps(best_result['params'], indent=2)}")
    
    fig.text(0.5, 0.01, best_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path)
        print(f"Hyperparameter tuning summary saved to {save_path}")
    
    # För Colab: visa direkt i notebook
    try:
        from IPython.display import display
        display(fig)
    except:
        pass
    
    return fig


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
    print("\n=== SVHN ResNet med Hyperparameter-tuning ===")
    
    # Konfigurera för Colab om det behövs
    is_colab, models_dir, tb_logdir = setup_colab()
    
    # TensorBoard writer för Colab
    tb_writer = None
    if tb_logdir:
        try:
            import torch.utils.tensorboard as tb
            tb_writer = tb.SummaryWriter(tb_logdir)
        except:
            print("Kunde inte skapa TensorBoard writer.")
    
    # Check CUDA availability
    cuda_available, device = check_cuda_availability()
    
    # Define configuration
    config = {
        "models_dir": models_dir,  # Från Colab-konfigurationen om det körs i Colab
        "batch_size": 128,
        "epochs": 15,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "dropout_rate": 0.2,  # Lower dropout for ResNet
        "device": str(device),
        "run_hyperparameter_tuning": True  # Genomför hyperparameter-tuning
    }
    
    # Create MLOps manager
    mlops_manager = MLOpsManager(base_dir=config["models_dir"], tb_writer=tb_writer)
    mlops_manager.save_config(config)
    
    print("\n1. Loading data with augmentation...")
    train_loader, val_loader, test_loader = get_augmented_svhn_data_loaders(
        batch_size=config["batch_size"],
        apply_augmentation=config["use_data_augmentation"] if "use_data_augmentation" in config else True
    )
    
    # Show sample images
    show_svhn_samples(
        train_loader, 
        num_samples=8, 
        save_path=os.path.join(mlops_manager.model_dir, "sample_images.png")
    )
    
    if config["run_hyperparameter_tuning"]:
        print("\n2. Running hyperparameter tuning for ResNet architectures...")
        
        # Define models to test
        model_classes = {
            "ResNet20": resnet20,
            "ResNet32": resnet32
        }
        
        # Define parameter grid
        param_grid = {
            "dropout_rate": [0.1, 0.2, 0.3],
            "learning_rate": [0.0005, 0.001, 0.002],
            "weight_decay": [1e-6, 1e-5, 1e-4],
            # Model-specific parameters
            "model_specific": {
                # If needed, you can add model-specific params here
            }
        }
        
        # Run hyperparameter tuning
        best_model, best_model_name, best_params, tuning_results = hyperparameter_tuning(
            model_classes=model_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            param_grid=param_grid,
            epochs=5,  # Fewer epochs for initial tuning
            base_dir=os.path.join(config["models_dir"], "hyperparameter_tuning"),
            tb_writer=tb_writer
        )
        
        print("\n3. Training best model configuration with full epochs...")
        
        # Create model with best parameters
        if best_model_name == "ResNet20":
            full_model = resnet20(dropout_rate=best_params.get("dropout_rate", 0.2))
        elif best_model_name == "ResNet32":
            full_model = resnet32(dropout_rate=best_params.get("dropout_rate", 0.2))
        else:
            full_model = resnet20(dropout_rate=0.2)  # Default
            
        full_model.to(device)
        
        # Train with full epochs
        full_model, full_history, full_time = train_model(
            model=full_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            mlops_manager=mlops_manager,
            epochs=config["epochs"],
            learning_rate=best_params.get("learning_rate", 0.001),
            weight_decay=best_params.get("weight_decay", 1e-5),
            checkpoint_interval=1  # Spara varje epok för säkerhets skull
        )
        
    else:
        print("\n2. Training ResNet20 model...")
        
        # Just train ResNet20 if not doing hyperparameter tuning
        resnet_model = resnet20(dropout_rate=config["dropout_rate"])
        resnet_model.to(device)
        
        full_model, full_history, full_time = train_model(
            model=resnet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            mlops_manager=mlops_manager,
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            checkpoint_interval=1
        )
    
    # Plot training results
    plot_training_results(
        full_history,
        model_name="ResNet (Final Model)",
        save_path=os.path.join(mlops_manager.model_dir, "training_results.png")
    )
    
    # Evaluate on test set
    test_accuracy = evaluate_model(full_model, test_loader, device)
    print(f"ResNet Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize feature maps
    print("\n4. Visualizing feature maps...")
    # Get a sample image for visualization
    sample_image, _ = next(iter(val_loader))
    sample_image = sample_image[0].to(device)
    
    visualize_feature_maps(
        full_model,
        sample_image,
        save_path=os.path.join(mlops_manager.model_dir, "feature_maps.png")
    )
    
    # Visualize misclassifications
    print("\n5. Visualizing misclassified examples...")
    visualize_misclassifications(
        full_model,
        test_loader,
        device,
        num_examples=10,
        save_path=os.path.join(mlops_manager.model_dir, "misclassified_examples.png")
    )
    
    # Final summary
    print("\n=== Training Summary ===")
    print(f"ResNet Model:")
    print(f"  Final Test accuracy: {test_accuracy:.4f}")
    print(f"  Training time: {full_time:.2f} seconds")
    
    print(f"\nModel and results saved to {mlops_manager.model_dir}")
    
    # För Colab: visa hur man kan titta på resultaten igen
    if is_colab and "drive" in models_dir:
        print("\nFör att titta på resultaten i en framtida Colab-session:")
        print("from IPython.display import Image")
        print(f"Image('{mlops_manager.model_dir}/training_results.png')")
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()