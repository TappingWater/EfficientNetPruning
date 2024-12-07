# utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import time
import os

def get_data_loaders(dataset_name='CIFAR10', data_dir='./data', batch_size=128, num_workers=8, pin_memory=True, valid_split=0.1, seed=42):
    """
    Create and return DataLoaders for training, validation, and testing.
    
    Args:
        dataset_name (str): Name of the dataset ('CIFAR10', 'ImageNet', etc.).
        data_dir (str): Directory where data is stored/downloaded.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        valid_split (float): Fraction of training data to use for validation.
        seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transforms
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std =[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])    
    # Load datasets
    if dataset_name == 'CIFAR10':
        full_train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                           download=True, transform=transform_train)
        
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                    download=True, transform=transform_val)
    elif dataset_name == 'ImageNet':
        full_train_dataset = torchvision.datasets.ImageNet(root=data_dir, split='train',
                                                           download=False, transform=transform_train)
        
        test_dataset = torchvision.datasets.ImageNet(root=data_dir, split='val',
                                                    download=False, transform=transform_val)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")    
    # Split training into train and validation
    train_size = int((1 - valid_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(seed)) 
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader


# Define loss function
criterion = nn.CrossEntropyLoss()


def train_epoch(model, device, train_loader, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run computations on.
        train_loader (DataLoader): DataLoader for the training set.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    elapsed = time.time() - start_time
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Time: {elapsed:.2f}s")
    return epoch_loss, epoch_acc

def validate_epoch(model, device, val_loader):
    """
    Validate the model for one epoch.
    
    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run computations on.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion: Loss function.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

def test_model(model, device, test_loader):
    """
    Test the model on the test set.
    
    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run computations on.
        test_loader (DataLoader): DataLoader for the test set.
        criterion: Loss function.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Test Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

def unfreeze_layers(model, layers_to_unfreeze):
    """
    Unfreeze specified layers in the model.
    
    Args:
        model (nn.Module): The neural network model.
        layers_to_unfreeze (list): List of layer name prefixes to unfreeze.
    """
    for name, param in model.named_parameters():
        for layer in layers_to_unfreeze:
            if name.startswith(layer):
                param.requires_grad = True
