# metrics.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
from thop import profile
from thop import clever_format

def get_model_size(model, path='temp.pth'):
    """
    Calculate the size of the model in megabytes (MB).

    Args:
        model (nn.Module): The PyTorch model.
        path (str): Temporary path to save the model.

    Returns:
        float: Size of the model in MB.
    """
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6  # Convert bytes to MB
    os.remove(path)
    return size

def measure_inference_time(model, device, input_size=(1, 3, 224, 224), iterations=100):
    """
    Measure the average inference time of the model.

    Args:
        model (nn.Module): The PyTorch model.
        device (torch.device): Device to run the model on.
        input_size (tuple): Size of the input tensor.
        iterations (int): Number of iterations to average.

    Returns:
        float: Average inference time in milliseconds.
    """
    model.eval()
    inputs = torch.randn(input_size).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            outputs = model(inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations * 1000  # Convert to ms
    return avg_time

def compute_flops(model, device, input_size=(1, 3, 224, 224)):
    """
    Compute the number of FLOPs (Floating Point Operations) of the model.

    Args:
        model (nn.Module): The PyTorch model.
        device (torch.device): Device to run the model on.
        input_size (tuple): Size of the input tensor.

    Returns:
        str: Formatted FLOPs value.
    """
    model.eval()
    inputs = torch.randn(input_size).to(device)
    try:
        macs, params = profile(model, inputs=(inputs, ))
        flops = 2 * macs  # Approximate FLOPs from MACs
        flops, params = clever_format([flops, params], "%.3f")
    except Exception as e:
        print(f"Error computing FLOPs: {e}")
        flops = "N/A"
    return flops

def get_memory_usage(model, device, input_size=(1, 3, 224, 224)):
    """
    Measure peak memory usage during a forward pass.

    Args:
        model (nn.Module): The PyTorch model.
        device (torch.device): Device to run the model on.
        input_size (tuple): Size of the input tensor.

    Returns:
        float or str: Peak memory usage in megabytes (MB), or 'N/A' if not CUDA.
    """
    if device.type != 'cuda':
        print("Memory usage monitoring is only available for CUDA devices.")
        return "N/A"
    model.eval()
    inputs = torch.randn(input_size).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        outputs = model(inputs)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1e6  # Convert to MB
    return peak_memory

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set and return accuracy and loss.

    Args:
        model (nn.Module): The PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion: Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_loss = running_loss / total
    test_accuracy = 100. * correct / total
    return test_loss, test_accuracy

def plot_metrics_table(metrics_dict, save_dir, model_name):
    """
    Generate and save a table image for the provided metrics.

    Args:
        metrics_dict (dict): Dictionary containing metric names and their values.
        save_dir (str): Directory to save the table image.
        model_name (str): Name of the model to display in the table.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define table data
    table_data = [
        ["Model Name", metrics_dict['model_name']],
        ["Pruning Ratio (%)", metrics_dict['pruning_ratio']],
        ["Pruning Type", metrics_dict['pruning_type']],
        ["Description", metrics_dict['description']],
        ["Model Size (MB)", f"{metrics_dict['model_size']:.2f}"],
        ["FLOPs", metrics_dict['flops']],
        ["Inference Time (ms)", f"{metrics_dict['inference_time']:.2f}"],
        ["Memory Usage (MB)", f"{metrics_dict['memory_usage']}" if metrics_dict['memory_usage'] != "N/A" else "N/A"],
        ["Test Loss", f"{metrics_dict['test_loss']:.4f}"],
        ["Test Accuracy (%)", f"{metrics_dict['test_accuracy']:.2f}%"]
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust size as needed
    ax.axis('off')
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust scaling as needed

    # Add title
    title = f"Model Metrics: {model_name}"
    if metrics_dict['pruning_ratio'] > 0 and metrics_dict['pruning_type']:
        title += f" | Pruned ({metrics_dict['pruning_ratio']}%, {metrics_dict['pruning_type']})"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Save the table image
    plt.savefig(os.path.join(save_dir, f"{model_name}_metrics_table.png"), bbox_inches='tight')
    plt.close()
    
    print(f"Metrics table saved in '{save_dir}' as '{model_name}_metrics_table.png'.")

def plot_training_validation_metrics(train_losses, train_accuracies, val_losses, val_accuracies, 
                                    save_dir, model_name, pruning_ratio=None, pruning_type=None):
    """
    Generate and save side-by-side plots for training and validation loss and accuracy.

    Args:
        train_losses (list): List of training loss values per epoch.
        train_accuracies (list): List of training accuracy values per epoch.
        val_losses (list): List of validation loss values per epoch.
        val_accuracies (list): List of validation accuracy values per epoch.
        save_dir (str): Directory to save the plots.
        model_name (str): Name of the model to display in the plot title.
        pruning_ratio (float, optional): Pruning ratio applied to the model.
        pruning_type (str, optional): Type of pruning applied to the model.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Two plots side by side
    
    # Plot Loss
    axs[0].plot(epochs, train_losses, label='Training Loss', color='blue')
    axs[0].plot(epochs, val_losses, label='Validation Loss', color='orange')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot Accuracy
    axs[1].plot(epochs, train_accuracies, label='Training Accuracy', color='green')
    axs[1].plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].legend()
    axs[1].grid(True)
    
    # Overall Title
    title = f"Training Metrics: {model_name}"
    if pruning_ratio is not None and pruning_type is not None:
        title += f" | Pruned ({pruning_ratio}%, {pruning_type})"
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plots
    plt.savefig(os.path.join(save_dir, f"{model_name}_training_validation_plots.png"))
    plt.close()
    
    print(f"Training and validation metrics plots saved in '{save_dir}' as '{model_name}_training_validation_plots.png'.")

def generate_and_save_metrics(model, device, test_loader, criterion, 
                              model_name='Model', pruning_ratio=0.0, pruning_type='', 
                              description='', train_losses=None, train_accuracies=None, 
                              val_losses=None, val_accuracies=None, save_dir='metrics_plots'):
    """
    Generate various metrics for the model, save a consolidated table image, and optionally plot training metrics.

    Args:
        model (nn.Module): The PyTorch model.
        device (torch.device): Device to run computations on.
        test_loader (DataLoader): DataLoader for the test set.
        criterion: Loss function.
        model_name (str): Name of the model.
        pruning_ratio (float): Pruning ratio applied to the model (0-100).
        pruning_type (str): Type of pruning applied to the model.
        description (str): Description of the model or pruning process.
        train_losses (list, optional): List of training loss values per epoch.
        train_accuracies (list, optional): List of training accuracy values per epoch.
        val_losses (list, optional): List of validation loss values per epoch.
        val_accuracies (list, optional): List of validation accuracy values per epoch.
        save_dir (str): Directory to save the plots.
    """
    print(f"--- Generating Metrics for {model_name} ---")
    
    print("Calculating Model Size...")
    model_size = get_model_size(model)
    
    print("Measuring Inference Time...")
    inference_time = measure_inference_time(model, device, input_size=(1,3,224,224), iterations=100)
    
    print("Computing FLOPs...")
    flops = compute_flops(model, device, input_size=(1,3,224,224))
    
    print("Measuring Memory Usage...")
    memory_usage = get_memory_usage(model, device, input_size=(1,3,224,224))
    
    # Evaluate model on test set
    print("Evaluating Model on Test Set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
    
    # Prepare metrics dictionary
    metrics = {
        'model_name': model_name,
        'pruning_ratio': pruning_ratio,
        'pruning_type': pruning_type,
        'description': description,
        'model_size': model_size,
        'flops': flops,
        'inference_time': inference_time,
        'memory_usage': memory_usage,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    print("Generating Metrics Table...")
    plot_metrics_table(metrics, save_dir=save_dir, model_name=model_name)
    
    # If training metrics are provided, plot them
    if all(v is not None for v in [train_losses, train_accuracies, val_losses, val_accuracies]):
        print("Generating Training and Validation Metrics Plots...")
        plot_training_validation_metrics(
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            save_dir=save_dir,
            model_name=model_name,
            pruning_ratio=pruning_ratio if pruning_ratio > 0 else None,
            pruning_type=pruning_type if pruning_type else None
        )
    
    print("--- Metrics Generation Completed ---\n")
