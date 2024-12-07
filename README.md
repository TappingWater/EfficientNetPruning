# Deep Learning Model Sparsification for Efficient Inference
**Chanaka Perera**

---

## Introduction

This project explores how to prune a deep learning model. Model training and evaluation were conducted on a CUDA-enabled device to speed up training. Additionally, for faster experimentation, I decided to use a lightweight model adapted to a smaller dataset as the base model for pruning. For this experiment, I used a pretrained EfficientNet-B0 model available in the PyTorch modules trained on ImageNet and adapted it to the CIFAR-10 dataset.

---

## EfficientNet-B0

**EfficientNet-B0** is a model that can be loaded easily by using PyTorch, trained on the ImageNet dataset. This model builds on traditional convolutional neural networks (CNNs) by enhancing performance and efficiency. Traditional CNNs scale by increasing depth (number of layers), width (number of channels), or resolution (input image size). However, this approach may lead to poor improvements in accuracy relative to the increased computational cost. EfficientNet uses a set of predefined scaling coefficients to scale all dimensions in a balanced manner. By proportionally increasing all dimensions, this model maintains feature representations well without too much computational overhead.

### Key Features of EfficientNet-B0

1. **Balanced Scaling:** EfficientNet scales depth, width, and resolution uniformly using compound scaling.
2. **Depthwise Separable Convolutions:** Reduces computational cost compared to standard convolutions.
3. **Squeeze-and-Excitation (SE) Modules:** Enhances feature representations by modeling interdependencies between channels.

### Depthwise and Pointwise Convolutions

EfficientNet consists of two primary operations:

1. **Depthwise Convolutions:**
   - **Function:** Applies a single filter per input channel.
   - **Benefit:** Reduces computational overhead compared to traditional CNNs that apply filters across all input channels simultaneously.
   - **Example:**
     - **Input:** 3 channels (R, G, B)
     - **Filters:** 3 separate 3x3 filters (one for each channel)
     - **Output:** 3 feature maps

2. **Pointwise Convolutions:**
   - **Function:** Combines filtered outputs to the desired number of output channels.
   - **Benefit:** Allows the model to learn information across different channels.
   - **Example:**
     - **Input:** 3 feature maps from depthwise convolution
     - **Filters:** 32 separate 1x1 filters
     - **Output:** 32 feature maps

### Squeeze-and-Excitation (SE) Modules

SE modules come after pointwise convolutions to prioritize more important features.

- **Squeeze Operation:**
  - **Function:** Global Information Aggregation by computing a single summary statistic (e.g., global average pooling) for each feature map (channel).
  - **Output:** A vector representing the importance of each channel.

- **Excitation Operation:**
  - **Function:** Generates adaptive weights for each channel by passing the squeezed vector through a small neural network (usually two fully connected layers with a non-linear activation in between).
  - **Output:** Scaled feature maps where each feature map is multiplied by its corresponding weight, enhancing important features and diminishing less relevant ones.

### Network Architecture Search

EfficientNet employs an **automated search method** for identifying the best configuration by exploring different search spaces. It uses algorithms such as reinforcement learning to find an effective configuration that maximizes performance while minimizing computational cost. This approach helps identify architectural patterns that are not as intuitive to humans.

---

## Dataset: CIFAR-10

- **CIFAR-10 Images:** Each image is 32x32 with RGB color channels.
- **Total Classification Classes:** 10 categories
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

---

## Training and Fine-Tuning a Base Model

### Pre-processing the CIFAR-10 Dataset

```python
from torch.utils.data import random_split

# Since the CIFAR-10 dataset is different from ImageNet
# We need to perform data pre-processing before training
transform_train = transforms.Compose([
    # ImageNet consists of 224x224 images; adjust input layer accordingly
    transforms.Resize(224),  
    # Data augmentation to simulate real-world conditions
    transforms.RandomHorizontalFlip(),  
    # Convert images to PyTorch tensors
    transforms.ToTensor(),
    # Normalize input channels based on ImageNet coefficients
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std =[0.229, 0.224, 0.225]),
])

# Pre-processing for the validation set (no augmentation)
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 training dataset
full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

# Split into Training and Validation (e.g., 45,000 training and 5,000 validation)
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
)

# Load CIFAR-10 test dataset (unchanged)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_val
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)
```
Dataloaders are a pytorch component that abstract a lot of low level code related to batch processing and paralell computing.

The following dataloader parameters control:
- **batch_size**: Number of samples that are evaluated before we update the model parameters. Larger batch sizes means faster training but require more memory and may lead to poor generalization.
- **shuffle**: Shuffles the data at each epoch which prevents the model from learning any specific patterns related to order and leads to better generalization.
- **num_workers**: Improves paralellization when working with a system with multiple CPU cores.
- **pin_memory**: The dataloader copies the tensor to the CUDA pineed memory improving data transfer to the GPU/TPU and reducing training time
batch_size


## Loss function, optimizer and learning rate scheduler
```python
# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer to include only trainable parameters
optimizer_finetune = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.01,  
    momentum=0.9,
    weight_decay=5e-4
)

# Define a learning rate scheduler for fine-tuning
scheduler_finetune = optim.lr_scheduler.StepLR(optimizer_finetune, step_size=5, gamma=0.1)
```

### Cross-Entropy Loss
It measures the difference between the predicted probabilities of each class and the true class labels:
- For each input, the model outputs a set of scores (logits) for each class.
- These logits are passed through a softmax function to convert them into probabilities.
- The cross-entropy loss calculates how far these probabilities are from the true class (encoded as a one-hot vector).
- The goal is to minimize this loss so that the predicted probabilities align closely with the true labels.
- For example, if the true class is "cat" (encoded as [1, 0, 0]) and the model predicts probabilities [0.7, 0.2, 0.1], the cross-entropy loss will measure how well the predicted distribution matches the true class.

### Optimizer (Stochastic Gradient Descent - SGD)
The optimizer updates the model's parameters during training using gradient descent. The key components here are:
- filter(lambda p: p.requires_grad, model.parameters()): We only update the parameters of the layers that are unfrozen.Helps to avoid overfitting.
- Learning Rate (lr=0.01): Determines the step size for each parameter update. A small learning rate makes convergence stable but slow, while a large learning rate can lead to overshooting or instability.
- Momentum (momentum=0.9): Adds a fraction of the previous update's direction to the current one, helping to overcome local minima and speeding up convergence.
- Weight Decay (weight_decay=5e-4): This adds L2 regularization, which discourages overly large weights and helps prevent overfitting.
- SGD computes gradients using backpropagation and updates the weights to minimize the loss.

### Learning Rate Scheduler
A learning rate scheduler adjusts the learning rate during training. This is important because:
- A high learning rate can make the model converge faster at the start, but it might not find the optimal solution.
- A low learning rate later in training helps fine-tune the model for better accuracy.
- The StepLR scheduler decreases the learning rate by a factor of gamma every step_size epochs:
step_size=5: Reduces the learning rate every 5 epochs.
gamma=0.1: Multiplies the current learning rate by 0.1. 
This means at epoch 6 during fine tuning our learning rate will be 0.01 if our initial learning rate is 0.1
