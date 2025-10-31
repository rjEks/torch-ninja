"""
MNIST Classifier for AWS SageMaker

This module implements a Convolutional Neural Network (CNN) for MNIST digit classification
with support for distributed training on both CPU and GPU environments.
"""

import argparse
import json
import logging
import os
import sagemaker_containers
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Net(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - Conv layer 1: 1 -> 10 channels, 5x5 kernel
    - Conv layer 2: 10 -> 20 channels, 5x5 kernel (with dropout)
    - Fully connected layer 1: 320 -> 50 neurons
    - Fully connected layer 2: 50 -> 10 neurons (output)
    """
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - Conv layer 1: 1 -> 10 channels, 5x5 kernel
    - Conv layer 2: 10 -> 20 channels, 5x5 kernel (with dropout)
    - Fully connected layer 1: 320 -> 50 neurons
    - Fully connected layer 2: 50 -> 10 neurons (output)
    """
    
    def __init__(self):
        """Initialize the CNN architecture."""
        super(Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # Dropout for regularization
        
        # Fully connected layers
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Log-softmax output of shape (batch_size, 10)
        """
        # First convolutional block: conv -> relu -> maxpool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Second convolutional block: conv -> dropout -> relu -> maxpool
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Flatten for fully connected layers
        x = x.view(-1, 320)
        
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        
        # Dropout for regularization
        x = F.dropout(x, training=self.training)
        
        # Output layer
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    """
    Create a DataLoader for training data.
    
    Args:
        batch_size: Number of samples per batch
        training_dir: Directory containing MNIST training data
        is_distributed: Whether to use distributed training
        **kwargs: Additional arguments for DataLoader (e.g., num_workers, pin_memory)
        
    Returns:
        DataLoader for training data
    """
    logger.info("Get train data loader")
    
    # Load MNIST training dataset with normalization
    dataset = datasets.MNIST(training_dir, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ]))
    
    # Use distributed sampler if training across multiple nodes
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train_sampler is None,  # Shuffle only if not using distributed sampler
        sampler=train_sampler, 
        **kwargs
    )


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    """
    Create a DataLoader for test data.
    
    Args:
        test_batch_size: Number of samples per batch for testing
        training_dir: Directory containing MNIST test data
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader for test data
    """
    logger.info("Get test data loader")
    
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])),
        batch_size=test_batch_size, 
        shuffle=True, 
        **kwargs
    )


def _average_gradients(model):
    """
    Average gradients across all processes in distributed training.
    
    This is used for gradient synchronization in distributed CPU training.
    
    Args:
        model: Neural network model
    """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args):
    """
    Main training function for MNIST classifier.
    
    Supports both single-machine and distributed training on CPU/GPU.
    
    Args:
        args: Namespace containing training configuration:
            - hosts: List of host machines for distributed training
            - backend: Communication backend for distributed training
            - num_gpus: Number of GPUs available
            - batch_size: Training batch size
            - test_batch_size: Test batch size
            - data_dir: Directory containing MNIST data
            - seed: Random seed for reproducibility
            - epochs: Number of training epochs
            - lr: Learning rate
            - momentum: SGD momentum
            - log_interval: Logging frequency
            - model_dir: Directory to save trained model
            - current_host: Current host machine name
    """
    # Check if distributed training is enabled
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    
    # Determine if CUDA is available
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    
    # Set up DataLoader kwargs for optimal performance
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment for multi-node training
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Create data loaders
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    # Create model and move to appropriate device
    model = Net().to(device)
    
    # Set up distributed or data parallel training
    if is_distributed and use_cuda:
        # Multi-machine multi-GPU case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # Single-machine multi-GPU or CPU case
        model = torch.nn.DataParallel(model)

    # Initialize SGD optimizer with momentum
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Average gradients for distributed CPU training
            if is_distributed and not use_cuda:
                _average_gradients(model)
            
            # Update weights
            optimizer.step()
            
            # Log training progress
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        
        # Test the model after each epoch
        test(model, test_loader, device)
    
    # Save the trained model
    save_model(model, args.model_dir)


def test(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on (CPU or CUDA)
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            
            # Get the index of the max log-probability (predicted class)
            pred = output.max(1, keepdim=True)[1]
            
            # Count correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def model_fn(model_dir):
    """
    Load the trained model for inference.
    
    This function is used by SageMaker for model serving.
    
    Args:
        model_dir: Directory containing the saved model
        
    Returns:
        Loaded model on appropriate device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    
    # Load model weights
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)


def save_model(model, model_dir):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained neural network model
        model_dir: Directory to save the model
    """
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    
    # Save model state dict (recommended approach)
    # Reference: http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    """
    Main entry point for the MNIST training script.
    
    Parses command-line arguments and initiates training.
    Designed to work with AWS SageMaker training infrastructure.
    """
    parser = argparse.ArgumentParser()

    # Data and model checkpoint directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment variables (provided by SageMaker)
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # Start training
    train(parser.parse_args())