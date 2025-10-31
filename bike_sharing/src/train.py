import torch
from torch import nn
from torch import optim
import time
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def train(train_loader, network, epoch, device, criterion, optimizer):
    """
    Train the neural network for one epoch.
    
    Args:
        train_loader: DataLoader containing training data
        network: Neural network model to train
        epoch: Current epoch number
        device: Device to run training on (CPU or CUDA)
        criterion: Loss function
        optimizer: Optimization algorithm
        
    Returns:
        Mean loss for the epoch
    """
    # Set network to training mode
    network.train()
    start = time.time()
    
    epoch_loss = []
    
    for batch in train_loader:
        # Extract data and labels from batch
        data, labels = batch
        
        # Move data to the specified device (CPU or GPU)
        data = data.to(device)
        labels = labels.to(device)
        
        # Zero the gradients before backward pass
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        predictions = network(data)
        loss = criterion(predictions, labels)
        epoch_loss.append(loss.cpu().data)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
    epoch_loss = np.asarray(epoch_loss)
    end = time.time()
    
    # Print training statistics
    print('#################### Train ####################')
    print('Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), end-start))
    
    return epoch_loss.mean()

