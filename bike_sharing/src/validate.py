import torch
from torch import nn
from torch import optim
import time
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def validate(test_loader, network, epoch, device, criterion):
    """
    Validate the neural network on test data.
    
    Args:
        test_loader: DataLoader containing validation/test data
        network: Neural network model to validate
        epoch: Current epoch number
        device: Device to run validation on (CPU or CUDA)
        criterion: Loss function
        
    Returns:
        Mean loss for the validation epoch
    """
    # Set network to evaluation mode
    network.eval()
    
    start = time.time()
    
    epoch_loss = []
    
    # For validation, we don't need to compute gradients
    with torch.no_grad():
        for batch in test_loader:
            # Extract data and labels from batch
            data, labels = batch
            
            # Move data to the specified device (CPU or GPU)
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass: compute predictions
            predictions = network(data)
            loss = criterion(predictions, labels)
            epoch_loss.append(loss.cpu().data)
    
    epoch_loss = np.asarray(epoch_loss)
    
    end = time.time()
    
    # Print validation statistics
    print('********** Validate **********')
    print('Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), end-start))
    
    return epoch_loss.mean()