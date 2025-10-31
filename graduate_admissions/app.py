import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sns


class Net(nn.Module):
    """
    Neural network for graduate admissions prediction.
    
    A fully connected network with configurable activation functions
    and optional dropout regularization.
    """
    
    def __init__(self, hidden_size, activation_fn='relu', apply_dropout=False, input_size=None, output_size=None):
        """
        Initialize the neural network.
        
        Args:
            hidden_size: Number of neurons in hidden layers
            activation_fn: Activation function ('relu', 'sigmoid', or 'tanh')
            apply_dropout: Whether to apply dropout regularization
            input_size: Number of input features (if None, must be set as global variable)
            output_size: Number of output classes (if None, must be set as global variable)
        """
        super(Net, self).__init__()
        
        # Use provided values or fall back to global variables for backward compatibility
        _input_size = input_size if input_size is not None else globals().get('input_size', 7)
        _output_size = output_size if output_size is not None else globals().get('output_size', 1)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(_input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, _output_size)
        
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        # Optional dropout layer for regularization
        self.dropout = None
        if apply_dropout:
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Log-softmax output for classification
        """
        # Select activation function based on configuration
        activation_fn = None
        if self.activation_fn == 'sigmoid':
            activation_fn = torch.sigmoid
        elif self.activation_fn == 'tanh':
            activation_fn = torch.tanh
        elif self.activation_fn == 'relu':
            activation_fn = F.relu
        
        # First hidden layer with activation
        x = activation_fn(self.fc1(x))
        
        # Second hidden layer with activation
        x = activation_fn(self.fc2(x))
        
        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        # Return log-softmax for classification
        return F.log_softmax(x, dim=-1)

    @staticmethod
    def train_and_evaluate_model(model, Xtrain, Ytrain, Xtest, Ytest, y_test, learn_rate=0.001):
        """
        Train and evaluate the neural network model.
        
        Args:
            model: Neural network model to train
            Xtrain: Training features
            Ytrain: Training labels
            Xtest: Test features
            Ytest: Test labels
            y_test: Test labels in original format (for accuracy calculation)
            learn_rate: Learning rate for optimization
            
        Returns:
            Dictionary containing model, training history, and evaluation metrics
        """
        epoch_data = []
        epochs = 1001
        
        # Initialize Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        
        # Use Negative Log-Likelihood Loss for classification
        loss_fn = nn.NLLLoss()
        
        test_accuracy = 0.0
        
        # Training loop
        for epoch in range(1, epochs):
            # Zero gradients before backward pass
            optimizer.zero_grad()
            
            # Forward pass on training data
            Ypred = model(Xtrain)
            
            # Calculate training loss
            loss = loss_fn(Ypred, Ytrain)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Evaluate on test data
            Ypred_test = model(Xtest)
            loss_test = loss_fn(Ypred_test, Ytest)
            
            # Get predictions by taking argmax
            _, pred = Ypred_test.data.max(1)
            
            # Calculate test accuracy
            test_accuracy = pred.eq(Ytest.data).sum().item() / y_test.values.size
            
            # Store epoch statistics
            epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), test_accuracy])
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print('Epoch - %d (%d%%) Train Loss - %.2f Test Loss - %.2f Test Accuracy - %.4f' %
                      (epoch, epoch / 150 * 10, loss.data.item(), loss_test.data.item(), test_accuracy))
        
        # Return training results
        return {
            'model': model,
            'epoch_data': epoch_data, 
            'num_epochs': epochs, 
            'optimizer': optimizer, 
            'loss_fn': loss_fn,
            'test_accuracy': test_accuracy,
            'predictions': Ypred_test.data.max(1),
            'actual_test_label': Ytest,
        }