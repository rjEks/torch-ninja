from torch import nn
from torch import optim


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network.
    
    A simple feedforward neural network with two hidden layers
    and ReLU activation functions.
    """
    
    def __init__(self, input_size, hidden_size, out_size): 
        """
        Initialize the MLP network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            out_size: Number of output features
        """
        super(MLP, self).__init__()
        
        # Feature extraction layers with ReLU activation
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, out_size),
            nn.ReLU(),
        )
        
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input tensor
            
        Returns:
            Output tensor after passing through the network
        """
        # Pass through feature extraction layers
        hidden = self.features(X)
        
        # Pass through output layer
        output = self.classifier(hidden)
        
        return output