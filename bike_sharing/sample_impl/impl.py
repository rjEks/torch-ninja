import torch
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils


def prepare_split(data):
    """
    Load and prepare the bike sharing dataset.
    
    Args:
        data: Path to the data file (not currently used - path is hardcoded)
        
    Returns:
        Tuple of (X_train, x_test, Y_train, y_test, features, target)
    """
    # Load bike sharing dataset
    data_bike_sharing = pd.read_csv("data/bike_sharing.csv", index_col=0)
    
    # One-hot encode the 'season' column
    data_bike_sharing = pd.get_dummies(data_bike_sharing, columns=["season"])
    
    # Select feature columns
    columns = ['registered', 'holiday', 'weekday', 
               'weathersit', 'temp', 'atemp',
               'season_fall', 'season_spring', 
               'season_summer', 'season_winter']
    
    features = data_bike_sharing[columns]
    target = data_bike_sharing[['cnt']]
    
    # Split into training and test sets (80/20 split)
    X_train, x_test, Y_train, y_test = train_test_split(features, target, test_size=0.2)
    
    return X_train, x_test, Y_train, y_test, features, target


def convert_pytorch_tensors(x_train, x_test, y_train, y_test):
    """
    Convert numpy arrays to PyTorch tensors.
    
    Args:
        x_train: Training features (numpy/pandas)
        x_test: Test features (numpy/pandas)
        y_train: Training labels (numpy/pandas)
        y_test: Test labels (numpy/pandas)
        
    Returns:
        Tuple of PyTorch tensors (X_train_tensor, x_test_tensor, Y_train_tensor, y_test_tensor)
    """
    X_train_tensor = torch.tensor(x_train.values, dtype=torch.float)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)
    
    Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)
    
    return X_train_tensor, x_test_tensor, Y_train_tensor, y_test_tensor


def create_dataset_dataloader(X_train_tensor, Y_train_tensor):
    """
    Create PyTorch Dataset and DataLoader from tensors.
    
    Args:
        X_train_tensor: Training features tensor
        Y_train_tensor: Training labels tensor
        
    Returns:
        Tuple of (train_data, train_loader)
    """
    train_data = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = data_utils.DataLoader(train_data, batch_size=100, shuffle=True)
    
    return train_data, train_loader


def settings(input_value):
    """
    Configure network architecture and loss function.
    
    Args:
        input_value: Number of input features
        
    Returns:
        Tuple of (input_value, output, hidden, loss_fn)
    """
    input_value = input_value
    output = 1  # Single output for regression
    hidden = 10  # Number of hidden neurons
    loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss for regression
    
    return input_value, output, hidden, loss_fn


def build_training_model(input, hidden, output, train_loader, features, target, loss_fn):
    """
    Build and train a neural network model.
    
    Args:
        input: Number of input features
        hidden: Number of hidden neurons
        output: Number of output features
        train_loader: DataLoader for training data
        features: Feature data (not used in current implementation)
        target: Target data (not used in current implementation)
        loss_fn: Loss function
        
    Returns:
        Trained PyTorch model
    """
    # Build model with input, hidden, and output layers
    model = torch.nn.Sequential(
        torch.nn.Linear(input, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, output)
    )
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    total_step = len(train_loader)
    num_epochs = 10000
    
    # Training loop
    for epoch in range(num_epochs + 1):
        for i, (features, target) in enumerate(train_loader):
            # Forward pass
            output = model(features)
            loss = loss_fn(output, target)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Print progress every 2000 epochs
            if epoch % 2000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    
    return model