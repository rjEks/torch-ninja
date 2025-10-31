import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Training configuration parameters
args = {
    'epoch_num': 300,     # Number of training epochs
    'lr': 5e-5,           # Learning rate
    'weight_decay': 5e-4, # L2 regularization coefficient
    'num_workers': 3,     # Number of DataLoader worker threads
    'batch_size': 20,     # Batch size for training
}


def train_test_split(df):
    """
    Split the dataset into training and test sets.
    
    Args:
        df: pandas DataFrame containing the dataset
        
    Returns:
        Tuple of (df_train, df_test) - training and test DataFrames
    """
    # Set random seed for reproducibility
    torch.manual_seed(1)
    
    # Generate random permutation of indices
    indexes = torch.randperm(len(df)).tolist()
    train_size = int(0.8 * len(df))
    
    # Split data into train and test sets
    df_train = df.iloc[indexes[:train_size]]
    df_test = df.iloc[indexes[train_size:]]
    
    # Save splits to CSV files
    df_train.to_csv('../data/train.csv', index=False)
    df_test.to_csv('../data/test.csv', index=False)
    
    return df_train, df_test


def get_device() -> str:
    """
    Determine and return the appropriate device (CUDA or CPU).
    
    Returns:
        torch.device object (either 'cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')
    
    return args['device']


def get_args() -> dict:
    """
    Get the training configuration arguments.
    
    Returns:
        Dictionary containing training configuration
    """
    return args


def create_data_loader(train, test, batch_size, num_workers):
    """
    Create DataLoader objects for training and testing.
    
    Args:
        train: Training dataset
        test: Test dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker threads for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train,
        batch_size,
        num_workers=num_workers,
        shuffle=True  # Shuffle training data for better generalization
    )
    
    test_loader = DataLoader(
        test,
        batch_size,
        num_workers=num_workers,
        shuffle=False  # Don't shuffle test data
    )
    
    return train_loader, test_loader


def set_criterion_and_optimizer(network, lr, weight_decay):
    """
    Initialize loss function and optimizer.
    
    Args:
        network: Neural network model
        lr: Learning rate
        weight_decay: L2 regularization coefficient
        
    Returns:
        Tuple of (criterion, optimizer)
    """
    # Use L1 loss (Mean Absolute Error) for regression
    criterion = nn.L1Loss().to(get_device())
    
    # Use Adam optimizer with specified learning rate and weight decay
    optimizer = optim.Adam(network.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    return criterion, optimizer