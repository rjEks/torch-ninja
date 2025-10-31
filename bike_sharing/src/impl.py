from Bycicle import Bicycle
from mlp_impl import MLP
import settings
import pandas as pd
import train
import validate
import torch


def running():
    """
    Main function to run the bicycle sharing prediction model.
    
    This function performs the following steps:
    1. Loads and splits the dataset
    2. Creates data loaders
    3. Initializes the neural network
    4. Trains the model
    5. Evaluates the model on test data
    """
    # Load the dataset
    df = pd.read_csv("../data/hour.csv")
    print("Total samples:", len(df))
    
    # Split into training and test sets
    df_train, df_test = settings.train_test_split(df)
    
    print("Train size:", len(df_train))
    print("Test size:", len(df_test))
    
    # Convert DataFrames to PyTorch datasets
    print("Converting dataset...")
    train_set = Bicycle("../data/train.csv")
    test_set = Bicycle("../data/test.csv")
    
    # Get training configuration
    print("Getting configuration arguments...")
    args = settings.get_args()
    
    # Create data loaders for batch processing
    print("Creating DataLoaders...")
    train_loader, test_loader = settings.create_data_loader(
        train_set, test_set, args["batch_size"], args["num_workers"]
    )
    
    # Check data dimensions
    print("Checking dimensions...")
    for batch in test_loader:
        sample, label = batch
        print("Sample size:", sample.size(), "Label size:", label.size())
        break
    
    # Set network architecture parameters
    print("Setting network variables...")
    input_size = train_set[0][0].size(0)
    hidden_size = 128
    out_size = 1
    
    # Initialize the neural network and move to appropriate device
    network = MLP(input_size, hidden_size, out_size).to(settings.get_device())
    print("Network architecture:")
    print(network)
    
    # Initialize loss function and optimizer
    print("Setting criterion and optimizer...")
    criterion, optimizer = settings.set_criterion_and_optimizer(
        network, args["lr"], args["weight_decay"]
    )
    
    # Lists to store loss values for monitoring
    train_losses, test_losses = [], []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args['epoch_num']):
        # Train for one epoch
        train_loss = train.train(
            train_loader, network, epoch, settings.get_device(), criterion, optimizer
        )
        train_losses.append(train_loss)
        
        # Validate on test set
        test_loss = validate.validate(
            test_loader, network, epoch, settings.get_device(), criterion
        )
        test_losses.append(test_loss)
    
    # Final evaluation on test set
    print("\nPerforming final evaluation...")
    Xtest = torch.stack([tup[0] for tup in test_set])
    Xtest = Xtest.to(settings.get_device())
    
    ytest = torch.stack([tup[1] for tup in test_set])
    ypred = network(Xtest).cpu().data
    
    # Combine predictions and actual values
    data = torch.cat((ytest, ypred), axis=1)
    
    # Create DataFrame with results (first column: actual, second column: predicted)
    df_results = pd.DataFrame(data.numpy(), columns=['ytest', 'ypred'])
    print("\nFirst 20 predictions:")
    print(df_results.head(20))


if __name__ == "__main__":
    running()