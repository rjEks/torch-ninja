"""
Bike Sharing Prediction Application

This script demonstrates training and inference for bike sharing demand prediction
using a simple neural network.
"""

from impl import *


def main():
    """Main function (currently a placeholder)."""
    print('Starting bike sharing prediction application...')


if __name__ == '__main__':
    """
    Main entry point for bike sharing prediction.
    
    Steps:
    1. Load and prepare the dataset
    2. Convert data to PyTorch tensors
    3. Create DataLoader for batch processing
    4. Configure model architecture
    5. Train the model
    6. Perform inference on test samples
    """
    
    # Load and split the dataset
    print("Preparing and splitting data...")
    X_train, x_test, Y_train, y_test, features, target = prepare_split("data/bike_sharing.csv")
    
    # Convert numpy arrays to PyTorch tensors
    print("Converting to PyTorch tensors...")
    X_train_tensor, x_test_tensor, Y_train_tensor, y_test_tensor = convert_pytorch_tensors(
        X_train, x_test, Y_train, y_test
    )
    
    # Create Dataset and DataLoader
    print("Creating DataLoader...")
    train_data, train_loader = create_dataset_dataloader(X_train_tensor, Y_train_tensor)
    
    # Configure initial settings
    print("Setting up model configuration...")
    input_value, output, hidden, loss_fn = settings(X_train_tensor.shape[1])
    
    # Build and train the model
    print("Training model...")
    model = build_training_model(
        input_value, hidden, output, train_loader, features, target, loss_fn
    )
    
    # Switch to evaluation mode
    model.eval()
    
    # Perform inference on test data
    print("\nPerforming inference...")
    with torch.no_grad():
        # Get predictions for all test samples
        y_pred = model(x_test_tensor)
        
        # Get a specific sample for detailed prediction
        sample = x_test.iloc[45]
        print("\nSample features:")
        print(sample)
        
        # Convert sample to tensor
        sample_tensor = torch.tensor(sample.values, dtype=torch.float)
        print("\nSample tensor:")
        print(sample_tensor)
        
        # Make prediction for the sample
        with torch.no_grad():
            y_pred = model(sample_tensor)
        
        print("\nPredicted value:", y_pred.item())
        print("Actual value:", y_test.iloc[45].values[0])
