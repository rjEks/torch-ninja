from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class Bicycle(Dataset):
    """
    Custom PyTorch Dataset for bicycle sharing data.
    
    This dataset loads bicycle sharing data from a CSV file and
    provides samples with features and labels.
    """
    
    def __init__(self, csv_path, scaler_feat=None, scaler_label=None):
        """
        Initialize the Bicycle dataset.
        
        Args:
            csv_path: Path to the CSV file containing the data
            scaler_feat: Optional feature scaler (not currently used)
            scaler_label: Optional label scaler (not currently used)
        """
        self.data = pd.read_csv(csv_path).to_numpy()
        
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (sample, label) where:
                - sample: Feature vector (columns 2-13)
                - label: Target value (last column)
        """
        # Extract features (columns 2 to 14, excluding 14)
        sample = self.data[idx][2:14]
        
        # Extract label (last column)
        label = self.data[idx][-1:]
        
        # Convert to PyTorch tensors
        sample = torch.from_numpy(sample.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        
        return sample, label
     
    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.data)