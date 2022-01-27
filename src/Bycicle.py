from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class Bycicle(Dataset):
    
    def __init__(self, csv_path, scaler_feat=None, scaler_label=None):
        self.dados = pd.read_csv(csv_path).to_numpy()
        
    def __getitem__(self, idx):
                
         sample = self.dados[idx][2:14]
         label  = self.dados[idx][-1:]
        
         sample = torch.from_numpy(sample.astype(np.float32))
         label = torch.from_numpy(label.astype(np.float32))
         
         return sample, label
     
    def __len__(self):
         return len(self.dados)