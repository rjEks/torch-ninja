from lightgbm import Dataset


from torch.utils.data import Dataset


class Bycicle(Dataset):
    
    def __init__(self, csv_path, scaler_feat=None, scaler_label=None) -> None:
        super().__init__()