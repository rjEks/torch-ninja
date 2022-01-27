import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

args ={
    'epoch_num': 300,     # Número de épocas.
    'lr': 5e-5,           # Taxa de aprendizado.
    'weight_decay': 5e-4, #  L2 (Regularização).
    'num_workers': 3,     # threads do dataloader.
    'batch_size': 20,     # Tamanho do batch.
}

def trainTestSplit(df):
    
    torch.manual_seed(1)
    
    indexes = torch.randperm(len(df)).tolist()
    train_size = int(0.8*len(df))
    
    df_train = df.iloc[indexes[:train_size]]
    df_test = df.iloc[indexes[train_size:]]
    
    df_train.to_csv('../data/train.csv',index=False)
    df_test.to_csv('../data/test.csv',index=False)    
    
    return df_train, df_test


def returnIsCuda() -> str:        
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')
    
    return args['device']


def setArgs() -> dict:
    return args

def createDataLoader(train, test, batch_size, num_workers):
    
    train_loader = DataLoader(
        train,
        batch_size,
        num_workers=num_workers,
        shuffle=True
        )

    test_loader = DataLoader(
        test,
        batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    return train_loader, test_loader

def setCrtiterionAndLoss(network,lr,weigth_decay):
    criterion = nn.L1Loss().to(returnIsCuda())
    optimizer = optim.Adam(network.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    return criterion , optimizer