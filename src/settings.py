import torch

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
    
    df_train.to_csv('train.csv',index=False)
    df_test.to_csv('test.csv',index=False)    


def returnIsCuda() -> bool:        
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')
    
    return args['device']


def setArgs(args) -> dict:
    return args

