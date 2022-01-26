import torch

args ={
    'epoch_num': 300,     # Número de épocas.
    'lr': 5e-5,           # Taxa de aprendizado.
    'weight_decay': 5e-4, #  L2 (Regularização).
    'num_workers': 3,     # threads do dataloader.
    'batch_size': 20,     # Tamanho do batch.
}
def returnIsCuda() -> bool:        
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')
    
    return args['device']