import torch
from torch import nn
from torch import optim
import time
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def train(train_loader, network, epoch, device, criterion, optimizer):
    
    #modo de treinamento
    network.train()
    start = time.time()
    
    epoch_loss = []
    
    for batch in train_loader:
        
        dado, rotulo = batch
        
        #Cast
        dado = dado.to(device)
        rotulo = rotulo.to(device)
        
        #Forward
        ypred = network(dado)
        loss = criterion(ypred, rotulo)
        epoch_loss.append(loss.cpu().data)
        
        #Backpropagation
        loss.backward()
        optimizer.step()
        
    epoch_loss = np.asarray(epoch_loss)
    end = time.time()
    print('#################### Train ####################')
    print('Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), end-start))
    
    return epoch_loss.mean()
        
    
    
