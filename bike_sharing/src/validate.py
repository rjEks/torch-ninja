import torch
from torch import nn
from torch import optim
import time
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def validate(test_loader, network, epoch, device, criterion):
    
  # Evaluation mode
  network.eval()
  
  start = time.time()
  
  epoch_loss  = []
  
  #Para validação nao precisamos calcular o gradiente
  with torch.no_grad(): 
    for batch in test_loader:

      dado, rotulo = batch

      # Cast do dado na GPU
      dado = dado.to(device)
      rotulo = rotulo.to(device)

      # Forward
      ypred = network(dado)
      loss = criterion(ypred, rotulo)
      epoch_loss.append(loss.cpu().data)

  epoch_loss = np.asarray(epoch_loss)
  
  end = time.time()
  print('********** Validate **********')
  print('Epoch %d, Loss: %.4f +/- %.4f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), end-start))
  
  return epoch_loss.mean()