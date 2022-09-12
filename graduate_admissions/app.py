import torch.optim as optim
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self,hidden_size, activation_fn = 'relu', apply_dropout=False): 
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.dropout = None
        if apply_dropout:
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        
        activation_fn = None
        if  self.activation_fn == 'sigmoid':
                activation_fn = F.torch.sigmoid

        elif self.activation_fn == 'tanh':
                activation_fn = F.torch.tanh

        elif self.activation_fn == 'relu':
                 activation_fn = F.relu

        x = activation_fn(self.fc1(x))
        x = activation_fn(self.fc2(x))

        if self.dropout != None:
            x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim = -1)

    def train_and_evaluate_model(model, learn_rate=0.001):
        
        epoch_data = []
        epochs = 1001

        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        loss_fn = nn.NLLLoss()

        test_accuracy = 0.0
        for epoch in range(1, epochs):

            optimizer.zero_grad()

            Ypred = model(Xtrain)

            loss = loss_fn(Ypred , Ytrain)
            loss.backward()

            optimizer.step()

            Ypred_test = model(Xtest)
            loss_test = loss_fn(Ypred_test, Ytest)

            _, pred = Ypred_test.data.max(1)

            test_accuracy = pred.eq(Ytest.data).sum().item() / y_test.values.size

            epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), test_accuracy])

            if epoch % 100 == 0:
                print ('epoch - %d (%d%%) train loss - %.2f test loss - %.2f Test accuracy - %.4f'\
                       % (epoch, epoch/150 * 10 , loss.data.item(), loss_test.data.item(), test_accuracy))


        return {'model' : model,
                'epoch_data' : epoch_data, 
                'num_epochs' : epochs, 
                'optimizer' : optimizer, 
                'loss_fn' : loss_fn,
                'test_accuracy' : test_accuracy,
                '_, pred' : Ypred_test.data.max(1),
                'actual_test_label' : Ytest,
                }