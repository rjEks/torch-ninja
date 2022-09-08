import torch
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils

def prepare_split(data):

    data_bike_sharing = pd.read_csv("data/bike_sharing.csv", index_col=0)
    data_bike_sharing = pd.get_dummies(data_bike_sharing, columns= ["season"])

    
    columns = ['registered', 'holiday', 'weekday', 
           'weathersit', 'temp', 'atemp',
           'season_fall', 'season_spring', 
           'season_summer', 'season_winter']
    
    features = data_bike_sharing[columns]
    target = data_bike_sharing[['cnt']]
    X_train, x_test, Y_train, y_test = train_test_split(features, target, test_size=0.2)

    return X_train, x_test, Y_train, y_test, features, target

def convert_pytorch_tensors(x_train,x_test,y_train,y_test):

    X_train_tensor = torch.tensor(x_train.values, dtype = torch.float)
    x_test_tensor = torch.tensor(x_test.values, dtype = torch.float)

    Y_train_tensor = torch.tensor(y_train.values, dtype = torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype = torch.float)

    return X_train_tensor, x_test_tensor, Y_train_tensor, y_test_tensor

def create_dataset_dataloader(X_train_tensor, Y_train_tensor):

    train_data = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = data_utils.DataLoader(train_data, batch_size=100, shuffle=True)

    return train_data, train_loader

def settings(input_value):
    input_value =  input_value
    output = 1
    hidden = 10
    loss_fn = torch.nn.MSELoss()

    return input_value, output, hidden, loss_fn

def build_training_model(input,hidden, output, train_loader, features, target, loss_fn):
    
    #Constroi model com camadas, inp, out e hid
    model = torch.nn.Sequential(torch.nn.Linear(input,hidden),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden, output))
    #Define otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    total_step = len(train_loader)
    num_epochs = 10000

    for epoch in range(num_epochs +1):
        for i, (features, target) in enumerate(train_loader):
            output = model(features)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
        
            loss.backward()
        
            optimizer.step()

            if epoch % 2000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    return model
