from Bycicle import Bycicle
from mlp_impl import MLP
import settings
import pandas as pd
import train
import validate
import torch

#Running cycles and epochs

def running():
    
    df = pd.read_csv("../data/hour.csv")
    print(len(df))
    
    df_train, df_test = settings.trainTestSplit(df)
    
    print("Train size: " + str(len(df_train)))
    print("Test Size: " + str(len(df_test)))
    
    print("Convert dataset")
    train_set = Bycicle("../data/train.csv")
    test_set = Bycicle("../data/test.csv")    
    
    print("Get Args")
    args = settings.setArgs()    
    
    print("Create Dataloader")
    train_loader, test_loader = settings.createDataLoader(train_set,test_set,args["batch_size"],args["num_workers"])
    
    print("Check Dimensions")
    for batch in test_loader:
        sample, label = batch
        print(sample.size(), label.size())
        break
    
    print("Set Variables")
    input_size = train_set[0][0].size(0)
    hidden_size = 128
    out_size = 1    
    
    network = MLP(input_size,hidden_size,out_size).to(settings.returnIsCuda())
    print(network)
    
    print("set criterion and loss")
    criterion, optimizer = settings.setCrtiterionAndLoss(network,args["lr"],args["weight_decay"])    
    
    train_losses, test_losses = [], []
    
    for epoch in range(args['epoch_num']):
        
        #Trainning
        train_losses.append(train.train(train_loader, network,epoch,settings.returnIsCuda(),criterion,optimizer))
        
        #Evaluate
        train_losses.append(validate.validate(train_loader, network,epoch,settings.returnIsCuda(),criterion))
    
    
    Xtest = torch.stack([tup[0] for tup in test_set])
    Xtest = Xtest.to(settings.returnIsCuda())

    ytest = torch.stack([tup[1] for tup in test_set])
    ypred = network(Xtest).cpu().data

    data = torch.cat((ytest, ypred), axis=1)

    df_results = pd.DataFrame(data, columns=['ypred', 'ytest'])
    df_results.head(20)
    
if __name__=="__main__":
    running()
    