from impl import *


def main():
    print('oi')

if __name__ == '__main__':
   
   #Prepare e split
   X_train, x_test, Y_train, y_test, features, target = prepare_split("data/bike_sharing.csv")

   #Convertendo de numpy para os tensores
   X_train_tensor, x_test_tensor, Y_train_tensor, y_test_tensor = convert_pytorch_tensors( X_train, x_test, Y_train, y_test)

   #Dataset e dataloader 
   train_data, train_loader = create_dataset_dataloader(X_train_tensor, Y_train_tensor)

   #setando configs iniciais
   input_value, output, hidden, loss_fn = settings(X_train_tensor.shape[1])

   #Build e treino do modelo
   model = build_training_model(input_value,hidden,output,train_loader,features,target,loss_fn)

   model.eval()

   with torch.no_grad():
    y_pred = model(x_test_tensor)
    sample = x_test.iloc[45]
    print(sample)

    sample_tensor = torch.tensor(sample.values, 
                             dtype = torch.float)
    print(sample_tensor)

    with torch.no_grad():
        y_pred = model(sample_tensor)

    print("Valor Predito : ", (y_pred.item()))
    print("Valor Real : ", (y_test.iloc[45]))

   