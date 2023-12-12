import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import L1_MAD_weighted, Euclidean_dist, DenseModel, adv_loss, CustomDataset


if __name__ == '__main__':

    
    data = pd.read_csv('data_6.csv',index_col=0)
    scaler = MinMaxScaler()

    X = data.iloc[:,:-1].to_numpy()
    Y = data.loc[:,'zfygpa'].to_numpy()
    
    test_size = 0.2
    (X_train, X_test, y_train, y_test) = train_test_split(X,
                                                          Y,
                                                          test_size = test_size,
                                                          random_state = 22,
                                                          shuffle = True
                                                          )

    X_train = torch.tensor(X_train).to(device = 'cuda:0',dtype = torch.float32)
    X_test = torch.tensor(X_test).to(device = 'cuda:0',dtype = torch.float32)
    y_train = torch.tensor(y_train).to(device = 'cuda:0',dtype = torch.float32)
    y_test = torch.tensor(y_test).to(device = 'cuda:0',dtype = torch.float32)

    training_data = torch.cat((X_train, torch.reshape(y_train,shape=(-1,1))), dim = 1)
    training_data = CustomDataset(training_data)
    data_loader = DataLoader(training_data, batch_size=32, shuffle=True)

    train = True
    loss_fn = torch.nn.MSELoss()
    
    if train:

        num_epochs = 100
        training_bar = tqdm(range(num_epochs))

        model = DenseModel(X_train.shape[1],1).to(device='cuda:0')
        model.train()
        optimizer = torch.optim.Adam(params=model.parameters(),lr = 1e-2,weight_decay=1e-5)

        for epoch in training_bar:

            for batch in data_loader:
                
                x,y = batch
                y = torch.reshape(y,shape=(-1,1))
                model.zero_grad()
                y_pred = model(x)
                
                train_loss = loss_fn(y_pred,y)
                optimizer.zero_grad()
                
                train_loss.backward()
                optimizer.step()

                training_bar.set_postfix(mse=float(train_loss))
        
        if train_loss.item() < 1e-5:
            torch.save(model,'trained_model.pt')
     
        model.eval()

        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = loss_fn(y_test_pred,torch.reshape(y_test,(-1,1)))
            print(f'The test loss is {test_loss.item()}')

    X_pert = torch.rand_like(X_train,requires_grad=True,device = 'cuda:0')    
    y_target = torch.zeros_like(y_train).to('cuda:0')

    model.eval()
    adv_optimizer = torch.optim.Adam([X_pert],lr = 1e-2)
    lamb = 1e-3
    num_iters = 3000
    pert_bar = tqdm(range(num_iters))

    for param in model.parameters():
        param.requires_grad = False

    for i in pert_bar:

        if i % 1000 == 0:
            lamb *= 1.2

        adv_logits = model(X_pert)
        loss = adv_loss(lamb,
                        adv_logits,
                        y_target,
                        X_train,
                        X_pert,
                        'L1_MAD',
                        False
                        )
        
        adv_optimizer.zero_grad()

        loss.backward()
        adv_optimizer.step()
        pert_bar.set_postfix(loss = float(loss)) 
    
    X_pert = X_pert.detach()
    
    with torch.no_grad():
        X_pert_pred = model(X_pert)
        X_train_pred = model(X_train)

        X_pert_pred = (X_pert_pred - X_pert_pred.mean()) / (X_pert_pred.max() - X_pert_pred.min())
        X_train_pred = (X_train_pred - X_train_pred.mean()) / (X_train_pred.max() - X_train_pred.min())
    
    print(X_train)
    print(X_pert)
    
    
    