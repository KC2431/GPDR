import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import ConfusionMatrix
from tqdm import tqdm
from utils import L1_MAD_weighted, Euclidean_dist, DenseModel, adv_loss, CustomDataset


if __name__ == '__main__':

    torch.manual_seed(20)

    data = pd.read_csv('diabetes.csv')
    scaler = MinMaxScaler()

    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                          Y,
                                                          test_size = test_size,
                                                          shuffle = False
                                                          )

    X_train = torch.tensor(scaler.fit_transform(X_train)).to(device = 'cuda:0',dtype = torch.float32)
    X_test = torch.tensor(scaler.fit_transform(X_test)).to(device = 'cuda:0',dtype = torch.float32)
    y_train = torch.tensor(y_train).to(device = 'cuda:0',dtype = torch.float32)
    y_test = torch.tensor(y_test).to(device = 'cuda:0',dtype = torch.float32)

    #training_data = torch.cat((X_train, torch.reshape(y_train,shape=(-1,1))), dim = 1)
    training_data = CustomDataset(X_train,y_train)
    data_loader = DataLoader(training_data, batch_size=10, shuffle=False)

    train = True
    loss_fn = torch.nn.BCELoss()
    
    if train:

        num_epochs = 200

        print('Training the model.')

        training_bar = tqdm(range(num_epochs))

        model = DenseModel(X_train.shape[1],1).to(device='cuda:0')
        optimizer = torch.optim.Adam(params=model.parameters(),lr = 1e-3,weight_decay=1e-2)
         
        for epoch in training_bar:
            for batch in data_loader:
            
                x,y = batch
                y_pred = model(x)
                    
                train_loss = loss_fn(y_pred,y.reshape(-1,1))
                optimizer.zero_grad()
                
                train_loss.backward()
                optimizer.step()

                training_bar.set_postfix(BCE_loss=float(train_loss))

        model.eval()

        print('Testing the model now.')

        with torch.no_grad():
            y_test_pred = model(X_test)
            y_test_pred = y_test_pred.round()

            cm = confusion_matrix(y_test.cpu().numpy(),y_test_pred.cpu().numpy())
            accu_score = accuracy_score(y_test.cpu().numpy(),y_test_pred.cpu().numpy())
            
            sns.heatmap(cm,annot=True)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            plt.title("Confusion matrix for PIMA dataset for test set")
            plt.show()

            print(f'The accuracy on test set is {accu_score*100:.2f}%')


    X_pert = torch.rand_like(X_train,requires_grad=True,device = 'cuda:0')    
    y_target = torch.zeros_like(y_train).to('cuda:0')
    y_target.add_(0.5)

    model.eval()
    adv_optimizer = torch.optim.Adam([X_pert],lr = 1e-3)
    lamb = 1e-3
    num_iters = 300
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
    
    print(scaler.inverse_transform(X_train.cpu().numpy()))
    print(scaler.inverse_transform(X_pert.cpu().numpy()))
    
    
    