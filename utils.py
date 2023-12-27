import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import median_abs_deviation
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.manual_seed(21)

# loss functions 
# ==========================================================================================================================================================
def L1_MAD_weighted(x_pert,x_orig):
    
    MAD = median_abs_deviation(x_orig.cpu().numpy(),axis = 0)
    diff = torch.abs(x_orig - x_pert)
    print(torch.tensor(MAD).shape)
    return (diff / torch.tensor(MAD).cuda()).sum(dim = 1)


def Euclidean_dist(X_pert,X_orig,weighted = False):

    loss = (X_orig - X_pert) ** 2
     
    if not weighted:
        loss = loss.sum(dim = 1)
    
    else:
        loss = torch.div(loss, X_orig.std(dim = 0))
        loss = loss.sum(dim = 1)

    return loss

def adv_loss(lamb,
            adv_logits,
            y_target,
            x_orig,
            x_pert,
            method,
            weighted):

    sq_diff = lamb * (adv_logits - y_target) ** 2
    
    if method == 'L1_MAD':
        dist_loss = L1_MAD_weighted(x_pert,x_orig)
    
    elif method == 'Euclid':
        if weighted:
            dist_loss = Euclidean_dist(x_pert,x_orig,weighted=True)
        else:
            dist_loss = Euclidean_dist(x_pert,x_orig,weighted=False)
    
    else:
        raise ValueError("Method can only be L1_MAD or Euclid")

    return (sq_diff + dist_loss).mean()
# ==========================================================================================================================================================

# Model Training
# =========================================================================================================================================================
def get_trained_model(file_name,
                      train,
                      batch_size,
                      shuffle,
                      train_test_ratio,
                      num_epochs,
                      lr,
                      weight_decay,
                      device):
    print('======================================================================')
    data = pd.read_csv(file_name)
    scaler = MinMaxScaler()

    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    
    X_scaled = scaler.fit_transform(X)

    test_size = train_test_ratio
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                          Y,
                                                          test_size = test_size,
                                                          shuffle = shuffle
                                                          )

    X_train = torch.tensor(X_train).to(device = 'cuda:0',dtype = torch.float32)
    X_test = torch.tensor(X_test).to(device = 'cuda:0',dtype = torch.float32)
    y_train = torch.tensor(y_train).to(device = 'cuda:0',dtype = torch.float32)
    y_test = torch.tensor(y_test).to(device = 'cuda:0',dtype = torch.float32)

    training_data = CustomDataset(X_train,y_train)
    data_loader = DataLoader(training_data, batch_size = batch_size, shuffle = shuffle)

    if train:

        num_epochs = 200
        loss_fn = torch.nn.BCELoss()

        print('Training the model.')

        model = DenseModel(X_train.shape[1],1).to(device='cuda:0')
        optimizer = torch.optim.Adam(params=model.parameters(),lr = lr,weight_decay=weight_decay)

        training_bar = tqdm(range(num_epochs))

        for epoch in training_bar:
            for batch in data_loader:
            
                x,y = batch
                y_pred = model(x)
                    
                train_loss = loss_fn(y_pred,y.reshape(-1,1))
                optimizer.zero_grad()
                
                train_loss.backward()
                optimizer.step()

                acc = (y_pred.round().view(y.shape[0]) == y).float().mean()

            training_bar.set_postfix(Acc=float(acc))

        
        model.eval()

        print('Testing the model now.')

        with torch.no_grad():

            y_test_pred = model(X_test)
            y_test_pred = y_test_pred.round()

            cm = confusion_matrix(
                                  y_test.cpu().numpy(),
                                  y_test_pred.view(X_test.shape[0]).cpu().numpy()
                                )
            accu_score = accuracy_score(
                                        y_test.cpu().numpy(),
                                        y_test_pred.cpu().numpy()
                                      )
            print(f'The accuracy on test set is {accu_score*100:.2f}%')
        
        print("Saving model as trained_model.pt")
        torch.save(model,'trained_model.pt')

    else:
        print('Loading trained model.')
        model = torch.load('trained_model.pt')
        model.eval()
    
    print('======================================================================')
    return model
    
# ==========================================================================================================================================================


# Class for model and custom dataset
# ==========================================================================================================================================================        
class DenseModel(torch.nn.Module):

    def __init__(self,input_shape,output_shape):

        super().__init__()
        self.inputShape = input_shape
        self.outputShape = output_shape
        self.model = torch.nn.Sequential(

                              torch.nn.Linear(
                                              in_features=self.inputShape,
                                              out_features=20,
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=20,
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=20,
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=self.outputShape,
                                              ),

                              torch.nn.Sigmoid()
                     
                     )

    def forward(self,input):
        
        return self.model(input)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        
        self.X  = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================================================================================================================================
