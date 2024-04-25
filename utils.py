import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
from scipy.stats import median_abs_deviation
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.manual_seed(10)

# Dataframe functions
# ==========================================================================================================================================================
def get_column_names(file_name):
    data = pd.read_csv(file_name)
    return data.columns

# loss functions 
# ==========================================================================================================================================================
def L1_MAD_weighted(x_pert,x_orig):
    
    MAD = median_abs_deviation(x_orig.cpu().numpy(),axis = 0)
    MAD = np.where(np.abs(MAD) < 0.0001, np.array(0.01), MAD)
    diff = torch.abs(x_orig - x_pert)
    return (diff / torch.tensor(MAD, device=x_orig.device)).sum(dim = 1)


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
            method):

    sq_diff = lamb * (adv_logits - y_target) ** 2
    
    if method == 'L1_MAD':
        dist_loss = L1_MAD_weighted(x_pert,x_orig)
    
    elif method == 'Euclid':
        dist_loss = Euclidean_dist(x_pert,x_orig,weighted=True)
        
    else:
        raise ValueError("Method can only be L1_MAD or Euclid")

    return (sq_diff + dist_loss).mean()

# ==========================================================================================================================================================
# Convex Hull projection and check
def pointIsInConvexHull(hull, point):
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    eps = np.finfo(np.float32).eps
    return contained(point, eps, A, b)


def contained(x, eps, A, b):
    return np.all(np.asarray(x) @ A.T + b.T < eps, axis=1)

# Model Training
# =========================================================================================================================================================
def get_trained_model(file_name,
                      train,
                      dataset,
                      select_features,
                      batch_size,
                      shuffle,
                      train_test_ratio,
                      num_epochs,
                      lr,
                      weight_decay,
                      device):
    print('======================================================================')

    available_dataset = ['diabetes','german_credit_data']
    if dataset not in available_dataset:
        raise ValueError(f"The dataset should be either {available_dataset[0]} or {available_dataset[1]}")

    data = pd.read_csv(file_name)
    scaler = MinMaxScaler()

    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    
    if select_features:
        max_features = 8 
        rfe = RFE(RandomForestClassifier(random_state=10), n_features_to_select=max_features)
        
        rfe.fit(X,Y)
        X = rfe.transform(X)

        columns_selected = data.iloc[:,:-1].columns[rfe.support_]
        print(columns_selected)
    else:
        columns_selected = data.iloc[:,:-1].columns

    test_size = train_test_ratio
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                          Y,
                                                          test_size = test_size,
                                                          shuffle = shuffle
                                                          )

    
    X_train = torch.tensor(scaler.fit_transform(X_train)).to(device = device,dtype = torch.float32)
    X_test = torch.tensor(scaler.transform(X_test)).to(device = device,dtype = torch.float32)
    y_train = torch.tensor(y_train).to(device = device,dtype = torch.float32)
    y_test = torch.tensor(y_test).to(device = device,dtype = torch.float32)

    training_data = CustomDataset(X_train,y_train)
    data_loader = DataLoader(training_data, batch_size = batch_size, shuffle = True)

    if train:

        num_epochs = num_epochs
        loss_fn = torch.nn.BCELoss()

        print('Training the model.')

        model = DenseModel(X_train.shape[1],1).to(device=device)
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

        print(f"Saving model as {dataset}_trained_model.pt")
        torch.save(model,f'{dataset}_trained_model.pt')

    else:
        print(f'Loading trained model for dataset {dataset}.')
        model = torch.load(f'{dataset}_trained_model.pt')
        model.eval()

        with torch.no_grad():
            print(f"The accuracy for the pretrained model is {torch.sum(model(X_test).round().squeeze() == y_test)}/{X_test.shape[0]}")
    
    print('======================================================================')
    
    Y = np.reshape(Y, (-1,1))
    X = torch.cat((X_train, X_test),dim = 0)
    scaled_data = pd.DataFrame(np.concatenate((X.cpu().numpy(),Y),axis=1))
    X_sampled = scaled_data[scaled_data.iloc[:,-1] == 1].sample(n=100,random_state=22).iloc[:,:-1]
    X_sampled = torch.tensor(X_sampled.values, dtype=torch.float32,device=device)
    
    return model, X_sampled, columns_selected, scaler
    
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
