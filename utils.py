import math
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.manual_seed(21)

# loss functions 
# ==========================================================================================================================================================
def L1_MAD_weighted(x_pert,x_orig):
    
    MAD = median_abs_deviation(x_orig.cpu().numpy(),axis = 0)
    diff = torch.abs(x_orig - x_pert)
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

# Adversarial attacks
# ==========================================================================================================================================================
def L1_MAD_attack(file_name,
                  device,
                  target,
                  lamb,
                  num_iters,
                  optim_lr,
                  model,
                  ):
    print('======================================================================')
    print('L1 weighted by MAD attack')
    data = pd.read_csv(file_name)
    scaler = MinMaxScaler()

    entire_X = data.iloc[:,:-1].values
    entire_Y = data.iloc[:,-1].values
    
    entire_X = scaler.fit_transform(entire_X)

    entire_X = torch.tensor(entire_X, dtype = torch.float32,device = device)
    entire_Y = torch.tensor(entire_Y, dtype=torch.float32,device = device)

    X_pert = torch.rand_like(entire_X,requires_grad=True,device = device)    
    y_target = torch.zeros_like(entire_Y, device=device)
    y_target.add_(target)

    adv_optimizer = torch.optim.Adam([X_pert],lr = optim_lr)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f'Performing attack on the dataset for {num_iters} epochs with the initial value of lambda as {lamb}.')
    pert_bar = tqdm(range(num_iters))

    for i in pert_bar:

        if i % 100 == 0:
            lamb *= 1.7

        adv_logits = model(X_pert)
        loss = adv_loss(lamb,
                        adv_logits,
                        y_target,
                        entire_X,
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

    entire_X = scaler.inverse_transform(entire_X.cpu().numpy())
    X_pert_inv_scaled = scaler.inverse_transform(X_pert.cpu().numpy())
    print(f'The mean L0 norm for the perturbation is {np.linalg.norm(entire_X.round(1) - X_pert_inv_scaled.round(1),ord=0,axis=1).mean()}')
    X_pert = torch.cat((torch.Tensor(X_pert_inv_scaled).abs(), X_pert_pred.cpu()), dim = 1)
    pert_data = pd.DataFrame(np.concatenate((X_pert_inv_scaled, X_pert_pred.cpu().numpy()), axis=1),columns=data.columns)
    pert_data.to_csv('results.csv',index=False)
    
    print('The Results have been saved as results.csv')

    print('======================================================================')
    return X_pert


def SAIF(model,
         file_name,
         labels,
         loss_fn,
         device,
         num_epochs,
         targeted = True,
         k = 3,
         eps = 1.0):


    print('======================================================================')
    print('SAIF method')

    data = pd.read_csv(file_name)
    scaler = MinMaxScaler()

    entire_X = data.iloc[:,:-1].values
    entire_Y = data.iloc[:,-1].values
    
    entire_X = scaler.fit_transform(entire_X)

    entire_X = torch.tensor(entire_X, dtype = torch.float32,device = device)
    entire_Y = torch.tensor(entire_Y, dtype=torch.float32,device = device)

    input_clone = entire_X.clone()
    input_clone.requires_grad = True

    y_target = torch.zeros_like(entire_Y, device=device)
    y_target.add_(labels)

    out = model(input_clone)
    loss = loss_fn(out,y_target.reshape(-1,1))
    loss.backward()

    p = -eps * input_clone.grad.sign()
    p = p.detach()
 
    kSmallest = torch.topk(-input_clone.grad,k = k,dim = 1)[1]
    kSmask = torch.zeros_like(input_clone.grad,device = device)
    kSmask.scatter_(1,kSmallest,1)
    s = kSmask.detach().float()

    r = 1

    epochs_bar = tqdm(range(num_epochs))

    for epoch in epochs_bar:

        s.requires_grad = True
        p.requires_grad = True
        out = model(entire_X + s*p)

        if targeted:
            loss = loss_fn(out,y_target.reshape(-1,1))
        else: 
            loss = -loss_fn(out,y_target)

        loss.backward()

        mp = p.grad
        ms = s.grad

        with torch.no_grad():

            v = -eps * mp.sign()
            
            kSmallest = torch.topk(-ms,k = k,dim = 1)[1]
            kSmask = torch.zeros_like(ms,device = device)
            kSmask.scatter_(1,kSmallest,1)
            
            z = torch.logical_and(kSmask, ms < 0).float()

            mu = 1 / (2 ** r * math.sqrt(epoch + 1))
            while loss_fn(model(entire_X + (s + mu * (z - s)) * (p + mu * (v - p))),y_target.reshape(-1,1)) > loss:
                r += 1
                mu = 1 / (2 ** r * math.sqrt(epoch + 1))

            p = p + mu * (v - p)
            s = s + mu * (z - s)

        epochs_bar.set_postfix(loss = float(loss))

    X_adv = entire_X + s*p
    X_adv_pred = model(X_adv)
    
    X_adv = scaler.inverse_transform(X_adv.cpu().numpy())
    entire_X = scaler.inverse_transform(entire_X.cpu().detach().numpy())
    
    L0norm_mean = np.linalg.norm(entire_X.round(1) - X_adv.round(1),ord=0,axis=1).mean()
    print(f'The mean L0 norm for the perturbation is {L0norm_mean}.')
    X_adv = torch.cat((torch.Tensor(X_adv).abs(), X_adv_pred.cpu()), dim = 1)
    pert_data = pd.DataFrame(X_adv.detach().numpy(),columns=data.columns)
    pert_data.to_csv('SAIFresults.csv',index=False)    

    print('The Results have been saved as SAIFresults.csv')
    print('======================================================================')

    return X_adv   
            
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