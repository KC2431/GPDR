import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

def L1_MAD_weighted(x_pert,x_orig):
    
    MAD = median_abs_deviation(x_orig.cpu().numpy(),axis = 0)
    diff = torch.abs(x_orig - x_pert)
    return (diff / torch.tensor(MAD).cuda()).sum(dim = 1).mean()

def adv_loss(lamb,
            logits,
            y_target,
            x_orig,
            x_pert,
            method):

    sq_diff = lamb * (logits - y_target) ** 2
    
    if method == 'L1_MAD':
        dist_loss = L1_MAD_weighted(x_pert,x_orig)
    elif method == 'L2_unwghtd':
        dist_loss = L2_unweighted_loss(x_pert,x_orig)
    else:
        raise ValueError("Method can only be L1_MAD or L2_unwghtd")

    return sq_diff.mean() + dist_loss

def L2_unweighted_loss(X_pert,X_orig):

    loss = (X_orig - X_pert) ** 2
    loss = loss.sum(dim = 1).mean()
    return loss


class DenseModel(torch.nn.Module):

    def __init__(self,input_shape,output_shape):

        super().__init__()
        self.inputShape = input_shape
        self.outputShape = output_shape
        self.model = torch.nn.Sequential(

                              torch.nn.Linear(
                                              in_features=self.inputShape,
                                              out_features=20
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=20
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=self.outputShape
                                              ),
                     
                     )

    def forward(self,input):
        
        return self.model(input)


if __name__ == '__main__':

    
    data = pd.read_csv('data_3.csv',index_col=0)
    
    X = data.iloc[:,:-1].to_numpy()
    Y = data.loc[:,'gpa'].to_numpy()

    test_size = 0.4
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

    train = True
    loss_fn = torch.nn.MSELoss()
    
    if train:
        num_epochs = 2000
        training_bar = tqdm(range(num_epochs))

        model = DenseModel(4,1).to(device='cuda:0')
        model.train()
        optimizer = torch.optim.Adam(params=model.parameters(),lr = 1e-3)

        for epoch in training_bar:

            model.zero_grad()
            y_pred = model(X_train)
            
            train_loss = loss_fn(y_pred,torch.reshape(y_train,(-1,1)))
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

    model = torch.load('trained_model.pt')
    model.eval()
    adv_optimizer = torch.optim.Adam([X_pert],lr = 1e-3)
    lamb = 1e-1
    num_iters = 6000
    pert_bar = tqdm(range(num_iters))

    for i in pert_bar:
        model.zero_grad()
        adv_logits = model(X_pert)
        
        loss = adv_loss(lamb,
                        adv_logits,
                        y_target,
                        X_train,
                        X_pert,
                        'L2_unwghtd'
                        )
        
        #loss = loss_fn(adv_logits,y_target.reshape(-1,1))
        adv_optimizer.zero_grad()

        loss.backward()
        adv_optimizer.step()
        pert_bar.set_postfix(loss = float(loss)) 
    print(X_train)
    print(model(X_pert))
    
    