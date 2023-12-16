from scipy.stats import median_abs_deviation
import torch
from torch.utils.data import DataLoader, Dataset

def L1_MAD_weighted(x_pert,x_orig):
    
    MAD = median_abs_deviation(x_orig.cpu().numpy(),axis = 0)
    diff = torch.abs(x_orig - x_pert)
    return (diff / torch.tensor(MAD).cuda()).sum(dim = 1)

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

def Euclidean_dist(X_pert,X_orig,weighted = False):

    loss = (X_orig - X_pert) ** 2
     
    if not weighted:
        loss = loss.sum(dim = 1)
    
    else:
        loss = torch.div(loss, X_orig.std(dim = 0))
        loss = loss.sum(dim = 1)

    return loss


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
