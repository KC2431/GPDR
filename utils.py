from scipy.stats import median_abs_deviation
import torch
from torch.utils.data import DataLoader, Dataset

def L1_MAD_weighted(x_pert,x_orig):
    
    MAD = median_abs_deviation(x_orig.cpu().numpy(),axis = 0)
    diff = torch.abs(x_orig - x_pert)
    return (diff / torch.tensor(MAD).cuda()).sum(dim = 1).sum()

def adv_loss(lamb,
            logits,
            y_target,
            x_orig,
            x_pert,
            method,
            weighted):

    sq_diff = lamb * (logits - y_target) ** 2
    
    if method == 'L1_MAD':
        dist_loss = L1_MAD_weighted(x_pert,x_orig)
    elif method == 'Euclid':
        if weighted:
            dist_loss = Euclidean_dist(x_pert,x_orig,weighted=True)
        else:
            dist_loss = Euclidean_dist(x_pert,x_orig,weighted=False)
    else:
        raise ValueError("Method can only be L1_MAD or L2_unwghtd")

    return sq_diff.mean() + dist_loss

def Euclidean_dist(X_pert,X_orig,weighted = False):

    loss = (X_orig - X_pert) ** 2
     
    if not weighted:
        loss = loss.sum(dim = 1).mean()
    
    else:
        loss = loss / X_orig.std(dim = 0)
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

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.X = self.data[:, :-1]  # All columns except the last one as input
        self.Y = self.data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]