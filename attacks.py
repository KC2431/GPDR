# Importing The libraries

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
from utils import adv_loss

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

        X_pert.requires_grad = True

        """
        if i % 50 == 0:
            lamb *= 1.7
        """

        adv_logits = model(X_pert)
        loss = adv_loss(lamb=lamb,
                        adv_logits=adv_logits,
                        y_target=y_target,
                        x_orig=entire_X,
                        x_pert=X_pert,
                        method='L1_MAD'
                        )

        adv_optimizer.zero_grad()
        loss.backward()
        adv_optimizer.step()

        pert_bar.set_postfix(loss = float(loss)) 
        

    X_pert = X_pert.detach()

    with torch.no_grad():
        X_pert_pred = model(X_pert)
    

    avg_L0_norm = torch.norm(torch.round(entire_X - X_pert,decimals = 3),p = 0, dim = 1).mean()

    entire_X = scaler.inverse_transform(entire_X.cpu().numpy())
    X_pert_inv_scaled = scaler.inverse_transform(X_pert.cpu().numpy())

    print(f'The mean L0 norm for the perturbation is {avg_L0_norm}')
    pert_data = pd.DataFrame(np.concatenate((X_pert_inv_scaled.round(3), X_pert_pred.cpu().numpy()), axis=1),columns=data.columns)
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
         k = 2,
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
    loss = -loss_fn(out,y_target.reshape(-1,1))
    loss.backward()

    p = -eps * input_clone.grad.sign()
    p = p.detach()
 
    kSmallest = torch.topk(-input_clone.grad,k = k,dim = 1)[1]
    kSmask = torch.zeros_like(input_clone.grad,device = device)
    kSmask.scatter_(1,kSmallest,1)
    s = kSmask.detach().float()

    r = 1

    print(f'Performing attack on the dataset for {num_epochs} epochs')

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
            
            X_adv = torch.clamp(entire_X + p, 0,1)
            p = X_adv - entire_X
            
        epochs_bar.set_postfix(loss = float(loss))

    X_adv = entire_X + s*p
    X_adv_pred = model(X_adv)
    
    avg_L0_norm = torch.norm(torch.round(entire_X - X_adv,decimals = 3),p = 0, dim = 1).mean()

    X_adv = scaler.inverse_transform(X_adv.cpu().numpy())
    entire_X = scaler.inverse_transform(entire_X.cpu().detach().numpy())
    
    print(f'The mean L0 norm for the perturbation is {avg_L0_norm}.')
    X_adv_df = torch.cat((torch.Tensor(X_adv), X_adv_pred.cpu()), dim = 1)
    pert_data = pd.DataFrame(X_adv_df.detach().numpy(),columns=data.columns)
    pert_data.to_csv('SAIFresults.csv',index=False)    

    print('The Results have been saved as SAIFresults.csv')
    print('======================================================================')

    return X_adv   
            
# ==========================================================================================================================================================
