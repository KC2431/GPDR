# Importing The libraries

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
from utils import adv_loss

torch.manual_seed(11)
# Adversarial attacks
# ==========================================================================================================================================================

def L1_MAD_attack(X,
                  Y,
                  device,
                  lamb,
                  num_iters,
                  optim_lr,
                  model,
                  ):
    """
    Args:
        file_name (_type_): _description_
        device (_type_): _description_
        target (_type_): _description_
        lamb (_type_): _description_
        num_iters (_type_): _description_
        optim_lr (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    print('======================================================================')
    print('L1 weighted by MAD attack')

    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


    entire_X = X
    entire_Y = Y

    X_pert = entire_X.clone() + 0.01 * torch.rand_like(entire_X)
    X_pert.requires_grad = True
    X_pert = X_pert.to(device)
        
    y_target = 1 - entire_Y
    y_target = y_target.to(device)
    

    adv_optimizer = torch.optim.Adam([X_pert],lr = optim_lr)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f'Performing attack on the dataset for {num_iters} epochs with the initial value of lambda as {lamb}.')
    pert_bar = tqdm(range(num_iters))

    for i in pert_bar:

        X_pert.requires_grad = True

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

        if i % 10 == 0:
            lamb *= 2.0
        
    X_pert = X_pert.detach()
    X_pert = torch.where(X_pert > 1, torch.ones_like(X_pert), X_pert)
    X_pert = torch.where(X_pert < 0, torch.zeros_like(X_pert), X_pert)

    with torch.no_grad():
        X_pert_pred = torch.round(model(X_pert))
        print(f"Number of successful counterfactuals : {torch.sum(X_pert_pred.squeeze() != entire_Y.squeeze())} / {entire_X.shape[0]}")
    
    avg_L0_norm = torch.abs(entire_X - X_pert)
    avg_L0_norm = torch.where(avg_L0_norm < 0.001, torch.tensor(0.0, device=device), avg_L0_norm)
    avg_L0_norm = torch.norm(avg_L0_norm, p = 0, dim = 1).mean()


    print(f'The mean L0 norm for the perturbation is {avg_L0_norm}')
    pert_data = pd.DataFrame(np.concatenate((X_pert.detach().cpu().numpy(), X_pert_pred.cpu().numpy()), axis=1),columns=columns)
    pert_data.to_csv('results.csv',index=False)
    
    print('The Results have been saved as results.csv')
    print('======================================================================')
    
    return X_pert


def SAIF(model,
         X,
         Y,
         loss_fn,
         device,
         num_epochs,
         targeted = True,
         k = 2,
         eps = 1.0):
    """SAIF

    Args:
        model (_type_): _description_
        file_name (_type_): _description_
        labels (_type_): _description_
        loss_fn (_type_): _description_
        device (_type_): _description_
        num_epochs (_type_): _description_
        targeted (bool, optional): _description_. Defaults to True.
        k (int, optional): _description_. Defaults to 1.
        eps (float, optional): _description_. Defaults to 1.0.

    Returns:
        X_adv: Adversarial instance of the input
    """


    print('======================================================================')
    print('SAIF method')

    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    entire_X = X
    entire_Y = Y

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    input_clone = entire_X.clone()
    input_clone.requires_grad = True

    y_target = 1 - entire_Y
    y_target = y_target.to(device)

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
    X_adv = torch.where(X_adv > 1, torch.ones_like(X_adv), X_adv)
    X_adv = torch.where(X_adv < 0, torch.zeros_like(X_adv), X_adv)

    with torch.no_grad():
        X_adv_pred = torch.round(model(X_adv))
        print(f"Number of successful counterfactuals : {torch.sum(X_adv_pred.squeeze() != entire_Y.squeeze())} / {entire_X.shape[0]}")
    
    avg_L0_norm = torch.abs(entire_X - X_adv)
    avg_L0_norm = torch.where(avg_L0_norm < 0.001, torch.tensor(0.0, device=device), avg_L0_norm)
    avg_L0_norm = torch.norm(avg_L0_norm, p = 0, dim = 1).mean() 

    X_adv = X_adv.cpu().numpy()
    entire_X = entire_X.cpu().detach().numpy()
    
    print(f'The mean L0 norm for the perturbation is {avg_L0_norm}.')
    X_adv_df = torch.cat((torch.Tensor(X_adv), X_adv_pred.cpu()), dim = 1)
    pert_data = pd.DataFrame(X_adv_df.detach().numpy(),columns=columns)
    pert_data.to_csv('SAIFresults.csv',index=False)    

    print('The Results have been saved as SAIFresults.csv')
    print('======================================================================')

    return X_adv   
            
# ==========================================================================================================================================================
