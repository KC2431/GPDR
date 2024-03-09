import numpy as np
import torch
from scipy.spatial import ConvexHull
import numpy as np
from utils import pointIsInConvexHull
import pandas as pd
from scipy.spatial import ConvexHull
import pickle

def get_farthest_points(points, target_points):
    distances = torch.cdist(points, target_points,p=1)
    farthest_indices = torch.argmax(distances,dim=0)
    return points[farthest_indices]
    


def convex_hull_proj(
        original_data_path: str,
        adv_data,
        X,
        selected_cols,
        save_hull,
        trained_model,
        device: str,
        lamb: float,
        optim_lr: float,
        num_iterates: int,
        scaler
        ):
    """covex_hull_proj

    Args:
        original_data_path (str): _description_
        adv_data_path (str): _description_
        X (_type_): For checking the L0 norm
        Y (tensor): For checking how many counterfactuals changed
        trained_model_path (str): 
        device (str): _description_
        lamb (float): _description_
        optim_lr (float): _description_
        num_iterates (int): _description_

    Returns:
        _type_: _description_
    """
    print('======================================================================')
    print('Performing Convex Hull Projection')

    model = trained_model
    
    for param in model.parameters():
        param.requires_grad = False

    df = pd.read_csv(original_data_path)
    saif = adv_data

    req_cols = selected_cols
    target_col = df.columns[-1]

    df = df[df[target_col] == 0]
    df = df[req_cols]
    
    df = scaler.transform(df.values)

    origSaif = saif.clone()
    saif_outcome_one  = saif
    
    hull = ConvexHull(df,qhull_options='Qx')


    if save_hull:
        pickle.dump(hull,open("convex.hull", "wb"))
 
    a = pointIsInConvexHull(hull, saif_outcome_one.cpu().numpy())

    print(f"Number of points in Convex Hull before projection {a.tolist().count(True)}/{saif_outcome_one.shape[0]}")
    saifNotInConvexHull = saif_outcome_one[~a].double()

    points = torch.tensor(df[hull.vertices], device=device)
    model = model.to(device)

    saifNotInConvexHullClone = saifNotInConvexHull.clone()
    lamb = lamb
    
    print(f'Value of lambda is {lamb}')

    saifNotInConvexHull.requires_grad = True
    points_optimizer = torch.optim.Adam([saifNotInConvexHull], lr=optim_lr)
    
    print("-----------------------------------------------------------------")
    for epoch in range(num_iterates):
        points_optimizer.zero_grad()
        farthest_points = get_farthest_points(points, saifNotInConvexHull)
        loss = torch.norm(farthest_points - saifNotInConvexHull, p=1, dim = 1) + lamb * torch.norm(saifNotInConvexHullClone - saifNotInConvexHull,p = 1, dim = 1)
        loss = loss.mean() 
        loss.backward()
        points_optimizer.step()
            
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.2f}')
            NumPointsInHull = pointIsInConvexHull(hull, saifNotInConvexHull.detach().cpu().numpy())
            NumPointsInHull_true = NumPointsInHull.tolist().count(True)
            print(f'Total points projected into hull {NumPointsInHull_true}/{saifNotInConvexHull.shape[0]}')
    
    print("-----------------------------------------------------------------")

    saifNotInConvexHull = saifNotInConvexHull.detach()
    saif_outcome_one[~a] = saifNotInConvexHull.float().clone()

    print(f"After the projection, number of points projected: {pointIsInConvexHull(hull, saif_outcome_one.cpu().numpy()).tolist().count(True)}/{saif_outcome_one.shape[0]}")

    pert = np.abs(scaler.inverse_transform(X.cpu().numpy()) - scaler.inverse_transform(saif_outcome_one.cpu().numpy()))
    pert = np.where(pert < 0.1, np.array(0.0), pert)
    print(f'The avg L0 norm after convex projection: {torch.norm(torch.tensor(pert),p=0)}')

    with torch.no_grad():
        pert_output = model(saif_outcome_one).round()
        orig_output = model(origSaif).round()
        print(f"Number of counterfactuals that were changed: {torch.sum(pert_output.squeeze() != orig_output.squeeze())}")
    
    print('======================================================================')
    return saifNotInConvexHullClone, saifNotInConvexHull
