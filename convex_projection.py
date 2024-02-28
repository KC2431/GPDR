import numpy as np
import torch
from scipy.spatial import ConvexHull
import numpy as np
from utils import pointIsInConvexHull
import pandas as pd
from scipy.spatial import ConvexHull


def get_farthest_points(points, target_points):
    distances = torch.cdist(points, target_points,p=1)
    farthest_indices = torch.argmax(distances,dim=0)
    return points[farthest_indices]
    


def convex_hull_proj(
        original_data_path: str,
        adv_data_path: str,
        X,
        Y,
        selected_cols,
        trained_model_path: str,
        device: str,
        lamb: float,
        optim_lr: float,
        num_iterates: int
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

    from sklearn.preprocessing import MinMaxScaler

    print(f"Dataset : {adv_data_path}")

    model = torch.load(trained_model_path)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    df = pd.read_csv(original_data_path)
    saif = pd.read_csv(adv_data_path)

    req_cols = selected_cols
    target_col = df.columns[-1]

    df = df[df[target_col] == 0]
    df = df[req_cols]
    
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df.values)

    

    origSaif = torch.tensor(saif[req_cols].values, dtype=torch.float32)
    saif_outcome_one  = saif[saif[target_col] == 0.0]
    saif_outcome_one = saif_outcome_one[req_cols]
    
    print(df.shape)
    hull = ConvexHull(df, qhull_options="Qx")
    print("Convex hull done")
    a = pointIsInConvexHull(hull, saif_outcome_one)

    print(f"Number of points in Convex Hull before projection {a.tolist().count(True)}/{saif_outcome_one.shape[0]}")
    saifNotInConvexHull = saif_outcome_one[~a].values

    points = torch.tensor(df[hull.vertices], device=device)
    saifNotInConvexHull = torch.tensor(saifNotInConvexHull, device=device, dtype = torch.double)
    
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
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            NumPointsInHull = pointIsInConvexHull(hull, saifNotInConvexHull.detach().cpu().numpy())
            NumPointsInHull_true = NumPointsInHull.tolist().count(True)
            print(f'Total points projected into hull {NumPointsInHull_true}/{saifNotInConvexHull.shape[0]}')
    
    print("-----------------------------------------------------------------")

    saifNotInConvexHull = saifNotInConvexHull.detach()
    saif_outcome_one[~a] = saifNotInConvexHull.cpu().numpy()

    print(f"After the projection, number of points projected: {pointIsInConvexHull(hull, saif_outcome_one).tolist().count(True)}/{saif_outcome_one.shape[0]}")

    saif.loc[saif["Outcome"] == 1.0, req_cols] = saif_outcome_one.values

    pert = np.abs(X.cpu().numpy() - saif[req_cols].values)
    pert = np.where(pert < 0.005, np.array(0.0), pert)
    print(f'The avg L0 norm after convex projection: {np.linalg.norm(pert, ord = 0, axis = 1).mean()}')

    with torch.no_grad():
        pert_output = model(torch.tensor(saif[req_cols].values, dtype=torch.float32).cuda()).round()
        orig_output = model(origSaif.cuda()).round()
        print(f"Number of counterfactuals that were changed: {torch.sum(pert_output.squeeze() != orig_output.squeeze())}")
    
    print('======================================================================')
    return saifNotInConvexHullClone, saifNotInConvexHull
