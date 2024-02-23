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
        X (_type_): _description_
        trained_model_path (str): _description_
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

    df = pd.read_csv(original_data_path)
    saif = pd.read_csv(adv_data_path)

    req_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    df = df[df['Outcome'] == 1]
    df = df[req_cols]
    
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df.values)

    origSaif = saif[req_cols]
    saif_outcome_one  = saif[saif['Outcome'] == 1.0]
    saif_outcome_one = saif_outcome_one[req_cols]
    

    hull = ConvexHull(df)
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
    points_optimizer = torch.optim.Adam([saifNotInConvexHull], lr=1e-2)

    for epoch in range(num_iterates):
        points_optimizer.zero_grad()
        farthest_points = get_farthest_points(points, saifNotInConvexHull.detach())
        loss = torch.norm(farthest_points - saifNotInConvexHull, p=1, dim = 1) + lamb * torch.norm(saifNotInConvexHullClone - saifNotInConvexHull,p = 1, dim = 1)
        loss = loss.mean() 
        loss.backward()
        points_optimizer.step()
            
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            NumPointsInHull = pointIsInConvexHull(hull, saifNotInConvexHull.detach().cpu().numpy())
            NumPointsInHull_true = NumPointsInHull.tolist().count(True)
            print(f'Total points projected into hull {NumPointsInHull_true}/{saifNotInConvexHull.shape[0]}')
    

    saifNotInConvexHull = saifNotInConvexHull.detach()
    saif_outcome_one[~a] = saifNotInConvexHull.cpu().numpy()
    saif.loc[saif["Outcome"] == 1.0, req_cols] = saif_outcome_one.values

    pert = np.abs(X.cpu().numpy() - saif[req_cols].values)
    pert = np.where(pert < 0.005, np.array(0.0), pert)
    print(f'The avg L0 norm after convex projection: {np.linalg.norm(pert, ord = 0, axis = 1).mean()} and the shape is {np.linalg.norm(pert, ord = 0, axis = 1).shape}')

    with torch.no_grad():
        pert_output = model(torch.tensor(saif[req_cols].values, dtype=torch.float32).cuda()).round()
        orig_output = model(torch.tensor(origSaif.values, dtype=torch.float32).cuda()).round()
        print(f"Number of counterfactuals that were changed: {torch.sum(orig_output.squeeze() != orig_output.squeeze())}")
    
    print('======================================================================')
    return saifNotInConvexHullClone, saifNotInConvexHull
