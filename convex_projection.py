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
        trained_model_path: str,
        device: str,
        lamb: float,
        optim_lr: float,
        num_iterates: int
        ):

    from sklearn.preprocessing import MinMaxScaler
    
    model = torch.load(trained_model_path)

    df = pd.read_csv(original_data_path)
    saif = pd.read_csv(adv_data_path)

    req_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    df = df[df['Outcome'] == 1]
    df = df[req_cols]
    
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df.values)

    saif_scaler = MinMaxScaler()
    saif = saif[req_cols]
    saif = saif_scaler.fit_transform(saif.values)

    hull = ConvexHull(df)
    a = pointIsInConvexHull(hull, saif)
    saifNotInConvexHull = saif[~a]

    points = torch.tensor(df, device=device)
    saif = torch.tensor(saifNotInConvexHull, device=device, dtype = torch.float32)
    
    with torch.no_grad():
        model = model.to(device)
    

    saifClone = saif.clone()
    lamb = lamb
    
    before_projection = model(saif).round().reshape(-1)

    saif = torch.tensor(saifNotInConvexHull, device=device, dtype = torch.double)

    print(f'The avg L1 norm diff before: {torch.norm(saifClone - saif, p = 1, dim = 1).mean()}')
    print(f'Value of lambda is {lamb}')

    saif.requires_grad = True
    points_optimizer = torch.optim.Adam([saif], lr=1e-2)

    for epoch in range(num_iterates):
        points_optimizer.zero_grad()
        farthest_points = get_farthest_points(points, saif)
        loss = torch.norm(farthest_points - saif, p=1, dim = 1) + lamb * torch.norm(saifClone - saif,p = 1, dim = 1)
        loss = loss.mean() 
        loss.backward()
        points_optimizer.step()
            
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            NumPointsInHull = pointIsInConvexHull(hull, saif.detach().cpu().numpy())
            NumPointsInHull_true = NumPointsInHull.tolist().count(True)
            print(f'Total points projected into hull {NumPointsInHull_true}/{saif.shape[0]}')
    
    print(f'The avg L1 norm diff after: {torch.norm(saifClone - saif, p = 1, dim = 1).mean()}')

    after_projection = model(torch.tensor(saif, device=device, dtype = torch.float32)).round().reshape(-1)

    print(torch.norm(before_projection - after_projection,p = 1))
    print(saif_scaler.inverse_transform(saifClone.cpu().numpy()))
    print(saif_scaler.inverse_transform(saif.detach().cpu().numpy()))
    
    return saifClone, saif
