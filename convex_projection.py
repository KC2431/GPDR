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
        trained_model_path: str,
        device: str,
        lamb: float,
        optim_lr: float,
        num_iterates: int
        ):

    from sklearn.preprocessing import MinMaxScaler

    print(f"Dataset : {adv_data_path}")
    model = torch.load(trained_model_path)

    df = pd.read_csv(original_data_path)
    saif = pd.read_csv(adv_data_path)

    req_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    df = df[df['Outcome'] == 1]
    origData = df
    df = df[req_cols]
    
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df.values)

    saif_scaler = MinMaxScaler()
    saif = saif[req_cols]
    saif = saif_scaler.fit_transform(saif.values)

    hull = ConvexHull(df)
    a = pointIsInConvexHull(hull, saif)

    print(f"Number of points in Convex Hull before projection {a.tolist().count(True)}/{saif.shape[0]}")
    saifNotInConvexHull = saif[~a]

    points = torch.tensor(df, device=device)
    saifNotInConvexHull = torch.tensor(saifNotInConvexHull, device=device, dtype = torch.double)
    
    with torch.no_grad():
        model = model.to(device)

    saifNotInConvexHullClone = saifNotInConvexHull.clone()
    lamb = lamb
    
    print(f'The avg L1 norm diff before: {torch.norm(saifNotInConvexHullClone.cpu() - saifNotInConvexHull.cpu(), p = 1, dim = 1).mean()}')
    print(f'Value of lambda is {lamb}')

    saifNotInConvexHull.requires_grad = True
    points_optimizer = torch.optim.Adam([saifNotInConvexHull], lr=1e-2)

    for epoch in range(num_iterates):
        points_optimizer.zero_grad()
        farthest_points = get_farthest_points(points, saifNotInConvexHull)
        loss = torch.norm(farthest_points - saifNotInConvexHull, p=1, dim = 1) + lamb * torch.norm(saifNotInConvexHullClone - saifNotInConvexHull,p = 1, dim = 1)
        loss = loss.mean() 
        loss.backward()
        points_optimizer.step()
            
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            NumPointsInHull = pointIsInConvexHull(hull, saifNotInConvexHull.detach().cpu().numpy())
            NumPointsInHull_true = NumPointsInHull.tolist().count(True)
            print(f'Total points projected into hull {NumPointsInHull_true}/{saifNotInConvexHull.shape[0]}')
    
    print(f'The avg L1 norm diff after: {torch.norm(saifNotInConvexHullClone - saifNotInConvexHull, p = 1, dim = 1).mean()}')

    saifNotInConvexHull = saifNotInConvexHull.detach()
    saif[~a] = saifNotInConvexHull.cpu().numpy()

    with torch.no_grad():
        output = model(torch.tensor(saif, dtype=torch.float32).cuda())

    saif = np.concatenate([saif, output.detach().cpu().numpy()], axis=1)
    pd.DataFrame(saif, columns=origData.columns).to_csv(f"{adv_data_path}_proj.csv",index = False)

    return saifNotInConvexHullClone, saifNotInConvexHull
