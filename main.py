from utils import *
from attacks import *
from convex_projection import convex_hull_proj

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_epochs = 500
    file_name = 'diabetes.csv'
    target = 0.5

    model = get_trained_model(file_name='diabetes.csv',
                              train=True,
                              batch_size=10,
                              shuffle=False,
                              train_test_ratio=0.2,
                              num_epochs=400,
                              lr=1e-2,
                              weight_decay=1e-3,
                              device=device)
    
    loss_fn = torch.nn.BCELoss()
    x_pert_SAIF = SAIF(model = model,
                       file_name=file_name,
                       labels=target,
                       loss_fn=loss_fn,
                       device=device,
                       num_epochs=num_epochs)
    
    X_pert_L1_MAD = L1_MAD_attack(file_name=file_name,
                                  device=device,
                                  target=target,
                                  lamb=10,
                                  num_iters=num_epochs,
                                  optim_lr=1e-2,
                                  model=model)

    
    NotInConvexHullData, InConvexHullData = convex_hull_proj(original_data_path='diabetes.csv',
                                                             adv_data_path='SAIFresults.csv',
                                                             trained_model_path='trained_model.pt',
                                                             device=device,
                                                             lamb=0.9,
                                                             optim_lr=1e-2,
                                                             num_iterates=201)    
    
    
