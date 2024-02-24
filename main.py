from utils import *
from attacks import *
from convex_projection import convex_hull_proj

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_epochs = 500

    model, X_test, y_test = get_trained_model(file_name='diabetes.csv',
                              train=False,
                              batch_size=35,
                              shuffle=False,
                              train_test_ratio=0.13,
                              num_epochs=500,
                              lr=1e-2,
                              weight_decay=1e-3,
                              device=device)
    
    loss_fn = torch.nn.BCELoss()
    x_pert_SAIF = SAIF(model = model,
                       X=X_test,
                       Y=y_test,
                       loss_fn=loss_fn,
                       device=device,
                       num_epochs=num_epochs)
    
    X_pert_L1_MAD = L1_MAD_attack(X=X_test,
                                  Y=y_test,
                                  device=device,
                                  lamb=1e-10,
                                  num_iters=num_epochs,
                                  optim_lr=1e-2,
                                  model=model)

    NotInConvexHullData, InConvexHullData = convex_hull_proj(original_data_path='diabetes.csv',
                                                             adv_data_path='SAIFresults.csv',
                                                             trained_model_path='trained_model.pt',
                                                             X=X_test,
                                                             Y=y_test,
                                                             device=device,
                                                             lamb=0.9,
                                                             optim_lr=1e-2,
                                                             num_iterates=201)    

    NotInConvexHullData, InConvexHullData = convex_hull_proj(original_data_path='diabetes.csv',
                                                             adv_data_path='results.csv',
                                                             trained_model_path='trained_model.pt',
                                                             X=X_test,
                                                             Y=y_test,
                                                             device=device,
                                                             lamb=0.9,
                                                             optim_lr=1e-2,
                                                             num_iterates=201)
