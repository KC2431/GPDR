from utils import *
from attacks import *
from convex_projection import convex_hull_proj

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_epochs = 500
    original_dataset = 'German_credit_data.csv'
    columns = get_column_names(original_dataset) 
    select_features = True

    if select_features:
        model, X_test, selected_cols = get_trained_model(file_name=original_dataset,
                               train=False,
                              dataset="german_credit_data",
                              select_features=select_features,
                              batch_size=32,
                              shuffle=False,
                              train_test_ratio=0.2,
                              num_epochs=500,
                              lr=1e-2,
                              weight_decay=1e-3,
                              device=device)
    else:
        model, X_test, selected_cols = get_trained_model(file_name=original_dataset,
                               train=False,
                              dataset="diabetes",
                              select_features=select_features,
                              batch_size=32,
                              shuffle=False,
                              train_test_ratio=0.2,
                              num_epochs=500,
                              lr=1e-2,
                              weight_decay=1e-3,
                              device=device)


    loss_fn = torch.nn.BCELoss()
    X_pert_SAIF = SAIF(model = model,
                        X=X_test,
                       loss_fn=loss_fn,
                       device=device,
                       num_epochs=num_epochs,
                       )
    
    X_pert_L1_MAD = L1_MAD_attack(X=X_test,
                                   device=device,
                                  lamb=1e-10,
                                  num_iters=num_epochs,
                                  optim_lr=1e-2,
                                  model=model
                                  )

    NotInConvexHullData, InConvexHullData = convex_hull_proj(original_data_path=original_dataset,
                                                              adv_data=X_pert_SAIF,
                                                             trained_model=model,
                                                             selected_cols=selected_cols,
                                                             X=X_test,
                                                             save_hull=True,
                                                             device=device,
                                                             lamb=0.6,
                                                             optim_lr=1e-2,
                                                             num_iterates=201)    

    NotInConvexHullData, InConvexHullData = convex_hull_proj(original_data_path=original_dataset,
                                                             adv_data=X_pert_L1_MAD,
                                                             trained_model=model,
                                                             selected_cols=selected_cols,
                                                             X=X_test,
                                                             save_hull=True,
                                                             device=device,
                                                             lamb=0.6,
                                                             optim_lr=1e-2,
                                                             num_iterates=201)
