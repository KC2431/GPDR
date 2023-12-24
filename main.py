from utils import *


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = get_trained_model(file_name='diabetes.csv',
                              train=True,
                              batch_size=20,
                              shuffle=False,
                              train_test_ratio=0.2,
                              num_epochs=200,
                              lr=2e-3,
                              weight_decay=1e-3,
                              device=device)
    
    """
    X_adv = SAIF(model,
                 entire_X,
                 y_target,
                 loss_fn,
                 device,
                 100
                 )
    X_adv_pred = model(X_adv)
    X_adv = scaler.inverse_transform(X_adv.cpu().numpy())
    print(X_adv)
    print(np.linalg.norm(X - X_adv.round(3),ord=0,axis=1).mean())
    X_adv = torch.cat((torch.Tensor(X_adv).abs(), X_adv_pred.cpu()), dim = 1)
    pert_data = pd.DataFrame(X_adv.detach().numpy(),columns=data.columns)
    pert_data.to_csv('SAIFresults.csv',index=False)
    """
    
    X_pert_L1_MAD = L1_MAD_attack('diabetes.csv',
                                   device,
                                   0.5,
                                   5e-3,
                                   300,
                                   2e-3,
                                   model)

    
    
    
    
