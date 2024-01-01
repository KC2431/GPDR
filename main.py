from utils import *
from attacks import *

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_epochs = 400

    model = get_trained_model(file_name='diabetes.csv',
                              train=True,
                              batch_size=20,
                              shuffle=False,
                              train_test_ratio=0.2,
                              num_epochs=200,
                              lr=2e-3,
                              weight_decay=1e-3,
                              device=device)
    
    loss_fn = torch.nn.BCELoss()
    x_pert_SAIF = SAIF(model = model,
                       file_name='diabetes.csv',
                       labels=0.5,
                       loss_fn=loss_fn,
                       device=device,
                       num_epochs=num_epochs)
    
    X_pert_L1_MAD = L1_MAD_attack(file_name='diabetes.csv',
                                  device=device,
                                  target=0.5,
                                  lamb=5e-3,
                                  num_iters=num_epochs,
                                  optim_lr=2e-3,
                                  model=model)

    
    
    
    
