import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm
from utils import L1_MAD_weighted, Euclidean_dist, DenseModel, adv_loss, CustomDataset


if __name__ == '__main__':

    torch.manual_seed(21)

    data = pd.read_csv('diabetes.csv')
    scaler = MinMaxScaler()

    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    
    X = scaler.fit_transform(X)

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                          Y,
                                                          test_size = test_size,
                                                          shuffle = False
                                                          )

    X_train = torch.tensor(X_train).to(device = 'cuda:0',dtype = torch.float32)
    X_test = torch.tensor(X_test).to(device = 'cuda:0',dtype = torch.float32)
    y_train = torch.tensor(y_train).to(device = 'cuda:0',dtype = torch.float32)
    y_test = torch.tensor(y_test).to(device = 'cuda:0',dtype = torch.float32)

    training_data = CustomDataset(X_train,y_train)
    data_loader = DataLoader(training_data, batch_size = 20, shuffle = False)

    train = True
    
    
    if train:

        num_epochs = 200
        loss_fn = torch.nn.BCELoss()

        print('Training the model.')

        model = DenseModel(X_train.shape[1],1).to(device='cuda:0')
        optimizer = torch.optim.Adam(params=model.parameters(),lr = 1e-3,weight_decay=1e-2)

        training_bar = tqdm(range(num_epochs))

        for epoch in training_bar:
            for batch in data_loader:
            
                x,y = batch
                y_pred = model(x)
                    
                train_loss = loss_fn(y_pred,y.reshape(-1,1))
                optimizer.zero_grad()
                
                train_loss.backward()
                optimizer.step()

                acc = (y_pred.round().view(y.shape[0]) == y).float().mean()

            training_bar.set_postfix(Acc=float(acc))

        model.eval()

        print('Testing the model now.')

        with torch.no_grad():

            y_test_pred = model(X_test)
            y_test_pred = y_test_pred.round()

            cm = confusion_matrix(
                                  y_test.cpu().numpy(),
                                  y_test_pred.view(X_test.shape[0]).cpu().numpy()
                                )
            accu_score = accuracy_score(
                                        y_test.cpu().numpy(),
                                        y_test_pred.cpu().numpy()
                                      )
            
            sns.heatmap(cm,annot=True)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            plt.title("Confusion matrix for PIMA dataset for test set")
            plt.show()

            print(f'The accuracy on test set is {accu_score*100:.2f}%')

    entire_X = torch.cat((X_train,X_test),dim=0)
    entire_Y = torch.cat((y_train,y_test),dim=0)

    X_pert = torch.rand_like(entire_X,requires_grad=True,device = 'cuda:0')    
    y_target = torch.zeros_like(entire_Y).to('cuda:0')
    y_target.add_(0.5)

    model.eval()
    adv_optimizer = torch.optim.Adam([X_pert],lr = 1e-3)
    lamb = 3e-3
    num_iters = 2100
    
    for param in model.parameters():
        param.requires_grad = False

    print(f'Performing attack on the dataset for {num_iters} epochs.')
    pert_bar = tqdm(range(num_iters))

    for i in pert_bar:

        if i % 100 == 0:
            lamb *= 1.8

        adv_logits = model(X_pert)
        loss = adv_loss(lamb,
                        adv_logits,
                        y_target,
                        entire_X,
                        X_pert,
                        'L1_MAD',
                        False
                        )
        
        adv_optimizer.zero_grad()

        loss.backward()
        adv_optimizer.step()
        pert_bar.set_postfix(loss = float(loss)) 
    
    X_pert = X_pert.detach()
    
    with torch.no_grad():
        X_pert_pred = model(X_pert)
    
    X_pert = scaler.inverse_transform(X_pert.cpu().numpy())
    X_pert = torch.cat((torch.Tensor(X_pert).abs(), X_pert_pred.cpu()), dim = 1)
    pert_data = pd.DataFrame(X_pert.numpy(),columns=data.columns)
    pert_data.to_csv('results.csv',index=False)
    
    print('The Results have been saved as results.csv')
    

    
    
    
    
