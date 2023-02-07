import torch
import numpy as np
from models.autoencoders import AE_MLP
from tqdm import tqdm

def train_embedding_model(params, train_loader, test_loader):
    device = torch.device('cpu')
    torch.manual_seed(params['seed']) # Seed for reproducibility
    np.random.seed(params['seed'])

    model = AE_MLP(input_dim=params['input_neurons'], output_dim=params['embedding_dims']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.BCELoss()
    n_train = len(train_loader)
    n_test = len(test_loader)

    train_loss=[]
    test_loss=[]
    early_stop = 0
    for epoch in tqdm(range(params["maxTrainSteps"])):
        run_train_loss = 0
        model.train()
        for i, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            reconstruction, _ = model(x)
            loss = criterion(reconstruction, x)
            loss.backward()
            optimizer.step()
            run_train_loss += loss.item()
            optimizer.zero_grad()
        train_loss.append(run_train_loss/n_train)

        run_test_loss = 0
        model.eval()
        for i, (x, _, _) in enumerate(test_loader):
            x = x.to(device)
            with torch.no_grad():
                reconstruction, _ = model(x)
                loss = criterion(reconstruction, x) # Only compute loss on masked part
                run_test_loss += loss.item()
        if run_train_loss/n_train < run_test_loss/n_test:
            early_stop += 1
        test_loss.append(run_test_loss/n_test)

        #print(f"Epoch: {epoch+1} \t Train Loss: {run_train_loss/n_train:.4f} \t Test Loss: {run_test_loss/n_test:.4f}")    
        torch.save({
                'params': params,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': test_loss,
                }, f'results/AE_model.pt')
        if early_stop == params['patience']:
            print("early stopping...")
            break

    return model, train_loss, test_loss