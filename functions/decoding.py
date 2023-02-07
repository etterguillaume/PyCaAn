import torch
from models.decoders import linear_decoder
from tqdm import tqdm

def train_linear_decoder(params, embedding_model, train_loader, test_loader):
    device = torch.device(params['device'])
    torch.manual_seed(params['seed']) # Seed for reproducibility
    
    for param in embedding_model.parameters(): 
        param.requires_grad = False # Freeze all weights from embedding model

    decoder = linear_decoder(input_dims=params['embedding_dims'], output_dims=1) #TODO flexibility for output dims
    criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=params['learning_rate'])
    n_train = len(train_loader)
    n_test = len(test_loader)
    
    train_loss=[]
    test_loss=[]
    early_stop = 0
    for epoch in tqdm(range(params["maxTrainSteps"])):
        run_train_loss = 0
        embedding_model.eval()
        decoder.train()
        for i, (x, position, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            _, embedding = embedding_model(x)
            pred = decoder(embedding)
            loss = criterion(pred.flatten(), position[:,0]) # TODO this is only for linear position
            loss.backward()
            optimizer.step()
            run_train_loss += loss.item()
        train_loss.append(run_train_loss/n_train)

        run_test_loss = 0
        embedding_model.eval()
        decoder.eval()
        for i, (x, position, _) in enumerate(test_loader):
            x = x.to(device)
            with torch.no_grad():
                _, embedding = embedding_model(x)
                pred = decoder(embedding)
                loss = criterion(pred.flatten(), position[:,0]) # TODO this is only for linear position
                run_test_loss += loss.item()
        if run_train_loss/n_train < run_test_loss/n_test:
            early_stop += 1
        test_loss.append(run_test_loss/n_test)

        #print(f"Epoch: {epoch+1} \t Train Loss: {run_train_loss/n_train:.4f} \t Test Loss: {run_test_loss/n_test:.4f}")    
        torch.save({
                'params': params,
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': test_loss,
                }, f'results/trained_models/embedding_decoder.pt')
        if early_stop == params['patience']:
            print("early stopping...")
            break

    return decoder, train_loss, test_loss