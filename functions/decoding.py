import torch

def train_embedding_decoder(params, embedding_model, train_loader, test_loader):
    device = torch.device('cpu')
    torch.manual_seed(params['seed']) # Seed for reproducibility
    np.random.seed(params['seed'])
    
    for param in embedding_model.parameters():
        param.requires_grad = False # Freeze all weights

    decoder_optimizer = torch.optim.AdamW(embedding_decoder.parameters(), lr=params['learning_rate'])
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_loss=[]
    test_loss=[]
    for epoch in range(params["maxTrainSteps"]):
        run_train_loss = 0
        model.eval()
        embedding_decoder.train()
        for i, (x, position, _) in enumerate(train_loader):
            decoder_optimizer.zero_grad()
            x = x.to(device)
            _, embedding = model(x)
            pred = embedding_decoder(embedding)
            loss = criterion(pred, position[:,0]) # TODO this is only for linear position
            loss.backward()
            decoder_optimizer.step()
            run_train_loss += loss.item()
        train_loss.append(run_train_loss/n_train)

        run_test_loss = 0
        model.eval()
        for i, (x, position, _) in enumerate(test_loader):
            x = x.to(device)
            with torch.no_grad():
                _, embedding = model(x)
                pred = embedding_decoder(embedding)
                loss = criterion(pred, position[:,0]) # TODO this is only for linear position
                run_test_loss += loss.item()
        test_loss.append(run_test_loss/n_test)

        print(f"Epoch: {epoch+1} \t Train Loss: {run_train_loss/n_train:.4f} \t Test Loss: {run_test_loss/n_test:.4f}")    
        torch.save({
                'params': params,
                'epoch': epoch,
                'model_state_dict': embedding_decoder.state_dict(),
                'optimizer_state_dict': decoder_optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': test_loss,
                }, f'results/embedding_decoder.pt')


    train_decoder(model, embedding_decoder, train_loader, test_loader, optimizer, MSE_criterion, device, params)