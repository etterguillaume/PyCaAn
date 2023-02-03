import torch

def train_embedder(model, train_loader, test_loader, optimizer, criterion, device, params):
    n_train = len(train_loader)
    n_test = len(test_loader)

    train_loss=[]
    test_loss=[]
    early_stop = 0
    for epoch in range(params["maxTrainSteps"]):
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

        print(f"Epoch: {epoch+1} \t Train Loss: {run_train_loss/n_train:.4f} \t Test Loss: {run_test_loss/n_test:.4f}")    
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

def train_decoder(model, embedding_decoder, train_loader, test_loader, decoder_optimizer, criterion, device, params):
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
            loss = criterion(pred, position) # TODO this is only for 
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
                loss = criterion(pred, position) # Only compute loss on masked part
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

#%%
# Plot reconstruction examples
# with torch.no_grad():
#     reconstruction, embedding = model(torch.tensor(data['caTrace'],dtype=torch.float))

# #%%
# #lower_bound_loss = criterion(torch.tensor(data['caTrace'],dtype=torch.float),torch.tensor(data['caTrace'],dtype=torch.float))
# #upper_bound_loss = criterion(torch.tensor(data['caTrace'],dtype=torch.float),torch.tensor(~data['caTrace'],dtype=torch.float))
# error = criterion(reconstruction, torch.tensor(data['caTrace'],dtype=torch.float))

# #%%
# plt.subplot(121)
# max_val=torch.max(torch.tensor(data['caTrace'],dtype=torch.float))
# cells2plot = 10
# for i in range(cells2plot):
#     plt.plot(torch.tensor(data['caTrace'],dtype=torch.float)[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
#             c=(1-i/50,.6,i/50),
#             linewidth=.3)    
#     plt.xlim([0,2000])
# plt.title(f'Original: {params["AE_dropout_rate"]}')

# max_val=torch.max(reconstruction)
# plt.subplot(122)
# for i in range(cells2plot):
#     plt.plot(reconstruction[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
#             c=(1-i/50,.6,i/50),
#             linewidth=.3)
#     plt.xlim([0,2000])
# plt.title(f'Reconstruction\nDropout rate: {params["AE_dropout_rate"]}')
#plt.plot(datapoints[:,0]-reconstruction[:,0])