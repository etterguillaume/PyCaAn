import numpy as np
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
import os
from plotly import graph_objects as go

def plot_summary(data, params, name, save_fig=True, extension='.png', plot=False):
    plt.figure(figsize=(3,3))
    plt.subplot(221)
    plt.plot(data['position'][0],data['position'][1]) # Plot trajectory
    plt.title('Location (cm)')
    plt.xlim([0,params['open_field_width']])
    plt.ylim([0,params['open_field_width']])
    plt.xticks([0,25,50])
    plt.yticks([0,25,50])

    plt.subplot(222)
    plt.imshow(data['corrProj'],aspect='auto',cmap='bone')
    #plt.imshow(np.max(data['SFPs'],axis=0),aspect='auto',cmap='bone')
    plt.axis('off')

    plt.subplot(212)
    max_val=np.max(data['caTrace'])
    for i in range(params['max_ca_traces_to_plot']):
        plt.plot(data['caTime'][0,:],
                 data['caTrace'][i,:]*params['plot_gain']+max_val*i/params['plot_gain'],
                 c=(1-i/50,.6,i/50),
                 linewidth=.3)

    if save_fig:
        if not os.path.exists(os.path.join(params['path_to_results'],'summary_figures')):
            os.mkdir(os.path.join(params['path_to_results'],'summary_figures'))
        
        save_path = os.path.join(params['path_to_results'],'summary_figures',f'{name}{extension}')
        plt.savefig(save_path)

    if plot==False:
        plt.close()

def plot_losses(train_loss, test_loss, loss_label='Loss', title='Model training'):
    plt.figure()
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel(loss_label)
    plt.title(title)
    plt.legend()
    
def interactive_plot_manifold3D(x,y,z,color):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        name='Embedded neuronal activity',
        mode = 'markers',
        marker=dict(
            size=4,
            color=color,
            colorscale='Viridis',
            showscale=True,
            opacity=0.8)))

    fig.update_layout(
        title=go.layout.Title(
            text='Embedded neuronal activity',
            xref="paper",
            x=0),
            xaxis_title="X",
            yaxis_title="Y",
            font=dict(
            family="Helvetica",
            size=18))

    fig.show()

#%% Plot reconstruction 
    plt.figure(figsize=(3,3))
    plt.subplot(221)
    x=total_inputs.flatten()
    y=total_reconstructions.flatten()
    a, b = np.polyfit(x, y, 1)
    plt.scatter(x,y)
    plt.plot(x, a*x+b,'r--')
    plt.xlabel('Original input value')
    plt.ylabel('Reconstructed input value')
    plt.title(f'Reconstruction\nR2={reconstruction_stats[0].round(4)}, p={reconstruction_stats[1].round(4)}')

    plt.subplot(222)
    plt.hist(total_losses)
    plt.xlabel('Loss')
    plt.ylabel('Number')
    plt.title(f'Embedder\nmean loss:{avg_loss.round(4)}')

#%% Plot decoding 
    plt.subplot(223)
    x=total_positions[:,0].flatten()
    y=total_predictions[:,0].flatten()
    a, b = np.polyfit(x, y, 1)
    plt.scatter(x,y)
    plt.plot(x, a*x+b,'r--')
    plt.xlabel('Actual position')
    plt.ylabel('Decoded position')
    plt.title(f'Decoder\nR2={decoder_stats[0].round(4)}, p={decoder_stats[1].round(4)}')

    plt.subplot(224)
    plt.hist(total_pred_losses)
    plt.xlabel('Loss')
    plt.ylabel('Number')
    plt.title(f'Decoder\nmean loss:{avg_pred_loss.round(4)}')

    plt.tight_layout()


#%% Plot reconstruction examples

    with torch.no_grad():
        reconstruction, embedding = model(torch.tensor(data['caTrace'],dtype=torch.float))

    #%%
    #lower_bound_loss = criterion(torch.tensor(data['caTrace'],dtype=torch.float),torch.tensor(data['caTrace'],dtype=torch.float))
    #upper_bound_loss = criterion(torch.tensor(data['caTrace'],dtype=torch.float),torch.tensor(~data['caTrace'],dtype=torch.float))
    error = criterion(reconstruction, torch.tensor(data['caTrace'],dtype=torch.float))

    #%%
    plt.subplot(121)
    max_val=torch.max(torch.tensor(data['caTrace'],dtype=torch.float))
    cells2plot = 10
    for i in range(cells2plot):
        plt.plot(torch.tensor(data['caTrace'],dtype=torch.float)[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
                c=(1-i/50,.6,i/50),
                linewidth=.3)    
        plt.xlim([0,2000])
    plt.title(f'Original: {params["AE_dropout_rate"]}')

    max_val=torch.max(reconstruction)
    plt.subplot(122)
    for i in range(cells2plot):
        plt.plot(reconstruction[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
                c=(1-i/50,.6,i/50),
                linewidth=.3)
        plt.xlim([0,2000])
    plt.title(f'Reconstruction\nDropout rate: {params["AE_dropout_rate"]}')
    plt.plot(datapoints[:,0]-reconstruction[:,0])