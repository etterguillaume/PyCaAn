import numpy as np
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

def plot_summary(data, params):
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
        plt.plot(data['caTime'][0,:],data['caTrace'][i,:]*params['plot_gain']+max_val*i/params['plot_gain'], c=(1-i/50,.6,i/50))