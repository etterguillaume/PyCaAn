#%% Imports
import yaml
import os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.metrics import extract_total_distance_travelled, extract_firing_properties
from argparse import ArgumentParser

plt.ioff() # Prevent plotting to save memory

#%%
def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def plot_summary_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    # Plot summary
    cmap = matplotlib.cm.get_cmap('viridis')
    numFrames, numNeurons = data['rawData'].shape
    total_distance_travelled = extract_total_distance_travelled(data['position'])

    fig=plt.figure(figsize=(8.27,11.7),) # A4 format
    fig.suptitle(data['path'])
    plt.subplot(4,3,1)
    plt.title('Correlation projection')
    plt.imshow(data['corrProj'])
    plt.axis('off')
    #plt.axis('square')

    plt.subplot(4,3,2)
    plt.title(f'SFPs: {numNeurons} neurons')
    plt.imshow(np.max(data['SFPs'],axis=0))
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title(f'Binarized activity')
    plt.imshow(data['binaryData'].T, cmap='GnBu', aspect='auto', interpolation='none')
    #plt.xlim(0,60*params['sampling_frequency'])
    #plt.ylim(0,50)

    # plt.subplot(4,3,5)
    # plt.title(f'Raw calcium')
    # for i in range(50):
    #     color = cmap(i/(50))
    #     plt.plot(data['caTime'],data['neuralData'][:,i]/10+i,
    #             c=color,
    #             linewidth=1,
    #             rasterized=True)
    # plt.xlim(0,60)
    # plt.axis('off')

    plt.subplot(4,3,7)
    plt.title(f'Distance travelled: {total_distance_travelled.round(2)} cm')
    plt.plot(data['position'][:,0],
            data['position'][:,1],
            rasterized=True)
    plt.axis('equal')

    plt.subplot(4,3,8)
    plt.title(f'Max speed: {np.max(data["velocity"]).round(2)}')
    plt.hist(data['velocity'],
             range=(0,75),
             bins=20,
             density=True,
             rasterized=True)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(working_directory,'summary.pdf'))
    fig.clf()
    plt.close()

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    plot_summary_session(data, params)

# %% Save figure
