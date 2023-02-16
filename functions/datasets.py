import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class generateDataset(Dataset):
    def __init__(self, data, params):
        self.neural_data = torch.tensor(data['procData'][:,0:params['input_neurons']],dtype=torch.float)
        self.position = torch.tensor(data['position'],dtype=torch.float)
        self.velocity = torch.tensor(data['velocity'],dtype=torch.float)

        #TODO implement other variables
        #TODO error checking

        # # Split into overlapping data chunks
        # numChunks = len(neural_data)-params['data_block_size']+1 # Compute number of chunks
        # self.neural_data = torch.zeros((numChunks, params['input_neurons'],params['data_block_size']))
        # self.position = torch.zeros((numChunks,2,params['data_block_size']))
        # self.velocity = torch.zeros((numChunks,1,params['data_block_size']))
        # for chunk in range(numChunks):
        #     self.neural_data[chunk,:,:] = torch.transpose(neural_data[chunk:chunk+params['data_block_size'],:],0,1)
        #     self.position[chunk,:,:] = torch.transpose(position[chunk:chunk+params['data_block_size'],:],0,1)
        #     self.velocity[chunk,0,:] = velocity[chunk:chunk+params['data_block_size']]

        # neural_data = torch.split(neural_data, params['data_block_size'])
        # position = torch.split(position, params['data_block_size'])
        # velocity = torch.split(velocity, params['data_block_size'])
        
        # #Remove last uneven block
        # if params['data_block_size'] > 1:
        #     neural_data = neural_data[:-1]
        #     position = position[:-1]
        #     velocity = velocity[:-1]

        # # Convert into tensor
        # numChunks = len(neural_data)
        # self.neural_data = torch.zeros((numChunks, params['input_neurons'],params['data_block_size']))
        # self.position = torch.zeros((numChunks,2,params['data_block_size']))
        # self.velocity = torch.zeros((numChunks,1,params['data_block_size']))

        # # Populate tensor
        # for chunk in range(numChunks):
        #     self.neural_data[chunk,:,:] = torch.transpose(neural_data[chunk],0,1)
        #     self.position[chunk,:,:] = torch.transpose(position[chunk],0,1)
        #     self.velocity[chunk,0,:] = velocity[chunk]

    def __len__(self):
        return len(self.neural_data)

    def __getitem__(self, idx):
        data = self.neural_data[idx]
        position = self.position[idx]
        velocity = self.velocity[idx]

        return data, position, velocity

def split_to_loaders(dataset, params):
    # Splits dataset into train/test portions, and constructs pytorch dataloader
    dataset_size=len(dataset)
    train_set_size = int(dataset_size * params['train_test_ratio'])
    test_set_size = dataset_size - train_set_size
    train_set, test_set = random_split(dataset, [train_set_size, test_set_size]) # Random split

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False)

    return train_loader, test_loader