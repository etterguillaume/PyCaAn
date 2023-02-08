import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class generateDataset(Dataset):
    def __init__(self, data, params):
        self.neural_data = torch.tensor(data['caTrace'][:,0:params['input_neurons']],dtype=torch.float)
        self.position = torch.tensor(data['position'],dtype=torch.float)
        self.velocity = torch.tensor(data['velocity'],dtype=torch.float)
        #TODO implement other variables
        #TODO error checking

        if params['data_block_size'] > 1: 
            # Split into data chunks
            self.neural_data = torch.split(self.neural_data, params['data_block_size'])
            self.position = torch.split(self.position, params['data_block_size'])
            self.velocity = torch.split(self.velocity, params['data_block_size'])
            # Remove last uneven block
            self.neural_data = self.neural_data[:-1]
            self.position = self.position[:-1]
            self.velocity = self.velocity[:-1]

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
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=True)

    return train_loader, test_loader