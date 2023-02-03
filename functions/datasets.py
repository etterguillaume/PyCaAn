import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class generateDataset(Dataset):
    def __init__(self, data, params, dataset_z=None, position_z=None, velocity_z=None):
        self.neural_data = torch.tensor(data['caTrace'],dtype=torch.float)
        self.position = torch.tensor(data['position'],dtype=torch.float)
        self.velocity = torch.tensor(data['velocity'],dtype=torch.float)
        
        self.dataset_z = dataset_z
        self.position_z = position_z
        self.velocity_z = velocity_z
        #TODO implement other variables
        #TODO error checking

    def __len__(self):
        return self.neural_data.shape[0]

    def __getitem__(self, idx):
        data = self.neural_data[idx, :]
        if self.dataset_z:
            data = (data-self.dataset_z[0])/self.dataset_z[1]
        
        position = self.position[idx]
        if self.position_z:
            position = (position-self.position_z[0])/self.position_z[1]

        velocity = self.velocity[idx]
        if self.velocity_z:
            velocity = (velocity-self.velocity_z[0])/self.velocity_z[1]

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