from torch.utils.data import Dataset

class generateDataset(Dataset):
    def __init__(self, neural_data, labels):
        self.neural_data = neural_data
        self.labels = labels

    def __len__(self):
        return self.neural_data.shape[0]

    def __getitem__(self, idx):
        data = self.neural_data[idx, :]
        label = self.labels[idx]

        return data, label