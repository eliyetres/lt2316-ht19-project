from torch.utils import data

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        X = self.data[index]
        y = self.labels[index]

        return X, y