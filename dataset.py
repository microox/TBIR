import torch
from torch.utils.data import Dataset

class FLICKR30K(Dataset):
    def __init__(self, mode):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode

    def __getitem__(self, index):
        # TODO: Overwrite the __getitem__ method of Dataset to return the data sample at the given index
        return

    def __len__(self):
        # TODO: Overwrite the __len__ method of Dataset to return the size of the dataset
        return

    def get_dimensions(self):
        # TODO: Implement this method to return the dimensions of the image and text features
        return