import torch
from torch.utils.data import Dataset
import pandas as pd


class FLICKR30K(Dataset):
    """
    implements map-style dataset for FLICKR30K

    see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, mode):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode

        # determine csv-file names, e.g. train_img.csv, train_cpt.csv
        csv_file_images = "%s_img.csv" % mode
        csv_file_captions = "%s_cpt.csv" % mode

        # load image data
        print("loading %s..." % csv_file_images)
        self.data_img = pd.read_csv(csv_file_images)

        # load captions and create BOW
        print("loading %s..." % csv_file_captions)
        data_cpt = pd.read_csv(csv_file_captions)
        self.data_bow = ...  # TODO: create BOW outside of data_set

    def __getitem__(self, index):
        # TODO how to represent captions? (list?)
        sample = {'image_features': image_features, 'capitons': captions}
        return sample

    def __len__(self):
        # TODO: Overwrite the __len__ method of Dataset to return the size of the dataset
        return

    def get_dimensions(self):
        # TODO: Implement this method to return the dimensions of the image and text features
        img_dim = ...
        txt_dim = ...
        return img_dim, txt_dim
