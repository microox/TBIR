import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import os


class FLICKR30K(Dataset):
    """
    implements map-style dataset for FLICKR30K

    see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, mode, datapath='data'):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode

        # determine csv-file names according to mode, e.g. train_img_features.csv
        csv_file_images = "%s_img_features.csv" % mode
        csv_file_captions = "%s_captions.csv" % mode

        # load image data
        print("loading %s..." % csv_file_images)
        self.data_img = pd.read_csv(os.path.join(datapath, csv_file_images))

        # create BOW from all captions in training set / TODO: also include val / test set to corpus
        print("loading %s..." % csv_file_captions)
        df_all_captions = pd.read_csv(os.path.join(datapath, csv_file_captions))  # TODO which input for vectorizer?
        corpus = [item for sublist in df_all_captions.iloc[:, 2:].values.tolist() for item in sublist]
        self.data_bow = self._create_bow(corpus)  # TODO: create BOW outside of data_set

    def __getitem__(self, index):
        # TODO how to represent captions? (list?)
        sample = {'image_features': image_features, 'capitons': captions}
        return sample

    def __len__(self):
        # TODO: Overwrite the __len__ method of Dataset to return the size of the dataset
        length = self.data_img.shape[0]
        return length

    def get_dimensions(self):
        # TODO: Implement this method to return the dimensions of the image and text features
        img_dim = self.data_img.shape
        cpt_dim = self.data_bow.shape
        return img_dim, cpt_dim

    def _create_bow(self, corpus):
        """
        create a Bag-of-Words model given a corpus of captions
        :param corpus: all captions belonging to training set
        :return:
        """
        # TODO: make function usable for test and validation set
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), max_features=1024,
                                     min_df=0, max_df=0.01, smooth_idf=True)
        return vectorizer.fit_transform(corpus)
