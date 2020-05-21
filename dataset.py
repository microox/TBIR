import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import os
import numpy as np


class FLICKR30K(Dataset):
    """
    implements map-style dataset for FLICKR30K

    see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, mode, datapath='data', vectorizer=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        if mode in ['val', 'test']:
            assert vectorizer is not None, \
                "you need to create or load a BOW model and pass it to the dataset when in mode %s" % mode
        self.mode = mode

        # determine csv-file names according to mode, e.g. train_img_features.csv
        csv_file_images = "%s_img_features.csv" % mode
        csv_file_captions = "%s_captions.csv" % mode

        # load image data
        print("loading %s..." % csv_file_images)
        self.data_img = pd.read_csv(os.path.join(datapath, csv_file_images))

        # create BOW from all captions in training set
        print("loading %s..." % csv_file_captions)
        data_captions = pd.read_csv(os.path.join(datapath, csv_file_captions))  # TODO self.df_all_captions for debugging ?
        corpus = [item for sublist in data_captions.iloc[:, 1:].values.tolist() for item in sublist]
        self.vectorizer = vectorizer
        if mode in ['train']:
            self.data_bow = self._create_bow(corpus)  # TODO: create BOW outside of data_set
        else:
            self.data_bow = self._transform_bow(vectorizer, corpus)

    def __getitem__(self, index):
        # TODO 1 image has 5 captions !
        # TODO: if self.mode = val, test, not int(index / 5)
        image_features = np.array(self.data_img.iloc[int(index / 5), 1:])
        captions = self.data_bow.getrow(index).toarray().ravel()
        image_name = self.data_img.iloc[int(index / 5), 0]
        return image_features, captions, image_name

    def __len__(self):
        length = self.data_img.shape[0] * 5
        return length

    def get_dimensions(self):
        img_dim = self.data_img.shape[1] - 1
        cpt_dim = self.data_bow.shape[1]  # TODO
        return img_dim, cpt_dim

    def _transform_bow(self, vectorizer, corpus):
        """
        create a Bag-of-Words model given a corpus of captions
        :param corpus: all captions belonging to training set
        :return:
        """
        print("transforming captions from %s set to bow model..." % self.mode)
        assert self.mode in ['val', 'test'] and vectorizer is not None
        return vectorizer.transform(corpus)

    def _create_bow(self, corpus):
        """
        create a Bag-of-Words model given a corpus of captions
        :param corpus: all captions belonging to training set
        :return:
        """
        print("fitting vectorizer on captions from training set and transforming captions to bow model...")
        assert self.mode in ['train']
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=2048,
                                          min_df=0, max_df=0.33, smooth_idf=True, stop_words='english')
        return self.vectorizer.fit_transform(corpus)
