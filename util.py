import numpy as np
import pandas as pd
import os


def get_train_val_test_split(train_path='data/train.txt',
                             val_path='data/val.txt',
                             test_path='data/test.txt',
                             verbose=True):
    """
    returns list with names of train, validation and test set
    """
    with open(train_path) as f:
        if verbose:
            print("parsing image names for training set...")
        train = f.readlines()
    with open(val_path) as f:
        if verbose:
            print("parsing image names for validation set...")
        val = f.readlines()
    with open(test_path) as f:
        if verbose:
            print("parsing image names for test set...")
        test = f.readlines()
    if verbose:
        print("converting image names to integers...")
    train = [int(string) for string in train]
    val = [int(string) for string in val]
    test = [int(string) for string in test]
    return train, val, test


def split_imgs(img_path, training_set, val_set, test_set, save_path='data', nr_of_feature_vectors=2048):
    column_names = ['img_name'] + ['feature_%s' % i for i in range(nr_of_feature_vectors)]
    df = pd.read_csv(img_path, sep=' ', header=None, names=column_names)
    df['img_name'] = df['img_name'].apply(lambda x: x.replace(".jpg", "")).astype(int)
    df = df.set_index('img_name')

    df_training = df[df.index.isin(training_set)]
    df_training.to_csv(os.path.join(save_path, 'train_img_features.csv'))
    df_val = df[df.index.isin(val_set)]
    df_val.to_csv(os.path.join(save_path, 'val_img_features.csv'))
    df_test = df[df.index.isin(test_set)]
    df_test.to_csv(os.path.join(save_path, 'test_img_features.csv'))

    return df_training, df_val, df_test


def split_captions(caption_path, training_set, val_set, test_set,
                   save_as_csv=True, save_path='data', verbose=True):
    """
    create data frame with index = img_name and cells = respective captions

    example:

    the following 5 lines from results_20130124.token (with full sentences shortened to sentence_i):
        1000092795.jpg#0	sentence_1
        1000092795.jpg#1	sentence_2
        1000092795.jpg#2	sentence_3
        1000092795.jpg#3	sentence_4
        1000092795.jpg#4	sentence_5

    would be converted to following row:
        {1000092795: [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5]}

    ATTENTION: this requires that the file located in caption path preserves is original structure!

    :param caption_path: path to caption file
    :param training_set: list with image names belonging to training set (e.g. 1000092795)
    :param val_set: list with image names belonging to validation set
    :param test_set: list with image names belonging to test set
    :param save_as_csv: if True, save captions as csv in save_path (easier to process later on)
    :param save_path: path, where to save csv files for captions; only used, when save_as_csv = True
    :param verbose: if true, print a message at each step
    :return: df_training, df_val, df_test (containing training, validation and test set
    """
    with open(caption_path) as f:
        if verbose:
            print("parsing image captions...")
        lines = f.readlines()

    # 0000000.jpg#0     This is an exemplary caption!
    #            will get converted to
    # ['000000', '0', 'This is an exemplary caption!']
    img_to_caption_as_list = [line.replace('.jpg#', '\t').replace('\n', '').split('\t') for line in lines]

    # group all captions belonging to the same image
    composite_list = np.array([img_to_caption_as_list[x:x + 5] for x in range(0, len(img_to_caption_as_list), 5)])

    # create dictionary with { img: [list of captions] }
    training = {}
    val = {}
    test = {}
    keys = np.unique(composite_list[:, :, 0]).astype('int')
    for i, key in enumerate(keys):
        if key in training_set:
            training[int(key)] = composite_list[:, :, 2][i]
        elif key in val_set:
            val[int(key)] = composite_list[:, :, 2][i]
        elif key in test_set:
            test[int(key)] = composite_list[:, :, 2][i]
        else:
            raise Exception("%s contains an image that does neither belong to train-, val- or test-set" % caption_path)

    # determine dataset
    if verbose:
        print("saving captions in csv-format...")

    df_training = pd.DataFrame(training).T
    df_val = pd.DataFrame(val).T
    df_test = pd.DataFrame(test).T

    if save_as_csv:
        df_training.to_csv(os.path.join(save_path, 'train_captions.csv'))
        df_val.to_csv(os.path.join(save_path, 'val_captions.csv'))
        df_test.to_csv(os.path.join(save_path, 'test_captions.csv'))

    return df_training, df_val, df_test


def img_to_caption_old(caption_path, training_list, verbose=True):
    with open(caption_path) as f:
        if verbose:
            print("parsing image captions...")
        lines = f.readlines()

    # 0000000.jpg#0     This is an exemplary caption!
    #            will get converted to
    # ['000000', '0', 'This is an exemplary caption!']
    # img_to_caption_as_list = [line.replace('.jpg#', '\t').split('\t') for line in lines if line in training_list]
    img_to_caption_as_list = [line.replace('.jpg#', '\t').replace('\n', '').split('\t') for line in lines]

    # only include images that are part of training_list
    if verbose:
        print("only include training set...")
    img_to_caption_as_list = [[int(l[0]), int(l[1]), str(l[2])]
                              for l in img_to_caption_as_list if int(l[0]) in training_list]

    # use dictionary for sampling captions:
    # i -> [img_name, caption_id, caption]
    if verbose:
        print("creating dictionary...")
    dictionary = {i: itcal for i, itcal in zip(range(len(img_to_caption_as_list)), img_to_caption_as_list)}
    return dictionary
