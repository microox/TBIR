import numpy as np


def get_train_val_test_split(train_path='data/train.txt',
                             val_path='data/val.txt',
                             test_path='data/test.txt',
                             verbose=True):
    '''
    returns list with names of train, validation and test set
    '''
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


def img_to_caption(caption_path, training_list, verbose=True):
    """
    create dictionary with image title : list of captions

    example:

    the following 5 lines from results_20130124.token (with full sentences shortened to sentence_i):
        1000092795.jpg#0	sentence_1
        1000092795.jpg#1	sentence_2
        1000092795.jpg#2	sentence_3
        1000092795.jpg#3	sentence_4
        1000092795.jpg#4	sentence_5

    would be converted to:
        {1000092795: [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5]}

    ATTENTION: this requires that the file located in caption path preserves is original structure!

    :param caption_path:
    :param training_list:
    :param verbose:
    :return:
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
    dict = {}
    keys = np.unique(composite_list[:, :, 0])
    for i, key in enumerate(keys):
        dict[int(key)] = composite_list[:, :, 2][i]

    return dict


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
    img_to_caption_as_list = [[int(l[0]), int(l[1]), str(l[2])] for l in img_to_caption_as_list if int(l[0]) in training_list]

    # use dictionary for sampling captions:
    # i -> [img_name, caption_id, caption]
    if verbose:
        print("creating dictionary...")
    dictionary = {i: itcal for i, itcal in zip(range(len(img_to_caption_as_list)), img_to_caption_as_list)}
    return dictionary
