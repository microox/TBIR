import argparse
from dataset import *
from torch.utils.data import DataLoader
import pickle

parser = argparse.ArgumentParser(description='Cross-modal Retrieval Demonstration')
parser.add_argument('--task', type=int, choices=[1, 2], default=1,
                    help='determines which task should be performed (choices: 1,2, default: 1)')
parser.add_argument('--name', default='BasicModel', type=str,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='id of gpu to use')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='if set, disables CUDA training')
parser.add_argument('--img_feature_path', type=str, metavar='P', default='data/image_features.csv',
                    help='specifies, where csv file with image feature (default: data/image_features.csv) ')
parser.add_argument('--caption_path', type=str, metavar='P', default='data/results_20130124.token',
                    help='specifies exact filepath where captions are stored (i.e. results_20130124.token)\n' +
                         'only used, if caption csv-files (train_captions.csv, val_captions.csv, test_captions.csv)' +
                         'do not exist yet')
parser.add_argument('--path_bow_vectorizer', type=str, metavar='P', default='data/bow_vectorizer',
                    help='specifies the path to the serialized bow_vectorizer')


def main():
    # enable GPU learning
    global args
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)

    print("starting with following args:\n%s" % vars(args))  # TODO: implement logger if time left

    with open(args.path_bow_vectorizer,'rb') as file:
        bow_vectorizer = pickle.load(file)

    test_set = FLICKR30K(mode='test', vectorizer=bow_vectorizer)
    test_loader = DataLoader(test_set, batch_size=4000, shuffle=False)
