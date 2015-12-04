import cv
from PIL import Image
import logistic
import csv
import numpy as np

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector


def getTrainingRes():
    """
    given a jpg image, vectorize the grayscale pixels to 
    a (width * height, 1) np array
    it is used to preprocess the data and transform it to feature space
    """
    def vectorize(filename):
        size = WIDTH, HEIGHT # (width, height)
        im = Image.open(filename) 
        resized_im = im.resize(size, Image.ANTIALIAS) # resize image
        im_grey = resized_im.convert('L') # convert the image to *greyscale*
        im_array = np.array(im_grey) # convert to np array
        oned_array = im_array.reshape(1, size[0] * size[1])
        return oned_array
    """
    load training data
    """
    # create a list for filenames of smiles pictures
    smilefiles = []
    with open('smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open('neutral.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load smile data
    PATH = "../data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)

    # load neutral data    
    PATH = "../data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    return lr
