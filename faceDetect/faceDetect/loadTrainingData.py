import cv
from PIL import Image
import logistic
import csv
import numpy as np

import defaults


def getTrainingRes():
    """
    given a jpg image, vectorize the grayscale pixels to 
    a (width * height, 1) np array
    it is used to preprocess the data and transform it to feature space
    """
    def vectorize(filename):
        size = defaults.WIDTH, defaults.HEIGHT # (width, height)
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
    with open(defaults.smile_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open(defaults.neutral_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec

    # create a list for filenames of disgust pictures
    disgustfiles = []
    with open(defaults.disgust_csv, 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            disgustfiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(smilefiles) + len(neutralfiles) + len(disgustfiles), defaults.dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load neutral data
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(defaults.neutral_imgs + filename)
        labels.append(0)

    # load smile data    
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(defaults.smile_imgs + filename)
        labels.append(1)

    # load disgust data
    offset = idx + 1
    for idx, filename in enumerate(disgustfiles):
        phi[idx + offset] = vectorize(defaults.disgust_imgs + filename)
        labels.append(2)

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(defaults.dim)
    lr.train(phi, labels)
    return lr
