from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import numpy
from scipy import ndimage
from six.moves import urllib
import tensorflow as tf
import cv2
import numpy as np

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000

path = "/home/dlagroup5/Assignment_3/Fingerprint/Phase2/Lum"
def maybe_download(filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images, norm_shift=False, norm_scale=True):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = data / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        cv2.imwrite("img.png",data[0,:,:,0])
        data = numpy.reshape(data, [num_images, -1])
    # print(data.shape)
    return data

def get_data():
    imgs_train = np.ndarray((3500,352*352),dtype="float32")
    imgs_test  = np.ndarray((1666,352*352),dtype="float32")
    ct = 0
    for fil in os.listdir(path):
        img = cv2.imread(os.path.join(path,fil),0)
        img = cv2.resize(img,(352,352))
        cv2.imwrite("./im_out/im{}.png".format(ct),img)
        img2 = np.reshape(img,(352*352))
        img2 = img2.astype('float32')

        if(ct<3500):
            imgs_train[ct,:] = img2
        else:
            imgs_test[ct-3500,:] = img2
        ct += 1
    imgs_test = imgs_test / PIXEL_DEPTH
    imgs_train = imgs_train / PIXEL_DEPTH
    # print(np.unique(imgs_test))
    return imgs_train, imgs_test


def prepare_MNIST_data(use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False):
    # train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    # test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    #
    # train_data = extract_data(train_data_filename, 60000, use_norm_shift, use_norm_scale)
    # test_data = extract_data(test_data_filename, 10000, use_norm_shift, use_norm_scale)
    train_data, test_data = get_data()
    return train_data, test_data

