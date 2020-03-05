"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
import cv2 as cv

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import math
import imageio

import tensorflow as tf

#FLAGS = tf.app.flags.FLAGS

def transform(images):
  return np.array(images)/127.5 - 1.
def inverse_transform(images):
  return (images+1.)/2
def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  
  """

  filenames = os.listdir(dataset)
  data_dir = os.path.join(os.getcwd(), dataset)
  data = glob.glob(os.path.join(data_dir, "*.png"))
  data = data + glob.glob(os.path.join(data_dir, "*.jpg"))
  return data

def imread(path, is_grayscale=False):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """

  if is_grayscale:
    return scipy.misc.imread(path, flatten=True).astype(np.float32)
   # img1=cv.imread(path).astype(np.float32)
   # return cv.cvtColor(img1,cv.COLOR_BGR2YCrCb)
  else:
   # img1=cv.imread(path).astype(np.float32)
   # return cv.cvtColor(img1,cv.COLOR_BGR2YCrCb)

    return scipy.misc.imread(path).astype(np.float32)

    
def imsave(image, path):

  imsaved = (inverse_transform(image)).astype(np.float)
  return imageio.imwrite(path, imsaved)

def get_image(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return image/255
def get_lable(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return image/255.
def imsave_lable(image, path):
  return imageio.imwrite(path, image*255)


def white_balance(img, percent=0):
#    img=cv.cvtColor(img, cv.COLOR_RGB2BGR)
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv.split(img):
        cumhist = np.cumsum(cv.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv.LUT(channel, lut.astype('uint8')))
    img=cv.merge(out_channels)
    #cv.imshow('WB1', img)
    print(percent)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
#    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def adjust_gamma(image, gamma=0.7):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image.astype(np.uint8), table.astype(np.uint8))


# def white_balance(img1):
#   #cv.imshow('WB2', img1)
#   #cv.waitKey(0)
#   #cv.destroyAllWindows()

#   result = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
#   avg_a = np.average(result[:, :, 1])
#   avg_b = np.average(result[:, :, 2])
#   result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
#   result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
#   img2 = cv.cvtColor(result, cv.COLOR_LAB2BGR)
#   print('WB1- passed')

#   return img2
