import numpy as np
import pandas as pd
import cv2
import os

def read_images_grey(dir, dim = 128):
    # Read in images as matrix
    # Shrink 424x424 --> dim x dim
    images = []
    image_files = sorted(os.listdir(dir))
    image_files = [dir + '/' + f for f in image_files]
    for imgf in image_files:
        # Loop through images in dir
        img = cv2.imread(imgf, 0)
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_CUBIC)
        length = np.prod(img.shape)
        img = np.reshape(img, length)
        images.append(img)
    # Save images as matrix
    images = np.vstack(images)
    return images

def get_image_names(dir):
    inames = sorted(os.listdir(dir))
    inames = [int(f.strip('.jpg')) for f in inames]
    return np.asarray(inames)

def force_bounds(a):
    for x in np.nditer(a, op_flags = ['readwrite']):
        if x[...] > 1:
            x[...] = 1
        if x[...] < 0:
            x[...] = 0
    return a
