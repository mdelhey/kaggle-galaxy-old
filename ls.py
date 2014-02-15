import numpy as np
import pandas as pd
import cv2
import os
#import sklearn as sk
from sklearn import linear_model
from functions import *

f_out = 'Submissions/ls_128.csv'
trn_dir = 'Data/images_train'
tst_dir = 'Data/images_test'

# X = individual images, n = 61,578
# Y = response for each, p = 37

# Read in training data & solutions
Xtrn = read_images_grey(trn_dir, dim = 128)
Ytrn = np.loadtxt(open('Data/train_solutions.csv', 'rb'), delimiter = ',', skiprows = 1)

# Train a regression model
#model = sk.linear_model.Ridge(alpha = 1)
model = sk.linear_model.LinearRegression()
model.fit(Xtrn, Ytrn[::, 1:])

# Read in test data, predict
Xtst = read_images_grey(tst_dir)
Ytst = model.predict(Xtst)

# Force [0,1] bounds
Ytst = force_bounds(Ytst)

# Add columns of ID's
Ytst_names = get_image_names(tst_dir)
Ytst = np.c_[Ytst_names, Ytst]

# Save as csv
np.savetxt(f_out, Ytst, delimiter = ',')
