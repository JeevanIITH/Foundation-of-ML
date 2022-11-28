import numpy as np
from  sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
import numpy as np
import scipy.io
import pandas as pd
import random

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
random.seed(0)


#parsing data from .mat file
img_data = scipy.io.loadmat('DrivFace/DrivFace.mat')
drivFaceData = img_data['drivFaceD'][0]

#X is input data
X_raw = img_data['drivFaceD'][0][0][0]
label_data = pd.read_csv("drivPoints.txt")
X=X_raw

#present using Y (label) as xF
Y=label_data['xF']

X=preprocessing.normalize(X)

random_state = 12883823
rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))

