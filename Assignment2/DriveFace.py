
import imp
from operator import truediv
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


#randomising data  , spliting data into training and testing 
id_list = list(range(len(X)))
random.shuffle(id_list)
id_train = id_list[0:303]
print(len(id_train))
id_test = id_list[303:]
print(len(id_test))


X_train = X[id_train]
print(X_train.shape)
Y_train = Y[id_train]
print(Y_train.shape)

X_test = X[id_test]
print(X_test.shape)
Y_test = Y[id_test]
print(Y_test.shape)


#passing data into linear regression trainig model . 

reg = LinearRegression().fit(X_train,Y_train)

#reg = LinearRegression().fit(X,Y)


print("PHI(x)=x")
print("Coefficioent is :",reg.coef_)

variance=explained_variance_score(Y_test, reg.predict(X_test) )

print("Explained variance is :",variance)



X1=[[1]*6400] * 303
X2=X_raw
X3=np.square(X2)
X = np.concatenate((X1, X2,X3), axis=1)

X_train = X[id_train]
print(X_train.shape)
Y_train = Y[id_train]
print(Y_train.shape)

X_test = X[id_test]
print(X_test.shape)
Y_test = Y[id_test]
print(Y_test.shape)

#passing data into linear regression trainig model . 

reg = LinearRegression().fit(X_train,Y_train)

#reg = LinearRegression().fit(X,Y)
print("PHI(x)=[1 x x^2]")
print("Coefficioent is :",reg.coef_)

variance=explained_variance_score(Y_test, reg.predict(X_test) )

print("Explained variance is :",variance)


