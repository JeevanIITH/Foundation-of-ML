import numpy as np
import scipy.io
import pandas as pd
import random
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
random.seed(0)

#Upload DrivFace.mat file in session storage before running the following code
img_data = scipy.io.loadmat('DrivFace/DrivFace.mat')
drivFaceData = img_data['drivFaceD'][0]
X_raw = img_data['drivFaceD'][0][0][0]
#print(X_raw.shape)

#Upload drivPoints.txt file in session storage before running the following code
label_data = pd.read_csv("DrivFace/drivPoints.txt")
#print(label_data.head(5))



Y = np.vstack((label_data['xF'], label_data['yF'], label_data['wF'], label_data['hF'])).T 

x_data=preprocessing.normalize(X_raw)
y_data=preprocessing.normalize(Y)

random_state = 12883823
rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=random_state)

hyper_parameters=[0.01,0.1,1,10,100]
print("3 fold 5 repeats cross validation \n")
for i in range(5):
    reg=linear_model.Ridge(alpha=hyper_parameters[i])
    score=cross_val_score(reg,x_data,y_data,cv=rkf)
    print(" mean score for alpha = "+str(hyper_parameters[i])+" is " +str(score.mean()))
    print("\n")


print("\n")
print("Best alpha is 0.01")
print("\n")
final=linear_model.Ridge(alpha=0.01,fit_intercept=False)
id_list = list(range(len(X_raw)))
split_ratio = 0.5
split_index = int(len(X_raw)*0.5)
random.shuffle(id_list)
id_train = id_list[0:split_index]
id_test = id_list[split_index:]

#Prepare feature data 
X_train1 = x_data[id_train]
X_test1 = x_data[id_test]
#print(X_train1.shape)
#print(X_test1.shape)

Y_train = y_data[id_train]
Y_test = y_data[id_test]
#print(Y_train.shape)
#print(Y_test.shape)

final.fit(X_train1,Y_train)
Y_pred1=final.predict(X_test1)

print("\n")
print("variance of rigid regression ")
print(f"Explained variance for xF: {explained_variance_score(Y_test[:,0], Y_pred1[:,0])}")
print(f"Explained variance for yF: {explained_variance_score(Y_test[:,1], Y_pred1[:,1])}")
print(f"Explained variance for wF: {explained_variance_score(Y_test[:,2], Y_pred1[:,2])}")
print(f"Explained variance for hF: {explained_variance_score(Y_test[:,3], Y_pred1[:,3])}")
print(f"Explained variance for all: {explained_variance_score(Y_test, Y_pred1)}")

print("\n")

print("SVR \n")
for i in range(5):
    model=svm.LinearSVR(C=hyper_parameters[i])
    score=cross_val_score(reg,x_data,y_data,cv=rkf)
    #print("score for C = "+str(hyper_parameters[i])+" is " +str(score))
    print("Mean score for C = "+str(hyper_parameters[i])+" is :"+ str(score.mean()))
    print("\n")


print("No best c for SVR ")
final=svm.LinearSVR(C=1)
final.fit(X_train1,Y_train[:,0])
Y_pred1=final.predict(X_test1)

print("\n")
print("variance of SVR ")
""" print(f"Explained variance for xF: {explained_variance_score(Y_test[:,0], Y_pred1[:,0])}")
print(f"Explained variance for yF: {explained_variance_score(Y_test[:,1], Y_pred1[:,1])}")
print(f"Explained variance for wF: {explained_variance_score(Y_test[:,2], Y_pred1[:,2])}")
print(f"Explained variance for hF: {explained_variance_score(Y_test[:,3], Y_pred1[:,3])}") """
print(f"Explained variance for all: {explained_variance_score(Y_test[:,0], Y_pred1)}")

print("\n")

print("Variance of linear regression WITH normalization \n ")
model1 = LinearRegression(fit_intercept=False).fit(X_train1, Y_train)
Y_pred1 = model1.predict(X_test1)
print(f"Explained variance for xF: {explained_variance_score(Y_test[:,0], Y_pred1[:,0])}")
print(f"Explained variance for yF: {explained_variance_score(Y_test[:,1], Y_pred1[:,1])}")
print(f"Explained variance for wF: {explained_variance_score(Y_test[:,2], Y_pred1[:,2])}")
print(f"Explained variance for hF: {explained_variance_score(Y_test[:,3], Y_pred1[:,3])}")
print(f"Explained variance for all: {explained_variance_score(Y_test, Y_pred1)}")

print("\n")
#Prepare feature data 
X_train1 = X_raw[id_train]
X_test1 = X_raw[id_test]


Y_train = Y[id_train]
Y_test = Y[id_test]



print("Variance of linear regression WITHOUT normalization \n ")
model1 = LinearRegression(fit_intercept=False).fit(X_train1, Y_train)
Y_pred1 = model1.predict(X_test1)
print(f"Explained variance for xF: {explained_variance_score(Y_test[:,0], Y_pred1[:,0])}")
print(f"Explained variance for yF: {explained_variance_score(Y_test[:,1], Y_pred1[:,1])}")
print(f"Explained variance for wF: {explained_variance_score(Y_test[:,2], Y_pred1[:,2])}")
print(f"Explained variance for hF: {explained_variance_score(Y_test[:,3], Y_pred1[:,3])}")
print(f"Explained variance for all: {explained_variance_score(Y_test, Y_pred1)}")





