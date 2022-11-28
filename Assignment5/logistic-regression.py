from sklearn import linear_model

from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random


xls = pd.ExcelFile('LSVT_voice_rehabilitation/LSVT_voice_rehabilitation.xlsx')  # pd.read_excel("LSVT_voice_rehabilitation.xlsx") will only read the 1st sheet
df1 = pd.read_excel(xls, 'Data')
df2 = pd.read_excel(xls, 'Binary response')

Y=np.array(df2['Binary class 1=acceptable, 2=unacceptable'])
#print(Y.shape)
Y=np.reshape(Y,(-1,1))
#print(Y.shape)
X=np.array(df1)
#print(X.shape)

x=preprocessing.normalize(X)
y=Y


y=np.array(y)
y=y.ravel()
#print(y.shape)
#print(y)
random_state = 12883823
rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=random_state)

hyper_parameters=[0.01,0.1,1,10,100]

print("\n 3 fold 5 repeats cross validation \n")
for i in range(5):
    model = linear_model.LogisticRegression(penalty='l2',C=hyper_parameters[i])
    score=cross_val_score(model,x,y,cv=rkf)
    print(" mean score for C = "+str(hyper_parameters[i])+" is " +str(score.mean()))
    print("\n")


print("All c except c=100 giving same mean score , so best is c = 1 (any one ) \n ")

model = linear_model.LogisticRegression(penalty='l2',C=1)

id_list = list(range(len(x)))
split_ratio = 0.5
split_index = int(len(x)*0.5)
random.shuffle(id_list)
id_train = id_list[0:split_index]
id_test = id_list[split_index:]

#Prepare feature data 
X_train1 = x[id_train]
X_test1 = x[id_test]
#print(X_train1.shape)
#print(X_test1.shape)

Y_train = y[id_train]
Y_test = y[id_test]
#print(Y_train.shape)
#print(Y_test.shape)


model.fit(X_train1,Y_train)
y_pred=model.predict(X_test1)

s=accuracy_score(Y_test,y_pred)

print("\n Accuracy of model logistic regression is : ", s)

print("\n 3 fold 5 repeats cross validation \n")
for i in range(5):
    model = svm.LinearSVC(loss='hinge',C=hyper_parameters[i])
    score=cross_val_score(model,x,y,cv=rkf)
    print(" mean score for C = "+str(hyper_parameters[i])+" is " +str(score.mean()))
    print("\n")

model=svm.LinearSVC(loss='hinge',C=1)
model.fit(X_train1,Y_train)
y_pred=model.predict(X_test1)

s=accuracy_score(Y_test,y_pred)

print("\n Accuracy of model logistic regression is : ", s)
print("\n")