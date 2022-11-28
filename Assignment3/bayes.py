

import imp
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from dataextracter import x_raw,y_raw
from sklearn.svm import SVC

norm= preprocessing.normalize(x_raw)

X_train, X_test, y_train, y_test = train_test_split(norm, y_raw, test_size=0.5, random_state=0)


nb=GaussianNB()
nb.fit(X_train,y_train)

#X_test=X_test[:100]
#y_test=y_test[:100]

#def plot_decision_boundary(pred_func,y_t):
# Set min and max values and give it some padding
x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
print(x_min)
print(x_max)
y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
h = 0.01
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole gid
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
# Plot the decision boundary
#plot_decision_boundary(lambda x: nb.predict(X_test),y_test)
plt.title("navie bayes classifier")
plt.show()


nb=LogisticRegression()
nb.fit(X_train,y_train)

X_test=X_test[:100]
y_test=y_test[:100]

#def plot_decision_boundary(pred_func,y_t):
# Set min and max values and give it some padding
x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
print(x_min)
print(x_max)
y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
h = 0.01
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole gid
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
# Plot the decision boundary
#plot_decision_boundary(lambda x: nb.predict(X_test),y_test)
plt.title("logistic regression")
plt.show()