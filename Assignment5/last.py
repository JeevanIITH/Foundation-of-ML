from turtle import shape
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# Loading the data and dropping the index axis
df = pd.read_csv('iris.data')
#df.shape
df['class'] = pd.factorize(df['class'])[0].astype(np.uint16)

Y = df['class']
df  = df.drop(['class'],axis=1)
X = df

X = np.array(X)
Y = np.array(Y)

X = X[:,:2]

C = 100  # SVM regularization parameter
models = (
    SVC(kernel="linear",gamma="auto", C=C),
    SVC(kernel="rbf", gamma=1, C=C),
    SVC(kernel="poly", degree=3, gamma="auto", C=C),
    SVC(kernel="poly", degree=2, gamma="auto"),
)

models = (clf.fit(X, Y) for clf in models)

titles = (
    "SVC with linear kernel",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
    "SVC with polynomial (degree 2) kernel",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0 = X[:,0]
X1 = X[:,1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel='sl',
        ylabel='sw',
    )
    
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()