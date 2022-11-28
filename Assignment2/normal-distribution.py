from random import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score






m=10



#for graph initilize list
graph_data=[]

# to get data for multiple d ( d dimensions)  looping d ranging from 1 to 50 

for d in range(1,100):

    #mean for normal distribution
    mean = np.random.rand(d)

    # for getting covariance , take random A , then A x A transpose produce symmetric 
    # covariance matrix must be symmetric
    A = np.random.rand(d, d) 
    cov = np.dot(A, A.transpose())

    # take any favorite W  ( not our lover matrix )
    w=[1]*d

    # producing X by multivariate guassian distribution
    x = np.random.multivariate_normal(mean, cov, m)


    y=[]

    #y is label data getting from normal distribution
    for i in range(m):
        y.append( np.random.normal(np.dot(w,x[i]),1,1) )

    # test and train data splitting
    x_train=x[:5]
    y_train=y[:5]
    x_test=x[5:]
    y_test=y[5:]

    # regression model training 
    reg = LinearRegression().fit(x_train,y_train)

    # variance calculating 
    variance= explained_variance_score(y_test,reg.predict(x_test))

    #storing data into graph data
    graph_data.append(variance)



#graph plotting 
x = [i for i in range(1, 100)]
y = graph_data

plt.plot(x,y)
plt.xlabel('d-dimensions')
plt.ylabel('Explained variance')

plt.title('Explained variance on  test set vs d')

plt.show()
