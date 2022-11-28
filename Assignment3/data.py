import os

import statistics
import numpy as np
from sklearn import preprocessing
x_raw=[]
y_raw=[]

class1=[]
class2=[]
class3=[]

for filename in os.listdir(os.getcwd()+'/Data'):
    input= open('Data/'+filename,'r')
    for line in input:
        l=line.split(';')
        if int(l[6])==0:
            class1.append([int(l[3]),int(l[4])])
        if int(l[6])==1:
            class2.append([int(l[3]),int(l[4])])
        if int(l[6])==2:
            class3.append([int(l[3]),int(l[4])])

class1=preprocessing.normalize(class1)
class2=preprocessing.normalize(class2)
class3=preprocessing.normalize(class3)

c1=np.array(class1)  
c2=np.array(class2)
c3=np.array(class3)    

#print(c1)
#print(arr[:, 0])
#class 1 
c1_1=statistics.mean(c1[:,0])
c1_2=statistics.mean(c1[:,1])

#class 2
c2_1=statistics.mean(c2[:,0])
c2_2=statistics.mean(c2[:,1])

#class 3
c3_1=statistics.mean(c3[:,0])
c3_2=statistics.mean(c3[:,1])


s_c1=np.cov(np.array([c1[0],c1[1]]) )
s_c2=np.cov(np.array([c2[0],c2[1]]) )
s_c3=np.cov(np.array([c3[0],c3[1]]))

print(s_c1)
print(s_c2)
print(s_c3)


