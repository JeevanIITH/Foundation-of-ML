
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn.metrics import accuracy_score






# Loading the data and dropping the index axis
df = pd.read_csv('kidney_disease.csv')
df  = df.drop(['id'],axis=1)


# separating data into different classes
real = ['sg','sc','pot','hemo','rc',]
integer = ['age','bp','al','su','bgr','bu','sod','pcv','wc',]
label = ['classification']
cat = list(set(df.columns) - set(real)-set(integer)-set(label))

# Removing parsing errors
df = df.replace('\t?',np.nan)
df = df.replace('\tyes','yes')
df = df.replace(' yes','yes')
df = df.replace('yes\t','yes')
df = df.replace('\tno','no')
df = df.replace('ckd\t','ckd')
df = df.replace('ckd',1)
df = df.replace('notckd',0)


# Filling the null values with mean you can also use other statistic like mode or median
for r in real:
    mean = np.array(df[r][~df[r].isna()]).astype('float').mean()
    mean=round(mean,0)
    print(mean)
    df[r].fillna(value=mean,inplace=True)
for i in integer:
    mean = np.array(df[i][~df[i].isna()]).astype('int').mean()
    mean=round(mean,0)
    df[i].fillna(value=int(mean),inplace=True)

df['rbc'].fillna(value=str('normal'),inplace=True)
df['pc'].fillna(value=str('normal'),inplace=True)


X = df.drop(label,axis=1)
Y =np.array( df[label] )
print(Y.shape)
Y=np.reshape(Y,(-1,))


# You need to convert the catagorical variables to binary u can use pd.get_dummies to do so 


xd=pd.get_dummies(X,columns=cat)

#partition into 3 parts
X_train, X_test, y_train, y_test = train_test_split(xd, Y, test_size=0.66, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1) # 0.25 x 0.8 = 0.2

X_train=np.array(X_train)
X_test=np.array(X_test)
X_val=np.array(X_val)

#print(X_train.shape)
#xd.to_csv('Output.csv')


def distane(x,z):
    d1=DistanceMetric.get_metric('minkowski')
    d2=DistanceMetric.get_metric('canberra')
    d3=DistanceMetric.get_metric('russellrao')


    xb=x[-20:-1]
    #print(xb.shape)
    zb=z[-20:-1]
    #np.array(list(zip(lat, lon)))
    xi=np.array([x[1],x[2],x[4],x[5],x[6],x[7],x[9],x[12],x[13]])
    zi=np.array([z[1],z[2],z[4],z[5],z[6],z[7],z[9],z[12],z[13]])

    xr=np.array([x[3],x[8],x[10],x[11],x[14]])
    zr=np.array([z[3],z[8],z[10],z[11],z[14]])


    da=d1.pairwise([xr,zr])[0][1]
    db=d2.pairwise([xi,zi])[0][1]
    dc=d3.pairwise([xb,zb])[0][1]
    #print(da.shape)
    return da+db+dc

kl=[1,3,5,7,9]
best_k=1
current_score=0
m_score=0
for k in kl:
    knn=KNeighborsClassifier(n_neighbors=k,metric=distane)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_val)
    current_score=accuracy_score(y_val,y_pred)
    print(current_score)
    if current_score>m_score:
        best_k=k
        m_score=current_score


print("Best K is ",best_k)

knn=KNeighborsClassifier(n_neighbors=best_k,metric=distane)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
t=accuracy_score(y_test,y_pred)
print("Accuracy score is :" ,t)
print("Test error is :",1-t)



    



