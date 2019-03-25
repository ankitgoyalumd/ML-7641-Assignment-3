# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:07:18 2019

@author: Ankit Goyal
"""
import pandas as pd
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree
from keras.layers import Dense
import numpy as np
import os
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn import metrics



os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1')
data=pd.read_csv('weather_data.csv')
data=data.dropna()

#Mapping RainTomoorow and Rain Today to 0 and 1
data['RainToday'] = data['RainToday'].map({'No': 0, 'Yes': 1})
data['RainTomorrow'] = data['RainTomorrow'].map({'No': 0, 'Yes': 1})


#Adding dummy variable for attributes with more than 2 categories
Categorical= ['WindGustDir', 'WindDir9am', 'WindDir3pm']
for each in Categorical:
    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)
df = pd.concat([data, dummies], axis=1)
fields_to_drop = ['Date', 'Location','WindGustDir', 'WindDir9am', 'WindDir3pm']
df = df.drop(fields_to_drop, axis=1)

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']
y.index=range(len(y))
X=np.array(X)
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled)
y.index=range(len(y))

## Johnson Lindenstrauss Lemma
print(johnson_lindenstrauss_min_dim(n_samples=112925, eps=[0.99,0.5, 0.1, 0.01]))
print(X.info())

def matchfn(y_true,y_pred):
    s=0
    for i in range(len(y_true)):
        if (y_true[i]==y_pred[i]):
            s=s+1
        else:
            s=s
    
    return max((s/len(y_pred)),(1-(s/len(y_pred))))

def components(K):
    Sum_of_squared_distances = []
    k=[]
    accuracy=[]
    score=[]
    for i in range(1,K):
        transformer = GaussianRandomProjection(n_components=i,eps=0.1)
        #transformer1 = GaussianRandomProjection(n_components=i,eps=0.5)
        #transformer2 = GaussianRandomProjection(n_components=i,eps=0.6)
        X_new = transformer.fit_transform(X)
        #label=transformer.predict(X)
        km = KMeans(n_clusters=2, random_state=0,max_iter=10000,tol=1e-9).fit(X_new)
        label=km.predict(X_new)
        accu=matchfn(y,label)
        #score_train1=metrics.silhouette_score(X_new,label, metric='euclidean')
        Sum_of_squared_distances.append(km.inertia_)
        k.append(i)
        accuracy.append(accu)
        #score.append(score_train1)
        #print(Sum_of_squared_distances)
    k=np.array(k)
    Sum_of_squared_distances=np.array(Sum_of_squared_distances)
    score=np.array(score)
    accuracy=np.array(accuracy)
    #line1,=plt.plot(k, Sum_of_squared_distances, 'bx-',marker='o')
    #line2,=plt.plot(k,score,color='g',marker='o')
    line3,=plt.plot(k,accuracy,color='r',marker='o')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    #plt.title('Elbow curve Optimal k')
    #plt.ylim(0,1)
    plt.show()
    return None

#components(20)

def eps():
    Sum_of_squared_distances = []
    k=[]
    score=[]
    eps=[0.8,0.6,0.4,0.2,0.05,0.01]
    for i in eps:
        transformer = GaussianRandomProjection(n_components=4,eps=i)
        X_new = transformer.fit_transform(X)
        #label=transformer.predict(X)
        km = KMeans(n_clusters=2, random_state=0,max_iter=10000,tol=1e-9).fit(X_new)
        #label=km.predict(X_new)
        #score_train1=metrics.silhouette_score(X_new,label, metric='euclidean')
        Sum_of_squared_distances.append(km.inertia_)
        k.append(i)
        #score.append(score_train1)
        print(Sum_of_squared_distances)
    k=np.array(k)
    Sum_of_squared_distances=np.array(Sum_of_squared_distances)
    score=np.array(score)
    line1,=plt.plot(k, Sum_of_squared_distances, 'bx-',marker='o')
    #line2,=plt.plot(k,score,color='g',marker='o')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow curve Optimal eps')
    plt.show()
    return None

#eps()        
        
    

transformer = GaussianRandomProjection(n_components=10,eps=0.6)
X_new=transformer.fit_transform(X)
X_new=pd.DataFrame(X_new)
km = KMeans(n_clusters=2, random_state=0,max_iter=10000,tol=1e-9).fit(X_new)
label=km.predict(X_new)

#print(X_new[0])
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_new[label==0][0],X_new[label==0][1],X_new[label==0][2],label='Class1',c='red')    
ax.scatter(X_new[label==1][0],X_new[label==1][1],X_new[label==1][2],label='Class2',c='blue') 
#ax.scatter(X[ans==2][0],X[ans==2][6],X[ans==2][27],label='Class2',c='green') 
ax.set_xlabel('0')
ax.set_ylabel('1')
ax.set_zlabel('2')
plt.show()

def comp1(K):
    Sum_of_squared_distances = []
    k=[]
    accuracy_train=[]
    accuracy_test=[]
    score=[]
    for i in range(1,K):
        print(i)
        agglo=GaussianRandomProjection(n_components=10,eps=0.6)
        #X_new_train,y_new_train=transformer.fit(X_train,y_train) 
        #X_new_test,y_new_test = transformer.transform(X_test,y_test)
        agglo.fit(X)
        X_reduced=agglo.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.20)
        km =MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=[8,8,8,8,8],random_state=1)
        km.fit(X_train,y_train)
        km.fit(X_test,y_test)
        #transformer1 = GaussianRandomProjection(n_components=i,eps=0.5)
        #transformer2 = GaussianRandomProjection(n_compo
        label_train=km.predict(X_train)
        label_test=km.predict(X_test)
        accu_train=km.score(X_test,y_test)
        accu_test=km.score(X_train,y_train)
        #score_train1=metrics.silhouette_score(X_new,label, metric='euclidean')
        #Sum_of_squared_distances.append(km.inenents=i,eps=0.6)       
        #label=transformer.predicn)rtia_)
        k.append(i)
        accuracy_train.append(accu_train)
        accuracy_test.append(accu_test)
        #score.append(score_train1)
        #print(accuracy)
    k=np.array(k)
    Sum_of_squared_distances=np.array(Sum_of_squared_distances)
    score=np.array(score)
    accuracy_train=np.array(accuracy_train)
    accuracy_test=np.asarray(accuracy_test)
    #line1,=plt.plot(k, Sum_of_squared_distances, 'bx-',marker='o')
    #line2,=plt.plot(k,score,color='g',marker='o')
    line3,=plt.plot(k,accuracy_train,color='r',marker='o',label='train_accuracy')
    line4,=plt.plot(k,accuracy_test,color='g',marker='o',label='test_accuracy')
    #plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.xlabel('k')
    plt.legend()
    plt.ylabel('accuracy')
    #plt.ylim(0,1)
    plt.show()
    return None
comp1(12)