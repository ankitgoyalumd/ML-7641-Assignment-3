# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 02:43:45 2019

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
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn import mixture
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.neural_network import MLPClassifier




os.getcwd()
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
#min_max_scaler = preprocessing.MinMaxScaler()
#np_scaled = min_max_scaler.fit_transform(X)
#X = pd.DataFrame(np_scaled)
#y.index=range(len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
def components(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    accu_train1=[]
    accu_train2=[]
    accu_train3=[]
    accu_train4=[]
    k_arr=[]
    print(type(k_arr))
    for i in range(1,15):
        g1 = mixture.GaussianMixture(n_components=i,covariance_type='diag')
        g2 = mixture.GaussianMixture(n_components=i,covariance_type='spherical')
        g3 = mixture.GaussianMixture(n_components=i,covariance_type='tied')
        g4 = mixture.GaussianMixture(n_components=i,covariance_type='full')
        g1.fit(X)
        g2.fit(X)
        g3.fit(X)
        g4.fit(X)
        #g.fit(X_test)
        #score_train=(np.sum(g.score(X_train)))
        #score_test=(np.sum(g.score(X_test)))
        score_train1=g1.bic(X)
        score_train2=g2.bic(X)
        score_train3=g3.bic(X)
        score_train4=g4.bic(X)
        #score_test=g.bic(X_test)
        #k_arr.append(i)
        accu_train1.append(score_train1)
        accu_train2.append(score_train2)
        accu_train3.append(score_train3)
        accu_train4.append(score_train4)
        #accu_test.append(score_test)
        k_arr.append(i)
        #print(accu_test)
    accu_train1=np.array(accu_train1)
    accu_train2=np.array(accu_train2)
    accu_train3=np.array(accu_train3)
    accu_train4=np.array(accu_train4)
    k_arr=np.array(k_arr)
    line1, = plt.plot(k_arr,accu_train1,color='r',marker='o',label='diag')
    line2, = plt.plot(k_arr,accu_train2,color='g',marker='o',label='spherical')
    line3, = plt.plot(k_arr,accu_train3,color='b',marker='o',label='tied')
    line4, = plt.plot(k_arr,accu_train4,color='m',marker='o',label='full')
    #line2, = plt.plot(k_arr,accu_train,color='b',label='test_accuracy',marker='p')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('BIC score')
    plt.xlabel('Number of components')
    plt.show()
    return None
#components(X,y)
def sh_score(x,y):
    accu_train1=[]
    accu_train2=[]
    accu_train3=[]
    accu_train4=[]
    k_arr=[]
    print(type(k_arr))
    for i in range(2,15):
        g1 = mixture.GaussianMixture(n_components=i,covariance_type='diag')
        g2 = mixture.GaussianMixture(n_components=i,covariance_type='spherical')
        g3 = mixture.GaussianMixture(n_components=i,covariance_type='tied')
        g4 = mixture.GaussianMixture(n_components=i,covariance_type='full')
        g1.fit(X)
        g2.fit(X)
        g3.fit(X)
        g4.fit(X)
        #accu_test.append(score_test)
        #g.fit(X_test)
        #score_train=(np.sum(g.score(X_train)))
        #score_test=(np.sum(g.score(X_test)))
        labels1=g1.predict(X)
        labels2=g1.predict(X)
        labels3=g1.predict(X)
        labels4=g1.predict(X)
        score_train1=metrics.silhouette_score(X,labels1, metric='euclidean')
        score_train2=metrics.silhouette_score(X,labels2, metric='euclidean')
        score_train3=metrics.silhouette_score(X,labels3, metric='euclidean')
        score_train4=metrics.silhouette_score(X,labels4, metric='euclidean')
        #score_test=g.bic(X_test)
        #k_arr.append(i)
        accu_train1.append(score_train1)
        accu_train2.append(score_train2)
        accu_train3.append(score_train3)
        accu_train4.append(score_train4)
        #accu_test.append(score_test)
        k_arr.append(i)
        print(accu_train1)
    accu_train1=np.array(accu_train1)
    accu_train2=np.array(accu_train2)
    accu_train3=np.array(accu_train3)
    accu_train4=np.array(accu_train4)
    k_arr=np.array(k_arr)
    line1, = plt.plot(k_arr,accu_train1,color='r',marker='o',label='diag')
    line2, = plt.plot(k_arr,accu_train2,color='g',marker='o',label='spherical')
    line3, = plt.plot(k_arr,accu_train3,color='b',marker='o',label='tied')
    line4, = plt.plot(k_arr,accu_train4,color='m',marker='o',label='full')
    #line2, = plt.plot(k_arr,accu_train,color='b',label='test_accuracy',marker='p')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('silhouette score')
    plt.xlabel('Number of components')
    plt.show()
    return None
#sh_score(X,y)
gn = mixture.GaussianMixture(n_components=2,covariance_type='spherical')
gn.fit(X)
arr=np.array((gn.predict(X)))
#print(arr[1:100])
    
def matchfn(y_true,y_pred):
    s=0
    for i in range(len(y_true)):
        if (y_true[i]==y_pred[i]):
            s=s
        else:
            s=s+1
    return s/len(y_pred)

print(matchfn(y,arr))

for i in range(200,1000,10):
    gn = mixture.GaussianMixture(n_components=2,covariance_type='full')
    gn.fit(X)
    arr=np.array((gn.predict(X)))
prob=gn.predict_proba(X)
#prob=prob/np.sum(prob)
mod1=(prob[:,1])
mod2=prob[:,0]
bins = np.linspace(-0.2, 1)
plt.hist(mod1, bins, alpha=0.7, label='mod1',density=True)
plt.hist(mod2, bins, alpha=0.7, label='mod2',density=True)
plt.legend(loc='upper left')
plt.show()
        
def comp(K):
    Sum_of_squared_distances = []
    k=[]
    accuracy_train=[]
    accuracy_test=[]
    score=[]
    for i in range(1,K):
        print(i)
        agglo=mixture.GaussianMixture(n_components=2,covariance_type='diag')
        #X_new_train,y_new_train=transformer.fit(X_train,y_train) 
        #X_new_test,y_new_test = transformer.transform(X_test,y_test)
        agglo.fit(X)
        X_reduced=agglo.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.20)
        km =MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=[i,i,i,i,i],random_state=1)
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
comp(14)