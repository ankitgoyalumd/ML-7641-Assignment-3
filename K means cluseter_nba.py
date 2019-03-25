# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 01:32:44 2019

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
from sklearn.neural_network import MLPClassifier





os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1/Dataset/nba')
data=pd.read_csv('nba.csv')
data=data.dropna()
print(data.info())
fields_to_drop=['TARGET_5Yrs','Name']
X = data.drop(fields_to_drop,axis=1)
y = data['TARGET_5Yrs']
y.index=range(len(y))
#X=pd.DataFrame(X)    
#min_max_scaler = preprocessing.MinMaxScaler()
#np_scaled = min_max_scaler.fit_transform(X)
#X = pd.DataFrame(np_scaled)
#y.index=range(len(y))



#print(kmeans.score(X))
def matchfn(y_true,y_pred):
    s=0
    for i in range(len(y_true)):
        if (y_true[i]==y_pred[i]):
            s=s+1
        else:
            s=s
    return s/len(y_pred)
def elbow_curve():
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-',marker='o')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow curve Optimal k')
    plt.show()
    return None
    
elbow_curve() 

print(X.head())
kmeans = KMeans(n_clusters=4, random_state=0,max_iter=10000,tol=1e-5).fit(X)
ans=kmeans.predict(X)
print(matchfn(y,ans))    
#print(X[ans==0][0])
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[ans==0]['MIN'],X[ans==0]['FGM'],X[ans==0]['FG%'],label='Class1',c='red')    
ax.scatter(X[ans==1]['MIN'],X[ans==1]['FGM'],X[ans==1]['FG%'],label='Class2',c='blue')
ax.scatter(X[ans==2]['MIN'],X[ans==2]['FGM'],X[ans==2]['FG%'],label='Class2',c='green') 
ax.scatter(X[ans==3]['MIN'],X[ans==3]['FGM'],X[ans==3]['FG%'],label='Class2',c='orange') 


#ax.scatter(X[ans==2][0],X[ans==2][6],X[ans==2][27],label='Class2',c='green') 
ax.set_xlabel('1')
ax.set_ylabel('3')
ax.set_zlabel('5')

#ax.legend()
plt.show()

plt.scatter(X[ans==0]['MIN'],X[ans==0]['FGM'],label='Class1',c='red')    
plt.scatter(X[ans==1]['MIN'],X[ans==1]['FGM'],label='Class2',c='blue')
plt.scatter(X[ans==2]['MIN'],X[ans==2]['FGM'],label='Class2',c='green') 
plt.scatter(X[ans==3]['MIN'],X[ans==3]['FGM'],label='Class2',c='orange') 

plt.xlabel('3')
plt.ylabel('5')
#print(X.head())
plt.show()

def components(K):
    Sum_of_squared_distances = []
    k=[]
    accuracy_train=[]
    accuracy_test=[]
    score=[]
    for i in range(1,K):
        print(i)
        agglo=KMeans(n_clusters=i)
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

components(14)
