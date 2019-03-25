# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 02:29:56 2019

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
from sklearn.decomposition import PCA
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
X=np.array(X)
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled)
y.index=range(len(y))
#Plotting the Cumulative Summation of the Explained Variance
pca1=PCA().fit(X)
plt.figure()
plt.plot(np.cumsum(pca1.explained_variance_ratio_),color='green',marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('weather dataset')
plt.show()
pca=PCA(n_components=8)
pca.fit(X)
#print(pca.explained_variance_)
x_new=pca.fit_transform(X)
pca_inv_data=pca.inverse_transform(np.eye(8))
#get indices of columns with maximum mean
arr=pca_inv_data.mean(axis=0)
ind=np.argsort(arr)
print(ind)
#X=pd.DataFrame(X)
#check the first few rows of the dataframe
#print(X)
#print(pca.components_)
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.7)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
plt.xlim(-0.5,1)
plt.ylim(-0.5,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()
myplot(x_new[:,0:3],np.transpose(pca.components_[0:3, :]))
plt.show()

#plot original variance

plt.figure(figsize = (10,6.5));
plt.semilogy(np.square(X.std(axis=0)) / np.square(X.std(axis=0)).sum(), '--o', label = 'variance ratio');
plt.semilogy(X.mean(axis=0) / np.square(X.mean(axis=0)).sum(), '--o', label = 'mean ratio');
plt.xlabel('original feature', fontsize = 20);
plt.ylabel('variance', fontsize = 20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0, 18]);
plt.legend(loc='lower left', fontsize=18);

#plot variance after PCA

fig = plt.figure(figsize=(10, 6.5))
plt.plot(pca_inv_data.mean(axis=0), '--o', label = 'mean')
plt.plot(np.square(pca_inv_data.std(axis=0)), '--o', label = 'variance')
plt.legend(loc='lower right')
plt.ylabel('feature contribution', fontsize=20);
plt.xlabel('feature index', fontsize=20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0, 18])
plt.legend(loc='lower left', fontsize=18)

X_1=X
#Get the reduced form of the data
drop_col=[17,9,6,7,12,13,10,14,2,8,3]
X_1.drop(X.columns[drop_col],axis=1,inplace=True)


#Apply clustering to the new algrothm
def matchfn(y_true,y_pred):
    s=0
    for i in range(len(y_true)):
        if (y_true[i]==y_pred[i]):
            s=s+1
        else:
            s=s
    return s/len(y_pred)
kmeans = KMeans(n_clusters=2, random_state=0,max_iter=10000,tol=1e-9).fit(X_1)
ans=kmeans.predict(X_1)
print(matchfn(y,ans))   
#plot the clusters
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_1[ans==0][11],X_1[ans==0][15],X_1[ans==0][16],label='Class1',c='red')    
ax.scatter(X_1[ans==1][11],X_1[ans==1][15],X_1[ans==1][16],label='Class2',c='blue') 
#ax.scatter(X[ans==2][0],X[ans==2][6],X[ans==2][27],label='Class2',c='green') 
ax.set_xlabel('11')
ax.set_ylabel('15')
ax.set_zlabel('16')

#ax.legend()
plt.show()

plt.scatter(X_1[ans==0][11],X_1[ans==0][16],label='Class1',c='red')    
plt.scatter(X_1[ans==1][11],X_1[ans==1][16],label='Class2',c='blue') 
#print(X.head())
plt.show()
x_new=pd.DataFrame(x_new)
print(x_new.head())
print(X.head())

def components(K):
    Sum_of_squared_distances = []
    k=[]
    accuracy_train=[]
    accuracy_test=[]
    score=[]
    for i in range(1,K):
        print(i)
        agglo=PCA(n_components=8)
        #X_new_train,y_new_train=transformer.fit(X_train,y_train) 
        #X_new_test,y_new_test = transformer.transform(X_test,y_test)
        agglo.fit(X)
        X_reduced=agglo.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.20)
        km =MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=[i,i,i,i,i,i,i],random_state=1)
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