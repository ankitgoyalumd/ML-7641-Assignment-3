# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:03:13 2019

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
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.neural_network import MLPClassifier




# get the data and pre process it
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
X=np.array(X)
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled)
y.index=range(len(y))

def kurt(data,label,n_max):
    kurt=[]
    kurt_var=[]
    k_arr=[]
    for i in range(1,n_max):
        transformer = FastICA(n_components=i,random_state=0,max_iter=1000)
        X_transformed = transformer.fit_transform(data)
        abso=np.absolute(kurtosis(X_transformed))
        avg=np.mean(abso)
        variance=np.var(abso)
        kurt.append(avg)
        k_arr.append(i)
        kurt_var.append(variance)
        print(kurt)
    kurt=np.array(kurt)
    k_arr=np.array(k_arr)
    kurt_var=np.array(kurt_var)
    line1, = plt.plot(k_arr,kurt,color='r',marker='o',label='mean_kurtosis')
    line2, = plt.plot(k_arr,kurt_var,color='b',marker='o',label='variance of kurtosis')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel(' kurtosis')
    plt.xlabel('Number of components')
    plt.show()
    return None
kurt(X,y,20)
ica=FastICA(n_components=11,random_state=0,max_iter=1000)
ica_2d = ica.fit_transform(X)
X_ica=ica.transform(X)
plt.scatter(ica_2d[:,0],ica_2d[:,1],  c = y, cmap = "RdGy",
            edgecolor = "None", alpha=1, vmin = 75, vmax = 150)
plt.colorbar()
plt.title('ICA Scatter Plot')     

def plot_samples(S, axis_list=None):
    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                color='steelblue', alpha=0.5)
    if axis_list is not None:
        colors = ['orange', 'red']
        for color, axis in zip(colors, axis_list):
            axis /= axis.std()
            x_axis, y_axis = axis
            # Trick to get legend to work
            plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color='g')
            plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
                       color='b')

    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3,3)
    plt.xlabel('x')
    plt.ylabel('y')
    return None

plt.figure()
plot_samples(ica_2d / np.std(ica_2d))
plt.title('ICA recovered signals')
plt.show()




accu_train1=[]
accu_test1=[]
k=[]
for i in range(2,12):
    X_train, X_test, y_train, y_test = train_test_split(X_ica, y, test_size=0.20)
    km =MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=[i,i,i,i,i,i,i,i],random_state=1)
    km.fit(X_train,y_train)
    km.fit(X_test,y_test)
    accu_train=km.score(X_test,y_test)
    accu_test=km.score(X_train,y_train)
    accu_train1.append(accu_train)
    accu_test1.append(accu_test)
    k.append(i)
accu_train1=np.array(accu_train1)
accu_test1=np.array(accu_test1)
k=np.array(k)
line1, = plt.plot(k,accu_train1,color='r',marker='o',label='Train Accuracy')
line2, = plt.plot(k,accu_test1,color='b',marker='o',label='Test Accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(' accuracy')
plt.xlabel('Number of nodes')
plt.show()

        
    
