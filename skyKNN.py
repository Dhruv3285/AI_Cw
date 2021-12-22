# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 14:20:22 2021

@author: dhruv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt2

dataset = pd.read_csv('skyserverV2.csv')

# dropping obselete features 
dataset.drop('objid',1,inplace=True) 
dataset.drop('rerun',1,inplace=True) 

def knnalg(X,y, metric):
  print("<---------------------"+ metric+ "metric--------------------------------->")
  # split dataset
  X_train, X_test, y_train, y_test = train_test_split(X.values, y, random_state=0, test_size= 0.2)

  #  Feature scaling
  sc_X = StandardScaler()
  X_train = sc_X.fit_transform(X_train)

  import math 
  math.sqrt(len(y_test))
  # gives 42 but we need odd k so 43 shall be used 

  # implementation of KNN
  classifier = KNeighborsClassifier(n_neighbors= 43, p=2, metric = metric)
  classifier.fit(X_train, y_train)
  KNeighborsClassifier(algorithm='auto', leaf_size=30, metric = metric,
                     metric_params=None, n_jobs=1, n_neighbors=43, p=2,
                     weights='uniform')

  # predict the test set results 
  y_pred = classifier.predict(X_test)

  # Evaluate model using matrix 
  cm = confusion_matrix(y_test, y_pred)
  print(f1_score(y_test, y_pred))
  print(accuracy_score(y_test, y_pred))

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
  disp.plot()


## running knn model using 2 versions of metrics 
c = dataset.iloc[:, 0:15]
d = dataset.iloc[:, 15]
knnalg(c,d,'euclidean')
knnalg(c,d,'manhattan')

# PCA(princple componenet analysis) 
def plot_correlation(datab):
    rcParams['figure.figsize']=15,20
    fig= plt.figure()
    sns.heatmap(datab.corr(), annot=True, fmt= ".2f")
    plt.show()

## running PCA to check feature correlations 
plot_correlation(dataset)

## dropping fields after PCA alalysis 
dataset.drop('specobjid',1,inplace=True) 
dataset.drop('mjd',1,inplace=True) 
dataset.drop('plate',1,inplace=True)
dataset.drop('z',1,inplace=True) 
dataset.drop('i',1,inplace=True) 
dataset.drop('r',1,inplace=True) 
dataset.drop('g',1,inplace=True) 

## rerunning KNN
v = dataset.iloc[:, 0:8]
w = dataset.iloc[:, 8]
knnalg(v,w,'euclidean')

## running PCA to see the new diagram 
plot_correlation(dataset)


k_scores = []

## function that runs manhattan metric with diffirent k values 
def knnEuclidean(kval, dataset):
  
      # split dataset
      A = dataset.iloc[:, 0:8]
      b = dataset.iloc[:, 8]
      A_train, A_test, b_train, b_test = train_test_split(A.values, b, random_state=0, test_size= 0.2)

      # Feature scaling
      sc_A = StandardScaler()
      A_train = sc_A.fit_transform(A_train)

      classifier = KNeighborsClassifier(n_neighbors= kval, p=2, metric = 'euclidean')
      classifier.fit(A_train, b_train)
      KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=1, n_neighbors=kval, p=2,
                     weights='uniform')
     
      b_pred = classifier.predict(A_test)

    # Evaluate model using cross validation
      scores = cross_val_score(classifier, A.values, b, cv=5, scoring='accuracy')
      k_scores.append(scores.mean())


k_range = range(30, 60)
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = knnEuclidean(k,dataset)



# plot to see clearly
plt2.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()






