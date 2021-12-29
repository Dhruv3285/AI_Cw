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
import math 

## dataset 
dataset = pd.read_csv('skyserver - original.csv')

## droppig QSO field 
dataset = dataset.drop(dataset[dataset['class'] == 'QSO'].sample(frac=1.00).index)
# replacing star class with 1 and universe class with 2
dataset['class'] = dataset['class'].replace({'STAR': 1})
dataset['class'] = dataset['class'].replace({'GALAXY': 2})

# rearranging the columns to have class in the last column 
dataset = dataset[['objid','ra','dec','u','g','r','i','z','run','rerun','camcol','field','specobjid','redshift','plate','mjd','fiberid','class']]

##method that runs KNN model on the dataset
def knnalg(X,y, metric,doCm):
    
  print("<---------------------"+ metric+ "metric--------------------------------->")
  # split dataset 
  X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, random_state=0, test_size= 0.2)

  #  calculate the ideal value of k that will be used in the model
  math.sqrt(len(y_test))  # gives 42 but we need odd k will he 43 

  # implementation of KNN classifier
  classifier = KNeighborsClassifier(n_neighbors= 43, p=2, metric = metric)
  classifier.fit(X_train, y_train)

  # predict the test set results 
  y_pred = classifier.predict(X_test)

  # Evaluate model using the confusion matrix 
  if(doCm==True):
      cm = confusion_matrix(y_test, y_pred)
      print(cm)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['STAR', 'GALAXY'] )
      disp.plot()
      
  # Evaluate using f1 score 
  print(f1_score(y_test, y_pred))
  # evaluate using accracy 
  print(accuracy_score(y_test, y_pred))


# dropping obselete features which are not required 
dataset.drop('objid',1,inplace=True) 
dataset.drop('rerun',1,inplace=True) 


## defining features and class 
c = dataset.iloc[:, 0:15]
d = dataset.iloc[:, 15]
# running knn model using 2 diffirent versions of metrics 
knnalg(c,d,'euclidean',True)
knnalg(c,d,'manhattan',True)

# PCA(princple componenet analysis) method
def plot_correlation(datab, ind):
    rcParams['figure.figsize']=15,20
    fig = plt.figure() 
    sns.heatmap(datab.corr(), annot=True, fmt= ".2f")
    plt.show()
    

## running PCA to check feature and class correlations 
plot_correlation(dataset,True)

## dropping unwanted fields after PCA alalysis 
dataset.drop('specobjid',1,inplace=True) 
dataset.drop('mjd',1,inplace=True) 
dataset.drop('z',1,inplace=True) 
dataset.drop('i',1,inplace=True) 
dataset.drop('r',1,inplace=True) 
dataset.drop('g',1,inplace=True) 

## rerunning KNN model with minumal fields using euclidean metric
v = dataset.iloc[:, 0:9]
w = dataset.iloc[:, 9]
knnalg(v,w,'euclidean',True)

## running PCA to check see new feature list 
plot_correlation(dataset,False)

k_scores = []
m_scores = []

## function that runs manhattan metric with diffirent k values 
def knnEuclidean(kval, dataset, mt):
  
      # split dataset
      A = dataset.iloc[:, 0:9]
      b = dataset.iloc[:, 9]
      A_train, A_test, b_train, b_test = train_test_split(A.values, b, random_state=0, test_size= 0.2)

      classifier = KNeighborsClassifier(n_neighbors= kval, p=2, metric = mt)
      classifier.fit(A_train, b_train)
     
      b_pred = classifier.predict(A_test)

    # Evaluate model using cross validation of two diffirent metrics 
      scores = cross_val_score(classifier, A.values, b, cv=5, scoring='accuracy')
      if(mt=='euclidean'):   
          k_scores.append(scores.mean())
      if(mt=='hamming'):
         m_scores.append(scores.mean()) 
          
## range of k value we used for cross validation 
k_range = range(35, 50)
# iterate of each k value and gun the classifier
#for k in k_range:
 #  knn = knnEuclidean(k,dataset,'euclidean')
  # knn = knnEuclidean(k,dataset,'hamming')
    
## defining new dataframe flot a plot
#df=pd.DataFrame({'x_values': k_range, 'y1_values': k_scores, 'y2_values': m_scores })
 
# multiple line plots
#plt.plot( 'x_values', 'y1_values', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="euclidean")
#plt.plot( 'x_values', 'y2_values', data=df, marker='o', markerfacecolor='green', markersize=12, color='lightgreen', linewidth=4,label= "hamming")
# show legend
#plt.legend()
# show graph
#plt.show()
print("done")




