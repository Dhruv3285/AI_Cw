# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:04:50 2021

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




data = pd.read_csv('diabetes_data_upload.csv')


data['Gender'] = data['Gender'].replace("Male",1).replace("Female",0)

data = data.replace("Yes",1).replace("No",0)
data['class'] = data['class'].replace("Positive",1).replace("Negative",0) 


# split dataset
X = data.iloc[:, 0:16]
y = data.iloc[:, 16]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.2)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

import math 
math.sqrt(len(y_test))
# gives 10 but we need odd k so 9 shall be used 


print("<---------------------euclidean metric--------------------------------->")

# implementation of KNN
classifier = KNeighborsClassifier(n_neighbors= 9, p=2, metric = 'euclidean')
classifier.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=1, n_neighbors=9, p=2,
                     weights='uniform')

# predict the test set results 
y_pred = classifier.predict(X_test)


# Evaluate model using matrix 
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()

print("<-----------------------manhattan metric with all the features------------------------------->")



classifier2 = KNeighborsClassifier(n_neighbors= 9, p=2, metric = 'manhattan')
classifier2.fit(X_train, y_train) 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=1, n_neighbors=9, p=2,
                     weights='uniform')

y_pred2 = classifier2.predict(X_test)

# Evaluate model using matrix 
cm2 = confusion_matrix(y_test, y_pred2)
print (cm2)
print(f1_score(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))

# PCA(princple componenet analysis) 
def plot_correlation(datab):
    rcParams['figure.figsize']=15,20
    fig= plt.figure()
    sns.heatmap(datab.corr(), annot=True, fmt= ".2f")
    plt.show()
    fig.savefig('corr.png')

plot_correlation(data)


k_scores = []

print("<-----------------------manhattan metric but age dropped out----------------->")
# dropping out gender
data.drop('Gender',1,inplace=True) 
 # split dataset
A = data.iloc[:, 0:14]
b = data.iloc[:, 14]
A_train, A_test, b_train, b_test = train_test_split(A, b, random_state=0, test_size= 0.2)

# Feature scaling
sc_A = StandardScaler()
A_train = sc_A.fit_transform(A_train)

classifier3 = KNeighborsClassifier(n_neighbors= 9, p=2, metric = 'manhattan')
classifier3.fit(A_train, b_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
     metric_params=None, n_jobs=1, n_neighbors=9, p=2,
     weights='uniform')

b_pred = classifier3.predict(A_test)

# Evaluate model using matrix 
cm3 = confusion_matrix(b_test, b_pred)
print (cm3)
print(f1_score(b_test, b_pred))
print(accuracy_score(b_test, b_pred))






def knnManhattan(kval, dataset):
      print("<-----------------------manhattan metric but age dropped out of it k="+str(kval)+"----------------->")
  
  
      # split dataset
      A = dataset.iloc[:, 0:14]
      b = dataset.iloc[:, 14]
      A_train, A_test, b_train, b_test = train_test_split(A, b, random_state=0, test_size= 0.2)

      # Feature scaling
      sc_A = StandardScaler()
      A_train = sc_A.fit_transform(A_train)

      classifier = KNeighborsClassifier(n_neighbors= kval, p=2, metric = 'manhattan')
      classifier.fit(A_train, b_train)
      KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=1, n_neighbors=kval, p=2,
                     weights='uniform')

      b_pred = classifier.predict(A_test)

    # Evaluate model using cross validation
      scores = cross_val_score(classifier, A, b, cv=5, scoring='accuracy')
      k_scores.append(scores.mean())


k_range = range(1, 31)
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = knnManhattan(k,data)

# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


