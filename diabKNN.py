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



data = pd.read_csv('diabetes_data_upload.csv')


print(data.columns)

data['Gender'] = data['Gender'].replace("Male",1).replace("Female",0)

#data['Polyuria'] = data['Polyuria'].replace("Yes",1).replace("No",0)

#data['Polydipsia'] = data['Polydipsia'].replace("Yes",1).replace("No",0)

#data['muscle stiffness'] = data['muscle stiffness'].replace("Yes",1).replace("No",0)
#data['Alopecia'] = data['Alopecia'].replace("Yes",1).replace("No",0)
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











