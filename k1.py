# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:23:29 2021

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


dataset = pd.read_csv('Stroke data set1.csv')
print( len(dataset) )
print( dataset.head() )



## changing smoking status to ints instead of strings 1->smoked or smokes, 0-> never smoked 
dataset['smoking_status']= dataset['smoking_status'].replace("smokes", 1).replace("formerly smoked", 1).replace("never smoked", 0) 

## change gender status into binary 
dataset['gender']= dataset['gender'].replace("Male", 1).replace("Female",0)

##
dataset['Residence_type']= dataset['Residence_type'].replace("Urban",1).replace("Rural",0)


## average out bmi where it is 0
for column in ['bmi']:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column]= dataset[column].replace(np.NaN, mean )

## drop these fields from dataset
dataset.drop('id', 1, inplace=True)
dataset.drop('work_type', 1, inplace=True)
dataset.drop('ever_married', 1, inplace=True)



## deleting rows where smoking status in unknown 
dataset = dataset[dataset.smoking_status != "Unknown"]


## deleting gender where it is labelled as other 
dataset = dataset[dataset.gender != "Other"]



#import os
#path = "."
#filename_write = os.path.join(path, "new file 3.csv")
#dataset.to_csv(filename_write, index=False) # Specify index = false to not write row numbers
#print("Done")





# split dataset
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.2)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

import math 
math.sqrt(len(y_test))
# gives 26 but we need odd k so 25 shall be used 


# implementation of KNN
classifier = KNeighborsClassifier(n_neighbors= 25, p=2, metric = 'euclidean')
classifier.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=1, n_neighbors=25, p=2,
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
















