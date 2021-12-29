# <<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:37:37 2021

@author: james
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

path = "."  #absolute or relative path to the folder containing the file. 
            #"." for current folder

filename_read = os.path.join(path, "skyserver - original.csv")  # Importing a csv file called skyserver which is our dataset
df = pd.read_csv(filename_read)  # Reading the csv file and saving it in the df variable (data frame)


df = df.drop(df[df['class'] == 'QSO'].sample(frac=1.00).index)  # Dropped all rows that contained 'QSO' in the class column
df['class'] = df['class'].replace({'STAR': 1})  # Renamed STAR entries as the number '1' in the class column
df['class'] = df['class'].replace({'GALAXY': 2})  # Renamed GALAXY entries as the number '2' in the class column

df = df[['objid','ra','dec','u','g','r','i','z','run','rerun','camcol','field','specobjid','redshift','plate','mjd','fiberid','class']]  # Reordered the columns in the data frame to move class column at the end of the data frame

X = df.iloc[:, 0:17] # X is selecting only the first 16 columns. Reordering the columns above allowed to make this task simple
y = df.iloc[:, 17] # y is selecting the 17th column only which is the class column 'star' or 'galaxy' 

X = X.values # Converts the X variable from a data frame which included column headers to a list of values instead
y = y.values # Y is converted to a list of values
# Reasoning for this is to convert it into an appropriate data form and to help make the train/test data split a lot easier 

target_names = ['STAR', 'GALAXY']  # Names of the targets which we will use for labelling in our confusion matrix output


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # X and y is split into training and testing data. 80% training data and 20% testing data

svm = SVC(gamma='scale', decision_function_shape='ovr')  # Support Vector Machine is initialised 
svm.fit(X_train,y_train)  # SVM is applied to X_train and y_train

y_pred = svm.predict(X_test)  # y_pred calculates the SVM prediction on the testing data for X

labels = target_names  # Renaming the target_names as labels which will be called in the confusion matrix

cm = confusion_matrix(y_test, y_pred)  # Confusion matrix function is applied on the y_test and the calculated y_pred variable from the SVM output

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # Confusion matrix is created by calling the results from the 'cm' variable above and also labels (axis of the matrix)
disp.plot()  # Confusion matrix is displayed in the terminal
plt.show() 

# Accuracy and F1 score is printed onto the terminal
print('Accuracy: %.16f' % accuracy_score(y_test, y_pred))  # Accuracy Score of the confusion matrix is calculated (which includes the SVM output)
print('F1: %.16f' % f1_score(y_test,y_pred))  # F1 Score of the confusion matrix is calculated (again using the SVM output)

# # Use 10-fold split
# kf1 = KFold(10, shuffle=True)    # A kfold is created which will use a 5-fold split
# fold1 = 1                      # Counter variable which will start from 1

# # The data is split five ways, for each fold, the 
# # Perceptron is trained, tested and evaluated for accuracy
# for train_index, validate_index in kf1.split(X, y):          # A for loop which will go through each train and validate value and in the split of the X and y column 
#     svm.fit(X[train_index],y[train_index])
#     y_test = y[validate_index]    
#     y_pred = svm.predict(X[validate_index])
#     print(f"Fold #{fold1}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")  # K fold is calculated from 1 to 5 alongside the accuracy of each of those
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     fold1 += 1
