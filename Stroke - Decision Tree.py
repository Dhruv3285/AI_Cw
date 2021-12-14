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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from csv import reader
from sklearn.svm import SVC



path = "."  #absolute or relative path to the folder containing the file. 
            #"." for current folder

filename_read = os.path.join(path, "Stroke data set1.csv")
df = pd.read_csv(filename_read)


X = df[['age','hypertension','heart_disease','avg_glucose_level','bmi']]
y = df[['stroke']]
#df = df.reindex(np.random.permutation(df.index))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train = X_train.reshape(1, -1)
#y_train = y_train.reshape(1, -1)
 
target_names = ['Stroke', 'No Stroke']

decision_tree = DecisionTreeClassifier(criterion = 'entropy')

decision_tree.fit(X_train,y_train)

y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))



#python function, defined here to plot confusion matrices
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, target_names, title='')


#graphical plots of confusion matrix using method above
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, target_names, title='Normalized confusion matrix')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=decision_tree.classes_)
disp.plot()
plt.show()

train_index = (X_train.index.values)
train_index = train_index.reshape(1, -1)
validate_index = (y_test.index.values)
validate_index = validate_index.reshape(1, -1)

# Use 5-fold split
kf = KFold(5, shuffle=True)

X1 = X.values
#X1 = X.index.values
#X1 = X1.reshape(1, -1)
y1 = y.values
#y1 = y.index.values
#y1 = y1.reshape(1, -1)



fold = 1
# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf.split(X1, y1):
    decision_tree.fit(X1[train_index],y1[train_index])
    y_test = y1[validate_index]
    y_pred = decision_tree.predict(X1[validate_index])

    print(f"Fold #{fold}, Training Size: {len(X1[train_index])}, Validation Size: {len(X1[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1


# plt.figure()

# plt.subplot(321)
# plt.scatter(df['age'],df['bmi'], c=y, s=50)

# plt.subplot(322)
# plt.scatter(df['avg_glucose_level'],df['heart_disease'], c=y, s=50)

# plt.subplot(323)
# plt.scatter(df['age'],df['avg_glucose_levels'], c=y, s=50)

# plt.subplot(324)
# plt.scatter(df['heart_disease'],df['age'], c=y, s=50)


#build a multiclass SVM 'ovo' for one-versus-one, and
#fit the data
# multi_svm = SVC(gamma='scale', decision_function_shape='ovo')  
# multi_svm.fit(X_train,y_train)

# #print(X.shape[1])

# y_pred = multi_svm.predict(X_test)

# target_names = ['stroke']
# variety = target_names

# cm = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix')
# print(cm_normalized)

# #confusion matric as a figure
# plt.figure()
# plot_confusion_matrix(cm_normalized, variety, title='Normalized confusion matrix')
# plt.show()

#print(variety)

















#print(y_test)
#print(y_pred)

#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
#print(X)
#print(y)
#print(y)
#print(X1)
#print(X1.tolist())
#print(y1)
#print(X.values.tolist())
#print(y)
#print(X.index.values)
#print(y.index.values)
#print((train_index)) #4088
# print((validate_index)) #1022
#print(len(train_index)) #4088
#print(len(validate_index)) #1022
# print(len(X)) #5110
# print(len(y)) #5110
# print((X_train)) #4088
# print((X_test)) #1022
# print((y_train)) #4088
# print((y_test)) #1022
# print(len(df)) #5110







