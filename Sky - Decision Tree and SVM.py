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

filename_read = os.path.join(path, "skyserver.csv")
df = pd.read_csv(filename_read)

df = df.reindex(np.random.permutation(df.index))

X = df.iloc[:, 0:16]
y = df.iloc[:, 17] 
#df = df.reindex(np.random.permutation(df.index))

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


target_names = ['STAR', 'GALAXY']


decision_tree = DecisionTreeClassifier(criterion = 'entropy')

decision_tree.fit(X_train,y_train)

y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

cm1 = confusion_matrix(y_test, y_pred)
decisionT = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=decision_tree.classes_)
decisionT.plot()
plt.show()

# Use 5-fold split
kf = KFold(5, shuffle=True)
fold = 1


# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf.split(X, y):
    decision_tree.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = decision_tree.predict(X[validate_index])

    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1

plt.figure()

plt.subplot(321)
plt.scatter(df['u'],df['g'], c=y, s=25)

plt.subplot(322)
plt.scatter(df['r'],df['i'], c=y, s=25)

plt.subplot(323)
plt.scatter(df['u'],df['z'], c=y, s=25)



#build a multiclass SVM 'ovo' for one-versus-one, and
#fit the data
multi_svm = SVC(gamma='scale', decision_function_shape='ovo')  
multi_svm.fit(X_train,y_train)


y_pred1 = multi_svm.predict(X_test)

variety = target_names

cm = confusion_matrix(y_test, y_pred1)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=variety)
disp.plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred1)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred1))







