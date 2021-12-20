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

filename_read = os.path.join(path, "skyserver.csv")    # Importing a csv file called skyserver which is our dataset
df = pd.read_csv(filename_read)  # Reading the csv file and saving it in the df variable

#df = df.reindex(np.random.permutation(df.index))

X = df.iloc[:, 0:16]     # X is selecting only the first 16 columns
y = df.iloc[:, 17]       # y is selecting the 17th column which is 'xxx'
#df = df.reindex(np.random.permutation(df.index))

X = X.values       # Converts the X variable from to a list of values which helps to make the train/test split easier (in the appropriate data form)
y = y.values       # Y is converted to a list of values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # X and y is split into training and testing data with test being 20%


target_names = ['STAR', 'GALAXY']  # Names of the targets which we will use for labelling in our analysis output


decision_tree = DecisionTreeClassifier(criterion = 'entropy')

decision_tree.fit(X_train,y_train)

y_pred = decision_tree.predict(X_test)   # y_pred is saving the prediction of the decision tree 

accuracy = accuracy_score(y_test, y_pred) # Accuracy score is calculated using y_test and y_pred

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))      # Score is outputted

cm1 = confusion_matrix(y_test, y_pred)                # Confusion matrix is created using the y_test and the calculated y_pred variable from the decision tree
decisionT = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=decision_tree.classes_)
decisionT.plot() 
plt.show()   # Confusion matrix is displayed

# Use 5-fold split
# kf = KFold(5, shuffle=True)    # A kfold is created which will use a 5-fold split
# fold = 1                      # Counter variable which will start from 1


# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
# for train_index, validate_index in kf.split(X, y):          # A for loop which will go through each train and validate value and in the split of the X and y column 
#     decision_tree.fit(X[train_index],y[train_index])
#     y_test = y[validate_index]    
#     y_pred = decision_tree.predict(X[validate_index])
#     print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")  # K fold is calculated from 1 to 5 alongside the accuracy of each of those
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     fold += 1

# plt.figure()

# plt.subplot(321)
# plt.scatter(df['u'],df['g'], c=y, s=25)

# plt.subplot(322)
# plt.scatter(df['r'],df['i'], c=y, s=25)

# plt.subplot(323)
# plt.scatter(df['u'],df['z'], c=y, s=25)



#build a multiclass SVM 'ovo' for one-versus-one, and
#fit the data

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)  # X and y is split into training and testing data with test being 20%

multi_svm = SVC(gamma='scale', decision_function_shape='ovo')  # Support Vector Machine is initialised 
multi_svm.fit(X_train1,y_train1)  # SVM is applied to X_train and y_train


y_pred1 = multi_svm.predict(X_test1)    # y_pred1 calculates the multi_svm prediction on the testing data for X

variety = target_names   

cm = confusion_matrix(y_test1, y_pred1)  # Confusion matrix is created using the y_test and the calculated y_pred variable from the SVM

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=variety)
disp.plot()   
plt.show()  # Confusion matrix is displayed

accuracy = accuracy_score(y_test1, y_pred1)
print('Accuracy: %.2f' % accuracy_score(y_test1, y_pred1))   # Accuracy of the confusion matrix is outputted

# Use 5-fold split
kf1 = KFold(10, shuffle=True)    # A kfold is created which will use a 5-fold split
fold1 = 1                      # Counter variable which will start from 1


# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index1, validate_index1 in kf1.split(X, y):          # A for loop which will go through each train and validate value and in the split of the X and y column 
    decision_tree.fit(X[train_index1],y[train_index1])
    y_test1 = y[validate_index1]    
    y_pred1 = decision_tree.predict(X[validate_index1])
    print(f"Fold #{fold1}, Training Size: {len(X[train_index1])}, Validation Size: {len(X[validate_index1])}")  # K fold is calculated from 1 to 5 alongside the accuracy of each of those
    print('Accuracy: %.2f' % accuracy_score(y_test1, y_pred1))
    fold1 += 1





# =======
# # -*- coding: utf-8 -*-
# """
# Created on Tue Nov  9 19:37:37 2021

# @author: james
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt  
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split 
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Perceptron
# from sklearn import metrics
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import KFold
# from csv import reader
# from sklearn.svm import SVC



# path = "."  #absolute or relative path to the folder containing the file. 
#             #"." for current folder

# filename_read = os.path.join(path, "skyserver.csv")
# df = pd.read_csv(filename_read)

# df = df.reindex(np.random.permutation(df.index))

# X = df.iloc[:, 0:16]
# y = df.iloc[:, 17] 
# #df = df.reindex(np.random.permutation(df.index))

# X = X.values
# y = y.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# target_names = ['STAR', 'GALAXY']


# decision_tree = DecisionTreeClassifier(criterion = 'entropy')

# decision_tree.fit(X_train,y_train)

# y_pred = decision_tree.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# cm1 = confusion_matrix(y_test, y_pred)
# decisionT = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=decision_tree.classes_)
# decisionT.plot()
# plt.show()

# # Use 5-fold split
# kf = KFold(5, shuffle=True)
# fold = 1


# # The data is split five ways, for each fold, the 
# # Perceptron is trained, tested and evaluated for accuracy
# for train_index, validate_index in kf.split(X, y):
#     decision_tree.fit(X[train_index],y[train_index])
#     y_test = y[validate_index]
#     y_pred = decision_tree.predict(X[validate_index])

#     print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     fold += 1

# plt.figure()

# plt.subplot(321)
# plt.scatter(df['u'],df['g'], c=y, s=25)

# plt.subplot(322)
# plt.scatter(df['r'],df['i'], c=y, s=25)

# plt.subplot(323)
# plt.scatter(df['u'],df['z'], c=y, s=25)



# #build a multiclass SVM 'ovo' for one-versus-one, and
# #fit the data
# multi_svm = SVC(gamma='scale', decision_function_shape='ovo')  
# multi_svm.fit(X_train,y_train)


# y_pred1 = multi_svm.predict(X_test)

# variety = target_names

# cm = confusion_matrix(y_test, y_pred1)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=variety)
# disp.plot()
# plt.show()

# accuracy = accuracy_score(y_test, y_pred1)
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred1))







# >>>>>>> 8b75d92fd798c731b23c421a9f100be2a1a7ea0b
