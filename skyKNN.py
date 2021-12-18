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





dataset = pd.read_csv('Skyserver_12_30_2019 4_49_58 PM.csv')

numOfRows = len(dataset.index)
print("total number of entries: "+str(numOfRows))

seriesObjQso = dataset.apply(lambda x: True if x['Class'] == "QSO" else False , axis=1)
seriesObjStar = dataset.apply(lambda x: True if x['Class'] == "STAR" else False , axis=1)
seriesObjGalaxy = dataset.apply(lambda x: True if x['Class'] == "GALAXY" else False , axis=1)


# Count number of True in qso
print("number of qs0: "+ str(len(dataset[seriesObjQso == True].index)))
# Count number of True in star
print("number of star: "+  str(len(dataset[seriesObjStar == True].index)))
# Count number of True in galaxy
print("number of galaxy: "+   str(len(dataset[seriesObjGalaxy == True].index)))


## deleting rows where class is qso beacuse we just want to classify between start and universe 
dataset = dataset[dataset.Class != "QSO"]


## we just want to have 10,000 datasets 
dataset = dataset.iloc[84419: , :]

print("total number of entries after deleting qso: "+str(len(dataset.index)))











