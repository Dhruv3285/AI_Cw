# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:18:25 2021

@author: james
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

path = "."  #absolute or relative path to the folder containing the file. 
            #"." for current folder

filename_read = os.path.join(path, "Covid Vaccinations.csv")
df = pd.read_csv(filename_read)
print(df[0:5])