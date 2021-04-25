
#God is my saviour
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

td = pd.read_csv("D:/data _science/PYTHON/SVM_Python/bank-additional-full.csv")
td.info() # All variables have missing values
td.shape #(41188, 21)