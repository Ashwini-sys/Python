#God is Holy
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing

df = pd.read_csv('D:/data _science/PYTHON/PCA_Python/pca34.csv')
df.info()
df= df.drop('Unnamed: 0', axis=1)
df.info()

#PCA 
pca3_nocs = PCA(n_components=3)
pca3_nocs_comp = pca3_nocs.fit_transform(df)
pca3_nocs_comp #Components

pca3_nocs_egvct = pca3_nocs.components_ #Eigen vectors
print(pca3_nocs_egvct)

pca3_nocs_egvl = pca3_nocs.explained_variance_ #Eigen Values
print(pca3_nocs_egvl)

pca3_nocs.explained_variance_ratio_
# array([0.65149154, 0.289459  , 0.05904946])

np.cumsum(pca3_nocs.explained_variance_ratio_)
#array([0.65149154, 0.94095054, 1.        ])