# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 08:52:59 2021

@author: khilesh
"""

import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sb
import pandas as pd
from scipy.spatial import distance_matrix
from random import sample
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
#!pip install pyclustertend
from pyclustertend import vat

data = pd.read_csv("D:/data _science/PYTHON/Cluster_Analysis_python/RBIdata.csv")
data = pd.DataFrame(data)
data.head()
#converting unnamed cl name to States
data.rename(columns = {'Unnamed: 0':'States'}, inplace = True)
data.set_index('States', inplace = True)
data
#setting index to states
data= data.set_index('States')#errror  "None of ['States'] are in the columns"
sample = data.sample(frac = 0.29, replace = False, random_state = 123)
len(sample)
sample

#distance matrix
DM = pd.DataFrame(distance_matrix(sample.values, sample.values),index=sample.index, columns=sample.index)
round(DM,2)
#plot of distance matrix
plt.plot(DM)
plt.ylabel("k-distances")
plt.grid(True)
plt.show()

#Visualize distance Matrix
from pyclustertend import vat

vat(sample)

#scaling of data
data_scaled=StandardScaler().fit_transform(data)
data_scaled

plt.figure(figsize=(10,8))

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='random', random_state = 42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss,'bx-')
plt.title('The Elbow Method')
plt.xlabel('No.of clusters')
plt.ylabel('wcss')
plt.show()

#Cluster membership
kmeans = KMeans(n_clusters=4)#just making a clusterin the backend
clusters = kmeans.fit_predict(data_scaled)#fitting the clusters to the dataset
clusters
Final_Clusters=clusters+1
cluster=list(Final_Clusters)
data['cluster']=cluster#Addition of the column to the dataset
data.head()

data[data['cluster']==1]

data[data['cluster']==2]

data[data['cluster']==3]

data[data['cluster']==4]

data.groupby('cluster').mean()

#plot clusters
plt.figure(figsize=(12,6))
sb.scatterplot(data['BirthRate'],data['MortalitityRate'],hue=Final_Clusters, 
               palette = ['green', 'orange', 'blue', 'red'])

#Silhouette Score
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(clusters)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(data_scaled,clusters,metric = 'euclidean')
#picturization of silhouette coefficient
y_ax_lower,y_ax_upper = 0,0
yticks = []

for i , c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[clusters == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),
             c_silhouette_vals,
             height = 1.0,
             edgecolor ='none',
             color = color)
    yticks.append((y_ax_lower + y_ax_upper)/2.)
    y_ax_lower += len(c_silhouette_vals)


silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color = "red",
            linestyle="--")
plt.yticks(yticks, cluster_labels +1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

#avrage silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data_scaled, clusters)
silhouette_avg

sample_silhouette_values = silhouette_samples(data_scaled, clusters)
sample_silhouette_values

from sklearn.metrics import silhouette_score

range_n_clusters = list(range(2,10))
print("Number of clusters from 2 to 9:\n", range_n_clusters)

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(data_scaled)
    centers = clusterer.cluster_centers_
    
    score = silhouette_score(data_scaled, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))























