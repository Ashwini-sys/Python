# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:48:51 2021

@author: khilesh
"""
#God is great
import os
os.chdir("C:/Users/khile/Desktop/WD_python")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

data = pd.read_csv("D:/data _science/PYTHON/Cluster_Analysis_python/Wholesale_customers_data.csv")
data = pd.DataFrame(data)

data.head(3)
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns = data.columns)
data_scaled.head(3)

import scipy.cluster.hierarchy as shc
#3 clustres
plt.figure(figsize = (10,7))
plt.title("Dedrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method = 'ward'))

#2 clusters
plt.figure(figsize = (10,7))
plt.title("Dedrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method = 'ward'))
plt.axhline(y = 7, color = "r", linestyle ='--')

#lets make cluster
#hierarchical clustering for 2 clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2, affinity='euclidean', linkage = 'ward')
cluster.fit_predict(data_scaled)
#scatter plot
plt.figure(figsize = (10,7))
plt.scatter(data_scaled['Milk'],data_scaled['Grocery'], c = cluster.labels_)

#K means
#importing needed libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns



df = pd.read_csv("D:/data _science/PYTHON/Cluster_Analysis_python/USArrests.csv")
df = pd.DataFrame(df) 

#bring states name as row name
#---------naming the 1st unnamed column as states
df.rename(columns = {'Unnamed: 0':'States'}, inplace = True)
        
#Setting states column as index
df.set_index('States', inplace = True)
df

# creating X variable from df without states
x = df[['Murder', 'Assault', 'Rape', 'UrbanPop']]

#scaling x variable
scaler = StandardScaler()
x_scaled = scaler.fit_transform( x )

#ploting for optimum nos of clusters
plt.figure(figsize=(10,8))

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Build cluster
#running  kmeans to our optional number of cluster
kmeans = KMeans(n_clusters = 4)
clusters = kmeans.fit_predict(x_scaled)
clusters

#naming clusters 1-4 instead of 0-3 and adding to dataFrame
y_kmeans1 = clusters + 1
cluster = list(y_kmeans1)
df['cluster'] = cluster
df.head()
#states and counts in different cluster1
df[df['cluster']==1]
len(df[df['cluster']==1])
#states and counts in different cluster2
df[df['cluster']==2]
len(df[df['cluster']==2])
#states and counts in different cluster3
df[df['cluster']==3]
len(df[df['cluster']==3])
#states and counts in different cluster4
df[df['cluster']==4]
len(df[df['cluster']==4])

#mean of cluster 1 to 4
kmeans_mean_cluster = pd.DataFrame(round(df.groupby('cluster').mean(),1)) # last 1 is for rounding means upto 1 dec pt
kmeans_mean_cluster

x = df['Murder']
y = df['Assault']
plt.figure(figsize=(12,6))
sns.scatterplot(x, y, hue=y_kmeans1,palette=['green', 'orange','blue','red'],legend= 'full')






































