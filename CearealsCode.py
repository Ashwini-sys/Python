# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:22:02 2021

@author: khilesh
"""

import os
os.chdir("C:/Users/khile/Desktop/WD_python")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.read_csv("D:/data _science/Project1stOnCereals/cereals_data (1).csv")
data = pd.DataFrame(data)

#str and dim
data.shape #r dim
data.info()
# need to remove na

data.calories.describe()
data.calories.skew()
data.calories.kurt()
data.calories.std()
data.calories.sem()

#histogram of all variables
data.hist()
#boxplot of all variables
data.boxplot()
########################################
#histogram
data.calories.hist()

#boxplot
data.boxplot('calories', vert = False)

#pairs panel
a =data.iloc[:, [3,4,9,15]]
a
sn.pairplot(a, palette= 'Paired')
###############################################3
#histogram and boxplot
#histogram of all 
data1=data.iloc[:,1:4]
data1.hist()

data2=data.iloc[:,4:8]
data2.hist()

data3=data.iloc[:,8:12]
data3.hist()

data4=data.iloc[:,12:16]
data4.hist()

#boxplot of all variables

data1.boxplot()
data2.boxplot()
data3.boxplot()
data4.boxplot()

del data2['sodium']
data2
data2.boxplot()
#boxplot of sodium
data.boxplot('sodium')

#boxplot of calories and sodium
data21 = data.iloc[:, [3,6]]
plt.boxplot(data21)

del data3['potass']
data3
data3.boxplot()

#boxplot of potass
data.boxplot('potass')

del data4['rating']
data4
data4.boxplot()

#boxplot of potass
data.boxplot('rating')

#boxplot with respect to all mfr
#data.boxplot(by = 'rating')

#boxplot of potass and rating
data5 = data[['potass', 'rating']]
data5.boxplot()

#description of all
data1.describe()
data2.describe()
data3.describe()
data4.describe()
data5.describe()

data.describe()

#skew and kurt of cups
data.cups.skew()
data.cups.kurt()

#ske and kurt of rating
data.rating.skew()
data.rating.kurt()

#ske and kurt of fat
data.fat.skew()
data.fat.kurt()

#-----------more categories------
#creating new variable as ratingcat
## More categories
def set_rating(rat):
    if rat['rating']>90:
        return '10'
    elif rat['rating']>80:
        return '9'
    elif rat['rating']>70: 
        return '8'
    elif rat['rating']>60: 
        return '7'
    elif rat['rating']>50: 
        return '6'
    elif rat['rating']>40: 
        return '5'
    elif rat['rating']>30: 
        return '4'
    elif rat['rating']>20: 
        return '3'
    else:
        return '2'
    
data=data.assign(ratingCat=data.apply(set_rating, axis =1))
print(data.head(3))


#----one continuous variable vs two categorical variables---
#grupby with ratingcat
k = data.groupby(['ratingCat', 'mfr'])
data_ratingCat = k['calories']
data_ratingCat.agg ('mean')
data_ratingCat.agg('describe')

r = data.ratingCat
plt.plot(r)



data.mfr.value_counts()
data.mfr.describe()
sn.countplot(x = 'mfr')
#pairplot
sn.pairplot(data)


#heatmap
data_cor = data.corr()
sn.heatmap(data_cor, 
            xticklabels=data_cor.columns.values,
            yticklabels=data_cor.columns.values)