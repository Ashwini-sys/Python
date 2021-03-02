
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:03:55 2021

@author: khilesh
"""
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/data _science/PYTHON/Linear_Regression_Python/HousePrices.csv")
df = pd.DataFrame(df)

df.info()#float64(3), int64(35), object(43)
df.shape#(2073, 81)

df.Property_Sale_Price.describe()#min 34900 and max is 755000 #response vriable

#histogram of Property_Sale_price
plt.hist(df.Property_Sale_Price, bins = 'auto', facecolor = 'red')
plt.xlabel("Propert_Sale_price")
plt.ylabel("Counts")
plt.title("Histogram of Property_Sale_Price")

#Boxplot of Property_Sale_price
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
df['Property_Sale_Price'].plot.box(color = props2, patch_artist = True, vert = False)

#Outliers

Q1 = np.percentile(df.Property_Sale_Price, 25, interpolation = 'midpoint')
Q2 = np.percentile(df.Property_Sale_Price, 50, interpolation = 'midpoint')
Q3 = np.percentile(df.Property_Sale_Price, 75, interpolation = 'midpoint')

print('Q1 25 percentile of the given data is,', Q1)
print('Q1 50 percentile of the given data is,', Q2)
print('Q1 75 percentile of the given data is,', Q3)

IQR = Q3-Q1

print('Interquartile range is', IQR)#84000.0

low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR

print('low_limit is', low_lim) #4000.0
print('up_limit is', up_lim) #340000.0

""" anything q3 +1.5 *IQR is Outlier on higher side"""

#Counting outliers
len(df.Property_Sale_Price[df.Property_Sale_Price > 340000])#78

78/2073   #4 % we will build model on this 78 obs at the end

#Lets remove /select outliers
df1  = df[df.Property_Sale_Price <= 340000]
df1.info()#1990; 2073-1995

2073-1995 #= 78

#Histogram of data df1
plt.hist(df1.Property_Sale_Price, bins = 'auto', facecolor = 'blue')
plt.xlabel("Propert_Sale_price")
plt.ylabel("Counts")
plt.title("Histogram of Property_Sale_Price")

#boxplot of data df1
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'blue')
df1['Property_Sale_Price'].plot.box(color = props2, patch_artist = True, vert = False)

#outliers

Q1 = np.percentile(df1.Property_Sale_Price, 25, interpolation = 'midpoint')
Q2 = np.percentile(df1.Property_Sale_Price, 50, interpolation = 'midpoint')
Q3 = np.percentile(df1.Property_Sale_Price, 75, interpolation = 'midpoint')

print('Q1 25 percentile of the given data is,', Q1)
print('Q1 50 percentile of the given data is,', Q2)
print('Q1 75 percentile of the given data is,', Q3)

IQR = Q3-Q1

print('Interquartile range is', IQR)# 76000.0

low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR

print('low_limit is', low_lim) #15000.00
print('up_limit is', up_lim) #319000.0

len(df1.Property_Sale_Price[df.Property_Sale_Price > 319000.0])#47

47/1995 # .23 means 2.4 %

#assigning UL to outliers  left side is our data points after removing outliers and right side is the value 
df1.Property_Sale_Price[df1.Property_Sale_Price > 319000.0] = 319000

#Histogram of data df1 after assigning to the UL
plt.hist(df1.Property_Sale_Price, bins = 'auto', facecolor = 'green')
plt.xlabel("Propert_Sale_price")
plt.ylabel("Counts")
plt.title("Histogram of Property_Sale_Price")

#boxplot of data df1 their are no outliers now!
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'blue')
df1['Property_Sale_Price'].plot.box(color = props2, patch_artist = True, vert = False)

df1.info()
df1.shape #(1995, 81)

#---lets see some categorical variables
df1.Zone_Class.value_counts()

#Count plot on single categorical variable
import seaborn as sns
sns.countplot(x = 'Zone_Class', data = df1)
#----------------------------------------------------------------------------------------------
""" Better club RM, FV, RH, C as RL_1 """
#----------------------------------------------------------------------------------------------
df1['Zone_Class'] = df1.get('Zone_Class').replace('RM', 'RL_1')
df1['Zone_Class'] = df1.get('Zone_Class').replace('FV', 'RL_1')
df1['Zone_Class'] = df1.get('Zone_Class').replace('RH', 'RL_1')
df1['Zone_Class'] = df1.get('Zone_Class').replace('C (all)', 'RL_1')

sns.countplot(x = 'Zone_Class', data = df1)

df1.Zone_Class.value_counts #now you have only 2 categories as 1092 and 306

#Can Zone_CLass be a good predictor
round (df1.Property_Sale_Price.groupby(df1.Zone_Class).describe(),2)
round (df1.Property_Sale_Price.groupby(df1.Zone_Class).min(),2)

#Independent Sample t test
ZC_RL = df1[df1.Zone_Class == 'RL']
ZC_RL_1 = df1[df1.Zone_Class == 'RL_1']
ZC_RL_1.info()

import scipy
scipy.stats.ttest_ind(ZC_RL.Property_Sale_Price, ZC_RL_1.Property_Sale_Price)

#-------------------------------------------------------------------------------------------
""" Pvalue is 2.5299368625343575e-31 so this is approx zero and less than 0.5
 then its good predictor """

''' so , next we should make dummy variable for Zone_Class'''

#---------------------------------------------------------------------------------------------
#*******Let's see anova********Property_Shape Variable

df1.info()
df1.Property_Shape.value_counts
#Count plot on single categorical variable

sns.countplot(x= 'Property_Shape', data = df1)

import statsmodels.api as sm
from statsmodels.formula.api import ols

mod = ols('Property_Sale_Price ~ Property_Shape', data = df1).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)


#-----------Correlation
df1.info()
crl = df1[['Property_Sale_Price', 'LotArea', 'GrLivArea']]
crl.head()
sns.heatmap(crl.corr())

#----------------------------------------------------------------------------------------------
#Dwell_Type

#histogram of Dwell_Type
plt.hist(df1.Dwell_Type, bins = 'auto', facecolor = 'green')
plt.xlabel("Dwell_Type")
plt.ylabel("Counts")
plt.title("Histogram of Dwell_Type")

#Boxplot 3 outlier
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'blue')
df1['Dwell_Type'].plot.box(color = props2, patch_artist = True, vert = False)

#barplot
sns.countplot(x = 'Dwell_Type', data = df1)

df.Dwell_Type.describe()#how to deal with this variable? can we grouped togethr some categories?is their any loss?

#----------------------------------------------------------------------------------------------------
#LotFrontage

#histogram of LotFrontage
plt.hist(df1.LotFrontage, bins = 'auto', facecolor = 'green')
plt.xlabel("LotFrontage")
plt.ylabel("Counts")
plt.title("Histogram of LotFrontage")

#Boxplot one outlier
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'blue')
df1['LotFrontage'].plot.box(color = props2, patch_artist = True, vert = False)

#outliers

Q1 = np.percentile(df1.LotFrontage, 25, interpolation = 'midpoint')
Q2 = np.percentile(df1.LotFrontage, 50, interpolation = 'midpoint')
Q3 = np.percentile(df1.LotFrontage, 75, interpolation = 'midpoint')

print('Q1 25 percentile of the given data is,', Q1)
print('Q1 50 percentile of the given data is,', Q2)
print('Q1 75 percentile of the given data is,', Q3)

IQR = Q3-Q1

print('Interquartile range is', IQR)#nan #why?

low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR

print('low_limit is', low_lim) #15000.00
print('up_limit is', up_lim) #319000.0

#-------------------------------------------------------------------------------------------------------
#LotArea
df1.info()

#histogram of LotArea
plt.hist(df1.LotArea, bins = 'auto', facecolor = 'green')
plt.xlabel("LotArea")
plt.ylabel("Counts")
plt.title("Histogram of LotArea")

#Boxplot one outlier
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'blue')
df1['LotArea'].plot.box(color = props2, patch_artist = True, vert = False)


sns.countplot(x= 'LotArea', data = df1)

#---------------------------------------------------------------------------------------------------
#Road_Type
#categorical variable
df1.Road_Type.value_counts()

sns.countplot(x = 'Road_Type', data = df1) #more on pave

#---------------------------------------------------------------------------------------------------
#Alley
#categorical variable
df1.Alley.value_counts()

sns.countplot(x = 'Alley', data = df1) #more on Grvl 
#so many values are missing so we can skip this var

#---------------------------------------------------------------------------------------------------
#LandContour
#categorical variable
df1.LandContour.value_counts()

sns.countplot(x = 'LandContour', data = df1) #more on Grvl

""" Better club Bnk, HLS, Low,  as Lvl_2 """

df1['LandContour'] = df1.get('LandContour').replace('Bnk', 'Lvl_2')
df1['LandContour'] = df1.get('LandContour').replace('HLS', 'Lvl_2')
df1['LandContour'] = df1.get('LandContour').replace('Low', 'Lvl_2')


sns.countplot(x = 'LandContour', data = df1)

df1.LandContour.value_counts 
#---------------------------------------------------------------------------------------------------
#Utilities
#categorical variable
df1.Utilities.value_counts()

sns.countplot(x = 'Utilities', data = df1) #more on AllPub
#---------------------------------------------------------------------------------------------------
#LotConfig
#categorical variable
df1.LotConfig.value_counts()

sns.countplot(x = 'LotConfig', data = df1) 

""" Better club Corner, CulDSac, FR2, FR3 as Outside """

df1['LotConfig'] = df1.get('LotConfig').replace('Corner', 'Outside')
df1['LotConfig'] = df1.get('LotConfig').replace('CulDSac', 'Outside')
df1['LotConfig'] = df1.get('LotConfig').replace('FR2', 'Outside')
df1['LotConfig'] = df1.get('LotConfig').replace('FR3', 'Outside')

sns.countplot(x = 'LotConfig', data = df1)

df1.LotConfig.value_counts 
#---------------------------------------------------------------------------------------------------
#LandSlope
df1.LandSlope.value_counts()

sns.countplot(x = 'LandSlope', data = df1) 

""" Better club Mod, Sev, as MOD """

df1['LandSlope'] = df1.get('LandSlope').replace('Mod', 'MOD')
df1['LandSlope'] = df1.get('LandSlope').replace('Sev', 'MOD')


sns.countplot(x = 'LandSlope', data = df1)

df1.LandSlope.value_counts 
df1.info()
#---------------------------------------------------------------------------------------------------
#Neighborhood













