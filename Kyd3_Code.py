# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:42:23 2021

@author: khilesh
"""
#god is my saviour

import os
os.chdir("C:/Users/khile/Desktop/WD_python")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
 
cs2m =pd.read_csv("D:/data _science/Basic_Notes/Manupuation/cs2m.csv")
cs2m = pd.DataFrame(cs2m)

#file grades
grades = pd.read_csv("D:/data _science/Basic_Notes/Manupuation/grades.csv")
grades = pd.DataFrame(grades)

cs2m.shape
grades.shape

len(grades.final)
len(cs2m.BP)

grades.firstname.unique().shape # count of all names in variable firstname
grades.firstname.unique()# all names in variable firstname

grades['quiz1'].dtype#type of data, its int64
cs2m.info()#all int 64
grades.info()#dtypes: float64(1), int64(17), object(4)

cs2m.describe()
cs2m['Age'].describe()
cs2m.Age.groupby(cs2m.Prgnt).describe()

#------Counts in categorical Variable-----
grades.ethnicity.value_counts()
#counts in categorical variable

grades.final.min()
grades.final.max()
grades.final.sum()
grades.final.skew()
grades.final.std()
grades.final.kurtosis()
round(grades.final.kurt(), 2)

from scipy.stats import sem
grades.final.sem()
#upto 4 decimal
round(grades.final.sem(),4)

cs2m.skew()
grades.std()# only numeric will be considered

#know top 3
cs2m.head(3)
cs2m.head()#default is 5

# know bottom 3
cs2m.tail(3)
cs2m.tail()
#histogram
plt.hist(grades.total)
plt.hist(grades.total, bins ='auto')

plt.hist(grades.total, bins = 'auto', facecolor = 'red')
plt.xlabel('total')
plt.ylabel('counts')
plt.title('Histogram of total')

grades.hist('total')

# Box plot
cs2m.boxplot('BP', vert = False)# vert will change orientation False gives horzontal
BP = cs2m['BP']
props1 = dict(boxes = 'red')
BP.plot.box(color = props1)

BP = cs2m['BP']

props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
BP.plot.box(color = props2)

cs2m['BP'].plot.box(color=props2, patch_artist = True, vert = True)
#patch_artist = filling color
cs2m['BP'].plot.box(color =props2, patch_artist = False, vert = False)

cs2m.boxplot()

props3 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
cs2m.plot.box(color=props3)

cs2m.plot.box(color=props3, patch_artist = True)

#boxplot of all verses prgnt
cs2m.boxplot(by = 'Prgnt')

#boxplot of total versus ethnicity
df = grades[['total', 'ethnicity']]
df.boxplot(by = 'ethnicity')

#boxplot of Age versus Prgnt
kf =cs2m[['Age', 'Prgnt']]
kf.boxplot(by = 'Prgnt')

#matplotlib.pyplot--------boxplot
plt.boxplot(cs2m.Chlstrl, 0, 'rs', 0)
#1st 0 = rectangle; 'rs' is colr for outlier last 0 is for horizontal (1 is for Vertical)

plt.boxplot(cs2m.Chlstrl, 1, 'rs', 0) #1 is for notch

plt.boxplot(cs2m.Chlstrl, 1, 'rs', 1) #1st 1 is for notch last 1 is for vertical

#--$$$$$$------*****DATA MANIPULATION******-------------

#.ix stands for indexing
# 0 =sr_no, 1=id, 2 = lastname, 3 = firstname, 4rth will be neglected
"""---------use iloc replacing to ix in new version of python--------------"""
grades.iloc[:, 0:4].head(3)
#rows only 20 to 22, column 1 to 4

grades.iloc[20:22, 0:4].head() #4rth in index will be ommitted

#rows from 1 to 12th row
cs2m1 = cs2m[0:12]# 12th row (actually 13th)will be omitted
cs2m1
cs2m.head()

#-------------Random Sample----------
#import random
from random import sample

#-----sample as per percentage------
cs2m.sample(frac=0.3, replace=False, random_state=123)

cs2m.sample(frac=0.3, replace=False)#diff set of sample appear

#-----------sample as per counts-----------
sp = cs2m.sample(10, random_state = 21)
sp

#----selecting choiced variables, all rows
#all rows and column 1,3,5
#0 is sr_no, will be ignored
cs2m.iloc[:, [1,3,5]].head(3)

#diff method for data frame selection
a = grades[['quiz1', 'gpa', 'final']]
a.head()

#cs2m.BP.compress((cs2m.BP == 170))#error .compress is not thier now in python
cs2m[cs2m.BP == 170]

#----Selection based on mathemtical argument----
#all rows where BP>140
cs2mBP_140 = cs2m[cs2m.BP > 140]
cs2mBP_140

#all rows where DrugR = 1
cs2mDrugR_1 = cs2m[cs2m.DrugR == 1]
cs2mDrugR_1.head(3)

#all rows where DrugR = 0
cs2mDrugR_1 =cs2m[cs2m.DrugR == 0]
cs2mDrugR_1.head(3)

#Clubbing more categories as one
#3 & 5 of ehtnicity as one group__pd.concat--
grades3 = grades[grades.ethnicity == 3]
grades3.head()

grades5 = grades[grades.ethnicity == 5]
grades5.head()

grades35 = pd.concat([grades3, grades5])
len(grades35.ethnicity)

#-------creation of a new variable------
#-------mathematical logic----where Age is L&H <32

cs2m['AgeLH']= np.where(cs2m['Age']<32, 'L', 'H')
cs2m.head()

#mathematical tratment
cs2m['sqrtBP'] = np.sqrt(cs2m.BP)
cs2m.head()
cs2m.shape

#-----------more categories------

def set_age(row):
    if row['Age'] < 20:
        return 'L'
    elif row ['Age'] >=20 and row['Age']<= 35:
        return'M'
    else:
        return 'H'
    
cs2m = cs2m.assign(AgeLH = cs2m.apply(set_age, axis = 1))    
print(cs2m.head(5))

#-------new variable-------

def set_age(row):
    if row['Age'] < 20:
        return 'L'
    elif row ['Age'] >=20 and row['Age']<= 35:
        return'M'
    else:
        return 'H'
    
cs2m = cs2m.assign(AgeLH = cs2m.apply(set_age, axis = 1))    
print(cs2m.head(5))

#-----remove variable----
import numpy as np
#-----------delecting a variable/s
del cs2m['sqrtBP']
cs2m.shape
cs2m.head()

#dropping variables....another way.......run in block 

cs2m_drop = cs2m.drop(['Age', 'BP', "DrugR"],1)# 1 for columns
cs2m_drop.head()  

#statistics mean & median of Age, indexed-prgnt
#-------like tapply
cs2m.Age.groupby(cs2m.Prgnt).mean()
round(cs2m.Age.groupby(cs2m.Prgnt).mean(),)

#ststistic across a categorical variable
cs2m.Age.groupby(cs2m.Prgnt).median()

#describe Age across pregnt: cs2m
cs2m.Age.groupby(cs2m.Prgnt).describe()

#---------scatter plots--------

plt.scatter(cs2m['Age'], cs2m[['BP']])# as excel , 1st will from X-Axis

#----****Pairs Plot****-------------
import seaborn

seaborn.pairplot(cs2m)# histograms + scatter plots

#lets take only continuous variables
file = cs2m[['Age', 'BP', 'Chlstrl']]
file.shape
seaborn.pairplot(file)

#entire data vs Prgnt
#density plots + scatter plots
seaborn.pairplot(cs2m, hue ='Prgnt')

#---lets play with arguments-----
#all variables
#run in block...awesome plot

seaborn.pairplot(cs2m, hue = 'Prgnt', diag_kind = 'kde',
                 plot_kws={'alpha':0.6, 's':80, 'edgecolor': 'black'})

#----Slected variable
#-----run in block....awesome plot

seaborn.pairplot(cs2m,
                 vars = ['Age', 'BP', 'Chlstrl'],
                 hue = 'AnxtyLH', diag_kind = 'kde',
                 plot_kws={'alpha':0.6, 's':80, 'edgecolor': 'black'},
                 size = 3)

#change the values of alpha andsize = 6
seaborn.pairplot(cs2m,
                 vars = ['Age', 'BP', 'Chlstrl'],
                 hue = 'AnxtyLH', diag_kind = 'kde',
                 plot_kws={'alpha':0.8, 's':80, 'edgecolor': 'black'},
                 size = 6)

#continuous vs one categorical
cs2m.Age.describe()
m = cs2m.groupby(['Prgnt'])
cs2m_Age = m['Age']
cs2m_Age.agg('mean')
cs2m_Age.agg('describe')

#----one continuous variable vs two categorical variables---
#for Age across Prnt and DrugR
k = cs2m.groupby(['Prgnt', 'DrugR'])
cs2m_Age = k['Age']
cs2m_Age.agg ('mean')
cs2m_Age.agg('describe')

#converssions of datatypes
a = cs2m
a.shape
a.info()

#---int64 to category factor---
a['Prgnt'] = a['Prgnt'].astype('category')
a.info()

#---int64 to float (numeric)---changed Age to float
a['Age'] = a['Age'].astype('float')
a.info()

#---int64 to float (numeric)---changed Age to int
a['Age'] = a['Age'].astype('int64')
a.info()

#---IQR and quantiles
stats.iqr(cs2m.Age)
cs2m.Age.quantile(0.25)
cs2m.Age.quantile(0.75)
cs2m.Age.quantile(0.5)

#-----cross tabualtion--
#------------ethnicity vs gender-------
pd.crosstab(grades.ethnicity, grades.gender, margins = True)
#margins = true gives row colums total
pd.crosstab(grades.ethnicity, grades.gender, margins = False)


#---------exporting file----
j =grades.sample(20)
j.head()

#save j at working directory
j.to_csv('j.csv')
#file created at desktop

#all 0s(zeros) in column B to be changed to 2
import pandas as pd
import numpy as np

fr = pd.read_csv("D:/data _science/PYTHON/Matplot/FindReplace.csv")
fr = pd.DataFrame(fr)
fr
#all 0s(zeros) in column B to be changed to 2
fr2 = fr.copy()
fr2["B"] = fr2["B"].replace(0,2)
fr2

#In column C, stella to be replaced by steffi
fr2["D"] =fr2["D"].replace("stella", "steffi")
fr2

#Column names to be A as Marks, B as Section , D as Names
fr2 = fr2.rename(columns ={"A":"Marks", "B":"Section", "D":"Names"})
fr2
