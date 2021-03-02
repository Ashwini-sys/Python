# Jesus is my Saviour!
# Jesus is Great!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
df = pd.read_csv('D:/data _science/PYTHON/Linear_Regression_Python/User_Data.csv')
df.info() # 400 obs, User ID, Gender, Age, EstimateSalary, Purchased
# input 
x = df.iloc[:, [2, 3]].values # Id and Gender not taken 
# output 
y = df.iloc[:, 4].values 

#__________________train test data

import sklearn
import sklearn.model_selection
from sklearn.model_selection import train_test_split 

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0) 

#__________standardized data
from sklearn.preprocessing import StandardScaler
 
sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain)  
xtest = sc_x.transform(xtest) 
# for y not needed as it is already in 0 and 1
print (xtrain[0:10, :]) 

#_________encoding text into 0 and 1
iris = pd.read_csv('D:/data _science/PYTHON/Linear_Regression_Python/iris.csv') # file should have column headings & Species with text 
iris.info()
spc = pd.get_dummies(iris.Species, prefix='Species')
spc.head()
iris1 = iris.join(spc) #add new var 'spc' which is having 3 vars inside!
iris1.drop(['Species'], axis=1, inplace=True) # drop the original Species as we do not need that
iris1.info() # see new 3 vars, headings and class/level is also proper!

"""
#________________________ if there are 4 levels in text
# convert them into 0 1 2 3  
# useful when these are in order, for treating them as continuous as applicable in this case!
"""
d = {'Maths' : pd.Series([10, 20, 30, 40], 
                       index =['Tessy', 'Shiny', 'Steffi', 'Michael']), 
      'Performance' : pd.Series(['Poor', 'Good', 'VGood', 'Excellent'], 
                        index =['Tessy', 'Shiny', 'Steffi', 'Michael'])}  
# creates Dataframe. 
df1 = pd.DataFrame(d)  
# print the data. 
df1 
#__________convert grades in numbers
df1.Performance = df1.Performance.map({'Poor': 0, 'Good': 1, 'VGood': 2, 'Excellent': 3 })
df1

#____________________________better treat Performance as continuous as there is an order!
'''
LabelEncoder is used 
for converting categories into numbers
USE FOR ONLY NOMINAL DATA 
'''
d2 = {'Marks' : pd.Series([10, 20, 30, 40]), 
      'Gender': pd.Series(['Male', 'Female', 'Male', 'Female'])}

# creates Dataframe. 
df2 = pd.DataFrame(d2)  
# print the data. 
df2 

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df2['Gender']= le.fit_transform(df2['Gender']) 
df2 #alphabetic order! 

'''
below example is a case
of pure nominal 
'''
d1 = {'zone' : pd.Series([1, 1, 3, 2, 2, 3, 2]), 
      'Performance' : pd.Series([200, 312, 510, 615, 325, 700, 725])}  

# creates Dataframe. 
df2 = pd.DataFrame(d1)  
# print the data. 
df2 

df_encoded = pd.get_dummies(df2.zone, prefix='zone')
df_encoded.head()

df_final = df2.join(df_encoded) #add new var 'df_encoded' which is having 3 vars inside!
df_final
df_final.drop(['zone'], axis=1, inplace=True) # drop the original zone as we do not need that
df_final.info() # see new 3 vars, headings and class/level is also proper!
df_final

'''
Clubbing categories
'''
#__________________clubbing more categories into less

d2 = {'zone' : pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'E', 'E', 'F']), 
      'Score' : pd.Series([200, 312, 510, 615, 325, 700, 725, 321, 834, 513])}

# creates Dataframe. 
df3 = pd.DataFrame(d2)  
# print the data. 
df3 

# keep B, D and F as B
# keep A, C and E as A 

df3['zone']=df3.get('zone').replace('D','B')
df3['zone']=df3.get('zone').replace('F','B')
df3['zone']=df3.get('zone').replace('C','A')
df3['zone']=df3.get('zone').replace('E','A')
df3 # now you have only 2 categories as A and B!

#__________taking only few categories as separate data @@@@@@@@@@@@@@@@
zoneBDF = df3[df3.zone.isin(['B','D', 'F'])]
zoneBDF.head()

zoneAC = df3[df3.zone.isin(['A','C'])]
zoneAC.head()

#__________detect outliers 

from scipy import stats 
#IQR = stats.iqr(data, interpolation = 'midpoint') 

#___________
d3 = {'zone' : pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'E', 'E', 'F']), 
      'Score' : pd.Series([200, 312, 310, 315, 325, 300, 325, 321, 1934, 1813])}

# creates Dataframe. 
df4 = pd.DataFrame(d3)  
# print the data. 
df4
df4.info()

import seaborn as sns
sns.boxplot(df4.Score)

Q1 = np.percentile(df4.Score, 25, interpolation = 'midpoint')  
Q2 = np.percentile(df4.Score, 50, interpolation = 'midpoint')  
Q3 = np.percentile(df4.Score, 75, interpolation = 'midpoint')   
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)   
IQR = Q3 - Q1  
print('Interquartile range is', IQR) # 3.0 #14.0

low_lim = Q1 - 1.5 * IQR 
up_lim = Q3 + 1.5 * IQR # 346
print('low_limit is', low_lim) #290.0
print('up_limit is', up_lim) #346.0

#_______counting outliers
len(df4.Score[df4.Score > 346])#2

#________ which row/s is outliers?    
df4.loc[df4.Score > 346] # 2 rows identified , 8th and 9th 

#__________droping > UL or selecting < 346
df5 = df4[df4.Score < 346.1]
df5 # 8 data points retrieved

df4.drop(df4.Score > 346, axis = 0)
data = df4.Score.drop([df4.Score > 346])
#_______________assigning UL to outliers 
df4.Score[df4.Score > 346] = 346
df4

#_________bar plot
import seaborn as sns

df = sns.load_dataset('tips') 
df.sex.value_counts()
# count plot on single categorical variable
sns.countplot(x ='sex', data = df)

#_______histogram
grades = pd.read_csv('D:/data _science/Basic_Notes/Manupuation/grades.csv')
grades = pd.DataFrame(df)
grades.info()
#_run in block
plt.hist(grades.total, bins = 'auto', facecolor = 'red')
plt.xlabel('total')
plt.ylabel('counts')
plt.title('Histogram of total')
#_run in block

# see the difference...grids..matplotlib
grades.hist('total')

#__boxplot_sns
sns.boxplot(grades.total)
#___boxplot_matplotlib
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
grades['total'].plot.box(color=props2, patch_artist = True, vert = False)

#____________scatter/joint
data = sns.load_dataset("mpg")   
# draw jointplot with 
# kde kind 
sns.jointplot(x = "mpg", y = "acceleration", 
              kind = "reg", data = data)

#________sample random
#___sample as per percentage
df = pd.read_csv('D:/data _science/Basic_Notes/Manupuation/grades.csv')
grades.sample(frac=0.3, replace=False, random_state=123)
#_______sample as per counts
sp = grades.sample(50, random_state = 21)
sp.head()

#_________________________deleting a variable
del grades['section']
grades.shape
grades.info()

# dropping variables....another way
grades_drop = grades.drop(['quiz1', 'quiz2', 'quiz3'], 1) # 1 for columns
grades_drop.info()

#_________missing values
data = pd.read_csv("D:/data _science/PYTHON/Linear_Regression_Python/employees.csv")
data = pd.DataFrame(data)
data.head(25)
data.info() #1000 see null values in First Name, Gender, Senior Management, Team
data.describe() #text will be ommitted
b = pd.notnull(data['Gender']) #only non-missing in Genderb
data[b] #855 cases appeared, 145 missed

#_____dictionary of lists
dict = {'First_Score': [100, 90, np.nan, 97],
        'Second_Score': [30, 45, 56, np.nan],
        'Third_Score':[np.nan, 40, 80, 98]}

df = pd.DataFrame(dict)
#__how many na?
df.info() # out of 4, 1 is na in each column

'''''
if you wish to have 
entire data, according to
complete cases (non-missing cases)
in a particular column
'''
c = pd.notnull(df['First_Score']) #only complete/non-missing in First_Score
c1 =df[c] #3 cases appeared, 1 missed
c1

#___________replace missing with mean (or any number)
df.First_Score.describe() # na will be ignored in calculations
df['First_Score'] = df['First_Score'].fillna(df['First_Score'].mean())
print(df)


