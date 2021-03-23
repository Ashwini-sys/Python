
#God is my saviour
import os
os.chdir("C:/Users/khile/Desktop/WD_python")

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


df = pd.read_csv("D:/data _science/PYTHON/K_Nearest_Nabour_Python/Mobile_data.csv")
df.info() # All variables have missing values

# target variable - Price_range 0-4 ; 0 is low and 3 is highest

df.price_range.value_counts()
sum(df.price_range.value_counts()) #2000


# % proportion of each grp
print(df['price_range'].value_counts(normalize=True))
# 0-3 0.25% each

# Variable 2- Battery Power- continous variable
df.battery_power.value_counts()
df.battery_power.describe()
"""count    2000.000000
mean     1238.518500
std       439.418206
min       501.000000
25%       851.750000
50%      1226.000000
75%      1615.250000
max      1998.000000
"""

#____histogram
plt.hist(df.battery_power, bins = 'auto', facecolor = 'red')
plt.xlabel('battery_power')
plt.ylabel('counts')
plt.title('Histogram of battery_power')

#__boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df['battery_power'].plot.box(color=props2, patch_artist = True, vert = False)


# Variable 3- clock_speed- continous variable
df.clock_speed.value_counts()
sum(df.clock_speed.value_counts()) #2000

df.clock_speed.describe()

#____histogram
plt.hist(df.clock_speed, bins = 'auto', facecolor = 'pink')
plt.xlabel('clock_speed')
plt.ylabel('counts')
plt.title('Histogram of clock_speed')

#__boxplot
props2 = dict(boxes = 'pink', whiskers = 'green', medians = 'black', caps = 'red')
df['clock_speed'].plot.box(color=props2, patch_artist = True, vert = False)

# No Outliers !!

# Variable 4- fc- continous variable

df.fc.value_counts()
sum(df.fc.value_counts()) #2000

df.fc.describe()

#____histogram
plt.hist(df.fc, bins = 'auto', facecolor = 'orange')
plt.xlabel('fc')
plt.ylabel('counts')
plt.title('Histogram of fc')

#__boxplot
props2 = dict(boxes = 'orange', whiskers = 'green', medians = 'black', caps = 'red')
df['fc'].plot.box(color=props2, patch_artist = True, vert = False)

# Counting Outliers
iqr = df.fc.describe()['75%'] - df.fc.describe()['25%']
up_lim = df.fc.describe()['75%']+1.5*iqr
len(df.fc[df.fc > up_lim]) # 18


# hard outlier on upper side = Q3 + 3*IQR
hard_up_lim = df.fc.describe()['75%']+ 3 *iqr
len(df.fc[df.fc > hard_up_lim]) # 0


# Outliers Noted!! bt since nt hard, v cn hold for nw,  try to fix later if needed

# Variable 5-  int_memory- continous variable

df.info()
'''
Internal Memory in Gigabytes
'''
df.int_memory.describe()
df.int_memory.value_counts()

sum(df.int_memory.value_counts()) #2000
#histogram
#run in block
plt.hist(df.int_memory, bins= 'auto', facecolor= 'red')
plt.xlabel('int_memory')
plt.ylabel('counts')
plt.title('Histogram of int_memory')

#boxplot #outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['int_memory'].plot.box(color=props2,patch_artist=True,vert=False)
# No outliers


#Variable - 6 , m_dep- ordinal variable
df.info()
'''
Mobile Depth in cm
'''
df.m_dep.describe()
df.m_dep.value_counts()
#histogram
#run in block
plt.hist(df.m_dep, bins= 'auto', facecolor= 'red')
plt.xlabel('m_dep')
plt.ylabel('counts')
plt.title('Histogram of m_dep')

#boxplot #no outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['m_dep'].plot.box(color=props2,patch_artist=True,vert=False)

# Countplot
import seaborn as sns
sns.countplot(x ='m_dep', data = df)


#Color Palette
palette = sns.color_palette("magma") # creating palette
sns.palplot(palette) # drawing palette, select frm it
# rocket, mako,flare, crest, viridis, cubehelix, YlOrBr, Spectral, coolwarm
# magma

sns.countplot(x ='m_dep', data = df,  palette=palette)


# Var 5, mobile_wt , luks numeric
df.info()
df.mobile_wt.value_counts()  #2000
df.mobile_wt.describe()

#____histogram
plt.hist(df.mobile_wt, bins = 'auto', facecolor = 'yellow')
plt.xlabel('mobile_wt')
plt.ylabel('counts')
plt.title('Histogram of mobile_wt')

#__boxplot
props2 = dict(boxes = 'yellow', whiskers = 'green', medians = 'black', caps = 'red')
df['mobile_wt'].plot.box(color=props2, patch_artist = True, vert = False)

# No Outliers !!


#Variable - 8 , n_cores- ordinal variable 
df.info()
'''
Number of cores of a processor
'''
df.n_cores.describe()
df.n_cores.value_counts()
#histogram
#run in block
plt.hist(df.n_cores, bins= 'auto', facecolor= 'red')
plt.xlabel('n_cores')
plt.ylabel('counts')
plt.title('Histogram of n_cores')

#boxplot #no outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['n_cores'].plot.box(color=props2,patch_artist=True,vert=False)


import seaborn as sns
sns.countplot(x ='n_cores', data = df)

## Variable- 9, pc , luks numeric
df.info()
df.pc.value_counts() 
df.pc.describe()

#____histogram
plt.hist(df.pc, bins = 'auto', facecolor = 'Pink')
plt.xlabel('pc')
plt.ylabel('counts')
plt.title('Histogram of pc')

#__boxplot
props2 = dict(boxes = 'Pink', whiskers = 'green', medians = 'black', caps = 'red')
df['pc'].plot.box(color=props2, patch_artist = True, vert = False)

# No outliers


#Variable - 9 , px_height
df.info()
'''
Pixel Resolution Height
'''
df.px_height.describe()
df.px_height.value_counts()
#histogram
#run in block
plt.hist(df.px_height, bins= 'auto', facecolor= 'red')
plt.xlabel('px_height')
plt.ylabel('counts')
plt.title('Histogram of px_height')

#boxplot #outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['px_height'].plot.box(color=props2,patch_artist=True,vert=False)

# Counting Outliers
iqr = df.px_height.describe()['75%'] - df.px_height.describe()['25%']
up_lim = df.px_height.describe()['75%']+1.5*iqr
len(df.px_height[df.px_height > up_lim]) # 2  soft outliers


# hard outlier on upper side = Q3 + 3*IQR
hard_up_lim = df.px_height.describe()['75%']+ 3 *iqr
len(df.px_height[df.px_height > hard_up_lim]) #0
 # no Hard outlier

#Variable - 10 , px_width - numeric variable

df.info()
df.px_width.value_counts() 
df.px_width.describe()

#____histogram
plt.hist(df.px_width, bins = 'auto', facecolor = 'blue')
plt.xlabel('px_width')
plt.ylabel('counts')
plt.title('Histogram of px_width')

#__boxplot
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'red')
df['px_width'].plot.box(color=props2, patch_artist = True, vert = False)

# No Outliers !!


#Variable - 11 , ram- numeric variable
df.info()
'''
Random Access Memory in MegaBytes
'''
df.ram.describe()
df.ram.value_counts()
#histogram
#run in block
plt.hist(df.ram, bins= 'auto', facecolor= 'red')
plt.xlabel('ram')
plt.ylabel('counts')
plt.title('Histogram of ram')

#boxplot #no outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['ram'].plot.box(color=props2,patch_artist=True,vert=False)

# no outliers

#Variable - 12 , sc_h - continous variable
df.info()
'''
Screen Height of mobile in cm
'''
df.sc_h.describe()
df.sc_h.value_counts()
#histogram
#run in block
plt.hist(df.sc_h, bins= 'auto', facecolor= 'red')
plt.xlabel('sc_h')
plt.ylabel('counts')
plt.title('Histogram of sc_h')

#boxplot #no outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['sc_h'].plot.box(color=props2,patch_artist=True,vert=False)

# No outliers

#Variable - 13 , sc_w- contious variable
df.info()
'''
Screen Width of mobile in cm
'''
df.sc_w.describe()
df.sc_w.value_counts()
#histogram
#run in block
plt.hist(df.sc_w, bins= 'auto', facecolor= 'red')
plt.xlabel('sc_w')
plt.ylabel('counts')
plt.title('Histogram of sc_w')

#boxplot #no outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['sc_w'].plot.box(color=props2,patch_artist=True,vert=False)

# many values are 0 in this column  so we need to take a median and replace 0 with that
df.sc_w.median() #5

# fill missing values 

df['sc_w']=df.get('sc_w').replace(0,df.sc_w.median())

# replaced all 0 with median=5

#Variable - 14 , talk_time
df.info()
'''
The longest time that a single battery charge will last when you are on call
'''
df.talk_time.describe()
df.talk_time.value_counts()
#histogram
#run in block
plt.hist(df.talk_time, bins= 'auto', facecolor= 'red')
plt.xlabel('talk_time')
plt.ylabel('counts')
plt.title('Histogram of talk_time')

#boxplot #no outliers
props2 = dict(boxes='red',whiskers='green',medians='black',caps='blue')
df['talk_time'].plot.box(color=props2,patch_artist=True,vert=False)

# no outliers

df.info()
x = df.iloc[:,:-1] #14 Variables
x.info()
y = df.iloc[:,-1]
y

#Splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state = 25,test_size=0.25)

len(x_train) #1500
len(x_test) #500
len(y_train) #1500
len(y_test) #500


#Building Model @ n_neighbors = 13
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13) 
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)

#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)

#Prediction Score
mpm_knn.score(x_test, y_test) #94.8% which is a good score

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict'])

(121+119+122+112)/500 #94.8 same as prediction score; diagonal values in matrix/total

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#try with n_neighbours = any no. for better acuracy
#here we are getting better accurcy by 13 only