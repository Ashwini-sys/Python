import os
os.chdir('D:/Ekta/1 DSP Program/Python/WD_Python')
## ourexported file will appear here


# Necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
from scipy import stats

#***************************************************************************
#Import data : File cereals

data = pd.read_csv("D:/Ekta/1 DSP Program/Python/Data for Practice/cereals.csv")
data = pd.DataFrame(data)


# str & dim of file cereals
data.shape
data.info()



## Checking NA's and Replacing
## checking for null values
data.isnull().sum()

## finding Means of all Vars
mean_data= np.mean(data)
mean_data

## Replacing NA's with Means of Vars
data= data.fillna(mean_data)
data.isnull().sum() ### checking if NA's Replaced
data.info()


#***************************************************************************#**********************************************************
## calories 
data.calories.sample(10)
data.calories.describe()
data.calories.skew()
data.calories.sem()

# Histogram
data.calories.hist()

# Boxplot
data.boxplot('calories', vert = False)


## matplot Histogram
plt.hist(data.calories, bins= 'auto', facecolor= 'red')
plt.xlabel  ('calories')
plt.ylabel  ('counts')
plt.title  ('Histogram of calories')

# matplot Boxplot
plotcolor = dict (boxes = 'red', whiskers= 'green',  medians= 'black',  caps= 'blue' ) 
data.calories.plot.box (color = plotcolor,patch_artist = True, vert= False)

#***************************************************************************#**************************************************************

# mfr
data.info()
data.mfr.value_counts()## table of categorical data
data.mfr.describe()

# Boxplot
s.countplot(x= 'mfr', data = data)


## Pie Chart
mfr= [ 'N' , 'Q', ' K ', ' R' , 'G' ,'P',  "A"]
data1= [6, 8, 23, 8, 22, 9, 1]
plt.pie(data1, labels = mfr)

#***************************************************************************#**************************************************************

# type
data.info()
data.type.value_counts()
data.type.describe()

# Boxplot
s.countplot(x= 'type' , data=data)

# Boxplot of all Vars versus type
data.boxplot(by='type',patch_artist = True)


#******************************************************************************

## protein
data.info()
data.protein.sample(10)
data.protein.describe()
data.protein.skew()
data.protein.sem()

# Histogram
data.protein.hist()

# Boxplot
data.boxplot('protein', vert = False)


#********************************************************************************

## fat
data.info()
data.fat.sample(10)
data.fat.describe()
data.fat.skew()
data.fat.sem()

# Histogram
data.fat.hist()

# Boxplot
data.boxplot('fat', vert = False)


#********************************************************************************

## sodium
data.info()
data.sodium.sample(10)
data.sodium.describe()
data.sodium.skew()
data.sodium.sem()

# Histogram 
data.sodium.hist()

# Boxplot
data.boxplot('sodium', vert = False)


#********************************************************************************

## fiber
data.info()
data.fiber.sample(10)
data.fiber.describe()
data.fiber.skew()
data.fiber.sem()

# Histogram
data.fiber.hist()

# Boxplot
data.boxplot('fiber', vert = False)


#********************************************************************************

## carbo
data.info()
data.carbo.sample(10)
data.carbo.describe()
data.carbo.skew()
data.carbo.sem()

# Histogram
data.carbo.hist()

# Boxplot
data.boxplot('carbo', vert = False)


#********************************************************************************

## sugars
data.info()
data.sugars.sample(10)
data.sugars.describe()
data.sugars.skew()
data.sugars.sem()

# Histogram
data.sugars.hist()

# Boxplot
data.boxplot('sugars', vert = False)


#********************************************************************************


## potass
data.info()
data.potass.sample(10)
data.potass.describe()
data.potass.skew()
data.potass.sem()

# Histogram
data.potass.hist()

# Boxplot
data.boxplot('potass', vert = False)


#********************************************************************************

## vitamins
data.info()
data.vitamins.sample(10)
data.vitamins.describe()
data.vitamins.skew()
data.vitamins.sem()

# Histogram
data.vitamins.hist()

# Boxplot
data.boxplot('vitamins', vert = False)


#********************************************************************************

## shelf
data.info()
data.shelf.sample(10)
data.shelf.describe()
data.shelf.skew()
data.shelf.sem()

# Histogram
data.shelf.hist()

# Boxplot
data.boxplot('shelf', vert = False)


#********************************************************************************

## weight
data.info()
data.weight.sample(10)
data.weight.describe()
data.weight.skew()
data.weight.sem()

# Histogram
data.weight.hist()

# Boxplot
data.boxplot('weight', vert = False)


#********************************************************************************

## cups
data.info()
data.cups.sample(10)
data.cups.describe()
data.cups.skew()
data.cups.sem()

# Histogram
data.cups.hist()

# Boxplot
data.boxplot('cups', vert = False)


#********************************************************************************

## rating
data.info()
data.rating.sample(10)
data.rating.describe()
data.rating.skew()
data.rating.sem()

# Histogram
data.rating.hist()

# Boxplot
data.boxplot('rating', vert = False)


#********************************************************************************
## corr test for whole data
data.corr()


# Pairpanels 
## creating a Subset of vars required for pairpanels
data.info()
a= data.iloc[:,[3,4,9,15]] ## creating Subset
a 
#   OR
 
pp= data[['calories' , 'rating' , 'sugars',  'protein', 'mfr', 'sodium' ]]
pp

## Pair Panels
s.pairplot(a, palette='spring_r') ## few vars
s.pairplot(pp, hue= 'mfr' , palette='spring_r') ## grouped by category mfr
s.pairplot(data, palette= 'Spring_r') ## whole data

#********************************************************************************

# Simple ScatterPlot
# ( 2 numerics)
plt.scatter(data.calories, data.rating, c='blue')

#*************************************


# Scatter Plot of calories/ rating/ mfr
# ( 2 category, 1 numeric)
data.mfr.value_counts()
fig, ax = plt. subplots( 1 ,1)
colors= {'K' : 'red', 'G' : 'green' , 'P': 'blue', 'Q' : 'yellow',  'R': 'pink',
         'N': 'orange', 'A' : 'purple'}
grouped= data.groupby('mfr')
for key, group in grouped   :
    group.plot(ax=ax, kind= 'scatter', x= 'calories', y= 'rating', label=key, color= colors[key] )
plt.xlabel('calories')
plt.ylabel('rating')
plt.title('Rating vs Calorie')
plt.grid()
plt.show()



#*************************************

# Scatter Plot of calories/ rating/ mfr-   rating sizing 
# ( 2 category, 1 numeric)
x = data[ 'calories' ]
y = data [ 'rating' ]
colors= {'K' : 'red', 'G' : 'green' , 'P': 'blue', 'Q' : 'yellow',  'R': 'pink',
         'N': 'orange', 'A' : 'purple'}
plt .scatter(x, y, s = 3.25*data['rating'], alpha= 0.8,
             c= data['mfr'].map(colors)
             )
plt.xlabel ( "calories" )
plt.ylabel ( "rating" )
plt.title ( 'Rating vs Calorie' )
plt.grid()

#******************************************************************************

# Strip plot
# ( 1 category, 1 numeric)                
s.stripplot (x="mfr", y = "calories",  data= data )
# ( 2 category, 1 numeric)
s.stripplot (x="mfr", y = "calories", hue= 'type' , palette="Set2",  data= data )

#******************************************************************************

## violin Plot ( 1 category, 1 numeric)
s.violinplot ( x ="mfr", y = "calories", data = data)

#******************************************************************************

## Relationship Plot
# ( 2 category, 2 numeric) 	
s.relplot(
    data = data ,
	    x = 'calories',
	    y = 'rating',
	    col= 'type',
	    hue= 'mfr', ## no. of plots
	    style= 'mfr'
	    )

#******************************************************************************














