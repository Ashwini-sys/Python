# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:16:24 2021

@author: khilesh
"""
import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = [1,2,3,4,5,6,7] # days
y = [25,32, 45, 21, 39, 52, 45] #Sales

plt.plot(x, y)

#you can set the plot title, and lables for x and y axes

x = [1,2,3,4,5, 6, 7] #days
y = [25,32, 45, 21, 39, 52, 45] #Sales
plt.plot(x,y, 'r') #red
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales per Day")
plt.show()

#------************ Line chart ***********----------
#-------------colrs--------
x = [1,2,3,4,5,6,7]#days
y = [25,32,45,21,39,52,45] #sales
plt.plot(x,y, 'm') #MAgneta
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title('Sales per day')
plt.show()

x = [1,2,3,4,5,6,7]#days
y = [25,32,45,21,39,52,45] #sales
plt.plot(x,y, 'gD')#green dotted-- , hat ^, filled circle 0, Diamond D
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title('Sales per day')
plt.show()

# ----------multilines plot------------
x = [1,2,3,4,5,6,7]#days
y = [25,32,45,21,39,52,45] #sales
z = [18,30,28, 18, 29, 47, 35]# Sold on Credit
plt.plot(x,y, 'g--')
plt.plot(x,z, 'r')
plt.xlabel("Days")
plt.ylabel('Sales & Credit per Day')
plt.legend(labels= ('Sales', 'credit sales'), loc = 'upper left')
plt.show()

#------------------Sub plot-----------
x = [1,2,3,4,5,6,7]#days
y = [25,32,45,21,39,52,45] #sales
z = [18,30,28, 18, 29, 47, 35]# Sold on Credit
plt.figure(figsize =(6,4))
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(x,y, 'g--')
ax2.plot(x,z, 'r')
plt.show()


#------------------Sub plot with title-----------

x = [1,2,3,4,5,6,7]#days
y = [25,32,45,21,39,52,45] #sales
z = [18,30,28, 18, 29, 47, 35]# Sold on Credit
plt.figure(figsize =(6,4))
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(x,y, 'g--')
ax1.set_title("Sales vs Days")
ax2.plot(x,z, 'r')
ax2.set_title("credit vs Days")
plt.show()

#-----------Grid-------
x = [1,2,3,4,5,6,7]#days
y = [25,32,45,21,39,52,45]#sales
plt.plot(x,y, 'r')#red
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales per Day")
plt.grid(True)
plt.show()

#-----------Grid-------
x = [1,2,3,4,5,6,7]#days
y = [25,32,45,21,39,52,45]#sales
plt.plot(x,y, 'r', lw = 2)#red
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales per Day")
plt.grid(color = 'b', ls ='-.', lw =0.35)
plt.show()

#------------Histogram-----------
fig,ax = plt.subplots(1,1)
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
ax.hist(a, bins =[0,25,50,75,100])
ax.set_title("histogram of result")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('marks')
ax.set_ylabel('no.of students')
plt.show()

#-------------with diff color and bin size 10--------
fig,ax = plt.subplots(1,1)
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
ax.hist(a,bins = [0,10,20,30,40,50,60,70,80,90,100], facecolor = 'm') 
ax.set_title("histogram of result")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('marks')
ax.set_ylabel('no. of students')
plt.show()

#----------bin size = 20----------#
fig,ax = plt.subplots(1,1)
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
ax.hist(a,bins = [0,20,40,60,80,100], facecolor = 'mediumspringgreen') 
ax.set_title("histogram of result")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('marks')
ax.set_ylabel('no. of students')
plt.show()

cs2m =pd.read_csv("D:/data _science/Basic_Notes/Manupuation/cs2m.csv")
cs2m = pd.DataFrame(cs2m)
fig,ax = plt.subplots(1, 1)
ax.hist(cs2m.Age, bins = [0,20,40,60,80,100], facecolor = 'deeppink')
ax.set_title("Histogram of Age")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('Age')
ax.set_ylabel('Count ladies')
plt.show()


#----Boxplot-------

import matplotlib.pyplot as plt
import numpy as np

#-------------creating dataset-----------
np.random.seed(10)
data = np.random.normal(100,20,200)
fig = plt.figure(figsize = (10,7))

#creating plot
plt.boxplot(data)
#show plot
plt.show()

#-*************More boxplots*************
data_1 = np.random.normal(100,10,200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]
fig = plt.figure(figsize= (10,7))
#creating axes instance
ax = fig.add_axes([0,0,1,1])#(xaxis, yaxis, width, height)
#creating plot
bp = ax.boxplot(data, vert = False)
plt.show()

#------Age, cs2m -------
Age = cs2m['Age']
#-------making colorful----
props2 = {'boxes': 'red', 'whiskers': 'green', 'medians': 'black', 'caps': 'blue'}
Age.plot.box(color = props2)
#fiilling whole box with red color
Age.plot.box(color=props2, patch_artist = True, vert = True)
#Horizontal plot, by vert = false
Age.plot.box(color=props2, patch_artist = True, vert = False)

cs2m.boxplot(color = props2, patch_artist = True)


#-------*****Pie chart************----------
# creating dataset
cars =  ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES' ]
data = [23, 17, 35, 29, 12, 41]
#creating plot
fig = plt.figure(figsize = (10,7))
plt.pie(data, labels = cars)
plt.show()

#-------pie from df grades---------
grades = pd.read_csv("D:/data _science/Basic_Notes/Manupuation/grades.csv")
grades = pd.DataFrame(grades)
grades.ethnicity.value_counts()
ethnicity = ['Australians', 'Brazilians', 'Americans', 'Chinese', 'Russians']
data = [5,11,20,24,45]
plt.pie(data, labels = ethnicity)

#---------Scatter plot-------
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,100,86,103,87,94,78,77,85,86]
plt.scatter(x, y, c ="blue")
plt.show()

mtcars = pd.read_csv("D:\data _science\PYTHON\Matplot\mtcars.csv")
mtcars = pd.DataFrame(mtcars)
mtcars.info()
plt.scatter(mtcars.hp, mtcars.mpg, c="red",
            linewidths = 2,
            marker ="o",
            edgecolors="k",
            s = 200,
            alpha= 0.8)
plt.xlabel("Horse Power")            
plt.ylabel("Mileage Per Gallan")            
plt.title( 'MPG VS HP')            
plt.grid() 
plt.show()           
            
mtcars.info()            
mtcars.cyl.value_counts()            
fig, ax = plt.subplots()
colors = {4:'red', 6:'green', 8:'blue'}
grouped = mtcars.groupby('cyl')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x= 'hp', y ='mpg', label = key, color=colors[key])
plt.xlabel("Horse Power")            
plt.ylabel("Mileage Per Gallan")            
plt.title( 'MPG VS HP')            
plt.grid() 
plt.show()   


x = mtcars['hp']
y = mtcars['mpg']
colors = {4:'red', 6:'green', 8:'blue'}
plt.scatter(x, y, s = 1.25*mtcars['hp'], alpha=0.8,
            c = mtcars['cyl'].map(colors))
plt.xlabel("Horse Power")            
plt.ylabel("Mileage Per Gallan")            
plt.title( 'MPG VS HP')            
plt.grid() 
lables = {4:'red', 6:'green', 8:'blue'}
plt.legend(loc='upper right')
plt.show()   