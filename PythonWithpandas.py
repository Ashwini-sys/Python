#God is great
import numpy as np
import pandas as pd
#creating series from list[without argument]

#creating pandas series with default index values
list = ["I", "am", "learning", "python"]
x = pd.Series(list)

#print the series
print (x)
len(x)

list1 = [11,12,13,14,15]
x1= pd.Series(list1)
#print the series
print(x1)
len(x1)

#series with our index
ind = [10,20,30,40,50,60,70]
lst = ['I', 'am', 'lucky', 'enough', 'in', 'learning', 'python']

#create  pandas series with define indexes
x3 = pd.Series(lst, index= ind)  
#print the Series
print(x3)
len(x3)
#back to series
#series to list
x3
lst2 = x3.tolist()
lst2
#Create series from multi list
#multi list

list2 = [['I'], ['am'], ['learning'], ['Python'],['Feeling'], ['Sooper'], ['Dooper'], ['Great!']]

# create pandas Series
x4 = pd.Series  ((i[0] for i in list2))
print(x4) 

#dictionary to series; Ex1
#create a dictionary
di  = {'D' : 10, 'B': 20, 'c': 30}
#Create series
srs = pd.Series(di)
print(srs)

# dictionary to series 
# create a dictionary
dictionary= {'Daniel': 10, 'Joshua' : 20, 'Mary' : 30}
#create series
series =  pd.Series(dictionary)  
print(series)

#series with missing values
#Series with NaN(Not a Number)
#create dictionary
dictionary1 = {'A': 50, 'B' : 10, 'C': 80}
#create a series
series1 = pd.Series(dictionary1, index = ['B', 'C', 'D', 'A'])
print(series1)
#dictionary to Data Frame
#dictionary to df to series
# Creating a dictionary
ditn = {'John': [30, 40, 45, 48],
        'Mary': [48, 44, 48, 35],
        'Sajani': [38, 48, 37, 42],
        'Tessy': [50, 48, 40, 39]}
#coverting it to data frame
df = pd.DataFrame(data=ditn)
df
#DF to series
#2nd column of df to series
#Converting second column i.e. 'Mary' to Series
ser1 = df.iloc[:, 1]
print("\nsecond column as a Series:\n")
print(ser1)
#Checking type
print(type(ser1))

#DF to series
#2nd column of df to series
#Converting second column i.e "mary" to Series
ser1 = df.iloc[:, 1]
print("\nsecond column as a Series:\n")
print(ser1)
#Checking type 
print(type(ser1))
######******** iloc = > index as per location *************
#multiple column of df to series 
#i.e. "MAry" and "Tessy" to series
ser1 = df.iloc[:, 1]
ser2 = df.iloc[:, 3]
print("\nMultiple columns as a Series:\n")
print(ser1)
print()
print(ser2)
ser2
#Checking type
print(type(ser1))
print(type(ser2))

#Series from Array
#numpy array
data = np.array(['Sajani', 'Shone', 'Tessy', 'Julie', 'Mary' ])
#creating series
s = pd.Series(data)
print(s)

#Series from array with index
#numpy array
data = np.array(['Sajani', 'Shone', 'Tessy', 'Julie', 'Mary'])
#creating series
s1 = pd.Series (data, index = [1000, 1001, 1002, 1003, 1004])
print(s1)
# series from array with index
#numpy array
data = np.array (['Sajani', 'Shone', 'Tessy', 'Julie', 'Mary']) 
#creating series
s1 = pd.Series(data, index=[1000, 1001, 1002, 1003, 1004])
print(s1)

###Accessing series elements
#access 1st elements
s
sr1 = s[0]    
sr1

#access the last element
s
sr1 = s[4]
sr1
len(s)
s[len(s)-1]

#Access series 
# access from 2nd to last elements
s
sr2 =   s[1:5]
sr2
len(s)
s[1: len(s)]

#**********Access series*********

#acess 1st and 3rd element
s
sr3 = s[[0,2]]
sr3
# access 1st 3 elements
s
sr3 = s[: 3]
sr3
#access 2nd and fourth element 
s
sr4 = s[1:4]
sr4
#Creating simple array
da = np.array(['g', 'e', 'e', 'k', 's', 'f', 'o', 'r', 'g', 'e', 'e','k', 's'])
k = pd.Series(da)
#retrive the last 10 elements
print(k[-10:])

#--access last 2
s
s5 = s[-2:]
s5

#--see first 3 elements
s.head(3)
s.tail(2)
#---Filtering series
#creating the series
sr = pd.Series([80, 25, 3, 25, 24, 6])
#create the index
index_ =  ['Coca Cola', 'Sprite', 'Coke', 'Fanta', 'Dew', 'ThumbsUp']
#set the index
sr.index = index_

#Print the series
print(sr)

#Filter a value say for Sprite
#filter values
result = sr.filter(regex='Sprite')
print(result)

#filter values
result1 = sr.filter(items = ['Coke', 'Dew'])
#print result1
print(result1)
#Operation
#----arithmatic operations
#creating 2 pandas Series
Series1  = pd.Series([1,2,3,4,5])
Series2 = pd.Series([6,7,8,9,10])

#adding the 2 series
Series3 = Series1+Series2
#Displaying the results
print(Series3)
#Substract the 2 Series
Series4 = Series1- Series2
#Displaying the result
print(Series4)
#multipy the 2 series
Series5 = Series1*Series2
#displaying the result
print(Series5)
#divide the 2 Series
#creating 2 Pandas Series
Series1 = pd.Series([1,2,3,4,5])
Series2 = pd.Series([6,7,8,9,10])
# dividing the 2 Series
Series6 = Series1/Series2
#displaying the result
print(Series6)
#Square root
Series7 = np.sqrt(Series1)
Series7
#Square
Series1.pow(2)
#--natural log
Series8 = np.log(Series1)
Series8
#--log to base 10
Series9 = np.log10(Series1)
Series9

#---------Ranking----------
# creating the series
sr = pd.Series([10,25,3,11,24,6])
#Create the Index
index_=['Coka Cola', 'Sprite', 'Coke', 'Fanta', 'Dew', 'ThumbsUp']
#set the index
sr.index = index_
#print the series
print(sr)

#Assign rank
"""Coke =3, lowest, rank = 1;
ThumbsUp =6, second lowest, rank = 2;
Coca cola = 10, third lowest, rank =3;
Fanta = 11, fourth lowest , rank = 4;
Dew = 24, fifth lowest, rank = 5;
Sprite = 25, highest, rank = 6;"""
result= sr.rank()

#print the result
print(result)

#Sorting
Series10 = sr.sort_values(ascending= True)
Series10

#Sorting high to low
Series11 = sr.sort_values(ascending = False)
Series11

######-------------Missing values-------------------------############
#importing file imp
imp = pd.read_csv("D:/data _science/PYTHON/BasicOfPython/imp.csv")
ser1 = imp.iloc[:,0]
ser1

#missing values
#------checking null values
ser1.isnull()
#-------count NaN's
ser1.isnull().sum()
#-------remove NaN values
ser2 = ser1.dropna(how = 'all')
ser2
#Missing Values
#-----mean of ser1
mean_ser1 = np.mean(ser1)
mean_ser1
#-----fill/replace NaN by mean
ser1_filled = ser1.fillna(mean_ser1)
ser1_filled
#concatenate
#-------concatenate
a = pd.Series (["Sajani", "Tessy", "Maria"])
a
b = pd.Series(["Aliviya", "Jennifer", "Andrea"])
b
ab = pd.concat([a,b],ignore_index= True)
ab
#see the differance ignore_index= False , ignore_index= True 
ab1 = pd.concat([a,b],ignore_index= False)
ab1
