# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:06:08 2021

@author: khilesh
"""
# Jesus is great!
import os
os.chidir('C:/Users/khile/Desktop/WD_Python')

import pandas as pd  
import numpy as np

cs2m = pd.read_csv("C:/Users/Dr Vinod/Desktop/DataSets1/cs2m.csv")
cs2m = pd.DataFrame(cs2m)
grades = pd.read_csv("C:/Users/Dr Vinod/Desktop/DataSets1/grades.csv")
grades = pd.DataFrame(grades)

#________________df creation
 
# initialize list of lists 
data = [['Tessy', 10], ['Sajani', 15], ['Maria', 14]]   
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Name', 'Age'])   
# print dataframe. 
df 

# 2 from dict
"""Python code demonstrate creating  
   DataFrame from dict narray / lists  
   By default addresses. 
"""
# intialise data of lists. 
data = {'Name':['Tom', 'John', 'Chris', 'Harold'], 
        'Age':[20, 21, 19, 18]}   
# Create DataFrame 
df = pd.DataFrame(data)   
# Print the output. 
df 

# 3 with indexes
"""Python code demonstrate creating  
  pandas DataFrame with indexed by   
  DataFrame using arrays. 
import pandas as pd 
"""  
# initialise data of lists. 
data = {'Name':['Tom', 'John', 'Chris', 'Harold'], 
        'Marks':[99, 98, 95, 90]}  
# Creates pandas DataFrame. 
df = pd.DataFrame(data, index =['rank1', 
                                'rank2', 
                                'rank3', 
                                'rank4'])   
# print the data 
df

# 4 
"""
Python code demonstrate how to create  
Pandas DataFrame by lists of dicts. 
import pandas as pd 
"""
# Initialise data to lists. 
data = [{'a': 1, 'b': 2, 'c':3}, 
        {'a':10, 'b': 20, 'c': 30}]   
# Creates DataFrame. 
df = pd.DataFrame(data)  
# Print the data 
df 

"""
Python code demonstrate to create 
Pandas DataFrame by passing lists of  
Dictionaries and row indices. 
"""
import pandas as pd  
# Intitialise data of lists  
data = [{'a': 4, 'b': 2, 'c':3}, {'a': 10, 'b': 20, 'c': 30}]   
# Creates padas DataFrame by passing  
# Lists of dictionaries and row index. 
df = pd.DataFrame(data, index =['first', 'second'])   
# Print the data 
df

"""
Another example to create pandas 
DataFrame from lists of dictionaries 
with both row index as well as column index.
"""
# Intitialise lists data. 
data = [{'a': 1, 'b': 2, 'c': 3}, 
        {'d': 5, 'e': 10, 'f': 20}] 
data
   
# With two column indices, values same  
# as dictionary keys 
df1 = pd.DataFrame(data, index =['first', 
                                 'second',
                                 ], 
                   columns =['a', 'b', 'c']) 
df1
# With two column indices with  
# one index with other name 
df2 = pd.DataFrame(data, index =['first', 
                                 'second',
                                 'third'], 
                   columns =['A', 'B', 'C']) 
   
# print for first data frame 
print (df1, "\n") 
   
# Print for second DataFrame. 
print (df2)

list = ['a', 'b', 1]
tuple = ('a', 'b', 1)
list = ['a', 'b', 2]
tuple = ('a', 'b', 2)

# 5

#Two lists can be merged by using list(zip()) function. 
#Now, create the pandas DataFrame by calling pd.DataFrame() function.
import pandas as pd      
# List1  
Name = ['Tessy', 'Maria', 'Sajani', 'Leema']    
# List2  
Age = [25, 30, 26, 22]     
# get the list of tuples from two lists.  
# and merge them by using zip().  
list_of_tuples = list(zip(Name, Age))      
# Assign data to tuples.  
list_of_tuples    
# Converting lists of tuples into  
# pandas Dataframe.  
df = pd.DataFrame(list_of_tuples, 
                  columns = ['Name', 'Age'])       
# Print data.  
df  

# 6
# Python code demonstrate creating 
# Pandas Dataframe from Dicts of series. 
  
import pandas as pd   
# Intialise data to Dicts of series. 
d = {'Maths' : pd.Series([10, 20, 30, 40], 
                       index =['Tom', 'Shone', 'Suby', 'Mithun']), 
      'Python' : pd.Series([10, 20, 30, 40], 
                        index =['Tom', 'Shone', 'Suby', 'Mithun'])}   
# creates Dataframe. 
df = pd.DataFrame(d)  
# print the data. 
df 


