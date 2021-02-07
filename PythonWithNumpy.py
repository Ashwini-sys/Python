# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:58:05 2021

@author: khilesh
"""
#Ganpati bappa morya

#--------creating list---------
import numpy as np

#create list

data1=[[1,2,3,4],[5,6,7,8]]
data1

#--------from above list to array -------
arr1 = np.array(data1)  
arr1

#--*** class ***----
#see class of data1 and arr1....double u/s!
data1.__class__  #list
arr1.__class__   #ndarray

#find dimension
arr1.ndim #2

#find shape
arr1.shape #(2,4)

#----Data type------
arr1.dtype  #int32

#array of 15 nos in 3*5 setup
#------------mind spellingof arange!-----------

np.arange(15).reshape(3,5)
a = np.arange(10)
a.__class__  #numpy.ndarray,0 to9

#-----------Replace  few element----------
a[5]
a[5:8]

#rep some no by 12
a[5:8] = 12
a  # 5 6 7 replaced with 12 12 12

##__________Access__________
a = np.arange(15).reshape(3,5)
a
a[0] #first item will appear
a[0,0] # forst item , within that 1st element
a[0,3]
a[0:3]
a[1,4]

"""In 2nd row [why not first?],
as per index, 4th element
mind it indexing starts with0 

"""
#create array
b = np.array([[11,12,13],[14,15,16],[17,18,19]])
b

#--Aceess---
#select 3rd row, what you will type, 2 or 3?
b[2]
b[0][2]
b[0,2]
b[:1]
b[2:]
b[2]
b[2,0:2]
b[:2]
b[:,2]
b[0:2, 0:2]

#-----------array operation---------
c = np.arange(6).reshape(2,3)
c

#------Transpose---------
cT = c.T
cT

r = np.random.randn(2,3)
r
#---------dot product--------
dp = np.dot(cT, r)
dp

#-----square root---------
s = np.arange(5)
s

np.sqrt(s)

#--------------exponential--------------

np.exp(s) # e to the power

#---------Greter then and Equal to----------
a = [1,2,3]
b = [3,2,4]

np.greater(a,b)

np.greater_equal(a,b)

#---------Equal and not Equal than-------
np.equal(a,b)

np.not_equal(a,b)
    
#----------------Mean, sum, standerd deviation-------------

m = np.array([[0,1,2],[3,4,5],[6,7,8]])
m

np.mean(m) #4.0
m.mean() #4.0

m.sum() #36
m.std() #2.582

#-------------------Combine--------------
#combining arrays, concatenate

import numpy as np
a1 = np.array([[1,2],[3,4]])
a1
a2 = np.array([[5,6],[7,8]]) 
a2
adc = np.concatenate((a1,a2), axis = 1)
adc
print(adc)

"""axis =1 [0 stands for rows, 1 stands for column]
adding column wise. rows of a2 will be added to corresponding rows and form
NEW COLUMNS"""

#concatenate
adr = np.concatenate((a1,a2), axis=0)
adr

#---Add----------
#----adding stack----
adsr = np.stack((a1, a2), axis = 0)
adsr  # same as concatenate , Row wise

#--Stack-----
adsc = np.stack((a1, a2), axis =1)
adsc # same as concatenate , Column wise

#--------Append------------*
arr_merged = np.append([[1,2],[3,4]],
                       [[10,20],[30,40]], axis = 0)

arr_merged
#try with axis = 1
arr_merged = np.append([[1,2],[3,4]],
                       [[10,20],[30,40]], axis = 1)

arr_merged

#-------Iterate-------------
#---------Iteration through array
#--------element by element
#1d array
arr = np.array([1, 2, 3])

for x in arr:
    print(x)
    
 #-----2nd array--------
 #-----------going row by row-----------
arr = np.array([[1,2,3],[4,5,6]])
 
for x in arr:
    print(x)
       
 """ going row by row, and Scaler element by element"""

for x in arr:
  for y in x:
      print(y)