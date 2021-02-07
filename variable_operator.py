# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:28:34 2021

@author: khilesh
"""
#----------Variable operator-------
#define structure in casting
x= str(65)
y = int(55)
z = float(38)
x
y
z
#checking type
x = 85
y = "Sajani"
print(type(x))
print(type(y))

#---arithmatic operator
#add
a = 9
b = 4
add = a+b
add
#sub
sub = a-b
sub
#mul
mul = a*b
mul
#div
div1 = a/b
div1

#modules
#-----modulo of both nos
mod = a%b
mod #9%4 = 1 -remainder is 1

#Quitient
quo = a//b
quo # 9//4 2(4 into 2 is 8, quotient = 2)

#Relational Operators
a = 13
b = 33

#a is grater than b
a>b
print(a>b)

#a is smallar than b
print(a<b)

#equal to
print(a ==b )#flase

#smallar than equal to
print(a <= b)


#-----------Assignment Operator------------
x = 4
x
#+=operator
x+=3 # same as x = x+3, 4+3 = 7
x
#gives same value
x=+3
x

#-= operator
#sub
x-=4
x

#*= operator

x = 3
x *= 3 #same as x = x*3, 3*3=9
x

#/= operator
x = 4
x /= 3
x  #x/3 = 4/3 =1.3

x = 4
x /= 4
x

#%= operator
x =3
x %= 3
x   #3/3= 0

x = 5
x %= 3
x #5/3 = 2

# //= operator
x = 3
x //= 3
x   # same as x = x//3

x = 4
x //= 4
x

# **= operator #explore
x = 4
x **= 4  #same as x = x**4 = 4*4*4*4 = 256 #4^4 is diff
x

#--------------*****And, Or, Not*****-------------------
x = True
y = False
print('x and y is', x and y)
print('x or y is', x or y)
print('not x is', not x)

# with x = 1 and y =0
x = 1
y = 0
print('1 and 0 is', 1 and 0) #1 and 0 is 0
print('1 or 0 is', 1 or 0) #1 or 0 is 1
print('not 1 is', not 1) #not 1 is False
print('not 0 is', not 0) #not 0 is True
print('not 1 is', not 0) #not 1 is True
print('not 0 is', not 1) #not 1 is False

#-----------------------------------------------------------------------
#True = 1  and False = 0
#True X True = True
#False X True = False
#True + True =True
#False + True = True
#-----------------------------------------------------------------------

#------------***********CONDITIONAL STATEMENTS***************-----------------------
#God is great
a = 33
b = 200
if b>a:
    print("b is greater than a") # Prinnted b is greater than a
    
#---------------------*********If with condition(naturally)****************------------------
# if the markes are more, print  an appropriate messege
marks = 93
name = "Joy"
if marks > 90:
    print(name, "is Excellent.") #Joy is Excellent.
    
#-------------- if else-----------------------------------------------
#if the marks are less , print an appropriate msg
marks = 83
name = 'Mary'
if marks > 90:
    print(name, "is Excellant.")
else:
    print("No worries this time. Try hard,", name) #No worries this time. Try hard, Mary


#if the marks are more , print an appropriate msg
#try with marks = 93
marks = 93
name = 'Mary'
if marks > 90:
    print(name, "is Excellant.")
else:
    print("No worries this time. Try hard,", name) #Mary is Excellant.
    
    
#--------------------if , elif, else ----------------------------
marks = 83
name = 'Ketty'
if marks > 90:
    print(name, "is Excellant.")
elif marks >= 70 & marks < 90 :
    print("Well Done!", name)
else:
    print("Need to improve", name)#Well Done! Ketty-elif state
    
#try with 46    
marks1 = 46
name = 'Ketty'
if marks1 > 90:
    print(name, "is Excellant.")
elif marks1 <= 70 & marks1 < 90 :
    print("Well Done!", name)
else:
    print("Need to improve", name)#Need to improve Ketty
    
#--------------------if , elif, elif, else ---------------------------------------------
marks = 98
name = "Mike"
if marks >= 90:
    print(name, "You  are Rocking!.")
elif 70 <= marks < 90:
        print("Well Done!", name)
elif 50 <= marks < 70:
    print("Re Do the exam,", name)
else:
    print("Call your parents,",name)#Mike You  are Rocking!.

# try with 42 marks
marks = 42
name = "Mike"
if marks >= 90:
    print(name, "You  are Rocking!.")
elif 70 <= marks < 90:
        print("Well Done!", name)
elif 50 <= marks < 70:
    print("Re Do the exam,", name)
else:
    print("Call your parents,",name)#Call your parents, Mike , coz score is less than equals to 50

#-------------------------Nasted if---------------------------------
age = 38
if (age >= 11):
    print("You are eligible to see the Football match.")
    if(age <= 20 or age >= 60):
        print("Ticket prize is $12")
    else:
        print("Tic kit price is $20")
else:
    print("You are not eligible to buy a ticket.")#You are eligible to see the Football match.
#Tic kit price is $20
##############################################
import random
age=(random.randint(7, 99))
##################################################
#for age = 10
age = 10
if (age >= 11):
    print("You are eligible to see the Football match.")
    if(age <= 20 or age >= 60):
        print("Ticket prize is $12")
    else:
        print("Tic kit price is $20")
else:
    print("You are not eligible to buy a ticket.") #You are not eligible to buy a ticket.



#---------------------For Loop we can use any on the space of "i "---------------------------------------------------------
x = [0,0,0,0,1,1,1,1,1]
for k in x:
    print(k)
#we acess all items in list in i and k
#-----------------------------For loop with else -----------------------------------
for i in x:
    print(i)
else:
    print("No item left")
    
#--------------------------Simple for loop-------------------------------------------

fruits = ["apple", "banana", "cherry"]    
for i in fruits:
    print(i)

#-----------------------for loop with condition ------------------------------------
x = [0,0,0,0,1,1,1,1,1]   
for i in x:
   if i == 0:
       print("No")
   else:
       print("Yes")
       
#----------------------- Simple for loop ----------------------------------------------
#list of numbers
n = [6, 5, 3, 8, 4, 2, 5, 4, 11]  
 
#variable to store the sum
sum = 0

#iterate over the list
for i in n:
    sum = sum + i
print("The sum is ", sum)#The sum is  48 - sum of all in n
#---------------------------------------------------------------------
#Simple for loop
n=[6,5,3,8,4,2,5,4,11]
#variavle to store the sun 
sum=0
for i in n:
    sum=sum+i
    print(sum)
print("The sum of all the element from list is  ",sum)

#----------------------------Simple For loop----------------------------------------

#simple for
subject = ['SQL', 'Excel', 'Python']

for i in subject:
    print("I Like", i)
#I Like SQL
#I Like Excel
#I Like Python

#----------------------------Range---------------------------------------
for x in range(6):
    print(x)#printed 0 to 5

#---------------------------For in range--------------------------------
#program to iterate through a list using indexing
subject = ['SQL', 'Excel', 'Python']

#iterate overthe list using index
for i in range(len(subject)):
    print("I Like", subject[i])
#len(<list name>)gives count of items inside the list,here3
#list items can be accessed by <list name>[<index no.>]

#----------------------------------Simple for loop----------------------------
for x in "banana":
    print(x)
    
#---------------------------------Break Statement---------------------------------
#----------break
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)
    if x == "banana":
        break
#apple
#banana    
##################
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)
    if x == "banana":break
        

#------------------------------------else with Break--------------------------
for x in range(6):
   if x == 3: break
   print(x)
else:
  print("Finally finished!")  # 0 1 2 - skipped all after 3
########################33
for x in range(6):
   if x == 3:
       break
   print(x)
else:
  print("Finally finished!")  # 
#---------------------------continue---------------------------------------------------------
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    if x == "banana":
        continue
    print(x)#apple cherry - skipped banana

#--------------------------range with start parameter-------------------------------
for x in range(2,6):
    print(x)# 2,3,4,5- 6 will not appear

#------------------range with start and specified increment------------------------
for x in range(2, 30, 3):
    print(x)  #shows range from 2 to 30 with the diff of 3 digits
    
#-------------------Nested for loop[more times 'for]--------------------------------

#Nested loop
adj = ["red", "big", "tasty"]    
fruits = ["appple", "banana", "cherry"]

for x in adj:
    for y in fruits:
        print(x,y)#red apple, red banana, red cherry shows all strings in adj for all in fruits 
 
#------------------------The while loop----------------------------------------
#print i as long as i is less than 6

#+= operator
x = 3
x += 5        
x # x+5 , 3+5 = 8

i = 1
while i < 6:
    print(i)
    i += 1 #it gives 1,2,3,4,5 upto 6 meaning not 6
    
    
#----------------------While with break loop----------------------------------------

#Break statement- with the break statement ae can stop the loop even if the while condition is true

i = 1
while i < 6:
    print(i)
    if i == 3:
        break
    i += 1 #1,2,3, after3 it breaks
    
#Continue
#while loop is continue
i = 0
while i < 6:
   i += 1
   if i == 3:
       continue
   print(i)#1,2,4,5,6 by skipping 3 it continues
   
#------------------------else with while-------------------------------------

#--while with else---
i = 1
while i < 6:
    print(i)
    i += 1
else:
    print("i is no longer less than 6")#printed till i is not less than 6
    
    
        
        
        
        
        
        