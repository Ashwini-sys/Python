# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:01:39 2021

@author: khilesh
"""

###----------------Simple User defined Function
def add_numbers(x,y):
    sum = x + y
    return sum
a = 5
b = 6
print("The sum is", add_numbers(a, b))

def Hello():
    print("Hello! Welcome to Happy learning Club")
    return

Hello()

#-----------------its giving output as formating name and age
def ItsMe(name, age):
    print("Hi! My name is {} and I am {} years old".format(name, age))
    return
ItsMe("Ashu", 23)


#------------------if return is their in code or not still runs 
#-----------------Arguements and parameteres
def evenOdd(x):
    if (x % 2 == 0):
        print("even")
    else:
        print("Odd")
        
9 % 4 

evenOdd(20)
evenOdd(33)

#-----------Math is f(x) = x^4 +1

def f(x):
    return x**4 +1

f(3) #3X3X3X3+1 = 3X27 + 1 = 81+1 = 82

#---------------get_ing-----------
def get_ing(wd):
    print(wd + 'ing')

get_ing("jogg")
get_ing("danc")
get_ing("swim")
get_ing("try")
get_ing("fly")
get_ing("study")


def same_initial(wd1, wd2):
    """Tests if two words start with the same charecter,
    and returns True/False. Case distinction is ignored ."""

    if wd1[0].lower() == wd2[0].lower():#converting first letter to lower case
        return True
    else:
        return False
same_initial('amazing', 'Awesome')#True
same_initial('amazing', 'awesome')#True
same_initial('Wonderful', 'Great')#False


def greetings(name):
    """First line in function is DocString"""
    print("Hello {}".format(name))
    return

greetings("Jancy")

def my_function(friend):
    for x in friend:
        print(x)
        
friend = ["Joy", "Chris", "Mary"] 

my_function(friend)       

#Lambda function adding a specific number

#----------lambda function
#anonymous function

x = lambda a: a+10
print(x(5))
print(x(6))


x = lambda a, b: a * b
print(x(5,6))

x = lambda a, b, c: a + b + c
print(x(5,6,2))

#----program to filter out only even items from list
my_list = [1,5,4,6,8,11,3,12]

new_list = list(filter(lambda x: (x%2 == 0), my_list))

print(new_list)

#------------ multiplying by 2--------------
my_list = [1,5,4,6,8,11,3,12]

new_list = list(map(lambda x: x * 2 , my_list))

print(new_list)















