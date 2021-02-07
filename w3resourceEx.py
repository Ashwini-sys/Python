# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:36:56 2021

@author: khilesh
"""
#----------Ganpati bappa morya!-----------

"""# 1) Write a Python function to find the Max of three numbers."""

def max_of_two( x, y ):
    if x > y:
        return x
    return y
def max_of_three( x, y, z ):
    return max_of_two( x, max_of_two( y, z ) )
print(max_of_three(3, 6, -5))  #6----what exactly happening here?
#---------------------------------------------------------------------------------
print(max_of_two(3, -6))
#-------------------------------------------------------------------------------------


# max of three numbers
def nehamax(x, y, z):
    if (x>y) and  (x>z):
        return x
    else: y > z
    return y
    return z
print(nehamax(3, 6, -5))

def max_of_three( x, y, z ):
    if x > y and x > z:
            return x
    elif y > x and y > z:
            return y
    else:
        return z
max_of_three(800, 40, 90)


def greater_of_three(x, y, z):
    if x > y:
        return x
    elif y > z:
        return y
    else:
        return z
print(greater_of_three(2,7,12))



"""# 2) Write a Python function to sum all the numbers in a list."""

def sum(numbers):
    total = 0
    for x in numbers:
        total += x
    return total
print(sum((8, 2, 3, 0, 7))) #20 += 8+0 = 8 8+2 = 10.... same for all the values in sum and gives total

#------------------------------------------------------------------------------------

"""# 3) Write a Python function to multiply all the numbers in a list."""
def multiply(numbers):  
    total = 1
    for x in numbers:
        total *= x  
    return total  
print(multiply((8, 2, 3, -1, 7)))#8*1=8, 8*2=16, 16*3=48, 48*-1=-48 -48*7= -336
#-336
#--------------------------------------------------------------------------------

"""# 4) Write a Python program to reverse a string."""
def string_reverse(str1):

    rstr1 = ''
    index = len(str1)
    while index > 0:
        rstr1 += str1[ index - 1 ]
        index = index - 1
    return rstr1
print(string_reverse('1234abcd')) #dcba4321 by indexing it - 1 from each index like 0th to 7th
#-------------------------------------------------------------------------------

"""# 5) Write a Python function to calculate the factorial of a number (a non-negative integer). The function accepts the number as an argument."""
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
n=int(input("Input a number to compute the factiorial : "))# Input a number to compute the factiorial : 4 then press enter in console
print(factorial(n))# 24

#-------------------------------------------------------------------------------
"""# 6) Write a Python function to check whether a number is in a given range"""

def test_range(n):
    if n in range(3,9):
        print( " %s is in the range"%str(n))
    else :
        print("The number is outside the given range.")
test_range(5) # 5 is in the range


def check_within_range (a, b, n):
    if ((a < n < b)|(a > n > b)):
        print("Within Range")
    else:
        print("Outside Range")
check_within_range(10, 3, 12)


# try for num 10 which is not in range 3 to 9
def test_range(n):
    if n in range(3,9):
        print( " %s is in the range"%str(n))
    else :
        print("The number is outside the given range.")
test_range(10) #The number is outside the given range.

#-------------------------------------------------------------------------------
"""7) Write a Python function that accepts a string and calculate the number of upper case letters and lower case letters."""

def string_test(s):
    d={"UPPER_CASE":0, "LOWER_CASE":0}
    for c in s:
        if c.isupper():
           d["UPPER_CASE"]+=1
        elif c.islower():
           d["LOWER_CASE"]+=1
        else:
           pass
    print ("Original String : ", s)
    print ("No. of Upper case characters : ", d["UPPER_CASE"])
    print ("No. of Lower case Characters : ", d["LOWER_CASE"])

string_test('The quick Brown Fox')
#O/P :-  Original String :  The quick Brown Fox
#No. of Upper case characters :  3
#No. of Lower case Characters :  13

#Try with diff string
def string_test(s):
    d={"UPPER_CASE":0, "LOWER_CASE":0}
    for c in s:
        if c.isupper():
           d["UPPER_CASE"]+=1
        elif c.islower():
           d["LOWER_CASE"]+=1
        else:
           pass
    print ("Original String : ", s)
    print ("No. of Upper case characters : ", d["UPPER_CASE"])
    print ("No. of Lower case Characters : ", d["LOWER_CASE"])

string_test('Hi Ashu')
#Original String :  Hi Ashu
#No. of Upper case characters :  2
#No. of Lower case Characters :  4

#-------------------------------------------------------------------------------------

""" # 8) Write a Python function that takes a list and returns a new list with unique elements of the first list."""

def unique_list(l):
  x = []
  for a in l:
    if a not in x:
      x.append(a)
  return x

print(unique_list([1,2,3,3,3,3,4,5])) #[1, 2, 3, 4, 5] unique no.
#-------------------------------------------------------------------------------------

"""# 9) Write a Python function that takes a number as a parameter and check the number is prime or not."""

def test_prime(n):
    if (n==1):
        return False
    elif (n==2):
        return True;
    else:
        for x in range(2,n):
            if(n % x==0): #modulas n=9 9% x ==0 firstly it stors 2 in x then stores 3 in x if it is divisiblr by 2 it gives true , for divisible by 3 false
                return False
        return True             
print(test_prime(9)) #False 9 is not prime no.

#-------------------------------------------------------------------------------------

"""#10) Write a Python program to print the even numbers from a given list."""

def is_even_num(l):
    enum = []
    for n in l:
        if n % 2 == 0:
            enum.append(n)
    return enum
print(is_even_num([1, 2, 3, 4, 5, 6, 7, 8, 9]))#[2, 4, 6, 8] shows the nos divicible by 2 means even no.

#-------------------------------------------------------------------------------------
"""11) Write a Python function to check whether a number is perfect or not."""

def perfect_number(n):
    sum = 0
    for x in range(1, n):
        if n % x == 0:
            sum += x
    return sum == n
print(perfect_number(6))#True #need to understand

"""Example : The first perfect number is 6, because 1, 2, and 3 are 
its proper positive divisors, and 1 + 2 + 3 = 6. Equivalently, the number 6 is
equal to half the sum of all its positive divisors: ( 1 + 2 + 3 + 6 ) / 2 = 6.
The next perfect number is 28 = 1 + 2 + 4 + 7 + 14. This is followed by
the perfect numbers 496 and 8128."""

#-------------------------------------------------------------------------------------

"""# 12) Write a Python function that checks whether a passed string is palindrome or not"""

def isPalindrome(string):
	left_pos = 0
	right_pos = len(string) - 1
	
	while right_pos >= left_pos:
		if not string[left_pos] == string[right_pos]:
			return False
		left_pos += 1
		right_pos -= 1
	return True
print(isPalindrome('aza')) #True#need to understand

#-------------------------------------------------------------------------------------
"""13) Write a Python function that prints out the first n rows of Pascal's triangle."""

def pascal_triangle(n):
   trow = [1]
   y = [0]
   for x in range(max(n,0)):
      print(trow)
      trow=[l+r for l,r in zip(trow+y, y+trow)]
   return n>=1
pascal_triangle(6) ##True#need to understand

#[1]
#[1, 1]
#[1, 2, 1]
#[1, 3, 3, 1]
#[1, 4, 6, 4, 1]
#[1, 5, 10, 10, 5, 1]
#-------------------------------------------------------------------------------------

"""# 14) Write a Python function to check whether a string is a pangram or not."""
# Pangrams are words or sentences containing every letter of the alphabet at least once. A to Z
import string, sys
def ispangram(str1, alphabet=string.ascii_lowercase):
    alphaset = set(alphabet)
    return alphaset <= set(str1.lower())
 
print ( ispangram('The quick brown fox jumps over the lazy dog')) #True
#-------------------------------------------------------------------------------------

""" #15) Write a Python program that accepts a hyphen-separated sequence of words as input and prints the words in a hyphen-separated sequence after sorting them alphabetically."""

items = ["green-red-blue" ]
items=[n for n in input().split('-')]
items.sort()
print('-'.join(items)) # need to ask

def sort_hyp(n):
    for i in n:
        l= n.split('-')
        l.sort()
    print('-'.join(l))
ip = input("Enter hyphenated sentence:")
sort_hyp(ip)

#-------------------------------------------------------------------------------------
""" #16) Write a Python function to create and print a list where the values are square of numbers between 1 and 30 (both included)."""
def printValues():
	l = list()
	for i in range(1,21):
		l.append(i**2)
	print(l)
		
printValues()
#[1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400]

#-------------------------------------------------------------------------------------
""" #17) Write a Python program to make a chain of function decorators (bold, italic, underline etc.) in Python"""
def make_bold(fn):
    def wrapped():
        return "<b>" + fn() + "</b>"
    return wrapped

def make_italic(fn):
    def wrapped():
        return "<i>" + fn() + "</i>"
    return wrapped

def make_underline(fn):
    def wrapped():
        return "<u>" + fn() + "</u>"
    return wrapped
@make_bold
@make_italic
@make_underline
def hello():
    return "hello world"
print(hello()) ## returns "<b><i><u>hello world</u></i></b>" need to understand

#-------------------------------------------------------------------------------------

"""# 18) Write a Python program to execute a string containing Python code."""

mycode = 'print("hello world")'
code = """
def mutiply(x,y):
    return x*y

print('Multiply of 2 and 3 is: ',mutiply(2,3))
"""
exec(mycode)
exec(code)
#hello world
#Multiply of 2 and 3 is:  6
#-------------------------------------------------------------------------------------

"""# 19)  Write a Python program to access a function inside a function."""
def test(a):
        def add(b):
                nonlocal a
                a += 1
                return a+b
        return add
func= test(4)
print(func(4)) #9#need to understand
#-------------------------------------------------------------------------------------

"""20) Write a Python program to detect the number of local variables declared in a function."""

def abc():
    x = 1
    y = 2
    str1= "w3resource"
    print("Python Exercises")

print(abc.__code__.co_nlocals)#3 need to understand

#-------------------------------------------------------------------------------------











