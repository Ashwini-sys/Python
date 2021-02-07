#For second one
#create a folder on desktop.
import os
os.chidir('C:/Users/khile/Desktop/WD_Python')

string = "I am Learning Python"
string
string1 = 'I am Learning Python'
string1
len(string1)
#-1 means last item
string[len(string)-1]
string[-1]
string[-2]
string[-3]
string[-4]
#giving space - 7
string[-7]
string[0:2]
string[3:6]
string[5:8]
string[ :4]
string[17: ]
string[:]
#replacing a by A adding two strings
a= "all is well"
a
b= "A" + a[1:]
b

#Tuple
#if their is parenthesis that means they are tuple
mytuple =(7, 'Lucky', 'Excellent', 5.5)
mytuple
len(mytuple)
mytuple[0]
mytuple[1]
mytuple[2]
mytuple[3]
mytuple[4]
#4-1 = 3rd index is 5.5
mytuple[-1]
mytuple[-2]
mytuple[-3]
mytuple[-4]
#2 excellant will ignored
mytuple[0:2]
mytuple+=("Wonderful",)

mytuple
#Tuple sorting
tuple1 = (11,33,22,44,55)
tuple1
tuple2 = sorted(tuple1)
#sorted according to desc order and giving list
tuple2
isinstance(tuple1, tuple)
isinstance(tuple2, tuple)
isinstance(tuple2, list)
isinstance(a, tuple)
a=(4)
a
b=(4, )
b
isinstance(b, tuple)
isinstance(b, list)
a= "Hello World!"
a
a = ("Hello World!")
a
isinstance(a, tuple)
a = ("Hello World!" ,"Hi")
a
isinstance(a, tuple)
b = ("Hello World!",)
b
isinstance(b, tuple)
k = "Hello World!"
k
m = tuple(k)
m
len(k)
isinstance(k, tuple)
isinstance(m, tuple)
#below we put key : value
pic = {"Bobby": "Dimple", "Sholay": "Hema", "Roja": "Madhoo", "3 Idiot": "Kareena"}
pic
pic["Roja"]
pic["3 Idiot"]
pic["Sholay"]
pic["Bobby"]
pic["Sholay"] = "Jaya"
pic
#create new pair
pic["Dangal"] = "Sana"
pic["Sultan"] = "Anuska"
#new dictionary
pic
pic = {"Bobby": "Dimple","Sholay": "Jaya","Roja": "Madhoo","3 Idiot": "Kareena","Dangal": "Sana","Sultan": "Anushka"}
pic.items()
pic.keys()
pic.values()
pic = {"Bobby": "Dimple", "Sholay": "Hema", "Roja": "Madhoo", "3 Idiot": "Kareena", "Dangal": "Sana"}

pic.popitem()
pic.popitem()

movies = ["Bobby", "Don", "Dangal"]
movies

cinema= ["Bobby", 1974, "Roja", 1990, "3 Idiot", 2008, ["Dimple", "Rishi"], ["Madhoo", "Arvind"], ["kareena", "Amir"]]
print(cinema[2])
print(cinema[4])
print(cinema[7][0])
print(cinema[8][0])
print(cinema[1])
# according to indexing its showing by indexing no. list within list
print(cinema[6][0])
print(cinema[7][1])
print(cinema[8][1])

a= [5, 10, 50, 100]
a
b = a
b
a[0] = 500
b
c= a[:]
c
a[0] = "Awesome"
a
c
set1 = {"Dimple", "Madhoo", "Kareena", "Tina"}
set1
set2 = {11,22,33,22}
#duplicate items will be ignored
set2
set3 = {"Dimple", "tina", 11}
set3
len(set1)
len(set2)
len(set3)
#Union of 2 sets (use pipe sym (|))
set4 = set1|set3
set4
#sets do not have ordres
set3[2]
#List into a set
alist =[11,22,33,22,44]
alist
len(alist)
#uses of sets
aset = set(alist)
#duplicates are ignored
aset
len(aset)
#indexing is possible in List
alist[2]
#sets are not indexed
aset[2]
a= {11, 22,33}
a
b = {12,23,33}
b
#UNION all in a and b
a|b
#Intersection Common in a and b
a&b
#Differance
a = {11,22,33}
a
b = {12, 23,33}
b
#Differencing all in a but not in b
#ignoring duplicates 
#All in a but those are not in b
a-b
#all in b but not in a
b-a
#Symmetrical diff
a
b
#Symetrical differance
#all in a, but not in b, and  all in b, but not in a
a^b
b^a
#union all in a and b
a|b









