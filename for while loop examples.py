#Write a program to print a multiplication table (a times table). At the start, it should
#ask the user which table to print. The output should look something like this:
#Which multiplication table would you like?
#5

# program to print multiplication table up to 10 ( with FOR Loop)

number = int(input('Which table would you like? '))
print('Here is your table:')
for i in range(1, 11):
    print(number, 'x', i, '=', number * i)

#********************************


# program to print multiplication table using  (while loop)

number = int(input('Which table would you like? '))
print('Here is your table:')
i = 1
while i <= 10:
    print(number, 'times', i, '=', number * i)
i = i + 1

#**********************************

# program to print multiplication table using a user-defined RANGE:
    
number = int(input('Which table would you like? '))
limit = int(input('How high would you like it to go? '))
print('Here is your table:')
for i in range(1, limit + 1):
    print(number, 'times', i, '=', number * i)
    
    
#*****************************************************************************************