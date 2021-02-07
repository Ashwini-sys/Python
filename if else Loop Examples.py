

# To check if a number is greater than 30, but less than or equal to 40, you would use
#something like this:
number=33
if number > 30 and number <= 40:
    print('The number is between 30 and 40')
    
# OR
number=33
if 30 < number <= 40:
    print("The number is between 30 and 40")
    
    
#**********************************************************************************

# To check for the letter Q in uppercase or lowercase:
answer='Q'
if answer == 'Q' or answer == 'q':
    print("you typed a 'Q' ")


#************************************************************************************

# program to calculate store discount
# 10% off for $10 or less, 20% off for more than $10
item_price = float(input('enter the price of the item: '))
if item_price <= 10.0:
    discount = item_price * 0.10
else:
    discount = item_price * 0.20
final_price = item_price - discount
print('You got ', discount, 'off, so your final price was', final_price)


#************************************************************************************


# A soccer team is looking for girls from ages 10 to 12 to play on their team. Write a
# program to ask the user’s age and whether the user is male or female (using “m” or
# “f”). Display a message indicating whether the person is eligible to play on the team.
# Bonus: make the program so that it doesn’t ask for the age unless the user is a girl

gender = input("Are you male or female? ('m' or 'f') ")
if gender == "f":
    age = int(input("What is your age? "))
    if 10 <= age <=12:
        print("You can play on the team.")
    else:
        print("Sorry, you are not the right age.")
else:
    print("Sorry, only girls are allowed on this team.")
    
    
 
#************************************************************************************  
 

    
#You’re on a long car trip and arrive at a gas station. It’s 200 km to the next station.
#Write a program to figure out if you need to buy gas here, or if you can wait for the
#next station.
#The program should ask these questions:
#Bonus: include a 5-liter buffer in your program, in case the fuel gauge isn’t accurate
'''How big is your gas tank, in liters?
■ How full is your tank (in percent—for example, half full = 50)?
■ How many km per liter does your car get?'''


# program to check if you need gas.
# Next station is 200 km away
tank_size = int(input('How big is your tank (liters)? '))
full = int(input('How full is your tank (eg. 50 for half full)? '))
mileage = int(input('What is your gas mileage (km per liter)? '))
range = tank_size * (full / 100) * mileage
print('You can go another', range, 'km.')
print('The next gas station is 200km away.')
if range <= 200:
    print('GET GAS NOW!')
else:
    print('You can wait for the next station.')

#To add a 5-liter buffer, change the line----- range to
range = (tank_size - 5) * (full / 100) * mileage



#************************************************************************************  



#Make a program where the user has to enter a secret password to use the program.
#just a simple one that displays a message like “You’re in!” when your friend enters the
#right password.

password = input(" Enter the password:")
if password == 'Ekta':
    print(' You’re in!')
else:
    print('Sorry, Wrong Password!, Try Again!')

# OR

password = 'Bigsecret'
guess = input(" Enter the password:")
if guess == password:
    print(' You’re in!')
else:
    print('Sorry, Wrong Password!, Try Again!')
    
    


#************************************************************************************  
