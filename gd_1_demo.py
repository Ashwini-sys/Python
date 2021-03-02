# Jesus is my Saviour!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

data = pd.read_csv("D:\data _science\PYTHON\Gradient Descent/gd_lr.csv")
data = pd.DataFrame(data)
# Preprocessing Input data

X = data.iloc[:, 0]
X
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
b = 0.75
a = 0.45
L = 0.01  # The learning Rate
epochs = 1 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)

#2iteration
b = 0.73
a = 0.41
L = 0.01  # The learning Rate
epochs = 2 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)

#3iteration
b = 0.70
a = 0.35
L = 0.01  # The learning Rate
epochs = 3 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)

#4iteration
b = 0.67
a = 0.29
L = 0.01  # The learning Rate
epochs = 3 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)

#5iteration
b = 0.65
a = 0.25
L = 0.01  # The learning Rate
epochs = 3 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)

#250iteration
b = 0.65
a = 0.25
L = 0.01  # The learning Rate
epochs = 250 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)

#300th iteration
b = 0.65
a = 0.25
L = 0.01  # The learning Rate
epochs = 300 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)


#500th iteration
b = 0.65
a = 0.25
L = 0.01  # The learning Rate
epochs = 500 # The number of iterations to perform gradient descent

for i in range(epochs): 
    Y_pred = b*X + a  # The current predicted value of Y
    D_b = -sum(X * (Y - Y_pred))  # Derivative wrt b
    D_a = -sum(Y - Y_pred)  # Derivative wrt a
    b = b - L * D_b  # Update b
    a = a - L * D_a  # Update a
    
print (b, a)

