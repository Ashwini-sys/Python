
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.models import Sequential


# load data
# mnist a set of 60,000 training images , plus 10,000 test images , each
# 28*28 pixel assembled by the nationalInstitute of Standards and Technology
# in the 1980's

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

'''
The images are encodedas Numpy arrays, and the labels are an array of digits, 
ranging from0 to 9. The images and labels have a one-to-one coresspondance
'''

# Size of train and test data
# checking the shape of the data
train_images.shape #  (60000, 28, 28)
train_labels.shape # (60000,)

# checking the shape of the data
test_images.shape # (10000, 28, 28)
test_labels.shape # (10000,)

# Exploring the 1st Image
first_img =train_images[0]
plt.imshow(first_img, cmap= 'gray') # Viewing the image
train_labels[0] # 5


# Reshaping & Scaling  the train_test images

# Train images reshaping
train_images= train_images.reshape((60000, 28, 28, 1 ))

# 1 is depth for b&w it is 1 for bgr it will be 3
train_images.shape

# Train Images converting to float32 & Scaling
train_images= train_images.astype("float32") /255



# test images reshaping
test_images= test_images.reshape((10000, 28, 28, 1 ))
test_images.shape

# test Images converting to float32 & Scaling
test_images= test_images.astype("float32") /255


# Converting to categories : train_labels
# Train_labels converting to categorical
train_labels  # array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

train_labels = to_categorical(train_labels)
train_labels  #  dtype=float32


# Converting to categories : test_labels
# test_labels converting to categorical
test_labels

test_labels = to_categorical(test_labels)
test_labels


# Model Architecture
# Building the network architecture
from keras import models 
from keras import layers

model = models.Sequential()

model.add(layers.Conv2D(32,(3, 3), activation = 'relu',
                        input_shape= (28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2))) 


model.add(layers.Conv2D(64 ,(3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64 ,(3, 3), activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64 , activation = 'relu'))
model.add(layers.Dense(10 , activation = 'softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', # method used for multicalss class target var
               optimizer = 'rmsprop', # type of optimizer like Gradient Descend
               metrics = ['accuracy']) # v want output based on accuracy
 
model.summary()

# Lets run the model
# Feeding the data in the model
history = model.fit(train_images, train_labels, epochs= 5, batch_size= 100,
                    validation_data= (test_images, test_labels))

# Lets see wat happens at 1st conv layer
model.summary()


# Wat happens in 1st Max Pooling
# max_pooling2d (MaxPooling2D) (None, 13, 13, 32)
# by Doing max pooling  with a pool size 2*2 with stride = pool size ie 2
# we get feature map of size 13*13
# Note that in maxpooling we are not estimating any parameter

# Wat happens in 2nd Conv layer?
# conv2d_1 (conv2D) (None, 11, 11, 64)    18496
# ((3*3*32) + 1) *64 {layers} = 18496
# Convoluting input image of 13*13 by 3*3 filter we get 11*11  output feature map  

# Wat happens in 2nd Max Pooling?
# max_pooling2d_1 (MaxPooling2 (None, 5,  5, 64) 0
# the above comment  for maxpooling is valid here also

# Wat happens in 3rd Conv layer?
# conv2d_1 (conv2D) (None, 3, 3, 64)    36928
# ((3*3*64) + 1) *64 {layers} = 36928
# Convoluting input image of 5*5 by 3*3 filter we get 3*3  output feature map  


# The last past: Flattening
# flatten (Flatten) (None, 576)
# 3 * 3* 64 = 576



















