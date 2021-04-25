# Image Recognition

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

# Train_images, 60,000
train_images.shape# (60000, 28, 28)

# train_labels
len(train_labels)
train_labels

# Test_images, 10,000
test_images.shape

# Test_labels
len(test_labels)
test_labels


# The Neural Network
# The neural Architecture
from keras import models
from keras import layers

network = models.Sequential()

network.add(layers.Dense(512, activation = 'relu', input_shape= (28, 28)))

'''
relu -rectified linear unit f(x) = max(0,x)
The function returns 0 if it recieves any negative  input,
but for any positive value x, it returns that value back
Thus it gives an output that has a range from 0 to infinity '''

network.add(layers.Dense(10, activation = "softmax"))
''' softmax layer, which means it will return an array of 10 probability scores '''

# The Compilation Step
network.compile(loss = 'categorical_crossentropy', # method used for multicalss class target var
               optimizer = 'rmsprop', # type of optimizer like Gradient Descend
               metrics = ['accuracy']) # v want output based on accuracy
 
'''
loss function to measure its performance on training dataset
Optimizer- the mechanizm through which the network will update itself
bsed on the data it sees and its loss function
metrics- Accuracy ie (the fraction of the images  that were correctly 
                      classified) '''


# Reshaping & Scaling  the train_test images
# Preparing the image data
'''
before training, we'll preprocess the data by reshaping it into the shape the 
network expects and scaling it so that all values  are in the [0,1] interval
'''
train_images.shape

train_images= train_images.reshape((60000, 28, 28, 1 ))
train_images.shape

# converting to float32 & Scaling
train_images= train_images.astype("float32") /255
train_images


# Reshaping & Scaling  the test_test images
# Preparing the image data

test_images= test_images.reshape((10000, 28, 28)) # reshaping 
test_images= test_images.astype("float32") /255 # scaling
test_images




# Converting Train_labels as categorical
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
train_labels


# Converting Test_labels as categorical
test_labels = to_categorical(test_labels)
test_labels


# Run the model on train data
# fitting  the model to its train data
network.fit(train_images, train_labels, epochs= 5, batch_size= 128)
# accuracy: 0.9899 epoch 5/5


# Test Data Performance
# Evaluating model on Test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_acc:", test_acc)
 



































