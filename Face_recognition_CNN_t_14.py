#Witnessig the mercy of God, everyday!

# Project on Face Recognition
 
# Specifying the folder where images are present
TrainingImagePath="D:/data _science/PYTHON/Convolutional_NueralNw_Python/Train"
validationImagePath = "D:/data _science/PYTHON/Convolutional_NueralNw_Python/Validation"

from keras.preprocessing.image import ImageDataGenerator
 
#____________base model
train_datagen = ImageDataGenerator(rescale=1./225)
 
test_datagen = ImageDataGenerator(rescale=1./225)
 
# Generating the Training Data
training_set = train_datagen.flow_from_directory(TrainingImagePath,
                                                 target_size=(64, 64),
                                                 batch_size=20,
                                                 class_mode='categorical')
'''Found 139 images belonging to 11 classes.''' 


# Generating the Validation Data
validation_set = test_datagen.flow_from_directory(validationImagePath,
                                                  target_size=(64, 64),
                                                  batch_size=20,
                                                  class_mode='categorical')
'''Found 59 images belonging to 11 classes.'''

'''______________________Create CNN deep learning model'''

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
 
'''Initializing the Convolutional Neural Network'''
classifier= Sequential()
 
''' STEP--1 Convolution
# Adding the first layer of CNN
# we are using the format (64,64,3) because we are using TensorFlow backend
# It means 3 matrix of size (64X64) pixels representing Red,
 Green and Blue components of pixels
'''
classifier.add(Convolution2D(32, kernel_size=(3, 3),
                             input_shape=(64,64,3), activation='relu')) 
'''# STEP--2 MAX Pooling'''
classifier.add(MaxPool2D(pool_size=(2,2)))
'''________________ ADDITIONAL LAYER of CONVOLUTION for better accuracy '''
classifier.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
 
classifier.add(MaxPool2D(pool_size=(2,2)))
'''________________ ADDITIONAL LAYER of CONVOLUTION for better accuracy '''
classifier.add(Convolution2D(64, kernel_size=(3, 3), activation='relu')) 
'''# STEP--3 FLattening'''
classifier.add(Flatten()) 
'''# STEP--4 Fully Connected Neural Network'''
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(11, activation='softmax'))
classifier.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 62, 62, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 31, 31, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 64)        36928     
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                589888    
_________________________________________________________________
dense_1 (Dense)              (None, 11)                715       
=================================================================
Total params: 646,923
Trainable params: 646,923
Non-trainable params: 0
_________________________________________________________________'''
 
'''Compiling the CNN'''

classifier.compile(loss='categorical_crossentropy', 
                   optimizer = 'adam', metrics=["accuracy"])
 
###########################################################
 
# Starting the model training
history = classifier.fit_generator(training_set, epochs=20, 
                         validation_data=validation_set)
'''
Epoch 16/20
7/7 [==============================] - 3s 440ms/step - loss: 0.0539 - accuracy: 0.9780 - 
val_loss: 2.8154 - val_accuracy: 0.4746
Epoch 17/20
7/7 [==============================] - 3s 464ms/step - loss: 0.0346 - accuracy: 1.0000 - 
val_loss: 2.4909 - val_accuracy: 0.5763
Epoch 18/20
7/7 [==============================] - 3s 478ms/step - loss: 0.0137 - accuracy: 0.9972 - 
val_loss: 2.5081 - val_accuracy: 0.6102
Epoch 19/20
7/7 [==============================] - 3s 407ms/step - loss: 0.0157 - accuracy: 1.0000 - 
val_loss: 2.4832 - val_accuracy: 0.6102
Epoch 20/20
7/7 [==============================] - 3s 419ms/step - loss: 0.0076 - accuracy: 1.0000 - 
val_loss: 2.5856 - val_accuracy: 0.5424

'''
 
#Displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#training acc for val acc
plt.plot(epochs, acc, 'bo',label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

#training acc for val loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#___________________Model 2_____________________
# Specifying the folder where images are present
TrainingImagePath="D:/data _science/PYTHON/Convolutional_NueralNw_Python/Train"
validationImagePath = "D:/data _science/PYTHON/Convolutional_NueralNw_Python/Validation"


from keras.preprocessing.image import ImageDataGenerator
 
#Adding parameters in the ImageDatagenerator
train_datagen = ImageDataGenerator(rescale=1./225, shear_range=0.5, 
                                   zoom_range=0.5, width_shift_range=0.2,
                                   height_shift_range=0.2)
 
# No transformations are done on the testing images
test_datagen = ImageDataGenerator(rescale=1./225)
 
# Generating the Training Data
training_set = train_datagen.flow_from_directory(TrainingImagePath,
                                                 target_size=(64, 64),
                                                 batch_size=20,
                                                 class_mode='categorical')
'''Found 139 images belonging to 11 classes.''' 

# Generating the Validation Data
validation_set = test_datagen.flow_from_directory(validationImagePath,
                                                  target_size=(64, 64),
                                                  batch_size=20,
                                                  class_mode='categorical')
'''Found 59 images belonging to 11 classes.'''

'''______________________Create CNN deep learning model'''

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
 
'''Initializing the Convolutional Neural Network'''
classifier= Sequential()
 
'''Convolution
'''
classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(64,64,3), activation='relu'))
 
'''MAX Pooling'''
classifier.add(MaxPool2D(pool_size=(2,2)))
 
'''ADDITIONAL LAYER of CONVOLUTION for better accuracy '''
classifier.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
 
classifier.add(MaxPool2D(pool_size=(2,2)))

'''ADDITIONAL LAYER of CONVOLUTION for better accuracy '''
classifier.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
 
'''FLattening'''
classifier.add(Flatten())
 
'''Fully Connected Neural Network'''
classifier.add(Dense(64, activation='relu')) # hidden layer
 
classifier.add(Dense(11, activation='softmax')) # output layer

classifier.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 62, 62, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 31, 31, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 64)        36928     
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                589888    
_________________________________________________________________
dense_1 (Dense)              (None, 11)                715       
=================================================================
Total params: 646,923
Trainable params: 646,923
Non-trainable params: 0
_________________________________________________________________'''
 
'''Compiler'''

classifier.compile(loss='categorical_crossentropy', 
                   optimizer = 'adam', metrics=["accuracy"])
 
###########################################################
# Starting the model training
history = classifier.fit_generator(training_set, epochs=100, 
                         validation_data=validation_set)
'''
Epoch 97/100
7/7 [==============================] - 3s 425ms/step - loss: 0.2843 - accuracy: 0.9283 - 
val_loss: 0.9457 - val_accuracy: 0.7966
Epoch 98/100
7/7 [==============================] - 3s 430ms/step - loss: 0.3497 - accuracy: 0.8937 - 
val_loss: 0.7466 - val_accuracy: 0.7966
Epoch 99/100
7/7 [==============================] - 3s 431ms/step - loss: 0.3195 - accuracy: 0.9013 - 
val_loss: 0.9642 - val_accuracy: 0.7627
Epoch 100/100
7/7 [==============================] - 3s 434ms/step - loss: 0.2929 - accuracy: 0.9264 - 
val_loss: 0.9903 - val_accuracy: 0.7966
'''
 
#Displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo',label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#___________________Pretrained model

#Instantiating the VGG16 convolutional base
# ImageNet dataset (1.4 million labeled images and 1,000 different classes
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False,
                  input_shape=(150, 150, 3))

conv_base.summary()
'''
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________

'''

#Extracting features using the pretrained convolutional base
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# Specifying the folder where images are present
TrainingImagePath="D:/data _science/PYTHON/Convolutional_NueralNw_Python/Train"
validationImagePath = "D:/data _science/PYTHON/Convolutional_NueralNw_Python/Validation"

datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

'''Note that because generators yield data indefinitely in a loop, you must 
break after every image has been seen once.'''

#Following three codes takes few minutes each as it is converting images into
#arrays
train_features, train_labels = extract_features(TrainingImagePath, 139) #error
'''Found 139 images belonging to 11 classes'''

validation_features, validation_labels = extract_features(validationImagePath, 59)
'''Found 59 images belonging to 11 classes.'''

train_features.shape #139, 4, 4, 512
train_features = np.reshape(train_features, (139, 4*4* 512))
train_features.shape #139, 8192

validation_features.shape
validation_features = np.reshape(validation_features, (59, 4*4* 512))
validation_features.shape #59, 8192

#Preparing the response data
train_labels.shape #139,
from keras.utils import to_categorical
train_labels_cat = to_categorical(train_labels)
train_labels_cat.shape #139, 11

validation_labels.shape #59,
valid_labels_cat = to_categorical(validation_labels)
valid_labels_cat.shape #59, 11

#Defining and training the densely connected classifier
from keras import models
from keras import layers
from keras import optimizers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(11, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

history = model.fit(train_features, train_labels_cat,epochs=30,batch_size=20,
                    validation_data=(validation_features, valid_labels_cat))

'''
Epoch 28/30
7/7 [==============================] - 0s 23ms/step - loss: 0.0186 - acc: 0.9923 - 
val_loss: 0.2962 - val_acc: 0.8814
Epoch 29/30
7/7 [==============================] - 0s 21ms/step - loss: 0.0088 - acc: 1.0000 - 
val_loss: 0.3045 - val_acc: 0.8983
Epoch 30/30
7/7 [==============================] - 0s 20ms/step - loss: 0.0272 - acc: 1.0000 - 
val_loss: 0.2072 - val_acc: 0.8983

'''

#Plotting the results
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

