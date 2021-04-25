# Witnessing God's mercy every second! Thanks God!!

import numpy as np 
import keras
keras.__version__
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
from keras.datasets import imdb

# load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

first_in_train_data = train_data[0]
first_in_train_data
'''
Out[10]: 
[1,
 14,
 22,
 16,.....
 16,
 5345,
 19,
 178,
 32]
'''
import numpy as np

def vectorize_maxwords(maxwords, uniqewords=10000):
    # Create an all-zero matrix of shape (len(maxwords), uniqewords)
    results = np.zeros((len(maxwords), uniqewords)) #1by10k matrix with all elements =0
    for row, column in enumerate(maxwords):
        results[row, column] = 1. # set specific indices of results[row, column] to 1s
    print(results)
    return results
    
# Our vectorized training data
x_train = vectorize_maxwords(train_data)
# Our vectorized test data
x_test = vectorize_maxwords(test_data)

x_train[0]
x_test[0]

first_in_x_train = x_train[0]
first_in_x_train.shape
'''
(10000,)
'''
# vectorize test & trainig LABELS
# float conversion is necessary
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

y_train[0]
y_test[0]

#_____________________________let's build model and optimizer
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configuring the optimizer
from keras import optimizers

# Using losses and metrics
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss=losses.binary_crossentropy,
                    metrics=[metrics.binary_accuracy])

# validation sets: x_train 
x_val = x_train[:10000] # upto 10,000
partial_x_train = x_train[10000:] # above 10,000; 10000th will be here; total = 15k


# validation sets: y_train 
y_val = y_train[:10000] # upto 10,000
partial_y_train = y_train[10000:] # above 10,000; 10000th will be here; total = 15k 


# Execute
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
'''
Epoch 18/20
30/30 [==============================] - 1s 46ms/step - loss: 0.0096 - binary_accuracy: 0.9985
 - val_loss: 0.6257 - val_binary_accuracy: 0.8686
Epoch 19/20
30/30 [==============================] - 1s 40ms/step - loss: 0.0046 - binary_accuracy: 1.0000
 - val_loss: 0.6815 - val_binary_accuracy: 0.8637
Epoch 20/20
30/30 [==============================] - 1s 38ms/step - loss: 0.0061 - binary_accuracy: 0.9992
 - val_loss: 0.7013 - val_binary_accuracy: 0.8672

'''

history_dict = history.history
history_dict #all 20 values are stored
history_dict.keys()
'''
dict_keys
([ dict_keys(['loss', 'binary_accuracy', 
              'val_loss', 'val_binary_accuracy'])])
'''

import matplotlib.pyplot as plt

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#_____accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# loss
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#____________________________ apply on all 25k train and test 
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
results # [0.290592223405838, 0.8838000297546387]
'''
Epoch 1/4
49/49 [==============================] - 2s 17ms/step - loss: 0.5559 - accuracy: 0.7446
Epoch 2/4
49/49 [==============================] - 1s 16ms/step - loss: 0.2753 - accuracy: 0.9063
Epoch 3/4
49/49 [==============================] - 1s 15ms/step - loss: 0.2006 - accuracy: 0.9310
Epoch 4/4
49/49 [==============================] - 1s 15ms/step - loss: 0.1670 - accuracy: 0.9430
'''

model.predict(x_test)
'''
array([[0.19204262],
       [0.9998943 ],
       [0.7866365 ],
       ...,
       [0.09956563],
       [0.05207038],
       [0.5161809 ]], dtype=float32)
'''





