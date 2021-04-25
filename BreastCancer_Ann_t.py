#Even the impossible is possible with God
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
pd.set_option('display.max_column',None)
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
#!pip install tensorflow
#!pip install keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

#Loading the data 
bc = pd.read_csv('D:/data _science/PYTHON/Nueral_Network_python/BreastCancer.csv')
bc.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 699 entries, 0 to 698
Data columns (total 12 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Unnamed: 0       699 non-null    int64  
 1   Id               699 non-null    int64  
 2   Cl.thickness     699 non-null    int64  
 3   Cell.size        699 non-null    int64  
 4   Cell.shape       699 non-null    int64  
 5   Marg.adhesion    699 non-null    int64  
 6   Epith.c.size     699 non-null    int64  
 7   Bare.nuclei      683 non-null    float64
 8   Bl.cromatin      699 non-null    int64  
 9   Normal.nucleoli  699 non-null    int64  
 10  Mitoses          699 non-null    int64  
 11  Class            699 non-null    object 
'''
bc.shape #699, 12
type(bc) 

#Checking Missing values
bc.isnull().sum()
'''
Unnamed: 0          0
Id                  0
Cl.thickness        0
Cell.size           0
Cell.shape          0
Marg.adhesion       0
Epith.c.size        0
Bare.nuclei        16
Bl.cromatin         0
Normal.nucleoli     0
Mitoses             0
Class               0
dtype: int64

'''
#Bare.nuclei    16 missing values

#Removing missing values
bc = bc.dropna()
bc.info() # 683 by 12
'''
Int64Index: 683 entries, 0 to 698
Data columns (total 12 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Unnamed: 0       683 non-null    int64  
 1   Id               683 non-null    int64  
 2   Cl.thickness     683 non-null    int64  
 3   Cell.size        683 non-null    int64  
 4   Cell.shape       683 non-null    int64  
 5   Marg.adhesion    683 non-null    int64  
 6   Epith.c.size     683 non-null    int64  
 7   Bare.nuclei      683 non-null    float64
 8   Bl.cromatin      683 non-null    int64  
 9   Normal.nucleoli  683 non-null    int64  
 10  Mitoses          683 non-null    int64  
 11  Class            683 non-null    object 
dtypes: float64(1), int64(10), object(1)
'''

#Checking Missing values
bc.isnull().sum() #No missing values
'''
Unnamed: 0         0
Id                 0
Cl.thickness       0
Cell.size          0
Cell.shape         0
Marg.adhesion      0
Epith.c.size       0
Bare.nuclei        0
Bl.cromatin        0
Normal.nucleoli    0
Mitoses            0
Class              0
dtype: int64
'''
#Assigning Predictors & Response variable
x = bc.iloc[:,2:11] #Excluding first 2 variables
x.info()
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 683 entries, 0 to 698
Data columns (total 9 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Cl.thickness     683 non-null    int64  
 1   Cell.size        683 non-null    int64  
 2   Cell.shape       683 non-null    int64  
 3   Marg.adhesion    683 non-null    int64  
 4   Epith.c.size     683 non-null    int64  
 5   Bare.nuclei      683 non-null    float64
 6   Bl.cromatin      683 non-null    int64  
 7   Normal.nucleoli  683 non-null    int64  
 8   Mitoses          683 non-null    int64  
dtypes: float64(1), int64(8)
'''
x.shape #(683, 9)
 
y = bc.iloc[:,11]
y.shape #(683,)
y.dtype # dtype('O') object

#Spliting the data into train & test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state = 0,test_size=0.25)

#Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Standardizing the x_train data
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train # array
x_train.shape # 512,9 

#Standardizing the x_test data
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_test # array 
x_test.shape  #171, 9  

#need tensorflow
#_________________________________Deep Learning, sequential model
#Importing libraries
from keras.models import Sequential
from keras.layers import Dense

#Processing train data
y_train[y_train=='malignant'] = 1
y_train[y_train=='benign'] = 0
y_train = np.asarray(y_train).astype('float32') # model needs float
x_train = np.asarray(x_train).astype('float32') # model needs float
x_train.shape #512, 9
y_train.shape #512

#Processing test data
y_test[y_test=='malignant'] = 1
y_test[y_test=='benign'] = 0
y_test = np.asarray(y_test).astype('float32')# model needs float
x_test = np.asarray(x_test).astype('float32') # model needs float
x_test.shape #171, 9
y_test.shape #171

#Model with hiddenlayer activation - 'relu' & outputlayer activation - 'sigmoid' 
bc_seq = Sequential()
bc_seq.add(Dense(9, activation = 'relu'))
bc_seq.add(Dense(30, activation = 'relu'))
bc_seq.add(Dense(20, activation = 'relu')) 
bc_seq.add(Dense(10, activation = 'relu'))
bc_seq.add(Dense(1, activation = 'sigmoid')) 

# compilation 
bc_seq.compile(loss= 'binary_crossentropy', 
               optimizer= 'adam', 
               metrics=['accuracy'])

bc_seq.fit(x_train,y_train, epochs=10) #Default taken 32 as batch size

bc_seq.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_6 (Dense)              (32, 9)                   90 
(9 IVs + 1 bias)*9 neurons = 90       
_________________________________________________________________
dense_7 (Dense)              (32, 30)                  300  
(9 output + 1 bias)*30 = 300     
_________________________________________________________________
dense_8 (Dense)              (32, 20)                  620  
(30 output + 1 bias)*20 = 620     
_________________________________________________________________
dense_9 (Dense)              (32, 10)                  210 
(20 outputs + 1 bias)* 10 = 210     
_________________________________________________________________
dense_10 (Dense)             (32, 1)                   11 
(10 outputs + 1 bias)*1 = 11       
=================================================================
Total params: 1,231
Trainable params: 1,231
Non-trainable params: 0
_________________________________________________________________'''

#Evaluating model on train data
train_loss, train_acc = bc_seq.evaluate(x_train, y_train)
'''
16/16 [==============================] - 0s 865us/step - loss: 0.0792 - accuracy: 0.9727
32*16 = 512; in batch = 32 obs; 32 obs are taken by default 
'''
print('train_acc:', train_acc) 
'''
train_acc: 0.97265625
'''

#Evaluating model on test data
test_loss, test_acc = bc_seq.evaluate(x_test, y_test)
'''
6/6 [==============================] - 0s 1ms/step - loss: 0.1734 - accuracy: 0.9357
6 bathes are craeted randomly 
'''

print('test_acc:', test_acc)
'''
test_acc: 0.9356725215911865
''' 

#________________________________Plotting the Model
import os
os.environ['PATH'] += os.pathsep + 'C:/Program Files/Graphviz/bin'
from keras.utils.vis_utils import plot_model
import pydot
import pydotplus
import graphviz

plot_model(bc_seq, show_shapes=True, show_layer_names=True)

#Experiment sequential model without hidden layer - Got very less accuary
bc_seq1 = Sequential()
bc_seq1.compile(loss= 'binary_crossentropy', 
                optimizer= 'adam', 
                metrics=['accuracy'])
bc_seq1.fit(x_train,y_train, epochs=5)

bc_seq1.summary() #No summary without hidden layers

#Evaluating model on train data
train_loss, train_acc = bc_seq1.evaluate(x_train, y_train)
print('train_acc:', train_acc) 

#Evaluating model on test data
test_loss, test_acc = bc_seq1.evaluate(x_test, y_test)
print('test_acc:', test_acc) 


##@@@@@@@@@@@@@@@@@@@@@@@@@
#________________________________________________________
#Applying Neural Network for classification
from sklearn.neural_network import MLPClassifier
#MLPClassifier - Multi Layer Preceptron Classifier

bc_mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
bc_mlp.fit(x_train,y_train)
bc_mlp.get_params()

'''
'activation': 'relu', #activation{‘identity’ -  f(x) = x, ‘logistic’ - , ‘tanh’ - f(x) = tanh(x), ‘relu’ - f(x) = max(0, x)}, default=’relu’
 'alpha': 0.0001,
 'batch_size': 'auto',
 'beta_1': 0.9,
 'beta_2': 0.999,
 'early_stopping': False,
 'epsilon': 1e-08, #Value for numerical stability in adam. Only used when solver=’adam’
 'hidden_layer_sizes': (30, 30, 30), #30 neuron each in three hidden layers 
 'learning_rate': 'constant', #0.001, #This will be used when solved is sgd
 'learning_rate_init': #0.001, #This will be used when solved is sgd
 'max_fun': 15000, #Only used when solver=’lbfgs’. Maximum number of loss function calls.
 'max_iter': 200, #For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
 'momentum': 0.9,
 'n_iter_no_change': 10,
 'nesterovs_momentum': True,
 'power_t': 0.5,
 'random_state': None,
 'shuffle': True,
 'solver': 'adam', #{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
 'tol': 0.0001,
 'validation_fraction': 0.1,
 'verbose': False,
 'warm_start': False'''
 
bc_mlp.n_layers_ # 5 It means input and output layers are by default included
bc_mlp.hidden_layer_sizes #No of neurons
 
#Prediction
bc_pred_test = bc_mlp.predict(x_test)
bc_pred_test

from sklearn.metrics import confusion_matrix, classification_report
pd.crosstab(y_test,bc_pred_test, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict    benign  malignant  All
Actual                           
benign        103          4  107
malignant       3         61   64
All           106         65  171'''

print(classification_report(y_test, bc_pred_test))
'''
              precision    recall  f1-score   support

      benign       0.97      0.96      0.97       107
   malignant       0.94      0.95      0.95        64

    accuracy                           0.96       171
   macro avg       0.96      0.96      0.96       171
weighted avg       0.96      0.96      0.96       171'''

#ROC Curve
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
bc_pred_prob = bc_mlp.predict_proba(x_test)
bc_pred_prob

fpr, tpr, thresholds = roc_curve(y_test, bc_pred_prob[:,1], pos_label='malignant')
auc = roc_auc_score(y_test, bc_pred_prob[:,1])
print(auc)

#Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve(area=%0.2f)' %auc)
plt.plot([0,1],[0,1], linewidth=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

#________________________________________Logistic REgression
from sklearn.linear_model import LogisticRegression
bc_lr = LogisticRegression(random_state=0)
bc_lr.fit(x_train, y_train)
bc_lr.get_params()
'''
'C': 1.0,
 'class_weight': None,
 'dual': False,
 'fit_intercept': True,
 'intercept_scaling': 1,
 'l1_ratio': None,
 'max_iter': 100,
 'multi_class': 'auto',
 'n_jobs': None,
 'penalty': 'l2',
 'random_state': 0,
 'solver': 'lbfgs',
 'tol': 0.0001,
 'verbose': 0,
 'warm_start': False'''

#Prediction
bc_lr_pred = bc_lr.predict(x_test)
bc_lr_pred

from sklearn.metrics import confusion_matrix, classification_report
pd.crosstab(y_test,bc_lr_pred, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict    benign  malignant  All
Actual                           
benign        103          4  107
malignant       5         59   64
All           108         63  171'''

print(classification_report(y_test, bc_lr_pred))
'''
              precision    recall  f1-score   support

      benign       0.95      0.96      0.96       107
   malignant       0.94      0.92      0.93        64

    accuracy                           0.95       171
   macro avg       0.95      0.94      0.94       171
weighted avg       0.95      0.95      0.95       171'''

#ROC Curve
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
bc_lr_prob = bc_lr.predict_proba(x_test)
bc_lr_prob

fpr, tpr, thresholds = roc_curve(y_test, bc_lr_prob[:,1], pos_label='malignant')
auc = roc_auc_score(y_test, bc_lr_prob[:,1])
print(auc)

#Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve(area=%0.2f)' %auc)
plt.plot([0,1],[0,1], linewidth=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

