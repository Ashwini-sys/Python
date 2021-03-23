
'''
No need to worry abotnormality assumption in decision tree no need to fix outliers
'''
# God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

df_1 = pd.read_csv('D:/data _science/PYTHON/DecisionTreeinPython/HeartDisease.csv')
df_1.info()
df_1.shape # (303, 14)
df_1.isnull().sum() #no missing values

#------------------------------------------------------------------------------
# 13 target , response variable
df_1.target.value_counts()
sum(df_1.target.value_counts())
303-303 #0 no missing values

#remove rows containing missing values after class
import seaborn as sns
sns.countplot(x ='target', data = df_1)

(len(df_1.target)-df_1.target.describe()['count'])/len(df_1.target)
#0.0 missing values

#------------------------------------------------------------------------------------------
#age
df_1.info()
df_1.age.value_counts()
sum(df_1.age.value_counts())

# No. Missing Values    # 820.0
print('No.missing values=',(len(df_1.age)-df_1.target.describe()['count']))
#0.0
# Countplot
import seaborn as sns
sns.countplot(x ='age', data = df_1) 

# % proportion of each grp
print(df_1['age'].value_counts(normalize=True))
df_1.info()
df_1.age.describe() 
#______histogram
plt.hist(df_1.age, bins = 'auto', facecolor = 'red')
plt.xlabel('age')
plt.ylabel('counts')
plt.title('Histogram of age') # looks good, normalized!

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_1['age'].plot.box(color=props2, patch_artist = True, vert = False)
#no outliers

#------------------------------------------------------------------------------------------
#gender
df_1.info()
df_1.gender.describe() 
df_1.gender.value_counts()
sum(df_1.gender.value_counts())
# No. Missing Values    # 820.0
print('No.missing values=',(len(df_1.age)-df_1.gender.describe()['count']))
#0.0
# Countplot
import seaborn as sns
sns.countplot(x ='gender', data = df_1) 

# % proportion of each grp
print(df_1['gender'].value_counts(normalize=True))

#------------------------------------------------------------------------------------------
#chest_pain
df_1.info()
df_1.chest_pain.value_counts()
sum(df_1.chest_pain.value_counts())
df_1.chest_pain.describe() 

# No. Missing Values    # 820.0
print('No.missing values=',(len(df_1.chest_pain)-df_1.gender.describe()['count']))
#0.0
# Countplot
import seaborn as sns
sns.countplot(x ='chest_pain', data = df_1) 

# % proportion of each grp
print(df_1['chest_pain'].value_counts(normalize=True))
#------------------------------------------------------------------------------------------
#rest_bps
df_1.info()
df_1.rest_bps.value_counts()
sum(df_1.rest_bps.value_counts())
df_1.rest_bps.describe() 

# No. Missing Values    # 820.0
print('No.missing values=',(len(df_1.rest_bps)-df_1.gender.describe()['count']))
#0.0
# Countplot
import seaborn as sns
sns.countplot(x ='rest_bps', data = df_1) 

# % proportion of each grp
print(df_1['rest_bps'].value_counts(normalize=True))
#run as block
#______histogram
plt.hist(df_1.rest_bps, bins = 'auto', facecolor = 'blue')
plt.xlabel('rest_bps')
plt.ylabel('counts')
plt.title('Histogram of rest_bps') # looks good, normalized!

#____boxplot
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'red')
df_1['rest_bps'].plot.box(color=props2, patch_artist = True, vert = False)
#some outliers r their
#will think about it latr!
#------------------------------------------------------------------------------------------
#cholestrol
df_1.info()
df_1.cholestrol.value_counts()
sum(df_1.cholestrol.value_counts())
df_1.cholestrol.describe() 

# No. Missing Values    # 820.0
print('No.missing values=',(len(df_1.cholestrol)-df_1.gender.describe()['count']))
#0.0
# % proportion of each grp
print(df_1['cholestrol'].value_counts(normalize=True))
#run as block
#______histogram
plt.hist(df_1.cholestrol, bins = 'auto', facecolor = 'green')
plt.xlabel('cholestrol')
plt.ylabel('counts')
plt.title('Histogram of cholestrol') # looks good, normalized!

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df_1['cholestrol'].plot.box(color=props2, patch_artist = True, vert = False)
#some outliers
#will think about it latr!

#------------------------------------------------------------------------------------------
#fasting_blood_sugar
df_1.info()
df_1.fasting_blood_sugar.value_counts()
sum(df_1.fasting_blood_sugar.value_counts())
df_1.fasting_blood_sugar.describe() 

# Countplot
import seaborn as sns
sns.countplot(x ='fasting_blood_sugar', data = df_1) 
# No. Missing Values    # 820.0
print('No.missing values=',(len(df_1.fasting_blood_sugar)-df_1.fasting_blood_sugar.describe()['count']))
#0.0
# % proportion of each grp
print(df_1['cholestrol'].value_counts(normalize=True))

#------------------------------------------------------------------------------------------
#rest_ecg
df_1.info()
df_1.rest_ecg.value_counts()
sum(df_1.rest_ecg.value_counts())
df_1.rest_ecg.describe() 

# Countplot
import seaborn as sns
sns.countplot(x ='rest_ecg', data = df_1) 
# No. Missing Values
print('No.missing values=',(len(df_1.rest_ecg)-df_1.rest_ecg.describe()['count']))
#No.missing values= 0.0
# % proportion of each grp
print(df_1['rest_ecg'].value_counts(normalize=True))
#------------------------------------------------------------------------------------------
#thalach continuous
df_1.info()
df_1.thalach.value_counts()
sum(df_1.thalach.value_counts())
df_1.thalach.describe() 

# No. Missing Values
print('No.missing values=',(len(df_1.thalach)-df_1.thalach.describe()['count']))
#No.missing values= 0.0
# % proportion of each grp
print(df_1['thalach'].value_counts(normalize=True))
#______histogram
plt.hist(df_1.thalach, bins = 'auto', facecolor = 'yellow')
plt.xlabel('thalach')
plt.ylabel('counts')
plt.title('Histogram of thalach') # looks good, normalized!

#____boxplot
props2 = dict(boxes = 'yellow', whiskers = 'green', medians = 'black', caps = 'red')
df_1['thalach'].plot.box(color=props2, patch_artist = True, vert = False)
#------------------------------------------------------------------------------------------
#exer_angina categorical
df_1.info()
df_1.exer_angina.value_counts()
sum(df_1.exer_angina.value_counts())
df_1.exer_angina.describe() 
# Countplot
import seaborn as sns
sns.countplot(x ='exer_angina', data = df_1) 
# No. Missing Values
print('No.missing values=',(len(df_1.exer_angina)-df_1.exer_angina.describe()['count']))
#No.missing values= 0.0
# % proportion of each grp
print(df_1['exer_angina'].value_counts(normalize=True))

#------------------------------------------------------------------------------------------
#old_peak continuous
df_1.info()
df_1.old_peak.value_counts()
sum(df_1.old_peak.value_counts())
df_1.old_peak.describe() 
# No. Missing Values
print('No.missing values=',(len(df_1.old_peak)-df_1.old_peak.describe()['count']))
#No.missing values= 0.0
# % proportion of each grp
print(df_1['old_peak'].value_counts(normalize=True))

#______histogram
plt.hist(df_1.old_peak, bins = 'auto', facecolor = 'green')
plt.xlabel('old_peak')
plt.ylabel('counts')
plt.title('Histogram of old_peak') # looks good, normalized!

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df_1['old_peak'].plot.box(color=props2, patch_artist = True, vert = False)
#Some outliers are their
#------------------------------------------------------------------------------------------
#slope continuous
df_1.info()
df_1.slope.value_counts()
sum(df_1.slope.value_counts())
df_1.slope.describe() 
# No. Missing Values
print('No.missing values=',(len(df_1.slope)-df_1.slope.describe()['count']))
#No.missing values= 0.0
# % proportion of each grp
print(df_1['slope'].value_counts(normalize=True))
# Countplot
import seaborn as sns
sns.countplot(x ='slope', data = df_1) 

#------------------------------------------------------------------------------------------
#ca continuous
df_1.info()
df_1.ca.value_counts()
sum(df_1.ca.value_counts())
df_1.ca.describe() 
# No. Missing Values
print('No.missing values=',(len(df_1.ca)-df_1.ca.describe()['count']))
#No.missing values= 0.0
# % proportion of each grp
print(df_1['ca'].value_counts(normalize=True))

import seaborn as sns
sns.countplot(x ='ca', data = df_1) 
#______histogram
plt.hist(df_1.ca, bins = 'auto', facecolor = 'green')
plt.xlabel('ca')
plt.ylabel('counts')
plt.title('Histogram of ca') # looks good, normalized!

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df_1['ca'].plot.box(color=props2, patch_artist = True, vert = False)
#Some outliers are their
'''
#Outliers
up_lim = df_1.ca.describe()['75%']+3*iqr
len(df_1.ca[df_1.ca > up_lim])
'''
#------------------------------------------------------------------------------------------
#thalassemia continuous
df_1.info()
df_1.thalassemia.value_counts()
sum(df_1.thalassemia.value_counts())
df_1.thalassemia.describe() 
# Countplot
import seaborn as sns
sns.countplot(x ='thalassemia', data = df_1) 
#------------------------------------------------------------------------------------------
#Heatmap
df_1.corr().target.sort_values()
sns.heatmap(df_1.corr())

#writing this to df_2
df_1.to_csv('df_2.csv')
df_2 = pd.read_csv('df_2.csv')
df_2.info()
df_2.shape
df_2 = df_2.drop(['Unnamed: 0'], axis = 1)
df_2.info()

#------------------------------------------------------------------------------------------
#libraries which we required for our decision tree building
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#classification_report
#Predictors
x = df_2.iloc[:,:13]

#Respond / target variable
y = df_2.iloc[:,13]

#Partitioning the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=123)

len(x_train) #212
len(x_test) #91
len(y_train) #212
len(y_test) #91

#Building Tree
clf = tree.DecisionTreeClassifier()
df_2_clf = clf.fit(x_train, y_train)

#Plotting Tree
fig, ax = plt.subplots(figsize=(20, 20))
tree.plot_tree(df_2_clf, ax=ax, fontsize=8,filled=True)
plt.show()

#Prediction on test data
y_pred = df_2_clf.predict(x_test)
len(y_pred)
print(y_pred)

#Confusion Matrix & Report
pd.crosstab(y_test,y_pred, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        32  13   45
1         9  37   46
All      41  50   91
'''
#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)#0.74

''' got 0.74 % accuracy'''

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)

#AUC -Area Under Curve
roc_auc = auc(fpr,tpr)
print(roc_auc)# 0.74
#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

'''
            precision    recall  f1-score   support

           0       0.78      0.69      0.73        45
           1       0.73      0.80      0.76        46

    accuracy                           0.75        91
   macro avg       0.75      0.75      0.75        91
weighted avg       0.75      0.75      0.75        91
'''
'''
1 pruning
2 remove outliers
3 bagging
4 random forect
5 adaptivebosting
6 XG booast
7 gradient boost
'''
#Pruning
#We will rebuild a new tree by using above data and see how it works by tweeking the parameteres

dtree = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 10, min_samples_leaf = 5, max_depth= 5)
dtree.fit(x_train,y_train)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=10, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='random')

predict3 = dtree.predict(x_train)
print(predict3)

predict4 = dtree.predict(x_test)
print(predict4)

#Accuracy of the model that we created with modified model parameters.
score2 = dtree.score(x_test, y_test)
score2

''' previously it was got 0.74 % accuracy, got 0.82 % accuracy after pruning'''

#bagging
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#classification_report
#Predictors
x = df_2.iloc[:,:13]

#Respond / target variable
y = df_2.iloc[:,13]

seed = 7
kfold = KFold(n_splits = 10, random_state = seed)
cart = DecisionTreeClassifier()
num_trees = 150

model = BaggingClassifier(base_estimator = cart, n_estimators = num_trees, random_state = seed)

results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())

''' previously it was got 0.79 % accuracy, got 0.76 % accuracy after pruning'''

#random forest
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

''' previously it was got 0.79 % accuracy, got 0.80 % accuracy after pruning'''


#tony's code
#Bagging 300 Trees
from sklearn.ensemble import BaggingClassifier
#Base estimator is clf
#Build Bagging Classifier bc
hd_bc = BaggingClassifier(base_estimator=clf, n_estimators=300, oob_score=True, n_jobs=-1)

#Bagging Classifier fitting with training data set
hd_bc.fit(x_train, y_train)    

#Predictions
y_predbc = hd_bc.predict(x_test)
print(y_predbc)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predbc, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        33  12   45
1         6  40   46
All      39  52   91'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predbc) #0.80

#Predictions _probabilities
yp_predbc = hd_bc.predict_proba(x_test)

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, yp_predbc[:,1])

#AUC -Area Under Curve
bc_roc_auc = auc(fpr,tpr)
print(bc_roc_auc) #0.91

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(bc_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predbc))
'''
                         precision    recall  f1-score   support

           0       0.85      0.73      0.79        45
           1       0.77      0.87      0.82        46

    accuracy                           0.80        91
   macro avg       0.81      0.80      0.80        91
weighted avg       0.81      0.80      0.80        91'''

#Random Forest
from sklearn.ensemble import RandomForestClassifier
#Create Model with 500 trees
rf = RandomForestClassifier(n_estimators=500,bootstrap=True,max_features='sqrt')

#Fitting the model
hd_rf = rf.fit(x_train, y_train)

#Prediction
y_predrf = hd_rf.predict(x_test)
len(y_predrf)
print(y_predrf)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predrf, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        34  11   45
1         7  39   46
All      41  50   91'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predrf) #0.80

#Predictions _probabilities
yp_predrf = hd_rf.predict_proba(x_test)

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, yp_predrf[:,1])

#AUC -Area Under Curve
rf_roc_auc = auc(fpr,tpr)
print(rf_roc_auc) #0.92

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(rf_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predrf))
'''
            precision    recall  f1-score   support

           0       0.83      0.76      0.79        45
           1       0.78      0.85      0.81        46

    accuracy                           0.80        91
   macro avg       0.80      0.80      0.80        91
weighted avg       0.80      0.80      0.80        91'''

#Importance of variables
#Extract Feature importance
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': hd_rf.feature_importances_}).\
                    sort_values('importance', ascending=False)

#Display
fi.head()
'''
        feature  importance
11           ca    0.132766
2    chest_pain    0.128034
7       thalach    0.118055
12  thalassemia    0.112282
9      old_peak    0.112229'''

#Adaptive Boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
adafit = ada.fit(x_train, y_train)
print(adafit)

#Prediction
y_predada = adafit.predict(x_test)
len(y_predada)
print(y_predada)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predada, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        31  14   45
1         9  37   46
All      40  51   91'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predada) #0.74

#Probabilities
yp_predada = adafit.predict_proba(x_test)[:,1]
print(yp_predada)
len(yp_predada)

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, yp_predada)

#AUC -Area Under Curve
adap_roc_auc = auc(fpr,tpr)
print(adap_roc_auc) #0.81

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(adap_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predada))
'''
               precision    recall  f1-score   support

           0       0.78      0.69      0.73        45
           1       0.73      0.80      0.76        46

    accuracy                           0.75        91
   macro avg       0.75      0.75      0.75        91
weighted avg       0.75      0.75      0.75        91'''











