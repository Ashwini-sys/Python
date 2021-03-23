 # Jesus is my Saviour!
import os
os.chdir('C:/Users/tonyk/Documents/DSP 34/Python/Logistic Regression')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

vac = pd.read_csv('df2_H1N1.csv')
vac.info()
vac.shape

x = vac.iloc[:,:-1]





#MLE - Maximum Likelihood estimation
import statsmodels.api as sm
sm_logit = sm.Logit(y,x).fit() #target variable and response variable
model_log.summary()

predictions = sm_logit.predict(x) #Predictions are in probabilities
predictions_nominal = [0 if x < 0.5 else 1 for x in predictions]

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, predictions_nominal)
print(cm)
'''
[[17091   700]
 [ 4152  1033]]'''

#Accuracy score - correct predictions / total number of data points
(17091+1033)/(22976) #0.788

#ROC & AUC
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(y, predictions)
roc_auc = auc(fpr, tpr) #Area under Curve 0.0.6883
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y, predictions_nominal, digits = 3))