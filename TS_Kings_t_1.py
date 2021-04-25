# GOD is King of Kings!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
pd.set_option('display.max_columns', None)
import numpy as np 
!pip install matplotlib 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
#%matplotlib inline #for showing plots in console
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

kings = pd.read_csv('D:/data _science/PYTHON/Time_Series_Python/kings.csv')
kings.info() # 42 
kings = kings.drop('obsno', axis=1)
kings.info()
kings.index #RangeIndex(start=0, stop=42, step=1)

#lineplot
plt.plot(kings, 'r')
plt.title('data = kings', fontsize=16)

#Histogram
kings.plot(kind='hist')
plt.title('Histogram of kings Data')

#Density plot
kings.plot(kind='kde')
plt.title('Density plot of kings Data')

#Boxplot
props2 = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
kings.plot.box(color = props2 , patch_artist = True, vert = False)
plt.title('Boxplot of kings Data')

#Decompose with multiplicative
from statsmodels.tsa.seasonal import seasonal_decompose
kings_decomp_m = seasonal_decompose(kings, period=1, model='mul') #Peroid is specified being the data is not having date as index
kings_decomp_m.plot() #No Trend & no seasonality
kings_decomp_m.observed
kings_decomp_m.trend
kings_decomp_m.seasonal
kings_decomp_m.resid

#Decompose with Additive
from statsmodels.tsa.seasonal import seasonal_decompose
kings_decomp_m = seasonal_decompose(kings, period=1, model='add')
kings_decomp_m.plot() #No Trend & no seasonality
kings_decomp_m.observed
kings_decomp_m.trend
kings_decomp_m.seasonal
kings_decomp_m.resid

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
kings_adf = adfuller(kings)

print('ADF Statistic: %f' % kings_adf[0])
print('p-value: %f' % kings_adf[1])
print('Critical Values:')
for key, value in kings_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -4.090230
p-value: 0.001005
Critical Values:
	1%: -3.601
	5%: -2.935
	10%: -2.606'''
#p-value: 0.001005 ie <= 0.05, Null Hypothesis rejected, so, the data is stationary
#H0: Data is not stationary

#Moving average/Rolloing average @3
kings_ma3 = kings.rolling(window=3).mean()
kings_ma3.head() # 1st & 2nd obs will be na
'''we have use moving avg for smoothing the data and then we can see better plot'''

#lineplot
plt.plot(kings_ma3, 'r')
plt.title('data = kings', fontsize=16)

#Residuals / errors
kings_ma3_res = kings - kings_ma3
kings_ma3_res.head()
kings_ma3_res = kings_ma3_res.dropna()
kings_ma3_res.head()

#Plotting histogram for residuals
plt.hist(kings_ma3_res)
plt.title('Histogram Residuals @MA3')

''' if we r getting symmetrical hist 
    then its good model that we did!
'''

#Plotting acf & pacf 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(kings_ma3_res, lags=20) 
plot_pacf(kings_ma3_res, lags=19)

#Squaring residuals/ errors
kings_ma3_se = pow(kings_ma3_res,2)
kings_ma3_se.head()

#average/mean of squared residuals/ errors
kings_ma3_mse = (kings_ma3_se.sum())/len(kings_ma3_se)
print(kings_ma3_mse) #128.7527777777778

#Root of average/mean of squared residuals/ errors
kings_ma3_rmse = sqrt(kings_ma3_mse) 
print(kings_ma3_rmse) #11.346928120763689

#Another method to find RMSE
kings_ma3 = kings.rolling(window=3).mean()
kings_ma3 = kings_ma3.dropna()
ma3_rmse = sqrt(mean_squared_error(kings[2:],kings_ma3))
print(ma3_rmse) #11.346928120763689

#Moving average/Rolloing average @8
kings_ma8 = kings.rolling(window=8).mean()
print(kings_ma8) # First 7 obs will be na

#lineplot
plt.plot(kings_ma8, 'g')
plt.title('data = kings', fontsize=16)

#Residuals / errors
kings_ma8_res = kings - kings_ma8
kings_ma8_res.head(10)
kings_ma8_res = kings_ma8_res.dropna() 
kings_ma8_res.head()

#Plotting histogram for residuals
plt.hist(kings_ma8_res)
plt.title('Histogram Residuals @MA8')

#Plotting acf & pacf 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(kings_ma8_res, lags=20) 
plot_pacf(kings_ma8_res, lags=16) # why 16? it should be less than 17 coz its half ofour datasize and our data side is 35

#Squaring residuals/ errors
kings_ma8_se = pow(kings_ma8_res,2)
kings_ma8_se.head(10)

#average/mean of squared residuals/ errors
kings_ma8_mse = (kings_ma8_se.sum())/len(kings_ma8_se)
print(kings_ma3_mse) #128.752778

#Root of average/mean of squared residuals/ errors
kings_ma8_rmse = sqrt(kings_ma8_mse) 
print(kings_ma8_rmse) #14.393373351253397

#Another method to find RMSE
kings_ma8 = kings.rolling(window=8).mean()
kings_ma8 = kings_ma8.dropna()
ma8_rmse = sqrt(mean_squared_error(kings[7:],kings_ma8))
print(ma8_rmse) #14.393373351253397

#Predicting with Moving average using Arima 
from statsmodels.tsa.arima.model import ARIMA
kings_ma_model = ARIMA(kings, order=(0,0,3)).fit()
kings_ma_model.summary()
#SARIMAX = Seasonal Auto Regressive Integrated Moving Average with Exogeneous Regressors
'''
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                    age   No. Observations:                   42
Model:                 ARIMA(0, 0, 3)   Log Likelihood                -173.643
Date:                Thu, 25 Mar 2021   AIC                            357.286
Time:                        19:48:27   BIC                            365.974
Sample:                             0   HQIC                           360.470
                                 - 42                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         55.3010      4.155     13.310      0.000      47.158      63.444
ma.L1          0.3525      0.256      1.375      0.169      -0.150       0.855
ma.L2          0.1218      0.197      0.617      0.537      -0.265       0.509
ma.L3          0.0434      0.194      0.224      0.823      -0.336       0.423
sigma2       227.6143     58.406      3.897      0.000     113.140     342.089
===================================================================================
Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):                 2.69
Prob(Q):                              0.90   Prob(JB):                         0.26
Heteroskedasticity (H):               1.03   Skew:                            -0.61
Prob(H) (two-sided):                  0.96   Kurtosis:                         3.19
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).'''

#Prediction
kings_ma_pred = kings_ma_model.forecast(10)
type(kings_ma_pred)

#Converting series to data frame
kings_ma_pred = pd.DataFrame(kings_ma_pred)
kings_ma_pred

#Plot
plt.plot(kings)
plt.plot(kings_ma_pred)
plt.legend(['Actual','Forecast'], bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Applying autoarima

import pmdarima
from pmdarima import auto_arima
model_autoarima = auto_arima(kings) 
model_autoarima.summary()

'''
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                   42
Model:               SARIMAX(0, 1, 1)   Log Likelihood                -170.064
Date:                Thu, 25 Mar 2021   AIC                            344.127
Time:                        20:03:34   BIC                            347.554
Sample:                             0   HQIC                           345.375
                                 - 42                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ma.L1         -0.7218      0.146     -4.957      0.000      -1.007      -0.436
sigma2       230.4371     57.900      3.980      0.000     116.956     343.919
===================================================================================
Ljung-Box (L1) (Q):                   0.12   Jarque-Bera (JB):                 0.38
Prob(Q):                              0.73   Prob(JB):                         0.83
Heteroskedasticity (H):               1.15   Skew:                            -0.15
Prob(H) (two-sided):                  0.80   Kurtosis:                         2.64
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).'''

#Prediction
kings_pred = model_autoarima.predict(n_periods=10)
kings_pred
kings_pred = pd.DataFrame(kings_pred)

#Plot
plt.plot(kings)
plt.plot(kings_pred)
plt.legend(['Train','Test','Prediction'], bbox_to_anchor=(1, 1), loc=2)
plt.show()