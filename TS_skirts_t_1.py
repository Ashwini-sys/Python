# God is Great!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
pd.set_option('display.max_columns', None)
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
#%matplotlib inline
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

skirts = pd.read_csv('D:/data _science/PYTHON/Time_Series_Python/skirts.csv', index_col=[1], parse_dates=True, squeeze=True)
skirts.info()
skirts = skirts.drop('obsno', axis=1)
skirts.head()
skirts.tail()
skirts.describe()
'''
              diam
count    46.000000
mean    759.260870
std     179.202856
min     523.000000
25%     593.000000
50%     765.000000
75%     909.750000
max    1049.000000'''

#Lineplot
skirts.plot()
plt.title('Data = Skirts')

#Histogram
skirts.plot(kind='hist')
plt.title('Histogram of Skirts Data')

#Density plot
skirts.plot(kind='kde')
plt.title('Density plot of skirts Data')

#Boxplot
props2 = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
skirts.plot.box(color = props2 , patch_artist = True, vert = False)
plt.title('Boxplot of Skirts Data')

#Decompose with multiplicative model
from statsmodels.tsa.seasonal import seasonal_decompose
skirts_decomp_m = seasonal_decompose(skirts, model='mul')
skirts_decomp_m.plot() #No Trend & no seasonality
skirts_decomp_m.observed
skirts_decomp_m.trend
skirts_decomp_m.seasonal
skirts_decomp_m.resid

#Decompose with additive model
from statsmodels.tsa.seasonal import seasonal_decompose
skirts_decomp_a = seasonal_decompose(skirts, model='add')
skirts_decomp_a.plot() #No Trend & no seasonality
skirts_decomp_a.observed
skirts_decomp_a.trend
skirts_decomp_a.seasonal
skirts_decomp_a.resid

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
skirts_adf = adfuller(skirts)

print('ADF Statistic: %f' % skirts_adf[0])
print('p-value: %f' % skirts_adf[1])
print('Critical Values:')
for key, value in skirts_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -1.917555
p-value: 0.323848
Critical Values:
	1%: -3.616
	5%: -2.941
	10%: -2.609'''
#p-value: 0.323848 ie > 0.5, Null Hypothesis accepted, Data is not stationary
#H0: Data is not stationary

'''When data is not stationary, apply differencing and check for stationarity'''
#Differencing @1
skirts_diff1 = skirts.diff() #default 1
skirts_diff1.head()
skirts_diff1 = skirts_diff1.dropna()
skirts_diff1

#Lineplot
plt.plot(skirts_diff1, 'r')
plt.title('Data = Skirts@Diff_1')

#Histogram
plt.hist(skirts_diff1)
plt.title('Histogram of Skirts@Diff_1')

#Density plot
skirts_diff1.plot(kind='kde')
plt.title('Density plot of skirts Data')

#Decompose with Additive model
#Multiplicative seasonality is not appropriate for zero and negative values
#So model multiplicative is not applicable
from statsmodels.tsa.seasonal import seasonal_decompose
skirts_diff1_decomp_a = seasonal_decompose(skirts_diff1, model='add')
skirts_diff1_decomp_a.plot() #No Trend & no seasonality
skirts_diff1_decomp_a.observed
skirts_diff1_decomp_a.trend
skirts_diff1_decomp_a.seasonal
skirts_diff1_decomp_a.resid

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
skirts_diff1_adf = adfuller(skirts_diff1)

print('ADF Statistic: %f' % skirts_diff1_adf[0])
print('p-value: %f' % skirts_diff1_adf[1])
print('Critical Values:')
for key, value in skirts_diff1_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -2.520768
p-value: 0.110503
Critical Values:
	1%: -3.639
	5%: -2.951
	10%: -2.614'''
#p-value: 0.110503 ie > 0.5, Null Hypothesis accepted, Data is not stationary
#H0: Data is not stationary

#Differencing @2
skirts_diff2 = skirts.diff(2) 
skirts_diff2.head()
skirts_diff2 = skirts_diff2.dropna()
skirts_diff2

#Lineplot
plt.plot(skirts_diff2, 'r')
plt.title('Data = Skirts@Diff_2')

#Histogram
plt.hist(skirts_diff2)
plt.title('Histogram of Skirts@Diff_2')

#Density plot
skirts_diff2.plot(kind='kde')
plt.title('Density plot of skirts Data')

#Decompose with additive model
from statsmodels.tsa.seasonal import seasonal_decompose
skirts_diff2_decomp_a = seasonal_decompose(skirts_diff2, model='add')
skirts_diff2_decomp_a.plot() #No Trend & no seasonality
skirts_diff2_decomp_a.observed
skirts_diff2_decomp_a.trend
skirts_diff2_decomp_a.seasonal
skirts_diff2_decomp_a.resid

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
skirts_diff2_adf = adfuller(skirts_diff2)

print('ADF Statistic: %f' % skirts_diff2_adf[0])
print('p-value: %f' % skirts_diff2_adf[1])
print('Critical Values:')
for key, value in skirts_diff2_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -1.614915
p-value: 0.475396
Critical Values:
	1%: -3.610
	5%: -2.939
	10%: -2.608'''

#p-value: 0.475396 ie > 0.5, Null Hypothesis accepted, Data is not stationary
#H0: Data is not stationary

#Being trend and seasonality is not visible in the data using
#single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
skirts_ses = SimpleExpSmoothing(skirts).fit()
skirts_ses.summary()
'''
                       SimpleExpSmoothing Model Results                       
==============================================================================
Dep. Variable:                   diam   No. Observations:                   46
Model:             SimpleExpSmoothing   SSE                          41869.930
Optimized:                       True   AIC                            317.429
Trend:                           None   BIC                            321.087
Seasonal:                        None   AICC                           318.405
Seasonal Periods:                None   Date:                 Thu, 25 Mar 2021
Box-Cox:                        False   Time:                         23:09:03
Box-Cox Coeff.:                  None                                         
==============================================================================
                       coeff                 code              optimized      
------------------------------------------------------------------------------
smoothing_level            0.9950000                alpha                 True
initial_level              608.00000                  l.0                 True
------------------------------------------------------------------------------'''

#Residual given by the model
skirts_ses_res = skirts_ses.resid
skirts_ses_res.head()

#Histogram of residuals
plt.hist(skirts_ses_res)
plt.title('births - Residual given by the Model')
plt.show()

#Plotting acf & pacf - residual
import statsesels
from statsesels.graphics.tsaplots import plot_acf
from statsesels.graphics.tsaplots import plot_pacf
plot_acf(skirts_ses_res, lags=20, title='births - Residual given by the Model - Autocorrelation') 
plot_pacf(skirts_ses_res, lags=20, title='births - Residual given by the Model - Partial Autocorrelation')

#Squaring residuals/ errors
skirts_ses_se = pow(skirts_ses_res,2)
skirts_ses_se.head()

#average/mean of squared residuals/ errors
skirts_ses_mse = (skirts_ses_se.sum())/len(skirts_ses_se)
print(skirts_ses_mse) #910.2158736656878

#Root of average/mean of squared residuals/ errors
skirts_ses_rmse = sqrt(skirts_ses_mse)
print(skirts_ses_rmse) #30.16978411698844

#Forecasting next 19 periods
skirts_pred = skirts_ses.forecast(steps=19)
print(skirts_pred)

#Plot of actual and forecast
plt.plot(skirts)
plt.plot(skirts_pred)
plt.legend(['Actual', 'Forecast'])
plt.show()

#Applying autoarima
from pmdarima import auto_arima

skirts_aa = auto_arima(skirts)
skirts_aa.summary()
'''
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                   46
Model:               SARIMAX(1, 2, 0)   Log Likelihood                -193.664
Date:                Thu, 25 Mar 2021   AIC                            391.329
Time:                        23:12:43   BIC                            394.897
Sample:                             0   HQIC                           392.652
                                 - 46                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.2997      0.152     -1.971      0.049      -0.598      -0.002
sigma2       388.7357    114.253      3.402      0.001     164.804     612.667
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                 1.79
Prob(Q):                              0.97   Prob(JB):                         0.41
Heteroskedasticity (H):               0.88   Skew:                            -0.14
Prob(H) (two-sided):                  0.80   Kurtosis:                         2.05
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).'''

#Residuals given by the mode
skirts_aa_res = skirts_aa.resid()
skirts_aa_res
#Addings index to the residuals & converting into dataframe
skirts_aa_res = pd.DataFrame(skirts_aa_res, index=skirts.index)
skirts_aa_res

#Calculating RMSE
skirts_aa_rmse = sqrt(mean_squared_error(skirts,skirts_aa_res))
print(skirts_aa_rmse) #782.2861317288002

#Residuals plot
plt.hist(skirts_aa_res)
plt.title('Histogram of Skirts_residuals')

#Forecasting next 19 periods
skirts_aa_pred = skirts_aa.predict(n_periods=19)
skirts_aa_pred

#Adding date index to the forecasting values and converting arrays to dataframe
skirts_aa_pred = pd.DataFrame(skirts_aa_pred, 
                              index=pd.date_range(start='1912-01-01',
                                                  periods=19,freq='YS'))
skirts_aa_pred

#Plot of actual and forecast
plt.plot(skirts)
plt.plot(skirts_aa_pred)
plt.legend(['Actual', 'Forecast'])
plt.show()
