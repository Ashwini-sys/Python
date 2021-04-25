#All things are possible with God
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

sovenir = pd.read_csv('D:/data _science/PYTHON/Time_Series_Python/sovenir.csv', date_parser=True)
sovenir.info()
sovenir.head()
sovenir.index = pd.DatetimeIndex(sovenir.month_year)
sovenir.info()
sovenir = sovenir.drop(['obsno', 'month_year'], axis=1)
sovenir.info()
sovenir.shape #84, 1
sovenir.head()

sovenir.describe()
'''
count      84.000000
mean    14315.587143
std     15748.840332
min      1664.810000
25%      5884.435000
50%      8771.770000
75%     16888.917500
max    104660.670000'''

#Lineplot
sovenir.plot()
plt.title('Data = sovenir')

#Lineplot - first 4years
sovenir[:48].plot()
plt.title('4yrs Data = sovenir')

#Histogram
sovenir.plot(kind='hist')
plt.title('Histogram of sovenir Data')

#Density plot
sovenir.plot(kind='kde')
plt.title('Density plot of sovenir Data')

#Boxplot
props2 = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
sovenir.plot.box(color = props2 , patch_artist = True, vert = False)
plt.title('Boxplot of sovenir Data')

#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# Season Decompose with Multiplicative model
sovenir_dec_m = seasonal_decompose(sovenir, model='multiplicative')

sovenir_dec_m.plot() #Trend & Seasonality visible

sovenir_dec_m.observed
sovenir_dec_m.trend.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months
sovenir_dec_m.seasonal
sovenir_dec_m.resid.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months

# Season Decompose with Additive model
sovenir_dec_a = seasonal_decompose(sovenir, model='additive')

sovenir_dec_a.plot() #Trend & Seasonality visible

sovenir_dec_a.observed
sovenir_dec_a.trend.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months
sovenir_dec_a.seasonal
sovenir_dec_a.resid.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months

#Applying log on the data
sovenir_log = np.log(sovenir.sales)

#Lineplot
sovenir_log.plot()
plt.title('Data = sovenir_log')

#Lineplot - first 4years
sovenir_log[:48].plot()
plt.title('4yrs Data = sovenir')

#Histogram
sovenir_log.plot(kind='hist')
plt.title('Histogram of sovenir_log Data')

#Density plot
sovenir_log.plot(kind='kde')
plt.title('Density plot of sovenir_log Data')

#Boxplot
props2 = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
sovenir_log.plot.box(color = props2 , patch_artist = True, vert = False)
plt.title('Boxplot of sovenir_log Data')

#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# Season Decompose with Multiplicative model
sovenir_log_dec_m = seasonal_decompose(sovenir_log, model='multiplicative')

sovenir_log_dec_m.plot() #Trend & Seasonality visible

sovenir_log_dec_m.observed
sovenir_log_dec_m.trend.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months
sovenir_log_dec_m.seasonal
sovenir_log_dec_m.resid.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months

# Season Decompose with Additive model
sovenir_log_dec_a = seasonal_decompose(sovenir_log, model='additive')

sovenir_log_dec_a.plot() #Trend & Seasonality visible

sovenir_log_dec_a.observed
sovenir_log_dec_a.trend.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months
sovenir_log_dec_a.seasonal
sovenir_log_dec_a.resid.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months

#Being trend & seasonality is observed in the data applying 
#Model with triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
sovenir_log_es = ExponentialSmoothing(sovenir_log, trend='add', seasonal='add').fit()
sovenir_log_es.summary()
'''
                       ExponentialSmoothing Model Results                       
================================================================================
Dep. Variable:                    sales   No. Observations:                   84
Model:             ExponentialSmoothing   SSE                              1.806
Optimized:                         True   AIC                           -290.519
Trend:                         Additive   BIC                           -251.626
Seasonal:                      Additive   AICC                          -279.996
Seasonal Periods:                    12   Date:                 Thu, 25 Mar 2021
Box-Cox:                          False   Time:                         22:05:35
Box-Cox Coeff.:                    None                                         
=================================================================================
                          coeff                 code              optimized      
---------------------------------------------------------------------------------
smoothing_level               0.4469032                alpha                 True
smoothing_trend              2.2196e-16                 beta                 True
smoothing_seasonal           1.4896e-15                gamma                 True
initial_level                 8.1596445                  l.0                 True
initial_trend                 0.0265265                  b.0                 True
initial_seasons.0            -0.7264619                  s.0                 True
initial_seasons.1            -0.4794325                  s.1                 True
initial_seasons.2            -0.0394028                  s.2                 True
initial_seasons.3            -0.3557417                  s.3                 True
initial_seasons.4            -0.3347759                  s.4                 True
initial_seasons.5            -0.2999557                  s.5                 True
initial_seasons.6            -0.1427969                  s.6                 True
initial_seasons.7            -0.1697504                  s.7                 True
initial_seasons.8            -0.0928354                  s.8                 True
initial_seasons.9            -0.0192853                  s.9                 True
initial_seasons.10            0.4355831                 s.10                 True
initial_seasons.11            1.1866500                 s.11                 True
---------------------------------------------------------------------------------'''

#Residual given by the model
sovenir_log_es_res = sovenir_log_es.resid
sovenir_log_es_res

#Histogram of residuals - train data
plt.hist(sovenir_log_es_res)
plt.title('sovenir_log - Residual given by the model')
plt.show()

#Plotting acf & pacf - residual
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(sovenir_log_es_res, title='sovenir_log - Residual given by the model - Autocorrelation') 
plot_pacf(sovenir_log_es_res, title='sovenir_log - Residual given by the model - Partial Autocorrelation')

#Squaring residuals/ errors
sovenir_log_es_se = pow(sovenir_log_es_res,2)
sovenir_log_es_se.head()

#average/mean of squared residuals/ errors
sovenir_log_es_mse = (sovenir_log_es_se.sum())/len(sovenir_log_es_se)
print(sovenir_log_es_mse) #0.02150398496214829

#Root of average/mean of squared residuals/ errors
sovenir_log_es_rmse = sqrt(sovenir_log_es_mse)
print(sovenir_log_es_rmse) #0.14664237096469862

#Forecasting next 19 periods
sovenir_log_es_pred = sovenir_log_es.forecast(19) #applied forecast not predict
sovenir_log_es_pred

#Plot Test vs Pred
plt.plot(sovenir_log)
plt.plot(sovenir_log_es_pred)
plt.xticks(rotation=45)
plt.legend(['Actual', 'Forecast'])
plt.show()

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
sovenir_log_adf = adfuller(sovenir_log)

print('ADF Statistic: %f' % sovenir_log_adf[0])
print('p-value: %f' % sovenir_log_adf[1])
print('Critical Values:')
for key, value in sovenir_log_adf[4].items():
    print('\t%s: %.3f' % (key, value))
    
'''
ADF Statistic: 0.206646
p-value: 0.972632
Critical Values:
	1%: -3.526
	5%: -2.903
	10%: -2.589'''
    
#p-value: 0.972632 ie > 0.5, Null Hypothesis accepted, Data is not stationary
#H0: Data is not stationary
'''When data is not stationary, apply differencing and check for stationarity'''

#Differencing @1
sovenir_log1 = sovenir_log.diff() #default 1
sovenir_log1.head()
sovenir_log1 = sovenir_log1.dropna()
sovenir_log1.head()

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
sovenir_log1_adf = adfuller(sovenir_log1)

print('ADF Statistic: %f' % sovenir_log1_adf[0])
print('p-value: %f' % sovenir_log1_adf[1])
print('Critical Values:')
for key, value in sovenir_log1_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -2.962946
p-value: 0.038503
Critical Values:
	1%: -3.527
	5%: -2.904
	10%: -2.589'''
    
#p-value: 0.038503 ie < 0.05, Null Hypothesis rejected, Data is stationary
#H0: Data is not stationary

#Applying autoarima
from pmdarima import auto_arima

sovenir_log1_mod1 = auto_arima(sovenir_log1)
sovenir_log1_mod1.summary()
'''
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                   83
Model:               SARIMAX(2, 0, 0)   Log Likelihood                 -66.728
Date:                Thu, 25 Mar 2021   AIC                            139.457
Time:                        22:37:57   BIC                            146.713
Sample:                             0   HQIC                           142.372
                                 - 83                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.4435      0.097     -4.573      0.000      -0.634      -0.253
ar.L2         -0.4253      0.168     -2.526      0.012      -0.755      -0.095
sigma2         0.2905      0.043      6.773      0.000       0.206       0.375
===================================================================================
Ljung-Box (L1) (Q):                   0.29   Jarque-Bera (JB):                23.93
Prob(Q):                              0.59   Prob(JB):                         0.00
Heteroskedasticity (H):               1.04   Skew:                            -0.87
Prob(H) (two-sided):                  0.92   Kurtosis:                         4.97
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).'''

#Residual given by the model
sovenir_log1_mod1_res = sovenir_log1_mod1.resid()
sovenir_log1_mod1_res
sovenir_log1_mod1_res = pd.DataFrame(sovenir_log1_mod1_res, index=sovenir_log1.index)
sovenir_log1_mod1_res

#Histogram of residuals
plt.hist(sovenir_log1_mod1_res)
plt.title('sovenir_log1 - Residual with train data')
plt.show()

#Plotting acf & pacf - residual
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(sovenir_log1_mod1_res, title='sovenir_log1 - Residual given by the Model - Autocorrelation') 
plot_pacf(sovenir_log1_mod1_res, title='sovenir_log1 - Residual given by the Model - Partial Autocorrelation')

#Squaring residuals/ errors
sovenir_log1_mod1_se = pow(sovenir_log1_mod1_res,2)
sovenir_log1_mod1_se.head()

#average/mean of squared residuals/ errors
sovenir_log1_mod1_mse = (sovenir_log1_mod1_se.sum())/len(sovenir_log1_mod1_se)
print(sovenir_log1_mod1_mse) #0.291135

#Root of average/mean of squared residuals/ errors
sovenir_log1_mod1_rmse = sqrt(sovenir_log1_mod1_mse)
print(sovenir_log1_mod1_rmse) #0.5395696486390393

#Forecasting next 19 periods
sovenir_log1_mod1_pred = sovenir_log1_mod1.predict(19)
sovenir_log1_mod1_pred = pd.DataFrame(sovenir_log1_mod1_pred, 
                                      index=pd.date_range(start='1994-01-01', 
                                                          periods=19, freq='MS'))
sovenir_log1_mod1_pred

#Plot Actual & forecast
plt.plot(sovenir_log1)
plt.plot(sovenir_log1_mod1_pred)
plt.legend(['Actual Data', 'Forecast'])
plt.xticks(rotation=45)
plt.show()