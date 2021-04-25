# God is King of Kings!
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

births = pd.read_csv('D:/data _science/PYTHON/Time_Series_Python/births.csv', date_parser=True)
births.info()
births.head()

#Indexing month_year
#2 digit year without century while converting 4 digit format, 
#python considers as present century so adjusting the year
births.index = pd.DatetimeIndex(births.month_year)+pd.DateOffset(years=-100)
births.info()

#Removing unecessary variables
births = births.drop(['obsno','month_year'], axis=1)
births.info()
births.shape #168, 1
births.head() 

#Variable births
births.describe()
'''
           births
count  168.000000
mean    25.059310
std      2.318791
min     20.000000
25%     23.280750
50%     24.957000
75%     26.878750
max     30.000000'''

#Lineplot
births.plot()
plt.title('Data = births')

#Only 4years of data
births[:49].plot()
plt.title('4 Yrs Data = births')

#Histogram
births.plot(kind='hist')
plt.title('Histogram of births Data')

#Density plot
births.plot(kind='kde')
plt.title('Density plot of births Data')

#Boxplot
props2  = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
births.plot.box(color = props2 , patch_artist = True, vert = False)
plt.title('Boxplot of births Data')

#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# Season Decompose with Multiplicative model
births_dec_m = seasonal_decompose(births, model='multiplicative')

births_dec_m.plot() #Trend & Seasonality visible

births_dec_m.observed
births_dec_m.trend.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months
births_dec_m.seasonal
births_dec_m.resid.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months

# Season Decompose with Additive model
births_dec_a = seasonal_decompose(births, model='additive')

births_dec_a.plot() #Trend & Seasonality visible

births_dec_a.observed
births_dec_a.trend.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months
births_dec_a.seasonal
births_dec_a.resid.head(20) #First 6 and last 6 values are Na's due calculation of seasonality indices of 12 months
 
'''
SimpleExpSmoothing is used when there is no trend and Seasonal
Holt is used when there is trend
ExponentialSmoothing is when there trend and seasonal'''

#For this data supposed to use last method but trying other methods for practice

#Model with single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
births_ses = SimpleExpSmoothing(births).fit()
births_ses.summary()
'''
                       SimpleExpSmoothing Model Results                       
==============================================================================
Dep. Variable:                 births   No. Observations:                  168
Model:             SimpleExpSmoothing   SSE                            280.315
Optimized:                       True   AIC                             90.007
Trend:                           None   BIC                             96.255
Seasonal:                        None   AICC                            90.253
Seasonal Periods:                None   Date:                 Fri, 26 Mar 2021
Box-Cox:                        False   Time:                         00:21:27
Box-Cox Coeff.:                  None                                         
==============================================================================
                       coeff                 code              optimized      
------------------------------------------------------------------------------
smoothing_level            0.4379774                alpha                 True
initial_level              25.611519                  l.0                 True
------------------------------------------------------------------------------'''

#forecasting/ predicting next 19 periods
births_pred = births_ses.forecast(steps=19)
print(births_pred)

#Plot actual and forecast
plt.plot(births)
plt.plot(births_pred)
plt.legend(['Actual', 'Forecast - SimpleExpSmoothing'],
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Model with double exponential smoothing
from statsmodels.tsa.holtwinters import Holt
births_holt = Holt(births).fit()
births_holt.summary()
'''                              Holt Model Results                              
==============================================================================
Dep. Variable:                 births   No. Observations:                  168
Model:                           Holt   SSE                            280.134
Optimized:                       True   AIC                             93.899
Trend:                       Additive   BIC                            106.395
Seasonal:                        None   AICC                            94.421
Seasonal Periods:                None   Date:                 Fri, 26 Mar 2021
Box-Cox:                        False   Time:                         00:23:58
Box-Cox Coeff.:                  None                                         
==============================================================================
                       coeff                 code              optimized      
------------------------------------------------------------------------------
smoothing_level            0.4354824                alpha                 True
smoothing_trend           1.8914e-12                 beta                 True
initial_level              25.574165                  l.0                 True
initial_trend              0.0144988                  b.0                 True
------------------------------------------------------------------------------'''

#forecasting/ predicting
births_pred1 = births_holt.forecast(steps=19)
print(births_pred1)

#Plot actual and forecast
plt.plot(births)
plt.plot(births_pred1)
plt.legend(['Actual', 'Forecast - Holt'], bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Model with triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
births_es = ExponentialSmoothing(births, seasonal_periods=12,
                                 trend='add', seasonal='add').fit()
births_es.summary()

'''                       ExponentialSmoothing Model Results                       
================================================================================
Dep. Variable:                   births   No. Observations:                  168
Model:             ExponentialSmoothing   SSE                             63.346
Optimized:                         True   AIC                           -131.859
Trend:                         Additive   BIC                            -81.875
Seasonal:                      Additive   AICC                          -127.268
Seasonal Periods:                    12   Date:                 Fri, 26 Mar 2021
Box-Cox:                          False   Time:                         00:25:39
Box-Cox Coeff.:                    None                                         
=================================================================================
                          coeff                 code              optimized      
---------------------------------------------------------------------------------
smoothing_level               0.9401515                alpha                 True
smoothing_trend              8.5161e-11                 beta                 True
smoothing_seasonal            5.805e-12                gamma                 True
initial_level                 27.155582                  l.0                 True
initial_trend                 0.0062209                  b.0                 True
initial_seasons.0            -0.5917886                  s.0                 True
initial_seasons.1            -2.0910713                  s.1                 True
initial_seasons.2             0.9121398                  s.2                 True
initial_seasons.3            -0.7605586                  s.3                 True
initial_seasons.4             0.3205229                  s.4                 True
initial_seasons.5            -0.1308801                  s.5                 True
initial_seasons.6             1.4655643                  s.6                 True
initial_seasons.7             1.2730839                  s.7                 True
initial_seasons.8             0.7820954                  s.8                 True
initial_seasons.9             0.8415282                  s.9                 True
initial_seasons.10           -1.0539487                 s.10                 True
initial_seasons.11           -0.3096054                 s.11                 True
---------------------------------------------------------------------------------'''

#Residual given by the model
births_es_res = births_es.resid
births_es_res

#Histogram of residuals
plt.hist(births_es_res)
plt.title('births - Residual given by the Model')
plt.show()

#Plotting acf & pacf - residual
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(births_es_res, lags=20, title='births - Residual given by the Model - Autocorrelation') 
plot_pacf(births_es_res, lags=20, title='births - Residual given by the Model - Partial Autocorrelation')

#Squaring residuals/ errors
births_es_se = pow(births_es_res,2)
births_es_se.head()

#average/mean of squared residuals/ errors
births_es_mse = (births_es_se.sum())/len(births_es_se)
print(births_es_mse) #0.3770609200564516

#Root of average/mean of squared residuals/ errors
births_es_rmse = sqrt(births_es_mse)
print(births_es_rmse) #0.6140528642197279

#forecasting/ predicting
births_pred2 = births_es.forecast(steps=19)
print(births_pred2)

#Plot actual and forecast
plt.plot(births)
plt.plot(births_pred2)
plt.legend(['Actual', 'Forecast - ExponentialSmoothing'], 
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
births_adf = adfuller(births)

print('ADF Statistic: %f' % births_adf[0])
print('p-value: %f' % births_adf[1])
print('Critical Values:')
for key, value in births_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -0.331281
p-value: 0.920956
Critical Values:
	1%: -3.474
	5%: -2.880
	10%: -2.577'''
#p-value: 0.920956 ie > 0.05, Null Hypothesis accepted and the data is not stationary    
#H0: Data is not stationary

'''When data is not stationary, apply differencing and check for stationarity'''

#Differencing @1
births1 = births.diff() #default 1
births1.head()
births1 = births.dropna()
births1.head()

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
births1_adf = adfuller(births1)

print('ADF Statistic: %f' % births1_adf[0])
print('p-value: %f' % births1_adf[1])
print('Critical Values:')
for key, value in births1_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -0.331281
p-value: 0.920956
Critical Values:
	1%: -3.474
	5%: -2.880
	10%: -2.577'''
#p-value: 0.920956 ie > 0.05, Null Hypothesis accepted and the data is not stationary    
#H0: Data is not stationary

#Applying auto - arima to forecast
from pmdarima import auto_arima

births_mod = auto_arima(births)
births_mod.summary()
'''
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  168
Model:               SARIMAX(2, 1, 1)   Log Likelihood                -271.935
Date:                Fri, 26 Mar 2021   AIC                            551.870
Time:                        00:35:55   BIC                            564.342
Sample:                             0   HQIC                           556.932
                                - 168                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.2509      0.095      2.643      0.008       0.065       0.437
ar.L2          0.3441      0.116      2.977      0.003       0.118       0.571
ma.L1         -0.9143      0.065    -14.166      0.000      -1.041      -0.788
sigma2         1.5133      0.194      7.788      0.000       1.132       1.894
===================================================================================
Ljung-Box (L1) (Q):                   0.20   Jarque-Bera (JB):                 2.24
Prob(Q):                              0.66   Prob(JB):                         0.33
Heteroskedasticity (H):               1.22   Skew:                             0.14
Prob(H) (two-sided):                  0.46   Kurtosis:                         2.51
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).'''

#Residual given by the model
births_mod_res = births_mod.resid()
births_mod_res

#Adding index and converting to datframe
births_mod_res = pd.DataFrame(births_mod_res, index=births.index)
births_mod_res

#Histogram of residuals
plt.hist(births_mod_res)
plt.title('births - Residual given by the model')
plt.show()

#Plotting acf & pacf - residual
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(births_mod_res, lags=20, title='births - Residual given by the model - Autocorrelation') 
plot_pacf(births_mod_res, lags=20, title='births - Residual given by the model - Partial Autocorrelation')

#Squaring residuals/ errors
births_mod_se = pow(births_mod_res,2)
births_mod_se.head()

#average/mean of squared residuals/ errors
births_mod_mse = (births_mod_se.sum())/len(births_mod_se)
print(births_mod_mse) #5.757373

#Root of average/mean of squared residuals/ errors
births_mod_rmse = sqrt(births_mod_mse)
print(births_mod_rmse) #2.399452542080286

#Forecasting next 19 periods
births_mod_pred = births_mod.predict(n_periods=19)
births_mod_pred

#Adding index to forecast and converting to dataframe

births_mod_pred = pd.DataFrame(births_mod_pred, 
                               index=pd.date_range(start='1960-01-01', 
                                                   periods=19, freq='MS'))
births_mod_pred

#Plot actual and forecast
plt.plot(births)
plt.plot(births_mod_pred)
plt.legend(['Actual', 'Forecast - Auto Arima'], 
           bbox_to_anchor=(1, 1), loc=2)
plt.show()