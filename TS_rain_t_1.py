#Showers of blessing
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

rain = pd.read_csv('D:/data _science/PYTHON/Time_Series_Python/rain.csv', index_col=[1], parse_dates=True)
rain.info()
rain = rain.drop('obsno', axis=1)
rain.info()
rain.shape #100,1
rain.head()
rain.describe()
'''
             rain
count  100.000000
mean    24.823900
std      4.214531
min     16.930000
25%     22.202500
50%     23.870000
75%     27.510000
max     38.100000'''

#Lineplot
rain.plot()
plt.title('Data = rain')

#Histogram
rain.plot(kind='hist')
plt.title('Histogram of rain Data')

#Density plot
rain.plot(kind='kde')
plt.title('Density plot of rain Data')

#Boxplot
props2 = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
rain.plot.box(color = props2 , patch_artist = True, vert = False)
plt.title('Boxplot of rain Data')

#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose

# Season Decompose with Multiplicative model
rain_dec_m = seasonal_decompose(rain, model='multiplicative')

rain_dec_m.plot() #No trend & No seasonality, residuals is 1 for all observations

rain_dec_m.observed
rain_dec_m.trend
rain_dec_m.seasonal
rain_dec_m.resid

# Season Decompose with Additive model
rain_dec_a = seasonal_decompose(rain, model='additive')

rain_dec_a.plot() #No trend & No seasonality, residuals is 0 for all observations

rain_dec_a.observed
rain_dec_a.trend
rain_dec_a.seasonal
rain_dec_a.resid

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
rain_adf = adfuller(rain)

print('ADF Statistic: %f' % rain_adf[0])
print('p-value: %f' % rain_adf[1])
print('Critical Values:')
for key, value in rain_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -10.502000
p-value: 0.000000
Critical Values:
	1%: -3.498
	5%: -2.891
	10%: -2.583'''    

#p-value: 0.0 ie < 0.05, Null Hypothesis rejected, so,  Data is stationary
#H0: Data is not stationary

#Moving average/Rolloing average @2
rain_ma2 = rain.rolling(window=2).mean()
print(rain_ma2) # 1st obs will be na

#Residuals / errors
rain_ma2_res = rain - rain_ma2
rain_ma2_res = rain_ma2_res.dropna() #na is trouble to get acf & pacf
rain_ma2_res.head()

#Plotting histogram for residuals
plt.hist(rain_ma2_res)
plt.title('Histogram Residuals @MA2')

#Squaring residuals/ errors
rain_ma2_se = pow(rain_ma2_res,2)
rain_ma2_se.head()

#average/mean of squared residuals/ errors
rain_ma2_mse = (rain_ma2_se.sum())/len(rain_ma2_se)
print(rain_ma2_mse) #9.439844

#Root of average/mean of squared residuals/ errors
rain_ma2_rmse = sqrt(rain_ma2_mse) 
print(rain_ma2_rmse) #3.072432984532689

#Another method to find RMSE
rain_ma2 = rain.rolling(window=2).mean()
rain_ma2 = rain_ma2.dropna()
ma2_rmse = sqrt(mean_squared_error(rain[1:],rain_ma2))
print(ma2_rmse) #3.072432984532689

#Plotting acf & pacf 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(rain_ma2_res) 
plot_pacf(rain_ma2_res)

#Moving average/Rolloing average @3
rain_ma3 = rain.rolling(window=3).mean()
print(rain_ma3) # 1st and 2nd obs will be na

#Residuals / errors
rain_ma3_res = rain - rain_ma3
rain_ma3_res = rain_ma3_res.dropna() #na is trouble to get acf & pacf
rain_ma3_res.head()

#Plotting histogram for residuals
plt.hist(rain_ma3_res)
plt.title('Histogram Residuals @MA3')

#Squaring residuals/ errors
rain_ma3_se = pow(rain_ma3_res,2)
rain_ma3_se.head()

#average/mean of squared residuals/ errors
rain_ma3_mse = (rain_ma3_se.sum())/len(rain_ma3_se)
print(rain_ma3_mse) #12.902231

#Root of average/mean of squared residuals/ errors
rain_ma3_rmse = sqrt(rain_ma3_mse) 
rain_ma3_rmse #3.5919675916046803

#Another method to find RMSE
rain_ma3 = rain.rolling(window=3).mean()
rain_ma3 = rain_ma3.dropna()
rain_ma3_rmse = sqrt(mean_squared_error(rain[2:],rain_ma3))
print(rain_ma3_rmse) #3.5919675916046803

#Plotting acf & pacf 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(rain_ma3_res)
plot_pacf(rain_ma3_res)

#Applying autoarima
import pmdarima
from pmdarima import auto_arima
rain_mod1 = auto_arima(rain)
rain_mod1.summary()

'''                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  100
Model:                        SARIMAX   Log Likelihood                -285.245
Date:                Fri, 26 Mar 2021   AIC                            574.490
Time:                        00:52:16   BIC                            579.701
Sample:                             0   HQIC                           576.599
                                - 100                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept     24.8239      0.467     53.106      0.000      23.908      25.740
sigma2        17.5847      2.583      6.807      0.000      12.522      22.648
===================================================================================
Ljung-Box (L1) (Q):                   0.45   Jarque-Bera (JB):                 7.88
Prob(Q):                              0.50   Prob(JB):                         0.02
Heteroskedasticity (H):               1.29   Skew:                             0.67
Prob(H) (two-sided):                  0.47   Kurtosis:                         3.30
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).'''

#Residual given by the model
rain_mod1_res = rain_mod1.resid()
rain_mod1_res = pd.DataFrame(rain_mod1_res, index=rain.index)
rain_mod1_res

#Lineplot of residuals
plt.plot(rain_mod1_res)
plt.title('Rain - Residual of the Model')
plt.show()

#Histogram of residuals
plt.hist(rain_mod1_res)
plt.title('Rain - Residual of the Model')
plt.show()

#Plotting acf & pacf - residual
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(rain_mod1_res, title='Rain - Residual given by the Model - Autocorrelation') 
plot_pacf(rain_mod1_res, title='Rain - Residual given by the Model - Partial Autocorrelation')

#Squaring residuals/ errors
rain_mod1_se = pow(rain_mod1_res,2)
rain_mod1_se.head()

#average/mean of squared residuals/ errors
rain_mod1_mse = (rain_mod1_se.sum())/len(rain_mod1_se)
print(rain_mod1_mse) #18.62254

#Root of average/mean of squared residuals/ errors
rain_mod1_rmse = sqrt(rain_mod1_mse) 
print(rain_mod1_rmse) #4.315384119871696

#Forecasting next 8 periods
rain_mod1_pred = rain_mod1.predict(n_periods=8)
rain_mod1_pred

#Adding index and coverting to dataframe
rain_mod1_pred = pd.DataFrame(rain_mod1_pred, 
                              index=pd.date_range(start='1913-01-01', 
                                                  periods=8, freq='YS'))
rain_mod1_pred

#Plot Actual & forecast
plt.plot(rain)
plt.plot(rain_mod1_pred)
plt.legend(['Actual','Forecast'], bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Model with single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
rain_ses = SimpleExpSmoothing(rain).fit()
rain_ses.summary()

'''                       SimpleExpSmoothing Model Results                       
==============================================================================
Dep. Variable:                   rain   No. Observations:                  100
Model:             SimpleExpSmoothing   SSE                           1758.465
Optimized:                       True   AIC                            290.703
Trend:                           None   BIC                            295.913
Seasonal:                        None   AICC                           291.124
Seasonal Periods:                None   Date:                 Fri, 26 Mar 2021
Box-Cox:                        False   Time:                         00:58:24
Box-Cox Coeff.:                  None                                         
==============================================================================
                       coeff                 code              optimized      
------------------------------------------------------------------------------
smoothing_level           1.4901e-08                alpha                 True
initial_level              24.823898                  l.0                 True
------------------------------------------------------------------------------'''

#Residuals given by the model
rain_ses_res = rain_ses.resid
rain_ses_res

#Histogram of residuals
plt.hist(rain_ses_res)
plt.title('rain - Residual given by the model')
plt.show()

#Plotting acf & pacf - residual
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(rain_ses_res, lags=20, title='rain - Residual given by the model - Autocorrelation') 
plot_pacf(rain_ses_res, lags=20, title='rain - Residual given by the model - Partial Autocorrelation')

#Squaring residuals/ errors
rain_ses_se = pow(rain_ses_res,2)
rain_ses_se.head()

#average/mean of squared residuals/ errors
rain_ses_mse = (rain_ses_se.sum())/len(rain_ses_se)
print(rain_ses_mse) #17.584652052034816

#Root of average/mean of squared residuals/ errors
rain_ses_rmse = sqrt(rain_ses_mse)
print(rain_ses_rmse) #4.193405781943219

#forecasting next 8 periods
rain_pred = rain_ses.forecast(steps=8)
print(rain_pred)

#Plot Actual & Forecast
plt.plot(rain)
plt.plot(rain_pred)
plt.legend(['Actual','Forecast'], bbox_to_anchor=(1, 1), loc=2)
plt.show()