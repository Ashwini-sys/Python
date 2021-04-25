# God is Gracious!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#!pip install seaborn 
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
#%matplotlib inline
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

ss = pd.read_csv('D:/data _science/PYTHON/Time_Series_Python/Super_Store.csv')
#It is giving UnicodeDecodeError so adding encoding while reading the file

ss = pd.read_csv('D:/data _science/PYTHON/Time_Series_Python/Super_Store.csv', encoding='latin1', parse_dates=True)

'''UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9a in position 6: 
    invalid start byte - so used encoding='latin1'''
ss.shape #2121, 21
ss.info()
ss.head()
ss.isnull().sum() #No missing values

#Order Date
#Adjusting the variable name
ss = ss.rename(columns={"Order Date":"Order_Date"})
ss.info()

#Changing the datatype being its a date
ss['Order_Date'] = pd.to_datetime(ss['Order_Date'])
ss['Order_year'] = pd.DatetimeIndex(ss['Order_Date']).year #Extracting year values and creating a new variable year
ss['Order_month'] = pd.DatetimeIndex(ss['Order_Date']).month #Extracting month values and creating a new variable year
ss.info()
ss.shape #2121, 23
ss.Order_Date.head()
ss.Order_Date.describe()
'''
count                    2121
unique                    889
top       2016-09-05 00:00:00
freq                       10
first     2014-01-06 00:00:00
last      2017-12-30 00:00:00''' #Repeated order dates are there

#Indexing data with Order_Date
ss = ss.set_index('Order_Date')
ss.head()

#Variable 1 - Row ID
ss['Row ID'].describe()
'''
count    2121.000000
mean     5041.643564
std      2885.740258
min         1.000000
25%      2568.000000
50%      5145.000000
75%      7534.000000
max      9991.000000'''
ss['Row ID'].value_counts().count() #All are unique ID nummbers, no use with this variable

#Variable 2 - Order ID
#Adjusting the variable name
ss = ss.rename(columns={"Order ID":"Order_ID"})
ss.Order_ID.describe()
'''
count               2121
unique              1764
top       CA-2017-125451
freq                   4''' #Few order Id's are repeated else unique values

#Plot of Order Monthly wise
#Resample helps to filter data sec, min, hour, day, week, month, year wise 
#Resample works only when index is date format..
plt.plot(ss.Order_ID.resample('M').count())
plt.xticks(rotation=45)
plt.title('Orders Monthly wise')
plt.xlabel('Months')
plt.ylabel('No of Orders')
plt.show()

#Plot No of Orders monthly wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Order_month')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Order_month')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Order_month')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Order_month')['Order_ID'].count())
plt.xlabel('Months')
plt.ylabel('No of Orders')
plt.title('No of Orders monthly wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 3 - Ship Date
#Adjusting the variable name
ss = ss.rename(columns={"Ship Date":"Ship_Date"})

#Changing the datatype being its a date
ss['Ship_Date'] = pd.to_datetime(ss['Ship_Date'])
ss.info()
ss.Ship_Date.head()
ss.Ship_Date.describe()
'''
count                    2121
unique                    960
top       2015-12-16 00:00:00
freq                       10
first     2014-01-10 00:00:00
last      2018-01-05 00:00:00''' #Repeated order dates are there

#Histogram
plt.hist(ss.Ship_Date)
plt.xticks(rotation=45)
plt.xlabel('Ship_Date')
plt.ylabel('No or Orders')
plt.title('No of Orders vs Ship_Date')
plt.show()

#Plot of Order Shipped Monthly wise
plt.plot(ss.Ship_Date.resample('M').count())
plt.xticks(rotation=45)
plt.title('Orders Shipped Monthly wise')
plt.show() #Similar plot as order Id

# Variable 4 - Ship Mode
#Adjusting the variable name
ss = ss.rename(columns={"Ship Mode":"Ship_Mode"})
ss.info()

ss.Ship_Mode.describe()
'''
count               2121
unique                 4
top       Standard Class
freq                1248''' 

ss.Ship_Mode.value_counts()
'''
Standard Class    1248
Second Class       427
First Class        327
Same Day           119'''

ss.Ship_Mode.value_counts().sum() #2121

#Barplot
sns.countplot(x='Ship_Mode', data=ss)
plt.xticks(rotation=45)
plt.xlabel('Ship_Mode')
plt.ylabel('No or Orders')
plt.title('No of Orders vs Ship_Mode')
plt.show()

#line plot
plt.plot(ss.groupby('Ship_Mode')['Order_ID'].count())
plt.xlabel('Ship_Mode')
plt.ylabel('No or Orders')
plt.title('No of Orders vs Ship_Mode')
plt.show()

#Plot No of Orders Ship_Mode wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Ship_Mode')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Ship_Mode')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Ship_Mode')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Ship_Mode')['Order_ID'].count())
plt.xlabel('Ship_Mode')
plt.ylabel('No of Orders')
plt.title('No of Orders Ship_Mode wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 5 - Customer ID
#Adjusting the variable name
ss = ss.rename(columns={"Customer ID":"Customer_ID"})
ss.info()

ss.Customer_ID.describe()
'''
count         2121
unique         707
top       SV-20365
freq            15''' 

ss.Customer_ID.value_counts() #707 different values 

ss.Customer_ID.value_counts().sum() #2121

#Histogram
plt.hist(ss.Customer_ID)
plt.xticks(rotation=90)
plt.xlabel('Customer_ID')
plt.ylabel('No or Orders')
plt.title('No of Orders vs Customer_ID')
plt.show()

#Variable 6 - Customer Name
#Adjusting the variable name
ss = ss.rename(columns={"Customer Name":"Customer_Name"})
ss.info()

ss.Customer_Name.describe()
'''
count            2121
unique            707
top       Seth Vernon
freq               15''' 

ss.Customer_Name.value_counts() #707 different values 

ss.Customer_Name.value_counts().sum() #2121

#Histogram
plt.hist(ss.Customer_Name)
plt.xticks(rotation=90)
plt.xlabel('Customer_Name')
plt.ylabel('No or Orders')
plt.title('No of Orders vs Customer_Name')
plt.show()

#Customer Id and Customer Name gives the same details

#Variable 7 - Segment
ss.Segment.describe()
'''
count         2121
unique           3
top       Consumer
freq          1113'''

ss.Segment.value_counts()
'''
Consumer       1113
Corporate       646
Home Office     362'''

ss.Segment.value_counts().sum() #2121

#Barplot
sns.countplot(x = 'Segment', data=ss)
plt.xlabel('Segment', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Segment', fontsize=16)
plt.show()

#line plot
plt.plot(ss.groupby('Segment')['Order_ID'].count())
plt.xlabel('Segment')
plt.ylabel('No or Orders')
plt.title('No of Orders vs Segment')
plt.show()

#Plot No of Orders Segment wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Segment')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Segment')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Segment')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Segment')['Order_ID'].count())
plt.xlabel('Segment')
plt.ylabel('No of Orders')
plt.title('No of Orders Segment wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 8 - Country
ss.Country.describe()
'''
count              2121
unique                1
top       United States
freq               2121'''
#All the data belongs to one country United States

#Variable 9 - City
ss.City.describe()
'''
count              2121
unique              371
top       New York City
freq                192'''

ss.City.value_counts() #371 different cities

ss.City.value_counts().sum() #2121

#Lineplot
fig = plt.gcf() 
fig.set_size_inches(12,6)
plt.plot(ss.groupby('City')['Order_ID'].count())
plt.xlabel('City', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs City', fontsize=16)
plt.xticks(rotation=90)
plt.show()

#Countplot/ Barplot
fig = plt.gcf() 
fig.set_size_inches(12,6)
sns.countplot(x='City', data=ss)
plt.xlabel('City', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs City', fontsize=16)
plt.xticks(rotation=90)
plt.show()

#Variable 10 - State
ss.State.describe()
'''
count           2121
unique            48
top       California
freq             444'''

ss.State.value_counts() #48 different States

ss.State.value_counts().sum() #2121

#Barplot
sns.countplot(x='State', data=ss)
plt.xlabel('State', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs State', fontsize=16)
plt.xticks(rotation=90)
plt.show()

#Lineplot
plt.plot(ss.groupby('State')['Order_ID'].count())
plt.xlabel('State', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs State', fontsize=16)
plt.xticks(rotation=90)
plt.show()

#Plot No of Orders State wise in respective years
fig = plt.gcf() 
fig.set_size_inches(10,6)
plt.plot(ss[ss.Order_year==2014].groupby('State')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('State')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('State')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('State')['Order_ID'].count())
plt.xlabel('State')
plt.ylabel('No of Orders')
plt.title('No of Orders State wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.xticks(rotation=90)
plt.show()

#Variable 11 - Postal Code 
#Adjusting the variable name
ss = ss.rename(columns={"Postal Code":"Postal_Code"})
ss.info()

ss.Postal_Code.describe()
'''
count     2121.000000
mean     55726.556341
std      32261.888225
min       1040.000000
25%      22801.000000
50%      60505.000000
75%      90032.000000
max      99301.000000'''

ss.Postal_Code.value_counts() #454 different Postal codes

ss.Postal_Code.value_counts().sum() #2121

#Histogram - Though postal codes are not continuous, we have just made histogram
#for seeing how it looks like
plt.hist(ss.Postal_Code)
plt.xlabel('Postal_Code', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Postal_Code', fontsize=16)
plt.show()

#Lineplot
plt.plot(ss.groupby('Postal_Code')['Order_ID'].count())
plt.xlabel('Postal_Code', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Postal_Code', fontsize=16)
plt.show()

#Plot No of Orders Postal_Code wise in respective years
fig = plt.gcf() 
fig.set_size_inches(10,6)
plt.plot(ss[ss.Order_year==2014].groupby('Postal_Code')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Postal_Code')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Postal_Code')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Postal_Code')['Order_ID'].count())
plt.xlabel('Postal_Code')
plt.ylabel('No of Orders')
plt.title('No of Orders Postal_Code wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.xticks(rotation=90)
plt.show()

#Variable 12 - Region
ss.Region.describe()
'''
count     2121
unique       4
top       West
freq       707'''

ss.Region.value_counts() 
'''
West       707
East       601
Central    481
South      332'''

ss.Region.value_counts().sum() #2121

#Barplot
sns.countplot(x='Region', data=ss)
plt.xlabel('Region', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Region', fontsize=16)
plt.show()

#Lineplot
plt.plot(ss.groupby('Region')['Order_ID'].count())
plt.xlabel('Region', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Region', fontsize=16)
plt.show()

#Plot No of Orders Region wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Region')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Region')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Region')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Region')['Order_ID'].count())
plt.xlabel('Region')
plt.ylabel('No of Orders')
plt.title('No of Orders Region wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 13 - Product ID
#Adjusting the variable name
ss = ss.rename(columns={"Product ID":"Product_ID"})
ss.info()

ss.Product_ID.describe()
'''
count                2121
unique                375
top       FUR-FU-10004270
freq                   16'''

ss.Product_ID.value_counts() #375 different States

ss.Product_ID.value_counts().sum() #2121

#Histogram
plt.hist(ss.Product_ID)
plt.xlabel('Product_ID', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Product_ID', fontsize=16)
plt.show()

#Lineplot
plt.plot(ss.groupby('Product_ID')['Order_ID'].count())
plt.xlabel('Product_ID', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Product_ID', fontsize=16)
plt.show()

#Variable 14 - Category
ss.Category.describe()
'''
count          2121
unique            1
top       Furniture
freq           2121'''
#The data is having only one category no much information from this variable 

#Variable 15 - Sub-Category
#Adjusting the variable name
ss = ss.rename(columns={"Sub-Category":"Sub_Category"})
ss.info()

ss.Sub_Category.describe()
'''
count            2121
unique              4
top       Furnishings
freq              957'''

ss.Sub_Category.value_counts()
'''
Furnishings    957
Chairs         617
Tables         319
Bookcases      228'''

ss.Sub_Category.value_counts().sum() #2121

#Barplot
sns.countplot(x='Sub_Category', data=ss)
plt.xlabel('Sub_Category', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Sub_Category', fontsize=16)
plt.show()

#Lineplot
plt.plot(ss.groupby('Sub_Category')['Order_ID'].count())
plt.xlabel('Sub_Category', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Sub_Category', fontsize=16)
plt.show()

#Plot No of Orders Sub_Category wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Sub_Category')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Sub_Category')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Sub_Category')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Sub_Category')['Order_ID'].count())
plt.xlabel('Sub_Category')
plt.ylabel('No of Orders')
plt.title('No of Orders Sub_Category wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 16 - Product Name
#Adjusting the variable name
ss = ss.rename(columns={"Product Name":"Product_Name"})
ss.info()

ss.Product_Name.describe()
'''
count                           2121
unique                           380
top       KI Adjustable-Height Table
freq                              18'''

ss.Product_Name.value_counts() #380 Different product names

ss.Product_Name.value_counts().sum() #2121

#Lineplot
fig = plt.gcf() 
fig.set_size_inches(15,8)
plt.plot(ss.groupby('Product_Name')['Order_ID'].count())
plt.xlabel('Product_Name', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Product_Name', fontsize=16)
plt.xticks(rotation=90)
plt.show()

#Variable 17 - Sales
ss.Sales.describe()
'''
count    2121.000000
mean      349.834887
std       503.179145
min         1.892000
25%        47.040000
50%       182.220000
75%       435.168000
max      4416.174000'''

ss.Sales.value_counts() #1636 different Sales value

ss.Sales.value_counts().sum() #2121

#Plot Sales Value Monthly wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Order_month')['Sales'].sum())
plt.plot(ss[ss.Order_year==2015].groupby('Order_month')['Sales'].sum())
plt.plot(ss[ss.Order_year==2016].groupby('Order_month')['Sales'].sum())
plt.plot(ss[ss.Order_year==2017].groupby('Order_month')['Sales'].sum())
plt.xlabel('Months')
plt.ylabel('Sum Sales value')
plt.title('Sum Sales Value Monthly wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 18 - Quantity
ss.Quantity.describe()
'''
count    2121.000000
mean        3.785007
std         2.251620
min         1.000000
25%         2.000000
50%         3.000000
75%         5.000000
max        14.000000'''

ss.Quantity.value_counts() #14 different quantities

ss.Quantity.value_counts().sum() #2121

#Barplot
sns.countplot(x='Quantity', data=ss)
plt.xlabel('Quantity', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Quantity', fontsize=16)
plt.show()

#Lineplot
plt.plot(ss.groupby('Quantity')['Order_ID'].count())
plt.xlabel('Quantity', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Quantity', fontsize=16)
plt.show()

#Plot No of Orders Quantity wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Quantity')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Quantity')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Quantity')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Quantity')['Order_ID'].count())
plt.xlabel('Quantity')
plt.ylabel('No of Orders')
plt.title('No of Orders Quantity wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Plot Sales value Quantity wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Quantity')['Sales'].sum())
plt.plot(ss[ss.Order_year==2015].groupby('Quantity')['Sales'].sum())
plt.plot(ss[ss.Order_year==2016].groupby('Quantity')['Sales'].sum())
plt.plot(ss[ss.Order_year==2017].groupby('Quantity')['Sales'].sum())
plt.xlabel('Quantity')
plt.ylabel('Sum Sale Value')
plt.title('Sum Sales Value Quantity wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Plot Sales value Order_month wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Order_month')['Sales'].sum())
plt.plot(ss[ss.Order_year==2015].groupby('Order_month')['Sales'].sum())
plt.plot(ss[ss.Order_year==2016].groupby('Order_month')['Sales'].sum())
plt.plot(ss[ss.Order_year==2017].groupby('Order_month')['Sales'].sum())
plt.xlabel('Order_month')
plt.ylabel('Sum Sale Value')
plt.title('Sum Sales Value Order_month wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 19 - Discount
ss.Discount.describe()
'''
count    2121.000000
mean        0.173923
std         0.181547
min         0.000000
25%         0.000000
50%         0.200000
75%         0.300000
max         0.700000'''

ss.Discount.value_counts() #11 different discounts

ss.Discount.value_counts().sum() #2121

#Barplot
sns.countplot(x='Discount', data=ss)
plt.xlabel('Discount', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Discount', fontsize=16)
plt.show()

#Lineplot
plt.plot(ss.groupby('Discount')['Order_ID'].count())
plt.xlabel('Discount', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Discount', fontsize=16)
plt.show()

#Plot No of Orders Discount wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Discount')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Discount')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Discount')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Discount')['Order_ID'].count())
plt.xlabel('Discount')
plt.ylabel('No of Orders')
plt.title('No of Orders Discount wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Plot Sales value Discount wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Discount')['Sales'].sum())
plt.plot(ss[ss.Order_year==2015].groupby('Discount')['Sales'].sum())
plt.plot(ss[ss.Order_year==2016].groupby('Discount')['Sales'].sum())
plt.plot(ss[ss.Order_year==2017].groupby('Discount')['Sales'].sum())
plt.xlabel('Discount')
plt.ylabel('Sum Sale Value')
plt.title('Sum Sales Value Discount wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Variable 20 - Profit
ss.Profit.describe()
'''
count    2121.000000
mean        8.699327
std       136.049246
min     -1862.312400
25%       -12.849000
50%         7.774800
75%        33.726600
max      1013.127000'''

ss[ss.Profit>=0].Profit.value_counts().sum() #1407 Positive values ie profit
ss[ss.Profit<0].Profit.value_counts().sum() #714 negative values ie loss

ss.Profit.value_counts().sum() #2121

#Lineplot
plt.plot(ss.groupby('Profit')['Order_ID'].count())
plt.xlabel('Profit', fontsize=14)
plt.ylabel('No of Orders', fontsize=14)
plt.title('No of Orders vs Profit', fontsize=16)
plt.show()

#Plot No of Orders Profit wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Profit')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2015].groupby('Profit')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2016].groupby('Profit')['Order_ID'].count())
plt.plot(ss[ss.Order_year==2017].groupby('Profit')['Order_ID'].count())
plt.xlabel('Profit')
plt.ylabel('No of Orders')
plt.title('No of Orders Profit wise in respective years')
plt.legend([2014,2015,2016,2017])
plt.show()

#Plot No of Orders Profit wise in respective years
plt.plot(ss[ss.Order_year==2014].groupby('Order_month')['Profit'].sum())
plt.plot(ss[ss.Order_year==2015].groupby('Order_month')['Profit'].sum())
plt.plot(ss[ss.Order_year==2016].groupby('Order_month')['Profit'].sum())
plt.plot(ss[ss.Order_year==2017].groupby('Order_month')['Profit'].sum())
plt.xlabel('Profit')
plt.ylabel('No of Orders')
plt.title('No of Orders Profit wise in respective years')
plt.legend([2014,2015,2016,2017],bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Building model on Sales
#Getting monthly wise sum data
ss_sales =  pd.DataFrame(ss.Sales.resample('M').sum())  
ss_sales.shape #48 observations

#Plot
plt.plot(ss_sales, 'r')
plt.xticks(rotation=45)
plt.title('Monthlywise Sales Data')
plt.xlabel('Months')
Plt.ylabel('Sales Value')
plt.show()

#Decompose with multiplicative model
from statsmodels.tsa.seasonal import seasonal_decompose
ss_sales_dec_m = seasonal_decompose(ss_sales, model='mul')
ss_sales_dec_m.plot() #Trend & seasonality can be observed
ss_sales_dec_m.observed 
ss_sales_dec_m.trend
ss_sales_dec_m.seasonal
ss_sales_dec_m.resid

#Decompose with additive model
from statsmodels.tsa.seasonal import seasonal_decompose
ss_sales_dec_a = seasonal_decompose(ss_sales, model='add')
ss_sales_dec_a.plot() #Trend & seasonality can be observed
ss_sales_dec_a.observed
ss_sales_dec_a.trend
ss_sales_dec_a.seasonal
ss_sales_dec_a.resid

#Test for stationarity
from statsmodels.tsa.stattools import adfuller
ss_sales_adf = adfuller(ss_sales)

print('ADF Statistic: %f' % ss_sales_adf[0])
print('p-value: %f' % ss_sales_adf[1])
print('Critical Values:')
for key, value in ss_sales_adf[4].items():
    print('\t%s: %.3f' % (key, value))
'''
ADF Statistic: -4.699026
p-value: 0.000085
Critical Values:
	1%: -3.578
	5%: -2.925
	10%: -2.601'''
#p-value: 0.000085 ie < 0.05, Null Hypothesis is rejected, so, Data is stationary
#H0 data is not stationary

#Applying auto arima method for prediction
from pmdarima import auto_arima

ss_sales_aa = auto_arima(ss_sales)
ss_sales_aa.summary()    
'''
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                   48
Model:               SARIMAX(1, 0, 0)   Log Likelihood                -502.820
Date:                Wed, 24 Mar 2021   AIC                           1011.640
Time:                        16:54:58   BIC                           1017.253
Sample:                             0   HQIC                          1013.761
                                 - 48                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept   1.084e+04   2695.066      4.021      0.000    5554.237    1.61e+04
ar.L1          0.3056      0.131      2.328      0.020       0.048       0.563
sigma2      7.318e+07      0.160   4.56e+08      0.000    7.32e+07    7.32e+07
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                 3.70
Prob(Q):                              0.98   Prob(JB):                         0.16
Heteroskedasticity (H):               1.88   Skew:                             0.64
Prob(H) (two-sided):                  0.22   Kurtosis:                         2.54
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
[2] Covariance matrix is singular or near-singular, with condition number 1.38e+24. Standard errors may be unstable.'''

#Residuals/Errors
ss_sales_res = pd.DataFrame(ss_sales_aa.resid(), index=ss_sales.index)
ss_sales_res 

#Model values - Fitted values
ss_sales_aa_v = pd.DataFrame(ss_sales_aa.predict_in_sample(),index=ss_sales.index)
ss_sales_aa_v

#Lineplot
ss_sales_res.plot()
plt.title('Line plot of Residuals')

#Histogram of residuals
plt.hist(ss_sales_res)
plt.title('Histogram of Residuals')

#Density plot
ss_sales_res.plot(kind='kde')
plt.title('Density plot of Residuals')

#Plotting acf & pacf for residuals
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(ss_sales_res, lags=20) 
plot_pacf(ss_sales_res, lags=20)

#Squaring residuals/ errors
ss_sales_se = pow(ss_sales_res,2)
ss_sales_se.head()

#average/mean of squared residuals/ errors
ss_sales_mse = (ss_sales_se.sum())/len(ss_sales_se)
print(ss_sales_mse) #7.353159e+07

#Root of average/mean of squared residuals/ errors
ss_sales_rmse = sqrt(ss_sales_mse) 
ss_sales_rmse #8575.056439986281

#Plot comparision Actual, Model Values & Residuals
plt.plot(ss_sales)
plt.plot(ss_sales_aa_v, 'g')
plt.plot(ss_sales_res, 'r')
plt.legend(['Actual','Model Values', 'Residuals'],
           bbox_to_anchor=(1, 1), loc=2)
plt.xticks(rotation=45)
plt.show()

#Predict
ss_sales_pred = ss_sales_aa.predict(n_periods=12) 
ss_sales_pred = pd.DataFrame(ss_sales_pred, 
                             index=pd.date_range(start='2018-01-31', 
                                                 end='2018-12-31', freq='M'))

#Plot comparision Actual, Model Values & Forecast
plt.plot(ss_sales)
plt.plot(ss_sales_aa_v)
plt.plot(ss_sales_pred)
plt.legend(['Actual','Model Values', 'Forecast'],
           bbox_to_anchor=(1, 1), loc=2)
plt.xticks(rotation=45)
plt.show()

#Model with triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
ss_sales_es = ExponentialSmoothing(ss_sales, seasonal_periods=12,
                                 trend='add', seasonal='add').fit()
ss_sales_es.summary()
'''
                       ExponentialSmoothing Model Results                       
================================================================================
Dep. Variable:                    Sales   No. Observations:                   48
Model:             ExponentialSmoothing   SSE                      587001664.255
Optimized:                         True   AIC                            815.328
Trend:                         Additive   BIC                            845.267
Seasonal:                      Additive   AICC                           838.914
Seasonal Periods:                    12   Date:                 Wed, 24 Mar 2021
Box-Cox:                          False   Time:                         17:49:09
Box-Cox Coeff.:                    None                                         
=================================================================================
                          coeff                 code              optimized      
---------------------------------------------------------------------------------
smoothing_level               0.0050000                alpha                 True
smoothing_trend                  0.0001                 beta                 True
smoothing_seasonal            0.4619643                gamma                 True
initial_level                 7892.3104                  l.0                 True
initial_trend                 92.537388                  b.0                 True
initial_seasons.0            -1649.7854                  s.0                 True
initial_seasons.1            -6052.6524                  s.1                 True
initial_seasons.2             6681.6456                  s.2                 True
initial_seasons.3             52.526600                  s.3                 True
initial_seasons.4            -979.52340                  s.4                 True
initial_seasons.5             5313.8152                  s.5                 True
initial_seasons.6             2928.7406                  s.6                 True
initial_seasons.7            -571.96390                  s.7                 True
initial_seasons.8             15924.170                  s.8                 True
initial_seasons.9             4411.9366                  s.9                 True
initial_seasons.10            13672.562                 s.10                 True
initial_seasons.11            22753.656                 s.11                 True
---------------------------------------------------------------------------------'''

#Residuals
ss_sales_res1 = pd.DataFrame(ss_sales_es.resid)
ss_sales_res1

#Lineplot
ss_sales_res1.plot()
plt.title('Line plot of Residuals')

#Histogram of residuals
plt.hist(ss_sales_res1)
plt.title('Histogram of Residuals')

#Density plot
ss_sales_res1.plot(kind='kde')
plt.title('Density plot of Residuals')

#Plotting acf & pacf for residuals
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(ss_sales_res1, lags=20) 
plot_pacf(ss_sales_res1, lags=20)

#Squaring residuals/ errors
ss_sales_se1 = pow(ss_sales_res1,2)
ss_sales_se1.head()

#average/mean of squared residuals/ errors
ss_sales_mse1 = (ss_sales_se1.sum())/len(ss_sales_se1)
print(ss_sales_mse1) #1.222920e+07

#Root of average/mean of squared residuals/ errors
ss_sales_rmse1 = sqrt(ss_sales_mse1) 
ss_sales_rmse1 #3497.0275004124182

#Plot Actual, Model Values and Residuals
plt.plot(ss_sales)
plt.plot(ss_sales_es.fittedvalues, 'g')
plt.plot(ss_sales_res1, 'r')
plt.legend(['Actual','Model Values', 'Residuals'])
plt.xticks(rotation=45)
plt.show()

#Predict/forecast
ss_sales_fore = ss_sales_es.forecast(12) 
ss_sales_fore

#Plot Actual, Model Values & forecast
plt.plot(ss_sales)
plt.plot(ss_sales_es.fittedvalues)
plt.plot(ss_sales_fore)
plt.legend(['Actual','Model Values', 'Predicted'],
           bbox_to_anchor=(1, 1), loc=2)
plt.xticks(rotation=45)
plt.show()

#Plot Actual, Model/Fitted Values & forecast Arima & Holt winter
plt.plot(ss_sales)
plt.plot(ss_sales_es.fittedvalues)
plt.plot(ss_sales_aa_v)
plt.plot(ss_sales_fore)
plt.plot(ss_sales_pred)
plt.legend(['Actual','Fitted Values - Holt', 'Fitted Values -Arima', 
            'Predicted Holt', 'Predicted Arima'], bbox_to_anchor=(1, 1), loc=2)
plt.xticks(rotation=45)
plt.show()