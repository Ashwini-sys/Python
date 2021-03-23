# God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

#df5 = pd.read_csv('df5.csv') # after bedroomAbgr, written as df5

df6 = pd.read_csv('D:/data _science/PYTHON/Linear_Regression_Python/df5.csv')
df6.info() 

#________23rd int64, TotRmsAbvGrd, #58
df6.TotRmsAbvGrd.describe() # no missing, catg
'''
count    1995.000000
mean        6.449123
std         1.550162
min         2.000000
25%         5.000000
50%         6.000000
75%         7.000000
max        14.000000
Name: TotRmsAbvGrd, dtype: float64
'''
df6.TotRmsAbvGrd.value_counts() #12 catgs 
sum(df6.TotRmsAbvGrd.value_counts()) # 1995
'''
6     578
7     444
5     391
8     258
4     127
9      94
10     47
3      22
11     19
12     13
14      1
2       1
Name: TotRmsAbvGrd, dtype: int64
'''
# 2 to 6 as 0, 7 and above as 1
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(2,0)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(3,0)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(4,0)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(5,0)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(6,0)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(7,1)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(8,1)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(9,1)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(10,1)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(11,1)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(12,1)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(13,1)
df6['TotRmsAbvGrd']=df6.get('TotRmsAbvGrd').replace(14,1)
df6.TotRmsAbvGrd.value_counts() #1119 and 876
sum(df6.TotRmsAbvGrd.value_counts()) # 1995

#______is this a good predictor?
# Indpndnt sample t test
tragr_0 = df6[df6.TotRmsAbvGrd == 0]
tragr_1 = df6[df6.TotRmsAbvGrd == 1]
import scipy
scipy.stats.ttest_ind(tragr_0.Property_Sale_Price, tragr_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

df6.info()
#________24th int64, Fireplaces, #60
df6.Fireplaces.describe() # no missing, ordinal
df6.Fireplaces.value_counts() #987,870,135,3
sum(df6.Fireplaces.value_counts()) # 1995
'''
as there is an order, we will treat this as continuous!
and tolearte 3 counts in 3 fireplaces, corr is also good!
'''
# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6.Fireplaces) # 0.46, good! 

df6.info()
#________25th int64, GarageCars, #65
df6.GarageCars.describe() # no missing, ordinal
sum(df6.GarageCars.value_counts()) # 1995
df6.GarageCars.value_counts()
'''
2    1179
1     510
3     187
0     113
4       6
Name: GarageCars, dtype: int64
'''
# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6.GarageCars) # 0.62, good! 

df6.info()
#________26th int64, GarageArea, #66
df6.GarageArea.describe() # no missing, cont
'''
count    1995.000000
mean      461.745865
std       203.638299
min         0.000000
25%       324.500000
50%       472.000000
75%       574.000000
max      1418.000000
Name: GarageArea, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df6.GarageArea, bins = 'auto', facecolor = 'red')
plt.xlabel('GarageArea')
plt.ylabel('counts')
plt.title('Histogram of GarageArea') # looks good!
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df6['GarageArea'].plot.box(color=props2, patch_artist = True, vert = False)

#first, fix outliers 
# outliers counts on higher side
iqr_66 = 574-324
ul_66 = 574 + 1.5*iqr_66
ul_66 #949
len(df6.GarageArea[df6.GarageArea > ul_66]) # 15
'''
put OLiers on higher threshold
'''
df6.GarageArea[df6.GarageArea > ul_66] = ul_66
len(df6.GarageArea[df6.GarageArea > ul_66]) # now 0, smile!
# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6.GarageArea) # 0.61, good! 
'''
think, GarageArea and GarageCars are not same!!
'''
#GarageYrBlt
df6.GarageYrBlt.describe()
'''
count    1882.000000
mean     1979.562168
std        24.154421
min      1906.000000
25%      1962.000000
50%      1981.000000
75%      2002.000000
max      2019.000000
Name: GarageYrBlt, dtype: float64
'''
df6.GarageYrBlt.value_counts() #only 102 datapoints are their Ignore!!

df6.info()
#________27th int64, WoodDeckSF, #70
df6.WoodDeckSF.describe() # no missing, cont..to be 0 n 1

#______histogram
#_run in block
plt.hist(df6.WoodDeckSF, bins = 'auto', facecolor = 'red')
plt.xlabel('WoodDeckSF')
plt.ylabel('counts')
plt.title('Histogram of WoodDeckSF') # looks good!
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df6['WoodDeckSF'].plot.box(color=props2, patch_artist = True, vert = False)

df6.WoodDeckSF.value_counts() # 1073 are zeros, we will make in 2 catgs, 0 and 1

df6.info()
#________28th int64, OpenPorchSF, #71
df6.OpenPorchSF.describe() # no missing, cont to be cnvrt catg 0 and 1
'''
count    1995.000000
mean       44.396992
std        64.293848
min         0.000000
25%         0.000000
50%        22.000000
75%        63.000000
max       547.000000
Name: OpenPorchSF, dtype: float64
'''
df6.OpenPorchSF.value_counts() # 930 are zeros, we will make in 2 catgs, 0 and 1

df6.info()
#________29th int64, EnclosedPorch, #72, too many zeros
df6.EnclosedPorch.describe()
df6.EnclosedPorch.value_counts() #1717 zeros....IGNORE

df6.info()
#________30th int64, 3SsnPorch, #73, too many zeros
df6['3SsnPorch'].describe() # Too many zeros
df6['3SsnPorch'].value_counts() # 1966 =0, IGNORE

#________31st int64, ScreenPorch, #74, too many zeros
df6['ScreenPorch'].describe() # Too many zeros
df6['ScreenPorch'].value_counts() # 1845 =0, IGNORE
df6.info()
#________32nd int64, PoolArea, #75, too many zeros
df6['PoolArea'].describe() # Too many zeros
df6['PoolArea'].value_counts() # 1988 =0, IGNORE

df6.info()
#________33rd int64, MiscVal, #79, too many zeros
df6['MiscVal'].describe() # Too many zeros
df6['MiscVal'].value_counts() # 1918 =0, IGNORE

df6.info()
#________34th int64, MoSold, #80
df6['MoSold'].describe() 
#______histogram
#_run in block
plt.hist(df6.MoSold, bins = 'auto', facecolor = 'red')
plt.xlabel('MoSold')
plt.ylabel('counts')
plt.title('Histogram of MoSold') # looks good!
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df6['MoSold'].plot.box(color=props2, patch_artist = True, vert = False)

# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6.MoSold) # 0.05, IGNORE! 

df6.info()
#________35th int64, YrSold, #81
df6['YrSold'].describe() 
'''
count    1995.000000
mean     2007.855639
std         1.327214
min      2006.000000
25%      2007.000000
50%      2008.000000
75%      2009.000000
max      2010.000000
Name: YrSold, dtype: float64
'''
df6['YrSold'].value_counts()
#______histogram
#_run in block
plt.hist(df6.YrSold, bins = 'auto', facecolor = 'red')
plt.xlabel('YrSold')
plt.ylabel('counts')
plt.title('Histogram of YrSold') # looks good!
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df6['YrSold'].plot.box(color=props2, patch_artist = True, vert = False)

# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6.YrSold) # 0.002, IGNORE! 

df6.info()
#________14TH INT #48, 2ndFlrSF
df6['2ndFlrSF'].describe() # no missing, continuous
df6['2ndFlrSF'].value_counts() # 1156 = zero; Let's make it 0 n 1

df6.insert(49, '2ndFlrSF_catg', df6['2ndFlrSF']) #copying & inserting
df6
df6['2ndFlrSF_catg'][df6['2ndFlrSF_catg'] > 0] = 'flr2sf'
df6['2ndFlrSF_catg'][df6['2ndFlrSF_catg'] == 0] = 'flr2sf_nil'
df6['2ndFlrSF_catg'].head()
df6['2ndFlrSF_catg'].value_counts() # 1156, 839

#______is this a good predictor?
# Indpndnt sample t test
flr2SF_0 = df6[df6['2ndFlrSF_catg'] == 'flr2sf_nil']
flr2SF_1 = df6[df6['2ndFlrSF_catg'] == 'flr2sf']
import scipy
scipy.stats.ttest_ind(flr2SF_0.Property_Sale_Price, flr2SF_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6['2ndFlrSF']) # 0.33! 

df6.info()
#________27TH INT #48, WoodDeckSF #71
df6['WoodDeckSF'].describe() # no missing, continuous
df6['WoodDeckSF'].value_counts() # 1156 = zero

df6.insert(72, 'woodDeckSF_catg', df6['WoodDeckSF'])
df6['woodDeckSF_catg'][df6['woodDeckSF_catg'] > 0] = 'woodDeck'
df6['woodDeckSF_catg'][df6['woodDeckSF_catg'] == 0] = 'woodDeck_nil'
df6['woodDeckSF_catg'].head()
df6['woodDeckSF_catg'].value_counts() # 1073, 922

#______is this a good predictor?
# Indpndnt sample t test
woodDeck_0 = df6[df6['woodDeckSF_catg'] == 'woodDeck_nil']
woodDeck_1 = df6[df6['woodDeckSF_catg'] == 'woodDeck']
import scipy
scipy.stats.ttest_ind(woodDeck_0.Property_Sale_Price, woodDeck_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6['WoodDeckSF']) # 0.31! 


df6.info()
#________28TH INT #48, OpenPorchSF #73
df6['OpenPorchSF'].describe() # no missing, continuous
df6['OpenPorchSF'].value_counts() # 930 = zero

df6.insert(74, 'OpenPorchSF_catg', df6['OpenPorchSF'])
df6['OpenPorchSF_catg'][df6['OpenPorchSF_catg'] > 0] = 'OpenPorch'
df6['OpenPorchSF_catg'][df6['OpenPorchSF_catg'] == 0] = 'OpenPorch_nil'
df6['OpenPorchSF_catg'].head()
df6['OpenPorchSF_catg'].value_counts() # 1065, 930

#______is this a good predictor?
# Indpndnt sample t test
OpenPorch_0 = df6[df6['OpenPorchSF_catg'] == 'OpenPorch_nil']
OpenPorch_1 = df6[df6['OpenPorchSF_catg'] == 'OpenPorch']
import scipy
scipy.stats.ttest_ind(OpenPorch_0.Property_Sale_Price, OpenPorch_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

# good or bad for our predictive modeling?
np.corrcoef(df6.Property_Sale_Price, df6['OpenPorchSF']) # 0.36! 

df6.to_csv('df6.csv')
'''
'Zone_Class','Property_Shape','LotConfig','Condition1','Dwelling_Type','HouseStyle',
'RoofStyle','Exterior1st','MasVnrType','ExterQual','Foundation','BsmtFinType1',
'HeatingQC','Bsmexposure','KitchenQual','GarageType','GarageFinish'
'''
## Creating Dummies 
cat_vars=['Zone_Class','Property_Shape','Utilities','LotConfig','Neighborhood',
          'Condition1','Dwelling_Type','HouseStyle','RoofStyle','Exterior1st',
          'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',
          'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
          'Electrical','GarageFinish']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df6[var], prefix=var)
    data1=df6.join(cat_list)
    df6=data1

#
#20 Categorical
import statsmodels
from statsmodels.formula.api import ols
fit = ols('''Property_Sale_Price ~ C(Zone_Class)+C(Property_Shape)+C(LotConfig)
          +C(Neighborhood)+C(Condition1)+C(Dwelling_Type)+C(HouseStyle)
          +C(RoofStyle)+C(Exterior1st)+C(MasVnrType)+C(ExterQual)+C(Foundation)
          +C(BsmtQual)+C(BsmtExposure)+C(BsmtFinType1)+C(HeatingQC)+C(KitchenQual)
          +C(GarageType)+C(GarageFinish)+C(SaleCondition)''', data=df6).fit()
          
fit.summary()
#---------------------------------------------------------------------------------

#Training and testing data spliting
y= df6.Property_Sale_Price
x=df6.drop('Property_Sale_Price',axis=1)

import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

#Model 
from sklearn import linear_model, metrics 
x = df6.iloc[:, [6, 7, 8, 11,12,13,16,17,19,20,21,23,24,25,27,29,31,32,33,34,35,36,37,38,39,41,42,43,44,46,47,48,50,51,53,54,55,57,58,61,64,65,66,71]]
print(x)
# output 
y = df6.iloc[:, 84] #84 = Property_Sale_Price
print(y)
reg = linear_model.LinearRegression() 
reg.fit(x, y) 
reg.coef_
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(x, y))) 
  


#need to ask
df4 = pd.read_csv('D:/data _science/PYTHON/Linear_Regression_Python/df3.csv')
df4.info() 
############______________correlation 
a = pd.DataFrame(df4.corr())
a = a.Property_Sale_Price[a.Property_Sale_Price>=0.5]
len(a) #21

