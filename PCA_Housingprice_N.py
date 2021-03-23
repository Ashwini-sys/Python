#God is righteous

import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing

hs = pd.read_csv('D:/data _science/PYTHON/PCA_Python/houseprice.csv')
hs.shape #1460, 81
hs.info()

#For linear Regression model studied all these variables individually &
#selected 28 variables as good predictors
'''SalePrice ~ MSZoning LotShape Neighborhood Condition1 BldgType RoofStyle 
MasVnrType ExterQual BsmtQual BsmtExposure BsmtFinType1 HeatingQC KitchenQual 
GarageType GarageFinish LotArea OverallQual YearBuilt YearRemodAdd BsmtFinSF1 
TotalBsmtSF 1stFlrSF GrLivArea TotRmsAbvGrd Fireplaces GarageCars WoodDeckSF 
OpenPorchSF'''

#Selecting the above mentioned variables
hs = hs.iloc[:,[2,7,12,13,15,21,25,27,30,32,33,40,53,58,60,4,17,19,20,34,38,43,46,54,56,61,66,67,80]]
hs.info()

#Saving the file with selected variables
hs.to_csv('houseprice_selected.csv', index=False)

#Loading the new file
hs =pd.read_csv('houseprice_selected.csv')
hs.shape #1460, 29
hs.info()

#Verifying the missing values
hs.isnull().sum() #There are missing values
hs = hs.dropna()
hs.shape #1340, 29
(1460-1340)/1460 #8% of data lost while removing missing values

#SalePrice
hs.SalePrice.describe()
'''
count      1340.000000
mean     186819.973881
std       78901.387378
min       35311.000000
25%      135000.000000
50%      168500.000000
75%      220000.000000
max      755000.000000'''

#______histogram
#_run in block
plt.hist(hs.SalePrice, bins = 'auto', facecolor = 'green')
plt.xlabel('SalePrice')
plt.ylabel('counts')
plt.title('Histogram of SalePrice')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['SalePrice'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr = hs.SalePrice.describe()['75%'] - hs.SalePrice.describe()['25%'] #85000.0
up_lim = hs.SalePrice.describe()['75%']+1.5*iqr # 347500.0
len(hs.SalePrice[hs.SalePrice > up_lim]) #56 outliers
up_lim_ext = hs.SalePrice.describe()['75%']+3*iqr # 475000.0
len(hs.SalePrice[hs.SalePrice > up_lim_ext]) #10 Extreme outliers

#Removed extreme outliers - 10 Observations were removed
hs = hs[hs.SalePrice <= 475000]    

#Other outliers assgning with the upperlimit
len(hs.SalePrice[hs.SalePrice >= 347500]) #46 Outliers
hs.SalePrice[hs.SalePrice >= 347500] = 347500

#____boxplot - After removing/ adjusting the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['SalePrice'].plot.box(color=props2, patch_artist = True, vert = False) #Ignoring these outliers

#MSZoning
hs.MSZoning.describe()
'''
count     1330
unique       5
top         RL
freq      1058'''

hs.MSZoning.value_counts()
'''
RL         1058
RM          191
FV           62
RH           11
C (all)       8'''

hs.MSZoning.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='MSZoning', data=hs)
plt.xlabel('MSZoning', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of MSZoning', size=16)
plt.show()

#Coverting the data into 2 categories RL & RL_1
hs['MSZoning'] = hs.get('MSZoning').replace('RM','RL_1')
hs['MSZoning'] = hs.get('MSZoning').replace('FV','RL_1')
hs['MSZoning'] = hs.get('MSZoning').replace('RH','RL_1')
hs['MSZoning'] = hs.get('MSZoning').replace('C (all)','RL_1')

hs.MSZoning.value_counts()
'''
RL      1058
RL_1     272'''
hs.MSZoning.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='MSZoning', data=hs)
plt.xlabel('MSZoning', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of MSZoning', size=16)
plt.show()

#LotShape
hs.LotShape.describe()
'''
count     1330
unique       4
top        Reg
freq       827'''

hs.LotShape.value_counts()
'''
Reg    827
IR1    454
IR2     39
IR3     10'''

hs.LotShape.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='LotShape', data=hs)
plt.xlabel('LotShape', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of LotShape', size=16)
plt.show()

#Coverting the data into 2 categories RL & RL_1
hs['LotShape'] = hs.get('LotShape').replace('IR1','RL_1')
hs['LotShape'] = hs.get('LotShape').replace('IR2','RL_1')
hs['LotShape'] = hs.get('LotShape').replace('IR3','RL_1')

hs.LotShape.value_counts()
'''
Reg     827
RL_1    503'''

hs.LotShape.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='LotShape', data=hs)
plt.xlabel('LotShape', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of LotShape', size=16)
plt.show()

#Neighborhood
hs.Neighborhood.describe()
'''
count      1330
unique       25
top       NAmes
freq        209'''

hs.Neighborhood.value_counts()
'''
NAmes      209
CollgCr    146
OldTown    100
Somerst     83
Gilbert     77
NWAmes      73
NridgHt     72
Edwards     70
Sawyer      69
SawyerW     53
Crawfor     50
BrkSide     47
Mitchel     42
Timber      38
NoRidge     38
IDOTRR      29
ClearCr     26
StoneBr     22
SWISU       20
Blmngtn     17
BrDale      15
MeadowV     12
Veenker     11
NPkVill      9
Blueste      2'''

hs.Neighborhood.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='Neighborhood', data=hs)
plt.xlabel('Neighborhood', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of Neighborhood', size=16)
plt.xticks(rotation=90)
plt.show()

#Coverting the data into 2 categories nghb & nghb_1
hs['Neighborhood']=hs.get('Neighborhood').replace('NAmes','nghb')
hs['Neighborhood']=hs.get('Neighborhood').replace('CollgCr','nghb')
hs['Neighborhood']=hs.get('Neighborhood').replace('OldTown','nghb')
hs['Neighborhood']=hs.get('Neighborhood').replace('Edwards','nghb')
hs['Neighborhood']=hs.get('Neighborhood').replace('Somerst','nghb')
hs['Neighborhood']=hs.get('Neighborhood').replace('Gilbert','nghb')
hs['Neighborhood']=hs.get('Neighborhood').replace('Sawyer','nghb')
hs['Neighborhood']=hs.get('Neighborhood').replace('NWAmes','nghb')

hs['Neighborhood']=hs.get('Neighborhood').replace('SawyerW','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('BrkSide','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('Mitchel','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('Crawfor','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('NridgHt','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('NoRidge','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('IDOTRR','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('Timber','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('ClearCr','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('SWISU','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('Blmngtn','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('MeadowV','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('StoneBr','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('BrDale','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('NPkVill','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('Veenker','nghb_1')
hs['Neighborhood']=hs.get('Neighborhood').replace('Blueste','nghb_1')

hs.Neighborhood.value_counts()
'''
nghb      827
nghb_1    503'''

hs.Neighborhood.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='Neighborhood', data=hs)
plt.xlabel('Neighborhood', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of Neighborhood', size=16)
plt.show()

#Condition1
hs.Condition1.describe()
'''
count     1330
unique       9
top       Norm
freq      1154'''

hs.Condition1.value_counts()
'''
Norm      1154
Feedr       63
Artery      43
RRAn        26
PosN        19
RRAe        10
PosA         8
RRNn         5
RRNe         2'''

hs.Condition1.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='Condition1', data=hs)
plt.xlabel('Condition1', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of Condition1', size=16)
plt.show()

#Coverting the data into 2 categories norm & norm_1
hs['Condition1']=hs.get('Condition1').replace('Feedr','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('Artery','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('RRAn','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('PosN','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('Feedr','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('RRAe','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('PosA','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('RRNn','Norm_1')
hs['Condition1']=hs.get('Condition1').replace('RRNe','Norm_1')

hs.Condition1.value_counts()
'''
Norm      1154
Norm_1     176'''

hs.Condition1.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='Condition1', data=hs)
plt.xlabel('Condition1', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of Condition1', size=16)
plt.show()

#BldgType
hs.BldgType.describe()
'''
count     1330
unique       5
top       1Fam
freq      1130'''

hs.BldgType.value_counts()
'''
1Fam      1130
TwnhsE     112
Twnhs       38
Duplex      28
2fmCon      22'''

hs.BldgType.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BldgType', data=hs)
plt.xlabel('BldgType', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BldgType', size=16)
plt.show()

#Coverting the data into 2 categories Fam & Fam_1
hs['BldgType']=hs.get('BldgType').replace('1Fam','Fam')
hs['BldgType']=hs.get('BldgType').replace('TwnhsE','Fam_1')
hs['BldgType']=hs.get('BldgType').replace('Duplex','Fam_1')
hs['BldgType']=hs.get('BldgType').replace('Twnhs','Fam_1')
hs['BldgType']=hs.get('BldgType').replace('2fmCon','Fam_1')

hs.BldgType.value_counts()
'''
Fam      1130
Fam_1     200'''

hs.BldgType.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BldgType', data=hs)
plt.xlabel('BldgType', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BldgType', size=16)
plt.show()

#RoofStyle
hs.RoofStyle.describe()
'''
count      1330
unique        6
top       Gable
freq       1038'''

hs.RoofStyle.value_counts()
'''
Gable      1038
Hip         263
Flat         11
Gambrel      10
Mansard       6
Shed          2'''

hs.RoofStyle.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='RoofStyle', data=hs)
plt.xlabel('RoofStyle', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of RoofStyle', size=16)
plt.show()

#Coverting the data into 2 categories Gable & Gable_1
hs['RoofStyle']=hs.get('RoofStyle').replace('Hip','Gable_1')
hs['RoofStyle']=hs.get('RoofStyle').replace('Flat', 'Gable_1')
hs['RoofStyle']=hs.get('RoofStyle').replace('Gambrel', 'Gable_1')
hs['RoofStyle']=hs.get('RoofStyle').replace('Mansard', 'Gable_1')
hs['RoofStyle']=hs.get('RoofStyle').replace('Shed', 'Gable_1')

hs.RoofStyle.value_counts()
'''
Gable      1038
Gable_1     292'''

hs.RoofStyle.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='RoofStyle', data=hs)
plt.xlabel('RoofStyle', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of RoofStyle', size=16)
plt.show()

#MasVnrType
hs.MasVnrType.describe()
'''
count     1330
unique       4
top       None
freq       763'''

hs.MasVnrType.value_counts()
'''
None       763
BrkFace    430
Stone      122
BrkCmn      15'''

hs.MasVnrType.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='MasVnrType', data=hs)
plt.xlabel('MasVnrType', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of MasVnrType', size=16)
plt.show()

#Coverting the into 2 categories Vnr_none, Vnr_present
hs['MasVnrType']=hs.get('MasVnrType').replace('None','Vnr_none')
hs['MasVnrType']=hs.get('MasVnrType').replace('BrkFace','Vnr_present')
hs['MasVnrType']=hs.get('MasVnrType').replace('Stone', 'Vnr_present')
hs['MasVnrType']=hs.get('MasVnrType').replace('BrkCmn', 'Vnr_present')

hs.MasVnrType.value_counts()
'''
Vnr_none       763
Vnr_present    567'''

hs.MasVnrType.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='MasVnrType', data=hs)
plt.xlabel('MasVnrType', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of MasVnrType', size=16)
plt.show()

#ExterQual
hs.ExterQual.describe()
'''
count     1330
unique       4
top         TA
freq       804'''

hs.ExterQual.value_counts()
'''
TA    804
Gd    475
Ex     44
Fa      7'''

hs.ExterQual.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='ExterQual', data=hs)
plt.xlabel('ExterQual', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of ExterQual', size=16)
plt.show()

#Coverting the data into 2 categories TA, TA_1
hs['ExterQual']=hs.get('ExterQual').replace('Gd','TA_1')
hs['ExterQual']=hs.get('ExterQual').replace('Ex','TA_1')
hs['ExterQual']=hs.get('ExterQual').replace('Fa','TA_1')

hs.ExterQual.value_counts()
'''
TA      804
TA_1    526'''

hs.ExterQual.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='ExterQual', data=hs)
plt.xlabel('ExterQual', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of ExterQual', size=16)
plt.show()

#BsmtQual
hs.BsmtQual.describe()
'''
count     1330
unique       4
top         TA
freq       594'''

hs.BsmtQual.value_counts()
'''
TA    594
Gd    593
Ex    111
Fa     32'''

hs.BsmtQual.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BsmtQual', data=hs)
plt.xlabel('BsmtQual', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BsmtQual', size=16)
plt.show()

##Coverting the data into 2 categories TA, TA_1
hs['BsmtQual']=hs.get('BsmtQual').replace('Gd','TA_1')
hs['BsmtQual']=hs.get('BsmtQual').replace('Ex','TA_1')
hs['BsmtQual']=hs.get('BsmtQual').replace('Fa','TA_1')

hs.BsmtQual.value_counts()
'''
TA_1    736
TA      594'''

hs.BsmtQual.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BsmtQual', data=hs)
plt.xlabel('BsmtQual', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BsmtQual', size=16)
plt.show()

#BsmtExposure
hs.BsmtExposure.describe()
'''
count     1330
unique       4
top         No
freq       887'''

hs.BsmtExposure.value_counts()
'''
No    887
Av    211
Gd    121
Mn    111'''

hs.BsmtExposure.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BsmtExposure', data=hs)
plt.xlabel('BsmtExposure', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BsmtExposure', size=16)
plt.show()

#Coverting the data into 2 categories No & Yes
hs['BsmtExposure']=hs.get('BsmtExposure').replace('Av','Yes')
hs['BsmtExposure']=hs.get('BsmtExposure').replace('Gd','Yes')
hs['BsmtExposure']=hs.get('BsmtExposure').replace('Mn','Yes')

hs.BsmtExposure.value_counts()
'''
No     887
Yes    443'''

hs.BsmtExposure.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BsmtExposure', data=hs)
plt.xlabel('BsmtExposure', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BsmtExposure', size=16)
plt.show()

#BsmtFinType1
hs.BsmtFinType1.describe()
'''
count     1330
unique       6
top        GLQ
freq       396'''

hs.BsmtFinType1.value_counts()
'''
GLQ    396
Unf    391
ALQ    208
BLQ    141
Rec    125
LwQ     69'''

hs.BsmtFinType1.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BsmtFinType1', data=hs)
plt.xlabel('BsmtFinType1', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BsmtFinType1', size=16)
plt.show()

#Coverting the data into 2 categories Ungl & Ungl_1
hs['BsmtFinType1']=hs.get('BsmtFinType1').replace('GLQ','Ungl')
hs['BsmtFinType1']=hs.get('BsmtFinType1').replace('Unf','Ungl')
hs['BsmtFinType1']=hs.get('BsmtFinType1').replace('ALQ','Ungl_1')
hs['BsmtFinType1']=hs.get('BsmtFinType1').replace('BLQ','Ungl_1')
hs['BsmtFinType1']=hs.get('BsmtFinType1').replace('Rec','Ungl_1')
hs['BsmtFinType1']=hs.get('BsmtFinType1').replace('LwQ','Ungl_1')

hs.BsmtFinType1.value_counts()
'''
Ungl      787
Ungl_1    543'''

hs.BsmtFinType1.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='BsmtFinType1', data=hs)
plt.xlabel('BsmtFinType1', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of BsmtFinType1', size=16)
plt.show()

#HeatingQC
hs.HeatingQC.describe()
'''
count     1330
unique       5
top         Ex
freq       695'''

hs.HeatingQC.value_counts()
'''
Ex    695
TA    380
Gd    218
Fa     36
Po      1'''

hs.HeatingQC.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='HeatingQC', data=hs)
plt.xlabel('HeatingQC', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of HeatingQC', size=16)
plt.show()

#Coverting the data into 2 categories Ex, Ex_1
hs['HeatingQC']=hs.get('HeatingQC').replace('TA','Ex_1')
hs['HeatingQC']=hs.get('HeatingQC').replace('Gd','Ex_1')
hs['HeatingQC']=hs.get('HeatingQC').replace('Fa','Ex_1')
hs['HeatingQC']=hs.get('HeatingQC').replace('Po','Ex_1')

hs.HeatingQC.value_counts()
'''
Ex      695
Ex_1    635'''

hs.HeatingQC.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='HeatingQC', data=hs)
plt.xlabel('HeatingQC', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of HeatingQC', size=16)
plt.show()

#KitchenQual
hs.KitchenQual.describe()
'''
count     1330
unique       4
top         TA
freq       650'''

hs.KitchenQual.value_counts()
'''
TA    650
Gd    569
Ex     88
Fa     23'''

hs.KitchenQual.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='KitchenQual', data=hs)
plt.xlabel('KitchenQual', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of KitchenQual', size=16)
plt.show()

#Coverting the data into 2 categories TA & TA_1
hs['KitchenQual']=hs.get('KitchenQual').replace('Gd', 'TA_1')
hs['KitchenQual']=hs.get('KitchenQual').replace('Fa', 'TA_1')
hs['KitchenQual']=hs.get('KitchenQual').replace('Ex', 'TA_1')

hs.KitchenQual.value_counts()
'''
TA_1    680
TA      650'''

hs.KitchenQual.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='KitchenQual', data=hs)
plt.xlabel('KitchenQual', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of KitchenQual', size=16)
plt.show()

#GarageType
hs.GarageType.describe()
'''
count       1330
unique         6
top       Attchd
freq         847'''

hs.GarageType.value_counts()
'''
Attchd     847
Detchd     369
BuiltIn     82
Basment     19
CarPort      7
2Types       6'''

hs.GarageType.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='GarageType', data=hs)
plt.xlabel('GarageType', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of GarageType', size=16)
plt.show()

#Coverting the data into 2 categories Attchd & Attchd_1
hs['GarageType']=hs.get('GarageType').replace('Detchd','Attchd_1')
hs['GarageType']=hs.get('GarageType').replace('BuiltIn', 'Attchd_1')
hs['GarageType']=hs.get('GarageType').replace('Basment', 'Attchd_1')
hs['GarageType']=hs.get('GarageType').replace('CarPort', 'Attchd_1')
hs['GarageType']=hs.get('GarageType').replace('2Types', 'Attchd_1')

hs.GarageType.value_counts()
'''
Attchd      847
Attchd_1    483'''

hs.GarageType.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='GarageType', data=hs)
plt.xlabel('GarageType', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of GarageType', size=16)
plt.show()

#GarageFinish
hs.GarageFinish.describe()
'''
count     1330
unique       3
top        Unf
freq       580'''

hs.GarageFinish.value_counts()
'''
Unf    580
RFn    413
Fin    337'''

hs.GarageFinish.value_counts().sum() #1330

#Barplot/ Countplot
sns.countplot(x='GarageFinish', data=hs)
plt.xlabel('GarageFinish', size=14)
plt.ylabel('Counts', size=14)
plt.title('Barplot of GarageFinish', size=16)
plt.show()

#15 Categorical variables + the response variable were shown above
#13 Continuous variables are shown below

#LotArea
hs.LotArea.describe()
'''
count      1330.000000
mean      10623.873684
std       10268.219818
min        1300.000000
25%        7731.000000
50%        9590.500000
75%       11668.500000
max      215245.000000'''

#______histogram
#_run in block
plt.hist(hs.LotArea, bins = 'auto', facecolor = 'green')
plt.xlabel('LotArea')
plt.ylabel('counts')
plt.title('Histogram of LotArea')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['LotArea'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_1 = hs.LotArea.describe()['75%'] - hs.LotArea.describe()['25%'] #3937.5

low_lim_1 = hs.LotArea.describe()['25%']-1.5*iqr_1 #1824.75
len(hs.LotArea[hs.LotArea < low_lim_1]) #13 outliers
up_lim_1 = hs.LotArea.describe()['75%']+1.5*iqr_1 # 17574.75
len(hs.LotArea[hs.LotArea > up_lim_1]) #63 outliers

#Removing outliers by replacing them with upper and lower limits
hs.LotArea[hs.LotArea < low_lim_1] = low_lim_1
hs.LotArea[hs.LotArea > up_lim_1]  = up_lim_1

#____boxplot after replacing the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['LotArea'].plot.box(color=props2, patch_artist = True, vert = False)

#OverallQual
#______histogram
#_run in block
plt.hist(hs.OverallQual, bins = 'auto', facecolor = 'green')
plt.xlabel('OverallQual')
plt.ylabel('counts')
plt.title('Histogram of OverallQual')

#______Barplot/ Countplot
sns.countplot(x = 'OverallQual', data=hs)
plt.xlabel('OverallQual')
plt.ylabel('counts')
plt.title('Barplot of OverallQual')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['OverallQual'].plot.box(color=props2, patch_artist = True, vert = False)

#YearBuilt
hs.YearBuilt.describe()
'''
count    1330.000000
mean     1972.848120
std        29.555617
min      1880.000000
25%      1955.250000
50%      1976.000000
75%      2001.000000
max      2010.000000'''

#______histogram
#_run in block
plt.hist(hs.YearBuilt, bins = 'auto', facecolor = 'green')
plt.xlabel('YearBuilt')
plt.ylabel('counts')
plt.title('Histogram of YearBuilt')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['YearBuilt'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_2 = hs.YearBuilt.describe()['75%'] - hs.YearBuilt.describe()['25%'] #45.75
low_lim_2 = hs.YearBuilt.describe()['25%']-1.5*iqr_2 # 1886.625
len(hs.YearBuilt[hs.YearBuilt < low_lim_2]) #7 outliers
'''only few outliers on lower side, will tolerate!'''

#YearRemodAdd
hs.YearRemodAdd.describe()
'''
count    1330.000000
mean     1985.560150
std        20.302633
min      1950.000000
25%      1968.000000
50%      1994.000000
75%      2004.000000
max      2010.000000'''

#______histogram
#_run in block
plt.hist(hs.YearRemodAdd, bins = 'auto', facecolor = 'green')
plt.xlabel('YearRemodAdd')
plt.ylabel('counts')
plt.title('Histogram of YearRemodAdd')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['YearRemodAdd'].plot.box(color=props2, patch_artist = True, vert = False)

#BsmtFinSF1
hs.BsmtFinSF1.describe()
'''
count    1330.000000
mean      458.509023
std       450.239094
min         0.000000
25%         0.000000
50%       410.000000
75%       732.000000
max      5644.000000'''

#______histogram
#_run in block
plt.hist(hs.BsmtFinSF1, bins = 'auto', facecolor = 'green')
plt.xlabel('BsmtFinSF1')
plt.ylabel('counts')
plt.title('Histogram of BsmtFinSF1')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['BsmtFinSF1'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_3 = hs.BsmtFinSF1.describe()['75%'] - hs.BsmtFinSF1.describe()['25%'] #732.0
up_lim_3 = hs.BsmtFinSF1.describe()['75%']+1.5*iqr_3 # 1830.0
len(hs.BsmtFinSF1[hs.BsmtFinSF1 > up_lim_3]) #3 outliers

#Replacing outliers with upper limit
hs.BsmtFinSF1[hs.BsmtFinSF1 > up_lim_3] = up_lim_3

#____boxplot after replacing the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['BsmtFinSF1'].plot.box(color=props2, patch_artist = True, vert = False)

#TotalBsmtSF
hs.TotalBsmtSF.describe()
'''
count    1330.000000
mean     1088.914286
std       399.036696
min       105.000000
25%       816.250000
50%      1015.000000
75%      1314.000000
max      6110.000000'''

#______histogram
#_run in block
plt.hist(hs.TotalBsmtSF, bins = 'auto', facecolor = 'green')
plt.xlabel('TotalBsmtSF')
plt.ylabel('counts')
plt.title('Histogram of TotalBsmtSF')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['TotalBsmtSF'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_4 = hs.TotalBsmtSF.describe()['75%'] - hs.TotalBsmtSF.describe()['25%'] #497.75
up_lim_4 = hs.TotalBsmtSF.describe()['75%']+1.5*iqr_4 # 2060.625
len(hs.TotalBsmtSF[hs.TotalBsmtSF > up_lim_4]) #18 outliers

#Replacing outliers with upper limit
hs.TotalBsmtSF[hs.TotalBsmtSF > up_lim_4] = up_lim_4

#____boxplot after replacing the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['TotalBsmtSF'].plot.box(color=props2, patch_artist = True, vert = False)

#1stFlrSF - FirstFlrSF
#Variable name starting with numeric troubles to execute functions, therefore
#changing the name of the variable
hs = hs.rename(columns={"1stFlrSF":"FirstFlrSF"})
hs.info()
hs.FirstFlrSF.describe()
'''
count    1330.000000
mean     1169.248120
std       378.837567
min       438.000000
25%       894.000000
50%      1096.500000
75%      1400.000000
max      4692.000000'''

#______histogram
#_run in block
plt.hist(hs.FirstFlrSF, bins = 'auto', facecolor = 'green')
plt.xlabel('FirstFlrSF')
plt.ylabel('counts')
plt.title('Histogram of FirstFlrSF')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['FirstFlrSF'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_5 = hs.FirstFlrSF.describe()['75%'] - hs.FirstFlrSF.describe()['25%'] #506.0
up_lim_5 = hs.FirstFlrSF.describe()['75%']+1.5*iqr_5 # 2159.0
len(hs.FirstFlrSF[hs.FirstFlrSF > up_lim_5]) #12 outliers

#Replacing the outliers with the upper limit
hs.FirstFlrSF[hs.FirstFlrSF > up_lim_5] = up_lim_5

#____boxplot after removing the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['FirstFlrSF'].plot.box(color=props2, patch_artist = True, vert = False)

#GrLivArea
hs.GrLivArea.describe()
'''
count    1330.000000
mean     1525.401504
std       498.687926
min       438.000000
25%      1158.000000
50%      1478.000000
75%      1785.500000
max      5642.000000'''

#______histogram
#_run in block
plt.hist(hs.GrLivArea, bins = 'auto', facecolor = 'green')
plt.xlabel('GrLivArea')
plt.ylabel('counts')
plt.title('Histogram of GrLivArea')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['GrLivArea'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_6 = hs.GrLivArea.describe()['75%'] - hs.GrLivArea.describe()['25%'] #627.5
up_lim_6 = hs.GrLivArea.describe()['75%']+1.5*iqr_6 # 2726.75
len(hs.GrLivArea[hs.GrLivArea > up_lim_6]) #25 outliers

#Replacing outliers with upper limit
hs.GrLivArea[hs.GrLivArea > up_lim_6] = up_lim_6

#____boxplot after replacing the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['GrLivArea'].plot.box(color=props2, patch_artist = True, vert = False)

#TotRmsAbvGrd
#______Barplot/ Countplot
sns.countplot(x = 'TotRmsAbvGrd', data=hs)
plt.xlabel('TotRmsAbvGrd')
plt.ylabel('counts')
plt.title('Barplot of TotRmsAbvGrd')

hs.TotRmsAbvGrd.value_counts()
'''
6     379
7     311
5     248
8     171
4      79
9      68
10     40
11     15
3      12
12      7'''

hs.TotRmsAbvGrd.value_counts().sum() #1330

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['TotRmsAbvGrd'].plot.box(color=props2, patch_artist = True, vert = False) #keeping outliers

#Fireplaces
#______Barplot/ Countplot
sns.countplot(x = 'Fireplaces', data=hs)
plt.xlabel('Fireplaces')
plt.ylabel('counts')
plt.title('Barplot of Fireplaces')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['Fireplaces'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_7 = hs.Fireplaces.describe()['75%'] - hs.Fireplaces.describe()['25%'] #1.0
up_lim_7 = hs.Fireplaces.describe()['75%']+1.5*iqr_7 # 2.5
len(hs.Fireplaces[hs.Fireplaces > up_lim_7]) #5 outliers
'''
as there is an order, we will treat this as continuous!
and tolearte 5 counts in 3 fireplaces, corr is also good!'''

#GarageCars
#______Barplot/ Countplot
sns.countplot(x = 'GarageCars', data=hs)
plt.xlabel('GarageCars')
plt.ylabel('counts')
plt.title('Barplot of GarageCars')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['GarageCars'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_8 = hs.GarageCars.describe()['75%'] - hs.GarageCars.describe()['25%'] #1.0
up_lim_8 = hs.GarageCars.describe()['75%']+1.5*iqr_8 # 3.5
len(hs.GarageCars[hs.GarageCars > up_lim_8]) #5 outliers
'''
as there is an order, we will treat this as continuous!
and tolearte 5 counts in 4 GarageCars, corr is also good!'''

#WoodDeckSF
hs.WoodDeckSF.describe()
'''
count    1330.000000
mean       98.684211
std       126.874559
min         0.000000
25%         0.000000
50%         0.000000
75%       172.000000
max       857.000000'''

#______Histogram
plt.hist(hs.WoodDeckSF, bins = 'auto', facecolor = 'green')
plt.xlabel('WoodDeckSF')
plt.ylabel('counts')
plt.title('Barplot of WoodDeckSF')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['WoodDeckSF'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_9 = hs.WoodDeckSF.describe()['75%'] - hs.WoodDeckSF.describe()['25%'] #172.0
up_lim_9 = hs.WoodDeckSF.describe()['75%']+1.5*iqr_9 # 430.0
len(hs.WoodDeckSF[hs.WoodDeckSF > up_lim_9]) #29 outliers

#Replacing with the upperlimit
hs.WoodDeckSF[hs.WoodDeckSF > up_lim_9] = up_lim_9

#____boxplot after replacing the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['WoodDeckSF'].plot.box(color=props2, patch_artist = True, vert = False)
    
#OpenPorchSF
hs.OpenPorchSF.describe()
'''
count    1330.000000
mean       47.472180
std        65.455638
min         0.000000
25%         0.000000
50%        28.000000
75%        69.000000
max       547.000000'''

#______Histogram
plt.hist(hs.OpenPorchSF, bins = 'auto', facecolor = 'green')
plt.xlabel('OpenPorchSF')
plt.ylabel('counts')
plt.title('Barplot of OpenPorchSF')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['OpenPorchSF'].plot.box(color=props2, patch_artist = True, vert = False)

#Getting the Iqr, up_lim & low_lim
iqr_10 = hs.OpenPorchSF.describe()['75%'] - hs.OpenPorchSF.describe()['25%'] #69
up_lim_10 = hs.OpenPorchSF.describe()['75%']+1.5*iqr_10 # 172.5
len(hs.OpenPorchSF[hs.OpenPorchSF > up_lim_10]) #67 outliers

#Replacing the outliers with upper limit
hs.OpenPorchSF[hs.OpenPorchSF > up_lim_10] = up_lim_10

#____boxplot after replacing the outliers
props2 = dict(boxes = 'green', whiskers = 'red', medians = 'black', caps = 'blue')
hs['OpenPorchSF'].plot.box(color=props2, patch_artist = True, vert = False)

#Finding the correlation b/w target variable & Continous variables
hs.corr()['SalePrice'] #By default it takes only continuous variables
'''
LotArea         0.407901
OverallQual     0.801526
YearBuilt       0.545806
YearRemodAdd    0.545212
BsmtFinSF1      0.345900
TotalBsmtSF     0.618067
FirstFlrSF      0.608780
GrLivArea       0.719647
TotRmsAbvGrd    0.472338
Fireplaces      0.453875
GarageCars      0.657451
WoodDeckSF      0.321283
OpenPorchSF     0.392774
SalePrice       1.000000'''

#Heatmap
sns.heatmap(hs.corr(), cmap="PiYG")

#Heatmap with correlation values
plt.figure(figsize = (15,15))
sns.heatmap(hs.corr(), annot=True)

#As a backup assigning to new variable
hs1 = hs 
hs1.shape
hs1.info()

#Standardizing the continuous data
from sklearn.preprocessing import StandardScaler 
scale = StandardScaler()
hs1_cat = hs1.select_dtypes(include=object) #Filtering object/ categorical variables
hs1_cat.info() #15 variables
hs1_cat.columns #Categorical variables names
''''MSZoning', 'LotShape', 'Neighborhood', 'Condition1', 'BldgType',
'RoofStyle', 'MasVnrType', 'ExterQual', 'BsmtQual', 'BsmtExposure',
'BsmtFinType1', 'HeatingQC', 'KitchenQual', 'GarageType','GarageFinish''''

hs1_con = hs1.select_dtypes(exclude=object) #Filtering non object/ categorical variables
hs1_con.info() #14 variables in including target variable
hs1_con = hs1_con.drop('SalePrice', axis=1) #dropping target variable
hs1_con.info() #13 Predictors
hs1_con.columns
''''LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1',
'TotalBsmtSF', 'FirstFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'Fireplaces',
'GarageCars', 'WoodDeckSF', 'OpenPorchSF''''

hs1_tar = pd.DataFrame(hs1.SalePrice) #Target variable
hs1_tar.info()

#Scaling the continuous variables without target variable
hs1_con = pd.DataFrame(scale.fit_transform(hs1_con), columns = hs1_con.columns)

pca = PCA(n_components=13)
hs_pca = pca.fit_transform(hs1_con)

#Array to dataframe
hs_pca = pd.DataFrame(hs_pca) #Principal components

pvctr_13 = pca.components_ #Eigen Vectors

pevl_13 = pca.explained_variance_ #Eigen values
#The amount of variance explained by each of the selected components.
'''
array([4.85966881, 1.70420578, 1.43689724, 0.92325719, 0.79813779,
       0.7294515 , 0.61129359, 0.60360117, 0.45882852, 0.35326224,
       0.30241558, 0.13814798, 0.0906144 ])'''

pca.explained_variance_ratio_
#Percentage of variance explained by each of the selected components.
'''
array([0.37353961, 0.13099419, 0.11044745, 0.07096639, 0.06134905,
       0.05606946, 0.04698723, 0.04639595, 0.03526796, 0.02715359,
       0.02324525, 0.01061878, 0.0069651 ])'''

#Cummulative sum of explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)
'''
array([0.37353961, 0.5045338 , 0.61498125, 0.68594763, 0.74729669,
       0.80336615, 0.85035338, 0.89674933, 0.93201729, 0.95917088,
       0.98241612, 0.9930349 , 1.        ])'''

# Plot of Cummulative sum of explained_variance_ratio_
plt.plot(np.cumsum(pca.explained_variance_ratio_), color= 'r')
plt.title('Cummulative - PCA Explained Variance Ratio')
plt.xlabel('PCA n_components')
plt.ylabel('PCA- explained_variance_ratio')
plt.xticks(np.arange(0,14,1))
plt.yticks(np.arange(0.3,1.1,0.05))
plt.grid(color='b', linestyle='dotted', linewidth=.5)
plt.show()

#Selecting 5 PCA's
hs_pca_5 = hs_pca.iloc[:, :5]
hs_pca_5.columns = ['PCA0', 'PCA1', 'PCA2', 'PCA3', 'PCA4']
hs_pca_5.info()

#Concatenating 15 categorical variables, 5 PCA's and Target variable
#Resetting the index
hs1_cat = hs1_cat.reset_index(drop=True)
hs_pca_5 = hs_pca_5.reset_index(drop=True)
hs1_tar = hs1_tar.reset_index(drop=True)

#Concatenating 3 dataframes
hs_pca1 = pd.concat([hs1_cat,hs_pca_5,hs1_tar], axis=1)
hs_pca1.info()

hs_pca1.columns
#Building a linear regression model
from statsmodels.formula.api import ols
model1 = ols('''SalePrice~MSZoning+LotShape+Neighborhood+Condition1+BldgType+
             RoofStyle+MasVnrType+ExterQual+BsmtQual+BsmtExposure+BsmtFinType1
             +HeatingQC+KitchenQual+GarageType+GarageFinish+PCA0+PCA1+PCA2+PCA3
             +PCA4''', data=hs_pca1).fit()
             
model1.summary()
'''
                                                       OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.849
Model:                            OLS   Adj. R-squared:                  0.847
Method:                 Least Squares   F-statistic:                     350.5
Date:                Wed, 17 Mar 2021   Prob (F-statistic):               0.00
Time:                        09:24:39   Log-Likelihood:                -15378.
No. Observations:                1330   AIC:                         3.080e+04
Df Residuals:                    1308   BIC:                         3.091e+04
Df Model:                          21                                         
Covariance Type:            nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                  1.736e+05   3164.515     54.853      0.000    1.67e+05     1.8e+05
MSZoning[T.RL_1]           2471.2928   2107.127      1.173      0.241   -1662.425    6605.011
LotShape[T.Reg]           -1044.7978   1578.748     -0.662      0.508   -4141.954    2052.358
Neighborhood[T.nghb_1]     1.425e+04   1566.284      9.098      0.000    1.12e+04    1.73e+04
Condition1[T.Norm_1]      -9133.6871   2144.925     -4.258      0.000   -1.33e+04   -4925.817
BldgType[T.Fam_1]         -1.676e+04   2291.880     -7.315      0.000   -2.13e+04   -1.23e+04
RoofStyle[T.Gable_1]       4038.7767   1829.396      2.208      0.027     449.905    7627.648
MasVnrType[T.Vnr_present]   232.6403   1639.670      0.142      0.887   -2984.031    3449.312
ExterQual[T.TA_1]          9809.9775   2372.658      4.135      0.000    5155.347    1.45e+04
BsmtQual[T.TA_1]           1879.9223   2102.785      0.894      0.371   -2245.277    6005.122
BsmtExposure[T.Yes]        4545.6330   1647.630      2.759      0.006    1313.347    7777.919
BsmtFinType1[T.Ungl_1]     1542.2401   1813.516      0.850      0.395   -2015.479    5099.959
HeatingQC[T.Ex_1]         -3937.1498   1765.867     -2.230      0.026   -7401.391    -472.909
KitchenQual[T.TA_1]        7011.0453   2041.440      3.434      0.001    3006.190     1.1e+04
GarageType[T.Attchd_1]      772.1039   1819.739      0.424      0.671   -2797.823    4342.031
GarageFinish[T.RFn]       -3995.1735   1941.553     -2.058      0.040   -7804.073    -186.274
GarageFinish[T.Unf]       -4134.9229   2293.598     -1.803      0.072   -8634.455     364.609
PCA0                       2.261e+04    580.577     38.947      0.000    2.15e+04    2.38e+04
PCA1                      -1170.4413    783.893     -1.493      0.136   -2708.266     367.383
PCA2                       2547.0692    701.270      3.632      0.000    1171.333    3922.805
PCA3                         81.2323    747.962      0.109      0.914   -1386.104    1548.568
PCA4                       -148.3333    809.798     -0.183      0.855   -1736.978    1440.311
==============================================================================
Omnibus:                      240.132   Durbin-Watson:                   2.013
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2054.468
Skew:                          -0.576   Prob(JB):                         0.00
Kurtosis:                       8.979   Cond. No.                         12.7
=============================================================================='''
#______________________________________________________________________________

#Building linear regression with all the variables without pca
hs2 = hs
hs2.info()

from statsmodels.formula.api import ols
model2 = ols('''SalePrice~MSZoning+LotShape+Neighborhood+Condition1
             +BldgType+RoofStyle+MasVnrType+ExterQual+BsmtQual
             +BsmtExposure+BsmtFinType1+HeatingQC+KitchenQual+
             GarageType+GarageFinish+LotArea+OverallQual+YearBuilt+
             YearRemodAdd+BsmtFinSF1+TotalBsmtSF+FirstFlrSF+GrLivArea
             +TotRmsAbvGrd+Fireplaces+GarageCars+WoodDeckSF+OpenPorchSF''',
             data= hs2).fit()

model2.summary()

'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.875
Model:                            OLS   Adj. R-squared:                  0.872
Method:                 Least Squares   F-statistic:                     312.6
Date:                Wed, 17 Mar 2021   Prob (F-statistic):               0.00
Time:                        10:55:47   Log-Likelihood:                -15255.
No. Observations:                1330   AIC:                         3.057e+04
Df Residuals:                    1300   BIC:                         3.073e+04
Df Model:                          29                                         
Covariance Type:            nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                 -9.325e+05   1.13e+05     -8.250      0.000   -1.15e+06   -7.11e+05
MSZoning[T.RL_1]           -897.4413   2059.393     -0.436      0.663   -4937.540    3142.657
LotShape[T.Reg]            -720.3973   1480.770     -0.487      0.627   -3625.357    2184.562
Neighborhood[T.nghb_1]     1.119e+04   1470.552      7.612      0.000    8308.329    1.41e+04
Condition1[T.Norm_1]      -9327.9185   1972.707     -4.728      0.000   -1.32e+04   -5457.881
BldgType[T.Fam_1]         -1.565e+04   2234.761     -7.004      0.000      -2e+04   -1.13e+04
RoofStyle[T.Gable_1]       3490.5718   1687.068      2.069      0.039     180.897    6800.246
MasVnrType[T.Vnr_present]    -6.3439   1531.774     -0.004      0.997   -3011.364    2998.676
ExterQual[T.TA_1]          5892.5814   2216.523      2.658      0.008    1544.228    1.02e+04
BsmtQual[T.TA_1]          -1257.3779   1974.656     -0.637      0.524   -5131.239    2616.484
BsmtExposure[T.Yes]        5132.4748   1526.325      3.363      0.001    2138.145    8126.805
BsmtFinType1[T.Ungl_1]        1.3303   1693.701      0.001      0.999   -3321.357    3324.017
HeatingQC[T.Ex_1]         -4081.1116   1636.336     -2.494      0.013   -7291.260    -870.963
KitchenQual[T.TA_1]        4748.3394   1921.927      2.471      0.014     977.921    8518.758
GarageType[T.Attchd_1]     -153.1697   1712.619     -0.089      0.929   -3512.969    3206.629
GarageFinish[T.RFn]       -3208.3845   1797.319     -1.785      0.074   -6734.348     317.579
GarageFinish[T.Unf]       -2652.1466   2137.502     -1.241      0.215   -6845.478    1541.185
LotArea                       1.4274      0.267      5.348      0.000       0.904       1.951
OverallQual                1.363e+04    882.952     15.437      0.000    1.19e+04    1.54e+04
YearBuilt                   168.0247     43.156      3.893      0.000      83.361     252.689
YearRemodAdd                288.2533     48.499      5.943      0.000     193.108     383.398
BsmtFinSF1                   17.9236      1.842      9.731      0.000      14.310      21.537
TotalBsmtSF                  21.4021      4.356      4.913      0.000      12.857      29.947
FirstFlrSF                   -4.2484      4.486     -0.947      0.344     -13.049       4.552
GrLivArea                    46.8819      3.171     14.785      0.000      40.661      53.102
TotRmsAbvGrd              -1680.5397    790.273     -2.127      0.034   -3230.890    -130.189
Fireplaces                 4056.4236   1232.850      3.290      0.001    1637.829    6475.018
GarageCars                 9594.7989   1469.476      6.529      0.000    6711.996    1.25e+04
WoodDeckSF                   13.9011      5.916      2.350      0.019       2.296      25.506
OpenPorchSF                  39.0353     13.828      2.823      0.005      11.908      66.163
==============================================================================
Omnibus:                      385.324   Durbin-Watson:                   1.989
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5182.498
Skew:                          -0.956   Prob(JB):                         0.00
Kurtosis:                      12.480   Cond. No.                     1.92e+06
=============================================================================='''
#______________________________________________________________________________
#Linear Regression model with PCA after removing variables with pvalue more than 0.05
#9 Variables having p value more than 0.05
#MSZoning, LotShape, MasVnrType, BsmtQual, BsmtFinType1, GarageType,, PCA1 PCA3, PCA4
#11 Predictors in the model
from statsmodels.formula.api import ols
model3 = ols('''SalePrice~Neighborhood+Condition1+BldgType+RoofStyle+ExterQual
             +BsmtExposure+HeatingQC+KitchenQual+GarageFinish+PCA0+PCA2''',
             data=hs_pca1).fit()
model3.summary()
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.848
Model:                            OLS   Adj. R-squared:                  0.847
Method:                 Least Squares   F-statistic:                     614.2
Date:                Thu, 18 Mar 2021   Prob (F-statistic):               0.00
Time:                        04:46:59   Log-Likelihood:                -15381.
No. Observations:                1330   AIC:                         3.079e+04
Df Residuals:                    1317   BIC:                         3.086e+04
Df Model:                          12                                         
Covariance Type:            nonrobust                                         
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
Intercept               1.753e+05   2366.371     74.071      0.000    1.71e+05     1.8e+05
Neighborhood[T.nghb_1]  1.389e+04   1531.584      9.068      0.000    1.09e+04    1.69e+04
Condition1[T.Norm_1]   -9733.2750   2114.868     -4.602      0.000   -1.39e+04   -5584.398
BldgType[T.Fam_1]      -1.476e+04   2054.350     -7.183      0.000   -1.88e+04   -1.07e+04
RoofStyle[T.Gable_1]    3671.8225   1789.520      2.052      0.040     161.201    7182.444
ExterQual[T.TA_1]       1.112e+04   2255.088      4.932      0.000    6697.582    1.55e+04
BsmtExposure[T.Yes]     4849.3403   1624.939      2.984      0.003    1661.589    8037.092
HeatingQC[T.Ex_1]      -4495.4291   1723.734     -2.608      0.009   -7876.994   -1113.864
KitchenQual[T.TA_1]     7593.7012   1990.188      3.816      0.000    3689.416    1.15e+04
GarageFinish[T.RFn]    -4356.7238   1921.226     -2.268      0.024   -8125.720    -587.727
GarageFinish[T.Unf]    -4901.2570   2146.874     -2.283      0.023   -9112.923    -689.591
PCA0                    2.227e+04    468.471     47.535      0.000    2.13e+04    2.32e+04
PCA2                    2559.0030    634.366      4.034      0.000    1314.524    3803.482
==============================================================================
Omnibus:                      242.892   Durbin-Watson:                   2.004
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2056.601
Skew:                          -0.591   Prob(JB):                         0.00
Kurtosis:                       8.976   Cond. No.                         10.6
=============================================================================='''
#______________________________________________________________________________
#Linear Regression model without PCA after removing vairables pvalue more than 0.05
#8 Variables having p value more than 0.05
#MSZoning, LotShape, MasVnrType, BsmtQual, BsmtFinType1, GarageType,
#GarageFinish,FirstFlrSF

from statsmodels.formula.api import ols
model4 = ols('''SalePrice~Neighborhood+Condition1+BldgType+RoofStyle+ExterQual
             +BsmtExposure+HeatingQC+KitchenQual+LotArea+OverallQual+YearBuilt+
             YearRemodAdd+BsmtFinSF1+TotalBsmtSF+GrLivArea+TotRmsAbvGrd
             +Fireplaces+GarageCars+WoodDeckSF+OpenPorchSF''',
             data= hs2).fit()

model4.summary()
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.874
Model:                            OLS   Adj. R-squared:                  0.872
Method:                 Least Squares   F-statistic:                     454.4
Date:                Wed, 17 Mar 2021   Prob (F-statistic):               0.00
Time:                        11:06:36   Log-Likelihood:                -15258.
No. Observations:                1330   AIC:                         3.056e+04
Df Residuals:                    1309   BIC:                         3.067e+04
Df Model:                          20                                         
Covariance Type:            nonrobust                                         
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
Intercept              -9.597e+05   9.89e+04     -9.706      0.000   -1.15e+06   -7.66e+05
Neighborhood[T.nghb_1]   1.14e+04   1450.434      7.857      0.000    8550.247    1.42e+04
Condition1[T.Norm_1]   -9361.0779   1966.809     -4.760      0.000   -1.32e+04   -5502.636
BldgType[T.Fam_1]      -1.612e+04   2157.337     -7.471      0.000   -2.04e+04   -1.19e+04
RoofStyle[T.Gable_1]    3631.0561   1644.358      2.208      0.027     405.190    6856.922
ExterQual[T.TA_1]       5996.5411   2172.398      2.760      0.006    1734.779    1.03e+04
BsmtExposure[T.Yes]     4974.6100   1510.244      3.294      0.001    2011.846    7937.373
HeatingQC[T.Ex_1]      -4143.9345   1622.743     -2.554      0.011   -7327.395    -960.474
KitchenQual[T.TA_1]     4558.5820   1908.697      2.388      0.017     814.142    8303.022
LotArea                    1.4692      0.247      5.947      0.000       0.985       1.954
OverallQual             1.363e+04    860.865     15.829      0.000    1.19e+04    1.53e+04
YearBuilt                183.0044     34.453      5.312      0.000     115.415     250.594
YearRemodAdd             284.7314     47.327      6.016      0.000     191.887     377.576
BsmtFinSF1                18.1980      1.765     10.313      0.000      14.736      21.660
TotalBsmtSF               17.9095      2.448      7.317      0.000      13.108      22.711
GrLivArea                 45.8958      3.054     15.026      0.000      39.904      51.888
TotRmsAbvGrd           -1503.5465    778.366     -1.932      0.054   -3030.528      23.435
Fireplaces              4264.2900   1196.812      3.563      0.000    1916.411    6612.170
GarageCars              9453.7273   1429.980      6.611      0.000    6648.425    1.23e+04
WoodDeckSF                13.8493      5.877      2.356      0.019       2.319      25.379
OpenPorchSF               38.1799     13.555      2.817      0.005      11.588      64.772
==============================================================================
Omnibus:                      384.098   Durbin-Watson:                   1.996
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4990.742
Skew:                          -0.962   Prob(JB):                         0.00
Kurtosis:                      12.293   Cond. No.                     1.67e+06
=============================================================================='''

pca_2 = PCA(n_components=2)
print(pca_2)
hs_pca_2 = pca_2.fit_transform(hs1_con)
print(hs_pca_2)
