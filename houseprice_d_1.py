# God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("D:/data _science/PYTHON/Linear_Regression_Python/HousePrices.csv")
df = pd.DataFrame(df) #2073, 81, float=3; int =35, object 43
# respose var/target var
df.Property_Sale_Price.describe() # min 34.9k; max 7.55lacs

#______histogram
#_run in block
plt.hist(df.Property_Sale_Price, bins = 'auto', facecolor = 'red')
plt.xlabel('Property_Sale_Price')
plt.ylabel('counts')
plt.title('Histogram of Property_Sale_Price')

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
df['Property_Sale_Price'].plot.box(color=props2, patch_artist = True, vert = False)

#________outliers
Q1 = np.percentile(df.Property_Sale_Price, 25, interpolation = 'midpoint')  
Q2 = np.percentile(df.Property_Sale_Price, 50, interpolation = 'midpoint')  
Q3 = np.percentile(df.Property_Sale_Price, 75, interpolation = 'midpoint')   
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)   
IQR = Q3 - Q1  
print('Interquartile range is', IQR) # 84000
df.info()
low_lim = Q1 - 1.5 * IQR 
up_lim = Q3 + 1.5 * IQR # 340000
print('low_limit is', low_lim) 
print('up_limit is', up_lim) #340000
# anything q3 + 1.5*iqr is outlier on higher side
#_______counting outliers
len(df.Property_Sale_Price[df.Property_Sale_Price > 340000]) # 78
78/2073 # 4% ; we will build model on these 78 obs at the end 
# let's remove/select
df1 = df[df.Property_Sale_Price <= 340000] # <=
df1.info() #1990 ; 2073-78 = 1995
2073-78 # 1995
#______histogram
#_run in block
plt.hist(df1.Property_Sale_Price, bins = 'auto', facecolor = 'blue')
plt.xlabel('Property_Sale_Price')
plt.ylabel('counts')
plt.title('Histogram of Property_Sale_Price')

# 2007, 600 real world data , log reg, mda; 200? 
#____boxplot
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'blue')
df1['Property_Sale_Price'].plot.box(color=props2, patch_artist = True, vert = False)

#________outliers
Q1 = np.percentile(df1.Property_Sale_Price, 25, interpolation = 'midpoint')  
Q2 = np.percentile(df1.Property_Sale_Price, 50, interpolation = 'midpoint')  
Q3 = np.percentile(df1.Property_Sale_Price, 75, interpolation = 'midpoint')   
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)   
IQR = Q3 - Q1  
print('Interquartile range is', IQR) # 76000

low_lim = Q1 - 1.5 * IQR 
up_lim = Q3 + 1.5 * IQR # 319000
print('low_limit is', low_lim) 
print('up_limit is', up_lim) #319000

#_______counting outliers
len(df1.Property_Sale_Price[df1.Property_Sale_Price > 319000]) # 47
47/1995 # 2.4%
#_______________assigning UL to outliers 
df1.Property_Sale_Price[df1.Property_Sale_Price > 319000] = 319000

#______histogram
#_run in block
plt.hist(df1.Property_Sale_Price, bins = 'auto', facecolor = 'green')
plt.xlabel('Property_Sale_Price')
plt.ylabel('counts')
plt.title('Histogram of Property_Sale_Price')

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'blue', medians = 'black', caps = 'blue')
df1['Property_Sale_Price'].plot.box(color=props2, patch_artist = True, vert = False)
len(df1)
df1.info()

#______let's see some categorical vars
df1.Zone_Class.value_counts() #1537, 320, 97, 22, 14
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='Zone_Class', data = df1) 
'''
better club RM, FV,RH,C (all), as RL_1
'''
df1['Zone_Class']=df1.get('Zone_Class').replace('RM','RL_1')
df1['Zone_Class']=df1.get('Zone_Class').replace('FV','RL_1')
df1['Zone_Class']=df1.get('Zone_Class').replace('RH','RL_1')
df1['Zone_Class']=df1.get('Zone_Class').replace('C (all)','RL_1')

df1.Zone_Class.value_counts() # now you have only 2 categories as 1541 and 454!

# can Zone_class be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Zone_Class).describe(),2)
round(df1.Property_Sale_Price.groupby(df1.Zone_Class).min(),2)

# Indpndnt sample t test
ZC_RL = df1[df1.Zone_Class == 'RL']
ZC_RL_1 = df1[df1.Zone_Class == 'RL_1']
ZC_RL_1.info()
import scipy
scipy.stats.ttest_ind(ZC_RL.Property_Sale_Price, ZC_RL_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

'''
so, next we should make dummy vars for Zone_Class!
'''
#______let's see 2nd Object/ categorical var Road_Type
df1.info()
df1.Road_Type.value_counts() #1985, 10 
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='Road_Type', data = df1) 
'''
better not include this
'''
#______let's see 3rd Object/ categorical var Alley
df1.info()
df1.Alley.value_counts() # 72, 57 
72+57 # 129
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='Alley', data = df1) 
(72+57)/1995 # 6.5% 
'''
better not include Alley
'''
#______let's see 4th Object/ categorical var Property_Shape
df1.info()
df1.Property_Shape.value_counts() # 1265, 663,53,14  
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='Property_Shape', data = df1)
'''
better club IR1, IR2 and IR3 as Reg_1
'''
df1['Property_Shape']=df1.get('Property_Shape').replace('IR1','Reg_1')
df1['Property_Shape']=df1.get('Property_Shape').replace('IR2','Reg_1')
df1['Property_Shape']=df1.get('Property_Shape').replace('IR3','Reg_1')

df1.Property_Shape.value_counts() # now you have only 2 categories as 1265 and 730!
import seaborn as sns
sns.countplot(x ='Property_Shape', data = df1)
# can Propert_Shape be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Property_Shape).describe(),2)
round(df1.Property_Sale_Price.groupby(df1.Property_Shape).min(),2)

# Indpndnt sample t test
PS_Reg = df1[df1.Property_Shape == 'Reg']
PS_Reg_1 = df1[df1.Property_Shape == 'Reg_1']
PS_Reg.info()
PS_Reg_1.info()
import scipy
scipy.stats.ttest_ind(PS_Reg.Property_Sale_Price, PS_Reg_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

'''
so, next we should make dummy vars for Property Shape!
'''
#______let's see 5th Object/ categorical var, LandContour
df1.info()
df1.LandContour.value_counts() # 1791, 92, 66, 46
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='LandContour', data = df1)
'''
better club Bnk, HLS, Low as Lvl_1
'''
df1['LandContour']=df1.get('LandContour').replace('Bnk','Lvl_1')
df1['LandContour']=df1.get('LandContour').replace('HLS','Lvl_1')
df1['LandContour']=df1.get('LandContour').replace('Low','Lvl_1')

df1.LandContour.value_counts() # now you have only 2 categories as 1791 and 204
204/1995 # 10.2%
import seaborn as sns
sns.countplot(x ='LandContour', data = df1)
# can LandCounter be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.LandContour).describe(),2)
round(df1.Property_Sale_Price.groupby(df1.LandContour).min(),2)
# Indpndnt sample t test
LV = df1[df1.LandContour == 'Lvl']
LV_1 = df1[df1.LandContour == 'Lvl_1']
import scipy
scipy.stats.ttest_ind(LV.Property_Sale_Price, LV_1.Property_Sale_Price)
# p_value is >0.05 = 0.4656; Ho Accept; Bad Predictor

#______let's see 6th Object/ categorical var, Utilities
df1.info()
df1.Utilities.value_counts() # 1992 and 3 
'''
too less in one group, No sense in including this in model!
'''
#______let's see 7th Object/ categorical var, LotConfig
df1.info()
df1.LotConfig.value_counts() # 1423, 377, 128, 63, 4
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='LotConfig', data = df1)
'''
better club Corner, CulDSac, FR2, FR3 as Inside_1
'''
df1['LotConfig']=df1.get('LotConfig').replace('Corner','Inside_1')
df1['LotConfig']=df1.get('LotConfig').replace('CulDSac','Inside_1')
df1['LotConfig']=df1.get('LotConfig').replace('FR2','Inside_1')
df1['LotConfig']=df1.get('LotConfig').replace('FR3','Inside_1')
df1.LotConfig.value_counts() # now you have only 2 categories as 1423 and 572
572/1995 # 28.7%
import seaborn as sns
sns.countplot(x ='LotConfig', data = df1)
# can LotConfig be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.LotConfig).describe(),2)
round(df1.Property_Sale_Price.groupby(df1.LotConfig).min(),2)
# Indpndnt sample t test
Inside = df1[df1.LotConfig == 'Inside']
Inside_1 = df1[df1.LotConfig == 'Inside_1']
import scipy
scipy.stats.ttest_ind(Inside.Property_Sale_Price, Inside_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor
'''
so, next we should make dummy vars for LotConfig!
'''

#______let's see 8th Object/ categorical var, LandSlope
df1.info()
df1.LandSlope.value_counts() # 1886, 96, 13
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='LandSlope', data = df1)
(96+13)/1995 # 5.5%
'''
better drop LandSlope
'''
#______let's see 9th Object/ categorical var, Neighborhood
df1.info()
df1.Neighborhood.value_counts() # 25 categories
sum(df1.Neighborhood.value_counts())
# count plot on single categorical variable
#import seaborn as sns
#sns.countplot(x ='Neighborhood', data = df1)
'''
let's make in two groups, nghb and nghb_1
'''
df1['Neighborhood']=df1.get('Neighborhood').replace('NAmes','nghb')
df1['Neighborhood']=df1.get('Neighborhood').replace('CollgCr','nghb')
df1['Neighborhood']=df1.get('Neighborhood').replace('OldTown','nghb')
df1['Neighborhood']=df1.get('Neighborhood').replace('Edwards','nghb')
df1['Neighborhood']=df1.get('Neighborhood').replace('Somerst','nghb')
df1['Neighborhood']=df1.get('Neighborhood').replace('Gilbert','nghb')
df1['Neighborhood']=df1.get('Neighborhood').replace('Sawyer','nghb')
df1['Neighborhood']=df1.get('Neighborhood').replace('NWAmes','nghb')

df1['Neighborhood']=df1.get('Neighborhood').replace('SawyerW','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('BrkSide','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('Mitchel','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('Crawfor','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('NridgHt','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('NoRidge','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('IDOTRR','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('Timber','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('ClearCr','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('SWISU','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('Blmngtn','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('MeadowV','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('StoneBr','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('BrDale','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('NPkVill','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('Veenker','nghb_1')
df1['Neighborhood']=df1.get('Neighborhood').replace('Blueste','nghb_1')

df1.Neighborhood.value_counts() # 1274 and 721 
sum(df1.Neighborhood.value_counts()) # 1995

import seaborn as sns
sns.countplot(x ='Neighborhood', data = df1)

# can Neighborhood be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Neighborhood).describe(),2)
'''
learnt from previous analysis
that if difference in means is like this : 1,80k and 1,67k
var is a good differentiator
so, we can take a call now only and decide to 
include in the model, however, better cross check through
indpndnt test
'''
round(df1.Property_Sale_Price.groupby(df1.Neighborhood).min(),2)
# Indpndnt sample t test
nghb = df1[df1.Neighborhood == 'nghb']
nghb_1 = df1[df1.Neighborhood == 'nghb_1']
import scipy
scipy.stats.ttest_ind(nghb.Property_Sale_Price, nghb_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor
'''
so, next we should make dummy vars for Neighborhood!
'''
#______let's see 10th Object/ categorical var, Condition1
df1.info()
df1.Condition1.value_counts() # 9 categories
sum(df1.Condition1.value_counts())
'''
let's make in two groups, Norm and Norm_1
'''
df1['Condition1']=df1.get('Condition1').replace('Feedr','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('Artery','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRAn','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('PosN','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('Feedr','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRAe','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('PosA','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRNn','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRNe','Norm_1')

df1.Condition1.value_counts() # 2 categories, 1719 & 276
sum(df1.Condition1.value_counts()) # 1995 
276/1995 # 14%
import seaborn as sns
sns.countplot(x ='Condition1', data = df1)

# can Condition1 be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Condition1).describe(),2)

# Indpndnt sample t test
Norm = df1[df1.Condition1 == 'Norm']
Norm_1 = df1[df1.Condition1 == 'Norm_1']
import scipy
scipy.stats.ttest_ind(Norm.Property_Sale_Price, Norm_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

'''
so, next we should make dummy vars for Condition1!
'''
#______let's see 11th Object/ categorical var, Condition2
df1.info()
df1.Condition2.value_counts() # 8 categories
sum(df1.Condition2.value_counts())
'''
let's drop it! 7 catgs even clubbed together are too less!
'''
#______let's see 12th Object/ categorical var, Dwelling_Type
df1.info()
df1.Dwelling_Type.value_counts() # 1643, 166, 87, 55, 44
sum(df1.Dwelling_Type.value_counts())
(166+87+55+44)/1995 # 17.6%

'''
let's make in two groups, Fam and Fam_1
'''
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('1Fam','Fam')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('TwnhsE','Fam_1')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('Duplex','Fam_1')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('Twnhs','Fam_1')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('2fmCon','Fam_1')

df1.Dwelling_Type.value_counts() # 1643, 352
sum(df1.Dwelling_Type.value_counts())

# can Dwelling_Type be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Dwelling_Type).describe(),2)

# Indpndnt sample t test
Fam = df1[df1.Dwelling_Type == 'Fam']
Fam_1 = df1[df1.Dwelling_Type == 'Fam_1']
import scipy
scipy.stats.ttest_ind(Fam.Property_Sale_Price, Fam_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

#______let's see 13th Object/ categorical var, HouseStyle
df1.info()
df1.HouseStyle.value_counts() # 997, 604 and more
sum(df1.HouseStyle.value_counts())
'''
let's make in three groups, Story_1, Story_2 Story_others 
'''
df1['HouseStyle']=df1.get('HouseStyle').replace('1Story','Story_1')
df1['HouseStyle']=df1.get('HouseStyle').replace('2Story','Story_2')
df1['HouseStyle']=df1.get('HouseStyle').replace('1.5Fin','Story_others')

df1['HouseStyle']=df1.get('HouseStyle').replace('SFoyer','Story_others')
df1['HouseStyle']=df1.get('HouseStyle').replace('SLvl','Story_others')
df1['HouseStyle']=df1.get('HouseStyle').replace('2.5Unf','Story_others')
df1['HouseStyle']=df1.get('HouseStyle').replace('1.5Unf','Story_others')
df1['HouseStyle']=df1.get('HouseStyle').replace('2.5Fin','Story_others')

df1.HouseStyle.value_counts() # 997, 604, 394
sum(df1.HouseStyle.value_counts())

# can HouseStyle be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.HouseStyle).describe(),2)
sns.countplot(x ='HouseStyle', data = df1)

# import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Property_Sale_Price ~ HouseStyle', data = df1).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)

#______let's see 14th Object/ categorical var, RoofStyle
df1.info()
df1.RoofStyle.value_counts() # 1578, 370, 19, 15, 11, 1
sum(df1.RoofStyle.value_counts())
'''
let's make in two groups, Gable, Gable_1 
'''
df1['RoofStyle']=df1.get('RoofStyle').replace('Hip','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Flat','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Gambrel','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Mansard','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Shed','Gable_1')
df1.RoofStyle.value_counts() # 1578, 417
sum(df1.RoofStyle.value_counts())
417/1995 #21%
sns.countplot(x ='RoofStyle', data = df1)
# can RoofStyle be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.RoofStyle).describe(),2)

# Indpndnt sample t test
Gable = df1[df1.RoofStyle == 'Gable']
Gable_1 = df1[df1.RoofStyle == 'Gable_1']
import scipy
scipy.stats.ttest_ind(Gable.Property_Sale_Price, Gable_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

#______let's see 15th Object/ categorical var, RoofMatl
df1.info()
df1.RoofMatl.value_counts() # 1964 and 7 catgs too low in counts 17,6,4,1,1,1,1
sum(df1.RoofMatl.value_counts())
''''
let's drop it, 7 catgs together also too small!
'''
#______let's see 16th Object/ categorical var, Exterior1st
df1.info()
df1.Exterior1st.value_counts() # 703 and 14 catgs 
sum(df1.Exterior1st.value_counts())

'''
let's make in two groups, VinylSd, VinylSd_1 
'''
df1['Exterior1st']=df1.get('Exterior1st').replace('HdBoard','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('MetalSd','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('MetalSd','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Wd Sdng','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Plywood','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('CemntBd','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('BrkFace','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('WdShing','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Stucco','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('AsbShng','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('BrkComm','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Stone','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('AsphShn','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('CBlock','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('ImStucc','VinylSd_1')

df1.Exterior1st.value_counts() # 703 and 1292
sum(df1.Exterior1st.value_counts())

sns.countplot(x ='Exterior1st', data = df1)
# can RoofStyle be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Exterior1st).describe(),2)
'''
as the diff in means is around 40k
based on previous learning
let's go for this predictor!
'''
#______let's see 17th Object/ categorical var, Exterior2nd
df1.info()
df1.Exterior2nd.value_counts() # many catgs
sum(df1.Exterior1st.value_counts())
'''
Exterior2nd is very much correlated with Exterior1st,
hence, we will drop Exterior2nd
'''
df1.to_csv('df1.csv')
df3 = pd.read_csv('df1.csv')
#df3.info()

#______let's see 18th Object/ categorical var, MasVnrType
df3.info()
df3.MasVnrType.value_counts() # 1220 (none), 589 (BrkFace), 153(Stone), 20(BrkCmn)
sum(df3.MasVnrType.value_counts()) # 1982
1995-1982 # 13 missing values
'''
let's make in two groups, Vnr_none, Vnr_present
'''
df3['MasVnrType']=df3.get('MasVnrType').replace('None','Vnr_none')
df3['MasVnrType']=df3.get('MasVnrType').replace('BrkFace','Vnr_present')
df3['MasVnrType']=df3.get('MasVnrType').replace('Stone','Vnr_present')
df3['MasVnrType']=df3.get('MasVnrType').replace('BrkCmn','Vnr_present')

df3.MasVnrType.value_counts() # 1220(Vnr_none), 762 (Vnr_present)
sum(df3.MasVnrType.value_counts()) # 1982, still 13 missing
# fill missing values 
df3['MasVnrType']=df3.get('MasVnrType').fillna('Vnr_none')

df3.MasVnrType.value_counts() # 1233(Vnr_none), 762 (Vnr_present)
sum(df3.MasVnrType.value_counts()) # 1995, missing replaced

sns.countplot(x ='MasVnrType', data = df3)
# can MasVnrType be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.MasVnrType).describe(),2)
'''
1) the difference in mean is around 40k, Good Predictor!
2) MasVnrArea is closely connected with MasVnrType,
hence, we decided to include only one in our model
that is MasVnrType
'''

#______let's see 19th Object/ categorical var, ExterQual
df3.info()
df3.ExterQual.value_counts() # 1268, 667, 41, 19
sum(df3.ExterQual.value_counts())
'''
let's make in two groups, TA AND TA_1 
'''
df3['ExterQual']=df3.get('ExterQual').replace('Gd','TA_1')
df3['ExterQual']=df3.get('ExterQual').replace('Ex','TA_1')
df3['ExterQual']=df3.get('ExterQual').replace('Fa','TA_1')

df3.ExterQual.value_counts() # 1268, 727
sum(df3.ExterQual.value_counts())

# can ExterQual be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.ExterQual).describe(),2)
'''
the difference is around 70k,
surely a good predictor, we will keep it! 
'''
#______let's see 20th Object/ categorical var, ExterCond, 28#
df3.info()
df3.ExterCond.value_counts() # 1741, 215, 35, 3, 1
sum(df3.ExterCond.value_counts())
'''
we see very much similarity between
ExterQual and ExterCond,so, we will keep only
ExterQual
'''
#______let's see 21st Object/ categorical var, Foundation, 29#
df3.info()
df3.Foundation.value_counts() #893, 859, 190, 40, 10,3
sum(df3.Foundation.value_counts())
'''
let's make in two groups, CBlock AND CBlock_1 
'''
df3['Foundation']=df3.get('Foundation').replace('PConc','CBlock_1')
df3['Foundation']=df3.get('Foundation').replace('BrkTil','CBlock_1')
df3['Foundation']=df3.get('Foundation').replace('Slab','CBlock_1')
df3['Foundation']=df3.get('Foundation').replace('Stone','CBlock_1')
df3['Foundation']=df3.get('Foundation').replace('Wood','CBlock_1')

df3.Foundation.value_counts() #893, 1103
sum(df3.Foundation.value_counts())
sns.countplot(x ='Foundation', data = df3)

# can Foundation be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.Foundation).describe(),2)
'''
the difference is around 40k,
surely a good predictor, we will keep Foundation! 
'''
#______let's see 21st Object/ categorical var, BsmtQual, 31#
df3.info()
df3.BsmtQual.value_counts() #889, 886, 111, 50
sum(df3.BsmtQual.value_counts()) #1936
1995-1936 # 59 missing cases!
'''
let's make in two groups, TA, TA_1
'''
df3['BsmtQual']=df3.get('BsmtQual').replace('Gd','TA_1')
df3['BsmtQual']=df3.get('BsmtQual').replace('Ex','TA_1')
df3['BsmtQual']=df3.get('BsmtQual').replace('Fa','TA_1')

df3.BsmtQual.value_counts() # 1047 and 889
sum(df3.BsmtQual.value_counts()) # 1936, still 59 missing
# fill missing values 
df3['BsmtQual']=df3.get('BsmtQual').fillna('TA_1')
df3.BsmtQual.value_counts() # 1106, 889
sum(df3.BsmtQual.value_counts()) # 1995, missing replaced
sns.countplot(x ='BsmtQual', data = df3)
# can Foundation be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.BsmtQual).describe(),2)
'''
the difference is around 40k,
surely a good predictor, we will keep BsmtQual! 
'''
#______let's see 22nd Object/ categorical var, BsmtCond, 32#
df3.info()
df3.BsmtCond.value_counts() # 1777, 96, 61, 2
sum(df3.BsmtCond.value_counts()) #1936
(96+61+2)/1936 # 8.2% 
'''
being low quantities, we can ignore!
'''
#______let's see 23rd Object/ categorical var, BsmtExposure , 33#
df3.info()
df3.BsmtExposure .value_counts() # 1334, 300, 154, 146
sum(df3.BsmtExposure .value_counts()) #1934

1995-1934 # 61 missing cases!
'''
let's make in two groups, No, Yes
'''
df3['BsmtExposure']=df3.get('BsmtExposure').replace('Av','Yes')
df3['BsmtExposure']=df3.get('BsmtExposure').replace('Mn','Yes')
df3['BsmtExposure']=df3.get('BsmtExposure').replace('Gd','Yes')

df3.BsmtExposure.value_counts() # 1334(No) and 600(Yes)
sum(df3.BsmtExposure.value_counts()) # 1934, still 61 missing

# fill missing values 
df3['BsmtExposure']=df3.get('BsmtExposure').fillna('No')
df3.BsmtExposure.value_counts() # 1395, 600
sum(df3.BsmtExposure.value_counts()) # 1995, missing replaced
sns.countplot(x ='BsmtExposure', data = df3)
# can BsmtExposure be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.BsmtExposure).describe(),2)
'''
as the diff seems to be large, we will keep it!
'''
#______let's see 24th Object/ categorical var, BsmtFinType1 , 34#
df3.info()
df3.BsmtFinType1 .value_counts() # 590, 540, 316, 209, 172, 109
sum(df3.BsmtFinType1 .value_counts()) #1936
1995-1936 # 59 missing
'''
let's make in two groups, Ungl, Ungl_1
'''
df3['BsmtFinType1']=df3.get('BsmtFinType1').replace('Unf','Ungl')
df3['BsmtFinType1']=df3.get('BsmtFinType1').replace('GLQ','Ungl')
df3['BsmtFinType1']=df3.get('BsmtFinType1').replace('ALQ','Ungl_1')
df3['BsmtFinType1']=df3.get('BsmtFinType1').replace('BLQ','Ungl_1')
df3['BsmtFinType1']=df3.get('BsmtFinType1').replace('Rec','Ungl_1')
df3['BsmtFinType1']=df3.get('BsmtFinType1').replace('LwQ','Ungl_1')

df3.BsmtFinType1.value_counts() # 1130 , 806 
sum(df3.BsmtFinType1.value_counts()) # 1936, still 59 missing

# fill missing values 
df3['BsmtFinType1']=df3.get('BsmtFinType1').fillna('Ungl')
df3.BsmtFinType1.value_counts() # 1189, 806
sum(df3.BsmtFinType1.value_counts()) # 1995, missing replaced
sns.countplot(x ='BsmtFinType1', data = df3)
# can BsmtFinType1 be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.BsmtFinType1).describe(),2)
'''
as diff is large, we will keep it! 
'''
#______let's see 25th Object/ categorical var, BsmtFinType2 , 36#
df3.info()
df3.BsmtFinType2 .value_counts() # 1695, 77, 68, 50, 23, 22
sum(df3.BsmtFinType2 .value_counts()) #1935
'''
too less in 5 catgs, ignore!
'''
#______let's see 26th Object/ categorical var, BsmtFinType2 , 36#
df3.info()
df3.BsmtFinType2 .value_counts() # 1695, 77, 68, 50, 23, 22
sum(df3.BsmtFinType2 .value_counts()) #1935
'''
too less in 5 catgs, ignore!
'''
#______let's see 27th Object/ categorical var, Heating , 40#
df3.info()
df3.Heating .value_counts() # 1948, 24, 10, 8, 3, 2
sum(df3.Heating .value_counts()) #1995
'''
ignore!
'''
#______let's see 28th Object/ categorical var, HeatingQC , 41#
df3.info()
df3.HeatingQC .value_counts() # 976, 616, 334, 68, 1
sum(df3.HeatingQC .value_counts()) #1995
'''
let's make in two groups, Ex, Ex_1
'''
df3['HeatingQC']=df3.get('HeatingQC').replace('TA','Ex_1')
df3['HeatingQC']=df3.get('HeatingQC').replace('Gd','Ex_1')
df3['HeatingQC']=df3.get('HeatingQC').replace('Fa','Ex_1')
df3['HeatingQC']=df3.get('HeatingQC').replace('Po','Ex_1')
df3.HeatingQC .value_counts() # 1019, 976
sum(df3.HeatingQC .value_counts()) #1995

# can HeatingQC be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.HeatingQC).describe(),2)
'''
as diff is large, we will keep it! 
'''
#______let's see 28th Object/ categorical var, CentralAir , 42#
df3.info()
df3.CentralAir .value_counts() # 1857, 138
sum(df3.CentralAir .value_counts()) #1995
'''
ignore! 
'''

#______let's see 28th Object/ categorical var, Electrical , 42#
df3.info()
df3.Electrical .value_counts() # 1823, 127, 40, 3, 1 
sum(df3.Electrical .value_counts()) #1994, 1 missing!!!!!!!!!!!
(127+40+3+1)/1994 # 8.6%
'''
ignore! 
'''
#______let's see 28th Object/ categorical var, KitchenQual , 54#
df3.info()
df3.KitchenQual.value_counts() # 1823, 127, 40, 3, 1 
sum(df3.KitchenQual.value_counts()) #1995 

'''
let's make in two groups, TA, TA_1
'''
df3['KitchenQual']=df3.get('KitchenQual').replace('Gd','TA_1')
df3['KitchenQual']=df3.get('KitchenQual').replace('Ex','TA_1')
df3['KitchenQual']=df3.get('KitchenQual').replace('Fa','TA_1')
df3.KitchenQual.value_counts() # 1041, 954
sum(df3.KitchenQual.value_counts()) #1995 
# can KitchenQual be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.KitchenQual).describe(),2)
'''
as diff is large, we will keep it!
'''

#______let's see 28th Object/ categorical var, Functional , 56#
df3.info()
df3.Functional.value_counts() # 1861, 49, 47, 16, 15, 6, 1 
sum(df3.Functional.value_counts()) #1995 
'''
Typ     1861
Min1      49
Min2      47
Mod       16
Maj1      15
Maj2       6
Sev        1
Name: Functional, dtype: int64

ignore! 
'''
#______let's see 28th Object/ categorical var, FireplaceQu , 58#
df3.info()
df3.FireplaceQu.value_counts() # 1823, 127, 40, 3, 1 
sum(df3.FireplaceQu.value_counts()) #1995 
1008/1995 # 50% 
'''
let's IGNORE!, too many missing values !
'''
#______let's see 28th Object/ categorical var, GarageType , 59#
df3.info()
df3.GarageType.value_counts() # 1197, 536, 103, 26, 12, 8 
sum(df3.GarageType.value_counts()) #1995 
1995-1882 # 113
113/1995 # 5.6%

'''
let's make in two groups, Attchd, Attchd_1
'''
df3['GarageType']=df3.get('GarageType').replace('Detchd','Attchd_1')
df3['GarageType']=df3.get('GarageType').replace('BuiltIn','Attchd_1')
df3['GarageType']=df3.get('GarageType').replace('Basment','Attchd_1')
df3['GarageType']=df3.get('GarageType').replace('CarPort','Attchd_1')
df3['GarageType']=df3.get('GarageType').replace('2Types','Attchd_1')

# fill missing values 
df3['GarageType']=df3.get('GarageType').fillna('Attchd')

df3.GarageType.value_counts() # 1310, 685 
sum(df3.GarageType.value_counts()) #1995
# can GarageType be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.GarageType).describe(),2)
'''
as diff is large, we will keep it!
'''
#______let's see 28th Object/ categorical var, GarageFinish , 61#
df3.info()
df3.GarageFinish.value_counts() 
'''
Unf    854
RFn    590
Fin    438
Name: GarageFinish, dtype: int64
'''
sum(df3.GarageFinish.value_counts()) #1882 
1995-1882 # 113
113/1995 # 5.6%

# fill missing values 
df3['GarageFinish']=df3.get('GarageFinish').fillna('Unf')
sum(df3.GarageFinish.value_counts()) #1995
# can GarageFinish be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.GarageFinish).describe(),2)
'''
looks significant difference
'''
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Property_Sale_Price ~ GarageFinish', data = df1).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
'''
let's keep this 
'''
#______let's see 28th Object/ categorical var, GarageQual , 64#
df3.info()
df3.GarageQual.value_counts() 
'''
TA    1793
Fa      65
Gd      18
Po       3
Ex       3
Name: GarageQual, dtype: int64
ignore! too less in catgs !
'''
#______let's see 28th Object/ categorical var, GarageCond , 65#
df3.info()
df3.GarageCond.value_counts() 
'''
TA    1807
Fa      47
Gd      17
Po       8
Ex       3
Name: GarageCond, dtype: int64
'''
#______let's see 28th Object/ categorical var, PavedDrive , 66#
df3.info()
df3.PavedDrive.value_counts() 
'''
Y    1835
N     116
P      44
Name: PavedDrive, dtype: int64, IGNORE!!
'''
#______let's see 28th Object/ categorical var, PoolQC , 73#
df3.info()
df3.PoolQC.value_counts() 
'''
Gd    4
Fa    2
Ex    1
Name: PoolQC, dtype: int64, IGNORE!
'''
#______let's see 28th Object/ categorical var, Fence , 74#
df3.info()
df3.Fence.value_counts() 
sum(df3.Fence.value_counts()) # 400, 
400/1995 # 80% missing, IGNORE!

#______let's see 28th Object/ categorical var, MiscFeature , 75#
df3.info()
df3.MiscFeature.value_counts() 
sum(df3.MiscFeature.value_counts())  
80/1995 # 96% MISSING! IGNORE!!

#______let's see 28th Object/ categorical var, SaleType , 79#
df3.info()
df3.SaleType.value_counts() 
sum(df3.SaleType.value_counts()) 
'''
WD       1761
New       130
COD        68
ConLD      14
Oth         5
ConLw       5
CWD         5
ConLI       4
Con         3
Name: SaleType, dtype: int64, IGNORE!
'''
#______let's see 28th Object/ categorical var, SaleCondition , 80#
df3.info()
df3.SaleCondition.value_counts() 
sum(df3.SaleCondition.value_counts())
'''
Normal     1660
Abnorml     154
Partial     134
Family       24
Alloca       18
AdjLand       5
'''
'''
let's make in two groups, Normal, Normal_1
'''
df3['SaleCondition']=df3.get('SaleCondition').replace('Abnorml','Normal_1')
df3['SaleCondition']=df3.get('SaleCondition').replace('Partial','Normal_1')
df3['SaleCondition']=df3.get('SaleCondition').replace('Family','Normal_1')
df3['SaleCondition']=df3.get('SaleCondition').replace('Alloca','Normal_1')
df3['SaleCondition']=df3.get('SaleCondition').replace('AdjLand','Normal_1')

df3.SaleCondition.value_counts() #1660, 335 
sum(df3.SaleCondition.value_counts()) # 1995

# can SaleCondition be a good predictor?
round(df3.Property_Sale_Price.groupby(df3.SaleCondition).describe(),2)

# Indpndnt sample t test
Normal = df1[df1.SaleCondition == 'Normal']
Normal_1 = df1[df1.SaleCondition == 'Normal_1']
import scipy
scipy.stats.ttest_ind(Gable.Property_Sale_Price, Gable_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

df3.to_csv('df3.csv')



























#______let's see ANOVA___demo only________________________________
df1.info()
df1.Property_Shape.value_counts() #1265, 658, 53, 14
# count plot on single categorical variable
sns.countplot(x ='Property_Shape', data = df1)
# import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Property_Sale_Price ~ Property_Shape', data = df1).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)

#___________correlation_____demo_____________________
df1.info()
crl = df1[['Property_Sale_Price', 'LotArea', 'GrLivArea']]
crl.head()
sns.heatmap(crl.corr()) # GrLivArea looks having good correlation!! 



'''
 #q1 = 20, q3 = 70; No Missing!
ll_1 = 20 - 1.5*(70-20)
ll_1 #-55
ul_1 = 70 + 1.5*(70-20)
ul_1 #145
#________outliers
Q1 = np.percentile(df4.Dwell_Type, 25, interpolation = 'midpoint')  
Q2 = np.percentile(df4.Dwell_Type, 50, interpolation = 'midpoint')  
Q3 = np.percentile(df4.Dwell_Type, 75, interpolation = 'midpoint')   
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)   
IQR = Q3 - Q1  
print('Interquartile range is', IQR) # 50
#df4.info()
low_lim = Q1 - 1.5 * IQR #-55
up_lim = Q3 + 1.5 * IQR # 145
print('low_limit is', low_lim) 
print('up_limit is', up_lim) #145
# anything q3 + 1.5*iqr is outlier on higher side
#_______counting outliers
len(df4.Dwell_Type[df4.Dwell_Type > 350]) # 1

# let's remove/select
df4 = df4[df4.Dwell_Type <= 300] # <=
df4.info() #1993 ; 2 removed 

#______histogram
#_run in block
plt.hist(df4.Dwell_Type, bins = 'auto', facecolor = 'blue')
plt.xlabel('Dwell_Type')
plt.ylabel('counts')
plt.title('Histogram of Dwell_Type')

#____boxplot
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'red')
df4['Dwell_Type'].plot.box(color=props2, patch_artist = True, vert = False)

df5 = df4 
# let's remove/select
df5 = df5[df5.Dwell_Type <= 145] # <=
df5.info() #1993 ; 2 removed 

#______histogram
#_run in block
plt.hist(df5.Dwell_Type, bins = 'auto', facecolor = 'blue')
plt.xlabel('Dwell_Type')
plt.ylabel('counts')
plt.title('Histogram of Dwell_Type')

#____boxplot
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'red')
df5['Dwell_Type'].plot.box(color=props2, patch_artist = True, vert = False)
'''






