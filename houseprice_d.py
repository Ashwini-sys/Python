# God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("D:/data _science/PYTHON/Linear_Regression_Python/HousePrices.csv")
df = pd.DataFrame(df)
df.info() #2073, 81, float=3; int =35, object 43
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
df1 = df[df.Property_Sale_Price <= 340000] 
df1.info() #1990 ; 2073-78 = 1995
2073-78 # 1995

#-----------------------------------------------------------------------------------------------------

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

"""#******************************************************************************************---------------------------------------------------------------------------------------------"""
#Zone_Class
#______let's see some categorical vars
df1.Zone_Class.value_counts() #1537, 320, 98, 22, 14
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
#______let's see 2nd Object/ categorical var
df1.info()
df1.Road_Type.value_counts() #1985, 10 
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='Road_Type', data = df1) 
'''
better not include this
'''
#______let's see 3rd Object/ categorical var
df1.info()
df1.Alley.value_counts() # 72, 57  
# count plot on single categorical variable
import seaborn as sns
sns.countplot(x ='Alley', data = df1) 
(72+57)/1995 # 6.5% 
'''
better not include Alley
'''
#______let's see 4th Object/ categorical var
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
# p_value is >0.05; Ho Accept; Bad Predictor


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
that if difference in means is like this - 1,80k and 1,67k
var is a good differentiator
so, we can take a call now only and decided to 
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
#Condition1
df1.info()
df1.Condition1.value_counts()
sum(df1.Condition1.value_counts())
# count plot on single categorical variable
#import seaborn as sns
#sns.countplot(x ='Neighborhood', data = df1)
'''
let's make in two groups, Norm and Norm_1
'''
df1['Condition1']=df1.get('Condition1').replace('Feedr','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('Artery','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRAn','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('PosN','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRAe','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('PosA','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRNn','Norm_1')
df1['Condition1']=df1.get('Condition1').replace('RRNe','Norm_1')

df1.Condition1.value_counts() # 1719 and 276 
sum(df1.Condition1.value_counts()) # 1995

import seaborn as sns
sns.countplot(x ='Condition1', data = df1)

# can Neighborhood be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Condition1).describe(),2)
'''
learnt from previous analysis
that if difference in means is like this - 1,80k and 1,67k
var is a good differentiator
so, we can take a call now only and decided to 
include in the model, however, better cross check through
indpndnt test
'''
round(df1.Property_Sale_Price.groupby(df1.Condition1).min(),2)
# Indpndnt sample t test
Norm = df1[df1.Condition1 == 'Norm']
Norm_1 = df1[df1.Condition1 == 'Norm_1']

import scipy
scipy.stats.ttest_ind(Norm.Property_Sale_Price, Norm_1.Property_Sale_Price)
#PValue is less than 0.05 so significant

#Condition2 
df1.info()

df1.Condition2.value_counts() # 1719 and 276 
sum(df1.Condition2.value_counts()) # 1995

import seaborn as sns
sns.countplot(x ='Condition2', data = df1)

#not significant cause so many missing values

#------------------------------------------------------------------------------------------
#Dwelling_Type
df1.info()

df1.Dwelling_Type.value_counts() #1643, 166, 87, 55, 44
sum(df1.Dwelling_Type.value_counts())

(166+87+55+44)/1995 # 17.6 %

""" Lets make as two group 1Fam and Fam_1"""

df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('1Fam','Fam')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('TwnhsE','Fam_1')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('Duplex','Fam_1')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('Twnhs','Fam_1')
df1['Dwelling_Type']=df1.get('Dwelling_Type').replace('2fmCon','Fam_1')

df1.Dwelling_Type.value_counts() # 1643 and 352 
sum(df1.Dwelling_Type.value_counts()) # 1995

import seaborn as sns
sns.countplot(x ='Dwelling_Type', data = df1)

# can Dwelling_Type be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Dwelling_Type).describe(),2)
round(df1.Property_Sale_Price.groupby(df1.Dwelling_Type).min(),2)

# Indpndnt sample t test
Fam = df1[df1.Dwelling_Type == 'Fam']
Fam_1 = df1[df1.Dwelling_Type == 'Fam_1']

import scipy
scipy.stats.ttest_ind(Fam.Property_Sale_Price, Fam_1.Property_Sale_Price)
#P value is less than 0.05 HO ==> rejected so its a significant

#-------------------------------------------------------------------------------------
#HouseStyle
df1.info()
df1.HouseStyle.value_counts() #604, 212, 88, 54, 16, 15, 9
sum(df1.HouseStyle.value_counts())
import seaborn as sns
sns.countplot(x ='HouseStyle', data = df1)

"""Lets gouped togethr as 2 category as story 1 and story 2 and Story_Others"""

df1['HouseStyle']=df1.get('HouseStyle').replace('1Story','Story_1')
df1['HouseStyle']=df1.get('HouseStyle').replace('2Story','Story_2')

df1['HouseStyle']=df1.get('HouseStyle').replace('1.5Fin','Story_Others')
df1['HouseStyle']=df1.get('HouseStyle').replace('SLvl','Story_Others')
df1['HouseStyle']=df1.get('HouseStyle').replace('SFoyer','Story_Others')
df1['HouseStyle']=df1.get('HouseStyle').replace('2.5Unf','Story_Others')
df1['HouseStyle']=df1.get('HouseStyle').replace('1.5Unf','Story_Others')
df1['HouseStyle']=df1.get('HouseStyle').replace('2.5Fin','Story_Others')

df1.HouseStyle.value_counts() #997, 604, 394
sum(df1.HouseStyle.value_counts())

import seaborn as sns
sns.countplot(x ='HouseStyle', data = df1)

# can HouseStyle be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.HouseStyle).describe(),2)
round(df1.Property_Sale_Price.groupby(df1.HouseStyle).min(),2)

#Independent sample t test
Story_1 = df1[df1.HouseStyle == 'Story_1']
Story_2 = df1[df1.HouseStyle == 'Story_2']
Story_Others = df1[df1.HouseStyle == 'Story_Others']

""" Null Hypothesis : no Significance differance between grp means
ALTernative Hypothesis : atleast 1 grp is different from others

Story 1 is close to story_others , so we can clubed them"""

#Anova test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Property_Sale_Price ~ HouseStyle', data = df1).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#here P value is significant we can consider this column
#-------------------------------------------------------------------------------------------

#RoofStyle
df1.info()
df1.RoofStyle.value_counts() #370, 19,15,11,2 
sum(df1.RoofStyle.value_counts())
import seaborn as sns
sns.countplot(x ='RoofStyle', data = df1)

"""Lets do two grps as Gable and Gable_1"""

df1['RoofStyle']=df1.get('RoofStyle').replace('Hip','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Flat','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Gambrel','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Mansard','Gable_1')
df1['RoofStyle']=df1.get('RoofStyle').replace('Shed','Gable_1')

df1.RoofStyle.value_counts() #1578, 417
sum(df1.RoofStyle.value_counts())

import seaborn as sns
sns.countplot(x ='RoofStyle', data = df1)
# can HouseStyle be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.RoofStyle).describe(),2)
round(df1.Property_Sale_Price.groupby(df1.RoofStyle).min(),2)

#Independent sample t test
Gable = df1[df1.RoofStyle == 'Gable']
Gable_1 = df1[df1.RoofStyle == 'Gable_1']

import scipy
scipy.stats.ttest_ind(Gable.Property_Sale_Price, Gable_1.Property_Sale_Price)
#P value is less than 0.05is rejected, its a good predictor

#-------------------------------------------------------------------------------------------
#RoofMatl
df1.info()
df1.RoofMatl.value_counts() #1964 and 7 catgrs are so low 17,6,4,1,1,1,1,1
sum(df1.RoofMatl.value_counts())
import seaborn as sns
sns.countplot(x ='RoofMatl', data = df1)
"""according to plot its more towords one cat. :so we can skip this var"""

#-------------------------------------------------------------------------------------------
#Exterior1st
df1.info()
df1.Exterior1st.value_counts() # 703 and 14 other categories
sum(df1.Exterior1st.value_counts())
import seaborn as sns
sns.countplot(x ='Exterior1st', data = df1)

"""Lets do 2 category VinylSd and VinylSd_1"""
df1['Exterior1st']=df1.get('Exterior1st').replace('HdBoard','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('MetalSd','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Wd Sdng','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Plywood','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('CemntBd','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('BrkFace','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('WdShing','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Stucco','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('AsbShng','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('Stone','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('BrkComm','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('AsphShn','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('ImStucc','VinylSd_1')
df1['Exterior1st']=df1.get('Exterior1st').replace('CBlock','VinylSd_1')

df1.Exterior1st.value_counts() #1292, 703
sum(df1.Exterior1st.value_counts())

import seaborn as sns
sns.countplot(x ='Exterior1st', data = df1)
# can HouseStyle be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Exterior1st).describe(),2)

"""MEAN value is 198832.06 and 155849.52 and the diff in both mean 
is arounf 40K based on previous learning so its be a good predicot"""

round(df1.Property_Sale_Price.groupby(df1.Exterior1st).min(),2)

#Independent sample t test
VinylSd = df1[df1.Exterior1st == 'VinylSd']
VinylSd_1 = df1[df1.Exterior1st == 'VinylSd_1']

import scipy
scipy.stats.ttest_ind(VinylSd.Property_Sale_Price, VinylSd_1.Property_Sale_Price)
#P value is less than 0.05is rejected, its a good predictor

#-------------------------------------------------------------------------------------------
#Exterior2nd
df1.info()
df1.Exterior2nd.value_counts() # 703 and 15 other categories
sum(df1.Exterior2nd.value_counts())
import seaborn as sns
sns.countplot(x ='Exterior2nd', data = df1)

#this is the corr code but need to convert data in numeric
#df1['Exterior1st'].corr(df1['Exterior2nd'])
#df1.corr()

#-----cross tabualtion--
#------------Exterior1st vs Exterior2nd-------
pd.crosstab(df1.Exterior1st, df1.Exterior2nd, margins = True)
#margins = true gives row colums total
df2 = df[df.Property_Sale_Price <= 340000]
#snehils code
pd.crosstab(df1.Exterior1st, df1.Exterior2nd)
pd.crosstab(df2.Exterior1st, df2.Exterior2nd).to_string()

"""Exteriror2nd is very much correlatd with Exterior1st hence we will drop Exterior2nd"""

#-------------------------------------------------------------------------------------------
#MasVnrType
df1.info()
df1.MasVnrType.value_counts() # 703 and 15 other categories
sum(df1.MasVnrType.value_counts())
import seaborn as sns
sns.countplot(x ='MasVnrType', data = df1)

df1.MasVnrType.describe() # na will be ignored in calculations
#df1['MasVnrType'] = df1['MasVnrType'].fillna(df1['MasVnrType'].mode())
#print(df1)

df1['MasVnrType']=df1.get('MasVnrType').fillna('Vnr_none')

"""Lets do 2 catgrs as Vnr_None ans Vnr_Present"""

df1['MasVnrType']=df1.get('MasVnrType').replace('None','Vnr_None')
df1['MasVnrType']=df1.get('MasVnrType').replace('Vnr_none','Vnr_None')
df1['MasVnrType']=df1.get('MasVnrType').replace('BrkFace','Vnr_Present')
df1['MasVnrType']=df1.get('MasVnrType').replace('Stone','Vnr_Present')
df1['MasVnrType']=df1.get('MasVnrType').replace('BrkCmn','Vnr_Present')

df1.MasVnrType.value_counts() # 1220 and 762 other categories
sum(df1.MasVnrType.value_counts())#1982 and 13 missing values
import seaborn as sns
sns.countplot(x ='MasVnrType', data = df1)
# can MasVnrType be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.MasVnrType).describe(),2)

#Independent sample t test
Vnr_None = df1[df1.MasVnrType == 'Vnr_None']
Vnr_Present = df1[df1.MasVnrType == 'Vnr_Present']

import scipy
scipy.stats.ttest_ind(Vnr_None.Property_Sale_Price, Vnr_Present.Property_Sale_Price)
##P value is less than 0.05is rejected, its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#ExterQual
df1.info()
df1.ExterQual.value_counts() #
sum(df1.ExterQual.value_counts())
import seaborn as sns
sns.countplot(x ='ExterQual', data = df1) 

"""Better to clubbed Gd, Ex and Fa as TA_1"""

df1['ExterQual']=df1.get('ExterQual').replace('Gd','TA_1')
df1['ExterQual']=df1.get('ExterQual').replace('Ex','TA_1')
df1['ExterQual']=df1.get('ExterQual').replace('Fa','TA_1')

df1.ExterQual.value_counts() # 1220 and 762 other categories
sum(df1.ExterQual.value_counts())#1992
import seaborn as sns
sns.countplot(x ='ExterQual', data = df1)
# can MasVnrType be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.ExterQual).describe(),2)

#Independent sample t test
TA = df1[df1.ExterQual == 'TA']
TA_1 = df1[df1.ExterQual == 'TA_1']

import scipy
scipy.stats.ttest_ind(TA.Property_Sale_Price, TA_1.Property_Sale_Price)
##P value is less than 0.05is rejected, its a good predictor

#--------------------------------------------------------------------------------------------------------------------
# 28th ExterCond
df1.info()
df1.ExterCond.value_counts() #1741,215,35,3,1
sum(df1.ExterCond.value_counts())
import seaborn as sns
sns.countplot(x ='ExterCond', data = df1) 

"""Better to clubbed Gd, Fa, EX, Po as TA_A and TA_B"""

df1['ExterCond']=df1.get('ExterCond').replace('TA','TA_A')

df1['ExterCond']=df1.get('ExterCond').replace('Gd','TA_B')
df1['ExterCond']=df1.get('ExterCond').replace('Fa','TA_B')
df1['ExterCond']=df1.get('ExterCond').replace('Ex','TA_B')
df1['ExterCond']=df1.get('ExterCond').replace('Po','TA_B')

df1.ExterCond.value_counts() # 1741 and 254 other categories
sum(df1.ExterCond.value_counts())#1992
import seaborn as sns
sns.countplot(x ='ExterCond', data = df1)
# can MasVnrType be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.ExterCond).describe(),2)

#Independent sample t test
TA_A = df1[df1.ExterCond == 'TA_A']
TA_B = df1[df1.ExterCond == 'TA_B']

import scipy
scipy.stats.ttest_ind(TA_A.Property_Sale_Price, TA_B.Property_Sale_Price)
##P value is 0.0004041813211341496 less than 0.05 is rejected, its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#Foundation
df1.info()
df1.Foundation.value_counts() #1741,215,35,3,1
sum(df1.Foundation.value_counts())
import seaborn as sns
sns.countplot(x ='Foundation', data = df1) 

"""Better to clubbed PConc, BrkTil, Slab, Stone,Wood as CBlock and CBlock_1"""

df1['Foundation']=df1.get('Foundation').replace('PConc','CBlock_1')
df1['Foundation']=df1.get('Foundation').replace('BrkTil','CBlock_1')
df1['Foundation']=df1.get('Foundation').replace('Slab','CBlock_1')
df1['Foundation']=df1.get('Foundation').replace('Stone','CBlock_1')
df1['Foundation']=df1.get('Foundation').replace('Wood','CBlock_1')

df1.Foundation.value_counts() # 1102 and 893 other categories
sum(df1.Foundation.value_counts())#1992
import seaborn as sns
sns.countplot(x ='Foundation', data = df1)
# can MasVnrType be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Foundation).describe(),2)

#Independent sample t test
CBlock = df1[df1.Foundation == 'CBlock']
CBlock_1 = df1[df1.Foundation == 'CBlock_1']

import scipy
scipy.stats.ttest_ind(CBlock.Property_Sale_Price, CBlock_1.Property_Sale_Price)
##P value is less than 0.05 is rejected, its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#BsmtQual
df1.info()
df1.BsmtQual.value_counts() #889,886,111,50
sum(df1.BsmtQual.value_counts())#1936 ,59 missing values
import seaborn as sns
sns.countplot(x ='BsmtQual', data = df1) 
#1995-1936
#filling missing vaues
df1['BsmtQual']=df1.get('BsmtQual').fillna('BsmtQual_1')
sum(df1.BsmtQual.value_counts())#now 1995

"""Better to clubbed  Gd, Ex and Fa as GD_1 and TA as GD"""

df1['BsmtQual']=df1.get('BsmtQual').replace('TA','GD')
df1['BsmtQual']=df1.get('BsmtQual').replace('Gd','GD_1')
df1['BsmtQual']=df1.get('BsmtQual').replace('Ex','GD_1')
df1['BsmtQual']=df1.get('BsmtQual').replace('Fa','GD_1')
df1['BsmtQual']=df1.get('BsmtQual').replace('BsmtQual_1','GD_1')


df1.BsmtQual.value_counts() # 1102 and 893 other categories
sum(df1.BsmtQual.value_counts())#1992
import seaborn as sns
sns.countplot(x ='BsmtQual', data = df1)
# can MasVnrType be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.BsmtQual).describe(),2)

#Independent sample t test
GD = df1[df1.BsmtQual == 'GD']
GD_1 = df1[df1.BsmtQual == 'GD_1']

import scipy
scipy.stats.ttest_ind(GD.Property_Sale_Price, GD_1.Property_Sale_Price)
##P value is less than 0.05 is rejected, its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#BsmtCond
df1.info()
df1.BsmtCond.value_counts() #889,886,111,50
sum(df1.BsmtCond.value_counts())#1936 ,59 missing values
import seaborn as sns
sns.countplot(x ='BsmtCond', data = df1) 
#1995-1936
#filling missing vaues
df1['BsmtCond']=df1.get('BsmtCond').fillna('BsmtCond_1')
sum(df1.BsmtCond.value_counts())#now 1995

"""Better to clubbed  Gd, Fa and PO as Bsmtc_TA_1 and TA as Bsmtc_TA"""

df1['BsmtCond']=df1.get('BsmtCond').replace('TA','Bsmtc_TA')

df1['BsmtCond']=df1.get('BsmtCond').replace('BsmtCond_1','Bsmtc_TA_1')
df1['BsmtCond']=df1.get('BsmtCond').replace('Gd','Bsmtc_TA_1')
df1['BsmtCond']=df1.get('BsmtCond').replace('Fa','Bsmtc_TA_1')
df1['BsmtCond']=df1.get('BsmtCond').replace('Po','Bsmtc_TA_1')
df1['BsmtCond']=df1.get('BsmtCond').replace('GD_1','Bsmtc_TA_1')

df1.BsmtCond.value_counts() # 1777 and 218 other categories
sum(df1.BsmtCond.value_counts())#1992
import seaborn as sns
sns.countplot(x ='BsmtCond', data = df1)
# can BsmtCond be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.BsmtCond).describe(),2)

#Independent sample t test
Bsmtc_TA = df1[df1.BsmtCond == 'Bsmtc_TA']
Bsmtc_TA_1 = df1[df1.BsmtCond == 'Bsmtc_TA_1']

import scipy
scipy.stats.ttest_ind(Bsmtc_TA.Property_Sale_Price, Bsmtc_TA_1.Property_Sale_Price)
##P value is less than 0.05 is rejected, its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#BsmtExposure
df1.info()
df1.BsmtExposure.value_counts() #1334and 300,154,146
sum(df1.BsmtExposure.value_counts())#1936 ,59 missing values
import seaborn as sns
sns.countplot(x ='BsmtExposure', data = df1) 
1995-1934
#filling missing vaues
df1['BsmtExposure']=df1.get('BsmtExposure').fillna('BsmtExposure_1')
sum(df1.BsmtExposure.value_counts())#now 1995

"""Better to clubbed  Gd, Mn and Av as Yes """
df1['BsmtExposure']=df1.get('BsmtExposure').replace('BsmtExposure_1','Yes')
df1['BsmtExposure']=df1.get('BsmtExposure').replace('Gd','Yes')
df1['BsmtExposure']=df1.get('BsmtExposure').replace('Mn','Yes')
df1['BsmtExposure']=df1.get('BsmtExposure').replace('Av','Yes')

df1.BsmtExposure.value_counts() # 1334 and 661 other categories
sum(df1.BsmtExposure.value_counts())#1992
import seaborn as sns
sns.countplot(x ='BsmtExposure', data = df1)
# can BsmtExposure be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.BsmtExposure).describe(),2)

#Independent sample t test
No = df1[df1.BsmtExposure == 'No']
Yes = df1[df1.BsmtExposure == 'Yes']

import scipy
scipy.stats.ttest_ind(No.Property_Sale_Price, Yes.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor
#--------------------------------------------------------------------------------------------------------------------
#BsmtFinType1
df1.info()
df1.BsmtFinType1.value_counts() #1334and 300,154,146
sum(df1.BsmtFinType1.value_counts())#1936 ,59 missing values
import seaborn as sns
sns.countplot(x ='BsmtFinType1', data = df1)
1995-1936
df1['BsmtFinType1']=df1.get('BsmtFinType1').fillna('BsmtFinType1_1')
sum(df1.BsmtFinType1.value_counts())#1995

"""Better to clubbed  ALQ, Unf, BLQ, LWQ and Rec  as GLQ_1 """

df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('BsmtFinType1_1','GLQ_1')

df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('ALQ','GLQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('Unf','GLQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('Rec','GLQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('BLQ','GLQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('LwQ','GLQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('BsmtFinType1','GLQ_1')

df1.BsmtFinType1.value_counts() # 1455 and 540 other categories
sum(df1.BsmtFinType1.value_counts())#1992
import seaborn as sns
sns.countplot(x ='BsmtFinType1', data = df1)
# can BsmtExposure be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.BsmtFinType1).describe(),2)

#Independent sample t test
GLQ = df1[df1.BsmtFinType1 == 'GLQ']
GLQ_1 = df1[df1.BsmtFinType1 == 'GLQ_1']

import scipy
scipy.stats.ttest_ind(GLQ.Property_Sale_Price, GLQ_1.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor
#############################################################################################
#by another aspect
#____let's see 24th Object/ categorical var, BsmtFinType1
#__let's see 24th Object/ categorical var, BsmtFinType1
df1.info()

# Since Categorical, using Cross Tab for seeing correlation Between BsmtFinType1
# nd BsmtFinType2
import pandas as pd
pd.crosstab(df1.BsmtFinType1, df1.BsmtFinType2)
# Shows No Correlation, Both cn be considered

df1.BsmtFinType1.value_counts() 
sum(df1.BsmtFinType1.value_counts()) # Missing Values Noted (1995-1936 = 59)


#  Counting Total Null Values 
df1['BsmtFinType1'].isnull().sum()

# Replacing Null Values with string 'None'
x= 'No_BsmtFin1'
df1['BsmtFinType1'] =  df1['BsmtFinType1']. fillna('No_BsmtFin1')
df1['BsmtFinType1'].isnull().sum()

# import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Property_Sale_Price ~ BsmtFinType1', data = df1).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
'''
Ho: there is no significant diff between sales Prices across d diff categories

Alt Ho: atleast 1 category's sales is different frm other categories sales
'''

# boxpLot of Property_Sale_Price versus BsmtFinType1
kf = df1[[ 'Property_Sale_Price' , 'BsmtFinType1' ]]
kf.boxplot (by = 'BsmtFinType1' )


# Tukey Test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df1.Property_Sale_Price, df1.BsmtFinType1, alpha=0.05)                      
print(tukey)

# let's make in 3 groups, No_BsmtFin1, GLQ and all others ALQ_1
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('Unf','ALQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('ALQ','ALQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('BLQ','ALQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('Rec','ALQ_1')
df1['BsmtFinType1']=df1.get('BsmtFinType1').replace('LwQ','ALQ_1')

import seaborn as sns
sns.countplot(x ='BsmtFinType1', data = df1)

# can BsmtFinType1 be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.BsmtFinType1).describe(),2)

# import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Property_Sale_Price ~ BsmtFinType1', data = df1).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)

# Tukey Test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df1.Property_Sale_Price, df1.BsmtFinType1, alpha=0.05)                      
print(tukey)
#Good predictor

#--------------------------------------------------------------------------------------------------------------------
#BsmtFinType2
df1.info()
df1.BsmtFinType2.value_counts() #1334and 300,154,146
sum(df1.BsmtFinType2.value_counts())#1935 ,62 missing values
import seaborn as sns
sns.countplot(x ='BsmtFinType2', data = df1)
#Replacing NA values
df1['BsmtFinType2']=df1.get('BsmtFinType2').fillna('BsmtFinType2_1')
sum(df1.BsmtFinType2.value_counts())#1995

"""Better to clubbed  ALQ, BLQ, LWQ, GLQ and Rec  as Unf_1 """

df1['BsmtFinType2']=df1.get('BsmtFinType2').replace('BsmtFinType2_1','Unf_1')
df1['BsmtFinType2']=df1.get('BsmtFinType2').replace('ALQ','Unf_1')
df1['BsmtFinType2']=df1.get('BsmtFinType2').replace('BLQ','Unf_1')
df1['BsmtFinType2']=df1.get('BsmtFinType2').replace('Rec','Unf_1')
df1['BsmtFinType2']=df1.get('BsmtFinType2').replace('LwQ','Unf_1')
df1['BsmtFinType2']=df1.get('BsmtFinType2').replace('GLQ','Unf_1')
df1['BsmtFinType2']=df1.get('BsmtFinType2').replace('GLQ_1','Unf')

df1.BsmtFinType2.value_counts() #1455 and 540

sum(df1.BsmtFinType2.value_counts())#1992
import seaborn as sns
sns.countplot(x ='BsmtFinType2', data = df1)
# can BsmtExposure be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.BsmtFinType2).describe(),2)

#Independent sample t test
Unf = df1[df1.BsmtFinType2 == 'Unf']
Unf_1 = df1[df1.BsmtFinType2 == 'Unf_1']

import scipy
scipy.stats.ttest_ind(Unf.Property_Sale_Price, Unf_1.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor
#--------------------------------------------------------------------------------------------------------------------
#Heating
df1.info()
df1.Heating.value_counts() #1334and 300,154,146
sum(df1.Heating.value_counts())#1995
import seaborn as sns
sns.countplot(x ='Heating', data = df1)
#Replacing NA values
sum(df1.Heating.value_counts())#1995

"""Better to clubbed  GasW, Grav, Wall, Floor and OthW  as GasA_1 """

df1['Heating']=df1.get('Heating').replace('GasW','GasA_1')
df1['Heating']=df1.get('Heating').replace('Grav','GasA_1')
df1['Heating']=df1.get('Heating').replace('Wall','GasA_1')
df1['Heating']=df1.get('Heating').replace('Floor','GasA_1')
df1['Heating']=df1.get('Heating').replace('OthW','GasA_1')


df1.Heating.value_counts() #1948 and 47
sum(df1.Heating.value_counts())# 1995

import seaborn as sns
sns.countplot(x ='Heating', data = df1)
# can BsmtExposure be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Heating).describe(),2)

#Independent sample t test
GasA = df1[df1.Heating == 'GasA']
GasA_1 = df1[df1.Heating == 'GasA_1']

import scipy
scipy.stats.ttest_ind(GasA.Property_Sale_Price, GasA_1.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor
#--------------------------------------------------------------------------------------------------------------------
#HeatingQC
df1.info()
df1.HeatingQC.value_counts() #976, 616, 334, 68 and 1
sum(df1.HeatingQC.value_counts())#1995
import seaborn as sns
sns.countplot(x ='HeatingQC', data = df1)
sum(df1.HeatingQC.value_counts())#1995

""" Better to Gd, TA, Fa, Po clubbed as Ex_1"""

df1['HeatingQC']=df1.get('HeatingQC').replace('Gd','Ex_1')
df1['HeatingQC']=df1.get('HeatingQC').replace('TA','Ex_1')
df1['HeatingQC']=df1.get('HeatingQC').replace('Fa','Ex_1')
df1['HeatingQC']=df1.get('HeatingQC').replace('Po','Ex_1')

df1.HeatingQC.value_counts() #1019 and 976
sum(df1.HeatingQC.value_counts())# 1995

import seaborn as sns
sns.countplot(x ='HeatingQC', data = df1)
# can BsmtExposure be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.HeatingQC).describe(),2)

#Independent sample t test
Ex = df1[df1.HeatingQC == 'Ex']
Ex_1 = df1[df1.HeatingQC == 'Ex_1']

import scipy
scipy.stats.ttest_ind(Ex.Property_Sale_Price, Ex_1.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#CentralAir
df1.info()
df1.CentralAir.value_counts() #976, 616, 334, 68 and 1
sum(df1.CentralAir.value_counts())#1995
import seaborn as sns
sns.countplot(x ='CentralAir', data = df1)
sum(df1.CentralAir.value_counts())#1995

"""Its a more towards no what we should do keep or ignore?"""
#--------------------------------------------------------------------------------------------------------------------
#Electrical
df1.info()
df1.Electrical.value_counts() #976, 616, 334, 68 and 1
sum(df1.Electrical.value_counts())#1994 , 1 missing value (no impact)
import seaborn as sns
sns.countplot(x ='Electrical', data = df1)

""" Better to club FuseA, FuseF, FuseP, Mix clubbed as SBrkr_1"""

df1['Electrical']=df1.get('Electrical').replace('FuseA','SBrkr_1')
df1['Electrical']=df1.get('Electrical').replace('FuseF','SBrkr_1')
df1['Electrical']=df1.get('Electrical').replace('FuseP','SBrkr_1')
df1['Electrical']=df1.get('Electrical').replace('Mix','SBrkr_1')

df1.Electrical.value_counts() #1019 and 976
sum(df1.Electrical.value_counts())# 1995

import seaborn as sns
sns.countplot(x ='Electrical', data = df1)
# can Electrical be a good predictor?
round(df1.Property_Sale_Price.groupby(df1.Electrical).describe(),2)

#Independent sample t test
SBrkr = df1[df1.Electrical == 'SBrkr']
SBrkr_1 = df1[df1.Electrical == 'SBrkr_1']

import scipy
scipy.stats.ttest_ind(Ex.Property_Sale_Price, Ex_1.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#KitchenQual
df1.info()
df1.KitchenQual.value_counts() #976, 616, 334, 68 and 1
sum(df1.KitchenQual.value_counts())#1995
import seaborn as sns
sns.countplot(x ='KitchenQual', data = df1)

""" Better to clubb Gd, Ex, Fa, as KitQ_1 , TA as KitQ"""

df1['KitchenQual']=df1.get('KitchenQual').replace('TA','KitQ')
df1['KitchenQual']=df1.get('KitchenQual').replace('Gd','KitQ_1')
df1['KitchenQual']=df1.get('KitchenQual').replace('Ex','KitQ_1')
df1['KitchenQual']=df1.get('KitchenQual').replace('Fa','KitQ_1')

df1.KitchenQual.value_counts() #1041, 954
sum(df1.KitchenQual.value_counts())#1995
import seaborn as sns
sns.countplot(x ='KitchenQual', data = df1)

round(df1.Property_Sale_Price.groupby(df1.KitchenQual).describe(),2)

#Independent sample t test
KitQ = df1[df1.KitchenQual == 'KitQ']
KitQ_1 = df1[df1.KitchenQual == 'KitQ_1']

import scipy
scipy.stats.ttest_ind(KitQ.Property_Sale_Price, KitQ_1.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor
#--------------------------------------------------------------------------------------------------------------------
#Functional 55th
df1.info()
df1.Functional.value_counts() #976, 616, 334, 68 and 1
sum(df1.Functional.value_counts())#1995
import seaborn as sns
sns.countplot(x ='Functional', data = df1)

"""More towards one category, wecan skip"""

49+47+16+15+6+1# =134 and 1861
#--------------------------------------------------------------------------------------------------------------------
#FireplaceQu
df1.info()
df1.FireplaceQu.value_counts() 
sum(df1.FireplaceQu.value_counts())#1008
import seaborn as sns
sns.countplot(x ='FireplaceQu', data = df1)
1995-1008#987 missing values

#--------------------------------------------------------------------------------------------------------------------
#GarageType
df1.info()
df1.GarageType.value_counts() 
sum(df1.GarageType.value_counts())#1882
import seaborn as sns
sns.countplot(x ='GarageType', data = df1)
1995-1882#113 missing values


#--------------------------------------------------------------------------------------------------------------------
#GarageFinish
df1.info()
df1.GarageFinish.value_counts() 
sum(df1.GarageFinish.value_counts())#1882
import seaborn as sns
sns.countplot(x ='GarageFinish', data = df1)
1995-1882#113 missing values
#Replacing NA values
df1['GarageFinish']=df1.get('GarageFinish').fillna('GarageFinish_1')
sum(df1.GarageFinish.value_counts())#1995

"""Better to clubbed RFn, Fin, GarageFinish_1 as GRF_Unf_1 and Unf as GRF_Unf """

df1['GarageFinish']=df1.get('GarageFinish').replace('Unf','GRF_Unf')

df1['GarageFinish']=df1.get('GarageFinish').replace('GarageFinish_1','GRF_Unf_1')
df1['GarageFinish']=df1.get('GarageFinish').replace('RFn','GRF_Unf_1')
df1['GarageFinish']=df1.get('GarageFinish').replace('Fin','GRF_Unf_1')

df1.GarageFinish.value_counts() #1141, 854
sum(df1.GarageFinish.value_counts())#1995
import seaborn as sns
sns.countplot(x ='GarageFinish', data = df1)

round(df1.Property_Sale_Price.groupby(df1.GarageFinish).describe(),2)

#Independent sample t test
GRF_Unf = df1[df1.GarageFinish == 'GRF_Unf']
GRF_Unf_1 = df1[df1.GarageFinish == 'GRF_Unf_1']

import scipy
scipy.stats.ttest_ind(GRF_Unf.Property_Sale_Price, GRF_Unf_1.Property_Sale_Price)
#P value is less thn 0.05, so its a good predictor

#--------------------------------------------------------------------------------------------------------------------
#GarageQual
df1.info()
df1.GarageQual.value_counts()# 1793 and other 5 categories
sum(df1.GarageQual.value_counts())#1882
import seaborn as sns
sns.countplot(x ='GarageQual', data = df1)
1995-1882#113 missing values
#Replacing NA values
df1['GarageQual']=df1.get('GarageQual').fillna('GarageQual_1')
sum(df1.GarageFinish.value_counts())#1995

"""Its more towords TA Category, so we can skip"""
#--------------------------------------------------------------------------------------------------------------------
#GarageCond
df1.info()
df1.GarageCond.value_counts()# 1793 and other 5 categories
sum(df1.GarageCond.value_counts())#1882
import seaborn as sns
sns.countplot(x ='GarageCond', data = df1)
1995-1882#113 missing values

47+17+8+3
75/1882 #0.039 percent so we can skip
"""Its more towords TA Category, so we can skip"""

#--------------------------------------------------------------------------------------------------------------------
#PavedDrive
df1.info()
df1.PavedDrive.value_counts()# 1835 and other 2 categories(116, 44)
sum(df1.PavedDrive.value_counts())#1995
import seaborn as sns
sns.countplot(x ='PavedDrive', data = df1)
116+44
160/1992 #0.08 percent so we can skip
"""Its more less than 10% towords Y Category, so we can skip"""

#--------------------------------------------------------------------------------------------------------------------
#PoolQC
df1.info()
df1.PoolQC.value_counts()# 4,2,1
sum(df1.PoolQC.value_counts())#7
import seaborn as sns
sns.countplot(x ='PoolQC', data = df1)

"""need to ignore coz only 7 data points are their and other all data is missing"""
#--------------------------------------------------------------------------------------------------------------------
#Fence
df1.info()
df1.Fence.value_counts()# 231, 87, 68, 14
sum(df1.Fence.value_counts())#400
import seaborn as sns
sns.countplot(x ='Fence', data = df1)

"""need to ignore coz only 400 data points are their and other all data is missing"""

#--------------------------------------------------------------------------------------------------------------------
#MiscFeature
df1.info()
df1.MiscFeature.value_counts()# 74,3,2,1
sum(df1.MiscFeature.value_counts())#80
import seaborn as sns
sns.countplot(x ='MiscFeature', data = df1)

"""need to ignore coz only 80 data points are their and other all data is missing"""

#--------------------------------------------------------------------------------------------------------------------
#SaleType
df1.info()
df1.SaleType.value_counts()# 1761 and other 8 categories
sum(df1.SaleType.value_counts())#80
import seaborn as sns
sns.countplot(x ='SaleType', data = df1)

130+68+14+5+5+5+4+3
234/1995 #11%

"""Its more less than 11% towords WD Category, so we can skip"""
#--------------------------------------------------------------------------------------------------------------------
#SaleCondition
df1.info()
df1.SaleCondition.value_counts()# 1761 and other 8 categories
sum(df1.SaleCondition.value_counts())#80
import seaborn as sns
sns.countplot(x ='SaleCondition', data = df1)

154+134+24+18+5
335/1995

"""Its more less than 17% towords WD Category, so we can skip"""

#------------------------------****Int vars****--------------------------------------------------------------------------------------
#OverallQual
df1.info()
df1.OverallQual.value_counts() #
sum(df1.OverallQual.value_counts())
import seaborn as sns
sns.countplot(x ='OverallQual', data = df1) 
#_run in block
plt.hist(df1.OverallQual, bins = 'auto', facecolor = 'blue')
plt.xlabel('OverallQual')
plt.ylabel('counts')
plt.title('Histogram of OverallQual')

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
df['OverallQual'].plot.box(color=props2, patch_artist = True, vert = False)
#no outliers
"""right side skewed, it may be a good predictor """

#-----------------------------------------------------------------------------------------------------------------
#OverallCond
df1.info()
df1.OverallCond.value_counts() #
sum(df1.OverallCond.value_counts())#1995
import seaborn as sns
sns.countplot(x ='OverallCond', data = df1)
#_run in block
plt.hist(df1.OverallCond, bins = 'auto', facecolor = 'blue')
plt.xlabel('OverallCond')
plt.ylabel('counts')
plt.title('Histogram of OverallCond')

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
df['OverallCond'].plot.box(color=props2, patch_artist = True, vert = False)
#no outliers
"""right side skewed, it may be a good predictor """

#-------------------------------------------------------------------------------------------
#YearBuilt
df1.info()
#______histogram
#_run in block
plt.hist(df1.YearBuilt, bins = 'auto', facecolor = 'green')
plt.xlabel('YearBuilt')
plt.ylabel('counts')
plt.title('Histogram of YearBuilt')
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
df['YearBuilt'].plot.box(color=props2, patch_artist = True, vert = False)
#2 outliers
#________outliers
Q1 = np.percentile(df1.YearBuilt, 25, interpolation = 'midpoint')  
Q2 = np.percentile(df1.YearBuilt, 50, interpolation = 'midpoint')  
Q3 = np.percentile(df1.YearBuilt, 75, interpolation = 'midpoint')   
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)   
IQR = Q3 - Q1  
print('Interquartile range is', IQR) # 49.0

low_lim = Q1 - 1.5 * IQR #1876.5
up_lim = Q3 + 1.5 * IQR # 2072
print('low_limit is', low_lim) 
print('up_limit is', up_lim) #2070.75

#_______________assigning UL to outliers 
df1.YearBuilt[df1.YearBuilt > 2070.75] = 2070.75
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
df['YearBuilt'].plot.box(color=props2, patch_artist = True, vert = False)

####not replaced outliers

"""left side skewed, it may be a good predictor """

#-------------------------------------------------------------------------------------------
#YearRemodAdd
df1.info()
df1.YearRemodAdd.value_counts()
sum(df1.YearRemodAdd.value_counts())#1995
 
# count plot on single numeric variable
sns.countplot(x ='YearRemodAdd', data = df1)

#______histogram
#_run in block
plt.hist(df1.YearRemodAdd, bins = 'auto', facecolor = 'green')
plt.xlabel('YearRemodAdd')
plt.ylabel('counts')
plt.title('Histogram of YearRemodAdd')

"""Not a normalize graph, need to normalize"""
#-------------------------------------------------------------------------------------------
#BsmtFinSF1
df1.info()
df1.BsmtFinSF1.value_counts() 
sum(df1.BsmtFinSF1.value_counts())#1995

#______histogram
#_run in block
plt.hist(df1.BsmtFinSF1, bins = 'auto', facecolor = 'green')
plt.xlabel('BsmtFinSF1')
plt.ylabel('counts')
plt.title('Histogram of BsmtFinSF1')



#-------------------------------------------------------------------------------------------
#BsmtFinSF2
df1.BsmtFinSF2.value_counts() 
sum(df1.BsmtFinSF2.value_counts())#1995

#______histogram
#_run in block
plt.hist(df1.BsmtFinSF2, bins = 'auto', facecolor = 'green')
plt.xlabel('BsmtFinSF2')
plt.ylabel('counts')
plt.title('Histogram of BsmtFinSF2')

"""0-1754 more counts of 0s"""
#-------------------------------------------------------------------------------------------
#BsmtUnfSF
df1.info()
df1.BsmtUnfSF.value_counts() 
sum(df1.BsmtUnfSF.value_counts())#1995

#______histogram
#_run in block
plt.hist(df1.BsmtUnfSF, bins = 'auto', facecolor = 'green')
plt.xlabel('BsmtUnfSF')
plt.ylabel('counts')
plt.title('Histogram of BsmtUnfSF')

"""" Right skewed histogram"""

#-------------------------------------------------------------------------------------------
#TotalBsmtSF
df1.info()
df1.TotalBsmtSF.value_counts() 
sum(df1.TotalBsmtSF.value_counts())#1995

#______histogram
#_run in block
plt.hist(df1.TotalBsmtSF, bins = 'auto', facecolor = 'green')
plt.xlabel('TotalBsmtSF')
plt.ylabel('counts')
plt.title('Histogram of TotalBsmtSF')

"""Its a symmetrical histogram"""

#-------------------------------------------------------------------------------------------
#
#Histogram
plt.hist(df1['1stFlrSF'], facecolor='green')
plt.xlabel('1stFlrSF')
plt.ylabel('counts')
plt.title('Histogram of 1stFlrSF')

#Box Plots
props2 = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
df1['1stFlrSF'].plot.box(color = props2 , patch_artist = True, vert = False) 

df1['1stFlrSF'].isnull().sum() #No missing values

Q1 = np.percentile(df1['1stFlrSF'], 25, interpolation = 'midpoint')
Q2 = np.percentile(df1['1stFlrSF'], 50, interpolation = 'midpoint')
Q3 = np.percentile(df1['1stFlrSF'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1 #466.0
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)  
print('Inter Quartile Range is', IQR)

low_lim = Q1 - IQR*1.5 #185.0
up_lim = Q3 + IQR*1.5#2049.0
print('low-limit is', low_lim)
print('Upper-limit is', up_lim)

df1['1stFlrSF'][df1['1stFlrSF'] >= up_lim] = up_lim #outliers replaced with upper 
#Box Plots
props2 = dict(boxes = 'red', whiskers ='green', medians = 'black', caps = 'blue')
df1['1stFlrSF'].plot.box(color = props2 , patch_artist = True, vert = False)

#-------------------------------------------------------------------------------------------
#2ndFlrSF
#Histogram
plt.hist(df1['2ndFlrSF'], facecolor='green')
plt.xlabel('2ndFlrSF')
plt.ylabel('counts')
plt.title('Histogram of 2ndFlrSF')

#Box Plots
props2 = dict(boxes = 'green', whiskers ='green', medians = 'black', caps = 'blue')
df1['2ndFlrSF'].plot.box(color = props2 , patch_artist = True, vert = False) 

df1['2ndFlrSF'].isnull().sum() #No missing values
#no outliers

#-------------------------------------------------------------------------------------------
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




