# God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

#df5 = pd.read_csv('df5.csv') # after bedroomAbgr, written as df5

df4 = pd.read_csv('D:/data _science/PYTHON/Linear_Regression_Python/df3.csv')
df4.info() 
#______int1 Dwell_Type, continuous
'''
looks in order, let's treat as continuous
'''
df4.Dwell_Type.describe() #No Missing!
'''
count    1995.000000
mean       61.242105
std       162.917385
min        20.000000
25%        20.000000
50%        50.000000
75%        70.000000
max      7080.000000 HOW 7080??????look at data description,
 --> should be max 190
Name: Dwell_Type, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df4.Dwell_Type, bins = 'auto', facecolor = 'red')
plt.xlabel('Dwell_Type')
plt.ylabel('counts')
plt.title('Histogram of Dwell_Type') # looks bad!
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df4['Dwell_Type'].plot.box(color=props2, patch_artist = True, vert = False)

'''
# looks serious outliers AND 
THERE SHOULD NOT BE MORE THAN 190!
Let's remove all above 190!
'''
# count
len(df4.Dwell_Type[df4.Dwell_Type > 190]) # only 2
"""Think of replacing by 70 and 30 , but it is only 2 datapoints so it will not affect our model"""
# let's put them at 190
#_______________assigning 190 to higher values
df4.Dwell_Type[df4.Dwell_Type > 190] = 190
len(df4.Dwell_Type[df4.Dwell_Type > 190]) # now 0

#________hist after treating higher values
plt.hist(df4.Dwell_Type, bins = 'auto', facecolor = 'blue')
plt.xlabel('Dwell_Type')
plt.ylabel('counts')
plt.title('Histogram of Dwell_Type') # looks good now!

#____boxplot after treating higher values
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'red')
df4['Dwell_Type'].plot.box(color=props2, patch_artist = True, vert = False)

df4.Dwell_Type.describe()
'''
still few outliers are visible, but we will tolerate these!!
'''
# good or bad for our predictive modeling?

np.corrcoef(df4.Property_Sale_Price, df4.Dwell_Type)#-0.04906216 Very bad we can not take this var in our model

#________2nd int LotFrontage #5; continuous
df4.info() 
df4.LotFrontage.describe() # missing present
1995-1678 #317 missing
'''
df4.LotFrontage.describe() 
Out[3]: 
count    1678.000000
mean       71.352205
std        27.935470
min        21.000000
25%        60.000000
50%        70.000000
75%        81.000000
max       313.000000
Name: LotFrontage, dtype: float64
'''

#______histogram
#_run in block
plt.hist(df4.LotFrontage, bins = 'auto', facecolor = 'blue')
plt.xlabel('LotFrontage')
plt.ylabel('counts')
plt.title('Histogram of LotFrontage')

#____boxplot
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'red')
df4['LotFrontage'].plot.box(color=props2, patch_artist = True, vert = False)

'''
first, fix outliers 
'''
# outliers counts on higher side
iqr_5 = 81-60 # Q3-Q2 , 75% and 25% Quartile UP LM
ul_5 = 81 + 1.5*iqr_5
ul_5 #112.5
len(df4.LotFrontage[df4.LotFrontage > ul_5]) # 61
'''
put OutLiers on higher threshold
'''
df4.LotFrontage[df4.LotFrontage > ul_5] = ul_5
len(df4.LotFrontage[df4.LotFrontage > ul_5]) # now 0, smile!
# outliers counts on lower side
iqr_5 = 81-60# Q3-Q2 , 75% and 25% Quartile LW LM
ll_5 = 60 -1.5*iqr_5
ll_5 #28.5
len(df4.LotFrontage[df4.LotFrontage < ll_5]) # 52
'''
put lower side OLiers on lower threshold
'''
df4.LotFrontage[df4.LotFrontage < ll_5] = ll_5
len(df4.LotFrontage[df4.LotFrontage < ll_5]) # now 0, smile!

#____boxplot, post treating Oliers
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df4['LotFrontage'].plot.box(color=props2, patch_artist = True, vert = False)
#__aha!!

#______histogram, post treating outliers
#_run in block
plt.hist(df4.LotFrontage, bins = 'auto', facecolor = 'green')
plt.xlabel('LotFrontage')
plt.ylabel('counts')
plt.title('Histogram of LotFrontage')

df4.LotFrontage.describe()
'''
count    1678.000000
mean       69.941299
std        19.922322
min        28.500000
25%        60.000000
50%        70.000000
75%        81.000000
max       112.500000
Name: LotFrontage, dtype: float64
'''
#___________replace missing with mean (or any number)
df4['LotFrontage'] = df4['LotFrontage'].fillna(df4['LotFrontage'].mean())
df4.LotFrontage.describe() # count = 1995; aha!

# good or bad for our predictive modeling?
np.corrcoef(df4.Property_Sale_Price, df4.LotFrontage) # 0.288, good! 

df4.info()
#________3rd int64, LotArea, #6
df4.LotArea.describe() # Nomissing 
'''
count      1995.000000
mean      10468.114286
std        7996.700030
min        1300.000000
25%        7512.500000
50%        9366.000000
75%       11424.000000
max      164660.000000
Name: LotArea, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df4.LotArea, bins = 'auto', facecolor = 'blue')
plt.xlabel('LotArea')
plt.ylabel('counts')
plt.title('Histogram of LotArea')

#____boxplot, oops! soooo many Olrs
props2 = dict(boxes = 'blue', whiskers = 'green', medians = 'black', caps = 'red')
df4['LotArea'].plot.box(color=props2, patch_artist = True, vert = False)

# outliers counts on higher side
iqr_6 = 11424 - 7512 
ul_6 = 11424 + 1.5*iqr_6
ul_6 #17292
len(df4.LotArea[df4.LotArea > ul_6]) # 121
'''
put OLiers on higher threshold
'''
df4.LotArea[df4.LotArea > ul_6] = ul_6
len(df4.LotArea[df4.LotArea > ul_6]) # now 0, smile!
# outliers counts on lower side
ll_6 = 7512 -1.5*iqr_6
ll_6 #1644
len(df4.LotArea[df4.LotArea < ll_6]) # 9
'''
put lower side OLiers on lower threshold
'''
df4.LotArea[df4.LotArea < ll_6] = ll_6
len(df4.LotArea[df4.LotArea < ll_6]) # now 0, smile!

#____boxplot, post treating Oliers
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df4['LotArea'].plot.box(color=props2, patch_artist = True, vert = False)
#__aha!!

# good or bad for our predictive modeling?
np.corrcoef(df4.Property_Sale_Price, df4.LotArea) # 0.32, good! 

df4.info()
#________4th int64, OverallQual, #19
df4.OverallQual.describe() # No missing, good news! we need to see outliers and corr only
'''
count    1995.000000
mean        6.153383
std         1.464571
min         1.000000
25%         5.000000
50%         6.000000
75%         7.000000
max        10.000000
Name: OverallQual, dtype: float64
''' 

#______histogram
#_run in block
plt.hist(df4.OverallQual, bins = 'auto', facecolor = 'red')
plt.xlabel('OverallQual')
plt.ylabel('counts')
plt.title('Histogram of OverallQual')

#____boxplot, only one/few olrs
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df4['OverallQual'].plot.box(color=props2, patch_artist = True, vert = False)

'''
we will tolerate these few outliers !
'''
# good or bad for our predictive modeling?
np.corrcoef(df4.Property_Sale_Price, df4.OverallQual) # 0.49, v good! 

df4.info()
#________5th int64, OverallQual, #20
df4.OverallCond.describe() # No missing, good news! we need to see outliers and corr only
'''
count    1995.000000
mean        5.845614
std         1.363011
min         1.000000
25%         5.000000
50%         5.000000
75%         7.000000
max         9.000000
Name: OverallCond, dtype: float64

PRIMA FACIE IT LOOKS LIKE THAT OverallCond will have v good corr
with OverallQual, but we found reverse!
OverallCond with RV was horribly poor!
let's drop this!
'''
# good or bad for our predictive modeling?
np.corrcoef(df4.OverallCond, df4.OverallQual) # 0.0106, v bad!
np.corrcoef(df4.Property_Sale_Price, df4.OverallCond) # -0.0459, v bad!

df4.info()
#________6th int64, YearBuilt, #21
df4.YearBuilt.describe()
'''
count    1995.000000
mean     1968.509774
std        29.988986
min      1872.000000
25%      1949.500000
50%      1970.000000
75%      1998.000000
max      2010.000000
Name: YearBuilt, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df4.YearBuilt, bins = 'auto', facecolor = 'purple')
plt.xlabel('YearBuilt')
plt.ylabel('counts')
plt.title('Histogram of YearBuilt')

#____boxplot, only one/few olrs
props2 = dict(boxes = 'purple', whiskers = 'green', medians = 'black', caps = 'red')
df4['YearBuilt'].plot.box(color=props2, patch_artist = True, vert = False)
'''
only few olrs on lower side, we will tolerate!
'''
# good or bad for our predictive modeling?

np.corrcoef(df4.Property_Sale_Price, df4.YearBuilt) # 0.47, good!

df4.info()
#________7th int64, YearRemodAdd, #22
df4.YearRemodAdd.describe()
'''
count    1995.000000
mean     1985.969424
std        20.504288
min      1950.000000
25%      1969.000000
50%      1994.000000
75%      2004.000000
max      2023.000000
Name: YearRemodAdd, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df4.YearRemodAdd, bins = 'auto', facecolor = 'pink')
plt.xlabel('YearRemodAdd')
plt.ylabel('counts')
plt.title('Histogram of YearRemodAdd')

#____boxplot, only one/few olrs
props2 = dict(boxes = 'pink', whiskers = 'green', medians = 'black', caps = 'red')
df4['YearRemodAdd'].plot.box(color=props2, patch_artist = True, vert = False)

#______@@@@@@@@@@@@  No Missing, No Outliers! 

# good or bad for our predictive modeling?
np.corrcoef(df4.Property_Sale_Price, df4.YearRemodAdd) # 0.51, good!

df4.info()
#________8th int64, MasVnrArea, #28
df4.MasVnrArea.describe()
'''
count    1982.000000
mean       89.879415
std       160.755925
min         0.000000
25%         0.000000
50%         0.000000
75%       141.500000
max      1600.000000
Name: MasVnrArea, dtype: float64

VERY STRANGE, 0s at min, 25%, 50%!
WE HAD DISCUSSED THIS DURING MasVnrType THAT WE WILL IGNORE MasVnrArea
'''
df4.MasVnrArea.value_counts() # 1215 are zeros!

df4.info()
#________9th int64, BsmtFinSF1, #36
df4.BsmtFinSF1.describe() # no missing
'''
count    1995.000000
mean      413.120301
std       419.346030
min         0.000000
25%         0.000000
50%       368.000000
75%       685.500000
max      5644.000000
Name: BsmtFinSF1, dtype: float64
'''

#______histogram
#_run in block
plt.hist(df4.BsmtFinSF1, bins = 'auto', facecolor = 'red')
plt.xlabel('BsmtFinSF1')
plt.ylabel('counts')
plt.title('Histogram of BsmtFinSF1')

#____boxplot, only one/few olrs
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df4['BsmtFinSF1'].plot.box(color=props2, patch_artist = True, vert = False)
#____count olrs
iqr_9 = 685-0
ul_9 = 685 + 1.5*iqr_9

len(df4.BsmtFinSF1[df4.BsmtFinSF1 > ul_9]) # 3
'''
put OLiers on higher threshold
'''
df4.BsmtFinSF1[df4.BsmtFinSF1 > ul_9] = ul_9
len(df4.BsmtFinSF1[df4.BsmtFinSF1 > ul_9]) # now 0, smile!

#____boxplot, post olrs treatment
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df4['BsmtFinSF1'].plot.box(color=props2, patch_artist = True, vert = False)
# aha!

# good or bad for our predictive modeling?
np.corrcoef(df4.Property_Sale_Price, df4.BsmtFinSF1) # 0.28, good!

df4.info()
#________10th int64, BsmtFinSF2, #38
df4.BsmtFinSF2.describe() # no missing, BUT, data is horrible!
'''
count    1995.000000
mean       50.372431
std       167.284792
min         0.000000
25%         0.000000
50%         0.000000
75%         0.000000
max      1474.000000
Name: BsmtFinSF2, dtype: float64

'''
#______histogram
#_run in block
plt.hist(df4.BsmtFinSF2, bins = 'auto', facecolor = 'red')
plt.xlabel('BsmtFinSF2')
plt.ylabel('counts')
plt.title('Histogram of BsmtFinSF2')

#____boxplot, looks HORRIBLE!
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df4['BsmtFinSF2'].plot.box(color=props2, patch_artist = True, vert = False)

'''
we will ignore this [BsmtFinSF2] and review BsmtFinType2 [catg] in detail 
'''
df4.info()
#________11th float, BsmtUnfSF, #39
df4.BsmtUnfSF.describe() # no missing
'''
count    1995.000000
mean      559.307268
std       438.577387
min         0.000000
25%       205.500000
50%       467.000000
75%       802.000000
max      2042.000000
Name: BsmtUnfSF, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df4.BsmtUnfSF, bins = 'auto', facecolor = 'red')
plt.xlabel('BsmtUnfSF')
plt.ylabel('counts')
plt.title('Histogram of BsmtUnfSF')

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df4['BsmtUnfSF'].plot.box(color=props2, patch_artist = True, vert = False)

#____count olrs
iqr_11 = 802 - 205
ul_11 = 802 + 1.5*iqr_11
ul_11 #1697.5
len(df4.BsmtUnfSF[df4.BsmtUnfSF > ul_11]) # 27
'''
put OLiers on higher threshold
'''
df4.BsmtUnfSF[df4.BsmtUnfSF > ul_11] = ul_11
len(df4.BsmtUnfSF[df4.BsmtUnfSF > ul_11]) # now 0, smile!

#____boxplot, post olrs treatment
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df4['BsmtUnfSF'].plot.box(color=props2, patch_artist = True, vert = False)

# good or bad for our predictive modeling?
np.corrcoef(df4.Property_Sale_Price, df4.BsmtUnfSF) # 0.26, good!

df4.info()
#________12th int64, TotalBsmtSF, #40
df4.TotalBsmtSF.describe() # no missing
'''
count    1995.000000
mean     1022.800000
std       403.538164
min         0.000000
25%       788.000000
50%       971.000000
75%      1254.500000
max      6110.000000
Name: TotalBsmtSF, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df4.TotalBsmtSF, bins = 'auto', facecolor = 'red')
plt.xlabel('TotalBsmtSF')
plt.ylabel('counts')
plt.title('Histogram of TotalBsmtSF')

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df4['TotalBsmtSF'].plot.box(color=props2, patch_artist = True, vert = False)

#____count olrs
iqr_12 = 1254 - 788
ul_12 = 1254 + 1.5*iqr_12
ul_12 #1953
len(df4.TotalBsmtSF[df4.TotalBsmtSF > ul_12]) # 16
'''
put OLiers on higher threshold
'''
df4.TotalBsmtSF[df4.TotalBsmtSF > ul_12] = ul_12
len(df4.TotalBsmtSF[df4.TotalBsmtSF > ul_12]) # now 0, smile!

#____boxplot, post olrs treatment
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df4['TotalBsmtSF'].plot.box(color=props2, patch_artist = True, vert = False)

# good or bad for our predictive modeling?
np.corrcoef(df4.Property_Sale_Price, df4.TotalBsmtSF) # 0.60, excellent! 

#__________let me write work done so far as df4 and import as df5
#___________now I will work in df5
#____df5 is having categorical and 12 int, FIXED, wow!
#df4.to_csv('df4.csv')

df5 = pd.read_csv('D:/data _science/PYTHON/Linear_Regression_Python/df4.csv')
df5.info() 

df5.info()
#________13th int64, 1stFlrSF, #46
df5['1stFlrSF'].describe() # no missing
'''
count    1995.000000
mean     1139.678697
std       351.879552
min       334.000000
25%       884.000000
50%      1078.000000
75%      1350.000000
max      4692.000000
Name: 1stFlrSF, dtype: float64
'''

#______histogram
#_run in block
plt.hist(df5['1stFlrSF'], bins = 'auto', facecolor = 'red')
plt.xlabel('1stFlrSF')
plt.ylabel('counts')
plt.title('Histogram of 1stFlrSF')

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df5['1stFlrSF'].plot.box(color=props2, patch_artist = True, vert = False)
#____count olrs
iqr_13 = 1350-884
ul_13 = 1350 + 1.5*iqr_13
ul_13 # 2049
len(df5['1stFlrSF'][df5['1stFlrSF'] > ul_13]) # 26
'''
put OLiers on higher threshold
'''
df5['1stFlrSF'][df5['1stFlrSF'] > ul_13] = ul_13 
len(df5['1stFlrSF'][df5['1stFlrSF'] > ul_13])  # now 0, smile!

#____boxplot
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df5['1stFlrSF'].plot.box(color=props2, patch_artist = True, vert = False)
#__aha!

# good or bad for our predictive modeling?
np.corrcoef(df5.Property_Sale_Price, df5['1stFlrSF']) # 0.55, good! 

df5.info()
#________14th int64, 2ndFlrSF, #47
df5['2ndFlrSF'].describe() # no missing
'''
count    1995.000000
mean      335.029574
std       425.940844
min         0.000000
25%         0.000000
50%         0.000000
75%       728.000000
max      1818.000000
Name: 2ndFlrSF, dtype: float64 min, 25%, 50% = 0 !!!!!!!!!!
'''
df5['2ndFlrSF'].value_counts() # 1156 are zeros!
1156/1995 # 58% are zeros !
# good idea to make 2 catgs, flr2sf_nil, flr2sf, lets see hist and bxplt

#______histogram
#_run in block
plt.hist(df5['2ndFlrSF'], bins = 'auto', facecolor = 'purple')
plt.xlabel('2ndFlrSF')
plt.ylabel('counts')
plt.title('Histogram of 2ndFlrSF')

#____boxplot
props2 = dict(boxes = 'purple', whiskers = 'green', medians = 'black', caps = 'red')
df5['2ndFlrSF'].plot.box(color=props2, patch_artist = True, vert = False)

# better convert into 2 catgs flr2sf_nil, flr2sf
np.corrcoef(df5.Property_Sale_Price, df5['2ndFlrSF']) # 0.33, good! 

df5.info()
#________15th int64, LowQualFinSF, #48
df5.LowQualFinSF.describe() # no missing
'''
count    1995.000000
mean        6.122807
std        48.883702
min         0.000000
25%         0.000000
50%         0.000000
75%         0.000000
max       528.000000   Opps! even 75% is also zero!
'''
df5.LowQualFinSF.value_counts() # 1957 are zeros!
1957/1995 # 98% are zeros! IGNORE

df5.info()
#________16th int64, GrLivArea, #49
df5.GrLivArea.describe() # no missing
'''
count    1995.000000
mean     1480.831078
std       479.146953
min       334.000000
25%      1120.000000
50%      1445.000000
75%      1750.000000
max      5642.000000
Name: GrLivArea, dtype: float64
'''
#______histogram
#_run in block
plt.hist(df5.GrLivArea, bins = 'auto', facecolor = 'red')
plt.xlabel('GrLivArea')
plt.ylabel('counts')
plt.title('Histogram of GrLivArea')

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df5['GrLivArea'].plot.box(color=props2, patch_artist = True, vert = False)

#____count olrs
iqr_15 = 1750-1120
ul_15 = 1750 + 1.5*iqr_15
ul_15 # 2695
len(df5['GrLivArea'][df5['GrLivArea'] > ul_15]) # 26
'''
put OLiers on higher threshold
'''
df5['GrLivArea'][df5['GrLivArea'] > ul_15] = ul_15 
len(df5['GrLivArea'][df5['GrLivArea'] > ul_15])  # now 0, smile!
#____boxplot
props2 = dict(boxes = 'green', whiskers = 'green', medians = 'black', caps = 'red')
df5['GrLivArea'].plot.box(color=props2, patch_artist = True, vert = False)
#_____aha!

# good or bad for our predictive modeling?
np.corrcoef(df5.Property_Sale_Price, df5.GrLivArea) # 0.70, excellent! 

df5.info()
#________17th int64, BsmtFullBath, #50
df5.BsmtFullBath.describe() # no missing
'''
count    1995.000000
mean        0.414536
std         0.516614
min         0.000000
25%         0.000000
50%         0.000000
75%         1.000000
max         3.000000
Name: BsmtFullBath, dtype: float64 LOOKS categorical
'''
df5.BsmtFullBath.value_counts() #1190, 785, 18, 2
sum(df5.BsmtFullBath.value_counts()) # 1995

# club catg 1,2,3 into 1
'''
let's make in two groups, 0, 1
'''
df5['BsmtFullBath']=df5.get('BsmtFullBath').replace(2,1)
df5['BsmtFullBath']=df5.get('BsmtFullBath').replace(3,1)

df5.BsmtFullBath.value_counts() #1190, 805
sum(df5.BsmtFullBath.value_counts()) # 1995

#______is this a good predictor?
# Indpndnt sample t test
bfb_0 = df5[df5.BsmtFullBath == 0]
bfb_1 = df5[df5.BsmtFullBath == 1]
import scipy
scipy.stats.ttest_ind(bfb_0.Property_Sale_Price, bfb_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

df5.info()
#________18th int64, BsmtHalfBath, #51
df5.BsmtHalfBath.describe() # no missing
'''
count    1995.000000
mean        0.057644
std         0.239494
min         0.000000
25%         0.000000
50%         0.000000
75%         0.000000
max         2.000000
Name: BsmtHalfBath, dtype: float64   looks catg!
'''
df5.BsmtHalfBath.value_counts() #1883, 109, 3
sum(df5.BsmtHalfBath.value_counts()) # 1995
1883/1995 # 94%
'''
94% one catg....IGNORE!
'''
df5.info()
#________19th int64, FullBath, #52
df5.FullBath.describe() # no missing
'''
count    1995.000000
mean        1.551880
std         0.530599
min         0.000000
25%         1.000000
50%         2.000000
75%         2.000000
max         3.000000
Name: FullBath, dtype: float64
'''
df5.FullBath.value_counts() #1063, 898, 24,10
sum(df5.FullBath.value_counts()) # 1995
# Club, 0 with 2, 3 with 1
'''
let's make in two groups, 1 and 2
'''
df5['FullBath']=df5.get('FullBath').replace(0,2)
df5['FullBath']=df5.get('FullBath').replace(3,1)

df5.FullBath.value_counts() #1073, 922
sum(df5.FullBath.value_counts()) # 1995
#______is this a good predictor?
# Indpndnt sample t test
fb_1 = df5[df5.FullBath == 1]
fb_2 = df5[df5.FullBath == 2]
import scipy
scipy.stats.ttest_ind(fb_1.Property_Sale_Price, fb_2.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

df5.info()
#________20th int64, HalfBath, #53
df5.HalfBath.describe() # no missing

df5.HalfBath.value_counts() #1281, 699, 15
sum(df5.HalfBath.value_counts()) # 1995
'''
let's make 2 catgs only
club 2 with 0 
'''
df5['HalfBath']=df5.get('HalfBath').replace(2,0)
df5.HalfBath.value_counts() #1296, 699
sum(df5.HalfBath.value_counts()) # 1995

#______is this a good predictor?
# Indpndnt sample t test
hb_0 = df5[df5.HalfBath == 0]
hb_1 = df5[df5.HalfBath == 1]
import scipy
scipy.stats.ttest_ind(hb_0.Property_Sale_Price, hb_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

df5.info()
#________21st int64, BedroomAbvGr, #54
df5.BedroomAbvGr.describe() # no missing
'''
count    1995.000000
mean        2.872682
std         0.801722
min         0.000000
25%         2.000000
50%         3.000000
75%         3.000000
max         8.000000
Name: BedroomAbvGr, dtype: float64
'''
df5.BedroomAbvGr.value_counts() #1108 and more
sum(df5.HalfBath.value_counts()) # 1995
'''
3    1108
2     486
4     289
1      67
5      30
6       9
0       5
8       1
Name: BedroomAbvGr, dtype: int64
'''
# let's club and rename
# 3, rename as 0, rest as 1 

df5['BedroomAbvGr']=df5.get('BedroomAbvGr').replace(3,0)
df5['BedroomAbvGr']=df5.get('BedroomAbvGr').replace(2,1)
df5['BedroomAbvGr']=df5.get('BedroomAbvGr').replace(4,1)
# 1 as 1 already there!
df5['BedroomAbvGr']=df5.get('BedroomAbvGr').replace(5,1)
df5['BedroomAbvGr']=df5.get('BedroomAbvGr').replace(6,1)
# 0 is already there!
df5['BedroomAbvGr']=df5.get('BedroomAbvGr').replace(8,1)

df5.BedroomAbvGr.value_counts() #1113 and 882
sum(df5.BedroomAbvGr.value_counts()) # 1995

#______is this a good predictor?
# Indpndnt sample t test
bragr_0 = df5[df5.BedroomAbvGr == 0]
bragr_1 = df5[df5.BedroomAbvGr == 1]
import scipy
scipy.stats.ttest_ind(bragr_0.Property_Sale_Price, bragr_1.Property_Sale_Price)
# p_value is <0.05; Ho Reject; Good Predictor

df5.info()
#________22nd int64, KitchenAbvGr, #55
df5.KitchenAbvGr.describe() # no missing, catg
df5.KitchenAbvGr.value_counts() #1887 in one catg
sum(df5.KitchenAbvGr.value_counts()) # 1995
1887/1995 # 95%
'''
95% in one catg, IGNORE!
'''
#__________let me write work done so far as df5 and import as df6
#___________now I will work in df6
#____df5 is having categorical and 21 int, FIXED, wow!
df5.to_csv('D:/data _science/PYTHON/Linear_Regression_Python/df5.csv')




