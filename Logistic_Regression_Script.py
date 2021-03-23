# God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

#import file h1n1_vaccine_prediction
data1 = pd.read_csv('D:/data _science/PYTHON/LogisticRegression/h1n1_vaccine_prediction.csv')
data1.info() #float64(23), int64(2), object(9)
data1.shape #(26707, 34)
#-----------------------------------------------------------------------------------------
# 33 h1n1_vaccine Targate variable
#Var no 1 h1n1_worry
data1.h1n1_vaccine.value_counts()
'''
0    21033
1     5674
'''
sum(data1.h1n1_vaccine.value_counts())
26707-26707 #0 no missing values

#remove rows containing missing values after class
import seaborn as sns
sns.countplot(x ='h1n1_vaccine', data = data1)

(len(data1.h1n1_vaccine)-data1.h1n1_vaccine.describe()['count'])/len(data1.h1n1_vaccine)
#0.0 no missing value
''' this var  h1n1 vaccine, imbalanced data , only about 20 % ppl have vaccinated 
use SMOTE (Synthetic Minority Oversampling Technique)'''

#-----------------------------------------------------------------------------------------
#Var no 1 h1n1_worry
data1.h1n1_worry.value_counts()
'''
2.0    10575
1.0     8153
3.0     4591
0.0     3296'''
sum(data1.h1n1_worry.value_counts())
26707-26615 #92 missing values
(92/26707)*100#0.34% We will not worry about missing value coz its less % h1n1 worry!
#remove rows containing missing values after class
import seaborn as sns
sns.countplot(x ='h1n1_worry', data = data1)

(len(data1.h1n1_worry)-data1.h1n1_worry.describe()['count'])/len(data1.h1n1_worry)
#0.003444789755494814 missing value
#Good predictor
np.corrcoef(data1.h1n1_vaccine, data1.h1n1_worry)

#-----------------------------------------------------------------------------------------
#Var no 2 h1n1_worry
data1.info()
data1.h1n1_awareness.value_counts()
'''
1.0    14598
2.0     9487
0.0     2506
'''
sum(data1.h1n1_awareness.value_counts())
26707-26591 #116 missing values
(116/26707)*100#0.43% 
import seaborn as sns
sns.countplot(x ='h1n1_awareness', data = data1)

(len(data1.h1n1_awareness)-data1.h1n1_awareness.describe()['count'])/len(data1.h1n1_awareness)
#0.00434343056127607  missing value
#Good predictor

#-----------------------------------------------------------------------------------------
##Var no 3 antiviral_medication
data1.info()
data1.antiviral_medication.value_counts()
'''
0.0    25335
1.0     1301'''
sum(data1.antiviral_medication.value_counts())
26707-26636 #116 missing values
(71/26707)*100#0.26% 
import seaborn as sns
sns.countplot(x ='antiviral_medication', data = data1)

(len(data1.antiviral_medication)-data1.antiviral_medication.describe()['count'])/len(data1.antiviral_medication)
# 0.002658479050436215 missing value
#Good predictor
#-----------------------------------------------------------------------------------------
##Var no 4 contact_avoidance
data1.info()
data1.contact_avoidance.value_counts()
'''
1.0    19228
0.0     7271'''
sum(data1.contact_avoidance.value_counts())
26707-26499 #208 missing values
(208/26707)*100#0.77% 
import seaborn as sns
sns.countplot(x ='antiviral_medication', data = data1)

(len(data1.contact_avoidance)-data1.contact_avoidance.describe()['count'])/len(data1.contact_avoidance)
#  0.007788220316770884 missing value
#Good predictor

#-----------------------------------------------------------------------------------------
# 5 bought_face_mask
data1.info()
data1.bought_face_mask.value_counts()
'''
0.0    24847
1.0     1841'''
sum(data1.bought_face_mask.value_counts())
26707-26688 #19 missing values
(19/26707)*100#0.07% 
import seaborn as sns
sns.countplot(x ='bought_face_mask', data = data1)

(len(data1.bought_face_mask)-data1.bought_face_mask.describe()['count'])/len(data1.bought_face_mask)
# Variable 5   bought_face_mask  
# bought face mask less than 1% people 
print('Count--')
print( data1.bought_face_mask .value_counts())
print('Count_sum--')
print(sum(data1.bought_face_mask .value_counts()))
print('describe--')
print('describe--', data1.bought_face_mask .describe())
print('% of each cat')
print(data1['bought_face_mask'].value_counts(normalize=True))
'''0.0    0.931018
1.0    0.068982'''
#will add this , and check it will affect on result or not!
#Good predictor
#-----------------------------------------------------------------------------------------
# 5 bought_face_mask
data1.info()
data1.wash_hands_frequently.value_counts()
'''
1.0    22015
0.0     4650'''
sum(data1.wash_hands_frequently.value_counts())
26707-26665 #42 missing values
(42/26707)*100#0.15% 
import seaborn as sns
sns.countplot(x ='wash_hands_frequently', data = data1)

(len(data1.wash_hands_frequently)-data1.wash_hands_frequently.describe()['count'])/len(data1.wash_hands_frequently)
#0.00157
print('% of each cat')
print(data1['wash_hands_frequently'].value_counts(normalize=True))
'''1.0    0.825614
0.0    0.174386'''
#Good predictor

#-----------------------------------------------------------------------------------------
#avoid_large_gatherings
data1.info()
data1.avoid_large_gatherings.value_counts()
'''
0.0    17073
1.0     9547'''
sum(data1.avoid_large_gatherings.value_counts())
26707-26620 #87 missing values
(87/26707)*100#0.32% 
import seaborn as sns
sns.countplot(x ='avoid_large_gatherings', data = data1)

(len(data1.avoid_large_gatherings)-data1.avoid_large_gatherings.describe()['count'])/len(data1.avoid_large_gatherings)
#0.0032575729209570526
print('% of each cat')
print(data1['avoid_large_gatherings'].value_counts(normalize=True))
'''0.0    0.64136
1.0    0.35864'''

#-----------------------------------------------------------------------------------------
#8 avoid_large_gatherings
data1.info()
data1.reduced_outside_home_cont.value_counts()
'''
0.0    17644
1.0     8981'''
sum(data1.reduced_outside_home_cont.value_counts())
26707-26625 #82 missing values
(82/26707)*100#0.30% 
import seaborn as sns
sns.countplot(x ='avoid_large_gatherings', data = data1)

(len(data1.reduced_outside_home_cont)-data1.reduced_outside_home_cont.describe()['count'])/len(data1.reduced_outside_home_cont)
# 0.0030703560864192908
print('% of each cat')
print(data1['reduced_outside_home_cont'].value_counts(normalize=True))
'''0.0    0.662685
1.0    0.337315'''
#we can ignore?

#-----------------------------------------------------------------------------------------
#9 avoid_touch_face
data1.info()
data1.avoid_touch_face.value_counts()
'''
1.0    18001
0.0     8578'''
sum(data1.avoid_touch_face.value_counts())
26707-26579 #128 missing values
(128/26707)*100#0.47% 
import seaborn as sns
sns.countplot(x ='avoid_touch_face', data = data1)

(len(data1.avoid_touch_face)-data1.avoid_touch_face.describe()['count'])/len(data1.avoid_touch_face)
# 0.004792750964166698
print('% of each cat')
print(data1['avoid_touch_face'].value_counts(normalize=True))
'''1.0    0.677264
0.0    0.322736'''

#We can avoid! Bad pred

#-----------------------------------------------------------------------------------------
# 10  dr_recc_h1n1_vacc 
data1.info()
data1.dr_recc_h1n1_vacc.value_counts()
'''
0.0    19139
1.0     5408'''
sum(data1.dr_recc_h1n1_vacc.value_counts())
26707-24547 #2160 missing values
(2160/26707)*100#8.087% 
import seaborn as sns
sns.countplot(x ='dr_recc_h1n1_vacc', data = data1)

(len(data1.dr_recc_h1n1_vacc)-data1.dr_recc_h1n1_vacc.describe()['count'])/len(data1.dr_recc_h1n1_vacc)
# 0.08087767252031303
print('% of each cat')
print(data1['dr_recc_h1n1_vacc'].value_counts(normalize=True))
'''0.0    0.779688
1.0    0.220312'''

#We can avoid! Bad pred

#-----------------------------------------------------------------------------------------
#11  dr_recc_seasonal_vacc 
data1.info()
data1.dr_recc_seasonal_vacc.value_counts()
'''
0.0    16453
1.0     8094'''
sum(data1.dr_recc_seasonal_vacc.value_counts())
26707-24547 #2160 missing values
(2160/26707)*100#8.087% 
import seaborn as sns
sns.countplot(x ='dr_recc_seasonal_vacc', data = data1)

(len(data1.dr_recc_seasonal_vacc)-data1.dr_recc_seasonal_vacc.describe()['count'])/len(data1.dr_recc_seasonal_vacc)
# 0.08087767252031303
print('% of each cat')
print(data1['dr_recc_seasonal_vacc'].value_counts(normalize=True))
'''0.0    0.670265
1.0    0.329735'''

#-----------------------------------------------------------------------------------------
# 12 chronic_medic_condition
data1.info()
data1.chronic_medic_condition.value_counts()
'''
0.0    18446
1.0     7290'''
sum(data1.chronic_medic_condition.value_counts())
26707-25736 #971 missing values
(971/26707)*100# 3.63% 
import seaborn as sns
sns.countplot(x ='chronic_medic_condition', data = data1)

(len(data1.chronic_medic_condition)-data1.chronic_medic_condition.describe()['count'])/len(data1.chronic_medic_condition)
#  0.03635750926723331
print('% of each cat')
print(data1['chronic_medic_condition'].value_counts(normalize=True))
'''0.0    0.716739
1.0    0.283261'''

#-----------------------------------------------------------------------------------------
# 13  cont_child_undr_6_mnths
data1.info()
data1.cont_child_undr_6_mnths.value_counts()
'''
0.0    23749
1.0     2138'''
sum(data1.cont_child_undr_6_mnths.value_counts())
26707-25887 #820 missing values
(820/26707)*100# 3.07% 
import seaborn as sns
sns.countplot(x ='cont_child_undr_6_mnths', data = data1)

(len(data1.cont_child_undr_6_mnths)-data1.cont_child_undr_6_mnths.describe()['count'])/len(data1.cont_child_undr_6_mnths)
#  0.03070356086419291
print('% of each cat')
print(data1['cont_child_undr_6_mnths'].value_counts(normalize=True))
'''0.0    0.91741
1.0    0.08259'''
#-----------------------------------------------------------------------------------------
#14  is_health_worker
data1.info()
data1.is_health_worker.value_counts()
'''
0.0    23004
1.0     2899'''
sum(data1.is_health_worker.value_counts())
26707-25903 #804 missing values
(804/26707)*100# 3.01% 
import seaborn as sns
sns.countplot(x ='is_health_worker', data = data1)

(len(data1.is_health_worker)-data1.is_health_worker.describe()['count'])/len(data1.is_health_worker)
#  0.03010446699367207
print('% of each cat')
print(data1['is_health_worker'].value_counts(normalize=True))
'''0.0    0.888082
1.0    0.111918'''

#-----------------------------------------------------------------------------------------
#15  has_health_insur  
data1.info()
data1.has_health_insur.value_counts()
'''
1.0    12697
0.0     1736'''
sum(data1.has_health_insur.value_counts())
26707-14433 #12274 missing values
(12274/26707)*100# 45.95% 
import seaborn as sns
sns.countplot(x ='has_health_insur', data = data1)

(len(data1.has_health_insur)-data1.has_health_insur.describe()['count'])/len(data1.has_health_insur)
#   0.45957988542329725
print('% of each cat')
print(data1['has_health_insur'].value_counts(normalize=True))
'''1.0    0.87972
0.0    0.12028'''

#-----------------------------------------------------------------------------------------
#15  has_health_insur  
data1.info()
data1.has_health_insur.value_counts()
'''
1.0    12697
0.0     1736'''
sum(data1.has_health_insur.value_counts())
26707-14433 #12274 missing values
(12274/26707)*100# 45.95% 
import seaborn as sns
sns.countplot(x ='has_health_insur', data = data1)

(len(data1.has_health_insur)-data1.has_health_insur.describe()['count'])/len(data1.has_health_insur)
#   0.45957988542329725
print('% of each cat')
print(data1['has_health_insur'].value_counts(normalize=True))
'''1.0    0.87972
0.0    0.12028'''
#-----------------------------------------------------------------------------------------
#16  is_h1n1_vacc_effective  
data1.info()
data1.is_h1n1_vacc_effective.value_counts()
'''
4.0    11683
5.0     7166
3.0     4723
2.0     1858
1.0      886'''
sum(data1.is_h1n1_vacc_effective.value_counts())
26707-26316 #391 missing values
(391/26707)*100# 1.46% 
import seaborn as sns
sns.countplot(x ='is_h1n1_vacc_effective', data = data1)

(len(data1.is_h1n1_vacc_effective)-data1.is_h1n1_vacc_effective.describe()['count'])/len(data1.is_h1n1_vacc_effective)
#   0.01464035646085296
print('% of each cat')
print(data1['is_h1n1_vacc_effective'].value_counts(normalize=True))
'''4.0    0.443950
5.0    0.272306
3.0    0.179473
2.0    0.070603
1.0    0.033668'''
#-----------------------------------------------------------------------------------------
#17  is_h1n1_risky  
data1.info()
data1.is_h1n1_risky.value_counts()
'''
2.0    9919
1.0    8139
4.0    5394
5.0    1750
3.0    1117'''
sum(data1.is_h1n1_risky.value_counts())
26707-26319 #388 missing values
(388/26707)*100# 1.45% 
import seaborn as sns
sns.countplot(x ='is_h1n1_risky', data = data1)

(len(data1.is_h1n1_risky)-data1.is_h1n1_risky.describe()['count'])/len(data1.is_h1n1_risky)
#   0.014528026360130303
print('% of each cat')
print(data1['is_h1n1_risky'].value_counts(normalize=True))
'''2.0    0.376876
1.0    0.309244
4.0    0.204947
5.0    0.066492
3.0    0.042441'''
#-----------------------------------------------------------------------------------------
#18  sick_from_h1n1_vacc  
data1.info()
data1.sick_from_h1n1_vacc.value_counts()
'''
2.0    9919
1.0    8139
4.0    5394
5.0    1750
3.0    1117'''
sum(data1.sick_from_h1n1_vacc.value_counts())
26707-26312 #395 missing values
(395/26707)*100# 1.47% 
import seaborn as sns
sns.countplot(x ='sick_from_h1n1_vacc', data = data1)

(len(data1.sick_from_h1n1_vacc)-data1.sick_from_h1n1_vacc.describe()['count'])/len(data1.sick_from_h1n1_vacc)
#   0.01479012992848317
print('% of each cat')
print(data1['sick_from_h1n1_vacc'].value_counts(normalize=True))
'''2.0    0.346952
1.0    0.341973
4.0    0.222332
5.0    0.083118
3.0    0.005625'''
#-----------------------------------------------------------------------------------------
#19  is_seas_vacc_effective  
data1.info()
data1.is_seas_vacc_effective.value_counts()

sum(data1.is_seas_vacc_effective.value_counts())
26707-26245 #462 missing values
(462/26707)*100# 1.72% 
import seaborn as sns
sns.countplot(x ='is_seas_vacc_effective', data = data1)

(len(data1.is_seas_vacc_effective)-data1.is_seas_vacc_effective.describe()['count'])/len(data1.is_seas_vacc_effective)
#  0.017298835511289176
print('% of each cat')
print(data1['is_seas_vacc_effective'].value_counts(normalize=True))

#-----------------------------------------------------------------------------------------
#20  is_seas_risky  
data1.info()
data1.is_seas_risky.value_counts()

sum(data1.is_seas_risky.value_counts())
26707-26193 #514 missing values
(514/26707)*100# 1.92% 
import seaborn as sns
sns.countplot(x ='is_seas_risky', data = data1)

(len(data1.is_seas_risky)-data1.is_seas_risky.describe()['count'])/len(data1.is_seas_risky)
# 0.019245890590481898
print('% of each cat')
print(data1['is_seas_risky'].value_counts(normalize=True))
#-----------------------------------------------------------------------------------------
#21  sick_from_seas_vacc  
data1.info()
data1.sick_from_seas_vacc.value_counts()

sum(data1.sick_from_seas_vacc.value_counts())
26707-26170 #537 missing values
(537/26707)*100# 2.01% 
import seaborn as sns
sns.countplot(x ='sick_from_seas_vacc', data = data1)

(len(data1.sick_from_seas_vacc)-data1.sick_from_seas_vacc.describe()['count'])/len(data1.sick_from_seas_vacc)
# 0.0201070880293556
print('% of each cat')
print(data1['sick_from_seas_vacc'].value_counts(normalize=True))
#-----------------------------------------------------------------------------------------
#22  age_bracket  
data1.info()
data1.age_bracket.value_counts()

sum(data1.age_bracket.value_counts())#no missing value

import seaborn as sns
sns.countplot(x ='age_bracket', data = data1)

(len(data1.age_bracket)-data1.age_bracket.describe()['count'])/len(data1.age_bracket)
# 0
print('% of each cat')
print(data1['age_bracket'].value_counts(normalize=True))
#-----------------------------------------------------------------------------------------
#23  qualification  
data1.info()
data1.qualification.value_counts()

sum(data1.qualification.value_counts())#25300

import seaborn as sns
sns.countplot(x ='qualification', data = data1)

(len(data1.qualification)-data1.qualification.describe()['count'])/len(data1.qualification)
#0.05268281723892612
print('% of each cat')
print(data1['qualification'].value_counts(normalize=True))
#-----------------------------------------------------------------------------------------
#24 race  
data1.info()
data1.race.value_counts()

sum(data1.race.value_counts())#25300

import seaborn as sns
sns.countplot(x ='race', data = data1)

(len(data1.race)-data1.race.describe()['count'])/len(data1.race)
#0
print('% of each cat')
print(data1['race'].value_counts(normalize=True))

#-----------------------------------------------------------------------------------------
#25 sex  
data1.info()
data1.sex.value_counts()

sum(data1.sex.value_counts())

import seaborn as sns
sns.countplot(x ='sex', data = data1)

(len(data1.sex)-data1.sex.describe()['count'])/len(data1.sex)
#0 no missing vlues
print('% of each cat')
print(data1['sex'].value_counts(normalize=True))

#-----------------------------------------------------------------------------------------
#26 income_level  
data1.info()
data1.income_level.value_counts()

sum(data1.income_level.value_counts())#22284

import seaborn as sns
sns.countplot(x ='income_level', data = data1)

(len(data1.income_level)-data1.income_level.describe()['count'])/len(data1.income_level)
#0.16561201183210394
print('% of each cat')
print(data1['income_level'].value_counts(normalize=True))

#-----------------------------------------------------------------------------------------
#27 marital_status  
data1.info()
data1.marital_status.value_counts()

sum(data1.marital_status.value_counts())#25299

import seaborn as sns
sns.countplot(x ='marital_status', data = data1)

(len(data1.marital_status)-data1.marital_status.describe()['count'])/len(data1.marital_status)
# 0.05272026060583368
print('% of each cat')
print(data1['marital_status'].value_counts(normalize=True))
#-----------------------------------------------------------------------------------------
#28 housing_status 
data1.info()
data1.housing_status.value_counts()

sum(data1.housing_status.value_counts())#25299

import seaborn as sns
sns.countplot(x ='housing_status', data = data1)

(len(data1.housing_status)-data1.housing_status.describe()['count'])/len(data1.housing_status)
# 0.07645935522522185
print('% of each cat')
print(data1['housing_status'].value_counts(normalize=True))
'''Own     0.759619
Rent    0.240381'''

#-----------------------------------------------------------------------------------------
#29 employment 
data1.info()
data1.employment.value_counts()

sum(data1.employment.value_counts())#25299

import seaborn as sns
sns.countplot(x ='employment', data = data1)

(len(data1.employment)-data1.employment.describe()['count'])/len(data1.employment)
# 0.05477964578574906
print('% of each cat')
print(data1['employment'].value_counts(normalize=True))
'''Employed              0.537157
Not in Labor Force    0.405284
Unemployed            0.057558'''
#-----------------------------------------------------------------------------------------
#30 census_msa 
data1.info()
data1.census_msa.value_counts()

sum(data1.census_msa.value_counts())

import seaborn as sns
sns.countplot(x ='census_msa', data = data1)

(len(data1.census_msa)-data1.census_msa.describe()['count'])/len(data1.census_msa)
# 0 no missing value
print('% of each cat')
print(data1['census_msa'].value_counts(normalize=True))
'''MSA, Not Principle  City    0.436028
MSA, Principle City         0.294455
Non-MSA                     0.269517'''
#-----------------------------------------------------------------------------------------
#31 no_of_adults 
data1.info()
data1.no_of_adults.value_counts()

sum(data1.no_of_adults.value_counts())#26458 some NAs

import seaborn as sns
sns.countplot(x ='no_of_adults', data = data1)

(len(data1.no_of_adults)-data1.no_of_adults.describe()['count'])/len(data1.no_of_adults)
#0.00932339835998053
print('% of each cat')
print(data1['no_of_adults'].value_counts(normalize=True))
'''1.0    0.547056
0.0    0.304483
2.0    0.105941
3.0    0.042520'''
#-----------------------------------------------------------------------------------------
#32 no_of_children 
data1.info()
data1.no_of_children.value_counts()

sum(data1.no_of_children.value_counts())#26458 some NAs

import seaborn as sns
sns.countplot(x ='no_of_children', data = data1)

(len(data1.no_of_children)-data1.no_of_children.describe()['count'])/len(data1.no_of_children)
#0.00932339835998053
print('% of each cat')
print(data1['no_of_children'].value_counts(normalize=True))
'''0.0    0.705722
1.0    0.120002
2.0    0.108247
3.0    0.066029'''
#-----------------------------------------------------------------------------------------
#32 no_of_children 
data1.info()
data1.no_of_children.value_counts()

sum(data1.no_of_children.value_counts())#26458 some NAs

import seaborn as sns
sns.countplot(x ='no_of_children', data = data1)

(len(data1.no_of_children)-data1.no_of_children.describe()['count'])/len(data1.no_of_children)
#0.00932339835998053
print('% of each cat')
print(data1['no_of_children'].value_counts(normalize=True))
'''0.0    0.705722
1.0    0.120002
2.0    0.108247
3.0    0.066029'''
#-----------------------------------------------------------------------------------------

'''#ignore has_health_insur this column coz near abt 50% data is missing not significant
#we will continue  with dr recommended h1n1 vaccine and seasonal flu vaccine var, 
#we will delete missing values from this 2  and other vars.'''

#Droping var 15  has_health_insur
data1 = data1.drop('has_health_insur', axis =1)
data1.info()

#droping all na from data1
data2 = data1.dropna()
data2.info()
data2.shape
1- 19642/26707 #0.26 %

#writing new file in directry os.chdir("C:/Users/khile/Desktop/WD_python")

data2.to_csv('df2_h1n1.csv')


 # God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

vac = pd.read_csv('df2_H1N1.csv')
vac.info()
vac.shape
vac = vac.drop(['Unnamed: 0', 'unique_id'], axis = 1) #Removed unnecessary columns
vac.info()

#MLE - Maximum Likelihood estimation
import statsmodels.formula.api as smf
sm_logit = smf.logit('''h1n1_vaccine~h1n1_worry+h1n1_awareness+antiviral_medication+
                     contact_avoidance+bought_face_mask+wash_hands_frequently+
                     avoid_large_gatherings+reduced_outside_home_cont+
                     avoid_touch_face+dr_recc_h1n1_vacc+dr_recc_seasonal_vacc+
                     chronic_medic_condition+cont_child_undr_6_mnths+
                     is_health_worker+is_h1n1_vacc_effective+is_h1n1_risky+
                     sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky+
                     sick_from_seas_vacc+C(age_bracket)+C(qualification)
                     +C(race)+C(sex)+C(income_level)+C(marital_status)+
                     C(marital_status)+C(housing_status)+C(employment)+
                     C(census_msa)+no_of_adults+no_of_children''',
                     data=vac).fit() #target variable and response variable

sm_logit.summary()

#Comments on summary
'''Income level,Age_bracket,qualification being these variable have an order
 they can be ordinal not categorical'''

#The variable having more than 0.05 p value can be removed 
#Categorical variable for which one value is  less than 0.05 and other is 
#greater than 0.05, those variables we are not removing
'''income_level, housing status, employement, #antiviral_medication, 
contact_avoidance,wash_hands_frequently, reduced_outside_home_cont, 
avoid_touch_face, sick_from_h1n1_vacc, no_of_adults, no_of_children'''

predictions = sm_logit.predict() #Predictions are in probabilities
predictions_nominal = [0 if x < 0.5 else 1 for x in predictions]

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(vac.h1n1_vaccine, predictions_nominal)
print(cm)
'''
[[14206   922]
 [ 2410  2104]]'''

#Accuracy score - correct predictions / total number of data points
(14206+2104)/(19642) #0.83

#ROC & AUC
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(vac.h1n1_vaccine, predictions)
roc_auc = auc(fpr, tpr) #Area under Curve 0.8420
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()
##############Calculation################

2113/4514# 0.46 prcision of grp 1
14223/15128 # 0.94 Recall of grp 0

# macro avg 
(0.855+0.695)/2 #0.775
(0.939+0.466)/2 #0.703

#weighted avg calculation
(0.855 * (15128/19642)) + 0.695 *(4514 / 19642)# 0.818
(0.939 * (15128/19642)) + 0.466 *(4514 / 19642)# 0.830

'''
1 SMOTE Application
2 removal of highr p values
3 standard scaling (ordinal and continues scale)
4 first , create a subset of all ordinal and con vars
5 scale the above subset of vars 
6 del/drop all origional  ordinaland continues var from data
7     
'''
#Classification Report
from sklearn.metrics import classification_report
print(classification_report(vac.h1n1_vaccine, predictions_nominal, digits = 3))
'''
              precision    recall  f1-score   support

           0      0.855     0.939     0.895     15128
           1      0.695     0.466     0.558      4514

    accuracy                          0.830     19642
   macro avg      0.775     0.703     0.727     19642
weighted avg      0.818     0.830     0.818     19642  '''
















