#God is my saviour
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


td = pd.read_csv("D:/data _science/PYTHON/SVM_Python/bank-additional-full.csv")
td.info() # All variables have missing values
td.shape #(41188, 21)

td.isnull().sum()

#Target Variable - y - has the client subscribed a term deposit? 
#(binary: 'yes','no')
td.y.describe()
'''
count     41188
unique        2
top          no
freq      36548'''

td.y.value_counts()
'''
no     36548
yes     4640'''

#Converting data to numeric
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
td.y = le.fit_transform(td.y)

td.y.describe()
'''
count    41188.000000
mean         0.112654
std          0.316173
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max          1.000000'''

td.y.value_counts()
'''
0    36548
1     4640'''

#Count/Bar Plot
sns.countplot(x='y', data=td)
plt.title('Counts of Term depost')
plt.xlabel('Client subscribed a Term Deposit')

#age - Age of the client
td.age.describe()
'''
count    41188.00000
mean        40.02406
std         10.42125
min         17.00000
25%         32.00000
50%         38.00000
75%         47.00000
max         98.00000'''

td.age.value_counts() #78 different values

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='age', data=td)
plt.xlabel('Age of the client', size=20)
plt.ylabel('Counts', size=20)
plt.title('Barplot of Age of the clients', size=24)
plt.xticks(rotation=90)
plt.show()

#Histogram
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.hist(td.age, facecolor='green')
plt.xlabel('Age of the client', size=20)
plt.ylabel('Counts', size=20)
plt.title('Histogram of Age of the clients', size=24)
plt.xticks(rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot age of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('age')['y'].count())
plt.plot(td[td.y==1].groupby('age')['y'].count())
plt.xlabel('Age of the client', size=16)
plt.ylabel('Counts', size=16)
plt.title('Age of the client vs Term Deposit', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.age.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Age of the clients') 

#Getting the Iqr, up_lim & low_lim
iqr = td.age.describe()['75%'] - td.age.describe()['25%'] #15
up_lim = td.age.describe()['75%']+1.5*iqr # 69.5
len(td.age[td.age > up_lim]) #469 outliers

for i in np.arange(69,99,5):
    outliers = len(td.age[td.age > i])
    print('At a limit of :', i, 'There are', outliers, 'outliers')
'''
At a limit of : 69 There are 469 outliers
At a limit of : 74 There are 269 outliers
At a limit of : 79 There are 150 outliers
At a limit of : 84 There are 58 outliers
At a limit of : 89 There are 10 outliers
At a limit of : 94 There are 3 outliers'''

#job 
'''Type of job (categorical: 'admin.','blue-collar', 'entrepreneur', 
'housemaid', 'management', 'retired','self-employed', 'services', 
'student', 'technician', 'unemployed', 'unknown')'''
    
td.job.describe()
'''
count      41188
unique        12
top       admin.
freq       10422'''

td.job.value_counts() 
'''
admin.           10422
blue-collar       9254
technician        6743
services          3969
management        2924
retired           1720
entrepreneur      1456
self-employed     1421
housemaid         1060
unemployed        1014
student            875
unknown            330'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='job', data=td)
plt.xlabel('Job of the client', size=20)
plt.ylabel('Counts', size=20)
plt.title('Barplot of job of the clients', size=24)
plt.xticks(rotation=30, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot job of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('job')['y'].count())
plt.plot(td[td.y==1].groupby('job')['y'].count())
plt.xlabel('Job of the client', size=16)
plt.ylabel('Counts', size=16)
plt.title('Job of the client vs Term Deposit', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.xticks(rotation=45)
plt.show()

#Converting the data to numeric
job = {'admin.':1, 'blue-collar':2, 'technician':3, 'services':4, 'management':5,
         'retired':6, 'entrepreneur':7, 'self-employed':8, 'housemaid':9, 
         'unemployed':10, 'student':11, 'unknown':12}
#Converting job names to numbers
td.job = [job[item] for item in td.job]
td.job.describe()
'''
count    41188.000000
mean         3.515587
std          2.701267
min          1.000000
25%          1.000000
50%          3.000000
75%          5.000000
max         12.000000'''

#marital 
'''Marital status (categorical: 'divorced','married','single','unknown'; 
note: 'divorced' means divorced or widowed)'''
td.marital.describe()
'''
count       41188
unique          4
top       married
freq        24928'''

td.marital.value_counts()
'''
married     24928
single      11568
divorced     4612
unknown        80'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='marital', data=td)
plt.xlabel('Marital status of the client', size=20)
plt.ylabel('Counts', size=20)
plt.title('Marital status of the client', size=24)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot Marital status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('marital')['y'].count())
plt.plot(td[td.y==1].groupby('marital')['y'].count())
plt.xlabel('Marital status of the client', size=16)
plt.ylabel('Counts', size=16)
plt.title('Marital status of the client vs Term Deposit', size=18)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting the data to numeric
marital = {'single':1, 'married':2, 'divorced':3, 'unknown':4}
#Converting job names to numbers
td.marital = [marital[item] for item in td.marital]
td.marital.describe()
'''
count    41188.000000
mean         1.835000
std          0.611053
min          1.000000
25%          1.000000
50%          2.000000
75%          2.000000
max          4.000000'''

#education 
'''Categorical: 'basic.4y', 'basic.6y' ,'basic.9y', 'high.school', 
'illiterate', 'professional.course', 'university.degree','unknown''''
td.education.describe()
'''
count                 41188
unique                    8
top       university.degree
freq                  12168'''

td.education.value_counts()
'''
university.degree      12168
high.school             9515
basic.9y                6045
professional.course     5243
basic.4y                4176
basic.6y                2292
unknown                 1731
illiterate                18'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='education', data=td)
plt.xlabel('education status of the client', size=20)
plt.ylabel('counts', size=20)
plt.title('education status of the client', size=24)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)    
plt.show()

#Plot education status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('education')['y'].count())
plt.plot(td[td.y==1].groupby('education')['y'].count())
plt.xlabel('education status of the client', size=16)
plt.ylabel('counts', size=16)
plt.title('education status of the client vs Term Deposit', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.xticks(rotation=45)
plt.show()

#Converting the data to numeric
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
td.education = le.fit_transform(td.education)
td.education.describe()
'''
count    41188.000000
mean         3.747184
std          2.136482
min          0.000000
25%          2.000000
50%          3.000000
75%          6.000000
max          7.000000'''

td.education.value_counts()
'''
6    12168
3     9515
2     6045
5     5243
0     4176
1     2292
7     1731
4       18'''

#default 
'''has credit in default? (categorical: 'no','yes','unknown')'''

td.default.describe()
'''
count     41188
unique        3
top          no
freq      32588'''

td.default.value_counts()
'''
no         32588
unknown     8597
yes            3'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='default', data=td)
plt.xlabel('default status of the client', size=20)
plt.ylabel('Counts', size=20)
plt.title('Default status of the client', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot default status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('default')['y'].count())
plt.plot(td[td.y==1].groupby('default')['y'].count())
plt.xlabel('Default status of the client', size=16)
plt.ylabel('Counts', size=16)
plt.title('Default status of the client vs Term Deposit', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting the data to numeric
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
td.default = le.fit_transform(td.default)
td.default.describe()
'''
count    41188.000000
mean         0.208872
std          0.406686
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max          2.000000'''

td.default.value_counts()
'''
0    32588
1     8597
2        3'''

#housing 
'''has a housing loan? (categorical: 'no','yes','unknown')'''

td.housing.describe()
'''
count     41188
unique        3
top         yes
freq      21576'''

td.housing.value_counts()
'''
yes        21576
no         18622
unknown      990'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='housing', data=td)
plt.xlabel('Housing loan status of the client', size=20)
plt.ylabel('Counts', size=20)
plt.title('Housing loan status of the client', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot housing status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('housing')['y'].count())
plt.plot(td[td.y==1].groupby('housing')['y'].count())
plt.xlabel('housing loan status of the client', size=16)
plt.ylabel('counts', size=16)
plt.title('housing loan status of the client vs Term Deposit', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting the data to numeric
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
td.housing = le.fit_transform(td.housing)
td.housing.describe()
'''
count    41188.000000
mean         1.071720
std          0.985314
min          0.000000
25%          0.000000
50%          2.000000
75%          2.000000
max          2.000000'''

td.housing.value_counts()
'''
2    21576
0    18622
1      990'''

#loan 
'''has a personal loan? (categorical: 'no','yes','unknown')'''

td.loan.describe()
'''
count     41188
unique        3
top          no
freq      33950'''

td.loan.value_counts()
'''
no         33950
yes         6248
unknown      990'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='loan', data=td)
plt.xlabel('Personal loan status of the client', size=16)
plt.ylabel('counts', size=16)
plt.title('Personal loan status of the client', size=20)
plt.xticks(fontsize=14)
plt.show()

#Plot loan status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('loan')['y'].count())
plt.plot(td[td.y==1].groupby('loan')['y'].count())
plt.xlabel('Personal loan status of the client', size=16)
plt.ylabel('counts', size=16)
plt.title('Personal loan status of the client vs Term Deposit', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting the data to numeric
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
td.loan = le.fit_transform(td.loan)
td.loan.describe()
'''
count    41188.000000
mean         0.327425
std          0.723616
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max          2.000000'''

td.loan.value_counts()
'''
0    33950
2     6248
1      990'''

#contact 
'''Contact communication type (categorical: 'cellular','telephone'''

td.contact.describe()
'''
count        41188
unique           2
top       cellular
freq         26144'''

td.contact.value_counts()
'''
cellular     26144
telephone    15044'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='contact', data=td)
plt.xlabel('communication type of the client', size=20)
plt.ylabel('counts', size=20)
plt.title('communication type of the client', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot contact status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('contact')['y'].count())
plt.plot(td[td.y==1].groupby('contact')['y'].count())
plt.xlabel('Communication type of the client', size=14)
plt.ylabel('Counts', size=14)
plt.title('Communication type of the client vs Term Deposit', size=16)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting the data to numeric
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
td.contact = le.fit_transform(td.contact)
td.contact.describe()
'''
count    41188.000000
mean         0.365252
std          0.481507
min          0.000000
25%          0.000000
50%          0.000000
75%          1.000000
max          1.000000'''

td.contact.value_counts()
'''
0    26144
1    15044'''

#month 
'''last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 
'nov', 'dec')'''

td.month.describe()
'''
count     41188
unique       10
top         may
freq      13769'''

td.month.value_counts()
'''
may    13769
jul     7174
aug     6178
jun     5318
nov     4101
apr     2632
oct      718
sep      570
mar      546
dec      182'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='month', data=td)
plt.xlabel('Monthly wise', size=20)
plt.ylabel('Counts', size=20)
plt.title('Monthly wise counts', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot month status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('month')['y'].count())
plt.plot(td[td.y==1].groupby('month')['y'].count())
plt.xlabel('Monthly wise', size=16)
plt.ylabel('Counts', size=16)
plt.title('Monthly wise counts', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting the data to numeric
month = {'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 
         'sep':9, 'oct':10, 'nov':11, 'dec':12}
#Converting month names to numbers
td.month = [month[item] for item in td.month]
td.month.describe()
'''
count    41188.000000
mean         6.607896
std          2.040998
min          3.000000
25%          5.000000
50%          6.000000
75%          8.000000
max         12.000000'''

td.month.value_counts()
'''
5     13769
7      7174
8      6178
6      5318
11     4101
4      2632
10      718
9       570
3       546
12      182'''

#day_of_week 
'''last contact day of the week (categorical: 'mon','tue','wed','thu','fri')'''

td.day_of_week.describe()
'''
count     41188
unique        5
top         thu
freq       8623'''

td.day_of_week.value_counts()
'''
thu    8623
mon    8514
wed    8134
tue    8090
fri    7827'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='day_of_week', data=td)
plt.xlabel('Week Day wise', size=20)
plt.ylabel('Counts', size=20)
plt.title('Week Day wise counts', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot day_of_week status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('day_of_week')['y'].count())
plt.plot(td[td.y==1].groupby('day_of_week')['y'].count())
plt.xlabel('Week Day  wise', size=16)
plt.ylabel('Counts', size=16)
plt.title('Week Day wise counts', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting the data to numeric
day_of_week = {'mon':2, 'tue':3, 'wed':4, 'thu':5, 'fri':6}
#Converting day_of_week names to numbers
td.day_of_week = [day_of_week[item] for item in td.day_of_week]
td.day_of_week.describe()
'''
count    41188.000000
mean         3.979581
std          1.411514
min          2.000000
25%          3.000000
50%          4.000000
75%          5.000000
max          6.000000'''

td.day_of_week.value_counts()
'''
5    8623
2    8514
4    8134
3    8090
6    7827'''

#duration 
'''last contact duration, in seconds'''

td.duration.describe()
'''
count    41188.000000
mean       258.285010
std        259.279249
min          0.000000
25%        102.000000
50%        180.000000
75%        319.000000
max       4918.000000'''

td.duration.value_counts() #1544 different numbers

#Histogram
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.hist(td.duration, facecolor='Blue')
plt.xlabel('Last contact duration', size=20)
plt.ylabel('Counts', size=20)
plt.title('Last contact duration', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('duration')['y'].count())
plt.plot(td[td.y==1].groupby('duration')['y'].count())
plt.xlabel('Last contact duration', size=16)
plt.ylabel('Counts', size=16)
plt.title('Last contact duration', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.duration.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Last contact duration')

#Getting the Iqr, up_lim & low_lim
iqr = td.duration.describe()['75%'] - td.duration.describe()['25%'] #217
up_lim = td.duration.describe()['75%']+1.5*iqr # 644.5
len(td.duration[td.duration > up_lim]) #2963 outliers

for i in np.arange(650,5000,500):
    outliers = len(td.duration[td.duration > i])
    print('At a limit of :', i, 'There are', outliers, 'outliers')
'''
At a limit of : 650 There are 2915 outliers
At a limit of : 1150 There are 597 outliers
At a limit of : 1650 There are 132 outliers
At a limit of : 2150 There are 42 outliers
At a limit of : 2650 There are 21 outliers
At a limit of : 3150 There are 12 outliers
At a limit of : 3650 There are 3 outliers
At a limit of : 4150 There are 2 outliers
At a limit of : 4650 There are 1 outliers'''

#campaign 
'''number of contacts performed during this campaign and for this client 
(numeric, includes last contact)'''

td.campaign.describe()
'''
count    41188.000000
mean         2.567593
std          2.770014
min          1.000000
25%          1.000000
50%          2.000000
75%          3.000000
max         56.000000'''

td.campaign.value_counts() #42 different numbers

#Barplot/Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='campaign', data=td)
plt.xlabel('No of contacts performed during the campaign', size=20)
plt.ylabel('Counts', size=20)
plt.title('No of contacts performed during the campaign', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('campaign')['y'].count())
plt.plot(td[td.y==1].groupby('campaign')['y'].count())
plt.xlabel('No of contacts performed during the campaign', size=16)
plt.ylabel('Counts', size=16)
plt.title('No of contacts performed during the campaign', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.campaign.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('No of contacts performed during the campaign')

#Getting the Iqr, up_lim & low_lim
iqr = td.campaign.describe()['75%'] - td.campaign.describe()['25%'] #2
up_lim = td.campaign.describe()['75%']+1.5*iqr # 6
len(td.campaign[td.campaign > up_lim]) #2406 outliers

for i in np.arange(6,50,5):
    outliers = len(td.campaign[td.campaign > i])
    print('At a limit of :', i, 'There are', outliers, 'outliers')
'''
At a limit of : 6 There are 2406 outliers
At a limit of : 11 There are 692 outliers
At a limit of : 16 There are 304 outliers
At a limit of : 21 There are 133 outliers
At a limit of : 26 There are 69 outliers
At a limit of : 31 There are 26 outliers
At a limit of : 36 There are 10 outliers
At a limit of : 41 There are 5 outliers
At a limit of : 46 There are 1 outliers'''

#pdays 
'''number of days that passed by after the client was last contacted from a 
previous campaign (numeric; 999 means client was not previously contacted)'''

td.pdays.describe()
'''
count    41188.000000
mean       962.475454
std        186.910907
min          0.000000
25%        999.000000
50%        999.000000
75%        999.000000
max        999.000000'''

td.pdays.value_counts() #27 different numbers

len(td[td.pdays==999]['pdays'])/len(td.pdays) #96.3% are not yet contacted
#Barplot/Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='pdays', data=td)
plt.xlabel('No of days passed after last contact', size=20)
plt.ylabel('Counts', size=20)
plt.title('No of days passed after last contact', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('pdays')['y'].count())
plt.plot(td[td.y==1].groupby('pdays')['y'].count())
plt.xlabel('No of days passed after last contact', size=16)
plt.ylabel('Counts', size=16)
plt.title('No of days passed after last contact', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.pdays.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('No of days passed after last contact')

#Getting the Iqr, up_lim & low_lim
iqr = td.pdays.describe()['75%'] - td.pdays.describe()['25%'] #0
low_lim = td.pdays.describe()['25%']-1.5*iqr # 999
len(td.pdays[td.pdays < low_lim]) #1515 outliers

for i in np.arange(0,50,5):
    outliers = len(td.pdays[td.pdays < i])
    print('At a limit of :', i, 'There are', outliers, 'outliers')
'''
At a limit of : 0 There are 0 outliers
At a limit of : 5 There are 659 outliers
At a limit of : 10 There are 1259 outliers
At a limit of : 15 There are 1453 outliers
At a limit of : 20 There are 1506 outliers
At a limit of : 25 There are 1512 outliers
At a limit of : 30 There are 1515 outliers
At a limit of : 35 There are 1515 outliers
At a limit of : 40 There are 1515 outliers
At a limit of : 45 There are 1515 outliers'''

#previous 
'''number of contacts performed before this campaign and for this client'''

td.previous.describe()
'''
count    41188.000000
mean         0.172963
std          0.494901
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max          7.000000'''

td.previous.value_counts() 
'''
0    35563
1     4561
2      754
3      216
4       70
5       18
6        5
7        1'''

#Barplot/Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='previous', data=td)
plt.xlabel('No of contacts performed before campaign', size=20)
plt.ylabel('Counts', size=20)
plt.title('No of contacts performed before campaign', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('previous')['y'].count())
plt.plot(td[td.y==1].groupby('previous')['y'].count())
plt.xlabel('No of contacts performed before campaign', size=16)
plt.ylabel('Counts', size=16)
plt.title('No of contacts performed before campaign', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.previous.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('No of contacts performed before campaign') #Outliers

#Getting the Iqr, up_lim & low_lim
iqr = td.previous.describe()['75%'] - td.previous.describe()['25%'] #0
up_lim = td.previous.describe()['75%']+1.5*iqr # 0
len(td.previous[td.previous > up_lim]) #5625 outliers

for i in np.arange(1,7,1):
    outliers = len(td.previous[td.previous > i])
    print('At a limit of :', i, 'There are', outliers, 'outliers')
'''
At a limit of : 1 There are 1064 outliers
At a limit of : 2 There are 310 outliers
At a limit of : 3 There are 94 outliers
At a limit of : 4 There are 24 outliers
At a limit of : 5 There are 6 outliers
At a limit of : 6 There are 1 outliers'''

#poutcome 
'''outcome of the previous marketing campaign (categorical: 'failure',
'nonexistent','success')'''

td.poutcome.describe()
'''
count           41188
unique              3
top       nonexistent
freq            35563'''

td.poutcome.value_counts() 
'''
nonexistent    35563
failure         4252
success         1373'''

#Barplot/Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='poutcome', data=td)
plt.xlabel('Outcome of the previous marketing campaign', size=20)
plt.ylabel('Counts', size=20)
plt.title('Outcome of the previous marketing campaign', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('poutcome')['y'].count())
plt.plot(td[td.y==1].groupby('poutcome')['y'].count())
plt.xlabel('Outcome of the previous marketing campaign', size=16)
plt.ylabel('Counts', size=16)
plt.title('Outcome of the previous marketing campaign', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Converting data to numeric
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
td.poutcome = le.fit_transform(td.poutcome)
td.poutcome.describe()
'''
count    41188.000000
mean         0.930101
std          0.362886
min          0.000000
25%          1.000000
50%          1.000000
75%          1.000000
max          2.000000'''

#emp.var.rate / emp_var_rate
'''employment variation rate - quarterly indicator (numeric)'''
#Renaming the column name, as the dots in the names troubles for some 
#direct coding. Replacing dots with underscore
td = td.rename(columns={"emp.var.rate":"emp_var_rate"})
td.info()

td.emp_var_rate.describe()
'''
count    41188.000000
mean         0.081886
std          1.570960
min         -3.400000
25%         -1.800000
50%          1.100000
75%          1.400000
max          1.400000'''

td.emp_var_rate.value_counts() 
'''
 1.4    16234
-1.8     9184
 1.1     7763
-0.1     3683
-2.9     1663
-3.4     1071
-1.7      773
-1.1      635
-3.0      172
-0.2       10'''

#Barplot/Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='emp_var_rate', data=td)
plt.xlabel('Employement variation Rate - Quaterly', size=20)
plt.ylabel('Counts', size=20)
plt.title('Employement variation Rate - Quaterly', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('emp_var_rate')['y'].count())
plt.plot(td[td.y==1].groupby('emp_var_rate')['y'].count())
plt.xlabel('Employement variation Rate - Quaterly', size=16)
plt.ylabel('Counts', size=16)
plt.title('Employement variation Rate - Quaterly', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.emp_var_rate.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Employement variation Rate - Quaterly') #No outliers

#cons.price.idx /cons_price_idx
'''consumer price index - monthly indicator'''
#Renaming the column name, as the dots in the names troubles for some 
#direct coding. Replacing dots with underscore
td = td.rename(columns={"cons.price.idx":"cons_price_idx"})
td.info()

td.cons_price_idx.describe()
'''
count    41188.000000
mean        93.575664
std          0.578840
min         92.201000
25%         93.075000
50%         93.749000
75%         93.994000
max         94.767000'''

td.cons_price_idx.value_counts()
'''
93.994    7763
93.918    6685
92.893    5794
93.444    5175
94.465    4374
93.200    3616
93.075    2458
92.201     770
92.963     715
92.431     447
92.649     357
94.215     311
94.199     303
92.843     282
92.379     267
93.369     264
94.027     233
94.055     229
93.876     212
94.601     204
92.469     178
93.749     174
92.713     172
94.767     128
93.798      67
92.756      10'''

#Barplot/Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='cons_price_idx', data=td)
plt.xlabel('Consumer price index - monthly indicator', size=20)
plt.ylabel('Counts', size=20)
plt.title('Consumer price index - monthly indicator', size=24)
plt.xticks(rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('cons_price_idx')['y'].count())
plt.plot(td[td.y==1].groupby('cons_price_idx')['y'].count())
plt.xlabel('Consumer price index - monthly indicator', size=16)
plt.ylabel('Counts', size=16)
plt.title('Consumer price index - monthly indicator', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.cons_price_idx.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Consumer price index - monthly indicator') #NO outliers

#cons.conf.idx /cons_conf_idx
'''consumer confidence index - monthly indicator'''
#Renaming the column name, as the dots in the names troubles for some 
#direct coding. Replacing dots with underscore
td = td.rename(columns={"cons.conf.idx":"cons_conf_idx"})
td.info()

td.cons_conf_idx.describe()
'''
count    41188.000000
mean       -40.502600
std          4.628198
min        -50.800000
25%        -42.700000
50%        -41.800000
75%        -36.400000
max        -26.900000'''

td.cons_conf_idx.value_counts()
'''
-36.4    7763
-42.7    6685
-46.2    5794
-36.1    5175
-41.8    4374
-42.0    3616
-47.1    2458
-31.4     770
-40.8     715
-26.9     447
-30.1     357
-40.3     311
-37.5     303
-50.0     282
-29.8     267
-34.8     264
-38.3     233
-39.8     229
-40.0     212
-49.5     204
-33.6     178
-34.6     174
-33.0     172
-50.8     128
-40.4      67
-45.9      10'''

#Barplot/Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='cons_conf_idx', data=td)
plt.xlabel('Consumer confidence index - monthly indicator', size=20)
plt.ylabel('Counts', size=20)
plt.title('Consumer confidence index - monthly indicator', size=24)
plt.xticks(rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('cons_conf_idx')['y'].count())
plt.plot(td[td.y==1].groupby('cons_conf_idx')['y'].count())
plt.xlabel('Consumer confidence index - monthly indicator', size=16)
plt.ylabel('Counts', size=16)
plt.title('Consumer confidence index - monthly indicator', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.cons_conf_idx.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Consumer confidence index - monthly indicator') #Outliers

#Getting the Iqr, up_lim & low_lim
iqr = td.cons_conf_idx.describe()['75%'] - td.cons_conf_idx.describe()['25%'] #6.30
up_lim = td.cons_conf_idx.describe()['75%']+1.5*iqr # -26.94
len(td.cons_conf_idx[td.cons_conf_idx > up_lim]) #447 outliers

for i in np.arange(-27,-26,.3):
    outliers = len(td.cons_conf_idx[td.cons_conf_idx > i])
    print('At a limit of :', i, 'There are', outliers, 'outliers')
'''
At a limit of : -27.0 There are 447 outliers
At a limit of : -26.7 There are 0 outliers
At a limit of : -26.4 There are 0 outliers
At a limit of : -26.099999999999998 There are 0 outliers'''

#euribor3m
'''euribor 3 month rate - daily indicator'''

td.euribor3m.describe()
'''
count    41188.000000
mean         3.621291
std          1.734447
min          0.634000
25%          1.344000
50%          4.857000
75%          4.961000
max          5.045000'''

td.euribor3m.value_counts() #316 Different values

#Histogram
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.hist(td.euribor3m, facecolor='Blue')
plt.xlabel('Euribor 3 month rate - daily indicator', size=20)
plt.ylabel('Counts', size=20)
plt.title('Euribor 3 month rate - daily indicator', size=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('euribor3m')['y'].count())
plt.plot(td[td.y==1].groupby('euribor3m')['y'].count())
plt.xlabel('Euribor 3 month rate - daily indicator', size=16)
plt.ylabel('Counts', size=16)
plt.title('Euribor 3 month rate - daily indicator', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.euribor3m.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Euribor 3 month rate - daily indicator')# No outliers

#nr.employed / nr_employed
'''number of employees - quarterly indicator'''
#Renaming the column name, as the dots in the names troubles for some 
#direct coding. Replacing dots with underscore
td = td.rename(columns={"nr.employed":"nr_employed"})
td.info()

td.nr_employed.describe()
'''
count    41188.000000
mean      5167.035911
std         72.251528
min       4963.600000
25%       5099.100000
50%       5191.000000
75%       5228.100000
max       5228.100000'''

td.nr_employed.value_counts() #316 Different values
'''
5228.1    16234
5099.1     8534
5191.0     7763
5195.8     3683
5076.2     1663
5017.5     1071
4991.6      773
5008.7      650
4963.6      635
5023.5      172
5176.3       10'''

#Histogram
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='nr_employed', data=td)
plt.xlabel('Number of employees - quarterly indicator', size=20)
plt.ylabel('Counts', size=20)
plt.title('Number of employees - quarterly indicator', size=24)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#Plot duration status of the client vs Term Deposit
plt.plot(td[td.y==0].groupby('nr_employed')['y'].count())
plt.plot(td[td.y==1].groupby('nr_employed')['y'].count())
plt.xlabel('Number of employees - quarterly indicator', size=16)
plt.ylabel('Counts', size=16)
plt.title('Number of employees - quarterly indicator', size=20)
plt.legend(labels = ('Without Term Deposit', 'Term Deposit'))
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
td.nr_employed.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Number of employees - quarterly indicator')

#Correlation
td.corr()['y']
'''
age               0.030399
job               0.055852
marital          -0.044538
education         0.057799
default          -0.099352
housing           0.011552
loan             -0.004909
contact          -0.144773
month             0.037187
day_of_week       0.010051
duration          0.405274
campaign         -0.066357
pdays            -0.324914
previous          0.230181
poutcome          0.129789
emp_var_rate     -0.298334
cons_price_idx   -0.136211
cons_conf_idx     0.054878
euribor3m        -0.307771
nr_employed      -0.354678
y                 1.000000'''

###############################################################################
td.to_csv('td.csv', index=False) #Saving the processed data as a new file
td = pd.read_csv('td.csv')

#Assigning predictors & response variables
td.info()
x = td.iloc[:,:-1] #20 variables
y = td.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state = 0,test_size=0.25)

len(x_train) #30891
len(x_test) #10297
len(y_train) #30891
len(y_test) #10297

#Building the model
#SVC - Suppot Vector Classifier
svc =svm.SVC()
td_svc = svc.fit(x_train, y_train)
td_svc.get_params()
{'C': 1.0, #Strictly positive, It trades off b/w maximum planes b/w separating planes
 'break_ties': False, #Default false for binary class, if class is >2 it should be true
 'cache_size': 200, #Kernel cache in MB
 'class_weight': None, #Class weight is used for giving more weights to less numbered classes
 'coef0': 0.0, #It is only significant in ‘poly’ and ‘sigmoid’.
 'decision_function_shape': 'ovr', #ovr - one-vs-rest for binary class, ovo-one vs other for multiclass
 'degree': 3, #Ignored by other kernels except polynomial kernel
 'gamma': 'scale', #scale: 1 / (n_features * X.var()) or auto: 1 / (n_features)
 'kernel': 'rbf', #{‘linear’, ‘poly’, ‘rbf’ (Gaussian), ‘sigmoid’, ‘precomputed’}, default=’rbf’
 'max_iter': -1, #-1 for no limit in interations
 'probability': False, #Whether to enable probability estimates, will slow down that method as it internally uses 5-fold cross-validation
 'random_state': None, #random number generation for shuffling the data for probability estimates. Ignored when probability is False
 'shrinking': True, #If the number of iterations is large, then shrinking can shorten the training time
 'tol': 0.001,#Tolerance for stopping criterion.
 'verbose': False} #if enabled, may not work properly in a multithreaded context.

#Prediction
y_pred = td_svc.predict(x_test)
len(y_pred)
print(y_pred)

#Confusion Matrix & Report
pd.crosstab(y_test,y_pred, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict     0    1    All
Actual                   
0        8986  153   9139
1         926  232   1158
All      9912  385  10297'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) #0.895

#finding fpr, tpr & thresholds
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)

#AUC -Area Under Curve
td_roc_auc = auc(fpr,tpr)
print(td_roc_auc) #0.59

#ROC Curve
plt.title('ROC Curve for Term Deposit -RBF')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(td_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.91      0.98      0.94      9139
           1       0.60      0.20      0.30      1158

    accuracy                           0.90     10297
   macro avg       0.75      0.59      0.62     10297
weighted avg       0.87      0.90      0.87     10297'''


#Building the model - 2
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced') #Will be used when data is not balanced
td_svc = svc.fit(x_train, y_train)
td_svc.get_params()

#Prediction
y_pred = td_svc.predict(x_test)
len(y_pred)
print(y_pred)

#Confusion Matrix & Report
pd.crosstab(y_test,y_pred, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict     0     1    All
Actual                    
0        7675  1464   9139
1         155  1003   1158
All      7830  2467  10297'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) #0.842

#finding fpr, tpr & thresholds
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)

#AUC -Area Under Curve
td_roc_auc = auc(fpr,tpr)
print(td_roc_auc) #0.852

#ROC Curve
plt.title('ROC Curve for Term Deposit - RBF with Balanced Weight')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(td_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.84      0.90      9139
           1       0.41      0.87      0.55      1158

    accuracy                           0.84     10297
   macro avg       0.69      0.85      0.73     10297
weighted avg       0.92      0.84      0.87     10297'''

###############################################################################
#Building Model with sample data
#Sampling the data
td1 = pd.read_csv('td.csv')
td1 = td1.sample(4000, random_state=21)
td1.info()

x1 = td1.iloc[:,:-1]
y1 = td1.iloc[:,-1]

#Building the model - 3
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', kernel='linear') 
td_svc = svc.fit(x1, y1)
td_svc.get_params()

#Prediction
y_pred_lin = td_svc.predict(x1)
len(y_pred_lin)
print(y_pred_lin)

#Confusion Matrix & Report
pd.crosstab(y1,y_pred_lin, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict     0    1   All
Actual                  
0        3172  380  3552
1         122  326   448
All      3294  706  4000'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y1,y_pred_lin) #0.874

#finding fpr, tpr & thresholds
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y1, y_pred_lin)

#AUC -Area Under Curve
td_roc_auc = auc(fpr,tpr)
print(td_roc_auc) #0.810

#ROC Curve
plt.title('ROC Curve for Term Deposit - Linear(Sample@4000)')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(td_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y1, y_pred_lin))
'''
              precision    recall  f1-score   support

           0       0.96      0.89      0.93      3552
           1       0.46      0.73      0.56       448

    accuracy                           0.87      4000
   macro avg       0.71      0.81      0.75      4000
weighted avg       0.91      0.87      0.89      4000'''

#Building the model - 4
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', kernel='poly') 
td_svc = svc.fit(x1, y1)
td_svc.get_params()

#Prediction
y_pred_poly = td_svc.predict(x1)
len(y_pred_poly)
print(y_pred_poly)

#Confusion Matrix & Report
pd.crosstab(y1,y_pred_poly, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict     0    1   All
Actual                  
0        3017  535  3552
1          77  371   448
All      3094  906  4000'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y1,y_pred_poly) #0.847

#finding fpr, tpr & thresholds
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y1, y_pred_poly)

#AUC -Area Under Curve
td_roc_auc = auc(fpr,tpr)
print(td_roc_auc) #0.838

#ROC Curve
plt.title('ROC Curve for Term Deposit - Poly(Sample@4000)')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(td_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y1, y_pred_poly))
'''
              precision    recall  f1-score   support

           0       0.98      0.85      0.91      3552
           1       0.41      0.83      0.55       448

    accuracy                           0.85      4000
   macro avg       0.69      0.84      0.73      4000
weighted avg       0.91      0.85      0.87      4000'''

#Building the model - 5
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', kernel='sigmoid') 
td_svc = svc.fit(x1, y1)
td_svc.get_params()

#Prediction
y_pred_sig = td_svc.predict(x1)
len(y_pred_sig)
print(y_pred_sig)

#Confusion Matrix & Report
pd.crosstab(y1,y_pred_sig, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict     0    1   All
Actual                  
0        3192  360  3552
1         165  283   448
All      3357  643  4000'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y1,y_pred_sig) #0.868

#finding fpr, tpr & thresholds
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y1, y_pred_sig)

#AUC -Area Under Curve
td_roc_auc = auc(fpr,tpr)
print(td_roc_auc) #0.765

#ROC Curve
plt.title('ROC Curve for Term Deposit - Sigmoid(Sample@4000)')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(td_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y1, y_pred_sig))
'''
              precision    recall  f1-score   support

           0       0.95      0.90      0.92      3552
           1       0.44      0.63      0.52       448

    accuracy                           0.87      4000
   macro avg       0.70      0.77      0.72      4000
weighted avg       0.89      0.87      0.88      4000'''

#Building the model - 6
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced') #rbf is default
td_svc = svc.fit(x1, y1)
td_svc.get_params()

#Prediction
y_pred_rbf = td_svc.predict(x1)
len(y_pred_rbf)
print(y_pred_rbf)

#Confusion Matrix & Report
pd.crosstab(y1,y_pred_rbf, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict     0    1   All
Actual                  
0        3114  438  3552
1         105  343   448
All      3219  781  4000'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y1,y_pred_rbf) #0.864

#finding fpr, tpr & thresholds
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y1, y_pred_rbf)

#AUC -Area Under Curve
td_roc_auc = auc(fpr,tpr)
print(td_roc_auc) #0.821

#ROC Curve
plt.title('ROC Curve for Term Deposit - RBF(Sample@4000)')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(td_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y1, y_pred_rbf))
'''
              precision    recall  f1-score   support

           0       0.97      0.88      0.92      3552
           1       0.44      0.77      0.56       448

    accuracy                           0.86      4000
   macro avg       0.70      0.82      0.74      4000
weighted avg       0.91      0.86      0.88      4000'''

###############################################################################
'''With Probability True'''
#Building the model - 7
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', probability=True) #rbf is default
td_svc = svc.fit(x1, y1)
td_svc.get_params()

#Prediction
yp_pred_rbf = td_svc.predict_proba(x1)[:,1]
len(yp_pred_rbf)
print(yp_pred_rbf)

#finding fpr, tpr & thresholds
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y1, yp_pred_rbf)

#AUC -Area Under Curve
td_roc_auc = auc(fpr,tpr)
print(td_roc_auc) #0.91

#ROC Curve
plt.title('ROC Curve for Term Deposit - RBF(Sample@4000)')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(td_roc_auc))
plt.legend(loc=4)
plt.show()

###############################################################################
'''With Probability True'''
#Building Model with sample data for comparision plot
#Sampling the data
td2 = pd.read_csv('td.csv')
td2 = td2.sample(25, random_state=21)
td2.info()

x2 = td2.iloc[:,:-1]
y2 = td2.iloc[:,-1]

'''With Probability True'''
#Sample model - 1
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', probability=True) #rbf is default
td_svc = svc.fit(x2, y2)
td_svc.get_params()

#Prediction
yp1_pred_rbf = td_svc.predict_proba(x2)[:,1]
len(yp1_pred_rbf)
print(yp1_pred_rbf)

#Sample model - 2
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', probability=True, kernel='linear') 
td_svc = svc.fit(x2, y2)
td_svc.get_params()

#Prediction
yp1_pred_lin = td_svc.predict_proba(x2)[:,1]
len(yp1_pred_lin)
print(yp1_pred_lin)

#Sample model - 3
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', probability=True, kernel='poly') 
td_svc = svc.fit(x2, y2)
td_svc.get_params()

#Prediction
yp1_pred_poly = td_svc.predict_proba(x2)[:,1]
len(yp1_pred_poly)
print(yp1_pred_poly)

#Sample model - 3
#SVC - Suppot Vector Classifier
svc =svm.SVC(class_weight='balanced', probability=True, kernel='sigmoid') 
td_svc = svc.fit(x2, y2)
td_svc.get_params()

#Prediction
yp1_pred_sig = td_svc.predict_proba(x2)[:,1]
len(yp1_pred_sig)
print(yp1_pred_sig)


#X is the no of observations ie 25 
x = np.arange(25)

#Plot comparision between the models
plt.plot(x,y2, lw=3)
plt.plot(x,yp1_pred_rbf)
plt.plot(x,yp1_pred_lin, color='red')
plt.plot(x,yp1_pred_poly)
plt.plot(x,yp1_pred_sig)
plt.legend(labels=('Actual','Rbf', 'Linear', 'Poly', 'Sigmoid'),
           bbox_to_anchor=(1, 1), loc=2)
plt.title('Comparing different kernel outputs with actual')
plt.xlabel('Sample of 25 observations')
plt.ylabel('Probabilities')
plt.show()