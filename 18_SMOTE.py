# God is Great! Jesus is my Saviour!

import os
os.chdir("C:/Users/khile/Desktop/WD_python")
import pandas as pd
import sklearn
from sklearn.utils import resample
data = pd.read_csv("D:/data _science/PYTHON/DecisionTreeinPython/UpSamp_ Demo.csv")
data = pd.DataFrame(data)
data.shape # 14 by 4
data.info()
data.default.value_counts() # 27% are catg 1, imbalanced data!
3/14 # 21% are 1

# separate minority and majority classes

not_default = data[data.default==0] #11
len(not_default) #11
default = data[data.default==1] # 3
len(default) # 3

#__________________________________ upsample minority_with replacement
from sklearn.utils import resample
default_upsampled1 = resample(default,
                          replace=True, # sample with replacement
                          n_samples=len(not_default), # match number in majority class
                          random_state=27) # reproducible results
default_upsampled1
# combine majority and upsampled minority

upsampled080321 = pd.concat([not_default, default_upsampled1]) 

# check new class counts
upsampled080321.default.value_counts() #11 11, You should be Happy!

upsampled080321.to_csv('upsampled03142021.csv')
