# God, Pardon me please!
# God is my Saviour!
import os
os.chdir("C:/Users/khile/Desktop/WD_python")
# our exported file will appear here  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy import stats 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler 

#__________standardized data
d = pd.read_csv('D:/data _science/Basic_Notes/Manupuation/cs2m.csv')
d.info()
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
d_sc = sc_x.fit_transform(d)
d_sc
ddf = pd.DataFrame(data = d_sc)
# if y/those vars in 0 and 1, do not scale it 
ddf #see the dataframe
