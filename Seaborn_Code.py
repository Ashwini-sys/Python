# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:12:25 2021

@author: khilesh
"""
import pandas as pd 
import seaborn as sns
#file mtcars
mtcars = pd.read_csv("D:\data _science\PYTHON\Matplot\mtcars.csv")
mtcars = pd.DataFrame(mtcars)
#file grades
grades = pd.read_csv("D:/data _science/Basic_Notes/Manupuation/grades.csv")
grades = pd.DataFrame(grades)
#file cs2m
cs2m =pd.read_csv("D:/data _science/Basic_Notes/Manupuation/cs2m.csv")
cs2m = pd.DataFrame(cs2m)
#tip
tip = sns.load_dataset("tips")
tip.info()
tip.describe()

#-----------strip plot----------
sns.stripplot(x='day', y='total_bill', data = tip)

#use to set style of background of plot
sns.set(style = 'darkgrid')
sns.stripplot(x=tip["total_bill"])

#------strip plot jitter-----------
sns.stripplot(x="day", y="total_bill", data = tip, jitter=0.2)

#---------------strip plot linewidth-------------
sns.stripplot(y="total_bill", x="day", data =tip, linewidth=0.7)

#-----------Relationship plot----------
sns.relplot(data = tip, 
            x = "total_bill", y = "tip", col = "time",
            hue = "smoker", style="smoker")

#--------------joint plots-------------
data = sns.load_dataset("attention")
data.info()

#draw joint plot with
#hex kind
sns.jointplot(x="solutions", y ="score",
              kind = "reg", data = data)    
#kind{"scatter"/ "kde" / "hist"/ "hex"/ "reg"/ "resid" }

#----------draw with kde kind---------
data = sns.load_dataset("mpg")
sns.jointplot(x = "mpg", y = "acceleration", kind = "scatter", data = data)

#------------------Count plots/Bar plots-------------
#------------count plot(bar plot)-----------
df = sns.load_dataset('tips') 
#Count plot on single categorical variable
sns.countplot(x ='sex', data = df)
#Count plots/Bar plots
grades.ethnicity.value_counts()
sns.countplot(x = 'ethnicity', data = grades)

#count plot/grouped by
#----------countplot 2 categorical vars
#count plot on two categorical variable
sns.countplot(x='sex', hue = "smoker", data = df)

sns.countplot(x = 'ethnicity', hue = 'gender', data = grades)

#change orientation
sns.countplot(y = 'sex', hue = 'smoker', data =df)

#change pallete
#use  a different colour palette in count plot
sns.countplot(x = 'sex', data = df, palette= "spring_r")  

#Box plot/Violin plot
sns.violinplot(x="ethnicity", y="final", data = grades)

sns.violinplot(x="Prgnt", y = "Age", data = cs2m)

#boxplot with violin plot of two variables
sns.violinplot(x="Prgnt", y = "Age", hue ="DrugR", data = cs2m)
#violin plot of Age
sns.violinplot(x=cs2m["Age"])# horizontal
sns.violinplot(y=cs2m["Age"])# vertical

#Box plot/Violin plot;Linewidth
sns.violinplot(x= "Prgnt", y="Age", hue ="DrugR", data = cs2m, linewidth = 3)

#Swarmplot
sns.swarmplot(x='Prgnt', y= 'Age', data =cs2m)

sns.swarmplot(y='total', x='ethnicity', data = grades)

sns.swarmplot(x='ethnicity', y='total', hue = "gender", data = grades)

#-----------------pair plot--------------------
cs2m.info()
da = cs2m[['Age', 'BP', 'Chlstrl', 'Prgnt']]
sns.pairplot(da, hue = 'Prgnt', kind='reg', palette= 'spring_r')

#----Palette-------
#*********seaborn has six variations of its default color palette: 
    #deep, muted, pastel, bright, dark, colorbind ***************
sns.pairplot(da, hue = 'Prgnt', kind='reg', palette= 'dark')
sns.pairplot(da, hue = 'Prgnt', kind='reg', palette= 'pastel')

#---------HEATMAP--------------
hm = sns.heatmap(data = cs2m)
hm = sns.heatmap(data = cs2m, cmap = 'tab10')
hm = sns.heatmap(data = cs2m, cmap = 'tab20_r')

#-----value shown heatmap----------
hm = sns.heatmap(data = cs2m, annot = True) # good for correl matrix
