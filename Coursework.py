#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.preprocessing import LabelEncoder
from collections import Counter

fetal_health = pd.read_csv(
    "C:/Users/sj21399/OneDrive - University of Bristol/Desktop/DATA ANALYTICS CW/fetal_health.csv")

# Check for null values
fetal_health.isnull().any

# split into target and input
a = fetal_health[fetal_health.columns.to_list()[:-1]]
b = fetal_health['fetal_health']

# assign label to target variable
b = LabelEncoder().fit_transform(b)


# See distribution of normal, pathological and suspected
def view(b):
    classifier = Counter(b)
    views = []
    for each, all in classifier.items():
        calc = all / len(b) * 100

    # plot the distribution
    fig = plt.bar(classifier.keys(), classifier.values())
    plt.savefig('Unbalanced dataset.png')
    plt.show()
    # fig.(r"C:/Users/sj21399/OneDrive - University of Bristol/Desktop/DATA ANALYTICS CW/ Unbalanced dataset.png")


view(b)
# This chart shows that our dataset is very unbalanced


# In[43]:


# !pip install imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Hybrid Sampling of Data

# define oversampling strategy
OS = RandomOverSampler(sampling_strategy='auto')
# fit and apply the transform
a, b = OS.fit_resample(a, b)

# define undersampling strategy
US = RandomUnderSampler(sampling_strategy='auto')
# fit and apply the transform
a, b = US.fit_resample(a, b)

view(b)

# The dataset has been balanced


# In[51]:


# !pip install pandas-profiling
from pandas_profiling import ProfileReport
import pandas_profiling as pdp

PROF = ProfileReport(fetal_health, title='Profiling Report of data', minimal=True, progress_bar=False,
                     missing_diagrams={
                         'heatmap': False,
                         'dendrogram': False,
                     })
PROF.to_file(output_file="Fetal Health Profile.html")
PROF

# In[57]:


correlation = fetal_health.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sea.heatmap(correlation, vmax=1.0, center=0, fmt='.2f', cmap="seismic",
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
plt.savefig('HeatMap.png')
plt.show()

#

# In[66]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(25, 15))

for all, column in enumerate(fetal_health.columns):
    plt.subplot(4, 6, all + 1)
    sea.histplot(data=fetal_health[column])
    plt.title(column)

plt.savefig('Histograms of all variables.png')
plt.tight_layout()
plt.show()

# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(25, 15))

for all, column in enumerate(fetal_health.columns):
    plt.subplot(4, 6, all + 1)
    sea.boxplot(data=fetal_health[column])
    plt.title(column)

plt.savefig('Boxplots of all variables.png')
plt.tight_layout()
plt.show()

# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(25, 15))

plt.pie(fetal_health['fetal_health'].value_counts(), autopct='%.2f%%', labels=['Normal', 'Suspects', 'Pathological'],
        colors=sea.color_palette('seismic'))

plt.savefig('Piechart.png')
plt.title('Distibution of cases')
plt.show()

# In[ ]:




