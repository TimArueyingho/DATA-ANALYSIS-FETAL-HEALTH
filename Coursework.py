#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
<<<<<<< HEAD
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler

data = pd.read_excel("fetal_health.xlsx")

data.describe()
data.isnull().values.any() 
#There are no null values

fig = plt.pie(data.fetal_health.value_counts(), labels = ["Normal", "Suspect", "Pathological"])

data.corr()
plt.figure(figsize=(20,10))
#plot heat map
map = sn.heatmap(data.corr().sort_values(by=["fetal_health"], ascending=False),annot=True, cmap="Blues")
#Prolonged decel, abnormal short term variability, % time with abnormal var have moderate correlation with fetal health
#accerlerations has low-to-mod neg correlation with fetal health

normal = data[data.fetal_health == 1]
suspect = data[data.fetal_health ==2]
pathological = data[data.fetal_health ==3]

fig, ax = plt.subplots(figsize = (15,5))
df = pd.concat([normal[["prolongued_decelerations"]].rename(columns={"prolongued_decelerations":"1"}),suspect[["prolongued_decelerations"]].rename(columns={"prolongued_decelerations":"2"}), pathological[["prolongued_decelerations"]].rename(columns={"prolongued_decelerations":"3"})], axis=1)
df.plot(kind="box", y = ["1","2","3"], label = ["normal","suspect","patholgical"],  ax = ax )
ax.set(title = 'Comparing prolonged decelerations by class',  ylabel="Prolonged_decelerations")
plt.show()

violin1 = sn.violinplot(x=data.fetal_health, y=data.prolongued_decelerations, data=data)
#Shows the distribution of prolongued decelerations is wider in pathological outcomes than in normal or suspect outcomes

violin2 = sn.violinplot(x=data.fetal_health, y=data.abnormal_short_term_variability, data=data)
#The median abonormal short_term_var is lower than that in suspect and pathological outcomes.

violin3 = sn.violinplot(x=data.fetal_health, y=data.percentage_of_time_with_abnormal_long_term_variability, data=data)

violin4 = sn.violinplot(x=data.fetal_health, y=data.accelerations, data=data)
#Suspect and pathological tended to have fewer accerlerations than normal

#Unbalanced testing
normal = data[data.fetal_health == 1]
suspect = data[data.fetal_health == 2]
pathological = data[data.fetal_health == 3]


#To create a representative train/test split, split each of the above classes into respective train and test sets.
normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=41)
suspect_train, suspect_test = train_test_split(suspect, test_size=0.2, random_state=41)
pathological_train, pathological_test = train_test_split(pathological, test_size=0.2, random_state=41)

#Then combine these into one training set and one test set.
training_data = pd.concat([normal_train, suspect_train, pathological_train], ignore_index=True)
test_data = pd.concat([normal_test, suspect_test, pathological_test], ignore_index=True)

#Then create the train and test X matrices and the train and test y arrays
X_train, X_test, y_train, y_test = training_data.drop(columns = ["fetal_health"]), test_data.drop(columns = ["fetal_health"]), training_data.fetal_health, test_data.fetal_health

models = {"KNeighborsClassifier":KNeighborsClassifier(n_neighbors=3), "GaussianNaiveBayes": GaussianNB(), "DecisionTree": DecisionTreeClassifier(max_depth=30), "RandomForest": RandomForestClassifier()}

#Unscaled
for named, model in models.items():
    mod = model.fit(X_train, y_train)
    ypred = mod.predict(X_test)
    accuracy = accuracy_score(y_test, ypred)
    precision = precision_score(y_test, ypred, average="macro")
    recall = recall_score(y_test, ypred, average="macro")
    f1score = f1_score(y_test, ypred, average="macro")
    print(f"Unscaled data: Accuracy for {named}: {accuracy}")
    print(f"Unscaled data: Precision for {named}: {precision}")
    print(f"Unscaled data: Recall for {named}: {recall}")
    print(f"Unscaled data: F1 score for {named}: {f1score}")
    
    p_hat = mod.predict_proba(X_test)
    roc = roc_auc_score(y_test, p_hat, multi_class="ovr")
    print(f"Unscaled data: ROC AUC score for {named}: {roc}\n")
    
#scaled data
sc = RobustScaler()
scaled_X_train, scaled_X_test = sc.fit_transform(X_train), sc.fit_transform(X_test)
scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns, index=X_train.index)
scaled_X_test = pd.DataFrame(scaled_X_test, columns = X_test.columns, index=X_test.index)

for named, model in models.items():
    mod = model.fit(scaled_X_train, y_train)
    ypred = mod.predict(scaled_X_test)
    accuracy = accuracy_score(y_test, ypred)
    precision = precision_score(y_test, ypred, average="micro")
    recall = recall_score(y_test, ypred, average="micro")
    f1score = f1_score(y_test, ypred, average="micro")
    print(f"Scaled data: Accuracy for {named}: {accuracy}")
    print(f"Sscaled data: Precision for {named}: {precision}")
    print(f"Scaled data: Recall for {named}: {recall}")
    print(f"Sscaled data: F1 score for {named}: {f1score}")
    p_hat = mod.predict_proba(X_test)
    roc = roc_auc_score(y_test, p_hat, multi_class="ovr")
    print(f"Scaled data: ROC AUC score for {named}: {roc}\n")

    
=======
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




>>>>>>> a0208eb30c57745e948c8bc63eb958a91d4ac136
