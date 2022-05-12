#!/usr/bin/env python
# coding: utf-8

# In[42]:

import pandas as pd
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from yellowbrick.features import FeatureImportances
#from pandas_profiling import ProfileReport


data = pd.read_excel(r"C:\Users\User\Documents\Digital Health\Data analysis for health\Coursework\fetal_health.xlsx")

describe = data.describe()
data.isnull().values.any() #There are no null values

fig = plt.pie(data.fetal_health.value_counts(), labels = ["Normal", "Suspect", "Pathological"], autopct="%1.1f%%", colors=["darkturquoise","cornflowerblue","mediumpurple"] )

data.corr()
plt.figure(figsize=(20,10))
#plot heat map
map = sn.heatmap(data.corr().sort_values(by=["fetal_health"], ascending=False),annot=True, cmap="Blues")
#Prolonged decelerations, abnormal short term variability, % time with abnormal var have moderate correlation with fetal health
#accerlerations has low-to-mod neg correlation with fetal health

normal = data[data.fetal_health == 1]
suspect = data[data.fetal_health ==2]
pathological = data[data.fetal_health ==3]

fig, ax = plt.subplots(figsize = (15,5))
df = pd.concat([normal[["prolongued_decelerations"]].rename(columns={"prolongued_decelerations":"1"}),suspect[["prolongued_decelerations"]].rename(columns={"prolongued_decelerations":"2"}), pathological[["prolongued_decelerations"]].rename(columns={"prolongued_decelerations":"3"})], axis=1)
df.plot(kind="box", y = ["1","2","3"], label = ["normal","suspect","patholgical"],  ax = ax )
ax.set(title = 'Comparing prolonged decelerations by class',  ylabel="Prolonged_decelerations")
plt.show()

plt.figure()
violin1 = sn.violinplot(x=data.fetal_health, y=data.prolongued_decelerations, data=data)
#Shows the distribution of prolongued decelerations is wider in pathological outcomes than in normal or suspect outcomes. 
#Median is similar for normal and suspect class = 0.000, and slightly higher for pathological = 0.001
plt.figure()
violin2 = sn.violinplot(x=data.fetal_health, y=data.abnormal_short_term_variability, data=data)
#The median abonormal short_term_var is lower in normal class (40) than that in suspect and pathological outcomes (~60).
plt.figure()
violin3 = sn.violinplot(x=data.fetal_health, y=data.percentage_of_time_with_abnormal_long_term_variability, data=data)
plt.figure()
#Large density of percentage time with abnormal long term variability is around 0 for class 1 but the median is very similar for normal and pathological classes (med=0). 
#For suspect the median is around 30
violin4 = sn.violinplot(x=data.fetal_health, y=data.accelerations, data=data)
#Suspect and pathological tended to have fewer accerlerations than normal - median 0.000 - with data focussed here. 
#Normal class has a wider range of accelerations with median roughly 0.025.

# PROF = ProfileReport(data, title='Profiling Report of data', minimal=True, progress_bar=False,
#                      missing_diagrams={
#                          'heatmap': False,
#                          'dendrogram': False,
#                      })
# PROF.to_file(output_file="Fetal Health Profile.html")
# PROF


plt.figure(figsize=(25, 15))
for all, column in enumerate(data.columns):
    plt.subplot(4, 6, all + 1)
    sn.histplot(data=data[column])
    plt.title(column)

plt.savefig('Histograms of all variables.png')
plt.tight_layout()
plt.show()

# In[68]:

plt.figure(figsize=(25, 15))

for all, column in enumerate(data.columns):
    plt.subplot(4, 6, all + 1)
    sn.boxplot(data=data[column])
    plt.title(column)

plt.savefig('Boxplots of all variables.png')
plt.tight_layout()
plt.show()

# In[71]:

plt.figure(figsize=(25, 15))

plt.pie(data['fetal_health'].value_counts(), autopct='%.2f%%', labels=['Normal', 'Suspects', 'Pathological'],
        colors=sn.color_palette('seismic'))

plt.savefig('Piechart.png')
plt.title('Distibution of cases')
plt.show()

#Next, we test 2 models with unbalanced, underbalanced, overbalanced and hybrid balanced data to choose the best strategy going forward

#Unbalanced testing

def model_tests(X_train, X_test,y_train,y_test):
    models = {"KNeighborsClassifier":KNeighborsClassifier(), "GaussianNaiveBayes": GaussianNB(), "DecisionTree": DecisionTreeClassifier(random_state=20), "RandomForest": RandomForestClassifier(random_state=20)}
    for named, model in models.items():
        mod = model.fit(X_train, y_train)
        ypred = mod.predict(X_test)
        accuracy = round((accuracy_score(y_test, ypred)),2)
        precision = round((precision_score(y_test, ypred, average="micro")),2)
        recall = round((recall_score(y_test, ypred, average="micro")),2)
        f1score = round((f1_score(y_test, ypred, average="micro")),2)
            
        print(f"Accuracy for {named}: {accuracy}")
        print(f"Micro precision for {named}: {precision}")
        print(f"Micro Recall for {named}: {recall}")
        print(f"Micro F1 score for {named}: {f1score}")
        
        p_hat = mod.predict_proba(X_test)
        roc = round((roc_auc_score(y_test, p_hat, multi_class="ovr")),2)
        print(f"ROC AUC score for {named}: {roc}\n")
        
        #precision, recall, f1 scores per class
        class_report = classification_report(y_test, ypred, labels=[1,2,3], target_names=["Normal", "Suspect", "Pathological"])
        print(f"Classification report by class for {named}")
        print(class_report)
        
                                  
#UNBALANCED DATA MODELLING
#To create a representative train/test split, split each of the above classes into respective train and test sets.
normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=41)
suspect_train, suspect_test = train_test_split(suspect, test_size=0.2, random_state=41)
pathological_train, pathological_test = train_test_split(pathological, test_size=0.2, random_state=41)

#Then combine these into one training set and one test set.
training_data = pd.concat([normal_train, suspect_train, pathological_train], ignore_index=True)
test_data = pd.concat([normal_test, suspect_test, pathological_test], ignore_index=True)

#Then create the train and test X matrices and the train and test y arrays
X_train, X_test, y_train, y_test = training_data.drop(columns = ["fetal_health"]), test_data.drop(columns = ["fetal_health"]), training_data.fetal_health, test_data.fetal_health

print(f"\n >>>Unbalanced dataset, initial tests")
model_tests(X_train, X_test, y_train, y_test)

# UNDERBALANCED STRATEGY

US= RandomUnderSampler(random_state= 12, sampling_strategy="auto")
X, y = US.fit_resample(data.drop(columns=["fetal_health"]), data["fetal_health"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print(f"\n>>>Under sampling initial results")
model_tests(X_train, X_test, y_train, y_test)


#HYBRID BALANCED STRATEGY
OS = RandomOverSampler(random_state = 12 ,sampling_strategy={2:800, 3:800})
X, y = OS.fit_resample(data.drop(columns=["fetal_health"]), data["fetal_health"])

US = RandomUnderSampler(sampling_strategy={1:800})
X, y = US.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(f"\n>>>HYBRID sampling initial results")
model_tests(X_train, X_test, y_train, y_test)

#OVERBALANCED STRATEGY - brings class 2 and 3 up to same number of samples as 1
OS = RandomOverSampler(sampling_strategy={2:1655, 3:1655}, random_state=40)
# fit and apply the transform
X, y = OS.fit_resample(data.drop(columns=["fetal_health"]), data["fetal_health"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print(f"\n>>>Over sampling initial results")
model_tests(X_train, X_test, y_train, y_test)

# FEATURE ENGINEERING

#FIRST SCALE THE X DATA 
def scale(X_train, X_test):
    sc = RobustScaler()
    scaled_X_train, scaled_X_test = sc.fit_transform(X_train), sc.fit_transform(X_test)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns, index=X_train.index)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns = X_test.columns, index=X_test.index)
    return scaled_X_train, scaled_X_test

scaled_X_train, scaled_X_test = scale(X_train, X_test)
print(f"\n >>> Results for scaled data")
model_tests(scaled_X_train, scaled_X_test, y_train, y_test)
#This improves model performance for KNN and GNB but not DTR and RFR actually!


# SECOND ENGINEERING PCA 
pca = PCA(n_components=12)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_train))
explained_variance_ratio = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(explained_variance_ratio)
#
# Create the visualization plot
#
plt.figure(dpi=128)
bars = ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8','PC9','PC10','PC11','PC12')
x_pos = np.arange(len(bars))

plt.bar(range(0,len(explained_variance_ratio)), explained_variance_ratio, align='center', label='Individual')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative',color = 'orange')
plt.ylabel('Explained variance ratio')
plt.xticks(x_pos, bars)
plt.legend(loc='best')
plt.title("Explained Variance Ratio by each principal component")
plt.show()

#8 Primary components give EVR greater than 0.8.

def component_data(X_train, X_test):
    sc = StandardScaler()
    scaled_X_train, scaled_X_test = sc.fit_transform(X_train), sc.fit_transform(X_test)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns, index=X_train.index)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns = X_test.columns, index=X_test.index)
    pca = PCA(n_components=8)
    X_trainpc = pca.fit_transform(scaled_X_train)
    X_testpc = pca.fit_transform(scaled_X_test)
    return X_trainpc, X_testpc

X_trainpc, X_testpc = component_data(X_train, X_test)
print(f"\n>>> PCA results: 8 components with standard scaler data")
model_tests(X_trainpc, X_testpc, y_train, y_test)

# FEATURE SELECTION

# In[66]
models = {"KNeighborsClassifier":KNeighborsClassifier(), "GaussianNaiveBayes": GaussianNB(), "RandomForestClassifier": RandomForestClassifier()}
for named, model in models.items(): 
    accuracyscores = []
    precisionscores = []
    recallscores= []
    for i in range(5, 16):
    #sequential feature selection for knn and gnb, selecting 15 best features
        sfs = SequentialFeatureSelector(model, n_features_to_select=i)
        sfs.fit(X, y)

    #transforming the training set and test set to be restricted to the selected number of features
        Xtraintemp = sfs.transform(X_train)
        Xtesttemp = sfs.transform(X_test)

    #fitting a knn model using the dataset containing the subset of features and making predictions
        model.fit(Xtraintemp, y_train)
        ypred = model.predict(Xtesttemp)

    #generating scores 
        accuracyscores.append(accuracy_score(y_test, ypred))
        precisionscores.append(precision_score(y_test, ypred, average = 'micro'))
        recallscores.append(recall_score(y_test, ypred, average = 'micro'))
    
    plt.figure()
    x = list(range(5,16))
    plt.plot(x,accuracyscores, label = 'accuracy scores')
    plt.plot(x,precisionscores, label = 'precision scores')
    plt.plot(x, recallscores, label = 'recall scores')
    plt.xlabel('number of features used')
    plt.xticks(x)
    plt.title(f'Changes features used in {named}')
    plt.legend()
    
# SECOND FEATURE SELECTION - trees
import sklearn.tree as skt

model = DecisionTreeClassifier()
x = model.fit(X_train, y_train)
plt.figure(figsize = (60,20))
new = skt.plot_tree(x, filled=True)
plt.show()

#Create a dictionary of features and their importances 
dictionary = {}

for columns, values in sorted (zip(X_train.columns, model.feature_importances_), key = lambda x: x[1], reverse = True):
    dictionary [columns] = values

importance_dataframe = pd.DataFrame({'Feature':dictionary.keys(),'Importance':dictionary.values()}, index=X_train.columns)

#Visualise
fig , ax = plt.subplots(figsize=(10,8))
tree = FeatureImportances(model)
tree.fit(X_train, y_train)
#fig.savefig('Important features.png',dpi=300)
plt.show()

#Data not scaled as scaled data performed worse except for KNN

five_best_features = X[['histogram_mean', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
    'abnormal_short_term_variability', 'accelerations']]

fivefeat_train, fivefeat_test, y_train, y_test = train_test_split(five_best_features, y, test_size = 0.2,random_state = 40)
print(f"\n >>> Results for 5 best features, NO PCA")
model_tests(fivefeat_train, fivefeat_test, y_train, y_test)

ten_best_features = X[['histogram_mean', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
   'abnormal_short_term_variability', 'accelerations', 	'prolongued_decelerations', 'histogram_mode', 'histogram_width','histogram_max', 'histogram_min']]

tenfeat_train, tenfeat_test, y_train, y_test = train_test_split(ten_best_features,y, test_size = 0.2, random_state = 40)

print(f"\n >>> Results for 10 best features, NO PCA")
model_tests(tenfeat_train, tenfeat_test, y_train, y_test)
# Top 10 features better than 5 feature

#FEATURE SELECTION ON PCA
model = DecisionTreeClassifier()
x = model.fit(X_trainpc, y_train)
plt.figure(figsize = (60,20))
new = skt.plot_tree(x, filled=True)
plt.show()

#Create a dictionary of features and their importances 
dictionary = {}

pc_dataframe_Xtrain = pd.DataFrame(data=X_trainpc, columns=["PC1", "PC2","PC3","PC4","PC5","PC6","PC7","PC8"])
pc_dataframe_Xtest = pd.DataFrame(data=X_testpc, columns=["PC1", "PC2","PC3","PC4","PC5","PC6","PC7","PC8"])

for columns, values in sorted (zip(pc_dataframe_Xtrain.columns, model.feature_importances_), key = lambda x: x[1], reverse = True):
    dictionary [columns] = values

pca_importance_dataframe = pd.DataFrame({'Feature':dictionary.keys(),'Importance':dictionary.values()})
#Visualise


three_best_PCAfeatures_train = pc_dataframe_Xtrain[["PC1", "PC2","PC3"]]
three_best_PCAfeatures_test = pc_dataframe_Xtest[["PC1", "PC2","PC3"]]

model_tests(three_best_PCAfeatures_train, three_best_PCAfeatures_test, y_train, y_test)
#Feature selection on 3 PCs worse

five_best_PCAfeatures_train = pc_dataframe_Xtrain[["PC1", "PC2","PC3","PC5","PC6"]]
five_best_PCAfeatures_test = pc_dataframe_Xtest[["PC1", "PC2","PC3","PC5","PC6"]]

model_tests(five_best_PCAfeatures_train, five_best_PCAfeatures_test, y_train, y_test)
#on 5 best PCs even worse

###
RF = RandomForestClassifier()
rfc = RF.fit(X_train, y_train)

#convert it to a dataframe
rf_importance_dataframe = pd.DataFrame({'Feature importance':RF.feature_importances_}, index=X_train.columns)
rf_importance_dataframe = rf_importance_dataframe.sort_values(by=["Feature importance"], axis=0, ascending=False)

fig , ax = plt.subplots(figsize=(10,8))
tree = FeatureImportances(rfc)
tree.fit(X_train, y_train)
#fig.savefig('Random forest.png',dpi=300)
plt.show()

#select the five best features
rf_5best_features = X[['abnormal_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
   'histogram_mean', 'histogram_median','accelerations']]

#split new dataset into training and testing sets
rf_5besttrain, rf_5besttest, y_train, y_test = train_test_split(rf_5best_features,y, test_size = 0.2, shuffle = True, random_state = 40)

print(f"\n>>> Results using RF's best 5 features, No PCA")
model_tests(rf_5besttrain, rf_5besttest, y_train, y_test)
#DTC slightly better with DTC's top 10 features worse than DTC top 5


rf_10best_features = X[['abnormal_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
   'histogram_mean', 'histogram_median','accelerations', 'mean_value_of_short_term_variability', 'mean_value_of_long_term_variability', 'histogram_mode','baseline value','prolongued_decelerations']]

#split new dataset into training and testing sets
rf_10besttrain, rf_10besttest, y_train, y_test = train_test_split(rf_10best_features,y, test_size = 0.2, shuffle = True, random_state = 40)

print(f"\n>>> Results using RF's best 5 features, No PCA")
model_tests(rf_10besttrain, rf_10besttest, y_train, y_test)

#RFR top 10 better for KNN, GNB, similar for DTC except ROC and slightly loweer than top 10 from DTC importances