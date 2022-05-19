#!/usr/bin/env python
# coding: utf-8

# In[42]:

import pandas as pd
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
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
from matplotlib import rcParams as rcp

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

#shows differing ranges of features
fig, ax = plt.subplots(figsize=(40,20))
sn.boxplot(data=data.drop(columns=["fetal_health"]))
ax.tick_params(axis='both', which='major', labelsize=30)
plt.xticks(rotation=90)
plt.ylabel("range", fontsize=40)
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
    models = {"KNeighborsClassifier":KNeighborsClassifier(), "GaussianNaiveBayes": GaussianNB(), "DecisionTree": DecisionTreeClassifier(random_state=10), "RandomForest": RandomForestClassifier(random_state=10)}
    for named, model in models.items():
        mod = model.fit(X_train, y_train)
        ypred = mod.predict(X_test)
        accuracy = round((accuracy_score(y_test, ypred)),2)
        precision = round((precision_score(y_test, ypred, average="macro")),2)
        recall = round((recall_score(y_test, ypred, average="macro")),2)
        f1score = round((f1_score(y_test, ypred, average="macro")),2)
            
        print(f"Accuracy for {named}: {accuracy}")
        print(f"Macro precision for {named}: {precision}")
        print(f"Macro Recall for {named}: {recall}")
        print(f"Macro F1 score for {named}: {f1score}")
        
        # p_hat = mod.predict_proba(X_test)
        # roc = round((roc_auc_score(y_test, p_hat, multi_class="ovr")),2)
        # print(f"ROC AUC score for {named}: {roc}\n")
        
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

models = {"KNeighborsClassifier":KNeighborsClassifier(), "GaussianNaiveBayes": GaussianNB(), "DecisionTree": DecisionTreeClassifier(random_state=10), "RandomForest": RandomForestClassifier(random_state=10)}
for named, model in models.items():
    mod = model.fit(X_train, y_train)
    ypred = mod.predict(X_test)
    accuracy = round((accuracy_score(y_test, ypred)),2)
    precision = round((precision_score(y_test, ypred, average="weighted")),2)
    recall = round((recall_score(y_test, ypred, average="weighted")),2)
    f1score = round((f1_score(y_test, ypred, average="weighted")),2)
        
    print(f"Accuracy for {named}: {accuracy}")
    print(f"Weighted precision for {named}: {precision}")
    print(f"Weighted Recall for {named}: {recall}")
    print(f"Weighted F1 score for {named}: {f1score}")
    
    # p_hat = mod.predict_proba(X_test)
    # roc = round((roc_auc_score(y_test, p_hat, multi_class="ovr")),2)
    # print(f"ROC AUC score for {named}: {roc}\n")
    
    #precision, recall, f1 scores per class
    class_report = classification_report(y_test, ypred, labels=[1,2,3], target_names=["Normal", "Suspect", "Pathological"])
    print(f"Classification report by class for {named}")
    print(class_report)


# UNDERBALANCED STRATEGY

US= RandomUnderSampler(random_state= 12, sampling_strategy="auto")
X, y = US.fit_resample(data.drop(columns=["fetal_health"]), data["fetal_health"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print(f"\n>>>Under sampling initial results")
model_tests(X_train, X_test, y_train, y_test)


#HYBRID BALANCED STRATEGY
OS = RandomOverSampler(random_state = 12 ,sampling_strategy={2:800, 3:800})
X, y = OS.fit_resample(data.drop(columns=["fetal_health"]), data["fetal_health"])

US = RandomUnderSampler(random_state = 12, sampling_strategy={1:800})
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
    scaled_X_train = sc.fit_transform(X_train)
    scaled_X_test = sc.transform(X_test)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns, index=X_train.index)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns = X_test.columns, index=X_test.index)
    return scaled_X_train, scaled_X_test

scaled_X_train, scaled_X_test = scale(X_train, X_test)
print(f"\n >>> Results for scaled data")
model_tests(scaled_X_train, scaled_X_test, y_train, y_test)
#This improves model performance for KNN and GNB but not DTR and RFR actually!

#Shows original data when scaled
fig, ax = plt.subplots(figsize=(40,20))
sn.boxplot(data=RobustScaler().fit_transform(data.drop(columns=["fetal_health"])))
ax.tick_params(axis='both', which='major', labelsize=30)
plt.xticks(ticks=range(0,21), labels=data.drop(columns=["fetal_health"]).columns, rotation=90)
plt.ylabel("range", fontsize=40)
plt.show()

# FEATURE SELECTION

# In[66]
models = {"KNeighborsClassifier":KNeighborsClassifier(), "GaussianNaiveBayes": GaussianNB(), "RandomForestClassifier": RandomForestClassifier(random_state=10)}
for named, model in models.items(): 
    accuracyscores = []
    precisionscores = []
    recallscores= []
    f1scores = []
    for i in range(5, 21):
    #sequential feature selection for knn and gnb, selecting i best features
        sfs = SequentialFeatureSelector(model, n_features_to_select=i)
        sfs.fit(scaled_X_train, y_train)

    #transforming the training set and test set to be restricted to the selected number of features
        Xtraintemp = sfs.transform(X_train)
        Xtesttemp = sfs.transform(X_test)

    #fitting a knn model using the dataset containing the subset of features and making predictions
        model.fit(Xtraintemp, y_train)
        ypred = model.predict(Xtesttemp)

    #generating scores 
        accuracyscores.append(accuracy_score(y_test, ypred))
        precisionscores.append(precision_score(y_test, ypred, average = 'macro'))
        recallscores.append(recall_score(y_test, ypred, average = 'macro'))
        f1scores.append(f1_score(y_test, ypred, average="macro"))
        print(f"For {model} with {i} features, selected features are: {sfs.get_feature_names_out()}")
    print(f"\n")
    plt.figure()
    x = list(range(5,21))
    plt.plot(x,accuracyscores, label = 'accuracy scores', color="green")
    plt.plot(x,precisionscores, label = 'precision scores', color="blue")
    plt.plot(x, recallscores, label = 'recall scores',color="red")
    plt.plot(x, f1scores, label = 'F1 scores',color="purple")
    plt.xlabel('number of features used')
    plt.xticks(x)
    plt.title(f'Changes features used in {named}')
    plt.legend()
    
# SECOND FEATURE SELECTION - trees
import sklearn.tree as skt

model = DecisionTreeClassifier(random_state=33)
x = model.fit(X_train, y_train)
plt.figure(figsize = (60,20))
new = skt.plot_tree(x, filled=True)
plt.show()

#Create a dictionary of features and their importances 
dictionary = {}

for columns, values in sorted (zip(X_train.columns, model.feature_importances_), key = lambda x: x[1], reverse = True):
    dictionary [columns] = values

importance_dataframe = pd.DataFrame({'Feature':dictionary.keys(),'Importance':dictionary.values()})

#Visualise
fig , ax = plt.subplots(figsize=(10,8))
tree = FeatureImportances(model)
tree.fit(X_train, y_train)
#fig.savefig('Important features.png',dpi=300)
plt.show()

#Data not scaled as scaled data performed worse except for KNN

models = {"KNeighborsClassifier":KNeighborsClassifier(), "GaussianNaiveBayes": GaussianNB(), "RandomForestClassifier": RandomForestClassifier(random_state=10)}
for named, model in models.items(): 
   
    accuracyscores = []
    precisionscores = []
    recallscores= []
    f1scores = []
    
    for i in range(len(dictionary.keys())):
        feature_names = list(dictionary.keys())[:i+1]
        best_X = X[feature_names]  

        best_X_train, best_X_test, y_train, y_test = train_test_split(best_X, y, test_size=0.2, random_state = 40)
        
        model.fit(best_X_train, y_train)
        ypred = model.predict(best_X_test)

    #generating scores 
        accuracyscores.append(accuracy_score(y_test, ypred))
        precisionscores.append(precision_score(y_test, ypred, average = 'macro'))
        recallscores.append(recall_score(y_test, ypred, average = 'macro'))
        f1scores.append(f1_score(y_test, ypred, average="macro"))
    
    plt.figure()
    x = range(1, len(dictionary.keys())+1)
    plt.plot(x,accuracyscores, label = 'accuracy scores', color="green")
    plt.plot(x,precisionscores, label = 'precision scores', color="blue")
    plt.plot(x, recallscores, label = 'recall scores',color="red")
    plt.plot(x, f1scores, label = 'F1 scores',color="purple")
    plt.xlabel('number of features used')
    plt.xticks(x)
    plt.title(f'{named}: Best features as determined by DecisionTree')
    plt.legend()
    

five_best_features = X[['histogram_mean', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
    'abnormal_short_term_variability', 'accelerations']]

fivefeat_train, fivefeat_test, y_train, y_test = train_test_split(five_best_features, y, test_size = 0.2,random_state = 40)
print(f"\n >>> Results for 5 best features, NO PCA")
model_tests(fivefeat_train, fivefeat_test, y_train, y_test)

ten_best_features = X[['histogram_mean', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
   'abnormal_short_term_variability', 'accelerations', 	'prolongued_decelerations', 'histogram_median', 'histogram_mode','baseline value', 'fetal_movement']]

tenfeat_train, tenfeat_test, y_train, y_test = train_test_split(ten_best_features,y, test_size = 0.2, random_state = 40)

print(f"\n >>> Results for 10 best features, NO PCA")
model_tests(tenfeat_train, tenfeat_test, y_train, y_test)
# Top 10 features better than 5 feature

#FEATURE SELECTION USING RF

###
RF = RandomForestClassifier(random_state=33)
rfc = RF.fit(X_train, y_train)

dictionary = {}

for columns, values in sorted (zip(X_train.columns, rfc.feature_importances_), key = lambda x: x[1], reverse = True):
    dictionary [columns] = values

#convert it to a dataframe
rf_importance_dataframe = pd.DataFrame({"Feature names": dictionary.keys(),'Feature importance':dictionary.values()})

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

print(f"\n>>> Results using RF's best 10 features, No PCA")
model_tests(rf_10besttrain, rf_10besttest, y_train, y_test)

#RFR top 10 better for KNN, GNB, similar for DTC except ROC and slightly loweer than top 10 from DTC importances

models = {"KNeighborsClassifier":KNeighborsClassifier(), "GaussianNaiveBayes": GaussianNB(), "RandomForestClassifier": RandomForestClassifier(random_state=10)}
for named, model in models.items(): 
   
    accuracyscores = []
    precisionscores = []
    recallscores= []
    f1scores = []
    
    for i in range(len(dictionary.keys())):
        feature_names = list(dictionary.keys())[:i+1]
        best_X = X[feature_names]  

        best_X_train, best_X_test, y_train, y_test = train_test_split(best_X, y, test_size=0.2, random_state = 40)
        
        model.fit(best_X_train, y_train)
        ypred = model.predict(best_X_test)

    #generating scores 
        accuracyscores.append(accuracy_score(y_test, ypred))
        precisionscores.append(precision_score(y_test, ypred, average = 'macro'))
        recallscores.append(recall_score(y_test, ypred, average = 'macro'))
        f1scores.append(f1_score(y_test, ypred, average="macro"))
    
    plt.figure()
    x = range(1, len(dictionary.keys())+1)
    plt.plot(x,accuracyscores, label = 'accuracy scores', color="green")
    plt.plot(x,precisionscores, label = 'precision scores', color="blue")
    plt.plot(x, recallscores, label = 'recall scores',color="red")
    plt.plot(x, f1scores, label = 'F1 scores',color="purple")
    plt.xlabel('number of features used')
    plt.xticks(x)
    plt.title(f'{named}: Best features as determined by RandomForest Importances')
    plt.legend()



# Agreed preprocessing step

def preprocess (data):
    data = data.copy()
   
    X = data[data.columns.to_list()[:-1]] #.............. #predicting this/target
    y = data['fetal_health'] 


    OS = RandomOverSampler(sampling_strategy={2:1655, 3:1655}, random_state=40)
    X,y = OS.fit_resample(X,y)


    main_data = X
    targets = y

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=10)
    
    #instantiate scaler
    scaler = RobustScaler()
    #fit training data of X
    scaler.fit(X_train)
    
    X_train = pd.DataFrame (scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_test = pd.DataFrame (scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    #we would return the split data so we can work on them
    return  X_train, X_test, y_train, y_test

X_train, X_test, y_train,y_test = preprocess(data)

## Hyperparameter tuning for KNN
#we will use the cross_validate function to assess accuracy scores across a range of k for knn
#number of neighbours is an important hyperparameter for knn. It is the basis for how the model makes predictions.
#taking too few neighbours will mean our model does not fit to our data well enough,
#while too many neighbours will result in overfitting.
#for each value of k, we will cross-validate on 5 splits and take the mean accuracy for train and validation
max_k = 30

#we will assess models using accuracy scores, and so begin by initialising variables to store these scores
trainacc = []
valacc = []

#we use a loop to assess the performance of the model on each split of K-fold 
for k in range(max_k):    
    
    knn = KNeighborsClassifier(n_neighbors=k+1)
    
    scores = cross_validate(knn, X_train, y_train, cv = 5, scoring =('accuracy'), return_train_score = True)
    
    trainacc.append(scores['train_score'].mean())
    valacc.append(scores['test_score'].mean())

#we can visualise this output in a graph
x = range(1, max_k+1)
plt.plot(x, trainacc, label='Training accuracy')
plt.plot(x, valacc, label='Validation accuracy')
plt.legend()
plt.xlabel('k')
plt.ylabel('accuracy score')

# Using gridsearch

n_neighbors = list(range(1,11))
weights = ['uniform', 'distance']
leaf_size = list(range(1,5))
p = [1,2]

params_knn = dict(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, p=p)

knn_grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = params_knn, scoring = 'accuracy')
knn_grid_search.fit(X_train, y_train)
knn_y_pred = knn_grid_search.predict(X_test)

print(knn_grid_search.best_params_)
print(f"\n >>> Results for optimised KNN")

accuracy = round((accuracy_score(y_test, knn_y_pred)),2)
precision = round((precision_score(y_test, knn_y_pred, average="weighted")),2)
recall = round((recall_score(y_test, knn_y_pred, average="weighted")),2)
f1score = round((f1_score(y_test, knn_y_pred, average="weighted")),2)
    
print(f"Accuracy for {named}: {accuracy}")
print(f"Weighted precision for {named}: {precision}")
print(f"Weighted Recall for {named}: {recall}")
print(f"Weighted F1 score for {named}: {f1score}")

class_report = classification_report(y_test, ypred, labels=[1,2,3], target_names=["Normal", "Suspect", "Pathological"])
print(f"Classification report by class for {named}")
print(class_report)

print("--------------------")
#Hyperparameter tuning with GNB
#The only hyperparameter in gnb is var_smoothing- default value is  10^−9
#conduct the grid search in the "logspace"

params_gnb= {"var_smoothing":[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15] }

#grid search was used to tune the hyperparameters

gnb_grid_search=GridSearchCV(estimator=GaussianNB().fit(X_train, y_train),param_grid=params_gnb,scoring='accuracy',verbose=1, cv=3, error_score='raise')
#fit gridsearch to model
gnb_grid_search.fit(X_train,y_train)
#find the best var_smoothing
print(gnb_grid_search.best_params_)

best_gnb=GaussianNB(var_smoothing=1e-10)
#fit the best parameter 
best_gnb_model=best_gnb.fit(X_train, y_train)
gnb_pred= best_gnb_model.predict(X_test)

print(f"\n -------------------\nResults for optimised GNB")
accuracy = round((accuracy_score(y_test, gnb_pred)),2)
precision = round((precision_score(y_test, gnb_pred, average="weighted")),2)
recall = round((recall_score(y_test, gnb_pred, average="weighted")),2)
f1score = round((f1_score(y_test, gnb_pred, average="weighted")),2)
    
print(f"Accuracy for {named}: {accuracy}")
print(f"Weighted precision for {named}: {precision}")
print(f"Weighted Recall for {named}: {recall}")
print(f"Weighted F1 score for {named}: {f1score}")

class_report = classification_report(y_test, ypred, labels=[1,2,3], target_names=["Normal", "Suspect", "Pathological"])
print(f"Classification report by class for {named}")
print(class_report)

# print(f"Accuracy for tuned GNB model : {accuracy_score (y_test, best_y_pred)}") 
# print(f"Precision for tuned GNB model: {precision_score (y_test, best_y_pred, average= 'micro')}")
# print(f"Recall for tuned GNB model : {recall_score(y_test, best_y_pred, average= 'micro')}")

#accuracy before 0.7541666666666667 vs after 0.7774420946626385
#precision before 0.7541666666666667 vs after 0.0.77744209466263857
#Recall  before 0.7541666666666667 vs after 0.7774420946626385

#2% increase post tuning

# RF hyperparm tuning
rf = RandomForestClassifier(random_state=10)

# use iterable parameters, else it gave me an error

rf_params = {
    'min_samples_split': [1,2,3,4],
    'min_samples_leaf': [1,2,3],
    'min_impurity_decrease': [0.0,0.1,0.2,0.3,0.4,0.5],
    'max_features':["sqrt","log2"],
    'n_estimators': [25, 50, 100],
    'criterion': ["gini","entropy","log_loss"],
    'max_samples':[0.5,1.0],
    
}

rf_grid_search = GridSearchCV(estimator=rf, 
                           param_grid=rf_params, 
                           cv=3, n_jobs=-1, verbose=1, scoring = "accuracy")

rf_grid_search.fit(X_train, y_train)
print("Best RF parameters:", rf_grid_search.best_params_)

rf_model= RandomForestClassifier(min_impurity_decrease = 0.0, min_samples_leaf = 1,random_state = 10, min_samples_split= 3)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
#pred_y
 
print("----------\nRF final results")
accuracy = round((accuracy_score(y_test, rf_pred)),2)
precision = round((precision_score(y_test, rf_pred, average="weighted")),2)
recall = round((recall_score(y_test, rf_pred, average="weighted")),2)
f1score = round((f1_score(y_test, rf_pred, average="weighted")),2)
    
print(f"Accuracy for {named}: {accuracy}")
print(f"Weighted precision for {named}: {precision}")
print(f"Weighted Recall for {named}: {recall}")
print(f"Weighted F1 score for {named}: {f1score}")

class_report = classification_report(y_test, ypred, labels=[1,2,3], target_names=["Normal", "Suspect", "Pathological"])
print(f"Classification report by class for {named}")
print(class_report)

#Cross validation to compare model performances
scoresknn = []
scoresgnb = []
scoresrf = []

X = RobustScaler().fit_transform(X)
cv = KFold(n_splits=10, shuffle=True)
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn_grid_search.fit(X_train, y_train)
    knn_y_pred = knn_grid_search.predict(X_test)
    best_gnb_model=best_gnb.fit(X_train, y_train)
    gnb_pred= best_gnb_model.predict(X_test)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    scoresknn.append(accuracy_score(y_test,knn_y_pred))
    scoresgnb.append(accuracy_score(y_test,gnb_pred))
    scoresrf.append(accuracy_score(y_test, rf_pred))

print(f'KNN: mean={np.mean(scoresknn)}, sd={np.std(scoresknn)}')
print(f'GNB: mean={np.mean(scoresgnb)},sd={np.std(scoresgnb)}')
print(f'RF: mean={np.mean(scoresrf)}, sd={np.std(scoresrf)}')

box_plot_data=[scoresknn,scoresgnb,scoresrf]
plt.boxplot(box_plot_data,labels=['KNN','GNB',"RF"])
plt.show()
