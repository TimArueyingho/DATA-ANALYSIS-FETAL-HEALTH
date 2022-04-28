import pandas as pd
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

    