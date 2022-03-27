import pandas as pd
import numpy as np


fetal_health = pd.read_csv("C:/Users/sj21399/OneDrive - University of Bristol/Desktop/DATA ANALYTICS CW/fetal_health.csv")
fetal_health

#Cleaning the data

df = pd.DataFrame(fetal_health)

#Removing Duplicates
df.drop_duplicates()

#check for missing values
df.isnull

#drop missing values
x = df.replace(0.0, np.nan)
fetal_health = x.dropna(axis = 1)
fetal_health

#DATA CLEANED
