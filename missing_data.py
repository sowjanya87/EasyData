# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

  
X = dataset.iloc[:, :-1].values
Y= dataset.iloc[:, 3].values

# Taking care of missing data

dataset=pd.read_csv('Data.csv')
from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values =np.nan,strategy='mean')
si=si.fit(X[:,1:3]) #FIT IMPUTER TO MATRIX X
X[:,1:3]=si.transform(X[:,1:3]) # replace missing data with mean of the columns

## Encoding categorical data for X
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0]) # to assign encoded values to categorical value

#To prevent categorical values are greater than other we create dummy variables
#dataset[:,0]=labelencoder_dataset.fit_transform(dataset[:,0])
from sklearn.preprocessing import OneHotEncoder
onehotencoder_x = OneHotEncoder(categorical_features=[0])
X=onehotencoder_x.fit_transform(X).toarray()

## Encoding categorical data Y
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state = 0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test) #dont need to fit as it is already fitted to training set

## Do we need to  scale fit and transform the dummy variables
# it depends upon interpretation  
# Do we need to feature scaling to Y...no we dont need to this one this is a 
#classification problem but for linear regresion we will need to 

