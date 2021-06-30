import pandas as pd
import numpy as np

import sklearn
print(sklearn.__version__)

# Importing the dataset
dataset = pd.read_csv('G:/Deep_Learning/ANN/Churn_Modelling.csv')

dataset.describe()
dataset.head()

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames
X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

#see count of out columns
y.value_counts()

##HoldOut Validation Approach
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
result = model.score(X_test, y_test)
print(result)

##K-Flod Cross Validation
from sklearn.model_selection import KFold
model = DecisionTreeClassifier()
kfold_valid = KFold(10)

from sklearn.model_selection import cross_val_score
result = cross_val_score(model, X, y, cv=kfold_valid)
print(result)
print(np.mean(result))

##Stratified K-Flod Cross Validation
## we use when we have Im-Balnced data set 
from sklearn.model_selection import stratifiedKFold
skfold = stratifiedKFold(n_split=5)
model = DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score
result = cross_val_score(model, X, y, cv=skfold)
print(result)
print(np.mean(result))

#Leave One Out Cross Validation
from sklearn.model_selection import LeaveOneOut
model = DecisionTreeClassifier()
leave_valid = LeaveOneOut()
result = cross_val_score(model, X, y, cv=leave_valid)
print(result)
print(np.mean(result))
