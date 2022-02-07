
"""
Data set: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
Random Forest Algo
@author: HIMANSHU

"""
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("F:/Image Processing using python/Data/data.csv")
print(df.describe().T)  # Values need to be normalized before fitting

print(df.isnull().sum())

df = df.rename(columns = {'diagnosis': 'Label'})
print(df.dtypes)

# Before processing understand the dataset

sns.countplot(x ='Label', data = df)
sns.distplot(df['radius_mean'], kde = False)

print(df.corr())
corrMatrix = df.corr()
fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(corrMatrix, annot = False, linewidth =.5, ax = ax)

# Replace categorical values with numbers

df['Label'].value_counts()
categories = {'B': 1, 'M': 2}
df['Label']= df['Label'].replace(categories)

# Now define the dependent variable need to be predicted (Label)
Y = df['Label'].values

X = df.drop(labels =["Label", "id", "Unnamed: 32"], axis=1)


feature_list = list(X.columns)

# Now normalize the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state =42)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 25, random_state = 42)

# Train the model on training data
model.fit(X_train, Y_train)


prediction = model.predict(X_test)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(Y_test, prediction))


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, prediction)
print(cm)

#Print individual accuracy values for each class, based on the confusion matrix
print("Benign = ", cm[0,0] / (cm[0,0]+cm[1,0]))
print("Malignant = ",   cm[1,1] / (cm[0,1]+cm[1,1]))


#importances = list(model_RF.feature_importances_)
feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)













