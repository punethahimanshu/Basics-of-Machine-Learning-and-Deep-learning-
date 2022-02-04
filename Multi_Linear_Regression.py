
"""
@author: HIMANSHU

#Dataset link:
https://cdn.scribbr.com/wp-content/uploads//2020/02/heart.data_.zip?_ga=2.217642335.893016210.1598387608-409916526.1598387608

"""
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv("F:/Image Processing using python/Data/heart.data/heart.data.csv")

print(df.head())

df = df.drop('Unnamed: 0', axis =1)

# Understanding the data with few plots

sns.lmplot(x = 'biking', y = 'heart.disease', data = df)
sns.lmplot(x = 'smoking', y = 'heart.disease', data = df)

x_df = df.drop('heart.disease', axis =1)
y_df = df['heart.disease']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_df, y_df, test_size=0.3, random_state =42)

from sklearn import linear_model

# Now create linear regression object

model = linear_model.LinearRegression()

model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))

# Prediction Test

prediction_test = model.predict(X_test)
print(Y_test, prediction_test)
print("Mean sq. errror between Y_test and predicted =", np.mean(prediction_test-Y_test)**2)


#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
print(model.coef_, model.intercept_)


