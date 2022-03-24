import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
fish_data = pd.read_csv("D:\Weather_data.csv")

print(fish_data.describe())
#fish_data=fish_data.head()
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#weatherFeatures = ["_pressurem"]
weatherFeatures = ["_dewptm","_hum","_pressurem"]

X = fish_data[weatherFeatures]
y = fish_data._tempm
print(X)
#print(y)
X_scaled  = preprocessing.scale(X) 
poly = PolynomialFeatures(3) 
X_final = poly.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.10, random_state=42) 
from sklearn import linear_model
regr = linear_model.Ridge(alpha = 0.5) 
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print(y_pred)
from sklearn.metrics import mean_squared_error, r2_score

# Display Model Intercept
print("Intercept: ", regr.intercept_)

# Display Model coefficients
print('Coefficients: \n', regr.coef_ )


print('Mean squared error: %.3f' % mean_squared_error(y_test,  y_pred))


# r2_score (Coefficient of determination) is a great evaluation metric - read more below
print('Coefficient of determination: %.3f' % r2_score(y_test, y_pred))