#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

#Load dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
df_raw = pd.read_csv(url)

#Transform variables
df_raw['sex_aux'] = df_raw['sex'].apply(lambda x:1 if x == 'female' else 0)
df_raw['smoker_aux'] = df_raw['smoker'].apply(lambda x:1 if x == 'yes' else 0)
df_raw['region_sw'] = df_raw['region'].apply(lambda x:1 if x == 'southwest' else 0)
df_raw['region_nw'] = df_raw['region'].apply(lambda x:1 if x == 'northwest' else 0)
df_raw['region_ne'] = df_raw['region'].apply(lambda x:1 if x == 'northeast' else 0)
df_raw = df_raw.drop(['sex', 'smoker', 'region'], axis = 1)
df_interim = df_raw.copy()

#Split data
X = df_interim[['age', 'bmi', 'children', 'sex_aux', 'smoker_aux', 'region_sw', 'region_nw', 'region_ne']] 
y = df_interim['charges'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)

#Model 1
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

#Model 2
model3 = LinearRegression(fit_intercept= False) 
model3.fit(X_train, y_train) 
y_pred2 = model3.predict(X_test) 

#Model 3
poly_features = PolynomialFeatures(degree = 2) 
X_poly_train = poly_features.fit_transform(X_train) 
X_poly_test = poly_features.fit_transform(X_test)
model4 = model.fit(X_poly_train, y_train)
y_pred3 = model4.predict(X_poly_test) 