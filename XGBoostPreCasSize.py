# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:30:37 2019

@author: XMM
"""

import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

def get_X_y(file):
    df_train = pd.read_csv(file, sep=",")
    X_train = df_train.iloc[:,:-2]
    y_train = df_train.cascade_unique_size
    return X_train, y_train


X_train, y_train = get_X_y("features/imeline.train.righ_cascade_features_list_25.csv")

X_test, y_test = get_X_y("features/imeline.validation.righ_cascade_features_list_25.csv")

my_imputer = Imputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)

my_model = XGBRegressor()

my_model.fit(X_train, y_train, verbose=False)

predictions = my_model.predict(X_test)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))