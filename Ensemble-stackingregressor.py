# This script performs ensemble stacking regression analysis for i iterations to evaluate the predictive performance of activity values using multiple machine learning models. It systematically trains and evaluates stacked regressors, generating key evaluation metrics (e.g., R², RMSE, MAE) for performance assessment.

import pandas as pd
import numpy as np
import collections
import statistics
import random
from collections import Counter
from random import randint
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_predict

#Extracting the data from dataset

for i in range(20):
    dataset = pd.read_csv('1.csv')
    df_X = dataset.drop(columns=['Activity Value [uM]'])
    df_Y = dataset['Activity Value [uM]']
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    frequency = {}
    # iterating over the list
    for item in df_Y:
   # checking the element in dictionary
       if item in frequency:
      # incrementing the counr
          frequency[item] += 1
       else:
      # initializing the count
          frequency[item] = 1

# printing the frequency
    print(frequency)

    c=collections.Counter(df_Y)
    most_frequent, the_count= c.most_common(5)[0]
    most_frequent2, the_count2  = c.most_common(5)[1]
    most_frequent3, the_count3  = c.most_common(5)[2]
    most_frequent4, the_count4  = c.most_common(5)[3]
    #most_frequent5, the_count5  = c.most_common(5)[4]
    
#spiliting the data in training and test set

    value = random.sample(range(100), 20)
    print(value)

    for n in value:

        X_train,X_test,y_train,y_test = train_test_split(df_X,df_Y, test_size=0.25, random_state=n)

#performing ensemble stacking

        params = {'n_estimators': 500,
         'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
        estimators = [
        ('rf', RandomForestRegressor(n_estimators=100,

         random_state=0)),

        ('svr', make_pipeline(StandardScaler(),

          SVR(kernel='rbf',C=0.6))),
        ('gb', GradientBoostingRegressor(random_state=0))

        ]  

        stack = StackingRegressor(estimators=estimators,final_estimator= RandomForestRegressor(n_estimators=100,

        random_state=0))

        Final_model =  stack.fit(X_train, y_train)

        predictions = Final_model.predict(X_test)
        print("Actual Value test:", y_test)
        print("Predicted Value test:", predictions)
        y_pred = cross_val_predict(Final_model, X_train, y_train, cv=cv)
        print("Actual Value validation:", y_train)
        print("Predicted Value vaidation:", y_pred)
# calculating the metrics
        scores = cross_val_score(Final_model, X_train, y_train, cv = cv)
        print("mean cross validation score: {}".format(np.mean(scores)))
        print(mean_absolute_error(y_test, predictions))
        print(mean_absolute_percentage_error(y_test, predictions))
        print(np.sqrt(mean_squared_error(y_test, predictions)))
        y_bar = y_test.mean()
        ss_tot = ((y_test-y_bar)**2).sum()
        ss_res = ((y_test-predictions)**2).sum()
        r2_score = 1 - (ss_res/ss_tot)
        print(n,'r2_score is',r2_score)
    print('The for loop is complete!')    


