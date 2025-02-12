# This script performs ensemble stacking classification analysis to predict activity types. It combines multiple base classifiers into a meta-classifier to enhance predictive accuracy. The script generates comprehensive evaluation metrics (e.g., accuracy, precision, recall, F1-score) to assess the model’s classification performance.

import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy import mean
from numpy import std
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Extracting the data from dataset(File1,2,3)

df = pd.read_csv('File1.csv')
#dataset.drop_duplicates(inplace=True)
df3_X = df.drop(columns=['Activity Type'])
df3_Y = df['Activity Type']

# finding the number of each activity class in the dataset
cv = KFold(n_splits=5, shuffle=True, random_state=42)
frequency = {}
# iterating over the list
for item in df3_Y:
   # checking the element in dictionary
   if item in frequency:
      # incrementing the count
      frequency[item] += 1
   else:
      # initializing the count
      frequency[item] = 1

# printing the frequency
print(frequency)
c=collections.Counter(df3_Y)
most_frequent, the_count= c.most_common(4)[0]
most_frequent2, the_count2  = c.most_common(4)[1]
most_frequent3, the_count3  = c.most_common(4)[2]
most_frequent4, the_count4  = c.most_common(4)[3]
# Ensure a balanced representation of each activity type 
Remaining__number = 1127
Remaining__number2 = 0

remain = float((the_count  - Remaining__number) / the_count)
remain2 = float((the_count2  - Remaining__number) / the_count2)
remain3 = float((the_count3  - Remaining__number) / the_count3)
remain4 = float((the_count4  - Remaining__number2) / the_count4)
df= df.drop(df.loc[df['Activity Type']==most_frequent].sample(frac= remain).index)
df= df.drop(df.loc[df['Activity Type']==most_frequent2].sample(frac= remain2).index)
df= df.drop(df.loc[df['Activity Type']==most_frequent3].sample(frac= remain3).index)
df= df.drop(df.loc[df['Activity Type']==most_frequent4].sample(frac= remain4).index)
df.to_csv('remove.csv', index=False)
dataset = pd.read_csv('remove.csv')
df3_X1 = dataset.drop(columns=['Activity Type'])
df3_Y1 = dataset['Activity Type']

# preforming enesemble stacking
lab = preprocessing.LabelEncoder()
df3_Y1 = lab.fit_transform(df3_Y1)

X_train,X_test,y_train,y_test = train_test_split(df3_X1,df3_Y1, test_size=0.25, random_state=None)


estimators = [
      ('lr', LogisticRegression(max_iter=10000, random_state=42)),
#     ('svc', make_pipeline(StandardScaler(),
#                        SVC(kernel='poly', random_state=42))),
     ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42))
]

final_estimator = LogisticRegression(max_iter=10000, random_state=42)

stack = StackingClassifier(
    estimators=estimators, final_estimator=final_estimator#,stack_method='predict_proba',cv=kfold
)

final_model = stack.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

#Caculating the metrics
scores = cross_val_score(final_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=stack.classes_)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=stack.classes_)

cm_display.plot()
plt.savefig('Confusion2_matrix.png')
          
