import pandas as pd
import numpy as np

target = 'result'

df = pd.read_csv('FullShotsData.csv')

print(df.dtypes)

print(df.describe())

print(df.shape)

#df['Scored'] = df.result=='Goal'

import seaborn as sns

sns.histplot(df['result'])

baseline = df[target].value_counts(normalize = True)

print(f' baseline model : {baseline[0]:.2f}')

k = df['lastAction'].value_counts()

df_edited = df.drop(labels=['id','minute','player','player_id','year','match_id','h_team','a_team','h_goals','a_goals','date','player_assisted','lastAction'],axis=1)
'''
from xgboost import XGBClassifier
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(
    OrdinalEncoder(),  
    SimpleImputer(), 
    XGBClassifier(n_estimators=200
                  , random_state=2
                  , n_jobs=-1
                  , max_depth=7
                  , learning_rate=0.2
                 )
)

from sklearn.model_selection import train_test_split

feature = df_edited.drop(labels='result',axis=1).columns

X_train, X_test, y_train, y_test = train_test_split(df_edited[feature],df_edited[target], stratify=(df_edited['result']),random_state=(42))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

pipe.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
y_pred = pipe.predict(X_val)
print('검증 정확도: ', accuracy_score(y_val, y_pred))

print(classification_report(y_pred, y_val))
'''