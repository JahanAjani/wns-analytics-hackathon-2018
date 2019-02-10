#!/usr/bin/env python

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import ensemble as em
import sklearn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# Load the data
train_org = pd.read_csv("../input/train.csv")
test_org = pd.read_csv("../input/test.csv")

train_org.head(5)

train_y = train_org['is_promoted']
print(train_y.shape)
#train_x = train_org.drop(labels=['is_promoted'], axis=1)

#print(train_x.head(2))
train_x = train_org.drop(labels=['is_promoted'], axis=1)
train_x = train_x.drop(labels=['employee_id'], axis=1)
test = test_org


train_x['gender'] = train_x['gender'].map( {'f': 0, 'm': 1} ).astype(int)

test['gender'] = test_org['gender'].map( {'f': 0, 'm': 1} ).astype(int)

train_x['education'].isnull().sum()

#train_x['education'] = train_x['education'].map( {"Master's & above": 0, "Bachelor's": 1} ).astype(int)
#train_x = train_x.dropna()

test['recruitment_channel'] = test['recruitment_channel'].map({'sourcing':2, 'other':3, 'referred':1})
test['region'] = test['region'].map({'region_7':7, 'region_22':22, 'region_19':19, 'region_23':23, 'region_26':26,
       'region_2':2, 'region_20':20, 'region_34':34, 'region_1':1, 'region_4':4,
       'region_29':29, 'region_31':31, 'region_15':15, 'region_14':14, 'region_11':11,
       'region_5':5, 'region_28':28, 'region_17':17, 'region_13':13, 'region_16':16,
       'region_25':25, 'region_10':10, 'region_27':27, 'region_30':30, 'region_12':12,
       'region_21':21, 'region_8':8, 'region_32':32, 'region_6':6, 'region_33':33,
       'region_24':24, 'region_3':3, 'region_9':9, 'region_18':18})
test['department']=test['department'].map( {'Sales & Marketing':0, 'Operations':1, 'Technology':2, 'Analytics':3,'R&D':4, 'Procurement':5, 'Finance':6, 'HR':7, 'Legal':8} ).astype(int)
test['education'] = test['education'].fillna('not req')
test['education'] = test['education'].map( {"Master's & above": 0, "Bachelor's": 1, 'Below Secondary': 2, "not req":3} ).astype(int)

train_x['recruitment_channel'].unique()
train_x['recruitment_channel'] = train_x['recruitment_channel'].map({'sourcing':2, 'other':3, 'referred':1})

train_x['region'].unique()
train_x['region'] = train_x['region'].map({'region_7':7, 'region_22':22, 'region_19':19, 'region_23':23, 'region_26':26,
       'region_2':2, 'region_20':20, 'region_34':34, 'region_1':1, 'region_4':4,
       'region_29':29, 'region_31':31, 'region_15':15, 'region_14':14, 'region_11':11,
       'region_5':5, 'region_28':28, 'region_17':17, 'region_13':13, 'region_16':16,
       'region_25':25, 'region_10':10, 'region_27':27, 'region_30':30, 'region_12':12,
       'region_21':21, 'region_8':8, 'region_32':32, 'region_6':6, 'region_33':33,
       'region_24':24, 'region_3':3, 'region_9':9, 'region_18':18})

train_x['region'].unique()

train_x['department'].unique()
train_x['department']=train_x['department'].map( {'Sales & Marketing':0, 'Operations':1, 'Technology':2, 'Analytics':3,'R&D':4, 'Procurement':5, 'Finance':6, 'HR':7, 'Legal':8} ).astype(int)

train_x['education'] = train_x['education'].fillna('not req')
train_x['education'].unique()
train_x['education'] = train_x['education'].map( {"Master's & above": 0, "Bachelor's": 1, 'Below Secondary': 2, "not req":3} ).astype(int)

test['education'].unique()
#test['education'] = test['education'].map( {"Master's & above": 0, "Bachelor's": 1, 'Below Secondary': 2} ).astype(int)
#train_x["education"]

xtrain,xtest,ytrain,ytest = train_test_split(train_x,train_y,train_size=.8,random_state=1234)

model3 = CatBoostClassifier(iterations=1200, learning_rate=0.02, depth=7, loss_function='Logloss', eval_metric='F1')

model3.fit(xtrain, ytrain, use_best_model=True, verbose=True, eval_set=(xtest,ytest))

model3.score(train_x[:3],train_y[:3])

sub_samp = pd.read_csv("../input/sample_submission.csv")

sub_samp.head(5)

test_y = test['employee_id']
print(test_y.shape)
test = test.drop(labels=['employee_id'], axis=1)

predt = model3.predict(test).astype('int')
#pd.DataFrame([test_y,predt],columns=['employee_id','is_promoted'])
#test[['id','target']].to_csv('catboost_submission.csv', index=False

StackingSubmission = pd.DataFrame({ 'employee_id': test_y,
                            'is_promoted': predt })
StackingSubmission['is_promoted'].unique()
StackingSubmission.to_csv("submission.csv", index=False)

