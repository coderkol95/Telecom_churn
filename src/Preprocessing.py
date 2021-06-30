import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split 
import pickle
 
df = pd.read_csv(r"./data/Telco-Customer-Churn.csv")
df.drop(["customerID"], axis=1, inplace=True)

df.SeniorCitizen=df.SeniorCitizen.apply(lambda x: str(x))

binary_feat = df.nunique()[df.nunique() == 2].keys().tolist()
numeric_feat = [col for col in df.select_dtypes([np.float64,np.int64]).columns.tolist() if col not in binary_feat]
categorical_feat = [ col for col in df.select_dtypes('object').columns.to_list() if col not in binary_feat + numeric_feat ]

binary_feat.remove('Churn')
target=df.Churn

'''
nos=["No phone service", "No"]
nos.extend( ["No internet service"]*6)
nos.extend(['Two year','Mailed check'])
'''

preprocessing = CT(
                    transformers=[
                        ('numeric_scaling', MinMaxScaler(), numeric_feat),
                        ('categorical_dummies', OneHotEncoder(drop='first'), categorical_feat),
                        ('binary_binarizing', OneHotEncoder(drop='if_binary'), binary_feat)
                                 ],
                        remainder='drop',
                    n_jobs=-1
                    )

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Churn'],axis=1),target, test_size=0.2, random_state=123456)

X_train = pd.DataFrame(preprocessing.fit_transform(X_train))
X_test = pd.DataFrame(preprocessing.transform(X_test))

y_train = y_train.map({'Yes':1,'No':0})
y_test = y_test.map({'Yes':1,'No':0})

train = pd.concat([X_train, pd.DataFrame(y_train.values)], axis=1)
test = pd.concat([X_test, pd.DataFrame(y_test.values)], axis=1)

dataset_X=df.drop(['Churn'],axis=1)
dataset_y=df['Churn']
dataset_y = dataset_y.map({'Yes':1,'No':0})

prep=preprocessing.fit(dataset_X)
dataset_X = pd.DataFrame(prep.transform(dataset_X))

dataset = pd.concat([dataset_X,dataset_y],axis=1)

with open(r'./bin/preprocessing.pkl','wb') as r:
    pickle.dump(prep,r) 

train.to_csv(r'./data/train.csv', index=False)
test.to_csv(r'./data/test.csv',index=False)
dataset.to_csv(r'./data/dataset.csv',index=False)
