import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split 
import pickle

#Reading the data 
df = pd.read_csv(r"./data/Telco-Customer-Churn.csv")
target=df.Churn
df.drop(["customerID"], axis=1, inplace=True)

df.SeniorCitizen=df.SeniorCitizen.apply(lambda x: str(x))


#Splitting features based on type for further preprocessing
binary_feat = df.nunique()[df.nunique() == 2].keys().tolist()
numeric_feat = [col for col in df.select_dtypes([np.float64,np.int64]).columns.tolist() if col not in binary_feat]
categorical_feat = [ col for col in df.select_dtypes('object').columns.to_list() if col not in binary_feat + numeric_feat ]

#Removing the target variable
binary_feat.remove('Churn')

# Creating a column transformer
# There were no outliers in TotalCharges, MonthlyCharges and tenure, so used MinMaxScaler instead of StandardScaler
preprocessing = CT(
                    transformers=[
                        ('numeric_scaling', MinMaxScaler(), numeric_feat),
                        ('categorical_dummies', OneHotEncoder(drop='first'), categorical_feat),
                        ('binary_binarizing', OneHotEncoder(drop='if_binary'), binary_feat)
                                 ],
                        remainder='drop',
                    n_jobs=-1
                    )



#Creating train and test splits
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Churn'],axis=1),target, test_size=0.2, random_state=123456)

#Fitting the transformer
X_train = pd.DataFrame(preprocessing.fit_transform(X_train))
X_test = pd.DataFrame(preprocessing.transform(X_test))

#Label encoding.
y_train = y_train.map({'Yes':1,'No':0})
y_test = y_test.map({'Yes':1,'No':0})


#Preparing datasets for use in modelling phase
train = pd.concat([X_train, pd.DataFrame(y_train.values)], axis=1)
test = pd.concat([X_test, pd.DataFrame(y_test.values)], axis=1)


#Also preparing the entire dataset for use by production model
dataset_X=df.drop(['Churn'],axis=1)
dataset_y=df['Churn']
dataset_y = dataset_y.map({'Yes':1,'No':0})

#The final transformer to be used by the production model in deployment phase
prep=preprocessing.fit(dataset_X)

#Transforming the entire dataset based on the fit
dataset_X = pd.DataFrame(prep.transform(dataset_X))

#The final dataset to be used for production model
dataset = pd.concat([dataset_X,dataset_y],axis=1)

#Writing out the preprocessing transformer
with open(r'./bin/preprocessing.pkl','wb') as r:
    pickle.dump(prep,r) 


#Writing out the files for use in modelling
train.to_csv(r'./data/train.csv', index=False)
test.to_csv(r'./data/test.csv',index=False)
dataset.to_csv(r'./data/dataset.csv',index=False)
