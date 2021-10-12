import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import recall_score, precision_score,accuracy_score,f1_score
import pickle

"""

Modelling:

Different machine learning models were trained on the train data and tested on the test data.
The top three models' hyperparameteres were tuned to obtain the best performance.
AdaBoost performed the best.

Here AdaBoost is being trained on the whole dataset for deployment.

"""

#Reading the dataset
dataset = pd.read_csv(r"./data/dataset.csv")

#Splitting the dataset into features and target
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#The function to be optimized by the HyperOpt
def optimize(params):
    
    """
    Objectives
    1. Initialize model with received params
    2. For the 5 splits, fit it on the train and score it on the test
    3. Calculate the mean F1 score across the splits and return it 
    
    """  
    s=StratifiedKFold(n_splits=5)
    mod = AdaBoostClassifier(**params)
    score=cross_val_score(mod,X,y,scoring='f1',cv=s).mean()    
    return -score.mean()

#Hyperparameters to be tuned
param_space_gb =  {
        'learning_rate': hp.uniform('learning_rate',0.01,1),
        'n_estimators': scope.int(hp.quniform('n_estimators',10,500,1))
}

#To initialize trials
trials=Trials()

#The optimization function
def score_hyperparams(params):
    score=optimize(params)
    return {'loss':score, 'status':STATUS_OK}

#The final assessment
result = fmin(
    fn=score_hyperparams,
    max_evals=50,
    space=param_space_gb,
    trials=trials,
    algo=tpe.suggest
)

print(result)

model = AdaBoostClassifier(learning_rate = result['learning_rate'], n_estimators=int(result['n_estimators'])) 
# model = AdaBoostClassifier(learning_rate = 0.888624248987685, n_estimators=56)   
# f1 score: 0.6039

model.fit(X,y)

with open(r'./bin/model.pkl','wb') as r:
    pickle.dump(model, r)
