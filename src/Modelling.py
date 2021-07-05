import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import AdaBoostClassifier
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll import scope
from functools import partial
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
def optimize(X,y,params):

    s=StratifiedKFold(n_splits=5)
    mod = AdaBoostClassifier(**params)
    score=cross_val_score(mod,X,y,scoring='f1',cv=s).mean()    
    return -score.mean()

#Hyperparameters to be tuned
param_space_gb =  {
        'learning_rate': hp.uniform('lr',0.01,1),
        'n_estimators': scope.int(hp.quniform('trees',50,1000,1))
}

#To store the results of the trials
trials=Trials()

#The optimization function
optimization_function= partial(optimize,X,y)

#The final assessment
result = fmin(
    fn=optimization_function,
    max_evals=20,
    space=param_space_gb,
    trials=trials,
    algo=tpe.suggest
)

print(result)
#model = AdaBoostClassifier(learning_rate= result['lr'], n_estimators= int(result['trees']))
model = AdaBoostClassifier(learning_rate= 0.877214687401072, n_estimators=62)                   #These were obtained as the best results. Running it repeatedly causes computation overhead.

model.fit(X,y)

with open(r'./bin/model.pkl','wb') as r:
    pickle.dump(model, r)