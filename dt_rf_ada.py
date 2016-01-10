import os
import json 
import numpy as np

# load data
with open('train.json') as data_file:
  data=json.load(data_file)

# process data
n=len(data)
nsplit=n/2

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def normalize_input(X):
    return (X.T / np.sum(X, axis=1)).T

X=[x['ingredients'] for x in data]
X= [dict(zip(x,np.ones(len(x)))) for x in X]

vec = DictVectorizer()
X= vec.fit_transform(X).toarray()
X= normalize_input(X)
X = X.astype(np.float32)

feature_names = np.array(vec.feature_names_)

lbl = LabelEncoder()

y= [y['cuisine'] for y in data]
y= lbl.fit_transform(y).astype(np.int32)

label_names = lbl.classes_ 

# Decision Tree Classifier
from sklearn import tree
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

accuracy_score_list = []
kfolds = KFold(39774,5)
for train_index,test_index in kfolds:
    print("Train:",len(train_index),"TEST:",len(test_index))
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    clf = tree.DecisionTreeClassifier(max_depth = 200).fit(Xtrain,ytrain)
    accuracy_score_list.append(accuracy_score(ytest,clf.predict(Xtest)))
    os.system('afplay /System/Library/Sounds/Glass.aiff')

    # 1-Level Decision Tree
    # 0.20070904558534006
    
    # 10-Level Decision Tree
    # 0.38422067681121197    
    
    # 100-Level Decision Tree
    # 0.58130952221028309 
    
    # 200-Level Decision Tree
    # 0.5957913502324097


# AdaBoosted Decision Tree Stump Classifier
from sklearn.ensemble import AdaBoostClassifier

accuracy_score_list = []
estimators = [50,100,200]
kfolds = KFold(39774,5)
for train_index,test_index in kfolds:
    print("Train:",len(train_index),"TEST:",len(test_index))
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    accuracy_score_list.append([AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 1),
                             algorithm="SAMME",
                             n_estimators=i).fit(Xtrain,ytrain).score(Xtest,ytest) for i in estimators])
    os.system('afplay /System/Library/Sounds/Glass.aiff')

reformatList = np.array(accuracy_score_list)
means = np.mean(reformatList,axis=0)

	# 50 Estimators
	# 0.302986996094924

	# 100 Estimators
	# 0.337456781269167

	# 200 Estimators
	# 0.394126576652964 

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

accuracy_score_list = []
estimators = [50,100,200]
kfolds = KFold(39774,5)
for train_index,test_index in kfolds:
    print("Train:",len(train_index),"TEST:",len(test_index))
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    accuracy_score_list.append([RandomForestClassifier(n_estimators=i,
                                max_features="sqrt",
                                n_jobs=-1,
                                bootstrap=True,
                                oob_score=True).fit(Xtrain,ytrain).oob_score_ for i in estimators])
    os.system('afplay /System/Library/Sounds/Glass.aiff')

reformatList = np.array(accuracy_score_list)
means = np.mean(reformatList,axis=0)

    # 50 Estimators
    # 0.68767913
        
    # 100 Estimators
    # 0.70144442
        
    # 200 Estimators
    # 0.70838991