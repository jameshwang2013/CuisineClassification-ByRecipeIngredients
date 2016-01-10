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

print "Load successful"

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

### Plot learning curves
#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# fit models
#X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.01, test_size=0.01)
#y_train = y_train[0:397]
#y_test = y_test[0:397]
X_train, X_test = X[:nsplit], X[nsplit:]
y_train, y_test = y[:nsplit], y[nsplit:]

key_param = 200

dt = tree.DecisionTreeClassifier(max_depth = key_param)
dt_fitted = dt.fit(X_train,y_train)
dt_score = dt_fitted.score(X_test,y_test)

ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 1),algorithm="SAMME",n_estimators=key_param)
ada_fitted = ada.fit(X_train,y_train)
ada_score = ada_fitted.score(X_test,y_test)

ada_real = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 1),algorithm="SAMME.R",n_estimators=key_param)
ada_fitted_real = ada_real.fit(X_train,y_train)
ada_score_real = ada_fitted_real.score(X_test,y_test)

rf = RandomForestClassifier(n_estimators=key_param,max_features="sqrt",n_jobs=-1,bootstrap=True,oob_score=True)
rf_fitted = rf.fit(X_train,y_train)
rf_score = rf_fitted.oob_score_

# set up plot
ada_acc_scores = np.zeros((key_param,))
for i, y_pred in enumerate(ada_fitted.staged_predict(X_test)):
    ada_acc_scores[i] = accuracy_score(y_pred, y_test)
							
ada_real_scores = np.zeros((key_param,))
for i, y_pred in enumerate(ada_fitted_real.staged_predict(X_test)):
    ada_real_scores[i] = accuracy_score(y_pred, y_test)

ensemble_clfs = [("RandomForestClassifier, max_features='sqrt'", 
			  RandomForestClassifier(oob_score=True,max_features="sqrt"))]

from collections import OrderedDict
oob_accuracy = OrderedDict((label, []) for label, _ in ensemble_clfs)
for label, clf in ensemble_clfs:
    for i in range(key_param):
        clf.set_params(n_estimators=i+1)
        clf.fit(X_train, y_train)
        oob_accuracy[label].append((i, clf.oob_score_))

print "Algs setup successful"
os.system('afplay /System/Library/Sounds/Glass.aiff')

# plot
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, key_param], [dt_score] * 2, 'k-',
        label='Decision Tree Test Accuracy')
								
ax.plot(np.arange(key_param) + 1, ada_acc_scores,
        label='AdaBoost-SAMME Test Accuracy',
        color='red')

print "Ada-SAMME plot successful"

ax.plot(np.arange(key_param) + 1, ada_real_scores,
        label='AdaBoost-SAMME.R Test Accuracy',
        color='blue')

print "Ada-SAMME.R plot successful"	
							
for label, clf_err in oob_accuracy.items():
    xs, ys = zip(*clf_err)
    ax.plot(xs, ys, label=label,color='green')

print "RF plot successful"

ax.set_ylim((0.0, 1.0))
ax.set_xlabel('n_estimators')
ax.set_ylabel('accuracy')

print "Axes setup successful"

#leg = ax.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(0.7)

print "Legend setup successful"

plt.show()
os.system('afplay /System/Library/Sounds/Glass.aiff')