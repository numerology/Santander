__author__ = 'Jiaxiao Zheng'

import numpy as np
import pandas as pd
import util
import preprocess
from scipy import sparse
from sklearn import linear_model, preprocessing, cross_validation, metrics
from sklearn.ensemble import ExtraTreesClassifier

train_df = pd.read_csv('data/train.csv', header = 0)
test_df = pd.read_csv('data/test.csv', header = 0)

'''
remove duplicate and constant
'''

remove = []
for col in train_df.columns:
    if(train_df[col].std() == 0):
        remove.append(col)

train_df.drop(remove, axis = 1, inplace = True)
test_df.drop(remove, axis = 1, inplace = True)

remove = []
cols = train_df.columns
for i in range(len(cols) - 1):
    v = train_df[cols[i]].values
    for j in range(i + 1, len(cols)):
        if np.array_equal(v, train_df[cols[j]].values):
            remove.append(cols[j])

train_df.drop(remove, axis = 1, inplace = True)
test_df.drop(remove, axis = 1, inplace = True)

test_id = test_df.ID
test = test_df.drop(['ID'], axis = 1)

train_df = train_df.replace(-999999, 2)
test_df = test_df.replace(-999999, 2)
X = train_df.iloc[:, 0:-1]
X_test = test_df.iloc[:, 1:]
train_df['n0'] = (X == 0).sum(axis = 1)
train_df['var38mc'] = np.isclose(train_df.var38, 117310.979016)
train_df['logvar38'] = train_df.loc[~train_df['var38mc'], 'var38'].map(np.log)
train_df.loc[train_df['var38mc'], 'logvar38'] = 0
test_df['n0'] = (X_test == 0).sum(axis = 1)

X = train_df.drop(['TARGET', 'ID'], axis = 1)
y = train_df.TARGET.values


#feature selection, try diff models
SEED = 39
#To prevent overfitting, we cannot run the feature selection on the whole training set
N = 5
mean_auc = 0.
model_running = linear_model.LogisticRegression(penalty = 'l2', C = 0.1, intercept_scaling = 10, class_weight = 'balanced')
for i in range(N):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
        X, y, test_size=.20,
        random_state=i*SEED)
  #  print(X_train)
  #  print(y_train)
  #
  #   if len(X_train.shape) == 1:
  #       X_train = np.matrix(X_train)
  #       X_train = np.transpose(X_train)
  #       X_cv = np.matrix(X_cv)
  #       X_cv = np.transpose(X_cv)
    model = ExtraTreesClassifier(random_state= i * SEED)
    selectedX, selectedX_test, fs = preprocess.model_feature_select(model, X_train, y_train, X_cv)

    model_running.fit(X, y)
    preds = model_running.predict_proba(X_cv)[:, 1]

    auc = metrics.roc_auc_score(y_cv, preds)
    print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
    mean_auc += auc

print('Current AUC under this selection is ' + str(mean_auc / N))
'''
Baseline: with all features: seems like 0.65
Before adding extracted features, auc approx 0.751
After adding n_0 and var38mc and logvar38, its 0.743

'''