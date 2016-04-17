__author__ = 'Jiaxiao Zheng'

import numpy as np
import pandas as pd
import xgboost as xgb
import util
from sklearn import cross_validation, metrics

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

X = train_df.drop(['TARGET', 'ID'], axis = 1).as_matrix()

y = train_df.TARGET.values

#params borrowed from the xgb_lalala script
bestParam = {
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'eta':0.0202048,
    'max_depth':5,
    'subsample':0.6815,
    'colsample_bytree':0.701
}
N = 5
mean_auc = 0.
for i in range(0, N):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
        X, y, test_size = 0.20, random_state = 39 * i
    )
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtest = xgb.DMatrix(X_cv)
    bst = xgb.train(bestParam, dtrain, num_round = 560)
    preds = bst.predict(dtest)
    auc = metrics.roc_auc_score(y_cv, preds)
    print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
    mean_auc += auc

print('Mean Auc is: ' + str(mean_auc/N))
