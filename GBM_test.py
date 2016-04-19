__author__ = 'Jiaxiao Zheng'

import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

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
test_df = test_df.drop(['ID'], axis = 1)

train_df = train_df.replace(-999999, 2)
test_df = test_df.replace(-999999, 2)
X = train_df.iloc[:, 0:-1]
X_test = test_df.iloc[:, 1:]
train_df['n0'] = (X == 0).sum(axis = 1)
train_df['var38mc'] = np.isclose(train_df.var38, 117310.979016)
train_df['logvar38'] = train_df.loc[~train_df['var38mc'], 'var38'].map(np.log)
train_df.loc[train_df['var38mc'], 'logvar38'] = 0
test_df['n0'] = (X_test == 0).sum(axis = 1)
test_df['var38mc'] = np.isclose(test_df.var38, 117310.979016)
test_df['logvar38'] = test_df.loc[~test_df['var38mc'], 'var38'].map(np.log)
test_df.loc[test_df['var38mc'], 'logvar38'] = 0

X = train_df.drop(['TARGET', 'ID'], axis = 1)
pca = PCA(n_components=2)
print(train_df.shape)
print(test_df.shape)
x_train_projected = pca.fit_transform(normalize(X, axis=0))
x_test_projected = pca.transform(normalize(test_df, axis=0))
Xtest = test_df
X.insert(1, 'PCAOne', x_train_projected[:, 0])
X.insert(1, 'PCATwo', x_train_projected[:, 1])
Xtest.insert(1, 'PCAOne', x_test_projected[:, 0])
Xtest.insert(1, 'PCATwo', x_test_projected[:, 1])
y = train_df.TARGET.values



clfs = GradientBoostingClassifier(loss = 'exponential', learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=20, random_state=39)


# clfs.fit(X, y)
# preds = clfs.predict_proba(X_test_sel)
'''
When kselect = 220
deviance results in auc: 0.8333
exponential results in auc: 0.8343

with pca 2 most significant components:
deviance: 0.8326
exponential: 0.8349

Conclusion, using pca is beneficial, somewhat we may want to keep both model, try to find the best k for features and n_estimators
'''
score_hist = []
N = 5

good_features = set([])
k_start = 50
step = 20
k_end = 400
k = k_start
# while (len(score_hist) < 2 or score_hist[-1] > score_hist[-2]):

#     selectK = SelectKBest(f_classif, k=k)

#     mean_auc = 0.
#     for i in range(N):
#         X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
#             X, y, test_size=.20,
#             random_state=i*39)
#       #  print(X_train)
#       #  print(y_train)
#       #
#       #   if len(X_train.shape) == 1:
#       #       X_train = np.matrix(X_train)
#       #       X_train = np.transpose(X_train)
#       #       X_cv = np.matrix(X_cv)
#       #       X_cv = np.transpose(X_cv)
#         X_train = selectK.fit_transform(X_train, y_train)
#         X_cv = selectK.transform(X_cv)

#         clfs.fit(X_train, y_train)
#         preds = clfs.predict_proba(X_cv)[:, 1]

#         auc = metrics.roc_auc_score(y_cv, preds)
#         print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
#         mean_auc += auc

#     mean_auc = mean_auc/N
#     print('When k = ' + str(k) + ', the auc is: ' + str(mean_auc))
#     score_hist.append(mean_auc)
#     k += step
#     if(k > k_end):
#         break

bestK = 90 #best k is around 90
#find the best n_estimator
score_hist = []
n_start = 100
step = 10
n_end = 200
n_est = n_start
selectK = SelectKBest(f_classif, k=bestK)
selectK.fit(X, y)
X_sel = selectK.transform(X)
while (len(score_hist) < 2 or score_hist[-1] > score_hist[-2]):
    clfs = GradientBoostingClassifier(
        loss = 'exponential',
        learning_rate=0.05,
        subsample=0.5,
        max_depth=6,
        n_estimators=n_est,
        random_state=39
    )
    mean_auc = util.cv_loop(X_sel, y, clfs, 5)
    print('When nEst = ' + str(n_est) + ', the auc is: ' + str(mean_auc))
    score_hist.append(mean_auc)
    n_est += step
    if(n_est > n_end):
        break

#best N-estimator = 100 with auc = 0.83445





