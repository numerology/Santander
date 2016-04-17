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
X = pd.to_numeric(X, errors = 'coerce').as_matrix()

y = train_df.TARGET.values

#Before run logis regression, need onehotencoding
#heuristic: if n_values > 20 its ordinal otherwise categorical
#    print(selectedX.columns)

d = X.shape[1]

X_ord = []
X_cat = []

for i in range(0,d):
#    print('transforming ' + str(i))
    current_col = X[:, i]
   # print(current_col)

    current_n_values = len(set(current_col))
#    print(current_col.dtype)

    if(current_n_values == 207):
        #whether it is var3
        X_cat.append(preprocess.ranking(current_col))
    else:
        if(current_n_values > 10):
            X_ord.append(current_col)
        else:
            X_cat.append(preprocess.ranking(current_col))
            continue
X_ord = np.array(X_ord).T
#X_ord = sparse.csr_matrix(X_ord)
X_cat = np.array(X_cat).T
# enc = preprocessing.OneHotEncoder()
# X_cat = enc.fit_transform(X_cat)
X = np.hstack([X_ord.astype(float), X_cat])


print(X.shape)



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
    selectedX, selectedX_test, fs = preprocess.model_feature_select(model_running, X_train, y_train, X_cv)

    model_running.fit(selectedX, y_train)
    preds = model_running.predict_proba(selectedX_test)[:, 1]

    auc = metrics.roc_auc_score(y_cv, preds)
    print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
    mean_auc += auc

print('Current AUC under this selection is ' + str(mean_auc / N))
'''
Baseline: with all features: seems like 0.65
Before adding extracted features, auc approx 0.751
After adding n_0 and var38mc and logvar38, its 0.743

Using Extratreeclassifier:
After adding n_0, logvar38, var38mc we boost to 0.7605
Then after 1hotencoding: worse, approx 0.746

Using logis self as selector:
After adding n0 logvar38 var38mc, without 1hotencoding: 0.770
After 1hotencoding: 0.758
Seems 1hotencoding screw things up, conjecture: many of the cat features are actually ordinal,
Solution: downgrade the threshold to 3.

Without 1hotencoding: 0.773
with 1hotencoding: still worse

What if we only input the ordinal features:
worse, 0.74
Adding var3?
Still worse 0.74
Change cat threshold back to 10:
does not help, but the variance is small, around 0.758

So the best way maybe treating all the features as ordinal, without 1hot encoding:
0.773

Here's the plan:
Running select from model using logistic regression as selector, pick individual good features.
Then using the Miroslaw's code to find good interaction terms

'''