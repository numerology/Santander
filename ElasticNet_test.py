__author__ = 'jxzheng'

import numpy as np
import util
import preprocess
import pandas as pd
from sklearn import cross_validation, metrics, preprocessing
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import ElasticNet
from sklearn.grid_search import GridSearchCV
from scipy import sparse


train_df = pd.read_csv('data/train.csv', header = 0)
test_df = pd.read_csv('data/test.csv', header = 0)
y = train_df.TARGET.values
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
train_df = train_df.drop(['TARGET', 'ID'], axis = 1)
X = train_df
X_test = test_df
train_df['n0'] = (X == 0).sum(axis = 1)
train_df['var38mc'] = np.isclose(train_df.var38, 117310.979016)
train_df['logvar38'] = train_df.loc[~train_df['var38mc'], 'var38'].map(np.log)
train_df.loc[train_df['var38mc'], 'logvar38'] = 0
test_df['n0'] = (X_test == 0).sum(axis = 1)
test_df['var38mc'] = np.isclose(test_df.var38, 117310.979016)
test_df['logvar38'] = test_df.loc[~test_df['var38mc'], 'var38'].map(np.log)
test_df.loc[test_df['var38mc'], 'logvar38'] = 0

X = train_df.as_matrix()
#X = X.as_matrix()
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
        if(current_n_values > 3):
            X_ord.append(current_col)
        else:
            X_cat.append(preprocess.ranking(current_col))
            continue
X_ord = np.array(X_ord).T
#X_ord = sparse.csr_matrix(X_ord)
X_cat = np.array(X_cat).T
# enc = preprocessing.OneHotEncoder()
# X_cat = enc.fit_transform(X_cat)
# X = sparse.hstack([X_ord.astype(float), X_cat])

X = np.hstack([X_ord.astype(float), X_cat])

pca = PCA(n_components=2)

x_train_projected = pca.fit_transform(normalize(X, axis=0))
#x_test_projected = pca.transform(normalize(test_df, axis=0))
#Xtest = test_df
#X.insert(1, 'PCAOne', x_train_projected[:, 0])
X = np.insert(X, 0, x_train_projected[:, 0], axis = 1)
X = np.insert(X, 0, x_train_projected[:, 0], axis = 1)
#X.insert(1, 'PCATwo', x_train_projected[:, 1])
#Xtest.insert(1, 'PCAOne', x_test_projected[:, 0])
#Xtest.insert(1, 'PCATwo', x_test_projected[:, 1])


model = ElasticNet(max_iter = 100, alpha = 0.001, l1_ratio = 0.11, selection = 'random')
'''
first trial
no ranking transform
no one hot encoding
get 0.74852
'''
'''
second trial:
with ranking transformation but no one hot encoding
0.7604

Third trial:
with alpha = 0.1 and l1ratio = 0.2, boost to approx 0.788

4th trial:
with select 90 best according to f_classif: down to 0.78787

5th trial: With one hot encoding, ranking and no pca components
threshold of n_value for categorical vars is 10:
down to 0.753

6th trial: change the threshold of cat to 2: 0.759, not interesting

7th trial, with selectk
'''
# selectK = SelectKBest(f_classif, k=150)
# X = selectK.fit_transform(X, y)


#mean_auc = util.cv_loop(X, y, model, N = 5, default_proba = True)
#print(mean_auc)
params = {'alpha':[0.001, 0.002, 0.003],
		  'l1_ratio':[0.11,0.13,0.15,0.20]
			}
#let's use alpha = 0.001, l1ratio = 0.11, giving us 0.792

clf = GridSearchCV(model, params, scoring = 'roc_auc', verbose = 1)
clf.fit(X, y)
print(clf.best_score_)
print(clf.best_params_)
