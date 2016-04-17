__author__ = 'Jiaxiao Zheng'
'''
Logistic regression tuning model
Vanilla logis regrfession won't work, need to do feature selection and transformation
Then tuning params
'''

import numpy as np
import pandas as pd
import util
import preprocess
from scipy import sparse
from sklearn import linear_model, preprocessing

#print basic info of training set

train_df = pd.read_csv('data/train.csv', header = 0)
test_df = pd.read_csv('data/test.csv', header = 0)


# print(preprocess.basic_info(Xtrain))
'''
Observation:
Many features are useless, having only one value,
Continuous vars are distorted except num_var35 and var15.
Strategy: discard useless features,
for continuous var which are extremely distorted, using its rank instead.

Update: inspired by engineering 2, first we run select from model to determine good individual features
Then select interaction terms from those
'''
#stat_info = util.load_obj('info_at_2016_04_14_22_38_05.pkl')
stat_info = util.load_obj('info_at_2016_04_14_19_41_35.pkl')
n_train = train_df.shape[0]
print(n_train)


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
test_df['var38mc'] = np.isclose(test_df.var38, 117310.979016)
test_df['logvar38'] = test_df.loc[~test_df['var38mc'], 'var38'].map(np.log)
test_df.loc[test_df['var38mc'], 'logvar38'] = 0

X = train_df.drop(['TARGET', 'ID'], axis = 1)
X = pd.to_numeric(X, errors = 'coerce').as_matrix()
test_id = test_df.ID
Xtest = test_df.drop(['ID'], axis = 1).as_matrix()

X_total = np.vstack((X, Xtest))

y = train_df.TARGET.values

#useful_X = util.load_dataset('usefulX.pkl')
n, d = X_total.shape
X_cat = []
X_ord = []
for i in range(0,d):
#    print('transforming ' + str(i))
    current_col = X_total[:, i]
    current_train_col = X[:, i]
   # print(current_col)

    current_n_values = len(set(current_col))
    current_n_values_train = len(set(current_train_col))
#    print(current_col.dtype)

    if(current_n_values_train == 207):
        #whether it is var3
        X_cat.append(preprocess.ranking(current_col))
    else:
        if(current_n_values_train > 10):
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

#select from model
X_train = X[0:n_train, :]
X_test = X[n_train:, :]

train_y = train_df['TARGET']

model = linear_model.LogisticRegression(penalty = 'l2', C = 0.1, intercept_scaling = 10, class_weight = 'balanced')
selectedX, selectedX_test, fs = preprocess.model_feature_select(model, X_train, train_y, X_test)

#find the list of float
n, d = selectedX.shape
print('n features after model selection is '+str(d))
list_float = []
for i in range(0, d):
    current_col = selectedX[:, i]
    c_n_values = len(set(current_col))
    if(c_n_values > 10):
        list_float.append(i)

#generate some interaction terms: d = 86... Try without interactions first

X_train_all, X_test_all, n_ord_good = preprocess.feature_select(model, X_train, train_y, Xtest, list_float, seed = 39, n_feat_limit = 0.8 * d)
util.save_dataset('post_selection', X_train_all, X_test_all)
print(n_ord_good)
util.save_obj(n_ord_good, 'n_ord_good.pkl')


# if True:
#     useful_X_ord = []
#     useful_X_cat = []
#     for i in range(0,d):
#
# #        col = used_cols[i]
#         print('transforming ' + col)
#         current_col_train = train_df[col]
#         current_col_test = test_df[col]
#         current_col = np.concatenate([current_col_train, current_col_test])
#
#        # print(len(current_col))
#         current_info = stat_info[i]
#         if(current_info['n_values'] == 1):
#             continue
#
#         if(col == 'num_var35' or col == 'var15'):
#             useful_X_ord.append(current_col)
#         else:
#             if(current_info['n_values'] > 4):
#                 transformed_col = preprocess.ranking(current_col)
#                 useful_X_ord.append(transformed_col)
#             else:
#                 useful_X_cat.append(current_col + np.min(current_col))
#
#     useful_X_ord = np.array(useful_X_ord).T
#     useful_X_cat = np.array(useful_X_cat).T
#     useful_X = np.hstack([useful_X_ord, useful_X_cat])
#     useful_X_train = useful_X[0:n_train, :]
#     useful_X_test = useful_X[n_train:, :]
#     n_ord = useful_X_ord.shape[1]
#     util.save_dataset('usefulX', useful_X_train, useful_X_test)
#     util.save_obj(n_ord,'number_of_ord.pkl')
# else:
#     useful_X_train = useful_X[0]
#     useful_X_test = useful_X[1]
#     n_ord = util.load_obj('number_of_ord.pkl')

#
#
#
# print('training samples:' + str(useful_X_train.shape[0]))
# print('testing samples:' + str(useful_X_test.shape[0]))
# print('no of features:' + str(useful_X_train.shape[1]))
#
# #Do a basic logis regression
#
# model = linear_model.LogisticRegression(penalty = 'l2', C = 0.1, intercept_scaling = 10, class_weight = 'balanced')
# X_ord_train = useful_X_train[:, 0:n_ord]
# X_cat_train = useful_X_train[:, n_ord:]
# X_cat_test = useful_X_test[:, n_ord:]
# # print('one hot encoding...')
# # X_cat = np.vstack([X_cat_train, X_cat_test])
# # X_cat, keymap = preprocess.OneHotEncoder(X_cat)
# # #X_cat = util.load_obj('X_cat.pkl')
# # util.save_obj(X_cat, 'X_cat.pkl')
# # print('one hot encoded.')
# # X_cat_train = X_cat[0:n_train, :]
# # X_cat_test = X_cat[n_train:, :]
# # print(X_ord_train.shape)
# # print(X_cat_train.shape)
# # useful_X_train = sparse.hstack([X_ord_train, X_cat_train]).tocsr()
# # useful_X_train
# print(useful_X_train.shape)
# # auc_score = util.cv_loop(useful_X_train, train_y, model, N = 10)
# # print('mean auc: ' + str(auc_score))
#
# #A basic logis regression gives us auc around 0.75
# #Given the huge number of feature we have, no interaction term this time, but still some feature selection
# X_train_all, X_test_all, n_ord_good = preprocess.feature_select(model, useful_X_train, y_train = train_y, Xtest=useful_X_test, list_of_float=range(0, n_ord), n_feat_limit = 12)
# util.save_dataset('post_selection', X_train_all, X_test_all)
# # This is very slow, but I think in the end it gives around 0.8
#
#



