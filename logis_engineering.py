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
from sklearn import linear_model

#print basic info of training set

train_df = pd.read_csv('data/train.csv', header = 0)
test_df = pd.read_csv('data/test.csv', header = 0)
cols = train_df.columns.values
used_cols = [x for x in cols if not (x == 'ID' or x == 'TARGET')]
Xtrain = train_df[used_cols]

# print(preprocess.basic_info(Xtrain))
'''
Observation:
Many features are useless, having only one value,
Continuous vars are distorted except num_var35 and var15.
Strategy: discard useless features,
for continuous var which are extremely distorted, using its rank instead.
'''
stat_info = util.load_obj('info_at_2016_04_14_22_38_05.pkl')
n_train = train_df.shape[0]
print(n_train)

useful_X = util.load_dataset('usefulX.pkl')

if useful_X is None:
    for i in range(0,len(used_cols)):

        col = used_cols[i]
        print('transforming ' + col)
        current_col_train = train_df[col]
        current_col_test = test_df[col]
        current_col = np.concatenate([current_col_train, current_col_test])

       # print(len(current_col))
        current_info = stat_info[i]
        if(current_info['n_values'] == 1):
            continue

        if(col == 'num_var35' or col == 'var15'):
            useful_X.append(current_col)
        else:
            if(current_info['n_values'] > 10):
                transformed_col = preprocess.ranking(current_col)
                useful_X.append(transformed_col)
            else:
                useful_X.append(current_col)

    useful_X = np.array(useful_X).T
    useful_X_train = useful_X[0:n_train, :]
    useful_X_test = useful_X[n_train, :]

else:
    useful_X_train = useful_X[0]
    useful_X_test = useful_X[1]

#util.save_dataset('usefulX', useful_X_train, useful_X_test)
train_y = train_df['TARGET']
print('training samples:' + str(useful_X_train.shape[0]))
print('testing samples:' + str(useful_X_test.shape[0]))
print('no of features:' + str(useful_X_train.shape[1]))

#Do a basic logis regression
model = linear_model.LogisticRegression(penalty = 'l2', C = 0.1, intercept_scaling = 10, class_weight = 'balanced')
auc_score = util.cv_loop(useful_X_train, train_y, model, N = 10)
print('mean auc: ' + str(auc_score))
#A basic logis regression gives us auc around 0.75






