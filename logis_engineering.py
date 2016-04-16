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
stat_info = util.load_obj('info_at_2016_04_14_19_41_35.pkl')
n_train = train_df.shape[0]
print(n_train)

useful_X = []

test = [2,4,5,8,7,1]
print(preprocess.ranking(test))

for i in range(0,len(used_cols)):

    col = used_cols[i]
    print('transforming ' + col)
    current_col_train = train_df[col]
    current_col_test = test_df[col]
    current_col = pd.concat([current_col_train, current_col_test])
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

util.save_dataset('usefulX', useful_X_train, useful_X_test)




