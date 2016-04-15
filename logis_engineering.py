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

print(preprocess.basic_info(Xtrain))