__author__ = 'Jiaxiao Zheng'
'''
Borrow code from boulder
origin:
https://github.com/BoulderDataScience/kaggle-santander/blob/master/submission_nn.py

Note: use gpu enabled computing can make both theano and tensorflow faster, but the 
benefit is not significant unless # of node > 1200
E.G. When I have 6000 node, one epoch takes 12s for gpu,
while cpu needs 120s
env: GPU: NVidia GTX 970, Cuda 7.0, cuDNN 4.0
CPU: i7-6700 3.4GHz
'''
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn import metrics, cross_validation

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.regularizers import l1l2, activity_l1l2

import operator

#from santander.preprocessing import ColumnDropper
class ColumnDropper(BaseEstimator):
    """
    Drop columns based on name
    """
    def __init__(self, drop=[]):
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.drop, axis=1)

def cv_auc(compiled_model, epochs, X, y, N, SEED=39, class_weight = {1:0.5, 0:0.5}):
    mean_auc = 0.
    skf = cross_validation.StratifiedKFold(y, n_folds = N, shuffle = True, random_state = SEED)

    i = 0
    for train_index, test_index in skf:
        # Update 4/25, use stratified k fold

        # X_train_cv, X_test_cv, y_train_cv, y_test_cv = cross_validation.train_test_split(
        #         X, y, test_size=.20,
        #         random_state=i*SEED)
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        if(epochs == -1):
            es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
            model.fit(X_train_cv, y_train_cv, nb_epoch=2000, shuffle = True, verbose = 1, 
                callbacks = [es], validation_data = (X_test_cv, y_test_cv), class_weight = class_weight)
        else:
            model.fit(X_train_cv, y_train_cv, nb_epoch=epochs, shuffle = True, verbose = 1,
                class_weight = class_weight)
        preds = model.predict_proba(X_test_cv)
        auc = metrics.roc_auc_score(y_test_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
        i += 1

    print 'Final AUC: %f' % (mean_auc/N)
    return mean_auc/N

#from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS
ZERO_VARIANCE_COLUMNS = [
    'ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0',
    'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46',
    'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3',
    'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3',
    'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3',
    'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3'
]


CORRELATED_COLUMNS = [
    'ind_var29_0', 'ind_var29', 'num_var6', 'num_var29', 'ind_var13_medio', 'num_var13_medio_0', 'num_var13_medio',
    'num_meses_var13_medio_ult3', 'ind_var18', 'num_var18_0', 'num_var18', 'num_var20_0', 'num_var20', 'ind_var26',
    'ind_var25', 'ind_var32', 'ind_var34', 'ind_var37', 'ind_var39', 'num_var29_0', 'delta_imp_amort_var18_1y3',
    'num_var26', 'num_var25', 'num_var32', 'num_var34', 'delta_imp_amort_var34_1y3', 'num_var37', 'num_var39',
    'saldo_var29', 'saldo_medio_var13_medio_ult1', 'delta_num_aport_var13_1y3', 'delta_num_aport_var17_1y3',
    'delta_num_aport_var33_1y3', 'delta_num_reemb_var13_1y3', 'num_reemb_var13_ult1', 'delta_num_reemb_var17_1y3',
    'delta_num_reemb_var33_1y3', 'num_reemb_var33_ult1', 'delta_num_trasp_var17_in_1y3',
    'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3',
    'num_trasp_var33_out_ult1', 'delta_num_venta_var44_1y3'
]

filename = 'submission_nn.csv'

pipeline = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
    ('std', StandardScaler())
])

df_train = pd.read_csv('data/train.csv')
# Even out the targets
df_train_1 = df_train[df_train["TARGET"] == 1]
print('pos samples: '+str(df_train_1.shape[0]))
class_weight = {}
class_weight[1] = float(df_train_1.shape[0])/df_train.shape[0]
class_weight[0] = 1.0 - class_weight[1]
print(class_weight)

df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)

pipeline = pipeline.fit(df_train)
X_train = pipeline.transform(df_train)
print(X_train.shape)
y_train = df_target



df_test = pd.read_csv('data/test.csv')
df_id = df_test['ID']
df_test = df_test.drop(['ID'], axis=1)
X_test = pipeline.transform(df_test)
ID_test = df_id

#TODO: Try different activation function: relu and else
'''
4/22
original setting: sigmoid 2 * 32nodes, no sample balance - 0.83447
relu 2 * 32nodes, no sample balance - slow and bad, invalid prob
softmax, too much loss...
tanh, 0.5auc
hardsigmoid still 0.5
linear two much loss
Conclusion: use sigmoid
Add 1 layer: 0.8359
Double # of nodes: 0.8368.
Conclusion: maybe more epochs needed, but the improvement is not so significant

Change loss:
MSE: 0.8234
MAE: 0.5
MSLE: 0.8177
squared_hinge: 0.7x
hinge: 0.7x
poisson: in the unbalanced case, because there are so many 0's, poisson is approximately logloss
but it still gives us 0.8368
cosine_proximity: invalid
'''

'''
4/23:
Adding balanced samples, # of 0 is 1.5 * # of 1,
grid search the best layer and node:
it says 3 layers with 500 nodes per layer
'''
n_nodes_list = [90, 100, 150, 200, 300]
n_layers_list = [3]
'''
4/23:
Then let's tune dropout and learning rate
both take 0.1
'''
lr_list = [0.01, 0.05, 0.1, 0.2, 0.4]
dropout_list = [0.1, 0.15, 0.25, 0.4, 0.6]


N_cv = 5

SEED = 39

print(X_train.shape)
print(y_train.shape)

# model = Sequential()
# model.add(Dense(1264, input_shape=(X_train.shape[1],), activation='sigmoid'))
# model.add(Dropout(0.25))
# model.add(Dense(1264, activation='sigmoid'))
# model.add(Dropout(0.25))
# model.add(Dense(1264, activation='sigmoid'))
# model.add(Dropout(0.25))
# model.add(Dense(1264, activation='sigmoid'))
# model.add(Dropout(0.25))
# model.add(Dense(1264, activation='sigmoid'))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='sigmoid'))

result = {}

for n_layer in n_layers_list:
    for n_node in n_nodes_list:
        model = Sequential()
        model.add(Dense(n_node, input_shape=(X_train.shape[1],), activation='sigmoid'))
        model.add(Dropout(0.5))
        for i in range(0, n_layer - 1):
            model.add(Dense(n_node, activation='sigmoid', 
                activity_regularizer = activity_l1l2(0, 0)))
            model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        nb_epoch = 30
        opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        current_auc = cv_auc(model, -1, X_train, y_train, N_cv, SEED = 39, class_weight = class_weight)
        result[(n_layer, n_node)] = current_auc

# for dropout in dropout_list:
#     for learning_rate in lr_list:
#         model = Sequential()
#         model.add(Dense(500, input_shape=(X_train.shape[1],), activation='sigmoid'))
#         model.add(Dropout(dropout))
#         for i in range(0, 2):
#             model.add(Dense(500, activation='sigmoid'))
#             model.add(Dropout(dropout))

#         model.add(Dense(1, activation='sigmoid'))
#         nb_epoch = 100
#         opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
#         model.compile(loss='binary_crossentropy', optimizer=opt)

#         current_auc = cv_auc(model, nb_epoch, X_train, y_train, N_cv, SEED = 39)
#         result[(dropout, learning_rate)] = current_auc


sorted_result = sorted(result.items(), key = operator.itemgetter(1))
good_results = sorted_result
with open('layer_nodes_nn_tillconverge_2.txt', 'w') as f:
    f.write(str(good_results))

#TODO: tuning params, lose function
# nb_epoch = 30
# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy', optimizer=opt)



# y_pred = model.predict_proba(X_test)
# submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, -1]})
# submission.to_csv(filename, index=False)
# print 'Wrote %s' % filename