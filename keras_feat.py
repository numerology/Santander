__author__ = 'Jiaxiao Zheng'
'''
Borrow code from boulder
origin:
https://github.com/BoulderDataScience/kaggle-santander/blob/master/submission_nn.py

Try different feature set fed into the NN, for example, features with pca, features after 
selection by xgboost, features with ranking
'''
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn import metrics, cross_validation

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

import operator
import xgboost as xgb

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

def cv_auc(compiled_model, epochs, X, y, N, SEED=39):
    mean_auc = 0.
    for i in range(N):
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = cross_validation.train_test_split(
                X, y, test_size=.20,
                random_state=i*SEED)
        model.fit(X_train_cv, y_train_cv, nb_epoch=epochs, verbose = 0)
        preds = model.predict_proba(X_test_cv)
        auc = metrics.roc_auc_score(y_test_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    print 'Final AUC: %f' % (mean_auc/N)
    return mean_auc/N


# Read data
df_train = pd.read_csv("data/train.csv", index_col='ID')
feature_cols = list(df_train.columns)
feature_cols.remove("TARGET")
df_test = pd.read_csv("data/test.csv", index_col='ID')

# Split up the data
X_all = df_train[feature_cols]
y_all = df_train["TARGET"]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X_all, y_all, test_size=0.2, random_state=5, stratify=y_all)

# Get top features from xgb model
model = xgb.XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=9,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=5
)

# Train cv
xgb_param = model.get_xgb_params()
dtrain = xgb.DMatrix(X_train.values, label=y_train.values, missing=np.nan)
cv_result = xgb.cv(
    xgb_param,
    dtrain,
    num_boost_round=model.get_params()['n_estimators'],
    nfold=5,
    metrics=['auc'],
    early_stopping_rounds=50)
best_n_estimators = cv_result.shape[0]
model.set_params(n_estimators=best_n_estimators)

# Train model
model.fit(X_train, y_train, eval_metric='auc')

# Predict training data
y_hat_train = model.predict(X_train)

# Predict test data
y_hat_test = model.predict(X_test)

# Print model report:
print("\nModel Report")
print("best n_estimators: {}".format(best_n_estimators))
print("AUC Score (Train): %f" % roc_auc_score(y_train, y_hat_train))
print("AUC Score (Test) : %f" % roc_auc_score(y_test,  y_hat_test))

# Get important features
feat_imp = list(pd.Series(model.booster().get_fscore()).sort_values(ascending=False).index)

df_train_1 = df_train[df_train["TARGET"] == 1]
print('pos samples: '+str(df_train_1.shape[0]))
df_train_0 = df_train[df_train["TARGET"] == 0].head(df_train_1.shape[0] * 2)
df_train = df_train_1.append(df_train_0)


not_important = [x for x in feature_cols if not x in feat_imp]
X_all = df_train.copy(deep=True)
pipeline = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS+not_important)),
    ('std', StandardScaler())
])
pipeline = pipeline.fit(X_all)
X_train = pipeline.transform(X_all)
y_train = df_train["TARGET"]
#X_all[feat_imp] = sklearn.preprocessing.scale(X_all, axis=0, with_mean=True, with_std=True, copy=True)
print(X_train.shape)
print(y_train.shape)



#TODO: Try different activation function: relu and else


N_cv = 5

SEED = 39

model = Sequential()
model.add(Dense(500, input_shape=(X_train.shape[1],), activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

nb_epoch = 1500
opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=opt)
print('AUC = ' + str(cv_auc(model, nb_epoch, X_train, y_train, N_cv, SEED = 39)))


#TODO: tuning params, lose function
# nb_epoch = 30
# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy', optimizer=opt)



# y_pred = model.predict_proba(X_test)
# submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, -1]})
# submission.to_csv(filename, index=False)
# print 'Wrote %s' % filename