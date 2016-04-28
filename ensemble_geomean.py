__author__ = 'Jiaxiao Zheng'

'''
Borrow code from KazAnova,

TODO:
KazAnova directly load the results from predumped file and test different weights, which is bad.
The reason is the predictions made on the training set is based on all samples there. In the cross-
validation, you are touching 'test' set you have.

A more accurate thing to do should be actually run learning using different models on the
splitted training set and do cv, which is potentially time consuming. A possible workaround could be as follows:

X_train and X_cv are generated randomly, but keep identical through all iterations. Then we can store the predictions
and there's no need to run the algorithms again and again.

We do gridsearch for the best weights.
'''

import sys
import numpy as np
import pandas as pd
import operator
import util
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import keras_test

import xgboost as xgb
from scipy.sparse import csr_matrix

"""
Script to do the final ensemble via geomean-rank-weighted-averaging

using many cross validations schemas for better AUC score

"""


#convert a single 1-dimensional array to rank (e.g sort the score from the smallest to the highest and give them scores as 1,2...len(array))
def ranking(score):
    """ method to create a score into rank"""
    data=[]
    for i in range(len(score)):
        data.append([score[i],i])
    data=sorted(data, key=operator.itemgetter(0), reverse=False)
    value=data[0][0]
    data[0][0]=1
    for i in range(1,len(score)):
        val=data[i][0]
        if val>value :
            value=val
            data[i][0]=(i+1)
        else :
            data[i][0]=data[i-1][0]
    data=sorted(data, key=operator.itemgetter(1), reverse=False)
    final_rank=[]
    for i in range(len(score)):
        final_rank.append(data[i][0])
    return final_rank

#retrieve specific column fron 2dimensional array as a 1dimensional array
def select_column(data, col) :
    array=[]
    for i in range(len(data)):
       array.append(data[i][col])
    return array

# put an array back to the given column j
def putcolumn(data,array,j) :
    for i in range(len(data)):
        data[i][j]=array[i]

# convert each one of the columns in the given array to ranks
def create_ranklist (data ) :
    for j in range(len(data[0])):
        putcolumn( data,ranking(select_column(data,j)),j)


# method to load a specific column
def loadcolumn(filename,col=4, skip=1, floats=True):
    pred=[]
    op=open(filename,'r')
    if skip==1:
        op.readline() #header
    for line in op:
        line=line.replace('\n','')
        sps=line.split(',')
        #load always the last columns
        if floats:
            pred.append(float(sps[col]))
        else :
            pred.append(str(sps[col]))
    op.close()
    return pred

def printfilcsve(X, filename):

    np.savetxt(filename,X)


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def main():

        # meta models to be used to assess how much weight to contribute to the final submission
        '''
        4 models:
        xgboost, nn(keras), gbm_adaboost, gbm_bernoulli
        '''
        reload(sys)
        sys.setdefaultencoding('ISO-8859-1')
        N_fold = 5
        SEED = 39

        Load = True

        df_train = pd.read_csv('data/train.csv')
        #prepare data for nn
        df_train_1 = df_train[df_train["TARGET"] == 1]
        print('pos samples: '+str(df_train_1.shape[0]))
        class_weight = {}
        class_weight[1] = float(df_train_1.shape[0])/df_train.shape[0]
        class_weight[0] = 1.0 - class_weight[1]
        print(class_weight)

        if(not Load):
            

            pipeline = Pipeline([
                ('cd',keras_test.ColumnDropper(drop=keras_test.ZERO_VARIANCE_COLUMNS+keras_test.CORRELATED_COLUMNS)),
                ('std', StandardScaler())
            ])

            df_target = df_train['TARGET']
            df_train = df_train.drop(['TARGET', 'ID'], axis=1)

            pipeline = pipeline.fit(df_train)
            X_train_nn = pipeline.transform(df_train)
            print('number of samples: ' + str(X_train_nn.shape))
            y_train = df_target

            np.save('data/X_train_nn', X_train_nn)
            print('finish saving nn')

            #prepare data for gbm
            remove = []
            for col in df_train.columns:
                if(df_train[col].std() == 0):
                    remove.append(col)
            df_train.drop(remove, axis = 1, inplace = True)

            remove = []
            cols = df_train.columns
            for i in range(len(cols) - 1):
                v = df_train[cols[i]].values
                for j in range(i + 1, len(cols)):
                    if np.array_equal(v, df_train[cols[j]].values):
                        remove.append(cols[j])
            df_train.drop(remove, axis = 1, inplace = True)


            df_train_gbm = df_train.replace(-999999, 2)
            X = df_train.as_matrix()
            df_train_gbm['n0'] = (X == 0).sum(axis = 1)
            df_train_gbm['var38mc'] = np.isclose(df_train_gbm.var38, 117310.979016)
            df_train_gbm['logvar38'] = df_train_gbm.loc[~df_train_gbm['var38mc'], 'var38'].map(np.log)
            df_train_gbm.loc[df_train_gbm['var38mc'], 'logvar38'] = 0
                       
            pca = PCA(n_components=2)
            X_train_gbm = df_train_gbm
            x_train_projected = pca.fit_transform(normalize(X, axis=0))
            X_train_gbm.insert(1, 'PCAOne', x_train_projected[:, 0])
            X_train_gbm.insert(1, 'PCATwo', x_train_projected[:, 1])

            X_train_xgb = X_train_gbm.copy()

            selectK = SelectKBest(f_classif, k=90)
            selectK.fit(X_train_gbm, y_train)
            X_train_gbm = selectK.transform(X_train_gbm)
            np.save('data/X_train_gbm', X_train_gbm)
            print('finish saving gbm')

            #prepare data for xgboost
            tokeep = ['num_var39_0',  # 0.00031104199066874026
              'ind_var13',  # 0.00031104199066874026
              'num_op_var41_comer_ult3',  # 0.00031104199066874026
              'num_var43_recib_ult1',  # 0.00031104199066874026
              'imp_op_var41_comer_ult3',  # 0.00031104199066874026
              'num_var8',  # 0.00031104199066874026
              'num_var42',  # 0.00031104199066874026
              'num_var30',  # 0.00031104199066874026
              'saldo_var8',  # 0.00031104199066874026
              'num_op_var39_efect_ult3',  # 0.00031104199066874026
              'num_op_var39_comer_ult3',  # 0.00031104199066874026
              'num_var41_0',  # 0.0006220839813374805
              'num_op_var39_ult3',  # 0.0006220839813374805
              'saldo_var13',  # 0.0009331259720062209
              'num_var30_0',  # 0.0009331259720062209
              'ind_var37_cte',  # 0.0009331259720062209
              'ind_var39_0',  # 0.001244167962674961
              'num_var5',  # 0.0015552099533437014
              'ind_var10_ult1',  # 0.0015552099533437014
              'num_op_var39_hace2',  # 0.0018662519440124418
              'num_var22_hace2',  # 0.0018662519440124418
              'num_var35',  # 0.0018662519440124418
              'ind_var30',  # 0.0018662519440124418
              'num_med_var22_ult3',  # 0.002177293934681182
              'imp_op_var41_efect_ult1',  # 0.002488335925349922
              'var36',  # 0.0027993779160186624
              'num_med_var45_ult3',  # 0.003110419906687403
              'imp_op_var39_ult1',  # 0.0037325038880248835
              'imp_op_var39_comer_ult3',  # 0.0037325038880248835
              'imp_trans_var37_ult1',  # 0.004043545878693624
              'num_var5_0',  # 0.004043545878693624
              'num_var45_ult1',  # 0.004665629860031105
              'ind_var41_0',  # 0.0052877138413685845
              'imp_op_var41_ult1',  # 0.0052877138413685845
              'num_var8_0',  # 0.005598755832037325
              'imp_op_var41_efect_ult3',  # 0.007153965785381027
              'num_op_var41_ult3',  # 0.007153965785381027
              'num_var22_hace3',  # 0.008087091757387248
              'num_var4',  # 0.008087091757387248
              'imp_op_var39_comer_ult1',  # 0.008398133748055987
              'num_var45_ult3',  # 0.008709175738724729
              'ind_var5',  # 0.009953343701399688
              'imp_op_var39_efect_ult3',  # 0.009953343701399688
              'num_meses_var5_ult3',  # 0.009953343701399688
              'saldo_var42',  # 0.01181959564541213
              'imp_op_var39_efect_ult1',  # 0.013374805598755831
              'PCATwo',  # 0.013996889580093312
              'num_var45_hace2',  # 0.014618973561430793
              'num_var22_ult1',  # 0.017107309486780714
              'saldo_medio_var5_ult1',  # 0.017418351477449457
              'PCAOne',  # 0.018040435458786936
              'saldo_var5',  # 0.0208398133748056
              'ind_var8_0',  # 0.021150855365474338
              'ind_var5_0',  # 0.02177293934681182
              'num_meses_var39_vig_ult3',  # 0.024572317262830483
              'saldo_medio_var5_ult3',  # 0.024883359253499222
              'num_var45_hace3',  # 0.026749611197511663
              'num_var22_ult3',  # 0.03452566096423017
              'saldo_medio_var5_hace3',  # 0.04074650077760498
              'saldo_medio_var5_hace2',  # 0.04292379471228616
              'n0',  # 0.04696734059097978
              'saldo_var30',  # 0.09611197511664074
              'var38',  # 0.1390357698289269
              'var15']  # 0.20964230171073095
            features = X_train_xgb.columns
            todrop = list(set(tokeep).difference(set(features)))
            X_train_xgb.drop(todrop, inplace=True, axis=1)
            print(X_train_xgb.shape)
          
            np.save('data/X_train_xgb', X_train_xgb)
            print('finish saving xgb')
            np.save('data/y_train', y_train)

            # util.save_obj('data/X_train_nn.pkl', X_train_nn)
            # util.save_obj('data/X_train_gbm.pkl', X_train_gbm)
            # util.save_obj('data/X_train_xgb.pkl', X_train_xgb)
            # util.save_obj('data/y_train.pkl', y_train)

        else:
            X_train_nn = np.load('data/X_train_nn.npy')
            X_train_gbm = np.load('data/X_train_gbm.npy')
            X_train_xgb = np.load('data/X_train_xgb.npy')
            y_train = np.load('data/y_train.npy')


        kfolder=StratifiedKFold(y_train, n_folds=N_fold, shuffle=False, random_state=SEED)
        i = 0
        mean_auc = 0.
        print('label shape' + str(y_train.shape))
        xgb_subsplit = 10
        for train_index, test_index in kfolder:
            # creaning and validation sets
            X_nn_train, X_nn_cv = X_train_nn[train_index, :], X_train_nn[test_index, :]
            X_gbm_train, X_gbm_cv = X_train_gbm[train_index, :], X_train_gbm[test_index, :]
            X_xgb_train, X_xgb_cv = X_train_xgb[train_index, :], X_train_xgb[test_index, :]
            y_train_cv, y_cv = y_train[train_index], y_train[test_index]
            print (" train size: %d. test size: %d, cols: %d " % ((X_nn_train.shape[0]) ,(X_nn_cv.shape[0]) ,(X_nn_train.shape[1]) ))

            #fitting
            model_nn = Sequential()
            model_nn.add(Dense(100, input_shape=(X_nn_train.shape[1],), activation='relu'))
            model_nn.add(Dropout(0.5))
            model_nn.add(Dense(100, activation='relu'))
            model_nn.add(Dropout(0.5))
            model_nn.add(Dense(1, activation='sigmoid'))
            opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model_nn.compile(loss = 'binary_crossentropy', optimizer = opt)
            es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
            model_nn.fit(X_nn_train, y_train_cv, nb_epoch=2, shuffle = True, verbose = 1,
                callbacks = [es], validation_split = 0.25, class_weight = class_weight)
            preds_nn = model_nn.predict_proba(X_nn_cv)

            model_gbm_ada = GradientBoostingClassifier(loss = 'exponential', learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10, random_state=39)
            model_gbm_ada.fit(X_gbm_train, y_train_cv)
            preds_gbm_ada = model_gbm_ada.predict_proba(X_gbm_cv)[:, 1]

            model_gbm_ber = GradientBoostingClassifier(loss = 'deviance', learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10, random_state=39)
            model_gbm_ber.fit(X_gbm_train, y_train_cv)
            preds_gbm_ber = model_gbm_ber.predict_proba(X_gbm_cv)[:, 1]

            #xgboost, xgb in itself is using some magical way to run fitting 5 times and take geomean
            #but here let's first run just for once
            
            num_rounds = 10

            params = {}
            params["objective"] = "binary:logistic"
            params["eta"] = 0.03
            params["subsample"] = 0.8
            params["colsample_bytree"] = 0.7
            params["silent"] = 1
            params["max_depth"] = 5
            params["min_child_weight"] = 1
            params["eval_metric"] = "auc"

            xgbKfold = StratifiedKFold(y_train_cv, n_folds=xgb_subsplit, shuffle=False, random_state=42)
            dcv = xgb.Dmatrix(X_xgb_cv, silent = True)
            xgb_preds = None
            index = 0
            for xgb_train_index, xgb_test_index in xgbKfold:
                visibletrain = X_xgb_train[xgb_train_index, :]
                blindtrain = X_xgb_train[xgb_test_index, :]
                y_visibletrain, y_blindtrain = y_train_cv[xgb_train_index], y_train_cv[xgb_test_index]
                dvisibletrain = xgb.DMatrix(visibletrain,
                            label = y_visibletrain,
                            silent=True)
                dblindtrain = xgb.DMatrix(blindtrain,
                            label = y_blindtrain,
                            silent=True)

                watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
                model_xgb = xgb.train(params, dvisibletrain, num_rounds,
                            evals=watchlist, early_stopping_rounds=50,
                            verbose_eval=False)
                current_preds_xgb = model_xgb.predict(dcv)
                if(xgb_preds is None):
                    xgb_preds = current_preds_xgb
                else:
                    xgb_preds *= current_preds_xgb
                index += 1

            preds_xgb = np.power(xgb_preds, 1./index)


            weight = [0.25, 0.25, 0.25, 0.25]
            preds_nn = np.power(preds_nn, weight[0])
            preds_gbm_ada = np.power(preds_gbm_ber, weight[1])
            preds_gbm_ber = np.power(preds_gbm_ber, weight[2])
            preds_xgb = np.power(preds_xgb, weight[3])

            preds = np.multiply(np.multiply(np.multiply(preds_nn[:,0], preds_gbm_ada), preds_gbm_ber), preds_xgb)
            print(y_cv.shape)
            print(preds.shape)
            auc = roc_auc_score(y_cv, preds)

            print "AUC (fold %d/%d): %f" % (i + 1, N_fold, auc)
            i+=1
            mean_auc += auc

        print('Final AUC is ' + str(mean_auc/N_fold))



        # print("len of target=%d" % (len(y))) # reconciliation check
        # weights=[0, # all weights to 1, e.g. average
        #          0,
        #          0,
        #          0.4,
        #          0,
        #          0.4,
        #          1
        #          ]     # the weights of the 4 level 3 meta models



        # number_of_folds=5 # for cv
        # usesccaling_to_0_1=True # some submissions need probas-ish
        # use_geo=True #false = uses linear rank average
        # Load=True
        # use_rank=True # IF we want to use rank
        # #basiclaly it says multiple the extra lvl3 model by 1, the xgboost model by 0.05 and the neural net with 0.25
        # if Load:
        #     Xmetatrain=None
        #     Xmetatest=None
        #     #append all the predictions into 1 list (array)
        #     for modelname in meta :
        #         mini_xtrain=np.loadtxt(modelname + '.train.csv')
        #         mini_xtest=np.loadtxt(modelname + '.test.csv')
        #         mean_train=np.mean(mini_xtrain)
        #         mean_test=np.mean(mini_xtest)
        #         print("model %s auc %f mean train/test %f/%f " % (modelname,roc_auc_score(y,mini_xtrain) ,mean_train,mean_test))
        #         if Xmetatrain==None:
        #             Xmetatrain=mini_xtrain
        #             Xmetatest=mini_xtest
        #         else :
        #             Xmetatrain=np.column_stack((Xmetatrain,mini_xtrain))
        #             Xmetatest=np.column_stack((Xmetatest,mini_xtest))
        #     # convert my scores to list

        #     X=Xmetatrain
        #     X_test=Xmetatest
        #     joblib.dump((X,X_test),"METADUMP.pkl" )
        # else :
        #     X,X_test=joblib.load("METADUMP.pkl")

        # outset="AUC_Geo_Rank_Weighted_Average" # Output base name


        # seedlist=[87, 111, 1337, 42 , 201628] # many seeds for more accurate results
        # train_stacker=[0.0 for i in range (0,len(X))]
        # mean_auc = 0.0
        # for seeder in seedlist:
        #     print("kfolding seed %d " % (seeder) )
        #     kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=seeder)
        #     #number_of_folds=0
        #     #X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time
        #     i=0 # iterator counter
        #     print ("starting cross validation with %d kfolds " % (number_of_folds))
        #     if number_of_folds>0:
        #         for train_index, test_index in kfolder:
        #             # creaning and validation sets
        #             X_train, X_cv = X[train_index], X[test_index]
        #             y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        #             print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))


        #             minmax=MinMaxScaler(feature_range=(0, 1))
        #             X_cv=X_cv.tolist()
        #             if use_rank:
        #                 create_ranklist(X_cv)

        #             #X_cv= minmax.fit_transform((X_cv))
        #             #print X_cv

        #             if use_geo: # use geo mean
        #                 preds=[1.0 for s in range (0,len(X_cv))]
        #                 for i in range (0,len(X_cv)) :
        #                     for j in range (0,len(weights)) :
        #                         preds[i]*=X_cv[i][j]**weights[j]
        #             else :
        #                 preds=[0.0 for s in range (0,len(X_cv))]
        #                 for i in range (0,len(X_cv)) :
        #                     for j in range (0,len(weights)) :
        #                         preds[i]+=X_cv[i][j]*weights[j]

        #             if usesccaling_to_0_1:
        #                 preds= minmax.fit_transform(preds)

        #             # compute Loglikelihood metric for this CV fold
        #             #scalepreds(preds)
        #             AUC = roc_auc_score(y_cv,preds)
        #             print "size train: %d  CV : %d AUC (fold %d/%d): %f" % ((X_train.shape[0]), len(X_cv), i + 1, number_of_folds, AUC)

        #             mean_auc += AUC
        #             #save the results
        #             no=0
        #             for real_index in test_index:
        #                      train_stacker[real_index]=(preds[no])
        #                      no+=1
        #             i+=1

        # mean_auc/=(len(seedlist)*5.0)
        # print ("Average AUC: %f" % mean_auc)
        # minmax=MinMaxScaler(feature_range=(0, 1))
        # X_test=X_test.tolist()
        # if use_rank:
        #     create_ranklist(X_test)

        # # combine all the ranked scores in a weighted manner for the test lvl 3 out-of-fold predictions


        # if use_geo: # use geo mean
        #     preds=[1.0 for s in range (0,len(X_test))]
        #     for i in range (0,len(X_test)) :
        #         for j in range (0,len(weights)) :
        #             preds[i]*=X_test[i][j]**weights[j]
        # else : # linear wighted rank average
        #     preds=[0.0 for s in range (0,len(X_test))]
        #     for i in range (0,len(X_test)) :
        #         for j in range (0,len(weights)) :
        #             preds[i]+=X_test[i][j]*weights[j]

        # if usesccaling_to_0_1:
        #     preds= minmax.fit_transform(preds)


        # #convert to numpy
        # preds=np.array(preds)
        # #write the results

        # save_results(preds, outset+"_submission_" +str(mean_auc) + ".csv")
        # print("Done.")






if __name__=="__main__":
  main()