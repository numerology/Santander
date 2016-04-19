# import os
# bashCommand = "export PYTHONPATH=${PYTHONPATH}:/Users/ChenYitao/gitLocal/scikit-learn"
# os.system(bashCommand)
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn.neural_network import MLPClassifier
import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA


SEED = 13  # seed to replicate results

def save_results(predictions,IDs,  filename):
    with open(filename, 'w') as f:
        f.write("ID,TARGET\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (IDs[i], pred))

def main():

    filename="yitao_chen" # name prefix - you could put your name, like marios_mich
    
    #model = LogisticRegression(C=1)  # Sample of LogisticRegression model. We will adjust the C value base don our 5-fold CV
    model=MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,4,5), random_state=1)

    # === load data into numpy arrays === #
    train = pd.read_csv("../input/train.csv", header = 0)
    test = pd.read_csv("../input/test.csv", header = 0)

    features = train.columns[1:-1]
    train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
    test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))

    remove = []
    c = train.columns
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])

    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)

    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    features = train.columns[1:-1]
    pca = PCA(n_components=2)
    x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
    x_test_projected = pca.transform(normalize(test[features], axis=0))
    train.insert(1, 'PCAOne', x_train_projected[:, 0])
    train.insert(1, 'PCATwo', x_train_projected[:, 1])
    test.insert(1, 'PCAOne', x_test_projected[:, 0])
    test.insert(1, 'PCATwo', x_test_projected[:, 1])
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
              'SumZeros',  # 0.04696734059097978
              'saldo_var30',  # 0.09611197511664074
              'var38',  # 0.1390357698289269
              'var15']  # 0.20964230171073095
    features = train.columns[1:-1]
    todrop = list(set(tokeep).difference(set(features)))
    train.drop(todrop, inplace=True, axis=1)
    test.drop(todrop, inplace=True, axis=1)

    # print train.shape
    # print test.shape

    X = np.array(train)[:,1:-1]
    y = np.array(train.TARGET)
    X_test = np.array(test)[:,1:]
 
 	# === input check ===
    # print X
    # print np.shape(X)
    # print np.shape(y)
    # print np.shape(X_test)

    # === search for best paramters === #
    alpha_vals = np.logspace(-7, -1, 10, endpoint = True, base=10.0)

    # do cross validation on other paramters to optimize
    # alpha
    score_hist = []
    for alpha in alpha_vals:
        param = {'alpha':alpha}
        model.set_params(**param)
        i = 0
        n = 5  # number of folds in strattified cv
        kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=SEED*(i+1))  
        mean_auc = 0.0
        for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
            # creaning and validation sets
            X_train, X_cv = X[train_index], X[test_index]
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
            print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
            
            # do scalling
            scaler=StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_cv = scaler.transform(X_cv)

            # train model
            model.fit(X_train,y_train)
            #  make predictions in probabilities
            preds=model.predict_proba(X_cv)[:,1]

            # compute AUC metric for this CV fold
            # err = mean_squared_error(y_cv, preds)
            # mean_err += err
            roc_auc = roc_auc_score(y_cv, preds)
            mean_auc += roc_auc
            i+=1
        mean_auc/=n
        score_hist.append((mean_auc, alpha))
        print ("alpha: %f Average AUC: %f" % (alpha, mean_auc) ) 
    print sorted(score_hist)    
    bestAlpha = sorted(score_hist)[-1][1] 
    print "Best alpha value: %f" % (bestAlpha)  

    #bestAlpha = 0.021544
    param = {'alpha':bestAlpha}
    model.set_params(**param)


    # === training & metrics === #
    mean_auc = 0.0
    n = 5  # number of folds in strattified cv
    kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=SEED)     
    i=0
    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
        # creaning and validation sets
        X_train, X_cv = X[train_index], X[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
        
        # do scalling
        scaler=StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_cv = scaler.transform(X_cv)

        # train model
        model.fit(X_train,y_train)
        #  make predictions in probabilities
        preds=model.predict_proba(X_cv)[:,1]

        # compute AUC metric for this CV fold
        roc_auc = roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc
        i+=1
        
    mean_auc/=n
    print (" Average AUC: %f" % (mean_auc) )        
    
    # do scalling
    scaler=StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test) 

    #make final model
    model.fit(X,y)
    #  make predictions in probabilities
    preds=model.predict_proba(X_test)[:,1]    
    #create submission file 
    save_results(preds, test.ID, "../submissions/mlp_nn_" + filename+"_" +str(mean_auc).replace(".","_") + ".csv") # putting the actuall AUC (of cv) in the file's name for reference
    
    print("Submission %s has been generated!" % ( "../submissions/mlp_nn_" + filename+"_" +str(mean_auc).replace(".","_") + ".csv"))


if __name__ == '__main__':
    main()