__author__ = 'Jiaxiao Zheng'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from scipy import sparse
from scipy.sparse import csr_matrix
from itertools import combinations
import time
import datetime
import cPickle as pickle
import util

def OneHotEncoder(data, keymap=None):
    """
    migrated from Miroslaw's code:

    OneHotEncoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix.

    Returns sparse binary matrix and keymap mapping categories to indicies.
    If a keymap is supplied on input it will be used instead of creating one
    and any categories appearing in the data that are not in the keymap are
    ignored
    """
    if keymap is None:
        keymap = []
        for col in data.T:
            uniques = set(list(col))
            keymap.append(dict((key, i) for i, key in enumerate(uniques)))
    total_pts = data.shape[0]
    outdat = []
    for i, col in enumerate(data.T):
        km = keymap[i]
        num_labels = len(km)
        spmat = sparse.lil_matrix((total_pts, num_labels))
        for j, val in enumerate(col):
            if val in km:
                spmat[j, km[val]] = 1
        outdat.append(spmat)
    outdat = sparse.hstack(outdat).tocsr()
    return outdat, keymap

def product(tuple1):
    """Calculates the product of a tuple"""
    prod = np.multiply(tuple1[0], tuple1[1])
    return prod

def group_float_data(data):
    #here i am not using hash, but the product
    #degree is fixed to 2
    new_data = []
    m, n = data.shape
    for indices in combinations(range(n), 2):
        new_data.append([product(tuple(v)) for v in data[:, indices]])
    return np.array(new_data).T

def group_int_data(data, degree=3, hash=hash):
    new_data = []
    m, n = data.shape
    for indices in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:, indices]])
    return np.array(new_data).T

def basic_info(data, use_col = None):
    '''
    :param data: data should be of pandas.dataframe
    :param use_col: the interesting columns
    :return: a dict containing basic statics of cols
    '''

    n, d = data.shape
    if(use_col is None):
        use_col = range(0, d)

    data = data[use_col]
    col_name = data.columns.values
    info_list = []
    for i in use_col:
        current_dict = {}
        set_values = []
        current_col = data[i]
        n_values = len(set(current_col))
        current_dict['n_values'] = n_values
        current_dict['max'] = max(current_col)
        current_dict['min'] = min(current_col)

        #couting frequent
        if(n_values > 10):
            current_dict['frequent'] = 'continuous, check the histogram'
            # plot a histogram instead
            hist, bins = np.histogram(current_col, bins = 20)
            width = (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            #   print(bins)
            #   print(hist)
            plt.bar(center, hist, align='center', width=width)
            plt.savefig('data/' + i + '_hist.png', bbox_inches='tight')

        else:
            frequent_dict = {}
            for values in set(current_col):
                frequent_dict[values] = 0

            for j in range(0, n):
                frequent_dict[current_col[j]] += 1

            current_dict['frequent'] = frequent_dict

        info_list.append(current_dict)

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    pickle.dump(info_list, 'info_at_' + str(timestamp) + '.pkl')
    return info_list

def add_interaction(data, float_cols, n_train, int_degree = 2):
    '''
    adding interaction features into the dataset

    :param data: data we are looking at
    :param float_cols: cols of numerical/ordinal features
    :param int_degree: cols of categorical features
    :return: expanded dataset with interactive features
    '''
    n, d = data.shape
    total_float = data[:, float_cols]
    int_cols = [x for x in range(0,d) if x not in float_cols]
    total_int = data[:, int_cols]

    print('transforming data')
    dp_float = group_float_data(total_float)
    total_float = np.hstack((total_float, dp_float))

    new_float_cols = range(0, total_float.shape[1])

    dp_int = group_int_data(total_int, degree = 2)
    new_total_int = np.hstack((total_int, dp_int))
    if(int_degree > 2):
        dt_int = group_int_data(total_int, degree = 3)
        new_total_int = np.hstack((new_total_int, dt_int))

    total_data = np.hstack((total_float, new_total_int))
    X_train = total_data[0:n_train ,:]
    X_test = total_data[n_train:, :]
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#    util.save_dataset('extended_data_degree_' + str(int_degree) + 'at' + str(timestamp), X_train, X_test)
    return X_train, X_test, new_float_cols

def float_categorize(float, ):



def feature_select(model, X, y_train, Xtest, list_of_float, seed = 39):
    '''
    :param model: model we are running selection for
    :param data: training X
    :param ytrain: y in training set
    :param list_of_float: list of ordinal features
    :param seed: random seed
    :return: transformed data set and training set with good features,
        and the number of ordinal features. In the stacked matrix, ordinal
        features appear first
    '''

    n, d = X.shape
    list_of_int = [x for x in range(0,d) if x not in list_of_float]
    X_float = X[:, list_of_float]
    X_test_float = Xtest[:, list_of_float]
    X_int = X[:, list_of_int]
    X_test_int = Xtest[:, list_of_int]

    Xts = range(0, len(list_of_float))
    print('performing feature selection on ordinal data')
    score_hist = []
    N = 10

    good_features = set([])
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        for f in range(len(Xts)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = np.hstack([X_float[:,j:j+1] for j in feats])
                score = util.cv_loop(Xt, y_train, model, N)
                scores.append((score, f))
                print "Feature: %i Mean AUC: %f" % (f, score)
        good_features.add(sorted(scores)[-1][1])
        score_hist.append(sorted(scores)[-1])
        print "Current features: %s" % sorted(list(good_features))

    # Remove last added feature from good_features
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    print(good_features)

    good_features_float = good_features

    X_float_train_good = X_float[:, good_features_float]
    X_float_test_good = X_test_float[:, good_features_float]

    Xts = [OneHotEncoder(X_int[:, [i]])[0] for i in range(len(list_of_int))]
    print('Performing feature selection on categorical features')
    score_hist = []
    N = 10

    good_features = set([])
    scores = []
    score_hist = []
    cnt = 0

    X_good_float = csr_matrix(X_float_train_good)
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        if cnt > 0:
            good_features.add(sorted(scores)[-1][1])
            scores = []
        for f in range(len(Xts)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                Xt = sparse.hstack([X_good_float, Xt]).tocsr()
                score = util.cv_loop(Xt, y_train, model, N)
                scores.append((score, f))
                print "Feature: %i Mean AUC: %f" % (f, score)

        score_hist.append(sorted(scores)[-1])
        cnt += 1
        print "Current features: %s" % sorted(list(good_features))


    # Remove last added feature from good_features

    good_features = sorted(list(good_features))
    print("In this run, final selection is:")
    print(good_features)
    good_features_int = good_features

    X_int_train_good = X_int[:, good_features_int]
    X_int_test_good = X_test_int[:, good_features_int]

    X_train_all = np.hstack((X_float_train_good, X_int_train_good))
    X_test_all = np.hstack((X_float_test_good, X_int_test_good))

    return X_train_all, X_test_all, X_float_test_good.shape[1]















