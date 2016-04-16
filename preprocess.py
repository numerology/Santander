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
import operator
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
        use_col = data.columns.values

    data = data[use_col]
    col_name = data.columns.values
    info_list = []
    for i in use_col:
        print('gathering info for '+ i)
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
            #plot a histogram instead
            hist, bins = np.histogram(current_col, bins = 20)
            width = (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            # print(bins)
            # print(hist)
            plt.bar(center, hist, align='center', width=width)
            plt.savefig('data/' + i + '_hist.png', bbox_inches='tight')
            plt.clf()

        else:
            frequent_dict = {}
            for values in set(current_col):
                frequent_dict[values] = 0

            for j in range(0, n):
                frequent_dict[current_col[j]] += 1

            current_dict['frequent'] = frequent_dict

        info_list.append(current_dict)

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    util.save_obj(info_list, 'info_at_' + str(timestamp) + '.pkl')
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

def float_categorize(float, threds_list = None, n_level = 4):
    '''
    convert float features into categorical ones by quantization

    :param float: float features
    :param threds_list: thresholds used to quantizing, if not automatic quantization is applied
    :param n_level: when automatic quantization is applied, the number of level of each feature
    :return: quantized version of float
    '''
    n, d = float.shape
    if(threds_list is None):
        #use automatic quantization
        result = []
        for i in range(0,d):
            current_col = float[:, i]
            ub = max(current_col)
            lb = min(current_col)
            step = (ub - lb) / (n_level - 1)
            thresh = util.frange(lb + step, ub, step)
            transformed_col = []
            for j in range(0, n):
                current_e = current_col[j]
                compare_res = thresh - current_e
                transformed_col.append(np.argmax(compare_res > 0) + 1)

            result.append(transformed_col)

    else:
        result = []
        for i in range(0,d):
            current_col = float[:, i]
            thresh = threds_list[i]
            transformed_col = []
            for j in range(0, n):
                current_e = current_col[j]
                compare_res = thresh - current_e
                transformed_col.append(np.argmax(compare_res > 0) + 1)

            result.append(transformed_col)

    return np.array(result).T


def ranking(score):
    '''
    method to create a score into rank, modified from KazAnova's code
    :param score: 1-d array which is to be converted, should contain both training and test samples
    :return: vector of ranking
    '''
    """ """
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


def int_extraction(int, y_train, n_train):
    '''
    extracted feature generation for categorical features:
    including: frequent/frequency of current value
               frequent/frequency of current value associated with a positive in the training set
    :param int: categorical feature set
    :param y_train: label of training set
    :param n_train:  number of training samples
    :return: dictionary
    '''
    n, d = int.shape
    frequent = []
    frequency = []
    pos_frequent = []
    pos_frequency = []

    for i in range(0, d):
        current_col = int[0:n_train, i]
        frequent_dict = {}
        pos_frequent_dict = {}
        frequency_dict = {}
        pos_frequency_dict = {}
        for val in set(current_col):
            frequent_dict[val] = 0
            pos_frequent_dict[val] = 0
        for j in range(0, n_train):
            frequent_dict[current_col[j]] += 1
            if(y_train[j] == 1):
                pos_frequent_dict[current_col[j]] += 1

        for val in frequent_dict:
            frequency_dict[val] = frequent_dict[val] / n_train
            pos_frequency_dict[val] = pos_frequent_dict[val] / n_train

        #transform
        transformed_col1 = []
        transformed_col2 = []
        transformed_col3 = []
        transformed_col4 = []
        for j in range(0, n):
            transformed_col1.append(frequent_dict[current_col[j]])
            transformed_col2.append(frequency_dict[current_col[j]])
            transformed_col3.append(pos_frequent_dict[current_col[j]])
            transformed_col4.append(pos_frequency_dict[current_col[j]])

        frequent.append(transformed_col1)
        frequency.append(transformed_col2)
        pos_frequent.append(transformed_col3)
        pos_frequency.append(transformed_col4)

        frequent = np.array(frequent).T
        frequency = np.array(frequency).T
        pos_frequent = np.array(pos_frequent).T
        pos_frequency = np.array(pos_frequency).T

        res = {}
        res['frequent'] = frequent
        res['frequency'] = frequency
        res['pos_frequent'] = pos_frequent
        res['pos_frequency'] = pos_frequency

        return res





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
    N = 5

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
    N = 5

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















