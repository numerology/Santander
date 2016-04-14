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
    util.save_dataset('extended_data_degree_' + str(int_degree) + 'at' + str(timestamp), X_train, X_test)
    return X_train, X_test






def feature_select(model, data, seed = 39):
    '''
    :param model: model we are running selection for
    :param data: training data
    :param seed: random seed
    :return: list of good features
    '''






