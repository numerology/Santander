__author__ = 'Jiaxiao Zheng'

'''
Basic utility function included
'''

import numpy as np
import pandas as pd
import logging
from scipy import sparse
import cPickle as pickle
from sklearn import metrics, cross_validation
import os


logger = logging.getLogger(__name__)

SEED = 42

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def save_dataset(filename, X, X_test, features=None, features_test=None):
    """Save the training and test sets augmented with the given features."""
    if features is not None:
        assert features.shape[1] == features_test.shape[1], "features mismatch"
        if sparse.issparse(X):
            features = sparse.lil_matrix(features)
            features_test = sparse.lil_matrix(features_test)
            X = sparse.hstack((X, features), 'csr')
            X_test = sparse.hstack((X_test, features_test), 'csr')
        else:
            X = np.hstack((X, features))
            X_test = np. hstack((X_test, features_test))

    logger.info("> saving %s to disk", filename)
    with open("data/%s.pkl" % filename, 'wb') as f:
        pickle.dump((X, X_test), f, pickle.HIGHEST_PROTOCOL)

def load_dataset(filename, use_cache=True):
    """Attempt to load data from cache."""
    data = None
    read_mode = 'rb' if '.pkl' in filename else 'r'
    if use_cache:
        try:
            with open("data/%s" % filename, read_mode) as f:
                data = pickle.load(f)
        except IOError:
            pass

    return data

def save_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filepath):
    data = None
    read_mode = 'rb' if '.pkl' in filepath else 'r'
    try:
        with open(filepath, read_mode) as f:
            data = pickle.load(f)
    except IOError:
        pass

    return data

def make_submission(filename, preds, ids):
    submission = pd.DataFrame({'id': ids, 'TARGET':preds})

    submission.to_csv(os.path.join('submission', filename + '.csv'), index = False)

def cv_loop(X, y, model, N, default_proba = False):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20,
            random_state=i*SEED)
      #  print(X_train)
      #  print(y_train)
      #
      #   if len(X_train.shape) == 1:
      #       X_train = np.matrix(X_train)
      #       X_train = np.transpose(X_train)
      #       X_cv = np.matrix(X_cv)
      #       X_cv = np.transpose(X_cv)

        model.fit(X_train, y_train)
        if(not default_proba):
            preds = model.predict_proba(X_cv)[:, 1]
        else:
            preds = model.predict(X_cv)

        preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds)) 

        # print(np.max(preds))
        # print(np.min(preds))

        auc = metrics.roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

