# Imports
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher

from utilities import normalize

import bisect
import copy

# cat2vec
from gensim.models.word2vec import Word2Vec
from random import shuffle

def label(x, column):
    """ 
        Performs label encoding.
        Example:
            Color: ['blue', 'green', 'blue', 'pink']
            is encoded by
            Color: [1, 2, 1, 3]
            
        :param df: Data
        :param column: Column to encode
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    x[column] = x[column].astype('category').cat.codes
    
    return x

def one_hot(x, column, rare=False, coeff=0.1):
    """ 
        Performs one-hot encoding.
        Example:
            Color: ['black', 'white', 'white']
            is encoded by
            Black: [1, 0, 0]
            White: [0, 1, 1]
            
        :param df: Data
        :param column: Column to encode
        :param rare: If True, rare categories are merged into one
        :param coeff: Coefficient defining rare values. 
                        A rare category occurs less than the (average number of occurrence * coefficient).
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    # Rare values management
    if rare:
        average = len(x[column]) / len(x[column].unique()) # train/test bias ?
        threshold = np.ceil(average * coeff)
        x.loc[x[column].value_counts()[x[column]].values < threshold, column] = "RARE_VALUE"
    
    # Usual one-hot encoding
    x = pd.concat([x, pd.get_dummies(x[column], prefix=column)], axis=1)
    x.drop([column], axis=1, inplace=True)
    
    return x


def likelihood(x, column, feat_type=None, mapping=None, return_param=False):
    """ 
        Performs likelihood encoding.
            
        :param df: Data
        :param column: Column to encode
        :param mapping: Dictionary {category : value}
        :param return_param: If True, the mapping is returned
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    # Numerical columns.
    if feat_type is None:
        feat_type = np.array(processing.get_types(x))
        
    numericals = x.columns[feat_type == 'Numerical']
    categories = x[column].unique()
        
    if mapping is None:
        mapping = dict()
        try: 
            # NOT OPTIMIZED: PCA will be computed for every variable
            pca = PCA()
            principal_axe = pca.fit(x[numericals].values).components_[0, :]
            # First principal component.
            pc1 = (principal_axe * x[numericals]).sum(axis=1)
        except:
            raise OSError('No numerical columns found, cannot apply likelihood encoding.')
        
        for i, category in enumerate(categories):
            mapping[category] = np.mean(pc1[x[column]==category])

    else:
        for category in categories:
            if category not in mapping:
                mapping[category] = 0

    x[column] = x[column].map(mapping)
    
    if return_param:
        return x, mapping
    return x


def count(x, column, mapping=None, return_param=False):
    """ Performs frequency encoding.
        Categories are replaced by their number of occurence.
        Soon: possibility of probability instead of count (normalization)
        
        :param df: Data
        :param column: Column to encode
        :param mapping: Dictionary {category : value}
        :param return_param: If True, the mapping is returned
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    categories = x[column].unique()
    
    if mapping is None:
        mapping = dict()
        for e in column:
            if e in mapping:
                mapping[e] += 1
            else:
                mapping[e] = 1
                
    else:
        for category in categories:
            if category not in mapping:
                mapping[category] = 0 # TODO
    
    x[column] = x[column].map(mapping)
    
    if return_param:
        return x, mapping
    return x
    
def target(x, column, target, mapping=None, return_param=False):
    """ 
        Performs target encoding.
            
        :param df: Data
        :param column: Column to encode
        :param target: Target column name
        :param mapping: Dictionary {category : value}
        :param return_param: If True, the mapping is returned
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    target = x[target]
    
    categories = x[column].unique()

    if mapping is None:
        mapping = dict()
        for i, category in enumerate(categories):
            mapping[category] = np.mean(target[x[column]==category]).round(3) # TODO

    else:
        for category in categories:
            if category not in mapping:
                mapping[category] = 0

    x[column] = x[column].map(mapping)
    
    if return_param:
        return x, mapping
    return x
    

def feature_hashing(X_train, X_test, verbose=True):
    """ Feature hashing
    """
    X_train_hash = copy.copy(X_train)
    X_test_hash = copy.copy(X_test)
    
    for i in range(X_train_hash.shape[1]):
        X_train_hash.iloc[:,i]=X_train_hash.iloc[:,i].astype('str')
        
    for i in range(X_test_hash.shape[1]):
        X_test_hash.iloc[:,i]=X_test_hash.iloc[:,i].astype('str')
        
    h = FeatureHasher(n_features=100,input_type="string")
    X_train_hash = h.transform(X_train_hash.values)
    X_test_hash = h.transform(X_test_hash.values)

    l.fit(X_train_hash,y_train)
    y_pred = l.predict_proba(X_test_hash)
    if verbose:
        print(log_loss(y_test,y_pred))

    r.fit(X_train_hash,y_train)
    y_pred = r.predict_proba(X_test_hash)
    if verbose:
        print(log_loss(y_test,y_pred))

def frequency(columns, probability=False):
    """ /!\ Warning: Take only column(s) and not DataFrame /!\
        Frequency encoding:
            Pandas series to frequency/probability distribution.
        
        If there are several series, the outputs will have the same format.
        Example:
          C1: ['b', 'a', 'a', 'b', 'b']
          C2: ['b', 'b', 'b', 'c', 'b']
          
          f1: ['a': 2, 'b'; 3, 'c': 0]
          f2: ['a': 0, 'b'; 4, 'c': 1]
          
          Output: [[2, 3, 0], [0, 4, 1]] (with probability = False)
          
        :param probability: True for probablities, False for frequencies.
        :return: Frequency/probability distribution.
        :rtype: list
    """ # TODO error if several columns have the same header

    # If there is only one column, just return frequencies
    if not isinstance(columns[0], (list, np.ndarray, pd.Series)):
        return columns.value_counts(normalize=probability).values
    
    frequencies = []
    
    # Compute frequencies for each column
    for column in columns:
        f = dict()
        for e in column:
            if e in f:
                f[e] += 1
            else:
                f[e] = 1
        frequencies.append(f)
        
    # Add keys from other columns in every dictionaries with a frequency of 0
    # We want the same format
    for i, f in enumerate(frequencies):
        for k in f.keys():
            for other_f in frequencies[:i]+frequencies[i+1:]:
                if k not in other_f:
                    other_f[k] = 0
           
    # Convert to frequency/probability distribution
    res = []         
    for f in frequencies:
        l = list(f.values())
        if probability:
            # normalize between 0 and 1 with a sum of 1
            l = normalize(l)
        # Convert dict into a list of values
        res.append(l)
        # Every list will follow the same order because the dicts contain the same keys
                    
    return res
    
    
def cat2vec(data, features, verbose=True):
    """ TODO
        Credit: Yonatan Hadar
    """
    size=6
    window=8
    x_w2v = copy.deepcopy(data.iloc[:,features])
    names = list(x_w2v.columns.values)
    
    for i in names:
        x_w2v[i]=x_w2v[i].astype('category')
        x_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in x_w2v[i].cat.categories]
    x_w2v = x_w2v.values.tolist()
    
    for i in x_w2v:
        shuffle(i)
    w2v = Word2Vec(x_w2v,size=size,window=window)

    X_train_w2v = copy.copy(X_train)
    X_test_w2v = copy.copy(X_test)
    
    for i in names:
        X_train_w2v[i] = X_train_w2v[i].astype('category')
        X_train_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in X_train_w2v[i].cat.categories]
    
    for i in names:
        X_test_w2v[i]=X_test_w2v[i].astype('category')
        X_test_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in X_test_w2v[i].cat.categories]
    X_train_w2v = X_train_w2v.values
    X_test_w2v = X_test_w2v.values
    x_w2v_train = np.random.random((len(X_train_w2v),size*X_train_w2v.shape[1]))
    
    for j in range(X_train_w2v.shape[1]):
        for i in range(X_train_w2v.shape[0]):
            if X_train_w2v[i,j] in w2v:
                x_w2v_train[i,j*size:(j+1)*size] = w2v[X_train_w2v[i,j]]

    x_w2v_test = np.random.random((len(X_test_w2v),size*X_test_w2v.shape[1]))
    for j in range(X_test_w2v.shape[1]):
        for i in range(X_test_w2v.shape[0]):
            if X_test_w2v[i,j] in w2v:
                x_w2v_test[i,j*size:(j+1)*size] = w2v[X_test_w2v[i,j]]

    l.fit(x_w2v_train,y_train)
    y_pred = l.predict_proba(x_w2v_test)
    if verbose:
        print(log_loss(y_test,y_pred))

    r.fit(x_w2v_train,y_train)
    y_pred = r.predict_proba(x_w2v_test)
    if verbose:
        print(log_loss(y_test,y_pred))
