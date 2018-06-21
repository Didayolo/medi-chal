# Imports
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split

from utilities import normalize

import bisect
import copy

# cat2vec
from gensim.models.word2vec import Word2Vec
from random import shuffle

def none(x, column):
    """ Remove column from x
    """
    x.drop([column], axis=1, inplace=True)
    return x

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


def count(x, column, mapping=None, probability=False, return_param=False):
    """ Performs frequency encoding.
        Categories are replaced by their number of occurence.
        Soon: possibility of probability instead of count (normalization)
        
        :param df: Data
        :param column: Column to encode
        :param mapping: Dictionary {category : value}
        :param probability: If True, return probability instead of frequency
        :param return_param: If True, the mapping is returned
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    categories = x[column].unique()
    
    if mapping is None:
        mapping = dict()
        for e in x[column]:
            if e in mapping:
                mapping[e] += 1
            else:
                mapping[e] = 1
                
    else:
        for category in categories:
            if category not in mapping:
                mapping[category] = 0 # TODO
                
    if probability:
        factor = 1.0 / sum(mapping.values())
        for k in mapping:
            mapping[k] = float(format(mapping[k] * factor, '.3f'))
    
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
            mapping[category] = np.mean(target[x[column]==category]).round(3)

    else:
        for category in categories:
            if category not in mapping:
                mapping[category] = 0

    x[column] = x[column].map(mapping)
    
    if return_param:
        return x, mapping
    return x
    

def feature_hashing(x, n_features=10, verbose=True):
    """ Feature hashing
        ...
    """
    h = FeatureHasher(n_features=n_features)
    x = x.to_dict('records')
    x = h.transform(x)
    x = x.toarray() # to array
    x = pd.DataFrame(x) # to df
    return x
    

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
    
    
def cat2vec(data, size=6, window=8, verbose=True):
    """ TODO
        Based on Yonatan Hadar's implementation
    """
    x_w2v = copy.deepcopy(data)
    names = list(x_w2v.columns.values)
    
    for i in names:
        x_w2v[i]=x_w2v[i].astype('category')
        x_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in x_w2v[i].cat.categories]
    x_w2v = x_w2v.values.tolist()
    
    for i in x_w2v:
        shuffle(i)
    w2v = Word2Vec(x_w2v, size=size, window=window)

    data_w2v = copy.copy(data)
    
    for i in names:
        data_w2v[i] = data_w2v[i].astype('category')
        data_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in data_w2v[i].cat.categories]
    
    data_w2v = data_w2v.values
    x_w2v_train = np.random.random((len(data_w2v),size*data_w2v.shape[1]))
    
    for j in range(data_w2v.shape[1]):
        for i in range(data_w2v.shape[0]):
            if data_w2v[i,j] in w2v:
                x_w2v_train[i,j*size:(j+1)*size] = w2v[data_w2v[i,j]]

    return pd.DataFrame(x_w2v_train)
    
 
# Deep category embedding
